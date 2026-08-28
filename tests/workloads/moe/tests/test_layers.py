"""Layer-level tests: causality, cache consistency, balancing behavior."""

import torch
import pytest

from latentmoe.layers.attnres import AttentionResidual
from latentmoe.layers.csa_hca import CompressedAttention
from latentmoe.layers.kda import KDAAttention
from latentmoe.layers.mhc import HyperConnections, sinkhorn
from latentmoe.layers.mla import GatedMLA
from latentmoe.layers.moe import LatentMoE

torch.manual_seed(0)

DIM, HEADS = 64, 4


def _csa(mode):
    return CompressedAttention(
        DIM, HEADS, mode, kv_latent_dim=32, q_latent_dim=48, rope_dim=16,
        csa_block=4, csa_topk=3, hca_block=8, local_window=16,
        indexer_heads=2, indexer_dim=8, out_groups=2, out_group_dim=24,
    ).eval()


@pytest.mark.parametrize("mode", ["csa", "hca"])
def test_compressed_attention_causal(mode):
    m = _csa(mode)
    T = 50
    x1 = torch.randn(1, T, DIM)
    x2 = x1.clone()
    x2[:, 30:] = torch.randn(1, T - 30, DIM)  # perturb the future
    with torch.no_grad():
        y1, y2 = m(x1), m(x2)
    assert torch.allclose(y1[:, :30], y2[:, :30], atol=1e-5), \
        (y1[:, :30] - y2[:, :30]).abs().max()
    assert not torch.allclose(y1[:, 30:], y2[:, 30:], atol=1e-3)


@pytest.mark.parametrize("mode", ["csa", "hca"])
@pytest.mark.parametrize("split", [1, 7, 32])
def test_compressed_attention_cache_matches_full(mode, split):
    m = _csa(mode)
    T = 41
    x = torch.randn(1, T, DIM)
    with torch.no_grad():
        full = m(x)
        cache = {}
        outs = []
        s = 0
        while s < T:
            e = min(T, s + split)
            outs.append(m(x[:, s:e], cache=cache))
            s = e
        inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_kda_layer_cache_matches_full():
    m = KDAAttention(DIM, HEADS, head_dim=16, conv_size=4, chunk=4).eval()
    T = 23
    x = torch.randn(1, T, DIM)
    with torch.no_grad():
        full = m(x)
        cache = {}
        inc = torch.cat([m(x[:, :9], cache=cache), m(x[:, 9:], cache=cache)], dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_kda_layer_causal():
    m = KDAAttention(DIM, HEADS, head_dim=16).eval()
    x1 = torch.randn(1, 20, DIM)
    x2 = x1.clone()
    x2[:, 12:] = torch.randn(1, 8, DIM)
    with torch.no_grad():
        y1, y2 = m(x1), m(x2)
    assert torch.allclose(y1[:, :12], y2[:, :12], atol=1e-5)


def test_mla_cache_matches_full():
    m = GatedMLA(DIM, HEADS, head_dim=16, kv_latent_dim=24, q_latent_dim=32).eval()
    T = 19
    x = torch.randn(1, T, DIM)
    with torch.no_grad():
        full = m(x)
        cache = {}
        outs = [m(x[:, :10], cache=cache)]
        for t in range(10, T):  # decode one token at a time
            outs.append(m(x[:, t:t + 1], cache=cache))
        inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_sinkhorn_doubly_stochastic():
    torch.manual_seed(7)
    logits = torch.randn(6, 6) * 3
    M = sinkhorn(logits, 50)
    assert torch.allclose(M.sum(0), torch.ones(6), atol=1e-3)
    assert torch.allclose(M.sum(1), torch.ones(6), atol=1e-3)
    assert (M >= 0).all()
    # spectral norm bounded by 1 (Birkhoff polytope property)
    assert torch.linalg.matrix_norm(M, 2) <= 1.0 + 1e-3


def test_hyperconnections_init_is_identity_like():
    hc = HyperConnections(4)
    x = torch.randn(2, 5, DIM)
    X = HyperConnections.expand(x, 4)
    assert torch.allclose(hc.read(X), x, atol=1e-5)  # a = 1/n mean of copies
    y = torch.randn(2, 5, DIM)
    X2 = hc.write(X, y)
    # near-identity B + c=1 broadcast: mean stream == x + y
    assert torch.allclose(HyperConnections.collapse(X2), x + y, atol=1e-2)


def test_attnres_zero_init_noop():
    ar = AttentionResidual(DIM)
    srcs = [torch.randn(2, 5, DIM) for _ in range(3)]
    assert torch.allclose(ar(srcs), torch.zeros(2, 5, DIM))


def _moe(balancer, latent=32, **kw):
    return LatentMoE(DIM, n_experts=8, top_k=2, expert_hidden=16,
                     n_shared=1, shared_hidden=32, latent_dim=latent,
                     balancer=balancer, **kw)


def test_moe_forward_shapes_and_losses():
    for latent in (32, None):
        for score in ("sigmoid", "sqrt_softplus"):
            m = _moe("bias", latent=latent, score_fn=score)
            m.train()
            x = torch.randn(2, 10, DIM)
            y, aux = m(x)
            assert y.shape == x.shape
            assert torch.isfinite(y).all()
            assert aux["seq_balance_loss"].ndim == 0
            (y.sum() + aux["seq_balance_loss"]).backward()
            assert m.w1.grad is not None and torch.isfinite(m.w1.grad).all()


def _imbalance(m, x, steps, use_balance_step):
    loads = []
    for _ in range(steps):
        m._load_acc.zero_()
        m(x)
        load = m._load_acc / m._load_acc.sum()
        loads.append(float(load.max() * m.e))
        if use_balance_step:
            m.balance_step()
    return loads


def _skew_router(m):
    """Skew the router toward experts 0/1 with continuous (tie-free) scores."""
    with torch.no_grad():
        m.w_router.weight.mul_(0.3)
        m.w_router.weight[0] += 0.4 * torch.ones(DIM) / DIM ** 0.5 * 8
        m.w_router.weight[1] += 0.3 * torch.ones(DIM) / DIM ** 0.5 * 8


def test_bias_balancer_reduces_imbalance():
    torch.manual_seed(1)
    m = _moe("bias", bias_update_rate=0.02).train()
    _skew_router(m)
    x = torch.randn(4, 64, DIM)
    with torch.no_grad():
        loads = _imbalance(m, x, 100, use_balance_step=True)
    early, late = sum(loads[:5]) / 5, sum(loads[-10:]) / 10
    assert early > 1.8, loads[:5]
    assert late < early * 0.75, (early, late, loads[::20])


def test_quantile_balancer_reduces_imbalance():
    torch.manual_seed(1)
    m = _moe("quantile", quantile_ema=0.5).train()
    _skew_router(m)
    x = torch.randn(4, 64, DIM)
    with torch.no_grad():
        loads = _imbalance(m, x, 60, use_balance_step=False)
    early, late = loads[0], sum(loads[-5:]) / 5
    assert early > 1.8, loads[:5]
    assert late < early * 0.85 and late < 1.6, (early, late, loads[::10])


def test_hash_routing_deterministic():
    m = _moe("none", hash_route=True).eval()
    x = torch.randn(1, 6, DIM)
    ids = torch.randint(0, 1000, (1, 6))
    y1, _ = m(x, token_ids=ids)
    y2, _ = m(x, token_ids=ids)
    assert torch.allclose(y1, y2)


def test_moe_qat_runs():
    m = _moe("bias", qat=True).train()
    x = torch.randn(2, 8, DIM)
    y, _ = m(x)
    y.sum().backward()
    assert torch.isfinite(m.w1.grad).all()
