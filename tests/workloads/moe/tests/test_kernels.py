"""Kernel reference-path tests (Triton twins are exercised on GPU machines;
on CPU these validate the reference math the Triton kernels mirror)."""

import torch
import pytest

from latentmoe.kernels.kda_scan import kda_chunkwise, kda_recurrent_ref
from latentmoe.kernels.grouped_gemm import grouped_gemm
from latentmoe.kernels.indexer import indexer_scores_ref
from latentmoe.kernels.mxfp import mx_quant_dequant_ref, mx_fake_quant, MX_BLOCK
from latentmoe.kernels.rmsnorm import rmsnorm_ref

torch.manual_seed(0)


def test_rmsnorm_matches_manual():
    x = torch.randn(4, 33)
    w = torch.randn(33)
    y = rmsnorm_ref(x, w, eps=1e-5)
    ref = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    assert torch.allclose(y, ref, atol=1e-5)


def test_mxfp4_grid():
    x = torch.randn(8, 64) * 3
    q = mx_quant_dequant_ref(x, "fp4_e2m1")
    # every block's values must lie on {0,.5,1,1.5,2,3,4,6} x shared E8M0 scale,
    # where scale = 2^(floor(log2(amax)) - 2) by the MX definition
    xb = x.reshape(-1, MX_BLOCK)
    qb = q.reshape(-1, MX_BLOCK)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    for orig, blk in zip(xb, qb):
        amax = orig.abs().max()
        if amax == 0:
            assert (blk == 0).all()
            continue
        scale = torch.exp2(torch.floor(torch.log2(amax)) - 2)
        vals = (blk.abs() / scale)
        ok = (vals.unsqueeze(-1) - grid.unsqueeze(0)).abs().min(-1).values < 1e-5
        assert ok.all(), (blk / scale)


def test_mxfp_idempotent_and_ste():
    for fmt in ("fp4_e2m1", "fp8_e4m3"):
        x = torch.randn(4, 96)
        q1 = mx_quant_dequant_ref(x, fmt)
        q2 = mx_quant_dequant_ref(q1, fmt)
        assert torch.allclose(q1, q2, atol=1e-6), fmt
    x = torch.randn(4, 64, requires_grad=True)
    y = mx_fake_quant(x, "fp4_e2m1")
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))  # straight-through


def _rand_kda(B=2, H=2, T=37, Dk=16, Dv=24, seed=1):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, Dk, generator=g)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, Dk, generator=g), dim=-1)
    v = torch.randn(B, H, T, Dv, generator=g)
    alpha = torch.exp(-5.0 * torch.sigmoid(torch.randn(B, H, T, Dk, generator=g)))
    beta = torch.sigmoid(torch.randn(B, H, T, generator=g))
    return q, k, v, alpha, beta


@pytest.mark.parametrize("chunk", [4, 16, 64])
def test_kda_chunkwise_matches_recurrent(chunk):
    q, k, v, alpha, beta = _rand_kda()
    o_ref, s_ref = kda_recurrent_ref(q, k, v, alpha, beta)
    o_ck, s_ck = kda_chunkwise(q, k, v, alpha, beta, chunk=chunk)
    assert torch.allclose(o_ref, o_ck, atol=2e-4), (o_ref - o_ck).abs().max()
    assert torch.allclose(s_ref, s_ck, atol=2e-4)


def test_kda_initial_state_and_continuation():
    q, k, v, alpha, beta = _rand_kda(T=32)
    o_full, s_full = kda_chunkwise(q, k, v, alpha, beta, chunk=8)
    # split at t=13 (not a chunk multiple): continue from returned state
    o1, s1 = kda_chunkwise(q[:, :, :13], k[:, :, :13], v[:, :, :13],
                           alpha[:, :, :13], beta[:, :, :13], chunk=8)
    o2, s2 = kda_chunkwise(q[:, :, 13:], k[:, :, 13:], v[:, :, 13:],
                           alpha[:, :, 13:], beta[:, :, 13:], state=s1, chunk=8)
    assert torch.allclose(torch.cat([o1, o2], dim=2), o_full, atol=2e-4)
    assert torch.allclose(s2, s_full, atol=2e-4)


@pytest.mark.parametrize("chunk", [4, 16])
def test_kda_gradients_match(chunk):
    def run(fn, ck=None):
        torch.manual_seed(3)
        q, k, v, alpha, beta = _rand_kda(B=1, H=1, T=48, Dk=8, Dv=8, seed=3)
        # push decay toward its e^-5 floor to stress the fp32 exponent range
        alpha = (alpha.clamp_min(1e-30) ** 3).clamp_min(6.75e-3).detach()
        for t in (q, k, v, alpha, beta):
            t.requires_grad_(True)
        o, _ = fn(q, k, v, alpha, beta) if ck is None else fn(q, k, v, alpha, beta, chunk=ck)
        (o.pow(2).sum()).backward()
        return [t.grad.clone() for t in (q, k, v, alpha, beta)]

    g_ref = run(kda_recurrent_ref)
    g_ck = run(kda_chunkwise, ck=chunk)
    for a, b in zip(g_ref, g_ck):
        assert torch.isfinite(b).all()
        assert torch.allclose(a, b, atol=2e-3), (a - b).abs().max()


def test_grouped_gemm_forward_backward():
    E, K, N = 4, 16, 24
    sizes = torch.tensor([5, 0, 7, 3])
    M = int(sizes.sum())
    x = torch.randn(M, K, requires_grad=True)
    w = torch.randn(E, K, N, requires_grad=True)
    y = grouped_gemm(x, w, sizes)
    # reference
    xr = x.detach().clone().requires_grad_(True)
    wr = w.detach().clone().requires_grad_(True)
    outs, start = [], 0
    for e, sz in enumerate(sizes.tolist()):
        outs.append(xr[start:start + sz] @ wr[e])
        start += sz
    y_ref = torch.cat(outs)
    assert torch.allclose(y, y_ref, atol=1e-5)
    g = torch.randn_like(y)
    y.backward(g)
    y_ref.backward(g)
    assert torch.allclose(x.grad, xr.grad, atol=1e-5)
    assert torch.allclose(w.grad, wr.grad, atol=1e-5)


def test_indexer_scores():
    B, T, H, D, S = 2, 5, 3, 8, 7
    qI, wI, kI = torch.randn(B, T, H, D), torch.randn(B, T, H), torch.randn(B, S, D)
    got = indexer_scores_ref(qI, wI, kI)
    want = torch.zeros(B, T, S)
    for b in range(B):
        for t in range(T):
            for s in range(S):
                acc = 0.0
                for h in range(H):
                    acc += float(wI[b, t, h]) * max(float(qI[b, t, h] @ kI[b, s]), 0.0)
                want[b, t, s] = acc
    assert torch.allclose(got, want, atol=1e-4)
