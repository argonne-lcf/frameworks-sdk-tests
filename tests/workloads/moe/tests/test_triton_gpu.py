"""Triton-kernel vs reference-twin equivalence ON DEVICE (CUDA or Intel XPU).

This is the module that actually validates the Triton kernels on real
hardware -- the rest of the suite runs the pure-PyTorch reference paths.
Skipped automatically when no Triton-capable device is present (e.g. macOS).

Run on the cluster:  pytest tests/ -q      (these tests activate by themselves)
Force references off for A/B:  LATENTMOE_DISABLE_TRITON=1 pytest tests/ -q
"""

import os

import pytest
import torch

from latentmoe.kernels import HAS_TRITON


def _gpu():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return None


GPU = _gpu()
pytestmark = pytest.mark.skipif(
    GPU is None or not HAS_TRITON or os.environ.get("LATENTMOE_DISABLE_TRITON"),
    reason="needs a CUDA/XPU device with Triton",
)

torch.manual_seed(0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rmsnorm_triton_matches_ref(dtype):
    from latentmoe.kernels.rmsnorm import _rmsnorm_triton, rmsnorm_ref

    x = torch.randn(64, 193, device=GPU, dtype=dtype)
    w = torch.randn(193, device=GPU, dtype=dtype)
    with torch.no_grad():
        got = _rmsnorm_triton(x, w, 1e-5)
    want = rmsnorm_ref(x, w, 1e-5)
    atol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(got.float(), want.float(), atol=atol), \
        (got - want).abs().max()


@pytest.mark.parametrize("fmt", ["fp4_e2m1", "fp8_e4m3"])
def test_mxfp_triton_matches_ref(fmt):
    from latentmoe.kernels.mxfp import _mx_quant_dequant_triton, mx_quant_dequant_ref

    x = torch.randn(32, 160, device=GPU) * 3
    got = _mx_quant_dequant_triton(x, fmt)
    want = mx_quant_dequant_ref(x, fmt)
    # identical closed-form grids; allow a vanishing fraction of half-step
    # rounding disagreements at exact grid midpoints
    mismatch = (got != want).float().mean().item()
    assert mismatch < 1e-3, (mismatch, (got - want).abs().max())


def test_kda_fused_recurrent_matches_ref():
    from latentmoe.kernels.kda_scan import _kda_recurrent_triton, kda_recurrent_ref

    B, H, T, Dk, Dv = 2, 3, 65, 32, 48
    g = torch.Generator(device="cpu").manual_seed(1)
    mk = lambda *s: torch.randn(*s, generator=g).to(GPU)  # noqa: E731
    q = mk(B, H, T, Dk)
    k = torch.nn.functional.normalize(mk(B, H, T, Dk), dim=-1)
    v = mk(B, H, T, Dv)
    alpha = torch.exp(-5.0 * torch.sigmoid(mk(B, H, T, Dk)))
    beta = torch.sigmoid(mk(B, H, T))
    state0 = mk(B, H, Dk, Dv).float()
    with torch.no_grad():
        o_t, s_t = _kda_recurrent_triton(q, k, v, alpha, beta, state0.clone())
        o_r, s_r = kda_recurrent_ref(q, k, v, alpha, beta, state0.clone())
    assert torch.allclose(o_t.float(), o_r.float(), atol=2e-3), (o_t - o_r).abs().max()
    assert torch.allclose(s_t, s_r, atol=2e-3), (s_t - s_r).abs().max()


def test_grouped_gemm_triton_matches_ref():
    from latentmoe.kernels.grouped_gemm import _grouped_gemm_ref, _grouped_gemm_triton

    E, K, N = 5, 48, 96
    sizes = torch.tensor([7, 0, 33, 1, 20])
    x = torch.randn(int(sizes.sum()), K, device=GPU)
    w = torch.randn(E, K, N, device=GPU)
    with torch.no_grad():
        got = _grouped_gemm_triton(x, w, sizes)
        want = _grouped_gemm_ref(x, w, sizes)
    assert torch.allclose(got, want, atol=1e-3, rtol=1e-3), (got - want).abs().max()
    # transposed-view path (what backward uses for dX)
    with torch.no_grad():
        got_t = _grouped_gemm_triton(torch.randn_like(got), w.transpose(1, 2).contiguous(), sizes)
    assert got_t.shape == (x.shape[0], K)


def test_indexer_triton_matches_ref():
    from latentmoe.kernels.indexer import _indexer_scores_triton, indexer_scores_ref

    B, T, H, D, S = 2, 37, 4, 8, 51  # D=8 exercises the BLOCK_D>=16 padding
    qI = torch.randn(B, T, H, D, device=GPU)
    wI = torch.randn(B, T, H, device=GPU)
    kI = torch.randn(B, S, D, device=GPU)
    with torch.no_grad():
        got = _indexer_scores_triton(qI, wI, kI)
    want = indexer_scores_ref(qI, wI, kI)
    assert torch.allclose(got, want, atol=1e-3), (got - want).abs().max()


@pytest.mark.parametrize("arch", ["v4", "k3", "hybrid"])
def test_model_forward_backward_on_gpu(arch):
    from latentmoe import LatentMoEModel, get_preset

    cfg = get_preset(arch, vocab_size=256, dim=64, n_layers=4, n_heads=4,
                     head_dim=16, kv_latent_dim=32, q_latent_dim=32, rope_dim=16,
                     csa_block=4, csa_topk=4, hca_block=8, local_window=16,
                     indexer_heads=2, indexer_dim=8, out_groups=2, out_group_dim=16,
                     kda_head_dim=16, kda_chunk=4, n_routed_experts=8, top_k=2,
                     moe_latent_dim=32, expert_hidden=16, shared_expert_hidden=32,
                     dense_ffn_hidden=64, mhc_streams=2, max_seq_len=128)
    model = LatentMoEModel(cfg).to(GPU)
    x = torch.randint(4, cfg.vocab_size, (2, 32), device=GPU)
    y = torch.randint(4, cfg.vocab_size, (2, 32), device=GPU)
    out = model(x, targets=y, return_logits=False)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("arch", ["v4", "k3"])
def test_model_autocast_bf16_on_gpu(arch):
    """Training-style mixed precision (what train.py does on GPU): bf16
    activations against fp32 master weights must flow through the Triton
    grouped GEMM -- regression test for the mixed-dtype tl.dot failure."""
    from latentmoe import LatentMoEModel, get_preset

    cfg = get_preset(arch, vocab_size=256, dim=64, n_layers=4, n_heads=4,
                     head_dim=16, kv_latent_dim=32, q_latent_dim=32, rope_dim=16,
                     csa_block=4, csa_topk=4, hca_block=8, local_window=16,
                     indexer_heads=2, indexer_dim=8, out_groups=2, out_group_dim=16,
                     kda_head_dim=16, kda_chunk=4, n_routed_experts=8, top_k=2,
                     moe_latent_dim=32, expert_hidden=16, shared_expert_hidden=32,
                     dense_ffn_hidden=64, mhc_streams=2, max_seq_len=128)
    model = LatentMoEModel(cfg).to(GPU)
    x = torch.randint(4, cfg.vocab_size, (2, 32), device=GPU)
    y = torch.randint(4, cfg.vocab_size, (2, 32), device=GPU)
    with torch.autocast(device_type=GPU.type, dtype=torch.bfloat16):
        out = model(x, targets=y, return_logits=False)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert p.grad.dtype == p.dtype, n     # fp32 grads on fp32 masters
            assert torch.isfinite(p.grad).all(), n


def test_model_cache_matches_full_on_gpu():
    """End-to-end: inference (Triton rmsnorm + fused-recurrent KDA + indexer)
    must agree with the training-path forward on device."""
    from latentmoe import LatentMoEModel, get_preset

    torch.manual_seed(0)
    cfg = get_preset("k3", vocab_size=256, dim=64, n_layers=4, n_heads=4,
                     head_dim=16, kv_latent_dim=32, q_latent_dim=32,
                     kda_head_dim=16, kda_chunk=4, n_routed_experts=8, top_k=2,
                     moe_latent_dim=32, expert_hidden=16, shared_expert_hidden=32,
                     dense_ffn_hidden=64, max_seq_len=128)
    model = LatentMoEModel(cfg).to(GPU).eval()
    x = torch.randint(4, cfg.vocab_size, (1, 33), device=GPU)
    with torch.no_grad():
        full = model(x)["logits"]
        caches = model.new_caches()
        outs = [model(x[:, :17], caches=caches)["logits"]]
        for t in range(17, 33):
            outs.append(model(x[:, t:t + 1], caches=caches)["logits"])
        inc = torch.cat(outs, dim=1)
    assert torch.allclose(full, inc, atol=2e-3), (full - inc).abs().max()
