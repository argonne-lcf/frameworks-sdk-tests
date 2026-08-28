"""MX (microscaling) block fake-quantization: MXFP4 / MXFP8.

Kimi-K3 trains routed-expert weights in MXFP4 with MXFP8 activations via
quantization-aware training; DeepSeek-V4 stores routed expert parameters in
FP4. MX format (OCP spec): blocks of 32 consecutive elements share one
power-of-two (E8M0) scale; elements are FP4-E2M1 or FP8-E4M3.

We implement *fake quantization* (quantize -> dequantize in the compute dtype)
with a straight-through estimator, which is exactly what QAT needs, and is
bit-faithful to the grids of the real formats:

    E2M1 grid:  +-{0, 0.5, 1, 1.5, 2, 3, 4, 6}
    E4M3 grid:  3 mantissa bits, max 448 (subnormals below 2^-6 approximated)

Both the reference and the Triton kernel use the same closed-form grid
rounding so they agree bit-for-bit:

    y    = |x| / scale
    p    = clamp(floor(log2(y)), p_lo, p_hi)
    step = 2^(p - mbits)
    q    = rint(y / step) * step          (then clamp to the format max)
"""

from __future__ import annotations

import torch

MX_BLOCK = 32

# (mbits, p_lo, p_hi, max_val, emax) per element format
_FMT = {
    "fp4_e2m1": (1, 0, 2, 6.0, 2),
    "fp8_e4m3": (3, -6, 8, 448.0, 8),
}


def _grid_round(y: torch.Tensor, fmt: str) -> torch.Tensor:
    mbits, p_lo, p_hi, max_val, _ = _FMT[fmt]
    y = y.clamp(max=max_val)
    safe = y.clamp_min(1e-30)
    p = torch.floor(torch.log2(safe)).clamp_(p_lo, p_hi)
    step = torch.exp2(p - mbits)
    return torch.round(y / step) * step


def mx_quant_dequant_ref(x: torch.Tensor, fmt: str = "fp4_e2m1") -> torch.Tensor:
    """Fake-quantize the last dim in blocks of MX_BLOCK with shared E8M0 scales."""
    mbits, p_lo, p_hi, max_val, emax = _FMT[fmt]
    orig_shape, dtype = x.shape, x.dtype
    n = orig_shape[-1]
    pad = (-n) % MX_BLOCK
    xf = x.float().reshape(-1, n)
    if pad:
        xf = torch.nn.functional.pad(xf, (0, pad))
    xb = xf.reshape(xf.shape[0], -1, MX_BLOCK)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    # E8M0 shared scale: 2^(floor(log2(amax)) - emax); zero blocks -> scale 1
    e = torch.floor(torch.log2(amax.clamp_min(1e-30))) - emax
    scale = torch.exp2(e)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    q = _grid_round(xb.abs() / scale, fmt) * scale * torch.sign(xb)
    q = q.reshape(xf.shape[0], -1)[:, :n].reshape(orig_shape)
    return q.to(dtype)


def _mx_quant_dequant_triton(x: torch.Tensor, fmt: str) -> torch.Tensor:
    import triton

    from ._mxfp_kernel import _mx_qdq

    mbits, p_lo, p_hi, max_val, emax = _FMT[fmt]
    orig_shape, dtype = x.shape, x.dtype
    n = orig_shape[-1]
    x2d = x.reshape(-1, n).contiguous()
    out = torch.empty_like(x2d)
    n_blocks = triton.cdiv(n, MX_BLOCK)
    _mx_qdq[(x2d.shape[0], n_blocks)](
        x2d, out, n, x2d.stride(0),
        MBITS=mbits, P_LO=p_lo, P_HI=p_hi, MAX_VAL=max_val, EMAX=emax,
        BLOCK=MX_BLOCK,
    )
    return out.reshape(orig_shape).to(dtype)


def mx_quant_dequant(x: torch.Tensor, fmt: str = "fp4_e2m1") -> torch.Tensor:
    from . import use_triton

    if use_triton(x):
        return _mx_quant_dequant_triton(x, fmt)
    return mx_quant_dequant_ref(x, fmt)


class MXFakeQuant(torch.autograd.Function):
    """Straight-through estimator: forward = MX quant-dequant, backward = identity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, fmt: str):
        return mx_quant_dequant(x, fmt)

    @staticmethod
    def backward(ctx, g):
        return g, None


def mx_fake_quant(x: torch.Tensor, fmt: str = "fp4_e2m1") -> torch.Tensor:
    return MXFakeQuant.apply(x, fmt)
