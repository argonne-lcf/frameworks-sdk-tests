"""Fused RMSNorm.

RMSNorm is everywhere in both architectures (pre-norm blocks, KDA output norm,
LatentMoE's post-aggregation norm, AttnRes key normalization), so it gets a
fused Triton kernel. Forward-only fusion: for training we fall back to the
autograd-friendly reference (compiled ops are fine there); the Triton path is
used under ``torch.no_grad()`` i.e. inference.
"""

from __future__ import annotations

import torch


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


def _rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    import triton

    from ._rmsnorm_kernel import _rmsnorm_fwd

    shape = x.shape
    x2d = x.reshape(-1, shape[-1]).contiguous()
    M, N = x2d.shape
    out = torch.empty_like(x2d)
    BLOCK_N = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK_N // 256, 1), 16)
    _rmsnorm_fwd[(M,)](x2d, weight, out, N, x2d.stride(0), eps, BLOCK_N=BLOCK_N, num_warps=num_warps)
    return out.reshape(shape)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    from . import use_triton

    if use_triton(x, weight) and not torch.is_grad_enabled():
        return _rmsnorm_triton(x, weight, eps)
    return rmsnorm_ref(x, weight, eps)
