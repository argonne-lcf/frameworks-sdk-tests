"""Grouped GEMM for MoE expert FFNs.

After routing, tokens are permuted so each expert's tokens are contiguous
(`group_sizes[e]` rows each). All experts' matmuls then run as ONE kernel
launch (the Triton path) instead of a Python loop -- this is the standard
fused-MoE trick and the compute core of expert parallelism.

    y = grouped_gemm(x, w, group_sizes)
      x: [M_total, K]  sorted by expert
      w: [E, K, N]     per-expert weights
      y: [M_total, N]

Autograd is provided:  dX = grouped_gemm(dY, w^T),  dW_e = X_e^T dY_e.
The Triton kernel takes W strides, so the transposed backward pass reuses the
same kernel on a free `w.transpose(1, 2)` view.
"""

from __future__ import annotations

import torch


def _grouped_gemm_ref(x, w, group_sizes):
    out = x.new_empty(x.shape[0], w.shape[-1])
    start = 0
    for e, sz in enumerate(group_sizes.tolist()):
        if sz:
            out[start : start + sz] = x[start : start + sz] @ w[e]
        start += sz
    return out


def _block_map(group_sizes, block_m: int, device):
    """Host-side map: for each M-block, which expert and which row range."""
    experts, starts = [], []
    start = 0
    for e, sz in enumerate(group_sizes.tolist()):
        nb = (sz + block_m - 1) // block_m
        for b in range(nb):
            experts.append(e)
            starts.append(start + b * block_m)
        start += sz
    if not experts:
        return None, None, None
    ends = []
    start = 0
    cum = torch.cumsum(torch.as_tensor(group_sizes.tolist()), 0).tolist()
    for e, s in zip(experts, starts):
        ends.append(cum[e])
    t = lambda v: torch.tensor(v, device=device, dtype=torch.int32)  # noqa: E731
    return t(experts), t(starts), t(ends)


def _grouped_gemm_triton(x, w, group_sizes):
    import triton

    from ._grouped_gemm_kernel import _grouped_mm

    M, K = x.shape
    E, _, N = w.shape
    out = x.new_empty(M, N)
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
    experts, starts, ends = _block_map(group_sizes, BLOCK_M, x.device)
    if experts is None:
        return out
    grid = (experts.shape[0], triton.cdiv(N, BLOCK_N))
    _grouped_mm[grid](
        x, w, out, experts, starts, ends,
        K, N,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out


def _forward(x, w, group_sizes):
    from . import use_triton

    if use_triton(x, w) and x.shape[0] > 0:
        return _grouped_gemm_triton(x, w, group_sizes)
    return _grouped_gemm_ref(x, w, group_sizes)


class _GroupedGemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, group_sizes):
        ctx.save_for_backward(x, w, group_sizes)
        return _forward(x, w, group_sizes)

    @staticmethod
    def backward(ctx, dy):
        x, w, group_sizes = ctx.saved_tensors
        dy = dy.contiguous()
        dx = _forward(dy, w.transpose(1, 2).contiguous(), group_sizes)
        dw = torch.zeros_like(w)
        start = 0
        for e, sz in enumerate(group_sizes.tolist()):
            if sz:
                dw[e] = x[start : start + sz].t() @ dy[start : start + sz]
            start += sz
        return dx, dw, None


def grouped_gemm(x: torch.Tensor, w: torch.Tensor, group_sizes: torch.Tensor) -> torch.Tensor:
    # Autocast does not reach inside a custom autograd.Function, so under bf16
    # autocast x arrives bf16 while the expert weights are fp32 masters --
    # mixed operands break tl.dot (and skip DPAS/tensor cores). Mirror what
    # autocast does for nn.Linear: cast weights to the activation dtype OUT
    # HERE, so the cast's autograd node returns fp32 grads to the master
    # weights automatically.
    if w.dtype != x.dtype:
        w = w.to(x.dtype)
    return _GroupedGemm.apply(x, w, group_sizes)
