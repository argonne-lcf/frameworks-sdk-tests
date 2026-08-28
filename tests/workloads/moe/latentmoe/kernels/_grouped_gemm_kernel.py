"""Triton grouped-GEMM kernel: one launch computes every expert's matmul.

Each program owns one (M-block, N-block) tile; a host-precomputed map tells it
which expert's weight matrix and which row range it serves.
"""

import triton
import triton.language as tl


@triton.jit
def _grouped_mm(X, W, Y, BLOCK_EXPERT, BLOCK_START, BLOCK_END,
                K, N,
                sxm, sxk, swe, swk, swn, sym, syn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert = tl.load(BLOCK_EXPERT + pid_m)
    row0 = tl.load(BLOCK_START + pid_m)
    row_end = tl.load(BLOCK_END + pid_m)

    offs_m = row0 + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < row_end
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(X + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(W + expert * swe + offs_k[:, None] * swk + offs_n[None, :] * swn,
                    mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w, allow_tf32=False)

    tl.store(Y + offs_m[:, None] * sym + offs_n[None, :] * syn,
             acc.to(Y.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])
