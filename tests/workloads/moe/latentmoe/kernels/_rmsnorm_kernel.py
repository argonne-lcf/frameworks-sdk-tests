"""Triton RMSNorm kernel (portable: only core triton.language ops)."""

import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(X, W, Y, N, stride_xm, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Y + row * stride_xm + cols, y.to(Y.dtype.element_ty), mask=mask)
