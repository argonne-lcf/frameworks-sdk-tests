"""Triton MX quant-dequant kernel. One program per (row, 32-elem block)."""

import triton
import triton.language as tl


@triton.jit
def _mx_qdq(X, Y, N, stride_m,
            MBITS: tl.constexpr, P_LO: tl.constexpr, P_HI: tl.constexpr,
            MAX_VAL: tl.constexpr, EMAX: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    cols = blk * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_m + cols, mask=mask, other=0.0).to(tl.float32)

    ax = tl.abs(x)
    amax = tl.max(ax, axis=0)
    # shared E8M0 scale for the block
    e = tl.floor(tl.log2(tl.maximum(amax, 1e-30))) - EMAX
    scale = tl.exp2(e)
    scale = tl.where(amax > 0, scale, 1.0)

    y = tl.minimum(ax / scale, MAX_VAL)
    p = tl.floor(tl.log2(tl.maximum(y, 1e-30)))
    p = tl.minimum(tl.maximum(p, P_LO * 1.0), P_HI * 1.0)
    step = tl.exp2(p - MBITS)
    # round half to even (matches torch.round in the reference twin), built
    # from core tl ops only -- tl.math.rint does not exist on the XPU backend.
    # v >= 0 here (abs upstream) and v <= 2^(MBITS+1), so fp32 is exact.
    v = y / step
    f = tl.floor(v)
    r = v - f
    base = f + tl.where(r > 0.5, 1.0, 0.0)
    f_is_even = (tl.floor(f * 0.5) * 2.0) == f
    q = tl.where(r == 0.5, tl.where(f_is_even, f, f + 1.0), base) * step
    sign = tl.where(x < 0, -1.0, 1.0)
    out = q * scale * sign
    tl.store(Y + row * stride_m + cols, out.to(Y.dtype.element_ty), mask=mask)
