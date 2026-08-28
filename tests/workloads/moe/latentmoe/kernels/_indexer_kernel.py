"""Triton lightning-indexer kernel: fused ReLU(q.k) head mixture."""

import triton
import triton.language as tl


@triton.jit
def _indexer_fwd(QI, WI, KI, OUT, T, S, H: tl.constexpr, D: tl.constexpr,
                 BLOCK_T: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    pt = tl.program_id(1)
    ps = tl.program_id(2)

    offs_t = pt * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_s = ps * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, BLOCK_D)
    mask_t = offs_t < T
    mask_s = offs_s < S
    mask_d = offs_d < D

    # K block: [BLOCK_S, D]
    k = tl.load(KI + b * S * D + offs_s[:, None] * D + offs_d[None, :],
                mask=mask_s[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_T, BLOCK_S), dtype=tl.float32)
    for h in range(H):
        q = tl.load(QI + ((b * T + offs_t[:, None]) * H + h) * D + offs_d[None, :],
                    mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        w = tl.load(WI + (b * T + offs_t) * H + h, mask=mask_t, other=0.0).to(tl.float32)
        dots = tl.dot(q, tl.trans(k), allow_tf32=False)
        acc += w[:, None] * tl.maximum(dots, 0.0)

    tl.store(OUT + b * T * S + offs_t[:, None] * S + offs_s[None, :], acc,
             mask=mask_t[:, None] & mask_s[None, :])
