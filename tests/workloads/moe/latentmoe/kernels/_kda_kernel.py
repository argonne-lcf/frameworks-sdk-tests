"""Fused recurrent KDA Triton kernel (forward only, inference path).

One program owns one (batch*head, value-block) pair and carries the state
tile S[BK, BV] in registers across the whole sequence. Portable Triton only.
"""

import triton
import triton.language as tl


@triton.jit
def _kda_fwd(Q, K, V, ALPHA, BETA, S, O, T, DK: tl.constexpr, DV: tl.constexpr,
             BK: tl.constexpr, BV: tl.constexpr):
    pid_bh = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < DK
    mask_v = offs_v < DV

    q_base = Q + pid_bh * T * DK
    k_base = K + pid_bh * T * DK
    a_base = ALPHA + pid_bh * T * DK
    v_base = V + pid_bh * T * DV
    b_base = BETA + pid_bh * T
    o_base = O + pid_bh * T * DV
    s_base = S + pid_bh * DK * DV

    # load initial state tile
    s_ptrs = s_base + offs_k[:, None] * DV + offs_v[None, :]
    s_mask = mask_k[:, None] & mask_v[None, :]
    state = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

    for t in range(T):
        k_t = tl.load(k_base + t * DK + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        a_t = tl.load(a_base + t * DK + offs_k, mask=mask_k, other=1.0).to(tl.float32)
        v_t = tl.load(v_base + t * DV + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        b_t = tl.load(b_base + t).to(tl.float32)
        q_t = tl.load(q_base + t * DK + offs_k, mask=mask_k, other=0.0).to(tl.float32)

        state = state * a_t[:, None]                    # Diag(alpha) S
        pred = tl.sum(state * k_t[:, None], axis=0)     # k^T S      [BV]
        state += b_t * k_t[:, None] * (v_t - pred)[None, :]
        o_t = tl.sum(state * q_t[:, None], axis=0)      # S^T q      [BV]
        tl.store(o_base + t * DV + offs_v, o_t, mask=mask_v)

    tl.store(s_ptrs, state, mask=s_mask)
