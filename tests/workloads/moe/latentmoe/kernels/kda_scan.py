"""Kimi Delta Attention (KDA) scan kernels.

KDA recurrence (Kimi-K3 / Kimi Linear; a gated delta rule with *channel-wise*
decay on the key dimension):

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T
    o_t = S_t^T q_t

with alpha_t in (e^{gmin}, 1)^{dk} (lower-bounded decay) and beta_t in (0,1).

Three implementations:

  * ``kda_recurrent_ref``  -- exact token-by-token scan; ground truth in tests.
  * ``kda_chunkwise``      -- exact chunked closed form (pure PyTorch matmuls,
        autograd-friendly). Derivation: writing D_t = Diag(prod_{s<=t} alpha_s)
        within a chunk and S_t = D_t (S_0 + sum_{s<=t} beta_s (k_s / d_s) u_s^T),
        the "pseudo values" u solve a unit-lower-triangular system

            (I + tril(K_hat K_tilde^T, -1) diag(beta)) U = V - K_hat S_0

        with K_hat = K * d and K_tilde = K / d, giving

            O      = Q_hat S_0 + (tril(Q_hat K_tilde^T) diag(beta)) U
            S_next = d_C * (S_0 + K_tilde^T diag(beta) U).

        The paper's 16-token tiles bound the cumulative log-decay to (-80, 0),
        i.e. d >= e^-80 ~ 2e-35: everything stays inside fp32 range, which is
        exactly why the chunk size defaults to 16. Internals run in fp32.
  * ``_kda_recurrent_triton`` -- fused recurrent Triton kernel (inference
        prefill/decode path on CUDA/XPU; state tiles live in registers).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _init_state(q, v, state):
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    if state is None:
        state = q.new_zeros(B, H, Dk, Dv, dtype=torch.float32)
    return state.float()


@torch.no_grad()
def _shapes_ok(q, k, v, alpha, beta):
    assert q.shape == k.shape == alpha.shape, (q.shape, k.shape, alpha.shape)
    assert beta.shape == q.shape[:3], (beta.shape, q.shape)
    assert v.shape[:3] == q.shape[:3]


def kda_recurrent_ref(q, k, v, alpha, beta, state=None):
    """Exact sequential scan. q,k,alpha: [B,H,T,Dk]; v: [B,H,T,Dv]; beta: [B,H,T].

    Returns (o [B,H,T,Dv] in q.dtype, final state [B,H,Dk,Dv] fp32).
    """
    _shapes_ok(q, k, v, alpha, beta)
    S = _init_state(q, v, state)
    qf, kf, vf, af, bf = (t.float() for t in (q, k, v, alpha, beta))
    outs = []
    for t in range(q.shape[2]):
        kt, vt, at, bt, qt = kf[..., t, :], vf[..., t, :], af[..., t, :], bf[..., t], qf[..., t, :]
        S = at.unsqueeze(-1) * S  # Diag(alpha_t) S_{t-1}
        # delta rule: S += beta k (v - k^T S)^T
        pred = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + bt[..., None, None] * kt.unsqueeze(-1) * (vt - pred).unsqueeze(-2)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, qt))
    return torch.stack(outs, dim=2).to(q.dtype), S


def kda_chunkwise(q, k, v, alpha, beta, state=None, chunk: int = 16):
    """Exact chunkwise form. Same signature/returns as kda_recurrent_ref."""
    # 16-token tile bound (paper): with log-decay lower-bounded at gmin=-5 the
    # within-tile cumulative decay stays >= e^-80, inside fp32 range; larger
    # tiles would underflow the k/d "un-decay" trick, so clamp.
    chunk = min(chunk, 16)
    _shapes_ok(q, k, v, alpha, beta)
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    in_dtype = q.dtype
    S = _init_state(q, v, state)

    pad = (-T) % chunk
    if pad:
        zpad = lambda t, val: F.pad(t, (0, 0, 0, pad), value=val)  # noqa: E731
        q, k, v = zpad(q.float(), 0.0), zpad(k.float(), 0.0), zpad(v.float(), 0.0)
        alpha = zpad(alpha.float(), 1.0)
        beta = F.pad(beta.float(), (0, pad), value=0.0)
    else:
        q, k, v, alpha, beta = (t.float() for t in (q, k, v, alpha, beta))

    Tp = T + pad
    nC = Tp // chunk
    # [B,H,nC,C,D]
    qc = q.view(B, H, nC, chunk, Dk)
    kc = k.view(B, H, nC, chunk, Dk)
    vc = v.view(B, H, nC, chunk, Dv)
    ac = alpha.view(B, H, nC, chunk, Dk)
    bc = beta.view(B, H, nC, chunk)

    d = torch.cumprod(ac, dim=-2)                       # within-chunk cumulative decay
    # exponent re-centering: divide d by its (detached) mid-tile value. The
    # chunk equations are invariant to this per-channel rescaling as long as
    # the incoming state is scaled by the anchor (see below) -- but it halves
    # the exponent range, so the d^2 that appears in BACKWARD of k/d stays
    # inside fp32 even for 16-token tiles with gmin=-5.
    anchor = d[..., chunk // 2, :].detach()             # [B,H,nC,Dk]
    d = d / anchor.unsqueeze(-2)
    k_hat = kc * d                                      # k_t * (d_t / c)
    k_tilde = kc / d                                    # k_t * (c / d_t)
    q_hat = qc * d

    eye = torch.eye(chunk, device=q.device, dtype=torch.float32)
    strict = torch.tril(torch.ones(chunk, chunk, device=q.device), diagonal=-1)
    incl = torch.tril(torch.ones(chunk, chunk, device=q.device), diagonal=0)

    # A[t,s] = (k_hat_t . k_tilde_s) * beta_s, strictly lower triangular
    A = torch.einsum("bhntk,bhnsk->bhnts", k_hat, k_tilde) * bc.unsqueeze(-2) * strict
    Ao = torch.einsum("bhntk,bhnsk->bhnts", q_hat, k_tilde) * bc.unsqueeze(-2) * incl

    outs = []
    for i in range(nC):
        Sa = anchor[:, :, i].unsqueeze(-1) * S           # c (x) S0
        rhs = vc[:, :, i] - torch.einsum("bhtk,bhkv->bhtv", k_hat[:, :, i], Sa)
        U = torch.linalg.solve_triangular(
            eye + A[:, :, i], rhs, upper=False, unitriangular=True
        )
        o = torch.einsum("bhtk,bhkv->bhtv", q_hat[:, :, i], Sa) + torch.einsum(
            "bhts,bhsv->bhtv", Ao[:, :, i], U
        )
        outs.append(o)
        bU = bc[:, :, i].unsqueeze(-1) * U
        d_last = d[:, :, i, -1] * anchor[:, :, i]        # true cumulative decay
        S = d_last.unsqueeze(-1) * S + (d[:, :, i, -1]).unsqueeze(-1) * torch.einsum(
            "bhtk,bhtv->bhkv", k_tilde[:, :, i], bU)

    o = torch.cat(outs, dim=2)[:, :, :T]
    return o.to(in_dtype), S


def _kda_recurrent_triton(q, k, v, alpha, beta, state=None):
    import triton

    from ._kda_kernel import _kda_fwd

    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    S = _init_state(q, v, state).contiguous()
    o = torch.empty(B, H, T, Dv, device=q.device, dtype=torch.float32)
    q, k, v, alpha, beta = (t.contiguous() for t in (q, k, v, alpha, beta))
    BK = triton.next_power_of_2(Dk)
    BV = min(64, triton.next_power_of_2(Dv))
    grid = (B * H, triton.cdiv(Dv, BV))
    _kda_fwd[grid](q, k, v, alpha, beta, S, o, T, Dk, Dv, BK=BK, BV=BV)
    return o.to(q.dtype), S


def kda_scan(q, k, v, alpha, beta, state=None, chunk: int = 16):
    """Dispatch: Triton fused-recurrent for inference on GPU, chunkwise for
    training / long prefill, plain recurrence for tiny T (decode steps)."""
    from . import use_triton

    if use_triton(q, k, v) and not torch.is_grad_enabled():
        return _kda_recurrent_triton(q, k, v, alpha, beta, state)
    if q.shape[2] <= 4 and not torch.is_grad_enabled():
        return kda_recurrent_ref(q, k, v, alpha, beta, state)
    return kda_chunkwise(q, k, v, alpha, beta, state, chunk=chunk)
