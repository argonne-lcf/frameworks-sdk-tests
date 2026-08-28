"""MuonClip: Muon (orthogonalized momentum) + AdamW split + QK-clip.

Both DeepSeek-V4 and Kimi-K3 pre-train with Muon. Per the V4 report
(Algorithm 1) each matrix parameter is updated with the orthogonalized
momentum direction:

    M_t = mu M_{t-1} + G_t
    O'  = HybridNewtonSchulz(mu M_t + G_t)          (Nesterov flavor)
    O   = O' * sqrt(max(n, m)) * gamma              (update-RMS matching)
    W_t = W_{t-1} (1 - eta lambda) - eta O

HybridNewtonSchulz = 8 iterations with (a,b,c) = (3.4445, -4.7750, 2.0315)
followed by 2 "polishing" iterations with (2, -1.5, 0.5) -- V4's refinement of
the standard 5-step schedule (better orthogonality at the same cost class).
V4 notes the iteration is stable in bf16 matmuls; we use bf16 on GPU, fp32
elsewhere.

Kimi-K3 refinement: momentum matrices of attention q/k/v projections are
partitioned ALONG THE HEAD DIMENSION and orthogonalized per head (equalizes
update scale across heads; cheaper on tall stacked projections). Any param
tagged `_muon_heads = H` gets this treatment -- the attention layers tag
themselves.

Non-matrix / sensitive params (embeddings, LM head, norms, mHC gains, biases,
routers, conv filters -- anything tagged `_no_muon`, plus every param with
ndim != 2) fall back to an internal AdamW, exactly like the reports.

QK-clip (Kimi-K2/K3 stability): after each step, attention layers whose max
attention logit exceeded tau get their per-head q/k projections rescaled --
call `optimizer.qk_clip(model, tau)` (train.py wires this), which delegates to
each layer's `qk_clip_`.
"""

from __future__ import annotations

import math
import os

import torch

_NS_COEFFS = [(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2


_GRAM_K_CHUNK = 2048  # largest inner dim measured per-call-DETERMINISTIC for
#   bf16 matmul on PVC: K=2048 clean, K=3072 nondeterministic (split-K
#   accumulation) — field-located via the 2D driver's replica fingerprints
#   (the only diverging tensors were the two 1024x3072-geometry MTP matrices)
#   and bracketed by ns_proj2k vs ns_fuse3k in tools/ns_determinism_smoke.py
#   (HANDOFF UPDATEs 23-24).


def _det_mm(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """P @ Q, deterministic-by-construction on device (batched OK).

    Output columns are produced in fixed slabs of <=_GRAM_K_CHUNK (bit-free:
    disjoint outputs, no accumulation crosses a slab), and an inner dim
    beyond the chunk accumulates K-slabs in fixed order with fp32 partials
    (each bf16 slab product is itself fp32-accumulated internally, so the
    only extra rounding vs monolithic is one bf16 round per partial —
    negligible for Newton-Schulz, which is bf16-tolerant by design).
    Identical FLOPs to the monolithic product. Small matmuls take the
    untouched single-call path bit-exactly.
    """
    K, N = P.shape[-1], Q.shape[-1]
    if K <= _GRAM_K_CHUNK and N <= _GRAM_K_CHUNK:
        return P @ Q
    outs = []
    for jn in range(0, N, _GRAM_K_CHUNK):
        q = Q[..., jn:jn + _GRAM_K_CHUNK]
        if K <= _GRAM_K_CHUNK:
            outs.append(P @ q)
            continue
        acc = None
        for jk in range(0, K, _GRAM_K_CHUNK):
            t = (P[..., jk:jk + _GRAM_K_CHUNK]
                 @ q[..., jk:jk + _GRAM_K_CHUNK, :]).float()
            acc = t if acc is None else acc + t
        outs.append(acc.to(P.dtype))
    return torch.cat(outs, dim=-1)


def _gram(X: torch.Tensor) -> torch.Tensor:
    """X @ X^T via the deterministic matmul."""
    return _det_mm(X, X.transpose(-2, -1))


def _ns_wide_safe(u: torch.Tensor) -> torch.Tensor:
    """Newton-Schulz for WIDE-geometry updates (max dim > _GRAM_K_CHUNK):
    on PROVEN-DIRTY devices, run on CPU — MKL fp32, the one substrate
    measured bit-deterministic in every run of this project — and ship the
    result back. On PVC (xpu), multiple bf16 GEMM variants at these
    geometries are per-call nondeterministic in BOTH orientations (constant
    noise for some variants, occasional for others, rank-dependent for one
    — HANDOFF UPDATEs 24-29), so no on-device invocation is trusted there.
    CUDA/cuBLAS documents run-to-run determinism and is left ON-DEVICE by
    default — but VERIFY on first NVIDIA contact with
    tools/ns_determinism_smoke.py + train_2d.py --debug-grad-drift before
    relying on replica identity. Measured cost of the offload at dim-1024
    on Sunspot: ~1-4% of a ~150 s step (12 ranks/node contend for the same
    CPUs). Overrides: LATENTMOE_NS_CPU_WIDE=1 forces the CPU offload on
    ANY device (paranoid mode, e.g. if the NVIDIA smoke ever shows dirt);
    LATENTMOE_NS_DEVICE_WIDE=1 forces on-device everywhere (A/B)."""
    if u.device.type == "cpu" or max(u.shape[-2], u.shape[-1]) <= _GRAM_K_CHUNK:
        return newton_schulz(u)
    if os.environ.get("LATENTMOE_NS_DEVICE_WIDE") == "1":
        return newton_schulz(u)
    if (u.device.type == "xpu"
            or os.environ.get("LATENTMOE_NS_CPU_WIDE") == "1"):
        return newton_schulz(u.cpu()).to(u.device)
    return newton_schulz(u)


@torch.no_grad()
def newton_schulz(G: torch.Tensor, coeffs=_NS_COEFFS) -> torch.Tensor:
    """Orthogonalize the last two dims (batched OK): G ~ U V^T of its SVD.

    Every reduction inside runs deterministic-by-construction (UPDATE 25):
    the norm in fp32 (device fp32 sums measured clean to 64M elements) and
    all three matmuls through _det_mm — so Muon updates are bit-identical
    across replicas at ANY parameter shape.
    """
    dtype = torch.bfloat16 if G.device.type in ("cuda", "xpu") else torch.float32
    X = G.to(dtype)
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.transpose(-2, -1)
    n = X.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7).to(dtype)
    X = X / n
    for a, b, c in coeffs:
        A = _gram(X)                        # K-chunked X @ X^T
        B = b * A + c * _det_mm(A, A)
        X = a * X + _det_mm(B, X)           # N-chunked when X is wide
    if transposed:
        X = X.transpose(-2, -1)
    return X.to(G.dtype)


def retag_model(model: torch.nn.Module) -> None:
    """(Re-)apply `_no_muon` / `_muon_heads` parameter tags structurally.

    The layers tag their own parameters at construction, but
    `copy.deepcopy(module)` silently DROPS custom attributes on nn.Parameter
    (Parameter.__deepcopy__ clones only data) -- which would put e.g. an
    embedding or conv filter into the Muon group on the copy and desynchronize
    otherwise-identical replicas (this bit us in the DualPipe mirror stages).
    from_model() always calls this, so grouping never depends on attribute
    survival.
    """
    from ..layers import (AttentionResidual, CompressedAttention, GatedMLA,
                          HyperConnections, KDAAttention, LatentMoE, RMSNorm)

    for m in model.modules():
        if isinstance(m, RMSNorm):
            m.weight._no_muon = True
        elif isinstance(m, torch.nn.Embedding):
            m.weight._no_muon = True
        elif isinstance(m, KDAAttention):
            for lin in (m.wq, m.wk, m.wv):
                lin.weight._muon_heads = m.h
            for p in (m.conv_q, m.conv_k, m.conv_v, m.a_log):
                p._no_muon = True
        elif isinstance(m, GatedMLA):
            for lin in (m.w_uq, m.w_uk, m.w_uv):
                lin.weight._muon_heads = m.h
        elif isinstance(m, CompressedAttention):
            m.w_uq.weight._muon_heads = m.h
            if m.mode == "csa":
                m.bias_a._no_muon = m.bias_b._no_muon = True
            else:
                m.bias._no_muon = True
        elif isinstance(m, LatentMoE):
            m.w_router.weight._no_muon = True
        elif isinstance(m, HyperConnections):
            for p in (m.a, m.b_logits, m.c):
                p._no_muon = True
        elif isinstance(m, AttentionResidual):
            m.pseudo_query._no_muon = True
            m.gamma._no_muon = True
    for name, p in model.named_parameters():
        if "lm_head" in name or name.endswith("head.weight"):
            p._no_muon = True


class MuonClip(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95,
                 weight_decay: float = 0.01, ns_gamma: float = 0.2,
                 adamw_betas=(0.9, 0.95), adamw_eps: float = 1e-8,
                 ns_batched: bool = True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_gamma=ns_gamma, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, ns_batched=ns_batched)
        super().__init__(params, defaults)

    # ---------------------------------------------------------------- #
    @classmethod
    def from_model(cls, model: torch.nn.Module, lr: float = 1e-3, **kw):
        """Split params into Muon matrices vs AdamW everything-else."""
        retag_model(model)
        muon, adamw = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and not getattr(p, "_no_muon", False):
                muon.append(p)
            elif p.ndim == 3 and not getattr(p, "_no_muon", False):
                muon.append(p)  # stacked expert weights [E, in, out]: per-expert NS
            else:
                adamw.append(p)
        opt = cls([{"params": muon, "use_muon": True},
                   {"params": adamw, "use_muon": False}], lr=lr, **kw)
        return opt

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_muon", group["params"] and group["params"][0].ndim >= 2):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        lr, mu, wd, gamma = (group["lr"], group["momentum"],
                             group["weight_decay"], group["ns_gamma"])
        batched = group.get("ns_batched", True)
        pending: dict = {}  # (n, m, dtype, device) -> [(p, update3d, n, m)]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum" not in state:
                state["momentum"] = torch.zeros_like(g)
            M = state["momentum"]
            M.mul_(mu).add_(g)
            update = mu * M + g  # Nesterov-style lookahead

            heads = getattr(p, "_muon_heads", None)
            if not batched:  # original per-parameter path (A/B baseline)
                if p.ndim == 3:  # stacked experts: batched NS over dim 0
                    O = newton_schulz(update)
                    n, m = p.shape[-2], p.shape[-1]
                elif heads and p.shape[0] % heads == 0:
                    hd = p.shape[0] // heads
                    O = newton_schulz(update.view(heads, hd, p.shape[1])).reshape_as(p)
                    n, m = hd, p.shape[1]
                else:
                    n, m = p.shape
                    if n < m and m > _GRAM_K_CHUNK:
                        # canonical orientation (UPDATE 26): see batched path
                        O = _ns_wide_safe(update.t().contiguous()).t()
                    else:
                        O = _ns_wide_safe(update)
                O = O * (gamma * math.sqrt(max(n, m)))
                p.mul_(1 - lr * wd).add_(O, alpha=-lr)
                continue
            # shape-batched: normalize every update to [k, n, m] slices, then
            # ONE Newton-Schulz per distinct (n, m) across all params (NS
            # normalizes per slice, so batching is mathematically identical)
            if p.ndim == 3:
                u3, n, m = update, p.shape[-2], p.shape[-1]
            elif heads and p.shape[0] % heads == 0:
                hd = p.shape[0] // heads
                u3, n, m = update.view(heads, hd, p.shape[1]), hd, p.shape[1]
            else:
                u3, n, m = update.unsqueeze(0), p.shape[0], p.shape[1]
            # Canonical orientation (UPDATE 26): a WIDE 2D update (n < m,
            # m past the chunk) enters NS pre-transposed, so NS's internal
            # flip makes X a strided transpose view — the memory layout
            # field-measured DETERMINISTIC on PVC; the contiguous rows<=cols
            # layout of the SAME geometry was not (fuse_proj vs wqkv,
            # dim-1024 96-rank probe). NS(X^T) = NS(X)^T exactly, so only
            # kernel rounding changes. `tr` rides the group key so a
            # canonicalized slab reproduces the proven invocation verbatim.
            tr = p.ndim == 2 and heads is None and n < m and m > _GRAM_K_CHUNK
            if tr:
                u3, n, m = u3.transpose(-2, -1), m, n
            pending.setdefault((n, m, tr, u3.dtype, u3.device), []).append(
                (p, u3, n, m, tr))
        for items in pending.values():
            O_all = _ns_wide_safe(
                torch.cat([u for _, u, _, _, _ in items], dim=0))
            off = 0
            for p, u, n, m, tr in items:
                k = u.shape[0]
                O = O_all[off: off + k]
                if tr:
                    O = O.transpose(-2, -1)
                O = O.reshape(p.shape) * (gamma * math.sqrt(max(n, m)))
                p.mul_(1 - lr * wd).add_(O, alpha=-lr)
                off += k

    def _adamw_step(self, group):
        lr, wd = group["lr"], group["weight_decay"]
        b1, b2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
            state["step"] += 1
            t = state["step"]
            ea, eas = state["exp_avg"], state["exp_avg_sq"]
            ea.mul_(b1).add_(g, alpha=1 - b1)
            eas.mul_(b2).addcmul_(g, g, value=1 - b2)
            denom = (eas / (1 - b2 ** t)).sqrt_().add_(eps)
            p.mul_(1 - lr * wd).addcdiv_(ea, denom, value=-lr / (1 - b1 ** t))

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def qk_clip(self, model: torch.nn.Module, tau: float = 100.0):
        """Apply QK-clip using the max-logit stats recorded in the last forward."""
        model.apply_qk_clip(tau)
