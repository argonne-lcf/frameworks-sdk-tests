"""Gated Multi-head Latent Attention (Kimi-K3's global-attention layers).

MLA (DeepSeek-V2 lineage): keys/values are up-projected from a single shared
low-rank latent c_t = W_dkv x_t, and *only the latent is cached* -- the KV
cache is kv_latent_dim wide instead of 2 * n_heads * head_dim. Kimi-K3 adds a
full-rank sigmoid output gate ("Gated MLA") and runs these layers NoPE
(positions come from the interleaved KDA layers; NoPE is what lets the hybrid
extrapolate to 1M tokens).

For clarity we materialize K/V from the cached latent at attention time; the
production "absorbed" trick (folding W_uk into the query and W_uv into the
output projection so attention runs directly in latent space) is a pure
refactoring of the same math.

The layer records max attention-logit stats per head when `track_qk` is set,
feeding MuonClip's QK-clip (Kimi-K2/K3 training-stability mechanism).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class GatedMLA(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int,
                 kv_latent_dim: int, q_latent_dim: int, eps: float = 1e-5,
                 dropout: float = 0.0):
        super().__init__()
        self.h, self.dh = n_heads, head_dim
        inner = n_heads * head_dim
        self.w_dq = nn.Linear(dim, q_latent_dim, bias=False)
        self.q_norm = RMSNorm(q_latent_dim, eps)
        self.w_uq = nn.Linear(q_latent_dim, inner, bias=False)
        self.w_dkv = nn.Linear(dim, kv_latent_dim, bias=False)
        self.kv_norm = RMSNorm(kv_latent_dim, eps)
        self.w_uk = nn.Linear(kv_latent_dim, inner, bias=False)
        self.w_uv = nn.Linear(kv_latent_dim, inner, bias=False)
        for lin in (self.w_uq, self.w_uk, self.w_uv):  # K3: per-head Muon orthogonalization
            lin.weight._muon_heads = n_heads
        self.w_gate = nn.Linear(dim, inner, bias=False)
        self.wo = nn.Linear(inner, dim, bias=False)
        self.dropout = dropout
        self.track_qk = False
        self.register_buffer("max_qk_logit", torch.zeros(n_heads), persistent=False)

    def forward(self, x: torch.Tensor, cache: dict | None = None, positions=None) -> torch.Tensor:
        B, T, _ = x.shape
        H, Dh = self.h, self.dh

        q = self.w_uq(self.q_norm(self.w_dq(x))).view(B, T, H, Dh)
        c = self.kv_norm(self.w_dkv(x))  # [B, T, latent] -- the whole KV cache

        if cache is not None:
            c = torch.cat([cache["kv_latent"], c], dim=1) if "kv_latent" in cache else c
            cache["kv_latent"] = c.detach()
        S = c.shape[1]

        k = self.w_uk(c).view(B, S, H, Dh)
        v = self.w_uv(c).view(B, S, H, Dh)

        if self.track_qk and self.training:
            with torch.no_grad():
                logits = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) / math.sqrt(Dh)
                self.max_qk_logit = logits.abs().amax(dim=(0, 2, 3))

        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, *, Dh]
        causal = T > 1  # decode (T==1): every cached position is attendable
        if causal and S != T:  # chunked prefill with an existing cache
            mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril_(S - T)
            o = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask,
                                               dropout_p=self.dropout if self.training else 0.0)
        else:
            o = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal,
                                               dropout_p=self.dropout if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, T, H * Dh)
        o = torch.sigmoid(self.w_gate(x)) * o
        return self.wo(o)

    @torch.no_grad()
    def qk_clip_(self, tau: float):
        """MuonClip's QK-clip: rescale per-head q/k projections when the max
        attention logit exceeded tau (scale sqrt(tau/max) on each side)."""
        for h in range(self.h):
            s_max = float(self.max_qk_logit[h])
            if s_max > tau:
                g = math.sqrt(tau / s_max)
                rows = slice(h * self.dh, (h + 1) * self.dh)
                self.w_uq.weight[rows].mul_(g)
                self.w_uk.weight[rows].mul_(g)
