"""Kimi Delta Attention layer (Kimi-K3 / Kimi Linear).

A linear-attention layer whose per-head state S in R^{dk x dv} is updated by a
*gated delta rule* with channel-wise, lower-bounded decay:

    q_t, k_t = L2Norm(Swish(ShortConv(W_{q,k} x_t)))     (per head)
    v_t      = Swish(ShortConv(W_v x_t))
    log a_t  = gmin * Sigmoid(e^{A_h} * (W_a x_t))       in (gmin, 0)^{dk}
    beta_t   = Sigmoid(W_b x_t)                          in (0, 1)
    S_t      = (I - beta_t k_t k_t^T) Diag(a_t) S_{t-1} + beta_t k_t v_t^T
    y_t      = W_o [ Sigmoid(W_g x_t) . RMSNorm(S_t^T q_t) ]

Constant O(dk*dv) state per head => O(1) memory in sequence length: this is
what carries Kimi-K3 to 1M-token contexts on 3 out of every 4 layers.
NoPE: the recurrence itself encodes position. Inference keeps (conv tail,
scan state) as its entire "KV cache".
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kernels.kda_scan import kda_scan
from .norm import RMSNorm


def _causal_depthwise_conv(x: torch.Tensor, weight: torch.Tensor, cache: dict | None, key: str):
    """x: [B, T, C]; weight: [C, 1, K] depthwise. Left-pads with cached tail."""
    B, T, C = x.shape
    K = weight.shape[-1]
    xt = x.transpose(1, 2)  # [B, C, T]
    if cache is not None and key in cache:
        xt = torch.cat([cache[key], xt], dim=-1)
        pad = 0
    else:
        pad = K - 1
    if cache is not None:
        cache[key] = xt[..., -(K - 1):].detach()
    y = F.conv1d(xt, weight, groups=C, padding=pad)
    if pad:
        y = y[..., : xt.shape[-1]]
    return y[..., -T:].transpose(1, 2)


class KDAAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, head_dim: int, conv_size: int = 4,
                 gmin: float = -5.0, chunk: int = 16, eps: float = 1e-5):
        super().__init__()
        self.h, self.dk, self.gmin, self.chunk = n_heads, head_dim, gmin, chunk
        inner = n_heads * head_dim
        self.wq = nn.Linear(dim, inner, bias=False)
        self.wk = nn.Linear(dim, inner, bias=False)
        self.wv = nn.Linear(dim, inner, bias=False)
        for lin in (self.wq, self.wk, self.wv):  # K3: per-head Muon orthogonalization
            lin.weight._muon_heads = n_heads
        self.conv_q = nn.Parameter(torch.randn(inner, 1, conv_size) / math.sqrt(conv_size))
        self.conv_k = nn.Parameter(torch.randn(inner, 1, conv_size) / math.sqrt(conv_size))
        self.conv_v = nn.Parameter(torch.randn(inner, 1, conv_size) / math.sqrt(conv_size))
        for p in (self.conv_q, self.conv_k, self.conv_v):
            p._no_muon = True
        self.w_a = nn.Linear(dim, inner, bias=False)
        self.a_log = nn.Parameter(torch.zeros(n_heads))
        self.a_log._no_muon = True
        self.w_beta = nn.Linear(dim, n_heads, bias=False)
        self.w_gate = nn.Linear(dim, inner, bias=False)
        self.out_norm = RMSNorm(head_dim, eps)
        self.wo = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor, cache: dict | None = None, positions=None) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.h, self.dk

        q = _causal_depthwise_conv(self.wq(x), self.conv_q, cache, "cq")
        k = _causal_depthwise_conv(self.wk(x), self.conv_k, cache, "ck")
        v = _causal_depthwise_conv(self.wv(x), self.conv_v, cache, "cv")
        q = F.normalize(F.silu(q).view(B, T, H, D), dim=-1)
        k = F.normalize(F.silu(k).view(B, T, H, D), dim=-1)
        v = F.silu(v).view(B, T, H, D)

        # channel-wise lower-bounded log decay
        z = self.w_a(x).view(B, T, H, D) * torch.exp(self.a_log)[None, None, :, None]
        alpha = torch.exp(self.gmin * torch.sigmoid(z))
        beta = torch.sigmoid(self.w_beta(x))  # [B, T, H]

        to_bhtd = lambda t: t.permute(0, 2, 1, 3).contiguous()  # noqa: E731
        state = cache.get("state") if cache is not None else None
        o, state = kda_scan(
            to_bhtd(q), to_bhtd(k), to_bhtd(v), to_bhtd(alpha),
            beta.permute(0, 2, 1).contiguous(), state, chunk=self.chunk,
        )
        if cache is not None:
            cache["state"] = state.detach()

        o = o.permute(0, 2, 1, 3)  # [B, T, H, D]
        o = self.out_norm(o).reshape(B, T, H * D)
        o = torch.sigmoid(self.w_gate(x)) * o
        return self.wo(o.to(x.dtype))
