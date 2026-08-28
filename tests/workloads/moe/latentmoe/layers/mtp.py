"""Multi-Token Prediction (MTP) head.

DeepSeek-V3/V4 style: one extra depth-1 transformer module predicts token t+2
from (final hidden state at t, embedding of token t+1):

    x_t = W_proj [ RMSNorm(h_t) ; RMSNorm(Emb(tok_{t+1})) ]   -> tiny block -> shared LM head

Trained with a weighted CE term; at inference it doubles as a self-speculative
draft head (propose t+2, verify with the main model next step).

Kimi-K3 flavor (mtp_fuse_layers=True): EAGLE-3-style fusion -- the head reads
low/mid/high-level features (outputs of an early, middle, and the final block)
instead of only the last hidden state.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe import DenseFFN
from .norm import RMSNorm


class _TinyBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_hidden: int, eps: float):
        super().__init__()
        self.h = n_heads
        self.n1, self.n2 = RMSNorm(dim, eps), RMSNorm(dim, eps)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.ffn = DenseFFN(dim, ffn_hidden, "swiglu")

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.wqkv(self.n1(x)).chunk(3, dim=-1)
        shp = (B, T, self.h, D // self.h)
        q, k, v = (t.view(shp).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        x = x + self.wo(o.transpose(1, 2).reshape(B, T, D))
        return x + self.ffn(self.n2(x))


class MTPHead(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_hidden: int,
                 fuse_layers: bool = False, eps: float = 1e-5):
        super().__init__()
        self.fuse = fuse_layers
        if fuse_layers:
            self.fuse_proj = nn.Linear(3 * dim, dim, bias=False)
        self.norm_h = RMSNorm(dim, eps)
        self.norm_e = RMSNorm(dim, eps)
        self.proj = nn.Linear(2 * dim, dim, bias=False)
        self.block = _TinyBlock(dim, n_heads, ffn_hidden, eps)
        self.out_norm = RMSNorm(dim, eps)

    def forward(self, h: torch.Tensor, next_emb: torch.Tensor,
                feats: list[torch.Tensor] | None = None) -> torch.Tensor:
        """h: [B,T,d] final hidden; next_emb: [B,T,d] embedding of token t+1.
        Returns hidden states whose LM-head logits predict token t+2."""
        if self.fuse and feats is not None:
            h = h + self.fuse_proj(torch.cat(feats, dim=-1))
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(next_emb)], dim=-1))
        return self.out_norm(self.block(x))
