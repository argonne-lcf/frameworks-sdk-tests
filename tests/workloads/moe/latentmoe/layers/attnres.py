"""Attention Residuals (AttnRes), Kimi-K3.

Each block gets, in addition to the ordinary residual stream, a learned
mixture over the *outputs of all preceding blocks* (and the embedding):

    alpha_{i->l} = phi(q_l, k_i) / sum_j phi(q_l, k_j),
    phi(q, k)   = exp(q^T RMSNorm(k)),   q_l = w_l  (a learnable pseudo-query)
    res_l       = sum_i alpha_{i->l} h_i

so layer l can selectively pull features from any earlier depth per token.
K3 uses a block variant (partition depth into ~8 blocks) to bound the memory
of keeping all previous outputs; we do the same with `attnres_block`:
sources = [embedding] + [completed block boundaries] + [layers in current block].

w_l inits to zero => uniform mixture; the output enters the residual through a
zero-init scale, so training starts exactly at the standard architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .norm import rms_normalize


class AttentionResidual(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.zeros(1))  # zero-init: starts as a no-op
        self.pseudo_query._no_muon = True
        self.gamma._no_muon = True

    def forward(self, sources: list[torch.Tensor]) -> torch.Tensor:
        """sources: list of [B, T, d] from earlier depths -> [B, T, d]."""
        K = torch.stack(sources, dim=2)                      # [B, T, S, d]
        Kn = rms_normalize(K)
        logits = torch.einsum("btsd,d->bts", Kn.float(), self.pseudo_query.float())
        alpha = torch.softmax(logits, dim=-1).to(K.dtype)
        mix = torch.einsum("bts,btsd->btd", alpha, K)
        return self.gamma.to(K.dtype) * mix
