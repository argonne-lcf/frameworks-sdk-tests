"""Manifold-Constrained Hyper-Connections (mHC), DeepSeek-V4.

Hyper-connections widen the residual stream to n_hc parallel streams
X in R^{n_hc x d}. Each block l mixes them with three maps:

    X_{l+1} = B_l X_l + c_l (x) F_l(a_l X_l)

  a_l in R^{n}      : combines streams into the block input
  B_l in R^{n x n}  : stream-mixing residual map
  c_l in R^{n}      : distributes the block output back onto the streams

V4's contribution: B_l is constrained to the *Birkhoff polytope* (doubly
stochastic matrices, row/col sums = 1, entries >= 0) via t_max=20 iterations of
Sinkhorn-Knopp on exp(B_tilde). This bounds ||B_l||_2 <= 1, so signal norms can
not blow up across hundreds of layers -- the "manifold constraint" that makes
hyper-connections trainable at scale.

Init makes the whole thing exactly a standard residual network: streams start
as n copies of the embedding, a = 1/n (mean), B ~ I (large diagonal logit),
c = 1 (broadcast add), and the model reads out the stream mean.

mHC parameters are optimized with AdamW, not Muon (they are not "matrices that
act on features" in the Muon sense) -- flagged via `_no_muon`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sinkhorn(logits: torch.Tensor, n_iters: int = 20) -> torch.Tensor:
    """Project exp(logits) onto the Birkhoff polytope by alternating normalization."""
    M = torch.exp(logits.float() - logits.float().max())
    for _ in range(n_iters):
        M = M / M.sum(dim=-2, keepdim=True).clamp_min(1e-12)  # columns
        M = M / M.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # rows
    return M


class HyperConnections(nn.Module):
    """Per-block mHC maps. The model owns the widened stream tensor."""

    def __init__(self, n_streams: int, sinkhorn_iters: int = 20, diag_init: float = 4.0):
        super().__init__()
        self.n = n_streams
        self.sinkhorn_iters = sinkhorn_iters
        self.a = nn.Parameter(torch.full((n_streams,), 1.0 / n_streams))
        self.b_logits = nn.Parameter(torch.eye(n_streams) * diag_init)
        self.c = nn.Parameter(torch.ones(n_streams))
        for p in (self.a, self.b_logits, self.c):
            p._no_muon = True

    def read(self, X: torch.Tensor) -> torch.Tensor:
        """X: [B, T, n, d] -> block input [B, T, d]."""
        return torch.einsum("btnd,n->btd", X, self.a.to(X.dtype))

    def write(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """X_{l+1} = B X + c (x) y.   y: [B, T, d]."""
        Bmat = sinkhorn(self.b_logits, self.sinkhorn_iters).to(X.dtype)
        mixed = torch.einsum("mn,btnd->btmd", Bmat, X)
        return mixed + self.c.to(X.dtype)[None, None, :, None] * y.unsqueeze(2)

    @staticmethod
    def expand(x: torch.Tensor, n_streams: int) -> torch.Tensor:
        return x.unsqueeze(2).expand(-1, -1, n_streams, -1).contiguous()

    @staticmethod
    def collapse(X: torch.Tensor) -> torch.Tensor:
        return X.mean(dim=2)
