from __future__ import annotations

import torch
import torch.nn as nn

from ..kernels import rmsnorm


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.weight._no_muon = True  # Muon paper/DeepSeek-V4: norms use AdamW

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)


def rms_normalize(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Weightless RMS normalization (used by AttnRes keys, KDA output, etc.)."""
    dtype = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).to(dtype)
