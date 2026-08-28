"""Rotary embedding for the decoupled RoPE dims of the V4 attention path.

Kimi-K3 layers are NoPE (no positional encoding at all -- the KDA recurrence
is inherently positional and the hybrid extrapolates to 1M tokens without any
PE modification), so only the CSA/HCA path uses this. `scaling` implements
simple NTK-style wavelength stretching for context extension.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, scaling: float = 1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / ((theta * scaling) ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def cos_sin(self, positions: torch.Tensor):
        """positions: [T] int64 -> cos, sin [T, dim/2] fp32."""
        freqs = positions.float()[:, None] * self.inv_freq[None, :].to(positions.device)
        return freqs.cos(), freqs.sin()

    def rotate(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Rotate last dim. x: [B, T, ..., dim] with the sequence at dim 1;
        positions: [T]."""
        cos, sin = self.cos_sin(positions)
        shape = [1, x.shape[1]] + [1] * (x.dim() - 3) + [self.dim // 2]
        cos, sin = cos.reshape(shape).to(x.dtype), sin.reshape(shape).to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
