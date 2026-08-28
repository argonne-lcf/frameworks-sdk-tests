"""Gated-linear-unit activations.

SwiGLU  (DeepSeek lineage):   silu(g) * u
SiTU-GLU (Kimi-K3):           [b1*tanh(g/b1) * sigmoid(g)] * [b2*tanh(u/b2)]

SiTU-GLU soft-caps *both* branches with tanh (gate capped at b1=4, linear
branch at b2=25), bounding expert outputs -- Kimi-K3 credits this with keeping
optimization stable at 896-expert / 56x sparsity scale.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return F.silu(g) * u


def situ_glu(g: torch.Tensor, u: torch.Tensor, beta1: float = 4.0, beta2: float = 25.0) -> torch.Tensor:
    gate = beta1 * torch.tanh(g / beta1) * torch.sigmoid(g)
    lin = beta2 * torch.tanh(u / beta2)
    return gate * lin


def glu_act(gu: torch.Tensor, kind: str, beta1: float = 4.0, beta2: float = 25.0) -> torch.Tensor:
    """gu: [..., 2H] fused gate/up projection output."""
    g, u = gu.chunk(2, dim=-1)
    if kind == "swiglu":
        return swiglu(g, u)
    if kind == "situ_glu":
        return situ_glu(g, u, beta1, beta2)
    raise ValueError(kind)
