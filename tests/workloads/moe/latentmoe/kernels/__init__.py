"""Triton kernels with pure-PyTorch reference twins.

Every hot path in this library exists twice:

  * ``*_triton``  -- a portable Triton kernel (no CUDA-specific intrinsics),
                     runs on NVIDIA (incl. Grace-Hopper) and Intel XPU via the
                     Triton backend shipped with the respective torch build.
  * ``*_ref``     -- a numerically equivalent pure-PyTorch implementation used
                     on CPU / MPS, and as the ground truth in the unit tests.

`use_triton(x)` decides at runtime which one to run for a given tensor.
Set LATENTMOE_DISABLE_TRITON=1 to force the reference paths everywhere.
"""

from __future__ import annotations

import os

import torch

try:  # Triton has no macOS wheels; Intel/NVIDIA Linux builds bundle it.
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on platform
    HAS_TRITON = False

_TRITON_DEVICES = ("cuda", "xpu")


def use_triton(*tensors: torch.Tensor) -> bool:
    """True if Triton is importable and all tensors live on a Triton-capable device."""
    if not HAS_TRITON or os.environ.get("LATENTMOE_DISABLE_TRITON"):
        return False
    return all(t.device.type in _TRITON_DEVICES for t in tensors)


from .rmsnorm import rmsnorm  # noqa: E402
from .mxfp import mx_quant_dequant, MXFakeQuant  # noqa: E402
from .kda_scan import kda_chunkwise, kda_recurrent_ref  # noqa: E402
from .grouped_gemm import grouped_gemm  # noqa: E402
from .indexer import indexer_scores  # noqa: E402

__all__ = [
    "HAS_TRITON",
    "use_triton",
    "rmsnorm",
    "mx_quant_dequant",
    "MXFakeQuant",
    "kda_chunkwise",
    "kda_recurrent_ref",
    "grouped_gemm",
    "indexer_scores",
]
