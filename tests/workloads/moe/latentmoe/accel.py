"""Device / backend abstraction layer.

The whole library is written against this thin shim so the same code runs on:
  * NVIDIA GPUs (``cuda``)             -- incl. Grace-Hopper GH200 (aarch64 + H100)
  * Intel GPUs (``xpu``)               -- Ponte Vecchio / Falcon Shores, PyTorch XPU build
  * Apple Silicon (``mps``) and CPU    -- for functional testing only

Rules used elsewhere in the code base:
  * never call ``torch.cuda.*`` directly -- use :func:`device_module`
  * never hardcode a distributed backend  -- use :func:`default_dist_backend`
  * Triton is optional; every Triton kernel has a pure-PyTorch reference twin
    (see ``latentmoe.kernels``).
"""

from __future__ import annotations

import functools
import os

import torch

__all__ = [
    "best_device",
    "device_module",
    "default_dist_backend",
    "supports_bf16",
    "autocast_dtype",
    "synchronize",
    "manual_seed_all",
]


@functools.lru_cache(maxsize=None)
def best_device() -> torch.device:
    """Pick the best available accelerator. Override with LATENTMOE_DEVICE=..."""
    forced = os.environ.get("LATENTMOE_DEVICE")
    if forced:
        return torch.device(forced)
    # torch.accelerator is the official device-agnostic API (torch >= 2.4/2.5)
    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        return torch.device(torch.accelerator.current_accelerator().type)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_module(device: torch.device | str | None = None):
    """Return the ``torch.<type>`` module (torch.cuda / torch.xpu / ...) for a device."""
    dev = torch.device(device) if device is not None else best_device()
    return getattr(torch, dev.type, None)


def default_dist_backend(device: torch.device | str | None = None) -> str:
    """Collective backend for the device: nccl (NVIDIA), xccl/ccl (Intel), gloo (CPU)."""
    dev = torch.device(device) if device is not None else best_device()
    if dev.type == "cuda":
        return "nccl"
    if dev.type == "xpu":
        # 'xccl' is the in-tree backend (torch>=2.7); oneccl_bindings register 'ccl'.
        if hasattr(torch.distributed, "is_xccl_available") and torch.distributed.is_xccl_available():
            return "xccl"
        return "ccl"
    return "gloo"


def supports_bf16(device: torch.device | str | None = None) -> bool:
    dev = torch.device(device) if device is not None else best_device()
    if dev.type == "cuda":
        return torch.cuda.is_bf16_supported()
    if dev.type == "xpu":
        return True  # all PyTorch-supported Intel GPUs support bf16
    if dev.type == "mps":
        try:
            torch.zeros(1, device="mps", dtype=torch.bfloat16)
            return True
        except Exception:
            return False
    return True  # CPU bf16 emulation always works (slowly)


def autocast_dtype(device: torch.device | str | None = None) -> torch.dtype:
    return torch.bfloat16 if supports_bf16(device) else torch.float32


def synchronize(device: torch.device | str | None = None) -> None:
    mod = device_module(device)
    if mod is not None and hasattr(mod, "synchronize"):
        mod.synchronize()


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    for name in ("cuda", "xpu", "mps"):
        mod = getattr(torch, name, None)
        if mod is not None and hasattr(mod, "manual_seed_all"):
            try:
                mod.manual_seed_all(seed)
            except Exception:
                pass
