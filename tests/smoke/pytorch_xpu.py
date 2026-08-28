#!/usr/bin/env python3
"""Verify that the loaded PyTorch build can execute and synchronize XPU work."""

from __future__ import annotations

import torch


def main() -> None:
    print(f"torch={torch.__version__}")
    print(f"torch_file={torch.__file__}")
    print(torch.__config__.show())

    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("PyTorch reports no available XPU")
    count = torch.xpu.device_count()
    if count < 1:
        raise RuntimeError("PyTorch reports zero XPU devices")

    device = torch.device("xpu:0")
    torch.xpu.set_device(device)
    left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    right = torch.eye(4, dtype=torch.float32, device=device)
    actual = left @ right
    torch.xpu.synchronize()
    torch.testing.assert_close(actual.cpu(), torch.arange(16).reshape(4, 4).float())

    xccl = getattr(torch.distributed, "is_xccl_available", lambda: False)()
    if not xccl:
        raise RuntimeError("torch.distributed reports that XCCL is unavailable")
    print(f"PASS xpu_count={count} xccl_available={xccl}")


if __name__ == "__main__":
    main()
