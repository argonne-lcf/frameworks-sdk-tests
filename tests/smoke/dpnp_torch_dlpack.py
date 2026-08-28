#!/usr/bin/env python3
"""Verify zero-copy DLPack sharing in both PyTorch/dpnp directions."""

from __future__ import annotations

import dpnp
import torch


def main() -> None:
    torch_array = torch.arange(4, dtype=torch.float32, device="xpu")
    dpnp_view = dpnp.from_dlpack(torch_array)
    torch_array[0] = -2
    if float(dpnp_view[0]) != -2:
        raise AssertionError("dpnp did not observe the PyTorch mutation")

    dpnp_array = dpnp.arange(4, dtype=dpnp.float32, device="gpu")
    torch_view = torch.from_dlpack(dpnp_array)
    dpnp_array[0] = -3
    torch.xpu.synchronize()
    if float(torch_view[0].cpu()) != -3:
        raise AssertionError("PyTorch did not observe the dpnp mutation")
    print(f"PASS torch_to_dpnp={dpnp_view.device} dpnp_to_torch={torch_view.device}")


if __name__ == "__main__":
    main()
