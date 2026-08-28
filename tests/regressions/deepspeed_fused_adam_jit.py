#!/usr/bin/env python3
"""Compile and load DeepSpeed's FusedAdam extension on XPU."""

from __future__ import annotations

import deepspeed
import torch
from deepspeed.ops.op_builder import FusedAdamBuilder


def main() -> None:
    if not torch.xpu.is_available():
        raise RuntimeError("DeepSpeed FusedAdam JIT validation requires an XPU")
    accelerator = deepspeed.accelerator.get_accelerator()
    print(f"deepspeed={deepspeed.__version__} accelerator={accelerator.device_name()}")
    extension = FusedAdamBuilder().load()
    if extension is None:
        raise RuntimeError("FusedAdamBuilder.load() returned None")
    print(f"PASS extension={extension.__name__}")


if __name__ == "__main__":
    main()
