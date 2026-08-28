#!/usr/bin/env python3
"""Verify that dpctl discovers at least one Level Zero GPU."""

from __future__ import annotations

import dpctl


def main() -> None:
    devices = dpctl.get_devices(backend="level_zero", device_type="gpu")
    if not devices:
        raise RuntimeError("dpctl found no Level Zero GPU devices")
    for index, device in enumerate(devices):
        print(f"gpu[{index}]={device}")
    print(f"PASS level_zero_gpus={len(devices)} cpu_visible={dpctl.has_cpu_devices()}")


if __name__ == "__main__":
    main()
