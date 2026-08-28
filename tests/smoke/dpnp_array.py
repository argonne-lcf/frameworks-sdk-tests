#!/usr/bin/env python3
"""Exercise a small dpnp computation on an XPU."""

from __future__ import annotations

import numpy as np
import dpnp


def main() -> None:
    values = dpnp.arange(100, device="gpu")
    squared = dpnp.square(values)
    np.testing.assert_array_equal(dpnp.asnumpy(squared), np.arange(100) ** 2)
    if "gpu" not in str(squared.device).lower():
        raise AssertionError(f"dpnp result is not on a GPU: {squared.device}")
    print(f"PASS device={squared.device} shape={squared.shape}")


if __name__ == "__main__":
    main()
