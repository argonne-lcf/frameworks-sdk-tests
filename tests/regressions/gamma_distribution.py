#!/usr/bin/env python3
"""Regression check for large XPU Gamma-distribution sampling."""

from __future__ import annotations

import argparse
import math

import torch


ALPHAS = (1.1644, 1.4164, 2.3335, 10.076)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10_000_000)
    parser.add_argument("--max-z", type=float, default=8.0)
    parser.add_argument("--variance-rtol", type=float, default=0.05)
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    if not torch.xpu.is_available():
        raise RuntimeError("Gamma regression requires an XPU")

    torch.manual_seed(1234)
    for alpha in ALPHAS:
        concentration = torch.tensor(alpha, device="xpu", dtype=torch.float64)
        rate = torch.tensor(1.0, device="xpu", dtype=torch.float64)
        sample = torch.distributions.Gamma(concentration, rate).sample((args.samples,))
        if not torch.isfinite(sample).all():
            raise AssertionError(f"alpha={alpha}: sample contains non-finite values")
        mean, variance = sample.mean().item(), sample.var().item()
        z_score = (mean - alpha) / math.sqrt(alpha / args.samples)
        variance_error = abs(variance - alpha) / alpha
        print(
            f"alpha={alpha:<7} mean={mean:.8f} z={z_score:.3f} "
            f"variance={variance:.8f} variance_rel_error={variance_error:.4%}"
        )
        if abs(z_score) > args.max_z:
            raise AssertionError(f"alpha={alpha}: |mean z-score| {abs(z_score):.3f} > {args.max_z}")
        if variance_error > args.variance_rtol:
            raise AssertionError(
                f"alpha={alpha}: variance relative error {variance_error:.4%} "
                f"> {args.variance_rtol:.4%}"
            )
    print(f"PASS samples={args.samples}")


if __name__ == "__main__":
    main()
