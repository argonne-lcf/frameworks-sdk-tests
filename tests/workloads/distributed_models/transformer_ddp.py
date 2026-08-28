#!/usr/bin/env python3
"""Bounded synthetic Transformer DDP training workload."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_setup import get_device, init_distributed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    if args.dimension % args.heads:
        raise ValueError("--dimension must be divisible by --heads")

    _, rank, world_size = init_distributed()
    device = get_device()
    torch.manual_seed(1234 + rank)
    model = torch.nn.Transformer(
        d_model=args.dimension,
        nhead=args.heads,
        num_encoder_layers=args.layers,
        num_decoder_layers=args.layers,
        batch_first=True,
    ).to(device)
    model = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 * world_size)

    for step in range(args.steps):
        source = torch.randn(
            args.batch_size, args.sequence_length, args.dimension, device=device
        )
        target = torch.randn_like(source)
        optimizer.zero_grad(set_to_none=True)
        output = model(source, target)
        loss = torch.nn.functional.mse_loss(output, target)
        if not torch.isfinite(loss):
            raise AssertionError(f"rank {rank}: non-finite loss")
        loss.backward()
        optimizer.step()
        reduced = loss.detach().clone()
        dist.all_reduce(reduced)
        if rank == 0:
            print(f"step={step} mean_loss={reduced.item() / world_size:.6f}")

    dist.destroy_process_group()
    if rank == 0:
        print(f"PASS Transformer DDP ranks={world_size}")


if __name__ == "__main__":
    main()
