#!/usr/bin/env python3
"""Exercise DTensor redistribution through a two-dimensional device mesh."""

from __future__ import annotations

import argparse

import torch

try:
    from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard
except ImportError:  # PyTorch 2.0--2.4 compatibility
    from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from torch_setup import get_device, get_device_type, init_distributed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument(
        "--mesh-cols",
        type=int,
        default=1,
        help="second mesh dimension; world size must be divisible by it",
    )
    args = parser.parse_args()
    if args.dim < 1 or args.mesh_cols < 1:
        raise ValueError("--dim and --mesh-cols must be positive")

    distributed, rank, world_size = init_distributed()
    if world_size < 2 or world_size % args.mesh_cols:
        raise ValueError(
            "world size must be at least two and divisible by --mesh-cols"
        )
    mesh_rows = world_size // args.mesh_cols
    if args.dim % mesh_rows:
        raise ValueError("--dim must be divisible by the first mesh dimension")

    device_type, device = get_device_type(), get_device()
    mesh_ranks = torch.arange(world_size).reshape(mesh_rows, args.mesh_cols)
    mesh = DeviceMesh(device_type, mesh_ranks, mesh_dim_names=("rows", "cols"))

    row = torch.arange(args.dim, dtype=torch.float32, device=device)
    full_a = row[:, None].expand(args.dim, args.dim).contiguous()
    full_b = torch.eye(args.dim, dtype=torch.float32, device=device)
    replicated = [Replicate(), Replicate()]
    dtensor_a = DTensor.from_local(full_a, device_mesh=mesh, placements=replicated)
    dtensor_b = DTensor.from_local(full_b, device_mesh=mesh, placements=replicated)

    product = dtensor_a @ dtensor_b
    sharded = product.redistribute(
        device_mesh=mesh, placements=[Shard(0), Replicate()]
    )
    round_trip = sharded.redistribute(device_mesh=mesh, placements=replicated)
    torch.testing.assert_close(round_trip.to_local(), full_a, rtol=0, atol=0)

    distributed.barrier()
    distributed.destroy_process_group()
    if rank == 0:
        print(
            f"PASS 2D DTensor redistribution mesh={mesh_rows}x{args.mesh_cols} "
            f"dim={args.dim}",
            flush=True,
        )


if __name__ == "__main__":
    main()
