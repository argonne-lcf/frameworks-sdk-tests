#!/usr/bin/env python3
"""Correctness-oriented 1D DTensor matrix multiplication workload."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.profiler import profile

try:
    from torch.distributed.tensor import DeviceMesh, DTensor, Shard
except ImportError:  # PyTorch 2.0--2.4 compatibility
    from torch.distributed._tensor import DeviceMesh, DTensor, Shard

from torch_setup import (
    get_device,
    get_device_type,
    get_profiler_activities,
    init_distributed,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-dir", type=Path, default=Path("dtensor_traces"))
    args = parser.parse_args()
    if args.dim < 1 or args.steps < 1:
        raise ValueError("--dim and --steps must be positive")

    distributed, rank, world_size = init_distributed()
    device_type, device = get_device_type(), get_device()
    mesh = DeviceMesh(device_type, torch.arange(world_size))
    local_a = torch.full((args.dim, args.dim), 2.0, device=device)
    local_b = torch.full((args.dim, args.dim), 2.0, device=device)
    dtensor_a = DTensor.from_local(local_a, device_mesh=mesh, placements=[Shard(1)])
    dtensor_b = DTensor.from_local(local_b, device_mesh=mesh, placements=[Shard(0)])

    profiler = (
        profile(activities=get_profiler_activities(), record_shapes=True)
        if args.profile
        else nullcontext()
    )
    result = None
    with profiler as captured:
        for step in range(args.steps):
            result = (dtensor_a @ dtensor_b).redistribute(mesh, [Shard(1)])
            distributed.barrier()
            if rank == 0:
                print(f"step={step}", flush=True)

    assert result is not None
    local_result = result.to_local()
    expected = torch.full_like(local_result, 4.0 * args.dim * world_size)
    torch.testing.assert_close(local_result, expected, rtol=0, atol=0)
    if args.profile:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        captured.export_chrome_trace(str(args.trace_dir / f"trace-{rank}-of-{world_size}.json"))
    distributed.destroy_process_group()
    if rank == 0:
        print(f"PASS DTensor matmul world_size={world_size} dim={args.dim}")


if __name__ == "__main__":
    main()
