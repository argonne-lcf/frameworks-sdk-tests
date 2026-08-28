#!/usr/bin/env python3
"""Validate GPU-aware MPI Allreduce directly on PyTorch XPU tensors."""

import os
import sys

from mpi4py import MPI
import torch


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if world_size < 2:
        if rank == 0:
            print("FAIL: this test requires at least two MPI ranks", flush=True)
        return 1

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        local_rank = local_comm.Get_rank()
        visible_xpus = torch.xpu.device_count()
        if visible_xpus < 1:
            print(f"[rank {rank}] FAIL: no XPU is visible", flush=True)
            local_ready = False
        else:
            device = torch.device(f"xpu:{local_rank % visible_xpus}")
            torch.xpu.set_device(device)
            local_ready = True

        all_ready = comm.allreduce(local_ready, op=MPI.LAND)
        if not all_ready:
            return 1

        elements = int(os.environ.get("TEST_MPI_ELEMENTS", "1024"))
        if elements < 1:
            raise ValueError("TEST_MPI_ELEMENTS must be positive")

        source = torch.full(
            (elements,), float(rank + 1), dtype=torch.float32, device=device
        )
        result = torch.empty_like(source)
        comm.Allreduce([source, MPI.FLOAT], [result, MPI.FLOAT], op=MPI.SUM)
        torch.xpu.synchronize()

        expected = float(world_size * (world_size + 1) // 2)
        max_error = float((result - expected).abs().max().cpu().item())
        local_ok = max_error == 0.0
        print(
            f"[rank {rank}/{world_size}] device={device} "
            f"expected={expected:g} max_abs_error={max_error:g} "
            f"status={'PASS' if local_ok else 'FAIL'}",
            flush=True,
        )

        all_ok = comm.allreduce(local_ok, op=MPI.LAND)
        if rank == 0:
            print(
                "GPU-aware mpi4py Allreduce: " + ("PASS" if all_ok else "FAIL"),
                flush=True,
            )
        return 0 if all_ok else 1
    finally:
        local_comm.Free()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"MPI/XPU all-reduce failed: {type(error).__name__}: {error}", flush=True)
        raise
