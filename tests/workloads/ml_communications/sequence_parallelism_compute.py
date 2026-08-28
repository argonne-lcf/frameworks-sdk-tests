#!/usr/bin/env python3
"""Bounded sequence-parallel communication/compute validation workload."""

from __future__ import annotations

import argparse
import os
import socket
import time

from mpi4py import MPI
import torch
import torch.distributed as dist


COMM = MPI.COMM_WORLD
ALL_GATHER_SINGLE = getattr(dist, "all_gather_single", None) or getattr(
    dist, "all_gather_into_tensor"
)
REDUCE_SCATTER_SINGLE = getattr(dist, "reduce_scatter_single", None) or getattr(
    dist, "reduce_scatter_tensor"
)


def synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            requested = "xpu"
        elif torch.cuda.is_available():
            requested = "cuda"
        else:
            requested = "cpu"

    local_comm = COMM.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        local_rank = local_comm.Get_rank()
    finally:
        local_comm.Free()

    if requested == "cpu":
        return torch.device("cpu")
    count = (
        torch.xpu.device_count() if requested == "xpu" else torch.cuda.device_count()
    )
    if count < 1:
        raise RuntimeError(f"no {requested} devices are visible")
    index = local_rank % count
    if requested == "xpu":
        torch.xpu.set_device(index)
    elif requested == "cuda":
        torch.cuda.set_device(index)
    else:
        raise ValueError(f"unsupported device: {requested}")
    return torch.device(requested, index)


def initialize(device: torch.device) -> None:
    rank, world_size = COMM.rank, COMM.size
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        master_addr = COMM.bcast(
            socket.gethostname() if rank == 0 else None, root=0
        )
        os.environ["MASTER_ADDR"] = master_addr
    if "MASTER_PORT" not in os.environ:
        if rank == 0:
            with socket.socket() as port_socket:
                port_socket.bind(("", 0))
                port = port_socket.getsockname()[1]
        else:
            port = None
        os.environ["MASTER_PORT"] = str(COMM.bcast(port, root=0))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if hasattr(dist, "get_default_backend_for_device"):
        backend = dist.get_default_backend_for_device(device.type)
    else:
        backend = {"xpu": "xccl", "cuda": "nccl"}.get(device.type, "gloo")
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size, init_method="env://"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    args = parser.parse_args()

    rank, world_size = COMM.rank, COMM.size
    if world_size < 2:
        raise ValueError("sequence-parallel validation requires at least two ranks")
    if min(
        args.sequence_length,
        args.hidden_size,
        args.iterations,
    ) < 1 or args.warmup_iterations < 0:
        raise ValueError("sizes/iterations must be positive and warmup non-negative")
    if args.sequence_length % world_size or args.hidden_size % world_size:
        raise ValueError("sequence length and hidden size must be divisible by world size")

    device = select_device(args.device)
    initialize(device)
    try:
        local_sequence = args.sequence_length // world_size
        dtype = torch.bfloat16 if device.type != "cpu" else torch.float32

        # Explicit collective oracles before the timed compute pattern.
        local = torch.full(
            (local_sequence, 1, args.hidden_size),
            float(rank + 1),
            dtype=dtype,
            device=device,
        )
        gathered = torch.empty(
            (args.sequence_length, 1, args.hidden_size), dtype=dtype, device=device
        )
        ALL_GATHER_SINGLE(gathered, local)
        for source_rank in range(world_size):
            chunk = gathered[
                source_rank * local_sequence : (source_rank + 1) * local_sequence
            ]
            expected = torch.full_like(chunk, float(source_rank + 1))
            torch.testing.assert_close(chunk, expected, rtol=0, atol=0)

        reduce_source = torch.ones_like(gathered)
        reduced = torch.empty_like(local)
        REDUCE_SCATTER_SINGLE(reduced, reduce_source)
        torch.testing.assert_close(
            reduced, torch.full_like(reduced, float(world_size)), rtol=0, atol=0
        )

        torch.manual_seed(1234)
        local = torch.rand_like(local)
        weight_in = torch.rand(
            (args.hidden_size // world_size, args.hidden_size),
            dtype=dtype,
            device=device,
        )
        weight_out = torch.rand(
            (args.hidden_size, args.hidden_size // world_size),
            dtype=dtype,
            device=device,
        )

        def step() -> None:
            ALL_GATHER_SINGLE(gathered, local)
            intermediate = torch.matmul(gathered, weight_in.t())
            intermediate = torch.matmul(intermediate, weight_out.t())
            REDUCE_SCATTER_SINGLE(local, intermediate)

        for _ in range(args.warmup_iterations):
            step()
        synchronize(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            step()
        synchronize(device)
        elapsed = time.perf_counter() - started

        if not bool(torch.isfinite(local).all().item()):
            raise FloatingPointError(f"rank {rank}: non-finite sequence-parallel output")
        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float32, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"PASS sequence parallelism ranks={world_size} device={device.type} "
                f"shape={args.sequence_length}x{args.hidden_size} "
                f"max_elapsed={elapsed_tensor.item():.6f}s",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
