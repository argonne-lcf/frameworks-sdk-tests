#!/usr/bin/env python3
"""Small MPI-to-torch.distributed adapter shared by model workloads."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as torch_dist
from mpi4py import MPI
from torch.profiler import ProfilerActivity


COMM = MPI.COMM_WORLD


def get_device_type() -> str:
    requested = os.environ.get("TEST_DEVICE")
    if requested:
        return requested
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _local_rank() -> int:
    for name in (
        "LOCAL_RANK",
        "PALS_LOCAL_RANKID",
        "MPI_LOCALRANKID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "PMI_LOCAL_RANK",
        "SLURM_LOCALID",
    ):
        value = os.environ.get(name)
        if value is not None:
            return int(value)

    hosts = COMM.allgather(socket.gethostname())
    host = hosts[COMM.rank]
    return sum(candidate == host for candidate in hosts[: COMM.rank])


def get_device() -> torch.device:
    device_type = get_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    index = _local_rank()
    count = torch.xpu.device_count() if device_type == "xpu" else torch.cuda.device_count()
    if count < 1:
        raise RuntimeError(f"no {device_type} devices are visible")
    index %= count
    if device_type == "xpu":
        torch.xpu.set_device(index)
    else:
        torch.cuda.set_device(index)
    return torch.device(device_type, index)


def get_profiler_activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if get_device_type() == "xpu":
        activities.append(ProfilerActivity.XPU)
    elif get_device_type() == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def init_distributed(backend: str | None = None):
    rank, world_size = COMM.rank, COMM.size
    device = get_device()
    if backend is None:
        if hasattr(torch_dist, "get_default_backend_for_device"):
            backend = torch_dist.get_default_backend_for_device(device.type)
        else:
            backend = {"xpu": "xccl", "cuda": "nccl"}.get(device.type, "gloo")

    master = os.environ.get("MASTER_ADDR")
    if not master:
        master = COMM.bcast(socket.gethostname() if rank == 0 else None, root=0)
        os.environ["MASTER_ADDR"] = master
    if "MASTER_PORT" not in os.environ:
        if rank == 0:
            with socket.socket() as port_socket:
                port_socket.bind(("", 0))
                master_port = port_socket.getsockname()[1]
        else:
            master_port = None
        master_port = COMM.bcast(master_port, root=0)
        os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(_local_rank())

    print(
        f"rank={rank}/{world_size} local_rank={os.environ['LOCAL_RANK']} "
        f"host={socket.gethostname()} device={device} backend={backend}",
        flush=True,
    )
    torch_dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return torch_dist, rank, world_size


if __name__ == "__main__":
    distributed, _, _ = init_distributed()
    distributed.destroy_process_group()
