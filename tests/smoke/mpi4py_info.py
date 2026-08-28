#!/usr/bin/env python3
"""Minimal MPI/mpi4py correctness test; valid at one or many ranks."""

from __future__ import annotations

import mpi4py
from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    result = comm.allreduce(rank + 1, op=MPI.SUM)
    expected = size * (size + 1) // 2
    if result != expected:
        raise AssertionError(f"rank {rank}: allreduce produced {result}, expected {expected}")
    print(
        f"PASS rank={rank}/{size} mpi4py={mpi4py.__version__} "
        f"vendor={MPI.get_vendor()[0]} allreduce={result}",
        flush=True,
    )


if __name__ == "__main__":
    main()
