#!/usr/bin/env python3
"""Write, reload, validate, and optionally retain one checkpoint per MPI rank."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from mpi4py import MPI
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--elements", type=int, default=1_048_576)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    args.output_folder.mkdir(parents=True, exist_ok=True)
    run_id = os.environ.get("FRAMEWORKS_TEST_RUN_ID", str(os.getppid()))
    path = args.output_folder / f"frameworks-checkpoint-{run_id}-{comm.rank}-of-{comm.size}.pt"
    expected = torch.arange(args.elements, dtype=torch.float32) + comm.rank
    torch.save({"rank": comm.rank, "tensor": expected}, path)
    comm.Barrier()
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if loaded["rank"] != comm.rank:
        raise AssertionError(f"checkpoint rank {loaded['rank']} != {comm.rank}")
    torch.testing.assert_close(loaded["tensor"], expected, rtol=0, atol=0)
    print(f"PASS rank={comm.rank}/{comm.size} path={path} bytes={path.stat().st_size}")
    comm.Barrier()
    if not args.keep:
        path.unlink()


if __name__ == "__main__":
    main()
