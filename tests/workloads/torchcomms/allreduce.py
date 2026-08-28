#!/usr/bin/env python3
import os
import sys
import torch
import torchcomms


def _iget(*names, default=None):
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return int(value)
    return default


def main():
    # Use a launcher that exports an actual world size (torchrun is preferred).
    rank = _iget("TORCHCOMM_RANK", "RANK", "PMIX_RANK", "PALS_RANKID", default=0)
    world = _iget("TORCHCOMM_SIZE", "WORLD_SIZE", "PMI_SIZE", default=1)
    local_rank = _iget(
        "TORCHCOMM_LOCAL_RANK",
        "LOCAL_RANK",
        "PALS_LOCAL_RANKID",
        "MPI_LOCALRANKID",
        default=rank,
    )
    os.environ.setdefault("TORCHCOMM_RANK", str(rank))
    os.environ.setdefault("TORCHCOMM_SIZE", str(world))
    os.environ.setdefault("TORCHCOMM_LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if world < 2:
        raise RuntimeError(
            "TorchComms all-reduce requires at least two ranks; launch with torchrun"
        )

    if not hasattr(torch, "xpu"):
        print(f"[rank {rank}] torch.xpu not available", flush=True)
        return 2

    ndev = torch.xpu.device_count()
    if ndev <= 0:
        print(f"[rank {rank}] no XPUs visible", flush=True)
        return 2

    dev = torch.device(f"xpu:{local_rank % ndev}")

    print(f"[rank {rank}/{world} local {local_rank}] using {dev}", flush=True)

    comm = None
    try:
        comm = torchcomms.new_comm("xccl", dev, name="main_comm")
        print(f"[rank {rank}] new_comm OK", flush=True)

        # Each rank contributes (rank+1), so SUM should be 1+2+...+world
        x = torch.ones(1024, device=dev, dtype=torch.float32) * float(rank + 1)

        # IMPORTANT: your torchcomms binding requires op + async_op explicitly
        work = comm.all_reduce(x, torchcomms.ReduceOp.SUM, async_op=True)
        work.wait()

        expected = float(world * (world + 1) / 2)
        got = float(x[0].item())

        if rank == 0:
            print(
                f"[rank 0] allreduce expected={expected} got={got} x[:4]={x[:4].tolist()}",
                flush=True,
            )

        # Check correctness on every rank
        if abs(got - expected) > 1e-4:
            print(f"[rank {rank}] FAIL expected={expected} got={got}", flush=True)
            return 1

        return 0

    except Exception as e:
        print(f"[rank {rank}] ERROR: {type(e).__name__}: {e}", flush=True)
        raise

    finally:
        if comm is not None:
            comm.finalize()
            print(f"[rank {rank}] finalize OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
