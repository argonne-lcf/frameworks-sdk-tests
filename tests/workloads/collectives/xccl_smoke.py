#!/usr/bin/env python3
import os
import sys
import socket
import datetime
import traceback

import torch
import torch.distributed as dist


def env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value not in (None, ""):
            return int(value)
    return default


def work_wait(work):
    """Wait for a c10d Work object, with a device-sync fallback."""
    for m in ("wait", "synchronize", "finish"):
        if hasattr(work, m):
            getattr(work, m)()
            return
    # fallback: do a device sync
    if torch.xpu.is_available():
        torch.xpu.synchronize()


def main():
    rank = env_int("TORCHCOMM_RANK", "RANK", "PALS_RANKID", "PMIX_RANK", default=0)
    world = env_int(
        "TORCHCOMM_SIZE", "WORLD_SIZE", "PALS_WORLD_SIZE", "PMI_SIZE", default=1
    )
    local_rank = env_int(
        "TORCHCOMM_LOCAL_RANK",
        "LOCAL_RANK",
        "PALS_LOCAL_RANKID",
        "MPI_LOCALRANKID",
        default=rank,
    )
    local_size = env_int(
        "TORCHCOMM_LOCAL_SIZE", "LOCAL_WORLD_SIZE", "PALS_LOCAL_SIZE", default=1
    )

    if world < 2:
        raise RuntimeError(
            "c10d validation requires at least two ranks; launch with torchrun"
        )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    master_addr = os.environ.get("MASTER_ADDR", "<unset>")
    master_port = os.environ.get("MASTER_PORT", "<unset>")
    host = socket.gethostname()

    ndev = torch.xpu.device_count()
    if ndev <= 0:
        raise RuntimeError("No XPU devices visible (torch.xpu.device_count()==0)")

    dev_index = local_rank % ndev
    device = torch.device(f"xpu:{dev_index}")
    torch.xpu.set_device(device)

    print(
        f"[rank {rank}/{world} local {local_rank}/{local_size}] "
        f"host={host} MASTER={master_addr}:{master_port} "
        f"PMIX_RANK={os.environ.get('PMIX_RANK')} "
        f"PALS_LOCAL_RANKID={os.environ.get('PALS_LOCAL_RANKID')} "
        f"ndev={ndev} device={device}",
        flush=True,
    )

    initialized = False
    exit_code = 0
    try:
        print(f"[rank {rank}] entering dist.init_process_group()", flush=True)
        dist.init_process_group(
            backend="xccl",
            timeout=datetime.timedelta(seconds=120),
        )
        initialized = True
        print(f"[rank {rank}] dist.init_process_group() OK", flush=True)

        # Build a simple tensor: rank+1 replicated
        x = torch.ones(8, device=device, dtype=torch.float32) * float(rank + 1)

        work = dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True)
        work.wait()

        torch.xpu.synchronize()

        expected = float(sum(range(1, world + 1)))
        got = float(x[0].item())

        if rank == 0:
            print(
                f"[rank 0] allreduce OK expected={expected} got={got} x[:4]={x[:4].tolist()}",
                flush=True,
            )

        # Optional: sanity check on every rank (tight tolerance)
        if abs(got - expected) > 1e-3:
            raise RuntimeError(f"[rank {rank}] mismatch: expected {expected}, got {got}")

    except Exception as e:
        print(f"[rank {rank}] ERROR: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        exit_code = 1
    finally:
        if initialized:
            try:
                dist.destroy_process_group()
                print(f"[rank {rank}] destroy_process_group() OK", flush=True)
            except Exception as e:
                print(f"[rank {rank}] destroy_process_group() failed: {e}", flush=True)
                traceback.print_exc()
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
