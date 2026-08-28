#!/usr/bin/env python
"""Minimal reproducer: oneCCL ATL(PMIx) init fences JOB-WIDE at first
communicator creation, deadlocking lazy-init SUBGROUP collectives under
asymmetric schedules (found via latentmoe train_2d.py, Sunspot, 2026-08-21;
HANDOFF UPDATE 10).

The pattern (mirrors a pipeline whose middle ranks wait in CPU-staged recvs):

  group A (first half of ranks):  subgroup all_reduce FIRST   <- first XCCL
                                  then gloo-send a token to B    comm creation
  group B (second half):          gloo-recv the token FIRST   <- blocks before
                                  then its own subgroup all_reduce   any XCCL

On torch-XCCL under PALS/PMIx (CCL_PROCESS_LAUNCHER=pmix), group A's first
communicator creation enters a PMIx fence that requires EVERY rank in the
job; group B never reaches an XCCL call because it is blocked in the gloo
recv waiting for A's send, which A can only post after the fence. Mutual
deadlock => the run HANGS (that hang IS the reproduction; Ctrl-C it).

    mpiexec -n 4 -ppn 4 ... python tools/pmix_fence_smoke.py            # hangs
    mpiexec -n 4 -ppn 4 ... python tools/pmix_fence_smoke.py --warmup   # passes

--warmup runs the fix: a WORLD barrier + a per-subgroup 1-element all_reduce
executed by all ranks at the same program point, so ATL init and all
subgroup communicators bootstrap symmetrically before the asymmetric phase.
On NCCL/Gloo both modes pass (no job-wide fence) -- run there to verify the
script itself, then on XCCL to reproduce.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from latentmoe.parallel.comm import init_distributed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", action="store_true",
                   help="symmetric world barrier + subgroup warmup first (the fix)")
    args = p.parse_args()

    rank, world, device = init_distributed()
    assert world >= 4 and world % 2 == 0, "need an even world >= 4"
    half = world // 2
    groups = [dist.new_group(ranks=list(range(half))),
              dist.new_group(ranks=list(range(half, world)))]
    mine = groups[rank // half]
    gloo = dist.new_group(backend="gloo")   # CPU side-channel (staged-p2p analog)
    log = lambda msg: print(f"[fence-smoke rank {rank}] {msg}", flush=True)

    if args.warmup:
        dist.barrier()                                       # world comm, all ranks
        dist.all_reduce(torch.ones(1, device=device), group=mine)
        log("warmup done (world + subgroup comms initialized symmetrically)")

    token = torch.zeros(1)
    if rank < half:
        log("group A: subgroup all_reduce (first XCCL comm here without --warmup)")
        dist.all_reduce(torch.ones(1, device=device), group=mine)
        log("group A: all_reduce done, sending gloo token")
        dist.send(token, dst=rank + half, group=gloo)
    else:
        log("group B: blocking in gloo recv BEFORE any XCCL call")
        dist.recv(token, src=rank - half, group=gloo)
        log("group B: token received, now my subgroup all_reduce")
        dist.all_reduce(torch.ones(1, device=device), group=mine)

    dist.barrier()
    if rank == 0:
        print(f"fence-smoke PASSED (world {world}, warmup={args.warmup}) -- "
              "no job-wide fence deadlock on this backend/launcher", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
