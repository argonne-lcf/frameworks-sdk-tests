#!/usr/bin/env python
"""Point-to-point smoke test for the active torch.distributed backend.

Isolates the exact p2p patterns DualPipe uses, in escalating order, so a hang
identifies the failing pattern by the last line printed. Run with exactly 2
ranks:

    mpiexec -n 2 -ppn 2 --env WORLD_SIZE=2 python tools/p2p_smoke.py
    (or torchrun --nproc-per-node 2 tools/p2p_smoke.py)

Patterns:
  A  blocking one-way:        rank0 send -> rank1 recv
  B  paired exchange:         both ranks isend first, then blocking recv
  C  fire-and-forget:         isend, blocking recv, drain send LATER
  D  deferred match:          rank0 isends 4 messages BEFORE rank1 posts any
                              recv (DualPipe warmup shape)
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from latentmoe.parallel.comm import init_distributed


def log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    rank, world, device = init_distributed()
    assert world == 2, "run with exactly 2 ranks"
    peer = 1 - rank
    backend = dist.get_backend()
    log(rank, f"backend={backend} device={device}")
    t = torch.full((256, 256), float(rank + 1), device=device)

    # A: blocking one-way
    t0 = time.perf_counter()
    if rank == 0:
        dist.send(t, peer)
    else:
        buf = torch.empty_like(t)
        dist.recv(buf, peer)
        assert float(buf[0, 0]) == 1.0
    dist.barrier()
    log(rank, f"A blocking one-way: OK ({time.perf_counter()-t0:.2f}s)")

    # B: paired exchange, isend posted before blocking recv on BOTH sides
    t0 = time.perf_counter()
    w = dist.isend(t, peer)
    buf = torch.empty_like(t)
    dist.recv(buf, peer)
    w.wait()
    assert float(buf[0, 0]) == float(peer + 1)
    dist.barrier()
    log(rank, f"B paired exchange: OK ({time.perf_counter()-t0:.2f}s)")

    # C: fire-and-forget with LATE drain (DualPipe's normal mode)
    t0 = time.perf_counter()
    pend = [dist.isend(t.clone(), peer) for _ in range(3)]
    bufs = [torch.empty_like(t) for _ in range(3)]
    for b in bufs:
        dist.recv(b, peer)
    for w in pend:
        w.wait()
    dist.barrier()
    log(rank, f"C fire-and-forget x3: OK ({time.perf_counter()-t0:.2f}s)")

    # D: deferred match -- sender posts several isends before the receiver
    #    posts ANY recv (pipeline warmup). Receiver deliberately sleeps.
    t0 = time.perf_counter()
    if rank == 0:
        pend = [dist.isend(t.clone(), peer) for _ in range(4)]
        log(rank, "D posted 4 unmatched isends, waiting...")
        for w in pend:
            w.wait()
    else:
        time.sleep(3.0)
        bufs = [torch.empty_like(t) for _ in range(4)]
        for b in bufs:
            dist.recv(b, peer)
    dist.barrier()
    log(rank, f"D deferred match: OK ({time.perf_counter()-t0:.2f}s)")

    log(rank, "ALL PATTERNS PASSED")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
