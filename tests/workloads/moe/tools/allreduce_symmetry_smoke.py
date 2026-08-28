#!/usr/bin/env python
"""Confirm/kill HANDOFF UPDATE 17 factor (1): do the two mirror pipe rows'
all-reduce communicators produce BIT-IDENTICAL results on identical inputs,
and at what message size does that break?

    mpiexec -n 96 -ppn 12 ... python tools/allreduce_symmetry_smoke.py --pipe 8

The rank grid matches train_2d.py (p-major: row p = ranks [p*E, (p+1)*E)).
Every COLUMN e fills its buffer with the same deterministic values in every
row (seeded by e alone), so row p and row P-1-p reduce IDENTICAL input
multisets on DIFFERENT communicators — exactly the situation of the 2D
driver's mirror-replica gradient sync. After each all-reduce, mirror pairs
exchange (bit-fingerprint, fp64 sum, 1M-element head slice) over gloo and
report bitmatch + max|delta| per (size, column).

Interpretation: "bitmatch YES" at all sizes kills factor (1);
"NO" above some size confirms it and locates the threshold. Expected from
the field crash: YES at ~30 MB (dim-256 runs passed the old bitwise assert),
NO at ~460 MB (dim-1024 runs tripped it).
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", type=int, default=8, help="pipe depth P (even)")
    ap.add_argument("--sizes", type=int, nargs="*",
                    default=[1_000_000, 8_000_000, 32_000_000, 115_000_000],
                    help="buffer sizes in fp32 elements (default up to ~460MB)")
    args = ap.parse_args()

    rank, world, device = init_distributed()
    P = args.pipe
    assert P % 2 == 0 and world % P == 0, (P, world)
    E = world // P
    p, e = rank // E, rank % E
    rows = [dist.new_group(ranks=[q * E + c for c in range(E)]) for q in range(P)]
    gloo = dist.new_group(backend="gloo")
    mirror = (P - 1 - p) * E + e            # global rank of my mirror replica

    dist.barrier()                          # symmetric ATL init (UPDATE 10)
    dist.all_reduce(torch.ones(1, device=device), group=rows[p])

    for n in args.sizes:
        torch.manual_seed(1000 + e)         # identical per column across rows
        buf = torch.randn(n, device=device)
        dist.all_reduce(buf, group=rows[p])
        fp = buf.view(torch.int32).long().sum().item()   # bit fingerprint
        s64 = buf.double().sum().item()
        head = buf[: min(n, 1_000_000)].cpu()

        mine_i = torch.tensor([fp], dtype=torch.int64)
        mine_f = torch.tensor([s64], dtype=torch.float64)
        peer_i, peer_f = torch.empty_like(mine_i), torch.empty_like(mine_f)
        peer_head = torch.empty_like(head)
        first = p < P - 1 - p
        for send_t, recv_t in ((mine_i, peer_i), (mine_f, peer_f),
                               (head, peer_head)):
            if first:
                dist.send(send_t, mirror, group=gloo)
                dist.recv(recv_t, mirror, group=gloo)
            else:
                dist.recv(recv_t, mirror, group=gloo)
                dist.send(send_t, mirror, group=gloo)

        # INTRA-communicator check (UPDATE 19 hypothesis): do two members of
        # the SAME all-reduce receive bit-identical bytes? Compare with my
        # row neighbor (column e XOR 1). "NO" here = member-dependent
        # reduction order (recursive-doubling-like) inside ONE collective.
        bud_i, bud_head = mine_i.clone(), head.clone()
        if E > 1:
            buddy = p * E + (e ^ 1)
            for send_t, recv_t in ((mine_i, bud_i), (head, bud_head)):
                if e < (e ^ 1):
                    dist.send(send_t, buddy, group=gloo)
                    dist.recv(recv_t, buddy, group=gloo)
                else:
                    dist.recv(recv_t, buddy, group=gloo)
                    dist.send(send_t, buddy, group=gloo)

        if first:                            # one report per mirror pair
            bit = "YES" if int(peer_i) == fp else "NO "
            intra = "YES" if int(bud_i) == fp else "NO "
            dmax = float((head - peer_head).abs().max())
            dintra = float((head - bud_head).abs().max())
            print(f"[symmetry] {n * 4 / 1e6:7.1f} MB | col {e:2d} "
                  f"rows ({p},{P - 1 - p}) | cross-comm bitmatch {bit} "
                  f"max|d| {dmax:.3e} | INTRA-comm bitmatch {intra} "
                  f"max|d| {dintra:.3e}", flush=True)

    dist.barrier()
    if rank == 0:
        print("symmetry smoke complete", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
