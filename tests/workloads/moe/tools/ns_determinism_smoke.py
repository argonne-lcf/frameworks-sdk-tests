#!/usr/bin/env python
"""Final drift suspect (HANDOFF UPDATE 20): is the OPTIMIZER's device compute
bit-deterministic — same result on every rank, and on repeated calls?

    mpiexec -n 96 -ppn 12 ... python tools/ns_determinism_smoke.py

Context: allreduce_symmetry_smoke exonerated the collectives (bitwise clean,
intra- and cross-communicator, up to 460 MB), and the mirror exchange is a
lossless byte copy — so post-sync gradients are bit-identical across
replicas, yet replicas drift ~1e-2 per the 2D driver's checksum. By
elimination the divergence enters in the optimizer step. Every rank builds
IDENTICAL seeded inputs and fingerprints, in order:

  sum32      fp32 tensor .sum()                (the drift checksum itself)
  mm_fp32    plain fp32 matmul
  mm_bf16    plain bf16 matmul                 (DPAS path)
  ns_fp32    newton_schulz forced fp32
  ns_real    newton_schulz as Muon runs it     (bf16 on cuda/xpu)
  ns_batched newton_schulz on a 3D stack       (shape-batched path)
  muon_step  one MuonClip step on a tiny model (end-to-end)

Each test reports: distinct fingerprints across ranks (1 = cross-rank
deterministic) and whether an immediate same-rank repeat bit-matches
(YES = per-call deterministic). The first test with distinct>1 or
repeat=NO is the drift source.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from latentmoe.optim.muon import _NS_COEFFS, newton_schulz
from latentmoe.parallel.comm import init_distributed


def fingerprint(t: torch.Tensor) -> int:
    return int(t.detach().to(torch.float32).cpu().view(torch.int32)
               .long().sum().item())


def ns_fp32(G):
    X = G.to(torch.float32)
    t = X.shape[-2] > X.shape[-1]
    if t:
        X = X.transpose(-2, -1)
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    for a, b, c in _NS_COEFFS:
        A = X @ X.transpose(-2, -1)
        X = a * X + (b * A + c * (A @ A)) @ X
    if t:
        X = X.transpose(-2, -1)
    return X


def main():
    rank, world, device = init_distributed()
    dist.barrier()

    torch.manual_seed(0)
    G = (torch.randn(1024, 512) * 1e-3).to(device)      # grad-scale matrix
    A = torch.randn(1024, 1024).to(device)
    B = torch.randn(1024, 512).to(device)
    G3 = (torch.randn(6, 256, 512) * 1e-3).to(device)   # stacked (batched NS)
    S4 = torch.randn(4_000_000).to(device)              # ~param-sized sum
    S64 = torch.randn(64_000_000).to(device)            # far past any kernel
    #                                                   # strategy threshold
    # UPDATE 23 suspects: the MTP head's stage-7-only Muon matrices at
    # dim 1024 (fuse_proj 1024x3072 = the model's largest-K NS input;
    # proj 1024x2048) + a raw bf16 matmul at K=3072
    GF = (torch.randn(1024, 3072) * 1e-3).to(device)
    GW = (torch.randn(3072, 1024) * 1e-3).to(device)
    GP = (torch.randn(1024, 2048) * 1e-3).to(device)
    A3 = torch.randn(1024, 3072).to(device)
    B3 = torch.randn(3072, 1024).to(device)

    def muon_like(in_f, out_f):
        def run():
            torch.manual_seed(3)
            m = torch.nn.Linear(in_f, out_f, bias=False).to(device)
            torch.manual_seed(7)
            m.weight.grad = (torch.randn_like(m.weight) * 1e-3)
            from latentmoe.optim import MuonClip
            MuonClip.from_model(m, lr=1e-2).step()
            return m.weight
        return run

    muon_step = muon_like(1024, 512)

    tests = [
        ("sum32", lambda: G.sum().reshape(1)),
        ("sum32_4m", lambda: S4.sum().reshape(1)),
        ("sum32_64m", lambda: S64.sum().reshape(1)),
        ("mm_fp32", lambda: A @ B),
        ("mm_bf16", lambda: (A.to(torch.bfloat16) @ B.to(torch.bfloat16))),
        ("ns_fp32", lambda: ns_fp32(G)),
        ("ns_real", lambda: newton_schulz(G)),           # bf16 on cuda/xpu
        ("ns_batched", lambda: newton_schulz(G3)),
        ("ns_fuse3k", lambda: newton_schulz(GF)),        # 1024x3072 (fuse_proj)
        ("ns_proj2k", lambda: newton_schulz(GP)),        # 1024x2048 (mtp proj)
        ("mm_bf16_k3", lambda: (A3.to(torch.bfloat16)
                                @ B3.to(torch.bfloat16))),
        # UPDATE 25 attribution rows — the other ops that differ between
        # the clean K=2048 geometry and the dirty 3072 one inside NS:
        ("norm_bf16_3m", lambda: GF.to(torch.bfloat16).norm().reshape(1)),
        ("bx_bf16_n3", lambda: (A.to(torch.bfloat16)          # (1024,1024)
                                @ A3.to(torch.bfloat16))),    # @(1024,3072)
        ("bmm_bf16_k3", lambda: (A3.to(torch.bfloat16).unsqueeze(0)
                                 @ B3.to(torch.bfloat16).unsqueeze(0))),
        # UPDATE 26 rows — training feeds NS 3D batch-of-1 slabs, and the
        # field showed the two ORIENTATIONS of the same geometry behave
        # differently: contiguous (1,1024,3072) was the dirty one, the
        # internally-transposed (1,3072,1024) the clean one.
        ("ns_slab_fuse", lambda: newton_schulz(GF.unsqueeze(0))),
        ("ns_slab_wqkv", lambda: newton_schulz(GW.unsqueeze(0))),
        ("muon_fuse", muon_like(3072, 1024)),   # weight (1024,3072), FIXED
        #                                       # canonical-orientation path
        ("muon_wqkv", muon_like(1024, 3072)),   # weight (3072,1024)
        ("muon_step", muon_step),
    ]
    for name, fn in tests:
        f1 = fingerprint(fn())
        f2 = fingerprint(fn())                            # same-rank repeat
        mine = torch.tensor([f1], dtype=torch.int64, device=device)
        gathered = [torch.zeros_like(mine) for _ in range(world)]
        dist.all_gather(gathered, mine)
        if rank == 0:
            distinct = len({int(g) for g in gathered})
            print(f"[ns-smoke] {name:10s} | distinct across {world} ranks: "
                  f"{distinct:3d} | same-rank repeat match: "
                  f"{'YES' if f1 == f2 else 'NO '}", flush=True)
        dist.barrier()
    if rank == 0:
        print("ns determinism smoke complete", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
