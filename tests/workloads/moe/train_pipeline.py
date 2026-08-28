#!/usr/bin/env python
"""DualPipe demo: bidirectional zero-bubble pipeline over a stage-split model.

    torchrun --nproc-per-node 2 train_pipeline.py --steps 10
    torchrun --nproc-per-node 4 train_pipeline.py --steps 10 --layers 8

Works on NCCL (NVIDIA/GH200), XCCL (Intel), and Gloo (CPU) -- the schedule
only needs point-to-point sends/recvs.

Every rank holds TWO stages (direction-0 stage `rank`, direction-1 stage
`P-1-rank`); microbatches stream in from both ends at once. After each step,
mirror replicas of every stage exchange+sum grads (`sync_mirror_grads`) and
take identical MuonClip steps, so the two copies remain bit-identical -- we
assert exactly that every few steps.
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from latentmoe import get_preset
from latentmoe.accel import manual_seed_all
from latentmoe.data import MarkovData
from latentmoe.layers import RMSNorm
from latentmoe.model import Block
from latentmoe.optim import MuonClip
from latentmoe.parallel import DualPipe, init_distributed
from latentmoe.parallel.dualpipe import sync_mirror_grads


class Stage(nn.Module):
    """A pipeline stage: [embed] -> blocks -> [final norm + LM head]."""

    def __init__(self, cfg, block_ids, is_first, is_last):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim) if is_first else None
        self.blocks = nn.ModuleList(
            Block(cfg, cfg.attn_plan()[i], cfg.ffn_plan()[i]) for i in block_ids
        )
        self.final = RMSNorm(cfg.dim, cfg.norm_eps) if is_last else None
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False) if is_last else None

    def forward(self, x):
        h = self.embed(x) if self.embed is not None else x
        for b in self.blocks:
            h, _ = b(h, None, None, None, [])
        if self.final is not None:
            return self.head(self.final(h))
        return h


def criterion(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                           labels.reshape(-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--chunks-per-rank", type=int, default=2, help="num_chunks = 2*world*this")
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    rank, world, device = init_distributed()
    assert world % 2 == 0, "DualPipe needs an even world size"
    num_chunks = 2 * world * args.chunks_per_rank

    # pipeline demo uses the standard residual stream (mhc/attnres carry
    # cross-block state that a stage split would have to ship explicitly)
    cfg = get_preset("k3", vocab_size=2048, dim=args.dim, n_layers=args.layers,
                     n_heads=max(2, args.dim // 64), head_dim=64,
                     n_routed_experts=8, top_k=2, moe_latent_dim=args.dim // 2,
                     expert_hidden=args.dim // 2, shared_expert_hidden=args.dim,
                     dense_ffn_hidden=args.dim * 2, kv_latent_dim=args.dim // 2,
                     q_latent_dim=args.dim // 2, residual="standard",
                     use_mtp=False, max_seq_len=args.seq_len)
    assert cfg.n_layers % world == 0
    per = cfg.n_layers // world

    # every rank builds the identical full stage list (same seed), then keeps
    # its two: direction-0 stage `rank`, direction-1 stage `world-1-rank`
    manual_seed_all(0)
    stages = [
        Stage(cfg, range(s * per, (s + 1) * per), s == 0, s == world - 1)
        for s in range(world)
    ]
    mod0 = stages[rank].to(device)
    mod1 = copy.deepcopy(stages[world - 1 - rank]).to(device)
    dp = DualPipe((mod0, mod1) if rank < world // 2 else (mod1, mod0), device=device)
    # note: DualPipe indexes modules by schedule phase; _dir() maps phase ->
    # direction, so hand it (phase0_module, phase1_module) in schedule order.

    opt = MuonClip.from_model(dp.module, lr=args.lr)
    data = MarkovData(cfg.vocab_size, seed=1234)
    gen = torch.Generator().manual_seed(7)  # same stream on all ranks

    mb, T = args.microbatch, args.seq_len
    for step in range(1, args.steps + 1):
        x, y = data.batch(num_chunks * mb, T, device, generator=gen)
        half = x.shape[0] // 2
        x_feed = x[:half] if rank == 0 else (x[half:] if rank == world - 1 else None)
        labels = y[:half] if rank == world - 1 else (y[half:] if rank == 0 else None)

        opt.zero_grad(set_to_none=True)
        loss = dp.step(x_feed, num_chunks=num_chunks, criterion=criterion,
                       labels=labels, chunk_shape=(mb, T, cfg.dim),
                       chunk_dtype=torch.float32)
        sync_mirror_grads(dp)
        opt.step()

        if step % 2 == 0:  # mirror replicas must remain identical
            checks = torch.tensor([sum(float(p.detach().sum()) for p in dp._mod(d).parameters())
                                   for d in (0, 1)], device=device, dtype=torch.float64)
            gathered = [torch.zeros_like(checks) for _ in range(world)]
            dist.all_gather(gathered, checks)
            twin = world - 1 - rank
            assert abs(gathered[rank][0] - gathered[twin][1]) < 1e-6, \
                f"stage {rank}: mirror replicas diverged"
        if loss is not None:
            print(f"[rank {rank}] step {step:3d} "
                  f"mean chunk loss (my direction) {2 * float(loss):.4f}", flush=True)

    dist.barrier()
    if rank == 0:
        print("dualpipe run complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
