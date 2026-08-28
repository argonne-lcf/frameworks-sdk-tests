#!/usr/bin/env python
"""2D parallelism demo: DualPipe pipeline x DeepEP-style expert parallelism.

    torchrun --nproc-per-node 4 train_2d.py --pipe 2 --steps 10           # 2x2
    torchrun --nproc-per-node 8 train_2d.py --pipe 4 --layers 8 --steps 10  # 4x2
    mpiexec -n 24 -ppn 12 ... python train_2d.py --pipe 4 --experts 24    # 4x6
    # macOS: prepend GLOO_SOCKET_IFNAME=lo0 and pass
    #        --rdzv-backend static --master-addr 127.0.0.1

World = P x E ranks on a grid: pipeline position p = rank // E, expert column
e = rank % E (p-major, so each EP group is a contiguous rank span -> mostly
intra-node all-to-alls at 12 ranks/node). Per-column pipe groups run the
DualPipe bidirectional zero-bubble schedule; per-position EP groups run
DeepEP-style dispatch/combine inside every MoE layer of the stage. Columns
are data-parallel (each feeds its own batch stream); expert compute within a
pipeline stage is sharded across its EP group.

Gradient flow per step:
  1. dp.step(): a2a autograd moves expert grads to their owner EP rank;
     each column ends with column-batch grads (mirror halves split by direction)
  2. sync_mirror_grads(): the two directional replicas of every stage
     exchange+sum -> full column-batch gradient on both
  3. ep_grad_sync(group=ep_group): all-reduce SUM / E across columns
     -> the exact global-mean gradient on all 2E replicas of every stage
All 2E replicas then take identical MuonClip steps and must remain identical
-- asserted every few steps.

MTP (t+2 prediction, model.py convention) is available in both flavors:
  --mtp       head + loss on the last stage; the labels ARE the next tokens,
              so it costs ZERO extra communication (drafter has its own
              embedding table -- see Stage docstring)
  --mtp-fuse  K3/EAGLE-3 feature fusion: early/middle/last-block taps ship
              through the pipe as MULTI-TENSOR payloads (chunk_specs), pass
              untouched through intermediate stages, and their grads flow
              back across stage boundaries through the same transport

All three residual schemes run under the pipeline (--residual):
  standard  h is the single payload tensor
  mhc       the [B,T,n,d] hyper-connection stream state ships whole
  attnres   the bounded source list ships (h == sources[-1], so it rides
            free); trimming mirrors model.py exactly

Demo simplifications, stated honestly:
  * criterion is CE (+ weighted MTP CE); MoE aux losses (seq balance) not added
  * quantile balancer stats are per-rank AND per-direction unless
    --global-balancer (each replica still routes its own chunks; grads sync
    exactly either way)
"""

from __future__ import annotations

import argparse
import copy
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from latentmoe import get_preset
from latentmoe.accel import manual_seed_all
from latentmoe.data import MarkovData
from latentmoe.layers import HyperConnections, MTPHead, RMSNorm
from latentmoe.layers.moe import LatentMoE
from latentmoe.model import Block
from latentmoe.optim import MuonClip
from latentmoe.parallel import Buffer, DualPipe, ep_grad_sync, init_distributed
from latentmoe.parallel.dualpipe import sync_mirror_grads, sync_union_grads


class Stage(nn.Module):
    """A pipeline stage: [embed] -> blocks -> [final norm + LM head (+ MTP)].

    MTP (t+2 prediction, model.py convention) lives on the LAST stage, where
    the loss and labels already are — and since LM labels ARE the next tokens,
    the head stage can embed them locally: no new communication. The drafter
    gets its OWN small embedding table (the main one lives on the first
    stage); K3 trains MTP as a separate EAGLE-3 drafter anyway, and tying
    would cost an end-to-end grad all-reduce for no benefit at this scale.
    """

    def __init__(self, cfg, block_ids, is_first, is_last, mtp=None, tap_ids=()):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim) if is_first else None
        self.blocks = nn.ModuleList(
            Block(cfg, cfg.attn_plan()[i], cfg.ffn_plan()[i]) for i in block_ids
        )
        self.tap_flags = [gid in tap_ids for gid in block_ids]
        self.residual = cfg.residual
        self.streams = cfg.mhc_streams              # mhc: payload is the
        self.arb = cfg.attnres_block                # [B,T,n,d] stream state;
        self.n_src = 1 + min(block_ids.start, cfg.attnres_block)  # attnres:
        # payload is the bounded source list (h == sources[-1], ships free)
        self.final = RMSNorm(cfg.dim, cfg.norm_eps) if is_last else None
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False) if is_last else None
        self.mtp = self.mtp_embed = None
        if mtp and is_last:
            self.mtp = MTPHead(cfg.dim, cfg.n_heads, cfg.dense_ffn_hidden,
                               fuse_layers=(mtp == "fuse"), eps=cfg.norm_eps)
            self.mtp_embed = nn.Embedding(cfg.vocab_size, cfg.dim)

    def forward(self, x, *in_taps):
        """Fusion mode ships tap features as extra payload tensors: upstream
        taps arrive as *in_taps (ascending layer order), pass through
        untouched, and this stage appends the taps its own blocks produce.

        Two independent precision knobs (set as attributes by the driver):
        `autocast_device` runs the compute under bf16 autocast; `wire_dtype`
        compresses ONLY the inter-stage payload (upcast on entry, downcast on
        exit; loss-stage outputs never hit the wire and stay uncast)."""
        wire = getattr(self, "wire_dtype", None)
        if wire is not None and self.embed is None:
            x = x.float()
            in_taps = tuple(t.float() for t in in_taps)
        amp = getattr(self, "autocast_device", None)
        if amp is not None:
            with torch.autocast(amp, dtype=torch.bfloat16):
                out = self._compute(x, *in_taps)
        else:
            out = self._compute(x, *in_taps)
        if wire is not None and self.final is None:
            outs = out if isinstance(out, tuple) else (out,)
            outs = tuple(o.to(wire) for o in outs)
            return outs if len(outs) > 1 else outs[0]
        return out

    def _compute(self, x, *rest):
        """Payload conventions (must match payload_spec):
        standard: (h, *taps)   mhc: (state[B,T,n,d], *taps)
        attnres:  (*sources, *taps) with h == sources[-1]."""
        mode, state, sources = self.residual, None, None
        if mode == "mhc":
            state = (HyperConnections.expand(self.embed(x), self.streams)
                     if self.embed is not None else x)
            h, taps = None, list(rest)
        elif mode == "attnres":
            if self.embed is not None:
                sources = [self.embed(x)]
            else:
                sources = list((x,) + rest[: self.n_src - 1])
                rest = rest[self.n_src - 1:]
            h, taps = sources[-1], list(rest)
        else:
            h = self.embed(x) if self.embed is not None else x
            taps = list(rest)
        for b, is_tap in zip(self.blocks, self.tap_flags):
            if mode == "mhc":
                state, _ = b(state, None, None, None, [])
                if is_tap or b is self.blocks[-1]:
                    h = HyperConnections.collapse(state)
            elif mode == "attnres":
                h, _ = b(h, None, None, None, list(sources))
                sources.append(h)
                if len(sources) > 1 + self.arb:
                    sources = [sources[0]] + sources[-self.arb:]
            else:
                h, _ = b(h, None, None, None, [])
            if is_tap:
                taps.append(h)
        if self.final is None:
            if mode == "mhc":
                return (state, *taps) if taps else state
            if mode == "attnres":
                return (*sources, *taps)
            return (h, *taps) if taps else h
        logits = self.head(self.final(h))
        if self.mtp is None:
            return logits
        if self.mtp.fuse:
            return (logits, h, *taps)  # criterion slices feats for MTP
        return logits, h               # criterion needs pre-norm h for MTP


def payload_spec(s, cfg, per, mb, T, tap_ids, dtype):
    """Payload arriving at stage s; must match Stage._compute's unpacking."""
    if cfg.residual == "mhc":
        specs = [((mb, T, cfg.mhc_streams, cfg.dim), dtype)]
    elif cfg.residual == "attnres":
        specs = [((mb, T, cfg.dim), dtype)] * (1 + min(s * per, cfg.attnres_block))
    else:
        specs = [((mb, T, cfg.dim), dtype)]
    return specs + [((mb, T, cfg.dim), dtype)] * sum(t < s * per for t in tap_ids)


def make_criterion(head_stage, mtp_weight):
    """Plain CE, or CE + weighted MTP CE mirroring model.py's t+2 convention
    (`head_stage` is this rank's replica of the last stage, or None).
    Returns (criterion, stats): stats accumulates per-chunk CE / MTP
    components so the driver can report them separately per step."""
    stats = {"ce": 0.0, "mtp": 0.0, "n": 0}

    def criterion(out, labels):
        if head_stage is None or head_stage.mtp is None:
            return F.cross_entropy(out.reshape(-1, out.shape[-1]).float(),
                                   labels.reshape(-1))
        logits, h, *feats = out
        h = h.float()  # taps/h may arrive bf16 off the wire; MTP head is fp32
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             labels.reshape(-1))
        loss = ce
        if labels.shape[1] > 2:
            next_tok = labels[:, :-1].clamp_min(0)        # token t+1 == label t
            mtp_h = head_stage.mtp(h[:, :-1], head_stage.mtp_embed(next_tok),
                                   feats=[f[:, :-1].float() for f in feats] or None)
            mtp_logits = head_stage.head(mtp_h)           # shared LM head
            mtp = F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1))                # predict token t+2
            loss = loss + mtp_weight * mtp
            stats["mtp"] += float(mtp.detach())
        stats["ce"] += float(ce.detach())
        stats["n"] += 1
        return loss

    return criterion, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipe", type=int, default=2,
                   help="pipeline depth P (even, >=2); EP size = world // P")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--chunks-per-rank", type=int, default=2,
                   help="num_chunks per column = 2*pipe*this")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mtp", action="store_true",
                   help="MTP t+2 head + loss on the head stage (no fusion)")
    p.add_argument("--mtp-fuse", action="store_true",
                   help="K3/EAGLE-3 fusion MTP: low/mid/high layer taps ship "
                        "through the pipe as multi-tensor payloads")
    p.add_argument("--no-bucket-grads", action="store_true",
                   help="EP grad sync: per-parameter all-reduce loop instead "
                        "of one flat bucket per dtype (A/B baseline)")
    p.add_argument("--global-balancer", action="store_true",
                   help="aggregate balancer statistics across the EP row "
                        "(paper behavior) instead of per-rank")
    p.add_argument("--no-batch-ns", action="store_true",
                   help="Muon: per-parameter Newton-Schulz instead of one "
                        "batched NS per distinct shape (A/B baseline)")
    p.add_argument("--bf16", action="store_true",
                   help="bf16 autocast for stage compute (CUDA/XPU only)")
    p.add_argument("--bf16-chunks", action="store_true",
                   help="ship inter-stage payloads in bf16 (halves staged-"
                        "Gloo bytes on Intel); independent of --bf16")
    p.add_argument("--residual", choices=["standard", "mhc", "attnres"],
                   default="standard",
                   help="residual scheme; mhc ships the [B,T,n,d] stream "
                        "state, attnres the bounded source list, as "
                        "multi-tensor payloads")
    p.add_argument("--legacy-sync", action="store_true",
                   help="use the original mirror-pair + per-row grad sync "
                        "instead of the union-group all-reduce (A/B; legacy "
                        "has cross-communicator ulp asymmetry, UPDATE 17)")
    p.add_argument("--debug-grad-drift", action="store_true",
                   help="checksum POST-SYNC grads across all stage replicas "
                        "every step (in-situ discriminator, UPDATE 21: "
                        "0.0 => sync clean under load, optimizer-side; "
                        ">0 => collectives diverge in-situ)")
    args = p.parse_args()

    rank, world, device = init_distributed()
    P = args.pipe
    assert P >= 2 and P % 2 == 0, "DualPipe needs an even pipeline depth"
    assert world % P == 0, f"world={world} not divisible by --pipe {P}"
    E = world // P
    pipe_rank, ep_rank = rank // E, rank % E
    num_chunks = 2 * P * args.chunks_per_rank

    # every rank creates ALL subgroups in the same order (torch requirement),
    # then keeps its own row (EP) and column (pipe)
    rows = [dist.new_group(ranks=[q * E + c for c in range(E)]) for q in range(P)]
    cols = [dist.new_group(ranks=[q * E + c for q in range(P)]) for c in range(E)]
    # union of the two mirror rows (2E ranks): every replica of a stage gets
    # its gradient from ONE collective => bit-identical by construction
    unions = [dist.new_group(ranks=[q * E + c for c in range(E)]
                             + [(P - 1 - q) * E + c for c in range(E)])
              for q in range(P // 2)]
    ep_group, pipe_group = rows[pipe_rank], cols[ep_rank]
    union_group = unions[min(pipe_rank, P - 1 - pipe_rank)]

    cfg = get_preset("k3", vocab_size=2048, dim=args.dim, n_layers=args.layers,
                     n_heads=max(2, args.dim // 64), head_dim=64,
                     n_routed_experts=args.experts, top_k=args.top_k,
                     moe_latent_dim=args.dim // 2, expert_hidden=args.dim // 2,
                     shared_expert_hidden=args.dim, dense_ffn_hidden=args.dim * 2,
                     kv_latent_dim=args.dim // 2, q_latent_dim=args.dim // 2,
                     residual=args.residual, use_mtp=False,
                     max_seq_len=args.seq_len)
    assert cfg.n_layers % P == 0, "layers must divide by pipeline depth"
    per = cfg.n_layers // P

    # identical full stage list everywhere (same seed); keep my two
    mtp_mode = "fuse" if args.mtp_fuse else ("plain" if args.mtp else None)
    # early/middle/last-block feature taps, mirroring model.py's selection
    tap_ids = ((max(0, cfg.n_layers // 4 - 1), cfg.n_layers // 2, cfg.n_layers - 1)
               if mtp_mode == "fuse" else ())
    assert len(set(tap_ids)) in (0, 3), "--mtp-fuse needs --layers >= 4"
    manual_seed_all(0)
    stages = [
        Stage(cfg, range(s * per, (s + 1) * per), s == 0, s == P - 1,
              mtp=mtp_mode, tap_ids=tap_ids)
        for s in range(P)
    ]
    mod0 = stages[pipe_rank].to(device)
    mod1 = copy.deepcopy(stages[P - 1 - pipe_rank]).to(device)

    buffer = Buffer(cfg.n_routed_experts, group=ep_group) if E > 1 else None
    if buffer is not None:
        for mod in (mod0, mod1):  # attach AFTER deepcopy: Buffer must be shared
            for m in mod.modules():
                if isinstance(m, LatentMoE):
                    m.dispatcher = buffer
                    if args.global_balancer:
                        m.stats_group = ep_group

    amp_dev = device.type if (args.bf16 and device.type in ("cuda", "xpu")) else None
    if args.bf16 and amp_dev is None and rank == 0:
        print(f"[2d] --bf16 ignored on device '{device.type}' "
              "(autocast is CUDA/XPU only)", flush=True)
    wire = torch.bfloat16 if args.bf16_chunks else None
    for mod in (mod0, mod1):
        mod.autocast_device = amp_dev
        mod.wire_dtype = wire
    chunk_dtype = wire or torch.float32

    dp = DualPipe((mod0, mod1) if pipe_rank < P // 2 else (mod1, mod0),
                  group=pipe_group, device=device)
    opt = MuonClip.from_model(dp.module, lr=args.lr,
                              ns_batched=not args.no_batch_ns)

    # the ranks that compute losses hold the head stage (stage P-1) as one of
    # their two replicas: dir0 on pipe_rank P-1, dir1 on pipe_rank 0
    head_stage = mod0 if pipe_rank == P - 1 else (mod1 if pipe_rank == 0 else None)
    crit, loss_stats = make_criterion(head_stage, cfg.mtp_loss_weight)

    # throughput accounting over the FULL model (stages[] holds every stage
    # once; mod0 aliases stages[pipe_rank]); ~train FLOPs = 6 * N_active / tok
    n_full = sum(p_.numel() for s in stages for p_ in s.parameters())
    moes = [m for s in stages for m in s.modules() if isinstance(m, LatentMoE)]
    n_active = n_full - sum(m.w1.numel() + m.w2.numel() for m in moes) \
        + sum((m.w1.numel() + m.w2.numel()) * m.k // m.e for m in moes)
    tokens_global = E * num_chunks * args.microbatch * args.seq_len

    if rank == 0:
        print(f"[2d] grid pipe={P} x ep={E} (world {world}) | "
              f"experts {cfg.n_routed_experts} ({cfg.n_routed_experts // E}/rank) | "
              f"chunks/column {num_chunks} | mtp={mtp_mode or 'off'} | "
              f"amp={amp_dev or 'off'} wire={'bf16' if wire else 'fp32'} | "
              f"sync={'legacy' if args.legacy_sync else 'union'} | "
              f"model {n_full/1e6:.1f}M (~{n_active/1e6:.1f}M active)", flush=True)

    # oneCCL/XCCL bootstrap: the process-wide ATL(PMIx) init FENCES ACROSS THE
    # ENTIRE JOB at first communicator creation. Middle pipe rows block in
    # staged recvs before ever touching XCCL, so if the first XCCL collectives
    # are the pipe-END rows' EP a2a's, that fence deadlocks (observed on
    # Sunspot, 4x6 grid, 2026-08-21). Warm up the world comm -- and each EP
    # row, symmetrically -- while every rank is still at the same point.
    dist.barrier()
    if E > 1:
        warm = torch.ones(1, device=device)
        dist.all_reduce(warm, group=ep_group)
    dist.all_reduce(torch.ones(1, device=device), group=union_group)

    # per-COLUMN data stream (columns are data-parallel); all ranks of a
    # column draw the same batch and slice their direction's half
    data = MarkovData(cfg.vocab_size, seed=1234)
    gen = torch.Generator().manual_seed(1000 + ep_rank)

    mb, T = args.microbatch, args.seq_len
    chunk_specs = None
    if mtp_mode == "fuse" or cfg.residual != "standard":
        chunk_specs = {
            0: payload_spec(pipe_rank, cfg, per, mb, T, tap_ids, chunk_dtype),
            1: payload_spec(P - 1 - pipe_rank, cfg, per, mb, T, tap_ids, chunk_dtype),
        }
    t_prev = time.perf_counter()
    for step in range(1, args.steps + 1):
        loss_stats.update(ce=0.0, mtp=0.0, n=0)
        x, y = data.batch(num_chunks * mb, T, device, generator=gen)
        half = x.shape[0] // 2
        x_feed = x[:half] if pipe_rank == 0 else (x[half:] if pipe_rank == P - 1 else None)
        labels = y[:half] if pipe_rank == P - 1 else (y[half:] if pipe_rank == 0 else None)

        opt.zero_grad(set_to_none=True)
        loss = dp.step(x_feed, num_chunks=num_chunks, criterion=crit,
                       labels=labels, chunk_shape=(mb, T, cfg.dim),
                       chunk_dtype=chunk_dtype, chunk_specs=chunk_specs)
        if args.legacy_sync:
            sync_mirror_grads(dp)                # replicas within my column
            if E > 1:
                ep_grad_sync(dp.module, group=ep_group,  # SUM/E across cols
                             bucketed=not args.no_bucket_grads)
        else:
            # one all-reduce per stage over BOTH mirror rows (UPDATE 18)
            sync_union_grads(dp, union_group, E,
                             bucketed=not args.no_bucket_grads)
        if args.debug_grad_drift:
            # In-situ discriminator (UPDATE 21): are POST-SYNC grads BITWISE
            # identical across all 2E replicas of each stage, inside a real
            # step? distinct==1 everywhere => sync is clean under load and
            # the divergence enters at/after opt.step(); distinct>1 => the
            # collectives differ in-situ (V3 only exonerated them isolated).
            fps, sums = [], []
            for d in (0, 1):
                acc = torch.zeros((), dtype=torch.int64, device=device)
                s64 = torch.zeros((), dtype=torch.float64, device=device)
                for p_ in dp._mod(d).parameters():
                    if p_.grad is not None:
                        g_ = p_.grad.detach().contiguous()
                        acc += g_.view(torch.int32).long().sum()
                        s64 += g_.double().sum()
                fps.append(acc)
                sums.append(s64)
            gi, gf = torch.stack(fps), torch.stack(sums)
            all_i = [torch.zeros_like(gi) for _ in range(world)]
            all_f = [torch.zeros_like(gf) for _ in range(world)]
            dist.all_gather(all_i, gi)
            dist.all_gather(all_f, gf)
            if rank == 0:  # every rank holds the full gather; report once
                bad = []
                for s in range(P):
                    # stage-s replicas: _mod(0) of row s, _mod(1) of row P-1-s
                    who = ([(g, 0) for g in range(world) if g // E == s]
                           + [(g, 1) for g in range(world)
                              if g // E == P - 1 - s])
                    distinct = len({int(all_i[g][d]) for g, d in who})
                    if distinct > 1:
                        fv = [float(all_f[g][d]) for g, d in who]
                        bad.append(f"stage {s}: {distinct} distinct "
                                   f"(fp64 spread {max(fv) - min(fv):.3e})")
                print(f"[2d] post-sync grads step {step}: "
                      + ("BIT-IDENTICAL across all replicas, all "
                         f"{P} stages" if not bad else "; ".join(bad)),
                      flush=True)
        opt.step()

        if step % 2 == 0:  # ALL 2E replicas of every stage must stay in lockstep
            checks = torch.tensor(
                [sum(float(p_.detach().sum()) for p_ in dp._mod(d).parameters())
                 for d in (0, 1)], device=device, dtype=torch.float64)
            gathered = [torch.zeros_like(checks) for _ in range(world)]
            dist.all_gather(gathered, checks)
            for s in {pipe_rank, P - 1 - pipe_rank}:
                vals = [gathered[g][0] for g in range(world) if g // E == s]
                vals += [gathered[g][1] for g in range(world) if g // E == P - 1 - s]
                spread = float(max(vals) - min(vals))
                scale = max(1.0, max(abs(float(v)) for v in vals))
                # Bitwise identity holds on gloo and at small sizes; at large
                # gradient buckets the two mirror rows' XCCL all-reduces can
                # differ by ulps, which bf16 Newton-Schulz amplifies to ~1e-2
                # absolute (HANDOFF UPDATE 17). Assert bounded RELATIVE drift:
                # real bugs (optimizer-group mismatch etc.) blow past this and
                # GROW step over step -- watch the printed spread.
                assert spread < 1e-3 * scale, \
                    (f"stage {s}: {2 * E} replicas diverged "
                     f"(spread {spread:.3e}, scale {scale:.3e})")
        if step % 2 == 0 and args.debug_grad_drift:
            # H-artifact discriminator (UPDATE 22): integer bit-fingerprints
            # of the PARAMS are exact and order-invariant; the fp64 checksum
            # above rides per-param fp32 .sum() device reductions, which are
            # order-SENSITIVE. If params fingerprint BIT-IDENTICAL while the
            # checksum shows spread, the "drift" is the instrument's own
            # reduction-order jitter, not real divergence.
            # PER-PARAM fingerprints (padded to a common length so the
            # gather is fixed-size; pad rows stay 0 everywhere and never
            # flag) — names the exact diverging tensor(s), UPDATE 23
            mxp = torch.tensor(
                [max(sum(1 for _ in dp._mod(d).parameters())
                     for d in (0, 1))], device=device)
            dist.all_reduce(mxp, op=dist.ReduceOp.MAX)
            mxp = int(mxp)
            ppf = torch.zeros((2, mxp), dtype=torch.int64, device=device)
            for d in (0, 1):
                for i, p_ in enumerate(dp._mod(d).parameters()):
                    ppf[d, i] = (p_.detach().contiguous().view(torch.int32)
                                 .long().sum())
            all_ppf = [torch.zeros_like(ppf) for _ in range(world)]
            dist.all_gather(all_ppf, ppf)
            if rank == 0:
                bad = []
                for s in range(P):
                    who = ([(g, 0) for g in range(world) if g // E == s]
                           + [(g, 1) for g in range(world)
                              if g // E == P - 1 - s])
                    fps = {tuple(all_ppf[g][d].tolist()) for g, d in who}
                    if len(fps) == 1:
                        continue
                    bad.append(f"stage {s}: {len(fps)} distinct")
                    div = [i for i in range(mxp)
                           if len({int(all_ppf[g][d][i]) for g, d in who}) > 1]
                    names = None  # rank 0 can name params of its own stages
                    if s == pipe_rank:
                        names = [n for n, _ in dp._mod(0).named_parameters()]
                    elif s == P - 1 - pipe_rank:
                        names = [n for n, _ in dp._mod(1).named_parameters()]
                    print(f"[2d] diverging tensors step {step} stage {s}: "
                          + ", ".join(names[i] if names else f"param[{i}]"
                                      for i in div), flush=True)
                print(f"[2d] params step {step}: "
                      + ("BIT-IDENTICAL across all replicas, all "
                         f"{P} stages" if not bad else "; ".join(bad)),
                      flush=True)
            # same-rank repeat of the fp32 checksum on UNCHANGED params: any
            # mismatch is direct local proof the measurement is noisy
            checks2 = torch.tensor(
                [sum(float(p_.detach().sum())
                     for p_ in dp._mod(d).parameters())
                 for d in (0, 1)], device=device, dtype=torch.float64)
            if not torch.equal(checks, checks2):
                print(f"[2d] rank {rank}: fp32 checksum NON-REPEATABLE on "
                      f"unchanged params (|d| "
                      f"{float((checks - checks2).abs().max()):.3e})",
                      flush=True)
        if step % 2 == 0 and rank == 0:
            print(f"[2d] replica drift ok at step {step} "
                  f"(spread {spread:.3e} / scale {scale:.3e})", flush=True)
        now = time.perf_counter()
        dt, t_prev = now - t_prev, now  # instantaneous, not cumulative avg
        if loss is not None:
            parts = ""
            if loss_stats["n"]:
                parts = (f" | ce {loss_stats['ce'] / loss_stats['n']:.4f}"
                         f" | mtp {loss_stats['mtp'] / loss_stats['n']:.4f}")
            tok_s = tokens_global / dt
            fl = 6 * n_active * tok_s / world
            fls = (f"{fl / 1e12:.2f} TF" if fl >= 1e11 else f"{fl / 1e9:.1f} GF")
            print(f"[pipe {pipe_rank} | ep {ep_rank}] step {step:3d} "
                  f"mean chunk loss (my direction) {2 * float(loss):.4f}{parts}"
                  f" | {dt:.3f}s/step | {tok_s / 1e3:.1f}k tok/s"
                  f" ({tok_s / world / 1e3:.2f}k/gpu, ~{fls}/gpu)", flush=True)

    dist.barrier()
    if rank == 0:
        print("2d (dualpipe x deepep) run complete")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
