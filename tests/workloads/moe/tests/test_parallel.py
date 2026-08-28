"""Distributed tests on Gloo/CPU (2 processes): DeepEP-style EP and DualPipe.

The gold standards:
  * EP MoE forward == single-process MoE forward on the same tokens, and
    post-sync gradients == single-process global-batch gradients.
  * DualPipe gradients == plain full-model backward gradients.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from latentmoe.layers.moe import LatentMoE

DIM, EXPERTS, TOPK = 32, 8, 2
SEQ, MB = 16, 2


def _init(rank, world, store_file):
    dist.init_process_group("gloo", init_method=f"file://{store_file}",
                            rank=rank, world_size=world)


def _make_moe(seed=0):
    torch.manual_seed(seed)
    return LatentMoE(DIM, EXPERTS, TOPK, expert_hidden=16, n_shared=1,
                     shared_hidden=16, latent_dim=16, balancer="none")


# --------------------------------------------------------------------- #
def _ep_worker(rank, world, store_file, bucketed=True):
    _init(rank, world, store_file)
    from latentmoe.parallel import Buffer, ep_grad_sync

    moe = _make_moe()
    torch.manual_seed(100 + rank)
    x = torch.randn(1, 12, DIM)

    # single-process reference on the *global* batch
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x)
    ref_moe = _make_moe()
    ref_out, _ = ref_moe(torch.cat(xs, dim=0))
    ref_loss = ref_out.pow(2).mean()
    ref_loss.backward()

    # expert-parallel run
    moe.dispatcher = Buffer(EXPERTS)
    out, _ = moe(x)
    # forward must match the reference rows for this rank
    assert torch.allclose(out, ref_out[rank:rank + 1], atol=1e-5), \
        (out - ref_out[rank:rank + 1]).abs().max()

    # loss defined like the reference: mean over the global batch =
    # mean over ranks of per-rank means (equal sizes)
    loss = out.pow(2).mean()
    loss.backward()
    ep_grad_sync(moe, bucketed=bucketed)
    for (n, p), (_, pr) in zip(moe.named_parameters(), ref_moe.named_parameters()):
        assert torch.allclose(p.grad, pr.grad, atol=1e-5), \
            (n, (p.grad - pr.grad).abs().max())
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2])
@pytest.mark.parametrize("bucketed", [True, False])
def test_expert_parallel_matches_local(tmp_path, world, bucketed):
    mp.spawn(_ep_worker,
             args=(world, str(tmp_path / f"ep_store_{int(bucketed)}"), bucketed),
             nprocs=world, join=True)


# --------------------------------------------------------------------- #
def _balancer_worker(rank, world, store_file):
    """Global balancer statistics (moe.stats_group) across an EP group.

    bias: the all-reduced load must reproduce the single-process bias update
    on the CONCATENATED batch exactly. quantile: all group members must end
    with IDENTICAL expert_bias -- and per-rank stats (flag off) must differ.
    """
    _init(rank, world, store_file)

    def make(balancer):
        torch.manual_seed(0)
        return LatentMoE(DIM, EXPERTS, TOPK, expert_hidden=16, n_shared=1,
                         shared_hidden=16, latent_dim=16, balancer=balancer)

    torch.manual_seed(200 + rank)
    x = torch.randn(2, 16, DIM)
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x)

    # ---- bias balancer: exact global-load reproduction ----
    moe = make("bias")
    moe.stats_group = dist.group.WORLD
    moe(x)
    moe.balance_step()
    ref = make("bias")
    ref(torch.cat(xs, dim=0))
    ref.balance_step()
    assert torch.equal(moe.expert_bias, ref.expert_bias), \
        (moe.expert_bias - ref.expert_bias).abs().max()

    # ---- quantile balancer: identical state across the group ----
    moe_q = make("quantile")
    moe_q.stats_group = dist.group.WORLD
    moe_q(x)
    gathered = [torch.zeros_like(moe_q.expert_bias) for _ in range(world)]
    dist.all_gather(gathered, moe_q.expert_bias)
    for g in gathered:
        assert torch.equal(moe_q.expert_bias, g), "global quantile state diverged"

    # flag OFF: per-rank stats must actually differ across ranks
    moe_l = make("quantile")
    moe_l(x)
    gathered_l = [torch.zeros_like(moe_l.expert_bias) for _ in range(world)]
    dist.all_gather(gathered_l, moe_l.expert_bias)
    assert not all(torch.equal(gathered_l[0], g) for g in gathered_l[1:]), \
        "per-rank quantile stats unexpectedly identical"
    dist.barrier()
    dist.destroy_process_group()


def test_global_balancer_stats(tmp_path):
    world = 2
    mp.spawn(_balancer_worker, args=(world, str(tmp_path / "bal_store")),
             nprocs=world, join=True)


# --------------------------------------------------------------------- #
class _TinyStage(nn.Module):
    def __init__(self, dim, is_first, is_last, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim) if is_first else None
        self.lin = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.head = nn.Linear(dim, vocab) if is_last else None

    def forward(self, x):
        h = self.embed(x) if self.embed is not None else x
        h = h + self.lin(h)
        return self.head(h) if self.head is not None else h


def _crit(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def _dualpipe_worker(rank, world, store_file):
    _init(rank, world, store_file)
    from latentmoe.parallel import DualPipe
    from latentmoe.parallel.dualpipe import sync_mirror_grads

    vocab, dim = 64, DIM
    num_chunks = 2 * world * 2
    torch.manual_seed(0)
    stages = [_TinyStage(dim, s == 0, s == world - 1, vocab) for s in range(world)]

    torch.manual_seed(42)
    x = torch.randint(0, vocab, (num_chunks * MB, SEQ))
    y = torch.randint(0, vocab, (num_chunks * MB, SEQ))

    # reference: plain forward/backward through all stages, global mean loss
    import copy
    ref_stages = copy.deepcopy(stages)
    h = x
    for s in ref_stages:
        h = s(h)
    ref_loss = _crit(h, y)
    ref_loss.backward()

    mod0 = stages[rank]
    mod1 = copy.deepcopy(stages[world - 1 - rank])
    dp = DualPipe((mod0, mod1) if rank < world // 2 else (mod1, mod0))

    half = x.shape[0] // 2
    x_feed = x[:half] if rank == 0 else (x[half:] if rank == world - 1 else None)
    labels = y[:half] if rank == world - 1 else (y[half:] if rank == 0 else None)
    dp.step(x_feed, num_chunks=num_chunks, criterion=_crit, labels=labels,
            chunk_shape=(MB, SEQ, dim), chunk_dtype=torch.float32)
    sync_mirror_grads(dp)

    # my direction-0 module is stage `rank`: grads must equal the reference
    ref = ref_stages[rank]
    for (n, p), (_, pr) in zip(dp._mod(0).named_parameters(), ref.named_parameters()):
        assert p.grad is not None, n
        assert torch.allclose(p.grad, pr.grad, atol=1e-4), \
            (rank, n, (p.grad - pr.grad).abs().max())

    # replica determinism: identical grads + identical optimizer settings must
    # keep the two copies of every stage bit-identical after a step (guards
    # against optimizer-group mismatches, e.g. deepcopy dropping param tags)
    from latentmoe.optim import MuonClip
    opt = MuonClip.from_model(dp.module, lr=1e-3)
    opt.step()
    mine = torch.cat([p.detach().flatten() for p in dp._mod(0).parameters()])
    mine1 = torch.cat([p.detach().flatten() for p in dp._mod(1).parameters()])
    peer = world - 1 - rank
    twin = torch.empty_like(mine)
    if rank < peer:
        dist.send(mine1, peer)
        dist.recv(twin, peer)
    else:
        dist.recv(twin, peer)
        dist.send(mine1, peer)
    assert torch.equal(mine, twin), \
        (rank, "replica params diverged after optimizer step",
         (mine - twin).abs().max())
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_dualpipe_grads_match_reference(tmp_path, world):
    if world > (os.cpu_count() or 2):
        pytest.skip("not enough cpus")
    mp.spawn(_dualpipe_worker, args=(world, str(tmp_path / f"dp_store_{world}")),
             nprocs=world, join=True)


# --------------------------------------------------------------------- #
def _ep_subgroup_worker(rank, world, store_file):
    """EP inside a SUBGROUP (world=4 -> two independent EP groups of 2).

    Exercises the group-relative -> global rank translation in the Gloo
    all-to-all fallback: each subgroup must reproduce its own subgroup-local
    single-process reference, oblivious to the other subgroup.
    """
    _init(rank, world, store_file)
    from latentmoe.parallel import Buffer, ep_grad_sync

    sub = world // 2
    groups = [dist.new_group(ranks=list(range(sub))),
              dist.new_group(ranks=list(range(sub, world)))]
    my_group = groups[rank // sub]

    moe = _make_moe()
    torch.manual_seed(100 + rank)
    x = torch.randn(1, 12, DIM)

    # reference on MY SUBGROUP's global batch only
    xs = [torch.zeros_like(x) for _ in range(sub)]
    dist.all_gather(xs, x, group=my_group)
    ref_moe = _make_moe()
    ref_out, _ = ref_moe(torch.cat(xs, dim=0))
    ref_out.pow(2).mean().backward()

    moe.dispatcher = Buffer(EXPERTS, group=my_group)
    out, _ = moe(x)
    sub_rank = dist.get_rank(my_group)
    assert torch.allclose(out, ref_out[sub_rank:sub_rank + 1], atol=1e-5), \
        (out - ref_out[sub_rank:sub_rank + 1]).abs().max()

    out.pow(2).mean().backward()
    ep_grad_sync(moe, group=my_group)
    for (n, p), (_, pr) in zip(moe.named_parameters(), ref_moe.named_parameters()):
        assert torch.allclose(p.grad, pr.grad, atol=1e-5), \
            (n, (p.grad - pr.grad).abs().max())
    dist.barrier()
    dist.destroy_process_group()


def test_expert_parallel_subgroup_matches_local(tmp_path):
    world = 4
    if world > (os.cpu_count() or 2):
        pytest.skip("not enough cpus")
    mp.spawn(_ep_subgroup_worker, args=(world, str(tmp_path / "ep_sub_store")),
             nprocs=world, join=True)


# --------------------------------------------------------------------- #
class _MoeStage(nn.Module):
    """Pipeline stage with a real LatentMoE inside (for the 2D test).

    No attention: the stage is token-wise, so chunked pipeline execution is
    mathematically identical to one full-batch forward (needed for an exact
    single-process reference). With mtp=True the last stage carries an MTPHead
    + its own drafter embedding (as in train_2d.py --mtp) and returns
    (logits, pre-head hidden) for the combined criterion.
    """

    def __init__(self, dim, is_first, is_last, vocab, mtp=False):
        super().__init__()
        from latentmoe.layers import MTPHead
        self.embed = nn.Embedding(vocab, dim) if is_first else None
        self.moe = _make_moe_uninit(dim)
        self.head = nn.Linear(dim, vocab) if is_last else None
        self.fuse = mtp == "fuse"
        self.mtp = self.mtp_embed = None
        if mtp and is_last:
            self.mtp = MTPHead(dim, 2, 16, fuse_layers=self.fuse)
            self.mtp_embed = nn.Embedding(vocab, dim)

    def forward(self, x, tap=None):
        wire = getattr(self, "wire_dtype", None)  # bf16-on-the-wire contract
        if wire is not None and self.embed is None:
            x = x.float()
            tap = tap.float() if tap is not None else None
        h = self.embed(x) if self.embed is not None else x
        y, _ = self.moe(h)
        h = h + y
        if self.head is None:
            out = (h, h) if self.fuse else h     # fuse: emit my tap downstream
            if wire is not None:
                out = (tuple(o.to(wire) for o in out)
                       if isinstance(out, tuple) else out.to(wire))
            return out
        logits = self.head(h)
        if self.mtp is None:
            return logits
        return (logits, h, tap) if self.fuse else (logits, h)


def _mtp_crit(head_stage, w=0.3):
    """CE + weighted t+2 MTP CE, same convention as train_2d.make_criterion."""

    def crit(out, labels):
        if head_stage is None or head_stage.mtp is None:
            return _crit(out, labels)
        if head_stage.fuse:
            logits, h, tap = out
            h, tap = h.float(), tap.float()      # tap may arrive bf16 off the wire
            feats = [tap[:, :-1], h[:, :-1], h[:, :-1]]
        else:
            logits, h = out
            feats = None
        loss = _crit(logits, labels)
        mtp_h = head_stage.mtp(h[:, :-1], head_stage.mtp_embed(labels[:, :-1]),
                               feats=feats)
        mtp_logits = head_stage.head(mtp_h)
        return loss + w * _crit(mtp_logits, labels[:, 1:])

    return crit


def _make_moe_uninit(dim):
    # like _make_moe but without reseeding: stage init stays sequential
    return LatentMoE(dim, EXPERTS, TOPK, expert_hidden=16, n_shared=1,
                     shared_hidden=16, latent_dim=16, balancer="none")


def _dualpipe_ep_worker(rank, world, store_file, mtp=False, wire_bf16=False,
                        union=False):
    """Full 2D grid (P=2 pipeline x E=2 expert parallel) on 4 ranks.

    Gold standard: post-sync grads == single-process chained-stage grads on
    the CONCATENATED batch of both data-parallel columns; after a MuonClip
    step, all 4 replicas of each stage remain bit-identical. With mtp truthy
    the last stage carries the MTP head and the criterion adds the t+2 loss.
    With wire_bf16 the payloads cross ranks in bf16: the fp32 reference
    comparison is skipped (wire rounding), but losses must be finite and the
    replica bit-identity gate still applies.
    """
    _init(rank, world, store_file)
    import copy

    from latentmoe.optim import MuonClip
    from latentmoe.parallel import Buffer, DualPipe, ep_grad_sync
    from latentmoe.parallel.dualpipe import sync_mirror_grads, sync_union_grads

    P = E = 2
    pipe_rank, ep_rank = rank // E, rank % E
    rows = [dist.new_group(ranks=[q * E + c for c in range(E)]) for q in range(P)]
    cols = [dist.new_group(ranks=[q * E + c for q in range(P)]) for c in range(E)]
    unions = [dist.new_group(ranks=[q * E + c for c in range(E)]
                             + [(P - 1 - q) * E + c for c in range(E)])
              for q in range(P // 2)]

    vocab, dim = 64, DIM
    num_chunks = 2 * P * 2
    torch.manual_seed(0)
    stages = [_MoeStage(dim, s == 0, s == P - 1, vocab, mtp=mtp) for s in range(P)]

    # per-column data; the reference sees both columns' batches concatenated
    def col_batch(e):
        g = torch.Generator().manual_seed(42 + e)
        x = torch.randint(0, vocab, (num_chunks * MB, SEQ), generator=g)
        y = torch.randint(0, vocab, (num_chunks * MB, SEQ), generator=g)
        return x, y

    x_me, y_me = col_batch(ep_rank)
    x_all = torch.cat([col_batch(e)[0] for e in range(E)])
    y_all = torch.cat([col_batch(e)[1] for e in range(E)])

    if not wire_bf16:  # fp32 reference (meaningless under wire rounding)
        ref_stages = copy.deepcopy(stages)  # deepcopy BEFORE attaching dispatchers
        h = ref_stages[0](x_all)
        for s in ref_stages[1:]:
            h = s(*h) if isinstance(h, tuple) else s(h)
        (_mtp_crit(ref_stages[-1])(h, y_all) if mtp else _crit(h, y_all)).backward()

    mod0 = stages[pipe_rank]
    mod1 = copy.deepcopy(stages[P - 1 - pipe_rank])
    buffer = Buffer(EXPERTS, group=rows[pipe_rank])
    cdt = torch.bfloat16 if wire_bf16 else torch.float32
    for mod in (mod0, mod1):
        mod.moe.dispatcher = buffer
        mod.wire_dtype = torch.bfloat16 if wire_bf16 else None
    dp = DualPipe((mod0, mod1) if pipe_rank < P // 2 else (mod1, mod0),
                  group=cols[ep_rank])
    head = mod0 if pipe_rank == P - 1 else (mod1 if pipe_rank == 0 else None)
    crit = _mtp_crit(head) if mtp else _crit

    specs = None
    if mtp == "fuse":  # (hidden, tap) arrives at stage 1; stage 0 is fed
        pair = [((MB, SEQ, dim), cdt)] * 2
        spec_into = lambda s: pair[:1] if s == 0 else pair
        specs = {0: spec_into(pipe_rank), 1: spec_into(P - 1 - pipe_rank)}

    half = x_me.shape[0] // 2
    x_feed = x_me[:half] if pipe_rank == 0 else (x_me[half:] if pipe_rank == P - 1 else None)
    labels = y_me[:half] if pipe_rank == P - 1 else (y_me[half:] if pipe_rank == 0 else None)
    loss = dp.step(x_feed, num_chunks=num_chunks, criterion=crit, labels=labels,
                   chunk_shape=(MB, SEQ, dim), chunk_dtype=cdt,
                   chunk_specs=specs)
    if union:  # ONE all-reduce per stage over both mirror rows (UPDATE 18)
        sync_union_grads(dp, unions[min(pipe_rank, P - 1 - pipe_rank)], E)
    else:
        sync_mirror_grads(dp)                       # column-batch grads
        ep_grad_sync(dp.module, group=rows[pipe_rank])  # global-mean grads

    if wire_bf16:
        if loss is not None:
            assert torch.isfinite(loss), loss
    else:
        ref = ref_stages[pipe_rank]             # my direction-0 stage
        for (n, p), (_, pr) in zip(dp._mod(0).named_parameters(), ref.named_parameters()):
            assert p.grad is not None, n
            assert torch.allclose(p.grad, pr.grad, atol=1e-4), \
                (rank, n, (p.grad - pr.grad).abs().max())

    # identical grads + identical optimizer -> all 2E=4 replicas of each
    # stage must stay bit-identical after the step
    opt = MuonClip.from_model(dp.module, lr=1e-3)
    opt.step()
    for s in range(P):  # with P=2 every rank holds a copy of both stages
        mod = dp._mod(0) if pipe_rank == s else dp._mod(1)
        flat = torch.cat([p.detach().flatten() for p in mod.parameters()])
        gathered = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(gathered, flat)
        for g in range(world):
            assert torch.equal(flat, gathered[g]), \
                (rank, f"stage {s} replica on rank {g} diverged")
    dist.barrier()
    dist.destroy_process_group()


# --------------------------------------------------------------------- #
class _TapStage(nn.Module):
    """Stages exchanging (hidden, tap) MULTI-TENSOR payloads. Middle stages
    pass the tap through UNTOUCHED — the identity edge (an output that IS a
    received leaf input) autograd must still route grads across."""

    def __init__(self, dim, vocab, kind):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(vocab, dim) if kind == "first" else None
        self.lin = nn.Linear(dim, dim)
        self.fuse = nn.Linear(2 * dim, dim) if kind == "last" else None
        self.head = nn.Linear(dim, vocab) if kind == "last" else None

    def forward(self, x, tap=None):
        if self.kind == "first":
            h = self.embed(x)
            tap = torch.tanh(self.lin(h))
            return h + tap, tap
        h = x + torch.tanh(self.lin(x))
        if self.kind == "mid":
            return h, tap                      # tap: identity pass-through
        return self.head(self.fuse(torch.cat([h, tap], dim=-1)))


def _tap_pipeline_worker(rank, world, store_file):
    """Multi-tensor transport vs plain chained backward (P=4, 2 mid stages)."""
    _init(rank, world, store_file)
    import copy

    from latentmoe.parallel import DualPipe
    from latentmoe.parallel.dualpipe import sync_mirror_grads

    P = world
    vocab, dim = 64, DIM
    num_chunks = 2 * P * 2
    torch.manual_seed(0)
    kinds = ["first"] + ["mid"] * (P - 2) + ["last"]
    stages = [_TapStage(dim, vocab, k) for k in kinds]

    torch.manual_seed(42)
    x = torch.randint(0, vocab, (num_chunks * MB, SEQ))
    y = torch.randint(0, vocab, (num_chunks * MB, SEQ))

    ref_stages = copy.deepcopy(stages)
    out = ref_stages[0](x)
    for s in ref_stages[1:]:
        out = s(*out)
    _crit(out, y).backward()

    mod0 = stages[rank]
    mod1 = copy.deepcopy(stages[P - 1 - rank])
    dp = DualPipe((mod0, mod1) if rank < P // 2 else (mod1, mod0))

    pair = [((MB, SEQ, dim), torch.float32)] * 2
    spec_into = lambda s: pair[:1] if s == 0 else pair   # stage 0 is fed
    specs = {0: spec_into(rank), 1: spec_into(P - 1 - rank)}

    half = x.shape[0] // 2
    x_feed = x[:half] if rank == 0 else (x[half:] if rank == P - 1 else None)
    labels = y[:half] if rank == P - 1 else (y[half:] if rank == 0 else None)
    dp.step(x_feed, num_chunks=num_chunks, criterion=_crit, labels=labels,
            chunk_specs=specs)
    sync_mirror_grads(dp)

    ref = ref_stages[rank]
    for (n, p), (_, pr) in zip(dp._mod(0).named_parameters(), ref.named_parameters()):
        assert p.grad is not None, n
        assert torch.allclose(p.grad, pr.grad, atol=1e-4), \
            (rank, n, (p.grad - pr.grad).abs().max())
    dist.barrier()
    dist.destroy_process_group()


def test_dualpipe_multitensor_matches_reference(tmp_path):
    world = 4
    if world > (os.cpu_count() or 2):
        pytest.skip("not enough cpus")
    mp.spawn(_tap_pipeline_worker, args=(world, str(tmp_path / "tap_store")),
             nprocs=world, join=True)


@pytest.mark.parametrize("mtp", [False, "plain", "fuse"])
def test_dualpipe_ep_2d_matches_reference(tmp_path, mtp):
    world = 4
    if world > (os.cpu_count() or 2):
        pytest.skip("not enough cpus")
    mp.spawn(_dualpipe_ep_worker,
             args=(world, str(tmp_path / f"dp2d_store_{mtp}"), mtp),
             nprocs=world, join=True)


def _residual_pipe_worker(rank, world, store_file, residual):
    """mHC / AttnRes residual schemes under DualPipe via multi-tensor
    payloads (driver Stage + payload_spec imported from train_2d, so the
    conventions are single-sourced). Gold standard: pipeline grads ==
    plain chained backward on the full batch."""
    _init(rank, world, store_file)
    import copy

    import train_2d as t2d
    from latentmoe import get_preset
    from latentmoe.parallel import DualPipe
    from latentmoe.parallel.dualpipe import sync_mirror_grads

    P = world
    cfg = get_preset("k3", vocab_size=128, dim=32, n_layers=P, n_heads=2,
                     head_dim=16, kda_head_dim=16, kda_chunk=4,
                     n_routed_experts=4, top_k=2, moe_latent_dim=16,
                     expert_hidden=16, shared_expert_hidden=32,
                     dense_ffn_hidden=64, kv_latent_dim=16, q_latent_dim=16,
                     residual=residual, use_mtp=False, mhc_streams=2,
                     balancer="none", max_seq_len=SEQ)
    per = cfg.n_layers // P
    num_chunks = 2 * P * 2
    torch.manual_seed(0)
    stages = [t2d.Stage(cfg, range(s * per, (s + 1) * per), s == 0, s == P - 1)
              for s in range(P)]

    torch.manual_seed(42)
    x = torch.randint(0, cfg.vocab_size, (num_chunks * MB, SEQ))
    y = torch.randint(0, cfg.vocab_size, (num_chunks * MB, SEQ))

    ref_stages = copy.deepcopy(stages)
    out = ref_stages[0](x)
    for s in ref_stages[1:]:
        out = s(*out) if isinstance(out, tuple) else s(out)
    _crit(out, y).backward()

    mod0 = stages[rank]
    mod1 = copy.deepcopy(stages[P - 1 - rank])
    dp = DualPipe((mod0, mod1) if rank < P // 2 else (mod1, mod0))
    f32 = torch.float32
    specs = {0: t2d.payload_spec(rank, cfg, per, MB, SEQ, (), f32),
             1: t2d.payload_spec(P - 1 - rank, cfg, per, MB, SEQ, (), f32)}

    half = x.shape[0] // 2
    x_feed = x[:half] if rank == 0 else (x[half:] if rank == P - 1 else None)
    labels = y[:half] if rank == P - 1 else (y[half:] if rank == 0 else None)
    dp.step(x_feed, num_chunks=num_chunks, criterion=_crit, labels=labels,
            chunk_specs=specs)
    sync_mirror_grads(dp)

    ref = ref_stages[rank]
    for (n, p), (_, pr) in zip(dp._mod(0).named_parameters(), ref.named_parameters()):
        assert p.grad is not None, n
        assert torch.allclose(p.grad, pr.grad, atol=1e-4), \
            (rank, residual, n, (p.grad - pr.grad).abs().max())
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("residual", ["mhc", "attnres"])
def test_residual_schemes_pipeline_match_reference(tmp_path, residual):
    world = 2
    mp.spawn(_residual_pipe_worker,
             args=(world, str(tmp_path / f"res_{residual}"), residual),
             nprocs=world, join=True)


def test_dualpipe_ep_2d_bf16_wire(tmp_path):
    """bf16 payloads (activations, taps, AND backward grads) over the pipe:
    finite losses + all-replica bit-identity through a MuonClip step."""
    world = 4
    if world > (os.cpu_count() or 2):
        pytest.skip("not enough cpus")
    mp.spawn(_dualpipe_ep_worker,
             args=(world, str(tmp_path / "dp2d_bf16"), "fuse", True),
             nprocs=world, join=True)


def test_dualpipe_ep_2d_union_sync(tmp_path):
    """sync_union_grads (one all-reduce per stage over BOTH mirror rows)
    must reproduce the single-process reference grads exactly like the
    legacy mirror+row sync, and keep all replicas torch.equal."""
    world = 4
    if world > (os.cpu_count() or 2):
        pytest.skip("not enough cpus")
    mp.spawn(_dualpipe_ep_worker,
             args=(world, str(tmp_path / "dp2d_union"), "plain", False, True),
             nprocs=world, join=True)
