"""DualPipe: bidirectional pipeline parallelism (DeepSeek-V3/V4 training).

A PyTorch-only port of the DualPipe algorithm (github.com/deepseek-ai/DualPipe,
MIT license; schedule reproduced from the reference implementation, comm layer
rewritten on plain isend/irecv so it also runs on Gloo/XCCL, not just NCCL).

Idea: microbatches enter the pipeline FROM BOTH ENDS simultaneously.
Rank r holds TWO stages of the model: stage r (serving direction 0, ranks
0 -> P-1) and stage P-1-r (serving direction 1, ranks P-1 -> 0). Forward of
one direction overlaps with backward of the other, cutting the bubble to
(P/2 - 1)(F&B + B - 3W) vs (P-1)(F+B) for 1F1B.

Zero-bubble split: a backward can be split into B (input grads, on the
critical path -- unblocks the upstream rank) and W (weight grads, deferred).
We implement the split with plain autograd:  B = autograd.grad(out, inp,
retain_graph=True); W = a closure `autograd.backward(out, g, inputs=params)`
pushed to WeightGradStore and popped in the schedule's W slots.

Parameter duplication note (as in the paper): the two copies of stage s live
on rank s (direction 0) and rank P-1-s (direction 1). After step(), call
`sync_mirror_grads` -- mirror copies exchange and SUM their grads, so both
copies apply the identical full-batch update and stay bit-identical.

Losses: criterion runs on the rank holding the final stage of each direction
(rank P-1 for direction 0, rank 0 for direction 1); each chunk's loss is
scaled by 1/num_chunks so the sum over both directions is the global mean.
"""

from __future__ import annotations

import queue

import torch
import torch.distributed as dist

from .comm import group_peer


class WeightGradStore:
    """FIFO of deferred weight-gradient closures (zero-bubble W chunks)."""

    enabled: bool = False
    _cache: "queue.Queue" = queue.Queue()

    @classmethod
    def put(cls, fn) -> None:
        cls._cache.put(fn)

    @classmethod
    def pop(cls) -> None:
        if not cls._cache.empty():
            cls._cache.get()()

    @classmethod
    def flush(cls) -> None:
        while not cls._cache.empty():
            cls._cache.get()()


class DualPipe(torch.nn.Module):
    def __init__(self, modules: tuple[torch.nn.Module, torch.nn.Module],
                 group=None, device=None):
        super().__init__()
        assert len(modules) == 2, "(stage for direction 0, stage for direction 1)"
        self.module = torch.nn.ModuleList(modules)
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        assert self.world % 2 == 0, "DualPipe needs an even number of ranks"
        self.half_rank = min(self.rank, self.world - 1 - self.rank)
        self.second_half = self.rank >= self.world // 2
        self.device = device or torch.device("cpu")

        # p2p transport. DualPipe requires fire-and-forget sends: an unmatched
        # isend must progress while the sender blocks in an unrelated recv.
        # Gloo (TCP) and NCCL provide that; oneCCL/XCCL point-to-point has
        # been observed to stall in this pattern, so on those backends we
        # stage p2p through a Gloo side-group on CPU (correct, portable, and
        # exactly what the CPU test suite validates). Override with
        # LATENTMOE_PIPE_COMM=native|staged. Collectives stay on the default
        # backend either way.
        import os as _os
        backend = str(dist.get_backend(group)).lower()
        mode = _os.environ.get("LATENTMOE_PIPE_COMM")
        if mode is None:
            mode = "staged" if ("ccl" in backend and "nccl" not in backend) else "native"
        self.staged = mode == "staged"
        self.p2p_group = dist.new_group(backend="gloo") if self.staged else group
        # peer ranks are computed group-relative (self.rank +/- 1 etc.) but
        # torch p2p ops take GLOBAL dst/src ranks -- translate once here. The
        # staged gloo side-group spans all ranks, so global ranks address it
        # correctly too.
        self._global = [group_peer(group, r) for r in range(self.world)]
        if dist.get_rank() == 0:
            print(f"[dualpipe] p2p transport: {mode} (default backend: {backend})",
                  flush=True)

    # ---------------- direction geometry ---------------- #
    def _dir(self, phase: int) -> int:
        """Schedule phase (0/1) -> data-flow direction (0/1)."""
        return phase if not self.second_half else 1 - phase

    def _mod(self, d: int) -> torch.nn.Module:
        return self.module[0] if d == self._dir(0) else self.module[1]

    def _prev(self, d: int) -> int:
        return self.rank - 1 if d == 0 else self.rank + 1

    def _next(self, d: int) -> int:
        return self.rank + 1 if d == 0 else self.rank - 1

    def _is_first(self, d: int) -> bool:
        return self.rank == (0 if d == 0 else self.world - 1)

    def _is_last(self, d: int) -> bool:
        return self.rank == (self.world - 1 if d == 0 else 0)

    # ---------------- p2p plumbing ---------------- #
    def _commit(self, ops):
        """ops: list of ('send'|'recv', tensor, peer).

        Sends are posted and drained only at the end of step() -- waiting for
        a send here could deadlock on rendezvous transports, since the
        matching recv may be scheduled a few chunks later on the peer. The
        sent tensors are kept alive by `_pending_sends` until drained.
        Recvs block: by the schedule's construction the matching send has
        already been posted on the peer.
        """
        recvs = []
        for kind, t, peer in ops:
            peer = self._global[peer]
            if kind == "send":
                tc = t.detach().contiguous()
                if self.staged:
                    tc = tc.cpu()
                self._pending_sends.append(
                    (dist.isend(tc, peer, group=self.p2p_group), tc))
            else:
                if self.staged and t.device.type != "cpu":
                    buf = torch.empty(t.shape, dtype=t.dtype, device="cpu")
                    recvs.append((dist.irecv(buf, peer, group=self.p2p_group), buf, t))
                else:
                    recvs.append((dist.irecv(t, peer, group=self.p2p_group), None, t))
        for w, buf, t in recvs:
            w.wait()
            if buf is not None:
                t.copy_(buf)

    # ---------------- chunk ops ---------------- #
    # Inter-stage payloads are TUPLES of tensors (single-tensor pipelines are
    # the 1-tuple case). Stages receive them as positional args and may return
    # a tensor or a tuple; payload order is preserved end to end, so the
    # peer's grad recv buffers (`empty_like` of its saved outputs) line up
    # with our grad sends by construction.
    def _feed_or_recv_input(self, d: int, ops):
        if self._is_first(d):
            self._inputs[d].append((self._feed[d].pop(0),))
        else:
            bufs = tuple(torch.empty(*shape, dtype=dtype, device=self.device)
                         for shape, dtype in self._chunk_specs[d])
            for b in bufs:
                ops.append(("recv", b, self._prev(d)))
            self._inputs[d].append(bufs)

    def _forward_chunk(self, phase: int):
        d = self._dir(phase)
        ops: list = []
        self._feed_or_recv_input(d, ops)
        self._commit(ops)
        xs = self._inputs[d].pop(0)
        inps = xs if self._is_first(d) else tuple(
            x.detach().requires_grad_(x.is_floating_point()) for x in xs)
        out = self._mod(d)(*inps)
        outs = out if isinstance(out, tuple) else (out,)
        loss = None
        if self._is_last(d):
            if self._criterion is not None:
                tgt = self._labels[d].pop(0)
                loss = self._criterion(out, tgt) / self._num_chunks
                self._losses[d].append(loss)
        else:
            self._commit([("send", o.detach(), self._next(d)) for o in outs])
        self._records[d].append((inps, outs, loss))

    def _backward_chunk(self, phase: int, enable_zb: bool = False):
        d = self._dir(phase)
        inps, outs, loss = self._records[d].pop(0)
        if loss is not None:
            roots, grads = (loss,), (None,)
        else:
            # explicit empty (not empty_like): outputs may be non-contiguous
            # (e.g. mHC stream state) and irecv needs contiguous buffers
            bufs = tuple(torch.empty(o.shape, dtype=o.dtype, device=o.device)
                         for o in outs)
            self._commit([("recv", b, self._next(d)) for b in bufs])
            roots, grads = outs, bufs
        grad_inps = () if self._is_first(d) else tuple(
            x for x in inps if x.is_floating_point() and x.requires_grad)
        if enable_zb:
            params = [p for p in self._mod(d).parameters() if p.requires_grad]
            gins = torch.autograd.grad(roots, grad_inps, grads, retain_graph=True,
                                       allow_unused=True) if grad_inps else ()
            WeightGradStore.put(
                lambda r=roots, g=grads, ps=params: torch.autograd.backward(r, g, inputs=ps))
        else:
            torch.autograd.backward(roots, grads)
            gins = tuple(x.grad for x in grad_inps)
        if grad_inps:
            self._commit([("send", g if g is not None else torch.zeros_like(x),
                           self._prev(d)) for g, x in zip(gins, grad_inps)])

    def _forward_backward_chunk(self, fwd_phase: int, bwd_phase: int):
        # Reference DualPipe overlaps these via module-provided
        # `overlapped_forward_backward`; without fused kernels we execute
        # forward first (its sends unblock the neighbor) then backward.
        self._forward_chunk(fwd_phase)
        self._backward_chunk(bwd_phase)

    @staticmethod
    def _weight_chunk():
        WeightGradStore.pop()

    # ---------------- the 8-phase schedule ---------------- #
    def step(self, x: torch.Tensor | None = None, *, num_chunks: int,
             criterion=None, labels: torch.Tensor | None = None,
             chunk_shape=None, chunk_dtype=torch.float32, chunk_specs=None):
        """Run one full DualPipe training step.

        x:      inputs, only on rank 0 (direction 0) / rank P-1 (direction 1)
        labels: only on rank P-1 (for direction-0 loss) / rank 0 (direction 1)
        chunk_shape/chunk_dtype: shape of one inter-stage activation chunk
        chunk_specs: optional {direction: [(shape, dtype), ...]} describing a
            MULTI-TENSOR payload arriving at this rank per direction (e.g.
            hidden state + MTP feature taps); overrides chunk_shape/dtype.
            Stages then take the payload as positional args and may return a
            tuple; the loss stage's raw return goes to the criterion.
        Returns summed loss (on loss ranks) or None.
        """
        P, R, hr = self.world, self.world // 2, self.half_rank
        assert num_chunks % 2 == 0 and num_chunks >= 2 * P, \
            "num_chunks must be even and >= 2 * world"
        HC = num_chunks // 2
        self._num_chunks = num_chunks
        self._criterion = criterion
        if chunk_specs is None:
            spec = [(tuple(chunk_shape), chunk_dtype)]
            chunk_specs = {0: spec, 1: spec}
        self._chunk_specs = chunk_specs
        self._inputs = {0: [], 1: []}
        self._records = {0: [], 1: []}
        self._losses = {0: [], 1: []}
        self._feed = {0: [], 1: []}
        self._labels = {0: [], 1: []}
        self._pending_sends = []
        for d in (0, 1):
            if self._is_first(d):
                assert x is not None, f"rank {self.rank} feeds direction {d}: pass x"
                self._feed[d] = list(x.chunk(HC, dim=0))
            if self._is_last(d):
                assert criterion is not None and labels is not None, \
                    f"rank {self.rank} computes direction-{d} loss: pass criterion+labels"
                self._labels[d] = list(labels.chunk(HC, dim=0))

        # phase 1: nF0
        for _ in range((R - hr - 1) * 2):
            self._forward_chunk(0)
        # phase 2: nF0F1
        for _ in range(hr + 1):
            self._forward_chunk(0)
            self._forward_chunk(1)
        # phase 3: nB1W1F1
        for _ in range(R - hr - 1):
            self._backward_chunk(1, enable_zb=True)
            self._weight_chunk()
            self._forward_chunk(1)
        # phase 4: nF0B1F1B0 (main loop)
        for _ in range(HC - P + hr + 1):
            self._forward_backward_chunk(0, 1)
            self._forward_backward_chunk(1, 0)
        # phase 5: nB1F1B0
        for _ in range(R - hr - 1):
            self._backward_chunk(1)
            self._forward_backward_chunk(1, 0)
        # phase 6: nB1B0 (zero-bubble kicks in at the midpoint)
        for i in range(hr + 1):
            self._backward_chunk(1, enable_zb=2 * i >= hr + 1)
            self._backward_chunk(0, enable_zb=2 * i + 1 >= hr + 1)
        # phase 7: nWB0
        for _ in range(R - hr - 1):
            self._weight_chunk()
            self._backward_chunk(0, enable_zb=True)
        # phase 8: nW
        WeightGradStore.flush()
        for w, _ in self._pending_sends:
            w.wait()
        self._pending_sends.clear()

        losses = self._losses[0] + self._losses[1]
        return torch.stack([l.detach() for l in losses]).sum() if losses else None


@torch.no_grad()
def sync_union_grads(dualpipe: DualPipe, union_group, n_columns: int,
                     bucketed: bool = True):
    """2D-grid replacement for sync_mirror_grads + per-row ep_grad_sync.

    ONE SUM-all-reduce per held stage over the UNION of the two mirror pipe
    rows (2E ranks), each rank contributing its direction-appropriate
    module, stages synced in ascending-id order (identical on every member,
    so the collectives line up). Design intent: every replica of a stage
    receives its gradient from the SAME collective on the SAME communicator,
    removing the legacy path's cross-communicator asymmetry AND its pairwise
    mirror p2p exchange (~457 MB/dir over staged-Gloo TCP on XCCL) — field
    result: -30% step time at dim-1024 (HANDOFF UPDATEs 18-19). NOTE the
    "replicas bit-identical by construction" prediction was FALSIFIED in the
    field (drift ~1e-6 relative persists, source still under investigation —
    UPDATEs 19-21, train_2d.py --debug-grad-drift). Each replica holds one
    direction of one column's chunk grads (losses pre-scaled by
    1/num_chunks), so the 2E-replica SUM divided by n_columns == E is the
    exact global-mean gradient — the same math as the legacy two-stage sync
    (test-pinned to the single-process reference).
    """
    from .comm import allreduce_grads

    P, p = dualpipe.world, dualpipe.rank
    for stage in sorted((p, P - 1 - p)):
        mod = dualpipe._mod(0) if stage == p else dualpipe._mod(1)
        grads = []
        for q in mod.parameters():
            if q.grad is None:
                q.grad = torch.zeros_like(q)
            grads.append(q.grad)
        allreduce_grads(grads, group=union_group, divisor=n_columns,
                        bucketed=bucketed)


@torch.no_grad()
def sync_mirror_grads(dualpipe: DualPipe):
    """SUM gradients between the two replicas of each stage.

    My direction-0 stage is stage `rank`; the other copy of stage `rank` is
    the direction-1 stage on rank P-1-rank (and symmetrically for my
    direction-1 stage). Each pair exchanges grads and sums, after which both
    replicas hold the identical full-batch gradient.
    """
    rank, world = dualpipe.rank, dualpipe.world
    group, staged = dualpipe.p2p_group, dualpipe.staged
    peer = dualpipe._global[world - 1 - rank]
    dir0_mod, dir1_mod = dualpipe._mod(0), dualpipe._mod(1)
    for mod in (dir0_mod, dir1_mod):
        for p in mod.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
    send0 = [p.grad for p in dir0_mod.parameters()]
    send1 = [p.grad for p in dir1_mod.parameters()]
    stage = (lambda g: g.contiguous().cpu()) if staged else (lambda g: g.contiguous())
    payload = [stage(g) for g in send0 + send1]
    reqs = [dist.isend(g, peer, group=group) for g in payload]
    # Peer posts [its dir0..., its dir1...] in the same order. Peer's dir0
    # stage == my dir1 stage and vice versa, so receive into swapped buffers.
    dev = "cpu" if staged else None
    peer_dir0 = [torch.empty(g.shape, dtype=g.dtype, device=dev or g.device) for g in send1]
    peer_dir1 = [torch.empty(g.shape, dtype=g.dtype, device=dev or g.device) for g in send0]
    for buf in peer_dir0 + peer_dir1:
        dist.recv(buf, peer, group=group)
    for w in reqs:
        w.wait()
    for p, extra in zip(dir0_mod.parameters(), peer_dir1):
        p.grad.add_(extra.to(p.grad.device))
    for p, extra in zip(dir1_mod.parameters(), peer_dir0):
        p.grad.add_(extra.to(p.grad.device))
