"""DeepEP-style expert-parallel dispatch/combine, in pure torch.distributed.

Mirrors the DeepEP `Buffer` API shape (dispatch -> handle -> combine) but is
built entirely on differentiable all-to-all collectives, so it runs unchanged
on NCCL (NVIDIA / Grace-Hopper), XCCL/oneCCL (Intel), and Gloo (CPU tests).
The real DeepEP gains its speed from NVSHMEM / RDMA kernels; the *interface
and dataflow* here are the same, which is what the rest of the code needs:

  1. `get_dispatch_layout`: how many of my (token, expert) assignments go to
     each rank / expert.
  2. `dispatch`: ship each assignment's activation row to the rank owning the
     expert; rows arrive grouped by local expert, ready for one grouped GEMM.
  3. `combine`: ship expert outputs back, weight by the router gates, reduce
     per token.

Backward of dispatch is a combine-shaped all-to-all and vice versa (autograd
handled by `all_to_all_autograd`), so EP training just works: run the loss
backward on every rank in lockstep and gradients flow across ranks through
the collectives.

`moe_forward` is the convenience the LatentMoE layer calls; `handle` caching
between decode steps (same routing -> reuse layout, no re-sync) mirrors
DeepEP's low-latency-mode trick.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .comm import all_to_all_autograd, all_to_all_single_any


class Buffer:
    def __init__(self, num_experts: int, group=None):
        self.group = group
        self.world = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        assert num_experts % self.world == 0, (
            f"num_experts={num_experts} must be divisible by the EP world size "
            f"({self.world}). Pass e.g. --experts "
            f"{self.world * max(1, round(num_experts / self.world))} "
            f"(any multiple of {self.world}).")
        self.num_experts = num_experts
        self.local_experts = num_experts // self.world

    @property
    def expert_slice(self) -> slice:
        return slice(self.rank * self.local_experts, (self.rank + 1) * self.local_experts)

    # ------------------------------------------------------------------ #
    def get_dispatch_layout(self, topk_idx: torch.Tensor):
        """topk_idx: [N, k] global expert ids ->
        (num_tokens_per_rank [world], num_tokens_per_expert [E])."""
        flat = topk_idx.reshape(-1)
        per_expert = torch.bincount(flat, minlength=self.num_experts)
        per_rank = per_expert.view(self.world, self.local_experts).sum(-1)
        return per_rank, per_expert

    # ------------------------------------------------------------------ #
    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor,
                 topk_weights: torch.Tensor, handle: dict | None = None):
        """x: [N, d]; topk_idx/topk_weights: [N, k].

        Returns (recv_x [M, d] grouped by local expert,
                 group_sizes [local_experts], handle).
        Pass a previous `handle` back in (same routing) to skip layout work --
        DeepEP's cached-handle decode path.
        """
        N, k = topk_idx.shape
        if handle is None:
            flat = topk_idx.reshape(-1)                       # [N*k]
            target_rank = flat // self.local_experts
            order = torch.argsort(target_rank, stable=True)   # send order, by rank
            in_splits = torch.bincount(target_rank, minlength=self.world)
            # exchange split sizes
            out_splits = all_to_all_single_any(
                in_splits.to(x.device) if in_splits.device != x.device else in_splits,
                [1] * self.world, [1] * self.world, self.group,
            ) if self.world > 1 else in_splits
            in_splits = in_splits.tolist()
            out_splits = out_splits.tolist()
            send_expert = (flat % self.local_experts)[order]
            if self.world > 1:
                recv_expert = all_to_all_single_any(
                    send_expert.contiguous(), out_splits, in_splits, self.group)
            else:
                recv_expert = send_expert
            recv_order = torch.argsort(recv_expert, stable=True)  # group by local expert
            group_sizes = torch.bincount(recv_expert, minlength=self.local_experts)
            handle = dict(order=order, in_splits=in_splits, out_splits=out_splits,
                          recv_order=recv_order, group_sizes=group_sizes,
                          n_tokens=N, top_k=k)

        send_x = x[handle["order"] // k]                       # replicate per assignment
        if self.world > 1:
            recv_x = all_to_all_autograd(send_x, handle["out_splits"],
                                         handle["in_splits"], self.group)
        else:
            recv_x = send_x
        recv_x = recv_x[handle["recv_order"]]
        return recv_x, handle["group_sizes"], handle

    # ------------------------------------------------------------------ #
    def combine(self, y: torch.Tensor, topk_weights: torch.Tensor, handle: dict):
        """y: [M, d] expert outputs in dispatch's grouped order -> [N, d]."""
        inv_recv = torch.empty_like(handle["recv_order"])
        inv_recv[handle["recv_order"]] = torch.arange(
            handle["recv_order"].numel(), device=y.device)
        y = y[inv_recv]                                        # back to arrival order
        if self.world > 1:
            y = all_to_all_autograd(y, handle["in_splits"], handle["out_splits"],
                                    self.group)
        N, k = handle["n_tokens"], handle["top_k"]
        inv_order = torch.empty_like(handle["order"])
        inv_order[handle["order"]] = torch.arange(handle["order"].numel(), device=y.device)
        y = y[inv_order].view(N, k, -1)                        # original assignment order
        return (y * topk_weights.unsqueeze(-1).to(y.dtype)).sum(dim=1)

    # ------------------------------------------------------------------ #
    def moe_forward(self, x, topk_idx, topk_weights, expert_fn, handle=None):
        recv_x, group_sizes, handle = self.dispatch(x, topk_idx, topk_weights, handle)
        y = expert_fn(recv_x, group_sizes)
        return self.combine(y, topk_weights, handle)
