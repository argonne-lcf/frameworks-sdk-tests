"""Backend-agnostic distributed helpers.

Backends: NCCL (NVIDIA / Grace-Hopper), XCCL or oneCCL (Intel GPUs), Gloo
(CPU -- used by the unit tests). The variable-split all-to-all at the heart of
expert parallelism is native on NCCL/XCCL; Gloo lacks it, so a functionally
identical isend/irecv fallback keeps everything testable on CPU.

`all_to_all_autograd` is the differentiable primitive DeepEP-style dispatch /
combine is built on: backward of an all-to-all is the reverse all-to-all.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from ..accel import best_device, default_dist_backend, device_module

_A2A_NATIVE_BACKENDS = ("nccl", "xccl", "ccl", "mpi")


def _first_env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return None


def init_distributed(device: torch.device | None = None):
    """Launcher-agnostic init. Returns (rank, world_size, device).

    Understands, in priority order:
      * torchrun            (RANK / WORLD_SIZE / LOCAL_RANK / MASTER_*)
      * mpiexec + PALS/PMI  (PMI_RANK / PMI_SIZE / PMI_LOCAL_RANK, PALS_*)
                            -- ALCF Aurora/Sunspot `mpiexec -n N -ppn P`
      * OpenMPI mpirun      (OMPI_COMM_WORLD_*)
      * Slurm srun          (SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID)

    When MASTER_ADDR is unset (typical under mpiexec), it is derived from the
    first line of $PBS_NODEFILE (rendezvous/TCPStore traffic only -- the
    actual collectives run over the fabric via XCCL/NCCL). Port from
    MASTER_PORT, default 29500.
    """
    # variable names collected from torchrun, PALS (both PMI and PMIx modes),
    # Cray MPICH, Intel MPI, OpenMPI and Slurm
    rank = int(_first_env("RANK", "PALS_RANKID", "PMIX_RANK", "PMI_RANK",
                          "OMPI_COMM_WORLD_RANK", "SLURM_PROCID") or 0)
    local_size = _first_env("PALS_LOCAL_SIZE", "PMI_LOCAL_SIZE", "MPI_LOCALNRANKS",
                            "I_MPI_LOCAL_SIZE", "OMPI_COMM_WORLD_LOCAL_SIZE")
    world_env = _first_env("WORLD_SIZE", "PALS_WORLD_SIZE", "PMI_SIZE", "PALS_SIZE",
                           "OMPI_COMM_WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS")
    if world_env is None:
        # PALS under PMIx exports a rank (PMIX_RANK) but NO size variable;
        # reconstruct it as ranks-per-node x number of allocated nodes.
        nodefile = os.environ.get("PBS_NODEFILE")
        if local_size and nodefile and os.path.exists(nodefile):
            with open(nodefile) as f:
                n_nodes = sum(1 for line in f if line.strip())
            world_env = str(int(local_size) * n_nodes)
    world = int(world_env or 1)
    local_rank = int(_first_env("LOCAL_RANK", "PALS_LOCAL_RANKID", "PMI_LOCAL_RANK",
                                "MPI_LOCALRANKID", "I_MPI_LOCAL_RANK",
                                "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID") or 0)
    if world > 1:
        # oneCCL (under torch-XCCL) can need its own local topology hints when
        # no recognized process launcher is in play; harmless elsewhere.
        os.environ.setdefault("CCL_LOCAL_RANK", str(local_rank))
        if local_size:
            os.environ.setdefault("CCL_LOCAL_SIZE", str(local_size))
    if os.environ.get("LATENTMOE_DIST_DEBUG"):
        src = {k: v for k, v in sorted(os.environ.items())
               if any(t in k for t in ("PMI", "PALS", "RANK", "WORLD", "MASTER", "SLURM", "OMPI"))}
        print(f"[latentmoe dist] resolved rank={rank} world={world} "
              f"local_rank={local_rank} from env: {src}", flush=True)
    if world > 1 and "MASTER_ADDR" not in os.environ:
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            with open(nodefile) as f:
                os.environ["MASTER_ADDR"] = f.readline().strip().split()[0]
        else:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29500")
    # normalize for anything downstream that reads the torchrun names
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(local_rank)
    device = best_device() if device is None else device
    if device.type in ("cuda", "xpu"):
        device_module(device).set_device(local_rank)
        device = torch.device(device.type, local_rank)
    elif device.type == "mps" and world > 1:
        device = torch.device("cpu")  # gloo cannot send/recv MPS tensors
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend=default_dist_backend(device), rank=rank,
                                world_size=world)
    return rank, world, device


def _backend(group=None) -> str:
    try:
        return dist.get_backend(group)
    except Exception:
        return "gloo"


def group_peer(group, r: int) -> int:
    """Group-relative rank -> global rank. torch.distributed p2p ops interpret
    dst/src as GLOBAL ranks regardless of the `group` argument, so any peer
    index computed inside a subgroup must be translated at the call boundary.
    (Identity on the default group, where the two numberings coincide.)"""
    return r if group is None else dist.get_global_rank(group, r)


def all_to_all_single_any(x: torch.Tensor, out_splits: list[int], in_splits: list[int],
                          group=None) -> torch.Tensor:
    """dist.all_to_all_single with a Gloo-compatible fallback."""
    world = dist.get_world_size(group)
    out = x.new_empty(sum(out_splits), *x.shape[1:])
    if _backend(group) in _A2A_NATIVE_BACKENDS:
        dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits, group=group)
        return out
    # Gloo fallback: post all sends, blocking-recv in rank order.
    rank = dist.get_rank(group)
    in_offs = [0] + list(torch.cumsum(torch.tensor(in_splits), 0).tolist())
    out_offs = [0] + list(torch.cumsum(torch.tensor(out_splits), 0).tolist())
    reqs = []
    xc = x.contiguous()
    for r in range(world):
        if r == rank or in_splits[r] == 0:
            continue
        chunk = xc[in_offs[r]: in_offs[r + 1]]
        reqs.append(dist.isend(chunk.contiguous(), dst=group_peer(group, r), group=group))
    out[out_offs[rank]: out_offs[rank + 1]] = xc[in_offs[rank]: in_offs[rank + 1]]
    for r in range(world):
        if r == rank or out_splits[r] == 0:
            continue
        recv = out[out_offs[r]: out_offs[r + 1]].contiguous()
        dist.recv(recv, src=group_peer(group, r), group=group)
        out[out_offs[r]: out_offs[r + 1]] = recv
    for w in reqs:
        w.wait()
    return out


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_splits, in_splits, group):
        ctx.out_splits, ctx.in_splits, ctx.group = out_splits, in_splits, group
        return all_to_all_single_any(x, out_splits, in_splits, group)

    @staticmethod
    def backward(ctx, g):
        return (all_to_all_single_any(g.contiguous(), ctx.in_splits, ctx.out_splits,
                                      ctx.group), None, None, None)


def all_to_all_autograd(x, out_splits, in_splits, group=None):
    return _AllToAll.apply(x, out_splits, in_splits, group)


@torch.no_grad()
def allreduce_grads(grads, group=None, divisor: float = 1.0,
                    bucketed: bool = True):
    """SUM-all-reduce a list of grad tensors over `group`, then divide by
    `divisor`. bucketed=True fuses into ONE collective per (device, dtype);
    bucketed=False keeps a per-tensor loop (A/B baseline)."""
    if not bucketed:
        for g in grads:
            dist.all_reduce(g, op=dist.ReduceOp.SUM, group=group)
            g.div_(divisor)
        return
    buckets: dict = {}
    for g in grads:
        buckets.setdefault((g.device, g.dtype), []).append(g)
    for gs in buckets.values():
        flat = torch.cat([g.reshape(-1) for g in gs])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        flat.div_(divisor)
        off = 0
        for g in gs:
            g.copy_(flat[off: off + g.numel()].view_as(g))
            off += g.numel()


@torch.no_grad()
def ep_grad_sync(model: torch.nn.Module, group=None, bucketed: bool = True):
    """Gradient sync for expert-parallel training with replicated parameters.

    With EP dispatch, each expert's gradient exists only on its owner rank
    (zeros elsewhere) while replicated params carry per-rank data gradients;
    all-reduce(SUM) / world unifies both cases into the exact global-mean
    gradient.

    bucketed=True (default) flattens grads into ONE all-reduce per
    (device, dtype) instead of one per parameter — same math, and identical
    on every group member (determinism argument unchanged: same inputs, same
    group, same algorithm), but per-element fp reduction order can differ
    from the per-param path at the last-ulp level. bucketed=False keeps the
    original per-parameter loop for A/B comparison.
    """
    world = dist.get_world_size(group)
    if world == 1:
        return
    grads = []
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    allreduce_grads(grads, group=group, divisor=world, bucketed=bucketed)
