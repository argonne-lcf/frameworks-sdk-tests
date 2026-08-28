# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Minimal reproducer for the inductor stride-assert failure in
``aten._scaled_dot_product_flash_attention_backward`` under torch.compile.

Distilled from ezpz's fsdp_tp example (ezpz.models.llama + ezpz.examples.
fsdp_tp.parallelize). torch-only: no ezpz, wandb, datasets, or triton -- just
torch + torch.distributed. Two modes:

  * SINGLE-PROCESS (no torchrun): a plain stack of Llama-style
    TransformerBlocks. Kept for quick smoke tests, but does NOT reproduce the
    assertion on its own -- the transposed q/k/v never crosses the fwd/bwd
    partition boundary without the TP/FSDP DTensor context.

  * DISTRIBUTED (launched with torchrun): 2D Tensor-Parallel + FSDP2, mirroring
    ezpz.examples.fsdp_tp.parallelize() EXACTLY (ColwiseParallel wq/wk/wv,
    RowwiseParallel wo, SequenceParallel norms, PrepareModuleInput gathering
    the sequence dim to Replicate for attention, then per-block fully_shard).
    This is the configuration (the original failure was TP4 x FSDP3) that
    produces the local transposed q/k/v saved buffer inductor's flash-backward
    stride guard rejects.

Root cause under test:
    Attention passes q/k/v to F.scaled_dot_product_attention as transposed,
    NON-contiguous views:

        xq = xq.transpose(1, 2)  # (B, L, N, H) -> (B, N, L, H)

    giving, for local shape (B=4, N=4, L=4096, H=128), strides
    (2097152, 128, 512, 1). Under torch.compile with the min-cut partitioner
    (activation_memory_budget < 1.0), a saved q/k/v tensor crosses the
    partition boundary into the flash-attention backward op. Inductor bakes in
    a CONTIGUOUS layout from the (XPU) meta kernel, but the real kernel keeps
    the transposed strides, so the emitted ``assert_size_stride`` guard trips:

        AssertionError: expected size 4==4, stride 128==524288 at dim=1;
        expected size 4096==4096, stride 512==128 at dim=2
        Error in op: torch.ops.aten._scaled_dot_product_flash_attention_backward.default

Usage:
    # single-process smoke test (does NOT reproduce):
    python repro_sdpa_flash_backward_stride.py

    # distributed TP4 x FSDP3 on one node's 12 XPU tiles. Launch from a shell
    # that already has BOTH the oneAPI SYCL libs (libsycl.so.9) and torchrun on
    # PATH (a fresh `ssh NODE "conda activate ..."` breaks one or the other).
    # TORCHINDUCTOR_COMPILE_THREADS=1 avoids the async-compile OOM on the
    # 208-core node x 12 ranks; CCL_* are required by this cluster's xccl.
    DTAG=$(date "+%Y%m%d_%H%M%S")
    PYTHONUNBUFFERED=1 TORCH_DIST_INIT_BARRIER=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
        EZPZ_REPRO_AC=1 EZPZ_REPRO_BUDGET=0.5 \
        CCL_ATL_TRANSPORT=ofi CCL_PROCESS_LAUNCHER=none \
        torchrun --standalone --nproc_per_node=12 \
        repro_sdpa_flash_backward_stride.py \
        2>&1 | tee ~/datavol/logs/pytorch/repro_sdpa_flash_backward_stride_${DTAG}.log

    # triage toggles (env):
    EZPZ_REPRO_CONTIGUOUS=1   # candidate fix: .contiguous() q/k/v before SDPA
    EZPZ_REPRO_BUDGET=1.0     # disable the min-cut partitioner (no partition)
    EZPZ_REPRO_LAYERS=4       # more transformer blocks
    EZPZ_REPRO_TP=4           # tensor-parallel degree (distributed mode)
    EZPZ_REPRO_AC=1           # wrap each block in full activation checkpointing
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- GLOBAL model dims. Under TP the per-rank (local) attention shape becomes
# (B, N_HEADS/TP, L, HEAD_DIM) = (4, 4, 4096, 128) at TP=4 -- exactly the
# failing buffer. Single-process mode uses these as-is (TP=1). ----
BATCH = 4
SEQ_LEN = 4096
N_HEADS = 16
N_KV_HEADS = 8  # GQA: local kv heads = N_KV_HEADS/TP, exercises repeat_kv
HEAD_DIM = 128
DIM = N_HEADS * HEAD_DIM  # 2048
FFN_HIDDEN = 4 * DIM
N_LAYERS = int(os.environ.get("EZPZ_REPRO_LAYERS", "2"))
ROPE_THETA = 10000.0

# activation_memory_budget < 1.0 forces the inductor min-cut partitioner to
# split forward/backward -- what makes a saved q/k/v cross into the flash
# backward op. 1.0 disables it. Matches ezpz's --act-mem-budget.
ACT_MEM_BUDGET = float(os.environ.get("EZPZ_REPRO_BUDGET", "0.5"))

# EZPZ_REPRO_CONTIGUOUS=1 applies the candidate fix (contiguous q/k/v before
# SDPA) to confirm the assertion disappears.
FORCE_CONTIGUOUS = os.environ.get("EZPZ_REPRO_CONTIGUOUS", "0") == "1"

# Tensor-parallel degree used only in distributed (torchrun) mode.
TP_SIZE = int(os.environ.get("EZPZ_REPRO_TP", "4"))

# EZPZ_REPRO_AC=1 wraps each block in full activation checkpointing before
# compile (ezpz/torchtitan apply_ac pattern). Under AC the forward is NOT
# saved but RECOMPUTED in backward, so the transposed q/k/v views are re-run
# and fed directly into the flash-attention backward op -- a prime trigger for
# the inductor stride-assert that plain (no-AC) runs may lay out differently.
USE_AC = os.environ.get("EZPZ_REPRO_AC", "0") == "1"


def precompute_freqs_cis(dim: int, end: int, theta: float = ROPE_THETA) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


class Attention(nn.Module):
    """Llama-style GQA attention, trimmed to the SDPA path (from ezpz.llama).

    ``n_heads`` / ``n_kv_heads`` are divided in place by the TP degree before
    the module runs (matching ezpz.parallelize), so under TP the ``.view``
    below uses the LOCAL head counts and SDPA sees (B, N/TP, L, H).
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_heads = N_HEADS
        self.n_kv_heads = N_KV_HEADS
        self.head_dim = HEAD_DIM
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        n_rep = self.n_heads // self.n_kv_heads

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        keys = repeat_kv(xk, n_rep)
        values = repeat_kv(xv, n_rep)

        # (B, L, N, H) -> (B, N, L, H). transpose() yields NON-contiguous views;
        # this is the layout inductor's flash-backward stride guard rejects.
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        if FORCE_CONTIGUOUS:
            xq = xq.contiguous()
            xk = xk.contiguous()
            xv = xv.contiguous()

        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(out)


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(DIM, FFN_HIDDEN, bias=False)
        self.w2 = nn.Linear(FFN_HIDDEN, DIM, bias=False)
        self.w3 = nn.Linear(DIM, FFN_HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = Attention()
        self.feed_forward = FeedForward()
        self.attention_norm = RMSNorm(DIM)
        self.ffn_norm = RMSNorm(DIM)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        return h + self.feed_forward(self.ffn_norm(h))


def _device_type() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    raise SystemExit("This reproducer needs an XPU or CUDA device.")


def _set_budget() -> None:
    if ACT_MEM_BUDGET != 1.0:
        import torch._functorch.config as functorch_config

        functorch_config.activation_memory_budget = ACT_MEM_BUDGET
        # Under TP, functional collectives thread the process-group communicator
        # into the compiled graph as a FakeScriptObject (e.g. primals_2). The
        # min-cut partitioner cannot size it and bails out. The process group
        # holds no activation memory, so treating it as zero size is sound and
        # lets the partitioner run (which is what pushes the transposed q/k/v
        # across the fwd/bwd boundary into the flash-attention backward op).
        functorch_config.unsafe_treat_script_objects_as_zero_size = True


def _sync(device_type: str) -> None:
    if device_type == "xpu":
        torch.xpu.synchronize()
    elif device_type == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Single-process path (no torchrun). Smoke test only; does NOT reproduce.
# ---------------------------------------------------------------------------
def _run_single_process() -> None:
    device_type = _device_type()
    device = torch.device(device_type)
    torch.manual_seed(0)
    _set_budget()

    freqs_cis = precompute_freqs_cis(HEAD_DIM, SEQ_LEN).to(device)
    blocks = [TransformerBlock().to(device).to(torch.bfloat16) for _ in range(N_LAYERS)]
    compiled = [torch.compile(b, fullgraph=True) for b in blocks]

    x = torch.randn(
        BATCH, SEQ_LEN, DIM, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    print(
        f"[single-process] device={device} budget={ACT_MEM_BUDGET} "
        f"contiguous={FORCE_CONTIGUOUS} layers={N_LAYERS} "
        f"shape=(B={BATCH}, N={N_HEADS}, L={SEQ_LEN}, H={HEAD_DIM})",
        flush=True,
    )

    h = x
    for block in compiled:
        h = block(h, freqs_cis)
    h.sum().backward()
    _sync(device_type)
    print("OK: forward + backward completed without a stride assertion.", flush=True)


# ---------------------------------------------------------------------------
# Distributed path (torchrun): 2D TP + FSDP2, mirroring ezpz.parallelize().
# ---------------------------------------------------------------------------
def _run_distributed() -> None:
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Replicate, Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        PrepareModuleInput,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )

    device_type = _device_type()
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])

    if device_type == "xpu":
        torch.xpu.set_device(local_rank)
        backend = "xccl"
    else:
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    dist.init_process_group(backend=backend)

    tp = TP_SIZE
    assert world_size % tp == 0, f"WORLD_SIZE({world_size}) must be divisible by TP({tp})"
    dp = world_size // tp

    _set_budget()
    torch.manual_seed(0)

    device_mesh = init_device_mesh(
        device_type, (dp, tp), mesh_dim_names=("dp", "tp")
    )
    tp_mesh = device_mesh["tp"]
    dp_mesh = device_mesh["dp"]

    if rank == 0:
        print(
            f"[distributed] device={device_type} world_size={world_size} "
            f"dp={dp} tp={tp} budget={ACT_MEM_BUDGET} "
            f"contiguous={FORCE_CONTIGUOUS} ac={USE_AC} layers={N_LAYERS} "
            f"global=(B={BATCH}, N={N_HEADS}, L={SEQ_LEN}, H={HEAD_DIM}) "
            f"local=(B={BATCH}, N={N_HEADS // tp}, L={SEQ_LEN}, H={HEAD_DIM})",
            flush=True,
        )

    freqs_cis = precompute_freqs_cis(HEAD_DIM, SEQ_LEN).to(device_type)

    # Per-block TP plan, verbatim from ezpz.examples.fsdp_tp.parallelize.
    layer_tp_plan = {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layouts=(Shard(1), None),
            desired_input_layouts=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
        "attention.wk": ColwiseParallel(),
        "attention.wv": ColwiseParallel(),
        "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        "ffn_norm": SequenceParallel(),
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    }

    blocks = []
    for _ in range(N_LAYERS):
        block = TransformerBlock().to(torch.bfloat16)
        # Divide head counts by TP BEFORE parallelize (ezpz ordering), so the
        # attention .view uses local head counts and SDPA sees (B, N/TP, L, H).
        block.attention.n_heads //= tp
        block.attention.n_kv_heads //= tp
        parallelize_module(block, tp_mesh, layer_tp_plan)
        block.to(device_type)

        # Wrapper order matches torchtitan parallelize_llama: TP -> AC ->
        # compile -> FSDP, i.e. FSDP is the OUTERMOST wrapper. Compiling on top
        # of fully_shard makes dynamo trace the FSDP pre-forward hook (which is
        # torch._dynamo.disable'd) and fail under fullgraph=True.
        if USE_AC:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
            )

            # Full activation checkpointing: the forward is NOT saved but
            # RECOMPUTED in backward, re-running the transposed q/k/v views
            # straight into the flash-attention backward op.
            block = checkpoint_wrapper(block, preserve_rng_state=False)

        # Per-block compile with fullgraph=True (ezpz apply_compile pattern).
        block = torch.compile(block, fullgraph=True)

        # FSDP2 shard each block on the dp mesh, outermost (the original
        # failure ran under FSDP sharded params).
        fully_shard(block, mesh=dp_mesh)
        blocks.append(block)

    compiled = blocks

    # Input feeds the first block's SequenceParallel norm as a Shard(1) local
    # shard: global seqlen = local * tp. So each rank holds SEQ_LEN // tp
    # tokens; PrepareModuleInput all-gathers back to full SEQ_LEN for attention.
    local_seq = SEQ_LEN // tp
    x = torch.randn(
        BATCH, local_seq, DIM, device=device_type, dtype=torch.bfloat16,
        requires_grad=True,
    )

    h = x
    for block in compiled:
        h = block(h, freqs_cis)
    h.sum().backward()
    _sync(device_type)

    dist.barrier()
    if rank == 0:
        print(
            "OK: forward + backward completed without a stride assertion.",
            flush=True,
        )
    dist.destroy_process_group()


def main() -> None:
    if os.environ.get("WORLD_SIZE") and int(os.environ["WORLD_SIZE"]) > 1:
        _run_distributed()
    else:
        _run_single_process()


if __name__ == "__main__":
    main()
