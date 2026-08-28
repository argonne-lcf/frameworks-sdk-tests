"""
Reproducer for the assert_size_stride crash in
_scaled_dot_product_flash_attention_backward under torch.compile.

Baseline (default): no FSDP, no TP, no torch.distributed collectives that
matter -- linear -> view -> transpose (non-contiguous, no .contiguous())
-> SDPA(is_causal=True) -> backward. This variant did NOT reproduce the
crash on real hardware.

USE_FSDP=1: wraps the module with fully_shard() BEFORE compiling it,
mirroring the real script's fully_shard(block) -> torch.compile(block,
fullgraph=True) order per TransformerBlock. This is the current lead --
FSDP2 frees/reallocates the all-gathered parameter buffer around each
forward when reshard_after_forward=True (the default), which is a known
category of bug when combined with a compiled backward.

Shapes match the crashing buffer from the real run: (4, 4, 4096, 128).

Run (baseline, single process is fine):
    python3 repro_sdpa_compile_bug.py

Run (FSDP2 test -- needs your normal distributed launcher, e.g.):
    USE_FSDP=1 ezpz launch --nhosts=1 -- python3 repro_sdpa_compile_bug.py
    USE_FSDP=1 RESHARD=0 ezpz launch --nhosts=1 -- python3 repro_sdpa_compile_bug.py

Run (GQA test -- exercises repeat_kv's expand().reshape() path, n_rep=4,
matching agpt-2b's local kv-head count at --tp=4):
    GQA=1 ezpz launch --nhosts=1 -- python3 repro_sdpa_compile_bug.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import ezpz
import ezpz.distributed
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

device = "xpu"
dtype = torch.bfloat16

# CONTIGUOUS_QKV=1  -> force xq/xk/xv contiguous before SDPA (earlier test)
# USE_FSDP=1        -> wrap the module with fully_shard BEFORE compiling it,
#                      mirroring the real script's fully_shard(block) then
#                      torch.compile(block, fullgraph=True) order.
# RESHARD=0         -> pass reshard_after_forward=False (only used if USE_FSDP=1)
# GQA=1             -> use n_kv_heads=1 (matches agpt-2b's local kv-head count
#                      at --tp=4: n_kv_heads=4 global / tp=4 = 1 local), so
#                      repeat_kv's expand().reshape() path actually runs
#                      (n_rep=4) instead of the n_rep==1 no-op. Not tested
#                      before this run.
CONTIGUOUS_QKV = os.environ.get("CONTIGUOUS_QKV", "0") == "1"
USE_FSDP = os.environ.get("USE_FSDP", "0") == "1"
RESHARD = os.environ.get("RESHARD", "1") == "1"
GQA = os.environ.get("GQA", "0") == "1"

bsz, seqlen, n_heads, head_dim = 4, 4096, 4, 128
n_kv_heads = 1 if GQA else n_heads
n_rep = n_heads // n_kv_heads
dim = n_heads * head_dim
kv_dim = n_kv_heads * head_dim


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Verbatim from ezpz/models/llama.py."""
    bs, slen, nkv, hd = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, nkv, n_rep, hd)
        .reshape(bs, slen, nkv * n_rep, hd)
    )


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, kv_dim, bias=False)
        self.wv = nn.Linear(dim, kv_dim, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        xq = self.wq(x).view(b, s, n_heads, head_dim)
        xk = self.wk(x).view(b, s, n_kv_heads, head_dim)
        xv = self.wv(x).view(b, s, n_kv_heads, head_dim)

        keys = repeat_kv(xk, n_rep)
        values = repeat_kv(xv, n_rep)

        # NOT contiguous after this -- this is the pattern in ezpz's llama.py
        xq = xq.transpose(1, 2)
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        if CONTIGUOUS_QKV:
            xq = xq.contiguous()
            xk = xk.contiguous()
            xv = xv.contiguous()

        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return out


def main():
    # Reuse ezpz's own distributed setup (same call the real script makes)
    # rather than hand-rolling a new dist-init path for XPU/oneCCL.
    rank = ezpz.distributed.setup_torch(tensor_parallel_size=1, seed=0)
    torch.manual_seed(0)

    model = Attn().to(device=device, dtype=dtype)

    if USE_FSDP:
        fsdp_kwargs = {
            "reshard_after_forward": RESHARD,
            "mp_policy": MixedPrecisionPolicy(param_dtype=dtype),
        }
        fully_shard(model, **fsdp_kwargs)

    # Matches the real script: fully_shard first, THEN compile the module.
    model_c = torch.compile(model, mode="max-autotune-no-cudagraphs", fullgraph=True)

    x = torch.randn(bsz, seqlen, dim, device=device, dtype=dtype, requires_grad=True)

    out = model_c(x)
    out.sum().backward()
    if rank == 0:
        print(
            f"OK - no crash (CONTIGUOUS_QKV={CONTIGUOUS_QKV}, "
            f"USE_FSDP={USE_FSDP}, RESHARD={RESHARD}, GQA={GQA})"
        )


if __name__ == "__main__":
    main()
