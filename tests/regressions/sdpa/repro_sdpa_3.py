"""
Minimal reproducer: assert_size_stride crash in
_scaled_dot_product_flash_attention_backward under torch.compile, when
grouped-query attention's repeat_kv() (expand().reshape()) feeds K/V.

No FSDP, no TP, no torch.distributed, no third-party packages -- plain
torch only.

Confirmed on hardware:
- N_KV_HEADS = N_HEADS (no GQA, repeat_kv is a no-op)      -> passes
- N_KV_HEADS < N_HEADS (GQA, repeat_kv actually runs)       -> crashes

Set DEVICE = "cuda" below to check whether this reproduces off-XPU too.

Run: python3 repro_minimal_upstream.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "xpu"
DTYPE = torch.bfloat16

BSZ, SEQLEN, N_HEADS, HEAD_DIM = 4, 4096, 4, 128
N_KV_HEADS = 1  # set equal to N_HEADS to see this pass instead
N_REP = N_HEADS // N_KV_HEADS
DIM = N_HEADS * HEAD_DIM
KV_DIM = N_KV_HEADS * HEAD_DIM


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep), expand+reshape form.

    Standard GQA helper (same as PyTorch's TP tutorial / torchtitan's
    Llama implementation).
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, KV_DIM, bias=False)
        self.wv = nn.Linear(DIM, KV_DIM, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        xq = self.wq(x).view(b, s, N_HEADS, HEAD_DIM)
        xk = self.wk(x).view(b, s, N_KV_HEADS, HEAD_DIM)
        xv = self.wv(x).view(b, s, N_KV_HEADS, HEAD_DIM)

        keys = repeat_kv(xk, N_REP)
        values = repeat_kv(xv, N_REP)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim), non-contiguous
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return out


def main():
    print(f"torch: {torch.__version__}, device: {DEVICE}")
    torch.manual_seed(0)

    model = Attention().to(device=DEVICE, dtype=DTYPE)
    model_c = torch.compile(model, mode="max-autotune-no-cudagraphs", fullgraph=True)

    x = torch.randn(BSZ, SEQLEN, DIM, device=DEVICE, dtype=DTYPE, requires_grad=True)

    out = model_c(x)
    out.sum().backward()
    torch.xpu.synchronize()
    print(f"OK - no crash (N_HEADS={N_HEADS}, N_KV_HEADS={N_KV_HEADS})")


if __name__ == "__main__":
    main()
