"""
Minimal reproducer for the assert_size_stride crash in
_scaled_dot_product_flash_attention_backward under torch.compile.

No FSDP, no TP, no torch.distributed. Just: linear -> view -> transpose
(non-contiguous, no .contiguous()) -> SDPA(is_causal=True) -> backward.
Shapes match the crashing buffer from the real run: (4, 4, 4096, 128).

Run: python3 repro_sdpa_compile_bug.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "xpu"
dtype = torch.bfloat16

bsz, seqlen, n_heads, head_dim = 4, 4096, 4, 128
dim = n_heads * head_dim


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        xq = self.wq(x).view(b, s, n_heads, head_dim)
        xk = self.wk(x).view(b, s, n_heads, head_dim)
        xv = self.wv(x).view(b, s, n_heads, head_dim)

        # NOT contiguous after this -- this is the pattern in ezpz's llama.py
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return out


def main():
    torch.manual_seed(0)
    model = Attn().to(device=device, dtype=dtype)
    model_c = torch.compile(model, mode="max-autotune-no-cudagraphs")

    x = torch.randn(bsz, seqlen, dim, device=device, dtype=dtype, requires_grad=True)

    out = model_c(x)
    out.sum().backward()
    torch.xpu.synchronize()
    print("OK - no crash")


if __name__ == "__main__":
    main()
