"""Lightning indexer (DeepSeek-V4 CSA sparse selection).

Cheap relevance scores between every query token t and every *compressed* KV
entry s, used to pick the top-k entries each query actually attends to:

    I[t, s] = sum_h  w[t, h] * ReLU( q^I[t, h, :] . k^I[s, :] )

The indexer is tiny (few heads, small dim) and, in DeepSeek-V4, runs in FP4;
we optionally MX-fake-quantize its inputs to model that. ReLU keeps scores
one-sided so the additive head mixture cannot cancel.
"""

from __future__ import annotations

import torch


def indexer_scores_ref(qI: torch.Tensor, wI: torch.Tensor, kI: torch.Tensor) -> torch.Tensor:
    """qI: [B, T, H, D]; wI: [B, T, H]; kI: [B, S, D]  ->  scores [B, T, S] (fp32)."""
    dots = torch.einsum("bthd,bsd->bths", qI.float(), kI.float()).relu_()
    return torch.einsum("bths,bth->bts", dots, wI.float())


def _indexer_scores_triton(qI, wI, kI):
    import triton

    from ._indexer_kernel import _indexer_fwd

    B, T, H, D = qI.shape
    S = kI.shape[1]
    out = torch.empty(B, T, S, device=qI.device, dtype=torch.float32)
    BLOCK_T, BLOCK_S = 32, 64
    BLOCK_D = max(16, triton.next_power_of_2(D))  # tl.dot needs dims >= 16
    grid = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(S, BLOCK_S))
    _indexer_fwd[grid](
        qI.contiguous(), wI.contiguous(), kI.contiguous(), out,
        T, S, H, D,
        BLOCK_T=BLOCK_T, BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )
    return out


def indexer_scores(qI, wI, kI, fp4_sim: bool = False):
    from . import use_triton
    from .mxfp import mx_quant_dequant

    if fp4_sim:  # V4 computes the indexer in FP4; simulate with MX fake-quant
        qI, kI = mx_quant_dequant(qI, "fp4_e2m1"), mx_quant_dequant(kI, "fp4_e2m1")
    if use_triton(qI, wI, kI) and not torch.is_grad_enabled():
        return _indexer_scores_triton(qI, wI, kI)
    return indexer_scores_ref(qI, wI, kI)
