"""Synthetic data for the demos and tests.

1. Markov-chain "language": a fixed sparse order-1 transition table -- real
   learnable structure, so the training loss visibly falls within tens of
   steps at toy scale.
2. Needle-recall sequences for long-context sanity checks: KEY k ... noise ...
   QUERY k -> the model must copy the payload token bound to k. This is the
   micro version of the 1M-token retrieval evals in both tech reports.
"""

from __future__ import annotations

import torch

_SPECIAL = 4  # 0=pad, 1=KEY, 2=QUERY, 3=SEP
KEY_TOK, QUERY_TOK, SEP_TOK = 1, 2, 3


class MarkovData:
    def __init__(self, vocab_size: int, branching: int = 4, seed: int = 1234):
        g = torch.Generator().manual_seed(seed)
        self.vocab = vocab_size
        # each token can be followed by `branching` successors, one favored
        self.succ = torch.randint(_SPECIAL, vocab_size, (vocab_size, branching), generator=g)
        probs = torch.rand(vocab_size, branching, generator=g) + 0.1
        probs[:, 0] += 2.0  # skewed -> low entropy -> learnable
        self.probs = probs / probs.sum(-1, keepdim=True)

    def batch(self, batch_size: int, seq_len: int, device="cpu", generator=None):
        toks = torch.empty(batch_size, seq_len + 1, dtype=torch.long)
        toks[:, 0] = torch.randint(_SPECIAL, self.vocab, (batch_size,), generator=generator)
        for t in range(seq_len):
            cur = toks[:, t]
            choice = torch.multinomial(self.probs[cur], 1, generator=generator).squeeze(-1)
            toks[:, t + 1] = self.succ[cur, choice]
        x, y = toks[:, :-1], toks[:, 1:]
        return x.to(device), y.contiguous().to(device)


def needle_batch(batch_size: int, seq_len: int, vocab_size: int, device="cpu",
                 generator=None):
    """[ KEY key payload SEP noise ... noise QUERY key ] -> next token = payload."""
    assert seq_len >= 8
    x = torch.randint(_SPECIAL, vocab_size, (batch_size, seq_len), generator=generator)
    key = torch.randint(_SPECIAL, vocab_size, (batch_size,), generator=generator)
    payload = torch.randint(_SPECIAL, vocab_size, (batch_size,), generator=generator)
    pos = torch.randint(0, max(1, seq_len - 8), (batch_size,), generator=generator)
    for b in range(batch_size):
        p = int(pos[b])
        x[b, p], x[b, p + 1], x[b, p + 2], x[b, p + 3] = KEY_TOK, key[b], payload[b], SEP_TOK
    x[:, -2], x[:, -1] = QUERY_TOK, key
    return x.to(device), payload.to(device)
