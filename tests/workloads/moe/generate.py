#!/usr/bin/env python
"""Inference demo: tiny caches + MTP self-speculative decoding.

    python generate.py --arch k3 --prompt-len 2048 --gen 64
    python generate.py --arch v4 --prompt-len 2048 --gen 64 --speculative
    python generate.py --ckpt model.pt --gen 64 --speculative

Shows the long-context machinery end to end:
  * chunked prefill through the per-layer caches
      - KDA layers:  O(1) state (dk x dv per head) regardless of context
      - MLA layers:  latent-only KV cache (kv_latent_dim per token)
      - CSA/HCA:     compressed entries (1 per block) + fixed raw tail
  * decode with the incremental caches
  * MTP head as a self-speculative drafter (propose t+2, verify next step)
  * reports cache bytes vs a dense-KV-cache baseline
"""

from __future__ import annotations

import argparse
import time

import torch

from latentmoe import LatentMoEModel, ModelConfig, get_preset
from latentmoe.accel import best_device, manual_seed_all, synchronize
from latentmoe.data import needle_batch


def cache_bytes(caches) -> int:
    total = 0
    for c in caches:
        for v in c.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def dense_kv_baseline(cfg: ModelConfig, seq: int, dtype_bytes: int = 2) -> int:
    return cfg.n_layers * seq * 2 * cfg.n_heads * cfg.head_dim * dtype_bytes


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["v4", "k3", "hybrid"], default="k3")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--prompt-len", type=int, default=2048)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--prefill-chunk", type=int, default=512)
    p.add_argument("--speculative", action="store_true", help="MTP self-drafting")
    p.add_argument("--exact-spec", action="store_true",
                   help="exact speculative decoding: snapshot caches before "
                        "each draft and rewind+replay on rejection, so output "
                        "is IDENTICAL to plain greedy (costs one extra "
                        "1-token forward per rejection)")
    p.add_argument("--markov", action="store_true",
                   help="prompt from the training distribution (MarkovData) "
                        "instead of a random needle sequence -- use with --ckpt "
                        "to see meaningful MTP acceptance rates")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    manual_seed_all(0)
    device = torch.device(args.device) if args.device else best_device()

    if args.ckpt:
        blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**blob["cfg"])
        model = LatentMoEModel(cfg)
        model.load_state_dict(blob["model"])
    else:
        cfg = get_preset(args.arch, vocab_size=4096, dim=256, n_layers=8,
                         n_heads=4, head_dim=64, n_routed_experts=16, top_k=4,
                         moe_latent_dim=128, expert_hidden=128,
                         shared_expert_hidden=256, dense_ffn_hidden=512,
                         kv_latent_dim=64, q_latent_dim=128,
                         max_seq_len=args.prompt_len + args.gen + 8)
        model = LatentMoEModel(cfg)
    model.to(device).eval()

    if args.markov:
        from latentmoe.data import MarkovData
        prompt, _ = MarkovData(cfg.vocab_size, seed=1234).batch(1, args.prompt_len, device)
        payload = torch.tensor([-1])
        print(f"arch={args.arch} device={device} prompt={args.prompt_len} Markov tokens")
    else:
        prompt, payload = needle_batch(1, args.prompt_len, cfg.vocab_size, device=device)
        print(f"arch={args.arch} device={device} prompt={args.prompt_len} tokens "
              f"(needle payload token: {int(payload[0])})")

    # ---------------- chunked prefill ----------------
    caches = model.new_caches()
    t0 = time.perf_counter()
    h_last = None
    out = None
    for s in range(0, prompt.shape[1], args.prefill_chunk):
        chunk = prompt[:, s: s + args.prefill_chunk]
        out = model(chunk, caches=caches, return_logits=True)
    synchronize(device)
    t_prefill = time.perf_counter() - t0

    cb = cache_bytes(caches)
    dense = dense_kv_baseline(cfg, prompt.shape[1])
    print(f"prefill: {t_prefill:.2f}s ({prompt.shape[1]/max(t_prefill,1e-9):.0f} tok/s) "
          f"| cache {cb/1e6:.2f} MB vs dense-KV "
          f"baseline {dense/1e6:.2f} MB ({100*cb/max(dense,1):.1f}%)")

    # ---------------- decode ----------------
    tokens = []
    last_logits = out["logits"][:, -1]
    draft = None
    accepted = 0
    proposed = 0
    t0 = time.perf_counter()
    step_inputs = last_logits.argmax(-1, keepdim=True)  # first generated token
    tokens.append(int(step_inputs[0, 0]))

    def mtp_draft(o, nxt):
        if model.mtp is None:
            return None
        feats = [f[:, -1:] for f in o["mtp_feats"]] if cfg.mtp_fuse_layers else None
        h = o["hidden_prenorm"][:, -1:]
        d_hidden = model.mtp(h, model.embed(nxt), feats=feats)
        return model.head(d_hidden)[:, -1].argmax(-1, keepdim=True)

    if args.speculative:
        draft = mtp_draft(out, step_inputs)

    while len(tokens) < args.gen:
        if args.speculative and draft is not None:
            proposed += 1
            snap = model.fork_caches(caches) if args.exact_spec else None
            pair = torch.cat([step_inputs, draft], dim=1)      # verify draft
            out = model(pair, caches=caches, return_logits=True)
            verify = out["logits"][:, 0].argmax(-1, keepdim=True)
            if int(verify[0, 0]) == int(draft[0, 0]):          # accepted: +2 tokens
                accepted += 1
                nxt = out["logits"][:, 1].argmax(-1, keepdim=True)
                tokens.append(int(draft[0, 0]))
                if len(tokens) < args.gen:
                    tokens.append(int(nxt[0, 0]))
                step_inputs = nxt
                draft = mtp_draft(out, nxt)
                continue
            # rejected: roll forward from the verified token
            if args.exact_spec:
                # rewind to the pre-draft caches and replay only the verified-
                # prefix token: cache is exactly as if the draft never
                # happened (one extra 1-token forward is the rejection price)
                for c, s in zip(caches, snap):
                    c.clear()
                    c.update(s)
                out = model(step_inputs, caches=caches, return_logits=True)
            # else: demo approximation -- the rejected draft token stays in
            # the cache (cheaper, slightly off-distribution)
            tokens.append(int(verify[0, 0]))
            step_inputs = verify
            draft = mtp_draft(out, verify)
        else:
            out = model(step_inputs, caches=caches, return_logits=True)
            step_inputs = out["logits"][:, -1].argmax(-1, keepdim=True)
            tokens.append(int(step_inputs[0, 0]))
            if args.speculative:
                draft = mtp_draft(out, step_inputs)
    synchronize(device)
    t_decode = time.perf_counter() - t0

    print(f"decode: {len(tokens)} tokens in {t_decode:.2f}s "
          f"({len(tokens)/max(t_decode,1e-9):.1f} tok/s)")
    if args.speculative and proposed:
        mode = "exact (rewind+replay)" if args.exact_spec else "approximate"
        print(f"MTP speculative acceptance: {accepted}/{proposed} "
              f"({100*accepted/proposed:.0f}%) [{mode}]")
    if not args.markov:
        print(f"first generated token: {tokens[0]} "
              f"(needle payload was {int(payload[0])} -- matches only after training)")
    print("generated:", tokens)


if __name__ == "__main__":
    main()
