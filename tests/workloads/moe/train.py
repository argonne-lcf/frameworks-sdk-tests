#!/usr/bin/env python
"""Train a LatentMoE model (DeepSeek-V4 / Kimi-K3 / hybrid, selectable at runtime).

Single device:
    python train.py --arch v4     --steps 100
    python train.py --arch k3     --steps 100
    python train.py --arch hybrid --steps 100 --qat

Expert parallelism (DeepEP-style dispatch across ranks; NCCL on NVIDIA,
XCCL/oneCCL on Intel, Gloo on CPU):
    torchrun --nproc-per-node 4 train.py --arch k3 --ep --steps 100

What each step exercises:
  * MuonClip optimizer (Muon w/ hybrid Newton-Schulz + per-head partition,
    AdamW for tagged params) and QK-clip on the tracked attention logits
  * aux-loss-free load balancing: bias updates ("bias") happen after each
    optimizer step; Quantile Balancing ("quantile") happens inside forward
  * MTP loss, sequence-wise balance loss
  * optionally MXFP4/MXFP8 QAT on the routed experts (--qat)
"""

from __future__ import annotations

import argparse
import time

import torch

from latentmoe import LatentMoEModel, get_preset
from latentmoe.accel import autocast_dtype, best_device, manual_seed_all, synchronize
from latentmoe.data import MarkovData
from latentmoe.optim import MuonClip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["v4", "k3", "hybrid"], default="k3")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--experts", type=int, default=16)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--qat", action="store_true", help="MXFP4/MXFP8 QAT on experts")
    p.add_argument("--ep", action="store_true", help="expert parallelism (torchrun)")
    p.add_argument("--no-bucket-grads", action="store_true",
                   help="EP grad sync: per-parameter all-reduce loop instead "
                        "of one flat bucket per dtype (A/B baseline)")
    p.add_argument("--global-balancer", action="store_true",
                   help="aggregate balancer statistics across EP ranks "
                        "(paper behavior) instead of per-rank")
    p.add_argument("--no-batch-ns", action="store_true",
                   help="Muon: per-parameter Newton-Schulz instead of one "
                        "batched NS per distinct shape (A/B baseline)")
    p.add_argument("--qk-clip-tau", type=float, default=100.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save", type=str, default="")
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    if min(
        args.steps,
        args.batch_size,
        args.seq_len,
        args.vocab,
        args.dim,
        args.layers,
        args.experts,
        args.top_k,
    ) < 1:
        raise ValueError("all size/count arguments must be positive")
    if args.top_k > args.experts:
        raise ValueError("--top-k cannot exceed --experts")
    manual_seed_all(0)

    rank, world = 0, 1
    device = torch.device(args.device) if args.device else best_device()
    buffer = None
    if args.ep:
        from latentmoe.parallel import Buffer, init_distributed
        rank, world, device = init_distributed(device)
        assert world > 1, (
            "--ep was requested but the launcher environment resolved to "
            "world_size=1, so no process group was created. Your launcher's "
            "rank/size env vars were not recognized. Diagnose with "
            "LATENTMOE_DIST_DEBUG=1 (prints what each rank sees), or set the "
            "standard vars explicitly, e.g.:\n"
            "  mpiexec ... bash -c 'export RANK=$PALS_RANKID "
            "LOCAL_RANK=$PALS_LOCAL_RANKID WORLD_SIZE=<n>; exec python train.py ...'")

    cfg = get_preset(
        args.arch,
        vocab_size=args.vocab, dim=args.dim, n_layers=args.layers,
        n_heads=max(4, args.dim // 64), head_dim=64,
        n_routed_experts=args.experts, top_k=args.top_k,
        moe_latent_dim=(args.dim // 2 if args.arch in ("k3", "hybrid") else None),
        expert_hidden=args.dim // 2, shared_expert_hidden=args.dim,
        dense_ffn_hidden=2 * args.dim, kv_latent_dim=args.dim // 4,
        q_latent_dim=args.dim // 2, max_seq_len=args.seq_len,
        qat_experts=args.qat,
    )
    model = LatentMoEModel(cfg).to(device)
    if args.ep:
        # identical init everywhere, then shard expert COMPUTE via dispatch
        import torch.distributed as dist
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        buffer = Buffer(cfg.n_routed_experts)
        model.set_dispatcher(buffer)
        if args.global_balancer:
            for m in model.moe_modules():
                m.stats_group = dist.group.WORLD

    n_params = sum(p.numel() for p in model.parameters())
    n_active = n_params - sum(
        m.w1.numel() + m.w2.numel() for m in model.moe_modules()
    ) + sum(
        (m.w1.numel() + m.w2.numel()) * m.k // m.e for m in model.moe_modules()
    )
    if rank == 0:
        print(f"arch={args.arch} device={device} world={world} "
              f"params={n_params/1e6:.1f}M (~{n_active/1e6:.1f}M active) "
              f"attn_plan={cfg.attn_plan()} residual={cfg.residual} "
              f"router={cfg.router_score}/{cfg.balancer} qat={cfg.qat_experts}")

    opt = MuonClip.from_model(model, lr=args.lr, ns_batched=not args.no_batch_ns)
    data = MarkovData(cfg.vocab_size, seed=1234)
    gen = torch.Generator().manual_seed(1000 + rank)
    amp_dtype = autocast_dtype(device)
    use_amp = device.type in ("cuda", "xpu") and amp_dtype == torch.bfloat16

    model.train()
    tracked_parameter = next(model.parameters())
    initial_parameter = tracked_parameter.detach().clone()
    # throughput accounting: every rank processes its OWN batch under EP
    tokens_per_step = args.batch_size * args.seq_len * world
    t0 = time.perf_counter()
    t_log, last_log_step = t0, 0
    for step in range(1, args.steps + 1):
        x, y = data.batch(args.batch_size, args.seq_len, device, generator=gen)
        model.set_track_qk(step % 10 == 1)  # sample max-logits periodically

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                out = model(x, targets=y, return_logits=False)
        else:
            out = model(x, targets=y, return_logits=False)
        loss = out["loss"]
        if not bool(torch.isfinite(loss).all().item()):
            raise FloatingPointError(f"non-finite training loss at step {step}: {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients:
            raise AssertionError(f"no gradients were produced at step {step}")
        gradients_finite = torch.stack(
            [torch.isfinite(gradient).all() for gradient in gradients]
        ).all()
        if not bool(gradients_finite.item()):
            raise FloatingPointError(f"non-finite gradient at step {step}")

        if args.ep:
            from latentmoe.parallel import ep_grad_sync
            ep_grad_sync(model, bucketed=not args.no_bucket_grads)

        opt.step()
        model.balance_step()                      # V3/V4 bias balancer update
        if step % 10 == 1:
            opt.qk_clip(model, tau=args.qk_clip_tau)   # MuonClip

        if rank == 0 and (step % args.log_every == 0 or step == 1):
            load = out.get("expert_load")
            imbalance = float(load.max() / load.mean().clamp_min(1e-9)) if load is not None else 0.0
            now = time.perf_counter()
            dt = (now - t_log) / max(1, step - last_log_step)  # windowed, not cumavg
            t_log, last_log_step = now, step
            tok_s = tokens_per_step / dt
            fl = 6 * n_active * tok_s / world  # ~train FLOPs (6*N_active/tok)
            fls = (f"{fl / 1e12:.2f} TF" if fl >= 1e11 else f"{fl / 1e9:.1f} GF")
            msg = (f"step {step:4d} | loss {float(out['ce_loss'].detach()):.4f}"
                   f" | mtp {float(out.get('mtp_loss', torch.zeros(())).detach()):.4f}"
                   f" | max/mean expert load {imbalance:.2f}"
                   f" | {dt:.3f}s/step | {tok_s / 1e3:.1f}k tok/s"
                   f" ({tok_s / world / 1e3:.2f}k/gpu, ~{fls}/gpu)")
            print(msg, flush=True)

    synchronize(device)
    update_norm = (tracked_parameter.detach() - initial_parameter).float().norm()
    if not torch.isfinite(update_norm) or update_norm.item() == 0.0:
        raise AssertionError("training did not produce a finite parameter update")
    if rank == 0:
        print(
            f"done in {time.perf_counter() - t0:.1f}s "
            f"update_norm={update_norm.item():.6g}"
        )
        if args.save:
            torch.save({"cfg": cfg.to_dict(), "model": model.state_dict()}, args.save)
            print(f"saved to {args.save}")


if __name__ == "__main__":
    main()
