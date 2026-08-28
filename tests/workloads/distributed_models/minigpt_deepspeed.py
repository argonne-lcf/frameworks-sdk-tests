#!/usr/bin/env python3
"""Small synthetic GPT-2/DeepSpeed training acceptance workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import GPT2Config, GPT2LMHeadModel

from torch_setup import init_distributed


class SyntheticTextDataset(Dataset):
    def __init__(self, length: int, sequence_length: int, vocabulary_size: int) -> None:
        self.length = length
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        generator = torch.Generator().manual_seed(index)
        tokens = torch.randint(
            self.vocabulary_size,
            (self.sequence_length,),
            dtype=torch.long,
            generator=generator,
        )
        return {"input_ids": tokens, "labels": tokens.clone()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("ds_config.json"))
    args = parser.parse_args()
    if min(
        args.layers,
        args.hidden_size,
        args.heads,
        args.sequence_length,
        args.samples,
        args.micro_batch_size,
        args.epochs,
    ) < 1:
        raise ValueError("all size/count arguments must be positive")
    if args.hidden_size % args.heads:
        raise ValueError("--hidden-size must be divisible by --heads")

    _, rank, world_size = init_distributed()
    vocabulary_size = 4096
    dataset = SyntheticTextDataset(args.samples, args.sequence_length, vocabulary_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=1234,
    )
    loader = DataLoader(
        dataset, batch_size=args.micro_batch_size, sampler=sampler
    )
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=vocabulary_size,
            n_positions=args.sequence_length,
            n_ctx=args.sequence_length,
            n_embd=args.hidden_size,
            n_layer=args.layers,
            n_head=args.heads,
            bos_token_id=0,
            eos_token_id=1,
        )
    )
    config = json.loads(args.config.read_text())
    config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        config_params=config,
        model_parameters=model.parameters(),
    )
    engine.train()
    tracked_parameter = next(engine.module.parameters())
    initial_parameter = tracked_parameter.detach().clone()
    completed = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            outputs = engine(
                batch["input_ids"].to(engine.device),
                labels=batch["labels"].to(engine.device),
            )
            if not torch.isfinite(outputs.loss):
                raise AssertionError(f"rank {rank}: non-finite loss")
            engine.backward(outputs.loss)
            engine.step()
            completed += 1
        if rank == 0:
            print(f"epoch={epoch} final_loss={outputs.loss.item():.6f}")
    if completed == 0:
        raise AssertionError("no DeepSpeed training steps completed")
    update_norm = (tracked_parameter.detach() - initial_parameter).float().norm()
    if not torch.isfinite(update_norm) or update_norm.item() == 0.0:
        raise AssertionError(f"rank {rank}: model parameter did not update cleanly")

    checksum = tracked_parameter.detach().float().sum()
    checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
    dist.all_gather(checksums, checksum)
    for other_rank, other in enumerate(checksums):
        if not torch.allclose(other, checksums[0], rtol=1e-5, atol=1e-5):
            raise AssertionError(
                f"rank {other_rank} parameter checksum diverged: "
                f"{other.item()} != {checksums[0].item()}"
            )
    if rank == 0:
        print(
            f"PASS DeepSpeed miniGPT ranks={world_size} steps={completed} "
            f"update_norm={update_norm.item():.6g}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
