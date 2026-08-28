#!/usr/bin/env python3
"""Bounded ResNet-50 DDP/FSDP training workload with synthetic default data."""

from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import resnet50

from torch_setup import get_device, init_distributed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--real-data", action="store_true", help="download and use CIFAR-10")
    args = parser.parse_args()

    _, rank, world_size = init_distributed()
    device = get_device()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    if args.real_data:
        if rank == 0:
            torchvision.datasets.CIFAR10("./data", train=True, download=True)
        dist.barrier()
        dataset = torchvision.datasets.CIFAR10(
            "./data", train=True, transform=transform, download=False
        )
    else:
        dataset = torchvision.datasets.FakeData(
            size=max(64, args.steps * args.batch_size * world_size),
            image_size=(3, 224, 224),
            num_classes=10,
            transform=transforms.ToTensor(),
            random_offset=29,
        )

    sampler = DistributedSampler(dataset, world_size, rank, seed=1234)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    base_model = resnet50(weights=None, num_classes=10).to(device)
    model = FSDP(base_model) if args.fsdp else DDP(base_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        started = time.perf_counter()
        total_loss = torch.zeros((), device=device)
        completed = 0
        for step, (images, labels) in enumerate(loader):
            if step >= args.steps:  # the source test accidentally used continue here
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise AssertionError(f"rank {rank}: non-finite loss")
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            completed += 1
        if completed == 0:
            raise AssertionError("no training steps completed")
        dist.all_reduce(total_loss)
        if rank == 0:
            elapsed = time.perf_counter() - started
            rate = completed * args.batch_size * world_size / elapsed
            print(
                f"epoch={epoch} steps={completed} "
                f"mean_loss={total_loss.item() / (completed * world_size):.6f} "
                f"throughput={rate:.2f} images/s"
            )
    dist.destroy_process_group()
    if rank == 0:
        print(f"PASS ResNet-50 {'FSDP' if args.fsdp else 'DDP'} workload")


if __name__ == "__main__":
    main()
