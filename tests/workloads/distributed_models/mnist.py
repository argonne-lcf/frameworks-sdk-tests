#!/usr/bin/env python3
"""Short DDP CNN training workload; synthetic by default and network-free."""

from __future__ import annotations

import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from torch_setup import get_device, init_distributed


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 10),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--real-data", action="store_true", help="download and use MNIST")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()

    _, rank, world_size = init_distributed()
    device = get_device()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    if args.real_data:
        if rank == 0:
            torchvision.datasets.MNIST(
                root="./data", train=True, transform=transform, download=True
            )
        dist.barrier()
        dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transform, download=False
        )
    else:
        dataset = torchvision.datasets.FakeData(
            size=max(128, args.steps * args.batch_size * world_size),
            image_size=(1, 28, 28),
            num_classes=10,
            transform=transforms.ToTensor(),
            random_offset=17,
        )

    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=1234)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
    )
    model = DDP(SimpleCNN().to(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        loss_sum = torch.zeros((), device=device)
        completed = 0
        for step, (images, labels) in enumerate(loader):
            if step >= args.steps:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise AssertionError(f"rank {rank}: non-finite loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            completed += 1
        if completed == 0:
            raise AssertionError("no training steps completed")
        dist.all_reduce(loss_sum)
        if rank == 0:
            print(f"epoch={epoch} steps={completed} mean_loss={loss_sum.item() / (completed * world_size):.6f}")

    if rank == 0 and args.checkpoint:
        torch.save(model.module.state_dict(), args.checkpoint)
    dist.destroy_process_group()
    if rank == 0:
        print("PASS MNIST DDP workload")


if __name__ == "__main__":
    main()
