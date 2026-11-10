#!/usr/bin/env python3
"""FSDP2/compile reproducer for autograd interpreter-shutdown SIGABRT.

Tracks argonne-lcf/AuroraBugTracking#182. With default CPU binding and
FI_MR_CACHE_MONITOR=disabled, an autograd callback-lifetime race can abort
Py_FinalizeEx after the workload completes.

Every rank builds a tiny fully-sharded model, runs one compiled forward and
backward, synchronizes, and destroys the process group. The workload itself
must succeed on every rank: each prints "work done <rank>" and the process
must then exit zero. A signal or nonzero exit status during shutdown is a
reproduction.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.nn import functional as F

if not torch.xpu.is_available():
    raise RuntimeError("FSDP shutdown regression requires an XPU")

# PALS (mpiexec) exports PALS_RANKID/PALS_LOCAL_RANKID, while torchrun exports
# RANK/LOCAL_RANK. The fallbacks only matter for direct single-process runs.
# The distributed env vars must be finalized before init_process_group.
os.environ["RANK"] = os.environ.get("PALS_RANKID") or os.environ.get("RANK") or "0"

# Defaults for direct single-process debugging; the multi-rank launchers
# export all three before this point.
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29513")

local_rank = os.environ.get("PALS_LOCAL_RANKID") or os.environ.get("LOCAL_RANK") or "0"

torch.xpu.set_device(int(local_rank))
dist.init_process_group("xccl")


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.RMSNorm(256, eps=1e-5)
        self.up = nn.Linear(256, 512, bias=False)
        self.down = nn.Linear(512, 256, bias=False)

    def forward(self, value):
        return self.down(F.silu(self.up(self.norm(value))))


model = Model().to("xpu")
model.compile(fullgraph=True)
fully_shard(
    model,
    mp_policy=MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    ),
)
value = torch.randn(1, 128, 256, device="xpu", dtype=torch.bfloat16)
model(value).float().square().mean().backward()
torch.xpu.synchronize()
dist.destroy_process_group()
print("work done", os.environ["RANK"], flush=True)
