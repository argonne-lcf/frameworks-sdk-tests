#!/usr/bin/env python3
"""Single-process interpreter-shutdown SIGABRT reproducer.

Reduced in argonne-lcf/AuroraBugTracking#182 from the FSDP shutdown
reproducer: no FSDP, no compilation, and no distributed communication. An
autograd worker thread releases queued Python callback references while the
main thread is inside Py_FinalizeEx; the process aborts with SIGABRT after
the "work done" line.

The workload itself must succeed, print "work done", and the process must
then exit zero. Any signal or nonzero exit status is a reproduction. Run
with single-CPU affinity (run_repro.sh applies taskset), which made the
failure reproducible in the upstream report.
"""

from __future__ import annotations

import os

import torch
from torch.utils.module_tracker import ModuleTracker

if not torch.xpu.is_available():
    raise RuntimeError("ModuleTracker shutdown regression requires an XPU")

torch.xpu.set_device(int(os.environ.get("PALS_LOCAL_RANKID", "0")))
model = torch.nn.Linear(1, 1, device="xpu")
value = torch.ones(1, 1, device="xpu")
with ModuleTracker():
    model(value).sum().backward()
torch.xpu.synchronize()
print("work done", os.environ.get("PALS_RANKID", "0"), flush=True)
