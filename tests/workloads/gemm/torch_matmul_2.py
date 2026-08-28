# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# --- Adapted from intel/torch-xpu-ops test/microbench/matmul.py ---
# Changes vs. upstream (flagged, not silent):
#   1. BUG FIX: upstream allocates m1/m2 via torch.rand(...).to(device) INSIDE
#      matmul(), i.e. inside the timed loop -> pays a host->device copy every
#      iteration (huge for our large-N shape). Allocation now happens once,
#      directly on-device, outside the timed region.
#   2. backward defaults to False, to match torch.mm(A, B, out=C) (forward-only,
#      pre-allocated output) instead of upstream's default backward=True.
#   3. shape_list replaced with our two shapes; explicit (m, k, n) tuples
#      instead of upstream's (m, k, n)->matmul(shape[0], shape[2], shape[1])
#      index juggling.
#   4. Added FLOP/s calculation: 2*m*n*k, same convention as the oneDNN
#      matmul_perf.cpp example, so numbers are directly comparable.
#   5. Added a device/tile identity printout (relevant to the open
#      single-tile-vs-multi-tile question from earlier).

import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20

# (m, k, n) -> A:(m,k) @ B:(k,n) = C:(m,n), matching torch.mm(A, B, out=C)
shape_list = [
    (2048, 2048, 65536),    # "small"
    (2048, 2048, 262144),   # "large"
]


def run_matmul(m1, m2, out, backward):
    if backward:
        # out= is not autograd-differentiable, so this branch can't reuse `out`
        output = torch.mm(m1, m2)
        gy = torch.empty_like(output)
        output.backward(gy)
    else:
        torch.mm(m1, m2, out=out)


if __name__ == "__main__":
    backward = False  # matches torch.mm(A, B, out=C); flip to True to include
                       # a backward pass (note: FLOP count below is forward-only)

    print(f"device: {torch.xpu.get_device_name(0)}")
    print(f"device_count: {torch.xpu.device_count()}")
    print()

    for m, k, n in shape_list:
        flops = 2 * m * n * k  # C[m,n] = A[m,k] @ B[k,n]; forward-only count
        for dtype in [torch.bfloat16,]:
            m1 = torch.rand(m, k, dtype=dtype, device=device)
            m2 = torch.rand(k, n, dtype=dtype, device=device)
            out = torch.empty(m, n, dtype=dtype, device=device)
            if backward:
                m1.requires_grad_(True)
                m2.requires_grad_(True)

            # warm up
            run_matmul(m1, m2, out, backward)
            torch.xpu.synchronize()

            print(f"shape: m={m}, k={k}, n={n} ; datatype: {dtype} ; backward: {backward}")
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(num_iter):
                    run_matmul(m1, m2, out, backward)
            print(prof.key_averages().table(sort_by="xpu_time_total"))

            # --- Method A: our original batched-sync mean (one synchronize
            # around the whole loop -> lets the queue pipeline back-to-back
            # launches; reports the mean of num_iter calls) ---
            torch.xpu.synchronize()
            t1 = time.time()
            for i in range(num_iter):
                run_matmul(m1, m2, out, backward)
            torch.xpu.synchronize()
            t2 = time.time()
            e2e_time = (t2 - t1) / num_iter
            tflops_mean_batched = flops / e2e_time / 1e12
            print(f"[batched-sync mean]  E2E: {e2e_time*1000:.4f} ms  ->  {tflops_mean_batched:.4f} TFLOP/s")

            # --- Method B: EleutherAI-style per-call sync (Event timing,
            # synchronize after every single call -> no pipelining; reports
            # both min (matches their np.amin) and mean of the SAME samples,
            # same process/tensors as Method A above) ---
            start_evt = torch.xpu.Event(enable_timing=True)
            end_evt = torch.xpu.Event(enable_timing=True)
            per_call_ms = []
            for i in range(num_iter):
                start_evt.record()
                run_matmul(m1, m2, out, backward)
                end_evt.record()
                torch.xpu.synchronize()
                per_call_ms.append(start_evt.elapsed_time(end_evt))

            min_time = min(per_call_ms) / 1000
            mean_time = (sum(per_call_ms) / len(per_call_ms)) / 1000
            tflops_min_percall = flops / min_time / 1e12
            tflops_mean_percall = flops / mean_time / 1e12
            print(f"[per-call-sync min]  E2E: {min_time*1000:.4f} ms  ->  {tflops_min_percall:.4f} TFLOP/s")
            print(f"[per-call-sync mean] E2E: {mean_time*1000:.4f} ms  ->  {tflops_mean_percall:.4f} TFLOP/s")
            print()
