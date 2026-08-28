# Adapted from EleutherAI/cookbook benchmarks/sizing/{mm_flops.py, utils.py}
# (https://github.com/EleutherAI/cookbook/blob/main/benchmarks/sizing/utils.py)
#
# Stripped: all non-mm Megatron-model benchmarks (bmm/linear/dropout/softmax/
# gelu/layernorm), argparse shape-sweep CLI, Tee file logging, CUDA-only
# print_benchmark_header.
#
# Dimension convention (note: upstream's benchmark_mm(m, n, k) uses n as the
# CONTRACTED dim and k as the output's last dim -- the opposite of what we've
# used elsewhere. Renamed to (m, k, n) here: A:(m,k) @ B:(k,n) = C:(m,n)).

import torch
import numpy as np
import datetime
from torch.profiler import profile, ProfilerActivity

#device = "xpu"
dev_index=0
device = torch.device(f"xpu:{dev_index}")
dtype = torch.bfloat16
#dtype = torch.float16
#dtype = torch.float32
#torch.set_float32_matmul_precision("high")
num_warmup_iterations = 10
num_iterations = 300

shape_list = [
    (2048, 2048, 65536),    # "small"
    (2048, 2048, 262144),   # "large"
]


def benchmark_mm(A, B, C, m, k, n):
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    times = []
    wall_times = []
    for _ in range(num_warmup_iterations + num_iterations):
        with torch.no_grad():
            wall_ts = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
            start.record()
            torch.mm(A, B, out=C)
            end.record()
        torch.xpu.synchronize()
        times.append(start.elapsed_time(end)) # ms
        wall_times.append(wall_ts)

    times = times[num_warmup_iterations:]
    min_time = min(times) / 1000    # ms -> s, best-of-N (matches upstream's np.amin)
    mean_time = (sum(times) / len(times)) / 1000  # ms -> s

    flops = 2 * m * n * k
    tflops = flops / ((np.array(times) / 1000) * 1e12)
    tflops_min = flops / (min_time * 1e12)
    tflops_mean = flops / (mean_time * 1e12)
    print(f"t_flops = {tflops}")
    print(f"Wall_times= {wall_times}")
    print(f"[min]  elapsed (best of {num_iterations}) for {m}x{k}x{n}: {min_time*1000:.4f} ms  ->  {tflops_min:.4f} TFLOP/s")
    print(f"[mean] elapsed (avg of {num_iterations}) for {m}x{k}x{n}: {mean_time*1000:.4f} ms  ->  {tflops_mean:.4f} TFLOP/s")
    return min_time, mean_time


if __name__ == "__main__":
    print(f"device: {torch.xpu.get_device_name(0)}")
    print(f"device_count: {torch.xpu.device_count()}")
    print()

    for m, k, n in shape_list:
        A = torch.randn(m, k, dtype=dtype, device=device)
        B = torch.randn(k, n, dtype=dtype, device=device)
        C = torch.empty(m, n, dtype=dtype, device=device)

        print(f"=== shape: m={m} k={k} n={n} dtype={dtype} ===")

        # warm up (avoids first-touch driver init / kernel JIT-compile
        # showing up in the profiler table below -- this is what caused the
        # small-shape table's zeModuleCreate/zeKernelCreate/zeInitDrivers
        # noise last run)
        with torch.no_grad():
            torch.mm(A, B, out=C)
        torch.xpu.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for _ in range(20):
                with torch.no_grad():
                    torch.mm(A, B, out=C)
            torch.xpu.synchronize()
        #prof.export_chrome_trace(f"trace_m{m}_k{k}_n{n}_{dtype}.json")
        print(prof.key_averages().table(sort_by="xpu_time_total"))

        benchmark_mm(A, B, C, m, k, n)
        print("-" * 80)
