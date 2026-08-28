# XCCL `empty_cache` Repro Package

This package isolates an Intel XPU/XCCL memory leak triggered by freeing collective-participating device allocations with `torch.xpu.empty_cache()`.

## Files

- `prove_list_allgather_hidden_temp.py`: proof-style repro with four modes.
- `run_repros.sh`: runs the four proof modes.

## Environment

Load the Frameworks SDK module first, or invoke this test through the root
runner, which loads it automatically.

```bash
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
```

## Run Proof Repro

```bash
./run_repros.sh
#or for running with Torchrun
./run_repros_torchrun.sh
```

The test exits nonzero if any mode loses more device memory per iteration than
`LEAK_TOLERANCE_GIB`. Expected summary lines on the affected stack include:

```text
list_hidden_temp   observed_drop_per_iter=0.373GiB
into_persistent   observed_drop_per_iter=0.000GiB
explicit_temp     observed_drop_per_iter=0.373GiB
temp_no_collective observed_drop_per_iter=0.000GiB
```

For a shorter/faster run, reduce tensor size:

```bash
TENSOR_SIZE=10000000 MAX_ITERS=30 ./run_repros.sh
```
