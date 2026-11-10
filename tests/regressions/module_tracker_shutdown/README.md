# ModuleTracker interpreter-shutdown SIGABRT (AuroraBugTracking #182)

The single-process reduction of
[AuroraBugTracking#182](https://github.com/argonne-lcf/AuroraBugTracking/issues/182):
no FSDP, no compilation, and no distributed communication. The multi-rank
FSDP variant is registered under
`tests/regressions/fsdp_shutdown_autograd_sigabrt/`.

## What it reproduces

`ModuleTracker` plus one XPU backward pass leaves an autograd worker thread
holding Python callback references. When the main thread enters
`Py_FinalizeEx`, the worker's callback deleter tries to acquire the GIL,
enters `PyThread_exit_thread`, and the unwinding reaches `std::terminate`,
so the process aborts with SIGABRT after the final `work done` line. The
failure reproduced with single-CPU affinity in the upstream report, so the
launcher pins the payload to one CPU with `taskset`.

## How to run

Through the harness (single XPU):

```bash
./run_tests run --id regression-module-tracker-shutdown-sigabrt --module frameworks
```

Directly:

```bash
module load frameworks
tests/regressions/module_tracker_shutdown/run_repro.sh
```

`PYTHON` (default `python`) overrides the interpreter.

## Expected result

On an unaffected stack the process prints `work done 0` and exits zero. On
the affected stack the same line is printed and the process then dies from
signal 6:

```text
work done 0
terminate called without an active exception
Aborted
```

## Files

- `repro_module_tracker_shutdown.py`: the workload, reduced in #182.
- `run_repro.sh`: harness entry point; pins the payload to one CPU.
