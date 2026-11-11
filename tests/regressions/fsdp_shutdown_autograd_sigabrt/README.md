# FSDP autograd interpreter-shutdown SIGABRT

Reproducer for
[AuroraBugTracking#182](https://github.com/argonne-lcf/AuroraBugTracking/issues/182),
where an autograd callback-lifetime race can abort interpreter shutdown after
an FSDP backward pass.

## What it reproduces

The payload builds a tiny fully-sharded, `torch.compile`d model, runs one
forward/backward, synchronizes, and destroys the process group. Every rank
must print `work done <rank>` and then exit cleanly.

An autograd worker releases queued Python callback references while the main
thread is inside `Py_FinalizeEx`. The worker's callback deleter tries to
acquire the GIL, enters `PyThread_exit_thread`, and unwinding reaches
`std::terminate`, so the process aborts with SIGABRT after its `work done`
line.

The launcher retains the default CPU binding and defaults
`FI_MR_CACHE_MONITOR=disabled` so the unrelated libfabric memhooks SIGSEGV
cannot mask this failure. Any rank dying from a signal, and therefore a
nonzero `mpiexec` exit status, is a reproduction.

## How to run

Through the harness (inside a PBS allocation with at least two nodes; the
case skips with an explicit reason otherwise):

```bash
./run_tests run --id regression-fsdp-shutdown-autograd-sigabrt --module frameworks
```

Directly, inside an allocation with at least two nodes:

```bash
module load frameworks
tests/regressions/fsdp_shutdown_autograd_sigabrt/run_repro.sh
```

Submitted to PBS:

```bash
qsub -A PROJECT -q workq -l select=2 -l walltime=00:30:00 \
    tests/regressions/fsdp_shutdown_autograd_sigabrt/run_repro.sh
```

`NRANKS_PER_NODE` (default `12`) sets ranks per node, and the world size is
`nodes × NRANKS_PER_NODE` from the allocation's nodefile. `MASTER_ADDR`
(default: first nodefile host) and `MASTER_PORT` (default `30032`) select the
rendezvous for `dist.init_process_group`. `MPIEXEC_TIMEOUT` (default `600`,
`0` disables) sets the PALS `mpiexec --timeout`. `PYTHON` (default `python`)
selects the interpreter, and `FRAMEWORKS_MODULE` (default `frameworks`)
selects the module loaded only when submitted directly with qsub.

`FI_MR_CACHE_MONITOR` can be overridden for triage, but defaults to `disabled`
to isolate the autograd shutdown failure.

The single-process `ModuleTracker` reduction of the same issue is registered
under `tests/regressions/module_tracker_shutdown/`.

## Expected result

On an unaffected stack every rank prints `work done <rank>` and `mpiexec`
exits zero. On the affected stack ranks abort with SIGABRT after their `work
done` lines, for example:

```text
work done 21
terminate called without an active exception
x4113c7s2b0n0.hsn.cm.aurora.alcf.anl.gov: rank 21 died from signal 6
```

## Files

- `repro_fsdp_shutdown.py`: the FSDP workload reduced in #182.
- `run_repro.sh`: harness entry point configuring the #182 launch conditions.
