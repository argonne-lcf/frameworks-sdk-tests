# FSDP libfabric memhooks shutdown SIGSEGV

Reproducer for
[AuroraBugTracking#180](https://github.com/argonne-lcf/AuroraBugTracking/issues/180),
where libfabric cleanup after an FSDP workload can crash when the MR-cache
monitor is `memhooks`.

## What it reproduces

The payload builds a tiny fully-sharded, `torch.compile`d model, runs one
forward/backward, synchronizes, and destroys the process group. Every rank
must print `work done <rank>` and then exit cleanly.

The launcher uses `--cpu-bind none` and defaults `FI_MR_CACHE_MONITOR` to
`memhooks`. During shutdown, the SYCL `CompressedRTDeviceBinaryImage`
destructor calls a libfabric memhooks interposer whose target mapping is
already unmapped, and ranks die with SIGSEGV. The resulting fault loop can
continue until the PALS timeout terminates the job.

Any rank dying from a signal, and therefore a nonzero `mpiexec` exit status,
is a reproduction.

## How to run

Through the harness (inside a PBS allocation with at least two nodes; the
case skips with an explicit reason otherwise):

```bash
./run_tests run --id regression-fsdp-shutdown-libfabric-memhooks --module frameworks
```

Directly, inside an allocation with at least two nodes:

```bash
module load frameworks
tests/regressions/fsdp_shutdown_libfabric_memhooks/run_repro.sh
```

Submitted to PBS:

```bash
qsub -A PROJECT -q workq -l select=2 -l walltime=00:30:00 \
    tests/regressions/fsdp_shutdown_libfabric_memhooks/run_repro.sh
```

`NRANKS_PER_NODE` (default `12`) sets ranks per node, and the world size is
`nodes × NRANKS_PER_NODE` from the allocation's nodefile. `MASTER_ADDR`
(default: first nodefile host) and `MASTER_PORT` (default `30032`) select the
rendezvous for `dist.init_process_group`. `MPIEXEC_TIMEOUT` (default `600`,
`0` disables) sets the PALS `mpiexec --timeout` so the fault loop terminates
the job instead of hanging. `PYTHON` (default `python`) selects the
interpreter, and `FRAMEWORKS_MODULE` (default `frameworks`) selects the
module loaded only when submitted directly with qsub.

`FI_MR_CACHE_MONITOR` can be overridden for triage with other monitors
(`userfaultfd`, `kdreg2`, ...); it defaults to `memhooks` for this reproducer.

## Expected result

On an unaffected stack every rank prints `work done <rank>` and `mpiexec`
exits zero. On the affected stack ranks die from SIGSEGV after their `work
done` lines, for example:

```text
work done 21
x4113c7s2b0n0.hsn.cm.aurora.alcf.anl.gov: rank 21 died from signal 11
```

## Files

- `repro_fsdp_shutdown.py`: the FSDP workload reduced in #180.
- `run_repro.sh`: harness entry point configuring the #180 launch conditions.
