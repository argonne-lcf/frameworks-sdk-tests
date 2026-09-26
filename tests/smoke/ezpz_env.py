#!/usr/bin/env python3
"""Validate that ezpz resolves this SDK's accelerator, backend, and scheduler.

ezpz is the launch/distributed-setup layer most ALCF PyTorch jobs sit on, but it
is *not* part of the Frameworks SDK: it is installed per-user, so the pairing of
"this SDK build" with "the ezpz on PYTHONPATH" is exactly what goes untested
until a job fails at scale. This case pins that pairing down.

It is deliberately single-process and MPI-free. ``ezpz.get_rank()`` and anything
else that reaches ``mpi4py.MPI`` aborts the interpreter outside an allocation
(``Fatal error in internal_Init_thread``), so importing ezpz must stay separable
from initializing a communicator -- and this test proves that it is.

The substantive check is agreement, not availability: ezpz must reach the same
verdict about the accelerator as the loaded PyTorch does. An ezpz that silently
answers "cpu"/"gloo" on a 12-XPU node is the failure this catches, and it is
invisible to a test that only imports the package.
"""

from __future__ import annotations

import importlib
import os
from typing import Callable, List, Tuple


# Attributes the suite's distributed cases and ordinary ALCF job scripts rely
# on. Resolving them exercises ezpz's lazy __getattr__ surface, which silently
# swallows a submodule that fails to import -- a missing dependency therefore
# shows up here as a missing attribute rather than as a stack trace at scale.
REQUIRED_ATTRIBUTES = (
    "setup_torch",
    "get_rank",
    "get_world_size",
    "get_local_rank",
    "get_torch_device",
    "get_torch_device_type",
    "get_torch_backend",
    "get_machine",
    "get_hostname",
    "cleanup",
)

# Accelerator backends that count as "not a CPU fallback".
ACCELERATOR_BACKENDS = {"xccl", "ccl", "nccl"}


def _probe(label: str, function: Callable[[], object]) -> Tuple[bool, object]:
    """Call an introspection helper, reporting failure instead of raising."""
    try:
        value = function()
    except BaseException as error:  # ABI/driver failures are not all Exceptions
        print(f"FAIL {label}: {type(error).__name__}: {error}", flush=True)
        return False, None
    print(f"{label}={value!r}", flush=True)
    return True, value


def main() -> None:
    failures: List[str] = []

    import ezpz

    print(f"ezpz={ezpz.__version__}", flush=True)
    # Which ezpz actually got imported matters: the SDK ships none, so this is
    # a user-site or venv copy and the path is the only way to tell them apart
    # when triaging a failure.
    print(f"ezpz_path={list(getattr(ezpz, '__path__', []))}", flush=True)
    print(f"PYTHONUSERBASE={os.environ.get('PYTHONUSERBASE', '')}", flush=True)

    for name in REQUIRED_ATTRIBUTES:
        try:
            attribute = getattr(ezpz, name)
        except AttributeError as error:
            # ezpz's __getattr__ appends the underlying import failure when it
            # has one, so this message usually names the real missing dependency.
            failures.append(f"ezpz.{name} unavailable: {error}")
            print(f"FAIL attribute {name}: {error}", flush=True)
        else:
            module = getattr(attribute, "__module__", "?")
            print(f"PASS attribute {name} ({module})", flush=True)

    ok_type, device_type = _probe(
        "device_type", ezpz.get_torch_device_type
    )
    ok_backend, backend = _probe("backend", ezpz.get_torch_backend)
    _probe("machine", ezpz.get_machine)
    _probe("hostname", ezpz.get_hostname)

    from ezpz.configs import get_scheduler

    _probe("scheduler", get_scheduler)

    if not ok_type:
        failures.append("ezpz could not report a torch device type")
    if not ok_backend:
        failures.append("ezpz could not report a torch backend")

    # The agreement check. torch is the ground truth for what the SDK can see;
    # ezpz disagreeing with it means every job launched through ezpz lands on
    # the wrong device or the wrong communication backend.
    import torch

    torch_xpu = bool(
        hasattr(torch, "xpu") and torch.xpu.is_available()
    ) and torch.xpu.device_count() > 0
    print(
        f"torch={torch.__version__} torch_xpu_available={torch_xpu} "
        f"torch_xpu_count={torch.xpu.device_count() if hasattr(torch, 'xpu') else 0}",
        flush=True,
    )

    if ok_type:
        ezpz_xpu = str(device_type).lower() == "xpu"
        if torch_xpu and not ezpz_xpu:
            failures.append(
                f"torch reports usable XPUs but ezpz selected device type "
                f"{device_type!r}; ezpz-launched jobs would run on the CPU"
            )
        elif not torch_xpu and ezpz_xpu:
            failures.append(
                f"ezpz selected device type {device_type!r} but torch reports "
                "no usable XPU"
            )
        else:
            print(
                f"PASS device agreement (torch_xpu={torch_xpu}, ezpz={device_type})",
                flush=True,
            )

    if ok_type and ok_backend:
        # A CPU device must not claim an accelerator backend, and an XPU device
        # falling back to gloo is a silently-degraded job, not a working one.
        backend_name = str(backend).lower()
        if str(device_type).lower() == "xpu":
            if backend_name not in ACCELERATOR_BACKENDS:
                failures.append(
                    f"device type 'xpu' resolved to backend {backend!r}; "
                    f"expected one of {sorted(ACCELERATOR_BACKENDS)}"
                )
            else:
                print(f"PASS backend agreement (xpu -> {backend})", flush=True)
        elif backend_name in ACCELERATOR_BACKENDS:
            failures.append(
                f"device type {device_type!r} resolved to accelerator backend "
                f"{backend!r}"
            )
        else:
            print(
                f"PASS backend agreement ({device_type} -> {backend})", flush=True
            )

    # Importing ezpz must not have initialized MPI. If it had, this process
    # would already have aborted on a login node; assert the invariant anyway so
    # a future ezpz that imports mpi4py eagerly is caught here and not by every
    # downstream tool that imports ezpz outside a job.
    mpi_module = importlib.import_module("mpi4py")
    initialized = getattr(mpi_module, "MPI", None)
    if initialized is not None and initialized.Is_initialized():
        failures.append("importing ezpz initialized MPI; it must stay lazy")
    else:
        print("PASS mpi_not_initialized_by_import", flush=True)

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", flush=True)
        raise SystemExit(f"{len(failures)} ezpz environment check(s) failed")

    print("PASS ezpz environment consistent with the loaded SDK", flush=True)


if __name__ == "__main__":
    main()
