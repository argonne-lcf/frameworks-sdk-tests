#!/usr/bin/env python3
"""Validate ezpz's distributed bring-up against the SDK's own XCCL stack.

``ezpz.setup_torch()`` is the single call most ALCF PyTorch jobs use to go from
"N processes exist" to "a working process group on the right device". It reads
the scheduler environment (PALS/PBS, or torchrun's), picks the device, picks the
backend, and initializes the process group. Every one of those steps can succeed
partially -- the classic failure is every rank binding device 0, which does not
raise, does not hang, and quietly destroys throughput and correctness.

This test therefore checks bring-up *results*, not just that the call returned:

  1. ezpz's rank/world/local-rank agree with the launcher's own environment;
  2. ranks are a permutation of ``range(world_size)`` -- no duplicates, which is
     what a botched rank assignment produces;
  3. local ranks map to *distinct* devices within a node, catching the
     everyone-on-device-0 failure directly;
  4. a real collective over ezpz's process group returns the mathematically
     correct answer, so the communicator ezpz built actually communicates.

Run under any launcher the SDK supports::

    mpiexec -n 12 -ppn 12 python tests/distributed/ezpz_distributed.py
    torchrun --standalone --nproc-per-node=2 tests/distributed/ezpz_distributed.py

``TEST_DEVICE=cpu`` forces the gloo path so the logic is exercisable without an
accelerator.
"""

from __future__ import annotations

import hashlib
import os
import socket
import sys
from typing import List, Optional

import torch
import torch.distributed as dist


TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "300"))


def _envint(*names: str) -> Optional[int]:
    """First integer value among *names*, mirroring the launcher variables."""
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            try:
                return int(raw)
            except ValueError:
                continue
    return None


def _launcher_view() -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Rank/world/local-rank according to the launcher, independent of ezpz.

    Aurora's PALS is the awkward case: ``mpiexec`` exports ``PALS_RANKID``,
    ``PALS_LOCAL_RANKID`` and ``PALS_LOCAL_SIZE``, but *no* global world-size
    variable (verified on a compute node -- there is no ``PALS_WORLD_SIZE`` or
    ``PALS_NRANKS``). Requiring one here would make every PALS launch fail to
    resolve, so the world size falls back to ``local_size * node_count`` and
    finally to torch.distributed itself once the group is up.
    """
    rank = _envint(
        "RANK", "PALS_RANKID", "PMIX_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK",
        "SLURM_PROCID",
    )
    world = _envint(
        "WORLD_SIZE", "PALS_WORLD_SIZE", "PALS_NRANKS", "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE", "SLURM_NTASKS",
    )
    local = _envint(
        "LOCAL_RANK", "PALS_LOCAL_RANKID", "MPI_LOCALRANKID",
        "MPICH_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "PMI_LOCAL_RANK",
        "SLURM_LOCALID",
    )
    if world is None:
        # PALS exports no world-size variable. Deriving one from
        # PALS_LOCAL_SIZE * node_count is wrong whenever the launch uses fewer
        # nodes than the allocation holds (observed: a 2-rank mpiexec inside a
        # 2-node job yields local_size=2, nodes=2 -> 4). There is no reliable
        # environment-only answer, so leave it unresolved and let the
        # torch.distributed cross-check below carry the world-size assertion.
        pass
    return rank, world, local


def main() -> int:
    import ezpz

    failures: List[str] = []

    env_rank, env_world, env_local = _launcher_view()
    if env_rank is None:
        print(
            "[fatal] cannot resolve a rank from the environment; launch this "
            "under mpiexec/PALS, torchrun, srun, or `ezpz launch`",
            file=sys.stderr,
            flush=True,
        )
        return 2

    # The call under test. It initializes the process group as a side effect and
    # returns this process's rank.
    reported_rank = ezpz.setup_torch()

    rank = ezpz.get_rank()
    world = ezpz.get_world_size()
    local_rank = ezpz.get_local_rank()
    device_type = ezpz.get_torch_device_type()
    backend = ezpz.get_torch_backend()

    if rank == 0:
        print(
            f"ezpz={ezpz.__version__} torch={torch.__version__} "
            f"world={world} device_type={device_type} backend={backend}",
            flush=True,
        )

    if not dist.is_initialized():
        print(
            f"[rank {rank}] FAIL setup_torch() returned without initializing "
            "a process group",
            flush=True,
        )
        return 1

    # setup_torch()'s return value is used as "my rank" by real job scripts; it
    # disagreeing with get_rank() would make those scripts address the wrong
    # shard while every other check still passes.
    if int(reported_rank) != rank:
        failures.append(
            f"setup_torch() returned {reported_rank} but get_rank() is {rank}"
        )

    # (1) ezpz must agree with the launcher that spawned this process.
    if rank != env_rank:
        failures.append(f"ezpz rank {rank} != launcher rank {env_rank}")
    if env_world is not None and world != env_world:
        failures.append(f"ezpz world size {world} != launcher world {env_world}")
    elif env_world is None and rank == 0:
        # PALS exports no world-size variable; say so rather than silently
        # skipping a check the reader assumes ran.
        print(
            "note: launcher exported no world-size variable; "
            "cross-checking against torch.distributed only",
            flush=True,
        )
    if env_local is not None and local_rank != env_local:
        failures.append(
            f"ezpz local rank {local_rank} != launcher local rank {env_local}"
        )
    if dist.get_rank() != rank or dist.get_world_size() != world:
        failures.append(
            f"torch.distributed reports rank {dist.get_rank()}/"
            f"{dist.get_world_size()}, ezpz reports {rank}/{world}"
        )

    device = ezpz.get_torch_device()
    if isinstance(device, str):
        device = torch.device(device)
    # A device without an explicit index means "current device", which is the
    # ambiguity this test exists to remove; resolve it the way ezpz would.
    if device.type != "cpu" and device.index is None:
        device = torch.device(device.type, torch.accelerator.current_device_index())
    collective_device = device if device.type != "cpu" else torch.device("cpu")
    barrier_kwargs = (
        {} if collective_device.type == "cpu" else {"device_ids": [collective_device.index]}
    )

    print(
        f"[rank {rank}/{world}] local_rank={local_rank} device={device} "
        f"host={socket.gethostname()}",
        flush=True,
    )

    # (2) Ranks must be a permutation of range(world): all_gather each rank's
    # own id and check the set. Duplicated ranks are a real, observed failure of
    # environment-derived rank assignment, and they do not raise on their own.
    rank_tensor = torch.tensor([rank], dtype=torch.int64, device=collective_device)
    gathered_ranks = [torch.zeros_like(rank_tensor) for _ in range(world)]
    dist.all_gather(gathered_ranks, rank_tensor)
    observed = sorted(int(item.item()) for item in gathered_ranks)
    if observed != list(range(world)):
        failures.append(
            f"ranks are not a permutation of range({world}): {observed}"
        )
    elif rank == 0:
        print(f"PASS rank permutation ({world} unique ranks)", flush=True)

    # (3) Distinct devices per node. Ranks sharing a host must hold distinct
    # device indices; everyone binding device 0 is silent and catastrophic.
    if collective_device.type != "cpu":
        # NOT hash(): Python randomizes string hashing per process (PYTHONHASH-
        # SEED), so every rank would compute a different key for the same host
        # and each rank would look like its own node -- making this check
        # vacuously pass. A stable digest is required for cross-rank grouping.
        host_hash = int.from_bytes(
            hashlib.blake2b(
                socket.gethostname().encode("utf-8"), digest_size=7
            ).digest(),
            "big",
        )
        pair = torch.tensor(
            [host_hash, int(collective_device.index or 0)],
            dtype=torch.int64,
            device=collective_device,
        )
        gathered_pairs = [torch.zeros_like(pair) for _ in range(world)]
        dist.all_gather(gathered_pairs, pair)
        per_host: dict[int, List[int]] = {}
        for item in gathered_pairs:
            host_key, device_index = int(item[0].item()), int(item[1].item())
            per_host.setdefault(host_key, []).append(device_index)
        collisions = {
            key: sorted(indices)
            for key, indices in per_host.items()
            if len(indices) != len(set(indices))
        }
        if collisions:
            example = next(iter(collisions.values()))
            failures.append(
                f"ranks on one host share device indices {example}; "
                "each local rank must bind a distinct device"
            )
        elif rank == 0:
            sizes = sorted({len(v) for v in per_host.values()})
            print(
                f"PASS distinct devices per host "
                f"({len(per_host)} host(s), ranks/host {sizes})",
                flush=True,
            )

    # (4) The communicator must actually compute. Sum of rank+1 has a closed
    # form, so a wrong answer is unambiguous rather than a tolerance question.
    value = torch.tensor(
        [float(rank + 1)], dtype=torch.float32, device=collective_device
    )
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = float(world * (world + 1) // 2)
    if abs(value.item() - expected) > 1e-3:
        failures.append(
            f"all_reduce over ezpz's process group produced {value.item()}, "
            f"expected {expected}"
        )
    elif rank == 0:
        print(f"PASS all_reduce sum={value.item():.0f} (expected {expected:.0f})",
              flush=True)

    dist.barrier(**barrier_kwargs)

    # Fail the whole job if any rank failed, not just rank 0: a per-rank device
    # bug would otherwise exit 0 because rank 0 happened to be correct.
    flag = torch.tensor(
        [1 if failures else 0], dtype=torch.int64, device=collective_device
    )
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    for failure in failures:
        print(f"[rank {rank}] FAIL {failure}", flush=True)

    failed = bool(flag.item())
    if rank == 0:
        print(f"RESULT {'FAIL' if failed else 'PASS'}", flush=True)

    ezpz.cleanup()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
