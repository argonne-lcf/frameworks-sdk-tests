"""Validate collectives across disjoint, sequential, and overlapping process groups."""

import os
import time

import torch
import torch.distributed as dist

import _common as C


MODE = os.environ.get("TEST_GROUPS", "ep")  # ep | pp | seq | disjoint | overlap
VALID_MODES = {"ep", "pp", "seq", "disjoint", "overlap"}
if MODE not in VALID_MODES:
    raise SystemExit(
        f"[fatal] unknown TEST_GROUPS={MODE!r}; expected one of "
        + ", ".join(sorted(VALID_MODES))
    )

ctx = C.Ctx(f"subgroups/{MODE}")
if MODE == "overlap" and (ctx.world < 4 or ctx.world % 4):
    dist.destroy_process_group()
    raise SystemExit("[fatal] overlap mode requires a world size divisible by 4")

# Build a 2-D mesh: EP groups are rows and PP groups are columns. EP must divide
# both the world and the ranks per node. Dividing only the world can make the
# supposedly local EP groups straddle nodes and invalidate the EP/PP comparison.
EP = int(os.environ.get("TEST_EP", "0")) or max(
    (
        divisor
        for divisor in range(1, min(6, ctx.world) + 1)
        if ctx.world % divisor == 0 and ctx.world // divisor >= 2
    ),
    default=1,
)
RPN = C._envint(
    "LOCAL_WORLD_SIZE",
    "PALS_LOCAL_SIZE",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "MPI_LOCALNRANKS",
)
if ctx.world % EP:
    raise SystemExit(f"[fatal] TEST_EP={EP} does not divide world {ctx.world}")
if RPN and RPN % EP:
    raise SystemExit(
        f"[fatal] TEST_EP={EP} does not divide {RPN} ranks per node; "
        "EP groups would straddle nodes"
    )
PP = ctx.world // EP

ep_members = list(range(ctx.rank // EP * EP, ctx.rank // EP * EP + EP))
pp_members = list(range(ctx.rank % EP, ctx.world, EP))

# new_group is collective over the whole world: every rank must create every
# group in the same order, including groups it never joins.
t0 = time.perf_counter()
ep_groups = [
    dist.new_group(list(range(group * EP, (group + 1) * EP)))
    for group in range(PP)
]
t_ep = time.perf_counter() - t0
t0 = time.perf_counter()
pp_groups = [
    dist.new_group(list(range(column, ctx.world, EP))) for column in range(EP)
]
t_pp = time.perf_counter() - t0

my_ep = ep_groups[ctx.rank // EP]
my_pp = pp_groups[ctx.rank % EP]

# Size from the largest group in play so per-peer bytes match across modes.
largest_group = ctx.world // 2 if MODE == "overlap" else max(EP, PP)
set_count = 2 if MODE in ("seq", "overlap") else 1
n = C.size_for(set_count * (largest_group + 1))

gather = getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor


def gather_set(tag, group, members):
    """Return operation, validator, and reset callbacks for one communicator."""
    source = torch.empty(n, dtype=C.DTYPE, device=ctx.device)
    output = torch.empty(n * len(members), dtype=C.DTYPE, device=ctx.device)
    C.fill(source, tag, ctx.rank)
    return (
        lambda: gather(output, source, group=group),
        lambda: C.check_finite(ctx, output)
        and C.check_regions(
            ctx,
            output,
            n,
            lambda index: (tag, members[index]),
            f"tag{tag} slot",
        ),
        output.zero_,
    )


report = None

if MODE == "pp":
    sets = [gather_set(1, my_pp, pp_members)]
elif MODE == "seq":
    sets = [
        gather_set(0, my_ep, ep_members),
        gather_set(1, my_pp, pp_members),
    ]
elif MODE == "overlap":
    # Two groups share one quarter of the world, a shape a partition cannot make.
    half = ctx.world // 2
    quarter = ctx.world // 4
    members_a = list(range(half))
    members_b = list(range(quarter, quarter + half))
    group_a = dist.new_group(members_a)
    group_b = dist.new_group(members_b)
    sets = (
        [gather_set(2, group_a, members_a)] if ctx.rank in members_a else []
    ) + ([gather_set(3, group_b, members_b)] if ctx.rank in members_b else [])
    # Collective order is per communicator, not per rank. Reverse it on some
    # shared ranks to expose backends that serialize completion on the device.
    if len(sets) == 2 and ctx.rank % 2:
        sets.reverse()
else:
    sets = [gather_set(0, my_ep, ep_members)]

if MODE == "disjoint":
    fire = sets[0][0]

    def timed(active):
        ctx.barrier()
        start = time.perf_counter()
        if active:
            fire()
        ctx.sync()
        return time.perf_counter() - start

    calibration_iterations = int(os.environ.get("TEST_GROUP_CALIB", "8"))
    if calibration_iterations < 3:
        raise SystemExit("[fatal] TEST_GROUP_CALIB must be at least 3")
    solo = [
        timed(ctx.rank < EP) for _ in range(calibration_iterations)
    ]
    concurrent = [timed(True) for _ in range(calibration_iterations)]
    solo_tail = sum(solo[-3:]) / 3
    concurrent_tail = sum(concurrent[-3:]) / 3
    ctx.log("one group alone: " + " ".join(f"{value:.3f}" for value in solo))
    ctx.log(
        "all groups conc: " + " ".join(f"{value:.3f}" for value in concurrent)
    )

    def report(_times):
        ctx.log(
            f"interference {concurrent_tail / max(solo_tail, 1e-9):.2f}x "
            f"({PP} disjoint groups at once {concurrent_tail:.3f}s vs one alone "
            f"{solo_tail:.3f}s; 1.00 = the fabric absorbs them)"
        )


ctx.log(
    f"mesh {EP}ep x {PP}pp groups {PP} ep + {EP} pp created in "
    f"{t_ep:.3f}s + {t_pp:.3f}s ({(t_ep + t_pp) / (EP + PP):.4f}s each)"
)
ctx.log(
    f"rank 0 ep={ep_members} pp={pp_members} per-peer "
    f"{C.human(n * torch.empty((), dtype=C.DTYPE).element_size())} "
    f"({n} elems) x {len(sets)} set(s)"
)

C.run(
    ctx,
    lambda: [operation() for operation, _, _ in sets],
    lambda: all(validate() for _, validate, _ in sets),
    lambda: [reset() for _, _, reset in sets],
    report,
)
