"""Prospective three-axis equal-MAC geometry and finite-context compute model.

This module is independent of the frozen v3 and Round A experiments.  The
array performs spatial BF16 multiplies and FP32 reductions.  Changing PK
changes the reduction tree/segment order; retaining precision does not imply
bitwise equivalence to v3's 512-leaf tree.

Timing profiles are declared *hypotheses*, anchored to the inherited PK=512
dot=20, commit=1 constants.  None is a synthesis result.  Operands are ready
in this compute projection: HBM, SRAM ports/banks, vector service and control
contention must be scheduled separately by the caller.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Iterable

TOTAL_MACS = 12_288
PM_MAX = 16
PN_MAX = 192
PK_VALUES = (32, 64, 128, 256, 512, 1024)
BF16_BYTES = 2
FP32_BYTES = 4


def ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


@dataclass(frozen=True, order=True)
class Core:
    pm: int
    pn: int
    pk: int

    def __post_init__(self) -> None:
        if not (1 <= self.pm <= PM_MAX and 1 <= self.pn <= PN_MAX
                and self.pk in PK_VALUES):
            raise ValueError("core requires PM[1,16], PN[1,192], PK in {32,64,128,256,512,1024}")

    @property
    def macs(self) -> int:
        return self.pm * self.pn * self.pk

    @property
    def record_bytes(self) -> int:
        """Padded FP32 partial-sum tile; capacity never depends on logical tails."""
        return self.pm * self.pn * FP32_BYTES

    @property
    def x_slice_bytes(self) -> int:
        return self.pm * self.pk * BF16_BYTES

    @property
    def w_slice_bytes(self) -> int:
        return self.pn * self.pk * BF16_BYTES


def geometry_id(cores: Iterable[Core]) -> str:
    return "+".join(f"{c.pm}x{c.pn}x{c.pk}" for c in sorted(cores))


def family(cores: tuple[Core, ...]) -> str:
    if len(cores) == 1:
        return "single"
    if len(cores) != 2:
        raise ValueError("geometry must contain one or two cores")
    return "homogeneous" if cores[0] == cores[1] else "heterogeneous"


@lru_cache(maxsize=1)
def enumerate_geometries() -> tuple[tuple[Core, ...], ...]:
    """All declared integer geometries with exactly 12,288 main multipliers.

    Both dual-core PK values vary independently.  Lexicographically sorted
    cores canonicalize mirrors; homogeneous designs occur exactly once.
    This is exhaustive inside the declared bounds, not over all hardware.
    """
    shapes = sorted(Core(m, n, k)
                    for m in range(1, PM_MAX + 1)
                    for n in range(1, PN_MAX + 1)
                    for k in PK_VALUES if m * n * k <= TOTAL_MACS)
    by_macs: dict[int, list[Core]] = defaultdict(list)
    for core in shapes:
        by_macs[core.macs].append(core)
    singles = [(c,) for c in shapes if c.macs == TOTAL_MACS]
    pairs = [(a, b) for a in shapes for b in by_macs[TOTAL_MACS - a.macs]
             if a <= b]
    return tuple(singles + pairs)


def enumeration_counts() -> dict[str, int]:
    geometries = enumerate_geometries()
    counts = Counter(family(g) for g in geometries)
    return {"total": len(geometries), "single": counts["single"],
            "dual": counts["homogeneous"] + counts["heterogeneous"],
            "homogeneous": counts["homogeneous"],
            "heterogeneous": counts["heterogeneous"],
            "heterogeneous_same_pk": sum(len(g) == 2 and g[0] != g[1]
                                          and g[0].pk == g[1].pk for g in geometries),
            "heterogeneous_unequal_pk": sum(len(g) == 2 and g[0].pk != g[1].pk
                                             for g in geometries)}


@dataclass(frozen=True)
class TimingProfile:
    name: str
    latency_by_pk: tuple[tuple[int, int], ...]
    initiation_interval: int = 1
    commit_cycles: int = 1
    hypothesis: str = "explicit user-supplied analytical timing hypothesis"

    def __post_init__(self) -> None:
        keys = [k for k, _ in self.latency_by_pk]
        if sorted(keys) != list(PK_VALUES) or len(set(keys)) != len(keys):
            raise ValueError("timing profile must give one dot latency for each supported PK")
        if (self.initiation_interval < 1 or self.commit_cycles < 1
                or any(v < 1 for _, v in self.latency_by_pk)):
            raise ValueError("positive dot latency, initiation interval and commit cycles required")

    def dot_latency(self, core: Core | int) -> int:
        pk = core.pk if isinstance(core, Core) else core
        for key, latency in self.latency_by_pk:
            if key == pk:
                return latency
        raise ValueError(f"unsupported PK {pk}")

    def completion_latency(self, core: Core | int) -> int:
        return self.dot_latency(core) + self.commit_cycles


def log2_stage_profile(stage_cycles: int) -> TimingProfile:
    """Anchor Ldot(512)=20 and remove log2(512/PK) reduction stages.

    One, two or four cycles per changed stage are explicit sensitivity choices.
    Fixed costs, routing and clock closure are unvalidated.  Tail masking
    uses the installed full-PK pipeline latency for every issue.
    """
    if stage_cycles not in (1, 2, 4):
        raise ValueError("declared sensitivities use one, two or four cycles per reduction stage")
    latencies = tuple((pk, 20 - stage_cycles * (9 - (pk.bit_length() - 1)))
                      for pk in PK_VALUES)
    return TimingProfile(f"log2_stage{stage_cycles}", latencies,
                         hypothesis=(f"Ldot(PK)=20-{stage_cycles}*log2(512/PK); "
                                     "PK512 anchor only; no physical validation"))


TIMING_PROFILES = {
    "conservative_flat20": TimingProfile(
        "conservative_flat20", tuple((pk, 20) for pk in PK_VALUES),
        hypothesis="all PK retain inherited 20-cycle dot; conservative sensitivity, unvalidated"),
    "log2_stage1": log2_stage_profile(1),
    "log2_stage2": log2_stage_profile(2),
    "log2_stage4": log2_stage_profile(4),
}
DEFAULT_TIMING = TIMING_PROFILES["log2_stage2"]


@dataclass(frozen=True)
class ContextLimits:
    """Installed bounds per core, independent of batch or expert shape.

    A record is one spatial PM×PN FP32 output tile retained through all K
    segments.  This differs from v3's whole-expert Current/Next contexts.
    Both limits apply; neither permits dynamic unbudgeted enlargement.
    """
    max_records: int = 8
    accumulator_bytes: int = 65_536

    def __post_init__(self) -> None:
        if self.max_records < 1 or self.accumulator_bytes < 1:
            raise ValueError("positive record and accumulator bounds required")

    def capacity(self, core: Core) -> int:
        capacity = min(self.max_records, self.accumulator_bytes // core.record_bytes)
        if capacity < 1:
            raise ValueError("core's single padded output record exceeds accumulator capacity")
        return capacity


@dataclass(frozen=True)
class GroupCost:
    first_record: int
    records: int
    issues: int
    cycles: int
    replacement_interval: int
    accumulator_bytes: int


@dataclass(frozen=True)
class ProjectionCost:
    cycles: int
    issues: int
    useful_macs: int
    issued_macs: int
    padding_macs: int
    m_waves: int
    n_tiles: int
    k_segments: int
    group_count: int
    resident_record_limit: int
    peak_resident_records: int
    record_bytes: int
    peak_accumulator_bytes: int
    x_slice_bytes: int
    w_slice_bytes: int
    accumulator_read_bytes: int
    accumulator_write_bytes: int
    dot_latency: int
    completion_latency: int
    initiation_interval: int
    last_m_rows: int
    last_n_columns: int
    last_k_elements: int

    @property
    def spatial_utilization(self) -> float:
        return self.useful_macs / self.issued_macs


def group_cost(records: int, k_segments: int, core: Core,
               timing: TimingProfile, first_record: int = 0) -> GroupCost:
    """Exact result-dependency schedule of one ready-operand group.

    For each ascending K segment, issue records in the same order.  The next
    segment of a record waits for its previous FP32 result to commit.  With
    q records, s segments, II=I and completion latency L, final commit is
      (s-1)*max(q*I,L) + (q-1)*I + L.
    Groups retire before replacement.  replacement_interval also preserves
    the global II when I>L; final completion itself needs no extra II tail.
    """
    if records < 1 or k_segments < 1:
        raise ValueError("positive group record and segment counts required")
    interval = timing.initiation_interval
    latency = timing.completion_latency(core)
    last_issue = (k_segments - 1) * max(records * interval, latency) + (records - 1) * interval
    return GroupCost(first_record, records, records * k_segments,
                     last_issue + latency, last_issue + max(latency, interval),
                     records * core.record_bytes)


def projection_groups(m: int, n: int, k: int, core: Core,
                      timing: TimingProfile = DEFAULT_TIMING,
                      limits: ContextLimits = ContextLimits()) -> tuple[GroupCost, ...]:
    """N-major/M-minor consecutive output records, bounded before K issues.

    Record r has N-tile r//ceil(M/PM), M-tile r%ceil(M/PM).  A group's
    records may cross an N boundary; operand reuse/traffic must follow that
    same order in a supply model.  Tail records occupy full physical slots.
    """
    if min(m, n, k) < 1:
        raise ValueError("positive logical GEMM dimensions required")
    count = ceil(m, core.pm) * ceil(n, core.pn)
    bound = limits.capacity(core)
    segments = ceil(k, core.pk)
    return tuple(group_cost(min(bound, count - first), segments, core, timing, first)
                 for first in range(0, count, bound))


@lru_cache(maxsize=262_144)
def projection(m: int, n: int, k: int, core: Core,
               timing: TimingProfile = DEFAULT_TIMING,
               limits: ContextLimits = ContextLimits()) -> ProjectionCost:
    """Compute cost with exact ceil/padding counts and finite partial outputs.

    Gate/Up must use paired_gate_up rather than an all-Gate/all-Up
    concatenation whose retired groups lose needed paired values.  First
    K segments start from zero, so no old accumulator read is counted.
    Accumulator traffic counts padded physical records, not only real tails.
    """
    if min(m, n, k) < 1:
        raise ValueError("positive logical GEMM dimensions required")
    nm, nn, nk = ceil(m, core.pm), ceil(n, core.pn), ceil(k, core.pk)
    records = nm * nn
    bound = limits.capacity(core)
    full, tail = divmod(records, bound)
    groups = full + bool(tail)
    q = min(records, bound)
    full_group = group_cost(bound, nk, core, timing)
    # Sum replacement intervals for all but the last group, then its commit.
    if tail:
        cycles = full * full_group.replacement_interval + group_cost(tail, nk, core, timing).cycles
    else:
        cycles = (full - 1) * full_group.replacement_interval + full_group.cycles
    issues = records * nk
    issued = issues * core.macs
    useful = m * n * k
    return ProjectionCost(
        cycles, issues, useful, issued, issued - useful, nm, nn, nk, groups,
        bound, q, core.record_bytes, q * core.record_bytes,
        core.x_slice_bytes, core.w_slice_bytes,
        records * (nk - 1) * core.record_bytes, issues * core.record_bytes,
        timing.dot_latency(core), timing.completion_latency(core), timing.initiation_interval,
        m - (nm - 1) * core.pm, n - (nn - 1) * core.pn, k - (nk - 1) * core.pk)


@lru_cache(maxsize=262_144)
def paired_gate_up(m: int, f: int, h: int, core: Core,
                   timing: TimingProfile = DEFAULT_TIMING,
                   limits: ContextLimits = ContextLimits()) -> ProjectionCost:
    """Two matching F-width projections sharing one finite paired group.

    Execute Gate group, then matching Up group, retaining both FP32 tiles
    until the caller's SiLU/elementwise product consumes them and produces
    BF16 Z.  Each F tail is padded independently.  A resident *paired*
    record occupies 2*PM*PN*4 bytes.  The returned m/n/k ceil counts and
    group count describe one projection/the paired groups; issue and MAC
    counts and accumulator traffic describe both projections.

    This cost excludes vector/Z service, and cannot be used to free Gate
    partial sums before the corresponding Up group has been consumed.
    """
    per_projection = ContextLimits(limits.max_records, limits.accumulator_bytes // 2)
    cost = projection(m, f, h, core, timing, per_projection)
    # Paired execution has 2*g serial projection groups and 2*g-1 II gaps.
    extra_ii = max(0, timing.initiation_interval - timing.completion_latency(core))
    return replace(cost, cycles=2 * cost.cycles + extra_ii,
                   issues=2 * cost.issues, useful_macs=2 * cost.useful_macs,
                   issued_macs=2 * cost.issued_macs, padding_macs=2 * cost.padding_macs,
                   record_bytes=2 * cost.record_bytes,
                   peak_accumulator_bytes=2 * cost.peak_accumulator_bytes,
                   accumulator_read_bytes=2 * cost.accumulator_read_bytes,
                   accumulator_write_bytes=2 * cost.accumulator_write_bytes)


def resource_requirements(core: Core, timing: TimingProfile = DEFAULT_TIMING,
                          limits: ContextLimits = ContextLimits()) -> dict[str, int | str]:
    """Installed full-slice capacity and ports needed for the hypothesized II.

    Double buffering and finite resident outputs are explicitly charged.
    These byte-rate requirements do not establish bank feasibility, timing,
    area or energy.  Shared SRAM paths may make the realized II larger.
    """
    records = limits.capacity(core)
    interval = timing.initiation_interval
    commit = timing.commit_cycles
    return {
        "precision": "BF16 inputs/weights; FP32 partial sums",
        "main_multipliers": core.macs,
        "x_double_buffer_bytes": 2 * core.x_slice_bytes,
        "weight_double_buffer_bytes": 2 * core.w_slice_bytes,
        "resident_records": records,
        "resident_partial_sum_bytes": records * core.record_bytes,
        "x_array_bytes_per_issue": core.x_slice_bytes,
        "weight_array_bytes_per_issue": core.w_slice_bytes,
        "accumulator_rmw_bytes_per_followup_issue": 2 * core.record_bytes,
        "x_array_required_bytes_per_cycle": ceil(core.x_slice_bytes, interval),
        "weight_array_required_bytes_per_cycle": ceil(core.w_slice_bytes, interval),
        "accumulator_required_bytes_per_cycle": max(ceil(2 * core.record_bytes, interval),
                                                   ceil(2 * core.record_bytes, commit)),
        "timing_scope": "requirements only; no physical port/bank or clock closure",
    }
