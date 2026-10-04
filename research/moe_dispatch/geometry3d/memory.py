"""Finite storage, native BF16 addressing and shared supply for prospective 3-D DSE.

This is an aggregate HBM/endpoint model, not Ramulator or implemented RTL.
Logical reduction K, physical PK, and the existing 4-column/512-K wire layout
are independent.  All addresses below refer to that single existing layout.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import heapq
import math
from typing import Any, Iterable

SRAM_BYTES = 2_158_592
TOTAL_MACS = 12_288
TRANSACTION_BYTES = 32
NATIVE_N = 4
NATIVE_K = 512
BANK_WORD_BYTES = 16


def ceildiv(n: int, d: int) -> int:
    if d <= 0:
        raise ValueError("positive divisor required")
    return (n + d - 1) // d


def align(n: int, granularity: int = TRANSACTION_BYTES) -> int:
    return ceildiv(n, granularity) * granularity


def _dims(core: Any) -> tuple[int, int, int]:
    if isinstance(core, (tuple, list)):
        out = tuple(core)
    else:
        out = (core.pm, core.pn, core.pk)
    if len(out) != 3 or min(out) <= 0:
        raise ValueError("positive PM, PN, PK required")
    return out


@dataclass(frozen=True)
class WeightSlice:
    transactions: tuple[int, ...]
    useful_bytes: int
    wire_bytes: int
    splice_read_bytes: int
    splice_write_bytes: int
    native_tiles: int

    @property
    def spans(self) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        for a in self.transactions:
            if spans and spans[-1][0] + spans[-1][1] == a:
                spans[-1] = (spans[-1][0], spans[-1][1] + TRANSACTION_BYTES)
            else:
                spans.append((a, TRANSACTION_BYTES))
        return tuple(spans)


@dataclass(frozen=True)
class NativeWeightLayout:
    """BF16 W[N,K], each real row independently padded to a 32-byte stride.

    N=4/K=512 are descriptor granularity, not extra wire padding. ``v3_tiles``
    explicitly selects the separate compiler-v3 K-segment/N-block W[4,kv]
    layout as a sensitivity. Splicing charges the native
    transaction read and packed useful-byte write; padding fills are supplied
    by zero generation and occupy the physical W slot, not an HBM alias.
    """
    n: int
    k: int
    base: int = 0
    layout: str = "row32"

    def __post_init__(self) -> None:
        if min(self.n, self.k) <= 0 or self.base < 0 or self.base % 32 or self.layout not in ("row32", "v3_tiles"):
            raise ValueError("positive logical dimensions and aligned base required")

    @property
    def storage_bytes(self) -> int:
        if self.layout == "row32":
            return self.n * align(self.k * 2)
        return ceildiv(self.n, 4) * sum(
            align(4 * min(512, self.k - s) * 2)
            for s in range(0, self.k, 512))

    def tile_base(self, n_block: int, k_segment: int) -> int:
        if not (0 <= n_block < ceildiv(self.n, 4) and
                0 <= k_segment < ceildiv(self.k, 512)):
            raise ValueError("native tile outside tensor")
        if self.layout == "row32":
            return self.base + n_block * 4 * align(self.k * 2) + k_segment * 512 * 2
        before = sum(align(4 * min(512, self.k - s) * 2)
                     for s in range(0, k_segment * 512, 512))
        kv = min(512, self.k - k_segment * 512)
        return self.base + ceildiv(self.n, 4) * before + n_block * align(4 * kv * 2)

    @lru_cache(maxsize=128)
    def tile(self, n0: int, cols: int, k0: int, kv: int) -> WeightSlice:
        if min(cols, kv) <= 0 or min(n0, k0) < 0 or n0 + cols > self.n or k0 + kv > self.k:
            raise ValueError("physical slice outside logical tensor")
        tx: set[int] = set()
        tiles: set[tuple[int, int]] = set()
        for col in range(n0, n0 + cols):
            if self.layout == "row32":
                start = self.base + col * align(self.k * 2) + k0 * 2
                end = start + kv * 2
                tx.update(range(start // 32 * 32, align(end), 32))
                tiles.update((col // 4, ks) for ks in range(k0 // 512, (k0 + kv - 1) // 512 + 1))
                continue
            for ks in range(k0 // 512, (k0 + kv - 1) // 512 + 1):
                native_kv = min(512, self.k - ks * 512)
                lo = max(k0, ks * 512) - ks * 512
                hi = min(k0 + kv, (ks + 1) * 512) - ks * 512
                base = self.tile_base(col // 4, ks) + (col % 4) * native_kv * 2
                start, end = base + lo * 2, base + hi * 2
                tx.update(range(start // 32 * 32, align(end), 32))
                tiles.add((col // 4, ks))
        useful = cols * kv * 2
        wire = len(tx) * 32
        return WeightSlice(tuple(sorted(tx)), useful, wire, wire, align(useful, 16), len(tiles))

    @staticmethod
    def coalesce(slices: Iterable[WeightSlice]) -> WeightSlice:
        items = tuple(slices)
        tx = tuple(sorted({a for item in items for a in item.transactions}))
        useful = sum(x.useful_bytes for x in items)
        # This helper requires disjoint logical slices. Overlapping requested
        # weights must retain their consumer references rather than be copied twice.
        return WeightSlice(tx, useful, len(tx) * 32, len(tx) * 32,
                           align(useful, 16), sum(x.native_tiles for x in items))


@dataclass(frozen=True)
class FabricProfile:
    hbm_bytes_per_cycle: int = 256
    hbm_latency_cycles: int = 64
    hbm_credits: int = 256
    landing_bytes_per_cycle: int = 256
    landing_pool_bytes: int = 40_960
    ingress_bytes: int = 8_192
    control_bytes: int = 16_384
    w_banks: int = 64
    x_banks: int = 24
    accumulator_banks: int = 12

    def __post_init__(self) -> None:
        if min(self.hbm_bytes_per_cycle, self.hbm_credits, self.landing_bytes_per_cycle,
               self.landing_pool_bytes, self.w_banks, self.x_banks, self.accumulator_banks) <= 0:
            raise ValueError("finite positive resources required")
        if self.hbm_latency_cycles < 0 or self.hbm_bytes_per_cycle % 32 or self.landing_bytes_per_cycle % 32:
            raise ValueError("HBM and landing grant 32-byte transactions")

    @property
    def credit_bandwidth_upper_bound(self) -> float:
        return min(self.hbm_bytes_per_cycle,
                   self.hbm_credits * 32 / max(1, self.hbm_latency_cycles))

    @property
    def landing_credit_bandwidth_upper_bound(self) -> float:
        # A response is committed by the landing port over at least one cycle.
        return min(self.hbm_bytes_per_cycle, self.landing_bytes_per_cycle,
                   self.hbm_credits * 32 / max(1, self.hbm_latency_cycles + 1))


@dataclass(frozen=True)
class CoreMemory:
    pm: int
    pn: int
    pk: int
    w_slots: int
    w_slot_bytes: int
    x_register_bytes: int
    accumulator_bytes: int
    z_bytes: int
    w_banks: int
    x_banks: int
    accumulator_banks: int
    group_tiles: int
    w_capacity_bytes: int = 0
    x_buffers: int = 0
    eligible: bool = True
    ineligibility_reason: str = ""

    @property
    def w_bytes(self) -> int:
        return self.w_capacity_bytes or self.w_slots * self.w_slot_bytes

    @property
    def w_bandwidth(self) -> int:
        return self.w_banks * 16

    @property
    def x_bandwidth(self) -> int:
        return self.x_banks * 16

    @property
    def accumulator_bandwidth(self) -> int:
        return self.accumulator_banks * 16


@dataclass(frozen=True)
class BudgetReport:
    cores: tuple[CoreMemory, ...]
    structures: dict[str, int]
    capacity_bytes: int
    total_bytes: int
    slack_bytes: int
    fits: bool
    profile: FabricProfile
    z_mode: str
    max_me: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def allocate_banks(total: int, weights: Iterable[int]) -> tuple[int, ...]:
    """Positive integer partition; all banks are allocated exactly once."""
    ws = tuple(weights)
    if not ws or min(ws) <= 0 or total < len(ws):
        raise ValueError("at least one bank per core required")
    remain = total - len(ws)
    ideals = [remain * w / sum(ws) for w in ws]
    vals = [1 + math.floor(x) for x in ideals]
    for i in sorted(range(len(ws)), key=lambda i: (-(ideals[i] % 1), i))[:total - sum(vals)]:
        vals[i] += 1
    return tuple(vals)


def memory_budget(cores: Iterable[Any], batch: int, hidden: int, max_f: int,
                  max_me: Iterable[int] | int | None = None, top_k: int = 6,
                  expert_count: int = 64, flows: Iterable[str] | None = None,
                  group_tiles: int = 8, z_mode: str = "full",
                  profile: FabricProfile = FabricProfile(),
                  bank_weights: Iterable[int] | None = None,
                  allocation: str = "proportional", buffer_limit: int = 32) -> BudgetReport:
    """Frozen installed partition, identical for every model/batch/family.

    X=512KiB, FP32 combined Y=1024KiB, Z=384KiB, private W=40KiB,
    ingress=8KiB, local X=12KiB, local accumulator/RF=96KiB, control=16KiB,
    routes=16KiB. All private capacities and physical banks partition once.
    The workload arguments describe live capacity checks, never resize SRAM.
    One-buffer designs are legal but cannot overlap their W fill and last use.
    W slot descriptors (32/core, 24B each) and controller state occupy control.
    """
    dims = tuple(_dims(c) for c in cores)
    if sum(m * n * k for m, n, k in dims) != TOTAL_MACS:
        raise ValueError("fixed 12,288 multiplier budget required")
    if min(batch, hidden, max_f, group_tiles) <= 0 or z_mode not in ("full", "streamed"):
        raise ValueError("positive workload dimensions and known Z mode required")
    if max_me is None:
        mes = (batch,) * len(dims)
    elif isinstance(max_me, int):
        mes = (max_me,) * len(dims)
    else:
        mes = tuple(max_me)
    if len(mes) != len(dims) or min(mes) <= 0 or max(mes) > batch:
        raise ValueError("one valid maximum Me per core required")
    flow_list = tuple(flows or ("ws",) * len(dims))
    if len(flow_list) != len(dims) or any(f not in ("ws", "is", "os") for f in flow_list):
        raise ValueError("one declared dataflow per core required")
    if allocation not in ("proportional", "equal") or buffer_limit <= 0 or buffer_limit > 32:
        raise ValueError("declared allocation and buffer limit 1..32 required")
    weights = tuple(bank_weights or ([m * n * k for m, n, k in dims]
                                   if allocation == "proportional" else [1] * len(dims)))
    if len(weights) != len(dims):
        raise ValueError("bank partition length differs from core count")
    wbank = allocate_banks(profile.w_banks, weights)
    xbank = allocate_banks(profile.x_banks, weights)
    abank = allocate_banks(profile.accumulator_banks, weights)
    def partition(byte_count: int) -> tuple[int, ...]:
        return tuple(x * 32 for x in allocate_banks(byte_count // 32, weights))
    wquotas = partition(40 * 1024)
    xquotas = partition(12 * 1024)
    aquotas = partition(96 * 1024)
    # Even a small core retains one routed/shared row; this minimum is part
    # of, rather than additional to, the frozen 384-KiB pool.
    zminimum = 8 * 1024
    zquotas = tuple(zminimum + x * 32 for x in allocate_banks(
        (384 * 1024 - len(dims) * zminimum) // 32, weights))
    structures = {"ingress_fifo": 8 * 1024,
                  "control_metadata": 16 * 1024,
                  "route_state": 16 * 1024,
                  "X_activation": 512 * 1024,
                  "combine_fp32": 1024 * 1024}
    entries = []
    for i, ((pm, pn, pk), me, flow) in enumerate(zip(dims, mes, flow_list)):
        wslot = align(pn * pk * 2)
        slots = min(buffer_limit, wquotas[i] // wslot)
        xbuffers = min(2, xquotas[i] // align(pm * pk * 2))
        reason = "" if slots and xbuffers else ("private W cannot hold one physical tile" if not slots else "local X cannot hold one physical M/PK slice")
        cm = CoreMemory(pm, pn, pk, slots, wslot, xquotas[i], aquotas[i], zquotas[i],
                        wbank[i], xbank[i], abank[i], group_tiles,
                        wquotas[i], xbuffers, not bool(reason), reason)
        entries.append(cm)
        structures.update({f"core{i}_W_slots": cm.w_bytes,
                           f"core{i}_X_buffers": cm.x_register_bytes,
                           f"core{i}_accumulator_RF": cm.accumulator_bytes,
                           f"core{i}_Z_activation": cm.z_bytes})
    # The SRAM budget includes logical operand RF bytes. Real area/power needs
    # synthesis; capacity equality alone does not imply equal total chip area.
    total = sum(structures.values())
    globals_fit = (batch * hidden * 2 <= 512 * 1024 and batch * hidden * 4 <= 1024 * 1024
                   and batch * top_k * 16 + expert_count * 64 <= 16 * 1024)
    metadata_fit = len(dims) * 32 * 24 + 4288 <= 16 * 1024
    return BudgetReport(tuple(entries), structures, SRAM_BYTES, total,
                        SRAM_BYTES - total, total <= SRAM_BYTES and globals_fit
                        and metadata_fit and all(c.eligible for c in entries),
                        profile, z_mode, mes)


@dataclass(frozen=True)
class TrafficReport:
    issues: int
    useful_macs: int
    issued_macs: int
    weight_hbm_bytes: int
    unique_weight_storage_bytes: int
    weight_splice_read_bytes: int
    weight_splice_write_bytes: int
    weight_operand_read_bytes: int
    x_sram_read_bytes: int
    x_operand_read_bytes: int
    accumulator_read_bytes: int
    accumulator_write_bytes: int
    output_write_bytes: int
    m_chunks: int
    n_groups: int
    k_segments: int
    dataflow: str

    @property
    def total_onchip_bytes(self) -> int:
        return sum((self.weight_splice_read_bytes, self.weight_splice_write_bytes,
                    self.weight_operand_read_bytes, self.x_sram_read_bytes,
                    self.x_operand_read_bytes, self.accumulator_read_bytes,
                    self.accumulator_write_bytes, self.output_write_bytes))


@lru_cache(maxsize=8192)
def _weight_group_bytes(n: int, k: int, pn: int, pk: int, group_tiles: int) -> int:
    # Each row has an aligned base. Distinct N groups cannot share a sector.
    # Non-16-aligned PK may refetch a boundary sector; supported PK32..1024
    # never does, including the valid logical K tail.
    return n * sum(align(min(k, k0 + pk) * 2) - (k0 * 2 // 32 * 32)
                   for k0 in range(0, k, pk))


def projection_traffic(m: int, n: int, k: int, core: Any, flow: str = "ws",
                       group_tiles: int = 8, x_capacity_bytes: int | None = None,
                       acc_capacity_bytes: int | None = None,
                       spill_capacity_bytes: int = 0) -> TrafficReport:
    """Exact bytes for declared bounded WS, IS and OS loop nests.

    WS: Ngroup,K,Mchunk; one weight group serves all admitted rows. IS:
    Mchunk,K,N; one X chunk serves N, weights repeat per finite X chunk.
    OS: Mblock,Ngroup,K; output group stays in RF, weights repeat per Mblock.
    When WS rows exceed accumulator capacity, groups execute in row chunks
    and weights reload. IS partial outputs use SRAM unless the whole output
    chunk fits; every intermediate K write/read is charged explicitly.
    """
    pm, pn, pk = _dims(core)
    if min(m, n, k, group_tiles) <= 0 or flow not in ("ws", "is", "os"):
        raise ValueError("positive dimensions and ws/is/os dataflow required")
    nm, nn, nk = ceildiv(m, pm), ceildiv(n, pn), ceildiv(k, pk)
    ng = ceildiv(nn, group_tiles)
    issues = nm * nn * nk
    unique = NativeWeightLayout(n, k).storage_bytes
    if flow == "ws":
        # One output Ngroup of all rows must remain live across K segments.
        rows_fit = m if acc_capacity_bytes is None else acc_capacity_bytes // (pn * group_tiles * 4)
        rows_fit = min(m, max(pm, rows_fit // pm * pm))
        if acc_capacity_bytes is not None and acc_capacity_bytes < pm * pn * group_tiles * 4:
            raise ValueError("accumulator cannot hold one WS output group")
        chunks = ceildiv(m, rows_fit)
        weight = chunks * _weight_group_bytes(n, k, pn, pk, group_tiles)
        xread = ng * m * k * 2
        accread = accwrite = 0
    elif flow == "is":
        rows_fit = m if x_capacity_bytes is None else x_capacity_bytes // (pk * 2)
        rows_fit = min(m, rows_fit // pm * pm if rows_fit >= pm else rows_fit)
        if rows_fit < min(pm, m) or (x_capacity_bytes is not None and x_capacity_bytes < pm * pk * 2):
            raise ValueError("X store cannot hold one IS M block")
        chunks = ceildiv(m, rows_fit)
        weight = chunks * _weight_group_bytes(n, k, pn, pk, 1)
        xread = m * k * 2
        whole = rows_fit * n * 4
        spills = acc_capacity_bytes is not None and acc_capacity_bytes < whole
        if spills and spill_capacity_bytes < m * n * 4:
            raise ValueError("IS partial outputs need an explicitly allocated finite spill backing")
        accread = m * n * 4 * (nk - 1) if spills else 0
        accwrite = accread
    else:
        chunks = nm
        weight = nm * _weight_group_bytes(n, k, pn, pk, group_tiles)
        xread = ng * m * k * 2
        accread = accwrite = 0
        if acc_capacity_bytes is not None and acc_capacity_bytes < pm * pn * group_tiles * 4:
            raise ValueError("accumulator cannot hold one OS output group")
    # Native transport -> splice -> finite packed W registers. These copies
    # are charged independently of arithmetic broadcasts on every issued tile.
    weight_useful = (chunks * n * k * 2)
    # Every physical result crosses the same finite local accumulator port;
    # these include RF accesses even when no SRAM spill is required.
    accread += nm * nn * (nk - 1) * pm * pn * 4
    accwrite += issues * pm * pn * 4
    return TrafficReport(issues, m * n * k, issues * pm * pn * pk,
                         weight, unique, weight, align(weight_useful, 16),
                         issues * pn * pk * 2, xread, issues * pm * pk * 2,
                         accread, accwrite, m * n * 4, chunks, ng, nk, flow)


@dataclass(frozen=True)
class RecordGroupTraffic:
    first_record: int
    records: int
    issues: int
    weight_hbm_bytes: int
    weight_splice_read_bytes: int
    weight_splice_write_bytes: int
    x_sram_read_bytes: int
    accumulator_read_bytes: int
    accumulator_write_bytes: int
    output_write_bytes: int
    unique_weight_tiles: int
    unique_x_slices: int


def record_group_traffic(m: int, n: int, k: int, core: Any,
                         first_record: int, records: int,
                         weight_reuse: bool = True, x_reuse: bool = True) -> RecordGroupTraffic:
    """Bytes matching compute.py's N-major/M-minor bounded record group.

    Each K segment fills one W slice per distinct N tile and one X slice per
    distinct M tile, if their operand references survive all relevant issues.
    Callers with insufficient slots must set reuse=False or split the group;
    this helper never reserves an unbounded weight or X working set itself.
    A paired Gate/Up group calls this twice using distinct tensor addresses.
    """
    pm, pn, pk = _dims(core)
    nm, nn, nk = ceildiv(m, pm), ceildiv(n, pn), ceildiv(k, pk)
    if min(m, n, k, records) <= 0 or first_record < 0 or first_record + records > nm * nn:
        raise ValueError("record group outside projection")
    indices = tuple(range(first_record, first_record + records))
    ns = tuple(dict.fromkeys(r // nm for r in indices))
    ms = tuple(dict.fromkeys(r % nm for r in indices))
    n_refs = ns if weight_reuse else tuple(r // nm for r in indices)
    m_refs = ms if x_reuse else tuple(r % nm for r in indices)
    wire = writes = xread = 0
    for k0 in range(0, k, pk):
        kv = min(pk, k - k0)
        sector_row_bytes = align((k0 + kv) * 2) - k0 * 2 // 32 * 32
        for nt in n_refs:
            cols = min(pn, n - nt * pn)
            wire += cols * sector_row_bytes
            writes += align(cols * kv * 2, 16)
        for mt in m_refs:
            rows = min(pm, m - mt * pm)
            # Actual SRAM reads are individually 16-byte row aligned; padded
            # operand rows are local zeros but occupy PM*PK*2 physical bytes.
            xread += rows * align(kv * 2, 16)
    output = sum(min(pm, m - (r % nm) * pm) * min(pn, n - (r // nm) * pn) * 4 for r in indices)
    return RecordGroupTraffic(first_record, records, records * nk, wire, wire,
                              writes, xread, records * (nk - 1) * pm * pn * 4,
                              records * nk * pm * pn * 4, output,
                              len(n_refs) * nk, len(m_refs) * nk)


@dataclass
class SharedPort:
    """A finite aggregate endpoint port shared by every core and operation."""
    bytes_per_cycle: int
    free_cycle: int = 0
    transferred_bytes: int = 0

    def reserve(self, at: int, byte_count: int) -> int:
        if min(at, byte_count) < 0 or self.bytes_per_cycle <= 0:
            raise ValueError("valid time, bytes and finite port required")
        start = max(at, self.free_cycle)
        self.free_cycle = start + ceildiv(byte_count, self.bytes_per_cycle)
        self.transferred_bytes += byte_count
        return self.free_cycle


@dataclass
class HBMEndpoint:
    """Explicit approximate endpoint screening server, shared across all cores.

    A continuous credit window caps its service rate; fixed latency precedes
    response readiness, and endpoint service is queued on one shared bus.
    It has no transaction arbitration, bank conflicts, or operand lookahead.
    ``SharedHBM`` is the finer finite-request validation model. The runner must
    retain a one-slot W operand through its last arithmetic use and serialize
    the following fill; this class cannot invent a second buffer.
    """
    profile: FabricProfile = field(default_factory=FabricProfile)
    free_cycle: int = 0
    transferred_bytes: int = 0
    reservations: list[dict[str, Any]] = field(default_factory=list)

    def reserve(self, at: int, byte_count: int, buffer_bytes: int | None = None,
                tag: Any = None) -> int:
        if min(at, byte_count) < 0 or byte_count % 32:
            raise ValueError("nonnegative time and aligned HBM bytes required")
        if byte_count == 0:
            return at
        hold = max(1, self.profile.hbm_latency_cycles + 1)
        bw = self.profile.landing_credit_bandwidth_upper_bound
        if buffer_bytes is not None:
            if buffer_bytes <= 0:
                raise ValueError("finite positive operand buffer required")
            bw = min(bw, buffer_bytes / hold)
        start = max(at + self.profile.hbm_latency_cycles, self.free_cycle)
        finish = start + math.ceil(byte_count / bw)
        self.free_cycle = finish
        self.transferred_bytes += byte_count
        self.reservations.append({"tag": tag, "request_cycle": at,
                                  "service_start": start, "finish_cycle": finish,
                                  "bytes": byte_count, "effective_bytes_per_cycle": bw})
        return finish

    def report(self) -> dict[str, Any]:
        return {"wire_bytes": self.transferred_bytes,
                "last_response_cycle": self.free_cycle,
                "scope": "continuous-credit endpoint screening approximation",
                "credit_bandwidth_upper_bound": self.profile.credit_bandwidth_upper_bound,
                "landing_credit_bandwidth_upper_bound": self.profile.landing_credit_bandwidth_upper_bound}


@dataclass
class HBMTransfer:
    tag: Any
    core: int
    at: int
    transactions: int
    reserved_bytes: int
    accepted: int = 0
    landed: int = 0
    reservation_active: bool = False
    ready_cycle: int | None = None
    consumed: bool = False


class SharedHBM:
    """Cycle/event server for shared 32-B requests and finite landing leases.

    Submit transfers before advancing through their timestamps. ``advance``
    returns newly ready handles. The consumer calls ``consume`` only after the
    last pool read, so queued cores cannot silently borrow live landing bytes.
    Requests round-robin among eligible cores. Credits retire at *landing
    commit*, one cycle after the landing write starts, never on HBM response.
    Physical per-bank address conflicts are outside this aggregate server.
    """
    def __init__(self, profile: FabricProfile = FabricProfile(),
                 per_core_capacity_bytes: Iterable[int] | None = None) -> None:
        self.profile = profile
        self.per_core_capacity_bytes = tuple(per_core_capacity_bytes or ())
        if self.per_core_capacity_bytes and (min(self.per_core_capacity_bytes) <= 0 or
                sum(self.per_core_capacity_bytes) != profile.landing_pool_bytes):
            raise ValueError("private landing quotas must partition the single pool")
        self.per_core_used = [0] * len(self.per_core_capacity_bytes)
        self.now = 0
        self.transfers: list[HBMTransfer] = []
        self.inflight: list[tuple[int, int, int]] = []
        self.returned: deque[tuple[int, int]] = deque()
        self.commits: list[tuple[int, int, int]] = []
        self.releases: list[tuple[int, int]] = []
        self.credit_used = self.credit_peak = 0
        self.pool_used = self.pool_peak = 0
        self.accepted_transactions = self.landed_transactions = 0
        self.last_core = -1
        self.ingress_peak_bytes = 0
        self.response_backpressure_cycles = 0

    def submit(self, core: int, at: int, transactions: int | Iterable[int],
               tag: Any = None, reservation_bytes: int | None = None) -> HBMTransfer:
        count = transactions if isinstance(transactions, int) else len(set(transactions))
        # The current cycle has just been processed by advance(). A consumer
        # may enqueue at that boundary; its first request is granted next cycle.
        if min(core, at) < 0 or count <= 0 or at < max(0, self.now - 1):
            raise ValueError("submit positive transactions before their event time")
        reserved = align(count * 32 if reservation_bytes is None else reservation_bytes)
        if reserved < count * 32 or reserved > self.profile.landing_pool_bytes:
            raise ValueError("transfer must fit a finite full landing reservation")
        if self.per_core_capacity_bytes and (core >= len(self.per_core_capacity_bytes)
                or reserved > self.per_core_capacity_bytes[core]):
            raise ValueError("transfer cannot fit its private landing quota")
        handle = HBMTransfer(tag if tag is not None else len(self.transfers), core, at, count, reserved)
        self.transfers.append(handle)
        return handle

    def consume(self, handle: HBMTransfer, at: int | None = None) -> None:
        if handle.ready_cycle is None or handle.consumed:
            raise ValueError("release only a ready, live landing lease")
        if at is not None and at < handle.ready_cycle:
            raise ValueError("release precedes readiness")
        handle.consumed = True
        if at is not None and at >= self.now:
            heapq.heappush(self.releases, (at, self.transfers.index(handle)))
        else:
            self._release(handle)

    def _release(self, handle: HBMTransfer) -> None:
        self.pool_used -= handle.reserved_bytes
        if self.per_core_used:
            self.per_core_used[handle.core] -= handle.reserved_bytes
        handle.reservation_active = False

    def _eligible(self, t: HBMTransfer) -> bool:
        return (t.at <= self.now and t.accepted < t.transactions and
                (t.reservation_active or (self.pool_used + t.reserved_bytes <= self.profile.landing_pool_bytes
                 and (not self.per_core_used or self.per_core_used[t.core] + t.reserved_bytes <= self.per_core_capacity_bytes[t.core]))))

    def next_event_time(self) -> int | None:
        events = [x[0] for x in self.inflight] + [x[0] for x in self.commits] + [x[0] for x in self.releases]
        if self.returned or (self.credit_used < self.profile.hbm_credits and any(self._eligible(t) for t in self.transfers)):
            events.append(self.now)
        events.extend(t.at for t in self.transfers if t.at > self.now and t.accepted < t.transactions)
        return min(events) if events else None

    def advance(self, until: int) -> list[HBMTransfer]:
        if until < self.now:
            raise ValueError("HBM event time must be monotonic")
        ready: list[HBMTransfer] = []
        while self.now <= until:
            while self.releases and self.releases[0][0] <= self.now:
                _, idx = heapq.heappop(self.releases)
                self._release(self.transfers[idx])
            while self.commits and self.commits[0][0] <= self.now:
                _, idx, count = heapq.heappop(self.commits)
                t = self.transfers[idx]
                t.landed += count
                self.credit_used -= count
                self.landed_transactions += count
                if t.landed == t.transactions:
                    t.ready_cycle = self.now
                    ready.append(t)
            ingress_used = sum(count for _, count in self.returned) * 32
            while self.inflight and self.inflight[0][0] <= self.now:
                if ingress_used + self.inflight[0][2] * 32 > self.profile.ingress_bytes:
                    self.response_backpressure_cycles += 1
                    break
                _, idx, count = heapq.heappop(self.inflight)
                self.returned.append((idx, count))
                ingress_used += count * 32
                self.ingress_peak_bytes = max(self.ingress_peak_bytes, ingress_used)
            land_left = self.profile.landing_bytes_per_cycle // 32
            while self.returned and land_left:
                idx, count = self.returned.popleft()
                grant = min(land_left, count)
                heapq.heappush(self.commits, (self.now + 1, idx, grant))
                if count > grant:
                    self.returned.appendleft((idx, count - grant))
                land_left -= grant
            grants = min(self.profile.hbm_bytes_per_cycle // 32,
                         self.profile.hbm_credits - self.credit_used)
            while grants:
                choices = [(i, t) for i, t in enumerate(self.transfers) if self._eligible(t)]
                if not choices:
                    break
                cores = sorted({t.core for _, t in choices})
                chosen_core = next((c for c in cores if c > self.last_core), cores[0])
                idx, t = next((i, t) for i, t in choices if t.core == chosen_core)
                if not t.reservation_active:
                    self.pool_used += t.reserved_bytes
                    if self.per_core_used:
                        self.per_core_used[t.core] += t.reserved_bytes
                    self.pool_peak = max(self.pool_peak, self.pool_used)
                    t.reservation_active = True
                # One request grant per core turn preserves transaction-level fairness.
                t.accepted += 1
                self.accepted_transactions += 1
                self.credit_used += 1
                self.credit_peak = max(self.credit_peak, self.credit_used)
                heapq.heappush(self.inflight, (self.now + self.profile.hbm_latency_cycles, idx, 1))
                self.last_core = chosen_core
                grants -= 1
            self.now += 1
            future = self.next_event_time()
            if future is None or future > until:
                self.now = until + 1
                break
            if future > self.now:
                self.now = future
        return ready

    def report(self) -> dict[str, Any]:
        return {"accepted_transactions": self.accepted_transactions,
                "landed_transactions": self.landed_transactions,
                "wire_bytes": self.accepted_transactions * 32,
                "credit_used": self.credit_used, "credit_peak": self.credit_peak,
                "landing_live_bytes": self.pool_used, "landing_peak_bytes": self.pool_peak,
                "ingress_peak_bytes": self.ingress_peak_bytes,
                "response_backpressure_cycles": self.response_backpressure_cycles,
                "nominal_hbm_bytes_per_cycle": self.profile.hbm_bytes_per_cycle,
                "credit_bandwidth_upper_bound": self.profile.credit_bandwidth_upper_bound,
                "landing_credit_bandwidth_upper_bound": self.profile.landing_credit_bandwidth_upper_bound,
                "scope": "fixed-latency aggregate HBM and finite landing leases; no channel/row or bank-conflict timing"}
