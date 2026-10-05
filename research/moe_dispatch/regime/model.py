"""Finite-resource fluid analytical screening, independent of frozen v3.

This is NOT cycle-exact RTL/Ramulator. Each bounded output group retains its
partial sums; phase-level fluid progress approximates overlap among compute,
ports and shared supply. Fixed phase barriers make this a prospective design
comparison, not a rigorous lower/upper bound on the existing simulator.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from functools import lru_cache
import heapq
import math
from ..geometry3d.compute import Core, TimingProfile, DEFAULT_TIMING, ceil, group_cost
from ..geometry3d.memory import CoreMemory, FabricProfile
from .resources import Partition, memory_budget, packed_projection_bytes, unique_bytes


@dataclass(frozen=True)
class Settings:
    valid_operand_traffic: bool = False  # False preserves frozen-v6 padded-port hypothesis
    gu_group_limit: int = 0
    down_group_limit: int = 0
    partition: Partition | None = None
    weight_format: str = "BF16"
    decoder_elements_per_cycle: int = 0  # zero is the explicitly ideal decoder oracle
    timing: TimingProfile = DEFAULT_TIMING
    allocation: str = "proportional"
    prefetch_slots: int = 32
    records: int = 8
    hbm: bool = True
    ports: bool = True
    control: bool = True
    flow: str = "bounded_ws"
    fabric: FabricProfile = FabricProfile()

    def __post_init__(self):
        if not (0<=self.gu_group_limit<=32 and 0<=self.down_group_limit<=32):
            raise ValueError('nonnegative bounded compiler N groups required')
        if self.weight_format not in ("BF16", "W8", "W4") or self.decoder_elements_per_cycle < 0:
            raise ValueError("declared weight format and nonnegative decoder throughput required")
        if self.flow not in ("bounded_ws", "bounded_os") or not 1 <= self.records <= 32:
            raise ValueError("declared WS/OS and positive bounded descriptors required")


@dataclass(frozen=True)
class Phase:
    name: str
    compute: int
    issues: int
    useful_macs: int
    issued_macs: int
    hbm_bytes: int
    weight_unique: int
    w_port_bytes: int
    x_port_bytes: int
    acc_port_bytes: int
    activation_bytes: int
    vector_elements: int
    control_cycles: int
    base_cycles: float
    base_reason: str
    prefetch_bandwidth: float
    peak_accumulator_bytes: int
    peak_x_bytes: int
    peak_w_bytes: int
    serialized_supply: bool = False
    final_drain_cycles: int = 1
    decoder_elements: int = 0


@lru_cache(maxsize=262144)
def projection_phase(m: int, n: int, k: int, core: Core, mem: CoreMemory,
                     settings: Settings, paired: bool = False) -> Phase:
    """Explicit loops: N-group, M-chunk, K-segment, M-block, N-tile.

    X is held while traversing the group's N tiles. A weight group is loaded
    again for each M chunk. At most eight output descriptors are retained;
    paired Gate/Up holds TWO FP32 planes per descriptor until SiLU (sixteen
    spatial output tiles maximum). Full physical tails occupy
    buffers; only valid native 32-B row sectors cross HBM.
    """
    nm, nn, nk = ceil(m, core.pm), ceil(n, core.pn), ceil(k, core.pk)
    mult = 2 if paired else 1
    bound = min(settings.records, mem.accumulator_bytes // (mult * core.record_bytes))
    if bound < 1 or not mem.eligible:
        raise ValueError("physical operand/partial sum exceeds its installed quota")
    mb = 1 if settings.flow == "bounded_os" or (paired and mem.w_slots == 1) else min(nm, bound)
    # Leave one physical W slot for lookahead whenever capacity permits.
    # The same rule applies to every geometry, not only asymmetric engines.
    ng = min(nn, max(1, (mem.w_slots - 1) // mult), bound // mb)
    limit=settings.gu_group_limit if paired else settings.down_group_limit
    if limit:ng=min(ng,limit)
    # A paired group holds both W sets to reuse them across its M blocks.
    # With one physical slot, mb=ng=1 streams Gate then Up sequentially.
    groups_m, groups_n = ceil(nm, mb), ceil(nn, ng)
    cycles = 0
    for mr, mc in ((mb, nm // mb), (nm % mb, int(bool(nm % mb)))):
        for nr, nc in ((ng, nn // ng), (nn % ng, int(bool(nn % ng)))):
            if mc and nc:
                cycles += mc * nc * group_cost(mult * mr * nr, nk, core, settings.timing).cycles
    issues = mult * nm * nn * nk
    # All supported PK are 32-B aligned in BF16. One full pass is exactly
    # n*align(2*k,32), regardless of physical N/K tails.
    unique = mult * packed_projection_bytes(n, k, settings.weight_format)
    wire = unique * groups_m
    fill = mult * groups_m * nn * nk * core.w_slice_bytes
    wread = issues * core.w_slice_bytes
    xfetch = groups_n * m * k * 2  # Gate/Up reuse the same X slice.
    xoperand = issues * core.x_slice_bytes
    # Retire each bounded group through its consumer BEFORE replacing its
    # partial sums. No whole Gate/Up or Down FP32 backing is assumed.
    vector = (3 if paired else 2) * m * n
    consumer_local = (8 if paired else 4) * m * n
    consumer_global = (2 if paired else 8) * m * n
    accbytes = mult * nm * nn * (2 * nk - 1) * core.record_bytes + consumer_local
    if settings.valid_operand_traffic:
        # Match Rust's valid_m/valid_n/valid_k port spans. The physical tile
        # reservation and full issued MAC slots remain unchanged. Zero lanes
        # are masked instead of fabricated SRAM payload reads/writes.
        fill=mult*groups_m*n*k*2
        wread=mult*nm*n*k*2
        xoperand=mult*nn*m*k*2
        n_words=sum(((min(core.pn,n-ni)*4+15)//16)*16 for ni in range(0,n,core.pn))
        accbytes=mult*m*n_words*(2*nk-1)+consumer_local
    ctrl = 2 * issues + 3 * (mult * groups_m * nn * nk) + 2 * (mult * groups_m * groups_n)
    consumer_minimum = vector / 64.0
    if settings.ports:
        consumer_minimum += consumer_local / mem.accumulator_bandwidth + consumer_global / 384.0
    # group_cost includes final commit; move it to the explicit endpoint drain
    # to charge it exactly once, including an HBM-limited last response.
    components = [("compute_or_K_dependency", float(cycles - settings.timing.completion_latency(core)) + consumer_minimum)]
    if settings.ports:
        components += [("weight_port", (wire + fill + wread) / mem.w_bandwidth),
                       ("X_port", (xfetch + xoperand) / mem.x_bandwidth),
                       ("accumulator_port", accbytes / mem.accumulator_bandwidth)]
    reason, base = max(components, key=lambda v: v[1])
    # One W slot cannot fill its successor before last operand read. Multiple
    # slots reserve one current consumer; remaining slots hide response delay.
    reuse_issues = min(mb, nm)
    last_use = max(1.0, reuse_issues * max(1.0,
                   (wread/issues if settings.valid_operand_traffic else core.w_slice_bytes) / mem.w_bandwidth if settings.ports else 1.0))
    hold = settings.fabric.hbm_latency_cycles + 1
    current_slots = min(mem.w_slots, mult * ng)
    spare_slots = mem.w_slots - current_slots
    loads = mult * groups_m * nn * nk
    average_native_slice = wire / loads
    if spare_slots == 0:
        group_wire = current_slots * average_native_slice
        # With no spare slot fill and last-read occupy it serially. Use native
        # wire bytes, not padded operand bytes, so N/K tails still pay a full
        # response latency for their smaller real payload.
        transfer = group_wire / settings.fabric.landing_credit_bandwidth_upper_bound
        window_bw = group_wire / (hold + transfer + current_slots * last_use)
    else:
        # Padding occupies operand capacity, but carries no native requests
        # and cannot act as extra useful lookahead sectors.
        window_bw = spare_slots * average_native_slice / hold
    return Phase("gate_up" if paired else "down", cycles, issues,
                 mult * m * n * k, issues * core.macs, wire, unique,
                 wire + fill + wread, xfetch + xoperand, accbytes, xfetch + consumer_global, vector,
                 ctrl, base, reason, window_bw,
                 mult * mb * ng * core.record_bytes,
                 core.x_slice_bytes * mem.x_buffers,
                 core.w_slice_bytes * mem.w_slots, spare_slots == 0,
                 settings.timing.completion_latency(core),
                 mult * groups_m * n * k if settings.weight_format != "BF16" else 0)


@lru_cache(maxsize=262144)
def expert_phases(m: int, h: int, f: int, core: Core, mem: CoreMemory,
                  settings: Settings) -> tuple[Phase, ...]:
    rows = mem.z_bytes // (2 * f)
    if rows < 1:
        raise ValueError("one Z row cannot fit")
    if rows < m and rows >= core.pm:
        rows = rows // core.pm * core.pm
    result = []
    for first in range(0, m, rows):
        actual = min(rows, m - first)
        result += [projection_phase(actual, f, h, core, mem, settings, True),
                   projection_phase(actual, h, f, core, mem, settings)]
    return tuple(result)


def private_estimate(phases: tuple[Phase, ...], m: int, h: int, f: int,
                     settings: Settings) -> float:
    result = 0.0
    for p in phases:
        demand = [p.base_cycles]
        if settings.hbm:
            bw = min(settings.fabric.landing_credit_bandwidth_upper_bound,
                     p.prefetch_bandwidth)
            demand.append(p.hbm_bytes / bw)
            if not p.serialized_supply:
                result += settings.fabric.hbm_latency_cycles
        if settings.ports:
            demand.append(p.activation_bytes / 384.0)
        if settings.control:
            demand.append(float(p.control_cycles))
        if settings.weight_format != "BF16" and settings.decoder_elements_per_cycle:
            demand.append(p.decoder_elements / settings.decoder_elements_per_cycle)
        result += max(demand) + p.final_drain_cycles
    return result


def fluid_rates(phases: dict[int, Phase], settings: Settings) -> dict[int, tuple[str, float]]:
    """Work-conserving weighted max-min fluid allocation of shared resources.

    Equal native-byte service is the reference fairness weight. When a phase
    reaches a local compute/port/window limit, its unused share is reclaimed.
    This is a continuous relaxation of arbitration, not native per-cycle RR.
    """
    local, demand = {}, {}
    capacities = {"vector_or_combine": 64.0}
    if settings.hbm:
        capacities["hbm_supply"] = settings.fabric.landing_credit_bandwidth_upper_bound
    if settings.ports:
        capacities["activation_port"] = 384.0
    if settings.control:
        capacities["control_port"] = 1.0
    if settings.weight_format != "BF16" and settings.decoder_elements_per_cycle:
        capacities["decoder"] = float(settings.decoder_elements_per_cycle)
    for c, p in phases.items():
        opts = [(p.base_reason, 1.0 / max(1.0, p.base_cycles))]
        if settings.hbm:
            opts.append(("hbm_supply", p.prefetch_bandwidth / p.hbm_bytes))
        local[c] = min(opts, key=lambda x: x[1])
        demand[c] = {"hbm_supply": p.hbm_bytes,
                     "activation_port": p.activation_bytes,
                     "control_port": p.control_cycles,
                     "vector_or_combine": p.vector_elements,
                     "decoder": p.decoder_elements}
    rates = {c: 0.0 for c in phases}
    reason = {c: local[c][0] for c in phases}
    weights = {c: 1.0 / max(1, phases[c].hbm_bytes) for c in phases}
    opened = set(phases)
    while opened:
        limits = [("local", c, (local[c][1] - rates[c]) / weights[c]) for c in opened]
        for r, capacity in capacities.items():
            used = sum(rates[c] * demand[c][r] for c in phases)
            slope = sum(weights[c] * demand[c][r] for c in opened)
            if slope:
                limits.append((r, None, max(0.0, capacity - used) / slope))
        tag, target, step = min(limits, key=lambda x: x[2])
        for c in opened:
            rates[c] += max(0.0, step) * weights[c]
        if tag == "local":
            opened.remove(target)
        else:
            stopped = {c for c in opened if demand[c][tag] > 0}
            for c in stopped:
                reason[c] = tag
            opened -= stopped
    return {c: (reason[c], rates[c]) for c in phases}


def simulate_layer(workload: dict, cores: tuple[Core, ...], settings: Settings = Settings(),
                   policy: str = "eft", detail: bool = False, fixed_owners: tuple[int, ...] | None = None) -> dict:
    if policy not in ("eft", "idle", "round_robin") and not (policy.startswith("threshold_") and policy.split("_")[-1].isdigit()):
        raise ValueError("unknown dispatch policy")
    if not workload["experts"] or any(not 0 < e["Me"] <= workload["batch"] for e in workload["experts"]):
        raise ValueError("expert population must be inside the declared batch")
    credit_tags = max(0, settings.fabric.hbm_credits - 256) * 2
    if len(cores) * settings.records * 256 + len(cores) * 32 * 24 + 4288 + credit_tags > 16384:
        raise ValueError("paired descriptor state exceeds the fixed control budget")
    hidden = workload.get("hidden", max(e["H"] for e in workload["experts"]))
    if any(e["H"] != hidden for e in workload["experts"]):
        raise ValueError("experts must share the declared layer hidden dimension")
    budget = memory_budget(cores, workload["batch"], hidden, max(e["F"] for e in workload["experts"]),
                           partition=settings.partition, allocation=settings.allocation, buffer_limit=settings.prefetch_slots,
                           profile=settings.fabric, top_k=workload.get("top_k", 6))
    if not budget.fits:
        raise ValueError("geometry exceeds fixed operand/global capacity")
    experts = workload["experts"]
    queues = [[] for _ in cores]
    available = [0.0] * len(cores)
    bindings = []
    ctrl_at = 0.0
    for idx, e in enumerate(experts):
        candidates = []
        for c, (core, mem) in enumerate(zip(cores, budget.cores)):
            try:
                phases = expert_phases(e["Me"], e["H"], e["F"], core, mem, settings)
                estimate = private_estimate(phases, e["Me"], e["H"], e["F"], settings)
                candidates.append((c, estimate, phases))
            except ValueError:
                pass
        if not candidates:
            raise ValueError("expert has no legal owner")
        if policy.startswith("threshold_") and len(cores) == 2 and cores[0] != cores[1]:
            threshold = int(policy.split("_")[1])
            stream = min(range(2), key=lambda c: (cores[c].pm, cores[c].macs, c))
            preferred = stream if e["Me"] <= threshold else 1 - stream
            narrowed = [x for x in candidates if x[0] == preferred]
            candidates = narrowed or candidates
        if fixed_owners is not None:
            if len(fixed_owners) != len(experts):
                raise ValueError("one fixed owner per expert required")
            chosen = [x for x in candidates if x[0] == fixed_owners[idx]]
            if not chosen:
                raise ValueError("fixed owner is illegal")
            c, estimate, phases = chosen[0]
        elif policy == "idle":
            c, estimate, phases = min(candidates, key=lambda x: (available[x[0]], x[0]))
        elif policy == "round_robin":
            preferred = idx % len(cores)
            c, estimate, phases = min(candidates, key=lambda x: (x[0] != preferred, x[0]))
        else:
            c, estimate, phases = min(candidates, key=lambda x: (available[x[0]] + x[1], x[0]))
        ctrl_at += 2 * len(candidates) + 2 if settings.control else 0
        available[c] += estimate
        queues[c].append((idx, phases, ctrl_at))
        bindings.append({"expert_index": idx, "core": c, "legal_cores": len(candidates),
                         "predicted_private_cycles": estimate, "bind_cycle": ctrl_at})

    # Weighted resource capacities are shared. Fluid phase progress is an
    # approximation, not instantaneous native transaction/operand scheduling.
    pending = []
    serial = 0
    active = {}
    pos = [0] * len(cores)
    finish = [0.0] * len(cores)
    counters = [{k: 0.0 for k in ("hbm_supply", "hbm_startup", "weight_port", "X_port",
                    "accumulator_port", "compute_or_K_dependency", "control_port",
                    "activation_port", "vector_or_combine", "decoder", "idle")} for _ in cores]
    phases_done = []
    def event(at, kind, core, task, phase):
        nonlocal serial
        serial += 1
        heapq.heappush(pending, (at, serial, kind, core, task, phase))
    clear = workload["batch"] * hidden * 4 / 384.0 if settings.ports else 0.0
    def start_next(c, at):
        if pos[c] < len(queues[c]):
            idx, ps, bound_at = queues[c][pos[c]]
            event(max(at, bound_at), "start", c, idx, 0)
        else:
            finish[c] = at
    for c in range(len(cores)):
        # Bind all finite layer descriptors first. This explicit conservative
        # barrier avoids a free second control port during phase execution.
        start_next(c, clear + ctrl_at)
    now = 0.0
    while pending or active:
        while pending and pending[0][0] <= now + 1e-7:
            at, _, kind, c, idx, pi = heapq.heappop(pending)
            ps = queues[c][pos[c]][1]
            if kind == "start":
                # Serialized-window service already pays every group's first
                # response delay. Avoid charging its first delay twice.
                latency = settings.fabric.hbm_latency_cycles if settings.hbm and not ps[pi].serialized_supply else 0
                counters[c]["hbm_startup"] += latency
                event(at + latency, "ready", c, idx, pi)
            elif kind == "ready":
                active[c] = [idx, pi, ps[pi], 1.0, at]
            elif kind == "continue":
                if pi < len(ps):
                    event(at, "start", c, idx, pi)
                else:
                    pos[c] += 1
                    start_next(c, at)
        if not active:
            if pending:
                later = pending[0][0]
                for c in range(len(cores)):
                    if finish[c] == 0:
                        counters[c]["idle"] += later - now
                now = later
                continue
            break
        rates = fluid_rates({c: entry[2] for c, entry in active.items()}, settings)
        until = min((active[c][3] / rates[c][1] for c in active), default=float("inf"))
        if pending:
            until = min(until, max(0.0, pending[0][0] - now))
        if until < 1e-10:
            until = 1e-7
        for c, (_, rate) in rates.items():
            active[c][3] -= until * rate
            counters[c][rates[c][0]] += until
        now += until
        for c in list(active):
            idx, pi, p, remaining, began = active[c]
            if remaining <= 1e-7:
                del active[c]
                # All bounded-group consumers are included in the shared
                # fluid service. Explicit final pipeline drain precedes the
                # next projection; no bulk unbacked FP32 outputs survive.
                end = now + p.final_drain_cycles
                event(end, "continue", c, idx, pi + 1)
                phases_done.append({"core": c, "expert_index": idx, "phase": p.name,
                                    "started": began, "stream_completed": now, "completed": end,
                                    **(asdict(p) if detail else {
                                        "useful_macs":p.useful_macs,"issued_macs":p.issued_macs,
                                        "hbm_bytes":p.hbm_bytes,"issues":p.issues,
                                        "x_port_bytes":p.x_port_bytes,"activation_bytes":p.activation_bytes,
                                        "control_cycles":p.control_cycles})})
    cycles = max(finish)
    useful = sum(p["useful_macs"] for p in phases_done)
    expected = sum(3 * e["Me"] * e["H"] * e["F"] for e in experts)
    assert useful == expected
    issued = sum(p["issued_macs"] for p in phases_done)
    result = {"workload": workload["id"], "batch": workload["batch"],
              "cycles": cycles, "latency_ms": cycles / 1e6,
              "useful_macs": useful, "issued_macs": issued,
              "padding_macs": issued - useful, "spatial_utilization": useful / issued,
              "wall_mac_utilization": useful / (12288 * cycles),
              "hbm_bytes": sum(p["hbm_bytes"] for p in phases_done),
              "native_unique_bytes": unique_bytes(workload, settings.weight_format),
              # Activation service includes X/Z fills and consumer output.
              # Removing the consumer gives actual fetches in either traffic
              # mode; subtracting padded issue slices would go negative when
              # valid M lanes are masked. Integer division handles Z chunks.
              "X_sram_bytes": sum(p["activation_bytes"] - (
                  p["useful_macs"] // experts[p["expert_index"]]["H"]
                  if p["phase"] == "gate_up" else
                  8 * (p["useful_macs"] // experts[p["expert_index"]]["F"]))
                  for p in phases_done),
              "global_activation_bytes": sum(p["activation_bytes"] for p in phases_done),
              "control_service_cycles": sum(p["control_cycles"] for p in phases_done) + ctrl_at,
              "core_finish_cycles": finish,
              "weight_format": settings.weight_format,
              "operand_traffic":'valid_spans' if settings.valid_operand_traffic else 'padded_v6_hypothesis',
              "compiler_groups":{'gate_up_limit':settings.gu_group_limit,'down_limit':settings.down_group_limit},
              "compulsory_traffic":{
                  "sum_HF":sum(e['H']*e['F'] for e in experts),
                  "sum_MH_plus_MF":sum(e['Me']*(e['H']+e['F']) for e in experts),
                  "accumulator_bytes":sum(e['Me']*(16*e['F']+8*e['H']) for e in experts),
                  "global_activation_bytes":sum(e['Me']*(4*e['F']+10*e['H']) for e in experts)+workload['batch']*hidden*4},
              "format_status": "BF16 reference" if settings.weight_format == "BF16" else "packed-transport sensitivity; accuracy and implementation not validated",
              "scope": "post-router MoE analytical fluid/finite-group model; 1ns hypothetical clock"}
    if detail:
        result.update(budget=budget.to_dict(), bindings=bindings, phases=phases_done,
                      exclusive_stream_counters=counters,
                      counters_warning="stream limiter attribution; not measured native stalls; startup overlaps idle counters; do not sum as wall time")
    return result
