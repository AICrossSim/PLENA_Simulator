"""Round-three BF16 finite-resource analytical model.

The byte counts describe explicit OS/WS/IS loop nests. Execution is an
event-driven *phase-fluid approximation*, not cycle-exact RTL/Ramulator:
each active projection makes weighted max-min progress through one shared
HBM and its installed private ports. Cold response latency, finite W
lookahead, output/K dependencies, and bounded task/operand storage are
charged. Port busy counters overlap and MUST NOT be added as wall time.

The legacy geometry3d module is only used for its immutable Core type. This
module never changes legacy timing/traffic implementations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import heapq
import inspect
import math
import random
from typing import Any, Iterable
from .config import Parameters

from research.moe_dispatch.geometry3d.compute import Core

TOTAL_MACS = 12_288
SRAM_BYTES = 2_158_592
KIB = 1024
PRIVATE_TOTALS = (40 * KIB, 12 * KIB, 96 * KIB, 384 * KIB)
FIXED_STORAGE = {"global_X": 512*KIB, "combined_Y": 1024*KIB,
                 "ingress": 8*KIB, "control": 16*KIB, "routes": 16*KIB}
SCOPE = "post-router BF16 MoE; finite-resource phase-fluid analytical estimate, not RTL/native HBM"


def ceildiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _partition(total: int, weights: tuple[int, ...], granularity: int = 1) -> tuple[int, ...]:
    units = total // granularity
    if units < len(weights):
        raise ValueError("positive quota required for every core")
    remain = units - len(weights)
    ideals = [remain * w / sum(weights) for w in weights]
    vals = [1 + int(x) for x in ideals]
    for i in sorted(range(len(vals)), key=lambda j: (-(ideals[j] % 1), j))[:units-sum(vals)]:
        vals[i] += 1
    return tuple(x * granularity for x in vals)


@dataclass(frozen=True)
class Design:
    cores: tuple[Core, ...]
    flows: tuple[str, ...] = ()
    w_bytes: tuple[int, ...] = ()
    x_bytes: tuple[int, ...] = ()
    acc_bytes: tuple[int, ...] = ()
    z_bytes: tuple[int, ...] = ()
    w_banks: tuple[int, ...] = ()
    x_banks: tuple[int, ...] = ()
    acc_banks: tuple[int, ...] = ()
    vector_lanes: tuple[int, ...] = ()
    total_macs: int = TOTAL_MACS
    records: int = 8
    label: str = ""
    landing_mode: str = "private"
    landing_pool_bytes: int = 0
    diagnostic_unbounded_w: bool = False

    def __post_init__(self):
        cs = tuple(self.cores)
        object.__setattr__(self, "cores", cs)
        if len(cs) not in (1, 2) or sum(c.macs for c in cs) != self.total_macs:
            raise ValueError("one/two cores and the declared exact multiplier budget required")
        weights = tuple(c.macs for c in cs)
        if self.landing_mode not in ("private", "shared"):
            raise ValueError("landing_mode must be private or shared")
        specs = (("w_bytes", PRIVATE_TOTALS[0], KIB), ("x_bytes", PRIVATE_TOTALS[1], KIB),
                 ("acc_bytes", PRIVATE_TOTALS[2], KIB), ("z_bytes", PRIVATE_TOTALS[3], KIB),
                 ("w_banks", 64, 1), ("x_banks", 24, 1), ("acc_banks", 12, 1),
                 ("vector_lanes", 64, 1))
        for name, total, granularity in specs:
            value = tuple(getattr(self, name))
            if not value:
                value = ((0,)*len(cs) if name == "w_bytes" and self.landing_mode == "shared"
                         else _partition(total, weights, granularity))
            minimum = 0 if name == "w_bytes" else 1
            if len(value) != len(cs) or min(value) < minimum:
                raise ValueError(f"{name}: one legal quota per core required")
            if granularity == 1 and sum(value) != total:
                raise ValueError(f"{name}: port/lane totals must sum to {total}")
            if granularity > 1 and any(v % granularity for v in value):
                raise ValueError(f"{name}: capacity lattice is 1 KiB")
            object.__setattr__(self, name, value)
        fs = tuple(str(f).upper() for f in self.flows) or ("OS",) * len(cs)
        if len(fs) != len(cs) or any(f not in ("OS", "WS", "IS") for f in fs):
            raise ValueError("one OS/WS/IS flow per core")
        object.__setattr__(self, "flows", fs)
        if self.landing_mode == "private":
            if self.landing_pool_bytes or any(v < 1 for v in self.w_bytes):
                raise ValueError("private mode requires positive W and no common pool")
        elif any(self.w_bytes) or self.landing_pool_bytes < sum(c.w_slice_bytes for c in cs):
            raise ValueError("shared mode has zero private W and reserves one current block per core")
        if self.landing_pool_bytes % KIB:
            raise ValueError("shared pool capacity lattice is 1 KiB")
        if not 1 <= self.records <= 8:
            raise ValueError("finite installed eight-record control window")
        if self.landing_pool_bytes + sum(FIXED_STORAGE.values()) + sum(sum(getattr(self, name)) for name in
                ("w_bytes", "x_bytes", "acc_bytes", "z_bytes")) != SRAM_BYTES:
            raise ValueError("storage ledger must be exactly 2,158,592 B")

    @property
    def pool_bytes(self):
        return self.landing_pool_bytes

    def effective_w_bytes(self, c):
        """Isolated cost quota; shared runtime still arbitrates one physical pool."""
        if self.diagnostic_unbounded_w:
            # Diagnostic only: capacity/cache/admission plus lookahead are
            # unbounded. The installed ledger remains a reference, not an
            # equal-resource claim. Physical read bandwidth stays finite.
            return 1 << 60
        if self.landing_mode == "private":
            return self.w_bytes[c]
        return self.landing_pool_bytes - sum(q.w_slice_bytes for j,q in enumerate(self.cores) if j != c)

    @property
    def geometry(self) -> str:
        return "+".join(f"{c.pm}x{c.pn}x{c.pk}" for c in self.cores)

    @property
    def family(self) -> str:
        return "single" if len(self.cores) == 1 else ("homogeneous" if self.cores[0] == self.cores[1] else "heterogeneous")

    def ledger(self) -> dict:
        return {"main_multipliers": self.total_macs, "installed_storage_B": SRAM_BYTES,
                "fixed_storage_B": FIXED_STORAGE, "private": {name: list(getattr(self, name))
                    for name in ("w_bytes", "x_bytes", "acc_bytes", "z_bytes", "w_banks",
                                 "x_banks", "acc_banks", "vector_lanes")},
                "vector_total_elements_per_cycle": 64, "capacity_quantum_B": KIB,
                "landing_mode":self.landing_mode,"shared_landing_pool_B":self.landing_pool_bytes,
                "shared_pool_counted_once":True,"non_iso_diagnostic":self.diagnostic_unbounded_w}


@dataclass(frozen=True)
class PhaseCost:
    name: str
    m: int
    n: int
    k: int
    paired: bool
    issues: int
    useful_macs: int
    issued_macs: int
    compute_cycles: float
    issue_busy: float
    dependency_floor: float
    hbm_bytes: int
    unique_hbm_bytes: int
    w_sram_bytes: int
    x_sram_bytes: int
    acc_sram_bytes: int
    rf_bytes: int
    vector_elements: int
    activation_bytes: int
    control_cycles: int
    w_tile_bytes: int
    first_weight_bytes: int
    weight_loads: int
    weight_live_slots: int
    w_slots: int
    peak_acc_bytes: int
    peak_x_bytes: int
    spill_bytes: int
    output_bytes: int


@dataclass(frozen=True)
class TaskCost:
    core: int
    phases: tuple[PhaseCost, ...]
    issues: int
    useful_macs: int
    issued_macs: int
    padding_macs: int
    issue_busy: float
    dependency_floor: float
    w_sram_bytes: int
    x_sram_bytes: int
    acc_sram_bytes: int
    hbm_bytes: int
    unique_hbm_bytes: int
    vector_elements: int
    isolated_cycles: float
    z_chunks: int
    spill_bytes: int

    @property
    def spatial_util(self):
        return self.useful_macs/self.issued_macs

    @property
    def irreducible_busy(self):
        return self.issue_busy


def _group_cycles(q: int, nk: int, II: float, L: float) -> float:
    return (nk-1)*max(q*II, L)+(q-1)*II+L


def _projection(m: int, n: int, k: int, design: Design, ci: int,
                params: Parameters, paired: bool) -> PhaseCost:
    core, flow = design.cores[ci], design.flows[ci]
    pm, pn, pk = core.pm, core.pn, core.pk
    nm, nn, nk = ceildiv(m, pm), ceildiv(n, pn), ceildiv(k, pk)
    mult = 2 if paired else 1
    wt, xt, rec = pn*pk*2, pm*pk*2, pm*pn*4
    slots = design.effective_w_bytes(ci)//wt
    if slots < 1 or design.x_bytes[ci] < xt or design.acc_bytes[ci] < mult*rec:
        raise ValueError("physical W/X/output record exceeds its private quota")
    unique = mult*n*ceildiv(k*2, 32)*32
    full_physical_weight = mult*nn*nk*wt
    cache_all_w = design.landing_mode == "private" and full_physical_weight <= design.effective_w_bytes(ci)
    qlimit = min(design.records, design.acc_bytes[ci]//(mult*rec))
    if qlimit < 1:
        raise ValueError("paired output context cannot fit")
    # OS owns RF outputs; WS owns the W tile in the array; IS owns X.
    if flow == "OS":
        ng = min(nn, qlimit)
        groups_n = ceildiv(nn, ng)
        groups_m = nm
        wreads = mult*nm*nn*nk*wt
        xreads = m*k*2*(1 if m*k*2 <= design.x_bytes[ci] else groups_n)
        xoperand = mult*nm*nn*nk*xt
        acc = mult*nm*nn*rec       # retire once; NO per-K SRAM RMW
        peak_acc = mult*ng*rec
        comp = nm*sum(_group_cycles(mult*min(ng, nn-j), nk,
                                   params.issue_interval, params.dot_latency(core)+1)
                      for j in range(0, nn, ng))
        live = 1
        spill = 0
    elif flow == "WS":
        # N/K outer, M waves inner: W is read from SRAM once per M group.
        # WS partials live in the budgeted accumulator SRAM, not in the
        # eight OS RF records or task descriptor window. Sequential M
        # operands can reuse W for every row that the actual SRAM can hold.
        mb = min(nm, design.acc_bytes[ci]//(mult*rec))
        groups_m = ceildiv(nm, mb)
        groups_n = nn
        wreads = mult*groups_m*nn*nk*wt
        cache_all_x = m*k*2 <= design.x_bytes[ci]
        xreads = m*k*2*(1 if cache_all_x else mult*nn)
        xoperand = mult*nm*nn*nk*xt
        acc = mult*nm*nn*(2*nk-1)*rec
        peak_acc = mult*mb*rec
        comp = nn*sum(_group_cycles(mult*min(mb, nm-j), nk,
                                   params.issue_interval, params.dot_latency(core)+1)
                      for j in range(0, nm, mb))
        live = 1                 # one array-resident W block
        spill = 0
    else:
        # M/K outer, N inner: one installed PMxPK X block stays in array.
        groups_m, groups_n = nm, 1
        wreads = mult*nm*nn*nk*wt
        xreads = m*k*2
        xoperand = nm*nk*xt      # X is NOT reread for every N tile
        acc = mult*nm*nn*(2*nk-1)*rec
        full_output = mult*pm*n*4
        spills = full_output > design.acc_bytes[ci]
        # Explicit off-chip backing when the IS partial plane cannot fit.
        spill = (mult*m*n*4*2*(nk-1)) if spills else 0
        acc += spill             # SRAM staging of spill/refill also costs
        peak_acc = min(full_output, design.acc_bytes[ci])
        comp = nm*_group_cycles(mult*nn, nk, params.issue_interval,
                                params.dot_latency(core)+1)
        live = 1
    if params.sram_weight_read_once_oracle:
        # U2 diagnostic oracle: remove repeated ARRAY reads across M waves,
        # retaining real HBM refetch, SRAM fill, and installed capacities.
        wreads = mult*nn*nk*wt
    refill = 1 if cache_all_w else groups_m
    # Whole-projection caching uses actual W slots; Next cannot silently evict
    # them while keeping the cache's once-only HBM byte count.
    if cache_all_w:
        live = mult*nn*nk
    wire = unique*refill
    padded_fill = mult*refill*nn*nk*wt
    issues = mult*nm*nn*nk
    comp = max(comp, issues*params.issue_interval)
    rf = mult*nm*nn*(2*nk-1)*rec if flow == "OS" else 0
    # Ingress read, packed W fill, and array read share the installed W banks.
    wport = wire+padded_fill+wreads
    vectors = (3 if paired else 2)*m*n
    # SiLU/GU consumes locally and writes BF16 Z. Down writes combined FP32.
    global_consumer = (2 if paired else 8)*m*n
    acc += (8 if paired else 12)*m*n
    xextra = 2*m*n if paired else 0  # BF16 Z write uses installed X/Z banks
    # Credited whole-X cache traffic requires reserving the actual resident
    # plane, not merely the currently issued physical PMxPK slice.
    peak_x = max(xt,m*k*2) if flow in ("OS","WS") and m*k*2<=design.x_bytes[ci] else xt
    # Per-core hardwired tile sequencers advance with the installed issue
    # pipeline. Only actual descriptor compare/bind operations are charged
    # globally in _one_layer; no invented serialized tile-control bottleneck.
    control = 0
    return PhaseCost("gate_up" if paired else "down", m, n, k, paired,
                     issues, mult*m*n*k, issues*core.macs, comp,
                     issues*params.issue_interval, nk*(params.dot_latency(core)+1),
                     wire+spill, unique, wport, xreads+xoperand+xextra, acc, rf,
                     vectors, xreads+global_consumer, control, wt,
                     min(wt, min(n, pn)*ceildiv(min(k, pk)*2,32)*32),
                     mult*refill*nn*nk, live, slots,
                     peak_acc, peak_x, spill, global_consumer)


def _window_bandwidth(phase: PhaseCost, params: Parameters,
                      reserved_next: bool | int = False) -> float:
    held = int(reserved_next) if isinstance(reserved_next,bool) else ceildiv(reserved_next,phase.w_tile_bytes)
    spare = phase.w_slots-phase.weight_live_slots-held
    average = max(32.0, (phase.hbm_bytes-phase.spill_bytes)/max(1, phase.weight_loads))
    # A projection fetched once into a whole-projection cache starts with
    # its reserved slots empty. All these slots can carry the initial fill;
    # they are not a permanently full cache demanding serialized refills.
    if phase.weight_loads == phase.weight_live_slots:
        initial_slots = max(1, phase.w_slots-held)
        return initial_slots*average/(params.hbm_latency+1)
    if spare > 0:
        return spare*average/(params.hbm_latency+1)
    # A one-slot array cannot refill before last use; serialized finite window.
    use = phase.compute_cycles/max(1, phase.weight_loads)
    return average/(params.hbm_latency+1+average/params.hbm_bandwidth+use)


def _isolated(phases: tuple[PhaseCost, ...], design: Design, c: int, p: Parameters) -> float:
    ans = 0.0
    for s in phases:
        resources = [s.compute_cycles,
                     s.w_sram_bytes/p.w_bandwidth(design, c),
                     s.x_sram_bytes/(design.x_banks[c]*p.bank_Bpc),
                     s.acc_sram_bytes/(design.acc_banks[c]*p.bank_Bpc),
                     s.vector_elements/(min(64, design.vector_lanes[c])*p.vector_scale),
                     s.hbm_bytes/min(p.hbm_bandwidth, float("inf") if design.diagnostic_unbounded_w else _window_bandwidth(s, p))]
        if p.charge_control:
            resources.append(float(s.control_cycles))
        # compute_cycles already includes the final reduction-tree drain.
        ans += p.hbm_latency+max(resources)
    return ans


@lru_cache(maxsize=32768)
def _task_cached(m: int, h: int, f: int, design: Design, c: int, p: Parameters) -> TaskCost:
    if min(m, h, f) <= 0 or not 0 <= c < len(design.cores):
        raise ValueError("positive expert shape and legal core required")
    zrows = design.z_bytes[c]//(2*f)
    if zrows < 1:
        raise ValueError("one BF16 Z row cannot fit")
    phases = []
    for begin in range(0, m, zrows):
        rows = min(zrows, m-begin)
        phases += [_projection(rows, f, h, design, c, p, True),
                   _projection(rows, h, f, design, c, p, False)]
    ps = tuple(phases)
    if p.sram_weight_read_once_oracle:
        seen=set()
        adjusted=[]
        from dataclasses import replace
        for s in ps:
            key=(s.n,s.k,s.paired)
            if key in seen:
                once=(2 if s.paired else 1)*ceildiv(s.n,design.cores[c].pn)*ceildiv(s.k,design.cores[c].pk)*s.w_tile_bytes
                s=replace(s,w_sram_bytes=s.w_sram_bytes-once)
            seen.add(key)
            adjusted.append(s)
        ps=tuple(adjusted)
    sums = lambda name: sum(getattr(x, name) for x in ps)
    issued, useful = sums("issued_macs"), sums("useful_macs")
    unique = 2*f*ceildiv(2*h,32)*32+h*ceildiv(2*f,32)*32
    return TaskCost(c, ps, sums("issues"), useful, issued, issued-useful,
                    sums("issue_busy"), max(x.dependency_floor for x in ps),
                    sums("w_sram_bytes"), sums("x_sram_bytes"), sums("acc_sram_bytes"),
                    sums("hbm_bytes"), unique, sums("vector_elements"),
                    _isolated(ps, design, c, p), ceildiv(m, zrows), sums("spill_bytes"))


def task_cost(expert: dict, design: Design, core: int, params: Parameters = Parameters()) -> TaskCost:
    """Cost of one expert; isolated_cycles is NOT a summable schedule LB."""
    return _task_cached(int(expert["Me"]), int(expert.get("H", 2048)),
                        int(expert.get("F", 1408)), design, core, params)


def _fluid_rates(actors: dict[Any, dict], capacities: dict[Any, float]) -> dict[Any, float]:
    """Work-conserving weighted max-min service of globally shared resources."""
    rates = {a: 0.0 for a in actors}
    weights = {a: 1/max(1.0, actors[a]["demand"].get("hbm", 0.0)) for a in actors}
    remaining = set(actors)
    while remaining:
        limits = [(max(0.0, actors[a]["limit"]-rates[a])/weights[a], "local", a)
                  for a in remaining]
        for r, cap in capacities.items():
            used = sum(rates[a]*actors[a]["demand"].get(r, 0.0) for a in actors)
            slope = sum(weights[a]*actors[a]["demand"].get(r, 0.0) for a in remaining)
            if slope:
                limits.append((max(0.0, cap-used)/slope, "resource", r))
        step, typ, who = min(limits, key=lambda v: (v[0], str(v[2])))
        for a in remaining:
            rates[a] += step*weights[a]
        if typ == "local":
            remaining.remove(who)
        else:
            remaining -= {a for a in remaining if actors[a]["demand"].get(who,0)>0}
    return rates


def _prediction(predictor, e, c, nominal, design, params):
    if predictor is None:
        return nominal
    if hasattr(predictor, "predict"):
        return max(1.0, float(predictor.predict(e, c, nominal)))
    return max(1.0, float(predictor(e, c, nominal)))


@dataclass(frozen=True)
class PoolRequest:
    key: Any
    tile_bytes: int
    need_at: float
    block_service_cycles: float
    max_blocks: int


class BytePoolAllocator:
    """Finite byte-window admission, one current block guaranteed per core.

    Actors are phase-fluid streams: grants represent finite lookahead byte
    windows, not claimed discrete Ramulator requests. A pending Next block
    reserves its full physical tile before the HBM request starts. Remaining
    whole blocks are admitted by the earliest predicted consumption deadline.
    All cores share this one ledger; different tile sizes are never rounded
    into a fictitious common slot count.
    """
    def __init__(self, pool_bytes: int, current_tiles: Iterable[int]):
        self.pool_bytes = int(pool_bytes)
        self.current_tiles = tuple(map(int,current_tiles))
        if min(self.current_tiles,default=1)<1 or sum(self.current_tiles)>pool_bytes:
            raise ValueError("pool cannot reserve every current block")

    def allocate(self, requests: Iterable[PoolRequest], next_reservations: Iterable[int]=()):
        rs=list(requests)
        extras=tuple(map(int,next_reservations))
        if any(x<0 for x in extras):
            raise ValueError("negative Next reservation")
        reserved=sum(self.current_tiles)+sum(extras)
        if reserved>self.pool_bytes:
            raise ValueError("Next overcommits the common byte pool")
        free=self.pool_bytes-reserved
        granted={r.key:0 for r in rs}
        pending=[]
        for serial,r in enumerate(rs):
            if r.tile_bytes<=0 or r.max_blocks<0 or r.block_service_cycles<0:
                raise ValueError("illegal pool descriptor")
            if r.max_blocks:
                heapq.heappush(pending,(r.need_at,serial,0,r))
        while pending:
            deadline,serial,index,r=heapq.heappop(pending)
            if r.tile_bytes>free:
                continue
            free-=r.tile_bytes
            granted[r.key]+=r.tile_bytes
            if index+1<r.max_blocks:
                heapq.heappush(pending,(r.need_at+(index+1)*r.block_service_cycles,
                                      serial,index+1,r))
        used=reserved+sum(granted.values())
        if used>self.pool_bytes:
            raise AssertionError("shared landing-byte conservation")
        return granted, {"pool_bytes":self.pool_bytes,"current_reserved_bytes":sum(self.current_tiles),
                         "next_reserved_bytes":sum(extras),"lookahead_bytes":sum(granted.values()),
                         "used_bytes":used,"unused_bytes":self.pool_bytes-used}


def _pool_can_reserve_next(design, next_state, tile_bytes):
    return (sum(q.w_slice_bytes for q in design.cores)
            +sum(v.get("slot_bytes",0) for v in next_state.values())+tile_bytes
            <=design.landing_pool_bytes)


def _set_window_limits(active, next_state, design, params, now):
    """Charge private finite windows exactly as before; pool windows once."""
    if design.landing_mode == "private":
        for key,a in active.items():
            s=a["cost"]
            if s is not None and a["demand"]["hbm"]>0:
                reserved=next_state.get(a["core"],{}).get("slot_bytes",0)
                a["limit"]=(float("inf") if design.diagnostic_unbounded_w else
                            _window_bandwidth(s,params,reserved))/a["demand"]["hbm"]
        return None
    pool=BytePoolAllocator(design.landing_pool_bytes,[q.w_slice_bytes for q in design.cores])
    requests=[]
    for key,a in active.items():
        s=a["cost"]
        if s is not None and a["demand"]["hbm"]>0:
            requests.append(PoolRequest(key,s.w_tile_bytes,now,
                s.compute_cycles/max(1,s.weight_loads),max(0,s.w_slots-1)))
    grants,ledger=pool.allocate(requests,[v.get("slot_bytes",0) for v in next_state.values()])
    for key,a in active.items():
        s=a["cost"]
        if s is None or a["demand"]["hbm"]<=0:
            continue
        blocks=grants.get(key,0)//s.w_tile_bytes
        average=max(32.0,(s.hbm_bytes-s.spill_bytes)/max(1,s.weight_loads))
        if design.diagnostic_unbounded_w:
            bandwidth=float("inf")
        elif blocks:
            bandwidth=blocks*average/(params.hbm_latency+1.0)
        else:
            use=s.compute_cycles/max(1,s.weight_loads)
            bandwidth=average/(params.hbm_latency+1.0+average/params.hbm_bandwidth+use)
        a["limit"]=bandwidth/a["demand"]["hbm"]
        a["admitted_lookahead_bytes"]=grants.get(key,0)
    return ledger


def _supply_segment(now,dt,active,rates,usages,params,design,experts,pool_ledger=None):
    ncores=len(design.cores)
    percore=[sum(rates[key]*a["demand"].get("hbm",0.0) for key,a in active.items()
                 if a["core"]==c) for c in range(ncores)]
    segment={"start":now,"end":now+dt,"hbm_rate_Bpc":usages["hbm"],
        "hbm_rate_Bpc_core":percore,"inflight_bytes":[x*(params.hbm_latency+1.0) for x in percore],
        "active_run":[("run",c) in active for c in range(ncores)],
        "run_expert":[experts[active[("run",c)]["task"]].get("id",active[("run",c)]["task"])
                      if ("run",c) in active else None for c in range(ncores)]}
    if pool_ledger is not None:
        segment["pool_ledger"]=pool_ledger
    return segment


def _idle_segment(start,end,ncores):
    return {"start":start,"end":end,"hbm_rate_Bpc":0.0,"hbm_rate_Bpc_core":[0.0]*ncores,
        "inflight_bytes":[0.0]*ncores,"active_run":[False]*ncores,"run_expert":[None]*ncores}


def _one_layer(workload: dict, design: Design, params: Parameters, policy: str,
               owners: tuple[int, ...] | None, predictor, detail: bool,
               cost_overrides: dict | None=None, shape_overrides: dict | None=None) -> dict:
    es = workload["experts"]
    if not es:
        return {"cycles":0.0,"latency_ms":0.0,"hbm_bytes":0,"useful_macs":0,"issued_macs":0,
                "tasks":[],"bindings":[],"phases":[],"core_finish_cycles":[0.0]*len(design.cores)}
    ncores = len(design.cores)
    cost_overrides=cost_overrides or {}
    shape_overrides=shape_overrides or {}
    if (cost_overrides or shape_overrides) and ncores!=1:
        raise ValueError("U1 cost/shape overrides are single-core only")
    def co_for(i,c):
        return cost_overrides[i] if i in cost_overrides else task_cost(es[i],design,c,params)
    def actual_core(i,c):
        return shape_overrides.get(i,design.cores[c])
    costs, predicted, queues = {}, {}, [[] for _ in range(ncores)]
    availability = [0.0]*ncores
    bindings = []
    binding_ready = {}
    rng = random.Random(params.seed)
    ctrl = 0.0
    order = list(range(len(es)))
    if owners is not None:
        if len(owners) != len(es):
            raise ValueError("one owner per expert required")
        # The solver's assignment is made executable in LPT order.
        order.sort(key=lambda i: (-co_for(i,int(owners[i])).isolated_cycles,i))
    threshold = 2
    if policy.startswith("threshold_"):
        try:
            threshold = int(policy.split("_")[-1])
        except ValueError:
            raise ValueError("threshold policy needs integer suffix")
    if policy == "adaptive":
        vals = sorted(e["Me"] for e in es if not e.get("is_shared",False))
        threshold = vals[len(vals)//2] if vals else 2
    # Fixed solver assignments use a declared LPT plan. Runtime policies bind
    # online below; no whole-layer free assignment or unbounded core FIFO.
    for i in (order if owners is not None else ()):
        options = []
        for c in range(ncores):
            try:
                co = co_for(i,c)
            except ValueError:
                continue
            costs[i,c] = co
            pred = _prediction(predictor,es[i],c,co.isolated_cycles,design,params)
            predicted[i,c] = pred
            options.append((c,pred))
        if not options:
            raise ValueError(f"expert {i} has no legal installed core")
        legal = len(options)
        if owners is not None:
            options = [o for o in options if o[0] == owners[i]]
            if not options:
                raise ValueError("fixed owner is infeasible")
        elif (policy.startswith("threshold_") or policy=="adaptive") and ncores==2:
            stream = min(range(ncores),key=lambda c:(design.cores[c].pm,design.cores[c].macs,c))
            preferred = stream if es[i]["Me"]<=threshold and not es[i].get("is_shared",False) else 1-stream
            selected = [o for o in options if o[0]==preferred]
            if "fallback" in policy and selected:
                fallback = min(options,key=lambda o:(availability[o[0]]+o[1],o[0]))
                selected = [fallback] if availability[fallback[0]]+fallback[1] < availability[preferred]+selected[0][1] else selected
            options = selected or options
        if policy == "random" and owners is None:
            chosen = options[rng.randrange(len(options))]
        elif policy in ("idle","greedy") and owners is None:
            chosen = min(options,key=lambda o:(availability[o[0]],o[0]))
        elif policy in ("round_robin","rr") and owners is None:
            chosen = min(options,key=lambda o:(o[0]!=i%ncores,o[0]))
        else:
            chosen = min(options,key=lambda o:(availability[o[0]]+o[1],o[0]))
        c,pred = chosen
        ctrl += (2*legal+2) if params.charge_control else 0
        bindings.append({"expert_index":i,"expert_id":es[i].get("id",i),"core":c,
                         "legal_cores":legal,"predicted_cycles":pred,"nominal_cycles":costs[i,c].isolated_cycles,"bind_cycle":ctrl,
                         "predicted_finish":availability[c]+pred})
        binding_ready[i] = ctrl
        availability[c] += pred
        queues[c].append(i)
    # Only an eight-entry control window is installed. Layer descriptors live
    # in the explicitly budgeted route store; no whole expert matrix is queued.
    batch = workload["batch"]
    hidden = workload.get("hidden",es[0].get("H",2048))
    clear_bytes=batch*hidden*4
    clear = clear_bytes/(12*params.bank_Bpc)
    starts = clear+ctrl
    pending, serial, active = [],0,{}
    event_keys = set()
    positions = [0]*ncores
    finishes = [starts]*ncores
    task_started, tasks = {},{}
    previous_completion = [None]*ncores
    complete_accepts_nominal = False
    if predictor is not None and hasattr(predictor,"on_complete"):
        sig=inspect.signature(predictor.on_complete)
        complete_accepts_nominal = "nominal" in sig.parameters or any(
            q.kind==inspect.Parameter.VAR_KEYWORD for q in sig.parameters.values())
    waiting = [] if owners is not None else list(order)
    eta_end = {}
    task_done_work = {}
    task_next_quarter = {}
    next_state = {}       # at most one reserved Next block per core
    phase_started = {}
    phase_rows = []
    stream_attribution = [dict() for _ in range(ncores)]
    segments = []
    busy = {"hbm":0.0,"W":0.0,"X":0.0,"acc":clear,"vector":0.0}
    core_busy = [0.0]*ncores
    actual_bytes = 0.0
    prefetched_bytes = 0.0
    wasted_prefetch = 0
    capacities = {"hbm":params.hbm_bandwidth,
                  "vector":64*params.vector_scale,"control":1.0}
    for c in range(ncores):
        capacities[("compute",c)] = 1.0
        capacities[("W",c)] = params.w_bandwidth(design,c)
        capacities[("X",c)] = design.x_banks[c]*params.bank_Bpc
        capacities[("acc",c)] = design.acc_banks[c]*params.bank_Bpc
        capacities[("vector",c)] = design.vector_lanes[c]*params.vector_scale
    def event(at,kind,c,i,pi=0):
        nonlocal serial
        key=(round(float(at),6),kind,c,i,pi)
        if key in event_keys:
            return
        event_keys.add(key)
        serial+=1
        heapq.heappush(pending,(float(at),serial,kind,c,i,pi))
    def capacity_to_bind(c, co, next_i):
        outstanding=len(queues[c])-positions[c]
        if outstanding>=2:
            return False
        if not outstanding:
            return True
        current=costs[queues[c][positions[c]],c]
        # Reserve a true first W/X/output/Z context for Next. Current keeps
        # enough installed resources to finish; predictions never free slots.
        wp=max(s.weight_live_slots*s.w_tile_bytes for s in current.phases)
        xp=max(s.peak_x_bytes for s in current.phases)
        ap=max(s.peak_acc_bytes for s in current.phases)
        zp=max(s.m*s.n*2 for s in current.phases if s.paired)
        first=co.phases[0]
        return ((wp+first.w_tile_bytes<=design.effective_w_bytes(c) and
                 (design.landing_mode == "private" or _pool_can_reserve_next(design,next_state,first.w_tile_bytes))) and
                xp+actual_core(next_i,c).x_slice_bytes<=design.x_bytes[c] and
                ap+2*actual_core(next_i,c).record_bytes<=design.acc_bytes[c] and
                zp+first.n*2<=design.z_bytes[c])
    def try_bind(at):
        nonlocal ctrl
        if owners is not None:
            return
        progress=True
        while progress and waiting:
            progress=False
            for i in waiting[:8]:
                physical=[]
                for c in range(ncores):
                    try:
                        co=co_for(i,c)
                    except ValueError:
                        continue
                    costs[i,c]=co
                    physical.append(c)
                if not physical:
                    raise ValueError(f"expert {i} has no legal installed core")
                options=[]
                for c in physical:
                    out=len(queues[c])-positions[c]
                    finish=at if not out else eta_end.get(queues[c][positions[c]],availability[c])
                    if out and finish-at>params.binding_lead_cycles+1e-7:
                        event(max(at+1e-6,finish-params.binding_lead_cycles),"bindcheck",c,
                              queues[c][positions[c]],0)
                        continue
                    if not capacity_to_bind(c,costs[i,c],i):
                        continue
                    pred=_prediction(predictor,es[i],c,costs[i,c].isolated_cycles,design,params)
                    options.append((c,pred,max(at,finish)))
                legal=len(options)
                if owners is None and (policy.startswith("threshold_") or policy=="adaptive") and ncores==2:
                    stream=min(range(ncores),key=lambda c:(design.cores[c].pm,design.cores[c].macs,c))
                    preferred=stream if es[i]["Me"]<=threshold and not es[i].get("is_shared",False) else 1-stream
                    pref=[o for o in options if o[0]==preferred]
                    if "fallback" not in policy:
                        options=pref if preferred in physical else options
                    elif pref:
                        best=min(options,key=lambda o:(o[2]+o[1],o[0]))
                        options=[best] if best[2]+best[1]<pref[0][2]+pref[0][1] else pref
                if not options:
                    continue
                if policy=="random":
                    c,pred,finish=options[rng.randrange(len(options))]
                elif policy in ("idle","greedy"):
                    c,pred,finish=min(options,key=lambda o:(o[2],o[0]))
                elif policy in ("rr","round_robin"):
                    c,pred,finish=min(options,key=lambda o:(o[0]!=i%ncores,o[0]))
                else:
                    c,pred,finish=min(options,key=lambda o:(o[2]+o[1],o[0]))
                was_empty=len(queues[c])==positions[c]
                ctrl=max(ctrl,at)+(2*legal+2 if params.charge_control else 0)
                predicted[i,c]=pred
                availability[c]=max(finish,ctrl)+pred
                eta_end[i]=availability[c]
                queues[c].append(i)
                waiting.remove(i)
                bindings.append({"expert_index":i,"expert_id":es[i].get("id",i),"core":c,
                    "legal_cores":legal,"physical_legal_cores":len(physical),"predicted_cycles":pred,
                    "nominal_cycles":costs[i,c].isolated_cycles,
                    "bind_cycle":ctrl,"predicted_finish":availability[c],"online":True,
                    "bounded_core_queue_depth":len(queues[c])-positions[c]})
                binding_ready[i] = ctrl
                if was_empty:
                    event(ctrl,"bound_ready",c,i,0)
                else:
                    current=queues[c][positions[c]]
                    event(max(ctrl,eta_end.get(current,finish)-params.hbm_latency-
                              costs[i,c].phases[0].first_weight_bytes/params.hbm_bandwidth),
                          "prefetch",c,current,0)
                progress=True
                break
    def begin_task(c,at):
        if positions[c]>=len(queues[c]):
            finishes[c]=at
            return
        i=queues[c][positions[c]]
        # A queued Next descriptor can become the queue head while its
        # serialized binding/ownership update is still in flight. Neither
        # the cold HBM request nor task execution may precede that grant.
        if at < binding_ready[i]:
            event(binding_ready[i],"bound_ready",c,i,0)
            return
        task_started[i]=at
        eta_end[i]=at+predicted[i,c]
        task_done_work[i]=0.0
        task_next_quarter[i]=1
        tasks[i]={"expert_index":i,"expert_id":es[i].get("id",i),"core":c,
                  "start":at,"predicted_cycles":predicted[i,c],"nominal_cycles":costs[i,c].isolated_cycles,"Me":es[i]["Me"],
                  "prefetch_request":None,"first_weight_ready":None,"current_end":previous_completion[c],
                  "shape":f"{actual_core(i,c).pm}x{actual_core(i,c).pn}x{actual_core(i,c).pk}"}
        pf=next_state.get(c)
        if pf and pf["task"]==i:
            tasks[i]["prefetch_request"]=pf["requested"]
            if pf.get("ready") is not None:
                tasks[i]["first_weight_ready"]=pf["ready"]
                event(at,"ready",c,i,0)
            else:
                pf["waiting"]=True
        else:
            event(at+params.hbm_latency,"ready",c,i,0)
        if params.prefetch and positions[c]+1<len(queues[c]):
            # Predictors influence prefetch timing, not correctness.
            predicted_end=at+predicted[i,c]
            event(max(at,predicted_end-params.hbm_latency-32/params.hbm_bandwidth),"prefetch",c,i,0)
    if owners is None:
        try_bind(starts)
    else:
        for c in range(ncores):
            begin_task(c,starts)
    now=0.0
    while pending or active:
        while pending and pending[0][0] <= now+1e-7:
            at,_,kind,c,i,pi=heapq.heappop(pending)
            event_keys.discard((round(at,6),kind,c,i,pi))
            if kind=="bindcheck":
                try_bind(at)
                continue
            co=costs[i,c]
            if kind=="bound_ready":
                if i not in task_started and positions[c]<len(queues[c]) and queues[c][positions[c]]==i:
                    begin_task(c,at)
            elif kind=="ready":
                s=co.phases[pi]
                subtract=0
                subtract_w=0
                pf=next_state.get(c)
                if pi==0 and pf and pf["task"]==i and pf.get("ready") is not None:
                    subtract=pf["bytes"]
                    subtract_w=pf["w_bytes"]
                    tasks[i]["first_weight_ready"]=pf["ready"]
                    # Reference released at first operand read, never on ETA.
                    del next_state[c]
                demand={"hbm":max(0,s.hbm_bytes-subtract),
                        "vector":s.vector_elements,("vector",c):s.vector_elements,
                        ("compute",c):s.compute_cycles,("W",c):s.w_sram_bytes-subtract_w,
                        ("X",c):s.x_sram_bytes,("acc",c):s.acc_sram_bytes,
                        "control":s.control_cycles if params.charge_control else 0}
                active[("run",c)]={"task":i,"phase":pi,"cost":s,"remaining":1.0,
                                      "began":at,"demand":demand,"limit":float("inf"),"core":c}
                # Demand and installed capacities are constant within a phase.
                # Preserve the original iteration/tie/division order once.
                options=[(r,demand.get(r,0)/cap) for r,cap in capacities.items() if demand.get(r,0)]
                active[("run",c)]["static_reason"]=str(max(options,key=lambda r:r[1])[0])
                phase_started[i,pi]=at
                if pi==0 and tasks[i]["first_weight_ready"] is None:
                    # Record the observed prefix completion of this actor's
                    # shared fluid HBM+local fill service, never a free global
                    # bandwidth estimate. This remains a phase-fluid prefix,
                    # not a native per-request tile-return trace.
                    active[("run",c)]["first_weight_fraction"]=min(1.0,max(
                        s.first_weight_bytes/max(1.0,demand["hbm"]),
                        (s.first_weight_bytes+s.w_tile_bytes)/max(1.0,demand[("W",c)])))
                if params.prefetch and positions[c]+1<len(queues[c]) and c not in next_state:
                    ni=queues[c][positions[c]+1]
                    event(max(at,eta_end[i]-params.hbm_latency-
                              costs[ni,c].phases[0].first_weight_bytes/params.hbm_bandwidth),
                          "prefetch",c,i,pi)
                try_bind(at)
            elif kind=="continue":
                if pi<len(co.phases):
                    event(at+params.hbm_latency,"ready",c,i,pi)
                else:
                    tasks[i]["finish"]=at
                    tasks[i]["actual_cycles"]=at-task_started[i]
                    tasks[i]["predicted_finish"]=task_started[i]+predicted[i,c]
                    if predictor is not None and hasattr(predictor,"on_complete"):
                        args=(es[i],c,predicted[i,c],tasks[i]["actual_cycles"])
                        if complete_accepts_nominal:
                            predictor.on_complete(*args,nominal=costs[i,c].isolated_cycles)
                        else:
                            predictor.on_complete(*args)
                    eta_end[i]=at
                    previous_completion[c]=at
                    positions[c]+=1
                    next_i=queues[c][positions[c]] if positions[c]<len(queues[c]) else None
                    begin_task(c,at)
                    if next_i is not None and next_i in tasks:
                        tasks[next_i]["current_end"]=at
                    try_bind(at)
            elif kind=="prefetch":
                if not params.prefetch:
                    continue
                if positions[c]>=len(queues[c]) or queues[c][positions[c]]!=i or c in next_state:
                    continue
                if positions[c]+1>=len(queues[c]):
                    continue
                ni=queues[c][positions[c]+1]
                first=costs[ni,c].phases[0]
                current=active.get(("run",c))
                if current is None:
                    continue
                cur=current["cost"]
                if cur.w_slots-cur.weight_live_slots<1:
                    continue
                if design.landing_mode == "shared" and not _pool_can_reserve_next(design,next_state,first.w_tile_bytes):
                    continue
                nbytes=ceildiv(first.first_weight_bytes,32)*32
                next_state[c]={"task":ni,"bytes":nbytes,"slot_bytes":first.w_tile_bytes,
                               "w_bytes":nbytes+first.w_tile_bytes,
                               "requested":at,"ready":None,"waiting":False}
                event(at+params.hbm_latency,"prefetch_ready",c,ni,0)
            elif kind=="prefetch_ready":
                pf=next_state.get(c)
                if not pf or pf["task"]!=i:
                    continue
                active[("prefetch",c)]={"task":i,"phase":0,"cost":None,"remaining":1.0,
                    "began":at,"core":c,"limit":float("inf"),
                    "demand":{"hbm":pf["bytes"],("W",c):pf["w_bytes"]}}
        if not active:
            if pending:
                if detail and pending[0][0] > now:
                    segments.append(_idle_segment(now,pending[0][0],ncores))
                now=max(now,pending[0][0])
                continue
            break
        # One physical pool ledger for shared landing; unchanged private windows.
        pool_ledger=_set_window_limits(active,next_state,design,params,now)
        rates=_fluid_rates(active,capacities)
        dt=min(a["remaining"]/max(1e-300,rates[key]) for key,a in active.items())
        for key,a in active.items():
            if "first_weight_fraction" in a:
                need=a["first_weight_fraction"]-(1-a["remaining"])
                if need>1e-9:
                    dt=min(dt,need/max(1e-300,rates[key]))
        for key,a in active.items():
            if key[0]!="run" or predictor is None:
                continue
            i,c=a["task"],a["core"]
            q=task_next_quarter[i]
            if q<=3:
                target=costs[i,c].useful_macs*q/4
                sofar=task_done_work[i]+a["cost"].useful_macs*(1-a["remaining"])
                if target>sofar+1e-7:
                    dt=min(dt,(target-sofar)/(a["cost"].useful_macs*max(1e-300,rates[key])))
        if pending:
            dt=min(dt,max(0.0,pending[0][0]-now))
        if not math.isfinite(dt) or dt>1e20:
            raise RuntimeError("resource deadlock / zero progress")
        if dt<=1e-10:
            dt=1e-7
        usages={r:sum(rates[key]*a["demand"].get(r,0) for key,a in active.items()) for r in capacities}
        if detail:
            segments.append(_supply_segment(now,dt,active,rates,usages,params,design,es,pool_ledger))
        actual_bytes+=usages["hbm"]*dt
        busy["hbm"]+=usages["hbm"]/capacities["hbm"]*dt
        busy["vector"]+=usages["vector"]/capacities["vector"]*dt
        for kind in ("W","X","acc"):
            denom=sum(capacities[(kind,c)] for c in range(ncores))
            busy[kind]+=sum(usages[(kind,c)] for c in range(ncores))/denom*dt
        for key,a in active.items():
            a["remaining"]-=dt*rates[key]
            if key[0]=="run":
                c=a["core"]
                core_busy[c]+=rates[key]*a["demand"][("compute",c)]*dt
                reason=a["static_reason"]
                stream_attribution[c][reason]=stream_attribution[c].get(reason,0)+dt
        now+=dt
        for key,a in active.items():
            target=a.get("first_weight_fraction")
            if target is not None and 1-a["remaining"]>=target-1e-9:
                tasks[a["task"]]["first_weight_ready"]=now
                del a["first_weight_fraction"]
        for key,a in list(active.items()):
            if key[0]!="run" or predictor is None:
                continue
            i,c=a["task"],a["core"]
            progress=(task_done_work[i]+a["cost"].useful_macs*(1-a["remaining"]))/costs[i,c].useful_macs
            while task_next_quarter[i]<=3 and progress+1e-8>=task_next_quarter[i]/4:
                quarter=task_next_quarter[i]/4
                task_next_quarter[i]+=1
                update=None
                if predictor is not None and hasattr(predictor,"on_progress"):
                    update=predictor.on_progress(es[i],c,now-task_started[i],quarter,
                                                  max(0.0,eta_end[i]-now))
                if update is not None:
                    eta_end[i]=now+max(0.0,float(update))
                # Reconsider ownership after actual progress even if the
                # estimator has no learning update at this point.
                event(max(now,eta_end[i]-params.binding_lead_cycles),"bindcheck",c,i,0)
                if positions[c]+1<len(queues[c]) and c not in next_state:
                    ni=queues[c][positions[c]+1]
                    event(max(now,eta_end[i]-params.hbm_latency-
                              costs[ni,c].phases[0].first_weight_bytes/params.hbm_bandwidth),
                          "prefetch",c,i,0)
        for key in list(active):
            a=active[key]
            if a["remaining"]>1e-8:
                continue
            del active[key]
            c,i,pi=a["core"],a["task"],a["phase"]
            if key[0]=="prefetch":
                pf=next_state[c]
                pf["ready"]=now
                prefetched_bytes+=pf["bytes"]
                if pf.get("waiting"):
                    tasks[i]["first_weight_ready"]=now
                    event(now,"ready",c,i,0)
            else:
                s=a["cost"]
                task_done_work[i]+=s.useful_macs
                # The phase compute demand already includes its final drain.
                end=now
                if detail:
                    phase_rows.append({"expert_index":i,"core":c,"phase":pi,"start":a["began"],
                                       "stream_finish":now,"finish":end,**asdict(s)})
                event(end,"continue",c,i,pi+1)
    if waiting:
        raise RuntimeError("unbound tasks remain after all execution stopped")
    cycles=max(finishes)
    chosen=[costs[b["expert_index"],b["core"]] for b in bindings]
    useful=sum(t.useful_macs for t in chosen)
    issued=sum(t.issued_macs for t in chosen)
    unique=sum(t.unique_hbm_bytes for t in chosen)
    expected=sum(3*e["Me"]*e.get("H",2048)*e.get("F",1408) for e in es)
    if useful!=expected:
        raise AssertionError("useful MAC conservation")
    # Fluid accumulation is floating-point; exported byte counts are exact
    # declared task traffic (prefetch subtracts then replaces the same bytes).
    hbm=sum(t.hbm_bytes for t in chosen)
    if abs(actual_bytes-hbm)>max(1.0,hbm*1e-8):
        raise AssertionError(f"HBM traffic conservation: {actual_bytes} vs {hbm}")
    result={"workload":workload.get("id","anonymous"),"batch":batch,"cycles":cycles,
            "latency_ms":cycles/1e6,"useful_macs":useful,"issued_macs":issued,
            "padding_macs":issued-useful,"spatial_utilization":useful/issued,
            "wall_mac_utilization":useful/(design.total_macs*cycles),
            "hbm_bytes":hbm,"native_unique_bytes":unique,
            "w_sram_bytes":sum(t.w_sram_bytes for t in chosen),
            "x_sram_bytes":sum(t.x_sram_bytes for t in chosen),
            "acc_sram_bytes":sum(t.acc_sram_bytes for t in chosen)+clear_bytes,
            "core_finish_cycles":finishes,"core_compute_busy":core_busy,
            "w_port_busy":busy["W"],"x_port_busy":busy["X"],"acc_port_busy":busy["acc"],
            "hbm_busy_frac":busy["hbm"]/cycles,"hbm_busy_cycles":busy["hbm"],
            "hbm_actual_Bpc":hbm/cycles,"core_finish_gap":max(finishes)-min(finishes),
            "idle_frac":max(0.0,1-sum(core_busy)/(ncores*cycles)),
            "prefetched_bytes":int(prefetched_bytes),"wasted_prefetch_bytes":wasted_prefetch,
            "scope":SCOPE,"non_iso_resource_reference":params.onchip_mode=="fixed_issue",
            "first_weight_timing_scope":"observed shared-fluid ingress/fill prefix; not native per-request tile timing"}
    result["U2_weight_read_once_oracle"]=params.sram_weight_read_once_oracle
    result["U1_per_expert_shape_oracle"]=bool(shape_overrides)
    if detail:
        result.update(segments=[dict(s,end=min(s["end"],cycles)) for s in segments if s["start"]<cycles],supply_observation_scope="phase-fluid rate x 65-cycle latency-residency estimate; not native credit/request trace",bindings=bindings,tasks=[tasks[i] for i in sorted(tasks)],phases=phase_rows,
                      stream_attribution=stream_attribution,ledger=design.ledger(),
                      counters_warning="resource occupancy/progress overlaps; not native stall counters; never sum as wall time")
    return result


def _chunk_workload(workload: dict, rows: int) -> list[dict]:
    """Sequential bounded input/combine chunks, preserving token/expert counts."""
    batch=workload["batch"]
    experts=workload["experts"]
    token_ids={}
    if all("token_indices" in e for e in experts):
        token_ids={i:list(e["token_indices"]) for i,e in enumerate(experts)}
    else:
        # Degree-preserving deterministic bipartite realization; no invented
        # captured trace. This path is ONLY an internal storage decomposition.
        routed=[i for i,e in enumerate(experts) if not e.get("is_shared",False)]
        counts={i:experts[i]["Me"] for i in routed}
        topk=workload.get("top_k",sum(counts.values())//batch)
        if sum(counts.values())!=batch*topk or max(counts.values(),default=0)>batch:
            raise ValueError("large count-only workload needs valid top-k degree sequence")
        token_ids={i:[] for i in range(len(experts))}
        for token in range(batch):
            chosen=sorted(counts,key=lambda i:(-counts[i],i))[:topk]
            if len(chosen)<topk or any(counts[i]<=0 for i in chosen):
                raise ValueError("cannot realize supplied expert degree sequence")
            for i in chosen:
                token_ids[i].append(token)
                counts[i]-=1
        if any(counts.values()):
            raise ValueError("degree sequence realization failed")
        for i,e in enumerate(experts):
            if e.get("is_shared",False):
                token_ids[i]=list(range(e["Me"]))
    chunks=[]
    for first in range(0,batch,rows):
        count=min(rows,batch-first)
        es=[]
        for i,e in enumerate(experts):
            ids=[t-first for t in token_ids[i] if first<=t<first+count]
            if ids:
                es.append({**e,"Me":len(ids),"token_indices":ids,"original_expert_index":i})
        chunks.append({**workload,"id":f"{workload.get('id','anonymous')}:storage{first}",
                       "batch":count,"experts":es})
    return chunks


def storage_chunks(workload: dict) -> list[dict]:
    """The exact installed global X/combine/route decomposition used by execution."""
    hidden=workload.get("hidden",workload["experts"][0].get("H",2048) if workload["experts"] else 2048)
    topk=workload.get("top_k",6)
    rows=max(1,min(FIXED_STORAGE["global_X"]//(2*hidden),
                   FIXED_STORAGE["combined_Y"]//(4*hidden),
                   (FIXED_STORAGE["routes"]-8*64)//(16*topk)))
    return [workload] if workload["batch"]<=rows else _chunk_workload(workload,rows)


def simulate(workload: dict, design: Design, params: Parameters = Parameters(),
             policy: str = "eft", owners: Iterable[int] | None = None,
             predictor=None, detail: bool = True,
             cost_overrides: dict | None=None,shape_overrides: dict | None=None) -> dict:
    """One globally shared-HBM post-router layer; deterministic for fixed seed.

    Main results require total_macs=12,288. Synthetic MAC scaling is explicit
    in Design.total_macs. Large synthetic windows execute sequential finite
    X/Y/route-storage chunks and pay off-chip activation spill traffic.
    """
    batch=workload["batch"]
    if batch<1 or any(not 0<e["Me"]<=batch for e in workload["experts"]):
        raise ValueError("Me must lie inside the declared batch")
    hidden=workload.get("hidden",workload["experts"][0].get("H",2048) if workload["experts"] else 2048)
    topk=workload.get("top_k",6)
    chunks=storage_chunks(workload)
    if len(chunks)==1:
        return _one_layer(workload,design,params,policy,None if owners is None else tuple(owners),predictor,detail,
                          cost_overrides,shape_overrides)
    if cost_overrides or shape_overrides:
        raise ValueError("chunked U1 override needs per-storage-chunk task costs, not whole-expert costs")
    owner_map=None if owners is None else tuple(owners)
    results=[]
    offset=0.0
    for cw in chunks:
        os=None if owner_map is None else tuple(owner_map[e["original_expert_index"]] for e in cw["experts"])
        r=_one_layer(cw,design,params,policy,os,predictor,detail)
        # Offsets are observation-only; physical spill/service stays identical.
        if detail:
            chunk_index=len(results)
            for t in r["tasks"]:
                t["chunk_index"]=chunk_index
                t["original_expert_index"]=cw["experts"][t["expert_index"]]["original_expert_index"]
                for name in ("start","finish","predicted_finish","first_weight_ready","current_end","prefetch_request"):
                    if t.get(name) is not None:
                        t[name]+=offset
            for b in r["bindings"]:
                b["chunk_index"]=chunk_index
                b["original_expert_index"]=cw["experts"][b["expert_index"]]["original_expert_index"]
                for name in ("bind_cycle","predicted_finish"):
                    if b.get(name) is not None:b[name]+=offset
            for segment in r.get("segments",[]):
                segment["start"]+=offset
                segment["end"]+=offset
            for ph in r["phases"]:
                ph["chunk_index"]=chunk_index
                ph["original_expert_index"]=cw["experts"][ph["expert_index"]]["original_expert_index"]
                for name in ("start","stream_finish","finish"):
                    if ph.get(name) is not None:ph[name]+=offset
        # Finite global X/combine backing. Output of each completed chunk is
        # drained before reuse; no free enlargement for B256/E256 studies.
        io=cw["batch"]*hidden*6
        spill_time=params.hbm_latency+io/params.hbm_bandwidth
        r["hbm_bytes"]+=io
        r["cycles"]+=spill_time
        r["activation_spill_bytes"]=io
        results.append(r)
        offset+=r["cycles"]
    useful=sum(r["useful_macs"] for r in results)
    issued=sum(r["issued_macs"] for r in results)
    result={"workload":workload.get("id","anonymous"),"batch":batch,"cycles":offset,"latency_ms":offset/1e6,
            "useful_macs":useful,"issued_macs":issued,"padding_macs":issued-useful,
            "spatial_utilization":useful/issued,"hbm_bytes":sum(r["hbm_bytes"] for r in results),
            "native_unique_bytes":sum(2*e.get("F",1408)*ceildiv(2*e.get("H",2048),32)*32+
                                      e.get("H",2048)*ceildiv(2*e.get("F",1408),32)*32 for e in workload["experts"]),
            "w_sram_bytes":sum(r["w_sram_bytes"] for r in results),
            "x_sram_bytes":sum(r["x_sram_bytes"] for r in results),
            "acc_sram_bytes":sum(r["acc_sram_bytes"] for r in results),
            "core_compute_busy":[sum(r["core_compute_busy"][c] for r in results) for c in range(len(design.cores))],
            "core_finish_cycles":[offset]*len(design.cores),
            "w_port_busy":sum(r["w_port_busy"] for r in results),
            "x_port_busy":sum(r["x_port_busy"] for r in results),
            "acc_port_busy":sum(r["acc_port_busy"] for r in results),
            "hbm_busy_frac":sum(r["hbm_bytes"] for r in results)/(params.hbm_bandwidth*offset),
            "core_finish_gap":0.0,"storage_chunks":len(chunks),
            "activation_spill_bytes":sum(r["activation_spill_bytes"] for r in results),
            "scope":SCOPE+"; sequential input/combine chunks, explicit HBM spill",
            "non_iso_resource_reference":params.onchip_mode=="fixed_issue"}
    if detail:
        result.update(segments=[s for r in results for s in r.get("segments",[])],
                      supply_observation_scope="phase-fluid latency-residency proxy, not native credit/request trace",
                      tasks=[t for r in results for t in r["tasks"]],
                      bindings=[b for r in results for b in r["bindings"]],
                      phases=[p for r in results for p in r["phases"]],chunk_results=results,ledger=design.ledger())
    return result


def micro_cost(shape: Core, m: int, h: int = 2048, f: int = 1408,
               flow: str = "OS", params: Parameters = Parameters()) -> TaskCost:
    """Micro-only explicit full bank/capacity quotas; NOT iso-MAC main result."""
    d=Design((shape,),flows=(flow,),total_macs=shape.macs,label="micro_explicit_full_private_quotas")
    return task_cost({"Me":m,"H":h,"F":f},d,0,params)


def shape_oracle(workload: dict, design: Design, params: Parameters=Parameters(),
                 *,detail: bool=True, owners=None,policy="eft") -> dict:
    """U1: per-expert best isolated shape; same ledger, zero reshape overhead.

    This is a diagnostic reconfiguration oracle, not a realizable NPU and not
    a proof of the globally optimal joint shape/prefetch schedule.
    """
    if len(design.cores)!=1 or design.total_macs!=12288:
        raise ValueError("U1 requires one 12,288-multiplier array")
    from dataclasses import replace
    from research.moe_dispatch.geometry3d.compute import enumerate_geometries
    choices=[replace(design,cores=cs,label="") for cs in enumerate_geometries() if len(cs)==1]
    overrides,shapes={},{}
    for i,e in enumerate(workload["experts"]):
        candidates=[]
        for d in choices:
            try:
                co=task_cost(e,d,0,params)
            except ValueError:
                continue
            candidates.append((co.isolated_cycles,d.geometry,co,d.cores[0]))
        if not candidates:
            raise ValueError("U1 expert has no legal shape within frozen private ledger")
        _,_,overrides[i],shapes[i]=min(candidates,key=lambda r:(r[0],r[1]))
    r=simulate(workload,design,params,cost_overrides=overrides,shape_overrides=shapes,detail=detail,owners=owners,policy=policy)
    r["oracle_selection"]="minimum isolated task cost among44 declared single geometries; zero shape switch"
    return r
