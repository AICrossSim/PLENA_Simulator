"""Finite shared weight-cache and DMA sensitivity model, not native-cycle E2E.

The compute costs are inherited unchanged from ``plena65536_affinity``.  A
column-strip OS traversal retains each PN-wide, full-K weight strip while all
M rows consume it.  This differs from the historical M-outer OS loop; it is
applied identically to every organization.  X/Z and output contexts use the
reserved SRAM, with ideal internal ports.  A single shared DMA serializes
bursts, but DMA and computation overlap on *single-core as well as dual-core*
designs.  Thus this model never manufactures a single-core no-overlap rule.

Capacity is bytes, not an expert-count approximation: the default 64 MiB
budget reserves 8 MiB for activation/local/context/metadata and provides
56 MiB for weights.  Pinned Shared reserves 33 MiB *inside* that pool, leaving
23 MiB.  Requests stage two strips per core, rather than two whole experts.
Warm initialization is explicitly charged as a residency assumption; it is
not available for free in a cold invocation.  No whole-expert cache thrashing
or speculative wrong-path fetch is invented: each demanded weight strip is
consumed once per window, unless an explicit future model changes that loop.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
from math import inf, isfinite

from research.moe_dispatch.plena65536_affinity.model import schedule
from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_shared.timing import PROFILES, divup

MIB = 1024 * 1024
DEFAULT_SRAM_BYTES = 64 * MIB
DEFAULT_RESERVED_BYTES = 8 * MIB
FROZEN_CORES = {
    "single": (Core(1, 512, 1, 128),),
    "homogeneous": (Core(1, 256, 1, 128), Core(1, 256, 1, 128)),
    "heterogeneous": (Core(1, 64, 1, 704), Core(1, 128, 1, 160)),
}


@dataclass(frozen=True)
class Block:
    key: tuple
    expert_id: int
    phase: str
    n_start: int
    n_stop: int
    nbytes: int
    cycles: int
    shared: bool


@dataclass
class Entry:
    nbytes: int
    shared: bool
    pinned: bool
    ready: bool = True
    refs: int = 0


class WeightCache:
    """Payload LRU with in-flight/consumer protection and real pin reservation.

    Pass the same object to repeated ``run_window`` calls only when weight
    identities actually persist.  By default the window ID is the namespace;
    callers wishing to reuse across windows must explicitly provide a common
    ``weight_namespace`` for the same layer/checkpoint.
    """
    def __init__(self, capacity_bytes=56 * MIB, policy="lru", pinned_bytes=0):
        if policy not in ("lru", "pin_shared"):
            raise ValueError("cache policy must be lru or pin_shared")
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("cache capacity must be a positive integer")
        if type(pinned_bytes) is not int or not 0 <= pinned_bytes <= capacity_bytes:
            raise ValueError("pinned reservation must fit inside weight capacity")
        self.capacity_bytes = capacity_bytes
        self.policy = policy
        self.pinned_bytes = pinned_bytes if policy == "pin_shared" else 0
        self.entries = OrderedDict()
        self.used_bytes = 0
        self.nonpinned_bytes = 0
        self.peak_bytes = 0
        self.evictions = 0
        self.evicted_bytes = 0
        self.shared_evicted_bytes = 0

    def _remove(self, key):
        entry = self.entries.pop(key)
        assert not entry.refs and not entry.pinned
        self.used_bytes -= entry.nbytes
        self.nonpinned_bytes -= entry.nbytes
        self.evictions += 1
        self.evicted_bytes += entry.nbytes
        self.shared_evicted_bytes += entry.nbytes if entry.shared else 0

    def reserve(self, block, *, ready=False):
        """Reserve a missing payload; return None if live references prevent it."""
        if block.key in self.entries:
            raise ValueError("reserve requires a missing cache key")
        pinned = self.policy == "pin_shared" and block.shared
        limit = self.capacity_bytes if pinned else self.capacity_bytes - self.pinned_bytes
        used = self.used_bytes if pinned else self.nonpinned_bytes
        if block.nbytes > limit:
            raise ValueError("One column strip exceeds its weight-cache partition")
        if pinned:
            pinned_loaded = self.used_bytes - self.nonpinned_bytes
            if pinned_loaded + block.nbytes > self.pinned_bytes:
                raise ValueError("Pinned keys exceed reservation; clear cache between distinct layers")
        # Check available victims first; a rejected request must not partially evict.
        excess = used + block.nbytes - limit
        if excess > 0:
            victims = []
            freed = 0
            for key, entry in self.entries.items():
                if entry.pinned or entry.refs:
                    continue
                victims.append(key)
                freed += entry.nbytes
                if freed >= excess:
                    break
            if freed < excess:
                return None
            for key in victims:
                self._remove(key)
        entry = Entry(block.nbytes, block.shared, pinned, ready=ready, refs=1)
        self.entries[block.key] = entry
        self.used_bytes += block.nbytes
        self.nonpinned_bytes += 0 if pinned else block.nbytes
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        assert self.used_bytes <= self.capacity_bytes
        assert self.nonpinned_bytes <= self.capacity_bytes - self.pinned_bytes
        return entry

    def acquire(self, block):
        entry = self.entries.get(block.key)
        if entry is not None:
            if entry.nbytes != block.nbytes or entry.shared != block.shared:
                raise ValueError("Cache-key collision across incompatible weight layouts")
            entry.refs += 1
            self.entries.move_to_end(block.key)
        return entry

    def release(self, block):
        entry = self.entries[block.key]
        assert entry.refs > 0 and entry.ready
        entry.refs -= 1
        self.entries.move_to_end(block.key)

    def seed(self, blocks):
        """Install already-resident payloads before timing, without DMA charges."""
        for block in blocks:
            if block.key in self.entries:
                continue
            entry = self.reserve(block, ready=True)
            if entry is None:
                raise ValueError("Initial resident set cannot be installed")
            entry.refs = 0


def blocks_for_job(core, job, namespace, profile=PROFILES[0]):
    """Exact weight bytes and feed/drain partition for the inherited job cost."""
    if profile.serialized:
        raise ValueError("This model currently implements pipelined compute profiles")
    m, h, f = job["M"], job["H"], job["F"]
    result = []
    for name, k, n in (("gate", h, f), ("up", h, f), ("down", f, h)):
        for start in range(0, n, core.pn):
            stop = min(start + core.pn, n)
            cycles = divup(m, core.pm) * divup(k, core.r) * core.b
            if stop == n and name in ("up", "down"):
                cycles += profile.drain(core)
            result.append(Block((str(namespace), job["expert_id"], name, start, stop, k),
                                job["expert_id"], name, start, stop, 2 * k * (stop - start),
                                cycles, job["is_shared"]))
    assert sum(block.cycles for block in result) == job["cycles"]
    return result


def run_window(cores, window, *, policy="workload_aware", threshold=2,
               cache_policy="lru", prefetch=True, initial="cold",
               bandwidth_gbs=512.0, request_latency_ns=64.0,
               total_sram_bytes=DEFAULT_SRAM_BYTES,
               reserved_sram_bytes=DEFAULT_RESERVED_BYTES,
               cache=None, weight_namespace=None, debug=False):
    """Run one complete frozen-assignment window with one finite global DMA.

    Times are nanoseconds, equal to 1 GHz cycles.  ``bandwidth_gbs`` is decimal
    GB/s, numerically bytes/ns.  A transfer occupies the one DMA server for
    ``request_latency_ns + nbytes / bandwidth_gbs``.  This coarse burst model
    deliberately does not claim native HBM bank/credit/controller accuracy.
    ``shared_resident`` seeds Shared only and reports the uncharged initial
    bytes.  An externally supplied cache is left intact for explicit repeats.
    """
    cores = tuple(cores)
    if bandwidth_gbs <= 0 or bandwidth_gbs != bandwidth_gbs:
        raise ValueError("Bandwidth must be positive (infinity is allowed)")
    if not isfinite(request_latency_ns) or request_latency_ns < 0:
        raise ValueError("Request latency must be finite and nonnegative")
    if initial not in ("cold", "shared_resident"):
        raise ValueError("initial must be cold or shared_resident")
    if type(prefetch) is not bool:
        raise ValueError("prefetch must be boolean")
    if not 0 <= reserved_sram_bytes < total_sram_bytes:
        raise ValueError("Reserved SRAM must leave positive weight capacity")
    plan = schedule(cores, window, policy=policy, threshold=threshold)
    namespace = weight_namespace
    if namespace is None:
        namespace = window.get("window_id", window.get("id", "anonymous_window"))
    streams = [[] for _ in cores]
    for job in plan["jobs"]:
        streams[job["core_id"]].extend(blocks_for_job(cores[job["core_id"]], job, namespace))
    shared_blocks = [b for stream in streams for b in stream if b.shared]
    shared_bytes = sum(b.nbytes for b in shared_blocks)
    capacity = total_sram_bytes - reserved_sram_bytes
    if cache is None:
        cache = WeightCache(capacity, cache_policy,
                            shared_bytes if cache_policy == "pin_shared" else 0)
    elif (cache.capacity_bytes != capacity or cache.policy != cache_policy
          or cache.pinned_bytes != (shared_bytes if cache_policy == "pin_shared" else 0)):
        raise ValueError("External cache must match frozen capacity and policy")
    if any(e.refs or not e.ready for e in cache.entries.values()):
        raise ValueError("Cache reuse requires a completed preceding invocation")
    explicitly_seeded_bytes = (sum(b.nbytes for b in shared_blocks if b.key not in cache.entries)
                               if initial == "shared_resident" else 0)
    if initial == "shared_resident":
        cache.seed(shared_blocks)
    initial_keys = set(cache.entries)
    initial_resident_bytes = cache.used_bytes
    initial_shared_bytes = sum(e.nbytes for e in cache.entries.values() if e.shared)
    evict0 = cache.evictions
    evictbytes0 = cache.evicted_bytes
    sharedevict0 = cache.shared_evicted_bytes
    old_peak = cache.peak_bytes
    cache.peak_bytes = cache.used_bytes
    indices = [0] * len(cores)
    pending = [deque() for _ in cores]
    active = [None] * len(cores)
    finish = [None] * len(cores)
    core_finish = [0.0] * len(cores)
    core_bytes = [0] * len(cores)
    core_hits = [0] * len(cores)
    per_core_dma_ns = [0.0] * len(cores)
    shared_hits = shared_misses = hbm_bytes = dma_requests = 0
    refetch_bytes = resident_reload_bytes = 0
    fetched_keys = set()
    dma = None
    dma_busy_ns = 0.0
    now = 0.0
    rr = 0
    max_pending = 2 if prefetch else 1
    peak_pending = [0] * len(cores)
    trace = [] if debug else None

    def log(kind, core, block, **kwargs):
        if trace is not None:
            trace.append({"event": kind, "time_ns": now, "core": core,
                          "key": list(block.key), "bytes": block.nbytes, **kwargs})

    while True:
        progressed = False
        if dma is not None and dma[0] <= now:
            _, owner, block, entry = dma
            entry.ready = True
            fetched_keys.add(block.key)
            log("dma_end", owner, block)
            dma = None
            progressed = True
        for owner in range(len(cores)):
            if active[owner] is not None and finish[owner] <= now:
                block = active[owner]
                log("compute_end", owner, block)
                cache.release(block)
                assert pending[owner].popleft() is block
                active[owner] = finish[owner] = None
                core_finish[owner] = now
                progressed = True
        # Ready current blocks issue before new demand/prefetch arbitration.
        for owner in range(len(cores)):
            if active[owner] is None and pending[owner]:
                block = pending[owner][0]
                if cache.entries[block.key].ready:
                    active[owner] = block
                    finish[owner] = now + block.cycles
                    log("compute_start", owner, block, end_ns=finish[owner])
                    progressed = True
        # Cache hits do not occupy the DMA. Miss requests use RR arbitration.
        rr_start = rr
        for offset in range(len(cores)):
            owner = (rr_start + offset) % len(cores)
            if len(pending[owner]) >= max_pending or indices[owner] == len(streams[owner]):
                continue
            block = streams[owner][indices[owner]]
            entry = cache.acquire(block)
            if entry is not None:
                core_hits[owner] += block.nbytes
                shared_hits += block.nbytes if block.shared else 0
                pending[owner].append(block)
                indices[owner] += 1
                log("cache_hit", owner, block)
                progressed = True
            elif dma is None:
                entry = cache.reserve(block)
                if entry is None:
                    continue
                duration = request_latency_ns + block.nbytes / bandwidth_gbs
                dma = (now + duration, owner, block, entry)
                core_bytes[owner] += block.nbytes
                hbm_bytes += block.nbytes
                shared_misses += block.nbytes if block.shared else 0
                dma_requests += 1
                dma_busy_ns += duration
                per_core_dma_ns[owner] += duration
                refetch_bytes += block.nbytes if block.key in fetched_keys else 0
                resident_reload_bytes += block.nbytes if block.key in initial_keys else 0
                pending[owner].append(block)
                indices[owner] += 1
                log("dma_start", owner, block, end_ns=now + duration)
                rr = (owner + 1) % len(cores)
                progressed = True
            peak_pending[owner] = max(peak_pending[owner], len(pending[owner]))
        if all(indices[c] == len(streams[c]) and not pending[c] for c in range(len(cores))):
            assert dma is None and all(v is None for v in active)
            break
        if progressed:
            continue
        next_events = [t for t in finish if t is not None]
        if dma is not None:
            next_events.append(dma[0])
        if not next_events:
            raise RuntimeError("Deadlock: no progress event (check cache partition and strip size)")
        next_time = min(next_events)
        if next_time < now:
            raise AssertionError("Time moved backwards")
        now = next_time

    assert sum(core_bytes) == hbm_bytes
    assert shared_hits + shared_misses == shared_bytes
    assert all(e.refs == 0 and e.ready for e in cache.entries.values())
    assert now + 1e-6 >= plan["cycles"]
    assert now + 1e-6 >= dma_busy_ns
    run_peak = cache.peak_bytes
    cache.peak_bytes = max(old_peak, run_peak)
    result = {
        "window_id": str(namespace), "batch": window["batch"],
        "policy": policy, "selected_policy": plan["selected_policy"],
        "selected_threshold": plan["selected_threshold"],
        "cache_policy": cache_policy, "prefetch": prefetch, "initial": initial,
        "bandwidth_gbs": bandwidth_gbs, "request_latency_ns": request_latency_ns,
        "total_cycles": now, "latency_ms": now / 1e6,
        "compute_cycles": plan["cycles"], "compute_ms": plan["cycles"] / 1e6,
        "ideal_compute_cycles": plan["cycles"],
        "exposed_memory_cycles": now - plan["cycles"],
        "stall_cycles": now - plan["cycles"],
        "stall_ms": (now - plan["cycles"]) / 1e6,
        "hbm_bytes": hbm_bytes, "hbm_MiB": hbm_bytes / MIB,
        "unique_weight_bytes": plan["unique_weight_bytes"],
        "weight_lower_bound_ns": hbm_bytes / bandwidth_gbs,
        "dma_busy_ns": dma_busy_ns, "dma_requests": dma_requests,
        "hbm_busy_cycles": dma_busy_ns,
        "dma_busy_fraction": dma_busy_ns / now,
        "effective_bandwidth_gbs": hbm_bytes / now,
        "bandwidth_utilization": hbm_bytes / (now * bandwidth_gbs),
        "shared_demand_bytes": shared_bytes, "shared_request_bytes": shared_bytes,
        "shared_hit_bytes": shared_hits,
        "shared_miss_bytes": shared_misses,
        "shared_hit_fraction": shared_hits / shared_bytes,
        "cache_evictions": cache.evictions - evict0,
        "evicted_bytes": cache.evicted_bytes - evictbytes0,
        "shared_evicted_bytes": cache.shared_evicted_bytes - sharedevict0,
        "refetch_bytes": refetch_bytes, "resident_reload_bytes": resident_reload_bytes,
        "initial_resident_bytes": initial_resident_bytes,
        "initial_shared_resident_bytes": initial_shared_bytes,
        "explicitly_seeded_bytes": explicitly_seeded_bytes,
        "sram_total_bytes": total_sram_bytes, "sram_reserved_bytes": reserved_sram_bytes,
        "weight_capacity_bytes": capacity, "pinned_reservation_bytes": cache.pinned_bytes,
        "cache_peak_bytes": run_peak, "cache_final_bytes": cache.used_bytes,
        "staging_slots_per_core": max_pending, "peak_staged_blocks": peak_pending,
        "core_compute_cycles": plan["core_cycles"], "core_finish_cycles": core_finish,
        "core0_compute_cycles": plan["core_cycles"][0],
        "core1_compute_cycles": plan["core_cycles"][1] if len(cores) > 1 else 0,
        "core_wait_cycles": [core_finish[c] - plan["core_cycles"][c] for c in range(len(cores))],
        "core_idle_tail_cycles": [now - value for value in core_finish],
        "core_hbm_bytes": core_bytes, "core_cache_hit_bytes": core_hits,
        "core_dma_service_ns": per_core_dma_ns,
        "useful_MACs": plan["useful_MACs"],
        "effective_GFLOPs": 2 * plan["useful_MACs"] / now,
        "shared_core": plan["boundary_stats"]["shared_core"],
        "core_expert_ids": [[j["expert_id"] for j in plan["jobs"] if j["core_id"] == c]
                            for c in range(len(cores))],
        "model": "column_strip_OS_finite_weight_cache_single_DMA_ideal_internal_ports",
    }
    if trace is not None:
        result["events"] = trace
    return result
