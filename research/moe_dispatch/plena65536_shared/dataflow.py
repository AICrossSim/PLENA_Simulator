"""Declared OS/WS/IS supply loops for the Shared-only compute experiment.

These formulas are traffic and capacity accounting, not a native PLENA timing
model.  In particular WS/IS require proposed operand retention and partial
store capabilities; merely changing a name does not add those to PLENA.

All three loops execute the same masked arithmetic tiles.  OS is m,n,k with
one output tile held through all K segments.  WS is k,n,m with one weight tile
held across all M tiles.  IS is k,m,n with one input tile held across all N
tiles.  WS/IS spill partials between K visits into a separately charged scratch
store.  A unified resident pool changes the boundary those reads cross: they
are not automatically additional external HBM reads.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from copy import deepcopy
from math import ceil
from typing import Any

BF16_BYTES = 2
FP32_BYTES = 4
HIDDEN = 2048
INTERMEDIATE = 2816
FLOWS = ("OS", "WS", "IS")


def _integer(value: Any, name: str, *, zero: bool = False) -> int:
    minimum = 0 if zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def shape_tuple(core: Any) -> tuple[int, int, int]:
    if isinstance(core, Mapping):
        shape = tuple(core[key] for key in ("pm", "pn", "pk"))
    elif all(hasattr(core, key) for key in ("pm", "pn", "pk")):
        shape = tuple(getattr(core, key) for key in ("pm", "pn", "pk"))
    else:
        shape = tuple(core)
    if len(shape) != 3:
        raise ValueError("core shape requires exactly PM, PN, PK")
    return tuple(_integer(value, "physical dimension") for value in shape)


def gemm_traffic(m: int, n: int, k: int, pm: int, pn: int, pk: int,
                 flow: str = "OS", *, double_buffer: bool = True) -> dict[str, Any]:
    """A disjoint output rectangle, with the complete reduction dimension.

    Masked traffic transfers just valid elements; full traffic transfers padded
    physical tiles.  Masked bytes are optimistic lower bounds and do not imply
    that physical SRAM banks support free partial transactions.

    Nonstationary operands are refilled each issue; only the named stationary
    operand is retained.  This does not exploit accidental reuse when an outer
    loop has a single iteration.  More extensive reuse requires a new contract.

    Array delivery means operands presented to a tile, even when the same
    weight/input latch remains resident.  Pool refills count latch loads.  The
    two boundaries must never be summed as if both were external traffic.
    """
    m, n, k = (_integer(m, "M", zero=True), _integer(n, "N", zero=True),
               _integer(k, "K"))
    pm, pn, pk = shape_tuple((pm, pn, pk))
    flow = flow.upper()
    if flow not in FLOWS:
        raise ValueError(f"unknown dataflow {flow!r}")
    nm, nn, nk = ceil(m / pm), ceil(n / pn), ceil(k / pk)
    records, issues = nm * nn, nm * nn * nk
    active = bool(records)
    m_pad, n_pad, k_pad = nm * pm, nn * pn, nk * pk
    w_mask, x_mask = n * k * BF16_BYTES, m * k * BF16_BYTES
    w_full, x_full = n_pad * k_pad * BF16_BYTES, m_pad * k_pad * BF16_BYTES
    if not active:
        w_mask = x_mask = w_full = x_full = 0
    # Factorization is exact, including an edge M/N/K tile.
    w_multiplier = nm if flow in ("OS", "IS") else 1
    x_multiplier = nn if flow in ("OS", "WS") else 1
    partial_mask, partial_full = m * n * FP32_BYTES, records * pm * pn * FP32_BYTES
    intermediate_visits = max(0, nk - 1)
    spill = flow != "OS"
    pool = {
        "weight_masked_bytes": w_mask * w_multiplier,
        "weight_full_bytes": w_full * w_multiplier,
        "x_masked_bytes": x_mask * x_multiplier,
        "x_full_bytes": x_full * x_multiplier,
        "partial_read_masked_bytes": partial_mask * intermediate_visits if spill else 0,
        "partial_read_full_bytes": partial_full * intermediate_visits if spill else 0,
        "partial_write_masked_bytes": partial_mask * intermediate_visits if spill else 0,
        "partial_write_full_bytes": partial_full * intermediate_visits if spill else 0,
        "final_output_masked_bytes": partial_mask,
        "final_output_full_bytes": partial_full,
    }
    array = {
        "weight_masked_bytes": nm * n * k * BF16_BYTES,
        "weight_full_bytes": issues * pn * pk * BF16_BYTES,
        "x_masked_bytes": nn * m * k * BF16_BYTES,
        "x_full_bytes": issues * pm * pk * BF16_BYTES,
        "partial_read_masked_bytes": partial_mask * intermediate_visits,
        "partial_read_full_bytes": partial_full * intermediate_visits,
        "partial_write_masked_bytes": partial_mask * nk,
        "partial_write_full_bytes": partial_full * nk,
    }
    if not active:
        array = {key: 0 for key in array}
    factor = 2 if double_buffer else 1
    w_tile = pn * pk * BF16_BYTES if active else 0
    x_tile = pm * pk * BF16_BYTES if active else 0
    local_acc = pm * pn * FP32_BYTES if active else 0
    # OS retains a single output tile locally.  Other loops require one output
    # tile accumulator plus full projection scratch; this is separate from the
    # eventual completed output, whose storage lifetime is phase-dependent.
    scratch = partial_full if active and spill and nk > 1 else 0
    capacity = {
        "weight_tile_bytes": w_tile,
        "x_tile_bytes": x_tile,
        "operand_buffers_bytes": factor * (w_tile + x_tile),
        "local_accumulator_bytes": local_acc,
        "partial_scratch_masked_bytes": partial_mask if scratch else 0,
        "partial_scratch_full_bytes": scratch,
        "completed_output_masked_bytes": partial_mask,
        "local_plus_partial_peak_bytes": factor * (w_tile + x_tile) + local_acc + scratch,
        "double_buffer": double_buffer,
    }
    return {
        "M": m, "N": n, "K": k, "pm": pm, "pn": pn, "pk": pk,
        "dataflow": flow,
        "loop_order": {"OS": "m,n,k", "WS": "k,n,m", "IS": "k,m,n"}[flow],
        "m_tiles": nm, "n_tiles": nn, "k_tiles": nk,
        "output_records": records, "issues": issues,
        "useful_macs": m * n * k, "issued_macs": issues * pm * pn * pk,
        "unique_weight_bytes": w_mask, "unique_x_bytes": x_mask,
        "weight_refill_factor": w_multiplier if active else 0,
        "x_refill_factor": x_multiplier if active else 0,
        "pool_to_local": pool, "array_delivery": array, "capacity": capacity,
        "native_support_claimed": False,
    }


def plena_shape_tuple(core: Any) -> tuple[int, int, int, int]:
    """Native mini-array geometry (b, logical R, output replicas rM/rN)."""
    names = ("b", "r", "rm", "rn")
    if isinstance(core, Mapping):
        if not all(key in core for key in names):
            names = ("b", "R", "rM", "rN")
        shape = tuple(core[key] for key in names)
    elif all(hasattr(core, key) for key in names):
        shape = tuple(getattr(core, key) for key in names)
    elif all(hasattr(core, key) for key in ("b", "R", "rM", "rN")):
        shape = tuple(getattr(core, key) for key in ("b", "R", "rM", "rN"))
    else:
        shape = tuple(core)
    if len(shape) != 4:
        raise ValueError("native geometry requires b,R,rM,rN")
    b, reduction, rm, rn = tuple(_integer(value, "native dimension") for value in shape)
    if reduction % b:
        raise ValueError("logical reduction R must be divisible by mini-array b")
    return b, reduction, rm, rn


def plena_gemm_traffic(m: int, n: int, k: int, b: int, reduction: int,
                       rm: int = 1, rn: int = 1, flow: str = "OS",
                       *, double_buffer: bool = True) -> dict[str, Any]:
    """Topology-consistent supply accounting for native OS mini-array PEs.

    Logical tiles are PM=rM*b, PN=rN*b, R.  G=R/b mini-arrays execute b
    temporal MAC steps each.  Physical multipliers are PM*PN*G, not PM*PN*R.
    Each PE has one FP32 partial.  Across large-K segments these G partials
    persist; M_SUM reduces them only after all K segments.  Proposed WS/IS
    *operand* retention loops keep OS PEs and save/restore all G PE contexts
    when switching output tiles.  They are not native switchable WS/IS PEs.

    Scratch counts deliberately preserve expanded G partials, rather than
    inserting an uncharged early M_SUM for every K segment.
    """
    b, reduction, rm, rn = plena_shape_tuple((b, reduction, rm, rn))
    pm, pn, groups = rm * b, rn * b, reduction // b
    got = deepcopy(gemm_traffic(m, n, k, pm, pn, reduction, flow,
                               double_buffer=double_buffer))
    got["geometry_backend"] = "plena_mini_array_OS_with_operand_loop_extension"
    got["native_geometry"] = {"b": b, "R": reduction, "rM": rm, "rN": rn,
                               "PM": pm, "PN": pn, "G": groups,
                               "physical_multipliers": pm * pn * groups,
                               "temporal_steps_per_segment": b}
    got["physical_multipliers"] = pm * pn * groups
    got["spatial_K_groups"] = groups
    got["logical_reduction_per_segment"] = reduction
    got["native_support_claimed"] = flow.upper() == "OS" and rm == rn == 1
    # Local accumulator is a set of G physical PE partials.  Proposed operand
    # reuse loops move that entire context across K visits, not one final sum.
    cap = got["capacity"]
    for key in ("local_accumulator_bytes", "partial_scratch_masked_bytes", "partial_scratch_full_bytes"):
        cap[key] *= groups
    cap["local_plus_partial_peak_bytes"] = (cap["operand_buffers_bytes"]
                                           + cap["local_accumulator_bytes"]
                                           + cap["partial_scratch_full_bytes"])
    cap["shadow_partial_context_bytes"] = cap["local_accumulator_bytes"]
    cap["local_plus_partial_peak_bytes"] += cap["shadow_partial_context_bytes"]
    cap["PE_partial_context_multiplier"] = groups
    for key in ("partial_read_masked_bytes", "partial_read_full_bytes",
                "partial_write_masked_bytes", "partial_write_full_bytes"):
        got["pool_to_local"][key] *= groups
    # Reads/writes *across* the context-store-to-PE boundary are zero for OS;
    # internal register updates are recorded independently below.
    for key in ("partial_read_masked_bytes", "partial_read_full_bytes",
                "partial_write_masked_bytes", "partial_write_full_bytes"):
        got["array_delivery"][key] = got["pool_to_local"][key]
    active_groups = min(groups, ceil(k / b)) if got["issues"] else 0
    got["M_SUM"] = {
        "invocations": got["output_records"],
        "only_after_complete_K": True,
        "tree_leaf_full_bytes": got["output_records"] * pm * pn * groups * FP32_BYTES,
        "tree_leaf_masked_bytes": m * n * active_groups * FP32_BYTES,
        "tree_output_masked_bytes": m * n * FP32_BYTES,
        "tree_output_full_bytes": got["output_records"] * pm * pn * FP32_BYTES,
        "physical_adders": pm * pn * max(0, groups - 1),
        "useful_tree_additions": m * n * max(0, active_groups - 1),
    }
    got["internal_PE_accumulator"] = {
        "useful_MAC_update_write_bytes": got["useful_macs"] * FP32_BYTES,
        "issued_MAC_update_write_bytes": got["issued_macs"] * FP32_BYTES,
        "not_shared_SRAM_traffic": True,
    }
    return got


@dataclass(frozen=True)
class Region:
    """Complete-K rectangle in one Shared projection, mapped to one core."""
    core_id: int
    m_start: int
    m_stop: int
    n_start: int
    n_stop: int


def _region(value: Any) -> Region:
    if isinstance(value, Region):
        return value
    if isinstance(value, Mapping):
        return Region(**{key: value[key] for key in Region.__dataclass_fields__})
    return Region(*value)


def _validate_cover(regions: Sequence[Region], m: int, n: int, core_count: int) -> None:
    """Exact cover without allocating a potentially large M*N mask."""
    area = 0
    for i, region in enumerate(regions):
        if not 0 <= region.core_id < core_count:
            raise ValueError("region refers to a nonexistent core")
        if not (0 <= region.m_start < region.m_stop <= m and
                0 <= region.n_start < region.n_stop <= n):
            raise ValueError("region lies outside the projection or is empty")
        area += (region.m_stop - region.m_start) * (region.n_stop - region.n_start)
        for other in regions[:i]:
            if (max(region.m_start, other.m_start) < min(region.m_stop, other.m_stop) and
                    max(region.n_start, other.n_start) < min(region.n_stop, other.n_stop)):
                raise ValueError("output ownership regions overlap")
    if area != m * n:
        raise ValueError("output ownership regions do not cover the projection")


def _sum_dict(items: Sequence[Mapping[str, int]]) -> dict[str, int]:
    if not items:
        return {}
    return {key: sum(item[key] for item in items) for key in items[0]}


def shared_traffic(batch: int, cores: Sequence[Any], flows: Sequence[str] | str,
                   partitions: Mapping[str, Sequence[Any]] | None = None,
                   *, hidden: int = HIDDEN, intermediate: int = INTERMEDIATE,
                   double_buffer: bool = True, native_plena: bool = False) -> dict[str, Any]:
    """Shared Gate/Up/Down traffic for explicitly disjoint output ownership.

    ``partitions`` keys are gate, up, down; every value is a list of Region or
    equivalent dict/tuple.  Without partitions, core 0 owns all projections.
    Gate/Up are accounted separately (no assumed free fused-X reuse).  The
    whole-layer unified pool has all X/Z available and deduplicates W loads.
    WS/IS local refills thus do not alter external unique cold weight traffic.

    No-pool streaming is a *different system*: every local operand refill,
    partial scratch access and final intermediate write/read crosses HBM.
    Completed Gate/Up are FP32, Z is BF16, final Y is FP32.  Elementwise GU->Z
    traffic is included, while its compute cost is intentionally not timed.
    """
    batch, hidden, intermediate = (_integer(batch, "batch"), _integer(hidden, "hidden"),
                                    _integer(intermediate, "intermediate"))
    shapes = [(plena_shape_tuple(core) if native_plena else shape_tuple(core)) for core in cores]
    if not shapes:
        raise ValueError("at least one core is required")
    if isinstance(flows, str):
        flows = [flows] * len(shapes)
    if len(flows) != len(shapes):
        raise ValueError("one dataflow is required for each core")
    flows = [flow.upper() for flow in flows]
    if any(flow not in FLOWS for flow in flows):
        raise ValueError("unknown core dataflow")
    dimensions = {"gate": (batch, intermediate, hidden),
                  "up": (batch, intermediate, hidden),
                  "down": (batch, hidden, intermediate)}
    if partitions is None:
        partitions = {name: [Region(0, 0, m, 0, n)]
                      for name, (m, n, _) in dimensions.items()}
    if set(partitions) != set(dimensions):
        raise ValueError("partition keys must be exactly gate, up, down")
    projections = []
    for name, (m, n, k) in dimensions.items():
        regions = [_region(value) for value in partitions[name]]
        _validate_cover(regions, m, n, len(shapes))
        for region in regions:
            if native_plena:
                traffic = plena_gemm_traffic(region.m_stop - region.m_start,
                                             region.n_stop - region.n_start, k,
                                             *shapes[region.core_id], flows[region.core_id],
                                             double_buffer=double_buffer)
            else:
                traffic = gemm_traffic(region.m_stop - region.m_start,
                                       region.n_stop - region.n_start, k,
                                       *shapes[region.core_id], flows[region.core_id],
                                       double_buffer=double_buffer)
            traffic.update({"projection": name, "core_id": region.core_id,
                            "region": region.__dict__})
            projections.append(traffic)
    pool = _sum_dict([item["pool_to_local"] for item in projections])
    array = _sum_dict([item["array_delivery"] for item in projections])
    w_unique = 3 * hidden * intermediate * BF16_BYTES
    x_unique = batch * hidden * BF16_BYTES
    z_unique = batch * intermediate * BF16_BYTES
    gu_fp32 = 2 * batch * intermediate * FP32_BYTES
    y_fp32 = batch * hidden * FP32_BYTES
    stream_weight = pool["weight_masked_bytes"]
    stream_x = pool["x_masked_bytes"]
    stream_partial = pool["partial_read_masked_bytes"] + pool["partial_write_masked_bytes"]
    # pool final outputs already includes GU/Y writes.  Activation refills
    # already includes Down's Z reads; GU elementwise finalization needs the
    # additional completed G/U read and the single BF16 Z write.
    stream_intermediates = pool["final_output_masked_bytes"] + gu_fp32 + z_unique
    per_core = []
    for core_id, shape in enumerate(shapes):
        own = [item for item in projections if item["core_id"] == core_id]
        phase_caps = {}
        for phase in dimensions:
            phase_caps[phase] = sum(item["capacity"]["local_plus_partial_peak_bytes"]
                                    for item in own if item["projection"] == phase)
        phase_caps["gate_up_concurrent_upper"] = phase_caps["gate"] + phase_caps["up"]
        local_peak = max(phase_caps.values(), default=0)
        per_core.append({"core_id": core_id, "shape": list(shape), "dataflow": flows[core_id],
                         "pool_to_local": _sum_dict([item["pool_to_local"] for item in own]),
                         "array_delivery": _sum_dict([item["array_delivery"] for item in own]),
                         "local_and_partial_phase_peak_bytes": local_peak,
                         "phase_capacity_bytes": phase_caps})
    # Conservative sequential GU->Down data liveness envelope.  This explicitly
    # does not claim free intercore context storage or include area/bank metadata.
    activation_peak = x_unique + gu_fp32 + z_unique + y_fp32
    local_partial_peak = sum(item["local_and_partial_phase_peak_bytes"] for item in per_core)
    external = {
        "cold_weight_only_unique_bytes": w_unique,
        "cold_weights_inputs_final_output_bytes": w_unique + x_unique + y_fp32,
        "fully_resident_weight_replay_bytes": 0,
        "no_pool_stream_weight_bytes": stream_weight,
        "no_pool_stream_activation_bytes": stream_x,
        "no_pool_stream_partial_bytes": stream_partial,
        "no_pool_stream_intermediate_and_final_bytes": stream_intermediates,
        "no_pool_stream_total_bytes": stream_weight + stream_x + stream_partial + stream_intermediates,
        "no_pool_stream_weight_refetch_factor": stream_weight / w_unique,
        "unified_pool_external_weight_refetch_factor": 1.0,
    }
    return {"batch": batch, "hidden": hidden, "intermediate": intermediate,
            "cores": [list(shape) for shape in shapes], "dataflows": flows,
            "geometry_backend": "native_plena_explicit_replication" if native_plena else "spatial_dot_extension",
            "useful_macs": 3 * batch * hidden * intermediate,
            "issued_macs": sum(item["issued_macs"] for item in projections),
            "issues": sum(item["issues"] for item in projections),
            "external": external, "pool_to_local": pool, "array_delivery": array,
            "capacity": {"all_shared_weights_bytes": w_unique,
                         "input_X_bf16_bytes": x_unique, "intermediate_Z_bf16_bytes": z_unique,
                         "completed_GU_fp32_bytes": gu_fp32, "output_Y_fp32_bytes": y_fp32,
                         "activation_liveness_upper_bytes": activation_peak,
                         "local_and_partial_peak_sum_bytes": local_partial_peak,
                         "resident_pool_conservative_payload_bytes": w_unique + activation_peak + local_partial_peak,
                         "excludes_metadata_banks_alignment": True},
            "per_core": per_core, "projections": projections,
            "contract": {"loop_accounting_only": True, "native_WS_IS_support_claimed": False,
                         "gate_up_joint_X_reuse_assumed": False,
                         "unified_pool_external_reads_do_not_equal_local_refills": True}}


def plena_shared_traffic(batch: int, cores: Sequence[Any], flows: Sequence[str] | str,
                         partitions: Mapping[str, Sequence[Any]] | None = None,
                         **kwargs: Any) -> dict[str, Any]:
    """Native topology wrapper accepting root's Core(b,r,rm,rn) objects."""
    return shared_traffic(batch, cores, flows, partitions, native_plena=True, **kwargs)


def add_rates(traffic: Mapping[str, Any], compute_cycles: int | float,
              *, frequency_hz: float = 1e9, external_bandwidth_GB_s: float = 256.0) -> dict[str, Any]:
    """Attach necessary average rates and standalone bandwidth service times.

    Fetch time is bytes/BW; it is not additional compute time, a latency-aware
    HBM simulation, nor proof that double buffering hides all stalls.
    """
    if compute_cycles <= 0 or frequency_hz <= 0 or external_bandwidth_GB_s <= 0:
        raise ValueError("cycles, frequency and bandwidth must be positive")
    seconds = compute_cycles / frequency_hz
    ext = traffic["external"]
    rates = {}
    for prefix, byte_key in (("unique_weight", "cold_weight_only_unique_bytes"),
                             ("cold_all", "cold_weights_inputs_final_output_bytes"),
                             ("no_pool_stream", "no_pool_stream_total_bytes")):
        payload = ext[byte_key]
        rates[f"{prefix}_required_GB_s"] = payload / seconds / 1e9
        rates[f"{prefix}_fetch_ms"] = payload / (external_bandwidth_GB_s * 1e9) * 1e3
    rates["compute_ms"] = seconds * 1e3
    rates["useful_GFLOP_s"] = 2 * traffic["useful_macs"] / seconds / 1e9
    rates["spatial_utilization_pct"] = 100 * traffic["useful_macs"] / traffic["issued_macs"]
    for boundary in ("pool_to_local", "array_delivery"):
        rates[boundary] = {key.replace("bytes", "GB_s"): value / seconds / 1e9
                           for key, value in traffic[boundary].items()}
    return rates
