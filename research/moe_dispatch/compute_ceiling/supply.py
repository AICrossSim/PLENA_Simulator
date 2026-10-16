"""Reverse supply requirements for ready-operand BF16 compute experiments.

This module does not simulate HBM, SRAM stalls, DMA, banks or caches.  It
answers what traffic/capacity an ideal compute schedule would ask a supply
system to sustain.  Traffic at three distinct boundaries is kept separate:

* external cold-load traffic: each selected weight is fetched once;
* shared-pool -> local buffers: a declared loop and reuse strategy;
* local buffers/latches -> array: operands delivered for every issued tile.

The primary loop jointly traverses matching Gate/Up output records for each
K segment, keeping their shared X panel resident until both projections have
used it.  Array deliveries still include the two projections' distinct issues.

Calling a loop WS/OS/IS does not, by itself, change arithmetic issue counts
or guarantee equal execution time.  Alternative loop traffic is reported
without attaching the caller's OS schedule time to those alternatives.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from math import ceil
from typing import Any

BF16_BYTES = 2
FP32_BYTES = 4
DEFAULT_FREQUENCY_HZ = 1_000_000_000


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _core_shape(core: Any) -> tuple[int, int, int]:
    if isinstance(core, Mapping):
        shape = tuple(core[key] for key in ("pm", "pn", "pk"))
    elif all(hasattr(core, key) for key in ("pm", "pn", "pk")):
        shape = (core.pm, core.pn, core.pk)
    else:
        shape = tuple(core)
    if len(shape) != 3:
        raise ValueError("a core shape must contain PM, PN, PK")
    return tuple(_positive_int(x, "core dimension") for x in shape)


def _shape_traffic(m: int, n: int, k: int, shape: tuple[int, int, int]) -> dict[str, Any]:
    """One projection with all output records retained across K segments.

    Masked counts omit tail positions; full counts transfer physical slices.
    A masked SRAM transfer may still consume a full bank transaction, so the
    masked figures are lower bounds, not physical port guarantees.
    """
    pm, pn, pk = shape
    nm, nn, nk = ceil(m / pm), ceil(n / pn), ceil(k / pk)
    records = nm * nn
    issues = records * nk
    padded_m, padded_n, padded_k = nm * pm, nn * pn, nk * pk
    record_full = pm * pn * FP32_BYTES
    array = {
        "weight_masked_bytes": nm * n * k * BF16_BYTES,
        "weight_full_slice_bytes": issues * pn * pk * BF16_BYTES,
        "x_masked_bytes": nn * m * k * BF16_BYTES,
        "x_full_slice_bytes": issues * pm * pk * BF16_BYTES,
        "partial_read_masked_bytes": m * n * max(0, nk - 1) * FP32_BYTES,
        "partial_read_full_record_bytes": records * max(0, nk - 1) * record_full,
        "partial_write_masked_bytes": m * n * nk * FP32_BYTES,
        "partial_write_full_record_bytes": issues * record_full,
    }
    # Main schedule: K-major, N-tile-major, M-tile-minor.  Hold the whole
    # expert's X K-panel across N, and latch one W tile across M.  These
    # resident buffers, as well as all output records, are charged below.
    main_pool = {
        "weight_masked_bytes": n * k * BF16_BYTES,
        "weight_full_slice_bytes": padded_n * padded_k * BF16_BYTES,
        "x_masked_bytes": m * k * BF16_BYTES,
        "x_full_slice_bytes": padded_m * padded_k * BF16_BYTES,
    }
    # WS group sizes are number of N tiles held per K segment.  Groups use
    # ng -> k -> m -> n, preserving a W group across all M tiles and an X
    # tile across q N tiles.  X is read again for the next N group.
    ws_alternatives = []
    for q in sorted({1, min(4, nn), min(8, nn), nn}):
        groups = ceil(nn / q)
        ws_alternatives.append({
            "n_tiles_per_group": q,
            "n_group_count": groups,
            "weight_masked_bytes": n * k * BF16_BYTES,
            "weight_full_slice_bytes": padded_n * padded_k * BF16_BYTES,
            "x_masked_bytes": groups * m * k * BF16_BYTES,
            "x_full_slice_bytes": groups * padded_m * padded_k * BF16_BYTES,
            "weight_group_buffer_bytes": q * pn * pk * BF16_BYTES,
            "x_tile_buffer_bytes": pm * pk * BF16_BYTES,
            "partial_record_bytes": nm * q * pm * pn * FP32_BYTES,
            "timing": "traffic/capacity only; group schedule must be retimed independently",
        })
    # IS uses k -> m -> n: one X tile remains while all N tiles pass, so
    # every W tile is fetched again from the shared pool for the next M tile.
    is_pool = {
        "weight_masked_bytes": nm * n * k * BF16_BYTES,
        "weight_full_slice_bytes": nm * padded_n * padded_k * BF16_BYTES,
        "x_masked_bytes": m * k * BF16_BYTES,
        "x_full_slice_bytes": padded_m * padded_k * BF16_BYTES,
        "x_tile_buffer_bytes": pm * pk * BF16_BYTES,
        "weight_tile_buffer_bytes": pn * pk * BF16_BYTES,
        "partial_record_bytes": records * record_full,
        "timing": "traffic/capacity only; M-major traversal must be retimed independently",
    }
    return {
        "M": m, "N": n, "K": k,
        "m_tiles": nm, "n_tiles": nn, "k_tiles": nk,
        "issues": issues, "records": records,
        "useful_macs": m * n * k,
        "issued_macs": issues * pm * pn * pk,
        "array_operand_and_partial_traffic": array,
        "os_kmajor_shared_pool_to_local": main_pool,
        "loop_alternatives": {"WS": ws_alternatives, "IS": is_pool},
        "capacity": {
            "all_partial_records_full_bytes": records * record_full,
            "all_partial_values_masked_bytes": m * n * FP32_BYTES,
            "x_k_panel_full_bytes": padded_m * pk * BF16_BYTES,
            "x_k_panel_double_buffer_bytes": 2 * padded_m * pk * BF16_BYTES,
            "one_weight_tile_full_bytes": pn * pk * BF16_BYTES,
            "weight_tile_double_buffer_bytes": 2 * pn * pk * BF16_BYTES,
        },
    }


def _sum_fields(items: Sequence[Mapping[str, int]]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for item in items:
        for key, value in item.items():
            out[key] += value
    return dict(out)


def _rate(bytes_count: int, cycles: int | float, frequency_hz: int) -> float:
    """Decimal GB/s.  At 1 GHz, 1 byte/cycle equals 1 GB/s."""
    return bytes_count * frequency_hz / cycles / 1_000_000_000


def latency_hiding_requirements(rate_GB_s: float, latency_ns: float,
                                request_payload_bytes: int = 32) -> dict[str, Any]:
    """Bandwidth-delay-product lower bounds, not a stall-free certification.

    Decimal GB/s times ns has units bytes.  Payload credit counts exclude
    command metadata, burst alignment, queueing and retries.  The landing
    allocation must additionally retain operands still being consumed.
    """
    if not isinstance(rate_GB_s, (int, float)) or rate_GB_s < 0:
        raise ValueError("rate_GB_s must be nonnegative")
    if not isinstance(latency_ns, (int, float)) or latency_ns < 0:
        raise ValueError("latency_ns must be nonnegative")
    request_payload_bytes = _positive_int(request_payload_bytes, "request payload bytes")
    inflight = ceil(rate_GB_s * latency_ns)
    return {"target_GB_s": rate_GB_s, "latency_ns": latency_ns,
            "request_payload_bytes": request_payload_bytes,
            "inflight_payload_bytes_lower_bound": inflight,
            "payload_credit_count_lower_bound": ceil(inflight / request_payload_bytes),
            "not_a_stall_free_guarantee": True}


def _latencies(values: Any, cores: Sequence[Any]) -> list[int]:
    if isinstance(values, Mapping):
        result = [values.get(i, values.get(str(i))) for i in range(len(cores))]
    elif isinstance(values, (int, float)):
        result = [values] * len(cores)
    else:
        result = list(values)
    if len(result) != len(cores):
        raise ValueError("one completion latency is required for every core")
    return [_positive_int(x, "completion latency") for x in result]


def _full_model_weight_capacity(window: Mapping[str, Any], experts: list[dict[str, Any]]) -> dict[str, Any]:
    """Whole model layer, using explicit metadata or homogeneous expert sizes.

    The default 64 is the named DeepSeek-V2-Lite experiment's expert count,
    not an inference about arbitrary models.  Other models must pass their
    routed_expert_count explicitly.
    """
    routed = [e for e in experts if not e["is_shared"]]
    shared = [e for e in experts if e["is_shared"]]
    count = _positive_int(window.get("routed_expert_count", window.get("expert_count", window.get("num_experts", 64))),
                          "routed expert count")
    if len(routed) > count:
        raise ValueError("more routed expert descriptors than routed_expert_count")
    if len(routed) == count:
        routed_bytes = sum(3 * e["H"] * e["F"] * BF16_BYTES for e in routed)
        provenance = "all routed expert dimensions explicitly supplied"
    else:
        dims = {(e["H"], e["F"]) for e in routed}
        if "routed_H" in window and "routed_F" in window:
            h = _positive_int(window["routed_H"], "routed_H")
            f = _positive_int(window["routed_F"], "routed_F")
            provenance = "routed_H/F metadata multiplied by routed_expert_count"
        elif len(dims) == 1:
            h, f = next(iter(dims))
            provenance = "homogeneous routed dimensions; model expert count explicitly/defaulted to 64"
        else:
            return {"all_routed_and_shared_weight_bytes": None,
                    "routed_expert_count": count,
                    "provenance": "unknown: full model heterogeneous dimensions were not supplied"}
        routed_bytes = count * 3 * h * f * BF16_BYTES
    if shared:
        shared_bytes = sum(3 * e["H"] * e["F"] * BF16_BYTES for e in shared)
    elif "shared_H" in window and "shared_F" in window:
        shared_bytes = (3 * _positive_int(window["shared_H"], "shared_H")
                        * _positive_int(window["shared_F"], "shared_F") * BF16_BYTES)
    else:
        shared_bytes = 0
    return {
        "all_routed_and_shared_weight_bytes": routed_bytes + shared_bytes,
        "routed_weight_bytes": routed_bytes,
        "shared_weight_bytes": shared_bytes,
        "routed_expert_count": count,
        "provenance": provenance,
    }


def reverse_requirements(window: Mapping[str, Any], cores: Sequence[Any],
                         assignments: Sequence[Mapping[str, Any]], cycles: int,
                         completion_latencies: Any, ii: int = 1) -> dict[str, Any]:
    """Return JSON-safe traffic, average rates, capacity and burst requirements.

    ``window['experts']`` contains ``{id, Me, H, F, is_shared}``.  Inactive
    Me=0 descriptors are allowed and do not add cold-load traffic.  The
    batch is ``window['batch']`` (or ``B``); all Shared experts have Me=batch.
    Cores accept dictionaries, attribute objects, or PM/PN/PK sequences.
    Assignments contain expert_id, owner (integer core index), and optionally
    start_cycle/finish_cycle.  Every active expert must appear exactly once.
    ``cycles`` is the caller's layer compute makespan, not recomputed here.
    ``completion_latencies`` are ordered per core; the timing includes dot
    and FP32 partial-sum commit.  Projection issue counts are recomputed.

    Gate/Up use a joint K-major record set, so shared-pool X transfers occur
    once across both projections.  The per-projection Up pool-X count is
    therefore zero, explicitly attributed to the Gate record's panel load.
    Array X deliveries, MACs and per-projection partials remain unchanged.

    No pipeline, loop-retiming, double-buffer or latency-hiding feasibility
    is inferred from a bandwidth quotient alone.
    """
    cycles = _positive_int(cycles, "layer compute cycles")
    ii = _positive_int(ii, "initiation interval")
    frequency_hz = _positive_int(window.get("frequency_hz", DEFAULT_FREQUENCY_HZ), "frequency_hz")
    if not cores:
        raise ValueError("at least one core is required")
    shapes = [_core_shape(core) for core in cores]
    latencies = _latencies(completion_latencies, cores)
    raw_experts = list(window["experts"])
    experts: list[dict[str, Any]] = []
    for raw in raw_experts:
        e = dict(raw)
        e["Me"] = _nonnegative_int(e["Me"], "expert Me")
        e["H"] = _positive_int(e["H"], "expert H")
        e["F"] = _positive_int(e["F"], "expert F")
        e["is_shared"] = bool(e.get("is_shared", False))
        experts.append(e)
    by_id = {e["id"]: e for e in experts}
    if len(by_id) != len(experts):
        raise ValueError("expert ids must be unique")
    active = [e for e in experts if e["Me"]]
    if not active:
        raise ValueError("at least one active expert is required")
    batch_value = window.get("batch", window.get("B"))
    if batch_value is None:
        shared_counts = [e["Me"] for e in active if e["is_shared"]]
        batch_value = shared_counts[0] if shared_counts else None
    if batch_value is None:
        raise ValueError("batch/B is required if there is no active Shared expert")
    batch = _positive_int(batch_value, "batch")
    if any(e["is_shared"] and e["Me"] != batch for e in active):
        raise ValueError("Shared expert Me must equal batch")
    if any(e["Me"] > batch for e in active):
        raise ValueError("expert Me cannot exceed batch in this single-layer token trace")
    hidden_dims = {e["H"] for e in active}
    if len(hidden_dims) != 1:
        raise ValueError("all active MoE experts must share the layer's input/output H")
    h = next(iter(hidden_dims))
    assignment_by_id: dict[Any, Mapping[str, Any]] = {}
    for assignment in assignments:
        eid = assignment["expert_id"]
        if eid in assignment_by_id:
            raise ValueError("active experts cannot be assigned to multiple cores")
        if eid not in by_id or not by_id[eid]["Me"]:
            raise ValueError("assignments may only refer to active expert ids")
        owner = _nonnegative_int(assignment["owner"], "owner")
        if owner >= len(cores):
            raise ValueError("assignment owner exceeds core count")
        start = assignment.get("start_cycle", 0)
        finish = assignment.get("finish_cycle", cycles)
        if not (0 <= start <= finish <= cycles):
            raise ValueError("assignment interval must lie inside the layer makespan")
        assignment_by_id[eid] = assignment
    if set(assignment_by_id) != {e["id"] for e in active}:
        raise ValueError("every active expert must be assigned exactly once")

    expert_results = []
    per_core_projection: list[list[dict[str, Any]]] = [[] for _ in cores]
    core_experts: list[list[dict[str, Any]]] = [[] for _ in cores]
    for e in active:
        owner = assignment_by_id[e["id"]]["owner"]
        pm, pn, pk = shapes[owner]
        projections = []
        for phase, n, k in (("Gate", e["F"], e["H"]),
                            ("Up", e["F"], e["H"]),
                            ("Down", e["H"], e["F"])):
            projection = _shape_traffic(e["Me"], n, k, (pm, pn, pk))
            projection["phase"] = phase
            projection["x_panel_refill_record_count"] = projection["records"]
            projections.append(projection)
            per_core_projection[owner].append(projection)
        # The compute model issues one combined Gate/Up K-major output
        # record set.  Both projections consume the same Me x PK X panel.
        # Attribute its single pool refill to Gate, rather than counting
        # another shared-pool transfer for Up.  This does not suppress the
        # Up issues' array/latch delivery counts or their partial writes.
        for projection in projections[:2]:
            projection["x_panel_refill_record_count"] = 2 * projection["records"]
            projection["joint_gate_up_X_panel"] = True
        projections[0]["pool_X_provenance"] = "one shared X panel load services both Gate and Up records for each K segment"
        projections[1]["pool_X_provenance"] = "zero new shared-pool load: Up reuses Gate's joint K-major X panel"
        projections[1]["os_kmajor_shared_pool_to_local"]["x_masked_bytes"] = 0
        projections[1]["os_kmajor_shared_pool_to_local"]["x_full_slice_bytes"] = 0
        core_experts[owner].append(e)
        paired_gate_up = 2 * projections[0]["capacity"]["all_partial_records_full_bytes"]
        max_down = projections[2]["capacity"]["all_partial_records_full_bytes"]
        expert_results.append({
            "expert_id": e["id"], "owner": owner, "Me": e["Me"],
            "H": e["H"], "F": e["F"], "is_shared": e["is_shared"],
            "start_cycle": assignment_by_id[e["id"]].get("start_cycle"),
            "finish_cycle": assignment_by_id[e["id"]].get("finish_cycle"),
            "unique_weight_bytes": 3 * e["H"] * e["F"] * BF16_BYTES,
            "logical_input_gather_bytes": e["Me"] * e["H"] * BF16_BYTES,
            "z_full_expert_bf16_bytes": e["Me"] * e["F"] * BF16_BYTES,
            "gate_up_paired_partial_full_bytes": paired_gate_up,
            "down_partial_full_bytes": max_down,
            "peak_partial_bytes_phases_do_not_overlap": max(paired_gate_up, max_down),
            "projections": projections,
        })

    per_core = []
    peak_weight_bpc = peak_x_bpc = peak_acc_read_bpc = peak_acc_write_bpc = 0.0
    peak_pool_weight_bpc = peak_pool_x_bpc = 0.0
    for index, (pm, pn, pk) in enumerate(shapes):
        ps = per_core_projection[index]
        array = _sum_fields([p["array_operand_and_partial_traffic"] for p in ps])
        pool = _sum_fields([p["os_kmajor_shared_pool_to_local"] for p in ps])
        assigned_e = [e for e in expert_results if e["owner"] == index]
        partial_peak = max((e["peak_partial_bytes_phases_do_not_overlap"] for e in assigned_e), default=0)
        x_panel_peak = max((p["capacity"]["x_k_panel_full_bytes"] for p in ps), default=0)
        busiest_start = min((assignment_by_id[e["id"]].get("start_cycle", 0) for e in core_experts[index]), default=0)
        busiest_finish = max((assignment_by_id[e["id"]].get("finish_cycle", cycles) for e in core_experts[index]), default=0)
        active_span = max(0, busiest_finish - busiest_start)
        weight_slice = pn * pk * BF16_BYTES
        x_slice = pm * pk * BF16_BYTES
        partial_slice = pm * pn * FP32_BYTES
        pool_weight_rate = max((weight_slice / (p["m_tiles"] * ii) for p in ps), default=0.0)
        # An entire X K-panel is prefetched during the previous K period.
        # This is a steady-state panel-replacement requirement: the first
        # panel still requires an explicitly earlier preload.
        pool_x_rate = max((p["capacity"]["x_k_panel_full_bytes"]
                           / max(p["x_panel_refill_record_count"] * ii, latencies[index])
                           for p in ps
                           if p["os_kmajor_shared_pool_to_local"]["x_full_slice_bytes"]), default=0.0)
        peak = {
            "weight_full_slice_bytes_per_issue": weight_slice,
            "x_full_slice_bytes_per_issue": x_slice,
            "partial_read_bytes_per_followup_issue": partial_slice,
            "partial_write_bytes_per_issue": partial_slice,
            "weight_full_operand_GB_s_at_II": _rate(weight_slice, ii, frequency_hz),
            "x_full_operand_GB_s_at_II": _rate(x_slice, ii, frequency_hz),
            "partial_read_GB_s_at_II": _rate(partial_slice, ii, frequency_hz),
            "partial_write_GB_s_at_II": _rate(partial_slice, ii, frequency_hz),
            "weight_pool_refill_GB_s_with_M_tile_latch_reuse": pool_weight_rate * frequency_hz / 1_000_000_000,
            "X_panel_steady_state_refill_GB_s": pool_x_rate * frequency_hz / 1_000_000_000,
            "refill_scope": "nominal issue spacing and K-period reuse; first K-panel requires earlier preload, not covered by this steady-state quotient",
        }
        if ps:
            peak_weight_bpc += weight_slice / ii
            peak_x_bpc += x_slice / ii
            peak_acc_read_bpc += partial_slice / ii
            peak_acc_write_bpc += partial_slice / ii
            peak_pool_weight_bpc += pool_weight_rate
            peak_pool_x_bpc += pool_x_rate
        arrays_avg = {key.replace("_bytes", "_average_GB_s"): _rate(value, cycles, frequency_hz)
                      for key, value in array.items()}
        pool_avg = {key.replace("_bytes", "_average_GB_s"): _rate(value, cycles, frequency_hz)
                    for key, value in pool.items()}
        per_core.append({
            "owner": index, "shape": {"pm": pm, "pn": pn, "pk": pk},
            "multipliers": pm * pn * pk,
            "assigned_expert_count": len(assigned_e),
            "issues": sum(p["issues"] for p in ps),
            "useful_macs": sum(p["useful_macs"] for p in ps),
            "issued_macs": sum(p["issued_macs"] for p in ps),
            "assignment_span_cycles": active_span,
            "completion_latency_cycles": latencies[index],
            "array_operand_and_partial_traffic": array,
            "array_average_rates_layer_denominator": arrays_avg,
            "os_kmajor_shared_pool_to_local": pool,
            "shared_pool_average_rates_layer_denominator": pool_avg,
            "worst_case_per_issue_ports": peak,
            "capacity": {
                "peak_partial_records_full_bytes": partial_peak,
                "x_k_panel_double_buffer_bytes": 2 * x_panel_peak,
                "weight_tile_double_buffer_bytes": 2 * weight_slice if ps else 0,
                "x_tile_double_buffer_bytes_if_no_panel_reuse": 2 * x_slice if ps else 0,
                "pipeline_output_values_inflight_full_bytes": ceil(latencies[index] / ii) * partial_slice if ps else 0,
                "excludes_pipeline_internal_register_bits": True,
            },
        })
    total_array = _sum_fields([c["array_operand_and_partial_traffic"] for c in per_core])
    total_pool = _sum_fields([c["os_kmajor_shared_pool_to_local"] for c in per_core])
    unique_w = sum(3 * e["H"] * e["F"] * BF16_BYTES for e in active)
    global_x = batch * h * BF16_BYTES
    global_y = batch * h * FP32_BYTES
    all_z = sum(e["Me"] * e["F"] * BF16_BYTES for e in active)
    per_core_z_peak = [max((e["Me"] * e["F"] * BF16_BYTES for e in es), default=0)
                       for es in core_experts]
    model_capacity = _full_model_weight_capacity(window, experts)
    core_buffer_peak = sum(c["capacity"]["peak_partial_records_full_bytes"]
                           + c["capacity"]["x_k_panel_double_buffer_bytes"]
                           + c["capacity"]["weight_tile_double_buffer_bytes"]
                           for c in per_core)
    resident_common = global_x + global_y + sum(per_core_z_peak) + core_buffer_peak
    return {
        "schema_version": 1,
        "window_id": window.get("window_id", window.get("id")),
        "batch": batch,
        "precision": "BF16 X/W/Z, FP32 dot products/partials/combined Y; no quantization",
        "frequency_hz": frequency_hz,
        "compute_cycles": cycles,
        "compute_latency_ms": cycles / frequency_hz * 1000,
        "ii_cycles": ii,
        "external": {
            "selected_unique_bf16_weight_bytes": unique_w,
            "cold_weight_average_required_GB_s": _rate(unique_w, cycles, frequency_hz),
            "fully_resident_weight_HBM_bytes_per_replay": 0,
            "compulsory_global_input_bf16_bytes": global_x,
            "combined_layer_output_fp32_bytes": global_y,
            "cold_weights_plus_input_and_output_bytes": unique_w + global_x + global_y,
            "cold_weights_plus_input_and_output_average_GB_s": _rate(unique_w + global_x + global_y, cycles, frequency_hz),
            "assumption": "each selected expert weight read once; input/output counts only if crossing this external boundary",
        },
        "onchip": {
            "os_kmajor_array_operand_and_partial_traffic": total_array,
            "os_kmajor_array_average_rates": {
                key.replace("_bytes", "_average_GB_s"): _rate(value, cycles, frequency_hz)
                for key, value in total_array.items()},
            "os_kmajor_shared_pool_to_local": total_pool,
            "os_kmajor_shared_pool_average_rates": {
                key.replace("_bytes", "_average_GB_s"): _rate(value, cycles, frequency_hz)
                for key, value in total_pool.items()},
            "joint_gate_up_X_panel_provenance": "pool X reads: Me*H*2 once for joint Gate/Up, Me*F*2 once for Down; array deliveries still count all three GEMMs",
            "logical_expert_input_gather_bytes": sum(e["Me"] * e["H"] * BF16_BYTES for e in active),
            "gate_up_vector_read_fp32_bytes": 2 * sum(e["Me"] * e["F"] * FP32_BYTES for e in active),
            "z_vector_write_bf16_bytes": all_z,
            "down_Z_array_reads_already_in_X_operand_traffic": True,
            "combined_Y_read_modify_write_bytes": 2 * sum(e["Me"] * e["H"] * FP32_BYTES for e in active),
        },
        "capacity": {
            "selected_layer_weights_bytes": unique_w,
            **model_capacity,
            "global_input_X_bf16_bytes": global_x,
            "combined_output_Y_fp32_bytes": global_y,
            "all_selected_experts_Z_bf16_bytes_if_retained": all_z,
            "sum_one_expert_Z_per_core_bytes": sum(per_core_z_peak),
            "all_gate_and_up_logical_fp32_values_if_retained": 2 * sum(e["Me"] * e["F"] * FP32_BYTES for e in active),
            "low_rank_U_bytes": 0,
            "low_rank_U_scope": "not applicable to this fixed-BF16 unquantized experiment",
            "sum_declared_per_core_buffer_peaks_bytes": core_buffer_peak,
            "fully_resident_selected_weight_os_working_capacity_bytes": unique_w + resident_common,
            "fully_resident_all_model_weight_os_working_capacity_bytes": (
                model_capacity["all_routed_and_shared_weight_bytes"] + resident_common
                if model_capacity["all_routed_and_shared_weight_bytes"] is not None else None),
            "capacity_scope": "sufficient byte payload allocation for one expert active per core under declared OS loops; conservative sum of core peaks, excludes metadata/alignment/banking/pipeline internals",
            "capacity_does_not_imply_bandwidth": True,
        },
        "peak_refill": {
            "simultaneous_array_weight_full_GB_s_upper_envelope": peak_weight_bpc * frequency_hz / 1_000_000_000,
            "simultaneous_array_X_full_GB_s_upper_envelope": peak_x_bpc * frequency_hz / 1_000_000_000,
            "simultaneous_partial_read_full_GB_s_upper_envelope": peak_acc_read_bpc * frequency_hz / 1_000_000_000,
            "simultaneous_partial_write_full_GB_s_upper_envelope": peak_acc_write_bpc * frequency_hz / 1_000_000_000,
            "simultaneous_pool_weight_GB_s_envelope_with_M_reuse": peak_pool_weight_bpc * frequency_hz / 1_000_000_000,
            "simultaneous_pool_X_panel_steady_state_GB_s_envelope": peak_pool_x_bpc * frequency_hz / 1_000_000_000,
            "envelope_scope": "sum of active cores' maximum per-issue demands; not measured concurrency or the minimum shared-pool refill rate",
            "latency_hiding": {
                "inflight_bytes_formula": "ceil(target_byte_rate_bytes_per_cycle * memory_latency_cycles)",
                "credits_formula": "ceil(inflight_bytes / request_payload_bytes)",
                "buffer_formula": "at least reserved in-flight landing bytes plus live operand bytes; add allocator/alignment margins",
                "stall_free_not_guaranteed_by_double_buffering": True,
            },
        },
        "per_core": per_core,
        "per_expert": expert_results,
        "formulas": {
            "GEMMs": "Gate/Up=[Me,H]@[H,F]; Down=[Me,F]@[F,H]",
            "tile_counts": "nm=ceil(Me/PM); nn=ceil(N/PN); nk=ceil(K/PK); issues=nm*nn*nk",
            "external_unique_weights": "sum_active_experts 2*(H*F + H*F + F*H)",
            "array_weight_full": "issues*PN*PK*2",
            "array_X_full": "issues*PM*PK*2",
            "partial_read_full": "nm*nn*(nk-1)*PM*PN*4; first K starts from zero",
            "partial_write_full": "issues*PM*PN*4; includes final projection commit",
            "os_pool_weight_full": "nn*nk*PN*PK*2; one W tile reused across all M tiles",
            "os_pool_X_full": "joint Gate/Up: nm*ceil(H/PK)*PM*PK*2 once; Down: nm*ceil(F/PK)*PM*PK*2 once; whole Me K-panel retained across all N output records",
            "os_pool_X_masked": "each expert: Me*(H+F)*2; Gate/Up share one X panel, Down consumes Z",
            "average_GB_s": "traffic_bytes * frequency_hz / compute_cycles / 1e9",
            "WS_loop": "N-group -> K -> M -> N-in-group; X reread for every N-group",
            "IS_loop": "K -> M -> N; W reread for every M-tile",
            "OS_loop": "Gate/Up jointly: K -> concatenated Gate/Up N records -> M; Down: K -> N -> M; all output records resident, whole X K-panel resident",
        },
        "limitations": [
            "Supply requirements only: no HBM/NoC/SRAM bank/port or DMA scheduling simulation.",
            "All average rates divide by the caller's compute-only layer makespan; averages cannot guarantee tile deadlines.",
            "Per-issue operand delivery is distinct from shared-pool refill: stationary latches may reuse delivered values.",
            "Masked tail traffic assumes address/transfer suppression; installed slices/records still use full physical capacity.",
            "WS/IS loop alternatives have separate residency/ordering; their runtime must be recomputed before comparing performance.",
            "HBM becomes zero per replay only when weights persist in the resident pool; initial load/prewarm is outside replay timing.",
            "GB-scale SRAM capacity is not evidence for sufficient SRAM/3D interconnect bandwidth, area, power or clock closure.",
        ],
    }
