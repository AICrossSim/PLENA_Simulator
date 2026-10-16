"""Full Shared + routed MoE supply and conservative resident-capacity ledger.

Every active expert retains its actual M_e and F.  The GEMM supply primitives
are the topology-consistent Shared experiment primitives: physical PEs remain
OS; WS/IS mean proposed operand-loop retention with complete G PE contexts.
No inactive expert weights are fetched.  External, pool-to-local, and array
boundaries are kept separate and never added as though all were HBM traffic.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from research.moe_dispatch.plena65536_shared.dataflow import (
    BF16_BYTES, FLOWS, FP32_BYTES, Region, add_rates, gemm_traffic,
    plena_gemm_traffic, plena_shape_tuple, plena_shared_traffic,
)


def _sum_dict(items: Sequence[Mapping[str, int]]) -> dict[str, int]:
    if not items:
        return {}
    return {key: sum(item[key] for item in items) for key in items[0]}


def resource_ledger(cores: Sequence[Any]) -> dict[str, Any]:
    """Charge fixed hardware once, independently of expert count or dataflow."""
    shapes = [plena_shape_tuple(core) for core in cores]
    result = {key: 0 for key in (
        "multipliers", "PE_FP32_bytes", "shadow_FP32_bytes", "M_SUM_adders",
        "M_SUM_tree_FP32_register_bytes", "full_operand_double_buffer_bytes",
        "native_square_groups")}
    for b, reduction, rm, rn in shapes:
        pm, pn, groups = b * rm, b * rn, reduction // b
        macs = pm * pn * groups
        adders = pm * pn * (groups - 1)
        result["multipliers"] += macs
        result["PE_FP32_bytes"] += FP32_BYTES * macs
        result["shadow_FP32_bytes"] += FP32_BYTES * macs
        result["M_SUM_adders"] += adders
        result["M_SUM_tree_FP32_register_bytes"] += FP32_BYTES * adders
        result["full_operand_double_buffer_bytes"] += 2 * BF16_BYTES * reduction * (pm + pn)
        result["native_square_groups"] += rm * rn * groups
    result["static_storage_bytes"] = sum(result[key] for key in (
        "PE_FP32_bytes", "shadow_FP32_bytes", "M_SUM_tree_FP32_register_bytes",
        "full_operand_double_buffer_bytes"))
    result["area_calibrated"] = False
    return result


def _endpoint(job: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in job:
            return float(job[name])
    return None


def _schedule_scratch_peak(entries: Sequence[Mapping[str, Any]], core_count: int) -> tuple[int, bool]:
    """Half-open execution intervals; per-core phase envelope is conservative."""
    if any(item["start"] is None or item["finish"] is None for item in entries):
        return sum(max((entry["scratch_bytes"] for entry in entries
                        if entry["core_id"] == core), default=0)
                   for core in range(core_count)), False
    events = []
    for item in entries:
        if item["finish"] < item["start"]:
            raise ValueError("job ends before it starts")
        if item["finish"] == item["start"]:
            continue
        events.append((item["start"], 1, item["core_id"], item["scratch_bytes"]))
        events.append((item["finish"], 0, item["core_id"], -item["scratch_bytes"]))
    live = [0] * core_count
    occupancy = [0] * core_count
    peak = 0
    for _, kind, core, change in sorted(events):
        occupancy[core] += 1 if kind else -1
        if occupancy[core] < 0 or occupancy[core] > 1:
            raise ValueError("expert jobs overlap on the same core")
        live[core] += change
        if live[core] < 0:
            raise ValueError("invalid scheduled capacity interval")
        # A core executes one expert job at a time.  Shared split subjobs may
        # legitimately execute concurrently only on different cores.
        peak = max(peak, sum(live))
    return peak, True


def _private_weight_bytes(projections: Sequence[Mapping[str, Any]]) -> int:
    """Union N ownership per projection, deduplicating row splits on a core."""
    result = 0
    for name in ("gate", "up", "down"):
        own = [p for p in projections if p["projection"] == name]
        intervals = sorted((p["region"]["n_start"], p["region"]["n_stop"]) for p in own)
        if not intervals:
            continue
        start, stop = intervals[0]
        width = 0
        for next_start, next_stop in intervals[1:]:
            if next_start > stop:
                width += stop - start
                start, stop = next_start, next_stop
            else:
                stop = max(stop, next_stop)
        width += stop - start
        result += width * own[0]["K"] * BF16_BYTES
    return result


def plena_mixed_traffic(window: Mapping[str, Any], cores: Sequence[Any],
                        flows: Sequence[str] | str, plan: Mapping[str, Any],
                        *, double_buffer: bool = True) -> dict[str, Any]:
    """Traffic for one actual routed window under an explicit expert schedule.

    Plan jobs each have ``expert_id`` and ``core_id`` for whole-expert jobs.
    A job can instead have ``partitions`` (Gate/Up/Down exact-cover regions) for
    Shared split across cores.  Optional ``start_cycles``/``finish_cycles`` or
    ``start``/``end`` declare execution intervals.  Exactly one job represents
    every active expert, including Shared.  All-active weight residency and
    all-expert intermediate residency are deliberately conservative envelopes.
    """
    if not cores:
        raise ValueError("at least one core is required")
    if isinstance(flows, str):
        flows = [flows] * len(cores)
    flows = [flow.upper() for flow in flows]
    if len(flows) != len(cores) or any(flow not in FLOWS for flow in flows):
        raise ValueError("one OS/WS/IS retention loop is required per core")
    batch, hidden = int(window["batch"]), int(window.get("hidden", 2048))
    if batch <= 0 or hidden <= 0:
        raise ValueError("window dimensions must be positive")
    experts = [expert for expert in window["experts"] if int(expert["Me"]) > 0]
    identities = [int(expert["id"]) for expert in experts]
    if len(set(identities)) != len(identities):
        raise ValueError("expert identities must be unique within a routed window")
    jobs = list(plan["jobs"])
    job_ids = [int(job["expert_id"]) for job in jobs]
    if sorted(job_ids) != sorted(identities):
        raise ValueError("schedule must execute every active expert exactly once")
    by_id = {int(expert["id"]): expert for expert in experts}
    leaves, scratch_entries = [], []
    private_weights = [0] * len(cores)
    private_activation = [0] * len(cores)
    for job in jobs:
        expert_id = int(job["expert_id"])
        expert = by_id[expert_id]
        m, h, f = int(expert["Me"]), int(expert["H"]), int(expert["F"])
        if h != hidden:
            raise ValueError("expert H does not match window H")
        partitions = job.get("partitions")
        if partitions is None:
            core_id = int(job["core_id"])
            partitions = {name: [Region(core_id, 0, m, 0, n)]
                          for name, n in (("gate", f), ("up", f), ("down", h))}
        leaf = plena_shared_traffic(m, cores, flows, partitions,
                                    hidden=h, intermediate=f, double_buffer=double_buffer)
        leaf.update(expert_id=expert_id, is_shared=bool(expert.get("is_shared", expert_id == -1)),
                    token_count=m)
        leaves.append(leaf)
        for core_id in range(len(cores)):
            projections = [p for p in leaf["projections"] if p["core_id"] == core_id]
            if not projections:
                continue
            gu = sum(p["capacity"]["partial_scratch_full_bytes"] for p in projections
                     if p["projection"] in ("gate", "up"))
            down = sum(p["capacity"]["partial_scratch_full_bytes"] for p in projections
                       if p["projection"] == "down")
            scratch_entries.append({
                "expert_id": expert_id, "core_id": core_id,
                "scratch_bytes": max(gu, down),
                "start": _endpoint(job, "start_cycles", "start", "start_cycle"),
                "finish": _endpoint(job, "finish_cycles", "end_cycles", "end", "finish", "end_cycle"),
            })
            private_weights[core_id] += _private_weight_bytes(projections)
            # Whole expert activation payload on every participating private
            # core is an explicit upper bound, including Shared column splits.
            private_activation[core_id] += leaf["capacity"]["activation_liveness_upper_bytes"]
    pool = _sum_dict([leaf["pool_to_local"] for leaf in leaves])
    array = _sum_dict([leaf["array_delivery"] for leaf in leaves])
    w_unique = sum(leaf["external"]["cold_weight_only_unique_bytes"] for leaf in leaves)
    global_x, global_y = batch * hidden * BF16_BYTES, batch * hidden * FP32_BYTES
    gu = sum(leaf["capacity"]["completed_GU_fp32_bytes"] for leaf in leaves)
    z = sum(leaf["capacity"]["intermediate_Z_bf16_bytes"] for leaf in leaves)
    expert_y = sum(leaf["capacity"]["output_Y_fp32_bytes"] for leaf in leaves)
    # Routed token_indices address the original X; pointer/gather transport is
    # not represented as an uncharged separately materialized routed X copy.
    activation_upper = global_x + gu + z + expert_y + global_y
    scratch_peak, uses_intervals = _schedule_scratch_peak(scratch_entries, len(cores))
    sum_core_scratch_upper = sum(max((e["scratch_bytes"] for e in scratch_entries
                                    if e["core_id"] == core), default=0)
                                for core in range(len(cores)))
    ledger = resource_ledger(cores)
    static = ledger["static_storage_bytes"]
    stream_weight = pool["weight_masked_bytes"]
    stream_x = pool["x_masked_bytes"]
    stream_partial = pool["partial_read_masked_bytes"] + pool["partial_write_masked_bytes"]
    leaf_intermediates = sum(leaf["external"]["no_pool_stream_intermediate_and_final_bytes"]
                            for leaf in leaves)
    combine = expert_y + global_y
    postprocess = {"GU_read_fp32_bytes": gu, "Z_write_bf16_bytes": z,
                   "expert_Y_combine_read_fp32_bytes": expert_y,
                   "combined_Y_write_fp32_bytes": global_y,
                   "route_metadata_transport_bytes": 0,
                   "route_gather_scatter_timing_included": False,
                   "elementwise_and_combine_timing_included": False}
    per_core = []
    for core_id in range(len(cores)):
        projections = [p for leaf in leaves for p in leaf["projections"] if p["core_id"] == core_id]
        per_core.append({"core_id": core_id, "dataflow": flows[core_id],
                         "pool_to_local": _sum_dict([p["pool_to_local"] for p in projections]),
                         "array_delivery": _sum_dict([p["array_delivery"] for p in projections]),
                         "private_distinct_weight_bytes": private_weights[core_id],
                         "private_activation_upper_bytes": private_activation[core_id],
                         "scratch_max_bytes": max((entry["scratch_bytes"] for entry in scratch_entries
                                                    if entry["core_id"] == core_id), default=0)})
    external = {
        "cold_weight_only_unique_bytes": w_unique,
        "cold_weights_inputs_final_output_bytes": w_unique + global_x + global_y,
        "fully_resident_weight_replay_bytes": 0,
        "no_pool_stream_weight_bytes": stream_weight,
        "no_pool_stream_activation_bytes": stream_x,
        "no_pool_stream_partial_bytes": stream_partial,
        "no_pool_stream_intermediate_and_final_bytes": leaf_intermediates + combine,
        "no_pool_stream_total_bytes": stream_weight + stream_x + stream_partial + leaf_intermediates + combine,
        "no_pool_stream_weight_refetch_factor": stream_weight / w_unique,
        "unified_pool_external_weight_refetch_factor": 1.0,
    }
    shared = [leaf for leaf in leaves if leaf["is_shared"]]
    routed = [leaf for leaf in leaves if not leaf["is_shared"]]
    return {
        "window_id": window.get("id", ""), "batch": batch, "hidden": hidden,
        "dataflows": flows, "active_routed_experts": len(routed),
        "shared_tokens": sum(leaf["token_count"] for leaf in shared),
        "routed_token_assignments": sum(leaf["token_count"] for leaf in routed),
        "useful_macs": sum(leaf["useful_macs"] for leaf in leaves),
        "shared_useful_macs": sum(leaf["useful_macs"] for leaf in shared),
        "routed_useful_macs": sum(leaf["useful_macs"] for leaf in routed),
        "issued_macs": sum(leaf["issued_macs"] for leaf in leaves),
        "issues": sum(leaf["issues"] for leaf in leaves),
        "shared_unique_weight_bytes": sum(leaf["external"]["cold_weight_only_unique_bytes"] for leaf in shared),
        "routed_unique_weight_bytes": sum(leaf["external"]["cold_weight_only_unique_bytes"] for leaf in routed),
        "external": external, "pool_to_local": pool, "array_delivery": array,
        "postprocess": postprocess, "per_core": per_core, "experts": leaves,
        "scratch_execution_intervals": scratch_entries, "resource_ledger": ledger,
        "capacity": {
            "active_unique_weights_bytes": w_unique, "original_X_bf16_bytes": global_x,
            "completed_all_GU_fp32_bytes": gu, "all_Z_bf16_bytes": z,
            "all_expert_Y_fp32_bytes": expert_y, "combined_Y_fp32_bytes": global_y,
            "activation_liveness_upper_bytes": activation_upper,
            "fixed_hardware_storage_bytes": static,
            "scheduled_scratch_upper_bytes": scratch_peak,
            "scheduled_scratch_uses_job_intervals": uses_intervals,
            "sum_core_max_scratch_upper_bytes": sum_core_scratch_upper,
            "local_scratch_tree_upper_bytes": static + scratch_peak,
            "unified_resident_payload_upper_bytes": w_unique + activation_upper + static + scratch_peak,
            "private_weights_sum_bytes": sum(private_weights),
            "private_activation_sum_upper_bytes": sum(private_activation),
            "private_resident_payload_sum_upper_bytes": sum(private_weights) + sum(private_activation)
                                                         + static + sum_core_scratch_upper,
            "all_active_weights_resident_assumed_for_capacity": True,
            "all_expert_intermediates_simultaneously_live_upper_bound": True,
            "scratch_phase_overlap_conservative_GU_pair": True,
            "excludes_metadata_banks_alignment": True,
        },
        "contract": {
            "loop_accounting_only": True, "native_WS_IS_support_claimed": False,
            "physical_PE_dataflow": "OS", "G_partial_context_spill_fully_charged": True,
            "external_weight_identity": "expert_id + matrix; physical address aliases do not merge experts",
            "unique_external_X_once_per_window": True,
            "routed_X_uses_global_token_index_addressing": True,
            "unified_pool_external_reads_do_not_equal_local_refills": True,
            "private_core_weight_duplicates_charged_to_capacity": True,
            "no_pool_is_distinct_external_streaming_system": True,
            "gate_up_joint_X_reuse_assumed": False,
            "route_metadata_and_nonGEMM_timing_included": False,
        },
    }
