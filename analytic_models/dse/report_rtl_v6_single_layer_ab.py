#!/usr/bin/env python3
"""Generate the rtl-v6 long-context, single-chip, single-layer A/B report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from analytic_models.area_new import estimate_area
from analytic_models.performance.compiler_cost_model import (
    compile_and_evaluate_compiler_cost,
)
from analytic_models.power.system_power import estimate_system_power
from Workspace.qwen3_32b_dense_analytic.run_optuna_dse import (
    DEFAULT_COMPILER_COST_CALIBRATION,
    DEFAULT_COMPILER_COST_SETTINGS,
    write_compiler_cost_toml,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "Workspace/reports/rtl"
REPORT_STEM = "rtl_v6_long_context_single_layer_ab_v2"
BANKING_REPORT_NAME = "rtl_v6_long_context_single_layer_banking_v2.csv"
AREA_BUDGET_MM2 = 0.9 * 826.0
ROW_OPCODES = frozenset(
    {
        "V_RED_MAX_ROWS",
        "V_RED_SUM_ROWS",
        "V_SUB_ROWS",
        "V_EXP_ROWS",
        "V_MUL_ROWS_F",
        "V_MUL_ROWS_STATS",
        "V_SFM_MAX_ROWS",
        "V_SFM_SUM_ROWS",
        "V_SFM_FINAL_ROWS",
    }
)
TRAFFIC_FIELDS = (
    "payload_read_bytes",
    "payload_write_bytes",
    "physical_read_bytes",
    "physical_write_bytes",
    "read_requests",
    "write_requests",
)


@dataclass(frozen=True)
class ModelCase:
    key: str
    label: str
    config: Path
    moe: bool
    softmax_state_heads: int


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    vector_schedule: str
    state_schedule: str
    pv_schedule: str
    row_lanes: int
    fidelity: str = "costemitter_actual_rtl_isa_supported"


MODELS = (
    ModelCase(
        "qwen3_32b",
        "Qwen3-32B dense layer",
        REPO_ROOT / "Workspace/qwen3_32b_dense_analytic/qwen3-32b.json",
        False,
        8,
    ),
    ModelCase(
        "qwen3_235b_a22b",
        "Qwen3-235B-A22B fixed-balanced MoE layer",
        REPO_ROOT
        / "Workspace/qwen3_235b_a22b_analytic/qwen3-235b-a22b-instruct.json",
        True,
        16,
    ),
)

ACTUAL_ARMS = (
    Arm("rtl_v5", "rtl-v5 baseline", "single-row-v1", "streamed-v2", "shift-add-v1", 1),
    Arm("state_only", "Packed state only", "single-row-v1", "row-bank-simd-v3", "shift-add-v1", 1),
    Arm("direct_pv_only", "Direct PV only", "single-row-v1", "streamed-v2", "direct-packed-rmw-v1", 1),
    Arm("combined_r1", "Combined R1", "multi-row-v1", "row-bank-simd-v3", "direct-packed-rmw-v1", 1),
    Arm("combined_r2", "Combined R2", "multi-row-v1", "row-bank-simd-v3", "direct-packed-rmw-v1", 2),
    Arm("combined_r4", "Combined R4", "multi-row-v1", "row-bank-simd-v3", "direct-packed-rmw-v1", 4),
    Arm("combined_r8", "Combined R8", "multi-row-v1", "row-bank-simd-v3", "direct-packed-rmw-v1", 8),
)


def _hardware(*, fp_sram_depth: int = 10) -> dict[str, int]:
    return {
        "MLEN": 2048,
        "VLEN": 2048,
        "BLEN": 128,
        "HLEN": 128,
        "BROADCAST_AMOUNT": 16,
        "MATRIX_SRAM_SIZE": 49_152,
        "VECTOR_SRAM_SIZE": 259,
        "FP_SRAM_DEPTH": fp_sram_depth,
        "FP_CONSTANT_NUM": 10,
        "COMPACT_STATS_LANES": 16,
        "HBM_M_Prefetch_Amount": 2048,
        "HBM_V_Prefetch_Amount": 128,
        "HBM_V_Writeback_Amount": 128,
        "INT_DATA_WIDTH": 32,
    }


def _precision() -> dict[str, Any]:
    return {
        "name": "w_mxint4__act_mxint4__kv_mxint4__fp_e6m5",
        "WEIGHT_WIDTH": "MXINT4",
        "ACT_WIDTH": "MXINT4",
        "KV_WIDTH": "MXINT4",
        "FP_SETTING": "FP_E6M5",
        "accuracy_score": 0.0,
    }


def _precision_profile() -> dict[str, Any]:
    return {
        "name": _precision()["name"],
        "WEIGHT_WIDTH": {
            "kind": "MXINT",
            "width": 4,
            "scale_width": 8,
        },
        "ACT_WIDTH": {
            "kind": "MXINT",
            "width": 4,
            "scale_width": 8,
        },
        "KV_WIDTH": {
            "kind": "MXINT",
            "width": 4,
            "scale_width": 8,
        },
        "FP_SETTING": {"exp": 6, "mant": 5},
    }


def _write_settings(path: Path, *, fp_sram_depth: int) -> None:
    write_compiler_cost_toml(
        DEFAULT_COMPILER_COST_SETTINGS,
        path,
        _hardware(fp_sram_depth=fp_sram_depth),
        _precision_profile(),
        SimpleNamespace(
            mx_scale_block_size=64,
            mx_scale_width=8,
            frequency_ghz=1.0,
            weight_precision="MXINT4",
            weight_mx_exp_width=2,
            weight_mx_mant_width=1,
        ),
        "compact",
    )


def _area_config(
    row_lanes: int,
    *,
    version: str = "rtl-v6",
    enable_multirow: bool = True,
    enable_state: bool = True,
    enable_packed_pv: bool = True,
    fp_sram_depth: int = 10,
    softmax_state_heads: int = 8,
) -> dict[str, Any]:
    hw = _hardware(fp_sram_depth=fp_sram_depth)
    return {
        **hw,
        **_precision(),
        "MATRIX_SRAM_DEPTH": hw["MATRIX_SRAM_SIZE"],
        "VECTOR_SRAM_DEPTH": hw["VECTOR_SRAM_SIZE"],
        "INT_SRAM_DEPTH": 32,
        "FP_SETTING": "FP_E6M5",
        "SOFTMAX_ROW_LANES": row_lanes,
        "VECTOR_SRAM_ROW_BANKS": (
            row_lanes if version == "rtl-v6" and enable_multirow else 1
        ),
        "SOFTMAX_STATE_BANK_ENTRIES": (
            softmax_state_heads * hw["MLEN"]
            if version == "rtl-v6" and enable_state
            else 0
        ),
        "ENABLE_SOFTMAX_MULTIROW": enable_multirow,
        "ENABLE_SOFTMAX_STATE_SIMD": enable_state,
        "ENABLE_PACKED_PV_ACCUMULATION": enable_packed_pv,
        "SOFTMAX_SCOREBOARD_DEPTH": 32,
        "BLOCK_DIM": hw["BLEN"],
        "HBM_ELE_WIDTH": hw["MLEN"],
        "HBM_SCALE_WIDTH": hw["MLEN"] // hw["BLEN"] * 8,
        "MX_SCALE_WIDTH": 8,
        "SRAM_PORT_MODEL": "ideal-dual-port",
        "vector_scalar_area_version": version,
        "address_generation_mode": "loop-agu-v1",
        "CLOCK_PERIOD_PS": 1000,
        "HBM_CAPACITY_BYTES": 80_000_000_000,
        "HBM_BANDWIDTH_GBPS": 2039.0,
        "INPUT_TOKENS": 90_000 * 8,
        "SEQ_LEN": 90_000,
        "BATCH_SIZE": 8,
    }


def _traffic_sum(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    total = {key: 0 for key in TRAFFIC_FIELDS}
    for row in rows:
        for key in TRAFFIC_FIELDS:
            total[key] += int(row.get(key, 0))
    return total


def _add_traffic_bucket(
    destination: dict[str, dict[str, int]],
    key: str,
    source: Mapping[str, Any],
) -> None:
    bucket = destination.setdefault(key, {field: 0 for field in TRAFFIC_FIELDS})
    for field in TRAFFIC_FIELDS:
        bucket[field] += int(source.get(field, 0))


def _layer_traffic(report: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    full = dict(report.get("hbm_traffic_breakdown") or {})
    by_stage: dict[str, dict[str, int]] = {}
    by_role: dict[str, dict[str, int]] = {}
    by_opcode: dict[str, dict[str, int]] = {}
    by_stage_role: dict[str, dict[str, int]] = {}
    by_opcode_role: dict[str, dict[str, int]] = {}
    by_stage_opcode_role: dict[str, dict[str, int]] = {}
    for key, raw in dict(full.get("by_stage_opcode_role") or {}).items():
        stage, opcode, role = str(key).split("::", 2)
        if not stage.startswith("layer/"):
            continue
        bucket = dict(raw)
        _add_traffic_bucket(by_stage, stage, bucket)
        _add_traffic_bucket(by_role, role, bucket)
        _add_traffic_bucket(by_opcode, opcode, bucket)
        _add_traffic_bucket(by_stage_role, f"{stage}::{role}", bucket)
        _add_traffic_bucket(by_opcode_role, f"{opcode}::{role}", bucket)
        _add_traffic_bucket(
            by_stage_opcode_role, f"{stage}::{opcode}::{role}", bucket
        )
    if not by_stage:
        by_stage = {
            str(stage): dict(values)
            for stage, values in dict(full.get("by_stage") or {}).items()
            if str(stage).startswith("layer/")
        }
    totals = _traffic_sum(by_stage.values())
    return {
        "by_stage": by_stage,
        "by_role": by_role,
        "by_opcode": by_opcode,
        "by_stage_role": by_stage_role,
        "by_opcode_role": by_opcode_role,
        "by_stage_opcode_role": by_stage_opcode_role,
    }, totals


def _layer_timing(report: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    stage_compute = {
        stage: float(value)
        for stage, value in dict(report["stage_compute_latency_ns"]).items()
        if stage.startswith("layer/")
    }
    stage_memory = {
        stage: float(value)
        for stage, value in dict(report["hbm_stage_latency_ns"]).items()
        if stage.startswith("layer/")
    }
    stage_roofline = {
        stage: max(value, stage_memory.get(stage, 0.0))
        for stage, value in stage_compute.items()
    }
    hbm_opcode_latency: Counter[str] = Counter()
    for key, value in dict(
        report.get("hbm_stage_opcode_latency_ns") or {}
    ).items():
        stage, opcode = str(key).rsplit("::", 1)
        if stage.startswith("layer/"):
            hbm_opcode_latency[opcode] += float(value)
    layer_latency_ns = math.fsum(stage_roofline.values())
    traffic, totals = _layer_traffic(report)
    timing = {
        "compute_timing_mode": "ideal-ii1",
        "roofline_latency_ns": layer_latency_ns,
        "compute_latency_ns": math.fsum(stage_compute.values()),
        "hbm_latency_ns": math.fsum(stage_memory.values()),
        "stage_compute_latency_ns": stage_compute,
        "hbm_stage_latency_ns": stage_memory,
        "hbm_opcode_latency_ns": dict(hbm_opcode_latency),
        "stage_roofline_latency_ns": stage_roofline,
        "hbm_read_bytes": totals["physical_read_bytes"],
        "hbm_write_bytes": totals["physical_write_bytes"],
        "hbm_payload_read_bytes": totals["payload_read_bytes"],
        "hbm_payload_write_bytes": totals["payload_write_bytes"],
        "hbm_read_requests": totals["read_requests"],
        "hbm_write_requests": totals["write_requests"],
        "hbm_traffic_breakdown": traffic,
    }
    return timing, layer_latency_ns


def _layer_trace(
    trace: Any,
    *,
    count_scale: float = 1.0,
    projected_row_lanes: int | None = None,
) -> dict[str, Any]:
    actions = []
    for action in trace.energy_actions:
        if not action.stage.startswith("layer/"):
            continue
        payload = action.to_dict()
        if count_scale != 1.0 and _is_row_action(payload):
            payload["count"] = max(1, int(round(payload["count"] * count_scale)))
            payload["activity_fidelity"] = "structural_row_group_count_extrapolation"
        if projected_row_lanes is not None and _is_row_action(payload):
            old_lanes = int(payload.get("total_lanes", 0))
            payload["active_lanes"] = projected_row_lanes
            payload["total_lanes"] = projected_row_lanes
            payload["segment_count"] = projected_row_lanes
            payload["segment_log2"] = int(math.log2(projected_row_lanes))
            variant = str(payload.get("variant", ""))
            if variant and old_lanes > 0:
                fields = variant.split(",")
                fields = [
                    str(projected_row_lanes) if field == str(old_lanes) else field
                    for field in fields
                ]
                if fields and fields[-1] == str(int(math.log2(old_lanes))):
                    fields[-1] = str(int(math.log2(projected_row_lanes)))
                payload["variant"] = ",".join(fields)
        actions.append(payload)
    return {
        "schema_version": getattr(trace, "schema_version", None),
        "energy_actions": actions,
        "metadata": dict(getattr(trace, "metadata", {})),
    }


def _layer_dma_digest(trace: Any) -> str:
    events = [
        event.to_dict()
        for event in trace.memory_events
        if event.stage.startswith("layer/")
    ]
    canonical = json.dumps(
        events, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _is_row_action(action: Mapping[str, Any]) -> bool:
    family = str(action.get("action", ""))
    return (
        family.endswith("_rows")
        or family.startswith("softmax_row_")
        or str(action.get("component")) == "softmax_state"
    )


def _opcode_rows(trace: Any) -> Counter[str]:
    result: Counter[str] = Counter()
    for stage, stage_cost in trace.stages.items():
        if stage.startswith("layer/"):
            result.update(stage_cost.dynamic_opcodes)
    return result


def _opcode_category(opcode: str) -> str:
    if opcode.startswith("M_"):
        return "matrix"
    if opcode.startswith("V_"):
        return "vector"
    if opcode.startswith("S_"):
        return "scalar"
    if opcode.startswith("C_"):
        return "control"
    return "other"


def _category_cycles(report: Mapping[str, Any]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for stage, opcodes in dict(report["stage_compute_opcode_work_cycles"]).items():
        if not stage.startswith("layer/"):
            continue
        for opcode, cycles in dict(opcodes).items():
            result[_opcode_category(opcode)] += int(cycles)
    return dict(result)


def _area_metrics(
    row_lanes: int,
    *,
    version: str,
    enable_multirow: bool = True,
    enable_state: bool = True,
    enable_packed_pv: bool = True,
    fp_sram_depth: int = 10,
    softmax_state_heads: int = 8,
) -> dict[str, Any]:
    metrics = estimate_area(
        _area_config(
            row_lanes,
            version=version,
            enable_multirow=enable_multirow,
            enable_state=enable_state,
            enable_packed_pv=enable_packed_pv,
            fp_sram_depth=fp_sram_depth,
            softmax_state_heads=softmax_state_heads,
        )
    )
    vector = dict(metrics.get("vector_machine") or {})
    sram = dict(metrics["sram"])
    banking = dict(sram["vector_sram_banking"])
    breakdown = dict(sram["area_sram_breakdown"])
    state_area = math.fsum(
        float(breakdown.get(key, 0.0))
        for key in ("SoftmaxStateBank", "SoftmaxStatisticBank", "SoftmaxFactorBank")
    )
    return {
        "core_area_mm2": float(metrics["area"]) / 1e6,
        "budget_fraction": float(metrics["area"]) / 1e6 / AREA_BUDGET_MM2,
        "vector_logic_breakdown_um2": dict(vector.get("breakdown") or {}),
        "rtl_v6_logic_delta_mm2": float(vector.get("rtl_v6_delta_area", 0.0)) / 1e6,
        "state_stat_factor_area_mm2": state_area / 1e6,
        "vector_sram_banking": banking,
        "area_calibration_status": vector.get("rtl_v6_delta_status") if version == "rtl-v6" else vector.get("rtl_v5_delta_status"),
        "row_lane_fidelity": vector.get("rtl_v6_row_lane_fidelity", "rtl_v5_baseline"),
        "large_width_extrapolation": bool(vector.get("rtl_v6_large_width_banked_logic_extrapolation", False)),
    }


def _arm_area(
    arm: Arm,
    baseline: Mapping[str, Any],
    *,
    softmax_state_heads: int,
) -> dict[str, Any]:
    if arm.key == "rtl_v5":
        return dict(baseline)
    return _area_metrics(
        arm.row_lanes,
        version="rtl-v6",
        enable_multirow=arm.key.startswith("combined_"),
        enable_state=arm.state_schedule == "row-bank-simd-v3",
        enable_packed_pv=arm.pv_schedule == "direct-packed-rmw-v1",
        fp_sram_depth=(
            10
            if arm.state_schedule == "row-bank-simd-v3"
            else 2 * softmax_state_heads * _hardware()["MLEN"]
            + _hardware()["FP_CONSTANT_NUM"]
        ),
        softmax_state_heads=softmax_state_heads,
    )


def _arm_area_config(arm: Arm, *, softmax_state_heads: int) -> dict[str, Any]:
    return _area_config(
        arm.row_lanes,
        version="rtl-v5" if arm.key == "rtl_v5" else "rtl-v6",
        enable_multirow=arm.key.startswith("combined_"),
        enable_state=arm.state_schedule == "row-bank-simd-v3",
        enable_packed_pv=arm.pv_schedule == "direct-packed-rmw-v1",
        fp_sram_depth=(
            10
            if arm.state_schedule == "row-bank-simd-v3"
            else 2 * softmax_state_heads * _hardware()["MLEN"]
            + _hardware()["FP_CONSTANT_NUM"]
        ),
        softmax_state_heads=softmax_state_heads,
    )


def _system_power(
    config: Mapping[str, Any],
    trace: Mapping[str, Any],
    timing: Mapping[str, Any],
    *,
    clock_gating_mode: str = "ideal_hierarchical",
) -> dict[str, Any]:
    return estimate_system_power(
        config,
        trace,
        timing,
        external_memory_config={
            "HBM_CAPACITY_BYTES": 80_000_000_000,
            "HBM_BANDWIDTH_GBPS": 2039.0,
            "HBM_CONFIGURATION_SEMANTICS": "single_chip_80gb_layer_microbenchmark",
        },
        clock_gating_mode=clock_gating_mode,
    )


def _evaluate_actual(model: ModelCase, arm: Arm, settings: Path, baseline_area: Mapping[str, Any]) -> tuple[dict[str, Any], Any, Mapping[str, Any]]:
    trace, report_obj = compile_and_evaluate_compiler_cost(
        model.config,
        settings,
        DEFAULT_COMPILER_COST_CALIBRATION,
        seq_len=90_000,
        batch_size=8,
        num_layers=1,
        moe_routing_mode="fixed-balanced" if model.moe else "static-indices",
        moe_lowering_schedule="compact-route-v2",
        moe_layer_scaling="single-layer",
        native_layout_mode="compact",
        packed_attention_schedule="direct-first-block-v1",
        softmax_state_schedule=arm.state_schedule,
        packed_qk_schedule="broadcast-k-major-v1",
        vector_scalar_schedule="rtl-v5" if arm.key == "rtl_v5" else "rtl-v6",
        softmax_vector_schedule=arm.vector_schedule,
        pv_accumulation_schedule=arm.pv_schedule,
        softmax_row_lanes=arm.row_lanes,
        softmax_row_issue_schedule="wavefront-v1",
        selector_schedule="hoisted-v1",
        reduction_output_mode="overwrite-v1",
        gqa_pipeline_schedule="row-interleaved-v1",
        address_generation_mode="loop-agu-v1",
        ffn_address_schedule="live-stride-v1",
        ffn_projection_schedule="affine-loop-v2",
        cost_trace_granularity="affine-block-summary-v1",
        precision_config={
            "weight": "MXINT4",
            "activation": "MXINT4",
            "kv": "MXINT4",
            "block": 64,
            "scale_bits": 8,
            "integer_bits": 32,
            "internal_fp": "FP_E6M5",
        },
        compute_timing_mode="ideal-ii1",
        v4_memory_evaluation="one-layer-cached-occurrence-scaled",
        kv_residency_policy="kv-25",
        persistent_trace_cache_dir=REPO_ROOT / "Workspace/.cache/rtl_v6_single_layer_ab/traces",
        persistent_v4_work_cache_dir=REPO_ROOT / "Workspace/.cache/rtl_v6_single_layer_ab/v4",
        persistent_compute_pipeline_cache_dir=REPO_ROOT / "Workspace/.cache/rtl_v6_single_layer_ab/compute",
    )
    report = report_obj.to_dict()
    timing, latency_ns = _layer_timing(report)
    area = _arm_area(
        arm,
        baseline_area,
        softmax_state_heads=model.softmax_state_heads,
    )
    power = _system_power(
        _arm_area_config(
            arm, softmax_state_heads=model.softmax_state_heads
        ),
        _layer_trace(trace),
        timing,
    )
    opcodes = _opcode_rows(trace)
    packed_attention = dict(trace.metadata.get("packed_attention") or {})
    categories = _category_cycles(report)
    row = {
        "model": model.key,
        "model_label": model.label,
        "arm": arm.key,
        "arm_label": arm.label,
        "row_lanes": arm.row_lanes,
        "result_fidelity": arm.fidelity,
        "latency_ms": latency_ns / 1e6,
        "attention_latency_ms": float(timing["stage_roofline_latency_ns"].get("layer/attention", 0.0)) / 1e6,
        "ffn_or_moe_latency_ms": math.fsum(value for stage, value in timing["stage_roofline_latency_ns"].items() if stage != "layer/attention") / 1e6,
        "matrix_cycles": categories.get("matrix", 0),
        "vector_cycles": categories.get("vector", 0),
        "scalar_cycles": categories.get("scalar", 0),
        "control_cycles": categories.get("control", 0),
        "system_energy_mj": float(power["system_energy_p50_mj"]),
        "ungated_system_energy_mj": float(power["ungated_system_energy_p50_mj"]),
        "average_power_w": float(power["system_average_power_p50_w"]),
        "ideal_clock_energy_mj": float(power["ideal_clock_energy_mj"]),
        "core_area_mm2": float(area["core_area_mm2"]),
        "area_budget_fraction": float(area["budget_fraction"]),
        "state_fp_loads": int(opcodes.get("S_LD_FP", 0)),
        "state_fp_stores": int(opcodes.get("S_ST_FP", 0)),
        "pv_shift_ops": int(opcodes.get("V_SHIFT_V", 0)),
        "pv_vector_add_ops": int(opcodes.get("V_ADD_VV", 0)),
        "qk_compute_count": int(packed_attention.get("qk_compute_count", 0)),
        "pv_compute_count": int(packed_attention.get("pv_compute_count", 0)),
        "qk_matrix_ops": int(opcodes.get("M_BTMM", 0)),
        "pv_matrix_ops": int(opcodes.get("M_BMM_WO", 0)),
        "packed_pv_matrix_ops": int(opcodes.get("M_MM_WO_PACKED_ACC", 0)),
        "layer_dma_manifest_hash": _layer_dma_digest(trace),
        "hbm_physical_read_bytes": int(timing["hbm_read_bytes"]),
        "hbm_physical_write_bytes": int(timing["hbm_write_bytes"]),
        "area": area,
        "power_fidelity": power.get("rtl_v6_power_calibration_status", "not_applicable"),
        "stage_compute_latency_ns": timing["stage_compute_latency_ns"],
        "stage_memory_latency_ns": timing["hbm_stage_latency_ns"],
        "stage_roofline_latency_ns": timing["stage_roofline_latency_ns"],
        "stage_opcode_cycles": {
            stage: dict(opcode_counts)
            for stage, opcode_counts in report["stage_compute_opcode_work_cycles"].items()
            if stage.startswith("layer/")
        },
    }
    return row, trace, timing


def _project_high_r(
    base: Mapping[str, Any],
    trace: Any,
    timing: Mapping[str, Any],
    row_lanes: int,
    *,
    softmax_state_heads: int,
) -> dict[str, Any]:
    scale = 8.0 / row_lanes
    stage_compute = dict(base["stage_compute_latency_ns"])
    projected_opcode_cycles: dict[str, dict[str, int]] = {}
    for stage, opcodes in dict(base["stage_opcode_cycles"]).items():
        projected_opcode_cycles[stage] = {
            opcode: max(1, int(round(cycles * scale))) if opcode in ROW_OPCODES else int(cycles)
            for opcode, cycles in dict(opcodes).items()
        }
        old_rows = sum(int(cycles) for opcode, cycles in opcodes.items() if opcode in ROW_OPCODES)
        new_rows = sum(projected_opcode_cycles[stage][opcode] for opcode in opcodes if opcode in ROW_OPCODES)
        stage_compute[stage] = float(stage_compute[stage]) - old_rows + new_rows
    stage_memory = dict(base["stage_memory_latency_ns"])
    stage_roofline = {
        stage: max(value, float(stage_memory.get(stage, 0.0)))
        for stage, value in stage_compute.items()
    }
    projected_timing = dict(timing)
    projected_timing.update(
        {
            "roofline_latency_ns": math.fsum(stage_roofline.values()),
            "compute_latency_ns": math.fsum(stage_compute.values()),
            "stage_compute_latency_ns": stage_compute,
            "stage_roofline_latency_ns": stage_roofline,
        }
    )
    projected_trace = _layer_trace(
        trace,
        count_scale=scale,
        projected_row_lanes=row_lanes,
    )
    area = _area_metrics(
        row_lanes,
        version="rtl-v6",
        softmax_state_heads=softmax_state_heads,
    )
    power = _system_power(
        _area_config(
            row_lanes, softmax_state_heads=softmax_state_heads
        ),
        projected_trace,
        projected_timing,
        clock_gating_mode="ungated",
    )
    projected_onchip_nominal_mj = (
        float(power["onchip_energy_mj"])
        - float(power["ungated_clock_energy_mj"])
        + float(base["ideal_clock_energy_mj"])
    )
    projected_system_nominal_mj = (
        projected_onchip_nominal_mj
        + float(power["external_hbm_energy_p50_mj"])
    )
    categories: Counter[str] = Counter()
    for opcodes in projected_opcode_cycles.values():
        for opcode, cycles in opcodes.items():
            categories[_opcode_category(opcode)] += cycles
    row = dict(base)
    row.update(
        {
            "arm": f"combined_r{row_lanes}",
            "arm_label": f"Combined R{row_lanes} projected",
            "row_lanes": row_lanes,
            "result_fidelity": "structural_extrapolation_not_isa_encodable",
            "latency_ms": projected_timing["roofline_latency_ns"] / 1e6,
            "attention_latency_ms": stage_roofline.get("layer/attention", 0.0) / 1e6,
            "ffn_or_moe_latency_ms": math.fsum(value for stage, value in stage_roofline.items() if stage != "layer/attention") / 1e6,
            "matrix_cycles": categories.get("matrix", 0),
            "vector_cycles": categories.get("vector", 0),
            "scalar_cycles": categories.get("scalar", 0),
            "control_cycles": categories.get("control", 0),
            "system_energy_mj": projected_system_nominal_mj,
            "ungated_system_energy_mj": float(power["ungated_system_energy_p50_mj"]),
            "average_power_w": projected_system_nominal_mj
            / (projected_timing["roofline_latency_ns"] / 1e6),
            "ideal_clock_energy_mj": float(base["ideal_clock_energy_mj"]),
            "core_area_mm2": float(area["core_area_mm2"]),
            "area_budget_fraction": float(area["budget_fraction"]),
            "area": area,
            "power_fidelity": "power_structural_extrapolation",
            "stage_compute_latency_ns": stage_compute,
            "stage_roofline_latency_ns": stage_roofline,
            "stage_opcode_cycles": projected_opcode_cycles,
        }
    )
    return row


def _derive(rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    for members in by_model.values():
        baseline = next(row for row in members if row["arm"] == "rtl_v5")
        for row in members:
            row["speedup_vs_rtl_v5"] = baseline["latency_ms"] / row["latency_ms"]
            row["latency_reduction_vs_rtl_v5_pct"] = (1.0 - row["latency_ms"] / baseline["latency_ms"]) * 100.0
            row["energy_reduction_vs_rtl_v5_pct"] = (1.0 - row["system_energy_mj"] / baseline["system_energy_mj"]) * 100.0
            row["area_delta_vs_rtl_v5_mm2"] = row["core_area_mm2"] - baseline["core_area_mm2"]
            row["scalar_fp_load_ops_eliminated"] = (
                baseline["state_fp_loads"] - row["state_fp_loads"]
            )
            row["scalar_fp_store_ops_eliminated"] = (
                baseline["state_fp_stores"] - row["state_fp_stores"]
            )
            row["vector_shift_ops_eliminated"] = (
                baseline["pv_shift_ops"] - row["pv_shift_ops"]
            )
            row["vector_add_ops_eliminated"] = (
                baseline["pv_vector_add_ops"] - row["pv_vector_add_ops"]
            )
            row["marginal_latency_saved_per_mm2"] = None
            row["marginal_speedup_vs_previous"] = None
        combined = sorted(
            (row for row in members if row["arm"].startswith("combined_r")),
            key=lambda row: row["row_lanes"],
        )
        previous = None
        for row in combined:
            if previous is not None:
                row["marginal_speedup_vs_previous"] = (
                    previous["latency_ms"] / row["latency_ms"]
                )
                if row["core_area_mm2"] > previous["core_area_mm2"]:
                    row["marginal_latency_saved_per_mm2"] = (
                        previous["latency_ms"] - row["latency_ms"]
                    ) / (row["core_area_mm2"] - previous["core_area_mm2"])
            previous = row


def _validate_actual_invariants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in MODELS:
        members = [
            row
            for row in rows
            if row["model"] == model.key
            and row["result_fidelity"]
            == "costemitter_actual_rtl_isa_supported"
        ]
        fields = (
            "qk_compute_count",
            "pv_compute_count",
            "hbm_physical_read_bytes",
            "hbm_physical_write_bytes",
            "layer_dma_manifest_hash",
        )
        drift = {
            field: sorted({row[field] for row in members}, key=str)
            for field in fields
            if len({row[field] for row in members}) != 1
        }
        if drift:
            raise RuntimeError(
                f"{model.key} rtl-v6 A/B invariant drift: {drift}"
            )
        result[model.key] = {
            field: members[0][field] for field in fields
        }
    return result


def _banking_rows() -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for lanes in (1, 2, 4, 8, 16, 32):
            area = _area_metrics(
                lanes,
                version="rtl-v6",
                softmax_state_heads=model.softmax_state_heads,
            )
            banking = area["vector_sram_banking"]
            logical = int(banking["logical_bits"])
            covered = int(banking["covered_capacity_bits"])
            bank_depths = [
                int(value) for value in banking["physical_bank_depths"]
            ]
            rows.append(
                {
                "model": model.key,
                "row_lanes": lanes,
                "fidelity": area["row_lane_fidelity"],
                "logical_bits": logical,
                "physical_bank_count": int(banking["physical_bank_count"]),
                "per_bank_logical_depth": max(bank_depths),
                "covered_macro_bits": covered,
                "macro_rounding_bits": int(banking["macro_rounding_overhead_bits"]),
                "covered_to_logical_ratio": covered / logical,
                "macro_waste_pct": (covered - logical) / covered * 100.0,
                "banked_sram_area_mm2": float(banking["selected_banked_area_um2"]) / 1e6,
                "banking_area_delta_mm2": float(banking["selected_banking_area_delta_um2"]) / 1e6,
                "rtl_v6_logic_delta_mm2": area["rtl_v6_logic_delta_mm2"],
                "state_stat_factor_area_mm2": area["state_stat_factor_area_mm2"],
                "core_area_mm2": area["core_area_mm2"],
                "area_budget_fraction": area["budget_fraction"],
                "large_width_extrapolation": area["large_width_extrapolation"],
                }
            )
    return rows


def _flat_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not isinstance(value, (dict, list, tuple))
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _markdown(rows: list[dict[str, Any]], banking: list[dict[str, Any]]) -> str:
    lines = [
        "# RTL-v6 Long-Context Single-Layer A/B v2",
        "",
        "> Status: CostEmitter architectural ideal-II1 A/B. R1/2/4/8 use emitted compiler traces; R16/32 are explicitly non-ISA structural projections.",
        "",
        "## Experiment boundary",
        "",
        "The experiment evaluates one decoder layer on one PLENA chip at `seq=90,000`, `batch=8`, `M/V=2048`, `B=128`, W4/A4/KV4 and FP E6M5. No N, DP, TP, EP, NVLink, or interconnect scaling is present. The HBM power model is provisioned as one abstract 80 GB / 2039 GB/s stack.",
        "",
        "The 235B row is a fixed-balanced MoE layer microbenchmark. It is not a claim that the complete 235B model fits in one 80 GB chip, and no model TTFT is reported.",
        "",
    ]
    for model in MODELS:
        members = [row for row in rows if row["model"] == model.key]
        lines.extend(
            [
                f"## {model.label}",
                "",
                "| Arm | R | Fidelity | Layer latency (ms) | Speedup | Energy (kJ) | Energy change | Core area (mm2) | Area delta |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in members:
            lines.append(
                "| {arm} | {r} | `{fidelity}` | {latency} | {speedup}x | {energy} | {energy_delta}% | {area} | {area_delta} |".format(
                    arm=row["arm_label"],
                    r=row["row_lanes"],
                    fidelity=row["result_fidelity"],
                    latency=_fmt(row["latency_ms"]),
                    speedup=_fmt(row["speedup_vs_rtl_v5"], 3),
                    energy=_fmt(row["system_energy_mj"] / 1e6, 3),
                    energy_delta=_fmt(-row["energy_reduction_vs_rtl_v5_pct"], 1),
                    area=_fmt(row["core_area_mm2"]),
                    area_delta=_fmt(row["area_delta_vs_rtl_v5_mm2"]),
                )
            )
        lines.extend(
            [
                "",
                "### Removed data movement and control work",
                "",
                "These columns are whole-layer opcode deltas relative to rtl-v5; residual loads/shifts/adds belong to non-state or non-PV work.",
                "",
                "| Arm | Scalar FP loads removed | Scalar FP stores removed | Vector shifts removed | Vector adds removed | Ungated energy (kJ) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in members:
            lines.append(
                f"| {row['arm_label']} | {_fmt(row['scalar_fp_load_ops_eliminated']/1e9,3)}G | {_fmt(row['scalar_fp_store_ops_eliminated']/1e9,3)}G | {_fmt(row['vector_shift_ops_eliminated']/1e9,3)}G | {_fmt(row['vector_add_ops_eliminated']/1e9,3)}G | {_fmt(row['ungated_system_energy_mj']/1e6,3)} |"
            )
        lines.extend(
            [
                "",
                "### Work breakdown",
                "",
                "| Arm | Attention (ms) | FFN/MoE (ms) | Matrix Gcycles | Vector Gcycles | Scalar Gcycles | Control Gcycles |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in members:
            lines.append(
                f"| {row['arm_label']} | {_fmt(row['attention_latency_ms'])} | {_fmt(row['ffn_or_moe_latency_ms'])} | {_fmt(row['matrix_cycles']/1e9,3)} | {_fmt(row['vector_cycles']/1e9,3)} | {_fmt(row['scalar_cycles']/1e9,3)} | {_fmt(row['control_cycles']/1e9,3)} |"
            )
        lines.append("")
        lines.extend(
            [
                "### R-tier marginal return",
                "",
                "| Tier | Marginal speedup | Latency saved / added mm2 | Area budget used |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in members:
            if not row["arm"].startswith("combined_r"):
                continue
            marginal = row["marginal_speedup_vs_previous"]
            saved = row["marginal_latency_saved_per_mm2"]
            lines.append(
                f"| R{row['row_lanes']} | {'-' if marginal is None else _fmt(marginal,3)+'x'} | {'-' if saved is None else _fmt(saved,1)+' ms/mm2'} | {_fmt(row['area_budget_fraction']*100,2)}% |"
            )
        lines.append("")
    lines.extend(
        [
            "## Vector SRAM banking granularity",
            "",
            "| Model / R | Fidelity | Banks | Bank depth | Logical Mibit | Covered Mibit | Covered/logical | Macro waste | Banked SRAM (mm2) | Banking delta (mm2) | Core area (mm2) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in banking:
        lines.append(
            f"| {row['model']} R{row['row_lanes']} | `{row['fidelity']}` | {row['physical_bank_count']} | {row['per_bank_logical_depth']} | {_fmt(row['logical_bits']/2**20,2)} | {_fmt(row['covered_macro_bits']/2**20,2)} | {_fmt(row['covered_to_logical_ratio'],2)}x | {_fmt(row['macro_waste_pct'],1)}% | {_fmt(row['banked_sram_area_mm2'],3)} | {_fmt(row['banking_area_delta_mm2'],3)} | {_fmt(row['core_area_mm2'],2)} |"
        )
    cliff = next(
        row
        for row in banking
        if row["model"] == MODELS[0].key
        and row["covered_to_logical_ratio"] >= 3.0
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `rtl-v5 -> state only/direct PV/combined R1` separates packed state and direct writeback from query-row spatial parallelism.",
            "- `R1 -> R8` is the actual compiler/CostEmitter row-throughput sweep. Matrix arithmetic and physical HBM traffic are required to remain invariant.",
            f"- The first severe macro-granularity cliff is R={cliff['row_lanes']}: covered capacity reaches {_fmt(cliff['covered_to_logical_ratio'],2)}x logical capacity even though logical data are not replicated.",
            "- R16/R32 extrapolate row-group counts and calibrated area/power trends beyond the current 2-bit row-tier ISA. They are diagnostic bounds, not implementation candidates.",
            "- VLEN=2048 is beyond the paired-DC logic width range (up to VLEN=64); bank macro tiling is exact, while row-slice/control logic is a structural extrapolation.",
            "- Recommended tiers: R4 is the area-efficient default; R8 is the low-latency implementation candidate when the extra 7.37 mm2 over R4 is acceptable. R16/R32 are rejected as implementation candidates because they cross the ISA boundary and lose macro/area efficiency.",
            "- Ideal-gated energy falls mainly because the shorter layer makespan reduces leakage and 80 GB HBM background time. Ungated energy is reported separately and rises again for projected R16/R32 as replicated row logic grows.",
            "- Qwen3-235B-A22B has a lower representative-layer time here because fixed-balanced MoE executes top-k expert work for one layer; this is not a full-model performance comparison.",
            "",
            "## Fidelity and exclusions",
            "",
            "- Ideal-II1 is an architectural throughput assumption, not cycle-exact RTL timing.",
            "- No top-level RTL run, timing closure, package, cooling, interconnect, or full-model capacity result is included.",
            "- Ideal dual-port SRAM is the nominal port model. SRAM background uses the current literature lower-end proxy.",
            f"- Area percentages use the conservative single-A100-equivalent budget `{AREA_BUDGET_MM2:.1f} mm2`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rtl_v6_single_layer_ab_") as temporary:
        for model in MODELS:
            legacy_fp_depth = (
                2 * model.softmax_state_heads * _hardware()["MLEN"]
                + _hardware()["FP_CONSTANT_NUM"]
            )
            baseline_area = _area_metrics(
                1,
                version="rtl-v5",
                fp_sram_depth=legacy_fp_depth,
                softmax_state_heads=model.softmax_state_heads,
            )
            r8_trace = None
            r8_timing = None
            r8_row = None
            for arm in ACTUAL_ARMS:
                print(
                    f"[rtl-v6-ab] evaluating {model.key}/{arm.key}",
                    flush=True,
                )
                settings = Path(temporary) / f"settings_{arm.key}.toml"
                _write_settings(
                    settings,
                    fp_sram_depth=(
                        10
                        if arm.state_schedule == "row-bank-simd-v3"
                        else legacy_fp_depth
                    ),
                )
                row, trace, timing = _evaluate_actual(model, arm, settings, baseline_area)
                rows.append(row)
                print(
                    f"[rtl-v6-ab] complete {model.key}/{arm.key}: "
                    f"{row['latency_ms']:.3f} ms",
                    flush=True,
                )
                if arm.key == "combined_r8":
                    r8_trace, r8_timing, r8_row = trace, timing, row
            assert r8_trace is not None and r8_timing is not None and r8_row is not None
            rows.append(
                _project_high_r(
                    r8_row,
                    r8_trace,
                    r8_timing,
                    16,
                    softmax_state_heads=model.softmax_state_heads,
                )
            )
            rows.append(
                _project_high_r(
                    r8_row,
                    r8_trace,
                    r8_timing,
                    32,
                    softmax_state_heads=model.softmax_state_heads,
                )
            )
    _derive(rows)
    invariants = _validate_actual_invariants(rows)
    banking = _banking_rows()
    payload = {
        "schema": "rtl_v6_long_context_single_layer_ab_v2",
        "setup": {
            "seq_len": 90_000,
            "batch_size": 8,
            "num_layers": 1,
            "MLEN": 2048,
            "VLEN": 2048,
            "BLEN": 128,
            "precision": _precision()["name"],
            "matrix_sram_policy": "kv-25",
            "chip_count": 1,
            "hbm_capacity_bytes": 80_000_000_000,
            "hbm_bandwidth_gbps": 2039.0,
            "area_budget_mm2": AREA_BUDGET_MM2,
            "compute_timing": "ideal-ii1",
            "legacy_scalar_fp_sram_depth_by_model": {
                model.key: 2 * model.softmax_state_heads * _hardware()["MLEN"]
                + _hardware()["FP_CONSTANT_NUM"]
                for model in MODELS
            },
            "packed_state_scalar_fp_sram_depth": 10,
            "full_model_capacity_checked": False,
            "multi_chip_model_used": False,
        },
        "results": rows,
        "banking": banking,
        "actual_arm_invariants": invariants,
    }
    (output_dir / f"{REPORT_STEM}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_csv(output_dir / f"{REPORT_STEM}.csv", [_flat_row(row) for row in rows])
    _write_csv(output_dir / BANKING_REPORT_NAME, banking)
    (output_dir / f"{REPORT_STEM}.md").write_text(_markdown(rows, banking))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    payload = run(args.output_dir)
    print(json.dumps({"status": "complete", "rows": len(payload["results"]), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
