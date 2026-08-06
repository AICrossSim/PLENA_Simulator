#!/usr/bin/env python3
"""Optuna DSE for Qwen3 dense and fixed-balanced MoE prefill workloads.

Area remains a feasibility constraint and accuracy comes from prefiltered
external precision profiles; both remain first-class report fields and
selectors. The default objective uses the native compiler CostEmitter with
ideal-II1 compute timing, RTL-v6
Vector/Scalar lowering, production-DMA V4 memory work, partial-resident K/V,
and an explicitly labelled stage-level multi-chip model. Legacy and
hazard-aware timing modes remain available for diagnostics. Area defaults to
the precision-aware proxy with ideal dual-port SRAM semantics.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import fcntl
import gc
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import optuna
import toml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PLENA_COMPILER_ROOT = REPO_ROOT / "PLENA_Compiler"
if str(PLENA_COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(PLENA_COMPILER_ROOT))

from compiler.aten.plena.native_layout import (  # noqa: E402
    PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1,
    PACKED_QK_SCHEDULE_HEAD_MAJOR_V1,
    SOFTMAX_STATE_SCHEDULE_SRAM_V1,
    SOFTMAX_STATE_SCHEDULE_STREAMED_V2,
    SOFTMAX_STATE_SCHEDULE_ROW_BANK_SIMD_V3,
    SequencePackingPlan,
    build_attention_head_packing,
    build_compact_stats_plan,
    build_softmax_state_layout,
)
from compiler.aten.plena.kv_residency import (  # noqa: E402
    MATRIX_SRAM_POLICIES,
    derive_matrix_sram_policy,
    plan_kv_residency,
)
from analytic_models.performance.multi_chip_model import (  # noqa: E402
    BASE_MATRIX_SRAM_TILES,
    DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS,
    ENDPOINT_AREA_MM2_PER_PORT,
    PARALLEL_MODELS,
    aggregate_area,
    estimate_decode_kv_handoff,
    estimate_multi_chip_latency,
    matrix_sram_requirements,
    matrix_sram_search_values,
    parse_positive_int_csv,
    projection_chunk_metadata,
    valid_ep_degrees,
    valid_tp_degrees,
)
from analytic_models.performance.multi_chip_dp_model import (  # noqa: E402
    TILE_AWARE_DP_MULTI_CHIP_MODEL,
    valid_dp_tp_ep_topologies,
)
from analytic_models.power import estimate_multi_chip_system_power  # noqa: E402
from analytic_models.power.multi_chip import (  # noqa: E402
    DEFAULT_INTERCONNECT_ENERGY,
)
from analytic_models.dse.artifacts import (  # noqa: E402
    DSECacheDirectories,
    GLOBAL_DSE_CACHE_SCHEMA,
    build_physical_candidate_bank,
    cache_entry_path,
    canonical_json_sha256,
    compact_trial_record,
    finalize_compact_artifacts,
    load_cached_json,
    load_or_create_json_cache_metadata,
    load_json,
    persist_trial_record,
    selector_trial_summary,
    trial_lifecycle_record,
    write_json,
)
from analytic_models.dse.domain import (  # noqa: E402
    CHIP_COUNT_SCALING_MODES,
    LEGAL_BLENS_BY_MLEN,
    SHAPE_DOMAIN_POLICY,
    canonical_sram_choices as _canonical_sram_choices,
    conditional_blen_param_name,
    conditional_parallel_config_param_name,
    conditional_ep_param_name,
    conditional_mlen_param_name,
    conditional_sram_param_name,
    conditional_tp_param_name,
    valid_blen_log2_values,
    valid_blen_values,
    valid_mlen_values,
    scale_chip_counts_for_reference,
)
from analytic_models.dse.profiles import (  # noqa: E402
    CURRENT_DSE_PROFILE,
    RTL_VALIDATION_PROFILE,
)
from analytic_models.dse.precision_search import (  # noqa: E402
    PRECISION_SEARCH_ENCODINGS,
    PRECISION_SIGNATURE_PARAM,
    PRECISION_SIGNATURE_SCHEMA,
    build_matrix_datapath_signatures,
    conditional_precision_variant_param_name,
    matrix_datapath_signature_distance,
    precision_variant_distance,
)
from analytic_models.dse.cli import (  # noqa: E402
    add_model_profile_argument,
    model_profile_consistency,
)
from analytic_models.dse.objective import (  # noqa: E402
    OBJECTIVE_DIRECTIONS,
    OBJECTIVE_NORMALIZATION,
    ObjectiveValues,
)
from analytic_models.dse.resources import (  # noqa: E402
    current_process_rss_gib,
    logical_cpu_capacity,
    mem_available_gib,
    peak_rss_gib,
    percentile,
    process_tree_cpu_seconds,
    process_tree_rss_gib,
    system_cpu_jiffies,
)
from analytic_models.dse.results import (  # noqa: E402
    select_area_reference_candidates,
    write_multi_chip_analysis,
)
from analytic_models.dse.workers import (  # noqa: E402
    DEFAULT_WORKER_POLICY,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent
DEFAULT_GLOBAL_DSE_CACHE_DIR = REPO_ROOT / "Workspace" / ".cache" / "dse"
RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
DEFAULT_MODEL_CONFIG = (
    REPO_ROOT / "Workspace/qwen3_32b_dense_analytic/qwen3-32b.json"
)
BASE_ANALYTIC_TOML = REPO_ROOT / "Workspace/qwen3_235b_a22b_analytic/analytic_smoke_hardware.toml"
ISA_LIB = REPO_ROOT / "analytic_models/performance/customISA_lib.json"
AREA_REPORT = RTL_ROOT / "build/synth/plena/latest/reports/plena_area.rpt"
POWER_REPORT = RTL_ROOT / "build/synth/plena/latest/reports/plena_power.rpt"
ELAB_AREA_REPORT = RTL_ROOT / "build/elab/plena/latest/reports/plena_generic_area.rpt"
ELAB_SUMMARY_REPORT = RTL_ROOT / "build/elab/plena/latest/logs/summary.log"
DEFAULT_FP_CONSTANT_NUM = 10
DEFAULT_BANDWIDTH_LIMIT_GBPS = 2039.0
DEFAULT_FREQUENCY_GHZ = 1.0
DEFAULT_INPUT_SEQ_LEN = 482
DEFAULT_OUTPUT_SEQ_LEN = 1
DEFAULT_DEVICE_NUM = 1
DEFAULT_LATENCY_BATCH_SIZE = 16
DEFAULT_HBM_CAPACITY_BYTES = 80_000_000_000
KV_CAPACITY_MODE_STREAMED_HANDOFF_V1 = "prefill-streamed-handoff-v1"
KV_CAPACITY_MODE_FULL_CACHE_V1 = "full-decoder-cache-v1"
KV_CAPACITY_MODES = (
    KV_CAPACITY_MODE_STREAMED_HANDOFF_V1,
    KV_CAPACITY_MODE_FULL_CACHE_V1,
)
DEFAULT_KV_CAPACITY_MODE = KV_CAPACITY_MODE_STREAMED_HANDOFF_V1
DEFAULT_KV_HANDOFF_STAGING_LAYERS = 1
DEFAULT_CHIP_COUNTS = (1, 2, 4, 8, 16)
DEFAULT_REFERENCE_A100_COUNT = 1
DEFAULT_CHIP_COUNT_SCALING = "per-a100-reference"
DEFAULT_ENDPOINT_OVERHEAD_FRACTION = 0.10
DEFAULT_NVLINK_BIDIRECTIONAL_GBPS = 3_600.0
DEFAULT_NVLINK_ONE_WAY_GBPS = DEFAULT_NVLINK_BIDIRECTIONAL_GBPS / 2.0
DEFAULT_MULTI_CHIP_MODEL = CURRENT_DSE_PROFILE.multi_chip_model
DEFAULT_NVLINK_PORT_COUNTS = (1, 2, 4)
DEFAULT_NVLINK_STARTUP_US = 2.5
FACTORIZED_MULTI_CHIP_MODELS = frozenset(
    {
        "factorized-tp-cp-v2",
        "tile-aware-tp-cp-ep-v3",
        TILE_AWARE_DP_MULTI_CHIP_MODEL,
    }
)
DEFAULT_WEIGHT_PARAM_COUNT = 32_000_000_000
DEFAULT_WEIGHT_ELEMENT_BITS = 4.0
DEFAULT_WEIGHT_PRECISION = "MXINT4"
DEFAULT_MX_SCALE_WIDTH = 8
DEFAULT_MX_SCALE_BLOCK_SIZE = 64
DEFAULT_WEIGHT_MX_EXP_WIDTH = 2
DEFAULT_WEIGHT_MX_MANT_WIDTH = 1
FRACTIONAL_LATENCY_MODEL_NAME = (
    "compiler_stage_roofline_ideal_ii1_v4_factorized_tp_cp_v10"
)
TILE_AWARE_LATENCY_MODEL_NAME = (
    "compiler_stage_roofline_ideal_ii1_v4_tile_aware_dp_tp_ep_v15_rtl_v6_streamed_kv_handoff"
)
DEFAULT_HLEN = 128
DEFAULT_BROADCAST_AMOUNT = 8
# Preserve the full RTL topology space by default.  Values above one are a
# diagnostic filter, not a hardware-validity constraint.
DEFAULT_MIN_MATRIX_K_SPLITS = 1
GA100_REFERENCE_AREA_MM2 = 826.0
DEFAULT_TARGET_AREA_MM2 = GA100_REFERENCE_AREA_MM2
DEFAULT_AREA_BUDGET_MM2 = GA100_REFERENCE_AREA_MM2 * 1.10
DEFAULT_TARGET_AREA_TOLERANCE_PCT = 5.0
DEFAULT_ACCURACY_PATH = (
    WORKSPACE_ROOT
    / "software_accuracy_inputs/software_precision_profiles_accuracy_gt_0p9.json"
)
DEFAULT_COMPILER_COST_SETTINGS = (
    REPO_ROOT
    / "Workspace/qwen3_32b_transactional_prefetch_sweep/runs/"
    "gqa_logical_kv_optimized_20260710/trial_0000/plena_settings.toml"
)
DEFAULT_COMPILER_COST_CALIBRATION = (
    REPO_ROOT / "analytic_models/performance/calibration/hbm_dma_service_v4.json"
)
DEFAULT_EXTERNAL_MEMORY_ENERGY = (
    REPO_ROOT
    / "analytic_models/power/calibration/external_memory_hbm3e_v1.json"
)
DEFAULT_SRAM_BACKGROUND_ENERGY = (
    REPO_ROOT
    / "analytic_models/power/calibration/sram_background_memexplorer_v1.json"
)
FRACTIONAL_OBJECTIVE_SCHEMA = (
    "latency_energy_identity_normalized_factorized_tp_cp_v2"
)
TILE_AWARE_OBJECTIVE_SCHEMA = (
    "latency_energy_identity_normalized_tile_aware_dp_tp_ep_v9_rtl_v6_hard_area_streamed_kv_handoff"
)
SEARCH_SCHEMA = "canonical_conditional_hardware_v7_factorized_tp_cp_ports"
TILE_AWARE_SEARCH_SCHEMA = (
    "canonical_conditional_hardware_v13_tile_aware_dp_tp_ep_rtl_v6_lineage_full_shape_domain"
)
SEARCH_ENCODINGS = ("canonical-conditional-v1", "legacy-policy-v1")
DEFAULT_OPTUNA_TRIALS = 2048
DEFAULT_OPTUNA_WORKERS = DEFAULT_WORKER_POLICY.worker_cap
DEFAULT_WORKER_RSS_RECYCLE_GIB = DEFAULT_WORKER_POLICY.rss_recycle_gib
DEFAULT_INITIAL_WORKER_RSS_GIB = DEFAULT_WORKER_POLICY.initial_rss_gib
DEFAULT_MEMORY_RESERVE_GIB = DEFAULT_WORKER_POLICY.launch_reserve_gib
DEFAULT_MEMORY_RESUME_GIB = DEFAULT_WORKER_POLICY.resume_gib
DEFAULT_MEMORY_EMERGENCY_GIB = DEFAULT_WORKER_POLICY.emergency_gib
DEFAULT_PROCESS_TREE_RSS_LIMIT_GIB = (
    DEFAULT_WORKER_POLICY.process_tree_limit_gib
)
DEFAULT_WORKER_STALL_TIMEOUT_SECONDS = (
    DEFAULT_WORKER_POLICY.stall_timeout_seconds
)
DEFAULT_WORKER_LAUNCH_BURST = DEFAULT_WORKER_POLICY.launch_burst
DEFAULT_WORKER_LAUNCH_INTERVAL_SECONDS = (
    DEFAULT_WORKER_POLICY.launch_interval_seconds
)
DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS = (
    DEFAULT_WORKER_POLICY.monitor_interval_seconds
)
DEFAULT_TPE_STARTUP_TRIALS = 256
DEFAULT_TPE_EI_CANDIDATES = 128
COMPILER_COST_OBJECTIVE_MODES = {
    "compute-objective",
    "roofline-objective",
    "objective",
}

DEFAULT_SEARCH_SPACE = {
    "MLEN": [256, 512, 1024, 2048, 4096, 8192],
    "BLEN": [32, 64, 128, 256, 512, 1024],
    "INT_DATA_WIDTH": [16, 32, 64],
}

@dataclass(frozen=True)
class DSEConfig:
    input_seq_len: int
    output_seq_len: int
    device_num: int
    latency_batch_size: int
    hbm_capacity_bytes: int
    hbm_bandwidth_gbps: float
    frequency_ghz: float
    mx_scale_width: int
    mx_scale_block_size: int
    fp_constant_num: int
    weight_param_count: float
    weight_element_bits: float
    weight_precision: str
    weight_mx_exp_width: int
    weight_mx_mant_width: int
    kv_capacity_mode: str = DEFAULT_KV_CAPACITY_MODE
    kv_handoff_staging_layers: int = DEFAULT_KV_HANDOFF_STAGING_LAYERS
    softmax_state_schedule: str = SOFTMAX_STATE_SCHEDULE_STREAMED_V2
    packed_qk_schedule: str = PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1

    @property
    def bandwidth_limit_bytes_per_cycle(self) -> float:
        # Decimal GB/s and GHz cancel to bytes/cycle.
        return self.hbm_bandwidth_gbps / self.frequency_ghz

    @property
    def bandwidth_limit_bits_per_cycle(self) -> float:
        return self.bandwidth_limit_bytes_per_cycle * 8.0

    @property
    def weight_effective_bits(self) -> float:
        return self.weight_element_bits + self.mx_scale_width / self.mx_scale_block_size


class TrialPrunedError(Exception):
    """Local pruning exception with a reason string."""


def area_budget_violation_reason(
    total_silicon_area_mm2: float,
    area_budget_mm2: float,
) -> str | None:
    """Return a stable prune reason when aggregate silicon exceeds budget."""
    area = float(total_silicon_area_mm2)
    budget = float(area_budget_mm2)
    if not math.isfinite(area) or not math.isfinite(budget) or budget <= 0.0:
        raise ValueError(
            "area feasibility requires finite positive values, got "
            f"area={area}, budget={budget}"
        )
    if area <= budget:
        return None
    return (
        "aggregate silicon area exceeds budget: "
        f"{area:.6f} mm2 > {budget:.6f} mm2"
    )


def aggregate_area_from_core_metrics(
    area_metrics: Mapping[str, Any],
    *,
    chip_count: int,
    multi_chip_model: str,
    endpoint_area_overhead_fraction: float,
    nvlink_port_count: int,
) -> dict[str, Any]:
    core_area_um2 = float(
        area_metrics.get("area_um2", area_metrics.get("area", 0.0))
    )
    core_area_mm2 = float(
        area_metrics.get("area_mm2", core_area_um2 / 1e6)
    )
    return aggregate_area(
        core_area_mm2=core_area_mm2,
        core_area_p10_mm2=float(
            area_metrics.get("area_uncertainty_p10_mm2", core_area_mm2)
        ),
        core_area_p50_mm2=float(
            area_metrics.get("area_uncertainty_p50_mm2", core_area_mm2)
        ),
        core_area_p90_mm2=float(
            area_metrics.get("area_uncertainty_p90_mm2", core_area_mm2)
        ),
        chip_count=int(chip_count),
        endpoint_overhead_fraction=(
            float(endpoint_area_overhead_fraction)
            if multi_chip_model == "ideal-linear-lower-bound-v1"
            else None
        ),
        nvlink_port_count=(
            int(nvlink_port_count)
            if multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
            else None
        ),
        endpoint_area_mm2_per_port=ENDPOINT_AREA_MM2_PER_PORT["nominal"],
    )


def stable_key(data: dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def canonical_sram_choices(
    *,
    policies: tuple[str, ...],
    k_blocks: int,
    mlen: int,
    projection_tiles: int,
) -> tuple[dict[str, Any], ...]:
    return _canonical_sram_choices(
        policies=policies,
        k_blocks=k_blocks,
        mlen=mlen,
        projection_tiles=projection_tiles,
        derive_policy=derive_matrix_sram_policy,
    )


def current_rss_gib() -> float:
    """Compatibility alias for the worker's peak RSS."""

    return peak_rss_gib()


def file_sha256(path: Path) -> str:
    """Return a full artifact digest for study compatibility checks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(*roots: Path) -> str:
    """Hash model source/calibration trees for safe cross-study caches."""

    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.suffix in {".py", ".json", ".csv"}
            and "__pycache__" not in candidate.parts
        ):
            digest.update(str(path.relative_to(REPO_ROOT)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def precision_profile_distance(
    left: str,
    right: str,
    profiles: Mapping[str, Mapping[str, Any]],
) -> float:
    """Give TPE useful structure for the precision-profile category."""

    if left == right:
        return 0.0
    lhs = profiles[str(left)]
    rhs = profiles[str(right)]
    distance = 0.0
    for role in ("ACT_WIDTH", "KV_WIDTH", "WEIGHT_WIDTH"):
        a = parse_mx_precision(lhs[role])
        b = parse_mx_precision(rhs[role])
        distance += 2.0 if a["family"] != b["family"] else 0.0
        distance += abs(float(a["width"]) - float(b["width"])) / 8.0
    a_fp = parse_mx_precision(lhs["FP_SETTING"])
    b_fp = parse_mx_precision(rhs["FP_SETTING"])
    distance += abs(float(a_fp["width"]) - float(b_fp["width"])) / 16.0
    distance += abs(
        float(lhs["accuracy_score"]) - float(rhs["accuracy_score"])
    ) * 10.0
    return distance


def bits_from_width_spec(spec: dict[str, Any], default_scale_width: int = DEFAULT_MX_SCALE_WIDTH) -> tuple[int, int]:
    kind = str(spec.get("kind", "MXFP")).upper()
    scale_width = int(spec.get("scale_width", spec.get("scale", default_scale_width)))
    if "INT" in kind:
        return int(spec.get("width", spec.get("bits", 8))), scale_width
    return 1 + int(spec["exp"]) + int(spec["mant"]), scale_width


def parse_mx_precision(spec: Any, default_scale_width: int = DEFAULT_MX_SCALE_WIDTH) -> dict[str, Any]:
    scale_width = default_scale_width
    if isinstance(spec, str):
        text = spec.upper().replace("_", "")
        if text.startswith("MXINT"):
            return {"family": "mxint", "width": int(text.removeprefix("MXINT")), "scale_width": scale_width}
        if text.startswith("MXFP"):
            text = text.removeprefix("MXFP")
        if text.startswith("E") and "M" in text:
            exp_text, mant_text = text[1:].split("M", 1)
            exp = int(exp_text)
            mant = int(mant_text)
            return {"family": "mxfp", "exp": exp, "mant": mant, "width": 1 + exp + mant, "scale_width": scale_width}
    if isinstance(spec, dict):
        kind = str(spec.get("kind", spec.get("type", ""))).upper().replace("_", "")
        scale_width = int(spec.get("scale_width", spec.get("scale", default_scale_width)))
        if "MXINT" in kind or "width" in spec or "bits" in spec:
            if kind.startswith("MXINT") and kind != "MXINT":
                width = int(kind.removeprefix("MXINT"))
            else:
                width = int(spec.get("width", spec.get("bits")))
            return {"family": "mxint", "width": width, "scale_width": scale_width}
        if "MXFP" in kind or {"exp", "mant"} <= set(spec):
            if kind.startswith("MXFP") and kind != "MXFP":
                fmt = kind.removeprefix("MXFP")
                exp_text, mant_text = fmt[1:].split("M", 1)
                exp = int(exp_text)
                mant = int(mant_text)
            else:
                exp = int(spec["exp"])
                mant = int(spec["mant"])
            return {"family": "mxfp", "exp": exp, "mant": mant, "width": 1 + exp + mant, "scale_width": scale_width}
    raise ValueError(f"unsupported MX precision spec: {spec!r}")


def weight_precision_element_bits(weight_precision: str, fallback: float) -> float:
    try:
        return float(parse_mx_precision(weight_precision)["width"])
    except Exception:
        return fallback


def profile_weight_spec(precision: dict[str, Any], config: DSEConfig) -> Any:
    return precision.get("WEIGHT_WIDTH", config.weight_precision)


def profile_weight_effective_bits(precision: dict[str, Any], config: DSEConfig) -> float:
    spec = profile_weight_spec(precision, config)
    parsed = parse_mx_precision(spec, config.mx_scale_width)
    block_size = config.mx_scale_block_size
    if isinstance(spec, dict):
        block_size = int(spec.get("block_size", spec.get("block", block_size)))
    return float(parsed["width"]) + float(parsed["scale_width"]) / block_size


def precision_label(spec: Any, default_scale_width: int = DEFAULT_MX_SCALE_WIDTH) -> str:
    parsed = parse_mx_precision(spec, default_scale_width)
    if parsed["family"] == "mxint":
        return f"MXINT{parsed['width']}"
    return f"MXFP_E{parsed['exp']}M{parsed['mant']}"


def rtl_precision_params(hw: dict[str, int], precision: dict[str, Any], config: DSEConfig) -> dict[str, int]:
    params = {
        "INT_DATA_WIDTH": hw["INT_DATA_WIDTH"],
        "BLOCK_DIM": hw["BLEN"],
    }

    act = parse_mx_precision(precision["ACT_WIDTH"], config.mx_scale_width)
    if act["family"] == "mxint":
        params.update({"ACT_MX_INT_ENABLE": 1, "ACT_MX_INT_WIDTH": act["width"]})
    else:
        params.update({
            "ACT_MX_INT_ENABLE": 0,
            "ACT_MXFP_EXP_WIDTH": act["exp"],
            "ACT_MXFP_MANT_WIDTH": act["mant"],
        })
    params["ACT_MX_SCALE_WIDTH"] = act["scale_width"]

    kv = parse_mx_precision(precision["KV_WIDTH"], config.mx_scale_width)
    if kv["family"] == "mxint":
        params.update({"KV_MX_INT_ENABLE": 1, "KV_MX_INT_WIDTH": kv["width"]})
    else:
        params.update({
            "KV_MX_INT_ENABLE": 0,
            "KV_MX_EXP_WIDTH": kv["exp"],
            "KV_MX_MANT_WIDTH": kv["mant"],
        })
    params["KV_MX_SCALE_WIDTH"] = kv["scale_width"]

    wt = parse_mx_precision(profile_weight_spec(precision, config), config.mx_scale_width)
    if wt["family"] == "mxint":
        params.update({"WT_MX_INT_ENABLE": 1, "WT_MX_INT_WIDTH": wt["width"]})
    else:
        params.update({
            "WT_MX_INT_ENABLE": 0,
            "WT_MX_EXP_WIDTH": wt["exp"],
            "WT_MX_MANT_WIDTH": wt["mant"],
        })
    params["WT_MX_SCALE_WIDTH"] = wt["scale_width"]
    params["MX_SCALE_WIDTH"] = max(act["scale_width"], kv["scale_width"], wt["scale_width"])

    fp_setting = precision["FP_SETTING"]
    fp_exp = int(fp_setting["exp"])
    fp_mant = int(fp_setting["mant"])
    for prefix in ("V_FP", "M_FP", "S_FP", "ROUND_FP"):
        params[f"{prefix}_EXP_WIDTH"] = fp_exp
        params[f"{prefix}_MANT_WIDTH"] = fp_mant
    return params


def derived_hardware(
    model: dict[str, Any],
    trial_params: dict[str, Any],
    config: DSEConfig,
    *,
    vector_scalar_schedule: str = "rtl-v5",
    softmax_row_lanes: int = 1,
) -> dict[str, int]:
    mlen = int(trial_params["MLEN"])
    vlen = int(trial_params["VLEN"])
    blen = int(trial_params["BLEN"])
    matrix_sram_tiles = int(trial_params.get("MATRIX_SRAM_TILES", 2))
    num_attention_heads = int(model.get("num_attention_heads", 64))
    num_key_value_heads = int(model.get("num_key_value_heads", 8))
    logical_broadcast = num_attention_heads // num_key_value_heads
    head_packing = build_attention_head_packing(
        mlen=mlen,
        hlen=DEFAULT_HLEN,
        head_dim=int(model["head_dim"]),
        logical_broadcast_amount=logical_broadcast,
        gqa_ratio=logical_broadcast,
        num_kv_heads=num_key_value_heads,
        mode="compact",
    )
    state_layout = build_softmax_state_layout(
        mlen=mlen,
        active_broadcast_heads=head_packing.broadcast_amount,
        schedule=config.softmax_state_schedule,
        fp_constant_num=config.fp_constant_num,
    )
    compact_stats_plan = build_compact_stats_plan(
        vlen=vlen,
        hlen=DEFAULT_HLEN,
        num_attention_heads=num_attention_heads,
        vector_scalar_schedule=vector_scalar_schedule,
    )
    return {
        "MLEN": mlen,
        "VLEN": vlen,
        "BLEN": blen,
        "HLEN": DEFAULT_HLEN,
        "BROADCAST_AMOUNT": logical_broadcast,
        "PHYSICAL_BROADCAST_AMOUNT": head_packing.broadcast_amount,
        "MATRIX_SRAM_TILES": matrix_sram_tiles,
        "MATRIX_SRAM_SIZE": matrix_sram_tiles * mlen,
        "VECTOR_SRAM_SIZE": 2 * model["head_dim"] + math.ceil(model["hidden_size"] / vlen),
        "INT_SRAM_DEPTH": 32,
        "FP_CONSTANT_NUM": config.fp_constant_num,
        "FP_SRAM_DEPTH": state_layout.required_depth,
        "FP_SRAM_REQUIRED_DEPTH": state_layout.required_depth,
        "COMPACT_STATS_LANES": compact_stats_plan.configured_lanes,
        "SOFTMAX_ROW_LANES": int(softmax_row_lanes),
        "SOFTMAX_STATE_BANK_ENTRIES": int(state_layout.state_bank_entries),
        "HBM_M_Prefetch_Amount": mlen,
        "HBM_V_Prefetch_Amount": blen,
        "HBM_V_Writeback_Amount": blen,
        "INT_DATA_WIDTH": int(trial_params["INT_DATA_WIDTH"]),
        "MATRIX_K_SPLITS": mlen // blen if mlen % blen == 0 else 0,
    }


def constraint_issues(
    model: dict[str, Any],
    hw: dict[str, int],
    precision: dict[str, Any],
    strict_bandwidth: bool,
    config: DSEConfig,
    *,
    min_matrix_k_splits: int = 1,
) -> tuple[list[str], list[str]]:
    issues = []
    warnings = []
    if hw["MLEN"] < hw["BLEN"]:
        issues.append("MLEN < BLEN")
    if hw["MLEN"] < hw["HLEN"]:
        issues.append("MLEN < HLEN")
    if hw["VLEN"] != hw["MLEN"]:
        issues.append("VLEN != MLEN")
    if hw["HLEN"] < int(model["head_dim"]):
        issues.append("HLEN < HEAD_DIM")
    if hw["MLEN"] % hw["BLEN"] != 0:
        issues.append("MLEN % BLEN != 0")
    elif hw["MLEN"] // hw["BLEN"] < min_matrix_k_splits:
        issues.append(
            "MLEN / BLEN gives fewer than "
            f"{min_matrix_k_splits} MatrixMachine K-splits"
        )
    if hw["MATRIX_SRAM_SIZE"] < 2 * hw["MLEN"]:
        issues.append("MATRIX_SRAM_SIZE < 2 * MLEN")
    vec_min = 2 * model["head_dim"] + math.ceil(model["hidden_size"] / hw["VLEN"])
    if hw["VECTOR_SRAM_SIZE"] < vec_min:
        issues.append(f"VECTOR_SRAM_SIZE < {vec_min}")
    if hw["INT_SRAM_DEPTH"] < 16:
        issues.append("INT_SRAM_DEPTH < 16")
    if (
        config.packed_qk_schedule == PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1
        and config.softmax_state_schedule
        not in {
            SOFTMAX_STATE_SCHEDULE_STREAMED_V2,
            SOFTMAX_STATE_SCHEDULE_ROW_BANK_SIMD_V3,
        }
    ):
        issues.append(
            "broadcast-k-major-v1 requires softmax streamed-v2 or row-bank-simd-v3"
        )
    if hw["FP_SRAM_DEPTH"] < hw["FP_SRAM_REQUIRED_DEPTH"]:
        issues.append(
            "FP_SRAM_DEPTH is smaller than the shared attention-state requirement"
        )

    act_width, act_scale_width = bits_from_width_spec(precision["ACT_WIDTH"], config.mx_scale_width)
    kv_width, kv_scale_width = bits_from_width_spec(precision["KV_WIDTH"], config.mx_scale_width)
    wt_width, wt_scale_width = bits_from_width_spec(
        profile_weight_spec(precision, config), config.mx_scale_width
    )
    matrix_width = max(wt_width, kv_width)
    vector_width = max(act_width, kv_width)
    matrix_scale_width = max(wt_scale_width, kv_scale_width)
    vector_scale_width = max(act_scale_width, kv_scale_width)
    m_bw = hw["MLEN"] * matrix_width + (hw["MLEN"] // hw["BLEN"]) * matrix_scale_width
    v_bw = hw["VLEN"] * vector_width + (hw["VLEN"] // hw["BLEN"]) * vector_scale_width
    kv_bw = hw["MLEN"] * kv_width
    limit_bits = config.bandwidth_limit_bits_per_cycle
    bandwidth_msg = (
        f"{limit_bits:g} bits/cycle "
        f"({config.hbm_bandwidth_gbps:g} GB/s @ {config.frequency_ghz:g}GHz, "
        f"{config.bandwidth_limit_bytes_per_cycle:g} bytes/cycle)"
    )
    if m_bw > limit_bits:
        (issues if strict_bandwidth else warnings).append(f"matrix bandwidth expression {m_bw} > {bandwidth_msg}")
    if v_bw > limit_bits:
        (issues if strict_bandwidth else warnings).append(f"vector bandwidth expression {v_bw} > {bandwidth_msg}")
    if kv_bw > limit_bits:
        (issues if strict_bandwidth else warnings).append(f"KV bandwidth expression {kv_bw} > {bandwidth_msg}")
    return issues, warnings


def legacy_bandwidth_diagnostics(
    hw: dict[str, int],
    precision: dict[str, Any],
    config: DSEConfig,
    *,
    chip_count: int,
) -> dict[str, float]:
    """Return the historical feed expressions as non-pruning diagnostics."""

    act_width, act_scale = bits_from_width_spec(
        precision["ACT_WIDTH"], config.mx_scale_width
    )
    kv_width, kv_scale = bits_from_width_spec(
        precision["KV_WIDTH"], config.mx_scale_width
    )
    wt_width, wt_scale = bits_from_width_spec(
        profile_weight_spec(precision, config), config.mx_scale_width
    )
    expressions = {
        "matrix": (
            hw["MLEN"] * max(wt_width, kv_width)
            + (hw["MLEN"] // hw["BLEN"]) * max(wt_scale, kv_scale)
        ),
        "vector": (
            hw["VLEN"] * max(act_width, kv_width)
            + (hw["VLEN"] // hw["BLEN"]) * max(act_scale, kv_scale)
        ),
        "kv": hw["MLEN"] * kv_width,
    }
    per_chip_limit = config.bandwidth_limit_bits_per_cycle / chip_count
    return {
        "legacy_matrix_feed_bits_per_cycle": float(expressions["matrix"]),
        "legacy_vector_feed_bits_per_cycle": float(expressions["vector"]),
        "legacy_kv_feed_bits_per_cycle": float(expressions["kv"]),
        "legacy_per_chip_limit_bits_per_cycle": float(per_chip_limit),
        "required_feed_ratio": (
            max(expressions.values()) / per_chip_limit
            if per_chip_limit > 0
            else float("inf")
        ),
    }


def build_area_proxy_inputs(hw: dict[str, int], precision: dict[str, Any], config: DSEConfig) -> dict[str, Any]:
    act_width, act_scale_width = bits_from_width_spec(precision["ACT_WIDTH"], config.mx_scale_width)
    kv_width, kv_scale_width = bits_from_width_spec(precision["KV_WIDTH"], config.mx_scale_width)
    fp_setting = precision["FP_SETTING"]
    weight_spec = profile_weight_spec(precision, config)
    weight_width, weight_scale_width = bits_from_width_spec(weight_spec, config.mx_scale_width)
    weight_parsed = parse_mx_precision(weight_spec, config.mx_scale_width)
    scale_width = int(max(act_scale_width, kv_scale_width, weight_scale_width))

    return {
        "MLEN": hw["MLEN"],
        "BLEN": hw["BLEN"],
        "VLEN": hw["VLEN"],
        "MATRIX_SRAM_DEPTH": hw["MATRIX_SRAM_SIZE"],
        "VECTOR_SRAM_DEPTH": hw["VECTOR_SRAM_SIZE"],
        "INT_SRAM_DEPTH": hw["INT_SRAM_DEPTH"],
        "FP_SRAM_DEPTH": hw["FP_SRAM_DEPTH"],
        "COMPACT_STATS_LANES": hw["COMPACT_STATS_LANES"],
        "SOFTMAX_ROW_LANES": hw.get("SOFTMAX_ROW_LANES", 1),
        "SOFTMAX_STATE_BANK_ENTRIES": hw.get("SOFTMAX_STATE_BANK_ENTRIES", 0),
        "HLEN": hw["HLEN"],
        "INT_DATA_WIDTH": hw["INT_DATA_WIDTH"],
        "ACT_ELEMENT_WIDTH": act_width,
        "KV_ELEMENT_WIDTH": kv_width,
        "FP_EXP_WIDTH": int(fp_setting["exp"]),
        "FP_MANT_WIDTH": int(fp_setting["mant"]),
        "WT_MX_EXP_WIDTH": int(weight_parsed.get("exp", config.weight_mx_exp_width)),
        "WT_MX_MANT_WIDTH": int(weight_parsed.get("mant", config.weight_mx_mant_width)),
        "WEIGHT_ELEMENT_BITS": int(weight_width),
        "MX_SCALE_WIDTH": scale_width,
        "BLOCK_DIM": hw["BLEN"],
        "HBM_ELE_WIDTH": hw["MLEN"],
        "HBM_SCALE_WIDTH": (hw["MLEN"] // hw["BLEN"]) * scale_width,
        "HBM_M_Prefetch_Amount": hw["HBM_M_Prefetch_Amount"],
        "HBM_V_Prefetch_Amount": hw["HBM_V_Prefetch_Amount"],
        "HBM_V_Writeback_Amount": hw["HBM_V_Writeback_Amount"],
        "ACT_WIDTH": precision["ACT_WIDTH"],
        "KV_WIDTH": precision["KV_WIDTH"],
        "WEIGHT_WIDTH": weight_spec,
        "FP_SETTING": f"FP_E{int(fp_setting['exp'])}M{int(fp_setting['mant'])}",
        "SRAM_PORT_MODEL": CURRENT_DSE_PROFILE.sram_port_model,
    }


def effective_mx_bits(width_spec: dict[str, Any], config: DSEConfig) -> float:
    element_bits, scale_width = bits_from_width_spec(width_spec, config.mx_scale_width)
    block_size = int(width_spec.get("block_size", width_spec.get("block", config.mx_scale_block_size)))
    return element_bits + scale_width / block_size


def calculate_batch_info(model: dict[str, Any], precision: dict[str, Any], config: DSEConfig) -> dict[str, Any]:
    weight_effective_bits = profile_weight_effective_bits(precision, config)
    model_weight_bytes = config.weight_param_count * weight_effective_bits / 8
    remaining_hbm_bytes = config.hbm_capacity_bytes - model_weight_bytes
    kv_bits = effective_mx_bits(precision["KV_WIDTH"], config)
    kv_bytes_per_layer_per_request = (
        config.input_seq_len
        * 2
        * model["num_key_value_heads"]
        * model["head_dim"]
        * kv_bits
        / 8
    )
    full_decoder_kv_cache_bytes_per_request = (
        kv_bytes_per_layer_per_request * model["num_hidden_layers"]
    )
    if config.kv_capacity_mode == KV_CAPACITY_MODE_STREAMED_HANDOFF_V1:
        resident_kv_layer_count = 1
        handoff_staging_layer_count = config.kv_handoff_staging_layers
        prefill_kv_capacity_bytes_per_request = (
            kv_bytes_per_layer_per_request
            * (resident_kv_layer_count + handoff_staging_layer_count)
        )
        kv_capacity_semantics = (
            "one_attention_layer_plus_handoff_staging;full_decoder_cache_is_shadow"
        )
    elif config.kv_capacity_mode == KV_CAPACITY_MODE_FULL_CACHE_V1:
        resident_kv_layer_count = int(model["num_hidden_layers"])
        handoff_staging_layer_count = 0
        prefill_kv_capacity_bytes_per_request = (
            full_decoder_kv_cache_bytes_per_request
        )
        kv_capacity_semantics = "full_decoder_kv_cache_resident_on_prefill_hbm"
    else:
        raise ValueError(
            f"unsupported KV capacity mode {config.kv_capacity_mode!r}; "
            f"expected one of {KV_CAPACITY_MODES}"
        )
    hbm_capacity_max_batch = max(
        1,
        math.floor(
            remaining_hbm_bytes / prefill_kv_capacity_bytes_per_request
        ),
    )
    return {
        "batch_size": config.latency_batch_size,
        "latency_batch_size": config.latency_batch_size,
        "hbm_capacity_max_batch": hbm_capacity_max_batch,
        "hbm_capacity_utilization_at_latency_batch": config.latency_batch_size / hbm_capacity_max_batch,
        "input_seq_len": config.input_seq_len,
        "output_seq_len": config.output_seq_len,
        "device_num": config.device_num,
        "hbm_capacity_bytes": config.hbm_capacity_bytes,
        "hbm_bandwidth_gbps": config.hbm_bandwidth_gbps,
        "bandwidth_limit_bits_per_cycle": config.bandwidth_limit_bits_per_cycle,
        "model_param_count": config.weight_param_count,
        "weight_effective_bits": weight_effective_bits,
        "model_weight_bytes": model_weight_bytes,
        "remaining_hbm_bytes": remaining_hbm_bytes,
        "kv_effective_bits": kv_bits,
        # Compatibility alias now follows the explicit prefill capacity
        # semantics. New consumers should use the unambiguous field below.
        "kv_bytes_per_request": prefill_kv_capacity_bytes_per_request,
        "kv_bytes_per_layer_per_request": kv_bytes_per_layer_per_request,
        "prefill_kv_capacity_bytes_per_request": (
            prefill_kv_capacity_bytes_per_request
        ),
        "full_decoder_kv_cache_bytes_per_request": (
            full_decoder_kv_cache_bytes_per_request
        ),
        "kv_capacity_mode": config.kv_capacity_mode,
        "kv_capacity_semantics": kv_capacity_semantics,
        "kv_resident_layer_count": resident_kv_layer_count,
        "kv_handoff_staging_layer_count": handoff_staging_layer_count,
        "full_decoder_kv_cache_capacity_included": (
            config.kv_capacity_mode == KV_CAPACITY_MODE_FULL_CACHE_V1
        ),
        "mx_scale_width": config.mx_scale_width,
        "mx_scale_block_size": config.mx_scale_block_size,
        "batch_policy": (
            "fixed_latency_batch_with_a100_80gb_weight_plus_"
            f"{config.kv_capacity_mode}_capacity_upper_bound"
        ),
    }


def estimate_model_weight_partition(
    model: Mapping[str, Any],
    total_weight_bytes: float,
    *,
    total_parameter_count: float | None = None,
) -> dict[str, float]:
    """Split model weights into shared and expert-resident byte pools."""

    num_experts = int(model.get("num_experts", 0) or 0)
    if num_experts <= 1:
        return {
            "shared_weight_bytes": float(total_weight_bytes),
            "expert_weight_bytes": 0.0,
            "expert_weight_fraction": 0.0,
        }
    mlp_types = tuple(model.get("mlp_types") or ())
    moe_layers = (
        sum(str(value).lower() == "moe" for value in mlp_types)
        if mlp_types
        else int(model["num_hidden_layers"])
    )
    hidden = int(model["hidden_size"])
    expert_intermediate = int(model["moe_intermediate_size"])
    expert_parameters = (
        moe_layers * num_experts * 3 * hidden * expert_intermediate
    )
    total_parameters = float(
        total_parameter_count
        or model.get("parameter_count")
        or model.get("num_parameters")
        or 0.0
    )
    if total_parameters <= 0:
        # Formal DSE supplies an explicit parameter count.  Library callers may
        # not, so retain a conservative architectural estimate instead of
        # incorrectly classifying every parameter as an expert weight.
        layers = int(model["num_hidden_layers"])
        dense_layers = max(0, layers - moe_layers)
        dense_intermediate = int(
            model.get("intermediate_size") or expert_intermediate
        )
        attention_parameters = layers * 4 * hidden * hidden
        dense_parameters = dense_layers * 3 * hidden * dense_intermediate
        router_parameters = moe_layers * hidden * num_experts
        vocabulary_parameters = (
            int(model.get("vocab_size", 0) or 0) * hidden
        )
        total_parameters = float(
            expert_parameters
            + attention_parameters
            + dense_parameters
            + router_parameters
            + vocabulary_parameters
        )
    expert_fraction = min(1.0, expert_parameters / total_parameters)
    expert_bytes = float(total_weight_bytes) * expert_fraction
    return {
        "shared_weight_bytes": float(total_weight_bytes) - expert_bytes,
        "expert_weight_bytes": expert_bytes,
        "expert_weight_fraction": expert_fraction,
    }


def calculate_multichip_hbm_capacity(
    batch_info: Mapping[str, Any],
    *,
    batch_size: int,
    chip_count: int,
    tp_degree: int,
    cp_degree: int,
    max_token_fraction: float,
    per_chip_hbm_capacity_bytes: float,
    factorized_parallel: bool,
    parallel_model: str,
    dp_degree: int = 1,
    ep_degree: int = 1,
    shared_weight_bytes: float | None = None,
    expert_weight_bytes: float = 0.0,
) -> dict[str, Any]:
    """Calculate resident prefill capacity without charging decode-side KV."""

    if min(
        batch_size,
        chip_count,
        tp_degree,
        cp_degree,
        dp_degree,
        ep_degree,
    ) <= 0:
        raise ValueError("batch, chip, and parallel degrees must be positive")
    if not 0.0 < max_token_fraction <= 1.0:
        raise ValueError(
            "max_token_fraction must be in (0, 1], got "
            f"{max_token_fraction}"
        )
    aggregate_weight_bytes = float(batch_info["model_weight_bytes"])
    aggregate_prefill_kv_bytes = (
        float(batch_info["prefill_kv_capacity_bytes_per_request"])
        * batch_size
    )
    full_decoder_kv_cache_bytes_shadow = (
        float(batch_info["full_decoder_kv_cache_bytes_per_request"])
        * batch_size
    )
    if parallel_model == "dp-tp-ep":
        if chip_count != dp_degree * tp_degree * ep_degree:
            raise ValueError("N must equal DP*TP*EP")
        shared_bytes = (
            aggregate_weight_bytes - float(expert_weight_bytes)
            if shared_weight_bytes is None
            else float(shared_weight_bytes)
        )
        expert_bytes = float(expert_weight_bytes)
        if min(shared_bytes, expert_bytes) < 0 or not math.isclose(
            shared_bytes + expert_bytes,
            aggregate_weight_bytes,
            rel_tol=0.0,
            abs_tol=max(1.0, aggregate_weight_bytes * 1e-12),
        ):
            raise ValueError("shared/expert weights must partition model bytes")
        per_chip_required_bytes = (
            shared_bytes / tp_degree
            + expert_bytes / (tp_degree * ep_degree)
            + aggregate_prefill_kv_bytes
            * max_token_fraction
            / tp_degree
        )
        aggregate_required_bytes = (
            shared_bytes * dp_degree * ep_degree
            + expert_bytes * dp_degree
            + aggregate_prefill_kv_bytes
        )
        weight_replication_factor = (
            (shared_bytes * dp_degree * ep_degree + expert_bytes * dp_degree)
            / aggregate_weight_bytes
            if aggregate_weight_bytes
            else 0.0
        )
    elif factorized_parallel:
        per_chip_required_bytes = (
            aggregate_weight_bytes / tp_degree
            + aggregate_prefill_kv_bytes
            * max_token_fraction
            / tp_degree
        )
        aggregate_required_bytes = (
            aggregate_weight_bytes * cp_degree
            + aggregate_prefill_kv_bytes
        )
    else:
        aggregate_required_bytes = (
            aggregate_weight_bytes + aggregate_prefill_kv_bytes
        )
        per_chip_required_bytes = (
            aggregate_required_bytes / chip_count
            if parallel_model == "tp-sp"
            else aggregate_weight_bytes / chip_count
            + aggregate_prefill_kv_bytes
        )
        weight_replication_factor = 1
    return {
        "aggregate_weight_bytes": aggregate_weight_bytes,
        "aggregate_prefill_kv_capacity_bytes": (
            aggregate_prefill_kv_bytes
        ),
        "full_decoder_kv_cache_bytes_shadow": (
            full_decoder_kv_cache_bytes_shadow
        ),
        "aggregate_hbm_required_bytes": aggregate_required_bytes,
        "per_chip_hbm_required_bytes": per_chip_required_bytes,
        "per_chip_hbm_capacity_feasible": (
            per_chip_required_bytes <= per_chip_hbm_capacity_bytes
        ),
        "weight_replication_factor": (
            weight_replication_factor
            if parallel_model == "dp-tp-ep"
            else cp_degree if factorized_parallel else 1
        ),
        "shared_weight_replication": (
            dp_degree * ep_degree
            if parallel_model == "dp-tp-ep"
            else cp_degree if factorized_parallel else 1
        ),
        "expert_weight_replication": (
            dp_degree if parallel_model == "dp-tp-ep" else 1
        ),
    }


def run_area_proxy(hw: dict[str, int], precision: dict[str, Any], config: DSEConfig) -> dict[str, Any]:
    from analytic_models.area import estimate_area

    proxy_inputs = build_area_proxy_inputs(hw, precision, config)
    metrics = estimate_area(proxy_inputs)
    metrics["area_mode"] = "proxy"
    return metrics


def area_extrapolation_warnings(hw: dict[str, int]) -> tuple[list[str], dict[str, float]]:
    ratios = {
        "matrix_mlen": hw["MLEN"] / 64.0,
        "matrix_blen": hw["BLEN"] / 16.0,
        "vector_vlen": hw["VLEN"] / 512.0,
        "hbm_mlen": hw["MLEN"] / 512.0,
        "hbm_vlen": hw["VLEN"] / 512.0,
    }
    warnings = [f"{name} exceeds calibration domain by {ratio:.2f}x" for name, ratio in ratios.items() if ratio > 1.0]
    k_splits = hw["MLEN"] / hw["BLEN"]
    ratios["matrix_k_splits"] = k_splits
    if k_splits < 2.0:
        warnings.append(
            "matrix_k_splits is below the calibrated minimum: "
            f"{k_splits:g} < 2 (BLEN=MLEN topology is exploratory)"
        )
    return warnings, ratios


def sequence_layout_metrics(
    *,
    seq_len: int,
    batch_size: int,
    mlen: int,
    native_layout_mode: str = "compact",
) -> dict[str, Any]:
    """Expose the shared compiler planner's physical sequence geometry."""

    plan = SequencePackingPlan.build(
        batch_size=batch_size,
        seq_len=seq_len,
        mlen=mlen,
        mode=native_layout_mode,
    )
    return {
        "active_sequence_rows": plan.logical_active_rows,
        "physical_sequence_rows": plan.compile_seq_rows,
        "rows_per_batch": plan.rows_per_attention_group,
        "sequence_row_utilization": plan.row_utilization,
        "sequence_padding_factor": plan.compile_seq_rows / plan.logical_active_rows,
        "batch_pack_factor": plan.batch_pack_factor,
        "attention_group_count": plan.attention_group_count,
    }


def compute_fidelity_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize how much RTL-v1 work is measured versus extrapolated.

    CostEmitter already retains the complete opcode-level validation record.
    These scalar fields make fidelity visible in CSVs and ranking reports,
    instead of burying it inside each trial JSON.
    """

    validation = report.get("compute_validation", {})
    status_cycles = validation.get("status_resource_cycles", {})
    total_cycles = sum(float(value) for value in status_cycles.values())

    def fraction(status: str) -> float:
        if total_cycles <= 0.0:
            return 0.0
        return float(status_cycles.get(status, 0.0)) / total_cycles

    return {
        "compute_fidelity_status": validation.get("status", "unknown"),
        "compute_measured_cycle_fraction": fraction("full_machine_measured"),
        "compute_structural_extrapolation_cycle_fraction": fraction(
            "structural_extrapolation"
        ),
        "compute_unsupported_cycle_fraction": fraction("unsupported_rtl"),
    }


def run_area_proxy_v2(
    hw: dict[str, int],
    precision: dict[str, Any],
    config: DSEConfig,
    vector_scalar_schedule: str = "rtl-v5",
    address_generation_mode: str = "loop-agu-v1",
) -> dict[str, Any]:
    from analytic_models.area_new import estimate_area

    proxy_inputs = build_area_proxy_inputs(hw, precision, config)
    proxy_inputs["vector_scalar_area_version"] = vector_scalar_schedule
    proxy_inputs["COMPACT_STATS_LANES"] = (
        hw["COMPACT_STATS_LANES"]
        if vector_scalar_schedule in {"rtl-v5", "rtl-v6"}
        else 16
    )
    proxy_inputs["address_generation_mode"] = address_generation_mode
    metrics = estimate_area(proxy_inputs)
    warnings, ratios = area_extrapolation_warnings(hw)
    warnings = list(
        dict.fromkeys(
            [*metrics.get("area_extrapolation_warnings", []), *warnings]
        )
    )
    metrics.update(
        {
            "area_mode": "proxy-v2",
            "area_um2": float(metrics["area"]),
            "area_mm2": float(metrics["area"]) / 1e6,
            "area_uncertainty_p10_mm2": float(
                metrics.get("area_uncertainty_p10", metrics["area"])
            )
            / 1e6,
            "area_uncertainty_p50_mm2": float(
                metrics.get("area_uncertainty_p50", metrics["area"])
            )
            / 1e6,
            "area_uncertainty_p90_mm2": float(
                metrics.get("area_uncertainty_p90", metrics["area"])
            )
            / 1e6,
            "area_extrapolation_warnings": warnings,
            "area_extrapolation_ratios": ratios,
        }
    )
    return metrics


def load_accuracy(
    path: Path,
    *,
    fallback_weight_precision: str = DEFAULT_WEIGHT_PRECISION,
    min_accuracy: float = 0.9,
) -> list[dict[str, Any]]:
    raw = load_json(path)
    profiles = raw.get("precision_profiles", [])
    if not profiles:
        raise ValueError("accuracy constraints must include at least one precision profile")
    normalized = []
    names: set[str] = set()
    tuples: set[str] = set()
    for idx, profile in enumerate(profiles):
        item = dict(profile)
        item.setdefault("name", f"profile_{idx}")
        item.setdefault("WEIGHT_WIDTH", fallback_weight_precision)
        for required in ("ACT_WIDTH", "KV_WIDTH", "WEIGHT_WIDTH", "FP_SETTING", "accuracy_score"):
            if required not in item:
                raise ValueError(f"precision profile {item['name']} missing {required}")
        score = float(item["accuracy_score"])
        if not math.isfinite(score) or score <= min_accuracy:
            raise ValueError(
                f"precision profile {item['name']} has accuracy_score={score}, expected > {min_accuracy}"
            )
        families = set()
        for role in ("ACT_WIDTH", "KV_WIDTH", "WEIGHT_WIDTH"):
            parsed = parse_mx_precision(item[role])
            if parsed["family"] == "mxint" and int(parsed["width"]) == 3:
                raise ValueError(
                    f"precision profile {item['name']} uses unsupported MXINT3 in {role}"
                )
            if int(parsed["width"]) not in (4, 8):
                raise ValueError(
                    f"precision profile {item['name']} uses unsupported V3 width in {role}: {parsed['width']}"
                )
            families.add(parsed["family"])
        if len(families) != 1:
            raise ValueError(
                f"precision profile {item['name']} mixes MXINT/MXFP families, unsupported by area_new"
            )
        name = str(item["name"])
        if name in names:
            raise ValueError(f"duplicate precision profile name: {name}")
        names.add(name)
        tuple_key = json.dumps(
            {key: item[key] for key in ("ACT_WIDTH", "KV_WIDTH", "WEIGHT_WIDTH", "FP_SETTING")},
            sort_keys=True,
        )
        if tuple_key in tuples:
            raise ValueError(f"duplicate precision tuple in profile {name}")
        tuples.add(tuple_key)
        normalized.append(item)
    return normalized


def write_analytic_toml(path: Path, hw: dict[str, int], config_args: DSEConfig) -> None:
    data = toml.load(BASE_ANALYTIC_TOML)
    config = data.setdefault("ANALYTIC", {}).setdefault("CONFIG", {})
    for key in (
        "MLEN",
        "VLEN",
        "BLEN",
        "HLEN",
        "BROADCAST_AMOUNT",
        "MATRIX_SRAM_SIZE",
        "VECTOR_SRAM_SIZE",
        "INT_SRAM_DEPTH",
        "FP_SRAM_DEPTH",
        "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount",
        "HBM_V_Writeback_Amount",
    ):
        config[key] = {"value": hw[key]}
    config["HBM_SIZE"] = {"value": config_args.hbm_capacity_bytes}
    config["HBM_WIDTH"] = {"value": round(config_args.bandwidth_limit_bits_per_cycle)}
    precision = data.setdefault("ANALYTIC", {}).setdefault("PRECISION", {})
    for key in ("HBM_M_WEIGHT_TYPE", "HBM_M_KV_TYPE", "HBM_V_ACT_TYPE", "HBM_V_KV_TYPE"):
        if key in precision:
            precision[key]["block"] = config_args.mx_scale_block_size
    precision.setdefault("HBM_V_INT_TYPE", {}).setdefault("DATA_TYPE", {})["width"] = hw["INT_DATA_WIDTH"]
    with path.open("w") as f:
        toml.dump(data, f)


def compiler_cost_precision_issue(
    precision: dict[str, Any], config_args: DSEConfig
) -> str | None:
    formats = {
        "weight": parse_mx_precision(
            profile_weight_spec(precision, config_args), config_args.mx_scale_width
        ),
        "activation": parse_mx_precision(precision["ACT_WIDTH"], config_args.mx_scale_width),
        "kv": parse_mx_precision(precision["KV_WIDTH"], config_args.mx_scale_width),
    }
    for role, fmt in formats.items():
        width = int(fmt["width"])
        if fmt["family"] == "mxint" and width == 3:
            return f"MXINT3 is unsupported by Compiler Cost Memory V3 ({role})"
        if width not in (4, 8):
            return f"Compiler Cost Memory V3 supports only 4/8-bit MX formats ({role}={width})"
    if config_args.mx_scale_block_size != 64:
        return f"active V3 DSE requires MX block size 64, got {config_args.mx_scale_block_size}"
    return None


def _mx_toml_section(spec: Any, config: DSEConfig) -> dict[str, Any]:
    parsed = parse_mx_precision(spec, config.mx_scale_width)
    if parsed["family"] == "mxint":
        element = {"type": "Int", "width": int(parsed["width"])}
    else:
        element = {
            "type": "Fp",
            "sign": True,
            "exponent": int(parsed["exp"]),
            "mantissa": int(parsed["mant"]),
        }
    return {
        "format": "Mx",
        "block": config.mx_scale_block_size,
        "ELEM": element,
        "SCALE": {
            "type": "Fp",
            "sign": False,
            "exponent": config.mx_scale_width,
            "mantissa": 0,
        },
    }


def _plain_fp_toml_section(fp_setting: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": "Plain",
        "DATA_TYPE": {
            "type": "Fp",
            "sign": True,
            "exponent": int(fp_setting["exp"]),
            "mantissa": int(fp_setting["mant"]),
        },
    }


def write_compiler_cost_toml(
    template: Path,
    path: Path,
    hw: dict[str, int],
    precision_profile: dict[str, Any],
    config_args: DSEConfig,
    native_layout_mode: str,
) -> None:
    data = toml.load(template)
    try:
        config = data["TRANSACTIONAL"]["CONFIG"]
    except KeyError as exc:
        raise ValueError(f"{template} has no TRANSACTIONAL.CONFIG section") from exc
    group_broadcast = min(
        hw["BROADCAST_AMOUNT"], hw["MLEN"] // hw["HLEN"]
    )
    attention_group_width = group_broadcast * hw["HLEN"]
    groups_per_storage_block = (
        max(1, hw["MLEN"] // attention_group_width)
        if native_layout_mode == "compact"
        else 1
    )
    # Compact storage processes one logical KV group at a time, but M_BMM still
    # consumes an aligned MLEN row.  Broadcasting across every head lane in the
    # shared storage block lets the compiler select the relevant score lanes.
    hardware_broadcast = group_broadcast * groups_per_storage_block
    values = {
        "MLEN": hw["MLEN"],
        "BLEN": hw["BLEN"],
        "VLEN": hw["VLEN"],
        "HLEN": hw["HLEN"],
        "BROADCAST_AMOUNT": hardware_broadcast,
        "MATRIX_SRAM_SIZE": hw["MATRIX_SRAM_SIZE"],
        "VECTOR_SRAM_SIZE": hw["VECTOR_SRAM_SIZE"],
        "FP_SRAM_DEPTH": hw["FP_SRAM_DEPTH"],
        "FP_CONSTANT_NUM": hw["FP_CONSTANT_NUM"],
        "COMPACT_STATS_LANES": hw["COMPACT_STATS_LANES"],
        "HBM_M_Prefetch_Amount": hw["HBM_M_Prefetch_Amount"],
        "HBM_V_Prefetch_Amount": hw["HBM_V_Prefetch_Amount"],
        "HBM_V_Writeback_Amount": hw["HBM_V_Writeback_Amount"],
        "CLOCK_PERIOD_PS": round(1000.0 / config_args.frequency_ghz),
    }
    optional_config_extensions = {
        "CLOCK_PERIOD_PS",
        "FP_SRAM_DEPTH",
        "FP_CONSTANT_NUM",
        "COMPACT_STATS_LANES",
    }
    for name, value in values.items():
        if name not in config and name not in optional_config_extensions:
            raise ValueError(f"transactional settings template is missing CONFIG.{name}")
        config[name] = {"value": int(value)}
    precision = data["TRANSACTIONAL"].setdefault("PRECISION", {})
    weight = _mx_toml_section(
        profile_weight_spec(precision_profile, config_args), config_args
    )
    activation = _mx_toml_section(precision_profile["ACT_WIDTH"], config_args)
    kv = _mx_toml_section(precision_profile["KV_WIDTH"], config_args)
    precision["HBM_M_WEIGHT_TYPE"] = weight
    precision["HBM_M_KV_TYPE"] = kv
    precision["HBM_V_ACT_TYPE"] = activation
    precision["HBM_V_KV_TYPE"] = kv
    internal_fp = _plain_fp_toml_section(precision_profile["FP_SETTING"])
    precision["MATRIX_SRAM_TYPE"] = internal_fp
    precision["VECTOR_SRAM_TYPE"] = internal_fp
    precision["SCALAR_FP"] = dict(internal_fp["DATA_TYPE"])
    precision["HBM_V_INT_TYPE"] = {
        "format": "Plain",
        "DATA_TYPE": {"type": "Int", "width": int(hw["INT_DATA_WIDTH"])},
    }
    with path.open("w") as handle:
        toml.dump(data, handle)


def run_compiler_cost(
    model_config: Path,
    settings_template: Path,
    calibration: Path,
    trial_dir: Path,
    hw: dict[str, int],
    precision: dict[str, Any],
    config_args: DSEConfig,
    compute_timing_mode: str,
    scheduled_shadow: bool,
    v4_memory_evaluation: str,
    native_layout_mode: str,
    packed_attention_schedule: str,
    softmax_state_schedule: str,
    packed_qk_schedule: str,
    vector_scalar_schedule: str,
    softmax_vector_schedule: str,
    pv_accumulation_schedule: str,
    softmax_row_lanes: int,
    selector_schedule: str,
    reduction_output_mode: str,
    gqa_pipeline_schedule: str,
    address_generation_mode: str,
    ffn_address_schedule: str,
    ffn_projection_schedule: str,
    power_shadow: bool,
    clock_gating_mode: str,
    external_memory_energy_artifact: Path,
    matrix_sram_policy: str,
    cost_trace_granularity: str,
    trial_report_materialization: str,
    cache_directories: DSECacheDirectories,
    v4_progress_callback=None,
    moe_routing_mode: str = "static-indices",
    moe_lowering_schedule: str = "compact-route-v2",
    moe_layer_scaling: str = "single-layer",
) -> dict[str, Any]:
    compiler_root = REPO_ROOT / "PLENA_Compiler"
    tools_root = REPO_ROOT / "PLENA_Tools"
    for dependency_root in (compiler_root, tools_root):
        if str(dependency_root) not in sys.path:
            sys.path.insert(0, str(dependency_root))
    from analytic_models.performance.compiler_cost_model import (
        compile_and_evaluate_compiler_cost,
    )

    settings_path = trial_dir / "compiler_cost_settings.toml"
    write_compiler_cost_toml(
        settings_template,
        settings_path,
        hw,
        precision,
        config_args,
        native_layout_mode,
    )
    trace = None
    report = None
    try:
        model_payload = load_json(model_config)
        trace, report = compile_and_evaluate_compiler_cost(
            model_config,
            settings_path,
            calibration,
            seq_len=config_args.input_seq_len,
            batch_size=config_args.latency_batch_size,
            num_layers=int(model_payload.get("num_hidden_layers", 1)),
            moe_routing_mode=moe_routing_mode,
            moe_lowering_schedule=moe_lowering_schedule,
            moe_layer_scaling=moe_layer_scaling,
            precision_config={
                "weight": profile_weight_spec(precision, config_args),
                "activation": precision["ACT_WIDTH"],
                "kv": precision["KV_WIDTH"],
                "block": config_args.mx_scale_block_size,
                "scale_bits": config_args.mx_scale_width,
                "integer_bits": hw["INT_DATA_WIDTH"],
                "internal_fp": precision["FP_SETTING"],
            },
            compute_timing_mode=compute_timing_mode,
            scheduled_shadow=scheduled_shadow,
            v4_memory_evaluation=v4_memory_evaluation,
            native_layout_mode=native_layout_mode,
            packed_attention_schedule=packed_attention_schedule,
            softmax_state_schedule=softmax_state_schedule,
            packed_qk_schedule=packed_qk_schedule,
            vector_scalar_schedule=vector_scalar_schedule,
            softmax_vector_schedule=softmax_vector_schedule,
            pv_accumulation_schedule=pv_accumulation_schedule,
            softmax_row_lanes=softmax_row_lanes,
            selector_schedule=selector_schedule,
            reduction_output_mode=reduction_output_mode,
            gqa_pipeline_schedule=gqa_pipeline_schedule,
            address_generation_mode=address_generation_mode,
            ffn_address_schedule=ffn_address_schedule,
            ffn_projection_schedule=ffn_projection_schedule,
            cost_trace_granularity=cost_trace_granularity,
            persistent_trace_cache_dir=cache_directories.compiler_traces,
            persistent_v4_work_cache_dir=cache_directories.compiler_v4_work,
            persistent_compute_pipeline_cache_dir=(
                cache_directories.compiler_compute_pipeline
            ),
            kv_residency_policy=matrix_sram_policy,
            v4_aggregation_backend="sufficient-statistics-v2",
            v4_progress_callback=v4_progress_callback,
            v4_geometry_batch_size=4096,
        )
        result = report.to_dict()
        result["trace"] = {
            "schema_version": trace.schema_version,
            "static_machine_instructions": sum(trace.static_opcodes.values()),
            "dynamic_machine_instructions": sum(trace.dynamic_opcodes.values()),
            "dynamic_opcodes": dict(sorted(trace.dynamic_opcodes.items())),
            "one_layer_dynamic_opcodes": dict(
                sorted(
                    (trace.metadata.get("one_layer_dynamic_opcodes") or {}).items()
                )
            ),
            "memory_stream_count": len(trace.memory_events),
            "compressed_memory_events": [
                event.to_dict() for event in trace.memory_events
            ],
            "dma_coverage": trace.metadata.get("dma_coverage"),
            "hardware": trace.metadata.get("hardware"),
            "native_layout": trace.metadata.get("native_layout"),
            "attention_schedule": trace.metadata.get("attention_schedule"),
            "packed_attention": trace.metadata.get("packed_attention"),
            "vector_scalar_optimization": trace.metadata.get(
                "vector_scalar_optimization"
            ),
            "workload": trace.metadata.get("workload"),
            "parallel_kernel_census": [
                entry.to_dict() for entry in trace.parallel_kernel_census
            ],
            "parallel_kernel_census_schema": trace.metadata.get(
                "parallel_kernel_census_schema"
            ),
            "parallel_kernel_census_coverage": trace.metadata.get(
                "parallel_kernel_census_coverage"
            ),
            "energy_action_lineage": trace.metadata.get(
                "energy_action_lineage"
            ),
            "compiler_metadata": {
                key: trace.metadata.get(key)
                for key in (
                    "address_generation_mode",
                    "ffn_address_schedule",
                    "ffn_projection_schedule",
                    "ffn_address_optimization",
                    "agu_residual_s_addi",
                    "broadcast_timing_model",
                    "broadcast_rtl_validated",
                    "broadcast_rtl_validation_status",
                    "cost_trace_granularity",
                    "compute_trace_fidelity",
                    "ordered_schedule_available",
                    "block_class_count",
                    "expanded_block_pair_equivalent",
                    "materialized_block_pair_count",
                    "moe_routing_mode",
                    "routing_plan_hash",
                    "moe_lowering_schedule",
                    "route_count",
                    "active_expert_ids",
                    "routes_per_expert",
                )
            },
            "persistent_trace_cache_hit": trace.metadata.get(
                "persistent_trace_cache_hit", False
            ),
            "persistent_trace_cache_key": trace.metadata.get(
                "persistent_trace_cache_key"
            ),
        }
        # Multi-chip energy partitioning needs the compressed action census,
        # not the full schedule tree.  Keeping only EnergyAction records makes
        # the payload small while preserving component, stage, precision, and
        # active-lane semantics.
        result["power_inputs"] = {
            "schema_version": trace.schema_version,
            "metadata": {
                "hardware": trace.metadata.get("hardware"),
                "native_layout": trace.metadata.get("native_layout"),
                "attention_schedule": trace.metadata.get(
                    "attention_schedule"
                ),
                "energy_action_schema": trace.metadata.get(
                    "energy_action_schema"
                ),
                "energy_action_lineage": trace.metadata.get(
                    "energy_action_lineage"
                ),
            },
            "energy_actions": [
                action.to_dict() for action in trace.energy_actions
            ],
        }
        if power_shadow:
            # Formal energy depends on chip count and the R-aware memory
            # partition, which are applied after CostEmitter returns.  Keep
            # only the compact action census here and evaluate system energy
            # exactly once in the objective.
            result["power_shadow"] = {
                "status": "deferred_multichip_partition",
                "power_model": "plena_multichip_system_energy_v1",
            }
        else:
            result["power_shadow"] = {
                "status": "disabled",
                "power_model": "plena_system_power_hbm3e_v1",
            }
        if trial_report_materialization != "full":
            settings_blob = settings_path.read_bytes()
            settings_hash = hashlib.sha256(settings_blob).hexdigest()
            settings_cache = cache_directories.compiler_settings
            settings_cache.mkdir(parents=True, exist_ok=True)
            shared_settings = settings_cache / f"{settings_hash}.toml"
            if not shared_settings.exists():
                shared_settings.write_bytes(settings_blob)
            result["compiler_settings_reference"] = {
                "sha256": settings_hash,
                "path": str(shared_settings),
            }
            settings_path.unlink(missing_ok=True)
        if trial_report_materialization == "full":
            write_json(trial_dir / "compiler_cost_report.json", result)
        return result
    finally:
        # CostTrace contains the full compressed schedule and DMA geometry.  A
        # process-local 64-entry frontend cache grew to 5-7 GiB per DSE worker
        # because Optuna assigns many hardware shapes to every process.  DSE
        # trials rarely reuse the immediately preceding shape, so release the
        # cache here while retaining CostEmitter's normal cache for other APIs.
        trace = None
        report = None
        try:
            from aten.cost_frontend import clear_cost_trace_cache
            from aten.isa_builder import parse_legacy_asm
            from analytic_models.performance.compiler_cost_model import (
                clear_v4_work_cache,
            )
            from analytic_models.performance.hbm_service_model import (
                clear_physical_memory_work_cache,
            )

            if cost_trace_granularity == "detailed":
                clear_cost_trace_cache()
                clear_v4_work_cache()
                clear_physical_memory_work_cache()
                parse_legacy_asm.cache_clear()
        finally:
            gc.collect()
            if sys.platform.startswith("linux"):
                try:
                    ctypes.CDLL(None).malloc_trim(0)
                except (AttributeError, OSError):
                    pass


def compiler_layout_record_fields(
    compiler_cost_report: dict[str, Any],
) -> dict[str, Any]:
    """Flatten shared native-layout metadata into stable trial columns."""

    trace_metadata = compiler_cost_report.get("trace", {})
    compiler_hardware = trace_metadata.get("hardware") or {}
    native_layout = trace_metadata.get("native_layout") or {}
    attention_layout = trace_metadata.get("attention_schedule") or {}
    packed_attention = trace_metadata.get("packed_attention") or {}
    vector_scalar = trace_metadata.get("vector_scalar_optimization") or {}
    compiler_metadata = trace_metadata.get("compiler_metadata") or {}
    if not native_layout:
        return {}
    head_layout = native_layout.get("head_packing", {})
    group_broadcast = attention_layout.get(
        "group_broadcast", head_layout.get("physical_broadcast_amount")
    )
    hardware_broadcast = attention_layout.get(
        "hardware_broadcast",
        head_layout.get(
            "hardware_broadcast_amount",
            attention_layout.get("physical_broadcast"),
        ),
    )
    execution_lane_utilization = head_layout.get(
        "execution_head_lane_utilization"
    )
    if (
        execution_lane_utilization is None
        and group_broadcast is not None
        and hardware_broadcast
    ):
        execution_lane_utilization = float(group_broadcast) / float(
            hardware_broadcast
        )
    return {
        "native_layout_schema_version": native_layout.get("schema_version"),
        "native_layout_mode": native_layout.get("mode"),
        "logical_token_rows": native_layout.get("logical_active_rows"),
        "physical_token_rows": native_layout.get("physical_rows"),
        "active_sequence_rows": native_layout.get("logical_active_rows"),
        "physical_sequence_rows": native_layout.get("physical_rows"),
        "sequence_row_utilization": native_layout.get("row_utilization"),
        "sequence_padding_factor": (
            None
            if not native_layout.get("logical_active_rows")
            else float(native_layout.get("physical_rows"))
            / float(native_layout.get("logical_active_rows"))
        ),
        "batch_pack_factor": native_layout.get("batch_pack_factor"),
        "rows_per_attention_group": native_layout.get(
            "rows_per_attention_group"
        ),
        "rows_per_batch": native_layout.get("rows_per_attention_group"),
        "attention_mask_kind": native_layout.get("mask_kind"),
        "logical_q_width": attention_layout.get("logical_q_width"),
        "physical_q_width": head_layout.get(
            "total_q_dim", attention_layout.get("physical_q_width")
        ),
        "head_lane_utilization": head_layout.get(
            "head_lane_utilization",
            attention_layout.get("head_lane_utilization"),
        ),
        "attention_execution_lane_utilization": execution_lane_utilization,
        "attention_group_count": native_layout.get("attention_group_count"),
        "compiler_mram_tile_capacity": compiler_hardware.get(
            "mram_tile_capacity"
        ),
        "attention_storage_block_count": head_layout.get("storage_block_count"),
        "attention_groups_per_storage_block": head_layout.get(
            "groups_per_storage_block"
        ),
        "attention_group_broadcast": group_broadcast,
        "attention_hardware_broadcast": hardware_broadcast,
        "attention_schedule_layout": attention_layout,
        "packed_attention_schedule": packed_attention.get("packed_attention_schedule"),
        "softmax_state_schedule": packed_attention.get(
            "softmax_state_schedule"
        ),
        "packed_qk_schedule": packed_attention.get("packed_qk_schedule"),
        "gqa_pipeline_schedule": packed_attention.get("gqa_pipeline_schedule"),
        "softmax_first_block_pipeline_width": packed_attention.get(
            "softmax_first_block_pipeline_width"
        ),
        "softmax_recurrent_pipeline_width": packed_attention.get(
            "softmax_recurrent_pipeline_width"
        ),
        "o_scale_pipeline_width": packed_attention.get("o_scale_pipeline_width"),
        "o_shift_ring_width": packed_attention.get("o_shift_ring_width"),
        "interleaved_softmax_rows": packed_attention.get(
            "interleaved_softmax_rows"
        ),
        "interleaved_o_rows": packed_attention.get("interleaved_o_rows"),
        "gqa_kv_double_buffered": packed_attention.get(
            "gqa_kv_double_buffered"
        ),
        "gqa_dma_overlap_eligible_occurrences": packed_attention.get(
            "gqa_dma_overlap_eligible_occurrences"
        ),
        "gqa_pipeline_fallback_reason": packed_attention.get(
            "gqa_pipeline_fallback_reason"
        ),
        "softmax_first_block_specialized_count": packed_attention.get(
            "softmax_first_block_specialized_count"
        ),
        "softmax_state_initializations_elided": packed_attention.get(
            "softmax_state_initializations_elided"
        ),
        "softmax_m_moves_elided": packed_attention.get(
            "softmax_m_moves_elided"
        ),
        "softmax_l_moves_elided": packed_attention.get(
            "softmax_l_moves_elided"
        ),
        "softmax_m_stores_elided": packed_attention.get(
            "softmax_m_stores_elided"
        ),
        "m_res_stores_elided": packed_attention.get("m_res_stores_elided"),
        "m_res_loads_elided": packed_attention.get("m_res_loads_elided"),
        "m_res_streamed_rows": packed_attention.get("m_res_streamed_rows"),
        "softmax_state_entries_required": packed_attention.get(
            "softmax_state_entries_required"
        ),
        "temporary_o_matrices_elided": packed_attention.get(
            "temporary_o_matrices_elided"
        ),
        "direct_o_lane_updates": packed_attention.get("direct_o_lane_updates"),
        "qk_compute_count": packed_attention.get("qk_compute_count"),
        "ideal_qk_compute_count": packed_attention.get(
            "ideal_qk_compute_count"
        ),
        "pv_compute_count": packed_attention.get("pv_compute_count"),
        "qk_recompute_factor": packed_attention.get("qk_recompute_factor"),
        "qk_broadcast_reuse_factor": packed_attention.get(
            "qk_broadcast_reuse_factor"
        ),
        "kv_reload_factor": packed_attention.get("kv_reload_factor"),
        "kv_tile_load_count": packed_attention.get("kv_tile_load_count"),
        "ideal_kv_tile_load_count": packed_attention.get(
            "ideal_kv_tile_load_count"
        ),
        "requested_kv_residency_fraction": packed_attention.get(
            "requested_kv_residency_fraction"
        ),
        "realized_kv_residency_fraction": packed_attention.get(
            "realized_kv_residency_fraction"
        ),
        "resident_kv_blocks": packed_attention.get("resident_kv_blocks"),
        "streamed_kv_blocks": packed_attention.get("streamed_kv_blocks"),
        "resident_stream_slot_tiles": packed_attention.get(
            "resident_stream_slot_tiles"
        ),
        "peak_live_tiles": packed_attention.get("peak_live_tiles"),
        "average_live_tiles": packed_attention.get("average_live_tiles"),
        "tile_utilization": packed_attention.get("tile_utilization"),
        "kv_cache_hits": packed_attention.get("kv_cache_hits"),
        "kv_cache_misses": packed_attention.get("kv_cache_misses"),
        "kv_cache_fidelity": packed_attention.get("kv_cache_fidelity"),
        "attention_kv_resident": attention_layout.get("kv_resident"),
        "attention_resident_kv_tiles": attention_layout.get(
            "resident_kv_tiles"
        ),
        "full_q_tiles": packed_attention.get("full_q_tiles"),
        "q_tail_rows": packed_attention.get("q_tail_rows"),
        "q_tail_utilization": packed_attention.get("q_tail_utilization"),
        "tail_bmm_occurrences": packed_attention.get(
            "tail_bmm_occurrences"
        ),
        "tail_full_width_work_cycles": packed_attention.get(
            "tail_full_width_work_cycles"
        ),
        "tail_isa_limitation": packed_attention.get("tail_isa_limitation"),
        "softmax_state_heads": packed_attention.get("softmax_state_heads"),
        "scalar_fp_sram_depth": packed_attention.get("scalar_fp_sram_depth"),
        "scalar_fp_sram_state_utilization": packed_attention.get(
            "scalar_fp_sram_state_utilization"
        ),
        "broadcast_timing_model": compiler_metadata.get(
            "broadcast_timing_model"
        ),
        "broadcast_rtl_validated": compiler_metadata.get(
            "broadcast_rtl_validated"
        ),
        "broadcast_rtl_validation_status": compiler_metadata.get(
            "broadcast_rtl_validation_status"
        ),
        "address_generation_mode": compiler_metadata.get(
            "address_generation_mode"
        ),
        "ffn_address_schedule": compiler_metadata.get("ffn_address_schedule"),
        "ffn_projection_schedule": compiler_metadata.get(
            "ffn_projection_schedule"
        ),
        "ffn_loop_plan_version": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_loop_plan_version"),
        "ffn_explicit_loop_depth": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_explicit_loop_depth"),
        "ffn_agu_streams_by_axis": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_agu_streams_by_axis"),
        "ffn_address_cycles_before": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_address_cycles_before"),
        "ffn_address_cycles_after": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_address_cycles_after"),
        "ffn_schedule_guard_status": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_schedule_guard_status"),
        "ffn_schedule_fallback_reason": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_schedule_fallback_reason"),
        "ffn_legacy_template_bypassed": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_legacy_template_bypassed"),
        "ffn_dead_k_pointer_updates_elided": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_dead_k_pointer_updates_elided"),
        "ffn_dead_prefetch_updates_elided": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_dead_prefetch_updates_elided"),
        "ffn_dead_output_updates_elided": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_dead_output_updates_elided"),
        "ffn_invariant_stride_loads": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_invariant_stride_loads"),
        "ffn_large_immediate_chunks_avoided": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_large_immediate_chunks_avoided"),
        "ffn_residual_address_opcodes": (
            compiler_metadata.get("ffn_address_optimization") or {}
        ).get("ffn_residual_address_opcodes"),
        "agu_residual_s_addi": compiler_metadata.get("agu_residual_s_addi"),
        "packed_attention_metadata": packed_attention,
        "vector_scalar_schedule": vector_scalar.get("vector_scalar_schedule"),
        "segmented_norm_square_ops_elided": vector_scalar.get(
            "segmented_norm_square_ops_elided"
        ),
        "segmented_norm_copy_ops_elided": vector_scalar.get(
            "segmented_norm_copy_ops_elided"
        ),
        "segmented_norm_constant_loads_elided": vector_scalar.get(
            "segmented_norm_constant_loads_elided"
        ),
        "inactive_norm_rows_elided": vector_scalar.get(
            "inactive_norm_rows_elided"
        ),
        "redundant_valid_masks_elided": vector_scalar.get(
            "redundant_valid_masks_elided"
        ),
        "valid_mask_build_count": vector_scalar.get("valid_mask_build_count"),
        "valid_mask_scope": vector_scalar.get("valid_mask_scope"),
        "rms_norm_address_loads_elided": vector_scalar.get(
            "rms_norm_address_loads_elided"
        ),
        "rms_norm_nops_elided": vector_scalar.get("rms_norm_nops_elided"),
        "vector_scalar_optimization_metadata": vector_scalar,
    }


def power_record_fields(power: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten stable system-energy fields while retaining the full report."""

    return {
        "power_status": power.get("status", "missing"),
        "power_model": power.get("power_model"),
        "power_scope": power.get("power_scope"),
        "power_calibration_status": power.get("calibration_status"),
        "multi_chip_energy_partition_fidelity": power.get(
            "multi_chip_energy_partition_fidelity"
        ),
        "multi_chip_unique_rank_power_workloads": power.get(
            "multi_chip_unique_rank_power_workloads"
        ),
        "onchip_energy_mj": power.get("onchip_energy_mj"),
        "onchip_average_power_w": power.get("onchip_average_power_w"),
        "sram_dynamic_energy_mj": power.get("sram_dynamic_energy_mj"),
        "sram_background_energy_mj": power.get(
            "sram_background_energy_mj"
        ),
        "sram_background_power_w": power.get("sram_background_power_w"),
        "sram_allocated_capacity_gb": power.get(
            "sram_allocated_capacity_gb"
        ),
        "sram_background_power_w_per_gb": power.get(
            "sram_background_power_w_per_gb"
        ),
        "sram_background_calibration_status": power.get(
            "sram_background_calibration_status"
        ),
        "asap7_sram_leakage_status": power.get(
            "asap7_sram_leakage_status"
        ),
        "clock_gating_mode": power.get("clock_gating_mode"),
        "clock_gating_status": power.get("clock_gating_status"),
        "clock_energy_mj": power.get("clock_energy_mj"),
        "ungated_clock_energy_mj": power.get("ungated_clock_energy_mj"),
        "ungated_onchip_energy_mj": power.get("ungated_onchip_energy_mj"),
        "ungated_onchip_average_power_w": power.get(
            "ungated_onchip_average_power_w"
        ),
        "clock_energy_savings_pct": power.get("clock_energy_savings_pct"),
        "clock_active_fraction_by_component": power.get(
            "clock_active_fraction_by_component"
        ),
        "unmodeled_clock_residual_area_um2": power.get(
            "unmodeled_clock_residual_area_um2"
        ),
        "external_memory_model": power.get("external_memory_model"),
        "external_memory_calibration_status": power.get(
            "external_memory_calibration_status"
        ),
        "external_memory_configuration_semantics": power.get(
            "external_memory_configuration_semantics"
        ),
        "external_hbm_capacity_bytes": power.get(
            "external_hbm_capacity_bytes"
        ),
        "external_hbm_configured_bandwidth_gbps": power.get(
            "external_hbm_configured_bandwidth_gbps"
        ),
        "hbm_background_energy_mj": power.get("hbm_background_energy_mj"),
        "hbm_read_energy_mj": power.get("hbm_read_energy_mj"),
        "hbm_write_energy_mj": power.get("hbm_write_energy_mj"),
        "external_hbm_energy_mj": power.get("external_hbm_energy_mj"),
        "external_hbm_energy_p10_mj": power.get(
            "external_hbm_energy_p10_mj"
        ),
        "external_hbm_energy_p50_mj": power.get(
            "external_hbm_energy_p50_mj"
        ),
        "external_hbm_energy_p90_mj": power.get(
            "external_hbm_energy_p90_mj"
        ),
        "external_hbm_average_power_w": power.get(
            "external_hbm_average_power_w"
        ),
        "hbm_physical_read_bytes": power.get("hbm_physical_read_bytes"),
        "hbm_physical_write_bytes": power.get("hbm_physical_write_bytes"),
        "hbm_payload_read_bytes": power.get("hbm_payload_read_bytes"),
        "hbm_payload_write_bytes": power.get("hbm_payload_write_bytes"),
        "physical_to_payload_traffic_ratio": power.get(
            "physical_to_payload_traffic_ratio"
        ),
        "achieved_average_bandwidth_gbps": power.get(
            "achieved_average_bandwidth_gbps"
        ),
        "bandwidth_utilization": power.get("bandwidth_utilization"),
        "external_hbm_energy_by_role": power.get(
            "external_hbm_energy_by_role"
        ),
        "external_hbm_energy_by_stage": power.get(
            "external_hbm_energy_by_stage"
        ),
        "external_hbm_energy_by_opcode": power.get(
            "external_hbm_energy_by_opcode"
        ),
        "interconnect_aggregate_bytes": power.get(
            "interconnect_aggregate_bytes"
        ),
        "interconnect_dynamic_energy_mj": power.get(
            "interconnect_dynamic_energy_mj"
        ),
        "interconnect_dynamic_energy_optimistic_c2c_mj": power.get(
            "interconnect_dynamic_energy_optimistic_c2c_mj"
        ),
        "interconnect_dynamic_energy_conservative_measured_path_mj": power.get(
            "interconnect_dynamic_energy_conservative_measured_path_mj"
        ),
        "interconnect_nominal_pj_per_bit": power.get(
            "interconnect_nominal_pj_per_bit"
        ),
        "system_energy_mj": power.get("system_energy_mj"),
        "system_energy_nominal_mj": power.get("system_energy_nominal_mj"),
        "system_energy_p10_mj": power.get("system_energy_p10_mj"),
        "system_energy_p50_mj": power.get("system_energy_p50_mj"),
        "system_energy_p90_mj": power.get("system_energy_p90_mj"),
        "system_energy_optimistic_c2c_mj": power.get(
            "system_energy_optimistic_c2c_mj"
        ),
        "system_energy_conservative_measured_path_mj": power.get(
            "system_energy_conservative_measured_path_mj"
        ),
        "system_average_power_w": power.get("system_average_power_w"),
        "system_average_power_p10_w": power.get(
            "system_average_power_p10_w"
        ),
        "system_average_power_p50_w": power.get(
            "system_average_power_p50_w"
        ),
        "system_average_power_p90_w": power.get(
            "system_average_power_p90_w"
        ),
        "system_energy_per_input_token_mj": power.get(
            "system_energy_per_input_token_mj"
        ),
        "power_uncertainty": power.get(
            "system_uncertainty_power_w",
            power.get("uncertainty_power_w"),
        ),
        "power_warnings": power.get("warnings", []),
        "power_excludes": power.get("excludes", []),
        "power_shadow": dict(power),
    }


def planned_layout_record_fields(
    record: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    seq_len: int,
    batch_size: int,
    native_layout_mode: str,
) -> dict[str, Any]:
    """Reconstruct layout metadata for old or pre-compiler trial records.

    The exhaustive run may span a schema transition: objective values remain
    valid, but an early ``compiler_cost_report.json`` can predate native-layout
    metadata.  Reusing the executable planners here avoids either dropping
    those trials or maintaining a third copy of the layout arithmetic.
    """

    mlen = int(record["MLEN"])
    hlen = int(record.get("HLEN", model["head_dim"]))
    num_heads = int(model["num_attention_heads"])
    num_kv_heads = int(model["num_key_value_heads"])
    gqa_ratio = num_heads // num_kv_heads
    logical_broadcast = int(record.get("BROADCAST_AMOUNT", gqa_ratio))
    sequence = SequencePackingPlan.build(
        batch_size=batch_size,
        seq_len=seq_len,
        mlen=mlen,
        mode=native_layout_mode,
    )
    heads = build_attention_head_packing(
        mlen=mlen,
        hlen=hlen,
        head_dim=int(model["head_dim"]),
        logical_broadcast_amount=logical_broadcast,
        gqa_ratio=gqa_ratio,
        num_kv_heads=num_kv_heads,
        mode=native_layout_mode,
    )
    return {
        "native_layout_schema_version": sequence.metadata()["schema_version"],
        "native_layout_mode": native_layout_mode,
        "logical_token_rows": sequence.logical_active_rows,
        "physical_token_rows": sequence.compile_seq_rows,
        "active_sequence_rows": sequence.logical_active_rows,
        "physical_sequence_rows": sequence.compile_seq_rows,
        "sequence_row_utilization": sequence.row_utilization,
        "sequence_padding_factor": (
            sequence.compile_seq_rows / sequence.logical_active_rows
        ),
        "batch_pack_factor": sequence.batch_pack_factor,
        "rows_per_attention_group": sequence.rows_per_attention_group,
        "rows_per_batch": sequence.rows_per_attention_group,
        "attention_mask_kind": sequence.mask_kind,
        "logical_q_width": num_heads * int(model["head_dim"]),
        "physical_q_width": heads.total_q_dim,
        "head_lane_utilization": heads.head_lane_utilization,
        "attention_execution_lane_utilization": (
            heads.execution_head_lane_utilization
        ),
        "attention_group_count": sequence.attention_group_count,
        "attention_storage_block_count": heads.storage_block_count,
        "attention_groups_per_storage_block": heads.groups_per_storage_block,
        "attention_group_broadcast": heads.broadcast_amount,
        "attention_hardware_broadcast": heads.hardware_broadcast_amount,
    }


def run_latency(
    model_path: Path,
    analytic_toml: Path,
    trial_dir: Path,
    batch_info: dict[str, Any],
    config: DSEConfig,
) -> tuple[float, dict[str, Any]]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "analytic_models/performance/qwen3_model.py"),
        "--model-path",
        str(model_path),
        "--config",
        str(analytic_toml),
        "--isa-lib",
        str(ISA_LIB),
        "--batch-size",
        str(batch_info["batch_size"]),
        "--input-seq",
        str(config.input_seq_len),
        "--output-seq",
        str(config.output_seq_len),
        "--device-num",
        str(config.device_num),
        "--phase",
        "prefill",
        "--quiet",
        "--json",
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    (trial_dir / "latency_stdout.json").write_text(completed.stdout)
    if completed.stderr:
        (trial_dir / "latency_stderr.log").write_text(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"latency command failed with {completed.returncode}")
    report = json.loads(completed.stdout)
    return float(report["prefill_ms"]), report


def shell_pairs(params: dict[str, int]) -> str:
    return " ".join(f"{key}={int(value)}" for key, value in sorted(params.items()))


def run_rtl_config(hw: dict[str, int], precision: dict[str, Any] | None = None, config: DSEConfig | None = None) -> dict[str, Any]:
    config_params = {
        "MLEN": hw["MLEN"],
        "BLEN": hw["BLEN"],
        "VLEN": hw["VLEN"],
        "HLEN": hw["HLEN"],
        "BROADCAST_AMOUNT": hw["BROADCAST_AMOUNT"],
        "MATRIX_SRAM_SIZE": hw["MATRIX_SRAM_SIZE"],
        "VECTOR_SRAM_SIZE": hw["VECTOR_SRAM_SIZE"],
        "HBM_M_Prefetch_Amount": hw["HBM_M_Prefetch_Amount"],
        "HBM_V_Prefetch_Amount": hw["HBM_V_Prefetch_Amount"],
        "HBM_V_Writeback_Amount": hw["HBM_V_Writeback_Amount"],
    }
    precision_params = {"INT_DATA_WIDTH": hw["INT_DATA_WIDTH"]}
    if precision is not None and config is not None:
        precision_params = rtl_precision_params(hw, precision, config)
    config_pairs = (
        shell_pairs(config_params)
    )
    precision_pairs = shell_pairs(precision_params)
    cmd = [
        "nix",
        "develop",
        "-c",
        "bash",
        "-lc",
        f"python src/definitions/config.py --config {config_pairs!r} --precision {precision_pairs!r} --mode ASIC",
    ]
    subprocess.run(cmd, cwd=RTL_ROOT, check=True)
    return {
        "rtl_config_params": config_params,
        "rtl_precision_params": precision_params,
    }


def run_area_synth() -> None:
    cmd = ["nix", "develop", "-c", "bash", "-lc", "just synth plena 1000 area"]
    subprocess.run(cmd, cwd=RTL_ROOT, check=True)


def run_rtl_elaborate() -> None:
    cmd = ["nix", "develop", "-c", "bash", "-lc", "just elaborate plena"]
    subprocess.run(cmd, cwd=RTL_ROOT, check=True)


def parse_area_power() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if AREA_REPORT.exists():
        text = AREA_REPORT.read_text(errors="replace")
        match = re.search(r"Total cell area:\s*([0-9.]+)", text)
        if match:
            metrics["area"] = float(match.group(1))
    if POWER_REPORT.exists():
        text = POWER_REPORT.read_text(errors="replace")
        top = re.search(r"^plena\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", text, re.MULTILINE)
        if top:
            metrics["switch_power_mw"] = float(top.group(1))
            metrics["internal_power_mw"] = float(top.group(2))
            metrics["leakage_power_pw"] = float(top.group(3))
            metrics["total_power_mw"] = float(top.group(4))
    return metrics


def parse_elaborate_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {"area_mode": "elaborate"}
    if ELAB_AREA_REPORT.exists():
        text = ELAB_AREA_REPORT.read_text(errors="replace")
        match = re.search(r"Total cell area:\s*([0-9.]+)", text)
        if match:
            metrics["area"] = float(match.group(1))
            metrics["generic_area"] = float(match.group(1))
    if ELAB_SUMMARY_REPORT.exists():
        text = ELAB_SUMMARY_REPORT.read_text(errors="replace")
        match = re.search(r"Elapsed seconds:\s*([0-9.]+)", text)
        if match:
            metrics["elaborate_elapsed_seconds"] = float(match.group(1))
    metrics.setdefault("area", 0.0)
    metrics["elaborate_area_report"] = str(ELAB_AREA_REPORT)
    metrics["elaborate_summary_report"] = str(ELAB_SUMMARY_REPORT)
    metrics["area_note"] = "generic elaborate area feature, not mapped 7nm synthesis area"
    return metrics


def snapshot_rtl_files() -> dict[Path, str]:
    paths = [
        RTL_ROOT / "src/definitions/plena_settings.toml",
        RTL_ROOT / "src/definitions/configuration.svh",
        RTL_ROOT / "src/definitions/precision.svh",
    ]
    return {path: path.read_text() for path in paths if path.exists()}


def restore_rtl_files(snapshot: dict[Path, str]) -> None:
    for path, content in snapshot.items():
        path.write_text(content)


def copy_rtl_reports(trial_dir: Path) -> None:
    out = trial_dir / "rtl_reports"
    out.mkdir(exist_ok=True)
    latest = RTL_ROOT / "build/synth/plena/latest"
    for rel in [
        "reports/plena_area.rpt",
        "reports/plena_power.rpt",
        "logs/summary.log",
        "logs/area.log",
        "logs/power.log",
    ]:
        src = latest / rel
        if src.exists():
            dst = out / rel.replace("/", "_")
            shutil.copy2(src, dst)


def copy_elaborate_reports(trial_dir: Path) -> None:
    out = trial_dir / "rtl_elaborate_reports"
    out.mkdir(exist_ok=True)
    latest = RTL_ROOT / "build/elab/plena/latest"
    for rel in [
        "reports/plena_generic_area.rpt",
        "reports/plena_reference.rpt",
        "reports/plena_resources.rpt",
        "reports/plena_design.rpt",
        "reports/plena_port.rpt",
        "logs/summary.log",
        "logs/elaborate.log",
    ]:
        src = latest / rel
        if src.exists():
            dst = out / rel.replace("/", "_")
            shutil.copy2(src, dst)


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")


def summarize_worker_resources(
    path: Path,
    *,
    requested_workers: int | None = None,
) -> dict[str, Any]:
    entries = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not entries:
        return {
            "recorded_attempts": 0,
            "worker_launches": 0,
            "worker_recycles": 0,
            "rss_triggered_recycles": 0,
            "memory_triggered_recycles": 0,
            "peak_worker_rss_gib": None,
            "ask_wall_time_seconds": 0.0,
            "evaluation_wall_time_seconds": 0.0,
        }
    attempts = [
        entry
        for entry in entries
        if entry.get("worker_id") is not None
        and entry.get("evaluation_seconds") is not None
    ]
    controller_samples = [
        entry
        for entry in entries
        if entry.get("state") == "controller_sample"
    ]
    worker_ids = {int(entry["worker_id"]) for entry in attempts}
    spawned_worker_ids = {
        int(entry["worker_id"])
        for entry in entries
        if entry.get("state") == "parent_spawned"
        and entry.get("worker_id") is not None
    }
    launched_worker_ids = spawned_worker_ids or worker_ids
    initial_pool_size = min(
        len(launched_worker_ids),
        (
            requested_workers
            if requested_workers is not None
            else len(launched_worker_ids)
        ),
    )
    ask_times = [float(entry["ask_seconds"]) for entry in attempts]
    evaluation_times = [
        float(entry["evaluation_seconds"]) for entry in attempts
    ]
    return {
        "recorded_attempts": len(attempts),
        "worker_launches": len(launched_worker_ids),
        "workers_with_attempts": len(worker_ids),
        "workers_without_attempts": len(launched_worker_ids - worker_ids),
        "spawned_worker_useful_fraction": (
            len(worker_ids) / len(launched_worker_ids)
            if launched_worker_ids
            else 0.0
        ),
        "worker_recycles": max(
            0, len(launched_worker_ids) - initial_pool_size
        ),
        "rss_triggered_recycles": len(
            {
                int(entry["worker_id"])
                for entry in attempts
                if bool(entry.get("rss_recycle_requested", False))
            }
        ),
        "memory_triggered_recycles": len(
            {
                int(entry["worker_id"])
                for entry in attempts
                if bool(entry.get("memory_recycle_requested", False))
            }
        ),
        "peak_worker_rss_gib": max(
            float(entry["peak_rss_gib"])
            for entry in entries
            if entry.get("peak_rss_gib") is not None
        ),
        "minimum_mem_available_gib": min(
            float(entry["mem_available_gib"])
            for entry in entries
            if entry.get("mem_available_gib") is not None
        ),
        "ask_wall_time_seconds": sum(
            float(entry["ask_seconds"]) for entry in attempts
        ),
        "mean_ask_wall_time_seconds": (
            sum(ask_times) / len(ask_times) if ask_times else 0.0
        ),
        "p95_ask_wall_time_seconds": (
            percentile(ask_times, 0.95) if ask_times else 0.0
        ),
        "evaluation_wall_time_seconds": sum(
            float(entry["evaluation_seconds"]) for entry in attempts
        ),
        "mean_evaluation_wall_time_seconds": (
            sum(evaluation_times) / len(evaluation_times)
            if evaluation_times
            else 0.0
        ),
        "p95_evaluation_wall_time_seconds": (
            percentile(evaluation_times, 0.95)
            if evaluation_times
            else 0.0
        ),
        "maximum_dynamic_concurrency": max(
            (
                int(entry.get("active_workers", 0))
                for entry in controller_samples
            ),
            default=0,
        ),
        "mean_system_cpu_utilization_pct": (
            sum(
                float(entry.get("system_cpu_utilization_pct", 0.0))
                for entry in controller_samples
            )
            / len(controller_samples)
            if controller_samples
            else None
        ),
        "mean_pool_cpu_core_equivalents": (
            sum(
                float(entry.get("pool_cpu_core_equivalents", 0.0))
                for entry in controller_samples
            )
            / len(controller_samples)
            if controller_samples
            else None
        ),
        "peak_pool_cpu_core_equivalents": max(
            (
                float(entry.get("pool_cpu_core_equivalents", 0.0))
                for entry in controller_samples
            ),
            default=0.0,
        ),
        "mean_pool_worker_cpu_utilization_pct": (
            sum(
                float(entry.get("pool_worker_cpu_utilization_pct", 0.0))
                for entry in controller_samples
            )
            / len(controller_samples)
            if controller_samples
            else None
        ),
        "mean_pool_cpu_capacity_utilization_pct": (
            sum(
                float(entry.get("pool_cpu_capacity_utilization_pct", 0.0))
                for entry in controller_samples
            )
            / len(controller_samples)
            if controller_samples
            else None
        ),
        "peak_active_process_tree_rss_gib": max(
            (
                float(entry.get("active_process_tree_rss_gib", 0.0))
                for entry in controller_samples
            ),
            default=0.0,
        ),
        "maximum_memory_prediction_error_gib": max(
            (
                float(entry.get("memory_prediction_error_gib", 0.0))
                for entry in controller_samples
            ),
            default=0.0,
        ),
        "state_counts": dict(
            Counter(str(entry["state"]) for entry in attempts)
        ),
        "parent_termination_count": sum(
            entry.get("state") == "parent_terminated" for entry in entries
        ),
        "parent_termination_reasons": dict(
            Counter(
                str(entry.get("reason", "unknown"))
                for entry in entries
                if entry.get("state") == "parent_terminated"
            )
        ),
    }


def write_best_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "trial",
        "state",
        "latency_ms",
        "latency_source",
        "compiler_compute_latency_ms",
        "compiler_memory_latency_ms",
        "compiler_roofline_latency_ms",
        "compiler_serial_latency_ms",
        "compiler_memory_model_version",
        "compiler_memory_evaluation_mode",
        "compiler_cost_cache_hit",
        "compiler_cost_cache_tier",
        "compiler_cost_cache_scope",
        "compiler_cost_cache_key",
        "area_cache_tier",
        "area_cache_scope",
        "compiler_compute_pipeline_makespan_cycles",
        "compiler_one_layer_compute_pipeline_makespan_cycles",
        "compiler_compute_pipeline_fidelity",
        "compiler_one_layer_compute_pipeline_fidelity",
        "compiler_compute_pipeline_cache_hit",
        "compiler_compute_pipeline_persistent_cache_hit",
        "power_status",
        "power_model",
        "power_scope",
        "onchip_energy_mj",
        "onchip_average_power_w",
        "energy_per_input_token_mj",
        "power_calibration_status",
        "power_uncertainty",
        "external_memory_model",
        "external_memory_calibration_status",
        "external_memory_configuration_semantics",
        "external_hbm_capacity_bytes",
        "external_hbm_configured_bandwidth_gbps",
        "hbm_background_energy_mj",
        "hbm_read_energy_mj",
        "hbm_write_energy_mj",
        "external_hbm_energy_mj",
        "external_hbm_energy_p10_mj",
        "external_hbm_energy_p50_mj",
        "external_hbm_energy_p90_mj",
        "external_hbm_average_power_w",
        "external_hbm_average_power_p10_w",
        "external_hbm_average_power_p50_w",
        "external_hbm_average_power_p90_w",
        "hbm_physical_read_bytes",
        "hbm_physical_write_bytes",
        "hbm_payload_read_bytes",
        "hbm_payload_write_bytes",
        "physical_to_payload_traffic_ratio",
        "achieved_average_bandwidth_gbps",
        "bandwidth_utilization",
        "external_hbm_energy_by_role",
        "external_hbm_energy_by_stage",
        "external_hbm_energy_by_opcode",
        "system_energy_mj",
        "system_energy_nominal_mj",
        "system_energy_p10_mj",
        "system_energy_p50_mj",
        "system_energy_p90_mj",
        "system_average_power_w",
        "system_average_power_p10_w",
        "system_average_power_p50_w",
        "system_average_power_p90_w",
        "system_energy_per_input_token_mj",
        "system_energy_optimistic_c2c_mj",
        "system_energy_conservative_measured_path_mj",
        "interconnect_aggregate_bytes",
        "interconnect_dynamic_energy_mj",
        "interconnect_dynamic_energy_optimistic_c2c_mj",
        "interconnect_dynamic_energy_conservative_measured_path_mj",
        "multi_chip_energy_partition_fidelity",
        "power_warnings",
        "power_excludes",
        "v3_memory_latency_ms",
        "v3_serial_latency_ms",
        "area",
        "area_um2",
        "area_mm2",
        "area_budget_constraint_mm2",
        "a100_area_constraint_mm2",
        "within_target_area_tolerance",
        "accuracy_score",
        "batch_size",
        "latency_batch_size",
        "hbm_capacity_max_batch",
        "input_seq_len",
        "MLEN",
        "BLEN",
        "VLEN",
        "INT_DATA_WIDTH",
        "native_layout_mode",
        "logical_token_rows",
        "physical_token_rows",
        "sequence_row_utilization",
        "batch_pack_factor",
        "logical_q_width",
        "physical_q_width",
        "head_lane_utilization",
        "precision_profile",
        "weight_precision",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in sorted(records, key=lambda r: (r.get("latency_ms", float("inf")), r.get("area", float("inf")))):
            writer.writerow(record)


def a100_constraints(trial: optuna.trial.FrozenTrial) -> tuple[float]:
    return (
        float(
            trial.user_attrs.get(
                "area_budget_constraint_mm2",
                trial.user_attrs.get("a100_area_constraint_mm2", 0.0),
            )
        ),
    )


def read_trial_records(
    run_dir: Path,
    *,
    model: Mapping[str, Any] | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    native_layout_mode: str | None = None,
    persist_layout_backfill: bool = False,
) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.glob("trial_*/trial_record.json")):
        try:
            record = load_json(path)
            original_layout_schema = record.get("native_layout_schema_version")
            if (
                model is not None
                and seq_len is not None
                and batch_size is not None
                and native_layout_mode is not None
                and record.get("MLEN") is not None
            ):
                record.update(
                    planned_layout_record_fields(
                        record,
                        model=model,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        native_layout_mode=native_layout_mode,
                    )
                )
            compiler_report_path = path.parent / "compiler_cost_report.json"
            if compiler_report_path.exists():
                compiler_fields = compiler_layout_record_fields(
                    load_json(compiler_report_path)
                )
                record.update(
                    {
                        name: value
                        for name, value in compiler_fields.items()
                        if value is not None
                    }
                )
            if (
                persist_layout_backfill
                and original_layout_schema is None
                and record.get("native_layout_schema_version") is not None
            ):
                write_json(path, record)
            records.append(record)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda record: int(record.get("trial", -1)))


def _settled_trial_count(study: optuna.Study) -> int:
    """Count grid points that need no retry.

    Failed attempts are deliberately excluded: a killed worker must not make a
    Cartesian-product point look complete merely because Optuna considers FAIL
    a terminal state.
    """

    settled = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    # Interrupted attempts can be explicitly re-enqueued.  Counting attempts
    # would then terminate an exhaustive grid early even though another grid
    # parameter tuple is still absent.  Canonical parameter JSON gives the
    # exact Cartesian-product coverage count required by GridSampler.
    return len(
        {
            json.dumps(trial.params, sort_keys=True, separators=(",", ":"))
            for trial in study.get_trials(deepcopy=False)
            if trial.state in settled
        }
    )


def _trial_requested_params(
    trial: optuna.trial.FrozenTrial,
) -> dict[str, Any]:
    """Return suggested params or the fixed params of a queued trial.

    Optuna keeps ``enqueue_trial()`` parameters in the ``fixed_params`` system
    attribute until a worker asks for that WAITING trial.  Reading only
    ``trial.params`` therefore makes queued points look like the same empty
    tuple and defeats retry deduplication.
    """

    if trial.params:
        return dict(trial.params)
    fixed = trial.system_attrs.get("fixed_params", {})
    return dict(fixed) if isinstance(fixed, Mapping) else {}


def reconcile_interrupted_trials(study: optuna.Study, run_dir: Path) -> dict[str, int]:
    """Repair RUNNING journal entries left by terminated worker processes.

    A trial writes ``trial_record.json`` before returning its objective values,
    so a narrow interruption window can leave a complete record paired with a
    RUNNING Optuna state.  Such records are recovered in place.  Other RUNNING
    trials are marked failed and their exact GridSampler parameters are queued
    for a fresh attempt.
    """

    counts = {
        "recovered_complete": 0,
        "recovered_pruned": 0,
        "requeued_running": 0,
        "requeued_failed": 0,
    }
    storage = study._storage
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.RUNNING:
            continue
        record_path = run_dir / f"trial_{trial.number:04d}" / "trial_record.json"
        try:
            record = load_json(record_path)
        except (OSError, json.JSONDecodeError):
            record = {}
        record_state = record.get("state")
        if record_state == "complete" and all(
            record.get(name) is not None
            for name in (
                "latency_ms",
                "system_energy_nominal_mj",
            )
        ):
            for name in (
                "area_budget_constraint_mm2",
                "a100_area_constraint_mm2",
                "area_mm2",
                "system_energy_nominal_mj",
            ):
                if record.get(name) is not None:
                    storage.set_trial_user_attr(trial._trial_id, name, record[name])
            storage.set_trial_state_values(
                trial._trial_id,
                optuna.trial.TrialState.COMPLETE,
                list(
                    ObjectiveValues.from_trial_record(
                        record
                    ).as_optuna_values()
                ),
            )
            counts["recovered_complete"] += 1
            continue
        if record_state == "pruned":
            storage.set_trial_state_values(
                trial._trial_id, optuna.trial.TrialState.PRUNED
            )
            counts["recovered_pruned"] += 1
            continue

        storage.set_trial_state_values(trial._trial_id, optuna.trial.TrialState.FAIL)
        interrupted = dict(record)
        interrupted.update(
            {
                "trial": trial.number,
                "state": "failed",
                "reason": "interrupted_worker_requeued",
            }
        )
        write_json(record_path, interrupted)
        requested_params = _trial_requested_params(trial)
        if requested_params:
            study.enqueue_trial(requested_params)
        counts["requeued_running"] += 1

    # GridSampler regards failed parameters as visited.  Requeue only failures
    # that do not already have a COMPLETE/PRUNED replacement, otherwise an
    # interrupted worker can leave a permanent hole in an apparently exhausted
    # grid.  Include WAITING entries to avoid enqueueing the same retry twice.
    refreshed = study.get_trials(deepcopy=False)
    settled_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }

    def params_key(trial: optuna.trial.FrozenTrial) -> str:
        return json.dumps(
            _trial_requested_params(trial),
            sort_keys=True,
            separators=(",", ":"),
        )

    settled_keys = {
        params_key(trial) for trial in refreshed if trial.state in settled_states
    }
    queued_keys = {
        params_key(trial)
        for trial in refreshed
        if trial.state
        in {optuna.trial.TrialState.WAITING, optuna.trial.TrialState.RUNNING}
    }
    for trial in refreshed:
        if trial.state != optuna.trial.TrialState.FAIL or not trial.params:
            continue
        key = params_key(trial)
        if key in settled_keys or key in queued_keys:
            continue
        study.enqueue_trial(_trial_requested_params(trial))
        queued_keys.add(key)
        counts["requeued_failed"] += 1
    return counts


def next_worker_id(run_dir: Path) -> int:
    ids = []
    for path in run_dir.glob("worker_*.log"):
        match = re.fullmatch(r"worker_(\d+)\.log", path.name)
        if match:
            ids.append(int(match.group(1)))
    return max(ids, default=-1) + 1


def finalize_redundant_waiting_trials(study: optuna.Study) -> int:
    """Fail queued retries whose grid parameters already have a result."""

    trials = study.get_trials(deepcopy=False)

    def key(trial: optuna.trial.FrozenTrial) -> str:
        return json.dumps(
            _trial_requested_params(trial),
            sort_keys=True,
            separators=(",", ":"),
        )

    settled = {
        key(trial)
        for trial in trials
        if trial.state
        in {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}
    }
    finalized = 0
    for trial in trials:
        if trial.state != optuna.trial.TrialState.WAITING:
            continue
        if key(trial) not in settled:
            continue
        study._storage.set_trial_state_values(
            trial._trial_id, optuna.trial.TrialState.FAIL
        )
        finalized += 1
    return finalized


def enqueue_missing_grid_trials(
    study: optuna.Study,
    search_space: Mapping[str, list[Any]],
) -> int:
    """Queue each still-missing Cartesian-product point exactly once.

    Optuna's :class:`GridSampler` can suggest duplicate points when many
    independent worker processes reach the tail of a distributed grid at the
    same time.  That behavior is harmless early in a run, but repeatedly
    launching workers for the last few points can spend most attempts on
    already-settled tuples.  On resume and in retry waves, explicitly enqueue
    the missing tuples so workers consume deterministic WAITING trials before
    asking the sampler for another suggestion.
    """

    names = tuple(search_space)

    def key(params: Mapping[str, Any]) -> str:
        return json.dumps(dict(params), sort_keys=True, separators=(",", ":"))

    trials = study.get_trials(deepcopy=False)
    settled_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    settled_keys = {
        key(_trial_requested_params(trial))
        for trial in trials
        if trial.state in settled_states
    }
    queued_keys = {
        key(_trial_requested_params(trial))
        for trial in trials
        if trial.state
        in {optuna.trial.TrialState.WAITING, optuna.trial.TrialState.RUNNING}
    }

    enqueued = 0
    values = (search_space[name] for name in names)
    for combination in itertools.product(*values):
        params = dict(zip(names, combination, strict=True))
        params_key = key(params)
        if params_key in settled_keys or params_key in queued_keys:
            continue
        study.enqueue_trial(params)
        queued_keys.add(params_key)
        enqueued += 1
    return enqueued


def canonical_grid_records(
    study: optuna.Study, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return one COMPLETE/PRUNED record for every settled grid tuple."""

    records_by_trial = {
        int(record.get("trial", -1)): record for record in records
    }
    selected: dict[str, tuple[optuna.trial.FrozenTrial, dict[str, Any]]] = {}
    for trial in study.get_trials(deepcopy=False):
        if trial.state not in {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        }:
            continue
        key = json.dumps(trial.params, sort_keys=True, separators=(",", ":"))
        record = dict(records_by_trial.get(trial.number, {}))
        record.setdefault("trial", trial.number)
        record["state"] = (
            "complete"
            if trial.state == optuna.trial.TrialState.COMPLETE
            else "pruned"
        )
        for name, value in trial.params.items():
            record.setdefault(name, value)
        existing = selected.get(key)
        if existing is None:
            selected[key] = (trial, record)
            continue
        existing_trial, _ = existing
        # A successful retry is preferred to a pruned duplicate; otherwise use
        # the latest attempt so diagnostics point to the final execution.
        if (
            trial.state == optuna.trial.TrialState.COMPLETE
            and existing_trial.state != optuna.trial.TrialState.COMPLETE
        ) or (
            trial.state == existing_trial.state
            and trial.number > existing_trial.number
        ):
            selected[key] = (trial, record)
    return sorted(
        (record for _, record in selected.values()),
        key=lambda record: int(record.get("trial", -1)),
    )


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "trial", "state", "reason",
        "normalized_latency", "normalized_energy",
        "objective_normalization",
        "latency_ms", "latency_source",
        "compiler_compute_latency_ms", "compiler_memory_latency_ms",
        "compiler_roofline_latency_ms",
        "compiler_serial_latency_ms", "compiler_memory_model_version",
        "compiler_memory_evaluation_mode", "compiler_cost_cache_hit",
        "compiler_cost_cache_tier", "compiler_cost_cache_scope",
        "compiler_cost_cache_key", "area_cache_tier", "area_cache_scope",
        "compiler_compute_pipeline_makespan_cycles",
        "compiler_one_layer_compute_pipeline_makespan_cycles",
        "compiler_compute_pipeline_fidelity",
        "compiler_one_layer_compute_pipeline_fidelity",
        "compiler_compute_pipeline_cache_hit",
        "compiler_compute_pipeline_persistent_cache_hit",
        "power_status", "power_model", "power_scope", "onchip_energy_mj",
        "onchip_average_power_w", "energy_per_input_token_mj",
        "power_calibration_status", "power_uncertainty",
        "external_memory_model", "external_memory_calibration_status",
        "external_memory_configuration_semantics",
        "external_hbm_capacity_bytes",
        "external_hbm_configured_bandwidth_gbps",
        "hbm_background_energy_mj", "hbm_read_energy_mj",
        "hbm_write_energy_mj", "external_hbm_energy_mj",
        "external_hbm_energy_p10_mj", "external_hbm_energy_p50_mj",
        "external_hbm_energy_p90_mj", "external_hbm_average_power_w",
        "external_hbm_average_power_p10_w",
        "external_hbm_average_power_p50_w",
        "external_hbm_average_power_p90_w",
        "hbm_physical_read_bytes", "hbm_physical_write_bytes",
        "hbm_payload_read_bytes", "hbm_payload_write_bytes",
        "physical_to_payload_traffic_ratio",
        "achieved_average_bandwidth_gbps", "bandwidth_utilization",
        "external_hbm_energy_by_role", "external_hbm_energy_by_stage",
        "external_hbm_energy_by_opcode",
        "system_energy_mj", "system_energy_nominal_mj",
        "system_energy_p10_mj",
        "system_energy_p50_mj", "system_energy_p90_mj",
        "system_average_power_w", "system_average_power_p10_w",
        "system_average_power_p50_w", "system_average_power_p90_w",
        "system_energy_per_input_token_mj",
        "system_energy_optimistic_c2c_mj",
        "system_energy_conservative_measured_path_mj",
        "interconnect_aggregate_bytes", "interconnect_dynamic_energy_mj",
        "interconnect_dynamic_energy_optimistic_c2c_mj",
        "interconnect_dynamic_energy_conservative_measured_path_mj",
        "multi_chip_energy_partition_fidelity",
        "power_warnings",
        "power_excludes",
        "v3_memory_latency_ms", "v3_serial_latency_ms",
        "area_um2", "area_mm2", "area_uncertainty_p10_mm2",
        "area_uncertainty_p50_mm2", "area_uncertainty_p90_mm2",
        "core_area_mm2", "endpoint_area_mm2", "physical_chip_area_mm2",
        "total_silicon_area_mm2", "total_silicon_area_p90_mm2",
        "endpoint_area_overhead_fraction", "endpoint_area_semantics",
        "endpoint_area_mm2_per_port",
        "area_budget_constraint_mm2", "a100_area_constraint_mm2",
        "within_target_area_tolerance",
        "accuracy_score", "precision_profile", "precision_search_encoding",
        "matrix_datapath_signature", "matrix_weight_operand_family",
        "matrix_activation_operand_family", "matrix_weight_port_bits",
        "matrix_activation_port_bits", "matrix_pe_bit_product",
        "matrix_output_fp_bits", "precision_variant_count",
        "weight_precision", "MLEN", "VLEN", "BLEN",
        "MATRIX_K_SPLITS", "HLEN", "BROADCAST_AMOUNT", "INT_DATA_WIDTH",
        "chip_count", "physical_chip_count", "chip_count_search_value",
        "chips_per_a100_reference", "chip_count_scaling",
        "reference_a100_count", "parallel_model",
        "multi_chip_model", "dp_degree", "tp_degree", "cp_degree", "ep_degree",
        "dp_tp_ep_legality", "tp_cp_legality", "tp_cp_ep_legality",
        "nvlink_port_count", "nvlink_bandwidth_semantics",
        "nvlink_peak_oneway_bandwidth_gbps",
        "search_encoding", "matrix_sram_config_id",
        "matrix_sram_policy", "matrix_sram_policy_aliases",
        "MATRIX_SRAM_TILES", "MATRIX_SRAM_SIZE", "matrix_sram_depth",
        "matrix_sram_width_bits", "matrix_sram_logical_bits",
        "matrix_sram_logical_mb",
        "projection_threshold_tiles", "attention_threshold_tiles",
        "matrix_sram_useful_saturation_tiles", "capacity_dominated",
        "projection_k_tiles", "projection_k_chunks",
        "compiler_mram_tile_capacity", "attention_kv_resident",
        "attention_resident_kv_tiles", "kv_tile_load_count",
        "ideal_kv_tile_load_count", "kv_reload_factor", "qk_recompute_factor",
        "VECTOR_SRAM_SIZE",
        "INT_SRAM_DEPTH", "FP_SRAM_DEPTH", "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount", "HBM_V_Writeback_Amount", "calibration_in_domain",
        "aggregate_hbm_capacity_bytes", "aggregate_hbm_bandwidth_gbps",
        "per_chip_hbm_capacity_bytes", "per_chip_hbm_bandwidth_gbps",
        "per_chip_equivalent_hbm_channels",
        "hbm_channel_calibration_status", "hbm_channel_extrapolation_ratio",
        "aggregate_hbm_required_bytes", "per_chip_hbm_required_bytes",
        "aggregate_prefill_kv_capacity_bytes",
        "full_decoder_kv_cache_bytes_shadow",
        "kv_capacity_mode", "kv_capacity_semantics",
        "kv_resident_layer_count", "kv_handoff_staging_layer_count",
        "full_decoder_kv_cache_capacity_included",
        "per_chip_hbm_capacity_feasible",
        "per_chip_compute_scale", "r_aware_v4_floor_ns",
        "r_aware_v4_residual_ns", "per_chip_hbm_physical_bytes",
        "aggregate_hbm_physical_bytes", "per_chip_achieved_bandwidth_gbps",
        "per_chip_bandwidth_utilization", "required_feed_ratio",
        "legacy_bandwidth_would_prune", "interconnect_bytes",
        "interconnect_latency_ms", "tp_collective_bytes",
        "tp_collective_latency_ns", "cp_kv_ring_bytes",
        "cp_kv_ring_latency_ns", "max_token_fraction",
        "ep_dispatch_latency_ns", "ep_return_latency_ns",
        "dependency_serial_nominal_ns", "request_origin_count",
        "active_request_origin_count", "idle_request_origin_count",
        "local_batch_by_origin", "batch_packing_utilization",
        "fixed_batch_requests_per_second", "fixed_batch_tokens_per_second",
        "max_causal_pair_fraction", "parallel_work_census_coverage",
        "parallel_kernel_census_coverage", "local_tile_counts_by_rank",
        "slowest_rank", "matrix_utilization_by_stage",
        "vector_utilization_by_stage", "padding_cycles",
        "replicated_compute_cycles", "tp_rounding_overhead",
        "cp_tail_overhead", "dp_batch_imbalance", "expert_weight_replication",
        "experts_per_rank", "expert_bucket_utilization",
        "ep_dispatch_bytes", "ep_return_bytes",
        "fractional_v2_latency", "tile_aware_v3_latency",
        "weight_replication_factor", "communication_overlap_bound",
        "full_overlap_lower_bound_ns", "nominal_stage_model_ns",
        "no_overlap_upper_bound_ns", "fp16_kv_handoff_bytes",
        "fp16_kv_handoff_max_source_bytes",
        "fp16_kv_handoff_latency_ms",
        "fp16_kv_handoff_source_chip_count",
        "fp16_kv_handoff_decode_chip_count",
        "fp16_kv_handoff_source_port_count",
        "fp16_kv_handoff_decode_port_count",
        "fp16_kv_handoff_effective_bandwidth_gbps",
        "fp16_kv_handoff_bottleneck",
        "fp16_kv_handoff_connection_waves",
        "prefill_latency_excluding_kv_handoff_ms",
        "kv_handoff_included_in_dse_latency",
        "prefill_kv_handoff_full_overlap_lower_bound_ms",
        "prefill_plus_kv_handoff_serial_shadow_ms",
        "kv_handoff_overlap_semantics", "multi_chip_fidelity",
        "active_sequence_rows", "physical_sequence_rows", "rows_per_batch",
        "sequence_row_utilization", "sequence_padding_factor",
        "native_layout_schema_version", "native_layout_mode",
        "logical_token_rows", "physical_token_rows", "batch_pack_factor",
        "rows_per_attention_group", "attention_mask_kind",
        "attention_group_count", "logical_q_width", "physical_q_width",
        "head_lane_utilization", "attention_storage_block_count",
        "attention_groups_per_storage_block",
        "attention_execution_lane_utilization",
        "attention_group_broadcast", "attention_hardware_broadcast",
        "ffn_address_schedule",
        "ffn_projection_schedule",
        "ffn_loop_plan_version",
        "ffn_explicit_loop_depth",
        "ffn_agu_streams_by_axis",
        "ffn_address_cycles_before",
        "ffn_address_cycles_after",
        "ffn_schedule_guard_status",
        "ffn_schedule_fallback_reason",
        "ffn_legacy_template_bypassed",
        "ffn_dead_k_pointer_updates_elided",
        "ffn_dead_prefetch_updates_elided",
        "ffn_dead_output_updates_elided",
        "ffn_invariant_stride_loads",
        "ffn_large_immediate_chunks_avoided",
        "ffn_residual_address_opcodes",
        "compute_fidelity_status", "compute_measured_cycle_fraction",
        "compute_structural_extrapolation_cycle_fraction",
        "compute_unsupported_cycle_fraction", "candidate_fidelity",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _strip_worker_cli(argv: list[str]) -> list[str]:
    takes_value = {
        "--workers",
        "--run-dir",
        "--worker-id",
        "--worker-trials",
        "--worker-max-trials-per-process",
    }
    flags = {"--worker-mode"}
    result = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        name = arg.split("=", 1)[0]
        if name in flags:
            index += 1
            continue
        if name in takes_value:
            index += 1 if "=" in arg else 2
            continue
        result.append(arg)
        index += 1
    return result


def worker_trial_quota(
    total_trials: int,
    workers: int,
    max_trials_per_process: int,
) -> int:
    """Choose a balanced worker quota, optionally limited by a recycle cap.

    A zero cap disables count-based recycling. The finite quota then only
    divides the current trial wave across the requested worker pool; RSS and
    system-memory guards can still recycle or terminate a worker.
    """

    if total_trials <= 0 or workers <= 0 or max_trials_per_process < 0:
        raise ValueError(
            "worker quota trial/worker inputs must be positive and the "
            "process cap must be nonnegative"
        )
    balanced_quota = max(1, total_trials // workers)
    if max_trials_per_process == 0:
        return balanced_quota
    return min(max_trials_per_process, balanced_quota)


def launch_worker_processes(
    run_dir: Path,
    workers: int,
    total_trials: int,
    start_worker_id: int = 0,
    *,
    max_trials_per_process: int = 0,
    memory_reserve_gib: float = DEFAULT_MEMORY_RESERVE_GIB,
    memory_resume_gib: float = DEFAULT_MEMORY_RESUME_GIB,
    memory_emergency_gib: float = DEFAULT_MEMORY_EMERGENCY_GIB,
    initial_worker_rss_gib: float = DEFAULT_INITIAL_WORKER_RSS_GIB,
    process_tree_rss_limit_gib: float = DEFAULT_PROCESS_TREE_RSS_LIMIT_GIB,
    stall_timeout_seconds: float = DEFAULT_WORKER_STALL_TIMEOUT_SECONDS,
    launch_burst: int = DEFAULT_WORKER_LAUNCH_BURST,
    launch_interval_seconds: float = DEFAULT_WORKER_LAUNCH_INTERVAL_SECONDS,
    monitor_interval_seconds: float = DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS,
    reconcile_callback: Callable[[], None] | None = None,
    persistent_pull_budget: bool = False,
    work_claim_available: Callable[[], bool] | None = None,
) -> tuple[list[int], int]:
    """Run a dynamically admitted, bounded-memory Optuna worker pool.

    Count-based recycling is optional. With a zero cap, workers retain their
    local summary LRU for a balanced share of the current wave and are recycled
    only by the existing RSS or system-memory guards.
    """

    if workers <= 0 or total_trials <= 0 or max_trials_per_process < 0:
        raise ValueError(
            "workers and total_trials must be positive; "
            "max_trials_per_process must be nonnegative"
        )
    if launch_burst <= 0:
        raise ValueError("worker launch burst must be positive")
    if launch_interval_seconds <= 0 or monitor_interval_seconds <= 0:
        raise ValueError("worker launch and monitor intervals must be positive")
    base_args = _strip_worker_cli(sys.argv[1:])
    active: list[tuple[subprocess.Popen[str], Any]] = []
    return_codes: list[int] = []
    trials_assigned = 0
    next_worker_id = start_worker_id
    memory_paused = False
    next_fill_after = 0.0
    resource_log_path = run_dir / "worker_resources.jsonl"
    terminated_pids: set[int] = set()
    phase_rss_samples: dict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=128)
    )
    heartbeat_progress: dict[int, tuple[tuple[Any, ...], float]] = {}
    cpu_samples: dict[int, deque[tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=256)
    )
    last_controller_sample = 0.0
    next_resource_sample = 0.0
    previous_cpu_sample = system_cpu_jiffies()
    cpu_capacity = logical_cpu_capacity()
    launch_quota = worker_trial_quota(
        total_trials,
        workers,
        max_trials_per_process,
    )
    if persistent_pull_budget and work_claim_available is None:
        raise ValueError(
            "persistent pull budget requires work_claim_available callback"
        )

    def completed_worker_peaks() -> list[float]:
        peaks = []
        if resource_log_path.exists():
            for line in resource_log_path.read_text().splitlines()[-128:]:
                try:
                    peaks.append(float(json.loads(line)["peak_rss_gib"]))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
        return peaks

    def predicted_peak_gib(phase: str | None = None) -> float:
        samples = (
            list(phase_rss_samples.get(phase, ()))
            if phase is not None
            else []
        )
        if not samples:
            samples = completed_worker_peaks()
        return max(
            initial_worker_rss_gib,
            percentile(samples, 0.90) if samples else initial_worker_rss_gib,
        )

    def read_heartbeat(process: subprocess.Popen[str]) -> dict[str, Any]:
        path = run_dir / f"worker_heartbeat_pid_{process.pid}.json"
        if not path.exists():
            return {}
        try:
            return load_json(path)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return {}

    def future_growth_headroom_gib() -> float:
        total = 0.0
        for process, _handle in active:
            if process.poll() is not None:
                continue
            current = process_tree_rss_gib(process.pid)
            phase = str(read_heartbeat(process).get("phase", "startup"))
            total += max(0.0, predicted_peak_gib(phase) - current)
        return total

    def spawn_one(quota: int) -> None:
        nonlocal next_worker_id
        worker_id = next_worker_id
        next_worker_id += 1
        log_handle = (run_dir / f"worker_{worker_id:03d}.log").open("a")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_args,
            "--run-dir", str(run_dir),
            "--workers", "1",
            "--worker-mode",
            "--worker-id", str(worker_id),
            "--worker-trials", str(quota),
        ]
        child_env = os.environ.copy()
        child_env.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "TORCH_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
            }
        )
        active.append(
            (
                subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=child_env,
                ),
                log_handle,
            )
        )

    def fill_pool() -> None:
        nonlocal trials_assigned, memory_paused, next_fill_after
        if time.monotonic() < next_fill_after:
            return
        if len(active) >= workers:
            return
        if persistent_pull_budget:
            if work_claim_available is None or not work_claim_available():
                return
        elif trials_assigned >= total_trials:
            return
        available = mem_available_gib()
        projected_growth = future_growth_headroom_gib()
        spawned = 0
        while (
            len(active) < workers
            and (
                (
                    persistent_pull_budget
                    and work_claim_available is not None
                    and work_claim_available()
                )
                or (
                    not persistent_pull_budget
                    and trials_assigned < total_trials
                )
            )
            and spawned < launch_burst
        ):
            threshold = (
                memory_resume_gib if memory_paused else memory_reserve_gib
            )
            worker_peak = predicted_peak_gib()
            if available - projected_growth - worker_peak < threshold:
                memory_paused = True
                break
            memory_paused = False
            quota = (
                max_trials_per_process
                if persistent_pull_budget
                else min(launch_quota, total_trials - trials_assigned)
            )
            spawn_one(quota)
            append_jsonl(
                resource_log_path,
                {
                    "worker_id": next_worker_id - 1,
                    "state": "parent_spawned",
                    "quota": quota,
                    "active_workers": len(active),
                    "mem_available_gib": available,
                    "predicted_worker_peak_gib": worker_peak,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            if not persistent_pull_budget:
                trials_assigned += quota
            spawned += 1
            projected_growth += worker_peak
        if spawned:
            # Give newly spawned interpreters time to materialize imports and
            # native buffers before trusting MemAvailable for another burst.
            next_fill_after = time.monotonic() + launch_interval_seconds

    def terminate_worker(
        entry: tuple[subprocess.Popen[str], Any],
        reason: str,
    ) -> None:
        process, _ = entry
        if process.poll() is not None or process.pid in terminated_pids:
            return
        terminated_pids.add(process.pid)
        append_jsonl(
            resource_log_path,
            {
                "worker_pid": process.pid,
                "state": "parent_terminated",
                "reason": reason,
                "mem_available_gib": mem_available_gib(),
                "process_tree_rss_gib": process_tree_rss_gib(process.pid),
                "timestamp": datetime.now().isoformat(),
            },
        )
        process.terminate()

    def cpu_low_for_stall_window(process: subprocess.Popen[str]) -> bool:
        now = time.monotonic()
        samples = cpu_samples[process.pid]
        samples.append((now, process_tree_cpu_seconds(process.pid)))
        while len(samples) > 2 and now - samples[1][0] >= 120.0:
            samples.popleft()
        if len(samples) < 2 or now - samples[0][0] < 120.0:
            return False
        elapsed = max(1e-9, now - samples[0][0])
        utilization = 100.0 * (samples[-1][1] - samples[0][1]) / elapsed
        return utilization < 5.0

    fill_pool()
    try:
        while active or (
            (
                persistent_pull_budget
                and work_claim_available is not None
                and work_claim_available()
            )
            or (
                not persistent_pull_budget
                and trials_assigned < total_trials
            )
        ):
            fill_pool()
            if not active:
                time.sleep(1.0)
                continue
            completed = [entry for entry in active if entry[0].poll() is not None]
            if not completed:
                monotonic_now = time.monotonic()
                if monotonic_now < next_resource_sample:
                    time.sleep(min(0.1, next_resource_sample - monotonic_now))
                    continue
                next_resource_sample = (
                    monotonic_now + monitor_interval_seconds
                )
                worker_rss = {
                    entry: process_tree_rss_gib(entry[0].pid)
                    for entry in active
                }
                heartbeats = {
                    entry: read_heartbeat(entry[0]) for entry in active
                }
                for entry, rss in worker_rss.items():
                    heartbeat = heartbeats[entry]
                    phase = str(heartbeat.get("phase", "startup"))
                    phase_rss_samples[phase].append(rss)
                    signature = (
                        heartbeat.get("trial"),
                        phase,
                        heartbeat.get("progress_done"),
                        heartbeat.get("progress_total"),
                        heartbeat.get("current_stream"),
                    )
                    previous = heartbeat_progress.get(entry[0].pid)
                    if previous is None or previous[0] != signature:
                        heartbeat_progress[entry[0].pid] = (
                            signature,
                            monotonic_now,
                        )
                oversized = [
                    entry
                    for entry, rss in worker_rss.items()
                    if rss > process_tree_rss_limit_gib
                ]
                for entry in oversized:
                    terminate_worker(
                        entry,
                        "process_tree_rss_exceeded:"
                        f"{worker_rss[entry]:.3f}>"
                        f"{process_tree_rss_limit_gib:.3f}GiB",
                    )
                stalled = []
                for entry in active:
                    process = entry[0]
                    heartbeat = heartbeats[entry]
                    if not heartbeat:
                        continue
                    progress = heartbeat_progress.get(process.pid)
                    if progress is None:
                        continue
                    age = monotonic_now - progress[1]
                    low_cpu = cpu_low_for_stall_window(process)
                    if (
                        age > stall_timeout_seconds
                        and low_cpu
                    ):
                        stalled.append((entry, heartbeat, age))
                for entry, heartbeat, age in stalled:
                    terminate_worker(
                        entry,
                        "progress_stall_low_cpu:"
                        f"{heartbeat.get('phase', 'unknown')}:{age:.1f}s",
                    )
                available = mem_available_gib()
                if available < memory_emergency_gib and active:
                    candidates = [
                        entry
                        for entry in active
                        if entry[0].pid not in terminated_pids
                    ]
                    largest = max(
                        candidates,
                        key=lambda entry: worker_rss.get(entry, 0.0),
                    ) if candidates else None
                    if largest is not None:
                        terminate_worker(
                            largest,
                            "system_memory_below_emergency_floor:"
                            f"{available:.3f}<"
                            f"{memory_emergency_gib:.3f}GiB",
                        )
                if monotonic_now - last_controller_sample >= 2.0:
                    current_cpu_sample = system_cpu_jiffies()
                    busy_delta = (
                        current_cpu_sample[0] - previous_cpu_sample[0]
                    )
                    total_delta = (
                        current_cpu_sample[1] - previous_cpu_sample[1]
                    )
                    cpu_utilization_pct = (
                        100.0 * busy_delta / total_delta
                        if total_delta > 0
                        else 0.0
                    )
                    previous_cpu_sample = current_cpu_sample
                    pool_cpu_core_equivalents = sum(
                        max(
                            0.0,
                            (samples[-1][1] - samples[-2][1])
                            / max(1e-9, samples[-1][0] - samples[-2][0]),
                        )
                        for samples in cpu_samples.values()
                        if len(samples) >= 2
                    )
                    active_rss_gib = sum(worker_rss.values())
                    predicted_active_peak_gib = sum(
                        predicted_peak_gib(
                            str(
                                heartbeats[entry].get(
                                    "phase", "startup"
                                )
                            )
                        )
                        for entry in active
                        if entry[0].poll() is None
                    )
                    append_jsonl(
                        resource_log_path,
                        {
                            "state": "controller_sample",
                            "active_workers": len(active),
                            "requested_workers": workers,
                            "system_cpu_utilization_pct": (
                                cpu_utilization_pct
                            ),
                            "logical_cpu_capacity": cpu_capacity,
                            "pool_cpu_core_equivalents": (
                                pool_cpu_core_equivalents
                            ),
                            "pool_cpu_capacity_utilization_pct": (
                                100.0
                                * pool_cpu_core_equivalents
                                / max(1, cpu_capacity)
                            ),
                            "pool_worker_cpu_utilization_pct": (
                                100.0
                                * pool_cpu_core_equivalents
                                / max(1, len(active))
                            ),
                            "active_process_tree_rss_gib": active_rss_gib,
                            "predicted_active_peak_rss_gib": (
                                predicted_active_peak_gib
                            ),
                            "memory_prediction_error_gib": (
                                active_rss_gib - predicted_active_peak_gib
                            ),
                            "mem_available_gib": available,
                            "predicted_future_growth_gib": (
                                future_growth_headroom_gib()
                            ),
                            "memory_paused": memory_paused,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    last_controller_sample = monotonic_now
                time.sleep(min(0.1, monitor_interval_seconds))
                continue
            for process, log_handle in completed:
                active.remove((process, log_handle))
                return_codes.append(int(process.returncode or 0))
                log_handle.close()
                heartbeat_progress.pop(process.pid, None)
                cpu_samples.pop(process.pid, None)
                if process.returncode and reconcile_callback is not None:
                    reconcile_callback()
            fill_pool()
    except BaseException:
        for process, _ in active:
            if process.poll() is None:
                process.terminate()
        for process, log_handle in active:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log_handle.close()
        raise
    return return_codes, next_worker_id


def optimize_with_serialized_ask(
    study: optuna.Study,
    objective,
    *,
    n_trials: int,
    ask_lock_path: Path,
    worker_id: int,
    resource_log_path: Path,
    rss_recycle_gib: float,
    memory_reserve_gib: float,
    target_complete_trials: int | None = None,
    max_total_attempts: int | None = None,
    budget_poll_seconds: float = 0.25,
) -> int:
    """Evaluate trials in parallel while serializing Optuna trial claims.

    Optuna 4.2 can let separate processes select the same WAITING trial before
    either backend makes the state transition visible to the other claimant.
    JournalStorage additionally has to replay the complete append-only log on
    every serialized claim, which is too expensive for a 13k-point grid.

    Only trial selection is serialized here.  The objective and terminal
    ``tell`` operate outside the lock, so CostEmitter evaluations retain full
    process-level parallelism.  Refreshing the journal while holding the lock
    ensures every claimant observes the preceding RUNNING transition.
    """

    if n_trials < 0:
        raise ValueError(f"n_trials must be non-negative, got {n_trials}")
    if target_complete_trials is not None and target_complete_trials <= 0:
        raise ValueError("target_complete_trials must be positive")
    if max_total_attempts is not None and max_total_attempts <= 0:
        raise ValueError("max_total_attempts must be positive")
    persistent_budget = target_complete_trials is not None
    if persistent_budget and max_total_attempts is None:
        raise ValueError(
            "persistent complete-trial budget requires max_total_attempts"
        )
    ask_lock_path.parent.mkdir(parents=True, exist_ok=True)
    completed_attempts = 0
    sqlite_path = ask_lock_path.parent / "study.sqlite3"
    sqlite_connection = (
        sqlite3.connect(sqlite_path, timeout=120)
        if sqlite_path.exists()
        else None
    )

    def state_counts() -> Counter[str]:
        if sqlite_connection is not None:
            rows = sqlite_connection.execute(
                """
                SELECT trials.state, COUNT(*)
                FROM trials
                JOIN studies ON studies.study_id = trials.study_id
                WHERE studies.study_name = ?
                GROUP BY trials.state
                """,
                (study.study_name,),
            ).fetchall()
            return Counter(
                {str(state).upper(): int(count) for state, count in rows}
            )
        return Counter(
            trial.state.name
            for trial in study.get_trials(deepcopy=False)
        )

    def budget_action(counts: Mapping[str, int]) -> str:
        assert target_complete_trials is not None
        assert max_total_attempts is not None
        complete = int(counts.get("COMPLETE", 0))
        running = int(counts.get("RUNNING", 0))
        waiting = int(counts.get("WAITING", 0))
        total = sum(int(value) for value in counts.values())
        if complete >= target_complete_trials:
            return "stop_complete"
        if complete + running >= target_complete_trials:
            return "wait_reserved"
        if waiting:
            return "claim"
        if total >= max_total_attempts:
            return "wait_attempts" if running else "stop_attempts"
        return "claim"

    def write_budget_wait_heartbeat(action: str) -> None:
        write_json(
            ask_lock_path.parent
            / f"worker_heartbeat_pid_{os.getpid()}.json",
            {
                "worker_id": worker_id,
                "pid": os.getpid(),
                "trial": None,
                "phase": "budget_wait",
                "budget_action": action,
                "updated_epoch": time.time(),
                "current_rss_gib": current_process_rss_gib(),
                "peak_rss_gib": current_rss_gib(),
                "mem_available_gib": mem_available_gib(),
            },
        )

    try:
        with ask_lock_path.open("a+") as ask_lock:
            while persistent_budget or completed_attempts < n_trials:
                if n_trials and completed_attempts >= n_trials:
                    break
                ask_started = time.perf_counter()
                fcntl.flock(ask_lock.fileno(), fcntl.LOCK_EX)
                try:
                    sync = getattr(study._storage, "_sync_with_backend", None)
                    if sync is not None:
                        sync()
                    if persistent_budget:
                        action = budget_action(state_counts())
                        if action.startswith("stop_"):
                            break
                        if action.startswith("wait_"):
                            trial = None
                        else:
                            trial = study.ask()
                    else:
                        action = "claim"
                        trial = study.ask()
                finally:
                    fcntl.flock(ask_lock.fileno(), fcntl.LOCK_UN)
                ask_seconds = time.perf_counter() - ask_started
                if trial is None:
                    write_budget_wait_heartbeat(action)
                    time.sleep(budget_poll_seconds)
                    continue

                evaluation_started = time.perf_counter()
                terminal_state = "complete"
                try:
                    values = objective(trial)
                except optuna.TrialPruned:
                    study.tell(
                        trial,
                        state=optuna.trial.TrialState.PRUNED,
                    )
                    terminal_state = "pruned"
                except KeyboardInterrupt:
                    study.tell(
                        trial,
                        state=optuna.trial.TrialState.FAIL,
                    )
                    raise
                except Exception:
                    # Match study.optimize(..., catch=(Exception,)): preserve
                    # the failed attempt and keep pulling from the shared
                    # budget instead of terminating the process.
                    study.tell(
                        trial,
                        state=optuna.trial.TrialState.FAIL,
                    )
                    terminal_state = "failed"
                else:
                    study.tell(trial, values=values)
                finally:
                    gc.collect()
                completed_attempts += 1
                peak_rss_gib = current_rss_gib()
                available_gib = mem_available_gib()
                rss_recycle_requested = (
                    peak_rss_gib >= rss_recycle_gib
                )
                memory_recycle_requested = (
                    available_gib < memory_reserve_gib
                )
                append_jsonl(
                    resource_log_path,
                    {
                        "worker_id": worker_id,
                        "trial": trial.number,
                        "state": terminal_state,
                        "ask_seconds": ask_seconds,
                        "evaluation_seconds": (
                            time.perf_counter() - evaluation_started
                        ),
                        "peak_rss_gib": peak_rss_gib,
                        "mem_available_gib": available_gib,
                        "rss_recycle_threshold_gib": rss_recycle_gib,
                        "rss_recycle_requested": rss_recycle_requested,
                        "memory_recycle_requested": (
                            memory_recycle_requested
                        ),
                    },
                )
                if rss_recycle_requested or memory_recycle_requested:
                    break
    finally:
        if sqlite_connection is not None:
            sqlite_connection.close()
    return completed_attempts


def create_optuna_storage(
    run_dir: Path,
    *,
    requested_backend: str,
    worker_mode: bool,
    workers: str,
) -> tuple[optuna.storages.BaseStorage, str]:
    """Create the persistent Optuna backend used by all worker processes.

    JournalStorage remains available for compatibility and small sequential
    studies.  Multi-process runs use SQLite WAL by default: external locking
    serializes only ``ask()``, while indexed RDB lookups avoid replaying a
    multi-megabyte journal for every one of thousands of claims.
    """

    if requested_backend not in {"auto", "journal", "sqlite"}:
        raise ValueError(f"unsupported Optuna storage backend {requested_backend!r}")
    journal_path = run_dir / "study.journal"
    sqlite_path = run_dir / "study.sqlite3"
    backend = requested_backend
    if backend == "auto":
        if sqlite_path.exists():
            backend = "sqlite"
        elif journal_path.exists():
            backend = "journal"
        else:
            parallel = workers == "auto" or int(workers) > 1
            backend = "sqlite" if parallel and not worker_mode else "journal"

    if backend == "sqlite":
        storage = optuna.storages.RDBStorage(
            f"sqlite:///{sqlite_path}",
            engine_kwargs={"connect_args": {"timeout": 120}},
        )
        # WAL persists in the database header. busy_timeout is also provided
        # through SQLAlchemy above; this direct connection establishes WAL
        # before workers are launched.
        with sqlite3.connect(sqlite_path, timeout=120) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=120000")
        return storage, backend

    try:
        from optuna.storages.journal import JournalFileBackend

        journal_backend = JournalFileBackend(str(journal_path))
    except ImportError:  # pragma: no cover - compatibility with older Optuna
        journal_backend = optuna.storages.JournalFileStorage(str(journal_path))
    return optuna.storages.JournalStorage(journal_backend), backend



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Optuna DSE for Qwen3 dense or fixed-balanced MoE prefill"
    )
    add_model_profile_argument(parser)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Legacy attempt budget; defaults to 2048 when no complete target is set",
    )
    parser.add_argument(
        "--target-complete-trials",
        type=int,
        default=None,
        help="Absolute COMPLETE-trial target for resumable sampled studies",
    )
    parser.add_argument(
        "--max-total-attempts",
        type=int,
        default=None,
        help="Absolute attempt cap used with --target-complete-trials",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help="Qwen model configuration consumed by CostEmitter and DSE constraints",
    )
    parser.add_argument("--accuracy-constraints", type=Path, default=DEFAULT_ACCURACY_PATH)
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument(
        "--area-mode",
        choices=("none", "proxy", "proxy-v2", "proxy-v2-mxint", "parse-existing", "synth", "elaborate"),
        default="proxy-v2",
    )
    parser.add_argument("--dry-run", action="store_true", help="Alias for --area-mode none")
    parser.add_argument(
        "--latency-batch-size",
        type=int,
        default=DEFAULT_LATENCY_BATCH_SIZE,
        help="Fixed batch size used for the prefill latency objective; HBM max batch is reported separately",
    )
    parser.add_argument("--input-seq-len", type=int, default=DEFAULT_INPUT_SEQ_LEN)
    parser.add_argument("--output-seq-len", type=int, default=DEFAULT_OUTPUT_SEQ_LEN)
    parser.add_argument("--device-num", type=int, default=DEFAULT_DEVICE_NUM)
    parser.add_argument(
        "--hbm-capacity-bytes",
        type=int,
        default=DEFAULT_HBM_CAPACITY_BYTES,
        help="HBM capacity per A100 reference; aggregate capacity is R times this value",
    )
    parser.add_argument(
        "--kv-capacity-mode",
        choices=KV_CAPACITY_MODES,
        default=DEFAULT_KV_CAPACITY_MODE,
        help=(
            "Prefill-chip KV capacity semantics. The default retains one "
            "attention layer plus a handoff staging buffer; the full decode "
            "KV cache remains a reported shadow."
        ),
    )
    parser.add_argument(
        "--kv-handoff-staging-layers",
        type=int,
        default=DEFAULT_KV_HANDOFF_STAGING_LAYERS,
        help=(
            "Additional one-layer-equivalent KV buffers reserved for decode "
            "handoff in prefill-streamed-handoff-v1 mode"
        ),
    )
    parser.add_argument(
        "--hbm-bandwidth-gbps",
        type=float,
        default=DEFAULT_BANDWIDTH_LIMIT_GBPS,
        help=(
            "HBM bandwidth per A100 reference in decimal GB/s; aggregate "
            "bandwidth is R times this value"
        ),
    )
    parser.add_argument(
        "--frequency-ghz",
        type=float,
        default=DEFAULT_FREQUENCY_GHZ,
        help="Clock frequency used only for GB/s to bytes/cycle bandwidth constraint conversion",
    )
    parser.add_argument("--mx-scale-width", type=int, default=DEFAULT_MX_SCALE_WIDTH)
    parser.add_argument("--mx-scale-block-size", type=int, default=DEFAULT_MX_SCALE_BLOCK_SIZE)
    parser.add_argument("--fp-constant-num", type=int, default=DEFAULT_FP_CONSTANT_NUM)
    parser.add_argument("--weight-param-count", type=float, default=DEFAULT_WEIGHT_PARAM_COUNT)
    parser.add_argument("--weight-element-bits", type=float, default=DEFAULT_WEIGHT_ELEMENT_BITS)
    parser.add_argument(
        "--weight-precision",
        choices=("MXINT4", "MXINT8", "MXFP_E1M2", "MXFP_E2M1", "MXFP_E4M3", "MXFP_E5M2"),
        default=None,
        help=(
            "Fallback weight precision for legacy accuracy profiles without WEIGHT_WIDTH; "
            "v4 profiles use their per-profile weight precision"
        ),
    )
    parser.add_argument("--weight-mx-exp-width", type=int, default=DEFAULT_WEIGHT_MX_EXP_WIDTH)
    parser.add_argument("--weight-mx-mant-width", type=int, default=DEFAULT_WEIGHT_MX_MANT_WIDTH)
    parser.add_argument(
        "--moe-routing-mode",
        choices=("static-indices", "fixed-balanced"),
        default="static-indices",
        help=(
            "MoE routing source. Formal large-shape MoE DSE uses the "
            "latency-only fixed-balanced histogram."
        ),
    )
    parser.add_argument(
        "--moe-lowering-schedule",
        choices=("compact-route-v2", "legacy-static-v1"),
        default="compact-route-v2",
    )
    parser.add_argument(
        "--moe-layer-scaling",
        choices=("single-layer", "repeat-static-plan", "repeat-fixed-balanced"),
        default="single-layer",
        help="How a selected MoE layer routing assumption is scaled across layers",
    )
    parser.add_argument("--fixed-mlen", type=int, default=None)
    parser.add_argument("--fixed-blen", type=int, default=None)
    parser.add_argument("--fixed-vlen", type=int, default=None)
    parser.add_argument("--fixed-int-data-width", type=int, default=None)
    parser.add_argument("--fixed-precision-profile", type=str, default=None)
    parser.add_argument("--fixed-chip-count", type=int, default=None)
    parser.add_argument("--fixed-matrix-sram-tiles", type=int, default=None)
    parser.add_argument(
        "--fixed-matrix-sram-policy",
        choices=MATRIX_SRAM_POLICIES,
        default=None,
    )
    parser.add_argument(
        "--chip-counts",
        default=",".join(str(value) for value in DEFAULT_CHIP_COUNTS),
        help=(
            "Comma-separated PLENA chip-count search values. Under the "
            "default per-a100-reference semantics these are multiplied by R "
            "to obtain physical chip counts."
        ),
    )
    parser.add_argument(
        "--chip-count-scaling",
        choices=CHIP_COUNT_SCALING_MODES,
        default=DEFAULT_CHIP_COUNT_SCALING,
        help=(
            "Interpret --chip-counts and --fixed-chip-count per A100 "
            "reference (default) or as absolute physical counts."
        ),
    )
    parser.add_argument(
        "--matrix-sram-tiles",
        default=",".join(str(value) for value in BASE_MATRIX_SRAM_TILES),
        help=(
            "Base Matrix SRAM tile-count domain. Useful non-power-of-two "
            "saturation points are added automatically."
        ),
    )
    parser.add_argument(
        "--matrix-sram-policies",
        default=",".join(MATRIX_SRAM_POLICIES),
        help=(
            "Comma-separated default Matrix SRAM strategy domain: "
            + ",".join(MATRIX_SRAM_POLICIES)
        ),
    )
    parser.add_argument(
        "--reference-a100-count",
        type=int,
        default=DEFAULT_REFERENCE_A100_COUNT,
        help="Number R of A100 references supplying aggregate area/HBM resources",
    )
    parser.add_argument(
        "--parallel-model",
        choices=("tp-sp", "tp-only", "both"),
        default="tp-sp",
        help="Legacy ideal-linear partition mode (ignored by factorized TP/CP)",
    )
    parser.add_argument(
        "--multi-chip-model",
        choices=(DEFAULT_MULTI_CHIP_MODEL,),
        default=DEFAULT_MULTI_CHIP_MODEL,
        help=(
            "Formal tile-aware DP/TP/EP model. CP-v3, fractional v2, and ideal-linear "
            "baselines are available only through analytic_models.legacy."
        ),
    )
    parser.add_argument(
        "--dp-degrees",
        default="auto",
        help=(
            "Request/data-parallel degree domain. 'auto' enumerates legal "
            "DP/TP/EP tuples; otherwise provide comma-separated integers."
        ),
    )
    parser.add_argument(
        "--fixed-dp-degree",
        type=int,
        default=None,
        help="Fix request/data parallel degree; whole requests are never split",
    )
    parser.add_argument(
        "--tp-degrees",
        default="auto",
        help=(
            "TP degree domain. 'auto' uses natural factors bounded by KV heads; "
            "otherwise provide comma-separated positive integers."
        ),
    )
    parser.add_argument(
        "--fixed-tp-degree",
        type=int,
        default=None,
        help="Fix TP degree inside each DP/EP topology",
    )
    parser.add_argument(
        "--ep-degrees",
        default="auto",
        help=(
            "EP degree domain for tile-aware MoE. Dense models resolve to EP=1."
        ),
    )
    parser.add_argument(
        "--fixed-ep-degree",
        type=int,
        default=None,
        help="Fix EP degree; EP is an independent physical axis for MoE",
    )
    parser.add_argument(
        "--nvlink-port-counts",
        default=",".join(map(str, DEFAULT_NVLINK_PORT_COUNTS)),
        help="Comma-separated NVLink endpoint port-count search domain",
    )
    parser.add_argument(
        "--fixed-nvlink-port-count",
        type=int,
        choices=DEFAULT_NVLINK_PORT_COUNTS,
        default=None,
        help="Fix the number of 900 GB/s bidirectional NVLink ports",
    )
    parser.add_argument(
        "--nvlink-bandwidth-semantics",
        choices=("peak",),
        default="peak",
        help="Use 100%% architectural peak bandwidth with no efficiency penalty",
    )
    parser.add_argument(
        "--nvlink-startup-us",
        type=float,
        default=DEFAULT_NVLINK_STARTUP_US,
        help="Nominal per-ring-step startup latency in microseconds",
    )
    parser.add_argument(
        "--decode-chip-count",
        type=int,
        default=None,
        help=(
            "Decode-system chip count for the reporting-only FP16 KV "
            "handoff; defaults to the A100 reference count R."
        ),
    )
    parser.add_argument(
        "--decode-nvlink-port-count",
        type=int,
        choices=DEFAULT_NVLINK_PORT_COUNTS,
        default=None,
        help=(
            "NVLink receive ports per decode chip for the handoff shadow; "
            "defaults to each trial's selected prefill port count."
        ),
    )
    parser.add_argument(
        "--endpoint-area-overhead-pct",
        type=float,
        default=DEFAULT_ENDPOINT_OVERHEAD_FRACTION * 100.0,
        help="Per-chip interconnect/decode-handoff endpoint area overhead",
    )
    parser.add_argument(
        "--nvlink-bandwidth-gbps",
        type=float,
        default=DEFAULT_NVLINK_BIDIRECTIONAL_GBPS,
        help="Legacy total bidirectional link bandwidth for old model",
    )
    parser.add_argument(
        "--min-matrix-k-splits",
        type=int,
        default=DEFAULT_MIN_MATRIX_K_SPLITS,
        help=(
            "Optional minimum MLEN/BLEN ratio. The default 1 keeps the full "
            "search space, including the structural-v4 BLEN=MLEN topology."
        ),
    )
    parser.add_argument(
        "--strict-bandwidth",
        "--legacy-bandwidth-prune",
        dest="legacy_bandwidth_prune",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Compatibility alias for --legacy-bandwidth-policy strict. The "
            "multi-chip default records violations without pruning."
        ),
    )
    parser.add_argument(
        "--legacy-bandwidth-policy",
        choices=("diagnostic", "strict"),
        default="diagnostic",
        help="Record the legacy expression or reproduce its historical hard prune",
    )
    parser.add_argument(
        "--compiler-cost-mode",
        choices=(
            "off",
            "shadow",
            "compute-objective",
            "roofline-objective",
            "objective",
        ),
        default="roofline-objective",
        help=(
            "Record the selected memory model beside the legacy model, use CostEmitter "
            "compute only, use the stage-wise selected-compute/R-aware-V4/NVLink "
            "roofline, or use serial compiler cost as the objective"
        ),
    )
    parser.add_argument(
        "--compiler-cost-settings",
        type=Path,
        default=DEFAULT_COMPILER_COST_SETTINGS,
        help="Transactional cycle settings template for compiler cost modes",
    )
    parser.add_argument(
        "--compiler-cost-calibration",
        type=Path,
        default=DEFAULT_COMPILER_COST_CALIBRATION,
        help="HBM service calibration artifact for compiler cost modes",
    )
    parser.add_argument(
        "--compiler-compute-timing",
        choices=("ideal-ii1", "rtl-v1", "legacy"),
        default=CURRENT_DSE_PROFILE.compute_timing,
        help=(
            "CostEmitter compute timing source; ideal-ii1 uses structural "
            "Matrix timing and one cycle per Vector/Scalar/control opcode"
        ),
    )
    parser.add_argument(
        "--compiler-scheduled-shadow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate the ordered hazard/overlap shadow in addition to the "
            "resource-work objective. Disabled by default because the current "
            "Qwen3-32B compressed replay takes about one minute per trial."
        ),
    )
    parser.add_argument(
        "--compiler-v4-memory-evaluation",
        choices=(
            "auto",
            "full-global-stateful",
            "one-layer-cached-occurrence-scaled",
            "one-layer-stateful-scaled",
        ),
        default="one-layer-cached-occurrence-scaled",
        help=(
            "V4 memory-shadow fidelity. DSE defaults to cached per-occurrence "
            "evaluation of one decoder layer followed by layer-stage scaling; "
            "stateful and full-global modes are reserved for validation."
        ),
    )
    parser.add_argument(
        "--compiler-trace-granularity",
        choices=("detailed", "affine-block-summary-v1"),
        default=CURRENT_DSE_PROFILE.cost_trace_granularity,
        help=(
            "DSE defaults to exact algebraic ideal-II1 summaries; detailed "
            "ordered traces are reserved for rtl-v1 and replay validation."
        ),
    )
    parser.add_argument(
        "--trial-report-materialization",
        choices=("summary-reference", "full"),
        default="summary-reference",
        help=(
            "Store one shared compiler report per semantic cache key and put "
            "only a reference in each trial directory by default."
        ),
    )
    parser.add_argument(
        "--artifact-retention",
        choices=("compact", "full"),
        default="compact",
        help=(
            "Compact keeps resume-safe trial summaries plus gzip details until "
            "successful finalization; full preserves all historical artifacts."
        ),
    )
    parser.add_argument(
        "--native-layout-mode",
        choices=("compact", "legacy"),
        default="compact",
        help="Native decoder row/head storage layout (default: compact)",
    )
    parser.add_argument(
        "--packed-attention-schedule",
        choices=("direct-first-block-v1", "legacy"),
        default="direct-first-block-v1",
        help=(
            "Packed-GQA schedule. The default specializes the first K block "
            "and accumulates PV directly into the destination head lane."
        ),
    )
    parser.add_argument(
        "--softmax-state-schedule",
        choices=(
            SOFTMAX_STATE_SCHEDULE_STREAMED_V2,
            SOFTMAX_STATE_SCHEDULE_SRAM_V1,
            SOFTMAX_STATE_SCHEDULE_ROW_BANK_SIMD_V3,
        ),
        default=CURRENT_DSE_PROFILE.softmax_state_schedule,
        help="Online-softmax state lifetime/storage schedule.",
    )
    parser.add_argument(
        "--packed-qk-schedule",
        choices=(
            PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1,
            PACKED_QK_SCHEDULE_HEAD_MAJOR_V1,
        ),
        default=PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1,
        help="Packed-GQA QK traversal and Matrix broadcast reuse schedule.",
    )
    parser.add_argument(
        "--vector-scalar-schedule",
        choices=("rtl-v6", "rtl-v5", "rtl-v4", "rtl-v3", "rtl-v2", "compiler-v1", "legacy"),
        default=CURRENT_DSE_PROFILE.vector_scalar_schedule,
        help=(
            "Native Vector/Scalar compiler lowering. The DSE default uses the "
            "latest compact-stat SIMD path; rtl-v3 remains available for A/B."
        ),
    )
    parser.add_argument(
        "--softmax-vector-schedule",
        choices=("single-row-v1", "multi-row-v1"),
        default=CURRENT_DSE_PROFILE.softmax_vector_schedule,
        help="Single-row compatibility or banked multi-row softmax lowering.",
    )
    parser.add_argument(
        "--pv-accumulation-schedule",
        choices=("shift-add-v1", "direct-packed-rmw-v1"),
        default=CURRENT_DSE_PROFILE.pv_accumulation_schedule,
        help="PV scratch/shift/add compatibility or direct packed-O writeback.",
    )
    parser.add_argument(
        "--softmax-row-lanes",
        default=",".join(
            str(value) for value in CURRENT_DSE_PROFILE.softmax_row_lanes
        ),
        help="Comma-separated row-lane tiers searched by rtl-v6.",
    )
    parser.add_argument(
        "--fixed-softmax-row-lanes",
        type=int,
        choices=(1, 2, 4, 8),
        help="Fix the rtl-v6 row-lane tier instead of searching it.",
    )
    parser.add_argument(
        "--selector-schedule",
        choices=("hoisted-v1", "legacy"),
        default="hoisted-v1",
        help="Packed-softmax reduction-selector placement.",
    )
    parser.add_argument(
        "--reduction-output-mode",
        choices=("overwrite-v1", "accumulate-v1"),
        default="overwrite-v1",
        help="Reduction destination initialization policy.",
    )
    parser.add_argument(
        "--gqa-pipeline-schedule",
        choices=("row-interleaved-v1", "row-serial"),
        default="row-interleaved-v1",
        help="Packed-GQA row issue schedule (default: row-interleaved-v1).",
    )
    parser.add_argument(
        "--address-generation-mode",
        choices=("loop-agu-v1", "legacy"),
        default="loop-agu-v1",
        help="Loop address generation lowering (default: loop-agu-v1).",
    )
    parser.add_argument(
        "--ffn-address-schedule",
        choices=("live-stride-v1", "legacy"),
        default="live-stride-v1",
        help=(
            "FFN pointer-liveness and invariant-stride lowering. The default "
            "removes dead final updates and avoids large-immediate expansion."
        ),
    )
    parser.add_argument(
        "--ffn-projection-schedule",
        choices=("affine-loop-v2", "legacy-auto-v1"),
        default="affine-loop-v2",
        help=(
            "FFN projection lowering. affine-loop-v2 uses one shared explicit "
            "loop plan for every Matrix SRAM capacity."
        ),
    )
    parser.add_argument(
        "--power-shadow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the on-chip plus external-HBM3E system-energy model. It "
            "is required because nominal system energy is the third formal "
            "Optuna objective."
        ),
    )
    parser.add_argument(
        "--clock-gating-mode",
        choices=("ideal-hierarchical", "ungated"),
        default=CURRENT_DSE_PROFILE.clock_gating_mode,
        help=(
            "On-chip clock-power semantics. The default is an explicitly "
            "ideal hierarchical-gating architectural lower bound; ungated "
            "retains the mapped-DC all-module upper bound."
        ),
    )
    parser.add_argument(
        "--external-memory-energy-artifact",
        type=Path,
        default=DEFAULT_EXTERNAL_MEMORY_ENERGY,
        help=(
            "Literature-parameterized external-memory energy artifact used "
            "by the power shadow"
        ),
    )
    parser.add_argument(
        "--sram-background-energy-artifact",
        type=Path,
        default=DEFAULT_SRAM_BACKGROUND_ENERGY,
        help=(
            "Capacity-proportional SRAM background-power artifact; dynamic "
            "SRAM accesses continue to use ASAP7 Liberty internal power"
        ),
    )
    parser.add_argument(
        "--interconnect-energy-artifact",
        type=Path,
        default=DEFAULT_INTERCONNECT_ENERGY,
        help="Literature proxy used for internal multi-chip link energy",
    )
    parser.add_argument("--keep-rtl-config", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--global-cache-dir",
        type=Path,
        default=DEFAULT_GLOBAL_DSE_CACHE_DIR,
        help=(
            "Versioned content-addressed CostEmitter/area cache shared by "
            "independent studies"
        ),
    )
    parser.add_argument(
        "--run-local-cache",
        action="store_true",
        help=(
            "Disable cross-study reuse and place the same versioned cache "
            "inside this run directory"
        ),
    )
    parser.add_argument(
        "--sampler",
        choices=("tpe", "nsga2", "grid"),
        default="tpe",
        help=(
            "Optuna sampler. Multi-objective TPE is the default; NSGA-II and "
            "the exhaustive grid remain available for compatibility."
        ),
    )
    parser.add_argument(
        "--search-encoding",
        choices=SEARCH_ENCODINGS,
        default="canonical-conditional-v1",
        help="Sample only legal/canonical physical configurations or reproduce legacy pruning",
    )
    parser.add_argument(
        "--precision-search-encoding",
        choices=PRECISION_SEARCH_ENCODINGS,
        default="hardware-signature-v1",
        help=(
            "Group precision profiles by Matrix PE operand ports and output "
            "format before conditionally sampling KV/accuracy variants, or "
            "reproduce the legacy flat profile category"
        ),
    )
    parser.add_argument(
        "--tpe-startup-trials",
        type=int,
        default=DEFAULT_TPE_STARTUP_TRIALS,
    )
    parser.add_argument(
        "--tpe-ei-candidates",
        type=int,
        default=DEFAULT_TPE_EI_CANDIDATES,
    )
    parser.add_argument("--target-area-mm2", type=float, default=None)
    parser.add_argument(
        "--area-budget-mm2",
        type=float,
        default=None,
        help=(
            "Aggregate silicon feasibility constraint; defaults to "
            "R * 826 * 1.10 mm2"
        ),
    )
    parser.add_argument(
        "--target-area-tolerance-pct", type=float, default=DEFAULT_TARGET_AREA_TOLERANCE_PCT
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Parallel Optuna worker processes; default auto uses min(logical CPUs, n-trials)",
    )
    parser.add_argument("--worker-mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-trials", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-max-trials-per-process",
        type=int,
        default=0,
        help=(
            "Recycle each parallel worker after this many trials; 0 disables "
            "count-based recycling and relies on RSS/memory guards (default: 0)"
        ),
    )
    parser.add_argument(
        "--worker-rss-recycle-gib",
        type=float,
        default=DEFAULT_WORKER_RSS_RECYCLE_GIB,
        help="Recycle a worker after a trial when its peak RSS reaches this threshold",
    )
    parser.add_argument(
        "--initial-worker-rss-gib",
        type=float,
        default=DEFAULT_INITIAL_WORKER_RSS_GIB,
        help=(
            "Memory token reserved for a newly spawned worker before observed "
            "RSS measurements are available"
        ),
    )
    parser.add_argument(
        "--memory-reserve-gib",
        type=float,
        default=DEFAULT_MEMORY_RESERVE_GIB,
        help="Do not launch replacement workers below this MemAvailable reserve",
    )
    parser.add_argument(
        "--memory-resume-gib",
        type=float,
        default=DEFAULT_MEMORY_RESUME_GIB,
        help="Resume filling the worker pool after memory pressure clears",
    )
    parser.add_argument(
        "--memory-emergency-gib",
        type=float,
        default=DEFAULT_MEMORY_EMERGENCY_GIB,
        help=(
            "Terminate the largest worker only below this early-OOM safety "
            "floor; launch reserve/resume thresholds never kill workers"
        ),
    )
    parser.add_argument(
        "--worker-process-tree-rss-limit-gib",
        type=float,
        default=DEFAULT_PROCESS_TREE_RSS_LIMIT_GIB,
        help="Terminate and requeue a worker whose process-tree RSS exceeds this limit",
    )
    parser.add_argument(
        "--worker-stall-timeout-seconds",
        "--worker-phase-timeout-seconds",
        dest="worker_stall_timeout_seconds",
        type=float,
        default=DEFAULT_WORKER_STALL_TIMEOUT_SECONDS,
        help=(
            "Terminate and requeue only after this many seconds without "
            "progress and at least 120 seconds below 5%% CPU"
        ),
    )
    parser.add_argument(
        "--worker-launch-burst",
        type=int,
        default=DEFAULT_WORKER_LAUNCH_BURST,
        help="Maximum worker processes started in one admission burst",
    )
    parser.add_argument(
        "--worker-launch-interval-seconds",
        type=float,
        default=DEFAULT_WORKER_LAUNCH_INTERVAL_SECONDS,
        help="Delay between worker admission bursts",
    )
    parser.add_argument(
        "--worker-monitor-interval-seconds",
        type=float,
        default=DEFAULT_WORKER_MONITOR_INTERVAL_SECONDS,
        help="Process-tree RSS, heartbeat, and pool-CPU sampling interval",
    )
    parser.add_argument(
        "--optuna-storage",
        choices=("auto", "journal", "sqlite"),
        default="auto",
        help=(
            "Persistent study backend. auto uses SQLite WAL for parallel runs "
            "and JournalStorage for sequential runs."
        ),
    )
    args = parser.parse_args()
    args.model_config = args.model_config.resolve()
    model = load_json(args.model_config)
    is_moe_model = int(model.get("num_experts", 0) or 0) > 0
    if is_moe_model:
        if args.moe_routing_mode != "fixed-balanced":
            raise ValueError(
                "Formal MoE DSE requires --moe-routing-mode fixed-balanced; "
                "static-indices requires an explicit per-input routing plan"
            )
        if args.moe_layer_scaling != "repeat-fixed-balanced":
            raise ValueError(
                "Multi-layer MoE DSE requires "
                "--moe-layer-scaling repeat-fixed-balanced"
            )
    elif (
        args.moe_routing_mode != "static-indices"
        or args.moe_layer_scaling != "single-layer"
    ):
        raise ValueError(
            "Dense DSE requires static-indices routing mode and single-layer "
            "MoE scaling semantics"
        )
    profile_consistent, profile_mismatches = model_profile_consistency(args)
    if not profile_consistent:
        raise ValueError(
            f"--model-profile {args.model_profile} is inconsistent with "
            + ", ".join(profile_mismatches)
            + "; select --model-profile custom for an explicit A/B run"
        )
    selected_profile_fidelity = (
        CURRENT_DSE_PROFILE.fidelity
        if args.model_profile == CURRENT_DSE_PROFILE.name
        else RTL_VALIDATION_PROFILE.fidelity
        if args.model_profile == RTL_VALIDATION_PROFILE.name
        else "custom_ab_configuration"
    )
    if args.artifact_retention == "full":
        args.trial_report_materialization = "full"
    if (
        args.compiler_compute_timing == "ideal-ii1"
        and args.compiler_scheduled_shadow
    ):
        raise ValueError(
            "--compiler-scheduled-shadow requires "
            "--compiler-compute-timing rtl-v1"
        )
    if args.compiler_trace_granularity == "affine-block-summary-v1":
        if args.compiler_compute_timing != "ideal-ii1":
            raise ValueError(
                "--compiler-trace-granularity affine-block-summary-v1 "
                "requires --compiler-compute-timing ideal-ii1"
            )
        if args.compiler_scheduled_shadow:
            raise ValueError(
                "affine-block-summary-v1 cannot be used with scheduled shadow"
            )
        if args.compiler_v4_memory_evaluation not in {
            "auto",
            "one-layer-cached-occurrence-scaled",
        }:
            raise ValueError(
                "affine-block-summary-v1 requires cached-occurrence V4"
            )
    if (
        args.compiler_compute_timing == "legacy"
        and args.address_generation_mode != "legacy"
    ):
        raise ValueError(
            "--compiler-compute-timing legacy requires "
            "--address-generation-mode legacy"
        )
    if (
        args.packed_qk_schedule == PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1
        and args.softmax_state_schedule
        not in {
            SOFTMAX_STATE_SCHEDULE_STREAMED_V2,
            SOFTMAX_STATE_SCHEDULE_ROW_BANK_SIMD_V3,
        }
    ):
        raise ValueError(
            "--packed-qk-schedule broadcast-k-major-v1 requires "
            "--softmax-state-schedule streamed-v2 or row-bank-simd-v3"
        )
    rtl_v6_requested = args.vector_scalar_schedule == "rtl-v6"
    softmax_row_lane_domain = (
        parse_positive_int_csv(args.softmax_row_lanes)
        if args.softmax_row_lanes
        else ((2, 4, 8) if rtl_v6_requested else (1,))
    )
    if any(value not in {1, 2, 4, 8} for value in softmax_row_lane_domain):
        raise ValueError("--softmax-row-lanes must be drawn from 1,2,4,8")
    if args.fixed_softmax_row_lanes is not None:
        softmax_row_lane_domain = (args.fixed_softmax_row_lanes,)
    if rtl_v6_requested:
        if args.softmax_vector_schedule != "multi-row-v1":
            raise ValueError("rtl-v6 DSE requires --softmax-vector-schedule multi-row-v1")
        if args.softmax_state_schedule != SOFTMAX_STATE_SCHEDULE_ROW_BANK_SIMD_V3:
            raise ValueError("rtl-v6 DSE requires --softmax-state-schedule row-bank-simd-v3")
        if args.pv_accumulation_schedule != "direct-packed-rmw-v1":
            raise ValueError("rtl-v6 DSE requires --pv-accumulation-schedule direct-packed-rmw-v1")
    elif (
        args.softmax_vector_schedule != "single-row-v1"
        or args.pv_accumulation_schedule != "shift-add-v1"
        or args.softmax_state_schedule == SOFTMAX_STATE_SCHEDULE_ROW_BANK_SIMD_V3
        or softmax_row_lane_domain != (1,)
    ):
        raise ValueError(
            "multi-row/state-bank/direct-PV and row lanes >1 require --vector-scalar-schedule rtl-v6"
        )
    if args.dry_run:
        args.area_mode = "none"
    if (
        args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
        and args.sampler == "grid"
    ):
        raise ValueError(
            "the formal multi-chip model uses a chip-count-conditional topology "
            "and is supported by TPE/NSGA-II, not the legacy Cartesian grid sampler"
        )
    if args.reference_a100_count <= 0:
        raise ValueError(
            "--reference-a100-count must be positive, got "
            f"{args.reference_a100_count}"
        )
    chip_count_search_values = parse_positive_int_csv(args.chip_counts)
    nvlink_port_counts = parse_positive_int_csv(args.nvlink_port_counts)
    if any(value not in DEFAULT_NVLINK_PORT_COUNTS for value in nvlink_port_counts):
        raise ValueError(
            "--nvlink-port-counts must be drawn from "
            f"{DEFAULT_NVLINK_PORT_COUNTS}"
        )
    if args.fixed_nvlink_port_count is not None:
        nvlink_port_counts = (args.fixed_nvlink_port_count,)
    requested_tp_degrees = (
        None
        if str(args.tp_degrees).strip().lower() == "auto"
        else parse_positive_int_csv(args.tp_degrees)
    )
    requested_dp_degrees = (
        None
        if str(args.dp_degrees).strip().lower() == "auto"
        else parse_positive_int_csv(args.dp_degrees)
    )
    requested_ep_degrees = (
        None
        if str(args.ep_degrees).strip().lower() == "auto"
        else parse_positive_int_csv(args.ep_degrees)
    )
    base_matrix_sram_tiles = parse_positive_int_csv(args.matrix_sram_tiles)
    matrix_sram_policies = tuple(
        dict.fromkeys(
            item.strip()
            for item in args.matrix_sram_policies.split(",")
            if item.strip()
        )
    )
    if not matrix_sram_policies or any(
        item not in MATRIX_SRAM_POLICIES for item in matrix_sram_policies
    ):
        raise ValueError(
            "--matrix-sram-policies must contain values from "
            f"{MATRIX_SRAM_POLICIES}, got {matrix_sram_policies}"
        )
    parallel_models = (
        ("dp-tp-ep",)
        if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL
        else ("tp-cp",)
        if args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
        else (
            PARALLEL_MODELS
            if args.parallel_model == "both"
            else (args.parallel_model,)
        )
    )
    search_encoding = (
        "legacy-policy-v1"
        if args.sampler == "grid"
        else args.search_encoding
    )
    effective_search_schema = (
        (
            TILE_AWARE_SEARCH_SCHEMA
            if args.multi_chip_model in {
                "tile-aware-tp-cp-ep-v3",
                TILE_AWARE_DP_MULTI_CHIP_MODEL,
            }
            else SEARCH_SCHEMA
        )
        if search_encoding == "canonical-conditional-v1"
        else "adaptive_hardware_matrix_sram_policy_v4"
    )
    if (
        args.sampler != "grid"
        and args.precision_search_encoding == "hardware-signature-v1"
    ):
        effective_search_schema = (
            f"{effective_search_schema}_precision_signature_v1"
        )
    objective_schema = (
        TILE_AWARE_OBJECTIVE_SCHEMA
        if args.multi_chip_model in {
            "tile-aware-tp-cp-ep-v3",
            TILE_AWARE_DP_MULTI_CHIP_MODEL,
        }
        else FRACTIONAL_OBJECTIVE_SCHEMA
    )
    latency_model_name = (
        TILE_AWARE_LATENCY_MODEL_NAME
        if args.multi_chip_model in {
            "tile-aware-tp-cp-ep-v3",
            TILE_AWARE_DP_MULTI_CHIP_MODEL,
        }
        else FRACTIONAL_LATENCY_MODEL_NAME
    )
    if args.nvlink_startup_us < 0:
        raise ValueError("--nvlink-startup-us must be nonnegative")
    decode_chip_count = (
        args.reference_a100_count
        if args.decode_chip_count is None
        else args.decode_chip_count
    )
    if decode_chip_count <= 0:
        raise ValueError("--decode-chip-count must be positive")
    if args.fixed_chip_count is not None:
        if args.fixed_chip_count <= 0:
            raise ValueError("--fixed-chip-count must be positive")
        chip_count_search_values = (args.fixed_chip_count,)
    chip_counts = scale_chip_counts_for_reference(
        chip_count_search_values,
        reference_a100_count=args.reference_a100_count,
        mode=args.chip_count_scaling,
    )
    chip_count_search_value_by_physical = dict(
        zip(chip_counts, chip_count_search_values, strict=True)
    )
    fixed_physical_chip_count = (
        chip_counts[0] if args.fixed_chip_count is not None else None
    )

    def run_mlen_values(chips: int) -> tuple[int, ...]:
        """Return shape-legal MLENs after applying explicit fixed knobs."""

        values = valid_mlen_values(int(chips))
        if args.fixed_blen is not None:
            values = tuple(
                mlen
                for mlen in values
                if int(args.fixed_blen) in valid_blen_values(mlen)
            )
        if not values:
            raise ValueError(
                f"no MLEN supports fixed BLEN={args.fixed_blen} for "
                f"chip_count={chips}"
            )
        return values

    def run_mlen_log2_values(chips: int) -> tuple[int, ...]:
        return tuple(
            int(math.log2(value)) for value in run_mlen_values(chips)
        )

    legal_chip_counts: list[int] = []
    if args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS:
        for chips in chip_counts:
            if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL:
                topologies = valid_dp_tp_ep_topologies(
                    model,
                    int(chips),
                    int(args.latency_batch_size),
                    routing_mode=args.moe_routing_mode,
                )
                topologies = tuple(
                    item
                    for item in topologies
                    if (requested_dp_degrees is None or item[0] in requested_dp_degrees)
                    and (args.fixed_dp_degree is None or item[0] == args.fixed_dp_degree)
                    and (requested_tp_degrees is None or item[1] in requested_tp_degrees)
                    and (args.fixed_tp_degree is None or item[1] == args.fixed_tp_degree)
                    and (requested_ep_degrees is None or item[2] in requested_ep_degrees)
                    and (args.fixed_ep_degree is None or item[2] == args.fixed_ep_degree)
                )
                if not topologies:
                    if args.fixed_chip_count is not None:
                        raise ValueError(
                            f"no legal DP/TP/EP topology for N={chips}, "
                            f"batch={args.latency_batch_size}"
                        )
                    continue
                legal_chip_counts.append(int(chips))
                continue
            legal = valid_tp_degrees(model, int(chips))
            if requested_tp_degrees is not None:
                legal = tuple(tp for tp in legal if tp in requested_tp_degrees)
            if args.fixed_tp_degree is not None:
                legal = tuple(tp for tp in legal if tp == args.fixed_tp_degree)
            if not legal:
                if args.fixed_chip_count is not None:
                    raise ValueError(
                        f"no legal TP degree for chip_count={chips}; natural "
                        f"domain={valid_tp_degrees(model, int(chips))}"
                    )
                continue
            legal_chip_counts.append(int(chips))
        chip_counts = tuple(legal_chip_counts)
        if not chip_counts:
            raise ValueError(
                "no chip count has a legal parallel topology for the requested "
                f"batch={args.latency_batch_size} and DP/TP/EP domains"
            )
    if not is_moe_model and args.fixed_ep_degree not in {None, 1}:
        raise ValueError(
            "Dense models require --fixed-ep-degree 1"
        )
    if (
        not is_moe_model
        and requested_ep_degrees is not None
        and 1 not in requested_ep_degrees
    ):
        raise ValueError("Dense DSE requires EP degree 1")
    if args.fixed_mlen is not None:
        if args.fixed_mlen not in DEFAULT_SEARCH_SPACE["MLEN"]:
            raise ValueError(
                f"--fixed-mlen must be one of {DEFAULT_SEARCH_SPACE['MLEN']}, "
                f"got {args.fixed_mlen}"
            )
        if args.fixed_chip_count is not None:
            if args.fixed_mlen not in run_mlen_values(
                fixed_physical_chip_count
            ):
                raise ValueError(
                    f"MLEN={args.fixed_mlen} is outside the canonical domain "
                    f"for physical chip_count={fixed_physical_chip_count}"
                )
        else:
            chip_counts = tuple(
                chips
                for chips in chip_counts
                if args.fixed_mlen in run_mlen_values(chips)
            )
            if not chip_counts:
                raise ValueError(
                    f"no requested chip count supports MLEN={args.fixed_mlen}"
                )
    if args.fixed_blen is not None:
        candidate_mlens = (
            (args.fixed_mlen,)
            if args.fixed_mlen is not None
            else tuple(DEFAULT_SEARCH_SPACE["MLEN"])
        )
        if not any(
            args.fixed_blen in valid_blen_values(mlen)
            for mlen in candidate_mlens
        ):
            raise ValueError(
                f"--fixed-blen={args.fixed_blen} is outside the canonical "
                "BLEN domain"
            )
    if args.fixed_matrix_sram_tiles is not None and args.fixed_matrix_sram_tiles < 2:
        raise ValueError("--fixed-matrix-sram-tiles must be at least 2")
    if (
        args.fixed_matrix_sram_tiles is not None
        and args.fixed_matrix_sram_policy is not None
    ):
        raise ValueError(
            "--fixed-matrix-sram-tiles and --fixed-matrix-sram-policy "
            "are mutually exclusive"
        )
    if args.fixed_matrix_sram_policy is not None:
        matrix_sram_policies = (args.fixed_matrix_sram_policy,)
    if args.endpoint_area_overhead_pct < 0:
        raise ValueError("--endpoint-area-overhead-pct must be nonnegative")
    if args.nvlink_bandwidth_gbps <= 0:
        raise ValueError("--nvlink-bandwidth-gbps must be positive")
    args.legacy_bandwidth_prune = (
        args.legacy_bandwidth_policy == "strict"
        if args.legacy_bandwidth_prune is None
        else bool(args.legacy_bandwidth_prune)
    )
    args.legacy_bandwidth_policy = (
        "strict" if args.legacy_bandwidth_prune else "diagnostic"
    )
    if args.target_area_mm2 is None:
        args.target_area_mm2 = (
            GA100_REFERENCE_AREA_MM2 * args.reference_a100_count
        )
    if args.area_budget_mm2 is None:
        args.area_budget_mm2 = (
            GA100_REFERENCE_AREA_MM2
            * 1.10
            * args.reference_a100_count
        )
    if args.latency_batch_size <= 0:
        raise ValueError(f"--latency-batch-size must be positive, got {args.latency_batch_size}")
    if args.frequency_ghz <= 0:
        raise ValueError(f"--frequency-ghz must be positive, got {args.frequency_ghz}")
    if args.mx_scale_block_size <= 0:
        raise ValueError(f"--mx-scale-block-size must be positive, got {args.mx_scale_block_size}")
    if args.hbm_capacity_bytes <= 0:
        raise ValueError(f"--hbm-capacity-bytes must be positive, got {args.hbm_capacity_bytes}")
    if args.kv_handoff_staging_layers < 0:
        raise ValueError("--kv-handoff-staging-layers must be nonnegative")
    if args.power_shadow and not args.external_memory_energy_artifact.exists():
        raise FileNotFoundError(
            "external-memory energy artifact does not exist: "
            f"{args.external_memory_energy_artifact}"
        )
    if args.power_shadow and not args.sram_background_energy_artifact.exists():
        raise FileNotFoundError(
            "SRAM background energy artifact does not exist: "
            f"{args.sram_background_energy_artifact}"
        )
    if not args.power_shadow:
        raise ValueError(
            "the latency-energy DSE requires --power-shadow because system "
            "energy is a formal objective"
        )
    if not args.interconnect_energy_artifact.exists():
        raise FileNotFoundError(
            "interconnect-energy artifact does not exist: "
            f"{args.interconnect_energy_artifact}"
        )
    if args.tpe_startup_trials <= 0 or args.tpe_ei_candidates <= 0:
        raise ValueError("TPE startup trials and EI candidates must be positive")
    if args.n_trials is not None and args.target_complete_trials is not None:
        raise ValueError(
            "--n-trials and --target-complete-trials are mutually exclusive"
        )
    if args.n_trials is None and args.target_complete_trials is None:
        args.n_trials = DEFAULT_OPTUNA_TRIALS
    if args.n_trials is not None and args.n_trials <= 0:
        raise ValueError(f"--n-trials must be positive, got {args.n_trials}")
    if (
        args.target_complete_trials is not None
        and args.target_complete_trials <= 0
    ):
        raise ValueError("--target-complete-trials must be positive")
    if args.max_total_attempts is not None and args.max_total_attempts <= 0:
        raise ValueError("--max-total-attempts must be positive")
    if args.target_complete_trials is not None:
        if args.sampler == "grid":
            raise ValueError("--target-complete-trials is not supported by grid")
        if args.max_total_attempts is None:
            args.max_total_attempts = math.ceil(
                args.target_complete_trials * 1.25
            )
        if args.max_total_attempts < args.target_complete_trials:
            raise ValueError(
                "--max-total-attempts must be at least "
                "--target-complete-trials"
            )
    if args.worker_max_trials_per_process < 0:
        raise ValueError(
            "--worker-max-trials-per-process must be nonnegative, got "
            f"{args.worker_max_trials_per_process}"
        )
    if args.worker_rss_recycle_gib <= 0:
        raise ValueError("--worker-rss-recycle-gib must be positive")
    if args.initial_worker_rss_gib <= 0:
        raise ValueError("--initial-worker-rss-gib must be positive")
    if args.memory_reserve_gib < 0:
        raise ValueError("--memory-reserve-gib must be nonnegative")
    if args.memory_resume_gib < args.memory_reserve_gib:
        raise ValueError(
            "--memory-resume-gib must be at least --memory-reserve-gib"
        )
    if not 0 <= args.memory_emergency_gib < args.memory_reserve_gib:
        raise ValueError(
            "--memory-emergency-gib must be nonnegative and below "
            "--memory-reserve-gib"
        )
    if args.worker_stall_timeout_seconds <= 0:
        raise ValueError("--worker-stall-timeout-seconds must be positive")
    if args.worker_launch_burst <= 0:
        raise ValueError("--worker-launch-burst must be positive")
    if (
        args.worker_launch_interval_seconds <= 0
        or args.worker_monitor_interval_seconds <= 0
    ):
        raise ValueError("worker launch and monitor intervals must be positive")
    if args.min_matrix_k_splits <= 0:
        raise ValueError(
            "--min-matrix-k-splits must be positive, got "
            f"{args.min_matrix_k_splits}"
        )
    if args.target_area_mm2 <= 0:
        raise ValueError(f"--target-area-mm2 must be positive, got {args.target_area_mm2}")
    if args.area_budget_mm2 <= 0:
        raise ValueError(f"--area-budget-mm2 must be positive, got {args.area_budget_mm2}")
    if args.target_area_tolerance_pct < 0:
        raise ValueError("--target-area-tolerance-pct must be nonnegative")
    if args.compiler_cost_mode != "off":
        missing = [
            name
            for name, value in (
                ("--compiler-cost-settings", args.compiler_cost_settings),
                ("--compiler-cost-calibration", args.compiler_cost_calibration),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"--compiler-cost-mode {args.compiler_cost_mode} requires " + ", ".join(missing)
            )
    if args.fixed_mlen is not None and args.fixed_vlen is not None and args.fixed_mlen != args.fixed_vlen:
        raise ValueError("--fixed-vlen must match --fixed-mlen when VLEN is tied to MLEN")
    if args.area_mode in {"synth", "elaborate", "parse-existing"} and not args.worker_mode:
        requested_workers = (
            logical_cpu_capacity()
            if args.workers == "auto"
            else int(args.workers)
        )
        if requested_workers != 1:
            raise ValueError(f"--area-mode {args.area_mode} requires --workers 1 because PLENA_RTL is shared")

    if args.weight_precision is None:
        if float(args.weight_element_bits).is_integer() and int(args.weight_element_bits) in (4, 8):
            effective_weight_precision = f"MXINT{int(args.weight_element_bits)}"
        else:
            effective_weight_precision = DEFAULT_WEIGHT_PRECISION
    else:
        effective_weight_precision = args.weight_precision
    effective_weight_bits = weight_precision_element_bits(effective_weight_precision, args.weight_element_bits)

    dse_config = DSEConfig(
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
        device_num=args.device_num,
        latency_batch_size=args.latency_batch_size,
        hbm_capacity_bytes=(
            args.hbm_capacity_bytes * args.reference_a100_count
        ),
        hbm_bandwidth_gbps=(
            args.hbm_bandwidth_gbps * args.reference_a100_count
        ),
        frequency_ghz=args.frequency_ghz,
        mx_scale_width=args.mx_scale_width,
        mx_scale_block_size=args.mx_scale_block_size,
        fp_constant_num=args.fp_constant_num,
        weight_param_count=args.weight_param_count,
        weight_element_bits=effective_weight_bits,
        weight_precision=effective_weight_precision,
        weight_mx_exp_width=args.weight_mx_exp_width,
        weight_mx_mant_width=args.weight_mx_mant_width,
        kv_capacity_mode=args.kv_capacity_mode,
        kv_handoff_staging_layers=args.kv_handoff_staging_layers,
        softmax_state_schedule=args.softmax_state_schedule,
        packed_qk_schedule=args.packed_qk_schedule,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (WORKSPACE_ROOT / "runs" / timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_directories = DSECacheDirectories.create(
        run_dir / "model_cache"
        if args.run_local_cache
        else args.global_cache_dir.expanduser().resolve()
    )
    cache_scope = "run-local" if args.run_local_cache else "cross-study-global"
    from analytic_models.performance.compiler_cost_model import (
        compiler_cost_source_fingerprint,
    )

    compiler_cost_source_hash = compiler_cost_source_fingerprint()
    area_model_source_hash = source_tree_sha256(
        REPO_ROOT / "analytic_models" / "area_new"
    )
    trials_jsonl = run_dir / (
        f"trials.worker_{args.worker_id:03d}.jsonl" if args.worker_mode else "trials.jsonl"
    )
    cache_path = run_dir / (
        f"area_cache.worker_{args.worker_id:03d}.json" if args.worker_mode else "area_cache.json"
    )
    cache: dict[str, Any] = {}
    compiler_cost_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
    compiler_report_metadata_cache: OrderedDict[
        str, dict[str, Any]
    ] = OrderedDict()

    def cached_area_report(
        key: str,
        compute: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        """Load or atomically publish one cross-worker area estimate."""

        if key in cache:
            return cache[key], "process"
        path = cache_entry_path(cache_directories.area_reports, key)
        lock_path = cache_directories.area_reports / f"{key}.lock"
        with lock_path.open("a+") as cache_lock:
            fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX)
            try:
                metrics = load_cached_json(path)
                tier = (
                    "global-json"
                    if metrics is not None
                    and cache_scope == "cross-study-global"
                    else "run-local-json"
                    if metrics is not None
                    else "cold"
                )
                if metrics is None:
                    path.unlink(missing_ok=True)
                    metrics = compute()
                    write_json(path, metrics)
                cache[key] = metrics
                return metrics, tier
            finally:
                fcntl.flock(cache_lock.fileno(), fcntl.LOCK_UN)

    def legal_tp_options(chips: int) -> tuple[int, ...]:
        options = valid_tp_degrees(model, int(chips))
        if requested_tp_degrees is not None:
            options = tuple(
                tp for tp in options if tp in requested_tp_degrees
            )
        if args.fixed_tp_degree is not None:
            options = tuple(
                tp for tp in options if tp == args.fixed_tp_degree
            )
        if not options:
            raise ValueError(f"no legal TP degree for chip_count={chips}")
        return options

    def legal_parallel_topologies(
        chips: int,
    ) -> tuple[tuple[int, int, int], ...]:
        if args.multi_chip_model != TILE_AWARE_DP_MULTI_CHIP_MODEL:
            raise ValueError(
                "DP/TP/EP topology enumeration is only valid for v4"
            )
        options = valid_dp_tp_ep_topologies(
            model,
            int(chips),
            dse_config.latency_batch_size,
            routing_mode=args.moe_routing_mode,
        )
        return tuple(
            topology
            for topology in options
            if (
                requested_dp_degrees is None
                or topology[0] in requested_dp_degrees
            )
            and (
                args.fixed_dp_degree is None
                or topology[0] == args.fixed_dp_degree
            )
            and (
                requested_tp_degrees is None
                or topology[1] in requested_tp_degrees
            )
            and (
                args.fixed_tp_degree is None
                or topology[1] == args.fixed_tp_degree
            )
            and (
                requested_ep_degrees is None
                or topology[2] in requested_ep_degrees
            )
            and (
                args.fixed_ep_degree is None
                or topology[2] == args.fixed_ep_degree
            )
        )

    def preferred_parallel_topology(
        chips: int,
        target_tp: int,
    ) -> tuple[int, int, int]:
        options = legal_parallel_topologies(chips)
        if not options:
            raise ValueError(f"no legal DP/TP/EP topology for N={chips}")
        return min(
            options,
            key=lambda item: (
                abs(item[1] - target_tp),
                -item[0],
                item[2],
                item,
            ),
        )

    precision_profiles = load_accuracy(
        args.accuracy_constraints,
        fallback_weight_precision=dse_config.weight_precision,
        min_accuracy=args.min_accuracy,
    )
    if args.fixed_precision_profile is not None:
        precision_profiles = [
            profile for profile in precision_profiles if profile["name"] == args.fixed_precision_profile
        ]
        if not precision_profiles:
            raise ValueError(f"unknown --fixed-precision-profile {args.fixed_precision_profile!r}")
    precision_by_name = {profile["name"]: profile for profile in precision_profiles}
    precision_search_encoding = (
        "profile-categorical-v1"
        if args.sampler == "grid"
        else args.precision_search_encoding
    )
    (
        matrix_datapath_signatures,
        profile_to_datapath_signature,
    ) = build_matrix_datapath_signatures(
        precision_profiles,
        default_scale_width=dse_config.mx_scale_width,
    )
    matrix_datapath_signature_by_id = {
        signature.signature_id: signature
        for signature in matrix_datapath_signatures
    }
    precision_profiles_by_signature = {
        signature.signature_id: tuple(
            precision_by_name[name] for name in signature.profile_names
        )
        for signature in matrix_datapath_signatures
    }
    search_space = {key: list(values) for key, values in DEFAULT_SEARCH_SPACE.items()}
    matrix_sram_search_space = matrix_sram_search_values(
        model,
        mlens=(
            [args.fixed_mlen]
            if args.fixed_mlen is not None
            else search_space["MLEN"]
        ),
        seq_len=dse_config.input_seq_len,
        chip_counts=chip_counts,
        parallel_models=(
            ("tp-sp",)
            if args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
            else parallel_models
        ),
        base_values=base_matrix_sram_tiles,
    )
    canonical_hardware_domain_size = 0
    for domain_mlen in (
        [args.fixed_mlen]
        if args.fixed_mlen is not None
        else search_space["MLEN"]
    ):
        supporting_chip_counts = tuple(
            chips
            for chips in chip_counts
            if int(domain_mlen) in run_mlen_values(chips)
        )
        legal_blen_count = (
            1
            if args.fixed_blen is not None
            else len(valid_blen_log2_values(int(domain_mlen)))
        )
        for domain_chips in supporting_chip_counts:
            for domain_parallel in parallel_models:
                if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL:
                    topology_options: tuple[
                        tuple[int | None, int | None, int | None], ...
                    ] = legal_parallel_topologies(int(domain_chips))
                elif args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS:
                    tp_options = legal_tp_options(int(domain_chips))
                    topology_options = tuple(
                        (
                            int(domain_chips) // int(tp),
                            int(tp),
                            1,
                        )
                        for tp in tp_options
                    )
                else:
                    topology_options = ((None, None, None),)
                for domain_dp, domain_tp, domain_ep in topology_options:
                    domain_cp = (
                        int(domain_chips) // int(domain_tp)
                        if domain_tp is not None
                        and args.multi_chip_model
                        != TILE_AWARE_DP_MULTI_CHIP_MODEL
                        else None
                    )
                    requirements = matrix_sram_requirements(
                        model,
                        mlen=int(domain_mlen),
                        seq_len=dse_config.input_seq_len,
                        chip_count=int(domain_chips),
                        parallel_model=domain_parallel,
                        cp_degree=domain_cp,
                        tp_degree=int(domain_tp or 1),
                        multi_chip_model=args.multi_chip_model,
                    )
                    k_blocks = math.ceil(
                        int(requirements["local_attention_seq_len"])
                        / int(domain_mlen)
                    )
                    sram_count = (
                        1
                        if args.fixed_matrix_sram_tiles is not None
                        or args.fixed_matrix_sram_policy is not None
                        else len(
                            canonical_sram_choices(
                                policies=matrix_sram_policies,
                                k_blocks=k_blocks,
                                mlen=int(domain_mlen),
                                projection_tiles=int(
                                    requirements["projection_threshold_tiles"]
                                ),
                            )
                        )
                    )
                    canonical_hardware_domain_size += (
                        legal_blen_count
                        * len(
                            [args.fixed_int_data_width]
                            if args.fixed_int_data_width is not None
                            else search_space["INT_DATA_WIDTH"]
                        )
                        * sram_count
                        * len(nvlink_port_counts)
                        * len(softmax_row_lane_domain)
                    )
    chip_log2_domain = tuple(int(math.log2(value)) for value in chip_counts)
    use_log2_chip_count = (
        args.sampler == "tpe"
        and all(value > 0 and value & (value - 1) == 0 for value in chip_counts)
        and chip_log2_domain
        == tuple(range(min(chip_log2_domain), max(chip_log2_domain) + 1))
    )
    snapshot = snapshot_rtl_files() if args.area_mode in {"synth", "elaborate"} else {}
    records: list[dict[str, Any]] = []

    if args.sampler == "grid":
        optuna_search_space = {"precision_profile": [p["name"] for p in precision_profiles]}
        if args.fixed_mlen is None:
            optuna_search_space["MLEN"] = search_space["MLEN"]
        if args.fixed_blen is None:
            optuna_search_space["BLEN"] = search_space["BLEN"]
        if args.fixed_int_data_width is None:
            optuna_search_space["INT_DATA_WIDTH"] = search_space["INT_DATA_WIDTH"]
        if args.fixed_chip_count is None:
            optuna_search_space["CHIP_COUNT"] = list(chip_counts)
        if (
            args.fixed_matrix_sram_tiles is None
            and args.fixed_matrix_sram_policy is None
        ):
            optuna_search_space["MATRIX_SRAM_POLICY"] = list(
                matrix_sram_policies
            )
        if len(parallel_models) > 1:
            optuna_search_space["PARALLEL_MODEL"] = list(parallel_models)
        sampler = optuna.samplers.GridSampler(optuna_search_space, seed=args.seed)
        grid_total_trials = math.prod(len(values) for values in optuna_search_space.values())
    elif args.sampler == "nsga2":
        grid_total_trials = None
        sampler = optuna.samplers.NSGAIISampler(
            seed=args.seed + (args.worker_id if args.worker_mode else 0),
            constraints_func=a100_constraints,
        )
    else:
        grid_total_trials = None
        categorical_distance_func: dict[str, Callable[[Any, Any], float]] = {}
        if precision_search_encoding == "profile-categorical-v1":
            categorical_distance_func["precision_profile"] = (
                lambda left, right: precision_profile_distance(
                    str(left), str(right), precision_by_name
                )
            )
        else:
            categorical_distance_func[PRECISION_SIGNATURE_PARAM] = (
                lambda left, right: matrix_datapath_signature_distance(
                    str(left),
                    str(right),
                    matrix_datapath_signature_by_id,
                )
            )
            for signature in matrix_datapath_signatures:
                if len(signature.profile_names) <= 1:
                    continue
                categorical_distance_func[
                    conditional_precision_variant_param_name(
                        signature.signature_id
                    )
                ] = (
                    lambda left, right: precision_variant_distance(
                        str(left), str(right), precision_by_name
                    )
                )
        if search_encoding == "legacy-policy-v1":
            categorical_distance_func["MATRIX_SRAM_POLICY"] = (
                lambda left, right: abs(
                    MATRIX_SRAM_POLICIES.index(str(left))
                    - MATRIX_SRAM_POLICIES.index(str(right))
                )
            )
        sampler = optuna.samplers.TPESampler(
            seed=args.seed + (args.worker_id if args.worker_mode else 0),
            n_startup_trials=args.tpe_startup_trials,
            n_ei_candidates=args.tpe_ei_candidates,
            multivariate=True,
            group=True,
            constant_liar=True,
            constraints_func=a100_constraints,
            categorical_distance_func=categorical_distance_func,
        )
    storage, optuna_storage_backend = create_optuna_storage(
        run_dir,
        requested_backend=args.optuna_storage,
        worker_mode=args.worker_mode,
        workers=args.workers,
    )
    study_name = f"{model.get('model_name', model.get('model_type', 'qwen'))}_dse"
    study = optuna.create_study(
        directions=list(OBJECTIVE_DIRECTIONS),
        sampler=sampler,
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    hardware_domain_fingerprint = stable_key(
        {
            "mlen": search_space["MLEN"],
            "blen": search_space["BLEN"],
            "shape_domain_policy": SHAPE_DOMAIN_POLICY,
            "int_width": search_space["INT_DATA_WIDTH"],
            "chip_counts": chip_counts,
            "chip_count_search_values": chip_count_search_values,
            "chip_count_scaling": args.chip_count_scaling,
            "reference_a100_count": args.reference_a100_count,
            "decode_chip_count": decode_chip_count,
            "decode_nvlink_port_count": args.decode_nvlink_port_count,
            "decode_kv_handoff_schema": "dual_endpoint_peak_bandwidth_v1",
            "parallel_models": parallel_models,
            "multi_chip_model": args.multi_chip_model,
            "dp_degrees": args.dp_degrees,
            "fixed_dp_degree": args.fixed_dp_degree,
            "tp_degrees": args.tp_degrees,
            "fixed_tp_degree": args.fixed_tp_degree,
            "ep_degrees": args.ep_degrees,
            "fixed_ep_degree": args.fixed_ep_degree,
            "parallel_topologies_by_chip": (
                {
                    str(chips): [
                        list(topology)
                        for topology in legal_parallel_topologies(chips)
                    ]
                    for chips in chip_counts
                }
                if args.multi_chip_model
                == TILE_AWARE_DP_MULTI_CHIP_MODEL
                else None
            ),
            "nvlink_port_counts": nvlink_port_counts,
            "nvlink_bandwidth_semantics": args.nvlink_bandwidth_semantics,
            "matrix_sram_policies": matrix_sram_policies,
            "softmax_row_lane_domain": list(softmax_row_lane_domain),
            "softmax_vector_schedule": args.softmax_vector_schedule,
            "pv_accumulation_schedule": args.pv_accumulation_schedule,
            "min_matrix_k_splits": args.min_matrix_k_splits,
            "kv_capacity_mode": dse_config.kv_capacity_mode,
            "kv_handoff_staging_layers": (
                dse_config.kv_handoff_staging_layers
            ),
        }
    )
    expected_study_attrs = {
        "model_profile": args.model_profile,
        "objective_schema": objective_schema,
        "objective_normalization": OBJECTIVE_NORMALIZATION,
        "objective_fields": [
            "normalized_latency",
            "normalized_energy",
        ],
        "search_schema": effective_search_schema,
        "search_encoding": search_encoding,
        "shape_domain_policy": SHAPE_DOMAIN_POLICY,
        "precision_search_encoding": precision_search_encoding,
        "precision_signature_schema": PRECISION_SIGNATURE_SCHEMA,
        "precision_signature_count": len(matrix_datapath_signatures),
        "precision_signature_fingerprint": stable_key(
            [
                {
                    **signature.metadata(),
                    "profile_names": list(signature.profile_names),
                }
                for signature in matrix_datapath_signatures
            ]
        ),
        "input_seq_len": dse_config.input_seq_len,
        "latency_batch_size": dse_config.latency_batch_size,
        "hardware_domain_fingerprint": hardware_domain_fingerprint,
        "model_config_sha256": file_sha256(args.model_config),
        "model_config": str(args.model_config),
        "moe_routing_mode": args.moe_routing_mode,
        "moe_lowering_schedule": args.moe_lowering_schedule,
        "moe_layer_scaling": args.moe_layer_scaling,
        "compiler_compute_timing": args.compiler_compute_timing,
        "compiler_v4_memory_evaluation": args.compiler_v4_memory_evaluation,
        "compiler_trace_granularity": args.compiler_trace_granularity,
        "trial_report_materialization": args.trial_report_materialization,
        "native_layout_mode": args.native_layout_mode,
        "packed_attention_schedule": args.packed_attention_schedule,
        "vector_scalar_schedule": args.vector_scalar_schedule,
        "softmax_vector_schedule": args.softmax_vector_schedule,
        "pv_accumulation_schedule": args.pv_accumulation_schedule,
        "softmax_row_lane_domain": list(softmax_row_lane_domain),
        "compact_stats_lane_policy": (
            "auto-tiered-v1"
            if args.vector_scalar_schedule in {"rtl-v5", "rtl-v6"}
            else "fixed-16-v1"
        ),
        "selector_schedule": args.selector_schedule,
        "reduction_output_mode": args.reduction_output_mode,
        "gqa_pipeline_schedule": args.gqa_pipeline_schedule,
        "address_generation_mode": args.address_generation_mode,
        "ffn_address_schedule": args.ffn_address_schedule,
        "ffn_projection_schedule": args.ffn_projection_schedule,
        "softmax_state_schedule": args.softmax_state_schedule,
        "packed_qk_schedule": args.packed_qk_schedule,
        "matrix_sram_policy_schema": "partial_resident_prefix_v1",
        "sram_port_model": CURRENT_DSE_PROFILE.sram_port_model,
        "clock_gating_mode": args.clock_gating_mode,
        "latency_model": latency_model_name,
        "multi_chip_model": args.multi_chip_model,
        "dp_degree_domain": args.dp_degrees,
        "tp_degree_domain": args.tp_degrees,
        "ep_degree_domain": args.ep_degrees,
        "nvlink_port_counts": list(nvlink_port_counts),
        "nvlink_bandwidth_semantics": args.nvlink_bandwidth_semantics,
        "nvlink_port_bidirectional_gbps": (
            DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS
        ),
        "nvlink_startup_us": args.nvlink_startup_us,
        "kv_capacity_mode": dse_config.kv_capacity_mode,
        "kv_handoff_staging_layers": (
            dse_config.kv_handoff_staging_layers
        ),
        "precision_profile_artifact_sha256": file_sha256(
            args.accuracy_constraints
        ),
        "interconnect_energy_artifact_sha256": file_sha256(
            args.interconnect_energy_artifact
        ),
        "sram_background_energy_artifact_sha256": file_sha256(
            args.sram_background_energy_artifact
        ),
    }
    if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL:
        expected_study_attrs.update(
            {
                "parallel_kernel_census_schema": (
                    "parallel_kernel_census_v2_schedule_lineage"
                ),
                "tile_accounting_schema": (
                    "whole_request_dp_rank_local_compiler_planner_v4"
                ),
                "parallel_topology_schema": (
                    "whole_request_dp_tp_ep_topology_v1"
                ),
                "communication_schema": (
                    "dependency_serial_tp_ring_ep_port_schedule_v1"
                ),
                "energy_action_lineage_schema": (
                    "energy_action_kernel_lineage_v3_structural_families"
                ),
                "rank_power_aggregation_schema": (
                    "sum_rank_energy_after_per_rank_clock_cap_v2"
                ),
            }
        )
    elif args.multi_chip_model == "tile-aware-tp-cp-ep-v3":
        expected_study_attrs.update(
            {
                "parallel_kernel_census_schema": (
                    "parallel_kernel_census_v2_schedule_lineage"
                ),
                "tile_accounting_schema": (
                    "balanced_partition_compiler_planner_v3"
                ),
                "ep_topology_schema": (
                    "ep_reuses_cp_contiguous_expert_partition_v1"
                ),
                "energy_action_lineage_schema": (
                    "energy_action_kernel_lineage_v3_structural_families"
                ),
                "rank_power_aggregation_schema": (
                    "sum_rank_energy_after_per_rank_clock_cap_v2"
                ),
            }
        )
    for attr, expected in expected_study_attrs.items():
        existing = study.user_attrs.get(attr)
        if existing is not None and existing != expected:
            raise ValueError(
                f"study schema mismatch for {attr}: existing={existing!r}, "
                f"requested={expected!r}; use a new run directory"
            )
        if existing is None and not args.worker_mode:
            study.set_user_attr(attr, expected)
    if (
        args.sampler == "tpe"
        and not args.worker_mode
        and not study.get_trials(deepcopy=False)
    ):
        def startup_anchor_mlen(chips: int, requested: int) -> int:
            """Choose a conservative startup point without shrinking DSE.

            The former chip-count MLEN caps were useful for avoiding obviously
            oversized startup points, but were incorrect as hard domain rules
            under per-reference scaling.  Apply the same shape preference to
            the normalized chip multiplier only when constructing anchors.
            """

            normalized_chips = int(
                chip_count_search_value_by_physical[int(chips)]
            )
            preferred_max = (
                2048
                if normalized_chips >= 16
                else 4096
                if normalized_chips >= 8
                else 8192
            )
            candidates = tuple(
                value
                for value in run_mlen_values(int(chips))
                if value <= preferred_max
            )
            return min(
                candidates,
                key=lambda value: (
                    abs(math.log2(value) - math.log2(int(requested))),
                    -value,
                ),
            )

        def anchor_params(
            *,
            profile_name: str,
            mlen: int,
            blen: int,
            int_width: int,
            chips: int,
            sram_policy: str,
            parallel: str,
            tp_degree: int,
            nvlink_ports: int,
        ) -> dict[str, Any]:
            if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL:
                dp_degree, tp_degree, ep_degree = (
                    preferred_parallel_topology(chips, tp_degree)
                )
                cp_degree = 1
            else:
                dp_degree = 1
                ep_degree = 1
                cp_degree = chips // tp_degree
            if precision_search_encoding == "profile-categorical-v1":
                params: dict[str, Any] = {
                    "precision_profile": profile_name,
                }
            else:
                signature_id = profile_to_datapath_signature[profile_name]
                signature = matrix_datapath_signature_by_id[signature_id]
                params = {PRECISION_SIGNATURE_PARAM: signature_id}
                if len(signature.profile_names) > 1:
                    params[
                        conditional_precision_variant_param_name(signature_id)
                    ] = profile_name
            if (
                args.fixed_matrix_sram_tiles is None
                and args.fixed_matrix_sram_policy is None
            ):
                requested_policy = (
                    sram_policy
                    if sram_policy in matrix_sram_policies
                    else matrix_sram_policies[0]
                )
                if search_encoding == "canonical-conditional-v1":
                    requirements = matrix_sram_requirements(
                        model,
                        mlen=mlen,
                        seq_len=dse_config.input_seq_len,
                        chip_count=chips,
                        parallel_model=parallel,
                        cp_degree=(
                            None
                            if args.multi_chip_model
                            == TILE_AWARE_DP_MULTI_CHIP_MODEL
                            else cp_degree
                        ),
                        tp_degree=tp_degree,
                        multi_chip_model=args.multi_chip_model,
                    )
                    k_blocks = math.ceil(
                        int(requirements["local_attention_seq_len"]) / mlen
                    )
                    choices = canonical_sram_choices(
                        policies=matrix_sram_policies,
                        k_blocks=k_blocks,
                        mlen=mlen,
                        projection_tiles=int(
                            requirements["projection_threshold_tiles"]
                        ),
                    )
                    choice_index = next(
                        (
                            int(choice["index"])
                            for choice in choices
                            if requested_policy in choice["policy_aliases"]
                        ),
                        0,
                    )
                    params[
                        conditional_sram_param_name(
                            mlen,
                            chips,
                            parallel,
                            tp_degree=(
                                tp_degree
                                if args.multi_chip_model
                                in FACTORIZED_MULTI_CHIP_MODELS
                                else None
                            ),
                            cp_degree=(
                                cp_degree
                                if args.multi_chip_model
                                in FACTORIZED_MULTI_CHIP_MODELS
                                and args.multi_chip_model
                                != TILE_AWARE_DP_MULTI_CHIP_MODEL
                                else None
                            ),
                            dp_degree=(
                                dp_degree
                                if args.multi_chip_model
                                == TILE_AWARE_DP_MULTI_CHIP_MODEL
                                else None
                            ),
                            ep_degree=(
                                ep_degree
                                if args.multi_chip_model
                                == TILE_AWARE_DP_MULTI_CHIP_MODEL
                                else None
                            ),
                        )
                    ] = choice_index
                else:
                    params["MATRIX_SRAM_POLICY"] = requested_policy
            if args.fixed_mlen is None:
                params[
                    conditional_mlen_param_name(chips)
                    if search_encoding == "canonical-conditional-v1"
                    else "MLEN_LOG2"
                ] = int(math.log2(mlen))
            if args.fixed_blen is None:
                params[
                    conditional_blen_param_name(mlen)
                    if search_encoding == "canonical-conditional-v1"
                    else "BLEN_LOG2"
                ] = int(math.log2(blen))
            if args.fixed_int_data_width is None:
                params["INT_WIDTH_LOG2"] = int(math.log2(int_width))
            if args.fixed_chip_count is None:
                if use_log2_chip_count:
                    params["CHIP_COUNT_LOG2"] = int(math.log2(chips))
                else:
                    params["CHIP_COUNT"] = chips
            if len(parallel_models) > 1:
                params["PARALLEL_MODEL"] = parallel
            if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL:
                topologies = legal_parallel_topologies(chips)
                topology_index = topologies.index(
                    (dp_degree, tp_degree, ep_degree)
                )
                if len(topologies) > 1:
                    params[
                        conditional_parallel_config_param_name(chips)
                    ] = topology_index
            elif (
                args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
                and args.fixed_tp_degree is None
            ):
                params[conditional_tp_param_name(chips)] = tp_degree
            if len(nvlink_port_counts) > 1:
                params["NVLINK_PORT_COUNT"] = nvlink_ports
            return params

        # Evaluate every validated software profile on one identical hardware
        # point first.  These matched observations let the sampler distinguish
        # KV/accuracy effects from PE-port area and compute effects.
        matched_chips = fixed_physical_chip_count or max(chip_counts)
        matched_mlen = args.fixed_mlen or startup_anchor_mlen(
            matched_chips,
            512,
        )
        matched_blens = valid_blen_values(matched_mlen)
        matched_blen = args.fixed_blen or min(
            matched_blens,
            key=lambda value: (abs(value - 64), value),
        )
        matched_tp_options = legal_tp_options(matched_chips)
        matched_tp = (
            args.fixed_tp_degree
            if args.fixed_tp_degree is not None
            else min(
                matched_tp_options,
                key=lambda value: (abs(value - 4), -value),
            )
        )
        for profile in precision_profiles:
            study.enqueue_trial(
                anchor_params(
                    profile_name=profile["name"],
                    mlen=matched_mlen,
                    blen=matched_blen,
                    int_width=args.fixed_int_data_width or 32,
                    chips=matched_chips,
                    sram_policy="streaming",
                    parallel=parallel_models[0],
                    tp_degree=matched_tp,
                    nvlink_ports=min(nvlink_port_counts),
                ),
                user_attrs={
                    "startup_anchor": "precision_matched_hardware_v1"
                },
                skip_if_exists=True,
            )

        # Give every physical PE signature one larger-array observation.  This
        # exposes the central area-reinvestment tradeoff: narrower operand
        # ports may fit more Matrix hardware under the same aggregate budget.
        reinvestment_chips = fixed_physical_chip_count or min(
            chip_counts,
            key=lambda value: (abs(value - 8), -value),
        )
        reinvestment_mlen = args.fixed_mlen or startup_anchor_mlen(
            reinvestment_chips,
            1024,
        )
        reinvestment_blens = valid_blen_values(reinvestment_mlen)
        reinvestment_blen = args.fixed_blen or min(
            reinvestment_blens,
            key=lambda value: (abs(value - 64), value),
        )
        reinvestment_tp_options = legal_tp_options(reinvestment_chips)
        reinvestment_tp = (
            args.fixed_tp_degree
            if args.fixed_tp_degree is not None
            else min(
                reinvestment_tp_options,
                key=lambda value: (abs(value - 4), -value),
            )
        )
        for signature in matrix_datapath_signatures:
            profile_name = min(
                signature.profile_names,
                key=lambda name: (
                    int(
                        parse_mx_precision(
                            precision_by_name[name]["KV_WIDTH"]
                        )["width"]
                    ),
                    -float(precision_by_name[name]["accuracy_score"]),
                    name,
                ),
            )
            study.enqueue_trial(
                anchor_params(
                    profile_name=profile_name,
                    mlen=reinvestment_mlen,
                    blen=reinvestment_blen,
                    int_width=args.fixed_int_data_width or 32,
                    chips=reinvestment_chips,
                    sram_policy="streaming",
                    parallel=parallel_models[0],
                    tp_degree=reinvestment_tp,
                    nvlink_ports=min(nvlink_port_counts),
                ),
                user_attrs={
                    "startup_anchor": "datapath_area_reinvestment_v1"
                },
                skip_if_exists=True,
            )

        # A compact second set varies topology independently of precision.
        # These anchors expose MLEN/BLEN ratio, chip count, and SRAM effects to
        # TPE without multiplying the full hardware/precision Cartesian space.
        hardware_profile = precision_profiles[0]["name"]
        for index, mlen in enumerate(search_space["MLEN"]):
            for ratio in (1, 2, 4, 8, 16):
                chips = fixed_physical_chip_count or chip_counts[
                    (index + int(math.log2(ratio))) % len(chip_counts)
                ]
                requested_mlen = args.fixed_mlen or mlen
                effective_mlen = startup_anchor_mlen(
                    chips,
                    requested_mlen,
                )
                target_blen = max(32, min(1024, effective_mlen // ratio))
                legal_blens = valid_blen_values(effective_mlen)
                blen = args.fixed_blen or min(
                    legal_blens,
                    key=lambda value: (
                        abs(math.log2(value) - math.log2(target_blen)),
                        -value,
                    ),
                )
                study.enqueue_trial(
                    # Rotate over legal TP decompositions for the selected N.
                    anchor_params(
                        profile_name=hardware_profile,
                        mlen=effective_mlen,
                        blen=blen,
                        int_width=args.fixed_int_data_width
                        or search_space["INT_DATA_WIDTH"][
                            index % len(search_space["INT_DATA_WIDTH"])
                        ],
                        chips=chips,
                        sram_policy=(
                            "projection-full" if ratio == 1 else "streaming"
                        ),
                        parallel=parallel_models[
                            index % len(parallel_models)
                        ],
                        tp_degree=(
                            args.fixed_tp_degree
                            if args.fixed_tp_degree is not None
                            else legal_tp_options(chips)[
                                (index + int(math.log2(ratio)))
                                % len(legal_tp_options(chips))
                            ]
                        ),
                        nvlink_ports=nvlink_port_counts[
                            (index + int(math.log2(ratio)))
                            % len(nvlink_port_counts)
                        ],
                    ),
                    user_attrs={"startup_anchor": "hardware_stratified_v3"},
                    skip_if_exists=True,
                )
    reconciliation = (
        reconcile_interrupted_trials(study, run_dir)
        if not args.worker_mode
        else {"recovered_complete": 0, "recovered_pruned": 0, "requeued": 0}
    )
    if any(reconciliation.values()):
        print(f"Reconciled interrupted trials: {reconciliation}")
    initial_finished_trials = sum(
        trial.state.is_finished() for trial in study.get_trials(deepcopy=False)
    )
    initial_complete_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.get_trials(deepcopy=False)
    )
    initial_settled_trials = _settled_trial_count(study)
    complete_budget_mode = (
        args.target_complete_trials is not None and not args.worker_mode
    )
    target_complete_trials = (
        int(args.target_complete_trials)
        if args.target_complete_trials is not None
        else None
    )
    max_total_attempts = (
        int(args.max_total_attempts)
        if args.max_total_attempts is not None
        else None
    )
    if args.worker_mode:
        trials_to_run = int(args.worker_trials or 0)
        target_settled_trials = initial_finished_trials + trials_to_run
    elif grid_total_trials is not None:
        trials_to_run = max(0, grid_total_trials - initial_settled_trials)
        target_settled_trials = grid_total_trials
        # Only the parent process may mutate the shared WAITING queue.  A
        # recycled worker starts after some points have settled; letting that
        # worker finalize duplicate WAITING trials races with peers that are
        # concurrently claiming those trials and can corrupt JournalStorage.
        #
        # Queue the exact missing Cartesian product even for a fresh run.  The
        # workers then consume deterministic WAITING entries instead of asking
        # GridSampler independently, which also avoids its documented
        # distributed duplicate suggestions near the end of a grid.
        if not args.worker_mode and trials_to_run > 0:
            finalize_redundant_waiting_trials(study)
            queued_missing = enqueue_missing_grid_trials(
                study, optuna_search_space
            )
            if queued_missing:
                print(f"Queued {queued_missing} exact missing grid trials")
    elif complete_budget_mode:
        missing_complete = max(
            0, int(target_complete_trials) - initial_complete_trials
        )
        remaining_attempts = max(
            0, int(max_total_attempts) - initial_finished_trials
        )
        trials_to_run = min(missing_complete, remaining_attempts)
        target_settled_trials = initial_finished_trials + trials_to_run
    else:
        trials_to_run = int(args.n_trials or 0)
        target_settled_trials = initial_finished_trials + trials_to_run
    def objective(
        trial: optuna.Trial,
    ) -> tuple[float, float, float, float]:
        trial_dir = run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(exist_ok=True)
        record: dict[str, Any] = {"trial": trial.number, "state": "running"}
        phase_started = time.perf_counter()
        current_phase = "startup"
        phase_seconds: Counter[str] = Counter()
        heartbeat_path = (
            run_dir / f"worker_heartbeat_pid_{os.getpid()}.json"
        )

        def heartbeat(
            phase: str,
            *,
            progress_done: int | None = None,
            progress_total: int | None = None,
            current_stream: int | None = None,
        ) -> None:
            nonlocal current_phase, phase_started
            now = time.perf_counter()
            phase_seconds[current_phase] += now - phase_started
            current_phase = phase
            phase_started = now
            write_json(
                heartbeat_path,
                {
                    "worker_id": args.worker_id,
                    "pid": os.getpid(),
                    "trial": trial.number,
                    "phase": phase,
                    "progress_done": progress_done,
                    "progress_total": progress_total,
                    "current_stream": current_stream,
                    "updated_epoch": time.time(),
                    "current_rss_gib": current_process_rss_gib(),
                    "peak_rss_gib": current_rss_gib(),
                    "mem_available_gib": mem_available_gib(),
                },
            )

        heartbeat("layout")
        try:
            if precision_search_encoding == "profile-categorical-v1":
                precision_name = trial.suggest_categorical(
                    "precision_profile",
                    [p["name"] for p in precision_profiles],
                )
                matrix_datapath_signature_id = (
                    profile_to_datapath_signature[precision_name]
                )
            else:
                matrix_datapath_signature_id = trial.suggest_categorical(
                    PRECISION_SIGNATURE_PARAM,
                    [
                        signature.signature_id
                        for signature in matrix_datapath_signatures
                    ],
                )
                signature_profiles = precision_profiles_by_signature[
                    matrix_datapath_signature_id
                ]
                if len(signature_profiles) == 1:
                    precision_name = str(signature_profiles[0]["name"])
                else:
                    precision_name = trial.suggest_categorical(
                        conditional_precision_variant_param_name(
                            matrix_datapath_signature_id
                        ),
                        [
                            str(profile["name"])
                            for profile in signature_profiles
                        ],
                    )
            precision = precision_by_name[precision_name]
            chip_count = (
                fixed_physical_chip_count
                if fixed_physical_chip_count is not None
                else (
                    1
                    << trial.suggest_int(
                        "CHIP_COUNT_LOG2",
                        min(chip_log2_domain),
                        max(chip_log2_domain),
                    )
                    if use_log2_chip_count
                    else trial.suggest_categorical(
                        "CHIP_COUNT", list(chip_counts)
                    )
                )
            )
            if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL:
                legal_topologies = legal_parallel_topologies(int(chip_count))
                if not legal_topologies:
                    raise TrialPrunedError(
                        f"no legal DP/TP/EP topology for N={chip_count}"
                    )
                topology_index = (
                    0
                    if len(legal_topologies) == 1
                    else trial.suggest_int(
                        conditional_parallel_config_param_name(
                            int(chip_count)
                        ),
                        0,
                        len(legal_topologies) - 1,
                    )
                )
                dp_degree, tp_degree, ep_degree = legal_topologies[
                    topology_index
                ]
                cp_degree = 1
                parallel_model = "dp-tp-ep"
            elif args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS:
                legal_tp = legal_tp_options(int(chip_count))
                if args.fixed_tp_degree is not None:
                    tp_degree = args.fixed_tp_degree
                    if tp_degree not in legal_tp:
                        raise TrialPrunedError(
                            f"TP={tp_degree} is illegal for N={chip_count}"
                        )
                else:
                    tp_degree = trial.suggest_categorical(
                        conditional_tp_param_name(int(chip_count)),
                        list(legal_tp),
                    )
                cp_degree = int(chip_count) // int(tp_degree)
                legal_ep = valid_ep_degrees(
                    model,
                    cp_degree,
                    routing_mode=args.moe_routing_mode,
                )
                if requested_ep_degrees is not None:
                    legal_ep = tuple(
                        ep for ep in legal_ep if ep in requested_ep_degrees
                    )
                if args.fixed_ep_degree is not None:
                    legal_ep = tuple(
                        ep
                        for ep in legal_ep
                        if ep == args.fixed_ep_degree
                    )
                if not legal_ep:
                    raise TrialPrunedError(
                        f"no legal EP degree for TP={tp_degree}, "
                        f"CP={cp_degree}, routing={args.moe_routing_mode}"
                    )
                ep_degree = (
                    legal_ep[0]
                    if len(legal_ep) == 1
                    else trial.suggest_categorical(
                        conditional_ep_param_name(tp_degree, cp_degree),
                        list(legal_ep),
                    )
                )
                dp_degree = 1
                parallel_model = "tp-cp"
            else:
                dp_degree = 1
                tp_degree = int(chip_count)
                ep_degree = 1
                parallel_model = (
                    parallel_models[0]
                    if len(parallel_models) == 1
                    else trial.suggest_categorical(
                        "PARALLEL_MODEL", list(parallel_models)
                    )
                )
                cp_degree = (
                    int(chip_count) if parallel_model == "tp-sp" else 1
                )
            nvlink_port_count = (
                nvlink_port_counts[0]
                if len(nvlink_port_counts) == 1
                else trial.suggest_categorical(
                    "NVLINK_PORT_COUNT",
                    list(nvlink_port_counts),
                )
            )
            mlen = (
                args.fixed_mlen
                if args.fixed_mlen is not None
                else (
                    1
                    << trial.suggest_int(
                        conditional_mlen_param_name(int(chip_count)),
                        min(run_mlen_log2_values(int(chip_count))),
                        max(run_mlen_log2_values(int(chip_count))),
                    )
                    if search_encoding == "canonical-conditional-v1"
                    else (
                        1 << trial.suggest_int("MLEN_LOG2", 8, 13)
                        if args.sampler == "tpe"
                        else trial.suggest_categorical(
                            "MLEN", search_space["MLEN"]
                        )
                    )
                )
            )
            if int(mlen) not in run_mlen_values(int(chip_count)):
                raise TrialPrunedError(
                    f"MLEN={mlen} is outside the canonical domain for "
                    f"chip_count={chip_count}"
                )
            if args.fixed_blen is not None:
                blen = args.fixed_blen
            elif search_encoding == "canonical-conditional-v1":
                legal_blen_logs = valid_blen_log2_values(int(mlen))
                blen = 1 << trial.suggest_int(
                    conditional_blen_param_name(int(mlen)),
                    min(legal_blen_logs),
                    max(legal_blen_logs),
                )
            else:
                blen = (
                    1 << trial.suggest_int("BLEN_LOG2", 5, 10)
                    if args.sampler == "tpe"
                    else trial.suggest_categorical(
                        "BLEN", search_space["BLEN"]
                    )
                )
            params = {
                "MLEN": mlen,
                "BLEN": blen,
                "INT_DATA_WIDTH": (
                    args.fixed_int_data_width
                    if args.fixed_int_data_width is not None
                    else (
                        1 << trial.suggest_int("INT_WIDTH_LOG2", 4, 6)
                        if args.sampler == "tpe"
                        else trial.suggest_categorical(
                            "INT_DATA_WIDTH",
                            search_space["INT_DATA_WIDTH"],
                        )
                    )
                ),
            }
            softmax_row_lanes = (
                softmax_row_lane_domain[0]
                if len(softmax_row_lane_domain) == 1
                else trial.suggest_categorical(
                    "SOFTMAX_ROW_LANES", list(softmax_row_lane_domain)
                )
            )
            params["SOFTMAX_ROW_LANES"] = int(softmax_row_lanes)
            params["VLEN"] = params["MLEN"]
            sram_requirements = matrix_sram_requirements(
                model,
                mlen=params["MLEN"],
                seq_len=dse_config.input_seq_len,
                chip_count=int(chip_count),
                parallel_model=parallel_model,
                cp_degree=(
                    None
                    if args.multi_chip_model
                    == TILE_AWARE_DP_MULTI_CHIP_MODEL
                    else cp_degree
                ),
                tp_degree=int(tp_degree),
                multi_chip_model=args.multi_chip_model,
            )
            local_k_blocks = math.ceil(
                int(sram_requirements["local_attention_seq_len"])
                / int(params["MLEN"])
            )
            matrix_sram_policy_aliases: tuple[str, ...]
            matrix_sram_config_id: str
            if args.fixed_matrix_sram_tiles is not None:
                matrix_sram_policy = "raw-tiles"
                matrix_sram_policy_aliases = ("raw-tiles",)
                residency_plan = plan_kv_residency(
                    k_blocks=local_k_blocks,
                    mlen=int(params["MLEN"]),
                    matrix_sram_tiles=int(args.fixed_matrix_sram_tiles),
                    policy="raw-tiles",
                )
                matrix_sram_config_id = (
                    f"tiles{residency_plan.matrix_sram_tiles}_"
                    f"resident{residency_plan.resident_prefix_blocks}"
                )
            elif args.fixed_matrix_sram_policy is not None:
                matrix_sram_policy = args.fixed_matrix_sram_policy
                matrix_sram_policy_aliases = (matrix_sram_policy,)
                residency_plan = derive_matrix_sram_policy(
                    policy=matrix_sram_policy,
                    k_blocks=local_k_blocks,
                    mlen=int(params["MLEN"]),
                    projection_tiles=int(
                        sram_requirements["projection_threshold_tiles"]
                    ),
                )
                matrix_sram_config_id = (
                    f"tiles{residency_plan.matrix_sram_tiles}_"
                    f"resident{residency_plan.resident_prefix_blocks}"
                )
            elif search_encoding == "canonical-conditional-v1":
                choices = canonical_sram_choices(
                    policies=matrix_sram_policies,
                    k_blocks=local_k_blocks,
                    mlen=int(params["MLEN"]),
                    projection_tiles=int(
                        sram_requirements["projection_threshold_tiles"]
                    ),
                )
                choice_index = trial.suggest_int(
                    conditional_sram_param_name(
                        int(params["MLEN"]),
                        int(chip_count),
                        parallel_model,
                        tp_degree=(
                            tp_degree
                            if args.multi_chip_model
                            in FACTORIZED_MULTI_CHIP_MODELS
                            else None
                        ),
                        cp_degree=(
                            cp_degree
                            if args.multi_chip_model
                            in FACTORIZED_MULTI_CHIP_MODELS
                            and args.multi_chip_model
                            != TILE_AWARE_DP_MULTI_CHIP_MODEL
                            else None
                        ),
                        dp_degree=(
                            dp_degree
                            if args.multi_chip_model
                            == TILE_AWARE_DP_MULTI_CHIP_MODEL
                            else None
                        ),
                        ep_degree=(
                            ep_degree
                            if args.multi_chip_model
                            == TILE_AWARE_DP_MULTI_CHIP_MODEL
                            else None
                        ),
                    ),
                    0,
                    len(choices) - 1,
                )
                selected_choice = choices[choice_index]
                matrix_sram_policy = str(
                    selected_choice["canonical_policy"]
                )
                matrix_sram_policy_aliases = tuple(
                    selected_choice["policy_aliases"]
                )
                matrix_sram_config_id = str(selected_choice["config_id"])
                residency_plan = selected_choice["plan"]
            else:
                matrix_sram_policy = trial.suggest_categorical(
                    "MATRIX_SRAM_POLICY", list(matrix_sram_policies)
                )
                matrix_sram_policy_aliases = (matrix_sram_policy,)
                residency_plan = derive_matrix_sram_policy(
                    policy=str(matrix_sram_policy),
                    k_blocks=local_k_blocks,
                    mlen=int(params["MLEN"]),
                    projection_tiles=int(
                        sram_requirements["projection_threshold_tiles"]
                    ),
                )
                matrix_sram_config_id = (
                    f"tiles{residency_plan.matrix_sram_tiles}_"
                    f"resident{residency_plan.resident_prefix_blocks}"
                )
            matrix_sram_tiles = residency_plan.matrix_sram_tiles
            params["MATRIX_SRAM_TILES"] = matrix_sram_tiles
            hw = derived_hardware(
                model,
                params,
                dse_config,
                vector_scalar_schedule=args.vector_scalar_schedule,
                softmax_row_lanes=int(softmax_row_lanes),
            )

            duplicate_policy = False
            if (
                search_encoding == "legacy-policy-v1"
                and
                args.fixed_matrix_sram_tiles is None
                and args.fixed_matrix_sram_policy is None
            ):
                selected_key = (
                    residency_plan.matrix_sram_tiles,
                    residency_plan.resident_prefix_blocks,
                )
                canonical_policy = str(matrix_sram_policy)
                for candidate_policy in matrix_sram_policies:
                    candidate = derive_matrix_sram_policy(
                        policy=candidate_policy,
                        k_blocks=local_k_blocks,
                        mlen=int(params["MLEN"]),
                        projection_tiles=int(
                            sram_requirements["projection_threshold_tiles"]
                        ),
                    )
                    if (
                        candidate.matrix_sram_tiles,
                        candidate.resident_prefix_blocks,
                    ) == selected_key:
                        canonical_policy = candidate_policy
                        break
                duplicate_policy = matrix_sram_policy != canonical_policy
            capacity_dominated = duplicate_policy
            residency_metadata = residency_plan.metadata(
                q_blocks=local_k_blocks,
                causal=True,
            )
            record.update(
                {
                    "precision_profile": precision_name,
                    "precision_search_encoding": precision_search_encoding,
                    **matrix_datapath_signature_by_id[
                        matrix_datapath_signature_id
                    ].metadata(),
                    "accuracy_score": float(
                        precision.get("accuracy_score", 1.0)
                    ),
                    "chip_count": int(chip_count),
                    "physical_chip_count": int(chip_count),
                    "chip_count_search_value": int(
                        chip_count_search_value_by_physical[int(chip_count)]
                    ),
                    "chips_per_a100_reference": (
                        float(chip_count) / args.reference_a100_count
                    ),
                    "chip_count_scaling": args.chip_count_scaling,
                    "reference_a100_count": args.reference_a100_count,
                    "parallel_model": parallel_model,
                    "multi_chip_model": args.multi_chip_model,
                    "dp_degree": int(dp_degree),
                    "tp_degree": int(tp_degree),
                    "cp_degree": (
                        int(cp_degree)
                        if args.multi_chip_model
                        != TILE_AWARE_DP_MULTI_CHIP_MODEL
                        else None
                    ),
                    "ep_degree": int(ep_degree),
                    "dp_tp_ep_legality": (
                        "valid_natural_head_sharding"
                        if args.multi_chip_model
                        in FACTORIZED_MULTI_CHIP_MODELS
                        else "legacy_not_applicable"
                    ),
                    "nvlink_port_count": int(nvlink_port_count),
                    "nvlink_bandwidth_semantics": (
                        args.nvlink_bandwidth_semantics
                    ),
                    "matrix_sram_policy": matrix_sram_policy,
                    "matrix_sram_policy_aliases": list(
                        matrix_sram_policy_aliases
                    ),
                    "matrix_sram_config_id": matrix_sram_config_id,
                    "search_encoding": search_encoding,
                    "MATRIX_SRAM_TILES": int(matrix_sram_tiles),
                    "MATRIX_SRAM_SIZE": hw["MATRIX_SRAM_SIZE"],
                    **hw,
                    **sram_requirements,
                    **{
                        f"planned_{key}": value
                        for key, value in residency_metadata.items()
                        if key
                        not in {
                            "resident_k_addresses",
                            "resident_v_addresses",
                            "tile_elements",
                        }
                    },
                    **projection_chunk_metadata(
                        model,
                        mlen=hw["MLEN"],
                        matrix_sram_tiles=int(matrix_sram_tiles),
                    ),
                    "softmax_state_schedule": args.softmax_state_schedule,
                    "packed_qk_schedule": args.packed_qk_schedule,
                    "broadcast_timing_model": (
                        "ordinary_matrix_structural_equivalent"
                        if args.packed_qk_schedule
                        == PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1
                        else "not_applicable"
                    ),
                    "broadcast_rtl_validated": False,
                    "rtl_validation_status": (
                        "broadcast_rtl_unvalidated"
                        if args.packed_qk_schedule
                        == PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1
                        else "schedule_dependent"
                    ),
                    "large_mlen_structural_extrapolation": bool(
                        hw["MLEN"] >= 4096
                    ),
                    "very_large_mlen_structural_extrapolation": bool(
                        hw["MLEN"] >= 8192
                    ),
                    "capacity_dominated": capacity_dominated,
                    "moe_routing_mode": (
                        args.moe_routing_mode if is_moe_model else None
                    ),
                    "moe_lowering_schedule": (
                        args.moe_lowering_schedule if is_moe_model else None
                    ),
                    "moe_layer_scaling": (
                        args.moe_layer_scaling if is_moe_model else None
                    ),
                    "routing_fidelity": (
                        "fixed_balanced_histogram"
                        if is_moe_model
                        else None
                    ),
                    "full_model_routing_fidelity": (
                        "approximate_repeated_balanced_routing"
                        if is_moe_model
                        else None
                    ),
                }
            )
            if duplicate_policy:
                raise TrialPrunedError(
                    "Matrix SRAM policy maps to a duplicate physical cache "
                    f"configuration: {matrix_sram_policy}"
                )
            hard_issues, bandwidth_issues = constraint_issues(
                model,
                hw,
                precision,
                False,
                dse_config,
                min_matrix_k_splits=args.min_matrix_k_splits,
            )
            if hard_issues:
                raise TrialPrunedError("; ".join(hard_issues))
            record.update(
                {
                    "model_config": str(args.model_config),
                    "latency_model": latency_model_name,
                    "precision_profile": precision_name,
                    "warnings": bandwidth_issues,
                    "accuracy_score": float(precision.get("accuracy_score", 1.0)),
                    "chip_count": int(chip_count),
                    "physical_chip_count": int(chip_count),
                    "chip_count_search_value": int(
                        chip_count_search_value_by_physical[int(chip_count)]
                    ),
                    "chips_per_a100_reference": (
                        float(chip_count) / args.reference_a100_count
                    ),
                    "chip_count_scaling": args.chip_count_scaling,
                    "reference_a100_count": args.reference_a100_count,
                    "parallel_model": parallel_model,
                    "multi_chip_model": args.multi_chip_model,
                    "dp_degree": int(dp_degree),
                    "tp_degree": int(tp_degree),
                    "cp_degree": (
                        int(cp_degree)
                        if args.multi_chip_model
                        != TILE_AWARE_DP_MULTI_CHIP_MODEL
                        else None
                    ),
                    "ep_degree": int(ep_degree),
                    "nvlink_port_count": int(nvlink_port_count),
                    **hw,
                    "ON_CHIP_ADDR_WIDTH": hw["INT_DATA_WIDTH"],
                    "INT_SRAM_WIDTH": hw["INT_DATA_WIDTH"],
                    "weight_precision": precision_label(
                        profile_weight_spec(precision, dse_config), dse_config.mx_scale_width
                    ),
                    "packed_attention_schedule": args.packed_attention_schedule,
                    "vector_scalar_schedule": args.vector_scalar_schedule,
                    "softmax_vector_schedule": args.softmax_vector_schedule,
                    "pv_accumulation_schedule": args.pv_accumulation_schedule,
                    "softmax_row_lanes": int(softmax_row_lanes),
                    "selector_schedule": args.selector_schedule,
                    "reduction_output_mode": args.reduction_output_mode,
                    "gqa_pipeline_schedule": args.gqa_pipeline_schedule,
                    "address_generation_mode": args.address_generation_mode,
                    "ffn_address_schedule": args.ffn_address_schedule,
                    "ffn_projection_schedule": args.ffn_projection_schedule,
                }
            )

            batch_info = calculate_batch_info(model, precision, dse_config)
            record.update(batch_info)
            bandwidth_diagnostics = legacy_bandwidth_diagnostics(
                hw,
                precision,
                dse_config,
                chip_count=int(chip_count),
            )
            record.update(bandwidth_diagnostics)
            record.update(
                sequence_layout_metrics(
                    seq_len=dse_config.input_seq_len,
                    batch_size=dse_config.latency_batch_size,
                    mlen=hw["MLEN"],
                    native_layout_mode=args.native_layout_mode,
                )
            )

            # Proxy area is independent of the emitted program. Reject an
            # infeasible physical design before long-context CostEmitter and
            # power evaluation, then reuse the same metrics below.
            early_area_metrics: dict[str, Any] | None = None
            area_cache_key = canonical_json_sha256(
                {
                    "schema": "area_report_cross_study_v1",
                    "area_source_sha256": area_model_source_hash,
                    "area_mode": args.area_mode,
                    "hw": hw,
                    "precision": precision,
                    "mx_scale_width": dse_config.mx_scale_width,
                    "mx_scale_block_size": dse_config.mx_scale_block_size,
                    "fp_constant_num": dse_config.fp_constant_num,
                    "vector_scalar_schedule": args.vector_scalar_schedule,
                    "address_generation_mode": args.address_generation_mode,
                    "sram_port_model": (
                        CURRENT_DSE_PROFILE.sram_port_model
                        if args.model_profile == CURRENT_DSE_PROFILE.name
                        else RTL_VALIDATION_PROFILE.sram_port_model
                        if args.model_profile == RTL_VALIDATION_PROFILE.name
                        else "custom_from_area_model"
                    ),
                }
            )
            if args.area_mode in {"none", "proxy", "proxy-v2", "proxy-v2-mxint"}:
                heartbeat("area-prefilter")
                def compute_proxy_area() -> dict[str, Any]:
                    if args.area_mode == "none":
                        return {"area": 0.0, "area_mode": "none"}
                    if args.area_mode == "proxy":
                        return run_area_proxy(hw, precision, dse_config)
                    return run_area_proxy_v2(
                        hw,
                        precision,
                        dse_config,
                        args.vector_scalar_schedule,
                        args.address_generation_mode,
                    )

                early_area_metrics, area_cache_tier = cached_area_report(
                    area_cache_key,
                    compute_proxy_area,
                )
                record["area_cache_tier"] = area_cache_tier
                record["area_cache_scope"] = cache_scope
                early_aggregate_area = aggregate_area_from_core_metrics(
                    early_area_metrics,
                    chip_count=int(chip_count),
                    multi_chip_model=args.multi_chip_model,
                    endpoint_area_overhead_fraction=(
                        args.endpoint_area_overhead_pct / 100.0
                    ),
                    nvlink_port_count=int(nvlink_port_count),
                )
                early_area_mm2 = float(
                    early_aggregate_area["total_silicon_area_mm2"]
                )
                early_area_constraint = (
                    early_area_mm2 - args.area_budget_mm2
                )
                trial.set_user_attr(
                    "area_budget_constraint_mm2", early_area_constraint
                )
                trial.set_user_attr(
                    "a100_area_constraint_mm2", early_area_constraint
                )
                trial.set_user_attr("area_mm2", early_area_mm2)
                early_violation = area_budget_violation_reason(
                    early_area_mm2,
                    args.area_budget_mm2,
                )
                if early_violation is not None:
                    record.update(
                        {
                            "area": early_area_mm2 * 1e6,
                            "area_um2": early_area_mm2 * 1e6,
                            "area_mm2": early_area_mm2,
                            **early_aggregate_area,
                            "area_budget_constraint_mm2": (
                                early_area_constraint
                            ),
                            "a100_area_constraint_mm2": (
                                early_area_constraint
                            ),
                            "area_mode": early_area_metrics.get("area_mode"),
                            "area_model": early_area_metrics.get("area_model"),
                            "area_prefilter": "exact_proxy_before_costemitter",
                        }
                    )
                    raise TrialPrunedError(early_violation)

            legacy_would_prune = (
                bandwidth_diagnostics["required_feed_ratio"] > 1.0
            )
            if legacy_would_prune and not bandwidth_issues:
                bandwidth_issues = [
                    "legacy per-chip bandwidth expression exceeds the "
                    f"R-aware limit by "
                    f"{bandwidth_diagnostics['required_feed_ratio']:.3f}x"
                ]
            shadow = {
                "mode": args.compiler_cost_mode,
                "legacy_bandwidth_policy": args.legacy_bandwidth_policy,
                "legacy_bandwidth_prune_enabled": args.legacy_bandwidth_prune,
                "legacy_would_prune": legacy_would_prune,
                "legacy_issues": bandwidth_issues,
                "v3_would_prune": False,
                "decision_disagrees": legacy_would_prune,
                "memory_status": "disabled",
                "v3_status": "disabled",
            }
            record["legacy_bandwidth_would_prune"] = legacy_would_prune
            compiler_cost_report = None
            if args.compiler_cost_mode != "off":
                precision_issue = compiler_cost_precision_issue(precision, dse_config)
                if precision_issue:
                    shadow.update(
                        {
                            "memory_status": "incompatible",
                            "memory_error": precision_issue,
                            "v3_status": "incompatible",
                            "v3_error": precision_issue,
                        }
                    )
                    if args.compiler_cost_mode in COMPILER_COST_OBJECTIVE_MODES:
                        raise TrialPrunedError(precision_issue)
                else:
                    try:
                        compiler_cache_key = canonical_json_sha256(
                            {
                                "schema": "compiler_report_cross_study_v1",
                                "compiler_cost_source_sha256": (
                                    compiler_cost_source_hash
                                ),
                                "hardware": hw,
                                "matrix_sram_policy": matrix_sram_policy,
                                "precision": {
                                    "weight": profile_weight_spec(
                                        precision, dse_config
                                    ),
                                    "activation": precision["ACT_WIDTH"],
                                    "kv": precision["KV_WIDTH"],
                                    "internal_fp": precision["FP_SETTING"],
                                    "scale_width": dse_config.mx_scale_width,
                                    "scale_block": dse_config.mx_scale_block_size,
                                    "int_width": hw["INT_DATA_WIDTH"],
                                },
                                "seq_len": dse_config.input_seq_len,
                                "batch_size": dse_config.latency_batch_size,
                                "settings_sha256": file_sha256(
                                    args.compiler_cost_settings
                                ),
                                "calibration_sha256": file_sha256(
                                    args.compiler_cost_calibration
                                ),
                                "compute_timing": args.compiler_compute_timing,
                                "scheduled_shadow": args.compiler_scheduled_shadow,
                                "v4_memory_evaluation": (
                                    args.compiler_v4_memory_evaluation
                                ),
                                "native_layout_mode": args.native_layout_mode,
                                "packed_attention_schedule": (
                                    args.packed_attention_schedule
                                ),
                                "softmax_state_schedule": (
                                    args.softmax_state_schedule
                                ),
                                "packed_qk_schedule": args.packed_qk_schedule,
                                "vector_scalar_schedule": (
                                    args.vector_scalar_schedule
                                ),
                                "softmax_vector_schedule": (
                                    args.softmax_vector_schedule
                                ),
                                "pv_accumulation_schedule": (
                                    args.pv_accumulation_schedule
                                ),
                                "softmax_row_lanes": int(softmax_row_lanes),
                                "selector_schedule": args.selector_schedule,
                                "reduction_output_mode": (
                                    args.reduction_output_mode
                                ),
                                "gqa_pipeline_schedule": (
                                    args.gqa_pipeline_schedule
                                ),
                                "address_generation_mode": (
                                    args.address_generation_mode
                                ),
                                "ffn_address_schedule": (
                                    args.ffn_address_schedule
                                ),
                                "ffn_projection_schedule": (
                                    args.ffn_projection_schedule
                                ),
                                "cost_trace_granularity": (
                                    args.compiler_trace_granularity
                                ),
                                "model_config_sha256": file_sha256(
                                    args.model_config
                                ),
                                "moe_routing_mode": args.moe_routing_mode,
                                "moe_lowering_schedule": (
                                    args.moe_lowering_schedule
                                ),
                                "moe_layer_scaling": args.moe_layer_scaling,
                                "power_shadow": args.power_shadow,
                                "clock_gating_mode": args.clock_gating_mode,
                                "external_memory_energy_artifact_sha256": (
                                    file_sha256(
                                        args.external_memory_energy_artifact
                                    )
                                ),
                            }
                        )
                        compiler_cost_report = compiler_cost_cache.get(
                            compiler_cache_key
                        )
                        compiler_cost_cache_tier = (
                            "process-lru"
                            if compiler_cost_report is not None
                            else "cold"
                        )
                        compiler_report_metadata = (
                            compiler_report_metadata_cache.get(
                                compiler_cache_key
                            )
                        )
                        heartbeat("compiler_report_cache")
                        compiler_cost_cache_hit = compiler_cost_report is not None
                        shared_cache_dir = cache_directories.compiler_reports
                        shared_cache_dir.mkdir(parents=True, exist_ok=True)
                        shared_report_path = cache_entry_path(
                            shared_cache_dir,
                            compiler_cache_key,
                        )
                        if compiler_cost_report is None:
                            shared_lock_path = (
                                shared_cache_dir
                                / f"{compiler_cache_key}.lock"
                            )
                            with shared_lock_path.open("a+") as cache_lock:
                                fcntl.flock(cache_lock.fileno(), fcntl.LOCK_EX)
                                try:
                                    if shared_report_path.exists():
                                        compiler_cost_report = load_cached_json(
                                            shared_report_path
                                        )
                                        if compiler_cost_report is None:
                                            # A worker killed before atomic
                                            # cache publication in an older
                                            # run may leave a truncated file.
                                            # Rebuild it while holding the
                                            # per-semantic-key lock.
                                            shared_report_path.unlink(
                                                missing_ok=True
                                            )
                                            shared_report_path.with_name(
                                                f"{shared_report_path.name}.meta.json"
                                            ).unlink(missing_ok=True)
                                        else:
                                            compiler_cost_cache_hit = True
                                            compiler_cost_cache_tier = (
                                                "global-json"
                                                if cache_scope
                                                == "cross-study-global"
                                                else "run-local-json"
                                            )
                                    if compiler_cost_report is None:
                                        heartbeat(
                                            "attention_census_kernel_lowering_"
                                            "ideal_ii1_v4"
                                        )
                                        compiler_cost_report = run_compiler_cost(
                                            args.model_config,
                                            args.compiler_cost_settings,
                                            args.compiler_cost_calibration,
                                            trial_dir,
                                            hw,
                                            precision,
                                            dse_config,
                                            args.compiler_compute_timing,
                                            args.compiler_scheduled_shadow,
                                            args.compiler_v4_memory_evaluation,
                                            args.native_layout_mode,
                                            args.packed_attention_schedule,
                                            args.softmax_state_schedule,
                                            args.packed_qk_schedule,
                                            args.vector_scalar_schedule,
                                            args.softmax_vector_schedule,
                                            args.pv_accumulation_schedule,
                                            int(softmax_row_lanes),
                                            args.selector_schedule,
                                            args.reduction_output_mode,
                                            args.gqa_pipeline_schedule,
                                            args.address_generation_mode,
                                            args.ffn_address_schedule,
                                            args.ffn_projection_schedule,
                                            args.power_shadow,
                                            args.clock_gating_mode,
                                            args.external_memory_energy_artifact,
                                            str(matrix_sram_policy),
                                            args.compiler_trace_granularity,
                                            args.trial_report_materialization,
                                            cache_directories,
                                            lambda progress: heartbeat(
                                                str(
                                                    progress.get(
                                                        "phase",
                                                        "v4_aggregation",
                                                    )
                                                ),
                                                progress_done=progress.get(
                                                    "progress_done"
                                                ),
                                                progress_total=progress.get(
                                                    "progress_total"
                                                ),
                                                current_stream=progress.get(
                                                    "current_stream"
                                                ),
                                            ),
                                            moe_routing_mode=(
                                                args.moe_routing_mode
                                            ),
                                            moe_lowering_schedule=(
                                                args.moe_lowering_schedule
                                            ),
                                            moe_layer_scaling=(
                                                args.moe_layer_scaling
                                            ),
                                        )
                                        write_json(
                                            shared_report_path,
                                            compiler_cost_report,
                                        )
                                    compiler_report_metadata = (
                                        load_or_create_json_cache_metadata(
                                            shared_report_path,
                                            compiler_cost_report,
                                        )
                                    )
                                finally:
                                    fcntl.flock(
                                        cache_lock.fileno(), fcntl.LOCK_UN
                                    )
                            compiler_cost_cache[compiler_cache_key] = (
                                compiler_cost_report
                            )
                            compiler_cost_cache.move_to_end(compiler_cache_key)
                            while len(compiler_cost_cache) > 4:
                                evicted_key, _ = compiler_cost_cache.popitem(
                                    last=False
                                )
                                compiler_report_metadata_cache.pop(
                                    evicted_key, None
                                )
                            compiler_report_metadata_cache[
                                compiler_cache_key
                            ] = compiler_report_metadata
                            compiler_report_metadata_cache.move_to_end(
                                compiler_cache_key
                            )
                        if compiler_report_metadata is None:
                            raise RuntimeError(
                                "compiler report cache metadata missing for "
                                f"{compiler_cache_key}"
                            )
                        if compiler_cost_cache_hit:
                            if args.artifact_retention == "full":
                                settings_path = (
                                    trial_dir / "compiler_cost_settings.toml"
                                )
                                write_compiler_cost_toml(
                                    args.compiler_cost_settings,
                                    settings_path,
                                    hw,
                                    precision,
                                    dse_config,
                                    args.native_layout_mode,
                                )
                            compiler_cost_cache.move_to_end(compiler_cache_key)
                            if args.trial_report_materialization == "full":
                                write_json(
                                    trial_dir / "compiler_cost_report.json",
                                    compiler_cost_report,
                                )
                        if args.trial_report_materialization == "summary-reference":
                            write_json(
                                trial_dir / "compiler_cost_reference.json",
                                {
                                    "compiler_cache_key": compiler_cache_key,
                                    "compiler_cache_scope": cache_scope,
                                    "shared_report": str(shared_report_path),
                                    "shared_report_sha256": (
                                        compiler_report_metadata[
                                            "file_sha256"
                                        ]
                                    ),
                                    "canonical_payload_sha256": (
                                        compiler_report_metadata[
                                            "canonical_payload_sha256"
                                        ]
                                    ),
                                    "compute_latency_ns": compiler_cost_report.get(
                                        "compute_latency_ns"
                                    ),
                                    "memory_latency_ns": compiler_cost_report.get(
                                        "memory_latency_ns"
                                    ),
                                    "roofline_latency_ns": compiler_cost_report.get(
                                        "roofline_latency_ns"
                                    ),
                                    "cost_trace_granularity": (
                                        args.compiler_trace_granularity
                                    ),
                                    "fidelity": compiler_cost_report.get(
                                        "trace", {}
                                    ).get("compiler_metadata", {}),
                                },
                            )
                        memory_model_version = str(
                            compiler_cost_report.get("memory_model_version", "unknown")
                        )
                        heartbeat("multi_chip_latency")
                        is_v3 = memory_model_version == "global_v3"
                        compute_latency_ms = (
                            compiler_cost_report["compute_latency_ns"] / 1e6
                        )
                        memory_latency_ms = (
                            compiler_cost_report["memory_latency_ns"] / 1e6
                        )
                        serial_latency_ms = (
                            compiler_cost_report["serial_latency_ns"] / 1e6
                        )
                        roofline_latency_ms = (
                            compiler_cost_report["roofline_latency_ns"] / 1e6
                        )
                        shadow.update(
                            {
                                "memory_status": "complete",
                                "memory_model_version": memory_model_version,
                                "memory_calibration_id": compiler_cost_report.get(
                                    "calibration_id"
                                ),
                                "memory_evaluation_mode": compiler_cost_report.get(
                                    "compatibility", {}
                                ).get("memory_evaluation_mode"),
                                "compiler_compute_latency_ms": compute_latency_ms,
                                "memory_latency_ms": memory_latency_ms,
                                "serial_latency_ms": serial_latency_ms,
                                "roofline_latency_ms": roofline_latency_ms,
                                "memory_calibration_in_domain": compiler_cost_report[
                                    "calibration_in_domain"
                                ],
                                # Historical aliases remain populated only
                                # when the selected artifact really is V3.
                                "v3_status": "complete" if is_v3 else "not_applicable",
                                "v3_compute_latency_ms": (
                                    compute_latency_ms if is_v3 else None
                                ),
                                "v3_memory_latency_ms": (
                                    memory_latency_ms if is_v3 else None
                                ),
                                "v3_total_latency_ms": (
                                    serial_latency_ms if is_v3 else None
                                ),
                                "v3_calibration_in_domain": (
                                    compiler_cost_report["calibration_in_domain"]
                                    if is_v3
                                    else None
                                ),
                                "compute_timing_mode": compiler_cost_report[
                                    "compute_timing_mode"
                                ],
                                "compute_validation_status": compiler_cost_report[
                                    "compute_validation"
                                ].get("status"),
                            }
                        )
                        record.update(
                            {
                                "compiler_compute_latency_ms": compute_latency_ms,
                                "compiler_memory_latency_ms": memory_latency_ms,
                                "compiler_roofline_latency_ms": roofline_latency_ms,
                                "compiler_serial_latency_ms": serial_latency_ms,
                                "compiler_memory_model_version": memory_model_version,
                                "compiler_cost_cache_hit": compiler_cost_cache_hit,
                                "compiler_cost_cache_tier": (
                                    compiler_cost_cache_tier
                                ),
                                "compiler_cost_cache_scope": cache_scope,
                                "compiler_cost_cache_key": compiler_cache_key,
                                "compiler_memory_calibration_id": compiler_cost_report.get(
                                    "calibration_id"
                                ),
                                "compiler_memory_evaluation_mode": compiler_cost_report.get(
                                    "compatibility", {}
                                ).get("memory_evaluation_mode"),
                                "v3_memory_latency_ms": (
                                    memory_latency_ms if is_v3 else None
                                ),
                                "v3_serial_latency_ms": (
                                    serial_latency_ms if is_v3 else None
                                ),
                                "compiler_stage_compute_latency_ns": compiler_cost_report.get(
                                    "stage_compute_latency_ns", {}
                                ),
                                "compiler_stage_roofline_latency_ns": compiler_cost_report.get(
                                    "stage_roofline_latency_ns", {}
                                ),
                                "compiler_stage_bound": compiler_cost_report.get("stage_bound", {}),
                                "compiler_hbm_traffic_breakdown": compiler_cost_report.get(
                                    "hbm_traffic_breakdown", {}
                                ),
                                "compiler_calibration_in_domain": compiler_cost_report.get(
                                    "calibration_in_domain"
                                ),
                                "compiler_compute_timing_mode": compiler_cost_report.get(
                                    "compute_timing_mode"
                                ),
                                "compiler_compute_timing_semantics": compiler_cost_report.get(
                                    "compute_timing_semantics"
                                ),
                                "compiler_hazards_modeled": compiler_cost_report.get(
                                    "hazards_modeled"
                                ),
                                "compiler_rtl_cycle_validation_claim": compiler_cost_report.get(
                                    "rtl_cycle_validation_claim"
                                ),
                                "compiler_matrix_timing_artifact_hash": compiler_cost_report.get(
                                    "matrix_timing_artifact_hash"
                                ),
                                "compiler_ideal_assumed_opcode_counts": compiler_cost_report.get(
                                    "ideal_assumed_opcode_counts", {}
                                ),
                                "compiler_compute_resource_work_cycles": compiler_cost_report.get(
                                    "compute_resource_work_cycles"
                                ),
                                "compiler_compute_pipeline_makespan_cycles": compiler_cost_report.get(
                                    "compute_pipeline_makespan_cycles"
                                ),
                                "compiler_one_layer_compute_pipeline_makespan_cycles": compiler_cost_report.get(
                                    "one_layer_compute_pipeline_makespan_cycles"
                                ),
                                "compiler_compute_pipeline_fidelity": compiler_cost_report.get(
                                    "compute_pipeline_fidelity"
                                ),
                                "compiler_one_layer_compute_pipeline_fidelity": compiler_cost_report.get(
                                    "one_layer_compute_pipeline_fidelity"
                                ),
                                "compiler_compute_pipeline_cache_hit": bool(
                                    compiler_cost_cache_hit
                                    or compiler_cost_report.get(
                                        "compute_pipeline_persistent_cache_hit"
                                    )
                                ),
                                "compiler_compute_pipeline_persistent_cache_hit": compiler_cost_report.get(
                                    "compute_pipeline_persistent_cache_hit"
                                ),
                                "compiler_compute_pipeline_cache_key": compiler_cost_report.get(
                                    "compute_pipeline_persistent_cache_key"
                                ),
                                "compiler_scalar_pipeline_busy_cycles": compiler_cost_report.get(
                                    "scalar_pipeline_busy_cycles"
                                ),
                                "compiler_scalar_rob_stall_cycles": compiler_cost_report.get(
                                    "scalar_rob_stall_cycles"
                                ),
                                "compiler_segment_parallel_reduction_cycles": compiler_cost_report.get(
                                    "segment_parallel_reduction_cycles"
                                ),
                                "compiler_gqa_stall_cycles_by_reason": compiler_cost_report.get(
                                    "gqa_stall_cycles_by_reason", {}
                                ),
                                "compiler_compute_pipeline": compiler_cost_report.get(
                                    "compute_pipeline", {}
                                ),
                                "compiler_compute_validation": compiler_cost_report.get(
                                    "compute_validation", {}
                                ),
                                "compiler_compute_calibration_in_domain": compiler_cost_report.get(
                                    "compute_calibration_in_domain"
                                ),
                                "legacy_compiler_compute_latency_ms": compiler_cost_report.get(
                                    "legacy_compute_latency_ns", 0.0
                                )
                                / 1e6,
                                "scheduled_shadow_status": compiler_cost_report.get(
                                    "scheduled_shadow", {}
                                ).get("status"),
                                "scheduled_shadow_fidelity": compiler_cost_report.get(
                                    "scheduled_shadow", {}
                                ).get("fidelity"),
                                "scheduled_shadow_reason": compiler_cost_report.get(
                                    "scheduled_shadow", {}
                                ).get("reason"),
                                "scheduled_shadow_validation": compiler_cost_report.get(
                                    "scheduled_shadow", {}
                                ).get("validation", {}),
                                "scheduled_shadow_stall_cycles_by_reason": compiler_cost_report.get(
                                    "scheduled_shadow", {}
                                ).get("stall_cycles_by_reason", {}),
                                "scheduled_shadow_resource_work_cycles": compiler_cost_report.get(
                                    "scheduled_shadow", {}
                                ).get("resource_work_cycles", {}),
                                "scheduled_shadow_makespan_cycles": compiler_cost_report.get(
                                    "scheduled_shadow_makespan_cycles"
                                ),
                                "scheduled_shadow_latency_ms": (
                                    None
                                    if compiler_cost_report.get(
                                        "scheduled_shadow_latency_ns"
                                    )
                                    is None
                                    else compiler_cost_report[
                                        "scheduled_shadow_latency_ns"
                                    ]
                                    / 1e6
                                ),
                            }
                        )
                        record.update(compute_fidelity_metrics(compiler_cost_report))
                        record.update(
                            compiler_layout_record_fields(compiler_cost_report)
                        )
                        power_shadow = dict(
                            compiler_cost_report.get("power_shadow") or {}
                        )
                        record.update(
                            {
                                "power_status": power_shadow.get(
                                    "status", "missing"
                                ),
                                "power_model": power_shadow.get("power_model"),
                                "power_scope": power_shadow.get("power_scope"),
                                "onchip_energy_mj": power_shadow.get(
                                    "onchip_energy_mj"
                                ),
                                "onchip_average_power_w": power_shadow.get(
                                    "onchip_average_power_w"
                                ),
                                "clock_gating_mode": power_shadow.get(
                                    "clock_gating_mode"
                                ),
                                "clock_gating_status": power_shadow.get(
                                    "clock_gating_status"
                                ),
                                "clock_energy_mj": power_shadow.get(
                                    "clock_energy_mj"
                                ),
                                "ungated_clock_energy_mj": power_shadow.get(
                                    "ungated_clock_energy_mj"
                                ),
                                "ungated_onchip_energy_mj": power_shadow.get(
                                    "ungated_onchip_energy_mj"
                                ),
                                "ungated_onchip_average_power_w": power_shadow.get(
                                    "ungated_onchip_average_power_w"
                                ),
                                "clock_energy_savings_pct": power_shadow.get(
                                    "clock_energy_savings_pct"
                                ),
                                "clock_active_fraction_by_component": power_shadow.get(
                                    "clock_active_fraction_by_component"
                                ),
                                "unmodeled_clock_residual_area_um2": power_shadow.get(
                                    "unmodeled_clock_residual_area_um2"
                                ),
                                "energy_per_input_token_mj": power_shadow.get(
                                    "energy_per_input_token_mj"
                                ),
                                "power_calibration_status": power_shadow.get(
                                    "calibration_status"
                                ),
                                "power_uncertainty": power_shadow.get(
                                    "system_uncertainty_power_w",
                                    power_shadow.get("uncertainty_power_w"),
                                ),
                                "external_memory_model": power_shadow.get(
                                    "external_memory_model"
                                ),
                                "external_memory_calibration_status": power_shadow.get(
                                    "external_memory_calibration_status"
                                ),
                                "external_memory_configuration_semantics": power_shadow.get(
                                    "external_memory_configuration_semantics"
                                ),
                                "external_hbm_capacity_bytes": power_shadow.get(
                                    "external_hbm_capacity_bytes"
                                ),
                                "external_hbm_configured_bandwidth_gbps": power_shadow.get(
                                    "external_hbm_configured_bandwidth_gbps"
                                ),
                                "hbm_background_energy_mj": power_shadow.get(
                                    "hbm_background_energy_mj"
                                ),
                                "hbm_read_energy_mj": power_shadow.get(
                                    "hbm_read_energy_mj"
                                ),
                                "hbm_write_energy_mj": power_shadow.get(
                                    "hbm_write_energy_mj"
                                ),
                                "external_hbm_energy_mj": power_shadow.get(
                                    "external_hbm_energy_mj"
                                ),
                                "external_hbm_energy_p10_mj": power_shadow.get(
                                    "external_hbm_energy_p10_mj"
                                ),
                                "external_hbm_energy_p50_mj": power_shadow.get(
                                    "external_hbm_energy_p50_mj"
                                ),
                                "external_hbm_energy_p90_mj": power_shadow.get(
                                    "external_hbm_energy_p90_mj"
                                ),
                                "external_hbm_average_power_w": power_shadow.get(
                                    "external_hbm_average_power_w"
                                ),
                                "external_hbm_average_power_p10_w": power_shadow.get(
                                    "external_hbm_average_power_p10_w"
                                ),
                                "external_hbm_average_power_p50_w": power_shadow.get(
                                    "external_hbm_average_power_p50_w"
                                ),
                                "external_hbm_average_power_p90_w": power_shadow.get(
                                    "external_hbm_average_power_p90_w"
                                ),
                                "hbm_physical_read_bytes": power_shadow.get(
                                    "hbm_physical_read_bytes"
                                ),
                                "hbm_physical_write_bytes": power_shadow.get(
                                    "hbm_physical_write_bytes"
                                ),
                                "hbm_payload_read_bytes": power_shadow.get(
                                    "hbm_payload_read_bytes"
                                ),
                                "hbm_payload_write_bytes": power_shadow.get(
                                    "hbm_payload_write_bytes"
                                ),
                                "physical_to_payload_traffic_ratio": power_shadow.get(
                                    "physical_to_payload_traffic_ratio"
                                ),
                                "achieved_average_bandwidth_gbps": power_shadow.get(
                                    "achieved_average_bandwidth_gbps"
                                ),
                                "bandwidth_utilization": power_shadow.get(
                                    "bandwidth_utilization"
                                ),
                                "external_hbm_energy_by_role": power_shadow.get(
                                    "external_hbm_energy_by_role"
                                ),
                                "external_hbm_energy_by_stage": power_shadow.get(
                                    "external_hbm_energy_by_stage"
                                ),
                                "external_hbm_energy_by_opcode": power_shadow.get(
                                    "external_hbm_energy_by_opcode"
                                ),
                                "system_energy_mj": power_shadow.get(
                                    "system_energy_mj"
                                ),
                                "system_energy_p10_mj": power_shadow.get(
                                    "system_energy_p10_mj"
                                ),
                                "system_energy_p50_mj": power_shadow.get(
                                    "system_energy_p50_mj"
                                ),
                                "system_energy_p90_mj": power_shadow.get(
                                    "system_energy_p90_mj"
                                ),
                                "system_average_power_w": power_shadow.get(
                                    "system_average_power_w"
                                ),
                                "ungated_system_energy_mj": power_shadow.get(
                                    "ungated_system_energy_mj"
                                ),
                                "ungated_system_average_power_w": power_shadow.get(
                                    "ungated_system_average_power_w"
                                ),
                                "system_average_power_p10_w": power_shadow.get(
                                    "system_average_power_p10_w"
                                ),
                                "system_average_power_p50_w": power_shadow.get(
                                    "system_average_power_p50_w"
                                ),
                                "system_average_power_p90_w": power_shadow.get(
                                    "system_average_power_p90_w"
                                ),
                                "system_energy_per_input_token_mj": power_shadow.get(
                                    "system_energy_per_input_token_mj"
                                ),
                                "power_warnings": power_shadow.get(
                                    "warnings", []
                                ),
                                "power_excludes": power_shadow.get(
                                    "excludes", []
                                ),
                                "power_shadow": power_shadow,
                            }
                        )
                        if not compiler_cost_report["calibration_in_domain"]:
                            shadow["memory_status"] = "out_of_domain"
                            if is_v3:
                                shadow["v3_status"] = "out_of_domain"
                            if args.compiler_cost_mode == "objective":
                                raise TrialPrunedError(
                                    "Compiler Cost memory calibration is out of domain: "
                                    + "; ".join(
                                        compiler_cost_report["compatibility"].get(
                                            "domain_issues", []
                                        )
                                    )
                                )
                    except Exception as exc:
                        shadow.update(
                            {
                                "memory_status": "failed",
                                "memory_error": f"{type(exc).__name__}: {exc}",
                                "v3_status": "failed",
                                "v3_error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if args.compiler_cost_mode in COMPILER_COST_OBJECTIVE_MODES:
                            raise
            record["bandwidth_shadow"] = shadow
            if args.legacy_bandwidth_prune and legacy_would_prune:
                raise TrialPrunedError("; ".join(bandwidth_issues))

            multi_chip_report = None
            if compiler_cost_report is not None:
                if args.multi_chip_model in {
                    "tile-aware-tp-cp-ep-v3",
                    TILE_AWARE_DP_MULTI_CHIP_MODEL,
                }:
                    lineage = dict(
                        (
                            compiler_cost_report.get("power_inputs") or {}
                        ).get("metadata", {}).get(
                            "energy_action_lineage"
                        )
                        or {}
                    )
                    if (
                        lineage.get("schema")
                        != "energy_action_kernel_lineage_v3_structural_families"
                        or float(lineage.get("coverage", 0.0)) != 1.0
                    ):
                        raise ValueError(
                            "tile-aware energy objective requires exact "
                            "structural EnergyAction lineage coverage"
                        )
                fp_setting = precision["FP_SETTING"]
                fp_width_bits = (
                    1
                    + int(fp_setting["exp"])
                    + int(fp_setting["mant"])
                )
                multi_chip_report = estimate_multi_chip_latency(
                    compiler_cost_report,
                    model,
                    chip_count=int(chip_count),
                    reference_a100_count=args.reference_a100_count,
                    parallel_model=parallel_model,
                    aggregate_hbm_bandwidth_gbps=(
                        dse_config.hbm_bandwidth_gbps
                    ),
                    aggregate_hbm_capacity_bytes=(
                        dse_config.hbm_capacity_bytes
                    ),
                    seq_len=dse_config.input_seq_len,
                    batch_size=dse_config.latency_batch_size,
                    fp_width_bits=fp_width_bits,
                    multi_chip_model=args.multi_chip_model,
                    dp_degree=int(dp_degree),
                    tp_degree=int(tp_degree),
                    ep_degree=int(ep_degree),
                    kv_width_bits=effective_mx_bits(
                        precision["KV_WIDTH"], dse_config
                    ),
                    nvlink_port_count=int(nvlink_port_count),
                    nvlink_port_bidirectional_gbps=(
                        DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS
                    ),
                    interconnect_startup_ns=(
                        args.nvlink_startup_us * 1_000.0
                    ),
                    one_way_link_bandwidth_gbps=(
                        args.nvlink_bandwidth_gbps / 2.0
                    ),
                    kv_cache_overlay={
                        "global_k_blocks": math.ceil(
                            dse_config.input_seq_len / hw["MLEN"]
                        ),
                        "local_k_blocks": local_k_blocks,
                        "global_tile_loads": plan_kv_residency(
                            k_blocks=math.ceil(
                                dse_config.input_seq_len / hw["MLEN"]
                            ),
                            mlen=hw["MLEN"],
                            matrix_sram_tiles=int(matrix_sram_tiles),
                            policy=str(matrix_sram_policy),
                        ).expected_tile_loads(
                            q_blocks=math.ceil(
                                dse_config.input_seq_len / hw["MLEN"]
                            ),
                            causal=True,
                        ),
                        "local_tile_loads": residency_plan.expected_tile_loads(
                            q_blocks=local_k_blocks,
                            causal=True,
                        ),
                        "matrix_sram_policy": matrix_sram_policy,
                        "resident_prefix_blocks": (
                            residency_plan.resident_prefix_blocks
                        ),
                    },
                )
                weight_partition = estimate_model_weight_partition(
                    model,
                    float(batch_info["model_weight_bytes"]),
                    total_parameter_count=dse_config.weight_param_count,
                )
                max_request_fraction = (
                    float(multi_chip_report["max_local_batch_size"])
                    / dse_config.latency_batch_size
                    if args.multi_chip_model
                    == TILE_AWARE_DP_MULTI_CHIP_MODEL
                    else float(multi_chip_report["max_token_fraction"])
                )
                capacity_report = calculate_multichip_hbm_capacity(
                    batch_info,
                    batch_size=dse_config.latency_batch_size,
                    chip_count=int(chip_count),
                    tp_degree=int(tp_degree),
                    cp_degree=int(cp_degree),
                    max_token_fraction=max_request_fraction,
                    per_chip_hbm_capacity_bytes=float(
                        multi_chip_report["per_chip_hbm_capacity_bytes"]
                    ),
                    factorized_parallel=(
                        args.multi_chip_model
                        in FACTORIZED_MULTI_CHIP_MODELS
                    ),
                    parallel_model=parallel_model,
                    dp_degree=int(dp_degree),
                    ep_degree=int(ep_degree),
                    shared_weight_bytes=weight_partition[
                        "shared_weight_bytes"
                    ],
                    expert_weight_bytes=weight_partition[
                        "expert_weight_bytes"
                    ],
                )
                per_chip_required_bytes = capacity_report[
                    "per_chip_hbm_required_bytes"
                ]
                aggregate_required_bytes = capacity_report[
                    "aggregate_hbm_required_bytes"
                ]
                aggregate_kv_bytes = capacity_report[
                    "aggregate_prefill_kv_capacity_bytes"
                ]
                full_decoder_kv_cache_bytes_shadow = capacity_report[
                    "full_decoder_kv_cache_bytes_shadow"
                ]
                hbm_capacity_feasible = capacity_report[
                    "per_chip_hbm_capacity_feasible"
                ]
                record.update(
                    {
                        "multi_chip": multi_chip_report,
                        "multi_chip_fidelity": multi_chip_report[
                            "multi_chip_fidelity"
                        ],
                        "aggregate_hbm_capacity_bytes": multi_chip_report[
                            "aggregate_hbm_capacity_bytes"
                        ],
                        "aggregate_hbm_bandwidth_gbps": multi_chip_report[
                            "aggregate_hbm_bandwidth_gbps"
                        ],
                        "per_chip_hbm_capacity_bytes": multi_chip_report[
                            "per_chip_hbm_capacity_bytes"
                        ],
                        "per_chip_hbm_bandwidth_gbps": multi_chip_report[
                            "per_chip_hbm_bandwidth_gbps"
                        ],
                        "per_chip_equivalent_hbm_channels": multi_chip_report[
                            "per_chip_equivalent_hbm_channels"
                        ],
                        "hbm_channel_calibration_status": multi_chip_report[
                            "hbm_channel_calibration_status"
                        ],
                        "hbm_channel_extrapolation_ratio": multi_chip_report[
                            "hbm_channel_extrapolation_ratio"
                        ],
                        "per_chip_hbm_required_bytes": per_chip_required_bytes,
                        "aggregate_hbm_required_bytes": (
                            aggregate_required_bytes
                        ),
                        "aggregate_prefill_kv_capacity_bytes": (
                            aggregate_kv_bytes
                        ),
                        "full_decoder_kv_cache_bytes_shadow": (
                            full_decoder_kv_cache_bytes_shadow
                        ),
                        "per_chip_hbm_capacity_feasible": hbm_capacity_feasible,
                        "shared_weight_replication": capacity_report.get(
                            "shared_weight_replication"
                        ),
                        "expert_weight_replication": capacity_report.get(
                            "expert_weight_replication"
                        ),
                        "weight_replication_factor": capacity_report.get(
                            "weight_replication_factor"
                        ),
                        "per_chip_compute_scale": multi_chip_report[
                            "per_chip_compute_scale"
                        ],
                        "dp_degree": int(dp_degree),
                        "request_origin_count": multi_chip_report.get(
                            "request_origin_count"
                        ),
                        "local_batch_by_origin": multi_chip_report.get(
                            "local_batch_by_origin"
                        ),
                        "active_request_origin_count": (
                            multi_chip_report.get(
                                "active_request_origin_count"
                            )
                        ),
                        "idle_request_origin_count": multi_chip_report.get(
                            "idle_request_origin_count"
                        ),
                        "batch_packing_utilization": (
                            multi_chip_report.get(
                                "batch_packing_utilization"
                            )
                        ),
                        "fixed_batch_requests_per_second": (
                            multi_chip_report.get(
                                "fixed_batch_requests_per_second"
                            )
                        ),
                        "fixed_batch_tokens_per_second": (
                            multi_chip_report.get(
                                "fixed_batch_tokens_per_second"
                            )
                        ),
                        "max_token_fraction": multi_chip_report.get(
                            "max_token_fraction"
                        ),
                        "max_causal_pair_fraction": multi_chip_report.get(
                            "max_causal_pair_fraction"
                        ),
                        "parallel_work_census_coverage": multi_chip_report.get(
                            "parallel_work_census_coverage"
                        ),
                        "parallel_kernel_census_coverage": (
                            multi_chip_report.get(
                                "parallel_kernel_census_coverage"
                            )
                        ),
                        "local_tile_counts_by_rank": multi_chip_report.get(
                            "local_tile_counts_by_rank"
                        ),
                        "slowest_rank": multi_chip_report.get("slowest_rank"),
                        "matrix_utilization_by_stage": (
                            multi_chip_report.get(
                                "matrix_utilization_by_stage"
                            )
                        ),
                        "vector_utilization_by_stage": (
                            multi_chip_report.get(
                                "vector_utilization_by_stage"
                            )
                        ),
                        "padding_cycles": multi_chip_report.get(
                            "padding_cycles"
                        ),
                        "replicated_compute_cycles": multi_chip_report.get(
                            "replicated_compute_cycles"
                        ),
                        "tp_rounding_overhead": multi_chip_report.get(
                            "tp_rounding_overhead"
                        ),
                        "cp_tail_overhead": multi_chip_report.get(
                            "cp_tail_overhead"
                        ),
                        "dp_batch_imbalance": multi_chip_report.get(
                            "dp_batch_imbalance"
                        ),
                        "experts_per_rank": multi_chip_report.get(
                            "experts_per_rank"
                        ),
                        "expert_bucket_utilization": multi_chip_report.get(
                            "expert_bucket_utilization"
                        ),
                        "ep_dispatch_bytes": multi_chip_report.get(
                            "ep_dispatch_bytes", 0.0
                        ),
                        "ep_return_bytes": multi_chip_report.get(
                            "ep_return_bytes", 0.0
                        ),
                        "fractional_v2_latency": multi_chip_report.get(
                            "fractional_v2_latency"
                        ),
                        "tile_aware_v3_latency": multi_chip_report.get(
                            "tile_aware_v3_latency"
                        ),
                        "per_chip_stage_compute_latency_ns": multi_chip_report[
                            "per_chip_stage_compute_latency_ns"
                        ],
                        "per_chip_stage_memory_latency_ns": multi_chip_report[
                            "per_chip_stage_memory_latency_ns"
                        ],
                        "r_aware_v4_floor_ns": sum(
                            multi_chip_report[
                                "per_chip_stage_v4_floor_ns"
                            ].values()
                        ),
                        "r_aware_v4_residual_ns": sum(
                            multi_chip_report[
                                "per_chip_stage_memory_latency_ns"
                            ].values()
                        )
                        - sum(
                            multi_chip_report[
                                "per_chip_stage_v4_floor_ns"
                            ].values()
                        ),
                        "per_chip_hbm_physical_bytes": multi_chip_report[
                            "per_chip_hbm_physical_bytes"
                        ],
                        "aggregate_hbm_physical_bytes": multi_chip_report[
                            "aggregate_hbm_physical_bytes"
                        ],
                        "per_chip_achieved_bandwidth_gbps": multi_chip_report[
                            "per_chip_achieved_bandwidth_gbps"
                        ],
                        "per_chip_bandwidth_utilization": multi_chip_report[
                            "per_chip_bandwidth_utilization"
                        ],
                        "interconnect_bytes": multi_chip_report[
                            "interconnect_bytes"
                        ],
                        "interconnect_latency_ns": multi_chip_report[
                            "interconnect_latency_ns"
                        ],
                        "interconnect_latency_ms": multi_chip_report[
                            "interconnect_latency_ns"
                        ]
                        / 1e6,
                        "tp_collective_bytes": multi_chip_report.get(
                            "tp_collective_bytes", 0.0
                        ),
                        "tp_collective_latency_ns": multi_chip_report.get(
                            "tp_collective_latency_ns", 0.0
                        ),
                        "cp_kv_ring_bytes": multi_chip_report.get(
                            "cp_kv_ring_bytes", 0.0
                        ),
                        "cp_kv_ring_latency_ns": multi_chip_report.get(
                            "cp_kv_ring_latency_ns", 0.0
                        ),
                        "ep_dispatch_latency_ns": multi_chip_report.get(
                            "ep_dispatch_latency_ns", 0.0
                        ),
                        "ep_return_latency_ns": multi_chip_report.get(
                            "ep_return_latency_ns", 0.0
                        ),
                        "dependency_serial_nominal_ns": (
                            multi_chip_report.get(
                                "dependency_serial_nominal_ns"
                            )
                        ),
                        "nvlink_peak_oneway_bandwidth_gbps": (
                            multi_chip_report.get(
                                "nvlink_peak_oneway_bandwidth_gbps"
                            )
                        ),
                        "weight_replication_factor": capacity_report.get(
                            "weight_replication_factor", 1
                        ),
                        "communication_overlap_bound": multi_chip_report.get(
                            "communication_overlap_bound"
                        ),
                        "full_overlap_lower_bound_ns": multi_chip_report.get(
                            "full_overlap_lower_bound_ns"
                        ),
                        "nominal_stage_model_ns": multi_chip_report.get(
                            "nominal_stage_model_ns"
                        ),
                        "no_overlap_upper_bound_ns": multi_chip_report.get(
                            "no_overlap_upper_bound_ns"
                        ),
                        **estimate_decode_kv_handoff(
                            model,
                            seq_len=dse_config.input_seq_len,
                            batch_size=dse_config.latency_batch_size,
                            source_chip_count=int(chip_count),
                            decode_chip_count=decode_chip_count,
                            source_port_count=int(nvlink_port_count),
                            decode_port_count=(
                                args.decode_nvlink_port_count
                                or int(nvlink_port_count)
                            ),
                            per_port_one_way_bandwidth_gbps=(
                                DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS / 2.0
                            ),
                            startup_ns=args.nvlink_startup_us * 1e3,
                        ),
                    }
                )
                power_config = build_area_proxy_inputs(
                    hw, precision, dse_config
                )
                power_config.update(
                    {
                        "CLOCK_PERIOD_PS": round(
                            1000.0 / dse_config.frequency_ghz
                        ),
                        "SEQ_LEN": dse_config.input_seq_len,
                        "BATCH_SIZE": dse_config.latency_batch_size,
                        "INPUT_TOKENS": (
                            dse_config.input_seq_len
                            * dse_config.latency_batch_size
                        ),
                        "HBM_CAPACITY_BYTES": dse_config.hbm_capacity_bytes,
                        "HBM_BANDWIDTH_GBPS": dse_config.hbm_bandwidth_gbps,
                        "HBM_CONFIGURATION_SEMANTICS": (
                            "abstract_80gb_a100_aligned"
                        ),
                    }
                )
                heartbeat("power")
                power_report = estimate_multi_chip_system_power(
                    power_config,
                    compiler_cost_report["power_inputs"],
                    compiler_cost_report,
                    multi_chip_report,
                    chip_count=int(chip_count),
                    parallel_model=parallel_model,
                    external_memory_config={
                        "HBM_CAPACITY_BYTES": dse_config.hbm_capacity_bytes,
                        "HBM_BANDWIDTH_GBPS": dse_config.hbm_bandwidth_gbps,
                        "HBM_CONFIGURATION_SEMANTICS": (
                            "abstract_80gb_a100_aligned"
                        ),
                    },
                    external_memory_artifact_path=(
                        args.external_memory_energy_artifact
                    ),
                    sram_background_path=(
                        args.sram_background_energy_artifact
                    ),
                    interconnect_energy_artifact_path=(
                        args.interconnect_energy_artifact
                    ),
                    clock_gating_mode=args.clock_gating_mode,
                )
                power_report = {"status": "complete", **power_report}
                record.update(power_record_fields(power_report))
                if not hbm_capacity_feasible:
                    raise TrialPrunedError(
                        "per-chip weight+prefill KV working-set capacity "
                        "exceeds the fixed aggregate HBM allocation"
                    )

            if args.compiler_cost_mode in COMPILER_COST_OBJECTIVE_MODES:
                if compiler_cost_report is None:
                    raise RuntimeError("compiler cost objective completed without a cost report")
                compute_latency_ms = record["compiler_compute_latency_ms"]
                if args.compiler_cost_mode == "compute-objective":
                    if multi_chip_report is None:
                        raise RuntimeError("missing multi-chip compute report")
                    latency_ms = (
                        sum(
                            multi_chip_report[
                                "per_chip_stage_compute_latency_ns"
                            ].values()
                        )
                        + multi_chip_report["interconnect_latency_ns"]
                    ) / 1e6
                    latency_source = (
                        "compiler_cost_ideal_ii1_multichip_compute_plus_nvlink"
                        if args.compiler_compute_timing == "ideal-ii1"
                        else "compiler_cost_rtl_v1_multichip_compute_plus_nvlink"
                        if args.compiler_compute_timing == "rtl-v1"
                        else "compiler_cost_legacy_compute"
                    )
                    objective_combination = (
                        "per_chip_compute_plus_nvlink_no_memory"
                    )
                    record["latency_model"] = latency_model_name
                elif args.compiler_cost_mode == "roofline-objective":
                    if multi_chip_report is None:
                        raise RuntimeError("missing multi-chip roofline report")
                    latency_ms = multi_chip_report["latency_ms"]
                    latency_source = (
                        "compiler_cost_stage_roofline_ideal_ii1_r_aware_v4_nvlink"
                        if args.compiler_compute_timing == "ideal-ii1"
                        else "compiler_cost_stage_roofline_rtl_v1_r_aware_v4_nvlink"
                        if args.compiler_compute_timing == "rtl-v1"
                        else "compiler_cost_stage_roofline_legacy_r_aware_v4_nvlink"
                    )
                    objective_combination = (
                        "sum_stage_max_per_chip_compute_r_aware_v4_plus_nvlink"
                    )
                    record["latency_model"] = (
                        latency_model_name
                        if args.compiler_compute_timing == "ideal-ii1"
                        else "compiler_stage_roofline_rtl_v1_v4_multichip_v1"
                        if args.compiler_compute_timing == "rtl-v1"
                        else "compiler_stage_roofline_legacy_v4_multichip_v1"
                    )
                else:
                    latency_ms = compiler_cost_report["true_full_model_latency_ns"] / 1e6
                    latency_source = (
                        "compiler_cost_"
                        + str(compiler_cost_report.get("memory_model_version", "memory"))
                    )
                    objective_combination = "transactional_serial"
                    record["latency_model"] = "compiler_integrated_serial_cost"
                record["latency_source"] = latency_source
                latency_report = {
                    "latency_source": latency_source,
                    "objective_combination": objective_combination,
                    "legacy_bandwidth_guard": args.legacy_bandwidth_prune,
                    "compiler_cost": (
                        compiler_cost_report
                        if args.trial_report_materialization == "full"
                        else {
                            "compiler_cache_key": compiler_cache_key,
                            "shared_report": str(shared_report_path),
                            "compute_latency_ns": compiler_cost_report.get(
                                "compute_latency_ns"
                            ),
                            "memory_latency_ns": compiler_cost_report.get(
                                "memory_latency_ns"
                            ),
                            "roofline_latency_ns": compiler_cost_report.get(
                                "roofline_latency_ns"
                            ),
                            "stage_compute_latency_ns": compiler_cost_report.get(
                                "stage_compute_latency_ns", {}
                            ),
                            "stage_roofline_latency_ns": compiler_cost_report.get(
                                "stage_roofline_latency_ns", {}
                            ),
                        }
                    ),
                }
            else:
                analytic_toml = trial_dir / "analytic_hardware.toml"
                write_analytic_toml(analytic_toml, hw, dse_config)
                latency_ms, latency_report = run_latency(
                    args.model_config,
                    analytic_toml,
                    trial_dir,
                    batch_info,
                    dse_config,
                )
            if "fp16_kv_handoff_latency_ms" in record:
                handoff_latency_ms = float(
                    record["fp16_kv_handoff_latency_ms"]
                )
                record.update(
                    {
                        "prefill_latency_excluding_kv_handoff_ms": latency_ms,
                        "kv_handoff_included_in_dse_latency": False,
                        "kv_handoff_included_in_energy_objective": False,
                        "prefill_kv_handoff_full_overlap_lower_bound_ms": max(
                            latency_ms, handoff_latency_ms
                        ),
                        "prefill_plus_kv_handoff_serial_shadow_ms": (
                            latency_ms + handoff_latency_ms
                        ),
                        "kv_handoff_overlap_semantics": (
                            "report_bounds_only_no_release_time_schedule"
                        ),
                    }
                )
            record["latency_ms"] = latency_ms
            persist_trial_record(
                trial_dir,
                record,
                artifact_retention=args.artifact_retention,
            )

            key = area_cache_key
            heartbeat("area")
            if early_area_metrics is not None:
                area_metrics = early_area_metrics
            elif key in cache:
                area_metrics = cache[key]
            elif args.area_mode == "none":
                area_metrics = {"area": 0.0, "area_mode": "none"}
            elif args.area_mode == "proxy":
                area_metrics = run_area_proxy(hw, precision, dse_config)
            elif args.area_mode in {"proxy-v2", "proxy-v2-mxint"}:
                area_metrics = run_area_proxy_v2(
                    hw,
                    precision,
                    dse_config,
                    args.vector_scalar_schedule,
                    args.address_generation_mode,
                )
            elif args.area_mode == "parse-existing":
                area_metrics = parse_area_power()
                area_metrics.setdefault("area", 0.0)
                area_metrics["area_mode"] = "parse-existing"
            elif args.area_mode == "elaborate":
                rtl_params = run_rtl_config(hw, precision, dse_config)
                record.update(rtl_params)
                run_rtl_elaborate()
                area_metrics = parse_elaborate_metrics()
                copy_elaborate_reports(trial_dir)
            else:
                rtl_params = run_rtl_config(hw, precision, dse_config)
                record.update(rtl_params)
                run_area_synth()
                area_metrics = parse_area_power()
                if "area" not in area_metrics:
                    raise RuntimeError("area synth completed but Total cell area was not parsed")
                copy_rtl_reports(trial_dir)
                area_metrics["area_mode"] = "synth"
            cache[key] = area_metrics
            if args.area_mode not in {
                "none",
                "proxy",
                "proxy-v2",
                "proxy-v2-mxint",
            }:
                write_json(cache_path, cache)

            aggregate_area_metrics = aggregate_area_from_core_metrics(
                area_metrics,
                chip_count=int(chip_count),
                multi_chip_model=args.multi_chip_model,
                endpoint_area_overhead_fraction=(
                    args.endpoint_area_overhead_pct / 100.0
                ),
                nvlink_port_count=int(nvlink_port_count),
            )
            area_mm2 = aggregate_area_metrics["total_silicon_area_mm2"]
            area_um2 = area_mm2 * 1e6
            area_p10_mm2 = aggregate_area_metrics[
                "total_silicon_area_p10_mm2"
            ]
            area_p50_mm2 = aggregate_area_metrics[
                "total_silicon_area_p50_mm2"
            ]
            area_p90_mm2 = aggregate_area_metrics[
                "total_silicon_area_p90_mm2"
            ]
            area_constraint = area_mm2 - args.area_budget_mm2
            tolerance = args.target_area_mm2 * args.target_area_tolerance_pct / 100.0
            trial.set_user_attr("area_budget_constraint_mm2", area_constraint)
            trial.set_user_attr("a100_area_constraint_mm2", area_constraint)
            trial.set_user_attr("area_mm2", area_mm2)
            trial.set_user_attr(
                "system_energy_nominal_mj",
                float(record["system_energy_nominal_mj"]),
            )

            area_warnings = list(
                area_metrics.get("area_extrapolation_warnings", [])
            )
            vector_scalar_area_status = "calibrated_existing_rtl"
            if args.vector_scalar_schedule in {"rtl-v3", "rtl-v4", "rtl-v5", "rtl-v6"}:
                vector_status = (area_metrics.get("vector_machine") or {}).get(
                    "vector_scalar_area_calibration_status"
                )
                scalar_status = (area_metrics.get("scalar_machine") or {}).get(
                    "vector_scalar_area_calibration_status"
                )
                if (
                    args.vector_scalar_schedule == "rtl-v3"
                    and vector_status
                    == scalar_status
                    == "calibrated_rtl_v3_delta_overlay"
                ):
                    vector_scalar_area_status = "calibrated_rtl_v3_delta_overlay"
                elif (
                    args.vector_scalar_schedule == "rtl-v6"
                    and vector_status == "structural_proxy_pending_paired_dc"
                ):
                    vector_scalar_area_status = str(vector_status)
                elif (
                    args.vector_scalar_schedule == "rtl-v5"
                    and vector_status
                    in {
                        "structural_extrapolation_from_compact_leaf_dc",
                        "fitted_from_paired_rtl_v5_dc",
                    }
                ):
                    vector_scalar_area_status = str(vector_status)
                elif (
                    args.vector_scalar_schedule == "rtl-v4"
                    and vector_status
                    == "fitted_from_paired_rtl_v4_dc"
                ):
                    vector_scalar_area_status = (
                        "fitted_from_paired_rtl_v4_dc"
                    )
                elif args.vector_scalar_schedule == "rtl-v4":
                    vector_scalar_area_status = "recalibration_pending_rtl_v4"
                elif args.vector_scalar_schedule == "rtl-v5":
                    vector_scalar_area_status = "recalibration_pending_rtl_v5"
                elif args.vector_scalar_schedule == "rtl-v6":
                    vector_scalar_area_status = "recalibration_pending_rtl_v6"
                else:
                    vector_scalar_area_status = "recalibration_pending_rtl_v3"
            elif args.vector_scalar_schedule == "rtl-v2":
                vector_scalar_area_status = "calibrated_pre_rtl_v3"
            if vector_scalar_area_status.startswith("recalibration_pending_"):
                area_warnings.append(
                    f"Vector/Scalar {args.vector_scalar_schedule} logic is not included "
                    "in the current area calibration; reported area uses the latest "
                    "available calibrated overlay"
                )
            area_warnings = list(dict.fromkeys(area_warnings))
            area_metrics["area_extrapolation_warnings"] = area_warnings
            matrix_sram_inputs = (
                (area_metrics.get("area_new_inputs") or {})
                .get("sram", {})
                .get("area_sram_inputs", {})
                .get("matrix", {})
            )
            if not matrix_sram_inputs:
                matrix_sram_inputs = (
                    (area_metrics.get("sram") or {})
                    .get("area_sram_inputs", {})
                    .get("matrix", {})
                )
            if not matrix_sram_inputs:
                matrix_sram_inputs = (
                    (area_metrics.get("area_new_breakdown") or {})
                    .get("SRAM", {})
                    .get("area_sram_inputs", {})
                    .get("matrix", {})
                )
            matrix_sram_depth = int(
                matrix_sram_inputs.get("depth", hw["MATRIX_SRAM_SIZE"])
            )
            matrix_sram_width = int(matrix_sram_inputs.get("width", 0))
            matrix_sram_logical_bits = matrix_sram_depth * matrix_sram_width
            matrix_sram_logical_mb = matrix_sram_logical_bits / 8.0 / 1e6
            sram_area_metrics = area_metrics.get("sram") or {}
            ideal_sram_mm2 = float(
                sram_area_metrics.get(
                    "ideal_dual_port_sram_area_um2",
                    area_metrics.get("sram_macro_area", 0.0),
                )
            ) / 1e6
            replicated_sram_mm2 = float(
                sram_area_metrics.get(
                    "replicated_single_port_sram_area_um2",
                    area_metrics.get("sram_macro_area", 0.0),
                )
            ) / 1e6

            record.update(
                {
                    "state": "complete",
                    "area": area_um2,
                    "area_um2": area_um2,
                    "area_mm2": area_mm2,
                    "area_uncertainty_p10_mm2": area_p10_mm2,
                    "area_uncertainty_p50_mm2": area_p50_mm2,
                    "area_uncertainty_p90_mm2": area_p90_mm2,
                    **aggregate_area_metrics,
                    "endpoint_area_overhead_fraction": (
                        args.endpoint_area_overhead_pct / 100.0
                        if args.multi_chip_model
                        == "ideal-linear-lower-bound-v1"
                        else None
                    ),
                    "endpoint_area_semantics": (
                        "fixed_nvlink_c2c_port_proxy_v1"
                        if args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
                        else "legacy_core_fraction"
                    ),
                    "matrix_sram_depth": matrix_sram_depth,
                    "matrix_sram_width_bits": matrix_sram_width,
                    "matrix_sram_logical_bits": matrix_sram_logical_bits,
                    "matrix_sram_tiles": hw["MATRIX_SRAM_TILES"],
                    "matrix_sram_logical_mb": matrix_sram_logical_mb,
                    "sram_port_model":
                        "ideal_dual_port_architectural_assumption",
                    "selected_sram_area_mm2": ideal_sram_mm2,
                    "ideal_dual_port_sram_area_mm2": ideal_sram_mm2,
                    "replicated_single_port_sram_area_mm2":
                        replicated_sram_mm2,
                    "dual_port_area_savings_mm2":
                        replicated_sram_mm2 - ideal_sram_mm2,
                    "dual_port_area_savings_pct": (
                        100.0
                        * (replicated_sram_mm2 - ideal_sram_mm2)
                        / replicated_sram_mm2
                        if replicated_sram_mm2
                        else 0.0
                    ),
                    "sram_port_energy_model": "ideal_independent_access",
                    "dual_port_overhead_included": False,
                    "area_budget_constraint_mm2": area_constraint,
                    "a100_area_constraint_mm2": area_constraint,
                    "within_target_area_tolerance": abs(area_mm2 - args.target_area_mm2) <= tolerance,
                    "area_mode": area_metrics.get("area_mode"),
                    "area_model": area_metrics.get("area_model"),
                    "area_breakdown": area_metrics.get(
                        "area_breakdown", area_metrics.get("area_proxy_breakdown", {})
                    ),
                    "area_metrics": area_metrics,
                    "area_extrapolation_warnings": area_warnings,
                    "vector_scalar_area_calibration_status": (
                        vector_scalar_area_status
                    ),
                    "calibration_in_domain": (
                        compiler_cost_report.get("calibration_in_domain")
                        if compiler_cost_report is not None
                        else None
                    ),
                }
            )
            fidelity_issues: list[str] = []
            compute_fidelity = record.get("compute_fidelity_status")
            if compute_fidelity not in {None, "validated"}:
                fidelity_issues.append(f"compute:{compute_fidelity}")
            if record["area_extrapolation_warnings"]:
                fidelity_issues.append("area:extrapolated")
            record["candidate_fidelity"] = (
                "validated" if not fidelity_issues else "exploratory"
            )
            record["candidate_fidelity_issues"] = fidelity_issues
            record["normalized_latency"] = float(record["latency_ms"])
            record["normalized_energy"] = float(
                record["system_energy_nominal_mj"]
            )
            record["objective_normalization"] = OBJECTIVE_NORMALIZATION
            for key_name in ("area_proxy_breakdown", "area_proxy_inputs", "area_new_breakdown", "area_new_inputs"):
                if key_name in area_metrics:
                    record[key_name] = area_metrics[key_name]
            area_violation = area_budget_violation_reason(
                area_mm2,
                args.area_budget_mm2,
            )
            if area_violation is not None:
                raise TrialPrunedError(area_violation)
            heartbeat("serialization")
            record["dse_phase_telemetry_seconds"] = dict(
                sorted(phase_seconds.items())
            )
            if compiler_cost_report is not None:
                record["compiler_phase_telemetry_seconds"] = (
                    compiler_cost_report.get("phase_telemetry_seconds", {})
                )
            persist_trial_record(
                trial_dir,
                record,
                artifact_retention=args.artifact_retention,
            )
            if args.artifact_retention == "full":
                write_json(
                    trial_dir / "latency_report.parsed.json",
                    latency_report,
                )
            return ObjectiveValues.from_trial_record(
                record
            ).as_optuna_values()
        except TrialPrunedError as exc:
            record.update({"state": "pruned", "reason": str(exc)})
            persist_trial_record(
                trial_dir,
                record,
                artifact_retention=args.artifact_retention,
            )
            raise optuna.TrialPruned(str(exc)) from exc
        except KeyboardInterrupt:
            record.update({"state": "failed", "reason": "KeyboardInterrupt"})
            persist_trial_record(
                trial_dir,
                record,
                artifact_retention=args.artifact_retention,
            )
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            record.update(
                {
                    "state": "failed",
                    "reason": reason,
                    "traceback": traceback.format_exc(),
                }
            )
            persist_trial_record(
                trial_dir,
                record,
                artifact_retention=args.artifact_retention,
            )
            raise
        finally:
            phase_seconds[current_phase] += time.perf_counter() - phase_started
            record.setdefault(
                "dse_phase_telemetry_seconds",
                dict(sorted(phase_seconds.items())),
            )
            append_jsonl(
                trials_jsonl,
                (
                    record
                    if args.artifact_retention == "full"
                    else trial_lifecycle_record(record)
                ),
            )
            records.append(record)

    worker_trial_budget = (
        args.worker_trials
        if args.worker_mode and args.worker_trials is not None
        else trials_to_run
    )
    resolved_workers = (
        min(
            DEFAULT_OPTUNA_WORKERS,
            logical_cpu_capacity(),
            max(1, worker_trial_budget),
        )
        if args.workers == "auto"
        else max(1, min(int(args.workers), max(1, worker_trial_budget)))
    )

    def reconcile_after_abnormal_worker_exit() -> None:
        refreshed = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
        )
        result = reconcile_interrupted_trials(refreshed, run_dir)
        if any(result.values()):
            print(f"Immediately reconciled interrupted worker trial: {result}")

    def persistent_work_claim_available() -> bool:
        if not complete_budget_mode:
            return False
        refreshed_trials = study.get_trials(deepcopy=False)
        counts = Counter(trial.state.name for trial in refreshed_trials)
        complete = int(counts.get("COMPLETE", 0))
        running = int(counts.get("RUNNING", 0))
        waiting = int(counts.get("WAITING", 0))
        total = sum(counts.values())
        if complete >= int(target_complete_trials):
            return False
        if complete + running >= int(target_complete_trials):
            return False
        return waiting > 0 or total < int(max_total_attempts)

    try:
        if not args.worker_mode and resolved_workers > 1 and trials_to_run > 0:
            worker_id = next_worker_id(run_dir)
            return_codes, worker_id = launch_worker_processes(
                run_dir,
                resolved_workers,
                trials_to_run,
                worker_id,
                max_trials_per_process=args.worker_max_trials_per_process,
                memory_reserve_gib=args.memory_reserve_gib,
                memory_resume_gib=args.memory_resume_gib,
                memory_emergency_gib=args.memory_emergency_gib,
                initial_worker_rss_gib=args.initial_worker_rss_gib,
                process_tree_rss_limit_gib=(
                    args.worker_process_tree_rss_limit_gib
                ),
                stall_timeout_seconds=args.worker_stall_timeout_seconds,
                launch_burst=args.worker_launch_burst,
                launch_interval_seconds=(
                    args.worker_launch_interval_seconds
                ),
                monitor_interval_seconds=(
                    args.worker_monitor_interval_seconds
                ),
                reconcile_callback=reconcile_after_abnormal_worker_exit,
                persistent_pull_budget=complete_budget_mode,
                work_claim_available=(
                    persistent_work_claim_available
                    if complete_budget_mode
                    else None
                ),
            )
            retry_wave = 0
            previous_progress = (
                initial_complete_trials
                if complete_budget_mode
                else initial_settled_trials
            )
            no_progress_waves = 0
            while True:
                refreshed = optuna.load_study(
                    study_name=study_name, storage=storage, sampler=sampler
                )
                retry_reconciliation = reconcile_interrupted_trials(
                    refreshed, run_dir
                )
                if any(retry_reconciliation.values()):
                    print(
                        "Reconciled incomplete retry wave: "
                        f"{retry_reconciliation}"
                    )
                settled = (
                    _settled_trial_count(refreshed)
                    if grid_total_trials is not None
                    else sum(
                        trial.state.is_finished()
                        for trial in refreshed.get_trials(deepcopy=False)
                    )
                )
                completed = sum(
                    trial.state == optuna.trial.TrialState.COMPLETE
                    for trial in refreshed.get_trials(deepcopy=False)
                )
                finished = sum(
                    trial.state.is_finished()
                    for trial in refreshed.get_trials(deepcopy=False)
                )
                progress = completed if complete_budget_mode else settled
                missing = (
                    int(target_complete_trials) - completed
                    if complete_budget_mode
                    else target_settled_trials - settled
                )
                if missing <= 0:
                    break
                if progress <= previous_progress:
                    no_progress_waves += 1
                else:
                    no_progress_waves = 0
                previous_progress = progress
                if no_progress_waves >= 3:
                    raise RuntimeError(
                        "DSE made no progress for three retry waves: "
                        f"complete={completed}, settled={settled}, missing={missing}"
                    )

                queued_missing = 0
                if grid_total_trials is not None:
                    finalize_redundant_waiting_trials(refreshed)
                    queued_missing = enqueue_missing_grid_trials(
                        refreshed, optuna_search_space
                    )
                attempts_to_launch = missing
                if complete_budget_mode:
                    remaining_attempts = int(max_total_attempts) - finished
                    if remaining_attempts <= 0:
                        raise RuntimeError(
                            "complete-trial target was not reached before "
                            f"max attempts: complete={completed}, "
                            f"target={target_complete_trials}, "
                            f"attempts={finished}, max={max_total_attempts}"
                        )
                    attempts_to_launch = min(missing, remaining_attempts)
                retry_wave += 1
                print(
                    f"Worker retry wave {retry_wave}: complete={completed}, "
                    f"settled={settled}, missing={missing}, "
                    f"attempts_to_launch={attempts_to_launch}, "
                    f"exact_grid_queued={queued_missing}"
                )
                retry_workers = min(resolved_workers, attempts_to_launch)
                retry_codes, worker_id = launch_worker_processes(
                    run_dir,
                    retry_workers,
                    attempts_to_launch,
                    worker_id,
                    max_trials_per_process=args.worker_max_trials_per_process,
                    memory_reserve_gib=args.memory_reserve_gib,
                    memory_resume_gib=args.memory_resume_gib,
                    memory_emergency_gib=args.memory_emergency_gib,
                    initial_worker_rss_gib=args.initial_worker_rss_gib,
                    process_tree_rss_limit_gib=(
                        args.worker_process_tree_rss_limit_gib
                    ),
                    stall_timeout_seconds=args.worker_stall_timeout_seconds,
                    launch_burst=args.worker_launch_burst,
                    launch_interval_seconds=(
                        args.worker_launch_interval_seconds
                    ),
                    monitor_interval_seconds=(
                        args.worker_monitor_interval_seconds
                    ),
                    reconcile_callback=reconcile_after_abnormal_worker_exit,
                    persistent_pull_budget=complete_budget_mode,
                    work_claim_available=(
                        persistent_work_claim_available
                        if complete_budget_mode
                        else None
                    ),
                )
                return_codes.extend(retry_codes)
            study = optuna.load_study(
                study_name=study_name, storage=storage, sampler=sampler
            )
            if any(code != 0 for code in return_codes):
                print(f"Warning: worker return codes: {return_codes}", file=sys.stderr)
        elif args.worker_mode or trials_to_run > 0:
            trial_count = args.worker_trials if args.worker_mode else trials_to_run
            if args.worker_mode:
                optimize_with_serialized_ask(
                    study,
                    objective,
                    n_trials=trial_count,
                    ask_lock_path=run_dir / "study.ask.lock",
                    worker_id=args.worker_id,
                    resource_log_path=run_dir / "worker_resources.jsonl",
                    rss_recycle_gib=args.worker_rss_recycle_gib,
                    memory_reserve_gib=args.memory_reserve_gib,
                    target_complete_trials=(
                        args.target_complete_trials
                        if args.target_complete_trials is not None
                        else None
                    ),
                    max_total_attempts=(
                        args.max_total_attempts
                        if args.target_complete_trials is not None
                        else None
                    ),
                )
            else:
                study.optimize(
                    objective,
                    n_trials=trial_count,
                    gc_after_trial=True,
                    catch=(Exception,),
                )
                if complete_budget_mode:
                    while True:
                        completed = sum(
                            trial.state == optuna.trial.TrialState.COMPLETE
                            for trial in study.get_trials(deepcopy=False)
                        )
                        finished = sum(
                            trial.state.is_finished()
                            for trial in study.get_trials(deepcopy=False)
                        )
                        missing = int(target_complete_trials) - completed
                        if missing <= 0:
                            break
                        remaining = int(max_total_attempts) - finished
                        if remaining <= 0:
                            raise RuntimeError(
                                "complete-trial target was not reached before "
                                f"max attempts: complete={completed}, "
                                f"target={target_complete_trials}"
                            )
                        study.optimize(
                            objective,
                            n_trials=min(missing, remaining),
                            gc_after_trial=True,
                            catch=(Exception,),
                        )
            if args.worker_mode:
                return 0
    finally:
        if snapshot and not args.keep_rtl_config:
            restore_rtl_files(snapshot)

    finalized_waiting_trials = (
        finalize_redundant_waiting_trials(study)
        if grid_total_trials is not None
        else 0
    )
    if finalized_waiting_trials:
        study = optuna.load_study(
            study_name=study_name, storage=storage, sampler=sampler
        )
    records = read_trial_records(
        run_dir,
        model=model,
        seq_len=dse_config.input_seq_len,
        batch_size=dse_config.latency_batch_size,
        native_layout_mode=args.native_layout_mode,
        persist_layout_backfill=True,
    )
    grid_records = (
        canonical_grid_records(study, records)
        if grid_total_trials is not None
        else records
    )
    completed_records = [
        record for record in grid_records if record.get("state") == "complete"
    ]
    with trials_jsonl.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_records_csv(run_dir / "all_trials.csv", records)
    if grid_total_trials is not None:
        write_records_csv(run_dir / "grid_trials.csv", grid_records)
    pareto_numbers = {trial.number for trial in study.best_trials}
    pareto_records = [record for record in completed_records if int(record["trial"]) in pareto_numbers]
    write_records_csv(run_dir / "pareto_trials.csv", pareto_records)
    physical_candidate_bank = build_physical_candidate_bank(
        completed_records
    )
    write_json(
        run_dir / "physical_candidate_bank.json",
        physical_candidate_bank,
    )
    write_multi_chip_analysis(
        run_dir,
        completed_records,
        target_area_mm2=args.target_area_mm2,
    )

    selections = select_area_reference_candidates(
        completed_records,
        target_area_mm2=args.target_area_mm2,
        area_budget_mm2=args.area_budget_mm2,
        target_area_tolerance_pct=args.target_area_tolerance_pct,
    )
    feasible = selections["feasible"]
    fastest = selections["fastest"]
    lowest_energy = selections["lowest_energy"]
    fidelity_qualified = selections["fidelity_qualified"]
    fastest_fidelity_qualified = selections["fastest_fidelity_qualified"]
    highest_accuracy = selections["highest_accuracy"]
    closest_to_target = selections["closest_to_target"]
    closest_below_target = selections["closest_below_target"]
    within_tolerance = selections["within_tolerance"]
    fastest_within_tolerance = selections["fastest_within_tolerance"]
    lowest_energy_within_tolerance = selections[
        "lowest_energy_within_tolerance"
    ]
    best_energy_delay_product = selections["best_energy_delay_product"]
    smallest_design_beating_a100_area_candidate = selections[
        "smallest_design_beating_a100_area_candidate"
    ]
    p90_feasible = selections["p90_feasible"]
    p90_fastest = selections["p90_fastest"]
    p90_closest_to_target = selections["p90_closest_to_target"]
    named_selector_records = (
        fastest,
        lowest_energy,
        fastest_fidelity_qualified,
        highest_accuracy,
        closest_to_target,
        closest_below_target,
        fastest_within_tolerance,
        lowest_energy_within_tolerance,
        best_energy_delay_product,
        smallest_design_beating_a100_area_candidate,
        p90_fastest,
        p90_closest_to_target,
    )
    best_records_by_trial = {
        int(record["trial"]): record
        for record in named_selector_records
        if record is not None
    }
    write_best_csv(
        run_dir / "best_trials.csv",
        list(best_records_by_trial.values()),
    )
    retained_detail_trial_ids = set(pareto_numbers) | set(
        best_records_by_trial
    )
    final_state_counts = Counter(
        str(record.get("state", "missing")) for record in records
    )
    worker_resource_summary = summarize_worker_resources(
        run_dir / "worker_resources.jsonl",
        requested_workers=resolved_workers,
    )
    write_json(
        run_dir / "a100_comparison.json",
        {
            "target_area_mm2": args.target_area_mm2,
            "area_budget_mm2": args.area_budget_mm2,
            "target_area_tolerance_pct": args.target_area_tolerance_pct,
            "reference": (
                f"{args.reference_a100_count} x NVIDIA A100 826 mm2 "
                "aggregate die-area reference with a 110% feasibility budget"
            ),
            "ga100_reference_area_mm2": GA100_REFERENCE_AREA_MM2,
            "reference_a100_count": args.reference_a100_count,
            "note": (
                "PLENA area is a calibrated logic plus SRAM-macro proxy and excludes physical "
                "HBM stacks/package. Candidate fidelity must be checked separately: large "
                "MLEN/BLEN points and unsupported RTL-v1 opcodes are exploratory."
            ),
            "feasible_trial_count": len(feasible),
            "fidelity_qualified_trial_count": len(fidelity_qualified),
            "fastest_under_area_budget": selector_trial_summary(fastest),
            "lowest_energy_under_area_budget": selector_trial_summary(
                lowest_energy
            ),
            "fastest_fidelity_qualified_under_area_budget":
                selector_trial_summary(fastest_fidelity_qualified),
            "highest_accuracy_under_area_budget": selector_trial_summary(
                highest_accuracy
            ),
            "highest_accuracy_under_a100_budget": selector_trial_summary(
                highest_accuracy
            ),
            "closest_area_to_target_mm2": selector_trial_summary(
                closest_to_target
            ),
            "closest_area_below_target_mm2": selector_trial_summary(
                closest_below_target
            ),
            "closest_area_below_826": selector_trial_summary(
                closest_below_target
            ),
            "within_target_area_tolerance_trial_ids": [
                int(record["trial"]) for record in within_tolerance
            ],
            "within_5_percent_of_a100_trial_ids": [
                int(record["trial"]) for record in within_tolerance
            ],
            "fastest_within_5pct_of_826": selector_trial_summary(
                fastest_within_tolerance
            ),
            "lowest_energy_within_5pct_of_826": (
                selector_trial_summary(lowest_energy_within_tolerance)
            ),
            "best_energy_delay_product_under_budget": (
                selector_trial_summary(best_energy_delay_product)
            ),
            "smallest_design_beating_a100_area_candidate": (
                selector_trial_summary(
                    smallest_design_beating_a100_area_candidate
                )
            ),
            "p90_conservative_feasible_trial_count": len(p90_feasible),
            "p90_conservative_fastest_under_area_budget":
                selector_trial_summary(p90_fastest),
            "p90_conservative_closest_area_to_target_mm2":
                selector_trial_summary(p90_closest_to_target),
        },
    )
    write_json(
        run_dir / "run_summary.json",
        {
            "run_dir": str(run_dir),
            "model_config": str(args.model_config),
            "latency_model": latency_model_name,
            "input_seq_len": dse_config.input_seq_len,
            "output_seq_len": dse_config.output_seq_len,
            "device_num": dse_config.device_num,
            "n_trials": (
                grid_total_trials
                or args.target_complete_trials
                or args.n_trials
            ),
            "target_complete_trials": args.target_complete_trials,
            "max_total_attempts": args.max_total_attempts,
            "final_state_counts": dict(final_state_counts),
            "effective_complete_rate": (
                final_state_counts.get("complete", 0) / sum(final_state_counts.values())
                if final_state_counts
                else 0.0
            ),
            "resume_initial_settled_trials": initial_settled_trials,
            "resume_trials_requested": trials_to_run,
            "interrupted_trial_reconciliation": reconciliation,
            "finalized_redundant_waiting_trials": finalized_waiting_trials,
            "workers": (
                resolved_workers
                if trials_to_run > 0 or args.worker_mode
                else min(DEFAULT_OPTUNA_WORKERS, logical_cpu_capacity())
                if args.workers == "auto"
                else int(args.workers)
            ),
            "workers_requested": args.workers,
            "worker_max_trials_per_process": args.worker_max_trials_per_process,
            "worker_rss_recycle_gib": args.worker_rss_recycle_gib,
            "initial_worker_rss_gib": args.initial_worker_rss_gib,
            "memory_reserve_gib": args.memory_reserve_gib,
            "memory_resume_gib": args.memory_resume_gib,
            "memory_emergency_gib": args.memory_emergency_gib,
            "worker_process_tree_rss_limit_gib": (
                args.worker_process_tree_rss_limit_gib
            ),
            "worker_stall_timeout_seconds": (
                args.worker_stall_timeout_seconds
            ),
            "worker_launch_burst": args.worker_launch_burst,
            "worker_launch_interval_seconds": (
                args.worker_launch_interval_seconds
            ),
            "worker_monitor_interval_seconds": (
                args.worker_monitor_interval_seconds
            ),
            "worker_resource_policy": DEFAULT_WORKER_POLICY.as_metadata(),
            "worker_resource_summary": worker_resource_summary,
            "global_cache_schema": GLOBAL_DSE_CACHE_SCHEMA,
            "cache_scope": cache_scope,
            "global_cache_dir": str(cache_directories.root),
            "compiler_cost_source_sha256": compiler_cost_source_hash,
            "area_model_source_sha256": area_model_source_hash,
            "compiler_cache_tier_counts": dict(
                Counter(
                    str(record.get("compiler_cost_cache_tier", "not_used"))
                    for record in completed_records
                )
            ),
            "area_cache_tier_counts": dict(
                Counter(
                    str(record.get("area_cache_tier", "not_used"))
                    for record in completed_records
                )
            ),
            "physical_candidate_count": len(physical_candidate_bank),
            "optuna_storage_backend": optuna_storage_backend,
            "serialized_optuna_ask": resolved_workers > 1,
            "sampler": args.sampler,
            "model_profile": args.model_profile,
            "model_profile_fidelity": selected_profile_fidelity,
            "objective_schema": objective_schema,
            "objective_normalization": OBJECTIVE_NORMALIZATION,
            "objective_directions": [
                "minimize_normalized_latency",
                "minimize_normalized_energy",
            ],
            "search_schema": effective_search_schema,
            "search_encoding": search_encoding,
            "precision_search_encoding": precision_search_encoding,
            "precision_signature_schema": PRECISION_SIGNATURE_SCHEMA,
            "hardware_domain_fingerprint": hardware_domain_fingerprint,
            "shape_domain_policy": SHAPE_DOMAIN_POLICY,
            "canonical_hardware_domain_size": canonical_hardware_domain_size,
            "canonical_full_candidate_domain_size": (
                canonical_hardware_domain_size * len(precision_profiles)
            ),
            "sampler_configuration": {
                "name": args.sampler,
                "tpe_startup_trials": args.tpe_startup_trials,
                "tpe_ei_candidates": args.tpe_ei_candidates,
                "multivariate": args.sampler == "tpe",
                "group": args.sampler == "tpe",
                "constant_liar": args.sampler == "tpe",
            },
            "accuracy_constraints": str(args.accuracy_constraints),
            "precision_profile_count": len(precision_profiles),
            "matrix_datapath_signature_count": len(
                matrix_datapath_signatures
            ),
            "precision_variants_per_signature": {
                signature.signature_id: len(signature.profile_names)
                for signature in matrix_datapath_signatures
            },
            "area_mode": args.area_mode,
            "target_area_mm2": args.target_area_mm2,
            "area_budget_mm2": args.area_budget_mm2,
            "target_area_tolerance_pct": args.target_area_tolerance_pct,
            "strict_bandwidth": args.legacy_bandwidth_prune,
            "legacy_bandwidth_prune": args.legacy_bandwidth_prune,
            "legacy_bandwidth_policy": args.legacy_bandwidth_policy,
            "compiler_trace_granularity": args.compiler_trace_granularity,
            "trial_report_materialization": args.trial_report_materialization,
            "artifact_retention": args.artifact_retention,
            "chip_counts": list(chip_counts),
            "physical_chip_counts": list(chip_counts),
            "chip_count_search_values": list(chip_count_search_values),
            "chip_count_scaling": args.chip_count_scaling,
            "parallel_models": list(parallel_models),
            "multi_chip_model": args.multi_chip_model,
            "parallel_kernel_census_schema": (
                "parallel_kernel_census_v2_schedule_lineage"
                if args.multi_chip_model in {
                    "tile-aware-tp-cp-ep-v3",
                    TILE_AWARE_DP_MULTI_CHIP_MODEL,
                }
                else None
            ),
            "tile_accounting_schema": (
                "whole_request_dp_rank_local_compiler_planner_v4"
                if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL
                else "balanced_partition_compiler_planner_v3"
                if args.multi_chip_model == "tile-aware-tp-cp-ep-v3"
                else None
            ),
            "ep_topology_schema": (
                "independent_ep_axis_contiguous_expert_partition_v1"
                if args.multi_chip_model == TILE_AWARE_DP_MULTI_CHIP_MODEL
                else "ep_reuses_cp_contiguous_expert_partition_v1"
                if args.multi_chip_model == "tile-aware-tp-cp-ep-v3"
                else None
            ),
            "energy_action_lineage_schema": (
                "energy_action_kernel_lineage_v3_structural_families"
                if args.multi_chip_model in {
                    "tile-aware-tp-cp-ep-v3",
                    TILE_AWARE_DP_MULTI_CHIP_MODEL,
                }
                else None
            ),
            "rank_power_aggregation_schema": (
                "sum_rank_energy_after_per_rank_clock_cap_v2"
                if args.multi_chip_model in {
                    "tile-aware-tp-cp-ep-v3",
                    TILE_AWARE_DP_MULTI_CHIP_MODEL,
                }
                else None
            ),
            "dp_degrees": args.dp_degrees,
            "fixed_dp_degree": args.fixed_dp_degree,
            "tp_degrees": args.tp_degrees,
            "fixed_tp_degree": args.fixed_tp_degree,
            "ep_degrees": args.ep_degrees,
            "fixed_ep_degree": args.fixed_ep_degree,
            "nvlink_port_counts": list(nvlink_port_counts),
            "nvlink_bandwidth_semantics": args.nvlink_bandwidth_semantics,
            "nvlink_port_bidirectional_bandwidth_gbps": (
                DEFAULT_NVLINK_PORT_BIDIRECTIONAL_GBPS
            ),
            "nvlink_startup_us": args.nvlink_startup_us,
            "reference_a100_count": args.reference_a100_count,
            "decode_chip_count": decode_chip_count,
            "decode_nvlink_port_count": args.decode_nvlink_port_count,
            "decode_kv_handoff_schema": "dual_endpoint_peak_bandwidth_v1",
            "matrix_sram_base_tiles": list(base_matrix_sram_tiles),
            "matrix_sram_search_tiles": list(matrix_sram_search_space),
            "matrix_sram_policies": list(matrix_sram_policies),
            "matrix_sram_policy_schema": "partial_resident_prefix_v1",
            "sram_port_model": CURRENT_DSE_PROFILE.sram_port_model,
            "endpoint_area_overhead_pct": (
                args.endpoint_area_overhead_pct
                if args.multi_chip_model == "ideal-linear-lower-bound-v1"
                else None
            ),
            "endpoint_area_model": (
                "fixed_nvlink_c2c_port_proxy_v1"
                if args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
                else "legacy_core_fraction"
            ),
            "endpoint_area_mm2_per_port": (
                dict(ENDPOINT_AREA_MM2_PER_PORT)
                if args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
                else None
            ),
            "nvlink_bidirectional_bandwidth_gbps": (
                args.nvlink_bandwidth_gbps
            ),
            "nvlink_effective_one_way_bandwidth_gbps": (
                "port_count * 450"
                if args.multi_chip_model in FACTORIZED_MULTI_CHIP_MODELS
                else args.nvlink_bandwidth_gbps / 2.0
            ),
            "compiler_cost_mode": args.compiler_cost_mode,
            "compiler_compute_timing": args.compiler_compute_timing,
            "compiler_scheduled_shadow": args.compiler_scheduled_shadow,
            "compiler_v4_memory_evaluation": args.compiler_v4_memory_evaluation,
            "packed_attention_schedule": args.packed_attention_schedule,
            "softmax_state_schedule": args.softmax_state_schedule,
            "packed_qk_schedule": args.packed_qk_schedule,
            "vector_scalar_schedule": args.vector_scalar_schedule,
            "softmax_vector_schedule": args.softmax_vector_schedule,
            "pv_accumulation_schedule": args.pv_accumulation_schedule,
            "softmax_row_lane_domain": list(softmax_row_lane_domain),
            "selector_schedule": args.selector_schedule,
            "reduction_output_mode": args.reduction_output_mode,
            "gqa_pipeline_schedule": args.gqa_pipeline_schedule,
            "address_generation_mode": args.address_generation_mode,
            "ffn_address_schedule": args.ffn_address_schedule,
            "ffn_projection_schedule": args.ffn_projection_schedule,
            "clock_gating_mode": args.clock_gating_mode,
            "compiler_cost_settings": (
                str(args.compiler_cost_settings) if args.compiler_cost_settings else None
            ),
            "compiler_cost_calibration": (
                str(args.compiler_cost_calibration)
                if args.compiler_cost_calibration
                else None
            ),
            "hbm_bandwidth_gbps": dse_config.hbm_bandwidth_gbps,
            "frequency_ghz": dse_config.frequency_ghz,
            "bandwidth_limit_bits_per_cycle": dse_config.bandwidth_limit_bits_per_cycle,
            "mx_scale_width": dse_config.mx_scale_width,
            "mx_scale_block_size": dse_config.mx_scale_block_size,
            "hbm_capacity_bytes": dse_config.hbm_capacity_bytes,
            "kv_capacity_mode": dse_config.kv_capacity_mode,
            "kv_capacity_semantics": (
                "one_attention_layer_plus_handoff_staging;"
                "full_decoder_cache_is_shadow"
                if dse_config.kv_capacity_mode
                == KV_CAPACITY_MODE_STREAMED_HANDOFF_V1
                else "full_decoder_kv_cache_resident_on_prefill_hbm"
            ),
            "kv_handoff_staging_layers": (
                dse_config.kv_handoff_staging_layers
            ),
            "external_memory_energy_artifact": str(
                args.external_memory_energy_artifact
            ),
            "external_memory_energy_artifact_sha256": file_sha256(
                args.external_memory_energy_artifact
            ),
            "sram_background_energy_artifact": str(
                args.sram_background_energy_artifact
            ),
            "sram_background_energy_artifact_sha256": file_sha256(
                args.sram_background_energy_artifact
            ),
            "interconnect_energy_artifact": str(
                args.interconnect_energy_artifact
            ),
            "interconnect_energy_artifact_sha256": file_sha256(
                args.interconnect_energy_artifact
            ),
            "precision_profile_artifact_sha256": file_sha256(
                args.accuracy_constraints
            ),
            "weight_param_count": dse_config.weight_param_count,
            "weight_element_bits": dse_config.weight_element_bits,
            "weight_precision_fallback": dse_config.weight_precision,
            "tie_vlen_to_mlen": True,
            "min_matrix_k_splits": args.min_matrix_k_splits,
            "fidelity_qualified_completed": sum(
                1
                for record in completed_records
                if record.get("candidate_fidelity") == "validated"
            ),
            "completed": sum(
                1 for r in grid_records if r.get("state") == "complete"
            ),
            "pruned": sum(
                1 for r in grid_records if r.get("state") == "pruned"
            ),
            "failed": sum(1 for r in records if r.get("state") == "failed"),
            "attempt_count": len(records),
            "unique_grid_record_count": len(grid_records),
        },
    )
    if args.artifact_retention == "compact":
        artifact_manifest = finalize_compact_artifacts(
            run_dir,
            retained_trial_ids=retained_detail_trial_ids,
        )
        write_json(run_dir / "artifact_manifest.json", artifact_manifest)
        run_summary_path = run_dir / "run_summary.json"
        run_summary = load_json(run_summary_path)
        run_summary["artifact_manifest"] = artifact_manifest
        write_json(run_summary_path, run_summary)
    try:
        from plot_four_objective_dse import plot_all

        plot_all(run_dir)
    except Exception as exc:
        print(
            f"Warning: latency-energy plots were not generated: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    print(f"Wrote DSE run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
