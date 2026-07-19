#!/usr/bin/env python3
"""Optuna DSE for Qwen3-32B prefill latency, accuracy, and chip area.

Accuracy comes from external precision profiles. The default objective uses
the native compiler CostEmitter with rtl-v1 compute timing, production-DMA V4
memory work, and the shared compact decoder layout; legacy analytic modes
remain available for diagnostics. Area defaults to the precision-aware
calibrated proxy and can optionally use PLENA_RTL synthesis/elaboration modes.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import fcntl
import gc
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import optuna
import toml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PLENA_COMPILER_ROOT = REPO_ROOT / "PLENA_Compiler"
if str(PLENA_COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(PLENA_COMPILER_ROOT))

from compiler.aten.plena.native_layout import (
    SequencePackingPlan,
    build_attention_head_packing,
)

WORKSPACE_ROOT = Path(__file__).resolve().parent
RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
MODEL_CONFIG = REPO_ROOT / "Workspace/qwen3_32b_dense_analytic/qwen3-32b.json"
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
DEFAULT_WEIGHT_PARAM_COUNT = 32_000_000_000
DEFAULT_WEIGHT_ELEMENT_BITS = 4.0
DEFAULT_WEIGHT_PRECISION = "MXINT4"
DEFAULT_MX_SCALE_WIDTH = 8
DEFAULT_MX_SCALE_BLOCK_SIZE = 64
DEFAULT_WEIGHT_MX_EXP_WIDTH = 2
DEFAULT_WEIGHT_MX_MANT_WIDTH = 1
LATENCY_MODEL_NAME = "compiler_integrated_compute_cost_rtl_v1"
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
COMPILER_COST_OBJECTIVE_MODES = {
    "compute-objective",
    "roofline-objective",
    "objective",
}

DEFAULT_SEARCH_SPACE = {
    "MLEN": [128, 256, 512, 1024, 2048],
    "BLEN": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def stable_key(data: dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


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


def derived_hardware(model: dict[str, Any], trial_params: dict[str, Any], config: DSEConfig) -> dict[str, int]:
    mlen = int(trial_params["MLEN"])
    vlen = int(trial_params["VLEN"])
    blen = int(trial_params["BLEN"])
    return {
        "MLEN": mlen,
        "VLEN": vlen,
        "BLEN": blen,
        "HLEN": DEFAULT_HLEN,
        "BROADCAST_AMOUNT": DEFAULT_BROADCAST_AMOUNT,
        "MATRIX_SRAM_SIZE": 2 * mlen,
        "VECTOR_SRAM_SIZE": 2 * model["head_dim"] + math.ceil(model["hidden_size"] / vlen),
        "INT_SRAM_DEPTH": 32,
        "FP_CONSTANT_NUM": config.fp_constant_num,
        "FP_SRAM_DEPTH": 3 * mlen + config.fp_constant_num,
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
    if hw["FP_SRAM_DEPTH"] < 3 * hw["MLEN"] + config.fp_constant_num:
        issues.append("FP_SRAM_DEPTH < 3 * MLEN + FP_CONSTANT_NUM")

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
    kv_bytes_per_request = (
        config.input_seq_len
        * model["num_hidden_layers"]
        * 2
        * model["num_key_value_heads"]
        * model["head_dim"]
        * kv_bits
        / 8
    )
    hbm_capacity_max_batch = max(1, math.floor(remaining_hbm_bytes / kv_bytes_per_request))
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
        "kv_bytes_per_request": kv_bytes_per_request,
        "mx_scale_width": config.mx_scale_width,
        "mx_scale_block_size": config.mx_scale_block_size,
        "batch_policy": "fixed_latency_batch_with_a100_80gb_weight_plus_kv_capacity_upper_bound",
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


def run_area_proxy_v2(hw: dict[str, int], precision: dict[str, Any], config: DSEConfig) -> dict[str, Any]:
    from analytic_models.area_new import estimate_area

    proxy_inputs = build_area_proxy_inputs(hw, precision, config)
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
        "HBM_M_Prefetch_Amount": hw["HBM_M_Prefetch_Amount"],
        "HBM_V_Prefetch_Amount": hw["HBM_V_Prefetch_Amount"],
        "HBM_V_Writeback_Amount": hw["HBM_V_Writeback_Amount"],
        "CLOCK_PERIOD_PS": round(1000.0 / config_args.frequency_ghz),
    }
    for name, value in values.items():
        if name not in config and name != "CLOCK_PERIOD_PS":
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
    vector_scalar_schedule: str,
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
        trace, report = compile_and_evaluate_compiler_cost(
            MODEL_CONFIG,
            settings_path,
            calibration,
            seq_len=config_args.input_seq_len,
            batch_size=config_args.latency_batch_size,
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
            vector_scalar_schedule=vector_scalar_schedule,
            persistent_trace_cache_dir=trial_dir.parent / "compiler_trace_cache",
            persistent_v4_work_cache_dir=(
                trial_dir.parent / "compiler_v4_work_cache"
            ),
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
            "dma_coverage": trace.metadata.get("dma_coverage"),
            "native_layout": trace.metadata.get("native_layout"),
            "attention_schedule": trace.metadata.get("attention_schedule"),
            "packed_attention": trace.metadata.get("packed_attention"),
            "vector_scalar_optimization": trace.metadata.get(
                "vector_scalar_optimization"
            ),
            "persistent_trace_cache_hit": trace.metadata.get(
                "persistent_trace_cache_hit", False
            ),
            "persistent_trace_cache_key": trace.metadata.get(
                "persistent_trace_cache_key"
            ),
        }
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
    native_layout = trace_metadata.get("native_layout") or {}
    attention_layout = trace_metadata.get("attention_schedule") or {}
    packed_attention = trace_metadata.get("packed_attention") or {}
    vector_scalar = trace_metadata.get("vector_scalar_optimization") or {}
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
        "attention_storage_block_count": head_layout.get("storage_block_count"),
        "attention_groups_per_storage_block": head_layout.get(
            "groups_per_storage_block"
        ),
        "attention_group_broadcast": group_broadcast,
        "attention_hardware_broadcast": hardware_broadcast,
        "attention_schedule_layout": attention_layout,
        "packed_attention_schedule": packed_attention.get("packed_attention_schedule"),
        "softmax_first_block_specialized_count": packed_attention.get(
            "softmax_first_block_specialized_count"
        ),
        "softmax_state_initializations_elided": packed_attention.get(
            "softmax_state_initializations_elided"
        ),
        "temporary_o_matrices_elided": packed_attention.get(
            "temporary_o_matrices_elided"
        ),
        "direct_o_lane_updates": packed_attention.get("direct_o_lane_updates"),
        "qk_compute_count": packed_attention.get("qk_compute_count"),
        "pv_compute_count": packed_attention.get("pv_compute_count"),
        "qk_recompute_factor": packed_attention.get("qk_recompute_factor"),
        "kv_reload_factor": packed_attention.get("kv_reload_factor"),
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
            for name in ("latency_ms", "area_mm2", "accuracy_score")
        ):
            for name in (
                "area_budget_constraint_mm2",
                "a100_area_constraint_mm2",
                "area_mm2",
            ):
                if record.get(name) is not None:
                    storage.set_trial_user_attr(trial._trial_id, name, record[name])
            storage.set_trial_state_values(
                trial._trial_id,
                optuna.trial.TrialState.COMPLETE,
                [
                    float(record["latency_ms"]),
                    float(record["area_mm2"]),
                    float(record["accuracy_score"]),
                ],
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
        "trial", "state", "reason", "latency_ms", "latency_source",
        "compiler_compute_latency_ms", "compiler_memory_latency_ms",
        "compiler_roofline_latency_ms",
        "compiler_serial_latency_ms", "compiler_memory_model_version",
        "compiler_memory_evaluation_mode", "compiler_cost_cache_hit",
        "v3_memory_latency_ms", "v3_serial_latency_ms",
        "area_um2", "area_mm2", "area_uncertainty_p10_mm2",
        "area_uncertainty_p50_mm2", "area_uncertainty_p90_mm2",
        "area_budget_constraint_mm2", "a100_area_constraint_mm2",
        "within_target_area_tolerance",
        "accuracy_score", "precision_profile", "weight_precision", "MLEN", "VLEN", "BLEN",
        "MATRIX_K_SPLITS", "HLEN", "BROADCAST_AMOUNT", "INT_DATA_WIDTH", "MATRIX_SRAM_SIZE", "VECTOR_SRAM_SIZE",
        "INT_SRAM_DEPTH", "FP_SRAM_DEPTH", "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount", "HBM_V_Writeback_Amount", "calibration_in_domain",
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


def launch_worker_processes(
    run_dir: Path,
    workers: int,
    total_trials: int,
    start_worker_id: int = 0,
    *,
    max_trials_per_process: int = 4,
) -> tuple[list[int], int]:
    """Run a bounded-memory pool of short-lived Optuna worker processes.

    CostEmitter traces use native/PyTorch allocations whose high-water pages
    are not always returned to the OS by ``gc`` or ``malloc_trim``. Recycling
    a worker after a small number of trials is the only reliable reclamation
    boundary. Replacement workers are launched as soon as one exits, so the
    requested process-level parallelism remains saturated without waiting for
    a whole wave of stragglers.
    """

    if workers <= 0 or total_trials <= 0 or max_trials_per_process <= 0:
        raise ValueError(
            "workers, total_trials, and max_trials_per_process must be positive"
        )
    base_args = _strip_worker_cli(sys.argv[1:])
    active: list[tuple[subprocess.Popen[str], Any]] = []
    return_codes: list[int] = []
    trials_assigned = 0
    next_worker_id = start_worker_id

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
        active.append(
            (
                subprocess.Popen(
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
                log_handle,
            )
        )

    def fill_pool() -> None:
        nonlocal trials_assigned
        while len(active) < workers and trials_assigned < total_trials:
            quota = min(max_trials_per_process, total_trials - trials_assigned)
            spawn_one(quota)
            trials_assigned += quota

    fill_pool()
    try:
        while active:
            completed = [entry for entry in active if entry[0].poll() is not None]
            if not completed:
                time.sleep(0.2)
                continue
            for process, log_handle in completed:
                active.remove((process, log_handle))
                return_codes.append(int(process.returncode or 0))
                log_handle.close()
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
) -> None:
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
    ask_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with ask_lock_path.open("a+") as ask_lock:
        for _ in range(n_trials):
            fcntl.flock(ask_lock.fileno(), fcntl.LOCK_EX)
            try:
                sync = getattr(study._storage, "_sync_with_backend", None)
                if sync is not None:
                    sync()
                trial = study.ask()
            finally:
                fcntl.flock(ask_lock.fileno(), fcntl.LOCK_UN)

            try:
                values = objective(trial)
            except optuna.TrialPruned:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            except KeyboardInterrupt:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                raise
            except Exception:
                # Match study.optimize(..., catch=(Exception,)): preserve the
                # failed attempt and continue with the worker's remaining
                # quota instead of terminating the process.
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            else:
                study.tell(trial, values=values)
            finally:
                gc.collect()


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


def select_area_reference_candidates(
    completed_records: list[dict[str, Any]],
    *,
    target_area_mm2: float,
    area_budget_mm2: float,
    target_area_tolerance_pct: float,
) -> dict[str, Any]:
    """Select report candidates with deterministic objective tie-breaking."""

    feasible = [
        record
        for record in completed_records
        if float(record["area_mm2"]) <= area_budget_mm2
    ]

    def fastest_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(record["latency_ms"]),
            -float(record["accuracy_score"]),
            float(record["area_mm2"]),
            int(record.get("trial", -1)),
        )

    fastest = min(feasible, key=fastest_key) if feasible else None
    fidelity_qualified = [
        record for record in feasible if record.get("candidate_fidelity") == "validated"
    ]
    fastest_fidelity_qualified = (
        min(fidelity_qualified, key=fastest_key) if fidelity_qualified else None
    )
    highest_accuracy = (
        max(
            feasible,
            key=lambda record: (
                float(record["accuracy_score"]),
                -float(record["latency_ms"]),
                -float(record["area_mm2"]),
                -int(record.get("trial", -1)),
            ),
        )
        if feasible
        else None
    )
    closest_to_target = (
        min(
            feasible,
            key=lambda record: (
                abs(float(record["area_mm2"]) - target_area_mm2),
                -float(record["accuracy_score"]),
                float(record["latency_ms"]),
                int(record.get("trial", -1)),
            ),
        )
        if feasible
        else None
    )
    below_target = [
        record for record in feasible if float(record["area_mm2"]) <= target_area_mm2
    ]
    closest_below_target = (
        max(
            below_target,
            key=lambda record: (
                float(record["area_mm2"]),
                float(record["accuracy_score"]),
                -float(record["latency_ms"]),
                -int(record.get("trial", -1)),
            ),
        )
        if below_target
        else None
    )
    tolerance_mm2 = target_area_mm2 * target_area_tolerance_pct / 100.0
    within_tolerance = [
        record
        for record in feasible
        if abs(float(record["area_mm2"]) - target_area_mm2) <= tolerance_mm2
    ]
    p90_feasible = [
        record
        for record in completed_records
        if float(record.get("area_uncertainty_p90_mm2", record["area_mm2"]))
        <= area_budget_mm2
    ]

    def p90_fastest_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(record["latency_ms"]),
            -float(record["accuracy_score"]),
            float(record.get("area_uncertainty_p90_mm2", record["area_mm2"])),
            int(record.get("trial", -1)),
        )

    p90_fastest = min(p90_feasible, key=p90_fastest_key) if p90_feasible else None
    p90_closest_to_target = (
        min(
            p90_feasible,
            key=lambda record: (
                abs(
                    float(
                        record.get("area_uncertainty_p90_mm2", record["area_mm2"])
                    )
                    - target_area_mm2
                ),
                -float(record["accuracy_score"]),
                float(record["latency_ms"]),
                int(record.get("trial", -1)),
            ),
        )
        if p90_feasible
        else None
    )
    return {
        "feasible": feasible,
        "fastest": fastest,
        "fidelity_qualified": fidelity_qualified,
        "fastest_fidelity_qualified": fastest_fidelity_qualified,
        "highest_accuracy": highest_accuracy,
        "closest_to_target": closest_to_target,
        "closest_below_target": closest_below_target,
        "within_tolerance": within_tolerance,
        "p90_feasible": p90_feasible,
        "p90_fastest": p90_fastest,
        "p90_closest_to_target": p90_closest_to_target,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna DSE for Qwen3-32B dense prefill")
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
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
        help="HBM capacity used for the weight+KV max-batch upper bound",
    )
    parser.add_argument(
        "--hbm-bandwidth-gbps",
        type=float,
        default=DEFAULT_BANDWIDTH_LIMIT_GBPS,
        help="HBM bandwidth in decimal GB/s. At 1 GHz, 2039 GB/s is 2039 bytes/cycle.",
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
    parser.add_argument("--fixed-mlen", type=int, default=None)
    parser.add_argument("--fixed-blen", type=int, default=None)
    parser.add_argument("--fixed-vlen", type=int, default=None)
    parser.add_argument("--fixed-int-data-width", type=int, default=None)
    parser.add_argument("--fixed-precision-profile", type=str, default=None)
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
        default=True,
        help="Preserve the historical fixed-bandwidth prune decision (default: true)",
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
        default="compute-objective",
        help=(
            "Record the selected memory model beside the legacy model, use CostEmitter "
            "compute with the legacy bandwidth guard, use the stage-wise RTL-v1/V4 "
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
        choices=("rtl-v1", "legacy"),
        default="rtl-v1",
        help="CostEmitter compute timing source; rtl-v1 is calibrated resource work",
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
        "--vector-scalar-schedule",
        choices=("compiler-v1", "legacy"),
        default="compiler-v1",
        help="Native Vector/Scalar compiler lowering (default: compiler-v1).",
    )
    parser.add_argument("--keep-rtl-config", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--sampler",
        choices=("nsga2", "grid"),
        default="nsga2",
        help="Optuna sampler. Use grid for exhaustive enumeration of the current categorical search space.",
    )
    parser.add_argument("--target-area-mm2", type=float, default=DEFAULT_TARGET_AREA_MM2)
    parser.add_argument(
        "--area-budget-mm2",
        type=float,
        default=DEFAULT_AREA_BUDGET_MM2,
        help="Hard area feasibility constraint; defaults to 110%% of the 826 mm2 GA100 reference",
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
        default=4,
        help=(
            "Recycle each parallel worker after this many trials to bound "
            "CostEmitter native allocator RSS (default: 4)"
        ),
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
    if args.dry_run:
        args.area_mode = "none"
    if args.latency_batch_size <= 0:
        raise ValueError(f"--latency-batch-size must be positive, got {args.latency_batch_size}")
    if args.frequency_ghz <= 0:
        raise ValueError(f"--frequency-ghz must be positive, got {args.frequency_ghz}")
    if args.mx_scale_block_size <= 0:
        raise ValueError(f"--mx-scale-block-size must be positive, got {args.mx_scale_block_size}")
    if args.hbm_capacity_bytes <= 0:
        raise ValueError(f"--hbm-capacity-bytes must be positive, got {args.hbm_capacity_bytes}")
    if args.n_trials <= 0:
        raise ValueError(f"--n-trials must be positive, got {args.n_trials}")
    if args.worker_max_trials_per_process <= 0:
        raise ValueError(
            "--worker-max-trials-per-process must be positive, got "
            f"{args.worker_max_trials_per_process}"
        )
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
    if args.compiler_cost_mode == "compute-objective" and not args.legacy_bandwidth_prune:
        raise ValueError(
            "--compiler-cost-mode compute-objective requires --legacy-bandwidth-prune"
        )
    if args.fixed_mlen is not None and args.fixed_vlen is not None and args.fixed_mlen != args.fixed_vlen:
        raise ValueError("--fixed-vlen must match --fixed-mlen when VLEN is tied to MLEN")
    if args.area_mode in {"synth", "elaborate", "parse-existing"} and not args.worker_mode:
        requested_workers = (os.cpu_count() or 1) if args.workers == "auto" else int(args.workers)
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
        hbm_capacity_bytes=args.hbm_capacity_bytes,
        hbm_bandwidth_gbps=args.hbm_bandwidth_gbps,
        frequency_ghz=args.frequency_ghz,
        mx_scale_width=args.mx_scale_width,
        mx_scale_block_size=args.mx_scale_block_size,
        fp_constant_num=args.fp_constant_num,
        weight_param_count=args.weight_param_count,
        weight_element_bits=effective_weight_bits,
        weight_precision=effective_weight_precision,
        weight_mx_exp_width=args.weight_mx_exp_width,
        weight_mx_mant_width=args.weight_mx_mant_width,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (WORKSPACE_ROOT / "runs" / timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_jsonl = run_dir / (
        f"trials.worker_{args.worker_id:03d}.jsonl" if args.worker_mode else "trials.jsonl"
    )
    cache_path = run_dir / (
        f"area_cache.worker_{args.worker_id:03d}.json" if args.worker_mode else "area_cache.json"
    )
    cache: dict[str, Any] = {}
    compiler_cost_cache: dict[str, dict[str, Any]] = {}

    model = load_json(MODEL_CONFIG)
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
    search_space = DEFAULT_SEARCH_SPACE
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
        sampler = optuna.samplers.GridSampler(optuna_search_space, seed=args.seed)
        grid_total_trials = math.prod(len(values) for values in optuna_search_space.values())
    else:
        grid_total_trials = None
        sampler = optuna.samplers.NSGAIISampler(
            seed=args.seed + (args.worker_id if args.worker_mode else 0),
            constraints_func=a100_constraints,
        )
    storage, optuna_storage_backend = create_optuna_storage(
        run_dir,
        requested_backend=args.optuna_storage,
        worker_mode=args.worker_mode,
        workers=args.workers,
    )
    study = optuna.create_study(
        directions=["minimize", "minimize", "maximize"],
        sampler=sampler,
        storage=storage,
        study_name="qwen3_32b_dense_dse",
        load_if_exists=True,
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
    initial_settled_trials = _settled_trial_count(study)
    if grid_total_trials is not None:
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
    else:
        trials_to_run = args.n_trials
        target_settled_trials = initial_finished_trials + args.n_trials

    def objective(trial: optuna.Trial) -> tuple[float, float, float]:
        trial_dir = run_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(exist_ok=True)
        record: dict[str, Any] = {"trial": trial.number, "state": "running"}
        try:
            precision_name = trial.suggest_categorical("precision_profile", [p["name"] for p in precision_profiles])
            precision = precision_by_name[precision_name]
            params = {
                "MLEN": args.fixed_mlen if args.fixed_mlen is not None else trial.suggest_categorical("MLEN", search_space["MLEN"]),
                "BLEN": args.fixed_blen if args.fixed_blen is not None else trial.suggest_categorical("BLEN", search_space["BLEN"]),
                "INT_DATA_WIDTH": (
                    args.fixed_int_data_width
                    if args.fixed_int_data_width is not None
                    else trial.suggest_categorical("INT_DATA_WIDTH", search_space["INT_DATA_WIDTH"])
                ),
            }
            params["VLEN"] = params["MLEN"]
            hw = derived_hardware(model, params, dse_config)
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
                    "model_config": str(MODEL_CONFIG),
                    "latency_model": LATENCY_MODEL_NAME,
                    "precision_profile": precision_name,
                    "warnings": bandwidth_issues,
                    "accuracy_score": float(precision.get("accuracy_score", 1.0)),
                    **hw,
                    "ON_CHIP_ADDR_WIDTH": hw["INT_DATA_WIDTH"],
                    "INT_SRAM_WIDTH": hw["INT_DATA_WIDTH"],
                    "weight_precision": precision_label(
                        profile_weight_spec(precision, dse_config), dse_config.mx_scale_width
                    ),
                    "packed_attention_schedule": args.packed_attention_schedule,
                    "vector_scalar_schedule": args.vector_scalar_schedule,
                }
            )

            batch_info = calculate_batch_info(model, precision, dse_config)
            record.update(batch_info)
            record.update(
                sequence_layout_metrics(
                    seq_len=dse_config.input_seq_len,
                    batch_size=dse_config.latency_batch_size,
                    mlen=hw["MLEN"],
                    native_layout_mode=args.native_layout_mode,
                )
            )

            legacy_would_prune = bool(bandwidth_issues)
            shadow = {
                "mode": args.compiler_cost_mode,
                "legacy_bandwidth_prune_enabled": args.legacy_bandwidth_prune,
                "legacy_would_prune": legacy_would_prune,
                "legacy_issues": bandwidth_issues,
                "v3_would_prune": False,
                "decision_disagrees": legacy_would_prune,
                "memory_status": "disabled",
                "v3_status": "disabled",
            }
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
                        compiler_cache_key = stable_key(
                            {
                                "hardware": hw,
                                "precision": precision,
                                "seq_len": dse_config.input_seq_len,
                                "batch_size": dse_config.latency_batch_size,
                                "settings": str(args.compiler_cost_settings),
                                "calibration": str(args.compiler_cost_calibration),
                                "compute_timing": args.compiler_compute_timing,
                                "scheduled_shadow": args.compiler_scheduled_shadow,
                                "v4_memory_evaluation": (
                                    args.compiler_v4_memory_evaluation
                                ),
                                "native_layout_mode": args.native_layout_mode,
                                "packed_attention_schedule": (
                                    args.packed_attention_schedule
                                ),
                                "vector_scalar_schedule": (
                                    args.vector_scalar_schedule
                                ),
                            }
                        )
                        compiler_cost_report = compiler_cost_cache.get(
                            compiler_cache_key
                        )
                        compiler_cost_cache_hit = compiler_cost_report is not None
                        if compiler_cost_report is None:
                            compiler_cost_report = run_compiler_cost(
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
                                args.vector_scalar_schedule,
                            )
                            compiler_cost_cache[compiler_cache_key] = (
                                compiler_cost_report
                            )
                        else:
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
                            write_json(
                                trial_dir / "compiler_cost_report.json",
                                compiler_cost_report,
                            )
                        memory_model_version = str(
                            compiler_cost_report.get("memory_model_version", "unknown")
                        )
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
                                "compiler_calibration_in_domain": compiler_cost_report.get(
                                    "calibration_in_domain"
                                ),
                                "compiler_compute_timing_mode": compiler_cost_report.get(
                                    "compute_timing_mode"
                                ),
                                "compiler_compute_timing_semantics": compiler_cost_report.get(
                                    "compute_timing_semantics"
                                ),
                                "compiler_compute_resource_work_cycles": compiler_cost_report.get(
                                    "compute_resource_work_cycles"
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

            if args.compiler_cost_mode in COMPILER_COST_OBJECTIVE_MODES:
                if compiler_cost_report is None:
                    raise RuntimeError("compiler cost objective completed without a cost report")
                compute_latency_ms = record["compiler_compute_latency_ms"]
                if args.compiler_cost_mode == "compute-objective":
                    latency_ms = compute_latency_ms
                    latency_source = (
                        "compiler_cost_rtl_v1_resource_work"
                        if args.compiler_compute_timing == "rtl-v1"
                        else "compiler_cost_legacy_compute"
                    )
                    objective_combination = "compute_only_with_legacy_bandwidth_guard"
                    record["latency_model"] = LATENCY_MODEL_NAME
                elif args.compiler_cost_mode == "roofline-objective":
                    latency_ms = record["compiler_roofline_latency_ms"]
                    latency_source = "compiler_cost_stage_roofline_rtl_v1_v4"
                    objective_combination = "sum_stage_max_compute_v4_memory"
                    record["latency_model"] = "compiler_stage_roofline_rtl_v1_v4"
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
                    "compiler_cost": compiler_cost_report,
                }
            else:
                analytic_toml = trial_dir / "analytic_hardware.toml"
                write_analytic_toml(analytic_toml, hw, dse_config)
                latency_ms, latency_report = run_latency(
                    MODEL_CONFIG, analytic_toml, trial_dir, batch_info, dse_config
                )
            record["latency_ms"] = latency_ms
            write_json(trial_dir / "trial_record.json", record)

            key = stable_key({"hw": hw, "precision": precision})
            if key in cache:
                area_metrics = cache[key]
            elif args.area_mode == "none":
                area_metrics = {"area": 0.0, "area_mode": "none"}
            elif args.area_mode == "proxy":
                area_metrics = run_area_proxy(hw, precision, dse_config)
            elif args.area_mode in {"proxy-v2", "proxy-v2-mxint"}:
                area_metrics = run_area_proxy_v2(hw, precision, dse_config)
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
            write_json(cache_path, cache)

            area_um2 = float(area_metrics.get("area_um2", area_metrics.get("area", 0.0)))
            area_mm2 = float(area_metrics.get("area_mm2", area_um2 / 1e6))
            area_p10_mm2 = float(area_metrics.get("area_uncertainty_p10_mm2", area_mm2))
            area_p50_mm2 = float(area_metrics.get("area_uncertainty_p50_mm2", area_mm2))
            area_p90_mm2 = float(area_metrics.get("area_uncertainty_p90_mm2", area_mm2))
            area_constraint = area_mm2 - args.area_budget_mm2
            tolerance = args.target_area_mm2 * args.target_area_tolerance_pct / 100.0
            trial.set_user_attr("area_budget_constraint_mm2", area_constraint)
            trial.set_user_attr("a100_area_constraint_mm2", area_constraint)
            trial.set_user_attr("area_mm2", area_mm2)

            record.update(
                {
                    "state": "complete",
                    "area": area_um2,
                    "area_um2": area_um2,
                    "area_mm2": area_mm2,
                    "area_uncertainty_p10_mm2": area_p10_mm2,
                    "area_uncertainty_p50_mm2": area_p50_mm2,
                    "area_uncertainty_p90_mm2": area_p90_mm2,
                    "area_budget_constraint_mm2": area_constraint,
                    "a100_area_constraint_mm2": area_constraint,
                    "within_target_area_tolerance": abs(area_mm2 - args.target_area_mm2) <= tolerance,
                    "area_mode": area_metrics.get("area_mode"),
                    "area_model": area_metrics.get("area_model"),
                    "area_breakdown": area_metrics.get(
                        "area_breakdown", area_metrics.get("area_proxy_breakdown", {})
                    ),
                    "area_metrics": area_metrics,
                    "area_extrapolation_warnings": area_metrics.get("area_extrapolation_warnings", []),
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
            for key_name in ("area_proxy_breakdown", "area_proxy_inputs", "area_new_breakdown", "area_new_inputs"):
                if key_name in area_metrics:
                    record[key_name] = area_metrics[key_name]
            write_json(trial_dir / "trial_record.json", record)
            write_json(trial_dir / "latency_report.parsed.json", latency_report)
            return record["latency_ms"], record["area_mm2"], record["accuracy_score"]
        except TrialPrunedError as exc:
            record.update({"state": "pruned", "reason": str(exc)})
            write_json(trial_dir / "trial_record.json", record)
            raise optuna.TrialPruned(str(exc)) from exc
        except KeyboardInterrupt:
            record.update({"state": "failed", "reason": "KeyboardInterrupt"})
            write_json(trial_dir / "trial_record.json", record)
            raise
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            record.update({"state": "failed", "reason": reason})
            write_json(trial_dir / "trial_record.json", record)
            raise
        finally:
            append_jsonl(trials_jsonl, record)
            records.append(record)

    worker_trial_budget = (
        args.worker_trials
        if args.worker_mode and args.worker_trials is not None
        else trials_to_run
    )
    resolved_workers = (
        min(os.cpu_count() or 1, max(1, worker_trial_budget))
        if args.workers == "auto"
        else max(1, min(int(args.workers), max(1, worker_trial_budget)))
    )
    try:
        if not args.worker_mode and resolved_workers > 1 and trials_to_run > 0:
            worker_id = next_worker_id(run_dir)
            return_codes, worker_id = launch_worker_processes(
                run_dir,
                resolved_workers,
                trials_to_run,
                worker_id,
                max_trials_per_process=args.worker_max_trials_per_process,
            )
            retry_wave = 0
            previous_settled = initial_settled_trials
            no_progress_waves = 0
            while True:
                refreshed = optuna.load_study(
                    study_name="qwen3_32b_dense_dse", storage=storage, sampler=sampler
                )
                if grid_total_trials is not None:
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
                missing = target_settled_trials - settled
                if missing <= 0:
                    break
                if settled <= previous_settled:
                    no_progress_waves += 1
                else:
                    no_progress_waves = 0
                previous_settled = settled
                if no_progress_waves >= 3:
                    raise RuntimeError(
                        "Exhaustive grid made no progress for three retry waves: "
                        f"settled={settled}, missing={missing}"
                    )

                finalize_redundant_waiting_trials(refreshed)
                queued_missing = enqueue_missing_grid_trials(
                    refreshed, optuna_search_space
                )
                retry_wave += 1
                print(
                    f"Grid retry wave {retry_wave}: settled={settled}, "
                    f"missing={missing}, exact_queued={queued_missing}"
                )
                retry_workers = min(resolved_workers, missing)
                retry_codes, worker_id = launch_worker_processes(
                    run_dir,
                    retry_workers,
                    missing,
                    worker_id,
                    max_trials_per_process=args.worker_max_trials_per_process,
                )
                return_codes.extend(retry_codes)
            study = optuna.load_study(
                study_name="qwen3_32b_dense_dse", storage=storage, sampler=sampler
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
                )
            else:
                study.optimize(
                    objective,
                    n_trials=trial_count,
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
            study_name="qwen3_32b_dense_dse", storage=storage, sampler=sampler
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
    write_best_csv(run_dir / "best_trials.csv", completed_records)
    pareto_numbers = {trial.number for trial in study.best_trials}
    pareto_records = [record for record in completed_records if int(record["trial"]) in pareto_numbers]
    write_records_csv(run_dir / "pareto_trials.csv", pareto_records)

    selections = select_area_reference_candidates(
        completed_records,
        target_area_mm2=args.target_area_mm2,
        area_budget_mm2=args.area_budget_mm2,
        target_area_tolerance_pct=args.target_area_tolerance_pct,
    )
    feasible = selections["feasible"]
    fastest = selections["fastest"]
    fidelity_qualified = selections["fidelity_qualified"]
    fastest_fidelity_qualified = selections["fastest_fidelity_qualified"]
    highest_accuracy = selections["highest_accuracy"]
    closest_to_target = selections["closest_to_target"]
    closest_below_target = selections["closest_below_target"]
    within_tolerance = selections["within_tolerance"]
    p90_feasible = selections["p90_feasible"]
    p90_fastest = selections["p90_fastest"]
    p90_closest_to_target = selections["p90_closest_to_target"]
    write_json(
        run_dir / "a100_comparison.json",
        {
            "target_area_mm2": args.target_area_mm2,
            "area_budget_mm2": args.area_budget_mm2,
            "target_area_tolerance_pct": args.target_area_tolerance_pct,
            "reference": "NVIDIA A100 826 mm2 die-area reference with a 110% feasibility budget",
            "ga100_reference_area_mm2": GA100_REFERENCE_AREA_MM2,
            "note": (
                "PLENA area is a calibrated logic plus SRAM-macro proxy and excludes physical "
                "HBM stacks/package. Candidate fidelity must be checked separately: large "
                "MLEN/BLEN points and unsupported RTL-v1 opcodes are exploratory."
            ),
            "feasible_trial_count": len(feasible),
            "fidelity_qualified_trial_count": len(fidelity_qualified),
            "fastest_under_area_budget": fastest,
            "fastest_fidelity_qualified_under_area_budget": fastest_fidelity_qualified,
            "highest_accuracy_under_area_budget": highest_accuracy,
            "closest_area_to_target_mm2": closest_to_target,
            "closest_area_below_target_mm2": closest_below_target,
            "within_target_area_tolerance": within_tolerance,
            "p90_conservative_feasible_trial_count": len(p90_feasible),
            "p90_conservative_fastest_under_area_budget": p90_fastest,
            "p90_conservative_closest_area_to_target_mm2": p90_closest_to_target,
        },
    )
    write_json(
        run_dir / "run_summary.json",
        {
            "run_dir": str(run_dir),
            "model_config": str(MODEL_CONFIG),
            "latency_model": LATENCY_MODEL_NAME,
            "input_seq_len": dse_config.input_seq_len,
            "output_seq_len": dse_config.output_seq_len,
            "device_num": dse_config.device_num,
            "n_trials": grid_total_trials or args.n_trials,
            "resume_initial_settled_trials": initial_settled_trials,
            "resume_trials_requested": trials_to_run,
            "interrupted_trial_reconciliation": reconciliation,
            "finalized_redundant_waiting_trials": finalized_waiting_trials,
            "workers": (
                resolved_workers
                if trials_to_run > 0 or args.worker_mode
                else (os.cpu_count() or 1)
                if args.workers == "auto"
                else int(args.workers)
            ),
            "workers_requested": args.workers,
            "worker_max_trials_per_process": args.worker_max_trials_per_process,
            "optuna_storage_backend": optuna_storage_backend,
            "serialized_optuna_ask": resolved_workers > 1,
            "sampler": args.sampler,
            "accuracy_constraints": str(args.accuracy_constraints),
            "precision_profile_count": len(precision_profiles),
            "area_mode": args.area_mode,
            "target_area_mm2": args.target_area_mm2,
            "area_budget_mm2": args.area_budget_mm2,
            "target_area_tolerance_pct": args.target_area_tolerance_pct,
            "strict_bandwidth": args.legacy_bandwidth_prune,
            "legacy_bandwidth_prune": args.legacy_bandwidth_prune,
            "compiler_cost_mode": args.compiler_cost_mode,
            "compiler_compute_timing": args.compiler_compute_timing,
            "compiler_scheduled_shadow": args.compiler_scheduled_shadow,
            "compiler_v4_memory_evaluation": args.compiler_v4_memory_evaluation,
            "packed_attention_schedule": args.packed_attention_schedule,
            "vector_scalar_schedule": args.vector_scalar_schedule,
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
    print(f"Wrote DSE run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
