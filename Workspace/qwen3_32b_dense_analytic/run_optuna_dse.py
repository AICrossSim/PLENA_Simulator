#!/usr/bin/env python3
"""Optuna DSE for Qwen3-32B dense analytic prefill + PLENA RTL area.

Workspace-local by design.  The script treats accuracy as an external input
through precision profiles, evaluates latency with the LLaMA-style dense
analytic prefill model, and optionally runs PLENA_RTL area-mode synthesis for
area/power.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import optuna
import toml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
COMPILER_COST_OBJECTIVE_MODES = {"compute-objective", "objective"}

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
    }


def constraint_issues(
    model: dict[str, Any],
    hw: dict[str, int],
    precision: dict[str, Any],
    strict_bandwidth: bool,
    config: DSEConfig,
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
    return warnings, ratios


def run_area_proxy_v2(hw: dict[str, int], precision: dict[str, Any], config: DSEConfig) -> dict[str, Any]:
    from analytic_models.area_new import estimate_area

    proxy_inputs = build_area_proxy_inputs(hw, precision, config)
    metrics = estimate_area(proxy_inputs)
    warnings, ratios = area_extrapolation_warnings(hw)
    metrics.update(
        {
            "area_mode": "proxy-v2",
            "area_um2": float(metrics["area"]),
            "area_mm2": float(metrics["area"]) / 1e6,
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
) -> None:
    data = toml.load(template)
    try:
        config = data["TRANSACTIONAL"]["CONFIG"]
    except KeyError as exc:
        raise ValueError(f"{template} has no TRANSACTIONAL.CONFIG section") from exc
    physical_broadcast = min(hw["BROADCAST_AMOUNT"], hw["MLEN"] // hw["HLEN"])
    values = {
        "MLEN": hw["MLEN"],
        "BLEN": hw["BLEN"],
        "VLEN": hw["VLEN"],
        "HLEN": hw["HLEN"],
        "BROADCAST_AMOUNT": physical_broadcast,
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
    )
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
    )
    result = report.to_dict()
    result["trace"] = {
        "schema_version": trace.schema_version,
        "static_machine_instructions": sum(trace.static_opcodes.values()),
        "dynamic_machine_instructions": sum(trace.dynamic_opcodes.values()),
        "memory_stream_count": len(trace.memory_events),
        "dma_coverage": trace.metadata.get("dma_coverage"),
    }
    write_json(trial_dir / "compiler_cost_report.json", result)
    return result


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


def read_trial_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.glob("trial_*/trial_record.json")):
        try:
            records.append(load_json(path))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda record: int(record.get("trial", -1)))


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "trial", "state", "reason", "latency_ms", "latency_source",
        "compiler_compute_latency_ms", "compiler_memory_latency_ms",
        "compiler_serial_latency_ms", "compiler_memory_model_version",
        "compiler_memory_evaluation_mode", "compiler_cost_cache_hit",
        "v3_memory_latency_ms", "v3_serial_latency_ms",
        "area_um2", "area_mm2", "area_budget_constraint_mm2", "a100_area_constraint_mm2",
        "within_target_area_tolerance",
        "accuracy_score", "precision_profile", "weight_precision", "MLEN", "VLEN", "BLEN",
        "HLEN", "BROADCAST_AMOUNT", "INT_DATA_WIDTH", "MATRIX_SRAM_SIZE", "VECTOR_SRAM_SIZE",
        "INT_SRAM_DEPTH", "FP_SRAM_DEPTH", "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount", "HBM_V_Writeback_Amount", "calibration_in_domain",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _strip_worker_cli(argv: list[str]) -> list[str]:
    takes_value = {"--workers", "--run-dir", "--worker-id", "--worker-trials"}
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


def launch_worker_processes(run_dir: Path, workers: int, total_trials: int, start_worker_id: int = 0) -> list[int]:
    base_args = _strip_worker_cli(sys.argv[1:])
    quotas = [total_trials // workers + (1 if index < total_trials % workers else 0) for index in range(workers)]
    processes: list[tuple[subprocess.Popen[str], Any]] = []
    for offset, quota in enumerate(quotas):
        if quota <= 0:
            continue
        worker_id = start_worker_id + offset
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
        processes.append((subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, text=True), log_handle))
    return_codes = []
    for process, log_handle in processes:
        return_codes.append(process.wait())
        log_handle.close()
    return return_codes


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
        "--strict-bandwidth",
        "--legacy-bandwidth-prune",
        dest="legacy_bandwidth_prune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preserve the historical fixed-bandwidth prune decision (default: true)",
    )
    parser.add_argument(
        "--compiler-cost-mode",
        choices=("off", "shadow", "compute-objective", "objective"),
        default="compute-objective",
        help=(
            "Record the selected memory model beside the legacy model, use CostEmitter "
            "compute with the legacy bandwidth guard, or use serial compiler cost as "
            "the objective"
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
        choices=("auto", "full-global-stateful", "one-layer-stateful-scaled"),
        default="one-layer-stateful-scaled",
        help=(
            "V4 memory-shadow fidelity. DSE defaults to one exact stateful "
            "decoder layer with layer-stage scaling; full global state is "
            "reserved for validation because it is minutes per trial."
        ),
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
    else:
        sampler = optuna.samplers.NSGAIISampler(
            seed=args.seed + (args.worker_id if args.worker_mode else 0),
            constraints_func=a100_constraints,
        )
    try:
        from optuna.storages.journal import JournalFileBackend

        journal_backend = JournalFileBackend(str(run_dir / "study.journal"))
    except ImportError:  # pragma: no cover - compatibility with older Optuna
        journal_backend = optuna.storages.JournalFileStorage(str(run_dir / "study.journal"))
    storage = optuna.storages.JournalStorage(journal_backend)
    study = optuna.create_study(
        directions=["minimize", "minimize", "maximize"],
        sampler=sampler,
        storage=storage,
        study_name="qwen3_32b_dense_dse",
        load_if_exists=True,
    )
    initial_finished_trials = sum(
        trial.state.is_finished() for trial in study.get_trials(deepcopy=False)
    )

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
                model, hw, precision, False, dse_config
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
                }
            )

            batch_info = calculate_batch_info(model, precision, dse_config)
            record.update(batch_info)

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

    resolved_workers = (
        min(os.cpu_count() or 1, args.n_trials)
        if args.workers == "auto"
        else max(1, min(int(args.workers), args.n_trials))
    )
    try:
        if not args.worker_mode and resolved_workers > 1:
            return_codes = launch_worker_processes(run_dir, resolved_workers, args.n_trials)
            target_finished = initial_finished_trials + args.n_trials
            next_worker_id = resolved_workers
            for _ in range(3):
                refreshed = optuna.load_study(
                    study_name="qwen3_32b_dense_dse", storage=storage, sampler=sampler
                )
                finished = sum(
                    trial.state.is_finished() for trial in refreshed.get_trials(deepcopy=False)
                )
                missing = target_finished - finished
                if missing <= 0:
                    break
                return_codes.extend(launch_worker_processes(run_dir, 1, missing, next_worker_id))
                next_worker_id += 1
            study = optuna.load_study(
                study_name="qwen3_32b_dense_dse", storage=storage, sampler=sampler
            )
            if any(code != 0 for code in return_codes):
                print(f"Warning: worker return codes: {return_codes}", file=sys.stderr)
        else:
            trial_count = args.worker_trials if args.worker_mode else args.n_trials
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

    records = read_trial_records(run_dir)
    completed_records = [record for record in records if record.get("state") == "complete"]
    with trials_jsonl.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    write_records_csv(run_dir / "all_trials.csv", records)
    write_best_csv(run_dir / "best_trials.csv", completed_records)
    pareto_numbers = {trial.number for trial in study.best_trials}
    pareto_records = [record for record in completed_records if int(record["trial"]) in pareto_numbers]
    write_records_csv(run_dir / "pareto_trials.csv", pareto_records)

    feasible = [record for record in completed_records if float(record["area_mm2"]) <= args.area_budget_mm2]
    fastest = min(feasible, key=lambda record: (record["latency_ms"], -record["accuracy_score"])) if feasible else None
    highest_accuracy = max(feasible, key=lambda record: (record["accuracy_score"], -record["latency_ms"])) if feasible else None
    closest_to_target = (
        min(feasible, key=lambda record: abs(float(record["area_mm2"]) - args.target_area_mm2))
        if feasible
        else None
    )
    below_target = [record for record in feasible if float(record["area_mm2"]) <= args.target_area_mm2]
    closest_below_target = max(below_target, key=lambda record: float(record["area_mm2"])) if below_target else None
    tolerance_mm2 = args.target_area_mm2 * args.target_area_tolerance_pct / 100.0
    within_tolerance = [
        record
        for record in feasible
        if abs(float(record["area_mm2"]) - args.target_area_mm2) <= tolerance_mm2
    ]
    write_json(
        run_dir / "a100_comparison.json",
        {
            "target_area_mm2": args.target_area_mm2,
            "area_budget_mm2": args.area_budget_mm2,
            "target_area_tolerance_pct": args.target_area_tolerance_pct,
            "reference": "NVIDIA GA100 826 mm2 die-area reference with a 110% feasibility budget",
            "ga100_reference_area_mm2": GA100_REFERENCE_AREA_MM2,
            "note": (
                "PLENA area is a calibrated logic plus SRAM-macro proxy and excludes physical "
                "HBM stacks/package; large MLEN/BLEN points extrapolate beyond DC calibration."
            ),
            "feasible_trial_count": len(feasible),
            "fastest_under_area_budget": fastest,
            "highest_accuracy_under_area_budget": highest_accuracy,
            "closest_area_to_target_mm2": closest_to_target,
            "closest_area_below_target_mm2": closest_below_target,
            "within_target_area_tolerance": within_tolerance,
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
            "n_trials": args.n_trials,
            "workers": resolved_workers,
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
            "completed": sum(1 for r in records if r.get("state") == "complete"),
            "pruned": sum(1 for r in records if r.get("state") == "pruned"),
            "failed": sum(1 for r in records if r.get("state") == "failed"),
        },
    )
    print(f"Wrote DSE run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
