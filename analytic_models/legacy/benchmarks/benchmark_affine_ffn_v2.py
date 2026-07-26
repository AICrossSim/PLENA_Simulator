#!/usr/bin/env python3
"""Reproduce the historical affine FFN lowering transition."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

from analytic_models.performance.compiler_cost_model import (
    compile_and_evaluate_compiler_cost,
)
from Workspace.qwen3_32b_dense_analytic.run_optuna_dse import (
    DEFAULT_ACCURACY_PATH,
    DEFAULT_COMPILER_COST_CALIBRATION,
    DEFAULT_COMPILER_COST_SETTINGS,
    DSEConfig,
    load_accuracy,
    profile_weight_spec,
    write_compiler_cost_toml,
)


def _config(record: dict[str, Any]) -> DSEConfig:
    return DSEConfig(
        input_seq_len=int(record["input_seq_len"]),
        output_seq_len=int(record.get("output_seq_len", 1)),
        device_num=int(record.get("device_num", 1)),
        latency_batch_size=int(record["latency_batch_size"]),
        hbm_capacity_bytes=int(record["hbm_capacity_bytes"]),
        hbm_bandwidth_gbps=float(record["hbm_bandwidth_gbps"]),
        frequency_ghz=1.0,
        mx_scale_width=int(record["mx_scale_width"]),
        mx_scale_block_size=int(record["mx_scale_block_size"]),
        fp_constant_num=int(record["FP_CONSTANT_NUM"]),
        weight_param_count=float(record["model_param_count"]),
        weight_element_bits=float(record["weight_effective_bits"]),
        weight_precision=str(record["weight_precision"]),
        weight_mx_exp_width=0,
        weight_mx_mant_width=0,
        softmax_state_schedule=str(record["softmax_state_schedule"]),
        packed_qk_schedule=str(record["packed_qk_schedule"]),
    )


def _hardware(record: dict[str, Any]) -> dict[str, int]:
    keys = (
        "MLEN",
        "BLEN",
        "VLEN",
        "HLEN",
        "BROADCAST_AMOUNT",
        "MATRIX_SRAM_SIZE",
        "VECTOR_SRAM_SIZE",
        "FP_SRAM_DEPTH",
        "FP_CONSTANT_NUM",
        "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount",
        "HBM_V_Writeback_Amount",
        "INT_DATA_WIDTH",
    )
    return {key: int(record[key]) for key in keys}


def _manifest_hash(trace: Any) -> str:
    payload = json.dumps(
        [event.to_dict() for event in trace.memory_events],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _evaluate(
    *,
    record: dict[str, Any],
    config: DSEConfig,
    profile: dict[str, Any],
    schedule: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="plena_ffn_affine_v2_") as directory:
        settings = Path(directory) / "settings.toml"
        write_compiler_cost_toml(
            DEFAULT_COMPILER_COST_SETTINGS,
            settings,
            _hardware(record),
            profile,
            config,
            "compact",
        )
        trace, report = compile_and_evaluate_compiler_cost(
            record["model_config"],
            settings,
            DEFAULT_COMPILER_COST_CALIBRATION,
            seq_len=config.input_seq_len,
            batch_size=config.latency_batch_size,
            num_layers=64,
            precision_config={
                "weight": profile_weight_spec(profile, config),
                "activation": profile["ACT_WIDTH"],
                "kv": profile["KV_WIDTH"],
                "block": config.mx_scale_block_size,
                "scale_bits": config.mx_scale_width,
                "integer_bits": record["INT_DATA_WIDTH"],
                "internal_fp": profile["FP_SETTING"],
            },
            compute_timing_mode="ideal-ii1",
            native_layout_mode="compact",
            packed_attention_schedule=str(record["packed_attention_schedule"]),
            softmax_state_schedule=str(record["softmax_state_schedule"]),
            packed_qk_schedule=str(record["packed_qk_schedule"]),
            vector_scalar_schedule=str(record["vector_scalar_schedule"]),
            selector_schedule=str(record["selector_schedule"]),
            reduction_output_mode=str(record["reduction_output_mode"]),
            gqa_pipeline_schedule=str(record["gqa_pipeline_schedule"]),
            address_generation_mode="loop-agu-v1",
            ffn_address_schedule="live-stride-v1",
            ffn_projection_schedule=schedule,
            cost_trace_granularity="affine-block-summary-v1",
            v4_memory_evaluation="one-layer-cached-occurrence-scaled",
            use_trace_cache=False,
            use_v4_work_cache=False,
            kv_residency_policy=str(record["matrix_sram_policy"]),
        )
    report_dict = report.to_dict()
    ffn = trace.stages["layer/ffn"].dynamic_opcodes
    return {
        "schedule": schedule,
        "compute_cycles": report_dict["compute_resource_work_cycles"],
        "one_layer_compute_cycles": report_dict[
            "one_layer_compute_resource_work_cycles"
        ],
        "stage_compute_ns": report_dict["stage_compute_latency_ns"],
        "stage_memory_ns": report_dict.get(
            "stage_memory_latency_ns",
            report_dict.get("hbm_stage_latency_ns", {}),
        ),
        "ffn_dynamic_opcodes": dict(sorted(ffn.items())),
        "memory_manifest_hash": _manifest_hash(trace),
        "memory_occurrences": sum(
            event.multiplicity for event in trace.memory_events
        ),
        "ffn_address_optimization": trace.metadata[
            "ffn_address_optimization"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_record", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    record = json.loads(args.trial_record.read_text())
    config = _config(record)
    profiles = load_accuracy(DEFAULT_ACCURACY_PATH)
    profile = next(
        item for item in profiles if item["name"] == record["precision_profile"]
    )
    results = [
        _evaluate(
            record=record,
            config=config,
            profile=profile,
            schedule=schedule,
        )
        for schedule in ("legacy-auto-v1", "affine-loop-v2")
    ]
    legacy, affine = results
    comparison = {
        "trial_record": str(args.trial_record),
        "hardware": {
            key: record[key]
            for key in ("MLEN", "BLEN", "VLEN", "MATRIX_SRAM_TILES")
        },
        "chip_count": record["chip_count"],
        "results": results,
        "matrix_work_equal": all(
            affine["ffn_dynamic_opcodes"].get(opcode, 0)
            == legacy["ffn_dynamic_opcodes"].get(opcode, 0)
            for opcode in ("M_MM", "M_MM_WO", "V_ADD_VV", "H_PREFETCH_M")
        ),
        "memory_manifest_equal": (
            affine["memory_manifest_hash"] == legacy["memory_manifest_hash"]
        ),
        "compute_reduction_pct": 100.0
        * (legacy["compute_cycles"] - affine["compute_cycles"])
        / legacy["compute_cycles"],
        "ffn_s_addi_reduction_pct": 100.0
        * (
            legacy["ffn_dynamic_opcodes"].get("S_ADDI_INT", 0)
            - affine["ffn_dynamic_opcodes"].get("S_ADDI_INT", 0)
        )
        / max(1, legacy["ffn_dynamic_opcodes"].get("S_ADDI_INT", 0)),
    }
    payload = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not comparison["matrix_work_equal"]:
        return 1
    if not comparison["memory_manifest_equal"]:
        return 1
    if affine["compute_cycles"] > legacy["compute_cycles"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
