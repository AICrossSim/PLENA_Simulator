#!/usr/bin/env python3
"""Reproduce attributable A/B experiments for compact-statistics RTL-v4.

The script evaluates the eight short-context combinations requested by the
validation plan and optional long-context baseline/combined pairs.  All counts
come from the compiler-generated CostTrace; the benchmark never subtracts
cycles with an analytical post-processing formula.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import time
from pathlib import Path
from typing import Any

from analytic_models.performance.compiler_cost_model import (
    compile_and_evaluate_compiler_cost,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "Workspace/qwen3_32b_dense_analytic/qwen3-32b.json"
DEFAULT_SETTINGS = (
    ROOT
    / "Workspace/qwen3_32b_dense_analytic/runs/"
    "smoke_streamed_kmajor_agu_v2_fixed_20260725/trial_0001/"
    "compiler_cost_settings.toml"
)
DEFAULT_HBM = ROOT / "analytic_models/performance/calibration/hbm_dma_service_v4.json"
DEFAULT_TIMING = ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v4.json"
DEFAULT_OUTPUT_DIR = ROOT / "Workspace/reports/compiler"

ARMS = {
    "baseline": ("rtl-v3", "legacy", "accumulate-v1"),
    "compact_only": ("rtl-v4", "legacy", "accumulate-v1"),
    "hoist_only": ("rtl-v3", "hoisted-v1", "accumulate-v1"),
    "overwrite_only": ("rtl-v3", "legacy", "overwrite-v1"),
    "compact_hoist": ("rtl-v4", "hoisted-v1", "accumulate-v1"),
    "compact_overwrite": ("rtl-v4", "legacy", "overwrite-v1"),
    "hoist_overwrite": ("rtl-v3", "hoisted-v1", "overwrite-v1"),
    "combined": ("rtl-v4", "hoisted-v1", "overwrite-v1"),
}

INVARIANT_MATRIX_OPCODES = (
    "M_MM",
    "M_MM_WO",
    "M_BTMM",
    "M_BMM_WO",
    "M_MV",
    "M_MV_WO",
)


def _settings_with_matrix_tiles(
    source: Path,
    destination: Path,
    *,
    mlen: int,
    tiles: int,
) -> Path:
    """Write a settings variant with the requested Matrix SRAM depth."""
    text = source.read_text()
    pattern = re.compile(
        r"(\[TRANSACTIONAL\.CONFIG\.MATRIX_SRAM_SIZE\]\s*\n"
        r"\s*value\s*=\s*)\d+"
    )
    updated, count = pattern.subn(rf"\g<1>{mlen * tiles}", text, count=1)
    if count != 1:
        raise ValueError(
            "could not find exactly one TRANSACTIONAL MATRIX_SRAM_SIZE setting"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(updated)
    return destination


def _run_case(
    *,
    name: str,
    model: Path,
    settings: Path,
    hbm: Path,
    timing: Path,
    seq_len: int,
    batch_size: int,
    vector_scalar_schedule: str,
    selector_schedule: str,
    reduction_output_mode: str,
    cache_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    trace, report = compile_and_evaluate_compiler_cost(
        model,
        settings,
        hbm,
        seq_len=seq_len,
        batch_size=batch_size,
        num_layers=64,
        packed_attention_schedule="direct-first-block-v1",
        softmax_state_schedule="streamed-v2",
        packed_qk_schedule="broadcast-k-major-v1",
        vector_scalar_schedule=vector_scalar_schedule,
        selector_schedule=selector_schedule,
        reduction_output_mode=reduction_output_mode,
        gqa_pipeline_schedule="row-interleaved-v1",
        address_generation_mode="loop-agu-v1",
        compute_timing_mode="ideal-ii1",
        rtl_timing_calibration=timing,
        persistent_trace_cache_dir=cache_dir / "trace",
        persistent_v4_work_cache_dir=cache_dir / "v4",
    )
    elapsed = time.perf_counter() - started
    trace_dict = trace.to_dict()
    report_dict = report.to_dict()
    return {
        "name": name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "modes": {
            "vector_scalar_schedule": vector_scalar_schedule,
            "selector_schedule": selector_schedule,
            "reduction_output_mode": reduction_output_mode,
        },
        "wall_time_sec": elapsed,
        "one_layer_compute_cycles": int(
            report_dict["one_layer_compute_resource_work_cycles"]
        ),
        "full_decoder_compute_cycles": int(
            report_dict["compute_resource_work_cycles"]
        ),
        "one_layer_category_cycles": {
            key: int(round(value))
            for key, value in report_dict["one_layer_category_latency_ns"].items()
        },
        "one_layer_stage_roofline_ns": report_dict["one_layer_latency_ns"],
        "full_decoder_stage_roofline_ns": report_dict["roofline_latency_ns"],
        "one_layer_opcodes": {
            key: int(value)
            for key, value in trace_dict["one_layer_dynamic_opcodes"].items()
        },
        "hbm": {
            key: report_dict[key]
            for key in (
                "one_layer_hbm_read_bytes",
                "one_layer_hbm_write_bytes",
                "one_layer_hbm_read_requests",
                "one_layer_hbm_write_requests",
            )
        },
        "optimization_metadata": trace_dict.get(
            "vector_scalar_optimization", {}
        ),
        "schedule_metadata": {
            key: trace_dict.get(key)
            for key in (
                "selector_schedule",
                "reduction_output_mode",
                "vector_scalar_schedule",
                "broadcast_rtl_validation_status",
            )
        },
    }


def _opcode_delta(
    baseline: dict[str, Any], case: dict[str, Any]
) -> list[dict[str, Any]]:
    before = baseline["one_layer_opcodes"]
    after = case["one_layer_opcodes"]
    return [
        {
            "arm": case["name"],
            "opcode": opcode,
            "before": before.get(opcode, 0),
            "after": after.get(opcode, 0),
            "delta": after.get(opcode, 0) - before.get(opcode, 0),
        }
        for opcode in sorted(set(before) | set(after))
        if before.get(opcode, 0) != after.get(opcode, 0)
    ]


def _summarize_short(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {case["name"]: case for case in cases}
    baseline = by_name["baseline"]
    baseline_cycles = baseline["one_layer_compute_cycles"]
    arm_summary = {}
    for name, case in by_name.items():
        saved = baseline_cycles - case["one_layer_compute_cycles"]
        arm_summary[name] = {
            "one_layer_compute_cycles": case["one_layer_compute_cycles"],
            "cycles_saved": saved,
            "reduction_pct": 100.0 * saved / baseline_cycles,
        }
    combined = by_name["combined"]
    matrix_invariant = {
        opcode: (
            baseline["one_layer_opcodes"].get(opcode, 0)
            == combined["one_layer_opcodes"].get(opcode, 0)
        )
        for opcode in INVARIANT_MATRIX_OPCODES
    }
    return {
        "arms": arm_summary,
        "acceptance": {
            "combined_compute_le_19_60m": (
                combined["one_layer_compute_cycles"] <= 19_600_000
            ),
            "combined_reduction_ge_22pct": (
                arm_summary["combined"]["reduction_pct"] >= 22.0
            ),
            "matrix_opcodes_unchanged": all(matrix_invariant.values()),
            "matrix_opcode_checks": matrix_invariant,
            "hbm_traffic_unchanged": baseline["hbm"] == combined["hbm"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--hbm-calibration", type=Path, default=DEFAULT_HBM)
    parser.add_argument("--rtl-timing", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-short", action="store_true")
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".compact_stats_v1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "compact_stats_selector_overwrite_v1_results.json"
    delta_path = output_dir / "compact_stats_selector_overwrite_v1_opcode_delta.csv"

    if args.skip_short and results_path.exists():
        payload = json.loads(results_path.read_text())
        payload["long_context"] = []
    else:
        payload = {
            "schema_version": "compact_stats_selector_overwrite_v1",
            "workload": {
                "model": str(args.model.resolve()),
                "mlen": 2048,
                "vlen": 2048,
                "blen": 1024,
                "num_layers": 64,
            },
            "short_context": [],
            "long_context": [],
        }
    opcode_rows: list[dict[str, Any]] = []

    if not args.skip_short:
        short_cases = []
        for name, modes in ARMS.items():
            case = _run_case(
                name=name,
                model=args.model,
                settings=args.settings,
                hbm=args.hbm_calibration,
                timing=args.rtl_timing,
                seq_len=482,
                batch_size=16,
                vector_scalar_schedule=modes[0],
                selector_schedule=modes[1],
                reduction_output_mode=modes[2],
                cache_dir=cache_dir,
            )
            short_cases.append(case)
            gc.collect()
        baseline = next(case for case in short_cases if case["name"] == "baseline")
        for case in short_cases:
            opcode_rows.extend(_opcode_delta(baseline, case))
        payload["short_context"] = short_cases
        payload["short_summary"] = _summarize_short(short_cases)

    if args.long_context:
        for tiles in (2, 8):
            settings = _settings_with_matrix_tiles(
                args.settings,
                cache_dir / f"settings_matrix_tiles_{tiles}.toml",
                mlen=2048,
                tiles=tiles,
            )
            for seq_len in (4096, 4097, 8192):
                for name in ("baseline", "combined"):
                    modes = ARMS[name]
                    case = _run_case(
                        name=f"{name}_s{seq_len}_t{tiles}",
                        model=args.model,
                        settings=settings,
                        hbm=args.hbm_calibration,
                        timing=args.rtl_timing,
                        seq_len=seq_len,
                        batch_size=1,
                        vector_scalar_schedule=modes[0],
                        selector_schedule=modes[1],
                        reduction_output_mode=modes[2],
                        cache_dir=cache_dir,
                    )
                    case["matrix_sram_tiles"] = tiles
                    payload["long_context"].append(case)
                    gc.collect()

    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if opcode_rows or not delta_path.exists():
        with delta_path.open("w", newline="") as handle:
            fields = ("arm", "opcode", "before", "after", "delta")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(opcode_rows)
    print(json.dumps(payload.get("short_summary", {}), indent=2, sort_keys=True))
    print(f"wrote {results_path}")
    print(f"wrote {delta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
