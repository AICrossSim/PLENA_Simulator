#!/usr/bin/env python3
"""Reproduce flattened MatrixMachine shape sweeps at fixed PE/SRAM budgets."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from analytic_models.area_new import estimate_area
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
DEFAULT_OUTPUT_DIR = ROOT / "Workspace/reports/compiler/flatten_shape_sweep_v1"

SHAPE_GROUPS = {
    262_144: ((512, 512), (1024, 256), (2048, 128)),
    524_288: ((1024, 512), (2048, 256)),
    1_048_576: ((1024, 1024), (2048, 512)),
}
WORKLOADS = (
    ("short_b16", 482, 16),
    ("long_4096", 4096, 1),
    ("long_4097", 4097, 1),
    ("long_8192", 8192, 1),
)
BASE_MATRIX_SRAM_ELEMENTS = 2048 * 4096
FP_CONSTANT_NUM = 10


def _replace_toml_value(text: str, section: str, value: int) -> str:
    pattern = re.compile(
        rf"(\[{re.escape(section)}\]\s*\n\s*value\s*=\s*)\d+"
    )
    updated, count = pattern.subn(rf"\g<1>{value}", text, count=1)
    if count != 1:
        raise ValueError(f"could not update {section}")
    return updated


def _shape_settings(source: Path, destination: Path, mlen: int, blen: int) -> dict[str, int]:
    model = json.loads(DEFAULT_MODEL.read_text())
    physical_broadcast = min(
        model["num_attention_heads"] // model["num_key_value_heads"],
        mlen // model["head_dim"],
    )
    hardware_broadcast = min(16, mlen // model["head_dim"])
    matrix_sram_depth = BASE_MATRIX_SRAM_ELEMENTS // mlen
    vector_sram_depth = 2 * model["head_dim"] + math.ceil(
        model["hidden_size"] / mlen
    )
    fp_sram_depth = FP_CONSTANT_NUM + 2 * mlen * physical_broadcast
    values = {
        "TRANSACTIONAL.CONFIG.MLEN": mlen,
        "TRANSACTIONAL.CONFIG.VLEN": mlen,
        "TRANSACTIONAL.CONFIG.BLEN": blen,
        "TRANSACTIONAL.CONFIG.BROADCAST_AMOUNT": hardware_broadcast,
        "TRANSACTIONAL.CONFIG.MATRIX_SRAM_SIZE": matrix_sram_depth,
        "TRANSACTIONAL.CONFIG.VECTOR_SRAM_SIZE": vector_sram_depth,
        "TRANSACTIONAL.CONFIG.FP_SRAM_DEPTH": fp_sram_depth,
        "TRANSACTIONAL.CONFIG.HBM_M_Prefetch_Amount": mlen,
        "TRANSACTIONAL.CONFIG.HBM_V_Prefetch_Amount": blen,
        "TRANSACTIONAL.CONFIG.HBM_V_Writeback_Amount": blen,
    }
    text = source.read_text()
    for section, value in values.items():
        text = _replace_toml_value(text, section, value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return {
        "matrix_sram_depth": matrix_sram_depth,
        "matrix_sram_elements": mlen * matrix_sram_depth,
        "matrix_sram_tiles": matrix_sram_depth // mlen,
        "vector_sram_depth": vector_sram_depth,
        "fp_sram_depth": fp_sram_depth,
        "physical_broadcast": physical_broadcast,
        "hardware_broadcast": hardware_broadcast,
    }


def _area_config(
    *,
    mlen: int,
    blen: int,
    memory: dict[str, int],
) -> dict[str, Any]:
    return {
        "MLEN": mlen,
        "VLEN": mlen,
        "BLEN": blen,
        "MATRIX_SRAM_DEPTH": memory["matrix_sram_depth"],
        "VECTOR_SRAM_DEPTH": memory["vector_sram_depth"],
        "INT_SRAM_DEPTH": 32,
        "FP_SRAM_DEPTH": memory["fp_sram_depth"],
        "INT_DATA_WIDTH": 32,
        "ACT_WIDTH": "MXFP_E5M2",
        "KV_WIDTH": "MXFP_E5M2",
        "WEIGHT_WIDTH": "MXFP_E1M2",
        "FP_SETTING": "FP_E5M6",
        "MX_SCALE_WIDTH": 8,
        "BLOCK_DIM": blen,
        "HBM_ELE_WIDTH": mlen,
        "HBM_SCALE_WIDTH": (mlen // blen) * 8,
        "HBM_M_Prefetch_Amount": mlen,
        "HBM_V_Prefetch_Amount": blen,
        "HBM_V_Writeback_Amount": blen,
        "vector_scalar_area_version": "rtl-v4",
        "address_generation_mode": "loop-agu-v1",
    }


def _evaluate_case(job: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    trace, report = compile_and_evaluate_compiler_cost(
        job["model"],
        job["settings"],
        job["hbm"],
        seq_len=job["seq_len"],
        batch_size=job["batch_size"],
        num_layers=64,
        packed_attention_schedule="direct-first-block-v1",
        softmax_state_schedule="streamed-v2",
        packed_qk_schedule="broadcast-k-major-v1",
        vector_scalar_schedule="rtl-v4",
        selector_schedule="hoisted-v1",
        reduction_output_mode="overwrite-v1",
        gqa_pipeline_schedule="row-interleaved-v1",
        address_generation_mode="loop-agu-v1",
        compute_timing_mode="ideal-ii1",
        rtl_timing_calibration=job["timing"],
        persistent_trace_cache_dir=job["cache_dir"] / "trace",
        persistent_v4_work_cache_dir=job["cache_dir"] / "v4",
    )
    trace_data = trace.to_dict()
    report_data = report.to_dict()
    opcodes = {
        key: int(value)
        for key, value in trace_data["one_layer_dynamic_opcodes"].items()
    }
    packed = trace_data.get("packed_attention", {})
    native_layout = trace_data.get("native_layout", {})
    return {
        "pe_budget": job["pe_budget"],
        "mlen": job["mlen"],
        "vlen": job["mlen"],
        "blen": job["blen"],
        "flatten_ratio": job["mlen"] / job["blen"],
        "workload": job["workload"],
        "seq_len": job["seq_len"],
        "batch_size": job["batch_size"],
        "wall_time_sec": time.perf_counter() - started,
        **job["memory"],
        "one_layer_compute_cycles": int(
            report_data["one_layer_compute_resource_work_cycles"]
        ),
        "one_layer_roofline_ns": float(report_data["one_layer_latency_ns"]),
        "full_decoder_roofline_ns": float(report_data["roofline_latency_ns"]),
        "one_layer_category_cycles": {
            key: int(round(value))
            for key, value in report_data["one_layer_category_latency_ns"].items()
        },
        "stage_compute_ns": report_data["stage_compute_latency_ns"],
        "stage_hbm_ns": report_data["one_layer_hbm_stage_latency_ns"],
        "stage_roofline_ns": report_data["stage_roofline_latency_ns"],
        "stage_bound": report_data["stage_bound"],
        "hbm_read_bytes": int(report_data["one_layer_hbm_read_bytes"]),
        "hbm_write_bytes": int(report_data["one_layer_hbm_write_bytes"]),
        "hbm_read_requests": int(report_data["one_layer_hbm_read_requests"]),
        "hbm_write_requests": int(report_data["one_layer_hbm_write_requests"]),
        "matrix_opcodes": {
            key: value for key, value in opcodes.items() if key.startswith("M_")
        },
        "vector_opcode_count": sum(
            value for key, value in opcodes.items() if key.startswith("V_")
        ),
        "scalar_opcode_count": sum(
            value for key, value in opcodes.items() if key.startswith("S_")
        ),
        "control_opcode_count": sum(
            value for key, value in opcodes.items() if key.startswith("C_")
        ),
        "packed_attention": packed,
        "native_layout": native_layout,
        "broadcast_rtl_validation_status": trace_data.get(
            "broadcast_rtl_validation_status"
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "pe_budget",
        "workload",
        "seq_len",
        "batch_size",
        "mlen",
        "blen",
        "flatten_ratio",
        "matrix_sram_tiles",
        "matrix_sram_depth",
        "matrix_sram_elements",
        "area_mm2",
        "matrix_area_mm2",
        "sram_area_mm2",
        "one_layer_compute_cycles",
        "one_layer_roofline_ms",
        "full_decoder_roofline_s",
        "matrix_cycles",
        "vector_cycles",
        "scalar_cycles",
        "control_cycles",
        "memory_cycles",
        "hbm_read_bytes",
        "hbm_write_bytes",
        "wall_time_sec",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            categories = row["one_layer_category_cycles"]
            writer.writerow(
                {
                    key: row.get(key)
                    for key in fields
                    if key
                    not in {
                        "matrix_cycles",
                        "vector_cycles",
                        "scalar_cycles",
                        "control_cycles",
                        "memory_cycles",
                    }
                }
                | {
                    "matrix_cycles": categories["matrix_compute"],
                    "vector_cycles": categories["vector_compute"],
                    "scalar_cycles": categories["scalar_compute"],
                    "control_cycles": categories["control"],
                    "memory_cycles": categories["memory"],
                }
            )


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    def point(workload: str, pe_budget: int, mlen: int, blen: int) -> dict[str, Any]:
        return next(
            row
            for row in rows
            if row["workload"] == workload
            and row["pe_budget"] == pe_budget
            and row["mlen"] == mlen
            and row["blen"] == blen
        )

    short_square = point("short_b16", 1_048_576, 1024, 1024)
    short_flat = point("short_b16", 1_048_576, 2048, 512)
    long_square = point("long_8192", 1_048_576, 1024, 1024)
    long_flat = point("long_8192", 1_048_576, 2048, 512)
    boundary_square = point("long_4096", 1_048_576, 1024, 1024)
    boundary_flat = point("long_4096", 1_048_576, 2048, 512)
    tail_square = point("long_4097", 1_048_576, 1024, 1024)
    tail_flat = point("long_4097", 1_048_576, 2048, 512)
    moderate_square = point("long_8192", 262_144, 512, 512)
    moderate_flat = point("long_8192", 262_144, 1024, 256)

    def faster(before: dict[str, Any], after: dict[str, Any]) -> float:
        return 100.0 * (
            before["one_layer_roofline_ns"] - after["one_layer_roofline_ns"]
        ) / before["one_layer_roofline_ns"]

    def growth(before: float, after: float) -> float:
        return 100.0 * (after - before) / before

    def direction(before: dict[str, Any], after: dict[str, Any]) -> str:
        improvement = faster(before, after)
        return (
            f"{improvement:.2f}% faster"
            if improvement >= 0.0
            else f"{abs(improvement):.2f}% slower"
        )

    long_categories_square = long_square["one_layer_category_cycles"]
    long_categories_flat = long_flat["one_layer_category_cycles"]
    short_area_latency_ratio = (
        short_flat["area_mm2"] * short_flat["one_layer_roofline_ns"]
    ) / (short_square["area_mm2"] * short_square["one_layer_roofline_ns"])
    long_area_latency_ratio = (
        long_flat["area_mm2"] * long_flat["one_layer_roofline_ns"]
    ) / (long_square["area_mm2"] * long_square["one_layer_roofline_ns"])

    lines = [
        "# Flattened MatrixMachine Shape Sweep v1",
        "",
        "All points use the latest combined compiler path (`rtl-v4`, selector",
        "hoisting, reduction overwrite, K-major broadcast, loop-AGU-v1) and",
        "ideal-II1 compute timing. Matrix SRAM capacity is held at 8,388,608",
        "FP entries (`MLEN * depth`) in every shape.",
        "",
        "> The large Matrix shapes are structural timing/area extrapolations,",
        "> and broadcast Matrix RTL remains unvalidated.",
        "",
        "## Main Findings",
        "",
        f"- At 1,048,576 PEs, `2048/512` is {direction(short_square, short_flat)} "
        "than `1024/1024` for `seq=482, batch=16`, while using "
        f"{growth(short_square['area_mm2'], short_flat['area_mm2']):.2f}% more area.",
        f"- At `seq=8192`, the same flattened shape is "
        f"{faster(long_square, long_flat):.2f}% faster. Its area-latency product "
        f"is {(1.0 - long_area_latency_ratio) * 100.0:.2f}% lower, whereas the "
        f"short-context product is {(short_area_latency_ratio - 1.0) * 100.0:.2f}% higher.",
        "- The long-context gain is not a direct MatrixMachine throughput gain. "
        f"Matrix work changes from {long_categories_square['matrix_compute'] / 1e6:.3f}M "
        f"to {long_categories_flat['matrix_compute'] / 1e6:.3f}M cycles, while "
        f"Vector falls from {long_categories_square['vector_compute'] / 1e6:.3f}M "
        f"to {long_categories_flat['vector_compute'] / 1e6:.3f}M and Scalar from "
        f"{long_categories_square['scalar_compute'] / 1e6:.3f}M to "
        f"{long_categories_flat['scalar_compute'] / 1e6:.3f}M.",
        f"- A moderate flattening from `512/512` to `1024/256` at 262,144 PEs "
        f"is {faster(moderate_square, moderate_flat):.2f}% faster at `seq=8192`; "
        "the more extreme `2048/128` point gives back part of that gain.",
        f"- Tail behavior is decisive. `2048/512` is "
        f"{direction(boundary_square, boundary_flat)} at `seq=4096`, "
        f"but is effectively tied ({direction(tail_square, tail_flat)}) at `seq=4097`. "
        "The single-row tail invokes full-width BMM because active-row BMM is unavailable.",
        f"- At `seq=8192`, changing `1024/1024` to `2048/512` increases physical "
        f"HBM reads by {growth(long_square['hbm_read_bytes'], long_flat['hbm_read_bytes']):.2f}% "
        f"and writes by {growth(long_square['hbm_write_bytes'], long_flat['hbm_write_bytes']):.2f}%. "
        "The evaluated points remain compute-bound, but this traffic matters for energy.",
        "",
        "The current implementation therefore has a real long-context shape",
        "benefit, but it should be described as a whole-machine tiling benefit",
        "caused largely by wider `MLEN=VLEN`, not as proof that the flattened",
        "systolic datapath alone has higher effective throughput.",
        "",
    ]
    for workload, _, _ in WORKLOADS:
        lines.extend([f"## {workload}", ""])
        for pe_budget, shapes in SHAPE_GROUPS.items():
            group = [
                row
                for row in rows
                if row["workload"] == workload and row["pe_budget"] == pe_budget
            ]
            group.sort(key=lambda row: row["flatten_ratio"])
            baseline = group[0]
            lines.extend(
                [
                    f"### PE budget {pe_budget:,}",
                    "",
                    "| MLEN/BLEN | Ratio | Area mm2 | Compute Mcy | Roofline ms/layer | Matrix Mcy | Vector Mcy | vs least-flat |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in group:
                reduction = (
                    100.0
                    * (
                        baseline["one_layer_roofline_ns"]
                        - row["one_layer_roofline_ns"]
                    )
                    / baseline["one_layer_roofline_ns"]
                )
                categories = row["one_layer_category_cycles"]
                lines.append(
                    "| "
                    f"{row['mlen']}/{row['blen']} | "
                    f"{row['flatten_ratio']:.1f} | "
                    f"{row['area_mm2']:.3f} | "
                    f"{row['one_layer_compute_cycles'] / 1e6:.3f} | "
                    f"{row['one_layer_roofline_ns'] / 1e6:.3f} | "
                    f"{categories['matrix_compute'] / 1e6:.3f} | "
                    f"{categories['vector_compute'] / 1e6:.3f} | "
                    f"{reduction:+.2f}% |"
                )
            lines.append("")
    lines.extend(
        [
            "## Claim Boundary",
            "",
            "- Equal PE count does not imply equal modeled area; the report gives",
            "  both values and fixes Matrix SRAM bit capacity instead of tile count.",
            "- `ideal-II1` makes every Vector/Scalar/Control instruction one cycle.",
            "  The result is a DSE architectural estimate, not cycle-exact RTL.",
            "- Matrix timing is measured only at small shapes and structurally",
            "  extrapolated here. Broadcast BMM remains RTL-unvalidated.",
            "- `MLEN=VLEN` couples Matrix shape to Vector tiling. The reported",
            "  end-to-end gain cannot be attributed to MatrixMachine alone.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--hbm", type=Path, default=DEFAULT_HBM)
    parser.add_argument("--timing", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=len(SHAPE_GROUPS) * 2 + 1)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_dir = output_dir / "settings"
    cache_root = output_dir / "cache"
    shape_metadata: dict[tuple[int, int], dict[str, int]] = {}
    settings_paths: dict[tuple[int, int], Path] = {}
    for shapes in SHAPE_GROUPS.values():
        for mlen, blen in shapes:
            key = (mlen, blen)
            if key in settings_paths:
                continue
            destination = settings_dir / f"m{mlen}_b{blen}.toml"
            shape_metadata[key] = _shape_settings(
                args.settings.resolve(), destination, mlen, blen
            )
            settings_paths[key] = destination

    jobs = []
    for pe_budget, shapes in SHAPE_GROUPS.items():
        for mlen, blen in shapes:
            for workload, seq_len, batch_size in WORKLOADS:
                jobs.append(
                    {
                        "pe_budget": pe_budget,
                        "mlen": mlen,
                        "blen": blen,
                        "workload": workload,
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "memory": shape_metadata[(mlen, blen)],
                        "model": args.model.resolve(),
                        "settings": settings_paths[(mlen, blen)],
                        "hbm": args.hbm.resolve(),
                        "timing": args.timing.resolve(),
                        "cache_dir": cache_root
                        / f"m{mlen}_b{blen}_{workload}",
                    }
                )

    rows = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(args.workers, len(settings_paths))
    ) as executor:
        futures = {executor.submit(_evaluate_case, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            row = future.result()
            area = estimate_area(
                _area_config(
                    mlen=job["mlen"],
                    blen=job["blen"],
                    memory=job["memory"],
                )
            )
            breakdown = area["area_breakdown"]
            row.update(
                {
                    "area_mm2": area["area"] / 1e6,
                    "area_p90_mm2": area["area_uncertainty_p90"] / 1e6,
                    "matrix_area_mm2": breakdown["MatrixMachine"] / 1e6,
                    "sram_area_mm2": area["sram_macro_area"] / 1e6,
                    "area_warnings": area["area_extrapolation_warnings"],
                    "one_layer_roofline_ms": row["one_layer_roofline_ns"] / 1e6,
                    "full_decoder_roofline_s": row["full_decoder_roofline_ns"]
                    / 1e9,
                }
            )
            rows.append(row)
            print(
                f"{row['workload']} {row['mlen']}/{row['blen']}: "
                f"{row['one_layer_roofline_ms']:.3f} ms/layer"
            )

    rows.sort(
        key=lambda row: (
            row["workload"],
            row["pe_budget"],
            row["flatten_ratio"],
        )
    )
    payload = {
        "schema_version": "flatten_shape_sweep_v1",
        "fixed_matrix_sram_elements": BASE_MATRIX_SRAM_ELEMENTS,
        "timing_semantics": "ideal-ii1",
        "compiler_modes": {
            "vector_scalar_schedule": "rtl-v4",
            "selector_schedule": "hoisted-v1",
            "reduction_output_mode": "overwrite-v1",
            "packed_qk_schedule": "broadcast-k-major-v1",
            "address_generation_mode": "loop-agu-v1",
        },
        "rows": rows,
    }
    (output_dir / "flatten_shape_sweep_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(output_dir / "flatten_shape_sweep_v1.csv", rows)
    _write_report(output_dir / "flatten_shape_sweep_v1.md", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
