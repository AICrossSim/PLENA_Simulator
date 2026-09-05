#!/usr/bin/env python3
"""Evaluate Matrix-SRAM context organizations on every saved SWE decode window.

This is a capacity and concurrency analysis, not a cycle model.  It answers
how many expert streams can have resident/pending panels under the existing
behavioral Matrix-SRAM payload without claiming that all streams execute in
parallel or that panel depth directly equals speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from transactional_emulator.testbench.moe_timing.replay.analyze_full_swe_grouping import (
    MODEL_SPECS,
    iter_decode_blocks,
    sha256_file,
    validate_archive,
)


MLEN = 64
MATRIX_SRAM_DEPTH_SETTING = 4096
MATRIX_SRAM_ELEMENT_BYTES = 2  # BF16 in plena_settings.toml.
TOTAL_CELLS = MATRIX_SRAM_DEPTH_SETTING // MLEN
CELL_ELEMENTS = MLEN * MLEN
CELL_BYTES = CELL_ELEMENTS * MATRIX_SRAM_ELEMENT_BYTES
TOTAL_BYTES = TOTAL_CELLS * CELL_BYTES
PANEL_CELLS = 4  # Compiler default mram_tile_capacity.
PANEL_MRAM_BYTES = PANEL_CELLS * CELL_BYTES
PANEL_HBM_BYTES = PANEL_CELLS * 4608  # 64x64 MXFP8+scale physical bytes/tile.


@dataclass(frozen=True)
class Organization:
    key: str
    display_name: str
    max_contexts: int
    fixed_depth: int | None
    note: str


ORGANIZATIONS = (
    Organization(
        "blocking_1ctx_d1",
        "Current compiler: one blocking 4-tile panel",
        1,
        1,
        "Capacity baseline; no ping-pong overlap.",
    ),
    Organization(
        "pingpong_1ctx_d2",
        "One expert context with ping-pong panels",
        1,
        2,
        "Separates load and compute for one stream only.",
    ),
    Organization(
        "static_4ctx_d2",
        "Four statically partitioned expert contexts",
        4,
        2,
        "Uses 32 of 64 cells; fixed ownership can strand capacity.",
    ),
    Organization(
        "static_8ctx_d2",
        "Eight statically partitioned expert contexts",
        8,
        2,
        "Uses all 64 cells as eight independent ping-pong streams.",
    ),
    Organization(
        "elastic_tagged_1to8ctx",
        "Elastic tagged pool with one to eight active contexts",
        8,
        None,
        "Allocates contexts per window and gives spare cells to deeper prefetch queues.",
    ),
)


def organization_for_jobs(org: Organization, jobs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return contexts, panel depth, and scheduling waves for each window."""
    if np.any(jobs <= 0):
        raise ValueError("each active window must contain at least one expert job")
    contexts = np.minimum(jobs, org.max_contexts)
    if org.fixed_depth is None:
        depth = TOTAL_CELLS // (PANEL_CELLS * contexts)
        depth = np.maximum(depth, 2)
    else:
        depth = np.full_like(jobs, org.fixed_depth)
    required_cells = contexts * depth * PANEL_CELLS
    if np.any(required_cells > TOTAL_CELLS):
        raise AssertionError(f"{org.key} exceeds Matrix-SRAM capacity")
    waves = np.ceil(jobs / contexts).astype(np.int64)
    return contexts, depth, waves


def iter_active_job_blocks(
    indices: np.ndarray,
    valid: np.ndarray,
    *,
    nominal_batch: int,
    num_experts: int,
) -> Iterable[np.ndarray]:
    """Yield logical expert-job counts for all active layer-step windows.

    Routed experts contribute one job each after grouping.  The existing
    Shared-MoE compiler presents the combined shared projection as one logical
    stream per window, so one shared job is reserved here.
    """
    for routes, active in iter_decode_blocks(
        indices,
        valid,
        nominal_batch=nominal_batch,
        num_experts=num_experts,
    ):
        keep = active > 0
        routes = routes[keep]
        present = np.zeros((routes.shape[0], num_experts), dtype=bool)
        valid_route = routes != num_experts
        window_ids = np.broadcast_to(np.arange(routes.shape[0])[:, None], routes.shape)[valid_route]
        present[window_ids, routes[valid_route].astype(np.int64)] = True
        routed_jobs = present.sum(axis=1, dtype=np.int64)
        yield routed_jobs + 1


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q, method="higher"))


def summarize_archive(path: Path, model_key: str, batches: list[int]) -> list[dict[str, Any]]:
    spec = MODEL_SPECS[model_key]
    rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as archive:
        validate_archive(archive, spec)
        indices = archive["decode_idx"].astype(np.int64, copy=False)
        valid = archive["valid"].astype(bool, copy=False)
        for batch in batches:
            job_parts = list(
                iter_active_job_blocks(
                    indices,
                    valid,
                    nominal_batch=batch,
                    num_experts=spec.num_experts,
                )
            )
            jobs = np.concatenate(job_parts)
            for org in ORGANIZATIONS:
                contexts, depth, waves = organization_for_jobs(org, jobs)
                rows.append(
                    {
                        "model": spec.display_name,
                        "workload": "SWE-bench test prompts",
                        "truth_scope": "all_samples_all_saved_16_decode_tokens_all_moe_layers",
                        "nominal_batch": batch,
                        "organization_id": org.key,
                        "organization": org.display_name,
                        "windows": int(jobs.size),
                        "mean_expert_jobs_including_shared": float(jobs.mean()),
                        "p50_expert_jobs": percentile(jobs, 50),
                        "p95_expert_jobs": percentile(jobs, 95),
                        "p99_expert_jobs": percentile(jobs, 99),
                        "max_expert_jobs": int(jobs.max()),
                        "mean_resident_contexts": float(contexts.mean()),
                        "min_panel_depth": int(depth.min()),
                        "mean_panel_depth": float(depth.mean()),
                        "max_panel_depth": int(depth.max()),
                        "one_wave_window_pct": float(100.0 * np.mean(waves == 1)),
                        "mean_scheduling_waves": float(waves.mean()),
                        "p95_scheduling_waves": percentile(waves, 95),
                        "max_scheduling_waves": int(waves.max()),
                        "fixed_cells_reserved": (
                            org.max_contexts * org.fixed_depth * PANEL_CELLS
                            if org.fixed_depth is not None
                            else "dynamic<=64"
                        ),
                        "note": org.note,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], provenance: dict[str, Any]) -> None:
    selected = [row for row in rows if row["organization_id"] in {"blocking_1ctx_d1", "static_8ctx_d2", "elastic_tagged_1to8ctx"}]
    lines = [
        "# SWE Decode Matrix-SRAM Context Capacity",
        "",
        "## Scope",
        "",
        "This is an exact capacity/concurrency analysis over every saved decode route window. "
        "The archives cover all 2,294 SWE-bench test prompts, all 16 saved decode tokens, and all MoE layers. "
        "They are not complete generations or agent trajectories. Scheduling waves are a residency-pressure metric, not cycle speedup.",
        "",
        "## Fixed Hardware Contract",
        "",
        f"- Matrix-SRAM behavioral capacity: {TOTAL_CELLS} cells x {CELL_BYTES} B = {TOTAL_BYTES} B ({TOTAL_BYTES / 1024:.0f} KiB), stored as BF16.",
        f"- Compiler panel: {PANEL_CELLS} cells = {PANEL_MRAM_BYTES} B in Matrix SRAM; the same four MXFP8 HBM tiles transfer {PANEL_HBM_BYTES} physical bytes.",
        "- One ping-pong expert context therefore consumes 8 cells (64 KiB), allowing at most eight simultaneous contexts at unchanged payload capacity.",
        "- One logical shared projection stream is included in every window.",
        "",
        "## Capacity Results",
        "",
        "| Model | Batch | Organization | Mean jobs | One-wave windows | Mean waves | p95 waves |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['model']} | {row['nominal_batch']} | {row['organization']} | "
            f"{row['mean_expert_jobs_including_shared']:.2f} | {row['one_wave_window_pct']:.2f}% | "
            f"{row['mean_scheduling_waves']:.2f} | {row['p95_scheduling_waves']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Expert grouping is what removes duplicate weight transfers. Matrix-SRAM organization does not reduce bytes by itself.",
            "- Ping-pong separates load and compute for one stream. Multiple tagged contexts additionally keep several experts ready so the DMA and scheduler are not globally blocked by one expert.",
            "- A context being resident does not imply an additional matrix core. Actual cycle benefit requires an asynchronous DMA path, bank/port arbitration, and a compute scheduler; those are separate replay/RTL validation gates.",
            "- The elastic organization is the architectural candidate: it preserves the 512 KiB payload and changes ownership metadata and banking rather than adding capacity.",
            "",
            "## Provenance",
            "",
            "```json",
            json.dumps(provenance, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--deepseek", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[2, 4, 8, 16])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {"qwen": args.qwen.resolve(), "deepseek": args.deepseek.resolve()}
    rows: list[dict[str, Any]] = []
    for model_key, path in inputs.items():
        rows.extend(summarize_archive(path, model_key, args.batches))
    provenance = {
        "schema_version": 1,
        "analysis_kind": "capacity_and_context_pressure_not_cycle_timing",
        "inputs": {
            key: {"path": str(path), "sha256": sha256_file(path)} for key, path in inputs.items()
        },
        "hardware": {
            "mlen": MLEN,
            "matrix_sram_depth_setting": MATRIX_SRAM_DEPTH_SETTING,
            "matrix_sram_storage": "BF16",
            "matrix_sram_cells": TOTAL_CELLS,
            "matrix_sram_payload_bytes": TOTAL_BYTES,
            "compiler_panel_cells": PANEL_CELLS,
            "compiler_panel_mram_bytes": PANEL_MRAM_BYTES,
            "compiler_panel_hbm_physical_bytes": PANEL_HBM_BYTES,
        },
    }
    write_csv(args.output_dir / "sram_context_capacity.csv", rows)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "SRAM_CONTEXT_CAPACITY_REPORT.md", rows, provenance)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
