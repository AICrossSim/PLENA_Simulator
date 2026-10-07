#!/usr/bin/env python3
"""Audit expert grouping over every saved SWE route window.

The route archives used here cover every SWE-bench test instance and every
saved decode token/layer, but only the first 16 decode tokens per instance.
This script keeps that boundary explicit. It computes exact population
statistics and physical weight-byte counts without emitting a multi-million
row intermediate file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


BLEN = 4
MX_PHYSICAL_BYTES_NUMERATOR = 9
MX_PHYSICAL_BYTES_DENOMINATOR = 8


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    hidden: int
    routed_intermediate: int
    shared_intermediate: int
    shared_experts: int
    top_k: int
    num_experts: int
    expected_layers: int

    @property
    def routed_weight_bytes(self) -> int:
        return mx_weight_bytes(3 * self.hidden * self.routed_intermediate)

    @property
    def shared_weight_bytes(self) -> int:
        # shared_intermediate already represents the combined shared width.
        return mx_weight_bytes(3 * self.hidden * self.shared_intermediate)


MODEL_SPECS = {
    "qwen": ModelSpec(
        key="qwen",
        display_name="Qwen3.5-35B-A3B-FP8",
        hidden=2048,
        routed_intermediate=512,
        shared_intermediate=512,
        shared_experts=1,
        top_k=8,
        num_experts=256,
        expected_layers=40,
    ),
    "deepseek": ModelSpec(
        key="deepseek",
        display_name="DeepSeek-V2-Lite-Chat",
        hidden=2048,
        routed_intermediate=1408,
        shared_intermediate=2816,
        shared_experts=2,
        top_k=6,
        num_experts=64,
        expected_layers=26,
    ),
}


def mx_weight_bytes(elements: int) -> int:
    """Return 64B-exact bytes for the compiler's 1.125 B/parameter MX layout."""
    numerator = elements * MX_PHYSICAL_BYTES_NUMERATOR
    if numerator % MX_PHYSICAL_BYTES_DENOMINATOR:
        raise ValueError(f"MX element count {elements} is not byte aligned")
    raw = numerator // MX_PHYSICAL_BYTES_DENOMINATOR
    return math.ceil(raw / 64) * 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Aggregate:
    windows: int = 0
    active_batch_sum: int = 0
    route_pairs: int = 0
    active_expert_loads: int = 0
    grouped_issued_rows: int = 0
    group_size_histogram: dict[int, int] = field(default_factory=dict)

    def update(self, counts: np.ndarray, active_batches: np.ndarray) -> None:
        if counts.ndim != 2:
            raise ValueError(f"counts must be [windows, experts], got {counts.shape}")
        if counts.shape[0] != active_batches.shape[0]:
            raise ValueError("counts and active batch arrays disagree")
        keep = active_batches > 0
        counts = counts[keep]
        active_batches = active_batches[keep]
        if not counts.size:
            return
        positive = counts[counts > 0]
        histogram = np.bincount(positive.astype(np.int64))
        self.windows += int(counts.shape[0])
        self.active_batch_sum += int(active_batches.sum())
        self.route_pairs += int(counts.sum())
        self.active_expert_loads += int(positive.size)
        self.grouped_issued_rows += int((np.ceil(positive / BLEN) * BLEN).sum())
        for size, frequency in enumerate(histogram):
            if size and frequency:
                self.group_size_histogram[size] = self.group_size_histogram.get(size, 0) + int(frequency)


def counts_from_routes(routes: np.ndarray, num_experts: int) -> np.ndarray:
    """Vectorized bincount for a bounded block of [window, routed_pair] ids."""
    windows = routes.shape[0]
    sentinel = num_experts
    valid = routes != sentinel
    window_ids = np.broadcast_to(np.arange(windows)[:, None], routes.shape)[valid]
    codes = window_ids * num_experts + routes[valid].astype(np.int64)
    return np.bincount(codes, minlength=windows * num_experts).reshape(windows, num_experts)


def iter_decode_blocks(
    indices: np.ndarray,
    valid: np.ndarray,
    *,
    nominal_batch: int,
    num_experts: int,
    cohort_chunk: int = 32,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    samples, steps, layers, top_k = indices.shape
    cohorts = math.ceil(samples / nominal_batch)
    sentinel = num_experts
    for cohort_begin in range(0, cohorts, cohort_chunk):
        cohort_end = min(cohorts, cohort_begin + cohort_chunk)
        cohort_count = cohort_end - cohort_begin
        packed = np.full(
            (cohort_count, nominal_batch, steps, layers, top_k),
            sentinel,
            dtype=np.int64,
        )
        packed_valid = np.zeros((cohort_count, nominal_batch, steps), dtype=bool)
        for local_cohort, cohort in enumerate(range(cohort_begin, cohort_end)):
            start = cohort * nominal_batch
            stop = min(samples, start + nominal_batch)
            width = stop - start
            packed[local_cohort, :width] = indices[start:stop]
            packed_valid[local_cohort, :width] = valid[start:stop]
        packed = np.where(packed_valid[..., None, None], packed, sentinel)
        routes = packed.transpose(0, 2, 3, 1, 4).reshape(-1, nominal_batch * top_k)
        active = packed_valid.sum(axis=1)
        active = np.repeat(active[:, :, None], layers, axis=2).reshape(-1)
        yield routes, active


def summarize_decode(
    indices: np.ndarray,
    valid: np.ndarray,
    *,
    spec: ModelSpec,
    nominal_batch: int,
) -> tuple[Aggregate, list[Aggregate]]:
    aggregate = Aggregate()
    per_layer = [Aggregate() for _ in range(indices.shape[2])]
    layers = indices.shape[2]
    for routes, active in iter_decode_blocks(
        indices,
        valid,
        nominal_batch=nominal_batch,
        num_experts=spec.num_experts,
    ):
        counts = counts_from_routes(routes, spec.num_experts)
        aggregate.update(counts, active)
        counts_by_layer = counts.reshape(-1, layers, spec.num_experts)
        active_by_layer = active.reshape(-1, layers)
        for layer in range(layers):
            per_layer[layer].update(counts_by_layer[:, layer], active_by_layer[:, layer])
    return aggregate, per_layer


def summarize_prefill_aggregate(
    prefill_counts: np.ndarray,
    *,
    nominal_batch: int,
) -> tuple[Aggregate, list[Aggregate]]:
    samples, layers, _experts = prefill_counts.shape
    overall = Aggregate()
    per_layer = [Aggregate() for _ in range(layers)]
    for start in range(0, samples, nominal_batch):
        stop = min(samples, start + nominal_batch)
        cohort = prefill_counts[start:stop].sum(axis=0)
        active = np.full(layers, stop - start, dtype=np.int64)
        overall.update(cohort, active)
        for layer in range(layers):
            per_layer[layer].update(cohort[layer : layer + 1], active[layer : layer + 1])
    return overall, per_layer


def pct(numerator: int | float, denominator: int | float) -> float:
    return 100.0 * float(numerator) / float(denominator) if denominator else 0.0


def aggregate_row(
    aggregate: Aggregate,
    *,
    spec: ModelSpec,
    phase: str,
    nominal_batch: int,
    layer: str | int,
    truth_scope: str,
) -> dict[str, Any]:
    histogram = aggregate.group_size_histogram
    pair_bytes = aggregate.route_pairs * spec.routed_weight_bytes + aggregate.windows * spec.shared_weight_bytes
    grouped_bytes = aggregate.active_expert_loads * spec.routed_weight_bytes + aggregate.windows * spec.shared_weight_bytes
    row = {
        "model": spec.display_name,
        "workload": "SWE-bench test prompts",
        "truth_scope": truth_scope,
        "phase": phase,
        "nominal_batch": nominal_batch,
        "layer": layer,
        "layer_step_windows": aggregate.windows,
        "mean_active_batch": aggregate.active_batch_sum / aggregate.windows if aggregate.windows else 0.0,
        "route_pairs": aggregate.route_pairs,
        "active_expert_weight_loads": aggregate.active_expert_loads,
        "mean_active_experts_per_window": (
            aggregate.active_expert_loads / aggregate.windows if aggregate.windows else 0.0
        ),
        **{
            f"m{size}_expert_load_pct": pct(histogram.get(size, 0), aggregate.active_expert_loads)
            for size in range(1, 8)
        },
        "m1_to_m4_expert_load_pct": pct(
            sum(histogram.get(size, 0) for size in range(1, 5)), aggregate.active_expert_loads
        ),
        "m8plus_expert_load_pct": pct(
            sum(frequency for size, frequency in histogram.items() if size >= 8),
            aggregate.active_expert_loads,
        ),
        "pair_major_row_occupancy_pct": pct(aggregate.route_pairs, aggregate.route_pairs * BLEN),
        "expert_major_row_occupancy_pct": pct(aggregate.route_pairs, aggregate.grouped_issued_rows),
        "pair_major_physical_weight_bytes": pair_bytes,
        "expert_major_physical_weight_bytes": grouped_bytes,
        "physical_weight_byte_reduction_pct": pct(pair_bytes - grouped_bytes, pair_bytes),
        "weight_traffic_ratio_pair_over_group": pair_bytes / grouped_bytes if grouped_bytes else 0.0,
        "routed_weight_bytes_per_expert": spec.routed_weight_bytes,
        "shared_weight_bytes_per_window": spec.shared_weight_bytes,
    }
    for size in range(1, 8):
        row[f"m{size}_expert_loads"] = histogram.get(size, 0)
    row["m8plus_expert_loads"] = sum(frequency for size, frequency in histogram.items() if size >= 8)
    return row


def validate_archive(archive: Any, spec: ModelSpec) -> None:
    required = {"decode_idx", "decode_weight", "valid", "prefill_counts", "sample_ids", "layer_ids", "meta"}
    missing = required.difference(archive.files)
    if missing:
        raise ValueError(f"{spec.key}: archive is missing {sorted(missing)}")
    indices = archive["decode_idx"]
    weights = archive["decode_weight"]
    prefill = archive["prefill_counts"]
    if indices.shape != weights.shape:
        raise ValueError(f"{spec.key}: route ids and weights disagree")
    if indices.shape[2:] != (spec.expected_layers, spec.top_k):
        raise ValueError(f"{spec.key}: unexpected decode shape {indices.shape}")
    if prefill.shape[1:] != (spec.expected_layers, spec.num_experts):
        raise ValueError(f"{spec.key}: unexpected prefill shape {prefill.shape}")
    if np.any(indices < 0) or np.any(indices >= spec.num_experts):
        raise ValueError(f"{spec.key}: route id outside [0, {spec.num_experts})")
    if not bool(np.all(np.isfinite(weights))):
        raise ValueError(f"{spec.key}: route weights contain non-finite values")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_model(
    path: Path,
    spec: ModelSpec,
    batches: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    prefill_rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as archive:
        validate_archive(archive, spec)
        indices = archive["decode_idx"].astype(np.int64, copy=False)
        valid = archive["valid"].astype(bool, copy=False)
        prefill = archive["prefill_counts"].astype(np.int64, copy=False)
        layer_ids = archive["layer_ids"].astype(np.int64).tolist()
        sample_ids = archive["sample_ids"].astype(str)
        metadata = json.loads(str(archive["meta"].item()))
        for batch in batches:
            overall, by_layer = summarize_decode(indices, valid, spec=spec, nominal_batch=batch)
            summary_rows.append(
                aggregate_row(
                    overall,
                    spec=spec,
                    phase="decode",
                    nominal_batch=batch,
                    layer="all",
                    truth_scope="all_samples_all_saved_decode_tokens_all_moe_layers",
                )
            )
            for layer_index, layer_aggregate in enumerate(by_layer):
                layer_rows.append(
                    aggregate_row(
                        layer_aggregate,
                        spec=spec,
                        phase="decode",
                        nominal_batch=batch,
                        layer=layer_ids[layer_index],
                        truth_scope="all_samples_all_saved_decode_tokens_one_moe_layer",
                    )
                )
            prefill_overall, _prefill_by_layer = summarize_prefill_aggregate(prefill, nominal_batch=batch)
            prefill_rows.append(
                aggregate_row(
                    prefill_overall,
                    spec=spec,
                    phase="prefill_aggregate_only",
                    nominal_batch=batch,
                    layer="all",
                    truth_scope="per_request_prefill_counts_summed_by_cohort_without_temporal_chunk_order",
                )
            )
        provenance = {
            "model": spec.display_name,
            "archive": str(path.resolve()),
            "archive_sha256": sha256_file(path),
            "samples": int(indices.shape[0]),
            "saved_decode_steps": int(indices.shape[1]),
            "moe_layers": int(indices.shape[2]),
            "top_k": int(indices.shape[3]),
            "valid_decode_sample_steps": int(valid.sum()),
            "planned_decode_sample_steps": int(valid.size),
            "decode_coverage_pct": pct(int(valid.sum()), int(valid.size)),
            "first_sample_id": str(sample_ids[0]),
            "last_sample_id": str(sample_ids[-1]),
            "source_metadata": metadata,
        }
    return summary_rows, layer_rows, prefill_rows, provenance


def write_report(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
) -> None:
    lines = [
        "# SWE-bench Shared-MoE 同专家分组全人口审计",
        "",
        "## 数据边界",
        "",
        "- 统计覆盖归档中的全部 2,294 个 SWE-bench test prompts、全部保存的 16 个 decode token、全部 MoE 层。",
        "- 这不是完整生成至 EOS，也不是真实 agent trajectory；归档只保存 decode16。",
        "- B2/B4/B8/B16 是按归档顺序组成的静态 cohort，不冒充 request arrival/retirement 的完整 continuous batching。",
        "- Prefill 只有每请求每层的聚合计数，没有 token 时间顺序，因此只报告聚合特征，不据此声称周期加速。",
        "",
        "## Decode 精确统计",
        "",
        "| 模型 | Batch | M=1 | M=2 | M=3 | M=4 | M=5..7 | M>=8 | M=1..4 | 分组后行占用 | 权重物理字节减少 | 仅权重流量比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['nominal_batch']} | {row['m1_expert_load_pct']:.2f}% | "
            f"{row['m2_expert_load_pct']:.2f}% | {row['m3_expert_load_pct']:.2f}% | "
            f"{row['m4_expert_load_pct']:.2f}% | "
            f"{sum(row[f'm{size}_expert_load_pct'] for size in range(5, 8)):.2f}% | "
            f"{row['m8plus_expert_load_pct']:.2f}% | {row['m1_to_m4_expert_load_pct']:.2f}% | "
            f"{row['expert_major_row_occupancy_pct']:.2f}% | "
            f"{row['physical_weight_byte_reduction_pct']:.2f}% | "
            f"{row['weight_traffic_ratio_pair_over_group']:.3f}x |"
        )
    lines += [
        "",
        "这里的“物理字节减少”严格由同一 layer-step 内 `route pairs -> unique experts` 计算，并包含每个窗口不变的 shared-expert 权重。它是 HBM 权重流量真值，不等于端到端周期加速。周期必须由 Compiler + Rust/Ramulator 重放另行测量。",
        "",
        "## Provenance",
        "",
    ]
    for item in provenance:
        lines.append(
            f"- {item['model']}: {item['samples']} samples, {item['saved_decode_steps']} saved decode steps, "
            f"{item['moe_layers']} layers, coverage {item['decode_coverage_pct']:.2f}%, SHA256 `{item['archive_sha256']}`."
        )
    (output_dir / "FULL_SWE_GROUPING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-npz", type=Path, required=True)
    parser.add_argument("--deepseek-npz", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if any(batch <= 0 for batch in args.batches):
        raise ValueError("all batch sizes must be positive")
    inputs = {"qwen": args.qwen_npz.resolve(), "deepseek": args.deepseek_npz.resolve()}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    layers: list[dict[str, Any]] = []
    prefill: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for key, path in inputs.items():
        model_summary, model_layers, model_prefill, model_provenance = run_model(
            path,
            MODEL_SPECS[key],
            args.batches,
        )
        summaries.extend(model_summary)
        layers.extend(model_layers)
        prefill.extend(model_prefill)
        provenance.append(model_provenance)

    write_csv(output_dir / "full_swe_decode_summary.csv", summaries)
    write_csv(output_dir / "full_swe_decode_per_layer.csv", layers)
    write_csv(output_dir / "full_swe_prefill_aggregate_only.csv", prefill)
    (output_dir / "provenance.json").write_text(
        json.dumps({"schema_version": 1, "sources": provenance}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir, summaries, provenance)
    print(json.dumps({"output_dir": str(output_dir), "decode_rows": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
