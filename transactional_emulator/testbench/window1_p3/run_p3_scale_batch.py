#!/usr/bin/env python3
"""Window 1 P3 scale runner built on top of the P2 route replay sidecars.

This wrapper intentionally does not change emulator timing semantics.  It
orchestrates the existing P2 steps:

1. generate true router traces from local model weights,
2. convert true-routing rows to replay traces,
3. replay traces through the Rust emulator harness,
4. export timing/tax/bytes/stage tables,
5. write scale/cost/failure reports for P3.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.window1_p1.p1_utils import PLENA_ROOT
from transactional_emulator.testbench.window1_p2 import (
    build_route_traces,
    export_pilot_results,
    generate_true_routing_with_weights,
    run_trace_batch,
)
from transactional_emulator.testbench.window1_p2.p2_utils import (
    INPUT_FILES,
    MODEL_CONFIGS,
    ensure_paths,
    iter_jsonl,
    load_json,
    write_csv,
    write_json,
)


DEFAULT_OUT_ROOT = PLENA_ROOT / "outputs" / "window1_p3"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYERS = "0,12,23"
FULL_SAMPLE_COUNTS = {
    "bfcl_v3": 5251,
    "gpqa_diamond": 198,
}
MEASUREMENT_NOTE = "self-consistent upper bound, absolute accuracy pending RTL (Window 2)"


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _limit_for(args: argparse.Namespace, benchmark: str) -> int | None:
    if benchmark == "bfcl_v3":
        return args.bfcl_limit
    if benchmark == "gpqa_diamond":
        return args.gpqa_limit
    raise ValueError(f"unknown benchmark: {benchmark}")


def _routing_output(out_root: Path, benchmark: str, model_key: str, layers: list[int], args: argparse.Namespace) -> Path:
    layer_text = "_".join(str(layer) for layer in layers)
    return (
        out_root
        / "true_routing"
        / f"{benchmark}_{model_key}_layers_{layer_text}_{args.write_phases}_d{args.decode_steps}.jsonl"
    )


def _route_trace_summary(out_root: Path, benchmark: str, model_key: str) -> Path:
    return out_root / "route_trace_summaries" / f"{benchmark}_{model_key}_route_traces_summary.json"


def _routing_summary_path(out_root: Path, benchmark: str, model_key: str) -> Path:
    return out_root / "routing_summaries" / f"{benchmark}_{model_key}_true_routing_summary.json"


def _routing_sample_log_path(out_root: Path) -> Path:
    return out_root / "routing_sample_times.jsonl"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _stats(values: list[int | float | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(clean),
        "min": min(clean),
        "median": float(statistics.median(clean)),
        "mean": float(statistics.mean(clean)),
        "p95": _percentile(clean, 0.95),
        "max": max(clean),
    }


def _input_count(benchmark: str, model_key: str) -> int:
    return sum(1 for row in iter_jsonl(INPUT_FILES[benchmark]) if row.get("model_key") == model_key)


def _completed_routing_samples(out_root: Path, args: argparse.Namespace, layers: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for benchmark in args.benchmarks:
        path = _routing_output(out_root, benchmark, args.model_key, layers, args)
        sample_ids = {str(row.get("sample_id")) for row in iter_jsonl(path)} if path.exists() else set()
        counts[benchmark] = len(sample_ids)
    return counts


def _run_true_routing(args: argparse.Namespace, benchmark: str, layers: list[int], out_root: Path) -> dict[str, Any]:
    output = _routing_output(out_root, benchmark, args.model_key, layers, args)
    failure_log = out_root / "failures" / "routing_failures.jsonl"
    sample_log = _routing_sample_log_path(out_root)
    routing_args = argparse.Namespace(
        benchmark=benchmark,
        model_key=args.model_key,
        limit=_limit_for(args, benchmark),
        sample_ids=None,
        max_input_tokens=args.max_input_tokens,
        layers=layers,
        decode_steps=args.decode_steps,
        threads=args.threads,
        batch_size=args.batch_size,
        output=output,
        resume=args.resume,
        sort_by_length=args.sort_by_length,
        write_phases=args.write_phases,
        keep_going=args.keep_going,
        failure_log=failure_log,
        sample_log=sample_log,
    )
    start = time.time()
    summary = generate_true_routing_with_weights.run(routing_args)
    summary["orchestrator_wall_seconds"] = time.time() - start
    summary["input_total_samples_for_model"] = _input_count(benchmark, args.model_key)
    write_json(_routing_summary_path(out_root, benchmark, args.model_key), summary)
    return summary


def _build_route_traces(args: argparse.Namespace, benchmark: str, layers: list[int], out_root: Path) -> list[Path]:
    input_path = _routing_output(out_root, benchmark, args.model_key, layers, args)
    phases = "prefill,decode" if args.write_phases == "both" else args.write_phases
    build_args = argparse.Namespace(
        input=input_path,
        out_dir=out_root / "route_traces",
        summary_out=_route_trace_summary(out_root, benchmark, args.model_key),
        sample_ids=None,
        layers=",".join(str(layer) for layer in layers),
        phases=phases,
        limit=None,
        mlen=args.mlen,
        blen=args.blen,
        emu_threads=args.emu_threads,
        allow_uniform_weights=False,
    )
    return build_route_traces.build_traces(build_args)


def _run_replay(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    replay_args = argparse.Namespace(
        trace_glob=[str(out_root / "route_traces" / "*.json")],
        root=out_root / "trace_replay",
        manifest_out=out_root / "trace_replay_batch_manifest.json",
        cwd=args.cwd,
        mlen=args.mlen,
        blen=args.blen,
        emu_threads=args.emu_threads,
        limit=args.replay_limit,
        stage_profile=args.stage_profile,
        skip_existing=args.skip_existing,
        keep_going=args.keep_going,
        keep_dumps=args.keep_dumps,
        prune_success_artifacts=args.prune_heavy_artifacts,
        experimental_overlap_prefetch_compute=False,
    )
    start = time.time()
    manifest = run_trace_batch.run(replay_args)
    manifest["orchestrator_wall_seconds"] = time.time() - start
    write_json(replay_args.manifest_out, manifest)
    failures = [row for row in manifest.get("runs", []) if row.get("status") == "failed"]
    _write_jsonl(out_root / "failures" / "replay_failures.jsonl", failures)
    return manifest


def _export(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    prefix = out_root / "gpqa_bfcl_emulation"
    export_args = argparse.Namespace(
        root=out_root / "trace_replay",
        out_prefix=prefix,
        out_json=out_root / "gpqa_bfcl_emulation_timing_summary.json",
    )
    payload = export_pilot_results.export(export_args)
    _write_distribution_stats(out_root, payload)
    return payload


def _prune_heavy_artifacts(out_root: Path) -> dict[str, Any]:
    """Drop replay intermediates that are not needed by the P3 exported tables."""
    keep_names = {
        "comparison_params.json",
        "compile_info.json",
        "generated_asm_code.asm",
        "generated_machine_code.mem",
        "gather_scatter_results.json",
        "p2_batch_stdout.log",
        "qwen3_trace_replay_manifest.json",
        "qwen3_trace_replay_results.json",
        "rust_emulator_run_stats.json",
        "rust_emulator_stdout.log",
        "stage_profile.json",
        "tensor_layouts.json",
        "trace.json",
    }
    prune_suffixes = {".pt"}
    prune_names = {
        "hbm_for_behave_sim.bin",
        "fp_sram.bin",
        "int_sram.bin",
        "vram_preload.bin",
        "vector_result.mem",
    }
    rows: list[dict[str, Any]] = []
    trace_root = out_root / "trace_replay"
    for build_dir in sorted(path for path in trace_root.glob("*") if path.is_dir()):
        result_path = build_dir / "qwen3_trace_replay_results.json"
        if not result_path.exists():
            continue
        for path in sorted(child for child in build_dir.iterdir() if child.is_file()):
            if path.name in keep_names:
                continue
            if path.suffix not in prune_suffixes and path.name not in prune_names:
                continue
            size = path.stat().st_size
            path.unlink()
            rows.append({"build_dir": str(build_dir), "path": str(path), "bytes": size})
    total = sum(int(row["bytes"]) for row in rows)
    summary = {
        "schema_version": 1,
        "pruned_file_count": len(rows),
        "pruned_bytes": total,
        "note": "Pruned only post-run intermediates; timing/result/stage/profile/trace/log evidence files are retained.",
        "files": rows,
    }
    write_json(out_root / "pruned_artifacts_manifest.json", summary)
    return summary


def _write_distribution_stats(out_root: Path, payload: dict[str, Any]) -> None:
    tax_by_trace = {row["trace_id"]: row for row in payload.get("routing_tax", [])}
    bytes_by_trace = {row["trace_id"]: row for row in payload.get("bytes_validation", [])}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("runs", []):
        benchmark = str(row.get("benchmark", ""))
        category = str(row.get("category") or "uncategorized")
        groups[(benchmark, "__all__")].append(row)
        groups[(benchmark, category)].append(row)

    stat_rows: list[dict[str, Any]] = []
    for (benchmark, category), rows in sorted(groups.items()):
        trace_ids = [row["trace_id"] for row in rows]
        cycle_stats = _stats([row.get("sim_latency_cycles") for row in rows])
        tax_stats = _stats([tax_by_trace.get(trace_id, {}).get("routing_tax_cycles") for trace_id in trace_ids])
        read_stats = _stats(
            [bytes_by_trace.get(trace_id, {}).get("physical_hbm_bytes_read_run_stats") for trace_id in trace_ids]
        )
        write_stats = _stats(
            [bytes_by_trace.get(trace_id, {}).get("physical_hbm_bytes_written_run_stats") for trace_id in trace_ids]
        )
        stat_rows.append(
            {
                "benchmark": benchmark,
                "category": category,
                "runs": len(rows),
                "cycles_min": cycle_stats["min"],
                "cycles_median": cycle_stats["median"],
                "cycles_mean": cycle_stats["mean"],
                "cycles_p95": cycle_stats["p95"],
                "cycles_max": cycle_stats["max"],
                "routing_tax_cycles_min": tax_stats["min"],
                "routing_tax_cycles_median": tax_stats["median"],
                "routing_tax_cycles_mean": tax_stats["mean"],
                "routing_tax_cycles_p95": tax_stats["p95"],
                "routing_tax_cycles_max": tax_stats["max"],
                "hbm_read_bytes_min": read_stats["min"],
                "hbm_read_bytes_median": read_stats["median"],
                "hbm_read_bytes_mean": read_stats["mean"],
                "hbm_read_bytes_p95": read_stats["p95"],
                "hbm_read_bytes_max": read_stats["max"],
                "hbm_written_bytes_min": write_stats["min"],
                "hbm_written_bytes_median": write_stats["median"],
                "hbm_written_bytes_mean": write_stats["mean"],
                "hbm_written_bytes_p95": write_stats["p95"],
                "hbm_written_bytes_max": write_stats["max"],
                "measurement_note": MEASUREMENT_NOTE,
            }
        )
    write_csv(out_root / "gpqa_bfcl_emulation_distribution_stats.csv", stat_rows)


def _collect_manifest_counts(manifest: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(row.get("status", "unknown") for row in manifest.get("runs", [])))


def _routing_sample_log_stats(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "rows": 0,
            "status_counts": {},
            "seconds": _stats([]),
        }
    rows = list(iter_jsonl(path))
    seconds = [row.get("seconds") for row in rows if row.get("status") == "passed"]
    return {
        "path": str(path),
        "rows": len(rows),
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in rows)),
        "seconds": _stats(seconds),
    }


def _write_cost_report(
    args: argparse.Namespace,
    out_root: Path,
    layers: list[int],
    routing_summaries: list[dict[str, Any]],
    replay_manifest: dict[str, Any] | None,
    export_payload: dict[str, Any] | None,
    prune_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    routing_by_benchmark = {row["benchmark"]: row for row in routing_summaries}
    trace_files = sorted((out_root / "route_traces").glob("*.json"))
    replay_runs = replay_manifest.get("runs", []) if replay_manifest else []
    replay_counts = _collect_manifest_counts(replay_manifest or {})
    routing_sample_stats = _routing_sample_log_stats(_routing_sample_log_path(out_root))
    replay_host_seconds = [
        float(row["host_batch_seconds"])
        for row in replay_runs
        if row.get("status") == "passed" and row.get("host_batch_seconds") is not None
    ]
    replay_pruned_bytes = sum(int(row.get("pruned_bytes") or 0) for row in replay_runs)
    replay_pruned_files = sum(int(row.get("pruned_file_count") or 0) for row in replay_runs)
    observed_processed = sum(int(row.get("processed_samples", 0)) for row in routing_summaries)
    completed_by_benchmark = _completed_routing_samples(out_root, args, layers)
    completed_total = sum(completed_by_benchmark.values())
    full_samples = sum(FULL_SAMPLE_COUNTS.get(benchmark, 0) for benchmark in args.benchmarks)
    full_runs = full_samples * len(layers)
    route_wall_seconds = sum(float(row.get("orchestrator_wall_seconds", 0.0)) for row in routing_summaries)
    replay_wall_seconds = float((replay_manifest or {}).get("orchestrator_wall_seconds", 0.0))
    total_observed_wall = route_wall_seconds + replay_wall_seconds
    avg_route_seconds_per_processed_sample = (
        route_wall_seconds / float(observed_processed) if observed_processed else None
    )
    avg_replay_seconds_per_passed_run = (
        statistics.mean(replay_host_seconds) if replay_host_seconds else None
    )
    projected_route_seconds = (
        avg_route_seconds_per_processed_sample * float(full_samples)
        if avg_route_seconds_per_processed_sample is not None
        else None
    )
    projected_replay_seconds = (
        avg_replay_seconds_per_passed_run * float(full_runs)
        if avg_replay_seconds_per_passed_run is not None
        else None
    )
    projected_total_seconds = (
        projected_route_seconds + projected_replay_seconds
        if projected_route_seconds is not None and projected_replay_seconds is not None
        else None
    )
    disk_sizes = {
        "true_routing_bytes": _dir_size_bytes(out_root / "true_routing"),
        "route_traces_bytes": _dir_size_bytes(out_root / "route_traces"),
        "trace_replay_bytes": _dir_size_bytes(out_root / "trace_replay"),
        "total_out_root_bytes": _dir_size_bytes(out_root),
    }
    observed_runs = len([row for row in replay_runs if row.get("status") in ("passed", "skipped_existing")])
    projected_total_bytes = (
        (disk_sizes["total_out_root_bytes"] / float(observed_runs)) * float(full_runs)
        if observed_runs
        else None
    )
    report = {
        "schema_version": 1,
        "out_root": str(out_root),
        "model_key": args.model_key,
        "model_name": MODEL_CONFIGS[args.model_key]["name"],
        "benchmarks": args.benchmarks,
        "layers": layers,
        "phase": args.write_phases,
        "decode_steps": args.decode_steps,
        "measurement_note": MEASUREMENT_NOTE,
        "routing_summaries": routing_by_benchmark,
        "completed_true_routing_samples_by_benchmark": completed_by_benchmark,
        "completed_true_routing_samples": completed_total,
        "route_trace_count": len(trace_files),
        "route_trace_dir": str(out_root / "route_traces"),
        "replay_manifest": str(out_root / "trace_replay_batch_manifest.json"),
        "replay_status_counts": replay_counts,
        "exported_runs": len((export_payload or {}).get("runs", [])),
        "routing_failure_log": str(out_root / "failures" / "routing_failures.jsonl"),
        "routing_sample_log": str(_routing_sample_log_path(out_root)),
        "routing_sample_log_stats": routing_sample_stats,
        "replay_failure_log": str(out_root / "failures" / "replay_failures.jsonl"),
        "pruned_artifacts": prune_summary
        or {
            "pruned_file_count": 0,
            "pruned_bytes": 0,
            "note": "Artifact pruning was not requested.",
        },
        "replay_pruned_artifacts": {
            "pruned_file_count": replay_pruned_files,
            "pruned_bytes": replay_pruned_bytes,
            "note": "Artifacts pruned immediately after successful replay runs.",
        },
        "observed": {
            "processed_samples_this_run": observed_processed,
            "completed_true_routing_samples": completed_total,
            "route_wall_seconds": route_wall_seconds,
            "replay_wall_seconds": replay_wall_seconds,
            "total_wall_seconds": total_observed_wall,
            "avg_route_seconds_per_processed_sample": avg_route_seconds_per_processed_sample,
            "avg_replay_seconds_per_passed_run": avg_replay_seconds_per_passed_run,
        },
        "full_scale_projection": {
            "full_samples": full_samples,
            "full_runs": full_runs,
            "projected_route_seconds": projected_route_seconds,
            "projected_replay_seconds": projected_replay_seconds,
            "projected_total_seconds": projected_total_seconds,
            "projected_total_hours": projected_total_seconds / 3600.0 if projected_total_seconds is not None else None,
            "projected_total_output_bytes": projected_total_bytes,
        },
        "disk_sizes": disk_sizes,
        "known_limits": [
            "P3 reuses P2 fixed-route replay semantics.",
            "Numbers are self-consistent upper bounds pending RTL calibration.",
            "Replay is decode tok1 only.",
            "Route replay uses zero-activation timing shape, as in P2.",
            "Current P2 true-routing generator supports qwen3 only.",
        ],
    }
    report_stem = args.report_name
    write_json(out_root / f"{report_stem}.json", report)
    _write_cost_report_md(out_root / f"{report_stem}.md", report)
    return report


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}s ({value / 3600.0:.2f}h)"


def _fmt_bytes(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    idx = 0
    while value_f >= 1024.0 and idx < len(units) - 1:
        value_f /= 1024.0
        idx += 1
    return f"{value_f:.2f} {units[idx]}"


def _write_cost_report_md(path: Path, report: dict[str, Any]) -> None:
    observed = report["observed"]
    projection = report["full_scale_projection"]
    disk = report["disk_sizes"]
    pruned = report["pruned_artifacts"]
    replay_pruned = report["replay_pruned_artifacts"]
    replay_counts = report["replay_status_counts"]
    routing_sample_stats = report["routing_sample_log_stats"]
    lines = [
        "# Window 1 P3 Scale Batch Report",
        "",
        f"- model: `{report['model_name']}` (`{report['model_key']}`)",
        f"- benchmarks: `{','.join(report['benchmarks'])}`",
        f"- layers: `{','.join(str(x) for x in report['layers'])}`",
        f"- phase: `{report['phase']}`, decode_steps: `{report['decode_steps']}`",
        f"- measurement note: {report['measurement_note']}",
        "",
        "## Observed Batch",
        "",
        f"- processed true-routing samples this run: `{observed['processed_samples_this_run']}`",
        f"- completed true-routing samples in output: `{observed['completed_true_routing_samples']}`",
        f"- cumulative routing sample log rows: `{routing_sample_stats['rows']}`",
        f"- routing sample status counts: `{routing_sample_stats['status_counts']}`",
        f"- routing sample seconds median/mean/p95: "
        f"`{routing_sample_stats['seconds']['median']}` / "
        f"`{routing_sample_stats['seconds']['mean']}` / "
        f"`{routing_sample_stats['seconds']['p95']}`",
        f"- route trace files: `{report['route_trace_count']}`",
        f"- replay status counts: `{replay_counts}`",
        f"- exported timing runs: `{report['exported_runs']}`",
        f"- replay-time pruned artifacts: `{replay_pruned['pruned_file_count']}` files, "
        f"`{_fmt_bytes(replay_pruned['pruned_bytes'])}`",
        f"- pruned heavy artifacts: `{pruned['pruned_file_count']}` files, `{_fmt_bytes(pruned['pruned_bytes'])}`",
        f"- route generation wall time: `{_fmt_seconds(observed['route_wall_seconds'])}`",
        f"- replay wall time: `{_fmt_seconds(observed['replay_wall_seconds'])}`",
        f"- total observed wall time: `{_fmt_seconds(observed['total_wall_seconds'])}`",
        f"- avg route seconds/sample: `{observed['avg_route_seconds_per_processed_sample']}`",
        f"- avg replay seconds/passed run: `{observed['avg_replay_seconds_per_passed_run']}`",
        "",
        "## Full-Scale Projection",
        "",
        f"- full samples: `{projection['full_samples']}`",
        f"- full representative-layer runs: `{projection['full_runs']}`",
        f"- projected route generation: `{_fmt_seconds(projection['projected_route_seconds'])}`",
        f"- projected replay: `{_fmt_seconds(projection['projected_replay_seconds'])}`",
        f"- projected total: `{_fmt_seconds(projection['projected_total_seconds'])}`",
        f"- projected output size: `{_fmt_bytes(projection['projected_total_output_bytes'])}`",
        "",
        "## Disk",
        "",
        f"- true routing: `{_fmt_bytes(disk['true_routing_bytes'])}`",
        f"- route traces: `{_fmt_bytes(disk['route_traces_bytes'])}`",
        f"- trace replay: `{_fmt_bytes(disk['trace_replay_bytes'])}`",
        f"- total output root: `{_fmt_bytes(disk['total_out_root_bytes'])}`",
        "",
        "## Limits",
        "",
    ]
    lines.extend(f"- {item}" for item in report["known_limits"])
    lines.extend(
        [
            "",
            "## Failure Logs",
            "",
            f"- routing: `{report['routing_failure_log']}`",
            f"- replay: `{report['replay_failure_log']}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_paths()
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "failures").mkdir(parents=True, exist_ok=True)
    (out_root / "failures" / "routing_failures.jsonl").touch(exist_ok=True)
    (out_root / "failures" / "replay_failures.jsonl").touch(exist_ok=True)
    layers = _parse_csv_ints(args.layers)
    routing_summaries: list[dict[str, Any]] = []

    if not args.skip_routing:
        for benchmark in args.benchmarks:
            routing_summaries.append(_run_true_routing(args, benchmark, layers, out_root))
    else:
        for benchmark in args.benchmarks:
            summary_path = _routing_summary_path(out_root, benchmark, args.model_key)
            if summary_path.exists():
                routing_summaries.append(load_json(summary_path))

    if not args.skip_build_traces:
        for benchmark in args.benchmarks:
            _build_route_traces(args, benchmark, layers, out_root)

    replay_manifest = None
    if not args.skip_replay:
        replay_manifest = _run_replay(args, out_root)
    else:
        manifest_path = out_root / "trace_replay_batch_manifest.json"
        if manifest_path.exists():
            replay_manifest = load_json(manifest_path)

    export_payload = None
    if not args.skip_export:
        export_payload = _export(args, out_root)
    else:
        export_path = out_root / "gpqa_bfcl_emulation_timing_summary.json"
        if export_path.exists():
            export_payload = load_json(export_path)

    prune_summary = _prune_heavy_artifacts(out_root) if args.prune_heavy_artifacts else None

    report = _write_cost_report(
        args,
        out_root,
        layers,
        routing_summaries,
        replay_manifest,
        export_payload,
        prune_summary,
    )
    write_json(out_root / "p3_scale_batch_summary.json", report)
    _append_jsonl(
        out_root / "p3_scale_batch_history.jsonl",
        {
            "timestamp": time.time(),
            "benchmarks": args.benchmarks,
            "bfcl_limit": args.bfcl_limit,
            "gpqa_limit": args.gpqa_limit,
            "layers": layers,
            "write_phases": args.write_phases,
            "decode_steps": args.decode_steps,
            "skip_routing": args.skip_routing,
            "skip_build_traces": args.skip_build_traces,
            "skip_replay": args.skip_replay,
            "skip_export": args.skip_export,
            "prune_heavy_artifacts": args.prune_heavy_artifacts,
            "observed": report["observed"],
            "replay_status_counts": report["replay_status_counts"],
            "exported_runs": report["exported_runs"],
            "replay_pruned_artifacts": {
                "pruned_file_count": report["replay_pruned_artifacts"].get("pruned_file_count"),
                "pruned_bytes": report["replay_pruned_artifacts"].get("pruned_bytes"),
            },
            "pruned_artifacts": {
                "pruned_file_count": report["pruned_artifacts"].get("pruned_file_count"),
                "pruned_bytes": report["pruned_artifacts"].get("pruned_bytes"),
            },
        },
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), default="qwen3")
    parser.add_argument("--benchmarks", type=_parse_csv_strings, default=_parse_csv_strings("bfcl_v3,gpqa_diamond"))
    parser.add_argument("--bfcl-limit", type=int, default=50)
    parser.add_argument("--gpqa-limit", type=int, default=20)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--write-phases", choices=("prefill", "decode", "both"), default="decode")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--mlen", type=int, default=128)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    parser.add_argument("--replay-limit", type=int)
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--keep-dumps", action="store_true")
    parser.add_argument("--skip-routing", action="store_true")
    parser.add_argument("--skip-build-traces", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--prune-heavy-artifacts", action="store_true")
    parser.add_argument("--report-name", default="s1_cost_report")
    args = parser.parse_args()

    unknown = [benchmark for benchmark in args.benchmarks if benchmark not in INPUT_FILES]
    if unknown:
        raise ValueError(f"unknown benchmarks: {unknown}")
    report = run(args)
    print(f"wrote {args.out_root / (args.report_name + '.md')}")
    print(
        "processed_samples="
        f"{report['observed']['processed_samples_this_run']} "
        f"completed_samples={report['observed']['completed_true_routing_samples']} "
        f"route_traces={report['route_trace_count']} "
        f"exported_runs={report['exported_runs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
