#!/usr/bin/env python3
"""Window 1 P3-rev representative subset and parallel replay runner.

This sidecar keeps the P2 replay semantics intact.  It only changes scale
strategy: select a reproducible representative subset, generate true routes for
missing samples, then replay selected route traces in parallel with checkpointed
progress.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.window1_p1.p1_utils import PLENA_ROOT
from transactional_emulator.testbench.window1_p2 import (
    build_route_traces,
    export_pilot_results,
    generate_true_routing_with_weights,
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
from transactional_emulator.testbench.window1_p2.run_trace_batch import _prune_success_artifacts


DEFAULT_OUT_ROOT = PLENA_ROOT / "outputs" / "window1_p3"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAYERS = [0, 12, 23]
DEFAULT_SEED = 20260704
MEASUREMENT_NOTE = "self-consistent upper bound, absolute accuracy pending RTL (Window 2)"
ROUTING_STAGE_NAMES = {
    "accumulator_init",
    "gather",
    "expert_weight_address",
    "expert_route_weight",
    "scatter_combine",
}


def _parse_csv_strings(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_csv_ints(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(",") if part.strip()]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _stable_seed(*parts: object) -> int:
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _stats(values: list[int | float | None]) -> dict[str, Any]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(clean)
    if len(ordered) == 1:
        p95 = ordered[0]
    else:
        rank = (len(ordered) - 1) * 0.95
        lo = int(rank)
        hi = min(lo + 1, len(ordered) - 1)
        frac = rank - lo
        p95 = ordered[lo] * (1.0 - frac) + ordered[hi] * frac
    return {
        "count": len(clean),
        "min": min(clean),
        "median": float(statistics.median(clean)),
        "mean": float(statistics.mean(clean)),
        "p95": float(p95),
        "max": max(clean),
    }


def _fmt_seconds(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    value_f = float(value)
    return f"{value_f:.1f}s ({value_f / 3600.0:.2f}h)"


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


def _input_records(benchmark: str, model_key: str) -> list[dict[str, Any]]:
    rows = []
    for idx, row in enumerate(iter_jsonl(INPUT_FILES[benchmark])):
        if row.get("model_key") != model_key:
            continue
        rows.append(
            {
                "benchmark": benchmark,
                "model_key": model_key,
                "sample_id": str(row["sample_id"]),
                "sample_index": idx,
                "category": str(row.get("category") or "uncategorized"),
                "input_tokens": int(row.get("input_tokens") or len(row.get("input_ids") or [])),
                "input_chars": row.get("input_chars"),
                "source_file": row.get("source_file"),
            }
        )
    return rows


def _largest_remainder_quotas(counts: Counter[str], target: int) -> dict[str, int]:
    if target <= 0:
        return {}
    total = sum(counts.values())
    if total <= 0:
        return {}
    categories = sorted(counts)
    raw = {category: (counts[category] * float(target)) / float(total) for category in categories}
    quotas = {category: min(counts[category], int(raw[category])) for category in categories}
    for category in categories:
        if counts[category] > 0 and quotas[category] == 0 and target >= len(categories):
            quotas[category] = 1
    while sum(quotas.values()) > target:
        candidates = [category for category in categories if quotas[category] > 1]
        category = min(candidates, key=lambda item: (raw[item] - int(raw[item]), quotas[item], item))
        quotas[category] -= 1
    while sum(quotas.values()) < target:
        candidates = [category for category in categories if quotas[category] < counts[category]]
        if not candidates:
            break
        category = max(candidates, key=lambda item: (raw[item] - int(raw[item]), counts[item], item))
        quotas[category] += 1
    return quotas


def _select_bfcl(rows: list[dict[str, Any]], target: int, seed: int) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    counts = Counter({category: len(items) for category, items in by_category.items()})
    quotas = _largest_remainder_quotas(counts, min(target, len(rows)))
    selected: list[dict[str, Any]] = []
    categories = []
    for category in sorted(by_category):
        items = sorted(by_category[category], key=lambda row: (row["sample_index"], row["sample_id"]))
        quota = int(quotas.get(category, 0))
        rng = random.Random(_stable_seed(seed, "bfcl_v3", category))
        shuffled = items[:]
        rng.shuffle(shuffled)
        chosen = sorted(shuffled[:quota], key=lambda row: (row["sample_index"], row["sample_id"]))
        selected.extend(chosen)
        categories.append(
            {
                "category": category,
                "available": len(items),
                "quota": quota,
                "selected_sample_ids": [row["sample_id"] for row in chosen],
            }
        )
    selected.sort(key=lambda row: (row["sample_index"], row["sample_id"]))
    return {
        "available": len(rows),
        "target": target,
        "selected_count": len(selected),
        "selection_method": "category_stratified_largest_remainder_fixed_seed",
        "categories": categories,
        "selected_samples": selected,
    }


def _select_gpqa(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = sorted(rows, key=lambda row: (row["sample_index"], row["sample_id"]))
    categories = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_category[row["category"]].append(row)
    for category in sorted(by_category):
        categories.append(
            {
                "category": category,
                "available": len(by_category[category]),
                "quota": len(by_category[category]),
                "selected_sample_ids": [row["sample_id"] for row in by_category[category]],
            }
        )
    return {
        "available": len(rows),
        "target": "all",
        "selected_count": len(selected),
        "selection_method": "all_samples_no_sampling",
        "categories": categories,
        "selected_samples": selected,
    }


def select_subset(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "seed": args.seed,
        "model_key": args.model_key,
        "model_name": MODEL_CONFIGS[args.model_key]["name"],
        "layers": args.layers,
        "phase": args.write_phases,
        "decode_steps": args.decode_steps,
        "measurement_note": MEASUREMENT_NOTE,
        "bfcl_target": args.bfcl_target,
        "benchmarks": {},
    }
    if "bfcl_v3" in args.benchmarks:
        payload["benchmarks"]["bfcl_v3"] = _select_bfcl(
            _input_records("bfcl_v3", args.model_key),
            args.bfcl_target,
            args.seed,
        )
    if "gpqa_diamond" in args.benchmarks:
        payload["benchmarks"]["gpqa_diamond"] = _select_gpqa(_input_records("gpqa_diamond", args.model_key))
    write_json(args.selected_out, payload)
    return payload


def _load_selection(args: argparse.Namespace) -> dict[str, Any]:
    if args.selected_out.exists() and not args.reselect:
        return load_json(args.selected_out)
    return select_subset(args)


def _sample_ids(selection: dict[str, Any], benchmark: str) -> list[str]:
    bench = selection["benchmarks"].get(benchmark, {})
    return [str(row["sample_id"]) for row in bench.get("selected_samples", [])]


def _sample_ids_for_args(selection: dict[str, Any], benchmark: str, args: argparse.Namespace) -> list[str]:
    bench = selection["benchmarks"].get(benchmark, {})
    rows = list(bench.get("selected_samples", []))
    shard_count = getattr(args, "sample_shard_count", None)
    shard_index = getattr(args, "sample_shard_index", None)
    if shard_count is not None:
        if shard_count <= 0:
            raise ValueError("--sample-shard-count must be positive")
        if shard_index is None or not (0 <= shard_index < shard_count):
            raise ValueError("--sample-shard-index must be in [0, sample_shard_count)")
        balanced = sorted(
            rows,
            key=lambda row: (-int(row.get("input_tokens") or 0), str(row.get("sample_id"))),
        )
        rows = [row for idx, row in enumerate(balanced) if idx % shard_count == shard_index]
        rows.sort(key=lambda row: (int(row.get("sample_index") or 0), str(row.get("sample_id"))))
    return [str(row["sample_id"]) for row in rows]


def _routing_output(out_root: Path, benchmark: str, model_key: str, layers: list[int], args: argparse.Namespace) -> Path:
    layer_text = "_".join(str(layer) for layer in layers)
    suffix = str(getattr(args, "routing_output_suffix", "") or "")
    return (
        out_root
        / "true_routing"
        / f"{benchmark}_{model_key}_layers_{layer_text}_{args.write_phases}_d{args.decode_steps}{suffix}.jsonl"
    )


def _routing_resume_sources(
    out_root: Path,
    benchmark: str,
    model_key: str,
    layers: list[int],
    args: argparse.Namespace,
) -> list[Path]:
    layer_text = "_".join(str(layer) for layer in layers)
    return sorted(
        (out_root / "true_routing").glob(
            f"{benchmark}_{model_key}_layers_{layer_text}_{args.write_phases}_d{args.decode_steps}*.jsonl"
        )
    )


def run_route_generation(args: argparse.Namespace, selection: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    log_suffix = str(getattr(args, "routing_log_suffix", "") or "")
    for benchmark in args.benchmarks:
        ids = _sample_ids_for_args(selection, benchmark, args)
        if not ids:
            continue
        routing_args = argparse.Namespace(
            benchmark=benchmark,
            model_key=args.model_key,
            limit=None,
            sample_ids=",".join(ids),
            max_input_tokens=args.max_input_tokens,
            layers=args.layers,
            decode_steps=args.decode_steps,
            threads=args.threads,
            batch_size=args.batch_size,
            output=_routing_output(args.out_root, benchmark, args.model_key, args.layers, args),
            resume=True,
            sort_by_length=args.sort_by_length,
            write_phases=args.write_phases,
            keep_going=args.keep_going,
            failure_log=args.out_root / "failures" / f"routing_failures{log_suffix}.jsonl",
            sample_log=args.out_root / f"routing_sample_times{log_suffix}.jsonl",
            resume_from=_routing_resume_sources(args.out_root, benchmark, args.model_key, args.layers, args),
        )
        start = time.time()
        summary = generate_true_routing_with_weights.run(routing_args)
        summary["orchestrator_wall_seconds"] = time.time() - start
        write_json(args.out_root / "routing_summaries" / f"p3_rev_{benchmark}_{args.model_key}.json", summary)
        summaries.append(summary)
    return summaries


def build_selected_traces(args: argparse.Namespace, selection: dict[str, Any]) -> list[Path]:
    written: list[Path] = []
    for benchmark in args.benchmarks:
        ids = _sample_ids_for_args(selection, benchmark, args)
        if not ids:
            continue
        phases = "prefill,decode" if args.write_phases == "both" else args.write_phases
        route_inputs = _routing_resume_sources(args.out_root, benchmark, args.model_key, args.layers, args)
        if not route_inputs:
            route_inputs = [_routing_output(args.out_root, benchmark, args.model_key, args.layers, args)]
        for route_input in route_inputs:
            build_args = argparse.Namespace(
                input=route_input,
                out_dir=args.out_root / "route_traces",
                summary_out=args.out_root
                / "route_trace_summaries"
                / f"p3_rev_{benchmark}_{args.model_key}_{route_input.stem}.json",
                sample_ids=",".join(ids),
                layers=",".join(str(layer) for layer in args.layers),
                phases=phases,
                limit=None,
                mlen=args.mlen,
                blen=args.blen,
                emu_threads=args.emu_threads,
                allow_uniform_weights=False,
            )
            written.extend(build_route_traces.build_traces(build_args))
    return written


def _trace_matches(trace: dict[str, Any], selection: dict[str, Any], layers: set[int], phases: set[str]) -> bool:
    workload = trace.get("workload", {})
    benchmark = str(workload.get("benchmark"))
    sample_id = str(workload.get("sample_id"))
    selected = set(_sample_ids(selection, benchmark))
    if sample_id not in selected:
        return False
    if int(trace.get("model", {}).get("layer_index")) not in layers:
        return False
    if str(workload.get("phase")) not in phases:
        return False
    return True


def selected_trace_paths(args: argparse.Namespace, selection: dict[str, Any]) -> list[Path]:
    layers = set(args.layers)
    phases = {"decode"} if args.write_phases == "decode" else {"prefill"} if args.write_phases == "prefill" else {"prefill", "decode"}
    paths = []
    for path in sorted((args.out_root / "route_traces").glob("*.json")):
        try:
            trace = load_json(path)
        except Exception:
            continue
        if _trace_matches(trace, selection, layers, phases):
            paths.append(path)
    def key(path: Path) -> tuple[str, int, str, int]:
        trace = load_json(path)
        workload = trace["workload"]
        return (
            str(workload.get("benchmark")),
            int(workload.get("sample_index") or 0),
            str(workload.get("sample_id")),
            int(trace["model"]["layer_index"]),
        )
    return sorted(paths, key=key)


def _filtered_trace_paths_by_id(trace_paths: list[Path], trace_id_file: Path | None) -> list[Path]:
    if trace_id_file is None:
        return trace_paths
    wanted = {
        line.strip()
        for line in trace_id_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not wanted:
        return []
    filtered = []
    for path in trace_paths:
        try:
            trace = load_json(path)
        except Exception:
            continue
        if str(trace.get("trace_id")) in wanted:
            filtered.append(path)
    return filtered


def progress_snapshot(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    selected_pairs = {
        (benchmark, str(row["sample_id"]))
        for benchmark, obj in selection.get("benchmarks", {}).items()
        for row in obj.get("selected_samples", [])
    }
    route_progress = {}
    for benchmark, obj in selection.get("benchmarks", {}).items():
        ids = {str(row["sample_id"]) for row in obj.get("selected_samples", [])}
        route_file = _routing_output(args.out_root, benchmark, args.model_key, args.layers, args)
        layer_text = "_".join(str(layer) for layer in args.layers)
        route_files = sorted(
            (args.out_root / "true_routing").glob(
                f"{benchmark}_{args.model_key}_layers_{layer_text}_{args.write_phases}_d{args.decode_steps}*.jsonl"
            )
        )
        got: dict[str, set[tuple[str, int, int | None]]] = {}
        for current_route_file in route_files:
            if not current_route_file.exists():
                continue
            for row in iter_jsonl(current_route_file):
                sample_id = str(row.get("sample_id"))
                if sample_id not in ids:
                    continue
                got.setdefault(sample_id, set()).add(
                    (
                        str(row.get("phase")),
                        int(row.get("layer")),
                        int(row["decode_step"]) if row.get("decode_step") is not None else None,
                    )
                )
        complete = sum(1 for sample_id in ids if len(got.get(sample_id, set())) >= len(args.layers))
        route_progress[benchmark] = {
            "selected_samples": len(ids),
            "complete_route_samples": complete,
            "route_rows": sum(len(value) for value in got.values()),
            "routing_output": str(route_file),
            "routing_outputs_scanned": [str(path) for path in route_files],
        }

    trace_paths = selected_trace_paths(args, selection)
    replay_complete = 0
    replay_rows = []
    for path in trace_paths:
        trace = load_json(path)
        build_dir = args.out_root / "trace_replay" / trace["trace_id"]
        status = "complete" if _result_ok(build_dir, require_stage_profile=args.stage_profile) else "missing"
        if status == "complete":
            replay_complete += 1
        replay_rows.append({"trace_id": trace["trace_id"], "status": status})

    exported_runs = 0
    selected_export = args.out_root / "p3_rev_emulation_timing.csv"
    if selected_export.exists():
        with selected_export.open(newline="", encoding="utf-8") as handle:
            exported_runs = sum(1 for _ in csv.DictReader(handle))
    failure_counts = {}
    failure_dir = args.out_root / "failures"
    if failure_dir.exists():
        for path in sorted(failure_dir.glob("*.jsonl")):
            failure_counts[path.name] = sum(1 for _ in path.open(encoding="utf-8"))

    payload = {
        "schema_version": 1,
        "updated_at_unix": time.time(),
        "selected_samples": sum(len(obj.get("selected_samples", [])) for obj in selection.get("benchmarks", {}).values()),
        "expected_representative_runs": len(selected_pairs) * len(args.layers),
        "route_progress": route_progress,
        "selected_trace_files": len(trace_paths),
        "replay_complete_runs": replay_complete,
        "exported_selected_runs": exported_runs,
        "failure_counts": failure_counts,
    }
    write_json(args.out_root / "p3_rev_progress_snapshot.json", payload)
    return payload


def _result_ok(build_dir: Path, require_stage_profile: bool = True) -> bool:
    result_path = build_dir / "qwen3_trace_replay_results.json"
    run_stats_path = build_dir / "rust_emulator_run_stats.json"
    if not result_path.exists() or not run_stats_path.exists():
        return False
    if require_stage_profile and not (build_dir / "stage_profile.json").exists():
        return False
    try:
        result = load_json(result_path)
    except Exception:
        return False
    return bool(result.get("functional_gate", True))


def _run_trace_command(
    trace_path: Path,
    build_dir: Path,
    args: argparse.Namespace,
    *,
    log_name: str,
    use_time: bool = False,
) -> tuple[int, Path, int | None]:
    cmd = [
        sys.executable,
        "-m",
        "transactional_emulator.testbench.window1_p2.qwen3_trace_replay_test",
        str(trace_path),
        "--build-dir",
        str(build_dir),
        "--mlen",
        str(args.mlen),
        "--blen",
        str(args.blen),
        "--emu-threads",
        str(args.emu_threads),
    ]
    if args.stage_profile:
        cmd.append("--stage-profile")
    run_cmd = cmd
    if use_time:
        run_cmd = ["/usr/bin/time", "-v", *cmd]
    build_dir.mkdir(parents=True, exist_ok=True)
    log_path = build_dir / log_name
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            run_cmd,
            cwd=args.cwd,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    rss_kb = None
    if use_time:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
        if match:
            rss_kb = int(match.group(1))
    return proc.returncode, log_path, rss_kb


def measure_rss(args: argparse.Namespace, trace_paths: list[Path]) -> dict[str, Any]:
    if not trace_paths:
        raise ValueError("no selected traces available for RSS probe")
    probe_trace = trace_paths[0]
    trace = load_json(probe_trace)
    probe_root = args.out_root / "p3_rev_rss_probe"
    build_dir = probe_root / trace["trace_id"]
    start = time.time()
    code, log_path, rss_kb = _run_trace_command(probe_trace, build_dir, args, log_name="rss_probe.log", use_time=True)
    row = {
        "schema_version": 1,
        "trace": str(probe_trace),
        "trace_id": trace["trace_id"],
        "build_dir": str(build_dir),
        "return_code": code,
        "passed": code == 0,
        "host_seconds": time.time() - start,
        "max_rss_kb": rss_kb,
        "log_path": str(log_path),
    }
    if code == 0 and args.prune_success_artifacts:
        row.update(_prune_success_artifacts(build_dir))
    write_json(args.out_root / "p3_rev_rss_probe.json", row)
    if code != 0:
        raise RuntimeError(f"RSS probe failed for {trace['trace_id']}; see {log_path}")
    return row


def _mem_available_kb() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0


def choose_width(args: argparse.Namespace, rss: dict[str, Any] | None) -> dict[str, Any]:
    if args.width is not None:
        width = max(1, int(args.width))
        reason = "explicit_width"
    else:
        rss_kb = int((rss or {}).get("max_rss_kb") or 0)
        cores = os.cpu_count() or 1
        mem_available_kb = _mem_available_kb()
        memory_budget_kb = int(mem_available_kb * float(args.memory_budget_fraction))
        if rss_kb <= 0:
            width = 1
            reason = "missing_rss_fallback"
        else:
            width = max(1, min(max(1, cores - 2), max(1, memory_budget_kb // rss_kb)))
            reason = "min_cores_minus_2_and_memory_budget_over_rss"
    if args.max_width is not None:
        width = min(width, max(1, int(args.max_width)))
        reason += "_capped"
    payload = {
        "schema_version": 1,
        "width": width,
        "reason": reason,
        "cpu_count": os.cpu_count(),
        "mem_available_kb": _mem_available_kb(),
        "memory_budget_fraction": args.memory_budget_fraction,
        "rss_probe": rss,
    }
    write_json(args.out_root / "p3_rev_parallel_width.json", payload)
    return payload


def _worker_run_trace(task: dict[str, Any]) -> dict[str, Any]:
    trace_path = Path(task["trace_path"])
    out_root = Path(task["out_root"])
    cwd = Path(task["cwd"])
    trace = load_json(trace_path)
    build_dir = out_root / "trace_replay" / trace["trace_id"]
    start = time.time()
    row = {
        "trace": str(trace_path),
        "trace_id": trace["trace_id"],
        "build_dir": str(build_dir),
        "worker_pid": os.getpid(),
    }
    try:
        if task["skip_existing"] and _result_ok(build_dir, require_stage_profile=task["stage_profile"]):
            row.update({"status": "skipped_existing", "host_batch_seconds": 0.0})
            return row
        ns = argparse.Namespace(
            cwd=cwd,
            mlen=task["mlen"],
            blen=task["blen"],
            emu_threads=task["emu_threads"],
            stage_profile=task["stage_profile"],
        )
        code, log_path, _ = _run_trace_command(
            trace_path,
            build_dir,
            ns,
            log_name="p3_rev_parallel_stdout.log",
            use_time=False,
        )
        row.update(
            {
                "return_code": code,
                "status": "passed" if code == 0 else "failed",
                "host_batch_seconds": time.time() - start,
                "log_path": str(log_path),
            }
        )
        if row["status"] == "passed" and task["prune_success_artifacts"]:
            row.update(_prune_success_artifacts(build_dir))
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "host_batch_seconds": time.time() - start,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return row


def _determinism_worker(task: dict[str, Any]) -> dict[str, Any]:
    trace_path = Path(task["trace_path"])
    build_root = Path(task["build_root"])
    cwd = Path(task["cwd"])
    trace = load_json(trace_path)
    build_dir = build_root / trace["trace_id"]
    start = time.time()
    row = {
        "trace": str(trace_path),
        "trace_id": trace["trace_id"],
        "build_dir": str(build_dir),
        "worker_pid": os.getpid(),
    }
    try:
        if task.get("skip_existing") and _result_ok(build_dir, require_stage_profile=task["stage_profile"]):
            row.update({"status": "skipped_existing", "host_batch_seconds": 0.0})
            if task.get("prune_success_artifacts"):
                row.update(_prune_success_artifacts(build_dir))
            return row
        ns = argparse.Namespace(
            cwd=cwd,
            mlen=task["mlen"],
            blen=task["blen"],
            emu_threads=task["emu_threads"],
            stage_profile=task["stage_profile"],
        )
        code, log_path, _ = _run_trace_command(
            trace_path,
            build_dir,
            ns,
            log_name=task["log_name"],
            use_time=False,
        )
        row.update(
            {
                "return_code": code,
                "status": "passed" if code == 0 else "failed",
                "host_batch_seconds": time.time() - start,
                "log_path": str(log_path),
            }
        )
        if row["status"] == "passed" and task.get("prune_success_artifacts"):
            row.update(_prune_success_artifacts(build_dir))
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "host_batch_seconds": time.time() - start,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return row


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_VOLATILE_JSON_KEYS = {
    "build_dir",
    "command",
    "config_path",
    "cwd",
    "ended_at_utc",
    "host_wall_time_seconds",
    "log_path",
    "stage_profile_path",
    "started_at_utc",
    "stats_path",
}


def _normalized_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalized_json(child)
            for key, child in sorted(value.items())
            if key not in _VOLATILE_JSON_KEYS
        }
    if isinstance(value, list):
        return [_normalized_json(child) for child in value]
    return value


def _sha256_json(path: Path, *, normalized: bool) -> str | None:
    if not path.exists():
        return None
    payload = load_json(path)
    if normalized:
        payload = _normalized_json(payload)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _comparison_hash(path: Path, mode: str) -> str | None:
    if mode == "raw":
        return _sha256_file(path)
    if mode == "json":
        return _sha256_json(path, normalized=False)
    if mode == "normalized_json":
        return _sha256_json(path, normalized=True)
    raise ValueError(f"unknown comparison mode: {mode}")


def _determinism_trace_subset(trace_paths: list[Path], sample_count: int) -> list[Path]:
    if sample_count <= 0:
        return []
    by_sample: dict[tuple[str, str], list[Path]] = defaultdict(list)
    order: list[tuple[str, str]] = []
    for path in trace_paths:
        trace = load_json(path)
        workload = trace["workload"]
        key = (str(workload.get("benchmark")), str(workload.get("sample_id")))
        if key not in by_sample:
            order.append(key)
        by_sample[key].append(path)
    chosen: list[Path] = []
    for key in order[:sample_count]:
        chosen.extend(sorted(by_sample[key], key=lambda path: int(load_json(path)["model"]["layer_index"])))
    return chosen


def run_determinism_gate(args: argparse.Namespace, trace_paths: list[Path], width_info: dict[str, Any] | None) -> dict[str, Any]:
    chosen = _determinism_trace_subset(trace_paths, args.determinism_samples)
    if not chosen:
        payload = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no determinism traces selected",
            "determinism_samples": args.determinism_samples,
        }
        write_json(args.out_root / "p3_rev_determinism.json", payload)
        return payload

    root = args.out_root / "p3_rev_determinism"
    serial_root = root / "serial"
    parallel_root = root / "parallel"
    compare_files = [
        {"file": "qwen3_trace_replay_results.json", "mode": "normalized_json"},
        {"file": "stage_profile.json", "mode": "json"},
        {"file": "rust_emulator_run_stats.json", "mode": "normalized_json"},
        {"file": "gather_scatter_results.json", "mode": "normalized_json"},
    ]

    serial_rows = []
    started = time.time()
    for path in chosen:
        row = _determinism_worker(
            {
                "trace_path": str(path),
                "build_root": str(serial_root),
                "cwd": str(args.cwd),
                "mlen": args.mlen,
                "blen": args.blen,
                "emu_threads": args.emu_threads,
                "stage_profile": args.stage_profile,
                "log_name": "serial_stdout.log",
                "skip_existing": True,
                "prune_success_artifacts": args.prune_success_artifacts,
            }
        )
        serial_rows.append(row)
        if row.get("status") != "passed" and not args.keep_going:
            break

    parallel_width = args.determinism_width
    if parallel_width is None:
        parallel_width = min(int((width_info or {}).get("width") or 1), len(chosen), 4)
    parallel_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, parallel_width)) as executor:
        futures = [
            executor.submit(
                _determinism_worker,
                {
                    "trace_path": str(path),
                    "build_root": str(parallel_root),
                    "cwd": str(args.cwd),
                    "mlen": args.mlen,
                    "blen": args.blen,
                    "emu_threads": args.emu_threads,
                    "stage_profile": args.stage_profile,
                    "log_name": "parallel_stdout.log",
                    "skip_existing": True,
                    "prune_success_artifacts": args.prune_success_artifacts,
                },
            )
            for path in chosen
        ]
        for future in concurrent.futures.as_completed(futures):
            parallel_rows.append(future.result())

    comparisons = []
    for path in chosen:
        trace = load_json(path)
        trace_id = trace["trace_id"]
        file_rows = []
        for item in compare_files:
            name = item["file"]
            mode = item["mode"]
            serial_hash = _comparison_hash(serial_root / trace_id / name, mode)
            parallel_hash = _comparison_hash(parallel_root / trace_id / name, mode)
            file_rows.append(
                {
                    "file": name,
                    "mode": mode,
                    "serial_sha256": serial_hash,
                    "parallel_sha256": parallel_hash,
                    "matches": serial_hash is not None and serial_hash == parallel_hash,
                }
            )
        comparisons.append(
            {
                "trace_id": trace_id,
                "trace": str(path),
                "files": file_rows,
                "all_files_match": all(row["matches"] for row in file_rows),
            }
        )

    payload = {
        "schema_version": 1,
        "status": "passed" if all(row["all_files_match"] for row in comparisons) else "failed",
        "started_at": started,
        "ended_at": time.time(),
        "wall_seconds": time.time() - started,
        "determinism_samples": args.determinism_samples,
        "trace_count": len(chosen),
        "parallel_width": parallel_width,
        "serial_status_counts": dict(Counter(str(row.get("status")) for row in serial_rows)),
        "parallel_status_counts": dict(Counter(str(row.get("status")) for row in parallel_rows)),
        "compare_files": compare_files,
        "comparisons": comparisons,
    }
    write_json(args.out_root / "p3_rev_determinism.json", payload)
    return payload


def _progress_payload(
    *,
    args: argparse.Namespace,
    run_id: str,
    total: int,
    started_at: float,
    rows: list[dict[str, Any]],
    running: int,
    width: int,
    manifest_jsonl: Path,
    failure_log: Path,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    completed = sum(status_counts.values())
    return {
        "schema_version": 1,
        "run_id": run_id,
        "pid": os.getpid(),
        "updated_at_unix": time.time(),
        "started_at_unix": started_at,
        "elapsed_seconds": time.time() - started_at,
        "selected_trace_count": total,
        "completed_events": completed,
        "running": running,
        "remaining_events": max(0, total - completed - running),
        "status_counts": dict(status_counts),
        "width": width,
        "manifest_jsonl": str(manifest_jsonl),
        "failure_log": str(failure_log),
    }


def _replay_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    name = getattr(args, "replay_run_name", None) or "p3_rev_parallel_replay"
    if name == "p3_rev_parallel_replay":
        return {
            "progress": args.out_root / "p3_rev_progress.json",
            "manifest_jsonl": args.out_root / "p3_rev_parallel_replay_manifest.jsonl",
            "manifest_json": args.out_root / "p3_rev_parallel_replay_manifest.json",
            "failure_log": args.out_root / "failures" / "p3_rev_replay_failures.jsonl",
        }
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "p3_rev_parallel_replay"
    return {
        "progress": args.out_root / f"{safe}_progress.json",
        "manifest_jsonl": args.out_root / f"{safe}_manifest.jsonl",
        "manifest_json": args.out_root / f"{safe}_manifest.json",
        "failure_log": args.out_root / "failures" / f"{safe}_failures.jsonl",
    }


def run_parallel_replay(args: argparse.Namespace, trace_paths: list[Path], width_info: dict[str, Any]) -> dict[str, Any]:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    width = int(width_info["width"])
    started_at = time.time()
    rows: list[dict[str, Any]] = []
    paths = _replay_artifact_paths(args)
    manifest_jsonl = paths["manifest_jsonl"]
    failure_log = paths["failure_log"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "failures").mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "trace_path": str(path),
            "out_root": str(args.out_root),
            "cwd": str(args.cwd),
            "mlen": args.mlen,
            "blen": args.blen,
            "emu_threads": args.emu_threads,
            "stage_profile": args.stage_profile,
            "skip_existing": args.skip_existing,
            "prune_success_artifacts": args.prune_success_artifacts,
        }
        for path in trace_paths
    ]
    progress_path = paths["progress"]
    _atomic_json(
        progress_path,
        _progress_payload(
            args=args,
            run_id=run_id,
            total=len(tasks),
            started_at=started_at,
            rows=rows,
            running=min(width, len(tasks)),
            width=width,
            manifest_jsonl=manifest_jsonl,
            failure_log=failure_log,
        ),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as executor:
        futures = [executor.submit(_worker_run_trace, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            row["run_id"] = run_id
            rows.append(row)
            _append_jsonl(manifest_jsonl, row)
            if row.get("status") == "failed":
                _append_jsonl(failure_log, row)
            running = sum(1 for item in futures if item.running())
            _atomic_json(
                progress_path,
                _progress_payload(
                    args=args,
                    run_id=run_id,
                    total=len(tasks),
                    started_at=started_at,
                    rows=rows,
                    running=running,
                    width=width,
                    manifest_jsonl=manifest_jsonl,
                    failure_log=failure_log,
                ),
            )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": time.time(),
        "orchestrator_wall_seconds": time.time() - started_at,
        "trace_count": len(trace_paths),
        "width": width,
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in rows)),
        "runs": rows,
    }
    write_json(paths["manifest_json"], manifest)
    return manifest


def _sum_stages(profile: dict[str, Any], names: set[str]) -> int:
    stages = profile.get("stages", {})
    if not isinstance(stages, dict):
        return 0
    return sum(int(stages.get(name, {}).get("wall_cycles") or 0) for name in names)


def export_selected(args: argparse.Namespace, selection: dict[str, Any]) -> dict[str, Any]:
    selected: set[tuple[str, str]] = set()
    for benchmark in args.benchmarks:
        selected.update((benchmark, sample_id) for sample_id in _sample_ids(selection, benchmark))
    export_args = argparse.Namespace(
        root=args.out_root / "trace_replay",
        out_prefix=args.out_root / "_p3_rev_unfiltered",
        out_json=args.out_root / "_p3_rev_unfiltered_timing_summary.json",
    )
    payload = export_pilot_results.export(export_args)
    runs = [row for row in payload["runs"] if (row["benchmark"], str(row["sample_id"])) in selected]
    trace_ids = {row["trace_id"] for row in runs}
    stage_stack = [row for row in payload["stage_stack"] if row["trace_id"] in trace_ids]
    routing_tax = [row for row in payload["routing_tax"] if row["trace_id"] in trace_ids]
    bytes_validation = [row for row in payload["bytes_validation"] if row["trace_id"] in trace_ids]
    summary_rows = _summary_rows(runs, routing_tax, bytes_validation)
    prefix = args.out_root / "p3_rev_emulation"
    write_csv(prefix.with_name(prefix.name + "_timing.csv"), runs)
    write_csv(prefix.with_name(prefix.name + "_stage_stack.csv"), stage_stack)
    write_csv(prefix.with_name(prefix.name + "_routing_tax.csv"), routing_tax)
    write_csv(prefix.with_name(prefix.name + "_bytes_validation.csv"), bytes_validation)
    write_csv(prefix.with_name(prefix.name + "_distribution_stats.csv"), summary_rows)
    filtered = {
        "schema_version": 1,
        "root": str(args.out_root / "trace_replay"),
        "selected_samples": str(args.selected_out),
        "measurement_note": MEASUREMENT_NOTE,
        "runs": runs,
        "stage_stack": stage_stack,
        "routing_tax": routing_tax,
        "bytes_validation": bytes_validation,
        "distribution_stats": summary_rows,
        "unfiltered_runs": len(payload["runs"]),
    }
    write_json(args.out_root / "p3_rev_emulation_timing_summary.json", filtered)
    return filtered


def _summary_rows(
    runs: list[dict[str, Any]],
    routing_tax: list[dict[str, Any]],
    bytes_validation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tax_by_trace = {row["trace_id"]: row for row in routing_tax}
    bytes_by_trace = {row["trace_id"]: row for row in bytes_validation}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        benchmark = str(row.get("benchmark", ""))
        category = str(row.get("category") or "uncategorized")
        groups[(benchmark, "__all__")].append(row)
        groups[(benchmark, category)].append(row)
    out = []
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
        out.append(
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
    return out


def _line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh)


def _failure_counts(args: argparse.Namespace) -> dict[str, Any]:
    paths: dict[str, Path] = {}
    for path in sorted((args.out_root / "failures").glob("routing_failures*.jsonl")):
        paths[path.name] = path
    for path in [
        args.out_root / "failures" / "replay_failures.jsonl",
        args.out_root / "failures" / "p3_rev_replay_failures.jsonl",
        _replay_artifact_paths(args)["failure_log"],
    ]:
        paths[path.name] = path
    counts = {name: _line_count(path) for name, path in sorted(paths.items())}
    return {
        "total_existing_failures": sum(count for count in counts.values() if count is not None),
        "by_file": counts,
    }


def _report_artifacts(args: argparse.Namespace) -> dict[str, str]:
    prefix = args.out_root / "p3_rev_emulation"
    replay_paths = _replay_artifact_paths(args)
    return {
        "selected_samples": str(args.selected_out),
        "timing_csv": str(prefix.with_name(prefix.name + "_timing.csv")),
        "stage_stack_csv": str(prefix.with_name(prefix.name + "_stage_stack.csv")),
        "routing_tax_csv": str(prefix.with_name(prefix.name + "_routing_tax.csv")),
        "bytes_validation_csv": str(prefix.with_name(prefix.name + "_bytes_validation.csv")),
        "distribution_stats_csv": str(prefix.with_name(prefix.name + "_distribution_stats.csv")),
        "timing_summary_json": str(args.out_root / "p3_rev_emulation_timing_summary.json"),
        "replay_progress_json": str(replay_paths["progress"]),
        "replay_manifest_jsonl": str(replay_paths["manifest_jsonl"]),
        "replay_manifest_json": str(replay_paths["manifest_json"]),
        "replay_failure_log": str(replay_paths["failure_log"]),
        "report_json": str(args.out_root / f"{args.report_name}.json"),
        "report_md": str(args.out_root / f"{args.report_name}.md"),
    }


def _progress_summary(progress: dict[str, Any] | None) -> dict[str, Any] | None:
    if not progress:
        return None
    route_progress = {}
    for benchmark, row in (progress.get("route_progress") or {}).items():
        route_progress[benchmark] = {
            "selected_samples": row.get("selected_samples"),
            "complete_route_samples": row.get("complete_route_samples"),
            "route_rows": row.get("route_rows"),
        }
    return {
        "selected_samples": progress.get("selected_samples"),
        "expected_representative_runs": progress.get("expected_representative_runs"),
        "selected_trace_files": progress.get("selected_trace_files"),
        "replay_complete_runs": progress.get("replay_complete_runs"),
        "exported_selected_runs": progress.get("exported_selected_runs"),
        "failure_counts": progress.get("failure_counts"),
        "route_progress": route_progress,
    }


def write_report(
    args: argparse.Namespace,
    selection: dict[str, Any],
    trace_paths: list[Path],
    rss: dict[str, Any] | None,
    width: dict[str, Any] | None,
    replay_manifest: dict[str, Any] | None,
    export_payload: dict[str, Any] | None,
    determinism: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_counts = {
        benchmark: len(_sample_ids(selection, benchmark))
        for benchmark in args.benchmarks
    }
    expected_runs = sum(selected_counts.values()) * len(args.layers)
    exported_runs = len((export_payload or {}).get("runs", []))
    functional_counts = Counter(str(row.get("functional_gate")) for row in (export_payload or {}).get("runs", []))
    cycle_counts = Counter(str(row.get("cycle_accounting_status")) for row in (export_payload or {}).get("runs", []))
    byte_rows = (export_payload or {}).get("bytes_validation", [])
    read_match = Counter(str(row.get("physical_read_matches")) for row in byte_rows)
    write_match = Counter(str(row.get("physical_written_matches")) for row in byte_rows)
    replay_counts = dict(Counter(str(row.get("status", "unknown")) for row in (replay_manifest or {}).get("runs", [])))
    replay_progress = None
    progress_path = _replay_artifact_paths(args)["progress"]
    if progress_path.exists():
        try:
            replay_progress = load_json(progress_path)
        except Exception:
            replay_progress = None
    if not replay_counts and replay_progress:
        replay_counts = dict(replay_progress.get("status_counts") or {})
    report = {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "selected_samples": str(args.selected_out),
        "selected_counts": selected_counts,
        "expected_representative_runs": expected_runs,
        "route_trace_count": len(trace_paths),
        "exported_runs": exported_runs,
        "rss_probe": rss,
        "parallel_width": width,
        "determinism": determinism,
        "progress_snapshot": progress,
        "progress_summary": _progress_summary(progress),
        "replay_progress": replay_progress,
        "replay_status_counts_latest_run": replay_counts,
        "functional_gate_counts": dict(functional_counts),
        "cycle_accounting_counts": dict(cycle_counts),
        "physical_read_match_counts": dict(read_match),
        "physical_written_match_counts": dict(write_match),
        "failure_counts": _failure_counts(args),
        "artifacts": _report_artifacts(args),
        "measurement_note": MEASUREMENT_NOTE,
        "known_limits": [
            "P3-rev reuses P2 fixed-route replay semantics.",
            "Numbers are self-consistent upper bounds pending RTL calibration.",
            "Replay is decode tok1 only.",
            "Route replay uses zero-activation timing shape, as in P2.",
            "Routing tax is stage-derived and excludes device router GEMM/top-k.",
        ],
    }
    write_json(args.out_root / f"{args.report_name}.json", report)
    _write_report_md(args.out_root / f"{args.report_name}.md", report, selection)
    return report


def _write_report_md(path: Path, report: dict[str, Any], selection: dict[str, Any]) -> None:
    width = report.get("parallel_width") or {}
    rss = report.get("rss_probe") or {}
    lines = [
        "# Window 1 P3-rev Representative Timing Report",
        "",
        f"- measurement note: {report['measurement_note']}",
        f"- selected samples: `{report['selected_samples']}`",
        f"- selected counts: `{report['selected_counts']}`",
        f"- expected representative runs: `{report['expected_representative_runs']}`",
        f"- route trace files currently available: `{report['route_trace_count']}`",
        f"- exported selected runs: `{report['exported_runs']}`",
        f"- progress summary: `{report.get('progress_summary')}`",
        "",
        "## Subset",
        "",
    ]
    for benchmark, bench in selection.get("benchmarks", {}).items():
        lines.append(
            f"- {benchmark}: selected `{bench.get('selected_count')}` / available `{bench.get('available')}`, "
            f"method `{bench.get('selection_method')}`"
        )
    lines.extend(
        [
            "",
            "## Parallel Replay",
            "",
            f"- RSS probe trace: `{rss.get('trace_id')}`",
            f"- RSS probe max RSS: `{_fmt_bytes(int(rss.get('max_rss_kb') or 0) * 1024 if rss else None)}`",
            f"- RSS probe wall time: `{_fmt_seconds(rss.get('host_seconds'))}`",
            f"- selected width: `{width.get('width')}`",
            f"- width reason: `{width.get('reason')}`",
            f"- latest replay status counts: `{report['replay_status_counts_latest_run']}`",
            f"- replay completed/running/remaining: "
            f"`{(report.get('replay_progress') or {}).get('completed_events')}` / "
            f"`{(report.get('replay_progress') or {}).get('running')}` / "
            f"`{(report.get('replay_progress') or {}).get('remaining_events')}`",
            f"- replay progress file: `{(report.get('artifacts') or {}).get('replay_progress_json')}`",
            f"- determinism status: `{(report.get('determinism') or {}).get('status')}`",
            f"- determinism trace count: `{(report.get('determinism') or {}).get('trace_count')}`",
            "",
            "## Gates",
            "",
            f"- functional gate counts: `{report['functional_gate_counts']}`",
            f"- cycle accounting counts: `{report['cycle_accounting_counts']}`",
            f"- physical read match counts: `{report['physical_read_match_counts']}`",
            f"- physical written match counts: `{report['physical_written_match_counts']}`",
            f"- failure counts: `{report['failure_counts']}`",
            "",
            "## Artifacts",
            "",
            f"- timing CSV: `{(report.get('artifacts') or {}).get('timing_csv')}`",
            f"- stage stack CSV: `{(report.get('artifacts') or {}).get('stage_stack_csv')}`",
            f"- routing tax CSV: `{(report.get('artifacts') or {}).get('routing_tax_csv')}`",
            f"- bytes validation CSV: `{(report.get('artifacts') or {}).get('bytes_validation_csv')}`",
            f"- distribution stats CSV: `{(report.get('artifacts') or {}).get('distribution_stats_csv')}`",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["known_limits"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_stage_status(args: argparse.Namespace, stage: str, status: str, **extra: Any) -> None:
    if getattr(args, "no_stage_status", False):
        return
    payload = {
        "schema_version": 1,
        "pid": os.getpid(),
        "updated_at_unix": time.time(),
        "stage": stage,
        "status": status,
        **extra,
    }
    _atomic_json(args.out_root / "p3_rev_stage_status.json", payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_paths()
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "failures").mkdir(parents=True, exist_ok=True)
    _write_stage_status(args, "start", "running", actions=sorted(args.actions))
    selection = _load_selection(args)
    _write_stage_status(
        args,
        "select",
        "completed",
        selected_counts={benchmark: len(_sample_ids(selection, benchmark)) for benchmark in args.benchmarks},
    )
    routing_summaries = None
    if "route" in args.actions:
        _write_stage_status(args, "route", "running")
        routing_summaries = run_route_generation(args, selection)
        write_json(args.out_root / "p3_rev_routing_summaries.json", {"summaries": routing_summaries})
        _write_stage_status(args, "route", "completed", summaries=routing_summaries)
    if "build" in args.actions:
        _write_stage_status(args, "build", "running")
        build_selected_traces(args, selection)
        _write_stage_status(args, "build", "completed")
    progress = progress_snapshot(args, selection)
    if "progress" in args.actions:
        _write_stage_status(args, "progress", "completed", progress=progress)
    trace_paths = selected_trace_paths(args, selection)
    replay_trace_paths = _filtered_trace_paths_by_id(trace_paths, args.replay_trace_id_file)
    rss = None
    if "rss" in args.actions:
        _write_stage_status(args, "rss", "running", selected_trace_count=len(trace_paths))
        rss = measure_rss(args, trace_paths)
        _write_stage_status(args, "rss", "completed", rss_probe=rss)
    elif (args.out_root / "p3_rev_rss_probe.json").exists():
        rss = load_json(args.out_root / "p3_rev_rss_probe.json")
    width = None
    if "width" in args.actions or "replay" in args.actions:
        _write_stage_status(args, "width", "running")
        width = choose_width(args, rss)
        _write_stage_status(args, "width", "completed", width=width)
    elif (args.out_root / "p3_rev_parallel_width.json").exists():
        width = load_json(args.out_root / "p3_rev_parallel_width.json")
    determinism = None
    if "determinism" in args.actions:
        if width is None:
            width = choose_width(args, rss)
        _write_stage_status(
            args,
            "determinism",
            "running",
            selected_trace_count=len(trace_paths),
            determinism_samples=args.determinism_samples,
        )
        determinism = run_determinism_gate(args, trace_paths, width)
        _write_stage_status(args, "determinism", "completed", determinism=determinism)
    elif (args.out_root / "p3_rev_determinism.json").exists():
        determinism = load_json(args.out_root / "p3_rev_determinism.json")
    replay_manifest = None
    if "replay" in args.actions:
        if width is None:
            width = choose_width(args, rss)
        _write_stage_status(args, "replay", "running", selected_trace_count=len(replay_trace_paths), width=width)
        replay_manifest = run_parallel_replay(args, replay_trace_paths, width)
        _write_stage_status(args, "replay", "completed", replay_manifest=replay_manifest)
    elif _replay_artifact_paths(args)["manifest_json"].exists():
        replay_manifest = load_json(_replay_artifact_paths(args)["manifest_json"])
    elif args.replay_run_name == "p3_rev_parallel_replay" and (args.out_root / "p3_rev_parallel_replay_manifest.json").exists():
        replay_manifest = load_json(args.out_root / "p3_rev_parallel_replay_manifest.json")
    export_payload = None
    if "export" in args.actions:
        _write_stage_status(args, "export", "running")
        export_payload = export_selected(args, selection)
        _write_stage_status(args, "export", "completed", exported_runs=len(export_payload.get("runs", [])))
    elif (args.out_root / "p3_rev_emulation_timing_summary.json").exists():
        export_payload = load_json(args.out_root / "p3_rev_emulation_timing_summary.json")
    _write_stage_status(args, "report", "running")
    progress = progress_snapshot(args, selection)
    report = write_report(args, selection, trace_paths, rss, width, replay_manifest, export_payload, determinism, progress)
    _write_stage_status(args, "done", "completed", report=str(args.out_root / f"{args.report_name}.md"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--selected-out", type=Path)
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), default="qwen3")
    parser.add_argument("--benchmarks", type=_parse_csv_strings, default=_parse_csv_strings("bfcl_v3,gpqa_diamond"))
    parser.add_argument("--bfcl-target", type=int, default=300)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--layers", type=_parse_csv_ints, default=DEFAULT_LAYERS)
    parser.add_argument("--write-phases", choices=("prefill", "decode", "both"), default="decode")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument(
        "--actions",
        type=lambda value: set(_parse_csv_strings(value)),
        default={"select", "route", "build", "rss", "width", "determinism", "replay", "export", "report"},
        help="Comma-separated actions: select,route,build,progress,rss,width,determinism,replay,export,report",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--mlen", type=int, default=128)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--prune-success-artifacts", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--max-width", type=int)
    parser.add_argument("--memory-budget-fraction", type=float, default=0.70)
    parser.add_argument("--determinism-samples", type=int, default=20)
    parser.add_argument("--determinism-width", type=int)
    parser.add_argument("--no-stage-status", action="store_true")
    parser.add_argument("--routing-log-suffix", default="")
    parser.add_argument("--routing-output-suffix", default="")
    parser.add_argument("--sample-shard-count", type=int)
    parser.add_argument("--sample-shard-index", type=int)
    parser.add_argument(
        "--replay-trace-id-file",
        type=Path,
        help="Replay only trace ids listed one per line; defaults to all selected traces.",
    )
    parser.add_argument(
        "--replay-run-name",
        default="p3_rev_parallel_replay",
        help="Artifact namespace for replay progress/manifest files.",
    )
    parser.add_argument("--report-name", default="p3_rev_report")
    args = parser.parse_args()
    if args.selected_out is None:
        args.selected_out = args.out_root / "p3_rev_selected_samples.json"
    args.layers = args.layers or DEFAULT_LAYERS
    unknown_actions = set(args.actions) - {
        "select",
        "route",
        "build",
        "progress",
        "rss",
        "width",
        "determinism",
        "replay",
        "export",
        "report",
    }
    if unknown_actions:
        raise ValueError(f"unknown actions: {sorted(unknown_actions)}")
    unknown_benchmarks = [benchmark for benchmark in args.benchmarks if benchmark not in INPUT_FILES]
    if unknown_benchmarks:
        raise ValueError(f"unknown benchmarks: {unknown_benchmarks}")
    report = run(args)
    print(f"wrote {args.out_root / (args.report_name + '.md')}")
    print(
        "selected="
        f"{report['selected_counts']} "
        f"traces={report['route_trace_count']} "
        f"exported={report['exported_runs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
