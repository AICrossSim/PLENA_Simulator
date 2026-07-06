#!/usr/bin/env python3
"""Lightweight P3-rev bridge from completed raw routing rows to route traces.

This watcher does not generate routing and does not run replay.  It only notices
when true_routing JSONL rows get ahead of built route_trace JSON files, then
delegates to run_p3_rev_subset's existing build/progress/report path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROUTE_TRACE_RE = re.compile(r"^qwen3_(bfcl_v3|gpqa_diamond)_(.*)_l(0|12|23)_decode_d1\.json$")


def _selected(out_root: Path) -> dict[str, set[str]]:
    manifest = json.loads((out_root / "p3_rev_selected_samples.json").read_text())
    return {bench: {str(row["sample_id"]) for row in obj["selected_samples"]} for bench, obj in manifest["benchmarks"].items()}


def _counts(out_root: Path, selected: dict[str, set[str]]) -> tuple[int, int, int, int]:
    built_seen: dict[str, dict[str, set[int]]] = {bench: {} for bench in selected}
    for path in (out_root / "route_traces").glob("*.json"):
        match = ROUTE_TRACE_RE.match(path.name)
        if not match:
            continue
        bench, sample_id, layer = match.group(1), match.group(2), int(match.group(3))
        if bench in selected and sample_id in selected[bench]:
            built_seen[bench].setdefault(sample_id, set()).add(layer)

    raw = set()
    for path in (out_root / "true_routing").glob("*.jsonl"):
        for line in path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            bench = str(row.get("benchmark", ""))
            sample_id = str(row.get("sample_id", ""))
            if (
                bench in selected
                and sample_id in selected[bench]
                and row.get("phase") == "decode"
                and int(row.get("decode_step", 1)) == 1
            ):
                raw.add((bench, sample_id, int(row["layer"])))

    built = {
        (bench, sample_id, layer)
        for bench, sample_layers in built_seen.items()
        for sample_id, layers in sample_layers.items()
        for layer in layers
    }
    expected = sum(len(samples) * 3 for samples in selected.values())
    return len(raw), len(built), len(raw - built), expected


def _run_incremental_build(repo: Path, out_root: Path, report_name: str) -> int:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = ":".join(
        [
            str(repo),
            str(repo / "PLENA_Compiler"),
            str(repo / "PLENA_Tools"),
            str(repo / "transactional_emulator" / "testbench"),
            env.get("PYTHONPATH", ""),
        ]
    )
    cmd = [
        os.environ.get("PYTHON", "python"),
        "-m",
        "transactional_emulator.testbench.window1_p3.run_p3_rev_subset",
        "--out-root",
        str(out_root),
        "--actions",
        "build,progress,report",
        "--stage-profile",
        "--skip-existing",
        "--keep-going",
        "--prune-success-artifacts",
        "--no-stage-status",
        "--report-name",
        report_name,
    ]
    return subprocess.run(cmd, cwd=repo, env=env, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--report-name", default="p3_rev_incremental_build_report")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    selected = _selected(args.out_root)
    poll = 0
    while True:
        poll += 1
        raw_count, built_count, raw_minus_built, expected = _counts(args.out_root, selected)
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "poll": poll,
                    "raw_selected_keys": raw_count,
                    "built_selected_keys": built_count,
                    "raw_minus_built": raw_minus_built,
                    "expected": expected,
                    "updated_at_unix": time.time(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if raw_minus_built:
            rc = _run_incremental_build(args.repo, args.out_root, args.report_name)
            print(json.dumps({"incremental_build_returncode": rc, "updated_at_unix": time.time()}), flush=True)
        if args.once or built_count >= expected:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
