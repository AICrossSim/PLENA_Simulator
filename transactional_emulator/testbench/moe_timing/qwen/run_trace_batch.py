#!/usr/bin/env python3
"""Run a batch of P2 route traces through the Qwen3 replay harness."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from transactional_emulator.testbench.moe_timing.qwen.utils import OUT_ROOT, REPO_ROOT, load_json, write_json


def _prune_success_artifacts(build_dir: Path) -> dict[str, int]:
    prune_suffixes = {".pt"}
    prune_names = {
        "hbm_for_behave_sim.bin",
        "fp_sram.bin",
        "int_sram.bin",
        "vram_preload.bin",
        "vector_result.mem",
    }
    pruned_files = 0
    pruned_bytes = 0
    for path in sorted(child for child in build_dir.iterdir() if child.is_file()):
        if path.suffix not in prune_suffixes and path.name not in prune_names:
            continue
        size = path.stat().st_size
        path.unlink()
        pruned_files += 1
        pruned_bytes += size
    return {"pruned_file_count": pruned_files, "pruned_bytes": pruned_bytes}


def _trace_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for pattern in args.trace_glob:
        paths.extend(sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:])))
    if args.limit is not None:
        paths = paths[: args.limit]
    return paths


def _prior_run_passed(result_path: Path) -> bool:
    """A results file exists even when the functional gate failed (it is written
    before the gate assertion), so `--skip-existing` must confirm the prior run
    actually passed — otherwise failed traces are silently never retried."""
    try:
        summary = load_json(result_path)
    except (OSError, ValueError):
        return False
    gate = summary.get("zero_input_smoke_gate") or {}
    return bool(gate.get("passed"))


def run(args: argparse.Namespace) -> dict:
    traces = _trace_paths(args)
    manifest = {
        "schema_version": 1,
        "started_at": time.time(),
        "trace_count": len(traces),
        "runs": [],
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    for idx, trace_path in enumerate(traces, start=1):
        trace = load_json(trace_path)
        build_dir = args.root / trace["trace_id"]
        result_path = build_dir / "qwen3_trace_replay_results.json"
        if args.skip_existing and result_path.exists() and _prior_run_passed(result_path):
            row = {
                "trace": str(trace_path),
                "trace_id": trace["trace_id"],
                "build_dir": str(build_dir),
                "status": "skipped_existing",
            }
            manifest["runs"].append(row)
            write_json(args.manifest_out, manifest)
            print(f"[{idx}/{len(traces)}] skip {trace['trace_id']}", flush=True)
            continue

        build_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay",
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
        if args.experimental_overlap_prefetch_compute:
            cmd.append("--experimental-overlap-prefetch-compute")
        if args.keep_dumps:
            cmd.append("--keep-dumps")
        print(f"[{idx}/{len(traces)}] run {trace['trace_id']}", flush=True)
        start = time.time()
        log_path = build_dir / "p2_batch_stdout.log"
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, cwd=args.cwd, text=True, stdout=log, stderr=subprocess.STDOUT)
        row = {
            "trace": str(trace_path),
            "trace_id": trace["trace_id"],
            "build_dir": str(build_dir),
            "command": cmd,
            "return_code": proc.returncode,
            "status": "passed" if proc.returncode == 0 else "failed",
            "host_batch_seconds": time.time() - start,
            "log_path": str(log_path),
        }
        if row["status"] == "passed" and args.prune_success_artifacts:
            row.update(_prune_success_artifacts(build_dir))
        manifest["runs"].append(row)
        write_json(args.manifest_out, manifest)
        print(
            f"[{idx}/{len(traces)}] {row['status']} {trace['trace_id']} ({row['host_batch_seconds']:.1f}s)",
            flush=True,
        )
        if proc.returncode != 0 and not args.keep_going:
            return manifest
    manifest["ended_at"] = time.time()
    write_json(args.manifest_out, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-glob", action="append", required=True)
    parser.add_argument("--root", type=Path, default=OUT_ROOT / "trace_replay")
    parser.add_argument("--manifest-out", type=Path, default=OUT_ROOT / "trace_replay_batch_manifest.json")
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    parser.add_argument("--mlen", type=int, default=128)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--keep-dumps", action="store_true")
    parser.add_argument("--experimental-overlap-prefetch-compute", action="store_true")
    parser.add_argument("--prune-success-artifacts", action="store_true")
    args = parser.parse_args()
    manifest = run(args)
    failed = [row for row in manifest["runs"] if row.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
