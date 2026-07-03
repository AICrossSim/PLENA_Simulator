#!/usr/bin/env python3
"""Run a batch of P2 route traces through the Qwen3 replay harness."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from transactional_emulator.testbench.window1_p2.p2_utils import OUT_ROOT, load_json, write_json


def _trace_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for pattern in args.trace_glob:
        paths.extend(sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:])))
    if args.limit is not None:
        paths = paths[: args.limit]
    return paths


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
        if args.skip_existing and result_path.exists():
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
        manifest["runs"].append(row)
        write_json(args.manifest_out, manifest)
        print(
            f"[{idx}/{len(traces)}] {row['status']} {trace['trace_id']} "
            f"({row['host_batch_seconds']:.1f}s)",
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
    parser.add_argument("--cwd", type=Path, default=Path("/scratch/shared/mcl123/plena/repos/PLENA_Simulator"))
    parser.add_argument("--mlen", type=int, default=128)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--keep-dumps", action="store_true")
    parser.add_argument("--experimental-overlap-prefetch-compute", action="store_true")
    args = parser.parse_args()
    manifest = run(args)
    failed = [row for row in manifest["runs"] if row.get("status") == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

