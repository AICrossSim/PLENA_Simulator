#!/usr/bin/env python3
"""Replay a route trace through the existing fixed-route MoE emulator harness."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.window1_p1.p1_utils import (
    DEFAULT_OUT_ROOT,
    ensure_python_paths,
    load_json,
    summarize_run,
    write_json,
)
from transactional_emulator.testbench.window1_p1.validate_route_trace import validate_trace


def _make_replay_reference(trace: dict[str, Any], build_dir: Path) -> Path:
    import torch

    source_path = Path(trace["artifacts"]["reference_pt"])
    reference = torch.load(source_path, map_location="cpu")
    if not isinstance(reference, dict):
        raise TypeError(f"reference artifact must be a dict-like .pt file: {source_path}")
    replay_reference = dict(reference)
    replay_reference["topk_indices"] = torch.tensor(trace["routing"]["topk_indices"], dtype=torch.long)
    replay_reference["topk_weights"] = torch.tensor(trace["routing"]["topk_weights"], dtype=torch.bfloat16)
    path = build_dir / "replay_reference.pt"
    build_dir.mkdir(parents=True, exist_ok=True)
    torch.save(replay_reference, path)
    return path


def replay_trace(args: argparse.Namespace) -> dict[str, Any]:
    ensure_python_paths()
    trace = load_json(args.trace)
    errors = validate_trace(trace)
    if errors:
        raise ValueError("Invalid route trace:\n" + "\n".join(errors))

    build_dir = args.build_dir or (DEFAULT_OUT_ROOT / "trace_replay" / trace["trace_id"])
    build_dir.mkdir(parents=True, exist_ok=True)
    replay_reference = _make_replay_reference(trace, build_dir)

    replay = trace["replay"]
    cmd = [
        sys.executable,
        "-m",
        replay["harness_module"],
        "--stage",
        replay["stage"],
        "--reference-path",
        str(replay_reference),
        "--l1-golden-path",
        trace["artifacts"]["l1_golden_pt"],
        "--build-dir",
        str(build_dir),
        "--mlen",
        str(replay["mlen"]),
        "--blen",
        str(replay["blen"]),
        "--emu-threads",
        str(replay["emu_threads"]),
    ]
    env = os.environ.copy()
    if args.stage_profile:
        env["PLENA_EMULATOR_STAGE_PROFILE"] = "1"
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[3]), env=env, text=True)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    replay_run, replay_stack = summarize_run("trace_replay", build_dir)
    direct = trace.get("direct_runs", {}).get("fixed_route", {})
    self_compare = {
        "against": "fixed_route_direct",
        "direct_cycles": direct.get("sim_latency_cycles"),
        "replay_cycles": replay_run.get("sim_latency_cycles"),
        "cycles_match": direct.get("sim_latency_cycles") == replay_run.get("sim_latency_cycles"),
        "direct_hbm_bytes_read": direct.get("hbm_bytes_read"),
        "replay_hbm_bytes_read": replay_run.get("hbm_bytes_read"),
        "hbm_read_match": direct.get("hbm_bytes_read") == replay_run.get("hbm_bytes_read"),
        "direct_hbm_bytes_written": direct.get("hbm_bytes_written"),
        "replay_hbm_bytes_written": replay_run.get("hbm_bytes_written"),
        "hbm_written_match": direct.get("hbm_bytes_written") == replay_run.get("hbm_bytes_written"),
    }
    summary = {
        "schema_version": 1,
        "trace_path": str(args.trace),
        "trace_id": trace["trace_id"],
        "build_dir": str(build_dir),
        "replay_reference": str(replay_reference),
        "command": cmd,
        "replay_run": replay_run,
        "replay_stage_stack_rows": replay_stack,
        "self_compare": self_compare,
        "passed": all(
            [
                replay_run.get("functional_gate_passed") is True,
                self_compare["cycles_match"],
                self_compare["hbm_read_match"],
                self_compare["hbm_written_match"],
            ]
        ),
    }
    write_json(build_dir / "trace_replay_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--no-stage-profile", dest="stage_profile", action="store_false")
    parser.set_defaults(stage_profile=True)
    args = parser.parse_args()
    summary = replay_trace(args)
    print(summary["self_compare"])
    if not summary["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
