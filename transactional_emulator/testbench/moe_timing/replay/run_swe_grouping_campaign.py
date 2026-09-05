#!/usr/bin/env python3
"""Run resumable pair-major versus expert-major SWE route replays.

The campaign consumes exact route slices selected by the existing grouped
manifests.  It intentionally keeps the timing workload (zero-valued weights)
separate from the nonzero functional gate: values do not affect the emulator's
instruction or HBM schedule, while avoiding hundreds of MiB of duplicate
nonzero test tensors per point.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_PREFIXES = {
    "qwen": "qwen35_swe_bench_b{batch}",
    "deepseek": "deepseek_v2_lite_swe_bench_b{batch}",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run(command: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + " ".join(command) + "\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; inspect {log_path}"
        )


def _validate_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["run_metrics"]
    if metrics["return_code"] != 0:
        raise ValueError(f"{path}: emulator return code is not zero")
    if not result["functional_gate"]["passed"]:
        raise ValueError(f"{path}: replay gate failed")
    if not result["hbm_weight_byte_gate"]["passed"]:
        raise ValueError(f"{path}: HBM byte gate failed")
    profile_path = Path(metrics["stage_profile_path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile["cycle_accounting_status"] != "profiled_time_matches_total":
        raise ValueError(f"{profile_path}: stage cycles do not close")
    if int(profile["total_simulation_cycles"]) != int(metrics["sim_latency_cycles"]):
        raise ValueError(f"{profile_path}: stage total differs from run total")
    return result


def _remove_regenerable_payloads(build_dir: Path) -> list[str]:
    """Keep provenance/results while removing deterministic bulky payloads."""

    removed: list[str] = []
    names = {
        "hbm_for_behave_sim.bin",
        "fp_sram.bin",
        "int_sram.bin",
        "vram_preload.bin",
        "golden_result.txt",
    }
    for path in build_dir.iterdir():
        if path.is_file() and (path.name in names or path.suffix == ".pt"):
            path.unlink()
            removed.append(path.name)
    return sorted(removed)


def _case_summary(result: dict[str, Any], *, removed: list[str]) -> dict[str, Any]:
    metrics = result["run_metrics"]
    trace = json.loads(Path(result["trace_path"]).read_text(encoding="utf-8"))
    return {
        "trace_id": result["trace_id"],
        "model": result["model_name"],
        "benchmark": result["benchmark"],
        "batch": result["rows"],
        "layer": result["layer"],
        "step": trace.get("provenance", {}).get("slice", {}).get("step"),
        "execution_order": result["execution_order"],
        "route_pairs": result["pair_count"],
        "active_experts": result["group_count"],
        "cycles": int(metrics["sim_latency_cycles"]),
        "physical_hbm_bytes": int(metrics["hbm_bytes_read"]),
        "functional_gate_kind": result["functional_gate"]["gate_kind"],
        "stage_cycles_closed": True,
        "hbm_bytes_match_formula": True,
        "result_path": str(Path(metrics["stats_path"]).parent / "moe_trace_replay_results.json"),
        "removed_regenerable_payloads": removed,
    }


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    workspace_root = args.workspace_root.expanduser().resolve()
    manifest_root = args.manifest_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "campaign_state.json"
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            raise ValueError(f"unsupported campaign state schema: {state.get('schema_version')}")
        if not isinstance(state.get("cases"), list) or not isinstance(state.get("failures"), list):
            raise ValueError(f"invalid campaign state: {state_path}")
        state.pop("finished_at_utc", None)
        state["last_resumed_at_utc"] = _utc_now()
        state["workspace_root"] = str(workspace_root)
        state["manifest_root"] = str(manifest_root)
    else:
        state = {
            "schema_version": 1,
            "scope": (
                "Representative exact SWE decode route slices selected by existing manifests; "
                "not the complete SWE trajectory and not all population windows."
            ),
            "timing_truth": (
                "Rust simulation cycles with Ramulator-backed 64B requests in serial opcode mode; "
                "comparative emulator timing, pending RTL absolute calibration."
            ),
            "started_at_utc": _utc_now(),
            "workspace_root": str(workspace_root),
            "manifest_root": str(manifest_root),
            "cases": [],
            "failures": [],
        }
    env = dict(os.environ)
    env["PLENA_WORKSPACE_ROOT"] = str(workspace_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(repo_root / "PLENA_Compiler"), str(repo_root / "PLENA_Tools")]
    )

    for model in args.models:
        pattern = MODEL_PREFIXES[model]
        for batch in args.batches:
            case_id = f"{model}_swe_b{batch}"
            case_dir = output_dir / case_id
            trace_path = case_dir / "trace.json"
            manifest_path = manifest_root / pattern.format(batch=batch) / "grouped_shared_manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(manifest_path)
            if not trace_path.exists():
                _run(
                    [
                        sys.executable,
                        "-m",
                        "transactional_emulator.testbench.moe_timing.replay.build_trace_from_grouped_manifest",
                        str(manifest_path),
                        "--out",
                        str(trace_path),
                        "--workspace-root",
                        str(workspace_root),
                    ],
                    cwd=repo_root,
                    log_path=case_dir / "build_trace.log",
                    env=env,
                )

            for order, leaf in (("pair_major", "pair"), ("expert_major", "grouped")):
                build_dir = case_dir / leaf
                result_path = build_dir / "moe_trace_replay_results.json"
                try:
                    if args.resume and result_path.exists():
                        result = _validate_result(result_path)
                        removed: list[str] = []
                    else:
                        command = [
                            sys.executable,
                            "-m",
                            "transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay",
                            str(trace_path),
                            "--build-dir",
                            str(build_dir),
                            "--input-mode",
                            "zeros",
                            "--weight-mode",
                            "zeros",
                            "--include-shared-expert",
                            "--execution-order",
                            order,
                            "--hbm-weight-layout",
                            "tile_major",
                            "--coalesce-hbm-bursts",
                            "--stage-profile",
                            "--timing-model",
                            "serial",
                        ]
                        _run(
                            command,
                            cwd=repo_root,
                            log_path=build_dir / "campaign.log",
                            env=env,
                        )
                        result = _validate_result(result_path)
                        removed = _remove_regenerable_payloads(build_dir) if args.cleanup else []
                    state["cases"] = [
                        row
                        for row in state["cases"]
                        if not (
                            row["model"] == result["model_name"]
                            and row["batch"] == result["rows"]
                            and row["execution_order"] == order
                        )
                    ]
                    state["cases"].append(_case_summary(result, removed=removed))
                    _write_json(state_path, state)
                except Exception as exc:
                    state["failures"].append(
                        {
                            "case": case_id,
                            "execution_order": order,
                            "error": str(exc),
                            "at_utc": _utc_now(),
                        }
                    )
                    _write_json(state_path, state)
                    raise

    state["finished_at_utc"] = _utc_now()
    _write_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_PREFIXES), default=sorted(MODEL_PREFIXES))
    parser.add_argument("--batches", nargs="+", type=int, default=[2, 4, 8, 16])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-cleanup", dest="cleanup", action="store_false")
    parser.set_defaults(cleanup=True)
    args = parser.parse_args()
    state = run_campaign(args)
    print(json.dumps({"state": str(args.output_dir / "campaign_state.json"), "cases": len(state["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
