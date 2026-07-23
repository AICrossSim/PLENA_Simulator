from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
PLENA_ROOT = REPO_ROOT.parents[1]
COMPILER_ROOT = REPO_ROOT / "PLENA_Compiler"
TOOLS_ROOT = REPO_ROOT / "PLENA_Tools"
TESTBENCH_ROOT = REPO_ROOT / "transactional_emulator" / "testbench"
# Default outputs inside the repo (git-ignored) so a fresh checkout writes to a
# predictable, self-contained location; override the base with PLENA_OUT_ROOT to
# redirect all moe_timing outputs elsewhere (e.g. a scratch volume).
OUT_ROOT_BASE = Path(os.environ.get("PLENA_OUT_ROOT", REPO_ROOT / "outputs"))
DEFAULT_OUT_ROOT = OUT_ROOT_BASE / "replay"


def ensure_python_paths() -> None:
    for path in (REPO_ROOT, COMPILER_ROOT, TOOLS_ROOT, TESTBENCH_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["PYTHONPATH"] = ":".join(
        [str(REPO_ROOT), str(COMPILER_ROOT), str(TOOLS_ROOT), str(TESTBENCH_ROOT), os.environ.get("PYTHONPATH", "")]
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gini_from_counts(counts: list[int]) -> float:
    if not counts:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    values = sorted(float(x) for x in counts)
    n = len(values)
    weighted = sum((idx + 1) * value for idx, value in enumerate(values))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def run_result_path(build_dir: Path) -> Path | None:
    for name in ("device_routing_results.json", "gather_scatter_results.json", "real_layer0_results.json"):
        path = build_dir / name
        if path.exists():
            return path
    return None


def load_run_bundle(build_dir: Path) -> dict[str, Any]:
    stats_path = build_dir / "rust_emulator_run_stats.json"
    stage_path = build_dir / "stage_profile.json"
    result_path = run_result_path(build_dir)
    return {
        "build_dir": str(build_dir),
        "run_stats_path": str(stats_path) if stats_path.exists() else None,
        "stage_profile_path": str(stage_path) if stage_path.exists() else None,
        "result_path": str(result_path) if result_path else None,
        "run_stats": load_json(stats_path) if stats_path.exists() else {},
        "stage_profile": load_json(stage_path) if stage_path.exists() else {},
        "result": load_json(result_path) if result_path else {},
    }


def summarize_run(run_id: str, build_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = load_run_bundle(build_dir)
    stats = bundle["run_stats"]
    profile = bundle["stage_profile"]
    result = bundle["result"]
    # full_vram_gate is the real numerical gate (routed_moe gather/scatter tests);
    # zero_input_smoke_gate is the qwen3 replay's timing-only shape smoke.
    gate = (
        result.get("device_routing_gate")
        or result.get("full_vram_gate")
        or result.get("zero_input_smoke_gate")
        or result.get("hbm_store_gate")
        or {}
    )
    row = {
        "run_id": run_id,
        "build_dir": str(build_dir),
        "stage": result.get("stage"),
        "sim_latency_cycles": stats.get("sim_latency_cycles"),
        "hbm_bytes_read": stats.get("hbm_bytes_read"),
        "hbm_bytes_written": stats.get("hbm_bytes_written"),
        "stage_total_simulation_cycles": profile.get("total_simulation_cycles"),
        "stage_total_wall_cycles": profile.get("total_stage_wall_cycles"),
        "cycle_accounting_status": profile.get("cycle_accounting_status"),
        "physical_byte_status": profile.get("physical_byte_status"),
        "functional_gate_passed": gate.get("passed"),
        "result_path": bundle["result_path"],
        "stage_profile_path": bundle["stage_profile_path"],
        "run_stats_path": bundle["run_stats_path"],
    }
    stage_rows: list[dict[str, Any]] = []
    stages = profile.get("stages", {})
    if isinstance(stages, dict):
        for stage_name, stage_data in stages.items():
            resources = stage_data.get("resource_proxy_cycles", {})
            stage_rows.append(
                {
                    "run_id": run_id,
                    "stage": stage_name,
                    "instructions": stage_data.get("instructions"),
                    "wall_cycles": stage_data.get("wall_cycles"),
                    "cycle_fraction": stage_data.get("cycle_fraction"),
                    "physical_hbm_bytes_read": stage_data.get(
                        "physical_hbm_bytes_read", stage_data.get("hbm_bytes_read")
                    ),
                    "physical_hbm_bytes_written": stage_data.get(
                        "physical_hbm_bytes_written", stage_data.get("hbm_bytes_written")
                    ),
                    "matrix_cycles": resources.get("matrix"),
                    "vector_cycles": resources.get("vector"),
                    "scalar_cycles": resources.get("scalar"),
                    "dma_cycles": resources.get("dma"),
                    "ramulator_proxy_cycles": resources.get("ramulator_proxy"),
                    "other_cycles": resources.get("other"),
                }
            )
    return row, stage_rows
