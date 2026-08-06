from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_dse_runner():
    path = (
        Path(__file__).resolve().parents[2]
        / "Workspace/qwen3_32b_dense_analytic/run_optuna_dse.py"
    )
    spec = importlib.util.spec_from_file_location("worker_monitoring_dse", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_worker_summary_distinguishes_spawned_and_used_workers(
    tmp_path: Path,
) -> None:
    dse = _load_dse_runner()
    path = tmp_path / "worker_resources.jsonl"
    rows = [
        {"state": "parent_spawned", "worker_id": worker_id}
        for worker_id in range(3)
    ]
    rows.extend(
        {
            "state": "complete",
            "worker_id": worker_id,
            "peak_rss_gib": 0.5,
            "mem_available_gib": 100.0,
            "ask_seconds": float(worker_id + 1),
            "evaluation_seconds": 2.0,
            "rss_recycle_requested": False,
            "memory_recycle_requested": False,
        }
        for worker_id in range(2)
    )
    rows.append(
        {
            "state": "controller_sample",
            "active_workers": 3,
            "system_cpu_utilization_pct": 30.0,
            "pool_cpu_core_equivalents": 2.5,
            "pool_worker_cpu_utilization_pct": 83.333,
            "pool_cpu_capacity_utilization_pct": 3.90625,
            "active_process_tree_rss_gib": 1.5,
            "memory_prediction_error_gib": -3.0,
        }
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = dse.summarize_worker_resources(path, requested_workers=3)

    assert summary["worker_launches"] == 3
    assert summary["workers_with_attempts"] == 2
    assert summary["workers_without_attempts"] == 1
    assert summary["spawned_worker_useful_fraction"] == 2 / 3
    assert summary["mean_ask_wall_time_seconds"] == 1.5
    assert summary["maximum_dynamic_concurrency"] == 3
    assert summary["peak_pool_cpu_core_equivalents"] == 2.5
