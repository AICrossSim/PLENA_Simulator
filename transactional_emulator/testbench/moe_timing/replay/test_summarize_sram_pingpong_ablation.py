from __future__ import annotations

import json
from pathlib import Path

import pytest

from transactional_emulator.testbench.moe_timing.replay.summarize_sram_pingpong_ablation import (
    VARIANTS,
    summarize,
)


def _write_case(root: Path, name: str, *, cycles: int, serial_cycles: int) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    is_scoreboard = "scoreboard" in name
    serialized = "serialized" in name
    panel = "pingpong" if name.startswith("pingpong") else "blocking"
    profile = {
        "cycle_accounting_status": "profiled_time_matches_total" if not is_scoreboard or serialized else "profiled_time_does_not_match_total",
        "total_profiled_cycles": cycles if not is_scoreboard or serialized else cycles + 10,
    }
    (directory / "stage_profile.json").write_text(json.dumps(profile), encoding="utf-8")
    log = ""
    if is_scoreboard:
        log = (
            "Scoreboard timing model summary data_stall_picos=1000 "
            "structural_stall_picos=2000 dma_wait_picos=3000 "
            "matrix_busy_pct=10 vector_busy_pct=20 scalar_busy_pct=30 dma_busy_pct=40\n"
        )
    (directory / "run.log").write_text(log, encoding="utf-8")
    result = {
        "trace_id": "same-trace",
        "weight_panel_mode": panel,
        "functional_gate": {"passed": True, "rel_rms": 0.001},
        "hbm_weight_byte_gate": {"passed": True},
        "run_metrics": {
            "stage_profile_path": str(directory / "stage_profile.json"),
            "log_path": str(directory / "run.log"),
            "timing_model": "scoreboard" if is_scoreboard else "serial",
            "scoreboard_serialize": serialized,
            "sim_latency_cycles": serial_cycles if serialized else cycles,
            "hbm_bytes_read": 4096,
        },
    }
    (directory / "moe_trace_replay_results.json").write_text(json.dumps(result), encoding="utf-8")


def _fixture(root: Path) -> None:
    values = {
        "blocking_serial": (100, 100),
        "blocking_scoreboard_serialized": (100, 100),
        "blocking_scoreboard": (90, 90),
        "pingpong_serial": (98, 98),
        "pingpong_scoreboard_serialized": (98, 98),
        "pingpong_scoreboard": (85, 85),
    }
    for name in VARIANTS:
        _write_case(root, name, cycles=values[name][0], serial_cycles=values[name][1])


def test_summarize_checks_controls_and_computes_incremental_gain(tmp_path: Path) -> None:
    _fixture(tmp_path)
    rows, summary = summarize(tmp_path)
    assert len(rows) == 6
    assert summary["pingpong_incremental_speedup_at_scoreboard"] == pytest.approx(90 / 85)
    assert summary["gates"]["serialized_scoreboard_matches_serial"]


def test_summarize_rejects_byte_drift(tmp_path: Path) -> None:
    _fixture(tmp_path)
    path = tmp_path / "pingpong_scoreboard" / "moe_trace_replay_results.json"
    result = json.loads(path.read_text())
    result["run_metrics"]["hbm_bytes_read"] += 64
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="changed physical HBM bytes"):
        summarize(tmp_path)
