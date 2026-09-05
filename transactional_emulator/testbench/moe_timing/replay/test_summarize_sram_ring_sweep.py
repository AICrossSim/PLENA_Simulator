from __future__ import annotations

import json
from pathlib import Path

import pytest

from transactional_emulator.testbench.moe_timing.replay.summarize_sram_ring_sweep import summarize


def _case(root: Path, depth: int, cycles: int, *, hbm_bytes: int = 4096) -> None:
    directory = root / f"depth{depth}"
    directory.mkdir()
    log = directory / "run.log"
    log.write_text(
        "Scoreboard timing model summary data_stall_picos=1000 "
        "structural_stall_picos=2000 dma_wait_picos=3000 "
        "matrix_busy_pct=10 vector_busy_pct=20 scalar_busy_pct=30 dma_busy_pct=40\n"
    )
    result = {
        "trace_id": "same-trace",
        "weight_panel_mode": "blocking" if depth == 1 else "ring",
        "panel_buffer_depth": depth,
        "mram_tile_capacity": 64,
        "functional_gate": {"passed": True, "rel_rms": 0.001},
        "hbm_weight_byte_gate": {"passed": True},
        "repeat_gate": {
            "passed": True,
            "series": {"sim_latency_cycles": [cycles, cycles, cycles]},
        },
        "run_metrics": {
            "timing_model": "scoreboard",
            "scoreboard_serialize": False,
            "sim_latency_cycles": cycles,
            "hbm_bytes_read": hbm_bytes,
            "log_path": str(log),
        },
    }
    (directory / "moe_trace_replay_results.json").write_text(json.dumps(result))


def test_fixed_capacity_sweep_selects_fastest_depth(tmp_path: Path) -> None:
    _case(tmp_path, 1, 100)
    _case(tmp_path, 2, 90)
    _case(tmp_path, 4, 80)
    rows, summary = summarize(tmp_path)
    assert [row["panel_buffer_depth"] for row in rows] == [1, 2, 4]
    assert summary["winner_depth"] == 4
    assert summary["winner_speedup"] == pytest.approx(1.25)


def test_fixed_capacity_sweep_rejects_byte_drift(tmp_path: Path) -> None:
    _case(tmp_path, 1, 100)
    _case(tmp_path, 2, 90, hbm_bytes=4160)
    with pytest.raises(ValueError, match="physical HBM bytes"):
        summarize(tmp_path)
