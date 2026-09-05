from __future__ import annotations

import json
from pathlib import Path

from transactional_emulator.testbench.moe_timing.replay.summarize_multi_expert_panel_pool import (
    RESULT_NAME,
    collect_rows,
)


def _write_result(path: Path, *, model: str, cycles: int, slots: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "model_name": model,
        "execution_order": "expert_major",
        "hbm_weight_layout": "tile_major",
        "mram_tile_capacity": 64,
        "trace_id": f"{model}-trace",
        "hidden": 512,
        "intermediate": 256,
        "shared_intermediate": 512,
        "pair_count": 64,
        "selected_experts": [0, 1],
        "multi_expert_panel_slots": slots,
        "functional_gate": {
            "gate_kind": "nonzero_reference",
            "passed": True,
            "rel_rms": 0.002,
        },
        "repeat_gate": {"passed": True, "repeats": 3},
        "hbm_weight_byte_gate": {"passed": True},
        "run_metrics": {
            "sim_latency_cycles": cycles,
            "hbm_bytes_read": 4096,
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_collect_rows_validates_and_compares_fixed_capacity_runs(tmp_path: Path) -> None:
    sram = tmp_path / "sram"
    pool = tmp_path / "pool"
    for key, model in (("qwen", "Qwen"), ("deepseek", "DeepSeek")):
        prefix = "" if key == "qwen" else "deep_"
        _write_result(sram / f"{prefix}h512_depth1" / RESULT_NAME, model=model, cycles=100, slots=None)
        _write_result(sram / f"{prefix}h512_depth2" / RESULT_NAME, model=model, cycles=90, slots=None)
        for slots, cycles in ((2, 89), (4, 88), (8, 88)):
            _write_result(
                pool / f"{key}_h512_slots{slots}" / RESULT_NAME,
                model=model,
                cycles=cycles,
                slots=slots,
            )

    rows = collect_rows(sram, pool)
    assert len(rows) == 10
    qwen_pool = [
        row
        for row in rows
        if row["model"] == "Qwen" and row["organization"] == "multi_expert_panel_pool"
    ]
    assert [row["resident_panel_slots"] for row in qwen_pool] == [2, 4, 8]
    assert qwen_pool[0]["speedup_vs_one_blocking_panel"] == 100 / 89
    assert qwen_pool[0]["speedup_vs_one_expert_pingpong"] == 90 / 89
