from transactional_emulator.testbench.moe_timing.replay.summarize_swe_grouping_campaign import summarize_state


def test_summary_computes_speedup_and_byte_reduction(monkeypatch, tmp_path):
    def fake_stages(_path, mode, model, batch):
        return [{"model": model, "batch": batch, "execution_mode": mode}]

    monkeypatch.setattr(
        "transactional_emulator.testbench.moe_timing.replay.summarize_swe_grouping_campaign.read_stage_rows",
        fake_stages,
    )
    state = {
        "cases": [
            {
                "model": "M",
                "batch": 4,
                "execution_order": "pair_major",
                "trace_id": "t",
                "cycles": 200,
                "physical_hbm_bytes": 100,
                "layer": 1,
                "step": 2,
                "route_pairs": 32,
                "active_experts": 10,
                "functional_gate_kind": "shape",
                "result_path": str(tmp_path / "pair.json"),
            },
            {
                "model": "M",
                "batch": 4,
                "execution_order": "expert_major",
                "trace_id": "t",
                "cycles": 100,
                "physical_hbm_bytes": 60,
                "layer": 1,
                "step": 2,
                "route_pairs": 32,
                "active_experts": 10,
                "functional_gate_kind": "shape",
                "result_path": str(tmp_path / "group.json"),
            },
        ]
    }
    rows, stages = summarize_state(state)
    assert rows[0]["cycle_speedup"] == 2.0
    assert rows[0]["physical_hbm_byte_reduction_pct"] == 40.0
    assert len(stages) == 2
