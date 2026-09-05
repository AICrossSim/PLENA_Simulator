import argparse
import json

from transactional_emulator.testbench.moe_timing.replay.run_swe_grouping_campaign import run_campaign


def test_resume_preserves_completed_cases(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    prior = {
        "schema_version": 1,
        "scope": "test",
        "timing_truth": "test",
        "started_at_utc": "earlier",
        "workspace_root": "old",
        "manifest_root": "old",
        "cases": [{"trace_id": "already-complete"}],
        "failures": [],
        "finished_at_utc": "earlier-finish",
    }
    (output_dir / "campaign_state.json").write_text(json.dumps(prior), encoding="utf-8")

    args = argparse.Namespace(
        workspace_root=tmp_path,
        manifest_root=tmp_path,
        output_dir=output_dir,
        models=[],
        batches=[],
        resume=True,
        cleanup=True,
    )
    state = run_campaign(args)

    assert state["cases"] == prior["cases"]
    assert state["started_at_utc"] == "earlier"
    assert "last_resumed_at_utc" in state
    assert state["finished_at_utc"] != "earlier-finish"
