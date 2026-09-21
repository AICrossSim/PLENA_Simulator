import pytest
import json
from types import SimpleNamespace
from .ltile_platform import DmaResources, memory_geometry, formal_blockers


def test_credits_require_physical_response_and_staging_storage():
    with pytest.raises(ValueError, match="response"):
        DmaResources(read_credits=64)
    with pytest.raises(ValueError, match="staging"):
        DmaResources(write_credits=64)
    assert DmaResources().buffer_and_tag_bytes == 4608


def test_capacity_cannot_change_without_memory_organization():
    c = {
        "impl": "HBM12",
        "dram": {"channel_width": 64, "org": {"count": [1, 2, 4, 4, 65536, 128], "dq": 64}, "timing": [2000]},
    }
    config = {"memory_system": {"channel_mapper": {"impl": "CacheLineInterleave"}, "controllers": [c] * 8}}
    a = memory_geometry(config)
    assert a["capacity_bytes"] == 16 * 1024**3 and a["peak_bytes_per_second"] == 128 * 10**9
    config["memory_system"]["controllers"] *= 2
    b = memory_geometry(config)
    assert (
        b["capacity_bytes"] == 2 * a["capacity_bytes"] and b["peak_bytes_per_second"] == 2 * a["peak_bytes_per_second"]
    )


def test_recurrent_gate_does_not_certify_full_model():
    reasons = formal_blockers(
        {"recurrence": {"passed": True}}, profile_sha256="x", required_bytes=18, capacity_bytes=16
    )
    assert len(reasons) == 8
    assert any("producer" in x for x in reasons)


def test_decode_entry_cannot_promote_recurrence_only_calibration(tmp_path):
    from .ltile_decode import run

    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            dict(gate_passed=True, model_contract="ltile_r3_compositional_v1", memory_backend={}, predictor_sources={})
        )
    )
    with pytest.raises(RuntimeError, match="recurrence-only"):
        run(tmp_path, tmp_path / "output", SimpleNamespace(identity={}), tmp_path, gate_path=gate)
    assert not (tmp_path / "output").exists()
