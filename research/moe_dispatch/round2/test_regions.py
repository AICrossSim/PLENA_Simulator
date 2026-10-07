from dataclasses import asdict
import json
import math
from types import SimpleNamespace

import pytest

from . import regions
from .common import Core, Design, encode_design, read_csv
from .sensitivity import SensitivityParameters, source_sha256


def witnesses(params):
    single = Design((Core(6, 16, 128),))
    hetero = Design((Core(4, 16, 128), Core(2, 16, 128)))
    return {
        "families": {
            "single": {"design": encode_design(single), "geomean_ms": 2.0},
            "heterogeneous": {"design": encode_design(hetero), "geomean_ms": 1.0},
        },
        "proof_complete": False,
        "gap_pct": 50.0,
        "open_lb_ms": 0.5,
        "resume": {"parameters": asdict(params)},
    }


def test_sobol_worker_preserves_frontend_parameter_and_certificate(monkeypatch, tmp_path):
    captured = []

    def search(workloads, params, **kwargs):
        captured.append(params)
        return witnesses(params)

    monkeypatch.setattr(regions, "ROOT", tmp_path)
    monkeypatch.setattr(regions, "search_workloads", search)
    row = regions._sobol_job((0, [30.4, 16, 1, 256, 1], [{"id": "development_fixture"}], 2, []))
    params = captured[0]
    assert isinstance(params, SensitivityParameters)
    assert params.issue_interval == 1
    design = Design((Core(4, 16, 128), Core(2, 16, 128)))
    assert sum(params.w_bandwidth(design, c) for c in (0, 1)) == pytest.approx(4096 / 30.4)
    assert row["weight_tile_service_cycles"] == 30.4
    assert "tile_issue_cycles" not in row
    assert row["timing_model_sha256"] == source_sha256()
    certificate = json.loads((tmp_path / "results/E3/search_certificates/sobol/0000.json").read_text())
    assert certificate["resume"]["parameters"] == asdict(params)


def test_flip_slices_all_use_frontend_class_and_keep_complete_scope(monkeypatch, tmp_path):
    captured = []

    def search(workloads, params, **kwargs):
        captured.append(params)
        return witnesses(params)

    class Executor:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, function, jobs, **kwargs): return map(function, jobs)

    out = tmp_path / "results/E3"
    out.mkdir(parents=True)
    (out / "FROZEN_SELECTION.json").write_text(json.dumps({"modes": {"pipelined": {
        "single": witnesses(SensitivityParameters())["families"]["single"]}}}))
    monkeypatch.setattr(regions, "ROOT", tmp_path)
    monkeypatch.setattr(regions, "inputs", lambda: {"development": [{"id": "development_fixture"}]})
    monkeypatch.setattr(regions, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(regions, "search_workloads", search)
    regions.flip(SimpleNamespace(jobs=1, point_seconds=2))
    assert len(captured) == 105
    assert all(isinstance(p, SensitivityParameters) and p.issue_interval == 1 for p in captured)
    assert captured[0].weight_tile_service_cycles == 1
    assert captured[20].weight_tile_service_cycles == 30.4
    assert all(p.weight_tile_service_cycles == 1 for p in captured[21:])
    assert all(isinstance(p.credits, int) for p in captured)
    assert len(read_csv(out / "flip_samples.csv")) == 105
    assert len(list((out / "search_certificates/flip").glob("*.json"))) == 105
    protocol = json.loads((out / "flip_protocol.json").read_text())
    assert set(protocol["ranges"]) == {"weight_tile_service_cycles", "bank_Bpc", "dotstagecycles", "credits", "vector_scale"}
    assert protocol["timing_model_sha256"] == source_sha256()
    assert protocol["baseline_parameters"] == asdict(SensitivityParameters())


def test_calibration_fits_both_development_endpoints_and_records_diffuse_selection(monkeypatch):
    def window(name, counts):
        return {"id": name, "batch": 2, "top_k": 6,
                "experts": [{"id": i, "Me": count, "is_shared": False}
                            for i, count in enumerate(counts)]}

    dev = [window("bfcl_development", [2] * 6),
           window("gpqa_development", [1] * 12),
           window("swe_development", [2] * 6)]
    calls = []

    def fit(loss, **kwargs):
        alpha = (2.0, 10.0)[len(calls)]
        value = loss(math.log(alpha))
        calls.append({"alpha": alpha, "loss": value})
        return SimpleNamespace(x=math.log(alpha), fun=value)

    monkeypatch.setattr(regions, "minimize_scalar", fit)
    calibration, rows = regions.calibration(dev)
    assert len(calls) == 2
    assert calibration["endpoint_fits"]["concentrated_swe"]["source_ids"] == ["swe_development"]
    assert calibration["endpoint_fits"]["diffuse_development"]["source_ids"] == ["gpqa_development"]
    assert calibration["selection_protocol"]["selected_diffuse_cohort"] == "gpqa"
    assert calibration["levels"][0] == pytest.approx(10)
    assert calibration["levels"][-1] == pytest.approx(2)
    assert calibration["fitted_endpoint_order_matches_expected"]
    assert all(math.isfinite(x["loss"]) for x in calls)
    assert len(rows) == 15
    assert {row["window_id"] for row in rows} == {w["id"] for w in dev}


def test_family_delta_interval_contains_all_optima_consistent_with_certified_bounds():
    result = {"families": {
        "single": {"certified_global_lb_ms": 5, "geomean_ms": 10},
        "heterogeneous": {"certified_global_lb_ms": 3, "geomean_ms": 6},
        "homogeneous": {"certified_global_lb_ms": 4, "geomean_ms": 8},
    }}
    bounds = regions.delta_intervals(result)
    assert bounds["delta_lower_vs_single_pct"] == pytest.approx(-70)
    assert bounds["delta_upper_vs_single_pct"] == pytest.approx(20)
    assert bounds["delta_lower_vs_homo_pct"] == pytest.approx(-62.5)
    assert bounds["delta_upper_vs_homo_pct"] == pytest.approx(50)
    for hetero in (3, 4.5, 6):
        for single in (5, 7.5, 10):
            assert bounds["delta_lower_vs_single_pct"] <= 100 * (hetero / single - 1) <= bounds["delta_upper_vs_single_pct"]
        for homo in (4, 6, 8):
            assert bounds["delta_lower_vs_homo_pct"] <= 100 * (hetero / homo - 1) <= bounds["delta_upper_vs_homo_pct"]
    # A faster incumbent alone does not resolve the ordering of family optima.
    assert result["families"]["heterogeneous"]["geomean_ms"] < result["families"]["single"]["geomean_ms"]
    assert bounds["delta_upper_vs_single_pct"] > 0


def test_family_delta_interval_handles_missing_zero_bounds_and_rejects_invalid_certificate():
    result = {"families": {
        "single": {"certified_global_lb_ms": 0, "geomean_ms": 10},
        "heterogeneous": {"certified_global_lb_ms": 0, "geomean_ms": 6},
    }}
    bounds = regions.delta_intervals(result)
    assert bounds["delta_lower_vs_single_pct"] == -100
    assert bounds["delta_upper_vs_single_pct"] is None
    assert bounds["delta_lower_vs_homo_pct"] is None
    result["families"]["heterogeneous"]["certified_global_lb_ms"] = 7
    with pytest.raises(ValueError, match="exceeds incumbent"):
        regions.delta_intervals(result)
