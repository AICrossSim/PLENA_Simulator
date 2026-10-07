from dataclasses import asdict
import json
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
