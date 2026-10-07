"""Units and isolated reruns must preserve the formal full-mode campaign."""
from types import SimpleNamespace
import csv
import sys
import pytest

from . import run
from .common import bounds, useful_macs
from .model import Core, Design, Parameters


def test_mac_floor_keeps_taskbook_units_across_issue_modes():
    w = {"id": "unit_floor", "batch": 4, "top_k": 1, "hidden": 512,
         "experts": [{"id": 0, "Me": 4, "H": 512, "F": 128, "is_shared": False}]}
    d = Design((Core(6, 4, 512),))
    expected = useful_macs(w) / 12288 / 1e6
    for mode in ("pipelined", "port_tight", "fixed_issue"):
        assert bounds(w, d, Parameters(onchip_mode=mode))["mac_floor"] == pytest.approx(expected)


def test_single_mode_cli_preserves_full_campaign_and_cold_recomputes(tmp_path, monkeypatch):
    original = tmp_path / "results/E2/micro.csv"
    original.parent.mkdir(parents=True)
    original.write_text("EXISTING_FULL_CAMPAIGN\n")
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "RESULTS_DIRECTORY", tmp_path / "results")
    monkeypatch.setattr(run, "MODES", ("pipelined", "port_tight", "fixed_issue"))
    monkeypatch.setattr(sys, "argv", ["run", "--stage", "E2micro", "--onchip-mode", "port_tight"])
    run.main()
    with (tmp_path / "results/isolated_modes/port_tight/E2/micro.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 396
    assert {r["onchip_mode"] for r in rows} == {"port_tight"}
    assert original.read_text() == "EXISTING_FULL_CAMPAIGN\n"
    # Cold second computation leaves one newly evaluated cost, not a hit.
    assert run._task_cached.cache_info().misses == 1
    assert run._task_cached.cache_info().hits == 0
