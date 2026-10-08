"""Small selection/protocol checks; no campaign jobs or result writes."""
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

import pytest

from research.moe_dispatch.round2 import model as old
from research.moe_dispatch.round2.common import encode_design
from research.moe_dispatch.round2.dispatch_fix import run as campaign
from research.moe_dispatch.round2.predictors import Predictor


def selection_rows(ratio_for):
    rows = []
    for threshold, flag in campaign.CONFIGS:
        for window in ("short", "long"):
            for mode in campaign.MODES:
                for index, design in enumerate(campaign.NAMES):
                    reference = (100.0 if window == "short" else 100000.0) * (index + 1)
                    ratio = ratio_for(threshold, flag, window, mode, design)
                    rows.append({
                        "t_big": threshold,
                        "large_first": flag,
                        "window_id": window,
                        "onchip_mode": mode,
                        "design": design,
                        "old_cycles": reference,
                        "fixed_cycles": reference * ratio,
                        "ratio": ratio,
                    })
    # Window matching must use identity instead of input adjacency.
    return rows[::2] + rows[1::2]


def test_selection_pairs_windows_and_chooses_one_common_configuration():
    def ratio_for(threshold, flag, window, mode, design):
        if not flag:
            return 1.0
        if threshold == 4:
            return {
                ("short", "pipelined"): 0.5,
                ("short", "port_tight"): 0.6,
                ("long", "pipelined"): 1.2,
                ("long", "port_tight"): 1.0,
            }[window, mode]
        if threshold == 3:
            return 0.8
        if threshold == 2:
            return (0.4 if window == "short" else 1.1) if design == "B0" else 1.25
        if threshold == 6:
            return (0.45 if window == "short" else 0.95) if design == "B1" else 1.15
        return 1.1

    rows = selection_rows(ratio_for)
    selected = campaign.select(rows, ["long", "short"])
    assert selected["chosen"] == {"t_big": 4, "large_first": True}
    assert selected["development_windows"] == ["long", "short"]
    assert selected["selected_development_ratio"] == pytest.approx(math.sqrt(0.6))
    logs = next(
        item["logs"] for item in selected["per_window_log_ratios"]
        if item["t_big"] == 4 and item["large_first"]
    )
    assert logs == pytest.approx([math.log(1.2) / 2, math.log(0.5 * 0.6) / 2])
    assert len(selected["scores"]) == 10
    # Two individual designs prefer a different threshold; selection stays common.
    assert math.sqrt(0.4 * 1.1) < selected["selected_development_ratio"]
    assert math.sqrt(0.45 * 0.95) < selected["selected_development_ratio"]
    # Summed absolute latency would prefer threshold3 because long windows dominate.
    total4 = sum(r["fixed_cycles"] for r in rows if r["t_big"] == 4 and r["large_first"])
    total3 = sum(r["fixed_cycles"] for r in rows if r["t_big"] == 3 and r["large_first"])
    assert total4 > total3


def test_bootstrap_is_deterministic_and_ties_choose_false_smallest_threshold():
    rows = selection_rows(lambda *args: 1.0)
    first = campaign.select(rows, ["short", "long"])
    second = campaign.select(rows, ["short", "long"])
    assert first == second
    assert first["chosen"] == {"t_big": 2, "large_first": False}
    assert first["bootstrap_draws"] == 200
    assert first["bootstrap_seed"] == 20261008
    assert first["threshold_inactive_when_large_first_false"] is True
    counts = first["bootstrap_selection_counts"]
    assert sum(item["count"] for item in counts) == 200
    assert sum(item["fraction"] for item in counts) == pytest.approx(1.0)
    assert next(item["count"] for item in counts if item["t_big"] == 2 and not item["large_first"]) == 200
    assert all(item["count"] == 0 for item in counts if (item["t_big"], item["large_first"]) != (2, False))


def test_large_first_false_makes_threshold_inactive_for_reused_ours_sequence():
    design = old.Design((old.Core(4, 4, 512), old.Core(2, 4, 512)), flows=("OS", "OS"))
    windows = [{
        "id": f"unit-{index}", "batch": 16, "hidden": 512, "top_k": 1,
        "experts": [{
            "id": i, "Me": rows, "H": 512, "F": 128, "is_shared": i == 4,
        } for i, rows in enumerate((1, 3, 6, 2, 1))],
    } for index in range(2)]
    sequences, states = [], []
    for threshold in (2, 3, 4, 6, 8):
        predictor = Predictor("ours")
        sequence = campaign._sequence(
            windows, design, old.Parameters(), "fixed", threshold, False, predictor,
        )
        sequences.append(campaign.digest(sequence))
        states.append((predictor.correction, predictor.counts, predictor.means, predictor.calls))
    assert len(set(sequences)) == 1
    assert all(state == states[0] for state in states[1:])
    assert sum(states[0][1]) == 10


def test_single_campaign_sequence_preserves_e4_predictor_none_path():
    design = old.Design((old.Core(6, 16, 128),), flows=("WS",))
    windows = [{
        "id": "single-sequence", "batch": 8, "hidden": 512, "top_k": 1,
        "experts": [{"id": 0, "Me": 3, "H": 512, "F": 128}],
    }]
    predictor = Predictor("ours")
    result = campaign._sequence(windows, design, old.Parameters(), "fixed", 4, True, predictor)
    assert campaign.digest(result) == campaign.digest([old.simulate(windows[0], design)])
    assert predictor.calls == 0 and predictor.counts == [0, 0]


def test_frozen_manifest_preserves_e4_hardware_and_parameters_per_mode():
    manifest_path = Path(campaign.__file__).with_name("frozen_designs.json")
    assert manifest_path.is_file(), "campaign must record frozen hardware before evaluation"
    manifest = json.loads(manifest_path.read_text())
    selection = campaign.selection_file()
    with (campaign.ROOT / "results/E4/per_window.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert set(manifest["modes"]) == set(campaign.MODES)
    for mode in campaign.MODES:
        assert manifest["parameters"][mode] == asdict(old.Parameters(onchip_mode=mode))
        assert set(manifest["modes"][mode]) == set(campaign.NAMES)
        designs = campaign.designs_from_selection(selection, mode)
        for name in campaign.NAMES:
            recorded = manifest["modes"][mode][name]
            assert recorded == json.loads(json.dumps(encode_design(designs[name])))
            references = [
                json.loads(row["design"]) for row in rows
                if row["entry"] == name and row["onchip_mode"] == mode
            ]
            assert references and all(reference == recorded for reference in references)
    assert manifest["modes"]["port_tight"]["B2"]["cores"] == [
        {"pm": 3, "pn": 16, "pk": 128},
        {"pm": 3, "pn": 16, "pk": 128},
    ]
