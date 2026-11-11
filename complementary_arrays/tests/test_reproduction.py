"""The pipeline reproduces the reference results (needs COMPLEMENTARY_ARRAYS_DATA and COMPLEMENTARY_ARRAYS_REFERENCE)."""

import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from complementary_arrays.dse import bootstrap, export, select
from complementary_arrays.dse import results as rs
from complementary_arrays.dse.candidates import Ranking, rule_candidates
from complementary_arrays.dse.organisations import FAMILIES, ORGS, expand
from complementary_arrays.dse.search import KAPPA
from complementary_arrays.dse.shapes import AnalyticModel, legal
from complementary_arrays.sim import workload as wl

from . import reference_arithmetic

MODEL, FMT, P = "deepseek", "w8g128", 98304
CELL = rs.cell_name(MODEL, FMT, P)


def _approx_equal(mine, ref, rel=1e-12):
    if isinstance(ref, dict):
        assert set(ref) <= set(mine)
        for k in ref:
            _approx_equal(mine[k], ref[k], rel)
    elif isinstance(ref, list):
        assert len(mine) == len(ref)
        for a, b in zip(mine, ref):
            _approx_equal(a, b, rel)
    elif isinstance(ref, float):
        assert mine == pytest.approx(ref, rel=rel)
    else:
        assert mine == ref


def _link_cell(reference, tmp_path, *parts):
    for part in parts:
        dst = tmp_path.joinpath(*part.split("/"), CELL)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(reference.joinpath(*part.split("/"), CELL))


def test_candidates(data_root, reference):
    rules = rule_candidates(Ranking(AnalyticModel(data_root, MODEL, FMT), KAPPA[FMT]), P)
    for org in FAMILIES[2]:
        ref = rs.read_json(rs.search_path(reference, "top10", CELL, org))
        mine = [c for c in rules[ORGS[org].kind] if legal(c["base"], P)]
        assert len(rules[ORGS[org].kind]) == ref["n_candidates"] and rules["C"] == ref["rule_candidates"]
        assert Counter((str(c["base"]), c["rule"]) for c in mine) == Counter(
            (str(r["base"]), r["rule"]) for r in ref["probe"]
        )
        probe = {str(r["base"]): r["probe"] for r in ref["probe"]}
        assert [str(c["base"]) for c in sorted(mine, key=lambda c: probe[str(c["base"])])] == [
            str(r["base"]) for r in ref["probe"]
        ]


def test_report_windows(data_root, reference):
    from complementary_arrays.sim.engine import Environment, counters, simulate

    chip = rs.read_json(rs.final_path(reference, CELL, "het11"))["best_main"]
    env = replace(Environment.from_dict(chip["env"]), shared_array=ORGS["het11"].shared_array)
    shapes = expand(ORGS["het11"], chip["base"])
    for B in wl.BATCHES:
        rows = wl.windows(data_root, MODEL, B, "report")
        for i in (0, len(rows) // 2, len(rows) - 1):
            out = simulate(rows[i]["ids"][None], shapes, wl.MODELS[MODEL], env, threads=1)
            assert counters(out)["cycles"][0] == chip["report"][f"B{B}"]["window_cycles"][i]


def test_report_means(reference, monkeypatch):
    monkeypatch.setattr(wl, "dot", reference_arithmetic.dot)
    for org in FAMILIES[2] + FAMILIES[4]:
        chip = (
            rs.read_json(rs.final_path(reference, CELL, org))["best_main"]
            if rs.final_path(reference, CELL, org).exists()
            else None
        )
        for x in chip["report"].values() if chip else ():
            w, c = np.asarray(x["window_weights"]), np.asarray(x["window_cycles"])
            assert wl.dot(w, c) / float(w.sum()) == x["cycles"]


def test_final_selection(reference, tmp_path):
    _link_cell(reference, tmp_path, "search/top10", "search/hetopt")
    chosen = select.select(tmp_path)
    assert len(chosen) == 6
    for _, org, stage in chosen:
        ref = rs.read_json(rs.final_path(reference, CELL, org))
        mine = rs.read_json(rs.final_path(tmp_path, CELL, org))
        assert stage == ref["selected_from"] and mine["best_main"] == ref["best_main"]
        assert (
            mine["candidates_select_main"] == ref["candidates_select_main"]
            and mine["provisional"] == ref["provisional"]
        )


def test_bootstrap(data_root, reference, tmp_path):
    _link_cell(reference, tmp_path, "search/top10")
    ref = next(
        r
        for r in rs.read_json(reference / "bootstrap.json")
        if (r["model"], r["fmt"], r["P"], r["streams"], r["version"]) == (MODEL, FMT, P, 2, "top10")
    )
    _approx_equal(bootstrap.heldout(data_root, tmp_path, MODEL, FMT, P, 2, "top10"), ref["heldout"])


def test_export(reference, tmp_path):
    for part in ("final", "search/fullhom", "search/hetcap"):
        (tmp_path / part).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / part).symlink_to(reference / part)
    (tmp_path / "bootstrap.json").symlink_to(reference / "bootstrap.json")
    assert json.loads(json.dumps(export.export(tmp_path))) == json.loads((reference / "export.json").read_text())


def test_systolic_frozen_windows(data_root, reference):
    from complementary_arrays.dse.frozen import DRAIN_COUNTERS
    from complementary_arrays.sim.engine import Environment, counters, simulate, token_counts

    ref = rs.read_json(reference / "frozen" / "systolic3" / CELL / "het11.json")
    env = replace(Environment.from_dict(ref["env"]), shared_array=False, **ref["engine_fields"])
    shapes = expand(ORGS["het11"], ref["base"])
    for B in wl.BATCHES:
        rows, x = wl.windows(data_root, MODEL, B, "report"), ref["report"][f"B{B}"]
        live = np.array([r["live"] for r in rows])
        for i in (0, len(rows) - 1):
            group = np.stack([rows[j]["ids"] for j in np.flatnonzero(live == live[i])])
            c = counters(
                simulate(
                    rows[i]["ids"][None],
                    shapes,
                    wl.MODELS[MODEL],
                    env,
                    threads=1,
                    provision_tokens=token_counts(group, wl.MODELS[MODEL]["experts"]),
                )
            )
            assert c["cycles"][0] == x["window_cycles"][i]
            assert [int(c[k][0]) for k in DRAIN_COUNTERS] == x["window_sys"][i]
