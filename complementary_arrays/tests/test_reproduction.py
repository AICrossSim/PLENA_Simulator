"""The pipeline reproduces the reference results (needs COMPLEMENTARY_ARRAYS_DATA, the snapshot data root, and
COMPLEMENTARY_ARRAYS_REFERENCE, reference results in the output layout: search/, final/, eval/final/)."""

from pathlib import Path

import numpy as np
import pytest

from complementary_arrays.dse import results as rs
from complementary_arrays.dse import snapshots as ss
from complementary_arrays.dse.evaluate import chip_setup
from complementary_arrays.dse.formats import ROUTES
from complementary_arrays.dse.organisations import FAMILY
from complementary_arrays.dse.select import select
from complementary_arrays.sim import workload as wl

from . import reference_arithmetic

LISTING = Path(__file__).resolve().parents[1] / "scripts" / "snapshots.sha256"


def _cells(reference):
    root = reference / "final"
    return sorted(p.name for p in root.iterdir() if all((p / f"{o}.json").exists() for o in FAMILY)) if root.exists() else []


def _parse(cell):
    model, fmt, _, P = cell.split("_")
    return model, fmt, int(P[1:])


def test_snapshot_digests(data_root, tmp_path):
    lines = [line for line in LISTING.read_text().splitlines() if (data_root / f"{line.split()[1]}.npz").exists()]
    if not lines:
        pytest.skip("no snapshot files of the listing in the data root")
    (tmp_path / "listing").write_text("\n".join(lines) + "\n")
    assert ss.check(data_root, tmp_path / "listing") == []


def test_eval_windows(data_root, reference):
    from complementary_arrays.sim.engine import counters, simulate, token_counts

    for cell in _cells(reference):
        model, fmt, _ = _parse(cell)
        for org in FAMILY:
            shapes, env = chip_setup(rs.read_json(reference / "final" / cell / f"{org}.json")["best_main"], org)
            with np.load(rs.eval_dir(reference, "final", cell) / f"s1_eval_{org}.npz") as z:
                for B in (32, 128):
                    rows = ss.windows(data_root, model, ROUTES[fmt], B, "eval")
                    provision = token_counts(np.stack([r["ids"] for r in rows]), wl.MODELS[model]["experts"])
                    for i in (0, len(rows) // 2, len(rows) - 1):
                        c = counters(simulate(rows[i]["ids"][None], shapes, wl.MODELS[model], env, threads=1,
                                              provision_tokens=provision))
                        assert c["cycles"][0] == z[f"B{B}_cycles"][i]


def test_eval_means(reference, monkeypatch):
    monkeypatch.setattr(wl, "dot", reference_arithmetic.dot)
    for cell in _cells(reference):
        for org in FAMILY:
            d = rs.eval_dir(reference, "final", cell)
            summary = rs.read_json(d / f"s1_eval_{org}.json")["batches"]
            with np.load(d / f"s1_eval_{org}.npz") as z:
                for key, x in summary.items():
                    w, c = z[f"{key}_weights"], z[f"{key}_cycles"]
                    assert wl.dot(w, c) / float(w.sum()) * len(set(z[f"{key}_layer"].tolist())) == x["step_cycles"]


def test_final_selection(reference):
    for cell in _cells(reference):
        for org in FAMILY:
            ref = rs.read_json(reference / "final" / cell / f"{org}.json")
            stages = [(s, rs.search_path(reference, s, cell, org)) for s in ref["candidates_select_main"]]
            stages = [(s, p) for s, p in stages if p.exists()]
            if ref["selected_from"] not in dict(stages):
                continue
            res = select(org, stages)
            assert res["best_main"] == ref["best_main"]
            assert all(res["candidates_select_main"][s] == ref["candidates_select_main"][s] for s, _ in stages)
