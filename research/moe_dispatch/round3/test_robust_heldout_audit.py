"""只读审计的对抗测试；不启动数值回放。"""
from dataclasses import asdict
import copy
import gzip
import json

import numpy as np
import pytest

from research.moe_dispatch.round3 import robust_heldout_audit as a
from research.moe_dispatch.round3.common import write_csv, write_json
from research.moe_dispatch.round3.config import parameters
from research.moe_dispatch.round3.model import Core, Design
from research.moe_dispatch.round2.robust import objectives, bootstrap_choices


def test_unclosed_campaign_never_passes(tmp_path):
    result = a.audit(tmp_path)
    assert not result["complete"] and not result["passed"]
    assert "尚未闭合" in result["failures"][0]


def test_complete_marker_cannot_hide_changed_frozen_selection(tmp_path):
    out = tmp_path / "E4/robust_heldout"
    out.mkdir(parents=True)
    for name in a.REQUIRED:
        (out / name).write_text("{}")
    selected = tmp_path / "E4/selected_designs.json"
    selected.write_text("{}")
    plan = {"selected_designs_sha256": "forged"}
    write_json(out / "CANDIDATE_PLAN.json", plan)
    run = {"candidate_plan_sha256": a._sha(out / "CANDIDATE_PLAN.json"),
           "exact_repeat_all": True, "source_unchanged": True, "selected_hardware_unchanged": True,
           "unique_points": 187, "group_candidate_rows": 334, "new_points": 175,
           "reused_points": 12, "new_physical_window_passes": 47250}
    write_json(out / "RUN_RECEIPT.json", run)
    write_json(out / "COMPLETE.json", {"all_unique_points_repeated": True, "diagnostic_only": True,
               "selected_hardware_unchanged": True, "unique_points": 187,
               "receipt_sha256": a._sha(out / "RUN_RECEIPT.json")})
    result = a.audit(tmp_path)
    assert result["complete"] and not result["passed"]
    assert "冻结主硬件SHA变化" in result["failures"][0]


def _point(tmp_path):
    hardware = asdict(Design((Core(6,4,512),), flows=("WS",), label=""))
    point = {"point_id": a._point_id("pipelined", hardware), "hardware": hardware,
             "onchip_mode": "pipelined", "physical_key": a._physical_key(hardware),
             "parameters": asdict(parameters())}
    windows = [{"id": "w" + str(i), "batch": 2} for i in range(3)]
    raw = [{"window_id": w["id"], "batch": w["batch"], "evaluation": {
               "legal": True, "lb_cycles": 10., "assignment": {
                   "status": "OPTIMAL", "optimal": True, "owners": [0]},
               "milp_sched": {"cycles": 100. + i, "hbm_bytes": 4096}}}
           for i, w in enumerate(windows)]
    path = tmp_path / "raw.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(raw, f)
    rec = {"point_id": point["point_id"], "hardware": hardware, "parameters": point["parameters"],
           "exact_repeat": True, "result_digest": a._digest(raw), "repeat_digest": a._digest(raw),
           "raw_file": path.name, "raw_sha256": a._sha(path),
           "kind": "new two-pass independently solved MILP+LPT physical replay",
           "solver_checks": [{"window_id": w["id"], "status": "OPTIMAL", "allocation_optimal": True,
                              "lb_cycles": 10., "owners": [0]} for w in windows],
           "rows": [{"window_id": w["id"], "batch": w["batch"], "cycles": 100. + i,
                     "latency_ms": (100. + i) / 1e6, "hbm_bytes": 4096,
                     "physical_digest": a._digest(raw[i]["evaluation"]["milp_sched"])}
                    for i, w in enumerate(windows)]}
    return point, windows, raw, rec


@pytest.mark.parametrize("corruption", ("raw_sha", "repeat_digest", "window_order", "reported_latency"))
def test_point_audit_rejects_forged_evidence(tmp_path, corruption):
    point, windows, raw, rec = _point(tmp_path)
    assert a._check_point(tmp_path, point, rec, windows, {}) == rec["rows"]
    bad = copy.deepcopy(rec)
    if corruption == "raw_sha":
        bad["raw_sha256"] = "wrong"
    elif corruption == "repeat_digest":
        bad["repeat_digest"] = "wrong"
    elif corruption == "window_order":
        bad["solver_checks"] = bad["solver_checks"][::-1]
    else:
        bad["rows"][0]["latency_ms"] *= 2
    with pytest.raises(AssertionError):
        a._check_point(tmp_path, point, bad, windows, {})


def _tables(tmp_path):
    hardware = [asdict(Design((Core(6,4,512),), flows=(flow,), label="")) for flow in ("WS", "IS")]
    ids = [a._point_id("pipelined", h) for h in hardware]
    points = {i: {"hardware": h, "physical_key": a._physical_key(h)} for i,h in zip(ids,hardware)}
    held = [{"id": "w" + str(i), "batch": b} for i,b in enumerate((2,2,4))]
    dev = [{"id": "d" + str(i), "batch": b} for i,b in enumerate((2,2,4))]
    vals = ([1.,1.,1.], [.5,.5,2.])
    rows = {i: [{"latency_ms": ms, "hbm_bytes": 4096} for ms in vector] for i,vector in zip(ids, vals)}
    group = {"onchip_mode": "pipelined", "constraint_group": "C0", "family": "single",
             "candidate_ids": ids, "frozen_id": ids[0], "development_ms": dict(zip(ids, vals))}
    selection = {"modes": {mode: {"C0": {"B1": hardware[0]}} for mode in a.MODES}}
    ordered = sorted(ids, key=lambda i: points[i]["physical_key"])
    computed = {i: objectives(np.asarray([r["latency_ms"] for r in rows[i]]), [w["batch"] for w in held]) for i in ids}
    wins = {name: min(ids, key=lambda i: (computed[i][name], points[i]["physical_key"])) for name in a.OBJECTIVES}
    agree = len(set(wins.values())) == 1
    counts = bootstrap_choices([group["development_ms"][i] for i in ordered], [w["batch"] for w in dev])
    metrics, winners, perwindow, stability = [], [], [], []
    common = {k: group[k] for k in ("onchip_mode", "constraint_group", "family")}
    for i in ids:
        metrics.append({**common, "point_id": i, "geometry": a._geometry(points[i]["hardware"]),
                        "hardware": a._canonical(points[i]["hardware"]),
                        "heldout_geomean_ms": float(np.exp(np.log(vals[ids.index(i)]).mean())),
                        "GM_ratio": computed[i]["geomean"], "CVaR10_ratio": computed[i]["cvar10"],
                        "worst_batch_ratio": computed[i]["minimax"],
                        "batch_ratios": a._canonical(computed[i]["batch_geomeans"]),
                        "is_primary_frozen_design": i == ids[0],
                        **{"selected_by_" + name: i == wins[name] for name in a.OBJECTIVES}})
        for w,r in zip(held,rows[i]):
            perwindow.append({**common, "point_id": i, "geometry": a._geometry(points[i]["hardware"]),
                              "window_id": w["id"], "batch": w["batch"], "latency_ms": r["latency_ms"],
                              "hbm_MiB": r["hbm_bytes"] / 2**20, "ratio_vs_C0_B1": r["latency_ms"]})
        for name in a.OBJECTIVES:
            c = counts[name][ordered.index(i)]
            stability.append({**common, "point_id": i, "objective": name, "draws": 200,
                              "selected_count": c, "selected_share": c / 200,
                              "most_selected": c == max(counts[name]), "diagnostic_only": True})
    for name,i in wins.items():
        winners.append({**common, "objective": name, "point_id": i, "geometry": a._geometry(points[i]["hardware"]),
                        "objective_ratio": computed[i][name], "three_objectives_same_physical_design": agree,
                        "matches_primary_frozen_design": i == ids[0], "candidate_count": 2, "diagnostic_only": True})
    for filename,value in (("robust_objectives.csv", metrics), ("winners.csv", winners),
                            ("per_window.csv", perwindow), ("selection_stability.csv", stability)):
        write_csv(tmp_path / filename, value)
    return [group], points, rows, selection, dev, held


def test_three_metrics_bootstrap_and_full_config_winners_are_recomputed(tmp_path):
    args = _tables(tmp_path)
    audited = a._audit_tables(tmp_path, *args)
    assert audited["objective_rows"] == 2 and audited["winner_rows"] == 3
    assert audited["max_objective_error"] < 1e-12
    path = tmp_path / "winners.csv"
    winners = a._csv(path)
    winners[0]["point_id"] = "forged-winner"
    write_csv(path, winners)
    with pytest.raises(AssertionError, match="赢家不是完整物理配置最小值"):
        a._audit_tables(tmp_path, *args)


def test_cvar_is_worst_decile_not_mean_or_geomean():
    values = [1.] * 9 + [10.]
    got = a._metrics(values, [2] * 5 + [4] * 5)
    assert got["cvar10"] == 10.
    assert got["geomean"] == pytest.approx(10. ** .1)
    assert got["minimax"] == pytest.approx(10. ** .2)


@pytest.mark.parametrize("file,column,new_value", (("robust_objectives.csv", "GM_ratio", "7"),
    ("selection_stability.csv", "selected_count", "201"),
    ("per_window.csv", "hbm_MiB", "100")))
def test_numeric_tables_cannot_be_forged_even_if_file_sha_is_reissued(tmp_path, file, column, new_value):
    args = _tables(tmp_path)
    path = tmp_path / file
    rows = a._csv(path)
    rows[0][column] = new_value
    write_csv(path, rows)
    with pytest.raises(AssertionError):
        a._audit_tables(tmp_path, *args)
