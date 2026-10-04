import json

from geometry3d.compute import Core, ContextLimits, TIMING_PROFILES, projection
from geometry3d.toy import issue_oracle, teacher_table, write_toy


def test_teacher_fixed_ownership_finite8_hand_counts():
    rows = teacher_table()
    assert [r["cycles"] for r in rows] == [224, 224, 112]
    assert [r["cycles_per_expert"] for r in rows] == [[112, 112], [224, 112], [112, 112]]
    assert [r["M_waves_per_expert"] for r in rows] == [[1, 1], [2, 1], [1, 1]]
    assert [r["groups_per_expert"] for r in rows] == [[4, 4], [8, 4], [4, 4]]
    assert [r["useful_macs"] for r in rows] == [393216] * 3
    assert [r["issued_macs"] for r in rows] == [786432, 589824, 393216]
    assert [r["padding_macs"] for r in rows] == [393216, 196608, 0]
    assert [r["issues"] for r in rows] == [64, 96, 64]
    assert [r["spatial_utilization"] for r in rows] == [.5, 2 / 3, 1]
    assert rows[0]["experts"][1]["start_cycle"] == 112
    assert all(e["start_cycle"] == 0 for r in rows[1:] for e in r["experts"])
    assert all(r["oracle_verified"] and r["main_multipliers"] == 12288 for r in rows)


def test_oracle_masks_actual_m_n_k_tails_and_obeys_commit_dependency():
    core = Core(3, 7, 64)
    limits = ContextLimits(3, 252)
    timing = TIMING_PROFILES["log2_stage2"]
    oracle = issue_oracle(5, 17, 129, core, timing, limits)
    p = projection(5, 17, 129, core, timing, limits)
    assert oracle["useful_macs"] == 5 * 17 * 129
    assert oracle["issues"] == 2 * 3 * 3
    assert oracle["cycles"] == p.cycles
    assert oracle["peak_outstanding_results"] <= limits.max_records
    previous = {}
    last_useful = []
    for event in oracle["events"]:
        record = (event["m0"], event["n0"])
        if event["k0"]:
            assert event["issue"] >= previous[record]
        previous[record] = event["commit"]
        if event["k0"] == 128:
            last_useful.append(event["useful_macs"])
    assert sorted(last_useful) == [6, 9, 14, 14, 21, 21]


def test_toy_is_versioned_and_written_only_as_new_outputs(tmp_path):
    report = write_toy(tmp_path)
    assert report["schema"] == "plena_geometry3d_teacher_finite8_v1"
    assert json.loads((tmp_path / "teacher_toy_geometry3d.json").read_text()) == json.loads(json.dumps(report))
    assert (tmp_path / "teacher_toy_geometry3d.csv").read_text().count("\n") == 4
