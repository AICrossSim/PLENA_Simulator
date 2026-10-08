"""Adversarial saved-evidence cases for the final acceptance audit."""
from __future__ import annotations
import copy
import json
import pytest

from research.moe_dispatch.round3.validate import (exact_reproduction, check_repeat_record, certificate_contract,
                       assert_floor, read_csv, check_physical_ledger, union_call_accounting,
                       indexed_campaign_rows, sensitivity_point_contract, single_regression_contract)
from research.moe_dispatch.round3.common import decode_design, frozen_designs, encode_design


def test_reproduction_rejects_changed_latency_even_if_abs_diff_forged_zero():
    row = dict(source="old", credits="256", design="B1", dispatch="ours",
               window_id="w", ms_ref="1", ms_new="1.001", abs_diff="0")
    with pytest.raises(AssertionError, match="Nonzero"):
        exact_reproduction([row])
    row["ms_new"] = "1"
    assert exact_reproduction([row]) == 1
    with pytest.raises(AssertionError, match="Duplicate"):
        exact_reproduction([row, row])


def test_repeat_receipt_rejects_warmup_divergence_when_heldout_matches():
    r = {"result_digest": "a", "repeat_digest": "a", "warmup_digest": "b",
         "repeat_warmup_digest": "c", "exact_repeat": True}
    with pytest.raises(AssertionError, match="warmup_digest"):
        check_repeat_record(r)


def test_development_reference_repeat_is_checked_independently_of_candidate():
    row = {"result_digest": "same", "repeat_digest": "same",
           "reference_digest": "EFT-first", "repeat_reference_digest": "EFT-second"}
    with pytest.raises(AssertionError, match="reference_digest"):
        check_repeat_record(row)


def test_mandatory_hbm_floor_rejects_one_percent_too_fast():
    # A window with 256 MiB unique weights at 256 B/cycle needs 1.048576 ms.
    floor = (256 * 1024**2) / 256
    with pytest.raises(AssertionError, match="below global LB"):
        assert_floor(1.04, floor, "corrupted timing")
    assert_floor(1.049, floor, "plausible timing")


def test_open_hardware_domain_is_not_closed_by_exact_assignment_leaf():
    c = {"proof_B_closed": True, "open_regions": [{"cardinality": 19}],
         "declared_lattice_points": 20,
         "certificate": [{"cardinality": 1, "status": "resolved_leaf", "allocation_optimal": True}],
         "witnesses": [], "successful_points": 0}
    with pytest.raises(AssertionError, match="open regions"):
        certificate_contract(c)
    c["proof_B_closed"] = False
    assert certificate_contract(c) == []
    c["open_regions"][0]["cardinality"] = 18
    with pytest.raises(AssertionError, match="cover declared"):
        certificate_contract(c)


def test_schema_rejects_fetch_bandwidth_in_compute_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("window_id,fetch_ms\nw,1.0\n")
    with pytest.raises(AssertionError, match="latency_ms"):
        read_csv(p, ("window_id", "latency_ms"))


def test_shared_pool_does_not_also_provide_free_private_storage():
    d = encode_design(frozen_designs("pipelined")["best_hetero"])
    d["landing_mode"] = "shared"
    d["landing_pool_bytes"] = sum(d["w_bytes"])
    with pytest.raises(ValueError):
        decode_design(d)
    d["w_bytes"] = [0, 0]
    valid = decode_design(d)
    assert valid.landing_pool_bytes == 40 * 1024


def test_search_repeat_accounting_rejects_unrun_second_campaign():
    c = {"proof_B_closed": False, "open_regions": [], "witnesses": [],
         "successful_points": 0, "simulator_calls": 120,
         "total_campaign_simulator_calls": 120}
    with pytest.raises(AssertionError, match="accounting"):
        certificate_contract(c)


def test_raw_admission_trace_rejects_more_bytes_than_installed_pool():
    d = encode_design(frozen_designs("pipelined")["best_hetero"])
    d.update(landing_mode="shared", landing_pool_bytes=sum(d["w_bytes"]), w_bytes=[0, 0])
    ledger = decode_design(d).ledger()
    p = {"current_reserved_bytes": 16384, "next_reserved_bytes": 8192,
         "lookahead_bytes": 16384, "used_bytes": 40960, "unused_bytes": 0, "pool_bytes": 40960}
    result = {"ledger": ledger, "segments": [{"pool_ledger": p}]}
    check_physical_ledger(result)
    p.update(lookahead_bytes=17408, used_bytes=41984, unused_bytes=-1024)
    with pytest.raises(AssertionError, match="admission exceeded"):
        check_physical_ledger(result)


def test_union_selections_do_not_double_count_actual_source_campaigns():
    c0 = {"total_campaign_simulator_calls": 720, "candidate_budget": 256,
          "node_budget": 2048, "evaluated_points": 10}
    c1 = {"total_campaign_simulator_calls": 648, "candidate_budget": 256,
          "node_budget": 2048, "evaluated_points": 9}
    row = {"actual_simulator_calls": 1368, "candidate_generation_budget": 512,
           "node_generation_budget": 4096, "generated_points_with_overlap": 19}
    assert union_call_accounting(row, c0, c1) == 1368
    # C0 and C1 are two selections of one executed union, not new runs.
    row["actual_simulator_calls"] *= 2
    with pytest.raises(AssertionError, match="actual source campaigns"):
        union_call_accounting(row, c0, c1)
    row["actual_simulator_calls"] //= 2
    row["generated_points_with_overlap"] = 18
    with pytest.raises(AssertionError, match="generation budget"):
        union_call_accounting(row, c0, c1)


def test_prediction_csv_archive_preserves_quoted_rows_and_exact_original(tmp_path):
    import gzip
    import hashlib
    from research.moe_dispatch.round3.validate import compressed_csv_contract
    original = b'window_id,note\r\nw0,"one\ntwo"\r\nw1,plain\r\n'
    archive = tmp_path / "tasks.csv.gz"
    archive.write_bytes(gzip.compress(original, mtime=0))
    manifest = {"archive": archive.name, "lossless": True,
                "compressed_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "uncompressed_sha256": hashlib.sha256(original).hexdigest(),
                "uncompressed_bytes": len(original), "rows": 2,
                "header": ["window_id", "note"]}
    assert compressed_csv_contract(tmp_path, manifest)["rows"] == 2
    manifest["rows"] = 3
    with pytest.raises(AssertionError, match="row count"):
        compressed_csv_contract(tmp_path, manifest)
    manifest["rows"] = 2
    manifest["uncompressed_sha256"] = "0" * 64
    with pytest.raises(AssertionError, match="archived original"):
        compressed_csv_contract(tmp_path, manifest)


def test_campaign_ids_reject_duplicate_substitution_even_when_row_count_matches():
    rows = [dict(sample_index=i, exact_repeat_identical=True) for i in range(3)]
    assert len(indexed_campaign_rows(rows, "sample_index", 3)) == 3
    bad = copy.deepcopy(rows)
    bad[2]["sample_index"] = 1
    with pytest.raises(AssertionError, match="Duplicate or out-of-range"):
        indexed_campaign_rows(bad, "sample_index", 3)
    with pytest.raises(AssertionError, match="Incomplete campaign"):
        indexed_campaign_rows([], "sample_index", 3)
    assert indexed_campaign_rows([], "sample_index", 3, partial=True) == {}
    bad = [dict(sample_index=3, exact_repeat_identical=True)]
    with pytest.raises(AssertionError, match="out-of-range"):
        indexed_campaign_rows(bad, "sample_index", 3, partial=True)


def test_partial_campaign_still_requires_whole_search_repeat_marker():
    rows = [dict(point_index=0, exact_repeat_identical=False)]
    with pytest.raises(AssertionError, match="two-run equality marker"):
        indexed_campaign_rows(rows, "point_index", 2, partial=True)


def sensitivity_fixture():
    # A saved five-family point; validator independently derives its dual
    # choice, latency ratio and call count instead of trusting the CSV.
    from research.moe_dispatch.round3.config import SEED
    params = {"credits": 520}
    families = {}
    scores = {"single": 5., "homogeneous": 4.5, "5+1": 4.3, "4+2": 4., "3+3": 4.2}
    for name, score in scores.items():
        rows = [6] if name == "single" else [3, 3] if name in ("homogeneous", "3+3") else [int(x) for x in name.split("+")]
        design = {"cores": [{"pm": m, "pn": 4, "pk": 512} for m in rows]}
        families[name] = {"family": name, "parameters": params, "proof_B_closed": False,
            "selected": {"score_ms": score, "design": design}, "global_lb_ms": 3.,
            "gap_pct": 0., "evaluated_points": 1, "simulator_calls": 2}
    data = {"families": families, "parameters": params, "seed": SEED,
            "development_window_ids": ["w"], "proof_complete": False}
    row = dict(exact_repeat_identical=True, proof_complete=False, single_ms=5., hetero_ms=4.,
               delta=-.2, delta_lower=-.4, delta_upper=4./3.-1, gap_pct=0.,
               single_design=json.dumps(families["single"]["selected"]["design"]),
               hetero_design=json.dumps(families["4+2"]["selected"]["design"]),
               compute_shapes_distinct=True, hetero_family="4+2", evaluated_points=5, simulator_calls=20)
    return data, row, params


def test_sensitivity_rejects_missing_family_wrong_parameter_and_stale_table_value():
    data, row, params = sensitivity_fixture()
    sensitivity_point_contract(data, row, params, ["w"])
    bad = copy.deepcopy(data)
    del bad["families"]["homogeneous"]
    with pytest.raises(AssertionError, match="five hardware families"):
        sensitivity_point_contract(bad, row, params, ["w"])
    with pytest.raises(AssertionError, match="planned point"):
        sensitivity_point_contract(data, row, {"credits": 256}, ["w"])
    badrow = copy.deepcopy(row)
    badrow["hetero_ms"] = 4.01
    with pytest.raises(AssertionError, match="differs from certificate"):
        sensitivity_point_contract(data, badrow, params, ["w"])
    bad = copy.deepcopy(data)
    bad["families"]["5+1"]["parameters"] = {"credits": 256}
    with pytest.raises(AssertionError, match="within sensitivity point"):
        sensitivity_point_contract(bad, row, params, ["w"])


def test_partial_orphan_certificate_checks_input_and_root_proof_without_csv():
    data, row, params = sensitivity_fixture()
    sensitivity_point_contract(data, None, params, ["w"])
    with pytest.raises(AssertionError, match="window IDs/order"):
        sensitivity_point_contract(data, None, params, ["wrong-window"])
    data["proof_complete"] = True
    with pytest.raises(AssertionError, match="root proof marker"):
        sensitivity_point_contract(data, None, params, ["w"])


def regression_fixture():
    rows = []; raw = {}
    for mode in ("pipelined", "port_tight"):
        for design in ("B0", "B1"):
            for wid in ("w0", "w1"):
                key = mode, design, wid
                rows.append(dict(onchip_mode=mode, design=design, window_id=wid,
                                 cycles_exact=True, hbm_bytes_exact=True, result_digest_exact=True))
                for dispatch in ("eft_old", "fixed"):
                    raw[key + (dispatch,)] = dict(workload=wid, cycles=100., hbm_bytes=1024,
                                                 tasks=[{"expert_id": 7, "core": 0}])
    return rows, raw


def test_single_regression_rechecks_raw_and_unique_protocols_despite_true_flags():
    rows, raw = regression_fixture()
    assert single_regression_contract(rows, raw, ["w0", "w1"]) == 8
    duplicate = copy.deepcopy(rows)
    duplicate[-1] = duplicate[-2]
    with pytest.raises(AssertionError, match="Duplicate or unexpected"):
        single_regression_contract(duplicate, raw, ["w0", "w1"])
    changed = copy.deepcopy(raw)
    changed["pipelined", "B0", "w0", "fixed"]["tasks"][0]["expert_id"] = 8
    with pytest.raises(AssertionError, match="raw fixed/EFT results differ"):
        single_regression_contract(rows, changed, ["w0", "w1"])
    with pytest.raises(AssertionError, match="Incomplete single-core"):
        single_regression_contract(rows[:-1], raw, ["w0", "w1"])
    assert single_regression_contract(rows[:-1], raw, ["w0", "w1"], partial=True) == 7
