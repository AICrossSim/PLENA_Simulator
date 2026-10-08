"""Adversarial saved-evidence cases for the final acceptance audit."""
from __future__ import annotations
import copy
import json
import pytest

from research.moe_dispatch.round3.validate import (exact_reproduction, check_repeat_record, certificate_contract,
                       assert_floor, read_csv, check_physical_ledger, union_call_accounting)
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
