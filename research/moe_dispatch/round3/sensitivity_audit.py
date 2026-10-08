"""Read-only completion audit for the two full sensitivity campaigns."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter

from .common import ROOT, metadata, sha, write_json, inputs
from .config import SEED

CERTIFICATE_FAMILIES = ("single", "homogeneous", "5+1", "4+2", "3+3")


def audit_flip_csv(path, rows):
    """Accept an empty observed-reversal set, but reject malformed evidence."""
    with path.open() as stream:
        reader = csv.DictReader(stream)
        flips = list(reader)
        assert reader.fieldnames and "sample_index" in reader.fieldnames
        assert "certified_reversal" in reader.fieldnames
    indices = [int(r["sample_index"]) for r in flips]
    assert len(indices) == len(set(indices)), "duplicate flip sample indices"
    assert set(indices) == {
        int(r["sample_index"]) for r in rows if float(r["delta"]) < 0
    }, "flip sample indices do not match the observed negative deltas"
    return {"flip_rows": len(flips), "flip_header_present": True,
            "zero_row_flip_csv_is_valid": True}


def audit(stage):
    synthetic = stage == "synthetic"
    directory = ROOT / ("E4/synthetic" if synthetic else "E5/sobol")
    csv_name = "reverse_search.csv" if synthetic else "sobol_samples.csv"
    index_name = "point_index" if synthetic else "sample_index"
    expected = 1440 if synthetic else 1792
    development_workloads = None if synthetic else inputs()["development"]
    development_ids = None if synthetic else [w["id"] for w in development_workloads]
    with (directory / csv_name).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected
    assert [int(r[index_name]) for r in rows] == list(range(expected))
    assert all(r["exact_repeat_identical"] == "True" for r in rows)
    flip_audit = {}
    protocol = json.loads((directory / ("SYNTHETIC_PROTOCOL.json" if synthetic
                                        else "SOBOL_PROTOCOL.json")).read_text())
    assert protocol["completed_points" if synthetic else "samples_completed"] == expected
    assert protocol["repeats"] == 2
    assert protocol["candidate_budget_per_family"] == 8
    assert protocol["node_budget_per_family"] == 32
    if not synthetic:
        assert protocol["seed"] == SEED
        flip_audit = audit_flip_csv(directory / "flip_points.csv", rows)
        with (directory / "sobol_indices.csv").open() as stream:
            indices = list(csv.DictReader(stream))
        assert [r["param"] for r in indices] == protocol["problem"]["names"]
        assert len(indices) == 5
        assert all(math.isfinite(float(r[k])) for r in indices
                   for k in ("S1", "S1_ci", "ST", "ST_ci"))
        assert all(float(r[k]) >= 0 for r in indices for k in ("S1_ci", "ST_ci"))
        assert all(r["all_searches_certified"] == str(all(
            sample["proof_complete"] == "True" for sample in rows)) for r in indices)
        import numpy as np
        from SALib.analyze import sobol as analyzer
        recomputed = analyzer.analyze(protocol["problem"],
            np.asarray([float(r["delta"]) for r in rows]),
            calc_second_order=False, seed=protocol["seed"])
        maximum_index_difference = 0.0
        for i, r in enumerate(indices):
            for column, key in (("S1", "S1"), ("S1_ci", "S1_conf"),
                                ("ST", "ST"), ("ST_ci", "ST_conf")):
                difference = abs(float(r[column]) - float(recomputed[key][i]))
                maximum_index_difference = max(maximum_index_difference, difference)
                assert difference == 0.0, (
                    "Sobol indices do not match independent CSV response replay")
        flip_audit["indices_independently_recomputed_exact"] = True
        flip_audit["maximum_recomputed_index_difference"] = maximum_index_difference
        flip_audit["indices_sha256"] = sha(directory / "sobol_indices.csv")
        flip_audit["flip_csv_sha256"] = sha(directory / "flip_points.csv")
        flip_audit["index_intervals_95pct"] = [{"param": r["param"],
            "S1": float(r["S1"]), "S1_low": float(r["S1"]) - float(r["S1_ci"]),
            "S1_high": float(r["S1"]) + float(r["S1_ci"]),
            "ST": float(r["ST"]), "ST_low": float(r["ST"]) - float(r["ST_ci"]),
            "ST_high": float(r["ST"]) + float(r["ST_ci"])} for r in indices]
    manifest = []
    calls = points = checked_witnesses = closed = 0
    statuses = Counter()
    chosen_families = Counter()
    chosen_shapes = Counter()
    chosen_h33_shapes = Counter()
    proof_a_closed = Counter()
    proof_b_closed = Counter()
    lowerbound_checks = 0
    for r in rows:
        index = int(r[index_name])
        raw_path = directory / "certificates" / f"{index:04d}.json"
        zipped = raw_path.with_suffix(".json.gz")
        if raw_path.exists():
            payload = raw_path.read_bytes()
            if zipped.exists():
                assert gzip.decompress(zipped.read_bytes()) == payload
            path = raw_path
        else:
            path = zipped
            payload = gzip.decompress(path.read_bytes())
        value = json.loads(payload)
        assert value["seed"] == SEED
        assert set(value["families"]) == set(CERTIFICATE_FAMILIES)
        assert value["proof_complete"] == all(
            item["proof_B_closed"] for item in value["families"].values())
        assert len(value["development_window_ids"]) == (1 if synthetic else 18)
        if synthetic:
            assert value["development_window_ids"] == [r["window_id"]]
            assert value["parameters"]["credits"] == 520
        else:
            assert value["development_window_ids"] == development_ids
            assert value["parameters"]["credits"] == int(r["credits"])
            for name in ("weight_tile_service_cycles", "bank_Bpc", "dotstagecycles", "vector_scale"):
                assert value["parameters"][name] == float(r[name])
            from .sensitivity import SensitivityParameters
            from .optimizer import universal_bound
            parameters = SensitivityParameters(**value["parameters"])
            window_floors = [universal_bound(w, parameters)["lb_cycles"]
                             for w in development_workloads]
        single = value["families"]["single"]["selected"]
        chosen = min((value["families"][f] for f in ("5+1", "4+2", "3+3")),
                     key=lambda item: (item["selected"]["score_ms"], item["family"]))
        upper = chosen["selected"]["score_ms"]
        assert float(r["single_ms"]) == single["score_ms"]
        assert float(r["hetero_ms"]) == upper
        assert r["hetero_family"] == chosen["family"]
        assert float(r["delta"]) == upper / single["score_ms"] - 1
        assert float(r["delta_lower"]) == min(value["families"][f]["global_lb_ms"]
            for f in ("5+1", "4+2", "3+3")) / single["score_ms"] - 1
        assert float(r["delta_upper"]) == upper / value["families"]["single"]["global_lb_ms"] - 1
        assert json.loads(r["single_design"]) == single["design"]
        assert json.loads(r["hetero_design"]) == chosen["selected"]["design"]
        cores = chosen["selected"]["design"]["cores"]
        distinct = len(cores) == 2 and cores[0] != cores[1]
        assert r["compute_shapes_distinct"] == str(distinct)
        chosen_families[chosen["family"]] += 1
        chosen_shapes["distinct" if distinct else "same"] += 1
        if chosen["family"] == "3+3":
            chosen_h33_shapes["distinct" if distinct else "same"] += 1
        expected_calls = 0
        for family, item in value["families"].items():
            assert item["family"] == family
            assert item["evaluated_points"] == 8
            assert item["selected"]["repeat_identical"] is True
            assert item["selected"]["score_ms"] + 1e-9 >= item["global_lb_ms"]
            assert item["declared_lattice_points"] == (
                sum(x["cardinality"] for x in item["certificate"])
                + sum(x["cardinality"] for x in item["open_regions"]))
            proof_a_closed[family] += int(item["proof_A_closed"])
            proof_b_closed[family] += int(item["proof_B_closed"])
            for witness in item["witnesses"]:
                if witness["status"] == "evaluated":
                    assert witness["repeat_identical"] is True
                    if not synthetic:
                        assert len(witness["latencies_ms"]) == len(window_floors)
                        for ms, floor in zip(witness["latencies_ms"], window_floors):
                            assert float(ms) * 1e6 + 1e-5 >= floor
                            lowerbound_checks += 1
                    checked_witnesses += 1
                    statuses.update(witness["solver_statuses"])
            points += item["evaluated_points"] * 2
            expected_calls += item["simulator_calls"] * 2
        assert int(r["simulator_calls"]) == expected_calls
        calls += expected_calls
        closed += int(value["proof_complete"])
        manifest.append({"index": index, "path": str(path.relative_to(ROOT)),
                         "stored_sha256": sha(path),
                         "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
                         "stored_bytes": path.stat().st_size,
                         "uncompressed_bytes": len(payload)})
    frozen = ("config.py", "model.py", "runtime.py", "optimizer.py", "search.py",
              "sensitivity.py", "native_enum.py", "native_enum.c")
    snapshot = ROOT / "archive/source_snapshots/sensitivity_start"
    source_matches = {name: sha(ROOT / name) == sha(snapshot / name) for name in frozen}
    assert all(source_matches.values())
    assert protocol["simulator_calls"] == calls
    write_json(directory / "CERTIFICATE_MANIFEST.json", manifest)
    result = metadata({"stage": stage, "expected_points": expected,
        "completed_points": len(rows), "certificate_count": len(manifest),
        "all_complete_search_repeats_identical": True,
        "all_physical_repeats_identical": True,
        "full_search_candidate_evaluations_both_repeats": points,
        "physical_simulator_calls_both_repeats": calls,
        "successful_first_search_witnesses_audited": checked_witnesses,
        "solver_status_counts_first_search": dict(statuses),
        "closed_global_proofs": closed,
        "selected_dual_family_counts": dict(chosen_families),
        "selected_dual_compute_shape_counts": dict(chosen_shapes),
        "selected_H33_compute_shape_counts": dict(chosen_h33_shapes),
        "delta_min": min(float(r["delta"]) for r in rows),
        "delta_max": max(float(r["delta"]) for r in rows),
        "response_recomputed_from_each_certificate_exact": True,
        "independent_development_window_lowerbound_checks": lowerbound_checks,
        "lowerbound_violations": 0,
        "closed_local_5pct_incumbent_certificates_by_family": dict(proof_a_closed),
        "closed_global_LPT_objective_proofs_by_family": dict(proof_b_closed),
        "proof_A_scope": "Sample-local no >=5% improvement over each family's executable incumbent; not the main-table B1/B2 joint gate.",
        "certificate_partition_conservation": True,
        "frozen_numerical_sources_match_start_snapshot": source_matches,
        "csv_sha256": sha(directory / csv_name),
        "certificate_manifest_sha256": sha(directory / "CERTIFICATE_MANIFEST.json"),
        "campaign_protocol_sha256": sha(directory / ("SYNTHETIC_PROTOCOL.json" if synthetic
                                                      else "SOBOL_PROTOCOL.json")),
        **flip_audit,
        "interpretation": "两遍完整重选与物理回放均一致；证明 B 开放时仅为已评估候选敏感性，不是全局最优架构结论。"})
    write_json(directory / "COMPLETION_AUDIT.json", result)
    print({k: result[k] for k in ("stage", "completed_points",
          "physical_simulator_calls_both_repeats", "closed_global_proofs")}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("synthetic", "sobol"), required=True)
    audit(parser.parse_args().stage)


if __name__ == "__main__":
    main()
