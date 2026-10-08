"""Read-only completion audit for the two full sensitivity campaigns."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter

from .common import ROOT, metadata, sha, write_json


def audit(stage):
    synthetic = stage == "synthetic"
    directory = ROOT / ("E4/synthetic" if synthetic else "E5/sobol")
    csv_name = "reverse_search.csv" if synthetic else "sobol_samples.csv"
    index_name = "point_index" if synthetic else "sample_index"
    expected = 1440 if synthetic else 1792
    with (directory / csv_name).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected
    assert [int(r[index_name]) for r in rows] == list(range(expected))
    assert all(r["exact_repeat_identical"] == "True" for r in rows)
    manifest = []
    calls = points = checked_witnesses = closed = 0
    statuses = Counter()
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
        assert len(value["development_window_ids"]) == (1 if synthetic else 18)
        if synthetic:
            assert value["development_window_ids"] == [r["window_id"]]
            assert value["parameters"]["credits"] == 520
        else:
            assert value["parameters"]["credits"] == int(r["credits"])
        expected_calls = 0
        for family, item in value["families"].items():
            assert item["family"] == family
            assert item["evaluated_points"] == 8
            assert item["selected"]["repeat_identical"] is True
            assert item["selected"]["score_ms"] + 1e-9 >= item["global_lb_ms"]
            assert item["declared_lattice_points"] == (
                sum(x["cardinality"] for x in item["certificate"])
                + sum(x["cardinality"] for x in item["open_regions"]))
            for witness in item["witnesses"]:
                if witness["status"] == "evaluated":
                    assert witness["repeat_identical"] is True
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
        "certificate_partition_conservation": True,
        "frozen_numerical_sources_match_start_snapshot": source_matches,
        "csv_sha256": sha(directory / csv_name),
        "certificate_manifest_sha256": sha(directory / "CERTIFICATE_MANIFEST.json"),
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
