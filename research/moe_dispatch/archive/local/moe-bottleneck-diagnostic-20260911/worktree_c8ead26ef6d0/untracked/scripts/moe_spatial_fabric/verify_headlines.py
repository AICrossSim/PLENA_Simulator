#!/usr/bin/env python3
"""Numerically verify every headline selection without changing its timing."""
import argparse
import csv
import gzip
import json
from pathlib import Path

from run_study import Study, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selections = list(csv.DictReader((args.evaluation / "selection_validation.csv").open()))
    points = list(csv.DictReader((args.evaluation / "all_points.csv").open()))
    chosen = []
    for selection in selections:
        family = "main" if selection["interface"] == "packet" else "main_packed"
        for window in ["b8_full", "b32_tokens8_31"]:
            chosen.append(next(row for row in points if row["family"] == family
                and row["window"] == window and all(row[key] == selection[key]
                    for key in ["shape", "ownership", "control"])))
    # The causal control-granularity comparison has the same source bytes.
    chosen.extend(row for row in points if row["family"] == "main"
        and row["window"] == "b8_full" and row["shape"] == "1+1+1+1+1+1"
        and row["ownership"] == "pinned_expert")
    assert len(chosen) == 19 and len({row["name"] for row in chosen}) == 19
    study = Study(args.binary.resolve(), args.output.resolve())
    numerical_hashes, matches = {}, []
    for original in chosen:
        name = original["name"]
        request = json.loads((args.evaluation / "requests" / (name + ".json")).read_text())
        request["compute"].update(name="headline_numeric_" + name, verify_values=True, record_trace=False)
        result = study.run(request, "headline_numeric", original["window"])
        old = json.loads(gzip.decompress((args.evaluation / "reports" / (name + ".json.gz")).read_bytes()))
        new = json.loads(gzip.decompress((args.output / "reports" / (result["name"] + ".json.gz")).read_bytes()))
        for key in ["total_cycles", "stats", "cores", "invocation_sha256", "service_sha256"]:
            assert old[key] == new[key], (name, key)
        value_hash = sha(json.dumps([new["output_fp32_bits"], new["output_bf16_bits"]]).encode())
        prior = numerical_hashes.setdefault(original["window"], value_hash)
        assert value_hash == prior, name
        matches.append(dict(source_point=name, numerical_point=result["name"],
                            value_sha256=value_hash, cycles=result["cycles"]))
    save(args.output / "validation.json", dict(passed=True, points=len(chosen), runs=2*len(chosen),
        all_numeric=True, all_integer_reference_bit_exact=True,
        all_timing_stats_and_event_hashes_equal_shape_only=True,
        all_architectures_same_output_per_window=True,
        binary_sha256=sha(args.binary.read_bytes()),
        runner_sha256=sha(Path(__file__).read_bytes()),
        matches=matches, receipts=study.receipts))
    print(f"COMPLETE {len(chosen)} x 2 numerical headline checks", flush=True)


if __name__ == "__main__":
    main()
