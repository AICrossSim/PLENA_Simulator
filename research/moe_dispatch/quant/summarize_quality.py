#!/usr/bin/env python3
"""Summarize verified numerical evidence without changing policy mathematics.

Equal-error selection across the two measured rank budgets is a post-hoc quality
diagnostic. It is not a causal scheduling policy and is never labelled as one.
"""
import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path


def mean(xs):
    return math.fsum(xs) / len(xs) if xs else None


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def summarize(output):
    receipt = json.loads((output / "q3_hardware_complete_receipt.json").read_text())
    if not receipt["complete"] or receipt["coverage"]["actual_ffn_rows"] != 196608:
        raise ValueError("hardware Q3 is not exhaustively complete")
    groups = defaultdict(dict)
    with (output / "q3_hardware_bf16_actual_ffn.csv").open() as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in ("layer", "rank_lanes", "bits", "method"))
            ident = (int(row["window_start"]), int(row["uniform_rank"]), row["strategy"])
            if ident in groups[key]:
                raise ValueError("duplicate hardware Q3 row")
            groups[key][ident] = row
    result = []
    for key, rows in sorted(groups.items()):
        starts = sorted({p[0] for p in rows})
        if len(starts) != 512 or len(rows) != 4096:
            raise ValueError("incomplete layer/format Q3 population")
        for reference_rank in (16, 32):
            baseline = [rows[(s, reference_rank, "uniform")] for s in starts]
            base_error = mean([float(r["relative_error"]) for r in baseline])
            base_bytes = mean([float(r["factor_bytes"]) for r in baseline])
            for policy in ("frequency_static", "gate_weighted_budget_oracle", "gate_weighted_causal"):
                # Both budgets use the same main/factor format and actual token
                # window. A quality-selected point is explicitly an offline
                # diagnostic rather than a realizable online policy.
                qualifying_global = []
                for budget in (16, 32):
                    candidates = [rows[(s, budget, policy)] for s in starts]
                    err = mean([float(r["relative_error"]) for r in candidates])
                    byte_count = mean([float(r["factor_bytes"]) for r in candidates])
                    if err <= base_error:
                        qualifying_global.append((byte_count, budget, err))
                best_global = min(qualifying_global) if qualifying_global else None
                matched = []
                same_budget = [rows[(s, reference_rank, policy)] for s in starts]
                for s, ref in zip(starts, baseline):
                    feasible = [rows[(s, budget, policy)] for budget in (16, 32)
                                if float(rows[(s, budget, policy)]["relative_error"]) <= float(ref["relative_error"])]
                    if feasible:
                        matched.append((min(feasible, key=lambda r:float(r["factor_bytes"])), ref))
                exact = [(r, b) for r, b in zip(same_budget, baseline)
                         if float(r["factor_bytes"]) == float(b["factor_bytes"])]
                result.append({
                    "layer": key[0], "rank_lanes": key[1], "bits": key[2], "method": key[3],
                    "uniform_reference_rank": reference_rank, "policy": policy, "windows": len(starts),
                    "uniform_mean_window_error": base_error, "uniform_mean_factor_bytes": base_bytes,
                    "same_requested_budget_mean_window_error": mean([float(r["relative_error"]) for r in same_budget]),
                    "same_requested_budget_mean_factor_bytes": mean([float(r["factor_bytes"]) for r in same_budget]),
                    "exact_byte_pairs": len(exact),
                    "exact_byte_subset_error_ratio": mean([float(r["relative_error"]) for r,b in exact]) / mean([float(b["relative_error"]) for r,b in exact]) if exact else None,
                    "globally_nonregressing_measured_budget": best_global[1] if best_global else None,
                    "globally_nonregressing_mean_byte_ratio": best_global[0] / base_bytes if best_global else None,
                    "globally_nonregressing_mean_error_ratio": best_global[2] / base_error if best_global else None,
                    "window_quality_selected_feasible_count": len(matched),
                    "window_quality_selected_subset_byte_ratio": mean([float(r["factor_bytes"]) for r,b in matched]) / mean([float(b["factor_bytes"]) for r,b in matched]) if matched else None,
                    "window_quality_selected_subset_error_ratio": mean([float(r["relative_error"]) for r,b in matched]) / mean([float(b["relative_error"]) for r,b in matched]) if matched else None,
                    "scope": "mean errors of actual 16-token windows; exact-byte subsets only; equal-error budget selection is post-hoc diagnostic, not causal runtime policy or full-tensor error",
                })
    write_csv(output / "q3_equal_error_byte_diagnostic.csv", result)
    (output / "q3_equal_error_byte_diagnostic.json").write_text(json.dumps({
        "schema": "plena_v3_measured_quality_budget_diagnostic_v1", "rows": len(result),
        "source_receipt": "q3_hardware_complete_receipt.json", "reference_budgets": [16, 32],
        "method": "compare actual fixed-format points only; no interpolation, sample reduction or budget/error relabelling",
        "metric": "arithmetic mean of 512 actual 16-token relative Frobenius errors; not full 8192-row relative Frobenius",
        "limitations": ["quality-selected budget is post-hoc", "byte equality can select a correlated subset", "no HF model accuracy or perplexity"],
    }, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    while not (args.output / "q3_hardware_complete_receipt.json").exists():
        if not args.wait:
            raise FileNotFoundError("full hardware Q3 receipt is not available")
        time.sleep(15)
    rows = summarize(args.output)
    print(json.dumps({"completed": True, "diagnostic_rows": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
