#!/usr/bin/env python3
"""Read-only BF16 compatibility checks; writes new E0 evidence only.

This does not run the historical mixed-format regime campaign. It replays
only the 945 BF16 v6 records and the three selected BF16/256 designs (405).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--workspace", type=Path,
                        default=Path("/scratch/shared/mcl123/plena"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty output directory; old evidence is immutable")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))
    from research.moe_dispatch.regime.campaign import load_inputs, evaluate
    from research.moe_dispatch.regime.model import Settings
    from research.moe_dispatch.geometry3d.study import cores_from
    from research.moe_dispatch.regime.search import settings_for

    v6 = args.workspace / "final_artifacts/moe_geometry3d_20261004_v6/payload/system"
    previous = args.workspace / "final_artifacts/moe_regime_20261005_v1/payload"
    captured = previous / "inputs"
    workloads = load_inputs(captured)
    heldout = workloads["heldout"]
    assert len(workloads["development"]) == 18 and len(heldout) == 135
    old_points = json.loads((v6 / "FROZEN_SELECTION.json").read_text())["points"]
    old_results = json.loads((v6 / "heldout_details.json").read_text())
    prior_points = [point for point in json.loads(
        (previous / "search/FROZEN_SELECTION.json").read_text())["points"]
        if point["weight_format"] == "BF16" and point["credits"] == 256]
    assert len(old_points) == 7 and len(prior_points) == 3

    checks = ("cycles", "hbm_bytes", "native_unique_bytes", "useful_macs",
              "issued_macs", "X_sram_bytes", "core_finish_cycles")
    receipts, all_rows = [], []
    for cohort, points in (("historical945", old_points),
                           ("previous_BF16_c256_three", prior_points)):
        rows = []
        for point in points:
            if cohort == "historical945":
                label = point["label"]
                settings = Settings(allocation=point["allocation"],
                                    flow=point["flow"],
                                    prefetch_slots=point["prefetch_slots"])
                reference = old_results[label]
            else:
                label = point["family"]
                settings = settings_for(256, "BF16", point)
                reference = json.loads((previous /
                    f"search/BF16_c256/heldout_{label}.json").read_text())
            results = evaluate(heldout, cores_from(point["geometry"]), settings,
                               point["policy"], repeats=2)
            assert len(reference) == len(results) == 135
            for now, prev in zip(results, reference):
                assert now["workload"] == prev["workload"]
                for metric in checks:
                    assert now[metric] == prev[metric], (
                        cohort, label, now["workload"], metric,
                        now[metric], prev[metric])
                rows.append({"design": f"{cohort}:{label}:{point['geometry']}",
                             "window_id": now["workload"], "batch": now["batch"],
                             "prev_ms": prev["latency_ms"],
                             "now_ms": now["latency_ms"],
                             "abs_diff": abs(now["latency_ms"] - prev["latency_ms"])})
            print(f"{cohort}: {label}: 135 exact records, two identical runs", flush=True)
        path = out / f"{cohort}_reproduce_check.csv"
        write_csv(path, rows)
        all_rows.extend(rows)
        receipts.append({"cohort": cohort, "rows": len(rows), "repeats": 2,
                         "all_exact": True, "sha256": sha(path)})
    write_csv(out / "reproduce_check.csv", all_rows)

    input_hashes = {name: sha(captured / name) for name in
        ("development.json", "mixed_development.json", "heldout.json", "mixed_heldout.json")}
    frozen = {"precision": "BF16", "assumed_core_clock_ghz": 1,
        "core_cycle_ns": 1, "hbm": {"kind": "HBM2 analytical credit service",
        "nominal_GB_per_s": 256, "request_bytes": 32, "response_cycles": 64,
        "landing_cycles": 1, "main_credits": 256,
        "main_service_cap_GB_per_s": 256*32/65,
        "sensitivity_credits": 512, "sensitivity_cap_GB_per_s": 512*32/65},
        "resources": {"multipliers": 12288, "storage_bytes": 2158592,
        "W_banks": 64, "X_banks": 24, "acc_banks": 12,
        "bytes_per_bank_cycle": 16, "vector_ops_per_cycle": 64},
        "model": {"name": "DeepSeek-V2-Lite", "H": 2048,
        "routed_F": 1408, "shared_F": 2816, "routed_topk": 6},
        "input_directory": str(captured), "input_sha256": input_hashes,
        "development_window_ids": [w["id"] for w in workloads["development"]],
        "heldout_window_ids": [w["id"] for w in heldout],
        "heldout_batch_counts": {str(b): sum(w["batch"] == b for w in heldout)
             for b in (2,4,8,16,64,96,128)},
        "origin": "Previously captured BFCL/GPQA/SWE routed populations",
        "timing_boundary": "post-router Gate/Up, SiLU/Z, Down, combine",
        "warnings": ["Not a new model inference run",
                     "Historical heldout captures have been accessed before",
                     "No qualified matching DeepSeek non-MoE per-layer timing found",
                     "Port counters overlap and are not additive wall time"]}
    write_json(out / "frozen_inputs.json", frozen)
    source_names = ("geometry3d/compute.py", "geometry3d/memory.py",
                    "geometry3d/model.py", "geometry3d/study.py",
                    "regime/model.py", "regime/resources.py",
                    "regime/metrics.py", "regime/search.py", "regime/campaign.py")
    source_base = repo / "research/moe_dispatch"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                     text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo,
                                     text=True).strip()
    receipt = {"branch": branch, "commit": commit, "exact_records": len(all_rows),
        "cohorts": receipts, "checked_metrics": list(checks),
        "source_sha256": {name: sha(source_base/name) for name in source_names},
        "script_sha256": sha(Path(__file__).resolve()), "input_sha256": input_hashes,
        "compatibility_source_sha256": {"v6_FROZEN_SELECTION": sha(v6/"FROZEN_SELECTION.json"),
            "v6_heldout_details": sha(v6/"heldout_details.json"),
            "regime_FROZEN_SELECTION": sha(previous/"search/FROZEN_SELECTION.json")},
        "scope": "BF16 exact compatibility only; not round2 new-design evidence"}
    write_json(out / "reproduction_receipt.json", receipt)
    provenance = [{"file": p.name, "sha256": sha(p), "commit": commit,
                   "scope": receipt["scope"]} for p in sorted(out.iterdir()) if p.is_file()]
    write_csv(out / "PROVENANCE.csv", provenance)
    (out/"README.md").write_text(
        "# E0 BF16 historical compatibility\n\n"
        "All 945 v6 plus 405 previous BF16/256 design records must be exact.\n"
        "Each full result is simulated twice and compared as a complete object.\n\n"
        f"Commit: `{commit}`\n\n"
        "```sh\n"
        f"{sys.executable} {Path(__file__).resolve()} --repo {repo} "
        f"--workspace {args.workspace} --out /absolute/new/empty/output\n"
        "```\n\nInput and source hashes: reproduction_receipt.json and frozen_inputs.json.\n")
    print(json.dumps({"exact_records": len(all_rows), "all_abs_diff_zero": True,
                      "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
