"""Assemble repeat receipts from completed runs without evaluating a model.

Completion of a capped search is distinct from deterministic repetition of
its evaluated witnesses. This receipt never certifies unopened regions.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, MODES, sha, write_json
from .search import _engine_hash


def data(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def rows(path):
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return value is True or str(value).lower() == "true"


def main():
    folder = ROOT / "results"
    checks = []
    references = {}

    def check(name, ok, **detail):
        checks.append({"name": name, "ok": bool(ok), **detail})

    gate = data(folder / "E0/PHASE1_GATE.json")
    historical = rows(folder / "E0/reproduce_check.csv")
    check("historical_reproduction", len(historical) == 1350 and
          all(float(row["abs_diff"]) == 0 for row in historical) and
          gate.get("exact_reproduction", {}).get("all_exact", False), rows=len(historical))
    lb = data(folder / "E3/lb_validity_protocol.json")
    check("regional_bound_audit", lb.get("checks") == 36000 and
          lb.get("repeats") == 2 and lb.get("all_ok"), protocol=lb)
    engine = _engine_hash()
    for mode in MODES:
        points = data(folder / f"E3/seed_points_{mode}.json")
        valid = [p for p in points if "invalid" not in p] if isinstance(points, list) else []
        check(f"seed_{mode}", bool(valid) and
              all(p.get("repeat_identical") and p.get("engine_sha256") == engine for p in valid),
              evaluated_valid_points=len(valid), declared_points=len(points) if isinstance(points, list) else 0)

    required = {
        "E2micro_cold_recompute": ("model.py",),
        "E3_main_search": ("model.py", "optimizer.py", "search.py", "main_search.py"),
        "E3_schedule_gaps": ("model.py", "optimizer.py", "main_search.py"),
        "E1": ("model.py", "optimizer.py", "run.py"),
        "E2layer": ("model.py", "run.py"),
        "E4": ("model.py", "optimizer.py", "run.py"),
        "E5": ("model.py", "predictors.py", "run.py"),
        "E6": ("run.py",),
        "E3_grid": ("model.py", "optimizer.py", "search.py", "regions.py"),
        "E3_extreme": ("model.py", "optimizer.py", "search.py", "regions.py", "extreme.py"),
        "E3_robust": ("model.py", "optimizer.py", "robust.py"),
        "E3_sobol": ("model.py", "optimizer.py", "search.py", "regions.py", "sensitivity.py"),
        "E3_flip": ("model.py", "optimizer.py", "search.py", "regions.py", "sensitivity.py"),
    }
    executions = [(path, data(path)) for path in sorted((folder / "executions").glob("*.json"))]
    for label, sources in required.items():
        compatible = [(p, r) for p, r in executions if r.get("label") == label and
                      r.get("returncode") == 0 and r.get("finished_utc") and
                      all(r.get("source_sha256", {}).get(name) == sha(ROOT / name) for name in sources)]
        receipt = max(compatible, key=lambda item: item[1]["finished_utc"]) if compatible else None
        check(label, receipt is not None, evidence=str(receipt[0].relative_to(ROOT)) if receipt else None)
        if receipt:
            references[str(receipt[0].relative_to(ROOT))] = sha(receipt[0])

    for name, key, count in (("workload_map", "point_index", 4320),
                             ("sobol_samples", "sample_index", 1792)):
        values = rows(folder / f"E3/{name}.csv")
        check(name + "_coverage", len(values) == count and
              {int(v[key]) for v in values} == set(range(count)), expected=count, actual=len(values))
    flip = rows(folder / "E3/flip_samples.csv")
    check("flip_coverage", len(flip) == 105 and
          len({(v["param"], v["value"]) for v in flip}) == 105, rows=len(flip))
    checks_ok = all(item["ok"] for item in checks)
    record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "All actually evaluated physical configurations and complete learned predictor sequences; unopened search regions are not certified or claimed evaluated",
        "all_configurations_identical": checks_ok,
        "checks": checks, "execution_receipt_sha256": references,
        "current_engine_sha256": engine,
        "source_sha256": sha(Path(__file__)),
        "protocol": "Performance drivers assert equality of two full result objects; E5 repeats warmup and heldout in two fresh identical states. Search wall-clock trajectories need not be identical; witness repetitions are.",
        "caveat": "This is an execution/repetition receipt, not RTL calibration, full search closure, or complete-model timing.",
    }
    write_json(folder / "REPEAT_CHECKS.json", record)
    print(json.dumps({"all_evaluated_repeats_accounted_for": checks_ok,
                      "checks": len(checks), "failed": [c["name"] for c in checks if not c["ok"]]}))
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
