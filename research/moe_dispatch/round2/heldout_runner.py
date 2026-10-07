"""Run all frozen held-out stages, with logged commands and strict gates."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from .common import ROOT, MODES, inputs, sha, write_json
from .search import _engine_hash, _workload_hash


def ready():
    folder = ROOT / "results/E3"
    names = ("FROZEN_SELECTION.json", "bnb_summary.json", "bnb_certificate.csv", "bnb_leaves.csv")
    if any(not (folder / n).exists() for n in names):
        return False
    try:
        selection = json.loads((folder / names[0]).read_text())
        if set(selection["modes"]) != set(MODES):
            return False
        dev = inputs()["development"]
        for mode in MODES:
            for family in ("single", "homogeneous", "heterogeneous"):
                row = selection["modes"][mode][family]
                if row["engine_sha256"] != _engine_hash() or row["workload_sha256"] != _workload_hash(dev):
                    return False
                if not row["repeat_identical"] or len(row["latencies_ms"]) != 18 or row["geomean_ms"] <= 0:
                    return False
        return True
    except (KeyError, json.JSONDecodeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--wait-selection", action="store_true")
    args = parser.parse_args()
    if args.wait_selection:
        while not ready():
            time.sleep(5)
    elif not ready():
        raise RuntimeError("Complete, current-engine frozen selection required")
    folder = ROOT / "results"
    status_file = folder / "HELDOUT_RUN_STATUS.json"
    guard = {name: sha(ROOT / name) for name in ("model.py", "optimizer.py", "search.py", "common.py", "run.py")}
    selection_hash = sha(folder / "E3/FROZEN_SELECTION.json")
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "jobs": args.jobs,
              "source_sha256": guard, "selection_sha256": selection_hash, "stages": [], "complete": False}
    write_json(status_file, record)
    for stage in ("E1", "E2layer", "E4", "E5", "E6"):
        if guard != {name: sha(ROOT / name) for name in guard} or selection_hash != sha(folder / "E3/FROZEN_SELECTION.json"):
            raise RuntimeError("Frozen held-out source or selection changed during campaign")
        cmd = [sys.executable, "-m", "research.moe_dispatch.round2.execute", "--label", stage,
               "--", sys.executable, "-m", "research.moe_dispatch.round2.run", "--stage", stage,
               "--jobs", str(args.jobs)]
        start = time.monotonic()
        code = subprocess.call(cmd, cwd=ROOT.parents[2])
        record["stages"].append({"stage": stage, "command": cmd, "returncode": code,
                                  "elapsed_seconds": time.monotonic() - start})
        write_json(status_file, record)
        if code:
            raise SystemExit(code)
    record.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(status_file, record)


if __name__ == "__main__":
    main()
