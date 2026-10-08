"""No-learning estimator with the same progress callbacks as other estimators.

The original None baseline remains untouched. This additional control isolates
estimation/learning from merely enabling existing runtime progress rechecks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from ...common import ROOT, sha, write_json
from ...predictors import Predictor as BasePredictor
from . import run as campaign

HERE = Path(__file__).resolve().parent/"nominal_control"


class NominalPredictor(BasePredictor):
    def __init__(self):
        super().__init__("static")
        self.name = "nominal"

    def predict(self, e, c, nominal, **kw):
        self.calls += 1
        return nominal

    def on_complete(self, e, c, predicted, actual, nominal=None):
        pass

    def on_progress(self, e, c, elapsed, quarter, remaining):
        return None

    def state_bits(self):
        return 0  # No adaptive estimator history; common controller state excluded.


def factory(name):
    assert name == "nominal"
    return NominalPredictor()


def init_worker():
    campaign.init_worker()
    # Local Python harness substitution only; no source/runtime mutation.
    campaign.Predictor = factory


def csv_out(name, rows):
    with (HERE/name).open("w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    init_worker()
    HERE.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    paths = [ROOT/f for f in campaign.MANIFEST["source_hashes"]]
    paths += [campaign.PARENT/"runtime.py", Path(campaign.__file__), Path(__file__),
              campaign.PARENT/"selection.json", campaign.PARENT/"dataflow_ablation_20261008/run.py"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    assert all(before[k] == v for k,v in campaign.MANIFEST["source_hashes"].items())
    specs = [(m,n,"nominal") for m in campaign.MODES for n in campaign.NAMES]
    rows, tasks, receipts = [], [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker) as pool:
        for spec, (rs,ts,receipt) in zip(specs,pool.map(campaign.job,specs)):
            rows.extend(rs)
            tasks.extend(ts)
            receipts.append(receipt)
            print({"done":spec,"geomean_ms":campaign.gm(r['latency_ms'] for r in rs)},flush=True)
    assert len(rows) == 810 and all(r['result_digest'] == r['repeat_digest'] for r in rows)
    csv_out("per_window.csv", rows)
    csv_out("task_predictions.csv", tasks)
    write_json(HERE/"repeat_checks.json", receipts)
    unchanged = {name: sha(ROOT/name) == value for name,value in before.items()}
    assert all(unchanged.values())
    write_json(HERE/"METADATA.json", {"started_utc": started,
        "finished_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-start,
        "command":sys.argv,"execution_commit":subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parents[2],text=True).strip(),
        "source_sha256":before,"source_unchanged":unchanged,"configurations":6,
        "development_windows":18,"heldout_windows":135,"repeats_per_configuration":2,
        "simulation_calls":1836,"dispatch_parameters":campaign.SELECTED,
        "scope":"no-learning nominal estimator; existing quarter-progress rechecks enabled; same common WS fixed runtime",
        "none_difference":"None skips predictor progress rechecks; nominal keeps them without any learned estimate or progress ETA correction"})
    print({"completed":6,"window_rows":len(rows),"task_rows":len(tasks)},flush=True)


if __name__ == "__main__":
    main()
