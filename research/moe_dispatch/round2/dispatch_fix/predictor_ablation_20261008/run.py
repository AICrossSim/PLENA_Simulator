"""Repeated predictor ablation, fixed dispatcher and common WS dataflow.

No re-selection of hardware or runtime parameters. Results live outside every
frozen E0--E6 artifact. Prediction affects estimated timing, never readiness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

from ...common import ROOT, canonical, decode_design, encode_design, inputs, sha, write_json
from ...model import Parameters
from ...predictors import Predictor
from .bounded_runtime import simulate, patch_metadata
from ..dataflow_ablation_20261008.run import digest, gm

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
NAMES = ("B1", "B2", "best_hetero")
MODES = ("pipelined", "port_tight")
PREDICTORS = ("none", "random", "static", "btb", "ema", "ours")
WS = None
MANIFEST = None
SELECTED = None


def init_worker():
    global WS, MANIFEST, SELECTED
    WS = inputs()
    MANIFEST = json.loads((PARENT/"frozen_designs.json").read_text())
    SELECTED = json.loads((PARENT/"selection.json").read_text())["chosen"]
    assert SELECTED == {"t_big": 3, "large_first": True}


def csv_out(name, rows):
    with (HERE/name).open("w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def job(spec):
    mode, name, pname = spec
    original = decode_design(MANIFEST["modes"][mode][name])
    d = replace(original, flows=("WS",)*len(original.cores))
    old_hw, new_hw = encode_design(original), encode_design(d)
    assert {k:v for k,v in old_hw.items() if k != "flows"} == {k:v for k,v in new_hw.items() if k != "flows"}
    p = Parameters(onchip_mode=mode)
    sequences, warm_digests, states = [], [], []
    for _ in range(2):
        pred = None if pname == "none" else Predictor(pname)
        warm = [simulate(w, d, p, predictor=pred, **SELECTED) for w in WS["development"]]
        warm_digests.append(digest(warm))
        sequences.append([simulate(w, d, p, predictor=pred, **SELECTED) for w in WS["heldout"]])
        states.append(None if pred is None else repr({**vars(pred), "rng": pred.rng.getstate()}))
    assert canonical(sequences[0]) == canonical(sequences[1]), (spec, "full heldout result mismatch")
    assert warm_digests[0] == warm_digests[1] and states[0] == states[1], (spec, "predictor state mismatch")
    rows, tasks = [], []
    for w, a, b in zip(WS["heldout"], *sequences):
        common = {"onchip_mode": mode, "design": name, "predictor": pname,
                  "window_id": w["id"], "batch": w["batch"]}
        assert a["hbm_bytes"]/a["cycles"] <= p.hbm_bandwidth + 1e-7
        rows.append({**common, "cycles": a["cycles"], "latency_ms": a["latency_ms"],
            "hbm_bytes": a["hbm_bytes"], "result_digest": digest(a), "repeat_digest": digest(b)})
        # Chunk-local expert IDs can repeat: pairing includes storage chunk.
        bindings = {(x.get("chunk_index", 0), x["expert_index"]): x for x in a["bindings"]}
        assert len(bindings) == len(a["tasks"])
        for t in a["tasks"]:
            bind = bindings[t.get("chunk_index", 0), t["expert_index"]]
            original_index = t.get("original_expert_index", t["expert_index"])
            e = w["experts"][original_index]
            assert bind["core"] == t["core"] and t["actual_cycles"] > 0
            tasks.append({**common, "expert_index": t["expert_index"],
                "chunk_index": t.get("chunk_index", 0), "original_expert_index": original_index,
                "expert_id": e.get("id", original_index), "core": t["core"],
                "is_shared": bool(e.get("is_shared", False)), "Me": e["Me"],
                "predicted_cycles": t["predicted_cycles"], "actual_cycles": t["actual_cycles"],
                "bind_cycle": bind["bind_cycle"], "actual_start_cycle": t["start"],
                "predicted_finish_at_bind": bind["predicted_finish"],
                "actual_finish_cycle": t["finish"],
                "first_weight_ready": t.get("first_weight_ready"), "current_end": t.get("current_end")})
    receipt = {"design": name, "onchip_mode": mode, "predictor": pname,
        "hardware": new_hw, "parameters": asdict(p), "dispatch_parameters": SELECTED,
        "predictor_state_bits": 0 if pname == "none" else Predictor(pname).state_bits(),
        "development_windows_per_repeat": len(WS["development"]), "heldout_windows_per_repeat": len(rows),
        "repeats": 2, "warmup_digest": warm_digests[0], "repeat_warmup_digest": warm_digests[1],
        "heldout_digest": digest(sequences[0]), "repeat_heldout_digest": digest(sequences[1]),
        "final_predictor_state_digest": digest(states[0]), "repeat_final_predictor_state_digest": digest(states[1]),
        "hardware_frozen_except_common_WS": True, "full_result_repeats_exact": True}
    return rows, tasks, receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    init_worker()
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    paths = [ROOT/f for f in MANIFEST["source_hashes"]]
    paths += [PARENT/"runtime.py", PARENT/"selection.json", PARENT/"frozen_designs.json",
              PARENT/"dataflow_ablation_20261008/run.py", Path(__file__), HERE/"bounded_runtime.py"]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    assert all(before[k] == v for k,v in MANIFEST["source_hashes"].items())
    specs = [(m, n, pred) for m in MODES for n in NAMES for pred in PREDICTORS]
    assert len(specs) == 36
    rows, tasks, receipts = [], [], []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker) as pool:
        for spec, (rs, ts, receipt) in zip(specs, pool.map(job, specs)):
            rows.extend(rs)
            tasks.extend(ts)
            receipts.append(receipt)
            print({"done": spec, "windows": len(rs), "tasks": len(ts),
                   "geomean_ms": gm(r["latency_ms"] for r in rs)}, flush=True)
    assert len(rows) == 4860 and all(r["result_digest"] == r["repeat_digest"] for r in rows)
    with (PARENT/"dataflow_ablation_20261008/per_window.csv").open() as f:
        baseline = {(r["onchip_mode"], r["design"], r["window_id"]): r for r in csv.DictReader(f)
                    if r["flows_core_order"] in ("WS", "WS/WS")}
    checks = []
    for r in rows:
        if r["predictor"] != ("none" if r["design"] == "B1" else "ours"):
            continue
        ref = baseline[r["onchip_mode"], r["design"], r["window_id"]]
        checks.append({**{k:r[k] for k in ("window_id", "design", "onchip_mode", "predictor")},
            "full_result_digest_exact": r["result_digest"] == ref["result_digest"],
            "cycles_delta": r["cycles"]-float(ref["cycles"]),
            "hbm_bytes_exact": r["hbm_bytes"] == int(ref["hbm_bytes"])})
    assert len(checks) == 810
    csv_out("per_window.csv", rows)
    csv_out("task_predictions.csv", tasks)
    csv_out("baseline_reproduction.csv", checks)
    write_json(HERE/"repeat_checks.json", receipts)
    unchanged = {name: sha(ROOT/name) == value for name,value in before.items()}
    assert all(unchanged.values())
    metadata = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic()-start,
        "execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT.parents[2], text=True).strip(),
        "command": sys.argv, "source_sha256": before, "source_unchanged": unchanged,
        "input_hashes": json.loads((ROOT/"results/E0/frozen_inputs.json").read_text()),
        "dispatch_parameters": SELECTED, "configurations": len(specs), "development_windows": 18,
        "heldout_windows": 135, "repeats_per_configuration": 2, "simulation_calls": 11016,
        "per_window_rows": len(rows), "task_prediction_rows": len(tasks), "existing_WS_exact_checks": len(checks),
        "bounded_runtime_patch": patch_metadata(),
        "baseline_max_absolute_cycle_delta": max(abs(r['cycles_delta']) for r in checks),
        "baseline_hbm_bytes_exact": all(r['hbm_bytes_exact'] for r in checks),
        "baseline_full_digest_exact_count": sum(r['full_result_digest_exact'] for r in checks),
        "scope": "post-router BF16 phase-fluid analytical model; no native HBM/RTL/full-model timing",
        "protocol": "common WS on frozen E4 hardware; same fixed dispatcher; predictor alone varies; 18 dev warmup then all heldout in frozen order; fresh state each repeat; no tuning",
        "none_definition": "nominal finite-resource shape cost; prefetch remains enabled; no learned correction",
        "random_definition": "random task duration estimate, not random core dispatch",
        "single_note": "single predictor ablation enabled explicitly; none reproduces original single baseline exactly",
        "oracle_note": "no new oracle; old E5 conditional frozen-plan replay is not optimal scheduling"}
    write_json(HERE/"METADATA.json", metadata)
    print({"completed": len(specs), "window_rows": len(rows), "task_rows": len(tasks),
           "elapsed_seconds": metadata['elapsed_seconds']}, flush=True)


if __name__ == "__main__":
    main()
