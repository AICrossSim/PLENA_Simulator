"""Sampled evaluation windows of one chip for the closed-loop HBM cross-check, with their ideal-HBM reference.

Sampling rule, per batch: sort the snapshot ids of the evaluation file and take positions 4, 12, ..., 60 (8 of 64
snapshots); layers 2, 7, ..., 37 of Qwen3.5 (8 of 40) and 1, 4, ..., 22 of DeepSeek-V2-Lite (8 of 26). That gives 64
windows per batch, each simulated as its own call with the ideal HBM stream; with --reference the cycles are checked
against the evaluation output of the same chip.

usage: python -m complementary_arrays.xcheck.sample OUT_DIR CHIP.json SNAPSHOT_DIR --tag TAG --batches 32,64,96,128
  CHIP.json: {"model": "qwen" | "deepseek", "shapes": per-stream array shapes, "env": {...}}, the simulator inputs of
  the chip ("workload" defaults to sim.workload.MODELS[model]). SNAPSHOT_DIR holds s1_eval_b<B>.npz (routes, live,
  metadata with snapshot and layer). --reference FILE: evaluation output with B<B>_unit, B<B>_layer, B<B>_cycles.
  Writes <TAG>_B<B>.npz and <TAG>_B<B>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

POSITIONS = tuple(range(4, 64, 8))
LAYERS = {"qwen": tuple(range(2, 40, 5)), "deepseek": tuple(range(1, 25, 3))}


def load_snapshots(path):
    with np.load(path) as z:
        meta = json.loads(str(z["metadata"]))
        routes, live = z["routes"], z["live"]
    return [dict(r, ids=routes[i, : live[i]]) for i, r in enumerate(meta)]


def pick(rows, model):
    units = sorted({r["snapshot"] for r in rows})
    if len(units) != 64:
        raise ValueError(f"expected 64 snapshots, found {len(units)}")
    snaps = [units[i] for i in POSITIONS]
    chosen = [r for r in rows if r["snapshot"] in snaps and r["layer"] in LAYERS[model]]
    chosen.sort(key=lambda r: (snaps.index(r["snapshot"]), r["layer"]))
    assert len(chosen) == len(POSITIONS) * len(LAYERS[model]), len(chosen)
    return chosen, snaps


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("chip", type=Path)
    p.add_argument("snapshots", type=Path)
    p.add_argument("--tag", required=True, help="output name, e.g. <model>_<format>_<design>_<org>")
    p.add_argument("--batches", default="32,64,96,128")
    p.add_argument("--reference", type=Path)
    p.add_argument("--pattern", default="s1_eval_b{batch}.npz")
    a = p.parse_args()
    from ..sim.engine import Environment, simulate, token_counts
    from ..sim.workload import MODELS

    chip = json.loads(a.chip.read_text())
    chip.setdefault("workload", MODELS[chip["model"]])
    env = Environment.from_dict(chip["env"])
    a.out.mkdir(parents=True, exist_ok=True)
    for b in (int(x) for x in a.batches.split(",")):
        every = load_snapshots(a.snapshots / a.pattern.format(batch=b))
        rows, snaps = pick(every, chip["model"])
        ids = np.stack([r["ids"] for r in rows]).astype(np.int32)
        # SRAM provisioned as in the evaluation call over every window of the batch
        tokens = token_counts(np.stack([r["ids"] for r in every]), chip["workload"]["experts"])
        out = np.stack(
            [simulate(ids[i : i + 1], chip["shapes"], chip["workload"], env, 1, tokens)[0] for i in range(len(rows))]
        )
        rec = dict(
            ids=ids,
            out=out,
            snapshot=np.array([r["snapshot"] for r in rows]),
            layer=np.array([r["layer"] for r in rows]),
        )
        same = None
        if a.reference:
            with np.load(a.reference) as z:
                unit, layer, cycles = z[f"B{b}_unit"], z[f"B{b}_layer"], z[f"B{b}_cycles"]
            rec["reference_cycles"] = np.array(
                [cycles[(unit == r["snapshot"]) & (layer == r["layer"])][0] for r in rows]
            )
            same = bool(np.array_equal(out[:, 0], rec["reference_cycles"]))
        tag = f"{a.tag}_B{b}"
        np.savez(a.out / f"{tag}.npz", **rec)
        info = dict(chip, batch=b, snapshots=snaps, layers=LAYERS[chip["model"]], provision_tokens=tokens)
        info["equal_to_reference"] = same
        (a.out / f"{tag}.json").write_text(json.dumps(info, indent=1, default=str))
        print(tag, "windows", len(rows), "equal_to_reference", same, "cycles", int(out[:, 0].sum()), flush=True)


if __name__ == "__main__":
    main()
