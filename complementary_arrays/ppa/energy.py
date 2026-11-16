"""Energy per token in the MoE layers and array efficiency of the evaluated chips, from mechanism records.

Each input JSON file maps keys to chip records with: cell ('<model>_<format>_...'), org, version, chip (base array
shapes), policy, env (simulator environment: converter_rate, slots, redeliv_packed) and batches {'B<b>':
{'energy_counts': activity counters}}. Records whose re-deliveries stay packed (env redeliv_packed = 1) are decoded
again on every pass and have no BF16 re-read from the main SRAM; the others write re-read blocks back with the ratios
of writeback.py. Rows with the same model, format, design and batch are compared across organisations.

usage: python -m complementary_arrays.ppa.energy OUT_STEM RECORDS.json [...] --writeback DIR [--gpu GPU.json]
  DIR holds <model>_<format>.json from writeback.py. GPU.json: a list of {model, format, batch, mJ_per_token} measured
  on the GPU (gpu/bench), added as a column and a ratio.
"""

import argparse
import copy
import csv
import json
import re
from pathlib import Path

from . import chip as pc

LAYERS = {"qwen": 40, "deepseek": 26}
MODEL = {"qwen": "Qwen3.5-35B-A3B", "deepseek": "DeepSeek-V2-Lite"}
FORMAT = {"w4g32p": "NVFP4", "fp8b128": "FP8"}
BATCHES = (2, 8, 32, 64, 128)  # batches with write-back ratios
ORDER = {"single2": 0, "hom11": 1, "het11": 2}


def design_of(record):
    if "design" in record:
        return record["design"]
    m = re.search(r"-B(\d+)$", record["version"])
    return f"designed for batch {m.group(1)}" if m else "one design for all batches"


def rows_of(record, writeback):
    model, fmt = record["cell"].split("_")[:2]
    arrays = pc.physical_arrays(record["org"], record["chip"])
    env = record["env"]
    area = pc.chip_area(arrays, fmt, record["policy"][0], env["converter_rate"], env["slots"])
    packed = bool(env.get("redeliv_packed", 0))
    for key, batch in record["batches"].items():
        counts, b = batch.get("energy_counts"), int(key[1:])
        if not counts or b not in BATCHES:
            continue
        counts = copy.deepcopy(counts)
        if packed:  # packed re-reads pass the weight port and the dequantizer again, not the BF16 path of the SRAM
            counts["cls_reread_bytes"] = [[0.0, 0.0, 0.0] for _ in counts["cls_reread_bytes"]]
        e = pc.energy_layer(counts, arrays, fmt, area, writeback.get((model, fmt)), b, packed)
        tok = 1e-9 * LAYERS[model] / b  # pJ per layer -> mJ per token over the MoE layers of one step
        yield dict(
            model=MODEL[model],
            format=FORMAT[fmt],
            design=design_of(record),
            version=record["version"],
            chip=record["org"],
            batch=b,
            reread_packed=int(packed),
            mJ_per_token_mid=e["total_mid"] * tok,
            mJ_low=e["total_low"] * tok,
            mJ_high=e["total_high"] * tok,
            mJ_mid_with_clock_tree=e["total_mid_clock"] * tok,
            mJ_mid_row_gating=e["total_mid_rowgate"] * tok,
            hbm_mJ=e["hbm_mid"] * tok,
            sram_and_buffer_mJ=e["sram_mid"] * tok,
            onchip_logic_mJ=e["onchip_logic"] * tok,
            array_mJ=(e["array_busy"] + e["array_idle"] + e["array_periph"]) * tok,
            decoder_mJ=e["decoder"] * tok,
            layer_cycles=e["cycles"],
            moe_step_us=e["cycles"] * LAYERS[model] / 1000.0,
            cycles_equal_eval=batch.get("cycles_equal_eval"),
            **pc.efficiency(arrays, counts),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path, help="output stem: writes OUT.csv and OUT.json")
    p.add_argument("records", type=Path, nargs="+")
    p.add_argument("--writeback", type=Path, required=True)
    p.add_argument("--gpu", type=Path)
    a = p.parse_args()
    writeback = {}
    for f in a.writeback.glob("*.json"):
        model, fmt = f.stem.split("_", 1)
        writeback[(model, fmt)] = json.loads(f.read_text())
    gpu = {}
    if a.gpu:
        for g in json.loads(a.gpu.read_text()):
            gpu[(MODEL.get(g["model"], g["model"]), FORMAT.get(g["format"], g["format"]), g["batch"])] = g[
                "mJ_per_token"
            ]
    records = {}
    for f in a.records:
        records.update(json.loads(f.read_text()))
    rows = [r for rec in records.values() for r in rows_of(rec, writeback)]
    for r in rows:
        g = gpu.get((r["model"], r["format"], r["batch"]))
        r["gpu_mJ_per_token"] = g
        r["gpu_over_this"] = g / r["mJ_per_token_mid"] if g else None
    index = {(r["model"], r["format"], r["design"], r["chip"], r["batch"]): r for r in rows}
    for r in rows:
        if r["chip"] != "het11":
            continue
        for other, key in (("hom11", "pair_over_identical"), ("single2", "pair_over_single")):
            q = index.get((r["model"], r["format"], r["design"], other, r["batch"]))
            if q:
                r[key] = r["mJ_per_token_mid"] / q["mJ_per_token_mid"]
    rows.sort(key=lambda r: (r["model"], r["format"], r["design"], ORDER[r["chip"]], r["batch"]))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    a.out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(
            f"{r['model'][:16]:16s} {r['format']:5s} {r['design']:26s} {r['chip']:7s} B{r['batch']:<3d} "
            f"{r['mJ_per_token_mid']:7.2f} mJ [{r['mJ_low']:5.2f}-{r['mJ_high']:6.2f}]  "
            f"{r['attained_tflops_per_mm2']:5.2f} TFLOPS/mm2"
        )


if __name__ == "__main__":
    main()
