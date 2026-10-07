#!/usr/bin/env python3
"""Isolate a fixed 4+4 versus 6+2 accumulator port partition.

This is a two-window diagnosis with a matched single-core control, not a DSE.
The executable and comparison implementation are explicitly pinned by the caller.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--comparison-script", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--nonsquare-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve(strict=True)
    source = args.comparison_script.resolve(strict=True)
    frozen_compare = output / "compare_moe_normal.py"
    shutil.copyfile(source, frozen_compare)
    spec = importlib.util.spec_from_file_location("frozen_comparison", frozen_compare)
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)

    source_dir = args.nonsquare_root.resolve(strict=True) / "architectures/per_channel"
    single = json.loads((source_dir / "single.json").read_text())
    original = json.loads((source_dir / "candidate.json").read_text())
    compare.require([(c["blen"], c["mlen"]) for c in original["cores"]]
                    == [(6, 512), (4, 256)], "unexpected starting shapes")
    compare.require([c["refinement"]["accumulator_elements_per_cycle"]
                     for c in original["cores"]] == [4, 4], "unexpected starting ports")
    candidate = copy.deepcopy(original)
    for core, port in zip(candidate["cores"], (6, 2), strict=True):
        core["refinement"]["accumulator_elements_per_cycle"] = port
    configs = (single, original, candidate)
    names = ("single_acc8", "heterogeneous_acc4_4", "heterogeneous_acc6_2")
    config_dir = output / "architectures"
    config_dir.mkdir()
    paths = []
    for config, name in zip(configs, names, strict=True):
        config["name"] = "port_control_" + name
        path = config_dir / (name + ".json")
        write_json(path, config)
        paths.append(path)
    prepared = args.prepared_root.resolve(strict=True)
    windows = [prepared / "windows" / ("qwen_full_decode_" + batch) for batch in ("b8", "b32")]
    bank = prepared / "banks/qwen/bank.json"
    tracked = [Path(__file__).resolve(), frozen_compare, binary, bank, bank.parent / "weights.bin", *paths]
    tracked.extend(w / f for w in windows for f in ("workload.json", "golden.json"))
    hashes = {str(p): digest(p) for p in tracked}
    hypothesis = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Only move two accumulator elements/cycle from small to large. It may shorten large-core service but worsen small-core tails; MAC share alone does not predict the outcome.",
        "scope": "Two archived fixed-bank route windows, synthetic numerical values, native HBM operator timing; not a search or full-model inference.",
        "repeats": 2, "expected_native_runs": 12,
        "controls": "Same 4096 PEs, total accumulator/activation/weight ports, configured SRAM, HBM, routes, shapes and normal N2 scheduling within each comparison.",
        "sha256": hashes,
    }
    write_json(output / "hypothesis.json", hypothesis)
    rows = []
    for window in windows:
        result = compare.run_comparison(
            binary, window / "workload.json", window / "golden.json", paths,
            output / "comparisons" / window.name, repeats=2, atol=0, rtol=0,
            max_hbm_bytes=1 << 30, timeout=1800, workers=2,
        )
        compare.require(result["all_gates_passed"], "comparison gate failure")
        for point in result["comparisons"]:
            report = point["result"]
            compare.require(all(c["jobs"] > 0 for c in report["cores"]), "idle configured core")
            rows.append({"window": window.name, "name": point["architecture"]["name"],
                         "total_ps": report["total_ps"], "hbm_read_bytes": report["hbm_read_bytes"],
                         "cores": [{"id": c["id"], "useful_macs": c["useful_macs"],
                                    "accumulator_port_busy_ps": c["refinement"]["accumulator_port_busy_ps"],
                                    "accumulator_port_wait_ps": c["refinement"]["accumulator_port_wait_ps"]}
                                   for c in report["cores"]]})
        write_json(output / "progress.json", {"status": "running", "rows": rows})
    compare.require(all(digest(p) == sha for p, sha in hashes.items()), "frozen input changed")
    write_json(output / "summary.json", {"status": "passed", "all_gates_passed": True,
                                          "successful_native_runs": 12, "rows": rows})
    print(json.dumps({"status": "passed", "summary": str(output / "summary.json")}))


if __name__ == "__main__":
    main()
