#!/usr/bin/env python3
"""Export a small nonzero workload and validate both checked-in architectures.

This checks the compiler-to-Rust path, not the full-dimension performance claim.
Requires Python 3.10+, NumPy, torch and PLENA_Tools on PYTHONPATH.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def matrix(rows, cols, seed):
    return [[(((r * 17 + c * 13 + seed * 7) % 41) - 20) / 128 for c in range(cols)] for r in range(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    exporter_path = args.compiler.resolve() / "aten/plena/moe_normal_export.py"
    spec = importlib.util.spec_from_file_location("moe_normal_export", exporter_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "moe_timing/replay"))
    from compare_moe_normal import require, run_comparison

    # Uneven expert groups exercise both core choices; odd D/F exercise tails.
    d, f, tokens = 31, 47, 12
    experts = [
        dict(id=i, gate=matrix(f, d, 3 * i + 1), up=matrix(f, d, 3 * i + 2), down=matrix(d, f, 3 * i + 3))
        for i in range(3)
    ]
    exporter.export_workload(
        root / "fixture",
        name="review_nonzero_tails_and_shared",
        inputs=matrix(tokens, d, 91),
        experts=experts,
        routes=[dict(token=t, slot=0, expert=0 if t < 10 else 1, weight=1.0) for t in range(tokens)],
        shared_expert=dict(expert=2, weight=0.25),
        provenance={"scope": "synthetic smoke test; not a performance benchmark"},
    )
    summary = run_comparison(
        args.binary.resolve(),
        root / "fixture/workload.json",
        root / "fixture/golden.json",
        [here / "single.json", here / "large_small.json"],
        root / "comparison",
        repeats=2,
        atol=0,
        rtol=0,
    )
    dual = summary["comparisons"][1]["result"]
    require(all(core["jobs"] > 0 for core in dual["cores"]), "smoke must exercise both cores")
    require(any(value & 0x7FFF for row in dual["output_bf16"] for value in row), "smoke must produce nonzero output")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "all_gates_passed": summary["all_gates_passed"],
                "result": str(root / "comparison/comparison.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
