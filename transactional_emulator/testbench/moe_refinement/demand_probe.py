#!/usr/bin/env python3
"""Change only DMA issue policy on frozen full-bank probe configurations.

All configurations receive the same new policy and SRAM budget. The original
prepared bank/windows/configurations remain immutable. This is a bounded
ablation, not a new DSE or a comparison against an older timing model.
"""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "moe_timing/replay"))
    from compare_moe_normal import digest, require, run_comparison

    root = args.prepared_root.resolve()
    frozen_path = root / "prepared.json"
    frozen_hash = digest(frozen_path)
    prepared = json.loads(frozen_path.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs = output / "architectures"
    configs.mkdir()
    paths, origins = [], []
    for group in prepared["architectures"].values():
        for record in group:
            source = root / record["path"]
            require(digest(source) == record["sha256"], "source architecture changed")
            architecture = json.loads(source.read_text())
            require(architecture["schema_version"] == 2, "demand probe requires refined outputs")
            architecture["dma"]["issue_policy"] = "demand_aware"
            target = configs / source.name
            target.write_text(json.dumps(architecture, indent=2, sort_keys=True) + "\n")
            paths.append(target)
            origins.append(dict(source=str(source), source_sha256=record["sha256"],
                                target=str(target), target_sha256=digest(target)))
    summary = dict(status="running", evidence_scope=__doc__.strip(),
                   prepared_sha256=frozen_hash, origins=origins,
                   driver_sha256=digest(Path(__file__)), comparisons=[])
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        for model in prepared["models"].values():
            require(digest(root / model["bank_manifest"]) == model["bank_sha256"], "bank catalog changed")
            for window in model["windows"]:
                workload, golden = root / window["workload"], root / window["golden"]
                require(digest(workload) == window["workload_sha256"], "route window changed")
                require(digest(golden) == window["golden_sha256"], "numerical reference changed")
                result_path = output / "comparisons" / window["name"]
                result = run_comparison(args.binary.resolve(), workload, golden, paths, result_path,
                                        repeats=2, atol=0, rtol=0, timeout=600, workers=2,
                                        max_hbm_bytes=prepared["controls"]["max_hbm_bytes"])
                require(result["all_gates_passed"], "demand comparison gates failed")
                summary["comparisons"].append(dict(window=window["name"],
                    path=str(result_path / "comparison.json"), metrics=[dict(
                        name=c["architecture"]["name"], total_ps=c["result"]["total_ps"],
                        hbm_read_bytes=c["result"]["hbm_read_bytes"],
                        demand_accepted=c["native"]["demand_accepted"],
                        prefetch_accepted=c["native"]["prefetch_accepted"],
                        aged_accepted=c["native"]["aged_accepted"],
                    ) for c in result["comparisons"]]))
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        require(digest(frozen_path) == frozen_hash, "prepared experiment changed")
        for record in origins:
            require(digest(record["source"]) == record["source_sha256"], "source configuration changed")
        summary["status"] = "passed"
        summary["all_gates_passed"] = True
    except Exception as error:
        summary.update(status="failed", all_gates_passed=False, error=str(error))
        raise
    finally:
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="passed", result=str(summary_path))))


if __name__ == "__main__":
    main()
