#!/usr/bin/env python3
"""Exercise immutable-bank windows and independently shaped normal cores.

Small synthetic correctness and accounting smoke, not a speedup benchmark.
Every native invocation has cold SRAM/HBM state and the same physical bank.
"""

import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def configuration(name, lanes, m_rows, active_n_tiles=None):
    """64 multipliers and matching aggregate SRAM/port resources in every case."""
    cores = []
    for i, (p, m) in enumerate(zip(lanes, m_rows, strict=True)):
        core = dict(id=f"core{i}", blen=p, mlen=16, activation_elements_per_cycle=4 * p,
                    vector_sram_bytes=16384 * p, accumulator_bytes=4096 * p,
                    weight_sram_bytes=8192 * p, weight_slots=3, read_cache_bytes=0)
        if active_n_tiles is not None:
            core["refinement"] = dict(m_rows=m, tail_policy="valid_rows", active_n_tiles=active_n_tiles,
                                      weight_read_elements_per_cycle=4 * p,
                                      accumulator_elements_per_cycle=p,
                                      operand_latch_bytes=2 * p * 16 * 2)
        cores.append(core)
    return dict(schema_version=2 if active_n_tiles is not None else 1, name=name, cores=cores,
                dispatch_threshold=3, large_core=0, small_core=len(cores) - 1,
                dispatch_policy="threshold", dispatch_cycles=1, dispatch_queue_bytes=4096,
                global_dma_credits=16, global_dma_staging_bytes=1024, combine_sram_bytes=16384,
                clock_period_ps=1000, mac_pipeline_cycles=16, vector_elements_per_cycle=64,
                matrix_timing="pipelined",
                dma=dict(issue_policy="per_channel", sector_reads=True, coalesce=True,
                         fair_credits=False, lookup_ii_cycles=2, frontend_sram_bytes=45056))


def negative_checks(binary, fixture, architecture, output):
    """Malformed bank/window bindings must fail before publishing a result."""
    checks = []
    for case in ("bank_hash", "catalog_bytes", "catalog_view", "image_bytes", "unknown_expert", "format"):
        directory = output / case
        directory.mkdir(parents=True)
        shutil.copytree(fixture["paths"]["bank_dir"], directory / "bank")
        workload = copy.deepcopy(fixture["workload"])
        workload["weight_bank"]["manifest"] = "bank/bank.json"
        workload["hbm_file"] = "bank/weights.bin"
        if case == "bank_hash":
            workload["weight_bank"]["sha256"] = "0" * 64
        elif case == "catalog_bytes":
            catalog = directory / "bank/bank.json"
            catalog.write_bytes(catalog.read_bytes() + b" ")
        elif case == "catalog_view":
            workload["experts"][0]["gate"]["element_base"] += 64
        elif case == "image_bytes":
            image = directory / "bank/weights.bin"
            payload = bytearray(image.read_bytes())
            payload[0] ^= 1
            image.write_bytes(payload)
        elif case == "unknown_expert":
            workload["routes"][0]["expert"] = 999
        elif case == "format":
            workload["metadata"]["block_size"] = 32
        path = directory / "workload.json"
        write_json(path, workload)
        report = directory / "unexpected_report.json"
        result = subprocess.run([str(binary), "--workload", str(path), "--architecture", str(architecture),
                                 "--output", str(report)], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 or report.exists():
            raise RuntimeError(f"malformed {case} unexpectedly produced a successful result")
        (directory / "rejection.log").write_text(result.stdout + result.stderr)
        checks.append(dict(case=case, rejected=True, exit_code=result.returncode,
                           diagnostic=result.stderr.strip()))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true", help="export fixtures/configs without executing Rust")
    mode.add_argument("--run-existing", action="store_true", help="execute previously prepared immutable fixtures/configs")
    args = parser.parse_args()
    compiler = args.compiler.resolve(strict=True)
    binary = args.binary.resolve()
    root = args.output_dir.resolve()
    if args.run_existing:
        if not root.is_dir():
            raise ValueError("--run-existing requires a prepared output directory")
    else:
        root.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    emulator = here.parents[1]
    sys.path.insert(0, str(compiler / "aten/plena"))
    exporter = importlib.import_module("moe_bank_export")
    sys.path.insert(0, str(here.parent / "moe_timing/replay"))
    from compare_moe_normal import require, run_comparison

    if args.run_existing:
        exporter.load_weight_bank(root / "bank/bank.json")
    else:
        exporter.create_weight_bank(root / "bank", input_dim=17, expert_hidden_dim=19, expert_ids=[0, 2, 7])
    inputs = exporter.full.generated_matrix(5, 17, 91, activation=True)
    fixtures = []
    for name, selected in (("window_a", [0, 0, 0, 0, 2]), ("window_b", [7, 7, 2, 2, 2])):
        if args.run_existing:
            fixture = dict(workload=json.loads((root / name / "workload.json").read_text()),
                           golden=json.loads((root / name / "golden.json").read_text()),
                           paths=dict(workload=str(root / name / "workload.json"),
                                      golden=str(root / name / "golden.json")))
            require([r["expert"] for r in fixture["workload"]["routes"]] == selected,
                    "prepared window routing differs from this smoke contract")
        else:
            fixture = exporter.export_bank_window(root / name, bank_manifest=root / "bank/bank.json", inputs=inputs,
                                                 routes=[dict(token=t, slot=0, expert=e, weight=0.75)
                                                         for t, e in enumerate(selected)], name=name)
        fixture["paths"]["bank_dir"] = str(root / "bank")
        fixtures.append(fixture)
    require(fixtures[0]["workload"]["experts"] == fixtures[1]["workload"]["experts"], "route changed bank addresses")
    require(fixtures[0]["workload"]["weight_bank"] == fixtures[1]["workload"]["weight_bank"], "route changed bank identity")
    configs = root / "architectures"
    if not args.run_existing:
        configs.mkdir()
    experiments = {"legacy": [configuration("legacy_single", [4], [4]),
                               configuration("legacy_homogeneous", [2, 2], [2, 2])]}
    for policy in ("per_channel", "demand_aware"):
        for active in (1, 2):
            name = f"refined_{policy}_n{active}"
            experiments[name] = [
                configuration(name + "_single", [4], [5], active),
                configuration(name + "_homogeneous", [2, 2], [3, 3], active),
                configuration(name + "_heterogeneous", [3, 1], [3, 1], active)]
            for architecture in experiments[name]:
                architecture["dma"]["issue_policy"] = policy
    paths = {}
    for experiment, architectures in experiments.items():
        paths[experiment] = []
        for architecture in architectures:
            path = configs / (architecture["name"] + ".json")
            if args.run_existing:
                require(json.loads(path.read_text()) == architecture, "prepared architecture configuration differs")
            else:
                write_json(path, architecture)
            paths[experiment].append(path)
    sources = [Path(__file__), *(compiler / "aten/plena" / name for name in
               ("moe_bank_export.py", "moe_normal_export.py", "moe_full_shape_export.py")),
               here.parent / "moe_timing/replay/compare_moe_normal.py",
               emulator / "src/bin/moe_dual_normal.rs", *sorted((emulator / "src/moe_normal").glob("*.rs"))]
    source_hashes = {str(path): digest(path) for path in sources}
    summary = dict(status="prepared", evidence_scope=__doc__.strip(),
                   bank_sha256=digest(root / "bank/bank.json"), hbm_sha256=digest(root / "bank/weights.bin"),
                   dimensions=dict(input_dim=17, expert_hidden_dim=19, tokens=5, all_expert_ids=[0, 2, 7]),
                   source_sha256=source_hashes,
                   architectures={p.name: digest(p) for group in paths.values() for p in group},
                   windows={f["workload"]["name"]: dict(workload_sha256=f["golden"]["workload_sha256"],
                                                        golden_sha256=digest(Path(f["paths"]["golden"]))) for f in fixtures})
    if args.prepare_only:
        write_json(root / "summary.json", summary)
        print(json.dumps(dict(status="prepared", result=str(root / "summary.json"))))
        return
    require(binary.is_file(), "native binary is missing")
    summary["binary_sha256"] = digest(binary)
    comparisons = []
    for fixture in fixtures:
        for experiment, architecture_paths in paths.items():
            output = root / "comparisons" / fixture["workload"]["name"] / experiment
            result = run_comparison(binary, Path(fixture["paths"]["workload"]), Path(fixture["paths"]["golden"]),
                                    architecture_paths, output, repeats=2, atol=0, rtol=0)
            require(result["all_gates_passed"], "comparison gates failed")
            for comparison in result["comparisons"]:
                cores = comparison["result"]["cores"]
                require(all(core["jobs"] > 0 for core in cores), "both cores must execute work")
                require(any(v & 0x7FFF for row in comparison["result"]["output_bf16"] for v in row),
                        "smoke must produce nonzero output")
            comparisons.append(dict(window=fixture["workload"]["name"], experiment=experiment,
                                     result=str(output / "comparison.json"), all_gates_passed=True))
    summary["negative_checks"] = negative_checks(binary, fixtures[0], paths["refined_per_channel_n1"][0],
                                                 root / ("negative_" + uuid.uuid4().hex))
    require(digest(root / "bank/bank.json") == summary["bank_sha256"], "bank catalog changed during smoke")
    require(digest(root / "bank/weights.bin") == summary["hbm_sha256"], "bank image changed during smoke")
    require(all(digest(Path(path)) == expected for path, expected in source_hashes.items()), "sources changed during smoke")
    summary.update(status="passed", all_gates_passed=True, comparisons=comparisons,
                   successful_native_runs=2 * len(fixtures) * sum(len(group) for group in paths.values()))
    write_json(root / "summary.json", summary)
    print(json.dumps(dict(status="passed", all_gates_passed=True, result=str(root / "summary.json"))))


if __name__ == "__main__":
    main()
