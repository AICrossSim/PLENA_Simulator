#!/usr/bin/env python3
"""Bounded full-dimension immutable-bank probe; not a DSE or full-model claim.

Archive BF16 inputs and routes are reused, while one complete synthetic expert
bank is generated independently of all windows. Native runs start cold. Three
predeclared organizations and one/two active N groups are compared; no winner
selection or post-result tuning is performed here.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np


MODELS = {"qwen": (2048, 512, 256), "deepseek": (2048, 1408, 64)}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def configuration(organization, active):
    # P, R, Mt, activation/weight port, accumulator port, weight/activation/acc SRAM.
    shapes = {
        "single": [(8, 512, 4, 1024, 8, 65536, 4194304, 1048576)],
        "homogeneous": [(4, 512, 4, 512, 4, 32768, 2097152, 524288)] * 2,
        "heterogeneous": [(4, 768, 4, 768, 4, 49152, 2097152, 524288),
                          (4, 256, 1, 256, 4, 16384, 2097152, 524288)],
    }
    cores = []
    for i, (p, r, mt, supply, acc_port, weight, vector, accumulator) in enumerate(shapes[organization]):
        cores.append(dict(id=f"core{i}", blen=p, mlen=r, weight_slots=3, read_cache_bytes=0,
                          activation_elements_per_cycle=supply, weight_sram_bytes=weight,
                          vector_sram_bytes=vector, accumulator_bytes=accumulator,
                          refinement=dict(m_rows=mt, tail_policy="valid_rows", active_n_tiles=active,
                                          weight_read_elements_per_cycle=supply,
                                          accumulator_elements_per_cycle=acc_port,
                                          # Reserve the same two-latch space in both schedule controls.
                                          operand_latch_bytes=2 * p * r * 2)))
    return dict(schema_version=2, name=f"normal_v2_n{active}_{organization}", cores=cores,
                dispatch_threshold=8, large_core=0, small_core=len(cores) - 1,
                dispatch_policy="work_conserving", dispatch_cycles=1, dispatch_queue_bytes=16384,
                global_dma_credits=128, global_dma_staging_bytes=8192, combine_sram_bytes=4194304,
                clock_period_ps=1000, mac_pipeline_cycles=16, vector_elements_per_cycle=512,
                matrix_timing="pipelined",
                dma=dict(issue_policy="per_channel", sector_reads=True, coalesce=True,
                         fair_credits=False, lookup_ii_cycles=1, frontend_sram_bytes=45056))


def prepare(args, root):
    root.mkdir(parents=True, exist_ok=False)
    compiler = args.compiler.resolve(strict=True)
    sys.path.insert(0, str(compiler / "aten/plena"))
    exporter = importlib.import_module("moe_bank_export")
    source_root = args.source_root.resolve(strict=True)
    configs = root / "architectures"
    configs.mkdir()
    architectures = {}
    for active in (1, 2):
        paths = []
        for organization in ("single", "homogeneous", "heterogeneous"):
            config = configuration(organization, active)
            path = configs / (config["name"] + ".json")
            write_json(path, config)
            paths.append(dict(path=str(path.relative_to(root)), sha256=digest(path)))
        architectures[f"active_n{active}"] = paths
    summary = dict(status="preparing", evidence_scope=__doc__.strip(), models={}, architectures=architectures,
                   compiler=str(compiler), probe_sha256=digest(Path(__file__)),
                   exporter_sha256={name: digest(compiler / "aten/plena" / name) for name in
                                    ("moe_bank_export.py", "moe_normal_export.py", "moe_full_shape_export.py")},
                   controls=dict(repeats=2, workers=2, timeout_seconds=600, max_hbm_bytes=1 << 30,
                                 hbm_channels=8, same_pe_total=4096, same_activation_supply=1024,
                                 same_weight_port_supply=1024, same_accumulator_port_supply=8))
    for model in args.models:
        d, f, experts = MODELS[model]
        write_json(root / "progress.json", dict(status="creating_bank", model=model))
        bank_dir = root / "banks" / model
        bank = exporter.create_weight_bank(bank_dir, input_dim=d, expert_hidden_dim=f,
                                            expert_ids=list(range(experts)), max_image_bytes=1 << 30)
        record = dict(input_dim=d, expert_hidden_dim=f, total_experts=experts,
                      bank_manifest=str((bank_dir / "bank.json").relative_to(root)),
                      bank_sha256=digest(bank_dir / "bank.json"), hbm_sha256=bank["hbm_sha256"],
                      hbm_bytes=bank["hbm_bytes"], windows=[])
        for tokens in (8, 32):
            name = f"{model}_full_decode_b{tokens}"
            archive = source_root / name / "workload.json"
            source_bytes = archive.read_bytes()
            old = json.loads(source_bytes)
            if (old["input_dim"], old["expert_hidden_dim"], len(old["inputs_bf16"])) != (d, f, tokens):
                raise ValueError(f"unexpected dimensions in {archive}")
            if old.get("shared_expert") is not None:
                raise ValueError("this probe expects the archived routed-only windows")
            inputs = (np.asarray(old["inputs_bf16"], dtype=np.uint32) << np.uint32(16)).view(np.float32)
            write_json(root / "progress.json", dict(status="exporting_window_and_reference", window=name))
            result = exporter.export_bank_window(root / "windows" / name, bank_manifest=bank_dir / "bank.json",
                                                 inputs=inputs, routes=old["routes"], name=f"fixed_bank_{name}",
                                                 provenance=dict(archive=str(archive), archive_sha256=hashlib.sha256(source_bytes).hexdigest(),
                                                                 reused="BF16 ready inputs and route tuples only; weights generated once in full bank"))
            record["windows"].append(dict(name=name,
                                          workload=str(Path(result["paths"]["workload"]).relative_to(root)),
                                          golden=str(Path(result["paths"]["golden"]).relative_to(root)),
                                          workload_sha256=result["golden"]["workload_sha256"],
                                          golden_sha256=digest(result["paths"]["golden"])))
        first, second = [json.loads((root / w["workload"]).read_text()) for w in record["windows"]]
        if first["experts"] != second["experts"] or first["weight_bank"] != second["weight_bank"]:
            raise AssertionError("routing changed physical bank catalog or identity")
        summary["models"][model] = record
        write_json(root / "prepared.json", summary)
    summary["status"] = "prepared"
    write_json(root / "prepared.json", summary)
    write_json(root / "progress.json", dict(status="prepared"))
    return summary


def execute(args, root):
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "moe_timing/replay"))
    from compare_moe_normal import require, run_comparison

    prepared = json.loads((root / "prepared.json").read_text())
    require(prepared["status"] == "prepared", "preparation did not complete")
    require(set(args.models) <= set(prepared["models"]), "requested model was not prepared")
    binary = args.binary.resolve(strict=True)
    binary_hash = digest(binary)
    require(prepared["probe_sha256"] == digest(Path(__file__)), "probe changed after configurations were frozen")
    result = dict(status="running", evidence_scope=__doc__.strip(), binary_sha256=binary_hash,
                  comparisons=[], successful_native_runs=0)
    for model in args.models:
        record = prepared["models"][model]
        bank = root / record["bank_manifest"]
        require(digest(bank) == record["bank_sha256"], "bank manifest changed")
        require(digest(bank.parent / "weights.bin") == record["hbm_sha256"], "bank image changed")
        for window in record["windows"]:
            require(digest(root / window["workload"]) == window["workload_sha256"], "window changed")
            require(digest(root / window["golden"]) == window["golden_sha256"], "reference changed")
            for experiment, configurations in prepared["architectures"].items():
                require(all(digest(root / c["path"]) == c["sha256"] for c in configurations), "architecture changed")
                write_json(root / "progress.json", dict(status="running", window=window["name"], experiment=experiment))
                output = root / "comparisons" / window["name"] / experiment
                comparison = run_comparison(binary, root / window["workload"], root / window["golden"],
                                            [root / c["path"] for c in configurations], output,
                                            repeats=2, hbm_channels=8, atol=0, rtol=0, timeout=600,
                                            workers=2, max_hbm_bytes=1 << 30)
                require(comparison["all_gates_passed"], "comparison gates failed")
                result["comparisons"].append(dict(model=model, window=window["name"], experiment=experiment,
                                                  result=str(output / "comparison.json"), all_gates_passed=True))
                result["successful_native_runs"] += 2 * len(configurations)
                write_json(root / "result.json", result)
    require(digest(binary) == binary_hash, "native binary changed during experiment")
    result.update(status="passed", all_gates_passed=True)
    write_json(root / "result.json", result)
    write_json(root / "progress.json", dict(status="passed"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run-only", action="store_true")
    parser.add_argument("--models", choices=MODELS, nargs="+", default=["qwen"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    result = None
    if not args.run_only:
        result = prepare(args, root)
    if not args.prepare_only:
        result = execute(args, root)
    print(json.dumps(dict(status=result["status"], output=str(root))))


if __name__ == "__main__":
    main()
