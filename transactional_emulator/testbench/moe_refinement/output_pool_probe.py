#!/usr/bin/env python3
"""Frozen output-context-pool experiment with an existing full expert bank.

A: isolated expert service and a fixed two-expert concurrent case.
B: identical archived route windows with fixed threshold placement.
WC: a separately frozen work-conserving confirmation at Q32, where every
expert group in these windows fits every participating core's context pool.
This is a bounded mechanism experiment, not a DSE or full-model inference.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
import uuid


ORGANIZATIONS = ("single", "homogeneous", "heterogeneous")
MODES = ("legacy_n2", "pool_q8", "pool_q16", "pool_q32")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configuration(organization, mode, dispatch="threshold"):
    # P,R,Mt,activation/weight port,accumulator port,weight/vector/accumulator SRAM.
    shapes = {
        "single": [(8, 512, 4, 1024, 8, 65536, 4194304, 1048576)],
        "homogeneous": [(4, 512, 4, 512, 4, 32768, 2097152, 524288)] * 2,
        "heterogeneous": [(6, 512, 4, 768, 4, 49152, 2097152, 524288),
                          (4, 256, 1, 256, 4, 16384, 2097152, 524288)],
    }
    require(mode != "legacy_n3" or organization == "single", "N3 control is single-only")
    active = 3 if mode == "legacy_n3" else 2
    cores = []
    for i, (p, r, mt, supply, acc_port, weight, vector, accumulator) in enumerate(shapes[organization]):
        refinement = dict(m_rows=mt, tail_policy="valid_rows", active_n_tiles=active,
                          weight_read_elements_per_cycle=supply,
                          accumulator_elements_per_cycle=acc_port,
                          operand_latch_bytes=active * p * r * 2)
        if mode.startswith("pool_q"):
            refinement["output_pool"] = dict(output_contexts=int(mode.removeprefix("pool_q")),
                                              operand_stages=2, scheduler_cycles=1)
        cores.append(dict(id=f"core{i}", blen=p, mlen=r, weight_slots=3, read_cache_bytes=0,
                          activation_elements_per_cycle=supply, weight_sram_bytes=weight,
                          vector_sram_bytes=vector, accumulator_bytes=accumulator, refinement=refinement))
    return dict(schema_version=2, name=f"output_pool_{dispatch}_{mode}_{organization}", cores=cores,
                dispatch_threshold=8, large_core=0, small_core=len(cores) - 1,
                dispatch_policy=dispatch, dispatch_cycles=1, dispatch_queue_bytes=16384,
                global_dma_credits=128, global_dma_staging_bytes=8192, combine_sram_bytes=4194304,
                clock_period_ps=1000, mac_pipeline_cycles=16, vector_elements_per_cycle=512,
                matrix_timing="pipelined",
                dma=dict(issue_policy="per_channel", sector_reads=True, coalesce=True,
                         fair_credits=False, lookup_ii_cycles=1, frontend_sram_bytes=45056))


def file_record(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=digest(path))


def window_record(workload_path, golden_path, name, role):
    workload = json.loads(Path(workload_path).read_text())
    groups = {}
    for route in workload["routes"]:
        groups[route["expert"]] = groups.get(route["expert"], 0) + 1
    return dict(name=name, role=role, workload=file_record(workload_path), golden=file_record(golden_path),
                tokens=len(workload["inputs_bf16"]), groups=groups,
                input_dim=workload["input_dim"], expert_hidden_dim=workload["expert_hidden_dim"])


def verify_record(record):
    require(digest(record["path"]) == record["sha256"], "frozen file changed: " + record["path"])


def preflight_configuration(config, windows):
    """Catch unaffordable frozen configurations before native experiments."""
    require(sum(c["blen"] * c["mlen"] for c in config["cores"]) == 4096, "multiplier budget differs")
    for field, total in (("weight_sram_bytes", 65536), ("vector_sram_bytes", 4194304),
                         ("accumulator_bytes", 1048576), ("activation_elements_per_cycle", 1024)):
        require(sum(c[field] for c in config["cores"]) == total, "resource total differs: " + field)
    for field, total in (("weight_read_elements_per_cycle", 1024), ("accumulator_elements_per_cycle", 8)):
        require(sum(c["refinement"][field] for c in config["cores"]) == total, "port budget differs: " + field)
    for core in config["cores"]:
        p, r, detail = core["blen"], core["mlen"], core["refinement"]
        packed_decoded = core["weight_slots"] * p * (r // 8) * 25
        require(packed_decoded + detail["operand_latch_bytes"] <= core["weight_sram_bytes"], "weight slots/latches do not fit")
    for window in windows:
        for rows in window["groups"].values():
            if config["dispatch_policy"] == "threshold":
                selected = config["large_core"] if rows >= 8 else config["small_core"]
                candidates = [config["cores"][selected]]
            else:
                candidates = config["cores"]  # WC Q32 confirmation must leave all targets eligible.
            for core in candidates:
                detail = core["refinement"]
                pool = detail.get("output_pool")
                if pool:
                    require((rows + detail["m_rows"] - 1) // detail["m_rows"] <= pool["output_contexts"],
                            "complete M cohort does not fit output pool")
                    control = 128 * pool["output_contexts"] + 64 * (core["weight_slots"] + pool["operand_stages"])
                    pending = pool["output_contexts"] * detail["m_rows"] * core["blen"] * 4
                    pipeline = core["blen"] * ((core["mlen"] - 1).bit_length() + config["mac_pipeline_cycles"]) * 4
                    accum = rows * max(window["input_dim"], window["expert_hidden_dim"]) * 4 + pipeline + control + pending
                    require(accum <= core["accumulator_bytes"], "pool control/results do not fit accumulator budget")


def prepare(args):
    root = args.prepared_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    compiler = args.compiler.resolve(strict=True)
    source = args.source_root.resolve(strict=True)
    sys.path.insert(0, str(compiler / "aten/plena"))
    exporter = importlib.import_module("moe_bank_export")
    bank_manifest = source / "banks/qwen/bank.json"
    bank, image, bank_hash = exporter.load_weight_bank(bank_manifest)
    require((bank["input_dim"], bank["expert_hidden_dim"], len(bank["experts"])) == (2048, 512, 256),
            "requires the existing complete Qwen-shaped bank")
    require({0, 7} <= {e["id"] for e in bank["experts"]}, "service expert IDs must exist in the bank")
    configured = {}
    for dispatch in ("threshold", "work_conserving"):
        modes = MODES if dispatch == "threshold" else ("legacy_n2", "pool_q32")
        entries = []
        for mode in modes:
            for organization in ORGANIZATIONS:
                config = configuration(organization, mode, dispatch)
                path = root / "architectures" / (config["name"] + ".json")
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, config)
                entries.append(dict(**file_record(path), organization=organization, mode=mode))
        config = configuration("single", "legacy_n3", dispatch)
        path = root / "architectures" / (config["name"] + ".json")
        write_json(path, config)
        entries.append(dict(**file_record(path), organization="single", mode="legacy_n3"))
        configured[dispatch] = entries

    service = []
    for rows in (1, 2, 8, 32):
        name = f"expert_me{rows}"
        write_json(root / "prepare_progress.json", dict(status="exporting_service", case=name))
        result = exporter.export_bank_window(
            root / "service" / name, bank_manifest=bank_manifest,
            inputs=exporter.full.generated_matrix(rows, 2048, 91, activation=True),
            routes=[dict(token=t, slot=0, expert=0, weight=1.0) for t in range(rows)], name=name,
            provenance=dict(scope="synthetic isolated expert service, not total-machine throughput",
                            placement="threshold8: Me<8 small_core, Me>=8 large_core; the other core is idle"))
        service.append(window_record(result["paths"]["workload"], result["paths"]["golden"], name, "isolated"))
    name = "concurrent_me1_me32"
    write_json(root / "prepare_progress.json", dict(status="exporting_service", case=name))
    result = exporter.export_bank_window(
        root / "service" / name, bank_manifest=bank_manifest,
        inputs=exporter.full.generated_matrix(33, 2048, 91, activation=True),
        routes=[dict(token=0, slot=0, expert=7, weight=1.0)]
               + [dict(token=t, slot=0, expert=0, weight=1.0) for t in range(1, 33)], name=name,
        provenance=dict(scope="synthetic simultaneous independent experts",
                        placement="threshold8: expert0 Me32 on core0, expert7 Me1 on core1 in dual configurations"))
    service.append(window_record(result["paths"]["workload"], result["paths"]["golden"], name, "concurrent"))
    archives = [window_record(source / "windows" / name / "workload.json", source / "windows" / name / "golden.json",
                              name, "archived_routes") for name in ("qwen_full_decode_b8", "qwen_full_decode_b32")]
    for window in service + archives:
        workload = json.loads(Path(window["workload"]["path"]).read_text())
        require(workload["experts"] == bank["experts"] and workload["weight_bank"]["sha256"] == bank_hash,
                "window changed full expert catalog or bank identity")
        image_path = (Path(window["workload"]["path"]).parent / workload["hbm_file"]).resolve()
        require(image_path == image.resolve(), "window does not reuse the same weight image")
        require(max(window["groups"].values()) <= 32, "Q32 WC confirmation needs every group to fit every core")
    phases = {
        "A": dict(label="isolated and concurrent service; fixed threshold8 placement", windows=service,
                  architectures=configured["threshold"], expected_native_runs=130),
        "B": dict(label="mechanism isolation on archived routes; fixed threshold8, not best scheduling",
                  windows=archives, architectures=configured["threshold"], expected_native_runs=52),
        "WC": dict(label="work-conserving confirmation; only Q32 so cohort eligibility is unchanged",
                   windows=archives, architectures=configured["work_conserving"], expected_native_runs=28),
    }
    for phase in phases.values():
        for config in phase["architectures"]:
            preflight_configuration(json.loads(Path(config["path"]).read_text()), phase["windows"])
    hypothesis = dict(
        schema_version=1, status="prepared", frozen_utc=datetime.now(timezone.utc).isoformat(),
        evidence_scope=__doc__.strip(),
        hypothesis="Decoupling bounded output contexts from two operand stages may hide feedback waits. Benefits must be compared with legacy single N3 inside the same budget. More Q need not help when weights, activation or accumulator ports limit service.",
        non_claims=["not a DSE", "not a guaranteed heterogeneous advantage", "no full-model inference",
                    "threshold mapping is an isolation control, not the best scheduler",
                    "single-expert dual measurements intentionally leave one core idle"],
        context_contract="Q counts (M-block,N-band) states per core; each band reserves ceil(Me/Mt) states. Pool uses two operand stages and three weight slots; all pool state is charged inside existing accumulator capacity.",
        strong_control="single legacy N3: three operand latches plus three weight slots occupy 62976 bytes, below 65536-byte weight budget",
        bank=dict(manifest=file_record(bank_manifest), image=file_record(image), bytes=bank["hbm_bytes"]),
        compiler=str(compiler), probe=file_record(Path(__file__)),
        exporters=[file_record(compiler / "aten/plena" / name) for name in
                   ("moe_bank_export.py", "moe_full_shape_export.py", "moe_normal_export.py")],
        controls=dict(repeats=2, workers=2, timeout_seconds=1800, max_hbm_bytes=1 << 30, hbm_channels=8,
                      total_multipliers=4096, total_activation_elements_per_cycle=1024,
                      total_weight_port_elements_per_cycle=1024, total_accumulator_elements_per_cycle=8,
                      weight_sram_bytes=65536, vector_sram_bytes=4194304, accumulator_bytes=1048576,
                      shared_lookup_ii_cycles=1, shared_dma_policy="per_channel", scheduler_cycles=1),
        phases=phases, expected_native_runs=210,
        runtime_estimate="About 10–20 minutes at two workers based on earlier runs; measured wall time will be reported. No long native execution is part of preparation.")
    for phase in phases.values():
        calculated = len(phase["windows"]) * len(phase["architectures"]) * 2
        require(calculated == phase["expected_native_runs"], "frozen experiment count mismatch")
    require(sum(phase["expected_native_runs"] for phase in phases.values()) == 210, "total run count mismatch")
    write_json(root / "prepared.json", hypothesis)
    write_json(root / "prepare_progress.json", dict(status="prepared", expected_native_runs=210))
    return dict(status="prepared", manifest=str(root / "prepared.json"), expected_native_runs=210)


def placement_gate(entry, window):
    architecture, result = entry["architecture"], entry["result"]
    observed = {core["id"] for core in result["cores"] if core["jobs"] > 0}
    if architecture["dispatch_policy"] == "threshold":
        expected = set()
        for job in result["job_completions"]:
            index = architecture["large_core"] if job["rows"] >= 8 else architecture["small_core"]
            expected_core = architecture["cores"][index]["id"]
            require(job["core"] == expected_core, "fixed threshold placement changed")
            expected.add(expected_core)
        require(observed == expected, "reported active cores do not match placement")
    if window["role"] == "isolated":
        require(len(observed) == 1, "isolated expert must use exactly one core")
    if window["role"] == "concurrent" and len(architecture["cores"]) == 2:
        require(len(observed) == 2, "two fixed concurrent experts must exercise both cores")
        jobs = result["job_completions"]
        require(max(j["start_ps"] for j in jobs) < min(j["compute_done_ps"] for j in jobs),
                "the two experts did not have overlapping execution intervals")
    return sorted(observed)


def run(args):
    root = args.prepared_root.resolve(strict=True)
    manifest = root / "prepared.json"
    manifest_hash = digest(manifest)
    frozen = json.loads(manifest.read_text())
    require(frozen["status"] == "prepared", "preparation is incomplete")
    verify_record(frozen["probe"])
    for source in frozen["exporters"]:
        verify_record(source)
    verify_record(frozen["bank"]["manifest"])
    verify_record(frozen["bank"]["image"])
    binary = args.binary.resolve(strict=True)
    binary_hash = digest(binary)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent / "moe_timing/replay"))
    from compare_moe_normal import run_comparison

    selected = list(frozen["phases"]) if args.phase == "all" else [args.phase]
    run_root = root / "results" / ("_".join(selected) + "_" + uuid.uuid4().hex)
    run_root.mkdir(parents=True)
    emulator = here.parents[1]
    sources = [Path(__file__), here.parent / "moe_timing/replay/compare_moe_normal.py",
               emulator / "src/bin/moe_dual_normal.rs", *sorted((emulator / "src/moe_normal").glob("*.rs"))]
    source_hashes = {str(p): digest(p) for p in sources}
    started = time.monotonic()
    result = dict(status="running", evidence_scope=__doc__.strip(), prepared_sha256=manifest_hash,
                  binary_sha256=binary_hash, source_sha256=source_hashes, phases=selected,
                  successful_native_runs=0, cases=[])
    write_json(run_root / "result.json", result)
    write_json(root / "latest_run.json", dict(run_directory=str(run_root), phases=selected, status="running"))
    for phase_name in selected:
        phase = frozen["phases"][phase_name]
        for window in phase["windows"]:
            for record in (window["workload"], window["golden"], *phase["architectures"]):
                verify_record(record)
            output = run_root / phase_name / window["name"]
            write_json(run_root / "progress.json", dict(status="running", phase=phase_name, window=window["name"],
                                                         successful_native_runs=result["successful_native_runs"]))
            controls = frozen["controls"]
            summary = run_comparison(binary, Path(window["workload"]["path"]), Path(window["golden"]["path"]),
                                     [Path(c["path"]) for c in phase["architectures"]], output,
                                     repeats=controls["repeats"], workers=controls["workers"],
                                     hbm_channels=controls["hbm_channels"], max_hbm_bytes=controls["max_hbm_bytes"],
                                     timeout=controls["timeout_seconds"], atol=0, rtol=0)
            require(summary["all_gates_passed"], "comparison gates failed")
            observations = []
            for entry in summary["comparisons"]:
                active = placement_gate(entry, window)
                stats = entry["result"]
                observations.append(dict(architecture=entry["architecture"]["name"], active_cores=active,
                                         total_ps=stats["total_ps"], useful_macs=stats["useful_macs"],
                                         issued_macs=stats["issued_macs"], hbm_read_bytes=stats["hbm_read_bytes"],
                                         cores=stats["cores"], job_completions=stats["job_completions"]))
            result["cases"].append(dict(phase=phase_name, name=window["name"], role=window["role"],
                                         result=str(output / "comparison.json"), observations=observations))
            result["successful_native_runs"] += controls["repeats"] * len(phase["architectures"])
            write_json(run_root / "result.json", result)
    require(result["successful_native_runs"] == sum(frozen["phases"][p]["expected_native_runs"] for p in selected),
            "completed run count differs from frozen plan")
    require(digest(manifest) == manifest_hash and digest(binary) == binary_hash, "plan or executable changed")
    require(all(digest(Path(p)) == sha for p, sha in source_hashes.items()), "source changed during execution")
    verify_record(frozen["bank"]["manifest"])
    verify_record(frozen["bank"]["image"])
    result.update(status="passed", all_gates_passed=True, wall_seconds=time.monotonic() - started)
    write_json(run_root / "result.json", result)
    write_json(run_root / "progress.json", dict(status="passed", successful_native_runs=result["successful_native_runs"]))
    write_json(root / "latest_run.json", dict(run_directory=str(run_root), phases=selected, status="passed"))
    return dict(status="passed", result=str(run_root / "result.json"), successful_native_runs=result["successful_native_runs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparing = sub.add_parser("prepare", help="freeze configurations and service inputs, no native execution")
    preparing.add_argument("--compiler", type=Path, required=True)
    preparing.add_argument("--source-root", type=Path, required=True, help="existing fixed_bank_full_qwen root")
    preparing.add_argument("--prepared-root", type=Path, required=True)
    executing = sub.add_parser("run", help="execute a selected phase with the frozen release binary")
    executing.add_argument("--prepared-root", type=Path, required=True)
    executing.add_argument("--binary", type=Path, required=True)
    executing.add_argument("--phase", choices=("A", "B", "WC", "all"), default="A")
    args = parser.parse_args()
    result = prepare(args) if args.command == "prepare" else run(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
