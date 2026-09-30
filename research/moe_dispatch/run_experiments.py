#!/usr/bin/env python3
"""Reproducible runner for the independent, non-Ramulator dispatch model.

No conclusions are generated. Each point is repeated twice and accepted only
when the complete raw report bytes match. Front states are per-core exclusive
front-end observations; they are NOT additive global latency components and
may overlap arithmetic activity, memory service, and the other core's states.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

from frontend import compiler as frontend, COMPILER_PATH, default_binary

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/moe_dispatch"
DEFAULT_BINARY = default_binary()
ORGANIZATIONS = (("6", [6]), ("33", [3, 3]), ("42", [4, 2]))
MODES = (("fixed", "fixed", "none"), ("dynamic", "dynamic", "none"),
         ("adaptive", "dynamic", "adaptive"))
BASE = {
    "hbm_bytes_per_ns": 256, "hbm_latency_ns": 64, "credits": 256,
    "arbiter": "rr", "ideal_hbm": False, "ideal_onchip": False,
    "control_cost": True, "window": 4, "onchip_bytes_per_ns": 384,
    "vector_elements_per_ns": 32, "dot_tail_ns": 20, "record_trace": False,
}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as f:
        temp = Path(f.name)
        f.write(content)
    os.replace(temp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def source_paths() -> dict[str, Path]:
    files = list(HERE.glob("*.py")) + list((HERE / "rust").glob("Cargo.*"))
    files += list((HERE / "rust/src").rglob("*.rs"))
    paths = {str(p.relative_to(HERE)): p for p in sorted(files) if p.is_file()}
    paths["compiler/compiler.py"] = COMPILER_PATH
    return paths


def source_inventory() -> dict[str, str]:
    return {name: file_sha(path) for name, path in source_paths().items()}


def toy_workloads() -> list[dict[str, Any]]:
    """Shape-mechanism inputs, not captured model routes or numerical weights.

    Tokens are disjoint per expert, top-k=1; e.g. Me4/2 means batch 6, with
    four rows routed to expert 0 and two to expert 1. There is no shared expert.
    """
    result = []
    for name, ms in (("toy_me4_2", [4, 2]), ("toy_me3_3", [3, 3]),
                     ("toy_onehot8", [8]), ("toy_uniform_me1", [1] * 6)):
        h, f, experts, tokens, at = 512, 128, [], [], 0
        for eid, me in enumerate(ms):
            ids = list(range(at, at + me))
            weights = {}
            for phase_index, (phase, n, k) in enumerate((("gate", f, h), ("up", f, h), ("down", h, f))):
                weights[phase] = {
                    "tensor_name": f"synthetic.expert{eid}.{phase}", "shape_nk": [n, k], "dtype": "BF16",
                    "hbm_base": (eid * 3 + phase_index) * frontend.EXPERT_PHASE_STRIDE,
                    "row_stride_bytes": frontend.align(k * 2), "payload_bytes": n * k * 2,
                    "physical_bytes": n * frontend.align(k * 2),
                    "source_sha256": digest([name, eid, phase, n, k]),
                    "source_shard": "synthetic-shape-only-no-numerical-weight-payload",
                }
            expert = {"id": eid, "is_shared": False, "Me": me, "H": h, "F": f,
                      "token_indices": ids, "route_slots": [0] * me,
                      "route_scores": [1.0] * me, "weights": weights}
            expert["dag"] = frontend.expert_dag(expert)
            experts.append(expert)
            tokens.extend({"token_index": t, "sample_id": f"synthetic-{t}",
                           "routes": [{"expert_id": eid, "slot": 0, "score": 1.0}]} for t in ids)
            at += me
        result.append({"id": name, "batch": sum(ms), "hidden": h, "top_k": 1,
                       "tokens": tokens, "experts": experts,
                       "input": {"shape": [sum(ms), h], "dtype": "BF16", "synthetic": True},
                       "scope": "synthetic routed shape mechanism, no captured numerical payload",
                       "routing_is_input_not_timed": True, "route_scores_renormalized": False})
    return result


def make_points(workloads: list[dict[str, Any]], suite: str, trace: bool) -> list[dict[str, Any]]:
    points = []
    if suite == "runtime":
        for w in workloads:
            if w["batch"] not in (2, 4, 8, 16):
                continue
            for org, lanes in ORGANIZATIONS:
                for mode, dispatch, prefetch in (
                    ("baseline", "fifo", False), ("dispatch_only", "dynamic", False),
                    ("prefetch_only", "fifo", True), ("both", "dynamic", True)):
                    config = dict(BASE, lanes=lanes, group=4, dispatch=dispatch,
                                  split="none", record_trace=trace, window=8,
                                  runtime_fsm=True, next_prefetch=prefetch)
                    points.append({"key": f"runtime__{w['id']}__{org}__{mode}",
                                   "suite": suite, "organization": org, "mode": mode,
                                   "condition": "base", "workload": w, "config": config})
        return points
    selected = toy_workloads() if suite == "toy" else workloads
    if suite == "sensitivity":
        selected = [w for w in workloads if w["batch"] == 8]
    for w in selected:
        for org, lanes in ORGANIZATIONS:
            for group in ([4] if suite == "sensitivity" else [1, 2, 4]):
                for mode, dispatch, split in MODES:
                    conditions = [("base", {})]
                    if suite == "sensitivity":
                        conditions = [(f"bw{bw}_lat{lat}", {"hbm_bytes_per_ns": bw, "hbm_latency_ns": lat})
                                      for bw in (64, 128, 256, 512) for lat in (32, 64, 128)]
                        conditions += [("ideal_hbm", {"ideal_hbm": True}),
                                       ("ideal_onchip", {"ideal_onchip": True}),
                                       ("ideal_both", {"ideal_hbm": True, "ideal_onchip": True})]
                    for condition, overrides in conditions:
                        config = dict(BASE, lanes=lanes, group=group, dispatch=dispatch,
                                      split=split, record_trace=trace, runtime_fsm=False, **overrides)
                        key = f"{suite}__{w['id']}__{org}__g{group}__{mode}__{condition}"
                        points.append({"key": key, "suite": suite, "organization": org,
                                       "mode": mode, "condition": condition,
                                       "workload": w, "config": config})
    return points


def attach_layout(point: dict[str, Any]) -> dict[str, Any]:
    if not hasattr(frontend, "engine_layout"):
        raise RuntimeError("compiler.engine_layout is not yet available; freeze the compiler helper first")
    workload = copy.deepcopy(point["workload"])
    workload["engine_layout"] = frontend.engine_layout(workload, point["config"]["lanes"], point["config"]["group"])
    return workload


def run_point(point: dict[str, Any], output: Path, binary: Path, provenance: dict[str, Any],
              repeats: int, timeout: int, prepare_only: bool) -> dict[str, Any]:
    workload = attach_layout(point)
    identity = {"binary_sha256": provenance["binary_sha256"], "sources": provenance["sources"],
                "workload": workload, "config": point["config"], "runner_schema": 1}
    point_hash = digest(identity)
    directory = output / "points" / (point["key"] + "__" + point_hash[:16])
    directory.mkdir(parents=True, exist_ok=True)
    wpath, cpath = directory / "workload.json", directory / "config.json"
    write_json(wpath, workload)
    write_json(cpath, point["config"])
    stamp = {"key": point["key"], "point_hash": point_hash,
             "binary_sha256": provenance["binary_sha256"], "source_bundle_sha256": provenance["source_bundle_sha256"],
             "workload_sha256": file_sha(wpath), "config_sha256": file_sha(cpath),
             "scope": "candidate analytical model; not native Ramulator or measured silicon",
             "repeats_required": repeats, "input_directory": str(directory)}
    write_json(directory / "point_manifest.json", stamp)
    if prepare_only:
        return {"point": point, "directory": directory, "point_hash": point_hash, "prepared": True}
    report_paths = [directory / f"report_repeat{i + 1}.json" for i in range(repeats)]
    receipt_path = directory / "repeat_receipt.json"
    cached = False
    if receipt_path.exists() and all(p.exists() for p in report_paths):
        receipt = json.loads(receipt_path.read_text())
        hashes = [file_sha(p) for p in report_paths]
        cached = (receipt.get("point_hash") == point_hash and len(set(hashes)) == 1
                  and receipt.get("report_sha256") == hashes[0] and receipt.get("repeat_count") == repeats)
    wall_seconds = []
    if not cached:
        for repeat, report_path in enumerate(report_paths, 1):
            start = time.monotonic()
            with (directory / f"stdout_repeat{repeat}.txt").open("w") as stdout, (directory / f"stderr_repeat{repeat}.txt").open("w") as stderr:
                completed = subprocess.run([str(binary), str(wpath), str(cpath), str(report_path)],
                                           stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            wall_seconds.append(time.monotonic() - start)
            if completed.returncode:
                raise RuntimeError(f"{point['key']} repeat {repeat} failed ({completed.returncode}); see {directory}")
        hashes = [file_sha(p) for p in report_paths]
        if len(set(hashes)) != 1:
            write_json(receipt_path, {**stamp, "repeat_match": False, "report_hashes": hashes})
            raise RuntimeError(f"nonidentical repeated raw reports: {point['key']}")
        write_json(receipt_path, {"point_hash": point_hash, "repeat_match": True, "repeat_count": repeats,
                                  "report_sha256": hashes[0], "host_wall_seconds_not_simulation_time": wall_seconds})
    report = json.loads(report_paths[0].read_text())
    if report.get("drained") is not True or report.get("ownership_k_order_capacity_checks") is not True:
        raise RuntimeError(f"missing drain/capacity validation: {point['key']}")
    if report.get("config") != point["config"]:
        # Config may acquire new default fields; all explicitly provided fields must match.
        actual = report.get("config", {})
        if any(actual.get(k) != v for k, v in point["config"].items()):
            raise RuntimeError(f"simulator did not use requested config: {point['key']}")
    return {"point": point, "directory": directory, "point_hash": point_hash,
            "report": report, "cached": cached, "report_sha256": file_sha(report_paths[0])}


def summary_row(item: dict[str, Any]) -> dict[str, Any]:
    p, r = item["point"], item["report"]
    cores = r["cores"]
    stat = lambda key: sum(c["stats"].get(key, 0) for c in cores)
    cfg = p["config"]
    row = {"point": p["key"], "point_hash": item["point_hash"], "suite": p["suite"],
           "workload": r["workload"], "batch": p["workload"]["batch"],
           "organization": p["organization"], "core_m": "+".join(map(str, cfg["lanes"])),
           "mode": p["mode"],
           "n_tile": 4, "k_tile": 512, "group": cfg["group"],
           "dispatch": cfg["dispatch"], "split": cfg["split"], "condition": p["condition"],
           "hbm_bytes_per_ns": cfg["hbm_bytes_per_ns"], "hbm_latency_ns": cfg["hbm_latency_ns"],
           "ideal_hbm": cfg["ideal_hbm"], "ideal_onchip": cfg["ideal_onchip"],
           "latency_cycles_at_1ghz": r["cycles"], "latency_ms": r["time_ms_at_1ghz"],
           "experts_done_ms": r["experts_done_cycles"] / 1e6,
           "combine_tail_ms": r["combine_tail_cycles"] / 1e6,
           "weight_mib": r["weight_bytes"] / 1024**2,
           "x_staging_mib": stat("x_stage_bytes") / 1024**2,
           "z_transfer_kib": stat("z_exchange_bytes") / 1024,
           "remote_copy_bytes": stat("remote_copy_bytes"), "all_copy_bytes": stat("copy_bytes"),
           "useful_macs": r["useful_macs"], "issued_macs": r["issued_macs"],
           "mac_spatial_utilization": r["useful_macs"] / r["issued_macs"] if r["issued_macs"] else 0,
           "control_service_ns_sum_not_wall_time": stat("control_cycles"),
           "vector_service_ns_sum_not_wall_time": stat("vector_service_cycles"),
           "dispatch_decisions": r["dispatch_decisions"], "deferrals": r["deferrals"],
           "credit_peak": r["credit_peak"], "repeats_equal": True,
           "raw_report_sha256": item["report_sha256"], "raw_directory": str(item["directory"])}
    row["latency_us"] = r["cycles"] / 1000
    row["tile_issues"] = stat("issues")
    row["hbm_read_bytes"] = r["weight_bytes"]
    row["hbm_write_bytes"] = 0  # Results remain in the accounted on-chip inbox.
    row["input_backpressure_cycles"] = r.get("input_backpressure_cycles", 0)
    for key in ("dma_accepted", "dma_landed", "dma_backpressure_cycles", "next_bindings",
                "next_prefetch_tiles", "next_ready_at_promotion", "next_inflight_at_promotion",
                "next_weight_wait_cycles"):
        row[key] = stat(key)
    for index in range(2):
        c = cores[index] if index < len(cores) else None
        for field, key in (("private_peak_bytes", "workspace_peak_bytes"),
                           ("weight_peak_bytes", "weight_peak_bytes"), ("x_peak_bytes", "x_peak_bytes"),
                           ("completion_ns", "done_cycle"), ("control_service_ns", "control_cycles")):
            row[f"core{index}_{field}"] = c["stats"].get(key, 0) if c else ""
        row[f"core{index}_private_capacity_bytes"] = c["capacity"] if c else ""
    return row


def front_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, core in enumerate(item["report"]["cores"]):
        states = core["stats"]["front_states"]
        observed = sum(states.values())
        for state, cycles in sorted(states.items()):
            rows.append({"point": item["point"]["key"], "point_hash": item["point_hash"],
                         "core": index, "m_lanes": core["m"], "front_state": state,
                         "cycles": cycles, "ms_at_1ghz": cycles / 1e6,
                         "fraction_of_this_core_front_observation": cycles / observed if observed else 0,
                         "this_core_observed_cycles": observed,
                         "interpretation": "exclusive core-front state; overlaps arithmetic/other core; not additive global latency breakdown"})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", type=Path, default=frontend.DEFAULT_WORKLOADS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--suite", choices=("runtime", "matrix", "toy", "sensitivity"), default="runtime")
    parser.add_argument("--pilot", action="store_true", help="matrix: B2 G4 (9 points); toy: Me4/2 G4; sensitivity: BW256/lat64")
    parser.add_argument("--filter", default="", help="regular expression on human-readable point key")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600, help="host seconds per repeat; never used as simulated latency")
    parser.add_argument("--prepare-only", action="store_true", help="emit plans/configs/manifests without starting the binary")
    parser.add_argument("--list", action="store_true", help="list selected point keys without compiling plans or running")
    parser.add_argument("--trace", action="store_true", help="record model timeline; large raw reports")
    args = parser.parse_args()
    if args.workers < 1 or args.repeats != 2:
        parser.error("workers must be positive; this protocol requires exactly two repeats")
    args.output = args.output.resolve()
    if args.workloads.exists():
        workload_data = frontend.load_workloads(args.workloads)
    elif args.suite == "toy":
        workload_data = {"workloads": [], "provenance": {"synthetic": True}}
    else:
        parser.error(f"workload bundle not found: {args.workloads}")
    workloads = workload_data["workloads"]
    if args.suite == "matrix":
        workloads = [w for w in workloads if w["batch"] in (2, 4, 8, 16)]
    points = make_points(workloads, args.suite, args.trace)
    if args.pilot:
        if args.suite == "matrix":
            points = [p for p in points if p["workload"]["batch"] == 2 and p["config"]["group"] == 4]
        elif args.suite == "toy":
            points = [p for p in points if p["workload"]["id"] == "toy_me4_2" and p["config"]["group"] == 4]
        else:
            points = [p for p in points if p["condition"] == "bw256_lat64"]
    if args.filter:
        pattern = re.compile(args.filter)
        points = [p for p in points if pattern.search(p["key"])]
    if not points:
        parser.error("no experiment points selected")
    if args.list:
        for p in points:
            print(p["key"])
        print(json.dumps({"points": len(points), "repeats": args.repeats, "runs": len(points) * args.repeats}))
        return
    if not args.binary.is_file():
        parser.error(f"analytical model binary not found: {args.binary}")
    sources = source_inventory()
    binary_sha = file_sha(args.binary)
    source_bundle = digest(sources)
    provenance = {"binary_sha256": binary_sha, "source_bundle_sha256": source_bundle,
                  "sources": sources, "workloads_provenance": workload_data.get("provenance", {})}
    # Execute a frozen binary copy; ongoing development cannot alter this run.
    prov_dir = args.output / "provenance" / (binary_sha[:16] + "-" + source_bundle[:16])
    prov_dir.mkdir(parents=True, exist_ok=True)
    frozen_binary = prov_dir / "moe-dispatch-analytical-v1"
    if not frozen_binary.exists():
        shutil.copy2(args.binary, frozen_binary)
    if file_sha(frozen_binary) != binary_sha:
        raise RuntimeError("frozen binary hash mismatch")
    for relative, sha in sources.items():
        dest = prov_dir / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_paths()[relative], dest)
        if file_sha(dest) != sha:
            raise RuntimeError("source changed while creating immutable provenance bundle")
    write_json(prov_dir / "manifest.json", provenance)
    started = time.monotonic()
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_point, p, args.output, frozen_binary, provenance,
                               args.repeats, args.timeout, args.prepare_only): p for p in points}
        for future in as_completed(futures):
            point = futures[future]
            try:
                item = future.result()
                results.append(item)
                print(json.dumps({"completed": len(results), "total": len(points), "point": point["key"],
                                  "prepared": item.get("prepared", False), "cached": item.get("cached", False),
                                  "latency_ms": item.get("report", {}).get("time_ms_at_1ghz")}), flush=True)
            except Exception as error:
                failures.append({"point": point["key"], "error": repr(error)})
                print(json.dumps({"failed": point["key"], "error": repr(error)}), file=sys.stderr, flush=True)
    results.sort(key=lambda item: item["point"]["key"])
    campaign_id = digest({"points": [item["point_hash"] for item in results], "suite": args.suite,
                          "binary": binary_sha, "source": source_bundle, "prepare_only": args.prepare_only})[:16]
    campaign = args.output / "campaigns" / (args.suite + "__" + campaign_id)
    campaign.mkdir(parents=True, exist_ok=True)
    if not args.prepare_only:
        write_csv(campaign / "summary.csv", [summary_row(item) for item in results])
        write_csv(campaign / "core_front_states.csv", [row for item in results for row in front_rows(item)])
    manifest = {**provenance, "suite": args.suite, "selected_points": len(points), "completed_points": len(results),
                "completed_simulation_runs": 0 if args.prepare_only else len(results) * args.repeats,
                "repeat_count": args.repeats, "prepare_only": args.prepare_only, "failures": failures,
                "host_elapsed_seconds_not_model_latency": time.monotonic() - started,
                "filter": args.filter, "pilot": args.pilot, "workers": args.workers,
                "points": [{"key": item["point"]["key"], "point_hash": item["point_hash"],
                            "directory": str(item["directory"])} for item in results],
                "front_state_semantics": "per-core exclusive observations, overlapping other cores/arithmetic; never sum as global latency breakdown"}
    write_json(campaign / "manifest.json", manifest)
    write_json(args.output / ("latest_" + args.suite + ".json"), {"campaign": str(campaign), "manifest": str(campaign / "manifest.json")})
    print(json.dumps({"campaign": str(campaign), "completed": len(results), "failures": len(failures)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
