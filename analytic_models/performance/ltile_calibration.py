"""Audit E/E2 evidence and validate analytical recurrence costs out of sample."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .ltile_cost import Machine, assembly_cost
from .ltile_program import build_program
from .ltile_dma import DmaBackend, sha


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def load_case(path):
    result = json.loads(path.read_text())
    if result.get("status") != "passed" or "timing" not in result:
        return None
    options = result.get("options", {})
    kind = result.get("kind", options.get("kind"))
    if kind not in ("mamba", "kda"):
        return None
    timing = result["timing"]
    counters = timing["counters"]
    hardware = timing.get("v2", {}).get("lanes", {})
    control = options.get("control", result.get("control", "old_isa"))
    if control not in ("row", "fsm", "old_isa"):
        return None
    dot = result.get("mode", "tree")
    if control == "old_isa":
        dot = "tree"
    batch = result.get("batch", result.get("E2_job", {}).get("batch", 1))
    if "state_values" in result and not result.get("fixture", {}).get("file", "").endswith("npz"):
        batch = result["state_values"] // ((64 * 128 * 64) if kind == "mamba" else (96 * 128 * 128))
    service = str(timing.get("dma_write_window", result.get("dma_service", "review")))
    if service not in ("1", "4", "8", "16", "32", "64"):
        service = "review"
    m = Machine(
        lanes=int(result.get("L", hardware.get("L", 256))),
        dot=dot,
        update_latency=int(hardware.get("update_latency", 6)),
        update_ii=int(hardware.get("update_II", 2)),
        dot_latency=int(hardware.get("feedback_core_latency", 6)),
        dot_ii=int(hardware.get("dot_II", 2)),
        sram_cycles=int(hardware.get("sram_port_multiplier", 1)),
        context_cycles=int(hardware.get("context_read_write_cycles_each", 1)),
    )
    actual = dict(
        issue=counters["issue_cycles"],
        scalar=timing["scalar_and_control_cycles"],
        sram=counters["bank_service_cycles"],
        arithmetic=counters["arithmetic_cycles"],
        dependency=counters["dependency_cycles"],
        dma=timing["dma_and_memory_wait_picos"] / timing["period_picos"],
        total=timing["total_picos"] / timing["period_picos"],
    )
    return dict(
        path=path,
        result=result,
        machine=m,
        kind=kind,
        batch=batch,
        tokens=result["tokens"],
        control=control,
        dot=dot,
        service=service,
        phased=options.get("phased", True),
        broadcast=options.get("broadcast", True),
        resident=options.get("resident", True),
        diagnostic=result.get("diagnostic_dma", False),
        actual=actual,
    )


def inventory(evidence):
    paths = []
    # Keep all 158 E executions. Diagnostic-DMA cases validate the instruction
    # model but never supply a performance headline or the B1 parameter split.
    for row in csv.DictReader((evidence / "E/executed_cycles.csv").open()):
        paths.append(evidence / "E" / row["result_path"])
    paths.extend(sorted((evidence / "E/runs").glob("*/result.json")))
    for directory in [
        evidence / "E2/runs",
        evidence / "E2/architecture_search/runs",
        evidence / "E2/analytic_model_20260920/execution/runs",
    ]:
        if directory.exists():
            paths.extend(sorted(directory.glob("*/result.json")))
    cases = []
    for path in dict.fromkeys(paths):
        case = load_case(path)
        if case is not None:
            cases.append(case)
    return cases


def training(case):
    m = case["machine"]
    return (
        case["batch"] == 1
        and case["tokens"] == 4
        and m.lanes == 256
        and m.update_latency == 6
        and m.dot_latency == 6
        and m.sram_cycles == 1
        and case["dot"] == "tree"
        and case["service"] in ("review", "1")
        and case["phased"]
        and case["broadcast"]
        and case["resident"]
        and not case["diagnostic"]
    )


def run(evidence, output, backend=None):
    if backend is None:
        raise ValueError("pinned memory backend required; aggregate DMA fit failed validation")
    output.mkdir(parents=True, exist_ok=True)
    cases = inventory(evidence)
    compiler = evidence / "E/compiler"
    cached_programs = {}
    jobs = {}
    for case in cases:
        key = tuple(
            case[k] for k in ("kind", "batch", "tokens", "control", "phased", "broadcast", "resident", "diagnostic")
        )
        if key not in cached_programs:
            asm, _size = build_program(
                *key[:4], compiler_root=compiler, phased=key[4], broadcast=key[5], resident=key[6], diagnostic=key[7]
            )
            cached_programs[key] = asm
        asm = cached_programs[key]
        case["split"] = "calibration" if training(case) else "validation"
        case["program_sha256"] = hashlib.sha256(asm.encode()).hexdigest()
        identity = (case["program_sha256"], case["machine"], case["service"])
        jobs.setdefault(identity, (asm, []))[1].append(case)

    def price_job(identity, asm):
        _, machine, service = identity
        cost = assembly_cost(asm, machine, trace_memory=True)
        memory = backend.price(cost, service)
        cost.memory_trace.clear()
        return cost, memory

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(price_job, key, value[0]): key for key, value in jobs.items()}
        for index, future in enumerate(as_completed(futures), 1):
            identity = futures[future]
            cost, memory = future.result()
            for case in jobs[identity][1]:
                case["predicted"], case["memory_result"] = cost, memory
            if index % 10 == 0 or index == len(futures):
                print(f"priced {index}/{len(futures)} unique schedules ({len(cases)} observations)", flush=True)

    # Retain the preregistered split, although the final compositional model
    # fits no observed cycles. Hardware parameters come from the frozen R3
    # contract; DRAM service comes from its independently fed memory backend.
    train = [c for c in cases if c["split"] == "calibration"]
    model = dict(
        schema=2,
        method="analytical compute/ports + pinned Ramulator DMA",
        fitted_cycle_parameters=0,
        source_clock_hz=1_000_000_000,
        **backend.identity,
        limitations=[
            "shared DRAM backend is not independent DRAM validation",
            "compute timing validates R3 assumptions, not complete RTL",
        ],
    )
    (output / "dma_model.json").write_text(json.dumps(model, indent=2) + "\n")

    rows = []
    for c in cases:
        predicted = c["predicted"].components()
        row = dict(
            name=c["path"].parent.name,
            model=c["kind"],
            batch=c["batch"],
            tokens=c["tokens"],
            control=c["control"],
            dot=c["dot"],
            L=c["machine"].lanes,
            dma_service=c["service"],
            phased=c["phased"],
            broadcast=c["broadcast"],
            resident=c["resident"],
            port_cycles=c["machine"].sram_cycles,
            feedback=c["machine"].dot_latency,
            split=c["split"],
            source=str(c["path"]),
            source_sha256=hashlib.sha256(c["path"].read_bytes()).hexdigest(),
            program_sha256=c["program_sha256"],
        )
        for component, actual in c["actual"].items():
            pred = predicted[component]
            row[f"{component}_rust"] = actual
            row[f"{component}_analytic"] = pred
            row[f"{component}_absolute_error"] = abs(pred - actual)
            row[f"{component}_relative_error"] = abs(pred - actual) / actual if actual else ""
        row["non_dma_exact"] = all(
            predicted[k] == c["actual"][k] for k in ("issue", "scalar", "sram", "arithmetic", "dependency")
        )
        rows.append(row)
    write_csv(output / "calibration.csv", rows)
    groups = {}
    for r in rows:
        key = tuple(
            r[k]
            for k in (
                "model",
                "batch",
                "tokens",
                "dot",
                "L",
                "dma_service",
                "phased",
                "broadcast",
                "resident",
                "port_cycles",
                "feedback",
            )
        )
        groups.setdefault(key, {})[r["control"]] = r
    pairs = []
    for key, g in groups.items():
        if "fsm" not in g:
            continue
        for baseline in ("old_isa", "row"):
            if baseline not in g:
                continue
            b, d = g[baseline], g["fsm"]
            sr = b["total_rust"] / d["total_rust"]
            sa = b["total_analytic"] / d["total_analytic"]
            error = abs(sa / sr - 1)
            pairs.append(
                dict(
                    model=b["model"],
                    batch=b["batch"],
                    tokens=b["tokens"],
                    L=b["L"],
                    dma_service=b["dma_service"],
                    dot=b["dot"],
                    baseline=baseline,
                    split="calibration" if b["split"] == d["split"] == "calibration" else "validation",
                    rust_speedup=sr,
                    analytic_speedup=sa,
                    speedup_relative_error=error,
                    within_2_percent=error <= 0.02,
                    within_5_percent=error <= 0.05,
                    effect_resolved=error * 3 < abs(sr - 1),
                    arithmetic_matched=baseline == "row",
                )
            )
    write_csv(output / "speedup_validation.csv", pairs)
    val = [r for r in rows if r["split"] == "validation"]
    pv = [r for r in pairs if r["split"] == "validation"]
    summary = dict(
        model_contract="ltile_r3_compositional_v1",
        memory_backend=backend.identity,
        predictor_sources={
            name: sha(Path(__file__).with_name(name))
            for name in ("ltile_cost.py", "ltile_program.py", "ltile_dma.py", "ltile_memory.cc")
        },
        cases=len(rows),
        unique_predicted_schedules=len(jobs),
        calibration_cases=len(train),
        validation_cases=len(val),
        total_error_mean=float(np.mean([r["total_relative_error"] for r in val])),
        total_error_max=max(r["total_relative_error"] for r in val),
        speedup_error_mean=float(np.mean([r["speedup_relative_error"] for r in pv])),
        speedup_error_max=max(r["speedup_relative_error"] for r in pv),
        dma_error_max=max(r["dma_relative_error"] for r in val),
        validation_non_dma_exact=sum(r["non_dma_exact"] for r in val),
        validation_pairs=len(pv),
        gate_passed=(
            all(r["non_dma_exact"] for r in rows)
            and all(r["within_5_percent"] for r in pv)
            and all(r["total_relative_error"] <= 0.05 and r["dma_relative_error"] <= 0.05 for r in val)
        ),
        source_counts=dict(Counter("E" if "/E/runs/" in str(c["path"]) else "E2_or_supplement" for c in cases)),
    )
    (output / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--memory-binary", type=Path, required=True)
    parser.add_argument("--memory-config", type=Path, required=True)
    parser.add_argument("--memory-cache", type=Path, required=True)
    args = parser.parse_args()
    backend = DmaBackend(args.memory_binary, args.memory_config, args.memory_cache) if args.memory_binary else None
    result = run(args.evidence.resolve(), args.output.resolve(), backend)
    if not result["gate_passed"]:
        raise SystemExit(1)
