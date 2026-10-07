"""Explore software projection loops and layouts on bounded M_MV/M_TMV.

No replay, M_MM.P, S4, slice hardware or measured-cycle fitting. This is a
restricted, executed-program search, not a proof of optimal original PLENA.
Legacy M_MM and deferred PE/tree reduction require separate calibration.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import get_context
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .ltile_dma import DmaBackend
from .ltile_layers import compiler_api
from .ltile_platform import ExecutionProfile
from .ltile_services import Services
from .projection_campaign import write_csv


def software_profile(controllers, request_tile, vector_rows):
    if vector_rows not in (58, 64):
        raise ValueError("software study uses 58 or 64 existing Vector rows")
    return ExecutionProfile(
        hbm_controllers=controllers, projection_schedule="resident",
        projection_request_tile=request_tile, projection_vector_rows=vector_rows,
        projection_codec_rows=6 if vector_rows == 58 else 0,
    )


def summarize(output, reference=None):
    """Derive comparisons from complete, hashed cases; never add stage minima."""
    output = Path(output)
    reference = Path(reference) if reference is not None else output
    manifest = json.loads((output / "manifest.json").read_text())
    if not (output / "paired.csv").is_file():
        raise ValueError("comparison requires a completed --paired campaign")
    selected = list(csv.DictReader((output / "selected.csv").open()))
    indexed = {}
    for folder in {output, reference}:
        for name in ("search.csv", "paired.csv"):
            for row in csv.DictReader((folder / name).open()):
                indexed[folder / row["result"]] = row
    comparisons, operators, stages, evidence = [], [], [], {}
    exclusive = ("issue", "scalar", "sram", "arithmetic", "dependency", "dma")

    def read(path):
        name = path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != indexed[path]["sha256"]:
            raise ValueError(f"case hash differs from its table: {name}")
        result = json.loads(path.read_text())
        parts = result["components"]
        if sum(parts[k] for k in exclusive) != parts["total"]:
            raise ValueError(f"exclusive costs do not reconcile: {name}")
        for s in result["sections"]:
            if sum(s[k] for k in exclusive) != s["total"]:
                raise ValueError(f"operator costs do not reconcile: {name}/{s['name']}")
        if sum(s["total"] for s in result["sections"]) != parts["total"]:
            raise ValueError(f"stages do not cover the program: {name}")
        for counter in ("hbm_read_bytes", "hbm_write_bytes"):
            if sum(s[counter] for s in result["sections"]) != result[counter]:
                raise ValueError(f"stage traffic differs: {name}/{counter}")
        evidence[str(path.resolve())] = digest
        return result

    for choice in selected:
        m, b = choice["model"], int(choice["batch"])
        q, v = int(choice["request_tile"]), int(choice["vector_rows"])
        names = {
            "previous_projection_vector": reference / f"{m}_b{b}_q{b}_v58_old_isa.json",
            "selected_projection_vector": output / f"{m}_b{b}_q{q}_v{v}_old_isa.json",
            "previous_projection_ltile": reference / f"{m}_b{b}_q{b}_v58_fsm.json",
            "selected_projection_ltile": output / choice["result"],
        }
        results = {arm: read(name) for arm, name in names.items()}
        summary = dict(model=m, batch=b, request_tile=q, vector_rows=v)
        for arm, result in results.items():
            summary[arm + "_cycles"] = result["components"]["total"]
            summary[arm + "_ms"] = result["components"]["total"] * 1000 / manifest["clock_hz"]
            summary[arm + "_hbm_bytes"] = result["hbm_read_bytes"] + result["hbm_write_bytes"]
            categories = {s["name"]: s["category"] for s in result["metadata"]["stages"]}
            grouped = {}
            for s in result["sections"]:
                category = categories[s["name"]]
                group = ("output_projection" if s["name"] == "output_projection" else
                         "input_and_gate_projection" if category == "projection" else category)
                operators.append(dict(model=m, batch=b, arm=arm, category=category, **s))
                target = grouped.setdefault(group, {k: 0 for k in (*exclusive, "total", "hbm_read_bytes", "hbm_write_bytes")})
                for key in target:
                    target[key] += s[key]
            for group, values in grouped.items():
                stages.append(dict(model=m, batch=b, arm=arm, stage=group, **values))
        new = summary["selected_projection_ltile_cycles"]
        summary.update(
            compiler_speedup_recurrence_fixed=summary["previous_projection_ltile_cycles"] / new,
            recurrent_extension_speedup_projection_fixed=summary["selected_projection_vector_cycles"] / new,
            combined_speedup=summary["previous_projection_vector_cycles"] / new,
        )
        comparisons.append(summary)
    for name, rows in (("comparison.csv", comparisons), ("stages.csv", stages), ("operators.csv", operators)):
        write_csv(output / name, rows)
    (output / "comparison_sources.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return comparisons


def execute_case(spec, *, args):
    compiler_api(args.compiler)
    memory, output, cache = args.memory_root, args.output, args.cache_root
    controllers, source = args.controllers, args.source_sha256
    model, batch, tile, rows, control = spec
    profile = software_profile(controllers, tile, rows)
    if args.transposed_k:
        profile = replace(profile, projection_schedule="transposed", projection_request_tile=16,
                          projection_k_tile=args.transposed_k)
    assert not profile.matrix.weight_replay and profile.matrix.projection_segments == 1
    backend = DmaBackend(memory / "ltile_memory", memory / "ramulator.json", cache / "memory")
    service = Services(args.compiler, profile, backend, cache / "services")
    result = service.layer(model, batch, control, "BF16",
                           supply=args.coefficient_supply if control == "fsm" else "packed")
    name = f"{model}_b{batch}_q{tile}_v{rows}_{control}"
    path = output / f"{name}.json"
    # Large memory event lists stay in the reproducible cache; public
    # per-case evidence is the actual program/source hashes and stages.
    result["software_study"] = dict(source_sha256=source, extra_projection_storage_bytes=0,
        original_plena_eligible=False, restriction="bounded Matrix service; rectangular views are a common substrate")
    path.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    row = dict(model=model, batch=batch, request_tile=tile, vector_rows=rows,
               control=control, **result["components"],
               hbm_read_bytes=result["hbm_read_bytes"], hbm_write_bytes=result["hbm_write_bytes"],
               result=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    categories = {s["name"]: s["category"] for s in result["metadata"]["stages"]}
    row["projection_cycles"] = sum(s["total"] for s in result["sections"] if categories[s["name"]] == "projection")
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiler", type=Path, default=Path(__file__).resolve().parents[2] / "PLENA_Compiler")
    p.add_argument("--memory-root", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cache-root", type=Path,
                   help="Reusable content-addressed cache; output evidence is always regenerated")
    p.add_argument("--models", nargs="+", choices=("mamba", "kda"), default=["mamba", "kda"])
    p.add_argument("--batches", nargs="+", type=int, choices=(1, 2, 4, 8, 16), default=[1, 2, 4, 8, 16])
    p.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument("--paired", action="store_true", help="Also execute old-Vector recurrence with the selected projection")
    p.add_argument("--reference-pairs", action="store_true",
                   help="With --paired, collect only the historical full-batch/58-row Vector reference")
    p.add_argument("--coefficient-supply", choices=("native", "packed"), default="native",
                   help="Hold the FSM/arithmetic fixed and compare compact native supply with software packing")
    p.add_argument("--resume", action="store_true", help="Resume an incomplete output using source-bound operator caches")
    p.add_argument("--report-only", action="store_true", help="Verify a completed paired campaign and write comparison/stage tables")
    p.add_argument("--reference-output", type=Path, help="Completed resident campaign for a transposed comparison")
    p.add_argument("--transposed-k", type=int, choices=(256, 512, 1024),
                   help="Existing M_TMV with offline-transposed weights; larger K changes reduction order")
    args = p.parse_args(argv)
    if args.reference_pairs and not args.paired:
        p.error("--reference-pairs requires --paired")
    if args.report_only:
        summarize(args.output, args.reference_output)
        return
    if args.memory_root is None:
        p.error("--memory-root is required when evaluating programs")
    compiler_api(args.compiler.resolve())
    output, memory = args.output.resolve(), args.memory_root.resolve()
    if args.resume and (output / "manifest.json").exists():
        raise ValueError("completed campaigns are immutable; choose a new output")
    output.mkdir(parents=True, exist_ok=args.resume)
    cache = args.cache_root.resolve() if args.cache_root else output / "cache"
    controllers = len(json.loads((memory / "ramulator.json").read_text())["memory_system"]["controllers"])
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    args.memory_root, args.output, args.cache_root = memory, output, cache
    args.compiler = args.compiler.resolve()
    args.controllers, args.source_sha256 = controllers, source
    execute = partial(execute_case, args=args)

    # Different budgets can yield identical traces. Evaluate them in separate
    # passes so the second pass reuses the content-addressed memory cache.
    cases = [(m, b, t, v, "fsm") for v in (58, 64) for m in args.models for b in args.batches
             for t in (1, 2, 4, 8, 16) if t <= b]
    if args.transposed_k:
        cases = [(m,b,b,58,"fsm") for m in args.models for b in args.batches]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        rows = []
        for budget in sorted({case[3] for case in cases}):
            rows.extend(pool.map(execute, [case for case in cases if case[3] == budget]))
    write_csv(output / "search.csv", rows)
    selected = [min((r for r in rows if r["model"] == m and r["batch"] == b),
                    key=lambda r: (r["total"], -r["request_tile"], r["vector_rows"]))
                for m in args.models for b in args.batches]
    if args.paired:
        # Both recurrence arms receive the same selected projection schedule.
        # Keep the historical software schedule as a third, separate arm.
        additional = set()
        for r in selected:
            if not args.reference_pairs:
                additional.add((r["model"], r["batch"], r["request_tile"], r["vector_rows"], "old_isa"))
            additional.add((r["model"], r["batch"], r["batch"], 58, "old_isa"))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            paired = list(pool.map(execute, sorted(additional)))
        write_csv(output / "paired.csv", paired)
    write_csv(output / "selected.csv", selected)
    manifest = dict(
        source_sha256=source, cases=len(rows), weight="BF16", clock_hz=10**9,
        projection_hardware="existing M_TMV/M_MV; no replay/slicing/segmentation",
        projection_k_tile=args.transposed_k or 256,
        coefficient_supply=args.coefficient_supply,
        reference_pairs_only=args.reference_pairs,
        transposed_static_weights=bool(args.transposed_k),
        extra_projection_storage_bytes=0, recurrence="BF16 ordinary Vector vs FP32 update/BF16 tree L_TILE",
        full_model=False, original_plena_eligible=False, numerical_validation="separate machine-code checks",
        scope="input norm through recurrent output projection; excludes residual and MoE",
        limitations=["M_MM not cycle-calibrated", "K grouping is an explicit BF16 rounding contract", "no runtime NVFP4",
                     "no inter-instruction overlap", "finite search, not globally optimal"],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if args.paired and not args.reference_pairs and (not args.transposed_k or args.reference_output):
        summarize(output, args.reference_output)
    print(f"Completed {len(rows)} software candidates; {len(selected)} selected cases", flush=True)


if __name__ == "__main__":
    main()
