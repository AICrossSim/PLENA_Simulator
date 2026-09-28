"""Reproduce compiled projection/sublayer ablations without local campaign paths.

Predictions use emitted instructions and Ramulator, never observed Rust cycles.
The historical baseline is a schedule on the common extended platform, not an
unmodified PLENA implementation. Every case retains its arithmetic and profile.
"""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .ltile_dma import DmaBackend
from .ltile_platform import ExecutionProfile
from .ltile_services import Services


STAGES = {
    "baseline": ("resident", False),
    "replay": ("resident", True),
    "compact": ("compact", True),
    "batch": ("batch", True),
}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, default=Path(__file__).resolve().parents[2] / "PLENA_Compiler")
    parser.add_argument("--memory-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing results are never overwritten"
    )
    parser.add_argument("--models", nargs="+", choices=("mamba", "kda"), default=["mamba", "kda"])
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 2, 4, 8, 16), default=[1, 2, 4, 8, 16])
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), default=list(STAGES))
    parser.add_argument("--control", choices=("fsm", "old_isa"), default="fsm")
    args = parser.parse_args(argv)
    compiler = args.compiler.resolve()
    memory = args.memory_root.resolve()
    if not (compiler / "aten/plena/ltile_native.py").is_file():
        parser.error("initialize the pinned Compiler submodule or pass --compiler")
    if not (memory / "ltile_memory").is_file() or not (memory / "ramulator.json").is_file():
        parser.error("--memory-root must contain the prepared ltile_memory and ramulator.json")
    for name in ("models", "batches", "stages"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"--{name} must not repeat cases")
    controllers = len(json.loads((memory / "ramulator.json").read_text())["memory_system"]["controllers"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    backend = DmaBackend(memory / "ltile_memory", memory / "ramulator.json", output / "cache/memory")
    rows, sections, records = [], [], []
    for stage in args.stages:
        schedule, replay = STAGES[stage]
        profile = replace(ExecutionProfile(), hbm_controllers=controllers, projection_schedule=schedule)
        profile = replace(profile, matrix=replace(profile.matrix, weight_replay=replay))
        services = Services(compiler, profile, backend, output / "cache/services")
        for model in args.models:
            for batch in args.batches:
                result = services.layer(
                    model, batch, args.control, "BF16", supply="native" if args.control == "fsm" else "packed"
                )
                case = dict(model=model, batch=batch, stage=stage, control=args.control)
                path = output / f"{model}_b{batch}_{stage}.json"
                path.write_text(json.dumps(dict(case=case, **result), indent=2) + "\n")
                components = result["components"]
                rows.append(
                    dict(
                        **case,
                        **components,
                        total_ms=components["total"] * 1000 / profile.machine.clock_hz,
                        hbm_read_bytes=result["hbm_read_bytes"],
                        hbm_write_bytes=result["hbm_write_bytes"],
                        profile_sha256=result["profile_sha256"],
                        assembly_sha256=result["assembly_sha256"],
                    )
                )
                categories = {s["name"]: s["category"] for s in result["metadata"]["stages"]}
                sections.extend(dict(**case, category=categories[s["name"]], **s) for s in result["sections"])
                records.append(dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    baselines = {(row["model"], row["batch"]): row["total"] for row in rows if row["stage"] == "baseline"}
    for row in rows:
        baseline = baselines.get((row["model"], row["batch"]))
        row["speedup_vs_schedule_baseline"] = baseline / row["total"] if baseline is not None else ""
    write_csv(output / "summary.csv", rows)
    write_csv(output / "operators.csv", sections)
    manifest = dict(
        scope="one recurrent sublayer: input norm through output projection; excludes outer residual and MoE",
        evidence="compiled analytical prediction; not a new numerical or GPU observation",
        original_plena_eligible=False,
        clock_hz=10**9,
        hbm_controllers=controllers,
        weight_format="BF16",
        control=args.control,
        memory=backend.identity,
        cases=records,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note="Each case records source hashes and resources. DMA is already included in operator totals.",
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} cases to {output}")


if __name__ == "__main__":
    main()
