"""Command-line interface for the resumable RunPod A100 benchmark campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aggregate import aggregate_campaign
from .inventory import write_inventory
from .manifest import load_manifest
from .runner import run_formal, run_preflight, run_replica_check


DEFAULT_MANIFEST = Path(__file__).with_name("manifests") / "runpod_a100_awq_v1.yaml"


def _csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _csv_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise argparse.ArgumentTypeError("GPU pool must contain unique non-negative integer IDs")
    return values


def _manifest_summary(path: Path) -> dict[str, object]:
    manifest = load_manifest(path)
    return {
        "campaign": manifest.campaign,
        "manifest_hash": manifest.fingerprint,
        "formal_points": len(manifest.formal_points),
        "preflight_points": len(manifest.preflight_points),
        "models": sorted(manifest.models),
        "workloads": sorted(manifest.workloads),
        "engine_configurations": len(
            {
                (
                    point.model_name,
                    point.tensor_parallel_size,
                    point.context_profile,
                    point.gpu_ids,
                )
                for point in manifest.formal_points
            }
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest", help="validate and expand the campaign manifest")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    inventory = subparsers.add_parser("inventory", help="capture GPU, topology, software, and RunPod metadata")
    inventory.add_argument("--output", type=Path, required=True)
    inventory.add_argument("--no-validate", action="store_true")

    preflight = subparsers.add_parser("preflight", help="select a common backend and freeze the environment")
    preflight.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    preflight.add_argument("--inventory", type=Path, required=True)
    preflight.add_argument("--output-root", type=Path, required=True)
    preflight.add_argument("--environment-lock", type=Path, required=True)
    preflight.add_argument("--image-digest", required=True)

    run = subparsers.add_parser("run", help="execute or resume formal points")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--environment-lock", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--image-digest", required=True)
    run.add_argument("--models", help="comma-separated manifest model names")
    run.add_argument("--workloads", help="comma-separated workload names")
    run.add_argument("--point-ids", help="comma-separated exact point IDs")
    run.add_argument(
        "--measurement-stage",
        choices=("screening", "confirmation", "short-sweep", "holdout"),
        default="screening",
    )
    run.add_argument(
        "--execution-mode",
        choices=("auto", "gpu-parallel", "sequential"),
        default="auto",
        help="auto parallelizes screening/short sweep and isolates confirmation/holdout",
    )
    run.add_argument(
        "--physical-gpu-pool",
        type=_csv_int_tuple,
        default=tuple(range(8)),
        help="comma-separated physical GPUs available to the scheduler (default: 0..7)",
    )
    run.add_argument("--max-concurrent-engines", type=int, default=8)

    aggregate = subparsers.add_parser("aggregate", help="validate repetitions and produce CSV/JSON summaries")
    aggregate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--allow-missing", action="store_true")

    replica = subparsers.add_parser("replica-check", help="measure synchronized concurrent replicas")
    replica.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    replica.add_argument("--environment-lock", type=Path, required=True)
    replica.add_argument("--formal-output-root", type=Path, required=True)
    replica.add_argument("--output-root", type=Path, required=True)
    replica.add_argument("--point-id", required=True)
    replica.add_argument("--image-digest", required=True)
    replica.add_argument("--gpu-groups", default="0,1,2,3:4,5,6,7")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-manifest":
        print(json.dumps(_manifest_summary(args.manifest), indent=2, sort_keys=True))
        return 0
    if args.command == "inventory":
        inventory = write_inventory(args.output, validate=not args.no_validate)
        print(json.dumps(inventory["validation"], indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        lock = run_preflight(
            manifest=load_manifest(args.manifest),
            inventory_path=args.inventory,
            output_root=args.output_root,
            environment_lock_path=args.environment_lock,
            image_digest=args.image_digest,
        )
        print(json.dumps(lock, indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        outcomes = run_formal(
            manifest=load_manifest(args.manifest),
            environment_lock_path=args.environment_lock,
            output_root=args.output_root,
            image_digest=args.image_digest,
            models=_csv_set(args.models),
            workloads=_csv_set(args.workloads),
            point_ids=_csv_set(args.point_ids),
            measurement_stage=args.measurement_stage,
            execution_mode=args.execution_mode,
            physical_gpu_pool=args.physical_gpu_pool,
            max_concurrent_engines=args.max_concurrent_engines,
        )
        print(json.dumps(outcomes, indent=2, sort_keys=True))
        return 0
    if args.command == "aggregate":
        report = aggregate_campaign(
            manifest=load_manifest(args.manifest),
            output_root=args.output_root,
            allow_missing=args.allow_missing,
        )
        print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
        return 0
    if args.command == "replica-check":
        groups = tuple(
            tuple(int(gpu) for gpu in group.split(","))
            for group in args.gpu_groups.split(":")
        )
        result = run_replica_check(
            manifest=load_manifest(args.manifest),
            environment_lock_path=args.environment_lock,
            formal_output_root=args.formal_output_root,
            output_root=args.output_root,
            point_id=args.point_id,
            gpu_groups=groups,
            image_digest=args.image_digest,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)
