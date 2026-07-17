#!/usr/bin/env python3
"""Aggregate generic and arrival-aware HBM V4 promotion evidence.

The output deliberately separates model-fit acceptance from system replay.
V4 is promotable only when the generic holdout and every required Qwen case
pass their own gates with a converged rtl-v1 arrival/service fixed point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_CASES = {
    "qwen3_32b_seq482_m512_b64_mxfp_e4m3_b8_c128",
    "qwen3_8b_seq64_m128_b16_mxint4_b64_c128",
    "qwen3_8b_seq128_m256_b32_mxint8_b64_c32",
    "qwen3_32b_seq128_m256_b64_mxfp_e1m2_b64_c8",
}


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative evidence path when one is available."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved)


def _case_argument(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case must be NAME=PATH") from exc
    if not name:
        raise argparse.ArgumentTypeError("case name cannot be empty")
    return name, Path(path)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _case_summary(name: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    replay = payload.get("arrival_replay", {})
    iterations = tuple(replay.get("iterations", ()))
    request_parity = bool(iterations) and all(
        all(bool(value) for value in row.get("request_parity", {}).values())
        for row in iterations
    )
    return {
        "name": name,
        "source": _portable_path(path),
        "accepted": bool(payload.get("accepted", False)),
        "acceptance": dict(payload.get("acceptance", {})),
        "calibration_id": payload.get("calibration_id"),
        "dma_semantic_version": payload.get("dma_semantic_version"),
        "observed_dma_timing_semantics": payload.get(
            "observed_dma_timing_semantics"
        ),
        "arrival_replay_converged": bool(replay.get("converged", False)),
        "arrival_replay_iterations": len(iterations),
        "arrival_replay_convergence_definition": replay.get(
            "convergence_definition"
        ),
        "arrival_replay_convergence_cycle_tolerances": dict(
            replay.get("convergence_cycle_tolerances", {})
        ),
        "arrival_replay_request_parity": request_parity,
        "initial_trace_used_as_seed_only": bool(
            replay.get("initial_trace_used_as_seed_only", False)
        ),
        "opcode_work": dict(payload.get("opcode_work", {})),
        "total_hbm_work": dict(payload.get("total_hbm_work", {})),
        "scheduled_makespan": {
            key: value
            for key, value in payload.get("scheduled_makespan", {}).items()
            if key
            in {
                "absolute_error_percent",
                "observed_cycles",
                "predicted_cycles",
                "observed_status",
                "predicted_status",
            }
        },
        "correctness_gate_modified": bool(
            payload.get("correctness_gate_modified", True)
        ),
        "numerical_execution_modified": bool(
            payload.get("numerical_execution_modified", True)
        ),
        "model_domain": list(payload.get("model_domain", ())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generic-validation", type=Path, required=True)
    parser.add_argument(
        "--case",
        type=_case_argument,
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        help="candidate model to mark promoted after every acceptance gate passes",
    )
    parser.add_argument(
        "--promoted-model-output",
        type=Path,
        help="destination for the accepted model with promotion metadata",
    )
    args = parser.parse_args()
    if (args.model is None) != (args.promoted_model_output is None):
        raise ValueError("--model and --promoted-model-output must be provided together")

    case_paths = dict(args.case)
    missing = sorted(REQUIRED_CASES - case_paths.keys())
    unexpected = sorted(case_paths.keys() - REQUIRED_CASES)
    if missing or unexpected:
        raise ValueError(
            f"system case set mismatch: missing={missing}, unexpected={unexpected}"
        )

    generic = _load(args.generic_validation)
    cases = [
        _case_summary(name, case_paths[name], _load(case_paths[name]))
        for name in sorted(REQUIRED_CASES)
    ]
    calibration_ids = {
        str(item["calibration_id"])
        for item in cases
        if item["calibration_id"] is not None
    }
    calibration_ids.add(str(generic.get("calibration_id")))
    calibration_consistent = len(calibration_ids) == 1
    requests_accepted = int(generic.get("request_manifest_mismatches", -1)) == 0
    all_system_cases_accepted = all(item["accepted"] for item in cases)
    all_replays_converged = all(
        item["arrival_replay_converged"] for item in cases
    )
    all_system_request_parity = all(
        item["arrival_replay_request_parity"] for item in cases
    )
    timing_only = all(
        not item["correctness_gate_modified"]
        and not item["numerical_execution_modified"]
        for item in cases
    )
    acceptance = {
        "generic_holdout_accepted": bool(generic.get("accepted", False)),
        "request_manifest_parity": requests_accepted,
        "all_required_system_cases_present": not missing and not unexpected,
        "all_system_cases_accepted": all_system_cases_accepted,
        "all_arrival_replays_converged": all_replays_converged,
        "all_system_request_manifests_match": all_system_request_parity,
        "calibration_id_consistent": calibration_consistent,
        "correctness_and_numerical_paths_unchanged": timing_only,
    }
    accepted = all(acceptance.values())
    output = {
        "schema_version": 4,
        "model": "production_dma_hbm_service_v4",
        "promotion_status": "accepted" if accepted else "candidate",
        "accepted": accepted,
        "calibration_id": next(iter(calibration_ids))
        if calibration_consistent
        else None,
        "acceptance": acceptance,
        "generic_validation": {
            "source": _portable_path(args.generic_validation),
            "accepted": bool(generic.get("accepted", False)),
            "overall_holdout": dict(generic.get("overall_holdout", {})),
            "store_holdout_p95_absolute_error_percent": generic.get(
                "store_holdout_p95_absolute_error_percent"
            ),
            "request_manifest_mismatches": generic.get(
                "request_manifest_mismatches"
            ),
        },
        "system_cases": cases,
        "scope": {
            "memory_model": "post_hoc production-DMA completion surrogate",
            "ramulator": "HBM2_2Gbps + MOP4CLXOR",
            "correctness_gate_modified": False,
            "online_cross_queue_ramulator_cosimulation": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered)
    if args.model is not None:
        if not accepted:
            raise ValueError("refusing to promote a V4 model with failed acceptance gates")
        model = _load(args.model)
        if model.get("calibration_id") != output["calibration_id"]:
            raise ValueError(
                "candidate model calibration ID differs from validation evidence"
            )
        metadata = model.setdefault("metadata", {})
        metadata.update(
            {
                "promotion_status": "accepted",
                "promotion_evidence": _portable_path(args.output),
                "promotion_evidence_sha256": hashlib.sha256(
                    rendered.encode()
                ).hexdigest(),
                "promotion_scope": (
                    "post_hoc production-DMA completion shadow; not online "
                    "cross-queue Ramulator co-simulation"
                ),
            }
        )
        args.promoted_model_output.parent.mkdir(parents=True, exist_ok=True)
        args.promoted_model_output.write_text(
            json.dumps(model, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"accepted": accepted, "output": str(args.output)}, indent=2))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
