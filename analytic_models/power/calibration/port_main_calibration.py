"""Port the research action-energy fit onto the unmodified main ISA domain."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from typing import Any


MAIN_VECTOR_FAMILIES = {
    "lane_add_sub_vf",
    "lane_add_sub_vv",
    "lane_movement_shift",
    "lane_multiply_vf",
    "lane_multiply_vv",
    "lane_sfu_exp",
    "lane_sfu_reciprocal",
    "reduction_max_full",
    "reduction_sum_full",
}
MAIN_SCALAR_FAMILIES = {
    "fp_add_sub_move",
    "fp_multiply",
    "fp_sfu_exp",
    "fp_sfu_reciprocal",
    "fp_sfu_sqrt",
    "integer_alu",
    "integer_multiply",
    "register_or_sram_access",
}
MAIN_HOLDOUT_KERNELS = {
    "hbm": {"matrix_prefetch", "vector_prefetch", "vector_writeback"},
    "matrix": {"array_compute"},
    "scalar": {
        "fp_alu",
        "fp_exp",
        "fp_mul",
        "fp_reciprocal",
        "fp_sqrt",
        "int_alu",
        "int_mul",
        "register_access",
    },
    "vector": {
        "add_vf",
        "add_vv",
        "exp",
        "mul_vf",
        "mul_vv",
        "reciprocal",
        "reduce_max",
        "reduce_sum",
        "shift",
    },
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty list")
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def port_logic(source: Path) -> dict[str, Any]:
    payload = _load(source)
    dynamic = payload["dynamic_nominal_pj"]
    dynamic["vector"] = {
        name: value for name, value in dynamic["vector"].items() if name in MAIN_VECTOR_FAMILIES
    }
    dynamic["scalar"] = {
        name: value for name, value in dynamic["scalar"].items() if name in MAIN_SCALAR_FAMILIES
    }
    allowed_envelopes = {
        "control.frontend_issue",
        "hbm_controller.matrix_prefetch",
        "hbm_controller.physical_transfer",
        "hbm_controller.vector_prefetch",
        "hbm_controller.vector_writeback",
        "matrix.array_compute.mxint",
        "matrix.array_compute.mxfp",
    }
    allowed_envelopes.update(f"vector.{name}" for name in MAIN_VECTOR_FAMILIES)
    allowed_envelopes.update(f"scalar.{name}" for name in MAIN_SCALAR_FAMILIES)
    payload["activity_envelope"] = {
        name: value
        for name, value in payload["activity_envelope"].items()
        if name in allowed_envelopes
    }
    payload.update(
        {
            "schema_version": "plena-main-action-energy-v1",
            "model": "main_compiler_action_energy_v1",
            "calibration_status": "rtl_activity_calibrated_candidate_main_compatible",
            "power_scope": "main ISA logic actions and on-chip HBM controller",
            "source_artifact_sha256": _hash(source),
            "port_exclusions": [
                "RTL-v3/v4/v5 segment and compact-stat operations",
                "loop AGU actions",
                "multi-chip and NVLink actions",
            ],
        }
    )
    payload.pop("source_points", None)
    payload["source_dataset"] = {
        "mapped_netlists": 31,
        "rtl_activity_replay_points": 395,
        "training_points": 290,
        "holdout_points": 105,
        "raw_points_committed": False,
    }
    payload["calibration_domain"]["main_vector_families"] = sorted(MAIN_VECTOR_FAMILIES)
    payload["calibration_domain"]["main_scalar_families"] = sorted(MAIN_SCALAR_FAMILIES)
    payload["provenance"]["main_port"] = (
        "family coefficients retained only where the unmodified main ISA exercises "
        "the same calibrated hardware datapath"
    )
    return payload


def port_validation(source: Path, logic_source: Path) -> dict[str, Any]:
    payload = _load(source)
    retained_points = [
        point
        for point in payload["holdout_points"]
        if point["microkernel"] in MAIN_HOLDOUT_KERNELS.get(point["component"], set())
    ]
    errors = [float(point["absolute_percentage_error"]) for point in retained_points]
    component_holdout = {}
    for component in sorted(MAIN_HOLDOUT_KERNELS):
        selected = [
            float(point["absolute_percentage_error"])
            for point in retained_points
            if point["component"] == component
        ]
        component_holdout[component] = {
            "count": len(selected),
            "median_ape": median(selected),
            "p95_ape": _percentile(selected, 0.95),
            "max_ape": max(selected),
        }
    return {
        "schema_version": "plena-main-power-validation-v1",
        "calibration_status": "rtl_activity_calibrated_candidate_main_compatible",
        "source_validation_sha256": _hash(source),
        "source_logic_sha256": _hash(logic_source),
        "gate_level_validation": payload.get("gate_level_validation"),
        "holdout_point_count": len(retained_points),
        "holdout_median_ape": median(errors),
        "holdout_p95_ape": _percentile(errors, 0.95),
        "component_holdout": component_holdout,
        "retained_holdout_points": retained_points,
        "source_qwen_mix_median_ape": median(payload["qwen_mix_errors"]) * 100,
        "source_qwen_mix_max_ape": max(payload["qwen_mix_errors"]) * 100,
        "minimum_action_slope_r2": payload["minimum_action_slope_r2"],
        "clock_holdout_median_ape": payload["clock_holdout_median_ape"],
        "clock_holdout_p95_ape": payload["clock_holdout_p95_ape"],
        "cached_power_evaluation_median_ms": payload["cached_power_evaluation_median_ms"],
        "matrix_invariants": payload["matrix_invariants"],
        "port_scope": {
            "retained_vector_families": sorted(MAIN_VECTOR_FAMILIES),
            "retained_scalar_families": sorted(MAIN_SCALAR_FAMILIES),
            "workload_action_mix_validation": "generated separately by validate_main_workloads.py",
        },
        "limitations": [
            "RTL activity is replayed on mapped DC netlists; gate-level simulation was not run",
            "holdout errors validate retained hardware families, not an end-to-end chip power measurement",
            "large MLEN/VLEN action energy is structural extrapolation beyond VLEN=64 calibration points",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic-source", type=Path, required=True)
    parser.add_argument("--validation-source", type=Path, required=True)
    parser.add_argument("--logic-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    args.logic_output.write_text(json.dumps(port_logic(args.logic_source), indent=2, sort_keys=True) + "\n")
    args.validation_output.write_text(
        json.dumps(port_validation(args.validation_source, args.logic_source), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
