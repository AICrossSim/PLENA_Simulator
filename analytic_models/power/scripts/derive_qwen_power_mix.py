#!/usr/bin/env python3
"""Derive the RTL-activity mixed-kernel histogram from a compiled CostTrace.

The output is deliberately independent of opcode latency and energy
coefficients. It records only the hardware action mix emitted by the compiler,
so the 128-action validation window can follow a real Qwen lowering instead of
using a hand-written average operation sequence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import importlib
import json
from pathlib import Path
import pickle
from typing import Any


ACTION_TO_MICROKERNEL = {
    "lane_add_sub_vv": "add_vv",
    "lane_add_sub_vf": "add_vf",
    "lane_add_sub_vseg": "add_vseg",
    "lane_multiply_vv": "mul_vv",
    "lane_multiply_vf": "mul_vf",
    "lane_multiply_vseg": "mul_vseg",
    "lane_sfu_exp": "exp",
    "lane_sfu_reciprocal": "reciprocal",
    "reduction_sum_full": "reduce_sum",
    "reduction_max_full": "reduce_max",
    "reduction_sum_segment": "reduce_sum_seg",
    "reduction_max_segment": "reduce_max_seg",
    "reduction_sum_segments": "reduce_sum_segs",
    "reduction_max_segments": "reduce_max_segs",
    "lane_movement_shift": "shift",
    "fp_add_sub_move": "fp_alu",
    "fp_multiply": "fp_mul",
    "fp_sfu_exp": "fp_exp",
    "fp_sfu_reciprocal": "fp_reciprocal",
    "fp_sfu_sqrt": "fp_sqrt",
    "fp_sfu_rsqrt": "fp_rsqrt",
    "integer_alu": "int_alu",
    "integer_multiply": "int_mul",
    "register_or_sram_access": "register_access",
    "matrix_prefetch": "matrix_prefetch",
    "vector_prefetch": "vector_prefetch",
    "vector_writeback": "vector_writeback",
    "frontend_issue": "frontend_issue",
}


def _load_trace(path: Path) -> Any:
    # Historical caches used both ``compiler.aten`` and top-level ``aten``.
    # Import both aliases before unpickling, then use the implementation from
    # the trace's own module so ScheduleNode identity checks remain valid.
    for module in ("compiler.aten.cost_emitter", "aten.cost_emitter"):
        try:
            importlib.import_module(module)
        except ImportError:
            pass
    with path.open("rb") as handle:
        return pickle.load(handle)


def derive(path: Path) -> dict[str, Any]:
    trace = _load_trace(path)
    module = importlib.import_module(type(trace).__module__)
    actions = module._build_energy_actions(trace)
    raw: dict[str, Counter[str]] = defaultdict(Counter)
    microkernels: dict[str, Counter[str]] = defaultdict(Counter)
    exclusions: Counter[str] = Counter()

    for action in actions:
        component = str(action.component)
        if component in {"matrix", "vector", "scalar", "control", "hbm_controller"}:
            raw[component][f"{action.action}:{action.precision}"] += int(action.count)
        if component == "matrix":
            if action.action == "array_compute":
                microkernels["matrix"]["array_compute"] += int(action.count)
            else:
                # Reduction and conversion use the separately measured leaf
                # bundle; a mini-array harness cannot reproduce those paths.
                exclusions[f"matrix.{action.action}"] += int(action.count)
            continue
        if component == "scalar" and action.action == "vector_lane_access":
            target = "lane_load" if action.precision == "S_LD_VLANE_FP" else "lane_store"
            microkernels["vector"][target] += int(action.count)
            continue
        component_key = "hbm" if component == "hbm_controller" else component
        target = ACTION_TO_MICROKERNEL.get(str(action.action))
        if target is not None:
            microkernels[component_key][target] += int(action.count)

    metadata = dict(getattr(trace, "metadata", {}))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "costtrace_hardware_action_histogram_v1",
        "source_trace": str(path),
        "source_trace_sha256": sha256(path.read_bytes()).hexdigest(),
        "source_compiler_revision": metadata.get("compiler_revision", "unknown"),
        "workload": metadata.get("workload", {}),
        "hardware": metadata.get("hardware", {}),
        "schedule": {
            "vector_scalar_schedule": metadata.get("vector_scalar_schedule"),
            "packed_attention_schedule": metadata.get("packed_attention_schedule"),
            "gqa_pipeline_schedule": metadata.get("gqa_pipeline_schedule"),
        },
        "components": {
            component: {
                "microkernel_weights": dict(sorted(weights.items())),
                "source_action_count": sum(weights.values()),
            }
            for component, weights in sorted(microkernels.items())
        },
        "raw_action_counts": {
            component: dict(sorted(counts.items())) for component, counts in sorted(raw.items())
        },
        "excluded_from_component_mix": dict(sorted(exclusions.items())),
    }
    semantic = json.dumps(
        {
            "algorithm": payload["algorithm"],
            "workload": payload["workload"],
            "hardware": payload["hardware"],
            "schedule": payload["schedule"],
            "components": payload["components"],
            "excluded": payload["excluded_from_component_mix"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    payload["semantic_hash"] = sha256(semantic).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = derive(args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"semantic_hash": payload["semantic_hash"], "components": payload["components"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
