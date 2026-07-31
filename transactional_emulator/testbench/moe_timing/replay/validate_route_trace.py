#!/usr/bin/env python3
"""Validate a PLENA MoE route trace contract without external dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.moe_timing.replay.utils import finite_number, load_json, write_json


def _require_mapping(parent: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return {}
    return value


#: Schema versions this validator accepts.
#:
#: v1 describes routed-only MoE. v2 adds the optional shared-expert fields; a v1
#: trace is a valid v2 trace with no shared branch, so both are read here rather
#: than forcing every existing trace to be regenerated.
SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})

#: The newest schema version this validator implements. Reported as
#: ``schema_version`` in the summary it writes; the version the trace itself
#: declared is reported separately as ``trace_schema_version``.
CURRENT_SCHEMA_VERSION = 2

#: Shared-expert gate policies. ``"none"`` covers DeepSeek-V2/V3, Kimi K2,
#: Llama-4 and GLM-4.5; ``"sigmoid"`` is Qwen2-MoE's ``shared_expert_gate``.
SHARED_GATE_POLICIES = frozenset({"none", "sigmoid"})


def _validate_shared_expert(model: dict[str, Any], errors: list[str]) -> None:
    """Check the optional v2 shared-expert block.

    Absent means "no shared expert", which is correct for GPT-OSS, Qwen3-MoE and
    Mixtral. Present means all three fields must agree: a trace claiming a shared
    branch but giving it zero width would replay as routed-only while still being
    labelled as shared, and the timing split would silently attribute nothing.
    """
    shared_experts = model.get("shared_experts")
    if shared_experts is None:
        for field in ("shared_intermediate_size", "shared_gate"):
            if model.get(field) is not None:
                errors.append(f"model.{field} is set but model.shared_experts is absent")
        return

    if not isinstance(shared_experts, int) or shared_experts < 0:
        errors.append("model.shared_experts must be a non-negative integer")
        return
    if shared_experts == 0:
        # Explicit zero is allowed (it states "this architecture has none"), but
        # then the other fields must not claim otherwise.
        if model.get("shared_intermediate_size"):
            errors.append("model.shared_intermediate_size is non-zero but shared_experts is 0")
        return

    shared_intermediate = model.get("shared_intermediate_size")
    if not isinstance(shared_intermediate, int) or shared_intermediate <= 0:
        errors.append("model.shared_intermediate_size must be a positive integer when shared_experts > 0")

    gate = model.get("shared_gate", "none")
    if gate not in SHARED_GATE_POLICIES:
        errors.append(f"model.shared_gate must be one of {sorted(SHARED_GATE_POLICIES)}, got {gate!r}")


def validate_trace(trace: dict[str, Any], *, allow_missing_artifacts: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(trace, dict):
        return ["trace must be a JSON object"]
    schema_version = trace.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(f"schema_version must be one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}, got {schema_version!r}")
    for key in ("trace_id", "created_by"):
        if not isinstance(trace.get(key), str) or not trace.get(key):
            errors.append(f"{key} must be a non-empty string")

    model = _require_mapping(trace, "model", errors)
    workload = _require_mapping(trace, "workload", errors)
    routing = _require_mapping(trace, "routing", errors)
    artifacts = _require_mapping(trace, "artifacts", errors)
    replay = _require_mapping(trace, "replay", errors)

    top_k = model.get("top_k")
    num_experts = model.get("num_experts")
    token_count = workload.get("token_count")
    for scope, fields in (
        (model, ("layer_index", "hidden_size", "intermediate_size", "num_experts", "top_k")),
        (workload, ("batch_size", "seq_len", "token_count")),
        (replay, ("mlen", "blen", "emu_threads")),
    ):
        for field in fields:
            # layer_index may be 0 (its >= 0 bound is checked separately below);
            # every other field must be strictly positive.
            if not isinstance(scope.get(field), int) or (scope[field] <= 0 and field != "layer_index"):
                errors.append(f"{field} must be a positive integer")
    if isinstance(model.get("layer_index"), int) and model["layer_index"] < 0:
        errors.append("layer_index must be >= 0")
    _validate_shared_expert(model, errors)

    topk_indices = routing.get("topk_indices")
    topk_weights = routing.get("topk_weights")
    if not isinstance(topk_indices, list):
        errors.append("routing.topk_indices must be an array")
        topk_indices = []
    if not isinstance(topk_weights, list):
        errors.append("routing.topk_weights must be an array")
        topk_weights = []
    if isinstance(token_count, int) and len(topk_indices) != token_count:
        errors.append(f"topk_indices row count {len(topk_indices)} != token_count {token_count}")
    if isinstance(token_count, int) and len(topk_weights) != token_count:
        errors.append(f"topk_weights row count {len(topk_weights)} != token_count {token_count}")

    for row_idx, row in enumerate(topk_indices):
        if not isinstance(row, list):
            errors.append(f"topk_indices[{row_idx}] must be an array")
            continue
        if isinstance(top_k, int) and len(row) != top_k:
            errors.append(f"topk_indices[{row_idx}] length {len(row)} != top_k {top_k}")
        for col_idx, expert_id in enumerate(row):
            if not isinstance(expert_id, int):
                errors.append(f"topk_indices[{row_idx}][{col_idx}] must be int")
            elif isinstance(num_experts, int) and not (0 <= expert_id < num_experts):
                errors.append(f"topk_indices[{row_idx}][{col_idx}]={expert_id} outside [0,{num_experts})")

    for row_idx, row in enumerate(topk_weights):
        if not isinstance(row, list):
            errors.append(f"topk_weights[{row_idx}] must be an array")
            continue
        if isinstance(top_k, int) and len(row) != top_k:
            errors.append(f"topk_weights[{row_idx}] length {len(row)} != top_k {top_k}")
        for col_idx, weight in enumerate(row):
            if not finite_number(weight):
                errors.append(f"topk_weights[{row_idx}][{col_idx}] must be finite number")

    expert_counts = routing.get("expert_counts")
    if not isinstance(expert_counts, list):
        errors.append("routing.expert_counts must be an array")
    elif isinstance(num_experts, int) and len(expert_counts) != num_experts:
        errors.append(f"expert_counts length {len(expert_counts)} != num_experts {num_experts}")
    elif any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in expert_counts):
        errors.append("expert_counts entries must be non-negative integers")
    elif isinstance(top_k, int) and sum(expert_counts) != len(topk_indices) * top_k:
        # keyed off actual row count (== token_count once the row-count check above passes)
        errors.append("expert_counts sum must equal token_count * top_k")

    if replay.get("stage") != "full_vram":
        errors.append("replay.stage must be full_vram for the current trace replay harness")
    # harness_module is executed via `python -m` by the replay runner.
    harness_module = replay.get("harness_module")
    if not isinstance(harness_module, str) or not harness_module:
        errors.append("replay.harness_module must be a non-empty string")

    if not allow_missing_artifacts:
        for key in ("reference_pt", "l1_golden_pt"):
            value = artifacts.get(key)
            if not isinstance(value, str) or not Path(value).exists():
                errors.append(f"artifacts.{key} does not exist: {value}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--allow-missing-artifacts", action="store_true")
    parser.add_argument("--summary-out", type=Path)
    args = parser.parse_args()

    trace = load_json(args.trace)
    errors = validate_trace(trace, allow_missing_artifacts=args.allow_missing_artifacts)
    summary = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "trace_schema_version": trace.get("schema_version") if isinstance(trace, dict) else None,
        "trace_path": str(args.trace),
        "valid": not errors,
        "errors": errors,
        "trace_id": trace.get("trace_id") if isinstance(trace, dict) else None,
    }
    if args.summary_out:
        write_json(args.summary_out, summary)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"valid route trace: {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
