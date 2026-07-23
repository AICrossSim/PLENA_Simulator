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


def validate_trace(trace: dict[str, Any], *, allow_missing_artifacts: bool = False) -> list[str]:
    errors: list[str] = []
    if not isinstance(trace, dict):
        return ["trace must be a JSON object"]
    if trace.get("schema_version") != 1:
        errors.append("schema_version must be 1")
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
    # harness_module is dereferenced (and executed via `python -m`) by the replay
    # runner, so it must be present and a non-empty string.
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
        "schema_version": 1,
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
