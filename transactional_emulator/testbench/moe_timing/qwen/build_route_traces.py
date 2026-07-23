#!/usr/bin/env python3
"""Convert P2 true-routing JSONL rows into P1 route trace schema files."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.moe_timing.replay.validate_route_trace import validate_trace
from transactional_emulator.testbench.moe_timing.qwen.utils import (
    MODEL_CONFIGS,
    OUT_ROOT,
    ensure_paths,
    iter_jsonl,
    routing_stats,
    stable_id,
    write_json,
)


def _parse_csv_ints(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(part) for part in value.split(",") if part.strip()}


def _parse_csv_strings(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {part.strip() for part in value.split(",") if part.strip()}


def _trace_id(record: dict[str, Any]) -> str:
    phase = str(record["phase"])
    suffix = f"_d{record.get('decode_step')}" if phase == "decode" and record.get("decode_step") is not None else ""
    return stable_id(
        f"{record['model_key']}_{record['benchmark']}_{record['sample_id']}_l{record['layer']}_{phase}{suffix}"
    )


def _weights_for(record: dict[str, Any], top_k: int, *, allow_uniform_weights: bool) -> tuple[list[list[float]], str]:
    weights = record.get("route_weights")
    if weights is not None:
        return [[float(x) for x in row] for row in weights], "true_router_topk_softmax_weights"
    if not allow_uniform_weights:
        raise ValueError(
            f"routing row {record.get('sample_id')} layer={record.get('layer')} phase={record.get('phase')} "
            "has routes but no route_weights; rerun generate_true_routing_with_weights.py or pass "
            "--allow-uniform-weights for timing-only shape replay"
        )
    rows = int(record["tokens"])
    return [[1.0 / float(top_k) for _ in range(top_k)] for _ in range(rows)], "uniform_weights_timing_only"


def trace_from_record(
    record: dict[str, Any],
    *,
    mlen: int,
    blen: int,
    emu_threads: int,
    allow_uniform_weights: bool,
) -> dict[str, Any]:
    model_cfg = MODEL_CONFIGS[record["model_key"]]
    top_k = int(model_cfg["top_k"])
    num_experts = int(model_cfg["num_experts"])
    indices = [[int(x) for x in row] for row in record["routes"]]
    weights, weight_source = _weights_for(record, top_k, allow_uniform_weights=allow_uniform_weights)
    stats = routing_stats(indices, num_experts)
    source = str(record.get("routing_source", "true_router_logits"))
    if weight_source == "uniform_weights_timing_only":
        source += "+uniform_weights_timing_only"
    trace = {
        "schema_version": 1,
        "trace_id": _trace_id(record),
        "created_by": "transactional_emulator.testbench.moe_timing.qwen.build_route_traces",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": {
            "name": model_cfg["name"],
            "layer_index": int(record["layer"]),
            "hidden_size": int(model_cfg["hidden_size"]),
            "intermediate_size": int(model_cfg["intermediate_size"]),
            "num_experts": num_experts,
            "top_k": top_k,
            "policy_name": model_cfg["policy_name"],
            "activation_policy": model_cfg["activation_policy"],
        },
        "workload": {
            "benchmark": record["benchmark"],
            "sample_id": str(record["sample_id"]),
            "phase": record["phase"],
            "batch_size": 1,
            "seq_len": int(record["tokens"]),
            "token_count": int(record["tokens"]),
            "input_tokens": int(record.get("input_tokens", record["tokens"])),
            "category": record.get("category", ""),
            "sample_index": record.get("sample_index"),
            "decode_step": record.get("decode_step"),
        },
        "routing": {
            "source": source,
            "weight_source": weight_source,
            "topk_indices": indices,
            "topk_weights": weights,
            **stats,
        },
        "logical_bytes": {
            "input_hidden_bf16_bytes": int(record["tokens"]) * int(model_cfg["hidden_size"]) * 2,
            "routed_hidden_bf16_bytes": int(record["tokens"]) * top_k * int(model_cfg["hidden_size"]) * 2,
            "route_weight_bf16_bytes": int(record["tokens"]) * top_k * 2,
            "expert_weight_table_note": "Physical HBM bytes come from Rust emulator WithStats; logical model bytes are context only.",
        },
        "artifacts": {
            "reference_pt": "generated_by_moe_timing_qwen_qwen3_trace_replay",
            "l1_golden_pt": "generated_by_moe_timing_qwen_qwen3_trace_replay",
        },
        "replay": {
            "harness_module": "transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay",
            "stage": "full_vram",
            "mlen": mlen,
            "blen": blen,
            "emu_threads": emu_threads,
        },
        "measurement_note": "self-consistent upper bound, absolute accuracy pending RTL (Window 2)",
    }
    errors = validate_trace(trace, allow_missing_artifacts=True)
    if errors:
        raise ValueError("Generated invalid trace:\n" + "\n".join(errors))
    return trace


def build_traces(args: argparse.Namespace) -> list[Path]:
    ensure_paths()
    sample_ids = _parse_csv_strings(args.sample_ids)
    layers = _parse_csv_ints(args.layers)
    phases = _parse_csv_strings(args.phases)
    out_dir = args.out_dir or (OUT_ROOT / "route_traces")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for record in iter_jsonl(args.input):
        if sample_ids is not None and str(record.get("sample_id")) not in sample_ids:
            continue
        if layers is not None and int(record.get("layer")) not in layers:
            continue
        if phases is not None and str(record.get("phase")) not in phases:
            continue
        if "routes" not in record:
            raise ValueError(f"routing row has no routes: {record.get('sample_id')} layer={record.get('layer')}")
        trace = trace_from_record(
            record,
            mlen=args.mlen,
            blen=args.blen,
            emu_threads=args.emu_threads,
            allow_uniform_weights=args.allow_uniform_weights,
        )
        path = out_dir / f"{trace['trace_id']}.json"
        write_json(path, trace)
        written.append(path)
        if args.limit is not None and len(written) >= args.limit:
            break
    summary = {
        "schema_version": 1,
        "input": str(args.input),
        "out_dir": str(out_dir),
        "written": [str(path) for path in written],
        "count": len(written),
    }
    write_json(args.summary_out or (OUT_ROOT / "route_traces_summary.json"), summary)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--sample-ids")
    parser.add_argument("--layers")
    parser.add_argument("--phases")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mlen", type=int, default=128)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--allow-uniform-weights", action="store_true")
    args = parser.parse_args()
    written = build_traces(args)
    print(f"wrote {len(written)} route traces")
    for path in written[:10]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
