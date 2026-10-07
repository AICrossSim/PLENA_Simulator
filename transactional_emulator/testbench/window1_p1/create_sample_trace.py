#!/usr/bin/env python3
"""Create the GPT-OSS layer0 tok1 sample route trace used by Window 1 P1."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.window1_p1.p1_utils import (
    DEFAULT_OUT_ROOT,
    gini_from_counts,
    load_json,
    summarize_run,
    write_json,
)
from transactional_emulator.testbench.window1_p1.validate_route_trace import validate_trace


def _to_nested_float(values: Any) -> list[list[float]]:
    return [[float(x) for x in row] for row in values.tolist()]


def _to_nested_int(values: Any) -> list[list[int]]:
    return [[int(x) for x in row] for row in values.tolist()]


def _routing_stats(indices: list[list[int]], num_experts: int) -> dict[str, Any]:
    counts = [0 for _ in range(num_experts)]
    for row in indices:
        for expert_id in row:
            counts[expert_id] += 1
    nonzero = [idx for idx, value in enumerate(counts) if value > 0]
    total_assignments = sum(counts)
    duplicate_factor = float(total_assignments) / float(len(nonzero)) if nonzero else 0.0
    return {
        "expert_counts": counts,
        "active_experts": nonzero,
        "hot_experts": sorted(nonzero, key=lambda idx: (-counts[idx], idx)),
        "cold_experts": [idx for idx, value in enumerate(counts) if value == 0],
        "duplicate_factor": duplicate_factor,
        "gini": gini_from_counts(counts),
    }


def create_trace(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    reference = torch.load(args.reference_path, map_location="cpu")
    if not isinstance(reference, dict):
        raise TypeError(f"reference must be a dict-like .pt file: {args.reference_path}")
    topk_indices = _to_nested_int(reference["topk_indices"])
    topk_weights = _to_nested_float(reference["topk_weights"])
    x = reference["x"]
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    model = reference.get("metadata", {}).get("model", {}) if isinstance(reference.get("metadata"), dict) else {}
    num_experts = int(args.num_experts)
    top_k = int(args.top_k or len(topk_indices[0]))
    stats = _routing_stats(topk_indices, num_experts)

    fixed_run, fixed_stack = summarize_run("fixed_route_direct", args.fixed_route_build_dir)
    device_run, device_stack = summarize_run("device_selected_direct", args.device_selected_build_dir)

    trace = {
        "schema_version": 1,
        "trace_id": args.trace_id,
        "created_by": "transactional_emulator.testbench.window1_p1.create_sample_trace",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": {
            "name": args.model_name,
            "layer_index": args.layer_index,
            "hidden_size": hidden,
            "intermediate_size": int(args.intermediate_size),
            "num_experts": num_experts,
            "top_k": top_k,
            "metadata_model": model,
        },
        "workload": {
            "benchmark": args.benchmark,
            "sample_id": args.sample_id,
            "phase": args.phase,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "token_count": rows,
        },
        "routing": {
            "source": "hf_reference_router_logits_topk",
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            **stats,
        },
        "logical_bytes": {
            "input_hidden_bf16_bytes": rows * hidden * 2,
            "routed_hidden_bf16_bytes": rows * top_k * hidden * 2,
            "expert_weight_mxfp8_table_note": "Logical expert-weight bytes are model/layout dependent; physical HBM bytes are taken from emulator WithStats.",
        },
        "artifacts": {
            "reference_pt": str(args.reference_path),
            "l1_golden_pt": str(args.l1_golden_path),
            "fixed_route_build_dir": str(args.fixed_route_build_dir),
            "device_selected_build_dir": str(args.device_selected_build_dir),
        },
        "replay": {
            "harness_module": "transactional_emulator.testbench.routed_moe.gpt_oss_moe_gather_scatter_test",
            "stage": "full_vram",
            "mlen": args.mlen,
            "blen": args.blen,
            "emu_threads": args.emu_threads,
        },
        "direct_runs": {
            "fixed_route": fixed_run,
            "device_selected": device_run,
            "routing_tax_cycles": int(device_run["sim_latency_cycles"]) - int(fixed_run["sim_latency_cycles"]),
            "routing_tax_hbm_bytes_read": int(device_run["hbm_bytes_read"]) - int(fixed_run["hbm_bytes_read"]),
        },
        "direct_stage_stack_rows": fixed_stack + device_stack,
    }
    errors = validate_trace(trace)
    if errors:
        raise ValueError("Generated invalid trace:\n" + "\n".join(errors))
    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_OUT_ROOT.parent / "real_workload" / "gpt_oss_layer0_tok1" / "reference" / "hf_layer0_moe_reference.pt")
    parser.add_argument("--l1-golden-path", type=Path, default=DEFAULT_OUT_ROOT.parent / "real_workload" / "gpt_oss_layer0_tok1" / "emulator" / "golden_output.pt")
    parser.add_argument("--fixed-route-build-dir", type=Path, default=DEFAULT_OUT_ROOT / "gpt_oss_layer0_tok1_fixed_route")
    parser.add_argument("--device-selected-build-dir", type=Path, default=DEFAULT_OUT_ROOT / "gpt_oss_layer0_tok1_device_selected")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT / "route_traces" / "gpt_oss_layer0_tok1_trace.json")
    parser.add_argument("--trace-id", default="gpt_oss_layer0_tok1_decode")
    parser.add_argument("--model-name", default="gpt-oss-20b")
    parser.add_argument("--benchmark", default="local_real_workload")
    parser.add_argument("--sample-id", default="gpt_oss_layer0_tok1")
    parser.add_argument("--phase", choices=("prefill", "decode", "smoke", "unknown"), default="decode")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--intermediate-size", type=int, default=2880)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--mlen", type=int, default=64)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--emu-threads", type=int, default=1)
    args = parser.parse_args()

    trace = create_trace(args)
    write_json(args.out, trace)
    print(f"wrote route trace: {args.out}")
    print(load_json(args.out)["direct_runs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

