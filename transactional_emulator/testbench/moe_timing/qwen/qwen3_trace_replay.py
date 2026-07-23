#!/usr/bin/env python3
"""Replay a Qwen3 true route trace through a fixed-route MoE emulator program.

This is a TIMING harness. It drives the emulator with the trace's fixed routing
and DUMMY (all-zero) expert weights to measure cycles and HBM bytes; its gate is a
zero-input shape/no-crash smoke, NOT a numerical-correctness check. Numerical
correctness of the gather / expert / scatter / route-weight math is validated
separately by the routed-MoE op tests (routed_moe/gpt_oss_moe_*_test.py and
real_layer0); this replay measures timing on top of that validated substrate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

# REPO_ROOT is parents[4] (qwen -> moe_timing -> testbench -> transactional_emulator -> repo);
# add the in-repo PLENA_Compiler submodule to sys.path so `import compiler` works
# when run standalone (not just under the justfile PYTHONPATH).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPILER_ROOT = _REPO_ROOT / "PLENA_Compiler"
if _COMPILER_ROOT.exists():
    sys.path.insert(0, str(_COMPILER_ROOT))

from compiler.aten.plena import PlenaCompiler  # noqa: E402
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw  # noqa: E402
from transactional_emulator.testbench.emulator_runner import (  # noqa: E402
    compare_emulator_output,
    run_emulator,
    run_emulator_repeat_gate,
)
from transactional_emulator.testbench.layout_utils import infer_hbm_tensor_layouts, prestage_bf16_vram_matrix  # noqa: E402
from transactional_emulator.testbench.models.gpt_oss.attention_semantics_test import _comparison_params  # noqa: E402
from transactional_emulator.testbench.routed_moe.gpt_oss_moe_gather_scatter_test import _align_to  # noqa: E402
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim  # noqa: E402
from transactional_emulator.testbench.moe_timing.replay.validate_route_trace import validate_trace  # noqa: E402
from transactional_emulator.testbench.moe_timing.qwen.utils import OUT_ROOT, ensure_paths, load_json, write_json  # noqa: E402
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402


def _flatten_int(rows: list[list[int]]) -> list[int]:
    return [int(value) for row in rows for value in row]


def _flatten_float(rows: list[list[float]]) -> list[float]:
    return [float(value) for row in rows for value in row]


def _expert_stride(prog: PlenaCompiler, shape: tuple[int, int]) -> int:
    raw_size = int(shape[0] * shape[1] * prog.real_data_ratio)
    return _align_to(raw_size, prog.mlen)


def _build_selected_dummy_weight_table(
    prog: PlenaCompiler,
    *,
    prefix: str,
    selected_experts: list[int],
    num_experts: int,
    shape: tuple[int, int],
    input_tensors: dict[str, torch.Tensor],
) -> tuple[list[Any], int, int]:
    stride = _expert_stride(prog, shape)
    base = prog._allocate_hbm(stride * num_experts)
    inputs = []
    zero = torch.zeros(shape, dtype=torch.bfloat16)
    for expert_id in selected_experts:
        name = f"{prefix}_e{expert_id}"
        inputs.append(prog.input(name, shape=shape, hbm_addr=base + expert_id * stride))
        input_tensors[name] = zero
    if not inputs:
        raise ValueError("selected_experts cannot be empty")
    return inputs, base, stride


def _synthetic_x(rows: int, hidden: int, *, mode: str, seed: int) -> torch.Tensor:
    if mode == "zeros":
        return torch.zeros(rows, hidden, dtype=torch.bfloat16)
    if mode == "random":
        torch.manual_seed(seed)
        return (torch.randn(rows, hidden) * 0.02).to(torch.bfloat16)
    raise ValueError(f"unknown input mode {mode!r}")


def build_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    ensure_paths()
    trace = load_json(args.trace)
    errors = validate_trace(trace, allow_missing_artifacts=True)
    if errors:
        raise ValueError("Invalid route trace:\n" + "\n".join(errors))

    model = trace["model"]
    workload = trace["workload"]
    routing = trace["routing"]
    if model["name"] != "Qwen3-30B-A3B":
        raise ValueError(f"qwen3_trace_replay supports Qwen3-30B-A3B, got {model['name']!r}")

    build_dir = args.build_dir or (OUT_ROOT / "trace_replay" / trace["trace_id"])
    build_dir = build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)

    rows = int(workload["token_count"])
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    num_experts = int(model["num_experts"])
    top_k = int(model["top_k"])
    topk_indices = [[int(value) for value in row] for row in routing["topk_indices"]]
    topk_weights = [[float(value) for value in row] for row in routing["topk_weights"]]
    pair_count = rows * top_k
    if len(topk_indices) != rows or len(topk_weights) != rows:
        raise ValueError("trace topk row count does not match token_count")
    selected_experts = sorted({expert_id for row in topk_indices for expert_id in row})

    prog = PlenaCompiler(mlen=args.mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    input_tensors: dict[str, torch.Tensor] = {}

    physical_rows = max(args.blen, math.ceil(rows / args.blen) * args.blen)
    vram_preload = torch.zeros(physical_rows * hidden, dtype=torch.bfloat16)
    x = _synthetic_x(rows, hidden, mode=args.input_mode, seed=args.seed)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="TraceReplayX",
        tensor=x,
        vram_addr=0,
        physical_shape=(physical_rows, hidden),
        vram_preload=vram_preload,
    )

    gate_inputs, gate_base, gate_stride = _build_selected_dummy_weight_table(
        prog,
        prefix="QwenGate",
        selected_experts=selected_experts,
        num_experts=num_experts,
        shape=(hidden, intermediate),
        input_tensors=input_tensors,
    )
    up_inputs, up_base, up_stride = _build_selected_dummy_weight_table(
        prog,
        prefix="QwenUp",
        selected_experts=selected_experts,
        num_experts=num_experts,
        shape=(hidden, intermediate),
        input_tensors=input_tensors,
    )
    down_inputs, down_base, down_stride = _build_selected_dummy_weight_table(
        prog,
        prefix="QwenDown",
        selected_experts=selected_experts,
        num_experts=num_experts,
        shape=(intermediate, hidden),
        input_tensors=input_tensors,
    )
    weight_templates = (gate_inputs[0], up_inputs[0], down_inputs[0])
    weight_table_bases = (gate_base, up_base, down_base)
    weight_table_strides = (gate_stride, up_stride, down_stride)

    zero = prog.fp_var("decoder_zero", size=1)
    one = prog.fp_var("decoder_one", size=args.blen)
    neg_alpha = prog.fp_var("decoder_neg_alpha", size=args.blen)
    limit_pos = prog.fp_var("decoder_unused_limit_pos", size=args.blen)
    limit_neg = prog.fp_var("decoder_unused_limit_neg", size=args.blen)
    shared_zero_row = prog.fp_var("decoder_shared_zero_row", size=args.mlen)
    topk_weight_var = prog.fp_var("trace_topk_weights", size=pair_count)
    route_fp_scratch = prog.fp_var("trace_route_fp_scratch", size=args.mlen)
    topk_weights_fp_base = topk_weight_var.address
    topk_indices_int_base = 0

    accumulator = prog.alloc(
        "TraceReplayAccumulator",
        rows=rows,
        cols=hidden,
        strict=False,
        physical_shape=(physical_rows, hidden),
    )
    prog.moe_true_zero_vram_rows_v0(
        accumulator,
        rows=list(range(rows)),
        hidden=hidden,
        zero_row=shared_zero_row,
        policy_name="qwen3_moe",
        name="trace_acc_zero",
    )

    for pair_idx in range(pair_count):
        token_idx = pair_idx // top_k
        gathered = prog.moe_gather_token_rows_from_vram_v0(
            x_vram,
            token_indices=[token_idx],
            hidden=hidden,
            zero_row=shared_zero_row,
            policy_name="qwen3_moe",
            name=f"trace_pair{pair_idx}_vram_gather_t{token_idx}",
        )
        expert_out = prog.moe_dynamic_expert_pair_v0(
            gathered,
            weight_templates,
            weight_table_bases=weight_table_bases,
            weight_table_strides=weight_table_strides,
            expert_indices_int_base=topk_indices_int_base,
            weights_fp_base=topk_weights_fp_base,
            pair_idx=pair_idx,
            bias_tables=None,
            rows=args.blen,
            intermediate=intermediate,
            constants=(zero, limit_pos, limit_neg, one, neg_alpha),
            zero_row=shared_zero_row,
            route_fp_scratch=route_fp_scratch,
            policy_name="qwen3_moe",
            activation_policy="standard_swiglu",
            name=f"trace_pair{pair_idx}",
        )
        prog.moe_scatter_add_active_rows_v0(
            accumulator,
            expert_out,
            token_indices=[token_idx],
            active_rows=[0],
            hidden=hidden,
            policy_name="qwen3_moe",
            name=f"trace_pair{pair_idx}_scatter",
        )

    isa = prog.compile()
    fp_preload_len = max(
        neg_alpha.address + neg_alpha.size,
        topk_weight_var.address + topk_weight_var.size,
        route_fp_scratch.address + route_fp_scratch.size,
        shared_zero_row.address + shared_zero_row.size,
    )
    fp_preload = [0.0] * fp_preload_len
    for idx in range(one.size):
        fp_preload[one.address + idx] = 1.0
    for idx in range(neg_alpha.size):
        fp_preload[neg_alpha.address + idx] = -1.0
    flat_weights = _flatten_float(topk_weights)
    for idx, value in enumerate(flat_weights):
        fp_preload[topk_weights_fp_base + idx] = value

    int_preload = torch.tensor(_flatten_int(topk_indices), dtype=torch.int32)
    # Zeros golden: expert weights are dummy zeros (this is a timing run), so the
    # expected output is zero regardless of input. The comparison is therefore a
    # shape/no-crash smoke, not numerical validation — see the module docstring.
    golden = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    comparison_params = _comparison_params(
        prog.get_vram_addr(accumulator.name),
        rows,
        hidden,
        args.mlen,
        physical_rows=accumulator.physical_shape[0],
    )
    tensor_layouts = infer_hbm_tensor_layouts(input_tensors)
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    data_order = sorted(input_tensors, key=lambda name: hbm_addrs[name])

    create_sim_env(
        input_tensors,
        isa,
        {
            "original_output": golden,
            "compile_info": {
                "trace_id": trace["trace_id"],
                "measurement_note": trace.get("measurement_note"),
                "input_mode": args.input_mode,
                "selected_experts": selected_experts,
            },
        },
        fp_preload=fp_preload,
        int_preload=int_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="qwen3_trace_replay",
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    (build_dir / "comparison_params.json").write_text(json.dumps(comparison_params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(isa)
    write_json(build_dir / "trace.json", trace)
    manifest = {
        "schema_version": 1,
        "trace_id": trace["trace_id"],
        "trace_path": str(args.trace),
        "benchmark": workload["benchmark"],
        "sample_id": workload["sample_id"],
        "phase": workload["phase"],
        "layer": model["layer_index"],
        "rows": rows,
        "hidden": hidden,
        "intermediate": intermediate,
        "num_experts": num_experts,
        "top_k": top_k,
        "pair_count": pair_count,
        "selected_experts": selected_experts,
        "selected_expert_count": len(selected_experts),
        "mlen": args.mlen,
        "blen": args.blen,
        "input_mode": args.input_mode,
        "topk_indices_int_base": topk_indices_int_base,
        "topk_weights_fp_base": topk_weights_fp_base,
        "weight_table_bases": {"gate": gate_base, "up": up_base, "down": down_base},
        "weight_table_strides": {"gate": gate_stride, "up": up_stride, "down": down_stride},
        "hbm_input_tensor_count": len(input_tensors),
        "asm_lines": len(isa.splitlines()),
        "measurement_note": "self-consistent upper bound, absolute accuracy pending RTL (Window 2)",
        "comparison_params": comparison_params,
    }
    write_json(build_dir / "qwen3_trace_replay_manifest.json", manifest)
    return {"trace": trace, "build_dir": build_dir, "manifest": manifest}


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    built = build_artifacts(args)
    build_dir: Path = built["build_dir"]
    manifest = built["manifest"]
    if args.no_run:
        return {"schema_version": 1, "build_dir": str(build_dir), "manifest": manifest, "ran": False}

    metrics = run_emulator(
        build_dir,
        threads=args.emu_threads,
        stage_profile=args.stage_profile,
        dump_cwd=build_dir,
        overlap_prefetch_compute=args.experimental_overlap_prefetch_compute,
    )
    results, params = compare_emulator_output(build_dir)
    gate = {
        "passed": bool(results.get("test_pass", results.get("allclose_pass", False))),
        "allclose_pass": bool(results.get("allclose_pass", False)),
        "relative_match_rate": results.get("relative_match_rate"),
        "max_error": results.get("max_error"),
        "relative_error": results.get("relative_error"),
        "zero_input_gate": manifest["input_mode"] == "zeros",
        # A shape / no-crash smoke on zero (dummy-weight) inputs, NOT a numerical
        # correctness check: with all-zero weights and a zeros golden the comparison
        # is trivially zeros==zeros. Numerical correctness of the gather / expert /
        # scatter / route-weight math is validated separately by the routed-MoE op
        # tests (routed_moe/gpt_oss_moe_*_test.py and real_layer0); this replay only
        # measures timing/bytes on top of that already-validated substrate.
        "gate_kind": "zero_input_shape_smoke",
    }
    repeat_summary = None
    if args.repeat_gate:
        repeat_summary = run_emulator_repeat_gate(
            build_dir,
            repeats=args.repeat_gate,
            threads=args.emu_threads,
            stage_profile=False,
            overlap_prefetch_compute=args.experimental_overlap_prefetch_compute,
        )
    summary = {
        **manifest,
        "run_metrics": metrics,
        "comparison_params_runtime": params,
        "emulator_compare_raw": {
            key: results[key]
            for key in (
                "mse",
                "mae",
                "max_error",
                "relative_error",
                "relative_match_rate",
                "allclose_match_rate",
                "match_rate",
                "allclose_pass",
                "test_pass",
                "atol",
                "rtol",
            )
            if key in results
        },
        "zero_input_smoke_gate": gate,
        "repeat_gate": repeat_summary,
    }
    write_json(build_dir / "qwen3_trace_replay_results.json", summary)
    write_json(build_dir / "gather_scatter_results.json", summary)
    if args.cleanup_dumps:
        removed = []
        for name in ("mram_dump.bin", "vram_dump.bin", "hbm_dump.bin", "fpsram_dump.bin", "intsram_dump.bin"):
            path = build_dir / name
            if path.exists():
                path.unlink()
                removed.append(name)
        summary["cleanup_removed_dumps"] = removed
        write_json(build_dir / "qwen3_trace_replay_results.json", summary)
        write_json(build_dir / "gather_scatter_results.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise AssertionError(f"trace replay zero-input smoke gate failed: {gate}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--input-mode", choices=("zeros", "random"), default="zeros")
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--repeat-gate", type=int, default=0)
    parser.add_argument("--experimental-overlap-prefetch-compute", action="store_true")
    parser.add_argument("--keep-dumps", dest="cleanup_dumps", action="store_false")
    parser.add_argument("--no-run", action="store_true")
    parser.set_defaults(cleanup_dumps=True)
    parser.set_defaults(mlen=128, blen=4)
    args = parser.parse_args()
    run_trace(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
