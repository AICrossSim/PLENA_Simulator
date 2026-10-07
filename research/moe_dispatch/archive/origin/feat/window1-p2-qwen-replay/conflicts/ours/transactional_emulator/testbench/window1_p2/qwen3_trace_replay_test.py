#!/usr/bin/env python3
"""Replay a Qwen3 true route trace through a fixed-route MoE emulator program."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from safetensors.torch import safe_open

for _root in (
    Path(__file__).resolve().parents[3],
    Path(__file__).resolve().parents[4],
    Path(__file__).resolve().parents[5],
):
    _compiler = _root / "PLENA_Compiler"
    if (_compiler / "aten").exists() and str(_compiler) not in sys.path:
        sys.path.insert(0, str(_compiler))

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.emulator_runner import (
    compare_emulator_output,
    run_emulator,
    run_emulator_repeat_gate,
)
from transactional_emulator.testbench.layout_utils import infer_hbm_tensor_layouts, prestage_bf16_vram_matrix
from transactional_emulator.testbench.models.gpt_oss.attention_semantics_test import (
    _comparison_params,
    _device_routing_vram_policy_golden,
)
from transactional_emulator.testbench.routed_moe.gpt_oss_moe_gather_scatter_test import _align_to
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.window1_p1.validate_route_trace import validate_trace
from transactional_emulator.testbench.window1_p2.p2_utils import OUT_ROOT, ensure_paths, load_json, write_json
from transactional_emulator.tools.create_sim_env import create_sim_env

NONZERO_FAULT_CHOICES = (
    "none",
    "expert_base_plus_one_block",
    "swap_first_two_experts",
    "shift_token_expert_map",
    "scale_element_misaligned",
)


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


class _SelectedExpertTensor:
    """Sparse expert tensor view keyed by real expert id."""

    def __init__(self, tensors: dict[int, torch.Tensor]):
        self._tensors = tensors

    def __getitem__(self, expert_id: int | torch.Tensor) -> torch.Tensor:
        return self._tensors[int(expert_id)]


def _load_selected_qwen3_experts(
    *,
    snapshot: Path,
    layer_idx: int,
    selected_experts: list[int],
) -> dict[str, Any]:
    """Load only the Qwen3 expert tensors reached by this fixed-route trace."""

    snapshot = snapshot.expanduser().resolve()
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing Qwen safetensors index: {index_path}")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    handles: dict[str, safe_open] = {}

    def tensor(name: str) -> torch.Tensor:
        shard_name = weight_map[name]
        if shard_name not in handles:
            handles[shard_name] = safe_open(snapshot / shard_name, framework="pt", device="cpu")
        return handles[shard_name].get_tensor(name).to(torch.bfloat16).contiguous()

    prefix = f"model.layers.{layer_idx}"
    gate: dict[int, torch.Tensor] = {}
    up: dict[int, torch.Tensor] = {}
    down: dict[int, torch.Tensor] = {}
    for expert_id in selected_experts:
        gate[expert_id] = tensor(f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight").T.contiguous()
        up[expert_id] = tensor(f"{prefix}.mlp.experts.{expert_id}.up_proj.weight").T.contiguous()
        down[expert_id] = tensor(f"{prefix}.mlp.experts.{expert_id}.down_proj.weight").T.contiguous()
    return {
        "snapshot": str(snapshot),
        "layer_idx": layer_idx,
        "gate_weight": _SelectedExpertTensor(gate),
        "up_weight": _SelectedExpertTensor(up),
        "down_weight": _SelectedExpertTensor(down),
        "gate_bias": None,
        "up_bias": None,
        "down_bias": None,
        "loaded_experts": selected_experts,
    }


def _build_selected_real_weight_table(
    prog: PlenaCompiler,
    *,
    prefix: str,
    selected_experts: list[int],
    num_experts: int,
    weights: _SelectedExpertTensor,
    input_tensors: dict[str, torch.Tensor],
) -> tuple[list[Any], int, int, dict[str, str]]:
    first_shape = tuple(int(dim) for dim in weights[selected_experts[0]].shape)
    stride = _expert_stride(prog, first_shape)
    base = prog._allocate_hbm(stride * num_experts)
    inputs = []
    tensor_names = {}
    for expert_id in selected_experts:
        tensor = weights[expert_id]
        if tuple(int(dim) for dim in tensor.shape) != first_shape:
            raise ValueError(f"{prefix} expert {expert_id} shape {tuple(tensor.shape)} != {first_shape}")
        name = f"{prefix}_e{expert_id}"
        inputs.append(prog.input(name, shape=first_shape, hbm_addr=base + expert_id * stride))
        input_tensors[name] = tensor
        tensor_names[str(expert_id)] = name
    if not inputs:
        raise ValueError("selected_experts cannot be empty")
    return inputs, base, stride, tensor_names


def _faulted_topk_indices(
    topk_indices: list[list[int]],
    *,
    selected_experts: list[int],
    fault_injection: str,
) -> tuple[list[list[int]], dict[str, Any]]:
    if fault_injection == "swap_first_two_experts":
        first_row = topk_indices[0] if topk_indices else []
        ordered_unique = []
        for value in first_row:
            if value not in ordered_unique:
                ordered_unique.append(value)
        if len(ordered_unique) < 2:
            ordered_unique = list(selected_experts)
        if len(ordered_unique) < 2:
            raise ValueError("swap_first_two_experts requires at least two selected experts")
        a, b = ordered_unique[0], ordered_unique[1]
        swapped = [[b if value == a else a if value == b else value for value in row] for row in topk_indices]
        return swapped, {"fault": fault_injection, "swapped_experts": [a, b]}
    if fault_injection == "shift_token_expert_map":
        if len(topk_indices) < 2:
            raise ValueError("shift_token_expert_map requires at least two token rows")
        shifted = [list(row) for row in (topk_indices[1:] + topk_indices[:1])]
        return shifted, {"fault": fault_injection, "shift": "topk rows rotated left by one token"}
    return [list(row) for row in topk_indices], {"fault": fault_injection}


def _corrupt_first_scale_region(
    *,
    build_dir: Path,
    selected_experts: list[int],
    gate_tensor_names: dict[str, str],
    hbm_addrs: dict[str, int],
    input_tensors: dict[str, torch.Tensor],
    target_expert_id: int | None = None,
) -> dict[str, Any]:
    expert_id = target_expert_id if target_expert_id in selected_experts else selected_experts[0]
    tensor_name = gate_tensor_names[str(expert_id)]
    tensor = input_tensors[tensor_name]
    rows, cols = (int(tensor.shape[0]), int(tensor.shape[1]))
    tensor_base = int(hbm_addrs[tensor_name])
    scale_start = tensor_base + rows * cols
    scale_len = rows * math.ceil(cols / 8)
    edit_len = scale_len
    hbm_path = build_dir / "hbm_for_behave_sim.bin"
    with hbm_path.open("r+b") as f:
        f.seek(scale_start)
        data = bytearray(f.read(edit_len))
        if len(data) < 2:
            raise ValueError(f"scale region too small to corrupt: {tensor_name}")
        data = data[1:] + data[:1]
        f.seek(scale_start)
        f.write(data)
    return {
        "fault": "scale_element_misaligned",
        "tensor": tensor_name,
        "expert_id": expert_id,
        "hbm_addr": tensor_base,
        "scale_start": scale_start,
        "scale_bytes_rotated": edit_len,
    }


def _decode_vram_bf16_matrix(build_dir: Path, params: dict[str, Any]) -> torch.Tensor:
    dump_path = build_dir / "vram_dump.bin"
    if not dump_path.exists():
        raise FileNotFoundError(f"missing emulator VRAM dump: {dump_path}")
    row_dim = int(params["row_dim"])
    rows = int(params["num_batches"])
    cols = int(params["elements_per_batch"])
    physical_rows = int(params.get("physical_rows", rows))
    start = int(params["start_row_idx"])
    raw = np.fromfile(dump_path, dtype="<u2")
    if raw.size % row_dim != 0:
        raise ValueError(f"VRAM dump element count {raw.size} is not divisible by row_dim={row_dim}")
    vram = torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(-1, row_dim)
    chunks = math.ceil(cols / row_dim)
    out = torch.zeros(rows, cols, dtype=torch.bfloat16)
    for chunk_idx in range(chunks):
        col_start = chunk_idx * row_dim
        col_end = min(cols, col_start + row_dim)
        source_row = start + chunk_idx * physical_rows
        block = vram[source_row : source_row + rows, : col_end - col_start]
        out[:, col_start:col_end] = block
    return out


def _nonzero_gate_summary(
    *,
    build_dir: Path,
    golden: torch.Tensor,
    comparison_params: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    actual = _decode_vram_bf16_matrix(build_dir, comparison_params)
    actual_f = actual.float()
    golden_f = golden.float()
    diff = actual_f - golden_f
    rel_rms = float(torch.linalg.vector_norm(diff) / torch.clamp(torch.linalg.vector_norm(golden_f), min=1e-12))
    abs_diff = diff.abs()
    max_flat = int(torch.argmax(abs_diff).item()) if abs_diff.numel() else 0
    row = max_flat // int(golden.shape[1])
    col = max_flat % int(golden.shape[1])
    summary = {
        "passed": rel_rms <= threshold,
        "rel_rms": rel_rms,
        "threshold": threshold,
        "rows": int(golden.shape[0]),
        "cols": int(golden.shape[1]),
        "max_abs_error": float(abs_diff.reshape(-1)[max_flat].item()) if abs_diff.numel() else 0.0,
        "max_error_position": {"row": row, "col": col},
        "max_error_values": {
            "actual": float(actual_f[row, col].item()) if actual.numel() else 0.0,
            "golden": float(golden_f[row, col].item()) if golden.numel() else 0.0,
            "diff": float(diff[row, col].item()) if diff.numel() else 0.0,
        },
    }
    write_json(build_dir / "nonzero_functional_gate.json", summary)
    return summary


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
        raise ValueError(f"qwen3_trace_replay_test supports Qwen3-30B-A3B, got {model['name']!r}")

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
    if args.fault_injection != "none" and args.functional_golden_mode != "nonzero-real":
        raise ValueError("--fault-injection is only valid with --functional-golden-mode nonzero-real")
    topk_indices_for_emulator, fault_metadata = _faulted_topk_indices(
        topk_indices,
        selected_experts=selected_experts,
        fault_injection=args.fault_injection,
    )

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

    selected_real_weights = None
    gate_tensor_names: dict[str, str] = {}
    if args.functional_golden_mode == "nonzero-real":
        if args.qwen_snapshot is None:
            raise ValueError("--qwen-snapshot is required for --functional-golden-mode nonzero-real")
        selected_real_weights = _load_selected_qwen3_experts(
            snapshot=args.qwen_snapshot,
            layer_idx=int(model["layer_index"]),
            selected_experts=selected_experts,
        )
        gate_inputs, gate_base, gate_stride, gate_tensor_names = _build_selected_real_weight_table(
            prog,
            prefix="QwenGate",
            selected_experts=selected_experts,
            num_experts=num_experts,
            weights=selected_real_weights["gate_weight"],
            input_tensors=input_tensors,
        )
        up_inputs, up_base, up_stride, _ = _build_selected_real_weight_table(
            prog,
            prefix="QwenUp",
            selected_experts=selected_experts,
            num_experts=num_experts,
            weights=selected_real_weights["up_weight"],
            input_tensors=input_tensors,
        )
        down_inputs, down_base, down_stride, _ = _build_selected_real_weight_table(
            prog,
            prefix="QwenDown",
            selected_experts=selected_experts,
            num_experts=num_experts,
            weights=selected_real_weights["down_weight"],
            input_tensors=input_tensors,
        )
    else:
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
    program_weight_table_bases = weight_table_bases
    if args.fault_injection == "expert_base_plus_one_block":
        program_weight_table_bases = tuple(
            int(base) + int(stride) for base, stride in zip(weight_table_bases, weight_table_strides)
        )
        fault_metadata = {
            "fault": args.fault_injection,
            "program_weight_table_bases": list(program_weight_table_bases),
            "true_weight_table_bases": list(weight_table_bases),
            "weight_table_strides": list(weight_table_strides),
        }

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
            weight_table_bases=program_weight_table_bases,
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

    int_preload = torch.tensor(_flatten_int(topk_indices_for_emulator), dtype=torch.int32)
    if args.functional_golden_mode == "nonzero-real":
        split = SimpleNamespace(
            gate_weight=selected_real_weights["gate_weight"],
            up_weight=selected_real_weights["up_weight"],
            gate_bias=None,
            up_bias=None,
        )
        golden, _ = _device_routing_vram_policy_golden(
            x=x,
            device_indices=torch.tensor(topk_indices, dtype=torch.long),
            device_weights=torch.tensor(topk_weights, dtype=torch.bfloat16),
            split=split,
            down_weight=selected_real_weights["down_weight"],
            down_bias=None,
            rows=rows,
            hidden=hidden,
            blen=args.blen,
            mlen=args.mlen,
            activation_policy=str(model.get("activation_policy", "standard_swiglu")),
        )
    else:
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
    if args.fault_injection == "scale_element_misaligned":
        fault_metadata = _corrupt_first_scale_region(
            build_dir=build_dir,
            selected_experts=selected_experts,
            gate_tensor_names=gate_tensor_names,
            hbm_addrs=hbm_addrs,
            input_tensors=input_tensors,
            target_expert_id=int(topk_indices[0][0]) if topk_indices and topk_indices[0] else None,
        )
    (build_dir / "comparison_params.json").write_text(json.dumps(comparison_params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(isa)
    write_json(build_dir / "trace.json", trace)
    if args.functional_golden_mode == "nonzero-real":
        np.savez_compressed(
            build_dir / "nonzero_functional_golden.npz",
            x=x.float().numpy(),
            golden=golden.float().numpy(),
            topk_indices=np.asarray(topk_indices, dtype=np.int32),
            topk_weights=np.asarray(topk_weights, dtype=np.float32),
        )
        write_json(
            build_dir / "nonzero_functional_metadata.json",
            {
                "schema_version": 1,
                "seed": args.seed,
                "input_source": "fixed_seed_random_hidden_states",
                "hidden_state_note": "sweep50 hidden states were not found on disk; fixed-seed random activations are used for this nonzero functional gate.",
                "snapshot": str(args.qwen_snapshot.expanduser().resolve()),
                "sample_id": workload["sample_id"],
                "sample_index": workload.get("sample_index"),
                "benchmark": workload["benchmark"],
                "phase": workload["phase"],
                "layer": model["layer_index"],
                "expert_ids": selected_experts,
                "precision_config": {
                    "mlen": args.mlen,
                    "blen": args.blen,
                    "real_data_ratio": hw.real_data_ratio,
                    "weight_format": "MXFP8 via create_mem_for_sim",
                    "activation_accumulation": "BF16 VRAM-style golden from active plena_settings.toml",
                },
                "fault_injection": fault_metadata,
            },
        )
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
        "functional_golden_mode": args.functional_golden_mode,
        "fault_injection": fault_metadata,
        "topk_indices_int_base": topk_indices_int_base,
        "topk_weights_fp_base": topk_weights_fp_base,
        "weight_table_bases": {"gate": gate_base, "up": up_base, "down": down_base},
        "program_weight_table_bases": {
            "gate": program_weight_table_bases[0],
            "up": program_weight_table_bases[1],
            "down": program_weight_table_bases[2],
        },
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
    }
    nonzero_gate = None
    if manifest["functional_golden_mode"] == "nonzero-real":
        golden_npz = np.load(build_dir / "nonzero_functional_golden.npz")
        golden = torch.from_numpy(golden_npz["golden"]).to(torch.bfloat16)
        nonzero_gate = _nonzero_gate_summary(
            build_dir=build_dir,
            golden=golden,
            comparison_params=manifest["comparison_params"],
            threshold=args.nonzero_rel_rms_threshold,
        )
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
        "full_vram_gate": gate,
        "nonzero_functional_gate": nonzero_gate,
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
    if nonzero_gate is not None:
        if args.expect_nonzero_gate_fail:
            if nonzero_gate["passed"]:
                raise AssertionError("expected nonzero functional gate to fail, but it passed")
            return summary
        if not nonzero_gate["passed"]:
            raise AssertionError(f"nonzero functional gate failed: {nonzero_gate}")
    if not gate["passed"]:
        raise AssertionError(f"trace replay functional gate failed: {gate}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--input-mode", choices=("zeros", "random"), default="zeros")
    parser.add_argument("--functional-golden-mode", choices=("zero", "nonzero-real"), default="zero")
    parser.add_argument("--qwen-snapshot", type=Path)
    parser.add_argument("--nonzero-rel-rms-threshold", type=float, default=0.01)
    parser.add_argument("--fault-injection", choices=NONZERO_FAULT_CHOICES, default="none")
    parser.add_argument("--expect-nonzero-gate-fail", action="store_true")
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
