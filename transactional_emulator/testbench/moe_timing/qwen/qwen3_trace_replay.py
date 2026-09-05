#!/usr/bin/env python3
"""Replay a true route trace through a fixed-route MoE emulator program.

The default is a timing run with all-zero expert weights. ``--weight-mode exact
--input-mode random`` switches to distinct nonzero MX-representable weights and
a BF16 torch reference, which is used to validate pair-major and expert-major
lowerings before timing full model dimensions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

# REPO_ROOT is parents[4] (qwen -> moe_timing -> testbench -> transactional_emulator -> repo).
# Resolve both in-repo dependencies from the checkout so standalone runs do not
# depend on a developer-specific PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPILER_ROOT = _REPO_ROOT / "PLENA_Compiler"
_TOOLS_ROOT = _REPO_ROOT / "PLENA_Tools"
for _dependency in (_REPO_ROOT, _COMPILER_ROOT, _TOOLS_ROOT):
    if _dependency.exists() and str(_dependency) not in sys.path:
        sys.path.insert(0, str(_dependency))

from compiler.aten.plena import PlenaCompiler  # noqa: E402
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw  # noqa: E402
from transactional_emulator.testbench.emulator_runner import (  # noqa: E402
    compare_emulator_output,
    run_emulator,
    run_emulator_repeat_gate,
)
from transactional_emulator.testbench.layout_utils import infer_hbm_tensor_layouts, prestage_bf16_vram_matrix  # noqa: E402
from transactional_emulator.testbench.aten.golden import quantize_to_mxfp  # noqa: E402
from transactional_emulator.testbench.gpt_oss_testkit import _exact_mxfp8_tensor  # noqa: E402
from transactional_emulator.testbench.models.gpt_oss.attention_semantics_test import _comparison_params  # noqa: E402
from transactional_emulator.testbench.routed_moe._shared_moe_reference import (  # noqa: E402
    combine_shared_and_routed_golden,
    deepseek_shared_expert_golden,
    qwen2_shared_expert_golden,
    swiglu_golden,
)
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
    storage_order: str = "row_major",
    weight_mode: str = "zeros",
    projection_index: int = 0,
) -> tuple[list[Any], int, int, dict[int, torch.Tensor]]:
    stride = _expert_stride(prog, shape)
    base = prog._allocate_hbm(stride * num_experts)
    inputs = []
    values_by_expert: dict[int, torch.Tensor] = {}
    zero = torch.zeros(shape, dtype=torch.bfloat16)
    for expert_id in selected_experts:
        name = f"{prefix}_e{expert_id}"
        inputs.append(
            prog.input(
                name,
                shape=shape,
                hbm_addr=base + expert_id * stride,
                hbm_storage_order=storage_order,
            )
        )
        if weight_mode == "zeros":
            value = zero
        elif weight_mode == "exact":
            value = (
                _exact_mxfp8_tensor(
                    shape,
                    stride=1 + ((expert_id + projection_index) % 4),
                    offset=(expert_id * 3 + projection_index) % 5,
                )
                * 0.125
            ).to(torch.bfloat16)
        else:
            raise ValueError(f"unsupported weight_mode {weight_mode!r}")
        input_tensors[name] = value
        values_by_expert[expert_id] = value
    if not inputs:
        raise ValueError("selected_experts cannot be empty")
    return inputs, base, stride, values_by_expert


def _build_static_weight(
    prog: PlenaCompiler,
    *,
    name: str,
    shape: tuple[int, int],
    input_tensors: dict[str, torch.Tensor],
    storage_order: str,
    weight_mode: str,
    pattern_index: int,
) -> tuple[Any, torch.Tensor]:
    input_var = prog.input(name, shape=shape, hbm_storage_order=storage_order)
    if weight_mode == "zeros":
        value = torch.zeros(shape, dtype=torch.bfloat16)
    elif weight_mode == "exact":
        value = (
            _exact_mxfp8_tensor(
                shape,
                stride=1 + (pattern_index % 4),
                offset=pattern_index % 5,
            )
            * 0.125
        ).to(torch.bfloat16)
    else:
        raise ValueError(f"unsupported weight_mode {weight_mode!r}")
    input_tensors[name] = value
    return input_var, value


def _synthetic_x(rows: int, hidden: int, *, mode: str, seed: int) -> torch.Tensor:
    if mode == "zeros":
        return torch.zeros(rows, hidden, dtype=torch.bfloat16)
    if mode == "random":
        torch.manual_seed(seed)
        return (torch.randn(rows, hidden) * 0.02).to(torch.bfloat16)
    raise ValueError(f"unknown input mode {mode!r}")


def _expert_major_groups(topk_indices: list[list[int]]) -> list[tuple[int, list[int]]]:
    """Return deterministic ``(expert_id, pair_indices)`` groups.

    Ordering by the first routed pair keeps the transformation stable while
    still allowing every later occurrence of the same expert to reuse its
    resident weight tiles.
    """
    groups: dict[int, list[int]] = {}
    top_k = len(topk_indices[0])
    for token_idx, row in enumerate(topk_indices):
        if len(row) != top_k:
            raise ValueError("topk_indices rows must have a uniform width")
        for route_slot, expert_id in enumerate(row):
            groups.setdefault(int(expert_id), []).append(token_idx * top_k + route_slot)
    return sorted(groups.items(), key=lambda item: (item[1][0], item[0]))


def _nonzero_routed_golden(
    x: torch.Tensor,
    topk_indices: list[list[int]],
    topk_weights: list[list[float]],
    gate_weights: dict[int, torch.Tensor],
    up_weights: dict[int, torch.Tensor],
    down_weights: dict[int, torch.Tensor],
) -> torch.Tensor:
    """BF16 reference for standard-SwiGLU routed experts."""

    rows, hidden = x.shape
    result = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    for token_idx, (experts, weights) in enumerate(zip(topk_indices, topk_weights, strict=True)):
        token = x[token_idx : token_idx + 1].to(torch.bfloat16)
        for expert_id, route_weight in zip(experts, weights, strict=True):
            gate = _mxfp_project(token, gate_weights[expert_id])
            up = _mxfp_project(token, up_weights[expert_id])
            activated = swiglu_golden(gate, up)
            expert_out = _mxfp_project(activated, down_weights[expert_id])
            weighted = (
                expert_out.float()
                * torch.tensor(route_weight, dtype=torch.bfloat16).float()
            ).to(torch.bfloat16)
            result[token_idx] = (result[token_idx].float() + weighted[0].float()).to(torch.bfloat16)
    return result


def _mxfp_project(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.matmul(lhs.float(), quantize_to_mxfp(rhs).float()).to(torch.bfloat16)


def _relative_rms(simulated: torch.Tensor, golden: torch.Tensor) -> float:
    """Return RMS(error) / RMS(reference), with an exact-zero fallback."""
    simulated_f32 = simulated.detach().float().reshape(-1)
    golden_f32 = golden.detach().float().reshape(-1)
    error_rms = torch.sqrt(torch.mean(torch.square(simulated_f32 - golden_f32)))
    reference_rms = torch.sqrt(torch.mean(torch.square(golden_f32)))
    if float(reference_rms) == 0.0:
        return 0.0 if float(error_rms) == 0.0 else math.inf
    return float(error_rms / reference_rms)


def build_artifacts(args: argparse.Namespace) -> dict[str, Any]:
    ensure_paths()
    trace = load_json(args.trace)
    errors = validate_trace(trace, allow_missing_artifacts=True)
    if errors:
        raise ValueError("Invalid route trace:\n" + "\n".join(errors))

    replay_contract = trace.get("replay", {})
    expected_mlen = replay_contract.get("mlen")
    expected_blen = replay_contract.get("blen")
    if expected_mlen is not None and int(args.mlen) != int(expected_mlen):
        raise ValueError(
            f"trace requires MLEN={expected_mlen}, but replay was requested with MLEN={args.mlen}"
        )
    if expected_blen is not None and int(args.blen) != int(expected_blen):
        raise ValueError(
            f"trace requires BLEN={expected_blen}, but replay was requested with BLEN={args.blen}"
        )

    model = trace["model"]
    workload = trace["workload"]
    routing = trace["routing"]
    build_dir = args.build_dir or (OUT_ROOT / "trace_replay" / trace["trace_id"])
    build_dir = build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)

    rows = int(workload["token_count"])
    hidden = int(model["hidden_size"])
    intermediate = int(model["intermediate_size"])
    num_experts = int(model["num_experts"])
    top_k = int(model["top_k"])
    shared_experts = int(model.get("shared_experts", 0))
    shared_intermediate = int(model.get("shared_intermediate_size", 0))
    shared_gate_kind = str(model.get("shared_gate", "none")).lower()
    if args.include_shared_expert:
        if shared_experts < 1 or shared_intermediate < 1:
            raise ValueError("--include-shared-expert requires shared expert dimensions in the route trace")
        if shared_intermediate % args.mlen:
            raise ValueError(
                f"shared intermediate={shared_intermediate} must be divisible by MLEN={args.mlen}"
            )
        if shared_gate_kind not in {"none", "sigmoid"}:
            raise ValueError(f"unsupported shared gate {shared_gate_kind!r}")
    topk_indices = [[int(value) for value in row] for row in routing["topk_indices"]]
    topk_weights = [[float(value) for value in row] for row in routing["topk_weights"]]
    pair_count = rows * top_k
    if len(topk_indices) != rows or len(topk_weights) != rows:
        raise ValueError("trace topk row count does not match token_count")
    selected_experts = sorted({expert_id for row in topk_indices for expert_id in row})

    multi_expert_panel_slots = int(args.multi_expert_panel_slots)
    if multi_expert_panel_slots < 1 or multi_expert_panel_slots > 16:
        raise ValueError("--multi-expert-panel-slots must be in [1, 16]")
    if args.weight_panel_mode != "blocking" and args.execution_order != "expert_major":
        raise ValueError("buffered weight panels require --execution-order expert_major")
    if multi_expert_panel_slots > 1:
        if args.execution_order != "expert_major":
            raise ValueError("multi-expert panel pooling requires --execution-order expert_major")
        if args.weight_panel_mode != "blocking":
            raise ValueError(
                "multi-expert panel pooling is a separate scheduling path; "
                "use --weight-panel-mode blocking"
            )
    if args.weight_panel_mode == "blocking":
        panel_buffer_depth = 1
        inferred_mram_tile_capacity = 4
    else:
        panel_buffer_depth = int(args.panel_buffer_depth)
        if panel_buffer_depth < 2 or panel_buffer_depth > 16:
            raise ValueError("--panel-buffer-depth must be in [2, 16]")
        if args.weight_panel_mode == "pingpong" and panel_buffer_depth != 2:
            raise ValueError("pingpong requires --panel-buffer-depth 2; use ring for deeper buffering")
        inferred_mram_tile_capacity = 4 * panel_buffer_depth
    inferred_mram_tile_capacity = max(inferred_mram_tile_capacity, 4 * multi_expert_panel_slots)
    mram_tile_capacity = (
        inferred_mram_tile_capacity
        if args.mram_tile_capacity is None
        else int(args.mram_tile_capacity)
    )
    required_panel_tiles = 4 * max(panel_buffer_depth, multi_expert_panel_slots)
    if mram_tile_capacity < required_panel_tiles:
        raise ValueError(
            f"--mram-tile-capacity={mram_tile_capacity} cannot hold "
            f"{max(panel_buffer_depth, multi_expert_panel_slots)} four-tile panels"
        )
    prog = PlenaCompiler(
        mlen=args.mlen,
        blen=args.blen,
        real_data_ratio=hw.real_data_ratio,
        mram_tile_capacity=mram_tile_capacity,
    )
    input_tensors: dict[str, torch.Tensor] = {}

    physical_rows = max(args.blen, math.ceil(rows / args.blen) * args.blen)
    shared_gate_elements = hidden if args.include_shared_expert and shared_gate_kind == "sigmoid" else 0
    vram_preload = torch.zeros(physical_rows * hidden + shared_gate_elements, dtype=torch.bfloat16)
    x = _synthetic_x(rows, hidden, mode=args.input_mode, seed=args.seed)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="TraceReplayX",
        tensor=x,
        vram_addr=0,
        physical_shape=(physical_rows, hidden),
        vram_preload=vram_preload,
    )
    shared_gate_weight = None
    shared_gate_weight_row = None
    if args.include_shared_expert and shared_gate_kind == "sigmoid":
        if args.weight_mode == "zeros":
            shared_gate_weight = torch.zeros((1, hidden), dtype=torch.bfloat16)
        else:
            shared_gate_weight = torch.linspace(-0.25, 0.25, hidden, dtype=torch.float32).reshape(1, hidden)
            shared_gate_weight = shared_gate_weight.to(torch.bfloat16)
        shared_gate_weight_row = prestage_bf16_vram_matrix(
            prog=prog,
            name="SharedExpertGateRow",
            tensor=shared_gate_weight,
            vram_addr=physical_rows * hidden,
            physical_shape=(1, hidden),
            vram_preload=vram_preload,
        )

    gate_inputs, gate_base, gate_stride, gate_values = _build_selected_dummy_weight_table(
        prog,
        prefix="QwenGate",
        selected_experts=selected_experts,
        num_experts=num_experts,
        shape=(hidden, intermediate),
        input_tensors=input_tensors,
        storage_order=args.hbm_weight_layout,
        weight_mode=args.weight_mode,
        projection_index=0,
    )
    up_inputs, up_base, up_stride, up_values = _build_selected_dummy_weight_table(
        prog,
        prefix="QwenUp",
        selected_experts=selected_experts,
        num_experts=num_experts,
        shape=(hidden, intermediate),
        input_tensors=input_tensors,
        storage_order=args.hbm_weight_layout,
        weight_mode=args.weight_mode,
        projection_index=1,
    )
    down_inputs, down_base, down_stride, down_values = _build_selected_dummy_weight_table(
        prog,
        prefix="QwenDown",
        selected_experts=selected_experts,
        num_experts=num_experts,
        shape=(intermediate, hidden),
        input_tensors=input_tensors,
        storage_order=args.hbm_weight_layout,
        weight_mode=args.weight_mode,
        projection_index=2,
    )
    weight_templates = (gate_inputs[0], up_inputs[0], down_inputs[0])
    weight_table_bases = (gate_base, up_base, down_base)
    weight_table_strides = (gate_stride, up_stride, down_stride)

    shared_weight_vars = None
    shared_weight_values = None
    if args.include_shared_expert:
        shared_gate_var, shared_gate_value = _build_static_weight(
            prog,
            name="SharedExpertWGate",
            shape=(hidden, shared_intermediate),
            input_tensors=input_tensors,
            storage_order=args.hbm_weight_layout,
            weight_mode=args.weight_mode,
            pattern_index=101,
        )
        shared_up_var, shared_up_value = _build_static_weight(
            prog,
            name="SharedExpertWUp",
            shape=(hidden, shared_intermediate),
            input_tensors=input_tensors,
            storage_order=args.hbm_weight_layout,
            weight_mode=args.weight_mode,
            pattern_index=102,
        )
        shared_down_var, shared_down_value = _build_static_weight(
            prog,
            name="SharedExpertWDown",
            shape=(shared_intermediate, hidden),
            input_tensors=input_tensors,
            storage_order=args.hbm_weight_layout,
            weight_mode=args.weight_mode,
            pattern_index=103,
        )
        shared_weight_vars = (shared_gate_var, shared_up_var, shared_down_var)
        shared_weight_values = (shared_gate_value, shared_up_value, shared_down_value)

    zero = prog.fp_var("decoder_zero", size=1)
    scalar_rows = max(rows, args.blen)
    one = prog.fp_var("decoder_one", size=scalar_rows)
    neg_alpha = prog.fp_var("decoder_neg_alpha", size=scalar_rows)
    limit_pos = prog.fp_var("decoder_unused_limit_pos", size=scalar_rows)
    limit_neg = prog.fp_var("decoder_unused_limit_neg", size=scalar_rows)
    shared_zero_row = prog.fp_var("decoder_shared_zero_row", size=args.mlen)
    topk_weight_var = prog.fp_var("trace_topk_weights", size=pair_count)
    route_fp_scratch = prog.fp_var("trace_route_fp_scratch", size=args.mlen)
    shared_gate_fp_scratch = (
        prog.fp_var("shared_gate_fp_scratch", size=scalar_rows)
        if args.include_shared_expert and shared_gate_kind == "sigmoid"
        else None
    )
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
        stage="accumulator_init",
        name="trace_acc_zero",
    )

    policy_name = str(model.get("policy_name", "trace_replay_moe"))
    activation_policy = str(model.get("activation_policy", "standard_swiglu"))
    expert_groups = _expert_major_groups(topk_indices)
    if args.execution_order == "pair_major":
        execution_groups = [
            (topk_indices[pair_idx // top_k][pair_idx % top_k], [pair_idx])
            for pair_idx in range(pair_count)
        ]
    else:
        execution_groups = expert_groups

    grouped_jobs: list[dict[str, Any]] = []
    for group_idx, (expert_id, pair_indices) in enumerate(execution_groups):
        token_indices = [pair_idx // top_k for pair_idx in pair_indices]
        if any(topk_indices[pair_idx // top_k][pair_idx % top_k] != expert_id for pair_idx in pair_indices):
            raise AssertionError(f"group {group_idx} mixes expert ids")
        if len(set(token_indices)) != len(token_indices):
            raise ValueError(f"expert {expert_id} was selected twice by one token")

        if args.execution_order == "pair_major":
            gathered = prog.moe_gather_token_rows_from_vram_v0(
                x_vram,
                token_indices=token_indices,
                hidden=hidden,
                zero_row=shared_zero_row,
                policy_name=policy_name,
                name=f"trace_pair{pair_indices[0]}_vram_gather_t{token_indices[0]}",
            )
            expert_out = prog.moe_dynamic_expert_pair_v0(
                gathered,
                weight_templates,
                weight_table_bases=weight_table_bases,
                weight_table_strides=weight_table_strides,
                expert_indices_int_base=topk_indices_int_base,
                weights_fp_base=topk_weights_fp_base,
                pair_idx=pair_indices[0],
                bias_tables=None,
                rows=args.blen,
                intermediate=intermediate,
                constants=(zero, limit_pos, limit_neg, one, neg_alpha),
                zero_row=shared_zero_row,
                route_fp_scratch=route_fp_scratch,
                policy_name=policy_name,
                activation_policy=activation_policy,
                name=f"trace_pair{pair_indices[0]}",
            )
            active_rows = [0]
        else:
            gathered = prog.moe_gather_token_rows_compact_from_vram_v0(
                x_vram,
                token_indices=token_indices,
                hidden=hidden,
                zero_row=shared_zero_row,
                policy_name=policy_name,
                name=f"trace_group{group_idx}_e{expert_id}_gather",
            )
            if multi_expert_panel_slots > 1:
                grouped_jobs.append(
                    {
                        "group_idx": group_idx,
                        "expert_id": expert_id,
                        "pair_indices": pair_indices,
                        "token_indices": token_indices,
                        "gathered": gathered,
                        "rows": len(pair_indices),
                        "projection_rows": max(
                            args.mlen,
                            gathered.physical_shape[0],
                            math.ceil(len(pair_indices) / args.blen) * args.blen,
                        ),
                        "name": f"trace_group{group_idx}_e{expert_id}",
                    }
                )
                continue
            else:
                expert_out = prog.moe_dynamic_expert_group_v0(
                    gathered,
                    weight_templates,
                    weight_table_bases=weight_table_bases,
                    weight_table_strides=weight_table_strides,
                    expert_indices_int_base=topk_indices_int_base,
                    weights_fp_base=topk_weights_fp_base,
                    pair_indices=pair_indices,
                    bias_tables=None,
                    rows=len(pair_indices),
                    intermediate=intermediate,
                    constants=(zero, limit_pos, limit_neg, one, neg_alpha),
                    zero_row=shared_zero_row,
                    route_fp_scratch=route_fp_scratch,
                    policy_name=policy_name,
                    activation_policy=activation_policy,
                    weight_panel_mode=args.weight_panel_mode,
                    panel_k_tiles=4,
                    panel_buffer_depth=panel_buffer_depth,
                    name=f"trace_group{group_idx}_e{expert_id}",
                )
                active_rows = list(range(len(pair_indices)))

        prog.moe_scatter_add_active_rows_v0(
            accumulator,
            expert_out,
            token_indices=token_indices,
            active_rows=active_rows,
            hidden=hidden,
            policy_name=policy_name,
            name=f"trace_group{group_idx}_scatter",
        )

    if grouped_jobs:
        representative_pairs = [job["pair_indices"][0] for job in grouped_jobs]
        gathered_inputs = [job["gathered"] for job in grouped_jobs]
        projection_rows = [job["projection_rows"] for job in grouped_jobs]
        job_names = [job["name"] for job in grouped_jobs]

        gate_outputs = prog.moe_dynamic_linear_projection_panel_pool_v0(
            gathered_inputs,
            weight_templates[0],
            expert_indices_int_base=topk_indices_int_base,
            pair_indices=representative_pairs,
            table_base=gate_base,
            per_expert_stride=gate_stride,
            names=[f"{name}_gate" for name in job_names],
            physical_shapes=[(rows_, weight_templates[0].physical_shape[1]) for rows_ in projection_rows],
            panel_pool_slots=multi_expert_panel_slots,
            panel_k_tiles=4,
        )
        up_outputs = prog.moe_dynamic_linear_projection_panel_pool_v0(
            gathered_inputs,
            weight_templates[1],
            expert_indices_int_base=topk_indices_int_base,
            pair_indices=representative_pairs,
            table_base=up_base,
            per_expert_stride=up_stride,
            names=[f"{name}_up" for name in job_names],
            physical_shapes=[(rows_, weight_templates[1].physical_shape[1]) for rows_ in projection_rows],
            panel_pool_slots=multi_expert_panel_slots,
            panel_k_tiles=4,
        )
        activated_outputs = [
            prog.moe_expert_activation_v0(
                gate,
                up,
                rows=job["rows"],
                intermediate=intermediate,
                constants=(zero, limit_pos, limit_neg, one, neg_alpha),
                activation_policy=activation_policy,
                policy_name=policy_name,
                stage="expert_activation",
                name=job["name"],
            )
            for job, gate, up in zip(grouped_jobs, gate_outputs, up_outputs, strict=True)
        ]
        expert_outputs = prog.moe_dynamic_linear_projection_panel_pool_v0(
            activated_outputs,
            weight_templates[2],
            expert_indices_int_base=topk_indices_int_base,
            pair_indices=representative_pairs,
            table_base=down_base,
            per_expert_stride=down_stride,
            names=[f"{name}_out" for name in job_names],
            physical_shapes=[(rows_, weight_templates[2].physical_shape[1]) for rows_ in projection_rows],
            panel_pool_slots=multi_expert_panel_slots,
            panel_k_tiles=4,
        )

        for job, expert_out in zip(grouped_jobs, expert_outputs, strict=True):
            active_rows = list(range(job["rows"]))
            route = prog.moe_materialize_route_weights_for_active_rows_v0(
                weights_fp_base=topk_weights_fp_base,
                pair_indices=job["pair_indices"],
                active_rows=active_rows,
                rows=job["rows"],
                hidden=hidden,
                zero_row=shared_zero_row,
                fp_scratch=route_fp_scratch,
                policy_name=policy_name,
                stage="expert_route_weight",
                name=f"{job['name']}_route",
            )
            prog.vram_mul(expert_out, route, num_rows=job["rows"])
            prog.moe_scatter_add_active_rows_v0(
                accumulator,
                expert_out,
                token_indices=job["token_indices"],
                active_rows=active_rows,
                hidden=hidden,
                policy_name=policy_name,
                name=f"trace_group{job['group_idx']}_scatter",
            )

    output_var = accumulator
    if args.include_shared_expert:
        assert shared_weight_vars is not None
        shared_out = prog.moe_shared_expert_v0(
            x_vram,
            shared_weight_vars,
            rows=rows,
            intermediate=shared_intermediate,
            constants=(zero, limit_pos, limit_neg, one, neg_alpha),
            gate_weight_row=shared_gate_weight_row,
            gate_fp_scratch=shared_gate_fp_scratch,
            zero_row=shared_zero_row,
            activation_policy="standard_swiglu",
            policy_name=policy_name,
            name="trace_shared_expert",
        )
        output_var = prog.moe_combine_shared_and_routed_v0(
            accumulator,
            shared_out,
            rows=rows,
            policy_name=policy_name,
            name="trace_complete_moe_combine",
        )

    isa = prog.compile()
    fp_preload_ends = [
        neg_alpha.address + neg_alpha.size,
        topk_weight_var.address + topk_weight_var.size,
        route_fp_scratch.address + route_fp_scratch.size,
        shared_zero_row.address + shared_zero_row.size,
    ]
    if shared_gate_fp_scratch is not None:
        fp_preload_ends.append(shared_gate_fp_scratch.address + shared_gate_fp_scratch.size)
    fp_preload_len = max(fp_preload_ends)
    fp_preload = [0.0] * fp_preload_len
    for idx in range(one.size):
        fp_preload[one.address + idx] = 1.0
    for idx in range(neg_alpha.size):
        fp_preload[neg_alpha.address + idx] = -1.0
    flat_weights = _flatten_float(topk_weights)
    for idx, value in enumerate(flat_weights):
        fp_preload[topk_weights_fp_base + idx] = value

    int_preload = torch.tensor(_flatten_int(topk_indices), dtype=torch.int32)
    if args.weight_mode == "zeros":
        golden = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    else:
        if args.input_mode != "random":
            raise ValueError("weight_mode='exact' requires input_mode='random' for a nonzero gate")
        golden = _nonzero_routed_golden(
            x,
            topk_indices,
            topk_weights,
            gate_values,
            up_values,
            down_values,
        )
        if args.include_shared_expert:
            assert shared_weight_values is not None
            shared_gate_value, shared_up_value, shared_down_value = shared_weight_values
            if shared_gate_kind == "sigmoid":
                assert shared_gate_weight is not None
                shared_golden = qwen2_shared_expert_golden(
                    x,
                    shared_gate_value,
                    shared_up_value,
                    shared_down_value,
                    shared_gate_weight,
                    project=_mxfp_project,
                    project_from_vram=_mxfp_project,
                )
            else:
                shared_golden = deepseek_shared_expert_golden(
                    x,
                    shared_gate_value,
                    shared_up_value,
                    shared_down_value,
                    project=_mxfp_project,
                    project_from_vram=_mxfp_project,
                )
            golden = combine_shared_and_routed_golden(golden, shared_golden)
    comparison_params = _comparison_params(
        prog.get_vram_addr(output_var.name),
        rows,
        hidden,
        args.mlen,
        physical_rows=accumulator.physical_shape[0],
    )
    tensor_layouts = infer_hbm_tensor_layouts(input_tensors)
    for layout in tensor_layouts.values():
        layout.update({"storage_order": args.hbm_weight_layout, "tile_size": args.mlen})
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
                "weight_mode": args.weight_mode,
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
        "model_name": model["name"],
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
        "weight_mode": args.weight_mode,
        "include_shared_expert": args.include_shared_expert,
        "shared_experts": shared_experts,
        "shared_intermediate": shared_intermediate,
        "shared_gate": shared_gate_kind,
        "execution_order": args.execution_order,
        "weight_panel_mode": args.weight_panel_mode,
        "panel_buffer_depth": panel_buffer_depth,
        "multi_expert_panel_slots": multi_expert_panel_slots,
        "mram_tile_capacity": mram_tile_capacity,
        "hbm_weight_layout": args.hbm_weight_layout,
        "coalesce_hbm_bursts": args.coalesce_hbm_bursts,
        "scoreboard_serialize": args.scoreboard_serialize,
        "group_count": len(expert_groups),
        "weight_load_reuse_factor": pair_count / len(expert_groups),
        "topk_indices_int_base": topk_indices_int_base,
        "topk_weights_fp_base": topk_weights_fp_base,
        "weight_table_bases": {"gate": gate_base, "up": up_base, "down": down_base},
        "weight_table_strides": {"gate": gate_stride, "up": up_stride, "down": down_stride},
        "routed_weight_bytes_per_expert": sum(weight_table_strides),
        "expected_routed_weight_bytes": (
            pair_count if args.execution_order == "pair_major" else len(expert_groups)
        )
        * sum(weight_table_strides),
        "expected_shared_weight_bytes": (
            0
            if not args.include_shared_expert
            else _expert_stride(prog, (hidden, shared_intermediate)) * 2
            + _expert_stride(prog, (shared_intermediate, hidden))
        ),
        "hbm_input_tensor_count": len(input_tensors),
        "asm_lines": len(isa.splitlines()),
        "measurement_note": (
            "Rust/Ramulator simulation cycles from fixed-route replay; "
            "absolute cycle accuracy still requires RTL primitive calibration."
        ),
        "comparison_params": comparison_params,
    }
    # Keep the historical filename for existing callers while exposing a
    # model-neutral artifact name for the generalized Qwen/DeepSeek harness.
    write_json(build_dir / "moe_trace_replay_manifest.json", manifest)
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
        timing_model=args.timing_model,
        scoreboard_serialize=args.scoreboard_serialize,
        coalesce_hbm_bursts=args.coalesce_hbm_bursts,
    )
    results, params = compare_emulator_output(build_dir)
    rel_rms = _relative_rms(results["simulated_values"], results["golden_values"])
    nonzero_gate_passed = rel_rms <= args.functional_rel_rms_threshold
    comparison_passed = bool(results.get("test_pass", results.get("allclose_pass", False)))
    expected_hbm_bytes = (
        int(manifest["expected_routed_weight_bytes"])
        + int(manifest["expected_shared_weight_bytes"])
    )
    measured_hbm_bytes = metrics.get("hbm_bytes_read")
    byte_gate = {
        "passed": measured_hbm_bytes == expected_hbm_bytes,
        "expected_weight_bytes": expected_hbm_bytes,
        "measured_physical_hbm_bytes": measured_hbm_bytes,
        "delta_bytes": None if measured_hbm_bytes is None else measured_hbm_bytes - expected_hbm_bytes,
        "scope": "weights_only; activations and optional shared gate row are prestaged in Vector SRAM",
    }
    gate = {
        "passed": nonzero_gate_passed if manifest["weight_mode"] == "exact" else comparison_passed,
        "allclose_pass": bool(results.get("allclose_pass", False)),
        "rel_rms": rel_rms,
        "rel_rms_threshold": args.functional_rel_rms_threshold,
        "relative_match_rate": results.get("relative_match_rate"),
        "max_error": results.get("max_error"),
        "relative_error": results.get("relative_error"),
        "zero_input_gate": manifest["input_mode"] == "zeros" or manifest["weight_mode"] == "zeros",
        "gate_kind": "nonzero_reference" if manifest["weight_mode"] == "exact" else "zero_output_shape_smoke",
    }
    repeat_summary = None
    if args.repeat_gate:
        repeat_summary = run_emulator_repeat_gate(
            build_dir,
            repeats=args.repeat_gate,
            threads=args.emu_threads,
            stage_profile=False,
            timing_model=args.timing_model,
            scoreboard_serialize=args.scoreboard_serialize,
            coalesce_hbm_bursts=args.coalesce_hbm_bursts,
            # Same isolation as the main run above. Without it the repeats fall back
            # to the shared emulator directory, so concurrent campaign workers race
            # on vram_dump.bin / fpsram_dump.bin and copy each other's dumps into
            # their own build dirs.
            dump_cwd=build_dir,
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
        "functional_gate": gate,
        # Compatibility alias for result readers written before nonzero gates
        # were added. New code should consume ``functional_gate``.
        "zero_input_smoke_gate": gate,
        "hbm_weight_byte_gate": byte_gate,
        "repeat_gate": repeat_summary,
    }
    write_json(build_dir / "moe_trace_replay_results.json", summary)
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
        write_json(build_dir / "moe_trace_replay_results.json", summary)
        write_json(build_dir / "qwen3_trace_replay_results.json", summary)
        write_json(build_dir / "gather_scatter_results.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not gate["passed"]:
        raise AssertionError(f"trace replay functional gate failed: {gate}")
    if not byte_gate["passed"]:
        raise AssertionError(f"trace replay HBM weight-byte gate failed: {byte_gate}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--emu-threads", type=int, default=1)
    parser.add_argument("--input-mode", choices=("zeros", "random"), default="zeros")
    parser.add_argument("--weight-mode", choices=("zeros", "exact"), default="zeros")
    parser.add_argument(
        "--include-shared-expert",
        action="store_true",
        help="Execute the trace's shared expert and final routed+shared combine.",
    )
    parser.add_argument(
        "--functional-rel-rms-threshold",
        type=float,
        default=0.01,
        help="Hard rel-RMS threshold for the nonzero functional reference gate.",
    )
    parser.add_argument(
        "--execution-order",
        choices=("pair_major", "expert_major"),
        default="pair_major",
        help="pair_major preserves the legacy lowering; expert_major groups equal expert ids and reuses weights.",
    )
    parser.add_argument(
        "--hbm-weight-layout",
        choices=("row_major", "tile_major"),
        default="row_major",
    )
    parser.add_argument(
        "--weight-panel-mode",
        choices=("blocking", "pingpong", "ring"),
        default="blocking",
        help="Opt-in grouped projection scheduling over disjoint four-tile Matrix-SRAM panels.",
    )
    parser.add_argument(
        "--panel-buffer-depth",
        type=int,
        default=2,
        help="Number of four-tile panels reserved by pingpong/ring mode; ignored by blocking.",
    )
    parser.add_argument(
        "--multi-expert-panel-slots",
        type=int,
        default=1,
        help=(
            "Opt-in global Matrix-SRAM panel pool across grouped experts. "
            "1 preserves the existing path; values 2..16 require expert_major "
            "and weight-panel-mode blocking."
        ),
    )
    parser.add_argument(
        "--mram-tile-capacity",
        type=int,
        help=(
            "Physical Matrix-SRAM capacity in MLEN-by-MLEN cells. Use the same "
            "value across a panel-depth ablation; PLENA's modeled 512-KiB SRAM is 64 cells."
        ),
    )
    parser.add_argument("--coalesce-hbm-bursts", action="store_true")
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument("--repeat-gate", type=int, default=0)
    parser.add_argument(
        "--timing-model",
        choices=("serial", "scoreboard"),
        default="serial",
        help="Emulator timing model: serial (default) or the pipelined scoreboard.",
    )
    parser.add_argument(
        "--scoreboard-serialize",
        action="store_true",
        help="Force scoreboard issue order to reproduce serial timing; validation control only.",
    )
    parser.add_argument("--keep-dumps", dest="cleanup_dumps", action="store_false")
    parser.add_argument("--no-run", action="store_true")
    parser.set_defaults(cleanup_dumps=True)
    args = parser.parse_args()
    if args.scoreboard_serialize and args.timing_model != "scoreboard":
        parser.error("--scoreboard-serialize requires --timing-model scoreboard")
    run_trace(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
