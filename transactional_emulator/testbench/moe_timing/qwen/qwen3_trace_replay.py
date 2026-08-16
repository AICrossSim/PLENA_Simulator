#!/usr/bin/env python3
"""Replay a Qwen3 true route trace through a routed MoE emulator program.

This is a TIMING harness. It drives the emulator with DUMMY (all-zero) expert
weights to measure cycles and HBM bytes, so the expert arithmetic it reports is
shape and movement, not numerics. Numerical correctness of the gather / expert /
scatter / route-weight math is validated separately by the routed-MoE op tests
(routed_moe/gpt_oss_moe_*_test.py and real_layer0); this replay measures timing
on top of that validated substrate.

Routing, however, is not dummy. The device selects its own experts with V_TOPK
from logits rebuilt out of the trace, and `_router_gate` requires the ids it
picks to be the ones the true router picked. That is the only check here with no
tolerance in it: the zero-input smoke gate compares an all-zero accumulator
against an all-zero golden, which holds for any routing at all.

Not covered: the router GEMM. The trace kept the top-k, not the logits, so the
hidden->num_experts projection cannot be replayed faithfully -- see
`router_logits` for what the reconstruction does and does not preserve.
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
from transactional_emulator.testbench.gpt_oss_testkit import _decode_bf16_dump, _decode_u32_dump  # noqa: E402
from transactional_emulator.testbench.layout_utils import infer_hbm_tensor_layouts, prestage_bf16_vram_matrix  # noqa: E402
from transactional_emulator.testbench.models.gpt_oss.attention_semantics_test import _comparison_params  # noqa: E402
from transactional_emulator.testbench.routed_moe.gpt_oss_moe_gather_scatter_test import _align_to  # noqa: E402
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim  # noqa: E402
from transactional_emulator.testbench.moe_timing.replay.validate_route_trace import validate_trace  # noqa: E402
from transactional_emulator.testbench.moe_timing.qwen.router_logits import reconstruct_router_logits  # noqa: E402
from transactional_emulator.testbench.moe_timing.qwen.utils import OUT_ROOT, ensure_paths, load_json, write_json  # noqa: E402
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402


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


#: Slots in the emulator's INT and FP scalar SRAMs, both `vec![...; 1024]` in
#: `transactional_emulator/src/accelerator/scalar_sram.rs`, and the same 1024 the
#: compiler's FPRAM allocator enforces.
#:
#: Not a tunable. The token count a trace can be replayed at is bounded by it:
#: the route-weight table takes top_k slots per token, so Qwen3's top_k of 8 runs
#: out somewhere near ninety tokens.
_SCALAR_SRAM_SLOTS = 1024

#: How far the device's route weights may sit from softmax over the logits staged
#: for it.
#:
#: Tighter than `router_logits._WEIGHT_RTOL`, and deliberately: that one absorbs
#: BF16's error between the reconstruction and the *trace*, while this compares
#: two evaluations of the same expression over the same BF16 row -- the device's
#: and torch's. Loosening this to match would stop the gate noticing a device that
#: computes softmax differently.
_ROUTER_WEIGHT_RTOL = 0.02
_ROUTER_WEIGHT_ATOL = 2e-3


def _check_scalar_sram_capacity(*, rows: int, top_k: int, fp_base: int, fp_tail: int) -> None:
    """Refuse a token count whose routing tables cannot fit in scalar SRAM.

    Without this the ceiling still holds -- the compiler's FPRAM allocator raises
    when the route-weight table runs off the end -- but it reports it as
    ``FPRAM overflow: need 128 at addr 897``, which names neither the trace nor
    the token count that caused it. This says which, and what would fit.

    Deliberately reads the base from an already-allocated variable rather than
    recomputing the layout, so it cannot drift from it: at MLEN 128 both refuse
    at exactly 94 tokens. The allocator stays the backstop if a future variable
    lands between the base read here and the two allocations below.
    """
    pair_count = rows * top_k
    if pair_count > _SCALAR_SRAM_SLOTS:
        raise ValueError(
            f"trace has {rows} tokens x top_k {top_k} = {pair_count} routed pairs, but the "
            f"expert-id table is indexed into {_SCALAR_SRAM_SLOTS} INT SRAM slots; "
            f"replay at most {_SCALAR_SRAM_SLOTS // top_k} tokens per trace"
        )
    needed = fp_base + pair_count + fp_tail
    if needed > _SCALAR_SRAM_SLOTS:
        affordable = (_SCALAR_SRAM_SLOTS - fp_base - fp_tail) // top_k
        raise ValueError(
            f"trace has {rows} tokens, which needs FP SRAM up to slot {needed} of "
            f"{_SCALAR_SRAM_SLOTS}: the route-weight table takes top_k={top_k} slots per "
            f"token on top of a {fp_base}-slot base and a {fp_tail}-slot scratch row. "
            f"Replay at most {affordable} tokens per trace"
        )


def _vram_extent(rows: int, cols: int, mlen: int) -> int:
    """Addresses a VRAM matrix of this physical shape occupies.

    Mirrors the column-block-major layout ``prestage_bf16_vram_matrix`` writes:
    the row is split into ``ceil(cols / mlen)`` blocks and each block holds
    ``rows * mlen`` addresses, so a matrix narrower than MLEN still costs a full
    block.
    """
    return math.ceil(cols / mlen) * rows * mlen


def _router_logits_layout(logits: torch.Tensor, *, mlen: int, blen: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Fold a ``(tokens, num_experts)`` logit block into the rows V_TOPK scans.

    V_TOPK reads ``num_experts`` contiguous VRAM addresses from the row base. When
    ``num_experts`` exceeds MLEN the emitter expects each token to occupy
    ``ceil(num_experts / mlen)`` *consecutive* MLEN-wide rows and addresses the
    first of them, so the block is reshaped rather than kept one row per token --
    at MLEN 64 a 128-expert row that stayed one row per token would have V_TOPK
    read 64 real logits and then 64 belonging to the next token.

    Any tail past ``num_experts`` is padding. ``prestage_bf16_vram_matrix`` fills
    it with zeros, which is below every reconstructed logit by construction, so
    padding cannot be selected.
    """
    tokens, num_experts = logits.shape
    blocks = math.ceil(num_experts / mlen)
    if blocks == 1:
        physical_rows = max(blen, math.ceil(tokens / blen) * blen)
        return logits, (physical_rows, num_experts)

    padded = torch.zeros(tokens, blocks * mlen, dtype=logits.dtype)
    padded[:, :num_experts] = logits
    folded = padded.reshape(tokens * blocks, mlen)
    physical_rows = max(blen, math.ceil(folded.shape[0] / blen) * blen)
    return folded, (physical_rows, mlen)


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

    # The router's logit row shares VRAM with the hidden state, so both extents
    # have to be known before either is staged.
    router_logits = reconstruct_router_logits(topk_indices, topk_weights, num_experts)
    logits_tensor, logits_physical = _router_logits_layout(router_logits, mlen=args.mlen, blen=args.blen)
    logits_base = _align_to(_vram_extent(physical_rows, hidden, args.mlen), args.mlen * args.mlen)
    vram_preload = torch.zeros(
        logits_base + _vram_extent(logits_physical[0], logits_physical[1], args.mlen),
        dtype=torch.bfloat16,
    )

    x = _synthetic_x(rows, hidden, mode=args.input_mode, seed=args.seed)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="TraceReplayX",
        tensor=x,
        vram_addr=0,
        physical_shape=(physical_rows, hidden),
        vram_preload=vram_preload,
    )
    logits_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="TraceReplayRouterLogits",
        tensor=logits_tensor,
        vram_addr=logits_base,
        physical_shape=logits_physical,
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
    # Before the two token-sized allocations, so the refusal names the trace
    # rather than surfacing as the allocator's address arithmetic.
    _check_scalar_sram_capacity(
        rows=rows,
        top_k=top_k,
        fp_base=shared_zero_row.address + shared_zero_row.size,
        fp_tail=args.mlen,
    )
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
        stage="accumulator_init",
        name="trace_acc_zero",
    )

    # Every token is selected before any expert address is computed. The pair loop
    # below loads expert ids from `topk_indices_int_base` with S_LD_INT; emitting
    # V_TOPK inside that loop would leave the first pairs reading a location the
    # router has not written yet -- zeros, so every early pair would route to
    # expert 0 and the program would still run to completion.
    for token_idx in range(rows):
        prog.moe_router_select_v0(
            logits_vram,
            token_idx=token_idx,
            weights_fp_base=topk_weights_fp_base + token_idx * top_k,
            indices_int_base=topk_indices_int_base + token_idx * top_k,
            num_experts=num_experts,
            top_k=top_k,
            policy_name="qwen3_moe",
            name=f"trace_router_t{token_idx}",
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

    # `topk_weights_fp_base` and `topk_indices_int_base` are left at zero on
    # purpose: V_TOPK writes both. Seeding them with the trace's values would
    # make the functional gate below unfalsifiable -- a router that emitted
    # nothing would still be read back as agreeing with the trace.
    int_preload = torch.zeros(pair_count, dtype=torch.int32)
    # Expert weights are dummy zeros, so the expected output is zero for any input.
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
        "router": {
            "on_device": True,
            "v_topk_count": rows,
            "logits_vram_addr": logits_base,
            "logits_rows": logits_physical[0],
            "logits_cols": logits_physical[1],
            "expert_blocks_per_token": math.ceil(num_experts / args.mlen),
            # Stated in the artifact, not only in a docstring: this is the number
            # someone will quote, and "the router is measured" is false in a way
            # that matters if the projection is not in it.
            "router_gemm_included": False,
            "reconstruction": (
                "logits rebuilt from the trace's topk_indices/topk_weights, which is all "
                "the capture kept; the selected experts and their softmax weights are the "
                "trace's, the 120 unselected logits are a floor rather than their true "
                "values, and the hidden->num_experts projection that produced them is not "
                "emitted"
            ),
        },
        "weight_table_bases": {"gate": gate_base, "up": up_base, "down": down_base},
        "weight_table_strides": {"gate": gate_stride, "up": up_stride, "down": down_stride},
        "hbm_input_tensor_count": len(input_tensors),
        "asm_lines": len(isa.splitlines()),
        "measurement_note": "self-consistent upper bound, absolute accuracy pending RTL (Window 2)",
        "comparison_params": comparison_params,
    }
    write_json(build_dir / "qwen3_trace_replay_manifest.json", manifest)
    # `staged_router_logits` is the tensor that went into VRAM, folded shape and
    # all -- not the block it was folded from. The gate derives its expectation
    # from this so it is checking the row the device actually read; recomputing
    # the reconstruction there would leave every transform between the two
    # unchecked by the one gate whose job is to compare them.
    return {
        "trace": trace,
        "build_dir": build_dir,
        "manifest": manifest,
        "staged_router_logits": logits_tensor,
    }


def _router_gate(
    build_dir: Path,
    manifest: dict[str, Any],
    trace: dict[str, Any],
    staged_logits: torch.Tensor,
) -> dict[str, Any]:
    """Did the device select the experts the true router selected?

    This is the check the replay did not have. Its existing gate compares the
    accumulator against an all-zero golden, which is what dummy expert weights
    produce for *any* routing -- so a program that sent every token to expert 0
    passed it. Expert ids are integers with no tolerance, so this one cannot.

    ``staged_logits`` is the tensor ``build_artifacts`` put in VRAM, so the
    weights are checked against softmax over the row the device actually read.
    Reconstructing it here instead would leave the staging path -- the fold, the
    narrowing, the placement -- unchecked by the gate that exists to compare the
    device against it, and would repeat the reconstruction for every run.
    """
    top_k = int(trace["model"]["top_k"])
    rows = int(trace["workload"]["token_count"])
    num_experts = int(trace["model"]["num_experts"])
    want_indices = torch.tensor(trace["routing"]["topk_indices"], dtype=torch.int64)

    indices_base = int(manifest["topk_indices_int_base"])
    weights_base = int(manifest["topk_weights_fp_base"])
    got_indices = _decode_u32_dump(build_dir / "intsram_dump.bin")[indices_base : indices_base + rows * top_k].reshape(
        rows, top_k
    )
    got_weights = _decode_bf16_dump(build_dir / "fpsram_dump.bin")[weights_base : weights_base + rows * top_k].reshape(
        rows, top_k
    )

    # Undo the VRAM fold: whatever shape the rows were staged in, one token's
    # logits are contiguous and start at a row boundary, so this recovers them
    # for both the one-row and the multi-row-per-token layouts.
    per_token = staged_logits.reshape(rows, -1)[:, :num_experts]
    want_weights = torch.softmax(torch.topk(per_token.float(), k=top_k, dim=-1).values, dim=-1)

    index_mismatches = (got_indices != want_indices).nonzero().tolist()
    weight_close = torch.allclose(got_weights.float(), want_weights, rtol=_ROUTER_WEIGHT_RTOL, atol=_ROUTER_WEIGHT_ATOL)
    return {
        "gate_kind": "device_selected_experts_match_trace",
        "passed": bool(not index_mismatches and weight_close),
        "expert_ids_match": bool(not index_mismatches),
        "route_weights_match": bool(weight_close),
        "tokens_checked": rows,
        "pairs_checked": rows * top_k,
        # Bounded: a wrong-by-one-expert run on a 4k-token trace would otherwise
        # print 32k coordinates into the results JSON.
        "index_mismatch_coordinates": index_mismatches[:16],
        "index_mismatch_count": len(index_mismatches),
        "max_weight_error": float((got_weights.float() - want_weights).abs().max()),
    }


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
    # Read before `compare_emulator_output`, and before the dump cleanup at the
    # end of this function can remove them.
    router_gate = _router_gate(build_dir, manifest, built["trace"], built["staged_router_logits"])
    results, params = compare_emulator_output(build_dir)
    gate = {
        "passed": bool(results.get("test_pass", results.get("allclose_pass", False))),
        "allclose_pass": bool(results.get("allclose_pass", False)),
        "relative_match_rate": results.get("relative_match_rate"),
        "max_error": results.get("max_error"),
        "relative_error": results.get("relative_error"),
        "zero_input_gate": manifest["input_mode"] == "zeros",
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
        "zero_input_smoke_gate": gate,
        "router_gate": router_gate,
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
    # Both are reported before either can raise, so a failing run still leaves a
    # readable artifact -- and both are checked, so a router fault is not masked
    # by the smoke gate happening to pass (with dummy zero weights, it always
    # does regardless of which experts were chosen).
    if not router_gate["passed"]:
        raise AssertionError(f"trace replay router gate failed: {router_gate}")
    if not gate["passed"]:
        raise AssertionError(f"trace replay zero-input smoke gate failed: {gate}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """The CLI, separated from ``main`` so callers can reach its defaults.

    A hand-built Namespace would drift from these silently: ``add_hw_args``
    contributes seven options this file never mentions, and a test that omits one
    fails on an AttributeError from deep inside ``setup_hw``.
    """
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_trace(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
