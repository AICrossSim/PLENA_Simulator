"""Transactional Rust proofs for chunked Mamba-2 and KDA prefill.

The hidden width and head count are compact so S128 remains practical in CI,
while Mamba keeps its 64-wide head and 128-wide state and KDA keeps its
128x128 recurrent matrix.  Production schedulers, descriptors, Matrix
projections, L_SCATTER_M and Rust X_STATE are used without a test-only opcode.

Long prompts are explicitly streamed through the fixed X_STATE chunk workspace.
This matters: lowering eight descriptors that all read the same first 16 rows
would produce legal machine code but would not execute an S128 prompt.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from analytic_models.reference.kimi_k3_kda import (
    KdaConvWeights,
    KdaShape,
    KdaXState,
    kda_state_engine_prefill,
)
from analytic_models.reference.nemotron3_mamba import (
    Mamba2Shape,
    Mamba2State,
    mamba_state_engine_prefill,
)
from analytic_models.reference.state_precision import StateStorage
from compiler.aten.kda.scheduler import KdaScheduleConfig, KimiK3KdaScheduler
from compiler.aten.mamba.scheduler import (
    MambaHbmLayout,
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)
from compiler.aten.plena import PlenaCompiler
from compiler.aten.state.isa_lowering import (
    KdaLayerMemoryMap,
    MambaLayerMemoryMap,
    LoweredMambaIsaProgram,
    lower_kda_trace_to_existing_isa,
    lower_mamba_trace_to_existing_isa,
)
from transactional_emulator.testbench.aten.configurable import setup_hw
from transactional_emulator.testbench.aten.golden import (
    _active_precision_settings,
    _rms_norm_vector_ref,
)
from transactional_emulator.testbench.emulator_runner import (
    compare_emulator_output,
    run_emulator,
)
from transactional_emulator.testbench.gpt_oss_testkit import (
    _comparison_params_for,
)
from transactional_emulator.testbench.layout_utils import (
    infer_hbm_tensor_layouts,
    prestage_bf16_vram_matrix,
)
from transactional_emulator.testbench.models.kimi3.connected_blocks_test import (
    BLEN,
    EPS,
    MLEN,
    TensorSet,
    _bf16,
    _bf16_layout,
    _linear,
    _rms,
    _set_matrix_kv_plain_bf16,
    _sigmoid,
)
from transactional_emulator.testbench.models.kimi3.kda_connected_test import (
    KDA_ARENA_BASE,
    KDA_DIM,
    KDA_HEADS,
    KDA_KERNEL,
    KDA_WEIGHT_BASE,
    KDA_WEIGHT_OFFSETS,
    KDA_WEIGHT_SLOT,
    _allocate_kda_moe_constants,
    _patch_hbm as _patch_kda_hbm,
    _register_kda_weights,
)
from transactional_emulator.testbench.models.nemotron3.mamba_connected_test import (
    _patch_hbm as _patch_mamba_hbm,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


CHUNK = 16
HIDDEN = 64
MAMBA_HEADS = 1
MAMBA_HEAD_DIM = 64
MAMBA_STATE_DIM = 128
MAMBA_GROUPS = 1
MAMBA_KERNEL = 4


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _mamba_weights() -> dict[str, torch.Tensor]:
    shape = Mamba2Shape(
        HIDDEN,
        MAMBA_HEADS,
        MAMBA_HEAD_DIM,
        MAMBA_STATE_DIM,
        MAMBA_GROUPS,
        MAMBA_KERNEL,
    )
    projection_width = _align(shape.projection_size, MLEN)
    projection = torch.zeros(HIDDEN, projection_width, dtype=torch.bfloat16)
    scales = torch.empty(shape.projection_size, dtype=torch.float32)
    scales[: shape.d_inner] = 0.25
    scales[shape.d_inner : shape.d_inner + shape.conv_channels] = 0.125
    scales[-shape.num_heads :] = 0.0625
    logical_columns = torch.arange(shape.projection_size)
    projection[logical_columns % HIDDEN, logical_columns] = scales.to(
        torch.bfloat16
    )

    output = torch.eye(HIDDEN, dtype=torch.bfloat16)
    conv = torch.tensor([0.125, -0.25, 0.375, 0.5], dtype=torch.bfloat16)
    return {
        "W_MAMBA_IN": projection,
        "W_MAMBA_OUT": output,
        "W_MAMBA_NORM": torch.ones(1, shape.d_inner, dtype=torch.bfloat16),
        "CONV_WEIGHT": conv.repeat(shape.conv_channels, 1),
        "CONV_BIAS": torch.zeros(shape.conv_channels, dtype=torch.bfloat16),
        "A_LOG": torch.full((shape.num_heads,), -0.5, dtype=torch.bfloat16),
        "DT_BIAS": torch.zeros(shape.num_heads, dtype=torch.bfloat16),
        "D_SKIP": torch.ones(shape.num_heads, dtype=torch.bfloat16),
    }


def _mamba_golden(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Mamba2State]:
    shape = Mamba2Shape(
        HIDDEN,
        MAMBA_HEADS,
        MAMBA_HEAD_DIM,
        MAMBA_STATE_DIM,
        MAMBA_GROUPS,
        MAMBA_KERNEL,
    )
    projected = _linear(hidden, weights["W_MAMBA_IN"])[
        :, : shape.projection_size
    ]
    state_output, state = mamba_state_engine_prefill(
        projected.unsqueeze(0),
        Mamba2State.zeros(shape, 1),
        weights["CONV_WEIGHT"],
        weights["A_LOG"],
        weights["DT_BIAS"],
        weights["D_SKIP"],
        shape,
        conv_bias=weights["CONV_BIAS"],
        state_storage=StateStorage.FP32,
    )
    state_output = _bf16(state_output.squeeze(0))
    gate = projected[:, : shape.d_inner]
    gated = _bf16(
        state_output.float()
        * _bf16(gate.float() * _sigmoid(gate).float()).float()
    )
    grouped = gated.reshape(hidden.shape[0], shape.groups, -1)
    normalized = torch.cat(
        [
            _rms_norm_vector_ref(group, EPS, _active_precision_settings())
            for group in grouped.unbind(1)
        ],
        dim=-1,
    )
    normalized = _bf16(normalized * weights["W_MAMBA_NORM"].float())
    return _linear(normalized, weights["W_MAMBA_OUT"]), state


def _kda_golden(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    conv_weight: torch.Tensor,
) -> tuple[torch.Tensor, KdaXState]:
    shape = KdaShape(
        HIDDEN,
        KDA_HEADS,
        KDA_DIM,
        KDA_DIM,
        KDA_KERNEL,
        chunk_size=CHUNK,
    )
    q = _linear(hidden, weights["W_kda_q"])
    k = _linear(hidden, weights["W_kda_k"])
    v = _linear(hidden, weights["W_kda_v"])
    decay = _linear(
        _linear(hidden, weights["W_kda_decay_a"]),
        weights["W_kda_decay_b"],
    )
    beta = _linear(hidden, weights["W_kda_beta"])[:, :KDA_HEADS]
    projected = torch.cat((q, k, v, decay, beta), dim=-1).unsqueeze(0)
    zero_bias = torch.zeros(KDA_DIM, dtype=torch.bfloat16)
    state_output, state = kda_state_engine_prefill(
        projected,
        KdaXState.zeros(shape, 1),
        KdaConvWeights(
            q=conv_weight,
            k=conv_weight,
            v=conv_weight,
            q_bias=zero_bias,
            k_bias=zero_bias,
            v_bias=zero_bias,
        ),
        torch.zeros(KDA_HEADS, dtype=torch.bfloat16),
        torch.zeros(KDA_HEADS, KDA_DIM, dtype=torch.bfloat16),
        shape,
        state_storage=StateStorage.FP32,
        conv_state_storage=StateStorage.BF16,
    )
    value = _bf16(state_output.squeeze(0))
    normalized = _bf16(
        _rms(value).float() * weights["W_kda_norm"].float()
    )
    output_gate = _linear(hidden, weights["W_kda_gate"])
    gated = _bf16(normalized.float() * _sigmoid(output_gate).float())
    return _linear(gated, weights["W_kda_out"]), state


def _emit_streamed_chunks(
    prog: PlenaCompiler,
    lowered: LoweredMambaIsaProgram,
    trace,
    prompt,
    fixed_hidden,
    outputs,
) -> int:
    trace_by_index = {event.index: event for event in trace.events}
    active_offset: int | None = None
    chunk_count = 0
    for lowered_event in lowered.events:
        trace_event = trace_by_index[lowered_event.event_index]
        descriptor = trace_event.descriptor
        if lowered_event.operation in {"IN_PROJECTION", "KDA_QKV_PROJECTION"}:
            assert descriptor is not None
            if descriptor.token_offset == active_offset:
                raise AssertionError("chunk projection was emitted twice")
            active_offset = descriptor.token_offset
            chunk_count += 1
            prog.emit_comment(
                f"STATE_PREFILL_CHUNK offset={active_offset} "
                f"valid={descriptor.valid_tokens}"
            )
            prog.vram_copy_region(
                fixed_hidden,
                prompt,
                num_rows=descriptor.valid_tokens,
                num_cols=HIDDEN,
                src_row_offset=active_offset,
            )
        prog.emit(lowered_event.assembly)
        if lowered_event.operation in {"OUT_PROJECTION", "KDA_OUT_PROJECTION"}:
            assert descriptor is not None
            prog.vram_copy_region(
                outputs,
                fixed_hidden,
                num_rows=descriptor.valid_tokens,
                num_cols=HIDDEN,
                dst_row_offset=descriptor.token_offset,
            )
    return chunk_count


def _prepare_program(
    build_dir: Path,
    *,
    tokens: int,
    workspace_end: int,
    hidden_addr: int,
) -> tuple[PlenaCompiler, torch.Tensor, object, object, object]:
    hw = setup_hw(
        argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None),
        build_dir,
    )
    _set_matrix_kv_plain_bf16()
    prog = PlenaCompiler(
        mlen=MLEN,
        blen=BLEN,
        real_data_ratio=hw.real_data_ratio,
        compact_matrix_loops=True,
    )
    prog.vram_allocator._vmm.mark_used(0, workspace_end, name="STATE_WORKSPACE")
    physical_prompt_rows = _align(tokens, MLEN)
    prompt_addr = _align(workspace_end, MLEN * MLEN)
    preload_size = prompt_addr + physical_prompt_rows * HIDDEN
    vram_preload = torch.zeros(preload_size, dtype=torch.bfloat16)
    hidden_value = (
        0.125
        + torch.arange(tokens * HIDDEN, dtype=torch.float32).reshape(tokens, HIDDEN)
        / 8192.0
    ).to(torch.bfloat16)
    fixed_hidden = prestage_bf16_vram_matrix(
        prog=prog,
        name="STATE_HIDDEN_CHUNK",
        tensor=torch.zeros(CHUNK, HIDDEN, dtype=torch.bfloat16),
        vram_addr=hidden_addr,
        physical_shape=(CHUNK, HIDDEN),
        vram_preload=vram_preload,
    )
    prompt = prestage_bf16_vram_matrix(
        prog=prog,
        name="STATE_PROMPT",
        tensor=hidden_value,
        vram_addr=prompt_addr,
        physical_shape=(physical_prompt_rows, HIDDEN),
        vram_preload=vram_preload,
    )
    outputs = prog.alloc(
        "STATE_PREFILL_OUTPUT",
        rows=tokens,
        cols=HIDDEN,
        strict=False,
        physical_shape=(physical_prompt_rows, HIDDEN),
    )
    return prog, vram_preload, fixed_hidden, prompt, outputs


def _run_and_compare(
    *,
    model: str,
    tokens: int,
    build_dir: Path,
    prog: PlenaCompiler,
    tensors: TensorSet,
    assembly: str,
    golden: torch.Tensor,
    outputs,
    fp_preload: list[float],
    vram_preload: torch.Tensor,
    hbm_size: int,
    patch_hbm,
    chunk_count: int,
    state_elements: int,
    conv_state_elements: int,
) -> dict[str, object]:
    layouts = infer_hbm_tensor_layouts(tensors.values)
    for name in tensors.bf16_names:
        layouts[name] = _bf16_layout(tensors.values[name])
    hbm_addrs = {
        name: prog._compiler.get_hbm_layout(name).hbm_base_addr
        for name in tensors.values
    }
    create_sim_env(
        tensors.values,
        assembly,
        {"original_output": golden},
        fp_preload=fp_preload,
        int_preload=[0] * 16,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=layouts,
    )
    create_mem_for_sim(
        data_size=MLEN,
        mode="behave_sim",
        asm=f"{model}_state_prefill_s{tokens}",
        specified_data_order=sorted(tensors.values, key=hbm_addrs.__getitem__),
        build_path=build_dir,
        input_tensors=tensors.values,
        tensor_layouts=layouts,
        hbm_addrs=hbm_addrs,
    )
    patch_hbm(build_dir / "hbm_for_behave_sim.bin")
    params = _comparison_params_for(
        outputs,
        rows=tokens,
        hidden=HIDDEN,
        mlen=MLEN,
        golden=golden,
    )
    params.update({"atol": 0.02, "rtol": 0.03, "min_allclose_match_rate": 100.0})
    (build_dir / "comparison_params.json").write_text(
        json.dumps(params, indent=2) + "\n"
    )
    (build_dir / "generated_asm_code.asm").write_text(assembly)
    (build_dir / "hbm_size.txt").write_text(f"{hbm_size}\n")
    profile_path = build_dir / "state_profile.json"
    metrics = run_emulator(
        build_dir,
        hbm_size=hbm_size,
        stage_profile=True,
        state_profile_out=profile_path,
        dump_cwd=build_dir,
    )
    results, _ = compare_emulator_output(build_dir, verbose=False)
    if float(results.get("allclose_match_rate", 0.0)) < 100.0:
        raise AssertionError(f"{model} S{tokens} prefill mismatch: {results}")
    state_profile = json.loads(profile_path.read_text())
    command_profile = state_profile["summary"]
    prefill_commands = [
        command
        for command in state_profile["commands"]
        if command["subop"] == "prefill"
    ]
    reset_commands = [
        command
        for command in state_profile["commands"]
        if command["subop"] == "reset"
    ]
    if len(prefill_commands) != chunk_count:
        raise AssertionError(
            f"{model}: expected {chunk_count} X_STATE chunks, "
            f"profile saw {len(prefill_commands)}"
        )
    if len(reset_commands) != 1:
        raise AssertionError(f"{model}: expected one X_STATE reset")
    if command_profile["valid_tokens"] != tokens:
        raise AssertionError(
            f"{model}: expected {tokens} state tokens, "
            f"profile saw {command_profile['valid_tokens']}"
        )
    summary = {
        "model": model,
        "phase": "prefill",
        "tokens": tokens,
        "chunk_size": CHUNK,
        "chunks": chunk_count,
        "x_state_commands": command_profile["commands"],
        "state_elements": state_elements,
        "conv_state_elements": conv_state_elements,
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "sim_cycles": metrics.get("sim_latency_cycles"),
        "max_abs_error": results.get("max_error"),
        "allclose_match_rate": results.get("allclose_match_rate"),
        "state_hbm_read_bytes": command_profile["state_hbm_read_bytes"],
        "state_hbm_write_bytes": command_profile["state_hbm_write_bytes"],
        "projection_read_stall_cycles": command_profile[
            "projection_read_stall_cycles"
        ],
        "bank_stall_cycles": command_profile["bank_stall_cycles"],
        "hbm_bytes": hbm_size,
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_mamba_prefill(build_dir: Path, *, tokens: int) -> dict[str, object]:
    build_dir.mkdir(parents=True, exist_ok=True)
    config = MambaScheduleConfig(
        phase=SchedulePhase.PREFILL,
        sequence_length=tokens,
        chunk_size=CHUNK,
        matrix_input_features=HIDDEN,
        mamba_layer_ids=(0,),
        mamba_num_heads=MAMBA_HEADS,
        mamba_head_dim=MAMBA_HEAD_DIM,
        mamba_state_dim=MAMBA_STATE_DIM,
        mamba_groups=MAMBA_GROUPS,
        mamba_conv_kernel=MAMBA_KERNEL,
        mamba_hbm_arena_base=0x1_0000,
    )
    scheduler = Nemotron3MambaScheduler(config)
    trace = scheduler.build()
    lowered = lower_mamba_trace_to_existing_isa(trace, descriptor_base=0)
    memories = {
        event.memory
        for event in lowered.events
        if isinstance(event.memory, MambaLayerMemoryMap)
    }
    if not memories:
        raise AssertionError("Mamba prefill produced no physical memory map")
    memory = next(iter(memories))
    workspace_end = max(item.normalization_scratch_vram_addr + MLEN for item in memories)
    prog, vram_preload, fixed_hidden, prompt, outputs = _prepare_program(
        build_dir,
        tokens=tokens,
        workspace_end=workspace_end,
        hidden_addr=memory.hidden_vram_addr,
    )
    prog.fp_var("zero", 1)
    prog.fp_var("attention_scale", 1)
    prog.fp_var("attention_negative_infinity", 1)
    prog.fp_var("attention_online_softmax_workspace", 253)
    mamba_eps = prog.fp_var("mamba_eps_backup", 1)
    mamba_reciprocal = prog.fp_var("mamba_reciprocal_backup", 1)
    mamba_one = prog.fp_var("mamba_one_backup", 1)
    group_width = MAMBA_HEADS * MAMBA_HEAD_DIM // MAMBA_GROUPS
    fp_preload = [0.0] * (mamba_one.address + mamba_one.size)
    fp_preload[mamba_eps.address] = EPS
    fp_preload[mamba_reciprocal.address] = 1.0 / group_width
    fp_preload[mamba_one.address] = 1.0
    weights = _mamba_weights()
    tensors = TensorSet(values={}, bf16_names=set())
    layout = MambaHbmLayout.build(config)
    for name, value, address, bf16 in (
        ("W_MAMBA_IN", weights["W_MAMBA_IN"], memory.input_projection_weight_hbm_addr, False),
        ("W_MAMBA_OUT", weights["W_MAMBA_OUT"], memory.output_projection_weight_hbm_addr, False),
        ("W_MAMBA_NORM", weights["W_MAMBA_NORM"], memory.norm_weight_hbm_addr, True),
    ):
        tensors.add(name, value, bf16=bf16)
        prog.input(
            name,
            shape=tuple(value.shape),
            physical_shape=tuple(value.shape),
            hbm_addr=address,
            real_data_ratio=2.0 if bf16 else None,
        )
    chunk_count = _emit_streamed_chunks(
        prog, lowered, trace, prompt, fixed_hidden, outputs
    )
    assembly = prog.compile()
    hidden_value = (
        0.125
        + torch.arange(tokens * HIDDEN, dtype=torch.float32).reshape(tokens, HIDDEN)
        / 8192.0
    ).to(torch.bfloat16)
    golden, final_state = _mamba_golden(hidden_value, weights)
    hbm_size = _align(max(prog._next_hbm_addr, layout.arena_end), 64)
    descriptor = next(event.descriptor for event in trace.events if event.descriptor)

    def patch(path: Path) -> None:
        _patch_mamba_hbm(
            path,
            descriptor_image=lowered.descriptor_image,
            descriptor_base=lowered.descriptor_base,
            layout_descriptor_image=lowered.layout_descriptor_image,
            layout_descriptor_base=lowered.layout_descriptor_base,
            writes={
                descriptor.payload.conv_weight_addr: weights["CONV_WEIGHT"],
                descriptor.payload.conv_bias_addr: weights["CONV_BIAS"],
                descriptor.payload.a_log_addr: weights["A_LOG"],
                descriptor.payload.dt_bias_addr: weights["DT_BIAS"],
                descriptor.payload.d_skip_addr: weights["D_SKIP"],
            },
            minimum_size=hbm_size,
        )

    return _run_and_compare(
        model="nemotron3_mamba",
        tokens=tokens,
        build_dir=build_dir,
        prog=prog,
        tensors=tensors,
        assembly=assembly,
        golden=golden,
        outputs=outputs,
        fp_preload=fp_preload,
        vram_preload=vram_preload,
        hbm_size=hbm_size,
        patch_hbm=patch,
        chunk_count=chunk_count,
        state_elements=final_state.ssm.numel(),
        conv_state_elements=final_state.conv.numel(),
    )


def build_kda_prefill(build_dir: Path, *, tokens: int) -> dict[str, object]:
    build_dir.mkdir(parents=True, exist_ok=True)
    config = KdaScheduleConfig(
        phase=SchedulePhase.PREFILL,
        sequence_length=tokens,
        chunk_size=CHUNK,
        matrix_input_features=HIDDEN,
        kda_layer_ids=(0,),
        kda_num_heads=KDA_HEADS,
        hbm_arena_base=KDA_ARENA_BASE,
        projection_weight_hbm_base=KDA_WEIGHT_BASE,
        projection_weight_layer_stride=len(KDA_WEIGHT_OFFSETS) * KDA_WEIGHT_SLOT,
        projection_weight_offsets=KDA_WEIGHT_OFFSETS,
    )
    scheduler = KimiK3KdaScheduler(config)
    trace = scheduler.build()
    lowered = lower_kda_trace_to_existing_isa(trace, descriptor_base=0)
    memories = {
        event.memory
        for event in lowered.events
        if isinstance(event.memory, KdaLayerMemoryMap)
    }
    if not memories:
        raise AssertionError("KDA prefill produced no physical memory map")
    memory = next(iter(memories))
    workspace_end = max(item.normalization_scratch_vram_addr + MLEN for item in memories)
    prog, vram_preload, fixed_hidden, prompt, outputs = _prepare_program(
        build_dir,
        tokens=tokens,
        workspace_end=workspace_end,
        hidden_addr=memory.hidden_vram_addr,
    )
    tensors = TensorSet(values={}, bf16_names=set())
    _constants, fp_preload = _allocate_kda_moe_constants(prog)
    weights = _register_kda_weights(prog, tensors, memory)
    chunk_count = _emit_streamed_chunks(
        prog, lowered, trace, prompt, fixed_hidden, outputs
    )
    assembly = prog.compile()
    hidden_value = (
        0.125
        + torch.arange(tokens * HIDDEN, dtype=torch.float32).reshape(tokens, HIDDEN)
        / 8192.0
    ).to(torch.bfloat16)
    conv_weight = torch.tensor(
        [0.125, -0.25, 0.375, 0.5], dtype=torch.bfloat16
    ).repeat(KDA_DIM, 1)
    golden, final_state = _kda_golden(hidden_value, weights, conv_weight)
    layout = scheduler.hbm_layout()
    hbm_size = _align(
        max(
            prog._next_hbm_addr,
            layout.realized_arena_bytes(len(trace.events)),
            lowered.layout_descriptor_base + len(lowered.layout_descriptor_image),
            KDA_WEIGHT_BASE + len(KDA_WEIGHT_OFFSETS) * KDA_WEIGHT_SLOT,
        ),
        64,
    )
    zeros = torch.zeros(KDA_DIM, dtype=torch.bfloat16)

    def patch(path: Path) -> None:
        _patch_kda_hbm(
            path,
            descriptor_image=lowered.descriptor_image,
            descriptor_base=lowered.descriptor_base,
            layout_descriptor_image=lowered.layout_descriptor_image,
            layout_descriptor_base=lowered.layout_descriptor_base,
            parameter_writes={
                layout.address("q_conv_weight", 0): conv_weight,
                layout.address("k_conv_weight", 0): conv_weight,
                layout.address("v_conv_weight", 0): conv_weight,
                layout.address("q_conv_bias", 0): zeros,
                layout.address("k_conv_bias", 0): zeros,
                layout.address("v_conv_bias", 0): zeros,
                layout.address("a_log", 0): torch.zeros(KDA_HEADS),
                layout.address("dt_bias", 0): zeros,
            },
            minimum_size=hbm_size,
        )

    return _run_and_compare(
        model="kimi_k3_kda",
        tokens=tokens,
        build_dir=build_dir,
        prog=prog,
        tensors=tensors,
        assembly=assembly,
        golden=golden,
        outputs=outputs,
        fp_preload=fp_preload,
        vram_preload=vram_preload,
        hbm_size=hbm_size,
        patch_hbm=patch,
        chunk_count=chunk_count,
        state_elements=final_state.recurrent.numel(),
        conv_state_elements=final_state.conv.numel(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mamba", "kda", "all"), default="all")
    parser.add_argument("--tokens", type=int, choices=(16, 128), default=16)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("transactional_emulator/testbench/build/state_prefill_connected"),
    )
    args = parser.parse_args()
    root = args.build_dir.expanduser().resolve()
    summaries = []
    if args.model in {"mamba", "all"}:
        summaries.append(build_mamba_prefill(root / "mamba", tokens=args.tokens))
    if args.model in {"kda", "all"}:
        summaries.append(build_kda_prefill(root / "kda", tokens=args.tokens))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
