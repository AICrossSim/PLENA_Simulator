"""Rust numerical proof for one physically connected Nemotron Mamba layer."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from compiler.aten.mamba.scheduler import (
    MambaHbmLayout,
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)
from compiler.aten.nemotron3.blocks import (
    NemotronMoeConstants,
    NemotronMoeShape,
    NemotronMoeWeights,
    emit_nemotron_moe_block,
)
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_routed_moe import moe_stage_marker
from compiler.aten.state.isa_lowering import (
    MambaLayerMemoryMap,
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
from transactional_emulator.testbench.gpt_oss_testkit import _comparison_params_for
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
    _exact,
    _linear,
    _register_expert_table,
    _register_weight,
    _rms,
    _set_matrix_kv_plain_bf16,
    _sigmoid,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


ARENA_BASE = 0x1_0000
HEADS = 64
HEAD_DIM = 64
STATE_DIM = 128
GROUPS = 8
KERNEL = 4
D_INNER = HEADS * HEAD_DIM
GROUP_STATE = GROUPS * STATE_DIM
CONV_CHANNELS = D_INNER + 2 * GROUP_STATE
PROJECTION = D_INNER + CONV_CHANNELS + HEADS


def _bf16_bytes(value: torch.Tensor) -> bytes:
    return value.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().astype("<u2", copy=False).tobytes()


def _patch_hbm(
    path: Path,
    *,
    descriptor_image: bytes,
    descriptor_base: int,
    layout_descriptor_image: bytes,
    layout_descriptor_base: int,
    writes: dict[int, torch.Tensor],
    minimum_size: int,
) -> None:
    with path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        current = stream.tell()
        if current < minimum_size:
            stream.write(b"\x00" * (minimum_size - current))
        stream.seek(descriptor_base)
        stream.write(descriptor_image)
        stream.seek(layout_descriptor_base)
        stream.write(layout_descriptor_image)
        for address, value in sorted(writes.items()):
            stream.seek(address)
            stream.write(_bf16_bytes(value))


def _read_vram_row(
    path: Path,
    *,
    address: int,
    width: int,
    physical_rows: int,
) -> torch.Tensor:
    """Read one logical row from VRAM's column-block-major BF16 layout."""
    blocks = []
    with path.open("rb") as stream:
        for column_block in range(math.ceil(width / MLEN)):
            element_offset = address + column_block * physical_rows * MLEN
            stream.seek(element_offset * 2)
            raw = stream.read(MLEN * 2)
            if len(raw) != MLEN * 2:
                raise AssertionError("VRAM dump ended inside a logical row")
            blocks.append(torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone())
    return torch.cat(blocks)[:width].reshape(1, width)


def _weights() -> dict[str, torch.Tensor]:
    in_proj = torch.zeros(MLEN, PROJECTION, dtype=torch.bfloat16)
    source = torch.arange(PROJECTION) % MLEN
    destination = torch.arange(PROJECTION)
    scales = torch.zeros(PROJECTION)
    scales[:D_INNER] = 0.5
    scales[D_INNER : D_INNER + D_INNER] = 0.25
    scales[D_INNER + D_INNER : D_INNER + CONV_CHANNELS] = 0.125
    in_proj[source, destination] = scales.to(torch.bfloat16)

    out_proj = torch.zeros(D_INNER, MLEN, dtype=torch.bfloat16)
    for row in range(D_INNER):
        out_proj[row, row % MLEN] = 1.0 / HEADS
    conv = torch.zeros(CONV_CHANNELS, KERNEL, dtype=torch.bfloat16)
    conv[:, -1] = 1.0
    return {
        "W_MAMBA_IN": in_proj,
        "W_MAMBA_OUT": out_proj,
        "W_MAMBA_NORM": torch.ones(1, D_INNER, dtype=torch.bfloat16),
        "CONV_WEIGHT": conv,
        "CONV_BIAS": torch.zeros(CONV_CHANNELS, dtype=torch.bfloat16),
        "A_LOG": torch.zeros(HEADS, dtype=torch.bfloat16),
        "DT_BIAS": torch.zeros(HEADS, dtype=torch.bfloat16),
        "D_SKIP": torch.ones(HEADS, dtype=torch.bfloat16),
    }


def _golden(hidden: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    projected = _linear(hidden, weights["W_MAMBA_IN"])
    gate = projected[:, :D_INNER]
    xbc = projected[:, D_INNER : D_INNER + CONV_CHANNELS]
    dt_raw = projected[:, -HEADS:].float()

    # Zero initial convolution state and a [0,0,0,1] depthwise kernel.
    convolved = _bf16(torch.nn.functional.silu(xbc.float()))
    x = convolved[:, :D_INNER].reshape(1, HEADS, HEAD_DIM).float()
    b = convolved[:, D_INNER : D_INNER + GROUP_STATE].reshape(1, GROUPS, STATE_DIM).float()
    c = convolved[:, D_INNER + GROUP_STATE :].reshape(1, GROUPS, STATE_DIM).float()
    b = b.repeat_interleave(HEADS // GROUPS, dim=1)
    c = c.repeat_interleave(HEADS // GROUPS, dim=1)
    # Match the state engine's f32 left-to-right reduction.  torch.sum may use
    # a different reduction tree, which is mathematically equivalent but can
    # move the final BF16 result by one or two ULPs over 128 state elements.
    x_np = x.numpy()
    b_np = b.numpy()
    c_np = c.numpy()
    dt_np = np.log1p(np.exp(dt_raw.numpy(), dtype=np.float32), dtype=np.float32)
    state_out_np = np.empty((1, HEADS, HEAD_DIM), dtype=np.float32)
    for head in range(HEADS):
        dt_value = np.float32(dt_np[0, head])
        for position in range(HEAD_DIM):
            x_value = np.float32(x_np[0, head, position])
            reduced = np.float32(0.0)
            for state_index in range(STATE_DIM):
                updated = np.float32(np.float32(dt_value * b_np[0, head, state_index]) * x_value)
                reduced = np.float32(reduced + np.float32(updated * c_np[0, head, state_index]))
            state_out_np[0, head, position] = np.float32(reduced + x_value)
    state_out = _bf16(torch.from_numpy(state_out_np).reshape(1, D_INNER))

    gated = _bf16(state_out.float() * _bf16(gate.float() * _sigmoid(gate).float()).float())
    grouped = gated.reshape(1, GROUPS, D_INNER // GROUPS)
    normalized = torch.cat(
        [_rms_norm_vector_ref(group, EPS, _active_precision_settings()) for group in grouped.unbind(1)],
        dim=-1,
    )
    normalized = _bf16(normalized * weights["W_MAMBA_NORM"].float())
    return _linear(normalized, weights["W_MAMBA_OUT"])


def _register_moe(
    prog: PlenaCompiler,
    tensors: TensorSet,
) -> tuple[NemotronMoeShape, NemotronMoeWeights]:
    router = torch.zeros(MLEN, 4, dtype=torch.bfloat16)
    for expert in range(4):
        router[:, expert] = _exact((MLEN,), expert + 1, expert, scale=1 / 32)
    up_values = [_exact((MLEN, MLEN), expert + 2, expert + 1, 1 / 32) for expert in range(4)]
    down_values = [_exact((MLEN, MLEN), expert + 3, expert + 2, 1 / 32) for expert in range(4)]
    weights = NemotronMoeWeights(
        router=_register_weight(prog, tensors, "W_NEMOTRON_ROUTER", router, bf16=True),
        routed_up=_register_expert_table(prog, tensors, prefix="W_NEMOTRON_EXPERT_UP", values=up_values),
        routed_down=_register_expert_table(prog, tensors, prefix="W_NEMOTRON_EXPERT_DOWN", values=down_values),
        shared_up=_register_weight(
            prog,
            tensors,
            "W_NEMOTRON_SHARED_UP",
            _exact((MLEN, MLEN), 4, 1, 1 / 32),
        ),
        shared_down=_register_weight(
            prog,
            tensors,
            "W_NEMOTRON_SHARED_DOWN",
            _exact((MLEN, MLEN), 3, 4, 1 / 32),
        ),
    )
    return (
        NemotronMoeShape(
            hidden=MLEN,
            intermediate=MLEN,
            shared_intermediate=MLEN,
            num_experts=4,
            top_k=2,
        ),
        weights,
    )


def _nemotron_moe_golden(
    hidden: torch.Tensor,
    tensors: TensorSet,
    correction: torch.Tensor,
    *,
    precision=None,
) -> torch.Tensor:
    logits = _linear(hidden, tensors.values["W_NEMOTRON_ROUTER"])
    choice = logits.float() + correction[:, :4].float()
    ranked = sorted(
        range(choice.shape[1]),
        key=lambda expert: (-float(choice[0, expert]), expert),
    )
    indices = torch.tensor([ranked[:2]], dtype=torch.long)
    selected = torch.sigmoid(logits.float().gather(1, indices))
    selected = selected / selected.sum(-1, keepdim=True) * 2.5
    # V_TOPK keeps the selected scores in FPRAM, then S_MAP_V_FP
    # materializes them as BF16 before the expert output is scaled.
    selected = _bf16(selected, precision=precision)

    accumulator = torch.zeros_like(hidden)
    for pair in range(2):
        expert = int(indices[0, pair])
        up = _linear(
            hidden,
            tensors.references[f"W_NEMOTRON_EXPERT_UP_{expert}"],
        )
        activated = _bf16(torch.clamp(up.float(), min=0.0), precision=precision)
        activated = _bf16(activated.float() * activated.float(), precision=precision)
        output = _linear(
            activated,
            tensors.references[f"W_NEMOTRON_EXPERT_DOWN_{expert}"],
        )
        output = _bf16(output.float() * selected[0, pair].float(), precision=precision)
        accumulator = _bf16(accumulator.float() + output.float(), precision=precision)

    shared = _linear(hidden, tensors.values["W_NEMOTRON_SHARED_UP"])
    shared = _bf16(torch.clamp(shared.float(), min=0.0), precision=precision)
    shared = _bf16(shared.float() * shared.float(), precision=precision)
    shared = _linear(shared, tensors.values["W_NEMOTRON_SHARED_DOWN"])
    return _bf16(accumulator.float() + shared.float(), precision=precision)


def build_and_run(stage: str, build_dir: Path) -> dict[str, object]:
    if stage not in {"mamba", "moe", "mamba_moe"}:
        raise ValueError(f"unsupported stage {stage!r}")
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(
        argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None),
        build_dir,
    )
    _set_matrix_kv_plain_bf16()
    config = MambaScheduleConfig(
        phase=SchedulePhase.DECODE,
        decode_tokens=1,
        matrix_input_features=MLEN,
        mamba_layer_ids=(0,),
        mamba_hbm_arena_base=ARENA_BASE,
    )
    scheduler = Nemotron3MambaScheduler(config)
    trace = scheduler.build()
    lowered = lower_mamba_trace_to_existing_isa(trace, descriptor_base=0)
    memories = {event.memory for event in lowered.events if isinstance(event.memory, MambaLayerMemoryMap)}
    if len(memories) != 1:
        raise AssertionError(f"expected one Mamba memory map, got {len(memories)}")
    memory = memories.pop()
    descriptor = next(event.descriptor for event in trace.events if event.descriptor)
    layout = MambaHbmLayout.build(config)

    prog = PlenaCompiler(mlen=MLEN, blen=BLEN, real_data_ratio=hw.real_data_ratio)
    prog.emit_comment(moe_stage_marker("non_moe", "Nemotron connected-test prelude"))
    prog.fp_var("zero", 1)
    prog.fp_var("attention_scale", 1)
    prog.fp_var("attention_negative_infinity", 1)
    prog.fp_var("attention_online_softmax_workspace", 253)
    mamba_eps = prog.fp_var("mamba_eps_backup", 1)
    mamba_reciprocal = prog.fp_var("mamba_reciprocal_backup", 1)
    mamba_one = prog.fp_var("mamba_one_backup", 1)
    zero_row = prog.fp_var("zero_row", MLEN)
    block_eps = prog.fp_var("block_eps", 1)
    block_reciprocal = prog.fp_var("block_reciprocal", 1)
    route_scale = prog.fp_var("route_scale", 2)
    fp_preload = [0.0] * (route_scale.address + route_scale.size)
    fp_preload[mamba_eps.address] = EPS
    fp_preload[mamba_reciprocal.address] = 1.0 / 512.0
    fp_preload[mamba_one.address] = 1.0
    fp_preload[block_eps.address] = EPS
    fp_preload[block_reciprocal.address] = 1.0 / MLEN
    fp_preload[route_scale.address : route_scale.address + 2] = [2.5, 2.5]
    prog.vram_allocator._vmm.mark_used(
        0,
        memory.normalization_scratch_vram_addr + MLEN,
        name="MAMBA_PHYSICAL_WORKSPACE",
    )
    vram_preload = torch.zeros(
        memory.normalization_scratch_vram_addr + 2 * MLEN * MLEN,
        dtype=torch.bfloat16,
    )
    hidden_value = (0.25 + torch.arange(MLEN, dtype=torch.float32).reshape(1, -1) / 512.0).to(torch.bfloat16)
    hidden = prestage_bf16_vram_matrix(
        prog=prog,
        name="HIDDEN",
        tensor=hidden_value,
        vram_addr=memory.hidden_vram_addr,
        physical_shape=(BLEN, MLEN),
        vram_preload=vram_preload,
    )
    values = _weights()
    tensors = TensorSet(values={}, bf16_names=set())
    for name, value, address, bf16 in (
        ("W_MAMBA_IN", values["W_MAMBA_IN"], memory.input_projection_weight_hbm_addr, False),
        ("W_MAMBA_OUT", values["W_MAMBA_OUT"], memory.output_projection_weight_hbm_addr, False),
        ("W_MAMBA_NORM", values["W_MAMBA_NORM"], memory.norm_weight_hbm_addr, True),
    ):
        tensors.add(name, value, bf16=bf16)
        prog.input(
            name,
            shape=tuple(value.shape),
            physical_shape=tuple(value.shape),
            hbm_addr=address,
            real_data_ratio=2.0 if bf16 else None,
        )
    prog._next_hbm_addr = max(prog._next_hbm_addr, layout.arena_end)
    current = hidden
    mamba_prefix = None
    golden_input = hidden_value
    correction_value = torch.tensor(
        [[0.0, 0.125, 0.25, -0.125] + [0.0] * (MLEN - 4)],
        dtype=torch.bfloat16,
    )
    correction_addr = (memory.normalization_scratch_vram_addr + MLEN * MLEN - 1) // (MLEN * MLEN) * (MLEN * MLEN)
    correction = prestage_bf16_vram_matrix(
        prog=prog,
        name="MOE_CORRECTION",
        tensor=correction_value,
        vram_addr=correction_addr,
        physical_shape=(BLEN, MLEN),
        vram_preload=vram_preload,
    )

    if stage == "mamba_moe":
        mamba_residual = prog.vram_copy(hidden, name="mamba_residual", num_rows=1)
        mamba_input = prog.vram_copy(hidden, name="mamba_input", num_rows=1)
        prog.rms_norm(
            mamba_input,
            eps_offset=block_eps.address,
            reci_hid_offset=block_reciprocal.address,
        )
        prog.vram_copy_region(hidden, mamba_input, num_rows=1, num_cols=MLEN)
        golden_input = _rms(hidden_value)
    if stage in {"mamba", "mamba_moe"}:
        prog.emit(lowered.assembly)
        golden = _golden(golden_input, values)
    else:
        moe_shape, moe_weights = _register_moe(prog, tensors)
        current = emit_nemotron_moe_block(
            prog,
            current,
            shape=moe_shape,
            weights=moe_weights,
            correction_bias=correction,
            constants=NemotronMoeConstants(
                zero_row=zero_row,
                routed_scale=route_scale,
            ),
            rows=1,
            name="standalone_nemotron_moe",
        )
        golden = _nemotron_moe_golden(hidden_value, tensors, correction_value)
    if stage == "mamba_moe":
        current = prog.vram_copy(mamba_residual, name="prefix_after_mamba", num_rows=1)
        prog.vram_add(current, hidden, num_rows=1)
        mamba_prefix = current
        golden = _bf16(hidden_value.float() + golden.float())

        moe_residual = prog.vram_copy(current, name="moe_residual", num_rows=1)
        moe_input = prog.vram_copy(current, name="moe_input", num_rows=1)
        prog.rms_norm(
            moe_input,
            eps_offset=block_eps.address,
            reci_hid_offset=block_reciprocal.address,
        )
        moe_input_golden = _rms(golden)
        moe_shape, moe_weights = _register_moe(prog, tensors)
        moe_out = emit_nemotron_moe_block(
            prog,
            moe_input,
            shape=moe_shape,
            weights=moe_weights,
            correction_bias=correction,
            constants=NemotronMoeConstants(
                zero_row=zero_row,
                routed_scale=route_scale,
            ),
            rows=1,
            name="connected_nemotron_moe",
        )
        prog.vram_add(moe_residual, moe_out, num_rows=1)
        current = moe_residual
        golden = _bf16(golden.float() + _nemotron_moe_golden(moe_input_golden, tensors, correction_value).float())
    assembly = prog.compile()

    input_tensors = tensors.values
    tensor_layouts = infer_hbm_tensor_layouts(input_tensors)
    for name in tensors.bf16_names:
        tensor_layouts[name] = _bf16_layout(input_tensors[name])
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_sim_env(
        input_tensors,
        assembly,
        {"original_output": golden},
        fp_preload=fp_preload,
        int_preload=[0] * 16,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    create_mem_for_sim(
        data_size=MLEN,
        mode="behave_sim",
        asm=f"nemotron3_connected_{stage}",
        specified_data_order=sorted(input_tensors, key=hbm_addrs.__getitem__),
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    hbm_size = (
        math.ceil(
            max(
                prog._next_hbm_addr,
                layout.arena_end,
                lowered.layout_descriptor_base + len(lowered.layout_descriptor_image),
            )
            / 64
        )
        * 64
    )
    _patch_hbm(
        build_dir / "hbm_for_behave_sim.bin",
        descriptor_image=lowered.descriptor_image,
        descriptor_base=lowered.descriptor_base,
        layout_descriptor_image=lowered.layout_descriptor_image,
        layout_descriptor_base=lowered.layout_descriptor_base,
        writes={
            descriptor.payload.conv_weight_addr: values["CONV_WEIGHT"],
            descriptor.payload.conv_bias_addr: values["CONV_BIAS"],
            descriptor.payload.a_log_addr: values["A_LOG"],
            descriptor.payload.dt_bias_addr: values["DT_BIAS"],
            descriptor.payload.d_skip_addr: values["D_SKIP"],
        },
        minimum_size=hbm_size,
    )

    params = _comparison_params_for(
        current,
        rows=1,
        hidden=MLEN,
        mlen=MLEN,
        golden=golden,
    )
    model_atol = 0.05 if stage == "mamba_moe" else 0.008
    params.update({"atol": model_atol, "rtol": 0.02, "min_allclose_match_rate": 100.0})
    if stage != "moe" and float(golden.abs().max()) < 0.05:
        raise AssertionError("Mamba golden signal is too small")
    (build_dir / "comparison_params.json").write_text(json.dumps(params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(assembly)
    (build_dir / "hbm_size.txt").write_text(f"{hbm_size}\n")

    metrics = run_emulator(
        build_dir,
        stage_profile=True,
        state_profile_out=build_dir / "state_profile.json",
        dump_cwd=build_dir,
    )
    results, _ = compare_emulator_output(build_dir, verbose=False)
    rate = float(results.get("allclose_match_rate", 0.0))
    if rate < 100.0:
        raise AssertionError(f"Mamba Rust comparison failed: max_abs={results.get('max_error')}, rate={rate}%")
    edge_max_abs_error = None
    if stage == "mamba_moe":
        assert mamba_prefix is not None
        vram_dump = build_dir / "vram_dump.bin"
        mamba_actual = _read_vram_row(
            vram_dump,
            address=prog.get_vram_addr(mamba_prefix.name),
            width=MLEN,
            physical_rows=mamba_prefix.physical_shape[0],
        )
        final_actual = _read_vram_row(
            vram_dump,
            address=prog.get_vram_addr(current.name),
            width=MLEN,
            physical_rows=current.physical_shape[0],
        )
        edge_golden = _bf16(
            mamba_actual.float() + _nemotron_moe_golden(_rms(mamba_actual), tensors, correction_value).float()
        )
        edge_max_abs_error = float((final_actual.float() - edge_golden.float()).abs().max())
        if not torch.allclose(final_actual.float(), edge_golden.float(), atol=0.008, rtol=0.02):
            raise AssertionError(f"Mamba-to-MoE physical handoff failed: max_abs={edge_max_abs_error}")
    summary = {
        "stage": stage,
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "max_abs_error": results.get("max_error"),
        "allclose_match_rate": rate,
        "edge_max_abs_error": edge_max_abs_error,
        "output_vram_addr": prog.get_vram_addr(current.name),
        "hbm_bytes": hbm_size,
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("mamba", "moe", "mamba_moe", "all"), default="all")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("transactional_emulator/testbench/build/nemotron3_mamba_connected"),
    )
    args = parser.parse_args()
    stages = ("mamba", "moe", "mamba_moe") if args.stage == "all" else (args.stage,)
    summaries = [build_and_run(stage, args.build_dir.expanduser().resolve() / stage) for stage in stages]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
