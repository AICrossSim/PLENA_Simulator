"""Rust numerical validation for a physically connected KDA decoder path.

The workload is intentionally compact (one head, one token, MLEN=64), but it
uses the production Compiler KDA scheduler/lowerer and the production Rust
X_STATE implementation.  The descriptor, recurrent state, convolution
parameters, Matrix weights, and optional MoE weights all share one HBM image.

Validated paths:

* ``kda``: hidden -> eight Matrix projections -> X_STATE -> gate/norm -> output
* ``kda_moe``: the KDA output address is consumed directly by routed/shared MoE
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from compiler.aten.kda.scheduler import KdaScheduleConfig, KimiK3KdaScheduler
from compiler.aten.kimi3.blocks import (
    AttnResConstants,
    KimiLatentMoeConstants,
    KimiLatentMoeShape,
    KimiLatentMoeWeights,
    emit_kimi_attn_res,
    emit_kimi_latent_moe_residual_block,
)
from compiler.aten.mamba.scheduler import SchedulePhase
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_routed_moe import KimiSituFPConstants
from compiler.aten.state.isa_lowering import (
    KdaLayerMemoryMap,
    lower_kda_trace_to_existing_isa,
)
from transactional_emulator.testbench.aten.configurable import setup_hw
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
    BETA,
    BLEN,
    EPS,
    LINEAR_BETA,
    MLEN,
    TensorSet,
    _bf16,
    _bf16_layout,
    _attn_res_golden,
    _exact,
    _linear,
    _moe_golden,
    _register_expert_table,
    _register_weight,
    _rms,
    _set_matrix_kv_plain_bf16,
    _sigmoid,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


KDA_HEADS = 1
KDA_DIM = 128
KDA_KERNEL = 4
KDA_ARENA_BASE = 0x1_0000
KDA_WEIGHT_BASE = 0x100_0000
KDA_WEIGHT_SLOT = 0x8000
KDA_WEIGHT_OFFSETS = tuple(index * KDA_WEIGHT_SLOT for index in range(9))


def _allocate_kda_moe_constants(
    prog: PlenaCompiler,
) -> tuple[KimiLatentMoeConstants, list[float]]:
    """Reserve KDA's fixed FPRAM ABI before allocating MoE constants."""

    zero = prog.fp_var("zero", 1)  # f0
    prog.fp_var("attention_scale", 1)  # f1
    prog.fp_var("attention_negative_infinity", 1)  # f2
    prog.fp_var("attention_online_softmax_workspace", 253)  # f3..f255
    kda_eps = prog.fp_var("kda_eps_backup", 1)  # f256
    kda_value_reciprocal = prog.fp_var("kda_value_reciprocal_backup", 1)  # f257
    one = prog.fp_var("one", BLEN)  # f258+, state backup and SiTU constant
    neg_one = prog.fp_var("neg_one", BLEN)
    beta = prog.fp_var("beta", BLEN)
    neg_two_beta = prog.fp_var("neg_two_beta", BLEN)
    linear_beta = prog.fp_var("linear_beta", BLEN)
    neg_two_linear_beta = prog.fp_var("neg_two_linear_beta", BLEN)
    zero_row = prog.fp_var("zero_row", MLEN)
    norm_eps = prog.fp_var("moe_norm_eps", 1)
    norm_reciprocal = prog.fp_var("moe_norm_reciprocal", 1)
    routed_eps = prog.fp_var("routed_norm_eps", 1)
    routed_reciprocal = prog.fp_var("routed_norm_reciprocal", 1)

    variables = (
        zero,
        kda_eps,
        kda_value_reciprocal,
        one,
        neg_one,
        beta,
        neg_two_beta,
        linear_beta,
        neg_two_linear_beta,
        zero_row,
        norm_eps,
        norm_reciprocal,
        routed_eps,
        routed_reciprocal,
    )
    preload = [0.0] * max(var.address + var.size for var in variables)

    def fill(var, value: float) -> None:
        for index in range(var.size):
            preload[var.address + index] = value

    fill(kda_eps, EPS)
    fill(kda_value_reciprocal, 1.0 / KDA_DIM)
    fill(one, 1.0)
    fill(neg_one, -1.0)
    fill(beta, BETA)
    fill(neg_two_beta, -2.0 / BETA)
    fill(linear_beta, LINEAR_BETA)
    fill(neg_two_linear_beta, -2.0 / LINEAR_BETA)
    fill(norm_eps, EPS)
    fill(norm_reciprocal, 1.0 / MLEN)
    fill(routed_eps, EPS)
    fill(routed_reciprocal, 1.0 / MLEN)

    constants = KimiLatentMoeConstants(
        situ=KimiSituFPConstants(
            zero=zero,
            one=one,
            neg_one=neg_one,
            beta=beta,
            neg_two_over_beta=neg_two_beta,
            linear_beta=linear_beta,
            neg_two_over_linear_beta=neg_two_linear_beta,
        ),
        zero_row=zero_row,
        norm_eps=norm_eps.address,
        norm_reciprocal_hidden=norm_reciprocal.address,
        routed_norm_eps=routed_eps.address,
        routed_norm_reciprocal_hidden=routed_reciprocal.address,
    )
    return constants, preload


def _register_fixed_weight(
    prog: PlenaCompiler,
    tensors: TensorSet,
    name: str,
    value: torch.Tensor,
    address: int,
    *,
    bf16: bool = False,
):
    tensors.add(name, value, bf16=bf16)
    return prog.input(
        name,
        shape=tuple(value.shape),
        physical_shape=tuple(value.shape),
        hbm_addr=address,
        real_data_ratio=2.0 if bf16 else None,
    )


def _register_kda_weights(
    prog: PlenaCompiler,
    tensors: TensorSet,
    memory: KdaLayerMemoryMap,
) -> dict[str, torch.Tensor]:
    eye = torch.eye(MLEN, dtype=torch.bfloat16)
    repeated_eye = torch.cat((eye, eye), dim=1)
    reduced_eye = torch.cat((eye, eye), dim=0)
    weights = {
        # q == k guarantees a positive recurrent dot product. Identity-like
        # weights also produce a large enough signal that an all-zero output
        # cannot pass under one-BF16-ULP tolerances.
        "W_kda_q": repeated_eye,
        "W_kda_k": repeated_eye,
        "W_kda_v": repeated_eye,
        "W_kda_gate": torch.zeros(MLEN, KDA_DIM, dtype=torch.bfloat16),
        "W_kda_out": reduced_eye,
        "W_kda_decay_a": torch.zeros(MLEN, KDA_DIM, dtype=torch.bfloat16),
        "W_kda_decay_b": torch.zeros(KDA_DIM, KDA_DIM, dtype=torch.bfloat16),
        "W_kda_beta": torch.zeros(MLEN, MLEN, dtype=torch.bfloat16),
        "W_kda_norm": torch.ones(1, KDA_DIM, dtype=torch.bfloat16),
    }
    addresses = {
        "W_kda_q": memory.q_weight_hbm_addr,
        "W_kda_k": memory.k_weight_hbm_addr,
        "W_kda_v": memory.v_weight_hbm_addr,
        "W_kda_gate": memory.gate_weight_hbm_addr,
        "W_kda_out": memory.output_weight_hbm_addr,
        "W_kda_decay_a": memory.decay_a_weight_hbm_addr,
        "W_kda_decay_b": memory.decay_b_weight_hbm_addr,
        "W_kda_beta": memory.beta_weight_hbm_addr,
        "W_kda_norm": memory.norm_weight_hbm_addr,
    }
    for name, value in weights.items():
        _register_fixed_weight(
            prog,
            tensors,
            name,
            value,
            addresses[name],
            bf16=name == "W_kda_norm",
        )
    return weights


def _register_moe(
    prog: PlenaCompiler,
    tensors: TensorSet,
) -> tuple[KimiLatentMoeShape, KimiLatentMoeWeights]:
    router = torch.zeros(MLEN, 4, dtype=torch.bfloat16)
    for expert in range(4):
        router[:, expert] = _exact((MLEN,), expert + 1, expert, scale=1 / 32)
    gate_values = [_exact((MLEN, MLEN), expert + 1, expert, 1 / 32) for expert in range(4)]
    up_values = [_exact((MLEN, MLEN), expert + 2, expert + 1, 1 / 32) for expert in range(4)]
    down_values = [_exact((MLEN, MLEN), expert + 3, expert + 2, 1 / 32) for expert in range(4)]
    weights = KimiLatentMoeWeights(
        router=_register_weight(prog, tensors, "W_moe_router", router, bf16=True),
        routed_down=_register_weight(prog, tensors, "W_moe_latent_down", _exact((MLEN, MLEN), 2, 2, 1 / 32)),
        routed_up=_register_weight(prog, tensors, "W_moe_latent_up", _exact((MLEN, MLEN), 3, 3, 1 / 32)),
        routed_gate=_register_expert_table(prog, tensors, prefix="W_expert_gate", values=gate_values),
        routed_up_expert=_register_expert_table(prog, tensors, prefix="W_expert_up", values=up_values),
        routed_down_expert=_register_expert_table(prog, tensors, prefix="W_expert_down", values=down_values),
        shared=(
            _register_weight(prog, tensors, "W_shared_gate", _exact((MLEN, MLEN), 4, 1, 1 / 32)),
            _register_weight(prog, tensors, "W_shared_up", _exact((MLEN, MLEN), 2, 3, 1 / 32)),
            _register_weight(prog, tensors, "W_shared_down", _exact((MLEN, MLEN), 3, 4, 1 / 32)),
        ),
    )
    shape = KimiLatentMoeShape(
        hidden=MLEN,
        routed_hidden=MLEN,
        intermediate=MLEN,
        shared_intermediate=MLEN,
        num_experts=4,
        top_k=2,
    )
    return shape, weights


def _kda_golden(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    q_raw = _linear(hidden, weights["W_kda_q"])
    k_raw = _linear(hidden, weights["W_kda_k"])
    v_raw = _linear(hidden, weights["W_kda_v"])
    gate = _linear(hidden, weights["W_kda_gate"])
    decay_low_rank = _linear(hidden, weights["W_kda_decay_a"])
    _decay = _linear(decay_low_rank, weights["W_kda_decay_b"])
    beta_projection = _linear(hidden, weights["W_kda_beta"])

    # The convolution state starts at zero and every channel uses [0,0,0,1].
    # Rust computes SiLU in FP32, then q/k normalization and recurrent update in
    # FP32; the state-engine output is quantized once when written to BF16 VRAM.
    q = torch.nn.functional.silu(q_raw.float())
    k = torch.nn.functional.silu(k_raw.float())
    value = torch.nn.functional.silu(v_raw.float())
    q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1.0e-6)
    k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1.0e-6)
    beta = torch.sigmoid(beta_projection[:, :1].float())
    state = beta.unsqueeze(-1) * value.unsqueeze(-1) * k.unsqueeze(-2)
    output = (state * q.unsqueeze(-2)).sum(dim=-1) * (KDA_DIM**-0.5)
    output = _bf16(output)

    normalized = _rms(output)
    normalized = _bf16(normalized.float() * weights["W_kda_norm"].float())
    gated = _bf16(normalized.float() * _sigmoid(gate).float())
    return _linear(gated, weights["W_kda_out"])


def _bf16_bytes(value: torch.Tensor) -> bytes:
    return value.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().astype("<u2", copy=False).tobytes()


def _patch_hbm(
    path: Path,
    *,
    descriptor_image: bytes,
    descriptor_base: int,
    layout_descriptor_image: bytes,
    layout_descriptor_base: int,
    parameter_writes: dict[int, torch.Tensor],
    minimum_size: int,
) -> None:
    with path.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        current_size = stream.tell()
        if current_size < minimum_size:
            stream.write(b"\x00" * (minimum_size - current_size))
        stream.seek(descriptor_base)
        stream.write(descriptor_image)
        stream.seek(layout_descriptor_base)
        stream.write(layout_descriptor_image)
        for address, value in sorted(parameter_writes.items()):
            stream.seek(address)
            stream.write(_bf16_bytes(value))


def build_and_run(stage: str, build_dir: Path, *, seed: int = 29) -> dict:
    if stage not in {"kda", "kda_moe", "kda_attnres_moe"}:
        raise ValueError(f"unsupported stage {stage!r}")
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(
        argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None),
        build_dir,
    )
    _set_matrix_kv_plain_bf16()
    torch.manual_seed(seed)

    config = KdaScheduleConfig(
        phase=SchedulePhase.DECODE,
        decode_tokens=1,
        matrix_input_features=MLEN,
        kda_layer_ids=(0,),
        kda_num_heads=KDA_HEADS,
        hbm_arena_base=KDA_ARENA_BASE,
        projection_weight_hbm_base=KDA_WEIGHT_BASE,
        projection_weight_layer_stride=len(KDA_WEIGHT_OFFSETS) * KDA_WEIGHT_SLOT,
        projection_weight_offsets=KDA_WEIGHT_OFFSETS,
    )
    scheduler = KimiK3KdaScheduler(config)
    trace = scheduler.build()
    kda_program = lower_kda_trace_to_existing_isa(trace, descriptor_base=0)
    memories = {event.memory for event in kda_program.events if isinstance(event.memory, KdaLayerMemoryMap)}
    if len(memories) != 1:
        raise AssertionError(f"expected one KDA memory map, got {len(memories)}")
    memory = memories.pop()

    prog = PlenaCompiler(mlen=MLEN, blen=BLEN, real_data_ratio=hw.real_data_ratio)
    tensors = TensorSet(values={}, bf16_names=set())
    moe_constants, fp_preload = _allocate_kda_moe_constants(prog)

    workspace_end = memory.normalization_scratch_vram_addr + MLEN
    prog.vram_allocator._vmm.mark_used(0, workspace_end, name="KDA_PHYSICAL_WORKSPACE")
    vram_preload = torch.zeros(workspace_end + 8 * MLEN * MLEN, dtype=torch.bfloat16)
    hidden_value = (torch.randn(1, MLEN) * 0.2).to(torch.bfloat16)
    hidden = prestage_bf16_vram_matrix(
        prog=prog,
        name="HIDDEN",
        tensor=hidden_value,
        vram_addr=memory.hidden_vram_addr,
        physical_shape=(BLEN, MLEN),
        vram_preload=vram_preload,
    )
    correction_value = torch.tensor(
        [[0.0, 0.125, 0.25, -0.125] + [0.0] * (MLEN - 4)],
        dtype=torch.bfloat16,
    )
    vram_tile = MLEN * MLEN
    correction_addr = ((workspace_end + vram_tile - 1) // vram_tile) * vram_tile
    correction = prestage_bf16_vram_matrix(
        prog=prog,
        name="MOE_CORRECTION",
        tensor=correction_value,
        vram_addr=correction_addr,
        physical_shape=(1, MLEN),
        vram_preload=vram_preload,
    )
    tensors.add("MOE_CORRECTION", correction_value)
    block_value = (torch.randn(1, MLEN) * 0.15).to(torch.bfloat16)
    block_addr = correction_addr + vram_tile
    block_residual = prestage_bf16_vram_matrix(
        prog=prog,
        name="ATTNRES_BLOCK",
        tensor=block_value,
        vram_addr=block_addr,
        physical_shape=(BLEN, MLEN),
        vram_preload=vram_preload,
    )
    score_weight_value = _exact((1, MLEN), 5, 2, scale=1 / 32)
    score_addr = block_addr + vram_tile
    score_weight = prestage_bf16_vram_matrix(
        prog=prog,
        name="ATTNRES_SCORE_WEIGHT",
        tensor=score_weight_value,
        vram_addr=score_addr,
        physical_shape=(BLEN, MLEN),
        vram_preload=vram_preload,
    )

    kda_weights = _register_kda_weights(prog, tensors, memory)
    projection_end = KDA_WEIGHT_BASE + len(KDA_WEIGHT_OFFSETS) * KDA_WEIGHT_SLOT
    prog._next_hbm_addr = max(prog._next_hbm_addr, projection_end)
    current = hidden
    golden = _bf16(hidden_value)
    prefix = None
    if stage == "kda_attnres_moe":
        prefix = prog.vram_copy(hidden, name="connected_prefix_before_kda", num_rows=1)
        kda_input = emit_kimi_attn_res(
            prog,
            (block_residual,),
            prefix,
            score_weight=score_weight,
            constants=AttnResConstants(
                eps=moe_constants.norm_eps,
                reciprocal_hidden=moe_constants.norm_reciprocal_hidden,
            ),
            rows=1,
            name="connected_attnres_before_kda",
        )
        kda_input_golden = _attn_res_golden((block_value,), golden, score_weight_value)
        prog.vram_copy_region(hidden, kda_input, num_rows=1, num_cols=MLEN)
        prog.free_tensor(kda_input)
        golden = kda_input_golden

    prog.emit(kda_program.assembly)
    golden = _kda_golden(golden, kda_weights)
    current = hidden

    if stage in {"kda_moe", "kda_attnres_moe"}:
        moe_shape, moe_weights = _register_moe(prog, tensors)
        moe_input = current
        add_residual = True
        prefix_after_mixer = None
        if stage == "kda_attnres_moe":
            assert prefix is not None
            prefix_after_mixer = prog.vram_copy(prefix, name="connected_prefix_after_kda", num_rows=1)
            prog.vram_add(prefix_after_mixer, current, num_rows=1)
            prefix_after_mixer_golden = _bf16(hidden_value.float() + golden.float())
            moe_input = emit_kimi_attn_res(
                prog,
                (block_residual,),
                prefix_after_mixer,
                score_weight=score_weight,
                constants=AttnResConstants(
                    eps=moe_constants.norm_eps,
                    reciprocal_hidden=moe_constants.norm_reciprocal_hidden,
                ),
                rows=1,
                name="connected_attnres_before_moe",
            )
            moe_input_golden = _attn_res_golden((block_value,), prefix_after_mixer_golden, score_weight_value)
            golden = moe_input_golden
            add_residual = False
        moe_output = emit_kimi_latent_moe_residual_block(
            prog,
            moe_input,
            shape=moe_shape,
            weights=moe_weights,
            correction_bias=correction,
            constants=moe_constants,
            rows=1,
            name="connected_kda_moe",
            add_residual=add_residual,
        )
        moe_golden = _moe_golden(golden, tensors, add_residual=add_residual)
        if stage == "kda_attnres_moe":
            assert prefix_after_mixer is not None
            current = prog.vram_copy(prefix_after_mixer, name="connected_prefix_after_moe", num_rows=1)
            prog.vram_add(current, moe_output, num_rows=1)
            golden = _bf16(prefix_after_mixer_golden.float() + moe_golden.float())
        else:
            current = moe_output
            golden = moe_golden

    asm = prog.compile()
    input_tensors = {name: value for name, value in tensors.values.items() if name != "MOE_CORRECTION"}
    layouts = infer_hbm_tensor_layouts(input_tensors)
    for name in tensors.bf16_names:
        layouts[name] = _bf16_layout(input_tensors[name])
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    data_order = sorted(input_tensors, key=hbm_addrs.__getitem__)
    create_sim_env(
        input_tensors,
        asm,
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
        asm=f"kimi3_connected_{stage}",
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=layouts,
        hbm_addrs=hbm_addrs,
    )

    layout = scheduler.hbm_layout()
    hbm_size = (
        math.ceil(
            max(
                prog._next_hbm_addr,
                layout.realized_arena_bytes(len(trace.events)),
                len(kda_program.descriptor_image),
                kda_program.layout_descriptor_base + len(kda_program.layout_descriptor_image),
            )
            / 64
        )
        * 64
    )
    conv_weight = torch.zeros(KDA_HEADS * KDA_DIM, KDA_KERNEL)
    conv_weight[:, -1] = 1.0
    zeros = torch.zeros(KDA_HEADS * KDA_DIM)
    parameter_writes = {
        layout.address("q_conv_weight", 0): conv_weight,
        layout.address("k_conv_weight", 0): conv_weight,
        layout.address("v_conv_weight", 0): conv_weight,
        layout.address("q_conv_bias", 0): zeros,
        layout.address("k_conv_bias", 0): zeros,
        layout.address("v_conv_bias", 0): zeros,
        layout.address("a_log", 0): torch.zeros(KDA_HEADS),
        layout.address("dt_bias", 0): zeros,
    }
    _patch_hbm(
        build_dir / "hbm_for_behave_sim.bin",
        descriptor_image=kda_program.descriptor_image,
        descriptor_base=kda_program.descriptor_base,
        layout_descriptor_image=kda_program.layout_descriptor_image,
        layout_descriptor_base=kda_program.layout_descriptor_base,
        parameter_writes=parameter_writes,
        minimum_size=hbm_size,
    )

    params = _comparison_params_for(
        current,
        rows=1,
        hidden=MLEN,
        mlen=MLEN,
        golden=golden,
    )
    params.update({"atol": 0.004, "rtol": 0.01, "min_allclose_match_rate": 100.0})
    if float(golden.abs().max()) < 0.05:
        raise AssertionError(f"{stage} golden signal is too small to reject an all-zero implementation")
    (build_dir / "comparison_params.json").write_text(json.dumps(params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(asm)
    (build_dir / "hbm_size.txt").write_text(f"{hbm_size}\n")

    metrics = run_emulator(
        build_dir,
        stage_profile=True,
        state_profile_out=build_dir / "state_profile.json",
        dump_cwd=build_dir,
    )
    results, _ = compare_emulator_output(build_dir, verbose=False)
    actual_rate = float(results.get("allclose_match_rate", 0.0))
    if actual_rate < 100.0:
        raise AssertionError(
            f"{stage} Rust numerical comparison failed: "
            f"max_abs={results.get('max_error')}, allclose_rate={actual_rate}%"
        )
    summary = {
        "stage": stage,
        "asm_lines": len(asm.splitlines()),
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "max_abs_error": results.get("max_error"),
        "allclose_match_rate": results.get("allclose_match_rate"),
        "output_vram_addr": prog.get_vram_addr(current.name),
        "kda_hidden_vram_addr": memory.hidden_vram_addr,
        "hbm_bytes": hbm_size,
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("kda", "kda_moe", "kda_attnres_moe", "all"),
        default="all",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("transactional_emulator/testbench/build/kimi3_kda_connected"),
    )
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    stages = ("kda", "kda_moe", "kda_attnres_moe") if args.stage == "all" else (args.stage,)
    summaries = [
        build_and_run(stage, args.build_dir.expanduser().resolve() / stage, seed=args.seed) for stage in stages
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
