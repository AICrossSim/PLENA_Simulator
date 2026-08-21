"""Numerical Rust-emulator validation for connected Kimi K3 decoder blocks.

This is deliberately a compact MLEN=64 workload.  It validates data ownership
and instruction semantics, not Kimi's model quality or full-size performance:

* ``mla``: hidden -> MLA projections/core/gate/out -> residual
* ``moe``: hidden -> corrected sigmoid router -> routed/shared SiTU -> residual
* ``chain``: the actual MLA output is consumed by MoE in one machine program
* ``attnres``: saved block + current prefix -> depth softmax -> weighted hidden
* ``attnres_chain``: official MLA/prefix/AttnRes/MoE/prefix ownership

Run this against the matching Compiler feature worktree with
``PLENA_COMPILER_ROOT=/path/to/PLENA_Compiler``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import tomlkit
import torch

from compiler.aten.kimi3.blocks import (
    AttnResConstants,
    KimiLatentMoeConstants,
    KimiLatentMoeShape,
    KimiLatentMoeWeights,
    MlaBlockShape,
    MlaBlockWeights,
    MlaNormConstants,
    emit_kimi_attn_res,
    emit_kimi_latent_moe_residual_block,
    emit_mla_residual_block,
)
from compiler.aten.plena import ExpertWeightTable, PlenaCompiler
from compiler.aten.plena.program_routed_moe import KimiSituFPConstants
from transactional_emulator.testbench.aten.configurable import setup_hw
from transactional_emulator.testbench.aten.golden import (
    _active_precision_settings,
    _rms_norm_vector_ref,
    quantize_to_vector_fp,
)
from transactional_emulator.testbench.emulator_runner import (
    compare_emulator_output,
    run_emulator,
)
from transactional_emulator.testbench.gpt_oss_testkit import (
    _comparison_params_for,
    _exact_mxfp8_tensor,
    _linear_projection_golden,
)
from transactional_emulator.testbench.layout_utils import (
    infer_hbm_tensor_layouts,
    prestage_bf16_vram_matrix,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4
EPS = 1e-5
BETA = 4.0
LINEAR_BETA = 25.0


@dataclass
class TensorSet:
    values: dict[str, torch.Tensor]
    bf16_names: set[str]
    references: dict[str, torch.Tensor] = field(default_factory=dict)

    def add(self, name: str, value: torch.Tensor, *, bf16: bool = False) -> torch.Tensor:
        value = value.contiguous()
        self.values[name] = value
        if bf16:
            self.bf16_names.add(name)
        return value


def _set_matrix_kv_plain_bf16() -> None:
    settings = Path(os.environ["PLENA_SETTINGS_TOML"])
    with settings.open() as stream:
        config = tomlkit.load(stream)
    for mode in ("TRANSACTIONAL", "ANALYTIC"):
        precision = config[mode]["PRECISION"]
        for key in ("HBM_M_KV_TYPE", "HBM_V_KV_TYPE"):
            precision[key] = tomlkit.table()
            precision[key]["format"] = "Plain"
            precision[key]["DATA_TYPE"] = tomlkit.table()
            precision[key]["DATA_TYPE"].update(
                {"type": "Fp", "sign": True, "exponent": 8, "mantissa": 7}
            )
    with settings.open("w") as stream:
        tomlkit.dump(config, stream)


def _bf16_layout(tensor: torch.Tensor, precision: str = "HBM_M_KV_TYPE") -> dict:
    rows, cols = tensor.shape
    return {
        "source_shape": [rows, cols],
        "storage_shape": [rows, cols],
        "source_rows": rows,
        "storage_rows": rows,
        "source_row_elements": cols,
        "storage_row_elements": cols,
        "precision": precision,
    }


def _exact(shape: tuple[int, ...], stride: int, offset: int = 0, scale: float = 1 / 64) -> torch.Tensor:
    return (_exact_mxfp8_tensor(shape, stride=stride, offset=offset) * scale).to(torch.bfloat16)


def _bf16(value: torch.Tensor) -> torch.Tensor:
    return quantize_to_vector_fp(value.to(torch.bfloat16).float(), _active_precision_settings())


def _rms(value: torch.Tensor) -> torch.Tensor:
    return _rms_norm_vector_ref(
        _bf16(value),
        EPS,
        _active_precision_settings(),
        vlen=MLEN,
    )


def _linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _linear_projection_golden(value, weight, mlen=MLEN, hbm_input=False)


def _sigmoid(value: torch.Tensor) -> torch.Tensor:
    out = _bf16(value)
    out = _bf16(out.float() * -1.0)
    out = _bf16(torch.exp(torch.clamp(out.float(), -88.0, 88.0)))
    out = _bf16(out.float() + 1.0)
    return _bf16(torch.reciprocal(out.float()))


def _situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    gate_exp = _bf16(gate.float() * (-2.0 / BETA))
    gate_exp = _bf16(torch.exp(torch.clamp(gate_exp.float(), -88.0, 88.0)))
    gate_denom = _bf16(gate_exp.float() + 1.0)
    gate_denom = _bf16(torch.reciprocal(gate_denom.float()))
    gate_num = _bf16(gate_exp.float() * -1.0)
    gate_num = _bf16(gate_num.float() + 1.0)
    gate_tanh = _bf16(gate_num.float() * gate_denom.float())
    gate_term = _bf16(gate_tanh.float() * _sigmoid(gate).float())
    gate_term = _bf16(gate_term.float() * BETA)

    up_exp = _bf16(up.float() * (-2.0 / LINEAR_BETA))
    up_exp = _bf16(torch.exp(torch.clamp(up_exp.float(), -88.0, 88.0)))
    up_denom = _bf16(up_exp.float() + 1.0)
    up_denom = _bf16(torch.reciprocal(up_denom.float()))
    up_num = _bf16(up_exp.float() * -1.0)
    up_num = _bf16(up_num.float() + 1.0)
    up_term = _bf16(up_num.float() * up_denom.float())
    up_term = _bf16(up_term.float() * LINEAR_BETA)
    return _bf16(gate_term.float() * up_term.float())


def _mla_golden(
    hidden: torch.Tensor,
    tensors: TensorSet,
    *,
    add_residual: bool = True,
) -> torch.Tensor:
    mixer = _rms(hidden)
    # Run the Q path in the golden even though one-token attention reduces to V;
    # this keeps shape/rounding mistakes visible during test construction.
    q_latent = _rms(_linear(mixer, tensors.values["W_mla_q_a"]))
    _q = _linear(q_latent, tensors.values["W_mla_q_b"])
    compressed = _linear(mixer, tensors.values["W_mla_kv_a"])
    kv_latent = _rms(compressed[:, :MLEN])
    kv_heads = _linear(kv_latent, tensors.values["W_mla_kv_b"])
    value = kv_heads[:, MLEN : 2 * MLEN]
    # This connected test configures HBM_M/V_KV_TYPE as plain BF16. For one
    # visible key softmax is exactly 1, so the attention value is the BF16 V
    # payload without an additional MXFP8 round trip.
    attention = value.to(torch.bfloat16)
    gate = _sigmoid(_linear(mixer, tensors.values["W_mla_gate"]))
    attention = _bf16(attention.float() * gate.float())
    output = _linear(attention, tensors.values["W_mla_out"])
    return _bf16(output.float() + _bf16(hidden).float()) if add_residual else output


def _moe_golden(
    hidden: torch.Tensor,
    tensors: TensorSet,
    *,
    add_residual: bool = True,
) -> torch.Tensor:
    mixer = _rms(hidden)
    logits = torch.matmul(mixer.float(), tensors.values["W_moe_router"].float()).to(torch.bfloat16)
    bias = tensors.values["MOE_CORRECTION"]
    ranking = logits.float() + bias[:, : logits.shape[1]].float()
    selected = sorted(range(logits.shape[1]), key=lambda idx: (-float(ranking[0, idx]), idx))[:2]
    raw = torch.tensor([float(logits[0, idx]) for idx in selected], dtype=torch.float32)
    route = torch.sigmoid(raw)
    route = (route / route.sum()).to(torch.bfloat16)

    routed_input = _linear(mixer, tensors.values["W_moe_latent_down"])
    accumulator = torch.zeros_like(routed_input)
    for slot, expert in enumerate(selected):
        gate = _linear(
            routed_input, tensors.references.get(f"W_expert_gate_{expert}", tensors.values.get(f"W_expert_gate_{expert}"))
        )
        up = _linear(
            routed_input, tensors.references.get(f"W_expert_up_{expert}", tensors.values.get(f"W_expert_up_{expert}"))
        )
        expert_out = _linear(
            _situ(gate, up),
            tensors.references.get(f"W_expert_down_{expert}", tensors.values.get(f"W_expert_down_{expert}")),
        )
        weighted = _bf16(expert_out.float() * route[slot].float())
        accumulator = _bf16(accumulator.float() + weighted.float())
    routed = _linear(_rms(accumulator), tensors.values["W_moe_latent_up"])

    shared_gate = _linear(mixer, tensors.values["W_shared_gate"])
    shared_up = _linear(mixer, tensors.values["W_shared_up"])
    shared = _linear(_situ(shared_gate, shared_up), tensors.values["W_shared_down"])
    output = _bf16(routed.float() + shared.float())
    return _bf16(output.float() + _bf16(hidden).float()) if add_residual else output


def _attn_res_golden(
    block_residuals: tuple[torch.Tensor, ...],
    prefix_sum: torch.Tensor,
    score_weight: torch.Tensor,
) -> torch.Tensor:
    candidates = (*block_residuals, prefix_sum)
    scores = []
    for candidate in candidates:
        product = _bf16(_rms(candidate).float() * _bf16(score_weight).float())
        scores.append(_bf16(product.float().sum(dim=-1, keepdim=True)))
    stacked = torch.cat(scores, dim=-1)
    maximum = stacked.max(dim=-1, keepdim=True).values
    exponentials = _bf16(torch.exp(_bf16(stacked.float() - maximum.float()).float()))
    denominator = torch.zeros_like(exponentials[:, :1])
    for index in range(exponentials.shape[1]):
        denominator = _bf16(denominator.float() + exponentials[:, index : index + 1].float())
    inverse = _bf16(torch.reciprocal(denominator.float()))
    probabilities = _bf16(exponentials.float() * inverse.float())
    output = _bf16(candidates[0].float() * probabilities[:, :1].float())
    for index, candidate in enumerate(candidates[1:], start=1):
        weighted = _bf16(candidate.float() * probabilities[:, index : index + 1].float())
        output = _bf16(output.float() + weighted.float())
    return output


def _register_weight(
    prog: PlenaCompiler,
    tensors: TensorSet,
    name: str,
    value: torch.Tensor,
    *,
    bf16: bool = False,
):
    tensors.add(name, value, bf16=bf16)
    return prog.input(
        name,
        shape=tuple(value.shape),
        physical_shape=tuple(value.shape),
        real_data_ratio=2.0 if bf16 else None,
    )


def _register_expert_table(
    prog: PlenaCompiler,
    tensors: TensorSet,
    *,
    prefix: str,
    values: list[torch.Tensor],
) -> ExpertWeightTable:
    rows, cols = values[0].shape
    if rows % prog.mlen or cols % prog.mlen:
        raise ValueError(f"expert table shape must be MLEN-aligned, got {(rows, cols)}")
    block_size = prog.mlen * prog.mlen
    stride = prog.hbm_tensor_size(block_size)
    raw_group_size = stride * len(values)
    tile_group_stride = 1 << (raw_group_size - 1).bit_length()
    prog._next_hbm_addr = (
        (prog._next_hbm_addr + tile_group_stride - 1) // tile_group_stride
    ) * tile_group_stride
    row_tiles = rows // prog.mlen
    col_tiles = cols // prog.mlen
    base = prog._allocate_hbm(row_tiles * col_tiles * tile_group_stride)
    template = prog.input(
        f"{prefix}_template",
        shape=(rows, cols),
        physical_shape=(rows, cols),
        hbm_addr=base,
    )
    for expert, value in enumerate(values):
        tensors.references[f"{prefix}_{expert}"] = value.contiguous()
        for row_tile in range(row_tiles):
            for col_tile in range(col_tiles):
                group = row_tile * col_tiles + col_tile
                name = f"{prefix}_tile_{row_tile}_{col_tile}_expert_{expert}"
                tile = value[
                    row_tile * prog.mlen : (row_tile + 1) * prog.mlen,
                    col_tile * prog.mlen : (col_tile + 1) * prog.mlen,
                ].contiguous()
                tensors.add(name, tile)
                prog.input(
                    name,
                    shape=(prog.mlen, prog.mlen),
                    physical_shape=(prog.mlen, prog.mlen),
                    hbm_addr=(
                        base + group * tile_group_stride + expert * stride
                    ),
                )
    return ExpertWeightTable(
        template=template,
        base=base,
        stride=stride,
        num_experts=len(values),
        tile_group_stride=tile_group_stride,
    )


def _allocate_fp_constants(prog: PlenaCompiler) -> tuple[MlaNormConstants, KimiLatentMoeConstants, list[float]]:
    zero = prog.fp_var("zero", 1)
    attn_scale = prog.fp_var("attn_scale", 1)
    neg_inf = prog.fp_var("negative_infinity", 1)
    prog.fp_var("online_softmax_reserved", 253)
    input_eps = prog.fp_var("input_eps", 1)
    input_reci = prog.fp_var("input_reciprocal", 1)
    q_eps = prog.fp_var("q_eps", 1)
    q_reci = prog.fp_var("q_reciprocal", 1)
    kv_eps = prog.fp_var("kv_eps", 1)
    kv_reci = prog.fp_var("kv_reciprocal", 1)
    one = prog.fp_var("one", BLEN)
    neg_one = prog.fp_var("neg_one", BLEN)
    beta = prog.fp_var("beta", BLEN)
    neg_two_beta = prog.fp_var("neg_two_beta", BLEN)
    linear_beta = prog.fp_var("linear_beta", BLEN)
    neg_two_linear_beta = prog.fp_var("neg_two_linear_beta", BLEN)
    zero_row = prog.fp_var("zero_row", MLEN)
    routed_eps = prog.fp_var("routed_eps", 1)
    routed_reci = prog.fp_var("routed_reciprocal", 1)

    all_vars = [
        zero,
        attn_scale,
        neg_inf,
        input_eps,
        input_reci,
        q_eps,
        q_reci,
        kv_eps,
        kv_reci,
        one,
        neg_one,
        beta,
        neg_two_beta,
        linear_beta,
        neg_two_linear_beta,
        zero_row,
        routed_eps,
        routed_reci,
    ]
    preload = [0.0] * max(var.address + var.size for var in all_vars)

    def fill(var, value: float) -> None:
        for index in range(var.size):
            preload[var.address + index] = value

    fill(attn_scale, (128**-0.5) / 0.25)
    fill(neg_inf, float("-inf"))
    for var in (input_eps, q_eps, kv_eps, routed_eps):
        fill(var, EPS)
    for var in (input_reci, q_reci, kv_reci, routed_reci):
        fill(var, 1.0 / MLEN)
    fill(one, 1.0)
    fill(neg_one, -1.0)
    fill(beta, BETA)
    fill(neg_two_beta, -2.0 / BETA)
    fill(linear_beta, LINEAR_BETA)
    fill(neg_two_linear_beta, -2.0 / LINEAR_BETA)

    mla = MlaNormConstants(
        input_eps=input_eps.address,
        input_reciprocal_hidden=input_reci.address,
        q_eps=q_eps.address,
        q_reciprocal_hidden=q_reci.address,
        kv_eps=kv_eps.address,
        kv_reciprocal_hidden=kv_reci.address,
        gate_one=one,
        gate_neg_one=neg_one,
    )
    moe = KimiLatentMoeConstants(
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
        norm_eps=input_eps.address,
        norm_reciprocal_hidden=input_reci.address,
        routed_norm_eps=routed_eps.address,
        routed_norm_reciprocal_hidden=routed_reci.address,
    )
    return mla, moe, preload


def build_and_run(stage: str, build_dir: Path, *, seed: int = 17) -> dict:
    build_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None)
    hw = setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()
    torch.manual_seed(seed)

    prog = PlenaCompiler(mlen=MLEN, blen=BLEN, real_data_ratio=hw.real_data_ratio)
    tensors = TensorSet(values={}, bf16_names=set())
    physical = (MLEN, MLEN)
    vram_preload = torch.zeros(6 * MLEN * MLEN, dtype=torch.bfloat16)
    hidden_value = (torch.randn(1, MLEN) * 0.2).to(torch.bfloat16)
    hidden = prestage_bf16_vram_matrix(
        prog=prog,
        name="HIDDEN",
        tensor=hidden_value,
        vram_addr=0,
        physical_shape=physical,
        vram_preload=vram_preload,
    )
    cos = prestage_bf16_vram_matrix(
        prog=prog,
        name="ROPE_COS",
        tensor=torch.ones(1, MLEN, dtype=torch.bfloat16),
        vram_addr=MLEN * MLEN,
        physical_shape=physical,
        vram_preload=vram_preload,
    )
    sin = prestage_bf16_vram_matrix(
        prog=prog,
        name="ROPE_SIN",
        tensor=torch.zeros(1, MLEN, dtype=torch.bfloat16),
        vram_addr=2 * MLEN * MLEN,
        physical_shape=physical,
        vram_preload=vram_preload,
    )
    correction_value = torch.tensor([[0.0, 0.125, 0.25, -0.125] + [0.0] * 60], dtype=torch.bfloat16)
    correction = prestage_bf16_vram_matrix(
        prog=prog,
        name="MOE_CORRECTION",
        tensor=correction_value,
        vram_addr=3 * MLEN * MLEN,
        physical_shape=(1, MLEN),
        vram_preload=vram_preload,
    )
    tensors.add("MOE_CORRECTION", correction_value)
    block_value = (torch.randn(1, MLEN) * 0.15).to(torch.bfloat16)
    block_residual = prestage_bf16_vram_matrix(
        prog=prog,
        name="ATTNRES_BLOCK",
        tensor=block_value,
        vram_addr=4 * MLEN * MLEN,
        physical_shape=physical,
        vram_preload=vram_preload,
    )
    score_weight_value = _exact((1, MLEN), 5, 2, scale=1 / 32)
    score_weight = prestage_bf16_vram_matrix(
        prog=prog,
        name="ATTNRES_SCORE_WEIGHT",
        tensor=score_weight_value,
        vram_addr=5 * MLEN * MLEN,
        physical_shape=physical,
        vram_preload=vram_preload,
    )
    mla_constants, moe_constants, fp_preload = _allocate_fp_constants(prog)

    eye = torch.eye(MLEN, dtype=torch.bfloat16)
    mla_weights = MlaBlockWeights(
        q_a=_register_weight(prog, tensors, "W_mla_q_a", _exact((MLEN, MLEN), 1)),
        q_b=_register_weight(prog, tensors, "W_mla_q_b", _exact((MLEN, 2 * MLEN), 2, 1)),
        kv_a=_register_weight(prog, tensors, "W_mla_kv_a", _exact((MLEN, 2 * MLEN), 3, 2)),
        kv_b=_register_weight(prog, tensors, "W_mla_kv_b", _exact((MLEN, 2 * MLEN), 4, 3)),
        out=_register_weight(prog, tensors, "W_mla_out", _exact((MLEN, MLEN), 2, 4)),
        q_rope_rotate=_register_weight(prog, tensors, "W_q_rope_rotate", eye, bf16=True),
        k_rope_rotate=_register_weight(prog, tensors, "W_k_rope_rotate", eye, bf16=True),
        gate=_register_weight(prog, tensors, "W_mla_gate", _exact((MLEN, MLEN), 3, 1)),
    )
    mla_shape = MlaBlockShape(
        hidden=MLEN,
        q_lora=MLEN,
        kv_lora=MLEN,
        qk_nope=MLEN,
        qk_rope=MLEN,
        v_head=MLEN,
        heads=1,
    )

    router = torch.zeros(MLEN, 4, dtype=torch.bfloat16)
    router[:, 0] = _exact((MLEN,), 1, scale=1 / 32)
    router[:, 1] = _exact((MLEN,), 2, 1, scale=1 / 32)
    router[:, 2] = _exact((MLEN,), 3, 2, scale=1 / 32)
    router[:, 3] = _exact((MLEN,), 4, 3, scale=1 / 32)
    routed_hidden = 5 * MLEN if stage == "moe_ksplit" else MLEN
    gate_values = [
        _exact((routed_hidden, MLEN), expert + 1, expert, 1 / 32)
        for expert in range(4)
    ]
    up_values = [
        _exact((routed_hidden, MLEN), expert + 2, expert + 1, 1 / 32)
        for expert in range(4)
    ]
    down_values = [
        _exact((MLEN, routed_hidden), expert + 3, expert + 2, 1 / 32)
        for expert in range(4)
    ]
    moe_weights = KimiLatentMoeWeights(
        router=_register_weight(prog, tensors, "W_moe_router", router, bf16=True),
        routed_down=_register_weight(
            prog,
            tensors,
            "W_moe_latent_down",
            _exact((MLEN, routed_hidden), 2, 2, 1 / 32),
        ),
        routed_up=_register_weight(
            prog,
            tensors,
            "W_moe_latent_up",
            _exact((routed_hidden, MLEN), 3, 3, 1 / 32),
        ),
        routed_gate=_register_expert_table(
            prog, tensors, prefix="W_expert_gate", values=gate_values
        ),
        routed_up_expert=_register_expert_table(
            prog, tensors, prefix="W_expert_up", values=up_values
        ),
        routed_down_expert=_register_expert_table(
            prog, tensors, prefix="W_expert_down", values=down_values
        ),
        shared=(
            _register_weight(prog, tensors, "W_shared_gate", _exact((MLEN, MLEN), 4, 1, 1 / 32)),
            _register_weight(prog, tensors, "W_shared_up", _exact((MLEN, MLEN), 2, 3, 1 / 32)),
            _register_weight(prog, tensors, "W_shared_down", _exact((MLEN, MLEN), 3, 4, 1 / 32)),
        ),
    )
    moe_shape = KimiLatentMoeShape(
        hidden=MLEN,
        routed_hidden=routed_hidden,
        intermediate=MLEN,
        shared_intermediate=MLEN,
        num_experts=4,
        top_k=2,
    )

    current = hidden
    golden = _bf16(hidden_value)
    if stage in {"mla", "chain"}:
        current = emit_mla_residual_block(
            prog,
            current,
            shape=mla_shape,
            weights=mla_weights,
            cos=cos,
            sin=sin,
            norms=mla_constants,
            rows=1,
            name="connected_mla",
        )
        golden = _mla_golden(golden, tensors)
    if stage in {"moe", "moe_ksplit", "chain"}:
        current = emit_kimi_latent_moe_residual_block(
            prog,
            current,
            shape=moe_shape,
            weights=moe_weights,
            correction_bias=correction,
            constants=moe_constants,
            rows=1,
            name="connected_moe",
        )
        golden = _moe_golden(golden, tensors)
    if stage == "attnres":
        current = emit_kimi_attn_res(
            prog,
            (block_residual,),
            current,
            score_weight=score_weight,
            constants=AttnResConstants(
                eps=mla_constants.input_eps,
                reciprocal_hidden=mla_constants.input_reciprocal_hidden,
            ),
            rows=1,
            name="connected_attnres",
        )
        golden = _attn_res_golden((block_value,), golden, score_weight_value)
    if stage == "attnres_chain":
        mixer_input = emit_kimi_attn_res(
            prog,
            (block_residual,),
            current,
            score_weight=score_weight,
            constants=AttnResConstants(
                eps=mla_constants.input_eps,
                reciprocal_hidden=mla_constants.input_reciprocal_hidden,
            ),
            rows=1,
            name="connected_attnres_before_mla",
        )
        mixer_input_golden = _attn_res_golden(
            (block_value,), golden, score_weight_value
        )
        mixer_out = emit_mla_residual_block(
            prog,
            mixer_input,
            shape=mla_shape,
            weights=mla_weights,
            cos=cos,
            sin=sin,
            norms=mla_constants,
            rows=1,
            name="connected_attnres_mla",
            add_residual=False,
        )
        mixer_out_golden = _mla_golden(
            mixer_input_golden,
            tensors,
            add_residual=False,
        )
        prefix_after_mixer = prog.vram_copy(
            current,
            name="connected_prefix_after_mixer",
            num_rows=1,
        )
        prog.vram_add(prefix_after_mixer, mixer_out, num_rows=1)
        prefix_after_mixer_golden = _bf16(golden.float() + mixer_out_golden.float())
        ffn_input = emit_kimi_attn_res(
            prog,
            (block_residual,),
            prefix_after_mixer,
            score_weight=score_weight,
            constants=AttnResConstants(
                eps=mla_constants.input_eps,
                reciprocal_hidden=mla_constants.input_reciprocal_hidden,
            ),
            rows=1,
            name="connected_attnres_before_moe",
        )
        ffn_input_golden = _attn_res_golden(
            (block_value,), prefix_after_mixer_golden, score_weight_value
        )
        moe_out = emit_kimi_latent_moe_residual_block(
            prog,
            ffn_input,
            shape=moe_shape,
            weights=moe_weights,
            correction_bias=correction,
            constants=moe_constants,
            rows=1,
            name="connected_attnres_moe",
            add_residual=False,
        )
        moe_out_golden = _moe_golden(
            ffn_input_golden,
            tensors,
            add_residual=False,
        )
        current = prog.vram_copy(
            prefix_after_mixer,
            name="connected_prefix_after_moe",
            num_rows=1,
        )
        prog.vram_add(current, moe_out, num_rows=1)
        golden = _bf16(prefix_after_mixer_golden.float() + moe_out_golden.float())

    asm = prog.compile()
    # H_STORE-created K/V scratch objects need real HBM backing, even though
    # their host preload is zero and gets overwritten before first consumption.
    for name, var in prog._inputs.items():
        if name not in tensors.values and ("_k_scratch" in name or "_v_scratch" in name):
            tensors.add(name, torch.zeros(var.physical_shape, dtype=torch.bfloat16))

    input_tensors = {name: value for name, value in tensors.values.items() if name != "MOE_CORRECTION"}
    layouts = infer_hbm_tensor_layouts(input_tensors)
    for name in tensors.bf16_names:
        layouts[name] = _bf16_layout(input_tensors[name])
    hbm_addrs = {
        name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors
    }
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
    params = _comparison_params_for(current, rows=1, hidden=MLEN, mlen=MLEN, golden=golden)
    # A two-block chain can accumulate one BF16 ULP at values around 0.5
    # (2^-8 = 0.00390625). Require every element to stay within that bound;
    # do not accept the comparator's separate, historically looser test_pass.
    params.update({"atol": 0.004, "rtol": 0.01, "min_allclose_match_rate": 100.0})
    (build_dir / "comparison_params.json").write_text(json.dumps(params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(asm)
    tensor_map = {
        name: {
            "vram_addr": prog.get_vram_addr(name),
            "shape": list(var.shape),
            "physical_shape": list(var.physical_shape),
        }
        for name, var in prog._tensors.items()
        if name in prog.vram_matrices
    }
    (build_dir / "vram_tensor_map.json").write_text(json.dumps(tensor_map, indent=2) + "\n")
    (build_dir / "hbm_size.txt").write_text(f"{math.ceil(prog._next_hbm_addr / 64) * 64}\n")

    metrics = run_emulator(build_dir, stage_profile=True, dump_cwd=build_dir)
    results, _ = compare_emulator_output(build_dir, verbose=False)
    minimum_rate = float(params["min_allclose_match_rate"])
    actual_rate = float(results.get("allclose_match_rate", 0.0))
    if actual_rate < minimum_rate:
        raise AssertionError(
            f"{stage} Rust numerical comparison failed: "
            f"max_abs={results.get('max_error')}, allclose_rate={actual_rate}% "
            f"(required {minimum_rate}%)"
        )
    summary = {
        "stage": stage,
        "asm_lines": len(asm.splitlines()),
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "max_abs_error": results.get("max_error"),
        "allclose_match_rate": results.get("allclose_match_rate"),
        "output_vram_addr": prog.get_vram_addr(current.name),
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "mla",
            "moe",
            "moe_ksplit",
            "chain",
            "attnres",
            "attnres_chain",
            "all",
        ),
        default="all",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("transactional_emulator/testbench/build/kimi3_connected"),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    stages = (
        ("mla", "moe", "chain", "attnres", "attnres_chain")
        if args.stage == "all"
        else (args.stage,)
    )
    summaries = [
        build_and_run(stage, args.build_dir.expanduser().resolve() / stage, seed=args.seed)
        for stage in stages
    ]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
