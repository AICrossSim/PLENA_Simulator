"""Whole-backbone synthetic Kimi K3 transactional proof.

One Rust invocation executes the pinned 93-layer KDA/MLA topology. Prefill is
streamed in 16-token causal chunks, followed by configurable single-token
decode passes.  The outer widths,
head count, and expert count are compact, but every producer-consumer edge,
AttnRes ownership rule, 69 independent KDA recurrent states, 24 independent
compressed MLA histories, dense/LatentMoE FFNs, and ordinary residual is real.

This is an executable topology and lifetime proof.  Its cycles are deliberately
not presented as real-shape Kimi K3 performance.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from analytic_models.reference.kimi_k3_kda import (
    KdaConvWeights,
    KdaShape,
    KdaXState,
    kda_state_engine_prefill,
)
from analytic_models.reference.state_precision import StateStorage
from compiler.aten.kda.scheduler import (
    KIMI_K3_KDA_LAYERS,
    KdaScheduleConfig,
    KimiK3KdaScheduler,
)
from compiler.aten.kimi3.blocks import (
    AttnResConstants,
    KimiLatentMoeConstants,
    KimiLatentMoeShape,
    KimiLatentMoeWeights,
    MlaBlockShape,
    MlaBlockWeights,
    MlaDecodeCache,
    MlaNormConstants,
    allocate_mla_decode_cache,
    emit_kimi_attn_res,
    emit_kimi_dense_ffn_residual_block,
    emit_kimi_latent_moe_residual_block,
    emit_mla_residual_block,
)
from compiler.aten.mamba.scheduler import SchedulePhase
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_routed_moe import KimiSituFPConstants
from compiler.aten.state.isa_lowering import (
    KdaLayerMemoryMap,
    lower_kda_trace_to_existing_isa,
)
from transactional_emulator.testbench.aten.configurable import setup_hw
from transactional_emulator.testbench.aten.golden import (
    _active_precision_settings,
    golden_flash_attention_mha_single_block,
)
from transactional_emulator.testbench.emulator_runner import (
    compare_emulator_output,
    run_emulator,
)
from transactional_emulator.testbench.gpt_oss_testkit import (
    MxfpWeightCache,
    _comparison_diagnostics,
    _comparison_params_for,
    _linear_projection_golden,
    _machine_code_line_count,
    _scan_cache_append_tokens,
)
from transactional_emulator.testbench.layout_utils import (
    infer_hbm_tensor_layouts,
    prestage_bf16_vram_matrix,
    read_bf16_vram_matrix,
)
from transactional_emulator.testbench.models.kimi3.connected_blocks_test import (
    BETA,
    EPS,
    LINEAR_BETA,
    TensorSet,
    _bf16,
    _bf16_layout,
    _exact,
    _register_expert_table,
    _register_weight,
    _rms,
    _set_matrix_kv_plain_bf16,
    _sigmoid,
    _situ,
)
from transactional_emulator.testbench.models.kimi3.kda_connected_test import (
    _patch_hbm,
)
from transactional_emulator.testbench.models.kimi3.mla_cache_connected_test import (
    _rope_tables,
    _rotate_half_matrix,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4
HIDDEN = 64
PREFILL_TOKENS = 16
DECODE_TOKENS = 4
TOTAL_TOKENS = PREFILL_TOKENS + DECODE_TOKENS
PREFILL_CHUNK = 16
NUM_LAYERS = 93
KDA_LAYERS = KIMI_K3_KDA_LAYERS
MLA_LAYERS = tuple(layer for layer in range(NUM_LAYERS) if layer not in KDA_LAYERS)
MOE_LAYERS = tuple(range(1, NUM_LAYERS))
CAPTURE_LAYERS = tuple(range(0, NUM_LAYERS, 12))

KDA_HEADS = 1
KDA_DIM = 128
KDA_KERNEL = 4
MLA_HEADS = 1
Q_LORA = 64
KV_LORA = 64
QK_NOPE = 64
QK_ROPE = 64
V_HEAD = 64
EXPERTS = 4
TOP_K = 2
# Keep the synthetic router away from BF16 top-k boundaries. Dedicated router
# tests cover close scores; this full-backbone fixture is a dataflow proof.
ROUTER_CORRECTION = (0.0, 2.0, 4.0, -2.0)

PREFILL_DESCRIPTOR_BASE = 0
# Descriptors grow with token count. Keep the persistent arena above their
# expected long-decode footprint, then place decode descriptors immediately
# after the realized prefill descriptor region.
KDA_ARENA_BASE = 0x080_0000
KDA_WEIGHT_BASE = 0x1B00_0000
KDA_WEIGHT_SLOT = 0x8000
KDA_WEIGHT_OFFSETS = tuple(index * KDA_WEIGHT_SLOT for index in range(9))
MIN_FULL_MODEL_ALLCLOSE_RATE = 99.8
MAX_FULL_MODEL_RELATIVE_L2_ERROR = 0.03
MAX_PHYSICAL_MLA_CACHE_ABS_ERROR = 0.02
MAX_PHYSICAL_MLA_CACHE_RELATIVE_L2_ERROR = 0.001
_WEIGHT_CACHE = MxfpWeightCache()


def _linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _linear_projection_golden(
        value,
        weight,
        mlen=MLEN,
        hbm_input=False,
        weight_cache=_WEIGHT_CACHE,
    )


def _configure_lengths(prefill_tokens: int, decode_tokens: int) -> None:
    if prefill_tokens <= 0 or decode_tokens <= 0:
        raise ValueError("prefill and decode token counts must be positive")
    if prefill_tokens + decode_tokens > 512:
        raise ValueError("Kimi route dump supports at most 512 total tokens")
    global PREFILL_TOKENS, DECODE_TOKENS, TOTAL_TOKENS
    PREFILL_TOKENS = prefill_tokens
    DECODE_TOKENS = decode_tokens
    TOTAL_TOKENS = prefill_tokens + decode_tokens


def _cache_precision_diagnostics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | int]:
    difference = actual.float() - expected.float()
    close = torch.isclose(actual.float(), expected.float(), atol=0.02, rtol=0.02)
    return {
        "max_abs_error": float(difference.abs().max()),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
        ),
        "allclose_match_rate": float(close.float().mean() * 100.0),
        "mismatch_values": int((~close).sum()),
        "compared_values": int(close.numel()),
    }


@dataclass
class KdaLayer:
    weights: dict[str, torch.Tensor]
    state: KdaXState
    conv_weight: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor


@dataclass
class MlaLayer:
    weights: MlaBlockWeights
    values: dict[str, torch.Tensor]
    input_norm: object
    q_norm: object
    kv_norm: object
    cache: MlaDecodeCache
    compressed_history: list[torch.Tensor]


@dataclass
class MoeLayer:
    weights: KimiLatentMoeWeights
    values: dict[str, object]
    input_norm: object
    routed_norm: object
    ordinal: int


@dataclass
class DenseLayer:
    weights: tuple[object, object, object]
    values: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    input_norm: object


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _hidden_inputs() -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(9304)
    prompt = (torch.randn(PREFILL_TOKENS, HIDDEN, generator=generator) * 0.05).to(torch.bfloat16)
    decode = [
        (torch.randn(1, HIDDEN, generator=generator) * 0.05 + token / 256.0).to(torch.bfloat16)
        for token in range(DECODE_TOKENS)
    ]
    return prompt, decode


def _kda_config(phase: SchedulePhase) -> KdaScheduleConfig:
    return KdaScheduleConfig(
        phase=phase,
        sequence_length=PREFILL_TOKENS if phase == SchedulePhase.PREFILL else 1,
        decode_tokens=DECODE_TOKENS,
        chunk_size=PREFILL_CHUNK,
        matrix_input_features=HIDDEN,
        kda_layer_ids=KDA_LAYERS,
        kda_num_heads=KDA_HEADS,
        kda_key_dim=KDA_DIM,
        kda_value_dim=KDA_DIM,
        kda_conv_kernel=KDA_KERNEL,
        hbm_arena_base=KDA_ARENA_BASE,
        projection_weight_hbm_base=KDA_WEIGHT_BASE,
        projection_weight_layer_stride=len(KDA_WEIGHT_OFFSETS) * KDA_WEIGHT_SLOT,
        projection_weight_offsets=KDA_WEIGHT_OFFSETS,
    )


def _lower_kda_phases():
    prefill_scheduler = KimiK3KdaScheduler(_kda_config(SchedulePhase.PREFILL))
    decode_scheduler = KimiK3KdaScheduler(_kda_config(SchedulePhase.DECODE))
    prefill_trace = prefill_scheduler.build()
    decode_trace = decode_scheduler.build()
    prefill_lowered = lower_kda_trace_to_existing_isa(prefill_trace, descriptor_base=PREFILL_DESCRIPTOR_BASE)
    decode_descriptor_base = _align(
        prefill_lowered.layout_descriptor_base + len(prefill_lowered.layout_descriptor_image),
        256,
    )
    decode_lowered = lower_kda_trace_to_existing_isa(decode_trace, descriptor_base=decode_descriptor_base)
    return (
        prefill_scheduler,
        prefill_trace,
        prefill_lowered,
        decode_scheduler,
        decode_trace,
        decode_lowered,
    )


def _validate_kda_hbm_regions(
    prefill_scheduler,
    prefill_trace,
    prefill_lowered,
    decode_scheduler,
    decode_trace,
    decode_lowered,
) -> dict[str, tuple[int, int]]:
    regions = {
        "prefill_descriptors": (
            prefill_lowered.descriptor_base,
            prefill_lowered.layout_descriptor_base + len(prefill_lowered.layout_descriptor_image),
        ),
        "decode_descriptors": (
            decode_lowered.descriptor_base,
            decode_lowered.layout_descriptor_base + len(decode_lowered.layout_descriptor_image),
        ),
        "state_arena": (
            KDA_ARENA_BASE,
            max(
                prefill_scheduler.hbm_layout().realized_arena_bytes(len(prefill_trace.events)),
                decode_scheduler.hbm_layout().realized_arena_bytes(len(decode_trace.events)),
            ),
        ),
        "projection_weights": (
            KDA_WEIGHT_BASE,
            KDA_WEIGHT_BASE + len(KDA_LAYERS) * len(KDA_WEIGHT_OFFSETS) * KDA_WEIGHT_SLOT,
        ),
    }
    ordered = [
        "prefill_descriptors",
        "decode_descriptors",
        "state_arena",
        "projection_weights",
    ]
    for left, right in pairwise(ordered):
        if regions[left][1] > regions[right][0]:
            raise ValueError(f"KDA HBM regions overlap: {left}={regions[left]} and {right}={regions[right]}")
    return regions


def _event_buckets(trace, lowered) -> dict[tuple[int, int], list[object]]:
    trace_by_index = {event.index: event for event in trace.events}
    buckets: dict[tuple[int, int], list[object]] = {}
    for lowered_event in lowered.events:
        event = trace_by_index[lowered_event.event_index]
        if event.layer_id is None:
            continue
        token = 0 if event.token_offset is None else event.token_offset
        buckets.setdefault((token, event.layer_id), []).append(lowered_event)
    return buckets


def _memories_by_layer(lowered) -> dict[int, KdaLayerMemoryMap]:
    result: dict[int, KdaLayerMemoryMap] = {}
    for event in lowered.events:
        if isinstance(event.memory, KdaLayerMemoryMap):
            previous = result.setdefault(event.memory.layer_id, event.memory)
            if (
                previous.hidden_vram_addr != event.memory.hidden_vram_addr
                or previous.q_weight_hbm_addr != event.memory.q_weight_hbm_addr
                or previous.output_weight_hbm_addr != event.memory.output_weight_hbm_addr
            ):
                raise AssertionError("one KDA layer changed its canonical memory map")
    return result


def _allocate_constants(
    prog: PlenaCompiler,
) -> tuple[MlaNormConstants, KimiLatentMoeConstants, AttnResConstants, list[float]]:
    zero = prog.fp_var("zero", 1)
    attention_scale = prog.fp_var("attention_scale", 1)
    negative_infinity = prog.fp_var("attention_negative_infinity", 1)
    prog.fp_var("attention_online_softmax_workspace", 253)
    state_eps = prog.fp_var("state_eps_backup", 1)
    state_reciprocal = prog.fp_var("state_reciprocal_backup", 1)
    lanes = PREFILL_CHUNK
    one = prog.fp_var("state_one_and_situ_one", lanes)
    neg_one = prog.fp_var("situ_neg_one", lanes)
    beta = prog.fp_var("situ_beta", lanes)
    neg_two_beta = prog.fp_var("situ_neg_two_beta", lanes)
    linear_beta = prog.fp_var("situ_linear_beta", lanes)
    neg_two_linear_beta = prog.fp_var("situ_neg_two_linear_beta", lanes)
    zero_row = prog.fp_var("moe_zero_row", MLEN)
    input_eps = prog.fp_var("input_eps", 1)
    input_reciprocal = prog.fp_var("input_reciprocal", 1)
    q_eps = prog.fp_var("q_eps", 1)
    q_reciprocal = prog.fp_var("q_reciprocal", 1)
    kv_eps = prog.fp_var("kv_eps", 1)
    kv_reciprocal = prog.fp_var("kv_reciprocal", 1)
    routed_eps = prog.fp_var("routed_eps", 1)
    routed_reciprocal = prog.fp_var("routed_reciprocal", 1)

    if (state_eps.address, state_reciprocal.address, one.address) != (256, 257, 258):
        raise AssertionError("KDA FPRAM backups no longer match the X_STATE ABI")
    variables = (
        zero,
        attention_scale,
        negative_infinity,
        state_eps,
        state_reciprocal,
        one,
        neg_one,
        beta,
        neg_two_beta,
        linear_beta,
        neg_two_linear_beta,
        zero_row,
        input_eps,
        input_reciprocal,
        q_eps,
        q_reciprocal,
        kv_eps,
        kv_reciprocal,
        routed_eps,
        routed_reciprocal,
    )
    preload = [0.0] * max(variable.address + variable.size for variable in variables)

    def fill(variable, value: float) -> None:
        for index in range(variable.size):
            preload[variable.address + index] = value

    fill(attention_scale, (QK_NOPE + QK_ROPE) ** -0.5)
    fill(negative_infinity, float("-inf"))
    fill(state_eps, EPS)
    fill(state_reciprocal, 1.0 / KDA_DIM)
    fill(one, 1.0)
    fill(neg_one, -1.0)
    fill(beta, BETA)
    fill(neg_two_beta, -2.0 / BETA)
    fill(linear_beta, LINEAR_BETA)
    fill(neg_two_linear_beta, -2.0 / LINEAR_BETA)
    for variable in (input_eps, q_eps, kv_eps, routed_eps):
        fill(variable, EPS)
    for variable in (
        input_reciprocal,
        q_reciprocal,
        kv_reciprocal,
        routed_reciprocal,
    ):
        fill(variable, 1.0 / HIDDEN)

    norms = MlaNormConstants(
        input_eps=input_eps.address,
        input_reciprocal_hidden=input_reciprocal.address,
        q_eps=q_eps.address,
        q_reciprocal_hidden=q_reciprocal.address,
        kv_eps=kv_eps.address,
        kv_reciprocal_hidden=kv_reciprocal.address,
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
        norm_reciprocal_hidden=input_reciprocal.address,
        routed_norm_eps=routed_eps.address,
        routed_norm_reciprocal_hidden=routed_reciprocal.address,
    )
    attnres = AttnResConstants(
        eps=input_eps.address,
        reciprocal_hidden=input_reciprocal.address,
    )
    return norms, moe, attnres, preload


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


def _register_router_weight(
    prog: PlenaCompiler,
    tensors: TensorSet,
    name: str,
    value: torch.Tensor,
):
    rows, experts = value.shape
    storage = torch.zeros(rows, _align(experts, prog.mlen), dtype=torch.bfloat16)
    storage[:, :experts] = value
    tensors.add(name, storage, bf16=True)
    return prog.input(
        name,
        shape=(rows, experts),
        physical_shape=tuple(storage.shape),
        real_data_ratio=2.0,
    )


def _load_bf16_matrix(prog: PlenaCompiler, value, name: str):
    return prog.load_batch(value, name=name, storage_precision=2, hbm_precision=1)


def _kda_weight_values(layer: int) -> dict[str, torch.Tensor]:
    eye = torch.eye(HIDDEN, dtype=torch.bfloat16)
    repeated_eye = torch.cat((eye, eye), dim=1)
    reduced_eye = torch.cat((eye, eye), dim=0)
    qk_scale = 0.5 + (layer % 5) / 32.0
    value_scale = 0.375 + (layer % 3) / 32.0
    output_scale = 0.5 + (layer % 7) / 64.0
    return {
        "q": (repeated_eye.float() * qk_scale).to(torch.bfloat16),
        "k": (repeated_eye.float() * qk_scale).to(torch.bfloat16),
        "v": (repeated_eye.float() * value_scale).to(torch.bfloat16),
        "gate": torch.zeros(HIDDEN, KDA_DIM, dtype=torch.bfloat16),
        "out": (reduced_eye.float() * output_scale).to(torch.bfloat16),
        "decay_a": torch.zeros(HIDDEN, KDA_DIM, dtype=torch.bfloat16),
        "decay_b": torch.zeros(KDA_DIM, KDA_DIM, dtype=torch.bfloat16),
        "beta": torch.zeros(HIDDEN, MLEN, dtype=torch.bfloat16),
        "norm": torch.ones(1, KDA_DIM, dtype=torch.bfloat16),
    }


def _register_kda_layers(
    prog: PlenaCompiler,
    tensors: TensorSet,
    memories: dict[int, KdaLayerMemoryMap],
) -> dict[int, KdaLayer]:
    result = {}
    for layer in KDA_LAYERS:
        memory = memories[layer]
        values = _kda_weight_values(layer)
        addresses = {
            "q": memory.q_weight_hbm_addr,
            "k": memory.k_weight_hbm_addr,
            "v": memory.v_weight_hbm_addr,
            "gate": memory.gate_weight_hbm_addr,
            "out": memory.output_weight_hbm_addr,
            "decay_a": memory.decay_a_weight_hbm_addr,
            "decay_b": memory.decay_b_weight_hbm_addr,
            "beta": memory.beta_weight_hbm_addr,
            "norm": memory.norm_weight_hbm_addr,
        }
        for key, value in values.items():
            _register_fixed_weight(
                prog,
                tensors,
                f"KDA_L{layer}_{key.upper()}",
                value,
                addresses[key],
                bf16=key == "norm",
            )
        shape = KdaShape(
            HIDDEN,
            KDA_HEADS,
            KDA_DIM,
            KDA_DIM,
            KDA_KERNEL,
            chunk_size=PREFILL_CHUNK,
        )
        result[layer] = KdaLayer(
            weights=values,
            state=KdaXState.zeros(shape, 1),
            conv_weight=torch.tensor([0.125, -0.25, 0.375, 0.5], dtype=torch.bfloat16).repeat(KDA_DIM, 1),
            a_log=torch.full((KDA_HEADS,), -0.5, dtype=torch.bfloat16),
            dt_bias=torch.zeros(KDA_HEADS, KDA_DIM, dtype=torch.bfloat16),
        )
    return result


def _register_mla_layer(
    prog: PlenaCompiler,
    tensors: TensorSet,
    shape: MlaBlockShape,
    layer: int,
) -> MlaLayer:
    prefix = f"KIMI_L{layer}_MLA"
    rotate = _rotate_half_matrix(QK_ROPE)
    values = {
        "q_a": _exact((HIDDEN, Q_LORA), layer + 1, 1, 1 / 128),
        "q_b": _exact((Q_LORA, shape.q_width), layer + 2, 3, 1 / 128),
        "kv_a": _exact((HIDDEN, shape.kv_a_width), layer + 3, 5, 1 / 128),
        "kv_b": _exact((KV_LORA, shape.kv_b_width), layer + 4, 7, 1 / 128),
        "out": _exact((shape.attention_width, HIDDEN), layer + 5, 9, 1 / 128),
        "q_rotate": rotate,
        "k_rotate": rotate,
        "gate": _exact((HIDDEN, shape.attention_width), layer + 6, 11, 1 / 128),
        "input_norm": torch.ones(PREFILL_CHUNK, HIDDEN, dtype=torch.bfloat16),
        "q_norm": torch.ones(PREFILL_CHUNK, Q_LORA, dtype=torch.bfloat16),
        "kv_norm": torch.ones(PREFILL_CHUNK, KV_LORA, dtype=torch.bfloat16),
    }
    weights = MlaBlockWeights(
        q_a=_register_weight(prog, tensors, f"{prefix}_Q_A", values["q_a"]),
        q_b=_register_weight(prog, tensors, f"{prefix}_Q_B", values["q_b"]),
        kv_a=_register_weight(prog, tensors, f"{prefix}_KV_A", values["kv_a"]),
        kv_b=_register_weight(prog, tensors, f"{prefix}_KV_B", values["kv_b"]),
        out=_register_weight(prog, tensors, f"{prefix}_OUT", values["out"]),
        q_rope_rotate=_register_weight(prog, tensors, f"{prefix}_Q_ROTATE", values["q_rotate"], bf16=True),
        k_rope_rotate=_register_weight(prog, tensors, f"{prefix}_K_ROTATE", values["k_rotate"], bf16=True),
        gate=_register_weight(prog, tensors, f"{prefix}_GATE", values["gate"]),
    )
    cache = allocate_mla_decode_cache(
        prog,
        shape=shape,
        max_tokens=TOTAL_TOKENS,
        name=f"kimi_layer{layer}_mla_cache",
    )
    cache.assert_hbm_contract(prog)
    for backing in cache.all_backings:
        tensors.add(
            backing.name,
            torch.zeros(backing.physical_shape, dtype=torch.bfloat16),
            bf16=True,
        )
    return MlaLayer(
        weights=weights,
        values=values,
        input_norm=_register_weight(
            prog,
            tensors,
            f"{prefix}_INPUT_NORM",
            values["input_norm"],
            bf16=True,
        ),
        q_norm=_register_weight(
            prog,
            tensors,
            f"{prefix}_Q_NORM",
            values["q_norm"],
            bf16=True,
        ),
        kv_norm=_register_weight(
            prog,
            tensors,
            f"{prefix}_KV_NORM",
            values["kv_norm"],
            bf16=True,
        ),
        cache=cache,
        compressed_history=[],
    )


def _register_moe_layer(
    prog: PlenaCompiler,
    tensors: TensorSet,
    layer: int,
    ordinal: int,
) -> MoeLayer:
    prefix = f"KIMI_L{layer}_MOE"
    router = torch.zeros(HIDDEN, EXPERTS, dtype=torch.bfloat16)
    for expert in range(EXPERTS):
        router[:, expert] = _exact((HIDDEN,), layer + expert + 1, expert, scale=1 / 96)
    gate_values = [_exact((HIDDEN, HIDDEN), layer + expert + 1, expert, 1 / 96) for expert in range(EXPERTS)]
    up_values = [_exact((HIDDEN, HIDDEN), layer + expert + 2, expert + 1, 1 / 96) for expert in range(EXPERTS)]
    down_values = [_exact((HIDDEN, HIDDEN), layer + expert + 3, expert + 2, 1 / 96) for expert in range(EXPERTS)]
    values: dict[str, object] = {
        "router": router,
        "routed_down": _exact((HIDDEN, HIDDEN), layer + 2, 2, 1 / 96),
        "routed_up": _exact((HIDDEN, HIDDEN), layer + 3, 3, 1 / 96),
        "gate": gate_values,
        "up": up_values,
        "down": down_values,
        "shared_gate": _exact((HIDDEN, HIDDEN), layer + 4, 1, 1 / 96),
        "shared_up": _exact((HIDDEN, HIDDEN), layer + 2, 3, 1 / 96),
        "shared_down": _exact((HIDDEN, HIDDEN), layer + 3, 4, 1 / 96),
        "input_norm": torch.ones(PREFILL_CHUNK, HIDDEN, dtype=torch.bfloat16),
        "routed_norm": torch.ones(PREFILL_CHUNK, HIDDEN, dtype=torch.bfloat16),
    }
    weights = KimiLatentMoeWeights(
        router=_register_router_weight(prog, tensors, f"{prefix}_ROUTER", router),
        routed_down=_register_weight(prog, tensors, f"{prefix}_ROUTED_DOWN", values["routed_down"]),
        routed_up=_register_weight(prog, tensors, f"{prefix}_ROUTED_UP", values["routed_up"]),
        routed_gate=_register_expert_table(prog, tensors, prefix=f"{prefix}_EXPERT_GATE", values=gate_values),
        routed_up_expert=_register_expert_table(prog, tensors, prefix=f"{prefix}_EXPERT_UP", values=up_values),
        routed_down_expert=_register_expert_table(prog, tensors, prefix=f"{prefix}_EXPERT_DOWN", values=down_values),
        shared=(
            _register_weight(prog, tensors, f"{prefix}_SHARED_GATE", values["shared_gate"]),
            _register_weight(prog, tensors, f"{prefix}_SHARED_UP", values["shared_up"]),
            _register_weight(prog, tensors, f"{prefix}_SHARED_DOWN", values["shared_down"]),
        ),
    )
    return MoeLayer(
        weights=weights,
        values=values,
        input_norm=_register_weight(
            prog,
            tensors,
            f"{prefix}_INPUT_NORM",
            values["input_norm"],
            bf16=True,
        ),
        routed_norm=_register_weight(
            prog,
            tensors,
            f"{prefix}_ROUTED_NORM",
            values["routed_norm"],
            bf16=True,
        ),
        ordinal=ordinal,
    )


def _register_dense_layer(prog: PlenaCompiler, tensors: TensorSet) -> DenseLayer:
    values = (
        _exact((HIDDEN, HIDDEN), 3, 1, 1 / 96),
        _exact((HIDDEN, HIDDEN), 5, 3, 1 / 96),
        _exact((HIDDEN, HIDDEN), 7, 5, 1 / 96),
    )
    return DenseLayer(
        weights=tuple(
            _register_weight(prog, tensors, f"KIMI_DENSE_{name}", value)
            for name, value in zip(("GATE", "UP", "DOWN"), values, strict=True)
        ),
        values=values,
        input_norm=_register_weight(
            prog,
            tensors,
            "KIMI_DENSE_INPUT_NORM",
            torch.ones(PREFILL_CHUNK, HIDDEN, dtype=torch.bfloat16),
            bf16=True,
        ),
    )


def _kda_cpu(
    hidden: torch.Tensor,
    layer: KdaLayer,
    *,
    precision,
) -> torch.Tensor:
    shape = KdaShape(
        HIDDEN,
        KDA_HEADS,
        KDA_DIM,
        KDA_DIM,
        KDA_KERNEL,
        chunk_size=PREFILL_CHUNK,
    )
    weights = layer.weights
    q = _linear(hidden, weights["q"])
    k = _linear(hidden, weights["k"])
    v = _linear(hidden, weights["v"])
    decay = _linear(_linear(hidden, weights["decay_a"]), weights["decay_b"])
    beta = _linear(hidden, weights["beta"])[:, :KDA_HEADS]
    projected = torch.cat((q, k, v, decay, beta), dim=-1).unsqueeze(0)
    zero_bias = torch.zeros(KDA_DIM, dtype=torch.bfloat16)
    state_output, layer.state = kda_state_engine_prefill(
        projected,
        layer.state,
        KdaConvWeights(
            q=layer.conv_weight,
            k=layer.conv_weight,
            v=layer.conv_weight,
            q_bias=zero_bias,
            k_bias=zero_bias,
            v_bias=zero_bias,
        ),
        layer.a_log,
        layer.dt_bias,
        shape,
        state_storage=StateStorage.FP32,
        conv_state_storage=StateStorage.BF16,
    )
    value = _bf16(state_output.squeeze(0), precision=precision)
    normalized = _bf16(
        _rms(value, precision=precision).float() * weights["norm"].float(),
        precision=precision,
    )
    gate = _sigmoid(_linear(hidden, weights["gate"]), precision=precision)
    gated = _bf16(normalized.float() * gate.float(), precision=precision)
    return _linear(gated, weights["out"])


def _mla_cache_projection_cpu(
    hidden: torch.Tensor,
    layer: MlaLayer,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    precision,
) -> torch.Tensor:
    values = layer.values
    mixer = _bf16(
        _rms(hidden, precision=precision).float() * values["input_norm"][:1].float(),
        precision=precision,
    )
    compressed = _linear(mixer, values["kv_a"])
    kv_latent = _bf16(
        _rms(compressed[:, :KV_LORA], precision=precision).float() * values["kv_norm"][:1].float(),
        precision=precision,
    )
    k_rope = compressed[:, KV_LORA:]
    k_rope_rot = _linear(k_rope, values["k_rotate"])
    k_rope = _bf16(
        _bf16(k_rope * cos, precision=precision) + _bf16(k_rope_rot * sin, precision=precision),
        precision=precision,
    )
    return mixer, torch.cat((kv_latent, k_rope), dim=-1).to(torch.bfloat16)


def _mla_cpu(
    hidden: torch.Tensor,
    layer: MlaLayer,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
    precision,
) -> torch.Tensor:
    values = layer.values
    mixer, compressed_rows = _mla_cache_projection_cpu(
        hidden,
        layer,
        cos=cos,
        sin=sin,
        precision=precision,
    )
    outputs = []
    for token in range(hidden.shape[0]):
        token_mixer = mixer[token : token + 1]
        q_latent = _linear(token_mixer, values["q_a"])
        q_latent = _bf16(
            _rms(q_latent, precision=precision).float() * values["q_norm"][:1].float(),
            precision=precision,
        )
        q_all = _linear(q_latent, values["q_b"])
        token_cos = cos[token : token + 1]
        token_sin = sin[token : token + 1]
        compressed_row = compressed_rows[token : token + 1]
        layer.compressed_history.append(compressed_row)
        history = torch.cat(layer.compressed_history, dim=0)
        history_latent = history[:, :KV_LORA]
        history_rope = history[:, KV_LORA:]

        head_outputs = []
        for head in range(MLA_HEADS):
            q_start = head * (QK_NOPE + QK_ROPE)
            q_head = q_all[:, q_start : q_start + QK_NOPE + QK_ROPE]
            q_rope = q_head[:, QK_NOPE:]
            q_rope_rot = _linear(q_rope, values["q_rotate"])
            q_rope = _bf16(
                _bf16(q_rope * token_cos, precision=precision) + _bf16(q_rope_rot * token_sin, precision=precision),
                precision=precision,
            )
            q_head = torch.cat((q_head[:, :QK_NOPE], q_rope), dim=-1)
            kv_start = head * (QK_NOPE + V_HEAD)
            kv_head = _linear(
                history_latent,
                values["kv_b"][:, kv_start : kv_start + QK_NOPE + V_HEAD],
            )
            key = torch.cat((kv_head[:, :QK_NOPE], history_rope), dim=-1)
            value = kv_head[:, QK_NOPE:]
            head_outputs.append(
                golden_flash_attention_mha_single_block(
                    q_head,
                    key,
                    value.float(),
                    (QK_NOPE + QK_ROPE) ** -0.5,
                    precision=precision,
                )
            )
        attention = torch.cat(head_outputs, dim=-1)
        gate = _sigmoid(_linear(token_mixer, values["gate"]), precision=precision)
        attention = _bf16(attention.float() * gate.float(), precision=precision)
        outputs.append(_linear(attention, values["out"]))
    return torch.cat(outputs, dim=0)


def _moe_cpu(
    hidden: torch.Tensor,
    layer: MoeLayer,
    correction: torch.Tensor,
    *,
    precision,
) -> tuple[torch.Tensor, list[list[int]]]:
    values = layer.values
    mixer = _bf16(
        _rms(hidden, precision=precision).float() * values["input_norm"][: hidden.shape[0]].float(),
        precision=precision,
    )
    logits = _linear(mixer, values["router"])
    routed_input = _linear(mixer, values["routed_down"])
    outputs = []
    routes = []
    for token in range(hidden.shape[0]):
        ranking = logits[token].float() + correction[0, :EXPERTS].float()
        selected = sorted(
            range(EXPERTS),
            key=lambda expert: (-float(ranking[expert]), expert),
        )[:TOP_K]
        routes.append(selected)
        raw = torch.tensor(
            [float(logits[token, expert]) for expert in selected],
            dtype=torch.float32,
        )
        route_weights = torch.sigmoid(raw)
        route_weights = _bf16(route_weights / route_weights.sum(), precision=precision)
        accumulator = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
        token_input = routed_input[token : token + 1]
        for slot, expert in enumerate(selected):
            gate = _linear(token_input, values["gate"][expert])
            up = _linear(token_input, values["up"][expert])
            expert_out = _linear(_situ(gate, up, precision=precision), values["down"][expert])
            weighted = _bf16(
                expert_out.float() * route_weights[slot].float(),
                precision=precision,
            )
            accumulator = _bf16(accumulator.float() + weighted.float(), precision=precision)
        accumulator = _bf16(
            _rms(accumulator, precision=precision).float() * values["routed_norm"][:1].float(),
            precision=precision,
        )
        routed = _linear(accumulator, values["routed_up"])
        token_mixer = mixer[token : token + 1]
        shared = _linear(
            _situ(
                _linear(token_mixer, values["shared_gate"]),
                _linear(token_mixer, values["shared_up"]),
                precision=precision,
            ),
            values["shared_down"],
        )
        outputs.append(_bf16(routed.float() + shared.float(), precision=precision))
    return torch.cat(outputs, dim=0), routes


def _dense_cpu(
    hidden: torch.Tensor,
    layer: DenseLayer,
    *,
    precision,
) -> torch.Tensor:
    normalized = _bf16(
        _rms(hidden, precision=precision).float() * torch.ones_like(hidden).float(),
        precision=precision,
    )
    gate = _linear(normalized, layer.values[0])
    up = _linear(normalized, layer.values[1])
    return _linear(_situ(gate, up, precision=precision), layer.values[2])


def _attnres_cpu(
    block_residuals: tuple[torch.Tensor, ...],
    prefix: torch.Tensor,
    score_weight: torch.Tensor,
    *,
    precision,
) -> torch.Tensor:
    candidates = (*block_residuals, prefix)
    scores = []
    for candidate in candidates:
        product = _bf16(
            _rms(candidate, precision=precision).float() * _bf16(score_weight, precision=precision).float(),
            precision=precision,
        )
        scores.append(_bf16(product.float().sum(dim=-1, keepdim=True), precision=precision))
    stacked = torch.cat(scores, dim=-1)
    maximum = stacked.max(dim=-1, keepdim=True).values
    exponentials = _bf16(
        torch.exp(_bf16(stacked.float() - maximum.float(), precision=precision).float()),
        precision=precision,
    )
    denominator = torch.zeros_like(exponentials[:, :1])
    for index in range(exponentials.shape[1]):
        denominator = _bf16(
            denominator.float() + exponentials[:, index : index + 1].float(),
            precision=precision,
        )
    probabilities = _bf16(
        exponentials.float() * _bf16(torch.reciprocal(denominator.float()), precision=precision).float(),
        precision=precision,
    )
    output = _bf16(
        candidates[0].float() * probabilities[:, :1].float(),
        precision=precision,
    )
    for index, candidate in enumerate(candidates[1:], start=1):
        output = _bf16(
            output.float()
            + _bf16(
                candidate.float() * probabilities[:, index : index + 1].float(),
                precision=precision,
            ).float(),
            precision=precision,
        )
    return output


def build_and_run(
    build_dir: Path,
    *,
    stage_profile: bool = False,
    reuse_emulator_output: bool = False,
) -> dict[str, object]:
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(
        argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None),
        build_dir,
    )
    _set_matrix_kv_plain_bf16()
    (
        prefill_scheduler,
        prefill_trace,
        prefill_lowered,
        decode_scheduler,
        decode_trace,
        decode_lowered,
    ) = _lower_kda_phases()
    prefill_buckets = _event_buckets(prefill_trace, prefill_lowered)
    decode_buckets = _event_buckets(decode_trace, decode_lowered)
    hbm_regions = _validate_kda_hbm_regions(
        prefill_scheduler,
        prefill_trace,
        prefill_lowered,
        decode_scheduler,
        decode_trace,
        decode_lowered,
    )
    prefill_memories = _memories_by_layer(prefill_lowered)
    decode_memories = _memories_by_layer(decode_lowered)
    if set(prefill_memories) != set(KDA_LAYERS):
        raise AssertionError("prefill lowering omitted a KDA layer")
    for layer in KDA_LAYERS:
        prefill_memory = prefill_memories[layer]
        decode_memory = decode_memories[layer]
        if (
            prefill_memory.q_weight_hbm_addr != decode_memory.q_weight_hbm_addr
            or prefill_memory.output_weight_hbm_addr != decode_memory.output_weight_hbm_addr
        ):
            raise AssertionError("prefill/decode changed persistent KDA weights")

    workspace_end = max(
        memory.normalization_scratch_vram_addr + MLEN
        for memory in (*prefill_memories.values(), *decode_memories.values())
    )
    prog = PlenaCompiler(
        mlen=MLEN,
        blen=BLEN,
        real_data_ratio=hw.real_data_ratio,
        compact_matrix_loops=True,
    )
    prog.vram_allocator._vmm.mark_used(0, workspace_end, name="KIMI_KDA_PHYSICAL_WORKSPACE")
    fixed_hbm_end = max(end for _, end in hbm_regions.values())
    prog._next_hbm_addr = _align(fixed_hbm_end, MLEN)
    mla_norms, moe_constants, attnres_constants, fp_preload = _allocate_constants(prog)

    prompt_value, decode_values = _hidden_inputs()
    rope_cos, rope_sin = _rope_tables(TOTAL_TOKENS, QK_ROPE)
    prefill_starts = tuple(range(0, PREFILL_TOKENS, PREFILL_CHUNK))
    tile = MLEN * MLEN
    cursor = _align(workspace_end, tile)

    def reserve() -> int:
        nonlocal cursor
        address = cursor
        cursor += tile
        return address

    prompt_addrs = [reserve() for _ in prefill_starts]
    decode_addrs = [reserve() for _ in range(DECODE_TOKENS)]
    prefill_cos_addrs = [reserve() for _ in prefill_starts]
    prefill_sin_addrs = [reserve() for _ in prefill_starts]
    decode_cos_addrs = [reserve() for _ in range(DECODE_TOKENS)]
    decode_sin_addrs = [reserve() for _ in range(DECODE_TOKENS)]
    correction_addr = reserve()
    vram_preload = torch.zeros(cursor, dtype=torch.bfloat16)
    prompt_chunks = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"KIMI_PROMPT_CHUNK_{start}",
            tensor=prompt_value[start : start + PREFILL_CHUNK],
            vram_addr=prompt_addrs[index],
            physical_shape=(min(PREFILL_CHUNK, PREFILL_TOKENS - start), HIDDEN),
            vram_preload=vram_preload,
        )
        for index, start in enumerate(prefill_starts)
    ]
    decode_inputs = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"KIMI_DECODE_TOKEN_{token}",
            tensor=value,
            vram_addr=decode_addrs[token],
            physical_shape=(BLEN, HIDDEN),
            vram_preload=vram_preload,
        )
        for token, value in enumerate(decode_values)
    ]
    prefill_cos_chunks = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"KIMI_PREFILL_ROPE_COS_{start}",
            tensor=rope_cos[start : start + PREFILL_CHUNK],
            vram_addr=prefill_cos_addrs[index],
            physical_shape=(MLEN, QK_ROPE),
            vram_preload=vram_preload,
        )
        for index, start in enumerate(prefill_starts)
    ]
    prefill_sin_chunks = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"KIMI_PREFILL_ROPE_SIN_{start}",
            tensor=rope_sin[start : start + PREFILL_CHUNK],
            vram_addr=prefill_sin_addrs[index],
            physical_shape=(MLEN, QK_ROPE),
            vram_preload=vram_preload,
        )
        for index, start in enumerate(prefill_starts)
    ]
    decode_cos = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"KIMI_DECODE_ROPE_COS_{token}",
            tensor=rope_cos[PREFILL_TOKENS + token : PREFILL_TOKENS + token + 1],
            vram_addr=decode_cos_addrs[token],
            physical_shape=(MLEN, QK_ROPE),
            vram_preload=vram_preload,
        )
        for token in range(DECODE_TOKENS)
    ]
    decode_sin = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"KIMI_DECODE_ROPE_SIN_{token}",
            tensor=rope_sin[PREFILL_TOKENS + token : PREFILL_TOKENS + token + 1],
            vram_addr=decode_sin_addrs[token],
            physical_shape=(MLEN, QK_ROPE),
            vram_preload=vram_preload,
        )
        for token in range(DECODE_TOKENS)
    ]
    correction_value = torch.tensor(
        [list(ROUTER_CORRECTION) + [0.0] * (MLEN - EXPERTS)],
        dtype=torch.bfloat16,
    )
    correction = prestage_bf16_vram_matrix(
        prog=prog,
        name="KIMI_MOE_CORRECTION",
        tensor=correction_value,
        vram_addr=correction_addr,
        physical_shape=(BLEN, MLEN),
        vram_preload=vram_preload,
    )

    prefill_hidden = prog.alloc_at(
        "KIMI_KDA_PREFILL_HIDDEN",
        rows=PREFILL_CHUNK,
        cols=HIDDEN,
        vram_addr=prefill_memories[KDA_LAYERS[0]].hidden_vram_addr,
        physical_shape=(PREFILL_CHUNK, HIDDEN),
    )
    decode_hidden = prog.alloc_at(
        "KIMI_KDA_DECODE_HIDDEN",
        rows=1,
        cols=HIDDEN,
        vram_addr=decode_memories[KDA_LAYERS[0]].hidden_vram_addr,
        physical_shape=(BLEN, HIDDEN),
    )
    checkpoint_stages = NUM_LAYERS * 2 + 1
    checkpoint_rows = checkpoint_stages * TOTAL_TOKENS
    checkpoint = prog.alloc(
        "KIMI_LAYER_CHECKPOINTS",
        rows=checkpoint_rows,
        cols=HIDDEN,
        strict=False,
        physical_shape=(_align(checkpoint_rows, MLEN), HIDDEN),
    )

    tensors = TensorSet(values={}, bf16_names=set())
    kda_layers = _register_kda_layers(prog, tensors, prefill_memories)
    mla_shape = MlaBlockShape(
        hidden=HIDDEN,
        q_lora=Q_LORA,
        kv_lora=KV_LORA,
        qk_nope=QK_NOPE,
        qk_rope=QK_ROPE,
        v_head=V_HEAD,
        heads=MLA_HEADS,
    )
    mla_layers = {layer: _register_mla_layer(prog, tensors, mla_shape, layer) for layer in MLA_LAYERS}
    dense_layer = _register_dense_layer(prog, tensors)
    moe_layers = {layer: _register_moe_layer(prog, tensors, layer, ordinal) for ordinal, layer in enumerate(MOE_LAYERS)}
    kda_input_norms = {
        layer: _register_weight(
            prog,
            tensors,
            f"KIMI_L{layer}_KDA_INPUT_NORM",
            torch.ones(PREFILL_CHUNK, HIDDEN, dtype=torch.bfloat16),
            bf16=True,
        )
        for layer in KDA_LAYERS
    }
    mixer_scores = {}
    ffn_scores = {}
    score_values = {}
    for layer in range(NUM_LAYERS):
        mixer_value = _exact((1, HIDDEN), layer + 1, 1, 1 / 256)
        ffn_value = _exact((1, HIDDEN), layer + 2, 3, 1 / 256)
        mixer_scores[layer] = _register_weight(
            prog,
            tensors,
            f"KIMI_L{layer}_ATTNRES_MIXER_SCORE",
            mixer_value,
            bf16=True,
        )
        ffn_scores[layer] = _register_weight(
            prog,
            tensors,
            f"KIMI_L{layer}_ATTNRES_FFN_SCORE",
            ffn_value,
            bf16=True,
        )
        score_values[(layer, "mixer")] = mixer_value
        score_values[(layer, "ffn")] = ffn_value
    final_score_value = _exact((1, HIDDEN), 11, 7, 1 / 256)
    final_score = _register_weight(prog, tensors, "KIMI_FINAL_ATTNRES_SCORE", final_score_value, bf16=True)
    final_norm_value = torch.ones(PREFILL_CHUNK, HIDDEN, dtype=torch.bfloat16)
    final_norm = _register_weight(prog, tensors, "KIMI_FINAL_NORM", final_norm_value, bf16=True)

    golden_checkpoint = torch.zeros(checkpoint_rows, HIDDEN, dtype=torch.bfloat16)
    expected_routes: dict[tuple[int, int], list[int]] = {}
    precision = _active_precision_settings()

    def record(stage: int, token_offset: int, rows: int, value, golden) -> None:
        prog.vram_copy_region(
            checkpoint,
            value,
            num_rows=rows,
            num_cols=HIDDEN,
            dst_row_offset=stage * TOTAL_TOKENS + token_offset,
        )
        golden_checkpoint[stage * TOTAL_TOKENS + token_offset : stage * TOTAL_TOKENS + token_offset + rows] = golden

    def run_layers(
        current,
        golden_current: torch.Tensor,
        *,
        rows: int,
        decode_token: int | None,
        cos,
        sin,
        cos_value: torch.Tensor,
        sin_value: torch.Tensor,
        prefill_start: int = 0,
    ):
        owned_current = False
        token_offset = prefill_start if decode_token is None else PREFILL_TOKENS + decode_token
        state_hidden = prefill_hidden if decode_token is None else decode_hidden
        block_residuals = []
        golden_block_residuals = []
        for layer in range(NUM_LAYERS):
            kind = "kda" if layer in KDA_LAYERS else "mla"
            prog.emit_comment(
                f"FULL_SYNTHETIC_KIMI phase={'prefill' if decode_token is None else 'decode'} "
                f"token={token_offset} layer={layer} kind={kind}"
            )
            if layer in CAPTURE_LAYERS:
                block_residuals.append(
                    prog.vram_copy(
                        current,
                        name=f"kimi_l{layer}_block_snapshot_t{token_offset}",
                        num_rows=rows,
                    )
                )
                golden_block_residuals.append(golden_current.clone())

            mixer_score = _load_bf16_matrix(
                prog,
                mixer_scores[layer],
                f"kimi_l{layer}_mixer_score_t{token_offset}",
            )
            mixer_input = emit_kimi_attn_res(
                prog,
                tuple(block_residuals),
                current,
                score_weight=mixer_score,
                constants=attnres_constants,
                rows=rows,
                name=f"kimi_l{layer}_attnres_mixer_t{token_offset}",
            )
            golden_mixer_input = _attnres_cpu(
                tuple(golden_block_residuals),
                golden_current,
                score_values[(layer, "mixer")],
                precision=precision,
            )
            prog.free_tensor(mixer_score)
            prefix_after_mixer = prog.vram_copy(
                current,
                name=f"kimi_l{layer}_prefix_after_mixer_t{token_offset}",
                num_rows=rows,
            )

            if kind == "kda":
                normalized = prog.vram_copy(
                    mixer_input,
                    name=f"kimi_l{layer}_kda_norm_t{token_offset}",
                    num_rows=rows,
                )
                prog.rms_norm(
                    normalized,
                    eps_offset=mla_norms.input_eps,
                    reci_hid_offset=mla_norms.input_reciprocal_hidden,
                )
                norm_weight = _load_bf16_matrix(
                    prog,
                    kda_input_norms[layer],
                    f"kimi_l{layer}_kda_norm_weight_t{token_offset}",
                )
                prog.vram_mul(normalized, norm_weight, num_rows=rows)
                prog.vram_copy_region(state_hidden, normalized, num_rows=rows, num_cols=HIDDEN)
                key = (
                    token_offset if decode_token is None else decode_token,
                    layer,
                )
                bucket = prefill_buckets[key] if decode_token is None else decode_buckets[key]
                for event in bucket:
                    prog.emit(event.assembly)
                mixer_out = state_hidden
                golden_state_input = _rms(golden_mixer_input, precision=precision)
                golden_mixer_out = _kda_cpu(golden_state_input, kda_layers[layer], precision=precision)
                prog.free_tensor(normalized)
                prog.free_tensor(norm_weight)
            else:
                data = mla_layers[layer]
                input_norm = _load_bf16_matrix(
                    prog,
                    data.input_norm,
                    f"kimi_l{layer}_mla_input_norm_t{token_offset}",
                )
                q_norm = _load_bf16_matrix(prog, data.q_norm, f"kimi_l{layer}_mla_q_norm_t{token_offset}")
                kv_norm = _load_bf16_matrix(
                    prog,
                    data.kv_norm,
                    f"kimi_l{layer}_mla_kv_norm_t{token_offset}",
                )
                mixer_out = emit_mla_residual_block(
                    prog,
                    mixer_input,
                    shape=mla_shape,
                    weights=data.weights,
                    cos=cos,
                    sin=sin,
                    norms=mla_norms,
                    input_norm_weight=input_norm,
                    q_norm_weight=q_norm,
                    kv_norm_weight=kv_norm,
                    rows=rows,
                    name=f"kimi_l{layer}_mla_t{token_offset}",
                    add_residual=False,
                    cache=data.cache,
                    token_index=token_offset,
                    causal=decode_token is None,
                )
                golden_mixer_out = _mla_cpu(
                    golden_mixer_input,
                    data,
                    cos=cos_value,
                    sin=sin_value,
                    precision=precision,
                )
                for value in (input_norm, q_norm, kv_norm):
                    prog.free_tensor(value)
            prog.free_tensor(mixer_input)
            prog.vram_add(prefix_after_mixer, mixer_out, num_rows=rows)
            golden_prefix_after_mixer = _bf16(
                golden_current.float() + golden_mixer_out.float(),
                precision=precision,
            )
            record(
                layer * 2,
                token_offset,
                rows,
                prefix_after_mixer,
                golden_prefix_after_mixer,
            )
            if mixer_out is not prefill_hidden and mixer_out is not decode_hidden:
                prog.free_tensor(mixer_out)

            ffn_score = _load_bf16_matrix(
                prog,
                ffn_scores[layer],
                f"kimi_l{layer}_ffn_score_t{token_offset}",
            )
            ffn_input = emit_kimi_attn_res(
                prog,
                tuple(block_residuals),
                prefix_after_mixer,
                score_weight=ffn_score,
                constants=attnres_constants,
                rows=rows,
                name=f"kimi_l{layer}_attnres_ffn_t{token_offset}",
            )
            golden_ffn_input = _attnres_cpu(
                tuple(golden_block_residuals),
                golden_prefix_after_mixer,
                score_values[(layer, "ffn")],
                precision=precision,
            )
            prog.free_tensor(ffn_score)

            if layer == 0:
                input_norm = _load_bf16_matrix(
                    prog,
                    dense_layer.input_norm,
                    f"kimi_dense_norm_t{token_offset}",
                )
                ffn_out = emit_kimi_dense_ffn_residual_block(
                    prog,
                    ffn_input,
                    weights=dense_layer.weights,
                    intermediate=HIDDEN,
                    constants=moe_constants,
                    input_norm_weight=input_norm,
                    rows=rows,
                    name=f"kimi_l0_dense_t{token_offset}",
                    add_residual=False,
                )
                golden_ffn_out = _dense_cpu(golden_ffn_input, dense_layer, precision=precision)
                prog.free_tensor(input_norm)
            else:
                data = moe_layers[layer]
                input_norm = _load_bf16_matrix(
                    prog,
                    data.input_norm,
                    f"kimi_l{layer}_moe_input_norm_t{token_offset}",
                )
                routed_norm = _load_bf16_matrix(
                    prog,
                    data.routed_norm,
                    f"kimi_l{layer}_moe_routed_norm_t{token_offset}",
                )
                route_base = (
                    token_offset * TOP_K if decode_token is None else PREFILL_TOKENS * TOP_K + decode_token * TOP_K
                )
                ffn_out = emit_kimi_latent_moe_residual_block(
                    prog,
                    ffn_input,
                    shape=KimiLatentMoeShape(
                        hidden=HIDDEN,
                        routed_hidden=HIDDEN,
                        intermediate=HIDDEN,
                        shared_intermediate=HIDDEN,
                        num_experts=EXPERTS,
                        top_k=TOP_K,
                    ),
                    weights=data.weights,
                    correction_bias=correction,
                    constants=moe_constants,
                    input_norm_weight=input_norm,
                    routed_norm_weight=routed_norm,
                    rows=rows,
                    int_sram_base=route_base,
                    name=f"kimi_l{layer}_moe_t{token_offset}",
                    add_residual=False,
                    loop_topk=decode_token is not None,
                )
                golden_ffn_out, routes = _moe_cpu(
                    golden_ffn_input,
                    data,
                    correction_value,
                    precision=precision,
                )
                for local_token, route in enumerate(routes):
                    expected_routes[(layer, token_offset + local_token)] = route
                prog.free_tensor(input_norm)
                prog.free_tensor(routed_norm)
            prog.free_tensor(ffn_input)

            next_prefix = prog.vram_copy(
                prefix_after_mixer,
                name=f"kimi_l{layer}_prefix_after_ffn_t{token_offset}",
                num_rows=rows,
            )
            prog.vram_add(next_prefix, ffn_out, num_rows=rows)
            golden_next = _bf16(
                golden_prefix_after_mixer.float() + golden_ffn_out.float(),
                precision=precision,
            )
            record(layer * 2 + 1, token_offset, rows, next_prefix, golden_next)
            prog.free_tensor(ffn_out)
            prog.free_tensor(prefix_after_mixer)
            if owned_current:
                prog.free_tensor(current)
            current = next_prefix
            golden_current = golden_next
            owned_current = True

        output_score = _load_bf16_matrix(prog, final_score, f"kimi_final_score_t{token_offset}")
        output = emit_kimi_attn_res(
            prog,
            tuple(block_residuals),
            current,
            score_weight=output_score,
            constants=attnres_constants,
            rows=rows,
            name=f"kimi_final_attnres_t{token_offset}",
        )
        golden_output = _attnres_cpu(
            tuple(golden_block_residuals),
            golden_current,
            final_score_value,
            precision=precision,
        )
        prog.free_tensor(output_score)
        output_norm = _load_bf16_matrix(prog, final_norm, f"kimi_final_norm_t{token_offset}")
        prog.rms_norm(
            output,
            eps_offset=mla_norms.input_eps,
            reci_hid_offset=mla_norms.input_reciprocal_hidden,
        )
        prog.vram_mul(output, output_norm, num_rows=rows)
        golden_output = _rms(golden_output, precision=precision)
        prog.free_tensor(output_norm)
        record(NUM_LAYERS * 2, token_offset, rows, output, golden_output)
        for snapshot in block_residuals:
            prog.free_tensor(snapshot)
        if owned_current:
            prog.free_tensor(current)
        return output

    for start, prompt_chunk, cos_chunk, sin_chunk in zip(
        prefill_starts,
        prompt_chunks,
        prefill_cos_chunks,
        prefill_sin_chunks,
        strict=True,
    ):
        rows = min(PREFILL_CHUNK, PREFILL_TOKENS - start)
        prefill_output = run_layers(
            prompt_chunk,
            prompt_value[start : start + rows],
            rows=rows,
            decode_token=None,
            cos=cos_chunk,
            sin=sin_chunk,
            cos_value=rope_cos[start : start + rows],
            sin_value=rope_sin[start : start + rows],
            prefill_start=start,
        )
        prog.free_tensor(prefill_output)
    for token, (decode_input, decode_value) in enumerate(zip(decode_inputs, decode_values, strict=True)):
        decode_output = run_layers(
            decode_input,
            decode_value,
            rows=1,
            decode_token=token,
            cos=decode_cos[token],
            sin=decode_sin[token],
            cos_value=rope_cos[PREFILL_TOKENS + token : PREFILL_TOKENS + token + 1],
            sin_value=rope_sin[PREFILL_TOKENS + token : PREFILL_TOKENS + token + 1],
        )
        prog.free_tensor(decode_output)

    cache_readbacks = []
    for layer, data in mla_layers.items():
        data.cache.assert_hbm_contract(prog)
        if len(data.compressed_history) != TOTAL_TOKENS:
            raise AssertionError(f"layer {layer} MLA history has {len(data.compressed_history)} rows")
        cache_readbacks.append(
            (
                layer,
                data,
                prog.load_batch(
                    data.cache.compressed.prefix(TOTAL_TOKENS),
                    name=f"kimi_layer{layer}_compressed_final_readback",
                    storage_precision=2,
                    hbm_precision=1,
                ),
            )
        )

    assembly = None
    if not reuse_emulator_output:
        assembly = prog.compile()
        cache_append_tokens = _scan_cache_append_tokens(
            assembly,
            (data.cache.compressed.backing.name for data in mla_layers.values()),
        )
        for layer, data, _readback in cache_readbacks:
            written_tokens = cache_append_tokens[data.cache.compressed.backing.name]
            if written_tokens != list(range(TOTAL_TOKENS)):
                raise AssertionError(f"layer {layer} compressed-cache lifetime mismatch: {written_tokens}")
            for scratch in data.cache.scratch_backings:
                if f"DECODE_CACHE_APPEND {scratch.name}" in assembly:
                    raise AssertionError("reconstructed MLA scratch became persistent")

    hbm_size = _align(
        max(
            prog._next_hbm_addr,
            fixed_hbm_end,
            decode_lowered.layout_descriptor_base + len(decode_lowered.layout_descriptor_image),
        ),
        64,
    )
    params = _comparison_params_for(
        checkpoint,
        rows=checkpoint_rows,
        hidden=HIDDEN,
        mlen=MLEN,
        golden=golden_checkpoint,
    )
    params.update(
        {
            "atol": 0.12,
            "rtol": 0.08,
            "min_allclose_match_rate": MIN_FULL_MODEL_ALLCLOSE_RATE,
        }
    )
    state_profile_path = build_dir / "state_profile.json"
    if reuse_emulator_output:
        required = (
            build_dir / "comparison_params.json",
            build_dir / "golden_output.pt",
            build_dir / "vram_dump.bin",
            build_dir / "intsram_dump.bin",
            state_profile_path,
            build_dir / "rust_emulator_run_stats.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"cannot reuse incomplete emulator output: {missing}")
        existing_params = json.loads((build_dir / "comparison_params.json").read_text())
        for key in ("start_row_idx", "num_rows", "num_batches", "elements_per_batch", "row_dim"):
            if existing_params.get(key) != params.get(key):
                raise AssertionError(f"reused comparison contract changed at {key}")
        stored_golden = torch.load(build_dir / "golden_output.pt", weights_only=True)
        if not torch.equal(stored_golden, golden_checkpoint.float()):
            raise AssertionError("reused emulator output was generated from a different CPU golden")
        metrics = json.loads((build_dir / "rust_emulator_run_stats.json").read_text())
        if metrics.get("return_code") != 0:
            raise AssertionError("reused emulator run did not exit successfully")
    else:
        assert assembly is not None
        layouts = infer_hbm_tensor_layouts(tensors.values)
        for name in tensors.bf16_names:
            layouts[name] = _bf16_layout(tensors.values[name])
        hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in tensors.values}
        create_sim_env(
            tensors.values,
            assembly,
            {"original_output": golden_checkpoint},
            fp_preload=fp_preload,
            int_preload=[0] * 1024,
            build_dir=str(build_dir),
            vram_preload=vram_preload,
            tensor_layouts=layouts,
        )
        create_mem_for_sim(
            data_size=MLEN,
            mode="behave_sim",
            asm=f"kimi3_full_synthetic_s{PREFILL_TOKENS}_decode{DECODE_TOKENS}",
            specified_data_order=sorted(tensors.values, key=hbm_addrs.__getitem__),
            build_path=build_dir,
            input_tensors=tensors.values,
            tensor_layouts=layouts,
            hbm_addrs=hbm_addrs,
        )
        layout = prefill_scheduler.hbm_layout()
        parameter_writes = {}
        zero_bias = torch.zeros(KDA_DIM, dtype=torch.bfloat16)
        for layer, data in kda_layers.items():
            parameter_writes.update(
                {
                    layout.address("q_conv_weight", layer): data.conv_weight,
                    layout.address("k_conv_weight", layer): data.conv_weight,
                    layout.address("v_conv_weight", layer): data.conv_weight,
                    layout.address("q_conv_bias", layer): zero_bias,
                    layout.address("k_conv_bias", layer): zero_bias,
                    layout.address("v_conv_bias", layer): zero_bias,
                    layout.address("a_log", layer): data.a_log,
                    layout.address("dt_bias", layer): data.dt_bias,
                }
            )
        for lowered in (prefill_lowered, decode_lowered):
            _patch_hbm(
                build_dir / "hbm_for_behave_sim.bin",
                descriptor_image=lowered.descriptor_image,
                descriptor_base=lowered.descriptor_base,
                layout_descriptor_image=lowered.layout_descriptor_image,
                layout_descriptor_base=lowered.layout_descriptor_base,
                parameter_writes=parameter_writes,
                minimum_size=hbm_size,
            )
        (build_dir / "comparison_params.json").write_text(json.dumps(params, indent=2) + "\n")
        (build_dir / "generated_asm_code.asm").write_text(assembly)
        (build_dir / "hbm_size.txt").write_text(f"{hbm_size}\n")
        metrics = run_emulator(
            build_dir,
            hbm_size=hbm_size,
            stage_profile=stage_profile,
            state_profile_out=state_profile_path,
            dump_cwd=build_dir,
        )
    results, _ = compare_emulator_output(build_dir, verbose=False)
    diagnostics = _comparison_diagnostics(
        results,
        checkpoint_stages=checkpoint_stages,
        total_tokens=TOTAL_TOKENS,
        hidden=HIDDEN,
        prefill_tokens=PREFILL_TOKENS,
    )
    if (
        float(results.get("allclose_match_rate", 0.0)) < MIN_FULL_MODEL_ALLCLOSE_RATE
        or diagnostics["relative_l2_error"] > MAX_FULL_MODEL_RELATIVE_L2_ERROR
    ):
        raise AssertionError(f"Kimi full synthetic hidden mismatch: metrics={results}, diagnostics={diagnostics}")

    state_profile = json.loads(state_profile_path.read_text())
    state_commands = [command for command in state_profile["commands"] if command["algorithm"] == "kda"]
    subop_counts = {
        subop: sum(command["subop"] == subop for command in state_commands) for subop in ("reset", "prefill", "step")
    }
    expected_subops = {
        "reset": len(KDA_LAYERS),
        "prefill": len(KDA_LAYERS) * len(prefill_starts),
        "step": len(KDA_LAYERS) * DECODE_TOKENS,
    }
    if subop_counts != expected_subops:
        raise AssertionError(f"KDA lifecycle mismatch: expected={expected_subops}, actual={subop_counts}")
    for layer in KDA_LAYERS:
        sequence = [
            command["subop"]
            for command in state_commands
            if command["layer_id"] == layer and command["subop"] in {"reset", "prefill", "step"}
        ]
        if sequence != (["reset"] + ["prefill"] * len(prefill_starts) + ["step"] * DECODE_TOKENS):
            raise AssertionError(f"layer {layer} KDA lifetime is out of order")

    final_layer = MOE_LAYERS[-1]
    expected_final_routes = np.asarray(
        [expected_routes[(final_layer, token)] for token in range(TOTAL_TOKENS)],
        dtype=np.int32,
    )
    actual_final_routes = np.fromfile(build_dir / "intsram_dump.bin", dtype="<i4", count=TOTAL_TOKENS * TOP_K).reshape(
        TOTAL_TOKENS, TOP_K
    )
    np.save(build_dir / "expected_final_routes.npy", expected_final_routes)
    np.save(build_dir / "actual_final_routes.npy", actual_final_routes)
    expected_sorted = np.sort(expected_final_routes, axis=1)
    actual_sorted = np.sort(actual_final_routes, axis=1)
    mismatched_tokens = np.flatnonzero(np.any(actual_sorted != expected_sorted, axis=1))
    if mismatched_tokens.size:
        examples = [
            {
                "token": int(token),
                "expected": expected_final_routes[token].tolist(),
                "actual": actual_final_routes[token].tolist(),
            }
            for token in mismatched_tokens[:8]
        ]
        raise AssertionError(
            "final Kimi MoE routes do not match the CPU reference: "
            f"mismatched_tokens={mismatched_tokens.size}/{TOTAL_TOKENS}, examples={examples}"
        )
    expected_route_count = len(MOE_LAYERS) * TOTAL_TOKENS
    if len(expected_routes) != expected_route_count:
        raise AssertionError(f"route lifecycle missing entries: {len(expected_routes)}/{expected_route_count}")

    simulated_values = results.get("simulated_values")
    if not isinstance(simulated_values, torch.Tensor):
        raise AssertionError("emulator comparison did not return simulated hidden checkpoints")
    actual_checkpoints = simulated_values.reshape(checkpoint_stages, TOTAL_TOKENS, HIDDEN).to(torch.bfloat16)
    actual_inputs = torch.cat((prompt_value, *decode_values), dim=0)
    actual_block_residuals: list[torch.Tensor] = []
    producer_conditioned_cache = {}
    for layer in range(NUM_LAYERS):
        actual_current = actual_inputs if layer == 0 else actual_checkpoints[layer * 2 - 1]
        if layer in CAPTURE_LAYERS:
            actual_block_residuals.append(actual_current)
        actual_mixer_input = _attnres_cpu(
            tuple(actual_block_residuals),
            actual_current,
            score_values[(layer, "mixer")],
            precision=precision,
        )
        if layer in MLA_LAYERS:
            _, cache_rows = _mla_cache_projection_cpu(
                actual_mixer_input,
                mla_layers[layer],
                cos=rope_cos,
                sin=rope_sin,
                precision=precision,
            )
            producer_conditioned_cache[layer] = cache_rows

    model_cache_diagnostics = {}
    physical_cache_diagnostics = {}
    for layer, data, readback in cache_readbacks:
        model_expected = torch.cat(data.compressed_history, dim=0)
        actual = read_bf16_vram_matrix(
            build_dir / "vram_dump.bin",
            address=prog.get_vram_addr(readback.name),
            rows=TOTAL_TOKENS,
            width=mla_shape.kv_a_width,
            physical_rows=readback.physical_shape[0],
            mlen=MLEN,
        )
        model_cache_diagnostics[layer] = _cache_precision_diagnostics(actual, model_expected)
        physical_cache_diagnostics[layer] = _cache_precision_diagnostics(
            actual,
            producer_conditioned_cache[layer],
        )
    (build_dir / "mla_cache_diagnostics.json").write_text(
        json.dumps(
            {
                "model_level": model_cache_diagnostics,
                "producer_conditioned": physical_cache_diagnostics,
            },
            indent=2,
        )
        + "\n"
    )
    invalid_caches = {
        layer: values
        for layer, values in physical_cache_diagnostics.items()
        if values["max_abs_error"] > MAX_PHYSICAL_MLA_CACHE_ABS_ERROR
        or values["relative_l2_error"] > MAX_PHYSICAL_MLA_CACHE_RELATIVE_L2_ERROR
    }
    if invalid_caches:
        raise AssertionError(f"producer-conditioned MLA cache gate failed: {invalid_caches}")

    persistent_names = [backing.name for data in mla_layers.values() for backing in data.cache.persistent_backings]
    if len(persistent_names) != len(MLA_LAYERS) or any(
        "reconstructed" in name or "head" in name for name in persistent_names
    ):
        raise AssertionError("MLA persistent HBM contains expanded K/V history")
    kda_weight_addresses = {memory.q_weight_hbm_addr for memory in prefill_memories.values()}
    if len(kda_weight_addresses) != len(KDA_LAYERS):
        raise AssertionError("KDA layers are not streaming independent weight slots")

    summary = {
        "model": "kimi_k3",
        "scope": "full_93_layer_compact_synthetic_transactional",
        "prefill_tokens": PREFILL_TOKENS,
        "prefill_chunk_tokens": PREFILL_CHUNK,
        "prefill_chunks": len(prefill_starts),
        "decode_tokens": DECODE_TOKENS,
        "layer_counts": {
            "kda": len(KDA_LAYERS),
            "mla": len(MLA_LAYERS),
            "latent_moe": len(MOE_LAYERS),
            "dense_ffn": 1,
            "attnres_capture_points": len(CAPTURE_LAYERS),
        },
        "checkpoint_rows": checkpoint_rows,
        "state_lifecycle": subop_counts,
        "compressed_mla_cache_rows_per_layer": TOTAL_TOKENS,
        "compressed_mla_cache_max_abs_error": max(
            (values["max_abs_error"] for values in model_cache_diagnostics.values()),
            default=0.0,
        ),
        "compressed_mla_cache_max_relative_l2_error": max(
            (values["relative_l2_error"] for values in model_cache_diagnostics.values()),
            default=0.0,
        ),
        "compressed_mla_cache_min_allclose_match_rate": min(
            (values["allclose_match_rate"] for values in model_cache_diagnostics.values()),
            default=100.0,
        ),
        "producer_conditioned_cache_max_abs_error": max(
            (values["max_abs_error"] for values in physical_cache_diagnostics.values()),
            default=0.0,
        ),
        "producer_conditioned_cache_max_relative_l2_error": max(
            (values["relative_l2_error"] for values in physical_cache_diagnostics.values()),
            default=0.0,
        ),
        "persistent_mla_cache_objects": len(persistent_names),
        "expanded_persistent_kv_objects": 0,
        "route_decisions": len(expected_routes) * TOP_K,
        "final_layer_routes_verified": int(expected_final_routes.size),
        "independent_kda_weight_streams": len(kda_weight_addresses),
        "sim_cycles": metrics.get("sim_latency_cycles"),
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "mse": results.get("mse"),
        "mae": results.get("mae"),
        "max_abs_error": results.get("max_error"),
        "mean_relative_error": results.get("relative_error"),
        "relative_match_rate": results.get("relative_match_rate"),
        "allclose_match_rate": results.get("allclose_match_rate"),
        "minimum_allclose_match_rate": MIN_FULL_MODEL_ALLCLOSE_RATE,
        "maximum_relative_l2_error": MAX_FULL_MODEL_RELATIVE_L2_ERROR,
        "comparison_diagnostics": diagnostics,
        "hbm_bytes": hbm_size,
        "instruction_count": _machine_code_line_count(metrics),
        "stage_profile_enabled": bool(metrics.get("stage_profile_requested")),
        "reused_emulator_output": reuse_emulator_output,
        "claim_boundary": ("compact synthetic execution proof; cycles are not real-shape Kimi K3 performance"),
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-tokens", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--stage-profile", action="store_true")
    parser.add_argument(
        "--reuse-emulator-output",
        action="store_true",
        help="rebuild CPU references and validate existing Rust dumps without rerunning the emulator",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("transactional_emulator/testbench/build/kimi3_full_synthetic_connected"),
    )
    args = parser.parse_args()
    _configure_lengths(args.prefill_tokens, args.decode_tokens)
    build_and_run(
        args.build_dir.expanduser().resolve(),
        stage_profile=args.stage_profile,
        reuse_emulator_output=args.reuse_emulator_output,
    )


if __name__ == "__main__":
    main()
