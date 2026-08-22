"""Whole-backbone synthetic Nemotron 3 transactional proof.

The program executes the pinned 52-layer M/E/* order in one Rust invocation:
S16 causal prefill followed by four single-token decode passes.  Widths and
weights are compact and deterministic, while producer-consumer edges, residuals,
23 independent Mamba states, six independent GQA caches, routed/shared MoE and
the physical Matrix/L_SCATTER_M/X_STATE path are real.

This fixture proves executable topology and lifetime correctness.  Its cycle
count is not a real-shape Nemotron performance estimate.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from analytic_models.reference.nemotron3_mamba import (
    Mamba2Shape,
    Mamba2State,
    mamba_state_engine_prefill,
)
from analytic_models.reference.state_precision import StateStorage
from compiler.aten.mamba.scheduler import (
    MambaHbmLayout,
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)
from compiler.aten.nemotron3.blocks import (
    NemotronAttentionShape,
    NemotronAttentionWeights,
    NemotronGqaDecodeCache,
    NemotronMoeConstants,
    NemotronMoeShape,
    NemotronMoeWeights,
    allocate_nemotron_gqa_decode_cache,
    emit_nemotron_attention_block,
    emit_nemotron_moe_block,
)
from compiler.aten.nemotron3.scheduler import (
    HybridLayerType,
    NEMOTRON3_PATTERN,
    SYMBOL_TO_LAYER,
)
from compiler.aten.plena import PlenaCompiler
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
from transactional_emulator.testbench.gpt_oss_testkit import (
    _comparison_params_for,
)
from transactional_emulator.testbench.layout_utils import (
    infer_hbm_tensor_layouts,
    prestage_bf16_vram_matrix,
    read_bf16_vram_matrix,
)
from transactional_emulator.testbench.models.kimi3.connected_blocks_test import (
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
from transactional_emulator.testbench.models.nemotron3.gqa_cache_connected_test import (
    _flash_attn_ref,
)
from transactional_emulator.testbench.models.nemotron3.mamba_connected_test import (
    _patch_hbm,
)
from transactional_emulator.testbench.models.state_prefill_connected_test import (
    _mamba_weights,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4
HIDDEN = 64
PREFILL_TOKENS = 16
DECODE_TOKENS = 4
TOTAL_TOKENS = PREFILL_TOKENS + DECODE_TOKENS
MAMBA_LAYERS = tuple(
    layer for layer, symbol in enumerate(NEMOTRON3_PATTERN) if symbol == "M"
)
ATTENTION_LAYERS = tuple(
    layer for layer, symbol in enumerate(NEMOTRON3_PATTERN) if symbol == "*"
)
MOE_LAYERS = tuple(
    layer for layer, symbol in enumerate(NEMOTRON3_PATTERN) if symbol == "E"
)
QUERY_HEADS = 4
KV_HEADS = 1
HEAD_DIM = 64
EXPERTS = 4
TOP_K = 2
MAMBA_HEADS = 1
MAMBA_HEAD_DIM = 64
MAMBA_STATE_DIM = 128
MAMBA_GROUPS = 1
MAMBA_KERNEL = 4
PREFILL_DESCRIPTOR_BASE = 0
DECODE_DESCRIPTOR_BASE = 0x10_0000
MAMBA_ARENA_BASE = 0x20_0000
EPS = 1.0e-5


@dataclass
class AttentionLayer:
    weights: NemotronAttentionWeights
    values: dict[str, torch.Tensor]
    cache: NemotronGqaDecodeCache
    key_history: list[list[torch.Tensor]]
    value_history: list[list[torch.Tensor]]


@dataclass
class MoeLayer:
    weights: NemotronMoeWeights
    values: dict[str, object]
    correction_input: object
    correction: torch.Tensor
    ordinal: int


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _hidden_inputs() -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(5204)
    prompt = (torch.randn(PREFILL_TOKENS, HIDDEN, generator=generator) * 0.08).to(
        torch.bfloat16
    )
    decode = [
        (torch.randn(1, HIDDEN, generator=generator) * 0.08 + token / 128.0).to(
            torch.bfloat16
        )
        for token in range(DECODE_TOKENS)
    ]
    return prompt, decode


def _state_config(phase: SchedulePhase) -> MambaScheduleConfig:
    return MambaScheduleConfig(
        phase=phase,
        sequence_length=PREFILL_TOKENS if phase == SchedulePhase.PREFILL else 1,
        decode_tokens=DECODE_TOKENS,
        chunk_size=PREFILL_TOKENS,
        matrix_input_features=HIDDEN,
        mamba_layer_ids=MAMBA_LAYERS,
        mamba_num_heads=MAMBA_HEADS,
        mamba_head_dim=MAMBA_HEAD_DIM,
        mamba_state_dim=MAMBA_STATE_DIM,
        mamba_groups=MAMBA_GROUPS,
        mamba_conv_kernel=MAMBA_KERNEL,
        mamba_hbm_arena_base=MAMBA_ARENA_BASE,
    )


def _lower_state_phases():
    prefill_trace = Nemotron3MambaScheduler(
        _state_config(SchedulePhase.PREFILL)
    ).build()
    decode_trace = Nemotron3MambaScheduler(
        _state_config(SchedulePhase.DECODE)
    ).build()
    prefill = lower_mamba_trace_to_existing_isa(
        prefill_trace, descriptor_base=PREFILL_DESCRIPTOR_BASE
    )
    decode = lower_mamba_trace_to_existing_isa(
        decode_trace, descriptor_base=DECODE_DESCRIPTOR_BASE
    )
    return prefill_trace, prefill, decode_trace, decode


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


def _memories_by_layer(lowered) -> dict[int, MambaLayerMemoryMap]:
    result: dict[int, MambaLayerMemoryMap] = {}
    for event in lowered.events:
        if isinstance(event.memory, MambaLayerMemoryMap):
            previous = result.setdefault(event.memory.layer_id, event.memory)
            if (
                previous.hidden_vram_addr != event.memory.hidden_vram_addr
                or previous.input_projection_weight_hbm_addr
                != event.memory.input_projection_weight_hbm_addr
                or previous.output_projection_weight_hbm_addr
                != event.memory.output_projection_weight_hbm_addr
            ):
                raise AssertionError("one Mamba layer changed its canonical memory map")
    return result


def _register_state_weights(
    prog: PlenaCompiler,
    tensors: TensorSet,
    memories: dict[int, MambaLayerMemoryMap],
) -> tuple[dict[str, torch.Tensor], dict[int, torch.Tensor]]:
    weights = _mamba_weights()
    norm_by_layer: dict[int, torch.Tensor] = {}
    for layer in MAMBA_LAYERS:
        memory = memories[layer]
        for suffix, key, address, bf16 in (
            (
                "in",
                "W_MAMBA_IN",
                memory.input_projection_weight_hbm_addr,
                False,
            ),
            (
                "out",
                "W_MAMBA_OUT",
                memory.output_projection_weight_hbm_addr,
                False,
            ),
            (
                "norm",
                "W_MAMBA_NORM",
                memory.norm_weight_hbm_addr,
                True,
            ),
        ):
            name = f"MAMBA_L{layer}_{suffix.upper()}"
            value = weights[key]
            tensors.add(name, value, bf16=bf16)
            prog.input(
                name,
                shape=tuple(value.shape),
                physical_shape=tuple(value.shape),
                hbm_addr=address,
                real_data_ratio=2.0 if bf16 else None,
            )
        norm_by_layer[layer] = weights["W_MAMBA_NORM"]
    return weights, norm_by_layer


def _register_attention_layer(
    prog: PlenaCompiler,
    tensors: TensorSet,
    layer: int,
) -> AttentionLayer:
    prefix = f"NEM_L{layer}_GQA"
    values = {
        "q": _exact((HIDDEN, QUERY_HEADS * HEAD_DIM), layer + 1, 1, 1 / 128),
        "k": _exact((HIDDEN, KV_HEADS * HEAD_DIM), layer + 2, 3, 1 / 128),
        "v": _exact((HIDDEN, KV_HEADS * HEAD_DIM), layer + 3, 5, 1 / 128),
        "out": _exact((QUERY_HEADS * HEAD_DIM, HIDDEN), layer + 4, 7, 1 / 256),
    }
    weights = NemotronAttentionWeights(
        q=_register_weight(prog, tensors, f"{prefix}_Q", values["q"]),
        k=_register_weight(prog, tensors, f"{prefix}_K", values["k"]),
        v=_register_weight(prog, tensors, f"{prefix}_V", values["v"]),
        out=_register_weight(prog, tensors, f"{prefix}_OUT", values["out"]),
    )
    cache = allocate_nemotron_gqa_decode_cache(
        prog,
        shape=NemotronAttentionShape(
            hidden=HIDDEN,
            query_heads=QUERY_HEADS,
            kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
        ),
        max_tokens=TOTAL_TOKENS,
        name=f"nemotron_layer{layer}_gqa_cache",
    )
    for backing in cache.backings:
        tensors.add(
            backing.name,
            torch.zeros(backing.physical_shape, dtype=torch.bfloat16),
            bf16=True,
        )
    return AttentionLayer(
        weights=weights,
        values=values,
        cache=cache,
        key_history=[[] for _ in range(KV_HEADS)],
        value_history=[[] for _ in range(KV_HEADS)],
    )


def _register_router_weight(
    prog: PlenaCompiler,
    tensors: TensorSet,
    name: str,
    value: torch.Tensor,
):
    """Register logical expert columns in an MLEN-aligned Matrix tile."""
    rows, experts = value.shape
    physical_experts = _align(experts, prog.mlen)
    storage = torch.zeros(rows, physical_experts, dtype=torch.bfloat16)
    storage[:, :experts] = value
    tensors.add(name, storage, bf16=True)
    return prog.input(
        name,
        shape=(rows, experts),
        physical_shape=tuple(storage.shape),
        real_data_ratio=2.0,
    )


def _register_moe_layer(
    prog: PlenaCompiler,
    tensors: TensorSet,
    layer: int,
    ordinal: int,
) -> MoeLayer:
    prefix = f"NEM_L{layer}_MOE"
    router = torch.zeros(HIDDEN, EXPERTS, dtype=torch.bfloat16)
    for expert in range(EXPERTS):
        router[:, expert] = _exact(
            (HIDDEN,), layer + expert + 1, expert, scale=1 / 64
        )
    up = [
        _exact((HIDDEN, HIDDEN), layer + expert + 2, expert + 1, 1 / 64)
        for expert in range(EXPERTS)
    ]
    down = [
        _exact((HIDDEN, HIDDEN), layer + expert + 3, expert + 2, 1 / 64)
        for expert in range(EXPERTS)
    ]
    shared_up = _exact((HIDDEN, HIDDEN), layer + 4, 1, 1 / 64)
    shared_down = _exact((HIDDEN, HIDDEN), layer + 3, 4, 1 / 64)
    correction = torch.tensor(
        [[0.0, 0.125, 0.25, -0.125] + [0.0] * (MLEN - EXPERTS)]
        * BLEN,
        dtype=torch.bfloat16,
    )
    correction_input = _register_weight(
        prog,
        tensors,
        f"{prefix}_CORRECTION",
        correction,
        bf16=True,
    )
    weights = NemotronMoeWeights(
        router=_register_router_weight(
            prog, tensors, f"{prefix}_ROUTER", router
        ),
        routed_up=_register_expert_table(
            prog, tensors, prefix=f"{prefix}_EXPERT_UP", values=up
        ),
        routed_down=_register_expert_table(
            prog, tensors, prefix=f"{prefix}_EXPERT_DOWN", values=down
        ),
        shared_up=_register_weight(
            prog, tensors, f"{prefix}_SHARED_UP", shared_up
        ),
        shared_down=_register_weight(
            prog, tensors, f"{prefix}_SHARED_DOWN", shared_down
        ),
    )
    return MoeLayer(
        weights=weights,
        values={
            "router": router,
            "up": up,
            "down": down,
            "shared_up": shared_up,
            "shared_down": shared_down,
        },
        correction_input=correction_input,
        correction=correction[:1],
        ordinal=ordinal,
    )


def _mamba_cpu(
    hidden: torch.Tensor,
    state: Mamba2State,
    weights: dict[str, torch.Tensor],
    *,
    precision,
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
    state_output, next_state = mamba_state_engine_prefill(
        projected.unsqueeze(0),
        state,
        weights["CONV_WEIGHT"],
        weights["A_LOG"],
        weights["DT_BIAS"],
        weights["D_SKIP"],
        shape,
        conv_bias=weights["CONV_BIAS"],
        state_storage=StateStorage.FP32,
    )
    value = _bf16(state_output.squeeze(0), precision=precision)
    gate = projected[:, : shape.d_inner]
    value = _bf16(
        value.float()
        * _bf16(
            gate.float() * _sigmoid(gate, precision=precision).float(),
            precision=precision,
        ).float(),
        precision=precision,
    )
    grouped = value.reshape(hidden.shape[0], shape.groups, -1)
    normalized = torch.cat(
        [
            _rms_norm_vector_ref(group, EPS, precision)
            for group in grouped.unbind(1)
        ],
        dim=-1,
    )
    normalized = _bf16(
        normalized * weights["W_MAMBA_NORM"].float(), precision=precision
    )
    return _linear(normalized, weights["W_MAMBA_OUT"]), next_state


def _attention_cpu(
    hidden: torch.Tensor,
    layer: AttentionLayer,
) -> torch.Tensor:
    q = _linear(hidden, layer.values["q"])
    k = _linear(hidden, layer.values["k"])
    v = _linear(hidden, layer.values["v"])
    outputs = []
    heads_per_kv = QUERY_HEADS // KV_HEADS
    precision = _active_precision_settings()
    for token in range(hidden.shape[0]):
        for kv_head in range(KV_HEADS):
            start = kv_head * HEAD_DIM
            layer.key_history[kv_head].append(k[token : token + 1, start : start + HEAD_DIM])
            layer.value_history[kv_head].append(v[token : token + 1, start : start + HEAD_DIM])
        head_outputs = []
        for q_head in range(QUERY_HEADS):
            kv_head = q_head // heads_per_kv
            start = q_head * HEAD_DIM
            head_outputs.append(
                _flash_attn_ref(
                    q[token : token + 1, start : start + HEAD_DIM],
                    torch.cat(layer.key_history[kv_head], dim=0),
                    torch.cat(layer.value_history[kv_head], dim=0).float(),
                    HEAD_DIM**-0.5,
                    precision=precision,
                )
            )
        outputs.append(_linear(torch.cat(head_outputs, dim=-1), layer.values["out"]))
    return torch.cat(outputs, dim=0)


def _moe_cpu(
    hidden: torch.Tensor,
    layer: MoeLayer,
    *,
    precision,
) -> tuple[torch.Tensor, list[list[int]]]:
    logits = _linear(hidden, layer.values["router"])
    bias = layer.correction[:, :EXPERTS].float()
    outputs = []
    routes = []
    for token in range(hidden.shape[0]):
        ranking = logits[token].float() + bias[0]
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
        route_weights = _bf16(
            route_weights / route_weights.sum() * 2.5,
            precision=precision,
        )
        accumulator = torch.zeros(1, HIDDEN, dtype=torch.bfloat16)
        token_hidden = hidden[token : token + 1]
        for slot, expert in enumerate(selected):
            up = _linear(token_hidden, layer.values["up"][expert])
            activated = _bf16(
                torch.clamp(up.float(), min=0.0), precision=precision
            )
            activated = _bf16(
                activated.float() * activated.float(), precision=precision
            )
            expert_out = _linear(activated, layer.values["down"][expert])
            weighted = _bf16(
                expert_out.float() * route_weights[slot].float(),
                precision=precision,
            )
            accumulator = _bf16(
                accumulator.float() + weighted.float(), precision=precision
            )
        shared = _linear(
            _bf16(
                torch.clamp(
                    _linear(token_hidden, layer.values["shared_up"]).float(),
                    min=0.0,
                )
                ** 2,
                precision=precision,
            ),
            layer.values["shared_down"],
        )
        outputs.append(
            _bf16(
                accumulator.float() + shared.float(), precision=precision
            )
        )
    return torch.cat(outputs, dim=0), routes


def _load_bf16_matrix(prog: PlenaCompiler, input_var, name: str):
    return prog.load_batch(
        input_var,
        name=name,
        storage_precision=2,
        hbm_precision=1,
    )


def build_and_run(build_dir: Path) -> dict[str, object]:
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(
        argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None),
        build_dir,
    )
    _set_matrix_kv_plain_bf16()
    prefill_trace, prefill_lowered, decode_trace, decode_lowered = (
        _lower_state_phases()
    )
    prefill_buckets = _event_buckets(prefill_trace, prefill_lowered)
    decode_buckets = _event_buckets(decode_trace, decode_lowered)
    prefill_memories = _memories_by_layer(prefill_lowered)
    decode_memories = _memories_by_layer(decode_lowered)
    if set(prefill_memories) != set(MAMBA_LAYERS):
        raise AssertionError("prefill lowering omitted a Mamba layer")
    for layer in MAMBA_LAYERS:
        if (
            prefill_memories[layer].input_projection_weight_hbm_addr
            != decode_memories[layer].input_projection_weight_hbm_addr
            or prefill_memories[layer].output_projection_weight_hbm_addr
            != decode_memories[layer].output_projection_weight_hbm_addr
        ):
            raise AssertionError("prefill/decode changed persistent Mamba weights")

    workspace_end = max(
        memory.normalization_scratch_vram_addr + MLEN
        for memory in (*prefill_memories.values(), *decode_memories.values())
    )
    prefill_hidden_addr = prefill_memories[MAMBA_LAYERS[0]].hidden_vram_addr
    decode_hidden_addr = decode_memories[MAMBA_LAYERS[0]].hidden_vram_addr
    prog = PlenaCompiler(
        mlen=MLEN,
        blen=BLEN,
        real_data_ratio=hw.real_data_ratio,
        compact_matrix_loops=True,
    )
    prog.vram_allocator._vmm.mark_used(
        0, workspace_end, name="NEMOTRON_STATE_WORKSPACE"
    )

    prog.fp_var("zero", 1)
    prog.fp_var("attention_scale", 1)
    prog.fp_var("attention_negative_infinity", 1)
    prog.fp_var("attention_online_softmax_workspace", 253)
    state_eps = prog.fp_var("state_eps_backup", 1)
    state_reciprocal = prog.fp_var("state_reciprocal_backup", 1)
    state_one = prog.fp_var("state_one_backup", 1)
    zero_row = prog.fp_var("moe_zero_row", MLEN)
    block_eps = prog.fp_var("block_eps", 1)
    block_reciprocal = prog.fp_var("block_reciprocal", 1)
    route_scale = prog.fp_var("route_scale", TOP_K)
    fp_preload = [0.0] * (route_scale.address + route_scale.size)
    fp_preload[1] = (HEAD_DIM**-0.5) / 0.25
    fp_preload[2] = float("-inf")
    fp_preload[state_eps.address] = EPS
    fp_preload[state_reciprocal.address] = (
        MAMBA_GROUPS / (MAMBA_HEADS * MAMBA_HEAD_DIM)
    )
    fp_preload[state_one.address] = 1.0
    fp_preload[block_eps.address] = EPS
    fp_preload[block_reciprocal.address] = 1.0 / HIDDEN
    for offset in range(TOP_K):
        fp_preload[route_scale.address + offset] = 2.5
    moe_constants = NemotronMoeConstants(
        zero_row=zero_row,
        routed_scale=route_scale,
    )

    prompt_value, decode_values = _hidden_inputs()
    prompt_addr = _align(workspace_end, MLEN * MLEN)
    decode_base = prompt_addr + MLEN * HIDDEN
    vram_preload = torch.zeros(
        decode_base + DECODE_TOKENS * BLEN * HIDDEN,
        dtype=torch.bfloat16,
    )
    prompt = prestage_bf16_vram_matrix(
        prog=prog,
        name="NEMOTRON_PROMPT",
        tensor=prompt_value,
        vram_addr=prompt_addr,
        physical_shape=(MLEN, HIDDEN),
        vram_preload=vram_preload,
    )
    decode_inputs = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"NEMOTRON_DECODE_TOKEN_{token}",
            tensor=value,
            vram_addr=decode_base + token * BLEN * HIDDEN,
            physical_shape=(BLEN, HIDDEN),
            vram_preload=vram_preload,
        )
        for token, value in enumerate(decode_values)
    ]
    prefill_fixed_hidden = prog.alloc_at(
        "NEMOTRON_MAMBA_PREFILL_HIDDEN",
        rows=PREFILL_TOKENS,
        cols=HIDDEN,
        vram_addr=prefill_hidden_addr,
        physical_shape=(PREFILL_TOKENS, HIDDEN),
    )
    decode_fixed_hidden = prog.alloc_at(
        "NEMOTRON_MAMBA_DECODE_HIDDEN",
        rows=1,
        cols=HIDDEN,
        vram_addr=decode_hidden_addr,
        physical_shape=(BLEN, HIDDEN),
    )
    checkpoint_rows = len(NEMOTRON3_PATTERN) * TOTAL_TOKENS
    checkpoint = prog.alloc(
        "NEMOTRON_LAYER_CHECKPOINTS",
        rows=checkpoint_rows,
        cols=HIDDEN,
        strict=False,
        physical_shape=(_align(checkpoint_rows, MLEN), HIDDEN),
    )

    tensors = TensorSet(values={}, bf16_names=set())
    mamba_weights, _ = _register_state_weights(
        prog, tensors, prefill_memories
    )
    prog._next_hbm_addr = max(
        prog._next_hbm_addr,
        MambaHbmLayout.build(_state_config(SchedulePhase.PREFILL)).arena_end,
    )
    block_norm_inputs = {
        layer: _register_weight(
            prog,
            tensors,
            f"NEM_L{layer}_BLOCK_NORM",
            torch.ones(PREFILL_TOKENS, HIDDEN, dtype=torch.bfloat16),
            bf16=True,
        )
        for layer in range(len(NEMOTRON3_PATTERN))
    }
    attention_layers = {
        layer: _register_attention_layer(prog, tensors, layer)
        for layer in ATTENTION_LAYERS
    }
    moe_layers = {
        layer: _register_moe_layer(prog, tensors, layer, ordinal)
        for ordinal, layer in enumerate(MOE_LAYERS)
    }
    mamba_states = {
        layer: Mamba2State.zeros(
            Mamba2Shape(
                HIDDEN,
                MAMBA_HEADS,
                MAMBA_HEAD_DIM,
                MAMBA_STATE_DIM,
                MAMBA_GROUPS,
                MAMBA_KERNEL,
            ),
            1,
        )
        for layer in MAMBA_LAYERS
    }
    golden_checkpoint = torch.zeros(
        checkpoint_rows, HIDDEN, dtype=torch.bfloat16
    )
    expected_routes: list[list[int] | None] = [None] * (
        len(MOE_LAYERS) * TOTAL_TOKENS
    )
    precision = _active_precision_settings()

    def run_layers(
        current,
        golden_current: torch.Tensor,
        *,
        rows: int,
        decode_token: int | None,
    ):
        owned_current = False
        token_offset = 0 if decode_token is None else PREFILL_TOKENS + decode_token
        state_hidden = (
            prefill_fixed_hidden if decode_token is None else decode_fixed_hidden
        )
        for layer, symbol in enumerate(NEMOTRON3_PATTERN):
            kind = SYMBOL_TO_LAYER[symbol]
            prog.emit_comment(
                f"FULL_SYNTHETIC_NEMOTRON phase={'prefill' if decode_token is None else 'decode'} "
                f"token={token_offset} layer={layer} kind={kind.value}"
            )
            residual = prog.vram_copy(
                current, name=f"nem_l{layer}_residual_t{token_offset}", num_rows=rows
            )
            normalized = prog.vram_copy(
                current,
                name=f"nem_l{layer}_normalized_t{token_offset}",
                num_rows=rows,
            )
            prog.rms_norm(
                normalized,
                eps_offset=block_eps.address,
                reci_hid_offset=block_reciprocal.address,
            )
            norm_weight = _load_bf16_matrix(
                prog,
                block_norm_inputs[layer],
                f"nem_l{layer}_norm_weight_t{token_offset}",
            )
            prog.vram_mul(normalized, norm_weight, num_rows=rows)
            prog.free_tensor(norm_weight)
            golden_normalized = _rms(golden_current, precision=precision)

            if kind is HybridLayerType.MAMBA:
                prog.vram_copy_region(
                    state_hidden,
                    normalized,
                    num_rows=rows,
                    num_cols=HIDDEN,
                )
                key = (0 if decode_token is None else decode_token, layer)
                bucket = (
                    prefill_buckets[key]
                    if decode_token is None
                    else decode_buckets[key]
                )
                for event in bucket:
                    prog.emit(event.assembly)
                mixer = state_hidden
                golden_mixer, mamba_states[layer] = _mamba_cpu(
                    golden_normalized,
                    mamba_states[layer],
                    mamba_weights,
                    precision=precision,
                )
            elif kind is HybridLayerType.ATTENTION:
                data = attention_layers[layer]
                mixer = emit_nemotron_attention_block(
                    prog,
                    normalized,
                    shape=NemotronAttentionShape(
                        hidden=HIDDEN,
                        query_heads=QUERY_HEADS,
                        kv_heads=KV_HEADS,
                        head_dim=HEAD_DIM,
                    ),
                    weights=data.weights,
                    rows=rows,
                    name=f"nem_l{layer}_gqa_t{token_offset}",
                    cache=data.cache,
                    token_index=token_offset,
                    causal=decode_token is None,
                )
                golden_mixer = _attention_cpu(golden_normalized, data)
            else:
                data = moe_layers[layer]
                correction = _load_bf16_matrix(
                    prog,
                    data.correction_input,
                    f"nem_l{layer}_correction_t{token_offset}",
                )
                route_base = data.ordinal * TOTAL_TOKENS * TOP_K + token_offset * TOP_K
                mixer = emit_nemotron_moe_block(
                    prog,
                    normalized,
                    shape=NemotronMoeShape(
                        hidden=HIDDEN,
                        intermediate=HIDDEN,
                        shared_intermediate=HIDDEN,
                        num_experts=EXPERTS,
                        top_k=TOP_K,
                    ),
                    weights=data.weights,
                    correction_bias=correction,
                    constants=moe_constants,
                    rows=rows,
                    int_sram_base=route_base,
                    name=f"nem_l{layer}_moe_t{token_offset}",
                )
                prog.free_tensor(correction)
                golden_mixer, routes = _moe_cpu(
                    golden_normalized, data, precision=precision
                )
                for local_token, route in enumerate(routes):
                    expected_routes[
                        data.ordinal * TOTAL_TOKENS + token_offset + local_token
                    ] = route

            prog.vram_add(residual, mixer, num_rows=rows)
            golden_next = _bf16(
                golden_current.float() + golden_mixer.float(),
                precision=precision,
            )
            prog.vram_copy_region(
                checkpoint,
                residual,
                num_rows=rows,
                num_cols=HIDDEN,
                dst_row_offset=layer * TOTAL_TOKENS + token_offset,
            )
            golden_checkpoint[
                layer * TOTAL_TOKENS
                + token_offset : layer * TOTAL_TOKENS
                + token_offset
                + rows
            ] = golden_next
            prog.free_tensor(normalized)
            if (
                mixer is not prefill_fixed_hidden
                and mixer is not decode_fixed_hidden
            ):
                prog.free_tensor(mixer)
            if owned_current:
                prog.free_tensor(current)
            current = residual
            golden_current = golden_next
            owned_current = True
        return current, golden_current

    prefill_output, _ = run_layers(
        prompt,
        prompt_value,
        rows=PREFILL_TOKENS,
        decode_token=None,
    )
    prog.free_tensor(prefill_output)
    for token, (decode_input, decode_value) in enumerate(
        zip(decode_inputs, decode_values, strict=True)
    ):
        decode_output, _ = run_layers(
            decode_input,
            decode_value,
            rows=1,
            decode_token=token,
        )
        prog.free_tensor(decode_output)

    cache_readbacks = []
    for layer, data in attention_layers.items():
        histories = (*data.key_history, *data.value_history)
        if any(len(history) != TOTAL_TOKENS for history in histories):
            raise AssertionError(
                f"layer {layer} CPU GQA history did not reach {TOTAL_TOKENS} rows"
            )
        for is_key, caches in ((True, data.cache.keys), (False, data.cache.values)):
            for head, cache in enumerate(caches):
                cache_readbacks.append(
                    (
                        layer,
                        is_key,
                        head,
                        cache,
                        prog.load_batch(
                            cache.prefix(TOTAL_TOKENS),
                            name=f"{cache.backing.name}_final_readback",
                            storage_precision=2,
                            hbm_precision=1,
                        ),
                    )
                )

    assembly = prog.compile()
    for _layer, _is_key, _head, cache, _readback in cache_readbacks:
        marker = f"DECODE_CACHE_APPEND {cache.backing.name} token="
        written_tokens = [
            int(line.split("token=", 1)[1].split()[0])
            for line in assembly.splitlines()
            if marker in line
        ]
        if written_tokens != list(range(TOTAL_TOKENS)):
            raise AssertionError(
                f"{cache.backing.name} write lifetime mismatch: {written_tokens}"
            )
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
        asm="nemotron3_full_synthetic_s16_decode4",
        specified_data_order=sorted(tensors.values, key=hbm_addrs.__getitem__),
        build_path=build_dir,
        input_tensors=tensors.values,
        tensor_layouts=layouts,
        hbm_addrs=hbm_addrs,
    )

    descriptors = [
        event.descriptor
        for event in prefill_trace.events
        if event.descriptor is not None
    ]
    parameter_writes: dict[int, torch.Tensor] = {}
    for descriptor in descriptors:
        payload = descriptor.payload
        parameter_writes.update(
            {
                payload.conv_weight_addr: mamba_weights["CONV_WEIGHT"],
                payload.conv_bias_addr: mamba_weights["CONV_BIAS"],
                payload.a_log_addr: mamba_weights["A_LOG"],
                payload.dt_bias_addr: mamba_weights["DT_BIAS"],
                payload.d_skip_addr: mamba_weights["D_SKIP"],
            }
        )
    hbm_size = _align(
        max(
            prog._next_hbm_addr,
            MambaHbmLayout.build(_state_config(SchedulePhase.PREFILL)).arena_end,
            decode_lowered.layout_descriptor_base
            + len(decode_lowered.layout_descriptor_image),
        ),
        64,
    )
    _patch_hbm(
        build_dir / "hbm_for_behave_sim.bin",
        descriptor_image=prefill_lowered.descriptor_image,
        descriptor_base=prefill_lowered.descriptor_base,
        layout_descriptor_image=prefill_lowered.layout_descriptor_image,
        layout_descriptor_base=prefill_lowered.layout_descriptor_base,
        writes=parameter_writes,
        minimum_size=hbm_size,
    )
    _patch_hbm(
        build_dir / "hbm_for_behave_sim.bin",
        descriptor_image=decode_lowered.descriptor_image,
        descriptor_base=decode_lowered.descriptor_base,
        layout_descriptor_image=decode_lowered.layout_descriptor_image,
        layout_descriptor_base=decode_lowered.layout_descriptor_base,
        writes=parameter_writes,
        minimum_size=hbm_size,
    )
    params = _comparison_params_for(
        checkpoint,
        rows=checkpoint_rows,
        hidden=HIDDEN,
        mlen=MLEN,
        golden=golden_checkpoint,
    )
    params.update(
        {"atol": 0.08, "rtol": 0.06, "min_allclose_match_rate": 100.0}
    )
    (build_dir / "comparison_params.json").write_text(
        json.dumps(params, indent=2) + "\n"
    )
    (build_dir / "generated_asm_code.asm").write_text(assembly)
    (build_dir / "hbm_size.txt").write_text(f"{hbm_size}\n")
    state_profile_path = build_dir / "state_profile.json"
    metrics = run_emulator(
        build_dir,
        hbm_size=hbm_size,
        stage_profile=True,
        state_profile_out=state_profile_path,
        dump_cwd=build_dir,
    )
    results, _ = compare_emulator_output(build_dir, verbose=False)
    if float(results.get("allclose_match_rate", 0.0)) < 100.0:
        raise AssertionError(f"Nemotron full synthetic hidden mismatch: {results}")

    state_profile = json.loads(state_profile_path.read_text())
    state_commands = [
        command
        for command in state_profile["commands"]
        if command["algorithm"] == "mamba2"
    ]
    subop_counts = {
        subop: sum(command["subop"] == subop for command in state_commands)
        for subop in ("reset", "prefill", "step")
    }
    expected_subops = {
        "reset": len(MAMBA_LAYERS),
        "prefill": len(MAMBA_LAYERS),
        "step": len(MAMBA_LAYERS) * DECODE_TOKENS,
    }
    if subop_counts != expected_subops:
        raise AssertionError(
            f"Mamba lifecycle mismatch: expected={expected_subops}, actual={subop_counts}"
        )
    for layer in MAMBA_LAYERS:
        layer_commands = [
            command for command in state_commands if command["layer_id"] == layer
        ]
        if [
            command["subop"]
            for command in layer_commands
            if command["subop"] in {"reset", "prefill", "step"}
        ] != ["reset", "prefill", "step", "step", "step", "step"]:
            raise AssertionError(f"layer {layer} state lifetime is out of order")

    actual_routes = np.fromfile(
        build_dir / "intsram_dump.bin",
        dtype="<i4",
        count=len(MOE_LAYERS) * TOTAL_TOKENS * TOP_K,
    ).reshape(len(MOE_LAYERS) * TOTAL_TOKENS, TOP_K)
    if any(route is None for route in expected_routes):
        raise AssertionError("CPU route log has unwritten entries")
    expected_route_array = np.asarray(expected_routes, dtype=np.int32)
    if not np.array_equal(
        np.sort(actual_routes, axis=1), np.sort(expected_route_array, axis=1)
    ):
        mismatch = np.argwhere(
            np.any(
                np.sort(actual_routes, axis=1)
                != np.sort(expected_route_array, axis=1),
                axis=1,
            )
        )[0, 0]
        raise AssertionError(
            f"MoE route mismatch at entry {mismatch}: "
            f"actual={actual_routes[mismatch].tolist()}, "
            f"expected={expected_route_array[mismatch].tolist()}"
        )

    cache_errors = {}
    vram_dump = build_dir / "vram_dump.bin"
    for layer, is_key, head, cache, readback in cache_readbacks:
        data = attention_layers[layer]
        expected = torch.cat(
            data.key_history[head] if is_key else data.value_history[head],
            dim=0,
        )
        actual = read_bf16_vram_matrix(
            vram_dump,
            address=prog.get_vram_addr(readback.name),
            rows=TOTAL_TOKENS,
            width=HEAD_DIM,
            physical_rows=readback.physical_shape[0],
            mlen=MLEN,
        )
        error = float((actual.float() - expected.float()).abs().max())
        cache_errors[cache.backing.name] = error
        if not torch.allclose(actual, expected, atol=0.01, rtol=0.02):
            raise AssertionError(
                f"GQA cache mismatch for {cache.backing.name}: max_abs={error}"
            )

    summary = {
        "model": "nemotron3",
        "scope": "full_52_layer_compact_synthetic_transactional",
        "prefill_tokens": PREFILL_TOKENS,
        "decode_tokens": DECODE_TOKENS,
        "layer_counts": {
            "mamba": len(MAMBA_LAYERS),
            "moe": len(MOE_LAYERS),
            "attention": len(ATTENTION_LAYERS),
        },
        "checkpoint_rows": checkpoint_rows,
        "state_lifecycle": subop_counts,
        "gqa_cache_rows_per_layer": TOTAL_TOKENS,
        "gqa_cache_max_abs_error": max(cache_errors.values(), default=0.0),
        "route_entries": int(expected_route_array.size),
        "route_patterns": len({tuple(row) for row in expected_route_array.tolist()}),
        "sim_cycles": metrics.get("sim_latency_cycles"),
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "max_abs_error": results.get("max_error"),
        "allclose_match_rate": results.get("allclose_match_rate"),
        "hbm_bytes": hbm_size,
        "instruction_count": sum(
            1
            for line in assembly.splitlines()
            if line.strip() and not line.lstrip().startswith(";")
        ),
        "claim_boundary": (
            "compact synthetic execution proof; cycles are not real-shape "
            "Nemotron performance"
        ),
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(
            "transactional_emulator/testbench/build/"
            "nemotron3_full_synthetic_connected"
        ),
    )
    args = parser.parse_args()
    build_and_run(args.build_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
