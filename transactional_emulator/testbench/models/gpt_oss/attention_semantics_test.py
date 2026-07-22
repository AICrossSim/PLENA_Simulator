"""GPT-OSS attention semantics smoke tests for PLENA.

This file intentionally tests the new attention building blocks before a full
block harness exists:

* ``projection`` verifies the BF16 HBM_M_KV matrix path used for Q/K/V/O
  projections.  It uses H_PREFETCH_M precision=KeyValue and emits no
  C_SET_SCALE_REG for the projection.
* ``core`` verifies packed GQA causal attention with optional GPT-OSS sink and
  sliding-window masks.  K/V are stored as Plain BF16 HBM_M_KV tensors.
* ``full`` verifies BF16 Q/K/V projections, BF16 K/V staging through
  H_STORE_V/H_PREFETCH_M KeyValue, packed attention, and BF16 O projection in
  one emulator program.
* ``true_core`` verifies the true GPT-OSS GQA shape (64 Q heads / 8 KV heads)
  with per-Q-head sinks by unrolling one Q head per packed-attention call.  Q/K/V
  are prestaged after projection+bias+RoPE, so this mode validates the attention
  core shape and sink semantics without claiming runtime RoPE lowering is done.
* ``runtime_rope`` verifies the missing bridge: BF16 projection + BF16 bias
  followed by device-side rotate-half projection and RoPE.
* ``true_full`` verifies the true-shape attention path with runtime BF16
  Q/K/V/O projection, BF16 bias, device-side RoPE, per-head sink, and packed
  64Q/8KV attention.
* ``true_decoder_block`` extends ``true_attn_block`` with the second pre-norm
  MoE sublayer: VRAM-resident post-attention RMSNorm feeds device top-k,
  dynamic true-expert weight selection, expert execution, scatter-combine, and
  the second residual add in one emulator program.
* ``true_decoder_chain`` is a sequential validation runner that invokes
  ``true_decoder_block`` layer by layer, handing emulator output to the next
  layer and running a host-chain control for cumulative drift diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import tomlkit
import torch
import torch.nn.functional as F
from safetensors import safe_open

from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.aten.golden import (
    _active_precision_settings,
    _rms_norm_vector_ref,
    quantize_to_vector_fp,
)
from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_and_assert, run_emulator
from transactional_emulator.testbench.layout_utils import infer_hbm_tensor_layouts, prestage_bf16_vram_matrix
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.routed_moe.gpt_oss_moe_gather_scatter_test import (
    _build_true_expert_bias_table,
    _build_true_expert_weight_table,
    _decode_bf16_dump,
    _decode_u32_dump,
    _device_routing_vram_exact_golden,
    _linear_projection_golden,
)
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.models.gpt_oss.attention_semantics_helpers import (
    _align_to_tile,
    _bias_parts,
    _comparison_params,
    _make_packed_rope_inputs,
    _make_rotate_half_matrix,
    _rel_rms,
    _resolve_sliding_window,
    _router_bias_block_rows,
    _router_margin_summary,
    _strict_tail_summary,
    _topk_match_summary,
    _write_json,
)

_DEFAULT_QWEN3_REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def _default_qwen3_snapshot() -> Path:
    override = os.environ.get("QWEN3_30B_A3B_SNAPSHOT")
    if override:
        return Path(override).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "hub" / "models--Qwen--Qwen3-30B-A3B" / "snapshots" / _DEFAULT_QWEN3_REVISION


_PRECISION_SETTINGS_CACHE = {}


def _cached_active_precision_settings():
    """Cache precision settings for hot Python goldens.

    Qwen MoE goldens call the BF16 VRAM quantizer many times.  Reading and
    parsing the settings TOML on every call dominates runtime without changing
    semantics, so cache per settings file and invalidate when this test mutates
    the settings in-place.
    """
    settings_key = os.environ.get("PLENA_SETTINGS_TOML", "")
    if settings_key not in _PRECISION_SETTINGS_CACHE:
        _PRECISION_SETTINGS_CACHE[settings_key] = _active_precision_settings()
    return _PRECISION_SETTINGS_CACHE[settings_key]


def _set_matrix_kv_plain_bf16() -> None:
    settings = Path(os.environ["PLENA_SETTINGS_TOML"])
    with settings.open() as f:
        config = tomlkit.load(f)
    for mode in ("TRANSACTIONAL", "ANALYTIC"):
        precision = config[mode]["PRECISION"]
        for key in ("HBM_M_KV_TYPE", "HBM_V_KV_TYPE"):
            precision[key] = tomlkit.table()
            precision[key]["format"] = "Plain"
            precision[key]["DATA_TYPE"] = tomlkit.table()
            precision[key]["DATA_TYPE"]["type"] = "Fp"
            precision[key]["DATA_TYPE"]["sign"] = True
            precision[key]["DATA_TYPE"]["exponent"] = 8
            precision[key]["DATA_TYPE"]["mantissa"] = 7
    with settings.open("w") as f:
        tomlkit.dump(config, f)
    _PRECISION_SETTINGS_CACHE.pop(str(settings), None)
    _PRECISION_SETTINGS_CACHE.pop(os.environ.get("PLENA_SETTINGS_TOML", ""), None)


def _bf16_vram(x: torch.Tensor) -> torch.Tensor:
    precision = _cached_active_precision_settings()
    return quantize_to_vector_fp(x.to(torch.bfloat16).float(), precision)


def _weighted_rms_norm_golden(x: torch.Tensor, weight_rows: torch.Tensor, eps: float) -> torch.Tensor:
    precision = _cached_active_precision_settings()
    x_vram = _bf16_vram(x)
    w_vram = _bf16_vram(weight_rows)
    normed = _rms_norm_vector_ref(x_vram, eps, precision)
    return quantize_to_vector_fp(normed.float() * w_vram.float(), precision)


def _head_rms_norm_golden(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last/head dimension for Qwen-style q_norm/k_norm."""
    precision = _cached_active_precision_settings()
    x_vram = _bf16_vram(x)
    w_vram = _bf16_vram(weight)
    normed = _rms_norm_vector_ref(x_vram, eps, precision)
    return quantize_to_vector_fp(normed.float() * w_vram.float(), precision)


def _standard_swiglu_vram_golden(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """VRAM-style standard SwiGLU: silu(gate) * up."""
    sigmoid = _bf16_vram(gate)
    sigmoid = _bf16_vram(sigmoid.float() * -1.0)
    sigmoid = _bf16_vram(torch.exp(torch.clamp(sigmoid.float(), -88.0, 88.0)))
    sigmoid = _bf16_vram(sigmoid.float() + 1.0)
    sigmoid = _bf16_vram(torch.reciprocal(sigmoid.float()))
    glu = _bf16_vram(gate.float() * sigmoid.float())
    return _bf16_vram(up.float() * glu.float())


def _device_routing_vram_policy_golden(
    *,
    x: torch.Tensor,
    device_indices: torch.Tensor,
    device_weights: torch.Tensor,
    split,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
    rows: int,
    hidden: int,
    blen: int,
    mlen: int,
    activation_policy: str,
    pair_count: int | None = None,
) -> tuple[torch.Tensor, dict | None]:
    """Decoder-block MoE golden with policy-specific expert activation."""
    if activation_policy == "gpt_oss_clamp_gated":
        if down_bias is None:
            raise ValueError("GPT-OSS clamp-gated golden requires down_bias")
        return _device_routing_vram_exact_golden(
            x=x,
            device_indices=device_indices,
            device_weights=device_weights,
            split=split,
            down_weight=down_weight,
            down_bias=down_bias,
            rows=rows,
            hidden=hidden,
            blen=blen,
            mlen=mlen,
            pair_count=pair_count,
        )
    if activation_policy != "standard_swiglu":
        raise NotImplementedError(f"unsupported decoder MoE activation_policy={activation_policy!r}")

    exact_golden = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    top_k = device_indices.shape[1]
    total_pairs = rows * top_k if pair_count is None else pair_count
    for pair_idx in range(total_pairs):
        token_idx = pair_idx // top_k
        topk_pos = pair_idx % top_k
        expert_id = int(device_indices[token_idx, topk_pos].item())
        weight = device_weights[token_idx, topk_pos].to(torch.bfloat16)
        x_slots = torch.zeros(blen, hidden, dtype=torch.bfloat16)
        x_slots[0:1] = x[token_idx : token_idx + 1].to(torch.bfloat16)

        gate = _linear_projection_golden(
            x_slots,
            split.gate_weight[expert_id],
            mlen=mlen,
            hbm_input=False,
        )
        up = _linear_projection_golden(
            x_slots,
            split.up_weight[expert_id],
            mlen=mlen,
            hbm_input=False,
        )
        if getattr(split, "gate_bias", None) is not None:
            gate = _bf16_vram(gate.float() + split.gate_bias[expert_id].reshape(1, -1).float())
        if getattr(split, "up_bias", None) is not None:
            up = _bf16_vram(up.float() + split.up_bias[expert_id].reshape(1, -1).float())
        hidden_slots = _standard_swiglu_vram_golden(gate, up)
        out = _linear_projection_golden(hidden_slots, down_weight[expert_id], mlen=mlen, hbm_input=False)
        if down_bias is not None:
            out = _bf16_vram(out.float() + down_bias[expert_id].reshape(1, -1).float())
        exact_golden[token_idx] = _bf16_vram(
            exact_golden[token_idx].float() + _bf16_vram(out[0].float() * weight.float()).float()
        )
    return exact_golden, None


def _load_qwen3_layer_tensors(snapshot: Path, layer_idx: int) -> dict[str, torch.Tensor | str]:
    """Load true Qwen3-MoE layer tensors in the shapes used by this harness.

    HF stores linear weights as [out_features, in_features].  The PLENA BF16
    projection helpers used here expect [in_features, out_features], matching
    the synthetic tensors above and the Qwen substrate scripts.
    """

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
    q_w = tensor(f"{prefix}.self_attn.q_proj.weight").T.contiguous()
    k_w = tensor(f"{prefix}.self_attn.k_proj.weight").T.contiguous()
    v_w = tensor(f"{prefix}.self_attn.v_proj.weight").T.contiguous()
    o_w = tensor(f"{prefix}.self_attn.o_proj.weight").T.contiguous()
    q_norm = tensor(f"{prefix}.self_attn.q_norm.weight")
    k_norm = tensor(f"{prefix}.self_attn.k_norm.weight")
    input_norm = tensor(f"{prefix}.input_layernorm.weight")
    post_norm = tensor(f"{prefix}.post_attention_layernorm.weight")
    router_w = tensor(f"{prefix}.mlp.gate.weight")

    num_experts = int(router_w.shape[0])
    gate_weights = []
    up_weights = []
    down_weights = []
    for expert_id in range(num_experts):
        gate_weights.append(tensor(f"{prefix}.mlp.experts.{expert_id}.gate_proj.weight").T.contiguous())
        up_weights.append(tensor(f"{prefix}.mlp.experts.{expert_id}.up_proj.weight").T.contiguous())
        down_weights.append(tensor(f"{prefix}.mlp.experts.{expert_id}.down_proj.weight").T.contiguous())
    gate_w = torch.stack(gate_weights, dim=0)
    up_w = torch.stack(up_weights, dim=0)
    down_w = torch.stack(down_weights, dim=0)

    hidden = int(router_w.shape[1])
    intermediate = int(gate_w.shape[2])
    return {
        "snapshot": str(snapshot),
        "layer_idx": layer_idx,
        "w_q": q_w,
        "w_k": k_w,
        "w_v": v_w,
        "w_o": o_w,
        "q_norm_weight": q_norm,
        "k_norm_weight": k_norm,
        "norm_weight": input_norm,
        "ffn_norm_weight": post_norm,
        "router_weight": router_w,
        "router_bias": torch.zeros(num_experts, dtype=torch.bfloat16),
        "gate_weight": gate_w,
        "up_weight": up_w,
        "down_weight": down_w,
        "gate_bias": torch.zeros(num_experts, intermediate, dtype=torch.bfloat16),
        "up_bias": torch.zeros(num_experts, intermediate, dtype=torch.bfloat16),
        "down_bias": torch.zeros(num_experts, hidden, dtype=torch.bfloat16),
    }


def _residual_add_golden(hidden: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
    precision = _cached_active_precision_settings()
    hidden_vram = _bf16_vram(hidden)
    sublayer_vram = _bf16_vram(sublayer)
    return quantize_to_vector_fp(hidden_vram.float() + sublayer_vram.float(), precision)


def _read_vram_matrix(
    path: Path,
    *,
    vram_addr: int,
    rows: int,
    cols: int,
    mlen: int,
    physical_rows: int,
) -> torch.Tensor:
    """Decode a BF16 VRAM matrix from column-block-major storage."""
    raw = np.fromfile(path, dtype="<u2")
    flat = torch.tensor(raw.astype(np.uint16), dtype=torch.uint16).view(torch.bfloat16)
    out = torch.zeros(rows, cols, dtype=torch.bfloat16)
    for col_block in range(math.ceil(cols / mlen)):
        col_start = col_block * mlen
        col_end = min(col_start + mlen, cols)
        width = col_end - col_start
        for row in range(rows):
            src = vram_addr + (col_block * physical_rows + row) * mlen
            out[row, col_start:col_end] = flat[src : src + width]
    return out


def _routing_boundary_verdict(
    *,
    reference_logits: torch.Tensor,
    comparison_logits: torch.Tensor,
    reference_indices: torch.Tensor,
    device_indices: torch.Tensor,
    top_k: int,
) -> dict:
    """Classify top-k mismatches as stable-token failures or boundary cases."""
    reference_margin = _router_margin_summary(reference_logits, top_k)
    comparison_margin = _router_margin_summary(comparison_logits, top_k)
    logit_error = (comparison_logits.float() - reference_logits.float()).abs()
    threshold = float(logit_error.max().item()) if logit_error.numel() else 0.0

    unsafe_tokens = []
    set_mismatch_tokens = []
    order_mismatch_tokens = []
    safe_set_mismatch_tokens = []
    safe_order_mismatch_tokens = []
    per_token = []
    for token_idx in range(reference_indices.shape[0]):
        ref_order = [int(v) for v in reference_indices[token_idx].tolist()]
        dev_order = [int(v) for v in device_indices[token_idx].tolist()]
        ref_gap = float(reference_margin["per_token"][token_idx]["rank_k_to_next_gap"])
        cmp_gap = float(comparison_margin["per_token"][token_idx]["rank_k_to_next_gap"])
        gap_floor = min(ref_gap, cmp_gap)
        unsafe = gap_floor <= threshold
        set_match = set(ref_order) == set(dev_order)
        order_match = ref_order == dev_order
        if unsafe:
            unsafe_tokens.append(token_idx)
        if not set_match:
            set_mismatch_tokens.append(token_idx)
            if not unsafe:
                safe_set_mismatch_tokens.append(token_idx)
        if not order_match:
            order_mismatch_tokens.append(token_idx)
            if not unsafe:
                safe_order_mismatch_tokens.append(token_idx)
        per_token.append(
            {
                "token_index": int(token_idx),
                "reference_gap": ref_gap,
                "comparison_gap": cmp_gap,
                "gap_floor": gap_floor,
                "unsafe": bool(unsafe),
                "set_match": bool(set_match),
                "order_match": bool(order_match),
                "reference_indices": ref_order,
                "device_indices": dev_order,
            }
        )

    if safe_set_mismatch_tokens:
        status = "fail_safe_set_mismatch"
    elif set_mismatch_tokens:
        status = "boundary_explained_set_mismatch"
    elif unsafe_tokens:
        status = "pass_with_weak_margins"
    elif safe_order_mismatch_tokens:
        status = "pass_set_match_safe_order_mismatch"
    else:
        status = "pass"

    return {
        "status": status,
        "top_k": int(top_k),
        "logit_error_max_abs": threshold,
        "unsafe_threshold": threshold,
        "unsafe_tokens": [int(v) for v in unsafe_tokens],
        "set_mismatch_tokens": [int(v) for v in set_mismatch_tokens],
        "order_mismatch_tokens": [int(v) for v in order_mismatch_tokens],
        "safe_set_mismatch_tokens": [int(v) for v in safe_set_mismatch_tokens],
        "safe_order_mismatch_tokens": [int(v) for v in safe_order_mismatch_tokens],
        "safe_token_count": int(reference_indices.shape[0] - len(unsafe_tokens)),
        "num_tokens": int(reference_indices.shape[0]),
        "reference_min_gap": reference_margin["min_rank_k_to_next_gap"],
        "comparison_min_gap": comparison_margin["min_rank_k_to_next_gap"],
        "per_token": per_token,
    }


def run_projection(args: argparse.Namespace) -> dict:
    build_dir = (args.build_dir / "projection").expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    rows = args.seq_len or 8
    hidden = args.hidden_size or 128
    out_features = args.out_features or 64
    x_override = _load_hidden_tensor(args.input_pt).to(torch.bfloat16) if args.input_pt is not None else None
    w_override = (
        _load_hidden_tensor(args.projection_weight_pt).to(torch.bfloat16)
        if args.projection_weight_pt is not None
        else None
    )
    golden_override = _load_hidden_tensor(args.golden_pt).to(torch.bfloat16) if args.golden_pt is not None else None
    if x_override is not None:
        if x_override.ndim != 2:
            raise ValueError(f"--input-pt for projection must be rank-2, got shape {tuple(x_override.shape)}")
        rows, hidden = map(int, x_override.shape)
    if w_override is not None:
        if w_override.ndim != 2:
            raise ValueError(
                f"--projection-weight-pt for projection must be rank-2, got shape {tuple(w_override.shape)}"
            )
        if int(w_override.shape[0]) == hidden:
            out_features = int(w_override.shape[1])
        elif int(w_override.shape[1]) == hidden:
            # HF Linear weights are [out_features, in_features]; PLENA projection
            # helpers consume [in_features, out_features].
            w_override = w_override.t().contiguous()
            out_features = int(w_override.shape[1])
        else:
            raise ValueError(
                f"--projection-weight-pt shape {tuple(w_override.shape)} incompatible with hidden={hidden}"
            )
    if rows % blen != 0:
        raise ValueError(f"rows={rows} must be a multiple of BLEN={blen}")
    padded_projection = bool(args.allow_padded_projection)
    if not padded_projection and (hidden % mlen != 0 or out_features % mlen != 0):
        raise ValueError("hidden and out_features must be multiples of MLEN for this smoke")
    physical_hidden = math.ceil(hidden / mlen) * mlen
    physical_out_features = math.ceil(out_features / mlen) * mlen

    torch.manual_seed(args.seed)
    x_logical = x_override if x_override is not None else (torch.randn(rows, hidden) * 0.2).to(torch.bfloat16)
    w_logical = w_override if w_override is not None else (torch.randn(hidden, out_features) * 0.2).to(torch.bfloat16)
    b = (torch.randn(out_features) * 0.02).to(torch.bfloat16) if args.projection_bias else None
    golden_logical = torch.matmul(x_logical.float(), w_logical.float()).to(torch.bfloat16)
    if args.projection_bias:
        golden_logical = (golden_logical.float() + b.float()).to(torch.bfloat16)
    if golden_override is not None:
        if tuple(golden_override.shape) != (rows, out_features):
            raise ValueError(
                f"--golden-pt for projection shape {tuple(golden_override.shape)} != {(rows, out_features)}"
            )
        golden_logical = golden_override
    x = torch.zeros(rows, physical_hidden, dtype=torch.bfloat16)
    x[:, :hidden] = x_logical
    w = torch.zeros(physical_hidden, physical_out_features, dtype=torch.bfloat16)
    w[:hidden, :out_features] = w_logical
    golden = torch.zeros(rows, physical_out_features, dtype=torch.bfloat16)
    golden[:, :out_features] = golden_logical

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    physical_rows = max(mlen, math.ceil(rows / blen) * blen)
    x_region = physical_rows * physical_hidden
    bias_base = math.ceil(x_region / (mlen * mlen)) * (mlen * mlen)
    bias_shape = (physical_rows, physical_out_features)
    bias_size = math.ceil((bias_shape[0] * bias_shape[1]) / (mlen * mlen)) * (mlen * mlen)
    vram_preload = torch.zeros(
        max(x_region + mlen * mlen, bias_base + bias_size if args.projection_bias else 0),
        dtype=torch.bfloat16,
    )
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="X",
        tensor=x,
        vram_addr=0,
        physical_shape=(physical_rows, physical_hidden),
        vram_preload=vram_preload,
    )
    bias_vram = None
    if args.projection_bias:
        bias_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="B_PROJ",
            tensor=torch.nn.functional.pad(b.reshape(1, -1).repeat(rows, 1), (0, physical_out_features - out_features)),
            vram_addr=bias_base,
            physical_shape=bias_shape,
            vram_preload=vram_preload,
        )
    w_input = prog.input("W_QKV_BF16", shape=(physical_hidden, physical_out_features), real_data_ratio=2.0)
    out = prog.linear_projection_bf16(
        x_vram,
        w_input,
        name="bf16_projection",
        physical_shape=(physical_rows, physical_out_features),
    )
    if args.projection_bias:
        prog.vram_add(out, bias_vram, num_rows=rows)
    isa = prog.compile()
    if "C_SET_SCALE_REG" in isa:
        raise AssertionError("BF16 projection emitted C_SET_SCALE_REG")
    if "H_PREFETCH_M" not in isa or ", 1, 1" not in isa:
        raise AssertionError("BF16 projection did not use H_PREFETCH_M precision=KeyValue")

    input_tensors = {"W_QKV_BF16": w}
    tensor_layouts = {
        "W_QKV_BF16": {
            "source_shape": [physical_hidden, physical_out_features],
            "storage_shape": [physical_hidden, physical_out_features],
            "source_rows": physical_hidden,
            "storage_rows": physical_hidden,
            "source_row_elements": physical_out_features,
            "storage_row_elements": physical_out_features,
            "precision": "HBM_M_KV_TYPE",
        }
    }
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_attention_projection_bf16",
        specified_data_order=list(input_tensors),
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    params = _comparison_params(
        prog._compiler.get_vram_addr(out.name),
        rows,
        physical_out_features,
        mlen,
        physical_rows=out.physical_shape[0],
    )
    _write_json(build_dir / "comparison_params.json", params)
    (build_dir / "generated_asm_code.asm").write_text(isa)

    if not args.no_run:
        run_and_assert(build_dir, "gpt_oss_attention_projection_bf16", mlen=mlen, blen=blen)
        results, _ = compare_emulator_output(build_dir)
        emu = results["simulated_values"].reshape(rows, physical_out_features).to(torch.bfloat16)
        simulated_output_pt = build_dir / "simulated_output.pt"
        torch.save(emu, simulated_output_pt)
        rel = _rel_rms(emu, golden)
        if rel > 0.01:
            raise AssertionError(f"BF16 projection rel_rms={rel:.6g} exceeds 1%")
    else:
        rel = float("nan")
        simulated_output_pt = None

    summary = {
        "mode": "projection",
        "rows": rows,
        "hidden": hidden,
        "out_features": out_features,
        "physical_hidden": physical_hidden,
        "physical_out_features": physical_out_features,
        "padded_projection": padded_projection,
        "scale_reg_count": isa.count("C_SET_SCALE_REG"),
        "h_prefetch_keyvalue_count": isa.count("H_PREFETCH_M"),
        "projection_bias": bool(args.projection_bias),
        "real_input": args.input_pt is not None,
        "real_weight": args.projection_weight_pt is not None,
        "external_golden": args.golden_pt is not None,
        "rel_rms": rel,
        "simulated_output_pt": str(simulated_output_pt) if simulated_output_pt is not None else None,
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _projection_weight_in_out(weight: torch.Tensor, in_features: int, *, label: str) -> tuple[torch.Tensor, int]:
    """Return a BF16 projection weight in PLENA [in,out] orientation."""
    if weight.ndim != 2:
        raise ValueError(f"{label} must be rank-2, got shape {tuple(weight.shape)}")
    if int(weight.shape[0]) == in_features:
        return weight.contiguous(), int(weight.shape[1])
    if int(weight.shape[1]) == in_features:
        return weight.t().contiguous(), int(weight.shape[0])
    raise ValueError(f"{label} shape {tuple(weight.shape)} incompatible with in_features={in_features}")


def run_deepseek_mla_q_path(args: argparse.Namespace) -> dict:
    """Integrated DeepSeek MLA frontend path: hidden -> proj -> RMSNorm -> proj.

    The separate real-input gates already cover the individual projections and
    RMSNorms.  This mode validates the VRAM handoff between those stages inside
    one device program.  It covers both the Q path
    (hidden -> q_a -> q_a_norm -> q_b) and the KV path
    (hidden -> compressed_kv -> kv_lora_norm -> kv_b), where the norm only sees
    the leading kv_lora lanes of the wider compressed_kv projection.
    """
    required = {
        "--input-pt": args.input_pt,
        "--projection-weight-pt": args.projection_weight_pt,
        "--norm-weight-pt": args.norm_weight_pt,
        "--second-projection-weight-pt": args.second_projection_weight_pt,
        "--golden-pt": args.golden_pt,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"deepseek_mla_q_path requires {', '.join(missing)}")

    build_dir = (args.build_dir / args.mode).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    x_logical = _load_hidden_tensor(args.input_pt).to(torch.bfloat16).contiguous()
    if x_logical.ndim != 2:
        raise ValueError(f"--input-pt must be rank-2, got shape {tuple(x_logical.shape)}")
    rows, hidden = map(int, x_logical.shape)
    if rows % blen != 0:
        raise ValueError(f"rows={rows} must be a multiple of BLEN={blen}")

    w1_raw = _load_hidden_tensor(args.projection_weight_pt).to(torch.bfloat16)
    w1_logical, first_width = _projection_weight_in_out(w1_raw, hidden, label="--projection-weight-pt")
    norm_weight = _load_hidden_tensor(args.norm_weight_pt).to(torch.bfloat16).reshape(-1).contiguous()
    norm_features = int(norm_weight.numel())
    if norm_features > first_width:
        raise ValueError(f"--norm-weight-pt length {norm_features} cannot exceed first projection width={first_width}")
    split_first_projection = norm_features < first_width
    if split_first_projection:
        # DeepSeek's packed kv_a projection is [kv_lora, k_rope].  RMSNorm is
        # defined only on kv_lora, while the k_rope tail is nonzero.  A VRAM view
        # over the packed row would let the hardware norm consume the tail lanes
        # because normalization walks the physical row.  Split the lora branch
        # here; k_rope is validated separately by the partial-RoPE/core gates.
        w1_logical = w1_logical[:, :norm_features].contiguous()
        first_width = norm_features
    w2_raw = _load_hidden_tensor(args.second_projection_weight_pt).to(torch.bfloat16)
    w2_logical, q_b_width = _projection_weight_in_out(
        w2_raw,
        norm_features,
        label="--second-projection-weight-pt",
    )
    golden_logical = _load_hidden_tensor(args.golden_pt).to(torch.bfloat16).contiguous()
    if tuple(golden_logical.shape) != (rows, q_b_width):
        raise ValueError(f"--golden-pt shape {tuple(golden_logical.shape)} != {(rows, q_b_width)}")

    physical_rows = max(mlen, math.ceil(rows / blen) * blen)
    physical_hidden = math.ceil(hidden / mlen) * mlen
    physical_first = math.ceil(first_width / mlen) * mlen
    physical_q_b = math.ceil(q_b_width / mlen) * mlen

    x = torch.zeros(rows, physical_hidden, dtype=torch.bfloat16)
    x[:, :hidden] = x_logical
    w1 = torch.zeros(physical_hidden, physical_first, dtype=torch.bfloat16)
    w1[:hidden, :first_width] = w1_logical
    w2 = torch.zeros(physical_first, physical_q_b, dtype=torch.bfloat16)
    w2[:norm_features, :q_b_width] = w2_logical
    golden = torch.zeros(rows, physical_q_b, dtype=torch.bfloat16)
    golden[:, :q_b_width] = golden_logical

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=2.0)
    x_base = 0
    x_region = math.ceil((physical_rows * physical_hidden) / mlen) * mlen
    norm_base = math.ceil(x_region / (mlen * mlen)) * (mlen * mlen)
    norm_region = math.ceil((physical_rows * physical_first) / mlen) * mlen
    vram_preload = torch.zeros(norm_base + norm_region + mlen * mlen, dtype=torch.bfloat16)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="DeepSeekMLAQInput",
        tensor=x,
        vram_addr=x_base,
        physical_shape=(physical_rows, physical_hidden),
        vram_preload=vram_preload,
    )
    norm_rows = norm_weight.reshape(1, -1).repeat(rows, 1).to(torch.bfloat16)
    norm_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="DeepSeekMLAQNormWeight",
        tensor=norm_rows,
        vram_addr=norm_base,
        physical_shape=(physical_rows, physical_first),
        vram_preload=vram_preload,
    )
    w1_input = prog.input(
        "DeepSeekMLA_FIRST_WEIGHT_BF16",
        shape=(physical_hidden, first_width),
        physical_shape=(physical_hidden, physical_first),
        real_data_ratio=2.0,
    )
    w2_input = prog.input(
        "DeepSeekMLA_SECOND_WEIGHT_BF16",
        shape=(norm_features, q_b_width),
        physical_shape=(physical_first, physical_q_b),
        real_data_ratio=2.0,
    )

    first = prog.linear_projection_bf16(
        x_vram,
        w1_input,
        name="deepseek_mla_first",
        physical_shape=(physical_rows, physical_first),
    )
    norm_input = first
    if norm_features != first_width:
        norm_input = prog.alloc_at(
            "deepseek_mla_norm_view",
            rows,
            norm_features,
            prog.get_vram_addr(first.name),
            physical_shape=(physical_rows, physical_first),
        )
    prog.rms_norm(norm_input, eps_offset=66, reci_hid_offset=67)
    prog.vram_mul(norm_input, norm_vram, num_rows=rows)
    second = prog.linear_projection_bf16(
        norm_input,
        w2_input,
        name="deepseek_mla_second",
        physical_shape=(physical_rows, physical_q_b),
    )
    isa = prog.compile()
    if "C_SET_SCALE_REG" in isa:
        raise AssertionError("DeepSeek MLA BF16 Q path emitted C_SET_SCALE_REG")

    input_tensors = {
        "DeepSeekMLA_FIRST_WEIGHT_BF16": w1,
        "DeepSeekMLA_SECOND_WEIGHT_BF16": w2,
    }
    tensor_layouts = {
        "DeepSeekMLA_FIRST_WEIGHT_BF16": _bf16_layout(physical_hidden, physical_first),
        "DeepSeekMLA_SECOND_WEIGHT_BF16": _bf16_layout(physical_first, physical_q_b),
    }
    fp_preload = [0.0] * 68
    fp_preload[66] = args.norm_eps
    fp_preload[67] = 1.0 / norm_features
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    data_order = sorted(input_tensors, key=lambda name: hbm_addrs[name])
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=args.mode,
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    params = _comparison_params(
        prog._compiler.get_vram_addr(second.name),
        rows,
        physical_q_b,
        mlen,
        physical_rows=physical_rows,
    )
    _write_json(build_dir / "comparison_params.json", params)
    (build_dir / "generated_asm_code.asm").write_text(isa)

    if not args.no_run:
        run_emulator(build_dir)
        results, _ = compare_emulator_output(build_dir)
        emu_full = results["simulated_values"].reshape(rows, physical_q_b).to(torch.bfloat16)
        emu = emu_full[:, :q_b_width].contiguous()
        simulated_output_pt = build_dir / "simulated_output.pt"
        simulated_full_output_pt = build_dir / "simulated_full_output.pt"
        torch.save(emu, simulated_output_pt)
        torch.save(emu_full, simulated_full_output_pt)
        rel = _rel_rms(emu, golden_logical)
        strict_summary = _strict_tail_summary(emu, golden_logical)
        padded_tail_max_abs = float(emu_full[:, q_b_width:].abs().max().item()) if physical_q_b > q_b_width else 0.0
        if rel > 0.02:
            raise AssertionError(f"DeepSeek MLA frontend path rel_rms={rel:.6g} exceeds 2%")
    else:
        rel = float("nan")
        strict_summary = {}
        padded_tail_max_abs = None
        simulated_output_pt = None
        simulated_full_output_pt = None

    summary = {
        "mode": args.mode,
        "rows": rows,
        "hidden": hidden,
        "first_width": first_width,
        "norm_features": norm_features,
        "second_width": q_b_width,
        "physical_rows": physical_rows,
        "physical_hidden": physical_hidden,
        "physical_first": physical_first,
        "physical_second": physical_q_b,
        "split_first_projection": split_first_projection,
        "norm_eps": args.norm_eps,
        "scale_reg_count": isa.count("C_SET_SCALE_REG"),
        "h_prefetch_keyvalue_count": isa.count("H_PREFETCH_M"),
        "rel_rms": rel,
        "strict_tail": strict_summary,
        "padded_tail_max_abs": padded_tail_max_abs,
        "simulated_output_pt": str(simulated_output_pt) if simulated_output_pt is not None else None,
        "simulated_full_output_pt": str(simulated_full_output_pt) if simulated_full_output_pt is not None else None,
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _make_true_attention_hbm_inputs(
    prog: PlenaCompiler,
    *,
    hidden: int,
    q_width: int,
    kv_width: int,
    out_features: int,
    mlen: int,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    rotate: torch.Tensor,
) -> tuple[tuple[object, object, object, object, object], dict[str, torch.Tensor], dict[str, dict]]:
    """Register the BF16 HBM inputs shared by true-shape attention adapters."""
    w_q_input = prog.input("W_Q_TRUE_BF16", shape=(hidden, q_width), real_data_ratio=2.0)
    w_k_input = prog.input("W_K_TRUE_BF16", shape=(hidden, kv_width), real_data_ratio=2.0)
    w_v_input = prog.input("W_V_TRUE_BF16", shape=(hidden, kv_width), real_data_ratio=2.0)
    w_o_input = prog.input("W_O_TRUE_BF16", shape=(q_width, out_features), real_data_ratio=2.0)
    rotate_input = prog.input("ROPE_ROTATE_TRUE_FULL_BF16", shape=(mlen, mlen), real_data_ratio=2.0)
    input_tensors = {
        "W_Q_TRUE_BF16": w_q,
        "W_K_TRUE_BF16": w_k,
        "W_V_TRUE_BF16": w_v,
        "W_O_TRUE_BF16": w_o,
        "ROPE_ROTATE_TRUE_FULL_BF16": rotate,
    }
    tensor_layouts = {
        "W_Q_TRUE_BF16": _bf16_layout(hidden, q_width),
        "W_K_TRUE_BF16": _bf16_layout(hidden, kv_width),
        "W_V_TRUE_BF16": _bf16_layout(hidden, kv_width),
        "W_O_TRUE_BF16": _bf16_layout(q_width, out_features),
        "ROPE_ROTATE_TRUE_FULL_BF16": _bf16_layout(mlen, mlen),
    }
    return (w_q_input, w_k_input, w_v_input, w_o_input, rotate_input), input_tensors, tensor_layouts


def _emit_true_qkv_projection_bf16(
    prog: PlenaCompiler,
    projection_input,
    *,
    w_q_input,
    w_k_input,
    w_v_input,
    bias_vrams: dict[str, object],
    bias_parts: set[str],
    physical_rows: int,
    q_width: int,
    kv_width: int,
    seq: int,
):
    """Emit BF16 Q/K/V projection plus optional BF16 bias for true attention."""
    q_var = prog.linear_projection_bias_bf16(
        projection_input,
        w_q_input,
        bias_var=bias_vrams["B_Q_TRUE"] if "q" in bias_parts else None,
        name="Q_true_proj",
        physical_shape=(physical_rows, q_width),
        bias_rows=seq,
    )
    k_var = prog.linear_projection_bias_bf16(
        projection_input,
        w_k_input,
        bias_var=bias_vrams["B_K_TRUE"] if "k" in bias_parts else None,
        name="K_true_proj",
        physical_shape=(physical_rows, kv_width),
        bias_rows=seq,
    )
    v_var = prog.linear_projection_bias_bf16(
        projection_input,
        w_v_input,
        bias_var=bias_vrams["B_V_TRUE"] if "v" in bias_parts else None,
        name="V_true_proj",
        physical_shape=(physical_rows, kv_width),
        bias_rows=seq,
    )
    return q_var, k_var, v_var


def _emit_true_output_projection_bf16(
    prog: PlenaCompiler,
    attn_out,
    *,
    w_o_input,
    bias_vrams: dict[str, object],
    bias_parts: set[str],
    physical_rows: int,
    out_features: int,
    seq: int,
):
    """Emit BF16 O projection plus optional BF16 bias for true attention."""
    return prog.linear_projection_bias_bf16(
        attn_out,
        w_o_input,
        bias_var=bias_vrams["B_O_TRUE"] if "o" in bias_parts else None,
        name="O_true_proj",
        physical_shape=(physical_rows, out_features),
        bias_rows=seq,
    )


def _stage_true_kv_heads_bf16(
    prog: PlenaCompiler,
    *,
    k_var,
    v_var,
    rotate_input,
    cos_vram,
    sin_vram,
    k_norm_weight_vram,
    qk_head_norm: bool,
    qk_norm_eps_offset: int | None,
    qk_norm_reci_head_offset: int | None,
    seq: int,
    mlen: int,
    hkv: int,
    physical_rows: int,
    input_tensors: dict[str, torch.Tensor],
    tensor_layouts: dict[str, dict],
) -> tuple[list[object], list[object]]:
    """Apply runtime RoPE to K heads, store K/V heads, and register layouts."""
    k_inputs = []
    v_inputs = []
    k_base = prog.get_vram_addr(k_var.name)
    v_base = prog.get_vram_addr(v_var.name)
    for kv_idx in range(hkv):
        k_head = prog.alloc_at(
            f"K_true_head_{kv_idx}",
            seq,
            mlen,
            k_base + kv_idx * physical_rows * mlen,
            physical_shape=(physical_rows, mlen),
        )
        prog.head_runtime_rope_bf16(
            k_head,
            rotate_input,
            cos_vram,
            sin_vram,
            norm_weight_var=k_norm_weight_vram if qk_head_norm else None,
            eps_offset=qk_norm_eps_offset if qk_head_norm else None,
            reci_hid_offset=qk_norm_reci_head_offset if qk_head_norm else None,
            num_rows=seq,
            name=f"K_true_runtime_rotate_h{kv_idx}",
        )
        v_head = prog.alloc_at(
            f"V_true_head_{kv_idx}",
            seq,
            mlen,
            v_base + kv_idx * physical_rows * mlen,
            physical_shape=(physical_rows, mlen),
        )
        k_staged = prog.store(
            k_head,
            name=f"K_true_staged_h{kv_idx}",
            precision=1,
            hbm_element_bytes=2,
            real_data_ratio=2.0,
        )
        v_staged = prog.store(
            v_head,
            name=f"V_true_staged_h{kv_idx}",
            precision=1,
            hbm_element_bytes=2,
            real_data_ratio=2.0,
        )
        k_inputs.append(k_staged)
        v_inputs.append(v_staged)
        input_tensors[k_staged.name] = torch.zeros(physical_rows, mlen, dtype=torch.bfloat16)
        input_tensors[v_staged.name] = torch.zeros(physical_rows, mlen, dtype=torch.bfloat16)
        tensor_layouts[k_staged.name] = _bf16_layout(physical_rows, mlen, precision="HBM_V_KV_TYPE")
        tensor_layouts[v_staged.name] = _bf16_layout(physical_rows, mlen, precision="HBM_V_KV_TYPE")
    return k_inputs, v_inputs


def _emit_true_q_heads_attention_bf16(
    prog: PlenaCompiler,
    *,
    q_var,
    rotate_input,
    cos_vram,
    sin_vram,
    q_norm_weight_vram,
    qk_head_norm: bool,
    qk_norm_eps_offset: int | None,
    qk_norm_reci_head_offset: int | None,
    seq: int,
    mlen: int,
    hq: int,
    hkv: int,
    head_dim: int,
    physical_rows: int,
    k_inputs: list[object],
    v_inputs: list[object],
    attn_out,
    scratch,
    causal_mask,
    scale: float,
    sink_base: int | None,
    q_pre: torch.Tensor,
    q_norm_weight: torch.Tensor | None,
    qk_norm_eps: float,
    q: torch.Tensor,
    debug_stop_after_first_q_norm: bool,
    debug_stop_after_first_q_head: bool,
) -> tuple[object, torch.Tensor | None, bool]:
    """Emit per-Q-head runtime RoPE and packed attention for true GPT-OSS shapes."""
    q_base = prog.get_vram_addr(q_var.name)
    out_base = prog.get_vram_addr(attn_out.name)
    ratio = hq // hkv
    for head in range(hq):
        q_head = prog.alloc_at(
            f"Q_true_head_{head}",
            seq,
            mlen,
            q_base + head * physical_rows * mlen,
            physical_shape=(physical_rows, mlen),
        )
        if debug_stop_after_first_q_norm:
            if qk_head_norm:
                prog.rms_norm(q_head, eps_offset=qk_norm_eps_offset, reci_hid_offset=qk_norm_reci_head_offset)
                prog.vram_mul(q_head, q_norm_weight_vram, num_rows=seq)
                golden = _head_rms_norm_golden(
                    q_pre.reshape(seq, hq, head_dim)[:, head : head + 1, :],
                    q_norm_weight,
                    qk_norm_eps,
                ).reshape(seq, head_dim)
            else:
                golden = q_pre.reshape(seq, hq, head_dim)[:, head, :].contiguous()
            return q_head, golden, True
        prog.head_runtime_rope_bf16(
            q_head,
            rotate_input,
            cos_vram,
            sin_vram,
            norm_weight_var=q_norm_weight_vram if qk_head_norm else None,
            eps_offset=qk_norm_eps_offset if qk_head_norm else None,
            reci_hid_offset=qk_norm_reci_head_offset if qk_head_norm else None,
            num_rows=seq,
            name=f"Q_true_runtime_rotate_h{head}",
        )
        if debug_stop_after_first_q_head:
            return q_head, q[:, head, :].contiguous(), True
        kv_idx = head // ratio
        prog.flash_attention_packed_group(
            q_head,
            k_inputs[kv_idx],
            v_inputs[kv_idx],
            group_heads=1,
            head_slot_dim=head_dim,
            output_base_address=out_base + head * physical_rows * mlen,
            scratch_base_address=prog.get_vram_addr(scratch.name),
            broadcast_amount=1,
            scale=scale,
            causal_mask=causal_mask,
            valid_cols=seq,
            sink_base_address=None if sink_base is None else sink_base + head,
            output_head_base=0,
            k_matrix_precision="keyvalue",
            k_set_scale=False,
            k_hbm_element_bytes=2,
            v_hbm_element_bytes=2,
        )
    return attn_out, None, False


def run_runtime_rope(args: argparse.Namespace) -> dict:
    build_dir = (args.build_dir / "runtime_rope").expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    seq = args.seq_len or 16
    hidden = args.hidden_size or 128
    packed_heads = 4
    head_dim = mlen // packed_heads
    if mlen % packed_heads != 0:
        raise ValueError(f"MLEN={mlen} must be divisible by packed_heads={packed_heads}")
    if seq > mlen:
        raise ValueError("runtime_rope smoke is single sequence tile only")
    if seq % blen != 0:
        raise ValueError(f"seq={seq} must be a multiple of BLEN={blen}")
    if hidden % mlen != 0:
        raise ValueError("hidden must be a multiple of MLEN")

    torch.manual_seed(args.seed)
    x = (torch.randn(seq, hidden) * 0.2).to(torch.bfloat16)
    w_q = (torch.randn(hidden, mlen) * 0.2).to(torch.bfloat16)
    b_q = (torch.randn(mlen) * 0.02).to(torch.bfloat16)
    rotate, cos, sin = _make_packed_rope_inputs(seq, packed_heads, head_dim, args.rope_theta)

    q_pre = torch.matmul(x.float(), w_q.float()).to(torch.bfloat16)
    q_pre = (q_pre.float() + b_q.float()).to(torch.bfloat16)
    q_heads = q_pre.reshape(seq, packed_heads, head_dim)
    cos_head = cos[:, :head_dim]
    sin_head = sin[:, :head_dim]
    golden = _apply_rope_bf16(q_heads, cos_head, sin_head).reshape(seq, mlen).contiguous()

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=2.0)
    physical_rows = max(mlen, math.ceil(seq / blen) * blen)
    x_region = physical_rows * hidden
    bias_base = _align_to_tile(x_region, mlen)
    bias_shape = (physical_rows, mlen)
    bias_size = _align_to_tile(bias_shape[0] * bias_shape[1], mlen)
    cos_base = bias_base + bias_size
    trig_shape = (physical_rows, mlen)
    trig_size = _align_to_tile(trig_shape[0] * trig_shape[1], mlen)
    sin_base = cos_base + trig_size
    vram_preload = torch.zeros(sin_base + trig_size, dtype=torch.bfloat16)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="X",
        tensor=x,
        vram_addr=0,
        physical_shape=(physical_rows, hidden),
        vram_preload=vram_preload,
    )
    bias_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="B_Q",
        tensor=b_q.reshape(1, -1).repeat(seq, 1),
        vram_addr=bias_base,
        physical_shape=bias_shape,
        vram_preload=vram_preload,
    )
    cos_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="ROPE_COS",
        tensor=cos,
        vram_addr=cos_base,
        physical_shape=trig_shape,
        vram_preload=vram_preload,
    )
    sin_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="ROPE_SIN",
        tensor=sin,
        vram_addr=sin_base,
        physical_shape=trig_shape,
        vram_preload=vram_preload,
    )

    w_q_input = prog.input("W_Q_BF16", shape=(hidden, mlen), real_data_ratio=2.0)
    rotate_input = prog.input("ROPE_ROTATE_BF16", shape=(mlen, mlen), real_data_ratio=2.0)
    q_var = prog.linear_projection_bf16(x_vram, w_q_input, name="Q_proj", physical_shape=(physical_rows, mlen))
    prog.vram_add(q_var, bias_vram, num_rows=seq)
    prog.runtime_rope_projection_bf16(q_var, rotate_input, cos_vram, sin_vram, name="Q_runtime_rotate")
    isa = prog.compile()
    if "C_SET_SCALE_REG" in isa:
        raise AssertionError("runtime BF16 projection/RoPE path emitted C_SET_SCALE_REG")

    input_tensors = {
        "W_Q_BF16": w_q,
        "ROPE_ROTATE_BF16": rotate,
    }
    tensor_layouts = {
        "W_Q_BF16": _bf16_layout(hidden, mlen),
        "ROPE_ROTATE_BF16": _bf16_layout(mlen, mlen),
    }
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    data_order = sorted(input_tensors, key=lambda name: hbm_addrs[name])
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_attention_runtime_rope",
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    params = _comparison_params(
        prog.get_vram_addr(q_var.name),
        seq,
        mlen,
        mlen,
        physical_rows=q_var.physical_shape[0],
    )
    _write_json(build_dir / "comparison_params.json", params)
    (build_dir / "generated_asm_code.asm").write_text(isa)

    if not args.no_run:
        run_and_assert(build_dir, "gpt_oss_attention_runtime_rope", mlen=mlen, blen=blen)
        results, _ = compare_emulator_output(build_dir)
        emu = results["simulated_values"].reshape(seq, mlen).to(torch.bfloat16)
        rel = _rel_rms(emu, golden)
        if rel > 0.01:
            raise AssertionError(f"runtime_rope rel_rms={rel:.6g} exceeds 1%")
    else:
        rel = float("nan")

    summary = {
        "mode": "runtime_rope",
        "seq": seq,
        "hidden": hidden,
        "packed_heads": packed_heads,
        "head_dim": head_dim,
        "runtime_projection": True,
        "runtime_bias": True,
        "runtime_rope": True,
        "scale_reg_count": isa.count("C_SET_SCALE_REG"),
        "rel_rms": rel,
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _gpt_oss_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor | None,
    *,
    scale: float,
    sliding_window: int | None,
    causal: bool = True,
) -> torch.Tensor:
    # q: [seq, hq, d], k/v: [seq, hkv, d]
    seq, hq, d = q.shape
    hkv = k.shape[1]
    ratio = hq // hkv
    qh = q.transpose(0, 1).float()
    kh = k.repeat_interleave(ratio, dim=1).transpose(0, 1).float()
    vh = v.repeat_interleave(ratio, dim=1).transpose(0, 1).float()
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    row = torch.arange(seq).reshape(1, seq, 1)
    col = torch.arange(seq).reshape(1, 1, seq)
    if causal:
        mask = col <= row
        if sliding_window is not None:
            mask = mask & (col > row - sliding_window)
        scores = scores.masked_fill(~mask, float("-inf"))
    if sinks is not None:
        sink_logits = sinks.float().reshape(hq, 1, 1).expand(hq, seq, 1)
        logits = torch.cat([scores, sink_logits], dim=-1)
        probs = F.softmax(logits, dim=-1)[..., :seq]
    else:
        probs = F.softmax(scores, dim=-1)
    out = torch.matmul(probs, vh)
    return out.transpose(0, 1).reshape(seq, hq * d).to(torch.bfloat16)


def _causal_score_mask_tensor(seq: int, mlen: int, *, sliding_window: int | None = None) -> torch.Tensor:
    """Build a BF16 score mask for single-tile causal/sliding attention."""
    if seq < 0 or seq > mlen:
        raise ValueError(f"seq must be in [0, {mlen}], got {seq}")
    row = torch.arange(mlen).reshape(mlen, 1)
    col = torch.arange(mlen).reshape(1, mlen)
    visible = (row < seq) & (col < seq) & (col <= row)
    if sliding_window is not None:
        visible = visible & (col > row - sliding_window)
    mask = torch.full((mlen, mlen), float("-inf"), dtype=torch.bfloat16)
    mask[visible] = torch.tensor(0.0, dtype=torch.bfloat16)
    return mask


def _build_score_mask_vram(
    prog: PlenaCompiler,
    name: str,
    *,
    seq: int,
    mlen: int,
    sliding_window: int | None = None,
):
    """Build a single-tile causal/sliding score mask directly in VRAM.

    Keep -inf on the scalar/VRAM path instead of loading it through HBM, where
    activation quantization can change the masking semantics.
    """
    mask = prog.alloc(name, mlen, mlen)
    mask_addr = prog.get_vram_addr(mask.name)
    fp_base = prog._ONLINE_SOFTMAX_FPSRAM_BASE
    gp_mask, gp_fp = prog.register_allocator.allocate_gp(2)
    lines = [
        f"; === Build causal score mask: MLEN={mlen}, seq={seq}, sliding_window={sliding_window} ===",
        f"S_ADDI_INT gp{gp_fp}, gp0, {fp_base}",
        "S_LD_FP f7, gp0, 2",
    ]
    lines.extend(load_large_int(gp_mask, mask_addr))
    for row in range(mlen):
        for col in range(mlen):
            visible = row < seq and col < seq and col <= row
            if sliding_window is not None:
                visible = visible and col > row - sliding_window
            lines.append(f"S_ST_FP {'f0' if visible else 'f7'}, gp{gp_fp}, {col}")
        lines.append(f"S_MAP_V_FP gp{gp_mask}, gp{gp_fp}, 0")
        lines.append(f"S_ADDI_INT gp{gp_mask}, gp{gp_mask}, {mlen}")
    prog.register_allocator.free_gp([gp_mask, gp_fp])
    prog.emit("\n".join(lines) + "\n")
    return mask


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _make_rope_tables(seq: int, head_dim: int, *, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    positions = torch.arange(seq).float()
    angles = torch.outer(positions, freqs)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    return torch.cat([cos_half, cos_half], dim=-1).to(torch.bfloat16), torch.cat([sin_half, sin_half], dim=-1).to(
        torch.bfloat16
    )


def _apply_rope_bf16(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x.float() * cos[:, None, :].float() + _rotate_half(x).float() * sin[:, None, :].float()).to(torch.bfloat16)


def run_true_core(args: argparse.Namespace) -> dict:
    """True-shape GPT-OSS attention core: 64Q/8KV via unrolled per-Q-head groups.

    By default this starts after projection+bias+RoPE: Q/K/V are prestaged as
    BF16 tensors.  With --runtime-rope, Q/K are prestaged before RoPE and the
    device computes rotate-half plus RoPE before running the same attention core.
    """
    build_dir = (args.build_dir / "true_core").expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    hq = args.num_attention_heads
    hkv = args.num_key_value_heads
    head_dim = args.hlen or mlen
    value_head_dim = args.value_head_dim or head_dim
    seq = args.seq_len or 16
    if head_dim != mlen:
        raise ValueError(f"true_core currently uses one Q head per MLEN row; got head_dim={head_dim}, MLEN={mlen}")
    if hq % hkv != 0:
        raise ValueError(f"num_attention_heads={hq} must be divisible by num_key_value_heads={hkv}")
    if seq > mlen:
        raise ValueError("true_core is single sequence tile only")
    if seq % blen != 0:
        raise ValueError(f"seq={seq} must be a multiple of BLEN={blen}")
    sliding_window, sliding_source = _resolve_sliding_window(args)
    runtime_rope = bool(args.runtime_rope)
    external_qkv = any(getattr(args, name) is not None for name in ("q_pt", "k_pt", "v_pt"))
    if external_qkv and not all(getattr(args, name) is not None for name in ("q_pt", "k_pt", "v_pt")):
        raise ValueError("--q-pt, --k-pt, and --v-pt must be provided together")
    if external_qkv and runtime_rope:
        raise ValueError(
            "external Q/K tensors are expected post-RoPE; do not combine --q-pt/--k-pt with --runtime-rope"
        )
    o_projection = args.o_weight_pt is not None
    if args.o_golden_pt is not None and not o_projection:
        raise ValueError("--o-golden-pt requires --o-weight-pt")

    torch.manual_seed(args.seed)
    if external_qkv:
        q = _load_hidden_tensor(args.q_pt).to(torch.bfloat16)
        k = _load_hidden_tensor(args.k_pt).to(torch.bfloat16)
        v_loaded = _load_hidden_tensor(args.v_pt).to(torch.bfloat16)
        if q.dim() == 2:
            q = q.reshape(seq, hq, head_dim)
        if k.dim() == 2:
            k = k.reshape(seq, hkv, head_dim)
        if v_loaded.dim() == 2:
            inferred_v_dim = v_loaded.shape[-1] // hkv
            v_loaded = v_loaded.reshape(seq, hkv, inferred_v_dim)
        if tuple(q.shape) != (seq, hq, head_dim):
            raise ValueError(f"--q-pt shape {tuple(q.shape)} != {(seq, hq, head_dim)}")
        if tuple(k.shape) != (seq, hkv, head_dim):
            raise ValueError(f"--k-pt shape {tuple(k.shape)} != {(seq, hkv, head_dim)}")
        value_head_dim = args.value_head_dim or v_loaded.shape[-1]
        if tuple(v_loaded.shape) != (seq, hkv, value_head_dim):
            raise ValueError(f"--v-pt shape {tuple(v_loaded.shape)} != {(seq, hkv, value_head_dim)}")
        q_pre = q
        k_pre = k
    else:
        if not (0 < value_head_dim <= head_dim):
            raise ValueError(f"value_head_dim must be in (0, head_dim], got {value_head_dim} for head_dim={head_dim}")
        q_pre = (torch.randn(seq, hq, head_dim) * 0.2).to(torch.bfloat16)
        k_pre = (torch.randn(seq, hkv, head_dim) * 0.2).to(torch.bfloat16)
        v_loaded = (torch.randn(seq, hkv, value_head_dim) * 0.2).to(torch.bfloat16)
        cos, sin = _make_rope_tables(seq, head_dim, theta=args.rope_theta)
        q = _apply_rope_bf16(q_pre, cos, sin)
        k = _apply_rope_bf16(k_pre, cos, sin)
    if not (0 < value_head_dim <= head_dim):
        raise ValueError(f"value_head_dim must be in (0, head_dim], got {value_head_dim} for head_dim={head_dim}")
    v = torch.zeros(seq, hkv, head_dim, dtype=torch.bfloat16)
    v[:, :, :value_head_dim] = v_loaded
    sinks = (torch.randn(hq) * 0.1).to(torch.bfloat16) if args.sink else None
    scale = float(args.attention_scale) if args.attention_scale is not None else 1.0 / math.sqrt(head_dim)
    if args.golden_pt is not None:
        golden = _load_hidden_tensor(args.golden_pt).to(torch.bfloat16)
        if golden.dim() == 2:
            golden = golden.reshape(seq, hq, head_dim)
        if tuple(golden.shape) != (seq, hq, head_dim):
            raise ValueError(f"--golden-pt shape {tuple(golden.shape)} != {(seq, hq, head_dim)}")
        golden = golden.reshape(seq, hq * head_dim).contiguous()
    else:
        golden = _gpt_oss_attention_ref(
            q,
            k,
            v,
            sinks,
            scale=scale,
            sliding_window=sliding_window,
            causal=not args.no_causal_mask,
        )

    o_weight_expanded = None
    o_logical_out_features = None
    o_physical_out_features = None
    o_golden_from_padded_rel = None
    if o_projection:
        o_weight_raw = _load_hidden_tensor(args.o_weight_pt).to(torch.bfloat16)
        logical_o_in = hq * value_head_dim
        padded_o_in = hq * head_dim
        if o_weight_raw.dim() != 2:
            raise ValueError(f"--o-weight-pt must be rank-2, got shape {tuple(o_weight_raw.shape)}")
        if o_weight_raw.shape[0] == logical_o_in:
            o_weight_logical = o_weight_raw
        elif o_weight_raw.shape[1] == logical_o_in:
            o_weight_logical = o_weight_raw.T.contiguous()
        else:
            raise ValueError(
                f"--o-weight-pt shape {tuple(o_weight_raw.shape)} is incompatible with logical input "
                f"{logical_o_in}; expected [in,out] or [out,in]"
            )
        o_logical_out_features = int(o_weight_logical.shape[1])
        o_physical_out_features = math.ceil(o_logical_out_features / mlen) * mlen
        o_weight_expanded = torch.zeros(padded_o_in, o_physical_out_features, dtype=torch.bfloat16)
        for head in range(hq):
            logical_start = head * value_head_dim
            padded_start = head * head_dim
            o_weight_expanded[padded_start : padded_start + value_head_dim, :o_logical_out_features] = o_weight_logical[
                logical_start : logical_start + value_head_dim, :
            ]
        golden_o = torch.matmul(golden.float(), o_weight_expanded[:, :o_logical_out_features].float()).to(
            torch.bfloat16
        )
        if args.o_golden_pt is not None:
            o_golden_ref = _load_hidden_tensor(args.o_golden_pt).to(torch.bfloat16)
            if tuple(o_golden_ref.shape) != (seq, o_logical_out_features):
                raise ValueError(f"--o-golden-pt shape {tuple(o_golden_ref.shape)} != {(seq, o_logical_out_features)}")
            o_golden_from_padded_rel = _rel_rms(golden_o, o_golden_ref)
            golden_o = o_golden_ref
        golden_padded = torch.zeros(seq, o_physical_out_features, dtype=torch.bfloat16)
        golden_padded[:, :o_logical_out_features] = golden_o
        golden = golden_padded

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=2.0)
    physical_rows = max(mlen, math.ceil(seq / blen) * blen)
    q_width = hq * head_dim
    q_region = physical_rows * q_width
    k_pre_base = q_region
    k_pre_region = hkv * physical_rows * mlen if runtime_rope else 0
    rope_base = _align_to_tile(k_pre_base + k_pre_region, mlen)
    trig_shape = (physical_rows, mlen)
    trig_size = _align_to_tile(trig_shape[0] * trig_shape[1], mlen)
    preload_words = q_region
    if runtime_rope:
        preload_words = rope_base + 2 * trig_size
    q_vram_preload = torch.zeros(preload_words, dtype=torch.bfloat16)
    q_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="Q_pre_true" if runtime_rope else "Q_rope_true",
        tensor=(q_pre if runtime_rope else q).reshape(seq, q_width),
        vram_addr=0,
        physical_shape=(physical_rows, q_width),
        vram_preload=q_vram_preload,
    )
    rotate_input = None
    cos_vram = sin_vram = None
    if runtime_rope:
        rotate, cos_packed, sin_packed = _make_packed_rope_inputs(seq, 1, head_dim, args.rope_theta)
        cos_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="ROPE_COS_TRUE",
            tensor=cos_packed,
            vram_addr=rope_base,
            physical_shape=trig_shape,
            vram_preload=q_vram_preload,
        )
        sin_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="ROPE_SIN_TRUE",
            tensor=sin_packed,
            vram_addr=rope_base + trig_size,
            physical_shape=trig_shape,
            vram_preload=q_vram_preload,
        )
        rotate_input = prog.input("ROPE_ROTATE_TRUE_BF16", shape=(mlen, mlen), real_data_ratio=2.0)

    k_inputs = []
    v_inputs = []
    input_tensors = {}
    tensor_layouts = {}
    for kv_idx in range(hkv):
        k_name = f"K_rope_h{kv_idx}"
        v_name = f"V_h{kv_idx}"
        if runtime_rope:
            k_pre_vram = prestage_bf16_vram_matrix(
                prog=prog,
                name=f"K_pre_h{kv_idx}",
                tensor=k_pre[:, kv_idx, :],
                vram_addr=k_pre_base + kv_idx * physical_rows * mlen,
                physical_shape=(physical_rows, mlen),
                vram_preload=q_vram_preload,
            )
            prog.runtime_rope_projection_bf16(
                k_pre_vram,
                rotate_input,
                cos_vram,
                sin_vram,
                name=f"K_runtime_rotate_h{kv_idx}",
            )
            k_input = prog.store(
                k_pre_vram,
                name=k_name,
                precision=1,
                hbm_element_bytes=2,
                real_data_ratio=2.0,
            )
            input_tensors[k_name] = torch.zeros(physical_rows, mlen, dtype=torch.bfloat16)
        else:
            k_input = prog.input(k_name, shape=(seq, mlen), physical_shape=(physical_rows, mlen))
            k_padded = torch.zeros(physical_rows, mlen, dtype=torch.bfloat16)
            k_padded[:seq, :] = k[:, kv_idx, :]
            input_tensors[k_name] = k_padded
        v_input = prog.input(v_name, shape=(seq, mlen), physical_shape=(physical_rows, mlen))
        k_inputs.append(k_input)
        v_inputs.append(v_input)
        v_padded = torch.zeros(physical_rows, mlen, dtype=torch.bfloat16)
        v_padded[:seq, :] = v[:, kv_idx, :]
        input_tensors[v_name] = v_padded
        tensor_layouts[k_name] = _bf16_layout(physical_rows, mlen)
        tensor_layouts[v_name] = _bf16_layout(physical_rows, mlen)
    if runtime_rope:
        input_tensors["ROPE_ROTATE_TRUE_BF16"] = rotate
        tensor_layouts["ROPE_ROTATE_TRUE_BF16"] = _bf16_layout(mlen, mlen)

    causal_mask = None
    if not args.no_causal_mask:
        mask_vram_name = (
            "_gpt_oss_true_core_sliding_mask" if sliding_window is not None else "_gpt_oss_true_core_causal_mask"
        )
        causal_mask = _build_score_mask_vram(
            prog,
            mask_vram_name,
            seq=seq,
            mlen=mlen,
            sliding_window=sliding_window,
        )

    out = prog.alloc("O_true_64q8kv", seq, q_width, strict=False, physical_shape=(physical_rows, q_width))
    scratch = prog.alloc("_gpt_oss_true_core_scratch", mlen, mlen, strict=True)

    fp_preload = [0.0, scale / 0.25, float("-inf")] + [0.0] * 253
    sink_base = None
    if sinks is not None:
        sink_base = 256
        if len(fp_preload) < sink_base + hq:
            fp_preload.extend([0.0] * (sink_base + hq - len(fp_preload)))
        for idx, value in enumerate(sinks.tolist()):
            fp_preload[sink_base + idx] = float(value)

    q_base = prog.get_vram_addr(q_vram.name)
    out_base = prog.get_vram_addr(out.name)
    ratio = hq // hkv
    for head in range(hq):
        q_head = prog.alloc_at(
            f"Q_head_{head}",
            seq,
            mlen,
            q_base + head * physical_rows * mlen,
            physical_shape=(physical_rows, mlen),
        )
        if runtime_rope:
            prog.runtime_rope_projection_bf16(
                q_head,
                rotate_input,
                cos_vram,
                sin_vram,
                name=f"Q_runtime_rotate_h{head}",
            )
        kv_idx = head // ratio
        prog.flash_attention_packed_group(
            q_head,
            k_inputs[kv_idx],
            v_inputs[kv_idx],
            group_heads=1,
            head_slot_dim=head_dim,
            output_base_address=out_base + head * physical_rows * mlen,
            scratch_base_address=prog.get_vram_addr(scratch.name),
            broadcast_amount=1,
            scale=scale,
            causal_mask=causal_mask,
            valid_cols=seq,
            k_matrix_precision="keyvalue",
            k_set_scale=False,
            k_hbm_element_bytes=2,
        )
    if o_projection:
        o_weight_input = prog.input(
            "O_proj_padded_weight",
            shape=(q_width, o_physical_out_features),
        )
        input_tensors["O_proj_padded_weight"] = o_weight_expanded
        tensor_layouts["O_proj_padded_weight"] = _bf16_layout(q_width, o_physical_out_features)
        out = prog.linear_projection_bf16(
            out,
            o_weight_input,
            name="O_true_padded_projection",
            physical_shape=(physical_rows, o_physical_out_features),
        )
    isa = prog.compile()

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    data_order = sorted(input_tensors, key=lambda name: hbm_addrs[name])
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=q_vram_preload,
        tensor_layouts=tensor_layouts,
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_attention_true_core",
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    compare_width = out.shape[1]
    params = _comparison_params(
        prog.get_vram_addr(out.name),
        seq,
        compare_width,
        mlen,
        physical_rows=out.physical_shape[0],
    )
    _write_json(build_dir / "comparison_params.json", params)
    (build_dir / "generated_asm_code.asm").write_text(isa)

    if not args.no_run:
        run_emulator(build_dir)
        results, _ = compare_emulator_output(build_dir)
        emu = results["simulated_values"].reshape(seq, compare_width).to(torch.bfloat16)
        simulated_output_pt = build_dir / "simulated_output.pt"
        torch.save(emu, simulated_output_pt)
        rel = _rel_rms(emu, golden)
        if rel > 0.02:
            raise AssertionError(f"true_core rel_rms={rel:.6g} exceeds 2%")
    else:
        rel = float("nan")
        emu = torch.empty(0, dtype=torch.bfloat16)
        simulated_output_pt = None

    padded_value_tail_max_abs = None
    if value_head_dim < head_dim and emu.numel():
        padded_value_tail_max_abs = float(
            emu.reshape(seq, hq, head_dim)[:, :, value_head_dim:].abs().max().item() if not o_projection else 0.0
        )

    summary = {
        "mode": "true_core",
        "seq": seq,
        "num_attention_heads": hq,
        "num_key_value_heads": hkv,
        "head_dim": head_dim,
        "value_head_dim": value_head_dim,
        "o_projection": bool(o_projection),
        "o_logical_out_features": o_logical_out_features,
        "o_physical_out_features": o_physical_out_features,
        "o_golden_from_padded_rel_rms": o_golden_from_padded_rel,
        "attention_scale": scale,
        "padded_value_tail_max_abs": padded_value_tail_max_abs,
        "sink": bool(args.sink),
        "sliding_window": sliding_window,
        "sliding_source": sliding_source,
        "rel_rms": rel,
        "sink_base": sink_base,
        "strategy": "unrolled_one_q_head_per_call",
        "runtime_projection": False,
        "runtime_rope": runtime_rope,
        "simulated_output_pt": str(simulated_output_pt) if simulated_output_pt is not None else None,
        "note": (
            "Q/K are prestaged before RoPE and device-side runtime RoPE feeds true-shape attention core."
            if runtime_rope
            else "Q/K/V are prestaged after bias/RoPE; this validates true-shape per-head sink attention core only."
        ),
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_qk_head_norm_smoke(args: argparse.Namespace) -> dict:
    """Isolate Qwen-style per-head RMSNorm on one BF16 VRAM head.

    The full Qwen-like attention probe produced all-NaN output once Q/K head
    RMSNorm was inserted.  This smoke keeps attention and RoPE out of the
    picture so a failure points at the norm lowering itself rather than the
    downstream online-softmax path.
    """
    build_dir = (args.build_dir / "qk_head_norm_smoke").expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    seq = args.seq_len or 8
    head_dim = args.hlen or mlen
    eps = args.qk_head_norm_eps
    if not (0 < head_dim <= mlen):
        raise ValueError(f"qk_head_norm_smoke expects 0 < head_dim <= MLEN; got {head_dim} and {mlen}")
    if seq > mlen:
        raise ValueError("qk_head_norm_smoke is single sequence tile only")
    if seq % blen != 0:
        raise ValueError(f"seq={seq} must be a multiple of BLEN={blen}")

    torch.manual_seed(args.seed)
    x = (torch.randn(seq, head_dim) * 0.2).to(torch.bfloat16)
    input_source = "random"
    if getattr(args, "input_pt", None) is not None:
        loaded_x = torch.load(args.input_pt.expanduser())
        if isinstance(loaded_x, dict):
            loaded_x = loaded_x.get("x", loaded_x.get("tensor", loaded_x.get("hidden", loaded_x)))
        if not torch.is_tensor(loaded_x):
            raise TypeError(f"--input-pt must contain a Tensor or tensor dict, got {type(loaded_x).__name__}")
        loaded_x = loaded_x.to(torch.bfloat16).contiguous()
        if tuple(loaded_x.shape) != (seq, head_dim):
            raise ValueError(f"--input-pt shape {tuple(loaded_x.shape)} != expected {(seq, head_dim)}")
        x = loaded_x
        input_source = str(args.input_pt.expanduser())
    weight = (1.0 + torch.randn(head_dim) * 0.05).to(torch.bfloat16)
    if getattr(args, "qk_norm_weight_pt", None) is not None:
        loaded_weight = torch.load(args.qk_norm_weight_pt.expanduser())
        if isinstance(loaded_weight, dict):
            loaded_weight = loaded_weight.get("weight", loaded_weight.get("tensor", loaded_weight))
        if not torch.is_tensor(loaded_weight):
            raise TypeError(
                f"--qk-norm-weight-pt must contain a Tensor or tensor dict, got {type(loaded_weight).__name__}"
            )
        loaded_weight = loaded_weight.to(torch.bfloat16).contiguous().reshape(-1)
        if tuple(loaded_weight.shape) != (head_dim,):
            raise ValueError(f"--qk-norm-weight-pt shape {tuple(loaded_weight.shape)} != expected {(head_dim,)}")
        weight = loaded_weight
    weight_rows = weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16)
    golden = _head_rms_norm_golden(x.reshape(seq, 1, head_dim), weight, eps).reshape(seq, head_dim)
    rotate = None
    if args.runtime_rope:
        rotate, cos, sin = _make_packed_rope_inputs(seq, 1, head_dim, args.rope_theta)
        if head_dim < mlen:
            rotate_padded = torch.zeros(mlen, mlen, dtype=torch.bfloat16)
            rotate_padded[:head_dim, :head_dim] = rotate
            cos_padded = torch.zeros(seq, mlen, dtype=torch.bfloat16)
            sin_padded = torch.zeros(seq, mlen, dtype=torch.bfloat16)
            cos_padded[:, :head_dim] = cos
            sin_padded[:, :head_dim] = sin
            rotate, cos, sin = rotate_padded, cos_padded, sin_padded
        cos_head, sin_head = _make_rope_tables(seq, head_dim, theta=args.rope_theta)
        golden = _apply_rope_bf16(golden.reshape(seq, 1, head_dim), cos_head, sin_head).reshape(seq, head_dim)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=2.0)
    physical_rows = max(mlen, math.ceil(seq / blen) * blen)
    x_size = physical_rows * mlen
    weight_base = _align_to_tile(x_size, mlen)
    weight_size = _align_to_tile(physical_rows * mlen, mlen)
    rope_base = _align_to_tile(weight_base + weight_size, mlen)
    trig_shape = (physical_rows, mlen)
    trig_size = _align_to_tile(trig_shape[0] * trig_shape[1], mlen)
    vram_words = rope_base + 2 * trig_size if args.runtime_rope else weight_base + weight_size + mlen * mlen
    vram_preload = torch.zeros(vram_words, dtype=torch.bfloat16)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="QKHeadNormX",
        tensor=x,
        vram_addr=0,
        physical_shape=(physical_rows, mlen),
        vram_preload=vram_preload,
    )
    weight_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="QKHeadNormWeight",
        tensor=weight_rows,
        vram_addr=weight_base,
        physical_shape=(physical_rows, mlen),
        vram_preload=vram_preload,
    )
    cos_vram = sin_vram = None
    if args.runtime_rope:
        cos_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="QKHeadNormROPECos",
            tensor=cos,
            vram_addr=rope_base,
            physical_shape=trig_shape,
            vram_preload=vram_preload,
        )
        sin_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="QKHeadNormROPESin",
            tensor=sin,
            vram_addr=rope_base + trig_size,
            physical_shape=trig_shape,
            vram_preload=vram_preload,
        )
    input_tensors = {}
    tensor_layouts = {}
    if args.runtime_rope:
        rotate_input = prog.input("QKHeadNormROPERotate", shape=(mlen, mlen), real_data_ratio=2.0)
        prog.head_runtime_rope_bf16(
            x_vram,
            rotate_input,
            cos_vram,
            sin_vram,
            norm_weight_var=weight_vram,
            eps_offset=66,
            reci_hid_offset=67,
            num_rows=seq,
            name="QKHeadNormRuntimeRoPE",
        )
        input_tensors["QKHeadNormROPERotate"] = rotate
        tensor_layouts["QKHeadNormROPERotate"] = _bf16_layout(mlen, mlen)
    else:
        prog.rms_norm(x_vram, eps_offset=66, reci_hid_offset=67)
        prog.vram_mul(x_vram, weight_vram, num_rows=seq)
    isa = prog.compile()

    fp_preload = [0.0] * 68
    fp_preload[66] = eps
    fp_preload[67] = 1.0 / head_dim
    compare_width = mlen if head_dim < mlen else head_dim
    comparison_golden = golden
    if compare_width != head_dim:
        comparison_golden = torch.zeros(seq, compare_width, dtype=torch.bfloat16)
        comparison_golden[:, :head_dim] = golden
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": comparison_golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    data_order = sorted(input_tensors, key=lambda name: hbm_addrs[name])
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="qk_head_norm_smoke",
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    if not input_tensors:
        (build_dir / "hbm_for_behave_sim.bin").touch(exist_ok=True)
    _write_json(
        build_dir / "comparison_params.json",
        _comparison_params(prog.get_vram_addr(x_vram.name), seq, compare_width, mlen, physical_rows=physical_rows),
    )
    (build_dir / "generated_asm_code.asm").write_text(isa)

    run_emulator(build_dir)
    results, _ = compare_emulator_output(build_dir)
    emu_full = results["simulated_values"].reshape(seq, compare_width).to(torch.bfloat16)
    emu = emu_full[:, :head_dim].contiguous()
    rel = _rel_rms(emu, golden)
    strict_summary = _strict_tail_summary(emu, golden)
    padded_tail_max_abs = (
        float(emu_full[:, head_dim:].abs().max().item()) if compare_width > head_dim and emu_full.numel() else None
    )
    summary = {
        "mode": "qk_head_norm_smoke",
        "seq": seq,
        "head_dim": head_dim,
        "compare_width": compare_width,
        "padded_tail_max_abs": padded_tail_max_abs,
        "physical_rows": physical_rows,
        "qk_head_norm_eps": eps,
        "runtime_rope": bool(args.runtime_rope),
        "input_source": input_source,
        "rel_rms": rel,
        "strict_tail": strict_summary,
        "nonfinite_output": int((~torch.isfinite(emu.float())).sum().item()),
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_external_rope(args: argparse.Namespace) -> dict:
    """Apply device-side RoPE using externally supplied BF16 cos/sin tables."""
    required = {
        "--input-pt": args.input_pt,
        "--cos-pt": args.cos_pt,
        "--sin-pt": args.sin_pt,
        "--golden-pt": args.golden_pt,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"external_rope requires {', '.join(missing)}")

    build_dir = (args.build_dir / "external_rope").expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    x = _load_hidden_tensor(args.input_pt).to(torch.bfloat16).contiguous()
    cos = _load_hidden_tensor(args.cos_pt).to(torch.bfloat16).contiguous()
    sin = _load_hidden_tensor(args.sin_pt).to(torch.bfloat16).contiguous()
    golden = _load_hidden_tensor(args.golden_pt).to(torch.bfloat16).contiguous()
    original_shape = tuple(x.shape)
    if x.ndim == 2:
        rows, head_dim = map(int, x.shape)
        head_count = 1
    elif x.ndim == 3:
        seq_rows, head_count, head_dim = map(int, x.shape)
        rows = seq_rows * head_count
        if tuple(golden.shape) != original_shape:
            raise ValueError(f"--golden-pt shape {tuple(golden.shape)} != {original_shape}")
        if tuple(cos.shape) == (seq_rows, head_dim):
            cos = cos[:, None, :].expand(seq_rows, head_count, head_dim).contiguous()
        if tuple(sin.shape) == (seq_rows, head_dim):
            sin = sin[:, None, :].expand(seq_rows, head_count, head_dim).contiguous()
        x = x.reshape(rows, head_dim).contiguous()
        golden = golden.reshape(rows, head_dim).contiguous()
        cos = cos.reshape(rows, head_dim).contiguous()
        sin = sin.reshape(rows, head_dim).contiguous()
    else:
        raise ValueError(f"--input-pt must be rank-2 or rank-3 for external_rope, got {tuple(x.shape)}")
    if tuple(cos.shape) != (rows, head_dim) or tuple(sin.shape) != (rows, head_dim):
        raise ValueError(f"cos/sin shapes {tuple(cos.shape)}/{tuple(sin.shape)} != {(rows, head_dim)}")
    if tuple(golden.shape) != (rows, head_dim):
        raise ValueError(f"--golden-pt shape {tuple(golden.shape)} != {(rows, head_dim)}")
    if rows % blen != 0:
        raise ValueError(f"rows={rows} must be a multiple of BLEN={blen}")
    if head_dim > mlen:
        raise ValueError(f"head_dim={head_dim} exceeds MLEN={mlen}")

    physical_rows = max(mlen, math.ceil(rows / blen) * blen)
    x_padded = torch.zeros(rows, mlen, dtype=torch.bfloat16)
    cos_padded = torch.zeros(rows, mlen, dtype=torch.bfloat16)
    sin_padded = torch.zeros(rows, mlen, dtype=torch.bfloat16)
    golden_padded = torch.zeros(rows, mlen, dtype=torch.bfloat16)
    x_padded[:, :head_dim] = x
    cos_padded[:, :head_dim] = cos
    sin_padded[:, :head_dim] = sin
    golden_padded[:, :head_dim] = golden
    rotate = torch.zeros(mlen, mlen, dtype=torch.bfloat16)
    rotate[:head_dim, :head_dim] = _make_rotate_half_matrix(head_dim)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=2.0)
    x_region = physical_rows * mlen
    cos_base = _align_to_tile(x_region, mlen)
    trig_size = _align_to_tile(physical_rows * mlen, mlen)
    sin_base = cos_base + trig_size
    vram_preload = torch.zeros(sin_base + trig_size + mlen * mlen, dtype=torch.bfloat16)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="ExternalRoPEX",
        tensor=x_padded,
        vram_addr=0,
        physical_shape=(physical_rows, mlen),
        vram_preload=vram_preload,
    )
    cos_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="ExternalRoPECos",
        tensor=cos_padded,
        vram_addr=cos_base,
        physical_shape=(physical_rows, mlen),
        vram_preload=vram_preload,
    )
    sin_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="ExternalRoPESin",
        tensor=sin_padded,
        vram_addr=sin_base,
        physical_shape=(physical_rows, mlen),
        vram_preload=vram_preload,
    )
    rotate_input = prog.input("ExternalRoPERotate", shape=(mlen, mlen), real_data_ratio=2.0)
    prog.runtime_rope_projection_bf16(x_vram, rotate_input, cos_vram, sin_vram, name="ExternalRoPERotateHalf")
    isa = prog.compile()

    input_tensors = {"ExternalRoPERotate": rotate}
    tensor_layouts = {"ExternalRoPERotate": _bf16_layout(mlen, mlen)}
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden_padded},
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="external_rope",
        specified_data_order=list(input_tensors),
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    _write_json(
        build_dir / "comparison_params.json",
        _comparison_params(prog.get_vram_addr(x_vram.name), rows, mlen, mlen, physical_rows=physical_rows),
    )
    (build_dir / "generated_asm_code.asm").write_text(isa)

    run_emulator(build_dir)
    results, _ = compare_emulator_output(build_dir)
    emu_full = results["simulated_values"].reshape(rows, mlen).to(torch.bfloat16)
    emu = emu_full[:, :head_dim].contiguous()
    simulated_output_pt = build_dir / "simulated_output.pt"
    simulated_full_output_pt = build_dir / "simulated_full_output.pt"
    torch.save(emu.reshape(original_shape), simulated_output_pt)
    torch.save(emu_full, simulated_full_output_pt)
    rel = _rel_rms(emu, golden)
    strict_summary = _strict_tail_summary(emu, golden)
    padded_tail_max_abs = float(emu_full[:, head_dim:].abs().max().item()) if head_dim < mlen else 0.0
    if rel > 0.01:
        raise AssertionError(f"external_rope rel_rms={rel:.6g} exceeds 1%")
    summary = {
        "mode": "external_rope",
        "rows": rows,
        "head_count": head_count,
        "head_dim": head_dim,
        "original_shape": list(original_shape),
        "physical_rows": physical_rows,
        "rel_rms": rel,
        "strict_tail": strict_summary,
        "padded_tail_max_abs": padded_tail_max_abs,
        "scale_reg_count": isa.count("C_SET_SCALE_REG"),
        "simulated_output_pt": str(simulated_output_pt),
        "simulated_full_output_pt": str(simulated_full_output_pt),
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_true_full(args: argparse.Namespace) -> dict:
    """True-shape GPT-OSS attention with runtime projection+bias+RoPE.

    This is the attention semantic gate immediately before a decoder-block
    harness.  It keeps sequence single-tile but uses the real 64Q/8KV head
    structure and computes Q/K/V/O projections inside the emulator.
    """
    mode_name = args.mode
    build_dir = (args.build_dir / mode_name).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    hq = args.num_attention_heads
    hkv = args.num_key_value_heads
    head_dim = args.hlen or mlen
    seq = args.seq_len or 16
    hidden = args.hidden_size or 128
    out_features = args.out_features or hidden
    if head_dim != mlen:
        raise ValueError(f"true_full uses one head per MLEN row; got head_dim={head_dim}, MLEN={mlen}")
    if hq % hkv != 0:
        raise ValueError(f"num_attention_heads={hq} must be divisible by num_key_value_heads={hkv}")
    if seq > mlen:
        raise ValueError("true_full is single sequence tile only")
    if seq % blen != 0:
        raise ValueError(f"seq={seq} must be a multiple of BLEN={blen}")
    if hidden % mlen != 0 or out_features % mlen != 0:
        raise ValueError("hidden and out_features must be multiples of MLEN")
    attention_block = args.mode in ("true_attn_block", "true_decoder_block")
    decoder_block = args.mode == "true_decoder_block"
    if attention_block and out_features != hidden:
        raise ValueError("true_attn_block requires out_features == hidden for residual add")
    if decoder_block:
        moe_policy_name = args.moe_policy_name
        moe_activation_policy = args.moe_activation_policy
        supported_moe_shape = {
            ("gpt_oss", 32, 4): "gpt_oss_clamp_gated",
            ("qwen3_moe", 128, 8): "standard_swiglu",
        }
        expected_activation = supported_moe_shape.get((moe_policy_name, args.moe_num_experts, args.moe_top_k))
        if expected_activation is None:
            raise ValueError(
                "true_decoder_block supports only GPT-OSS 32/top4 and Qwen3-MoE 128/top8 for now; "
                f"got policy={moe_policy_name!r}, experts={args.moe_num_experts}, top_k={args.moe_top_k}"
            )
        if moe_activation_policy != expected_activation:
            raise ValueError(
                f"policy={moe_policy_name!r} expects activation_policy={expected_activation!r}, "
                f"got {moe_activation_policy!r}"
            )
        moe_intermediate = args.moe_intermediate_size or hidden
        if moe_intermediate % mlen != 0:
            raise ValueError(f"moe_intermediate={moe_intermediate} must be a multiple of MLEN={mlen}")
    else:
        moe_policy_name = None
        moe_activation_policy = None
        moe_intermediate = None
    sliding_window, sliding_source = _resolve_sliding_window(args)
    projection_bias = bool(args.projection_bias)
    bias_parts = _bias_parts(args)
    norm_eps = args.norm_eps
    norm_eps_offset = 64
    norm_reci_hid_offset = 65
    qk_norm_eps = args.qk_head_norm_eps
    # Q/K head-norm is applied after the attention masks have been materialized.
    # The mask builders and online-softmax scratch own the low FPRAM region
    # (ONLINE_SOFTMAX_FPSRAM_BASE + 3 * MLEN), so keep these constants in a
    # high slot.  66/67 looked natural next to the block RMSNorm constants, but
    # MLEN=128 mask construction overwrites them and turns the norm into
    # rsqrt(0) -> inf.
    qk_norm_eps_offset = 512
    qk_norm_reci_head_offset = 513
    ffn_norm_eps_offset = None
    ffn_norm_reci_hid_offset = None
    topk_weights_fp_base = None
    topk_indices_int_base = None
    decoder_moe_input_var = None
    decoder_moe_residual_var = None

    torch.manual_seed(args.seed)
    q_width = hq * head_dim
    kv_width = hkv * head_dim
    qwen_real_layer = bool(args.qwen_real_layer)
    qwen_router_matrix_bf16 = bool(getattr(args, "qwen_router_matrix_bf16", False))
    qwen_router_packed_skinny_bf16 = bool(getattr(args, "qwen_router_packed_skinny_bf16", False))
    qwen_real_metadata = None
    if qwen_router_matrix_bf16 and qwen_router_packed_skinny_bf16:
        raise ValueError("--qwen-router-matrix-bf16 and --qwen-router-packed-skinny-bf16 are mutually exclusive")
    if qwen_router_matrix_bf16 and (not decoder_block or moe_policy_name != "qwen3_moe"):
        raise ValueError("--qwen-router-matrix-bf16 is only valid for Qwen3-MoE true_decoder_block runs")
    if qwen_router_packed_skinny_bf16 and (not decoder_block or moe_policy_name != "qwen3_moe"):
        raise ValueError("--qwen-router-packed-skinny-bf16 is only valid for Qwen3-MoE true_decoder_block runs")
    if qwen_real_layer:
        if not decoder_block:
            raise ValueError("--qwen-real-layer is currently supported only with true_decoder_block")
        if moe_policy_name != "qwen3_moe" or moe_activation_policy != "standard_swiglu":
            raise ValueError("--qwen-real-layer requires --moe-policy-name qwen3_moe and standard_swiglu")
        if projection_bias:
            raise ValueError("Qwen3-30B-A3B config has attention_bias=false; do not pass --projection-bias")
        if args.sink:
            raise ValueError("Qwen3-30B-A3B attention has no GPT-OSS sink; do not pass --sink")
        if not args.qk_head_norm:
            raise ValueError("--qwen-real-layer requires --qk-head-norm")
    x = (torch.randn(seq, hidden) * 0.2).to(torch.bfloat16)
    input_source = "random"
    if getattr(args, "input_pt", None) is not None:
        loaded_x = torch.load(args.input_pt.expanduser())
        if isinstance(loaded_x, dict):
            for key in ("hidden", "output", "tensor", "x"):
                if key in loaded_x:
                    loaded_x = loaded_x[key]
                    break
        if not torch.is_tensor(loaded_x):
            raise TypeError(f"--input-pt must contain a Tensor or tensor dict, got {type(loaded_x).__name__}")
        loaded_x = loaded_x.to(torch.bfloat16)
        if tuple(loaded_x.shape) != (seq, hidden):
            raise ValueError(f"--input-pt shape {tuple(loaded_x.shape)} != expected {(seq, hidden)}")
        x = loaded_x.contiguous()
        input_source = str(args.input_pt.expanduser())
    norm_weight = (1.0 + torch.randn(hidden) * 0.05).to(torch.bfloat16) if attention_block else None
    norm_weight_rows = norm_weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16) if attention_block else None
    q_norm_weight = (1.0 + torch.randn(head_dim) * 0.05).to(torch.bfloat16) if args.qk_head_norm else None
    k_norm_weight = (1.0 + torch.randn(head_dim) * 0.05).to(torch.bfloat16) if args.qk_head_norm else None
    q_norm_weight_rows = q_norm_weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16) if args.qk_head_norm else None
    k_norm_weight_rows = k_norm_weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16) if args.qk_head_norm else None
    ffn_norm_weight = (1.0 + torch.randn(hidden) * 0.05).to(torch.bfloat16) if decoder_block else None
    ffn_norm_weight_rows = ffn_norm_weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16) if decoder_block else None
    w_q = (torch.randn(hidden, q_width) * 0.2).to(torch.bfloat16)
    w_k = (torch.randn(hidden, kv_width) * 0.2).to(torch.bfloat16)
    w_v = (torch.randn(hidden, kv_width) * 0.2).to(torch.bfloat16)
    w_o = (torch.randn(q_width, out_features) * 0.2).to(torch.bfloat16)
    b_q = (torch.randn(q_width) * 0.02).to(torch.bfloat16) if projection_bias else None
    b_k = (torch.randn(kv_width) * 0.02).to(torch.bfloat16) if projection_bias else None
    b_v = (torch.randn(kv_width) * 0.02).to(torch.bfloat16) if projection_bias else None
    b_o = (torch.randn(out_features) * 0.02).to(torch.bfloat16) if projection_bias else None
    sinks = (torch.randn(hq) * 0.1).to(torch.bfloat16) if args.sink else None
    if decoder_block:
        num_experts = args.moe_num_experts
        top_k = args.moe_top_k
        router_weight = (torch.randn(num_experts, hidden) * 0.2).to(torch.bfloat16)
        router_bias = (torch.randn(num_experts) * 0.02).to(torch.bfloat16)
        gate_weight = (torch.randn(num_experts, hidden, moe_intermediate) * 0.2).to(torch.bfloat16)
        up_weight = (torch.randn(num_experts, hidden, moe_intermediate) * 0.2).to(torch.bfloat16)
        down_weight = (torch.randn(num_experts, moe_intermediate, hidden) * 0.2).to(torch.bfloat16)
        gate_bias = (torch.randn(num_experts, moe_intermediate) * 0.02).to(torch.bfloat16)
        up_bias = (torch.randn(num_experts, moe_intermediate) * 0.02).to(torch.bfloat16)
        down_bias = (torch.randn(num_experts, hidden) * 0.02).to(torch.bfloat16)
        split = SimpleNamespace(
            gate_weight=gate_weight,
            up_weight=up_weight,
            gate_bias=gate_bias,
            up_bias=up_bias,
        )
    if qwen_real_layer:
        real = _load_qwen3_layer_tensors(args.qwen_snapshot, args.layer_idx or 0)
        if real["w_q"].shape != (hidden, q_width):
            raise ValueError(f"Qwen q_proj shape {tuple(real['w_q'].shape)} != {(hidden, q_width)}")
        if real["w_k"].shape != (hidden, kv_width):
            raise ValueError(f"Qwen k_proj shape {tuple(real['w_k'].shape)} != {(hidden, kv_width)}")
        if real["w_v"].shape != (hidden, kv_width):
            raise ValueError(f"Qwen v_proj shape {tuple(real['w_v'].shape)} != {(hidden, kv_width)}")
        if real["w_o"].shape != (q_width, out_features):
            raise ValueError(f"Qwen o_proj shape {tuple(real['w_o'].shape)} != {(q_width, out_features)}")
        if real["router_weight"].shape != (num_experts, hidden):
            raise ValueError(f"Qwen router shape {tuple(real['router_weight'].shape)} != {(num_experts, hidden)}")
        if real["gate_weight"].shape != (num_experts, hidden, moe_intermediate):
            raise ValueError(
                f"Qwen gate shape {tuple(real['gate_weight'].shape)} != {(num_experts, hidden, moe_intermediate)}"
            )
        if real["up_weight"].shape != (num_experts, hidden, moe_intermediate):
            raise ValueError(
                f"Qwen up shape {tuple(real['up_weight'].shape)} != {(num_experts, hidden, moe_intermediate)}"
            )
        if real["down_weight"].shape != (num_experts, moe_intermediate, hidden):
            raise ValueError(
                f"Qwen down shape {tuple(real['down_weight'].shape)} != {(num_experts, moe_intermediate, hidden)}"
            )
        w_q = real["w_q"]
        w_k = real["w_k"]
        w_v = real["w_v"]
        w_o = real["w_o"]
        norm_weight = real["norm_weight"]
        norm_weight_rows = norm_weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16)
        q_norm_weight = real["q_norm_weight"]
        k_norm_weight = real["k_norm_weight"]
        q_norm_weight_rows = q_norm_weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16)
        k_norm_weight_rows = k_norm_weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16)
        ffn_norm_weight = real["ffn_norm_weight"]
        ffn_norm_weight_rows = ffn_norm_weight.reshape(1, -1).repeat(seq, 1).to(torch.bfloat16)
        router_weight = real["router_weight"]
        router_bias = real["router_bias"]
        gate_weight = real["gate_weight"]
        up_weight = real["up_weight"]
        down_weight = real["down_weight"]
        gate_bias = real["gate_bias"]
        up_bias = real["up_bias"]
        down_bias = real["down_bias"]
        split = SimpleNamespace(
            gate_weight=gate_weight,
            up_weight=up_weight,
            gate_bias=gate_bias,
            up_bias=up_bias,
        )
        qwen_real_metadata = {
            "snapshot": real["snapshot"],
            "layer_idx": real["layer_idx"],
        }

    projection_x = _weighted_rms_norm_golden(x, norm_weight_rows, norm_eps) if attention_block else x
    q_pre = torch.matmul(projection_x.float(), w_q.float()).to(torch.bfloat16)
    k_pre = torch.matmul(projection_x.float(), w_k.float()).to(torch.bfloat16)
    v_pre = torch.matmul(projection_x.float(), w_v.float()).to(torch.bfloat16)
    if "q" in bias_parts:
        q_pre = (q_pre.float() + b_q.float()).to(torch.bfloat16)
    if "k" in bias_parts:
        k_pre = (k_pre.float() + b_k.float()).to(torch.bfloat16)
    if "v" in bias_parts:
        v_pre = (v_pre.float() + b_v.float()).to(torch.bfloat16)
    q = q_pre.reshape(seq, hq, head_dim)
    k = k_pre.reshape(seq, hkv, head_dim)
    v = v_pre.reshape(seq, hkv, head_dim)
    if args.qk_head_norm:
        q = _head_rms_norm_golden(q, q_norm_weight, qk_norm_eps)
        k = _head_rms_norm_golden(k, k_norm_weight, qk_norm_eps)
    cos_head, sin_head = _make_rope_tables(seq, head_dim, theta=args.rope_theta)
    q = _apply_rope_bf16(q, cos_head, sin_head)
    k = _apply_rope_bf16(k, cos_head, sin_head)
    scale = 1.0 / math.sqrt(head_dim)
    attn_golden = _gpt_oss_attention_ref(q, k, v, sinks, scale=scale, sliding_window=sliding_window)
    golden = torch.matmul(attn_golden.float(), w_o.float()).to(torch.bfloat16)
    if "o" in bias_parts:
        golden = (golden.float() + b_o.float()).to(torch.bfloat16)
    if attention_block:
        golden = _residual_add_golden(x, golden)
    if decoder_block:
        post_attn = golden
        moe_input = _weighted_rms_norm_golden(post_attn, ffn_norm_weight_rows, norm_eps)
        router_logits = torch.matmul(moe_input.float(), router_weight.t().float()).to(torch.bfloat16)
        router_logits = (router_logits.float() + router_bias.reshape(1, -1).float()).to(torch.bfloat16)
        top_values, top_indices = torch.topk(router_logits, k=top_k, dim=-1)
        top_weights = torch.softmax(top_values, dim=1, dtype=top_values.dtype).to(torch.bfloat16)
        moe_golden, moe_clamp_counts = _device_routing_vram_policy_golden(
            x=moe_input,
            device_indices=top_indices,
            device_weights=top_weights,
            split=split,
            down_weight=down_weight,
            down_bias=down_bias,
            rows=seq,
            hidden=hidden,
            blen=blen,
            mlen=mlen,
            activation_policy=moe_activation_policy,
        )
        golden = _residual_add_golden(post_attn, moe_golden)
    else:
        moe_clamp_counts = None

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=2.0)
    physical_rows = max(mlen, math.ceil(seq / blen) * blen)
    x_region = physical_rows * hidden
    if attention_block:
        residual_base = _align_to_tile(x_region, mlen)
        residual_size = _align_to_tile(physical_rows * hidden, mlen)
        norm_weight_base = _align_to_tile(residual_base + residual_size, mlen)
        norm_weight_size = _align_to_tile(physical_rows * hidden, mlen)
        preload_cursor = _align_to_tile(norm_weight_base + norm_weight_size, mlen)
    else:
        residual_base = None
        norm_weight_base = None
        preload_cursor = _align_to_tile(x_region, mlen)
    q_norm_weight_base = k_norm_weight_base = None
    if args.qk_head_norm:
        q_norm_weight_base = _align_to_tile(preload_cursor, mlen)
        q_norm_weight_size = _align_to_tile(physical_rows * mlen, mlen)
        k_norm_weight_base = _align_to_tile(q_norm_weight_base + q_norm_weight_size, mlen)
        k_norm_weight_size = _align_to_tile(physical_rows * mlen, mlen)
        preload_cursor = _align_to_tile(k_norm_weight_base + k_norm_weight_size, mlen)
    ffn_norm_weight_base = router_w_base = router_bias_base = None
    gate_bias_base = up_bias_base = down_bias_base = None
    if decoder_block:
        ffn_norm_weight_base = _align_to_tile(preload_cursor, mlen)
        ffn_norm_weight_size = _align_to_tile(physical_rows * hidden, mlen)
        router_w_base = _align_to_tile(ffn_norm_weight_base + ffn_norm_weight_size, mlen)
        router_w_physical = (num_experts, hidden)
        router_w_size = _align_to_tile(router_w_physical[0] * router_w_physical[1], mlen)
        router_bias_base = _align_to_tile(router_w_base + router_w_size, mlen)
        router_bias_rows = _router_bias_block_rows(router_bias, rows=seq, num_experts=num_experts, mlen=mlen)
        router_bias_physical = (
            max(blen, math.ceil(router_bias_rows.shape[0] / blen) * blen),
            mlen,
        )
        router_bias_size = _align_to_tile(router_bias_physical[0] * router_bias_physical[1], mlen)
        gate_bias_base = _align_to_tile(router_bias_base + router_bias_size, mlen)
        gate_bias_physical = (num_experts * blen, moe_intermediate)
        gate_bias_size = _align_to_tile(gate_bias_physical[0] * gate_bias_physical[1], mlen)
        up_bias_base = _align_to_tile(gate_bias_base + gate_bias_size, mlen)
        up_bias_physical = (num_experts * blen, moe_intermediate)
        up_bias_size = _align_to_tile(up_bias_physical[0] * up_bias_physical[1], mlen)
        down_bias_base = _align_to_tile(up_bias_base + up_bias_size, mlen)
        down_bias_physical = (num_experts * blen, hidden)
        down_bias_size = _align_to_tile(down_bias_physical[0] * down_bias_physical[1], mlen)
        preload_cursor = _align_to_tile(down_bias_base + down_bias_size, mlen)
    bias_base = _align_to_tile(preload_cursor, mlen)
    bias_plan: list[tuple[str, torch.Tensor, tuple[int, int]]] = []
    if projection_bias:
        if "q" in bias_parts:
            bias_plan.append(("B_Q_TRUE", b_q.reshape(1, -1).repeat(seq, 1), (physical_rows, q_width)))
        if "k" in bias_parts:
            bias_plan.append(("B_K_TRUE", b_k.reshape(1, -1).repeat(seq, 1), (physical_rows, kv_width)))
        if "v" in bias_parts:
            bias_plan.append(("B_V_TRUE", b_v.reshape(1, -1).repeat(seq, 1), (physical_rows, kv_width)))
        if "o" in bias_parts:
            bias_plan.append(("B_O_TRUE", b_o.reshape(1, -1).repeat(seq, 1), (physical_rows, out_features)))
    bias_total = sum(_align_to_tile(shape[0] * shape[1], mlen) for _, _, shape in bias_plan)
    rope_base = bias_base + bias_total
    trig_shape = (physical_rows, mlen)
    trig_size = _align_to_tile(trig_shape[0] * trig_shape[1], mlen)
    vram_words = max(x_region + mlen * mlen, rope_base + 2 * trig_size)
    vram_preload = torch.zeros(vram_words, dtype=torch.bfloat16)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="X_true",
        tensor=x,
        vram_addr=0,
        physical_shape=(physical_rows, hidden),
        vram_preload=vram_preload,
    )
    residual_vram = None
    norm_weight_vram = None
    q_norm_weight_vram = None
    k_norm_weight_vram = None
    ffn_norm_weight_vram = None
    router_w_vram = None
    router_w_matrix_input = None
    router_w_packed_skinny_input = None
    router_bias_vram = None
    gate_bias_table = None
    up_bias_table = None
    down_bias_table = None
    if attention_block:
        residual_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="X_residual_true",
            tensor=x,
            vram_addr=residual_base,
            physical_shape=(physical_rows, hidden),
            vram_preload=vram_preload,
        )
        norm_weight_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="AttentionNormWeight",
            tensor=norm_weight_rows,
            vram_addr=norm_weight_base,
            physical_shape=(physical_rows, hidden),
            vram_preload=vram_preload,
        )
    if args.qk_head_norm:
        q_norm_weight_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="QHeadNormWeight",
            tensor=q_norm_weight_rows,
            vram_addr=q_norm_weight_base,
            physical_shape=(physical_rows, mlen),
            vram_preload=vram_preload,
        )
        k_norm_weight_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="KHeadNormWeight",
            tensor=k_norm_weight_rows,
            vram_addr=k_norm_weight_base,
            physical_shape=(physical_rows, mlen),
            vram_preload=vram_preload,
        )
    if decoder_block:
        ffn_norm_weight_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="FFNNormWeight",
            tensor=ffn_norm_weight_rows,
            vram_addr=ffn_norm_weight_base,
            physical_shape=(physical_rows, hidden),
            vram_preload=vram_preload,
        )
        router_w_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="DecoderRouterWBF16",
            tensor=router_weight,
            vram_addr=router_w_base,
            physical_shape=router_w_physical,
            vram_preload=vram_preload,
        )
        router_bias_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="DecoderRouterBias",
            tensor=router_bias_rows,
            vram_addr=router_bias_base,
            physical_shape=router_bias_physical,
            vram_preload=vram_preload,
        )
        gate_bias_table = _build_true_expert_bias_table(
            prog=prog,
            name="DecoderGateBiasTable",
            bias=gate_bias,
            width=moe_intermediate,
            blen=blen,
            vram_addr=gate_bias_base,
            vram_preload=vram_preload,
        )
        up_bias_table = _build_true_expert_bias_table(
            prog=prog,
            name="DecoderUpBiasTable",
            bias=up_bias,
            width=moe_intermediate,
            blen=blen,
            vram_addr=up_bias_base,
            vram_preload=vram_preload,
        )
        down_bias_table = _build_true_expert_bias_table(
            prog=prog,
            name="DecoderDownBiasTable",
            bias=down_bias,
            width=hidden,
            blen=blen,
            vram_addr=down_bias_base,
            vram_preload=vram_preload,
        )
        norm_weight_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="AttentionNormWeight",
            tensor=norm_weight_rows,
            vram_addr=norm_weight_base,
            physical_shape=(physical_rows, hidden),
            vram_preload=vram_preload,
        )
    bias_vrams = {}
    next_bias_addr = bias_base
    for name, tensor, physical_shape in bias_plan:
        bias_vrams[name] = prestage_bf16_vram_matrix(
            prog=prog,
            name=name,
            tensor=tensor,
            vram_addr=next_bias_addr,
            physical_shape=physical_shape,
            vram_preload=vram_preload,
        )
        next_bias_addr += _align_to_tile(physical_shape[0] * physical_shape[1], mlen)
    rotate, cos_packed, sin_packed = _make_packed_rope_inputs(seq, 1, head_dim, args.rope_theta)
    cos_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="ROPE_COS_TRUE_FULL",
        tensor=cos_packed,
        vram_addr=rope_base,
        physical_shape=trig_shape,
        vram_preload=vram_preload,
    )
    sin_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="ROPE_SIN_TRUE_FULL",
        tensor=sin_packed,
        vram_addr=rope_base + trig_size,
        physical_shape=trig_shape,
        vram_preload=vram_preload,
    )

    projection_input = x_vram
    if attention_block:
        prog.rms_norm(projection_input, eps_offset=norm_eps_offset, reci_hid_offset=norm_reci_hid_offset)
        prog.vram_mul(projection_input, norm_weight_vram, num_rows=seq)

    (
        (w_q_input, w_k_input, w_v_input, w_o_input, rotate_input),
        input_tensors,
        tensor_layouts,
    ) = _make_true_attention_hbm_inputs(
        prog,
        hidden=hidden,
        q_width=q_width,
        kv_width=kv_width,
        out_features=out_features,
        mlen=mlen,
        w_q=w_q,
        w_k=w_k,
        w_v=w_v,
        w_o=w_o,
        rotate=rotate,
    )
    q_var, k_var, v_var = _emit_true_qkv_projection_bf16(
        prog,
        projection_input,
        w_q_input=w_q_input,
        w_k_input=w_k_input,
        w_v_input=w_v_input,
        bias_vrams=bias_vrams,
        bias_parts=bias_parts,
        physical_rows=physical_rows,
        q_width=q_width,
        kv_width=kv_width,
        seq=seq,
    )

    weight_templates = None
    weight_table_bases = None
    weight_table_strides = None
    if decoder_block:
        gate_inputs, gate_table_base, gate_stride = _build_true_expert_weight_table(
            prog,
            prefix="DecoderWGate",
            weights=gate_weight,
            input_tensors=input_tensors,
        )
        up_inputs, up_table_base, up_stride = _build_true_expert_weight_table(
            prog,
            prefix="DecoderWUp",
            weights=up_weight,
            input_tensors=input_tensors,
        )
        down_inputs, down_table_base, down_stride = _build_true_expert_weight_table(
            prog,
            prefix="DecoderWDown",
            weights=down_weight,
            input_tensors=input_tensors,
        )
        weight_templates = (gate_inputs[0], up_inputs[0], down_inputs[0])
        weight_table_bases = (gate_table_base, up_table_base, down_table_base)
        weight_table_strides = (gate_stride, up_stride, down_stride)
        tensor_layouts.update(
            infer_hbm_tensor_layouts(
                {
                    name: tensor
                    for name, tensor in input_tensors.items()
                    if name.startswith(("DecoderWGate", "DecoderWUp", "DecoderWDown"))
                }
            )
        )
        if qwen_router_matrix_bf16:
            router_w_matrix_input = prog.input(
                "DecoderRouterWMatrixBF16",
                shape=(hidden, num_experts),
                real_data_ratio=2.0,
            )
            input_tensors["DecoderRouterWMatrixBF16"] = router_weight.t().contiguous()
            tensor_layouts["DecoderRouterWMatrixBF16"] = _bf16_layout(hidden, num_experts)
        if qwen_router_packed_skinny_bf16:
            packed_router_weight = _pack_qwen_router_weight_skinny(
                router_weight.t().contiguous(),
                mlen=mlen,
                blen=blen,
                k_tiles_per_packed_tile=args.qwen_router_packed_skinny_k_tiles,
            )
            router_w_packed_skinny_input = prog.input(
                "DecoderRouterWPackedSkinnyBF16",
                shape=tuple(packed_router_weight.shape),
                physical_shape=tuple(packed_router_weight.shape),
                real_data_ratio=2.0,
            )
            input_tensors["DecoderRouterWPackedSkinnyBF16"] = packed_router_weight
            tensor_layouts["DecoderRouterWPackedSkinnyBF16"] = _bf16_layout(*packed_router_weight.shape)
    k_inputs, v_inputs = _stage_true_kv_heads_bf16(
        prog,
        k_var=k_var,
        v_var=v_var,
        rotate_input=rotate_input,
        cos_vram=cos_vram,
        sin_vram=sin_vram,
        k_norm_weight_vram=k_norm_weight_vram if args.qk_head_norm else None,
        qk_head_norm=args.qk_head_norm,
        qk_norm_eps_offset=qk_norm_eps_offset if args.qk_head_norm else None,
        qk_norm_reci_head_offset=qk_norm_reci_head_offset if args.qk_head_norm else None,
        seq=seq,
        mlen=mlen,
        hkv=hkv,
        physical_rows=physical_rows,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
    )

    attn_out = prog.alloc("O_true_full_attention", seq, q_width, strict=False, physical_shape=(physical_rows, q_width))
    scratch = prog.alloc("_gpt_oss_true_full_scratch", mlen, mlen, strict=True)
    if sliding_window is not None:
        causal_mask = prog._build_sliding_causal_score_mask("_gpt_oss_true_full_sliding_mask", sliding_window)
    else:
        causal_mask = prog._build_causal_score_mask("_gpt_oss_true_full_causal_mask")

    fp_preload = [0.0, scale / 0.25, float("-inf")] + [0.0] * 253
    if args.qk_head_norm:
        if len(fp_preload) <= qk_norm_reci_head_offset:
            fp_preload.extend([0.0] * (qk_norm_reci_head_offset + 1 - len(fp_preload)))
        fp_preload[qk_norm_eps_offset] = qk_norm_eps
        fp_preload[qk_norm_reci_head_offset] = 1.0 / head_dim
    if attention_block:
        if len(fp_preload) <= norm_reci_hid_offset:
            fp_preload.extend([0.0] * (norm_reci_hid_offset + 1 - len(fp_preload)))
        fp_preload[norm_eps_offset] = norm_eps
        fp_preload[norm_reci_hid_offset] = 1.0 / hidden
    sink_base = None
    if sinks is not None:
        sink_base = 256
        if len(fp_preload) < sink_base + hq:
            fp_preload.extend([0.0] * (sink_base + hq - len(fp_preload)))
        for idx, value in enumerate(sinks.tolist()):
            fp_preload[sink_base + idx] = float(value)

    debug_stop_after_first_q_norm = bool(getattr(args, "debug_stop_after_first_q_norm", False))
    debug_stop_after_first_q_head = bool(getattr(args, "debug_stop_after_first_q_head", False))
    out, debug_golden, debug_stop_active = _emit_true_q_heads_attention_bf16(
        prog,
        q_var=q_var,
        rotate_input=rotate_input,
        cos_vram=cos_vram,
        sin_vram=sin_vram,
        q_norm_weight_vram=q_norm_weight_vram if args.qk_head_norm else None,
        qk_head_norm=args.qk_head_norm,
        qk_norm_eps_offset=qk_norm_eps_offset if args.qk_head_norm else None,
        qk_norm_reci_head_offset=qk_norm_reci_head_offset if args.qk_head_norm else None,
        seq=seq,
        mlen=mlen,
        hq=hq,
        hkv=hkv,
        head_dim=head_dim,
        physical_rows=physical_rows,
        k_inputs=k_inputs,
        v_inputs=v_inputs,
        attn_out=attn_out,
        scratch=scratch,
        causal_mask=causal_mask,
        scale=scale,
        sink_base=sink_base,
        q_pre=q_pre,
        q_norm_weight=q_norm_weight if args.qk_head_norm else None,
        qk_norm_eps=qk_norm_eps,
        q=q,
        debug_stop_after_first_q_norm=debug_stop_after_first_q_norm,
        debug_stop_after_first_q_head=debug_stop_after_first_q_head,
    )
    if debug_stop_active:
        golden = debug_golden
    if not debug_stop_active:
        out = _emit_true_output_projection_bf16(
            prog,
            attn_out,
            w_o_input=w_o_input,
            bias_vrams=bias_vrams,
            bias_parts=bias_parts,
            physical_rows=physical_rows,
            out_features=out_features,
            seq=seq,
        )
        if attention_block:
            prog.vram_add(out, residual_vram, num_rows=seq)
    if decoder_block and not debug_stop_active:
        # The attention path uses raw FPRAM offsets for scale/-inf, RMSNorm
        # constants, and per-head sinks (slots 1/2, 64/65, and 256+).  Reserve
        # the low region before allocating MoE FP variables so the second
        # sublayer cannot silently overwrite those constants.  Q/K head norm
        # also owns 512/513, so the decoder FFN norm must start after those
        # slots.  Slot 0 remains the true zero row expected by vram_fill_zero.
        zero = prog.fp_var("decoder_zero", size=1)
        _decoder_reserved_fp = prog.fp_var("decoder_reserved_attention_fp_slots", size=qk_norm_reci_head_offset)
        ffn_norm_eps_var = prog.fp_var("decoder_ffn_norm_eps", size=1)
        ffn_norm_reci_hid_var = prog.fp_var("decoder_ffn_norm_reci_hidden", size=1)
        ffn_norm_eps_offset = ffn_norm_eps_var.address
        ffn_norm_reci_hid_offset = ffn_norm_reci_hid_var.address
        constant_rows = blen
        limit_pos = prog.fp_var("decoder_gpt_oss_limit_pos", size=constant_rows)
        limit_neg = prog.fp_var("decoder_gpt_oss_limit_neg", size=constant_rows)
        one = prog.fp_var("decoder_one", size=constant_rows)
        neg_alpha = prog.fp_var("decoder_neg_alpha", size=constant_rows)
        shared_zero_row = prog.fp_var("decoder_shared_zero_row", size=mlen)

        moe_residual = prog.alloc(
            "DecoderMoeResidual",
            rows=seq,
            cols=hidden,
            strict=False,
            physical_shape=(physical_rows, hidden),
        )
        prog.moe_true_zero_vram_rows_v0(
            moe_residual,
            rows=list(range(seq)),
            hidden=hidden,
            zero_row=shared_zero_row,
            policy_name=moe_policy_name,
            name="decoder_moe_residual_zero",
        )
        prog.vram_add(moe_residual, out, num_rows=seq)

        prog.rms_norm(out, eps_offset=ffn_norm_eps_offset, reci_hid_offset=ffn_norm_reci_hid_offset)
        prog.vram_mul(out, ffn_norm_weight_vram, num_rows=seq)
        decoder_moe_input_var = out
        if qwen_router_packed_skinny_bf16:
            logits = prog.qwen3_router_logits_packed_skinny_bf16_rowpacked_v0(
                out,
                router_w_packed_skinny_input,
                rows=seq,
                hidden=hidden,
                num_experts=num_experts,
                k_tiles_per_packed_tile=args.qwen_router_packed_skinny_k_tiles,
                name="decoder_router_logits",
            )
        elif qwen_router_matrix_bf16:
            logits = prog.qwen3_router_logits_matrix_bf16_rowpacked_v0(
                out,
                router_w_matrix_input,
                rows=seq,
                hidden=hidden,
                num_experts=num_experts,
                mram_tile_capacity=args.qwen_router_mram_tile_capacity,
                name="decoder_router_logits",
            )
        else:
            logits = prog.moe_router_logits_bf16_v0(
                out,
                router_w_vram,
                rows=seq,
                hidden=hidden,
                num_experts=num_experts,
                policy_name=moe_policy_name,
                name="decoder_router_logits",
            )
        prog.vram_add(logits, router_bias_vram, num_rows=logits.shape[0])
        topk_weight_var = prog.fp_var("decoder_topk_weights", size=seq * top_k)
        topk_weights_fp_base = topk_weight_var.address
        topk_indices_int_base = 0
        for token_idx in range(seq):
            prog.moe_router_select_v0(
                logits,
                token_idx=token_idx,
                weights_fp_base=topk_weights_fp_base + token_idx * top_k,
                indices_int_base=topk_indices_int_base + token_idx * top_k,
                policy_name=moe_policy_name,
                num_experts=num_experts,
                top_k=top_k,
                name=f"decoder_token{token_idx}",
            )

        accumulator = prog.alloc(
            "DecoderMoeAccumulator",
            rows=seq,
            cols=hidden,
            strict=False,
            physical_shape=(physical_rows, hidden),
        )
        prog.moe_true_zero_vram_rows_v0(
            accumulator,
            rows=list(range(seq)),
            hidden=hidden,
            zero_row=shared_zero_row,
            policy_name=moe_policy_name,
            name="decoder_moe_acc_zero",
        )
        route_fp_scratch = prog.fp_var("decoder_route_fp_scratch", size=mlen)
        pair_count = seq * top_k
        for pair_idx in range(pair_count):
            token_idx = pair_idx // top_k
            gathered = prog.moe_gather_token_rows_from_vram_v0(
                out,
                token_indices=[token_idx],
                hidden=hidden,
                zero_row=shared_zero_row,
                policy_name=moe_policy_name,
                name=f"decoder_pair{pair_idx}_vram_gather_t{token_idx}",
            )
            expert_out = prog.moe_dynamic_expert_pair_v0(
                gathered,
                weight_templates,
                weight_table_bases=weight_table_bases,
                weight_table_strides=weight_table_strides,
                expert_indices_int_base=topk_indices_int_base,
                weights_fp_base=topk_weights_fp_base,
                pair_idx=pair_idx,
                bias_tables=(gate_bias_table, up_bias_table, down_bias_table),
                rows=blen,
                intermediate=moe_intermediate,
                constants=(zero, limit_pos, limit_neg, one, neg_alpha),
                zero_row=shared_zero_row,
                route_fp_scratch=route_fp_scratch,
                policy_name=moe_policy_name,
                activation_policy=moe_activation_policy,
                name=f"decoder_pair{pair_idx}",
            )
            prog.moe_scatter_add_active_rows_v0(
                accumulator,
                expert_out,
                token_indices=[token_idx],
                active_rows=[0],
                hidden=hidden,
                policy_name=moe_policy_name,
                name=f"decoder_pair{pair_idx}_scatter",
            )
        prog.vram_add(accumulator, moe_residual, num_rows=seq)
        decoder_moe_residual_var = moe_residual
        out = accumulator
        decoder_fp_end = max(
            ffn_norm_reci_hid_var.address + ffn_norm_reci_hid_var.size,
            topk_weight_var.address + topk_weight_var.size,
            route_fp_scratch.address + route_fp_scratch.size,
            shared_zero_row.address + shared_zero_row.size,
            neg_alpha.address + neg_alpha.size,
        )
        if len(fp_preload) < decoder_fp_end:
            fp_preload.extend([0.0] * (decoder_fp_end - len(fp_preload)))
        fp_preload[ffn_norm_eps_offset] = norm_eps
        fp_preload[ffn_norm_reci_hid_offset] = 1.0 / hidden
        for idx in range(limit_pos.size):
            fp_preload[limit_pos.address + idx] = 7.0
        for idx in range(limit_neg.size):
            fp_preload[limit_neg.address + idx] = -7.0
        for idx in range(one.size):
            fp_preload[one.address + idx] = 1.0
        for idx in range(neg_alpha.size):
            fp_preload[neg_alpha.address + idx] = -1.0 if moe_activation_policy == "standard_swiglu" else -1.702
    isa = prog.compile()
    if "C_SET_SCALE_REG" in isa and not decoder_block:
        isa_lines = isa.splitlines()
        non_store_scale_lines = []
        for idx, line in enumerate(isa_lines):
            if "C_SET_SCALE_REG" not in line:
                continue
            context = isa_lines[max(0, idx - 4) : idx]
            if not any("Store Activation Generation" in item for item in context):
                non_store_scale_lines.append(line)
        if non_store_scale_lines:
            (build_dir / "generated_asm_code.debug.asm").write_text(isa)
            raise AssertionError(
                "true_full BF16 projection/attention path emitted non-store C_SET_SCALE_REG: "
                + "; ".join(non_store_scale_lines[:8])
            )

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    data_order = sorted(input_tensors, key=lambda name: hbm_addrs[name])
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_attention_true_full",
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    compare_out_features = out.shape[1]
    params = _comparison_params(
        prog.get_vram_addr(out.name),
        seq,
        compare_out_features,
        mlen,
        physical_rows=out.physical_shape[0],
    )
    _write_json(build_dir / "comparison_params.json", params)
    (build_dir / "generated_asm_code.asm").write_text(isa)

    strict_summary = None
    decoder_runtime_summary = None
    if not args.no_run:
        if args.probe_rel_only or attention_block:
            run_emulator(build_dir)
        else:
            run_and_assert(build_dir, "gpt_oss_attention_true_full", mlen=mlen, blen=blen)
        results, _ = compare_emulator_output(build_dir)
        emu = results["simulated_values"].reshape(seq, compare_out_features).to(torch.bfloat16)
        rel = _rel_rms(emu, golden)
        strict_summary = _strict_tail_summary(emu, golden)
        if getattr(args, "save_golden_pt", None) is not None:
            args.save_golden_pt.expanduser().parent.mkdir(parents=True, exist_ok=True)
            torch.save(golden.float(), args.save_golden_pt.expanduser())
        if getattr(args, "save_output_pt", None) is not None:
            args.save_output_pt.expanduser().parent.mkdir(parents=True, exist_ok=True)
            torch.save(emu.float(), args.save_output_pt.expanduser())
        if decoder_block:
            device_indices_flat = _decode_u32_dump(build_dir / "intsram_dump.bin")[
                topk_indices_int_base : topk_indices_int_base + seq * top_k
            ]
            device_weights_flat = _decode_bf16_dump(build_dir / "fpsram_dump.bin")[
                topk_weights_fp_base : topk_weights_fp_base + seq * top_k
            ]
            device_indices = device_indices_flat.reshape(seq, top_k)
            device_weights = device_weights_flat.reshape(seq, top_k).to(torch.bfloat16)
            vram_dump = build_dir / "vram_dump.bin"
            device_moe_input = _read_vram_matrix(
                vram_dump,
                vram_addr=prog.get_vram_addr(decoder_moe_input_var.name),
                rows=seq,
                cols=hidden,
                mlen=mlen,
                physical_rows=decoder_moe_input_var.physical_shape[0],
            )
            device_moe_residual = _read_vram_matrix(
                vram_dump,
                vram_addr=prog.get_vram_addr(decoder_moe_residual_var.name),
                rows=seq,
                cols=hidden,
                mlen=mlen,
                physical_rows=decoder_moe_residual_var.physical_shape[0],
            )
            device_context_logits = torch.matmul(
                device_moe_input.float(),
                router_weight.t().float(),
            ).to(torch.bfloat16)
            device_context_logits = (device_context_logits.float() + router_bias.reshape(1, -1).float()).to(
                torch.bfloat16
            )
            context_values, context_indices = torch.topk(device_context_logits, k=top_k, dim=-1)
            context_weights = torch.softmax(context_values, dim=1, dtype=context_values.dtype).to(torch.bfloat16)
            device_selected_moe, device_selected_clamp_counts = _device_routing_vram_policy_golden(
                x=device_moe_input,
                device_indices=device_indices,
                device_weights=device_weights,
                split=split,
                down_weight=down_weight,
                down_bias=down_bias,
                rows=seq,
                hidden=hidden,
                blen=blen,
                mlen=mlen,
                activation_policy=moe_activation_policy,
            )
            device_selected_final = _residual_add_golden(device_moe_residual, device_selected_moe)
            torch.save(device_selected_final.float(), build_dir / "device_selected_decoder_golden.pt")
            torch.save(device_moe_input.float(), build_dir / "device_moe_input.pt")
            torch.save(device_moe_residual.float(), build_dir / "device_post_attention_residual.pt")
            device_selected_strict = _strict_tail_summary(emu, device_selected_final)
            decoder_runtime_summary = {
                "topk_weights_fp_base": topk_weights_fp_base,
                "topk_indices_int_base": topk_indices_int_base,
                "device_selected_exact_rel_rms": _rel_rms(emu, device_selected_final),
                "device_selected_exact_strict_tail": device_selected_strict,
                "host_topk_vs_device_topk": _topk_match_summary(
                    host_indices=top_indices,
                    host_weights=top_weights,
                    device_indices=device_indices,
                    device_weights=device_weights,
                ),
                "host_router_margin": _router_margin_summary(router_logits, top_k),
                "device_context_topk_vs_device_topk": _topk_match_summary(
                    host_indices=context_indices,
                    host_weights=context_weights,
                    device_indices=device_indices,
                    device_weights=device_weights,
                ),
                "host_vs_device_routing_boundary": _routing_boundary_verdict(
                    reference_logits=router_logits,
                    comparison_logits=device_context_logits,
                    reference_indices=top_indices,
                    device_indices=device_indices,
                    top_k=top_k,
                ),
                "device_context_vs_device_routing_boundary": _routing_boundary_verdict(
                    reference_logits=device_context_logits,
                    comparison_logits=device_context_logits,
                    reference_indices=context_indices,
                    device_indices=device_indices,
                    top_k=top_k,
                ),
                "device_context_router_margin": _router_margin_summary(device_context_logits, top_k),
                "device_context_vs_host_router_logits_rel_rms": _rel_rms(device_context_logits, router_logits),
                "device_context_vs_host_router_logits_max_abs_error": float(
                    (device_context_logits.float() - router_logits.float()).abs().max().item()
                ),
                "device_moe_input_vs_host_moe_input_rel_rms": _rel_rms(device_moe_input, moe_input),
                "device_post_attention_vs_host_post_attention_rel_rms": _rel_rms(
                    device_moe_residual,
                    post_attn,
                ),
                "device_selected_clamp_counts": device_selected_clamp_counts,
            }
        if rel > 0.02 and not args.probe_rel_only:
            device_selected_rel = (
                decoder_runtime_summary.get("device_selected_exact_rel_rms")
                if decoder_runtime_summary is not None
                else None
            )
            if device_selected_rel is None or not math.isfinite(device_selected_rel) or device_selected_rel > 0.02:
                raise AssertionError(f"true_full rel_rms={rel:.6g} exceeds 2%")
    else:
        rel = float("nan")

    summary = {
        "mode": mode_name,
        "seq": seq,
        "hidden": hidden,
        "out_features": out_features,
        "compare_out_features": compare_out_features,
        "debug_stop_after_first_q_head": bool(debug_stop_after_first_q_head and debug_stop_active),
        "debug_stop_after_first_q_norm": bool(debug_stop_after_first_q_norm and debug_stop_active),
        "num_attention_heads": hq,
        "num_key_value_heads": hkv,
        "head_dim": head_dim,
        "sink": bool(args.sink),
        "projection_bias": projection_bias,
        "bias_parts": "".join(sorted(bias_parts)),
        "runtime_projection": True,
        "runtime_bias": projection_bias,
        "runtime_rope": True,
        "qk_head_norm": bool(args.qk_head_norm),
        "qk_head_norm_eps": qk_norm_eps if args.qk_head_norm else None,
        "qk_norm_eps_offset": qk_norm_eps_offset if args.qk_head_norm else None,
        "qk_norm_reci_head_offset": qk_norm_reci_head_offset if args.qk_head_norm else None,
        "input_source": input_source,
        "qwen_real_layer": qwen_real_layer,
        "qwen_real_metadata": qwen_real_metadata,
        "saved_output_pt": str(args.save_output_pt.expanduser()) if getattr(args, "save_output_pt", None) else None,
        "saved_golden_pt": str(args.save_golden_pt.expanduser()) if getattr(args, "save_golden_pt", None) else None,
        "attention_block": attention_block,
        "decoder_block": decoder_block,
        "pre_norm": attention_block,
        "residual_add": attention_block,
        "norm_eps": norm_eps if attention_block or decoder_block else None,
        "norm_eps_offset": norm_eps_offset if attention_block else None,
        "norm_reci_hid_offset": norm_reci_hid_offset if attention_block else None,
        "ffn_norm_eps_offset": ffn_norm_eps_offset if decoder_block else None,
        "ffn_norm_reci_hid_offset": ffn_norm_reci_hid_offset if decoder_block else None,
        "projection_precision": "qkvo_bf16",
        "moe_input_source": "vram_bf16" if decoder_block else None,
        "moe_num_experts": args.moe_num_experts if decoder_block else None,
        "moe_top_k": args.moe_top_k if decoder_block else None,
        "moe_policy_name": moe_policy_name if decoder_block else None,
        "moe_activation_policy": moe_activation_policy if decoder_block else None,
        "moe_intermediate": moe_intermediate if decoder_block else None,
        "qwen_router_matrix_bf16": qwen_router_matrix_bf16 if decoder_block else None,
        "qwen_router_packed_skinny_bf16": qwen_router_packed_skinny_bf16 if decoder_block else None,
        "qwen_router_mram_tile_capacity": (args.qwen_router_mram_tile_capacity if qwen_router_matrix_bf16 else None),
        "qwen_router_packed_skinny_k_tiles": (
            args.qwen_router_packed_skinny_k_tiles if qwen_router_packed_skinny_bf16 else None
        ),
        "moe_clamp_counts_golden": moe_clamp_counts,
        "topk_weights_fp_base": topk_weights_fp_base if decoder_block else None,
        "topk_indices_int_base": topk_indices_int_base if decoder_block else None,
        "decoder_runtime_summary": decoder_runtime_summary,
        "sliding_window": sliding_window,
        "sliding_source": sliding_source,
        "rel_rms": rel,
        "sink_base": sink_base,
        "scale_reg_count": isa.count("C_SET_SCALE_REG"),
        "h_store_keyvalue_count": isa.count("H_STORE_V"),
        "strict_tail": strict_summary,
        "strategy": "runtime_qkvo_projection_unrolled_one_q_head_per_call",
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _load_hidden_tensor(path: Path) -> torch.Tensor:
    loaded = torch.load(path.expanduser(), map_location="cpu")
    if isinstance(loaded, dict):
        for key in ("hidden", "output", "tensor", "x"):
            if key in loaded:
                loaded = loaded[key]
                break
    if not torch.is_tensor(loaded):
        raise TypeError(f"{path} must contain a Tensor or tensor dict, got {type(loaded).__name__}")
    return loaded


def _clone_args(args: argparse.Namespace, **overrides) -> SimpleNamespace:
    values = vars(args).copy()
    values.update(overrides)
    return SimpleNamespace(**values)


def run_true_decoder_chain(args: argparse.Namespace) -> dict:
    """Run true decoder blocks sequentially with explicit tensor handoff.

    This is not a fused multi-layer PLENA program.  It is a validation harness
    that reuses the already-validated true_decoder_block program, feeds each
    layer's emulator output into the next layer, and runs a parallel host-chain
    control to expose cumulative drift and routing-margin sensitivity.
    """
    if args.no_run:
        raise ValueError("true_decoder_chain requires emulator execution; --no-run is not supported")
    if args.chain_layers < 1:
        raise ValueError(f"--chain-layers must be >= 1, got {args.chain_layers}")
    if args.mode != "true_decoder_chain":
        raise ValueError(f"run_true_decoder_chain expected mode=true_decoder_chain, got {args.mode}")

    root = (args.build_dir / "true_decoder_chain").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    device_input: Path | None = args.input_pt.expanduser() if getattr(args, "input_pt", None) else None
    host_input: Path | None = args.input_pt.expanduser() if getattr(args, "input_pt", None) else None
    layer_records = []
    host_control_records = []

    for layer_idx in range(args.chain_layers):
        layer_seed = args.seed + layer_idx
        device_dir = root / f"layer{layer_idx}_device_chain"
        device_output = device_dir / f"layer{layer_idx}_emu_output.pt"
        device_conditional_golden = device_dir / f"layer{layer_idx}_device_input_golden.pt"
        device_args = _clone_args(
            args,
            mode="true_decoder_block",
            build_dir=device_dir,
            layer_idx=layer_idx,
            seed=layer_seed,
            input_pt=device_input,
            save_output_pt=device_output,
            save_golden_pt=device_conditional_golden,
            probe_rel_only=True,
            no_run=False,
        )
        device_summary = run_true_full(device_args)

        record = {
            "layer_idx": layer_idx,
            "seed": layer_seed,
            "device_build_dir": str(device_dir),
            "device_input_pt": str(device_input) if device_input is not None else None,
            "device_output_pt": str(device_output),
            "device_conditional_golden_pt": str(device_conditional_golden),
            "device_summary": device_summary,
        }

        if layer_idx == 0:
            host_input = device_conditional_golden
            record["device_input_vs_host_input_rel_rms"] = 0.0
        else:
            host_dir = root / f"layer{layer_idx}_host_chain_control"
            host_output = host_dir / f"layer{layer_idx}_host_control_emu_output.pt"
            host_golden = host_dir / f"layer{layer_idx}_host_chain_golden.pt"
            if host_input is None:
                raise RuntimeError("host_input unexpectedly missing for chained host-control layer")
            host_args = _clone_args(
                args,
                mode="true_decoder_block",
                build_dir=host_dir,
                layer_idx=layer_idx,
                seed=layer_seed,
                input_pt=host_input,
                save_output_pt=host_output,
                save_golden_pt=host_golden,
                probe_rel_only=True,
                no_run=False,
            )
            host_summary = run_true_full(host_args)
            device_in_tensor = _load_hidden_tensor(device_input)
            host_in_tensor = _load_hidden_tensor(host_input)
            record["device_input_vs_host_input_rel_rms"] = _rel_rms(device_in_tensor, host_in_tensor)
            record["host_control_build_dir"] = str(host_dir)
            record["host_chain_golden_pt"] = str(host_golden)
            record["host_control_summary"] = host_summary
            host_control_records.append(
                {
                    "layer_idx": layer_idx,
                    "seed": layer_seed,
                    "host_build_dir": str(host_dir),
                    "host_input_pt": str(host_input),
                    "host_output_pt": str(host_output),
                    "host_golden_pt": str(host_golden),
                    "summary": host_summary,
                }
            )
            host_input = host_golden

        layer_records.append(record)
        device_input = device_output

    final_device_output = device_input
    final_host_golden = host_input
    final_device_vs_host_chain_rel = None
    if final_device_output is not None and final_host_golden is not None:
        final_device_vs_host_chain_rel = _rel_rms(
            _load_hidden_tensor(final_device_output),
            _load_hidden_tensor(final_host_golden),
        )

    summary = {
        "mode": "true_decoder_chain",
        "note": (
            "Sequential validation runner: each layer is a separate true_decoder_block "
            "emulator run, with emulator output handed to the next layer. This is not "
            "a fused multi-layer PLENA program."
        ),
        "chain_layers": args.chain_layers,
        "initial_input_pt": str(args.input_pt.expanduser()) if getattr(args, "input_pt", None) else None,
        "final_device_output_pt": str(final_device_output) if final_device_output is not None else None,
        "final_host_chain_golden_pt": str(final_host_golden) if final_host_golden is not None else None,
        "final_device_vs_host_chain_rel_rms": final_device_vs_host_chain_rel,
        "layers": layer_records,
        "host_controls": host_control_records,
    }
    _write_json(root / "chain_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_core(args: argparse.Namespace) -> dict:
    build_dir = (args.build_dir / "core").expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    hq = 4
    hkv = 1
    head_dim = args.hlen or (mlen // hq)
    seq = args.seq_len or 16
    if head_dim * hq != mlen:
        raise ValueError("core smoke expects hq*head_dim == MLEN")
    if seq > mlen:
        raise ValueError("core smoke is single sequence tile only")
    if seq % blen != 0:
        raise ValueError(f"seq={seq} must be a multiple of BLEN={blen}")
    sliding_window, sliding_source = _resolve_sliding_window(args)

    torch.manual_seed(args.seed)
    q = (torch.randn(seq, hq, head_dim) * 0.2).to(torch.bfloat16)
    k = (torch.randn(seq, hkv, head_dim) * 0.2).to(torch.bfloat16)
    v = (torch.randn(seq, hkv, head_dim) * 0.2).to(torch.bfloat16)
    sinks = (torch.randn(hq) * 0.1).to(torch.bfloat16) if args.sink else None
    scale = 1.0 / math.sqrt(head_dim)
    golden = _gpt_oss_attention_ref(q, k, v, sinks, scale=scale, sliding_window=sliding_window)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=2.0)
    rows_per_batch = max(mlen, seq)
    q_flat = q.reshape(seq, hq * head_dim)
    q_vram_flat = torch.zeros(rows_per_batch * mlen, dtype=torch.bfloat16)
    q_input = prog.input("Q", shape=(seq, mlen), prestaged_vram_addr=0, physical_shape=(rows_per_batch, mlen))
    q_group = prog.load_batch(q_input, name="Q")
    q_vram_flat[: seq * mlen] = q_flat.reshape(-1)

    kv_slots = mlen // head_dim
    k_padded = torch.zeros(rows_per_batch, kv_slots, head_dim, dtype=torch.bfloat16)
    v_padded = torch.zeros(rows_per_batch, kv_slots, head_dim, dtype=torch.bfloat16)
    k_padded[:seq, :hkv, :] = k
    v_padded[:seq, :hkv, :] = v
    k_input = prog.input("K", shape=(seq, mlen), physical_shape=(rows_per_batch, mlen), real_data_ratio=2.0)
    v_input = prog.input("V", shape=(seq, mlen), physical_shape=(rows_per_batch, mlen), real_data_ratio=2.0)

    o = prog.alloc("O", seq, mlen, strict=False, physical_shape=(rows_per_batch, mlen))
    scratch = prog.alloc("_gpt_oss_attn_scratch", mlen * kv_slots, mlen, strict=True)
    if sliding_window is not None:
        causal_mask = prog._build_sliding_causal_score_mask("_gpt_oss_sliding_mask", sliding_window)
    else:
        causal_mask = True

    fp_preload = [0.0, scale / 0.25, float("-inf")] + [0.0] * 253
    sink_base = None
    if sinks is not None:
        sink_base = 256
        if len(fp_preload) < sink_base + hq:
            fp_preload.extend([0.0] * (sink_base + hq - len(fp_preload)))
        for idx, value in enumerate(sinks.tolist()):
            fp_preload[sink_base + idx] = float(value)

    prog.flash_attention_packed_group(
        q_group,
        k_input,
        v_input,
        group_heads=hq,
        head_slot_dim=head_dim,
        output_base_address=prog.get_vram_addr(o.name),
        scratch_base_address=prog.get_vram_addr(scratch.name),
        broadcast_amount=kv_slots,
        scale=scale,
        causal_mask=causal_mask,
        valid_cols=seq,
        sink_base_address=sink_base,
        k_matrix_precision="keyvalue",
        k_set_scale=False,
        k_hbm_element_bytes=2,
        v_hbm_element_bytes=2,
    )
    isa = prog.compile()

    input_tensors = {
        "K": k_padded.reshape(rows_per_batch, mlen),
        "V": v_padded.reshape(rows_per_batch, mlen),
    }
    tensor_layouts = {
        name: {
            "source_shape": [rows_per_batch, mlen],
            "storage_shape": [rows_per_batch, mlen],
            "source_rows": rows_per_batch,
            "storage_rows": rows_per_batch,
            "source_row_elements": mlen,
            "storage_row_elements": mlen,
            "precision": "HBM_M_KV_TYPE",
        }
        for name in input_tensors
    }
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=q_vram_flat,
        tensor_layouts=tensor_layouts,
    )
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_attention_core",
        specified_data_order=list(input_tensors),
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    params = _comparison_params(prog.get_vram_addr(o.name), seq, mlen, mlen, physical_rows=rows_per_batch)
    _write_json(build_dir / "comparison_params.json", params)
    (build_dir / "generated_asm_code.asm").write_text(isa)

    if not args.no_run:
        run_and_assert(build_dir, "gpt_oss_attention_core", mlen=mlen, blen=blen)
        results, _ = compare_emulator_output(build_dir)
        emu = results["simulated_values"].reshape(seq, mlen).to(torch.bfloat16)
        rel = _rel_rms(emu, golden)
        if rel > 0.02:
            raise AssertionError(f"attention core rel_rms={rel:.6g} exceeds 2%")
    else:
        rel = float("nan")

    summary = {
        "mode": "core",
        "seq": seq,
        "sink": bool(args.sink),
        "sliding_window": sliding_window,
        "sliding_source": sliding_source,
        "rel_rms": rel,
        "sink_base": sink_base,
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _bf16_layout(rows: int, cols: int, *, precision: str = "HBM_M_KV_TYPE") -> dict:
    return {
        "source_shape": [rows, cols],
        "storage_shape": [rows, cols],
        "source_rows": rows,
        "storage_rows": rows,
        "source_row_elements": cols,
        "storage_row_elements": cols,
        "precision": precision,
    }


def _pack_qwen_router_weight_skinny(
    weight_hidden_by_expert: torch.Tensor,
    *,
    mlen: int,
    blen: int,
    k_tiles_per_packed_tile: int,
) -> torch.Tensor:
    """Pack ``[hidden, experts]`` router weight for the packed-skinny helper."""
    hidden, num_experts = weight_hidden_by_expert.shape
    if hidden % mlen != 0:
        raise ValueError(f"router hidden={hidden} must be divisible by MLEN={mlen}")
    if num_experts % blen != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by BLEN={blen}")
    tiles_per_mlen = mlen // blen
    if k_tiles_per_packed_tile <= 0 or k_tiles_per_packed_tile > tiles_per_mlen:
        raise ValueError(f"k_tiles_per_packed_tile={k_tiles_per_packed_tile} must be in 1..{tiles_per_mlen}")

    num_k_tiles = hidden // mlen
    num_groups = math.ceil(num_k_tiles / k_tiles_per_packed_tile)
    num_microcols = math.ceil(num_experts / blen)
    physical_col_blocks = max(tiles_per_mlen, num_microcols)
    packed = torch.zeros(num_groups * mlen, physical_col_blocks * mlen, dtype=torch.bfloat16)

    for group_idx in range(num_groups):
        for micro_col_idx in range(num_microcols):
            expert_start = micro_col_idx * blen
            expert_end = min(expert_start + blen, num_experts)
            row_start = group_idx * mlen
            col_base = micro_col_idx * mlen
            for local_k_tile in range(k_tiles_per_packed_tile):
                k_tile_idx = group_idx * k_tiles_per_packed_tile + local_k_tile
                if k_tile_idx >= num_k_tiles:
                    break
                k_start = k_tile_idx * mlen
                k_end = k_start + mlen
                skinny_col = col_base + local_k_tile * blen
                packed[row_start : row_start + mlen, skinny_col : skinny_col + (expert_end - expert_start)] = (
                    weight_hidden_by_expert[k_start:k_end, expert_start:expert_end].to(torch.bfloat16)
                )
    return packed.contiguous()


def run_full(args: argparse.Namespace) -> dict:
    build_dir = (args.build_dir / "full").expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    setup_hw(args, build_dir)
    _set_matrix_kv_plain_bf16()

    mlen = args.mlen
    blen = args.blen
    hq = 4
    hkv = 1
    head_dim = args.hlen or (mlen // hq)
    seq = args.seq_len or 16
    hidden = args.hidden_size or 128
    out_features = args.out_features or hidden
    if head_dim * hq != mlen:
        raise ValueError("full attention smoke expects hq*head_dim == MLEN")
    if seq > mlen:
        raise ValueError("full attention smoke is single sequence tile only")
    if seq % blen != 0:
        raise ValueError(f"seq={seq} must be a multiple of BLEN={blen}")
    if hidden % mlen != 0 or out_features % mlen != 0:
        raise ValueError("hidden and out_features must be multiples of MLEN")
    sliding_window, sliding_source = _resolve_sliding_window(args)
    vo_mx = bool(args.vo_mx)
    projection_bias = bool(args.projection_bias)
    runtime_rope = bool(args.runtime_rope)
    bias_parts = _bias_parts(args)

    torch.manual_seed(args.seed)
    x = (torch.randn(seq, hidden) * 0.2).to(torch.bfloat16)
    w_q = (torch.randn(hidden, mlen) * 0.2).to(torch.bfloat16)
    w_k = torch.zeros(hidden, mlen, dtype=torch.bfloat16)
    w_v = torch.zeros(hidden, mlen, dtype=torch.bfloat16)
    w_k[:, :head_dim] = (torch.randn(hidden, head_dim) * 0.2).to(torch.bfloat16)
    w_v[:, :head_dim] = (torch.randn(hidden, head_dim) * 0.2).to(torch.bfloat16)
    w_o = (torch.randn(mlen, out_features) * 0.2).to(torch.bfloat16)
    b_q = (torch.randn(mlen) * 0.02).to(torch.bfloat16) if projection_bias else None
    b_k = (torch.randn(mlen) * 0.02).to(torch.bfloat16) if projection_bias else None
    b_v = (torch.randn(mlen) * 0.02).to(torch.bfloat16) if projection_bias else None
    b_o = (torch.randn(out_features) * 0.02).to(torch.bfloat16) if projection_bias else None
    # K/V are staged in MLEN-wide physical rows, but this small diagnostic has
    # only one KV head.  Keep padded lanes zero so projection bias cannot create
    # non-model data in lanes outside the logical head_dim.
    if projection_bias:
        b_k[head_dim:] = 0
        b_v[head_dim:] = 0
    sinks = (torch.randn(hq) * 0.1).to(torch.bfloat16) if args.sink else None

    q_proj = torch.matmul(x.float(), w_q.float()).to(torch.bfloat16)
    k_proj = torch.matmul(x.float(), w_k.float()).to(torch.bfloat16)
    v_proj = torch.matmul(x.float(), w_v.float()).to(torch.bfloat16)
    if "q" in bias_parts:
        q_proj = (q_proj.float() + b_q.float()).to(torch.bfloat16)
    if "k" in bias_parts:
        k_proj = (k_proj.float() + b_k.float()).to(torch.bfloat16)
    if "v" in bias_parts:
        v_proj = (v_proj.float() + b_v.float()).to(torch.bfloat16)
    q = q_proj.reshape(seq, hq, head_dim)
    k = k_proj[:, :head_dim].reshape(seq, hkv, head_dim)
    v = v_proj[:, :head_dim].reshape(seq, hkv, head_dim)
    rotate = cos = sin = None
    if runtime_rope:
        rotate, cos, sin = _make_packed_rope_inputs(seq, hq, head_dim, args.rope_theta)
        cos_head = cos[:, :head_dim]
        sin_head = sin[:, :head_dim]
        q = _apply_rope_bf16(q, cos_head, sin_head)
        k = _apply_rope_bf16(k, cos_head, sin_head)
    scale = 1.0 / math.sqrt(head_dim)
    attn_golden = _gpt_oss_attention_ref(q, k, v, sinks, scale=scale, sliding_window=sliding_window)
    golden = torch.matmul(attn_golden.float(), w_o.float()).to(torch.bfloat16)
    if "o" in bias_parts:
        golden = (golden.float() + b_o.float()).to(torch.bfloat16)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=2.0)
    physical_rows = max(mlen, math.ceil(seq / blen) * blen)
    x_region = physical_rows * hidden
    bias_base = math.ceil(x_region / (mlen * mlen)) * (mlen * mlen)
    bias_plan: list[tuple[str, torch.Tensor, tuple[int, int]]] = []
    if projection_bias:
        if "q" in bias_parts:
            bias_plan.append(("B_Q", b_q.reshape(1, -1).repeat(seq, 1), (physical_rows, mlen)))
        if "k" in bias_parts:
            bias_plan.append(("B_K", b_k.reshape(1, -1).repeat(seq, 1), (physical_rows, mlen)))
        if "v" in bias_parts:
            bias_plan.append(("B_V", b_v.reshape(1, -1).repeat(seq, 1), (physical_rows, mlen)))
        if "o" in bias_parts:
            bias_plan.append(("B_O", b_o.reshape(1, -1).repeat(seq, 1), (physical_rows, out_features)))
    bias_total = sum(math.ceil((shape[0] * shape[1]) / (mlen * mlen)) * (mlen * mlen) for _, _, shape in bias_plan)
    rope_base = bias_base + bias_total
    trig_shape = (physical_rows, mlen)
    trig_size = _align_to_tile(trig_shape[0] * trig_shape[1], mlen)
    vram_words = max(x_region + mlen * mlen, bias_base + bias_total)
    if runtime_rope:
        vram_words = max(vram_words, rope_base + 2 * trig_size)
    vram_preload = torch.zeros(vram_words, dtype=torch.bfloat16)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="X",
        tensor=x,
        vram_addr=0,
        physical_shape=(physical_rows, hidden),
        vram_preload=vram_preload,
    )
    bias_vrams = {}
    next_bias_addr = bias_base
    for name, tensor, physical_shape in bias_plan:
        bias_vrams[name] = prestage_bf16_vram_matrix(
            prog=prog,
            name=name,
            tensor=tensor,
            vram_addr=next_bias_addr,
            physical_shape=physical_shape,
            vram_preload=vram_preload,
        )
        next_bias_addr += math.ceil((physical_shape[0] * physical_shape[1]) / (mlen * mlen)) * (mlen * mlen)
    cos_vram = sin_vram = None
    if runtime_rope:
        cos_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="ROPE_COS",
            tensor=cos,
            vram_addr=rope_base,
            physical_shape=trig_shape,
            vram_preload=vram_preload,
        )
        sin_vram = prestage_bf16_vram_matrix(
            prog=prog,
            name="ROPE_SIN",
            tensor=sin,
            vram_addr=rope_base + trig_size,
            physical_shape=trig_shape,
            vram_preload=vram_preload,
        )

    w_q_input = prog.input("W_Q_BF16", shape=(hidden, mlen), real_data_ratio=2.0)
    w_k_input = prog.input("W_K_BF16", shape=(hidden, mlen), real_data_ratio=2.0)
    rotate_input = None
    if runtime_rope:
        rotate_input = prog.input("ROPE_ROTATE_BF16", shape=(mlen, mlen), real_data_ratio=2.0)
    if vo_mx:
        w_v_input = prog.input("W_V_MX", shape=(hidden, mlen), real_data_ratio=1.125)
        w_o_input = prog.input("W_O_MX", shape=(mlen, out_features), real_data_ratio=1.125)
    else:
        w_v_input = prog.input("W_V_BF16", shape=(hidden, mlen), real_data_ratio=2.0)
        w_o_input = prog.input("W_O_BF16", shape=(mlen, out_features), real_data_ratio=2.0)

    q_var = prog.linear_projection_bf16(x_vram, w_q_input, name="Q_proj", physical_shape=(physical_rows, mlen))
    k_var = prog.linear_projection_bf16(x_vram, w_k_input, name="K_proj", physical_shape=(physical_rows, mlen))
    if vo_mx:
        v_var = prog.linear_projection(x_vram, w_v_input, name="V_proj_mx", physical_shape=(physical_rows, mlen))
    else:
        v_var = prog.linear_projection_bf16(x_vram, w_v_input, name="V_proj", physical_shape=(physical_rows, mlen))
    if "q" in bias_parts:
        prog.vram_add(q_var, bias_vrams["B_Q"], num_rows=seq)
    if "k" in bias_parts:
        prog.vram_add(k_var, bias_vrams["B_K"], num_rows=seq)
    if "v" in bias_parts:
        prog.vram_add(v_var, bias_vrams["B_V"], num_rows=seq)
    if runtime_rope:
        prog.runtime_rope_projection_bf16(q_var, rotate_input, cos_vram, sin_vram, name="Q_runtime_rotate")
        prog.runtime_rope_projection_bf16(k_var, rotate_input, cos_vram, sin_vram, name="K_runtime_rotate")
    k_staged = prog.store(
        k_var,
        name="K_staged_bf16",
        precision=1,
        hbm_element_bytes=2,
        real_data_ratio=2.0,
    )
    v_staged = prog.store(
        v_var,
        name="V_staged_bf16",
        precision=1,
        hbm_element_bytes=2,
        real_data_ratio=2.0,
    )

    attn_out = prog.alloc("attn_out", seq, mlen, strict=False, physical_shape=(physical_rows, mlen))
    scratch = prog.alloc("_gpt_oss_full_attn_scratch", mlen * (mlen // head_dim), mlen, strict=True)
    if sliding_window is not None:
        causal_mask = prog._build_sliding_causal_score_mask("_gpt_oss_full_sliding_mask", sliding_window)
    else:
        causal_mask = True

    fp_preload = [0.0, scale / 0.25, float("-inf")] + [0.0] * 253
    sink_base = None
    if sinks is not None:
        sink_base = 256
        if len(fp_preload) < sink_base + hq:
            fp_preload.extend([0.0] * (sink_base + hq - len(fp_preload)))
        for idx, value in enumerate(sinks.tolist()):
            fp_preload[sink_base + idx] = float(value)

    prog.flash_attention_packed_group(
        q_var,
        k_staged,
        v_staged,
        group_heads=hq,
        head_slot_dim=head_dim,
        output_base_address=prog.get_vram_addr(attn_out.name),
        scratch_base_address=prog.get_vram_addr(scratch.name),
        broadcast_amount=mlen // head_dim,
        scale=scale,
        causal_mask=causal_mask,
        valid_cols=seq,
        sink_base_address=sink_base,
        k_matrix_precision="keyvalue",
        k_set_scale=False,
        k_hbm_element_bytes=2,
        v_hbm_element_bytes=2,
    )
    if vo_mx:
        out = prog.linear_projection(
            attn_out, w_o_input, name="O_proj_mx", physical_shape=(physical_rows, out_features)
        )
    else:
        out = prog.linear_projection_bf16(
            attn_out, w_o_input, name="O_proj", physical_shape=(physical_rows, out_features)
        )
    if "o" in bias_parts:
        prog.vram_add(out, bias_vrams["B_O"], num_rows=seq)
    isa = prog.compile()

    input_tensors = {
        "W_Q_BF16": w_q,
        "W_K_BF16": w_k,
        k_staged.name: torch.zeros(physical_rows, mlen, dtype=torch.bfloat16),
        v_staged.name: torch.zeros(physical_rows, mlen, dtype=torch.bfloat16),
    }
    if runtime_rope:
        input_tensors["ROPE_ROTATE_BF16"] = rotate
    if vo_mx:
        input_tensors["W_V_MX"] = w_v
        input_tensors["W_O_MX"] = w_o
    else:
        input_tensors["W_V_BF16"] = w_v
        input_tensors["W_O_BF16"] = w_o
    tensor_layouts = infer_hbm_tensor_layouts(input_tensors)
    tensor_layouts.update(
        {
            "W_Q_BF16": _bf16_layout(hidden, mlen),
            "W_K_BF16": _bf16_layout(hidden, mlen),
            k_staged.name: _bf16_layout(physical_rows, mlen, precision="HBM_V_KV_TYPE"),
            v_staged.name: _bf16_layout(physical_rows, mlen, precision="HBM_V_KV_TYPE"),
        }
    )
    if runtime_rope:
        tensor_layouts["ROPE_ROTATE_BF16"] = _bf16_layout(mlen, mlen)
    if not vo_mx:
        tensor_layouts["W_V_BF16"] = _bf16_layout(hidden, mlen)
        tensor_layouts["W_O_BF16"] = _bf16_layout(mlen, out_features)
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    data_order = sorted(input_tensors, key=lambda name: hbm_addrs[name])
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_attention_full",
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )
    params = _comparison_params(
        prog.get_vram_addr(out.name),
        seq,
        out_features,
        mlen,
        physical_rows=out.physical_shape[0],
    )
    _write_json(build_dir / "comparison_params.json", params)
    (build_dir / "generated_asm_code.asm").write_text(isa)

    if not args.no_run:
        if args.probe_rel_only:
            run_emulator(build_dir)
        else:
            run_and_assert(build_dir, "gpt_oss_attention_full", mlen=mlen, blen=blen)
        results, _ = compare_emulator_output(build_dir)
        emu = results["simulated_values"].reshape(seq, out_features).to(torch.bfloat16)
        rel = _rel_rms(emu, golden)
        if rel > 0.02 and not args.probe_rel_only:
            raise AssertionError(f"full attention rel_rms={rel:.6g} exceeds 2%")
    else:
        rel = float("nan")

    summary = {
        "mode": "full",
        "seq": seq,
        "hidden": hidden,
        "out_features": out_features,
        "sink": bool(args.sink),
        "vo_mx": vo_mx,
        "projection_bias": projection_bias,
        "bias_parts": "".join(sorted(bias_parts)),
        "runtime_rope": runtime_rope,
        "projection_precision": "qk_bf16_vo_mxfp8" if vo_mx else "qkvo_bf16",
        "sliding_window": sliding_window,
        "sliding_source": sliding_source,
        "rel_rms": rel,
        "sink_base": sink_base,
        "h_store_keyvalue_count": isa.count("H_STORE_V"),
        "h_prefetch_keyvalue_count": isa.count("H_PREFETCH_M"),
    }
    _write_json(build_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument(
        "--mode",
        choices=(
            "projection",
            "core",
            "full",
            "true_core",
            "true_full",
            "true_attn_block",
            "true_decoder_block",
            "true_decoder_chain",
            "runtime_rope",
            "qk_head_norm_smoke",
            "deepseek_mla_q_path",
            "deepseek_mla_kv_path",
            "external_rope",
        ),
        required=True,
    )
    parser.add_argument(
        "--build-dir", type=Path, default=Path(__file__).parent / "build" / "gpt_oss_attention_semantics"
    )
    parser.add_argument("--out-features", type=int, default=None)
    parser.add_argument(
        "--allow-padded-projection",
        action="store_true",
        help="Allow projection mode to zero-pad non-MLEN hidden/out dimensions for DeepSeek-style shape probes.",
    )
    parser.add_argument("--sink", action="store_true")
    parser.add_argument("--sliding-window", type=int, default=None)
    parser.add_argument("--no-causal-mask", action="store_true")
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--layer-idx", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=64)
    parser.add_argument("--num-key-value-heads", type=int, default=8)
    parser.add_argument(
        "--value-head-dim",
        type=int,
        default=None,
        help=(
            "Logical V head dimension for true_core. Defaults to HLEN/head_dim. "
            "When smaller than head_dim, V is zero-padded to the QK head slot; "
            "this probes DeepSeek-style qk_dim != v_dim attention."
        ),
    )
    parser.add_argument("--q-pt", type=Path, default=None, help="Optional post-RoPE Q tensor for true_core.")
    parser.add_argument("--k-pt", type=Path, default=None, help="Optional post-RoPE K tensor for true_core.")
    parser.add_argument(
        "--v-pt", type=Path, default=None, help="Optional V tensor for true_core; may be narrower than HLEN."
    )
    parser.add_argument(
        "--golden-pt",
        type=Path,
        default=None,
        help="Optional padded attention-core golden tensor for true_core external-QKV mode.",
    )
    parser.add_argument("--cos-pt", type=Path, default=None, help="Optional external BF16 RoPE cos table.")
    parser.add_argument("--sin-pt", type=Path, default=None, help="Optional external BF16 RoPE sin table.")
    parser.add_argument(
        "--projection-weight-pt",
        type=Path,
        default=None,
        help="Optional rank-2 BF16 projection weight for projection mode; accepts PLENA [in,out] or HF [out,in].",
    )
    parser.add_argument(
        "--second-projection-weight-pt",
        type=Path,
        default=None,
        help="Optional second rank-2 BF16 projection weight for integrated two-projection probes.",
    )
    parser.add_argument(
        "--norm-weight-pt",
        type=Path,
        default=None,
        help="Optional rank-1 BF16 RMSNorm scale for integrated projection/RMSNorm probes.",
    )
    parser.add_argument(
        "--attention-scale",
        type=float,
        default=None,
        help="Optional attention softmax scale override for true_core, needed for DeepSeek YaRN scaling.",
    )
    parser.add_argument(
        "--o-weight-pt",
        type=Path,
        default=None,
        help=(
            "Optional BF16 O-projection weight for true_core.  The tensor may be "
            "[logical_value_heads, out] or HF-style [out, logical_value_heads]. "
            "For DeepSeek-style value_head_dim < head_dim, the harness expands it "
            "into the padded head slot with zero tail rows."
        ),
    )
    parser.add_argument(
        "--o-golden-pt",
        type=Path,
        default=None,
        help="Optional direct HF O-projection output for true_core + --o-weight-pt validation.",
    )
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument(
        "--projection-bias",
        action="store_true",
        help="Add BF16 projection bias. Validated for projection mode; full mode remains a diagnostic probe.",
    )
    parser.add_argument(
        "--bias-parts",
        default="qkvo",
        help="Diagnostic subset of projection bias terms to enable in full mode: any of q/k/v/o, all, or none.",
    )
    parser.add_argument(
        "--runtime-rope",
        action="store_true",
        help="Apply device-side rotate-half projection plus RoPE after Q/K projection.",
    )
    parser.add_argument(
        "--qk-head-norm",
        action="store_true",
        help="Apply Qwen-style RMSNorm over each Q/K head before RoPE.",
    )
    parser.add_argument(
        "--qk-head-norm-eps",
        type=float,
        default=1e-6,
        help="Epsilon for --qk-head-norm; Qwen3-MoE uses config.rms_norm_eps.",
    )
    parser.add_argument(
        "--norm-eps",
        type=float,
        default=1e-5,
        help="Epsilon for block RMSNorm before attention/FFN; GPT-OSS uses 1e-5, Qwen3-MoE uses 1e-6.",
    )
    parser.add_argument(
        "--qk-norm-weight-pt",
        type=Path,
        default=None,
        help="Optional BF16 head RMSNorm weight tensor for qk_head_norm_smoke.",
    )
    parser.add_argument(
        "--debug-stop-after-first-q-head",
        action="store_true",
        help="Diagnostic true_full mode: stop after the first Q head RMSNorm+RoPE and compare that head.",
    )
    parser.add_argument(
        "--debug-stop-after-first-q-norm",
        action="store_true",
        help="Diagnostic true_full mode: stop after the first Q head RMSNorm+weight before RoPE.",
    )
    parser.add_argument(
        "--vo-mx",
        action="store_true",
        help="Precision probe: keep Q/K projections BF16 but use MXFP8 V/O projection weights.",
    )
    parser.add_argument(
        "--probe-rel-only",
        action="store_true",
        help="Run emulator and report rel_rms without failing the process on numerical mismatch.",
    )
    parser.add_argument(
        "--input-pt",
        type=Path,
        default=None,
        help="Optional BF16 hidden-state tensor to use instead of the generated random input.",
    )
    parser.add_argument(
        "--qwen-real-layer",
        action="store_true",
        help="Load true Qwen3-30B-A3B layer weights instead of random synthetic true_decoder_block weights.",
    )
    parser.add_argument(
        "--qwen-snapshot",
        type=Path,
        default=_default_qwen3_snapshot(),
        help="Local Qwen3-30B-A3B snapshot containing config.json and safetensors index "
        "(or set QWEN3_30B_A3B_SNAPSHOT).",
    )
    parser.add_argument(
        "--qwen-router-matrix-bf16",
        action="store_true",
        help=(
            "Use the BF16 matrix-machine router logits path for Qwen3-MoE, "
            "then pack logits into the existing V_TOPK token-major ABI."
        ),
    )
    parser.add_argument(
        "--qwen-router-mram-tile-capacity",
        type=int,
        default=4,
        help=(
            "MRAM K-tile chunk size for --qwen-router-matrix-bf16.  Qwen full-block "
            "runs use MLEN=128, so the default 4 tiles gives the same 512-element "
            "K chunk as the standalone MLEN=64/cap8 router probe."
        ),
    )
    parser.add_argument(
        "--qwen-router-packed-skinny-bf16",
        action="store_true",
        help=(
            "Use the packed-skinny BF16 matrix router table for Qwen3-MoE, "
            "then pack logits into the existing V_TOPK token-major ABI."
        ),
    )
    parser.add_argument(
        "--qwen-router-packed-skinny-k-tiles",
        type=int,
        default=8,
        help="Number of skinny K tiles packed into one full router-weight MRAM tile.",
    )
    parser.add_argument(
        "--save-output-pt",
        type=Path,
        default=None,
        help="Optional path to save the emulator output tensor after a run.",
    )
    parser.add_argument(
        "--save-golden-pt",
        type=Path,
        default=None,
        help="Optional path to save the host golden tensor for the current run.",
    )
    parser.add_argument("--moe-num-experts", type=int, default=32)
    parser.add_argument("--moe-top-k", type=int, default=4)
    parser.add_argument("--moe-intermediate-size", type=int, default=None)
    parser.add_argument(
        "--moe-policy-name",
        choices=("gpt_oss", "qwen3_moe"),
        default="gpt_oss",
        help="MoE router/expert policy for true_decoder_block.",
    )
    parser.add_argument(
        "--moe-activation-policy",
        choices=("gpt_oss_clamp_gated", "standard_swiglu"),
        default="gpt_oss_clamp_gated",
        help="Expert activation policy for true_decoder_block.",
    )
    parser.add_argument(
        "--chain-layers",
        type=int,
        default=2,
        help="Number of sequential true_decoder_block runs for true_decoder_chain mode.",
    )
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "projection":
        run_projection(args)
    elif args.mode == "core":
        run_core(args)
    elif args.mode == "true_core":
        run_true_core(args)
    elif args.mode in ("true_full", "true_attn_block", "true_decoder_block"):
        run_true_full(args)
    elif args.mode == "true_decoder_chain":
        run_true_decoder_chain(args)
    elif args.mode == "runtime_rope":
        run_runtime_rope(args)
    elif args.mode == "qk_head_norm_smoke":
        run_qk_head_norm_smoke(args)
    elif args.mode in ("deepseek_mla_q_path", "deepseek_mla_kv_path"):
        run_deepseek_mla_q_path(args)
    elif args.mode == "external_rope":
        run_external_rope(args)
    else:
        run_full(args)


if __name__ == "__main__":
    main()
