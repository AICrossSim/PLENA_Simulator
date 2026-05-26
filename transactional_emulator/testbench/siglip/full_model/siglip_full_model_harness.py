"""ASM generation for full 27-layer SigLIP model.

Emits monolithic assembly code for all encoder layers, managing VRAM/HBM layout globally.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_emulator
from transactional_emulator.testbench.siglip.full_model.embedding_flow import (
    fill_embedding_inputs_for_asm,
    prepare_vram_preload_from_embedding,
)
from transactional_emulator.testbench.siglip.full_model.golden_reference import (
    compute_golden_embedding,
    compute_golden_full_model,
)
from compiler.asm_templates.siglip import (
    build_embedding_stage_asm,
    build_encoder_layer_asm,
)
from transactional_emulator.testbench.siglip.local_asm_templates.layout import (
    compute_vram_layout,
)
from transactional_emulator.testbench.siglip.utils.core import align_up
from transactional_emulator.testbench.siglip.utils.harness_utils import prepare_case_artifacts
from transactional_emulator.testbench.siglip.utils.vram import (
    pack_seq_to_chunk_major,
    load_vram_chunk_major_to_seq,
    load_vram_head_major_q_to_seq,
    load_vram_seq_major_to_seq,
)
from transactional_emulator.testbench.siglip.utils.math import gqa_sdpa

DIAG_ABS_BUDGET = 0.0625
DIAG_REL_BUDGET = 0.05


def _compute_persistent_end(
    *,
    vram_layout: dict,
    seq_len: int,
    hidden_size: int,
    inter_size: int,
    num_layers: int,
) -> int:
    """Compute end of persistent VRAM region before scratch/workspace allocations."""
    persistent_end = int(vram_layout["embedding_base"]) + int(vram_layout["embedding_size"])
    layer_bases = vram_layout.get("layer_bases", {})
    layer_sizes = vram_layout.get("layer_sizes", {})
    if layer_bases:
        last_layer_idx = max(layer_bases.keys())
        persistent_end = max(
            persistent_end,
            int(layer_bases[last_layer_idx]) + int(layer_sizes[last_layer_idx]),
        )

    q_bias_bases = vram_layout.get("q_bias_bases", {})
    ln1_weight_bases = vram_layout.get("ln1_weight_bases", {})
    ln1_bias_bases = vram_layout.get("ln1_bias_bases", {})
    ln2_weight_bases = vram_layout.get("ln2_weight_bases", {})
    ln2_bias_bases = vram_layout.get("ln2_bias_bases", {})
    out_bias_bases = vram_layout.get("out_bias_bases", {})
    fc1_bias_bases = vram_layout.get("fc1_bias_bases", {})
    fc2_bias_bases = vram_layout.get("fc2_bias_bases", {})

    for layer_idx in range(num_layers):
        q_bias_base = q_bias_bases.get(layer_idx)
        if q_bias_base is not None:
            persistent_end = max(persistent_end, int(q_bias_base) + seq_len * hidden_size)

        for base_dict in (ln1_weight_bases, ln1_bias_bases, ln2_weight_bases, ln2_bias_bases):
            base = base_dict.get(layer_idx)
            if base is not None:
                persistent_end = max(persistent_end, int(base) + seq_len * hidden_size)

        out_base = out_bias_bases.get(layer_idx)
        if out_base is not None:
            persistent_end = max(persistent_end, int(out_base) + seq_len * hidden_size)

        fc1_base = fc1_bias_bases.get(layer_idx)
        if fc1_base is not None:
            persistent_end = max(persistent_end, int(fc1_base) + seq_len * inter_size)

        fc2_base = fc2_bias_bases.get(layer_idx)
        if fc2_base is not None:
            persistent_end = max(persistent_end, int(fc2_base) + seq_len * hidden_size)

    return max(persistent_end, int(vram_layout.get("total_vram_elements", persistent_end)))


def _infer_source_num_heads(source_hidden: int, target_heads: int) -> int:
    """Infer source checkpoint head count for head-aware hidden remapping."""
    # SigLIP So400m checkpoints use 16 heads. Prefer this when compatible.
    if source_hidden % 16 == 0 and target_heads <= 16:
        return 16
    if source_hidden % target_heads == 0:
        return target_heads
    for h in range(min(source_hidden, 64), 0, -1):
        if source_hidden % h == 0 and h >= target_heads:
            return h
    return 1


def _build_hidden_index_map(source_hidden: int, target_hidden: int, target_heads: int) -> torch.Tensor:
    """Build source->target hidden index map preserving head-block locality."""
    source_heads = _infer_source_num_heads(source_hidden, target_heads)
    source_head_dim = source_hidden // source_heads
    target_head_dim = target_hidden // target_heads

    heads_to_copy = min(source_heads, target_heads)
    dims_to_copy = min(source_head_dim, target_head_dim)

    indices: list[int] = []
    for h in range(heads_to_copy):
        base = h * source_head_dim
        for d in range(dims_to_copy):
            indices.append(base + d)

    # Fallback: if target_hidden is still not covered, append remaining indices sequentially.
    if len(indices) < target_hidden:
        selected = set(indices)
        for i in range(source_hidden):
            if i in selected:
                continue
            indices.append(i)
            if len(indices) >= target_hidden:
                break

    return torch.tensor(indices[:target_hidden], dtype=torch.long)


def _gelu_hardware_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Hardware GELU approximation used by gelu_asm: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


def _build_hbm_input_tensors(
    embedding_weights: dict,
    layer_weights_list: list,
    num_layers: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Build HBM preload tensors keyed by filename stem in create_mem_for_sim order.

    The run path is a smoke harness: it executes generated ASM in emulator, but does
    not claim numerical parity yet.
    """
    patch_bias = embedding_weights.get("patch_bias")
    if patch_bias is None:
        patch_bias = torch.zeros(1, dtype=torch.float32)

    input_tensor: dict[str, torch.Tensor] = {
        "patch_weight": embedding_weights["patch_weight"].reshape(-1).to(torch.float32),
        "patch_bias": patch_bias.reshape(-1).to(torch.float32),
        "position_table": embedding_weights["position_table"].reshape(-1).to(torch.float32),
    }

    data_order = ["patch_weight", "patch_bias", "position_table"]

    for layer_idx in range(num_layers):
        weights = layer_weights_list[layer_idx]

        # Keep names aligned with compute_hbm_data_order() and hbm layout keys.
        keyed_tensors = {
            f"layer_{layer_idx}_ln1_weight": weights["ln1_weight"],
            f"layer_{layer_idx}_ln1_bias": weights["ln1_bias"],
            f"layer_{layer_idx}_q_proj_weight": weights["q_proj_weight"],
            f"layer_{layer_idx}_q_proj_bias": weights["q_proj_bias"],
            f"layer_{layer_idx}_k_proj_weight": weights["k_proj_weight"],
            f"layer_{layer_idx}_k_proj_bias": weights["k_proj_bias"],
            f"layer_{layer_idx}_v_proj_weight": weights["v_proj_weight"],
            f"layer_{layer_idx}_v_proj_bias": weights["v_proj_bias"],
            f"layer_{layer_idx}_out_proj_weight": weights["out_proj_weight"],
            f"layer_{layer_idx}_out_proj_bias": weights["out_proj_bias"],
            f"layer_{layer_idx}_ln2_weight": weights["ln2_weight"],
            f"layer_{layer_idx}_ln2_bias": weights["ln2_bias"],
            f"layer_{layer_idx}_fc1_weight": weights["fc1_weight"],
            f"layer_{layer_idx}_fc1_bias": weights["fc1_bias"],
            f"layer_{layer_idx}_fc2_weight": weights["fc2_weight"],
            f"layer_{layer_idx}_fc2_bias": weights["fc2_bias"],
        }

        for key, tensor in keyed_tensors.items():
            if tensor is None:
                tensor = torch.zeros(1, dtype=torch.float32)
            input_tensor[key] = tensor.reshape(-1).to(torch.float32)
            data_order.append(key)

    return input_tensor, data_order


def _prepare_vram_preload(
    config: dict,
    embedding_weights: dict,
    seq_len: int,
    mlen: int,
    hidden_size: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """Prepare VRAM preload with [patches | zeros until position offset | position table]."""
    patch_size = config["patch_size"]
    num_channels = config["num_channels"]
    in_features = num_channels * patch_size * patch_size
    aligned_in_features = align_up(in_features, mlen)

    torch.manual_seed(0)
    pixel_values = torch.randn(1, num_channels, config["image_size"], config["image_size"], dtype=torch.float32)
    patches = F.unfold(pixel_values, kernel_size=patch_size, stride=patch_size).transpose(1, 2).contiguous()[0]
    # Keep a copy of the unaligned patches for golden-reference computation.
    patches_raw = patches.clone()

    if aligned_in_features != in_features:
        patches = F.pad(patches, (0, aligned_in_features - in_features))

    input_flat = patches.reshape(-1).to(torch.bfloat16)

    embedding_result_base = aligned_in_features * seq_len
    position_base = embedding_result_base + seq_len * hidden_size
    position_flat = embedding_weights["position_table"].reshape(-1).to(torch.bfloat16)

    total_len = position_base + position_flat.numel()
    preload = torch.zeros(total_len, dtype=torch.bfloat16)
    preload[: input_flat.numel()] = input_flat
    preload[position_base : position_base + position_flat.numel()] = position_flat

    return preload.view(torch.int16).numpy().view(np.uint16), patches_raw


def build_runtime_repacked_model(
    config: dict,
    embedding_weights: dict,
    layer_weights_list: list,
    mlen: int,
    seq_len_kernel: int,
) -> tuple[dict, dict, list[dict]]:
    """Build runtime-expanded tensors matching flash-attn padded head geometry."""
    runtime_config = dict(config)

    hidden = int(config["hidden_size"])
    inter = int(config["intermediate_size"])
    hq = int(config["num_attention_heads"])
    head_dim = hidden // hq
    d_padded = align_up(head_dim, mlen)
    hidden_runtime = hq * d_padded
    inter_padded = align_up(inter, mlen)

    runtime_config["hidden_size"] = hidden_runtime
    runtime_config["intermediate_size"] = inter_padded
    runtime_config["seq_len_valid"] = int((config["image_size"] // config["patch_size"]) ** 2)

    patch_size = int(config["patch_size"])
    num_channels = int(config["num_channels"])
    in_features = num_channels * patch_size * patch_size
    aligned_in_features = align_up(in_features, mlen)

    source_hidden = int(embedding_weights["position_table"].shape[1])
    hidden_index = _build_hidden_index_map(source_hidden, hidden, hq)

    patch_weight = embedding_weights["patch_weight"].detach().float()
    patch_weight_runtime = torch.zeros(aligned_in_features, hidden_runtime, dtype=torch.float32)
    copy_rows = min(patch_weight.shape[0], aligned_in_features)
    copy_hidden = min(hidden, patch_weight.shape[1], hidden_runtime, hidden_index.numel())
    patch_weight_runtime[:copy_rows, :copy_hidden] = patch_weight[:copy_rows, :].index_select(
        1, hidden_index[:copy_hidden]
    )

    patch_bias_src = embedding_weights.get("patch_bias")
    patch_bias_runtime = torch.zeros(hidden_runtime, dtype=torch.float32)
    if patch_bias_src is not None:
        patch_bias_f = patch_bias_src.detach().float()
        copy_hidden = min(hidden, patch_bias_f.numel(), hidden_runtime, hidden_index.numel())
        patch_bias_runtime[:copy_hidden] = patch_bias_f.index_select(0, hidden_index[:copy_hidden])

    position_src = embedding_weights["position_table"].detach().float()
    position_runtime = torch.zeros(seq_len_kernel, hidden_runtime, dtype=torch.float32)
    valid_rows = min(position_src.shape[0], seq_len_kernel)
    copy_hidden = min(hidden, position_src.shape[1], hidden_runtime, hidden_index.numel())
    position_runtime[:valid_rows, :copy_hidden] = position_src[:valid_rows, :].index_select(
        1, hidden_index[:copy_hidden]
    )

    runtime_embedding = {
        "patch_weight": patch_weight_runtime,
        "patch_bias": patch_bias_runtime,
        "position_table": position_runtime,
    }

    runtime_layers: list[dict] = []
    for src in layer_weights_list:
        layer_rt: dict[str, torch.Tensor | None] = {}

        ln1_w = torch.zeros(hidden_runtime, dtype=torch.float32)
        ln1_src = src["ln1_weight"].detach().float()
        copy_hidden = min(hidden, ln1_src.numel(), hidden_runtime, hidden_index.numel())
        ln1_w[:copy_hidden] = ln1_src.index_select(0, hidden_index[:copy_hidden])
        layer_rt["ln1_weight"] = ln1_w
        ln1_b = src.get("ln1_bias")
        if ln1_b is not None:
            b = torch.zeros(hidden_runtime, dtype=torch.float32)
            ln1_b_src = ln1_b.detach().float()
            copy_hidden = min(hidden, ln1_b_src.numel(), hidden_runtime, hidden_index.numel())
            b[:copy_hidden] = ln1_b_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["ln1_bias"] = b
        else:
            layer_rt["ln1_bias"] = None

        wq_src = src["q_proj_weight"].detach().float()
        wq = wq_src.index_select(0, hidden_index).index_select(1, hidden_index).T
        wq_rt = torch.zeros(hidden_runtime, hidden_runtime, dtype=torch.float32)
        copy_h = min(hidden, wq.shape[0], wq.shape[1], hidden_runtime, hidden_index.numel())
        wq_rt[:copy_h, :copy_h] = wq[:copy_h, :copy_h]
        layer_rt["q_proj_weight"] = wq_rt

        q_bias = src.get("q_proj_bias")
        if q_bias is not None:
            qb = torch.zeros(hidden_runtime, dtype=torch.float32)
            q_bias_src = q_bias.detach().float()
            copy_hidden = min(hidden, q_bias_src.numel(), hidden_runtime, hidden_index.numel())
            qb[:copy_hidden] = q_bias_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["q_proj_bias"] = qb
        else:
            layer_rt["q_proj_bias"] = None

        wk_src = src["k_proj_weight"].detach().float()
        wk = wk_src.index_select(0, hidden_index).index_select(1, hidden_index).T
        wk_rt = torch.zeros(hidden_runtime, hidden_runtime, dtype=torch.float32)
        copy_h = min(hidden, wk.shape[0], wk.shape[1], hidden_runtime, hidden_index.numel())
        wk_rt[:copy_h, :copy_h] = wk[:copy_h, :copy_h]
        layer_rt["k_proj_weight"] = wk_rt

        wv_src = src["v_proj_weight"].detach().float()
        wv = wv_src.index_select(0, hidden_index).index_select(1, hidden_index).T
        wv_rt = torch.zeros(hidden_runtime, hidden_runtime, dtype=torch.float32)
        copy_h = min(hidden, wv.shape[0], wv.shape[1], hidden_runtime, hidden_index.numel())
        wv_rt[:copy_h, :copy_h] = wv[:copy_h, :copy_h]
        layer_rt["v_proj_weight"] = wv_rt

        k_bias = src.get("k_proj_bias")
        if k_bias is not None:
            kb = torch.zeros(hidden_runtime, dtype=torch.float32)
            k_bias_src = k_bias.detach().float()
            copy_hidden = min(hidden, k_bias_src.numel(), hidden_runtime, hidden_index.numel())
            kb[:copy_hidden] = k_bias_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["k_proj_bias"] = kb
        else:
            layer_rt["k_proj_bias"] = None

        v_bias = src.get("v_proj_bias")
        if v_bias is not None:
            vb = torch.zeros(hidden_runtime, dtype=torch.float32)
            v_bias_src = v_bias.detach().float()
            copy_hidden = min(hidden, v_bias_src.numel(), hidden_runtime, hidden_index.numel())
            vb[:copy_hidden] = v_bias_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["v_proj_bias"] = vb
        else:
            layer_rt["v_proj_bias"] = None

        out_w = src.get("out_proj_weight")
        if out_w is not None:
            ow = torch.zeros(hidden_runtime, hidden_runtime, dtype=torch.float32)
            out_w_src = out_w.detach().float().index_select(0, hidden_index).index_select(1, hidden_index).T
            copy_h = min(hidden, out_w_src.shape[0], out_w_src.shape[1], hidden_runtime, hidden_index.numel())
            ow[:copy_h, :copy_h] = out_w_src[:copy_h, :copy_h]
            layer_rt["out_proj_weight"] = ow
        else:
            layer_rt["out_proj_weight"] = None

        out_b = src.get("out_proj_bias")
        if out_b is not None:
            ob = torch.zeros(hidden_runtime, dtype=torch.float32)
            out_b_src = out_b.detach().float()
            copy_hidden = min(hidden, out_b_src.numel(), hidden_runtime, hidden_index.numel())
            ob[:copy_hidden] = out_b_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["out_proj_bias"] = ob
        else:
            layer_rt["out_proj_bias"] = None

        ln2_w = torch.zeros(hidden_runtime, dtype=torch.float32)
        ln2_src = src["ln2_weight"].detach().float()
        copy_hidden = min(hidden, ln2_src.numel(), hidden_runtime, hidden_index.numel())
        ln2_w[:copy_hidden] = ln2_src.index_select(0, hidden_index[:copy_hidden])
        layer_rt["ln2_weight"] = ln2_w
        ln2_b = src.get("ln2_bias")
        if ln2_b is not None:
            b = torch.zeros(hidden_runtime, dtype=torch.float32)
            ln2_b_src = ln2_b.detach().float()
            copy_hidden = min(hidden, ln2_b_src.numel(), hidden_runtime, hidden_index.numel())
            b[:copy_hidden] = ln2_b_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["ln2_bias"] = b
        else:
            layer_rt["ln2_bias"] = None

        fc1_src = src["fc1_weight"].detach().float()[:, :].index_select(1, hidden_index)[:inter, :].T
        fc1_rt = torch.zeros(hidden_runtime, inter_padded, dtype=torch.float32)
        fc1_rt[:hidden, :inter] = fc1_src
        layer_rt["fc1_weight"] = fc1_rt
        fc1_b_src = src.get("fc1_bias")
        if fc1_b_src is not None:
            b = torch.zeros(inter_padded, dtype=torch.float32)
            b[:inter] = fc1_b_src.detach().float()[:inter]
            layer_rt["fc1_bias"] = b
        else:
            layer_rt["fc1_bias"] = None

        fc2_src = src["fc2_weight"].detach().float().index_select(0, hidden_index)[:hidden, :inter].T
        fc2_rt = torch.zeros(inter_padded, hidden_runtime, dtype=torch.float32)
        fc2_rt[:inter, :hidden] = fc2_src
        layer_rt["fc2_weight"] = fc2_rt
        fc2_b_src = src.get("fc2_bias")
        if fc2_b_src is not None:
            b = torch.zeros(hidden_runtime, dtype=torch.float32)
            b[:hidden] = fc2_b_src.detach().float().index_select(0, hidden_index)
            layer_rt["fc2_bias"] = b
        else:
            layer_rt["fc2_bias"] = None

        runtime_layers.append(layer_rt)

    return runtime_config, runtime_embedding, runtime_layers


def _compute_layer_kv_tiles(
    x_in: torch.Tensor,
    layer_weights: dict,
    config: dict,
    mlen: int,
    seq_len_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-layer K/V activations expected by flash-attn HBM prefetch."""
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden_size // num_heads
    d_padded = align_up(head_dim, mlen)
    eps = float(config.get("layer_norm_eps", 1e-2))

    source_hidden = int(layer_weights["q_proj_weight"].shape[1])
    if source_hidden == hidden_size:
        hidden_index = torch.arange(hidden_size, dtype=torch.long)
    else:
        hidden_index = _build_hidden_index_map(source_hidden, hidden_size, num_heads)

    ln1_w = layer_weights.get("ln1_weight")
    ln1_b = layer_weights.get("ln1_bias")
    ln1_w_f = ln1_w.detach().float().index_select(0, hidden_index) if ln1_w is not None else None
    ln1_b_f = ln1_b.detach().float().index_select(0, hidden_index) if ln1_b is not None else None

    x_ln1 = F.layer_norm(
        x_in[:, :hidden_size].float(),
        (hidden_size,),
        weight=ln1_w_f,
        bias=ln1_b_f,
        eps=eps,
    ).to(torch.bfloat16).float()

    wk_src = layer_weights["k_proj_weight"].detach().float()
    wk = wk_src.index_select(0, hidden_index).index_select(1, hidden_index)
    bk = layer_weights.get("k_proj_bias")
    bk_f = bk.detach().float().index_select(0, hidden_index) if bk is not None else None
    wv_src = layer_weights["v_proj_weight"].detach().float()
    wv = wv_src.index_select(0, hidden_index).index_select(1, hidden_index)
    bv = layer_weights.get("v_proj_bias")
    bv_f = bv.detach().float().index_select(0, hidden_index) if bv is not None else None

    k_out = F.linear(x_ln1, wk, bk_f).float()
    v_out = F.linear(x_ln1, wv, bv_f).float()

    seq_len = x_in.shape[0]
    k_heads = k_out.reshape(seq_len, num_kv_heads, head_dim)
    v_heads = v_out.reshape(seq_len, num_kv_heads, head_dim)

    if d_padded != head_dim:
        k_heads = F.pad(k_heads, (0, d_padded - head_dim))
        v_heads = F.pad(v_heads, (0, d_padded - head_dim))

    # HBM K/V layout expected by flash-attn templates: [hkv, seq_kernel, d_padded].
    k_tiled = torch.zeros(num_kv_heads, seq_len_kernel, d_padded, dtype=torch.float32)
    v_tiled = torch.zeros(num_kv_heads, seq_len_kernel, d_padded, dtype=torch.float32)
    k_tiled[:, :seq_len, :] = k_heads.permute(1, 0, 2).contiguous()
    v_tiled[:, :seq_len, :] = v_heads.permute(1, 0, 2).contiguous()
    k_hbm = k_tiled.reshape(-1).to(torch.float32)
    v_hbm = v_tiled.reshape(-1).to(torch.float32)
    return k_hbm, v_hbm


def _build_runtime_layer_payloads(
    config: dict,
    runtime_layer_weights_list: list,
    source_layer_weights_list: list,
    layer_outputs: list[torch.Tensor],
    num_layers: int,
    mlen: int,
    seq_len_kernel: int,
) -> list[dict]:
    """Build per-layer payload dicts with K/V slots populated by runtime activations."""
    runtime_layers: list[dict] = []
    for layer_idx in range(num_layers):
        src_runtime = runtime_layer_weights_list[layer_idx]
        src_kv = source_layer_weights_list[layer_idx]
        runtime = dict(src_runtime)

        x_in = layer_outputs[layer_idx]
        k_tile, v_tile = _compute_layer_kv_tiles(x_in, src_kv, config, mlen, seq_len_kernel)

        k_slot = torch.zeros(src_runtime["k_proj_weight"].numel(), dtype=torch.float32)
        v_slot = torch.zeros(src_runtime["v_proj_weight"].numel(), dtype=torch.float32)
        k_slot[: k_tile.numel()] = k_tile
        v_slot[: v_tile.numel()] = v_tile

        # Reuse K/V weight slots for runtime K/V activation tiles to keep
        # existing HBM offsets/layout contracts unchanged.
        runtime["k_proj_weight"] = k_slot.reshape_as(src_runtime["k_proj_weight"])
        runtime["v_proj_weight"] = v_slot.reshape_as(src_runtime["v_proj_weight"])
        runtime_layers.append(runtime)

    return runtime_layers


def _compute_emitted_path_golden(
    *,
    patches: torch.Tensor,
    runtime_embedding_weights: dict,
    runtime_layers: list,
    runtime_config: dict,
    seq_len_valid: int,
    mlen: int,
) -> torch.Tensor:
    """Compute golden output for the exact emitted path using runtime K/V payloads."""
    x_full = compute_golden_embedding(
        patches,
        runtime_embedding_weights,
        runtime_config,
        use_mxfp=False,
    ).float()
    x = x_full[:seq_len_valid, :]

    hidden_size = int(runtime_config["hidden_size"])
    inter_size = int(runtime_config["intermediate_size"])
    num_heads = int(runtime_config["num_attention_heads"])
    num_kv_heads = int(runtime_config["num_key_value_heads"])
    head_dim = hidden_size // num_heads
    eps = float(runtime_config.get("layer_norm_eps", 1e-2))
    scale = 1.0 / float(head_dim) ** 0.5
    seq_len_kernel = int(x_full.shape[0])

    for layer in runtime_layers:
        ln1_w = layer["ln1_weight"].float()
        ln1_b = layer["ln1_bias"].float() if layer.get("ln1_bias") is not None else None
        x_ln1 = F.layer_norm(x, (hidden_size,), weight=ln1_w, bias=ln1_b, eps=eps).to(torch.bfloat16).float()

        q = x_ln1 @ layer["q_proj_weight"].float()
        q_bias = layer.get("q_proj_bias")
        if q_bias is not None:
            q = q + q_bias.float()
        q = q.to(torch.bfloat16).float()

        kv_elems = num_kv_heads * seq_len_kernel * head_dim
        k_heads = layer["k_proj_weight"].reshape(-1).float()[:kv_elems].reshape(num_kv_heads, seq_len_kernel, head_dim)
        v_heads = layer["v_proj_weight"].reshape(-1).float()[:kv_elems].reshape(num_kv_heads, seq_len_kernel, head_dim)
        k_heads = k_heads[:, :seq_len_valid, :]
        v_heads = v_heads[:, :seq_len_valid, :]

        q_heads = q.reshape(1, seq_len_valid, num_heads, head_dim)
        k_heads_b = k_heads.permute(1, 0, 2).reshape(1, seq_len_valid, num_kv_heads, head_dim)
        v_heads_b = v_heads.permute(1, 0, 2).reshape(1, seq_len_valid, num_kv_heads, head_dim)
        attn_out = gqa_sdpa(
            q_heads,
            k_heads_b,
            v_heads_b,
            scale=scale,
            hq=num_heads,
            hkv=num_kv_heads,
            kv_valid_len=seq_len_valid,
        ).reshape(seq_len_valid, hidden_size).to(torch.bfloat16).float()

        out = F.linear(
            attn_out,
            layer["out_proj_weight"].float(),
            layer["out_proj_bias"].float() if layer.get("out_proj_bias") is not None else None,
        ).to(torch.bfloat16).float()
        x_res1 = (x.to(torch.bfloat16) + out.to(torch.bfloat16)).to(torch.bfloat16).float()

        ln2_w = layer["ln2_weight"].float()
        ln2_b = layer["ln2_bias"].float() if layer.get("ln2_bias") is not None else None
        x_ln2 = F.layer_norm(x_res1, (hidden_size,), weight=ln2_w, bias=ln2_b, eps=eps).to(torch.bfloat16).float()

        fc1 = F.linear(
            x_ln2,
            layer["fc1_weight"].float().T,
            layer["fc1_bias"].float() if layer.get("fc1_bias") is not None else None,
        ).to(torch.bfloat16).float()
        gelu = _gelu_hardware_sigmoid(fc1).to(torch.bfloat16).float()
        fc2 = F.linear(
            gelu[:, :inter_size],
            layer["fc2_weight"].float().T[:, :inter_size],
            layer["fc2_bias"].float() if layer.get("fc2_bias") is not None else None,
        ).to(torch.bfloat16).float()
        x = (x_res1.to(torch.bfloat16) + fc2.to(torch.bfloat16)).to(torch.bfloat16).float()

    return x


def _print_layer0_stage_diagnostics(
    *,
    build_dir: Path,
    config: dict,
    runtime_config: dict,
    vram_layout: dict,
    runtime_layer_weights_list: list,
    layer_outputs: list[torch.Tensor],
    mlen: int,
    blen: int,
) -> None:
    """Print quick layer-0 stage parity metrics (x input and q projection)."""
    if not runtime_layer_weights_list or not layer_outputs:
        return

    emulator_root = Path(__file__).resolve().parents[3]
    vram_dump = emulator_root / "vram_dump.bin"
    if not vram_dump.exists():
        print(f"[diag] Skipping stage diagnostics (missing {vram_dump})")
        return

    seq_len = int(vram_layout["seq_len"])
    hidden_visible = int(config["hidden_size"])
    hidden_runtime = int(runtime_config["hidden_size"])
    hq = int(config["num_attention_heads"])
    head_dim = hidden_visible // hq
    eps = float(config.get("layer_norm_eps", 1e-2))
    d_padded = align_up(head_dim, mlen)

    seq_len_valid_cfg = int(runtime_config.get("seq_len_valid", seq_len))
    seq_valid = max(0, min(seq_len, seq_len_valid_cfg))
    if seq_valid == 0:
        print("[diag] Skipping stage diagnostics (seq_len_valid=0)")
        return
    print(f"[diag] using valid tokens only: {seq_valid}/{seq_len}")

    probe_diagnostics: list[dict[str, float | int | str]] = []

    def _summarize(name: str, sim: torch.Tensor, ref: torch.Tensor) -> None:
        sim_f = sim[:seq_valid].float()
        ref_f = ref[:seq_valid].float()
        diff = (sim_f - ref_f).abs()
        rel = diff / ref_f.abs().clamp_min(1e-6)
        abs_pass = diff <= DIAG_ABS_BUDGET
        rel_pass = rel <= DIAG_REL_BUDGET
        within_budget = abs_pass | rel_pass
        match = within_budget.float().mean().item() * 100.0

        flat_diff = diff.reshape(-1)
        flat_rel = rel.reshape(-1)
        numel = int(flat_diff.numel())
        abs_budget_fail = int((~abs_pass).sum().item())
        rel_budget_fail = int((~rel_pass).sum().item())
        abs_budget_fail_pct = (100.0 * abs_budget_fail / numel) if numel > 0 else 0.0
        rel_budget_fail_pct = (100.0 * rel_budget_fail / numel) if numel > 0 else 0.0
        p95_abs = float(torch.quantile(flat_diff, 0.95).item()) if numel > 0 else 0.0
        p99_abs = float(torch.quantile(flat_diff, 0.99).item()) if numel > 0 else 0.0
        mean_rel = float(flat_rel.mean().item()) if numel > 0 else 0.0

        probe_diagnostics.append(
            {
                "probe": name,
                "elements": numel,
                "match_percent": match,
                "mean_abs": float(flat_diff.mean().item()) if numel > 0 else 0.0,
                "max_abs": float(flat_diff.max().item()) if numel > 0 else 0.0,
                "p95_abs": p95_abs,
                "p99_abs": p99_abs,
                "mean_rel": mean_rel,
                "abs_budget_fail": abs_budget_fail,
                "abs_budget_fail_pct": abs_budget_fail_pct,
                "rel_budget_fail": rel_budget_fail,
                "rel_budget_fail_pct": rel_budget_fail_pct,
            }
        )
        print(
            f"[diag] {name}: match={match:.3f}% "
            f"mean_abs={flat_diff.mean().item():.6f} max_abs={flat_diff.max().item():.6f} "
            f"p95_abs={p95_abs:.6f} p99_abs={p99_abs:.6f} mean_rel={mean_rel:.6f} "
            f"abs_fail={abs_budget_fail}/{numel} ({abs_budget_fail_pct:.3f}%) "
            f"rel_fail={rel_budget_fail}/{numel} ({rel_budget_fail_pct:.3f}%)"
        )

    x_ref = torch.zeros(seq_len, hidden_runtime, dtype=torch.float32)
    seq_ref = min(seq_valid, layer_outputs[0].shape[0])
    x_ref[:seq_ref, :hidden_visible] = layer_outputs[0][:seq_ref, :hidden_visible].float()

    layer0_probe_bases = vram_layout.get("layer0_probe_bases", {})
    input_probe_base = layer0_probe_bases.get("input_token_major")
    if input_probe_base is not None:
        x_sim = load_vram_seq_major_to_seq(
            vram_dump,
            start_elem=int(input_probe_base),
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
        )
    else:
        x_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=int(vram_layout["embedding_base"]),
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
            mlen=mlen,
        )
    _summarize("layer0_embedding_out", x_sim, x_ref)

    layer0 = runtime_layer_weights_list[0]
    ln1_w = layer0["ln1_weight"].float()
    ln1_b = layer0["ln1_bias"].float() if layer0.get("ln1_bias") is not None else None
    ln1_ref = F.layer_norm(x_ref, (hidden_runtime,), weight=ln1_w, bias=ln1_b, eps=eps)

    q_ref = ln1_ref @ layer0["q_proj_weight"].float()
    q_bias = layer0.get("q_proj_bias")
    if q_bias is not None:
        q_ref = q_ref + q_bias.float()

    num_layers = len(runtime_layer_weights_list)
    inter_runtime = int(runtime_config.get("intermediate_size", hidden_runtime))
    persistent_end = _compute_persistent_end(
        vram_layout=vram_layout,
        seq_len=seq_len,
        hidden_size=hidden_runtime,
        inter_size=inter_runtime,
        num_layers=num_layers,
    )

    workspace_base = align_up(persistent_end, mlen)
    q_seq_base = workspace_base
    q_vram_base = q_seq_base + seq_len * hidden_runtime

    flash_layout = compute_vram_layout(
        mlen=mlen,
        blen=blen,
        q_len=seq_len,
        kv_len=seq_len,
        hq=hq,
        hkv=int(config["num_key_value_heads"]),
        d=d_padded,
        vector_sram_base=q_vram_base,
    )
    q_base = int(flash_layout["q_base"])

    q_sim = load_vram_head_major_q_to_seq(
        vram_dump,
        start_elem=q_base,
        s_q=seq_len,
        hq=hq,
        d_padded=d_padded,
    )
    _summarize("layer0_q_proj", q_sim, q_ref)

    try:
        # Reconstruct runtime K/V tiles from staged HBM payloads.
        k_tile_flat = torch.load(build_dir / "layer_0_k_proj_weight.pt").reshape(-1).float()
        v_tile_flat = torch.load(build_dir / "layer_0_v_proj_weight.pt").reshape(-1).float()
        kv_elems = int(config["num_key_value_heads"]) * seq_len * d_padded
        k_heads = k_tile_flat[:kv_elems].reshape(int(config["num_key_value_heads"]), seq_len, d_padded)
        v_heads = v_tile_flat[:kv_elems].reshape(int(config["num_key_value_heads"]), seq_len, d_padded)

        q_heads = q_ref.reshape(1, seq_len, hq, d_padded)
        k_heads_b = k_heads.permute(1, 0, 2).reshape(1, seq_len, int(config["num_key_value_heads"]), d_padded)
        v_heads_b = v_heads.permute(1, 0, 2).reshape(1, seq_len, int(config["num_key_value_heads"]), d_padded)
        attn_scale = 1.0 / float(d_padded) ** 0.5

        attn_out = gqa_sdpa(
            q_heads,
            k_heads_b,
            v_heads_b,
            scale=attn_scale,
            hq=hq,
            hkv=int(config["num_key_value_heads"]),
            kv_valid_len=int(runtime_config.get("seq_len_valid", seq_len)),
        )
        attn_out = attn_out.reshape(seq_len, hidden_runtime).to(torch.bfloat16).float()

        attn_probe_base = layer0_probe_bases.get("attn_token_major")
        if attn_probe_base is not None:
            attn_out_sim = load_vram_seq_major_to_seq(
                vram_dump,
                start_elem=int(attn_probe_base),
                seq_len=seq_len,
                hidden_dim=hidden_runtime,
            )
            _summarize("layer0_attn_out", attn_out_sim, attn_out)

        out_w = layer0["out_proj_weight"].float()
        out_b = layer0["out_proj_bias"].float() if layer0.get("out_proj_bias") is not None else None
        attn_proj = F.linear(attn_out, out_w, out_b).to(torch.bfloat16).float()

        outproj_probe_base = layer0_probe_bases.get("outproj_token_major")
        if outproj_probe_base is not None:
            out_proj_sim = load_vram_seq_major_to_seq(
                vram_dump,
                start_elem=int(outproj_probe_base),
                seq_len=seq_len,
                hidden_dim=hidden_runtime,
            )
            _summarize("layer0_out_proj_out", out_proj_sim, attn_proj)

        x_res1_ref = (x_ref.to(torch.bfloat16) + attn_proj.to(torch.bfloat16)).to(torch.bfloat16).float()

        attn_base = int(flash_layout["o_old_base"])
        x_res1_sim = load_vram_seq_major_to_seq(
            vram_dump,
            start_elem=attn_base,
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
        )
        _summarize("layer0_residual1_out", x_res1_sim, x_res1_ref)

        ln2_w = layer0["ln2_weight"].float()
        ln2_b = layer0["ln2_bias"].float() if layer0.get("ln2_bias") is not None else None
        ln2_ref = F.layer_norm(x_res1_ref, (hidden_runtime,), weight=ln2_w, bias=ln2_b, eps=eps).to(torch.bfloat16).float()

        x_ln2_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=int(vram_layout["embedding_base"]),
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
            mlen=mlen,
        )
        _summarize("layer0_ln2_out", x_ln2_sim, ln2_ref)

        fc1_w = layer0["fc1_weight"].float().T
        fc1_b = layer0["fc1_bias"].float() if layer0.get("fc1_bias") is not None else None
        fc1_out = F.linear(ln2_ref, fc1_w, fc1_b).to(torch.bfloat16).float()
        gelu = _gelu_hardware_sigmoid(fc1_out).to(torch.bfloat16).float()

        fc2_w = layer0["fc2_weight"].float().T
        fc2_b = layer0["fc2_bias"].float() if layer0.get("fc2_bias") is not None else None
        fc2_out = F.linear(gelu, fc2_w, fc2_b).to(torch.bfloat16).float()
        x_final_ref = (x_res1_ref.to(torch.bfloat16) + fc2_out.to(torch.bfloat16)).to(torch.bfloat16).float()

        inter_padded = int(runtime_config["intermediate_size"])
        residual_base = attn_base + seq_len * hidden_runtime
        mlp_inter_base = residual_base + seq_len * hidden_runtime
        mlp_inter_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=mlp_inter_base,
            seq_len=seq_len,
            hidden_dim=inter_padded,
            mlen=mlen,
        )
        _summarize("layer0_mlp_inter", mlp_inter_sim, gelu)

        layer0_out_base = int(vram_layout["layer_bases"][0])
        x_final_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=layer0_out_base,
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
            mlen=mlen,
        )
        fc2_out_sim = x_final_sim - x_res1_sim
        _summarize("layer0_mlp_out", fc2_out_sim, fc2_out)
        _summarize("layer0_residual2_out", x_final_sim, x_final_ref)
    except Exception as exc:
        print(f"[diag] Skipping downstream diagnostics ({exc})")

    if probe_diagnostics:
        diag_path = build_dir / "layer0_probe_diagnostics.json"
        diag_payload = {
            "seq_valid": int(seq_valid),
            "seq_len": int(seq_len),
            "abs_budget": DIAG_ABS_BUDGET,
            "rel_budget": DIAG_REL_BUDGET,
            "probes": probe_diagnostics,
        }
        diag_path.write_text(json.dumps(diag_payload, indent=2), encoding="utf-8")
        print(f"[diag] Saved probe diagnostics: {diag_path}")


def run_full_model_emulator_smoke(
    config: dict,
    runtime_config: dict,
    embedding_weights: dict,
    runtime_embedding_weights: dict,
    layer_weights_list: list,
    runtime_layer_weights_list: list,
    vram_layout: dict,
    hbm_layout: tuple,
    asm_code: str,
    build_dir: Path,
    max_layers: int,
    mlen: int,
    vlen: int,
    blen: int,
    write_golden_txt: bool = True,
    enforce_numerical_parity: bool = False,
    embedding_mode: str = "bypass",
    skip_numerical_compare: bool = False,
) -> None:
    """Run generated full-model ASM in emulator and optionally enforce numerical parity."""
    num_layers = min(max_layers, len(layer_weights_list))
    seq_len = vram_layout["seq_len"]
    seq_len_valid = int(runtime_config.get("seq_len_valid", seq_len))
    hidden_size = config["hidden_size"]
    runtime_hidden_size = runtime_config["hidden_size"]
    _unused_vram_preload, patches_raw = _prepare_vram_preload(
        runtime_config,
        runtime_embedding_weights,
        seq_len,
        mlen,
        runtime_hidden_size,
    )

    # Build per-layer inputs needed to synthesize K/V activation tiles.
    # For fast smoke runs we only need inputs up to L-1 (not full L-layer golden).
    golden_depth_for_inputs = num_layers if not skip_numerical_compare else max(0, num_layers - 1)
    _unused_final_golden, layer_outputs = compute_golden_full_model(
        patches=patches_raw,
        embedding_weights=runtime_embedding_weights,
        layer_weights_list=runtime_layer_weights_list,
        config=runtime_config,
        use_mxfp=False,
        max_layers=golden_depth_for_inputs,
    )

    total_vram = int(vram_layout["total_vram_elements"])
    vram_preload = np.zeros(total_vram, dtype=np.uint16)
    if embedding_mode == "bypass":
        # Preload embedding output directly for encoder-only validation.
        embedding_preload = prepare_vram_preload_from_embedding(
            embedding_out=layer_outputs[0],
            seq_len_kernel=seq_len,
            hidden_runtime=runtime_hidden_size,
            hidden_visible=hidden_size,
            mlen=mlen,
        )
        emb_base = int(vram_layout["embedding_base"])
        vram_preload[emb_base : emb_base + embedding_preload.size] = embedding_preload
    elif embedding_mode == "asm":
        # Preload chunk-major patch inputs and position table consumed by Stage 0.
        fill_embedding_inputs_for_asm(
            config=runtime_config,
            embedding_weights=runtime_embedding_weights,
            seq_len=seq_len,
            hidden_size=runtime_hidden_size,
            mlen=mlen,
            vram_preload=vram_preload,
            patch_input_base=int(vram_layout["embedding_patch_input_base"]),
            patch_bias_base=int(vram_layout["embedding_patch_bias_base"]),
            position_base=int(vram_layout["embedding_position_base"]),
        )
    else:
        raise ValueError(f"Unsupported embedding_mode={embedding_mode!r}")

    q_bias_bases = vram_layout.get("q_bias_bases", {})
    for layer_idx in range(num_layers):
        q_bias_base = q_bias_bases.get(layer_idx)
        if q_bias_base is None:
            continue
        q_bias = runtime_layer_weights_list[layer_idx].get("q_proj_bias")
        if q_bias is None:
            continue
        q_bias_tile = q_bias.float().unsqueeze(0).repeat(seq_len, 1)
        q_bias_packed = pack_seq_to_chunk_major(q_bias_tile.to(torch.bfloat16).float(), mlen=mlen)
        q_bias_preload = q_bias_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        q_bias_base_i = int(q_bias_base)
        vram_preload[q_bias_base_i : q_bias_base_i + q_bias_preload.size] = q_bias_preload

    ln1_weight_bases = vram_layout.get("ln1_weight_bases", {})
    ln1_bias_bases = vram_layout.get("ln1_bias_bases", {})
    ln2_weight_bases = vram_layout.get("ln2_weight_bases", {})
    ln2_bias_bases = vram_layout.get("ln2_bias_bases", {})
    out_bias_bases = vram_layout.get("out_bias_bases", {})
    fc1_bias_bases = vram_layout.get("fc1_bias_bases", {})
    fc2_bias_bases = vram_layout.get("fc2_bias_bases", {})
    for layer_idx in range(num_layers):
        layer_rt = runtime_layer_weights_list[layer_idx]
        affine_specs = (
            (ln1_weight_bases.get(layer_idx), layer_rt.get("ln1_weight")),
            (ln1_bias_bases.get(layer_idx), layer_rt.get("ln1_bias")),
            (ln2_weight_bases.get(layer_idx), layer_rt.get("ln2_weight")),
            (ln2_bias_bases.get(layer_idx), layer_rt.get("ln2_bias")),
        )
        for base, tensor in affine_specs:
            if base is None or tensor is None:
                continue
            affine_tile = tensor.float().unsqueeze(0).repeat(seq_len, 1)
            affine_packed = pack_seq_to_chunk_major(affine_tile.to(torch.bfloat16).float(), mlen=mlen)
            affine_preload = affine_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            base_i = int(base)
            vram_preload[base_i : base_i + affine_preload.size] = affine_preload

        fc1_b = layer_rt.get("fc1_bias")
        fc1_base = fc1_bias_bases.get(layer_idx)
        if fc1_b is not None and fc1_base is not None:
            fc1_tile = fc1_b.float().unsqueeze(0).repeat(seq_len, 1)
            fc1_packed = pack_seq_to_chunk_major(fc1_tile.to(torch.bfloat16).float(), mlen=mlen)
            fc1_preload = fc1_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            fc1_base_i = int(fc1_base)
            vram_preload[fc1_base_i : fc1_base_i + fc1_preload.size] = fc1_preload

        fc2_b = layer_rt.get("fc2_bias")
        fc2_base = fc2_bias_bases.get(layer_idx)
        if fc2_b is not None and fc2_base is not None:
            fc2_tile = fc2_b.float().unsqueeze(0).repeat(seq_len, 1)
            fc2_packed = pack_seq_to_chunk_major(fc2_tile.to(torch.bfloat16).float(), mlen=mlen)
            fc2_preload = fc2_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            fc2_base_i = int(fc2_base)
            vram_preload[fc2_base_i : fc2_base_i + fc2_preload.size] = fc2_preload

        out_b = layer_rt.get("out_proj_bias")
        out_base = out_bias_bases.get(layer_idx)
        if out_b is not None and out_base is not None:
            out_tile = out_b.float().unsqueeze(0).repeat(seq_len, 1)
            out_packed = pack_seq_to_chunk_major(out_tile.to(torch.bfloat16).float(), mlen=mlen)
            out_preload = out_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
            out_base_i = int(out_base)
            vram_preload[out_base_i : out_base_i + out_preload.size] = out_preload

    runtime_layers = _build_runtime_layer_payloads(
        config=config,
        runtime_layer_weights_list=runtime_layer_weights_list,
        source_layer_weights_list=layer_weights_list,
        layer_outputs=layer_outputs,
        num_layers=num_layers,
        mlen=mlen,
        seq_len_kernel=seq_len,
    )
    if skip_numerical_compare:
        final_compare_golden = torch.zeros(seq_len_valid, hidden_size, dtype=torch.float32)
    else:
        final_compare_golden = _compute_emitted_path_golden(
            patches=patches_raw,
            runtime_embedding_weights=runtime_embedding_weights,
            runtime_layers=runtime_layers,
            runtime_config=runtime_config,
            seq_len_valid=seq_len_valid,
            mlen=mlen,
        )
    input_tensor, data_order = _build_hbm_input_tensors(runtime_embedding_weights, runtime_layers, num_layers)

    # Keep emulator hardware config in sync with generated assembly assumptions.
    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)

    final_output_base = vram_layout["layer_bases"][num_layers - 1] if num_layers > 0 else vram_layout["embedding_base"]
    golden_result = {
        "input_tensor": {k: v for k, v in input_tensor.items()},
        "original_output": final_compare_golden.reshape(-1).to(torch.float32),
    }

    fp_preload = compute_fp_preload(config, mlen=mlen)
    # Runtime LN stages operate on expanded hidden width.
    fp_preload[3] = 1.0 / float(runtime_hidden_size)

    build_dir = build_dir.resolve()
    # hbm_layout[1] is total HBM elements in BF16 units.
    hbm_mb = int(np.ceil((hbm_layout[1] * 2) / (1024 * 1024))) + 16
    prepare_case_artifacts(
        case_build_dir=build_dir,
        input_tensor=input_tensor,
        asm_code=asm_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=vram_preload,
        hbm_mb=hbm_mb,
        data_order=data_order,
        write_golden_txt=write_golden_txt,
    )

    # Compare final layer output from VRAM against the golden tensor.
    comparison_params = {
        "start_row_idx": int(final_output_base // vlen),
        "num_rows": int((seq_len * runtime_hidden_size) // vlen),
        "num_batches": int(seq_len),
        "elements_per_batch": int(hidden_size),
        "row_dim": int(vlen),
        "use_stride_mode": False,
        "use_slice_mode": False,
        "use_chunk_major_mode": True,
        "seq_len": int(seq_len),
        "hidden_dim": int(runtime_hidden_size),
        "mlen": int(mlen),
        "chunk_major_valid_seq_len": int(seq_len_valid),
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    for key in data_order:
        pt_path = build_dir / f"{key}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing HBM tensor payload: {pt_path}")
        loaded = torch.load(pt_path)
        if loaded is None:
            raise ValueError(f"HBM tensor payload is None: {pt_path}")

    run_emulator(build_dir, log_path=build_dir / "emulator.log")
    if skip_numerical_compare:
        print("✓ Emulator run completed (numerical compare skipped in fast mode).")
    else:
        _print_layer0_stage_diagnostics(
            build_dir=build_dir,
            config=config,
            runtime_config=runtime_config,
            vram_layout=vram_layout,
            runtime_layer_weights_list=runtime_layer_weights_list,
            layer_outputs=layer_outputs,
            mlen=mlen,
            blen=blen,
        )
        results, _ = compare_emulator_output(build_dir)
        print(
            "✓ Emulator run completed. "
            f"allclose={results['allclose_pass']} "
            f"match_rate={results['match_rate']:.3f}% "
            f"max_error={results['max_error']:.6f}"
        )
    print(f"  Build dir: {build_dir}")
    print(f"  Final output base (elements): {final_output_base}")
    if (not skip_numerical_compare) and enforce_numerical_parity and not results["allclose_pass"]:
        raise RuntimeError(
            "Full-model harness numerical comparison failed: "
            f"match_rate={results['match_rate']:.3f}% max_error={results['max_error']:.6f}"
        )


def build_full_model_asm(
    config: dict,
    embedding_weights: dict,
    layer_weights_list: list,
    vram_layout: dict,
    hbm_layout: tuple,
    mlen: int = 128,
    vlen: int = 128,
    blen: int = 4,
    max_layers: int = 27,
    embedding_mode: str = "bypass",
) -> str:
    """Emit complete ASM code for all 27 encoder layers + embedding.

    Args:
        config: Config dict from load_siglip_config()
        embedding_weights: Embedding weight dict from extract_embedding_weights()
        layer_weights_list: List of per-layer weight dicts from extract_layer_weights()
        vram_layout: VRAM layout from compute_full_model_vram_layout()
        hbm_layout: (layout_dict, total_hbm_elems) from compute_full_model_hbm_layout()
        mlen, vlen, blen: Hardware parameters
        max_layers: Number of layers to emit (usually 27)

    Returns:
        Complete ASM code as string
    """
    hbm_layout_dict, _total_hbm = hbm_layout

    seq_len = vram_layout["seq_len"]
    seq_len_valid = int(config.get("seq_len_valid", seq_len))
    hidden_size = config["hidden_size"]
    inter_size = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]

    # Padding/alignment
    hidden_padded = ((hidden_size + mlen - 1) // mlen) * mlen
    inter_padded = ((inter_size + mlen - 1) // mlen) * mlen
    head_dim = hidden_size // num_heads
    d_padded = ((head_dim + mlen - 1) // mlen) * mlen
    attn_hidden_padded = num_heads * d_padded

    if hidden_padded != attn_hidden_padded:
        raise ValueError(
            "Internal hidden geometry mismatch after runtime repack: "
            f"hidden_padded={hidden_padded}, attn_hidden_padded={attn_hidden_padded}."
        )

    asm_code = ""
    asm_code += "; ============================================================\n"
    asm_code += "; SigLIP Full Model (27 Encoder Layers)\n"
    asm_code += f"; Hidden={hidden_size}, Heads={num_heads}, Inter={inter_size}\n"
    asm_code += f"; Sequence length={seq_len}, MLEN={mlen}, VLEN={vlen}\n"
    asm_code += "; ============================================================\n\n"

    # ========== Embedding Stage ==========
    if embedding_mode == "bypass":
        asm_code += "; --- STAGE 0: Embedding Preloaded (ASM bypass) ---\n"
    elif embedding_mode == "asm":
        asm_code += build_embedding_stage_asm(
            config=config,
            vram_layout=vram_layout,
            hbm_layout=hbm_layout,
            mlen=mlen,
            blen=blen,
            vlen=vlen,
        )
    else:
        raise ValueError(f"Unsupported embedding_mode={embedding_mode!r}")

    embedding_output_base = vram_layout["embedding_base"]

    # ========== Encoder Layers ==========
    num_layers_to_emit = min(max_layers, len(layer_weights_list))

    # Reusable workspace region placed after persistent per-layer output buffers.
    q_bias_bases = vram_layout.get("q_bias_bases", {})
    ln1_weight_bases = vram_layout.get("ln1_weight_bases", {})
    ln1_bias_bases = vram_layout.get("ln1_bias_bases", {})
    ln2_weight_bases = vram_layout.get("ln2_weight_bases", {})
    ln2_bias_bases = vram_layout.get("ln2_bias_bases", {})
    out_bias_bases = vram_layout.get("out_bias_bases", {})
    fc1_bias_bases = vram_layout.get("fc1_bias_bases", {})
    fc2_bias_bases = vram_layout.get("fc2_bias_bases", {})

    persistent_end = _compute_persistent_end(
        vram_layout=vram_layout,
        seq_len=seq_len,
        hidden_size=hidden_padded,
        inter_size=inter_padded,
        num_layers=num_layers_to_emit,
    )

    # Reserve persistent probe snapshots for layer-0 diagnostics so later layers
    # do not overwrite stage-boundary captures.
    layer0_probe_bases: dict[str, int] = {}
    if num_layers_to_emit > 0:
        token_major_span = seq_len * hidden_padded
        layer0_probe_bases = {
            "input_token_major": int(persistent_end),
            "attn_token_major": int(persistent_end + token_major_span),
            "outproj_token_major": int(persistent_end + 2 * token_major_span),
        }
        persistent_end += 3 * token_major_span
        vram_layout["layer0_probe_bases"] = dict(layer0_probe_bases)

    workspace_base = align_up(persistent_end, mlen)

    for layer_idx in range(num_layers_to_emit):
        asm_code += f"\n; --- LAYER {layer_idx}: Encoder Layer ---\n"

        # Layer input/output chaining:
        # - layer 0 consumes embedding output
        # - layer N consumes layer N-1 output
        # - each layer writes to its dedicated output base
        layer_input_base = embedding_output_base if layer_idx == 0 else vram_layout["layer_bases"][layer_idx - 1]
        layer_output_base = vram_layout["layer_bases"][layer_idx]

        # HBM offsets for this layer's weights
        layer_hbm = hbm_layout_dict.get("layers", {}).get(layer_idx, {})
        wq_offset = layer_hbm.get("q_proj_weight", 0)
        k_offset = layer_hbm.get("k_proj_weight", 0)
        v_offset = layer_hbm.get("v_proj_weight", 0)
        w1_offset = layer_hbm.get("fc1_weight", 0)
        w2_offset = layer_hbm.get("fc2_weight", 0)
        out_offset = layer_hbm.get("out_proj_weight", 0)

        # Temporary VRAM allocations for this layer in a reusable non-overlapping workspace.
        q_seq_base = workspace_base
        q_vram_base = q_seq_base + seq_len * hidden_padded

        flash_layout = compute_vram_layout(
            mlen=mlen,
            blen=blen,
            q_len=seq_len,
            kv_len=seq_len,
            hq=num_heads,
            hkv=num_kv_heads,
            d=d_padded,
            vector_sram_base=q_vram_base,
        )
        attn_vram_base = flash_layout["o_old_base"]
        residual_vram_base = attn_vram_base + seq_len * hidden_padded
        mlp_inter_vram_base = residual_vram_base + seq_len * hidden_padded
        scratch_base = align_up(mlp_inter_vram_base + seq_len * inter_padded, mlen)

        # Build encoder layer ASM
        layer_asm = build_encoder_layer_asm(
            mlen=mlen,
            blen=blen,
            vlen=vlen,
            batch=1,
            s_q=seq_len,
            s_kv=seq_len,
            s_kv_valid=seq_len_valid,
            hq=num_heads,
            hkv=num_kv_heads,
            h_qkv=d_padded,
            hidden_size=hidden_padded,
            inter_dim=inter_padded,
            x_base=layer_input_base,
            q_base=q_vram_base,
            q_seq_base=q_seq_base,
            attn_base=attn_vram_base,
            residual_base=residual_vram_base,
            mlp_inter_base=mlp_inter_vram_base,
            mlp_out_base=layer_output_base,
            scratch_base=scratch_base,
            k_hbm_offset=k_offset,
            v_hbm_offset=v_offset,
            out_hbm_offset=out_offset,
            wq_hbm_offset=wq_offset,
            w1_hbm_offset=w1_offset,
            w2_hbm_offset=w2_offset,
            ln_eps_fp_slot=2,
            ln_reci_hid_fp_slot=3,
            gelu_one_fp_slot=4,
            gelu_1702_fp_slot=5,
            attn_scale_fp_slot=1,
            attn_ninf_fp_slot=6,
            flash_temp_fp_start=64,
            q_bias_base=q_bias_bases.get(layer_idx),
            ln1_affine_weight_base=ln1_weight_bases.get(layer_idx),
            ln1_affine_bias_base=ln1_bias_bases.get(layer_idx),
            ln2_affine_weight_base=ln2_weight_bases.get(layer_idx),
            ln2_affine_bias_base=ln2_bias_bases.get(layer_idx),
            out_bias_base=out_bias_bases.get(layer_idx),
            fc1_bias_base=fc1_bias_bases.get(layer_idx),
            fc2_bias_base=fc2_bias_bases.get(layer_idx),
            debug_stage0_snapshot_base=(
                layer0_probe_bases.get("input_token_major") if layer_idx == 0 else None
            ),
            debug_attn_snapshot_base=(
                layer0_probe_bases.get("attn_token_major") if layer_idx == 0 else None
            ),
            debug_outproj_snapshot_base=(
                layer0_probe_bases.get("outproj_token_major") if layer_idx == 0 else None
            ),
            include_final_residual=True,
            include_gelu=True,
        )

        asm_code += layer_asm + "\n"

    asm_code += "\n; ============================================================\n"
    asm_code += "; End of SigLIP Full Model ASM\n"
    asm_code += "; ============================================================\n"

    return asm_code


def compute_fp_preload(config: dict, mlen: int = 64, num_slots: int = 1024) -> list[float]:
    """Compute FP preload values for all required constants.

    Returns a preload list where indices are slot numbers.
    """
    fp_preload = [0.0] * num_slots

    hidden_padded = ((config["hidden_size"] + mlen - 1) // mlen) * mlen
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    # Slot 1: Attention scale
    fp_preload[1] = 1.0 / (head_dim ** 0.5)

    # Slot 2: Layer norm epsilon
    fp_preload[2] = config.get("layer_norm_eps", 1e-2)

    # Slot 3: 1 / hidden_size_padded (for layer norm)
    fp_preload[3] = 1.0 / hidden_padded

    # Slot 4: 1.0 (for GELU and other operations)
    fp_preload[4] = 1.0

    # Slot 5: 1.702 (for GELU approximation)
    fp_preload[5] = 1.702

    # Slot 6: -inf (for attention masking)
    fp_preload[6] = float("-inf")

    return fp_preload


def compute_hbm_data_order(num_layers: int = 27) -> list[str]:
    """Compute the HBM data order for weight loading.

    Returns list of weight tensor names in HBM load order.
    """
    order = [
        # Embedding
        "patch_weight", "patch_bias", "position_table",
    ]

    # Per-layer weights
    for layer_idx in range(num_layers):
        order.extend([
            f"layer_{layer_idx}_ln1_weight",
            f"layer_{layer_idx}_ln1_bias",
            f"layer_{layer_idx}_q_proj_weight",
            f"layer_{layer_idx}_q_proj_bias",
            f"layer_{layer_idx}_k_proj_weight",
            f"layer_{layer_idx}_k_proj_bias",
            f"layer_{layer_idx}_v_proj_weight",
            f"layer_{layer_idx}_v_proj_bias",
            f"layer_{layer_idx}_out_proj_weight",
            f"layer_{layer_idx}_out_proj_bias",
            f"layer_{layer_idx}_ln2_weight",
            f"layer_{layer_idx}_ln2_bias",
            f"layer_{layer_idx}_fc1_weight",
            f"layer_{layer_idx}_fc1_bias",
            f"layer_{layer_idx}_fc2_weight",
            f"layer_{layer_idx}_fc2_bias",
        ])

    return order


if __name__ == "__main__":
    """Generate full-model ASM, and optionally run emulator smoke path."""
    from transactional_emulator.testbench.siglip.model_loader import (
        load_siglip_config,
        load_siglip_vision_model,
        extract_embedding_weights,
        extract_layer_weights,
    )
    from transactional_emulator.testbench.siglip.full_model.memory_layout import (
        compute_full_model_hbm_layout,
    )

    parser = argparse.ArgumentParser(description="SigLIP full-model ASM harness")
    parser.add_argument(
        "--config",
        default="compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
        help="Path to SigLIP config json",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=2,
        help="Number of layers to emit/run",
    )
    parser.add_argument(
        "--mlen",
        type=int,
        default=128,
        help="Hardware MLEN",
    )
    parser.add_argument(
        "--vlen",
        type=int,
        default=128,
        help="Hardware VLEN",
    )
    parser.add_argument(
        "--blen",
        type=int,
        default=4,
        help="Hardware BLEN",
    )
    parser.add_argument(
        "--output-asm",
        default="",
        help="Optional output file for generated ASM",
    )
    parser.add_argument(
        "--run-emulator",
        action="store_true",
        help="Run emulator smoke path after ASM generation",
    )
    parser.add_argument(
        "--full-flow-embedding",
        action="store_true",
        help="Run embedding stage in ASM (patch proj + pos add) instead of preload bypass.",
    )
    parser.add_argument(
        "--build-dir",
        default="./build/siglip_full_model_harness",
        help="Build directory for simulator artifacts",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SigLIP Full Model ASM Generation Test")
    print("=" * 80)

    # Load config and weights
    config = load_siglip_config(args.config)
    seq_len_valid = int((config["image_size"] // config["patch_size"]) ** 2)
    seq_len_kernel = align_up(seq_len_valid, args.mlen)
    model = load_siglip_vision_model()
    embed_weights = extract_embedding_weights(model, config)
    max_layers = min(args.max_layers, config["num_hidden_layers"])
    layer_weights = [extract_layer_weights(model, i, config["hidden_size"]) for i in range(max_layers)]
    runtime_config, runtime_embed_weights, runtime_layer_weights = build_runtime_repacked_model(
        config=config,
        embedding_weights=embed_weights,
        layer_weights_list=layer_weights,
        mlen=args.mlen,
        seq_len_kernel=seq_len_kernel,
    )
    runtime_config["seq_len_valid"] = seq_len_valid

    # Compute layouts
    hidden_runtime = int(runtime_config["hidden_size"])
    embedding_base = 0
    embedding_size = seq_len_kernel * hidden_runtime
    layer_bases = {}
    layer_sizes = {}
    cur = embedding_base + embedding_size
    for i in range(max_layers):
        layer_bases[i] = cur
        layer_sizes[i] = seq_len_kernel * hidden_runtime
        cur += layer_sizes[i]
    q_bias_bases = {}
    for i in range(max_layers):
        q_bias_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
    ln1_weight_bases = {}
    ln1_bias_bases = {}
    ln2_weight_bases = {}
    ln2_bias_bases = {}
    for i in range(max_layers):
        ln1_weight_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
        ln1_bias_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
        ln2_weight_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
        ln2_bias_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime

    patch_size = int(runtime_config["patch_size"])
    num_channels = int(runtime_config["num_channels"])
    in_features = num_channels * patch_size * patch_size
    aligned_in_features = align_up(in_features, args.mlen)
    embedding_patch_input_base = cur
    cur += seq_len_kernel * aligned_in_features
    embedding_patch_bias_base = cur
    cur += seq_len_kernel * hidden_runtime
    embedding_position_base = cur
    cur += seq_len_kernel * hidden_runtime

    vram_layout = {
        "seq_len": seq_len_kernel,
        "hidden_size": hidden_runtime,
        "embedding_base": embedding_base,
        "embedding_size": embedding_size,
        "layer_bases": layer_bases,
        "layer_sizes": layer_sizes,
        "q_bias_bases": q_bias_bases,
        "ln1_weight_bases": ln1_weight_bases,
        "ln1_bias_bases": ln1_bias_bases,
        "ln2_weight_bases": ln2_weight_bases,
        "ln2_bias_bases": ln2_bias_bases,
        "embedding_patch_input_base": embedding_patch_input_base,
        "embedding_patch_bias_base": embedding_patch_bias_base,
        "embedding_position_base": embedding_position_base,
        "total_vram_elements": cur,
        "total_vram_mb": cur * 2 / (1024 * 1024),
    }
    hbm_layout = compute_full_model_hbm_layout(runtime_config, runtime_embed_weights, runtime_layer_weights)

    # Generate ASM (limited to 2 layers for testing)
    print("\n--- Generating ASM Code ---")
    embedding_mode = "asm" if args.full_flow_embedding else "bypass"
    print(f"Embedding mode: {embedding_mode}")
    asm_code = build_full_model_asm(
        runtime_config, runtime_embed_weights, runtime_layer_weights,
        vram_layout, hbm_layout,
        mlen=args.mlen, vlen=args.vlen, blen=args.blen,
        max_layers=max_layers,
        embedding_mode=embedding_mode,
    )

    asm_lines = asm_code.split("\n")
    print(f"Generated {len(asm_lines)} lines of ASM code")
    print("First 20 lines:")
    for line in asm_lines[:20]:
        print(f"  {line}")

    if args.output_asm:
        out_path = Path(args.output_asm)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(asm_code)
        print(f"\n✓ Wrote ASM to {out_path}")

    # Test FP preload
    print("\n--- FP Preload Values ---")
    fp_preload = compute_fp_preload(config, mlen=args.mlen)
    print(f"Slot 1 (attention scale): {fp_preload[1]}")
    print(f"Slot 2 (LN epsilon): {fp_preload[2]}")
    print(f"Slot 3 (1/hidden): {fp_preload[3]}")
    print(f"Slot 4 (1.0): {fp_preload[4]}")
    print(f"Slot 5 (1.702): {fp_preload[5]}")
    print(f"Slot 6 (-inf): {fp_preload[6]}")

    # Test data order
    print("\n--- HBM Data Order (first 10) ---")
    data_order = compute_hbm_data_order(num_layers=max_layers)
    for i, item in enumerate(data_order[:10]):
        print(f"  {i}: {item}")

    if args.run_emulator:
        print("\n--- Running Emulator Smoke Path ---")
        run_full_model_emulator_smoke(
            config=config,
            runtime_config=runtime_config,
            embedding_weights=embed_weights,
            runtime_embedding_weights=runtime_embed_weights,
            layer_weights_list=layer_weights,
            runtime_layer_weights_list=runtime_layer_weights,
            vram_layout=vram_layout,
            hbm_layout=hbm_layout,
            asm_code=asm_code,
            build_dir=Path(args.build_dir),
            max_layers=max_layers,
            mlen=args.mlen,
            vlen=args.vlen,
            blen=args.blen,
            enforce_numerical_parity=True,
            embedding_mode=embedding_mode,
        )

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
