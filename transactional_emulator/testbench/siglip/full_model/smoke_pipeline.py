"""Smoke harness pipeline helpers for full-model SigLIP runs."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_emulator
from transactional_emulator.testbench.siglip.full_model.embedding_flow import (
    fill_embedding_inputs_for_asm,
    prepare_vram_preload_from_embedding,
)
from transactional_emulator.testbench.siglip.full_model.golden_reference import (
    compute_golden_embedding,
    compute_golden_full_model,
)
from transactional_emulator.testbench.siglip.full_model.runtime_prep import build_hidden_index_map
from transactional_emulator.testbench.siglip.utils.core import align_up
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    prepare_case_artifacts,
    write_comparison_params,
)
from transactional_emulator.testbench.siglip.utils.math import gqa_sdpa
from transactional_emulator.testbench.siglip.utils.vram import pack_seq_to_chunk_major

__all__ = [
    "SmokeRuntimeInputs",
    "prepare_smoke_case_artifacts_and_compare_params",
    "prepare_smoke_runtime_inputs",
    "validate_payloads_and_execute_smoke",
]


@dataclass(frozen=True)
class SmokeRuntimeInputs:
    """Prepared runtime artifacts needed to execute smoke harness stages."""

    num_layers: int
    seq_len: int
    seq_len_valid: int
    hidden_size: int
    runtime_hidden_size: int
    layer_outputs: list[torch.Tensor]
    runtime_layers: list[dict]
    final_compare_golden: torch.Tensor
    input_tensor: dict[str, torch.Tensor]
    data_order: list[str]
    vram_preload: np.ndarray


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
        hidden_index = build_hidden_index_map(source_hidden, hidden_size, num_heads)

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

        runtime["k_proj_weight"] = k_slot.reshape_as(src_runtime["k_proj_weight"])
        runtime["v_proj_weight"] = v_slot.reshape_as(src_runtime["v_proj_weight"])
        runtime_layers.append(runtime)

    return runtime_layers


def _gelu_hardware_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Hardware GELU approximation used by gelu_asm: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


def _compute_emitted_path_golden(
    *,
    patches: torch.Tensor,
    runtime_embedding_weights: dict,
    runtime_layers: list,
    runtime_config: dict,
    seq_len_valid: int,
) -> torch.Tensor:
    """Compute golden output for the exact emitted path using runtime K/V payloads."""
    x_full = compute_golden_embedding(
        patches,
        runtime_embedding_weights,
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


def _build_hbm_input_tensors(
    embedding_weights: dict,
    layer_weights_list: list,
    num_layers: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Build HBM preload tensors keyed by filename stem in create_mem_for_sim order."""
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


def _populate_runtime_bias_preloads(
    *,
    vram_preload: np.ndarray,
    runtime_layer_weights_list: list,
    vram_layout: dict,
    num_layers: int,
    seq_len: int,
    mlen: int,
) -> None:
    """Populate per-layer bias/affine buffers in VRAM preload."""

    def _store_preload(base: int | None, tensor: torch.Tensor | None) -> None:
        if base is None or tensor is None:
            return
        tiled = tensor.float().unsqueeze(0).repeat(seq_len, 1)
        packed = pack_seq_to_chunk_major(tiled.to(torch.bfloat16).float(), mlen=mlen)
        preload = packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        base_i = int(base)
        vram_preload[base_i : base_i + preload.size] = preload

    q_bias_bases = vram_layout.get("q_bias_bases", {})
    ln1_weight_bases = vram_layout.get("ln1_weight_bases", {})
    ln1_bias_bases = vram_layout.get("ln1_bias_bases", {})
    ln2_weight_bases = vram_layout.get("ln2_weight_bases", {})
    ln2_bias_bases = vram_layout.get("ln2_bias_bases", {})
    out_bias_bases = vram_layout.get("out_bias_bases", {})
    fc1_bias_bases = vram_layout.get("fc1_bias_bases", {})
    fc2_bias_bases = vram_layout.get("fc2_bias_bases", {})

    for layer_idx in range(num_layers):
        layer_rt = runtime_layer_weights_list[layer_idx]
        preload_specs = (
            (q_bias_bases.get(layer_idx), layer_rt.get("q_proj_bias")),
            (ln1_weight_bases.get(layer_idx), layer_rt.get("ln1_weight")),
            (ln1_bias_bases.get(layer_idx), layer_rt.get("ln1_bias")),
            (ln2_weight_bases.get(layer_idx), layer_rt.get("ln2_weight")),
            (ln2_bias_bases.get(layer_idx), layer_rt.get("ln2_bias")),
            (fc1_bias_bases.get(layer_idx), layer_rt.get("fc1_bias")),
            (fc2_bias_bases.get(layer_idx), layer_rt.get("fc2_bias")),
            (out_bias_bases.get(layer_idx), layer_rt.get("out_proj_bias")),
        )
        for base, tensor in preload_specs:
            _store_preload(base, tensor)


def prepare_smoke_runtime_inputs(
    *,
    config: dict,
    runtime_config: dict,
    runtime_embedding_weights: dict,
    layer_weights_list: list,
    runtime_layer_weights_list: list,
    vram_layout: dict,
    max_layers: int,
    mlen: int,
    skip_numerical_compare: bool,
    embedding_mode: str,
) -> SmokeRuntimeInputs:
    """Prepare runtime tensors and preload buffers for smoke execution."""
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

    _populate_runtime_bias_preloads(
        vram_preload=vram_preload,
        runtime_layer_weights_list=runtime_layer_weights_list,
        vram_layout=vram_layout,
        num_layers=num_layers,
        seq_len=seq_len,
        mlen=mlen,
    )

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
        )

    input_tensor, data_order = _build_hbm_input_tensors(runtime_embedding_weights, runtime_layers, num_layers)

    return SmokeRuntimeInputs(
        num_layers=num_layers,
        seq_len=seq_len,
        seq_len_valid=seq_len_valid,
        hidden_size=hidden_size,
        runtime_hidden_size=runtime_hidden_size,
        layer_outputs=layer_outputs,
        runtime_layers=runtime_layers,
        final_compare_golden=final_compare_golden,
        input_tensor=input_tensor,
        data_order=data_order,
        vram_preload=vram_preload,
    )


def prepare_smoke_case_artifacts_and_compare_params(
    *,
    build_dir: Path,
    vram_layout: dict,
    num_layers: int,
    hbm_layout: tuple,
    asm_code: str,
    final_compare_golden: torch.Tensor,
    fp_preload: list[float],
    mlen: int,
    vlen: int,
    seq_len: int,
    seq_len_valid: int,
    hidden_size: int,
    runtime_hidden_size: int,
    input_tensor: dict[str, torch.Tensor],
    data_order: list[str],
    vram_preload: np.ndarray,
    write_golden_txt: bool,
) -> tuple[Path, int]:
    """Prepare case artifacts and comparison params for smoke execution."""
    final_output_base = (
        vram_layout["layer_bases"][num_layers - 1] if num_layers > 0 else vram_layout["embedding_base"]
    )
    golden_result = {
        "input_tensor": {k: v for k, v in input_tensor.items()},
        "original_output": final_compare_golden.reshape(-1).to(torch.float32),
    }

    resolved_build_dir = build_dir.resolve()
    # hbm_layout[1] is total HBM elements in BF16 units.
    hbm_mb = int(np.ceil((hbm_layout[1] * 2) / (1024 * 1024))) + 16
    prepare_case_artifacts(
        case_build_dir=resolved_build_dir,
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
    write_comparison_params(
        resolved_build_dir,
        start_row_idx=int(final_output_base // vlen),
        num_rows=int((seq_len * runtime_hidden_size) // vlen),
        num_batches=int(seq_len),
        elements_per_batch=int(hidden_size),
        use_stride_mode=False,
        use_slice_mode=False,
        extra_params={
            "row_dim": int(vlen),
            "use_chunk_major_mode": True,
            "seq_len": int(seq_len),
            "hidden_dim": int(runtime_hidden_size),
            "mlen": int(mlen),
            "chunk_major_valid_seq_len": int(seq_len_valid),
        },
    )

    return resolved_build_dir, int(final_output_base)


def validate_payloads_and_execute_smoke(
    *,
    build_dir: Path,
    data_order: list[str],
    skip_numerical_compare: bool,
    config: dict,
    runtime_config: dict,
    vram_layout: dict,
    runtime_layer_weights_list: list,
    layer_outputs: list,
    mlen: int,
    blen: int,
    diagnostics_fn: Callable[..., None],
) -> dict | None:
    """Validate generated payloads and execute emulator run (+ optional compare)."""
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
        return None

    diagnostics_fn(
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
    return results
