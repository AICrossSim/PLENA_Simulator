import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.emulator_runner import compare_emulator_output
from transactional_emulator.testbench.siglip.local_asm_templates.embedding_blocks import (
    append_position_add_asm,
    build_embedding_projection_asm,
)
from transactional_emulator.testbench.siglip.local_asm_templates.encoder_layer_blocks import build_encoder_layer_asm
from transactional_emulator.testbench.siglip.local_asm_templates.layout import compute_hbm_offsets, compute_vram_layout
from transactional_emulator.testbench.siglip.mlp.siglip_mlp_test import gelu_with_bf16_intermediates, quantize_to_mxfp
from transactional_emulator.testbench.siglip.utils.core import (
    json_default,
    resolve_position_embedding,
    resolve_vision_encoder_layer,
    tensor_metrics,
)
from transactional_emulator.testbench.siglip.utils.vram import (
    load_vram_bf16,
    load_vram_chunk_major_to_seq,
    pack_seq_to_chunk_major,
)
from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    gqa_sdpa,
    projection_matmul_k_split_visible,
    quantize_flattened_like_hbm,
)
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    format_siglip_extended_run_config,
    load_siglip_harness_run_config,
    prepare_case_and_run_emulator,
    resolve_siglip_vram_dump_path,
    write_comparison_params,
)


EMBED_HBM_ORDER = ["act_tensor", "weights", "position_tensor"]
ENC_HBM_ORDER = ["WQ", "K", "V", "W1", "W2"]


def _load_vram_embedding_result(
    vram_bin_file: Path,
    *,
    start_elem: int,
    seq_len: int,
    hidden_dim: int,
) -> torch.Tensor:
    flat = load_vram_bf16(vram_bin_file, num_elements=seq_len * hidden_dim, start_elem=start_elem)
    return flat.reshape(seq_len, hidden_dim)


def _extract_patches(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    # [B, C, H, W] -> [B, num_patches, C*patch*patch]
    patches = F.unfold(pixel_values.float(), kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).contiguous()


def emit_and_run_integration_test(build_dir: Path):
    torch.manual_seed(0)

    run_cfg = load_siglip_harness_run_config(
        build_dir=build_dir,
        mlen_default=128,
        vlen_default=None,
        q_chunk_default=None,
        max_chunks_default=1,
        inter_dim_default=1024,
    )
    mlen = run_cfg.mlen
    vlen = run_cfg.vlen
    blen = 4
    batch = 1
    max_q_chunk = run_cfg.max_q_chunk
    max_chunks = run_cfg.max_chunks

    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)
    build_dir.mkdir(parents=True, exist_ok=True)
    print(format_siglip_extended_run_config(run_cfg))

    model_id = "google/siglip-so400m-patch14-384"
    print(f"Loading model: {model_id}")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    vision_root = getattr(model, "vision_model", model)
    layer0 = resolve_vision_encoder_layer(model, layer_idx=0)

    image_size = int(getattr(vision_root.config, "image_size", 384))
    patch_size = int(getattr(vision_root.config, "patch_size", 14))
    num_channels = int(getattr(vision_root.config, "num_channels", 3))
    hidden_size = int(getattr(vision_root.config, "hidden_size", 1152))
    hq = int(getattr(vision_root.config, "num_attention_heads", 16))
    h_qkv = hidden_size // hq
    hkv = int(getattr(layer0.self_attn, "num_key_value_heads", hq))

    inter_dim = int(os.environ.get("SIGLIP_INTER_DIM", "1024"))
    if inter_dim <= 0:
        raise ValueError("SIGLIP_INTER_DIM must be > 0")

    s_full = (image_size // patch_size) ** 2
    s_kv_valid = int(os.environ.get("SIGLIP_KV_VALID_LEN", str(s_full)))
    if s_kv_valid <= 0 or s_kv_valid > s_full:
        raise ValueError(f"SIGLIP_KV_VALID_LEN must be in [1, {s_full}], got {s_kv_valid}")
    s_kv_kernel = ((s_kv_valid + mlen - 1) // mlen) * mlen
    mask_padded_kv = os.environ.get("SIGLIP_MASK_PADDED_KV", "1") == "1"

    d_padded = mlen
    hidden_size_padded = hq * d_padded
    aligned_inter_dim = ((inter_dim + mlen - 1) // mlen) * mlen

    eps = 1e-2
    use_padded_attn_scale = os.environ.get("SIGLIP_USE_PADDED_ATTN_SCALE", "0") == "1"
    scale_dim = d_padded if use_padded_attn_scale else h_qkv
    scale = 1.0 / math.sqrt(scale_dim)
    real_data_ratio = MXFP_REAL_DATA_RATIO

    print(
        f"Integration config: s_full={s_full}, hidden={hidden_size}, hq={hq}, hkv={hkv}, "
        f"d_padded={d_padded}, max_q_chunk={max_q_chunk}, max_chunks={max_chunks}"
    )

    # ---------------- Stage A: Patch embedding + position add ----------------
    pixel_values = torch.randn(batch, num_channels, image_size, image_size, dtype=torch.float32)
    patches = _extract_patches(pixel_values, patch_size=patch_size)[0]  # [s_full, in_features]
    in_features = patches.shape[1]

    patch_weight = vision_root.embeddings.patch_embedding.weight.detach().contiguous()
    patch_weight_2d = patch_weight.reshape(patch_weight.shape[0], -1).T.contiguous()
    pos_table = resolve_position_embedding(vision_root)
    position_tensor_hw = pos_table[:s_full, :hidden_size].contiguous().float()
    patch_bias = vision_root.embeddings.patch_embedding.bias.detach().float()
    if patch_bias.numel() != hidden_size:
        raise ValueError(
            f"Unexpected patch bias shape {tuple(patch_bias.shape)} for hidden_size={hidden_size}"
        )
    # Actual SigLIP embedding includes projection bias before positional add.
    # Keep a separate model reference; hardware stage remains projection + position.
    position_tensor_model = position_tensor_hw + patch_bias.unsqueeze(0)

    effective_batch = ((s_full + blen - 1) // blen) * blen
    act_tensor = patches.to(torch.bfloat16)
    if effective_batch != s_full:
        act_tensor = F.pad(act_tensor, (0, 0, 0, effective_batch - s_full))
        position_tensor_hw = F.pad(position_tensor_hw, (0, 0, 0, effective_batch - s_full))
        position_tensor_model = F.pad(position_tensor_model, (0, 0, 0, effective_batch - s_full))

    aligned_in_features = ((in_features + mlen - 1) // mlen) * mlen
    if aligned_in_features != in_features:
        act_tensor = F.pad(act_tensor, (0, aligned_in_features - in_features))
        patch_weight_2d = F.pad(patch_weight_2d, (0, 0, 0, aligned_in_features - in_features))
        in_features = aligned_in_features

    act_mxfp = quantize_to_mxfp(act_tensor.float()).to(torch.bfloat16)
    weights_mxfp = quantize_to_mxfp(patch_weight_2d.float()).to(torch.bfloat16)
    position_mxfp = quantize_to_mxfp(position_tensor_hw).to(torch.bfloat16)
    position_model_mxfp = quantize_to_mxfp(position_tensor_model).to(torch.bfloat16)

    embed_proj_golden = torch.mm(act_mxfp, weights_mxfp).to(torch.bfloat16).float()
    embed_final_golden = (embed_proj_golden + position_mxfp.float()).to(torch.bfloat16).float()
    embed_model_ref = (embed_proj_golden + position_model_mxfp.float()).to(torch.bfloat16).float()

    embed_asm, embed_result_base = build_embedding_projection_asm(
        title="SigLIP Patch+Position Integration Stage",
        shape_batch=s_full,
        in_features=in_features,
        out_features=hidden_size,
        effective_batch=effective_batch,
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        weight_hbm_offset=int((((in_features * effective_batch) * real_data_ratio + 63) // 64) * 64),
        weight_hbm_end=int((((in_features * effective_batch + in_features * hidden_size) * real_data_ratio + 63) // 64) * 64),
    )
    embed_pos_base = embed_result_base + effective_batch * hidden_size
    embed_asm = append_position_add_asm(
        gen_assembly_code=embed_asm,
        result_vram_offset=embed_result_base,
        position_vram_offset=embed_pos_base,
        batch=effective_batch,
        out_features=hidden_size,
        vlen=vlen,
    )

    embed_case = build_dir / "stage_embed"
    embed_input = {
        "act_tensor": act_mxfp,
        "weights": weights_mxfp,
        "position_tensor": position_mxfp,
    }
    embed_golden = {
        "input_tensor": embed_input,
        "original_output": embed_final_golden[:s_full].reshape(-1),
    }
    embed_fp_preload = [0.0, 1e-6, 1.0 / in_features]
    embed_hbm_mb = int(np.ceil((
        (in_features * effective_batch + in_features * hidden_size + effective_batch * hidden_size)
        * real_data_ratio
    ) / (1024 * 1024))) + 16

    prepare_case_and_run_emulator(
        case_build_dir=embed_case,
        input_tensor=embed_input,
        asm_code=embed_asm,
        golden_result=embed_golden,
        fp_preload=embed_fp_preload,
        vram_preload=None,
        hbm_mb=embed_hbm_mb,
        data_order=EMBED_HBM_ORDER,
        copy_vram_dump=True,
        vram_dump_source=resolve_siglip_vram_dump_path(),
    )

    # Decode embedding output via comparator reorder path to avoid layout assumptions.
    embed_results = None
    embed_decode_error = None
    embed_decode_row_dims = [vlen] if vlen == 64 else [vlen, 64]
    for row_dim in embed_decode_row_dims:
        write_comparison_params(
            embed_case,
            start_row_idx=int(embed_result_base // row_dim),
            num_rows=int((effective_batch * hidden_size) // row_dim),
            num_batches=int(effective_batch),
            elements_per_batch=int(hidden_size),
            use_stride_mode=True,
            extra_params={"row_dim": int(row_dim)},
        )
        try:
            embed_results, _ = compare_emulator_output(embed_case)
            break
        except Exception as exc:
            embed_decode_error = exc

    if embed_results is None:
        raise RuntimeError(
            f"Failed to decode stage-embed output with row dims {embed_decode_row_dims}: {embed_decode_error}"
        )

    sim_vals = embed_results.get("simulated_values")
    if sim_vals is None:
        embed_sim = _load_vram_embedding_result(
            embed_case / "vram_dump.bin",
            start_elem=embed_result_base,
            seq_len=effective_batch,
            hidden_dim=hidden_size,
        )[:s_full]
    else:
        if isinstance(sim_vals, torch.Tensor):
            sim_tensor = sim_vals.detach().cpu().float()
        else:
            sim_tensor = torch.tensor(sim_vals, dtype=torch.float32)
        if sim_tensor.numel() % hidden_size != 0:
            raise RuntimeError(
                f"Stage-embed decoded output has {sim_tensor.numel()} values, not divisible by hidden_size={hidden_size}"
            )
        decoded_seq = sim_tensor.numel() // hidden_size
        embed_sim = sim_tensor.reshape(decoded_seq, hidden_size)[:s_full]
    embed_metrics = tensor_metrics(embed_sim, embed_final_golden[:s_full], atol=1e-2, rtol=1e-2)
    embed_model_metrics = tensor_metrics(embed_sim, embed_model_ref[:s_full], atol=1e-2, rtol=1e-2)

    parity_debug = {
        "seed": 0,
        "in_features": int(in_features),
        "effective_batch": int(effective_batch),
        "embed_result_base": int(embed_result_base),
        "position_base": int(embed_pos_base),
        "act_mean": float(act_mxfp.float().mean().item()),
        "weights_mean": float(weights_mxfp.float().mean().item()),
        "position_hw_mean": float(position_mxfp.float().mean().item()),
        "position_model_mean": float(position_model_mxfp.float().mean().item()),
        "embed_decode_row_dims_tried": embed_decode_row_dims,
        "embed_compare_match_rate": float(embed_results.get("match_rate", 0.0)),
        "embed_hw_match_rate": float(embed_metrics["match_rate"]),
        "embed_model_match_rate": float(embed_model_metrics["match_rate"]),
    }
    with open(embed_case / "parity_debug.json", "w") as f:
        json.dump(parity_debug, f, indent=2, default=json_default)

    print(
        f"Embedding stage (HW ref): allclose={embed_metrics['allclose_pass']} "
        f"match_rate={embed_metrics['match_rate']:.3f}% mae={embed_metrics['mae']:.4f}"
    )
    print(
        f"Embedding stage (Model ref): allclose={embed_model_metrics['allclose_pass']} "
        f"match_rate={embed_model_metrics['match_rate']:.3f}% mae={embed_model_metrics['mae']:.4f}"
    )

    embed_min_match = float(os.environ.get("SIGLIP_EMBED_MIN_MATCH", "0"))
    if embed_min_match > 0.0 and float(embed_metrics["match_rate"]) < embed_min_match:
        raise RuntimeError(
            f"Embedding HW-ref match_rate {embed_metrics['match_rate']:.3f}% below "
            f"SIGLIP_EMBED_MIN_MATCH={embed_min_match:.3f}%"
        )

    compare_actual_model = os.environ.get("SIGLIP_COMPARE_ACTUAL_MODEL", "1") == "1"
    actual_layer_out = None
    if compare_actual_model:
        try:
            with torch.no_grad():
                layer0.eval()
                attention_mask = torch.zeros(
                    (1, 1, s_full, s_full),
                    dtype=embed_sim.dtype,
                    device=embed_sim.device,
                )
                actual_out = layer0(
                    embed_sim.unsqueeze(0).float(),
                    attention_mask=attention_mask,
                )
                if isinstance(actual_out, (tuple, list)):
                    actual_out = actual_out[0]
                actual_layer_out = actual_out[0].detach().float().contiguous()
            print("Enabled comparison to actual SigLIP encoder layer output")
        except Exception as exc:
            compare_actual_model = False
            print(f"Actual model comparison disabled due to runtime error: {exc}")

    # ---------------- Stage B: QKV + encoder layer ----------------
    x_in_full = embed_sim
    x_in_full_padded = torch.zeros(s_full, hidden_size_padded, dtype=torch.float32)
    x_in_full_padded[:, :hidden_size] = x_in_full.float()

    ln1_weight = torch.zeros(hidden_size_padded, dtype=torch.float32)
    ln1_bias = torch.zeros(hidden_size_padded, dtype=torch.float32)
    ln1_weight[:hidden_size] = layer0.layer_norm1.weight[:hidden_size].detach().float()
    ln1_bias[:hidden_size] = layer0.layer_norm1.bias[:hidden_size].detach().float()

    x_ln1_full = F.layer_norm(
        x_in_full_padded,
        (hidden_size_padded,),
        weight=ln1_weight,
        bias=ln1_bias,
        eps=layer0.layer_norm1.eps,
    ).to(torch.bfloat16).float()

    attn_mod = layer0.self_attn
    wq_raw = attn_mod.q_proj.weight[:hidden_size, :hidden_size].detach().float().t().contiguous()
    q_bias_raw = attn_mod.q_proj.bias[:hidden_size].detach().float().contiguous()

    wk = attn_mod.k_proj.weight[:hidden_size, :hidden_size].detach().float()
    bk = attn_mod.k_proj.bias[:hidden_size].detach().float()
    wv = attn_mod.v_proj.weight[:hidden_size, :hidden_size].detach().float()
    bv = attn_mod.v_proj.bias[:hidden_size].detach().float()

    k_all = F.linear(x_ln1_full[:, :hidden_size], wk, bk)
    v_all = F.linear(x_ln1_full[:, :hidden_size], wv, bv)

    k_full = k_all.reshape(1, s_full, hkv, h_qkv).contiguous()
    v_full = v_all.reshape(1, s_full, hkv, h_qkv).contiguous()

    w1_raw = layer0.mlp.fc1.weight[:inter_dim, :hidden_size].detach().float().t().contiguous()
    w2_raw = layer0.mlp.fc2.weight[:hidden_size, :inter_dim].detach().float().t().contiguous()
    if aligned_inter_dim != inter_dim:
        w1_raw = F.pad(w1_raw, (0, aligned_inter_dim - inter_dim)).contiguous()
        w2_raw = F.pad(w2_raw, (0, 0, 0, aligned_inter_dim - inter_dim)).contiguous()

    wq_padded = torch.zeros(hidden_size_padded, hidden_size_padded, dtype=torch.float32)
    wq_padded[:hidden_size, :hidden_size] = wq_raw
    q_bias_padded = torch.zeros(hidden_size_padded, dtype=torch.float32)
    q_bias_padded[:hidden_size] = q_bias_raw

    k_padded = torch.zeros(hkv, s_kv_kernel, d_padded, dtype=torch.float32)
    v_padded = torch.zeros(hkv, s_kv_kernel, d_padded, dtype=torch.float32)
    k_padded[:, :s_kv_valid, :h_qkv] = k_full[0, :s_kv_valid, :, :].permute(1, 0, 2).float()
    v_padded[:, :s_kv_valid, :h_qkv] = v_full[0, :s_kv_valid, :, :].permute(1, 0, 2).float()

    w1_padded = torch.zeros(hidden_size_padded, aligned_inter_dim, dtype=torch.float32)
    w1_padded[:hidden_size, :] = w1_raw.float()
    w2_padded = torch.zeros(aligned_inter_dim, hidden_size_padded, dtype=torch.float32)
    w2_padded[:, :hidden_size] = w2_raw.float()

    k_hbm = quantize_flattened_like_hbm(k_padded)
    v_hbm = quantize_flattened_like_hbm(v_padded)
    wq_hbm = quantize_flattened_like_hbm(wq_padded)
    w1_hbm = quantize_flattened_like_hbm(w1_padded)
    w2_hbm = quantize_flattened_like_hbm(w2_padded)

    k_flat = k_padded.reshape(-1).to(torch.float32)
    v_flat = v_padded.reshape(-1).to(torch.float32)
    wq_flat = wq_padded.reshape(-1).to(torch.float32)
    w1_flat = w1_padded.reshape(-1).to(torch.float32)
    w2_flat = w2_padded.reshape(-1).to(torch.float32)

    (wq_off, k_off, v_off, w1_off, w2_off), _ = compute_hbm_offsets(
        [wq_flat.numel(), k_flat.numel(), v_flat.numel(), w1_flat.numel(), w2_flat.numel()],
        real_data_ratio=real_data_ratio,
        align_elems=64,
    )

    hbm_mb = int(np.ceil(((wq_flat.numel() + k_flat.numel() + v_flat.numel() + w1_flat.numel() + w2_flat.numel()) * real_data_ratio) / (1024 * 1024))) + 16

    fp_preload = [0.0] * 1024
    fp_preload[1] = float(scale)
    fp_preload[2] = eps
    fp_preload[3] = 1.0 / hidden_size_padded
    fp_preload[4] = 1.0
    fp_preload[5] = 1.702
    fp_preload[6] = float("-inf")

    chunk_reports = []
    chunk_idx = 0

    for start in range(0, s_full, max_q_chunk):
        if max_chunks > 0 and chunk_idx >= max_chunks:
            break

        end = min(start + max_q_chunk, s_full)
        s_q_actual = end - start
        s_q_kernel = max_q_chunk

        # Use embedding stage output (embed_sim) as Stage B input, padded to kernel size
        x_chunk_actual = embed_sim[start:end].contiguous()
        x_chunk_padded = torch.zeros(s_q_kernel, hidden_size, dtype=torch.float32)
        x_chunk_padded[:s_q_actual, :] = x_chunk_actual
        # Pad to hidden_size_padded for encoder
        x_chunk_padded_full = torch.zeros(s_q_kernel, hidden_size_padded, dtype=torch.float32)
        x_chunk_padded_full[:, :hidden_size] = x_chunk_padded
        x_chunk_bf16 = x_chunk_padded_full.to(torch.bfloat16).float()

        q_bias_tile = q_bias_padded.unsqueeze(0).repeat(s_q_kernel, 1)
        x_chunk_packed = pack_seq_to_chunk_major(x_chunk_bf16, mlen=mlen)
        q_bias_packed = pack_seq_to_chunk_major(q_bias_tile, mlen=mlen)

        x_vram_flat = x_chunk_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        q_bias_flat = q_bias_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

        q_bias_base = int(x_vram_flat.size)
        q_seq_base = int(q_bias_base + q_bias_flat.size)
        q_base = int(q_seq_base + x_vram_flat.size)
        vram_preload = np.concatenate([x_vram_flat, q_bias_flat])

        # Golden computation uses embedding stage output
        # LN1 on embedded input
        x_ln1_chunk = F.layer_norm(
            x_chunk_bf16[:s_q_actual],
            (hidden_size_padded,),
            eps=eps,
        ).to(torch.bfloat16).float()

        q_seq_proj = projection_matmul_k_split_visible(
            x_ln1_chunk,
            wq_hbm,
            mlen=mlen,
        )
        q_seq_proj = (q_seq_proj + q_bias_padded.unsqueeze(0).to(torch.bfloat16).float()).to(torch.bfloat16).float()

        q_ln1 = q_seq_proj.reshape(batch, s_q_actual, hq, d_padded)
        k_tile_gqa = k_hbm.permute(1, 0, 2).reshape(batch, s_kv_kernel, hkv, d_padded)
        v_tile_gqa = v_hbm.permute(1, 0, 2).reshape(batch, s_kv_kernel, hkv, d_padded)
        attn = gqa_sdpa(
            q_ln1,
            k_tile_gqa,
            v_tile_gqa,
            scale,
            hq,
            hkv,
            kv_valid_len=(s_kv_valid if mask_padded_kv else None),
        ).reshape(s_q_actual, hidden_size_padded)
        x_res1 = x_chunk_bf16[:s_q_actual] + attn

        x_ln2 = F.layer_norm(x_res1, (hidden_size_padded,), eps=eps)
        mlp_mid = gelu_with_bf16_intermediates(projection_matmul_k_split_visible(x_ln2, w1_hbm, mlen=mlen))
        mlp_out = projection_matmul_k_split_visible(mlp_mid, w2_hbm, mlen=mlen)
        final_golden = x_res1 + mlp_out

        x_base = 0
        layout = compute_vram_layout(
            mlen=mlen,
            blen=blen,
            q_len=s_q_kernel,
            hq=hq,
            hkv=hkv,
            d=d_padded,
            vector_sram_base=q_base,
        )

        attn_base = layout["o_old_base"]
        residual_base = attn_base + s_q_kernel * hidden_size_padded
        mlp_inter_base = residual_base + s_q_kernel * hidden_size_padded
        mlp_out_base = mlp_inter_base + s_q_kernel * aligned_inter_dim
        scratch_base = mlp_out_base + s_q_kernel * hidden_size_padded

        asm = build_encoder_layer_asm(
            mlen=mlen,
            blen=blen,
            vlen=vlen,
            batch=batch,
            s_q=s_q_kernel,
            s_kv=s_kv_kernel,
            s_kv_valid=s_kv_valid,
            hq=hq,
            hkv=hkv,
            h_qkv=d_padded,
            hidden_size=hidden_size_padded,
            inter_dim=aligned_inter_dim,
            x_base=x_base,
            q_base=q_base,
            q_seq_base=q_seq_base,
            attn_base=attn_base,
            residual_base=residual_base,
            mlp_inter_base=mlp_inter_base,
            mlp_out_base=mlp_out_base,
            scratch_base=scratch_base,
            k_hbm_offset=int(k_off),
            v_hbm_offset=int(v_off),
            wq_hbm_offset=int(wq_off),
            w1_hbm_offset=int(w1_off),
            w2_hbm_offset=int(w2_off),
            ln_eps_fp_slot=2,
            ln_reci_hid_fp_slot=3,
            gelu_one_fp_slot=4,
            gelu_1702_fp_slot=5,
            attn_scale_fp_slot=1,
            attn_ninf_fp_slot=6,
            flash_temp_fp_start=64,
            q_bias_base=q_bias_base,
            include_final_residual=True,
            include_gelu=True,
        )

        chunk_dir = build_dir / f"chunk_{chunk_idx:02d}_{start:04d}_{end:04d}"
        input_tensor = {
            "WQ": wq_flat,
            "K": k_flat,
            "V": v_flat,
            "W1": w1_flat,
            "W2": w2_flat,
            "Q": x_chunk_padded.reshape(-1),
        }
        golden_result = {
            "input_tensor": {
                "Q": input_tensor["Q"],
                "WQ": input_tensor["WQ"],
                "K": input_tensor["K"],
                "V": input_tensor["V"],
                "W1": input_tensor["W1"],
                "W2": input_tensor["W2"],
            },
            "original_output": final_golden.reshape(-1),
        }

        prepare_case_and_run_emulator(
            case_build_dir=chunk_dir,
            input_tensor=input_tensor,
            asm_code=asm,
            golden_result=golden_result,
            fp_preload=fp_preload,
            vram_preload=vram_preload,
            hbm_mb=hbm_mb,
            data_order=ENC_HBM_ORDER,
            copy_vram_dump=True,
            vram_dump_source=resolve_siglip_vram_dump_path(),
        )

        write_comparison_params(
            chunk_dir,
            start_row_idx=int(mlp_out_base // mlen),
            num_rows=int((s_q_actual * hidden_size_padded) // mlen),
            num_batches=int(s_q_actual),
            elements_per_batch=int(hidden_size_padded),
            use_slice_mode=False,
            use_stride_mode=True,
            extra_params={"row_dim": int(mlen)},
        )

        results, _ = compare_emulator_output(chunk_dir)

        final_sim = load_vram_chunk_major_to_seq(
            chunk_dir / "vram_dump.bin",
            start_elem=mlp_out_base,
            seq_len=s_q_actual,
            hidden_dim=hidden_size_padded,
            mlen=mlen,
        )
        stage_metrics = {
            "embedding_stage_hw": embed_metrics,
            "embedding_stage_model": embed_model_metrics,
            "final_out": tensor_metrics(final_sim, final_golden, atol=1e-2, rtol=1e-2),
            "q_proj": tensor_metrics(
                load_vram_bf16(chunk_dir / "vram_dump.bin", s_q_actual * hidden_size_padded, q_base)
                .reshape(hq, s_q_actual, d_padded)
                .permute(1, 0, 2)
                .reshape(s_q_actual, hidden_size_padded),
                q_seq_proj,
                atol=1e-2,
                rtol=1e-2,
            ),
        }

        if actual_layer_out is not None:
            actual_chunk = actual_layer_out[start:end, :hidden_size]
            sim_chunk_visible = final_sim[:, :hidden_size]
            stage_metrics["actual_siglip_layer"] = tensor_metrics(
                sim_chunk_visible,
                actual_chunk,
                atol=1e-2,
                rtol=1e-2,
            )

        report = {
            "chunk_idx": int(chunk_idx),
            "token_start": int(start),
            "token_end": int(end),
            "results": results,
            "stage_metrics": stage_metrics,
            "config": {
                "s_full": s_full,
                "s_kv_valid": s_kv_valid,
                "s_kv_kernel": s_kv_kernel,
                "mlen": mlen,
                "vlen": vlen,
                "hq": hq,
                "hkv": hkv,
                "hidden_size": hidden_size,
                "hidden_size_padded": hidden_size_padded,
                "d_padded": d_padded,
                "inter_dim": inter_dim,
                "aligned_inter_dim": aligned_inter_dim,
                "scale": scale,
            },
        }
        with open(chunk_dir / "comparison_results.json", "w") as f:
            json.dump(report, f, indent=2, default=json_default)

        print(
            f"Chunk {chunk_idx} ({start}-{end}): "
            f"allclose={results['allclose_pass']} "
            f"match_rate={results['match_rate']:.3f}% "
            f"mae={results['mae']:.4f}"
        )
        chunk_reports.append(report)
        chunk_idx += 1

    summary = {
        "chunk_count": len(chunk_reports),
        "allclose_pass": all(r["results"]["allclose_pass"] for r in chunk_reports) if chunk_reports else False,
        "mean_mse": float(np.mean([r["results"]["mse"] for r in chunk_reports])) if chunk_reports else None,
        "mean_mae": float(np.mean([r["results"]["mae"] for r in chunk_reports])) if chunk_reports else None,
        "max_abs_error": float(np.max([r["results"]["max_error"] for r in chunk_reports])) if chunk_reports else None,
        "min_match_rate": float(np.min([r["results"]["match_rate"] for r in chunk_reports])) if chunk_reports else None,
        "embedding_allclose": bool(embed_metrics["allclose_pass"]),
        "embedding_match_rate": float(embed_metrics["match_rate"]),
        "embedding_model_allclose": bool(embed_model_metrics["allclose_pass"]),
        "embedding_model_match_rate": float(embed_model_metrics["match_rate"]),
    }
    if compare_actual_model and chunk_reports:
        model_stage = [
            r["stage_metrics"].get("actual_siglip_layer")
            for r in chunk_reports
            if isinstance(r.get("stage_metrics", {}).get("actual_siglip_layer"), dict)
        ]
        if model_stage:
            summary["actual_model_allclose"] = all(m["allclose_pass"] for m in model_stage)
            summary["actual_model_mean_mse"] = float(np.mean([m["mse"] for m in model_stage]))
            summary["actual_model_mean_mae"] = float(np.mean([m["mae"] for m in model_stage]))
            summary["actual_model_min_match_rate"] = float(np.min([m["match_rate"] for m in model_stage]))
    with open(build_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print(
        "Integration test complete: "
        f"chunks={summary['chunk_count']} allclose={summary['allclose_pass']} "
        f"embedding_match={summary['embedding_match_rate']:.3f}%"
    )


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "build" / "siglip_patch_qkv_encoder_integration"
    emit_and_run_integration_test(out_dir)
