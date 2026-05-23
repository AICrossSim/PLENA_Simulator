import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.mlp.siglip_mlp_test import gelu_with_bf16_intermediates

from transactional_emulator.testbench.siglip.local_asm_templates.layout import (
    compute_hbm_offsets,
    compute_vram_layout,
)
from transactional_emulator.testbench.siglip.local_asm_templates.encoder_layer_blocks import build_encoder_layer_asm

from transactional_emulator.testbench.emulator_runner import compare_emulator_output
from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.siglip.utils.core import (
    ENCODER_HBM_DATA_ORDER,
    json_default,
)
from transactional_emulator.testbench.siglip.utils.vram import (
    pack_seq_to_chunk_major,
)
from transactional_emulator.testbench.siglip.utils.math import (
    gqa_sdpa,
    projection_matmul_k_split_visible,
    quantize_flattened_like_hbm,
)
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    format_chunk_result_line,
    format_harness_summary_line,
    format_siglip_run_config,
    load_siglip_harness_run_config,
    prepare_case_and_run_emulator,
    clear_chunk_dirs,
    summarize_chunk_reports,
    warn_q_chunk_mismatch,
    write_chunk_report,
    write_summary_report,
)
from transactional_emulator.testbench.siglip.utils.siglip_tensors import (
    load_or_prepare_full_siglip_tensors,
)


def emit_and_run_asm_test(build_dir: Path):
    """Run the encoder layer with full Q chunks but a single KV tile debug path.

    Uses d_padded = mlen = 128 (pad h_qkv 72 → 128) so per-head VRAM
    addresses are always vlen-aligned per the SRAM layout contract.
    Q in VRAM: head-major [hq, s_q, d_padded].
    K/V in HBM: head-major [hkv, mlen, d_padded].
    Golden and all downstream ops use hidden_size_padded = hq * d_padded = 2048.
    """
    batch = 1
    blen = 4
    run_cfg = load_siglip_harness_run_config(
        build_dir=build_dir,
        mlen_default=128,
        vlen_default=128,
        q_chunk_default=128,
        max_chunks_default=1,
        inter_dim_default=1024,
    )
    mlen = run_cfg.mlen
    max_q_chunk = run_cfg.max_q_chunk
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    vlen = run_cfg.vlen
    # Keep emulator hardware config in sync with generated assembly.
    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)
    warn_q_chunk_mismatch(max_q_chunk=max_q_chunk, mlen=mlen)
    build_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_cfg.cache_path
    max_chunks = run_cfg.max_chunks
    print(format_siglip_run_config(run_cfg))
    clear_chunk_dirs(build_dir)
    tensors = load_or_prepare_full_siglip_tensors(cache_path=cache_path)
    s_full = int(tensors["s_full"])
    hidden_size = int(tensors["hidden_size"])   # 1152 (real SigLIP)
    hq = int(tensors["hq"])                     # 16
    hkv = int(tensors["hkv"])                   # 16
    h_qkv = int(tensors["h_qkv"])               # 72
    aligned_inter_dim = int(tensors["aligned_inter_dim"])
    x_in_full = tensors["x_in_full"]            # [729, 1152]
    k_full = tensors["k_full"]                  # [1, 729, 16, 72]
    v_full = tensors["v_full"]                  # [1, 729, 16, 72]
    wq_padded = tensors["wq_padded"]           # [hidden_padded, hidden_padded]
    q_bias_padded = tensors["q_bias_padded"]   # [hidden_padded]
    w1_raw = tensors["w1_raw"]                 # [hidden_size, aligned_inter_dim]
    w2_raw = tensors["w2_raw"]                 # [aligned_inter_dim, hidden_size]

    # Pad head_dim to mlen for VRAM alignment contract compliance.
    d_padded = mlen                              # 128
    hidden_size_padded = hq * d_padded           # 2048
    # KV tile size: limit to one MLEN tile so H_PREFETCH_M loads exactly
    # mlen * d_padded = 16384 elements per tile (no multi-tile advancement needed).
    s_kv_tile = mlen                             # 128 KV tokens per tile

    # Attention scale uses original h_qkv (not padded) for correct softmax scaling.
    eps = 1e-2
    scale = 1.0 / math.sqrt(h_qkv)

    # ----- Build K/V HBM tensors: head-major padded [hkv, s_kv_tile, d_padded] -----
    # K_tile[h, t, :h_qkv] = real K; zeros for h_qkv:d_padded.
    k_tile_real = k_full[0, :s_kv_tile, :, :]   # [s_kv_tile, hkv, h_qkv]
    k_padded = torch.zeros(hkv, s_kv_tile, d_padded, dtype=torch.float32)
    k_padded[:, :, :h_qkv] = k_tile_real.permute(1, 0, 2).float()  # [hkv, s_kv_tile, h_qkv]
    k_flat = k_padded.reshape(-1).to(torch.float32)

    v_tile_real = v_full[0, :s_kv_tile, :, :]   # [s_kv_tile, hkv, h_qkv]
    v_padded = torch.zeros(hkv, s_kv_tile, d_padded, dtype=torch.float32)
    v_padded[:, :, :h_qkv] = v_tile_real.permute(1, 0, 2).float()  # [hkv, s_kv_tile, h_qkv]
    v_flat = v_padded.reshape(-1).to(torch.float32)

    # ----- Q projection weights in HBM -----
    wq_hbm = quantize_flattened_like_hbm(wq_padded)
    wq_flat = wq_padded.reshape(-1).to(torch.float32)

    # ----- Pad MLP weights to hidden_size_padded -----
    # w1: [hidden_size, inter_dim] → [hidden_size_padded, inter_dim]
    w1_padded = torch.zeros(hidden_size_padded, aligned_inter_dim, dtype=torch.float32)
    w1_padded[:hidden_size, :] = w1_raw.float()
    w1_hbm = quantize_flattened_like_hbm(w1_padded)
    w1_flat = w1_padded.reshape(-1).to(torch.float32)

    # w2: [inter_dim, hidden_size] → [inter_dim, hidden_size_padded]
    w2_padded = torch.zeros(aligned_inter_dim, hidden_size_padded, dtype=torch.float32)
    w2_padded[:, :hidden_size] = w2_raw.float()
    w2_hbm = quantize_flattened_like_hbm(w2_padded)
    w2_flat = w2_padded.reshape(-1).to(torch.float32)

    wq_elems = int(wq_flat.numel())
    k_elems = int(k_flat.numel())
    v_elems = int(v_flat.numel())
    w1_elems = int(w1_flat.numel())
    w2_elems = int(w2_flat.numel())
    (wq_offset_elems, k_offset_elems, v_offset_elems, w1_offset_elems, w2_offset_elems), _ = compute_hbm_offsets(
        [wq_elems, k_elems, v_elems, w1_elems, w2_elems],
        real_data_ratio=real_data_ratio,
        align_elems=64,
    )
    wq_hbm_offset = int(wq_offset_elems)
    k_hbm_offset = int(k_offset_elems)
    v_hbm_offset = int(v_offset_elems)
    w1_hbm_offset = int(w1_offset_elems)
    w2_hbm_offset = int(w2_offset_elems)

    hbm_mb = int(np.ceil(((wq_elems + k_elems + v_elems + w1_elems + w2_elems) * real_data_ratio) / (1024 * 1024))) + 16

    attn_scale_fp_slot = 1
    ln_eps_fp_slot = 2
    ln_reci_hid_fp_slot = 3
    gelu_one_fp_slot = 4
    gelu_1702_fp_slot = 5
    attn_ninf_fp_slot = 6
    flash_temp_fp_start = 64

    fp_preload = [0.0] * 1024
    fp_preload[gelu_one_fp_slot] = 1.0
    fp_preload[ln_eps_fp_slot] = eps
    fp_preload[ln_reci_hid_fp_slot] = 1.0 / hidden_size_padded  # normalize over padded dims
    fp_preload[gelu_1702_fp_slot] = 1.702
    fp_preload[attn_scale_fp_slot] = float(scale)
    fp_preload[attn_ninf_fp_slot] = float("-inf")

    chunk_idx = 0
    for start in range(0, s_full, max_q_chunk):
        if max_chunks > 0 and chunk_idx >= max_chunks:
            break
        end = min(start + max_q_chunk, s_full)
        s_q_actual = end - start
        s_q_kernel = max_q_chunk

        # ---- Build padded X for VRAM ----
        # X in VRAM: [s_q_kernel, hidden_size_padded], padded from hidden_size=1152.
        x_chunk_actual = x_in_full[start:end].contiguous()  # [s_q_actual, 1152]
        x_chunk_padded = torch.zeros(s_q_kernel, hidden_size_padded, dtype=x_chunk_actual.dtype)
        x_chunk_padded[:s_q_actual, :hidden_size] = x_chunk_actual
        # x_chunk_padded is [128, 2048] — used as input to LN1.

        x_ln1 = F.layer_norm(x_chunk_padded[:s_q_actual].float(), (hidden_size_padded,), eps=eps)

        # Preload X and a per-token Q-bias buffer. Q projection output and
        # head-major Q are produced on-chip.
        q_bias_tile = q_bias_padded.unsqueeze(0).repeat(s_q_kernel, 1)
        x_chunk_packed = pack_seq_to_chunk_major(x_chunk_padded, mlen=mlen)
        q_bias_packed = pack_seq_to_chunk_major(q_bias_tile, mlen=mlen)
        x_vram_flat = x_chunk_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        q_bias_flat = q_bias_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        q_bias_base = int(x_vram_flat.size)
        q_seq_base = int(q_bias_base + q_bias_flat.size)
        q_base = int(q_seq_base + x_vram_flat.size)
        vram_preload = np.concatenate([x_vram_flat, q_bias_flat])

        # ---- Golden computation (padded dims, self-consistent with hardware) ----
        # Use k/v sliced to s_kv_tile tokens (same tile that hardware sees).
        k_tile_golden = k_padded.permute(1, 0, 2).float()  # [s_kv_tile, hkv, d_padded]
        v_tile_golden = v_padded.permute(1, 0, 2).float()  # [s_kv_tile, hkv, d_padded]

        x_in_padded = x_chunk_padded.float()
        q_seq_proj = projection_matmul_k_split_visible(
            x_ln1,
            wq_hbm[:hidden_size_padded, :hidden_size_padded],
            mlen=mlen,
        )
        q_seq_proj = (q_seq_proj + q_bias_padded.unsqueeze(0)).to(torch.bfloat16).float()
        q_ln1 = q_seq_proj.reshape(batch, s_q_actual, hq, d_padded)
        k_tile_gqa = k_tile_golden.reshape(batch, s_kv_tile, hkv, d_padded)
        v_tile_gqa = v_tile_golden.reshape(batch, s_kv_tile, hkv, d_padded)
        attn = gqa_sdpa(q_ln1, k_tile_gqa, v_tile_gqa, scale, hq, hkv).reshape(s_q_actual, hidden_size_padded)
        x_res1 = x_in_padded[:s_q_actual] + attn

        x_ln2 = F.layer_norm(x_res1, (hidden_size_padded,), eps=eps)
        mlp_pre_gelu = projection_matmul_k_split_visible(x_ln2, w1_hbm, mlen=mlen)
        mlp_mid = gelu_with_bf16_intermediates(mlp_pre_gelu)
        mlp_out = projection_matmul_k_split_visible(mlp_mid, w2_hbm, mlen=mlen)
        golden = (x_res1 + mlp_out).reshape(-1)

        x_base = 0

        # Layout: X, Q-bias, Q-seq scratch, then Q head-major.
        # Flash attention derives o_old_base from vector_sram_base=q_base
        layout = compute_vram_layout(
            mlen=mlen,
            blen=blen,
            q_len=s_q_kernel,
            kv_len=s_kv_tile,
            hq=hq,
            hkv=hkv,
            d=d_padded,      # pass d_padded directly
            vector_sram_base=q_base,
        )

        attn_base = layout["o_old_base"]
        residual_base = attn_base + s_q_kernel * hidden_size_padded
        mlp_inter_base = residual_base + s_q_kernel * hidden_size_padded
        mlp_out_base = mlp_inter_base + s_q_kernel * aligned_inter_dim
        scratch_base = mlp_out_base + s_q_kernel * hidden_size_padded

        gen_assembly_code = build_encoder_layer_asm(
            mlen=mlen,
            blen=blen,
            vlen=vlen,
            batch=batch,
            s_q=s_q_kernel,
            s_kv=s_kv_tile,
            hq=hq,
            hkv=hkv,
            h_qkv=d_padded,             # pass d_padded so flash_attn uses aligned head dim
            hidden_size=hidden_size_padded,  # all ops use padded hidden size
            inter_dim=aligned_inter_dim,
            x_base=x_base,
            q_base=q_base,
            attn_base=attn_base,
            residual_base=residual_base,
            mlp_inter_base=mlp_inter_base,
            mlp_out_base=mlp_out_base,
            scratch_base=scratch_base,
            q_seq_base=q_seq_base,
            k_hbm_offset=k_hbm_offset,
            v_hbm_offset=v_hbm_offset,
            wq_hbm_offset=wq_hbm_offset,
            w1_hbm_offset=w1_hbm_offset,
            w2_hbm_offset=w2_hbm_offset,
            ln_eps_fp_slot=ln_eps_fp_slot,
            ln_reci_hid_fp_slot=ln_reci_hid_fp_slot,
            gelu_one_fp_slot=gelu_one_fp_slot,
            gelu_1702_fp_slot=gelu_1702_fp_slot,
            attn_scale_fp_slot=attn_scale_fp_slot,
            attn_ninf_fp_slot=attn_ninf_fp_slot,
            flash_temp_fp_start=flash_temp_fp_start,
            include_final_residual=True,
            include_gelu=True,
            q_bias_base=q_bias_base,
        )

        chunk_build_dir = build_dir / f"chunk_{chunk_idx:02d}_{start:04d}_{end:04d}"
        chunk_build_dir.mkdir(parents=True, exist_ok=True)

        # input_tensor for HBM loading: Q is informational only, not used for HBM.
        input_tensor = {
            "WQ": wq_flat,
            "Q": x_chunk_padded.reshape(-1).to(torch.float32),
            "K": k_flat,
            "V": v_flat,
            "W1": w1_flat,
            "W2": w2_flat,
        }
        golden_result = {
            "input_tensor": {
                "Q": input_tensor["Q"].reshape(-1),
                "WQ": input_tensor["WQ"].reshape(-1),
                "K": input_tensor["K"].reshape(-1),
                "V": input_tensor["V"].reshape(-1),
                "W1": input_tensor["W1"].reshape(-1),
                "W2": input_tensor["W2"].reshape(-1),
            },
            "original_output": golden,
        }

        prepare_case_and_run_emulator(
            case_build_dir=chunk_build_dir,
            input_tensor=input_tensor,
            asm_code=gen_assembly_code,
            golden_result=golden_result,
            fp_preload=fp_preload,
            vram_preload=vram_preload,
            hbm_mb=hbm_mb,
            data_order=ENCODER_HBM_DATA_ORDER,
        )

        comparison_params = {
            "start_row_idx": int(mlp_out_base // mlen),
            "num_rows": int((s_q_actual * hidden_size_padded) // mlen),
            "num_batches": int(s_q_actual),
            "elements_per_batch": int(hidden_size_padded),
            "row_dim": int(mlen),
            "use_slice_mode": False,
            "use_stride_mode": True,
        }
        with open(chunk_build_dir / "comparison_params.json", "w") as f:
            json.dump(comparison_params, f, indent=2)

        results, params = compare_emulator_output(chunk_build_dir)
        print(format_chunk_result_line(chunk_idx=chunk_idx, start=start, end=end, results=results, include_mae=True))
        write_chunk_report(
            chunk_build_dir,
            chunk_idx,
            start,
            end,
            results,
            params,
            json_default=json_default,
        )
        chunk_idx += 1

    summary = summarize_chunk_reports(build_dir)
    write_summary_report(build_dir, summary, json_default=json_default)
    print(
        format_harness_summary_line(
            label="Single-KV-tile SigLIP encoder layer test",
            summary=summary,
            token_count=s_full,
        )
    )


if __name__ == '__main__':
    build_dir = Path(__file__).parent / "build" / "siglip_encoder_layer_single_kv_tile_asm"
    emit_and_run_asm_test(build_dir)
