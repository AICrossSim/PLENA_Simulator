import math
import json
from pathlib import Path

import numpy as np
import torch

from transactional_emulator.testbench.siglip.mlp.siglip_mlp_test import quantize_to_mxfp, gelu_with_bf16_intermediates
from transactional_emulator.testbench.siglip.utils.math import gqa_sdpa

from transactional_emulator.testbench.siglip.local_asm_templates.layout import (
    compute_hbm_offsets,
    compute_vram_layout,
)
from transactional_emulator.testbench.siglip.local_asm_templates.encoder_layer_blocks import build_encoder_layer_asm

from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.harness_utils import prepare_case_artifacts


def emit_and_run_asm_test(build_dir: Path):
    """Build and run a standalone SigLIP encoder layer test against emulator output."""
    batch = 1
    s_q = 64
    s_kv = 64
    hq = 4
    hkv = 1
    h_qkv = 16
    hidden_size = hq * h_qkv  # 64
    inter_dim = 128

    mlen = 64
    blen = 4
    vlen = 64
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(0)
    # keep magnitudes small to avoid FP packing issues in the test harness
    q = (torch.randn(batch, s_q, hq, h_qkv) * 0.1).contiguous()
    k = (torch.randn(batch, s_kv, hkv, h_qkv) * 0.1).contiguous()
    v = (torch.randn(batch, s_kv, hkv, h_qkv) * 0.1).contiguous()
    w1 = (torch.randn(hidden_size, inter_dim) * 0.05).contiguous()
    w2 = (torch.randn(inter_dim, hidden_size) * 0.05).contiguous()

    # Pad KV to mlen-wide for template compatibility
    k_padded = torch.zeros(batch, s_kv, mlen // h_qkv, h_qkv)
    v_padded = torch.zeros(batch, s_kv, mlen // h_qkv, h_qkv)
    k_padded[:, :, :hkv, :] = k
    v_padded[:, :, :hkv, :] = v

    attn_scale_fp_slot = 1
    attn_ninf_fp_slot = 6

    # Golden path: LN1 -> attention -> residual1 -> LN2 -> MLP -> residual2
    eps = 1e-2
    scale = 1.0 / math.sqrt(h_qkv)
    # Round to BF16 then back to float32 to match the hardware's VRAM data type.
    x_in = q.reshape(s_q, hidden_size).to(torch.bfloat16).float()

    k_mxfp = quantize_to_mxfp(k.float())
    v_mxfp = quantize_to_mxfp(v.float())
    k_hbm = quantize_to_mxfp(k_padded.reshape(s_q, hidden_size).float())
    v_hbm = quantize_to_mxfp(v_padded.reshape(s_q, hidden_size).float())
    w1_mxfp = quantize_to_mxfp(w1.float())
    w2_mxfp = quantize_to_mxfp(w2.float())

    x_ln1 = torch.nn.functional.layer_norm(x_in, (hidden_size,), eps=eps)
    q_ln1 = x_ln1.reshape(batch, s_q, hq, h_qkv)
    attn = gqa_sdpa(q_ln1, k_mxfp, v_mxfp, scale, hq, hkv).reshape(s_q, hidden_size)
    x_res1 = x_in + attn

    x_ln2 = torch.nn.functional.layer_norm(x_res1, (hidden_size,), eps=eps)
    mlp_mid = torch.matmul(x_ln2, w1_mxfp)
    mlp_mid = gelu_with_bf16_intermediates(mlp_mid)
    mlp_out = torch.matmul(mlp_mid.float(), w2_mxfp)
    golden = (x_res1 + mlp_out).reshape(-1)

    # Prepare HBM/VRAM tensors.
    # Write Q as raw BF16 uint16 bit-patterns so create_sim_env writes bytes
    # verbatim. This matches BF16 Vector SRAM interpretation in the emulator.
    q_vram_flat = q.reshape(-1).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    k_flat = k_hbm.reshape(-1).to(torch.float32)
    v_flat = v_hbm.reshape(-1).to(torch.float32)
    w1_flat = w1_mxfp.reshape(-1).to(torch.float32)
    w2_flat = w2_mxfp.reshape(-1).to(torch.float32)

    # Compute VRAM layout and HBM offsets for K and V
    layout = compute_vram_layout(mlen=mlen, blen=blen, q_len=s_q, kv_len=s_kv, hq=hq, hkv=hkv, d=h_qkv, vector_sram_base=0)

    # HBM sizes in elements (K -> V -> W1 -> W2)
    k_elems = int(k_flat.numel())
    v_elems = int(v_flat.numel())
    w1_elems = int(w1_flat.numel())
    w2_elems = int(w2_flat.numel())
    (k_offset_elems, v_offset_elems, w1_offset_elems, w2_offset_elems), _ = compute_hbm_offsets(
        [k_elems, v_elems, w1_elems, w2_elems], real_data_ratio=real_data_ratio, align_elems=64
    )

    # convert element offsets to element counts (HBM offsets in this codebase are element counts)
    k_hbm_offset = int(k_offset_elems)
    v_hbm_offset = int(v_offset_elems)
    w1_hbm_offset = int(w1_offset_elems)
    w2_hbm_offset = int(w2_offset_elems)

    # VRAM layout for full encoder block
    x_base = 0
    attn_base = layout["o_old_base"]
    residual_base = attn_base + s_q * hidden_size
    mlp_inter_base = residual_base + s_q * hidden_size
    mlp_out_base = mlp_inter_base + s_q * inter_dim
    scratch_base = mlp_out_base + s_q * hidden_size

    # Keep constants in stable low slots and move flash-attn temp region up.
    ln_eps_fp_slot = 2
    ln_reci_hid_fp_slot = 3
    gelu_one_fp_slot = 4
    gelu_1702_fp_slot = 5
    flash_temp_fp_start = 64

    gen_assembly_code = build_encoder_layer_asm(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        batch=batch,
        s_q=s_q,
        s_kv=s_kv,
        hq=hq,
        hkv=hkv,
        h_qkv=h_qkv,
        hidden_size=hidden_size,
        inter_dim=inter_dim,
        x_base=x_base,
        attn_base=attn_base,
        residual_base=residual_base,
        mlp_inter_base=mlp_inter_base,
        mlp_out_base=mlp_out_base,
        scratch_base=scratch_base,
        k_hbm_offset=k_hbm_offset,
        v_hbm_offset=v_hbm_offset,
        w1_hbm_offset=w1_hbm_offset,
        w2_hbm_offset=w2_hbm_offset,
        ln_eps_fp_slot=ln_eps_fp_slot,
        ln_reci_hid_fp_slot=ln_reci_hid_fp_slot,
        gelu_one_fp_slot=gelu_one_fp_slot,
        gelu_1702_fp_slot=gelu_1702_fp_slot,
        attn_scale_fp_slot=attn_scale_fp_slot,
        attn_ninf_fp_slot=attn_ninf_fp_slot,
        flash_temp_fp_start=flash_temp_fp_start,
    )

    build_dir.mkdir(parents=True, exist_ok=True)
    gen_file = build_dir / "generated_asm_code.asm"
    with open(gen_file, "w") as f:
        f.write(gen_assembly_code)

    # Keep HBM tensors in float32 for stable serialization in the test harness.
    input_tensor = {
        "Q": q.reshape(-1).to(torch.float32),
        "K": k_flat,
        "V": v_flat,
        "W1": w1_flat,
        "W2": w2_flat,
    }
    golden_result = {
        "input_tensor": {
            "Q": input_tensor["Q"].reshape(-1),
            "K": input_tensor["K"].reshape(-1),
            "V": input_tensor["V"].reshape(-1),
            "W1": input_tensor["W1"].reshape(-1),
            "W2": input_tensor["W2"].reshape(-1),
        },
        "original_output": golden,
    }

    # Dedicated FP slots for constants used by flash-attn, layernorm, and GELU.
    fp_preload = [0.0] * 1024
    fp_preload[gelu_one_fp_slot] = 1.0
    fp_preload[ln_eps_fp_slot] = eps
    fp_preload[ln_reci_hid_fp_slot] = 1.0 / hidden_size
    fp_preload[gelu_1702_fp_slot] = 1.702
    fp_preload[attn_scale_fp_slot] = float(scale)
    fp_preload[attn_ninf_fp_slot] = float("-inf")

    prepare_case_artifacts(
        case_build_dir=build_dir,
        input_tensor=input_tensor,
        asm_code=gen_assembly_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=q_vram_flat,
        hbm_mb=256,
        # Q is preloaded to VRAM and should not occupy HBM base offset 0.
        data_order=["K", "V", "W1", "W2"],
    )

    start_row_idx = mlp_out_base // mlen
    num_rows = (s_q * hidden_size) // mlen
    comparison_params = {
        "start_row_idx": int(start_row_idx),
        "num_rows": int(num_rows),
        "num_batches": s_q,
        "elements_per_batch": hidden_size,
        "use_slice_mode": False,
        "use_stride_mode": True,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    # Expand runtime HBM allocation to avoid out-of-bounds during strided accesses.
    hbm_file = Path(build_dir) / "hbm_for_behave_sim.bin"
    if hbm_file.exists():
        # Allocate extra HBM space for the emulator runtime to avoid
        # out-of-bounds reads during strided transfers. Use 2x the
        # preload size (rounded to 64 bytes) — matches heuristics used
        # elsewhere in the test-suite.
        preload_bytes = hbm_file.stat().st_size
        # Allocate a larger HBM region (4x) to ensure emulator's strided
        # reads don't exceed the mapped region during early iteration.
        hbm_size_bytes = (((4 * preload_bytes) + 63) // 64) * 64
        (build_dir / "hbm_size.txt").write_text(str(int(hbm_size_bytes)))

    run_and_assert(build_dir, "siglip_encoder_layer_asm", mlen=mlen, blen=blen)


if __name__ == '__main__':
    build_dir = Path(__file__).parent / "build" / "siglip_encoder_layer_asm"
    emit_and_run_asm_test(build_dir)
