import math
from pathlib import Path

import numpy as np
import torch

from transactional_emulator.testbench.siglip.mlp.siglip_mlp_test import gelu_with_bf16_intermediates
from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    mha_sdpa,
    projection_matmul_k_split_visible,
    quantize_flattened_like_hbm,
)
from transactional_emulator.testbench.siglip.utils.siglip_tensors import (
    build_runtime_hidden_positions,
    _scatter_hidden_square_matrix,
    _scatter_hidden_to_inter_matrix,
    _scatter_inter_to_hidden_matrix,
    _scatter_seq_hidden_tensor,
)
from transactional_emulator.testbench.siglip.utils.vram import pack_seq_to_chunk_major

from transactional_emulator.testbench.siglip.local_asm_templates.layout import (
    compute_hbm_offsets,
    compute_vram_layout,
)
from transactional_emulator.testbench.siglip.local_asm_templates.encoder_layer_blocks import build_encoder_layer_asm

from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    prepare_case_artifacts,
    write_comparison_params,
)


def emit_and_run_asm_test(build_dir: Path):
    """Build and run a standalone SigLIP encoder layer test against emulator output."""
    batch = 1
    s_q = 64
    s_kv = 64
    hq = 4
    hkv = hq
    h_qkv = 64

    mlen = 64
    blen = 4
    vlen = 64
    d_padded = mlen
    hidden_size_visible = hq * h_qkv
    hidden_size = hq * d_padded
    inter_dim = 512
    real_data_ratio = MXFP_REAL_DATA_RATIO

    torch.manual_seed(0)
    # keep magnitudes small to avoid FP packing issues in the test harness
    q = (torch.randn(batch, s_q, hq, h_qkv) * 0.1).contiguous()
    k = (torch.randn(batch, s_kv, hkv, h_qkv) * 0.1).contiguous()
    v = (torch.randn(batch, s_kv, hkv, h_qkv) * 0.1).contiguous()
    o_vis = torch.eye(hidden_size_visible, dtype=torch.float32).contiguous()
    w1_vis = (torch.randn(hidden_size_visible, inter_dim) * 0.05).contiguous()
    w2_vis = (torch.randn(inter_dim, hidden_size_visible) * 0.05).contiguous()

    runtime_hidden_positions = build_runtime_hidden_positions(hidden_size_visible, hq, d_padded)

    # Build KV in padded head-major layout for HBM: [hkv, s_kv, d_padded].
    # Real values occupy [:h_qkv], and [h_qkv:d_padded] are explicit zero gaps.
    k_head_major = torch.zeros(hkv, s_kv, d_padded, dtype=torch.float32)
    v_head_major = torch.zeros(hkv, s_kv, d_padded, dtype=torch.float32)
    k_head_major[:, :, :h_qkv] = k[0].permute(1, 0, 2).contiguous().float()
    v_head_major[:, :, :h_qkv] = v[0].permute(1, 0, 2).contiguous().float()

    attn_scale_fp_slot = 1
    attn_ninf_fp_slot = 6

    # Golden path: LN1 -> attention -> residual1 -> LN2 -> MLP -> residual2
    eps = 1e-6
    scale = 1.0 / math.sqrt(h_qkv)
    # Round to BF16 then back to float32 to match the hardware's VRAM data type.
    x_in_vis = q.reshape(s_q, hidden_size_visible)
    x_in = _scatter_seq_hidden_tensor(x_in_vis, hidden_size, runtime_hidden_positions).to(torch.bfloat16).float()

    wq_vis = torch.eye(hidden_size_visible, dtype=torch.float32).contiguous()
    wq = _scatter_hidden_square_matrix(wq_vis, hidden_size, runtime_hidden_positions)
    o = _scatter_hidden_square_matrix(o_vis, hidden_size, runtime_hidden_positions)
    w1 = _scatter_hidden_to_inter_matrix(w1_vis, hidden_size, runtime_hidden_positions)
    w2 = _scatter_inter_to_hidden_matrix(w2_vis, hidden_size, runtime_hidden_positions)

    k_hbm = quantize_flattened_like_hbm(k_head_major.float())
    v_hbm = quantize_flattened_like_hbm(v_head_major.float())
    k_mxfp = k_hbm.permute(1, 0, 2).unsqueeze(0).contiguous()
    v_mxfp = v_hbm.permute(1, 0, 2).unsqueeze(0).contiguous()
    o_hbm = quantize_flattened_like_hbm(o.float())
    wq_hbm = quantize_flattened_like_hbm(wq.float())
    w1_hbm = quantize_flattened_like_hbm(w1.float())
    w2_hbm = quantize_flattened_like_hbm(w2.float())

    x_ln1 = torch.nn.functional.layer_norm(x_in, (hidden_size,), eps=eps)
    q_ln1 = x_ln1.reshape(batch, s_q, hq, d_padded)
    attn = mha_sdpa(q_ln1, k_mxfp, v_mxfp, scale, hq, hkv).reshape(s_q, hidden_size)
    attn_out = projection_matmul_k_split_visible(attn, o_hbm, mlen=mlen)
    x_res1 = x_in + attn_out

    x_ln2 = torch.nn.functional.layer_norm(x_res1, (hidden_size,), eps=eps)
    mlp_mid = projection_matmul_k_split_visible(x_ln2, w1_hbm, mlen=mlen)
    mlp_mid = gelu_with_bf16_intermediates(mlp_mid)
    mlp_out = projection_matmul_k_split_visible(mlp_mid, w2_hbm, mlen=mlen)
    golden = (x_res1 + mlp_out).reshape(-1)

    # Prepare HBM/VRAM tensors.
    # The encoder pipeline now expects X at x_base in chunk-major layout.
    # X is preloaded in chunk-major layout expected by the encoder path.
    x_chunk_packed = pack_seq_to_chunk_major(x_in, mlen=mlen)
    x_vram_flat = x_chunk_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    wq_flat = wq_hbm.reshape(-1).to(torch.float32)
    k_flat = k_head_major.reshape(-1).to(torch.float32)
    v_flat = v_head_major.reshape(-1).to(torch.float32)
    o_flat = o_hbm.reshape(-1).to(torch.float32)
    w1_flat = w1_hbm.reshape(-1).to(torch.float32)
    w2_flat = w2_hbm.reshape(-1).to(torch.float32)
    q_base = int(x_vram_flat.size)

    # Compute VRAM layout and HBM offsets for K and V
    layout = compute_vram_layout(mlen=mlen, blen=blen, q_len=s_q, hq=hq, hkv=hkv, d=d_padded, vector_sram_base=q_base)

    # HBM sizes in elements (WQ -> K -> V -> WO -> W1 -> W2)
    wq_elems = int(wq_flat.numel())
    k_elems = int(k_flat.numel())
    v_elems = int(v_flat.numel())
    o_elems = int(o_flat.numel())
    w1_elems = int(w1_flat.numel())
    w2_elems = int(w2_flat.numel())
    (wq_offset_elems, k_offset_elems, v_offset_elems, o_offset_elems, w1_offset_elems, w2_offset_elems), _ = compute_hbm_offsets(
        [wq_elems, k_elems, v_elems, o_elems, w1_elems, w2_elems], real_data_ratio=real_data_ratio, align_elems=64
    )

    # convert element offsets to element counts (HBM offsets in this codebase are element counts)
    wq_hbm_offset = int(wq_offset_elems)
    k_hbm_offset = int(k_offset_elems)
    v_hbm_offset = int(v_offset_elems)
    o_hbm_offset = int(o_offset_elems)
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
        h_qkv=d_padded,
        hidden_size=hidden_size,
        inter_dim=inter_dim,
        x_base=x_base,
        attn_base=attn_base,
        residual_base=residual_base,
        mlp_inter_base=mlp_inter_base,
        mlp_out_base=mlp_out_base,
        scratch_base=scratch_base,
        q_base=q_base,
        wq_hbm_offset=wq_hbm_offset,
        k_hbm_offset=k_hbm_offset,
        v_hbm_offset=v_hbm_offset,
        out_hbm_offset=o_hbm_offset,
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
        "WQ": wq_flat,
        "Q": x_in.reshape(-1).to(torch.float32),
        "K": k_flat,
        "V": v_flat,
        "WO": o_flat,
        "W1": w1_flat,
        "W2": w2_flat,
    }
    golden_result = {
        "input_tensor": {
            "WQ": input_tensor["WQ"].reshape(-1),
            "Q": input_tensor["Q"].reshape(-1),
            "K": input_tensor["K"].reshape(-1),
            "V": input_tensor["V"].reshape(-1),
            "WO": input_tensor["WO"].reshape(-1),
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
        vram_preload=x_vram_flat,
        hbm_mb=256,
        # X activations are preloaded to VRAM for this standalone harness,
        # so HBM data starts from WQ.
        data_order=["WQ", "K", "V", "WO", "W1", "W2"],
    )

    start_row_idx = mlp_out_base // mlen
    num_rows = (s_q * hidden_size) // mlen
    write_comparison_params(
        build_dir,
        start_row_idx=int(start_row_idx),
        num_rows=int(num_rows),
        num_batches=int(s_q),
        elements_per_batch=int(hidden_size),
        use_slice_mode=False,
        use_stride_mode=False,
        extra_params={
            "row_dim": int(mlen),
            "use_chunk_major_mode": True,
            "seq_len": int(s_q),
            "hidden_dim": int(hidden_size),
            "mlen": int(mlen),
            "chunk_major_valid_seq_len": int(s_q),
            "visible_lane_positions": [int(i) for i in runtime_hidden_positions[:hidden_size_visible]],
        },
    )

    # Keep a larger runtime HBM envelope than preload size for safety: some
    # emulator codepaths perform strided accesses beyond the serialized prefix.
    hbm_file = Path(build_dir) / "hbm_for_behave_sim.bin"
    if hbm_file.exists():
        # Allocate extra HBM space for the emulator runtime to avoid
        # out-of-bounds reads during strided transfers. Use 4x the
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
