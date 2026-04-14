"""
End-to-end GQA Flash Attention test for kev/aten path (fused codegen).

Reuses main branch's `flash_attn_asm` template directly (already available on
the kev/aten branch under `compiler/asm_templates/flashattn/`).  This shows
that the ATen backend *can* emit the same fused GQA ASM at the same cost as
main's prefill test — no regression in codegen quality for GQA.

Dims match main's `flashattn_prefill_test.py`:
  batch=1, s_q=s_kv=64, hq=4, hkv=1, h_qkv=16 (hq/hkv=blen=4, hq*h_qkv=mlen)

Golden reference uses SDPA (the original test depended on an outdated
aria_lm_ops.flash_attn2_gemv API that no longer exists on this branch).
"""

import math
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F

from compiler.asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm
from compiler.asm_templates.flashattn import flash_attn_asm
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from emulator_runner import run_and_assert


def gqa_sdpa(q, k, v, qk_scale, num_q_heads, num_kv_heads):
    q_t = q.transpose(1, 2)  # (b, hq, s_q, d)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    ratio = num_q_heads // num_kv_heads
    k_t = k_t.repeat_interleave(ratio, dim=1)
    v_t = v_t.repeat_interleave(ratio, dim=1)
    o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=qk_scale)
    return o.transpose(1, 2)  # (b, s_q, hq, d)


if __name__ == "__main__":
    print("=" * 80)
    print("GQA Flash Attention — kev/aten test harness using fused flash_attn_asm")
    print("=" * 80)

    batch_size = 1
    s_q = 64
    s_kv = 64
    num_q_heads = 4
    num_kv_heads = 1
    h_qkv = 16
    hidden_size = h_qkv * num_q_heads  # 64
    mlen = 64
    vlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    qk_scale = 1.0 / math.sqrt(h_qkv)

    torch.manual_seed(42)
    q = torch.randn(batch_size, s_q, num_q_heads, h_qkv, dtype=torch.bfloat16) * 0.5
    k = torch.randn(batch_size, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16) * 0.5
    v = torch.randn(batch_size, s_kv, num_kv_heads, h_qkv, dtype=torch.bfloat16) * 0.5

    # Reshape K and V to match main's expected padded layout (hkv -> mlen/h_qkv)
    if num_kv_heads < (mlen // h_qkv):
        k_padded = torch.zeros(batch_size, s_kv, mlen // h_qkv, h_qkv, dtype=k.dtype)
        v_padded = torch.zeros(batch_size, s_kv, mlen // h_qkv, h_qkv, dtype=v.dtype)
        k_padded[:, :, :num_kv_heads, :] = k
        v_padded[:, :, :num_kv_heads, :] = v
    else:
        k_padded = k
        v_padded = v

    # Golden
    golden = gqa_sdpa(q.float(), k.float(), v.float(), qk_scale, num_q_heads, num_kv_heads)

    input_tensor = {
        "Q": q.reshape(1, -1),
        "K": k_padded.reshape(1, -1),
        "V": v_padded.reshape(1, -1),
    }

    gen_assembly_code = "; GQA Flash Attention — fused codegen (kev/aten harness)\n"
    gen_assembly_code += (
        f"; Config: batch={batch_size}, s_q={s_q}, s_kv={s_kv}, hq={num_q_heads}, hkv={num_kv_heads}, d={h_qkv}\n"
    )

    q_hbm_size = int(s_q * num_q_heads * h_qkv * batch_size * real_data_ratio)
    k_hbm_size = int(s_kv * (mlen // h_qkv) * h_qkv * batch_size * real_data_ratio)
    k_hbm_offset = q_hbm_size
    v_hbm_offset = q_hbm_size + k_hbm_size

    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[k_hbm_offset, v_hbm_offset],
    )
    gen_assembly_code += preload_act_asm(
        vlen=mlen,
        preload_len=4,
        batch=batch_size,
        hidden_size=h_qkv * num_q_heads * s_q,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
    gen_assembly_code += flash_attn_asm(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        batch=batch_size,
        hq=num_q_heads,
        hkv=num_kv_heads,
        d=h_qkv,
        q_len=s_q,
        kv_len=s_kv,
        alive_registers_int=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        alive_registers_fp=[1, 2, 3, 4, 5, 6, 7],
        vector_sram_base_address=0,
        fp_sram_start_address=3,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2,
    )

    lines = gen_assembly_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA")

    # Mirror main's VSRAM layout math
    o_old_base_address = (
        s_q * num_q_heads * h_qkv  # Q
        + mlen * mlen * num_q_heads // num_kv_heads  # S
        + mlen * mlen * num_q_heads // num_kv_heads  # PV
    )
    result_vram_offset = o_old_base_address
    effective_batch = batch_size * s_q
    result_start_row = result_vram_offset // vlen
    num_result_rows = (effective_batch * hidden_size) // vlen

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": golden.reshape(s_q, num_q_heads * h_qkv),
    }

    fp_preload = [0.0, qk_scale, float("-inf")] + [0.0] * 45
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=str(build_dir))
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_gqa_fused",
        data=None,
        specified_data_order=["Q", "K", "V"],
        build_path=build_dir,
    )

    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": s_q,
        "elements_per_batch": hidden_size,
        "row_dim": vlen,
        "use_stride_mode": False,
        "use_slice_mode": True,
        "slice_per_row": h_qkv,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_assembly_code)

    print(f"\nOutput at VRAM row {result_start_row}, {num_result_rows} rows")
    run_and_assert(build_dir, "flash_attention_gqa_fused", mlen=mlen, blen=blen)
