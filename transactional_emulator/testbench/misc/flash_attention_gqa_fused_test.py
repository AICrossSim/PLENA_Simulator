"""
End-to-end GQA Flash Attention test for kev/aten path (fused codegen).

Reuses the `flash_attn_asm` template directly from the
`compiler.asm_templates.flashattn` package.  This shows
that the ATen backend *can* emit the same fused GQA ASM at the same cost as
main's prefill test — no regression in codegen quality for GQA.

Dims match main's `flashattn_prefill_test.py`:
  batch=1, s_q=s_kv=64, hq=4, hkv=1, h_qkv=16 (hq/hkv=blen=4, hq*h_qkv=mlen)

Golden reference uses SDPA (the original test depended on an outdated
aria_lm_ops.flash_attn2_gemv API that no longer exists on this branch).
"""

import argparse
import math
import json
import toml
from pathlib import Path


import torch
import torch.nn.functional as F

from compiler.asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm
from compiler.asm_templates.flashattn import flash_attn_asm
from compiler.sim_env_utils import create_mem_for_sim
from plena_utils import load_precision_from_toml
from verification.create_sim_env import create_sim_env
from transactional_emulator.testbench.emulator_runner import run_and_assert
from runtime_paths import settings_path


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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kv-heads", type=int, default=1,
        help="Number of KV heads. Each holds BROADCAST_AMOUNT query heads, so "
             "the packed Q/O width is kv_heads*MLEN. Defaults to 1.",
    )
    parser.add_argument(
        "--kv-head-reuse", action="store_true",
        help="Hoist the KV-head loop inside the key-tile loop so one resident "
             "K/V tile serves every KV head instead of being re-read per head.",
    )
    args = parser.parse_args()

    batch_size = 1
    s_q = 64
    s_kv = 64
    h_qkv = 16
    mlen = 64
    vlen = 64
    blen = 4
    # One KV group covers the array's broadcast lanes, so the query-head count
    # follows the KV-head count. Each group's Q/O occupies its own MLEN-wide
    # block (the packed group layout), so the packed width is kv_heads*MLEN.
    broadcast_amount = mlen // h_qkv
    num_kv_heads = args.kv_heads
    num_q_heads = broadcast_amount * num_kv_heads
    if num_kv_heads < 1 or num_kv_heads > broadcast_amount:
        raise SystemExit(
            f"--kv-heads must be in 1..{broadcast_amount}: a KV cache row is "
            f"MLEN={mlen} wide and holds kv_heads*{h_qkv} real columns."
        )
    # The residual-stream width one KV group contributes.
    hidden_size = h_qkv * broadcast_amount  # 64
    packed_group_layout = num_kv_heads > 1
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    qk_scale = 1.0 / math.sqrt(h_qkv)
    # Slots 0-2 hold zero, the QK scale and -inf; the softmax state starts above
    # them and is bounded by the scalar FP SRAM the emulator is configured with.
    FP_SRAM_START = 3
    fp_sram_depth = int(
        toml.load(settings_path())["TRANSACTIONAL"]["CONFIG"]["FP_SRAM_DEPTH"]["value"]
    )

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

    # Packed group layout: query head h belongs to KV group h//ratio and occupies
    # lane h%ratio of that group's MLEN-wide row, and each group's rows form
    # their own contiguous (s_q, MLEN) block. Grouping the head axis and moving
    # it ahead of the row axis produces exactly that order. With one KV group
    # the permute is the identity, so the single-group layout is unchanged.
    def to_group_layout(tensor):
        return (
            tensor.reshape(batch_size, s_q, num_kv_heads, broadcast_amount, h_qkv)
            .permute(0, 2, 1, 3, 4)
            .reshape(batch_size, num_kv_heads * s_q, mlen)
        )

    q_packed = to_group_layout(q)
    golden_packed = to_group_layout(golden)

    input_tensor = {
        "Q": q_packed.reshape(1, -1),
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
        fp_sram_start_address=FP_SRAM_START,
        # Without a depth the lowering sweeps every query row at once, whose
        # softmax state is 3 * s_q * ratio slots and overruns the scalar FP
        # SRAM. Passing the hardware depth tiles the rows to fit it.
        fp_sram_depth=fp_sram_depth - FP_SRAM_START,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2,
        # Each KV group's Q and O live in their own (s_q, MLEN) block, so the
        # group stride is a whole block. Left at the defaults for one group so
        # the single-group assembly is byte-identical to the established one.
        packed_group_layout=packed_group_layout,
        q_group_stride=s_q * mlen if packed_group_layout else None,
        o_group_stride=s_q * mlen if packed_group_layout else None,
        kv_head_reuse=args.kv_head_reuse,
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
    # O carries one MLEN-wide row per query row per KV group.
    result_rows = batch_size * num_kv_heads * s_q
    result_start_row = result_vram_offset // vlen
    num_result_rows = (result_rows * mlen) // vlen

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": golden_packed.reshape(num_kv_heads * s_q, mlen),
    }

    fp_preload = [0.0, qk_scale, float("-inf")] + [0.0] * 45
    build_dir = Path(__file__).parent / "build" / "flash_attention_gqa_fused"
    build_dir.mkdir(parents=True, exist_ok=True)

    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=str(build_dir))
    create_mem_for_sim(
        precision_settings=load_precision_from_toml(
            settings_path(), mode="TRANSACTIONAL"
        ),
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
        "num_batches": num_kv_heads * s_q,
        "elements_per_batch": mlen,
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
