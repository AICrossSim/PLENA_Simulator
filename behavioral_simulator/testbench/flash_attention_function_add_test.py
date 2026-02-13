"""
Flash Attention + Full Matrix Add Test (Function style)

目标：
1. 把现有 Flash Attention 主流程包装成 @prog.function。
2. 提供一个整矩阵相加函数（内部按 mlen x mlen block 调用 vram_block_add_to）。
3. 生成仿真环境与 comparison_params，供环境自动比较。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import json
import torch

from plena_program import PLENAProgram
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    print("=" * 80)
    print("Flash Attention Function + Full Matrix Add Function Test")
    print("=" * 80)

    # ========================================================================
    # 参数设置
    # ========================================================================
    seq_len = 128
    head_dim = 128
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    num_q_blocks = seq_len // mlen
    num_k_blocks = seq_len // mlen
    row_blocks = seq_len // mlen
    col_blocks = head_dim // mlen
    scale = 1.0 / math.sqrt(head_dim)
    norm_eps = 1e-6
    norm_reci_hid = 1.0 / head_dim

    torch.manual_seed(42)

    # ========================================================================
    # 输入与 Golden
    # ========================================================================
    Q = torch.randn(seq_len, head_dim) * 0.5
    K = torch.randn(seq_len, head_dim) * 0.5
    V = torch.randn(seq_len, head_dim) * 0.5

    golden_S = torch.matmul(Q.float(), K.float().T)
    golden_P = torch.softmax(golden_S * scale, dim=-1)
    golden_O = torch.matmul(golden_P, V.float())
    golden_out = golden_O + Q.float()  # residual add
    golden_out = golden_out * torch.rsqrt(golden_out.pow(2).mean(dim=-1, keepdim=True) + norm_eps)

    print(f"Q/K/V shape: {Q.shape}")
    print(f"Golden output shape: {golden_out.shape}")

    # ========================================================================
    # PLENAProgram
    # ========================================================================
    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    q_input = prog.input("Q", shape=(seq_len, head_dim))
    k_input = prog.input("K", shape=(seq_len, head_dim))
    v_input = prog.input("V", shape=(seq_len, head_dim))

    @prog.function
    def flash_attention_fn(q_in, k_in, v_in):
        Q_batch = prog.load_batch(q_in, name="Q")
        Q_sub = prog.register_vram_sub_matrix(Q_batch, name="Q_sub")
        K_sub = prog.register_sub_matrix(k_in, name="K_sub")
        V_sub = prog.register_sub_matrix(v_in, name="V_sub")

        S_block = prog.alloc("S_block", mlen, mlen)
        PV = prog.alloc("PV", mlen, head_dim)
        O = prog.alloc("O", seq_len, head_dim)

        for q_idx in range(num_q_blocks):
            prog.init_online_softmax(q_idx, O)
            for k_idx in range(num_k_blocks):
                prog.reset_mram()
                K_sub.load_row(k_idx)

                prog.vram_sub_projection_T_to(
                    Q_sub.row(q_idx),
                    K_sub.row(k_idx),
                    S_block,
                    target_row_idx=0,
                    target_col_idx=0,
                )

                prog.online_softmax_block(S_block, scale)
                prog.compute_pv(S_block, V_sub, k_idx, PV)
                prog.scale_o_row(O, q_idx)
                prog.vram_add(O, PV, dst_row_offset=q_idx * mlen)

            prog.final_scale_o(q_idx, O)

        return O, Q_batch

    @prog.function
    def full_matrix_add_fn(a, b):
        out = prog.alloc("add_out", seq_len, head_dim)
        for r in range(row_blocks):
            for c in range(col_blocks):
                prog.vram_block_add_to(
                    src1=a,
                    src1_row_idx=r,
                    src1_col_idx=c,
                    src2=b,
                    src2_row_idx=r,
                    src2_col_idx=c,
                    target=out,
                    target_row_idx=r,
                    target_col_idx=c,
                )
        return out

    O_attn, Q_batch = flash_attention_fn(q_input, k_input, v_input)
    O_final = full_matrix_add_fn(O_attn, Q_batch)
    O_final = prog.norm(
        O_final,
        mode="rms",
        eps_offset=3,
        reci_hid_offset=4,
    )

    prog.result(O_final)
    gen_code = prog.compile()
    print(f"Generated ISA lines: {len(gen_code.splitlines())}")

    # ========================================================================
    # 仿真环境
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "Q": Q.reshape(1, -1),
        "K": K.reshape(1, -1),
        "V": V.reshape(1, -1),
    }
    golden_result = {"original_output": golden_out}

    # FP SRAM preload: [0]=0.0, [1]=scale, [2]=-inf, [3]=eps, [4]=1/head_dim
    fp_preload = [0.0, scale, float("-inf"), norm_eps, norm_reci_hid] + [0.0] * 5
    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_function_add_test",
        data=None,
        specified_data_order=["Q", "K", "V"],
        build_path=build_dir,
    )

    symbol_table = prog._compiler.symbol_table.table
    out_info = symbol_table[O_final.name]

    comparison_params = {
        "start_row_idx": out_info.vram_addr // mlen,
        "num_rows": (seq_len * head_dim) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": head_dim,
        "row_dim": mlen,
        "use_stride_mode": True,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"Build dir: {build_dir}")
    print(f"Result location: row {out_info.vram_addr // mlen}, {(seq_len * head_dim) // mlen} rows")
    print(f"Comparison params: {comparison_params}")
