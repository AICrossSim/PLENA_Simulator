"""
测试 Online Softmax 实现 - 使用 PLENAProgram API（无手写 ISA）

算法（与老 _online_softmax_asm 一致）：
初始化：m_old[row] = -inf, l_old[row] = 0
for row in range(mlen):
    1. m_old_saved = m_old[row]               # 保存旧 max
    2. S[row] *= scale                         # 缩放
    3. row_max = max(S[row])                   # 当前行最大值
    4. m_old[row] = max(m_old[row], row_max)   # 更新 max（m_curr）
    5. m_res[row] = exp(m_old_saved - m_curr)  # 衰减因子
    6. S[row] -= m_curr                        # 减去最大值
    7. P[row] = exp(S[row])                    # 指数化
    8. sum_p = sum(P[row])                     # 求和
    9. l_old[row] = l_old[row] * m_res + sum_p # 更新累加和
最终：P[row] /= l_old[row]                     # 归一化
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json
from plena_program import PLENAProgram
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim


if __name__ == "__main__":
    print("=" * 80)
    print("测试 Online Softmax - 纯 API 实现（无手写 ISA）")
    print("=" * 80)

    # ========================================================================
    # 参数设置
    # ========================================================================
    mlen = 64
    blen = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    scale = 1.0

    torch.manual_seed(42)

    # ========================================================================
    # 创建测试数据
    # ========================================================================
    X = torch.randn(mlen, mlen) * 0.5

    print(f"\n输入数据：")
    print(f"  X: {X.shape}, range [{X.min():.3f}, {X.max():.3f}]")
    print(f"  scale: {scale}")

    # ========================================================================
    # Golden 计算
    # ========================================================================
    print("\n--- Golden Computation ---")
    golden_P = torch.softmax(X.float() * scale, dim=1)
    print(f"  P = softmax(X * scale, dim=1): {golden_P.shape}")
    print(f"  P[0,:4]: {golden_P[0,:4].tolist()}")
    print(f"  P[0,:].sum(): {golden_P[0,:].sum():.6f} (应该≈1.0)")

    # ========================================================================
    # PLENAProgram 实现
    # ========================================================================
    print("\n--- PLENAProgram 实现 ---")
    
    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # 加载输入
    x_input = prog.input("X", shape=(mlen, mlen))
    X_batch = prog.load_batch(x_input, name="X")

    # 分配 VRAM 矩阵
    S = prog.alloc("S", mlen, mlen)

    # 预留 fp_preload 区域（地址 0-2）：0=0.0, 1=scale, 2=-inf
    # 这样 FPVar 分配不会覆盖预加载的常量
    prog._compiler.sub_matrix_manager.fpram_allocator.next_free = 3
    
    # 分配 FPRAM 变量（从地址 3 开始）
    scale_fp = prog.fp_var("scale_fp", size=1)        # scale factor
    m_old = prog.fp_var("m_old", size=mlen)            # 每行的历史 max
    m_res = prog.fp_var("m_res", size=mlen)            # 每行的 exp(m_old - m_new)
    l_old = prog.fp_var("l_old", size=mlen)            # 每行的累加和
    row_max_tmp = prog.fp_var("row_max_tmp", size=1)   # 当前行 max（临时）
    m_old_saved = prog.fp_var("m_old_saved", size=1)   # 保存 m_old（临时）
    sum_p_tmp = prog.fp_var("sum_p_tmp", size=1)       # 当前行 sum（临时）
    inv_l = prog.fp_var("inv_l", size=mlen)            # 1/l（最终归一化）

    print(f"  FPRAM 分配: scale_fp@{scale_fp.address}, m_old@{m_old.address}, "
          f"m_res@{m_res.address}, l_old@{l_old.address}")
    print(f"  临时: row_max_tmp@{row_max_tmp.address}, m_old_saved@{m_old_saved.address}, "
          f"sum_p_tmp@{sum_p_tmp.address}")

    # ========================================================================
    # Step 0: 初始化
    # ========================================================================
    print(f"\n  Step 0: S = X, m_old = -inf, l_old = 0")
    prog.vram_add(S, X_batch)
    prog.fpvar_fill_from_fpram(scale_fp, src_fpram_addr=1)   # scale
    prog.fpvar_fill_from_fpram(m_old, src_fpram_addr=2)      # -inf
    prog.fpvar_fill_from_fpram(l_old, src_fpram_addr=0)      # 0

    # ========================================================================
    # 逐行 Online Softmax（纯 API 调用）
    # ========================================================================
    print(f"\n  逐行 Online Softmax (共 {mlen} 行):")
    
    compiler = prog._compiler  # 用于 element-level FPRAM 操作

    for row in range(mlen):
        if row % 16 == 0:
            print(f"    处理行 {row}/{mlen}...")
        
        # 1. m_old_saved = m_old[row]（保存旧 max）
        compiler.fpvar_copy_asm(m_old.address + row, m_old_saved.address, 1)
        
        # 2. S[row] *= scale
        prog.tile_row_mul_fp_broadcast(S, scale_fp.address, row)
        
        # 3. row_max = max(S[row])
        prog.tile_row_max(row_max_tmp.address, S, row)
        
        # 4. m_old[row] = max(m_old[row], row_max)  -> m_curr
        compiler.fpvar_max_asm(m_old.address + row, row_max_tmp.address, m_old.address + row, 1)
        
        # 5. m_res[row] = exp(m_old_saved - m_curr)
        compiler.fpvar_sub_asm(m_old_saved.address, m_old.address + row, m_old_saved.address, 1)
        compiler.fpvar_exp_asm(m_old_saved.address, m_res.address + row, 1)
        
        # 6. S[row] -= m_curr (= m_old[row])
        prog.tile_row_sub_fp(S, m_old.address + row, row)
        
        # 7. P[row] = exp(S[row])
        prog.tile_row_exp(S, row)
        
        # 8. sum_p = sum(P[row])
        prog.tile_row_sum(sum_p_tmp.address, S, row)
        
        # 9. l_old[row] = l_old[row] * m_res[row] + sum_p
        compiler.fpvar_mul_asm(l_old.address + row, m_res.address + row, l_old.address + row, 1)
        compiler.fpvar_add_asm(l_old.address + row, sum_p_tmp.address, l_old.address + row, 1)

    # ========================================================================
    # 最终归一化：P /= l
    # ========================================================================
    print(f"\n  最终归一化: P /= l")
    prog.fpvar_reci(l_old, inv_l)
    for row in range(mlen):
        prog.tile_row_mul_fp(S, inv_l.address + row, row)

    print(f"\n  完成！")

    # ========================================================================
    # 生成 ISA 代码
    # ========================================================================
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\n生成 {len(lines)} 行 ISA 代码")

    # ========================================================================
    # 创建仿真环境
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {"X": X}
    golden_result = {"original_output": golden_P}

    # FP SRAM 预加载：[0]=0.0, [1]=scale, [2]=-inf
    fp_preload = [0.0, scale, float('-inf')] + [0.0] * 7

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="fpvar_softmax_test",
        data=None,
        specified_data_order=["X"],
        build_path=build_dir
    )

    symbol_table = prog._compiler.symbol_table.table
    s_info = symbol_table[S.name]

    comparison_params = {
        "start_row_idx": s_info.vram_addr // mlen,
        "num_rows": (mlen * mlen) // mlen,
        "num_batches": mlen,
        "elements_per_batch": mlen,
        "row_dim": mlen,
        "use_stride_mode": True
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\n仿真环境已创建: {build_dir}")
    print(f"  S 位置: VRAM row {s_info.vram_addr // mlen}")
    print(f"\n运行仿真:")
    print(f"  cd PLENA_Simulator && just behave_sim fpvar_softmax_test")
