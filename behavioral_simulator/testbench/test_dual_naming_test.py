"""
测试双层命名机制：display_name vs internal_name（三层 linear）

使用 PLENAProgram API 定义一个 linear 函数，调用三次，
验证函数内的局部变量名不会冲突（通过 symbol table 和 ISA 生成），
并验证 free_tensor 和 store 可以正确工作。

测试场景：
- 全局输入：X, W1, W2, W3
- Y1 = linear(X, W1)
- store Y1 to HBM, free Y1
- Y2 = linear(Y1, W2)
- store Y2 to HBM, free Y2
- Y3 = linear(Y2, W3)
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
    print("测试三层 linear：Y3 = linear(linear(linear(X, W1), W2), W3)")
    print("=" * 80)

    # ========================================================================
    # 参数设置
    # ========================================================================
    batch = 128
    hidden_size = 128
    out_features = 128
    mlen = 64
    blen = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)

    num_col_blocks = out_features // mlen

    torch.manual_seed(42)

    # ========================================================================
    # 创建测试数据
    # ========================================================================
    X = torch.randn(batch, hidden_size) * 0.1
    W1 = torch.randn(hidden_size, hidden_size) * 0.1
    W2 = torch.randn(hidden_size, hidden_size) * 0.1
    W3 = torch.randn(hidden_size, out_features) * 0.1

    print(f"\n输入数据：")
    print(f"  X:  {X.shape}")
    print(f"  W1: {W1.shape}")
    print(f"  W2: {W2.shape}")
    print(f"  W3: {W3.shape}")

    golden_Y1 = torch.matmul(X.float(), W1.float())
    golden_Y2 = torch.matmul(golden_Y1.float(), W2.float())
    golden_Y3 = torch.matmul(golden_Y2.float(), W3.float())

    print(f"\n期望输出：")
    print(f"  Y1 = X @ W1:   {golden_Y1.shape}")
    print(f"  Y2 = Y1 @ W2:  {golden_Y2.shape}")
    print(f"  Y3 = Y2 @ W3:  {golden_Y3.shape}")

    # ========================================================================
    # 使用 PLENAProgram API 定义计算
    # ========================================================================
    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    x_input = prog.input("X", shape=(batch, hidden_size))
    w1_input = prog.input("W1", shape=(hidden_size, hidden_size))
    w2_input = prog.input("W2", shape=(hidden_size, hidden_size))
    w3_input = prog.input("W3", shape=(hidden_size, out_features))

    w1_sub = prog.register_sub_matrix(w1_input, name="W1_sub")
    w2_sub = prog.register_sub_matrix(w2_input, name="W2_sub")
    w3_sub = prog.register_sub_matrix(w3_input, name="W3_sub")

    @prog.function
    def linear(x_in, w_sub_matrix):
        Y = prog.alloc("Y", batch, out_features)

        act = prog.load_batch(x_in, name="X")
        act_sub = prog.register_vram_sub_matrix(act, name="X_sub")

        num_row_blocks = batch // mlen
        num_col_blocks = out_features // mlen

        for row_idx in range(num_row_blocks):
            for col_idx in range(num_col_blocks):
                prog.reset_mram()
                w_sub_matrix.load_col(col_idx)

                prog.vram_sub_projection_to(
                    vram_row=act_sub.row(row_idx),
                    mram_col=w_sub_matrix.col(col_idx),
                    target=Y,
                    target_row_idx=row_idx,
                    target_col_idx=col_idx
                )
        return Y

    @prog.function
    def linear_and_store(x_in, w_sub_matrix):
        Y = linear(x_in, w_sub_matrix)
        Y_stored = prog.store(Y, name="Y_stored")
        # Y (VRAMMatrixVar) is auto-freed here since it's not returned
        # Y_stored (InputVar) is returned, can be loaded back later
        return Y_stored

    # 第一层：Y1 = X @ W1 → store to HBM
    Y1_stored = linear_and_store(x_input, w1_sub)
    print(f"  Y1 stored to HBM, VRAM freed")

    # 第二层：Y2 = Y1 @ W2 → store to HBM
    Y2_stored = linear_and_store(Y1_stored, w2_sub)
    print(f"  Y2 stored to HBM, VRAM freed")

    # 第三层：Y3 = Y2 @ W3（最后一层不需要 store）
    Y3 = linear(Y2_stored, w3_sub)
    print(f"  Y3: {Y3.shape}, internal={Y3.name}")

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

    input_tensor = {
        "X": X,
        "W1": W1,
        "W2": W2,
        "W3": W3,
    }

    golden_result = {
        "original_output": golden_Y3,
    }

    create_sim_env(input_tensor, gen_code, golden_result, [], build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="test_dual_naming_test",
        data=None,
        specified_data_order=["X", "W1", "W2", "W3"],
        build_path=build_dir
    )

    symbol_table = prog._compiler.symbol_table.table
    y3_info = symbol_table[Y3.name]

    comparison_params = {
        "start_row_idx": y3_info.vram_addr // mlen,
        "num_rows": (batch * out_features) // mlen,
        "num_batches": batch,
        "elements_per_batch": out_features,
        "row_dim": mlen,
        "use_stride_mode": True
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"  仿真环境已创建: {build_dir}")
    print(f"  Result location: row {y3_info.vram_addr // mlen}, {(batch * out_features) // mlen} rows")
    print(f"  Comparison params: {comparison_params}")
