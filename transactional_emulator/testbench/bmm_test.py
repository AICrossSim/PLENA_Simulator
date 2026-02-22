import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
# from acc_simulator.quantize.quantized_layers.linear import MXFPLinearPTQ
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import batched_matmul_asm, preload_addr_reg_asm, reset_reg_asm
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim

if __name__ == "__main__":
    # Testing the operation (hidden_size, hidden_size) @ (hidden_size, batch_size)
    m = 64
    k = 128
    n = 128
    batch_size = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0]
    mlen = 64
    blen = 4

    # Gen Weight and Test Data
    # generate_and_save_random_weights(hidden_size, hidden_size, get_weights_path('model_weights.pt'))
    
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, m, k)
    weight_2_tensor = torch.randn(batch_size, k, n)
    original_output = torch.bmm(input_tensor, weight_2_tensor)

    golden_result = {
        "input_tensor": input_tensor,
        "weights": weight_2_tensor,
        "original_output": original_output
    }

    gen_assembly_code = "; Linear Test Generation \n"
    
    # Set the addr offset for weight and bias
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1],
        addr_reg_val=[int(m * k * batch_size * real_data_ratio)]
    )

    gen_assembly_code += reset_reg_asm(
        alive_registers=[1]
    )

    gen_assembly_code += batched_matmul_asm(
        mlen=mlen,
        blen=blen,
        b=batch_size,
        m=m,
        k=k,
        n=n,
        alive_registers=[1,2,3],
        w_base_hbm_offset_reg=1,
        w_prefetch_amount=k,
        a_base_hbm_offset_reg=0,
        a_prefetch_amount=4,
        result_base_address=2048,
    )


    build_path = Path(__file__).parent / "build"
    input_data = {"input_tensor": input_tensor, "model_weights": weight_2_tensor}
    create_sim_env(input_data, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="linear", data=None, specified_data_order = ["input_tensor", "model_weights"], build_path=build_path)

    # Save comparison parameters for view_mem.py
    import json

    # Result is (B, M, N) written row-major.  With stride_len = n // mlen = 2,
    # M_MM_WO writes blen rows spaced stride_len VRAM rows apart.  The two n_groups
    # (cols 0..63 and cols 64..127) interleave: n_group 0 fills even rows, n_group 1
    # fills odd rows.  Together they pack the full row-major (B, M, N) result into
    # contiguous VRAM rows starting at result_base_address // mlen.
    # Total VRAM rows = B * M * (N // mlen) = 4 * 64 * 2 = 512.
    result_start_row = 2048 // mlen  # 32: past activation area (rows 0-31)
    num_result_rows = batch_size * m * (n // mlen)  # 512
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": batch_size,
        "elements_per_batch": m * n,
        "row_dim": mlen,
        "use_stride_mode": False,
        "row_stride": 1,
    }
    with open(build_path / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating assembly code")
    print("================================================")
