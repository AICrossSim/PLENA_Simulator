import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
# from acc_simulator.quantize.quantized_layers.linear import MXFPLinearPTQ
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import batched_matmul_asm, preload_addr_reg_asm, reset_reg_asm
from create_sim_env import create_sim_env
from sim_env_utils import create_mem_for_sim

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
        w_base_hbm_offset_reg=0,
        w_prefetch_amount=k,
        a_base_hbm_offset_reg=0,
        a_prefetch_amount=4,
        result_base_address=0,
    )


    create_sim_env(input_tensor, weight_2_tensor, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="linear", data=None, specified_data_order = ["input_tensor", "model_weights"])

    print("================================================")
    print("Finished generating assembly code")
    print("================================================")
