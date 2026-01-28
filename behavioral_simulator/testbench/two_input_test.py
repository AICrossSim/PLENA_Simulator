from re import I
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
# from acc_simulator.quantize.quantized_layers.linear import MXFPLinearPTQ
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import rms_norm_asm, projection_asm, preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from create_sim_env import create_sim_env
from sim_env_utils import create_mem_for_sim
from tools.memory_mapping.hbm_addr_map import align_addr_to_hbm_bandwidth
import torch.nn.functional as F

if __name__ == "__main__":
 
    hidden_size = 64
    vlen = 64
    batch_size = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1e-6, 1/hidden_size]
    preload_amount = 4
    hbm_data_width = 64
    
    torch.manual_seed(42)
    input_tensor1 = torch.randn(batch_size, hidden_size)
    input_tensor2 = input_tensor1

    input_tensor = {
        "input_tensor1": input_tensor1,
        "input_tensor2": input_tensor1
    }

    weights = input_tensor1
    original_output = input_tensor1

    golden_result = {
        "input_tensor": input_tensor,
        "weights": weights,
        "original_output": original_output
    }
    
    gen_assembly_code = "; Two Input Test Generation \n"
    
    # Set the addr offset for weight and bias
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1],
        addr_reg_val=[int(align_addr_to_hbm_bandwidth(batch_size * hidden_size * real_data_ratio, hbm_data_width))]
    )
    print("batch_size * hidden_size * real_data_ratio", batch_size * hidden_size * real_data_ratio)
    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1]
    )
    
    # # Gen Activation Preload
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=preload_amount,
        batch=batch_size,
        hidden_size=hidden_size,
        alive_registers=[1,2,3],  # [a_actual_register, set_stride_register, result_register]
        act_vram_offset=0,
        activation_offset_reg=0
    )

    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=preload_amount,
        batch=batch_size,
        hidden_size=hidden_size,
        alive_registers=[1,2,3],  # [a_actual_register, set_stride_register, result_register]
        act_vram_offset=batch_size*hidden_size,
        activation_offset_reg=1
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3,4]
    )

    create_sim_env(input_tensor, weights, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="dllm", data=None, specified_data_order = ["input_tensor1","input_tensor2"])