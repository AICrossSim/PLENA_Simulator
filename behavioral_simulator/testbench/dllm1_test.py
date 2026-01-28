import sys
import random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
# from acc_simulator.quantize.quantized_layers.linear import MXFPLinearPTQ
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import  preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from create_sim_env import create_sim_env
from sim_env_utils import create_mem_for_sim
import torch.nn.functional as F

from tools.memory_mapping.hbm_addr_map import align_addr_to_hbm_bandwidth
from transformers import AutoTokenizer


if __name__ == "__main__":


    # Testing the operation (hidden_size, hidden_size) @ (hidden_size, batch_size)
    vocal_size = 2
    hidden_size = 128
    vlen = 64
    batch_size = 4
    preload_amount = 1
    real_data_ratio = (8*8 + 8) / (8 * 8)
    hbm_data_width = 64
    fp_preload = [0.0, 1e-6]
    int_preload = [random.randint(0, 10) for _ in range(vlen)]
    
    torch.manual_seed(42)
    logits = torch.randn(batch_size, hidden_size)
    weights = logits
    original_output = logits

    # Generate vlen random int32 data
    int_preload = torch.randint(low=0, high=10, size=(vlen,), dtype=torch.int32)


    print('logits.shape= ', logits.shape)
    print('original_output.shape= ', original_output.shape)

    
    input_tensor = {
        "logits": logits,
        "int": int_preload,
    }

    golden_result = {
        "input_tensor": input_tensor,
        "weights": weights,
        "original_output": original_output
    }
    print('original_output.shape = ',original_output.shape)
    print('original_output = ',original_output)
    
    gen_assembly_code = "; DLLM Test Generation \n"
    
    # Set the addr offset for mask

    # gen_assembly_code += preload_addr_reg_asm(
    #     addr_reg_to_set=[1,2],
    #     available_registers=[1,2],
    #     addr_reg_val=[int(align_addr_to_hbm_bandwidth(batch_size * hidden_size * vocal_size * real_data_ratio, hbm_data_width)),int(2*align_addr_to_hbm_bandwidth(batch_size * hidden_size * vocal_size * real_data_ratio, hbm_data_width))]
    # )


    # Reset the registers
    # gen_assembly_code += reset_reg_asm(
    #     alive_registers=[1,2,3,4,5,6]
    # )
    
    # Gen logtis Preload (B,L,V)
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=preload_amount,
        batch=batch_size,
        hidden_size=hidden_size,
        alive_registers=[1,2,3,4,5],  # [a_actual_register, set_stride_register, result_register, outer_loop_register, inner_loop_register]
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=hidden_size
    )

    # # Preload Integer Activation to the later section.
    # gen_assembly_code += f"S_ADDI_INT gp1, gp0, {hidden_size * batch_size} \n"
    # gen_assembly_code += f"S_ADDI_INT gp2, gp0, {int(hidden_size * batch_size * real_data_ratio)} \n"
    # gen_assembly_code += "H_PREFETCH_V gp1, gp2, a0, 0, 2, 1 \n"
    
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, int_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="dllm", data=None, specified_data_order = ["logits", "int"])