from re import I
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
# from acc_simulator.quantize.quantized_layers.linear import MXFPLinearPTQ
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import rms_norm_asm, projection_asm, preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from tools.memory_mapping.hbm_addr_map import align_addr_to_hbm_bandwidth
import torch.nn.functional as F

if __name__ == "__main__":
 
    
    vlen = 64
    load_len = vlen * vlen
    batch_size = 1
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1e-6]
    preload_amount = 4
    hbm_data_width = 64
    
    torch.manual_seed(42)
    fp_preload = [0, 1]
    # fp_sram = [fp_preload[0]] * vlen

    gen_assembly_code = "; S MAP V Test Generation \n"
    torch.manual_seed(42)
    
    input_tensor1 = torch.randn(batch_size, load_len)
    input_tensor = {
        "input_tensor1": input_tensor1,
        "input_tensor2": input_tensor1
    }
    weights = input_tensor1
    original_output = torch.max(input_tensor1.reshape(batch_size, vlen, vlen), dim=-1).values

    golden_result = {
        "input_tensor": input_tensor,
        "weights": weights,
        "original_output": original_output
    }
    # Gen Activation Preload for fp0
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=preload_amount,
        batch=batch_size,
        hidden_size=load_len,
        alive_registers=[1,2,3],  # [a_actual_register, set_stride_register, result_register]
        act_vram_offset=0,
        activation_offset_reg=0
    )
    gen_assembly_code += f"S_ADDI_INT gp1, gp0, 0 \n"
    for i in range(vlen):
        gen_assembly_code += f"V_RED_MAX f1, gp1, 0 \n"
        gen_assembly_code += f"S_ADDI_INT gp1, gp1, {vlen} \n"
        gen_assembly_code += f"S_ST_FP f1, gp0, {i} \n"

    # gen_assembly_code += "; V_RED_MAX f1, gp0, 0 \n"
    gen_assembly_code += f"S_ADDI_INT gp1, gp0, 0 \n"
    gen_assembly_code += f"S_MAP_V_FP gp1, gp0, 0 \n"
    
    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, weights, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="dllm", data=None, specified_data_order = ["input_tensor1","input_tensor2"], build_path=build_path)