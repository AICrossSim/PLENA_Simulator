from re import I
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
# from acc_simulator.quantize.quantized_layers.linear import MXFPLinearPTQ
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import rms_norm_asm, projection_asm, preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from transactional_emulator.tools.create_sim_env import create_sim_env
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
    # V_RED_MAX accumulates: f1 = max(f1, max(row)).  f1 is never reset between rows,
    # so FP SRAM stores the running (cumulative) max, not per-row max.
    row_maxima = torch.max(input_tensor1.reshape(batch_size, vlen, vlen), dim=-1).values  # (1, 64)
    # Fold in the initial f1 value from fp_preload[1]
    row_maxima[:, 0] = torch.clamp(row_maxima[:, 0], min=fp_preload[1])
    original_output = torch.cummax(row_maxima, dim=-1).values  # (1, 64) running max

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
        alive_registers=[1,2,3,4,5],  # [a_actual_register, set_stride_register, result_register, outer_loop, inner_loop]
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
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="dllm", data=None, specified_data_order = ["input_tensor1","input_tensor2"], build_path=build_path)

    import json
    # Write comparison_params.json for view_mem.py
    # Output is in FP SRAM (not VRAM), so use compare_fpsram mode.
    # FP SRAM positions 0..vlen-1 hold the per-row max values.
    comparison_params = {
        "compare_fpsram": True,
        "fpsram_num_elements": vlen,
        "fpsram_l_start": 0,
        "fpsram_m_res_start": 0,
        "start_row_idx": 0,
        "num_rows": 0,
        "num_batches": batch_size,
        "elements_per_batch": vlen,
        "use_stride_mode": False,
    }
    build_path_obj = Path(__file__).parent / "build"
    with open(build_path_obj / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    # Save golden FP SRAM values for view_mem.py comparison
    golden_fpsram = {
        "golden_exp_sum_new": original_output.float().flatten(),
        "golden_exp_m_res": original_output.float().flatten(),
        "fpsram_l_start": 0,
        "fpsram_m_res_start": 0,
    }
    torch.save(golden_fpsram, build_path_obj / "golden_fpsram.pt")
    print("comparison_params.json and golden_fpsram.pt written")