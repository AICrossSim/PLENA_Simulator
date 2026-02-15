import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
# from acc_simulator.quantize.quantized_layers.linear import MXFPLinearPTQ
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import projection_asm, preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import build_sim_env


# TODOs: Need to integrate the MX quantizer here.
# quantized_layer = MXFPLinearPTQ.from_linear(
#     layer=original_layer,
#     x_meta=my_x_meta,
#     w_meta=my_w_meta,
#     b_meta=my_b_meta,
#     layer_type="XWB",
#     online_rotate=False,
#     clip_search_y=False
# )


if __name__ == "__main__":
    # Testing the operation (hidden_size, hidden_size) @ (hidden_size, batch_size)
    hidden_size = 128
    batch_size = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1]

    # Gen Weight and Test Data
    # generate_and_save_random_weights(hidden_size, hidden_size, get_weights_path('model_weights.pt'))
    
    torch.manual_seed(42)
    act_tensor = torch.randn(batch_size, hidden_size)
    original_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
    weights = original_layer.state_dict()
    
    original_output = original_layer(act_tensor)

    input_tensor = {
        "act_tensor": act_tensor,
        "weights": weights['weight'].t(),
    }

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": original_output
    }

    gen_assembly_code = "; Loop Test Generation \n"
    # Set the addr offset for weight and bias
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[int(hidden_size * batch_size * real_data_ratio), int((hidden_size * (batch_size + 1) + hidden_size * hidden_size) * real_data_ratio)]
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3]
    )
    
    # Gen Activation Preload
    gen_assembly_code += preload_act_asm(
        vlen=64,
        preload_len=4,
        batch=4,
        hidden_size=128,
        alive_registers=[1,2,3,4,5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=hidden_size
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3,4,5]
    )


    # gen_assembly_code += "S_LD_FP f1, gp0, 1 \n"
    # gen_assembly_code += "C_LOOP_START gp1, 3 \n"
    # gen_assembly_code += "V_ADD_VF gp2, gp2, f1, 0\n"
    # gen_assembly_code += "C_LOOP_END gp1 \n"

    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    build_sim_env(data_size=256, mode="behave_sim", asm="linear", data=None, specified_data_order = ["act_tensor", "weights"], build_path=build_path)

    print("================================================")
    print("Finished generating assembly code")
    print("================================================")
