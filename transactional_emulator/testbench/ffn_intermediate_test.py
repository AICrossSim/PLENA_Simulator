import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
# from acc_simulator.quantize.quantized_layers.linear import MXFPLinearPTQ
from test_data_gen import get_weights_path, generate_and_save_random_weights
from compiler.asm_templates import ffn_up_silu_asm, preload_addr_reg_asm, reset_reg_asm, preload_act_asm, rms_norm_asm
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim


class LlamaFeedForward(nn.Module):
    """
    Standard FeedForward layer used in Llama architectures:
    y = W2(activation(W1(x)))
    where activation is typically SwiGLU in Llama2.

    Args:
        dim (int): input and output dimension (hidden size)
        inter_dim (int): intermediate/fc dimension
        activation (callable): nonlinearity to use (default: SwiGLU)
    """
    def __init__(self, dim: int, inter_dim: int, activation: str = "silu"):
        super().__init__()
        # Llama uses SwiGLU: x * silu(x)
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        print("w1:", self.w1.weight.shape)
        self.w2 = nn.Linear(dim, inter_dim, bias=False)
        print("w2:", self.w2.weight.shape)
        self.w3 = nn.Linear(inter_dim, dim, bias=False)
        print("w3:", self.w3.weight.shape)
        self.act = torch.nn.SiLU() if activation == "silu" else getattr(torch.nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(self.act(self.w1(x)) * self.w2(x))
    
    

if __name__ == "__main__":
    # Testing the operation (hidden_size, hidden_size) @ (hidden_size, batch_size)
    hidden_size = 128
    inter_dim = 256
    batch_size = 4
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1]
    mlen = 64
    blen = 4
    vlen = 64
    seq_len = 2

    torch.manual_seed(42)
    act_tensor = torch.rand(batch_size, seq_len, hidden_size)

    ffn = LlamaFeedForward(dim=hidden_size, inter_dim=inter_dim)

    weight_up_layer = torch.randn(inter_dim, hidden_size)
    weight_gate_layer = torch.randn(inter_dim, hidden_size)
    weight_down_layer = torch.randn(hidden_size, inter_dim)

    # Set weights for w1, w2, w3 to the generated tensors
    with torch.no_grad():
        ffn.w1.weight.copy_(weight_up_layer)
        ffn.w2.weight.copy_(weight_gate_layer)
        ffn.w3.weight.copy_(weight_down_layer)

    # Compute intermediate result: up_proj(x) = w1(x)
    # This is what we want to check - only the up projection (no SILU)
    act_reshaped = act_tensor.reshape(batch_size * seq_len, hidden_size)
    up_proj_result = torch.matmul(act_reshaped, weight_up_layer.t())  # (batch*seq_len, inter_dim)
    silu_up = torch.nn.functional.silu(up_proj_result)  # Apply SILU to up projection
    
    # Also compute full output for reference (not used for comparison)
    original_output = ffn(act_tensor)

    input_tensor = {
        "act_tensor": act_reshaped,
        "weight_up_layer": weight_up_layer.t(),
        "weight_gate_layer": weight_gate_layer.t(),
        "weight_down_layer": weight_down_layer.t(),
    }

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": up_proj_result.flatten()  # Use up_proj result as golden
    }

    gen_assembly_code = "; FFN Test Generation \n"

    # Set the addr offset for weight (only need up for up+silu test)
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1],
        addr_reg_val=[int(hidden_size * batch_size * seq_len * real_data_ratio)]
    )

    print("up_addr_hbm_val:", int(hidden_size * batch_size * seq_len * real_data_ratio))

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1]
    )

    # Preload Activation (needs 5 registers)
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=batch_size * seq_len,
        hidden_size=hidden_size,
        alive_registers=[1,2,3,4,5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=hidden_size
    )

    # FFN Up + SILU Generation (only up projection + SILU, no gate, no down)
    # Uses loop version - needs 10 registers (gp1-gp10)
    gen_assembly_code += ffn_up_silu_asm(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=inter_dim,
        alive_registers=[1,2,3,4,5,6,7,8,9,10],
        up_weight_hbm_offset_reg=1,
        const_one_fp_address=1,
        activation_base_address=0
    )
    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3,4,5,6,7,8,9,10]
    )

    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm=None, data=None, specified_data_order = ["act_tensor", "weight_up_layer", "weight_gate_layer", "weight_down_layer"], build_path=build_path)

    # Save comparison parameters for view_mem.py
    # Result (up projection only) is stored at up_result_register location
    # which is batch * seq_len * hidden_size (not at activation_base_address)
    import json
    effective_batch = batch_size * seq_len
    intermediate_vram_offset = batch_size * seq_len * hidden_size  # Location where result is stored
    result_start_row = intermediate_vram_offset // vlen
    num_result_rows = (effective_batch * inter_dim) // vlen  # Intermediate size
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": effective_batch,
        "elements_per_batch": inter_dim  # Intermediate dimension
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating assembly code (Up projection only, no SILU)")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print(f"Expected shape: ({effective_batch}, {inter_dim})")
    print("Checking GEMM/accumulator operations in up projection")
    print("================================================")