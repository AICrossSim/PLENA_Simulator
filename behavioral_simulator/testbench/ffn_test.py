import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import Tensor, nn
from compiler.asm_templates import ffn_asm, preload_addr_reg_asm, reset_reg_asm, preload_act_asm
from create_sim_env import create_sim_env
from sim_env_utils import create_mem_for_sim
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware


def quantize_to_mxfp(tensor):
    """
    Quantize tensor to MXFP format matching hardware (E4M3 with 8-bit scale per block of 8).
    Returns the dequantized tensor (what hardware sees after HBM->VRAM load).
    """
    orig_shape = tensor.shape
    tensor_2d = tensor.reshape(-1, tensor.shape[-1])
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor_2d, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8]
    )
    return bm_x.reshape(orig_shape)


class LlamaFeedForward(nn.Module):
    """
    Standard FeedForward layer used in Llama architectures:
    y = W3(activation(W1(x)) * W2(x))
    where activation is SwiGLU in Llama2.
    """
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)  # up projection
        self.w2 = nn.Linear(dim, inter_dim, bias=False)  # gate projection
        self.w3 = nn.Linear(inter_dim, dim, bias=False)  # down projection
        self.act = torch.nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.w3(self.act(self.w1(x)) * self.w2(x))

    def forward_with_bf16_intermediates(self, x: Tensor) -> Tensor:
        """
        Forward pass matching hardware: float32 accumulation, BF16 intermediate storage.
        Each stage writes to VRAM (BF16), causing precision loss that accumulates.
        """
        x_f32 = x.float()
        w1_f32, w2_f32, w3_f32 = self.w1.weight.float(), self.w2.weight.float(), self.w3.weight.float()

        # Stage 1 & 2: up/gate projections (parallel in hardware)
        up = torch.nn.functional.linear(x_f32, w1_f32).to(torch.bfloat16)
        gate = torch.nn.functional.linear(x_f32, w2_f32).to(torch.bfloat16)

        # Stage 3: SiLU(up) * gate
        silu_gate = (self.act(up.float()) * gate.float()).to(torch.bfloat16)

        # Stage 4: down projection
        return torch.nn.functional.linear(silu_gate.float(), w3_f32).to(torch.bfloat16)


if __name__ == "__main__":
    hidden_size = 128
    inter_dim = 256
    batch_size = 4
    seq_len = 2
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1.0]  # [0]=0.0, [1]=1.0 for SiLU
    mlen = 64
    blen = 4
    vlen = 64

    torch.manual_seed(42)
    act_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    ffn = LlamaFeedForward(dim=hidden_size, inter_dim=inter_dim).bfloat16()

    # Use proper weight initialization (Xavier/Glorot) to keep values in reasonable range
    # This mimics real neural network initialization where weights are scaled by 1/sqrt(fan_in)
    scale_up = 1.0 / (hidden_size ** 0.5)
    scale_down = 1.0 / (inter_dim ** 0.5)
    weight_up_layer = torch.randn(inter_dim, hidden_size, dtype=torch.bfloat16) * scale_up
    weight_gate_layer = torch.randn(inter_dim, hidden_size, dtype=torch.bfloat16) * scale_up
    weight_down_layer = torch.randn(hidden_size, inter_dim, dtype=torch.bfloat16) * scale_down

    # Quantize all inputs to MXFP to match hardware precision
    act_mxfp = quantize_to_mxfp(act_tensor).to(act_tensor.dtype)
    weight_up_mxfp = quantize_to_mxfp(weight_up_layer).to(act_tensor.dtype)
    weight_gate_mxfp = quantize_to_mxfp(weight_gate_layer).to(act_tensor.dtype)
    weight_down_mxfp = quantize_to_mxfp(weight_down_layer).to(act_tensor.dtype)

    # Set quantized weights
    with torch.no_grad():
        ffn.w1.weight.copy_(weight_up_mxfp)
        ffn.w2.weight.copy_(weight_gate_mxfp)
        ffn.w3.weight.copy_(weight_down_mxfp)

    # Compute golden with MXFP-quantized inputs and BF16 intermediates
    # This matches hardware behavior where VRAM stores BF16 after each stage
    original_output = ffn.forward_with_bf16_intermediates(act_mxfp)

    input_tensor = {
        "act_tensor": act_mxfp.reshape(batch_size * seq_len, hidden_size),
        "weight_up_layer": weight_up_mxfp.t(),
        "weight_gate_layer": weight_gate_mxfp.t(),
        "weight_down_layer": weight_down_mxfp.t(),
    }

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": original_output.flatten()
    }

    gen_assembly_code = "; FFN Test Generation \n"

    # Set the addr offset for weights
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2, 3],
        available_registers=[1, 2, 3],
        addr_reg_val=[
            int(hidden_size * batch_size * seq_len * real_data_ratio),
            int(hidden_size * batch_size * seq_len * real_data_ratio) + int(hidden_size * inter_dim * real_data_ratio),
            int(hidden_size * batch_size * seq_len * real_data_ratio) + int(hidden_size * inter_dim * real_data_ratio) + int(inter_dim * hidden_size * real_data_ratio)
        ]
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(alive_registers=[1,2,3])

    # Preload Activation
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

    # FFN with loop instructions
    gen_assembly_code += ffn_asm(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        intermediate_size=inter_dim,
        alive_registers=[1,2,3,4,5,6,7,8,9,10],
        up_weight_hbm_offset_reg=1,
        gate_weight_hbm_offset_reg=2,
        down_weight_hbm_offset_reg=3,
        const_one_fp_address=1,
        activation_base_address=0,
        use_loop_instructions=True
    )

    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm=None, data=None,
                       specified_data_order=["act_tensor", "weight_up_layer", "weight_gate_layer", "weight_down_layer"])

    # Save comparison parameters for view_mem.py
    import json
    result_vram_offset = 0
    effective_batch = batch_size * seq_len
    result_start_row = result_vram_offset // vlen
    num_result_rows = (effective_batch * hidden_size) // vlen
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": effective_batch,
        "elements_per_batch": hidden_size
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating FFN test assembly code")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
