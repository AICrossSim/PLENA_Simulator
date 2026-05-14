#!/usr/bin/env python3
from pathlib import Path
import sys

import json
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[3]))

from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from transactional_emulator.testbench.siglip.local_asm_templates.mlp_blocks import build_mlp_pipeline_asm
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.emulator_runner import run_and_assert


def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    quantized, _, _, _ = _mx_fp_quantize_hardware(
        tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8]
    )
    return quantized.reshape(tensor.shape)


def gelu_with_bf16_intermediates(tensor: torch.Tensor) -> torch.Tensor:
    tensor_f32 = tensor.float()
    step1 = (1.702 * tensor_f32).to(torch.bfloat16)
    step2 = (-step1.float()).to(torch.bfloat16)
    step3 = torch.exp(step2.float()).to(torch.bfloat16)
    step4 = (1.0 + step3.float()).to(torch.bfloat16)
    step5 = (1.0 / step4.float()).to(torch.bfloat16)
    return (tensor_f32 * step5.float()).to(torch.bfloat16)


class SiglipMLP(nn.Module):
    def __init__(self, hidden_size: int = 1152, intermediate_size: int = 4304):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    hidden_size = 64
    intermediate_size = 128
    batch_size = 4
    seq_len = 1

    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    fp_preload = [0.0, 1.0, 0.0, 0.0, 1.702]

    mlen = 64
    blen = 4
    vlen = 64

    torch.manual_seed(42)

    act_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    effective_batch = batch_size * seq_len
    act_matrix = act_tensor.reshape(effective_batch, hidden_size)
    act_hbm = quantize_to_mxfp(act_matrix).to(torch.bfloat16)

    mlp = SiglipMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
    weight_up = torch.randn(intermediate_size, hidden_size, dtype=torch.bfloat16)
    weight_down = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)
    with torch.no_grad():
        mlp.fc1.weight.copy_(weight_up)
        mlp.fc2.weight.copy_(weight_down)

    weight_up_hbm = quantize_to_mxfp(weight_up).to(torch.bfloat16)
    weight_down_hbm = quantize_to_mxfp(weight_down).to(torch.bfloat16)

    aligned_intermediate = ((intermediate_size + mlen - 1) // mlen) * mlen
    pad_rows = aligned_intermediate - intermediate_size
    weight_up_padded = torch.nn.functional.pad(weight_up_hbm.t(), (0, pad_rows)).contiguous()
    weight_down_padded = torch.nn.functional.pad(weight_down_hbm.t(), (0, 0, 0, pad_rows)).contiguous()

    up_proj = torch.matmul(act_hbm.float(), weight_up_padded.float())
    activated = gelu_with_bf16_intermediates(up_proj)
    final_output = torch.matmul(activated.float(), weight_down_padded.float())
    final_output = final_output[:, :hidden_size].contiguous()

    input_tensor = {
        "act_tensor": act_hbm,
        "weight_up_layer": weight_up_padded,
        "weight_down_layer": weight_down_padded,
    }

    golden_result = {"input_tensor": input_tensor, "original_output": final_output.flatten()}

    gen_assembly_code, final_vram_offset = build_mlp_pipeline_asm(
        title="SigLIP MLP FC1 + GELU + FC2 Test",
        effective_batch=effective_batch,
        hidden_size=hidden_size,
        aligned_intermediate=aligned_intermediate,
        real_data_ratio=real_data_ratio,
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        weight_up_numel=weight_up_padded.numel(),
        weight_hbm_reg=1,
        weight_down_hbm_reg=2,
        gelu_one_fp_slot=1,
        gelu_1702_fp_slot=4,
    )

    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)

    create_mem_for_sim(
        data_size=16,
        mode="behave_sim",
        asm=None,
        data=None,
        specified_data_order=["act_tensor", "weight_up_layer", "weight_down_layer"],
        build_path=build_path,
    )

    result_start_row = final_vram_offset // vlen
    num_result_rows = (effective_batch * hidden_size) // vlen
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": effective_batch,
        "elements_per_batch": hidden_size,
        "use_stride_mode": True,
    }
    with open(build_path / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating SigLIP MLP test")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Expected shape: ({effective_batch}, {hidden_size})")
    print("================================================")

    run_and_assert(build_path, "siglip_mlp", mlen=mlen, blen=blen)