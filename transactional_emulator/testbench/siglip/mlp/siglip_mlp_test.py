#!/usr/bin/env python3
from pathlib import Path

import torch
from torch import nn

from transactional_emulator.testbench.siglip.mlp.asm_pipeline import build_mlp_pipeline_asm
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    gelu_fp_preload,
    gelu_with_bf16_intermediates,
    quantize_to_mxfp,
)
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    prepare_case_artifacts,
    write_comparison_params,
)


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

    real_data_ratio = MXFP_REAL_DATA_RATIO
    fp_preload = gelu_fp_preload(one_slot=1, coeff_slot=4)

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
    prepare_case_artifacts(
        case_build_dir=build_path,
        input_tensor=input_tensor,
        asm_code=gen_assembly_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=None,
        hbm_mb=16,
        data_order=["act_tensor", "weight_up_layer", "weight_down_layer"],
    )

    result_start_row = final_vram_offset // vlen
    num_result_rows = (effective_batch * hidden_size) // vlen
    write_comparison_params(
        build_path,
        start_row_idx=result_start_row,
        num_rows=num_result_rows,
        num_batches=effective_batch,
        elements_per_batch=hidden_size,
        use_stride_mode=True,
    )

    print("================================================")
    print("Finished generating SigLIP MLP test")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Expected shape: ({effective_batch}, {hidden_size})")
    print("================================================")

    run_and_assert(build_path, "siglip_mlp", mlen=mlen, blen=blen)
