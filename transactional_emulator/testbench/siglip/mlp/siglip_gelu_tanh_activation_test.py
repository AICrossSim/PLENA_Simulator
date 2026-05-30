#!/usr/bin/env python3
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler.asm_templates import gelu_tanh_asm, preload_act_asm, reset_reg_asm
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    prepare_case_artifacts,
    write_comparison_params,
)
from transactional_emulator.testbench.siglip.utils.math import quantize_to_mxfp


def gelu_tanh_with_bf16_intermediates(x: torch.Tensor) -> torch.Tensor:
    """GELU tanh approximation with BF16 truncation at hardware-visible steps."""
    x_f32 = x.float()

    x2 = (x_f32 * x_f32).to(torch.bfloat16)
    x3 = (x2.float() * x_f32).to(torch.bfloat16)
    cubic_term = (0.044715 * x3.float()).to(torch.bfloat16)
    poly = (x_f32 + cubic_term.float()).to(torch.bfloat16)

    z = (0.7978845608028654 * poly.float()).to(torch.bfloat16)  # sqrt(2/pi)
    two_z = (z.float() + z.float()).to(torch.bfloat16)
    exp_2z = torch.exp(two_z.float()).to(torch.bfloat16)

    num = (exp_2z.float() - 1.0).to(torch.bfloat16)
    den = (num.float() + 2.0).to(torch.bfloat16)
    tanh_z = (num.float() * (1.0 / den.float())).to(torch.bfloat16)

    one_plus_tanh = (1.0 + tanh_z.float()).to(torch.bfloat16)
    half_x = (0.5 * x_f32).to(torch.bfloat16)
    return (half_x.float() * one_plus_tanh.float()).to(torch.bfloat16)


if __name__ == "__main__":
    hidden_size = 128
    batch_size = 4
    vlen = 64

    torch.manual_seed(42)
    act_tensor = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
    act_mxfp = quantize_to_mxfp(act_tensor).to(torch.bfloat16)

    # FP SRAM layout: [0]=0.0, [1]=1.0, [2]=0.5, [3]=0.044715, [4]=sqrt(2/pi)
    fp_preload = [0.0, 1.0, 0.5, 0.044715, 0.7978845608028654]

    golden_output = gelu_tanh_with_bf16_intermediates(act_mxfp)
    pytorch_tanh_output = F.gelu(act_mxfp.float(), approximate="tanh").to(torch.bfloat16)

    diff_vs_pytorch = torch.abs(golden_output.float() - pytorch_tanh_output.float())
    mae_vs_pytorch = diff_vs_pytorch.mean().item()
    max_err_vs_pytorch = diff_vs_pytorch.max().item()
    close_rate_1e2 = (torch.isclose(golden_output.float(), pytorch_tanh_output.float(), atol=1e-2, rtol=1e-2).float().mean() * 100).item()
    close_rate_2e2 = (torch.isclose(golden_output.float(), pytorch_tanh_output.float(), atol=2e-2, rtol=2e-2).float().mean() * 100).item()

    print("PyTorch GELU(tanh) comparison against BF16-step hardware-style golden:")
    print(f"  MAE: {mae_vs_pytorch:.6f}")
    print(f"  Max Error: {max_err_vs_pytorch:.6f}")
    print(f"  Match Rate @ atol=rtol=1e-2: {close_rate_1e2:.2f}%")
    print(f"  Match Rate @ atol=rtol=2e-2: {close_rate_2e2:.2f}%")

    input_tensor = {"act_tensor": act_mxfp}
    golden_result = {"input_tensor": input_tensor, "original_output": golden_output.flatten()}

    total_elements = batch_size * hidden_size
    scratch0_base = total_elements
    scratch1_base = total_elements * 2

    gen_assembly_code = "; SigLIP GELU tanh activation test\n"
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=batch_size,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=hidden_size,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])
    gen_assembly_code += gelu_tanh_asm(
        const_one_fp_address=1,
        const_half_fp_address=2,
        const_cubic_fp_address=3,
        const_sqrt_2_over_pi_fp_address=4,
        alive_registers=[1, 2, 3, 4, 5],
        activation_base_address=0,
        scratchpad0_base_address=scratch0_base,
        scratchpad1_base_address=scratch1_base,
        vlen=vlen,
        batch_size=batch_size,
        hidden_dim=hidden_size,
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
        data_order=["act_tensor"],
    )

    result_start_row = 0
    num_result_rows = (batch_size * hidden_size) // vlen
    write_comparison_params(
        build_path,
        start_row_idx=result_start_row,
        num_rows=num_result_rows,
        num_batches=batch_size,
        elements_per_batch=hidden_size,
        use_stride_mode=True,
    )

    print("================================================")
    print("Finished generating SigLIP GELU tanh activation test")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Expected shape: ({batch_size}, {hidden_size})")
    print("================================================")

    run_and_assert(build_path, "siglip_gelu_tanh_activation", mlen=vlen, blen=4)
