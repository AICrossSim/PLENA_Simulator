#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.local_asm_templates.layernorm_blocks import build_layernorm_inplace_asm
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    prepare_case_artifacts,
    write_comparison_params,
)
from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    quantize_to_mxfp,
)

def emit_and_run_asm_test(build_path: Path) -> None:
    hidden_size = 64
    batch_size = 4
    seq_len = 1
    effective_batch = batch_size * seq_len

    real_data_ratio = MXFP_REAL_DATA_RATIO

    mlen = 64
    blen = 4
    vlen = 64

    eps = 1e-5
    eps_fp_slot = 1
    reci_hid_fp_slot = 2

    torch.manual_seed(42)

    act_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    act_matrix = act_tensor.reshape(effective_batch, hidden_size)
    act_hbm = quantize_to_mxfp(act_matrix).to(torch.bfloat16)

    gamma = torch.randn(hidden_size, dtype=torch.bfloat16)
    beta = torch.randn(hidden_size, dtype=torch.bfloat16)

    golden_output = F.layer_norm(
        act_hbm.float(),
        (hidden_size,),
        weight=gamma.float(),
        bias=beta.float(),
        eps=eps,
    ).to(torch.bfloat16)

    input_tensor = {
        "act_tensor": act_hbm,
    }
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": golden_output.flatten(),
    }

    scratch_vram_addr = effective_batch * hidden_size
    affine_weight_vram_offset = scratch_vram_addr + vlen
    affine_bias_vram_offset = affine_weight_vram_offset + hidden_size

    gen_assembly_code = build_layernorm_inplace_asm(
        title="SigLIP LayerNorm Affine Test",
        effective_batch=effective_batch,
        hidden_size=hidden_size,
        vlen=vlen,
        eps_fp_slot=eps_fp_slot,
        reci_hid_fp_slot=reci_hid_fp_slot,
        affine_weight_vram_offset=affine_weight_vram_offset,
        affine_bias_vram_offset=affine_bias_vram_offset,
    )

    fp_preload = [0.0] * 64
    fp_preload[eps_fp_slot] = eps
    fp_preload[reci_hid_fp_slot] = 1.0 / hidden_size

    build_path.mkdir(parents=True, exist_ok=True)

    total_vram_elems = affine_bias_vram_offset + hidden_size
    vram_preload = np.zeros(total_vram_elems, dtype=np.uint16)

    gamma_vector = gamma.contiguous().view(torch.uint16).cpu().numpy()
    beta_vector = beta.contiguous().view(torch.uint16).cpu().numpy()

    vram_preload[affine_weight_vram_offset : affine_weight_vram_offset + gamma_vector.size] = gamma_vector
    vram_preload[affine_bias_vram_offset : affine_bias_vram_offset + beta_vector.size] = beta_vector

    act_hbm_size_mb = int((act_hbm.numel() * real_data_ratio + (1024 * 1024 - 1)) // (1024 * 1024))
    total_hbm_size_mb = act_hbm_size_mb + 10

    prepare_case_artifacts(
        case_build_dir=build_path,
        input_tensor=input_tensor,
        asm_code=gen_assembly_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=vram_preload,
        hbm_mb=total_hbm_size_mb,
        data_order=["act_tensor"],
    )

    comparison_params = write_comparison_params(
        build_path,
        start_row_idx=0,
        num_rows=(effective_batch * hidden_size) // vlen,
        num_batches=effective_batch,
        elements_per_batch=hidden_size,
        use_stride_mode=True,
    )

    print("================================================")
    print("Finished generating SigLIP LayerNorm affine test")
    print(f"Result location: row 0, {comparison_params['num_rows']} rows")
    print(f"Expected shape: ({effective_batch}, {hidden_size})")
    print("================================================")

    run_and_assert(build_path, "siglip_layernorm_affine", mlen=mlen, blen=blen)


if __name__ == "__main__":
    build_path = Path(__file__).parent / "build" / "siglip_layernorm_affine"
    emit_and_run_asm_test(build_path)
