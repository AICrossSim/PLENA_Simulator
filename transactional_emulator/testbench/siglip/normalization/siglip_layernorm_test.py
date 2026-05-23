#!/usr/bin/env python3
from pathlib import Path

import json
import torch

from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from transactional_emulator.testbench.siglip.local_asm_templates.layernorm_blocks import build_layernorm_inplace_asm
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.harness_utils import prepare_case_artifacts


def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    quantized, _, _, _ = _mx_fp_quantize_hardware(
        tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8]
    )
    return quantized.reshape(tensor.shape)


def emit_and_run_asm_test(build_path: Path) -> None:
    hidden_size = 64
    batch_size = 4
    seq_len = 1
    effective_batch = batch_size * seq_len

    real_data_ratio = (8 * 8 + 8) / (8 * 8)

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

    # Golden uses quantized activations to match what hardware sees after HBM preload.
    golden_output = torch.nn.functional.layer_norm(
        act_hbm.float(),
        (hidden_size,),
        eps=eps,
    ).to(torch.bfloat16)

    input_tensor = {
        "act_tensor": act_hbm,
    }
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": golden_output.flatten(),
    }

    gen_assembly_code = build_layernorm_inplace_asm(
        title="SigLIP LayerNorm Test",
        effective_batch=effective_batch,
        hidden_size=hidden_size,
        vlen=vlen,
        eps_fp_slot=eps_fp_slot,
        reci_hid_fp_slot=reci_hid_fp_slot,
    )

    fp_preload = [0.0] * 64
    fp_preload[eps_fp_slot] = eps
    fp_preload[reci_hid_fp_slot] = 1.0 / hidden_size

    build_path.mkdir(parents=True, exist_ok=True)

    act_hbm_size_mb = int((act_hbm.numel() * real_data_ratio + (1024 * 1024 - 1)) // (1024 * 1024))
    total_hbm_size_mb = act_hbm_size_mb + 10

    prepare_case_artifacts(
        case_build_dir=build_path,
        input_tensor=input_tensor,
        asm_code=gen_assembly_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=None,
        hbm_mb=total_hbm_size_mb,
        data_order=["act_tensor"],
    )

    comparison_params = {
        "start_row_idx": 0,
        "num_rows": (effective_batch * hidden_size) // vlen,
        "num_batches": effective_batch,
        "elements_per_batch": hidden_size,
        "use_stride_mode": True,
    }
    with open(build_path / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating SigLIP LayerNorm test")
    print(f"Result location: row 0, {comparison_params['num_rows']} rows")
    print(f"Expected shape: ({effective_batch}, {hidden_size})")
    print("================================================")

    run_and_assert(build_path, "siglip_layernorm", mlen=mlen, blen=blen)


if __name__ == "__main__":
    build_path = Path(__file__).parent / "build" / "siglip_layernorm"
    emit_and_run_asm_test(build_path)
