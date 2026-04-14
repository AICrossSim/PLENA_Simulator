"""
ATen Compiler Test: nn.RMSNorm(64) -> PLENA ISA via compile_module.

Traces the model with torch.export, compiles to ISA, runs the Rust emulator,
and checks numerical accuracy against the CPU golden reference.

Note: RMSNorm weight goes through HBM but PLENA's rms_norm_plena does not
apply it explicitly — this test validates the normalization pass only.
The golden reference omits the weight scaling to match PLENA hardware behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import torch
import torch.nn as nn
import json

from plena.compiler.aten_compiler import compile_module
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


if __name__ == "__main__":
    print("=" * 80)
    print("ATen Compiler Test: nn.RMSNorm(128) - Matched Dims")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    hidden_size = 128
    batch_size = 4
    mlen = 128
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    # ========================================================================
    # Model + test data
    # ========================================================================
    model = nn.RMSNorm(hidden_size)
    x = torch.randn(batch_size, hidden_size)

    print(f"\nInput x: {x.shape}")
    print(f"Weight:  {model.weight.shape}")

    # ========================================================================
    # CPU golden reference
    # PLENA rms_norm_plena does not apply the weight scaling (it uses FPRAM
    # preloaded eps and 1/hidden_size). Golden must match this behavior:
    # compute RMS norm without weight multiplication.
    # ========================================================================
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    golden = x * torch.rsqrt(variance + 1e-6)
    print(f"\nGolden output (no weight scaling): {golden.shape}")
    print(f"  golden[0,:4]: {golden[0, :4].tolist()}")

    # ========================================================================
    # Compile with ATen compiler
    # ========================================================================
    print("\n--- ATen Compiler (torch.export -> PLENA ISA) ---")
    isa_str, info = compile_module(model, (x,), mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    prog = info["prog"]
    tensor_map = info["tensor_map"]
    input_names = info["input_names"]
    hbm_input_order = info["hbm_input_order"]
    output_var = info["output_var"]
    state_dict_tensors = info["state_dict_tensors"]

    lines = isa_str.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")
    print(f"Input names (activations): {input_names}")
    print(f"HBM input order: {hbm_input_order}")
    print(f"State dict tensors: {list(state_dict_tensors.keys())}")
    print(f"Output var: {output_var}")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Build input_tensor dict: weight from state_dict, activation from x
    input_tensor = {}
    for name in hbm_input_order:
        if name in state_dict_tensors:
            input_tensor[name] = state_dict_tensors[name]
        else:
            # User input (activation)
            input_tensor[name] = x

    golden_result = {"original_output": golden}

    # FP SRAM preload: [0]=0.0, [1]=eps(1e-6), [2]=1/hidden_size
    # rms_norm_plena uses eps_offset=1, reci_hid_offset=2
    fp_preload = [0.0, 1e-6, 1.0 / hidden_size] + [0.0] * 7

    create_sim_env(
        input_tensor,
        isa_str,
        golden_result,
        fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="aten_compiler_rms_norm_matched",
        data=None,
        specified_data_order=hbm_input_order,
        build_path=build_dir,
    )

    # ========================================================================
    # Comparison params
    # RMS norm is in-place: result is at same VRAM location as input
    # ========================================================================
    # output_var from rms_norm_plena is the in-place result at the input VRAM addr
    x_vram_addr = prog._compiler.get_vram_addr(output_var.name)

    comparison_params = {
        "start_row_idx": x_vram_addr // mlen,
        "num_rows": (batch_size * hidden_size) // mlen,
        "num_batches": batch_size,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": hidden_size > mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(isa_str)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {x_vram_addr // mlen} (in-place)")

    # ========================================================================
    # Run emulator and check
    # ========================================================================
    run_and_assert(build_dir, "aten_compiler_rms_norm_matched", mlen=mlen, blen=blen)
