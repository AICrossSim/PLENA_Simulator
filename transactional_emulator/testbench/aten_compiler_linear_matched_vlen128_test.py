"""
ATen Compiler Test: nn.Linear(64, 64, bias=False) -> PLENA ISA via compile_module.

Traces the model with torch.export, compiles to ISA, runs the Rust emulator,
and checks numerical accuracy against the CPU golden reference.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import torch
import json

from plena.compiler.aten_compiler import compile_module, quantize_to_mxfp
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


if __name__ == "__main__":
    print("=" * 80)
    print("ATen Compiler Test: nn.Linear(64, 64, bias=False)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    in_features = 128
    out_features = 256
    batch_size = 8
    mlen = 128
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    # ========================================================================
    # Model + test data
    # ========================================================================
    model = torch.nn.Linear(in_features, out_features, bias=False)
    x = torch.randn(batch_size, in_features)

    print(f"\nInput x: {x.shape}")
    print(f"Weight:  {model.weight.shape}  (nn.Linear stores [out, in])")

    # ========================================================================
    # CPU golden reference (apply MXFP8 to weight since it goes through HBM)
    # ========================================================================
    W_T = model.weight.T.contiguous()  # (in, out)
    W_T_q = quantize_to_mxfp(W_T)
    golden = x @ W_T_q
    print(f"\nGolden output: {golden.shape}")
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

    # Build input_tensor dict: activation uses raw x, weights use transposed float
    # The dict keys must match hbm_input_order names
    input_tensor = {}
    for name in hbm_input_order:
        if name in state_dict_tensors:
            input_tensor[name] = state_dict_tensors[name]
        else:
            # User input (activation) — use the example input
            input_tensor[name] = x

    golden_result = {"original_output": golden}

    # FP SRAM preload: [0]=0.0, [1]=eps, [2]=1/in_features
    fp_preload = [0.0, 1e-6, 1.0 / in_features] + [0.0] * 7

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
        asm="aten_compiler_linear",
        data=None,
        specified_data_order=hbm_input_order,
        build_path=build_dir,
    )

    # ========================================================================
    # Comparison params
    # ========================================================================
    y_vram_addr = prog._compiler.get_vram_addr(output_var.name)

    comparison_params = {
        "start_row_idx": y_vram_addr // mlen,
        "num_rows": (batch_size * out_features) // mlen,
        "num_batches": batch_size,
        "elements_per_batch": out_features,
        "row_dim": mlen,
        "use_stride_mode": False,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(isa_str)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Output location: VRAM row {y_vram_addr // mlen}")

    # ========================================================================
    # Run emulator and check
    # ========================================================================
    run_and_assert(build_dir, "aten_compiler_linear", mlen=mlen, blen=blen)
