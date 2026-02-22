"""
ATen-style Conv2d Test (im2col + linear projection)

Conv2d is mapped to a linear projection by:
  1. im2col: input [B, C_in, H, W]  ->  col [B*OH*OW, C_in*K*K]
  2. weight reshape: [C_out, C_in, K, K] -> [C_in*K*K, C_out]
  3. output = col @ weight_2d  ->  [B*OH*OW, C_out]

Uses the PLENA ATen-style registry:
    import plena.ops as ops
    Y = ops.conv2d(prog, X_col_batch, W_2d_input)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.conv2d(X_col, W_2d)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import json

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops

from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


def im2col(input_4d: torch.Tensor, kernel_size: int, stride: int = 1, padding: int = 0) -> torch.Tensor:
    """Transform input [B, C, H, W] -> [B*OH*OW, C*K*K] via torch.nn.Unfold."""
    B, C, H, W = input_4d.shape
    unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
    col = unfold(input_4d.float())          # [B, C*K*K, OH*OW]
    col = col.permute(0, 2, 1).reshape(-1, C * kernel_size * kernel_size)  # [B*OH*OW, C*K*K]
    return col


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style Conv2d Test  (plena.ops.conv2d via im2col + linear)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    B        = 1
    C_in     = 4
    H        = 11
    W        = 11
    K_size   = 4      # C_in * K * K = 4*4*4 = 64 == mlen  ✓
    C_out    = 64
    stride   = 1
    padding  = 0
    mlen     = 64
    blen     = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    OH = (H - K_size + 2 * padding) // stride + 1  # 8
    OW = (W - K_size + 2 * padding) // stride + 1  # 8
    M  = B * OH * OW                                 # 64 == mlen  ✓
    K_col = C_in * K_size * K_size                   # 64 == mlen  ✓
    N  = C_out                                        # 64 == mlen  ✓

    torch.manual_seed(42)

    # ========================================================================
    # Raw tensors
    # ========================================================================
    input_4d = torch.randn(B, C_in, H, W)
    weight_4d = torch.randn(C_out, C_in, K_size, K_size)

    print(f"\nInput:  {input_4d.shape}")
    print(f"Weight: {weight_4d.shape}")
    print(f"OH={OH}, OW={OW}, M={M}, K_col={K_col}, N={N}")

    # ========================================================================
    # im2col transform (done by host driver before dispatching to PLENA)
    # ========================================================================
    X_col  = im2col(input_4d, K_size, stride=stride, padding=padding)  # [M, K_col]
    W_2d   = weight_4d.float().reshape(C_out, -1).T.contiguous()        # [K_col, C_out]

    print(f"\nim2col X_col:  {X_col.shape}")
    print(f"W_2d:          {W_2d.shape}")

    # ========================================================================
    # Load ATen-style operator registry
    # ========================================================================
    registry = OpRegistry.load()
    print(f"\nLoaded ops: {registry.list_ops()}")

    # ========================================================================
    # CPU golden reference (matmul on im2col tensors)
    # ========================================================================
    print("\n--- CPU Golden Reference ---")
    registry.set_backend(Backend.CPU)
    golden_Y = ops.conv2d(X_col, W_2d)        # [M, N] = [64, 64]
    print(f"  golden_Y: {golden_Y.shape}")
    print(f"  golden_Y[0,:4]: {golden_Y[0,:4].tolist()}")

    # Cross-check against F.conv2d
    ref_conv = F.conv2d(input_4d, weight_4d, stride=stride, padding=padding)  # [B, C_out, OH, OW]
    ref_2d   = ref_conv.permute(0, 2, 3, 1).reshape(M, N)
    max_diff = (golden_Y - ref_2d).abs().max().item()
    print(f"  Max diff vs F.conv2d: {max_diff:.2e}  (should be ~0)")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs: X_col in VRAM (load_batch), W_2d stays in HBM
    x_col_input = prog.input("X_col", shape=(M, K_col))
    w_2d_input  = prog.input("W_2d",  shape=(K_col, N))
    X_col_batch = prog.load_batch(x_col_input, name="X_col")

    # ATen-style dispatch: conv2d_plena() is called with (prog, X_col_batch, w_2d_input)
    Y = ops.conv2d(prog, X_col_batch, w_2d_input)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor  = {"X_col": X_col, "W_2d": W_2d}
    golden_result = {"original_output": golden_Y}

    # FP SRAM preload: [0]=0.0, [1]=eps(1e-6), [2]=1/K_col
    fp_preload = [0.0, 1e-6, 1.0 / K_col] + [0.0] * 7

    create_sim_env(
        input_tensor, gen_code, golden_result, fp_preload,
        build_dir=str(build_dir)
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="linear_aten",
        data=None,
        specified_data_order=["X_col", "W_2d"],
        build_path=build_dir,
    )

    symbol_table = prog._compiler.symbol_table.table
    y_info = symbol_table[Y.name]

    comparison_params = {
        "start_row_idx":      y_info.vram_addr // mlen,
        "num_rows":           (M * N) // mlen,
        "num_batches":        M,
        "elements_per_batch": N,
        "row_dim":            mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Y location: VRAM row {y_info.vram_addr // mlen}")
    run_and_assert(build_dir, "conv2d", mlen=mlen, blen=blen)
