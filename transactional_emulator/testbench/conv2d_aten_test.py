"""
ATen-style Conv2d Test — TRUE on-chip im2col on PLENA

Conv2d is mapped to an on-chip im2col + linear projection entirely on PLENA:
  1. Host provides raw NCHW input and a mask vector in HBM.
     The input is stored row-padded to W_padded=64 so that every HBM row
     start is 64-element aligned (required by H_PREFETCH_V).
  2. PLENA assembles im2col rows in VRAM using:
       H_PREFETCH_V  (load K contiguous elements per (c, kr) pair)
       V_MUL_VV      (zero out elements beyond K using mask vector)
       V_SHFT_V      (right-shift to target column position)
       V_ADD_VV      (accumulate into output row)
  3. The assembled im2col matrix is multiplied by the weight via the
     systolic array (same linear_plena path as before).

Parameters chosen so that:
  - OW = 1 (ow = 0 always) → HBM offsets (c*H + oh+kr)*W_padded are
    always multiples of 64, satisfying H_PREFETCH_V alignment.
  - M = OH * OW = 64 = mlen  ✓
  - K_col = C_in * K * K = 64 = mlen  ✓
  - N = C_out = 64 = mlen  ✓

Uses the PLENA ATen-style registry:
    import plena.ops as ops
    Y = ops.conv2d(prog, input_raw_var, mask_vec_var, w_2d_input, conv_params)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.conv2d(X_col, W_2d)   # CPU: pre-im2col'd tensors
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
    col = col.permute(0, 2, 1).reshape(-1, C * kernel_size * kernel_size)
    return col


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style Conv2d Test  (TRUE on-chip im2col via V_SHFT_V)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    B        = 1
    C_in     = 4
    H        = 67     # OH = (67-4)//1+1 = 64
    W        = 4      # OW = (4-4)//1+1  = 1  → ow=0 always (64-aligned HBM)
    W_padded = 64     # HBM row stride; must be a multiple of 64
    K_size   = 4      # C_in * K * K = 4*4*4 = 64 == mlen  ✓
    C_out    = 64
    stride   = 1
    padding  = 0
    mlen     = 64
    blen     = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    OH = (H - K_size + 2 * padding) // stride + 1   # 64
    OW = (W - K_size + 2 * padding) // stride + 1   # 1
    M  = B * OH * OW                                  # 64 == mlen  ✓
    K_col = C_in * K_size * K_size                    # 64 == mlen  ✓
    N  = C_out                                         # 64 == mlen  ✓

    assert OW == 1, f"OW must be 1 for 64-aligned H_PREFETCH_V, got {OW}"
    assert W_padded % 64 == 0, f"W_padded must be a multiple of 64, got {W_padded}"
    assert M == mlen, f"M={M} must equal mlen={mlen}"

    torch.manual_seed(42)

    # ========================================================================
    # Raw tensors
    # ========================================================================
    input_4d  = torch.randn(B, C_in, H, W)
    weight_4d = torch.randn(C_out, C_in, K_size, K_size)

    print(f"\nInput:  {input_4d.shape}")
    print(f"Weight: {weight_4d.shape}")
    print(f"OH={OH}, OW={OW}, M={M}, K_col={K_col}, N={N}, W_padded={W_padded}")

    # ========================================================================
    # im2col on CPU (used only for the golden reference)
    # ========================================================================
    X_col = im2col(input_4d, K_size, stride=stride, padding=padding)  # [M, K_col]
    W_2d  = weight_4d.float().reshape(C_out, -1).T.contiguous()        # [K_col, C_out]

    print(f"\nim2col X_col (CPU): {X_col.shape}")
    print(f"W_2d:               {W_2d.shape}")

    # ========================================================================
    # Mask vector: K_size ones, (mlen - K_size) zeros
    # ========================================================================
    mask_vec = torch.zeros(mlen)
    mask_vec[:K_size] = 1.0
    print(f"\nmask_vec[:8]: {mask_vec[:8].tolist()}  (first {K_size} are 1.0)")

    # ========================================================================
    # Padded input: each of the C_in*H rows is zero-padded to W_padded=64
    # so that row starts (c*H + h)*W_padded are multiples of 64 in HBM.
    # ========================================================================
    input_padded = torch.zeros(C_in * H, W_padded)
    input_padded[:, :W] = input_4d.reshape(C_in * H, W)
    print(f"\ninput_padded: {input_padded.shape}  (actual data in cols 0..{W-1})")

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
    ref_conv = F.conv2d(input_4d, weight_4d, stride=stride, padding=padding)
    ref_2d   = ref_conv.permute(0, 2, 3, 1).reshape(M, N)
    max_diff = (golden_Y - ref_2d).abs().max().item()
    print(f"  Max diff vs F.conv2d: {max_diff:.2e}  (should be ~0)")

    # ========================================================================
    # PLENA backend — TRUE on-chip im2col via V_SHFT_V
    # ========================================================================
    print("\n--- PLENA Backend (on-chip im2col via V_SHFT_V) ---")
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare HBM inputs (auto-allocated in declaration order):
    #   [0] raw input  shape (C_in*H, W_padded) — row-padded NCHW
    #   [1] mask vector     shape (1, mlen)
    #   [2] weight (2-D)    shape (K_col, N)
    input_raw_var = prog.input("input_raw", shape=(C_in * H, W_padded))
    mask_vec_var  = prog.input("mask_vec",  shape=(1, mlen))
    w_2d_input    = prog.input("W_2d",      shape=(K_col, N))

    conv_params = dict(
        C_in=C_in, H=H, W=W, K=K_size,
        OH=OH, OW=OW, M=M, K_col=K_col,
        vlen=mlen, W_padded=W_padded,
    )

    # Dispatch: conv2d_plena assembles im2col in VRAM then runs linear
    Y = ops.conv2d(prog, input_raw_var, mask_vec_var, w_2d_input, conv_params)

    # Compile to ISA
    gen_code = prog.compile()
    asm_lines = gen_code.splitlines()
    print(f"\nGenerated {len(asm_lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Tensors stored in HBM in the same order as prog.input declarations
    input_tensor = {
        "input_raw": input_padded.float(),
        "mask_vec":  mask_vec.reshape(1, mlen).float(),
        "W_2d":      W_2d,
    }
    golden_result = {"original_output": golden_Y}

    # FP SRAM preload: [0]=0.0, [1]=eps(1e-6), [2]=1/K_col
    fp_preload = [0.0, 1e-6, 1.0 / K_col] + [0.0] * 7

    create_sim_env(
        input_tensor, gen_code, golden_result, fp_preload,
        build_dir=str(build_dir)
    )

    create_mem_for_sim(
        data_size=512,
        mode="behave_sim",
        asm="linear_aten",
        data=None,
        specified_data_order=["input_raw", "mask_vec", "W_2d"],
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
