"""
ATen-style Conv2d Test — TRUE on-chip im2col (documented ISA only) + PLENA systolic matmul

Conv2d is implemented as:
  1. PLENA executes on-chip im2col entirely on the hardware:
       raw NCHW input in HBM -> im2col matrix in VRAM
     using only formally documented instructions:
       H_PREFETCH_V, V_MUL_VV, V_RED_SUM, S_ST_FP, S_MAP_V_FP
  2. PLENA runs the systolic matmul on the im2col matrix.

No V_SHFT_V (opcode 0x32) is used — it is present in operation.svh but is not
formally supported per the PhD lead.

All instructions used are fully documented in the ISA spec.

HBM alignment constraint:
  H_PREFETCH_V requires the HBM element address to be a multiple of 64.
  With W_padded=64 and OW=1 (ow=0 always): offset=(c*H+oh+kr)*64 — always aligned.

Parameters chosen so all tile dimensions = mlen = 64:
  C_in=4, H=67, W=4, K=4 -> K_col = 4*4*4 = 64
  OH = 67-4+1 = 64, OW = 4-4+1 = 1, M = 64
  W_padded = 64

Uses the PLENA ATen-style registry:
    import plena.ops as ops
    Y = ops.conv2d(prog, input_raw_var, w_2d_var,
                   C_in=C_in, H=H, W=W, K=K_size,
                   OH=OH, OW=OW, M=M, W_padded=W_padded)

CPU golden reference (im2col pre-computed on CPU):
    registry.set_backend(Backend.CPU)
    golden = ops.conv2d(X_col, W_2d)

fp_preload requirements:
    f0 = 0.0  (hardware constant, always zero)
    f1 = 1.0  (required to build basis vectors for im2col extraction)
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
    unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
    col = unfold(input_4d.float())  # [B, C*K*K, OH*OW]
    col = col.permute(0, 2, 1).reshape(-1, input_4d.shape[1] * kernel_size * kernel_size)
    return col


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style Conv2d Test  (TRUE on-chip im2col, no V_SHFT_V)")
    print("=" * 80)

    # ========================================================================
    # Parameters — chosen so all tile dimensions = mlen = 64
    #   K_col = C_in * K * K = 4 * 4 * 4 = 64 = mlen  ✓
    #   OH    = H - K + 1    = 67 - 4 + 1 = 64 = mlen  ✓
    #   OW    = W - K + 1    = 4  - 4 + 1 = 1           (OW=1 ensures ow=0 always)
    #   M     = OH * OW      = 64             = mlen  ✓
    #   N     = C_out        = 64             = mlen  ✓
    #   W_padded = 64  (pad input rows to 64 for H_PREFETCH_V 64-element alignment)
    # ========================================================================
    B = 1
    C_in = 4
    H = 67
    W = 4
    K_size = 4
    C_out = 64
    stride = 1
    padding = 0
    W_padded = 64  # must be multiple of 64 for HBM alignment
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)  # MX8 block format

    OH = (H - K_size + 2 * padding) // stride + 1  # 64
    OW = (W - K_size + 2 * padding) // stride + 1  # 1
    M = B * OH * OW  # 64
    K_col = C_in * K_size * K_size  # 64
    N = C_out  # 64

    torch.manual_seed(42)

    # ========================================================================
    # Raw tensors
    # ========================================================================
    input_4d = torch.randn(B, C_in, H, W)
    weight_4d = torch.randn(C_out, C_in, K_size, K_size)

    print(f"\nInput:  {input_4d.shape}")
    print(f"Weight: {weight_4d.shape}")
    print(f"OH={OH}, OW={OW}, M={M}, K_col={K_col}, N={N}, W_padded={W_padded}")

    # ========================================================================
    # CPU im2col (for golden reference only — PLENA does im2col on-chip)
    # ========================================================================
    X_col = im2col(input_4d, K_size, stride=stride, padding=padding)  # [M, K_col]
    W_2d = weight_4d.float().reshape(C_out, -1).T.contiguous()  # [K_col, C_out]

    print(f"\nim2col X_col: {X_col.shape}")
    print(f"W_2d:         {W_2d.shape}")

    # ========================================================================
    # Raw input arranged for HBM: (C_in*H, W_padded)
    #   Row c*H + h  holds input[0, c, h, 0..W-1] at columns 0..W-1,
    #   columns W..W_padded-1 are zero-padded (never accessed by the im2col ASM
    #   when OW=1 and ow=0 always).
    # ========================================================================
    raw_input = torch.zeros(C_in * H, W_padded)
    for c in range(C_in):
        raw_input[c * H : (c + 1) * H, :W] = input_4d[0, c, :, :]

    print(f"raw_input (HBM layout): {raw_input.shape}")

    # ========================================================================
    # Load ATen-style operator registry
    # ========================================================================
    registry = OpRegistry.load()
    print(f"\nLoaded ops: {registry.list_ops()}")

    # ========================================================================
    # CPU golden reference (im2col on CPU, then matmul)
    # ========================================================================
    print("\n--- CPU Golden Reference ---")
    registry.set_backend(Backend.CPU)
    golden_Y = ops.conv2d(X_col, W_2d)  # [M, N]
    print(f"  golden_Y: {golden_Y.shape}")
    print(f"  golden_Y[0,:4]: {golden_Y[0, :4].tolist()}")

    # Cross-check against F.conv2d
    ref_conv = F.conv2d(input_4d, weight_4d, stride=stride, padding=padding)
    ref_2d = ref_conv.permute(0, 2, 3, 1).reshape(M, N)
    max_diff = (golden_Y - ref_2d).abs().max().item()
    print(f"  Max diff vs F.conv2d: {max_diff:.2e}  (should be ~0)")

    # ========================================================================
    # PLENA backend — TRUE on-chip im2col (no V_SHFT_V) + systolic matmul
    # ========================================================================
    print("\n--- PLENA Backend (TRUE on-chip im2col, no V_SHFT_V) ---")
    registry.set_backend(Backend.PLENA)

    # fp_preload:
    #   f0 = 0.0  (hardware constant zero; S_ADD_FP f{ex},f0,f0 zeros V_RED_SUM accum)
    #   f1 = 1.0  (used to build K basis vectors e_0..e_{K-1} in VRAM)
    #   f2 = 0.0  (initial value of V_RED_SUM accumulator — reset before each call)
    fp_preload = [0.0, 1.0, 0.0] + [0.0] * 7

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # HBM inputs:
    #   input_raw_var — raw NCHW input, shape (C_in*H, W_padded)
    #   w_2d_var      — weight matrix,  shape (K_col, N)
    input_raw_var = prog.input("input_raw", shape=(C_in * H, W_padded))
    w_2d_var = prog.input("W_2d", shape=(K_col, N))

    import os
    use_shift = os.environ.get("CONV_USE_SHIFT", "0") == "1"
    print(f"\n[CONV_USE_SHIFT={use_shift}]\n")
    Y = ops.conv2d(
        prog,
        input_raw_var,
        w_2d_var,
        C_in=C_in,
        H=H,
        W=W,
        K=K_size,
        OH=OH,
        OW=OW,
        M=M,
        W_padded=W_padded,
        use_shift=use_shift,
    )

    # Compile to ISA
    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "input_raw": raw_input.float(),
        "W_2d": W_2d,
    }
    golden_result = {"original_output": golden_Y}

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="linear_aten",
        data=None,
        specified_data_order=["input_raw", "W_2d"],
        build_path=build_dir,
    )

    y_vram_addr = prog._compiler.get_vram_addr(Y.name)

    comparison_params = {
        "start_row_idx": y_vram_addr // mlen,
        "num_rows": (M * N) // mlen,
        "num_batches": M,
        "elements_per_batch": N,
        "row_dim": mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Y location: VRAM row {y_vram_addr // mlen}")
    run_and_assert(build_dir, "conv2d", mlen=mlen, blen=blen)
