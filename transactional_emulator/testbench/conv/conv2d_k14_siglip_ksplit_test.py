"""
Real SigLIP Conv2d Test — C_in=3 (RGB), K=14, K_col=588 (10 MRAM tiles, K-split)

Uses real SigLIP patch-embedding parameters: C_in=3 RGB channels, 14×14 kernel,
giving K_col = 3*14*14 = 588 = 10 tiles of 64.

K-split feature: K_col=588 exceeds the 4-tile MRAM limit (MAX_K_TILES=4, max K_col=256).
  10 tiles are split into 3 chunks: [0..3], [4..7], [8..9]  → sizes [4, 4, 2]
  Each chunk produces a partial output (M, C_out); partial sums are accumulated
  via VRAM vector add, yielding the final result.

Hardware constraints:
  - MRAM capacity: 4 × mlen² = 16384 elements (4 tiles of 64×64)
  - Max K_col per chunk: 4 × mlen = 256
  - K_col=588 needs 10 tiles → ceil(10/4) = 3 chunks of [4,4,2] tiles

Geometry (M = OH*OW = 64 = mlen):
  H=77, W=14  →  OH = 77-14+1 = 64, OW = 14-14+1 = 1  →  M = 64
  W_padded=64  (next multiple of 64 >= W=14)

HBM alignment:
  hbm_offset = (c*H + oh+kr) * W_padded + ow
  With W_padded=64, ow=0: always 64-element aligned ✓
"""

import sys
import json
from pathlib import Path


import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena_compiler import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.model_layer_test_builder import quantize_to_mxfp


def im2col_cpu(input_4d, kernel_size):
    """CPU reference im2col: (B,C,H,W) → (B*OH*OW, C*K*K)."""
    unfold = torch.nn.Unfold(kernel_size=kernel_size)
    col = unfold(input_4d.float())
    return col.permute(0, 2, 1).reshape(-1, input_4d.shape[1] * kernel_size * kernel_size)


if __name__ == "__main__":
    print("=" * 80)
    print("Real SigLIP Conv2d Test  (C_in=3 RGB, K=14, K_col=588, 10 tiles, K-split)")
    print("  MRAM limit: 4 tiles × 64² = 16384 → max K_col per chunk = 256")
    print("  K_col=588 = 10 tiles → 3 chunks: [0..3], [4..7], [8..9]")
    print("=" * 80)

    # K_col = C_in * K * K = 3 * 14 * 14 = 588 = 10 tiles of 64
    # M = OH * OW = 64 = mlen (one output row block)
    B = 1
    C_in = 3  # real SigLIP RGB input channels
    H = 77  # H - K + 1 = 77 - 14 + 1 = 64 = mlen → OH = 64
    W = 14  # W - K + 1 = 14 - 14 + 1 = 1          → OW = 1
    K_size = 14
    C_out = 64
    stride = 1
    padding = 0
    W_padded = 64  # next multiple of 64 >= W=14
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)  # MXFP8: 9 bytes per 8 data bytes

    OH = (H - K_size + 2 * padding) // stride + 1  # 64
    OW = (W - K_size + 2 * padding) // stride + 1  # 1
    M = B * OH * OW  # 64
    K_col = C_in * K_size * K_size  # 588
    K_col_padded = ((K_col + mlen - 1) // mlen) * mlen  # 640 (next multiple of 64)

    num_tiles = (K_col + mlen - 1) // mlen  # 10

    print(f"\nC_in={C_in} (RGB), K={K_size}")
    print(f"K_col={K_col}  →  {num_tiles} tiles of {mlen}  (MRAM: needs K-split, MAX_K_TILES=4)")
    print(f"H={H}, W={W}, OH={OH}, OW={OW}, M={M}, C_out={C_out}, W_padded={W_padded}")

    torch.manual_seed(42)

    input_4d = torch.randn(B, C_in, H, W)
    weight_4d = torch.randn(C_out, C_in, K_size, K_size)

    print(f"\nInput:  {input_4d.shape}")
    print(f"Weight: {weight_4d.shape}")

    # CPU golden
    X_col = im2col_cpu(input_4d, K_size)
    W_2d = weight_4d.float().reshape(C_out, -1).T.contiguous()  # (K_col, C_out)

    print(f"\nim2col:    {X_col.shape}")
    print(f"weight_2d: {W_2d.shape}")

    W_2d_q = quantize_to_mxfp(W_2d)

    registry = OpRegistry.load()
    registry.set_backend(Backend.CPU)
    golden_Y = ops.conv2d(X_col, W_2d_q)

    print(f"golden:    {golden_Y.shape}")
    print(f"golden[0,:4]: {golden_Y[0, :4].tolist()}")

    # HBM layout for raw input: (C_in*H, W_padded)
    raw_input = torch.zeros(C_in * H, W_padded)
    for c in range(C_in):
        raw_input[c * H : (c + 1) * H, :W] = input_4d[0, c, :, :]

    print(f"raw_input (HBM layout): {raw_input.shape}")

    # PLENA backend
    print(f"\n--- PLENA Backend (tiled im2col + K-split, K_col={K_col}, {num_tiles} tiles) ---")
    registry.set_backend(Backend.PLENA)

    # fp_preload: slot 1 = 1.0  (fp_one_reg=1, default for conv2d_plena)
    fp_preload = [0.0, 1.0, 0.0] + [0.0] * 7

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Zero-pad W_2d to K_col_padded rows so the last MRAM tile is clean (no HBM OOB reads)
    W_2d_padded = torch.zeros(K_col_padded, C_out)
    W_2d_padded[:K_col] = W_2d

    input_raw_var = prog.input("input_raw", shape=(C_in * H, W_padded))
    w_2d_var = prog.input("W_2d", shape=(K_col_padded, C_out))

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
    )

    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA code")

    build_dir = Path(__file__).parent / "build" / "conv2d_siglip_real_k14"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {"input_raw": raw_input.float(), "W_2d": W_2d_padded}
    golden_result = {"original_output": golden_Y}

    create_sim_env(
        input_tensor,
        gen_code,
        golden_result,
        fp_preload,
        build_dir=str(build_dir),
    )

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
        "num_rows": (M * C_out) // mlen,
        "num_batches": M,
        "elements_per_batch": C_out,
        "row_dim": mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Y location: VRAM row {y_vram_addr // mlen}")
    run_and_assert(build_dir, "conv2d_siglip_real_k14", mlen=mlen, blen=blen)
