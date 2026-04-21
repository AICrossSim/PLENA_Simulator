"""
Tiled im2col Conv2d Test — K_col > VLEN=64

Verifies that conv2d_plena correctly handles K_col > 64 by tiling the
column dimension of the im2col matrix.

Parameters chosen so K_col = 2 * mlen = 128 (two tiles of 64):
  C_in=2, K=8  →  K_col = 2*8*8 = 128 = 2 tiles
  H=8, W=8, OH=1, OW=1, M=1   (H=K → OH=1; OW=1 ensures ow=0 always)
  C_out=64, W_padded=64

HBM alignment:
  With W_padded=64 and ow=0: offset = (c*H+oh+kr)*64 — always 64-element aligned.
"""

import json
from pathlib import Path


import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena_compiler import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.emulator_runner import run_and_assert


def im2col_cpu(input_4d, kernel_size):
    """CPU reference im2col: (B,C,H,W) → (B*OH*OW, C*K*K)."""
    unfold = torch.nn.Unfold(kernel_size=kernel_size)
    col = unfold(input_4d.float())
    return col.permute(0, 2, 1).reshape(-1, input_4d.shape[1] * kernel_size * kernel_size)


if __name__ == "__main__":
    print("=" * 80)
    print("Tiled im2col Conv2d Test  (K_col=128, 2 tiles)")
    print("=" * 80)

    # K_col = C_in * K * K = 2 * 8 * 8 = 128 = 2 * mlen  (2 tiles)
    # M = OH * OW = 64 = mlen  (required: batch must be multiple of mlen)
    B = 1
    C_in = 2
    H = 71  # H - K + 1 = 71 - 8 + 1 = 64 = mlen → OH = 64
    W = 8  # W - K + 1 = 8 - 8 + 1 = 1         → OW = 1
    K_size = 8
    C_out = 64
    stride = 1
    padding = 0
    W_padded = 64
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    OH = (H - K_size + 2 * padding) // stride + 1  # 1
    OW = (W - K_size + 2 * padding) // stride + 1  # 1
    M = B * OH * OW  # 1
    K_col = C_in * K_size * K_size  # 128
    N = C_out  # 64

    print(f"\nC_in={C_in}, K={K_size}, K_col={K_col} ({K_col // mlen} tiles of {mlen})")
    print(f"H={H}, W={W}, OH={OH}, OW={OW}, M={M}, C_out={C_out}, W_padded={W_padded}")

    torch.manual_seed(42)

    input_4d = torch.randn(B, C_in, H, W)
    weight_4d = torch.randn(C_out, C_in, K_size, K_size)

    print(f"\nInput:  {input_4d.shape}")
    print(f"Weight: {weight_4d.shape}")

    # CPU golden
    X_col = im2col_cpu(input_4d, K_size)
    W_2d = weight_4d.float().reshape(C_out, -1).T.contiguous()

    print(f"\nim2col:    {X_col.shape}")
    print(f"weight_2d: {W_2d.shape}")

    registry = OpRegistry.load()
    registry.set_backend(Backend.CPU)
    golden_Y = ops.conv2d(X_col, W_2d)

    print(f"golden:    {golden_Y.shape}")
    print(f"golden[0,:4]: {golden_Y[0, :4].tolist()}")

    # HBM layout for raw input
    raw_input = torch.zeros(C_in * H, W_padded)
    for c in range(C_in):
        raw_input[c * H : (c + 1) * H, :W] = input_4d[0, c, :, :]

    print(f"raw_input (HBM layout): {raw_input.shape}")

    # PLENA backend
    print("\n--- PLENA Backend (tiled im2col, K_col=128 > VLEN=64) ---")
    registry.set_backend(Backend.PLENA)

    # fp_preload: slot 1 = 1.0  (fp_one_reg=1, default for conv2d_plena)
    fp_preload = [0.0, 1.0, 0.0] + [0.0] * 7

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    input_raw_var = prog.input("input_raw", shape=(C_in * H, W_padded))
    w_2d_var = prog.input("W_2d", shape=(K_col, N))

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

    build_dir = Path(__file__).parent / "build" / "conv2d_tiled_im2col"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {"input_raw": raw_input.float(), "W_2d": W_2d}
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
    run_and_assert(build_dir, "conv2d_tiled_im2col", mlen=mlen, blen=blen)
