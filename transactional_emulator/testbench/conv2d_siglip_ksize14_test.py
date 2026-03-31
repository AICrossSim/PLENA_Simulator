"""
SigLIP-Inspired Conv2d Test — C_in=3 (RGB), K=8, K_col=192 (3 tiles of 64)

Uses real SigLIP input-channel count (C_in=3 = RGB) with K=8 kernel,
giving K_col = 3*8*8 = 192 = 3 tiles of 64.

Hardware constraints:
  - MRAM capacity: 4 × mlen² = 16384 elements (4 tiles of 64×64)
  - Max K_col for single-pass matmul: 4 × mlen = 256
  - K_col=192 uses 3 MRAM tiles (3×4096=12288 ≤ 16384) ✓
  - K_col=588 (real SigLIP 3×14×14) would need 10 MRAM tiles → overflow

Real SigLIP uses C_in=3, K=14, K_col=588 which exceeds the 4-tile MRAM limit.
Supporting K_col > 256 would require a K-split linear_plena (partial sums via
VRAM vector add), which is a separate architectural task.

Geometry (M = OH*OW = 64 = mlen):
  H=71, W=8  →  OH = 71-8+1 = 64, OW = 8-8+1 = 1  →  M = 64
  W_padded=64  (next multiple of 64 >= W=8)

HBM alignment:
  hbm_offset = (c*H + oh+kr) * W_padded + ow
  With W_padded=64, ow=0: always 64-element aligned ✓
"""

import sys
import json
from pathlib import Path

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "tools"))

import torch

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops

from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


def im2col_cpu(input_4d, kernel_size):
    """CPU reference im2col: (B,C,H,W) → (B*OH*OW, C*K*K)."""
    unfold = torch.nn.Unfold(kernel_size=kernel_size)
    col = unfold(input_4d.float())
    return col.permute(0, 2, 1).reshape(-1, input_4d.shape[1] * kernel_size * kernel_size)


if __name__ == "__main__":
    print("=" * 80)
    print("SigLIP-Inspired Conv2d Test  (C_in=3 RGB, K=8, K_col=192, 3 tiles)")
    print("  MRAM limit: 4 tiles × 64² = 16384 → max K_col=256")
    print("  K_col=192 = 3 tiles (3×4096=12288 ≤ 16384) ✓")
    print("=" * 80)

    # K_col = C_in * K * K = 3 * 8 * 8 = 192 = 3 tiles of 64
    # M = OH * OW = 64 = mlen (one output row block)
    B = 1
    C_in = 3  # real SigLIP RGB input channels
    H = 71  # H - K + 1 = 71 - 8 + 1 = 64 = mlen → OH = 64
    W = 8  # W - K + 1 = 8 - 8 + 1 = 1          → OW = 1
    K_size = 8
    C_out = 64
    stride = 1
    padding = 0
    W_padded = 64  # next multiple of 64 >= W=8
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)  # MXFP8: 9 bytes per 8 data bytes

    OH = (H - K_size + 2 * padding) // stride + 1  # 64
    OW = (W - K_size + 2 * padding) // stride + 1  # 1
    M = B * OH * OW  # 64
    K_col = C_in * K_size * K_size  # 192

    num_tiles = (K_col + mlen - 1) // mlen  # 3

    print(f"\nC_in={C_in} (RGB), K={K_size}")
    print(
        f"K_col={K_col}  →  {num_tiles} tiles of {mlen}  "
        f"(MRAM: {num_tiles}×{mlen * mlen}={num_tiles * mlen * mlen} ≤ 16384)"
    )
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

    registry = OpRegistry.load()
    registry.set_backend(Backend.CPU)
    golden_Y = ops.conv2d(X_col, W_2d)

    print(f"golden:    {golden_Y.shape}")
    print(f"golden[0,:4]: {golden_Y[0, :4].tolist()}")

    # HBM layout for raw input: (C_in*H, W_padded)
    raw_input = torch.zeros(C_in * H, W_padded)
    for c in range(C_in):
        raw_input[c * H : (c + 1) * H, :W] = input_4d[0, c, :, :]

    print(f"raw_input (HBM layout): {raw_input.shape}")

    # PLENA backend
    print(f"\n--- PLENA Backend (tiled im2col, K_col={K_col}, {num_tiles} tiles) ---")
    registry.set_backend(Backend.PLENA)

    # fp_preload: slot 1 = 1.0  (fp_one_reg=1, default for conv2d_plena)
    fp_preload = [0.0, 1.0, 0.0] + [0.0] * 7

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    input_raw_var = prog.input("input_raw", shape=(C_in * H, W_padded))
    w_2d_var = prog.input("W_2d", shape=(K_col, C_out))

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

    build_dir = Path(__file__).parent / "build"
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
    run_and_assert(build_dir, "conv2d_siglip_cin3_k8", mlen=mlen, blen=blen)
