"""Configurable ATen-style Conv2D test — on-chip im2col + PLENA systolic matmul.

Presets:
  baseline   C_in=4, K=4,  K_col=64   (1 tile)
  tiled      C_in=2, K=8,  K_col=128  (2 tiles)
  siglip     C_in=3, K=8,  K_col=192  (3 tiles, RGB-like)
  ksplit     C_in=3, K=14, K_col=588  (10 tiles, K-split + MXFP8 quant)
"""

from pathlib import Path
import argparse
import json
import math
import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.emulator_runner import run_and_assert


PRESETS = {
    "baseline": dict(C_in=4, K=4, C_out=64, H=67, W=4),
    "tiled": dict(C_in=2, K=8, C_out=64, H=71, W=8),
    "siglip": dict(C_in=3, K=8, C_out=64, H=71, W=8),
    "ksplit": dict(C_in=3, K=14, C_out=64, H=77, W=14, quantize=True),
}

MLEN, BLEN, MAX_K_TILES = 64, 4, 4
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)


def im2col_cpu(x_4d, kernel_size):
    u = torch.nn.Unfold(kernel_size=kernel_size)
    return u(x_4d.float()).permute(0, 2, 1).reshape(-1, x_4d.shape[1] * kernel_size * kernel_size)


def _raw_input(x_4d, c_in, h, w_padded):
    raw = torch.zeros(c_in * h, w_padded)
    for c in range(c_in):
        raw[c * h : (c + 1) * h, : x_4d.shape[3]] = x_4d[0, c, :, :]
    return raw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=list(PRESETS), default="baseline")
    parser.add_argument("--cin", type=int, default=None)
    parser.add_argument("--kernel", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--cout", type=int, default=64)
    parser.add_argument("--mlen", type=int, default=MLEN)
    parser.add_argument("--blen", type=int, default=BLEN)
    parser.add_argument("--quantize", action="store_true", default=None)
    parser.add_argument("--build-dir", type=Path, default=None)
    args = parser.parse_args()

    p = PRESETS[args.preset].copy()
    for k in ["cin", "kernel", "height", "width", "cout"]:
        v = getattr(args, k, None)
        if v is not None:
            p[k[0].upper() + k[1:]] = v
    if args.quantize is not None:
        p["quantize"] = args.quantize

    C_in, K, C_out, H, W = p["C_in"], p["K"], p["C_out"], p["H"], p["W"]
    do_quantize = p.get("quantize", False)
    mlen, blen = args.mlen, args.blen
    W_padded = ((W + mlen - 1) // mlen) * mlen
    OH = H - K + 1
    OW = W - K + 1
    M = OH * OW
    K_col = C_in * K * K
    N = C_out

    label = f"Conv2D C_in={C_in} K={K} K_col={K_col}"
    if do_quantize:
        label += " K-split+MXFP8"
    print(f"{'=' * 80}\n{label}\n{'=' * 80}")

    torch.manual_seed(42)
    x_4d = torch.randn(1, C_in, H, W)
    w_4d = torch.randn(C_out, C_in, K, K)

    X_col = im2col_cpu(x_4d, K)
    W_2d = w_4d.float().reshape(C_out, -1).T.contiguous()
    raw_input = _raw_input(x_4d, C_in, H, W_padded)

    print(f"OH={OH}, OW={OW}, M={M}, K_col={K_col}, N={N}, W_padded={W_padded}")

    registry = OpRegistry.load()
    registry.set_backend(Backend.CPU)
    golden = ops.conv2d(X_col, W_2d)
    print(f"  golden: {golden.shape}  golden[0,:4]: {golden[0, :4].tolist()}")

    registry.set_backend(Backend.PLENA)
    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=REAL_DATA_RATIO)
    fp_preload = [0.0, 1.0, 0.0] + [0.0] * 7

    if not do_quantize:
        # Simple: single conv2d call
        raw_var = prog.input("input_raw", shape=(C_in * H, W_padded))
        w_var = prog.input("W_2d", shape=(K_col, N))
        Y = ops.conv2d(prog, raw_var, w_var, C_in=C_in, H=H, W=W, K=K, OH=OH, OW=OW, M=M, W_padded=W_padded)
    else:
        # K-split: chunk K_col into MAX_K_TILES-tile chunks, accumulate partial outputs
        from transactional_emulator.testbench.sliced_layer_test_builder import quantize_to_mxfp

        raw_var = prog.input("input_raw", shape=(C_in * H, W_padded))
        K_quant = quantize_to_mxfp(W_2d, None)
        chunk_size = MAX_K_TILES * mlen
        n_chunks = math.ceil(K_col / chunk_size)

        partial_outputs = []
        for ci in range(n_chunks):
            k_start = ci * chunk_size
            k_end = min(k_start + chunk_size, K_col)
            k_chunk = k_end - k_start
            w_chunk = K_quant[:, k_start:k_end].contiguous()
            w_var = prog.input(f"W_2d_c{ci}", shape=(k_chunk, N))
            Y_part = ops.conv2d(
                prog, raw_var, w_var, C_in=C_in, H=H, W=W, K=K, OH=OH, OW=OW, M=M, W_padded=W_padded, k_start=k_start
            )
            partial_outputs.append(Y_part)

        # Accumulate partials with vector add
        if len(partial_outputs) > 1:
            acc = partial_outputs[0]
            for part in partial_outputs[1:]:
                isa = prog._compiler
                gp = isa.register_allocator.allocate_gp(3)
                isa.generated_code += (
                    f"; Vector-add partial output {part.name} into {acc.name}\n"
                    f"S_ADDI_INT gp{gp[0]}, gp0, {isa.get_vram_addr(acc.name)}\n"
                    f"S_ADDI_INT gp{gp[1]}, gp0, {isa.get_vram_addr(part.name)}\n"
                    f"S_ADDI_INT gp{gp[2]}, gp0, {M * N}\n"
                    f"V_ADD_VV gp{gp[0]}, gp{gp[1]}, gp{gp[2]}\n"
                )
                isa.register_allocator.free_gp(gp)
            Y = acc
        else:
            Y = partial_outputs[0]

    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA code")

    build_dir = args.build_dir or (Path(__file__).parent / "build" / f"conv2d_{args.preset}")
    build_dir.mkdir(parents=True, exist_ok=True)

    y_addr = prog._compiler.get_vram_addr(Y.name)
    comparison_params = {
        "start_row_idx": y_addr // mlen,
        "num_rows": (M * N) // mlen,
        "num_batches": M,
        "elements_per_batch": N,
        "row_dim": mlen,
    }

    input_tensor = {"input_raw": raw_input.float()}
    if not do_quantize:
        input_tensor["W_2d"] = W_2d
    else:
        for ci in range(n_chunks):
            k_start = ci * chunk_size
            k_end = min(k_start + chunk_size, K_col)
            input_tensor[f"W_2d_c{ci}"] = K_quant[:, k_start:k_end].contiguous()

    create_sim_env(input_tensor, gen_code, {"original_output": golden}, fp_preload, build_dir=str(build_dir))
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="linear_aten",
        data=None,
        specified_data_order=list(input_tensor.keys()),
        build_path=build_dir,
    )

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    run_and_assert(build_dir, f"conv2d_{args.preset}", mlen=mlen, blen=blen)
