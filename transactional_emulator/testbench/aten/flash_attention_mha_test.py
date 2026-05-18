"""
ATen per-head MHA flash attention golden test.

This covers the ATen MHA primitive path, not the fused GQA direct_emit template.
It verifies the generated ISA against PyTorch scaled_dot_product_attention in
the transactional emulator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler.aten.ops.registry import Backend, OpRegistry
import compiler.aten.ops as ops
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim


def mha_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    q_t = q.unsqueeze(0).unsqueeze(0)
    k_t = k.unsqueeze(0).unsqueeze(0)
    v_t = v.unsqueeze(0).unsqueeze(0)
    o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
    return o.squeeze(0).squeeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--unroll-attention", action="store_true")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "flash_attention_mha",
    )
    args = parser.parse_args()

    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    scale = 1.0 / math.sqrt(args.head_dim)

    if args.seq_len % mlen != 0:
        raise ValueError(f"seq_len={args.seq_len} must be a multiple of MLEN={mlen}")
    if args.head_dim % mlen != 0:
        raise ValueError(f"head_dim={args.head_dim} must be a multiple of MLEN={mlen}")

    build_dir = args.build_dir.resolve()

    torch.manual_seed(42)
    q = torch.randn(args.seq_len, args.head_dim) * 0.5
    k = torch.randn(args.seq_len, args.head_dim) * 0.5
    v = torch.randn(args.seq_len, args.head_dim) * 0.5
    golden = mha_sdpa(q.float(), k.float(), v.float(), scale)

    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)
    prog.unroll_attention = args.unroll_attention

    q_input = prog.input("Q", shape=(args.seq_len, args.head_dim), prestaged_vram_addr=0)
    k_input = prog.input("K", shape=(args.seq_len, args.head_dim))
    v_input = prog.input("V", shape=(args.seq_len, args.head_dim))
    Q = prog.load_batch(q_input, name="Q")

    O = ops.flash_attention(prog, Q, k_input, v_input, scale)
    gen_code = prog.compile()

    build_dir.mkdir(parents=True, exist_ok=True)
    input_tensor = {
        "Q": q.reshape(1, -1),
        "K": k.reshape(1, -1),
        "V": v.reshape(1, -1),
    }
    golden_result = {"original_output": golden.reshape(args.seq_len, args.head_dim)}
    fp_preload = [0.0, scale, float("-inf")] + [0.0] * 45
    q_vram_flat = q.reshape(-1).to(torch.float16)

    create_sim_env(
        input_tensor,
        gen_code,
        golden_result,
        fp_preload,
        build_dir=str(build_dir),
        vram_preload=q_vram_flat,
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_mha_aten",
        data=None,
        specified_data_order=["Q", "K", "V"],
        build_path=build_dir,
    )

    o_vram_addr = prog._compiler.get_vram_addr(O.name)
    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (args.seq_len * args.head_dim) // mlen,
        "num_batches": args.seq_len,
        "elements_per_batch": args.head_dim,
        "row_dim": mlen,
        "use_stride_mode": False,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"Generated {len(gen_code.splitlines())} lines of ISA")
    print(f"Output at VRAM row {o_vram_addr // mlen}")
    run_and_assert(build_dir, "flash_attention_mha_aten", mlen=mlen, blen=blen)


if __name__ == "__main__":
    main()
