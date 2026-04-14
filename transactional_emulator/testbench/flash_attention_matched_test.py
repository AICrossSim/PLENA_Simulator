"""
Matched flash-attention test for ATen path.
Dims chosen for FLOP-parity with main's flashattn_prefill_test.py:
  Main GQA: hq=4, hkv=1, h_qkv=16, s_q=s_kv=64 → 4 * 2*64*64*16 = 524,288 FLOPs
  ATen MHA: seq=64, head_dim=64 → 2*64*64*64 = 524,288 FLOPs ✓
"""

import sys
import math
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops
from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


if __name__ == "__main__":
    print("=" * 80)
    print("Matched Flash Attention Test — FLOP-equivalent to main prefill")
    print("=" * 80)

    seq_len = 64
    head_dim = 64
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)

    Q = torch.randn(seq_len, head_dim) * 0.5
    K = torch.randn(seq_len, head_dim) * 0.5
    V = torch.randn(seq_len, head_dim) * 0.5
    print(f"Q/K/V shape: {Q.shape}, scale: {scale:.6f}")

    registry = OpRegistry.load()
    registry.set_backend(Backend.CPU)
    golden_O = ops.flash_attention(Q, K, V, scale)

    registry.set_backend(Backend.PLENA)
    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)
    q_input = prog.input("Q", shape=(seq_len, head_dim))
    k_input = prog.input("K", shape=(seq_len, head_dim))
    v_input = prog.input("V", shape=(seq_len, head_dim))
    Q_batch = prog.load_batch(q_input, name="Q")
    O = ops.flash_attention(prog, Q_batch, k_input, v_input, scale)

    gen_code = prog.compile()
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "Q": Q.reshape(1, -1),
        "K": K.reshape(1, -1),
        "V": V.reshape(1, -1),
    }
    golden_result = {"original_output": golden_O}
    fp_preload = [0.0, scale, float("-inf")] + [0.0] * 7

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_matched",
        data=None,
        specified_data_order=["Q", "K", "V"],
        build_path=build_dir,
    )

    o_vram_addr = prog._compiler.get_vram_addr(O.name)
    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (seq_len * head_dim) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": head_dim,
        "row_dim": mlen,
        "use_stride_mode": True,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)
    print(f"\nMatched flash-attention: ASM = {len(gen_code.splitlines())} lines")
    run_and_assert(build_dir, "flash_attention_matched", mlen=mlen, blen=blen)
