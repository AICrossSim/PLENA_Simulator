"""
Naive GQA flash-attention on the ATen path: 4 separate flash_attention calls,
one per Q head, sharing the SAME K/V across heads (GQA 4:1).

This matches main's flashattn_prefill_test dims:
  hq=4, hkv=1, h_qkv=16, s_q=s_kv=64

The purpose is to quantify main's head-fusion benefit.  Main's GQA inner loop
packs 4 Q heads into the blen=4 systolic dimension so one M_MM produces 4
heads of output in parallel.  Naive 4-call ATen does them sequentially.
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


if __name__ == "__main__":
    print("=" * 80)
    print("Naive-GQA Flash Attention on ATen path (4 × single-head calls)")
    print("=" * 80)

    # Matched dims — main's flashattn_prefill_test
    batch_size = 1
    s_q = 64
    s_kv = 64
    num_q_heads = 4
    num_kv_heads = 1
    h_qkv = 16
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    scale = 1.0 / math.sqrt(h_qkv)
    torch.manual_seed(42)

    # Tensors in the same shape conventions main uses
    Q_full = torch.randn(s_q, num_q_heads, h_qkv) * 0.5
    K_full = torch.randn(s_kv, num_kv_heads, h_qkv) * 0.5
    V_full = torch.randn(s_kv, num_kv_heads, h_qkv) * 0.5

    # Register backend
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Shared K/V (kv head 0)
    K_slice = K_full[:, 0, :]  # (s_kv, h_qkv=16)
    V_slice = V_full[:, 0, :]
    # flash_attention_plena requires K/V of shape (seq_len, head_dim)
    # To fit mlen constraint, pad h_qkv=16 to mlen=64 (zero padding)
    K_padded = torch.zeros(s_kv, mlen)
    V_padded = torch.zeros(s_kv, mlen)
    K_padded[:, :h_qkv] = K_slice
    V_padded[:, :h_qkv] = V_slice

    k_input = prog.input("K", shape=(s_kv, mlen))
    v_input = prog.input("V", shape=(s_kv, mlen))

    outputs = []
    for q_head in range(num_q_heads):
        Q_slice = Q_full[:, q_head, :]  # (s_q, h_qkv=16)
        Q_padded = torch.zeros(s_q, mlen)
        Q_padded[:, :h_qkv] = Q_slice

        q_input = prog.input(f"Q{q_head}", shape=(s_q, mlen))
        Q_batch = prog.load_batch(q_input, name=f"Q{q_head}")
        O_head = ops.flash_attention(prog, Q_batch, k_input, v_input, scale)
        outputs.append(O_head)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nNaive-GQA ATen ASM: {len(lines)} lines")

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)
    print(f"  written to: {build_dir}/generated_asm_code.asm")
