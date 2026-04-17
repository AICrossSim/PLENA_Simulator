"""
GQA flash attention via the proper ATen dispatch.

Uses `ops.flash_attention(prog, Q, K, V, scale, hq=4, hkv=1, h_qkv=16)` —
dispatches through the registry to `flash_attention_plena`, which detects
GQA and emits fused codegen using main's `flash_attn_asm` template.

Dims match main's prefill: batch=1, s_q=s_kv=64, hq=4, hkv=1, h_qkv=16.
"""

import math
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops
from compiler.aten.plena_compiler import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


def gqa_sdpa(q, k, v, scale, hq, hkv):
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2).repeat_interleave(hq // hkv, dim=1)
    v_t = v.transpose(1, 2).repeat_interleave(hq // hkv, dim=1)
    o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
    return o.transpose(1, 2)


if __name__ == "__main__":
    print("=" * 80)
    print("GQA Flash Attention via ATen dispatch (ops.flash_attention)")
    print("=" * 80)

    batch_size = 1
    s_q = 64
    s_kv = 64
    hq = 4
    hkv = 1
    h_qkv = 16
    hidden_size = hq * h_qkv  # 64 = mlen
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    scale = 1.0 / math.sqrt(h_qkv)

    torch.manual_seed(42)
    q = torch.randn(batch_size, s_q, hq, h_qkv) * 0.5
    k = torch.randn(batch_size, s_kv, hkv, h_qkv) * 0.5
    v = torch.randn(batch_size, s_kv, hkv, h_qkv) * 0.5

    # Pad KV to mlen-wide for main-template compatibility (hkv=1 → 4 slots, 3 zero)
    k_padded = torch.zeros(batch_size, s_kv, mlen // h_qkv, h_qkv)
    v_padded = torch.zeros(batch_size, s_kv, mlen // h_qkv, h_qkv)
    k_padded[:, :, :hkv, :] = k
    v_padded[:, :, :hkv, :] = v

    # Golden via SDPA
    golden = gqa_sdpa(q.float(), k.float(), v.float(), scale, hq, hkv)

    # PLENA program using proper ATen dispatch
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)
    # Q is prestaged at VRAM addr=0 by the test harness (matches main's prefill
    # test which also preloads Q to VRAM row 0 via preload_act_asm).
    q_input = prog.input("Q", shape=(s_q, hidden_size), prestaged_vram_addr=0)
    k_input = prog.input("K", shape=(s_kv, mlen))  # padded to mlen
    v_input = prog.input("V", shape=(s_kv, mlen))
    Q_batch = prog.load_batch(q_input, name="Q")  # no ISA emitted (prestaged)

    # Dispatch through ops.flash_attention with GQA params
    O = ops.flash_attention(prog, Q_batch, k_input, v_input, scale, hq=hq, hkv=hkv, h_qkv=h_qkv)

    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA")

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "Q": q.reshape(1, -1),
        "K": k_padded.reshape(1, -1),
        "V": v_padded.reshape(1, -1),
    }
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": golden.reshape(s_q, hidden_size),
    }

    fp_preload = [0.0, scale, float("-inf")] + [0.0] * 45
    # Q is prestaged in VRAM at addr=0: provide flat fp16 VRAM image starting
    # with Q's elements (row-major, hidden_size=64 elements per row).
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
        asm="flash_attention_gqa_aten",
        data=None,
        specified_data_order=["Q", "K", "V"],
        build_path=build_dir,
    )

    o_vram_addr = prog._compiler.get_vram_addr(O.name)
    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (s_q * hidden_size) // mlen,
        "num_batches": s_q,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": False,
        "use_slice_mode": True,
        "slice_per_row": h_qkv,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nOutput at VRAM row {o_vram_addr // mlen}")
    run_and_assert(build_dir, "flash_attention_gqa_aten", mlen=mlen, blen=blen)
