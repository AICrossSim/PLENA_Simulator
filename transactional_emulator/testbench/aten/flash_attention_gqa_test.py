"""GQA flash attention via the proper ATen dispatch.

    python flash_attention_gqa_test.py [--mlen 128] [--blen 16] \
        [--batch-size 2] [--seq-len 64]

Uses `ops.flash_attention(prog, Q, K, V, scale, hq=4, hkv=1, h_qkv=16, ...)` --
dispatches through the registry to `flash_attention_plena`, which detects
GQA and emits fused codegen using main's `flash_attn_asm` template.

Dims follow the unified [batch_size, seq_len, heads, head_dim] interface:
Q is [batch_size, seq_len, hq, h_qkv] and K/V are [batch_size, seq_len, hkv,
h_qkv]. The packed GQA lowering supports one sequence tile, so seq_len and
kv_seq_len must each be <= MLEN; multiple batches are emitted as a per-batch
loop. With the defaults (batch_size=1, seq_len=mlen) this reproduces the
previous single-tile prefill test exactly.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.golden import quantize_to_mxfp
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw


def gqa_sdpa(q, k, v, scale, hq, hkv, *, causal: bool = False):
    # q: [batch, seq, hq, h_qkv]; k/v: [batch, seq, hkv, h_qkv]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2).repeat_interleave(hq // hkv, dim=1)
    v_t = v.transpose(1, 2).repeat_interleave(hq // hkv, dim=1)
    o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale, is_causal=causal)
    return o.transpose(1, 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--causal-mask", action="store_true", help="Apply causal mask in GQA attention.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "flash_attention_gqa",
        help="Directory for generated simulator artifacts.",
    )
    args = parser.parse_args()

    mlen = args.mlen
    blen = args.blen

    # GQA head counts are fixed (architectural constants, not hardware tile params)
    hq = 4
    hkv = 1
    h_qkv = mlen // hq  # per-head dim: scales with mlen (e.g. 16 for mlen=64, 32 for mlen=128)

    # Unified [batch_size, seq_len, heads, head_dim] interface.
    batch_size = args.batch_size if args.batch_size is not None else 1
    s_q = args.seq_len if args.seq_len is not None else mlen
    s_kv = s_q
    hidden_size = hq * h_qkv  # equals mlen

    if mlen % hq != 0:
        raise ValueError(f"MLEN ({mlen}) must be divisible by hq ({hq})")
    if mlen % blen != 0:
        raise ValueError(f"MLEN ({mlen}) must be divisible by BLEN ({blen})")
    if batch_size <= 0:
        raise ValueError(f"batch_size ({batch_size}) must be positive")
    # Packed GQA lowering supports exactly one sequence tile per batch.
    if s_q > mlen:
        raise ValueError(
            f"seq_len ({s_q}) exceeds the one-tile packed-GQA limit MLEN ({mlen}). "
            f"Supported range: 1 <= seq_len <= MLEN."
        )
    rows = batch_size * s_q
    if rows % blen != 0:
        raise ValueError(f"rows = batch_size*seq_len = {batch_size}*{s_q} = {rows} must be a multiple of BLEN ({blen})")

    # Per-batch physical rows must be a whole number of MLEN tiles (the GQA loop
    # advances Q/K/V/O bases by MLEN-aligned per-batch strides).
    rows_per_batch = max(mlen, s_q)
    if rows_per_batch % mlen != 0:
        rows_per_batch = ((rows_per_batch + mlen - 1) // mlen) * mlen

    scale = 1.0 / math.sqrt(h_qkv)

    args.hlen = h_qkv  # HLEN must equal per-head dim for packed attention

    build_dir = args.build_dir.expanduser().resolve()
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(
        f"GQA Flash Attention via ATen dispatch  (mlen={mlen}, blen={blen}, "
        f"batch={batch_size}, seq={s_q}, hq={hq}, hkv={hkv}, h_qkv={h_qkv})"
    )
    print("=" * 80)

    torch.manual_seed(args.seed)
    # [batch_size, seq_len, heads, head_dim]
    q = torch.randn(batch_size, s_q, hq, h_qkv) * 0.5
    k = torch.randn(batch_size, s_kv, hkv, h_qkv) * 0.5
    v = torch.randn(batch_size, s_kv, hkv, h_qkv) * 0.5

    # Pad KV heads to mlen-wide for main-template compatibility (hkv=1 -> 4 slots,
    # 3 zero) and pad each batch's sequence rows up to rows_per_batch tiles.
    kv_head_slots = mlen // h_qkv
    k_padded = torch.zeros(batch_size, rows_per_batch, kv_head_slots, h_qkv)
    v_padded = torch.zeros(batch_size, rows_per_batch, kv_head_slots, h_qkv)
    k_padded[:, :s_kv, :hkv, :] = k
    v_padded[:, :s_kv, :hkv, :] = v

    # Hardware-accurate golden: MXFP8-quantize K, V before GQA SDPA.
    k_q = quantize_to_mxfp(k)
    v_q = quantize_to_mxfp(v)
    golden = gqa_sdpa(q.float(), k_q.float(), v_q.float(), scale, hq, hkv, causal=args.causal_mask)

    # PLENA program using proper ATen dispatch
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)

    # Q is prestaged at VRAM addr=0 by the test harness (matches main's prefill
    # test which also preloads Q to VRAM row 0 via preload_act_asm). Each batch
    # occupies a rows_per_batch-tall physical block; the logical Q is
    # [batch_size*seq_len, hidden_size].
    q_phys_rows = batch_size * rows_per_batch
    kv_phys_rows = batch_size * rows_per_batch
    q_input = prog.input(
        "Q",
        shape=(rows, hidden_size),
        physical_shape=(q_phys_rows, mlen),
        prestaged_vram_addr=0,
    )
    k_input = prog.input(
        "K",
        shape=(batch_size * s_kv, mlen),
        physical_shape=(kv_phys_rows, mlen),
    )
    v_input = prog.input(
        "V",
        shape=(batch_size * s_kv, mlen),
        physical_shape=(kv_phys_rows, mlen),
    )
    Q_batch = prog.load_batch(q_input, name="Q")  # no ISA emitted (prestaged)

    # Dispatch through ops.flash_attention with GQA + batch/seq params.
    O = ops.flash_attention(
        prog,
        Q_batch,
        k_input,
        v_input,
        scale,
        hq=hq,
        hkv=hkv,
        h_qkv=h_qkv,
        causal_mask=True if args.causal_mask else None,
        batch_size=batch_size,
        seq_len=s_q,
        kv_seq_len=s_kv,
    )

    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA")

    # HBM images: K/V occupy batch_size * rows_per_batch physical rows of mlen,
    # each batch's seq rows packed at the front of its tile block.
    input_tensor = {
        "Q": q.reshape(batch_size, s_q, hidden_size).reshape(1, -1),
        "K": k_padded.reshape(batch_size * rows_per_batch, mlen).reshape(1, -1),
        "V": v_padded.reshape(batch_size * rows_per_batch, mlen).reshape(1, -1),
    }
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": golden.reshape(rows, hidden_size),
    }

    # FP SRAM slot 1 holds the softmax scale that the online-softmax kernel
    # multiplies QK^T by. M_BTMM already applies the emulator's fixed bmm_scale
    # (0.25), so the kernel must apply scale/0.25 to recover the caller's QK
    # scale of `scale`. (For h_qkv=16 this equals 1.0 and the kernel skips the
    # multiply, masking the issue; for larger h_qkv the value is actually read.)
    softmax_scale = scale / 0.25
    fp_preload = [0.0, softmax_scale, float("-inf")] + [0.0] * 45

    # Q is prestaged in VRAM at addr=0: provide flat fp16 VRAM image. Each batch
    # occupies a rows_per_batch-tall mlen-wide block; the first seq_len rows hold
    # the [seq_len, hidden_size] data, remaining rows are zero padding.
    q_vram = torch.zeros(batch_size, rows_per_batch, mlen)
    q_vram[:, :s_q, :hidden_size] = q.reshape(batch_size, s_q, hidden_size)
    q_vram_flat = q_vram.reshape(-1).to(torch.float16)

    # Explicit HBM layouts for the matrix-prefetch writer. The input_tensor dict
    # passes K/V/Q flattened to (1, -1), so create_mem_for_sim cannot infer the
    # real per-tile row geometry from tensor.shape. Without an explicit layout it
    # falls back to whatever stale tensor_layouts.json is left in the build dir
    # from a previous (e.g. larger-mlen) run, which lays K/V out at the wrong row
    # stride and zero-pads away almost all the data -> K/V load as ~0 into matrix
    # SRAM and the attention output collapses to exp(0)=uniform softmax. Each
    # K/V/Q tile is mlen-wide; K/V occupy kv_phys_rows physical rows, Q occupies
    # q_phys_rows. (Mirrors flash_attention_mha_test.py's tensor_layouts.)
    tensor_layouts = {
        "Q": {
            "physical_shape": [q_phys_rows, mlen],
            "source_rows": q_phys_rows,
            "storage_rows": q_phys_rows,
            "source_row_elements": mlen,
            "storage_row_elements": mlen,
        },
        "K": {
            "physical_shape": [kv_phys_rows, mlen],
            "source_rows": kv_phys_rows,
            "storage_rows": kv_phys_rows,
            "source_row_elements": mlen,
            "storage_row_elements": mlen,
        },
        "V": {
            "physical_shape": [kv_phys_rows, mlen],
            "source_rows": kv_phys_rows,
            "storage_rows": kv_phys_rows,
            "source_row_elements": mlen,
            "storage_row_elements": mlen,
        },
    }

    create_sim_env(
        input_tensor,
        gen_code,
        golden_result,
        fp_preload,
        build_dir=str(build_dir),
        vram_preload=q_vram_flat,
        tensor_layouts=tensor_layouts,
    )

    # At MLEN>=256 the compiler tile-aligns HBM allocations, leaving gaps between
    # tensors; a contiguous writer would place K/V where the prefetch never reads
    # (-> zero K/V tiles). Pin each tensor at its compiler-assigned hbm_base_addr.
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensor}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_gqa_aten",
        data=None,
        specified_data_order=["Q", "K", "V"],
        build_path=build_dir,
        input_tensors=input_tensor,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )

    o_vram_addr = prog._compiler.get_vram_addr(O.name)
    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (rows * hidden_size) // mlen,
        "num_batches": rows,
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
