"""Synthetic sequence-tiled packed-GQA attention test.

This test exercises the canonical logical-KV-group packed-GQA path. It supports
multiple batches, sequence tiling, HLEN padding, resident K/V, and streaming K/V.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.aten.golden import quantize_to_mxfp
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


def _gqa_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, *, causal: bool) -> torch.Tensor:
    # q: [batch, seq, hq, head_dim]; k/v: [batch, seq, hkv, head_dim]
    batch, seq_len, hq, _head_dim = q.shape
    hkv = k.shape[2]
    if hq % hkv != 0:
        raise ValueError(f"hq ({hq}) must be divisible by hkv ({hkv})")
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2).repeat_interleave(hq // hkv, dim=1)
    v_t = v.transpose(1, 2).repeat_interleave(hq // hkv, dim=1)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale, is_causal=causal)
    return out.transpose(1, 2).reshape(batch, seq_len, hq, -1)


def _pack_q_for_kv_groups(
    q: torch.Tensor,
    *,
    rows_per_batch: int,
    mlen: int,
    hkv: int,
    broadcast_amount: int,
    head_slot_dim: int,
) -> torch.Tensor:
    batch_size, seq_len, hq, head_dim = q.shape
    ratio = hq // hkv
    chunks_per_kv = math.ceil(ratio / broadcast_amount)
    packed = torch.zeros(hkv, chunks_per_kv, batch_size, rows_per_batch, mlen, dtype=q.dtype)
    for kv_head in range(hkv):
        for local_head in range(ratio):
            q_head = kv_head * ratio + local_head
            chunk = local_head // broadcast_amount
            lane = local_head % broadcast_amount
            start = lane * head_slot_dim
            packed[kv_head, chunk, :, :seq_len, start : start + head_dim] = q[:, :, q_head, :]
    return packed.reshape(-1)


def _pack_kv_heads(
    tensor: torch.Tensor,
    *,
    rows_per_batch: int,
    mlen: int,
) -> list[torch.Tensor]:
    batch_size, seq_len, hkv, head_dim = tensor.shape
    packed = []
    for kv_head in range(hkv):
        head_tensor = torch.zeros(batch_size, rows_per_batch, mlen, dtype=tensor.dtype)
        head_tensor[:, :seq_len, :head_dim] = tensor[:, :, kv_head, :]
        packed.append(head_tensor.reshape(1, -1))
    return packed


def _pack_output_golden(
    out: torch.Tensor,
    *,
    rows_per_batch: int,
    mlen: int,
    hkv: int,
    broadcast_amount: int,
    head_slot_dim: int,
) -> torch.Tensor:
    batch_size, seq_len, hq, head_dim = out.shape
    ratio = hq // hkv
    chunks_per_kv = math.ceil(ratio / broadcast_amount)
    packed = torch.zeros(hkv, chunks_per_kv, batch_size, rows_per_batch, mlen, dtype=out.dtype)
    for kv_head in range(hkv):
        for local_head in range(ratio):
            q_head = kv_head * ratio + local_head
            chunk = local_head // broadcast_amount
            lane = local_head % broadcast_amount
            start = lane * head_slot_dim
            packed[kv_head, chunk, :, :seq_len, start : start + head_dim] = out[:, :, q_head, :]
    return packed.reshape(hkv * chunks_per_kv * batch_size * rows_per_batch, mlen)


def _tensor_layout(rows: int, cols: int) -> dict[str, object]:
    return {
        "physical_shape": [rows, cols],
        "source_rows": rows,
        "storage_rows": rows,
        "source_row_elements": cols,
        "storage_row_elements": cols,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--hq", type=int, default=4)
    parser.add_argument("--hkv", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--causal", action="store_true", help="Apply a causal attention mask")
    parser.add_argument("--no-run", action="store_true", help="Generate artifacts without running the emulator")
    parser.add_argument(
        "--timing-mode",
        choices=("legacy", "rtl-v1"),
        default="rtl-v1",
        help="Transactional timing model (default rtl-v1); numerical comparison is unchanged.",
    )
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Write memory_profile.json, including the rtl-v1 timeline.",
    )
    parser.add_argument(
        "--event-trace",
        action="store_true",
        help="Write event_trace.json with issue/start/result-ready/completion cycles.",
    )
    parser.add_argument(
        "--require-rtl-validated",
        action="store_true",
        help=(
            "Write artifacts, then fail if rtl-v1 uses unsupported or "
            "out-of-calibration-domain opcodes."
        ),
    )
    args = parser.parse_args()
    if args.require_rtl_validated and args.timing_mode != "rtl-v1":
        parser.error("--require-rtl-validated requires --timing-mode rtl-v1")
    return args


if __name__ == "__main__":
    args = _parse_args()
    mlen = int(args.mlen)
    blen = int(args.blen)
    hq = int(args.hq)
    hkv = int(args.hkv)
    head_dim = int(args.head_dim)
    batch_size = int(args.batch_size if args.batch_size is not None else 1)
    seq_len = int(args.seq_len if args.seq_len is not None else 96)

    if seq_len <= mlen:
        raise ValueError(f"seq_len ({seq_len}) must exceed MLEN ({mlen}) to exercise sequence tiling")
    if hq % hkv != 0:
        raise ValueError(f"hq ({hq}) must be divisible by hkv ({hkv})")
    ratio = hq // hkv
    if head_dim > mlen:
        raise ValueError(f"head_dim ({head_dim}) must fit in MLEN ({mlen})")
    head_slot_dim = int(args.hlen if args.hlen is not None else head_dim)
    if head_dim > head_slot_dim:
        raise ValueError(
            f"head_dim ({head_dim}) must not exceed HLEN/head_slot_dim ({head_slot_dim})"
        )
    logical_broadcast_amount = (
        int(args.broadcast_amount)
        if getattr(args, "broadcast_amount", None) is not None
        else mlen // head_slot_dim
    )
    if logical_broadcast_amount <= 0:
        raise ValueError(f"broadcast_amount ({logical_broadcast_amount}) must be positive")
    physical_broadcast_amount = min(logical_broadcast_amount, mlen // head_slot_dim)
    if physical_broadcast_amount <= 0:
        raise ValueError(
            f"MLEN ({mlen}) must fit at least one HLEN slot ({head_slot_dim})"
        )
    broadcast_amount = physical_broadcast_amount
    chunks_per_kv = math.ceil(ratio / broadcast_amount)
    if seq_len % blen != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of BLEN ({blen})")

    rows_per_batch = ((seq_len + mlen - 1) // mlen) * mlen
    scale = 1.0 / math.sqrt(head_dim)

    args.hlen = head_slot_dim
    args.broadcast_amount = physical_broadcast_amount
    build_dir = Path(__file__).parent / "build" / "flash_attention_packed_gqa_tiled"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(
        "Packed-GQA sequence-tiled attention "
        f"(mlen={mlen}, blen={blen}, seq={seq_len}, hq={hq}, hkv={hkv}, "
        f"head_dim={head_dim}, head_slot_dim={head_slot_dim}, "
        f"logical_broadcast={logical_broadcast_amount}, "
        f"physical_broadcast={physical_broadcast_amount}, chunks_per_kv={chunks_per_kv}, "
        f"rows_per_batch={rows_per_batch})"
    )
    print("=" * 80)

    torch.manual_seed(args.seed)
    q = torch.randn(batch_size, seq_len, hq, head_dim) * 0.5
    k = torch.randn(batch_size, seq_len, hkv, head_dim) * 0.5
    v = torch.randn(batch_size, seq_len, hkv, head_dim) * 0.5

    k_q = quantize_to_mxfp(k)
    v_q = quantize_to_mxfp(v)
    golden = _gqa_sdpa(q.float(), k_q.float(), v_q.float(), scale, causal=args.causal)
    packed_golden = _pack_output_golden(
        golden,
        rows_per_batch=rows_per_batch,
        mlen=mlen,
        hkv=hkv,
        broadcast_amount=broadcast_amount,
        head_slot_dim=head_slot_dim,
    )

    q_vram_flat = _pack_q_for_kv_groups(
        q,
        rows_per_batch=rows_per_batch,
        mlen=mlen,
        hkv=hkv,
        broadcast_amount=broadcast_amount,
        head_slot_dim=head_slot_dim,
    ).to(torch.float16)
    k_inputs = _pack_kv_heads(k, rows_per_batch=rows_per_batch, mlen=mlen)
    v_inputs = _pack_kv_heads(v, rows_per_batch=rows_per_batch, mlen=mlen)

    mram_tiles = int(args.mram_tiles if args.mram_tiles is not None else 4)
    prog = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        real_data_ratio=hw.real_data_ratio,
        mram_tile_capacity=mram_tiles,
    )
    prog.hlen = head_slot_dim
    prog.broadcast_amount = broadcast_amount

    q_input = prog.input(
        "Q_full",
        shape=(batch_size * seq_len, hkv * chunks_per_kv * mlen),
        physical_shape=(batch_size * rows_per_batch, hkv * chunks_per_kv * mlen),
        prestaged_vram_addr=0,
    )
    q_full = prog.load_batch(q_input, name="Q_full")
    output = prog.alloc(
        "O_full",
        batch_size * seq_len,
        hkv * chunks_per_kv * mlen,
        strict=False,
        physical_shape=(batch_size * rows_per_batch, hkv * chunks_per_kv * mlen),
    )
    prog.vram_fill_zero(output)
    scratch_rows = mlen * (broadcast_amount + ratio)
    scratch = prog.alloc("packed_attn_scratch", scratch_rows, mlen, strict=True)
    output_base = prog.get_vram_addr(output.name)
    scratch_base = prog.get_vram_addr(scratch.name)
    causal_mask = None
    causal_mask_data = None
    if args.causal:
        causal_mask_data = torch.zeros(mlen, mlen)
        causal_mask_data.masked_fill_(torch.triu(torch.ones(mlen, mlen), diagonal=1).bool(), float("-inf"))
        causal_input = prog.input("causal_mask", shape=(mlen, mlen), physical_shape=(mlen, mlen))
        causal_mask = prog.load_batch(causal_input, name="CAUSAL_MASK")

    kv_pairs = []
    input_tensor: dict[str, torch.Tensor] = {"Q_full": q_vram_flat.reshape(1, -1)}
    tensor_layouts = {
        "Q_full": _tensor_layout(
            batch_size * rows_per_batch * hkv * chunks_per_kv,
            mlen,
        )
    }
    data_order = []
    if causal_mask_data is not None:
        input_tensor["causal_mask"] = causal_mask_data.reshape(1, -1)
        tensor_layouts["causal_mask"] = _tensor_layout(mlen, mlen)
        data_order.append("causal_mask")
    for kv_head in range(hkv):
        k_name = f"K_{kv_head}"
        v_name = f"V_{kv_head}"
        K = prog.input(
            k_name,
            shape=(batch_size * seq_len, head_dim),
            physical_shape=(batch_size * rows_per_batch, mlen),
        )
        V = prog.input(
            v_name,
            shape=(batch_size * seq_len, head_dim),
            physical_shape=(batch_size * rows_per_batch, mlen),
        )
        kv_pairs.append((K, V))
        input_tensor[k_name] = k_inputs[kv_head]
        input_tensor[v_name] = v_inputs[kv_head]
        tensor_layouts[k_name] = _tensor_layout(batch_size * rows_per_batch, mlen)
        tensor_layouts[v_name] = _tensor_layout(batch_size * rows_per_batch, mlen)
        data_order.extend([k_name, v_name])

    schedule = prog.flash_attention_packed_gqa(
        q_full,
        output,
        kv_pairs,
        batch_size=batch_size,
        seq_len=seq_len,
        rows_per_batch=rows_per_batch,
        gqa_ratio=ratio,
        physical_broadcast=broadcast_amount,
        head_slot_dim=head_slot_dim,
        scratch_base_address=scratch_base,
        scale=scale,
        causal_mask=causal_mask,
    )
    print(f"Packed GQA schedule: {schedule}")
    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA")

    fp_preload = [0.0, scale / 0.25, float("-inf")] + [0.0] * 45
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": packed_golden,
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

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in data_order}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="flash_attention_packed_gqa_tiled",
        data=None,
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=input_tensor,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )

    comparison_params = {
        "start_row_idx": output_base // mlen,
        "num_rows": hkv * chunks_per_kv * batch_size * rows_per_batch,
        "num_batches": hkv * chunks_per_kv * batch_size * rows_per_batch,
        "elements_per_batch": mlen,
        "row_dim": mlen,
        "use_stride_mode": False,
        "use_slice_mode": False,
        "atol": 0.2,
        "rtol": 0.2,
        "min_allclose_match_rate": 90.0,
    }
    with (build_dir / "comparison_params.json").open("w") as f:
        json.dump(comparison_params, f, indent=2)
    with (build_dir / "generated_asm_code.asm").open("w") as f:
        f.write(gen_code)

    print(f"\nOutput at VRAM row {output_base // mlen}")
    if not args.no_run:
        run_and_assert(
            build_dir,
            "flash_attention_packed_gqa_tiled",
            mlen=mlen,
            blen=blen,
            profile_memory=args.profile_memory,
        timing_mode=args.timing_mode,
        event_trace=(build_dir / "event_trace.json") if args.event_trace else None,
        require_rtl_validated=args.require_rtl_validated,
    )
