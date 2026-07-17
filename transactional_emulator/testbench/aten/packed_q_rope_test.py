"""Transactional numerical test for the looped packed-Q RoPE emitter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena_frontend import _pad_rope_inputs_for_head_slots
from compiler.aten.reference import _make_rotate_half_matrix
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.aten.rope_test import make_rope_tables
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.sliced_layer_test_builder import (
    _active_precision_settings,
    _load_to_vector_fp,
    quantize_to_mxfp,
    quantize_to_vector_fp,
)
from transactional_emulator.tools.create_sim_env import create_sim_env


def _layout(rows: int, cols: int) -> dict[str, object]:
    return {
        "physical_shape": [rows, cols],
        "source_rows": rows,
        "storage_rows": rows,
        "source_row_elements": cols,
        "storage_row_elements": cols,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    mlen = int(args.mlen)
    blen = int(args.blen)
    batch_size = int(args.batch_size if args.batch_size is not None else 2)
    seq_len = int(args.seq_len if args.seq_len is not None else mlen + blen)
    head_dim = int(args.head_dim)
    head_slot_dim = int(args.hlen if args.hlen is not None else head_dim)
    groups = int(args.groups)
    if head_dim > head_slot_dim:
        raise ValueError(f"head_dim={head_dim} exceeds HLEN={head_slot_dim}")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim={head_dim} must be even")
    if head_slot_dim > mlen:
        raise ValueError(f"HLEN={head_slot_dim} exceeds MLEN={mlen}")
    physical_broadcast = min(
        int(args.broadcast_amount if args.broadcast_amount is not None else mlen // head_slot_dim),
        mlen // head_slot_dim,
    )
    if physical_broadcast <= 0:
        raise ValueError("packed Q requires at least one physical head slot")
    rows_per_slab = math.ceil(seq_len / mlen) * mlen
    slab_count = groups * batch_size

    args.hlen = head_slot_dim
    args.broadcast_amount = physical_broadcast
    if args.mram_tiles is None:
        args.mram_tiles = 1
    build_dir = Path(__file__).parent / "build" / "packed_q_rope"
    hw = setup_hw(args, build_dir)

    torch.manual_seed(args.seed)
    q = torch.zeros(groups, batch_size, rows_per_slab, mlen)
    cos_active, sin_active = make_rope_tables(seq_len, head_dim)
    for group in range(groups):
        for batch in range(batch_size):
            for lane in range(physical_broadcast):
                start = lane * head_slot_dim
                active = torch.randn(seq_len, head_dim) * 0.5
                q[group, batch, :seq_len, start : start + head_dim] = active

    rotation = _make_rotate_half_matrix(head_dim)
    packed_rotation, packed_cos, packed_sin = _pad_rope_inputs_for_head_slots(
        rotation,
        cos_active,
        sin_active,
        padded_seq_len=rows_per_slab,
        group_width=mlen,
        head_slot_dim=head_slot_dim,
        broadcast_amount=physical_broadcast,
    )
    precision = _active_precision_settings()
    q_vram = quantize_to_vector_fp(q, precision)
    rotation_mram = quantize_to_vector_fp(
        quantize_to_mxfp(packed_rotation, precision["HBM_M_WEIGHT_TYPE"]),
        precision,
    )
    cos_vram = _load_to_vector_fp(packed_cos, precision["HBM_V_ACT_TYPE"], precision)
    sin_vram = _load_to_vector_fp(packed_sin, precision["HBM_V_ACT_TYPE"], precision)
    q_rot = quantize_to_vector_fp(torch.matmul(q_vram, rotation_mram), precision)
    golden = quantize_to_vector_fp(
        quantize_to_vector_fp(q_vram * cos_vram, precision)
        + quantize_to_vector_fp(q_rot * sin_vram, precision),
        precision,
    )
    q_flat = q.reshape(-1).to(torch.float16)
    golden_flat = golden.reshape(slab_count * rows_per_slab, mlen)

    prog = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        real_data_ratio=hw.real_data_ratio,
        mram_tile_capacity=int(args.mram_tiles),
    )
    q_input = prog.input(
        "Q_full",
        shape=(batch_size * seq_len, groups * mlen),
        physical_shape=(batch_size * rows_per_slab, groups * mlen),
        prestaged_vram_addr=0,
    )
    rotation_input = prog.input("R", shape=(mlen, mlen), physical_shape=(mlen, mlen))
    cos_input = prog.input(
        "COS",
        shape=(rows_per_slab, mlen),
        physical_shape=(rows_per_slab, mlen),
    )
    sin_input = prog.input(
        "SIN",
        shape=(rows_per_slab, mlen),
        physical_shape=(rows_per_slab, mlen),
    )
    q_var = prog.load_batch(q_input, name="Q_full")
    cos_var = prog.load_batch(cos_input, name="COS")
    sin_var = prog.load_batch(sin_input, name="SIN")
    prog.rope_packed_q(
        q_var,
        rotation_input,
        cos_var,
        sin_var,
        slab_count=slab_count,
        rows_per_slab=rows_per_slab,
        active_rows=seq_len,
    )
    asm = prog.compile()

    tensors = {
        "Q_full": q_flat.reshape(1, -1),
        "R": packed_rotation,
        "COS": packed_cos,
        "SIN": packed_sin,
    }
    layouts = {
        "Q_full": _layout(slab_count * rows_per_slab, mlen),
        "R": _layout(mlen, mlen),
        "COS": _layout(rows_per_slab, mlen),
        "SIN": _layout(rows_per_slab, mlen),
    }
    create_sim_env(
        tensors,
        asm,
        {"input_tensor": tensors, "original_output": golden_flat},
        [],
        build_dir=str(build_dir),
        vram_preload=q_flat,
        tensor_layouts=layouts,
    )
    data_order = ["R", "COS", "SIN"]
    hbm_addrs = {
        name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in data_order
    }
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="packed_q_rope",
        data=None,
        specified_data_order=data_order,
        build_path=build_dir,
        input_tensors=tensors,
        tensor_layouts=layouts,
        hbm_addrs=hbm_addrs,
    )
    comparison = {
        "start_row_idx": prog.get_vram_addr(q_var.name) // mlen,
        "num_rows": slab_count * rows_per_slab,
        "num_batches": slab_count * rows_per_slab,
        "elements_per_batch": mlen,
        "row_dim": mlen,
        "use_stride_mode": False,
        "atol": 0.02,
        "rtol": 0.02,
        "min_allclose_match_rate": 99.0,
    }
    with (build_dir / "comparison_params.json").open("w") as f:
        json.dump(comparison, f, indent=2)
    with (build_dir / "generated_asm_code.asm").open("w") as f:
        f.write(asm)

    print(
        f"Packed Q RoPE: slabs={slab_count}, seq={seq_len}, rows_per_slab={rows_per_slab}, "
        f"head_dim={head_dim}, HLEN={head_slot_dim}, broadcast={physical_broadcast}, "
        f"ISA lines={len(asm.splitlines())}"
    )
    if not args.no_run:
        run_and_assert(build_dir, "packed_q_rope", mlen=mlen, blen=blen)
