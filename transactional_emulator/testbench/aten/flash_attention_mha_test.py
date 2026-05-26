"""Configurable ATen per-head MHA flash attention golden test."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (REPO_ROOT, REPO_ROOT / "PLENA_Compiler"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from compiler.aten.ops.registry import Backend, OpRegistry  # noqa: E402
import compiler.aten.ops as ops  # noqa: E402
from transactional_emulator.testbench.aten.configurable import AtenTemplateTestbench  # noqa: E402


def mha_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    q_t = q.unsqueeze(1)
    k_t = k.unsqueeze(1)
    v_t = v.unsqueeze(1)
    return F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale).squeeze(1)


def pad_batch_rows(tensor: torch.Tensor, physical_rows: int, physical_cols: int) -> torch.Tensor:
    padded = torch.zeros(tensor.shape[0], physical_rows, physical_cols, dtype=tensor.dtype)
    padded[:, : tensor.shape[1], : tensor.shape[2]] = tensor
    return padded.reshape(tensor.shape[0] * physical_rows, physical_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--unroll-attention", action="store_true")
    AtenTemplateTestbench.add_common_args(
        parser,
        default_build_dir=Path(__file__).parent / "build" / "flash_attention_mha",
    )
    args = parser.parse_args()
    tb = AtenTemplateTestbench(args, name="flash_attention_mha_aten", default_build_dir=args.build_dir)

    if args.seq_len > tb.hw.mlen:
        raise ValueError(f"seq_len={args.seq_len} exceeds one-tile MHA test MLEN={tb.hw.mlen}")
    if args.head_dim > tb.hw.mlen:
        raise ValueError(f"head_dim={args.head_dim} exceeds one-tile MHA test MLEN={tb.hw.mlen}")

    scale = 1.0 / math.sqrt(args.head_dim)
    q = torch.randn(args.batch_size, args.seq_len, args.head_dim) * 0.5
    k = torch.randn(args.batch_size, args.seq_len, args.head_dim) * 0.5
    v = torch.randn(args.batch_size, args.seq_len, args.head_dim) * 0.5
    golden = mha_sdpa(q.float(), k.float(), v.float(), scale)

    physical_rows = max(args.seq_len, tb.hw.mlen)
    physical_cols = max(args.head_dim, tb.hw.mlen)
    physical_shape = (args.batch_size * physical_rows, physical_cols)
    q_padded = pad_batch_rows(q, physical_rows, physical_cols)
    k_padded = pad_batch_rows(k, physical_rows, physical_cols)
    v_padded = pad_batch_rows(v, physical_rows, physical_cols)

    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)
    prog = tb.compiler()
    prog.unroll_attention = args.unroll_attention
    q_input = prog.input(
        "Q",
        shape=(args.batch_size * args.seq_len, args.head_dim),
        physical_shape=physical_shape,
        prestaged_vram_addr=0,
    )
    k_input = prog.input(
        "K",
        shape=(args.batch_size * args.seq_len, args.head_dim),
        physical_shape=physical_shape,
    )
    v_input = prog.input(
        "V",
        shape=(args.batch_size * args.seq_len, args.head_dim),
        physical_shape=physical_shape,
    )
    q_var = prog.load_batch(q_input, name="Q")
    o = ops.flash_attention(
        prog,
        q_var,
        k_input,
        v_input,
        scale,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        kv_seq_len=args.seq_len,
    )
    gen_code = prog.compile()

    o_addr = prog._compiler.get_vram_addr(o.name)
    tensor_layouts = {
        name: {
            "source_row_elements": physical_shape[1],
            "storage_row_elements": physical_shape[1],
            "physical_shape": list(physical_shape),
        }
        for name in ("Q", "K", "V")
    }
    tb.prepare_and_run(
        asm_name="flash_attention_mha_aten",
        gen_code=gen_code,
        input_tensor={"Q": q_padded, "K": k_padded, "V": v_padded},
        golden_result={"original_output": golden.reshape(args.batch_size * args.seq_len, args.head_dim)},
        fp_preload=[0.0, scale, float("-inf")] + [0.0] * 45,
        specified_data_order=["Q", "K", "V"],
        comparison_params=tb.comparison_params(
            vram_addr=o_addr,
            rows=args.batch_size * args.seq_len,
            logical_cols=args.head_dim,
            physical_cols=o.physical_shape[1],
            physical_rows=o.physical_shape[0],
            use_slice_mode=o.physical_shape[1] > args.head_dim,
            slice_per_row=args.head_dim if o.physical_shape[1] > args.head_dim else None,
        ),
        vram_preload=q_padded.reshape(-1).to(torch.float16),
        tensor_layouts=tensor_layouts,
    )


if __name__ == "__main__":
    main()
