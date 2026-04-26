"""HBM-to-HBM split and cat copy kernels.

split: Read a wide tensor from HBM, write column slices to separate HBM regions.
cat:   Read from separate HBM regions, write them side-by-side into one wide tensor.

No computation — pure data movement through VRAM.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_TESTBENCH_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_TESTBENCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tile_tensor_program import TileTensorProgram
from tile_tensor_kernel_programs.testbench_runner import emit_single_output_testbench


def build_split_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    total_cols: int,
    split_sizes: list[int],
    hbm_base_addr: int = 0,
    src_hbm_addr: int | None = None,
    dst_hbm_addrs: list[int] | None = None,
) -> tuple[TileTensorProgram, list[object], list[object]]:
    """Split a [1, seq_len, 1, total_cols] tensor along the last dim.

    Args:
        split_sizes: list of column widths, must sum to total_cols.
                     e.g. [128, 128, 128, 512] for q/k/v/mlp split.
        hbm_base_addr: base address for scratch HBM allocation.
        src_hbm_addr: explicit HBM address for the source tensor.
        dst_hbm_addrs: explicit HBM addresses for each destination tensor.
                       Must have the same length as split_sizes if provided.

    Returns:
        (prog, output_tensors, output_bufs)
    """
    if sum(split_sizes) != total_cols:
        raise ValueError(f"split_sizes {split_sizes} sum to {sum(split_sizes)}, expected {total_cols}")
    if dst_hbm_addrs is not None and len(dst_hbm_addrs) != len(split_sizes):
        raise ValueError(
            f"dst_hbm_addrs length {len(dst_hbm_addrs)} != split_sizes length {len(split_sizes)}"
        )

    batch_size = 2
    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=mlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
        hbm_base_addr=hbm_base_addr,
    )

    src_shape = (batch_size, seq_len, 1, total_cols)
    src_in = prog.input("SRC_IN", src_shape, hbm_addr=src_hbm_addr)
    src = prog.tensor("SRC", src_shape)
    for batch_index in range(batch_size):
        prog.copy(src_in[batch_index, :, :, :], src[batch_index, :, :, :])

    output_tensors = []
    output_bufs = []
    col_offset = 0
    for i, width in enumerate(split_sizes):
        dst_shape = (batch_size, seq_len, 1, width)
        dst_addr = dst_hbm_addrs[i] if dst_hbm_addrs is not None else None
        out_buf = prog.input(f"DST_{i}", dst_shape, hbm_addr=dst_addr)
        dst = prog.tensor(f"DST_T_{i}", dst_shape)

        for batch_index in range(batch_size):
            prog.copy(src[batch_index, :, 0:1, col_offset:col_offset + width], dst[batch_index, :, :, :])
            prog.copy(dst[batch_index, :, :, :], out_buf[batch_index, :, :, :])

        output_tensors.append(dst)
        output_bufs.append(out_buf)
        col_offset += width

    return prog, output_tensors, output_bufs


def build_cat_program(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    part_widths: list[int],
    hbm_base_addr: int = 0,
    src_hbm_addrs: list[int] | None = None,
    dst_hbm_addr: int | None = None,
) -> tuple[TileTensorProgram, object, object]:
    """Cat multiple [1, seq_len, 1, width_i] tensors along the last dim.

    Args:
        part_widths: list of column widths for each input part.
        hbm_base_addr: base address for scratch HBM allocation.
        src_hbm_addrs: explicit HBM addresses for each input part tensor.
                       Must have the same length as part_widths if provided.
        dst_hbm_addr: explicit HBM address for the output tensor.

    Returns:
        (prog, output_tensor, output_buf)
    """
    if src_hbm_addrs is not None and len(src_hbm_addrs) != len(part_widths):
        raise ValueError(
            f"src_hbm_addrs length {len(src_hbm_addrs)} != part_widths length {len(part_widths)}"
        )

    total_cols = sum(part_widths)
    batch_size = 2

    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=mlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
        hbm_base_addr=hbm_base_addr,
    )

    # Create inputs for each part
    parts = []
    for i, width in enumerate(part_widths):
        part_shape = (batch_size, seq_len, 1, width)
        src_addr = src_hbm_addrs[i] if src_hbm_addrs is not None else None
        part_in = prog.input(f"PART_{i}", part_shape, hbm_addr=src_addr)
        part_t = prog.tensor(f"PART_T_{i}", part_shape)
        for batch_index in range(batch_size):
            prog.copy(part_in[batch_index, :, :, :], part_t[batch_index, :, :, :])
        parts.append(part_t)

    # Create output
    out_shape = (batch_size, seq_len, 1, total_cols)
    out_buf = prog.input("OUT", out_shape, hbm_addr=dst_hbm_addr)
    dst = prog.tensor("DST", out_shape)

    # Copy each part into the correct column slice
    col_offset = 0
    for i, (part_t, width) in enumerate(zip(parts, part_widths)):
        for batch_index in range(batch_size):
            prog.copy(part_t[batch_index, :, :, :], dst[batch_index, :, 0:1, col_offset:col_offset + width])
        col_offset += width

    for batch_index in range(batch_size):
        prog.copy(dst[batch_index, :, :, :], out_buf[batch_index, :, :, :])
    return prog, dst, out_buf


# ---------------------------------------------------------------------------
# Golden references
# ---------------------------------------------------------------------------

def build_split_golden(x: torch.Tensor, split_sizes: list[int]) -> list[torch.Tensor]:
    return list(torch.split(x, split_sizes, dim=-1))


def build_cat_golden(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    batch_size = 2
    seq_len = 128
    hidden_size = 128
    mlp_dim = 512
    split_sizes = [hidden_size, hidden_size, hidden_size, mlp_dim]
    total_cols = sum(split_sizes)
    kind = os.getenv("TILE_TENSOR_HBM_COPY_KIND", "cat").strip().lower()
    if kind not in {"cat", "split", "both"}:
        raise ValueError("TILE_TENSOR_HBM_COPY_KIND must be one of: cat, split, both")

    # ---- Test cat ----
    if kind in {"cat", "both"}:
        print("Testing cat...")
        part_widths = [hidden_size, mlp_dim]
        prog_cat, _, out_buf_cat = build_cat_program(
            mlen=mlen,
            blen=blen,
            seq_len=seq_len,
            part_widths=part_widths,
        )

        part0 = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.25
        part1 = torch.randn(batch_size, seq_len, 1, mlp_dim, dtype=torch.float32) * 0.25
        golden_cat = build_cat_golden([part0, part1]).reshape(batch_size * seq_len, sum(part_widths))

        emit_single_output_testbench(
            prog=prog_cat,
            out_buf=out_buf_cat,
            input_tensors={
                "PART_0": part0,
                "PART_1": part1,
                "OUT": torch.zeros(batch_size, seq_len, 1, sum(part_widths), dtype=torch.float32),
            },
            golden_output=golden_cat,
            asm_name="tile_tensor_kernel_cat",
            artifact_prefix="tile_tensor_kernel_cat",
            build_dir=_TESTBENCH_DIR / "build",
        )

    # ---- Test split ----
    # Clear HBM binary so the split test's data starts at address 0
    # (each emit_single_output_testbench appends to hbm_for_behave_sim.bin)
    if kind in {"split", "both"}:
        hbm_bin = _TESTBENCH_DIR / "build" / "hbm_for_behave_sim.bin"
        if hbm_bin.exists():
            hbm_bin.unlink()

        print("Testing split...")
        prog_split, _, out_bufs = build_split_program(
            mlen=mlen,
            blen=blen,
            seq_len=seq_len,
            total_cols=total_cols,
            split_sizes=split_sizes,
        )

        src_data = torch.randn(batch_size, seq_len, 1, total_cols, dtype=torch.float32) * 0.25
        golden_parts = build_split_golden(src_data, split_sizes)

        # Verify last split output (mlp part)
        input_tensors = {"SRC_IN": src_data}
        for i, width in enumerate(split_sizes):
            input_tensors[f"DST_{i}"] = torch.zeros(batch_size, seq_len, 1, width, dtype=torch.float32)

        golden_last = golden_parts[-1].reshape(batch_size * seq_len, mlp_dim)
        emit_single_output_testbench(
            prog=prog_split,
            out_buf=out_bufs[-1],
            input_tensors=input_tensors,
            golden_output=golden_last,
            asm_name="tile_tensor_kernel_split",
            artifact_prefix="tile_tensor_kernel_split",
            build_dir=_TESTBENCH_DIR / "build",
        )

    print("Split and cat kernels created successfully!")
