"""
Tile-first sub-matrix manager (standalone, no inheritance).
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from sub_matrix_manager import (
    BLEN,
    HBMAllocator,
    MLEN,
    FPRAMAllocator,
    FPRAMObjectLayout,
    MRAMAllocator,
    MatrixBlockLayout,
    MemoryObjectInfo,
    SubMatrixInfo,
    VRAMAllocator,
    VRAMMatrixBlockLayout,
    VRAMSubMatrixInfo,
)


@dataclass(frozen=True)
class TileCoord:
    row_block: int
    col_block: int
    row_start: int
    col_start: int
    rows: int
    cols: int


class TiledSubMatrixManager:
    """Standalone manager with tile-aware planning and ISA helpers."""

    def __init__(self, mlen: int = MLEN, blen: int = BLEN, fpram_total_size: int = 1024):
        self.mlen = mlen
        self.blen = blen

        self.hbm_matrices: Dict[str, MatrixBlockLayout] = {}
        self.vram_matrices: Dict[str, VRAMMatrixBlockLayout] = {}
        self.fpram_matrices: Dict[str, FPRAMObjectLayout] = {}

        self.hbm_allocator = HBMAllocator()
        self.vram_allocator = VRAMAllocator()
        self.mram_allocator = MRAMAllocator()
        self.fpram_allocator = FPRAMAllocator(total_size=fpram_total_size)

        self.loaded_sub_blocks: Dict[str, SubMatrixInfo] = {}
        self._address_cache: Dict[str, int] = {}

    def __contains__(self, name: str) -> bool:
        return (
            name in self.hbm_matrices
            or name in self.vram_matrices
            or name in self.fpram_matrices
        )

    def __getitem__(self, name: str) -> MemoryObjectInfo:
        if name not in self:
            raise KeyError(f"Object '{name}' not found")

        info = MemoryObjectInfo(name=name, kind="Unknown")
        hbm_layout = self.hbm_matrices.get(name)
        vram_layout = self.vram_matrices.get(name)
        fpram_layout = self.fpram_matrices.get(name)

        if hbm_layout is not None:
            rows, cols = hbm_layout.full_shape
            info.shape = hbm_layout.full_shape
            info.size = rows * cols
            info.hbm_addr = hbm_layout.hbm_base_addr
            info.hbm_size = hbm_layout.hbm_size
            info.kind = "Matrix"

        if vram_layout is not None:
            rows, cols = vram_layout.full_shape
            info.shape = vram_layout.full_shape
            info.size = rows * cols
            info.vram_addr = vram_layout.vram_base_addr
            info.kind = "VRAMMatrix" if hbm_layout is None else "Batch"

        if fpram_layout is not None:
            info.shape = (1, fpram_layout.size)
            info.size = fpram_layout.size
            info.fpram_addr = fpram_layout.fpram_addr
            info.fpram_size = fpram_layout.size
            info.kind = "FPRAMObject"

        return info

    def get(self, name: str, default: Optional[MemoryObjectInfo] = None) -> Optional[MemoryObjectInfo]:
        try:
            return self[name]
        except KeyError:
            return default

    def get_hbm_layout(self, name: str) -> MatrixBlockLayout:
        if name not in self.hbm_matrices:
            raise KeyError(f"HBM matrix '{name}' not found")
        return self.hbm_matrices[name]

    def get_vram_layout(self, name: str) -> VRAMMatrixBlockLayout:
        if name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{name}' not found")
        return self.vram_matrices[name]

    def add_hbm_object(
        self,
        name: str,
        shape: Tuple[int, int],
        hbm_addr: int,
        dtype: str = "fp16",
        kind: str = "HBMObject",
        real_data_ratio: float = 1.125,
    ) -> MemoryObjectInfo:
        del dtype, kind
        rows, cols = shape
        hbm_size = int(rows * cols * real_data_ratio)
        self.hbm_allocator.reserve(name=name, addr=hbm_addr, size=hbm_size, strict=False)
        self.register_matrix(name, shape, hbm_addr, real_data_ratio)
        return self[name]

    def add_vram_object(
        self,
        name: str,
        shape: Tuple[int, int],
        vram_addr: Optional[int] = None,
        dtype: str = "fp16",
        kind: str = "VRAMObject",
        allocate_if_none: bool = True,
    ) -> MemoryObjectInfo:
        del dtype, kind
        rows, cols = shape
        if vram_addr is None:
            if not allocate_if_none:
                raise ValueError("vram_addr is None and allocate_if_none is False")
            vram_addr = self.vram_allocator.allocate(size=rows * cols, name=name)
        self.register_vram_matrix(name=name, shape=shape, vram_base_addr=vram_addr)
        return self[name]

    def free_hbm_object(self, name: str, strict: bool = True) -> Optional[MemoryObjectInfo]:
        if name not in self.hbm_matrices:
            if strict:
                raise KeyError(f"HBM object '{name}' not found")
            return None
        info = self[name]
        self.hbm_allocator.free(name, strict=False)
        self.hbm_matrices.pop(name, None)
        return info

    def free_vram_object(self, name: str, strict: bool = True) -> Optional[MemoryObjectInfo]:
        if name not in self.vram_matrices:
            if strict:
                raise KeyError(f"VRAM object '{name}' not found")
            return None
        info = self[name]
        self.vram_allocator.free(name, strict=strict)
        self.vram_matrices.pop(name, None)
        return info

    def register_matrix(
        self,
        name: str,
        shape: Tuple[int, int],
        hbm_base_addr: int,
        real_data_ratio: float = 1.125,
    ) -> MatrixBlockLayout:
        rows, cols = shape
        row_aligned = rows % self.mlen == 0 or rows == 1
        col_aligned = cols % self.mlen == 0 or cols == 1
        if not row_aligned or not col_aligned:
            raise ValueError(f"HBM matrix {name} shape {shape} must align to mlen={self.mlen}")
        layout = MatrixBlockLayout(
            name=name,
            full_shape=shape,
            block_size=self.mlen,
            hbm_base_addr=hbm_base_addr,
            hbm_size=int(rows * cols * real_data_ratio),
        )
        self.hbm_matrices[name] = layout
        return layout

    def register_vram_matrix(
        self,
        name: str,
        shape: Tuple[int, int],
        vram_base_addr: int,
    ) -> VRAMMatrixBlockLayout:
        rows, cols = shape
        if rows % self.mlen != 0 or cols % self.mlen != 0:
            raise ValueError(f"VRAM matrix {name} shape {shape} must align to mlen={self.mlen}")
        layout = VRAMMatrixBlockLayout(
            name=name,
            full_shape=shape,
            vram_base_addr=vram_base_addr,
            block_size=self.mlen,
        )
        self.vram_matrices[name] = layout
        return layout

    def ensure_vram_layout(self, name: str) -> VRAMMatrixBlockLayout:
        if name in self.vram_matrices:
            return self.vram_matrices[name]
        if name not in self:
            raise KeyError(f"Object '{name}' not found")
        info = self[name]
        if info.vram_addr is None:
            raise ValueError(f"Object '{name}' has no VRAM address")
        return self.register_vram_matrix(name=name, shape=info.shape, vram_base_addr=info.vram_addr)

    def iter_tiles(
        self,
        shape: Tuple[int, int],
        tile_rows: Optional[int] = None,
        tile_cols: Optional[int] = None,
    ) -> Iterator[TileCoord]:
        rows, cols = shape
        tr = tile_rows or self.mlen
        tc = tile_cols or self.mlen
        if rows % tr != 0 or cols % tc != 0:
            raise ValueError(f"Shape {shape} must align to tile ({tr}, {tc})")
        for row_start in range(0, rows, tr):
            for col_start in range(0, cols, tc):
                yield TileCoord(
                    row_block=row_start // tr,
                    col_block=col_start // tc,
                    row_start=row_start,
                    col_start=col_start,
                    rows=tr,
                    cols=tc,
                )

    def iter_hbm_tiles(self, name: str) -> Iterator[Tuple[TileCoord, SubMatrixInfo]]:
        layout = self.get_hbm_layout(name)
        for tile in self.iter_tiles(layout.full_shape):
            yield tile, layout.get_sub_block(tile.row_block, tile.col_block)

    def iter_vram_tiles(self, name: str) -> Iterator[Tuple[TileCoord, VRAMSubMatrixInfo]]:
        layout = self.ensure_vram_layout(name)
        for tile in self.iter_tiles(layout.full_shape):
            yield tile, layout.get_sub_block(tile.row_block, tile.col_block)

    def get_vram_tile_addr(self, name: str, row_block: int, col_block: int) -> int:
        layout = self.ensure_vram_layout(name)
        return layout.get_sub_block(row_block, col_block).vram_addr

    def get_hbm_row_blocks(self, name: str, row_idx: int) -> List[SubMatrixInfo]:
        return self.get_hbm_layout(name).get_row_blocks(row_idx)

    def get_hbm_col_blocks(self, name: str, col_idx: int) -> List[SubMatrixInfo]:
        return self.get_hbm_layout(name).get_col_blocks(col_idx)

    def mark_mram_addr(self, name: str, row_idx: int, col_idx: int, mram_addr: int):
        sub = self.get_hbm_layout(name).get_sub_block(row_idx, col_idx)
        sub.mram_addr = mram_addr
        self.loaded_sub_blocks[f"{name}[{row_idx}][{col_idx}]"] = sub

    def get_loaded_block_addr(self, name: str, row_idx: int, col_idx: int) -> int:
        key = f"{name}[{row_idx}][{col_idx}]"
        if key not in self.loaded_sub_blocks:
            raise KeyError(f"Sub-block {key} not loaded")
        addr = self.loaded_sub_blocks[key].mram_addr
        if addr is None:
            raise KeyError(f"Sub-block {key} has no MRAM addr")
        return addr

    def load_activation_with_format_convert_asm(
        self,
        name: str,
        hbm_base_addr: int,
        batch: int,
        hidden_size: int,
        vram_dest_addr: int,
        hbm_addr_reg: int,
        gp_regs: List[int],
    ) -> str:
        if len(gp_regs) < 5:
            raise ValueError("Need at least 5 GP registers")
        gp_hbm = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_vram = gp_regs[2]

        if hidden_size % self.mlen != 0 or batch % self.mlen != 0:
            raise ValueError(
                f"load_batch tile-first requires shape aligned to mlen={self.mlen}, got {(batch, hidden_size)}"
            )

        col_blocks = hidden_size // self.mlen
        lines = []
        lines.append(f"; Tile load+format-convert {name}: HBM[{hbm_base_addr}] -> VRAM[{vram_dest_addr}]")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {hidden_size}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")
        lines.append(f"S_ADDI_INT gp{gp_hbm}, gp0, {batch * hidden_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_hbm}")

        for c in range(col_blocks):
            hbm_offset = c * self.mlen
            vram_col_base = vram_dest_addr + c * batch * self.mlen
            for b in range(0, batch, self.blen):
                lines.append(f"S_ADDI_INT gp{gp_hbm}, gp0, {hbm_offset + b * hidden_size}")
                lines.append(f"S_ADDI_INT gp{gp_vram}, gp0, {vram_col_base + b * self.mlen}")
                lines.append(f"H_PREFETCH_V gp{gp_vram}, gp{gp_hbm}, a{hbm_addr_reg}, 1, 0")

        return "\n".join(lines) + "\n"

    def store_activation_with_format_convert_asm(
        self,
        name: str,
        vram_src_addr: int,
        batch: int,
        hidden_size: int,
        hbm_dest_addr: int,
        hbm_addr_reg: int,
        gp_regs: List[int],
    ) -> str:
        if len(gp_regs) < 3:
            raise ValueError("Need at least 3 GP registers")
        gp_hbm = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_vram = gp_regs[2]

        if hidden_size % self.mlen != 0 or batch % self.mlen != 0:
            raise ValueError(
                f"store_to_hbm tile-first requires shape aligned to mlen={self.mlen}, got {(batch, hidden_size)}"
            )

        col_blocks = hidden_size // self.mlen
        lines = []
        lines.append(f"; Tile store+format-convert {name}: VRAM[{vram_src_addr}] -> HBM[{hbm_dest_addr}]")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {hidden_size}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")

        for c in range(col_blocks):
            hbm_offset = c * self.mlen
            vram_col_base = vram_src_addr + c * batch * self.mlen
            for b in range(0, batch, self.blen):
                lines.append(f"S_ADDI_INT gp{gp_hbm}, gp0, {hbm_offset + b * hidden_size}")
                lines.append(f"S_ADDI_INT gp{gp_vram}, gp0, {vram_col_base + b * self.mlen}")
                lines.append(f"H_STORE_V gp{gp_vram}, gp{gp_hbm}, a{hbm_addr_reg}, 0")

        return "\n".join(lines) + "\n"

    def vram_block_add_tile_asm(
        self,
        src1_addr: int,
        src2_addr: int,
        dst_addr: int,
        gp_regs: List[int],
    ) -> str:
        if len(gp_regs) < 4:
            raise ValueError("Need at least 4 GP registers")
        gp_dst, gp_src1, gp_src2, gp_loop = gp_regs[:4]
        lines = []
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"S_ADDI_INT gp{gp_src1}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_src2}, gp0, {src2_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {self.mlen}")
        lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_src1}, gp{gp_src2}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}")
        lines.append(f"S_ADDI_INT gp{gp_src1}, gp{gp_src1}, {self.mlen}")
        lines.append(f"S_ADDI_INT gp{gp_src2}, gp{gp_src2}, {self.mlen}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        return "\n".join(lines) + "\n"

    def vram_sub_projection_tile_asm(
        self,
        vram_row_tile_addrs: List[int],
        mram_tile_addrs: List[int],
        result_vram_addr: int,
        transpose_mram: bool,
        gp_regs: List[int],
    ) -> str:
        if len(gp_regs) < 9:
            raise ValueError("Need at least 9 GP registers")
        if len(vram_row_tile_addrs) != len(mram_tile_addrs):
            raise ValueError("Tile count mismatch between VRAM row and MRAM tiles")

        gp_act = gp_regs[0]
        gp_mat = gp_regs[1]
        gp_result = gp_regs[2]
        gp_loop_outer = gp_regs[3]
        gp_loop_middle = gp_regs[4]
        gp_loop_inner = gp_regs[5]
        gp_act_row_base = gp_regs[6]
        gp_mat_col_base = gp_regs[7]
        gp_result_col_base = gp_regs[8]

        tiles_per_mlen = self.mlen // self.blen
        hidden_tiles = len(vram_row_tile_addrs)

        act_hidden_stride = vram_row_tile_addrs[1] - vram_row_tile_addrs[0] if hidden_tiles > 1 else 0
        mat_hidden_stride = mram_tile_addrs[1] - mram_tile_addrs[0] if hidden_tiles > 1 else 0
        output_row_stride = self.blen * self.mlen
        mat_output_col_stride = self.blen * self.mlen if transpose_mram else self.blen

        mm_op = "M_TMM" if transpose_mram else "M_MM"
        mm_args = "gp{a}, gp{m}" if transpose_mram else "gp{m}, gp{a}"
        mm_args = mm_args.format(a=gp_act, m=gp_mat)

        lines = []
        lines.append(f"S_ADDI_INT gp{gp_mat_col_base}, gp0, {mram_tile_addrs[0]}")
        lines.append(f"S_ADDI_INT gp{gp_result_col_base}, gp0, {result_vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop_outer}, {tiles_per_mlen}")
        lines.append(f"S_ADDI_INT gp{gp_act_row_base}, gp0, {vram_row_tile_addrs[0]}")
        lines.append(f"S_ADDI_INT gp{gp_result}, gp{gp_result_col_base}, 0")
        lines.append(f"C_LOOP_START gp{gp_loop_middle}, {tiles_per_mlen}")
        lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act_row_base}, 0")
        lines.append(f"S_ADDI_INT gp{gp_mat}, gp{gp_mat_col_base}, 0")
        lines.append(f"C_LOOP_START gp{gp_loop_inner}, {hidden_tiles}")
        lines.append(f"{mm_op} 0, {mm_args}")
        lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {act_hidden_stride}")
        lines.append(f"S_ADDI_INT gp{gp_mat}, gp{gp_mat}, {mat_hidden_stride}")
        lines.append(f"C_LOOP_END gp{gp_loop_inner}")
        lines.append(f"M_MM_WO gp{gp_result}, gp0, 0")
        lines.append(f"S_ADDI_INT gp{gp_act_row_base}, gp{gp_act_row_base}, {output_row_stride}")
        lines.append(f"S_ADDI_INT gp{gp_result}, gp{gp_result}, {output_row_stride}")
        lines.append(f"C_LOOP_END gp{gp_loop_middle}")
        lines.append(f"S_ADDI_INT gp{gp_mat_col_base}, gp{gp_mat_col_base}, {mat_output_col_stride}")
        lines.append(f"S_ADDI_INT gp{gp_result_col_base}, gp{gp_result_col_base}, {self.blen}")
        lines.append(f"C_LOOP_END gp{gp_loop_outer}")
        return "\n".join(lines) + "\n"

    def reset(self):
        self.hbm_matrices.clear()
        self.vram_matrices.clear()
        self.fpram_matrices.clear()
        self.hbm_allocator.reset()
        self.vram_allocator.reset()
        self.mram_allocator.reset()
        self.fpram_allocator.reset()
        self.loaded_sub_blocks.clear()
        self._address_cache.clear()
