"""
Tile-first developer compiler (standalone, no inheritance).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.asm_templates import (
    preload_act_asm,
    preload_addr_reg_asm,
    reset_reg_asm,
    store_act_asm,
)
from sub_matrix_manager import BLEN, MLEN
from tiled_sub_matrix_manager import TiledSubMatrixManager


class RegisterAllocator:
    """Register allocator copied for standalone tiled compiler."""

    def __init__(self, start_gp: int = 1, start_addr: int = 0, start_fp: int = 1):
        self.gp_registers = list(range(start_gp, 16))
        self.addr_registers = list(range(start_addr, 8))
        self.fp_registers = list(range(start_fp, 8))
        self.used_gp: List[int] = []
        self.used_addr: List[int] = []
        self.used_fp: List[int] = []

    def allocate_gp(self, count: int = 1) -> List[int]:
        if len(self.gp_registers) < count:
            raise RuntimeError(f"Not enough GP registers. Need {count}, have {len(self.gp_registers)}")
        allocated = self.gp_registers[:count]
        self.gp_registers = self.gp_registers[count:]
        self.used_gp.extend(allocated)
        return allocated

    def free_gp(self, registers: List[int]):
        for reg in registers:
            if reg in self.used_gp:
                self.used_gp.remove(reg)
                self.gp_registers.append(reg)
        self.gp_registers.sort()

    def allocate_addr(self, count: int = 1) -> List[int]:
        if len(self.addr_registers) < count:
            raise RuntimeError(
                f"Not enough address registers. Need {count}, have {len(self.addr_registers)}"
            )
        allocated = self.addr_registers[:count]
        self.addr_registers = self.addr_registers[count:]
        self.used_addr.extend(allocated)
        return allocated

    def free_addr(self, registers: List[int]):
        for reg in registers:
            if reg in self.used_addr:
                self.used_addr.remove(reg)
                self.addr_registers.append(reg)
        self.addr_registers.sort()


class TiledDeveloperCompiler:
    """Standalone tiled compiler with independent management and tile-first execution."""

    class InterruptManager:
        def __init__(self, compiler: "TiledDeveloperCompiler"):
            self.compiler = compiler
            self.enabled = False
            self._k_count = 0
            self._tile_count = 0
            self.current_matrix: str = ""
            self.current_activation: str = ""
            self._mlen = 64
            self._blen = 4
            self._batch = 64
            self._q_block_idx = 0
            self._k_block_idx = 0
            self._s_tile_address = 0

        def reset(self):
            self._k_count = 0
            self._tile_count = 0
            self._q_block_idx = 0
            self._k_block_idx = 0
            self._s_tile_address = 0

        def enable(self):
            self.enabled = True

        def disable(self):
            self.enabled = False

        def trigger_k_start(self) -> str:
            return ""

        def trigger_k_prefetch_done(self) -> str:
            return ""

        def trigger_s_tile_done(self) -> str:
            return ""

        def trigger_k_end(self) -> str:
            return ""

    def __init__(self, mlen: int = MLEN, blen: int = BLEN, fpram_total_size: int = 1024):
        self.mlen = mlen
        self.blen = blen
        self.sub_matrix_manager = TiledSubMatrixManager(
            mlen=mlen,
            blen=blen,
            fpram_total_size=fpram_total_size,
        )
        self.register_allocator = RegisterAllocator()
        self.generated_code = ""
        self.interrupt = self.InterruptManager(self)

    def _reset_default_scale_stride_asm(self, gp_scale: int, gp_stride: int) -> str:
        tile_elems = self.mlen * self.mlen
        lines = []
        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {tile_elems}")
        lines.append(f"C_SET_SCALE_REG gp{gp_scale}")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {self.mlen}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")
        return "\n".join(lines) + "\n"

    # ---------------------------------------------------------------------
    # Management helpers
    # ---------------------------------------------------------------------

    def get_code(self) -> str:
        return self.generated_code

    def reset(self):
        self.generated_code = ""
        self.sub_matrix_manager.reset()
        self.register_allocator = RegisterAllocator()
        self.interrupt.reset()

    def get_symbol_table(self) -> Dict[str, Dict[str, object]]:
        names = sorted(
            set(self.sub_matrix_manager.hbm_matrices)
            | set(self.sub_matrix_manager.vram_matrices)
            | set(self.sub_matrix_manager.fpram_matrices)
        )
        table: Dict[str, Dict[str, object]] = {}
        for name in names:
            info = self.sub_matrix_manager[name]
            table[name] = {
                "kind": info.kind,
                "shape": info.shape,
                "size": info.size,
                "hbm_addr": info.hbm_addr,
                "hbm_size": info.hbm_size,
                "vram_addr": info.vram_addr,
                "fpram_addr": info.fpram_addr,
                "fpram_size": info.fpram_size,
            }
        return table

    def get_tensor_info(self, name: str):
        return self.sub_matrix_manager[name]

    def add_hbm_object(
        self,
        name: str,
        shape: Tuple[int, int],
        hbm_addr: int,
        dtype: str = "fp16",
        kind: str = "HBMObject",
        real_data_ratio: float = 1.125,
    ):
        return self.sub_matrix_manager.add_hbm_object(
            name=name,
            shape=shape,
            hbm_addr=hbm_addr,
            dtype=dtype,
            kind=kind,
            real_data_ratio=real_data_ratio,
        )

    def free_hbm_object(self, name: str, strict: bool = False):
        return self.sub_matrix_manager.free_hbm_object(name, strict=strict)

    def allocate_vram_matrix(self, name: str, rows: int, cols: int) -> int:
        vram_addr = self.sub_matrix_manager.vram_allocator.allocate(size=rows * cols, name=name)
        self.sub_matrix_manager.add_vram_object(
            name=name,
            shape=(rows, cols),
            vram_addr=vram_addr,
            dtype="fp16",
            kind="VRAMMatrix",
            allocate_if_none=False,
        )
        self.generated_code += f"; Allocate VRAM Matrix {name}: ({rows}, {cols}) at VRAM[{vram_addr}]\n"
        return vram_addr

    def reset_mram(self) -> str:
        self.sub_matrix_manager.mram_allocator.reset()
        self.sub_matrix_manager.loaded_sub_blocks.clear()
        isa_code = "; === Reset MRAM ===\n"
        self.generated_code += isa_code
        return isa_code

    # ---------------------------------------------------------------------
    # Rewritten key execution functions (tile-first)
    # ---------------------------------------------------------------------

    def load_tile_from_hbm(
        self,
        hbm_addr: int,
        vram_addr: int,
        batch: int,
        hidden_size: int,
        hbm_stride: Optional[int] = None,
        hbm_scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
        vlen: int = 64,
        preload_len: int = 4,
    ) -> str:
        """Load a tile-aligned matrix region from HBM to VRAM (legacy ISA path)."""
        addr_reg = self.register_allocator.allocate_addr(1)[0]
        gp_regs_for_addr = self.register_allocator.allocate_gp(1)
        gp_regs_for_preload = self.register_allocator.allocate_gp(5)

        isa_code = ""
        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg],
            available_registers=gp_regs_for_addr,
            addr_reg_val=[hbm_addr],
        )
        isa_code += reset_reg_asm(alive_registers=gp_regs_for_preload)
        isa_code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=batch,
            hidden_size=hidden_size,
            alive_registers=gp_regs_for_preload,
            act_vram_offset=vram_addr,
            activation_offset_reg=addr_reg,
            stride_size=(hidden_size if hbm_stride is None else hbm_stride),
            scale_size=hbm_scale_size,
            hbm_start_offset=hbm_start_offset,
        )
        isa_code += self._reset_default_scale_stride_asm(
            gp_scale=gp_regs_for_preload[0],
            gp_stride=gp_regs_for_preload[1],
        )

        self.register_allocator.free_gp(gp_regs_for_addr)
        self.register_allocator.free_gp(gp_regs_for_preload)
        self.register_allocator.free_addr([addr_reg])
        return isa_code

    def store_tile_to_hbm(
        self,
        vram_addr: int,
        hbm_addr: int,
        batch: int,
        hidden_size: int,
        hbm_stride: Optional[int] = None,
        hbm_scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
        hbm_addr_reg: Optional[int] = None,
        vlen: int = 64,
        store_amount: int = 4,
    ) -> str:
        """Store a tile-aligned matrix region from VRAM to HBM (legacy ISA path)."""
        need_free_addr = False
        if hbm_addr_reg is None:
            hbm_addr_reg = self.register_allocator.allocate_addr(1)[0]
            need_free_addr = True

        gp_regs_for_addr = self.register_allocator.allocate_gp(2)
        gp_regs = self.register_allocator.allocate_gp(5)

        isa_code = ""
        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[hbm_addr_reg],
            available_registers=gp_regs_for_addr,
            addr_reg_val=[hbm_addr],
        )
        isa_code += store_act_asm(
            vlen=vlen,
            batch=batch,
            hidden_size=hidden_size,
            alive_registers=gp_regs,
            act_vram_offset=vram_addr,
            hbm_addr_reg=hbm_addr_reg,
            stride_size=(hidden_size if hbm_stride is None else hbm_stride),
            scale_size=hbm_scale_size,
            hbm_start_offset=hbm_start_offset,
            store_amount=store_amount,
        )
        isa_code += self._reset_default_scale_stride_asm(
            gp_scale=gp_regs[0],
            gp_stride=gp_regs[1],
        )

        self.register_allocator.free_gp(gp_regs_for_addr)
        self.register_allocator.free_gp(gp_regs)
        if need_free_addr:
            self.register_allocator.free_addr([hbm_addr_reg])
        return isa_code

    def add_tile(
        self,
        dst_matrix: str,
        dst_row_idx: int,
        dst_col_idx: int,
        src_matrix: str,
        src_row_idx: int,
        src_col_idx: int,
        target_matrix: Optional[str] = None,
        target_row_idx: Optional[int] = None,
        target_col_idx: Optional[int] = None,
    ) -> str:
        """Tile add: target[rt][ct] = dst[rd][cd] + src[rs][cs]."""
        if target_matrix is None:
            target_matrix = dst_matrix
        if target_row_idx is None:
            target_row_idx = dst_row_idx
        if target_col_idx is None:
            target_col_idx = dst_col_idx

        self.sub_matrix_manager.ensure_vram_layout(dst_matrix)
        self.sub_matrix_manager.ensure_vram_layout(src_matrix)
        self.sub_matrix_manager.ensure_vram_layout(target_matrix)

        dst_addr = self.sub_matrix_manager.get_vram_tile_addr(dst_matrix, dst_row_idx, dst_col_idx)
        src_addr = self.sub_matrix_manager.get_vram_tile_addr(src_matrix, src_row_idx, src_col_idx)
        tgt_addr = self.sub_matrix_manager.get_vram_tile_addr(
            target_matrix, target_row_idx, target_col_idx
        )

        gp_regs = self.register_allocator.allocate_gp(4)
        isa_code = (
            f"; add_tile {target_matrix}[{target_row_idx}][{target_col_idx}] = "
            f"{dst_matrix}[{dst_row_idx}][{dst_col_idx}] + {src_matrix}[{src_row_idx}][{src_col_idx}]\n"
        )
        isa_code += self.sub_matrix_manager.vram_block_add_tile_asm(
            src1_addr=dst_addr,
            src2_addr=src_addr,
            dst_addr=tgt_addr,
            gp_regs=gp_regs,
        )
        self.register_allocator.free_gp(gp_regs)
        self.generated_code += isa_code
        return isa_code

    # NOTE: kept as comments for ISA usage reference; methods are intentionally disabled for now.
    # def vram_sub_projection_to(
    #     self,
    #     vram_mat_name: str,
    #     vram_row_idx: int,
    #     mram_mat_name: str,
    #     mram_col_idx: int,
    #     target_matrix: str,
    #     target_row_idx: int,
    #     target_col_idx: int,
    # ) -> str:
    #     ...
    #
    # def vram_sub_projection_T_to(
    #     self,
    #     vram_mat_name: str,
    #     vram_row_idx: int,
    #     mram_mat_name: str,
    #     mram_row_idx: int,
    #     target_matrix: str,
    #     target_row_idx: int,
    #     target_col_idx: int,
    # ) -> str:
    #     ...
