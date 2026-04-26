"""ISAEmitter: turns prepared tile/FP operations into ISA strings.

Owns all `emit_*` methods (HBM/VRAM transfer, BTMM, matmul, FP kernels,
row operations, etc.). Managers and TileTensorProgram hold a reference
to an ISAEmitter and call its methods directly rather than going through
the program object.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from compiler.asm_templates import preload_addr_reg_asm

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403


class ISAEmitter:
    """Emit ISA strings for already-prepared tensor/FP operations."""

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program

    def emit_hbm_tile_to_mram(
        self,
        *,
        hbm_addr: int,
        mram_addr: int,
        hbm_offset: int = 0,
        hbm_scale: Optional[int] = None,
        hbm_stride: Optional[int] = None,
    ) -> None:
        addr_reg = self.program.compiler.register_allocator.allocate_addr(1)[0]
        gp_addr = self.program.compiler.register_allocator.allocate_gp(2)
        gp_exec = self.program.compiler.register_allocator.allocate_gp(3)
        gp_scale, gp_stride, gp_mram = gp_exec
        scale_val = self.program.tile_elems if hbm_scale is None else int(hbm_scale)
        stride_val = self.program.mlen if hbm_stride is None else int(hbm_stride)

        isa = ""
        isa += preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg],
            available_registers=gp_addr,
            addr_reg_val=[hbm_addr],
        )
        isa += f"S_ADDI_INT gp{gp_scale}, gp0, {scale_val}\n"
        isa += f"C_SET_SCALE_REG gp{gp_scale}\n"
        isa += f"S_ADDI_INT gp{gp_stride}, gp0, {stride_val}\n"
        isa += f"C_SET_STRIDE_REG gp{gp_stride}\n"
        isa += f"S_ADDI_INT gp{gp_mram}, gp0, {mram_addr}\n"
        isa += f"S_ADDI_INT gp{gp_scale}, gp0, {hbm_offset}\n"
        isa += f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{addr_reg}, 1, 0\n"
        isa += f"S_ADDI_INT gp{gp_scale}, gp0, {self.program.tile_elems}\n"
        isa += f"C_SET_SCALE_REG gp{gp_scale}\n"
        isa += f"S_ADDI_INT gp{gp_stride}, gp0, {self.program.mlen}\n"
        isa += f"C_SET_STRIDE_REG gp{gp_stride}\n"
        self.program.compiler.generated_code += isa

        self.program.compiler.register_allocator.free_gp(gp_addr)
        self.program.compiler.register_allocator.free_gp(gp_exec)
        self.program.compiler.register_allocator.free_addr([addr_reg])

    def emit_load_tile_from_hbm(
        self,
        *,
        hbm_addr: int,
        vram_addr: int,
        hbm_stride: Optional[int] = None,
        hbm_scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
    ) -> None:
        isa = self.program.compiler.load_tile_from_hbm(
            hbm_addr=hbm_addr,
            vram_addr=vram_addr,
            batch=self.program.mlen,
            hidden_size=self.program.mlen,
            hbm_stride=self.program.mlen if hbm_stride is None else int(hbm_stride),
            hbm_scale_size=self.program.tile_elems if hbm_scale_size is None else int(hbm_scale_size),
            hbm_start_offset=int(hbm_start_offset),
            vlen=self.program.mlen,
            preload_len=self.program.blen,
        )
        self.program.compiler.generated_code += isa

    def emit_store_tile_to_hbm(
        self,
        *,
        vram_addr: int,
        hbm_addr: int,
        hbm_stride: Optional[int] = None,
        hbm_scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
    ) -> None:
        isa = self.program.compiler.store_tile_to_hbm(
            vram_addr=vram_addr,
            hbm_addr=hbm_addr,
            batch=self.program.mlen,
            hidden_size=self.program.mlen,
            hbm_stride=self.program.mlen if hbm_stride is None else int(hbm_stride),
            hbm_scale_size=self.program.tile_elems if hbm_scale_size is None else int(hbm_scale_size),
            hbm_start_offset=int(hbm_start_offset),
            vlen=self.program.mlen,
            store_amount=self.program.blen,
        )
        self.program.compiler.generated_code += isa

    def emit_zero_vram_tile(self, vram_addr: int) -> None:
        gp_regs = self.program.compiler.register_allocator.allocate_gp(2)
        gp, gp_loop = gp_regs
        lines = [f"; zero tile vram[{vram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp}, gp0, {vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {self.program.mlen}")
        lines.append(f"V_MUL_VF gp{gp}, gp{gp}, f0, 0")
        lines.append(f"S_ADDI_INT gp{gp}, gp{gp}, {self.program.mlen}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_map_v_fp_tile(
        self,
        *,
        vram_addr: int,
        fpram_addr: int,
        row_count: int,
        row_width: int,
        task_id: str = "map_v_fp_tile",
    ) -> None:
        if row_count <= 0 or row_width <= 0:
            raise ValueError(f"emit_map_v_fp_tile expects positive row_count/row_width, got {row_count}/{row_width}")
        if row_width != self.program.mlen:
            raise ValueError(
                f"emit_map_v_fp_tile currently requires row_width == mlen == {self.program.mlen}, got {row_width}"
            )
        gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp_regs
        lines = [f"; map fp tile task {task_id} fpram[{fpram_addr}] -> vram[{vram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {fpram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
        lines.append(f"S_MAP_V_FP gp{gp_dst}, gp{gp_src}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_width}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_width}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_map_fp_v_tile(
        self,
        *,
        fpram_addr: int,
        vram_addr: int,
        row_count: int,
        row_width: int,
        task_id: str = "map_fp_v_tile",
    ) -> None:
        if row_count <= 0 or row_width <= 0:
            raise ValueError(f"emit_map_fp_v_tile expects positive row_count/row_width, got {row_count}/{row_width}")
        if row_width != self.program.mlen:
            raise ValueError(
                f"emit_map_fp_v_tile currently requires row_width == mlen == {self.program.mlen}, got {row_width}"
            )
        gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp_regs
        lines = [f"; map fp tile task {task_id} vram[{vram_addr}] -> fpram[{fpram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fpram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
        lines.append(f"S_MAP_FP_V gp{gp_dst}, gp{gp_src}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_width}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_width}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_btmm(
        self,
        *,
        lhs_packed_vram_addr: int,
        rhs_mram_addr: int,
        task_id: str = "btmm",
    ) -> None:
        gp_regs = self.program.compiler.register_allocator.allocate_gp(2)
        gp_mram_base, gp_lhs_base = gp_regs
        lines = [
            (
                f"; btmm task {task_id} lhs_packed=vram[{lhs_packed_vram_addr}] "
                f"rhs_mram={rhs_mram_addr} lanes={self.program.btmm_lane_count} head_width={self.program.btmm_hlen}"
            ),
            f"S_ADDI_INT gp{gp_mram_base}, gp0, {rhs_mram_addr}",
            f"S_ADDI_INT gp{gp_lhs_base}, gp0, {lhs_packed_vram_addr}",
            f"M_BTMM gp0, gp{gp_mram_base}, gp{gp_lhs_base}",
        ]
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        self.program.compiler.register_allocator.free_gp(gp_regs)

    def emit_btmm_wo(
        self,
        *,
        base_addr: int,
        tile_count: int,
        task_id: str = "btmm_wo",
    ) -> None:
        gp_out = self.program.compiler.register_allocator.allocate_gp(1)[0]
        lines = [
            (
                f"; btmm write-only task {task_id} out=vram[{base_addr}] "
                f"tiles={tile_count} lanes={self.program.btmm_lane_count} head_width={self.program.btmm_hlen}"
            ),
            f"S_ADDI_INT gp{gp_out}, gp0, {base_addr}",
            f"M_BMM_WO gp{gp_out}, 0",
        ]
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        self.program.compiler.register_allocator.free_gp([gp_out])

    def emit_matmul(
        self,
        *,
        lhs_vram_addrs: Sequence[int],
        rhs_mram_addrs: Sequence[int],
        dst_vram_addr: int,
        task_id: str = "matmul",
        zero_dst: bool = False,
    ) -> None:
        if len(lhs_vram_addrs) != len(rhs_mram_addrs):
            raise ValueError("lhs_vram_addrs and rhs_mram_addrs must have equal lengths")
        if zero_dst:
            self.emit_zero_vram_tile(dst_vram_addr)

        gp_regs = self.program.compiler.register_allocator.allocate_gp(5)
        gp_act, gp_mat, gp_out, gp_stride, gp_loop = gp_regs
        tiles_per_mlen = self.program.mlen // self.program.blen
        lines = [f"; matmul task {task_id}"]
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")
        lhs_prog = self.program._arith_progression([int(addr) for addr in lhs_vram_addrs])
        rhs_prog = self.program._arith_progression([int(addr) for addr in rhs_mram_addrs])

        for oc in range(tiles_per_mlen):
            for orow in range(tiles_per_mlen):
                if lhs_prog is not None and rhs_prog is not None:
                    lhs_start, pair_count, lhs_step = lhs_prog
                    rhs_start, _, rhs_step = rhs_prog
                    act_addr = lhs_start + orow * self.program.blen * self.program.mlen
                    mat_addr = rhs_start + oc * self.program.blen
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
                    lines.append(f"C_LOOP_START gp{gp_loop}, {pair_count}")
                    lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {lhs_step}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp{gp_mat}, {rhs_step}")
                    lines.append(f"C_LOOP_END gp{gp_loop}")
                else:
                    for lhs_addr, rhs_addr in zip(lhs_vram_addrs, rhs_mram_addrs):
                        act_addr = lhs_addr + orow * self.program.blen * self.program.mlen
                        mat_addr = rhs_addr + oc * self.program.blen
                        lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                        lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
                        lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                out_addr = dst_vram_addr + orow * self.program.blen * self.program.mlen + oc * self.program.blen
                lines.append(f"S_ADDI_INT gp{gp_out}, gp0, {out_addr}")
                lines.append(f"M_MM_WO gp{gp_out}, gp0, 0")

        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_slot_matmul(
        self,
        *,
        lhs_vram_addr: int,
        rhs_mram_addr: int,
        rhs_col_offset: int,
        dst_vram_addr: int,
        dst_col_offset: int,
        col_count: int,
        task_id: str = "slot_matmul",
        zero_dst: bool = False,
    ) -> None:
        if col_count <= 0:
            raise ValueError("emit_slot_matmul requires one positive col_count")
        if col_count % self.program.blen != 0:
            raise ValueError(
                f"emit_slot_matmul requires col_count divisible by blen={self.program.blen}, got {col_count}"
            )
        if zero_dst:
            self.emit_zero_vram_tile(dst_vram_addr)

        gp_regs = self.program.compiler.register_allocator.allocate_gp(5)
        gp_act, gp_mat, gp_out, gp_stride, gp_loop = gp_regs
        tiles_per_mlen = self.program.mlen // self.program.blen
        tiles_per_slot = col_count // self.program.blen
        lines = [f"; slot matmul task {task_id}"]
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for oc in range(tiles_per_slot):
            act_addr = lhs_vram_addr
            mat_addr = rhs_mram_addr + rhs_col_offset + oc * self.program.blen
            out_addr = dst_vram_addr + dst_col_offset + oc * self.program.blen
            lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
            lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
            lines.append(f"S_ADDI_INT gp{gp_out}, gp0, {out_addr}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {tiles_per_mlen}")
            lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
            lines.append(f"M_MM_WO gp{gp_out}, gp0, 0")
            lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {self.program.blen * self.program.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_out}, gp{gp_out}, {self.program.blen * self.program.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")

        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_tile_binary(
        self,
        *,
        lhs_vram_addr: int,
        rhs_vram_addr: int,
        dst_vram_addr: int,
        op: str = "add",
        task_id: str = "tile_binary",
    ) -> None:
        op_to_insn = {
            "add": "V_ADD_VV",
            "sub": "V_SUB_VV",
            "mul": "V_MUL_VV",
        }
        if op not in op_to_insn:
            raise ValueError(f"Unsupported tile binary op={op!r}")
        gp_regs = self.program.compiler.register_allocator.allocate_gp(4)
        gp_dst, gp_lhs, gp_rhs, gp_loop = gp_regs
        lines = [f"; tile binary task {task_id} op={op}"]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_lhs}, gp0, {lhs_vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_rhs}, gp0, {rhs_vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {self.program.mlen}")
        if op == "sub":
            lines.append(f"{op_to_insn[op]} gp{gp_dst}, gp{gp_rhs}, gp{gp_lhs}, 0")
        else:
            lines.append(f"{op_to_insn[op]} gp{gp_dst}, gp{gp_lhs}, gp{gp_rhs}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.program.mlen}")
        lines.append(f"S_ADDI_INT gp{gp_lhs}, gp{gp_lhs}, {self.program.mlen}")
        lines.append(f"S_ADDI_INT gp{gp_rhs}, gp{gp_rhs}, {self.program.mlen}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_tile_add(
        self,
        *,
        lhs_vram_addr: int,
        rhs_vram_addr: int,
        dst_vram_addr: int,
        task_id: str = "tile_add",
    ) -> None:
        self.emit_tile_binary(
            lhs_vram_addr=lhs_vram_addr,
            rhs_vram_addr=rhs_vram_addr,
            dst_vram_addr=dst_vram_addr,
            op="add",
            task_id=task_id,
        )

    def emit_fp_kernel(
        self,
        *,
        src1_addrs: Sequence[int],
        dst_addrs: Sequence[int],
        src2_addrs: Optional[Sequence[int]] = None,
        op: str,
        task_id: str = "fp_kernel",
    ) -> None:
        unary_copy = {"copy", "fill"}
        unary_math = {"exp": "S_EXP_FP", "reci": "S_RECI_FP", "sqrt": "S_SQRT_FP"}
        binary_math = {"add": "S_ADD_FP", "sub": "S_SUB_FP", "mul": "S_MUL_FP", "max": "S_MAX_FP"}
        if len(src1_addrs) != len(dst_addrs):
            raise ValueError("emit_fp_kernel expects matched src1/dst lengths")
        if src2_addrs is not None and len(src2_addrs) != len(dst_addrs):
            raise ValueError("emit_fp_kernel expects matched src2/dst lengths")
        if op in unary_copy:
            gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
            gp_src, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src_prog = self.program._arith_progression([int(addr) for addr in src1_addrs])
            dst_prog = self.program._arith_progression([int(addr) for addr in dst_addrs])
            if src_prog is not None and dst_prog is not None:
                src_start, count, src_step = src_prog
                dst_start, _, dst_step = dst_prog
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_start}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {src_step}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {dst_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for src_addr, dst_addr in zip(src1_addrs, dst_addrs):
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {int(src_addr)}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                    lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            self.program.compiler.register_allocator.free_gp(gp_regs)
            self.program.compiler.generated_code += "\n".join(lines) + "\n"
            return
        if op in unary_math:
            gp_regs = self.program.compiler.register_allocator.allocate_gp(3)
            gp_src, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src_prog = self.program._arith_progression([int(addr) for addr in src1_addrs])
            dst_prog = self.program._arith_progression([int(addr) for addr in dst_addrs])
            if src_prog is not None and dst_prog is not None:
                src_start, count, src_step = src_prog
                dst_start, _, dst_step = dst_prog
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_start}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                if op in {"exp", "reci"}:
                    lines.append(f"{unary_math[op]} f1, f1, 0")
                else:
                    lines.append(f"{unary_math[op]} f1, f1")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {src_step}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {dst_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for src_addr, dst_addr in zip(src1_addrs, dst_addrs):
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {int(src_addr)}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
                    if op in {"exp", "reci"}:
                        lines.append(f"{unary_math[op]} f1, f1, 0")
                    else:
                        lines.append(f"{unary_math[op]} f1, f1")
                    lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            self.program.compiler.register_allocator.free_gp(gp_regs)
            self.program.compiler.generated_code += "\n".join(lines) + "\n"
            return
        if op in binary_math:
            if src2_addrs is None:
                raise ValueError(f"emit_fp_kernel op={op!r} requires src2_addrs")
            gp_regs = self.program.compiler.register_allocator.allocate_gp(4)
            gp_a, gp_b, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src1_prog = self.program._arith_progression([int(addr) for addr in src1_addrs])
            src2_prog = self.program._arith_progression([int(addr) for addr in src2_addrs])
            dst_prog = self.program._arith_progression([int(addr) for addr in dst_addrs])
            if src1_prog is not None and src2_prog is not None and dst_prog is not None:
                src1_start, count, src1_step = src1_prog
                src2_start, _, src2_step = src2_prog
                dst_start, _, dst_step = dst_prog
                lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_start}")
                lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_start}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
                lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
                lines.append(f"{binary_math[op]} f1, f1, f2")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
                lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, {src1_step}")
                lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, {src2_step}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {dst_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for src1_addr, src2_addr, dst_addr in zip(src1_addrs, src2_addrs, dst_addrs):
                    lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {int(src1_addr)}")
                    lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {int(src2_addr)}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
                    lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
                    lines.append(f"{binary_math[op]} f1, f1, f2")
                    lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            self.program.compiler.register_allocator.free_gp(gp_regs)
            self.program.compiler.generated_code += "\n".join(lines) + "\n"
            return
        raise ValueError(f"Unsupported emit_fp_kernel op={op!r}")

    def emit_row_operation(
        self,
        *,
        src_vram_addr: int,
        dst_vram_addr: Optional[int] = None,
        op: str,
        row_count: int,
        dst_addrs: Optional[Sequence[int]] = None,
        rhs_addrs: Optional[Sequence[int]] = None,
        mask_val: Optional[int] = None,
        task_id: str = "row_operations",
    ) -> None:
        if row_count <= 0:
            return
        unary_ops = {"exp", "reci"}
        reduce_ops = {"reduce_max": "V_RED_MAX", "reduce_sum": "V_RED_SUM"}
        binary_ops = {"mul": "V_MUL_VF", "add": "V_ADD_VF", "sub": "V_SUB_VF"}
        if op not in unary_ops | set(reduce_ops) | set(binary_ops):
            raise ValueError(f"Unsupported emit_row_operation op={op!r}")

        gp_regs = self.program.compiler.register_allocator.allocate_gp(5)
        gp_src, gp_fp, gp_dst, gp_loop, gp_mask = gp_regs
        lines = [f"; row operation task {task_id} op={op} rows={row_count}"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {int(src_vram_addr)}")
        dst_vram_addr = int(src_vram_addr if dst_vram_addr is None else dst_vram_addr)
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_vram_addr}")
        use_mask = mask_val is not None
        if use_mask:
            lines.append(f"; row operation mask {int(mask_val)}")
            lines.append(f"S_ADDI_INT gp{gp_mask}, gp0, {int(mask_val)}")
            lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")

        if op in unary_ops:
            lines.append(f"C_LOOP_START gp{gp_loop}, {int(row_count)}")
            if op == "exp":
                lines.append(f"V_EXP_V gp{gp_dst}, gp{gp_src}, {1 if use_mask else 0}")
            else:
                lines.append(f"V_RECI_V gp{gp_dst}, gp{gp_src}, {1 if use_mask else 0}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.program.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.program.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        elif op in reduce_ops:
            if dst_addrs is None or len(dst_addrs) != row_count:
                raise ValueError(f"emit_row_operation op={op!r} expects one dst fp addr per row")
            dst_prog = self.program._arith_progression([int(addr) for addr in dst_addrs])
            if dst_prog is None:
                for row_index, dst_addr in enumerate(dst_addrs):
                    row_addr = int(src_vram_addr) + row_index * self.program.mlen
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_dst}, 0")
                    lines.append(f"{reduce_ops[op]} f1, gp{gp_src}, {1 if use_mask else 0}")
                    lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            else:
                dst_start, count, dst_step = dst_prog
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_dst}, 0")
                lines.append(f"{reduce_ops[op]} f1, gp{gp_src}, {1 if use_mask else 0}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.program.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {dst_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            if rhs_addrs is None or len(rhs_addrs) not in (1, row_count):
                raise ValueError(f"emit_row_operation op={op!r} expects one rhs fp addr or one per row")
            rhs_prog = self.program._arith_progression([int(addr) for addr in rhs_addrs]) if len(rhs_addrs) > 1 else None
            if len(rhs_addrs) == 1:
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {int(rhs_addrs[0])}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {int(row_count)}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                if op == "sub":
                    lines.append(f"V_SUB_VF gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                else:
                    lines.append(f"{binary_ops[op]} gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.program.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.program.mlen}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            elif rhs_prog is not None:
                rhs_start, count, rhs_step = rhs_prog
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {rhs_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                if op == "sub":
                    lines.append(f"V_SUB_VF gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                else:
                    lines.append(f"{binary_ops[op]} gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.program.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.program.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {rhs_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for row_index, rhs_addr in enumerate(rhs_addrs):
                    row_addr = int(src_vram_addr) + row_index * self.program.mlen
                    dst_row_addr = dst_vram_addr + row_index * self.program.mlen
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_row_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {int(rhs_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                    if op == "sub":
                        lines.append(f"V_SUB_VF gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                    else:
                        lines.append(f"{binary_ops[op]} gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}")

        if use_mask:
            lines.append("S_ADDI_INT gp{0}, gp0, 0".format(gp_mask))
            lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")

        self.program.compiler.register_allocator.free_gp(gp_regs)
        self.program.compiler.generated_code += "\n".join(lines) + "\n"

