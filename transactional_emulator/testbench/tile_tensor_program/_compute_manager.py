"""ComputeManager: validates operands, ensures residency, emits ISA."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403


class ComputeManager:
    """Execute already-prepared tensor/FP operations and emit ISA.

    ComputeManager should not invent binding policy. It assumes the write path
    has already been prepared by ValueManager and mainly does:

    - ensure operands in the correct place close to use
    - validate lane/view layout for execution kernels
    - emit ISA
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.isa_emitter = program.isa_emitter
        self.ops: List[Dict[str, object]] = []

    def execute(self, signal: List[object]) -> Dict[str, object]:
        operands, op_kind = signal
        record = {"op_kind": op_kind, "operands": operands}
        self.ops.append(record)
        if op_kind == "matmul":
            return self._execute_matmul(operands)
        return {
            "op_kind": op_kind,
            "inputs": operands,
            "outputs": operands.get("outputs", []) if isinstance(operands, dict) else operands,
        }

    def _execute_matmul(self, operands: object) -> Dict[str, object]:
        if not isinstance(operands, tuple) or len(operands) != 4 or operands[0] != "matmul":
            raise RuntimeError("matmul execute expects ('matmul', src_pairs, dst_value, dst_tile)")
        _, src_pairs, dst_value, _ = operands
        if not isinstance(dst_value, ValueTile):
            raise RuntimeError("matmul execute expects one destination ValueTile")

        lhs_vram_addrs: List[int] = []
        rhs_mram_addrs: List[int] = []
        for pair in src_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            lhs_value, rhs_value = pair
            if not isinstance(lhs_value, ValueTile) or not isinstance(rhs_value, ValueTile):
                raise RuntimeError("matmul execute expects ValueTile sources")
            self.program.value_manager.ensure_value_tile_in_place(lhs_value, "vram")
            self.program.value_manager.ensure_value_tile_in_place(rhs_value, "mram")
            lhs_vram_addr = lhs_value.residency.get("vram_addr")
            rhs_mram_addr = rhs_value.residency.get("mram_addr")
            if lhs_vram_addr is None:
                raise RuntimeError(f"matmul execute requires lhs value in VRAM: {lhs_value.value_tile_id}")
            if rhs_mram_addr is None:
                raise RuntimeError(f"matmul execute requires rhs value in MRAM: {rhs_value.value_tile_id}")
            lhs_vram_addrs.append(int(lhs_vram_addr))
            rhs_mram_addrs.append(int(rhs_mram_addr))

        self.program.value_manager.ensure_value_tile_in_place(dst_value, "vram")
        dst_vram_addr = dst_value.residency.get("vram_addr")
        if dst_vram_addr is None:
            raise RuntimeError(f"matmul execute requires dst value in VRAM: {dst_value.value_tile_id}")

        task_id = self._matmul_task_id_from_value(dst_value)
        self.isa_emitter.emit_matmul(
            lhs_vram_addrs=lhs_vram_addrs,
            rhs_mram_addrs=rhs_mram_addrs,
            dst_vram_addr=int(dst_vram_addr),
            task_id=task_id,
            zero_dst=True,
        )
        return {
            "op_kind": "matmul",
            "inputs": operands,
            "outputs": [dst_value],
            "dst": dst_value,
            "task_id": task_id,
        }

    def view_matmul(
        self,
        lhs_values: List[ValueTile],
        rhs_tile: TensorTile | InputTile,
        dst_tile: TensorTile | InputTile,
        dst_value: ValueTile,
        *,
        task_id: str,
        zero_dst: bool,
    ) -> Dict[str, object]:
        if not lhs_values:
            raise RuntimeError("view_matmul expects one non-empty lhs ValueTile list")
        if not all(isinstance(value, ValueTile) for value in lhs_values):
            raise RuntimeError("view_matmul expects lhs_values to contain ValueTile objects only")
        rhs_views = self.program.value_manager._tile_compute_views(rhs_tile)
        dst_views = self.program.value_manager._tile_compute_views(dst_tile)
        if not rhs_views or not dst_views:
            raise RuntimeError("view_matmul expects non-empty rhs/dst view lanes")
        rhs_value = self.program.value_manager.value_tiles.get(rhs_views[0].backing_value_tile_id)
        if not isinstance(rhs_value, ValueTile):
            raise RuntimeError("view_matmul requires rhs backing value")

        for lhs_value in lhs_values:
            self.program.value_manager.ensure_value_tile_in_place(lhs_value, "vram")
        self.program.value_manager.ensure_value_tile_in_place(rhs_value, "mram")
        self.program.value_manager.ensure_value_tile_in_place(dst_value, "vram")

        rhs_mram_addr = rhs_value.residency.get("mram_addr")
        dst_vram_addr = dst_value.residency.get("vram_addr")
        lhs_vram_addrs = [value.residency.get("vram_addr") for value in lhs_values]
        if rhs_mram_addr is None or dst_vram_addr is None or any(addr is None for addr in lhs_vram_addrs):
            raise RuntimeError("view_matmul requires lhs in VRAM, rhs in MRAM, dst in VRAM")
        if len(rhs_views) != len(dst_views):
            raise RuntimeError(
                f"view_matmul requires matching rhs/dst slot counts, got rhs={len(rhs_views)} dst={len(dst_views)}"
            )
        if len(lhs_values) != len(rhs_views):
            raise RuntimeError(
                f"view_matmul requires lhs_values to align with lanes, got lhs={len(lhs_values)} slots={len(rhs_views)}"
            )

        lane_logs: List[Dict[str, object]] = []
        for lane_index, (lhs_addr, rhs_view, dst_view, lhs_value) in enumerate(
            zip(lhs_vram_addrs, rhs_views, dst_views, lhs_values)
        ):
            if lhs_addr is None:
                raise RuntimeError(f"view_matmul lane {lane_index} is missing one lhs VRAM address")
            if rhs_view.col_count != dst_view.col_count:
                raise RuntimeError(
                    f"view_matmul lane {lane_index} slot width mismatch rhs={rhs_view.col_count} dst={dst_view.col_count}"
                )
            self.isa_emitter.emit_slot_matmul(
                lhs_vram_addr=int(lhs_addr),
                rhs_mram_addr=int(rhs_mram_addr),
                rhs_col_offset=int(rhs_view.col_offset),
                dst_vram_addr=int(dst_vram_addr),
                dst_col_offset=int(dst_view.col_offset),
                col_count=int(rhs_view.col_count),
                task_id=f"{task_id}.lane{lane_index}",
                zero_dst=(zero_dst and lane_index == 0),
            )
            lane_logs.append(
                {
                    "lane_index": lane_index,
                    "lhs_value": lhs_value.value_tile_id,
                    "lhs_vram_addr": int(lhs_addr),
                    "rhs_view": rhs_view.view_id,
                    "rhs_col_offset": int(rhs_view.col_offset),
                    "dst_view": dst_view.view_id,
                    "dst_col_offset": int(dst_view.col_offset),
                    "col_count": int(rhs_view.col_count),
                }
            )
        return {
            "op_kind": "view_matmul",
            "inputs": [lhs_values, rhs_tile, dst_tile],
            "outputs": [dst_value],
            "dst": dst_value,
            "task_id": task_id,
            "lane_logs": lane_logs,
        }

    def btmm(
        self,
        *,
        lhs_packed_value: ValueTile,
        rhs_value: ValueTile,
        task_id: str = "btmm",
    ) -> Dict[str, object]:
        self.program.value_manager.ensure_value_tile_in_place(lhs_packed_value, "vram")
        self.program.value_manager.ensure_value_tile_in_place(rhs_value, "mram")

        lhs_vram_addr = lhs_packed_value.residency.get("vram_addr")
        rhs_mram_addr = rhs_value.residency.get("mram_addr")
        if lhs_vram_addr is None or rhs_mram_addr is None:
            raise RuntimeError("btmm requires lhs_packed_value in VRAM and rhs_value in MRAM")

        self.isa_emitter.emit_btmm(
            lhs_packed_vram_addr=int(lhs_vram_addr),
            rhs_mram_addr=int(rhs_mram_addr),
            task_id=task_id,
        )
        return {
            "op_kind": "btmm",
            "lhs": lhs_packed_value,
            "rhs": rhs_value,
            "btmm_finished": True,
            "task_id": task_id,
        }

    def btmm_write(
        self,
        *,
        btmm_state: Dict[str, object],
        tile_count: Optional[int] = None,
        reason: str = "btmm_write",
        logical_shape: Optional[Tuple[int, int]] = None,
        metadata: Optional[Dict[str, object]] = None,
        task_id: str = "btmm_wo",
    ) -> Dict[str, object]:
        if not btmm_state.get("btmm_finished"):
            raise RuntimeError("btmm_write requires btmm_state.btmm_finished == True")

        resolved_tile_count = self.program.btmm_lane_count if tile_count is None else int(tile_count)
        if resolved_tile_count <= 0:
            raise ValueError(f"btmm_write requires one positive tile_count, got {resolved_tile_count}")

        out_values, base_addr = self.program.value_manager.allocate_contiguous_vram_value_tiles(
            tile_count=resolved_tile_count,
            logical_shape=logical_shape if logical_shape is not None else (self.program.mlen, self.program.mlen),
            metadata=metadata,
            reason=reason,
        )
        self.isa_emitter.emit_btmm_wo(
            base_addr=base_addr,
            tile_count=resolved_tile_count,
            task_id=task_id,
        )
        return {
            "op_kind": "btmm_wo",
            "btmm_state": btmm_state,
            "dst_values": out_values,
            "base_addr": base_addr,
            "tile_count": resolved_tile_count,
            "task_id": task_id,
        }

    def _matmul_task_id_from_value(self, value: ValueTile) -> str:
        source_tile_id = value.metadata.get("source_tile_id")
        if not isinstance(source_tile_id, str):
            return f"matmul.{value.value_tile_id}"
        dst_tile = self.program.tensor_manager.tensor_tiles.get(source_tile_id)
        if dst_tile is None:
            return f"matmul.{value.value_tile_id}"
        row_block, col_block = dst_tile.coord
        return f"matmul.r{row_block}.c{col_block}"

    def fp_kernel(
        self,
        src1: Sequence[FPVar],
        dst: Sequence[FPVar],
        *,
        src2: Optional[Sequence[FPVar]] = None,
        op: str = "add",
        task_id: str = "fp_kernel",
    ) -> Dict[str, object]:
        unary_ops = {"copy", "fill", "exp", "reci", "sqrt"}
        binary_ops = {"add", "sub", "mul", "max"}
        valid_ops = unary_ops | binary_ops
        if op not in valid_ops:
            raise ValueError(f"Unsupported fp_kernel op {op!r}; expected one of {sorted(valid_ops)}")
        if op in binary_ops and src2 is None:
            raise ValueError(f"Binary fp_kernel op {op!r} requires src2")
        if op in unary_ops and src2 is not None:
            raise ValueError(f"Unary fp_kernel op {op!r} does not accept src2")

        src1_vars = list(src1)
        dst_vars = list(dst)
        src2_vars = list(src2) if src2 is not None else None
        if len(src1_vars) != len(dst_vars):
            if op in {"copy", "fill"} and len(src1_vars) == 1 and len(dst_vars) > 1:
                src1_vars = src1_vars * len(dst_vars)
            else:
                raise ValueError(f"fp_kernel expects matched src1/dst lengths, got {len(src1_vars)} vs {len(dst_vars)}")
        if src2_vars is not None and len(src2_vars) != len(dst_vars):
            raise ValueError(f"fp_kernel expects matched src2/dst lengths, got {len(src2_vars)} vs {len(dst_vars)}")

        self.isa_emitter.emit_fp_kernel(
            src1_addrs=[_require_fp_addr(var) for var in src1_vars],
            dst_addrs=[_require_fp_addr(var) for var in dst_vars],
            src2_addrs=[_require_fp_addr(var) for var in src2_vars] if src2_vars is not None else None,
            op=op,
            task_id=task_id,
        )
        record = {
            "op_kind": "fp_kernel",
            "task_id": task_id,
            "op": op,
            "src1": [var.name for var in src1_vars],
            "src2": [var.name for var in src2_vars] if src2_vars is not None else None,
            "dst": [var.name for var in dst_vars],
        }
        self.ops.append(record)
        return record

    def pure_fp_compute(
        self,
        src1: Sequence[FPVar],
        dst: Sequence[FPVar],
        *,
        src2: Optional[Sequence[FPVar]] = None,
        op: str = "add",
        task_id: str = "pure_fp_compute",
    ) -> Dict[str, object]:
        return self.fp_kernel(src1, dst, src2=src2, op=op, task_id=task_id)

    def row_operations(
        self,
        src: RowOperandLike,
        *,
        dst_operand: Optional[RowOperandLike] = None,
        dst: Optional[Sequence[FPVar]] = None,
        rhs: Optional[Sequence[FPVar]] = None,
        op: str,
        task_id: str = "row_operations",
    ) -> Dict[str, object]:
        if isinstance(src, ValueTileView):
            backing_value = self.program.value_manager.value_tiles.get(src.backing_value_tile_id)
            if not isinstance(backing_value, ValueTile):
                raise RuntimeError(f"row_operations view source is missing backing value: {src.view_id}")
            self.program.value_manager.ensure_value_tile_in_place(backing_value, "vram")
            src_vram_addr = backing_value.residency.get("vram_addr")
            row_count = int(src.row_count)
            mask_unit = int(self.program.btmm_hlen)
            col_offset = int(src.col_offset)
            col_count = int(src.col_count)
            if mask_unit <= 0:
                raise RuntimeError(f"row_operations requires positive mask_unit, got {mask_unit}")
            if col_offset % mask_unit != 0 or col_count % mask_unit != 0:
                raise RuntimeError(
                    f"row_operations view mask expects col_offset/col_count aligned to mask_unit={mask_unit}, "
                    f"got col_offset={col_offset} col_count={col_count}"
                )
            lane_start = col_offset // mask_unit
            lane_count = col_count // mask_unit
            mask_val = ((1 << lane_count) - 1) << lane_start
            src_name = src.view_id
        else:
            self.program.value_manager.ensure_value_tile_in_place(src, "vram")
            src_vram_addr = src.residency.get("vram_addr")
            row_count = int(src.logical_shape[0])
            mask_val = None
            src_name = src.value_tile_id
        if src_vram_addr is None:
            raise RuntimeError(f"row_operations requires src in VRAM: {src_name}")
        if dst_operand is None:
            dst_operand = src
        if isinstance(dst_operand, ValueTileView):
            dst_backing_value = self.program.value_manager.value_tiles.get(dst_operand.backing_value_tile_id)
            if not isinstance(dst_backing_value, ValueTile):
                raise RuntimeError(f"row_operations view destination is missing backing value: {dst_operand.view_id}")
            self.program.value_manager.ensure_value_tile_in_place(dst_backing_value, "vram")
            dst_vram_addr = dst_backing_value.residency.get("vram_addr")
        else:
            self.program.value_manager.ensure_value_tile_in_place(dst_operand, "vram")
            dst_vram_addr = dst_operand.residency.get("vram_addr")
        if dst_vram_addr is None:
            raise RuntimeError(f"row_operations requires dst in VRAM: {task_id}")

        dst_addrs = [_require_fp_addr(var) for var in dst] if dst is not None else None
        rhs_addrs = [_require_fp_addr(var) for var in rhs] if rhs is not None else None
        self.isa_emitter.emit_row_operation(
            src_vram_addr=int(src_vram_addr),
            dst_vram_addr=int(dst_vram_addr),
            dst_addrs=dst_addrs,
            rhs_addrs=rhs_addrs,
            row_count=row_count,
            mask_val=mask_val,
            op=op,
            task_id=task_id,
        )
        record = {
            "op_kind": "row_operations",
            "task_id": task_id,
            "op": op,
            "src": src_name,
            "dst_operand": getattr(dst_operand, "view_id", getattr(dst_operand, "value_tile_id", None)),
            "dst": [var.name for var in dst] if dst is not None else None,
            "rhs": [var.name for var in rhs] if rhs is not None else None,
            "mask_val": mask_val,
        }
        self.ops.append(record)
        return record


