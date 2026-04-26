"""TileTensorProgram: top-level user-facing program builder."""

from __future__ import annotations

import inspect
import sys
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from compiler.asm_templates import preload_addr_reg_asm
from tiled_developer_compiler import TiledDeveloperCompiler
from operation_report_delta import build_delta_report, parse_operation_report

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403
from ._hardware_manager import HardwareManager
from ._isa_emitter import ISAEmitter
from ._thread_manager import ThreadManager, _LoopHintRange
from ._value_manager import ValueManager
from ._tensor_manager import TensorManager
from ._compute_manager import ComputeManager


class TileTensorProgram:
    """User-facing program builder over the logical/value/compute pipeline.

    This class exposes the testbench authoring API (`input`, `tensor`, `copy`,
    `matmul`, `atomic_add`, FP helpers, reporting, and compile hooks) while
    delegating the real work to TensorManager, ValueManager, and ComputeManager.

    In practice it acts as the orchestration layer for the current runtime law:

        mapt -> mapv -> compute -> mapv_back -> mapt_back

    The important modern write-path rule is:

        resolve view -> prepare_updated_view_value -> compute -> bind/writeback

    FP-domain operations are intentionally separate from the tensor
    value/view pipeline.
    """

    def __init__(
        self,
        *,
        mlen: int,
        blen: int,
        btmm_hlen: Optional[int] = None,
        real_data_ratio: float = 1.0,
        vram_tile_capacity: int = 0,
        mram_tile_capacity: int = 0,
        fpram_capacity: int = 0,
        hbm_base_addr: int = 0,
    ) -> None:
        self.mlen = int(mlen)
        self.blen = int(blen)
        self.btmm_hlen = int(btmm_hlen) if btmm_hlen is not None else (self.mlen // self.blen)
        if self.btmm_hlen <= 0 or self.mlen % self.btmm_hlen != 0:
            raise ValueError(
                f"Invalid btmm_hlen={self.btmm_hlen}; require positive divisor of mlen={self.mlen}"
            )
        self.btmm_lane_count = self.mlen // self.btmm_hlen
        self.real_data_ratio = float(real_data_ratio)
        self.vram_tile_capacity = int(vram_tile_capacity)
        self.mram_tile_capacity = int(mram_tile_capacity)
        self.fpram_capacity = int(fpram_capacity)
        self.tile_elems = self.mlen * self.mlen
        self._next_hbm_addr = int(hbm_base_addr)

        self.compiler = TiledDeveloperCompiler(
            mlen=self.mlen,
            blen=self.blen,
            fpram_total_size=(self.fpram_capacity or 1024),
        )
        self.hardware = HardwareManager(self)
        self.isa_emitter = ISAEmitter(self)
        self.thread_manager = ThreadManager(self)
        self.value_manager = ValueManager(self)
        self.tensor_manager = TensorManager(self)
        self.compute_manager = ComputeManager(self)
        self._auto_name_counters: Dict[str, int] = {}
        self.loop_hints: List[Dict[str, int | str]] = []
        self.operation_snapshots: List[Dict[str, object]] = []
        self._active_parallel_region_ids: List[int] = []
        self._parallel_region_counter = 0
        self._parallel_snapshot_keys: set[Tuple[int, str, int, str]] = set()
        self._parallel_execution_lowered = False

    def input(self, name: str, logical_shape: LogicalShape, *, hbm_addr: Optional[int] = None) -> Input:
        return self.tensor_manager.input(name, logical_shape, hbm_addr=hbm_addr)

    def tensor(self, name: str, logical_shape: LogicalShape) -> Tensor | Vector:
        tensor = self.tensor_manager.tensor(name, logical_shape)
        if isinstance(tensor, Vector):
            self._initialize_vector_backing(tensor)
        return tensor

    def vector(self, name: str, logical_shape: LogicalShape) -> Vector:
        vector = self.tensor_manager.vector(name, logical_shape)
        self._initialize_vector_backing(vector)
        return vector

    def parallel_region3d(
        self,
        extents: Tuple[int, int, int] | List[int],
        *,
        name: Optional[str] = None,
    ) -> _ParallelRegionScope:
        return self.thread_manager.parallel_region3d(extents, name=name)

    def parallel_region2d(
        self,
        extents: Tuple[int, int] | List[int],
        *,
        name: Optional[str] = None,
    ) -> _ParallelRegion2DScope:
        return self.thread_manager.parallel_region2d(extents, name=name)

    def where(self, predicate: object, on_true: object, on_false: object) -> ParallelExpr:
        return self.thread_manager.where(predicate, on_true, on_false)

    def if_then_else(self, predicate: object, on_true: object, on_false: object) -> ParallelExpr:
        return self.thread_manager.if_then_else(predicate, on_true, on_false)

    def max(self, lhs: object, rhs: object) -> ParallelExpr:
        return ParallelExpr(
            kind="op",
            op="max",
            args=(_coerce_parallel_expr(lhs), _coerce_parallel_expr(rhs)),
        )

    def exp(self, operand: object) -> ParallelExpr:
        return ParallelExpr(kind="unary_op", op="exp", args=(_coerce_parallel_expr(operand),))

    def reci(self, operand: object) -> ParallelExpr:
        return ParallelExpr(kind="unary_op", op="reci", args=(_coerce_parallel_expr(operand),))

    def sqrt(self, operand: object) -> ParallelExpr:
        return ParallelExpr(kind="unary_op", op="sqrt", args=(_coerce_parallel_expr(operand),))

    def pair(self, axis: object) -> ParallelExpr:
        return self.thread_manager.pair(axis)

    def half_index(self, axis: object) -> ParallelExpr:
        return self.thread_manager.half_index(axis)

    def parallel_execution_plans(self) -> List[ParallelExecutionPlan]:
        return self.thread_manager.parallel_execution_plans()

    def lower_parallel_execution_plans(self) -> None:
        self.thread_manager.lower_parallel_execution_plans()


    def _auto_name(self, prefix: str) -> str:
        count = self._auto_name_counters.get(prefix, 0)
        self._auto_name_counters[prefix] = count + 1
        return f"{prefix}.{count}"

    def fp_var(self, name: str, value: float = 0.0, size: int = 1) -> FPVar | FPFragment:
        return self.tensor_manager.fp_var(name, value=value, size=size)

    def fp_fragment(self, name: str, shape: Tuple[int, ...] | int, *, init: float = 0.0) -> FPFragment:
        return self.tensor_manager.fp_fragment(name=name, shape=shape, init=init)

    def alloc_fragment(
        self,
        name: str,
        logical_shape: LogicalShape,
        *,
        init_zero: bool = False,
        dtype: str = "fp32",
    ) -> Tensor | Vector:
        fragment = self.tensor_manager.alloc_fragment(
            name=name,
            logical_shape=logical_shape,
            init_zero=init_zero,
            dtype=dtype,
        )
        if isinstance(fragment, Vector):
            self._initialize_vector_backing(fragment, init_zero=init_zero)
        if init_zero and isinstance(fragment, Tensor):
            self.clear(fragment)
        return fragment

    def constant(self, name: str, value: float, size: int = 1) -> FPVar | FPFragment:
        return self.fp_var(name, value=value, size=size)

    def create_value_tile_in_fpram(
        self,
        tile: TensorTile | InputTile,
        fragment: FPFragment,
        *,
        bind: bool = True,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ValueTile:
        return self.value_manager.create_value_tile_in_fpram_for_tile(
            tile,
            fragment,
            bind=bind,
            metadata=metadata,
        )

    def free_tensor_tile(self, operand: object, *, weak: Optional[bool] = None) -> Optional[str] | List[str]:
        if isinstance(operand, TensorTile) and not isinstance(operand, VectorTile):
            return self.value_manager.free_tensor_tile(operand, weak=weak)
        tiles = self.tensor_manager._resolve_tiles_from_operand(operand)
        value_tile_ids: List[str] = []
        for tile in tiles:
            if not isinstance(tile, TensorTile) or isinstance(tile, VectorTile):
                raise TypeError(
                    f"free_tensor_tile expects Tensor/TensorSlice/TensorTile operands, got {type(tile).__name__}"
                )
            value_tile_id = self.value_manager.free_tensor_tile(tile, weak=weak)
            if value_tile_id is not None:
                value_tile_ids.append(value_tile_id)
        return value_tile_ids

    def map_tile_to_fp_fragment(
        self,
        tile: VectorTile,
        fragment: FPFragment,
    ) -> FPFragment:
        return self.value_manager.bind_tile_to_fp_fragment(tile, fragment)

    def _initialize_vector_backing(self, vector: Vector, *, init_zero: bool = False) -> None:
        for tile in _tiles_in_grid_order(vector.tiles):
            if self.value_manager.fp_fragment_bindings.get(tile.tile_id):
                continue
            fragment_name = self._auto_name(f"{vector.name}.fp_tile")
            fragment = self.tensor_manager.fp_fragment(
                name=fragment_name,
                shape=tile.tile_shape,
                init=0.0,
            )
            self.value_manager.bind_tile_to_fp_fragment(tile, fragment)
            tile.metadata["fp_fragment_name"] = fragment.name
            if init_zero:
                self.fp_fill(fragment, 0.0)

    def pipelined(self, extent: int, num_stages: int = 1) -> range:
        self.loop_hints.append(
            {
                "kind": "pipelined",
                "extent": int(extent),
                "num_stages": int(num_stages),
            }
        )
        return _LoopHintRange(self, kind="pipelined", extent=int(extent))

    def parallel(self, extent: int) -> range:
        region_id = self._parallel_region_counter
        self._parallel_region_counter += 1
        self.loop_hints.append(
            {
                "kind": "parallel",
                "extent": int(extent),
                "region_id": region_id,
            }
        )
        return _LoopHintRange(self, kind="parallel", extent=int(extent), region_id=region_id)

    def _sorted_value_tile_ids(self, place: str) -> List[str]:
        if place == "vram":
            items = sorted(self.value_manager._value_tiles_in_vram.items(), key=lambda item: (int(item[1]), item[0]))
            return [value_tile_id for value_tile_id, _ in items]
        if place == "mram":
            items = sorted(self.value_manager._value_tiles_in_mram.items(), key=lambda item: (int(item[1]), item[0]))
            return [value_tile_id for value_tile_id, _ in items]
        if place == "hbm":
            return sorted(self.value_manager._value_tiles_in_hbm.keys())
        raise ValueError(f"Unsupported value-tile residency place: {place}")

    def _tile_by_id(self, tile_id: str) -> Optional[TileLike]:
        return (
            self.tensor_manager.tensor_tiles.get(tile_id)
            or self.tensor_manager.input_tiles.get(tile_id)
            or self.tensor_manager.vector_tiles.get(tile_id)
        )

    def _logical_row_segment_labels(
        self,
        logical_shape: LogicalShape,
        row_start: int,
        row_end: int,
    ) -> List[str]:
        if len(logical_shape) == 4:
            batch, seq, _, _ = logical_shape
            labels: List[str] = []
            for batch_index in range(int(batch)):
                batch_row_start = batch_index * int(seq)
                batch_row_end = batch_row_start + int(seq)
                overlap_start = max(int(row_start), batch_row_start)
                overlap_end = min(int(row_end), batch_row_end)
                if overlap_start >= overlap_end:
                    continue
                labels.append(
                    f"batch={batch_index},seq={overlap_start - batch_row_start}:{overlap_end - batch_row_start}"
                )
            return labels
        if len(logical_shape) == 3:
            return [f"x={int(row_start)}:{int(row_end)}"]
        return [f"row={int(row_start)}:{int(row_end)}"]

    def _tile_slice_labels(self, tile: TileLike) -> List[str]:
        logical_shape = tuple(tile.metadata.get("logical_shape", ()))
        owner_name = _tile_owner_name(tile)
        row_start = int(tile.coord[0]) * int(self.mlen)
        row_end = row_start + int(tile.tile_shape[0])
        row_labels = self._logical_row_segment_labels(logical_shape, row_start, row_end)

        if len(logical_shape) == 4:
            head_dim = int(tile.metadata.get("head_dim", self.mlen))
            grouped_narrow = bool(tile.metadata.get("grouped_narrow"))
            if grouped_narrow:
                group_head_start = int(tile.metadata.get("group_head_start", 0))
                packed_head_count = int(tile.metadata.get("packed_head_count", 1))
                labels: List[str] = []
                for row_label in row_labels:
                    for head_offset in range(packed_head_count):
                        labels.append(f"{owner_name}[{row_label},head={group_head_start + head_offset},d=0:{head_dim}]")
                return labels

            col_start = int(tile.coord[1]) * int(self.mlen)
            col_end = col_start + int(tile.tile_shape[1])
            head_start = col_start // head_dim if head_dim > 0 else 0
            head_end = (col_end - 1) // head_dim + 1 if head_dim > 0 and col_end > col_start else head_start
            d_start = col_start % head_dim if head_dim > 0 else 0
            d_end = col_end % head_dim if head_dim > 0 else 0
            if d_end == 0 and col_end > col_start:
                d_end = head_dim
            if head_end - head_start == 1:
                col_label = f"head={head_start},d={d_start}:{d_end}"
            elif d_start == 0 and d_end == head_dim:
                col_label = f"head={head_start}:{head_end},d=0:{head_dim}"
            else:
                col_label = f"flat_col={col_start}:{col_end}"
            return [f"{owner_name}[{row_label},{col_label}]" for row_label in row_labels]

        col_start = int(tile.coord[1]) * int(self.mlen)
        col_end = col_start + int(tile.tile_shape[1])
        col_label = f"col={col_start}:{col_end}"
        return [f"{owner_name}[{row_label},{col_label}]" for row_label in row_labels]

    def _value_tile_slice_refs_snapshot(self) -> Dict[str, List[str]]:
        refs: Dict[str, List[str]] = {}
        for value_tile_id in sorted(self.value_manager.value_tiles.keys()):
            tile_ids = sorted(self.value_manager.value_tile_tensor_refs.get(value_tile_id, set()))
            if not tile_ids:
                continue
            labels: set[str] = set()
            for tile_id in tile_ids:
                tile = self._tile_by_id(tile_id)
                if tile is None:
                    labels.add(tile_id)
                    continue
                labels.update(self._tile_slice_labels(tile))
            if labels:
                refs[value_tile_id] = sorted(labels)
        return refs

    def _fp_fragment_value_refs_snapshot(self) -> Dict[str, List[str]]:
        refs: Dict[str, List[str]] = {}
        for value_tile_id, value_tile in sorted(self.value_manager.value_tiles.items()):
            fragment_name = value_tile.metadata.get("fp_fragment_name")
            if not isinstance(fragment_name, str):
                continue
            refs.setdefault(fragment_name, []).append(value_tile_id)
        for fragment_name in refs:
            refs[fragment_name].sort()
        return refs

    def _active_fp_fragments_snapshot(self) -> List[str]:
        fragment_names = {
            fragment_name
            for fragment_name in self.value_manager.fp_fragment_bindings.values()
            if isinstance(fragment_name, str)
        }
        fragment_names.update(self._fp_fragment_value_refs_snapshot().keys())
        return sorted(fragment_names)

    def _should_skip_parallel_snapshot(self, op_kind: str) -> bool:
        if not self._active_parallel_region_ids:
            return False
        frame = inspect.currentframe()
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None and frame.f_back.f_back is not None else None
        filename = caller.f_code.co_filename if caller is not None else "<unknown>"
        lineno = caller.f_lineno if caller is not None else -1
        region_id = self._active_parallel_region_ids[-1]
        dedupe_key = (region_id, filename, lineno, op_kind)
        if dedupe_key in self._parallel_snapshot_keys:
            return True
        self._parallel_snapshot_keys.add(dedupe_key)
        return False

    def _record_operation_snapshot(self, op_kind: str, **details: object) -> None:
        if self._should_skip_parallel_snapshot(op_kind):
            return
        self.operation_snapshots.append(
            {
                "index": len(self.operation_snapshots),
                "op_kind": op_kind,
                "details": {key: value for key, value in details.items() if value is not None},
                "vram_value_tiles": self._sorted_value_tile_ids("vram"),
                "mram_value_tiles": self._sorted_value_tile_ids("mram"),
                "hbm_value_tiles": self._sorted_value_tile_ids("hbm"),
                "fpram_fp_fragments": self._active_fp_fragments_snapshot(),
                "value_tile_slice_refs": self._value_tile_slice_refs_snapshot(),
                "fp_fragment_value_refs": self._fp_fragment_value_refs_snapshot(),
            }
        )

    def _format_operation_label(self, snapshot: Dict[str, object]) -> str:
        op_kind = str(snapshot.get("op_kind", "unknown"))
        details = snapshot.get("details", {})
        if not isinstance(details, dict):
            return op_kind
        if op_kind == "matmul":
            path = details.get("path")
            src1 = details.get("src1")
            src2 = details.get("src2")
            dst = details.get("dst")
            extras = [item for item in (f"path={path}" if path else None, f"src1={src1}" if src1 else None, f"src2={src2}" if src2 else None, f"dst={dst}" if dst else None) if item]
            return f"{op_kind} ({', '.join(extras)})" if extras else op_kind
        if op_kind == "atomic_ops":
            op = details.get("op")
            src1 = details.get("src1")
            src2 = details.get("src2")
            dst = details.get("dst")
            extras = [item for item in (f"op={op}" if op else None, f"src1={src1}" if src1 else None, f"src2={src2}" if src2 else None, f"dst={dst}" if dst else None) if item]
            return f"{op_kind} ({', '.join(extras)})" if extras else op_kind
        if op_kind == "row_op":
            op = details.get("op")
            src = details.get("src")
            rhs = details.get("rhs")
            out = details.get("out")
            task_id = details.get("task_id")
            extras = [item for item in (f"op={op}" if op else None, f"src={src}" if src else None, f"rhs={rhs}" if rhs else None, f"out={out}" if out else None, f"task_id={task_id}" if task_id else None) if item]
            return f"{op_kind} ({', '.join(extras)})" if extras else op_kind
        if op_kind in {"pure_fp_compute", "fill"}:
            control = details.get("control")
            src = details.get("src")
            dst = details.get("dst")
            task_id = details.get("task_id")
            extras = [item for item in (f"control={control}" if control else None, f"src={src}" if src else None, f"dst={dst}" if dst else None, f"task_id={task_id}" if task_id else None) if item]
            return f"{op_kind} ({', '.join(extras)})" if extras else op_kind
        return op_kind

    def write_operation_report(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        lines: List[str] = []
        for snapshot in self.operation_snapshots:
            op_index = int(snapshot.get("index", 0))
            op_kind = self._format_operation_label(snapshot)
            details = snapshot.get("details", {})
            detail_text = ""
            if isinstance(details, dict) and details:
                detail_text = " | " + ", ".join(f"{key}={value}" for key, value in details.items())
            lines.append(f"op[{op_index}]: {op_kind}{detail_text}")
            lines.append(f"  vram_value_tiles: {', '.join(snapshot.get('vram_value_tiles', [])) or '(empty)'}")
            lines.append(f"  mram_value_tiles: {', '.join(snapshot.get('mram_value_tiles', [])) or '(empty)'}")
            lines.append(f"  hbm_value_tiles: {', '.join(snapshot.get('hbm_value_tiles', [])) or '(empty)'}")
            lines.append(f"  fpram_fp_fragments: {', '.join(snapshot.get('fpram_fp_fragments', [])) or '(empty)'}")

            value_refs = snapshot.get("value_tile_slice_refs", {})
            lines.append("  value_tile_slice_refs:")
            if isinstance(value_refs, dict) and value_refs:
                for value_tile_id, tile_ids in value_refs.items():
                    lines.append(f"    {value_tile_id}: {', '.join(tile_ids)}")
            else:
                lines.append("    (empty)")

            fragment_refs = snapshot.get("fp_fragment_value_refs", {})
            lines.append("  fp_fragment_value_refs:")
            if isinstance(fragment_refs, dict) and fragment_refs:
                for fragment_name, value_tile_ids in fragment_refs.items():
                    lines.append(f"    {fragment_name}: {', '.join(value_tile_ids)}")
            else:
                lines.append("    (empty)")
            lines.append("")
        report_text = "\n".join(lines)
        output_path.write_text(report_text, encoding="utf-8")
        delta_path = output_path.with_name(f"{output_path.stem}_delta.txt")
        delta_text = build_delta_report(parse_operation_report(report_text))
        delta_path.write_text(delta_text, encoding="utf-8")

    def mapf(self, operand: object) -> List[FPVar]:
        return self.tensor_manager.mapf(operand)

    def mapf_t(self, tensor_operand: object, fp_operand: object, *, control: str = "mixed") -> Dict[str, object]:
        self._require_single_batch_tensor_op("mapf_t", tensor_operand)
        return self.tensor_manager.mapf_t(tensor_operand, fp_operand, control=control)

    def _logical_batch_extent_for_selectors(
        self,
        logical_shape: LogicalShape,
        selectors: Tuple[object, ...],
    ) -> Optional[int]:
        if len(logical_shape) != 4:
            return None
        if not selectors:
            return int(logical_shape[0])
        batch_selector = selectors[0]
        if isinstance(batch_selector, int):
            return 1
        if isinstance(batch_selector, slice):
            start, stop = _slice_item_to_range(batch_selector, int(logical_shape[0]))
            return max(0, int(stop) - int(start))
        return int(logical_shape[0])

    def _operand_batch_extent(self, operand: object) -> Optional[int]:
        if operand is None:
            return None
        if isinstance(operand, (Input, Tensor, Vector, InputTranspose, TensorTranspose, VectorTranspose)):
            shape = tuple(operand.logical_shape)
            return int(shape[0]) if len(shape) == 4 else None
        if isinstance(operand, ParallelAccess):
            return self._logical_batch_extent_for_selectors(tuple(operand.logical_shape), tuple(operand.selectors))
        if isinstance(operand, (InputSlice, TensorSlice, VectorSlice)):
            return self._logical_batch_extent_for_selectors(tuple(operand.base.logical_shape), tuple(operand.selectors))
        if isinstance(operand, ElementRef):
            shape = tuple(getattr(operand.base, "logical_shape", ()))
            return 1 if len(shape) == 4 else None
        if isinstance(operand, (InputTile, TensorTile, VectorTile)):
            shape = tuple(operand.metadata.get("logical_shape", ()))
            return 1 if len(shape) == 4 else None
        return None

    def _operand_debug_name(self, operand: object) -> str:
        if operand is None:
            return "None"
        if isinstance(operand, (InputSlice, TensorSlice, VectorSlice)):
            return getattr(operand.base, "name", type(operand.base).__name__)
        if isinstance(operand, ElementRef):
            return getattr(operand.base, "name", type(operand.base).__name__)
        if isinstance(operand, (InputTile, TensorTile, VectorTile)):
            return _tile_owner_name(operand)
        return str(getattr(operand, "name", type(operand).__name__))

    def _require_single_batch_tensor_op(self, op_name: str, *operands: object) -> None:
        for operand in operands:
            batch_extent = self._operand_batch_extent(operand)
            if batch_extent is None or int(batch_extent) <= 1:
                continue
            raise NotImplementedError(
                f"{op_name} currently requires each BSHD tensor operation to address exactly one batch; "
                f"got {self._operand_debug_name(operand)} with batch_extent={batch_extent}"
            )

    def fp_kernel(
        self,
        src1: object,
        dst: object,
        *,
        src2: Optional[object] = None,
        control: str = "add",
        task_id: str = "fp_kernel",
    ) -> Dict[str, object]:
        self._require_single_batch_tensor_op(task_id, src1, src2, dst)
        src1_vars = self.mapf(src1)
        if isinstance(dst, ElementRef):
            if control in {"copy", "fill"}:
                if len(src1_vars) != 1:
                    raise ValueError(f"ElementRef dst with control={control!r} expects one source FPVar")
                bound = self.tensor_manager.bind_element_pointer(dst, src1_vars[0], mode="alias")
                record = {
                    "op_kind": "fp_kernel_bind",
                    "task_id": task_id,
                    "op": control,
                    "src1": [src1_vars[0].name],
                    "dst": [bound.name],
                }
                self.compute_manager.ops.append(record)
                return record
            dst_var = self.tensor_manager.allocate_element_result_fpvar(dst)
            record = self.compute_manager.fp_kernel(
                src1_vars,
                [dst_var],
                src2=self.mapf(src2) if src2 is not None else None,
                op=control,
                task_id=task_id,
            )
            self.tensor_manager.bind_element_pointer(dst, dst_var, mode="result")
            return record
        return self.compute_manager.fp_kernel(
            src1_vars,
            self.tensor_manager.mapf_dst(dst, control=control, src1_vars=src1_vars),
            src2=self.mapf(src2) if src2 is not None else None,
            op=control,
            task_id=task_id,
        )

    def pure_fp_compute(
        self,
        src1: object,
        dst: object,
        *,
        src2: Optional[object] = None,
        control: str = "add",
        task_id: str = "pure_fp_compute",
    ) -> Dict[str, object]:
        self._require_single_batch_tensor_op(task_id, src1, src2, dst)
        src1_vars = self.mapf(src1)
        if isinstance(dst, ElementRef):
            if control in {"copy", "fill"}:
                if len(src1_vars) != 1:
                    raise ValueError(f"ElementRef dst with control={control!r} expects one source FPVar")
                bound = self.tensor_manager.bind_element_pointer(dst, src1_vars[0], mode="alias")
                record = {
                    "op_kind": "pure_fp_bind",
                    "task_id": task_id,
                    "op": control,
                    "src1": [src1_vars[0].name],
                    "dst": [bound.name],
                }
                self.compute_manager.ops.append(record)
                return record
            dst_var = self.tensor_manager.allocate_element_result_fpvar(dst)
            record = self.compute_manager.pure_fp_compute(
                src1_vars,
                [dst_var],
                src2=self.mapf(src2) if src2 is not None else None,
                op=control,
                task_id=task_id,
            )
            self.tensor_manager.bind_element_pointer(dst, dst_var, mode="result")
            return record
        record = self.compute_manager.pure_fp_compute(
            src1_vars,
            self.tensor_manager.mapf_dst(dst, control=control, src1_vars=src1_vars),
            src2=self.mapf(src2) if src2 is not None else None,
            op=control,
            task_id=task_id,
        )
        return record

    def copy(self, src: object, dst: object) -> object:
        if _is_fp_domain_operand(src) or _is_fp_domain_operand(dst):
            return self.fp_copy(src, dst)
        self._require_single_batch_tensor_op("copy", src, dst)
        src_groups = self.tensor_manager.mapt([src, 0])
        dst_groups = self.tensor_manager.mapt([dst, 0])
        if len(src_groups) != len(dst_groups):
            raise RuntimeError(
                f"copy expects matching tile counts, got src={len(src_groups)} dst={len(dst_groups)}"
            )
        signal_4 = []
        for src_group, dst_group in zip(src_groups, dst_groups):
            if len(src_group) != 1 or len(dst_group) != 1:
                raise RuntimeError("copy currently expects mapt(control=0) groups with one tile each")
            src_tile = src_group[0]
            dst_tile = dst_group[0]
            if not isinstance(src_tile, (TensorTile, InputTile)) or not isinstance(dst_tile, (TensorTile, InputTile)):
                raise RuntimeError("copy expects tensor/input tile groups only")
            src_value = self.value_manager.resolve_value_tile(src_tile)
            if isinstance(dst_tile, InputTile):
                self.value_manager._write_value_back_to_input_tile(src_value, dst_tile)
            else:
                self.value_manager._bind_value_to_tensor_tile(src_value, dst_tile)
            signal_4.append(
                {
                    "control": "copy_bind" if not isinstance(dst_tile, InputTile) else "copy_writeback",
                    "dst_tile": dst_tile,
                    "dst_value_id": src_value.value_tile_id,
                }
            )
        out = self.tensor_manager.mapt_back(signal_4, dst_groups)
        if signal_4:
            copy_trace: List[Dict[str, object]] = []
            for src_group, dst_group in zip(src_groups, dst_groups):
                src_tile = src_group[0]
                dst_tile = dst_group[0]
                src_value = self.value_manager.resolve_value_tile(src_tile)
                dst_value = self.value_manager.resolve_value_tile(dst_tile)
                copy_trace.append(
                    {
                        "src_tile": self.value_manager._tile_debug_state(src_tile),
                        "dst_tile": self.value_manager._tile_debug_state(dst_tile),
                        "src_value": self.value_manager._value_debug_state(src_value),
                        "dst_value": self.value_manager._value_debug_state(dst_value),
                    }
                )
            self._record_operation_snapshot(
                "copy",
                src=getattr(src, "name", type(src).__name__),
                dst=getattr(dst, "name", type(dst).__name__),
                tile_copies=copy_trace,
            )
        return out

    def atomic_ops(
        self,
        src1: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        src2: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        dst: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        *,
        op: str = "add",
    ) -> object:
        """Run elementwise tile ops with alias-safe destination updates.

        The public API looks like one simple tilewise binary op, but the runtime
        has two materially different execution paths:

        - wide/full-tile path using direct ValueTile operands in VRAM
        - narrow/grouped path using ValueTileView-aware compute and rebinding

        When the destination aliases one source (for example `A + B -> B`), the
        wide-tile path first detaches the old destination binding and
        materializes a fresh writable value so reads remain stable during the
        update.
        """
        if op not in {"add", "sub", "mul"}:
            raise ValueError(f"atomic_ops only supports add/sub/mul, got op={op!r}")
        self._require_single_batch_tensor_op(f"atomic_{op}", src1, src2, dst)

        src1_groups = self.tensor_manager.mapt([src1, 0])
        src2_groups = self.tensor_manager.mapt([src2, 0])
        dst_groups = self.tensor_manager.mapt([dst, 0])
        if len(src1_groups) != len(src2_groups) or len(src1_groups) != len(dst_groups):
            raise RuntimeError(
                f"atomic_ops expects matching tile counts, got src1={len(src1_groups)} src2={len(src2_groups)} dst={len(dst_groups)}"
            )

        signal_4: List[Dict[str, object]] = []
        for group_index, (src1_group, src2_group, dst_group_tiles) in enumerate(zip(src1_groups, src2_groups, dst_groups)):
            if len(src1_group) != 1 or len(src2_group) != 1 or len(dst_group_tiles) != 1:
                raise RuntimeError("atomic_ops currently expects mapt(control=0) groups with one tile each")
            lhs_tile = src1_group[0]
            rhs_tile = src2_group[0]
            dst_tile = dst_group_tiles[0]
            if not isinstance(lhs_tile, (TensorTile, InputTile)) or not isinstance(rhs_tile, (TensorTile, InputTile)) or not isinstance(dst_tile, (TensorTile, InputTile)):
                raise RuntimeError("atomic_ops expects tensor/input tile groups only")

            lhs_value = self.value_manager.resolve_value_tile(lhs_tile)
            rhs_value = self.value_manager.resolve_value_tile(rhs_tile)
            dst_aliases_source = lhs_tile.tile_id == dst_tile.tile_id or rhs_tile.tile_id == dst_tile.tile_id
            if isinstance(dst_tile, TensorTile):
                dst_view = self.value_manager.resolve_value_tile_view(dst_tile)
                prepared_write = self.value_manager.prepare_updated_view_value(
                    dst_tile,
                    dst_view,
                    ensure_old_place="vram" if dst_aliases_source else None,
                    new_place="vram",
                )
                dst_value = prepared_write.new_value
                if lhs_tile.tile_id == dst_tile.tile_id:
                    lhs_value = prepared_write.old_value
                if rhs_tile.tile_id == dst_tile.tile_id:
                    rhs_value = prepared_write.old_value
                if prepared_write.requires_preserve_copy:
                    old_vram_addr = prepared_write.old_value.residency.get("vram_addr")
                    new_vram_addr = prepared_write.new_value.residency.get("vram_addr")
                    if old_vram_addr is None or new_vram_addr is None:
                        raise RuntimeError(
                            "atomic_ops preserve copy requires old/new values resident in VRAM"
                        )
                    self.isa_emitter.emit_zero_vram_tile(int(new_vram_addr))
                    self.isa_emitter.emit_tile_binary(
                        lhs_vram_addr=int(new_vram_addr),
                        rhs_vram_addr=int(old_vram_addr),
                        dst_vram_addr=int(new_vram_addr),
                        op="add",
                        task_id=f"atomic_preserve_copy.{dst_tile.tile_id}.{group_index}",
                    )
            else:
                dst_value = self.value_manager._prepare_mapv_destination_value(dst_tile, "vram")
            self.value_manager.ensure_value_tile_in_place(lhs_value, "vram")
            self.value_manager.ensure_value_tile_in_place(rhs_value, "vram")
            self.value_manager.ensure_value_tile_in_place(dst_value, "vram")

            lhs_vram_addr = lhs_value.residency.get("vram_addr")
            rhs_vram_addr = rhs_value.residency.get("vram_addr")
            dst_vram_addr = dst_value.residency.get("vram_addr")
            if lhs_vram_addr is None or rhs_vram_addr is None or dst_vram_addr is None:
                raise RuntimeError("atomic_ops wide-tile path requires all operands in VRAM")
            self.isa_emitter.emit_tile_binary(
                lhs_vram_addr=int(lhs_vram_addr),
                rhs_vram_addr=int(rhs_vram_addr),
                dst_vram_addr=int(dst_vram_addr),
                op=op,
                task_id=f"atomic_{op}.{group_index}",
            )
            signal_4.append(
                {
                    "control": f"atomic_{op}_tile",
                    "dst_tile": dst_tile,
                    "dst_value_id": dst_value.value_tile_id,
                }
            )

        out = self.tensor_manager.mapt_back(signal_4, dst_groups)
        self._record_operation_snapshot(
            "atomic_ops",
            op=op,
            src1=getattr(src1, "name", type(src1).__name__),
            src2=getattr(src2, "name", type(src2).__name__),
            dst=getattr(dst, "name", type(dst).__name__),
        )
        return out

    def atomic_add(
        self,
        src1: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        src2: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        dst: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
    ) -> object:
        return self.atomic_ops(src1, src2, dst, op="add")

    def atomic_sub(
        self,
        src1: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        src2: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        dst: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
    ) -> object:
        return self.atomic_ops(src1, src2, dst, op="sub")

    def atomic_mul(
        self,
        src1: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        src2: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        dst: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
    ) -> object:
        return self.atomic_ops(src1, src2, dst, op="mul")

    def fill(self, dst: object, src: object) -> object:
        if isinstance(dst, (FPVar, FPFragment, FPFragmentSlice, Vector, VectorSlice, VectorTile, ElementRef)):
            out = self.fp_fill(dst, src)
            self._record_operation_snapshot(
                "fill",
                control="copy",
                src=getattr(src, "name", src if isinstance(src, (int, float, str)) else type(src).__name__),
                dst=getattr(dst, "name", type(dst).__name__),
            )
            return out
        raise NotImplementedError(f"fill currently supports FP-domain destinations only, got {type(dst).__name__}")

    def matmul(self, src1: Tensor | Input, src2: Tensor | Input | TensorTranspose | InputTranspose, dst: Tensor | Input) -> object:
        """Route one matmul request to the correct execution strategy.

        The current runtime supports multiple matmul families behind one API:

        - default tilewise matmul using `mapt -> mapv -> compute`
        - view-based lane matmul for grouped narrow-head layouts
        - BTMM/QKT path when the RHS is explicitly transposed and shapes match

        This function is therefore both an entry point and a router. The exact
        path is selected from logical shape/layout information before compute
        packets are materialized.
        """
        self._require_single_batch_tensor_op("matmul", src1, src2, dst)
        if self._should_use_btmm_qkt_matmul(src1, src2, dst):
            out = self._matmul_btmm_qkt_path(src1, _unwrap_transposed_operand(src2), dst)
            self._record_operation_snapshot(
                "matmul",
                path="btmm_qkt",
                src1=getattr(src1, "name", type(src1).__name__),
                src2=getattr(_unwrap_transposed_operand(src2), "name", type(_unwrap_transposed_operand(src2)).__name__),
                dst=getattr(dst, "name", type(dst).__name__),
            )
            return out
        if _is_transposed_operand(src2):
            raise RuntimeError("BTMM/QKT matmul only supports explicit transpose syntax as prog.matmul(q, k.T, p)")
        if self._should_use_view_matmul(src1, src2, dst):
            out = self._matmul_view_path(src1, src2, dst)
            self._record_operation_snapshot(
                "matmul",
                path="view",
                src1=getattr(src1, "name", type(src1).__name__),
                src2=getattr(src2, "name", type(src2).__name__),
                dst=getattr(dst, "name", type(dst).__name__),
            )
            return out
        signal_0 = [src1, src2, dst, 0]
        signal_1 = self.tensor_manager.mapt(signal_0)
        signal_4 = []
        for a in signal_1:
            a.append(["vram", "mram", "vram"])
            tmp = self.value_manager.mapv(a)
            signal_2 = self.compute_manager.execute([tmp, "matmul"])
            signal_3 = self.value_manager.mapv_back([signal_2, tmp])
            signal_4.append(signal_3)
        out = self.tensor_manager.mapt_back(signal_4, signal_1)
        self._record_operation_snapshot(
            "matmul",
            path="default",
            src1=getattr(src1, "name", type(src1).__name__),
            src2=getattr(src2, "name", type(src2).__name__),
            dst=getattr(dst, "name", type(dst).__name__),
        )
        return out

    def _should_use_btmm_qkt_matmul(
        self,
        src1: Tensor | Input,
        src2: Tensor | Input | TensorTranspose | InputTranspose,
        dst: Tensor | Input,
    ) -> bool:
        if not _is_transposed_operand(src2):
            return False
        src2_base = _unwrap_transposed_operand(src2)
        logical_shapes = [getattr(src, "logical_shape", ()) for src in (src1, src2_base, dst)]
        if not all(len(shape) == 4 for shape in logical_shapes):
            return False
        _, src1_seq, src1_heads, src1_dim = logical_shapes[0]
        _, src2_seq, src2_heads, src2_dim = logical_shapes[1]
        _, dst_seq, dst_heads, dst_dim = logical_shapes[2]
        if src1_heads != src2_heads or src1_heads != dst_heads:
            return False
        if src1_dim != self.btmm_hlen or src2_dim != self.btmm_hlen:
            return False
        if dst_seq != src1_seq or dst_dim != src2_seq:
            return False
        if dst_dim % self.mlen != 0:
            return False
        return True

    def _should_use_view_matmul(self, src1: Tensor | Input, src2: Tensor | Input, dst: Tensor | Input) -> bool:
        logical_shapes = [getattr(src, "logical_shape", ()) for src in (src1, src2, dst)]
        if not all(len(shape) == 4 for shape in logical_shapes):
            return False
        _, _, _, src2_head_dim = logical_shapes[1]
        _, _, _, dst_head_dim = logical_shapes[2]
        if src2_head_dim <= 0 or self.mlen % src2_head_dim != 0:
            return False
        if dst_head_dim != src2_head_dim:
            return False
        return True

    def _matmul_view_path(self, src1: Tensor | Input, src2: Tensor | Input, dst: Tensor | Input) -> object:
        signal_1 = self.tensor_manager.mapt_view_matmul(src1, src2, dst)
        signal_4: List[Dict[str, object]] = []

        for dst_tile, terms, group_start in signal_1:
            if not terms:
                continue

            dst_view = self.value_manager.resolve_value_tile_view(dst_tile)
            prepared_write = self.value_manager.prepare_updated_view_value(
                dst_tile,
                dst_view,
                ensure_old_place="vram",
                new_place="vram",
            )
            dst_value = prepared_write.new_value

            for term_index, (lhs_tiles, rhs_tile) in enumerate(terms):
                if not lhs_tiles:
                    continue
                lhs_values = [self.value_manager.resolve_value_tile(tile) for tile in lhs_tiles]
                self.compute_manager.view_matmul(
                    lhs_values=lhs_values,
                    rhs_tile=rhs_tile,
                    dst_tile=dst_tile,
                    dst_value=dst_value,
                    task_id=f"view_matmul.{dst_tile.tile_id}.term{term_index}",
                    zero_dst=(term_index == 0),
                )

            signal_4.append(
                {
                    "control": "view_matmul",
                    "dst_tile_id": dst_tile.tile_id,
                    "dst_value_id": dst_value.value_tile_id,
                    "dst_tile": dst_tile,
                }
            )

        out = self.tensor_manager.mapt_back(signal_4, signal_1)
        return out

    def _matmul_btmm_qkt_path(self, src1: Tensor | Input, src2: Tensor | Input, dst: Tensor | Input) -> object:
        signal_1 = self.tensor_manager.mapt([src1, src2, dst, 1])
        signal_4: List[Dict[str, object]] = []

        for thread_index, thread in enumerate(signal_1):
            if not isinstance(thread, dict):
                raise RuntimeError(f"BTMM QKT matmul expected one dict thread, got {type(thread).__name__}")
            lhs_tiles = thread.get("lhs_tiles")
            rhs_tiles = thread.get("rhs_tiles")
            dst_tiles = thread.get("dst_tiles")
            if not isinstance(lhs_tiles, list) or not isinstance(rhs_tiles, list) or not isinstance(dst_tiles, list):
                raise RuntimeError("BTMM QKT matmul thread is missing lhs_tiles/rhs_tiles/dst_tiles lists")
            if len(lhs_tiles) != 1 or len(rhs_tiles) != 1:
                raise RuntimeError(
                    f"BTMM QKT matmul currently expects one lhs tile and one rhs tile per thread, "
                    f"got lhs={len(lhs_tiles)} rhs={len(rhs_tiles)}"
                )
            if not dst_tiles:
                continue

            lhs_tile = lhs_tiles[0]
            rhs_tile = rhs_tiles[0]
            if not isinstance(lhs_tile, (TensorTile, InputTile)) or not isinstance(rhs_tile, (TensorTile, InputTile)):
                raise RuntimeError("BTMM QKT matmul thread tiles must be tensor/input tiles")
            if not all(isinstance(tile, (TensorTile, InputTile)) for tile in dst_tiles):
                raise RuntimeError("BTMM QKT matmul destination group must contain tensor/input tiles only")

            lhs_value = self.value_manager._resolve_mapv_source_value(lhs_tile, "vram")
            rhs_value = self.value_manager._resolve_mapv_source_value(rhs_tile, "mram")
            if not isinstance(lhs_value, ValueTile) or not isinstance(rhs_value, ValueTile):
                raise RuntimeError("BTMM QKT matmul currently expects full-tile source values")

            task_id = (
                f"btmm_qkt.r{thread.get('lhs_row_block', 0)}"
                f".k{thread.get('rhs_row_block', 0)}"
                f".g{thread.get('group_start', 0)}"
                f".t{thread_index}"
            )
            btmm_state = self.btmm(
                lhs_packed_value=lhs_value,
                rhs_value=rhs_value,
                task_id=task_id,
            )
            write_state = self.btmm_write(
                btmm_state=btmm_state,
                tile_count=len(dst_tiles),
                reason=task_id,
                logical_shape=(self.mlen, self.mlen),
                metadata={
                    "source_thread": task_id,
                    "group_start": thread.get("group_start"),
                    "lhs_row_block": thread.get("lhs_row_block"),
                    "rhs_row_block": thread.get("rhs_row_block"),
                },
                task_id=f"{task_id}.wo",
            )

            out_values = write_state.get("dst_values")
            if not isinstance(out_values, list) or len(out_values) != len(dst_tiles):
                raise RuntimeError(
                    f"BTMM QKT writeback expected {len(dst_tiles)} output value tiles, got {len(out_values) if isinstance(out_values, list) else 'invalid'}"
                )

            for dst_tile, dst_value in zip(dst_tiles, out_values):
                if not isinstance(dst_value, ValueTile):
                    raise RuntimeError("BTMM QKT writeback produced one non-ValueTile output")
                if isinstance(dst_tile, InputTile):
                    self.value_manager._write_value_back_to_input_tile(dst_value, dst_tile)
                else:
                    self.value_manager._bind_value_to_tensor_tile(dst_value, dst_tile)

            signal_4.append(
                {
                    "control": "btmm_qkt_matmul",
                    "dst_tiles": dst_tiles,
                    "dst_tile": dst_tiles[0],
                    "task_id": task_id,
                    "thread_index": thread_index,
                    "base_addr": write_state.get("base_addr"),
                }
            )

        out = self.tensor_manager.mapt_back(signal_4, signal_1)
        return out

    def btmm(
        self,
        *,
        lhs_packed_value: ValueTile,
        rhs_value: ValueTile,
        task_id: str = "btmm",
    ) -> Dict[str, object]:
        return self.compute_manager.btmm(
            lhs_packed_value=lhs_packed_value,
            rhs_value=rhs_value,
            task_id=task_id,
        )

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
        return self.compute_manager.btmm_write(
            btmm_state=btmm_state,
            tile_count=tile_count,
            reason=reason,
            logical_shape=logical_shape,
            metadata=metadata,
            task_id=task_id,
        )

    def fp_copy(self, src: object, dst: object) -> Dict[str, object]:
        return self.fp_kernel(src, dst, control="copy", task_id="fp_copy")

    def fp_fill(self, dst: object, src: object) -> Dict[str, object]:
        return self.fp_kernel(src, dst, control="copy", task_id="fp_fill")

    def fp_fill_from_addr(self, dst: object, src_fpram_addr: int) -> Dict[str, object]:
        src_var = self._fp_var_from_addr(int(src_fpram_addr))
        return self.fp_fill(dst, src_var)

    def fp_add(self, src1: object, src2: object, dst: object) -> Dict[str, object]:
        return self.fp_kernel(src1, dst, src2=src2, control="add", task_id="fp_add")

    def fp_sub(self, src1: object, src2: object, dst: object) -> Dict[str, object]:
        return self.fp_kernel(src1, dst, src2=src2, control="sub", task_id="fp_sub")

    def fp_mul(self, src1: object, src2: object, dst: object) -> Dict[str, object]:
        return self.fp_kernel(src1, dst, src2=src2, control="mul", task_id="fp_mul")

    def fp_max(self, src1: object, src2: object, dst: object) -> Dict[str, object]:
        return self.fp_kernel(src1, dst, src2=src2, control="max", task_id="fp_max")

    def fp_exp(self, src: object, dst: object) -> Dict[str, object]:
        return self.fp_kernel(src, dst, control="exp", task_id="fp_exp")

    def fp_reci(self, src: object, dst: object) -> Dict[str, object]:
        return self.fp_kernel(src, dst, control="reci", task_id="fp_reci")

    def fp_sqrt(self, src: object, dst: object) -> Dict[str, object]:
        return self.fp_kernel(src, dst, control="sqrt", task_id="fp_sqrt")

    def row_op(
        self,
        src: Tensor | Input | Vector | TensorSlice | InputSlice | VectorSlice,
        rhs: Optional[object] = None,
        op: str = "exp",
        *,
        out: Optional[object] = None,
        dim: int = -1,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        self._require_single_batch_tensor_op(task_id or f"row_op.{op}", src, rhs, out)
        if dim != -1:
            raise NotImplementedError(f"row_op currently supports dim=-1 only, got {dim}")
        src_slice_ranges: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        if isinstance(src, (TensorSlice, InputSlice, VectorSlice)):
            src_slice_ranges = _logical_selectors_to_physical_ranges(src.base.logical_shape, src.selectors)
        src_groups = self.tensor_manager.mapt([src, 0])
        if not src_groups:
            return []
        records: List[Dict[str, object]] = []
        mutates_src = op in {"exp", "reci", "mul", "add", "sub"}
        rhs_vars = self.mapf(rhs) if rhs is not None and op in {"mul", "add", "sub"} else None
        out_vars = self.mapf(out) if out is not None else None
        rhs_cursor = 0
        out_cursor = 0
        for group_index, src_group in enumerate(src_groups):
            if len(src_group) != 1 or not isinstance(src_group[0], (TensorTile, InputTile, VectorTile)):
                raise RuntimeError("row_op currently expects one full tile per mapt group")
            src_tile = src_group[0]
            if isinstance(src_tile, VectorTile):
                record = self._row_op_vector_tile(
                    src_tile,
                    src_slice_ranges=src_slice_ranges,
                    rhs_vars=rhs_vars,
                    out_vars=out_vars,
                    op=op,
                    rhs_cursor=rhs_cursor,
                    out_cursor=out_cursor,
                    task_id=task_id or f"row_op.{op}.{group_index}",
                )
                rhs_cursor = int(record.get("rhs_cursor", rhs_cursor))
                out_cursor = int(record.get("out_cursor", out_cursor))
                records.append(record)
                continue
            if src_slice_ranges is not None:
                src_operand = self.value_manager.resolve_row_operand_for_ranges(
                    src_tile,
                    src_slice_ranges[0],
                    src_slice_ranges[1],
                    "vram",
                )
            else:
                src_operand = self.value_manager.resolve_row_operand(src_tile, "vram")
            if src_slice_ranges is not None:
                target_view = src_operand if isinstance(src_operand, ValueTileView) else None
            else:
                target_view = None
            dst_operand: RowOperandLike = src_operand
            if mutates_src:
                if isinstance(src_tile, TensorTile):
                    if target_view is None:
                        target_view = self.value_manager.resolve_value_tile_view(src_tile)
                    prepared_write = self.value_manager.prepare_updated_view_value(
                        src_tile,
                        target_view,
                        ensure_old_place="vram",
                        new_place="vram",
                    )
                    if not prepared_write.reuse_old:
                        if prepared_write.requires_preserve_copy:
                            old_vram_addr = prepared_write.old_value.residency.get("vram_addr")
                            new_vram_addr = prepared_write.new_value.residency.get("vram_addr")
                            if old_vram_addr is None or new_vram_addr is None:
                                raise RuntimeError(
                                    "row_op preserve copy requires old/new values resident in VRAM"
                                )
                            self.isa_emitter.emit_zero_vram_tile(int(new_vram_addr))
                            self.isa_emitter.emit_tile_binary(
                                lhs_vram_addr=int(new_vram_addr),
                                rhs_vram_addr=int(old_vram_addr),
                                dst_vram_addr=int(new_vram_addr),
                                op="add",
                                task_id=f"row_op_preserve_copy.{src_tile.tile_id}.{group_index}",
                            )
                        if isinstance(src_operand, ValueTileView):
                            dst_operand = prepared_write.target_view
                        else:
                            dst_operand = prepared_write.new_value
                else:
                    dst_operand = self.value_manager._prepare_mapv_destination_value(src_tile, "vram")
            row_count = int(src_operand.row_count if isinstance(src_operand, ValueTileView) else src_operand.logical_shape[0])

            group_out: Optional[List[FPVar]] = None
            if op in {"reduce_max", "reduce_sum"}:
                if out_vars is None:
                    raise ValueError(f"row_op op={op!r} requires out")
                if out_cursor + row_count > len(out_vars):
                    raise ValueError(f"row_op op={op!r} out size is smaller than required rows")
                group_out = out_vars[out_cursor : out_cursor + row_count]
                out_cursor += row_count

            group_rhs: Optional[List[FPVar]] = None
            if op in {"mul", "add", "sub"}:
                if rhs_vars is None:
                    raise ValueError(f"row_op op={op!r} requires rhs")
                if len(rhs_vars) == 1:
                    group_rhs = list(rhs_vars)
                else:
                    if rhs_cursor + row_count > len(rhs_vars):
                        raise ValueError(f"row_op op={op!r} rhs size is smaller than required rows")
                    group_rhs = rhs_vars[rhs_cursor : rhs_cursor + row_count]
                    rhs_cursor += row_count

            record = self.compute_manager.row_operations(
                src_operand,
                dst_operand=dst_operand,
                dst=group_out,
                rhs=group_rhs,
                op=op,
                task_id=task_id or f"row_op.{op}.{group_index}",
            )
            records.append(record)
        self._record_operation_snapshot(
            "row_op",
            op=op,
            src=getattr(src, "name", type(src).__name__),
            rhs=getattr(rhs, "name", rhs if isinstance(rhs, (int, float, str)) else type(rhs).__name__) if rhs is not None else None,
            out=getattr(out, "name", type(out).__name__) if out is not None else None,
            task_id=task_id or "row_op",
        )
        return records

    def _row_op_vector_tile(
        self,
        src_tile: VectorTile,
        *,
        src_slice_ranges: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        rhs_vars: Optional[List[FPVar]],
        out_vars: Optional[List[FPVar]],
        op: str,
        rhs_cursor: int,
        out_cursor: int,
        task_id: str,
    ) -> Dict[str, object]:
        fragment = self.value_manager.resolve_fp_fragment(src_tile)
        row_groups = _vector_tile_row_fp_groups(
            src_tile=src_tile,
            fragment=fragment,
            mlen=self.mlen,
            btmm_hlen=self.btmm_hlen,
            src_slice_ranges=src_slice_ranges,
        )
        if not row_groups:
            return {
                "op_kind": "row_op_vector",
                "task_id": task_id,
                "tile": src_tile.tile_id,
                "rows": 0,
                "rhs_cursor": rhs_cursor,
                "out_cursor": out_cursor,
            }

        if op in {"exp", "reci"}:
            for row_index, row_vars in enumerate(row_groups):
                self.compute_manager.fp_kernel(
                    row_vars,
                    row_vars,
                    op=op,
                    task_id=f"{task_id}.row{row_index}",
                )
        elif op in {"add", "sub", "mul"}:
            if rhs_vars is None:
                raise ValueError(f"row_op op={op!r} requires rhs")
            if len(rhs_vars) == 1:
                row_rhs_vars = [rhs_vars[0] for _ in row_groups]
            else:
                if rhs_cursor + len(row_groups) > len(rhs_vars):
                    raise ValueError(f"row_op op={op!r} rhs size is smaller than required rows")
                row_rhs_vars = rhs_vars[rhs_cursor : rhs_cursor + len(row_groups)]
                rhs_cursor += len(row_groups)
            for row_index, (row_vars, rhs_var) in enumerate(zip(row_groups, row_rhs_vars)):
                self.compute_manager.fp_kernel(
                    row_vars,
                    row_vars,
                    src2=[rhs_var] * len(row_vars),
                    op=op,
                    task_id=f"{task_id}.row{row_index}",
                )
        elif op in {"reduce_sum", "reduce_max"}:
            if out_vars is None:
                raise ValueError(f"row_op op={op!r} requires out")
            if out_cursor + len(row_groups) > len(out_vars):
                raise ValueError(f"row_op op={op!r} out size is smaller than required rows")
            row_out_vars = out_vars[out_cursor : out_cursor + len(row_groups)]
            out_cursor += len(row_groups)
            for row_index, (row_vars, out_var) in enumerate(zip(row_groups, row_out_vars)):
                if not row_vars:
                    continue
                if op == "reduce_sum":
                    self.compute_manager.fp_kernel(
                        [self.mapf(0.0)[0]],
                        [out_var],
                        op="copy",
                        task_id=f"{task_id}.row{row_index}.init",
                    )
                    for cell_index, cell_var in enumerate(row_vars):
                        self.compute_manager.fp_kernel(
                            [out_var],
                            [out_var],
                            src2=[cell_var],
                            op="add",
                            task_id=f"{task_id}.row{row_index}.cell{cell_index}",
                        )
                else:
                    self.compute_manager.fp_kernel(
                        [row_vars[0]],
                        [out_var],
                        op="copy",
                        task_id=f"{task_id}.row{row_index}.init",
                    )
                    for cell_index, cell_var in enumerate(row_vars[1:], start=1):
                        self.compute_manager.fp_kernel(
                            [out_var],
                            [out_var],
                            src2=[cell_var],
                            op="max",
                            task_id=f"{task_id}.row{row_index}.cell{cell_index}",
                        )
        else:
            raise NotImplementedError(f"row_op vector path does not support op={op!r}")

        return {
            "op_kind": "row_op_vector",
            "task_id": task_id,
            "tile": src_tile.tile_id,
            "fragment": fragment.name,
            "rows": len(row_groups),
            "op": op,
            "rhs_cursor": rhs_cursor,
            "out_cursor": out_cursor,
        }

    def elementwise(
        self,
        src1: object,
        dst: object,
        *,
        src2: Optional[object] = None,
        op: str = "add",
        task_id: Optional[str] = None,
    ) -> Dict[str, object]:
        self._require_single_batch_tensor_op(task_id or f"elementwise.{op}", src1, src2, dst)
        if _is_parallel_graph_operand(src1) or _is_parallel_graph_operand(dst) or _is_parallel_graph_operand(src2):
            if not isinstance(dst, ParallelAccess):
                raise ValueError("parallel elementwise dst must be one ParallelAccess target")
            expr = _coerce_parallel_expr(src1)
            if op == "copy":
                self.thread_manager.record_parallel_assignment_from_access(dst, expr)
            elif op == "add":
                self.thread_manager.record_parallel_assignment_from_access(dst, expr + src2)
            elif op == "sub":
                self.thread_manager.record_parallel_assignment_from_access(dst, expr - src2)
            elif op == "mul":
                self.thread_manager.record_parallel_assignment_from_access(dst, expr * src2)
            else:
                raise ValueError(f"parallel elementwise does not support op={op!r}")
            region = self.thread_manager.current_parallel_graph()
            return {
                "op_kind": "parallel_graph_elementwise",
                "task_id": task_id or f"elementwise.{op}",
                "region": region.name,
                "op": op,
                "dst": _parallel_access_identity(dst),
            }
        return self.pure_fp_compute(
            src1,
            dst,
            src2=src2,
            control=op,
            task_id=task_id or f"elementwise.{op}",
        )

    def clear(self, tensor: Tensor) -> None:
        self._require_single_batch_tensor_op("clear", tensor)
        cleared_values: set[str] = set()
        for tile in _tiles_in_grid_order(tensor.tiles):
            value = self.value_manager.resolve_value_tile(tile)
            self.value_manager.ensure_value_tile_in_place(value, "vram")
            if value.value_tile_id in cleared_values:
                continue
            vram_addr = value.residency.get("vram_addr")
            if vram_addr is None:
                raise RuntimeError(f"clear expected VRAM residency for {value.value_tile_id}")
            self.isa_emitter.emit_zero_vram_tile(int(vram_addr))
            cleared_values.add(value.value_tile_id)

    def _fp_var_from_addr(self, fp_mem_addr: int) -> FPVar:
        for fp_var in self.tensor_manager.fp_vars.values():
            if fp_var.fp_mem_addr == fp_mem_addr:
                return fp_var
        raise KeyError(f"No FPVar found at fp_mem_addr={fp_mem_addr}")

    def _arith_progression(self, values: Sequence[int]) -> Optional[Tuple[int, int, int]]:
        if not values:
            return None
        if len(values) == 1:
            return int(values[0]), 1, 0
        first = int(values[0])
        step = int(values[1]) - first
        for idx, value in enumerate(values[1:], start=1):
            if int(value) != first + idx * step:
                return None
        return first, len(values), step

    def alloc_hbm_addr(self, elems: int) -> int:
        size = int(elems * self.real_data_ratio)
        base = self._next_hbm_addr
        self._next_hbm_addr += size
        return base

    def add_hbm_object(self, name: str, shape: Tuple[int, int], *, hbm_addr: Optional[int] = None) -> int:
        base_addr = self.alloc_hbm_addr(shape[0] * shape[1]) if hbm_addr is None else int(hbm_addr)
        self.compiler.add_hbm_object(
            name=name,
            shape=shape,
            hbm_addr=base_addr,
            real_data_ratio=self.real_data_ratio,
        )
        self.hardware.hbm_objects[name] = {
            "name": name,
            "shape": shape,
            "base_addr": base_addr,
        }
        return base_addr

    def build_fp_preload(self, min_size: int = 0) -> List[float]:
        """Return the FP_MEM initialisation array ordered by address.

        Entries come from fp_var() declarations; any slots beyond the
        declared range up to min_size are zero-padded.
        """
        values = list(self.tensor_manager._fp_mem_values)
        size = max(len(values), int(min_size))
        values.extend([0.0] * (size - len(values)))
        return values

    def _normalize_large_addi_immediates(self, asm_code: str) -> str:
        lines: List[str] = []
        for raw_line in asm_code.splitlines():
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                lines.append(line)
                continue

            parts = stripped.split(None, 1)
            if len(parts) != 2 or parts[0] != "S_ADDI_INT":
                lines.append(line)
                continue

            operands = [item.strip() for item in parts[1].split(",")]
            if len(operands) != 3:
                lines.append(line)
                continue

            rd, rs1, imm_text = operands
            try:
                imm_value = int(imm_text)
            except ValueError:
                lines.append(line)
                continue

            if 0 <= imm_value <= 262143:
                lines.append(line)
                continue

            if rs1 != "gp0":
                lines.append(line)
                continue

            upper = imm_value >> 12
            lower = imm_value & 0xFFF
            lines.append(f"S_LUI_INT {rd}, {upper}")
            lines.append(f"S_ADDI_INT {rd}, {rd}, {lower}")
        normalized = "\n".join(lines)
        if asm_code.endswith("\n"):
            normalized += "\n"
        return normalized

    def compile(self) -> str:
        if not self._parallel_execution_lowered:
            self.lower_parallel_execution_plans()
        self.compiler.generated_code = self._normalize_large_addi_immediates(self.compiler.generated_code)
        return self.compiler.generated_code


