"""TileTensor runtime for logical tiles, backing values, and ISA emission.

This module is the main execution-side runtime used by the TileTensor
testbench. The current design is centered on three layers:

1. TensorManager
   Logical tensor/input objects, tile creation, slice resolution, and `mapt`
   grouping. It does not decide residency or backing allocation.

2. ValueManager
   `tile -> ValueTile` bindings, `ValueTileView` resolution, residency changes,
   and write preparation.

3. ComputeManager
   Last-mile operand validation, residency ensure-at-use, and ISA emission.

The important runtime law is:

    logical tile -> ValueTileView -> compute -> bind/writeback

Core write-path functions are:

- `resolve_value_tile(...)`
- `resolve_value_tile_view(...)`
- `prepare_updated_view_value(...)`
- `prepare_vram_backing_value(...)`

View-update / preserve-copy policy
---------------------------------

For tensor destinations, the runtime intentionally treats full-tile physical
copy in VRAM as the last-resort fallback. The current decision order is:

1. Reuse old backing in place
   - If the destination view has no conflicting live refs, the write reuses
     `old_value` directly.
   - When this path writes back to VRAM, stale MRAM/HBM residency for the same
     value is invalidated so later readers do not observe outdated copies.

2. Replace whole logical tile without preserve copy
   - If refs conflict but the destination view covers the whole logical tile,
     the runtime allocates/prepares one fresh writable backing value and does
     not preserve old contents, because the tile will be fully overwritten.

3. Partial-update successor without physical copy
   - If refs conflict and the write only updates part of the tile, the runtime
     first tries `_prepare_partial_update_vram_successor(...)`.
   - This path is the preferred efficient partial-update route: when the old
     value already has one stable HBM backing plus current VRAM storage, the
     successor inherits the VRAM physical storage directly and the old version
     remains recoverable from HBM.

4. Physical ISA copy fallback
   - Only if the partial-update successor path is unavailable does the runtime
     fall back to preserve copy in VRAM.
   - The current implementation materializes this as:
       `emit_zero_vram_tile(new)`
       `emit_tile_binary(lhs=new, rhs=old, dst=new, op="add")`
   - This is intentionally the slowest, last-resort path.
"""

from __future__ import annotations

import inspect
import sys
from math import ceil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.asm_templates import preload_addr_reg_asm
from tiled_developer_compiler import TiledDeveloperCompiler
from operation_report_delta import build_delta_report, parse_operation_report


TileCoord = Tuple[int, int]
LogicalShape = Tuple[int, ...]
SliceItem = int | slice
FPIndex = Tuple[int, ...]


@dataclass
class FPVar:
    name: str
    dtype: str = "fp32"
    size: int = 1
    storage: str = "fpram"
    fp_mem_addr: Optional[int] = None  # Address in FP_MEM; loaded via S_LD_FP before VF ops


@dataclass
class FPFragment:
    program: "TileTensorProgram"
    name: str
    shape: Tuple[int, ...]
    vars: Dict[FPIndex, FPVar] = field(default_factory=dict)
    dtype: str = "fp32"
    storage: str = "fpram"
    metadata: Dict[str, object] = field(default_factory=dict)

    def __getitem__(self, item: SliceItem | Tuple[SliceItem, ...]) -> "FPFragmentSlice":
        if not isinstance(item, tuple):
            item = (item,)
        return FPFragmentSlice(base=self, selectors=item)


@dataclass
class FPFragmentSlice:
    base: FPFragment
    selectors: Tuple[SliceItem, ...]


@dataclass(frozen=True)
class ElementRef:
    base: object
    indices: Tuple[int, ...]


@dataclass(frozen=True)
class ParallelAxis:
    program: "TileTensorProgram"
    region_id: int
    axis: int
    name: str
    extent: int

    def _as_expr(self) -> "ParallelExpr":
        return ParallelExpr(kind="axis", value=self)

    def __add__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__add__(other)

    def __radd__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__radd__(other)

    def __sub__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__sub__(other)

    def __rsub__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__rsub__(other)

    def __mul__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__mul__(other)

    def __rmul__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__rmul__(other)

    def __mod__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__mod__(other)

    def __rmod__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__rmod__(other)

    def __lt__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__lt__(other)

    def __le__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__le__(other)

    def __gt__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__gt__(other)

    def __ge__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__ge__(other)

    def __eq__(self, other: object) -> "ParallelExpr":  # type: ignore[override]
        return self._as_expr().__eq__(other)


@dataclass(frozen=True)
class ParallelAccess:
    base: object
    selectors: Tuple[object, ...]

    @property
    def program(self) -> "TileTensorProgram":
        return self.base.program

    @property
    def logical_shape(self) -> LogicalShape:
        shape = tuple(getattr(self.base, "logical_shape", ()))
        if not shape:
            raise RuntimeError(f"ParallelAccess base {type(self.base).__name__} does not expose logical_shape")
        return shape

    def append_selectors(self, item: SliceItem | Tuple[SliceItem, ...]) -> "ParallelAccess":
        if not isinstance(item, tuple):
            item = (item,)
        return ParallelAccess(base=self.base, selectors=self.selectors + tuple(item))

    def __getitem__(self, item: SliceItem | Tuple[SliceItem, ...]) -> "ParallelAccess":
        return self.append_selectors(item)

    def __setitem__(self, item: SliceItem | Tuple[SliceItem, ...], value: object) -> None:
        self.program.thread_manager.record_parallel_assignment_from_access(self.append_selectors(item), value)

    def _as_expr(self) -> "ParallelExpr":
        return ParallelExpr(kind="load", value=self)

    def __add__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__add__(other)

    def __radd__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__radd__(other)

    def __sub__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__sub__(other)

    def __rsub__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__rsub__(other)

    def __mul__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__mul__(other)

    def __rmul__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__rmul__(other)

    def __mod__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__mod__(other)

    def __rmod__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__rmod__(other)

    def __lt__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__lt__(other)

    def __le__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__le__(other)

    def __gt__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__gt__(other)

    def __ge__(self, other: object) -> "ParallelExpr":
        return self._as_expr().__ge__(other)

    def __eq__(self, other: object) -> "ParallelExpr":  # type: ignore[override]
        return self._as_expr().__eq__(other)


@dataclass(frozen=True)
class ParallelExpr:
    kind: str
    value: object = None
    args: Tuple["ParallelExpr", ...] = ()
    op: Optional[str] = None

    def _binary(self, other: object, *, op: str) -> "ParallelExpr":
        return ParallelExpr(
            kind="op",
            op=op,
            args=(self, _coerce_parallel_expr(other)),
        )

    def __add__(self, other: object) -> "ParallelExpr":
        return self._binary(other, op="add")

    def __radd__(self, other: object) -> "ParallelExpr":
        return _coerce_parallel_expr(other)._binary(self, op="add")

    def __sub__(self, other: object) -> "ParallelExpr":
        return self._binary(other, op="sub")

    def __rsub__(self, other: object) -> "ParallelExpr":
        return _coerce_parallel_expr(other)._binary(self, op="sub")

    def __mul__(self, other: object) -> "ParallelExpr":
        return self._binary(other, op="mul")

    def __rmul__(self, other: object) -> "ParallelExpr":
        return _coerce_parallel_expr(other)._binary(self, op="mul")

    def __mod__(self, other: object) -> "ParallelExpr":
        return self._binary(other, op="mod")

    def __rmod__(self, other: object) -> "ParallelExpr":
        return _coerce_parallel_expr(other)._binary(self, op="mod")

    def __lt__(self, other: object) -> "ParallelExpr":
        return self._binary(other, op="lt")

    def __le__(self, other: object) -> "ParallelExpr":
        return self._binary(other, op="le")

    def __gt__(self, other: object) -> "ParallelExpr":
        return self._binary(other, op="gt")

    def __ge__(self, other: object) -> "ParallelExpr":
        return self._binary(other, op="ge")

    def __eq__(self, other: object) -> "ParallelExpr":  # type: ignore[override]
        return self._binary(other, op="eq")


@dataclass
class ParallelAssignment:
    dst: ParallelAccess
    expr: ParallelExpr
    task_id: str
    sources: List[ParallelAccess] = field(default_factory=list)


@dataclass(frozen=True)
class ParallelCycleGroup:
    i_index: int
    j_index: int
    k_base: int
    k_count: int
    elem_width: int
    element_count: int


@dataclass(frozen=True)
class ParallelInputCacheSlotPlan:
    slot_id: int
    access: ParallelAccess
    pattern_kind: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelOutputCacheSlotPlan:
    slot_id: int
    access: ParallelAccess
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelLoadOp:
    slot_id: int
    access: ParallelAccess
    ensure_place: str = "vram"
    load_kind: str = "mapv_to_fpram"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelComputeOp:
    task_id: str
    dst_slot_id: int
    expr: ParallelExpr
    input_slot_ids: List[int] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ParallelWritebackOp:
    slot_id: int
    access: ParallelAccess
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParallelCyclePlan:
    group: ParallelCycleGroup
    input_slots: List[ParallelInputCacheSlotPlan] = field(default_factory=list)
    output_slots: List[ParallelOutputCacheSlotPlan] = field(default_factory=list)
    load_ops: List[ParallelLoadOp] = field(default_factory=list)
    compute_ops: List[ParallelComputeOp] = field(default_factory=list)
    writeback_ops: List[ParallelWritebackOp] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParallelExecutionPlan:
    region_name: str
    cycle_groups: List[ParallelCycleGroup] = field(default_factory=list)
    cycle_plans: List[ParallelCyclePlan] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParallelRegionGraph:
    region_id: int
    name: str
    extents: Tuple[int, int, int]
    axes: Tuple[ParallelAxis, ParallelAxis, ParallelAxis]
    assignments: List[ParallelAssignment] = field(default_factory=list)
    cache_plan: Dict[str, object] = field(default_factory=dict)
    execution_plan: Optional[ParallelExecutionPlan] = None

    def finalize(self, program: "TileTensorProgram") -> None:
        unique_input_identities: Dict[str, ParallelAccess] = {}
        predicate_kinds: set[str] = set()
        output_cache_count = 0
        for assignment in self.assignments:
            for access in assignment.sources:
                unique_input_identities.setdefault(_parallel_access_identity(access), access)
            for predicate in _collect_parallel_predicates(assignment.expr):
                predicate_kinds.add(predicate)
            output_cache_count = max(output_cache_count, 1)
        self.cache_plan = {
            "thread_group_size": int(program.mlen),
            "cache_shape": (1, int(program.mlen)),
            "cache_cycle_model": "uniform",
            "input_cache_count": int(len(unique_input_identities)),
            "output_cache_count": int(output_cache_count),
            "d_axis_groups": int(ceil(self.extents[2] / program.mlen)),
            "input_accesses": sorted(unique_input_identities.keys()),
            "predicate_kinds": sorted(predicate_kinds),
        }
        self.execution_plan = _build_parallel_execution_plan(self, program=program)


class _ParallelRegionScope:
    def __init__(
        self,
        program: "TileTensorProgram",
        *,
        extents: Tuple[int, int, int],
        name: Optional[str] = None,
    ) -> None:
        self.program = program
        self.extents = tuple(int(extent) for extent in extents)
        self.name = name or self.program._auto_name("parallel_region")
        self.region_id = self.program._parallel_region_counter
        self.program._parallel_region_counter += 1
        self.region: Optional[ParallelRegionGraph] = None

    def __enter__(self) -> Tuple[ParallelAxis, ParallelAxis, ParallelAxis]:
        axes = (
            ParallelAxis(self.program, self.region_id, 0, "s", self.extents[0]),
            ParallelAxis(self.program, self.region_id, 1, "h", self.extents[1]),
            ParallelAxis(self.program, self.region_id, 2, "d", self.extents[2]),
        )
        self.region = ParallelRegionGraph(
            region_id=self.region_id,
            name=self.name,
            extents=self.extents,
            axes=axes,
        )
        self.program.thread_manager._active_parallel_graphs.append(self.region)
        return axes

    def __exit__(self, exc_type, exc, tb) -> None:
        region = self.region
        if region is None:
            return
        popped = self.program.thread_manager._active_parallel_graphs.pop()
        if popped is not region:
            raise RuntimeError("parallel region stack became inconsistent")
        if exc_type is None:
            region.finalize(self.program)
            self.program.thread_manager.parallel_regions.append(region)
            if region.execution_plan is not None:
                self.program.thread_manager._emit_parallel_execution_plan(region, region.execution_plan)
                self.program._parallel_execution_lowered = True


class _ParallelRegion2DScope:
    def __init__(
        self,
        program: "TileTensorProgram",
        *,
        extents: Tuple[int, int],
        name: Optional[str] = None,
    ) -> None:
        self.program = program
        self.extents = tuple(int(extent) for extent in extents)
        self.name = name or self.program._auto_name("parallel_region2d")
        self.region_id = self.program._parallel_region_counter
        self.program._parallel_region_counter += 1
        self.region: Optional[ParallelRegionGraph] = None

    def __enter__(self) -> Tuple[ParallelAxis, ParallelAxis]:
        axes = (
            ParallelAxis(self.program, self.region_id, 0, "_", 1),
            ParallelAxis(self.program, self.region_id, 1, "h", self.extents[0]),
            ParallelAxis(self.program, self.region_id, 2, "s", self.extents[1]),
        )
        self.region = ParallelRegionGraph(
            region_id=self.region_id,
            name=self.name,
            extents=(1, self.extents[0], self.extents[1]),
            axes=axes,
        )
        self.region.cache_plan["lowering_kind"] = "fp2d"
        self.program.thread_manager._active_parallel_graphs.append(self.region)
        return axes[1], axes[2]

    def __exit__(self, exc_type, exc, tb) -> None:
        region = self.region
        if region is None:
            return
        popped = self.program.thread_manager._active_parallel_graphs.pop()
        if popped is not region:
            raise RuntimeError("parallel region stack became inconsistent")
        if exc_type is None:
            self.program.thread_manager.parallel_regions.append(region)
            self.program.thread_manager._emit_parallel2d_fp_region(region)


@dataclass
class InputTile:
    tile_id: str
    input_name: str
    coord: TileCoord
    tile_shape: Tuple[int, int]
    binding: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class TensorTile:
    tile_id: str
    tensor_name: str
    coord: TileCoord
    tile_shape: Tuple[int, int]
    binding: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class VectorTile(TensorTile):
    pass


@dataclass
class ValueTile:
    value_tile_id: str
    logical_shape: Tuple[int, int]
    from_input_tile: bool = False
    source_input_tile_id: Optional[str] = None
    residency: Dict[str, Optional[int | str]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ValueTileView:
    backing_value_tile_id: str
    owner_tile_id: str
    row_offset: int
    row_count: int
    col_offset: int
    col_count: int
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def view_id(self) -> str:
        lane = self.metadata.get("lane_index")
        if isinstance(lane, int):
            return f"{self.owner_tile_id}.lane{lane}"
        return self.owner_tile_id


@dataclass
class PreparedWrite:
    """Explicit write-preparation result for one tensor view update.

    `prepare_updated_view_value(...)` returns this object so callers do not need
    to reverse-engineer write semantics from scattered booleans.
    """
    old_value: ValueTile
    new_value: ValueTile
    target_view: ValueTileView
    reuse_old: bool
    requires_preserve_copy: bool = False


@dataclass
class Input:
    program: "TileTensorProgram"
    name: str
    logical_shape: LogicalShape
    tiles: Dict[TileCoord, InputTile] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tiles = self.program.tensor_manager.create_input_tiles(self.name, self.logical_shape)

    def __getitem__(self, item: SliceItem | Tuple[SliceItem, ...]) -> "InputSlice":
        if not isinstance(item, tuple):
            item = (item,)
        if _contains_parallel_selector(item):
            return ParallelAccess(base=self, selectors=item)
        if _is_full_element_index(item, len(self.logical_shape)):
            return ElementRef(base=self, indices=tuple(int(index) for index in item))
        return InputSlice(base=self, selectors=item)

    def __setitem__(self, item: SliceItem | Tuple[SliceItem, ...], value: object) -> None:
        self.program.thread_manager.record_parallel_assignment_from_index(self, item, value)

    @property
    def T(self) -> "InputTranspose":
        return InputTranspose(base=self)


@dataclass
class Tensor:
    program: "TileTensorProgram"
    name: str
    logical_shape: LogicalShape
    tiles: Dict[TileCoord, TensorTile] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tiles = self.program.tensor_manager.create_tensor_tiles(self.name, self.logical_shape)

    def __getitem__(self, item: SliceItem | Tuple[SliceItem, ...]) -> "TensorSlice":
        if not isinstance(item, tuple):
            item = (item,)
        if _contains_parallel_selector(item):
            return ParallelAccess(base=self, selectors=item)
        if _is_full_element_index(item, len(self.logical_shape)):
            return ElementRef(base=self, indices=tuple(int(index) for index in item))
        return TensorSlice(base=self, selectors=item)

    def __setitem__(self, item: SliceItem | Tuple[SliceItem, ...], value: object) -> None:
        self.program.thread_manager.record_parallel_assignment_from_index(self, item, value)

    @property
    def T(self) -> "TensorTranspose":
        return TensorTranspose(base=self)


@dataclass
class Vector(Tensor):
    tiles: Dict[TileCoord, VectorTile] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tiles = self.program.tensor_manager.create_vector_tiles(self.name, self.logical_shape)

    def __getitem__(self, item: SliceItem | Tuple[SliceItem, ...]) -> "VectorSlice":
        if not isinstance(item, tuple):
            item = (item,)
        if _contains_parallel_selector(item):
            return ParallelAccess(base=self, selectors=item)
        if _is_full_element_index(item, len(self.logical_shape)):
            return ElementRef(base=self, indices=tuple(int(index) for index in item))
        return VectorSlice(base=self, selectors=item)

    def __setitem__(self, item: SliceItem | Tuple[SliceItem, ...], value: object) -> None:
        self.program.thread_manager.record_parallel_assignment_from_index(self, item, value)

    @property
    def T(self) -> "VectorTranspose":
        return VectorTranspose(base=self)


@dataclass
class InputSlice:
    base: Input
    selectors: Tuple[SliceItem, ...]


@dataclass
class TensorSlice:
    base: Tensor
    selectors: Tuple[SliceItem, ...]


@dataclass
class VectorSlice:
    base: Vector
    selectors: Tuple[SliceItem, ...]


@dataclass(frozen=True)
class InputTranspose:
    base: Input

    @property
    def program(self) -> "TileTensorProgram":
        return self.base.program

    @property
    def name(self) -> str:
        return f"{self.base.name}.T"

    @property
    def logical_shape(self) -> LogicalShape:
        return self.base.logical_shape

    @property
    def tiles(self) -> Dict[TileCoord, InputTile]:
        return self.base.tiles


@dataclass(frozen=True)
class TensorTranspose:
    base: Tensor

    @property
    def program(self) -> "TileTensorProgram":
        return self.base.program

    @property
    def name(self) -> str:
        return f"{self.base.name}.T"

    @property
    def logical_shape(self) -> LogicalShape:
        return self.base.logical_shape

    @property
    def tiles(self) -> Dict[TileCoord, TensorTile]:
        return self.base.tiles


@dataclass(frozen=True)
class VectorTranspose:
    base: Vector

    @property
    def program(self) -> "TileTensorProgram":
        return self.base.program

    @property
    def name(self) -> str:
        return f"{self.base.name}.T"

    @property
    def logical_shape(self) -> LogicalShape:
        return self.base.logical_shape

    @property
    def tiles(self) -> Dict[TileCoord, VectorTile]:
        return self.base.tiles


TileLike = TensorTile | InputTile | VectorTile
TensorLike = Tensor | Input | Vector
TransposedTensorLike = TensorTranspose | InputTranspose | VectorTranspose
SourceValueLike = ValueTile
RowOperandLike = ValueTile | ValueTileView
ViewMatmulTerm = Tuple[List[TileLike], TileLike]
ViewMatmulThread = Tuple[TileLike, List[ViewMatmulTerm], int]
BTMMHeadGroupThread = Dict[str, object]
CopyMapvPacket = Tuple[str, ValueTile, TileLike]
MatmulMapvPacket = Tuple[str, List[List[SourceValueLike]], ValueTile, TileLike]
GemmMapvPacket = Tuple[str, List[List[SourceValueLike]], ValueTile, ValueTile, TileLike]
MapvPacket = CopyMapvPacket | MatmulMapvPacket | GemmMapvPacket


class HardwareManager:
    """Registry for simulated HBM/VRAM/MRAM objects and placement metadata.

    This layer tracks hardware-visible objects only. It does not own tensor
    grouping, value/scatter binding policy, or compute semantics.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.hbm_objects: Dict[str, Dict[str, object]] = {}
        self.vram_objects: Dict[str, Dict[str, object]] = {}
        self.mram_objects: Dict[str, Dict[str, object]] = {}


class ThreadManager:
    """Manage parallel thread regions, expression graphs, and cache planning.

    This layer is intentionally FP-first for now. It owns the symbolic
    `parallel_region3d` flow and keeps the graph/cache planning state out of
    `TileTensorProgram` so later lowering can evolve independently.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self._active_parallel_graphs: List[ParallelRegionGraph] = []
        self.parallel_regions: List[ParallelRegionGraph] = []
        self._parallel2d_scratch_var_cache: Dict[Tuple[int, int], List[FPVar]] = {}

    def parallel_region3d(
        self,
        extents: Tuple[int, int, int] | List[int],
        *,
        name: Optional[str] = None,
    ) -> _ParallelRegionScope:
        normalized = tuple(int(extent) for extent in extents)
        if len(normalized) != 3 or any(extent <= 0 for extent in normalized):
            raise ValueError(f"parallel_region3d expects three positive extents, got {extents}")
        return _ParallelRegionScope(self.program, extents=normalized, name=name)

    def parallel_region2d(
        self,
        extents: Tuple[int, int] | List[int],
        *,
        name: Optional[str] = None,
    ) -> _ParallelRegion2DScope:
        normalized = tuple(int(extent) for extent in extents)
        if len(normalized) != 2 or any(extent <= 0 for extent in normalized):
            raise ValueError(f"parallel_region2d expects two positive extents, got {extents}")
        return _ParallelRegion2DScope(self.program, extents=normalized, name=name)

    def where(self, predicate: object, on_true: object, on_false: object) -> ParallelExpr:
        return ParallelExpr(
            kind="select",
            args=(
                _coerce_parallel_expr(predicate),
                _coerce_parallel_expr(on_true),
                _coerce_parallel_expr(on_false),
            ),
        )

    def if_then_else(self, predicate: object, on_true: object, on_false: object) -> ParallelExpr:
        return self.where(predicate, on_true, on_false)

    def pair(self, axis: object) -> ParallelExpr:
        # RoPE-style lane pairing helper: pair(2k)=2k+1, pair(2k+1)=2k.
        return ParallelExpr(kind="pair_index", args=(_coerce_parallel_expr(axis),))

    def half_index(self, axis: object) -> ParallelExpr:
        # RoPE coefficient group helper: half_index(d)=d//2.
        # Runtime planning assumes coefficients may be pre-expanded to full-lane layout.
        return ParallelExpr(kind="half_index", args=(_coerce_parallel_expr(axis),))

    def current_parallel_graph(self) -> ParallelRegionGraph:
        if not self._active_parallel_graphs:
            raise RuntimeError("parallel graph write requested outside active parallel_region3d")
        return self._active_parallel_graphs[-1]

    def record_parallel_assignment_from_index(
        self,
        base: object,
        item: SliceItem | Tuple[SliceItem, ...],
        value: object,
    ) -> None:
        if not isinstance(item, tuple):
            item = (item,)
        self.record_parallel_assignment_from_access(ParallelAccess(base=base, selectors=tuple(item)), value)

    def record_parallel_assignment_from_access(self, dst_access: ParallelAccess, value: object) -> None:
        region = self.current_parallel_graph()
        if not isinstance(dst_access.base, Tensor):
            raise TypeError(
                "parallel assignment destination must be a Tensor-backed access; "
                f"got {type(dst_access.base).__name__}"
            )
        if len(dst_access.selectors) != len(dst_access.logical_shape):
            raise ValueError(
                f"parallel assignment target must fully index rank-{len(dst_access.logical_shape)} tensor, "
                f"got selectors={dst_access.selectors}"
            )
        expr = _coerce_parallel_expr(value)
        if region.cache_plan.get("lowering_kind") == "fp2d":
            self._validate_parallel2d_fp_assignment(region, dst_access, expr)
        else:
            self._validate_parallel_assignment(region, dst_access, expr)
        assignment = ParallelAssignment(
            dst=dst_access,
            expr=expr,
            task_id=self.program._auto_name(f"{region.name}.assign"),
            sources=_collect_parallel_accesses(expr),
        )
        region.assignments.append(assignment)

    def _validate_parallel2d_fp_assignment(
        self,
        region: ParallelRegionGraph,
        dst_access: ParallelAccess,
        expr: ParallelExpr,
    ) -> None:
        if not isinstance(dst_access.base, Vector):
            raise TypeError(
                "parallel_region2d currently supports FP-backed Vector destinations only; "
                f"got {type(dst_access.base).__name__}"
            )
        axis_refs = [selector for selector in dst_access.selectors if isinstance(selector, ParallelAxis)]
        axis_ids = {int(axis.axis) for axis in axis_refs}
        expected_axis_ids = {1, 2}
        if int(region.extents[1]) == 1:
            expected_axis_ids = {2}
        if axis_ids != expected_axis_ids:
            raise ValueError(
                "parallel_region2d destination must index with its active axes; "
                f"got selectors={dst_access.selectors}"
            )
        axis_region_ids = {axis.region_id for axis in axis_refs}
        if axis_region_ids != {region.region_id}:
            raise ValueError("parallel_region2d destination mixes axes from another parallel region")
        self._validate_parallel_expr(expr, region=region)

    def _validate_parallel_assignment(
        self,
        region: ParallelRegionGraph,
        dst_access: ParallelAccess,
        expr: ParallelExpr,
    ) -> None:
        axis_refs = [selector for selector in dst_access.selectors if isinstance(selector, ParallelAxis)]
        if len(axis_refs) != 3:
            raise ValueError(
                "parallel assignment destination must index with exactly the active 3D parallel axes; "
                f"got selectors={dst_access.selectors}"
            )
        axis_region_ids = {axis.region_id for axis in axis_refs}
        if axis_region_ids != {region.region_id}:
            raise ValueError("parallel assignment destination mixes axes from another parallel region")
        self._validate_parallel_expr(expr, region=region)

    def _validate_parallel_expr(
        self,
        expr: ParallelExpr,
        *,
        region: ParallelRegionGraph,
    ) -> None:
        if expr.kind in {"literal", "axis", "fpvar"}:
            return
        if expr.kind == "load":
            access = expr.value
            if not isinstance(access, ParallelAccess):
                raise TypeError(f"parallel load expected ParallelAccess, got {type(access).__name__}")
            axis_region_ids = {
                selector.region_id
                for selector in access.selectors
                if isinstance(selector, ParallelAxis)
            }
            if axis_region_ids and axis_region_ids != {region.region_id}:
                raise ValueError("parallel expression mixes axes from another parallel region")
            return
        if expr.kind == "select":
            if len(expr.args) != 3:
                raise ValueError("parallel select expression expects exactly three arguments")
            self._validate_parallel_predicate(expr.args[0], region=region)
            for arg in expr.args[1:]:
                self._validate_parallel_expr(arg, region=region)
            return
        if expr.kind == "op":
            if expr.op not in {"add", "sub", "mul", "max"} or len(expr.args) != 2:
                raise NotImplementedError(
                    f"parallel expression currently supports only binary add/sub/mul/max, got {expr.op!r}"
                )
            for arg in expr.args:
                self._validate_parallel_expr(arg, region=region)
            return
        if expr.kind == "unary_op":
            if expr.op not in {"exp", "reci", "sqrt"} or len(expr.args) != 1:
                raise NotImplementedError(
                    f"parallel expression currently supports unary exp/reci/sqrt, got {expr.op!r}"
                )
            self._validate_parallel_expr(expr.args[0], region=region)
            return
        if expr.kind in {"pair_index", "half_index"}:
            for arg in expr.args:
                self._validate_parallel_expr(arg, region=region)
            return
        raise NotImplementedError(f"Unsupported parallel expression kind: {expr.kind}")

    def _validate_parallel_predicate(
        self,
        expr: ParallelExpr,
        *,
        region: ParallelRegionGraph,
    ) -> None:
        if expr.kind == "op" and expr.op in {"lt", "le", "gt", "ge", "eq"} and len(expr.args) == 2:
            for arg in expr.args:
                self._validate_parallel_index_expr(arg, region=region)
            return
        raise NotImplementedError(
            "parallel predicates currently support only binary comparisons "
            f"(lt/le/gt/ge/eq), got {expr.kind}:{getattr(expr, 'op', None)!r}"
        )

    def _validate_parallel_index_expr(
        self,
        expr: ParallelExpr,
        *,
        region: ParallelRegionGraph,
    ) -> None:
        if expr.kind in {"literal", "axis"}:
            return
        if expr.kind == "op" and expr.op in {"add", "sub", "mul", "mod"} and len(expr.args) == 2:
            for arg in expr.args:
                self._validate_parallel_index_expr(arg, region=region)
            return
        raise NotImplementedError(
            f"parallel predicate index expression currently supports only axis/literal/add/sub/mul/mod, got {expr.kind}:{getattr(expr, 'op', None)!r}"
        )

    def parallel_execution_plans(self) -> List[ParallelExecutionPlan]:
        plans: List[ParallelExecutionPlan] = []
        for region in self.parallel_regions:
            if region.execution_plan is not None:
                plans.append(region.execution_plan)
        return plans

    def lower_parallel_execution_plans(self) -> None:
        if self.program._parallel_execution_lowered:
            return
        for region in self.parallel_regions:
            if region.execution_plan is None:
                continue
            self._emit_parallel_execution_plan(region, region.execution_plan)
        self.program._parallel_execution_lowered = True

    def _emit_parallel_execution_plan(
        self,
        region: ParallelRegionGraph,
        execution_plan: ParallelExecutionPlan,
    ) -> None:
        if not execution_plan.cycle_plans:
            raise RuntimeError(f"parallel region {region.name} finalized without any cycle plans")
        self._prepare_parallel_region_output_bindings(region)
        region_output_values: Dict[str, Tuple[TensorTile, ValueTile]] = {}
        for cycle_plan in execution_plan.cycle_plans:
            self._emit_parallel_cycle_plan(region, cycle_plan, region_output_values=region_output_values)
        for dst_tile, output_value in region_output_values.values():
            self.program.value_manager._bind_value_to_tensor_tile(output_value, dst_tile)

    def _emit_parallel2d_fp_region(self, region: ParallelRegionGraph) -> None:
        if region.cache_plan.get("lowering_kind") != "fp2d":
            raise RuntimeError(f"parallel2d fp lowering got non-fp2d region {region.name}")
        _, head_count, lane_count = (int(extent) for extent in region.extents)
        for assignment in region.assignments:
            if not isinstance(assignment.dst.base, Vector):
                raise TypeError(
                    "parallel_region2d lowering supports only FP-backed Vector destinations; "
                    f"got {type(assignment.dst.base).__name__}"
                )
            for head_index in range(head_count):
                dst_vars = self._parallel2d_access_vars(
                    assignment.dst,
                    head_index=head_index,
                    lane_count=lane_count,
                )
                self._emit_parallel2d_fp_expr_kernel(
                    assignment.expr,
                    dst_vars=dst_vars,
                    head_index=head_index,
                    lane_count=lane_count,
                    task_id=f"{assignment.task_id}.h{head_index}",
                )

    def _emit_parallel2d_fp_expr_kernel(
        self,
        expr: ParallelExpr,
        *,
        dst_vars: Sequence[FPVar],
        head_index: int,
        lane_count: int,
        task_id: str,
    ) -> None:
        self._parallel2d_materialize_expr_into(
            expr,
            head_index=head_index,
            lane_count=lane_count,
            task_id=task_id,
            dst_vars=list(dst_vars),
            temp_depth=0,
        )

    def _parallel2d_materialize_expr_into(
        self,
        expr: ParallelExpr,
        *,
        head_index: int,
        lane_count: int,
        task_id: str,
        dst_vars: Sequence[FPVar],
        temp_depth: int,
    ) -> None:
        if expr.kind in {"load", "fpvar", "literal"}:
            leaf_vars = self._parallel2d_expr_vars(expr, head_index=head_index, lane_count=lane_count)
            self.program.emit_fp_kernel(
                src1_addrs=[_require_fp_addr(var) for var in leaf_vars],
                dst_addrs=[_require_fp_addr(var) for var in dst_vars],
                op="copy",
                task_id=task_id,
            )
            return
        if expr.kind == "op":
            if expr.op not in {"add", "sub", "mul", "max"} or len(expr.args) != 2:
                raise NotImplementedError(
                    f"parallel_region2d FP lowering supports binary add/sub/mul/max, got {expr.op!r}"
                )
            lhs_expr, rhs_expr = expr.args
            if lhs_expr.kind in {"load", "fpvar", "literal"}:
                src1_vars = self._parallel2d_expr_vars(lhs_expr, head_index=head_index, lane_count=lane_count)
            else:
                self._parallel2d_materialize_expr_into(
                    lhs_expr,
                    head_index=head_index,
                    lane_count=lane_count,
                    task_id=f"{task_id}.lhs",
                    dst_vars=dst_vars,
                    temp_depth=temp_depth,
                )
                src1_vars = list(dst_vars)

            if rhs_expr.kind in {"load", "fpvar", "literal"}:
                src2_vars = self._parallel2d_expr_vars(rhs_expr, head_index=head_index, lane_count=lane_count)
            else:
                src2_vars = self._parallel2d_scratch_vars(
                    lane_count=lane_count,
                    slot_index=temp_depth,
                )
                self._parallel2d_materialize_expr_into(
                    rhs_expr,
                    head_index=head_index,
                    lane_count=lane_count,
                    task_id=f"{task_id}.rhs",
                    dst_vars=src2_vars,
                    temp_depth=temp_depth + 1,
                )
            self.program.emit_fp_kernel(
                src1_addrs=[_require_fp_addr(var) for var in src1_vars],
                src2_addrs=[_require_fp_addr(var) for var in src2_vars],
                dst_addrs=[_require_fp_addr(var) for var in dst_vars],
                op=str(expr.op),
                task_id=task_id,
            )
            return
        if expr.kind == "unary_op":
            if expr.op not in {"exp", "reci", "sqrt"} or len(expr.args) != 1:
                raise NotImplementedError(
                    f"parallel_region2d FP lowering supports unary exp/reci/sqrt, got {expr.op!r}"
                )
            arg_expr = expr.args[0]
            if arg_expr.kind in {"load", "fpvar", "literal"}:
                src_vars = self._parallel2d_expr_vars(arg_expr, head_index=head_index, lane_count=lane_count)
            else:
                self._parallel2d_materialize_expr_into(
                    arg_expr,
                    head_index=head_index,
                    lane_count=lane_count,
                    task_id=f"{task_id}.src",
                    dst_vars=dst_vars,
                    temp_depth=temp_depth,
                )
                src_vars = list(dst_vars)
            self.program.emit_fp_kernel(
                src1_addrs=[_require_fp_addr(var) for var in src_vars],
                dst_addrs=[_require_fp_addr(var) for var in dst_vars],
                op=str(expr.op),
                task_id=task_id,
            )
            return
        raise NotImplementedError(
            f"parallel_region2d FP lowering does not support expr kind {expr.kind!r}"
        )

    def _parallel2d_scratch_vars(
        self,
        *,
        lane_count: int,
        slot_index: int,
    ) -> List[FPVar]:
        key = (int(lane_count), int(slot_index))
        cached = self._parallel2d_scratch_var_cache.get(key)
        if cached is not None:
            return cached
        fragment = self.program.fp_fragment(
            self.program._auto_name(f"parallel2d.scratch{slot_index}"),
            (int(lane_count),),
            init=0.0,
        )
        scratch_vars = self.program.mapf(fragment)
        self._parallel2d_scratch_var_cache[key] = scratch_vars
        return scratch_vars

    def _parallel2d_expr_vars(
        self,
        expr: ParallelExpr,
        *,
        head_index: int,
        lane_count: int,
    ) -> List[FPVar]:
        if expr.kind == "load":
            access = expr.value
            if not isinstance(access, ParallelAccess):
                raise RuntimeError(f"parallel_region2d load expr missing ParallelAccess: {expr}")
            return self._parallel2d_access_vars(access, head_index=head_index, lane_count=lane_count)
        if expr.kind == "fpvar":
            fp_var = expr.value
            if not isinstance(fp_var, FPVar):
                raise RuntimeError(f"parallel_region2d fpvar expr missing FPVar payload: {expr}")
            return [fp_var] * int(lane_count)
        if expr.kind == "literal":
            literal_var = self.program.mapf(float(expr.value))[0]
            return [literal_var] * int(lane_count)
        raise NotImplementedError(
            f"parallel_region2d FP kernel operands currently support load/fpvar/literal, got {expr.kind}"
        )

    def _parallel2d_access_vars(
        self,
        access: ParallelAccess,
        *,
        head_index: int,
        lane_count: int,
    ) -> List[FPVar]:
        if not isinstance(access.base, Vector):
            raise TypeError(
                "parallel_region2d FP access supports only Vector operands; "
                f"got {type(access.base).__name__}"
            )
        resolved: List[FPVar] = []
        for lane_index in range(int(lane_count)):
            logical_index = self._parallel2d_access_logical_index(
                access,
                head_index=head_index,
                lane_index=lane_index,
            )
            resolved.append(
                self.program.tensor_manager._resolve_element_fpvar(
                    ElementRef(base=access.base, indices=logical_index)
                )
            )
        return resolved

    def _parallel2d_access_logical_index(
        self,
        access: ParallelAccess,
        *,
        head_index: int,
        lane_index: int,
    ) -> Tuple[int, ...]:
        logical_index: List[int] = []
        for selector in access.selectors:
            if isinstance(selector, ParallelAxis):
                if int(selector.axis) == 1:
                    logical_index.append(int(head_index))
                elif int(selector.axis) == 2:
                    logical_index.append(int(lane_index))
                else:
                    raise RuntimeError(f"parallel_region2d does not expose axis {selector.axis}")
            elif isinstance(selector, ParallelExpr):
                logical_index.append(
                    self._parallel2d_index_expr_value(
                        selector,
                        head_index=head_index,
                        lane_index=lane_index,
                    )
                )
            elif isinstance(selector, int):
                logical_index.append(int(selector))
            else:
                raise NotImplementedError(
                    f"parallel_region2d does not support selector {selector!r} of type {type(selector).__name__}"
                )
        return tuple(logical_index)

    def _parallel2d_index_expr_value(
        self,
        expr: ParallelExpr,
        *,
        head_index: int,
        lane_index: int,
    ) -> int:
        if expr.kind == "literal":
            return int(expr.value)
        if expr.kind == "axis":
            axis = expr.value
            if not isinstance(axis, ParallelAxis):
                raise RuntimeError("parallel_region2d axis expression missing axis metadata")
            if int(axis.axis) == 1:
                return int(head_index)
            if int(axis.axis) == 2:
                return int(lane_index)
            raise RuntimeError(f"parallel_region2d does not expose axis {axis.axis}")
        if expr.kind == "op" and len(expr.args) == 2:
            lhs = self._parallel2d_index_expr_value(expr.args[0], head_index=head_index, lane_index=lane_index)
            rhs = self._parallel2d_index_expr_value(expr.args[1], head_index=head_index, lane_index=lane_index)
            if expr.op == "add":
                return lhs + rhs
            if expr.op == "sub":
                return lhs - rhs
            if expr.op == "mul":
                return lhs * rhs
            if expr.op == "mod":
                return lhs % rhs
        raise NotImplementedError(
            f"Unsupported parallel_region2d index expression: {expr.kind}:{getattr(expr, 'op', None)!r}"
        )

    def _emit_parallel_cycle_plan(
        self,
        region: ParallelRegionGraph,
        cycle_plan: ParallelCyclePlan,
        *,
        region_output_values: Dict[str, Tuple[TensorTile, ValueTile]],
    ) -> None:
        group = cycle_plan.group
        if not cycle_plan.output_slots:
            raise RuntimeError(f"parallel cycle {region.name} has no output slots")
        if not cycle_plan.compute_ops:
            raise RuntimeError(f"parallel cycle {region.name} has no compute ops")
        if int(group.element_count) != int(self.program.mlen):
            raise NotImplementedError(
                "parallel lowering requires element_count == mlen per cycle "
                f"(elem_width={group.elem_width}, k_count={group.k_count}, "
                f"element_count={group.element_count}, mlen={self.program.mlen})"
            )

        cache_tag = f"{region.name}.i{group.i_index}.j{group.j_index}.k{group.k_base}"
        (
            input_slot_bases,
            output_slot_bases,
            input_slot_names,
            output_slot_names,
        ) = self._allocate_parallel_cycle_cache_slots(cycle_plan, cache_tag)
        output_slot_values = self._resolve_parallel_cycle_output_values(
            cycle_plan,
            group,
            region_output_values=region_output_values,
        )

        try:
            self._emit_parallel_cycle_loads(cycle_plan, group, cache_tag, input_slot_bases)
            self._emit_parallel_cycle_compute(cycle_plan, group, input_slot_bases, output_slot_bases)
            self._emit_parallel_cycle_writebacks(
                cycle_plan,
                group,
                cache_tag,
                output_slot_bases,
                output_slot_values,
            )
        finally:
            self._free_parallel_cycle_cache_slots(input_slot_names, output_slot_names)

    def _allocate_parallel_cycle_cache_slots(
        self,
        cycle_plan: ParallelCyclePlan,
        cache_tag: str,
    ) -> Tuple[Dict[int, int], Dict[int, int], List[str], List[str]]:
        allocator = self.program.compiler.sub_matrix_manager.fpram_allocator
        cache_floor = int(self.program.tensor_manager._next_fp_mem_addr)
        if allocator.next_free < cache_floor:
            allocator.next_free = cache_floor
        allocator.free_stack[:] = [block for block in allocator.free_stack if int(block.addr) >= cache_floor]

        input_slot_bases: Dict[int, int] = {}
        output_slot_bases: Dict[int, int] = {}
        input_slot_names: List[str] = []
        output_slot_names: List[str] = []
        for input_slot in cycle_plan.input_slots:
            slot_name = f"__parallel_input_cache__.{cache_tag}.slot{input_slot.slot_id}"
            input_slot_bases[input_slot.slot_id] = int(allocator.allocate(slot_name, self.program.mlen))
            input_slot_names.append(slot_name)
        for output_slot in cycle_plan.output_slots:
            slot_name = f"__parallel_output_cache__.{cache_tag}.slot{output_slot.slot_id}"
            output_slot_bases[output_slot.slot_id] = int(allocator.allocate(slot_name, self.program.mlen))
            output_slot_names.append(slot_name)
        return input_slot_bases, output_slot_bases, input_slot_names, output_slot_names

    def _resolve_parallel_cycle_output_values(
        self,
        cycle_plan: ParallelCyclePlan,
        group: ParallelCycleGroup,
        *,
        region_output_values: Dict[str, Tuple[TensorTile, ValueTile]],
    ) -> Dict[int, ValueTile]:
        output_slot_values: Dict[int, ValueTile] = {}
        for output_slot in cycle_plan.output_slots:
            output_tile = self._parallel_access_cycle_dst_tile(output_slot.access, group)
            output_slot_values[output_slot.slot_id] = self._get_or_create_parallel_region_output_value(
                output_tile,
                region_output_values=region_output_values,
            )
        return output_slot_values

    def _emit_parallel_cycle_loads(
        self,
        cycle_plan: ParallelCyclePlan,
        group: ParallelCycleGroup,
        cache_tag: str,
        input_slot_bases: Dict[int, int],
    ) -> None:
        for load_op in cycle_plan.load_ops:
            src_vram_addr = self._parallel_access_cycle_src_vram_row_addr(load_op.access, group)
            self.program.emit_map_fp_v_tile(
                fpram_addr=input_slot_bases[load_op.slot_id],
                vram_addr=src_vram_addr,
                row_count=1,
                row_width=self.program.mlen,
                task_id=f"parallel_load.{cache_tag}.slot{load_op.slot_id}",
            )

    def _emit_parallel_cycle_compute(
        self,
        cycle_plan: ParallelCyclePlan,
        group: ParallelCycleGroup,
        input_slot_bases: Dict[int, int],
        output_slot_bases: Dict[int, int],
    ) -> None:
        for compute_op in cycle_plan.compute_ops:
            access_order = _collect_parallel_accesses(compute_op.expr)
            access_slot_map = {
                _parallel_access_identity(access): slot_id
                for access, slot_id in zip(access_order, compute_op.input_slot_ids)
            }
            dst_base = output_slot_bases[compute_op.dst_slot_id]
            if self._try_emit_parallel_pairwise_cloop_compute(
                compute_op=compute_op,
                group=group,
                access_slot_map=access_slot_map,
                input_slot_bases=input_slot_bases,
                dst_base=int(dst_base),
            ):
                continue
            for lane_offset in range(self.program.mlen):
                dst_addr = int(dst_base + lane_offset)
                self._emit_parallel_expr_to_addr(
                    expr=compute_op.expr,
                    dst_addr=dst_addr,
                    lane_offset=lane_offset,
                    group=group,
                    access_slot_map=access_slot_map,
                    input_slot_bases=input_slot_bases,
                    task_id=f"{compute_op.task_id}.lane{lane_offset}",
                )

    def _emit_parallel_cycle_writebacks(
        self,
        cycle_plan: ParallelCyclePlan,
        group: ParallelCycleGroup,
        cache_tag: str,
        output_slot_bases: Dict[int, int],
        output_slot_values: Dict[int, ValueTile],
    ) -> None:
        for writeback_op in cycle_plan.writeback_ops:
            output_value = output_slot_values[writeback_op.slot_id]
            dst_vram_addr = output_value.residency.get("vram_addr")
            if dst_vram_addr is None:
                raise RuntimeError("parallel output writeback expected new value tile in VRAM")
            dst_row = self._parallel_access_cycle_row(writeback_op.access, group)
            dst_vram_row_addr = int(dst_vram_addr) + (int(dst_row) % self.program.mlen) * self.program.mlen
            self.program.emit_map_v_fp_tile(
                vram_addr=dst_vram_row_addr,
                fpram_addr=output_slot_bases[writeback_op.slot_id],
                row_count=1,
                row_width=self.program.mlen,
                task_id=f"parallel_writeback.{cache_tag}.slot{writeback_op.slot_id}",
            )

    def _free_parallel_cycle_cache_slots(
        self,
        input_slot_names: List[str],
        output_slot_names: List[str],
    ) -> None:
        allocator = self.program.compiler.sub_matrix_manager.fpram_allocator
        for slot_name in input_slot_names:
            allocator.free(slot_name, strict=False)
        for slot_name in output_slot_names:
            allocator.free(slot_name, strict=False)

    def _prepare_parallel_region_output_bindings(self, region: ParallelRegionGraph) -> None:
        detached_tile_ids: set[str] = set()
        if region.execution_plan is None:
            return
        for cycle_plan in region.execution_plan.cycle_plans:
            for output_slot in cycle_plan.output_slots:
                dst_tile = self._parallel_access_cycle_dst_tile(output_slot.access, cycle_plan.group)
                if dst_tile.tile_id in detached_tile_ids:
                    continue
                self.program.value_manager._unbind_tile_value_pointer(dst_tile.tile_id)
                detached_tile_ids.add(dst_tile.tile_id)

    def _create_parallel_region_output_value(self, dst_tile: TensorTile) -> ValueTile:
        value = ValueTile(
            value_tile_id=self.program.value_manager._next_value_tile_id(),
            logical_shape=dst_tile.tile_shape,
            metadata={"source_tile_id": dst_tile.tile_id, "parallel_region_output": True},
        )
        vram_name = f"{value.value_tile_id}.vram"
        vram_addr = self.program.value_manager.allocate_value_tile_address(
            size=self.program.tile_elems,
            name=vram_name,
            place="vram",
            value_tile=value,
        )
        value.residency["vram_addr"] = vram_addr
        value.residency["vram_name"] = vram_name
        self.program.value_manager.value_tiles[value.value_tile_id] = value
        self.program.value_manager._value_tiles_in_vram[value.value_tile_id] = int(vram_addr)
        return value

    def _get_or_create_parallel_region_output_value(
        self,
        dst_tile: TensorTile,
        *,
        region_output_values: Dict[str, Tuple[TensorTile, ValueTile]],
    ) -> ValueTile:
        existing = region_output_values.get(dst_tile.tile_id)
        if existing is not None:
            return existing[1]
        created = self._create_parallel_region_output_value(dst_tile)
        region_output_values[dst_tile.tile_id] = (dst_tile, created)
        return created

    def _try_emit_parallel_pairwise_cloop_compute(
        self,
        *,
        compute_op: ParallelComputeOp,
        group: ParallelCycleGroup,
        access_slot_map: Dict[str, int],
        input_slot_bases: Dict[int, int],
        dst_base: int,
    ) -> bool:
        rope_inputs = self._match_parallel_pairwise_rope_expr(compute_op.expr)
        if rope_inputs is None:
            return False
        if self.program.mlen % 2 != 0:
            return False

        x_slot = access_slot_map.get(_parallel_access_identity(rope_inputs["x_direct"]))
        cos_slot = access_slot_map.get(_parallel_access_identity(rope_inputs["cos"]))
        sin_slot = access_slot_map.get(_parallel_access_identity(rope_inputs["sin"]))
        neg_sin_slot = access_slot_map.get(_parallel_access_identity(rope_inputs["neg_sin"]))
        if any(slot is None for slot in (x_slot, cos_slot, sin_slot, neg_sin_slot)):
            return False

        gp_regs = self.program.compiler.register_allocator.allocate_gp(6)
        gp_x, gp_cos, gp_sin, gp_neg_sin, gp_dst, gp_loop = gp_regs
        try:
            lines = [f"; parallel pairwise cloop compute {compute_op.task_id}"]
            lines.append(f"S_ADDI_INT gp{gp_x}, gp0, {int(input_slot_bases[int(x_slot)])}")
            lines.append(f"S_ADDI_INT gp{gp_cos}, gp0, {int(input_slot_bases[int(cos_slot)])}")
            lines.append(f"S_ADDI_INT gp{gp_sin}, gp0, {int(input_slot_bases[int(sin_slot)])}")
            lines.append(f"S_ADDI_INT gp{gp_neg_sin}, gp0, {int(input_slot_bases[int(neg_sin_slot)])}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_base)}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {self.program.mlen // 2}")
            lines.append(f"S_LD_FP f1, gp{gp_x}, 0")
            lines.append(f"S_LD_FP f2, gp{gp_x}, 1")
            lines.append(f"S_LD_FP f3, gp{gp_cos}, 0")
            lines.append(f"S_LD_FP f4, gp{gp_neg_sin}, 0")
            lines.append(f"S_MUL_FP f5, f1, f3")
            lines.append(f"S_MUL_FP f6, f2, f4")
            lines.append(f"S_ADD_FP f5, f5, f6")
            lines.append(f"S_ST_FP f5, gp{gp_dst}, 0")
            lines.append(f"S_LD_FP f4, gp{gp_sin}, 0")
            lines.append(f"S_MUL_FP f5, f1, f4")
            lines.append(f"S_MUL_FP f6, f2, f3")
            lines.append(f"S_ADD_FP f5, f5, f6")
            lines.append(f"S_ST_FP f5, gp{gp_dst}, 1")
            lines.append(f"S_ADDI_INT gp{gp_x}, gp{gp_x}, 2")
            lines.append(f"S_ADDI_INT gp{gp_cos}, gp{gp_cos}, 2")
            lines.append(f"S_ADDI_INT gp{gp_sin}, gp{gp_sin}, 2")
            lines.append(f"S_ADDI_INT gp{gp_neg_sin}, gp{gp_neg_sin}, 2")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 2")
            lines.append(f"C_LOOP_END gp{gp_loop}")
            self.program.compiler.generated_code += "\n".join(lines) + "\n"
        finally:
            self.program.compiler.register_allocator.free_gp(gp_regs)
        return True

    def _match_parallel_pairwise_rope_expr(
        self,
        expr: ParallelExpr,
    ) -> Optional[Dict[str, ParallelAccess]]:
        if expr.kind != "select" or len(expr.args) != 3:
            return None
        predicate, even_expr, odd_expr = expr.args
        if not self._parallel_expr_matches_even_mod2(predicate):
            return None
        if even_expr.kind != "op" or even_expr.op != "add" or len(even_expr.args) != 2:
            return None
        if odd_expr.kind != "op" or odd_expr.op != "add" or len(odd_expr.args) != 2:
            return None

        even_terms = self._collect_parallel_mul_terms(even_expr)
        odd_terms = self._collect_parallel_mul_terms(odd_expr)
        if even_terms is None or odd_terms is None:
            return None

        x_base_id = self._resolve_parallel_pairwise_data_base_id(even_terms + odd_terms)
        if x_base_id is None:
            return None

        even_direct = self._find_parallel_term(even_terms, data_base_id=x_base_id, pair=False)
        even_pair = self._find_parallel_term(even_terms, data_base_id=x_base_id, pair=True)
        odd_direct = self._find_parallel_term(odd_terms, data_base_id=x_base_id, pair=False)
        odd_pair = self._find_parallel_term(odd_terms, data_base_id=x_base_id, pair=True)
        if any(item is None for item in (even_direct, even_pair, odd_direct, odd_pair)):
            return None

        if id(even_direct["data"].base) != x_base_id or id(even_pair["data"].base) != x_base_id:
            return None
        if id(odd_direct["data"].base) != x_base_id or id(odd_pair["data"].base) != x_base_id:
            return None
        if _parallel_access_identity(even_direct["coeff"]) != _parallel_access_identity(odd_direct["coeff"]):
            return None

        return {
            "x_direct": even_direct["data"],
            "x_pair": even_pair["data"],
            "cos": even_direct["coeff"],
            "neg_sin": even_pair["coeff"],
            "sin": odd_pair["coeff"],
        }

    def _analyze_parallel_mul_term(
        self,
        expr: ParallelExpr,
    ) -> Optional[Dict[str, ParallelAccess]]:
        if expr.kind != "op" or expr.op != "mul" or len(expr.args) != 2:
            return None
        lhs, rhs = expr.args
        if lhs.kind != "load" or rhs.kind != "load":
            return None
        lhs_access = lhs.value
        rhs_access = rhs.value
        if not isinstance(lhs_access, ParallelAccess) or not isinstance(rhs_access, ParallelAccess):
            return None
        return {"lhs": lhs_access, "rhs": rhs_access}

    def _collect_parallel_mul_terms(
        self,
        expr: ParallelExpr,
    ) -> Optional[List[Dict[str, ParallelAccess]]]:
        if expr.kind != "op" or expr.op != "add" or len(expr.args) != 2:
            return None
        terms = [self._analyze_parallel_mul_term(term) for term in expr.args]
        if any(term is None for term in terms):
            return None
        return [term for term in terms if term is not None]

    def _resolve_parallel_pairwise_data_base_id(
        self,
        terms: List[Dict[str, ParallelAccess]],
    ) -> Optional[int]:
        direct_bases = set()
        pair_bases = set()
        for term in terms:
            for access in (term["lhs"], term["rhs"]):
                if self._parallel_access_is_pair(access):
                    pair_bases.add(id(access.base))
                else:
                    direct_bases.add(id(access.base))
        candidate_bases = direct_bases & pair_bases
        if len(candidate_bases) != 1:
            return None
        return next(iter(candidate_bases))

    def _find_parallel_term(
        self,
        terms: List[Optional[Dict[str, ParallelAccess]]],
        *,
        data_base_id: int,
        pair: bool,
    ) -> Optional[Dict[str, ParallelAccess]]:
        for term in terms:
            if term is None:
                continue
            lhs_access = term["lhs"]
            rhs_access = term["rhs"]
            lhs_is_data = id(lhs_access.base) == data_base_id and self._parallel_access_is_pair(lhs_access) == pair
            rhs_is_data = id(rhs_access.base) == data_base_id and self._parallel_access_is_pair(rhs_access) == pair
            if lhs_is_data == rhs_is_data:
                continue
            if lhs_is_data:
                return {"data": lhs_access, "coeff": rhs_access}
            return {"data": rhs_access, "coeff": lhs_access}
        return None

    def _parallel_access_is_pair(self, access: ParallelAccess) -> bool:
        selectors = tuple(access.selectors)
        return bool(
            selectors
            and isinstance(selectors[-1], ParallelExpr)
            and selectors[-1].kind == "pair_index"
        )

    def _parallel_expr_matches_even_mod2(self, expr: ParallelExpr) -> bool:
        if expr.kind != "op" or expr.op != "eq" or len(expr.args) != 2:
            return False
        lhs, rhs = expr.args
        if rhs.kind == "literal" and int(rhs.value) == 0:
            return self._parallel_expr_is_mod2(lhs)
        if lhs.kind == "literal" and int(lhs.value) == 0:
            return self._parallel_expr_is_mod2(rhs)
        return False

    def _parallel_expr_is_mod2(self, expr: ParallelExpr) -> bool:
        return (
            expr.kind == "op"
            and expr.op == "mod"
            and len(expr.args) == 2
            and expr.args[1].kind == "literal"
            and int(expr.args[1].value) == 2
        )

    def _emit_parallel_expr_to_addr(
        self,
        *,
        expr: ParallelExpr,
        dst_addr: int,
        lane_offset: int,
        group: ParallelCycleGroup,
        access_slot_map: Dict[str, int],
        input_slot_bases: Dict[int, int],
        task_id: str,
    ) -> None:
        gp_regs = self.program.compiler.register_allocator.allocate_gp(1)
        gp_dst = gp_regs[0]
        fp_reg = self._emit_parallel_expr_to_fp_reg(
            expr=expr,
            lane_offset=lane_offset,
            group=group,
            access_slot_map=access_slot_map,
            input_slot_bases=input_slot_bases,
            task_id=task_id,
            preferred_fp_reg=1,
            scratch_fp_regs=(2, 3, 4, 5, 6, 7),
        )
        try:
            lines = [
                f"; parallel expr store {task_id}",
                f"S_ADDI_INT gp{gp_dst}, gp0, {int(dst_addr)}",
                f"S_ST_FP f{fp_reg}, gp{gp_dst}, 0",
            ]
            self.program.compiler.generated_code += "\n".join(lines) + "\n"
        finally:
            self.program.compiler.register_allocator.free_gp(gp_regs)

    def _emit_parallel_expr_to_fp_reg(
        self,
        *,
        expr: ParallelExpr,
        lane_offset: int,
        group: ParallelCycleGroup,
        access_slot_map: Dict[str, int],
        input_slot_bases: Dict[int, int],
        task_id: str,
        preferred_fp_reg: int,
        scratch_fp_regs: Tuple[int, ...],
    ) -> int:
        if expr.kind == "select":
            predicate = expr.args[0]
            branch_expr = expr.args[1] if self._parallel_predicate_value(predicate, lane_offset, group) else expr.args[2]
            return self._emit_parallel_expr_to_fp_reg(
                expr=branch_expr,
                lane_offset=lane_offset,
                group=group,
                access_slot_map=access_slot_map,
                input_slot_bases=input_slot_bases,
                task_id=task_id,
                preferred_fp_reg=preferred_fp_reg,
                scratch_fp_regs=scratch_fp_regs,
            )
        if expr.kind == "literal":
            literal_var = self.program.mapf(float(expr.value))[0]
            return self._emit_parallel_load_addr_to_fp_reg(
                int(_require_fp_addr(literal_var)),
                task_id=f"{task_id}.literal",
                fp_dst=preferred_fp_reg,
            )
        if expr.kind == "fpvar":
            fp_var = expr.value
            if not isinstance(fp_var, FPVar):
                raise RuntimeError(f"parallel fpvar expr missing FPVar payload: {expr}")
            return self._emit_parallel_load_addr_to_fp_reg(
                int(_require_fp_addr(fp_var)),
                task_id=f"{task_id}.fpvar",
                fp_dst=preferred_fp_reg,
            )
        if expr.kind == "load":
            access = expr.value
            if not isinstance(access, ParallelAccess):
                raise RuntimeError(f"parallel load expr missing ParallelAccess: {expr}")
            if isinstance(access.base, Vector):
                return self._emit_parallel_vector_access_to_fp_reg(
                    access=access,
                    lane_offset=lane_offset,
                    group=group,
                    task_id=f"{task_id}.vector_load",
                    fp_dst=preferred_fp_reg,
                )
            slot_id = access_slot_map[_parallel_access_identity(access)]
            lane_index = self._parallel_access_lane_index(access, lane_offset)
            return self._emit_parallel_load_addr_to_fp_reg(
                int(input_slot_bases[slot_id] + lane_index),
                task_id=f"{task_id}.load",
                fp_dst=preferred_fp_reg,
            )
        if expr.kind == "op":
            if len(expr.args) != 2 or expr.op not in {"add", "sub", "mul", "max"}:
                raise NotImplementedError(f"parallel expr op lowering supports add/sub/mul/max only, got {expr.op!r}")
            if not scratch_fp_regs:
                raise RuntimeError(f"parallel expr {task_id} ran out of hard-coded FP scratch registers")
            rhs_preferred = scratch_fp_regs[0]
            rhs_scratch = tuple(reg for reg in scratch_fp_regs[1:] if reg != preferred_fp_reg)
            lhs_reg = self._emit_parallel_expr_to_fp_reg(
                expr=expr.args[0],
                lane_offset=lane_offset,
                group=group,
                access_slot_map=access_slot_map,
                input_slot_bases=input_slot_bases,
                task_id=f"{task_id}.lhs",
                preferred_fp_reg=preferred_fp_reg,
                scratch_fp_regs=scratch_fp_regs,
            )
            rhs_reg = self._emit_parallel_expr_to_fp_reg(
                expr=expr.args[1],
                lane_offset=lane_offset,
                group=group,
                access_slot_map=access_slot_map,
                input_slot_bases=input_slot_bases,
                task_id=f"{task_id}.rhs",
                preferred_fp_reg=rhs_preferred,
                scratch_fp_regs=rhs_scratch,
            )
            op_to_insn = {"add": "S_ADD_FP", "sub": "S_SUB_FP", "mul": "S_MUL_FP", "max": "S_MAX_FP"}
            self.program.compiler.generated_code += (
                f"; parallel expr {task_id}.{expr.op}\n"
                f"{op_to_insn[str(expr.op)]} f{lhs_reg}, f{lhs_reg}, f{rhs_reg}\n"
            )
            return lhs_reg
        raise NotImplementedError(f"Unsupported parallel expr kind for lowering: {expr.kind}")

    def _emit_parallel_load_addr_to_fp_reg(
        self,
        addr: int,
        *,
        task_id: str,
        fp_dst: int,
    ) -> int:
        gp_regs = self.program.compiler.register_allocator.allocate_gp(1)
        gp_src = gp_regs[0]
        lines = [
            f"; parallel load scalar {task_id}",
            f"S_ADDI_INT gp{gp_src}, gp0, {int(addr)}",
            f"S_LD_FP f{fp_dst}, gp{gp_src}, 0",
        ]
        self.program.compiler.generated_code += "\n".join(lines) + "\n"
        self.program.compiler.register_allocator.free_gp(gp_regs)
        return fp_dst

    def _emit_parallel_vector_access_to_fp_reg(
        self,
        *,
        access: ParallelAccess,
        lane_offset: int,
        group: ParallelCycleGroup,
        task_id: str,
        fp_dst: int,
    ) -> int:
        fp_addr = self._parallel_vector_access_fp_addr(access, lane_offset=lane_offset, group=group)
        return self._emit_parallel_load_addr_to_fp_reg(fp_addr, task_id=task_id, fp_dst=fp_dst)

    def _parallel_vector_access_fp_addr(
        self,
        access: ParallelAccess,
        *,
        lane_offset: int,
        group: ParallelCycleGroup,
    ) -> int:
        if not isinstance(access.base, Vector):
            raise RuntimeError(f"parallel vector fp addr expected Vector base, got {type(access.base).__name__}")
        logical_index = self._parallel_access_lane_logical_index(access, lane_offset=lane_offset, group=group)
        fp_var = self.program.tensor_manager._resolve_element_fpvar(ElementRef(base=access.base, indices=logical_index))
        return int(_require_fp_addr(fp_var))

    def _parallel_predicate_value(
        self,
        expr: ParallelExpr,
        lane_offset: int,
        group: ParallelCycleGroup,
    ) -> bool:
        if expr.kind == "op" and len(expr.args) == 2:
            lhs = self._parallel_index_expr_value(expr.args[0], lane_offset=lane_offset, group=group)
            rhs = self._parallel_index_expr_value(expr.args[1], lane_offset=lane_offset, group=group)
            if expr.op == "lt":
                return lhs < rhs
            if expr.op == "le":
                return lhs <= rhs
            if expr.op == "gt":
                return lhs > rhs
            if expr.op == "ge":
                return lhs >= rhs
            if expr.op == "eq":
                return lhs == rhs
        raise NotImplementedError(f"Unsupported parallel predicate lowering: {expr.kind}")

    def _parallel_index_expr_value(
        self,
        expr: ParallelExpr,
        *,
        lane_offset: int,
        group: ParallelCycleGroup,
    ) -> int:
        if expr.kind == "literal":
            return int(expr.value)
        if expr.kind == "axis":
            axis = expr.value
            if axis is None:
                raise RuntimeError("parallel axis expression missing axis metadata")
            axis_id = int(axis.axis)
            if axis_id == 0:
                return int(group.i_index)
            if axis_id == 1:
                return int(group.j_index)
            if axis_id == 2:
                if int(group.k_count) > 1 and int(group.elem_width) < int(self.program.mlen):
                    return int(group.k_base) + (int(lane_offset) % int(group.elem_width))
                return int(group.k_base) + int(lane_offset)
            raise RuntimeError(f"Unsupported parallel axis id: {axis_id}")
        if expr.kind == "op" and len(expr.args) == 2:
            lhs = self._parallel_index_expr_value(expr.args[0], lane_offset=lane_offset, group=group)
            rhs = self._parallel_index_expr_value(expr.args[1], lane_offset=lane_offset, group=group)
            if expr.op == "add":
                return lhs + rhs
            if expr.op == "sub":
                return lhs - rhs
            if expr.op == "mul":
                return lhs * rhs
            if expr.op == "mod":
                return lhs % rhs
        raise NotImplementedError(f"Unsupported parallel index expression lowering: {expr.kind}:{getattr(expr, 'op', None)!r}")

    def _parallel_access_lane_index(self, access: ParallelAccess, lane_offset: int) -> int:
        selectors = tuple(access.selectors)
        if not selectors:
            return int(lane_offset)
        last = selectors[-1]
        if isinstance(last, ParallelExpr) and last.kind == "pair_index":
            return int(lane_offset) ^ 1
        return int(lane_offset)

    def _parallel_access_packs_axis1_lanes(
        self,
        access: ParallelAccess,
        group: ParallelCycleGroup,
    ) -> bool:
        if int(group.k_count) <= 1 or int(group.elem_width) >= int(self.program.mlen):
            return False
        selectors = tuple(access.selectors)
        if not selectors:
            return False
        lane_selector = selectors[-1]
        if isinstance(lane_selector, ParallelAxis):
            lane_axis_ok = int(lane_selector.axis) == 2
        elif isinstance(lane_selector, ParallelExpr):
            lane_axis_ok = lane_selector.kind in {"pair_index", "half_index"}
        else:
            lane_axis_ok = False
        has_axis1 = any(
            isinstance(selector, ParallelAxis) and int(selector.axis) == 1
            for selector in selectors[:-1]
        )
        return bool(lane_axis_ok and has_axis1)

    def _parallel_access_lane_logical_index(
        self,
        access: ParallelAccess,
        *,
        lane_offset: int,
        group: ParallelCycleGroup,
    ) -> Tuple[int, ...]:
        logical_index: List[int] = []
        lane_axis_index = len(access.logical_shape) - 1
        multi_lane = int(group.k_count) > 1
        elem_width = int(group.elem_width)
        packed_axis1 = self._parallel_access_packs_axis1_lanes(access, group)
        for axis_pos, selector in enumerate(access.selectors):
            if isinstance(selector, ParallelAxis):
                if int(selector.axis) == 0:
                    logical_index.append(int(group.i_index))
                elif int(selector.axis) == 1:
                    if multi_lane and packed_axis1:
                        logical_index.append(int(group.j_index) + int(lane_offset) // elem_width)
                    else:
                        logical_index.append(int(group.j_index))
                elif int(selector.axis) == 2:
                    if multi_lane:
                        if packed_axis1 or axis_pos == lane_axis_index:
                            # Innermost: element position within lane
                            logical_index.append(int(self._parallel_access_lane_index(access, lane_offset)) % elem_width)
                        else:
                            # Non-innermost: head/group index
                            logical_index.append(int(group.k_base) + int(lane_offset) // elem_width)
                    else:
                        logical_index.append(int(group.k_base) + (self._parallel_access_lane_index(access, lane_offset) if axis_pos == lane_axis_index else int(lane_offset)))
                else:
                    raise RuntimeError(f"Unsupported parallel axis id: {selector.axis}")
            elif isinstance(selector, ParallelExpr):
                if axis_pos != lane_axis_index:
                    raise NotImplementedError(
                        f"parallel lane logical index only supports selector expr on innermost axis, got axis_pos={axis_pos}"
                    )
                if multi_lane:
                    # Innermost expr (pair_index etc.): element position within lane
                    logical_index.append(int(self._parallel_access_lane_index(access, lane_offset)) % elem_width)
                else:
                    logical_index.append(int(group.k_base) + int(self._parallel_access_lane_index(access, lane_offset)))
            elif isinstance(selector, int):
                logical_index.append(int(selector))
            else:
                raise NotImplementedError(
                    f"parallel lane logical index does not support selector {selector!r} of type {type(selector).__name__}"
                )
        return tuple(logical_index)

    def _parallel_access_cycle_src_vram_row_addr(
        self,
        access: ParallelAccess,
        group: ParallelCycleGroup,
    ) -> int:
        tile = self._parallel_access_cycle_src_tile(access, group)
        row = self._parallel_access_cycle_row(access, group)
        local_row = row % self.program.mlen
        value = self.program.value_manager.resolve_value_tile(tile)
        self.program.value_manager.ensure_value_tile_in_place(value, "vram")
        vram_addr = value.residency.get("vram_addr")
        if vram_addr is None:
            raise RuntimeError(f"parallel lowering expected VRAM residency for {value.value_tile_id}")
        return int(vram_addr) + local_row * self.program.mlen

    def _parallel_access_cycle_row(
        self,
        access: ParallelAccess,
        group: ParallelCycleGroup,
    ) -> int:
        concrete_selectors = self._parallel_access_concrete_selectors(access, group)
        row_range, col_range = _logical_selectors_to_physical_ranges(access.base.logical_shape, concrete_selectors)
        expected_width = int(group.element_count)
        if (row_range[1] - row_range[0]) != 1 or (col_range[1] - col_range[0]) != expected_width:
            raise NotImplementedError(
                "parallel lowering currently supports one full-width contiguous row per cycle; "
                f"got row_range={row_range}, col_range={col_range}, expected_width={expected_width}, mlen={self.program.mlen}"
            )
        return int(row_range[0])

    def _parallel_access_cycle_src_tile(
        self,
        access: ParallelAccess,
        group: ParallelCycleGroup,
    ) -> TileLike:
        tile = self._parallel_access_cycle_tile(access, group)
        if not isinstance(tile, (TensorTile, InputTile)):
            raise RuntimeError("parallel source access did not resolve to TensorTile/InputTile")
        return tile

    def _parallel_access_cycle_dst_tile(
        self,
        access: ParallelAccess,
        group: ParallelCycleGroup,
    ) -> TensorTile:
        tile = self._parallel_access_cycle_tile(access, group)
        if not isinstance(tile, TensorTile):
            raise RuntimeError(
                f"parallel destination access must resolve to TensorTile, got {type(tile).__name__}"
            )
        return tile

    def _parallel_access_cycle_tile(
        self,
        access: ParallelAccess,
        group: ParallelCycleGroup,
    ) -> TileLike:
        concrete_selectors = self._parallel_access_concrete_selectors(access, group)
        row_range, col_range = _logical_selectors_to_physical_ranges(access.base.logical_shape, concrete_selectors)
        expected_width = int(group.element_count)
        if (row_range[1] - row_range[0]) != 1 or (col_range[1] - col_range[0]) != expected_width:
            raise NotImplementedError(
                "parallel lowering currently supports one full-width contiguous row per cycle; "
                f"got row_range={row_range}, col_range={col_range}, expected_width={expected_width}, mlen={self.program.mlen}"
            )
        row = int(row_range[0])
        col = int(col_range[0])
        tile_coord = (row // self.program.mlen, col // self.program.mlen)
        tiles = getattr(access.base, "tiles", None)
        if not isinstance(tiles, dict):
            raise RuntimeError(f"parallel access base {type(access.base).__name__} does not expose tiles")
        tile = tiles.get(tile_coord)
        if not isinstance(tile, (TensorTile, InputTile)):
            raise RuntimeError(f"parallel access did not resolve to TensorTile/InputTile at coord={tile_coord}")
        return tile

    def _parallel_access_concrete_selectors(
        self,
        access: ParallelAccess,
        group: ParallelCycleGroup,
    ) -> Tuple[SliceItem, ...]:
        concrete: List[SliceItem] = []
        multi_lane = int(group.k_count) > 1
        packed_axis1 = self._parallel_access_packs_axis1_lanes(access, group)
        if multi_lane and packed_axis1:
            axis_to_value = {
                0: int(group.i_index),
                1: slice(int(group.j_index), int(group.j_index) + int(group.k_count)),
                2: slice(0, int(group.elem_width)),
            }
            expr_range = slice(0, int(group.elem_width))
        elif multi_lane:
            # Multi-lane: axis 2 selects k_count head/group indices;
            # pair_index/half_index cover the per-head elem_width range.
            k_axis_range = slice(int(group.k_base), int(group.k_base) + int(group.k_count))
            expr_range = slice(0, int(group.elem_width))
            axis_to_value = {
                0: int(group.i_index),
                1: int(group.j_index),
                2: k_axis_range,
            }
        else:
            k_axis_range = slice(int(group.k_base), int(group.k_base) + int(self.program.mlen))
            expr_range = slice(int(group.k_base), int(group.k_base) + int(self.program.mlen))
            axis_to_value = {
                0: int(group.i_index),
                1: int(group.j_index),
                2: k_axis_range,
            }
        for selector in access.selectors:
            if isinstance(selector, ParallelAxis):
                concrete.append(axis_to_value[int(selector.axis)])
            elif isinstance(selector, ParallelExpr):
                if selector.kind == "pair_index":
                    concrete.append(expr_range)
                elif selector.kind == "half_index":
                    concrete.append(expr_range)
                else:
                    raise NotImplementedError(f"Unsupported selector expr for concrete access lowering: {selector.kind}")
            else:
                concrete.append(selector)
        return tuple(concrete)


class _LoopHintRange:
    def __init__(self, program: "TileTensorProgram", *, kind: str, extent: int, region_id: Optional[int] = None) -> None:
        self.program = program
        self.kind = kind
        self.extent = int(extent)
        self.region_id = region_id

    def __iter__(self):
        for index in range(self.extent):
            if self.kind == "parallel" and self.region_id is not None:
                self.program._active_parallel_region_ids.append(self.region_id)
            try:
                yield index
            finally:
                if self.kind == "parallel" and self.region_id is not None:
                    self.program._active_parallel_region_ids.pop()


class ValueManager:
    """Resolve logical tiles into backing values/views and manage residency.

    The value layer is responsible for:

    - direct `tile -> ValueTile` bindings
    - `ValueTileView` resolution over shared backing values
    - write preparation for mutating tensor destinations
    - HBM/VRAM/MRAM residency transitions
    - rebinding and release when compute produces updated values

    This class is the main implementation of the runtime's value layer. The
    preferred write-preparation entrypoint is `prepare_updated_view_value(...)`.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.value_tiles: Dict[str, ValueTile] = {}
        self.full_tile_bindings: Dict[str, str] = {}
        self.fp_fragment_bindings: Dict[str, str] = {}
        self.value_tile_tensor_refs: Dict[str, set[str]] = {}
        self.narrow_group_bindings: Dict[Tuple[object, ...], str] = {}
        self._value_tiles_in_vram: Dict[str, int] = {}
        self._value_tiles_in_mram: Dict[str, int] = {}
        self._value_tiles_in_hbm: Dict[str, object] = {}
        self._mram_fifo: List[str] = []
        self._protected_vram_value_tile_ids: set[str] = set()
        self._value_tile_counter = 0

    @property
    def bindings(self) -> Dict[str, str]:
        # Compatibility alias for older scaffold/debug helpers.
        return self.full_tile_bindings

    def _next_value_tile_id(self) -> str:
        value_tile_id = f"value_tile.{self._value_tile_counter}"
        self._value_tile_counter += 1
        return value_tile_id

    def mapv(self, signal: List[object]) -> MapvPacket:
        """Resolve one mapped logical packet into concrete value-layer operands.

        Input packets come from TensorManager's `mapt` stage plus residency
        targets and, optionally, one control tag. The function performs late
        source resolution so compute sees the correct runtime object type:

        - wide/full tiles -> ValueTile
        - narrow/grouped tiles -> shared backing ValueTile

        Destination resolution is also late here so updates can detach old
        bindings and materialize one fresh writable value only when compute is
        ready to run.
        """
        control = None
        if signal and isinstance(signal[-1], str):
            control = signal[-1]
            residency_targets = signal[-2]
            signal_items = signal[:-2]
        else:
            residency_targets = signal[-1]
            signal_items = signal[:-1]

        if control == "copy_tile_pair":
            if len(signal_items) != 2 or not all(_is_tile_object(item) for item in signal_items):
                raise RuntimeError("copy_tile_pair mapv expects [src_tile, dst_tile, residency_targets, control]")
            src_tile, dst_tile = signal_items
            src_value = self._resolve_mapv_source_value(src_tile, residency_targets[0])
            if not isinstance(src_value, ValueTile):
                raise RuntimeError("copy mapv expects one full source ValueTile")
            return ("copy", src_value, dst_tile)

        pair_groups, dst_tile = self._split_mapv_signal(signal_items)
        mapped_pairs: List[List[object]] = []
        for pair in pair_groups:
            if len(pair) != 2:
                continue
            src1_tile, src2_tile = pair
            v1 = self._resolve_mapv_source_value(src1_tile, residency_targets[0])
            v2 = self._resolve_mapv_source_value(src2_tile, residency_targets[1])
            mapped_pairs.append([v1, v2])

        if dst_tile is None:
            raise RuntimeError("mapv expects one destination tensor tile")
        if isinstance(dst_tile, TensorTile):
            dst_view = self.resolve_value_tile_view(dst_tile)
            prepared_write = self.prepare_updated_view_value(
                dst_tile,
                dst_view,
                ensure_old_place=None,
                new_place=residency_targets[2],
            )
            v3 = prepared_write.new_value
        else:
            v3 = self._prepare_mapv_destination_value(dst_tile, residency_targets[2])
        return ("matmul", mapped_pairs, v3, dst_tile)

    def _resolve_mapv_source_value(self, tile: TensorTile | InputTile | VectorTile, place: str) -> SourceValueLike:
        if isinstance(tile, VectorTile):
            raise RuntimeError(
                f"VectorTile {tile.tile_id} maps to FPFragment rather than ValueTile; "
                "use mapf or ElementRef-based FP kernels"
            )
        value = self._resolve_tile_backing_value(tile)
        return value

    def _resolve_alias_owner_tile(self, tile: TileLike) -> TileLike:
        if not bool(tile.metadata.get("slice_materialized", False)):
            return tile
        source_tile_id = tile.metadata.get("source_tile_id")
        if not isinstance(source_tile_id, str):
            return tile
        owner_tile = self.program.tensor_manager.tensor_tiles.get(source_tile_id)
        if owner_tile is None:
            owner_tile = self.program.tensor_manager.input_tiles.get(source_tile_id)
        if not isinstance(owner_tile, (TensorTile, InputTile, VectorTile)):
            return tile
        return owner_tile

    def _prepare_mapv_destination_value(self, tile: TensorTile | InputTile | VectorTile, place: str) -> ValueTile:
        if isinstance(tile, VectorTile):
            raise RuntimeError(
                f"VectorTile {tile.tile_id} does not prepare one destination ValueTile; "
                "bind it to FPFragment through ValueManager"
            )
        canonical_tile = self._resolve_alias_owner_tile(tile)
        if canonical_tile is not tile and not self._is_narrow_tensor_tile(tile):
            tile = canonical_tile
        if isinstance(tile, TensorTile) and not self._is_narrow_tensor_tile(tile):
            old_value = self.resolve_value_tile(tile)
            old_value_tile_id = self._detach_tile_value_pointer(tile.tile_id)
            if old_value_tile_id is None:
                raise RuntimeError(f"Wide destination tile {tile.tile_id} had no bound value to detach")
            new_value = self.prepare_vram_backing_value(old_value)
            self._attach_tile_value_pointer(tile.tile_id, new_value.value_tile_id)
            self.free_value_tile(old_value_tile_id)
            return new_value
        dst_source_value = self.resolve_value_tile(tile)
        value = self.prepare_vram_backing_value(dst_source_value)
        return value

    def _is_packed_narrow_tile(self, tile: TileLike) -> bool:
        return int(tile.metadata.get("packed_head_count", 1)) > 1 or bool(tile.metadata.get("packed_head_group", False))

    def _is_grouped_narrow_backing_tile(self, tile: TileLike) -> bool:
        return self._is_packed_narrow_tile(tile)

    def _is_narrow_tensor_tile(self, tile: TileLike) -> bool:
        width_class = tile.metadata.get("tile_width_class")
        if width_class == "narrow":
            return True
        if width_class == "full":
            return False
        return int(tile.tile_shape[1]) < int(self.program.mlen)

    def _view_group_key_for_tile(self, tile: TileLike) -> Tuple[object, ...]:
        owner_name = _tile_owner_name(tile)
        if bool(tile.metadata.get("packed_head_group", False)):
            head_index = int(tile.metadata.get("group_head_start", tile.metadata.get("head_index", 0)))
        else:
            head_index = int(tile.metadata.get("head_index", 0))
        row_block = int(tile.metadata.get("row_block", tile.coord[0]))
        return (owner_name, head_index, row_block)

    def _view_slot_key_for_tile(self, tile: TileLike) -> Tuple[object, ...]:
        owner_name = _tile_owner_name(tile)
        head_index = int(tile.metadata.get("slot_head_index", tile.metadata.get("head_index", 0)))
        row_block = int(tile.metadata.get("row_block", tile.coord[0]))
        col_offset = int(tile.metadata.get("scatter_col_offset", tile.coord[1] * self.program.mlen))
        col_count = int(tile.metadata.get("scatter_col_count", tile.tile_shape[1]))
        return (owner_name, head_index, row_block, col_offset, col_count)

    def _tiles_sharing_backing(self, tile: TensorTile | InputTile) -> List[TensorTile | InputTile]:
        if not self._is_narrow_tensor_tile(tile):
            return [tile]
        if self._is_packed_narrow_tile(tile):
            return [tile]
        return self._iter_group_tiles(tile)

    def _bind_tiles_to_value(self, tiles: Sequence[TensorTile | InputTile], value_tile_id: str) -> List[str]:
        detached_ids: List[str] = []
        for tile in tiles:
            old_value_tile_id = self._detach_tile_value_pointer(tile.tile_id)
            if old_value_tile_id is not None and old_value_tile_id != value_tile_id:
                detached_ids.append(old_value_tile_id)
            self._attach_tile_value_pointer(tile.tile_id, value_tile_id)
        return detached_ids

    def _rebind_view_group_value(self, tile: TensorTile | InputTile, new_value: ValueTile) -> None:
        group_tiles = self._tiles_sharing_backing(tile)
        if self._is_narrow_tensor_tile(tile):
            self.narrow_group_bindings[self._view_group_key_for_tile(tile)] = new_value.value_tile_id
        detached_ids = self._bind_tiles_to_value(group_tiles, new_value.value_tile_id)
        for old_value_tile_id in sorted(set(detached_ids)):
            self.free_value_tile(old_value_tile_id)

    def _iter_value_tile_views(self, value_tile_id: str) -> List[ValueTileView]:
        tile_ids = sorted(self.value_tile_tensor_refs.get(value_tile_id, set()))
        views: List[ValueTileView] = []
        for tile_id in tile_ids:
            tile = self.program.tensor_manager.tensor_tiles.get(tile_id)
            if tile is None:
                tile = self.program.tensor_manager.input_tiles.get(tile_id)
            if not isinstance(tile, (TensorTile, InputTile)):
                continue
            for view in self._tile_compute_views(tile):
                if view.backing_value_tile_id == value_tile_id:
                    views.append(view)
        return views

    def _views_overlap(self, lhs: ValueTileView, rhs: ValueTileView) -> bool:
        lhs_row_end = int(lhs.row_offset) + int(lhs.row_count)
        rhs_row_end = int(rhs.row_offset) + int(rhs.row_count)
        lhs_col_end = int(lhs.col_offset) + int(lhs.col_count)
        rhs_col_end = int(rhs.col_offset) + int(rhs.col_count)
        return not (
            lhs_row_end <= int(rhs.row_offset)
            or rhs_row_end <= int(lhs.row_offset)
            or lhs_col_end <= int(rhs.col_offset)
            or rhs_col_end <= int(lhs.col_offset)
        )

    def _same_view_identity(self, lhs: ValueTileView, rhs: ValueTileView) -> bool:
        return (
            lhs.backing_value_tile_id == rhs.backing_value_tile_id
            and lhs.owner_tile_id == rhs.owner_tile_id
            and int(lhs.row_offset) == int(rhs.row_offset)
            and int(lhs.row_count) == int(rhs.row_count)
            and int(lhs.col_offset) == int(rhs.col_offset)
            and int(lhs.col_count) == int(rhs.col_count)
        )

    def view_has_conflicting_refs(self, view: ValueTileView) -> bool:
        for other_view in self._iter_value_tile_views(view.backing_value_tile_id):
            if self._same_view_identity(view, other_view):
                continue
            if self._views_overlap(view, other_view):
                return True
        return False

    def prepare_updated_view_value(
        self,
        tile: TensorTile | InputTile,
        view: ValueTileView,
        *,
        ensure_old_place: Optional[str] = None,
        new_place: str = "vram",
    ) -> PreparedWrite:
        """Prepare one mutating tensor-view write.

        This is the main write-path helper for tensor destinations.

        Returned `PreparedWrite` tells the caller:
        - whether the write is in-place (`reuse_old`)
        - which backing value should receive the write (`new_value`)
        - which view on the new backing should be targeted (`target_view`)
        - whether a partial-update preserve copy is still required
          (`requires_preserve_copy`)
        """
        old_value = self.value_tiles.get(view.backing_value_tile_id)
        if not isinstance(old_value, ValueTile):
            raise RuntimeError(f"View {view.view_id} is missing backing value {view.backing_value_tile_id}")
        if ensure_old_place is not None:
            self.ensure_value_tile_in_place(old_value, ensure_old_place)
        if not self.view_has_conflicting_refs(view):
            self.ensure_value_tile_in_place(old_value, new_place)
            if new_place == "vram":
                self._drop_stale_non_vram_residency(old_value)
            return PreparedWrite(
                old_value=old_value,
                new_value=old_value,
                target_view=view,
                reuse_old=True,
                requires_preserve_copy=False,
            )
        requires_preserve_copy = False
        self.protect_value_tile(old_value, "vram")
        try:
            if self._view_covers_logical_tile(tile, view):
                new_value = self.prepare_vram_backing_value(old_value, preserve_existing=True)
            else:
                new_value = self._prepare_partial_update_vram_successor(old_value)
                if new_value is None:
                    new_value = self.prepare_vram_backing_value(old_value, preserve_existing=True)
                    requires_preserve_copy = True
            self._rebind_view_group_value(tile, new_value)
        finally:
            self.stop_protect_value_tile(old_value, "vram")
        self.ensure_value_tile_in_place(new_value, new_place)
        return PreparedWrite(
            old_value=old_value,
            new_value=new_value,
            target_view=self.rebind_view(view, new_value),
            reuse_old=False,
            requires_preserve_copy=requires_preserve_copy,
        )

    def resolve_value_tile_view(self, tile: TensorTile | InputTile) -> ValueTileView:
        backing_value = self.resolve_value_tile(tile)
        if self._is_packed_narrow_tile(tile):
            return ValueTileView(
                backing_value_tile_id=backing_value.value_tile_id,
                owner_tile_id=tile.tile_id,
                row_offset=0,
                row_count=int(tile.tile_shape[0]),
                col_offset=0,
                col_count=int(tile.tile_shape[1]),
                metadata={"slot_key": self._view_group_key_for_tile(tile), "kind": "packed_tile"},
            )
        if self._is_narrow_tensor_tile(tile):
            slot_key = self._view_slot_key_for_tile(tile)
            return ValueTileView(
                backing_value_tile_id=backing_value.value_tile_id,
                owner_tile_id=tile.tile_id,
                row_offset=0,
                row_count=int(tile.tile_shape[0]),
                col_offset=int(slot_key[3]),
                col_count=int(slot_key[4]),
                metadata={"slot_key": slot_key, "kind": "narrow_tile"},
            )
        return ValueTileView(
            backing_value_tile_id=backing_value.value_tile_id,
            owner_tile_id=tile.tile_id,
            row_offset=0,
            row_count=int(tile.tile_shape[0]),
            col_offset=0,
            col_count=int(tile.tile_shape[1]),
            metadata={"kind": "full_tile"},
        )

    def _tile_compute_views(self, tile: TensorTile | InputTile) -> List[ValueTileView]:
        if not self._is_packed_narrow_tile(tile):
            return [self.resolve_value_tile_view(tile)]
        backing_value = self.resolve_value_tile(tile)
        packed_heads = int(tile.metadata.get("packed_head_count", 1))
        slot_width = int(tile.metadata.get("scatter_slot_width", tile.tile_shape[1]))
        views: List[ValueTileView] = []
        for lane_index in range(packed_heads):
            views.append(
                ValueTileView(
                    backing_value_tile_id=backing_value.value_tile_id,
                    owner_tile_id=tile.tile_id,
                    row_offset=0,
                    row_count=int(tile.tile_shape[0]),
                    col_offset=lane_index * slot_width,
                    col_count=slot_width,
                    metadata={"lane_index": lane_index, "kind": "packed_lane"},
                )
            )
        return views

    def resolve_row_operand(self, tile: TensorTile | InputTile, place: str = "vram") -> RowOperandLike:
        if self._is_narrow_tensor_tile(tile):
            view = self.resolve_value_tile_view(tile)
            return view
        value = self.resolve_value_tile(tile)
        return value

    def resolve_row_operand_for_ranges(
        self,
        tile: TensorTile | InputTile,
        row_range: Tuple[int, int],
        col_range: Tuple[int, int],
        place: str = "vram",
    ) -> RowOperandLike:
        if not self._is_narrow_tensor_tile(tile):
            return self.resolve_row_operand(tile, place)

        row_block, col_block = tile.coord
        row_start = row_block * self.program.mlen
        row_end = row_start + int(tile.tile_shape[0])
        col_start = col_block * self.program.mlen
        col_end = col_start + int(tile.tile_shape[1])
        if not _ranges_overlap((row_start, row_end), row_range) or not _ranges_overlap((col_start, col_end), col_range):
            raise RuntimeError(
                f"Requested row operand slice row_range={row_range} col_range={col_range} does not overlap tile {tile.tile_id}"
            )

        overlap_col_start = max(col_start, col_range[0])
        overlap_col_end = min(col_end, col_range[1])
        overlap_col_offset = int(overlap_col_start - col_start)
        overlap_col_count = int(overlap_col_end - overlap_col_start)
        if overlap_col_count <= 0:
            raise RuntimeError(f"Resolved empty column overlap for tile {tile.tile_id}")

        if overlap_col_offset == 0 and overlap_col_count == int(tile.tile_shape[1]):
            return self.resolve_row_operand(tile, place)

        slot_width = int(tile.metadata.get("scatter_slot_width", overlap_col_count))
        if overlap_col_offset % slot_width != 0 or overlap_col_count % slot_width != 0:
            raise RuntimeError(
                f"Slice overlap for tile {tile.tile_id} is not aligned to slot width {slot_width}: "
                f"offset={overlap_col_offset} count={overlap_col_count}"
            )
        backing_value = self.resolve_value_tile(tile)
        return ValueTileView(
            backing_value_tile_id=backing_value.value_tile_id,
            owner_tile_id=tile.tile_id,
            row_offset=0,
            row_count=int(tile.tile_shape[0]),
            col_offset=int(overlap_col_offset),
            col_count=int(overlap_col_count),
            metadata={
                "slot_width": slot_width,
                "lane_index": overlap_col_offset // slot_width,
                "source": "slice_range",
            },
        )

    def rebind_view(self, view: ValueTileView, new_value: ValueTile) -> ValueTileView:
        return ValueTileView(
            backing_value_tile_id=new_value.value_tile_id,
            owner_tile_id=view.owner_tile_id,
            row_offset=int(view.row_offset),
            row_count=int(view.row_count),
            col_offset=int(view.col_offset),
            col_count=int(view.col_count),
            metadata=dict(view.metadata),
        )

    def _drop_stale_non_vram_residency(self, value: ValueTile) -> None:
        mram_name = value.residency.pop("mram_name", None)
        if mram_name is not None:
            self.program.compiler.sub_matrix_manager.mram_allocator.free(str(mram_name), strict=False)
        value.residency.pop("mram_addr", None)
        self._value_tiles_in_mram.pop(value.value_tile_id, None)
        self._mram_fifo[:] = [item for item in self._mram_fifo if item != value.value_tile_id]

        # HBM residency is preserved: the tile remains valid in HBM while also
        # resident in VRAM. Only MRAM is evicted on HBM→VRAM moves.

    def _view_covers_logical_tile(self, tile: TensorTile | InputTile, view: ValueTileView) -> bool:
        return (
            int(view.row_offset) == 0
            and int(view.col_offset) == 0
            and int(view.row_count) == int(tile.tile_shape[0])
            and int(view.col_count) == int(tile.tile_shape[1])
        )

    def _prepare_partial_update_vram_successor(self, old_value: ValueTile) -> Optional[ValueTile]:
        has_hbm_backing = (
            old_value.residency.get("hbm_addr") is not None
            and old_value.residency.get("hbm_name") is not None
            and bool(old_value.residency.get("hbm_ready"))
        )
        old_vram_addr = old_value.residency.get("vram_addr")
        if not has_hbm_backing or old_vram_addr is None:
            return None

        new_value = ValueTile(
            value_tile_id=self._next_value_tile_id(),
            logical_shape=old_value.logical_shape,
            metadata=dict(old_value.metadata),
        )
        new_value.from_input_tile = old_value.from_input_tile
        new_value.source_input_tile_id = old_value.source_input_tile_id
        new_value.residency["vram_addr"] = old_value.residency.pop("vram_addr")
        new_value.residency["vram_name"] = old_value.residency.pop("vram_name", None)
        new_value.residency["vram_owner_from"] = old_value.value_tile_id
        self._value_tiles_in_vram.pop(old_value.value_tile_id, None)
        self._value_tiles_in_vram[new_value.value_tile_id] = int(new_value.residency["vram_addr"])
        self.value_tiles[new_value.value_tile_id] = new_value
        return new_value

    def protect_value_tile(self, value: ValueTile, place: str = "vram") -> None:
        if place != "vram":
            raise ValueError(f"Unsupported protect place: {place}")
        already_protected = value.value_tile_id in self._protected_vram_value_tile_ids
        self._protected_vram_value_tile_ids.add(value.value_tile_id)

    def stop_protect_value_tile(self, value: Optional[ValueTile] = None, place: str = "vram") -> None:
        if place != "vram":
            raise ValueError(f"Unsupported protect place: {place}")
        if value is None:
            if not self._protected_vram_value_tile_ids:
                return
            old_value_ids = sorted(self._protected_vram_value_tile_ids)
            self._protected_vram_value_tile_ids.clear()
            return
        if value.value_tile_id not in self._protected_vram_value_tile_ids:
            return
        self._protected_vram_value_tile_ids.remove(value.value_tile_id)

    def _is_protected_value_tile(self, value_tile_id: str, place: str = "vram") -> bool:
        if place != "vram":
            return False
        return value_tile_id in self._protected_vram_value_tile_ids

    def _create_value_tile_for_tile(self, tile: TensorTile | InputTile, *, bind_tile_pointer: bool = True) -> ValueTile:
        if bind_tile_pointer:
            existing_id = self.full_tile_bindings.get(tile.tile_id)
            if existing_id is not None:
                existing = self.value_tiles.get(existing_id)
                if existing is not None:
                    return existing
        value_tile = ValueTile(
            value_tile_id=self._next_value_tile_id(),
            logical_shape=tile.tile_shape,
            from_input_tile=isinstance(tile, InputTile),
            source_input_tile_id=tile.tile_id if isinstance(tile, InputTile) else None,
            metadata={"source_tile_id": tile.tile_id},
        )
        if isinstance(tile, InputTile):
            hbm_name = f"{tile.input_name}.hbm"
            logical_shape = tuple(tile.metadata.get("logical_shape", ()))
            hbm_stride = _logical_shape_to_hbm_stride(logical_shape)
            hbm_offset = _tile_coord_to_hbm_offset(tile.coord, logical_shape, self.program.mlen)
            hbm_addr = self.allocate_value_tile_address(
                size=self.program.tile_elems,
                name=f"{value_tile.value_tile_id}.hbm",
                place="hbm",
                value_tile=value_tile,
                hbm_name=hbm_name,
                hbm_offset=hbm_offset,
                hbm_stride=hbm_stride if hbm_stride > 0 else self.program.mlen,
            )
            value_tile.residency["hbm_addr"] = hbm_addr
            value_tile.residency["hbm_name"] = hbm_name
            value_tile.residency["hbm_offset"] = hbm_offset
            value_tile.residency["hbm_stride"] = hbm_stride if hbm_stride > 0 else self.program.mlen
            value_tile.residency["hbm_ready"] = True
        self.value_tiles[value_tile.value_tile_id] = value_tile
        if bind_tile_pointer:
            self._bind_tile_pointer(tile.tile_id, value_tile.value_tile_id)
        return value_tile

    def create_value_tile_in_fpram_for_tile(
        self,
        tile: TensorTile | InputTile,
        fragment: FPFragment,
        *,
        bind: bool = True,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ValueTile:
        value = self.create_value_tile_in_fpram_from_fp_fragment(
            fragment,
            logical_shape=tile.tile_shape,
            metadata={
                **(dict(metadata) if metadata is not None else {}),
                "source_tile_id": tile.tile_id,
                "source_fragment_name": fragment.name,
            },
        )
        if bind:
            if isinstance(tile, InputTile):
                self._write_value_back_to_input_tile(value, tile)
            else:
                self._bind_value_to_tensor_tile(value, tile)
        return value

    def _iter_group_tiles(self, tile: TensorTile | InputTile) -> List[TensorTile | InputTile]:
        owner_tiles = self._owner_tiles_for_tile(tile)
        group_key = self._view_group_key_for_tile(tile)
        candidates: List[TensorTile | InputTile] = []
        for candidate in _tiles_in_grid_order(owner_tiles):
            if not isinstance(candidate, (TensorTile, InputTile)):
                continue
            if not self._is_narrow_tensor_tile(candidate):
                continue
            if self._view_group_key_for_tile(candidate) != group_key:
                continue
            candidates.append(candidate)
        return candidates

    def _owner_tiles_for_tile(self, tile: TensorTile | InputTile) -> Dict[TileCoord, TensorTile | InputTile]:
        if isinstance(tile, TensorTile):
            owner = self.program.tensor_manager.tensors.get(tile.tensor_name)
            if owner is None:
                raise RuntimeError(f"Unknown tensor owner for tile {tile.tile_id}: {tile.tensor_name}")
            return owner.tiles
        owner = self.program.tensor_manager.inputs.get(tile.input_name)
        if owner is None:
            raise RuntimeError(f"Unknown input owner for tile {tile.tile_id}: {tile.input_name}")
        return owner.tiles

    def _split_mapv_signal(self, items: List[object]) -> Tuple[List[List[object]], Optional[TileLike]]:
        pair_groups: List[List[object]] = []
        dst_tile: Optional[TileLike] = None
        for item in items:
            if isinstance(item, list) and len(item) == 2 and all(_is_tile_object(part) for part in item):
                pair_groups.append(item)
                continue
            if isinstance(item, list) and len(item) == 1 and isinstance(item[0], (TensorTile, InputTile, VectorTile)):
                dst_tile = item[0]
                continue
        return pair_groups, dst_tile

    def _resolve_tile_backing_value(self, tile: TensorTile | InputTile) -> ValueTile:
        canonical_tile = self._resolve_alias_owner_tile(tile)
        if canonical_tile is not tile and not self._is_narrow_tensor_tile(tile):
            tile = canonical_tile
        if self._is_narrow_tensor_tile(tile):
            existing_id = self.full_tile_bindings.get(tile.tile_id)
            if existing_id is not None:
                existing = self.value_tiles.get(existing_id)
                if existing is not None:
                    return existing
            group_key = self._view_group_key_for_tile(tile)
            group_value_id = self.narrow_group_bindings.get(group_key)
            if group_value_id is not None:
                existing = self.value_tiles.get(group_value_id)
                if existing is not None:
                    self._bind_tiles_to_value(self._tiles_sharing_backing(tile), existing.value_tile_id)
                    return existing
            value = self._create_value_tile_for_tile(tile, bind_tile_pointer=False)
            self.narrow_group_bindings[group_key] = value.value_tile_id
            self._bind_tiles_to_value(self._tiles_sharing_backing(tile), value.value_tile_id)
            return value
        existing_id = self.full_tile_bindings.get(tile.tile_id)
        if existing_id is not None:
            existing = self.value_tiles.get(existing_id)
            if existing is not None:
                return existing
        return self._create_value_tile_for_tile(tile, bind_tile_pointer=True)

    def resolve_value_tile(self, tile: TensorTile | InputTile) -> ValueTile:
        return self._resolve_tile_backing_value(tile)

    def get_value_tile(self, tile: TensorTile | InputTile) -> ValueTile:
        # Compatibility wrapper around resolve_value_tile().
        return self.resolve_value_tile(tile)

    def bind_tile_to_fp_fragment(self, tile: VectorTile, fragment: FPFragment) -> FPFragment:
        self.fp_fragment_bindings[tile.tile_id] = fragment.name
        return fragment

    def resolve_fp_fragment(self, tile: VectorTile) -> FPFragment:
        fragment_name = self.fp_fragment_bindings.get(tile.tile_id)
        if not isinstance(fragment_name, str):
            raise RuntimeError(f"VectorTile {tile.tile_id} is not bound to one FPFragment")
        fragment = self.program.tensor_manager.fp_fragments.get(fragment_name)
        if not isinstance(fragment, FPFragment):
            raise RuntimeError(
                f"VectorTile {tile.tile_id} binding points to missing FPFragment {fragment_name!r}"
            )
        return fragment

    def _value_tile_has_live_refs(self, value_tile_id: str) -> bool:
        if self.value_tile_tensor_refs.get(value_tile_id):
            return True
        return False

    def _value_debug_state(self, value: ValueTile) -> Dict[str, object]:
        tensor_refs = sorted(self.value_tile_tensor_refs.get(value.value_tile_id, set()))
        residency = value.residency
        return {
            "value_tile_id": value.value_tile_id,
            "from_input_tile": bool(value.from_input_tile),
            "source_input_tile_id": value.source_input_tile_id,
            "vram_addr": residency.get("vram_addr"),
            "mram_addr": residency.get("mram_addr"),
            "hbm_name": residency.get("hbm_name"),
            "hbm_addr": residency.get("hbm_addr"),
            "hbm_offset": residency.get("hbm_offset"),
            "hbm_stride": residency.get("hbm_stride"),
            "hbm_scale_size": residency.get("hbm_scale_size"),
            "hbm_ready": residency.get("hbm_ready"),
            "tensor_refs": tensor_refs,
            "last_move": value.metadata.get("last_move"),
        }

    def _tile_debug_state(self, tile: TensorTile | InputTile) -> Dict[str, object]:
        state: Dict[str, object] = {
            "tile_id": tile.tile_id,
            "coord": tile.coord,
            "tile_shape": tile.tile_shape,
            "kind": type(tile).__name__,
        }
        if isinstance(tile, InputTile):
            state["owner"] = tile.input_name
        elif isinstance(tile, TensorTile):
            state["owner"] = tile.tensor_name
        logical_shape = tile.metadata.get("logical_shape")
        if logical_shape is not None:
            state["logical_shape"] = logical_shape
        return state

    def prepare_vram_backing_value(
        self,
        value: Optional[ValueTile] = None,
        *,
        preserve_existing: bool = False,
    ) -> ValueTile:
        if value is not None and not preserve_existing and not self._value_tile_has_live_refs(value.value_tile_id):
            self.ensure_value_tile_in_place(value, "vram")
            return value
        new_value_tile = ValueTile(
            value_tile_id=self._next_value_tile_id(),
            logical_shape=value.logical_shape if value is not None else (self.program.mlen, self.program.mlen),
            metadata=dict(value.metadata) if value is not None else {},
        )
        if value is not None:
            new_value_tile.from_input_tile = value.from_input_tile
            new_value_tile.source_input_tile_id = value.source_input_tile_id
            has_live_refs = self._value_tile_has_live_refs(value.value_tile_id)
            can_transfer_vram = (
                value.residency.get("vram_addr") is not None
                and not self._is_protected_value_tile(value.value_tile_id, "vram")
                and not has_live_refs
            )
            if can_transfer_vram:
                new_value_tile.residency["vram_addr"] = value.residency.pop("vram_addr")
                new_value_tile.residency["vram_name"] = value.residency.pop("vram_name", None)
                new_value_tile.residency["vram_owner_from"] = value.value_tile_id
                old_addr = self._value_tiles_in_vram.pop(value.value_tile_id, None)
                if old_addr is not None:
                    self._value_tiles_in_vram[new_value_tile.value_tile_id] = old_addr
            elif (
                has_live_refs
                and (
                    value.residency.get("vram_addr") is not None
                    or value.residency.get("hbm_addr") is not None
                    or value.residency.get("hbm_ready")
                )
            ):
                self.ensure_value_tile_in_place(value, "hbm")
        if new_value_tile.residency.get("vram_addr") is None:
            vram_name = f"{new_value_tile.value_tile_id}.vram"
            vram_addr = self.allocate_value_tile_address(
                size=self.program.tile_elems,
                name=vram_name,
                place="vram",
                value_tile=new_value_tile,
            )
            new_value_tile.residency["vram_addr"] = vram_addr
            new_value_tile.residency["vram_name"] = vram_name
            self._value_tiles_in_vram[new_value_tile.value_tile_id] = vram_addr
        self.value_tiles[new_value_tile.value_tile_id] = new_value_tile
        return new_value_tile

    def create_value_tile_in_fpram(
        self,
        *,
        logical_shape: Tuple[int, int],
        fpram_addr: int,
        fpram_size: int,
        fpram_name: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ValueTile:
        value_tile = ValueTile(
            value_tile_id=self._next_value_tile_id(),
            logical_shape=tuple(int(dim) for dim in logical_shape),
            metadata=dict(metadata) if metadata is not None else {},
        )
        value_tile.residency["fpram_addr"] = int(fpram_addr)
        value_tile.residency["fpram_name"] = str(fpram_name)
        value_tile.residency["fpram_size"] = int(fpram_size)
        value_tile.residency["fpram_ready"] = True
        self.value_tiles[value_tile.value_tile_id] = value_tile
        return value_tile

    def create_value_tile_in_fpram_from_fp_fragment(
        self,
        fragment: FPFragment,
        *,
        logical_shape: Optional[Tuple[int, int]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ValueTile:
        fragment_shape = tuple(int(dim) for dim in fragment.shape)
        tile_rows, tile_cols = _fp_fragment_shape_to_tile_shape(
            fragment_shape,
            mlen=self.program.mlen,
            btmm_hlen=self.program.btmm_hlen,
        )
        fp_vars = [fragment.vars[index] for index in _iter_fp_indices(fragment_shape)]
        fp_addrs = [_require_fp_addr(fp_var) for fp_var in fp_vars]
        fp_prog = self.program._arith_progression(fp_addrs)
        expected_cells = tile_rows * tile_cols
        if len(fp_addrs) != expected_cells:
            raise RuntimeError(
                f"FPFragment {fragment.name!r} expected {expected_cells} FP cells for one tile, got {len(fp_addrs)}"
            )
        fp_base_addr = int(fp_addrs[0]) if fp_addrs else 0
        fp_dense = bool(fp_prog is not None and fp_prog[1] == expected_cells and fp_prog[2] == 1)

        return self.create_value_tile_in_fpram(
            logical_shape=logical_shape if logical_shape is not None else (tile_rows, tile_cols),
            fpram_addr=int(fp_base_addr),
            fpram_size=int(expected_cells),
            fpram_name=fragment.name,
            metadata={
                **(dict(metadata) if metadata is not None else {}),
                "fp_fragment_name": fragment.name,
                "fp_fragment_shape": fragment_shape,
                "fp_materialized_tile_shape": (tile_rows, tile_cols),
                "fp_fragment_dense": fp_dense,
            },
        )

    def _resolve_value_fp_fragment(self, value: ValueTile) -> FPFragment:
        fragment_name = value.metadata.get("fp_fragment_name")
        if not isinstance(fragment_name, str):
            raise RuntimeError(
                f"fpram-backed value tile {value.value_tile_id} is missing fp_fragment_name metadata"
            )
        fragment = self.program.tensor_manager.fp_fragments.get(fragment_name)
        if not isinstance(fragment, FPFragment):
            raise RuntimeError(
                f"fpram-backed value tile {value.value_tile_id} references missing FPFragment {fragment_name!r}"
            )
        return fragment

    def _temporary_fpram_row_scratch(self, row_width: int, *, value_tile_id: str, row_index: int) -> Tuple[str, int]:
        allocator = self.program.compiler.sub_matrix_manager.fpram_allocator
        floor = int(self.program.tensor_manager._next_fp_mem_addr)
        if allocator.next_free < floor:
            allocator.next_free = floor
        allocator.free_stack[:] = [
            block for block in allocator.free_stack
            if int(block.addr) >= floor
        ]
        scratch_name = f"__fpram_row_scratch__.{value_tile_id}.row{row_index}"
        scratch_addr = allocator.allocate(scratch_name, row_width)
        return scratch_name, int(scratch_addr)

    def evaluate_contiguous_vram_value_tile_window(
        self,
        *,
        tile_count: int,
        reason: str = "contiguous_vram_window",
    ) -> Dict[str, object]:
        if tile_count <= 0:
            raise ValueError(f"tile_count must be positive, got {tile_count}")

        allocator = self.program.compiler.sub_matrix_manager.vram_allocator
        tile_size = self.program.tile_elems
        window_size = tile_count * tile_size
        candidates: List[Dict[str, object]] = []

        for block in sorted(allocator.free_stack, key=lambda item: (item.size, item.addr)):
            if block.size < window_size:
                continue
            waste = int(block.size - window_size)
            candidates.append(
                {
                    "kind": "free_stack",
                    "addr": int(block.addr),
                    "size": int(block.size),
                    "cost": waste,
                    "waste": waste,
                    "block_name": block.name,
                }
            )

        aligned_bump_addr = ((int(allocator.next_free) + tile_size - 1) // tile_size) * tile_size
        candidates.append(
            {
                "kind": "bump",
                "addr": aligned_bump_addr,
                "size": window_size,
                "cost": window_size,
                "waste": 0,
                "block_name": "<bump>",
            }
        )

        candidates.sort(key=lambda item: (int(item["cost"]), int(item["waste"]), int(item["addr"])))
        chosen = dict(candidates[0])
        plan = {
            "reason": reason,
            "tile_count": tile_count,
            "tile_size": tile_size,
            "window_size": window_size,
            "chosen": chosen,
            "candidates": candidates,
        }
        return plan

    def allocate_contiguous_vram_value_tiles(
        self,
        *,
        tile_count: int,
        logical_shape: Optional[Tuple[int, int]] = None,
        metadata: Optional[Dict[str, object]] = None,
        reason: str = "contiguous_vram_window",
    ) -> Tuple[List[ValueTile], int]:
        plan = self.evaluate_contiguous_vram_value_tile_window(tile_count=tile_count, reason=reason)
        alloc_name = f"contiguous_values.{self._next_value_tile_id()}.vram"
        window_size = int(plan["window_size"])
        tile_size = int(plan["tile_size"])
        base_addr = self.program.compiler.sub_matrix_manager.vram_allocator.allocate(size=window_size, name=alloc_name)

        template_metadata = dict(metadata) if metadata is not None else {}
        reserved_values: List[ValueTile] = []
        for lane in range(tile_count):
            value = ValueTile(
                value_tile_id=self._next_value_tile_id(),
                logical_shape=logical_shape if logical_shape is not None else (self.program.mlen, self.program.mlen),
                metadata={
                    **template_metadata,
                    "contiguous_lane_index": lane,
                },
            )
            vram_addr = base_addr + lane * tile_size
            value.residency["vram_addr"] = vram_addr
            value.residency["vram_name"] = alloc_name
            value.residency["vram_lane_index"] = lane
            self.value_tiles[value.value_tile_id] = value
            self._value_tiles_in_vram[value.value_tile_id] = vram_addr
            self._touch_fifo("vram", value.value_tile_id)
            reserved_values.append(value)

        return reserved_values, base_addr

    def ensure_value_tile_in_place(self, value: ValueTile, place: str) -> ValueTile:
        if place == "vram":
            if value.residency.get("vram_addr") is not None:
                return value
            if value.residency.get("fpram_ready"):
                vram_name = value.residency.get("vram_name") or f"{value.value_tile_id}.vram"
                vram_addr = self.allocate_value_tile_address(
                    size=self.program.tile_elems,
                    name=str(vram_name),
                    place="vram",
                    value_tile=value,
                )
                value.residency["vram_addr"] = vram_addr
                value.residency["vram_name"] = vram_name
                self.move_tile(value, "fpram", "vram")
                self._value_tiles_in_vram[value.value_tile_id] = vram_addr
                return value
            # Fresh output/scratch values may not have any HBM provenance yet.
            # For those, materialize directly in VRAM instead of forcing an HBM round-trip.
            if value.residency.get("hbm_addr") is None and not value.residency.get("hbm_ready"):
                vram_name = value.residency.get("vram_name") or f"{value.value_tile_id}.vram"
                vram_addr = self.allocate_value_tile_address(
                    size=self.program.tile_elems,
                    name=str(vram_name),
                    place="vram",
                    value_tile=value,
                )
                value.residency["vram_addr"] = vram_addr
                value.residency["vram_name"] = vram_name
                self._value_tiles_in_vram[value.value_tile_id] = vram_addr
                return value
            self.ensure_value_tile_in_place(value, "hbm")
            if value.residency.get("vram_addr") is None:
                vram_name = value.residency.get("vram_name") or f"{value.value_tile_id}.vram"
                vram_addr = self.allocate_value_tile_address(
                    size=self.program.tile_elems,
                    name=str(vram_name),
                    place="vram",
                    value_tile=value,
                )
                value.residency["vram_addr"] = vram_addr
                value.residency["vram_name"] = vram_name
            self.move_tile(value, "hbm", "vram")
            if value.residency.get("vram_addr") is not None:
                self._value_tiles_in_vram[value.value_tile_id] = value.residency["vram_addr"]
            self.program._record_operation_snapshot(
                "value_residency",
                stage="ensure",
                target_place="vram",
                value=self._value_debug_state(value),
            )
            return value
        if place == "mram":
            if value.residency.get("mram_addr") is not None:
                return value
            if value.residency.get("mram_addr") is None:
                mram_name = f"{value.value_tile_id}.mram"
                mram_addr = self.allocate_value_tile_address(
                    name=mram_name,
                    size=self.program.tile_elems,
                    place="mram",
                    value_tile=value,
                )
                value.residency["mram_addr"] = mram_addr
                value.residency["mram_name"] = mram_name
            self.ensure_value_tile_in_place(value, "hbm")
            self.move_tile(value, "hbm", "mram")
            self._value_tiles_in_mram[value.value_tile_id] = value.residency["mram_addr"]
            return value
        if place == "fpram":
            if value.residency.get("fpram_ready"):
                return value
            if value.residency.get("vram_addr") is not None and value.metadata.get("fp_fragment_name") is not None:
                self.move_tile(value, "vram", "fpram")
                return value
            raise RuntimeError(
                f"Value tile {value.value_tile_id} is not fpram-backed; current implementation only "
                "supports values created initially in fpram"
            )
        if place == "hbm":
            if value.residency.get("hbm_ready"):
                self._value_tiles_in_hbm[value.value_tile_id] = True
                return value
            if value.residency.get("fpram_ready"):
                self.ensure_value_tile_in_place(value, "vram")
                self.move_tile(value, "vram", "hbm")
                value.residency["hbm_ready"] = True
                self._value_tiles_in_hbm[value.value_tile_id] = {
                    "addr": value.residency.get("hbm_addr"),
                    "name": value.residency.get("hbm_name"),
                    "offset": value.residency.get("hbm_offset"),
                    "stride": value.residency.get("hbm_stride"),
                }
                return value
            if value.residency.get("vram_addr") is None:
                if value.residency.get("hbm_addr") is not None:
                    value.residency["hbm_ready"] = True
                    self._value_tiles_in_hbm[value.value_tile_id] = {
                        "addr": value.residency.get("hbm_addr"),
                        "name": value.residency.get("hbm_name"),
                        "offset": value.residency.get("hbm_offset"),
                        "stride": value.residency.get("hbm_stride"),
                    }
                    return value
                raise RuntimeError(
                    f"Value tile {value.value_tile_id} is neither in HBM nor VRAM; refusing to ensure HBM to avoid loops"
                )
            if value.residency.get("hbm_addr") is None:
                hbm_name = f"{value.value_tile_id}.hbm"
                hbm_addr = self.allocate_value_tile_address(
                    size=self.program.tile_elems,
                    name=hbm_name,
                    place="hbm",
                    value_tile=value,
                )
                value.residency["hbm_addr"] = hbm_addr
                value.residency["hbm_name"] = hbm_name
                value.residency["hbm_offset"] = 0
                value.residency["hbm_stride"] = self.program.mlen
            self.move_tile(value, "vram", "hbm")
            value.residency["hbm_ready"] = True
            self._value_tiles_in_hbm[value.value_tile_id] = {
                "addr": value.residency.get("hbm_addr"),
                "name": value.residency.get("hbm_name"),
                "offset": value.residency.get("hbm_offset"),
                "stride": value.residency.get("hbm_stride"),
            }
            self.program._record_operation_snapshot(
                "value_residency",
                stage="ensure",
                target_place="hbm",
                value=self._value_debug_state(value),
            )
            return value
        raise ValueError(f"Unsupported place for ensure_value_tile_in_place: {place}")

    def move_tile(self, value: ValueTile, src_place: str, dst_place: str) -> None:
        if src_place == "fpram" and dst_place == "vram":
            fpram_addr = value.residency.get("fpram_addr")
            vram_addr = value.residency.get("vram_addr")
            fragment_shape = value.metadata.get("fp_fragment_shape")
            if vram_addr is None:
                raise RuntimeError(
                    f"move_tile fpram->vram requires vram_addr for {value.value_tile_id}"
                )
            if not isinstance(fragment_shape, tuple):
                raise RuntimeError(
                    f"fpram-backed value tile {value.value_tile_id} is missing fp_fragment_shape metadata"
                )
            fragment = self._resolve_value_fp_fragment(value)
            row_count, row_width = _fp_fragment_shape_to_tile_shape(
                tuple(int(dim) for dim in fragment_shape),
                mlen=self.program.mlen,
                btmm_hlen=self.program.btmm_hlen,
            )
            slow_rows = 0
            for row_index in range(int(row_count)):
                row_fp_vars = _fp_fragment_row_fp_vars(
                    fragment,
                    row_index=row_index,
                    row_width=int(row_width),
                    btmm_hlen=self.program.btmm_hlen,
                )
                row_addrs = [_require_fp_addr(fp_var) for fp_var in row_fp_vars]
                row_prog = self.program._arith_progression(row_addrs)
                row_vram_addr = int(vram_addr) + row_index * int(row_width)
                if row_prog is not None and row_prog[1] == int(row_width) and row_prog[2] == 1:
                    self.program.emit_map_v_fp_tile(
                        vram_addr=row_vram_addr,
                        fpram_addr=int(row_prog[0]),
                        row_count=1,
                        row_width=int(row_width),
                        task_id=f"fpram_to_vram.{value.value_tile_id}.row{row_index}",
                    )
                    continue

                slow_rows += 1
                scratch_name, scratch_addr = self._temporary_fpram_row_scratch(
                    int(row_width),
                    value_tile_id=value.value_tile_id,
                    row_index=row_index,
                )
                scratch_addrs = [scratch_addr + offset for offset in range(int(row_width))]
                try:
                    self.program.emit_fp_kernel(
                        src1_addrs=row_addrs,
                        dst_addrs=scratch_addrs,
                        op="copy",
                        task_id=f"fpram_row_gather.{value.value_tile_id}.row{row_index}",
                    )
                    self.program.emit_map_v_fp_tile(
                        vram_addr=row_vram_addr,
                        fpram_addr=int(scratch_addr),
                        row_count=1,
                        row_width=int(row_width),
                        task_id=f"fpram_to_vram.{value.value_tile_id}.row{row_index}.scratch",
                    )
                finally:
                    self.program.compiler.sub_matrix_manager.fpram_allocator.free(scratch_name, strict=False)
            value.metadata["last_move"] = ("fpram", "vram")
            value.residency.pop("fpram_addr", None)
            value.residency.pop("fpram_name", None)
            value.residency.pop("fpram_size", None)
            value.residency.pop("fpram_ready", None)
            value.residency.pop("hbm_addr", None)
            value.residency.pop("hbm_name", None)
            value.residency.pop("hbm_offset", None)
            value.residency.pop("hbm_stride", None)
            value.residency.pop("hbm_scale_size", None)
            value.residency.pop("hbm_ready", None)
            value.residency.pop("mram_addr", None)
            value.residency.pop("mram_name", None)
            self._value_tiles_in_hbm.pop(value.value_tile_id, None)
            self._value_tiles_in_mram.pop(value.value_tile_id, None)
            self._mram_fifo[:] = [item for item in self._mram_fifo if item != value.value_tile_id]
            return
        if src_place == "vram" and dst_place == "hbm":
            vram_addr = value.residency.get("vram_addr")
            hbm_params = self._hbm_base_offset_scale_for_value(value)
            hbm_addr = hbm_params["hbm_addr"]
            hbm_name = hbm_params["hbm_name"]
            if vram_addr is None or hbm_addr is None or hbm_name is None:
                raise RuntimeError(
                    f"move_tile vram->hbm requires vram_addr/hbm_addr/hbm_name for {value.value_tile_id}"
                )
            self.program.emit_store_tile_to_hbm(
                vram_addr=int(vram_addr),
                hbm_addr=int(hbm_params["hbm_base_addr"]),
                hbm_stride=int(hbm_params["hbm_stride"]),
                hbm_scale_size=int(hbm_params["hbm_scale_size"]),
                hbm_start_offset=int(hbm_params["hbm_offset"]),
            )
            value.metadata["last_move"] = ("vram", "hbm")
            self.program._record_operation_snapshot(
                "value_residency",
                stage="move_tile",
                src_place="vram",
                dst_place="hbm",
                hbm_params=dict(hbm_params),
                value=self._value_debug_state(value),
            )
            return
        if src_place == "hbm" and dst_place == "vram":
            hbm_params = self._hbm_base_offset_scale_for_value(value)
            hbm_addr = hbm_params["hbm_addr"]
            vram_addr = value.residency.get("vram_addr")
            hbm_name = hbm_params["hbm_name"]
            if hbm_addr is None or vram_addr is None or hbm_name is None:
                raise RuntimeError(f"move_tile hbm->vram requires both hbm_addr and vram_addr for {value.value_tile_id}")
            self.program.emit_load_tile_from_hbm(
                hbm_addr=int(hbm_params["hbm_base_addr"]),
                vram_addr=int(vram_addr),
                hbm_stride=int(hbm_params["hbm_stride"]),
                hbm_scale_size=int(hbm_params["hbm_scale_size"]),
                hbm_start_offset=int(hbm_params["hbm_offset"]),
            )
            value.metadata["last_move"] = ("hbm", "vram")
            self._drop_stale_non_vram_residency(value)
            self.program._record_operation_snapshot(
                "value_residency",
                stage="move_tile",
                src_place="hbm",
                dst_place="vram",
                hbm_params=dict(hbm_params),
                value=self._value_debug_state(value),
            )
            return
        if src_place == "hbm" and dst_place == "mram":
            hbm_params = self._hbm_base_offset_scale_for_value(value)
            hbm_addr = hbm_params["hbm_addr"]
            mram_addr = value.residency.get("mram_addr")
            if hbm_addr is None or mram_addr is None:
                raise RuntimeError(f"move_tile hbm->mram requires both hbm_addr and mram_addr for {value.value_tile_id}")
            self.program.emit_hbm_tile_to_mram(
                hbm_addr=int(hbm_params["hbm_base_addr"]),
                mram_addr=int(mram_addr),
                hbm_offset=int(hbm_params["hbm_offset"]),
                hbm_scale=int(hbm_params["hbm_scale_size"]),
                hbm_stride=int(hbm_params["hbm_stride"]),
            )
            value.metadata["last_move"] = ("hbm", "mram")
            return
        if src_place == "vram" and dst_place == "fpram":
            vram_addr = value.residency.get("vram_addr")
            if vram_addr is None:
                raise RuntimeError(
                    f"move_tile vram->fpram requires vram_addr for {value.value_tile_id}"
                )
            fragment = self._resolve_value_fp_fragment(value)
            fragment_shape = tuple(int(dim) for dim in fragment.shape)
            row_count, row_width = _fp_fragment_shape_to_tile_shape(
                fragment_shape,
                mlen=self.program.mlen,
                btmm_hlen=self.program.btmm_hlen,
            )
            for row_index in range(int(row_count)):
                row_fp_vars = _fp_fragment_row_fp_vars(
                    fragment,
                    row_index=row_index,
                    row_width=int(row_width),
                    btmm_hlen=self.program.btmm_hlen,
                )
                row_addrs = [_require_fp_addr(fp_var) for fp_var in row_fp_vars]
                row_prog = self.program._arith_progression(row_addrs)
                row_vram_addr = int(vram_addr) + row_index * int(row_width)
                if row_prog is not None and row_prog[1] == int(row_width) and row_prog[2] == 1:
                    self.program.emit_map_fp_v_tile(
                        fpram_addr=int(row_prog[0]),
                        vram_addr=row_vram_addr,
                        row_count=1,
                        row_width=int(row_width),
                        task_id=f"vram_to_fpram.{value.value_tile_id}.row{row_index}",
                    )
                    continue

                scratch_name, scratch_addr = self._temporary_fpram_row_scratch(
                    int(row_width),
                    value_tile_id=value.value_tile_id,
                    row_index=row_index,
                )
                scratch_addrs = [scratch_addr + offset for offset in range(int(row_width))]
                try:
                    self.program.emit_map_fp_v_tile(
                        fpram_addr=int(scratch_addr),
                        vram_addr=row_vram_addr,
                        row_count=1,
                        row_width=int(row_width),
                        task_id=f"vram_to_fpram.{value.value_tile_id}.row{row_index}.scratch",
                    )
                    self.program.emit_fp_kernel(
                        src1_addrs=scratch_addrs,
                        dst_addrs=row_addrs,
                        op="copy",
                        task_id=f"fpram_row_scatter.{value.value_tile_id}.row{row_index}",
                    )
                finally:
                    self.program.compiler.sub_matrix_manager.fpram_allocator.free(scratch_name, strict=False)

            fp_vars = [fragment.vars[index] for index in _iter_fp_indices(fragment_shape)]
            fp_addrs = [_require_fp_addr(fp_var) for fp_var in fp_vars]
            value.residency["fpram_name"] = fragment.name
            value.residency["fpram_size"] = len(fp_addrs)
            value.residency["fpram_ready"] = True
            if fp_addrs:
                value.residency["fpram_addr"] = int(fp_addrs[0])
            value.metadata["last_move"] = ("vram", "fpram")
            return
        raise ValueError(f"Unsupported move_tile path: {src_place} -> {dst_place}")

    def _hbm_base_offset_scale_for_value(self, value: ValueTile) -> Dict[str, object]:
        explicit_hbm_name = value.residency.get("hbm_name")
        explicit_hbm_addr = value.residency.get("hbm_addr")
        explicit_hbm_offset = value.residency.get("hbm_offset")
        explicit_hbm_stride = value.residency.get("hbm_stride")
        if (
            explicit_hbm_name is not None
            and explicit_hbm_addr is not None
            and explicit_hbm_offset is not None
            and explicit_hbm_stride is not None
        ):
            hbm_object = self.program.hardware.hbm_objects.get(str(explicit_hbm_name))
            if hbm_object is None:
                raise RuntimeError(
                    f"Value tile {value.value_tile_id} references missing explicit HBM object {explicit_hbm_name}"
                )
            hbm_shape = tuple(hbm_object.get("shape", (self.program.mlen, self.program.mlen)))
            hbm_scale_size = int(value.residency.get("hbm_scale_size", int(hbm_shape[0]) * int(hbm_shape[1])))
            hbm_base_addr = int(hbm_object["base_addr"])
            return {
                "hbm_name": str(explicit_hbm_name),
                "hbm_addr": int(explicit_hbm_addr),
                "hbm_base_addr": hbm_base_addr,
                "hbm_offset": int(explicit_hbm_offset),
                "hbm_stride": int(explicit_hbm_stride),
                "hbm_scale_size": hbm_scale_size,
            }
        if value.from_input_tile and value.source_input_tile_id is not None:
            input_tile = self.program.tensor_manager.input_tiles.get(value.source_input_tile_id)
            if input_tile is not None:
                input_obj = self.program.tensor_manager.inputs.get(input_tile.input_name)
                hbm_name = (
                    input_obj.metadata.get("hbm_group_obj", f"{input_tile.input_name}.hbm")
                    if input_obj is not None
                    else f"{input_tile.input_name}.hbm"
                )
                logical_shape = tuple(input_tile.metadata.get("logical_shape", ()))
                hbm_stride = _logical_shape_to_hbm_stride(logical_shape)
                hbm_offset = _tile_coord_to_hbm_offset(input_tile.coord, logical_shape, self.program.mlen)
                hbm_object = self.program.hardware.hbm_objects.get(str(hbm_name))
                if hbm_object is None:
                    raise RuntimeError(
                        f"Input-backed value tile {value.value_tile_id} references missing HBM object {hbm_name}"
                    )
                hbm_shape = tuple(hbm_object.get("shape", (self.program.mlen, self.program.mlen)))
                hbm_scale_size = int(hbm_shape[0]) * int(hbm_shape[1])
                hbm_base_addr = int(hbm_object["base_addr"])
                hbm_addr = hbm_base_addr + int(hbm_offset)
                value.residency["hbm_name"] = str(hbm_name)
                value.residency["hbm_addr"] = hbm_addr
                value.residency["hbm_offset"] = int(hbm_offset)
                value.residency["hbm_stride"] = int(hbm_stride)
                value.residency["hbm_scale_size"] = int(hbm_scale_size)
                return {
                    "hbm_name": str(hbm_name),
                    "hbm_addr": hbm_addr,
                    "hbm_base_addr": hbm_base_addr,
                    "hbm_offset": int(hbm_offset),
                    "hbm_stride": int(hbm_stride),
                    "hbm_scale_size": int(hbm_scale_size),
                }

        hbm_name = value.residency.get("hbm_name")
        hbm_addr = value.residency.get("hbm_addr")
        if hbm_name is None or hbm_addr is None:
            raise RuntimeError(f"Value tile {value.value_tile_id} is missing HBM metadata")
        hbm_object = self.program.hardware.hbm_objects.get(str(hbm_name))
        if hbm_object is None:
            raise RuntimeError(f"Unknown HBM object for value tile {value.value_tile_id}: {hbm_name}")
        hbm_base_addr = int(hbm_object["base_addr"])
        hbm_shape = tuple(hbm_object.get("shape", (self.program.mlen, self.program.mlen)))
        explicit_hbm_scale_size = value.residency.get("hbm_scale_size")
        hbm_scale_size = int(explicit_hbm_scale_size) if explicit_hbm_scale_size is not None else int(hbm_shape[0]) * int(hbm_shape[1])
        hbm_offset = int(value.residency.get("hbm_offset", int(hbm_addr) - hbm_base_addr))
        hbm_stride = int(value.residency.get("hbm_stride", self.program.mlen))
        if explicit_hbm_scale_size is None:
            value.residency["hbm_scale_size"] = int(hbm_scale_size)
        return {
            "hbm_name": str(hbm_name),
            "hbm_addr": int(hbm_addr),
            "hbm_base_addr": hbm_base_addr,
            "hbm_offset": int(hbm_offset),
            "hbm_stride": int(hbm_stride),
            "hbm_scale_size": int(hbm_scale_size),
        }

    def allocate_value_tile_address(
        self,
        *,
        size: int,
        name: str,
        place: str,
        value_tile: Optional[ValueTile] = None,
        hbm_name: Optional[str] = None,
        hbm_offset: int = 0,
        hbm_stride: Optional[int] = None,
    ) -> int:
        if place == "vram":
            self._evict_fifo_if_needed("vram")
            if value_tile is not None:
                self._touch_fifo("vram", value_tile.value_tile_id)
            addr = self.program.compiler.sub_matrix_manager.vram_allocator.allocate(size=size, name=name)
            return addr
        if place == "mram":
            self._evict_fifo_if_needed("mram")
            if value_tile is not None:
                self._touch_fifo("mram", value_tile.value_tile_id)
            addr = self.program.compiler.sub_matrix_manager.mram_allocator.allocate(name=name, size=size)
            return addr
        if place == "hbm":
            resolved_name = hbm_name or name
            if resolved_name not in self.program.hardware.hbm_objects:
                base_addr = self.program.add_hbm_object(
                    resolved_name,
                    (self.program.mlen, self.program.mlen),
                )
            else:
                base_addr = self.program.hardware.hbm_objects[resolved_name]["base_addr"]
            hbm_object = self.program.hardware.hbm_objects[resolved_name]
            hbm_shape = tuple(hbm_object.get("shape", (self.program.mlen, self.program.mlen)))
            hbm_scale_size = int(hbm_shape[0]) * int(hbm_shape[1])
            addr = base_addr + int(hbm_offset)
            if value_tile is not None:
                scale_size = int(value_tile.residency.get("hbm_scale_size", hbm_scale_size))
                self._value_tiles_in_hbm[value_tile.value_tile_id] = {
                    "addr": addr,
                    "name": resolved_name,
                    "offset": int(hbm_offset),
                    "stride": self.program.mlen if hbm_stride is None else int(hbm_stride),
                    "scale_size": scale_size,
                }
                if "hbm_scale_size" not in value_tile.residency:
                    value_tile.residency["hbm_scale_size"] = hbm_scale_size
            return addr
        raise ValueError(f"Unsupported place for allocate_value_tile_address: {place}")

    def _touch_fifo(self, place: str, value_tile_id: str) -> None:
        if place == "vram":
            if value_tile_id in self._value_tiles_in_vram:
                addr = self._value_tiles_in_vram.pop(value_tile_id)
                self._value_tiles_in_vram[value_tile_id] = addr
            return
        fifo = self._mram_fifo
        fifo[:] = [item for item in fifo if item != value_tile_id]
        fifo.append(value_tile_id)

    def _evict_fifo_if_needed(self, place: str) -> None:
        if place == "vram":
            capacity = getattr(self.program, "vram_tile_capacity", 0)
            if capacity > 0 and len(self._value_tiles_in_vram) >= capacity:
                self._evict_one_value_tile("vram")
            return
        if place == "mram":
            capacity = getattr(self.program, "mram_tile_capacity", 0)
            if capacity > 0 and len(self._value_tiles_in_mram) >= capacity:
                self._evict_one_value_tile("mram")
            return

    def _evict_one_value_tile(self, place: str) -> None:
        residency_table = self._value_tiles_in_vram if place == "vram" else self._value_tiles_in_mram
        addr_key = "vram_addr" if place == "vram" else "mram_addr"
        name_key = "vram_name" if place == "vram" else "mram_name"
        allocator = (
            self.program.compiler.sub_matrix_manager.vram_allocator
            if place == "vram"
            else self.program.compiler.sub_matrix_manager.mram_allocator
        )
        if place == "vram":
            resident_ids = list(residency_table.keys())
            if not resident_ids:
                raise RuntimeError(f"{place.upper()} allocation requested but no resident value tile was available for FIFO eviction")
            skipped_protected = 0
            while resident_ids:
                evict_id = resident_ids.pop(0)
                if self._is_protected_value_tile(evict_id, "vram"):
                    addr = residency_table.pop(evict_id)
                    residency_table[evict_id] = addr
                    skipped_protected += 1
                    if skipped_protected >= len(residency_table):
                        raise RuntimeError(
                            f"VRAM eviction stalled because all resident value tiles are currently protected"
                        )
                    continue
                evict_value = self.value_tiles.get(evict_id)
                if evict_value is None:
                    raise RuntimeError(
                        f"{place.upper()} residency table references missing value tile {evict_id}; internal residency state is inconsistent"
                    )
                self.ensure_value_tile_in_place(evict_value, "hbm")
                alloc_name = evict_value.residency.get(name_key)
                if alloc_name is not None:
                    allocator.free(str(alloc_name), strict=False)
                evict_value.residency.pop(addr_key, None)
                evict_value.residency.pop(name_key, None)
                residency_table.pop(evict_id, None)
                return
            raise RuntimeError(f"{place.upper()} allocation requested but no resident value tile was available for FIFO eviction")

        fifo = self._mram_fifo
        while fifo:
            evict_id = fifo.pop(0)
            evict_value = self.value_tiles.get(evict_id)
            if evict_value is None:
                raise RuntimeError(
                    f"{place.upper()} FIFO references missing value tile {evict_id}; internal residency state is inconsistent"
                )
            alloc_name = evict_value.residency.get(name_key)
            if alloc_name is not None:
                allocator.free(str(alloc_name), strict=False)
            evict_value.residency.pop(addr_key, None)
            evict_value.residency.pop(name_key, None)
            residency_table.pop(evict_id, None)
            return
        raise RuntimeError(f"{place.upper()} allocation requested but no resident value tile was available for FIFO eviction")

    def mapv_back(self, signal: List[object]) -> Dict[str, object]:
        compute_output, mapv_input = signal
        dst_value = compute_output.get("dst") if isinstance(compute_output, dict) else None
        if not isinstance(mapv_input, tuple) or not mapv_input:
            raise RuntimeError("mapv_back expects one tuple mapv packet")
        control = mapv_input[0]
        if control == "copy":
            if len(mapv_input) != 3:
                raise RuntimeError("copy mapv_back expects ('copy', src_value, dst_tile)")
            _, src_value, dst_tile = mapv_input
            if not isinstance(src_value, ValueTile):
                raise RuntimeError("copy mapv_back expects one source ValueTile")
            if not isinstance(dst_tile, (TensorTile, InputTile)):
                raise RuntimeError("copy mapv_back expects one destination tile")
            if isinstance(dst_tile, InputTile):
                self._write_value_back_to_input_tile(src_value, dst_tile)
            else:
                self._bind_value_to_tensor_tile(src_value, dst_tile)
            return {
                "mapped_values": compute_output,
                "mapv_input": mapv_input,
                "dst_tile_id": dst_tile.tile_id,
                "dst_value_tile_id": src_value.value_tile_id,
                "control": control,
            }

        if len(mapv_input) != 4:
            raise RuntimeError("matmul mapv_back expects ('matmul', src_pairs, dst_value, dst_tile)")
        _, _, _, dst_tile = mapv_input
        if not isinstance(dst_value, ValueTile):
            raise RuntimeError("mapv_back expects compute output to contain one destination ValueTile")
        if not isinstance(dst_tile, (TensorTile, InputTile)):
            raise RuntimeError("mapv_back expects mapv input to contain one destination tile")
        if isinstance(dst_tile, InputTile):
            self._write_value_back_to_input_tile(dst_value, dst_tile)
        else:
            self._bind_value_to_tensor_tile(dst_value, dst_tile)
        return {
            "mapped_values": compute_output,
            "mapv_input": mapv_input,
            "dst_tile_id": dst_tile.tile_id,
            "dst_value_tile_id": dst_value.value_tile_id,
            "control": control,
        }

    def _write_value_back_to_input_tile(self, value: ValueTile, dst_tile: InputTile) -> None:
        original_value = value
        input_obj = self.program.tensor_manager.inputs.get(dst_tile.input_name)
        if input_obj is None:
            raise RuntimeError(f"Unknown input owner for input tile {dst_tile.tile_id}: {dst_tile.input_name}")
        hbm_name = input_obj.metadata.get("hbm_group_obj", f"{dst_tile.input_name}.hbm")
        logical_shape = tuple(dst_tile.metadata.get("logical_shape", ()))
        hbm_stride = _logical_shape_to_hbm_stride(logical_shape)
        hbm_offset = _tile_coord_to_hbm_offset(dst_tile.coord, logical_shape, self.program.mlen)
        hbm_object = self.program.hardware.hbm_objects.get(str(hbm_name))
        if hbm_object is None:
            raise RuntimeError(f"Unknown HBM object for input writeback: {hbm_name}")
        hbm_shape = tuple(hbm_object.get("shape", (self.program.mlen, self.program.mlen)))
        hbm_addr = int(hbm_object["base_addr"]) + int(hbm_offset)

        prev_hbm_name = value.residency.get("hbm_name")
        prev_hbm_addr = value.residency.get("hbm_addr")
        prev_hbm_offset = value.residency.get("hbm_offset")
        prev_hbm_stride = value.residency.get("hbm_stride")
        target_changed = (
            prev_hbm_name != str(hbm_name)
            or prev_hbm_addr != hbm_addr
            or prev_hbm_offset != hbm_offset
            or prev_hbm_stride != hbm_stride
        )

        # Preserve the current value contents before retargeting its HBM identity
        # to the destination input/output object. Otherwise a non-VRAM resident
        # value could be reloaded from the destination HBM slot instead of its
        # original backing.
        self.ensure_value_tile_in_place(value, "vram")
        writeback_value = value
        shared_tensor_refs = bool(self.value_tile_tensor_refs.get(value.value_tile_id))
        if shared_tensor_refs and target_changed:
            old_vram_addr = value.residency.pop("vram_addr", None)
            old_vram_name = value.residency.pop("vram_name", None)
            if old_vram_addr is None:
                raise RuntimeError(
                    f"shared writeback split requires VRAM residency, got {value.value_tile_id}"
                )
            writeback_value = ValueTile(
                value_tile_id=self._next_value_tile_id(),
                logical_shape=value.logical_shape,
                metadata=dict(value.metadata),
            )
            writeback_value.residency["vram_addr"] = int(old_vram_addr)
            if old_vram_name is not None:
                writeback_value.residency["vram_name"] = old_vram_name
            self.value_tiles[writeback_value.value_tile_id] = writeback_value
            self._value_tiles_in_vram.pop(value.value_tile_id, None)
            self._value_tiles_in_vram[writeback_value.value_tile_id] = int(old_vram_addr)

        writeback_value.residency["hbm_addr"] = hbm_addr
        writeback_value.residency["hbm_name"] = str(hbm_name)
        writeback_value.residency["hbm_offset"] = hbm_offset
        writeback_value.residency["hbm_stride"] = hbm_stride
        writeback_value.residency["hbm_scale_size"] = int(hbm_shape[0]) * int(hbm_shape[1])
        if target_changed:
            # A value may already be "hbm_ready" in a temporary spill object.
            # Final output writeback must retarget and actually store into the
            # destination input/output HBM object instead of early-returning.
            writeback_value.residency["hbm_ready"] = False
        self.move_tile(writeback_value, "vram", "hbm")
        writeback_value.residency["hbm_ready"] = True
        self._value_tiles_in_hbm[writeback_value.value_tile_id] = {
            "addr": writeback_value.residency.get("hbm_addr"),
            "name": writeback_value.residency.get("hbm_name"),
            "offset": writeback_value.residency.get("hbm_offset"),
            "stride": writeback_value.residency.get("hbm_stride"),
            "scale_size": writeback_value.residency.get("hbm_scale_size"),
        }
        if self._is_narrow_tensor_tile(dst_tile):
            self._rebind_view_group_value(dst_tile, writeback_value)
        else:
            self._bind_tile_pointer(dst_tile.tile_id, writeback_value.value_tile_id)
        writeback_value.metadata["input_writeback_tile_id"] = dst_tile.tile_id
        writeback_value.metadata["input_writeback_name"] = dst_tile.input_name
        self.program._record_operation_snapshot(
            "value_writeback",
            src_value=self._value_debug_state(original_value),
            writeback_value=self._value_debug_state(writeback_value),
            dst_tile=self._tile_debug_state(dst_tile),
            target_hbm={
                "hbm_name": str(hbm_name),
                "hbm_addr": hbm_addr,
                "hbm_offset": hbm_offset,
                "hbm_stride": hbm_stride,
                "hbm_scale_size": int(hbm_shape[0]) * int(hbm_shape[1]),
            },
            target_changed=target_changed,
            shared_tensor_refs=shared_tensor_refs,
        )

    def _detach_input_backing_identity(self, value: ValueTile) -> None:
        if not value.from_input_tile and value.source_input_tile_id is None:
            return
        # Keep the explicit HBM residency fields intact, but stop treating this
        # value as one logical alias of its original input tile in later fallback
        # HBM reconstruction paths.
        value.from_input_tile = False
        value.source_input_tile_id = None

    def _bind_value_to_tensor_tile(self, value: ValueTile, dst_tile: TensorTile) -> None:
        canonical_tile = self._resolve_alias_owner_tile(dst_tile)
        if isinstance(canonical_tile, TensorTile) and canonical_tile is not dst_tile and not self._is_narrow_tensor_tile(dst_tile):
            dst_tile = canonical_tile
        self._detach_input_backing_identity(value)
        if self._is_narrow_tensor_tile(dst_tile):
            self._rebind_view_group_value(dst_tile, value)
            return
        self._bind_tile_pointer(dst_tile.tile_id, value.value_tile_id)

    def _bind_tile_pointer(self, tile_id: str, value_tile_id: str) -> None:
        old_value_tile_id = self.full_tile_bindings.get(tile_id)
        if old_value_tile_id == value_tile_id:
            self.value_tile_tensor_refs.setdefault(value_tile_id, set()).add(tile_id)
            return
        if old_value_tile_id is not None:
            detached_old_value_tile_id = self._detach_tile_value_pointer(tile_id)
            self._attach_tile_value_pointer(tile_id, value_tile_id)
            if detached_old_value_tile_id is not None:
                self.free_value_tile(detached_old_value_tile_id)
            return
        self._attach_tile_value_pointer(tile_id, value_tile_id)

    def _attach_tile_value_pointer(self, tile_id: str, value_tile_id: str) -> None:
        self.full_tile_bindings[tile_id] = value_tile_id
        self.value_tile_tensor_refs.setdefault(value_tile_id, set()).add(tile_id)

    def _detach_tile_value_pointer(self, tile_id: str) -> Optional[str]:
        old_value_tile_id = self.full_tile_bindings.pop(tile_id, None)
        if old_value_tile_id is None:
            return None
        old_refs = self.value_tile_tensor_refs.get(old_value_tile_id)
        if old_refs is not None:
            old_refs.discard(tile_id)
            if not old_refs:
                self.value_tile_tensor_refs.pop(old_value_tile_id, None)
        return old_value_tile_id

    def _unbind_tile_value_pointer(self, tile_id: str) -> None:
        old_value_tile_id = self._detach_tile_value_pointer(tile_id)
        if old_value_tile_id is None:
            return
        self.free_value_tile(old_value_tile_id)

    def _is_input_backed_value_tile(self, value_tile_id: str) -> bool:
        value = self.value_tiles.get(value_tile_id)
        if value is not None and (value.from_input_tile or value.source_input_tile_id is not None):
            return True
        return any(ref_id in self.program.tensor_manager.input_tiles for ref_id in self.value_tile_tensor_refs.get(value_tile_id, set()))

    def _free_value_tile_vram_residency(self, value_tile_id: str) -> bool:
        value = self.value_tiles.get(value_tile_id)
        if value is None or self._is_protected_value_tile(value_tile_id, "vram"):
            return False
        vram_name = value.residency.pop("vram_name", None)
        value.residency.pop("vram_addr", None)
        self._value_tiles_in_vram.pop(value_tile_id, None)
        if vram_name is None:
            return False
        has_other_live_owner = any(
            other_id != value_tile_id and other.residency.get("vram_name") == vram_name
            for other_id, other in self.value_tiles.items()
        )
        if not has_other_live_owner:
            self.program.compiler.sub_matrix_manager.vram_allocator.free(str(vram_name), strict=False)
        return True

    def _non_input_value_refs(self, value_tile_id: str) -> List[str]:
        return sorted(
            ref_id
            for ref_id in self.value_tile_tensor_refs.get(value_tile_id, set())
            if ref_id not in self.program.tensor_manager.input_tiles
        )

    def free_tensor_tile(self, tile: TensorTile, *, weak: Optional[bool] = None) -> Optional[str]:
        if isinstance(tile, VectorTile):
            raise TypeError("free_tensor_tile only supports TensorTile; VectorTile uses FPFragment backing")
        value_tile_id = self.full_tile_bindings.get(tile.tile_id)
        if value_tile_id is None:
            return None
        if weak:
            self._detach_tile_value_pointer(tile.tile_id)
            self.program._record_operation_snapshot(
                "free_tensor_tile",
                mode="weak",
                tile=self._tile_debug_state(tile),
                value_tile_id=value_tile_id,
            )
            return value_tile_id

        if weak is None:
            detached_tile_ids = [tile.tile_id]
            self._detach_tile_value_pointer(tile.tile_id)
            released_vram = False
            if not self._non_input_value_refs(value_tile_id):
                if self._is_input_backed_value_tile(value_tile_id):
                    released_vram = self._free_value_tile_vram_residency(value_tile_id)
                else:
                    self.free_value_tile(value_tile_id)
                    released_vram = True
            self.program._record_operation_snapshot(
                "free_tensor_tile",
                mode="auto",
                tile=self._tile_debug_state(tile),
                value_tile_id=value_tile_id,
                detached_tile_ids=detached_tile_ids,
                released_vram=released_vram,
            )
            return value_tile_id

        ref_tile_ids = sorted(self.value_tile_tensor_refs.get(value_tile_id, set()))
        detach_tile_ids = ref_tile_ids
        input_backed = self._is_input_backed_value_tile(value_tile_id)
        if input_backed:
            detach_tile_ids = [
                ref_tile_id
                for ref_tile_id in ref_tile_ids
                if ref_tile_id not in self.program.tensor_manager.input_tiles
            ]
        for ref_tile_id in detach_tile_ids:
            self._detach_tile_value_pointer(ref_tile_id)
        self.narrow_group_bindings = {
            group_key: bound_value_tile_id
            for group_key, bound_value_tile_id in self.narrow_group_bindings.items()
            if bound_value_tile_id != value_tile_id
        }
        released_vram = False
        if input_backed:
            released_vram = self._free_value_tile_vram_residency(value_tile_id)
        else:
            self.free_value_tile(value_tile_id)
            released_vram = True
        self.program._record_operation_snapshot(
            "free_tensor_tile",
            mode="strong",
            tile=self._tile_debug_state(tile),
            value_tile_id=value_tile_id,
            detached_tile_ids=detach_tile_ids,
            preserved_input_tile_ids=[ref_id for ref_id in ref_tile_ids if ref_id not in detach_tile_ids],
            released_vram=released_vram,
        )
        return value_tile_id

    def free_value_tile(self, value_tile_id: str) -> None:
        value = self.value_tiles.get(value_tile_id)
        if value is None:
            return
        if self.value_tile_tensor_refs.get(value_tile_id):
            return
        if self._is_protected_value_tile(value_tile_id, "vram"):
            return
        vram_name = value.residency.pop("vram_name", None)
        if vram_name is not None:
            has_other_live_owner = any(
                other_id != value_tile_id and other.residency.get("vram_name") == vram_name
                for other_id, other in self.value_tiles.items()
            )
            if not has_other_live_owner:
                self.program.compiler.sub_matrix_manager.vram_allocator.free(str(vram_name), strict=False)
        mram_name = value.residency.pop("mram_name", None)
        if mram_name is not None:
            self.program.compiler.sub_matrix_manager.mram_allocator.free(str(mram_name), strict=False)
        value.residency.pop("vram_addr", None)
        value.residency.pop("mram_addr", None)
        self._value_tiles_in_vram.pop(value_tile_id, None)
        self._value_tiles_in_mram.pop(value_tile_id, None)
        self._value_tiles_in_hbm.pop(value_tile_id, None)
        self._mram_fifo[:] = [item for item in self._mram_fifo if item != value_tile_id]
        self.narrow_group_bindings = {
            group_key: bound_value_tile_id
            for group_key, bound_value_tile_id in self.narrow_group_bindings.items()
            if bound_value_tile_id != value_tile_id
        }
        self.value_tiles.pop(value_tile_id, None)


class TensorManager:
    """Manage logical tensors, tiles, slices, and tensor-thread grouping.

    TensorManager operates on logical objects only. It owns shape flattening,
    tile metadata, slice resolution, and `mapt` grouping. It deliberately does
    not create ValueTile / ValueTileView objects and does not decide
    residency placement; that work stays in ValueManager.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.inputs: Dict[str, Input] = {}
        self.tensors: Dict[str, Tensor] = {}
        self.vectors: Dict[str, Vector] = {}
        self.fp_fragments: Dict[str, FPFragment] = {}
        self.input_tiles: Dict[str, InputTile] = {}
        self.tensor_tiles: Dict[str, TensorTile] = {}
        self.vector_tiles: Dict[str, VectorTile] = {}
        self._input_tile_counter = 0
        self._tensor_tile_counter = 0
        # FPVar management: one FP_MEM slot per scalar constant.
        # _fp_mem_values is ordered by address so build_fp_preload can
        # return the initialisation array directly.
        # Addresses [0, 32) are reserved for system/hardware constants;
        # user fp_var() declarations start at address 32.
        self.fp_vars: Dict[str, FPVar] = {}
        self._fp_mem_values: List[float] = [0.0] * 32
        self._next_fp_mem_addr: int = 32
        self._literal_fp_vars: Dict[Tuple[str, float], FPVar] = {}

    def fp_var(self, name: str, value: float = 0.0, size: int = 1) -> FPVar | FPFragment:
        """Allocate FP-domain storage.

        The new default rule is that one FPVar represents one scalar slot.
        For compatibility, requesting size > 1 returns one FPFragment whose
        cells are backed by one scalar FPVar each.

        Usage:
            scale = program.fp_var("scale", value=1.0 / math.sqrt(dim))
        """
        if size <= 0:
            raise ValueError(f"FP allocation size must be positive, got {size}")
        if size != 1:
            fragment = self.fp_fragment(name=name, shape=(int(size),), init=value)
            return fragment
        if name in self.fp_vars:
            raise ValueError(f"FPVar {name!r} already declared")
        addr = self._next_fp_mem_addr
        self._next_fp_mem_addr += 1
        var = FPVar(name=name, fp_mem_addr=addr)
        self.fp_vars[name] = var
        self._fp_mem_values.append(float(value))
        return var

    def fp_fragment(
        self,
        name: str,
        shape: Tuple[int, ...] | int,
        *,
        init: float = 0.0,
        dtype: str = "fp32",
    ) -> FPFragment:
        if isinstance(shape, int):
            shape = (shape,)
        normalized_shape = tuple(int(dim) for dim in shape)
        if not normalized_shape or any(dim <= 0 for dim in normalized_shape):
            raise ValueError(f"FPFragment shape must contain positive extents, got {shape}")
        if name in self.fp_fragments or name in self.fp_vars:
            raise ValueError(f"FPFragment {name!r} already declared")

        fragment = FPFragment(program=self.program, name=name, shape=normalized_shape, dtype=dtype)
        for index in _iter_fp_indices(normalized_shape):
            cell_name = f"{name}{_format_fp_index(index)}"
            fragment.vars[index] = self.fp_var(cell_name, value=init, size=1)  # type: ignore[assignment]

        self.fp_fragments[name] = fragment
        return fragment

    def alloc_fragment(
        self,
        name: str,
        logical_shape: LogicalShape,
        *,
        init_zero: bool = False,
        dtype: str = "fp32",
    ) -> Tensor | Vector:
        if len(logical_shape) == 4:
            tensor = self.tensor(name, logical_shape)
            tensor.metadata["fragment_kind"] = "tensor"
            tensor.metadata["dtype"] = dtype
            tensor.metadata["init_zero"] = bool(init_zero)
            return tensor
        if len(logical_shape) == 3:
            vector = self.vector(name, logical_shape)
            vector.metadata["fragment_kind"] = "vector"
            vector.metadata["dtype"] = dtype
            vector.metadata["init_zero"] = bool(init_zero)
            return vector
        raise NotImplementedError(
            f"alloc_fragment supports 4D tensor fragments and 3D vector fragments only, got {logical_shape}"
        )

    def mapf(self, operand: object) -> List[FPVar]:
        if isinstance(operand, (int, float)):
            literal_value = float(operand)
            key = ("fp32", literal_value)
            literal_var = self._literal_fp_vars.get(key)
            if literal_var is None:
                literal_name = self.program._auto_name("fp_literal")
                created = self.fp_var(literal_name, value=literal_value, size=1)
                if not isinstance(created, FPVar):
                    raise RuntimeError("literal fp allocation expected one FPVar")
                literal_var = created
                self._literal_fp_vars[key] = literal_var
            return [literal_var]
        if isinstance(operand, FPVar):
            return [operand]
        if isinstance(operand, FPFragment):
            return [operand.vars[index] for index in _iter_fp_indices(operand.shape)]
        if isinstance(operand, FPFragmentSlice):
            return self._resolve_fp_fragment_slice(operand.base, operand.selectors)
        if isinstance(operand, Vector):
            return self._resolve_vector_fp_vars(operand)
        if isinstance(operand, VectorSlice):
            return self._resolve_vector_slice_fp_vars(operand)
        if isinstance(operand, VectorTile):
            return self._resolve_vector_tile_fp_vars(operand)
        if isinstance(operand, ElementRef):
            return [self._resolve_element_fpvar(operand)]
        if isinstance(operand, (list, tuple)):
            resolved: List[FPVar] = []
            for item in operand:
                resolved.extend(self.mapf(item))
            return resolved
        raise NotImplementedError(f"Unsupported operand for mapf: {type(operand).__name__}")

    def mapf_dst(self, operand: object, *, control: str, src1_vars: Optional[Sequence[FPVar]] = None) -> List[FPVar]:
        if isinstance(operand, (list, tuple)):
            resolved: List[FPVar] = []
            for item in operand:
                resolved.extend(self.mapf_dst(item, control=control, src1_vars=src1_vars))
            return resolved
        return self.mapf(operand)

    def _resolve_vector_fp_vars(self, vector: Vector) -> List[FPVar]:
        resolved: List[FPVar] = []
        for logical_index in _iter_logical_indices(vector.logical_shape):
            resolved.append(self._resolve_element_fpvar(ElementRef(base=vector, indices=logical_index)))
        return resolved

    def _resolve_vector_slice_fp_vars(self, vector_slice: VectorSlice) -> List[FPVar]:
        resolved: List[FPVar] = []
        for logical_index in _iter_selected_logical_indices(vector_slice.base.logical_shape, vector_slice.selectors):
            resolved.append(self._resolve_element_fpvar(ElementRef(base=vector_slice.base, indices=logical_index)))
        return resolved

    def _resolve_vector_tile_fp_vars(self, tile: VectorTile) -> List[FPVar]:
        fragment = self.program.value_manager.resolve_fp_fragment(tile)
        row_groups = _vector_tile_row_fp_groups(
            src_tile=tile,
            fragment=fragment,
            mlen=self.program.mlen,
            btmm_hlen=self.program.btmm_hlen,
            src_slice_ranges=None,
        )
        return [fp_var for row in row_groups for fp_var in row]

    def _resolve_element_operand_context(
        self,
        operand: ElementRef,
    ) -> Tuple[object, Tuple[int, ...], TileLike, int, int]:
        base = operand.base
        logical_shape = tuple(getattr(base, "logical_shape", ()))
        if not logical_shape:
            raise RuntimeError(f"ElementRef base {type(base).__name__} does not expose logical_shape")
        if len(operand.indices) != len(logical_shape):
            raise RuntimeError(
                f"ElementRef expected {len(logical_shape)} indices for {type(base).__name__}, got {len(operand.indices)}"
            )

        normalized_indices = tuple(_normalize_index(index, extent) for index, extent in zip(operand.indices, logical_shape))
        physical_row, physical_col = _logical_indices_to_physical_coord(logical_shape, normalized_indices)
        tile_coord = (physical_row // self.program.mlen, physical_col // self.program.mlen)
        tile_col_start = tile_coord[1] * self.program.mlen
        tile_row_start = tile_coord[0] * self.program.mlen

        tiles = getattr(base, "tiles", None)
        if not isinstance(tiles, dict):
            raise RuntimeError(f"ElementRef base {type(base).__name__} does not expose tiles")
        tile = tiles.get(tile_coord)
        if not isinstance(tile, (TensorTile, InputTile, VectorTile)):
            raise RuntimeError(
                f"ElementRef {getattr(base, 'name', type(base).__name__)}{normalized_indices} "
                f"did not resolve to one tile at coord={tile_coord}"
            )
        return (
            base,
            normalized_indices,
            tile,
            int(physical_row - tile_row_start),
            int(physical_col - tile_col_start),
        )

    def _ensure_element_tile_fp_fragment(
        self,
        *,
        base: object,
        normalized_indices: Tuple[int, ...],
        tile: TensorTile | InputTile,
    ) -> FPFragment:
        backing_value = self.program.value_manager.resolve_value_tile(tile)
        if backing_value.residency.get("fpram_ready"):
            return self.program.value_manager._resolve_value_fp_fragment(backing_value)

        has_materialized_storage = any(
            backing_value.residency.get(key) is not None
            for key in ("vram_addr", "mram_addr", "hbm_addr")
        ) or bool(backing_value.residency.get("hbm_ready"))
        if has_materialized_storage:
            raise RuntimeError(
                "ElementRef write requires one FP-backed tile before mutating materialized tensor storage; "
                f"tile={tile.tile_id} base={getattr(base, 'name', type(base).__name__)} indices={normalized_indices}"
            )

        fragment_name = self.program._auto_name(f"{getattr(base, 'name', 'tensor')}.element_fp_tile")
        zero_var = self.mapf(0.0)[0]
        fragment = FPFragment(
            program=self.program,
            name=fragment_name,
            shape=tile.tile_shape,
            dtype="fp32",
        )
        for fp_index in _iter_fp_indices(tile.tile_shape):
            fragment.vars[fp_index] = zero_var
        self.fp_fragments[fragment_name] = fragment
        self.program.create_value_tile_in_fpram(
            tile,
            fragment,
            bind=True,
            metadata={
                "element_ref_direct_backing": True,
                "source_tensor": getattr(base, "name", type(base).__name__),
                "source_tile_id": tile.tile_id,
            },
        )
        return fragment

    def _element_fragment_and_index(
        self,
        operand: ElementRef,
        *,
        ensure_write_backing: bool = False,
    ) -> Tuple[FPFragment, FPIndex, object, Tuple[int, ...], TileLike]:
        base, normalized_indices, tile, local_row, local_col = self._resolve_element_operand_context(operand)
        if isinstance(tile, VectorTile):
            fragment = self.program.value_manager.resolve_fp_fragment(tile)
        else:
            backing_value = self.program.value_manager.resolve_value_tile(tile)
            if ensure_write_backing:
                fragment = self._ensure_element_tile_fp_fragment(
                    base=base,
                    normalized_indices=normalized_indices,
                    tile=tile,
                )
            elif not backing_value.residency.get("fpram_ready"):
                raise RuntimeError(
                    f"ElementRef {getattr(base, 'name', type(base).__name__)}{normalized_indices} requires one fpram-backed "
                    f"value tile; backing value {backing_value.value_tile_id} is no longer resident in fpram"
                )
            else:
                fragment = self.program.value_manager._resolve_value_fp_fragment(backing_value)
        fp_index = _physical_tile_coord_to_fp_index(
            fragment.shape,
            local_row=local_row,
            local_col=local_col,
            mlen=self.program.mlen,
            btmm_hlen=self.program.btmm_hlen,
        )
        return fragment, fp_index, base, normalized_indices, tile

    def _resolve_element_fpvar(self, operand: ElementRef, *, create_for_write: bool = False) -> FPVar:
        fragment, fp_index, base, normalized_indices, _tile = self._element_fragment_and_index(
            operand,
            ensure_write_backing=create_for_write,
        )
        fp_var = fragment.vars.get(fp_index)
        if not isinstance(fp_var, FPVar):
            raise RuntimeError(
                f"ElementRef {getattr(base, 'name', type(base).__name__)}{normalized_indices} resolved to missing fp cell {fp_index}"
            )
        return fp_var

    def bind_element_pointer(self, operand: ElementRef, fp_var: FPVar, *, mode: str = "alias") -> FPVar:
        fragment, fp_index, base, normalized_indices, tile = self._element_fragment_and_index(
            operand,
            ensure_write_backing=True,
        )
        fragment.vars[fp_index] = fp_var
        return fp_var

    def allocate_element_result_fpvar(self, operand: ElementRef) -> FPVar:
        _fragment, _fp_index, base, normalized_indices, tile = self._element_fragment_and_index(
            operand,
            ensure_write_backing=True,
        )
        created = self.fp_var(
            self.program._auto_name(f"{getattr(base, 'name', 'tensor')}.element_fp"),
            value=0.0,
            size=1,
        )
        if not isinstance(created, FPVar):
            raise RuntimeError("ElementRef result allocation expected one scalar FPVar")
        return created

    def mapf_t(self, tensor_operand: object, fp_operand: object, *, control: str = "mixed") -> Dict[str, object]:
        tensor_tiles = self.mapt([tensor_operand, 0]) if tensor_operand is not None else []
        fp_vars = self.mapf(fp_operand)
        packet = {
            "control": control,
            "tensor_operand": tensor_operand,
            "tensor_groups": tensor_tiles,
            "fp_operand": fp_operand,
            "fp_vars": fp_vars,
        }
        return packet

    def _resolve_fp_fragment_slice(
        self,
        fragment: FPFragment,
        selectors: Tuple[SliceItem, ...],
    ) -> List[FPVar]:
        normalized = list(selectors) + [slice(None)] * max(0, len(fragment.shape) - len(selectors))
        selected_indices: List[FPIndex] = []
        for index in _iter_fp_indices(fragment.shape):
            keep = True
            for dim_idx, selector in enumerate(normalized[: len(fragment.shape)]):
                start, stop = _slice_item_to_range(selector, fragment.shape[dim_idx])
                if index[dim_idx] < start or index[dim_idx] >= stop:
                    keep = False
                    break
            if keep:
                selected_indices.append(index)
        return [fragment.vars[index] for index in selected_indices]

    def _next_input_tile_id(self) -> str:
        tile_id = f"input_tile.{self._input_tile_counter}"
        self._input_tile_counter += 1
        return tile_id

    def _next_tensor_tile_id(self) -> str:
        tile_id = f"tensor_tile.{self._tensor_tile_counter}"
        self._tensor_tile_counter += 1
        return tile_id

    def create_input_tiles(self, input_name: str, logical_shape: LogicalShape) -> Dict[TileCoord, InputTile]:
        rows, cols = _logical_shape_to_physical_shape(logical_shape)
        row_blocks = ceil(rows / self.program.mlen)
        col_blocks = ceil(cols / self.program.mlen)
        tiles: Dict[TileCoord, InputTile] = {}
        for row_block in range(row_blocks):
            for col_block in range(col_blocks):
                row_count = min(self.program.mlen, rows - row_block * self.program.mlen)
                col_count = min(self.program.mlen, cols - col_block * self.program.mlen)
                input_tile = InputTile(
                    tile_id=self._next_input_tile_id(),
                    input_name=input_name,
                    coord=(row_block, col_block),
                    tile_shape=(row_count, col_count),
                    metadata=self._build_tile_metadata(logical_shape, row_block, col_block, row_count, col_count),
                )
                tiles[(row_block, col_block)] = input_tile
                self.input_tiles[input_tile.tile_id] = input_tile
        return tiles

    def create_tensor_tiles(self, tensor_name: str, logical_shape: LogicalShape) -> Dict[TileCoord, TensorTile]:
        rows, cols = _logical_shape_to_physical_shape(logical_shape)
        row_blocks = ceil(rows / self.program.mlen)
        col_blocks = ceil(cols / self.program.mlen)
        tiles: Dict[TileCoord, TensorTile] = {}
        for row_block in range(row_blocks):
            for col_block in range(col_blocks):
                row_count = min(self.program.mlen, rows - row_block * self.program.mlen)
                col_count = min(self.program.mlen, cols - col_block * self.program.mlen)
                tensor_tile = TensorTile(
                    tile_id=self._next_tensor_tile_id(),
                    tensor_name=tensor_name,
                    coord=(row_block, col_block),
                    tile_shape=(row_count, col_count),
                    metadata=self._build_tile_metadata(logical_shape, row_block, col_block, row_count, col_count),
                )
                tiles[(row_block, col_block)] = tensor_tile
                self.tensor_tiles[tensor_tile.tile_id] = tensor_tile
        return tiles

    def create_vector_tiles(self, vector_name: str, logical_shape: LogicalShape) -> Dict[TileCoord, VectorTile]:
        rows, cols = _logical_shape_to_physical_shape(logical_shape)
        row_blocks = ceil(rows / self.program.mlen)
        col_blocks = ceil(cols / self.program.mlen)
        tiles: Dict[TileCoord, VectorTile] = {}
        for row_block in range(row_blocks):
            for col_block in range(col_blocks):
                row_count = min(self.program.mlen, rows - row_block * self.program.mlen)
                col_count = min(self.program.mlen, cols - col_block * self.program.mlen)
                vector_tile = VectorTile(
                    tile_id=self._next_tensor_tile_id(),
                    tensor_name=vector_name,
                    coord=(row_block, col_block),
                    tile_shape=(row_count, col_count),
                    metadata=self._build_tile_metadata(logical_shape, row_block, col_block, row_count, col_count),
                )
                tiles[(row_block, col_block)] = vector_tile
                self.vector_tiles[vector_tile.tile_id] = vector_tile
                self.tensor_tiles[vector_tile.tile_id] = vector_tile
        return tiles

    def _build_tile_metadata(
        self,
        logical_shape: LogicalShape,
        row_block: int,
        col_block: int,
        row_count: int,
        col_count: int,
    ) -> Dict[str, object]:
        """Build per-tile logical metadata used by later grouping/mapping stages.

        For 4D BSHD tensors, the current convention treats one physical tile as
        one logical window over flattened `(seq, head * head_dim)` storage.
        When `head_dim < mlen`, one physical tile may pack multiple adjacent
        heads. The metadata below records both views:

        - per-head view: `head_index`, `head_col_offset`, `d_tile_index`
        - packed-group view: `group_head_start`, `packed_head_count`
        - scatter layout view: `grouped_narrow`, `packed_head_group`,
          `scatter_slot_width`

        Downstream `mapt_head_group`, scatter-group matmul, and group-head
        elementwise paths all rely on these fields instead of re-deriving the
        packing rules independently.
        """
        metadata: Dict[str, object] = {
            "mlen": self.program.mlen,
            "logical_shape": logical_shape,
            "row_block": row_block,
            "col_block": col_block,
            "row_count": row_count,
            "col_count": col_count,
            "tile_width_class": "narrow" if int(col_count) < int(self.program.mlen) else "full",
        }
        if len(logical_shape) == 4:
            b, s, h, d = logical_shape
            if int(b) > 1 and int(s) % int(self.program.mlen) != 0:
                raise ValueError(
                    f"BSHD tensors with batch>1 require S to be a multiple of mlen={self.program.mlen}; "
                    f"got shape={logical_shape}"
                )
            row_blocks_per_batch = (
                max(1, int(s) // int(self.program.mlen))
                if int(s) % int(self.program.mlen) == 0
                else max(1, ceil(int(s) / int(self.program.mlen)))
            )
            batch_index = int(row_block) // row_blocks_per_batch
            seq_block = int(row_block) % row_blocks_per_batch
            seq_start = seq_block * int(self.program.mlen)
            seq_end = min(int(s), seq_start + int(row_count))
            physical_col_start = col_block * self.program.mlen
            head_index = physical_col_start // d if d > 0 else 0
            head_col_offset = physical_col_start % d if d > 0 else 0
            grouped_narrow = d > 0 and d < self.program.mlen
            packed_head_count = min(max(self.program.mlen // d, 1), max(h - head_index, 0)) if grouped_narrow else 1
            metadata.update(
                {
                    "layout": "bshd",
                    "batch": b,
                    "seq": s,
                    "heads": h,
                    "head_dim": d,
                    "batch_index": batch_index,
                    "seq_block": seq_block,
                    "seq_start": seq_start,
                    "seq_end": seq_end,
                    "row_blocks_per_batch": row_blocks_per_batch,
                    "head_index": head_index,
                    "head_col_offset": head_col_offset,
                    "d_tile_index": head_col_offset // self.program.mlen if self.program.mlen > 0 else 0,
                    "grouped_narrow": grouped_narrow,
                    "packed_head_group": grouped_narrow,
                    "tile_width_class": "narrow" if grouped_narrow or int(col_count) < int(self.program.mlen) else "full",
                    "group_head_start": head_index,
                    "packed_head_count": packed_head_count,
                    "scatter_slot_width": d if grouped_narrow else col_count,
                }
            )
        elif len(logical_shape) == 3:
            x, y, z = logical_shape
            metadata.update(
                {
                    "layout": "vector3d",
                    "vector_extents": (x, y, z),
                    "vector_row_dim": x,
                    "vector_col_dims": (y, z),
                }
            )
        else:
            metadata["layout"] = "2d"
        return metadata

    def input(self, name: str, logical_shape: LogicalShape, *, hbm_addr: Optional[int] = None) -> Input:
        physical_shape = _logical_shape_to_physical_shape(logical_shape)
        hbm_group_name = f"{name}.hbm"
        if hbm_group_name not in self.program.hardware.hbm_objects:
            self.program.add_hbm_object(hbm_group_name, physical_shape, hbm_addr=hbm_addr)
        input_obj = Input(program=self.program, name=name, logical_shape=logical_shape)
        input_obj.metadata["hbm_group_obj"] = hbm_group_name
        self.inputs[name] = input_obj
        return input_obj

    def tensor(self, name: str, logical_shape: LogicalShape) -> Tensor | Vector:
        if len(logical_shape) == 3:
            return self.vector(name, logical_shape)
        tensor = Tensor(program=self.program, name=name, logical_shape=logical_shape)
        self.tensors[name] = tensor
        return tensor

    def vector(self, name: str, logical_shape: LogicalShape) -> Vector:
        if len(logical_shape) != 3:
            raise ValueError(f"vector expects one 3D logical shape, got {logical_shape}")
        vector = Vector(program=self.program, name=name, logical_shape=logical_shape)
        self.vectors[name] = vector
        return vector

    def mapt(self, signal: List[object]) -> List[object]:
        """Group logical tensor tiles into per-thread compute packets.

        `mapt` is the logical staging step before value resolution. Depending
        on the control mode, it can:

        - enumerate tiles directly for copy / elementwise paths
        - build BSHD matmul groups
        - build head-group packets for grouped-narrow tensors
        - build BTMM/QKT-specific thread packets

        The output is intentionally still a tensor-layer structure. Value/scatter
        objects are resolved later by `mapv`, not here.
        """
        if len(signal) == 2:
            operand, control = signal
            if control == 0:
                resolved_tiles = self._resolve_tiles_from_operand(operand)
                return [[tile] for tile in resolved_tiles]
            if control == "head_group":
                return self.mapt_head_group(operand)
            resolved_tiles = self._resolve_tiles_from_operand(operand)
            raise NotImplementedError(f"Basic mapt resolve does not support control={control!r}")

        src1, src2, dst, control = signal
        if control not in (0, 1):
            raise NotImplementedError(f"Unsupported mapt control: {control}")
        if (
            len(getattr(src1, "logical_shape", ())) == 4
            and len(getattr(src2, "logical_shape", ())) == 4
            and len(getattr(dst, "logical_shape", ())) == 4
        ):
            if control == 1:
                return self.mapt_btmm_head_group_qkt(src1, src2, dst)  # type: ignore[return-value]
            return self._mapt_bshd_matmul_groups(src1, src2, dst)

        src1_tiles = _tiles_in_grid_order(src1.tiles)
        src2_tiles = _tiles_in_grid_order(src2.tiles)
        dst_tiles = _tiles_in_grid_order(dst.tiles)
        groups: List[List[object]] = []
        for dst_tile in dst_tiles:
            lhs_group = [tile for tile in src1_tiles if tile.coord[0] == dst_tile.coord[0]]
            rhs_group = [tile for tile in src2_tiles if tile.coord[1] == dst_tile.coord[1]]
            groups.append([*lhs_group, *rhs_group, dst_tile])
        return groups

    def mapt_head_group(self, operand: object) -> List[Dict[str, object]]:
        resolved_tiles = self._resolve_tiles_from_operand(operand)
        if not resolved_tiles:
            return []
        if not all(isinstance(tile, (TensorTile, InputTile, VectorTile)) for tile in resolved_tiles):
            raise RuntimeError("mapt_head_group expects tile operands only")

        first_tile = resolved_tiles[0]
        logical_shape = getattr(getattr(operand, "base", operand), "logical_shape", ())
        if len(logical_shape) != 4:
            return [
                {
                    "control": "head_group",
                    "tiles": [tile],
                    "row_block": int(tile.metadata.get("row_block", tile.coord[0])),
                    "group_start": int(tile.metadata.get("head_index", 0)),
                    "group_heads": 1,
                    "lane_heads": [int(tile.metadata.get("head_index", 0))],
                    "group_key": (
                        int(tile.metadata.get("row_block", tile.coord[0])),
                        int(tile.metadata.get("head_index", 0)),
                    ),
                }
                for tile in resolved_tiles
            ]

        groups: Dict[Tuple[int, int], Dict[str, object]] = {}
        for tile in resolved_tiles:
            row_block = int(tile.metadata.get("row_block", tile.coord[0]))
            group_start = int(tile.metadata.get("group_head_start", tile.metadata.get("head_index", 0)))
            packed_head_count = int(tile.metadata.get("packed_head_count", 1))
            lane_heads = [int(tile.metadata.get("head_index", 0))]
            if packed_head_count > 1:
                lane_heads = [group_start + lane for lane in range(packed_head_count)]
            group_key = (row_block, group_start)
            packet = groups.get(group_key)
            if packet is None:
                packet = {
                    "control": "head_group",
                    "tiles": [],
                    "row_block": row_block,
                    "group_start": group_start,
                    "group_heads": 0,
                    "lane_heads": [],
                    "group_key": group_key,
                }
                groups[group_key] = packet
            packet["tiles"].append(tile)
            existing_heads = set(packet["lane_heads"])
            for head in lane_heads:
                if head not in existing_heads:
                    packet["lane_heads"].append(head)
                    existing_heads.add(head)
            packet["group_heads"] = len(packet["lane_heads"])

        packets = list(groups.values())
        packets.sort(key=lambda item: (int(item["row_block"]), int(item["group_start"])))
        return packets

    def _mapt_bshd_matmul_groups(self, src1: object, src2: object, dst: object) -> List[List[object]]:
        src1_shape = tuple(getattr(src1, "logical_shape", ()))
        src2_shape = tuple(getattr(src2, "logical_shape", ()))
        dst_shape = tuple(getattr(dst, "logical_shape", ()))
        if src1_shape[0] != src2_shape[0] or src1_shape[0] != dst_shape[0]:
            raise ValueError(
                f"BSHD matmul requires matched batch size, got src1={src1_shape[0]} "
                f"src2={src2_shape[0]} dst={dst_shape[0]}"
            )
        src1_tiles = _tiles_in_grid_order(src1.tiles)
        src2_tiles = _tiles_in_grid_order(src2.tiles)
        dst_tiles = _tiles_in_grid_order(dst.tiles)
        src1_by_batch_head_seq_k: Dict[Tuple[int, int, int, int], object] = {}
        src2_by_batch_head_k_col: Dict[Tuple[int, int, int, int], object] = {}
        groups: List[List[object]] = []

        for tile in src1_tiles:
            batch_index = _bshd_tile_batch_index(tile)
            head_index = int(tile.metadata.get("head_index", 0))
            seq_block = _bshd_tile_seq_block(tile)
            k_index = int(tile.metadata.get("d_tile_index", tile.coord[1]))
            src1_by_batch_head_seq_k[(batch_index, head_index, seq_block, k_index)] = tile

        for tile in src2_tiles:
            batch_index = _bshd_tile_batch_index(tile)
            head_index = int(tile.metadata.get("head_index", 0))
            k_index = _bshd_tile_seq_block(tile)
            d_tile_index = int(tile.metadata.get("d_tile_index", 0))
            src2_by_batch_head_k_col[(batch_index, head_index, k_index, d_tile_index)] = tile

        for dst_tile in dst_tiles:
            batch_index = _bshd_tile_batch_index(dst_tile)
            head_index = int(dst_tile.metadata.get("head_index", 0))
            seq_block = _bshd_tile_seq_block(dst_tile)
            d_tile_index = int(dst_tile.metadata.get("d_tile_index", 0))
            lhs_candidates = [
                key
                for key in src1_by_batch_head_seq_k.keys()
                if key[0] == batch_index and key[1] == head_index and key[2] == seq_block
            ]
            k_values = sorted(key[3] for key in lhs_candidates)
            group: List[object] = []
            for k_index in k_values:
                lhs_tile = src1_by_batch_head_seq_k.get((batch_index, head_index, seq_block, k_index))
                rhs_tile = src2_by_batch_head_k_col.get((batch_index, head_index, k_index, d_tile_index))
                if lhs_tile is None or rhs_tile is None:
                    continue
                group.append([lhs_tile, rhs_tile])
            group.append([dst_tile])
            groups.append(group)
        return groups

    def mapt_btmm_head_group_qkt(
        self,
        src1: object,
        src2: object,
        dst: object,
    ) -> List[BTMMHeadGroupThread]:
        if not (
            len(getattr(src1, "logical_shape", ())) == 4
            and len(getattr(src2, "logical_shape", ())) == 4
            and len(getattr(dst, "logical_shape", ())) == 4
        ):
            raise NotImplementedError("mapt_btmm_head_group_qkt currently supports BSHD tensors only")

        src1_batch, src1_seq, src1_heads, src1_dim = getattr(src1, "logical_shape")
        src2_batch, src2_seq, src2_heads, src2_dim = getattr(src2, "logical_shape")
        dst_batch, dst_seq, dst_heads, dst_dim = getattr(dst, "logical_shape")
        if src1_batch != src2_batch or src1_batch != dst_batch:
            raise ValueError(
                f"BTMM QKT mapt requires matched batch size, got src1={src1_batch} "
                f"src2={src2_batch} dst={dst_batch}"
            )
        if src1_heads != src2_heads or src1_heads != dst_heads:
            raise ValueError(
                f"BTMM QKT mapt requires matched head count, got src1={src1_heads} src2={src2_heads} dst={dst_heads}"
            )
        if src1_dim != self.program.btmm_hlen or src2_dim != self.program.btmm_hlen:
            raise ValueError(
                f"BTMM QKT mapt requires src1/src2 head_dim == btmm_hlen == {self.program.btmm_hlen}, "
                f"got src1={src1_dim} src2={src2_dim}"
            )
        if src1_seq != dst_seq:
            raise ValueError(f"BTMM QKT mapt requires dst seq to match src1 seq, got dst={dst_seq} src1={src1_seq}")
        if src2_seq != dst_dim:
            raise ValueError(
                f"BTMM QKT mapt requires dst last dim to match src2 seq, got dst={dst_dim} src2={src2_seq}"
            )
        if dst_dim % self.program.mlen != 0:
            raise ValueError(
                f"BTMM QKT mapt requires dst last dim multiple of mlen={self.program.mlen}, got {dst_dim}"
            )

        lhs_tiles = _tiles_in_grid_order(src1.tiles)
        rhs_tiles = _tiles_in_grid_order(src2.tiles)
        dst_tiles = _tiles_in_grid_order(dst.tiles)
        lhs_groups: Dict[Tuple[int, int, int], TileLike] = {}
        rhs_groups: Dict[Tuple[int, int, int], TileLike] = {}
        dst_by_key: Dict[Tuple[int, int, int, int], TileLike] = {}
        threads: List[BTMMHeadGroupThread] = []

        for tile in lhs_tiles:
            batch_index = _bshd_tile_batch_index(tile)
            seq_block = _bshd_tile_seq_block(tile)
            group_block = int(tile.coord[1])
            lhs_groups[(batch_index, seq_block, group_block)] = tile

        for tile in rhs_tiles:
            batch_index = _bshd_tile_batch_index(tile)
            seq_block = _bshd_tile_seq_block(tile)
            group_block = int(tile.coord[1])
            rhs_groups[(batch_index, seq_block, group_block)] = tile

        dst_col_blocks_per_head = dst_dim // self.program.mlen
        for tile in dst_tiles:
            batch_index = _bshd_tile_batch_index(tile)
            seq_block = _bshd_tile_seq_block(tile)
            head_index = int(tile.metadata.get("head_index", 0))
            rhs_row_block = int(tile.coord[1]) - head_index * dst_col_blocks_per_head
            dst_by_key[(batch_index, seq_block, rhs_row_block, head_index)] = tile

        group_heads = self.program.btmm_lane_count
        q_row_blocks = max(1, ceil(src1_seq / self.program.mlen))
        k_row_blocks = max(1, ceil(src2_seq / self.program.mlen))
        group_blocks = max(1, ceil(src1_heads / group_heads))

        for batch_index in range(int(src1_batch)):
            for lhs_row_block in range(q_row_blocks):
                for rhs_row_block in range(k_row_blocks):
                    for group_block in range(group_blocks):
                        lhs_tile = lhs_groups.get((batch_index, lhs_row_block, group_block))
                        rhs_tile = rhs_groups.get((batch_index, rhs_row_block, group_block))
                        if lhs_tile is None or rhs_tile is None:
                            continue

                        head_start = group_block * group_heads
                        dst_group_tiles: List[TileLike] = []
                        lane_heads: List[int] = []
                        for lane in range(group_heads):
                            head_index = head_start + lane
                            if head_index >= dst_heads:
                                break
                            dst_tile = dst_by_key.get((batch_index, lhs_row_block, rhs_row_block, head_index))
                            if dst_tile is None:
                                continue
                            lane_heads.append(head_index)
                            dst_group_tiles.append(dst_tile)

                        if not dst_group_tiles:
                            continue

                        threads.append(
                            {
                                "control": "tensor_tile_group",
                                "lhs_tiles": [lhs_tile],
                                "rhs_tiles": [rhs_tile],
                                "dst_tiles": dst_group_tiles,
                                "batch_index": batch_index,
                                "group_block": group_block,
                                "group_start": head_start,
                                "group_heads": len(dst_group_tiles),
                                "lane_heads": lane_heads,
                                "lhs_row_block": lhs_row_block,
                                "rhs_row_block": rhs_row_block,
                            }
                        )
        return threads

    def mapt_view_matmul(
        self,
        src1: object,
        src2: object,
        dst: object,
    ) -> List[ViewMatmulThread]:
        if not (
            len(getattr(src1, "logical_shape", ())) == 4
            and len(getattr(src2, "logical_shape", ())) == 4
            and len(getattr(dst, "logical_shape", ())) == 4
        ):
            raise NotImplementedError("mapt_view_matmul currently supports BSHD tensors only")

        src1_batch = int(getattr(src1, "logical_shape", ())[0])
        src2_batch = int(getattr(src2, "logical_shape", ())[0])
        dst_batch = int(getattr(dst, "logical_shape", ())[0])
        if src1_batch != src2_batch or src1_batch != dst_batch:
            raise ValueError(
                f"scatter-group mapt requires matched batch size, got src1={src1_batch} "
                f"src2={src2_batch} dst={dst_batch}"
            )
        src1_head_dim = int(getattr(src1, "logical_shape", ())[-1])
        src2_head_dim = int(getattr(src2, "logical_shape", ())[-1])
        dst_head_dim = int(getattr(dst, "logical_shape", ())[-1])
        if src2_head_dim <= 0 or self.program.mlen % src2_head_dim != 0:
            raise ValueError(
                f"scatter-group mapt requires src2 head_dim to divide mlen={self.program.mlen}, got {src2_head_dim}"
            )
        if dst_head_dim != src2_head_dim:
            raise ValueError(
                f"scatter-group mapt expects dst head_dim == src2 head_dim, got dst={dst_head_dim} src2={src2_head_dim}"
            )

        group_heads = self.program.mlen // src2_head_dim
        src1_by_batch_head_seq_k: Dict[Tuple[int, int, int, int], object] = {}
        src2_by_batch_seq_group: Dict[Tuple[int, int, int], object] = {}
        threads: List[ViewMatmulThread] = []

        for tile in _tiles_in_grid_order(src1.tiles):
            batch_index = _bshd_tile_batch_index(tile)
            head_index = int(tile.metadata.get("head_index", 0))
            seq_block = _bshd_tile_seq_block(tile)
            k_index = int(tile.metadata.get("d_tile_index", tile.coord[1]))
            src1_by_batch_head_seq_k[(batch_index, head_index, seq_block, k_index)] = tile

        for tile in _tiles_in_grid_order(src2.tiles):
            batch_index = _bshd_tile_batch_index(tile)
            seq_block = _bshd_tile_seq_block(tile)
            group_block = int(tile.coord[1])
            src2_by_batch_seq_group[(batch_index, seq_block, group_block)] = tile

        for dst_tile in _tiles_in_grid_order(dst.tiles):
            batch_index = _bshd_tile_batch_index(dst_tile)
            seq_block = _bshd_tile_seq_block(dst_tile)
            group_block = int(dst_tile.coord[1])
            group_start = group_block * group_heads
            lane_heads: List[int] = []
            lhs_candidates: List[List[object]] = []

            for lane in range(group_heads):
                head_index = group_start + lane
                lane_k_tiles = [
                    tile
                    for (tile_batch, tile_head, tile_seq, _), tile in src1_by_batch_head_seq_k.items()
                    if tile_batch == batch_index and tile_head == head_index and tile_seq == seq_block
                ]
                if not lane_k_tiles:
                    continue
                lane_heads.append(head_index)
                lhs_candidates.append(sorted(lane_k_tiles, key=lambda tile: int(tile.metadata.get("d_tile_index", 0))))

            rhs_terms: List[ViewMatmulTerm] = []
            rhs_row_blocks = sorted(
                row
                for (tile_batch, row, col_group) in src2_by_batch_seq_group.keys()
                if tile_batch == batch_index and col_group == group_block
            )
            for rhs_row_block in rhs_row_blocks:
                rhs_tile = src2_by_batch_seq_group.get((batch_index, rhs_row_block, group_block))
                if rhs_tile is None:
                    continue
                term_lhs_tiles: List[object] = []
                for lane_tiles in lhs_candidates:
                    if rhs_row_block >= len(lane_tiles):
                        term_lhs_tiles = []
                        break
                    term_lhs_tiles.append(lane_tiles[rhs_row_block])
                if not term_lhs_tiles:
                    continue
                rhs_terms.append((term_lhs_tiles, rhs_tile))

            threads.append((dst_tile, rhs_terms, group_start))
        return threads

    def mapt_back(self, signal_4: List[object], signal_1: List[object]) -> object:
        if not signal_1:
            return None
        if signal_4:
            controls = {
                item.get("control")
                for item in signal_4
                if isinstance(item, dict) and item.get("control") is not None
            }
            if len(controls) > 1:
                raise RuntimeError(f"mapt_back received mixed map controls: {sorted(controls)}")
        dst_tile = self._extract_dst_tile_from_group(signal_1[0])
        if dst_tile is None:
            return None
        if isinstance(dst_tile, TensorTile):
            return self.tensors.get(dst_tile.tensor_name) or self.vectors.get(dst_tile.tensor_name) or self.inputs.get(dst_tile.tensor_name)
        if isinstance(dst_tile, InputTile):
            return self.inputs.get(dst_tile.input_name)
        return None

    def _extract_dst_tile_from_group(self, group: object) -> Optional[object]:
        if isinstance(group, dict):
            dst_tile = group.get("dst_tile")
            if isinstance(dst_tile, (TensorTile, InputTile, VectorTile)):
                return dst_tile
            dst_tiles = group.get("dst_tiles")
            if isinstance(dst_tiles, list):
                for item in dst_tiles:
                    if isinstance(item, (TensorTile, InputTile, VectorTile)):
                        return item
            return None
        if not isinstance(group, list) or not group:
            return None
        tail = group[-1]
        if isinstance(tail, list) and len(tail) == 1 and isinstance(tail[0], (TensorTile, InputTile, VectorTile)):
            return tail[0]
        if isinstance(tail, (TensorTile, InputTile, VectorTile)):
            return tail
        return None

    def _resolve_tiles_from_operand(self, operand: object) -> List[object]:
        if isinstance(operand, Input):
            return _tiles_in_grid_order(operand.tiles)
        if isinstance(operand, Tensor):
            return _tiles_in_grid_order(operand.tiles)
        if isinstance(operand, Vector):
            return _tiles_in_grid_order(operand.tiles)
        if isinstance(operand, InputSlice):
            return self._resolve_slice_tiles(operand.base.tiles, operand.base.logical_shape, operand.selectors)
        if isinstance(operand, TensorSlice):
            return self._resolve_slice_tiles(operand.base.tiles, operand.base.logical_shape, operand.selectors)
        if isinstance(operand, VectorSlice):
            return self._resolve_slice_tiles(operand.base.tiles, operand.base.logical_shape, operand.selectors)
        if isinstance(operand, (InputTile, TensorTile, VectorTile)):
            return [operand]
        raise NotImplementedError(f"Unsupported operand for mapt(control=0): {type(operand).__name__}")

    def _resolve_slice_tiles(
        self,
        tiles: Dict[TileCoord, object],
        logical_shape: LogicalShape,
        selectors: Tuple[SliceItem, ...],
    ) -> List[object]:
        row_range, col_range = _logical_selectors_to_physical_ranges(logical_shape, selectors)
        resolved: List[object] = []
        for tile in _tiles_in_grid_order(tiles):
            row_block, col_block = tile.coord
            row_start = row_block * self.program.mlen
            row_end = row_start + tile.tile_shape[0]
            col_start = col_block * self.program.mlen
            col_end = col_start + tile.tile_shape[1]
            if _ranges_overlap((row_start, row_end), row_range) and _ranges_overlap((col_start, col_end), col_range):
                # Views are aliases: return the owner tile directly instead of
                # materializing a derived tile object with independent identity.
                resolved.append(tile)
        return resolved


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
        self.program.emit_matmul(
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
            self.program.emit_slot_matmul(
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

        self.program.emit_btmm(
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
        self.program.emit_btmm_wo(
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

        self.program.emit_fp_kernel(
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
        self.program.emit_row_operation(
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
                    self.emit_zero_vram_tile(int(new_vram_addr))
                    self.emit_tile_binary(
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
            self.emit_tile_binary(
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
                            self.emit_zero_vram_tile(int(new_vram_addr))
                            self.emit_tile_binary(
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
            self.emit_zero_vram_tile(int(vram_addr))
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

    def emit_hbm_tile_to_mram(
        self,
        *,
        hbm_addr: int,
        mram_addr: int,
        hbm_offset: int = 0,
        hbm_scale: Optional[int] = None,
        hbm_stride: Optional[int] = None,
    ) -> None:
        addr_reg = self.compiler.register_allocator.allocate_addr(1)[0]
        gp_addr = self.compiler.register_allocator.allocate_gp(2)
        gp_exec = self.compiler.register_allocator.allocate_gp(3)
        gp_scale, gp_stride, gp_mram = gp_exec
        scale_val = self.tile_elems if hbm_scale is None else int(hbm_scale)
        stride_val = self.mlen if hbm_stride is None else int(hbm_stride)

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
        isa += f"S_ADDI_INT gp{gp_scale}, gp0, {self.tile_elems}\n"
        isa += f"C_SET_SCALE_REG gp{gp_scale}\n"
        isa += f"S_ADDI_INT gp{gp_stride}, gp0, {self.mlen}\n"
        isa += f"C_SET_STRIDE_REG gp{gp_stride}\n"
        self.compiler.generated_code += isa

        self.compiler.register_allocator.free_gp(gp_addr)
        self.compiler.register_allocator.free_gp(gp_exec)
        self.compiler.register_allocator.free_addr([addr_reg])

    def emit_load_tile_from_hbm(
        self,
        *,
        hbm_addr: int,
        vram_addr: int,
        hbm_stride: Optional[int] = None,
        hbm_scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
    ) -> None:
        isa = self.compiler.load_tile_from_hbm(
            hbm_addr=hbm_addr,
            vram_addr=vram_addr,
            batch=self.mlen,
            hidden_size=self.mlen,
            hbm_stride=self.mlen if hbm_stride is None else int(hbm_stride),
            hbm_scale_size=self.tile_elems if hbm_scale_size is None else int(hbm_scale_size),
            hbm_start_offset=int(hbm_start_offset),
            vlen=self.mlen,
            preload_len=self.blen,
        )
        self.compiler.generated_code += isa

    def emit_store_tile_to_hbm(
        self,
        *,
        vram_addr: int,
        hbm_addr: int,
        hbm_stride: Optional[int] = None,
        hbm_scale_size: Optional[int] = None,
        hbm_start_offset: int = 0,
    ) -> None:
        isa = self.compiler.store_tile_to_hbm(
            vram_addr=vram_addr,
            hbm_addr=hbm_addr,
            batch=self.mlen,
            hidden_size=self.mlen,
            hbm_stride=self.mlen if hbm_stride is None else int(hbm_stride),
            hbm_scale_size=self.tile_elems if hbm_scale_size is None else int(hbm_scale_size),
            hbm_start_offset=int(hbm_start_offset),
            vlen=self.mlen,
            store_amount=self.blen,
        )
        self.compiler.generated_code += isa

    def emit_zero_vram_tile(self, vram_addr: int) -> None:
        gp_regs = self.compiler.register_allocator.allocate_gp(2)
        gp, gp_loop = gp_regs
        lines = [f"; zero tile vram[{vram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp}, gp0, {vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {self.mlen}")
        lines.append(f"V_MUL_VF gp{gp}, gp{gp}, f0, 0")
        lines.append(f"S_ADDI_INT gp{gp}, gp{gp}, {self.mlen}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.compiler.register_allocator.free_gp(gp_regs)
        self.compiler.generated_code += "\n".join(lines) + "\n"

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
        if row_width != self.mlen:
            raise ValueError(
                f"emit_map_v_fp_tile currently requires row_width == mlen == {self.mlen}, got {row_width}"
            )
        gp_regs = self.compiler.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp_regs
        lines = [f"; map fp tile task {task_id} fpram[{fpram_addr}] -> vram[{vram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {fpram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
        lines.append(f"S_MAP_V_FP gp{gp_dst}, gp{gp_src}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_width}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_width}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.compiler.register_allocator.free_gp(gp_regs)
        self.compiler.generated_code += "\n".join(lines) + "\n"

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
        if row_width != self.mlen:
            raise ValueError(
                f"emit_map_fp_v_tile currently requires row_width == mlen == {self.mlen}, got {row_width}"
            )
        gp_regs = self.compiler.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp_regs
        lines = [f"; map fp tile task {task_id} vram[{vram_addr}] -> fpram[{fpram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fpram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
        lines.append(f"S_MAP_FP_V gp{gp_dst}, gp{gp_src}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_width}")
        lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_width}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.compiler.register_allocator.free_gp(gp_regs)
        self.compiler.generated_code += "\n".join(lines) + "\n"

    def emit_btmm(
        self,
        *,
        lhs_packed_vram_addr: int,
        rhs_mram_addr: int,
        task_id: str = "btmm",
    ) -> None:
        gp_regs = self.compiler.register_allocator.allocate_gp(2)
        gp_mram_base, gp_lhs_base = gp_regs
        lines = [
            (
                f"; btmm task {task_id} lhs_packed=vram[{lhs_packed_vram_addr}] "
                f"rhs_mram={rhs_mram_addr} lanes={self.btmm_lane_count} head_width={self.btmm_hlen}"
            ),
            f"S_ADDI_INT gp{gp_mram_base}, gp0, {rhs_mram_addr}",
            f"S_ADDI_INT gp{gp_lhs_base}, gp0, {lhs_packed_vram_addr}",
            f"M_BTMM gp0, gp{gp_mram_base}, gp{gp_lhs_base}",
        ]
        self.compiler.generated_code += "\n".join(lines) + "\n"
        self.compiler.register_allocator.free_gp(gp_regs)

    def emit_btmm_wo(
        self,
        *,
        base_addr: int,
        tile_count: int,
        task_id: str = "btmm_wo",
    ) -> None:
        gp_out = self.compiler.register_allocator.allocate_gp(1)[0]
        lines = [
            (
                f"; btmm write-only task {task_id} out=vram[{base_addr}] "
                f"tiles={tile_count} lanes={self.btmm_lane_count} head_width={self.btmm_hlen}"
            ),
            f"S_ADDI_INT gp{gp_out}, gp0, {base_addr}",
            f"M_BMM_WO gp{gp_out}, 0",
        ]
        self.compiler.generated_code += "\n".join(lines) + "\n"
        self.compiler.register_allocator.free_gp([gp_out])

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

        gp_regs = self.compiler.register_allocator.allocate_gp(5)
        gp_act, gp_mat, gp_out, gp_stride, gp_loop = gp_regs
        tiles_per_mlen = self.mlen // self.blen
        lines = [f"; matmul task {task_id}"]
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")
        lhs_prog = self._arith_progression([int(addr) for addr in lhs_vram_addrs])
        rhs_prog = self._arith_progression([int(addr) for addr in rhs_mram_addrs])

        for oc in range(tiles_per_mlen):
            for orow in range(tiles_per_mlen):
                if lhs_prog is not None and rhs_prog is not None:
                    lhs_start, pair_count, lhs_step = lhs_prog
                    rhs_start, _, rhs_step = rhs_prog
                    act_addr = lhs_start + orow * self.blen * self.mlen
                    mat_addr = rhs_start + oc * self.blen
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
                    lines.append(f"C_LOOP_START gp{gp_loop}, {pair_count}")
                    lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                    lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {lhs_step}")
                    lines.append(f"S_ADDI_INT gp{gp_mat}, gp{gp_mat}, {rhs_step}")
                    lines.append(f"C_LOOP_END gp{gp_loop}")
                else:
                    for lhs_addr, rhs_addr in zip(lhs_vram_addrs, rhs_mram_addrs):
                        act_addr = lhs_addr + orow * self.blen * self.mlen
                        mat_addr = rhs_addr + oc * self.blen
                        lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
                        lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
                        lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
                out_addr = dst_vram_addr + orow * self.blen * self.mlen + oc * self.blen
                lines.append(f"S_ADDI_INT gp{gp_out}, gp0, {out_addr}")
                lines.append(f"M_MM_WO gp{gp_out}, gp0, 0")

        self.compiler.register_allocator.free_gp(gp_regs)
        self.compiler.generated_code += "\n".join(lines) + "\n"

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
        if col_count % self.blen != 0:
            raise ValueError(
                f"emit_slot_matmul requires col_count divisible by blen={self.blen}, got {col_count}"
            )
        if zero_dst:
            self.emit_zero_vram_tile(dst_vram_addr)

        gp_regs = self.compiler.register_allocator.allocate_gp(5)
        gp_act, gp_mat, gp_out, gp_stride, gp_loop = gp_regs
        tiles_per_mlen = self.mlen // self.blen
        tiles_per_slot = col_count // self.blen
        lines = [f"; slot matmul task {task_id}"]
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for oc in range(tiles_per_slot):
            act_addr = lhs_vram_addr
            mat_addr = rhs_mram_addr + rhs_col_offset + oc * self.blen
            out_addr = dst_vram_addr + dst_col_offset + oc * self.blen
            lines.append(f"S_ADDI_INT gp{gp_act}, gp0, {act_addr}")
            lines.append(f"S_ADDI_INT gp{gp_mat}, gp0, {mat_addr}")
            lines.append(f"S_ADDI_INT gp{gp_out}, gp0, {out_addr}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {tiles_per_mlen}")
            lines.append(f"M_MM 0, gp{gp_mat}, gp{gp_act}")
            lines.append(f"M_MM_WO gp{gp_out}, gp0, 0")
            lines.append(f"S_ADDI_INT gp{gp_act}, gp{gp_act}, {self.blen * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_out}, gp{gp_out}, {self.blen * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")

        self.compiler.register_allocator.free_gp(gp_regs)
        self.compiler.generated_code += "\n".join(lines) + "\n"

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
        gp_regs = self.compiler.register_allocator.allocate_gp(4)
        gp_dst, gp_lhs, gp_rhs, gp_loop = gp_regs
        lines = [f"; tile binary task {task_id} op={op}"]
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_lhs}, gp0, {lhs_vram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_rhs}, gp0, {rhs_vram_addr}")
        lines.append(f"C_LOOP_START gp{gp_loop}, {self.mlen}")
        if op == "sub":
            lines.append(f"{op_to_insn[op]} gp{gp_dst}, gp{gp_rhs}, gp{gp_lhs}, 0")
        else:
            lines.append(f"{op_to_insn[op]} gp{gp_dst}, gp{gp_lhs}, gp{gp_rhs}, 0")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}")
        lines.append(f"S_ADDI_INT gp{gp_lhs}, gp{gp_lhs}, {self.mlen}")
        lines.append(f"S_ADDI_INT gp{gp_rhs}, gp{gp_rhs}, {self.mlen}")
        lines.append(f"C_LOOP_END gp{gp_loop}")
        self.compiler.register_allocator.free_gp(gp_regs)
        self.compiler.generated_code += "\n".join(lines) + "\n"

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
            gp_regs = self.compiler.register_allocator.allocate_gp(3)
            gp_src, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src_prog = self._arith_progression([int(addr) for addr in src1_addrs])
            dst_prog = self._arith_progression([int(addr) for addr in dst_addrs])
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
            self.compiler.register_allocator.free_gp(gp_regs)
            self.compiler.generated_code += "\n".join(lines) + "\n"
            return
        if op in unary_math:
            gp_regs = self.compiler.register_allocator.allocate_gp(3)
            gp_src, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src_prog = self._arith_progression([int(addr) for addr in src1_addrs])
            dst_prog = self._arith_progression([int(addr) for addr in dst_addrs])
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
            self.compiler.register_allocator.free_gp(gp_regs)
            self.compiler.generated_code += "\n".join(lines) + "\n"
            return
        if op in binary_math:
            if src2_addrs is None:
                raise ValueError(f"emit_fp_kernel op={op!r} requires src2_addrs")
            gp_regs = self.compiler.register_allocator.allocate_gp(4)
            gp_a, gp_b, gp_dst, gp_loop = gp_regs
            lines = [f"; fp kernel task {task_id} op={op}"]
            src1_prog = self._arith_progression([int(addr) for addr in src1_addrs])
            src2_prog = self._arith_progression([int(addr) for addr in src2_addrs])
            dst_prog = self._arith_progression([int(addr) for addr in dst_addrs])
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
            self.compiler.register_allocator.free_gp(gp_regs)
            self.compiler.generated_code += "\n".join(lines) + "\n"
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

        gp_regs = self.compiler.register_allocator.allocate_gp(5)
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
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        elif op in reduce_ops:
            if dst_addrs is None or len(dst_addrs) != row_count:
                raise ValueError(f"emit_row_operation op={op!r} expects one dst fp addr per row")
            dst_prog = self._arith_progression([int(addr) for addr in dst_addrs])
            if dst_prog is None:
                for row_index, dst_addr in enumerate(dst_addrs):
                    row_addr = int(src_vram_addr) + row_index * self.mlen
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
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {dst_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            if rhs_addrs is None or len(rhs_addrs) not in (1, row_count):
                raise ValueError(f"emit_row_operation op={op!r} expects one rhs fp addr or one per row")
            rhs_prog = self._arith_progression([int(addr) for addr in rhs_addrs]) if len(rhs_addrs) > 1 else None
            if len(rhs_addrs) == 1:
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {int(rhs_addrs[0])}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {int(row_count)}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                if op == "sub":
                    lines.append(f"V_SUB_VF gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                else:
                    lines.append(f"{binary_ops[op]} gp{gp_dst}, gp{gp_src}, f1, {1 if use_mask else 0}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}")
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
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {rhs_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for row_index, rhs_addr in enumerate(rhs_addrs):
                    row_addr = int(src_vram_addr) + row_index * self.mlen
                    dst_row_addr = dst_vram_addr + row_index * self.mlen
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

        self.compiler.register_allocator.free_gp(gp_regs)
        self.compiler.generated_code += "\n".join(lines) + "\n"

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


def _logical_shape_to_physical_shape(logical_shape: LogicalShape) -> Tuple[int, int]:
    if len(logical_shape) == 4:
        b, s, h, d = logical_shape
        return b * s, h * d
    if len(logical_shape) == 3:
        x, y, z = logical_shape
        return x, y * z
    if len(logical_shape) == 2:
        return logical_shape[0], logical_shape[1]
    raise NotImplementedError(f"Unsupported logical shape: {logical_shape}")


def _logical_selectors_to_physical_ranges(
    logical_shape: LogicalShape,
    selectors: Tuple[SliceItem, ...],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    normalized = list(selectors) + [slice(None)] * max(0, len(logical_shape) - len(selectors))
    if len(logical_shape) == 4:
        b, s, h, d = logical_shape
        b_sel, s_sel, h_sel, d_sel = normalized[:4]
        b_range = _slice_item_to_range(b_sel, b)
        s_range = _slice_item_to_range(s_sel, s)
        h_range = _slice_item_to_range(h_sel, h)
        d_range = _slice_item_to_range(d_sel, d)
        row_range = (b_range[0] * s + s_range[0], (b_range[1] - 1) * s + s_range[1])
        col_range = (h_range[0] * d + d_range[0], (h_range[1] - 1) * d + d_range[1])
        return row_range, col_range
    if len(logical_shape) == 3:
        rows, outer, inner = logical_shape
        row_sel, outer_sel, inner_sel = normalized[:3]
        row_range = _slice_item_to_range(row_sel, rows)
        col_range = _logical_3d_selectors_to_flat_col_range(
            outer_extent=outer,
            inner_extent=inner,
            outer_selector=outer_sel,
            inner_selector=inner_sel,
        )
        return row_range, col_range
    if len(logical_shape) == 2:
        rows, cols = logical_shape
        row_sel, col_sel = normalized[:2]
        return _slice_item_to_range(row_sel, rows), _slice_item_to_range(col_sel, cols)
    raise NotImplementedError(f"Unsupported logical shape for selectors: {logical_shape}")


def _slice_item_to_range(selector: SliceItem, extent: int) -> Tuple[int, int]:
    if isinstance(selector, int):
        index = selector if selector >= 0 else extent + selector
        return index, index + 1
    start = 0 if selector.start is None else selector.start
    stop = extent if selector.stop is None else selector.stop
    return start, stop


def _ranges_overlap(lhs: Tuple[int, int], rhs: Tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


def _tiles_in_grid_order(tiles: Dict[TileCoord, object]) -> List[object]:
    return [tile for _, tile in sorted(tiles.items(), key=lambda item: item[0])]


def _bshd_tile_batch_index(tile: object) -> int:
    return int(getattr(tile, "metadata", {}).get("batch_index", 0))


def _bshd_tile_seq_block(tile: object) -> int:
    return int(getattr(tile, "metadata", {}).get("seq_block", getattr(tile, "coord", (0, 0))[0]))


def _is_tile_object(tile: object) -> bool:
    return isinstance(tile, (TensorTile, InputTile, VectorTile))


def _is_full_element_index(item: Tuple[SliceItem, ...], rank: int) -> bool:
    return len(item) == rank and all(isinstance(index, int) for index in item)


def _contains_parallel_selector(item: Tuple[object, ...]) -> bool:
    return any(isinstance(selector, (ParallelAxis, ParallelExpr, ParallelAccess)) for selector in item)


def _normalize_index(index: int, extent: int) -> int:
    normalized = int(index)
    if normalized < 0:
        normalized += int(extent)
    if normalized < 0 or normalized >= int(extent):
        raise IndexError(f"Index {index} is out of range for extent {extent}")
    return normalized


def _tile_owner_name(tile: TileLike) -> str:
    if isinstance(tile, InputTile):
        return tile.input_name
    return tile.tensor_name


def _logical_shape_to_hbm_stride(logical_shape: LogicalShape) -> int:
    if len(logical_shape) == 4:
        _, _, heads, head_dim = logical_shape
        return int(heads) * int(head_dim)
    rows, cols = _logical_shape_to_physical_shape(logical_shape)
    return int(cols if cols > 0 else rows)


def _tile_coord_to_hbm_offset(coord: TileCoord, logical_shape: LogicalShape, mlen: int) -> int:
    _, stride = _logical_shape_to_physical_shape(logical_shape)[0], _logical_shape_to_hbm_stride(logical_shape)
    return int(coord[0]) * int(mlen) * int(stride) + int(coord[1]) * int(mlen)


def _logical_3d_selectors_to_flat_col_range(
    *,
    outer_extent: int,
    inner_extent: int,
    outer_selector: SliceItem,
    inner_selector: SliceItem,
) -> Tuple[int, int]:
    outer_range = _slice_item_to_range(outer_selector, outer_extent)
    inner_range = _slice_item_to_range(inner_selector, inner_extent)
    outer_full = outer_range == (0, outer_extent)
    inner_full = inner_range == (0, inner_extent)
    if isinstance(outer_selector, int):
        base = outer_range[0] * inner_extent
        return base + inner_range[0], base + inner_range[1]
    if inner_full:
        return outer_range[0] * inner_extent, outer_range[1] * inner_extent
    if outer_range[1] - outer_range[0] == 1:
        base = outer_range[0] * inner_extent
        return base + inner_range[0], base + inner_range[1]
    if outer_full and inner_range[1] - inner_range[0] == inner_extent:
        return 0, outer_extent * inner_extent
    raise NotImplementedError(
        "3D vector slicing currently supports full-inner slices or one selected outer lane; "
        f"got outer={outer_selector!r} inner={inner_selector!r}"
    )


def _logical_indices_to_physical_coord(
    logical_shape: LogicalShape,
    indices: Tuple[int, ...],
) -> Tuple[int, int]:
    if len(logical_shape) != len(indices):
        raise ValueError(f"logical_indices rank mismatch: shape={logical_shape} indices={indices}")
    if len(logical_shape) == 4:
        b, s, h, d = logical_shape
        bi, si, hi, di = indices
        return bi * s + si, hi * d + di
    if len(logical_shape) == 3:
        x, y, z = logical_shape
        xi, yi, zi = indices
        return xi, yi * z + zi
    if len(logical_shape) == 2:
        ri, ci = indices
        return ri, ci
    raise NotImplementedError(f"Unsupported logical shape for element indices: {logical_shape}")


def _physical_tile_coord_to_fp_index(
    fragment_shape: Tuple[int, ...],
    *,
    local_row: int,
    local_col: int,
    mlen: int,
    btmm_hlen: int,
) -> FPIndex:
    normalized = tuple(int(dim) for dim in fragment_shape)
    if len(normalized) == 2:
        rows, cols = normalized
        if local_row < 0 or local_row >= rows or local_col < 0 or local_col >= cols:
            raise IndexError(
                f"Local tile coord ({local_row}, {local_col}) is out of range for FP fragment shape {fragment_shape}"
            )
        return int(local_row), int(local_col)
    if normalized == (mlen, mlen):
        return int(local_row), int(local_col)
    if btmm_hlen > 0 and normalized == (mlen, mlen // btmm_hlen, btmm_hlen):
        return int(local_row), int(local_col // btmm_hlen), int(local_col % btmm_hlen)
    raise ValueError(
        f"Unsupported fp fragment shape for tile-element mapping: {fragment_shape}; "
        f"expected ({mlen}, {mlen}) or ({mlen}, {mlen // btmm_hlen if btmm_hlen > 0 else 'invalid'}, {btmm_hlen})"
    )


def _vector_tile_row_fp_groups(
    *,
    src_tile: VectorTile,
    fragment: FPFragment,
    mlen: int,
    btmm_hlen: int,
    src_slice_ranges: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> List[List[FPVar]]:
    row_block, col_block = src_tile.coord
    row_start = row_block * mlen
    row_end = row_start + int(src_tile.tile_shape[0])
    col_start = col_block * mlen
    col_end = col_start + int(src_tile.tile_shape[1])

    if src_slice_ranges is None:
        use_row_start, use_row_end = row_start, row_end
        use_col_start, use_col_end = col_start, col_end
    else:
        req_row_range, req_col_range = src_slice_ranges
        use_row_start = max(row_start, int(req_row_range[0]))
        use_row_end = min(row_end, int(req_row_range[1]))
        use_col_start = max(col_start, int(req_col_range[0]))
        use_col_end = min(col_end, int(req_col_range[1]))
        if use_row_start >= use_row_end or use_col_start >= use_col_end:
            return []

    groups: List[List[FPVar]] = []
    for physical_row in range(use_row_start, use_row_end):
        local_row = physical_row - row_start
        row_vars: List[FPVar] = []
        for physical_col in range(use_col_start, use_col_end):
            local_col = physical_col - col_start
            fp_index = _physical_tile_coord_to_fp_index(
                fragment.shape,
                local_row=local_row,
                local_col=local_col,
                mlen=mlen,
                btmm_hlen=btmm_hlen,
            )
            fp_var = fragment.vars.get(fp_index)
            if not isinstance(fp_var, FPVar):
                raise RuntimeError(
                    f"VectorTile {src_tile.tile_id} bound to fragment {fragment.name!r} is missing fp cell {fp_index}"
                )
            row_vars.append(fp_var)
        groups.append(row_vars)
    return groups


def _unwrap_transposed_operand(operand: object) -> object:
    if isinstance(operand, (TensorTranspose, InputTranspose, VectorTranspose)):
        return operand.base
    return operand


def _is_transposed_operand(operand: object) -> bool:
    return isinstance(operand, (TensorTranspose, InputTranspose, VectorTranspose))


def _is_narrow_tile(tile: TileLike) -> bool:
    mlen = int(tile.metadata.get("mlen", tile.tile_shape[0]))
    return tile.tile_shape[0] != mlen or tile.tile_shape[1] != mlen


def _is_fp_domain_operand(operand: object) -> bool:
    return isinstance(
        operand,
        (
            FPVar,
            FPFragment,
            FPFragmentSlice,
            Vector,
            VectorSlice,
            VectorTile,
            ElementRef,
        ),
    )


def _is_parallel_graph_operand(operand: object) -> bool:
    return isinstance(operand, (ParallelAxis, ParallelAccess, ParallelExpr))


def _coerce_parallel_expr(value: object) -> ParallelExpr:
    if isinstance(value, ParallelExpr):
        return value
    if isinstance(value, ParallelAxis):
        return value._as_expr()
    if isinstance(value, ParallelAccess):
        return value._as_expr()
    if isinstance(value, FPVar):
        return ParallelExpr(kind="fpvar", value=value)
    if isinstance(value, (int, float)):
        return ParallelExpr(kind="literal", value=float(value))
    raise TypeError(f"Unsupported parallel expression operand: {type(value).__name__}")


def _collect_parallel_accesses(expr: ParallelExpr) -> List[ParallelAccess]:
    accesses: List[ParallelAccess] = []
    if expr.kind == "load":
        access = expr.value
        if isinstance(access, ParallelAccess):
            accesses.append(access)
        return accesses
    for arg in expr.args:
        accesses.extend(_collect_parallel_accesses(arg))
    return accesses


def _collect_parallel_predicates(expr: ParallelExpr) -> List[str]:
    predicates: List[str] = []
    if expr.kind == "select" and expr.args:
        predicate = expr.args[0]
        predicates.append(_infer_parallel_predicate_kind(predicate))
    for arg in expr.args:
        predicates.extend(_collect_parallel_predicates(arg))
    return predicates


def _parallel_access_identity(access: ParallelAccess) -> str:
    base_name = getattr(access.base, "name", type(access.base).__name__)
    return f"{base_name}{tuple(access.selectors)!r}"


def _parallel_expr_identity(expr: ParallelExpr) -> str:
    if expr.kind == "literal":
        return f"literal({expr.value})"
    if expr.kind == "fpvar":
        fp_var = expr.value
        if isinstance(fp_var, FPVar):
            return f"fpvar({fp_var.name})"
        return "fpvar(?)"
    if expr.kind == "axis":
        axis = expr.value
        if isinstance(axis, ParallelAxis):
            return f"axis({axis.name})"
        return "axis(?)"
    if expr.kind == "load":
        access = expr.value
        if isinstance(access, ParallelAccess):
            return f"load({_parallel_access_identity(access)})"
        return "load(?)"
    if expr.kind in {"pair_index", "half_index"}:
        args = ",".join(_parallel_expr_identity(arg) for arg in expr.args)
        return f"{expr.kind}({args})"
    if expr.kind == "unary_op":
        args = ",".join(_parallel_expr_identity(arg) for arg in expr.args)
        return f"{expr.op}({args})"
    if expr.kind == "op":
        args = ",".join(_parallel_expr_identity(arg) for arg in expr.args)
        return f"{expr.op}({args})"
    if expr.kind == "select":
        args = ",".join(_parallel_expr_identity(arg) for arg in expr.args)
        return f"select({args})"
    return expr.kind


def _infer_parallel_load_metadata(access: ParallelAccess) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    selectors = tuple(access.selectors)
    if selectors and isinstance(selectors[-1], ParallelExpr):
        lane_expr = selectors[-1]
        if lane_expr.kind == "pair_index":
            metadata["companion_kind"] = "pair_swap"
        elif lane_expr.kind == "half_index":
            metadata["coefficient_layout"] = "preexpanded_full_lane"
    return metadata


def _infer_parallel_predicate_kind(expr: ParallelExpr) -> str:
    if (
        expr.kind == "op"
        and expr.op == "eq"
        and len(expr.args) == 2
        and ((expr.args[0].kind == "op" and expr.args[0].op == "mod") or (expr.args[1].kind == "op" and expr.args[1].op == "mod"))
    ):
        return "even_mask"
    return "generic"


def _build_parallel_execution_plan(
    region: ParallelRegionGraph,
    *,
    program: "TileTensorProgram",
) -> ParallelExecutionPlan:
    ext_i, ext_j, ext_k = (int(dim) for dim in region.extents)
    elem_width = _infer_parallel_elem_width(region=region, program=program)
    if elem_width <= 0 or program.mlen % elem_width != 0:
        raise ValueError(
            f"parallel execution plan requires elem_width to divide mlen: elem_width={elem_width}, mlen={program.mlen}"
        )
    if elem_width == int(program.mlen):
        k_step = int(program.mlen)
        k_count_per_cycle = 1
    else:
        k_step = max(1, program.mlen // elem_width)
        k_count_per_cycle = k_step
    cycle_groups: List[ParallelCycleGroup] = []
    cycle_plans: List[ParallelCyclePlan] = []
    pack_axis1_lanes = elem_width < int(program.mlen) and ext_k == elem_width
    if pack_axis1_lanes:
        if ext_j % k_count_per_cycle != 0:
            raise NotImplementedError(
                "parallel execution plan currently requires axis-1 lane packing to divide ext_j evenly; "
                f"ext_j={ext_j}, lanes_per_cycle={k_count_per_cycle}, elem_width={elem_width}, mlen={program.mlen}"
            )
        for i_index in range(ext_i):
            for j_index in range(0, ext_j, k_count_per_cycle):
                group = ParallelCycleGroup(
                    i_index=int(i_index),
                    j_index=int(j_index),
                    k_base=0,
                    k_count=int(k_count_per_cycle),
                    elem_width=int(elem_width),
                    element_count=int(k_count_per_cycle * elem_width),
                )
                cycle_groups.append(group)
                cycle_plans.append(_build_parallel_cycle_plan(region=region, group=group))
    else:
        for i_index in range(ext_i):
            for j_index in range(ext_j):
                for k_base in range(0, ext_k, k_step):
                    if elem_width == int(program.mlen):
                        k_count = 1
                        element_count = int(program.mlen)
                    else:
                        k_count = min(k_count_per_cycle, ext_k - k_base)
                        element_count = int(k_count * elem_width)
                    group = ParallelCycleGroup(
                        i_index=int(i_index),
                        j_index=int(j_index),
                        k_base=int(k_base),
                        k_count=int(k_count),
                        elem_width=int(elem_width),
                        element_count=element_count,
                    )
                    cycle_groups.append(group)
                    cycle_plans.append(_build_parallel_cycle_plan(region=region, group=group))
    return ParallelExecutionPlan(
        region_name=region.name,
        cycle_groups=cycle_groups,
        cycle_plans=cycle_plans,
        metadata={
            "elem_width": int(elem_width),
            "k_count_per_cycle": int(k_count_per_cycle),
            "cycle_element_budget": int(program.mlen),
        },
    )


def _infer_parallel_elem_width(
    *,
    region: ParallelRegionGraph,
    program: "TileTensorProgram",
) -> int:
    candidate_widths: set[int] = set()
    for assignment in region.assignments:
        dst_shape = tuple(getattr(assignment.dst.base, "logical_shape", ()))
        if len(dst_shape) < 1:
            continue
        candidate_widths.add(int(dst_shape[-1]))
    if not candidate_widths:
        return int(program.mlen)
    if len(candidate_widths) != 1:
        raise ValueError(
            f"parallel execution plan currently expects one unified innermost width, got {sorted(candidate_widths)}"
        )
    innermost_width = int(next(iter(candidate_widths)))
    if innermost_width % int(program.mlen) == 0:
        return int(program.mlen)
    if innermost_width == int(program.btmm_hlen):
        return int(program.btmm_hlen)
    raise ValueError(
        "parallel execution plan only supports innermost width that is either "
        f"btmm_hlen ({int(program.btmm_hlen)}) or a multiple of mlen ({int(program.mlen)}); "
        f"got {innermost_width}"
    )


def _build_parallel_cycle_plan(
    *,
    region: ParallelRegionGraph,
    group: ParallelCycleGroup,
) -> ParallelCyclePlan:
    input_slot_map: Dict[str, int] = {}
    input_slots: List[ParallelInputCacheSlotPlan] = []
    output_slot_map: Dict[str, int] = {}
    output_slots: List[ParallelOutputCacheSlotPlan] = []
    load_ops: List[ParallelLoadOp] = []
    compute_ops: List[ParallelComputeOp] = []
    writeback_ops: List[ParallelWritebackOp] = []

    for assignment in region.assignments:
        dst_access = assignment.dst
        dst_identity = _parallel_access_identity(dst_access)
        dst_slot_id = output_slot_map.get(dst_identity)
        if dst_slot_id is None:
            dst_slot_id = len(output_slots)
            output_slot_map[dst_identity] = dst_slot_id
            output_slots.append(
                ParallelOutputCacheSlotPlan(
                    slot_id=dst_slot_id,
                    access=dst_access,
                    metadata={
                        "group_i": group.i_index,
                        "group_j": group.j_index,
                        "group_k_base": group.k_base,
                        "group_k_count": group.k_count,
                    },
                )
            )
            writeback_ops.append(
                ParallelWritebackOp(
                    slot_id=dst_slot_id,
                    access=dst_access,
                    metadata={
                        "writeback_kind": "value_view_update",
                        "group_i": group.i_index,
                        "group_j": group.j_index,
                        "group_k_base": group.k_base,
                        "group_k_count": group.k_count,
                    },
                )
            )

        input_slot_ids: List[int] = []
        for access in assignment.sources:
            access_identity = _parallel_access_identity(access)
            slot_id = input_slot_map.get(access_identity)
            if slot_id is None:
                slot_id = len(input_slots)
                input_slot_map[access_identity] = slot_id
                load_metadata = _infer_parallel_load_metadata(access)
                source_kind = "direct_fpfragment" if isinstance(access.base, Vector) else "mapv_to_fpram"
                input_slots.append(
                    ParallelInputCacheSlotPlan(
                        slot_id=slot_id,
                        access=access,
                        pattern_kind="uniform",
                        metadata={
                            **load_metadata,
                            "source_kind": source_kind,
                            "group_i": group.i_index,
                            "group_j": group.j_index,
                            "group_k_base": group.k_base,
                            "group_k_count": group.k_count,
                        },
                    )
                )
                if not isinstance(access.base, Vector):
                    load_ops.append(
                        ParallelLoadOp(
                            slot_id=slot_id,
                            access=access,
                            metadata={
                                **load_metadata,
                                "group_i": group.i_index,
                                "group_j": group.j_index,
                                "group_k_base": group.k_base,
                                "group_k_count": group.k_count,
                            },
                        )
                    )
            input_slot_ids.append(slot_id)

        predicate_kinds = _collect_parallel_predicates(assignment.expr)
        compute_ops.append(
            ParallelComputeOp(
                task_id=assignment.task_id,
                dst_slot_id=dst_slot_id,
                expr=assignment.expr,
                input_slot_ids=input_slot_ids,
                metadata={
                    "predicate_kinds": predicate_kinds,
                    "processing_kind": "per_output_element_ordered",
                },
            )
        )

    return ParallelCyclePlan(
        group=group,
        input_slots=input_slots,
        output_slots=output_slots,
        load_ops=load_ops,
        compute_ops=compute_ops,
        writeback_ops=writeback_ops,
        metadata={
            "writeback_at_cycle_end": True,
            "processing_mode": "per_output_element_ordered",
        },
    )


def _iter_fp_indices(shape: Tuple[int, ...]) -> List[FPIndex]:
    if not shape:
        return [()]
    indices: List[FPIndex] = [()]
    for dim in shape:
        next_indices: List[FPIndex] = []
        for prefix in indices:
            for value in range(int(dim)):
                next_indices.append(prefix + (value,))
        indices = next_indices
    return indices


def _iter_logical_indices(shape: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    if not shape:
        return [()]
    indices: List[Tuple[int, ...]] = [()]
    for dim in shape:
        next_indices: List[Tuple[int, ...]] = []
        for prefix in indices:
            for value in range(int(dim)):
                next_indices.append(prefix + (value,))
        indices = next_indices
    return indices


def _iter_selected_logical_indices(
    shape: Tuple[int, ...],
    selectors: Tuple[SliceItem, ...],
) -> List[Tuple[int, ...]]:
    normalized = list(selectors) + [slice(None)] * max(0, len(shape) - len(selectors))
    selected: List[Tuple[int, ...]] = []
    for logical_index in _iter_logical_indices(shape):
        keep = True
        for dim_idx, selector in enumerate(normalized[: len(shape)]):
            start, stop = _slice_item_to_range(selector, int(shape[dim_idx]))
            if logical_index[dim_idx] < start or logical_index[dim_idx] >= stop:
                keep = False
                break
        if keep:
            selected.append(logical_index)
    return selected


def _format_fp_index(index: FPIndex) -> str:
    return "".join(f"[{value}]" for value in index)


def _require_fp_addr(fp_var: FPVar) -> int:
    if fp_var.fp_mem_addr is None:
        raise RuntimeError(f"FPVar {fp_var.name!r} has no fp_mem_addr")
    return int(fp_var.fp_mem_addr)


def _fp_fragment_shape_to_tile_shape(
    shape: Tuple[int, ...],
    *,
    mlen: int,
    btmm_hlen: int,
) -> Tuple[int, int]:
    normalized = tuple(int(dim) for dim in shape)
    if len(normalized) == 2 and 0 < normalized[0] <= mlen and 0 < normalized[1] <= mlen:
        return normalized[0], normalized[1]
    if normalized == (mlen, mlen):
        return mlen, mlen
    if btmm_hlen > 0:
        expected_lane_count = mlen // btmm_hlen
        if normalized == (mlen, expected_lane_count, btmm_hlen):
            return mlen, mlen
    raise ValueError(
        "fpram-interacting FPFragment must have shape "
        f"({mlen}, {mlen}) or ({mlen}, {mlen // btmm_hlen if btmm_hlen > 0 else 'invalid'}, {btmm_hlen}), "
        f"got {shape}"
    )


def _fp_fragment_row_fp_vars(
    fragment: FPFragment,
    *,
    row_index: int,
    row_width: int,
    btmm_hlen: int,
) -> List[FPVar]:
    shape = tuple(int(dim) for dim in fragment.shape)
    if row_index < 0 or row_index >= int(shape[0]):
        raise IndexError(f"row_index {row_index} out of range for FPFragment {fragment.name!r} with shape {shape}")
    if shape == (row_width, row_width):
        return [fragment.vars[(row_index, col_index)] for col_index in range(int(row_width))]
    if btmm_hlen > 0:
        packed_head_count = row_width // btmm_hlen
        if shape == (row_width, packed_head_count, btmm_hlen):
            row_vars: List[FPVar] = []
            for head_index in range(packed_head_count):
                for col_index in range(btmm_hlen):
                    row_vars.append(fragment.vars[(row_index, head_index, col_index)])
            return row_vars
    raise ValueError(
        f"FPFragment {fragment.name!r} with shape {shape} cannot be materialized as one {row_width}-wide row"
    )
