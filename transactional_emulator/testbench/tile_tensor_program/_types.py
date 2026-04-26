"""TileTensor data classes: FP types, parallel types, tile/tensor types.

Includes the dataclasses, scope helpers, and module-level type aliases used
by the rest of the `tile_tensor_program` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple


__all__ = [
    "TileCoord",
    "LogicalShape",
    "SliceItem",
    "FPIndex",
    "FPVar",
    "FPFragment",
    "FPFragmentSlice",
    "ElementRef",
    "ParallelAxis",
    "ParallelAccess",
    "ParallelExpr",
    "ParallelAssignment",
    "ParallelCycleGroup",
    "ParallelInputCacheSlotPlan",
    "ParallelOutputCacheSlotPlan",
    "ParallelLoadOp",
    "ParallelComputeOp",
    "ParallelWritebackOp",
    "ParallelCyclePlan",
    "ParallelExecutionPlan",
    "ParallelRegionGraph",
    "_ParallelRegionScope",
    "_ParallelRegion2DScope",
    "InputTile",
    "TensorTile",
    "VectorTile",
    "ValueTile",
    "ValueTileView",
    "PreparedWrite",
    "Input",
    "Tensor",
    "Vector",
    "InputSlice",
    "TensorSlice",
    "VectorSlice",
    "InputTranspose",
    "TensorTranspose",
    "VectorTranspose",
    "TileLike",
    "TensorLike",
    "TransposedTensorLike",
    "SourceValueLike",
    "RowOperandLike",
    "ViewMatmulTerm",
    "ViewMatmulThread",
    "BTMMHeadGroupThread",
    "CopyMapvPacket",
    "MatmulMapvPacket",
    "GemmMapvPacket",
    "MapvPacket",
]


TileCoord = Tuple[int, int]
LogicalShape = Tuple[int, ...]
SliceItem = int | slice
FPIndex = Tuple[int, ...]


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





# Bottom-of-file import: helpers used by dataclass method bodies.
# Placed here (not at top) to avoid a circular import — `_helpers` does
# `from ._types import *` at its top, which would fail before the classes
# defined above were registered in this module's namespace.
from ._helpers import (
    _build_parallel_execution_plan,
    _coerce_parallel_expr,
    _collect_parallel_predicates,
    _contains_parallel_selector,
    _is_full_element_index,
    _parallel_access_identity,
)
