"""TileTensor program runtime for tile/value mapping and ISA-oriented execution.

This file is the main local execution layer behind the TileTensor tests. It
owns four responsibilities:

1. logical tensor/input objects and slice-to-tile resolution
2. value/scatter/scatter-group binding and residency management
3. compute-path routing for copy / atomic ops / matmul variants
4. ISA emission bookkeeping for the transactional-emulator testbench

The implementation is no longer a placeholder scaffold. The current design
follows the workspace split of:

    mapt -> mapv -> compute -> mapv_back -> mapt_back

where TensorManager stays at logical-tile grouping level and ValueManager owns
late value/scatter resolution.
"""

from __future__ import annotations

import sys
from math import ceil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.asm_templates import preload_addr_reg_asm
from tiled_developer_compiler import TiledDeveloperCompiler


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
class ValueTile:
    value_tile_id: str
    logical_shape: Tuple[int, int]
    from_input_tile: bool = False
    source_input_tile_id: Optional[str] = None
    residency: Dict[str, Optional[int | str]] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class Scatter:
    scatter_id: str
    backing_value_tile_id: str
    scatter_group_id: str
    row_offset: int
    row_count: int
    col_offset: int
    col_count: int
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ScatterGroup:
    group_id: str
    backing_value_tile_id: str
    scatter_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


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
        return InputSlice(base=self, selectors=item)

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
        return TensorSlice(base=self, selectors=item)

    @property
    def T(self) -> "TensorTranspose":
        return TensorTranspose(base=self)


@dataclass
class InputSlice:
    base: Input
    selectors: Tuple[SliceItem, ...]


@dataclass
class TensorSlice:
    base: Tensor
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


TileLike = TensorTile | InputTile
TensorLike = Tensor | Input
TransposedTensorLike = TensorTranspose | InputTranspose
SourceValueLike = ValueTile | Scatter
RowOperandLike = ValueTile | Scatter
ScatterGroupMatmulTerm = Tuple[List[TileLike], TileLike]
ScatterGroupMatmulThread = Tuple[TileLike, List[ScatterGroupMatmulTerm], int]
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


class ValueManager:
    """Resolve logical tiles into value/scatter objects and manage residency.

    The value layer is responsible for:

    - wide-tile direct bindings to ValueTile
    - narrow-tile bindings through Scatter / ScatterGroup
    - late destination materialization for writable outputs
    - HBM/VRAM/MRAM residency transitions
    - rebinding and release when compute produces updated values

    This is the main implementation of the workspace's `mapv` / `mapv_back`
    stage.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.value_tiles: Dict[str, ValueTile] = {}
        self.scatters: Dict[str, Scatter] = {}
        self.scatter_groups: Dict[str, ScatterGroup] = {}
        # Wide/full tiles bind directly to value tiles. Narrow tiles must resolve
        # through scatter -> scatter_group -> backing value.
        self.full_tile_bindings: Dict[str, str] = {}
        self.value_tile_tensor_refs: Dict[str, set[str]] = {}
        self.tile_scatter_bindings: Dict[str, str] = {}
        self.scatter_group_slots: Dict[str, Dict[Tuple[object, ...], str]] = {}
        self._value_tiles_in_vram: Dict[str, int] = {}
        self._value_tiles_in_mram: Dict[str, int] = {}
        self._value_tiles_in_hbm: Dict[str, object] = {}
        self._mram_fifo: List[str] = []
        self._protected_vram_value_tile_ids: set[str] = set()
        self._value_tile_counter = 0
        self._scatter_counter = 0
        self._scatter_group_counter = 0

    @property
    def bindings(self) -> Dict[str, str]:
        # Compatibility alias for older scaffold/debug helpers.
        return self.full_tile_bindings

    def _next_value_tile_id(self) -> str:
        value_tile_id = f"value_tile.{self._value_tile_counter}"
        self._value_tile_counter += 1
        return value_tile_id

    def _next_scatter_id(self) -> str:
        scatter_id = f"scatter.{self._scatter_counter}"
        self._scatter_counter += 1
        return scatter_id

    def _next_scatter_group_id(self) -> str:
        group_id = f"scatter_group.{self._scatter_group_counter}"
        self._scatter_group_counter += 1
        return group_id

    def mapv(self, signal: List[object]) -> MapvPacket:
        """Resolve one mapped logical packet into concrete value-layer operands.

        Input packets come from TensorManager's `mapt` stage plus residency
        targets and, optionally, one control tag. The function performs late
        source resolution so compute sees the correct runtime object type:

        - wide/full tiles -> ValueTile
        - narrow logical tiles -> Scatter
        - grouped narrow backing tiles -> backing ValueTile

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
        v3 = self._prepare_mapv_destination_value(dst_tile, residency_targets[2])
        return ("matmul", mapped_pairs, v3, dst_tile)

    def _resolve_mapv_source_value(self, tile: TensorTile | InputTile, place: str) -> SourceValueLike:
        if self._is_grouped_narrow_backing_tile(tile):
            group = self._get_or_create_scatter_group_for_tile(tile)
            value = self._resolve_tile_backing_value(tile)
            self.ensure_value_tile_in_place(value, place)
            self.program.operation_log.append(
                {
                    "kind": "mapv_source_grouped_narrow_backing",
                    "tile": tile.tile_id,
                    "scatter_group": group.group_id,
                    "value": value.value_tile_id,
                    "place": place,
                }
            )
            return value
        if self._is_narrow_tensor_tile(tile):
            scatter = self._get_or_create_scatter_for_tile(tile)
            backing_value = self._resolve_tile_backing_value(tile)
            self.ensure_value_tile_in_place(backing_value, place)
            group = self.scatter_groups[scatter.scatter_group_id]
            self.program.operation_log.append(
                {
                    "kind": "mapv_source_narrow",
                    "tile": tile.tile_id,
                    "scatter": scatter.scatter_id,
                    "scatter_group": group.group_id,
                    "backing_value": backing_value.value_tile_id,
                    "place": place,
                }
            )
            return scatter
        value = self._resolve_tile_backing_value(tile)
        self.ensure_value_tile_in_place(value, place)
        return value

    def _resolve_alias_owner_tile(self, tile: TensorTile | InputTile) -> TensorTile | InputTile:
        if not bool(tile.metadata.get("slice_materialized", False)):
            return tile
        source_tile_id = tile.metadata.get("source_tile_id")
        if not isinstance(source_tile_id, str):
            return tile
        owner_tile = self.program.tensor_manager.tensor_tiles.get(source_tile_id)
        if owner_tile is None:
            owner_tile = self.program.tensor_manager.input_tiles.get(source_tile_id)
        if not isinstance(owner_tile, (TensorTile, InputTile)):
            return tile
        return owner_tile

    def map_scatters_to_group(self, scatters: Sequence[object]) -> Optional[ScatterGroup]:
        if not scatters:
            return None
        if not all(isinstance(scatter, Scatter) for scatter in scatters):
            return None

        typed_scatters = [scatter for scatter in scatters if isinstance(scatter, Scatter)]
        if not typed_scatters:
            return None

        group_id = typed_scatters[0].scatter_group_id
        if any(scatter.scatter_group_id != group_id for scatter in typed_scatters[1:]):
            return None

        group = self.scatter_groups.get(group_id)
        if group is None:
            return None

        first = typed_scatters[0]
        expected_row_offset = first.row_offset
        expected_row_count = first.row_count
        expected_col_count = first.col_count
        expected_offsets = [first.col_offset + idx * expected_col_count for idx in range(len(typed_scatters))]

        for scatter, expected_col_offset in zip(typed_scatters, expected_offsets):
            if scatter.row_offset != expected_row_offset:
                return None
            if scatter.row_count != expected_row_count:
                return None
            if scatter.col_count != expected_col_count:
                return None
            if scatter.col_offset != expected_col_offset:
                return None

        return group

    def _prepare_mapv_destination_value(self, tile: TensorTile | InputTile, place: str) -> ValueTile:
        canonical_tile = self._resolve_alias_owner_tile(tile)
        if canonical_tile is not tile and not self._is_narrow_tensor_tile(tile):
            tile = canonical_tile
        if isinstance(tile, TensorTile) and not self._is_narrow_tensor_tile(tile):
            old_value = self.resolve_value_tile(tile)
            old_value_tile_id = self._detach_tile_value_pointer(tile.tile_id)
            if old_value_tile_id is None:
                raise RuntimeError(f"Wide destination tile {tile.tile_id} had no bound value to detach")
            new_value = self.create_value_tile_in_vram(old_value)
            self._attach_tile_value_pointer(tile.tile_id, new_value.value_tile_id)
            self.ensure_value_tile_in_place(new_value, place)
            self.free_value_tile(old_value_tile_id)
            return new_value
        dst_source_value = self.resolve_value_tile(tile)
        value = self.create_value_tile_in_vram(dst_source_value)
        self.ensure_value_tile_in_place(value, place)
        return value

    def prepare_updated_wide_tile_value(
        self,
        tile: TensorTile | InputTile,
        *,
        ensure_old_place: Optional[str] = None,
        new_place: str = "vram",
    ) -> Tuple[ValueTile, ValueTile, str]:
        canonical_tile = self._resolve_alias_owner_tile(tile)
        if canonical_tile is not tile and not self._is_narrow_tensor_tile(tile):
            tile = canonical_tile
        if not isinstance(tile, TensorTile) or self._is_narrow_tensor_tile(tile):
            raise RuntimeError(
                f"prepare_updated_wide_tile_value expects one wide tensor tile, got {type(tile).__name__} {tile.tile_id}"
            )

        old_value = self.resolve_value_tile(tile)
        if ensure_old_place is not None:
            self.ensure_value_tile_in_place(old_value, ensure_old_place)
        old_value_tile_id = self._detach_tile_value_pointer(tile.tile_id)
        if old_value_tile_id is None:
            raise RuntimeError(f"Wide destination tile {tile.tile_id} had no bound value to detach")

        # Keep the old source residency stable while we materialize the new dst value.
        self.protect_value_tile(old_value, "vram")
        try:
            new_value = self.create_value_tile_in_vram(old_value)
        finally:
            self.stop_protect_value_tile(old_value, "vram")
        self._attach_tile_value_pointer(tile.tile_id, new_value.value_tile_id)
        self.ensure_value_tile_in_place(new_value, new_place)
        return old_value, new_value, old_value_tile_id

    def _is_packed_narrow_tile(self, tile: TensorTile | InputTile) -> bool:
        return int(tile.metadata.get("packed_head_count", 1)) > 1 or bool(tile.metadata.get("packed_head_group", False))

    def _is_grouped_narrow_backing_tile(self, tile: TensorTile | InputTile) -> bool:
        return self._is_packed_narrow_tile(tile)

    def _is_narrow_tensor_tile(self, tile: TensorTile | InputTile) -> bool:
        width_class = tile.metadata.get("tile_width_class")
        if width_class == "narrow":
            return True
        if width_class == "full":
            return False
        return int(tile.tile_shape[1]) < int(self.program.mlen)

    def _get_or_create_scatter_for_tile(self, tile: TensorTile | InputTile) -> Scatter:
        existing_scatter_id = self.tile_scatter_bindings.get(tile.tile_id)
        if existing_scatter_id is not None:
            existing_scatter = self.scatters.get(existing_scatter_id)
            if existing_scatter is not None:
                self._attach_tile_to_scatter(tile, existing_scatter)
                return existing_scatter

        group = self._get_or_create_scatter_group_for_tile(tile)
        scatter = self._require_group_scatter_for_tile(group, tile)
        self._attach_tile_to_scatter(tile, scatter)
        return scatter

    def try_map_tile_to_scatter(self, tile: TensorTile | InputTile) -> Optional[Scatter]:
        if self._is_packed_narrow_tile(tile):
            return None
        return self._get_or_create_scatter_for_tile(tile)

    def map_tile_to_scatter(self, tile: TensorTile | InputTile) -> Scatter:
        scatter = self.try_map_tile_to_scatter(tile)
        if scatter is None:
            raise RuntimeError(
                f"Tile {tile.tile_id} does not map to one direct scatter; resolve its scatter group instead"
            )
        return scatter

    def try_map_tile_to_scatter_group(self, tile: TensorTile | InputTile) -> Optional[ScatterGroup]:
        if not self._is_narrow_tensor_tile(tile):
            return None
        if self._is_packed_narrow_tile(tile):
            return self._get_or_create_scatter_group_for_tile(tile)
        scatter = self._get_or_create_scatter_for_tile(tile)
        group = self.scatter_groups.get(scatter.scatter_group_id)
        if group is None:
            raise RuntimeError(f"Tile {tile.tile_id} resolved to missing scatter group {scatter.scatter_group_id}")
        return group

    def map_tile_to_scatter_group(self, tile: TensorTile | InputTile) -> ScatterGroup:
        group = self.try_map_tile_to_scatter_group(tile)
        if group is None:
            raise RuntimeError(f"Wide tile {tile.tile_id} does not map to a scatter group")
        return group

    def resolve_row_operand(self, tile: TensorTile | InputTile, place: str = "vram") -> RowOperandLike:
        if self._is_narrow_tensor_tile(tile):
            group = self._get_or_create_scatter_group_for_tile(tile)
            scatter = self._require_group_scatter_for_tile(group, tile)
            backing_value = self.value_tiles.get(scatter.backing_value_tile_id)
            if not isinstance(backing_value, ValueTile):
                raise RuntimeError(
                    f"Row operand narrow tile {tile.tile_id} resolved to scatter {scatter.scatter_id} without backing value"
                )
            self.ensure_value_tile_in_place(backing_value, place)
            return scatter
        value = self.resolve_value_tile(tile)
        self.ensure_value_tile_in_place(value, place)
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

        group = self._get_or_create_scatter_group_for_tile(tile)
        owner_name = tile.tensor_name if isinstance(tile, TensorTile) else tile.input_name
        row_block_index = int(tile.metadata.get("row_block", tile.coord[0]))
        slot_width = int(tile.metadata.get("scatter_slot_width", overlap_col_count))
        if overlap_col_offset % slot_width != 0 or overlap_col_count % slot_width != 0:
            raise RuntimeError(
                f"Slice overlap for tile {tile.tile_id} is not aligned to scatter slot width {slot_width}: "
                f"offset={overlap_col_offset} count={overlap_col_count}"
            )
        group_head_start = int(tile.metadata.get("group_head_start", tile.metadata.get("head_index", 0)))
        lane_index = overlap_col_offset // slot_width
        slot_head_index = group_head_start + lane_index
        slot_key = (owner_name, slot_head_index, row_block_index, overlap_col_offset, overlap_col_count)
        scatter_id = self.scatter_group_slots.get(group.group_id, {}).get(slot_key)
        if scatter_id is None:
            raise RuntimeError(
                f"Scatter group {group.group_id} did not define slot for tile {tile.tile_id} "
                f"slice key={slot_key}"
            )
        scatter = self.scatters.get(scatter_id)
        if not isinstance(scatter, Scatter):
            raise RuntimeError(f"Scatter group {group.group_id} slot {slot_key} mapped to missing scatter {scatter_id}")
        backing_value = self.value_tiles.get(scatter.backing_value_tile_id)
        if not isinstance(backing_value, ValueTile):
            raise RuntimeError(
                f"Row operand narrow slice for tile {tile.tile_id} resolved to scatter {scatter.scatter_id} without backing value"
            )
        self.ensure_value_tile_in_place(backing_value, place)
        return scatter

    def protect_value_tile(self, value: ValueTile, place: str = "vram") -> None:
        if place != "vram":
            raise ValueError(f"Unsupported protect place: {place}")
        already_protected = value.value_tile_id in self._protected_vram_value_tile_ids
        self._protected_vram_value_tile_ids.add(value.value_tile_id)
        self.program.operation_log.append(
            {
                "kind": "protect_value_tile",
                "place": place,
                "value": value.value_tile_id,
                "already_protected": already_protected,
            }
        )

    def stop_protect_value_tile(self, value: Optional[ValueTile] = None, place: str = "vram") -> None:
        if place != "vram":
            raise ValueError(f"Unsupported protect place: {place}")
        if value is None:
            if not self._protected_vram_value_tile_ids:
                return
            old_value_ids = sorted(self._protected_vram_value_tile_ids)
            self._protected_vram_value_tile_ids.clear()
            self.program.operation_log.append(
                {
                    "kind": "stop_protect_value_tile",
                    "place": place,
                    "old_values": old_value_ids,
                }
            )
            return
        if value.value_tile_id not in self._protected_vram_value_tile_ids:
            return
        self._protected_vram_value_tile_ids.remove(value.value_tile_id)
        self.program.operation_log.append(
            {
                "kind": "stop_protect_value_tile",
                "place": place,
                "old_value": value.value_tile_id,
            }
        )

    def _is_protected_value_tile(self, value_tile_id: str, place: str = "vram") -> bool:
        if place != "vram":
            return False
        return value_tile_id in self._protected_vram_value_tile_ids

    def create_transient_scatter_group_like(self, template_group: ScatterGroup) -> ScatterGroup:
        template_value = self.value_tiles.get(template_group.backing_value_tile_id)
        if not isinstance(template_value, ValueTile):
            raise RuntimeError(
                f"Transient scatter-group clone requires template backing value for {template_group.group_id}"
            )
        temp_value = ValueTile(
            value_tile_id=self._next_value_tile_id(),
            logical_shape=template_value.logical_shape,
            from_input_tile=template_value.from_input_tile,
            source_input_tile_id=template_value.source_input_tile_id,
            metadata={**dict(template_value.metadata), "transient": True},
        )
        vram_name = f"{temp_value.value_tile_id}.vram"
        vram_addr = self.allocate_value_tile_address(
            size=self.program.tile_elems,
            name=vram_name,
            place="vram",
            value_tile=temp_value,
        )
        temp_value.residency["vram_addr"] = vram_addr
        temp_value.residency["vram_name"] = vram_name
        self.value_tiles[temp_value.value_tile_id] = temp_value
        self._value_tiles_in_vram[temp_value.value_tile_id] = vram_addr
        self._touch_fifo("vram", temp_value.value_tile_id)
        self.program.emit_zero_vram_tile(int(vram_addr))
        temp_value.metadata["transient"] = True
        group = ScatterGroup(
            group_id=self._next_scatter_group_id(),
            backing_value_tile_id=temp_value.value_tile_id,
            scatter_ids=[],
            metadata={
                "template_group_id": template_group.group_id,
                "transient": True,
            },
        )
        self.scatter_groups[group.group_id] = group
        self.scatter_group_slots[group.group_id] = {}

        template_slot_map = self.scatter_group_slots.get(template_group.group_id, {})
        for slot_key, template_scatter_id in sorted(template_slot_map.items(), key=lambda item: item[0]):
            template_scatter = self.scatters.get(template_scatter_id)
            if template_scatter is None:
                raise RuntimeError(
                    f"Transient scatter-group clone references missing scatter {template_scatter_id}"
                )
            scatter = Scatter(
                scatter_id=self._next_scatter_id(),
                backing_value_tile_id=temp_value.value_tile_id,
                scatter_group_id=group.group_id,
                row_offset=template_scatter.row_offset,
                row_count=template_scatter.row_count,
                col_offset=template_scatter.col_offset,
                col_count=template_scatter.col_count,
                metadata={
                    "slot_key": slot_key,
                    "slot_shape": template_scatter.metadata.get("slot_shape"),
                    "template_scatter_id": template_scatter.scatter_id,
                    "transient": True,
                },
            )
            self.scatters[scatter.scatter_id] = scatter
            self.scatter_group_slots[group.group_id][slot_key] = scatter.scatter_id
            group.scatter_ids.append(scatter.scatter_id)
        self.program.operation_log.append(
            {
                "kind": "create_transient_scatter_group",
                "group": group.group_id,
                "template_group": template_group.group_id,
                "backing_value": temp_value.value_tile_id,
                "slot_count": len(group.scatter_ids),
            }
        )
        return group

    def _require_group_scatter_for_tile(self, group: ScatterGroup, tile: TensorTile | InputTile) -> Scatter:
        slot_map = self.scatter_group_slots.get(group.group_id, {})
        slot_key = self._scatter_slot_key_for_tile(tile)
        scatter_id = slot_map.get(slot_key)
        if scatter_id is None:
            raise RuntimeError(
                f"Scatter group {group.group_id} did not define a scatter slot for tile {tile.tile_id} key={slot_key}"
            )
        scatter = self.scatters.get(scatter_id)
        if scatter is None:
            raise RuntimeError(
                f"Scatter binding for tile {tile.tile_id} points to missing scatter {scatter_id}"
            )
        if scatter.scatter_group_id != group.group_id:
            raise RuntimeError(
                f"Scatter/group mismatch for tile {tile.tile_id}: "
                f"scatter={scatter.scatter_id} group={scatter.scatter_group_id} expected_group={group.group_id}"
            )
        return scatter

    def _create_group_scatter(
        self,
        group: ScatterGroup,
        tile: TensorTile | InputTile,
    ) -> Scatter:
        slot_key = self._scatter_slot_key_for_tile(tile)
        scatter = Scatter(
            scatter_id=self._next_scatter_id(),
            backing_value_tile_id=group.backing_value_tile_id,
            scatter_group_id=group.group_id,
            row_offset=0,
            row_count=int(self.program.mlen),
            col_offset=int(tile.coord[1] * self.program.mlen),
            col_count=int(tile.tile_shape[1]),
            metadata={
                "owner_tile_id": tile.tile_id,
                "tile_coord": tile.coord,
                "slot_key": slot_key,
                "slot_shape": (self.program.mlen, int(tile.tile_shape[1])),
            },
        )
        self.scatters[scatter.scatter_id] = scatter
        self.scatter_group_slots.setdefault(group.group_id, {})[slot_key] = scatter.scatter_id
        group.scatter_ids.append(scatter.scatter_id)
        self.program.operation_log.append(
            {
                "kind": "create_scatter",
                "scatter": scatter.scatter_id,
                "group": group.group_id,
                "backing_value": scatter.backing_value_tile_id,
                "tile": tile.tile_id,
                "col_count": scatter.col_count,
            }
        )
        return scatter

    def _attach_tile_to_scatter(self, tile: TensorTile | InputTile, scatter: Scatter) -> None:
        old_scatter_id = self.tile_scatter_bindings.get(tile.tile_id)
        if old_scatter_id is not None and old_scatter_id != scatter.scatter_id:
            self._release_scatter_binding(tile.tile_id)
        group_id = scatter.scatter_group_id
        self.tile_scatter_bindings[tile.tile_id] = scatter.scatter_id
        self._unbind_tile_value_pointer(tile.tile_id)

    def _get_or_create_scatter_group_for_tile(self, tile: TensorTile | InputTile) -> ScatterGroup:
        group_key = self._scatter_group_key_for_tile(tile)
        for group in self.scatter_groups.values():
            if group.metadata.get("group_key") == group_key:
                if not bool(tile.metadata.get("slice_materialized", False)):
                    self._populate_scatter_group(group, tile)
                return group

        if self._is_packed_narrow_tile(tile):
            backing_value = self._create_value_tile_for_tile(tile, bind_tile_pointer=False)
        else:
            backing_value = self.resolve_value_tile(tile)
        group = ScatterGroup(
            group_id=self._next_scatter_group_id(),
            backing_value_tile_id=backing_value.value_tile_id,
            scatter_ids=[],
            metadata={"group_key": group_key},
        )
        self.scatter_groups[group.group_id] = group
        self.scatter_group_slots[group.group_id] = {}
        self.program.operation_log.append(
            {
                "kind": "create_scatter_group",
                "group": group.group_id,
                "backing_value": backing_value.value_tile_id,
                "tile": tile.tile_id,
            }
        )
        self._populate_scatter_group(group, tile)
        return group

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
            hbm_stride = int(tile.metadata.get("logical_shape", (0, 0, 0, 0))[-1] * tile.metadata.get("heads", 1))
            hbm_offset = tile.coord[0] * self.program.mlen * hbm_stride + tile.coord[1] * self.program.mlen
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

    def _populate_scatter_group(self, group: ScatterGroup, tile: TensorTile | InputTile) -> None:
        slot_map = self.scatter_group_slots.setdefault(group.group_id, {})
        if self._is_packed_narrow_tile(tile):
            packed_heads = int(tile.metadata.get("packed_head_count", 0))
            slot_width = int(tile.metadata.get("scatter_slot_width", 0))
            group_head_start = int(tile.metadata.get("group_head_start", tile.metadata.get("head_index", 0)))
            owner_name = tile.tensor_name if isinstance(tile, TensorTile) else tile.input_name
            row_block = int(tile.metadata.get("row_block", tile.coord[0]))
            for lane in range(packed_heads):
                slot_head_index = group_head_start + lane
                slot_key = (owner_name, slot_head_index, row_block, lane * slot_width, slot_width)
                if slot_key in slot_map:
                    continue
                scatter = Scatter(
                    scatter_id=self._next_scatter_id(),
                    backing_value_tile_id=group.backing_value_tile_id,
                    scatter_group_id=group.group_id,
                    row_offset=0,
                    row_count=int(self.program.mlen),
                    col_offset=lane * slot_width,
                    col_count=slot_width,
                    metadata={
                        "owner_tile_id": tile.tile_id,
                        "tile_coord": tile.coord,
                        "slot_key": slot_key,
                        "slot_shape": (self.program.mlen, slot_width),
                        "grouped_narrow_lane": lane,
                        "group_head_start": group_head_start,
                    },
                )
                self.scatters[scatter.scatter_id] = scatter
                slot_map[slot_key] = scatter.scatter_id
                group.scatter_ids.append(scatter.scatter_id)
                self.program.operation_log.append(
                    {
                        "kind": "create_scatter",
                        "scatter": scatter.scatter_id,
                        "group": group.group_id,
                        "backing_value": scatter.backing_value_tile_id,
                        "tile": tile.tile_id,
                        "col_count": scatter.col_count,
                        "grouped_narrow_lane": lane,
                    }
                )
            return
        for candidate in self._iter_group_tiles(tile):
            slot_key = self._scatter_slot_key_for_tile(candidate)
            if slot_key in slot_map:
                continue
            self._create_group_scatter(group, candidate)

    def _iter_group_tiles(self, tile: TensorTile | InputTile) -> List[TensorTile | InputTile]:
        owner_tiles = self._owner_tiles_for_tile(tile)
        group_key = self._scatter_group_key_for_tile(tile)
        candidates: List[TensorTile | InputTile] = []
        for candidate in _tiles_in_grid_order(owner_tiles):
            if not isinstance(candidate, (TensorTile, InputTile)):
                continue
            if not self._is_narrow_tensor_tile(candidate):
                continue
            if self._scatter_group_key_for_tile(candidate) != group_key:
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

    def _ensure_scatter_in_place(self, scatter: Scatter, place: str) -> ValueTile:
        backing_value = self.value_tiles[scatter.backing_value_tile_id]
        self.ensure_value_tile_in_place(backing_value, place)
        group = self.scatter_groups.get(scatter.scatter_group_id)
        self.program.operation_log.append(
            {
                "kind": "ensure_scatter_in_place",
                "scatter": scatter.scatter_id,
                "scatter_group": scatter.scatter_group_id,
                "backing_value": backing_value.value_tile_id,
                "place": place,
                "group_scatter_count": len(group.scatter_ids) if group is not None else 0,
            }
        )
        return backing_value

    def _scatter_group_key_for_tile(self, tile: TensorTile | InputTile) -> Tuple[object, ...]:
        owner_name = tile.tensor_name if isinstance(tile, TensorTile) else tile.input_name
        if bool(tile.metadata.get("packed_head_group", False)):
            head_index = int(tile.metadata.get("group_head_start", tile.metadata.get("head_index", 0)))
        else:
            head_index = int(tile.metadata.get("head_index", 0))
        row_block = int(tile.metadata.get("row_block", tile.coord[0]))
        return (owner_name, head_index, row_block)

    def _scatter_slot_key_for_tile(self, tile: TensorTile | InputTile) -> Tuple[object, ...]:
        owner_name = tile.tensor_name if isinstance(tile, TensorTile) else tile.input_name
        head_index = int(tile.metadata.get("slot_head_index", tile.metadata.get("head_index", 0)))
        row_block = int(tile.metadata.get("row_block", tile.coord[0]))
        col_offset = int(tile.metadata.get("scatter_col_offset", tile.coord[1] * self.program.mlen))
        col_count = int(tile.metadata.get("scatter_col_count", tile.tile_shape[1]))
        return (owner_name, head_index, row_block, col_offset, col_count)

    def _split_mapv_signal(self, items: List[object]) -> Tuple[List[List[object]], Optional[TensorTile]]:
        pair_groups: List[List[object]] = []
        dst_tile: Optional[TensorTile] = None
        for item in items:
            if isinstance(item, list) and len(item) == 2 and all(_is_tile_object(part) for part in item):
                pair_groups.append(item)
                continue
            if isinstance(item, list) and len(item) == 1 and isinstance(item[0], TensorTile):
                dst_tile = item[0]
                continue
        return pair_groups, dst_tile

    def _resolve_tile_backing_value(self, tile: TensorTile | InputTile) -> ValueTile:
        canonical_tile = self._resolve_alias_owner_tile(tile)
        if canonical_tile is not tile and not self._is_narrow_tensor_tile(tile):
            tile = canonical_tile
        if self._is_packed_narrow_tile(tile):
            group = self._get_or_create_scatter_group_for_tile(tile)
            value = self.value_tiles.get(group.backing_value_tile_id)
            if not isinstance(value, ValueTile):
                raise RuntimeError(
                    f"Grouped-narrow tile {tile.tile_id} resolved to scatter group {group.group_id} "
                    f"without backing value {group.backing_value_tile_id}"
                )
            return value
        if self._is_narrow_tensor_tile(tile):
            scatter = self._get_or_create_scatter_for_tile(tile)
            value = self.value_tiles.get(scatter.backing_value_tile_id)
            if not isinstance(value, ValueTile):
                raise RuntimeError(
                    f"Narrow tile {tile.tile_id} resolved to scatter {scatter.scatter_id} "
                    f"without backing value {scatter.backing_value_tile_id}"
                )
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

    def _value_tile_has_live_refs(self, value_tile_id: str) -> bool:
        if self.value_tile_tensor_refs.get(value_tile_id):
            return True
        if self._value_tile_has_scatter_group_refs(value_tile_id):
            return True
        return False

    def create_value_tile_in_vram(self, value: Optional[ValueTile] = None) -> ValueTile:
        new_value_tile = ValueTile(
            value_tile_id=self._next_value_tile_id(),
            logical_shape=value.logical_shape if value is not None else (self.program.mlen, self.program.mlen),
            metadata=dict(value.metadata) if value is not None else {},
        )
        if value is not None:
            new_value_tile.from_input_tile = value.from_input_tile
            new_value_tile.source_input_tile_id = value.source_input_tile_id
            if (
                self._value_tile_has_live_refs(value.value_tile_id)
                and (
                    value.residency.get("vram_addr") is not None
                    or value.residency.get("hbm_addr") is not None
                    or value.residency.get("hbm_ready")
                )
            ):
                self.ensure_value_tile_in_place(value, "hbm")
            if (
                value.residency.get("vram_addr") is not None
                and not self._is_protected_value_tile(value.value_tile_id, "vram")
            ):
                new_value_tile.residency["vram_addr"] = value.residency.pop("vram_addr")
                new_value_tile.residency["vram_name"] = value.residency.pop("vram_name", None)
                new_value_tile.residency["vram_owner_from"] = value.value_tile_id
                old_addr = self._value_tiles_in_vram.pop(value.value_tile_id, None)
                if old_addr is not None:
                    self._value_tiles_in_vram[new_value_tile.value_tile_id] = old_addr
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
        self.program.operation_log.append(
            {
                "kind": "create_value_tile_in_vram",
                "new_value": new_value_tile.value_tile_id,
                "source_value": getattr(value, "value_tile_id", None),
                "vram_addr": new_value_tile.residency.get("vram_addr"),
            }
        )
        return new_value_tile

    def _detach_scatter_group_backing_value(self, group: ScatterGroup) -> str:
        old_value_tile_id = group.backing_value_tile_id
        if not old_value_tile_id:
            raise RuntimeError(f"Scatter group {group.group_id} has no backing value to detach")
        group.backing_value_tile_id = ""
        for scatter_id in group.scatter_ids:
            scatter = self.scatters.get(scatter_id)
            if scatter is None:
                continue
            scatter.backing_value_tile_id = ""
        self.program.operation_log.append(
            {
                "kind": "detach_scatter_group_backing_value",
                "scatter_group": group.group_id,
                "old_value": old_value_tile_id,
            }
        )
        return old_value_tile_id

    def _attach_scatter_group_backing_value(self, group: ScatterGroup, new_value: ValueTile) -> None:
        group.backing_value_tile_id = new_value.value_tile_id
        for scatter_id in group.scatter_ids:
            scatter = self.scatters.get(scatter_id)
            if scatter is None:
                continue
            scatter.backing_value_tile_id = new_value.value_tile_id
        self.program.operation_log.append(
            {
                "kind": "attach_scatter_group_backing_value",
                "scatter_group": group.group_id,
                "new_value": new_value.value_tile_id,
            }
        )

    def rebind_scatter_group_backing_value(self, group: ScatterGroup, new_value: ValueTile) -> None:
        old_value_tile_id = group.backing_value_tile_id
        if old_value_tile_id == new_value.value_tile_id:
            return
        if old_value_tile_id:
            old_value_tile_id = self._detach_scatter_group_backing_value(group)
        self._attach_scatter_group_backing_value(group, new_value)
        if old_value_tile_id:
            self.program.operation_log.append(
                {
                    "kind": "rebind_scatter_group_backing_value",
                    "scatter_group": group.group_id,
                    "old_value": old_value_tile_id,
                    "new_value": new_value.value_tile_id,
                }
            )
            self.free_value_tile(old_value_tile_id)

    def prepare_updated_scatter_group_backing(
        self,
        group: ScatterGroup,
        *,
        ensure_old_vram: bool = False,
    ) -> Tuple[ValueTile, ValueTile]:
        old_value = self.value_tiles.get(group.backing_value_tile_id)
        if not isinstance(old_value, ValueTile):
            raise RuntimeError(
                f"Scatter group {group.group_id} is missing backing value {group.backing_value_tile_id}"
            )
        if ensure_old_vram:
            self.ensure_value_tile_in_place(old_value, "vram")
        old_value_tile_id = self._detach_scatter_group_backing_value(group)
        new_value = self.create_value_tile_in_vram(old_value)
        self._attach_scatter_group_backing_value(group, new_value)
        self.program.operation_log.append(
            {
                "kind": "rebind_scatter_group_backing_value",
                "scatter_group": group.group_id,
                "old_value": old_value_tile_id,
                "new_value": new_value.value_tile_id,
            }
        )
        self.free_value_tile(old_value_tile_id)
        return old_value, new_value

    def prepare_updated_dst_operand(
        self,
        dst: Scatter | ScatterGroup,
    ) -> Tuple[ScatterGroup, ValueTile]:
        if isinstance(dst, ScatterGroup):
            _, new_value = self.prepare_updated_scatter_group_backing(dst)
            return dst, new_value
        dst_group = self.scatter_groups.get(dst.scatter_group_id)
        if dst_group is None:
            raise RuntimeError(f"Destination scatter {dst.scatter_id} is missing group {dst.scatter_group_id}")
        _, new_value = self.prepare_updated_scatter_group_backing(dst_group, ensure_old_vram=True)
        return dst_group, new_value

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
        self.program.operation_log.append(
            {
                "kind": "evaluate_contiguous_vram_value_tile_window",
                "reason": reason,
                "tile_count": tile_count,
                "window_size": window_size,
                "chosen_kind": chosen["kind"],
                "chosen_addr": chosen["addr"],
                "chosen_cost": chosen["cost"],
                "candidate_count": len(candidates),
            }
        )
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

        self.program.operation_log.append(
            {
                "kind": "allocate_contiguous_vram_value_tiles",
                "reason": reason,
                "alloc_name": alloc_name,
                "base_addr": base_addr,
                "tile_count": tile_count,
                "value_tiles": [value.value_tile_id for value in reserved_values],
            }
        )
        return reserved_values, base_addr

    def ensure_value_tile_in_place(self, value: ValueTile, place: str) -> ValueTile:
        if place == "vram":
            if value.residency.get("vram_addr") is not None:
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
        if place == "hbm":
            if value.residency.get("hbm_ready"):
                self._value_tiles_in_hbm[value.value_tile_id] = True
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
            return value
        raise ValueError(f"Unsupported place for ensure_value_tile_in_place: {place}")

    def move_tile(self, value: ValueTile, src_place: str, dst_place: str) -> None:
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
            self.program.operation_log.append(
                {
                    "kind": "move_tile",
                    "value": value.value_tile_id,
                    "src": "vram",
                    "dst": "hbm",
                    "vram_addr": vram_addr,
                    "hbm_name": hbm_name,
                    "hbm_addr": hbm_addr,
                    "hbm_base_addr": hbm_params["hbm_base_addr"],
                    "hbm_start_offset": hbm_params["hbm_offset"],
                    "hbm_stride": hbm_params["hbm_stride"],
                    "hbm_scale_size": hbm_params["hbm_scale_size"],
                    "from_input_tile": value.from_input_tile,
                    "source_input_tile_id": value.source_input_tile_id,
                }
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
            self.program.operation_log.append(
                {
                    "kind": "move_tile",
                    "value": value.value_tile_id,
                    "src": "hbm",
                    "dst": "vram",
                    "hbm_addr": hbm_addr,
                    "hbm_base_addr": hbm_params["hbm_base_addr"],
                    "vram_addr": vram_addr,
                    "hbm_name": hbm_name,
                    "hbm_start_offset": hbm_params["hbm_offset"],
                    "hbm_stride": hbm_params["hbm_stride"],
                    "hbm_scale_size": hbm_params["hbm_scale_size"],
                    "from_input_tile": value.from_input_tile,
                    "source_input_tile_id": value.source_input_tile_id,
                }
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
            self.program.operation_log.append(
                {
                    "kind": "move_tile",
                    "value": value.value_tile_id,
                    "src": "hbm",
                    "dst": "mram",
                    "hbm_addr": hbm_addr,
                    "hbm_base_addr": hbm_params["hbm_base_addr"],
                    "mram_addr": mram_addr,
                    "hbm_name": hbm_params["hbm_name"],
                    "hbm_offset": hbm_params["hbm_offset"],
                    "hbm_stride": hbm_params["hbm_stride"],
                    "hbm_scale_size": hbm_params["hbm_scale_size"],
                    "from_input_tile": value.from_input_tile,
                    "source_input_tile_id": value.source_input_tile_id,
                }
            )
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
                if len(logical_shape) == 4:
                    _, _, heads, head_dim = logical_shape
                    hbm_stride = int(heads * head_dim)
                else:
                    hbm_stride = self.program.mlen
                hbm_offset = input_tile.coord[0] * self.program.mlen * hbm_stride + input_tile.coord[1] * self.program.mlen
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
        hbm_scale_size = int(hbm_shape[0]) * int(hbm_shape[1])
        hbm_offset = int(value.residency.get("hbm_offset", int(hbm_addr) - hbm_base_addr))
        hbm_stride = int(value.residency.get("hbm_stride", self.program.mlen))
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
            self.program.operation_log.append(
                {
                    "kind": "allocate_value_tile_address",
                    "place": "vram",
                    "name": name,
                    "size": size,
                    "addr": addr,
                    "value": getattr(value_tile, "value_tile_id", None),
                }
            )
            return addr
        if place == "mram":
            self._evict_fifo_if_needed("mram")
            if value_tile is not None:
                self._touch_fifo("mram", value_tile.value_tile_id)
            addr = self.program.compiler.sub_matrix_manager.mram_allocator.allocate(name=name, size=size)
            self.program.operation_log.append(
                {
                    "kind": "allocate_value_tile_address",
                    "place": "mram",
                    "name": name,
                    "size": size,
                    "addr": addr,
                    "value": getattr(value_tile, "value_tile_id", None),
                }
            )
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
                self._value_tiles_in_hbm[value_tile.value_tile_id] = {
                    "addr": addr,
                    "name": resolved_name,
                    "offset": int(hbm_offset),
                    "stride": self.program.mlen if hbm_stride is None else int(hbm_stride),
                    "scale_size": hbm_scale_size,
                }
                value_tile.residency["hbm_scale_size"] = hbm_scale_size
            self.program.operation_log.append(
                {
                    "kind": "allocate_value_tile_address",
                    "place": "hbm",
                    "name": resolved_name,
                    "size": size,
                    "addr": addr,
                    "offset": int(hbm_offset),
                    "stride": self.program.mlen if hbm_stride is None else int(hbm_stride),
                    "scale_size": hbm_scale_size,
                    "value": getattr(value_tile, "value_tile_id", None),
                }
            )
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
                self.program.operation_log.append(
                    {
                        "kind": "fifo_evict",
                        "place": place,
                        "value": evict_value.value_tile_id,
                        "alloc_name": alloc_name,
                        "addr": evict_value.residency.get(addr_key),
                    }
                )
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
            self.program.operation_log.append(
                {
                    "kind": "fifo_evict",
                    "place": place,
                    "value": evict_value.value_tile_id,
                    "alloc_name": alloc_name,
                    "addr": evict_value.residency.get(addr_key),
                }
            )
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
        input_obj = self.program.tensor_manager.inputs.get(dst_tile.input_name)
        if input_obj is None:
            raise RuntimeError(f"Unknown input owner for input tile {dst_tile.tile_id}: {dst_tile.input_name}")
        hbm_name = input_obj.metadata.get("hbm_group_obj", f"{dst_tile.input_name}.hbm")
        logical_shape = tuple(dst_tile.metadata.get("logical_shape", ()))
        if len(logical_shape) == 4:
            _, _, heads, head_dim = logical_shape
            hbm_stride = int(heads * head_dim)
        else:
            hbm_stride = self.program.mlen
        hbm_offset = dst_tile.coord[0] * self.program.mlen * hbm_stride + dst_tile.coord[1] * self.program.mlen
        hbm_object = self.program.hardware.hbm_objects.get(str(hbm_name))
        if hbm_object is None:
            raise RuntimeError(f"Unknown HBM object for input writeback: {hbm_name}")
        hbm_shape = tuple(hbm_object.get("shape", (self.program.mlen, self.program.mlen)))
        hbm_addr = int(hbm_object["base_addr"]) + int(hbm_offset)
        prev_hbm_name = value.residency.get("hbm_name")
        prev_hbm_addr = value.residency.get("hbm_addr")
        prev_hbm_offset = value.residency.get("hbm_offset")
        prev_hbm_stride = value.residency.get("hbm_stride")
        value.residency["hbm_addr"] = hbm_addr
        value.residency["hbm_name"] = str(hbm_name)
        value.residency["hbm_offset"] = hbm_offset
        value.residency["hbm_stride"] = hbm_stride
        value.residency["hbm_scale_size"] = int(hbm_shape[0]) * int(hbm_shape[1])
        target_changed = (
            prev_hbm_name != value.residency["hbm_name"]
            or prev_hbm_addr != value.residency["hbm_addr"]
            or prev_hbm_offset != value.residency["hbm_offset"]
            or prev_hbm_stride != value.residency["hbm_stride"]
        )
        if target_changed:
            # A value may already be "hbm_ready" in a temporary spill object.
            # Final output writeback must retarget and actually store into the
            # destination input/output HBM object instead of early-returning.
            value.residency["hbm_ready"] = False
        if value.residency.get("vram_addr") is not None:
            self.move_tile(value, "vram", "hbm")
            value.residency["hbm_ready"] = True
        else:
            self.ensure_value_tile_in_place(value, "hbm")
        self._value_tiles_in_hbm[value.value_tile_id] = {
            "addr": value.residency.get("hbm_addr"),
            "name": value.residency.get("hbm_name"),
            "offset": value.residency.get("hbm_offset"),
            "stride": value.residency.get("hbm_stride"),
            "scale_size": value.residency.get("hbm_scale_size"),
        }
        if self._is_narrow_tensor_tile(dst_tile):
            dst_group = self._get_or_create_scatter_group_for_tile(dst_tile)
            self.rebind_scatter_group_backing_value(dst_group, value)
        else:
            self._bind_tile_pointer(dst_tile.tile_id, value.value_tile_id)
        value.metadata["input_writeback_tile_id"] = dst_tile.tile_id
        value.metadata["input_writeback_name"] = dst_tile.input_name
        self.program.operation_log.append(
            {
                "kind": "input_writeback",
                "value": value.value_tile_id,
                "dst_tile": dst_tile.tile_id,
                "dst_input": dst_tile.input_name,
                "hbm_name": value.residency.get("hbm_name"),
                "hbm_addr": value.residency.get("hbm_addr"),
                "hbm_offset": value.residency.get("hbm_offset"),
                "hbm_stride": value.residency.get("hbm_stride"),
            }
        )

    def _bind_value_to_tensor_tile(self, value: ValueTile, dst_tile: TensorTile) -> None:
        canonical_tile = self._resolve_alias_owner_tile(dst_tile)
        if isinstance(canonical_tile, TensorTile) and canonical_tile is not dst_tile and not self._is_narrow_tensor_tile(dst_tile):
            dst_tile = canonical_tile
        if self._is_narrow_tensor_tile(dst_tile):
            dst_group = self._get_or_create_scatter_group_for_tile(dst_tile)
            self.rebind_scatter_group_backing_value(dst_group, value)
            return
        self._bind_tile_pointer(dst_tile.tile_id, value.value_tile_id)

    def _bind_tile_pointer(self, tile_id: str, value_tile_id: str) -> None:
        tile_obj = self.program.tensor_manager.tensor_tiles.get(tile_id) or self.program.tensor_manager.input_tiles.get(tile_id)
        if tile_obj is not None and self._is_narrow_tensor_tile(tile_obj):
            raise RuntimeError(
                f"Narrow tensor tile {tile_id} must bind to scatter only, not directly to value tile {value_tile_id}"
            )
        self._release_scatter_binding(tile_id)
        old_value_tile_id = self.full_tile_bindings.get(tile_id)
        if old_value_tile_id == value_tile_id:
            self.value_tile_tensor_refs.setdefault(value_tile_id, set()).add(tile_id)
            return
        if old_value_tile_id is not None:
            detached_old_value_tile_id = self._detach_tile_value_pointer(tile_id)
            self._attach_tile_value_pointer(tile_id, value_tile_id)
            if detached_old_value_tile_id is not None:
                self.program.operation_log.append(
                    {
                        "kind": "rebind_tile_pointer",
                        "tile": tile_id,
                        "old_value": detached_old_value_tile_id,
                        "new_value": value_tile_id,
                    }
                )
                self.free_value_tile(detached_old_value_tile_id)
            return
        self._attach_tile_value_pointer(tile_id, value_tile_id)

    def _attach_tile_value_pointer(self, tile_id: str, value_tile_id: str) -> None:
        self.full_tile_bindings[tile_id] = value_tile_id
        self.value_tile_tensor_refs.setdefault(value_tile_id, set()).add(tile_id)
        self.program.operation_log.append(
            {
                "kind": "attach_tile_value_pointer",
                "tile": tile_id,
                "new_value": value_tile_id,
            }
        )

    def _detach_tile_value_pointer(self, tile_id: str) -> Optional[str]:
        old_value_tile_id = self.full_tile_bindings.pop(tile_id, None)
        if old_value_tile_id is None:
            return None
        old_refs = self.value_tile_tensor_refs.get(old_value_tile_id)
        if old_refs is not None:
            old_refs.discard(tile_id)
            if not old_refs:
                self.value_tile_tensor_refs.pop(old_value_tile_id, None)
        self.program.operation_log.append(
            {
                "kind": "detach_tile_value_pointer",
                "tile": tile_id,
                "old_value": old_value_tile_id,
            }
        )
        return old_value_tile_id

    def _unbind_tile_value_pointer(self, tile_id: str) -> None:
        old_value_tile_id = self._detach_tile_value_pointer(tile_id)
        if old_value_tile_id is None:
            return
        self.free_value_tile(old_value_tile_id)

    def _release_scatter_binding(self, tile_id: str) -> None:
        scatter_id = self.tile_scatter_bindings.pop(tile_id, None)
        if scatter_id is None:
            return
        scatter = self.scatters.get(scatter_id)
        group_id = scatter.scatter_group_id if scatter is not None else None
        self.program.operation_log.append(
            {
                "kind": "release_scatter_binding",
                "tile": tile_id,
                "scatter": scatter_id,
                "scatter_group": group_id,
            }
        )
        if group_id is not None:
            self._release_scatter_group(group_id)

    def _release_scatter_group(self, group_id: str) -> None:
        group = self.scatter_groups.get(group_id)
        if group is None:
            return
        if self._scatter_group_has_tile_refs(group_id):
            return
        backing_value_tile_id = group.backing_value_tile_id
        scatter_ids = list(group.scatter_ids)
        slot_map = self.scatter_group_slots.pop(group_id, {})
        for scatter_id in scatter_ids:
            scatter = self.scatters.pop(scatter_id, None)
            if scatter is None:
                continue
            slot_key = scatter.metadata.get("slot_key")
            if slot_key is not None and slot_map.get(slot_key) == scatter_id:
                slot_map.pop(slot_key, None)
            self.program.operation_log.append(
                {
                    "kind": "free_scatter",
                    "scatter": scatter_id,
                    "scatter_group": group_id,
                    "backing_value": scatter.backing_value_tile_id,
                }
            )
        group.scatter_ids.clear()
        self.scatter_groups.pop(group_id, None)
        self.program.operation_log.append(
            {
                "kind": "free_scatter_group",
                "scatter_group": group_id,
                "backing_value": backing_value_tile_id,
            }
        )
        self.free_value_tile(backing_value_tile_id)

    def _value_tile_has_scatter_group_refs(self, value_tile_id: str) -> bool:
        for group in self.scatter_groups.values():
            if group.backing_value_tile_id == value_tile_id:
                return True
        return False

    def _scatter_group_has_tile_refs(self, group_id: str) -> bool:
        for scatter_id in self.tile_scatter_bindings.values():
            scatter = self.scatters.get(scatter_id)
            if scatter is not None and scatter.scatter_group_id == group_id:
                return True
        return False

    def free_value_tile(self, value_tile_id: str) -> None:
        value = self.value_tiles.get(value_tile_id)
        if value is None:
            return
        if self.value_tile_tensor_refs.get(value_tile_id):
            return
        if self._value_tile_has_scatter_group_refs(value_tile_id):
            return
        self.program.operation_log.append(
            {
                "kind": "free_value_tile",
                "value": value_tile_id,
                "vram_addr": value.residency.get("vram_addr"),
                "mram_addr": value.residency.get("mram_addr"),
                "hbm_addr": value.residency.get("hbm_addr"),
            }
        )
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
        self.value_tiles.pop(value_tile_id, None)


class TensorManager:
    """Manage logical tensors, tiles, slices, and tensor-thread grouping.

    TensorManager operates on logical objects only. It owns shape flattening,
    tile metadata, slice resolution, and `mapt` grouping. It deliberately does
    not create ValueTile / Scatter / ScatterGroup objects and does not decide
    residency placement; that work stays in ValueManager.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.inputs: Dict[str, Input] = {}
        self.tensors: Dict[str, Tensor] = {}
        self.fp_fragments: Dict[str, FPFragment] = {}
        self.input_tiles: Dict[str, InputTile] = {}
        self.tensor_tiles: Dict[str, TensorTile] = {}
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
            self.program.operation_log.append(
                {
                    "kind": "compat_fp_var_fragment",
                    "name": name,
                    "size": int(size),
                    "fragment_shape": fragment.shape,
                }
            )
            return fragment
        if name in self.fp_vars:
            raise ValueError(f"FPVar {name!r} already declared")
        addr = self._next_fp_mem_addr
        self._next_fp_mem_addr += 1
        var = FPVar(name=name, fp_mem_addr=addr)
        self.fp_vars[name] = var
        self._fp_mem_values.append(float(value))
        self.program.operation_log.append(
            {
                "kind": "alloc_fp_var",
                "name": name,
                "fp_mem_addr": addr,
                "value": float(value),
            }
        )
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
        self.program.operation_log.append(
            {
                "kind": "alloc_fp_fragment",
                "name": name,
                "shape": normalized_shape,
                "init": float(init),
                "cell_count": len(fragment.vars),
            }
        )
        return fragment

    def alloc_fragment(
        self,
        name: str,
        logical_shape: LogicalShape,
        *,
        init_zero: bool = False,
        dtype: str = "fp32",
    ) -> Tensor | FPFragment:
        if len(logical_shape) == 4:
            tensor = self.tensor(name, logical_shape)
            tensor.metadata["fragment_kind"] = "tensor"
            tensor.metadata["dtype"] = dtype
            tensor.metadata["init_zero"] = bool(init_zero)
            return tensor
        if len(logical_shape) == 3:
            fragment = self.fp_fragment(name=name, shape=logical_shape, init=0.0, dtype=dtype)
            fragment.metadata["fragment_kind"] = "fp"
            fragment.metadata["init_zero"] = bool(init_zero)
            return fragment
        raise NotImplementedError(
            f"alloc_fragment supports 4D tensor fragments and 3D fp fragments only, got {logical_shape}"
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
        if isinstance(operand, (list, tuple)):
            resolved: List[FPVar] = []
            for item in operand:
                resolved.extend(self.mapf(item))
            return resolved
        raise NotImplementedError(f"Unsupported operand for mapf: {type(operand).__name__}")

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
        self.program.operation_log.append(
            {
                "kind": "mapf_t",
                "control": control,
                "tensor_group_count": len(tensor_tiles),
                "fp_var_count": len(fp_vars),
            }
        )
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
        else:
            metadata["layout"] = "2d"
        return metadata

    def input(self, name: str, logical_shape: LogicalShape) -> Input:
        physical_shape = _logical_shape_to_physical_shape(logical_shape)
        hbm_group_name = f"{name}.hbm"
        if hbm_group_name not in self.program.hardware.hbm_objects:
            self.program.add_hbm_object(hbm_group_name, physical_shape)
        input_obj = Input(program=self.program, name=name, logical_shape=logical_shape)
        input_obj.metadata["hbm_group_obj"] = hbm_group_name
        self.inputs[name] = input_obj
        return input_obj

    def tensor(self, name: str, logical_shape: LogicalShape) -> Tensor:
        tensor = Tensor(program=self.program, name=name, logical_shape=logical_shape)
        self.tensors[name] = tensor
        return tensor

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
        if not all(isinstance(tile, (TensorTile, InputTile)) for tile in resolved_tiles):
            raise RuntimeError("mapt_head_group expects tensor/input tiles only")

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
        self.program.operation_log.append(
            {
                "kind": "mapt_head_group",
                "group_count": len(packets),
                "operand_type": type(operand).__name__,
            }
        )
        return packets

    def _mapt_bshd_matmul_groups(self, src1: object, src2: object, dst: object) -> List[List[object]]:
        src1_tiles = _tiles_in_grid_order(src1.tiles)
        src2_tiles = _tiles_in_grid_order(src2.tiles)
        dst_tiles = _tiles_in_grid_order(dst.tiles)
        src1_by_head_row_k: Dict[Tuple[int, int, int], object] = {}
        src2_by_head_k_col: Dict[Tuple[int, int, int], object] = {}
        groups: List[List[object]] = []

        for tile in src1_tiles:
            head_index = int(tile.metadata.get("head_index", 0))
            row_block = int(tile.metadata.get("row_block", tile.coord[0]))
            k_index = int(tile.metadata.get("d_tile_index", tile.coord[1]))
            src1_by_head_row_k[(head_index, row_block, k_index)] = tile

        for tile in src2_tiles:
            head_index = int(tile.metadata.get("head_index", 0))
            k_index = int(tile.metadata.get("row_block", tile.coord[0]))
            d_tile_index = int(tile.metadata.get("d_tile_index", 0))
            src2_by_head_k_col[(head_index, k_index, d_tile_index)] = tile

        for dst_tile in dst_tiles:
            head_index = int(dst_tile.metadata.get("head_index", 0))
            row_block = int(dst_tile.metadata.get("row_block", dst_tile.coord[0]))
            d_tile_index = int(dst_tile.metadata.get("d_tile_index", 0))
            lhs_candidates = [
                key for key in src1_by_head_row_k.keys() if key[0] == head_index and key[1] == row_block
            ]
            k_values = sorted(key[2] for key in lhs_candidates)
            group: List[object] = []
            for k_index in k_values:
                lhs_tile = src1_by_head_row_k.get((head_index, row_block, k_index))
                rhs_tile = src2_by_head_k_col.get((head_index, k_index, d_tile_index))
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

        _, src1_seq, src1_heads, src1_dim = getattr(src1, "logical_shape")
        _, src2_seq, src2_heads, src2_dim = getattr(src2, "logical_shape")
        _, dst_seq, dst_heads, dst_dim = getattr(dst, "logical_shape")
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
        lhs_groups: Dict[Tuple[int, int], TileLike] = {}
        rhs_groups: Dict[Tuple[int, int], TileLike] = {}
        dst_by_key: Dict[Tuple[int, int, int], TileLike] = {}
        threads: List[BTMMHeadGroupThread] = []

        for tile in lhs_tiles:
            row_block = int(tile.metadata.get("row_block", tile.coord[0]))
            group_block = int(tile.coord[1])
            lhs_groups[(row_block, group_block)] = tile

        for tile in rhs_tiles:
            row_block = int(tile.metadata.get("row_block", tile.coord[0]))
            group_block = int(tile.coord[1])
            rhs_groups[(row_block, group_block)] = tile

        dst_col_blocks_per_head = dst_dim // self.program.mlen
        for tile in dst_tiles:
            row_block = int(tile.metadata.get("row_block", tile.coord[0]))
            head_index = int(tile.metadata.get("head_index", 0))
            rhs_row_block = int(tile.coord[1]) - head_index * dst_col_blocks_per_head
            dst_by_key[(row_block, rhs_row_block, head_index)] = tile

        group_heads = self.program.btmm_lane_count
        q_row_blocks = max(1, ceil(src1_seq / self.program.mlen))
        k_row_blocks = max(1, ceil(src2_seq / self.program.mlen))
        group_blocks = max(1, ceil(src1_heads / group_heads))

        for lhs_row_block in range(q_row_blocks):
            for rhs_row_block in range(k_row_blocks):
                for group_block in range(group_blocks):
                    lhs_tile = lhs_groups.get((lhs_row_block, group_block))
                    rhs_tile = rhs_groups.get((rhs_row_block, group_block))
                    if lhs_tile is None or rhs_tile is None:
                        continue

                    head_start = group_block * group_heads
                    dst_group_tiles: List[TileLike] = []
                    lane_heads: List[int] = []
                    for lane in range(group_heads):
                        head_index = head_start + lane
                        if head_index >= dst_heads:
                            break
                        dst_tile = dst_by_key.get((lhs_row_block, rhs_row_block, head_index))
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
                            "group_block": group_block,
                            "group_start": head_start,
                            "group_heads": len(dst_group_tiles),
                            "lane_heads": lane_heads,
                            "lhs_row_block": lhs_row_block,
                            "rhs_row_block": rhs_row_block,
                        }
                    )
        return threads

    def mapt_scatter_group_matmul(
        self,
        src1: object,
        src2: object,
        dst: object,
    ) -> List[ScatterGroupMatmulThread]:
        if not (
            len(getattr(src1, "logical_shape", ())) == 4
            and len(getattr(src2, "logical_shape", ())) == 4
            and len(getattr(dst, "logical_shape", ())) == 4
        ):
            raise NotImplementedError("mapt_scatter_group_matmul currently supports BSHD tensors only")

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
        src1_by_head_row_k: Dict[Tuple[int, int, int], object] = {}
        src2_by_row_group: Dict[Tuple[int, int], object] = {}
        threads: List[ScatterGroupMatmulThread] = []

        for tile in _tiles_in_grid_order(src1.tiles):
            head_index = int(tile.metadata.get("head_index", 0))
            row_block = int(tile.metadata.get("row_block", tile.coord[0]))
            k_index = int(tile.metadata.get("d_tile_index", tile.coord[1]))
            src1_by_head_row_k[(head_index, row_block, k_index)] = tile

        for tile in _tiles_in_grid_order(src2.tiles):
            row_block = int(tile.metadata.get("row_block", tile.coord[0]))
            group_block = int(tile.coord[1])
            src2_by_row_group[(row_block, group_block)] = tile

        for dst_tile in _tiles_in_grid_order(dst.tiles):
            row_block = int(dst_tile.metadata.get("row_block", dst_tile.coord[0]))
            group_block = int(dst_tile.coord[1])
            group_start = group_block * group_heads
            lane_heads: List[int] = []
            lhs_candidates: List[List[object]] = []

            for lane in range(group_heads):
                head_index = group_start + lane
                lane_k_tiles = [
                    tile
                    for (tile_head, tile_row, _), tile in src1_by_head_row_k.items()
                    if tile_head == head_index and tile_row == row_block
                ]
                if not lane_k_tiles:
                    continue
                lane_heads.append(head_index)
                lhs_candidates.append(sorted(lane_k_tiles, key=lambda tile: int(tile.metadata.get("d_tile_index", 0))))

            rhs_terms: List[ScatterGroupMatmulTerm] = []
            rhs_row_blocks = sorted(
                row for (row, col_group) in src2_by_row_group.keys()
                if col_group == group_block
            )
            for rhs_row_block in rhs_row_blocks:
                rhs_tile = src2_by_row_group.get((rhs_row_block, group_block))
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
            return self.tensors.get(dst_tile.tensor_name) or self.inputs.get(dst_tile.tensor_name)
        if isinstance(dst_tile, InputTile):
            return self.inputs.get(dst_tile.input_name)
        return None

    def _extract_dst_tile_from_group(self, group: object) -> Optional[object]:
        if isinstance(group, dict):
            dst_tile = group.get("dst_tile")
            if isinstance(dst_tile, (TensorTile, InputTile)):
                return dst_tile
            dst_tiles = group.get("dst_tiles")
            if isinstance(dst_tiles, list):
                for item in dst_tiles:
                    if isinstance(item, (TensorTile, InputTile)):
                        return item
            return None
        if not isinstance(group, list) or not group:
            return None
        tail = group[-1]
        if isinstance(tail, list) and len(tail) == 1 and isinstance(tail[0], (TensorTile, InputTile)):
            return tail[0]
        if isinstance(tail, (TensorTile, InputTile)):
            return tail
        return None

    def _resolve_tiles_from_operand(self, operand: object) -> List[object]:
        if isinstance(operand, Input):
            return _tiles_in_grid_order(operand.tiles)
        if isinstance(operand, Tensor):
            return _tiles_in_grid_order(operand.tiles)
        if isinstance(operand, InputSlice):
            return self._resolve_slice_tiles(operand.base.tiles, operand.base.logical_shape, operand.selectors)
        if isinstance(operand, TensorSlice):
            return self._resolve_slice_tiles(operand.base.tiles, operand.base.logical_shape, operand.selectors)
        if isinstance(operand, (InputTile, TensorTile)):
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
    """Structure-only compute manager placeholder."""

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
            if isinstance(lhs_value, Scatter) or isinstance(rhs_value, Scatter):
                raise RuntimeError("matmul execute does not accept narrow scatter inputs; use scatter_group_matmul")
            if not isinstance(lhs_value, ValueTile) or not isinstance(rhs_value, ValueTile):
                raise RuntimeError("matmul execute expects ValueTile sources")
            lhs_vram_addr = lhs_value.residency.get("vram_addr")
            rhs_mram_addr = rhs_value.residency.get("mram_addr")
            if lhs_vram_addr is None:
                raise RuntimeError(f"matmul execute requires lhs value in VRAM: {lhs_value.value_tile_id}")
            if rhs_mram_addr is None:
                raise RuntimeError(f"matmul execute requires rhs value in MRAM: {rhs_value.value_tile_id}")
            lhs_vram_addrs.append(int(lhs_vram_addr))
            rhs_mram_addrs.append(int(rhs_mram_addr))

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
        self.program.operation_log.append(
            {
                "kind": "compute_matmul",
                "task_id": task_id,
                "lhs_count": len(lhs_vram_addrs),
                "rhs_count": len(rhs_mram_addrs),
                "lhs_vram_addrs": list(lhs_vram_addrs),
                "rhs_mram_addrs": list(rhs_mram_addrs),
                "dst_value": dst_value.value_tile_id,
                "dst_vram_addr": int(dst_vram_addr),
            }
        )
        return {
            "op_kind": "matmul",
            "inputs": operands,
            "outputs": [dst_value],
            "dst": dst_value,
            "task_id": task_id,
        }

    def scatter_group_matmul(
        self,
        lhs_values: List[ValueTile],
        rhs_group: ScatterGroup,
        dst_group: ScatterGroup,
    ) -> Dict[str, object]:
        if not lhs_values:
            raise RuntimeError("scatter_group_matmul expects one non-empty lhs ValueTile list")
        if not all(isinstance(value, ValueTile) for value in lhs_values):
            raise RuntimeError("scatter_group_matmul expects lhs_values to contain ValueTile objects only")

        rhs_value = self.program.value_manager.value_tiles.get(rhs_group.backing_value_tile_id)
        dst_value = self.program.value_manager.value_tiles.get(dst_group.backing_value_tile_id)
        if not isinstance(rhs_value, ValueTile) or not isinstance(dst_value, ValueTile):
            raise RuntimeError("scatter_group_matmul requires rhs/destination backing values")

        for lhs_value in lhs_values:
            self.program.value_manager.ensure_value_tile_in_place(lhs_value, "vram")
        self.program.value_manager.ensure_value_tile_in_place(rhs_value, "mram")
        self.program.value_manager.ensure_value_tile_in_place(dst_value, "vram")

        rhs_mram_addr = rhs_value.residency.get("mram_addr")
        dst_vram_addr = dst_value.residency.get("vram_addr")
        lhs_vram_addrs = [value.residency.get("vram_addr") for value in lhs_values]
        if rhs_mram_addr is None or dst_vram_addr is None or any(addr is None for addr in lhs_vram_addrs):
            raise RuntimeError("scatter_group_matmul requires lhs in VRAM, rhs in MRAM, dst in VRAM")

        task_id = f"scatter_group_matmul.{rhs_group.group_id}.{dst_group.group_id}"
        rhs_slots = self._ordered_group_scatters(rhs_group)
        dst_slots = self._ordered_group_scatters(dst_group)
        if len(rhs_slots) != len(dst_slots):
            raise RuntimeError(
                f"scatter_group_matmul requires matching rhs/dst slot counts, got rhs={len(rhs_slots)} dst={len(dst_slots)}"
            )
        if len(lhs_values) != len(rhs_slots):
            raise RuntimeError(
                f"scatter_group_matmul requires lhs_values to align with group lanes, got lhs={len(lhs_values)} slots={len(rhs_slots)}"
            )

        lane_logs: List[Dict[str, object]] = []
        for lane_index, (lhs_addr, rhs_slot, dst_slot, lhs_value) in enumerate(
            zip(lhs_vram_addrs, rhs_slots, dst_slots, lhs_values)
        ):
            if lhs_addr is None:
                raise RuntimeError(f"scatter_group_matmul lane {lane_index} is missing one lhs VRAM address")
            if rhs_slot.col_count != dst_slot.col_count:
                raise RuntimeError(
                    f"scatter_group_matmul lane {lane_index} slot width mismatch rhs={rhs_slot.col_count} dst={dst_slot.col_count}"
                )
            self.program.emit_slot_matmul(
                lhs_vram_addr=int(lhs_addr),
                rhs_mram_addr=int(rhs_mram_addr),
                rhs_col_offset=int(rhs_slot.col_offset),
                dst_vram_addr=int(dst_vram_addr),
                dst_col_offset=int(dst_slot.col_offset),
                col_count=int(rhs_slot.col_count),
                task_id=f"{task_id}.lane{lane_index}",
                zero_dst=(lane_index == 0),
            )
            lane_logs.append(
                {
                    "lane_index": lane_index,
                    "lhs_value": lhs_value.value_tile_id,
                    "lhs_vram_addr": int(lhs_addr),
                    "rhs_scatter": rhs_slot.scatter_id,
                    "rhs_col_offset": int(rhs_slot.col_offset),
                    "dst_scatter": dst_slot.scatter_id,
                    "dst_col_offset": int(dst_slot.col_offset),
                    "col_count": int(rhs_slot.col_count),
                }
            )
        self.program.operation_log.append(
            {
                "kind": "compute_scatter_group_matmul",
                "task_id": task_id,
                "lhs_values": [value.value_tile_id for value in lhs_values],
                "rhs_group": rhs_group.group_id,
                "rhs_value": rhs_value.value_tile_id,
                "dst_group": dst_group.group_id,
                "dst_value": dst_value.value_tile_id,
                "lhs_vram_addrs": [int(addr) for addr in lhs_vram_addrs if addr is not None],
                "rhs_mram_addr": int(rhs_mram_addr),
                "dst_vram_addr": int(dst_vram_addr),
                "lanes": lane_logs,
            }
        )
        return {
            "op_kind": "scatter_group_matmul",
            "inputs": [lhs_values, rhs_group, dst_group],
            "outputs": [dst_value],
            "dst": dst_value,
            "dst_group": dst_group,
            "task_id": task_id,
        }

    def _ordered_group_scatters(self, group: ScatterGroup) -> List[Scatter]:
        scatters: List[Scatter] = []
        for scatter_id in group.scatter_ids:
            scatter = self.program.value_manager.scatters.get(scatter_id)
            if scatter is None:
                raise RuntimeError(f"Scatter group {group.group_id} references missing scatter {scatter_id}")
            scatters.append(scatter)
        scatters.sort(key=lambda scatter: (int(scatter.row_offset), int(scatter.col_offset), scatter.scatter_id))
        return scatters

    def scatter_group_binary(
        self,
        lhs_group: ScatterGroup,
        rhs_group: ScatterGroup,
        dst_group: ScatterGroup,
        *,
        op: str = "add",
    ) -> Dict[str, object]:
        # Intentionally unmasked: scatter-group <-> scatter-group binary assumes the
        # participating groups are packing-aligned inside their backing values, so the
        # whole-tile VV op is the right lowering here. If we later add binary ops for a
        # single Scatter, that path should use masked VV/VF instead of extending this one.
        lhs_value = self.program.value_manager.value_tiles.get(lhs_group.backing_value_tile_id)
        rhs_value = self.program.value_manager.value_tiles.get(rhs_group.backing_value_tile_id)
        dst_old_value = self.program.value_manager.value_tiles.get(dst_group.backing_value_tile_id)
        dst_value: Optional[ValueTile] = None
        if (
            not isinstance(lhs_value, ValueTile)
            or not isinstance(rhs_value, ValueTile)
            or not isinstance(dst_old_value, ValueTile)
        ):
            raise RuntimeError("scatter_group_binary requires lhs/rhs/dst backing values")

        self.program.value_manager.ensure_value_tile_in_place(lhs_value, "vram")
        self.program.value_manager.ensure_value_tile_in_place(rhs_value, "vram")

        lhs_vram_addr = lhs_value.residency.get("vram_addr")
        rhs_vram_addr = rhs_value.residency.get("vram_addr")
        if lhs_vram_addr is None or rhs_vram_addr is None:
            raise RuntimeError("scatter_group_binary requires lhs/rhs in VRAM")

        deferred_free_value_tile_id: Optional[str] = None
        if dst_old_value.value_tile_id in {lhs_value.value_tile_id, rhs_value.value_tile_id}:
            self.program.value_manager.protect_value_tile(dst_old_value, "vram")
            try:
                old_value_tile_id = self.program.value_manager._detach_scatter_group_backing_value(dst_group)
                dst_value = self.program.value_manager.create_value_tile_in_vram(dst_old_value)
                self.program.value_manager._attach_scatter_group_backing_value(dst_group, dst_value)
                self.program.operation_log.append(
                    {
                        "kind": "rebind_scatter_group_backing_value",
                        "scatter_group": dst_group.group_id,
                        "old_value": old_value_tile_id,
                        "new_value": dst_value.value_tile_id,
                    }
                )
                deferred_free_value_tile_id = old_value_tile_id
            finally:
                self.program.value_manager.stop_protect_value_tile(dst_old_value, "vram")
        else:
            _, dst_value = self.program.value_manager.prepare_updated_scatter_group_backing(dst_group)
        self.program.value_manager.ensure_value_tile_in_place(dst_value, "vram")
        dst_vram_addr = dst_value.residency.get("vram_addr")
        if dst_vram_addr is None:
            raise RuntimeError("scatter_group_binary requires dst in VRAM")

        task_id = f"scatter_group_{op}.{lhs_group.group_id}.{rhs_group.group_id}.{dst_group.group_id}"
        self.program.emit_tile_binary(
            lhs_vram_addr=int(lhs_vram_addr),
            rhs_vram_addr=int(rhs_vram_addr),
            dst_vram_addr=int(dst_vram_addr),
            op=op,
            task_id=task_id,
        )
        self.program.operation_log.append(
            {
                "kind": "compute_scatter_group_binary",
                "op": op,
                "task_id": task_id,
                "lhs_group": lhs_group.group_id,
                "rhs_group": rhs_group.group_id,
                "dst_group": dst_group.group_id,
                "lhs_value": lhs_value.value_tile_id,
                "rhs_value": rhs_value.value_tile_id,
                "dst_value": dst_value.value_tile_id,
                "dst_old_value": dst_old_value.value_tile_id,
                "lhs_vram_addr": int(lhs_vram_addr),
                "rhs_vram_addr": int(rhs_vram_addr),
                "dst_vram_addr": int(dst_vram_addr),
            }
        )
        if deferred_free_value_tile_id is not None:
            self.program.value_manager.free_value_tile(deferred_free_value_tile_id)
        return {
            "op_kind": f"scatter_group_{op}",
            "inputs": [lhs_group, rhs_group, dst_group],
            "outputs": [dst_value],
            "dst": dst_value,
            "dst_group": dst_group,
            "task_id": task_id,
        }

    def scatter_group_add(
        self,
        lhs_group: ScatterGroup,
        rhs_group: ScatterGroup,
        dst_group: ScatterGroup,
    ) -> Dict[str, object]:
        return self.scatter_group_binary(lhs_group, rhs_group, dst_group, op="add")

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
        self.program.operation_log.append(
            {
                "kind": "compute_btmm",
                "task_id": task_id,
                "lhs_value": lhs_packed_value.value_tile_id,
                "rhs_value": rhs_value.value_tile_id,
                "lhs_vram_addr": int(lhs_vram_addr),
                "rhs_mram_addr": int(rhs_mram_addr),
                "btmm_finished": True,
            }
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
        self.program.operation_log.append(
            {
                "kind": "compute_btmm_wo",
                "task_id": task_id,
                "base_addr": base_addr,
                "tile_count": resolved_tile_count,
                "value_tiles": [value.value_tile_id for value in out_values],
            }
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
        dst: Optional[Sequence[FPVar]] = None,
        rhs: Optional[Sequence[FPVar]] = None,
        op: str,
        task_id: str = "row_operations",
    ) -> Dict[str, object]:
        if isinstance(src, Scatter):
            backing_value = self.program.value_manager.value_tiles.get(src.backing_value_tile_id)
            if not isinstance(backing_value, ValueTile):
                raise RuntimeError(f"row_operations scatter source is missing backing value: {src.scatter_id}")
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
                    f"row_operations scatter mask expects col_offset/col_count aligned to mask_unit={mask_unit}, "
                    f"got col_offset={col_offset} col_count={col_count}"
                )
            lane_start = col_offset // mask_unit
            lane_count = col_count // mask_unit
            mask_val = ((1 << lane_count) - 1) << lane_start
            src_name = src.scatter_id
        else:
            self.program.value_manager.ensure_value_tile_in_place(src, "vram")
            src_vram_addr = src.residency.get("vram_addr")
            row_count = int(src.logical_shape[0])
            mask_val = None
            src_name = src.value_tile_id
        if src_vram_addr is None:
            raise RuntimeError(f"row_operations requires src in VRAM: {src_name}")

        dst_addrs = [_require_fp_addr(var) for var in dst] if dst is not None else None
        rhs_addrs = [_require_fp_addr(var) for var in rhs] if rhs is not None else None
        self.program.emit_row_operation(
            src_vram_addr=int(src_vram_addr),
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

    plus several specialized routes such as scatter-group matmul and BTMM/QKT
    execution.
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
        self._next_hbm_addr = 0

        self.compiler = TiledDeveloperCompiler(
            mlen=self.mlen,
            blen=self.blen,
            fpram_total_size=(self.fpram_capacity or 1024),
        )
        self.hardware = HardwareManager(self)
        self.value_manager = ValueManager(self)
        self.tensor_manager = TensorManager(self)
        self.compute_manager = ComputeManager(self)
        self.operation_log: List[Dict[str, object]] = []
        self._auto_name_counters: Dict[str, int] = {}

    def input(self, name: str, logical_shape: LogicalShape) -> Input:
        return self.tensor_manager.input(name, logical_shape)

    def tensor(self, name: str, logical_shape: LogicalShape) -> Tensor:
        return self.tensor_manager.tensor(name, logical_shape)

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
    ) -> Tensor | FPFragment:
        fragment = self.tensor_manager.alloc_fragment(
            name=name,
            logical_shape=logical_shape,
            init_zero=init_zero,
            dtype=dtype,
        )
        if init_zero and isinstance(fragment, Tensor):
            self.clear(fragment)
        return fragment

    def constant(self, name: str, value: float, size: int = 1) -> FPVar | FPFragment:
        return self.fp_var(name, value=value, size=size)

    def pipelined(self, extent: int, num_stages: int = 1) -> range:
        self.operation_log.append(
            {
                "kind": "pipelined_hint",
                "extent": int(extent),
                "num_stages": int(num_stages),
            }
        )
        return range(int(extent))

    def mapf(self, operand: object) -> List[FPVar]:
        return self.tensor_manager.mapf(operand)

    def mapf_t(self, tensor_operand: object, fp_operand: object, *, control: str = "mixed") -> Dict[str, object]:
        return self.tensor_manager.mapf_t(tensor_operand, fp_operand, control=control)

    def fp_kernel(
        self,
        src1: object,
        dst: object,
        *,
        src2: Optional[object] = None,
        control: str = "add",
        task_id: str = "fp_kernel",
    ) -> Dict[str, object]:
        return self.compute_manager.fp_kernel(
            self.mapf(src1),
            self.mapf(dst),
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
        return self.compute_manager.pure_fp_compute(
            self.mapf(src1),
            self.mapf(dst),
            src2=self.mapf(src2) if src2 is not None else None,
            op=control,
            task_id=task_id,
        )

    def copy(self, src: object, dst: object) -> object:
        if isinstance(src, (FPVar, FPFragment, FPFragmentSlice)) or isinstance(dst, (FPVar, FPFragment, FPFragmentSlice)):
            return self.fp_copy(src, dst)
        self.operation_log.append(
            {
                "kind": "copy_start",
                "src": getattr(src, "name", type(src).__name__),
                "dst": getattr(dst, "name", type(dst).__name__),
            }
        )
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
            tmp = self.value_manager.mapv([src_tile, dst_tile, ["vram", "vram"], "copy_tile_pair"])
            signal_3 = self.value_manager.mapv_back([{"op_kind": "copy"}, tmp])
            signal_4.append(signal_3)
        self.operation_log.append({"kind": "copy_end", "groups": len(signal_4)})
        return self.tensor_manager.mapt_back(signal_4, dst_groups)

    def atomic_ops(
        self,
        src1: Tensor | Input,
        src2: Tensor | Input,
        dst: Tensor | Input,
        *,
        op: str = "add",
    ) -> object:
        """Run elementwise tile ops with alias-safe destination updates.

        The public API looks like one simple tilewise binary op, but the runtime
        has two materially different execution paths:

        - wide/full-tile path using direct ValueTile operands in VRAM
        - narrow/grouped path using ScatterGroup-aware compute and rebinding

        When the destination aliases one source (for example `A + B -> B`), the
        wide-tile path first detaches the old destination binding and
        materializes a fresh writable value so reads remain stable during the
        update.
        """
        if op not in {"add", "sub", "mul"}:
            raise ValueError(f"atomic_ops only supports add/sub/mul, got op={op!r}")
        logical_shapes = [getattr(src, "logical_shape", None) for src in (src1, src2, dst)]
        if logical_shapes[0] is None or logical_shapes[1] is None or logical_shapes[2] is None:
            raise TypeError("atomic_ops expects tensor/input operands with logical_shape")
        if logical_shapes[0] != logical_shapes[1] or logical_shapes[0] != logical_shapes[2]:
            raise ValueError(
                f"atomic_ops requires matching logical shapes, got src1={logical_shapes[0]} src2={logical_shapes[1]} dst={logical_shapes[2]}"
            )

        self.operation_log.append(
            {
                "kind": "atomic_ops_start",
                "op": op,
                "src1": getattr(src1, "name", type(src1).__name__),
                "src2": getattr(src2, "name", type(src2).__name__),
                "dst": getattr(dst, "name", type(dst).__name__),
            }
        )

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

            lhs_group = self.value_manager.try_map_tile_to_scatter_group(lhs_tile)
            rhs_group = self.value_manager.try_map_tile_to_scatter_group(rhs_tile)
            dst_group = self.value_manager.try_map_tile_to_scatter_group(dst_tile)
            if lhs_group is not None or rhs_group is not None or dst_group is not None:
                if lhs_group is None or rhs_group is None or dst_group is None:
                    raise RuntimeError(
                        f"atomic_ops requires all operands to share one mapping style, got lhs_group={lhs_group is not None} rhs_group={rhs_group is not None} dst_group={dst_group is not None}"
                    )
                result = self.compute_manager.scatter_group_binary(lhs_group, rhs_group, dst_group, op=op)
                dst_value = result.get("dst")
                if not isinstance(dst_value, ValueTile):
                    raise RuntimeError(f"atomic_ops scatter-group path did not produce one destination ValueTile for op={op!r}")
                signal_4.append(
                    {
                        "control": f"atomic_{op}_scatter_group",
                        "dst_tile": dst_tile,
                        "dst_group_id": dst_group.group_id,
                        "dst_value_id": dst_value.value_tile_id,
                    }
                )
                continue

            lhs_value = self.value_manager.resolve_value_tile(lhs_tile)
            rhs_value = self.value_manager.resolve_value_tile(rhs_tile)
            dst_value_old_id: Optional[str] = None
            dst_aliases_source = lhs_tile.tile_id == dst_tile.tile_id or rhs_tile.tile_id == dst_tile.tile_id
            if (
                dst_aliases_source
                and isinstance(dst_tile, TensorTile)
                and self.value_manager.try_map_tile_to_scatter_group(dst_tile) is None
                and not self.value_manager._is_narrow_tensor_tile(dst_tile)
            ):
                old_dst_value, dst_value, dst_value_old_id = self.value_manager.prepare_updated_wide_tile_value(
                    dst_tile,
                    ensure_old_place="vram",
                    new_place="vram",
                )
                if lhs_tile.tile_id == dst_tile.tile_id:
                    lhs_value = old_dst_value
                if rhs_tile.tile_id == dst_tile.tile_id:
                    rhs_value = old_dst_value
            else:
                dst_value = self.value_manager._prepare_mapv_destination_value(dst_tile, "vram")
            try:
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
                self.operation_log.append(
                    {
                        "kind": "compute_atomic_ops_tile",
                        "op": op,
                        "task_id": f"atomic_{op}.{group_index}",
                        "lhs_value": lhs_value.value_tile_id,
                        "rhs_value": rhs_value.value_tile_id,
                        "dst_value": dst_value.value_tile_id,
                        "lhs_vram_addr": int(lhs_vram_addr),
                        "rhs_vram_addr": int(rhs_vram_addr),
                        "dst_vram_addr": int(dst_vram_addr),
                    }
                )
                signal_4.append(
                    {
                        "control": f"atomic_{op}_tile",
                        "dst_tile": dst_tile,
                        "dst_value_id": dst_value.value_tile_id,
                    }
                )
            finally:
                if dst_value_old_id is not None:
                    self.value_manager.free_value_tile(dst_value_old_id)

        self.operation_log.append({"kind": "atomic_ops_end", "op": op, "groups": len(signal_4)})
        return self.tensor_manager.mapt_back(signal_4, dst_groups)

    def atomic_add(self, src1: Tensor | Input, src2: Tensor | Input, dst: Tensor | Input) -> object:
        return self.atomic_ops(src1, src2, dst, op="add")

    def atomic_sub(self, src1: Tensor | Input, src2: Tensor | Input, dst: Tensor | Input) -> object:
        return self.atomic_ops(src1, src2, dst, op="sub")

    def atomic_mul(self, src1: Tensor | Input, src2: Tensor | Input, dst: Tensor | Input) -> object:
        return self.atomic_ops(src1, src2, dst, op="mul")

    def fill(self, dst: object, src: object) -> object:
        if isinstance(dst, (FPVar, FPFragment, FPFragmentSlice)):
            return self.fp_fill(dst, src)
        raise NotImplementedError(f"fill currently supports FP-domain destinations only, got {type(dst).__name__}")

    def matmul(self, src1: Tensor | Input, src2: Tensor | Input | TensorTranspose | InputTranspose, dst: Tensor | Input) -> object:
        """Route one matmul request to the correct execution strategy.

        The current runtime supports multiple matmul families behind one API:

        - default tilewise matmul using `mapt -> mapv -> compute`
        - scatter-group matmul for grouped narrow-head layouts
        - BTMM/QKT path when the RHS is explicitly transposed and shapes match

        This function is therefore both an entry point and a router. The exact
        path is selected from logical shape/layout information before compute
        packets are materialized.
        """
        self.operation_log.append(
            {
                "kind": "matmul_start",
                "src1": getattr(src1, "name", type(src1).__name__),
                "src2": getattr(src2, "name", type(src2).__name__),
                "dst": getattr(dst, "name", type(dst).__name__),
            }
        )
        if self._should_use_btmm_qkt_matmul(src1, src2, dst):
            out = self._matmul_btmm_qkt_path(src1, _unwrap_transposed_operand(src2), dst)
            self.operation_log.append({"kind": "matmul_end", "groups": "btmm_qkt"})
            return out
        if _is_transposed_operand(src2):
            raise RuntimeError("BTMM/QKT matmul only supports explicit transpose syntax as prog.matmul(q, k.T, p)")
        if self._should_use_scatter_group_matmul(src1, src2, dst):
            out = self._matmul_scatter_group_path(src1, src2, dst)
            self.operation_log.append({"kind": "matmul_end", "groups": "scatter_group"})
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
        self.operation_log.append({"kind": "matmul_end", "groups": len(signal_4)})
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

    def _should_use_scatter_group_matmul(self, src1: Tensor | Input, src2: Tensor | Input, dst: Tensor | Input) -> bool:
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

    def _matmul_scatter_group_path(self, src1: Tensor | Input, src2: Tensor | Input, dst: Tensor | Input) -> object:
        signal_1 = self.tensor_manager.mapt_scatter_group_matmul(src1, src2, dst)
        signal_4: List[Dict[str, object]] = []

        for dst_tile, terms, group_start in signal_1:
            dst_group = self.value_manager.try_map_tile_to_scatter_group(dst_tile)
            if dst_group is None:
                raise RuntimeError(f"Destination tile {dst_tile.tile_id} is not one narrow/scatter-group tile")
            dst_value = self.value_manager.value_tiles.get(dst_group.backing_value_tile_id)
            if not isinstance(dst_value, ValueTile):
                raise RuntimeError(f"Destination group {dst_group.group_id} is missing backing value")

            if not terms:
                continue

            _, dst_value = self.value_manager.prepare_updated_scatter_group_backing(dst_group)
            accumulator_ready = False
            current_dst_value = dst_value
            self.value_manager.protect_value_tile(current_dst_value, "vram")
            try:
                for term_index, (lhs_tiles, rhs_tile) in enumerate(terms):
                    if not lhs_tiles:
                        continue

                    lhs_values = [self.value_manager._resolve_mapv_source_value(tile, "vram") for tile in lhs_tiles]
                    if any(isinstance(value, Scatter) for value in lhs_values):
                        raise RuntimeError(
                            f"scatter-group matmul term expected lhs full ValueTiles, got {lhs_values}"
                        )
                    lhs_full_values = [value for value in lhs_values if isinstance(value, ValueTile)]
                    rhs_group = self.value_manager.try_map_tile_to_scatter_group(rhs_tile)
                    if rhs_group is None:
                        raise RuntimeError("scatter-group matmul term rhs tile is not one narrow/scatter-group tile")

                    target_group = dst_group
                    transient_group: Optional[ScatterGroup] = None
                    for lhs_value in lhs_full_values:
                        self.value_manager.protect_value_tile(lhs_value, "vram")
                    if accumulator_ready:
                        transient_group = self.value_manager.create_transient_scatter_group_like(dst_group)
                        target_group = transient_group

                    try:
                        self.scatter_group_matmul(
                            lhs_values=lhs_full_values,
                            rhs_group=rhs_group,
                            dst_group=target_group,
                        )

                        if accumulator_ready:
                            updated_dst_value = self.scatter_group_add(dst_group, target_group, dst_group)
                            self.value_manager.stop_protect_value_tile(current_dst_value, "vram")
                            current_dst_value = updated_dst_value
                            self.value_manager.protect_value_tile(current_dst_value, "vram")
                            self.value_manager._release_scatter_group(target_group.group_id)
                        else:
                            accumulator_ready = True
                    finally:
                        for lhs_value in lhs_full_values:
                            self.value_manager.stop_protect_value_tile(lhs_value, "vram")

                    self.operation_log.append(
                        {
                            "kind": "matmul_scatter_group_term",
                            "thread_dst_tile": dst_tile.tile_id,
                            "thread_group_start": group_start,
                            "term_index": term_index,
                            "lhs_values": [value.value_tile_id for value in lhs_full_values],
                            "rhs_group": rhs_group.group_id,
                            "dst_group": target_group.group_id,
                            "accumulate_into_dst": accumulator_ready,
                        }
                    )
            finally:
                self.value_manager.stop_protect_value_tile(current_dst_value, "vram")

            signal_4.append(
                {
                    "control": "scatter_group_matmul",
                    "dst_tile_id": dst_tile.tile_id,
                    "dst_group_id": dst_group.group_id,
                    "dst_tile": dst_tile,
                }
            )

        out = self.tensor_manager.mapt_back(signal_4, signal_1)
        self.operation_log.append(
            {
                "kind": "matmul_scatter_group_dispatch",
                "signal": "scatter_group_matmul",
                "groups": len(signal_4),
            }
        )
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
            self.operation_log.append(
                {
                    "kind": "matmul_btmm_qkt_thread",
                    "task_id": task_id,
                    "thread_index": thread_index,
                    "lhs_tile": lhs_tile.tile_id,
                    "rhs_tile": rhs_tile.tile_id,
                    "dst_tiles": [tile.tile_id for tile in dst_tiles],
                    "group_start": thread.get("group_start"),
                    "group_heads": thread.get("group_heads"),
                    "lhs_row_block": thread.get("lhs_row_block"),
                    "rhs_row_block": thread.get("rhs_row_block"),
                }
            )

        out = self.tensor_manager.mapt_back(signal_4, signal_1)
        self.operation_log.append(
            {
                "kind": "matmul_btmm_qkt_dispatch",
                "signal": "btmm_qkt_matmul",
                "groups": len(signal_4),
            }
        )
        return out

    def scatter_group_matmul(
        self,
        lhs_values: List[ValueTile],
        rhs_group: ScatterGroup,
        dst_group: ScatterGroup,
    ) -> ValueTile:
        result = self.compute_manager.scatter_group_matmul(lhs_values, rhs_group, dst_group)
        dst_value = result.get("dst")
        if not isinstance(dst_value, ValueTile):
            raise RuntimeError("scatter_group_matmul did not produce one destination ValueTile")
        return dst_value

    def scatter_group_add(
        self,
        lhs_group: ScatterGroup,
        rhs_group: ScatterGroup,
        dst_group: ScatterGroup,
    ) -> ValueTile:
        result = self.compute_manager.scatter_group_add(lhs_group, rhs_group, dst_group)
        dst_value = result.get("dst")
        if not isinstance(dst_value, ValueTile):
            raise RuntimeError("scatter_group_add did not produce one destination ValueTile")
        return dst_value

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
        src: Tensor | Input | TensorSlice | InputSlice,
        rhs: Optional[object] = None,
        op: str = "exp",
        *,
        out: Optional[object] = None,
        dim: int = -1,
        task_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        if dim != -1:
            raise NotImplementedError(f"row_op currently supports dim=-1 only, got {dim}")
        src_slice_ranges: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        if isinstance(src, (TensorSlice, InputSlice)):
            src_slice_ranges = _logical_selectors_to_physical_ranges(src.base.logical_shape, src.selectors)
        src_groups = self.tensor_manager.mapt([src, 0])
        if not src_groups:
            return []
        records: List[Dict[str, object]] = []
        mutates_src = op in {"exp", "mul", "add", "sub"}
        rhs_vars = self.mapf(rhs) if rhs is not None and op in {"mul", "add", "sub"} else None
        out_vars = self.mapf(out) if out is not None else None
        rhs_cursor = 0
        out_cursor = 0
        for group_index, src_group in enumerate(src_groups):
            if len(src_group) != 1 or not isinstance(src_group[0], (TensorTile, InputTile)):
                raise RuntimeError("row_op currently expects one full tile per mapt group")
            src_tile = src_group[0]
            deferred_free_value_tile_id: Optional[str] = None
            try:
                if mutates_src:
                    if (
                        isinstance(src_tile, TensorTile)
                        and self.value_manager.try_map_tile_to_scatter_group(src_tile) is not None
                    ):
                        dst_group = self.value_manager.map_tile_to_scatter_group(src_tile)
                        old_value = self.value_manager.value_tiles.get(dst_group.backing_value_tile_id)
                        if not isinstance(old_value, ValueTile):
                            raise RuntimeError(
                                f"row_op mutating scatter-group tile {src_tile.tile_id} is missing backing value"
                            )
                        self.value_manager.ensure_value_tile_in_place(old_value, "vram")
                        self.value_manager.protect_value_tile(old_value, "vram")
                        try:
                            old_value_tile_id = self.value_manager._detach_scatter_group_backing_value(dst_group)
                            new_value = self.value_manager.create_value_tile_in_vram(old_value)
                            self.value_manager._attach_scatter_group_backing_value(dst_group, new_value)
                            self.operation_log.append(
                                {
                                    "kind": "rebind_scatter_group_backing_value",
                                    "scatter_group": dst_group.group_id,
                                    "old_value": old_value_tile_id,
                                    "new_value": new_value.value_tile_id,
                                }
                            )
                            deferred_free_value_tile_id = old_value_tile_id
                        finally:
                            self.value_manager.stop_protect_value_tile(old_value, "vram")

                        old_vram_addr = old_value.residency.get("vram_addr")
                        new_vram_addr = new_value.residency.get("vram_addr")
                        if old_vram_addr is None or new_vram_addr is None:
                            raise RuntimeError("row_op mutating scatter-group path requires old/new values in VRAM")
                        self._emit_copy_vram_tile(int(new_vram_addr), int(old_vram_addr))
                    elif isinstance(src_tile, TensorTile) and not self.value_manager._is_narrow_tensor_tile(src_tile):
                        old_value, new_value, old_value_tile_id = self.value_manager.prepare_updated_wide_tile_value(
                            src_tile,
                            ensure_old_place="vram",
                            new_place="vram",
                        )
                        old_vram_addr = old_value.residency.get("vram_addr")
                        new_vram_addr = new_value.residency.get("vram_addr")
                        if old_vram_addr is None or new_vram_addr is None:
                            raise RuntimeError("row_op mutating wide-tile path requires old/new values in VRAM")
                        self._emit_copy_vram_tile(int(new_vram_addr), int(old_vram_addr))
                        deferred_free_value_tile_id = old_value_tile_id
                    else:
                        self.value_manager._prepare_mapv_destination_value(src_tile, "vram")

                if src_slice_ranges is not None:
                    src_operand = self.value_manager.resolve_row_operand_for_ranges(
                        src_tile,
                        src_slice_ranges[0],
                        src_slice_ranges[1],
                        "vram",
                    )
                else:
                    src_operand = self.value_manager.resolve_row_operand(src_tile, "vram")
                row_count = int(src_operand.row_count if isinstance(src_operand, Scatter) else src_operand.logical_shape[0])

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
                    dst=group_out,
                    rhs=group_rhs,
                    op=op,
                    task_id=task_id or f"row_op.{op}.{group_index}",
                )
                records.append(record)
            finally:
                if deferred_free_value_tile_id is not None:
                    self.value_manager.free_value_tile(deferred_free_value_tile_id)
        return records

    def elementwise(
        self,
        src1: object,
        dst: object,
        *,
        src2: Optional[object] = None,
        op: str = "add",
        task_id: Optional[str] = None,
    ) -> Dict[str, object]:
        return self.pure_fp_compute(
            src1,
            dst,
            src2=src2,
            control=op,
            task_id=task_id or f"elementwise.{op}",
        )

    def clear(self, tensor: Tensor) -> None:
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
        self.operation_log.append({"kind": "clear_tensor", "tensor": tensor.name, "values": len(cleared_values)})

    def batch_matmul(
        self,
        src1: Tensor | Input,
        src2: Tensor | Input,
        *,
        transpose_b: bool = False,
        out: Tensor | Input,
    ) -> object:
        if not transpose_b:
            raise NotImplementedError("batch_matmul currently supports transpose_b=True only")
        return self.matmul(src1, src2, out)

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

    def _emit_copy_vram_tile(self, dst_vram_addr: int, src_vram_addr: int) -> None:
        if int(dst_vram_addr) == int(src_vram_addr):
            self.compiler.generated_code += f"; skip copy tile because src/dst alias at vram[{int(dst_vram_addr)}]\n"
            return
        self.emit_zero_vram_tile(int(dst_vram_addr))
        self.emit_tile_binary(
            lhs_vram_addr=int(dst_vram_addr),
            rhs_vram_addr=int(src_vram_addr),
            dst_vram_addr=int(dst_vram_addr),
            op="add",
            task_id=f"copy_vram_tile.{int(src_vram_addr)}.{int(dst_vram_addr)}",
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
        op: str,
        row_count: int,
        dst_addrs: Optional[Sequence[int]] = None,
        rhs_addrs: Optional[Sequence[int]] = None,
        mask_val: Optional[int] = None,
        task_id: str = "row_operations",
    ) -> None:
        if row_count <= 0:
            return
        unary_ops = {"exp"}
        reduce_ops = {"reduce_max": "V_RED_MAX", "reduce_sum": "V_RED_SUM"}
        binary_ops = {"mul": "V_MUL_VF", "add": "V_ADD_VF", "sub": "V_SUB_VF"}
        if op not in unary_ops | set(reduce_ops) | set(binary_ops):
            raise ValueError(f"Unsupported emit_row_operation op={op!r}")

        gp_regs = self.compiler.register_allocator.allocate_gp(5)
        gp_src, gp_fp, gp_dst, gp_loop, gp_mask = gp_regs
        lines = [f"; row operation task {task_id} op={op} rows={row_count}"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {int(src_vram_addr)}")
        use_mask = mask_val is not None
        if use_mask:
            lines.append(f"; row operation mask {int(mask_val)}")
            lines.append(f"S_ADDI_INT gp{gp_mask}, gp0, {int(mask_val)}")
            lines.append(f"C_SET_V_MASK_REG gp{gp_mask}")

        if op in unary_ops:
            lines.append(f"C_LOOP_START gp{gp_loop}, {int(row_count)}")
            lines.append(f"V_EXP_V gp{gp_src}, gp{gp_src}, {1 if use_mask else 0}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}")
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
                    lines.append(f"V_SUB_VF gp{gp_src}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                else:
                    lines.append(f"{binary_ops[op]} gp{gp_src}, gp{gp_src}, f1, {1 if use_mask else 0}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            elif rhs_prog is not None:
                rhs_start, count, rhs_step = rhs_prog
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {rhs_start}")
                lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                if op == "sub":
                    lines.append(f"V_SUB_VF gp{gp_src}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                else:
                    lines.append(f"{binary_ops[op]} gp{gp_src}, gp{gp_src}, f1, {1 if use_mask else 0}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {self.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {rhs_step}")
                lines.append(f"C_LOOP_END gp{gp_loop}")
            else:
                for row_index, rhs_addr in enumerate(rhs_addrs):
                    row_addr = int(src_vram_addr) + row_index * self.mlen
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {int(rhs_addr)}")
                    lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                    if op == "sub":
                        lines.append(f"V_SUB_VF gp{gp_src}, gp{gp_src}, f1, {1 if use_mask else 0}, 0")
                    else:
                        lines.append(f"{binary_ops[op]} gp{gp_src}, gp{gp_src}, f1, {1 if use_mask else 0}")

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

    def get_fp_table(self) -> Dict[str, Dict[str, object]]:
        table: Dict[str, Dict[str, object]] = {}
        for name, fp_var in self.tensor_manager.fp_vars.items():
            table[name] = {
                "kind": "fp_var",
                "addr": fp_var.fp_mem_addr,
                "size": fp_var.size,
                "dtype": fp_var.dtype,
                "storage": fp_var.storage,
            }
        for name, fragment in self.tensor_manager.fp_fragments.items():
            table[name] = {
                "kind": "fp_fragment",
                "shape": list(fragment.shape),
                "vars": {
                    ",".join(str(item) for item in index): {
                        "name": fp_var.name,
                        "addr": fp_var.fp_mem_addr,
                    }
                    for index, fp_var in fragment.vars.items()
                },
            }
        return table

    def get_tensor_table(self) -> Dict[str, Dict[str, object]]:
        table: Dict[str, Dict[str, object]] = {}
        for name, input_obj in self.tensor_manager.inputs.items():
            physical_shape = _logical_shape_to_physical_shape(input_obj.logical_shape)
            table[name] = {
                "kind": "input",
                "shape": physical_shape,
                "logical_shape": input_obj.logical_shape,
                "logical_layout": "bshd" if len(input_obj.logical_shape) == 4 else "2d",
                "hbm_group_obj": input_obj.metadata.get("hbm_group_obj", f"{name}.hbm"),
                "tiles": dict(input_obj.tiles),
            }
        for name, tensor in self.tensor_manager.tensors.items():
            physical_shape = _logical_shape_to_physical_shape(tensor.logical_shape)
            table[name] = {
                "kind": "tensor",
                "shape": physical_shape,
                "logical_shape": tensor.logical_shape,
                "logical_layout": "bshd" if len(tensor.logical_shape) == 4 else "2d",
                "tiles": dict(tensor.tiles),
            }
        return table

    def write_operation_report(self, output_path: str | Path) -> None:
        lines: List[str] = []
        for index, entry in enumerate(self.operation_log):
            kind = str(entry.get("kind", "unknown"))
            payload = ", ".join(
                f"{key}={value}"
                for key, value in entry.items()
                if key != "kind"
            )
            lines.append(f"log[{index}]: {kind}" + (f" | {payload}" if payload else ""))
        for index, op in enumerate(self.compute_manager.ops):
            op_kind = op.get("op_kind", "unknown")
            operands = op.get("operands")
            operand_summary = ""
            if isinstance(operands, tuple) and operands:
                control = operands[0]
                if control == "matmul" and len(operands) == 4:
                    _, src_pairs, dst, dst_tile = operands
                    operand_summary = (
                        f" src_pairs={len(src_pairs)}"
                        f" dst_value={getattr(dst, 'value_tile_id', None)}"
                        f" dst_tile={getattr(dst_tile, 'tile_id', None)}"
                        f" control={control}"
                    )
                elif control == "copy" and len(operands) == 3:
                    _, src_value, dst_tile = operands
                    operand_summary = (
                        f" src_value={getattr(src_value, 'value_tile_id', None)}"
                        f" dst_tile={getattr(dst_tile, 'tile_id', None)}"
                        f" control={control}"
                    )
            elif isinstance(operands, dict):
                src_pairs = operands.get("src_pairs", [])
                dst = operands.get("dst")
                dst_tile = operands.get("dst_tile")
                operand_summary = (
                    f" src_pairs={len(src_pairs)}"
                    f" dst_value={getattr(dst, 'value_tile_id', None)}"
                    f" dst_tile={getattr(dst_tile, 'tile_id', None)}"
                    f" control={operands.get('control')}"
                )
            lines.append(f"compute[{index}]: {op_kind}{operand_summary}")
        Path(output_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def write_tile_distribution_report(self, output_path: str | Path) -> None:
        lines = [
            f"inputs: {len(self.tensor_manager.inputs)}",
            f"tensors: {len(self.tensor_manager.tensors)}",
            f"input_tiles: {len(self.tensor_manager.input_tiles)}",
            f"tensor_tiles: {len(self.tensor_manager.tensor_tiles)}",
            f"value_tiles: {len(self.value_manager.value_tiles)}",
            f"value_tiles_in_vram: {len(self.value_manager._value_tiles_in_vram)}",
            f"value_tiles_in_mram: {len(self.value_manager._value_tiles_in_mram)}",
            f"value_tiles_in_hbm: {len(self.value_manager._value_tiles_in_hbm)}",
        ]
        Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

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
            self.operation_log.append(
                {
                    "kind": "normalize_large_addi_immediate",
                    "rd": rd,
                    "rs1": rs1,
                    "imm": imm_value,
                    "upper": upper,
                    "lower": lower,
                }
            )
        normalized = "\n".join(lines)
        if asm_code.endswith("\n"):
            normalized += "\n"
        return normalized

    def compile(self) -> str:
        self.compiler.generated_code = self._normalize_large_addi_immediates(self.compiler.generated_code)
        return self.compiler.generated_code


def _logical_shape_to_physical_shape(logical_shape: LogicalShape) -> Tuple[int, int]:
    if len(logical_shape) == 4:
        b, s, h, d = logical_shape
        return b * s, h * d
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


def _is_tile_object(tile: object) -> bool:
    return isinstance(tile, (TensorTile, InputTile))


def _unwrap_transposed_operand(operand: object) -> object:
    if isinstance(operand, (TensorTranspose, InputTranspose)):
        return operand.base
    return operand


def _is_transposed_operand(operand: object) -> bool:
    return isinstance(operand, (TensorTranspose, InputTranspose))


def _is_narrow_tile(tile: TensorTile | InputTile) -> bool:
    mlen = int(tile.metadata.get("mlen", tile.tile_shape[0]))
    return tile.tile_shape[0] != mlen or tile.tile_shape[1] != mlen


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


def _format_fp_index(index: FPIndex) -> str:
    return "".join(f"[{value}]" for value in index)


def _require_fp_addr(fp_var: FPVar) -> int:
    if fp_var.fp_mem_addr is None:
        raise RuntimeError(f"FPVar {fp_var.name!r} has no fp_mem_addr")
    return int(fp_var.fp_mem_addr)
