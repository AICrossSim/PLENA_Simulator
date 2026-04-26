"""ValueManager: backing-value bindings, view resolution, residency, write prep."""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403


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
        self.isa_emitter = program.isa_emitter
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
                    self.isa_emitter.emit_map_v_fp_tile(
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
                    self.isa_emitter.emit_fp_kernel(
                        src1_addrs=row_addrs,
                        dst_addrs=scratch_addrs,
                        op="copy",
                        task_id=f"fpram_row_gather.{value.value_tile_id}.row{row_index}",
                    )
                    self.isa_emitter.emit_map_v_fp_tile(
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
            self.isa_emitter.emit_store_tile_to_hbm(
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
            self.isa_emitter.emit_load_tile_from_hbm(
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
            self.isa_emitter.emit_hbm_tile_to_mram(
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
                    self.isa_emitter.emit_map_fp_v_tile(
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
                    self.isa_emitter.emit_map_fp_v_tile(
                        fpram_addr=int(scratch_addr),
                        vram_addr=row_vram_addr,
                        row_count=1,
                        row_width=int(row_width),
                        task_id=f"vram_to_fpram.{value.value_tile_id}.row{row_index}.scratch",
                    )
                    self.isa_emitter.emit_fp_kernel(
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


