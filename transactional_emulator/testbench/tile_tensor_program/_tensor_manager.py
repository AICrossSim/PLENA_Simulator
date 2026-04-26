"""TensorManager: logical tensor/input objects, tile creation, slice resolution."""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional, Sequence, Tuple

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403


class TensorManager:
    """Manage logical tensors, tiles, slices, and tensor-thread grouping.

    TensorManager operates on logical objects only. It owns shape flattening,
    tile metadata, slice resolution, and `mapt` grouping. It deliberately does
    not create ValueTile / ValueTileView objects and does not decide
    residency placement; that work stays in ValueManager.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.isa_emitter = program.isa_emitter
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


