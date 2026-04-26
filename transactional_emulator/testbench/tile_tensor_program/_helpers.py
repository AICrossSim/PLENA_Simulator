"""Module-level helper functions used by managers and TileTensorProgram."""

from __future__ import annotations

from typing import Dict, List, Tuple

from ._types import *  # noqa: F401,F403


__all__ = [
    "_logical_shape_to_physical_shape",
    "_logical_selectors_to_physical_ranges",
    "_slice_item_to_range",
    "_ranges_overlap",
    "_tiles_in_grid_order",
    "_bshd_tile_batch_index",
    "_bshd_tile_seq_block",
    "_is_tile_object",
    "_is_full_element_index",
    "_contains_parallel_selector",
    "_normalize_index",
    "_tile_owner_name",
    "_logical_shape_to_hbm_stride",
    "_tile_coord_to_hbm_offset",
    "_logical_3d_selectors_to_flat_col_range",
    "_logical_indices_to_physical_coord",
    "_physical_tile_coord_to_fp_index",
    "_vector_tile_row_fp_groups",
    "_unwrap_transposed_operand",
    "_is_transposed_operand",
    "_is_narrow_tile",
    "_is_fp_domain_operand",
    "_is_parallel_graph_operand",
    "_coerce_parallel_expr",
    "_collect_parallel_accesses",
    "_collect_parallel_predicates",
    "_parallel_access_identity",
    "_parallel_expr_identity",
    "_infer_parallel_load_metadata",
    "_infer_parallel_predicate_kind",
    "_build_parallel_execution_plan",
    "_infer_parallel_elem_width",
    "_build_parallel_cycle_plan",
    "_iter_fp_indices",
    "_iter_logical_indices",
    "_iter_selected_logical_indices",
    "_format_fp_index",
    "_require_fp_addr",
    "_fp_fragment_shape_to_tile_shape",
    "_fp_fragment_row_fp_vars",
]


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
