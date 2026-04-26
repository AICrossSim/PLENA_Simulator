"""ThreadManager: parallel thread regions, expression graphs, cache planning."""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional, Tuple

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403


class ThreadManager:
    """Manage parallel thread regions, expression graphs, and cache planning.

    This layer is intentionally FP-first for now. It owns the symbolic
    `parallel_region3d` flow and keeps the graph/cache planning state out of
    `TileTensorProgram` so later lowering can evolve independently.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.isa_emitter = program.isa_emitter
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
            self.isa_emitter.emit_fp_kernel(
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
            self.isa_emitter.emit_fp_kernel(
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
            self.isa_emitter.emit_fp_kernel(
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
            self.isa_emitter.emit_map_fp_v_tile(
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
            self.isa_emitter.emit_map_v_fp_tile(
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


