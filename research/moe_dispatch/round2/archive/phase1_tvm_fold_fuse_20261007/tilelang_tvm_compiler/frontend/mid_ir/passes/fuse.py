"""pass_5_fuse: collapse each Async region into a single MultiLaneOp.

Why this pass exists
--------------------

After ``async_wrap`` + ``view``, each cluster body holds:

  * ``Async(body=[one_can_async_op])`` — one async per op (strict)
  * bare ``can_async=False`` ops (Reduce, Elementwise w/ Broadcast)
  * possibly nested For / cluster / grid

We collapse each Async into one ``MultiLaneOp`` that:

  * carries the underlying op as ``inner``
  * names the enclosing cluster axes via ``cluster_axis_names``
    (outermost-to-innermost order)
  * exposes ``dim_map`` — for each lane-aware (non-global) buffer the
    op references, the list of physical dims that correspond to each
    cluster axis. Today every buffer's cluster dim is physical dim 0
    (pass_4b prepends the phase index there), so dim_map values are
    always ``[0]``. Multi-axis cluster fusion would put extra entries
    in this list.

What stays untouched
--------------------

  * Bare can_async=False ops in the cluster body — these are per-row
    ops that lower to ``for lane in range(cluster)`` at pass_8.
    pass_5 doesn't wrap them; they keep BufferRefs with view_perm.
  * RawStore, For, ParallelAxis structure
  * Buffer shapes
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cluster_guard import should_skip_cluster
from ..ir import (
    BufferRef,
    Broadcast,
    VarRef,
    Dma,
    Gemm,
    Elementwise,
    Reduce,
    RawStore,
    For,
    Async,
    MultiLaneOp,
    ParallelAxis,
    ParallelKind,
    MidFunc,
    Stmt,
)


class FuseError(RuntimeError):
    pass


# Per-``run`` lookup: name -> VarRef for non-cluster ParallelAxes. The
# CLUSTER walker reads it to find its sibling number axis's identity.
_NUMBER_VAR_BY_NAME: dict = {}


@dataclass
class _ClusterAxis:
    """One enclosing cluster axis as seen by fuse.

    Names are kept for HLIR dump / API surfaces (``cluster_axis_names``
    is a list of strings). Identity comparisons live on the VarRef
    fields; ``_collapse_lane_axis`` only compares by identity.

    Used by ``_collapse_lane_axis`` to recognise both:
      * Per-lane indices written as
        ``add(phase_var, mul(number_var, count))`` (produced by
        pass_4b_view for non-global buffers).
      * Bare ``VarRef`` matching the original lane var
        (``by``-equivalent), kept verbatim for global / global.* refs
        whose indices view skipped.
    Both forms collapse to ``ranged_slice(mul(number_var, count),
    count)`` so multi-lane sync ops read the full cluster's chunk in
    one go.
    """

    phase_name: str
    number_name: str
    count: int
    original_name: str
    phase_var: VarRef
    number_var: VarRef
    original_var: VarRef


# ---------------------------------------------------------------------------
# Cluster stack — track enclosing cluster axes outermost → innermost
# ---------------------------------------------------------------------------


def _collect_op_refs(op) -> list[BufferRef]:
    """Return every BufferRef the op directly references. Used to
    build dim_map."""
    refs: list[BufferRef] = []
    if isinstance(op, Dma):
        refs.extend([op.src, op.dst])
    elif isinstance(op, Gemm):
        refs.extend([op.a, op.b, op.c])
    elif isinstance(op, Elementwise):
        refs.append(op.dst)
        for s in op.srcs:
            if isinstance(s, Broadcast):
                refs.append(s.src)
            else:
                refs.append(s)
    elif isinstance(op, Reduce):
        refs.extend([op.dst, op.src])
    elif isinstance(op, RawStore):
        refs.append(op.dst)
    return refs


def _build_dim_map(op, cluster_axis_names: list[str]) -> dict[str, list[int]]:
    """For each non-global buffer the op touches, record the physical
    dims that map to each cluster axis (in cluster_axis_names order).

    Reads ``ref.buffer.cluster_dim`` directly — split sets it on
    cluster-expanded buffers and view/burn_view permute it along with
    any axis reshuffle. For ``global.*`` user caches (not
    cluster-expanded), ``cluster_dim is None`` and the buffer is
    excluded from the map.
    """
    n_axes = len(cluster_axis_names)
    out: dict[str, list[int]] = {}
    refs = op.list_refs() if hasattr(op, "list_refs") else _collect_op_refs(op)
    for ref in refs:
        if ref.buffer.scope == "global":
            continue
        cdim = ref.buffer.cluster_dim
        if cdim is None:
            continue
        # Single-axis cluster today: emit ``[cdim]`` for every entry
        # in cluster_axis_names. Multi-axis support would carry an
        # ordered list on the BufferDef.
        out[ref.buffer.name] = [cdim] * n_axes
    return out


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _walk(stmt: Stmt, cluster_stack: list[_ClusterAxis]) -> Stmt:
    if isinstance(stmt, ParallelAxis):
        if stmt.kind == ParallelKind.CLUSTER:
            if stmt.parent_grid_axis_name is None:
                raise FuseError(
                    f"cluster axis {stmt.axis_name!r} missing parent_grid_axis_name; pass_3_split should have set it"
                )
            # Read the user-visible original axis name straight off the
            # ParallelAxis (set by pass_3_split). Parsing string
            # suffixes (``"_phase"`` / ``"_number"``) used to work but
            # made the contract fragile against any future renaming
            # scheme; ``original_axis_name`` is the explicit channel.
            phase = stmt.axis_name
            original = stmt.original_axis_name
            if original is None:
                raise FuseError(f"cluster axis {phase!r} missing original_axis_name; pass_3_split should have set it")
            if stmt.axis_var is None or stmt.original_axis_var is None:
                raise FuseError(
                    f"cluster axis {phase!r}: identity fields (axis_var / original_axis_var) must be set by split"
                )
            number_var = _NUMBER_VAR_BY_NAME.get(stmt.parent_grid_axis_name)
            if number_var is None:
                raise FuseError(
                    f"cluster {phase!r}: number axis "
                    f"{stmt.parent_grid_axis_name!r} VarRef not recorded; "
                    f"split must emit ``number -> phase`` nesting"
                )
            new_stack = cluster_stack + [
                _ClusterAxis(
                    phase_name=phase,
                    number_name=stmt.parent_grid_axis_name,
                    count=stmt.extent,
                    original_name=original,
                    phase_var=stmt.axis_var,
                    number_var=number_var,
                    original_var=stmt.original_axis_var,
                )
            ]
            return ParallelAxis(
                axis_name=stmt.axis_name,
                extent=stmt.extent,
                body=[_walk(s, new_stack) for s in stmt.body],
                kind=stmt.kind,
                thread_tag=stmt.thread_tag,
                parent_grid_axis_name=stmt.parent_grid_axis_name,
                original_axis_name=stmt.original_axis_name,
                axis_var=stmt.axis_var,
                original_axis_var=stmt.original_axis_var,
            )
        # Non-cluster ParallelAxis. Record axis_var by name so a nested
        # CLUSTER can pick up the matching number VarRef.
        if stmt.axis_var is not None:
            _NUMBER_VAR_BY_NAME[stmt.axis_name] = stmt.axis_var
        return ParallelAxis(
            axis_name=stmt.axis_name,
            extent=stmt.extent,
            body=[_walk(s, cluster_stack) for s in stmt.body],
            kind=stmt.kind,
            thread_tag=stmt.thread_tag,
            parent_grid_axis_name=stmt.parent_grid_axis_name,
            original_axis_name=stmt.original_axis_name,
            axis_var=stmt.axis_var,
            original_axis_var=stmt.original_axis_var,
        )
    if isinstance(stmt, For):
        return For(
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=[_walk(s, cluster_stack) for s in stmt.body],
            kind=stmt.kind,
            loop_var_var=stmt.loop_var_var,
        )
    if isinstance(stmt, Async):
        return _fuse_async(stmt, cluster_stack)
    if isinstance(stmt, MultiLaneOp):
        # Already fused (idempotency). Recurse into inner just in case
        # — though typically inner is a leaf.
        return stmt
    # Leaf: pass through.
    return stmt


def _match_lane_composite(idx, axes: list[_ClusterAxis]):
    """If ``idx`` is exactly a lane index — either the bare original
    lane var, or the ``add(phase, mul(number, count))`` split form
    pass_4b_view produces — return its matching ``_ClusterAxis``.
    Otherwise return ``None``.

    This is the recogniser for "this expression IS one lane axis";
    ``_collapse_lane_axis`` uses it both for whole-axis indices and as
    the kernel of the ``lane ± const`` head-offset case.
    """
    if isinstance(idx, VarRef):
        for ax in axes:
            if idx == ax.original_var:
                return ax
        return None
    if isinstance(idx, dict) and idx.get("op") == "add":
        args = idx.get("args", [])
        if len(args) == 2 and isinstance(args[0], VarRef):
            phase = args[0]
            inner = args[1]
            if isinstance(inner, dict) and inner.get("op") == "mul":
                m_args = inner.get("args", [])
                if len(m_args) == 2 and isinstance(m_args[0], VarRef) and isinstance(m_args[1], int):
                    number, count = m_args[0], m_args[1]
                    for ax in axes:
                        if phase == ax.phase_var and number == ax.number_var and count == ax.count:
                            return ax
    return None


def _ranged_slice_for_axis(ax: _ClusterAxis, extra_offset=None):
    """Build ``ranged_slice(mul(number, count) [+ extra_offset], count)``
    for one cluster axis. ``extra_offset`` (a constant head offset, or
    None) is folded into the slice's START so the ranged_slice stays at
    the TOP of the index expression — downstream ``_ref_extents`` and
    ``_render_idx_as_primexpr`` only recognise a top-level ranged_slice.
    """
    base = {"op": "mul", "args": [ax.number_var, ax.count]}
    if extra_offset is not None:
        base = {"op": "add", "args": [base, extra_offset]}
    return {"op": "ranged_slice", "args": [base, ax.count]}


def _collapse_lane_axis(idx, axes: list[_ClusterAxis]):
    """Fold a per-lane index expression back into a cluster-wide
    ``ranged_slice``.

    pass_4b_view turns the original lane var (e.g. ``by``) into
    ``add(phase_var, mul(number_var, count))`` — that's the correct
    per-lane expression for op kinds that fire once per lane (Reduce,
    Broadcast Elementwise). For multi-lane ops (Async-wrapped DMA /
    btmm / pure Elementwise) the cluster fires the op exactly once
    across all lanes, so the same axis position should describe a span
    of ``count`` consecutive lane indices starting at the cluster's
    base — encoded as ``ranged_slice(mul(number_var, count), count)``.

    Three cases handled:
      1. The bare original lane var, or the ``add(phase, mul(number,
         count))`` split form — collapse straight to a ranged_slice.
      2. ``lane_composite ± const`` (a head-offset write, e.g.
         ``Y_hbm[..., by + 8, ...]``) — the constant is folded INTO
         the ranged_slice's start so the ranged_slice stays top-level
         and keeps ``extent == count``. Without this the constant add
         buries the ranged_slice one level down and ``_ref_extents``
         falls back to extent 1, writing only one lane's worth.
      3. Anything else — recurse into children (the lane composite may
         live deep inside a compound, e.g. ``mul(by_expr, stride)``).
    """
    # Case 1: idx is itself a lane axis.
    ax = _match_lane_composite(idx, axes)
    if ax is not None:
        return _ranged_slice_for_axis(ax)

    if not isinstance(idx, dict):
        return idx

    # Case 2: ``lane_composite + const`` or ``const + lane_composite``.
    # (Subtraction of a const is normalised by upstream IR builders to
    # an add of a negative IntImm, so matching ``add`` covers both.)
    if idx.get("op") == "add":
        args = idx.get("args", [])
        if len(args) == 2:
            a0, a1 = args
            for lane_arg, other in ((a0, a1), (a1, a0)):
                lane_ax = _match_lane_composite(lane_arg, axes)
                if lane_ax is not None and isinstance(other, int):
                    return _ranged_slice_for_axis(lane_ax, extra_offset=other)

    # Case 3: recurse into children.
    return {
        "op": idx.get("op"),
        "args": [_collapse_lane_axis(a, axes) for a in idx.get("args", [])],
    }


def _collapse_ref(ref: BufferRef, axes: list[_ClusterAxis]) -> BufferRef:
    """Apply ``_collapse_lane_axis`` to every index of a user-declared
    global ref (``global`` HBM or ``global.vram`` / ``global.mram`` /
    ``global.fpram`` on-chip caches).

    Non-global refs already had their lane axis baked into the
    prepended phase by pass_4b_view, so we only rewrite globals. For
    on-chip globals (``global.*``), the kernel author still indexes
    them with the un-split logical lane var ``by`` — fuse_pass widens
    that to a cluster-wide ``ranged_slice`` so the multi-lane sync
    wrap reads the full cluster's chunk (by_phase drops out)."""
    if not (ref.buffer.scope == "global" or ref.buffer.scope.startswith("global.")):
        return ref
    return BufferRef(
        buffer=ref.buffer,
        indices=[_collapse_lane_axis(i, axes) for i in ref.indices],
        view_perm=ref.view_perm,
    )


def _collapse_src(src, axes: list[_ClusterAxis]):
    if isinstance(src, Broadcast):
        return Broadcast(
            src=_collapse_ref(src.src, axes),
            broadcast_dims=list(src.broadcast_dims),
        )
    return _collapse_ref(src, axes)


def _collapse_lane_in_op(op, axes: list[_ClusterAxis]):
    """Rebuild ``op`` with HBM refs widened to cluster-wide ranged
    slices. Only HBM refs are touched; on-chip refs are unchanged.

    axes (per-axis info on each operand) are passed through verbatim
    — collapsing HBM lane slices only changes ``indices``/extents at
    the *value* level, not the axis-role tagging. axis-aware passes
    that need the new extent should consult ``ref.buffer.shape``.
    """
    if isinstance(op, Dma):
        return Dma(
            src=_collapse_ref(op.src, axes),
            dst=_collapse_ref(op.dst, axes),
            src_axes=list(op.src_axes),
            dst_axes=list(op.dst_axes),
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Gemm):
        return Gemm(
            a=_collapse_ref(op.a, axes),
            b=_collapse_ref(op.b, axes),
            c=_collapse_ref(op.c, axes),
            transpose_a=op.transpose_a,
            transpose_b=op.transpose_b,
            kind=op.kind,
            a_axes=list(op.a_axes),
            b_axes=list(op.b_axes),
            c_axes=list(op.c_axes),
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Elementwise):
        return Elementwise(
            dst=_collapse_ref(op.dst, axes),
            srcs=[_collapse_src(s, axes) for s in op.srcs],
            op=op.op,
            dst_axes=list(op.dst_axes),
            src_axes=[list(s) for s in op.src_axes],
            axis=op.axis,
            size=op.size,
            marker=op.marker,
            can_async=op.can_async,
        )
    return op


def _fuse_async(stmt: Async, cluster_stack: list[_ClusterAxis]) -> Stmt:
    """One Async → one MultiLaneOp. Async body must hold exactly one
    op (the strict one-async-one-op invariant from pass_4).

    During fusion we also collapse the per-lane index expressions
    inside the inner op's HBM refs back into cluster-wide ranged
    slices, since the resulting MultiLaneOp fires once across all
    lanes (not once per lane).
    """
    if not cluster_stack:
        raise FuseError(
            f"Async #{stmt.scope_id} found outside any cluster — this shouldn't happen if pass_4_async ran first"
        )
    if len(stmt.body) != 1:
        raise FuseError(
            f"Async #{stmt.scope_id} must hold exactly one op (got {len(stmt.body)}); pass_4 enforces one-async-one-op"
        )
    inner = stmt.body[0]
    if isinstance(inner, (Async, MultiLaneOp, ParallelAxis, For)):
        raise FuseError(f"Async #{stmt.scope_id} body must be a leaf op, got {type(inner).__name__}")
    inner = _collapse_lane_in_op(inner, cluster_stack)
    axis_names = [ax.phase_name for ax in cluster_stack]
    axis_vars = [ax.phase_var for ax in cluster_stack]
    return MultiLaneOp(
        inner=inner,
        cluster_axis_names=axis_names,
        cluster_axis_vars=axis_vars,
        dim_map=_build_dim_map(inner, axis_names),
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc) -> MidFunc:
    """Collapse Async regions into MultiLaneOp nodes."""
    if should_skip_cluster(func):
        return func
    _NUMBER_VAR_BY_NAME.clear()
    return MidFunc(
        name=func.name,
        params=list(func.params),
        allocs=list(func.allocs),
        body=[_walk(s, cluster_stack=[]) for s in func.body],
        lane_axes=list(func.lane_axes),
        cluster_counts=list(func.cluster_counts),
        attrs=dict(func.attrs),
    )


__all__ = ["FuseError", "run"]
