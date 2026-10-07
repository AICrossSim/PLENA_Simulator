"""pass_1_fold: raw tir.PrimFunc → mid_ir.MidFunc.

What this pass folds
--------------------

Five raw-TIR shapes get collapsed into one mid_ir node each:

    tl.tileop.copy(region(src), region(dst))
        → Dma(src, dst)

    tl.tileop.gemm_py(region(A), region(B), region(C), ...)
    [optionally wrapped by ``T.attr(0, "plena.gemm_kind", "btmm")``]
        → Gemm(A, B, C, kind=<btmm | overwrite>)

    tl.tileop.reduce(region(src), region(dst), dim, clear)
        → Reduce(dst, src, op, axis)

    for i in T.Parallel(N):
        dst[..., i] = A[..., i] OP B[..., i]
        → Elementwise(dst, [A, B], BinOp.<op>)            # whole-buffer
    for i in T.Parallel(N):
        dst[..., i] = T.exp(A[..., i])
        → Elementwise(dst, [A], UnaryOp.EXP, axis=-1)

    for i in T.serial(N):
        dst[i] = scalar_expr_of(A[i], B[i])               # 1D fp scalar update
        → Elementwise(dst, [A, B], BinOp.<op>) or Reduce / Broadcast

Anything that doesn't match one of the above is preserved as a raw
``RawStmt`` wrapper for the next pass to look at — but for the
flash_attention_min op set everything is expected to fold.

Structure-preserving wrappers (For with thread_tag, AttrStmt for
KIND, SeqStmt, BlockRealize) are translated to mid_ir's For + body
list as appropriate. Raw structure isn't carried over verbatim — the
output is purely mid_ir nodes.

Scope
-----

Only handles the rounded ops the kernel test set exercises today:
add / sub / mul / max / exp / reci / copy / 0-fill (zero_v) / sum-reduce
/ max-reduce. Anything else (FloatImm in store other than 0, DivNode
RHS, Cast in expr) raises ``FoldError`` so we notice early — better
than silently emitting a malformed mid_ir node.

Limitations / explicit gaps for later
-------------------------------------

  * Compound RHS (a*b + c*d) is rejected — relies on
    ``lower_compound_fp_stores`` running first.
  * IfThenElse: kernels don't use it; raises FoldError.
  * Match-buffers / non-trivial Block.alloc_buffers: passed through
    via best-effort BufferDef synthesis.
  * Reduce's ``clear`` flag is read but mid_ir doesn't represent it
    yet — we store it on Reduce.attrs for the lowering pass to read.
"""

from __future__ import annotations


from tvm import tir

from ..ir import (
    BinOp,
    UnaryOp,
    ReduceOp,
    AxisRole,
    AxisInfo,
    BufferDef,
    BufferRef,
    Slice,
    VarRef,
    Elementwise,
    Broadcast,
    Reduce,
    Gemm,
    Dma,
    RawStore,
    For,
    MidFunc,
    ParallelAxis,
    ParallelKind,
)


_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REDUCE = "tl.tileop.reduce"
_TILEOP_REGION = "tl.tileop.region"
_KIND_KEY = "plena.gemm_kind"
_LANE_AXIS_FUNC_ATTR = "plena.lane_axis"


class FoldError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Per-fold Var registry
# ---------------------------------------------------------------------------
#
# Canonicalises ``tir.Var`` -> ``VarRef`` within one fold call.
#
# Keyed by ``id(var)`` so a given ``tir.Var`` object always yields the
# same ``VarRef`` instance — identity comparisons on ``VarRef`` are
# stable across multiple visits of the same underlying var.
#
# Same-name different-object vars are explicitly *allowed*. That's the
# whole point of moving off bare-string indices: each ``tir.Var`` keeps
# its own identity even when ``name_hint`` collides (e.g. two unrelated
# ``row`` vars in the same PrimFunc). Two ``VarRef``s wrapping
# different ``tir.Var``s with the same name will compare unequal via
# ``var.same_as`` — exactly the contract we want.
#
# Reset at every call to :func:`run`.


class _VarRegistry:
    def __init__(self) -> None:
        self._by_id: dict[int, VarRef] = {}
        # Keep a strong reference to each Var so its id() can't be
        # recycled inside this fold call.
        self._anchor: list[object] = []

    def ref(self, var) -> VarRef:
        existing = self._by_id.get(id(var))
        if existing is not None:
            return existing
        new_ref = VarRef(var)
        self._by_id[id(var)] = new_ref
        self._anchor.append(var)
        return new_ref


# Active registry for the current ``run`` invocation. Module-level
# (vs threaded through every helper) because the recursion already
# fans out through many small functions; passing it would force a
# wide signature change for no benefit. Reset at the top of ``run``.
_active_registry: _VarRegistry | None = None


def _vref(var) -> VarRef:
    """Canonical ``VarRef`` for ``var`` in the active fold call."""
    if _active_registry is None:
        raise FoldError("_vref called outside a fold ``run`` — registry not active")
    return _active_registry.ref(var)


def _assert_no_str_in_indices(stmts) -> None:
    """Walk ``stmts`` and assert no BufferRef.indices entry is a bare
    ``str``. Bare-string indices were the pre-VarRef cheat; fold output
    must be VarRef-only."""

    def visit_ref(ref: BufferRef) -> None:
        for i, idx in enumerate(ref.indices):
            _check_idx(idx, ref.buffer.name, i)

    def _check_idx(idx, buf_name, pos) -> None:
        if isinstance(idx, str):
            raise FoldError(
                f"fold produced a bare-string index in BufferRef "
                f"{buf_name}[..pos {pos}..] = {idx!r}. mid_ir now requires "
                f"VarRef; investigate the producer."
            )
        if isinstance(idx, dict):
            for a in idx.get("args", []):
                _check_idx(a, buf_name, pos)

    def visit_src(src) -> None:
        if isinstance(src, Broadcast):
            visit_ref(src.src)
        else:
            visit_ref(src)

    def walk(s) -> None:
        if isinstance(s, Elementwise):
            visit_ref(s.dst)
            for x in s.srcs:
                visit_src(x)
        elif isinstance(s, Reduce):
            visit_ref(s.dst)
            visit_ref(s.src)
        elif isinstance(s, Dma):
            visit_ref(s.src)
            visit_ref(s.dst)
        elif isinstance(s, Gemm):
            visit_ref(s.a)
            visit_ref(s.b)
            visit_ref(s.c)
        elif isinstance(s, RawStore):
            visit_ref(s.dst)
        # Recurse into structural nodes.
        body = getattr(s, "body", None)
        if isinstance(body, list):
            for c in body:
                walk(c)

    for s in stmts:
        walk(s)


# ---------------------------------------------------------------------------
# Call kind helpers (mirror prior passes; kept local for self-containment)
# ---------------------------------------------------------------------------


def _call_kind(call: tir.Call) -> str | None:
    if not isinstance(call, tir.Call):
        return None
    op_name = getattr(call.op, "name", "")
    if op_name and not op_name.startswith("tir."):
        return op_name
    if op_name == "tir.call_extern" and call.args:
        head = call.args[0]
        if isinstance(head, tir.StringImm):
            return str(head.value)
    return None


def _call_args(call: tir.Call) -> list:
    op_name = getattr(call.op, "name", "")
    if op_name == "tir.call_extern" and call.args:
        return list(call.args[1:])
    return list(call.args)


# ---------------------------------------------------------------------------
# Buffer table
# ---------------------------------------------------------------------------


def _scope_string(buf: tir.Buffer, default: str) -> str:
    s = getattr(buf, "scope", None)
    if callable(s):
        try:
            return str(s())
        except Exception:
            return default
    if isinstance(s, str):
        return s
    return default


def _shape_ints(buf: tir.Buffer) -> list[int]:
    out = []
    for d in buf.shape:
        if isinstance(d, tir.IntImm):
            out.append(int(d.value))
        elif isinstance(d, int):
            out.append(int(d))
        else:
            raise FoldError(f"buffer {buf.name!r} has symbolic dim {d!r}; mid_ir is int-shape-only at this stage")
    return out


def _buffer_def(buf: tir.Buffer, default_scope: str = "global") -> BufferDef:
    shape = _shape_ints(buf)
    scope = _scope_string(buf, default_scope)
    # Hard rule: 1D ``shared`` (VRAM tile) buffers are rejected.
    # Reason: fold's broadcast detection in ``_wrap_src`` only flips a
    # same-rank operand into ``Broadcast`` when the src ref's rank is
    # strictly shorter than dst's. A 1D shared dst paired with a 1D
    # ``fp[0]``-style scalar src therefore fails to fold (same rank,
    # but the src is logically a scalar broadcast). Force authors to
    # write ``(1, N)`` instead; downstream tile/row machinery is built
    # around ≥2D shared anyway.
    if scope.startswith("shared") and len(shape) == 1:
        raise FoldError(
            f"1D shared buffer {buf.name!r} (shape={shape}) is not "
            f"supported. Use ``T.alloc_shared((1, N), ...)`` instead — "
            f"1D shared loses the broadcast-axis fold needed for "
            f"`vram[i] op fp_scalar[0]` to lower to a vector op."
        )
    return BufferDef(
        name=buf.name,
        shape=shape,
        dtype=str(buf.dtype),
        scope=scope,
    )


# ---------------------------------------------------------------------------
# Raw-TIR → mid_ir IndexExpr conversion
# ---------------------------------------------------------------------------


def _index_expr(expr) -> int | VarRef | dict:
    """Convert a TIR PrimExpr appearing as an index into a mid_ir
    IndexExpr (int / VarRef / dict). Compound arithmetic becomes a
    ``{"op": "<add/mul/...>", "args": [...]}`` dict; passes that need
    to manipulate it can parse the dict.

    A vectorized index (``tir.Ramp(base, stride, lanes)``) is treated
    as a contiguous slice and encoded as
    ``{"op": "ramp", "args": [base, stride, lanes]}``. Callers that
    can collapse a full-range ramp (base=0, stride=1, lanes=buffer_dim)
    into a ``Slice`` should do so before calling here, but the encoding
    survives if they don't.

    ``tir.Var`` -> ``VarRef`` (identity-typed). The active fold-call
    registry canonicalises so visits of the same Var object always
    yield the same VarRef, and two distinct Vars sharing a name raise.
    """
    if isinstance(expr, (int,)):
        return int(expr)
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if isinstance(expr, tir.Var):
        return _vref(expr)
    if isinstance(expr, tir.Add):
        return {"op": "add", "args": [_index_expr(expr.a), _index_expr(expr.b)]}
    if isinstance(expr, tir.Sub):
        return {"op": "sub", "args": [_index_expr(expr.a), _index_expr(expr.b)]}
    if isinstance(expr, tir.Mul):
        return {"op": "mul", "args": [_index_expr(expr.a), _index_expr(expr.b)]}
    if isinstance(expr, tir.FloorDiv):
        return {"op": "fdiv", "args": [_index_expr(expr.a), _index_expr(expr.b)]}
    if isinstance(expr, tir.FloorMod):
        return {"op": "fmod", "args": [_index_expr(expr.a), _index_expr(expr.b)]}
    if isinstance(expr, tir.Ramp):
        return {
            "op": "ramp",
            "args": [_index_expr(expr.base), _index_expr(expr.stride), int(expr.lanes)],
        }
    raise FoldError(f"unsupported index expression type {type(expr).__name__}: {expr!r}")


def _is_full_range_ramp_for_dim(idx, dim_extent: int) -> bool:
    """True if ``idx`` is a Ramp(0, 1, dim_extent) — equivalent to a
    whole-axis Slice on a buffer dim of size ``dim_extent``."""
    if isinstance(idx, tir.Ramp):
        if (
            isinstance(idx.base, tir.IntImm)
            and int(idx.base.value) == 0
            and isinstance(idx.stride, tir.IntImm)
            and int(idx.stride.value) == 1
            and int(idx.lanes) == dim_extent
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Region call → BufferRef
# ---------------------------------------------------------------------------


def _region_to_ref(call: tir.Call, buf_table: dict[str, BufferDef]) -> BufferRef:
    """Convert ``tl.tileop.region(BufferLoad(buf, [starts]), mode, *extents)``
    into a BufferRef.

    Indexing convention for mid_ir:
      * Where extent equals the buffer's full extent on that axis,
        the index is a Slice (whole-axis access).
      * Otherwise the index is the BufferLoad's start (a literal int,
        a var name, or a compound expression).

    This matches the way the kernel author wrote the access: a sliced
    access ``Q_hbm[0, q*rows, by, 0]`` has whole-axis on dims 1 and 3
    (not really — but on our flash_attention_min slice with extents
    [1, rows, 1, hlen] only dims 1 and 3 cover the full HBM axis, so
    mark only those as Slice). The point: mid_ir's BufferRef tells
    later passes whether a dim is "fully consumed" so cluster-dim
    rewrites know what to leave alone vs index into.
    """
    # tilelang's reduce ABI sometimes passes a bare BufferLoad as the
    # src/dst arg instead of wrapping it in a tl.tileop.region call.
    # Treat that as a whole-buffer reference: any dim whose index is
    # 0-IntImm OR a full-range Ramp(0, 1, dim_extent) becomes Slice;
    # everything else is the literal index. (Ramp shows up because
    # tilelang's reduce passes vectorized loads — Ramp(0,1,N) means
    # "the whole N-wide range".)
    if isinstance(call, tir.BufferLoad):
        load = call
        buf_def = buf_table.get(load.buffer.name)
        if buf_def is None:
            buf_def = _buffer_def(load.buffer, default_scope="shared")
            buf_table[load.buffer.name] = buf_def
        indices: list = []
        for axis, idx in enumerate(load.indices):
            dim_extent = int(buf_def.shape[axis]) if axis < len(buf_def.shape) else None
            if isinstance(idx, tir.IntImm) and int(idx.value) == 0:
                indices.append(Slice())
            elif dim_extent is not None and _is_full_range_ramp_for_dim(idx, dim_extent):
                indices.append(Slice())
            else:
                indices.append(_index_expr(idx))
        return BufferRef(buffer=buf_def, indices=indices)

    args = _call_args(call)
    if not args:
        raise FoldError("empty region call")
    load = args[0]
    if not isinstance(load, tir.BufferLoad):
        raise FoldError(f"region first arg is not a BufferLoad: {load!r}")
    starts = list(load.indices)
    extents = list(args[2:])  # args[1] is mode
    if len(starts) != len(extents):
        raise FoldError(f"region rank mismatch: starts={len(starts)} extents={len(extents)}")
    # Reconcile to a BufferDef (the buffer may have been seen via
    # decl_buffer here for the first time).
    buf_def = buf_table.get(load.buffer.name)
    if buf_def is None:
        buf_def = _buffer_def(load.buffer, default_scope="shared")
        buf_table[load.buffer.name] = buf_def
    indices: list = []
    for axis, (s, e) in enumerate(zip(starts, extents)):
        if not isinstance(e, tir.IntImm):
            raise FoldError(f"region extent on axis {axis} of {buf_def.name!r} is non-static: {e!r}")
        e_int = int(e.value)
        # Slice when extent matches buffer dim AND the start is 0
        # OR start is a full-range Ramp (vectorized whole-axis load).
        is_zero_start = isinstance(s, tir.IntImm) and int(s.value) == 0
        is_full_ramp = _is_full_range_ramp_for_dim(s, e_int)
        if e_int == buf_def.shape[axis] and (is_zero_start or is_full_ramp):
            indices.append(Slice())
        elif e_int > 1:
            # Partial range with extent > 1: preserve both start and extent
            # via a compound "ranged_slice" expression so downstream passes
            # (to_plena _ref_extents / _render_idx) can recover the tile
            # width. Keeps the mid_ir IndexExpr taxonomy unchanged.
            indices.append(
                {
                    "op": "ranged_slice",
                    "args": [_index_expr(s), e_int],
                }
            )
        else:
            indices.append(_index_expr(s))
    return BufferRef(buffer=buf_def, indices=indices)


# ---------------------------------------------------------------------------
# BufferStore RHS → mid_ir Op recogniser
# ---------------------------------------------------------------------------


_BIN_NODE_TO_OP = {
    tir.Add: BinOp.ADD,
    tir.Sub: BinOp.SUB,
    tir.Mul: BinOp.MUL,
    # Max / Min are tir.Max / tir.Min; tested below.
}


def _peel_cast_roundtrip(expr, target_dtype: str | None = None):
    """Strip TVM-inserted fp16↔fp32 cast roundtrips.

    TVM lowers ``fp16 = 1.0 / fp16`` to
    ``Cast(fp16, Cast(fp32, 1.0) / Cast(fp32, x_fp16))``. PLENA HW does
    the reciprocal in fp16 natively, so for fold-pattern matching we
    want to see the inner expression as if no widening happened.

    The strategy:
      1. If ``expr`` is ``Cast(T, x)`` where ``x.dtype == T`` → return
         ``x`` (no-op cast).
      2. If ``expr`` is ``Cast(T, Cast(_, x))`` where ``x.dtype == T``
         → return ``x`` (widen-then-narrow roundtrip).
      3. If ``expr`` is ``Cast(T, arith_expr)`` where ``arith_expr`` is
         a Div / Add / Sub / Mul / Max / Min / unary Call whose operands
         are themselves ``Cast(_, leaf)`` widening originals from dtype
         ``T`` → return a rebuilt ``arith_expr`` whose operands are the
         peeled leaves (so the whole thing is now in dtype ``T``).
      4. Otherwise return ``expr`` unchanged.
    """
    if not isinstance(expr, tir.Cast):
        return expr
    target = str(expr.dtype) if target_dtype is None else target_dtype
    inner = expr.value
    # Rule 2: nested cast roundtrip.
    if isinstance(inner, tir.Cast):
        innermost = inner.value
        if str(innermost.dtype) == target:
            return innermost
        return expr
    # Rule 1: redundant same-dtype cast.
    inner_dtype = getattr(inner, "dtype", None)
    if inner_dtype is not None and str(inner_dtype) == target:
        return inner

    # Rule 3: widen-op-narrow over arithmetic. Peel each operand and,
    # if every operand is either a leaf already in ``target`` or a
    # ``Cast(_, x)`` from ``target``, rebuild the arith expression at
    # ``target`` dtype.
    def _peel_to_target(e):
        if isinstance(e, tir.Cast):
            x = e.value
            if str(getattr(x, "dtype", "")) == target:
                return x
            # Constant under cast (e.g. Cast(fp32, FloatImm(1.0))) —
            # re-emit the constant at target dtype directly.
            if isinstance(x, tir.IntImm):
                return tir.IntImm(target, int(x.value))
            if isinstance(x, tir.FloatImm):
                return tir.FloatImm(target, float(x.value))
            return None
        # Bare literals: re-emit at target dtype so binop dtypes match.
        if isinstance(e, tir.IntImm):
            return tir.IntImm(target, int(e.value))
        if isinstance(e, tir.FloatImm):
            return tir.FloatImm(target, float(e.value))
        if str(getattr(e, "dtype", "")) == target:
            return e
        return None

    cls = type(inner)
    if cls in (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.Max, tir.Min):
        a = _peel_to_target(inner.a)
        b = _peel_to_target(inner.b)
        if a is not None and b is not None:
            return cls(a, b)
        return expr
    if isinstance(inner, tir.Call) and len(inner.args) == 1:
        a = _peel_to_target(inner.args[0])
        if a is not None:
            return tir.Call(target, inner.op, [a])
        return expr
    return expr


def _try_bin_op(node) -> BinOp | None:
    cls = type(node)
    if cls in _BIN_NODE_TO_OP:
        return _BIN_NODE_TO_OP[cls]
    if cls is tir.Max:
        return BinOp.MAX
    if cls is tir.Min:
        return BinOp.MIN
    return None


def _try_unary_call(node) -> UnaryOp | None:
    """Recognise T.exp(x), T.sqrt(x). Reciprocal shows up as
    ``1.0 / x`` (a tir.Div), not as a Call — handled separately."""
    if not isinstance(node, tir.Call):
        return None
    op_name = getattr(node.op, "name", "")
    if op_name == "tir.exp":
        return UnaryOp.EXP
    if op_name == "tir.sqrt":
        return UnaryOp.SQRT
    return None


# ---------------------------------------------------------------------------
# Per-loop folder (T.Parallel / T.serial elementwise)
# ---------------------------------------------------------------------------


def _store_to_ref(store: tir.BufferStore, buf_table: dict[str, BufferDef]) -> BufferRef:
    name = store.buffer.name
    if name not in buf_table:
        buf_table[name] = _buffer_def(store.buffer, default_scope="shared")
    return BufferRef(
        buffer=buf_table[name],
        indices=[_index_expr(i) for i in store.indices],
    )


def _load_to_ref(load: tir.BufferLoad, buf_table: dict[str, BufferDef]) -> BufferRef:
    name = load.buffer.name
    if name not in buf_table:
        buf_table[name] = _buffer_def(load.buffer, default_scope="shared")
    return BufferRef(
        buffer=buf_table[name],
        indices=[_index_expr(i) for i in load.indices],
    )


def _try_fold_parallel(for_stmt: tir.For, buf_table: dict[str, BufferDef]) -> Elementwise | None:
    """``for i in T.Parallel(N): dst[..., i] = expr(loads_at_..._i)``
    → Elementwise.

    Produces ``axis=-1`` (acts on the last axis only — other axes are
    independent / fired once per element) and ``size=N`` (one HW
    vector instruction covers ``N`` elements in one issue)."""
    if for_stmt.kind != tir.ForKind.PARALLEL:
        return None
    body = for_stmt.body
    if not isinstance(body, tir.BufferStore):
        return None
    extent = int(for_stmt.extent.value) if isinstance(for_stmt.extent, tir.IntImm) else 1
    return _try_fold_store(
        body,
        for_stmt.loop_var,
        buf_table,
        axis=-1,
        size=extent,
    )


def _index_exprs_equal(a, b) -> bool:
    """Cheap structural equality on two mid_ir IndexExpr values. Used
    to decide whether a src's index tuple is a prefix of dst's (which
    is how we detect a broadcast)."""
    if isinstance(a, Slice) and isinstance(b, Slice):
        return True
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, VarRef) and isinstance(b, VarRef):
        return a == b  # identity-based equality
    if isinstance(a, dict) and isinstance(b, dict):
        if a.get("op") != b.get("op"):
            return False
        aa, bb = a.get("args", []), b.get("args", [])
        if len(aa) != len(bb):
            return False
        return all(_index_exprs_equal(x, y) for x, y in zip(aa, bb))
    return False


def _wrap_src(
    load: tir.BufferLoad,
    dst_indices: list,
    buf_table: dict[str, BufferDef],
    dst_buf: BufferDef | None = None,
) -> BufferRef | Broadcast | None:
    """Convert a BufferLoad src into either a plain BufferRef (when
    its index tuple matches dst's exactly) or a Broadcast (when it's
    a prefix — fewer trailing dims).

    Broadcast detection rule: src.indices == dst.indices[:len(src)]
    AND len(src) < len(dst). The missing trailing dims become
    ``broadcast_dims``.

    Special case: FPRAM scalar fragment stores (rank-1
    ``local.fragment`` dst) lower to one ``S_*_FP`` per call, with
    every operand carrying its own independent scalar address. Index
    matching is irrelevant — return the BufferRef as-is so the FPRAM
    scalar emitter can use ``src.indices`` directly. Used by kernels
    like RoPE that compute pair-swap addresses ``X_FP[2*i+1]`` against
    ``__tmp_fp_0[2*i]``.

    Anything else (mismatched non-prefix shapes on SIMD-style refs)
    returns None so fold fails loudly.
    """
    src_ref = _load_to_ref(load, buf_table)
    src_idx = src_ref.indices
    is_fpram_scalar_dst = (
        dst_buf is not None
        and dst_buf.scope in ("local.fragment", "fragment", "fragment.fpram")
        and len(dst_buf.shape) == 1
    )
    if is_fpram_scalar_dst:
        # FPRAM scalar lower-cycle: each operand is just a scalar
        # address, no SIMD axis to align.
        return src_ref
    if len(src_idx) == len(dst_indices):
        # Same rank — every position must be index-equal to dst's
        # for this to be a valid per-element op.
        if all(_index_exprs_equal(s, d) for s, d in zip(src_idx, dst_indices)):
            return src_ref
    elif len(src_idx) < len(dst_indices):
        # Broadcast: src must equal a prefix of dst.
        prefix = dst_indices[: len(src_idx)]
        if all(_index_exprs_equal(s, p) for s, p in zip(src_idx, prefix)):
            broadcast_dims = list(range(len(src_idx), len(dst_indices)))
            return Broadcast(src=src_ref, broadcast_dims=broadcast_dims)
    return None


def _to_raw_store(store: tir.BufferStore, buf_table: dict[str, BufferDef]) -> RawStore:
    """Wrap a BufferStore as an opaque RawStore.

    The ``dst`` BufferRef is computed from the store's indices via the
    same ``_index_expr`` machinery as elsewhere — so e.g. ``buf[MLEN+k]``
    becomes ``BufferRef(buf, [{op:add, args:[64, "k"]}])`` and the
    ``value`` payload is the raw ``store.value`` PrimExpr (opaque).
    """
    return RawStore(
        dst=_store_to_ref(store, buf_table),
        value=store.value,
    )


def _axes_for_ref(ref: BufferRef, simd_axis: int | None, simd_size: int) -> list[AxisInfo]:
    """Build per-axis AxisInfo for a BufferRef given the op's SIMD context.

    Each axis carries its **buffer-declared extent** plus a role:
      * SIMD with extent ``simd_size`` for the axis the op vectorises
        along (``simd_axis`` index, negative wraps).
      * BATCH with the buffer's declared extent for every other axis.
        The op fan-outs that many times along that dim; whether or
        not an outer for-loop is currently in scope at fold time is
        a fold detail (the for may even get absorbed) — the axis's
        fan-out cardinality stays the same regardless. Lower wraps
        outer fors based on this extent.

    ``simd_axis is None`` → whole-buffer SIMD: every dim is SIMD at
    its full extent.

    CLUSTER role is never assigned at this point — the cluster axis
    is prepended later by pass_3_split / pass_4b_view alongside the
    indices change.
    """
    out: list[AxisInfo] = []
    rank = len(ref.indices)
    normalised_simd = simd_axis + rank if (simd_axis is not None and simd_axis < 0) else simd_axis
    for dim, extent_decl in enumerate(ref.buffer.shape):
        full = int(extent_decl)
        if normalised_simd is not None and dim == normalised_simd:
            out.append(AxisInfo(role=AxisRole.SIMD, extent=int(simd_size)))
        elif simd_axis is None:
            out.append(AxisInfo(role=AxisRole.SIMD, extent=full))
        else:
            out.append(AxisInfo(role=AxisRole.BATCH, extent=full))
    return out


def _axes_for_broadcast_src(
    bc: Broadcast,
    simd_axis: int | None,
    simd_size: int,
) -> list[AxisInfo]:
    """Per-axis AxisInfo for the inner ref of a Broadcast.

    The dst rank exceeds the src rank by ``len(bc.broadcast_dims)``;
    those dst-side axes are tagged BROADCAST and don't appear in the
    src's axes list at all. The src's own axes use the same rules as
    ``_axes_for_ref`` but with the SIMD axis index translated to its
    position inside the src's (shorter) shape, if applicable.
    """
    # Dst-side SIMD axis index. If it lives in a broadcast_dim, the
    # src side has no corresponding axis to be SIMD; we leave all of
    # the src's axes as BATCH (its values are broadcast onto the SIMD
    # dim from outside).
    rank_src = len(bc.src.indices)
    if simd_axis is None:
        return [AxisInfo(role=AxisRole.SIMD, extent=int(d)) for d in bc.src.buffer.shape]
    # Map dst-side simd index → src-side index by counting how many
    # broadcast_dims sit at or before it.
    bd_set = set(bc.broadcast_dims)
    if simd_axis in bd_set:
        src_simd: int | None = None
    else:
        # Number of broadcast_dims with smaller index — drop them from
        # the dst index to land on the matching src dim.
        offset = sum(1 for b in bd_set if b < simd_axis)
        src_simd = simd_axis - offset
        if not (0 <= src_simd < rank_src):
            src_simd = None
    return _axes_for_ref(bc.src, src_simd, simd_size)


def _try_fold_store(
    store: tir.BufferStore,
    parallel_var: tir.Var | None,
    buf_table: dict[str, BufferDef],
    axis: int | None = None,
    size: int = 1,
) -> Elementwise | None:
    """Recognise the RHS of ``store`` as a mid_ir-expressible
    Elementwise. Returns None on no-match — caller is responsible for
    falling back to ``RawStore`` rather than raising. Never raises.

    ``size`` is the per-issue element count for the resulting
    Elementwise (see :class:`Elementwise.size`): 1 for a scalar store
    (SISD), N for a folded ``T.Parallel(N)`` vector store (SIMD).
    """
    if not store.indices:
        return None
    if parallel_var is not None:
        last = store.indices[-1]
        if not (isinstance(last, tir.Var) and last.same_as(parallel_var)):
            return None
    # Compound indices (e.g. ``buf[2 * i + 1] = ...``) are kept as
    # affine PrimExpr in the resulting Elementwise/BufferRef. Used by
    # kernels like RoPE that compute pair offsets ``e = 2*i,
    # o = 2*i + 1`` and write into a per-pair fragment. Downstream
    # ``_render_idx_as_primexpr`` materialises them.
    dst = _store_to_ref(store, buf_table)
    # Peel TVM's fp16↔fp32 cast roundtrip so reciprocal / binop / unary
    # matchers below see a clean fp16 expression tree.
    expr = _peel_cast_roundtrip(store.value, target_dtype=str(store.buffer.dtype))

    def _build_axes(srcs):
        return (
            _axes_for_ref(dst, axis, size),
            [
                _axes_for_broadcast_src(s, axis, size) if isinstance(s, Broadcast) else _axes_for_ref(s, axis, size)
                for s in srcs
            ],
        )

    # Constant fill: only 0 maps cleanly to a HW vector op. Anything
    # else falls back to RawStore (the caller wraps it).
    if isinstance(expr, (tir.IntImm, tir.FloatImm)):
        if float(expr.value) != 0.0:
            return None
        dst_axes, src_axes = _build_axes([])
        return Elementwise(
            dst=dst,
            srcs=[],
            op=UnaryOp.COPY,
            axis=axis,
            size=size,
            dst_axes=dst_axes,
            src_axes=src_axes,
        )

    # Unary: T.exp(x), T.sqrt(x).
    unary = _try_unary_call(expr)
    if unary is not None:
        if len(expr.args) != 1:
            return None
        a = _peel_cast_roundtrip(expr.args[0])
        if not isinstance(a, tir.BufferLoad):
            return None
        wrapped = _wrap_src(a, dst.indices, buf_table, dst_buf=dst.buffer)
        if wrapped is None:
            return None
        dst_axes, src_axes = _build_axes([wrapped])
        return Elementwise(
            dst=dst,
            srcs=[wrapped],
            op=unary,
            axis=axis,
            size=size,
            dst_axes=dst_axes,
            src_axes=src_axes,
        )

    # Reciprocal: 1.0 / x.
    if isinstance(expr, tir.Div):
        a = _peel_cast_roundtrip(expr.a)
        b = _peel_cast_roundtrip(expr.b)
        if isinstance(a, (tir.IntImm, tir.FloatImm)) and float(a.value) == 1.0 and isinstance(b, tir.BufferLoad):
            wrapped = _wrap_src(b, dst.indices, buf_table, dst_buf=dst.buffer)
            if wrapped is None:
                return None
            dst_axes, src_axes = _build_axes([wrapped])
            return Elementwise(
                dst=dst,
                srcs=[wrapped],
                op=UnaryOp.RECI,
                axis=axis,
                size=size,
                dst_axes=dst_axes,
                src_axes=src_axes,
            )
        return None

    # Pure copy: dst[idx] = src[idx].
    if isinstance(expr, tir.BufferLoad):
        wrapped = _wrap_src(expr, dst.indices, buf_table, dst_buf=dst.buffer)
        if wrapped is None:
            return None
        dst_axes, src_axes = _build_axes([wrapped])
        return Elementwise(
            dst=dst,
            srcs=[wrapped],
            op=UnaryOp.COPY,
            axis=axis,
            size=size,
            dst_axes=dst_axes,
            src_axes=src_axes,
        )

    # Binary: A op B (each a BufferLoad — may broadcast independently).
    binop = _try_bin_op(expr)
    if binop is not None:
        srcs: list[BufferRef | Broadcast] = []
        for arg in (expr.a, expr.b):
            arg = _peel_cast_roundtrip(arg)
            if isinstance(arg, tir.BufferLoad):
                wrapped = _wrap_src(arg, dst.indices, buf_table, dst_buf=dst.buffer)
                if wrapped is None:
                    return None
                srcs.append(wrapped)
            else:
                # Scalar literal / compound expr in binop → not foldable.
                return None
        dst_axes, src_axes = _build_axes(srcs)
        return Elementwise(
            dst=dst,
            srcs=srcs,
            op=binop,
            axis=axis,
            size=size,
            dst_axes=dst_axes,
            src_axes=src_axes,
        )

    return None


# ---------------------------------------------------------------------------
# Reduce / Gemm / Dma extern recognisers
# ---------------------------------------------------------------------------


_REDUCE_OPS_BY_NAME = {
    "max": ReduceOp.MAX,
    "sum": ReduceOp.SUM,
    "min": ReduceOp.MIN,
}


def _fold_reduce(call: tir.Call, buf_table: dict[str, BufferDef]) -> Reduce:
    """``tl.tileop.reduce(src, dst, op_name, dim, clear)``.

    Tilelang's reduce ABI varies — args[0] / args[1] are always
    src/dst (either a region call or a bare BufferLoad). The op-name
    StringImm and the dim IntImm can sit in different positions
    depending on tilelang version (we've seen op at arg[2] / dim at
    arg[3], and dim at arg[2] / op at arg[4]). Scan args[2:] to
    pick them out by type.
    """
    args = _call_args(call)
    if len(args) < 4:
        raise FoldError(f"tl.tileop.reduce: expected ≥4 args, got {len(args)}")
    src_ref = _region_to_ref(args[0], buf_table)
    dst_ref = _region_to_ref(args[1], buf_table)

    op_name: str | None = None
    axis: int | None = None
    for cand in args[2:]:
        if op_name is None and isinstance(cand, tir.StringImm):
            op_name = str(cand.value).lower()
        elif axis is None and isinstance(cand, tir.IntImm):
            # First IntImm after the regions is the dim. (clear=0/1
            # also IntImm, but we only need one.)
            axis = int(cand.value)
        if op_name is not None and axis is not None:
            break
    if op_name is None:
        raise FoldError(f"tl.tileop.reduce: cannot determine op kind from args={args!r}")
    if axis is None:
        raise FoldError(f"tl.tileop.reduce: cannot determine dim from args={args!r}")
    op = _REDUCE_OPS_BY_NAME.get(op_name)
    if op is None:
        raise FoldError(f"unknown reduce op {op_name!r}")
    # Build axes: dst is one rank lower than src; the collapsed
    # axis is tagged REDUCE on src, every other axis is BATCH.
    src_rank = len(src_ref.indices)
    normalised_axis = axis + src_rank if axis < 0 else axis
    src_axes: list[AxisInfo] = []
    for dim, ext in enumerate(src_ref.buffer.shape):
        full = int(ext)
        if dim == normalised_axis:
            src_axes.append(AxisInfo(role=AxisRole.REDUCE, extent=full))
        else:
            src_axes.append(_axes_for_ref(src_ref, None, 0)[dim])
            # ``_axes_for_ref(..., None, 0)`` returned SIMD across every dim;
            # we only want SIMD treatment when dim is REDUCE's neighbor on the
            # SIMD axis, which Reduce doesn't have. Force BATCH instead.
            src_axes[-1] = AxisInfo(role=AxisRole.BATCH, extent=full)
    dst_axes: list[AxisInfo] = []
    for dim, ext in enumerate(dst_ref.buffer.shape):
        dst_axes.append(AxisInfo(role=AxisRole.BATCH, extent=int(ext)))
    return Reduce(
        dst=dst_ref,
        src=src_ref,
        op=op,
        axis=axis,
        dst_axes=dst_axes,
        src_axes=src_axes,
    )


def _fold_dma(call: tir.Call, buf_table: dict[str, BufferDef]) -> Dma:
    args = _call_args(call)
    if len(args) < 2:
        raise FoldError(f"tl.tileop.copy: expected 2 args, got {len(args)}")
    src_ref = _region_to_ref(args[0], buf_table)
    dst_ref = _region_to_ref(args[1], buf_table)

    # Default DMA axis tagging: the innermost dim is SIMD (one HW
    # vector load/store moves a contiguous mlen-aligned run along it),
    # every other dim is BATCH (the kernel fans out one HW issue per
    # index along it). View pass prepends a CLUSTER axis when the
    # buffer is lane-aware. This matches the per-axis story Elementwise
    # uses for default ``axis=-1, size=last_dim``.
    def _default_axes(ref: BufferRef) -> list[AxisInfo]:
        shape = ref.buffer.shape
        out: list[AxisInfo] = []
        for i, d in enumerate(shape):
            role = AxisRole.SIMD if i == len(shape) - 1 else AxisRole.BATCH
            out.append(AxisInfo(role=role, extent=int(d)))
        return out

    return Dma(
        src=src_ref,
        dst=dst_ref,
        src_axes=_default_axes(src_ref),
        dst_axes=_default_axes(dst_ref),
    )


def _fold_gemm(call: tir.Call, kind: str, buf_table: dict[str, BufferDef]) -> Gemm:
    args = _call_args(call)
    if len(args) < 3:
        raise FoldError(f"tl.tileop.gemm_py: expected ≥3 args, got {len(args)}")
    a = _region_to_ref(args[0], buf_table)
    b = _region_to_ref(args[1], buf_table)
    c = _region_to_ref(args[2], buf_table)
    # tilelang's gemm extern ABI: args[3..] include transpose flags as
    # IntImm 0/1. Order is (transpose_a, transpose_b) per gemm_macros
    # docstring. Accept either position; default both False.
    ta, tb = False, False
    flags = [a for a in args[3:] if isinstance(a, tir.IntImm)]
    if len(flags) >= 1:
        ta = bool(int(flags[0].value))
    if len(flags) >= 2:
        tb = bool(int(flags[1].value))

    # Tag Gemm operand axes with their algebra roles. At fold time the
    # refs are rank-2 (pre-split lane prepend), so the labelling is
    # unambiguous from the matmul algebra:
    #
    #     c = a @ b            -> c is [M, N]
    #     a is [M, K]  (transpose_a flips to [K, M])
    #     b is [K, N]  (transpose_b flips to [N, K])
    #
    # split prepends an extra CLUSTER axis on lane-aware operands;
    # view/burn_view permute the axes alongside indices. Downstream
    # lowering (e.g. ``_lower_bare_per_head_gemm``) reads ``c_axes`` to
    # locate GEMM_M without scanning shape extents.
    def _pair(ref, roles):
        rank = len(ref.buffer.shape)
        if rank != 2:
            # Fold sees pre-split rank-2 operands. If a kernel author
            # ever hands us a rank-3 tile (unlikely but possible),
            # leave axes empty — downstream paths will need to handle
            # it explicitly.
            return []
        return [AxisInfo(role=roles[i], extent=int(ref.buffer.shape[i])) for i in range(rank)]

    a_roles = (AxisRole.GEMM_K, AxisRole.GEMM_M) if ta else (AxisRole.GEMM_M, AxisRole.GEMM_K)
    b_roles = (AxisRole.GEMM_N, AxisRole.GEMM_K) if tb else (AxisRole.GEMM_K, AxisRole.GEMM_N)
    c_roles = (AxisRole.GEMM_M, AxisRole.GEMM_N)
    return Gemm(
        a=a,
        b=b,
        c=c,
        transpose_a=ta,
        transpose_b=tb,
        kind=kind,
        a_axes=_pair(a, a_roles),
        b_axes=_pair(b, b_roles),
        c_axes=_pair(c, c_roles),
    )


# ---------------------------------------------------------------------------
# Walker — produces a flat list of mid_ir Stmt
# ---------------------------------------------------------------------------


def _tir_for_kind_name(stmt: tir.For) -> str:
    """Return the lowercase tilelang ForKind name (``"serial"`` /
    ``"parallel"`` / ``"unrolled"`` / ...). Used to pick between For
    and ParallelAxis(CLUSTER) when a T.Parallel doesn't fold into an
    Elementwise."""
    try:
        return tir.ForKind(int(stmt.kind)).name.lower()
    except Exception:
        return "serial"


def _mid_for_kind(name: str) -> str:
    """Map a tilelang for-kind name to the mid-IR For.kind string.
    For.kind is one of ``"serial"`` or ``"unroll"``; other tilelang
    kinds shouldn't reach For (T.Parallel becomes ParallelAxis(CLUSTER))."""
    if name == "unrolled" or name == "unroll":
        return "unroll"
    return "serial"


def _outer_loop_matches_buffer_axis(dst: BufferRef, loop_var: tir.Var, extent: int) -> bool:
    """True when ``dst.indices`` references ``loop_var`` (by identity)
    on a non-last axis whose buffer extent equals ``extent``. Used to
    decide whether an outer ``for row`` is redundant on top of an
    already-whole-buffer Elementwise."""
    target = _vref(loop_var)
    shape = dst.buffer.shape
    if len(dst.indices) != len(shape):
        return False
    for axis, idx in enumerate(dst.indices):
        if axis == len(dst.indices) - 1:
            continue  # inner axis = the one Elementwise(axis=-1) already covers
        if isinstance(idx, VarRef) and idx == target and int(shape[axis]) == extent:
            return True
    return False


def _index_expr_uses_varref(idx, target: VarRef) -> bool:
    """Recursively look for ``target`` (by identity) inside an IndexExpr.

    Used to decide whether an outer ``for row`` can be absorbed into an
    already-folded inner Elementwise: if any Broadcast src still
    references the outer var, absorbing the for would leave the var
    unbound.
    """
    if isinstance(idx, VarRef):
        return idx == target
    if isinstance(idx, dict):
        return any(_index_expr_uses_varref(a, target) for a in idx.get("args", []))
    return False


def _elementwise_refs_var(ew, target: VarRef) -> bool:
    """True if any ``Broadcast`` src of ``ew`` references ``target``
    (by identity) in its indices.

    Only the broadcast case is problematic: ``_wrap_src`` keeps the
    Broadcast's indices as a prefix of dst's, so the outer var
    is preserved literally in the source. Absorbing the outer for then
    leaves the var referenced but unbound.

    Same-rank BufferRef srcs already have indices that match dst's one
    for one, so absorbing the outer for is symmetric — both dst and
    src lose their reference to the var simultaneously and the
    whole-buffer Elementwise stands on its own.
    """
    for src in ew.srcs:
        if not isinstance(src, Broadcast):
            continue
        for idx in src.src.indices:
            if _index_expr_uses_varref(idx, target):
                return True
    return False


def _is_serial_for(stmt: tir.For) -> bool:
    return stmt.kind == tir.ForKind.SERIAL


def _walk_stmt(stmt, buf_table: dict[str, BufferDef], current_kind: str | None) -> list:
    """Walk one TIR Stmt, return a list of mid_ir Stmt items.

    A single TIR construct may unfold into 0, 1, or more mid_ir items
    (e.g. a SeqStmt becomes its concatenated children; an AttrStmt for
    KIND becomes nothing of its own — its body inherits ``current_kind``).
    """
    if stmt is None:
        return []
    if isinstance(stmt, tir.SeqStmt):
        out = []
        for c in stmt.seq:
            out.extend(_walk_stmt(c, buf_table, current_kind))
        return out
    if isinstance(stmt, tir.BlockRealize):
        for b in stmt.block.alloc_buffers:
            if b.name not in buf_table:
                buf_table[b.name] = _buffer_def(b, default_scope="shared")
        return _walk_stmt(stmt.block.body, buf_table, current_kind)
    if isinstance(stmt, tir.AttrStmt):
        if stmt.attr_key == _KIND_KEY:
            v = stmt.value
            kind = v.value if isinstance(v, tir.StringImm) else str(v)
            return _walk_stmt(stmt.body, buf_table, current_kind=kind)
        if stmt.attr_key == "thread_extent" and isinstance(stmt.node, tir.IterVar):
            iv = stmt.node
            inner = _walk_stmt(stmt.body, buf_table, current_kind)
            ext_val = stmt.value
            if not isinstance(ext_val, tir.IntImm):
                raise FoldError(f"thread_extent {iv.var.name!r} non-static: {ext_val!r}")
            # blockIdx grid binding from T.Kernel → ParallelAxis(BLOCK_IDX).
            # Mid-IR keeps multi-thread semantics: this is NOT a serial
            # for, it's an SPMD parallel axis. Pass_8_to_plena flattens
            # to a serial outer for at HLIR generation time.
            return [
                ParallelAxis(
                    axis_name=iv.var.name,
                    extent=int(ext_val.value),
                    body=inner,
                    kind=ParallelKind.BLOCK_IDX,
                    thread_tag=str(iv.thread_tag) if iv.thread_tag else None,
                    axis_var=_vref(iv.var),
                )
            ]
        # Unknown attr — pass through.
        return _walk_stmt(stmt.body, buf_table, current_kind)
    if isinstance(stmt, tir.For):
        # Try fold first.
        if stmt.kind == tir.ForKind.PARALLEL:
            ew = _try_fold_parallel(stmt, buf_table)
            if ew is not None:
                return [ew]
        # Serial outer wrapping a single fold-able store (1D fp scalar
        # update like ``L_INV[row] = 1 / L_NEW[row]``): fold the inner
        # body. The serial for is *absorbed* by the Elementwise — its
        # extent becomes ``size`` so downstream lowering knows the op
        # covers that many elements along ``axis=-1`` (otherwise the
        # default ``size=1`` mis-types it as a single scalar and FPRAM
        # unrolling won't kick in).
        if _is_serial_for(stmt) and isinstance(stmt.body, tir.BufferStore):
            extent = int(stmt.extent.value) if isinstance(stmt.extent, tir.IntImm) else 1
            ew = _try_fold_store(
                stmt.body,
                parallel_var=stmt.loop_var,
                buf_table=buf_table,
                axis=-1,
                size=extent,
            )
            if ew is not None:
                return [ew]
        # Serial / unroll outer wrapping an inner T.Parallel that
        # already folds to a whole-buffer Elementwise: the inner fold
        # produced an op covering every element along ``axis=-1``; the
        # outer ``for row`` re-iterates over a dim the op already
        # covers. Absorb the outer loop if its extent matches the dim
        # the inner Elementwise iterates over (typically the row axis
        # of dst). Without this we'd emit the same whole-buffer op
        # ``rows`` times.
        if _is_serial_for(stmt) or _tir_for_kind_name(stmt) == "unrolled":
            inner_body = stmt.body
            if isinstance(inner_body, tir.For) and inner_body.kind == tir.ForKind.PARALLEL:
                inner_ew = _try_fold_parallel(inner_body, buf_table)
                if (
                    inner_ew is not None
                    and isinstance(stmt.extent, tir.IntImm)
                    and _outer_loop_matches_buffer_axis(
                        inner_ew.dst,
                        stmt.loop_var,
                        int(stmt.extent.value),
                    )
                    # Don't absorb if any Broadcast src still uses
                    # the outer loop var — e.g. ``dst[row,col] =
                    # a[row,col] * b[row]`` folds the parallel ``col``
                    # away but the ``b[row]`` Broadcast still needs
                    # ``row`` bound by an enclosing for. Absorbing it
                    # would leave ``row`` referenced but unbound,
                    # crashing ExprMaterializer later. Pass through
                    # as a regular For instead so the outer loop
                    # keeps its scope.
                    and not _elementwise_refs_var(
                        inner_ew,
                        _vref(stmt.loop_var),
                    )
                ):
                    return [inner_ew]
        # Pass through as a regular For. Body is recursively walked;
        # any nested BufferStore that doesn't fold becomes a RawStore.
        if not isinstance(stmt.extent, tir.IntImm):
            raise FoldError(f"non-static loop extent on {stmt.loop_var.name!r}: {stmt.extent!r}")
        body = _walk_stmt(stmt.body, buf_table, current_kind)
        kind_name = _tir_for_kind_name(stmt)
        if kind_name == "parallel":
            # T.Parallel that didn't fold into an Elementwise (because
            # the inner store had a non-elementwise pattern). Surface
            # as a LOGICAL_GRID parallel axis — semantically still N
            # concurrent program instances. Pass_3 may later split it
            # into (LOGICAL_GRID number, CLUSTER phase) just like a
            # blockIdx-bound axis.
            return [
                ParallelAxis(
                    axis_name=stmt.loop_var.name,
                    extent=int(stmt.extent.value),
                    body=body,
                    kind=ParallelKind.LOGICAL_GRID,
                    thread_tag=None,
                    axis_var=_vref(stmt.loop_var),
                )
            ]
        return [
            For(
                loop_var=stmt.loop_var.name,
                extent=int(stmt.extent.value),
                body=body,
                kind=_mid_for_kind(kind_name),
                loop_var_var=_vref(stmt.loop_var),
            )
        ]
    if isinstance(stmt, tir.LetStmt):
        # Should be eliminated by inline_let_stmts; if it lingers,
        # walk through and warn implicitly by losing the binding.
        return _walk_stmt(stmt.body, buf_table, current_kind)
    if isinstance(stmt, tir.IfThenElse):
        raise FoldError("tir.IfThenElse not supported by mid_ir")
    if isinstance(stmt, tir.Allocate):
        # Create a BufferDef from the Allocate (raw form has only
        # buffer_var). Best-effort: name from var, shape from extents.
        # The body's BufferLoad/Store will see the real name if there's
        # a corresponding decl_buffer; if not, this synth def stands in.
        name = stmt.buffer_var.name
        if name not in buf_table:
            try:
                buf_table[name] = BufferDef(
                    name=name,
                    shape=[int(e.value) for e in stmt.extents],
                    dtype=str(stmt.dtype),
                    scope="shared",
                )
            except Exception as e:
                raise FoldError(f"could not build BufferDef from raw Allocate {name!r}: {e}")
        return _walk_stmt(stmt.body, buf_table, current_kind)
    if isinstance(stmt, tir.Evaluate):
        val = stmt.value
        if not isinstance(val, tir.Call):
            return []
        kind = _call_kind(val)
        if kind == _TILEOP_COPY:
            return [_fold_dma(val, buf_table)]
        if kind == _TILEOP_GEMM:
            return [_fold_gemm(val, kind=current_kind or "overwrite", buf_table=buf_table)]
        if kind == _TILEOP_REDUCE:
            return [_fold_reduce(val, buf_table)]
        # Unknown extern: drop with a deliberate marker. Production
        # could accumulate these into a side list for diagnostics.
        return []
    if isinstance(stmt, tir.BufferStore):
        ew = _try_fold_store(stmt, parallel_var=None, buf_table=buf_table)
        if ew is not None:
            return [ew]
        raise FoldError(
            f"unrecognised BufferStore — every store must lower to a "
            f"single elementwise / reduce / broadcast pattern. "
            f"dst={stmt.buffer.name}{list(stmt.indices)} := {stmt.value!r}"
        )
    raise FoldError(f"unhandled stmt type {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: tir.PrimFunc, name: str = "kernel") -> MidFunc:
    """Fold a raw tir.PrimFunc into mid_ir."""
    global _active_registry
    _active_registry = _VarRegistry()
    try:
        return _run_locked(func, name)
    finally:
        _active_registry = None


def _run_locked(func: tir.PrimFunc, name: str) -> MidFunc:
    buf_table: dict[str, BufferDef] = {}

    # Seed param buffers (always global by convention).
    params: list[BufferDef] = []
    for var in func.params:
        buf = func.buffer_map.get(var)
        if buf is None:
            continue
        bd = _buffer_def(buf, default_scope="global")
        # Force "global" — tilelang doesn't tag params with a scope.
        bd = BufferDef(name=bd.name, shape=bd.shape, dtype=bd.dtype, scope="global")
        buf_table[bd.name] = bd
        params.append(bd)

    body = _walk_stmt(func.body, buf_table, current_kind=None)

    # Fold output invariant: no ``str`` may appear in any BufferRef
    # indices. Bare-string indices were the cheat the VarRef rewrite
    # exists to remove. If something slipped through, fail loudly here
    # rather than letting fuse/to_plena silently mishandle it.
    _assert_no_str_in_indices(body)

    # Allocs are everything in buf_table that isn't a param.
    param_names = {p.name for p in params}
    allocs = [b for n, b in buf_table.items() if n not in param_names]

    # Lane axes from func attr (T.func_attr({"plena.lane_axis": "by"}) or list).
    lane_axes: list[str] = []
    if func.attrs is not None and _LANE_AXIS_FUNC_ATTR in func.attrs:
        raw = func.attrs[_LANE_AXIS_FUNC_ATTR]
        if isinstance(raw, tir.StringImm):
            lane_axes = [str(raw.value)]
        elif isinstance(raw, str):
            lane_axes = [raw]
        elif hasattr(raw, "__iter__"):
            lane_axes = [str(s.value) if isinstance(s, tir.StringImm) else str(s) for s in raw]

    # Carry select prim_func attrs forward so downstream passes (e.g.
    # to_plena reading ``plena.layout``) can find them. We unwrap TVM
    # ObjectRef strings to plain Python so dict access works uniformly.
    attrs_out: dict[str, object] = {}
    if func.attrs is not None:
        for k in ("plena.layout",):
            if k in func.attrs:
                v = func.attrs[k]
                if isinstance(v, tir.StringImm):
                    attrs_out[k] = str(v.value)
                else:
                    attrs_out[k] = str(v)
        # ``plena.hoisted_constants`` is a {buffer_name: value} map
        # stamped by the ``hoist_float_constants`` pre-pass. Unwrap to
        # a plain ``Dict[str, float]`` so to_plena can iterate it
        # without TVM-side type acrobatics.
        if "plena.hoisted_constants" in func.attrs:
            raw = func.attrs["plena.hoisted_constants"]
            attrs_out["plena.hoisted_constants"] = {
                str(name): float(val.value if hasattr(val, "value") else val) for name, val in raw.items()
            }

    return MidFunc(
        name=name,
        params=params,
        allocs=allocs,
        body=body,
        lane_axes=lane_axes,
        cluster_counts=[],  # filled by pass_3
        attrs=attrs_out,
    )


__all__ = ["FoldError", "run"]
