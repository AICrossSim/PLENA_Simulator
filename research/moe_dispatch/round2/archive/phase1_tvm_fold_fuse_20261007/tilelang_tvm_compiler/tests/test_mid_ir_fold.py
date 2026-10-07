"""Unit tests for mid_ir.passes.fold (raw TIR → mid_ir).

Coverage:
  * dma  (tl.tileop.copy)
  * gemm (tl.tileop.gemm_py with + without KIND="btmm")
  * reduce (tl.tileop.reduce)
  * elementwise binary (T.Parallel + add/sub/mul/max)
  * elementwise unary (T.exp, 1/x, copy)
  * **broadcast** — src.indices is a prefix of dst.indices
  * zero fill (constant 0.0 / 0)
  * blockIdx grid wrappers preserved as For(thread_tag=...)

Run:
    python -m tilelang_tvm_compiler.tests.test_mid_ir_fold
"""

from __future__ import annotations

import sys

import tvm
from tvm import tir

from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.fold import (
    run as fold_run,
)


def _ii(n: int, dtype: str = "int32") -> tir.IntImm:
    return tir.IntImm(dtype, n)


def _extern(name: str, *args):
    return tir.call_extern("handle", name, *args)


def _region(buf: tir.Buffer, starts, extents):
    return _extern(
        "tl.tileop.region",
        tir.BufferLoad(buf, starts),
        _ii(0),
        *extents,
    )


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


def _wrap(body, params=(), buffer_map=None) -> tir.PrimFunc:
    return tir.PrimFunc(
        params=list(params),
        body=body,
        ret_type=None,
        buffer_map=buffer_map or {},
    )


# ---------------------------------------------------------------------------
# 1. dma / gemm / reduce
# ---------------------------------------------------------------------------


def test_fold_dma() -> int:
    print("test_fold_dma")
    f16 = "float16"
    Q_hbm = tir.decl_buffer([1, 64, 4, 16], dtype=f16, name="Q_hbm", scope="global")
    Q_sh = tir.decl_buffer([64, 16], dtype=f16, name="Q_sh", scope="shared.dyn")
    body = tir.Evaluate(
        _extern(
            "tl.tileop.copy",
            _region(Q_hbm, [_ii(0), _ii(0), tir.Var("by", "int32"), _ii(0)], [_ii(1), _ii(64), _ii(1), _ii(16)]),
            _region(Q_sh, [_ii(0), _ii(0)], [_ii(64), _ii(16)]),
        )
    )
    func = _wrap(body, params=[Q_hbm.data], buffer_map={Q_hbm.data: Q_hbm})
    mid = fold_run(func, name="t_dma")
    failures = 0
    failures += _check("body length", len(mid.body), 1)
    if mid.body and isinstance(mid.body[0], ir.Dma):
        dma = mid.body[0]
        failures += _check("src buffer", dma.src.buffer.name, "Q_hbm")
        failures += _check("dst buffer", dma.dst.buffer.name, "Q_sh")
        # dst is whole-buffer (extents == buffer shape, starts 0): both Slice
        failures += _check(
            "dst indices all-Slice",
            all(isinstance(i, ir.Slice) for i in dma.dst.indices),
            True,
        )
        # src has extents [1,64,1,16], buffer shape [1,64,4,16] → axes
        # 0 and 1 and 3 cover full dim; axis 2 is sliced (extent 1, start `by`).
        failures += _check("src indices[2]", dma.src.indices[2], "by")
    else:
        print(f"  [FAIL] body[0] is not Dma: {mid.body}")
        failures += 1
    return failures


def test_fold_gemm_btmm() -> int:
    print("test_fold_gemm_btmm")
    f16 = "float16"
    Q = tir.decl_buffer([64, 16], dtype=f16, name="Q", scope="shared.dyn")
    K = tir.decl_buffer([64, 16], dtype=f16, name="K", scope="shared.dyn")
    S = tir.decl_buffer([64, 64], dtype=f16, name="S", scope="local.fragment")
    body = tir.AttrStmt(
        _ii(0),
        "plena.gemm_kind",
        tir.StringImm("btmm"),
        tir.Evaluate(
            _extern(
                "tl.tileop.gemm_py",
                _region(Q, [_ii(0)] * 2, list(Q.shape)),
                _region(K, [_ii(0)] * 2, list(K.shape)),
                _region(S, [_ii(0)] * 2, list(S.shape)),
                _ii(0),  # transpose_a
                _ii(1),  # transpose_b
            )
        ),
    )
    func = _wrap(body)
    mid = fold_run(func, name="t_gemm")
    failures = 0
    failures += _check("body length", len(mid.body), 1)
    if mid.body and isinstance(mid.body[0], ir.Gemm):
        gemm = mid.body[0]
        failures += _check("kind", gemm.kind, "btmm")
        failures += _check("transpose_b", gemm.transpose_b, True)
        failures += _check("transpose_a", gemm.transpose_a, False)
    else:
        print(f"  [FAIL] body[0] is not Gemm: {mid.body}")
        failures += 1
    return failures


def test_fold_gemm_per_head() -> int:
    print("test_fold_gemm_per_head — no KIND attr → kind='overwrite'")
    f16 = "float16"
    A = tir.decl_buffer([64, 64], dtype=f16, name="A", scope="local.fragment")
    B = tir.decl_buffer([64, 16], dtype=f16, name="B", scope="shared.dyn")
    C = tir.decl_buffer([64, 16], dtype=f16, name="C", scope="local.fragment")
    body = tir.Evaluate(
        _extern(
            "tl.tileop.gemm_py",
            _region(A, [_ii(0)] * 2, list(A.shape)),
            _region(B, [_ii(0)] * 2, list(B.shape)),
            _region(C, [_ii(0)] * 2, list(C.shape)),
        )
    )
    func = _wrap(body)
    mid = fold_run(func)
    failures = 0
    if mid.body and isinstance(mid.body[0], ir.Gemm):
        failures += _check("kind", mid.body[0].kind, "overwrite")
    else:
        failures += 1
    return failures


def test_fold_reduce() -> int:
    print("test_fold_reduce")
    f16 = "float16"
    src = tir.decl_buffer([64, 64], dtype=f16, name="src", scope="local.fragment")
    dst = tir.decl_buffer([64], dtype=f16, name="dst", scope="local.fragment")
    body = tir.Evaluate(
        _extern(
            "tl.tileop.reduce",
            _region(src, [_ii(0), _ii(0)], [_ii(64), _ii(64)]),
            _region(dst, [_ii(0)], [_ii(64)]),
            _ii(1),  # dim
            _ii(0),  # clear
            tir.StringImm("max"),  # op
        )
    )
    func = _wrap(body)
    mid = fold_run(func)
    failures = 0
    if mid.body and isinstance(mid.body[0], ir.Reduce):
        red = mid.body[0]
        failures += _check("axis", red.axis, 1)
        failures += _check("op", red.op, ir.ReduceOp.MAX)
        failures += _check("src", red.src.buffer.name, "src")
        failures += _check("dst", red.dst.buffer.name, "dst")
    else:
        failures += 1
    return failures


# ---------------------------------------------------------------------------
# 2. elementwise patterns (T.Parallel + binary / unary / zero)
# ---------------------------------------------------------------------------


def test_fold_parallel_add() -> int:
    print("test_fold_parallel_add")
    f16 = "float16"
    A = tir.decl_buffer([64, 16], dtype=f16, name="A", scope="shared.dyn")
    B = tir.decl_buffer([64, 16], dtype=f16, name="B", scope="shared.dyn")
    C = tir.decl_buffer([64, 16], dtype=f16, name="C", scope="shared.dyn")
    row = tir.Var("row", "int32")
    col = tir.Var("col", "int32")
    inner = tir.For(
        col,
        _ii(0),
        _ii(16),
        tir.ForKind.PARALLEL,
        tir.BufferStore(
            C,
            tir.BufferLoad(A, [row, col]) + tir.BufferLoad(B, [row, col]),
            [row, col],
        ),
    )
    outer = tir.For(row, _ii(0), _ii(64), tir.ForKind.SERIAL, inner)
    func = _wrap(outer)
    mid = fold_run(func)
    failures = 0
    # Walk: outer For(row) → body has the fused Elementwise.
    if (
        mid.body
        and isinstance(mid.body[0], ir.For)
        and mid.body[0].body
        and isinstance(mid.body[0].body[0], ir.Elementwise)
    ):
        ew = mid.body[0].body[0]
        failures += _check("op", ew.op, ir.BinOp.ADD)
        failures += _check("# srcs", len(ew.srcs), 2)
        failures += _check(
            "all srcs are BufferRef (no broadcast)",
            all(isinstance(s, ir.BufferRef) for s in ew.srcs),
            True,
        )
    else:
        print(f"  [FAIL] expected For(row) → Elementwise, got {mid.body}")
        failures += 1
    return failures


def test_fold_parallel_zero() -> int:
    print("test_fold_parallel_zero")
    f16 = "float16"
    Z = tir.decl_buffer([64, 16], dtype=f16, name="Z", scope="shared.dyn")
    row = tir.Var("row", "int32")
    col = tir.Var("col", "int32")
    inner = tir.For(
        col,
        _ii(0),
        _ii(16),
        tir.ForKind.PARALLEL,
        tir.BufferStore(Z, tir.FloatImm(f16, 0.0), [row, col]),
    )
    outer = tir.For(row, _ii(0), _ii(64), tir.ForKind.SERIAL, inner)
    func = _wrap(outer)
    mid = fold_run(func)
    failures = 0
    if mid.body and isinstance(mid.body[0], ir.For) and isinstance(mid.body[0].body[0], ir.Elementwise):
        ew = mid.body[0].body[0]
        failures += _check("op (zero is COPY w/ srcs=[])", ew.op, ir.UnaryOp.COPY)
        failures += _check("# srcs (zero sentinel)", len(ew.srcs), 0)
    else:
        failures += 1
    return failures


def test_fold_parallel_exp() -> int:
    print("test_fold_parallel_exp")
    f16 = "float16"
    A = tir.decl_buffer([64, 64], dtype=f16, name="A", scope="local.fragment")
    row = tir.Var("row", "int32")
    col = tir.Var("col", "int32")
    inner = tir.For(
        col,
        _ii(0),
        _ii(64),
        tir.ForKind.PARALLEL,
        tir.BufferStore(
            A,
            tir.exp(tir.BufferLoad(A, [row, col])),
            [row, col],
        ),
    )
    outer = tir.For(row, _ii(0), _ii(64), tir.ForKind.SERIAL, inner)
    func = _wrap(outer)
    mid = fold_run(func)
    failures = 0
    if mid.body and isinstance(mid.body[0], ir.For) and isinstance(mid.body[0].body[0], ir.Elementwise):
        ew = mid.body[0].body[0]
        failures += _check("op", ew.op, ir.UnaryOp.EXP)
        failures += _check("# srcs", len(ew.srcs), 1)
    else:
        failures += 1
    return failures


# ---------------------------------------------------------------------------
# 3. **broadcast** — the case I was missing
# ---------------------------------------------------------------------------


def test_fold_broadcast_sub_fp() -> int:
    """``S[r, c] = S[r, c] - M_CURR[r]`` — M_CURR is rank 1, S is rank 2.
    Broadcast over the col axis."""
    print("test_fold_broadcast_sub_fp — S[r,c] = S[r,c] - M_CURR[r]")
    f16 = "float16"
    S = tir.decl_buffer([64, 64], dtype=f16, name="S", scope="local.fragment")
    M_CURR = tir.decl_buffer([64], dtype=f16, name="M_CURR", scope="local.fragment")
    row = tir.Var("row", "int32")
    col = tir.Var("col", "int32")
    inner = tir.For(
        col,
        _ii(0),
        _ii(64),
        tir.ForKind.PARALLEL,
        tir.BufferStore(
            S,
            tir.BufferLoad(S, [row, col]) - tir.BufferLoad(M_CURR, [row]),
            [row, col],
        ),
    )
    outer = tir.For(row, _ii(0), _ii(64), tir.ForKind.SERIAL, inner)
    func = _wrap(outer)
    mid = fold_run(func)
    failures = 0
    if not (mid.body and isinstance(mid.body[0], ir.For) and isinstance(mid.body[0].body[0], ir.Elementwise)):
        print(f"  [FAIL] expected For(row) → Elementwise, got {mid.body}")
        return 1
    ew = mid.body[0].body[0]
    failures += _check("op", ew.op, ir.BinOp.SUB)
    failures += _check("# srcs", len(ew.srcs), 2)
    # First src is S (same rank as dst → BufferRef).
    failures += _check(
        "src[0] is BufferRef",
        isinstance(ew.srcs[0], ir.BufferRef),
        True,
    )
    # Second src is M_CURR (rank 1, dst is rank 2 → Broadcast).
    failures += _check(
        "src[1] is Broadcast",
        isinstance(ew.srcs[1], ir.Broadcast),
        True,
    )
    if isinstance(ew.srcs[1], ir.Broadcast):
        failures += _check(
            "broadcast dims",
            ew.srcs[1].broadcast_dims,
            [1],
        )
        failures += _check(
            "broadcast src buffer",
            ew.srcs[1].src.buffer.name,
            "M_CURR",
        )
    return failures


def test_fold_broadcast_mul_fp() -> int:
    """``O[r, c] = O[r, c] * L_INV[r]`` — same broadcast pattern."""
    print("test_fold_broadcast_mul_fp — O[r,c] = O[r,c] * L_INV[r]")
    f16 = "float16"
    O_loc = tir.decl_buffer([64, 16], dtype=f16, name="O_loc", scope="local.fragment")
    L_INV = tir.decl_buffer([64], dtype=f16, name="L_INV", scope="local.fragment")
    row = tir.Var("row", "int32")
    col = tir.Var("col", "int32")
    inner = tir.For(
        col,
        _ii(0),
        _ii(16),
        tir.ForKind.PARALLEL,
        tir.BufferStore(
            O_loc,
            tir.BufferLoad(O_loc, [row, col]) * tir.BufferLoad(L_INV, [row]),
            [row, col],
        ),
    )
    outer = tir.For(row, _ii(0), _ii(64), tir.ForKind.SERIAL, inner)
    func = _wrap(outer)
    mid = fold_run(func)
    failures = 0
    if not (mid.body and isinstance(mid.body[0], ir.For) and isinstance(mid.body[0].body[0], ir.Elementwise)):
        return 1
    ew = mid.body[0].body[0]
    failures += _check("op", ew.op, ir.BinOp.MUL)
    failures += _check("src[1] is Broadcast", isinstance(ew.srcs[1], ir.Broadcast), True)
    if isinstance(ew.srcs[1], ir.Broadcast):
        failures += _check("broadcast dims", ew.srcs[1].broadcast_dims, [1])
    return failures


def test_fold_broadcast_left_operand() -> int:
    """Same shape but broadcast on LHS operand: ``O[r,c] = SCALE[r] * O[r,c]``."""
    print("test_fold_broadcast_left_operand")
    f16 = "float16"
    O_loc = tir.decl_buffer([64, 16], dtype=f16, name="O_loc", scope="local.fragment")
    SCALE = tir.decl_buffer([64], dtype=f16, name="SCALE", scope="local.fragment")
    row = tir.Var("row", "int32")
    col = tir.Var("col", "int32")
    inner = tir.For(
        col,
        _ii(0),
        _ii(16),
        tir.ForKind.PARALLEL,
        tir.BufferStore(
            O_loc,
            tir.BufferLoad(SCALE, [row]) * tir.BufferLoad(O_loc, [row, col]),
            [row, col],
        ),
    )
    outer = tir.For(row, _ii(0), _ii(64), tir.ForKind.SERIAL, inner)
    func = _wrap(outer)
    mid = fold_run(func)
    if not (mid.body and isinstance(mid.body[0], ir.For) and isinstance(mid.body[0].body[0], ir.Elementwise)):
        return 1
    ew = mid.body[0].body[0]
    failures = 0
    failures += _check("src[0] is Broadcast", isinstance(ew.srcs[0], ir.Broadcast), True)
    failures += _check("src[1] is BufferRef", isinstance(ew.srcs[1], ir.BufferRef), True)
    return failures


def test_fold_conv2d_zero_pad_init() -> int:
    """conv2d's ``for k: in_FP_padded[MLEN + k] = 0`` — the dst index is
    a compound expression, not a bare loop var. fold can't express this
    as Elementwise (it's not a whole-axis cover); the For + RawStore
    must survive."""
    print("test_fold_conv2d_zero_pad_init — for k: padded[MLEN + k] = 0")
    f16 = "float16"
    padded = tir.decl_buffer([67], dtype=f16, name="in_FP_padded", scope="local.fragment")
    k = tir.Var("k", "int32")
    body = tir.For(
        k,
        _ii(0),
        _ii(3),
        tir.ForKind.SERIAL,
        tir.BufferStore(padded, tir.FloatImm(f16, 0.0), [tir.IntImm("int32", 64) + k]),
    )
    func = _wrap(body)
    mid = fold_run(func)
    failures = 0
    if not (mid.body and isinstance(mid.body[0], ir.For)):
        print(f"  [FAIL] expected For, got {mid.body}")
        return 1
    f = mid.body[0]
    failures += _check("loop var", f.loop_var, "k")
    failures += _check("extent", f.extent, 3)
    failures += _check(
        "body is one RawStore",
        len(f.body) == 1 and isinstance(f.body[0], ir.RawStore),
        True,
    )
    return failures


def test_fold_conv2d_serial_copy() -> int:
    """conv2d's ``for i in T.serial(MLEN): in_FP_padded[i] = in_FP_aux[i]``
    — both indices are the bare loop var, full coverage. Should fold
    into an Elementwise(COPY)."""
    print("test_fold_conv2d_serial_copy — for i: padded[i] = aux[i]")
    f16 = "float16"
    padded = tir.decl_buffer([67], dtype=f16, name="in_FP_padded", scope="local.fragment")
    aux = tir.decl_buffer([64], dtype=f16, name="in_FP_aux", scope="local.fragment")
    i = tir.Var("i", "int32")
    body = tir.For(
        i,
        _ii(0),
        _ii(64),
        tir.ForKind.SERIAL,
        tir.BufferStore(padded, tir.BufferLoad(aux, [i]), [i]),
    )
    func = _wrap(body)
    mid = fold_run(func)
    failures = 0
    if not (mid.body and isinstance(mid.body[0], ir.Elementwise)):
        print(f"  [FAIL] expected Elementwise, got {mid.body}")
        return 1
    ew = mid.body[0]
    failures += _check("op", ew.op, ir.UnaryOp.COPY)
    failures += _check("# srcs", len(ew.srcs), 1)
    return failures


def test_fold_conv2d_shifted_copy() -> int:
    """conv2d's ``for m in T.serial(MLEN): shift_FP[m] = in_FP_padded[m + kw_idx]``
    — the src index has a compound expression that doesn't match dst.
    fold can't express this as Elementwise; For + RawStore preserved."""
    print("test_fold_conv2d_shifted_copy — for m: shift[m] = padded[m + kw]")
    f16 = "float16"
    shift = tir.decl_buffer([64], dtype=f16, name="shift_FP", scope="local.fragment")
    padded = tir.decl_buffer([67], dtype=f16, name="in_FP_padded", scope="local.fragment")
    m = tir.Var("m", "int32")
    kw = tir.Var("kw_idx", "int32")
    body = tir.For(
        m,
        _ii(0),
        _ii(64),
        tir.ForKind.SERIAL,
        tir.BufferStore(shift, tir.BufferLoad(padded, [m + kw]), [m]),
    )
    func = _wrap(body)
    mid = fold_run(func)
    failures = 0
    if not (mid.body and isinstance(mid.body[0], ir.For)):
        print(f"  [FAIL] expected For, got {mid.body}")
        return 1
    f = mid.body[0]
    failures += _check("loop var", f.loop_var, "m")
    failures += _check(
        "body is one RawStore",
        len(f.body) == 1 and isinstance(f.body[0], ir.RawStore),
        True,
    )
    return failures


def test_fold_unfoldable_falls_back_to_for() -> int:
    """src ``B[r, k]`` doesn't match dst ``[r, c]`` (different var on
    last axis). Fold can't recognize this as elementwise:
      * outer T.serial(row) → For(serial)
      * inner T.Parallel(col) → ParallelAxis(CLUSTER) (T.Parallel
        always becomes a CLUSTER parallel axis when it can't be
        folded into an Elementwise)
      * the BufferStore lands as a RawStore inside the parallel axis.

    Fold stays conservative: anything it doesn't recognize survives
    structurally without losing the parallelism hint."""
    print("test_fold_unfoldable_falls_back_to_for")
    f16 = "float16"
    A = tir.decl_buffer([64, 16], dtype=f16, name="A", scope="local.fragment")
    B = tir.decl_buffer([64, 16], dtype=f16, name="B", scope="local.fragment")
    C = tir.decl_buffer([64, 16], dtype=f16, name="C", scope="local.fragment")
    row = tir.Var("row", "int32")
    col = tir.Var("col", "int32")
    k = tir.Var("k", "int32")
    inner = tir.For(
        col,
        _ii(0),
        _ii(16),
        tir.ForKind.PARALLEL,
        tir.BufferStore(
            C,
            tir.BufferLoad(A, [row, col]) + tir.BufferLoad(B, [row, k]),
            [row, col],
        ),
    )
    outer = tir.For(row, _ii(0), _ii(64), tir.ForKind.SERIAL, inner)
    func = _wrap(outer)
    mid = fold_run(func)
    failures = 0
    if not (mid.body and isinstance(mid.body[0], ir.For)):
        print(f"  [FAIL] expected outer For, got {mid.body}")
        return 1
    outer_for = mid.body[0]
    failures += _check("outer For loop_var", outer_for.loop_var, "row")
    failures += _check("outer For kind", outer_for.kind, "serial")
    if not (outer_for.body and isinstance(outer_for.body[0], ir.ParallelAxis)):
        print(f"  [FAIL] expected inner ParallelAxis, got {outer_for.body}")
        return failures + 1
    inner_par = outer_for.body[0]
    failures += _check("inner ParallelAxis axis_name", inner_par.axis_name, "col")
    # Unfolded T.Parallel becomes LOGICAL_GRID — kernel-body parallel axis,
    # NOT a CLUSTER (CLUSTER is created by pass_3 split, not fold).
    failures += _check("inner ParallelAxis kind", inner_par.kind, ir.ParallelKind.LOGICAL_GRID)
    failures += _check(
        "inner body is RawStore",
        len(inner_par.body) == 1 and isinstance(inner_par.body[0], ir.RawStore),
        True,
    )
    return failures


# ---------------------------------------------------------------------------
# 4. blockIdx wrappers preserved
# ---------------------------------------------------------------------------


def test_fold_preserves_blockidx() -> int:
    """blockIdx grid bindings become ParallelAxis(BLOCK_IDX), not For —
    mid_ir keeps multi-thread semantics until pass_8."""
    print("test_fold_preserves_blockidx")
    f16 = "float16"
    Z = tir.decl_buffer([64, 16], dtype=f16, name="Z", scope="shared.dyn")
    by = tir.Var("by", "int32")
    by_iv = tir.IterVar(
        dom=tvm.ir.Range.from_min_extent(_ii(0), _ii(4)),
        var=by,
        iter_type=tir.IterVar.ThreadIndex,
        thread_tag="blockIdx.y",
    )
    col = tir.Var("col", "int32")
    body = tir.AttrStmt(
        by_iv,
        "thread_extent",
        _ii(4),
        tir.For(
            col,
            _ii(0),
            _ii(16),
            tir.ForKind.PARALLEL,
            tir.BufferStore(Z, tir.FloatImm(f16, 0.0), [tir.IntImm("int32", 0), col]),
        ),
    )
    func = _wrap(body)
    func = func.with_attr("plena.lane_axis", "by")
    mid = fold_run(func)
    failures = 0
    failures += _check("lane_axes", mid.lane_axes, ["by"])
    if mid.body and isinstance(mid.body[0], ir.ParallelAxis):
        outer = mid.body[0]
        failures += _check("outer kind", outer.kind, ir.ParallelKind.BLOCK_IDX)
        failures += _check("outer thread_tag", outer.thread_tag, "blockIdx.y")
        failures += _check("outer axis_name", outer.axis_name, "by")
        failures += _check("outer extent", outer.extent, 4)
    else:
        failures += _check("outer is ParallelAxis", type(mid.body[0]).__name__, "ParallelAxis")
    return failures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_fold_dma()
    failures += test_fold_gemm_btmm()
    failures += test_fold_gemm_per_head()
    failures += test_fold_reduce()
    failures += test_fold_parallel_add()
    failures += test_fold_parallel_zero()
    failures += test_fold_parallel_exp()
    failures += test_fold_broadcast_sub_fp()
    failures += test_fold_broadcast_mul_fp()
    failures += test_fold_broadcast_left_operand()
    failures += test_fold_unfoldable_falls_back_to_for()
    failures += test_fold_conv2d_zero_pad_init()
    failures += test_fold_conv2d_serial_copy()
    failures += test_fold_conv2d_shifted_copy()
    failures += test_fold_preserves_blockidx()
    print()
    if failures == 0:
        print("PASS — all mid_ir.fold tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
