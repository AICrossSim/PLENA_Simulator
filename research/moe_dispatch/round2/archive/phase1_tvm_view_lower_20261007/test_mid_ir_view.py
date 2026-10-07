"""Unit tests for mid_ir.passes.view (pass_4b).

Coverage:
  * Non-global ref gets phase prepended + view_perm set
    (BSHD by default, BHSD for btmm_out and per_head_lhs)
  * HBM ref doesn't get rank-grown but lane var is substituted
    with the composite expression
  * Broadcast.broadcast_dims shifts by 1 (rank grew)
  * Global view conflict (same buffer, two different perms) raises
  * cluster_guard skip (no lane_axes / D >= MLEN) → no-op
  * Outside cluster body: refs not rewritten

Run:
    python -m tilelang_tvm_compiler.tests.test_mid_ir_view
"""

from __future__ import annotations

import sys

from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.view import (
    ViewConflictError,
    run as view_run,
)


LANE = 4


def _mk_buf(name, shape, scope="shared"):
    return ir.BufferDef(name=name, shape=shape, dtype="float16", scope=scope)


def _ref(buf, indices):
    return ir.BufferRef(buf, list(indices))


def _slice_ref(buf, n):
    """Build a BufferRef with `n` Slice indices. Used to model a
    pre-grow ref (rank N) into a now-grown buffer (rank N+1)."""
    return ir.BufferRef(buf, [ir.Slice() for _ in range(n)])


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


def _cluster(body):
    return ir.ParallelAxis(
        axis_name="by_phase",
        extent=LANE,
        body=body,
        kind=ir.ParallelKind.CLUSTER,
        thread_tag=None,
        parent_grid_axis_name="by_number",
    )


def _grid(body):
    return ir.ParallelAxis(
        axis_name="by_number",
        extent=1,
        body=body,
        kind=ir.ParallelKind.BLOCK_IDX,
        thread_tag="blockIdx.y",
    )


def _wrap(body, allocs=()):
    return ir.MidFunc(
        name="t",
        params=[],
        allocs=list(allocs),
        body=list(body),
        lane_axes=["by"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dma_lane_ref_bshd() -> int:
    """DMA dst = on-chip → BSHD perm; phase prepended."""
    print("test_dma_lane_ref_bshd — DMA dst gets BSHD view + prepend")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16], scope="shared")  # post-grow
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Dma(
                                src=_ref(Q_hbm, [0, ir.Slice(), "by", ir.Slice()]),
                                dst=_slice_ref(Q_sh, n=2),
                                marker=ir.Marker.DMA,
                                can_async=True,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[Q_sh],
    )
    out = view_run(fn)
    dma = out.body[0].body[0].body[0]
    failures = 0
    # On-chip dst: prepended phase, BSHD perm = [1, 0, 2]
    failures += _check("Q_sh indices", dma.dst.indices, ["by_phase", ir.Slice(), ir.Slice()])
    failures += _check("Q_sh view_perm (BSHD)", dma.dst.view_perm, [1, 0, 2])
    return failures


def test_btmm_output_bhsd() -> int:
    """BTMM C (S_loc) → BHSD = identity perm."""
    print("test_btmm_output_bhsd")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16], scope="shared")
    K_sh = _mk_buf("K_sh", [LANE, 64, 16], scope="shared")
    S_loc = _mk_buf("S_loc", [LANE, 64, 64], scope="fragment")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Gemm(
                                a=_slice_ref(Q_sh, 2),
                                b=_slice_ref(K_sh, 2),
                                c=_slice_ref(S_loc, 2),
                                kind="btmm",
                                transpose_b=True,
                                marker=ir.Marker.BTMM,
                                can_async=True,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[Q_sh, K_sh, S_loc],
    )
    out = view_run(fn)
    g = out.body[0].body[0].body[0]
    failures = 0
    failures += _check("a (Q_sh) BSHD", g.a.view_perm, [1, 0, 2])
    failures += _check("b (K_sh) BSHD", g.b.view_perm, [1, 0, 2])
    failures += _check("c (S_loc) BHSD identity", g.c.view_perm, [0, 1, 2])
    return failures


def test_per_head_matmul_lhs_bhsd() -> int:
    """per-head matmul (kind=overwrite) LHS → BHSD."""
    print("test_per_head_matmul_lhs_bhsd")
    S = _mk_buf("S", [LANE, 64, 64], scope="fragment")
    V = _mk_buf("V", [LANE, 64, 16], scope="shared")
    P = _mk_buf("P", [LANE, 64, 16], scope="fragment")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Gemm(
                                a=_slice_ref(S, 2),
                                b=_slice_ref(V, 2),
                                c=_slice_ref(P, 2),
                                kind="overwrite",
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[S, V, P],
    )
    out = view_run(fn)
    g = out.body[0].body[0].body[0]
    failures = 0
    failures += _check("a (S) BHSD identity", g.a.view_perm, [0, 1, 2])
    failures += _check("b (V) BSHD", g.b.view_perm, [1, 0, 2])
    failures += _check("c (P) BSHD", g.c.view_perm, [1, 0, 2])
    return failures


def test_hbm_ref_lane_var_subst() -> int:
    """HBM ref's "by" → composite; rank unchanged; no view_perm set."""
    print("test_hbm_ref_lane_var_subst")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16], scope="shared")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Dma(
                                src=_ref(Q_hbm, [0, ir.Slice(), "by", ir.Slice()]),
                                dst=_slice_ref(Q_sh, 2),
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[Q_sh],
    )
    out = view_run(fn)
    src = out.body[0].body[0].body[0].src
    failures = 0
    failures += _check("HBM rank unchanged", len(src.indices), 4)
    failures += _check("HBM view_perm None", src.view_perm, None)
    expected_by = {
        "op": "add",
        "args": ["by_phase", {"op": "mul", "args": ["by_number", LANE]}],
    }
    failures += _check("HBM[2] composite", src.indices[2], expected_by)
    return failures


def test_broadcast_dims_shift() -> int:
    """Elementwise(SUB, [S, Broadcast(M, [1])]) — dst rank grows by 1
    (prepend), so broadcast_dims must shift by 1 too. Use D<MLEN
    everywhere to avoid the cluster_guard skip path."""
    print("test_broadcast_dims_shift")
    S = _mk_buf("S", [LANE, 64, 16], scope="fragment")  # D=16 < MLEN
    M = _mk_buf("M", [LANE, 16], scope="fragment")  # D=16 < MLEN
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Elementwise(
                                dst=_slice_ref(S, 2),
                                srcs=[
                                    _slice_ref(S, 2),
                                    ir.Broadcast(src=_slice_ref(M, 1), broadcast_dims=[1]),
                                ],
                                op=ir.BinOp.SUB,
                                marker=ir.Marker.LANE_OP,
                                can_async=False,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[S, M],
    )
    out = view_run(fn)
    ew = out.body[0].body[0].body[0]
    failures = 0
    failures += _check("dst[0] prepended", ew.dst.indices[0], "by_phase")
    # Elementwise with a Broadcast src → BHSD (matches the BTMM
    # output it usually consumes); no permute needed.
    failures += _check("dst view_perm BHSD identity", ew.dst.view_perm, [0, 1, 2])
    bcast = ew.srcs[1]
    failures += _check("Broadcast preserved", type(bcast).__name__, "Broadcast")
    failures += _check("broadcast_dims shifted", bcast.broadcast_dims, [2])
    failures += _check("M ref BHSD identity", bcast.src.view_perm, [0, 1])
    return failures


def test_global_consistency_conflict() -> int:
    """Same buffer used as Gemm[btmm].c (BHSD) AND Gemm[btmm].a (BSHD)
    — conflict, raises."""
    print("test_global_consistency_conflict")
    X = _mk_buf("X", [LANE, 64, 16], scope="fragment")
    K = _mk_buf("K", [LANE, 64, 16], scope="shared")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            # btmm output → X gets BHSD
                            ir.Gemm(
                                a=_slice_ref(X, 2),
                                b=_slice_ref(K, 2),
                                c=_slice_ref(X, 2),
                                kind="btmm",
                                transpose_b=True,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[X, K],
    )
    try:
        view_run(fn)
    except ViewConflictError as e:
        print(f"  [OK]   raised ViewConflictError: {str(e)[:80]}...")
        return 0
    print("  [FAIL] expected ViewConflictError")
    return 1


def test_skip_when_no_lane_axes() -> int:
    """No lane_axes declared → guard skips."""
    print("test_skip_when_no_lane_axes")
    Q = _mk_buf("Q", [LANE, 64, 16])
    fn = ir.MidFunc(
        name="t",
        params=[],
        allocs=[Q],
        body=[ir.Dma(src=_slice_ref(Q, 3), dst=_slice_ref(Q, 3))],
        lane_axes=[],
    )
    out = view_run(fn)
    return _check("body unchanged", out.body[0].src.view_perm, None)


def test_skip_when_d_ge_mlen() -> int:
    """All non-global D >= MLEN → guard skips."""
    print("test_skip_when_d_ge_mlen")
    A = _mk_buf("A", [4, 64], scope="shared")  # D=64=MLEN
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Dma(src=_slice_ref(A, 2), dst=_slice_ref(A, 2)),
                        ]
                    )
                ]
            )
        ],
        allocs=[A],
    )
    out = view_run(fn)
    dma = out.body[0].body[0].body[0]
    return _check("view_perm not set (skipped)", dma.src.view_perm, None)


def test_outside_cluster_untouched() -> int:
    """Op directly inside a grid (no cluster) — refs not rewritten."""
    print("test_outside_cluster_untouched")
    A = _mk_buf("A", [LANE, 64, 16])
    fn = _wrap(
        [
            _grid(
                [
                    ir.Dma(src=_slice_ref(A, 3), dst=_slice_ref(A, 3)),
                ]
            )
        ],
        allocs=[A],
    )
    out = view_run(fn)
    dma = out.body[0].body[0]
    return _check("view_perm None", dma.src.view_perm, None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_dma_lane_ref_bshd()
    failures += test_btmm_output_bhsd()
    failures += test_per_head_matmul_lhs_bhsd()
    failures += test_hbm_ref_lane_var_subst()
    failures += test_broadcast_dims_shift()
    failures += test_global_consistency_conflict()
    failures += test_skip_when_no_lane_axes()
    failures += test_skip_when_d_ge_mlen()
    failures += test_outside_cluster_untouched()
    print()
    if failures == 0:
        print("PASS — all mid_ir.view tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
