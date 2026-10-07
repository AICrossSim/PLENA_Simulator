"""Unit tests for mid_ir.passes.fuse (pass_5).

Coverage:
  * Async wrapping a Dma → MultiLaneOp(inner=Dma, ...)
  * cluster_axis_names = list of enclosing CLUSTER axes (outer→inner)
  * dim_map: every non-global buffer the op touches gets [0]
  * HBM buffer NOT in dim_map
  * Bare can_async=False ops (Reduce) stay unwrapped
  * Outside cluster: skipped
  * Nested clusters → multi-axis cluster_axis_names
  * cluster_guard skip → no-op

Run:
    python -m tilelang_tvm_compiler.tests.test_mid_ir_fuse
"""

from __future__ import annotations

import sys

from tvm import tir as _tir

from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.fuse import (
    FuseError,
    run as fuse_run,
)


LANE = 4


def _mk_buf(name, shape, scope="shared"):
    return ir.BufferDef(name=name, shape=shape, dtype="float16", scope=scope)


def _ref(buf, indices):
    return ir.BufferRef(buf, list(indices))


def _slice_ref(buf, n):
    return ir.BufferRef(buf, [ir.Slice() for _ in range(n)])


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


# Build a coherent ``by`` lane-axis nest with all the VarRef identity
# fields populated, the way split would emit it. Exposes the wrapped
# tir.Vars so test bodies that need to reference ``by`` in indices can
# use the matching VarRef.
def _lane_vars():
    by = _tir.Var("by", "int32")
    by_number = _tir.Var("by_number", "int32")
    by_phase = _tir.Var("by_phase", "int32")
    return {
        "original": ir.VarRef(by),
        "number": ir.VarRef(by_number),
        "phase": ir.VarRef(by_phase),
    }


_LANE = _lane_vars()


def _cluster(body, axis_name="by_phase", parent="by_number"):
    return ir.ParallelAxis(
        axis_name=axis_name,
        extent=LANE,
        body=body,
        kind=ir.ParallelKind.CLUSTER,
        thread_tag=None,
        parent_grid_axis_name=parent,
        original_axis_name="by",
        axis_var=_LANE["phase"],
        original_axis_var=_LANE["original"],
    )


def _grid(body, axis_name="by_number", tag="blockIdx.y"):
    return ir.ParallelAxis(
        axis_name=axis_name,
        extent=1,
        body=body,
        kind=ir.ParallelKind.BLOCK_IDX,
        thread_tag=tag,
        original_axis_name="by",
        axis_var=_LANE["number"],
        original_axis_var=_LANE["original"],
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


def test_async_dma_collapses_to_multi_lane() -> int:
    print("test_async_dma_collapses_to_multi_lane")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16], scope="shared")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Async(
                                body=[
                                    ir.Dma(
                                        src=_ref(Q_hbm, [0, ir.Slice(), _LANE["original"], ir.Slice()]),
                                        dst=_slice_ref(Q_sh, 3),
                                        marker=ir.Marker.DMA,
                                        can_async=True,
                                    ),
                                ],
                                scope_id=0,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[Q_sh],
    )
    out = fuse_run(fn)
    cluster = out.body[0].body[0]
    failures = 0
    failures += _check("body length", len(cluster.body), 1)
    failures += _check("body[0] is MultiLaneOp", type(cluster.body[0]).__name__, "MultiLaneOp")
    if isinstance(cluster.body[0], ir.MultiLaneOp):
        mlo = cluster.body[0]
        failures += _check("inner is Dma", type(mlo.inner).__name__, "Dma")
        failures += _check("cluster_axis_names", mlo.cluster_axis_names, ["by_phase"])
        # Q_hbm is global → not in dim_map; Q_sh is non-global → [0]
        failures += _check("dim_map keys", set(mlo.dim_map.keys()), {"Q_sh"})
        failures += _check("dim_map['Q_sh']", mlo.dim_map["Q_sh"], [0])
    return failures


def test_async_btmm_collapses() -> int:
    """BTMM: dim_map should mention all 3 lane-aware buffers."""
    print("test_async_btmm_collapses")
    Q = _mk_buf("Q", [LANE, 64, 16], scope="shared")
    K = _mk_buf("K", [LANE, 64, 16], scope="shared")
    S = _mk_buf("S", [LANE, 64, 64], scope="fragment")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Async(
                                body=[
                                    ir.Gemm(
                                        a=_slice_ref(Q, 3),
                                        b=_slice_ref(K, 3),
                                        c=_slice_ref(S, 3),
                                        kind="btmm",
                                        transpose_b=True,
                                        marker=ir.Marker.BTMM,
                                        can_async=True,
                                    ),
                                ],
                                scope_id=0,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[Q, K, S],
    )
    out = fuse_run(fn)
    mlo = out.body[0].body[0].body[0]
    failures = 0
    failures += _check("type", type(mlo).__name__, "MultiLaneOp")
    failures += _check("dim_map keys", set(mlo.dim_map.keys()), {"Q", "K", "S"})
    for n in ("Q", "K", "S"):
        failures += _check(f"dim_map[{n}]", mlo.dim_map[n], [0])
    return failures


def test_reduce_stays_bare() -> int:
    """Reduce (can_async=False) is not in an Async, so fuse leaves it
    as-is."""
    print("test_reduce_stays_bare")
    S = _mk_buf("S", [LANE, 64, 16], scope="fragment")
    M = _mk_buf("M", [LANE, 64], scope="fragment")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Reduce(
                                dst=_slice_ref(M, 2),
                                src=_slice_ref(S, 3),
                                op=ir.ReduceOp.MAX,
                                axis=2,
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
    out = fuse_run(fn)
    inner = out.body[0].body[0].body[0]
    return _check("body[0] still Reduce", type(inner).__name__, "Reduce")


def test_mixed_async_and_bare() -> int:
    """async+bare interleaved → mixed MultiLaneOp + bare ops."""
    print("test_mixed_async_and_bare")
    A = _mk_buf("A", [LANE, 64, 16])
    S = _mk_buf("S", [LANE, 64, 16], scope="fragment")
    M = _mk_buf("M", [LANE, 64], scope="fragment")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Async(
                                body=[
                                    ir.Dma(
                                        src=_slice_ref(A, 3), dst=_slice_ref(A, 3), marker=ir.Marker.DMA, can_async=True
                                    ),
                                ],
                                scope_id=0,
                            ),
                            ir.Reduce(
                                dst=_slice_ref(M, 2),
                                src=_slice_ref(S, 3),
                                op=ir.ReduceOp.MAX,
                                axis=2,
                                marker=ir.Marker.LANE_OP,
                                can_async=False,
                            ),
                            ir.Async(
                                body=[
                                    ir.Dma(
                                        src=_slice_ref(A, 3), dst=_slice_ref(A, 3), marker=ir.Marker.DMA, can_async=True
                                    ),
                                ],
                                scope_id=1,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[A, S, M],
    )
    out = fuse_run(fn)
    body = out.body[0].body[0].body
    failures = 0
    failures += _check("body length", len(body), 3)
    failures += _check("[0]", type(body[0]).__name__, "MultiLaneOp")
    failures += _check("[1]", type(body[1]).__name__, "Reduce")
    failures += _check("[2]", type(body[2]).__name__, "MultiLaneOp")
    return failures


def test_global_buffer_not_in_dim_map() -> int:
    print("test_global_buffer_not_in_dim_map")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16], scope="shared")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Async(
                                body=[
                                    ir.Dma(
                                        src=_slice_ref(Q_hbm, 4),
                                        dst=_slice_ref(Q_sh, 3),
                                        marker=ir.Marker.DMA,
                                        can_async=True,
                                    ),
                                ],
                                scope_id=0,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[Q_sh],
    )
    out = fuse_run(fn)
    mlo = out.body[0].body[0].body[0]
    return _check("Q_hbm not in dim_map", "Q_hbm" in mlo.dim_map, False)


def test_async_outside_cluster_raises() -> int:
    """An Async outside any cluster (shouldn't happen but defend) →
    FuseError."""
    print("test_async_outside_cluster_raises")
    A = _mk_buf("A", [LANE, 64, 16])
    fn = _wrap(
        [
            ir.Async(
                body=[
                    ir.Dma(src=_slice_ref(A, 3), dst=_slice_ref(A, 3), can_async=True),
                ],
                scope_id=0,
            ),
        ],
        allocs=[A],
    )
    try:
        fuse_run(fn)
    except FuseError as e:
        print(f"  [OK]   raised FuseError: {str(e)[:60]}...")
        return 0
    return 1


def test_skip_no_lane_axes() -> int:
    print("test_skip_no_lane_axes")
    A = _mk_buf("A", [LANE, 64, 16])
    fn = ir.MidFunc(
        name="t",
        params=[],
        allocs=[A],
        body=[ir.Dma(src=_slice_ref(A, 3), dst=_slice_ref(A, 3))],
        lane_axes=[],
    )
    out = fuse_run(fn)
    return _check("body unchanged", type(out.body[0]).__name__, "Dma")


def test_index_var_identity_not_name() -> int:
    """Two ``tir.Var`` objects sharing the same ``name_hint`` must not
    collide as cluster-axis references — the pre-VarRef cheat compared
    by name and would have silently replaced the unrelated one. With
    VarRef identity, ``_collapse_lane_axis`` must skip the unrelated
    var.
    """
    print("test_index_var_identity_not_name")
    # ``by`` here is a completely unrelated tir.Var that just happens
    # to share the lane-axis name. It must NOT be collapsed.
    unrelated_by = _tir.Var("by", "int32")
    unrelated_ref = ir.VarRef(unrelated_by)

    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16], scope="shared")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Async(
                                body=[
                                    ir.Dma(
                                        src=_ref(Q_hbm, [0, ir.Slice(), unrelated_ref, ir.Slice()]),
                                        dst=_slice_ref(Q_sh, 3),
                                        marker=ir.Marker.DMA,
                                        can_async=True,
                                    ),
                                ],
                                scope_id=0,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[Q_sh],
    )
    out = fuse_run(fn)
    mlo = out.body[0].body[0].body[0]
    failures = 0
    if not isinstance(mlo, ir.MultiLaneOp):
        failures += _check("inner is MultiLaneOp", type(mlo).__name__, "MultiLaneOp")
        return failures
    # The unrelated ``by`` at index slot 2 must stay a bare VarRef
    # (NOT be collapsed to a ranged_slice). The pre-VarRef code would
    # have name-matched ``"by"`` and replaced it.
    src_indices = mlo.inner.src.indices
    failures += _check("unrelated by preserved as VarRef", isinstance(src_indices[2], ir.VarRef), True)
    if isinstance(src_indices[2], ir.VarRef):
        # Must be the exact unrelated var, not the lane's original.
        failures += _check("identity preserved", src_indices[2].var is unrelated_by, True)
    return failures


def test_skip_d_ge_mlen() -> int:
    print("test_skip_d_ge_mlen")
    A = _mk_buf("A", [4, 64], scope="shared")  # D=64=MLEN → skip
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Async(
                                body=[
                                    ir.Dma(src=_slice_ref(A, 2), dst=_slice_ref(A, 2), can_async=True),
                                ],
                                scope_id=0,
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[A],
    )
    out = fuse_run(fn)
    # Should be a no-op: Async still there.
    return _check("Async preserved (skipped)", type(out.body[0].body[0].body[0]).__name__, "Async")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_async_dma_collapses_to_multi_lane()
    failures += test_async_btmm_collapses()
    failures += test_reduce_stays_bare()
    failures += test_mixed_async_and_bare()
    failures += test_global_buffer_not_in_dim_map()
    failures += test_async_outside_cluster_raises()
    failures += test_skip_no_lane_axes()
    failures += test_index_var_identity_not_name()
    failures += test_skip_d_ge_mlen()
    print()
    if failures == 0:
        print("PASS — all mid_ir.fuse tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
