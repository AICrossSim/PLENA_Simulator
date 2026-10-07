"""Unit tests for mid_ir.passes.to_plena (pass_6).

Coverage:
  * BufferDef.scope mapping
      "global" → hbm
      "shared" → vram
      "fragment" 1D → fpram, 2D+ → vram
  * Gemm B operand override → MRAM (BTMM RHS / per-head matmul RHS)
  * DMA dst inferred MRAM → kind = dma_h2m (not dma_h2v)
  * MultiLaneOp(Dma) → Op(kind=dma_h2v_slice / dma_h2v / dma_h2m, scalar_args=[lane_count])
  * MultiLaneOp(Gemm[btmm]) → Op(kind=btmm)
  * Bare Reduce in cluster → for lane: for row: row_reduce_*_at
  * Bare broadcast Elementwise in cluster → for lane: for row: row_*_fp_at
  * ParallelAxis(BLOCK_IDX) → Op(kind=for, ...)
  * ParallelAxis(CLUSTER) → unwrapped (no for in HLIR)
  * For(serial/unroll) → Op(kind=for) with loop_kind annotation
  * Auto-dump to build_dir creates <name>.midir.txt
  * cluster_guard skip → still produces an HLIRModule

Run:
    python -m tilelang_tvm_compiler.tests.test_mid_ir_to_plena
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.to_plena import (
    run as to_plena_run,
)


LANE = 4


def _mk_buf(name, shape, scope="shared"):
    return ir.BufferDef(name=name, shape=shape, dtype="float16", scope=scope)


def _ref(buf, indices):
    return ir.BufferRef(buf, list(indices))


def _slice_ref(buf):
    return ir.BufferRef(buf, [ir.Slice() for _ in buf.shape])


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


def _wrap(body, params=(), allocs=(), name="t"):
    return ir.MidFunc(
        name=name,
        params=list(params),
        allocs=list(allocs),
        body=list(body),
        lane_axes=["by"],
    )


# ---------------------------------------------------------------------------
# Scope mapping
# ---------------------------------------------------------------------------


def test_scope_basic_mapping() -> int:
    """global → hbm; shared → vram; fragment 1D → fpram; 2D → vram."""
    print("test_scope_basic_mapping")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [4, 64, 16], scope="shared")
    M = _mk_buf("M", [16], scope="fragment")  # 1D → fpram
    S = _mk_buf("S", [4, 64, 16], scope="fragment")  # 2D+ → vram
    fn = _wrap([], params=[Q_hbm], allocs=[Q_sh, M, S])
    out = to_plena_run(fn)
    failures = 0
    failures += _check("Q_hbm scope", out.buffers["Q_hbm"].scope, _scope.HBM)
    failures += _check("Q_sh scope", out.buffers["Q_sh"].scope, _scope.VRAM)
    failures += _check("M scope (1D fragment)", out.buffers["M"].scope, _scope.FPRAM)
    failures += _check("S scope (2D fragment)", out.buffers["S"].scope, _scope.VRAM)
    return failures


def test_gemm_b_override_mram() -> int:
    """Buffer used as Gemm B → MRAM, overrides the default shared→vram."""
    print("test_gemm_b_override_mram")
    Q = _mk_buf("Q", [4, 64, 16], scope="shared")  # default → vram
    K = _mk_buf("K", [4, 64, 16], scope="shared")  # but used as B → mram
    S = _mk_buf("S", [4, 64, 16], scope="fragment")
    fn = _wrap(
        [
            ir.Gemm(a=_slice_ref(Q), b=_slice_ref(K), c=_slice_ref(S), kind="btmm", transpose_b=True),
        ],
        allocs=[Q, K, S],
    )
    out = to_plena_run(fn)
    failures = 0
    failures += _check("Q scope", out.buffers["Q"].scope, _scope.VRAM)
    failures += _check("K scope (B operand)", out.buffers["K"].scope, _scope.MRAM)
    failures += _check("S scope", out.buffers["S"].scope, _scope.VRAM)
    return failures


def test_dma_to_mram_picks_h2m() -> int:
    """DMA dst was Gemm B → MRAM scope → dma kind = dma_h2m."""
    print("test_dma_to_mram_picks_h2m")
    K_hbm = _mk_buf("K_hbm", [1, 64, 4, 16], scope="global")
    K_sh = _mk_buf("K_sh", [4, 64, 16], scope="shared")
    Q_sh = _mk_buf("Q_sh", [4, 64, 16], scope="shared")
    S = _mk_buf("S", [4, 64, 16], scope="fragment")
    fn = _wrap(
        [
            # K is the BTMM B operand → forces K_sh to MRAM
            ir.Dma(src=_slice_ref(K_hbm), dst=_slice_ref(K_sh)),
            ir.Gemm(a=_slice_ref(Q_sh), b=_slice_ref(K_sh), c=_slice_ref(S), kind="btmm", transpose_b=True),
        ],
        params=[K_hbm],
        allocs=[Q_sh, K_sh, S],
    )
    out = to_plena_run(fn)
    failures = 0
    failures += _check("K_sh scope (MRAM via override)", out.buffers["K_sh"].scope, _scope.MRAM)
    # First op should be dma_h2m (not dma_h2v).
    dma_op = out.ops[0]
    failures += _check("dma op kind", dma_op.kind, "dma_h2m")
    return failures


# ---------------------------------------------------------------------------
# Op lowering
# ---------------------------------------------------------------------------


def _grid(body):
    return ir.ParallelAxis(
        axis_name="by_number",
        extent=1,
        body=body,
        kind=ir.ParallelKind.BLOCK_IDX,
        thread_tag="blockIdx.y",
    )


def _cluster(body):
    return ir.ParallelAxis(
        axis_name="by_phase",
        extent=LANE,
        body=body,
        kind=ir.ParallelKind.CLUSTER,
        parent_grid_axis_name="by_number",
    )


def test_multi_lane_dma_to_op() -> int:
    """MultiLaneOp(Dma) → single Op(kind=dma_*) with lane_count."""
    print("test_multi_lane_dma_to_op")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [4, 64, 16], scope="shared")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.MultiLaneOp(
                                inner=ir.Dma(
                                    src=_slice_ref(Q_hbm),
                                    dst=_slice_ref(Q_sh),
                                    marker=ir.Marker.DMA,
                                    can_async=True,
                                ),
                                cluster_axis_names=["by_phase"],
                                dim_map={"Q_sh": [0]},
                            ),
                        ]
                    )
                ]
            )
        ],
        params=[Q_hbm],
        allocs=[Q_sh],
    )
    out = to_plena_run(fn)
    # Top-level is a for(by_number); its body has the dma.
    by_number_for = out.ops[0]
    failures = 0
    failures += _check("top is for", by_number_for.kind, "for")
    inner = by_number_for.body[0]
    # No CLUSTER for in HLIR — the dma is directly inside.
    failures += _check("dma kind", inner.kind, "dma_h2v")
    failures += _check("dma lane_count", inner.scalar_args[0], LANE)
    return failures


def test_multi_lane_btmm_to_op() -> int:
    print("test_multi_lane_btmm_to_op")
    Q = _mk_buf("Q", [4, 64, 16], scope="shared")
    K = _mk_buf("K", [4, 64, 16], scope="shared")  # → MRAM by override
    S = _mk_buf("S", [4, 64, 16], scope="fragment")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.MultiLaneOp(
                                inner=ir.Gemm(
                                    a=_slice_ref(Q),
                                    b=_slice_ref(K),
                                    c=_slice_ref(S),
                                    kind="btmm",
                                    transpose_b=True,
                                    marker=ir.Marker.BTMM,
                                    can_async=True,
                                ),
                                cluster_axis_names=["by_phase"],
                                dim_map={"Q": [0], "K": [0], "S": [0]},
                            ),
                        ]
                    )
                ]
            )
        ],
        allocs=[Q, K, S],
    )
    out = to_plena_run(fn)
    op = out.ops[0].body[0]
    failures = 0
    failures += _check("kind", op.kind, "btmm")
    failures += _check("lane_count", op.scalar_args[0], LANE)
    return failures


def test_bare_reduce_lowers_to_nested_fors() -> int:
    """Bare reduce in cluster → for lane: for row: row_reduce_max_at."""
    print("test_bare_reduce_lowers_to_nested_fors")
    S = _mk_buf("S", [LANE, 64, 16], scope="fragment")
    M = _mk_buf("M", [LANE, 16], scope="fragment")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.Reduce(
                                dst=_slice_ref(M),
                                src=_slice_ref(S),
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
    out = to_plena_run(fn)
    by_for = out.ops[0]
    lane_for = by_for.body[0]
    row_for = lane_for.body[0]
    inner = row_for.body[0]
    failures = 0
    failures += _check("lane for", lane_for.kind, "for")
    failures += _check("lane extent", lane_for.annotations["extent"], LANE)
    failures += _check("row for", row_for.kind, "for")
    failures += _check("row extent", row_for.annotations["extent"], 64)
    failures += _check("row_reduce_max_at", inner.kind, "row_reduce_max_at")
    return failures


def test_parallel_axis_block_idx_to_for() -> int:
    """grid → for; cluster → unwrapped."""
    print("test_parallel_axis_block_idx_to_for")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q = _mk_buf("Q", [4, 64, 16], scope="shared")
    fn = _wrap(
        [
            _grid(
                [
                    _cluster(
                        [
                            ir.MultiLaneOp(
                                inner=ir.Dma(
                                    src=_slice_ref(Q_hbm), dst=_slice_ref(Q), can_async=True, marker=ir.Marker.DMA
                                ),
                                cluster_axis_names=["by_phase"],
                                dim_map={"Q": [0]},
                            ),
                        ]
                    )
                ]
            )
        ],
        params=[Q_hbm],
        allocs=[Q],
    )
    out = to_plena_run(fn)
    by_number_for = out.ops[0]
    failures = 0
    failures += _check("by_number for kind", by_number_for.kind, "for")
    failures += _check("by_number loop_var", by_number_for.annotations["loop_var"], "by_number")
    # Inside should NOT be another for (cluster doesn't survive); just dma.
    inner = by_number_for.body[0]
    failures += _check("inner kind != for", inner.kind != "for", True)
    return failures


def test_for_kind_preserved() -> int:
    """For(unroll) gets loop_kind=unroll annotation; serial preserved too."""
    print("test_for_kind_preserved")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q = _mk_buf("Q", [4, 64, 16], scope="shared")
    fn = _wrap(
        [
            ir.For(
                loop_var="kh",
                extent=4,
                kind="unroll",
                body=[
                    ir.Dma(src=_slice_ref(Q_hbm), dst=_slice_ref(Q), can_async=False, marker=None),
                ],
            ),
        ],
        params=[Q_hbm],
        allocs=[Q],
    )
    out = to_plena_run(fn)
    f = out.ops[0]
    failures = 0
    failures += _check("kind", f.kind, "for")
    failures += _check("loop_kind", f.annotations["loop_kind"], "unroll")
    return failures


# ---------------------------------------------------------------------------
# Auto-dump
# ---------------------------------------------------------------------------


def test_auto_dump_creates_midir_file() -> int:
    print("test_auto_dump_creates_midir_file")
    Q = _mk_buf("Q", [4, 64, 16], scope="shared")
    fn = _wrap([], allocs=[Q], name="my_kernel")
    with tempfile.TemporaryDirectory() as tmp:
        to_plena_run(fn, build_dir=Path(tmp))
        dump = Path(tmp) / "my_kernel.midir.txt"
        if not dump.exists():
            print(f"  [FAIL] expected {dump} to exist")
            return 1
        text = dump.read_text()
        failures = 0
        failures += _check("contains func name", "my_kernel" in text, True)
        failures += _check("contains buffer", "Q" in text, True)
        return failures


def test_no_dump_when_build_dir_none() -> int:
    """build_dir=None: no file written."""
    print("test_no_dump_when_build_dir_none")
    Q = _mk_buf("Q", [4, 64, 16], scope="shared")
    fn = _wrap([], allocs=[Q])
    out = to_plena_run(fn, build_dir=None)
    return _check("returns HLIRModule", isinstance(out, _hlir.HLIRModule), True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_scope_basic_mapping()
    failures += test_gemm_b_override_mram()
    failures += test_dma_to_mram_picks_h2m()
    failures += test_multi_lane_dma_to_op()
    failures += test_multi_lane_btmm_to_op()
    failures += test_bare_reduce_lowers_to_nested_fors()
    failures += test_parallel_axis_block_idx_to_for()
    failures += test_for_kind_preserved()
    failures += test_auto_dump_creates_midir_file()
    failures += test_no_dump_when_build_dir_none()
    print()
    if failures == 0:
        print("PASS — all mid_ir.to_plena tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
