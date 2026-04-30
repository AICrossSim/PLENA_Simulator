"""TVM-compiler test for a single BTMM, modeled after attention.py.

Same structure as the runtime-compiler kernel programs:
  - build the kernel (here: import the TIR PrimFunc)
  - prepare inputs
  - compute golden in pure numpy (parallel to torch in attention.py)
  - call the test harness, which compiles + bundles everything

When you add a more complex TVM kernel later (multi-step matmul, real
attention, etc.) it follows the same pattern: define inputs + golden +
call emit_single_output_testbench. The harness handles the rest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_TESTBENCH_DIR = _THIS_DIR.parent
_REPO_ROOT = _TESTBENCH_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "compiler"))
sys.path.insert(0, str(_TESTBENCH_DIR))

from tilelang_tvm_compiler import emit_single_output_testbench
from tilelang_tvm_compiler.kernels.minimal_btmm import (
    GROUP_HEADS,
    HLEN,
    MLEN,
    minimal_btmm,
)


def build_btmm_golden(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """BTMM semantics: per-head Q @ K.T.

    A: (MLEN, GROUP_HEADS, HLEN)   -- played as Q
    B: (MLEN, GROUP_HEADS, HLEN)   -- played as K (we matmul A with B.T per head)
    -> C: (MLEN, GROUP_HEADS, MLEN)
       C[m, h, n] = sum_k A[m, h, k] * B[n, h, k]
    """
    return np.einsum("mhk,nhk->mhn", A, B)


def main() -> int:
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((MLEN, GROUP_HEADS, HLEN)) * 0.5).astype(np.float32)
    B = (rng.standard_normal((MLEN, GROUP_HEADS, HLEN)) * 0.5).astype(np.float32)
    C = np.zeros((MLEN, GROUP_HEADS, MLEN), dtype=np.float32)
    golden = build_btmm_golden(A, B)

    build_dir = _TESTBENCH_DIR / "build"
    paths = emit_single_output_testbench(
        prim_func=minimal_btmm,
        out_buffer="C_hbm",
        input_tensors={"A_hbm": A, "B_hbm": B, "C_hbm": C},
        golden_output=golden,
        asm_name="tvm_btmm_kernel",
        artifact_prefix="tvm_btmm_kernel",
        build_dir=build_dir,
    )

    print("=" * 60)
    print("TVM compiler -- BTMM kernel test artifact bundle")
    print("=" * 60)
    for k, v in paths.items():
        print(f"  {k:>10}: {v}")
    print("=" * 60)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
