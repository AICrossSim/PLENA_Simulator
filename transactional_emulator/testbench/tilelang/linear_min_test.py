"""
TVM tilelang linear_min test — end-to-end test using tilelang_tvm_compiler.

This test demonstrates using the tilelang_tvm_compiler to compile a kernel
(linear_min) and run it through the transactional emulator for numerical
verification.

The tilelang_tvm_compiler compiles TVM PrimFunc kernels written using the
tilelang DSL to PLENA ISA text. This test:
  1. Compiles linear_min via subprocess (the compiler runs in .venv with TVM)
  2. Generates input tensors and CPU golden reference
  3. Creates simulation environment files (HBM, fp_sram, etc.)
  4. Runs the Rust transactional emulator
  5. Compares VRAM output against the golden reference

Prerequisites:
  - TVM and tilelang must be installed in the venv (.venv by default)
  - To use a different venv, set TILELANG_VENV=".venv-tvm" environment variable
  - The transactional emulator must be built: cargo build --release in transactional_emulator/

Run with: just test-tilelang-linear-min
"""

from pathlib import Path
import sys
import os
import importlib.util

# Ensure imports work from testbench directory
REPO_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "PLENA_Compiler"))
sys.path.insert(0, str(REPO_ROOT / "PLENA_Tools"))

# Configurable venv for TVM compiler subprocess
TILELANG_VENV = os.environ.get("TILELANG_VENV", ".venv")

import torch

# Import test_helper directly as a module file to avoid loading TVM via __init__.py
# The TVM compiler runs in a subprocess, so we don't need TVM in the main process.
_test_helper_path = REPO_ROOT / "PLENA_Compiler" / "tilelang_tvm_compiler" / "test_helper.py"
_spec = importlib.util.spec_from_file_location("test_helper", _test_helper_path)
test_helper = importlib.util.module_from_spec(_spec)
sys.modules["test_helper"] = test_helper  # Required for dataclass decorator to work
_spec.loader.exec_module(test_helper)
TvmTestbenchSpec = test_helper.TvmTestbenchSpec
run = test_helper.run

from transactional_emulator.testbench.emulator_runner import run_and_assert


# ============================================================================
# Parameters
# ============================================================================
MLEN = 64
M_BLOCKS = 1   # M = M_BLOCKS * MLEN = 64 rows
N_BLOCKS = 2   # N = N_BLOCKS * MLEN = 128 cols (output features)
K_BLOCKS = 2   # K = K_BLOCKS * MLEN = 128 (input features)

M = M_BLOCKS * MLEN
N = N_BLOCKS * MLEN
K = K_BLOCKS * MLEN


def build_inputs_and_golden(seed: int) -> dict:
    """Generate input tensors and compute CPU golden reference.

    linear_min expects inputs in BSHD format (batch, seq, heads, dim):
      - A_hbm: (1, M, 1, K) - input activation
      - B_hbm: (1, N, 1, K) - weight matrix (transposed layout)
      - C_hbm: (1, M, 1, N) - output

    The kernel computes: C = A @ B^T (matrix multiply with transposed B)
    """
    torch.manual_seed(seed)

    # Create input tensors in 4D BSHD format
    # A: input activations (M, K) -> (1, M, 1, K)
    # B: weight matrix (N, K) -> (1, N, 1, K) - note: B is stored row-major
    A = torch.randn(1, M, 1, K, dtype=torch.float16)
    B = torch.randn(1, N, 1, K, dtype=torch.float16)

    # CPU golden: C = A @ B^T
    # Reshape for matmul: A (1,M,1,K) -> (M,K), B (1,N,1,K) -> (N,K)
    A_2d = A.view(M, K)
    B_2d = B.view(N, K)
    C_2d = A_2d @ B_2d.T  # (M, K) @ (K, N) = (M, N)

    # Golden in flat format for comparison
    golden_flat = C_2d.contiguous()

    return {
        "hbm_inputs": {
            "A_hbm": A,
            "B_hbm": B,
        },
        "golden_flat": golden_flat,
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    """Build comparison parameters for view_mem.py.

    The comparator needs to know:
    - start_row_idx: which VRAM row the output starts at
    - num_rows: how many MLEN-wide rows to compare
    - num_batches: batch dimension for reassembly
    - elements_per_batch: output columns per batch
    """
    golden = io["golden_flat"]
    total_elements = golden.numel()
    num_rows = total_elements // MLEN

    # For staged output, the output is copied to VRAM starting at row 0
    return {
        "start_row_idx": 0,
        "num_rows": num_rows,
        "num_batches": M,  # treat each M row as a "batch"
        "elements_per_batch": N,  # each row has N output elements
        "row_dim": MLEN,
        "use_stride_mode": True,
    }


def main():
    print("=" * 80)
    print("TVM tilelang linear_min Test")
    print("=" * 80)
    print(f"  M={M} (M_BLOCKS={M_BLOCKS}), N={N} (N_BLOCKS={N_BLOCKS}), K={K} (K_BLOCKS={K_BLOCKS})")
    print(f"  MLEN={MLEN}")
    print()

    spec = TvmTestbenchSpec(
        asm_name="linear_min",
        kernel="tilelang_tvm_compiler.kernels.linear_min:make_linear_min",
        kernel_kwargs={
            "m_blocks": M_BLOCKS,
            "n_blocks": N_BLOCKS,
            "k_blocks": K_BLOCKS,
            "with_bias": False,
        },
        build_inputs_and_golden=build_inputs_and_golden,
        build_comparison_params=build_comparison_params,
        mlen=MLEN,
        stage_output="C_hbm",  # Stage output from HBM to VRAM for comparison
        seed=42,
        venv_name=TILELANG_VENV,  # Configurable via TILELANG_VENV env var
    )

    # Run the test helper to compile + generate simulation environment
    run(spec)

    # Run emulator and verify results
    build_dir = Path(__file__).parent.parent / "build"
    run_and_assert(build_dir, "linear_min", mlen=MLEN, blen=4)


if __name__ == "__main__":
    main()
