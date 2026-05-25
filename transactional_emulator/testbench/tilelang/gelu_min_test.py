"""
TVM tilelang gelu_min test — end-to-end test using tilelang_tvm_compiler.

This test demonstrates using the tilelang_tvm_compiler to compile the GELU
activation kernel and run it through the transactional emulator.

GELU (tanh approximation):
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

The tilelang_tvm_compiler automatically:
  - Hoists float constants to FPRAM
  - Decomposes tanh using exp/reci/add/sub/mul primitives
  - Generates PLENA ISA for FP scalar operations

Prerequisites:
  - TVM and tilelang must be installed in the venv (.venv by default)
  - To use a different venv, set TILELANG_VENV=".venv-tvm" environment variable
  - The transactional emulator must be built: cargo build --release in transactional_emulator/

Run with: just test-tilelang-gelu-min
"""

from pathlib import Path
import sys
import os
import math
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
ROWS = MLEN          # Must equal MLEN for gelu_min
HLEN = 16            # Head dimension
HEAD_COUNT = 4       # Number of attention heads
NUM_S_BLOCKS = 1     # Number of sequence blocks
BATCH = 1

SEQ_LEN = NUM_S_BLOCKS * ROWS


def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
    """GELU with tanh approximation (matches the tilelang kernel).

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    inner = sqrt_2_over_pi * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def build_inputs_and_golden(seed: int) -> dict:
    """Generate input tensors and compute CPU golden reference.

    gelu_min expects inputs in BSHD format:
      - X_hbm: (batch, seq_len, head_count, hlen) - input
      - Y_hbm: (batch, seq_len, head_count, hlen) - output

    The kernel applies GELU element-wise.
    """
    torch.manual_seed(seed)

    # Input tensor in BSHD format
    X = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float16)

    # CPU golden: GELU activation with tanh approximation
    # Convert to float32 for precision, then back to float16
    golden = gelu_tanh_approx(X.float()).half()

    # Golden in flat format for comparison
    # Flatten to (batch * seq_len, head_count * hlen) = (rows, cols)
    golden_flat = golden.view(BATCH * SEQ_LEN, HEAD_COUNT * HLEN).contiguous()

    return {
        "hbm_inputs": {
            "X_hbm": X,
        },
        "golden_flat": golden_flat,
    }


def parse_buffer_addrs(raw_addrs: dict) -> dict:
    """Parse buffer addresses from the compiler output.

    The compiler dumps buffer addresses including auto-hoisted FP constants.
    We extract the Y_hbm address for staging.
    """
    result = {}
    for name, entry in raw_addrs.items():
        if isinstance(entry, dict):
            result[name] = {
                "scope": entry.get("scope"),
                "address": entry.get("address"),
                "shape": entry.get("shape"),
                "dtype": entry.get("dtype"),
            }
    return result


def build_comparison_params(io: dict, addrs: dict) -> dict:
    """Build comparison parameters for view_mem.py."""
    golden = io["golden_flat"]
    total_elements = golden.numel()
    num_rows = total_elements // MLEN

    return {
        "start_row_idx": 0,
        "num_rows": num_rows,
        "num_batches": BATCH * SEQ_LEN,
        "elements_per_batch": HEAD_COUNT * HLEN,
        "row_dim": MLEN,
        "use_stride_mode": True,
    }


def main():
    print("=" * 80)
    print("TVM tilelang gelu_min Test")
    print("=" * 80)
    print(f"  ROWS={ROWS}, HLEN={HLEN}, HEAD_COUNT={HEAD_COUNT}")
    print(f"  NUM_S_BLOCKS={NUM_S_BLOCKS}, SEQ_LEN={SEQ_LEN}")
    print(f"  MLEN={MLEN}")
    print()

    spec = TvmTestbenchSpec(
        asm_name="gelu_min",
        kernel="tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min",
        kernel_kwargs={
            "rows": ROWS,
            "hlen": HLEN,
            "head_count": HEAD_COUNT,
            "num_s_blocks": NUM_S_BLOCKS,
            "batch": BATCH,
        },
        build_inputs_and_golden=build_inputs_and_golden,
        build_comparison_params=build_comparison_params,
        parse_buffer_addrs=parse_buffer_addrs,
        mlen=MLEN,
        btmm_hlen=HLEN,
        btmm_lane_count=MLEN // HLEN,
        stage_output="Y_hbm",
        seed=42,
        venv_name=TILELANG_VENV,  # Configurable via TILELANG_VENV env var
    )

    # Run the test helper to compile + generate simulation environment
    run(spec)

    # Run emulator and verify results
    build_dir = Path(__file__).parent.parent / "build"
    run_and_assert(build_dir, "gelu_min", mlen=MLEN, blen=4)


if __name__ == "__main__":
    main()
