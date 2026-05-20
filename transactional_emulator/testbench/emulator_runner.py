"""
Shared helper for running the Rust transactional emulator and comparing results.
Used by ATen-style testbench scripts for end-to-end numerical verification.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from check_mem import compare_vram_with_golden, print_comparison_results
from config_utils import update_plena_config


def run_emulator(build_dir: Path) -> None:
    """Run the Rust transactional emulator with build artifacts from build_dir."""
    emulator_dir = Path(__file__).parent.parent  # transactional_emulator/
    binary = emulator_dir / "target" / "release" / "transactional_emulator"

    if not binary.exists():
        raise FileNotFoundError(
            f"Emulator binary not found: {binary}\n"
            "Run './run.sh build <test>' once to compile it first."
        )

    asm_path = build_dir / "generated_machine_code.mem"
    hbm_path = build_dir / "hbm_for_behave_sim.bin"
    fpsram_path = build_dir / "fp_sram.bin"
    intsram_path = build_dir / "int_sram.bin"

    cmd = [
        str(binary),
        "--opcode", str(asm_path),
        "--hbm", str(hbm_path),
        "--fpsram", str(fpsram_path),
        "--intsram", str(intsram_path),
        "--quiet",
    ]

    # tch's download-libtorch stores libtorch in the Cargo build cache.
    # The binary needs LD_LIBRARY_PATH (Linux) / DYLD_LIBRARY_PATH (macOS) to
    # find it at runtime.  The cargo-bundled libtorch is the one the binary
    # was compiled against (tch-rs 0.20.0); if the parent process inherits a
    # newer libtorch (e.g. from a Python venv with torch 2.10.0), we'd hit
    # ABI mismatches that surface as "Cannot access data pointer of Tensor
    # that doesn't have storage" panics inside transfer_mx_from_hbm.
    libtorch_pattern = str(
        emulator_dir / "target" / "release" / "build" / "torch-sys-*"
        / "out" / "libtorch" / "libtorch" / "lib"
    )
    libtorch_dirs = glob.glob(libtorch_pattern)
    if not libtorch_dirs and os.environ.get("PLENA_EMULATOR_LIBTORCH"):
        libtorch_dirs = [os.environ["PLENA_EMULATOR_LIBTORCH"]]
    env = {**os.environ, "RUST_BACKTRACE": "1"}
    if libtorch_dirs:
        new_ldpath = libtorch_dirs[0]
        existing_ldpath = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{new_ldpath}:{existing_ldpath}" if existing_ldpath else new_ldpath
        )
        # macOS: DYLD_LIBRARY_PATH is what dyld actually consults.  Prepend
        # the cargo-bundled libtorch so it wins over any venv torch the
        # parent shell put on the path.
        existing_dyld = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = (
            f"{new_ldpath}:{existing_dyld}" if existing_dyld else new_ldpath
        )

    # Capture stdout + stderr so:
    #   * Python callers (verify scripts driving the binary via subprocess)
    #     can read the simulator's `Simulation completed. Latency {ns}` line
    #     and `HBM Statistics ...` line (both printed via `eprintln!` in
    #     main.rs — they go to *stderr*).  Previously we inherited the
    #     terminal directly which meant pipe-captured callers saw nothing.
    #   * On panic / non-zero exit, the Rust backtrace lands in the
    #     RuntimeError message instead of vanishing into terminal void.
    # Always forward both streams so an interactive caller sees the live
    # latency output as if we hadn't captured at all.
    result = subprocess.run(cmd, cwd=str(emulator_dir), env=env,
                            capture_output=True, text=True)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Transactional emulator failed (exit code {result.returncode})\n"
            f"--- emulator stderr (first 4KB) ---\n{result.stderr[:4096]}"
        )


def compare_emulator_output(build_dir: Path) -> tuple:
    """
    Compare emulator VRAM output against the golden reference.

    Returns:
        (results dict, params dict)
    """
    emulator_dir = Path(__file__).parent.parent  # transactional_emulator/
    vram_file = emulator_dir / "vram_dump.bin"
    golden_file = build_dir / "golden_result.txt"
    params_file = build_dir / "comparison_params.json"

    with open(params_file) as f:
        params = json.load(f)

    results = compare_vram_with_golden(
        vram_file,
        golden_file,
        exp_width=8,
        man_width=7,
        num_bytes_per_val=2,
        row_dim=params.get("row_dim", 64),
        start_row_idx=params["start_row_idx"],
        num_batches=params["num_batches"],
        num_rows=params["num_rows"],
        elements_per_batch=params["elements_per_batch"],
        use_stride_mode=params.get("use_stride_mode", True),
        use_slice_mode=params.get("use_slice_mode", False),
        slice_per_row=params.get("slice_per_row", None),
        row_stride=params.get("row_stride", 1),
    )
    return results, params


def run_and_assert(build_dir: Path, op_name: str, mlen: int = 64, blen: int = 4) -> None:
    """
    Sync HW config, run the Rust emulator, compare output, exit(1) on failure.

    Args:
        build_dir: Path to the build directory with sim env files.
        op_name:   Operator name used in pass/fail messages.
        mlen:      Matrix tile length — synced to plena_settings.toml before running.
        blen:      Batch tile length — synced to plena_settings.toml before running.
    """
    # VLEN must equal mlen so the emulator's row-address alignment check passes.
    update_plena_config(vlen=mlen, mlen=mlen, blen=blen, verbose=False)

    print("\n--- Running Rust transactional emulator ---")
    run_emulator(build_dir)

    print("\n--- Comparing emulator output vs golden ---")
    results, params = compare_emulator_output(build_dir)
    print_comparison_results(results, verbose=True, comparison_params=params)

    if results["allclose_pass"]:
        print(f"\n[ATen-style {op_name} test PASSED - ISA generated + emulator verified]")
    else:
        print(f"\n[ATen-style {op_name} test FAILED - emulator numerical check failed]")
        sys.exit(1)
