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

from transactional_emulator.tools.check_mem import compare_vram_with_golden, print_comparison_results
from transactional_emulator.tools.config_utils import update_plena_config


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _use_color() -> bool:
    return os.environ.get("NO_COLOR") is None


def _color(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{color}{text}{_Ansi.RESET}"


def _section(title: str, color: str = _Ansi.CYAN) -> None:
    print(_color(f"\n{title}", _Ansi.BOLD + color))


def _note(text: str) -> None:
    print(_color(f"  # {text}", _Ansi.DIM))


def _path_status(path: Path) -> str:
    if path.exists():
        return _color("found", _Ansi.GREEN)
    return _color("missing", _Ansi.YELLOW)


def _print_artifact(label: str, path: Path, meaning: str) -> None:
    print(f"  {_color(label + ':', _Ansi.BOLD)} {path}")
    print(f"    {_color('meaning:', _Ansi.CYAN)} {meaning}")
    print(f"    {_color('status:', _Ansi.CYAN)}  {_path_status(path)}")


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

    print(_color("  Rust emulator launch plan", _Ansi.BOLD))
    _note("The lines below are the files passed to the Rust binary.")
    _print_artifact("Emulator binary", binary, "compiled Rust executable that performs the simulation")
    _print_artifact("Opcode memory", asm_path, "generated machine code / instruction stream")
    _print_artifact("HBM image", hbm_path, "initial high-bandwidth memory contents")
    _print_artifact("FP SRAM image", fpsram_path, "initial floating-point SRAM contents")
    _print_artifact("INT SRAM image", intsram_path, "initial integer SRAM contents")
    _print_artifact("VRAM dump", emulator_dir / "vram_dump.bin", "output file written by the emulator after execution")

    cmd = [
        str(binary),
        "--opcode",
        str(asm_path),
        "--hbm",
        str(hbm_path),
        "--fpsram",
        str(fpsram_path),
        "--intsram",
        str(intsram_path),
        "--quiet",
    ]
    print(f"  {_color('Command:', _Ansi.BOLD)} {' '.join(cmd)}")
    _note("--quiet suppresses most emulator logs; any text below this line comes from the Rust binary.")

    # tch's download-libtorch stores libtorch in the Cargo build cache.
    # The binary needs LD_LIBRARY_PATH to find it at runtime.
    libtorch_pattern = str(
        emulator_dir
        / "target"
        / "release"
        / "build"
        / "torch-sys-*"
        / "out"
        / "libtorch"
        / "libtorch"
        / "lib"
    )
    libtorch_dirs = glob.glob(libtorch_pattern)
    env = {**os.environ, "RUST_BACKTRACE": "1"}
    if libtorch_dirs:
        existing_ldpath = env.get("LD_LIBRARY_PATH", "")
        new_ldpath = libtorch_dirs[0]
        env["LD_LIBRARY_PATH"] = f"{new_ldpath}:{existing_ldpath}" if existing_ldpath else new_ldpath
        print(f"  {_color('Runtime library path:', _Ansi.BOLD)} {new_ldpath}")
        _note("This lets the Rust binary find libtorch at runtime.")
    else:
        print(f"  {_color('Runtime library path:', _Ansi.BOLD)} not found in Cargo build cache")
        _note("If the emulator fails to load libtorch, build it once so torch-sys downloads libtorch.")

    print(_color("\n  --- Begin raw Rust emulator output ---", _Ansi.MAGENTA))
    result = subprocess.run(cmd, cwd=str(emulator_dir), env=env, text=True, capture_output=True)
    if result.stdout:
        print(_color("  [stdout]", _Ansi.BLUE))
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(_color("  [stderr]", _Ansi.YELLOW if result.returncode == 0 else _Ansi.RED))
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if not result.stdout and not result.stderr:
        print(_color("  (no emulator text output; this is normal when --quiet is effective)", _Ansi.DIM))
    print(_color("  --- End raw Rust emulator output ---", _Ansi.MAGENTA))

    if result.returncode != 0:
        raise RuntimeError(f"Transactional emulator failed (exit code {result.returncode})")
    print(_color("  Emulator finished successfully with exit code 0.", _Ansi.GREEN))


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

    print("  Comparison inputs:")
    print(f"    Emulator VRAM dump: {vram_file}")
    print(f"    Golden reference:   {golden_file}")
    print(f"    Compare parameters: {params_file}")

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
        op_name: Operator name used in pass/fail messages.
        mlen: Matrix tile length synced to plena_settings.toml before running.
        blen: Batch tile length synced to plena_settings.toml before running.
    """
    _section("=== Transactional Emulator Verification ===", _Ansi.CYAN)
    print(f"{_color('Operator label:', _Ansi.BOLD)} {op_name}")
    print(f"{_color('Build directory:', _Ansi.BOLD)} {build_dir}")
    print("This helper checks the generated ISA by running the Rust emulator,")
    print("then comparing emulator VRAM output against the CPU/PyTorch golden result.")

    _section("[1/3] Sync simulator hardware config", _Ansi.CYAN)
    print(f"  Setting VLEN={mlen}, MLEN={mlen}, BLEN={blen} in plena_settings.toml")
    print("  Note: VLEN is forced to match MLEN so row-address alignment checks pass.")
    # VLEN must equal mlen so the emulator's row-address alignment check passes.
    update_plena_config(vlen=mlen, mlen=mlen, blen=blen, verbose=False)

    _section("[2/3] Run Rust transactional emulator", _Ansi.CYAN)
    print("  The emulator consumes the generated memory images and writes vram_dump.bin.")
    run_emulator(build_dir)

    _section("[3/3] Compare emulator output vs golden", _Ansi.CYAN)
    print("  The comparison reads selected rows from vram_dump.bin and aligns them")
    print("  with values parsed from golden_result.txt.")
    results, params = compare_emulator_output(build_dir)
    print_comparison_results(results, verbose=True, comparison_params=params)

    if results["allclose_pass"]:
        print(_color(f"\n[PASS] {op_name}: ISA generated and emulator output matches golden within tolerance.", _Ansi.GREEN))
    else:
        print(_color(f"\n[FAIL] {op_name}: emulator numerical check failed.", _Ansi.RED))
        sys.exit(1)
