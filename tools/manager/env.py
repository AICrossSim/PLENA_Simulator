"""Run-environment constants for the manager.

The compile step runs in a subprocess: the compiler (tvm/tilelang) lives in
``.venv`` and torch needs the nix gcc libstdc++ on LD_LIBRARY_PATH. These mirror
exactly what the existing test_helper subprocess + run_all_tvm_tests.sh use.
"""

from __future__ import annotations

import os
from pathlib import Path

# tools/manager/env.py -> PLENA_Simulator/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
COMPILER_DIR = PROJECT_ROOT / "compiler"
TOOLS_DIR = PROJECT_ROOT / "tools"
DEFAULT_TOML = PROJECT_ROOT / "plena_settings.toml"

# torch's C extensions need this (same path the justfile / stepwise driver use).
# Allow override from the ambient env if the caller already set it.
NIX_GCC_LIB = "/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib"

# managerbuild/ layout (MANAGER_DESIGN.md §2.6): hbm_bin/ persistent,
# ir/ auto-refreshed per kernel call.
MANAGERBUILD = PROJECT_ROOT / "managerbuild"
HBM_DIR = MANAGERBUILD / "hbm_bin"
IR_DIR = MANAGERBUILD / "ir"
HBM_BIN = HBM_DIR / "hbm_for_behave_sim.bin"
FP_SRAM_BIN = HBM_DIR / "fp_sram.bin"
LAYOUT_JSON = HBM_DIR / "layout.json"


def compile_env() -> dict:
    """Environment dict for the compile subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(COMPILER_DIR)
    env.setdefault("LD_LIBRARY_PATH", NIX_GCC_LIB)
    return env


# The release emulator binary links libtorch from the venv's prebuilt torch
# (.envrc: TORCH_LIB_PATH = torch.utils.cmake_prefix_path + '/lib'). Resolve it
# the same way so libtorch_cpu.so is found at runtime.
VENV_TORCH_LIB = PROJECT_ROOT / ".venv" / "lib" / "python3.12" / "site-packages" / "torch" / "lib"


def emulator_env() -> dict:
    """Environment for the emulator subprocess: nix gcc libstdc++ + venv
    libtorch on LD_LIBRARY_PATH."""
    env = os.environ.copy()
    parts = [str(VENV_TORCH_LIB), NIX_GCC_LIB]
    existing = env.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    env["LD_LIBRARY_PATH"] = ":".join(parts)
    return env


def ir_dir_for(kernel_name: str) -> Path:
    return IR_DIR / kernel_name
