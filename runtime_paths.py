"""Resolve repository paths for the simulator, testbenches and analytic models.

This module sits at the repository root, so the root is its own parent. The
``PLENA_SIMULATOR_PATH`` and ``PLENA_SETTINGS_TOML`` environment variables
override the derived locations when the tree is checked out elsewhere or a
testbench needs alternative machine settings.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent


def simulator_root() -> Path:
    """Return the PLENA_Simulator checkout containing the analytic models."""

    value = os.environ.get("PLENA_SIMULATOR_PATH")
    root = Path(value).expanduser().resolve() if value else _DEFAULT_ROOT
    if not (root / "analytic_models").is_dir():
        raise RuntimeError(
            f"{root} is not a PLENA_Simulator checkout (no analytic_models/); "
            "set PLENA_SIMULATOR_PATH to the correct tree"
        )
    return root


def sibling_repository(name: str) -> Path:
    """Return a repository checked out alongside the simulator, e.g. PLENA_RTL."""

    return simulator_root().parent / name


def settings_path() -> Path:
    """Return the machine-configuration TOML used by the emulator and models."""

    value = os.environ.get("PLENA_SETTINGS_TOML")
    return (
        Path(value).expanduser().resolve()
        if value
        else simulator_root() / "plena_settings.toml"
    )
