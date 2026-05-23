"""GUI parameter overrides for testbench scripts.

The web GUI ("Transformer Block Test" panel) runs the existing testbench
scripts as subprocesses and passes per-run knobs (batch size, sequence
length, tile dimensions, ...) via ``PLENA_TB_*`` environment variables.

Each script reads a few well-known parameters through :func:`gi`, falling
back to the literal default baked into the script when the variable is unset
or empty. This keeps ``just test-*`` behaviour byte-for-byte identical when
the GUI is not driving the run.
"""

import os

__all__ = ["gi", "gs", "active_overrides"]

_PREFIX = "PLENA_TB_"


def gi(name: str, default: int) -> int:
    """Return int override ``PLENA_TB_<NAME>`` if set/parseable, else ``default``."""
    raw = os.environ.get(_PREFIX + name.upper())
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def gs(name: str, default: str) -> str:
    """Return string override ``PLENA_TB_<NAME>`` if set, else ``default``."""
    raw = os.environ.get(_PREFIX + name.upper())
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def active_overrides() -> dict[str, str]:
    """All currently-set ``PLENA_TB_*`` overrides (for logging/debug)."""
    return {
        k[len(_PREFIX):].lower(): v
        for k, v in os.environ.items()
        if k.startswith(_PREFIX) and v.strip() != ""
    }
