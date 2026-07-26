#!/usr/bin/env python3
# ruff: noqa: E402
"""Extract ASAP7 SRAM read/write energy tables from public Liberty files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytic_models.power.sram_energy import (
    DEFAULT_CATALOG,
    DEFAULT_LIB_DIR,
    build_sram_energy_catalog,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib-dir", type=Path, default=DEFAULT_LIB_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    result = build_sram_energy_catalog(args.lib_dir, output=args.output)
    print(f"Wrote {result['macro_count']} SRAM macro entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
