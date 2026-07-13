#!/usr/bin/env python3
"""Build the committed ASAP7 SRAM macro area catalogue.

The source repository is too large and external to vendor into this project.
This utility extracts only logical depth/width and physical area from each
macro's Liberty and LEF files. The resulting CSV is sufficient for deterministic
macro tiling in :mod:`analytic_models.area_new.sram_model` without running DC.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


NAME_RE = re.compile(r"^srambank_(?P<rows>\d+)x(?P<banks>\d+)x(?P<width>\d+)_6t122$")
AREA_RE = re.compile(r"\barea\s*:\s*([0-9.]+)\s*;")
SIZE_RE = re.compile(r"\bSIZE\s+([0-9.]+)\s+BY\s+([0-9.]+)\s*;")

FIELDS = [
    "macro",
    "rows",
    "banks",
    "depth",
    "width",
    "bits",
    "area_um2",
    "lef_width_um",
    "lef_height_um",
    "lef_area_um2",
    "area_per_bit_um2",
    "lib_path",
    "lef_path",
]


def parse_name(name: str) -> dict[str, int]:
    """Decode rows, banks, width, depth, and capacity from a macro name."""
    match = NAME_RE.match(name)
    if not match:
        raise ValueError(f"unrecognized SRAM macro name: {name}")
    rows = int(match.group("rows"))
    banks = int(match.group("banks"))
    width = int(match.group("width"))
    return {
        "rows": rows,
        "banks": banks,
        "depth": rows * banks,
        "width": width,
        "bits": rows * banks * width,
    }


def parse_lib_area(path: Path) -> tuple[str, float]:
    """Read the Liberty cell name and area in um^2."""
    text = path.read_text(errors="ignore")
    cell_match = re.search(r"\bcell\s*\(\s*([^)]+)\s*\)", text)
    if not cell_match:
        raise ValueError(f"cell name not found in {path}")
    area_match = AREA_RE.search(text[cell_match.start() :])
    if not area_match:
        raise ValueError(f"cell area not found in {path}")
    return cell_match.group(1).strip(), float(area_match.group(1))


def parse_lef_size(path: Path) -> tuple[float | None, float | None]:
    """Read LEF width/height in um, returning missing values as ``None``."""
    if not path.exists():
        return None, None
    match = SIZE_RE.search(path.read_text(errors="ignore"))
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def build_table(source: Path) -> list[dict[str, object]]:
    """Extract all recognized SRAM macros from an ASAP7 collateral checkout."""
    lib_dir = source / "generated" / "LIB"
    lef_dir = source / "generated" / "LEF"
    rows: list[dict[str, object]] = []
    for lib_path in sorted(lib_dir.glob("srambank_*.lib")):
        macro, area = parse_lib_area(lib_path)
        dims = parse_name(macro)
        lef_path = lef_dir / f"{macro}.lef"
        lef_w, lef_h = parse_lef_size(lef_path)
        lef_area = None if lef_w is None or lef_h is None else lef_w * lef_h
        rows.append(
            {
                "macro": macro,
                **dims,
                "area_um2": area,
                "lef_width_um": "" if lef_w is None else lef_w,
                "lef_height_um": "" if lef_h is None else lef_h,
                "lef_area_um2": "" if lef_area is None else lef_area,
                "area_per_bit_um2": area / dims["bits"],
                "lib_path": str(lib_path),
                "lef_path": str(lef_path) if lef_path.exists() else "",
            }
        )
    if not rows:
        raise FileNotFoundError(f"no srambank LIB files found under {lib_dir}")
    return rows


def write_csv(rows: list[dict[str, object]], out: Path) -> None:
    """Write a stable, reviewable macro table with a fixed column order."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse source checkout and output CSV paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("Workspace/external/asap7_sram_0p0"),
        help="Path to The-OpenROAD-Project/asap7_sram_0p0 checkout",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analytic_models/area_new/calibration/asap7_sram_macro_table.csv"),
    )
    return parser.parse_args()


def main() -> int:
    """Generate the table and return a shell-compatible exit status."""
    args = parse_args()
    rows = build_table(args.source)
    write_csv(rows, args.out)
    print(f"wrote {len(rows)} ASAP7 SRAM macro rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
