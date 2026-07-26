"""Extract and apply ASAP7 SRAM macro read/write internal-power tables."""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections.abc import Iterable
from functools import lru_cache
from statistics import fmean
from typing import Any

POWER_DIR = Path(__file__).resolve().parent
REPO_ROOT = POWER_DIR.parents[1]
DEFAULT_LIB_DIR = REPO_ROOT / "Workspace/external/asap7_sram_0p0/generated/LIB"
DEFAULT_CATALOG = POWER_DIR / "calibration/sram_energy_asap7_v1.json"


def _balanced_block(text: str, start: int) -> str:
    """Return a Liberty block beginning at ``start`` including its braces."""

    opening = text.find("{", start)
    if opening < 0:
        raise ValueError("Liberty block has no opening brace")
    depth = 0
    quoted = False
    escaped = False
    for index in range(opening, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("unterminated Liberty block")


def _blocks(text: str, pattern: str) -> Iterable[str]:
    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        yield _balanced_block(text, match.start())


def _table_numbers(block: str, table: str) -> list[float]:
    values: list[float] = []
    for table_block in _blocks(block, rf"\b{re.escape(table)}\s*\([^)]*\)\s*\{{"):
        for match in re.finditer(r"\bvalues\s*\((.*?)\)\s*;", table_block, re.DOTALL):
            values.extend(
                float(item)
                for item in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))
            )
    return values


def _clock_access_energy(clock_pin: str, condition: str) -> tuple[float, dict[str, Any]]:
    """Read one clock-edge access energy from VDD internal-power entries.

    The generated libraries use derived dynamic power units of mW and time in
    ns.  Summing rise and fall power over a 1 ns clock period therefore has the
    same numeric value in pJ.  Only VDD entries are used; VSS entries describe
    the alternate supply rail and must not be added a second time.
    """

    selected: list[dict[str, Any]] = []
    wanted = "write" if condition == "write" else "!write"
    for block in _blocks(clock_pin, r"\binternal_power\s*\([^)]*\)\s*\{"):
        when_match = re.search(r'\bwhen\s*:\s*"([^"]+)"', block)
        pg_match = re.search(r"\brelated_pg_pin\s*:\s*(\w+)", block)
        if when_match is None or when_match.group(1).replace(" ", "") != wanted:
            continue
        if pg_match is None or pg_match.group(1) != "VDD":
            continue
        rise = _table_numbers(block, "rise_power")
        fall = _table_numbers(block, "fall_power")
        rise_nominal = fmean(rise) if rise else 0.0
        fall_nominal = fmean(fall) if fall else 0.0
        selected.append(
            {
                "condition": wanted,
                "related_pg_pin": "VDD",
                "rise_power_mw": rise_nominal,
                "fall_power_mw": fall_nominal,
            }
        )
    if not selected:
        raise ValueError(f"clk internal_power has no VDD condition {wanted!r}")
    energy_pj = sum(row["rise_power_mw"] + row["fall_power_mw"] for row in selected)
    return energy_pj, {"entries": selected, "clock_period_ns": 1.0}


def parse_asap7_sram_lib(path: str | Path) -> dict[str, Any]:
    """Extract nominal read/write energy and corner metadata from one macro."""

    path = Path(path)
    text = path.read_text(errors="replace")
    macro_match = re.search(r"\bcell\s*\(\s*([^\s)]+)\s*\)", text)
    geometry = re.search(r"srambank_(\d+)x(\d+)x(\d+)_", path.stem)
    if macro_match is None or geometry is None:
        raise ValueError(f"cannot identify SRAM macro geometry in {path}")
    rows, banks, width = map(int, geometry.groups())
    pin_match = re.search(r"\bpin\s*\(\s*clk\s*\)\s*\{", text)
    if pin_match is None:
        raise ValueError(f"{path} has no clk pin")
    clock_pin = _balanced_block(text, pin_match.start())
    read_energy, read_source = _clock_access_energy(clock_pin, "read")
    write_energy, write_source = _clock_access_energy(clock_pin, "write")
    try:
        source_path = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        source_path = str(path)
    return {
        "macro": macro_match.group(1),
        "rows": rows,
        "banks": banks,
        "depth": rows * banks,
        "width": width,
        "bits": rows * banks * width,
        "read_energy_pj": read_energy,
        "write_energy_pj": write_energy,
        "cell_leakage_power": 0.0,
        "sram_leakage_status": "unavailable",
        "nom_voltage_v": float(re.search(r"\bnom_voltage\s*:\s*([0-9.]+)", text).group(1)),
        "nom_temperature_c": float(re.search(r"\bnom_temperature\s*:\s*([0-9.]+)", text).group(1)),
        "lib_path": source_path,
        "extraction": {
            "semantics": "VDD conditional clk internal-power, rise+fall over 1ns",
            "read": read_source,
            "write": write_source,
        },
    }


def build_sram_energy_catalog(
    lib_dir: str | Path = DEFAULT_LIB_DIR,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Build the compact energy table used by the power proxy."""

    lib_dir = Path(lib_dir)
    macros = [parse_asap7_sram_lib(path) for path in sorted(lib_dir.glob("*.lib"))]
    if len(macros) != 36:
        raise ValueError(f"expected 36 ASAP7 SRAM Liberty files, found {len(macros)}")
    payload = {
        "schema_version": 1,
        "model": "asap7_sram_liberty_internal_power_v1",
        "corner": {"process": "TT", "voltage_v": 0.7, "temperature_c": 25.0},
        "energy_unit": "pJ/access",
        "macro_count": len(macros),
        "sram_leakage_status": "unavailable",
        "macros": macros,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


@lru_cache(maxsize=8)
def _load_sram_energy_catalog(selected_text: str) -> dict[str, Any]:
    selected = Path(selected_text)
    if selected.exists():
        return json.loads(selected.read_text())
    return build_sram_energy_catalog()


def load_sram_energy_catalog(path: str | Path | None = None) -> dict[str, Any]:
    """Load one immutable catalog per path for low-overhead DSE evaluation."""

    selected = Path(path or DEFAULT_CATALOG).resolve()
    return _load_sram_energy_catalog(str(selected))


def macro_energy_lookup(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["macro"]): dict(row) for row in catalog["macros"]}


__all__ = [
    "build_sram_energy_catalog",
    "load_sram_energy_catalog",
    "macro_energy_lookup",
    "parse_asap7_sram_lib",
]
