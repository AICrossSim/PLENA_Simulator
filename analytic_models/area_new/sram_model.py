"""Estimate PLENA on-chip SRAM area from an ASAP7 macro catalogue.

Behavioral SRAM arrays are black boxes in the normal full-chip DC flow, so
their bitcell area cannot be inferred from the synthesized wrapper logic. The
default model therefore tiles real single-port macros from the OpenROAD ASAP7
LIB/LEF collateral and chooses the minimum-area legal tiling for each logical
memory.

The estimate includes Matrix, Vector, scalar integer, and scalar FP SRAMs. A
multi-port logical memory is conservatively implemented by full macro copies.
This is a DSE floorplanning proxy in um^2, not a foundry SRAM compiler result.
The older coefficient equation remains available only as an explicit fallback.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from .precision import PrecisionError, derive_compute_sides, parse_precision

CALIBRATION_DIR = Path(__file__).with_name("calibration")

DEFAULT_COEFFICIENTS = {
    "matrix": {"a": 0.020, "b": 1.0, "c": 0.4, "d": 4.0, "e": 0.15, "f": 20.0},
    "vector": {"a": 0.025, "b": 1.2, "c": 0.5, "d": 5.0, "e": 0.25, "f": 30.0},
    "scalar": {"a": 0.018, "b": 0.8, "c": 0.3, "d": 2.0, "e": 0.10, "f": 10.0},
}

_FP_RE = re.compile(r"^FP_?E(\d+)M(\d+)$", re.IGNORECASE)
DEFAULT_MACRO_TABLE = CALIBRATION_DIR / "asap7_sram_macro_table.csv"


def _load_macro_table(explicit_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load normalized macro dimensions and LEF area from compact CSV data."""
    path = explicit_path or os.environ.get("PLENA_AREA_NEW_SRAM_MACRO_TABLE")
    if path is None:
        path = DEFAULT_MACRO_TABLE
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "macro": row["macro"],
                    "depth": int(row["depth"]),
                    "width": int(row["width"]),
                    "bits": int(row["bits"]),
                    "area_um2": float(row["area_um2"]),
                    "area_per_bit_um2": float(row["area_per_bit_um2"]),
                }
            )
    return rows


def _load_coefficients(explicit_path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load the legacy register-array coefficient fallback."""
    path = explicit_path or os.environ.get("PLENA_AREA_NEW_SRAM_COEFFICIENTS")
    if path is None:
        path = CALIBRATION_DIR / "sram_model_coefficients.json"
    path = Path(path)
    if not path.exists():
        return DEFAULT_COEFFICIENTS
    with path.open() as f:
        raw = json.load(f)
    coeffs = raw.get("coefficients", raw)
    out = {name: dict(DEFAULT_COEFFICIENTS[name]) for name in DEFAULT_COEFFICIENTS}
    for name, values in coeffs.items():
        if name in out and isinstance(values, dict):
            out[name].update({str(k): float(v) for k, v in values.items()})
    return out


def _fp_width(config: dict[str, Any], *, prefix: str = "") -> int:
    """Resolve sign + exponent + mantissa width from DSE or RTL-style keys."""
    exp_key = f"{prefix}FP_EXP_WIDTH" if prefix else "FP_EXP_WIDTH"
    mant_key = f"{prefix}FP_MANT_WIDTH" if prefix else "FP_MANT_WIDTH"
    if exp_key in config and mant_key in config:
        return 1 + int(config[exp_key]) + int(config[mant_key])
    if "FP_SETTING" in config:
        match = _FP_RE.match(str(config["FP_SETTING"]).strip())
        if match:
            return 1 + int(match.group(1)) + int(match.group(2))
    return 1 + int(config.get("FP_EXP_WIDTH", config.get("S_FP_EXP_WIDTH", 5))) + int(
        config.get("FP_MANT_WIDTH", config.get("S_FP_MANT_WIDTH", 6))
    )


def _depth(config: dict[str, Any], *names: str, default: int) -> int:
    """Resolve the first available depth alias used by old and new callers."""
    for name in names:
        if name in config:
            return int(config[name])
    return default


def _generic_area(depth: int, width: int, banks: int, ports: int, coeffs: dict[str, float]) -> float:
    """Evaluate the deprecated linear register-array area equation."""
    return (
        coeffs["a"] * depth * width
        + coeffs["b"] * depth
        + coeffs["c"] * width
        + coeffs["d"] * banks
        + coeffs["e"] * ports * width
        + coeffs["f"]
    )


def _macro_tiling_area(
    depth: int,
    width: int,
    ports: int,
    macro_table: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Estimate SRAM area by tiling ASAP7 single-port macros.

    Every candidate macro is tiled independently in depth and width. The
    number of physical copies is multiplied by ``ports`` because the source
    catalogue contains single-port macros. The minimum-area candidate wins;
    returned details expose capacity over-provisioning for auditability.
    """
    if not macro_table:
        raise ValueError("empty SRAM macro table")
    best: tuple[float, dict[str, Any]] | None = None
    effective_ports = max(1, int(ports))
    for macro in macro_table:
        depth_tiles = math.ceil(depth / macro["depth"])
        width_tiles = math.ceil(width / macro["width"])
        tile_count = depth_tiles * width_tiles * effective_ports
        area = tile_count * macro["area_um2"]
        detail = {
            "macro": macro["macro"],
            "macro_depth": macro["depth"],
            "macro_width": macro["width"],
            "macro_area_um2": macro["area_um2"],
            "macro_area_per_bit_um2": macro["area_per_bit_um2"],
            "depth_tiles": depth_tiles,
            "width_tiles": width_tiles,
            "port_copies": effective_ports,
            "tile_count": tile_count,
            "covered_depth": depth_tiles * macro["depth"],
            "covered_width": width_tiles * macro["width"],
            "covered_bits": depth_tiles * macro["depth"] * width_tiles * macro["width"] * effective_ports,
        }
        if best is None or area < best[0]:
            best = (area, detail)
    assert best is not None
    return best


def _matrix_features(config: dict[str, Any]) -> dict[str, Any]:
    """Derive logical MatrixSRAM geometry from T-side precision and MLEN."""
    sides = derive_compute_sides(
        config["ACT_WIDTH"],
        config["KV_WIDTH"],
        config.get("WEIGHT_WIDTH", "MXINT4"),
        default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)),
    )
    mlen = int(config["MLEN"])
    block_dim = int(config.get("BLOCK_DIM", config.get("BLEN", 4)))
    parallel_dim = int(config.get("PARALLEL_DIM", 1))
    depth = _depth(config, "MATRIX_SRAM_DEPTH", "MATRIX_SRAM_SIZE", default=max(32, 2 * mlen))
    element_width = int(sides["t_width"])
    scale_width = int(sides["scale_width"])
    # Matrix rows carry T-side values and their shared scales. Precision only
    # changes row width; configured SRAM depth remains an architectural knob.
    width = mlen * parallel_dim * (element_width + scale_width)
    banks = 2 * max(1, math.ceil(mlen / max(parallel_dim, 1)))
    return {
        "mode": sides["mode"],
        "depth": depth,
        "width": width,
        "banks": banks,
        "ports": 2,
        "mlen": mlen,
        "block_dim": block_dim,
        "parallel_dim": parallel_dim,
        "element_width": element_width,
        "scale_width": scale_width,
    }


def _vector_features(config: dict[str, Any]) -> dict[str, Any]:
    """Derive VectorSRAM geometry for FP, ACT, KV, and scale payloads."""
    act = parse_precision(config["ACT_WIDTH"], default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)))
    kv = parse_precision(config["KV_WIDTH"], default_scale_width=int(config.get("MX_SCALE_WIDTH", 8)))
    if act.kind != kv.kind:
        raise PrecisionError(f"mixed ACT/KV vector SRAM precision is unsupported: {act.name}, {kv.name}")
    vlen = int(config["VLEN"])
    mlen = int(config.get("MLEN", vlen))
    blen = int(config.get("BLEN", config.get("BLOCK_DIM", 4)))
    block_dim = int(config.get("BLOCK_DIM", blen))
    depth = _depth(config, "VECTOR_SRAM_DEPTH", "VECTOR_SRAM_SIZE", default=max(32, 2 * 128 + math.ceil(mlen / vlen)))
    fp_width = _fp_width(config)
    act_width = act.element_width
    kv_width = kv.element_width
    scale_width = max(act.scale_width, kv.scale_width, int(config.get("MX_SCALE_WIDTH", 8)))
    scale_blocks = max(1, math.ceil(vlen / max(block_dim, 1)))
    # A vector row must accommodate all three payload classes used by the RTL.
    # Two scale streams account for block-scaled activation/KV metadata.
    width = vlen * (fp_width + act_width + kv_width) + 2 * scale_blocks * scale_width
    return {
        "mode": act.kind.lower(),
        "depth": depth,
        "width": width,
        "banks": 3,
        "ports": 2,
        "vlen": vlen,
        "mlen": mlen,
        "blen": blen,
        "block_dim": block_dim,
        "fp_width": fp_width,
        "act_width": act_width,
        "kv_width": kv_width,
        "scale_width": scale_width,
    }


def _scalar_int_features(config: dict[str, Any]) -> dict[str, Any]:
    """Return scalar integer SRAM geometry."""
    depth = _depth(config, "INT_SRAM_DEPTH", default=32)
    width = int(config.get("INT_DATA_WIDTH", 32))
    return {"depth": depth, "width": width, "banks": 1, "ports": 1}


def _scalar_fp_features(config: dict[str, Any]) -> dict[str, Any]:
    """Return scalar floating-point SRAM geometry."""
    depth = _depth(config, "FP_SRAM_DEPTH", default=512)
    width = _fp_width(config, prefix="S_") if "S_FP_EXP_WIDTH" in config or "S_FP_MANT_WIDTH" in config else _fp_width(config)
    return {"depth": depth, "width": width, "banks": 1, "ports": 1}


def estimate_sram_area(
    config: dict[str, Any],
    *,
    coefficients_path: str | Path | None = None,
    macro_table_path: str | Path | None = None,
    use_macro_table: bool | None = None,
) -> dict[str, Any]:
    """Estimate precision-aware SRAM subsystem area in um^2.

    Required config keys are ``ACT_WIDTH``, ``KV_WIDTH``, ``MLEN``, and
    ``VLEN``. Depth aliases from both old and new DSE scripts are accepted.

    Args:
        config: Hardware dimensions, logical SRAM depths, and precision knobs.
        coefficients_path: Optional legacy linear-model coefficients.
        macro_table_path: Optional replacement ASAP7-compatible macro table.
        use_macro_table: Select macro tiling explicitly. ``None`` uses the
            ``PLENA_AREA_NEW_SRAM_MODEL`` environment setting and defaults to
            macro tiling.

    Returns:
        Total SRAM area, per-memory breakdown, derived logical geometries, and
        selected macro tilings. All area values are in um^2.
    """
    if use_macro_table is None:
        use_macro_table = os.environ.get("PLENA_AREA_NEW_SRAM_MODEL", "macro").lower() != "coefficients"
    coeffs = _load_coefficients(coefficients_path)
    macro_table = _load_macro_table(macro_table_path) if use_macro_table else []
    matrix = _matrix_features(config)
    vector = _vector_features(config)
    scalar_int = _scalar_int_features(config)
    scalar_fp = _scalar_fp_features(config)

    macro_details: dict[str, Any] = {}
    if macro_table:
        matrix_area, macro_details["matrix"] = _macro_tiling_area(
            **{k: matrix[k] for k in ["depth", "width", "ports"]}, macro_table=macro_table
        )
        vector_area, macro_details["vector"] = _macro_tiling_area(
            **{k: vector[k] for k in ["depth", "width", "ports"]}, macro_table=macro_table
        )
        scalar_int_area, macro_details["scalar_int"] = _macro_tiling_area(
            **{k: scalar_int[k] for k in ["depth", "width", "ports"]}, macro_table=macro_table
        )
        scalar_fp_area, macro_details["scalar_fp"] = _macro_tiling_area(
            **{k: scalar_fp[k] for k in ["depth", "width", "ports"]}, macro_table=macro_table
        )
        model = "asap7_sram_macro_tiling"
    else:
        matrix_area = _generic_area(
            **{k: matrix[k] for k in ["depth", "width", "banks", "ports"]}, coeffs=coeffs["matrix"]
        )
        vector_area = _generic_area(
            **{k: vector[k] for k in ["depth", "width", "banks", "ports"]}, coeffs=coeffs["vector"]
        )
        scalar_int_area = _generic_area(
            **{k: scalar_int[k] for k in ["depth", "width", "banks", "ports"]}, coeffs=coeffs["scalar"]
        )
        scalar_fp_area = _generic_area(
            **{k: scalar_fp[k] for k in ["depth", "width", "banks", "ports"]}, coeffs=coeffs["scalar"]
        )
        model = "fitted_linear_coefficients"

    breakdown = {
        "MatrixSRAM": matrix_area,
        "VectorSRAM": vector_area,
        "ScalarIntSRAM": scalar_int_area,
        "ScalarFPSRAM": scalar_fp_area,
    }
    return {
        "area": sum(breakdown.values()),
        "area_sram_proxy": sum(breakdown.values()),
        "area_sram_breakdown": breakdown,
        "area_sram_inputs": {
            "matrix": matrix,
            "vector": vector,
            "scalar_int": scalar_int,
            "scalar_fp": scalar_fp,
        },
        "area_sram_model": model,
        "area_sram_macro_tiling": macro_details,
        "area_sram_coefficients": coeffs,
    }


__all__ = ["estimate_sram_area"]
