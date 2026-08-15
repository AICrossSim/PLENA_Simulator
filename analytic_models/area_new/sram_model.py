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
    *,
    sram_port_model: str = "replicated-single-port",
) -> tuple[float, dict[str, Any]]:
    """Estimate SRAM area by tiling ASAP7 single-port macros.

    Every candidate macro is tiled independently in depth and width. The
    historical model physically replicates single-port macros. The
    ``ideal-dual-port`` architectural sensitivity keeps the same logical port
    count but charges one macro copy and no dual-port peripheral overhead.
    """
    if not macro_table:
        raise ValueError("empty SRAM macro table")
    best: tuple[float, dict[str, Any]] | None = None
    if sram_port_model not in {"replicated-single-port", "ideal-dual-port"}:
        raise ValueError(f"unsupported SRAM port model {sram_port_model!r}")
    logical_ports = max(1, int(ports))
    port_copies = (
        logical_ports
        if sram_port_model == "replicated-single-port"
        else 1
    )
    for macro in macro_table:
        depth_tiles = math.ceil(depth / macro["depth"])
        width_tiles = math.ceil(width / macro["width"])
        tile_count = depth_tiles * width_tiles * port_copies
        area = tile_count * macro["area_um2"]
        detail = {
            "macro": macro["macro"],
            "macro_depth": macro["depth"],
            "macro_width": macro["width"],
            "macro_area_um2": macro["area_um2"],
            "macro_area_per_bit_um2": macro["area_per_bit_um2"],
            "depth_tiles": depth_tiles,
            "width_tiles": width_tiles,
            "logical_ports": logical_ports,
            "port_copies": port_copies,
            "port_area_multiplier": port_copies,
            "sram_port_model": sram_port_model,
            "tile_count": tile_count,
            "covered_depth": depth_tiles * macro["depth"],
            "covered_width": width_tiles * macro["width"],
            "covered_bits": depth_tiles * macro["depth"] * width_tiles * macro["width"] * port_copies,
        }
        if best is None or area < best[0]:
            best = (area, detail)
    assert best is not None
    return best


def _distributed_bank_depths(depth: int, bank_count: int) -> tuple[int, ...]:
    """Distribute logical rows across statically interleaved physical banks."""

    if depth <= 0:
        raise ValueError(f"SRAM depth must be positive, got {depth}")
    if bank_count <= 0:
        raise ValueError(f"SRAM bank count must be positive, got {bank_count}")
    base, remainder = divmod(depth, bank_count)
    return tuple(base + (1 if bank < remainder else 0) for bank in range(bank_count))


def _banked_macro_tiling_area(
    *,
    logical_depth: int,
    logical_width: int,
    physical_bank_count: int,
    physical_bank_width: int,
    ports: int,
    macro_table: list[dict[str, Any]],
    sram_port_model: str,
) -> tuple[float, dict[str, Any]]:
    """Tile a row-interleaved SRAM as independent, non-replicated banks.

    Banking distributes logical rows; it does not create copies of the stored
    payload. Each bank is tiled independently because macro depth rounding is
    a physical cost of exposing more simultaneous row accesses.
    """

    bank_depths = _distributed_bank_depths(logical_depth, physical_bank_count)
    bank_details: list[dict[str, Any]] = []
    total_area = 0.0
    covered_capacity_bits = 0
    covered_bits_with_port_copies = 0
    total_tiles = 0
    for bank, bank_depth in enumerate(bank_depths):
        if bank_depth == 0:
            bank_details.append(
                {
                    "bank": bank,
                    "logical_depth": 0,
                    "logical_width": physical_bank_width,
                    "area_um2": 0.0,
                    "unused_empty_bank": True,
                }
            )
            continue
        area, detail = _macro_tiling_area(
            depth=bank_depth,
            width=physical_bank_width,
            ports=ports,
            macro_table=macro_table,
            sram_port_model=sram_port_model,
        )
        detail = dict(detail)
        detail.update(
            {
                "bank": bank,
                "logical_depth": bank_depth,
                "logical_width": physical_bank_width,
                "logical_bits": bank_depth * physical_bank_width,
                "area_um2": area,
            }
        )
        bank_details.append(detail)
        total_area += area
        total_tiles += int(detail["tile_count"])
        covered_capacity_bits += int(detail["covered_depth"]) * int(detail["covered_width"])
        covered_bits_with_port_copies += int(detail["covered_bits"])

    r1_area, r1_detail = _macro_tiling_area(
        depth=logical_depth,
        width=logical_width,
        ports=ports,
        macro_table=macro_table,
        sram_port_model=sram_port_model,
    )
    logical_bits = logical_depth * logical_width
    common_macros = {
        detail.get("macro")
        for detail in bank_details
        if not detail.get("unused_empty_bank", False)
    }
    port_copies = (
        max(1, int(ports))
        if sram_port_model == "replicated-single-port"
        else 1
    )
    return total_area, {
        "macro": next(iter(common_macros)) if len(common_macros) == 1 else "mixed",
        "logical_depth": logical_depth,
        "logical_width": logical_width,
        "logical_bits": logical_bits,
        "physical_bank_count": physical_bank_count,
        "physical_bank_width": physical_bank_width,
        "physical_bank_depths": list(bank_depths),
        "storage_replication_factor": 1,
        "logical_ports_per_bank": max(1, int(ports)),
        "logical_ports": max(1, int(ports)),
        "port_copies": port_copies,
        "port_area_multiplier": port_copies,
        "sram_port_model": sram_port_model,
        "tile_count": total_tiles,
        "covered_capacity_bits": covered_capacity_bits,
        "covered_bits": covered_bits_with_port_copies,
        "macro_rounding_overhead_bits": covered_capacity_bits - logical_bits,
        "macro_rounding_overhead_pct": (
            100.0 * (covered_capacity_bits - logical_bits) / logical_bits
            if logical_bits
            else 0.0
        ),
        "banked_area_um2": total_area,
        "r1_area_um2": r1_area,
        "banking_area_delta_um2": total_area - r1_area,
        "r1_tiling": r1_detail,
        "banks": bank_details,
    }


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
    row_banks = int(
        config.get(
            "VECTOR_SRAM_ROW_BANKS",
            config.get("SOFTMAX_ROW_LANES", 1),
        )
    )
    if row_banks not in {1, 2, 4, 8, 16, 32}:
        raise ValueError(f"unsupported VECTOR_SRAM_ROW_BANKS={row_banks}")
    return {
        "mode": act.kind.lower(),
        "depth": depth,
        "width": width,
        "banks": 3,
        "ports": 2,
        "physical_bank_count": row_banks,
        "physical_bank_width": width,
        "banking_semantics": "physical_row_modulo_softmax_row_lanes",
        "storage_replication_factor": 1,
        "row_bank_fidelity": (
            "rtl_v6_calibrated_tier"
            if row_banks <= 8
            else "structural_extrapolation_not_isa_encodable"
        ),
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


def _softmax_state_features(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the dedicated banked m/l state-store geometry for rtl-v6."""

    entries = int(config.get("SOFTMAX_STATE_BANK_ENTRIES", 0))
    if entries <= 0:
        return None
    row_lanes = int(config.get("SOFTMAX_ROW_LANES", 1))
    if row_lanes not in {1, 2, 4, 8, 16, 32}:
        raise ValueError(f"unsupported SOFTMAX_ROW_LANES={row_lanes}")
    fp_width = (
        _fp_width(config, prefix="S_")
        if "S_FP_EXP_WIDTH" in config or "S_FP_MANT_WIDTH" in config
        else _fp_width(config)
    )
    return {
        "depth": math.ceil(entries / row_lanes),
        "width": row_lanes * (2 * fp_width + 1),
        "banks": row_lanes,
        "ports": 2,
        "physical_bank_count": row_lanes,
        "physical_bank_width": 2 * fp_width + 1,
        "logical_depth": entries,
        "logical_width": 2 * fp_width + 1,
        "entries": entries,
        "entry_width": 2 * fp_width + 1,
        "row_lanes": row_lanes,
        "storage_semantics": "banked_m_l_plus_resettable_valid_bitmap",
        "row_bank_fidelity": (
            "rtl_v6_calibrated_tier"
            if row_lanes <= 8
            else "structural_extrapolation_not_isa_encodable"
        ),
    }


def _softmax_transient_features(
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return the statistic and factor stores used by the rtl-v6 row engine."""

    state = _softmax_state_features(config)
    if state is None:
        return {}
    entries = int(state["entries"])
    row_lanes = int(state["row_lanes"])
    fp_width = (int(state["entry_width"]) - 1) // 2
    value = {
        "depth": math.ceil(entries / row_lanes),
        "width": row_lanes * (fp_width + 1),
        "banks": row_lanes,
        "ports": 2,
        "physical_bank_count": row_lanes,
        "physical_bank_width": fp_width + 1,
        "logical_depth": entries,
        "logical_width": fp_width + 1,
        "entries": entries,
        "entry_width": fp_width + 1,
        "row_lanes": row_lanes,
        "storage_semantics": "banked_fp_value_plus_valid_bitmap",
    }
    return {
        "softmax_statistic": dict(value),
        "softmax_factor": dict(value),
    }


def estimate_sram_area(
    config: dict[str, Any],
    *,
    coefficients_path: str | Path | None = None,
    macro_table_path: str | Path | None = None,
    use_macro_table: bool | None = None,
    sram_port_model: str = "replicated-single-port",
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
        sram_port_model: Either the historical physical replication of
            single-port macros or an ideal dual-port architectural assumption.

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
    softmax_state = _softmax_state_features(config)
    softmax_transients = _softmax_transient_features(config)

    if sram_port_model not in {"replicated-single-port", "ideal-dual-port"}:
        raise ValueError(f"unsupported SRAM port model {sram_port_model!r}")

    features = {
        "matrix": matrix,
        "vector": vector,
        "scalar_int": scalar_int,
        "scalar_fp": scalar_fp,
    }
    if softmax_state is not None:
        features["softmax_state"] = softmax_state
        features.update(softmax_transients)

    def evaluate_port_model(
        port_model: str,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        details: dict[str, Any] = {}
        areas: dict[str, float] = {}
        if macro_table:
            for name, values in features.items():
                physical_bank_count = int(values.get("physical_bank_count", 1))
                if physical_bank_count > 1:
                    areas[name], details[name] = _banked_macro_tiling_area(
                        logical_depth=int(values.get("logical_depth", values["depth"])),
                        logical_width=int(values.get("logical_width", values["width"])),
                        physical_bank_count=physical_bank_count,
                        physical_bank_width=int(values.get("physical_bank_width", values["width"])),
                        ports=int(values["ports"]),
                        macro_table=macro_table,
                        sram_port_model=port_model,
                    )
                else:
                    areas[name], details[name] = _macro_tiling_area(
                        **{key: values[key] for key in ["depth", "width", "ports"]},
                        macro_table=macro_table,
                        sram_port_model=port_model,
                    )
                    logical_bits = int(values["depth"]) * int(values["width"])
                    details[name].update(
                        {
                            "logical_depth": int(values["depth"]),
                            "logical_width": int(values["width"]),
                            "logical_bits": logical_bits,
                            "physical_bank_count": 1,
                            "physical_bank_width": int(values["width"]),
                            "physical_bank_depths": [int(values["depth"])],
                            "storage_replication_factor": 1,
                            "covered_capacity_bits": (
                                int(details[name]["covered_depth"])
                                * int(details[name]["covered_width"])
                            ),
                            "macro_rounding_overhead_bits": (
                                int(details[name]["covered_depth"])
                                * int(details[name]["covered_width"])
                                - logical_bits
                            ),
                            "banked_area_um2": areas[name],
                            "r1_area_um2": areas[name],
                            "banking_area_delta_um2": 0.0,
                        }
                    )
        else:
            for name, values in features.items():
                effective_ports = (
                    values["ports"]
                    if port_model == "replicated-single-port"
                    else 1
                )
                coeff_key = (
                    "scalar"
                    if name.startswith("scalar_") or name.startswith("softmax_")
                    else name
                )
                physical_bank_count = int(values.get("physical_bank_count", 1))
                logical_depth = int(values.get("logical_depth", values["depth"]))
                logical_width = int(values.get("logical_width", values["width"]))
                physical_bank_width = int(values.get("physical_bank_width", values["width"]))
                if physical_bank_count > 1:
                    bank_depths = _distributed_bank_depths(logical_depth, physical_bank_count)
                    areas[name] = sum(
                        _generic_area(
                            depth=bank_depth,
                            width=physical_bank_width,
                            banks=1,
                            ports=effective_ports,
                            coeffs=coeffs[coeff_key],
                        )
                        for bank_depth in bank_depths
                        if bank_depth > 0
                    )
                    r1_area = _generic_area(
                        depth=logical_depth,
                        width=logical_width,
                        banks=1,
                        ports=effective_ports,
                        coeffs=coeffs[coeff_key],
                    )
                else:
                    bank_depths = (int(values["depth"]),)
                    areas[name] = _generic_area(
                        depth=values["depth"],
                        width=values["width"],
                        banks=values["banks"],
                        ports=effective_ports,
                        coeffs=coeffs[coeff_key],
                    )
                    r1_area = areas[name]
                details[name] = {
                    "logical_ports": values["ports"],
                    "port_copies": effective_ports,
                    "port_area_multiplier": effective_ports,
                    "sram_port_model": port_model,
                    "logical_depth": logical_depth,
                    "logical_width": logical_width,
                    "logical_bits": logical_depth * logical_width,
                    "physical_bank_count": physical_bank_count,
                    "physical_bank_width": physical_bank_width,
                    "physical_bank_depths": list(bank_depths),
                    "storage_replication_factor": 1,
                    "banked_area_um2": areas[name],
                    "r1_area_um2": r1_area,
                    "banking_area_delta_um2": areas[name] - r1_area,
                }
        return areas, details

    selected_areas, macro_details = evaluate_port_model(sram_port_model)
    ideal_areas, ideal_details = evaluate_port_model("ideal-dual-port")
    replicated_areas, replicated_details = evaluate_port_model("replicated-single-port")
    matrix_area = selected_areas["matrix"]
    vector_area = selected_areas["vector"]
    scalar_int_area = selected_areas["scalar_int"]
    scalar_fp_area = selected_areas["scalar_fp"]
    softmax_state_area = selected_areas.get("softmax_state", 0.0)
    softmax_statistic_area = selected_areas.get("softmax_statistic", 0.0)
    softmax_factor_area = selected_areas.get("softmax_factor", 0.0)
    if macro_table:
        model = "asap7_sram_macro_tiling"
    else:
        model = "fitted_linear_coefficients"

    breakdown = {
        "MatrixSRAM": matrix_area,
        "VectorSRAM": vector_area,
        "ScalarIntSRAM": scalar_int_area,
        "ScalarFPSRAM": scalar_fp_area,
    }
    if softmax_state is not None:
        breakdown["SoftmaxStateBank"] = softmax_state_area
        breakdown["SoftmaxStatisticBank"] = softmax_statistic_area
        breakdown["SoftmaxFactorBank"] = softmax_factor_area
    ideal_total = sum(ideal_areas.values())
    replicated_total = sum(replicated_areas.values())
    return {
        "area": sum(breakdown.values()),
        "area_sram_proxy": sum(breakdown.values()),
        "area_sram_breakdown": breakdown,
        "area_sram_inputs": {
            "matrix": matrix,
            "vector": vector,
            "scalar_int": scalar_int,
            "scalar_fp": scalar_fp,
            **({"softmax_state": softmax_state} if softmax_state is not None else {}),
            **softmax_transients,
        },
        "area_sram_model": model,
        "sram_port_model": (
            "ideal_dual_port_architectural_assumption"
            if sram_port_model == "ideal-dual-port"
            else "replicated_single_port_macros"
        ),
        "selected_sram_area_um2": sum(breakdown.values()),
        "ideal_dual_port_sram_area_um2": ideal_total,
        "replicated_single_port_sram_area_um2": replicated_total,
        "dual_port_area_savings_um2": replicated_total - ideal_total,
        "dual_port_area_savings_pct": (
            100.0 * (replicated_total - ideal_total) / replicated_total
            if replicated_total
            else 0.0
        ),
        "dual_port_overhead_included": False,
        "area_sram_macro_tiling": macro_details,
        "vector_sram_banking": {
            "logical_bits": int(macro_details["vector"].get("logical_bits", vector["depth"] * vector["width"])),
            "physical_bank_count": int(macro_details["vector"].get("physical_bank_count", 1)),
            "physical_bank_depths": list(macro_details["vector"].get("physical_bank_depths", [vector["depth"]])),
            "row_width_bits": int(vector["width"]),
            "covered_capacity_bits": int(macro_details["vector"].get("covered_capacity_bits", 0)),
            "macro_rounding_overhead_bits": int(macro_details["vector"].get("macro_rounding_overhead_bits", 0)),
            "selected_banked_area_um2": float(selected_areas["vector"]),
            "selected_r1_area_um2": float(macro_details["vector"].get("r1_area_um2", selected_areas["vector"])),
            "selected_banking_area_delta_um2": float(macro_details["vector"].get("banking_area_delta_um2", 0.0)),
            "ideal_dual_port_banked_area_um2": float(ideal_areas["vector"]),
            "replicated_single_port_banked_area_um2": float(replicated_areas["vector"]),
            "ideal_dual_port_r1_area_um2": float(ideal_details["vector"].get("r1_area_um2", ideal_areas["vector"])),
            "replicated_single_port_r1_area_um2": float(replicated_details["vector"].get("r1_area_um2", replicated_areas["vector"])),
            "storage_replication_factor": 1,
            "banking_semantics": vector["banking_semantics"],
            "row_bank_fidelity": vector["row_bank_fidelity"],
            "dual_port_overhead_included": False,
        },
        "area_sram_coefficients": coeffs,
    }


__all__ = ["estimate_sram_area"]
