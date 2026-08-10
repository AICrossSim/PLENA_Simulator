"""Precision-aware full-chip and multi-chip area models for PLENA.

Areas are returned in square micrometres. Logic blocks use exact RTL structural
counts with unit areas fitted to retained aggregate DC tables. SRAM uses ASAP7
macro geometry, and link PHYs use an explicitly labelled literature projection.
"""

from __future__ import annotations

from typing import Any, Mapping

from .precision import PrecisionError, derive_compute_sides, parse_precision
from .matrix import (
    estimate_matrix_machine_area,
    matrix_area_from_sides,
    structural_counts,
)
from .vector import estimate_vector_area
from .scalar import estimate_scalar_area, loop_address_generator_counts
from .hbm_interface import estimate_hbm_interface_area
from .top import estimate_top_area
from .sram import estimate_buffer_area, estimate_sram_area
from .link import estimate_link_phy_area
from .hbm_phy import estimate_hbm_phy_area
from .geometry import solve_geometry_for_area
from .evidence import weakest_tier

__all__ = [
    "PrecisionError",
    "derive_compute_sides",
    "parse_precision",
    "estimate_matrix_machine_area",
    "structural_counts",
    "estimate_vector_area",
    "estimate_scalar_area",
    "loop_address_generator_counts",
    "estimate_hbm_interface_area",
    "estimate_top_area",
    "estimate_sram_area",
    "estimate_buffer_area",
    "estimate_link_phy_area",
    "estimate_hbm_phy_area",
    "estimate_area",
    "estimate_system_area",
    "solve_geometry_for_area",
]


def estimate_area(config: Mapping[str, Any], **kwargs) -> dict[str, Any]:
    """Estimate the full decode-chip silicon area.

    The decomposition is

    ``(1 + top_ratio) * (matrix + vector + scalar + hbm_if) + SRAM``.

    ``top_ratio`` is derived from the independently fitted top-level residual,
    rather than assumed. No missing block is silently replaced with zero.
    """

    corner = str(kwargs.pop("corner", "reference"))
    coefficients_path = kwargs.pop("coefficients_path", None)
    macro_table_path = kwargs.pop("macro_table_path", None)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported estimate_area options: {names}")

    logic_options = {"corner": corner, "coefficients_path": coefficients_path}
    matrix = estimate_matrix_machine_area(config, **logic_options)
    vector = estimate_vector_area(config, **logic_options)
    scalar = estimate_scalar_area(config, **logic_options)
    hbm_interface = estimate_hbm_interface_area(config, **logic_options)
    top = estimate_top_area(config, **logic_options)
    sram = estimate_sram_area(config, macro_table_path=macro_table_path)

    logic_blocks = {
        "MatrixMachine": float(matrix["area"]),
        "VectorMachine": float(vector["area"]),
        "ScalarMachine": float(scalar["area"]),
        "HBMInterface": float(hbm_interface["area"]),
    }
    logic_base = sum(logic_blocks.values())
    top_area = float(top["area"])
    top_ratio = top_area / logic_base
    sram_breakdown = {name: float(area) for name, area in sram["breakdown"].items()}
    sram_area = float(sram["area"])
    total = (1.0 + top_ratio) * logic_base + sram_area
    enhancement_area = (
        float(vector.get("enhanced_area", vector["area"]))
        - float(vector.get("calibrated_area", vector["area"]))
        + float(scalar.get("enhanced_area", scalar["area"]))
        - float(scalar.get("calibrated_area", scalar["area"]))
    )
    breakdown = {**logic_blocks, "TopOverhead": top_area, **sram_breakdown}
    block_evidence = {
        "MatrixMachine": matrix["evidence"],
        "VectorMachine": vector["evidence"],
        "ScalarMachine": scalar["evidence"],
        "HBMInterface": hbm_interface["evidence"],
        "TopOverhead": top["evidence"],
        "SRAMMacros": sram["evidence"],
    }
    return {
        "area": total,
        "chip_area": total,
        "calibrated_chip_area": total - enhancement_area,
        "enhanced_chip_area": total,
        "enhancement_area": enhancement_area,
        "area_model": "plena_full_chip_precision_structural",
        "matrix_machine_area": float(matrix["area"]),
        "vector_machine_area": float(vector["area"]),
        "scalar_machine_area": float(scalar["area"]),
        "hbm_interface_area": float(hbm_interface["area"]),
        "top_overhead_area": top_area,
        "logic_area": logic_base + top_area,
        "sram_macro_area": sram_area,
        "top_overhead_ratio": top_ratio,
        "breakdown": breakdown,
        "block_evidence": block_evidence,
        "evidence_tier": weakest_tier(block_evidence.values()),
        "matrix_machine": matrix,
        "vector_machine": vector,
        "scalar_machine": scalar,
        "hbm_interface": hbm_interface,
        "top": top,
        "sram": sram,
    }


def estimate_system_area(
    config: Mapping[str, Any],
    *,
    chip_count: int,
    ports_per_chip: int = 0,
    link_bandwidth_gbps: float = 900.0 * 8.0,
    hbm_interface_units_per_chip: int = 0,
    target_node_nm: float = 7.0,
    **area_options: Any,
) -> dict[str, Any]:
    """Estimate aggregate silicon for chips, HBM PHYs, and C2C PHY ports.

    ``chip_area`` is the per-die total including the chip's HBM PHY, so the
    system total minus ``chip_count`` times ``chip_area`` is exactly the link
    silicon. The HBM PHY term is what makes attached memory bandwidth a
    genuine area trade-off rather than a free configuration choice.
    """

    chips = int(chip_count)
    ports = int(ports_per_chip)
    if chips <= 0 or ports < 0:
        raise ValueError("chip_count must be positive and ports_per_chip non-negative")
    chip = estimate_area(config, **area_options)
    link = estimate_link_phy_area(
        bandwidth_gbps=link_bandwidth_gbps,
        target_node_nm=target_node_nm,
    )
    hbm_phy = estimate_hbm_phy_area(
        hbm_interface_units_per_chip,
        target_node_nm=target_node_nm,
    )
    logic_sram_area = float(chip["area"])
    hbm_phy_area = float(hbm_phy["area"])
    chip_area = logic_sram_area + hbm_phy_area
    link_area = float(link["area"])
    total = chips * (chip_area + ports * link_area)
    evidence_records = [{"tier": chip["evidence_tier"]}, link["evidence"]]
    if hbm_phy_area > 0.0:
        evidence_records.append(hbm_phy["evidence"])
    return {
        "area": total,
        "system_area": total,
        "chip_count": chips,
        "ports_per_chip": ports,
        "chip_area": chip_area,
        "chip_logic_sram_area": logic_sram_area,
        "hbm_phy_area_per_chip": hbm_phy_area,
        "hbm_interface_units_per_chip": int(hbm_interface_units_per_chip),
        "link_phy_area_per_port": link_area,
        "link_phy_area_per_chip": ports * link_area,
        "breakdown": {
            "DecodeChips": chips * logic_sram_area,
            "HBMPhys": chips * hbm_phy_area,
            "LinkPHYs": chips * ports * link_area,
        },
        "chip": chip,
        "hbm_phy": hbm_phy,
        "link_phy": link,
        "evidence_tier": weakest_tier(evidence_records),
    }
