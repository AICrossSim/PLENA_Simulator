"""Declared per-channel HBM PHY and beachfront area.

The anchor is 11.0 mm^2 of chip-side PHY, I/O, and beachfront silicon per
1024-bit HBM stack interface at 7 nm. It combines the published die-edge
(shoreline) occupancy of roughly 11 mm per HBM stack interface (Chen et al.,
"Overcoming Design Challenges for High Bandwidth Memory Interface with
CoWoS," EMCSI 2022, doi:10.1109/EMCSI39492.2022.10050234) with a declared
1.0 mm beachfront macro depth. The per-stack figure is divided evenly over
the sixteen 64-bit interface units of a 1024-bit stack, so chip area scales
linearly with the number of interface units a design attaches. This is a
declared structural estimate, not PLENA synthesis data; replacing the
declared depth with a synthesized or vendor PHY floorplan upgrades the tier
without any caller change.
"""

from __future__ import annotations

from typing import Any

from .evidence import STRUCTURAL_ESTIMATE

SOURCE_SHORELINE_MM_PER_STACK = 11.0
SOURCE_DOI = "10.1109/EMCSI39492.2022.10050234"
DECLARED_BEACHFRONT_DEPTH_MM = 1.0
STACK_INTERFACE_UNITS = 16
ANCHOR_NODE_NM = 7.0
ANCHOR_STACK_AREA_MM2 = SOURCE_SHORELINE_MM_PER_STACK * DECLARED_BEACHFRONT_DEPTH_MM
AREA_MM2_PER_INTERFACE_UNIT = ANCHOR_STACK_AREA_MM2 / STACK_INTERFACE_UNITS


def estimate_hbm_phy_area(
    interface_units: int,
    *,
    target_node_nm: float = ANCHOR_NODE_NM,
) -> dict[str, Any]:
    """Estimate chip-side HBM PHY area for a number of 64-bit interface units."""

    units = int(interface_units)
    if units < 0:
        raise ValueError("interface_units must be non-negative")
    if float(target_node_nm) <= 0.0:
        raise ValueError("target_node_nm must be positive")
    node_scale = (float(target_node_nm) / ANCHOR_NODE_NM) ** 2
    per_unit_mm2 = AREA_MM2_PER_INTERFACE_UNIT * node_scale
    area_mm2 = per_unit_mm2 * units
    return {
        "area": area_mm2 * 1e6,
        "area_mm2": area_mm2,
        "area_mm2_per_interface_unit": per_unit_mm2,
        "interface_units": units,
        "area_model": "hbm_phy_declared_shoreline_depth",
        "evidence": {
            "tier": STRUCTURAL_ESTIMATE,
            "source": SOURCE_DOI,
            "source_shoreline_mm_per_stack": SOURCE_SHORELINE_MM_PER_STACK,
            "declared_beachfront_depth_mm": DECLARED_BEACHFRONT_DEPTH_MM,
            "stack_interface_units": STACK_INTERFACE_UNITS,
            "anchor_node_nm": ANCHOR_NODE_NM,
            "target_node_nm": float(target_node_nm),
            "node_area_scaling_exponent": 2.0,
            "synthesized_for_plena": False,
        },
    }
