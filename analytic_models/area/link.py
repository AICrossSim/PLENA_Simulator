"""Literature-scaled chip-to-chip PHY area.

The 5 nm density anchor is 552 Gb/s/mm^2 from Gangasani et al., "A 1.6Tb/s
Chiplet over XSR-MCM Channels using 113Gb/s PAM-4 Transceiver with Dynamic
Receiver-Driven Adaptation of TX-FFE and Programmable Roaming Taps in 5nm
CMOS," ISSCC 2022, doi:10.1109/ISSCC42614.2022.9731636. Area is scaled with
the square of feature size; this is a declared analytic projection, not PLENA
synthesis data. At 7 nm, a bidirectional 900 GB/s-class port is 25.565 mm^2.
"""

from __future__ import annotations

from typing import Any

from .evidence import PUBLISHED_DENSITY_SCALING

SOURCE_DENSITY_GBPS_PER_MM2 = 552.0
SOURCE_NODE_NM = 5.0
DEFAULT_TARGET_NODE_NM = 7.0
DEFAULT_PORT_BANDWIDTH_GBPS = 900.0 * 8.0
SOURCE_DOI = "10.1109/ISSCC42614.2022.9731636"


def estimate_link_phy_area(
    *,
    bandwidth_gbps: float = DEFAULT_PORT_BANDWIDTH_GBPS,
    target_node_nm: float = DEFAULT_TARGET_NODE_NM,
    source_density_gbps_per_mm2: float = SOURCE_DENSITY_GBPS_PER_MM2,
    source_node_nm: float = SOURCE_NODE_NM,
) -> dict[str, Any]:
    """Estimate one chip-side C2C PHY port from published bandwidth density."""

    values = (
        bandwidth_gbps,
        target_node_nm,
        source_density_gbps_per_mm2,
        source_node_nm,
    )
    if any(float(value) <= 0.0 for value in values):
        raise ValueError("bandwidth, density, and process nodes must be positive")
    target_density = (
        float(source_density_gbps_per_mm2)
        * (float(source_node_nm) / float(target_node_nm)) ** 2
    )
    area_mm2 = float(bandwidth_gbps) / target_density
    return {
        "area": area_mm2 * 1e6,
        "area_mm2": area_mm2,
        "bandwidth_gbps": float(bandwidth_gbps),
        "target_density_gbps_per_mm2": target_density,
        "area_model": "c2c_phy_published_density_scaling",
        "evidence": {
            "tier": PUBLISHED_DENSITY_SCALING,
            "source": SOURCE_DOI,
            "source_node_nm": float(source_node_nm),
            "target_node_nm": float(target_node_nm),
            "source_density_gbps_per_mm2": float(source_density_gbps_per_mm2),
            "node_area_scaling_exponent": 2.0,
            "synthesized_for_plena": False,
        },
    }
