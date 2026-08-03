"""Literature-parameterized external HBM energy bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping

from analytic_models.latency.schemas import MemoryLatencyReport


@dataclass(frozen=True)
class ExternalHbmEnergy:
    background_low_pj: float
    background_nominal_pj: float
    background_high_pj: float
    read_pj: float
    write_pj: float
    provenance: dict[str, Any]

    @property
    def nominal_pj(self) -> float:
        return self.background_nominal_pj + self.read_pj + self.write_pj


def estimate_external_hbm_energy(
    memory_report: MemoryLatencyReport,
    runtime_picos: int,
    config: Mapping[str, Any],
) -> ExternalHbmEnergy:
    """Charge physical HBM traffic and a capacity-proportional background."""

    if runtime_picos <= 0:
        raise ValueError("runtime_picos must be positive")
    capacity_bytes = int(config["capacity_bytes"])
    if capacity_bytes <= 0:
        raise ValueError("external HBM capacity must be positive")
    coefficients = config["coefficients"]
    background = coefficients["background_power_mw_per_gb"]
    capacity_gb = capacity_bytes / 1_000_000_000
    runtime_ns = runtime_picos / 1_000

    # mW * ns is pJ; physical bytes include partial-store RMW reads from V4.
    background_energy = {
        name: capacity_gb * float(background[name]) * runtime_ns
        for name in ("p10", "p50", "p90")
    }
    read_pj = memory_report.physical_read_bytes * 8 * float(coefficients["read_energy_pj_per_bit"])
    write_pj = memory_report.physical_write_bytes * 8 * float(coefficients["write_energy_pj_per_bit"])
    return ExternalHbmEnergy(
        background_low_pj=background_energy["p10"],
        background_nominal_pj=background_energy["p50"],
        background_high_pj=background_energy["p90"],
        read_pj=read_pj,
        write_pj=write_pj,
        provenance={
            "model": config.get("model", "external-hbm"),
            "technology": config.get("technology", "unknown"),
            "calibration_status": config.get("calibration_status", "unspecified"),
            "traffic_semantics": "HBM-V4 physical 64-byte line traffic including store RMW",
            "capacity_bytes": capacity_bytes,
        },
    )


__all__ = ["ExternalHbmEnergy", "estimate_external_hbm_energy"]
