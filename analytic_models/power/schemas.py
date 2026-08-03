"""Stable schemas for compiler-action power estimation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnergyAction:
    """One compressed hardware action emitted by a compiler schedule."""

    stage: str
    component: str
    action: str
    count: int
    source_opcode: str
    precision: str = ""
    active_instances: int = 0
    total_instances: int = 0
    active_bits: int = 0
    busy_picos: int = 0
    bytes: int = 0
    fidelity: str = "exact"

    def __post_init__(self) -> None:
        for name in (
            "count",
            "active_instances",
            "total_instances",
            "active_bits",
            "busy_picos",
            "bytes",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"EnergyAction.{name} must be nonnegative")
        if self.active_instances and self.total_instances:
            if self.active_instances > self.total_instances:
                raise ValueError("active instances cannot exceed total instances")


@dataclass(frozen=True)
class ActionHardwareConfig:
    """Physical widths needed to interpret final-schedule actions."""

    mlen: int
    blen: int
    vlen: int
    fp_format: str = "FP_E6M5"
    matrix_mode: str = "mxfp"
    matrix_t_bits: int = 8
    matrix_l_bits: int = 8
    int_width: int = 32
    clock_period_picos: int = 1_000

    def __post_init__(self) -> None:
        for name in (
            "mlen",
            "blen",
            "vlen",
            "matrix_t_bits",
            "matrix_l_bits",
            "int_width",
            "clock_period_picos",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mlen % self.blen:
            raise ValueError("MLEN must be divisible by BLEN")
        if self.matrix_mode not in {"mxint", "mxfp"}:
            raise ValueError("matrix_mode must be 'mxint' or 'mxfp'")

    @property
    def fp_width(self) -> int:
        prefix = self.fp_format.upper()
        if not prefix.startswith("FP_E") or "M" not in prefix:
            raise ValueError(f"unsupported FP format {self.fp_format!r}")
        exponent, mantissa = prefix.removeprefix("FP_E").split("M", 1)
        return 1 + int(exponent) + int(mantissa)

    @classmethod
    def from_mapping(
        cls,
        value: ActionHardwareConfig | Mapping[str, Any],
    ) -> ActionHardwareConfig:
        if isinstance(value, cls):
            return value
        aliases = {
            "mlen": ("mlen", "MLEN"),
            "blen": ("blen", "BLEN"),
            "vlen": ("vlen", "VLEN"),
            "fp_format": ("fp_format", "FP_FORMAT"),
            "matrix_mode": ("matrix_mode", "MATRIX_MODE"),
            "matrix_t_bits": ("matrix_t_bits", "MATRIX_T_BITS"),
            "matrix_l_bits": ("matrix_l_bits", "MATRIX_L_BITS"),
            "int_width": ("int_width", "INT_WIDTH", "INT_DATA_WIDTH"),
            "clock_period_picos": ("clock_period_picos", "CLOCK_PERIOD_PS"),
        }
        result: dict[str, Any] = {}
        for target, names in aliases.items():
            for name in names:
                if name in value:
                    result[target] = value[name]
                    break
        return cls(**result)


@dataclass(frozen=True)
class ComponentProperty:
    logic_area_um2: float
    logic_leakage_mw: float
    clock_density_pj_per_cycle_um2: float
    fixed_clock_pj_per_active_cycle: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "logic_area_um2",
            "logic_leakage_mw",
            "clock_density_pj_per_cycle_um2",
            "fixed_clock_pj_per_active_cycle",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"ComponentProperty.{name} must be nonnegative")


@dataclass(frozen=True)
class ComponentPhysicalProperties:
    """Versioned physical inputs kept separate from workload actions."""

    schema_version: str
    calibration_id: str
    hardware: ActionHardwareConfig
    components: dict[str, ComponentProperty]
    corner: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: ComponentPhysicalProperties | Mapping[str, Any] | str | Path,
    ) -> ComponentPhysicalProperties:
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            with Path(value).open() as handle:
                value = json.load(handle)
        return cls(
            schema_version=str(value["schema_version"]),
            calibration_id=str(value["calibration_id"]),
            hardware=ActionHardwareConfig.from_mapping(value["hardware"]),
            components={
                str(name): ComponentProperty(**raw)
                for name, raw in value.get("components", {}).items()
            },
            corner=dict(value.get("corner", {})),
            provenance=dict(value.get("provenance", {})),
        )


@dataclass(frozen=True)
class ActionEnergyReport:
    actions: tuple[EnergyAction, ...]
    nominal_energy_pj: float
    low_energy_pj: float
    high_energy_pj: float
    by_component_pj: dict[str, float]
    by_stage_pj: dict[str, float]
    opcode_coverage: float
    active_shape_coverage: float
    sram_descriptor_coverage: float
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "ActionEnergyReport",
    "ActionHardwareConfig",
    "ComponentPhysicalProperties",
    "ComponentProperty",
    "EnergyAction",
]
