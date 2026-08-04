"""Versioned HBM operating points for controlled decode sensitivity studies."""

from __future__ import annotations

from dataclasses import dataclass

HBM_TECHNOLOGY_SCHEMA = "plena-hbm-technology"
INTERFACE_UNIT_BITS = 64
MEMEXPLORER_SOURCE_URL = "https://arxiv.org/pdf/2604.16007"


@dataclass(frozen=True)
class HBMTechnology:
    """One explicit rate, capacity, and interface-width operating point."""

    generation: str
    pin_rate_gbps: float
    io_width_bits: int
    capacity_gb_per_stack: float
    source_url: str
    source_label: str
    background_power_mw_per_gb: float
    read_energy_pj_per_bit: float
    write_energy_pj_per_bit: float
    energy_source_label: str
    energy_source_url: str
    emulator_generation: str | None = None
    emulator_pin_rate_gbps: float | None = None
    schema_version: str = HBM_TECHNOLOGY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != HBM_TECHNOLOGY_SCHEMA:
            raise ValueError("unsupported HBM technology schema")
        if not self.generation or not self.source_url or not self.source_label:
            raise ValueError("HBM identity and source fields must be non-empty")
        if self.pin_rate_gbps <= 0 or self.capacity_gb_per_stack <= 0:
            raise ValueError("HBM rate and capacity must be positive")
        if not self.energy_source_label or not self.energy_source_url:
            raise ValueError("HBM energy coefficients must name and link their source")
        for name in (
            "background_power_mw_per_gb",
            "read_energy_pj_per_bit",
            "write_energy_pj_per_bit",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.io_width_bits <= 0 or self.io_width_bits % INTERFACE_UNIT_BITS:
            raise ValueError("HBM width must contain complete interface units")
        if (self.emulator_generation is None) != (
            self.emulator_pin_rate_gbps is None
        ):
            raise ValueError("emulator generation and rate must be paired")
        if (
            self.emulator_pin_rate_gbps is not None
            and self.emulator_pin_rate_gbps <= 0
        ):
            raise ValueError("emulator HBM rate must be positive")

    @property
    def interface_units_per_stack(self) -> int:
        return self.io_width_bits // INTERFACE_UNIT_BITS

    @property
    def capacity_gb_per_interface_unit(self) -> float:
        return self.capacity_gb_per_stack / self.interface_units_per_stack

    @property
    def peak_bandwidth_bytes_per_s_per_stack(self) -> float:
        return self.io_width_bits * self.pin_rate_gbps * 1e9 / 8

    @property
    def emulator_rate_matches(self) -> bool:
        return (
            self.emulator_generation == self.generation
            and self.emulator_pin_rate_gbps is not None
            and abs(self.emulator_pin_rate_gbps - self.pin_rate_gbps) <= 1e-12
        )

    def overrides(
        self,
        interface_units: int,
        *,
        clock_hz: float,
    ) -> dict[str, int]:
        """Return capacity and per-cycle interface width for explicit units."""

        units = (
            self.interface_units_per_stack
            if interface_units == 0
            else int(interface_units)
        )
        if units <= 0 or clock_hz <= 0:
            raise ValueError("HBM units and clock must be positive")
        bandwidth_bytes_per_s = (
            units
            * INTERFACE_UNIT_BITS
            * self.pin_rate_gbps
            * 1e9
            / 8
        )
        return {
            "HBM_WIDTH": int(round(bandwidth_bytes_per_s / clock_hz)) * 8,
            "HBM_SIZE": int(
                round(units * self.capacity_gb_per_interface_unit * 1e9)
            ),
            "channels": units,
        }


HBM_TECHNOLOGIES = {
    "HBM2": HBMTechnology(
        generation="HBM2",
        pin_rate_gbps=2.0,
        io_width_bits=1024,
        capacity_gb_per_stack=16.0,
        source_url="https://github.com/CMU-SAFARI/ramulator2",
        source_label="Ramulator2 HBM2_2Gbps modeled operating point",
        background_power_mw_per_gb=75.0,
        read_energy_pj_per_bit=4.2,
        write_energy_pj_per_bit=5.0,
        energy_source_label=(
            "Model assumption: HBM3E experimental read/write coefficients from "
            "MemExplorer Table 1 scaled up 1.4x for HBM2; the 75 mW/GB "
            "background midpoint is retained"
        ),
        energy_source_url=MEMEXPLORER_SOURCE_URL,
        emulator_generation="HBM2",
        emulator_pin_rate_gbps=2.0,
    ),
    "HBM2E": HBMTechnology(
        generation="HBM2E",
        pin_rate_gbps=3.2,
        io_width_bits=1024,
        capacity_gb_per_stack=16.0,
        source_url=(
            "https://www.micron.com/content/dam/micron/global/public/products/"
            "technical-marketing-brief/micron-hbm2e-memory-wp.pdf"
        ),
        source_label="Micron HBM2E technical brief",
        background_power_mw_per_gb=75.0,
        read_energy_pj_per_bit=3.6,
        write_energy_pj_per_bit=4.3,
        energy_source_label=(
            "Model assumption: HBM3E experimental read/write coefficients from "
            "MemExplorer Table 1 scaled up 1.2x for HBM2E; the 75 mW/GB "
            "background midpoint is retained"
        ),
        energy_source_url=MEMEXPLORER_SOURCE_URL,
    ),
    "HBM3": HBMTechnology(
        generation="HBM3",
        pin_rate_gbps=6.4,
        io_width_bits=1024,
        capacity_gb_per_stack=24.0,
        source_url=(
            "https://news.skhynix.com/sk-hynix-at-nvidia-gtc-2022-"
            "demonstrating-the-worlds-fastest-dram-hbm3/"
        ),
        source_label="SK hynix HBM3 product announcement",
        background_power_mw_per_gb=75.0,
        read_energy_pj_per_bit=3.2,
        write_energy_pj_per_bit=3.8,
        energy_source_label=(
            "Model assumption: HBM3E experimental read/write coefficients from "
            "MemExplorer Table 1 scaled up 1.07x for HBM3; the 75 mW/GB "
            "background midpoint is retained"
        ),
        energy_source_url=MEMEXPLORER_SOURCE_URL,
        emulator_generation="HBM3",
        emulator_pin_rate_gbps=2.0,
    ),
    "HBM3E": HBMTechnology(
        generation="HBM3E",
        pin_rate_gbps=9.2,
        io_width_bits=1024,
        capacity_gb_per_stack=24.0,
        source_url="https://www.micron.com/products/memory/hbm/hbm3e",
        source_label="Micron HBM3E product page",
        background_power_mw_per_gb=75.0,
        read_energy_pj_per_bit=3.0,
        write_energy_pj_per_bit=3.6,
        energy_source_label=(
            "MemExplorer Table 1 HBM3E experiment; 75 mW/GB is the midpoint "
            "of the reported 50-100 mW/GB background range"
        ),
        energy_source_url=MEMEXPLORER_SOURCE_URL,
    ),
    "HBM4": HBMTechnology(
        generation="HBM4",
        pin_rate_gbps=11.0,
        io_width_bits=2048,
        capacity_gb_per_stack=36.0,
        source_url="https://investors.micron.com/node/50236/pdf",
        source_label=(
            "Micron 36GB 12H HBM4 announcement; 11.0 Gb/s is a "
            "conservative lower-bound point for the stated >11 Gb/s rate"
        ),
        background_power_mw_per_gb=75.0,
        read_energy_pj_per_bit=2.2,
        write_energy_pj_per_bit=2.4,
        energy_source_label=(
            "MemExplorer Table 1 HBM4 values derived from the stated 40% "
            "efficiency improvement over HBM3E"
        ),
        energy_source_url=MEMEXPLORER_SOURCE_URL,
    ),
}


def hbm_technology(generation: str) -> HBMTechnology:
    try:
        return HBM_TECHNOLOGIES[str(generation).upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported HBM generation {generation!r}") from exc


__all__ = [
    "HBM_TECHNOLOGIES",
    "HBM_TECHNOLOGY_SCHEMA",
    "HBMTechnology",
    "INTERFACE_UNIT_BITS",
    "MEMEXPLORER_SOURCE_URL",
    "hbm_technology",
]
