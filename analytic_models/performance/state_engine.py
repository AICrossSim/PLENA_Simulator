"""Common recurrent-state engine model for Nemotron Mamba-2 and Kimi K3 KDA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum


class StateAlgorithm(StrEnum):
    MAMBA2 = "mamba2"
    KDA = "kda"


class StateSramLayout(StrEnum):
    ROW_MAJOR = "row_major"
    DUAL_AXIS_CYCLIC = "dual_axis_cyclic"


class StateStorage(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    MX8_B128 = "mx8_b128"

    @property
    def element_bytes(self) -> int:
        return 4 if self == StateStorage.FP32 else 2 if self in {StateStorage.BF16, StateStorage.FP16} else 1


@dataclass(frozen=True)
class StateGeometry:
    algorithm: StateAlgorithm
    heads: int
    rows: int
    columns: int
    precision: StateStorage
    conv_precision: StateStorage
    conv_channels: int
    conv_kernel: int
    sram_passes: int
    fmas_per_element: int

    def __post_init__(self) -> None:
        for name in (
            "heads",
            "rows",
            "columns",
            "conv_channels",
            "conv_kernel",
            "sram_passes",
            "fmas_per_element",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def nemotron3_mamba2(cls, precision: StateStorage = StateStorage.FP32) -> StateGeometry:
        # Per head: [P=64, N=128] in FP32.
        return cls(
            StateAlgorithm.MAMBA2,
            64,
            64,
            128,
            precision,
            precision,
            conv_channels=6144,
            conv_kernel=4,
            sram_passes=1,
            fmas_per_element=2,
        )

    @classmethod
    def kimi_k3_kda(
        cls,
        precision: StateStorage = StateStorage.FP32,
        conv_precision: StateStorage = StateStorage.BF16,
    ) -> StateGeometry:
        # Profiled FlashKDA state is FP32; its short-convolution history is BF16.
        return cls(
            StateAlgorithm.KDA,
            96,
            128,
            128,
            precision,
            conv_precision,
            conv_channels=96 * (2 * 128 + 128),
            conv_kernel=4,
            sram_passes=2,
            fmas_per_element=3,
        )

    @property
    def element_bytes(self) -> int:
        return self.precision.element_bytes

    @property
    def elements_per_head(self) -> int:
        return self.rows * self.columns

    @property
    def bytes_per_head(self) -> int:
        return self.bytes_per_layer // self.heads

    @property
    def elements_per_layer(self) -> int:
        return self.heads * self.elements_per_head

    @property
    def bytes_per_layer(self) -> int:
        value_bytes = self.elements_per_layer * self.element_bytes
        return value_bytes + self.state_scale_bytes_per_layer

    @property
    def state_scale_bytes_per_layer(self) -> int:
        if self.precision != StateStorage.MX8_B128:
            return 0
        return self.heads * self.rows * math.ceil(self.columns / 128)

    @property
    def conv_elements_per_layer(self) -> int:
        return self.conv_channels * self.conv_kernel

    @property
    def conv_bytes_per_layer(self) -> int:
        value_bytes = self.conv_elements_per_layer * self.conv_precision.element_bytes
        scale_bytes = (
            self.conv_channels * math.ceil(self.conv_kernel / 128)
            if self.conv_precision == StateStorage.MX8_B128
            else 0
        )
        return value_bytes + scale_bytes

    @property
    def persistent_bytes_per_layer(self) -> int:
        return self.bytes_per_layer + self.conv_bytes_per_layer


@dataclass(frozen=True)
class StateEngineDesign:
    frequency_hz: int = 1_000_000_000
    head_lanes: int = 1
    row_lanes: int = 4
    column_lanes: int = 8
    banks_per_head_lane: int = 32
    ports_per_bank: int = 1
    fma_lanes_per_head_lane: int = 32
    hbm_bytes_per_cycle: int = 64
    head_tile_slots: int = 2
    layout: StateSramLayout = StateSramLayout.ROW_MAJOR

    def __post_init__(self) -> None:
        for name in (
            "frequency_hz",
            "head_lanes",
            "row_lanes",
            "column_lanes",
            "banks_per_head_lane",
            "ports_per_bank",
            "fma_lanes_per_head_lane",
            "hbm_bytes_per_cycle",
            "head_tile_slots",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def state_values_per_cycle(self) -> int:
        return self.head_lanes * self.fma_lanes_per_head_lane


@dataclass(frozen=True)
class StateAddress:
    bank: int
    offset: int


class BankedStateLayout:
    """Map a logical per-head state matrix to single-ported SRAM banks."""

    def __init__(self, geometry: StateGeometry, design: StateEngineDesign) -> None:
        self.geometry = geometry
        self.design = design

    def address(self, row: int, column: int) -> StateAddress:
        geometry = self.geometry
        design = self.design
        if not 0 <= row < geometry.rows or not 0 <= column < geometry.columns:
            raise IndexError("state coordinate is out of range")

        if design.layout == StateSramLayout.ROW_MAJOR:
            linear = row * geometry.columns + column
            return StateAddress(linear % design.banks_per_head_lane, linear // design.banks_per_head_lane)

        # A logical row_lanes x column_lanes tile is striped over distinct
        # banks. The offset selects the coarse tile; no state is duplicated.
        row_group, local_row = divmod(row, design.row_lanes)
        column_group, local_column = divmod(column, design.column_lanes)
        column_groups = math.ceil(geometry.columns / design.column_lanes)
        logical_bank = local_row * design.column_lanes + local_column
        bank = logical_bank % design.banks_per_head_lane
        bank_group = logical_bank // design.banks_per_head_lane
        offset = (row_group * column_groups + column_group) * math.ceil(
            design.row_lanes * design.column_lanes / design.banks_per_head_lane
        ) + bank_group
        return StateAddress(bank, offset)


@dataclass(frozen=True)
class StateBankStats:
    packets: int
    values: int
    ideal_cycles: int
    service_cycles: int

    @property
    def stall_cycles(self) -> int:
        return self.service_cycles - self.ideal_cycles


@dataclass(frozen=True)
class StateEngineResult:
    geometry: StateGeometry
    design: StateEngineDesign
    bank_stats: StateBankStats
    compute_cycles: int
    state_sram_cycles: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    hbm_cycles: int
    total_cycles: int

    @property
    def latency_us(self) -> float:
        return self.total_cycles / self.design.frequency_hz * 1e6

    @property
    def head_tile_sram_bytes(self) -> int:
        return self.design.head_lanes * self.design.head_tile_slots * self.geometry.bytes_per_head

    def to_dict(self) -> dict:
        return {
            "geometry": asdict(self.geometry),
            "design": asdict(self.design),
            "bank_stats": {**asdict(self.bank_stats), "stall_cycles": self.bank_stats.stall_cycles},
            "metrics": {
                "compute_cycles": self.compute_cycles,
                "state_sram_cycles": self.state_sram_cycles,
                "hbm_read_bytes": self.hbm_read_bytes,
                "hbm_write_bytes": self.hbm_write_bytes,
                "hbm_cycles": self.hbm_cycles,
                "total_cycles": self.total_cycles,
                "latency_us": self.latency_us,
                "head_tile_sram_bytes": self.head_tile_sram_bytes,
                "total_fma_lanes": self.design.state_values_per_cycle,
                "total_state_banks": self.design.head_lanes * self.design.banks_per_head_lane,
                "conv_state_sram_bytes": self.geometry.conv_bytes_per_layer,
                "persistent_state_bytes": self.geometry.persistent_bytes_per_layer,
            },
        }


class RecurrentStateEngineModel:
    """Model one layer's fused recurrent core and per-head state streaming."""

    def __init__(self, geometry: StateGeometry, design: StateEngineDesign) -> None:
        self.geometry = geometry
        self.design = design
        self.layout = BankedStateLayout(geometry, design)

    def bank_stats(self) -> StateBankStats:
        geometry = self.geometry
        design = self.design
        one_head_packets = one_head_values = one_head_ideal = one_head_service = 0
        for row_start in range(0, geometry.rows, design.row_lanes):
            for column_start in range(0, geometry.columns, design.column_lanes):
                banks = []
                for row in range(row_start, min(row_start + design.row_lanes, geometry.rows)):
                    for column in range(column_start, min(column_start + design.column_lanes, geometry.columns)):
                        banks.append(self.layout.address(row, column).bank)
                counts = [0] * design.banks_per_head_lane
                for bank in banks:
                    counts[bank] += 1
                one_head_packets += 1
                one_head_values += len(banks)
                one_head_ideal += math.ceil(len(banks) / (design.banks_per_head_lane * design.ports_per_bank))
                one_head_service += max(math.ceil(count / design.ports_per_bank) for count in counts)

        head_waves = math.ceil(geometry.heads / design.head_lanes)
        factor = head_waves * geometry.sram_passes
        return StateBankStats(
            packets=one_head_packets * factor,
            values=one_head_values * factor,
            ideal_cycles=one_head_ideal * factor,
            service_cycles=one_head_service * factor,
        )

    def evaluate(self, *, state_resident: bool = False) -> StateEngineResult:
        geometry = self.geometry
        design = self.design
        bank_stats = self.bank_stats()
        compute_cycles = math.ceil(
            geometry.elements_per_layer
            * geometry.fmas_per_element
            / (design.head_lanes * design.fma_lanes_per_head_lane)
        )
        state_sram_cycles = bank_stats.service_cycles

        # KDA performs two on-chip passes because prediction must finish before
        # error/update. The head tile is loaded and written only once.
        hbm_read_bytes = 0 if state_resident else geometry.persistent_bytes_per_layer
        hbm_write_bytes = 0 if state_resident else geometry.persistent_bytes_per_layer
        hbm_cycles = math.ceil((hbm_read_bytes + hbm_write_bytes) / design.hbm_bytes_per_cycle)
        recurrent_cycles = max(compute_cycles, state_sram_cycles)
        if state_resident:
            total_cycles = recurrent_cycles
        elif design.head_tile_slots >= 2:
            total_cycles = max(recurrent_cycles, hbm_cycles)
        else:
            total_cycles = recurrent_cycles + hbm_cycles
        return StateEngineResult(
            geometry=geometry,
            design=design,
            bank_stats=bank_stats,
            compute_cycles=compute_cycles,
            state_sram_cycles=state_sram_cycles,
            hbm_read_bytes=hbm_read_bytes,
            hbm_write_bytes=hbm_write_bytes,
            hbm_cycles=hbm_cycles,
            total_cycles=total_cycles,
        )
