"""Caller-owned configuration of a 3D-stacked DRAM attached to a PLENA chip.

Ported from DeepStack (Mo et al., arXiv:2604.04750), tile-ai/DeepStack@8509061:

* ``DramTimingConfig`` and ``StackedDramConfig`` follow ``DramTimingConfig`` and
  ``DramInterfaceConfig`` in ``src/deepstack/mosaic/arch/custom_profile.py``.
  The GPU-only fields (L2 bandwidth multiplier, uncached utilisation, the bank
  data-rate clock used by DeepStack's bank oracle) are dropped, and the memory
  clock moves into the DRAM configuration because PLENA has no separate GPU
  profile to hold it.
* ``BufferingPolicy`` and ``ThermalPolicy`` follow the caller-supplied branch of
  ``DramDsePolicy``, ``apply_littles_law`` and ``compute_thermal_freq_scale`` in
  ``src/deepstack/mosaic/dse_space/case_study_dram_layer/dram_layer_config.py``.
  DeepStack's per-SM quantities become per-requester quantities.
* ``DramEnergyConfig`` keeps the DRAM terms of ``ChipEnergyConfig`` in
  ``src/deepstack/mosaic/cost/energy.py``.

No field carries a hardware default and this package ships no reference
calibration: every value describes the caller's own design.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ._validate import finite_number, fraction, nonnegative_int, positive_int


@dataclass(frozen=True)
class DramTimingConfig:
    """Row service and round-trip latency of one DRAM bank, in DRAM cycles.

    Reading a row takes ``sectors_per_row * sector_cycles`` cycles and opening
    the next one costs ``recharge_cycles``. The ratio of the two is the
    efficiency of a fully connected stack, where every layer streams and no
    recharge can be hidden behind another layer.
    """

    row_bytes: int
    sector_bytes: int
    sector_cycles: int
    recharge_cycles: int
    round_trip_latency_cycles: float
    latency_clock_hz: float

    def __post_init__(self) -> None:
        row = positive_int(self.row_bytes, "row_bytes")
        sector = positive_int(self.sector_bytes, "sector_bytes")
        if row % sector:
            raise ValueError("row_bytes must be an integer multiple of sector_bytes")
        positive_int(self.sector_cycles, "sector_cycles")
        nonnegative_int(self.recharge_cycles, "recharge_cycles")
        finite_number(self.round_trip_latency_cycles, "round_trip_latency_cycles", allow_zero=True)
        finite_number(self.latency_clock_hz, "latency_clock_hz")

    @property
    def sectors_per_row(self) -> int:
        return self.row_bytes // self.sector_bytes

    @property
    def row_read_cycles(self) -> int:
        return self.sectors_per_row * self.sector_cycles

    @property
    def full_row_cycles(self) -> int:
        return self.row_read_cycles + self.recharge_cycles

    @property
    def fully_connected_efficiency(self) -> float:
        return self.row_read_cycles / self.full_row_cycles

    @property
    def round_trip_latency_seconds(self) -> float:
        return self.round_trip_latency_cycles / self.latency_clock_hz


@dataclass(frozen=True)
class StackedDramConfig:
    """Interface of a DRAM stack with ``total_layers`` of which ``connected_layers`` stream.

    Peak bandwidth is given in exactly one of two forms:

    * transfer composition: ``connected_layers * channels_per_connected_layer *
      bytes_per_channel_transfer * transfers_per_memory_clock *
      memory_frequency_hz``;
    * ``direct_peak_bandwidth_per_connected_layer_bytes_per_s`` times
      ``connected_layers``.

    The efficiency of a fully connected stack is given either directly
    (``fully_connected_efficiency``) or derived from ``bank_timing``.
    ``transaction_bytes * channels_per_connected_layer`` is the DRAM "wave":
    with ``apply_wave_quantization`` every transfer is rounded up to whole waves.
    """

    total_layers: int
    connected_layers: int
    channels_per_connected_layer: int
    capacity_per_layer_bytes: int
    transaction_bytes: int
    bytes_per_channel_transfer: float | None = None
    transfers_per_memory_clock: float | None = None
    memory_frequency_hz: float | None = None
    direct_peak_bandwidth_per_connected_layer_bytes_per_s: float | None = None
    fully_connected_efficiency: float | None = None
    bank_timing: DramTimingConfig | None = None
    apply_wave_quantization: bool = False

    def __post_init__(self) -> None:
        total = positive_int(self.total_layers, "total_layers")
        connected = positive_int(self.connected_layers, "connected_layers")
        if connected > total:
            raise ValueError("connected_layers must not exceed total_layers")
        positive_int(self.channels_per_connected_layer, "channels_per_connected_layer")
        positive_int(self.capacity_per_layer_bytes, "capacity_per_layer_bytes")
        positive_int(self.transaction_bytes, "transaction_bytes")

        composition = {
            "bytes_per_channel_transfer": self.bytes_per_channel_transfer,
            "transfers_per_memory_clock": self.transfers_per_memory_clock,
            "memory_frequency_hz": self.memory_frequency_hz,
        }
        direct_peak = self.direct_peak_bandwidth_per_connected_layer_bytes_per_s
        if direct_peak is None:
            missing = sorted(name for name, value in composition.items() if value is None)
            if missing:
                raise ValueError(
                    "transfer composition requires "
                    + ", ".join(missing)
                    + " (or direct_peak_bandwidth_per_connected_layer_bytes_per_s instead)"
                )
            for name, value in composition.items():
                finite_number(value, name)
        else:
            finite_number(direct_peak, "direct_peak_bandwidth_per_connected_layer_bytes_per_s")
            given = sorted(name for name, value in composition.items() if value is not None)
            if given:
                raise ValueError(
                    "direct peak bandwidth and transfer composition are mutually exclusive: " + ", ".join(given)
                )

        if self.bank_timing is None:
            if self.fully_connected_efficiency is None:
                raise ValueError("supply fully_connected_efficiency or bank_timing")
            fraction(self.fully_connected_efficiency, "fully_connected_efficiency", allow_zero=False)
        else:
            if not isinstance(self.bank_timing, DramTimingConfig):
                raise TypeError("bank_timing must be a DramTimingConfig")
            if self.fully_connected_efficiency is not None:
                raise ValueError("fully_connected_efficiency and bank_timing are mutually exclusive")
        if not isinstance(self.apply_wave_quantization, bool):
            raise TypeError("apply_wave_quantization must be a bool")

    @property
    def resolved_fully_connected_efficiency(self) -> float:
        if self.bank_timing is not None:
            return self.bank_timing.fully_connected_efficiency
        assert self.fully_connected_efficiency is not None
        return float(self.fully_connected_efficiency)

    @property
    def capacity_bytes(self) -> int:
        return self.total_layers * self.capacity_per_layer_bytes

    @property
    def wave_bytes(self) -> int:
        return self.transaction_bytes * self.channels_per_connected_layer

    def peak_bandwidth_bytes_per_s(self, connected_layers: int | None = None) -> float:
        connected = (
            self.connected_layers if connected_layers is None else positive_int(connected_layers, "connected_layers")
        )
        direct_peak = self.direct_peak_bandwidth_per_connected_layer_bytes_per_s
        if direct_peak is not None:
            return connected * float(direct_peak)
        assert self.bytes_per_channel_transfer is not None
        assert self.transfers_per_memory_clock is not None
        assert self.memory_frequency_hz is not None
        return (
            connected
            * self.channels_per_connected_layer
            * float(self.bytes_per_channel_transfer)
            * float(self.transfers_per_memory_clock)
            * float(self.memory_frequency_hz)
        )

    def with_layers(self, total_layers: int, connected_layers: int | None = None) -> StackedDramConfig:
        """Return the same interface with a different stack height.

        ``connected_layers`` defaults to ``total_layers`` (a fully connected
        stack), as in DeepStack's ``update_ddr``.
        """

        total = positive_int(total_layers, "total_layers")
        connected = total if connected_layers is None else connected_layers
        return replace(self, total_layers=total, connected_layers=connected)


@dataclass(frozen=True)
class BufferingPolicy:
    """Little's-law bound on the bandwidth a finite landing buffer can sustain.

    ``requesters`` independent streams share the stack bandwidth ``B``. Keeping
    a stream busy needs ``(B / requesters) * latency * buffering_factor`` bytes
    in flight; with a smaller ``buffer_bytes_per_requester`` the usable
    bandwidth shrinks in proportion. DeepStack applies this per SM with its
    shared memory as the buffer; on PLENA a requester is an independent DMA
    stream and its buffer is the on-chip SRAM reserved for that stream.

    The round-trip latency comes either from this policy (cycles and their
    clock, both given) or, when both are omitted, from the DRAM's
    ``bank_timing``.
    """

    requesters: int
    buffer_bytes_per_requester: int
    buffering_factor: float
    round_trip_latency_cycles: float | None = None
    round_trip_latency_clock_hz: float | None = None

    def __post_init__(self) -> None:
        positive_int(self.requesters, "requesters")
        positive_int(self.buffer_bytes_per_requester, "buffer_bytes_per_requester")
        finite_number(self.buffering_factor, "buffering_factor")
        cycles = self.round_trip_latency_cycles
        clock = self.round_trip_latency_clock_hz
        if (cycles is None) != (clock is None):
            raise ValueError("round_trip_latency_cycles and round_trip_latency_clock_hz must be given together")
        if cycles is not None:
            finite_number(cycles, "round_trip_latency_cycles", allow_zero=True)
            finite_number(clock, "round_trip_latency_clock_hz")

    def round_trip_latency_seconds(self, bank_timing: DramTimingConfig | None) -> float:
        if self.round_trip_latency_cycles is not None:
            assert self.round_trip_latency_clock_hz is not None
            return float(self.round_trip_latency_cycles) / float(self.round_trip_latency_clock_hz)
        if bank_timing is not None:
            return bank_timing.round_trip_latency_seconds
        raise ValueError("buffering needs round_trip_latency_cycles/round_trip_latency_clock_hz or a DRAM bank_timing")


@dataclass(frozen=True)
class ThermalPolicy:
    """Compute-frequency derating as the stack above the logic die grows.

    The junction-to-ambient resistance is ``resistance_base_c_per_w +
    resistance_per_layer_c_per_w * layers``. At ``baseline_layers`` the chip may
    dissipate ``design_power_w``; a taller stack lowers the allowed power in
    proportion to its resistance. The dynamic part of the remaining budget sets
    the frequency through ``power ~ frequency ** dynamic_power_exponent``.
    Shorter stacks than the baseline give a scale above one, as in DeepStack.
    """

    resistance_base_c_per_w: float
    resistance_per_layer_c_per_w: float
    baseline_layers: int
    design_power_w: float
    static_power_w: float
    dynamic_power_exponent: float

    def __post_init__(self) -> None:
        finite_number(self.resistance_base_c_per_w, "resistance_base_c_per_w")
        finite_number(self.resistance_per_layer_c_per_w, "resistance_per_layer_c_per_w", allow_zero=True)
        positive_int(self.baseline_layers, "baseline_layers")
        design = finite_number(self.design_power_w, "design_power_w")
        static = finite_number(self.static_power_w, "static_power_w", allow_zero=True)
        if static >= design:
            raise ValueError("static_power_w must be below design_power_w")
        finite_number(self.dynamic_power_exponent, "dynamic_power_exponent")

    def frequency_scale(self, total_layers: int) -> float:
        layers = positive_int(total_layers, "total_layers")
        resistance = self.resistance_base_c_per_w + self.resistance_per_layer_c_per_w * layers
        baseline_resistance = self.resistance_base_c_per_w + self.resistance_per_layer_c_per_w * self.baseline_layers
        max_power = self.design_power_w * baseline_resistance / resistance
        dynamic_available = max_power - self.static_power_w
        dynamic_budget = self.design_power_w - self.static_power_w
        if dynamic_available <= 0.0:
            raise ValueError(f"thermal policy leaves no dynamic-power budget at {layers} layers")
        return (dynamic_available / dynamic_budget) ** (1.0 / self.dynamic_power_exponent)


@dataclass(frozen=True)
class DramEnergyConfig:
    """Access energy of the DRAM interface, in picojoules per transferred bit."""

    read_pj_per_bit: float
    write_pj_per_bit: float

    def __post_init__(self) -> None:
        finite_number(self.read_pj_per_bit, "read_pj_per_bit", allow_zero=True)
        finite_number(self.write_pj_per_bit, "write_pj_per_bit", allow_zero=True)

    def energy_pj(self, read_bytes: float, write_bytes: float) -> float:
        return 8.0 * (read_bytes * self.read_pj_per_bit + write_bytes * self.write_pj_per_bit)
