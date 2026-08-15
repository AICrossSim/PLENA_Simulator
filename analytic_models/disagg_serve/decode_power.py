"""Decode-chip power and energy-efficiency model.

Total decode power is the sum of five explicitly reported terms:

    P_total = P_memory + P_compute + P_sram + P_leakage + P_link

The memory term follows the transactional formulation used for heterogeneous
NPU memory studies: a capacity-proportional background component plus per-bit
read and write components driven by realised bandwidth.

    P_memory = rho_bg * C + e_read * BW_read + e_write * BW_write

where C is the provisioned capacity in GB, BW is in bits/s, and the three
coefficients come from the technology table in ``hbm_technology``.

The compute coefficient is anchored so the reference configuration reproduces
a literature-reported model output.  SRAM, leakage and link costs are then
reported separately as structural sensitivities.  This deliberately errs on
the conservative side because the literature residual also contains some
non-array logic; it never presents an uncalibrated structural estimate as a DC
or measured-silicon result.

The memory term carries the technology-specific detail — background power scales
with capacity and the dynamic terms with per-bit read and write energy — while
the compute term is a single uniformly scaled contribution.

Which term dominates depends on the operating point and is not a property of
decode being memory bound in *time*. At the declared reference assumptions the
memory term is approximately 87.1 W of the 300.09 W literature target, leaving
an unrounded residual target of approximately 213.0 W. The stored 0.203 pJ/MAC
coefficient is rounded to three decimal places and reproduces that total within
0.1%. At a long-context point whose KV cache dwarfs the weights, memory can
become the majority. Both terms are therefore reported separately rather than
one being treated as a correction to the other.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from .handoff import LINK_ENERGY_PJ_PER_BIT, LINK_ENERGY_SOURCE
    from .hbm_technology import HBMTechnology, hbm_technology
except ImportError:
    from handoff import LINK_ENERGY_PJ_PER_BIT, LINK_ENERGY_SOURCE
    from hbm_technology import HBMTechnology, hbm_technology

#: Whole-chip energy per multiply-accumulate at the reference operand width,
#: including on-chip SRAM, vector unit, control and clock distribution.
#:
#: The literature anchor is a 2048 x 128 array with four HBM3E stacks reported
#: at 300.09 W. Subtracting the memory power this module predicts for that
#: configuration at a steady-state
#: 70% bandwidth utilisation leaves 213 W for the rest of the chip, which over
#: 2.62e14 MAC/s at 8-bit operands is 0.813 pJ, or 0.203 pJ scaled to the 4-bit
#: reference width. ``calibrate_reference_mac_energy`` reproduces this figure
#: and ``REFERENCE_CONFIGURATION`` records the anchor.
REFERENCE_MAC_ENERGY_PJ = 0.203
REFERENCE_MAC_BITS = 4

ANALYTIC_ENERGY_TIER = "analytic_anchored"
DC_ENERGY_TIER = "dc_calibrated"
ENERGY_TIERS = frozenset({ANALYTIC_ENERGY_TIER, DC_ENERGY_TIER})

# The per-bit dynamic read energy is the median over the vendored ASAP7 SRAM
# macro library extraction (sram_energy_asap7_v1.json: rise+fall VDD clk
# internal power over a 1 ns period, TT/0.7 V/25 C, 36 macros). The previous
# Horowitz constant-field estimate (0.0243 pJ/bit) sat at the optimistic edge
# of that extracted range (0.023-0.114 pJ/bit); the library median replaces
# it so the coefficient is a characterized macro figure rather than a scaled
# textbook anchor. The ASAP7 macro census still supplies accessed
# capacity/width.
_SRAM_ENERGY_TABLE_PATH = Path(__file__).with_name("sram_energy_asap7_v1.json")


def _sram_read_pj_per_bit_median(path: Path = _SRAM_ENERGY_TABLE_PATH) -> float:
    macros = json.loads(path.read_text(encoding="utf-8"))["macros"]
    per_bit = []
    for macro in macros:
        entry = macro["extraction"]["read"]["entries"][0]
        width = int(macro["bits"]) // int(macro["depth"])
        per_bit.append(
            (float(entry["rise_power_mw"]) + float(entry["fall_power_mw"])) / width
        )
    per_bit.sort()
    middle = len(per_bit) // 2
    if len(per_bit) % 2:
        return per_bit[middle]
    return (per_bit[middle - 1] + per_bit[middle]) / 2.0


SRAM_ACCESS_ENERGY_PJ_PER_BIT = _sram_read_pj_per_bit_median()
SRAM_ENERGY_SOURCE = {
    "source_title": "ASAP7 SRAM macro library internal-power extraction",
    "source_artifact": "sram_energy_asap7_v1.json",
    "macro_library": "The-OpenROAD-Project/asap7_sram_0p0",
    "corner": "TT / 0.7 V / 25 C",
    "statistic": "median read pJ/bit over 36 macros",
    "macro_geometry_source": "ASAP7 SRAM macro table",
    "evidence_scope": "macro library internal-power extraction; not PLENA netlist power",
}

# A gate-level Design Compiler campaign measured leakage on the MatrixMachine
# alone: eight timing-closed points, ASAP7 RVT_TT at PVT_0P7V_25C, MLEN 16-64,
# MXFP operands. Leakage tracked area to 1.44% across all eight, giving a
# density of 9.2097e-07 mW/um^2 -- 9.21e-04 W/mm^2. That is a real measurement
# and it is recorded here in full, but it is *not* adopted as the coefficient,
# for two reasons that the number itself cannot resolve:
#
#   Temperature. Subthreshold leakage rises steeply with junction temperature,
#   and 25 C is the coldest corner in the library. A datacentre part runs at
#   85-125 C. The campaign synthesised no hot corner, so the derating factor
#   between the measured point and an operating point is unmeasured here; any
#   factor used would be an assumption, not evidence.
#
#   Scope. The measured design is 98.3% dense compute array. This coefficient
#   is charged against whole-chip non-memory logic, whose cell mix, utilisation
#   and threshold-voltage distribution differ from a systolic datapath.
#
# The declared 0.05 W/mm^2 therefore stays as the default. It is conservative
# with respect to the measurement rather than contradicted by it: the measured
# 25 C array density bounds realistic leakage from below, and the declared
# value sits above that bound. Adopting the 25 C figure would make every design
# look better on a corner mismatch. The sensitivity is small either way and is
# recorded in MATRIX_MACHINE_LEAKAGE_MEASUREMENT so the choice is auditable.
LOGIC_LEAKAGE_W_PER_MM2 = 0.05

#: Measured leakage density from the gate-level campaign, recorded as a scoped
#: datum and a lower bound. Deliberately not wired into the default path.
MATRIX_MACHINE_LEAKAGE_MEASUREMENT = {
    "w_per_mm2": 9.209669e-04,
    "mw_per_um2": 9.209669e-07,
    "n_points": 8,
    "spread_pct": 1.44,
    "corner": "ASAP7 RVT_TT / PVT_0P7V_25C",
    "temperature_c": 25,
    "block_scope": "matrix_machine only (98.3% compute array); not full-chip logic",
    "geometry_scope": "MLEN 16-64, BLEN 4-8, MXFP operands, um^2",
    "evidence_tier": "gate_level_dc_measured",
    "artifact": (
        "analytic_models/area/calibration/matrix_gate_level_validation.json"
    ),
    "relation_to_default": "lower_bound",
    "default_over_measured_ratio": LOGIC_LEAKAGE_W_PER_MM2 / 9.209669e-04,
    "not_adopted_because": (
        "25 C is the coldest library corner and no hot-corner point was "
        "synthesised, so the temperature derating to an 85-125 C operating "
        "junction is unmeasured; and one dense compute block is not the "
        "whole-chip logic this coefficient is charged against"
    ),
    "upgrade_path": (
        "re-report the same mapped netlists at a hot operating condition and "
        "extend the campaign to the full chip; that replaces the declared "
        "coefficient with a measured one at matching corner and scope"
    ),
    "sensitivity": (
        "at a representative decode point the leakage term is about 0.2% of "
        "total power at the declared coefficient and about 0.004% at the "
        "measured 25 C density, so the choice moves tokens/joule by roughly "
        "0.2%; the gap widens only for large, lightly loaded arrays"
    ),
}

LEAKAGE_SOURCE = {
    "coefficient_w_per_mm2": LOGIC_LEAKAGE_W_PER_MM2,
    "evidence_scope": (
        "declared 7 nm structural sensitivity; not DC calibrated at "
        "whole-chip scope or at an operating junction temperature"
    ),
    "conservative_direction": "above the measured 25 C lower bound",
    "measured_lower_bound": MATRIX_MACHINE_LEAKAGE_MEASUREMENT,
}

#: Independent gate-level corroboration of REFERENCE_MAC_ENERGY_PJ, the most
#: load-bearing coefficient in this module. The same campaign priced two mapped
#: MatrixMachine netlists over a declared toggle envelope; the analytic anchor
#: falls inside that envelope at a toggle rate that agrees across geometries.
#: This is a cross-check, not a calibration: the toggle rate is assumed and
#: propagated by the synthesis tool, not measured from decode switching, so the
#: energy tier is unchanged and no coefficient is refitted from it.
COMPUTE_ENERGY_CROSS_CHECK = {
    "anchor_pj_per_mac": REFERENCE_MAC_ENERGY_PJ,
    "envelope_pj_per_mac": [0.1128, 1.1256],
    "declared_toggle_rates": [0.05, 0.10, 0.25, 0.50],
    "implied_toggle_rate": {"MXFP_E1M2_32x4": 0.0797, "MXFP_E1M2_16x4": 0.0835},
    "implied_toggle_rate_range": [0.0732, 0.0925],
    "geometries_bracketing_the_anchor": 6,
    "corner": "ASAP7 RVT_TT / PVT_0P7V_25C, 1000 ps",
    "block_scope": "matrix_machine only; not whole-chip energy per MAC",
    "evidence_tier": "gate_level_declared_activity_estimate",
    "artifact": (
        "analytic_models/area/calibration/matrix_gate_level_validation.json"
    ),
    "caveat": (
        "declared-activity vectorless analysis, not annotated decode "
        "switching; it brackets the anchor and does not replace it"
    ),
    "coefficient_changed": False,
}

#: The literature model output used to anchor the compute coefficient.
REFERENCE_CONFIGURATION = {
    "array": (2048, 128),
    "hbm_generation": "HBM3E",
    "hbm_stacks": 4,
    "operand_bits": 8,
    "bandwidth_utilisation": 0.70,
    "read_fraction": 0.95,
    "write_fraction": 0.05,
    "reference_clock_hz": 1.0e9,
    "array_active_fraction": 1.0,
    "compute_scaling_rule": (
        "residual whole-chip power scales linearly with multiplier count and "
        "activity, linearly with clock, and quadratically with operand width"
    ),
    "reference_total_watts": 300.09,
    "reference_average_decode_watts": 257.4,
    "source_title": (
        "MemExplorer: Navigating the Heterogeneous Memory Design Space for "
        "Agentic Inference NPUs"
    ),
    "source_url": "https://arxiv.org/pdf/2604.16007",
    "source_tables": "Tables 5 and 6",
    "source_scope": (
        "literature-reported analytic/synthesis model output; not measured silicon"
    ),
}


@dataclass(frozen=True)
class DecodePower:
    """Average power and energy efficiency for one decode operating point."""

    memory_watts: float
    compute_watts: float
    tokens_per_second: float
    memory_source_label: str
    sram_watts: float = 0.0
    leakage_watts: float = 0.0
    link_watts: float = 0.0
    token_latency_s: float | None = None
    energy_tier: str = ANALYTIC_ENERGY_TIER
    energy_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "memory_watts",
            "compute_watts",
            "sram_watts",
            "leakage_watts",
            "link_watts",
            "tokens_per_second",
        ):
            value = float(getattr(self, name))
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not self.memory_source_label:
            raise ValueError("memory_source_label must be non-empty")
        if self.token_latency_s is not None:
            latency = float(self.token_latency_s)
            if latency <= 0 or not math.isfinite(latency):
                raise ValueError("token_latency_s must be finite and positive")
            object.__setattr__(self, "token_latency_s", latency)
        if self.energy_tier not in ENERGY_TIERS:
            raise ValueError(f"unsupported energy tier {self.energy_tier!r}")
        if self.energy_id is not None and not self.energy_id:
            raise ValueError("energy_id must be non-empty")

    @property
    def total_watts(self) -> float:
        return (
            self.memory_watts
            + self.compute_watts
            + self.sram_watts
            + self.leakage_watts
            + self.link_watts
        )

    @property
    def energy_per_token_j(self) -> float:
        if self.tokens_per_second <= 0:
            return 0.0
        return self.total_watts / self.tokens_per_second

    @property
    def tokens_per_joule(self) -> float:
        """Decode tokens delivered per joule, the decode energy-efficiency metric."""

        if self.total_watts <= 0 or self.tokens_per_second <= 0:
            return 0.0
        return self.tokens_per_second / self.total_watts

    @property
    def edp_j_s(self) -> float:
        """Energy-delay product using per-token energy and decode TPOT."""

        if self.energy_per_token_j <= 0:
            return 0.0
        latency = (
            self.token_latency_s
            if self.token_latency_s is not None
            else 1.0 / self.tokens_per_second
        )
        return self.energy_per_token_j * latency

    @property
    def memory_fraction(self) -> float:
        total = self.total_watts
        return self.memory_watts / total if total > 0 else 0.0

    def summary(self) -> dict[str, float | str]:
        return {
            "memory_watts": self.memory_watts,
            "compute_watts": self.compute_watts,
            "sram_watts": self.sram_watts,
            "leakage_watts": self.leakage_watts,
            "link_watts": self.link_watts,
            "total_watts": self.total_watts,
            "tokens_per_second": self.tokens_per_second,
            "energy_per_token_j": self.energy_per_token_j,
            "tokens_per_joule": self.tokens_per_joule,
            "token_latency_s": self.token_latency_s,
            "edp_j_s": self.edp_j_s,
            "energy_tier": self.energy_tier,
            "energy_id": self.energy_id,
            "memory_fraction": self.memory_fraction,
            "memory_source": self.memory_source_label,
        }


def memory_power_watts(
    technology: HBMTechnology,
    *,
    capacity_bytes: float,
    read_bytes_per_second: float,
    write_bytes_per_second: float,
) -> float:
    """Return average memory power for a provisioned capacity and traffic mix."""

    if capacity_bytes < 0 or read_bytes_per_second < 0 or write_bytes_per_second < 0:
        raise ValueError("capacity and bandwidth must be non-negative")
    background_w = technology.background_power_mw_per_gb * (capacity_bytes / 1e9) / 1e3
    read_w = technology.read_energy_pj_per_bit * read_bytes_per_second * 8 * 1e-12
    write_w = technology.write_energy_pj_per_bit * write_bytes_per_second * 8 * 1e-12
    return background_w + read_w + write_w


def compute_power_watts(
    *,
    multipliers: int,
    clock_hz: float,
    mac_bits: int,
    array_active_fraction: float,
) -> float:
    """Return average on-chip (non-memory) power for the configured array.

    Multiplier energy scales with the product of operand widths, so an ``M``-bit
    multiply costs ``(M / REFERENCE_MAC_BITS)^2`` times the reference energy.
    """

    if multipliers <= 0 or clock_hz <= 0 or mac_bits <= 0:
        raise ValueError("array geometry and clock must be positive")
    if not 0.0 <= array_active_fraction <= 1.0:
        raise ValueError("array_active_fraction must lie in [0, 1]")
    width_scale = (mac_bits / REFERENCE_MAC_BITS) ** 2
    mac_energy_j = REFERENCE_MAC_ENERGY_PJ * width_scale * 1e-12
    return multipliers * clock_hz * array_active_fraction * mac_energy_j


def sram_power_watts(
    *,
    read_bytes_per_second: float,
    write_bytes_per_second: float,
    energy_pj_per_bit: float = SRAM_ACCESS_ENERGY_PJ_PER_BIT,
) -> float:
    """Return structural SRAM dynamic power from physical accessed bytes."""

    for name, value in (
        ("read_bytes_per_second", read_bytes_per_second),
        ("write_bytes_per_second", write_bytes_per_second),
        ("energy_pj_per_bit", energy_pj_per_bit),
    ):
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    return (
        (read_bytes_per_second + write_bytes_per_second)
        * 8.0
        * energy_pj_per_bit
        * 1e-12
    )


def logic_leakage_power_watts(
    *,
    logic_area_mm2: float,
    leakage_w_per_mm2: float = LOGIC_LEAKAGE_W_PER_MM2,
) -> float:
    """Scale the declared leakage sensitivity by non-memory logic area."""

    if logic_area_mm2 < 0 or not math.isfinite(logic_area_mm2):
        raise ValueError("logic_area_mm2 must be finite and non-negative")
    if leakage_w_per_mm2 < 0 or not math.isfinite(leakage_w_per_mm2):
        raise ValueError("leakage_w_per_mm2 must be finite and non-negative")
    return logic_area_mm2 * leakage_w_per_mm2


def link_power_watts(
    *,
    transferred_bytes_per_second: float,
    link_generation: str = "nvlink4",
) -> float:
    """Return collective-link dynamic power from transferred physical bits."""

    if transferred_bytes_per_second < 0 or not math.isfinite(
        transferred_bytes_per_second
    ):
        raise ValueError(
            "transferred_bytes_per_second must be finite and non-negative"
        )
    try:
        energy_pj_per_bit = LINK_ENERGY_PJ_PER_BIT[link_generation]
    except KeyError as exc:
        raise ValueError(
            f"unsupported link generation {link_generation!r}"
        ) from exc
    return transferred_bytes_per_second * 8.0 * energy_pj_per_bit * 1e-12


def analytic_energy_identity() -> str:
    """Content identity for every coefficient used by the analytic tier."""

    payload = {
        "tier": ANALYTIC_ENERGY_TIER,
        "reference_mac_energy_pj": REFERENCE_MAC_ENERGY_PJ,
        "reference_mac_bits": REFERENCE_MAC_BITS,
        "reference_configuration": REFERENCE_CONFIGURATION,
        "sram_access_energy_pj_per_bit": SRAM_ACCESS_ENERGY_PJ_PER_BIT,
        "sram_source": SRAM_ENERGY_SOURCE,
        "logic_leakage_w_per_mm2": LOGIC_LEAKAGE_W_PER_MM2,
        "leakage_source": LEAKAGE_SOURCE,
        "compute_energy_cross_check": COMPUTE_ENERGY_CROSS_CHECK,
        "link_energy_pj_per_bit": LINK_ENERGY_PJ_PER_BIT,
        "link_source": LINK_ENERGY_SOURCE,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"analytic-decode-energy-{digest}"


def decode_power(
    technology: HBMTechnology,
    *,
    capacity_bytes: float,
    read_bytes_per_second: float,
    write_bytes_per_second: float,
    multipliers: int,
    clock_hz: float,
    mac_bits: int,
    array_active_fraction: float,
    tokens_per_second: float,
    chip_count: int = 1,
    sram_read_bytes_per_second: float = 0.0,
    sram_write_bytes_per_second: float = 0.0,
    logic_area_mm2: float = 0.0,
    link_bytes_per_second: float = 0.0,
    link_generation: str = "nvlink4",
    token_latency_s: float | None = None,
) -> DecodePower:
    """Compose power for a whole decode system from per-chip inputs.

    Capacity, traffic rates, multiplier count, clock, operand width, and array
    activity describe one chip. ``tokens_per_second`` is the system-wide
    throughput. The per-chip memory and compute estimates are each multiplied
    by ``chip_count`` exactly once.
    """

    if chip_count <= 0:
        raise ValueError("chip_count must be positive")
    if tokens_per_second < 0:
        raise ValueError("tokens_per_second must be non-negative")
    memory_w = memory_power_watts(
        technology,
        capacity_bytes=capacity_bytes,
        read_bytes_per_second=read_bytes_per_second,
        write_bytes_per_second=write_bytes_per_second,
    )
    compute_w = compute_power_watts(
        multipliers=multipliers,
        clock_hz=clock_hz,
        mac_bits=mac_bits,
        array_active_fraction=array_active_fraction,
    )
    sram_w = sram_power_watts(
        read_bytes_per_second=sram_read_bytes_per_second,
        write_bytes_per_second=sram_write_bytes_per_second,
    )
    leakage_w = logic_leakage_power_watts(
        logic_area_mm2=logic_area_mm2,
    )
    link_w = link_power_watts(
        transferred_bytes_per_second=link_bytes_per_second,
        link_generation=link_generation,
    )
    return DecodePower(
        memory_watts=memory_w * chip_count,
        compute_watts=compute_w * chip_count,
        sram_watts=sram_w * chip_count,
        leakage_watts=leakage_w * chip_count,
        link_watts=link_w,
        tokens_per_second=tokens_per_second,
        token_latency_s=token_latency_s,
        memory_source_label=technology.energy_source_label,
        energy_tier=ANALYTIC_ENERGY_TIER,
        energy_id=analytic_energy_identity(),
    )


def calibrate_reference_mac_energy(
    *,
    reference_total_watts: float | None = None,
) -> float:
    """Re-derive :data:`REFERENCE_MAC_ENERGY_PJ` from the literature anchor.

    Kept executable so the coefficient can be re-derived if the reference
    configuration or the memory coefficients change.
    """

    reference = REFERENCE_CONFIGURATION
    technology = hbm_technology(str(reference["hbm_generation"]))
    stacks = int(reference["hbm_stacks"])
    utilisation = float(reference["bandwidth_utilisation"])
    read_fraction = float(reference["read_fraction"])
    write_fraction = float(reference["write_fraction"])
    if abs(read_fraction + write_fraction - 1.0) > 1e-12:
        raise ValueError("reference read and write fractions must sum to one")
    peak = stacks * technology.peak_bandwidth_bytes_per_s_per_stack
    memory_w = memory_power_watts(
        technology,
        capacity_bytes=stacks * technology.capacity_gb_per_stack * 1e9,
        read_bytes_per_second=peak * utilisation * read_fraction,
        write_bytes_per_second=peak * utilisation * write_fraction,
    )
    total_w = (
        float(reference["reference_total_watts"])
        if reference_total_watts is None
        else reference_total_watts
    )
    rows, columns = reference["array"]
    macs_per_second = (
        rows
        * columns
        * float(reference["reference_clock_hz"])
        * float(reference["array_active_fraction"])
    )
    width_scale = (int(reference["operand_bits"]) / REFERENCE_MAC_BITS) ** 2
    return (total_w - memory_w) / macs_per_second / width_scale * 1e12


__all__ = [
    "ANALYTIC_ENERGY_TIER",
    "COMPUTE_ENERGY_CROSS_CHECK",
    "DC_ENERGY_TIER",
    "DecodePower",
    "ENERGY_TIERS",
    "LEAKAGE_SOURCE",
    "LINK_ENERGY_PJ_PER_BIT",
    "LINK_ENERGY_SOURCE",
    "LOGIC_LEAKAGE_W_PER_MM2",
    "MATRIX_MACHINE_LEAKAGE_MEASUREMENT",
    "REFERENCE_CONFIGURATION",
    "REFERENCE_MAC_BITS",
    "REFERENCE_MAC_ENERGY_PJ",
    "SRAM_ACCESS_ENERGY_PJ_PER_BIT",
    "SRAM_ENERGY_SOURCE",
    "analytic_energy_identity",
    "calibrate_reference_mac_energy",
    "compute_power_watts",
    "decode_power",
    "link_power_watts",
    "logic_leakage_power_watts",
    "memory_power_watts",
    "sram_power_watts",
]
