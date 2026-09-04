"""End-to-end Matrix-SRAM L-Compute campaign for the two hybrid models.

This campaign is intentionally independent from the older Vector-SRAM
``L_CFG`` experiments.  C is the executable single-base fixed-descriptor
lowering and D is the compact executable phased-descriptor lowering.  D' is a
separate, strongest fixed-wiring bank-only control: it gives every logical tile
an ordinary compiler-selected base phase.  Any capacity packing, issue-count,
and bank-service effects are reported separately rather than being collapsed
into a single "bank" number.

The executable point uses BF16 uniformly for Matrix-SRAM values, projection
fields and recurrent state.  The official GPU observation that both Nemotron
Mamba and Kimi KDA retain FP32 state is baseline evidence only; it is not
silently substituted into PLENA's storage geometry.  Each compiler-sized state
group is explicitly streamed from HBM into the existing Matrix SRAM, remains
there only for its deterministic recurrence steps, and is explicitly written
back.  This is scratchpad use, not a cache: there are no tags, hits, replacement
or implicit persistence.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .hybrid_lcompute_campaign import _cached_gpu_report, _generic_compute, _model, _scenario
from .hybrid_routing import RoutingProfile, load_pinned_nemotron_profile
from .matrix_state_residency import build_report as build_state_residency_report
from .nemotron3_workload import InferencePhase, Precision, StageWork, WorkloadReport


class MatrixVariant(StrEnum):
    A_ORIGINAL = "A_original_fixed"
    B_ARLO = "B_arlo_static"
    # Executable control with one base and one fixed-diagonal descriptor.  This
    # is intentionally not called the strongest fixed-hardware control: D'
    # below may assign a different ordinary base phase to each logical tile.
    C_FIXED = "C_fixed_single_base_descriptor"
    # Member names are retained for compatibility with older analysis code;
    # the frozen implementation keeps alpha=1 and programs only a tile phase.
    D_AFFINE = "D_compact_phased_mapping"
    E_AFFINE_OVERLAP = "E_phased_plus_static_overlap"


class StateMode(StrEnum):
    PLENA_BF16 = "plena_bf16_matrix_streamed"


@dataclass(frozen=True)
class MatrixHardwarePoint:
    mlen: int = 2048
    blen: int = 32
    banks: int = 64
    bank_width: int = 32
    hbm_bytes_per_cycle: int = 1560
    hbm_burst_bytes: int = 64
    clock_hz: int = 1_000_000_000
    vector_lanes: int = 2048
    exp_latency: int = 2
    reduction_latency: int = 8
    # Preserve the paper's approximately 1 MiB bit budget. Uniform BF16 gives
    # 256 MLEN-wide rows, matching the Compiler's MatrixSramPoint.
    matrix_sram_rows: int = 256
    matrix_element_bits: int = 16
    # One bank word is 32 BF16 values = 512 bits. No hidden port widening.
    matrix_bank_port_bits_per_cycle: int = 512
    reference_bank_port_bits_per_cycle: int = 512
    view_slots: int = 4

    def __post_init__(self) -> None:
        if self.mlen != self.banks * self.bank_width:
            raise ValueError("MLEN must equal banks * bank_width")
        if self.banks < 1 or self.banks & (self.banks - 1):
            raise ValueError("Matrix bank count must be a power of two")
        if min(self.blen, self.hbm_bytes_per_cycle, self.hbm_burst_bytes) <= 0:
            raise ValueError("hardware dimensions and bandwidth must be positive")
        if (
            min(
                self.matrix_bank_port_bits_per_cycle,
                self.reference_bank_port_bits_per_cycle,
            )
            <= 0
        ):
            raise ValueError("Matrix SRAM port widths must be positive")

    @property
    def matrix_macs_per_cycle(self) -> int:
        return self.mlen * self.blen

    @property
    def matrix_sram_bytes(self) -> int:
        return self.mlen * self.matrix_sram_rows * self.matrix_element_bits // 8

    @property
    def matrix_element_bytes(self) -> int:
        if self.matrix_element_bits % 8:
            raise ValueError("Matrix element width must contain whole bytes")
        return self.matrix_element_bits // 8

    @property
    def matrix_bank_word_bits(self) -> int:
        return self.bank_width * self.matrix_element_bits

    @property
    def matrix_bank_word_beats(self) -> int:
        return math.ceil(self.matrix_bank_word_bits / self.matrix_bank_port_bits_per_cycle)

    def resource_proxies(self) -> dict[str, Any]:
        skew_bits = int(math.log2(self.banks))
        bank_word_bits = self.matrix_bank_word_bits
        packet_bits = self.mlen * self.matrix_element_bits
        rotator_stages = skew_bits
        return {
            "scope": "pre-RTL structural proxy; not area, power, timing or PPA",
            "additional_sram_payload_bytes": 0,
            "additional_cache_tags_or_replacement_bits": 0,
            "additional_mac_lanes": 0,
            "configuration_register_bits": self.view_slots * 64,
            "configured_view_slots": self.view_slots,
            "l_tile_primitive_count": 3,
            "sequencer_state_bits_upper_bound": 256,
            "sequencer_loop_counters": 3,
            # Nemotron packs 32 independent 64-value rows into MLEN=2048;
            # Kimi needs 16 128-value rows. Report the worst supported case.
            "segment_scalar_broadcast_lanes": 32,
            "segment_scalar_width_bits": 16,
            "segment_scalar_broadcast_mux_input_bits": 32 * 16,
            "fused_primitives_reuse_existing_vector_mul_add": True,
            "skew_bits": skew_bits,
            # D' showed no value in a programmable row coefficient. Keep only
            # one compact inter-tile phase accumulator per configured view.
            "additional_programmable_bank_select_adders_upper_bound": 0,
            "programmable_bank_coefficient_bits": 0,
            "programmable_row_bank_coefficient": False,
            "tile_phase_accumulator_count_upper_bound": self.view_slots,
            "tile_phase_accumulator_width_bits": skew_bits,
            "tile_phase_accumulator_bits_upper_bound": self.view_slots * skew_bits,
            "fixed_diagonal_bank_select_adders_existing": self.banks,
            "fixed_diagonal_bank_select_adder_width_bits": skew_bits,
            "cyclic_lane_restore_bank_words": self.banks,
            "lane_restore_word_width_bits": bank_word_bits,
            "lane_restore_mux_stages": rotator_stages,
            "conservative_one_bit_mux_equivalents": (self.banks * rotator_stages * bank_word_bits),
            "additional_matrix_sram_read_ports_per_bank": 0,
            "additional_matrix_sram_write_ports_per_bank": 0,
            "matrix_bank_word_bits": bank_word_bits,
            "matrix_bank_port_bits_per_cycle": self.matrix_bank_port_bits_per_cycle,
            "matrix_bank_word_beats": self.matrix_bank_word_beats,
            "reference_bank_port_bits_per_cycle": self.reference_bank_port_bits_per_cycle,
            "incremental_port_bits_per_bank": max(
                0,
                self.matrix_bank_port_bits_per_cycle - self.reference_bank_port_bits_per_cycle,
            ),
            "incremental_port_bits_all_banks": self.banks
            * max(
                0,
                self.matrix_bank_port_bits_per_cycle - self.reference_bank_port_bits_per_cycle,
            ),
            "additional_operand_staging_bytes": 0,
            "existing_vector_operand_buffer_reused": True,
            "existing_matrix_to_vector_datapath_reused": True,
            "additional_payload_datapath_bits": 0,
            "maximum_operand_hold_bytes": packet_bits // 8,
            "matrix_to_vector_operand_staging_bits": packet_bits,
            "matrix_vector_bypass_payload_bits_per_cycle": packet_bits,
            "matrix_to_vector_operand_mux_bits": packet_bits,
            "vector_to_matrix_writeback_mux_bits": packet_bits,
            "matrix_sram_capacity_bytes": self.matrix_sram_bytes,
            "note": (
                "C uses PLENA's existing fixed diagonal bank mapping; D keeps that row "
                "mapping and compactly programs an inter-tile bank phase. The cyclic restore network "
                "reuses the existing one-word-per-bank row interface. With one read port "
                "per bank, a two-source Vector operation "
                "uses the existing Vector operand buffer to hold the first restored packet "
                "while the second packet arrives; no new operand SRAM is introduced. The "
                "existing Matrix-to-Vector and Vector-to-Matrix payload path is reused; "
                "only view selection, segment broadcast control and the L_TILE sequencer "
                "are modeled structural additions. A one-cycle 2048-value BF16 packet "
                "uses the reference 512-bit bank word. Synthesis "
                "is required before quoting LUTs, gates, frequency, power or area."
            ),
        }

    def resource_proxies_by_variant(self) -> dict[str, dict[str, Any]]:
        """Expose which structural cost belongs to each ablation row."""

        packet_bits = self.mlen * self.matrix_element_bits
        skew_bits = int(math.log2(self.banks))
        result: dict[str, dict[str, Any]] = {}
        for variant in MatrixVariant:
            packet_path = variant not in {
                MatrixVariant.A_ORIGINAL,
                MatrixVariant.B_ARLO,
            }
            programmable_phase = variant in {
                MatrixVariant.D_AFFINE,
                MatrixVariant.E_AFFINE_OVERLAP,
            }
            result[variant] = {
                "packetized_matrix_access": packet_path,
                "compiler_programmable_tile_pitch": packet_path,
                "compiler_programmable_tile_phase": programmable_phase,
                "compiler_programmable_alpha": False,
                "architectural_variant": packet_path,
                "configuration_register_bits": (self.view_slots * 64 if packet_path else 0),
                "additional_programmable_skew_address_adders_upper_bound": 0,
                "tile_phase_accumulator_count_upper_bound": (
                    self.view_slots if programmable_phase else 0
                ),
                "tile_phase_accumulator_width_bits": skew_bits,
                # The fixed diagonal mapper is PLENA prior work and exists in
                # every ablation row. It must not appear as an incremental
                # L-Compute resource only when packet mode is enabled.
                "fixed_diagonal_address_adders_existing": self.banks,
                "incremental_fixed_diagonal_address_adders": 0,
                "skew_adder_width_bits": skew_bits,
                "cyclic_lane_restore_payload_bits": packet_bits if packet_path else 0,
                "additional_operand_staging_bytes": 0,
                "existing_vector_operand_buffer_reused": packet_path,
                "existing_matrix_to_vector_datapath_reused": packet_path,
                "additional_payload_datapath_bits": 0,
                "sequencer_state_bits_upper_bound": 256 if packet_path else 0,
                "segment_scalar_broadcast_lanes": 32 if packet_path else 0,
                "maximum_operand_hold_bytes": packet_bits // 8 if packet_path else 0,
                "matrix_sram_read_ports_per_bank": 1,
                "matrix_sram_write_ports_per_bank": 1,
                "additional_matrix_sram_read_ports_per_bank": 0,
                "additional_matrix_sram_write_ports_per_bank": 0,
                "matrix_sram_capacity_bytes": self.matrix_sram_bytes,
                "matrix_bank_port_bits_per_cycle": self.matrix_bank_port_bits_per_cycle,
                "matrix_bank_word_beats": self.matrix_bank_word_beats,
                "layout_added_sram_payload_bytes": 0,
                "layout_added_spill_bytes": 0,
                "overlap_requires_runtime_scheduler": False,
            }
        return result


@dataclass(frozen=True)
class RecurrentPacketSpec:
    model: str
    compiler_key: str
    layer_type: str
    layers: int
    heads: int
    packet_heads: int
    elements_per_head: int
    recurrence_rows: int
    passes: int
    bf16_field_loads_per_row_pass: int

    @property
    def groups(self) -> int:
        if self.heads % self.packet_heads:
            raise ValueError(f"{self.model}: packet heads do not divide heads")
        return self.heads // self.packet_heads

    @property
    def words_per_head(self) -> int:
        return self.elements_per_head // 32

    @property
    def packet_values(self) -> int:
        return self.packet_heads * self.elements_per_head

    @property
    def row_passes(self) -> int:
        return self.groups * self.passes * self.recurrence_rows

    @property
    def bf16_field_packets(self) -> int:
        return self.row_passes * self.bf16_field_loads_per_row_pass


NEMOTRON_PACKET = RecurrentPacketSpec(
    model="nemotron3",
    compiler_key="nemotron_mamba_decode_recurrence",
    layer_type="mamba",
    layers=23,
    heads=64,
    # One BF16 packet fills all 2048 lanes: 32 heads x 64 values.
    packet_heads=32,
    elements_per_head=64,
    recurrence_rows=128,
    passes=1,
    bf16_field_loads_per_row_pass=3,
)

KIMI_PACKET = RecurrentPacketSpec(
    model="kimi_k3",
    compiler_key="kimi_k3_decode_recurrence",
    layer_type="kda",
    layers=69,
    heads=96,
    # One BF16 packet fills all 2048 lanes: 16 heads x 128 values.
    packet_heads=16,
    elements_per_head=128,
    recurrence_rows=128,
    passes=2,
    bf16_field_loads_per_row_pass=2,
)

PACKETS = {spec.model: spec for spec in (NEMOTRON_PACKET, KIMI_PACKET)}


EVIDENCE_LEVELS = {
    "isa_and_banked_matrix_sram": ("Executable Compiler encoding and Rust transactional implementation."),
    "connected_projection": (
        "MLEN=64 synthetic numerical Matrix projection -> fixed-diagonal, "
        "Compiler-pitched writeback -> "
        "view-qualified consumer; exact output comparison."
    ),
    "multi_token_recurrence": (
        "Four-token official recurrence geometry (Nemotron 64x128x64 and Kimi "
        "96x128x128) compiled to machine words and numerically executed by Rust, "
        "with explicit HBM -> BF16 Matrix SRAM -> HBM state and output."
    ),
    "official_shape_packets": (
        "Physical 2048-value Matrix-SRAM replay using official head and row "
        "dimensions and exact Compiler-emitted addresses. C is the executable "
        "single-base fixed descriptor; D is the compact phased descriptor; D' is "
        "the strongest fixed-wiring bank-only control."
    ),
    "full_model_timeline": (
        "Official 52/93-layer structure, tensor dimensions, measured GPU calibration "
        "and symbolic PLENA weights in a resource-accounted analytic timeline. Every "
        "Mamba/KDA layer consumes the measured Compiler L_TILE lowering, but this remains "
        "formula-based rather than first-to-last-layer numerical Rust execution."
    ),
    "not_demonstrated": (
        "No real-weight first-to-last-layer Rust execution for Nemotron 3 or Kimi K3; "
        "no RTL timing, synthesis, PPA or silicon energy measurement."
    ),
}


def load_connected_recurrence_evidence(
    simulator_root: Path,
    *,
    expected_contract_version: int,
) -> dict[str, Any]:
    """Load the checked Compiler-to-Rust official-geometry BF16 results."""

    path = simulator_root / "artifacts" / "matrix_lcompute_connected_bf16" / "summary.json"
    if not path.exists():
        return {
            "status": "NOT_RUN",
            "source": str(path),
            "required_command": "just test-matrix-lcompute-recurrence <compiler-root>",
        }
    payload = json.loads(path.read_text())
    expected = {
        ("nemotron3_mamba2", "fixed"),
        ("nemotron3_mamba2", "affine"),
        ("kimi_k3_kda", "fixed"),
        ("kimi_k3_kda", "affine"),
    }
    actual = {(str(record["model"]), str(record["layout"])) for record in payload}
    if actual != expected:
        raise ValueError(f"connected BF16 result coverage differs: {actual} != {expected}")
    for record in payload:
        if int(record.get("schema_version", 0)) != 2:
            raise ValueError("connected recurrence result does not use schema v2")
        if int(record.get("matrix_view_contract_version", 0)) != expected_contract_version:
            raise ValueError(
                "connected recurrence Matrix-view contract differs from the active Compiler"
            )
        if record.get("precision") != "bf16_uniform_matrix_recurrence":
            raise ValueError("connected recurrence result is not uniform BF16")
        if int(record.get("tokens", 0)) != 4:
            raise ValueError("connected recurrence result must cover four tokens")
        if int(record.get("l_tile_exec_count", 0)) <= 0:
            raise ValueError("connected recurrence result contains no L_TILE_EXEC")
        for field in ("assembly_sha256", "machine_code_sha256", "input_hbm_sha256"):
            value = str(record.get(field, ""))
            if len(value) != 64:
                raise ValueError(f"connected recurrence result has invalid {field}")
    return {
        "status": "EXECUTED_AND_NUMERICALLY_VERIFIED",
        "source": str(path.relative_to(simulator_root)),
        "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "records": payload,
    }


def build_round3_triage() -> list[dict[str, str]]:
    """Current audit state; retained under the old key for artifact compatibility."""

    return [
        {
            "finding": "Fixed and compact-phased controls must have the same model and SRAM budget",
            "status": "ENFORCED",
            "disposition": (
                "Both lowerings use the official formula, head count, PLENA BF16 state, one-MiB "
                "Matrix SRAM, 64 one-read/one-write banks, and the same Vector arithmetic. "
                "C is explicitly labelled a single-base descriptor control. The stronger D' "
                "uses fixed diagonal wiring plus per-tile base phases and is the bank-only "
                "control. Issue/spill/chunk costs are reported separately."
            ),
        },
        {
            "finding": "A programmable row coefficient needs an architectural justification",
            "status": "NOT_JUSTIFIED_BY_BANK_SERVICE",
            "disposition": (
                "The executable phased descriptor removes stalls relative to C, but D' maps "
                "the same official state coordinates with fixed wiring and legal base phases. "
                "D over D' is 1.00x for bank service, so the arbitrary row coefficient was "
                "removed before RTL. The remaining tile-phase stride is compact descriptor/issue "
                "encoding and is evaluated under that name."
            ),
        },
        {
            "finding": "The executable precision must not silently widen the Matrix SRAM port",
            "status": "FIXED_BY_UNIFORM_BF16",
            "disposition": (
                "The PLENA point uses 32 BF16 values per 512-bit bank word, exactly matching "
                "the reference width. Official GPU FP32 state remains baseline metadata."
            ),
        },
        {
            "finding": "The old projection/recurrence full-overlap subtraction violated dependencies",
            "status": "WITHDRAWN",
            "disposition": (
                "E receives no overlap credit until the Compiler emits a capacity-legal "
                "head-group double-buffer schedule and the Simulator replays it. Projection "
                "must complete before its dependent recurrence consumes the data."
            ),
        },
        {
            "finding": "Column-lane restoration and 32-bit state packing were incorrect",
            "status": "FIXED",
            "disposition": (
                "Column reads now restore the selected lane for every bank word. The executable "
                "design is uniformly BF16, so no FP32 Matrix-state packing is used. Positive "
                "and negative Rust tests cover lane restoration."
            ),
        },
        {
            "finding": "Ordinary Attention/MoE Matrix accesses could regress",
            "status": "CHECKED_AT_ALL_BASE_PHASES",
            "disposition": (
                "Row and column accesses are replayed at all 64 allocation base phases. "
                "Variants retain identical service cycles on ordinary stages."
            ),
        },
        {
            "finding": "Complete-model speedup is not transactional numerical execution",
            "status": "BOUNDARY",
            "disposition": (
                "The 52/93-layer results are analytic schedules using official dimensions, "
                "Compiler-emitted recurrence streams, measured routing/calibration, and "
                "symbolic weights. Real-weight first-to-last Rust execution is not claimed."
            ),
        },
        {
            "finding": "Prior prefill handoff speedups used incomparable denominators",
            "status": "WITHDRAWN",
            "disposition": (
                "Only the identity-GEMM instruction/MAC census and exact numbered-value "
                "transpose equivalence remain. No prefill latency speedup is reported."
            ),
        },
    ]


def _physical_word(
    *,
    tile: int,
    row: int,
    word: int,
    spec: RecurrentPacketSpec,
    hardware: MatrixHardwarePoint,
    alpha: int,
    gamma: int,
    base_bank_row: int = 0,
    tile_pitch_rows: int = 1,
    tile_phase_stride: int = 0,
) -> tuple[int, int]:
    words_per_row = spec.elements_per_head // hardware.bank_width
    row_groups = math.ceil(words_per_row / hardware.banks)
    bank_row = base_bank_row + tile * tile_pitch_rows + row * row_groups + word // hardware.banks
    bank = (
        alpha * bank_row
        + tile_phase_stride * tile
        + gamma * (bank_row // hardware.banks)
        + word
    ) % hardware.banks
    return bank, bank_row


def measure_packet(
    spec: RecurrentPacketSpec,
    *,
    hardware: MatrixHardwarePoint | None = None,
    alpha: int = 1,
    gamma: int = 1,
    base_bank_row: int = 0,
    row: int = 0,
    tile_pitch_rows: int = 1,
    tile_phase_stride: int = 0,
) -> dict[str, Any]:
    """Move one real paper-shape packet through numbered physical bank words."""

    hardware = hardware or MatrixHardwarePoint()
    if not 0 < spec.packet_values <= hardware.mlen:
        raise ValueError(
            f"{spec.model}: packet has {spec.packet_values} values, outside one MLEN={hardware.mlen} issue"
        )
    if spec.elements_per_head % hardware.bank_width:
        raise ValueError(f"{spec.model}: head row does not contain whole bank words")
    words_per_head = spec.elements_per_head // hardware.bank_width
    if not 0 <= alpha < hardware.banks or not 0 <= gamma < hardware.banks:
        raise ValueError("alpha and gamma must fit the Matrix bank index")
    if not 0 <= tile_phase_stride < hardware.banks:
        raise ValueError("tile phase stride must fit the Matrix bank index")
    row_groups = math.ceil(words_per_head / hardware.banks)
    if tile_pitch_rows < 0:
        raise ValueError("tile pitch must be non-negative")

    cells: dict[tuple[int, int], tuple[int, ...]] = {}
    expected: list[tuple[int, ...]] = []
    bank_load: Counter[int] = Counter()
    for tile in range(spec.packet_heads):
        for word in range(words_per_head):
            logical = tuple(
                tile * spec.elements_per_head + word * hardware.bank_width + lane for lane in range(hardware.bank_width)
            )
            coord = _physical_word(
                tile=tile,
                row=row,
                word=word,
                spec=spec,
                hardware=hardware,
                alpha=alpha,
                gamma=gamma,
                base_bank_row=base_bank_row,
                tile_pitch_rows=tile_pitch_rows,
                tile_phase_stride=tile_phase_stride,
            )
            if coord in cells:
                raise AssertionError(f"physical Matrix word aliases at {coord}")
            cells[coord] = logical
            expected.append(logical)
            bank_load[coord[0]] += 1

    restored = []
    for tile in range(spec.packet_heads):
        for word in range(words_per_head):
            coord = _physical_word(
                tile=tile,
                row=row,
                word=word,
                spec=spec,
                hardware=hardware,
                alpha=alpha,
                gamma=gamma,
                base_bank_row=base_bank_row,
                tile_pitch_rows=tile_pitch_rows,
                tile_phase_stride=tile_phase_stride,
            )
            restored.append(cells[coord])
    if restored != expected:
        raise AssertionError("inverse lane restoration changed logical packet order")

    wrong_layout_detected = False
    wrong_tile_phase = (tile_phase_stride + 1) % hardware.banks
    try:
        wrong = [
            cells[
                _physical_word(
                    tile=tile,
                    row=row,
                    word=word,
                    spec=spec,
                    hardware=hardware,
                    alpha=alpha,
                    gamma=gamma,
                    base_bank_row=base_bank_row,
                    tile_pitch_rows=tile_pitch_rows,
                    tile_phase_stride=wrong_tile_phase,
                )
            ]
            for tile in range(spec.packet_heads)
            for word in range(words_per_head)
        ]
        wrong_layout_detected = wrong != expected
    except KeyError:
        wrong_layout_detected = True
    bank_words = spec.packet_heads * words_per_head
    service = max(bank_load.values(), default=0)
    ideal = math.ceil(bank_words / hardware.banks)
    return {
        "model": spec.model,
        "packet": {
            "tiles": spec.packet_heads,
            "elements_per_tile": spec.elements_per_head,
            "values": spec.packet_values,
            "bank_words": bank_words,
        },
        "map": {
            "alpha": alpha,
            "gamma": gamma,
            "tile_pitch_rows": tile_pitch_rows,
            "tile_phase_stride": tile_phase_stride,
        },
        "service_cycles": service,
        "ideal_cycles": ideal,
        "bank_stall_cycles": service - ideal,
        "banks_touched": len(bank_load),
        "worst_bank_words": service,
        "roundtrip_values_checked": spec.packet_values,
        "wrong_mapping_changes_data": wrong_layout_detected,
        "packet_occupancy_bytes": spec.packet_values * hardware.matrix_element_bits // 8,
        "packet_physical_span_rows": ((spec.packet_heads - 1) * tile_pitch_rows + row_groups),
        "packet_physical_span_bytes": (
            ((spec.packet_heads - 1) * tile_pitch_rows + row_groups) * hardware.mlen * hardware.matrix_element_bits // 8
        ),
    }


def measure_fixed_phased_packet(
    spec: RecurrentPacketSpec,
    *,
    hardware: MatrixHardwarePoint | None = None,
) -> dict[str, Any]:
    """Measure D': the strongest control on PLENA's existing fixed wiring.

    The control has no compact programmable tile-phase stride. Instead, the
    compiler gives each logical head tile an ordinary column-aligned base
    phase.  For the official packets the phases are ``tile * words_per_head``;
    all 64 bank words of one recurrent row therefore occupy different banks.

    This representation needs one base/view binding per tile, so it is not the
    compact executable C descriptor.  It is nevertheless the mandatory fair
    *bank-only* control: a compact phased mapper cannot claim a conflict
    benefit that fixed wiring plus legal compiler placement already obtains.
    """

    hardware = hardware or MatrixHardwarePoint()
    if spec.packet_values != hardware.mlen:
        raise ValueError(f"{spec.model}: fixed-phased control expects one full MLEN packet")
    if spec.elements_per_head % hardware.bank_width:
        raise ValueError(f"{spec.model}: head row does not contain whole bank words")
    words_per_head = spec.elements_per_head // hardware.bank_width
    if spec.packet_heads * words_per_head != hardware.banks:
        raise ValueError(f"{spec.model}: fixed phases require exactly one bank word per bank")

    cells: dict[tuple[int, int], tuple[int, int, int]] = {}
    packet_load: Counter[int] = Counter()
    for row in range(spec.recurrence_rows):
        for tile in range(spec.packet_heads):
            base_bank = tile * words_per_head
            for word in range(words_per_head):
                # Existing PLENA diagonal wiring: bank = base + physical row + word.
                bank = (base_bank + row + word) % hardware.banks
                coord = (bank, row)
                affine_coord = (
                    (row + words_per_head * tile + word) % hardware.banks,
                    row,
                )
                if coord != affine_coord:
                    raise AssertionError("fixed per-tile phase does not reproduce compact phased placement")
                logical = (row, tile, word)
                previous = cells.setdefault(coord, logical)
                if previous != logical:
                    raise AssertionError(f"fixed-phased placement aliases {logical} with {previous} at {coord}")
                if row == 0:
                    packet_load[bank] += 1

    service = max(packet_load.values(), default=0)
    ideal = math.ceil(sum(packet_load.values()) / hardware.banks)
    return {
        "model": spec.model,
        "mapping": {
            "fixed_alpha": 1,
            "fixed_gamma": 0,
            "programmable_skew": False,
            "per_tile_base_bank_phase": "tile * words_per_head",
            "base_bindings": spec.packet_heads,
        },
        "service_cycles": service,
        "ideal_cycles": ideal,
        "bank_stall_cycles": service - ideal,
        "banks_touched": len(packet_load),
        "roundtrip_values_checked": len(cells) * hardware.bank_width,
        "physical_rows": spec.recurrence_rows,
        "capacity_bytes": (spec.recurrence_rows * hardware.mlen * hardware.matrix_element_bytes),
        "evidence": (
            "complete official-shape state coordinates on fixed diagonal wiring; "
            "compiler-selected ordinary base phase per logical tile"
        ),
        "same_physical_coordinates_as_compact_tile_phase": True,
        # Compatibility key retained for older report readers.
        "same_physical_coordinates_as_affine_tile_skew": True,
        "compact_phase_vs_explicit_bases_bank_speedup": 1.0,
    }


def measure_matrix_line(
    *,
    axis: str,
    rows: int,
    cols: int,
    hardware: MatrixHardwarePoint,
    alpha: int,
    gamma: int,
    index: int = 0,
    base_bank_row: int = 0,
) -> dict[str, Any]:
    """Move one ordinary Matrix row/column through the same diagonal cells."""

    if cols % hardware.bank_width:
        raise ValueError("Matrix line width must contain complete bank words")
    if axis not in {"row", "column"}:
        raise ValueError(f"unknown Matrix line axis {axis}")
    if axis == "row" and not 0 <= index < rows:
        raise ValueError("row index out of range")
    if axis == "column" and not 0 <= index < cols:
        raise ValueError("column index out of range")

    positions = (
        [(index, col) for col in range(0, cols, hardware.bank_width)]
        if axis == "row"
        else [(row, index - index % hardware.bank_width) for row in range(rows)]
    )
    cells: dict[tuple[int, int], tuple[int, int]] = {}
    expected: list[tuple[int, int]] = []
    loads: Counter[int] = Counter()
    words_per_row = cols // hardware.bank_width
    row_groups = math.ceil(words_per_row / hardware.banks)
    for row, word_col in positions:
        word = word_col // hardware.bank_width
        bank_row = base_bank_row + row * row_groups + word // hardware.banks
        bank = (alpha * bank_row + gamma * (bank_row // hardware.banks) + word) % hardware.banks
        logical = (row, word_col)
        coord = (bank, bank_row)
        if coord in cells:
            raise AssertionError(f"ordinary Matrix line aliases at {coord}")
        cells[coord] = logical
        expected.append(logical)
        loads[bank] += 1
    restored = [
        cells[
            (
                (
                    alpha * (base_bank_row + row * row_groups + (word_col // hardware.bank_width) // hardware.banks)
                    + gamma
                    * (
                        (base_bank_row + row * row_groups + (word_col // hardware.bank_width) // hardware.banks)
                        // hardware.banks
                    )
                    + word_col // hardware.bank_width
                )
                % hardware.banks,
                base_bank_row + row * row_groups + (word_col // hardware.bank_width) // hardware.banks,
            )
        ]
        for row, word_col in positions
    ]
    if restored != expected:
        raise AssertionError("ordinary Matrix lane restoration changed value order")
    service = max(loads.values(), default=0)
    ideal = math.ceil(len(positions) / hardware.banks)
    return {
        "axis": axis,
        "logical_values_checked": cols if axis == "row" else rows,
        "bank_words_checked": len(positions),
        "ideal_cycles": ideal,
        "service_cycles": service,
        "bank_stall_cycles": service - ideal,
    }


def build_ordinary_no_regression_evidence(
    *,
    compiler: dict[str, Any],
    physical: dict[str, Any],
    hardware: MatrixHardwarePoint,
) -> dict[str, Any]:
    fixed = physical["global_fixed_map"]
    maps = {
        MatrixVariant.C_FIXED: (int(fixed["alpha"]), int(fixed["gamma"])),
        # Ordinary Attention/MoE accesses do not select a phased recurrence
        # view, so D/E retain their exact original mapping and timing.
        MatrixVariant.D_AFFINE: (int(fixed["alpha"]), int(fixed["gamma"])),
        MatrixVariant.E_AFFINE_OVERLAP: (
            int(fixed["alpha"]),
            int(fixed["gamma"]),
        ),
    }
    records = {
        variant: {
            access: {
                "allocation_phases_checked": hardware.banks,
                "service_cycles": max(record["service_cycles"] for record in phase_records),
                "ideal_cycles": max(record["ideal_cycles"] for record in phase_records),
                "bank_stall_cycles": max(record["bank_stall_cycles"] for record in phase_records),
                "logical_values_checked_per_phase": phase_records[0]["logical_values_checked"],
            }
            for access, phase_records in {
                "moe_and_projection_row": [
                    measure_matrix_line(
                        axis="row",
                        rows=128,
                        cols=hardware.mlen,
                        hardware=hardware,
                        alpha=alpha,
                        gamma=gamma,
                        base_bank_row=base,
                    )
                    for base in range(hardware.banks)
                ],
                "attention_qkt_column": [
                    measure_matrix_line(
                        axis="column",
                        rows=128,
                        cols=hardware.mlen,
                        hardware=hardware,
                        alpha=alpha,
                        gamma=gamma,
                        base_bank_row=base,
                    )
                    for base in range(hardware.banks)
                ],
            }.items()
        }
        for variant, (alpha, gamma) in maps.items()
    }
    for access in ("moe_and_projection_row", "attention_qkt_column"):
        cycles = {record[access]["service_cycles"] for record in records.values()}
        if len(cycles) != 1:
            raise AssertionError(f"ordinary Matrix {access} regressed across layouts")
    stages = [
        case["stage"]
        for case in compiler["real_packets"]["cases"]
        if case["stage"]
        in {
            "gqa_attention_qkt",
            "mla_attention_qkt",
            "moe_gate_projection",
            "latent_moe_gate_projection",
        }
    ]
    return {
        "source_stages": stages,
        "records": records,
        "all_service_cycles_identical": True,
        "values_checked": {
            "per_row": hardware.mlen * hardware.banks,
            "per_column": 128 * hardware.banks,
        },
        "allocation_base_phases_checked": hardware.banks,
    }


def build_physical_evidence(
    hardware: MatrixHardwarePoint | None = None,
) -> dict[str, Any]:
    """Describe executable mappings and the mandatory fair bank-only control.

    C is an executable *single-base descriptor* control. D/E retain PLENA's
    fixed diagonal row map and add a compact per-view tile-phase stride. D' is
    separately evaluated with PLENA's fixed
    diagonal wiring plus one legal compiler-selected base phase per tile.  D'
    is the strongest bank-only control; C is not.  Therefore C-to-D may expose
    descriptor/chunk/issue differences, while only D'-to-D can be called a
    programmable-skew bank comparison.
    """

    hardware = hardware or MatrixHardwarePoint()
    fixed_phased = {spec.model: measure_fixed_phased_packet(spec, hardware=hardware) for spec in PACKETS.values()}
    return {
        "degrees_of_freedom": {
            "C": {
                "role": "executable single-base fixed descriptor",
                "alpha": 1,
                "tile_phase_stride": 0,
                "global_gamma": 0,
                "compiler_controls": [
                    "base_bank_phase",
                    "tile_pitch_rows",
                    "group_phase",
                    "chunking",
                ],
            },
            "D_prime": {
                "role": "strongest fixed-wiring bank-only control",
                "alpha": 1,
                "tile_phase_stride": 0,
                "global_gamma": 0,
                "compiler_controls": [
                    "per_tile_base_bank_phase",
                    "tile_pitch_rows",
                    "group_phase",
                    "chunking",
                ],
                "compact_single_descriptor": False,
            },
            "D": {
                "role": "compact executable phased descriptor",
                "alpha": 1,
                "global_gamma": 0,
                "compiler_controls": [
                    "base_bank_phase",
                    "tile_pitch_rows",
                    "tile_phase_stride",
                    "chunking",
                ],
            },
            "fairness_check": (
                "same capacity, banks, ports, arithmetic and model formulas; "
                "D' owns the same physical placement freedom; D encodes per-tile "
                "phases compactly in one descriptor"
            ),
        },
        "global_fixed_map": {
            "alpha": 1,
            "gamma": 0,
            "source": "PLENA fixed diagonal Matrix-SRAM wiring",
        },
        "fixed_phased_bank_control": fixed_phased,
        "comparison_source": (
            "C versus D is executable descriptor/chunk/issue evidence. D' versus D "
            "is the fair physical bank comparison. For both official state packets, "
            "D' and D occupy identical bank coordinates and have 1.00x bank speedup."
        ),
        "hardware": asdict(hardware),
    }


@functools.lru_cache(maxsize=1)
def load_compiler_evidence(compiler_root: str) -> dict[str, Any]:
    root = Path(compiler_root).resolve()
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    loaded = sys.modules.get("compiler")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        if not loaded_file.is_relative_to(root):
            raise RuntimeError(
                "another PLENA Compiler checkout is already loaded in this Python "
                f"process ({loaded_file}); requested {root}. Start a fresh process "
                "with PLENA_COMPILER_ROOT set before pytest collection."
            )
    previous_settings = os.environ.get("PLENA_SETTINGS_TOML")
    os.environ["PLENA_SETTINGS_TOML"] = str(settings)
    sys.path.insert(0, str(root))
    try:
        from compiler.aten.plena.hybrid_compile_report import build_report as build_issue_report
        from compiler.aten.plena.hybrid_l_tile_schedule import (
            build_official_hybrid_l_tile_report,
        )
        from compiler.aten.plena.matrix_packet_report import build_report as build_packet_report
        from compiler.aten.plena.matrix_prefill_handoff import (
            build_prefill_handoff_report,
        )
        from compiler.aten.plena.matrix_recurrence_lowering import (
            build_matrix_recurrence_report,
        )
        from compiler.aten.plena.mview import L_MVIEW_CONTRACT_VERSION

        issue = build_issue_report(
            root / "doc/Model_Lib",
            packet_elements=2048,
            storage_atom=32,
            banks=64,
            bank_width=32,
            blen=32,
            mamba_recurrent_row_elements=64,
            kda_recurrent_row_elements=128,
        )
        return {
            "compiler_root": str(root),
            "plena_settings_toml": str(settings),
            "matrix_view_contract_version": L_MVIEW_CONTRACT_VERSION,
            "issue": issue,
            "hybrid_l_tile_schedule": build_official_hybrid_l_tile_report(root / "doc/Model_Lib"),
            "matrix_recurrence": build_matrix_recurrence_report(),
            "real_packets": build_packet_report(),
            "prefill_handoff": build_prefill_handoff_report(),
        }
    finally:
        sys.path.pop(0)
        if previous_settings is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = previous_settings


def _issue_counts(compiler: dict[str, Any], spec: RecurrentPacketSpec) -> dict[str, int]:
    old = compiler["issue"]["assembly"][spec.compiler_key]
    return {
        "A": int(old["baseline"]["dynamic_issued_instructions"]),
        "B": int(old["postincrement_only"]["dynamic_issued_instructions"]),
    }


def _validate_real_packet_contract(
    compiler: dict[str, Any],
    spec: RecurrentPacketSpec,
    *,
    co_layout: bool,
) -> dict[str, Any]:
    """Validate the executable Compiler packet stream consumed by the DSE.

    The Rust sequencer has one Matrix-SRAM read port per bank.  Destination,
    source and scalar reads are therefore three ordered phases, followed by a
    write phase; treating them as one co-issued packet invents bandwidth that
    the implementation does not have.  This guard also proves that every
    packet carries the complete phased mapping, including its tile phase.
    """

    stage = "nemotron3_mamba2_matrix_recurrence" if spec.model == "nemotron3" else "kimi_k3_kda_matrix_recurrence"
    lowering = "matrix_recurrence_affine" if co_layout else "matrix_recurrence_fixed"
    matches = [
        case for case in compiler["real_packets"]["cases"] if case["stage"] == stage and case["lowering"] == lowering
    ]
    if len(matches) != 1:
        raise AssertionError(f"{spec.model}: expected one {lowering} packet case")
    case = matches[0]
    l_tile_groups = [entry for entry in case["coissued_histogram"] if entry["opcode"] == "L_TILE_EXEC"]
    expected_phases = {
        ("read", "l_tile_dst_read"),
        ("read", "l_tile_source_read"),
        ("read", "l_tile_scale_read"),
        ("write", "l_tile_dst_write"),
    }
    observed_phases = {(str(entry["direction"]), str(entry["axis"])) for entry in l_tile_groups}
    if observed_phases != expected_phases:
        raise AssertionError(f"{spec.model}: L_TILE phases {observed_phases} != {expected_phases}")
    if any(int(entry["same_cycle_operands"]) != 1 for entry in l_tile_groups):
        raise AssertionError(f"{spec.model}: sequential one-read-port L_TILE phase was co-issued")

    allocations = case["working_set"]["allocations"]
    for group in case["service_groups"]:
        if group["opcode"] not in {"L_TILE_EXEC", "H_PREFETCH_V.MV", "H_STORE_V.MV"}:
            continue
        operands = group["operands"]
        if len(operands) != 1:
            raise AssertionError(f"{spec.model}: physical phase has {len(operands)} operands")
        operand = operands[0]
        matches = [
            allocation
            for allocation in allocations
            if allocation["base"] == operand["matrix_address"]
            and allocation["shape"]["rows"] == operand["view_rows"]
            and allocation["shape"]["cols"] == operand["view_cols"]
            and allocation["mapping"]["tile_pitch_rows"] == operand["tile_pitch_rows"]
        ]
        if len(matches) != 1:
            raise AssertionError(f"{spec.model}: packet does not name one working-set allocation: {operand}")
        allocation = matches[0]
        mapping = allocation["mapping"]
        phased = int(mapping["tile_phase_stride"]) != 0
        expected_alpha = 1
        expected_tile_phase = int(mapping["tile_phase_stride"])
        if (
            operand["view_phased"] is not phased
            or int(operand["view_alpha"]) != expected_alpha
            or int(operand["view_tile_phase_stride"]) != expected_tile_phase
        ):
            raise AssertionError(f"{spec.model}: packet lost its frozen phased mapping: {operand}")
        if int(operand["tiles"]) > int(allocation["shape"]["tile_count"]):
            raise AssertionError(f"{spec.model}: packet exceeds configured tile count")

    state_values = spec.heads * spec.recurrence_rows * spec.elements_per_head
    metrics = case["lowering_metrics"]
    if int(metrics["state_transfer_values"]) < 2 * state_values:
        raise AssertionError(f"{spec.model}: state is not explicitly loaded and stored")
    return {
        "stage": stage,
        "lowering": lowering,
        "physical_l_tile_phases": l_tile_groups,
        "dynamic_packet_repeats": case["dynamic_packet_repeats"],
        "source": case["source"],
        "service_groups": case["service_groups"],
        "working_set": case["working_set"],
        "lowering_metrics": metrics,
    }


def measure_real_service_groups(
    *,
    compiler: dict[str, Any],
    spec: RecurrentPacketSpec,
    variant: MatrixVariant,
    state_mode: StateMode,
    hardware: MatrixHardwarePoint,
    fixed_alpha: int,
    fixed_gamma: int,
) -> dict[str, Any]:
    """Replay the exact ordered SRAM phases emitted by the Compiler.

    This mirrors ``Accelerator::execute_l_tile`` rather than assuming a wider
    machine: destination, source and compact-scalar reads are sequential, the
    existing Vector ALU executes one primitive, and destination writeback is a
    fourth phase.  DMA packets also traverse the same physical bank mapper.
    """

    phased_mapping = variant in {
        MatrixVariant.D_AFFINE,
        MatrixVariant.E_AFFINE_OVERLAP,
    }
    contract = _validate_real_packet_contract(
        compiler,
        spec,
        co_layout=phased_mapping,
    )
    groups = contract["service_groups"]
    total_logical_values = 0
    total_bank_words = 0
    total_operands = 0
    total_groups = 0
    service_cycles = 0
    ideal_cycles = 0
    vector_arithmetic_cycles = 0
    worst_group_service = 0
    service_histogram: Counter[tuple[str, str, str, int, int]] = Counter()
    allocations_by_base = {
        int(allocation["base"]): str(allocation["name"]) for allocation in contract["working_set"]["allocations"]
    }

    def physical_coord(operand: dict[str, Any], *, tile: int, row: int, word: int) -> tuple[int, int]:
        address = operand["matrix_address"]
        if not isinstance(address, int) or address % hardware.bank_width:
            raise AssertionError(f"{spec.model}: Matrix address {address} is not bank-word aligned")
        base_bank_row = address // hardware.mlen
        base_bank = (address % hardware.mlen) // hardware.bank_width
        words_per_row = int(operand["view_cols"]) // hardware.bank_width
        row_groups = math.ceil(words_per_row / hardware.banks)
        bank_row = base_bank_row + tile * int(operand["tile_pitch_rows"]) + row * row_groups + word // hardware.banks
        if phased_mapping:
            alpha = fixed_alpha
            if int(operand["view_alpha"]) != fixed_alpha:
                raise AssertionError(
                    f"{spec.model}: executable view changed fixed diagonal alpha"
                )
            tile_phase_stride = int(operand["view_tile_phase_stride"])
        else:
            # C is the executable single-base fixed descriptor. Base phase,
            # pitch and chunking come from its own Compiler lowering. D' is
            # evaluated separately as the strongest bank-only control.
            alpha = fixed_alpha
            tile_phase_stride = 0
        bank = (
            base_bank
            + alpha * bank_row
            + tile_phase_stride * tile
            + fixed_gamma * (bank_row // hardware.banks)
            + word
        ) % hardware.banks
        return bank, bank_row

    def requested_words(
        group: dict[str, Any], operand: dict[str, Any], repeat: int
    ) -> tuple[list[tuple[int, int, int]], int]:
        rows = int(operand["view_rows"])
        cols = int(operand["view_cols"])
        tiles = int(operand["tiles"])
        words_per_row = cols // hardware.bank_width
        if group["axis"] == "view_dma":
            return (
                [(tile, row, word) for tile in range(tiles) for row in range(rows) for word in range(words_per_row)],
                tiles * rows * cols,
            )

        line_axis = operand["view_line_axis"]
        line_period = operand["view_line_period"]
        tile_start = operand["view_tile_start"]
        if line_axis not in {"row", "column"} or not isinstance(line_period, int):
            raise AssertionError(f"{spec.model}: incomplete L_TILE line metadata")
        if not isinstance(tile_start, int) or line_period <= 0:
            raise AssertionError(f"{spec.model}: invalid L_TILE line metadata")
        line = repeat % line_period
        broadcast_tile = bool(operand["view_broadcast_tile"])
        packet_tiles = [0 if broadcast_tile else tile_start + offset for offset in range(tiles)]
        if line_axis == "row":
            return (
                [(tile, line, word) for tile in packet_tiles for word in range(words_per_row)],
                tiles * cols,
            )
        return (
            [(tile, row, line // hardware.bank_width) for tile in packet_tiles for row in range(rows)],
            tiles * rows,
        )

    for group in groups:
        operands = list(group["operands"])
        if len(operands) != 1:
            raise AssertionError(f"{spec.model}: one-read-port physical phase has {len(operands)} operands")
        repeats = int(group["repeats"])
        total_groups += repeats
        for repeat in range(repeats):
            bank_load: Counter[int] = Counter()
            logical_values_this_group = 0
            for operand in operands:
                stride = operand["address_stride_elements"]
                if not isinstance(stride, int) or stride != 0:
                    raise AssertionError(f"{spec.model}: L_TILE lowering must use invariant Matrix bases")
                cols = int(operand["view_cols"])
                if cols % hardware.bank_width:
                    raise AssertionError("Matrix view row is not a whole number of bank words")
                words, logical_values = requested_words(group, operand, repeat)
                logical_values_this_group += logical_values
                cells: dict[tuple[int, int], tuple[int, int, int]] = {}
                for logical in words:
                    coord = physical_coord(operand, tile=logical[0], row=logical[1], word=logical[2])
                    previous = cells.setdefault(coord, logical)
                    if previous != logical:
                        raise AssertionError(f"{spec.model}: Matrix view aliases {previous} and {logical} at {coord}")
                for bank, _bank_row in cells:
                    bank_load[bank] += 1

            service = max(bank_load.values(), default=0)
            words_this_group = len(cells)
            ideal = math.ceil(words_this_group / hardware.banks)
            # A narrower sensitivity port may require multiple physical beats.
            service_cycles += service * hardware.matrix_bank_word_beats
            ideal_cycles += ideal * hardware.matrix_bank_word_beats
            total_bank_words += words_this_group
            total_logical_values += logical_values_this_group
            total_operands += len(operands)
            worst_group_service = max(
                worst_group_service,
                service * hardware.matrix_bank_word_beats,
            )
            operand = operands[0]
            allocation_name = allocations_by_base[int(operand["matrix_address"])]
            service_histogram[
                (
                    str(group["axis"]),
                    str(operand.get("l_tile_primitive") or "DMA"),
                    allocation_name,
                    ideal * hardware.matrix_bank_word_beats,
                    service * hardware.matrix_bank_word_beats,
                )
            ] += 1

        if group["axis"] == "l_tile_source_read":
            primitive = operands[0]["l_tile_primitive"]
            latency = {
                "SCALE_ACCUM": 3,
                "DOT_REDUCE": 2,
                "OUTER_UPDATE": 2,
            }.get(primitive)
            if latency is None:
                raise AssertionError(f"unknown L_TILE primitive {primitive}")
            vector_arithmetic_cycles += repeats * latency

    metrics = contract["lowering_metrics"]
    dynamic_instructions = int(metrics["dynamic_issued_instructions"])
    l_tile_instructions = int(metrics["l_tile_exec_count"])
    non_l_tile_issue_cycles = dynamic_instructions - l_tile_instructions
    local_cycles = non_l_tile_issue_cycles + service_cycles + vector_arithmetic_cycles
    return {
        "dynamic_matrix_instructions": dynamic_instructions,
        "dynamic_l_tile_instructions": l_tile_instructions,
        "dynamic_service_groups": total_groups,
        "dynamic_operands": total_operands,
        "logical_values_replayed": total_logical_values,
        "bank_words": total_bank_words,
        "bank_word_beats": hardware.matrix_bank_word_beats,
        "ideal_cycles": ideal_cycles,
        "service_cycles": service_cycles,
        "bank_stall_cycles": service_cycles - ideal_cycles,
        "non_l_tile_issue_cycles": non_l_tile_issue_cycles,
        "vector_arithmetic_cycles": vector_arithmetic_cycles,
        "local_recurrence_cycles": local_cycles,
        "state_transfer_values": int(metrics["state_transfer_values"]),
        "state_transfer_values_by_direction": {
            str(direction): int(values) for direction, values in metrics["state_transfer_values_by_direction"].items()
        },
        "field_logical_values": int(metrics["field_logical_values"]),
        "field_transfer_values": int(metrics["field_transfer_values"]),
        "worst_group_service_cycles": worst_group_service,
        "service_histogram": [
            {
                "phase": phase,
                "primitive": primitive,
                "allocation": allocation,
                "ideal_cycles": ideal,
                "service_cycles": service,
                "dynamic_groups": count,
            }
            for (
                phase,
                primitive,
                allocation,
                ideal,
                service,
            ), count in sorted(service_histogram.items())
        ],
        "lowering": contract["lowering"],
        "architectural": True,
        "state_mode": state_mode,
        "source": (
            "Python replay of Compiler service_groups in Rust L_TILE phase order, "
            "using complete base/pitch/fixed-alpha/tile-phase mapping"
        ),
    }


def attach_real_service_evidence(
    *,
    compiler: dict[str, Any],
    physical: dict[str, Any],
    hardware: MatrixHardwarePoint,
) -> None:
    """Attach exact fixed/phased service totals from executable lowerings."""

    fixed = physical["global_fixed_map"]
    for spec in PACKETS.values():
        modes: dict[str, Any] = {}
        for state_mode in (StateMode.PLENA_BF16,):
            modes[state_mode] = {
                variant: measure_real_service_groups(
                    compiler=compiler,
                    spec=spec,
                    variant=variant,
                    state_mode=state_mode,
                    hardware=hardware,
                    fixed_alpha=int(fixed["alpha"]),
                    fixed_gamma=int(fixed["gamma"]),
                )
                for variant in (
                    MatrixVariant.C_FIXED,
                    MatrixVariant.D_AFFINE,
                    MatrixVariant.E_AFFINE_OVERLAP,
                )
            }
        physical.setdefault(spec.model, {})["real_lowering_service"] = modes


def recurrent_core_metrics(
    *,
    spec: RecurrentPacketSpec,
    variant: MatrixVariant,
    state_mode: StateMode,
    compiler: dict[str, Any],
    physical: dict[str, Any],
    batch_size: int,
) -> dict[str, int]:
    counts = _issue_counts(compiler, spec)
    state_values = spec.heads * spec.recurrence_rows * spec.elements_per_head
    # Layout and chunking may change issue/service traffic, but never the
    # mathematical recurrence.  Keep this invariant explicit in every record.
    arithmetic_element_ops = (
        # dt*x and skip are one row/head; state update is 2 mul + add and
        # C readout is mul + accumulate.
        6 * spec.heads * spec.elements_per_head + 5 * state_values
        if spec.model == "nemotron3"
        # decay SCALE_ACCUM, prediction DOT, beta error, rank-1 update,
        # and updated-state readout.
        else 3 * spec.heads * spec.elements_per_head + 9 * state_values
    )

    if variant == MatrixVariant.A_ORIGINAL:
        return {
            "cycles": counts["A"] * batch_size,
            "issued": counts["A"] * batch_size,
            "issue_cycles": counts["A"] * batch_size,
            "matrix_service_cycles": 0,
            "matrix_ideal_cycles": 0,
            "vector_arithmetic_cycles": 0,
            "packet_ops": 0,
            "service": 0,
            "ideal": 0,
            "stall": 0,
            "logical_state_values": state_values,
            "arithmetic_element_ops": arithmetic_element_ops,
            "explicit_state_hbm_read_bytes": state_values * physical["hardware"]["matrix_element_bits"] // 8,
            "explicit_state_hbm_write_bytes": state_values * physical["hardware"]["matrix_element_bits"] // 8,
        }
    if variant == MatrixVariant.B_ARLO:
        return {
            "cycles": counts["B"] * batch_size,
            "issued": counts["B"] * batch_size,
            "issue_cycles": counts["B"] * batch_size,
            "matrix_service_cycles": 0,
            "matrix_ideal_cycles": 0,
            "vector_arithmetic_cycles": 0,
            "packet_ops": 0,
            "service": 0,
            "ideal": 0,
            "stall": 0,
            "logical_state_values": state_values,
            "arithmetic_element_ops": arithmetic_element_ops,
            "explicit_state_hbm_read_bytes": state_values * physical["hardware"]["matrix_element_bits"] // 8,
            "explicit_state_hbm_write_bytes": state_values * physical["hardware"]["matrix_element_bits"] // 8,
        }

    exact = physical[spec.model]["real_lowering_service"][state_mode][variant]
    directions = exact["state_transfer_values_by_direction"]
    return {
        "cycles": int(exact["local_recurrence_cycles"]) * batch_size,
        "issued": int(exact["dynamic_matrix_instructions"]) * batch_size,
        "issue_cycles": int(exact["non_l_tile_issue_cycles"]) * batch_size,
        "matrix_service_cycles": int(exact["service_cycles"]) * batch_size,
        "matrix_ideal_cycles": int(exact["ideal_cycles"]) * batch_size,
        "vector_arithmetic_cycles": int(exact["vector_arithmetic_cycles"]) * batch_size,
        "packet_ops": int(exact["dynamic_service_groups"]) * batch_size,
        "service": int(exact["service_cycles"]) * batch_size,
        "ideal": int(exact["ideal_cycles"]) * batch_size,
        "stall": int(exact["bank_stall_cycles"]) * batch_size,
        "logical_state_values": state_values * batch_size,
        "arithmetic_element_ops": arithmetic_element_ops * batch_size,
        "explicit_state_hbm_read_bytes": (
            int(directions.get("load", 0)) + int(directions.get("reload_intermediate", 0))
        )
        * int(physical["hardware"]["matrix_element_bits"])
        // 8
        * batch_size,
        "explicit_state_hbm_write_bytes": (
            int(directions.get("store", 0)) + int(directions.get("store_intermediate", 0))
        )
        * int(physical["hardware"]["matrix_element_bits"])
        // 8
        * batch_size,
    }


_MAMBA_RECURRENCE = {"mamba_state_update", "mamba_state_output"}
_KDA_RECURRENCE = {
    "kda_state_decay_prediction",
    "kda_delta_update_output",
}


def _is_recurrence(stage: StageWork, model: str) -> bool:
    return stage.name in (_MAMBA_RECURRENCE if model == "nemotron3" else _KDA_RECURRENCE)


def _coefficient_preparation_metrics(
    stage: StageWork,
    *,
    model: str,
    hardware: MatrixHardwarePoint,
) -> dict[str, int]:
    """Price coefficient transforms that remain outside ``L_TILE``.

    The KDA Compiler implements

    ``decay = exp(lower_bound * sigmoid(rate * (gate + dt_bias)))`` and
    ``beta = sigmoid(beta_logit)``.

    A sigmoid contributes three ordinary elementwise passes plus one exponent
    pass.  Decay additionally contributes add/rate/lower-bound passes and a
    second exponent.  Keep the passes separate when rounding to VLEN; merging
    their operation counts would invent cross-operation lane packing.  Mamba's
    coefficient work already lives in the separate ``mamba_dt_exp`` stage.
    """

    if model != "kimi_k3" or stage.exp_ops == 0:
        return {"cycles": 0, "elementwise_ops": 0, "exp_ops": 0}
    vectors = math.ceil(stage.exp_ops / hardware.vector_lanes)
    if stage.name == "kda_state_decay_prediction":
        elementwise_passes = 6
        exp_passes = 2
    elif stage.name == "kda_delta_update_output":
        elementwise_passes = 3
        exp_passes = 1
    else:
        raise AssertionError(f"unclassified KDA coefficient stage {stage.name}")
    return {
        "cycles": vectors * (elementwise_passes + exp_passes * hardware.exp_latency),
        "elementwise_ops": elementwise_passes * stage.exp_ops,
        "exp_ops": exp_passes * stage.exp_ops,
    }


def simulate_report(
    report: WorkloadReport,
    *,
    model: str,
    variant: MatrixVariant,
    state_mode: StateMode,
    hardware: MatrixHardwarePoint,
    compiler: dict[str, Any],
    physical: dict[str, Any],
) -> dict[str, Any]:
    spec = PACKETS[model]
    core = (
        recurrent_core_metrics(
            spec=spec,
            variant=variant,
            state_mode=state_mode,
            compiler=compiler,
            physical=physical,
            batch_size=report.scenario.batch_size,
        )
        if report.scenario.phase == InferencePhase.DECODE
        else None
    )
    totals: dict[str, Any] = {
        "cycles": 0,
        "hbm_cycles": 0,
        "matrix_cycles": 0,
        "vector_cycles": 0,
        "lcompute_cycles": 0,
        "recurrence_cycles": 0,
        "recurrence_coefficient_prep_cycles": 0,
        "recurrence_coefficient_prep_elementwise_ops": 0,
        "recurrence_coefficient_prep_exp_ops": 0,
        "recurrence_issue_cycles": 0,
        "recurrence_vector_arithmetic_cycles": 0,
        "matrix_sram_ideal_cycles": 0,
        "overlap_cycles": 0,
        "logical_hbm_read_bytes": 0,
        "logical_hbm_write_bytes": 0,
        "logical_weight_read_bytes": 0,
        "physical_hbm_read_bytes": 0,
        "physical_hbm_write_bytes": 0,
        "packet_ops": 0,
        "matrix_sram_service_cycles": 0,
        "bank_stall_cycles": 0,
        "dynamic_issued_instructions": 0,
        "logical_recurrence_state_values": 0,
        "recurrent_arithmetic_element_ops": 0,
        "by_layer_type": defaultdict(lambda: defaultdict(int)),
    }
    recurrence_done: set[int] = set()
    recurrence_hbm_done: set[int] = set()
    for stage in report.stages:
        matrix, vector = _generic_compute(stage, hardware)  # type: ignore[arg-type]

        if report.scenario.phase == InferencePhase.DECODE and _is_recurrence(stage, model):
            # L_TILE consumes already prepared decay/beta coefficients.  Their
            # exponentials therefore remain ordinary Vector work in every
            # ablation variant; only the recurrence arithmetic below is
            # replaced.  Keeping this cost explicit prevents C/D/E from
            # receiving a hidden advantage over the A/B instruction streams.
            coefficient_prep = _coefficient_preparation_metrics(
                stage,
                model=model,
                hardware=hardware,
            )
            coefficient_prep_cycles = coefficient_prep["cycles"]
            matrix = vector = 0
            vector = coefficient_prep_cycles
            totals["recurrence_coefficient_prep_cycles"] += coefficient_prep_cycles
            totals["recurrence_coefficient_prep_elementwise_ops"] += coefficient_prep["elementwise_ops"]
            totals["recurrence_coefficient_prep_exp_ops"] += coefficient_prep["exp_ops"]
            totals["by_layer_type"][stage.layer_type]["recurrence_coefficient_prep_cycles"] += coefficient_prep_cycles
            if stage.layer_id not in recurrence_done:
                recurrence_done.add(stage.layer_id)
                assert core is not None
                if variant in {MatrixVariant.A_ORIGINAL, MatrixVariant.B_ARLO}:
                    # The A/B recurrence stream also consumes prepared
                    # coefficients. Preserve the preparation work charged to
                    # this stage instead of overwriting it with the core issue
                    # proxy. C/D/E account for the core in L-Compute below.
                    vector = core["cycles"] + coefficient_prep_cycles
                else:
                    totals["lcompute_cycles"] += core["cycles"]
                    totals["by_layer_type"][stage.layer_type]["lcompute_cycles"] += core["cycles"]
                totals["recurrence_cycles"] += core["cycles"]
                totals["recurrence_issue_cycles"] += core["issue_cycles"]
                totals["recurrence_vector_arithmetic_cycles"] += core["vector_arithmetic_cycles"]
                totals["matrix_sram_ideal_cycles"] += core["matrix_ideal_cycles"]
                totals["by_layer_type"][stage.layer_type]["recurrence_cycles"] += core["cycles"]
                totals["dynamic_issued_instructions"] += core["issued"]
                totals["logical_recurrence_state_values"] += core["logical_state_values"]
                totals["recurrent_arithmetic_element_ops"] += core["arithmetic_element_ops"]
                totals["packet_ops"] += core["packet_ops"]
                totals["matrix_sram_service_cycles"] += core["service"]
                totals["bank_stall_cycles"] += core["stall"]
                # The recurrence depends on this layer's projection.  Earlier
                # reports subtracted their full overlap despite that dependency.
                # E remains equal to D until a legal group-level double-buffer
                # schedule is emitted and replayed.

        logical_read = stage.traffic.logical_hbm_read_bytes
        logical_write = stage.traffic.logical_hbm_write_bytes
        if (
            report.scenario.phase == InferencePhase.DECODE
            and variant not in {MatrixVariant.A_ORIGINAL, MatrixVariant.B_ARLO}
            and _is_recurrence(stage, model)
        ):
            # Replace the workload's one-read/one-write state estimate with
            # the exact state DMA stream emitted by this layout.  Fixed C may
            # need intermediate spill/reload; compact phased D/E do not. Prepared
            # projection fields remain on chip and are not invented as HBM
            # traffic in the integrated timeline.
            logical_read = 0
            logical_write = 0
            if stage.layer_id not in recurrence_hbm_done:
                recurrence_hbm_done.add(stage.layer_id)
                assert core is not None
                logical_read = core["explicit_state_hbm_read_bytes"]
                logical_write = core["explicit_state_hbm_write_bytes"]
        physical_read = (
            math.ceil(logical_read / hardware.hbm_burst_bytes) * hardware.hbm_burst_bytes if logical_read else 0
        )
        physical_write = (
            math.ceil(logical_write / hardware.hbm_burst_bytes) * hardware.hbm_burst_bytes if logical_write else 0
        )
        hbm = math.ceil((physical_read + physical_write) / hardware.hbm_bytes_per_cycle)
        totals["hbm_cycles"] += hbm
        totals["matrix_cycles"] += matrix
        totals["vector_cycles"] += vector
        totals["logical_hbm_read_bytes"] += logical_read
        totals["logical_hbm_write_bytes"] += logical_write
        totals["logical_weight_read_bytes"] += stage.traffic.weight_read_bytes
        totals["physical_hbm_read_bytes"] += physical_read
        totals["physical_hbm_write_bytes"] += physical_write
        layer = totals["by_layer_type"][stage.layer_type]
        layer["hbm_cycles"] += hbm
        layer["matrix_cycles"] += matrix
        layer["vector_cycles"] += vector

    totals["cycles"] = (
        totals["hbm_cycles"]
        + totals["matrix_cycles"]
        + totals["vector_cycles"]
        + totals["lcompute_cycles"]
        - totals["overlap_cycles"]
    )
    totals["latency_us_proxy"] = totals["cycles"] / (hardware.clock_hz / 1_000_000)
    totals["by_layer_type"] = {
        name: {
            **values,
            "cycles": values["hbm_cycles"]
            + values["matrix_cycles"]
            + values["vector_cycles"]
            + values.get("lcompute_cycles", 0)
            - values.get("overlap_cycles", 0),
        }
        for name, values in totals["by_layer_type"].items()
    }
    return totals


def _routing_for_scenario(
    *,
    model: str,
    phase: InferencePhase,
    batch_size: int,
    context_length: int,
    sequence_length: int,
    decode_index: int,
    profile: RoutingProfile | None,
    strict_profile: bool = False,
) -> tuple[int | None, tuple[tuple[int, int], ...]]:
    if profile is not None:
        profile_matches = (
            model == "nemotron3" and batch_size == profile.batch_size and context_length == profile.context_length
        )
        key = "prefill" if phase == InferencePhase.PREFILL else "decode"
        index = 0 if key == "prefill" else decode_index
        matches = [step for step in profile.steps if (step.phase, step.index) == (key, index)]
        token_count_matches = len(matches) == 1 and matches[0].token_count == batch_size * sequence_length
        if profile_matches and token_count_matches:
            return None, matches[0].unique_experts_by_layer
        if strict_profile:
            raise ValueError(
                "strict routing profile does not match the requested "
                f"model={model}, phase={key}, batch={batch_size}, context={context_length}, "
                f"sequence={sequence_length}, step={index}"
            )
    elif strict_profile:
        raise ValueError("strict routing requires an explicit RoutingProfile")
    topk, experts = (6, 128) if model == "nemotron3" else (16, 896)
    return min(experts, batch_size * sequence_length * topk), ()


def build_reports(
    *,
    model: str,
    phase: InferencePhase,
    batch_size: int,
    tokens: int,
    context_length: int,
    state_mode: StateMode,
    compiler_root: Path,
    profile: RoutingProfile | None,
    strict_profile: bool = False,
    weight_precision: Precision | None = None,
) -> list[WorkloadReport]:
    if state_mode is not StateMode.PLENA_BF16:
        raise ValueError(f"unsupported executable state mode {state_mode}")
    state_precision = Precision.BF16
    workload = _model(
        model,
        compiler_root,
        activation_precision=Precision.BF16,
        weight_precision=weight_precision,
        state_precision=state_precision,
    )
    reports = []
    repetitions = 1 if phase == InferencePhase.PREFILL else tokens
    for index in range(repetitions):
        sequence_length = tokens if phase == InferencePhase.PREFILL else 1
        unique, by_layer = _routing_for_scenario(
            model=model,
            phase=phase,
            batch_size=batch_size,
            context_length=context_length,
            sequence_length=sequence_length,
            decode_index=index,
            profile=profile,
            strict_profile=strict_profile,
        )
        reports.append(
            workload.build(
                _scenario(
                    phase,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    context_length=context_length + (index if phase == InferencePhase.DECODE else 0),
                    moe_unique_experts=unique,
                    moe_unique_experts_by_layer=by_layer,
                )
            )
        )
    return reports


def run_ablation(
    *,
    model: str,
    phase: InferencePhase,
    batch_size: int,
    tokens: int,
    context_length: int,
    state_mode: StateMode,
    hardware: MatrixHardwarePoint,
    compiler_root: Path,
    compiler: dict[str, Any],
    physical: dict[str, Any],
    profile: RoutingProfile | None = None,
    strict_profile: bool = False,
    weight_precision: Precision | None = None,
) -> dict[str, Any]:
    schedule_model = compiler["hybrid_l_tile_schedule"]["variants"]["affine"][model]
    expected_recurrent_layers = 23 if model == "nemotron3" else 69
    if schedule_model["recurrent_layer_count"] != expected_recurrent_layers:
        raise AssertionError(
            f"{model}: hybrid Compiler schedule has {schedule_model['recurrent_layer_count']} recurrent layers"
        )
    if not schedule_model["all_recurrent_layers_emit_l_tile"]:
        raise AssertionError(f"{model}: not every recurrent layer emits L_TILE")
    recurrence_key = "nemotron3_mamba2" if model == "nemotron3" else "kimi_k3_kda"
    per_layer_exec = compiler["matrix_recurrence"]["models"][recurrence_key]["capacity_points"][str(1024 * 1024)][
        "affine"
    ]["metrics"]["l_tile_exec_count"]
    expected_exec = expected_recurrent_layers * int(per_layer_exec)
    if schedule_model["l_tile_exec_count"] != expected_exec:
        raise AssertionError(
            f"{model}: full schedule has {schedule_model['l_tile_exec_count']} "
            f"L_TILE instructions, expected {expected_exec}"
        )
    reports = build_reports(
        model=model,
        phase=phase,
        batch_size=batch_size,
        tokens=tokens,
        context_length=context_length,
        state_mode=state_mode,
        compiler_root=compiler_root,
        profile=profile,
        strict_profile=strict_profile,
        weight_precision=weight_precision,
    )
    records = []
    for variant in MatrixVariant:
        pieces = [
            simulate_report(
                report,
                model=model,
                variant=variant,
                state_mode=state_mode,
                hardware=hardware,
                compiler=compiler,
                physical=physical,
            )
            for report in reports
        ]
        numeric = (
            "cycles",
            "hbm_cycles",
            "matrix_cycles",
            "vector_cycles",
            "lcompute_cycles",
            "recurrence_cycles",
            "recurrence_coefficient_prep_cycles",
            "recurrence_coefficient_prep_elementwise_ops",
            "recurrence_coefficient_prep_exp_ops",
            "recurrence_issue_cycles",
            "recurrence_vector_arithmetic_cycles",
            "matrix_sram_ideal_cycles",
            "overlap_cycles",
            "logical_hbm_read_bytes",
            "logical_hbm_write_bytes",
            "logical_weight_read_bytes",
            "physical_hbm_read_bytes",
            "physical_hbm_write_bytes",
            "packet_ops",
            "matrix_sram_service_cycles",
            "bank_stall_cycles",
            "dynamic_issued_instructions",
            "logical_recurrence_state_values",
            "recurrent_arithmetic_element_ops",
        )
        record = {name: sum(piece[name] for piece in pieces) for name in numeric}
        record["variant"] = variant
        record["latency_us_proxy"] = record["cycles"] / (hardware.clock_hz / 1_000_000)
        record["hbm_cycle_share"] = record["hbm_cycles"] / record["cycles"]
        record["matrix_cycle_share"] = record["matrix_cycles"] / record["cycles"]
        record["vector_cycle_share"] = record["vector_cycles"] / record["cycles"]
        record["lcompute_cycle_share"] = record["lcompute_cycles"] / record["cycles"]
        record["recurrence_cycle_share"] = record["recurrence_cycles"] / record["cycles"]
        layer_types = {name for piece in pieces for name in piece["by_layer_type"]}
        record["by_layer_type"] = {
            layer_type: {
                metric: sum(piece["by_layer_type"].get(layer_type, {}).get(metric, 0) for piece in pieces)
                for metric in (
                    "cycles",
                    "hbm_cycles",
                    "matrix_cycles",
                    "vector_cycles",
                    "lcompute_cycles",
                    "recurrence_cycles",
                    "recurrence_coefficient_prep_cycles",
                    "overlap_cycles",
                )
            }
            for layer_type in sorted(layer_types)
        }
        records.append(record)

    by_variant = {record["variant"]: record for record in records}
    for record in records:
        record["speedup_vs_A"] = by_variant[MatrixVariant.A_ORIGINAL]["cycles"] / record["cycles"]
        record["speedup_vs_B"] = by_variant[MatrixVariant.B_ARLO]["cycles"] / record["cycles"]
        record["speedup_vs_C_fixed"] = by_variant[MatrixVariant.C_FIXED]["cycles"] / record["cycles"]
        record["speedup_vs_D_phased"] = by_variant[MatrixVariant.D_AFFINE]["cycles"] / record["cycles"]

    ordinary = ("attention", "moe", "mla", "latent_moe", "dense", "attn_res")
    for layer_type in ordinary:
        observed = {record["by_layer_type"].get(layer_type, {}).get("cycles", 0) for record in records}
        if len(observed) > 1:
            raise AssertionError(f"ordinary {layer_type} path regressed across layouts")
    if phase == InferencePhase.PREFILL:
        if len({record["cycles"] for record in records}) != 1:
            raise AssertionError("decode-only Matrix packet mode changed prefill")

    layer_counts = reports[0].to_dict()["layer_counts"]
    expected = (
        {"mamba": 23, "moe": 23, "attention": 6}
        if model == "nemotron3"
        else {"kda": 69, "mla": 24, "latent_moe": 92, "dense": 1}
    )
    for name, count in expected.items():
        if layer_counts.get(name) != count:
            raise AssertionError(f"{model}: expected {count} {name} layers, got {layer_counts}")
    return {
        "model": model,
        "phase": phase,
        "batch_size": batch_size,
        "tokens": tokens,
        "context_length": context_length,
        "state_mode": state_mode,
        "weight_precision": reports[0].weight_precision,
        "weight_precision_policy": (
            reports[0].weight_precision_policy.to_dict()
            if reports[0].weight_precision_policy is not None
            else {
                "name": f"uniform_{reports[0].weight_precision}",
                "default_precision": reports[0].weight_precision,
                "source": "explicit DSE sensitivity override",
            }
        ),
        "routing_profile_required": strict_profile,
        "routing_profile_used": profile is not None,
        "layer_counts": layer_counts,
        "ordinary_attention_moe_cycles_identical": True,
        "compiler_l_tile_schedule": {
            "recurrent_layer_count": schedule_model["recurrent_layer_count"],
            "all_recurrent_layers_emit_l_tile": schedule_model["all_recurrent_layers_emit_l_tile"],
            "assembly_sha256": schedule_model["assembly_sha256"],
            "l_tile_exec_count": schedule_model["l_tile_exec_count"],
            "execution_boundary": schedule_model["architectural_boundary"],
        },
        "records": records,
    }


def _precision_evidence(root: Path) -> dict[str, Any]:
    kda_path = root / "analytic_models/performance/profiles/kda_state_precision.json"
    kda = json.loads(kda_path.read_text())
    selected = [
        record
        for record in kda["records"]
        if record["tokens"] == 2048
        and record["schedule"] == "token"
        and record["storage"] in {"fp32", "bf16", "fp16", "mx8_b128"}
    ]
    return {
        "kimi_kda_s2048_token_schedule": selected,
        "nemotron_mamba_s32768": _cached_gpu_report()["b200_supplemental"]["mamba_precision_s32768"],
        "interpretation": (
            "The PLENA execution point uses BF16 state. The official GPU path uses "
            "FP32; other formats remain numerical storage experiments and are not "
            "checkpoint-quality claims."
        ),
    }


def _dse_packet(
    *,
    name: str,
    row_width: int,
    packet_width: int,
    hardware: MatrixHardwarePoint,
) -> dict[str, Any]:
    if packet_width % row_width:
        raise ValueError("DSE packet width must contain complete logical rows")
    if row_width % hardware.bank_width:
        return {
            "name": name,
            "row_width": row_width,
            "packet_width": packet_width,
            "supported": False,
            "reason": (f"row width {row_width} is not a whole {hardware.bank_width}-value Matrix bank word"),
        }
    spec = RecurrentPacketSpec(
        model=name,
        compiler_key=name,
        layer_type="dse",
        layers=1,
        heads=packet_width // row_width,
        packet_heads=packet_width // row_width,
        elements_per_head=row_width,
        recurrence_rows=1,
        passes=1,
        bf16_field_loads_per_row_pass=1,
    )
    fixed = measure_packet(
        spec,
        hardware=hardware,
        alpha=1,
        gamma=0,
        tile_pitch_rows=row_width // hardware.bank_width,
    )
    phased = measure_packet(
        spec,
        hardware=hardware,
        alpha=1,
        gamma=0,
        tile_pitch_rows=0,
        tile_phase_stride=row_width // hardware.bank_width,
    )
    return {
        "name": name,
        "row_width": row_width,
        "packet_width": packet_width,
        "supported": True,
        "C_fixed": fixed,
        "D_compact_phased": phased,
        "compact_phase_speedup_over_fixed": (
            fixed["service_cycles"] / phased["service_cycles"]
        ),
        "values_checked_per_variant": packet_width,
    }


def build_layout_dse(
    *,
    hardware: MatrixHardwarePoint,
    experiments: dict[str, Any],
) -> dict[str, Any]:
    """Sweep the frozen fixed-diagonal/compact-phase geometry."""

    bank_geometry = []
    for bank_width in (8, 16, 32, 64):
        point = MatrixHardwarePoint(
            banks=hardware.mlen // bank_width,
            bank_width=bank_width,
        )
        bank_geometry.append(
            {
                "banks": point.banks,
                "bank_width": point.bank_width,
                "mlen": point.mlen,
                "mamba": _dse_packet(
                    name="nemotron3_mamba2",
                    row_width=64,
                    packet_width=point.mlen,
                    hardware=point,
                ),
                "kda": _dse_packet(
                    name="kimi_k3_kda",
                    row_width=128,
                    packet_width=point.mlen,
                    hardware=point,
                ),
                "resource_proxy": {
                    "additional_programmable_bank_select_adders": 0,
                    "fixed_diagonal_bank_select_adders_existing": point.banks,
                    "lane_restore_payload_bits": (point.mlen * point.matrix_element_bits),
                    "ports_per_bank": {"read": 1, "write": 1},
                },
            }
        )

    row_width = [
        _dse_packet(
            name=f"row_{width}",
            row_width=width,
            packet_width=hardware.mlen,
            hardware=hardware,
        )
        for width in (32, 64, 128, 256)
    ]
    packet_width = [
        _dse_packet(
            name=f"packet_{width}",
            row_width=row,
            packet_width=width,
            hardware=hardware,
        )
        for row in (64, 128)
        for width in (512, 1024, 2048)
    ]

    def record(mode: StateMode, model: str, case: str) -> dict[str, Any]:
        return next(
            item for item in experiments[mode][model][case]["records"] if item["variant"] == MatrixVariant.D_AFFINE
        )

    precision = {
        StateMode.PLENA_BF16: {
            model: {
                "cycles": record(StateMode.PLENA_BF16, model, "decode_b1_t1")["cycles"],
                "physical_hbm_read_bytes": record(StateMode.PLENA_BF16, model, "decode_b1_t1")[
                    "physical_hbm_read_bytes"
                ],
                "bank_stall_cycles": record(StateMode.PLENA_BF16, model, "decode_b1_t1")["bank_stall_cycles"],
            }
            for model in ("nemotron3", "kimi_k3")
        },
        "mixed_precision_boundary": (
            "The executable Matrix recurrence is uniformly BF16. FP32 is retained only "
            "as the official GPU reference, while FP16/MX8 remain numerical studies."
        ),
    }
    batch = {
        model: [
            {
                "batch": batch_size,
                "cycles": record(
                    StateMode.PLENA_BF16,
                    model,
                    f"decode_b{batch_size}_t1",
                )["cycles"],
                "bank_stall_cycles": record(
                    StateMode.PLENA_BF16,
                    model,
                    f"decode_b{batch_size}_t1",
                )["bank_stall_cycles"],
            }
            for batch_size in (1, 2, 4, 8, 16, 32, 64)
        ]
        for model in ("nemotron3", "kimi_k3")
    }
    return {
        "candidate_mapping": (
            "fixed diagonal alpha=1 plus compact tile_phase_stride; no programmable row coefficient"
        ),
        "method": (
            "Each C/D point writes numbered logical values, reads them through the "
            "selected physical map, restores lane order, and checks exact equality. "
            "Only bank geometry or packet shape changes; both paths retain one "
            "read and one write port per bank."
        ),
        "bank_count_and_bank_width": bank_geometry,
        "tile_logical_row_width": row_width,
        "packet_width": packet_width,
        "state_precision_full_timeline": precision,
        "batch_full_timeline": batch,
    }


def build_prefill_handoff_timeline_delta(
    *,
    experiments: dict[str, Any],
    handoff: dict[str, Any],
) -> dict[str, Any]:
    """Keep the verified handoff census while withdrawing its speedup claim.

    The legacy MAC count comes from emitted Compiler assembly and the view path
    moves numbered values through a transpose view.  They do not form two
    measured executions of the same prefill timeline, however, so composing
    either one serially with the analytic workload would manufacture a speedup.
    """

    legacy = handoff["legacy_identity_gemm"]
    view = handoff["matrix_view_handoff"]
    shape = handoff["shape"]
    _ = experiments
    view_config_instructions = int(view["configuration_dynamic_instructions"]) * int(shape["kda_layers"])

    return {
        "scope": (
            "Kimi K3 BF16/MX8 candidate only; Compiler instruction census and "
            "numbered-value transpose equivalence, not a latency comparison"
        ),
        "performance_claim_withdrawn": True,
        "withdrawn_claims": ["prefill_b1_s16_3.387x", "prefill_b1_s128_1.713x"],
        "withdrawal_reason": (
            "the old and view paths were not measured as two executions of the "
            "same prefill timeline; serially adding an asserted analytic denominator "
            "inflated the result"
        ),
        "evidence_kind": {
            "legacy": "Compiler-emitted instruction and MAC census",
            "view": "numbered values moved through Matrix cells and compared exactly",
        },
        "legacy_logical_macs_eliminated": int(legacy["logical_macs_all_kda_layers"]),
        "legacy_emitted_padded_macs_eliminated": int(legacy["emitted_padded_macs_all_kda_layers"]),
        "legacy_matrix_cycles_formula_not_used_for_speedup": int(legacy["emitted_matrix_cycles_all_kda_layers"]),
        "view_handoff_macs": int(view["handoff_macs"]),
        "view_configuration_instructions_if_repeated_per_layer": view_config_instructions,
        "values_moved_and_compared": int(view["value_evidence"]["values_checked"]),
        "official_fp32_speedup_claimed": False,
        "cases": {},
    }


def build_static_overlap_feasibility(
    *,
    compiler: dict[str, Any],
    hardware: MatrixHardwarePoint,
) -> dict[str, Any]:
    """Prove whether the one-MiB point can hold a second state group.

    Variant E may hide the next group's HBM load only when the current phased
    working set and a second state group occupy disjoint Matrix-SRAM cells.
    This is a capacity test, not a latency estimate.
    """

    capacity_words = hardware.matrix_sram_rows * hardware.banks
    bytes_per_bank_word = hardware.bank_width * hardware.matrix_element_bits // 8
    records: dict[str, Any] = {}
    for model, spec in PACKETS.items():
        contract_name = "nemotron3_mamba2" if model == "nemotron3" else "kimi_k3_kda"
        working_set = compiler["matrix_recurrence"]["models"][contract_name]["capacity_points"][
            str(hardware.matrix_sram_bytes)
        ]["affine"]["working_set"]
        current_words = int(working_set["capacity_facts"]["bank_words"])
        group_heads = int(working_set["group_heads"])
        words_per_row = spec.elements_per_head // hardware.bank_width
        second_state_words = group_heads * spec.recurrence_rows * words_per_row
        required_words = current_words + second_state_words
        excess_words = max(0, required_words - capacity_words)
        records[model] = {
            "current_affine_working_set_bank_words": current_words,
            "second_state_group_bank_words": second_state_words,
            "required_bank_words": required_words,
            "capacity_bank_words": capacity_words,
            "fits_same_capacity": required_words <= capacity_words,
            "minimum_additional_bytes": excess_words * bytes_per_bank_word,
            "minimum_capacity_bytes": required_words * bytes_per_bank_word,
            "group_heads": group_heads,
        }
    return {
        "method": (
            "current phased working-set occupancy plus one disjoint state group; "
            "no tag, cache, replacement policy or runtime decision"
        ),
        "models": records,
        "variant_e_credit_allowed": all(record["fits_same_capacity"] for record in records.values()),
        "conclusion": (
            "The fixed one-MiB point cannot legally double-buffer the current "
            "head group. E therefore receives zero overlap credit. A future E "
            "point must either emit a smaller-group ping-pong schedule or charge "
            "the reported capacity increase."
        ),
    }


def build_campaign(
    *,
    compiler_root: Path,
    simulator_root: Path,
) -> dict[str, Any]:
    hardware = MatrixHardwarePoint()
    compiler = load_compiler_evidence(str(compiler_root))
    packet_contracts = {
        spec.model: {
            "fixed": _validate_real_packet_contract(compiler, spec, co_layout=False),
            "affine": _validate_real_packet_contract(compiler, spec, co_layout=True),
        }
        for spec in PACKETS.values()
    }
    physical = build_physical_evidence(hardware)
    attach_real_service_evidence(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
    ordinary_no_regression = build_ordinary_no_regression_evidence(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
    routing = load_pinned_nemotron_profile()
    measured_decode_steps = sum(step.phase == "decode" for step in routing.steps)
    experiments: dict[str, Any] = {}
    # Compiler, analytic model and Rust share one executable BF16 state point.
    for state_mode in (StateMode.PLENA_BF16,):
        mode_cases: dict[str, Any] = {}
        for model in ("nemotron3", "kimi_k3"):
            cases: dict[str, Any] = {}
            for batch in (1, 2, 4, 8, 16, 32, 64):
                cases[f"decode_b{batch}_t1"] = run_ablation(
                    model=model,
                    phase=InferencePhase.DECODE,
                    batch_size=batch,
                    tokens=1,
                    context_length=2048,
                    state_mode=state_mode,
                    hardware=hardware,
                    compiler_root=compiler_root,
                    compiler=compiler,
                    physical=physical,
                    profile=(routing if model == "nemotron3" and batch == 1 else None),
                )
            for decode_tokens in (4, 32, 128):
                cases[f"decode_b1_t{decode_tokens}"] = run_ablation(
                    model=model,
                    phase=InferencePhase.DECODE,
                    batch_size=1,
                    tokens=decode_tokens,
                    context_length=2048,
                    state_mode=state_mode,
                    hardware=hardware,
                    compiler_root=compiler_root,
                    compiler=compiler,
                    physical=physical,
                    profile=(routing if model == "nemotron3" else None),
                )
            for prefill_tokens in (16, 128):
                cases[f"prefill_b1_s{prefill_tokens}"] = run_ablation(
                    model=model,
                    phase=InferencePhase.PREFILL,
                    batch_size=1,
                    tokens=prefill_tokens,
                    context_length=prefill_tokens,
                    state_mode=state_mode,
                    hardware=hardware,
                    compiler_root=compiler_root,
                    compiler=compiler,
                    physical=physical,
                    profile=None,
                )
            mode_cases[model] = cases
        experiments[state_mode] = mode_cases

    bandwidth_sweep = {}
    for bandwidth in (64, 256, 512, 1024, 1560, 8192):
        point = MatrixHardwarePoint(hbm_bytes_per_cycle=bandwidth)
        bandwidth_sweep[str(bandwidth)] = {
            model: run_ablation(
                model=model,
                phase=InferencePhase.DECODE,
                batch_size=1,
                tokens=1,
                context_length=2048,
                state_mode=StateMode.PLENA_BF16,
                hardware=point,
                compiler_root=compiler_root,
                compiler=compiler,
                physical=physical,
                profile=(routing if model == "nemotron3" else None),
            )
            for model in ("nemotron3", "kimi_k3")
        }
    port_width_sweep: dict[str, Any] = {}
    for name, bits in (
        ("bf16_reference_port", 512),
        ("bf16_half_width_two_beat_control", 256),
    ):
        point = MatrixHardwarePoint(matrix_bank_port_bits_per_cycle=bits)
        point_physical = build_physical_evidence(point)
        attach_real_service_evidence(
            compiler=compiler,
            physical=point_physical,
            hardware=point,
        )
        port_width_sweep[name] = {
            "hardware": asdict(point),
            "resource_proxies": point.resource_proxies(),
            "models": {
                model: run_ablation(
                    model=model,
                    phase=InferencePhase.DECODE,
                    batch_size=1,
                    tokens=1,
                    context_length=2048,
                    state_mode=StateMode.PLENA_BF16,
                    hardware=point,
                    compiler_root=compiler_root,
                    compiler=compiler,
                    physical=point_physical,
                    profile=(routing if model == "nemotron3" else None),
                )
                for model in ("nemotron3", "kimi_k3")
            },
        }
    layout_dse = build_layout_dse(
        hardware=hardware,
        experiments=experiments,
    )
    prefill_handoff_delta = build_prefill_handoff_timeline_delta(
        experiments=experiments,
        handoff=compiler["prefill_handoff"],
    )
    overlap_feasibility = build_static_overlap_feasibility(
        compiler=compiler,
        hardware=hardware,
    )
    if not overlap_feasibility["variant_e_credit_allowed"]:
        for models in experiments.values():
            for cases in models.values():
                for case in cases.values():
                    by_variant = {record["variant"]: record for record in case["records"]}
                    if (
                        by_variant[MatrixVariant.E_AFFINE_OVERLAP]["cycles"]
                        != by_variant[MatrixVariant.D_AFFINE]["cycles"]
                    ):
                        raise AssertionError("E received overlap credit without a capacity-legal schedule")

    return {
        "schema_version": 6,
        "outcome": (
            "The executable phased descriptor removes stalls relative to the constrained "
            "single-base C lowering, but the fair fixed-wiring D' control reaches the same "
            "conflict-free bank floor. Therefore arbitrary programmable row skew is removed "
            "before RTL; C-to-D gains are descriptor/chunk/issue/spill effects. Whole-model "
            "results use the official 52/93-layer analytic schedule with symbolic weights; "
            "no projection overlap or real-weight first-to-last Rust execution is fabricated."
        ),
        "status": "EVALUATED_PRE_RTL_WITH_EXPLICIT_BOUNDARIES",
        "scope": "Compiler plus analytic/transactional Simulator; no RTL, synthesis or PPA",
        "round3_triage": build_round3_triage(),
        "evidence_levels": EVIDENCE_LEVELS,
        "hardware": asdict(hardware),
        "resource_proxies": hardware.resource_proxies(),
        "resource_proxies_by_variant": hardware.resource_proxies_by_variant(),
        "static_overlap_feasibility": overlap_feasibility,
        "physical_packet_evidence": physical,
        "ordinary_attention_moe_no_regression": ordinary_no_regression,
        "compiler": {
            "source_root": compiler["compiler_root"],
            "plena_settings_toml": compiler["plena_settings_toml"],
            "matrix_view_contract_version": compiler["matrix_view_contract_version"],
            "issue_counts": {spec.model: _issue_counts(compiler, spec) for spec in PACKETS.values()},
            "real_packet_histograms": [
                {
                    "model": case["model"],
                    "stage": case["stage"],
                    "lowering": case["lowering"],
                    "consumer_descriptor": case.get("consumer_descriptor"),
                    "histogram": case["histogram"],
                    "coissued_histogram": case["coissued_histogram"],
                }
                for case in compiler["real_packets"]["cases"]
            ],
            "validated_recurrence_packets": packet_contracts,
            "matrix_recurrence_contract": compiler["matrix_recurrence"],
            "hybrid_l_tile_schedule": compiler["hybrid_l_tile_schedule"],
            "packet_extraction_coverage": compiler["real_packets"]["coverage"],
            "packet_capacity_contract": compiler["real_packets"]["capacity_contract"],
            "packet_scope_boundary": compiler["real_packets"]["scope_boundary"],
            "prefill_to_decode_handoff": compiler["prefill_handoff"],
        },
        "experiments": experiments,
        "bandwidth_sweep": bandwidth_sweep,
        "matrix_port_width_sweep": port_width_sweep,
        "layout_dse": layout_dse,
        "precision_evidence": _precision_evidence(simulator_root),
        "connected_recurrence_evidence": load_connected_recurrence_evidence(
            simulator_root,
            expected_contract_version=compiler["matrix_view_contract_version"],
        ),
        "prefill_handoff_evidence": compiler["prefill_handoff"],
        "prefill_handoff_timeline_delta": prefill_handoff_delta,
        "state_residency": build_state_residency_report(),
        "gpu_evidence": _cached_gpu_report(),
        "routing_evidence": {
            "nemotron3": {
                "model_id": routing.model_id,
                "revision": routing.revision,
                "source_sha256": routing.source_sha256,
                "batch_size": routing.batch_size,
                "context_length": routing.context_length,
                "measured_decode_steps": measured_decode_steps,
                "decode_b1_t128_policy": (
                    f"use the {measured_decode_steps} measured steps, then use the "
                    "conservative maximum-active-expert bound for every missing step"
                ),
            },
            "kimi_k3": {
                "policy": (
                    "no measured batch routing trace is available; use the conservative maximum-active-expert bound"
                )
            },
        },
        "claim_boundaries": {
            "lcompute_credit": (
                "B to C/D: moving the complete recurrence from row-wise Vector issue into "
                "generic Matrix-view L_TILE execution. C to D is compact descriptor, "
                "chunk, issue and spill credit; it is not pure bank-conflict credit."
            ),
            "programmable_skew_bank_credit": (
                "D over D': 1.00x for both official BF16 state packets because fixed "
                "diagonal wiring plus per-tile base phases occupies identical cells"
            ),
            "compiler_credit": "B over A only",
            "overlap_credit": "E over D only",
            "coefficient_preparation": (
                "KDA decay/beta elementwise and exponential transforms remain "
                "ordinary Vector work and are charged identically to A/B/C/D/E; "
                "L_TILE replaces only the "
                "prepared-coefficient recurrence. Mamba dt/exp is already a "
                "separate stage."
            ),
            "bf16_state": (
                "uniform BF16, explicit HBM -> existing Matrix SRAM -> HBM state groups; "
                "compiler-managed scratchpad, no cache"
            ),
            "prefill": (
                "full workload timeline supported at PLENA BF16. Official GPU FP32 state "
                "remains calibration metadata. The emitted identity-GEMM census and BF16 "
                "numbered-value "
                "view handoff are reported separately. Their previous serial speedup "
                "composition is withdrawn because it did not compare two measured paths."
            ),
            "weights": "official shapes and checkpoint storage policy; symbolic PLENA weights",
            "performance": "cycle model calibrated by Compiler and GPU evidence, not silicon",
            "energy": "not reported because no validated power model exists",
            "ppa": "not reported because no RTL was synthesized",
        },
    }


def _record(campaign: dict[str, Any], mode: str, model: str, case: str, variant: str) -> dict[str, Any]:
    records = campaign["experiments"][mode][model][case]["records"]
    return next(record for record in records if record["variant"] == variant)


def write_artifacts(campaign: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    (output_dir / "state_residency.json").write_text(json.dumps(campaign["state_residency"], indent=2) + "\n")
    rows = []
    for mode, models in campaign["experiments"].items():
        for model, cases in models.items():
            for case_name, case in cases.items():
                for record in case["records"]:
                    rows.append(
                        {
                            "state_mode": mode,
                            "model": model,
                            "case": case_name,
                            "phase": case["phase"],
                            "batch": case["batch_size"],
                            "tokens": case["tokens"],
                            "variant": record["variant"],
                            "cycles": record["cycles"],
                            "latency_us_proxy": record["latency_us_proxy"],
                            "speedup_vs_A": record["speedup_vs_A"],
                            "speedup_vs_B": record["speedup_vs_B"],
                            "speedup_vs_C_fixed": record["speedup_vs_C_fixed"],
                            "speedup_vs_D_phased": record["speedup_vs_D_phased"],
                            "bank_stall_cycles": record["bank_stall_cycles"],
                            "matrix_sram_service_cycles": record["matrix_sram_service_cycles"],
                            "matrix_sram_ideal_cycles": record["matrix_sram_ideal_cycles"],
                            "recurrence_cycles": record["recurrence_cycles"],
                            "recurrence_coefficient_prep_cycles": record["recurrence_coefficient_prep_cycles"],
                            "recurrence_coefficient_prep_elementwise_ops": record[
                                "recurrence_coefficient_prep_elementwise_ops"
                            ],
                            "recurrence_coefficient_prep_exp_ops": record["recurrence_coefficient_prep_exp_ops"],
                            "recurrence_issue_cycles": record["recurrence_issue_cycles"],
                            "recurrence_vector_arithmetic_cycles": record["recurrence_vector_arithmetic_cycles"],
                            "vector_cycle_share": record["vector_cycle_share"],
                            "lcompute_cycle_share": record["lcompute_cycle_share"],
                            "physical_hbm_read_bytes": record["physical_hbm_read_bytes"],
                            "physical_hbm_write_bytes": record["physical_hbm_write_bytes"],
                        }
                    )
    with (output_dir / "ablation.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for mode in campaign["experiments"]:
        for model in ("nemotron3", "kimi_k3"):
            for batch in (1, 2, 4, 8, 16, 32, 64):
                case = f"decode_b{batch}_t1"
                records = {
                    variant: _record(campaign, str(mode), model, case, str(variant)) for variant in MatrixVariant
                }
                fixed_phased = campaign["physical_packet_evidence"]["fixed_phased_bank_control"][model]
                summary.append(
                    {
                        "state_mode": mode,
                        "model": model,
                        "batch": batch,
                        "A_original_cycles": records[MatrixVariant.A_ORIGINAL]["cycles"],
                        "B_arlo_cycles": records[MatrixVariant.B_ARLO]["cycles"],
                        "C_fixed_l_tile_cycles": records[MatrixVariant.C_FIXED]["cycles"],
                        "D_compact_phased_l_tile_cycles": records[MatrixVariant.D_AFFINE]["cycles"],
                        "E_static_overlap_cycles": records[MatrixVariant.E_AFFINE_OVERLAP]["cycles"],
                        "D_speedup_vs_A": records[MatrixVariant.D_AFFINE]["speedup_vs_A"],
                        "D_speedup_vs_B": records[MatrixVariant.D_AFFINE]["speedup_vs_B"],
                        "D_speedup_vs_C_single_base": records[MatrixVariant.D_AFFINE]["speedup_vs_C_fixed"],
                        "D_prime_fixed_phased_bank_service_cycles": fixed_phased["service_cycles"],
                        "D_prime_fixed_phased_bank_stall": fixed_phased["bank_stall_cycles"],
                        "D_vs_D_prime_pure_bank_speedup": fixed_phased[
                            "compact_phase_vs_explicit_bases_bank_speedup"
                        ],
                        "E_speedup_vs_D": records[MatrixVariant.E_AFFINE_OVERLAP]["speedup_vs_D_phased"],
                        "C_bank_stall": records[MatrixVariant.C_FIXED]["bank_stall_cycles"],
                        "D_bank_stall": records[MatrixVariant.D_AFFINE]["bank_stall_cycles"],
                        "coefficient_prep_cycles_all_variants": records[MatrixVariant.D_AFFINE][
                            "recurrence_coefficient_prep_cycles"
                        ],
                        "whole_model_evidence": (
                            "formula-based analytic timeline with official dimensions, "
                            "Compiler-emitted recurrence and symbolic weights"
                        ),
                    }
                )
    with (output_dir / "headline.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(summary[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary)

    attribution = []
    for mode in campaign["experiments"]:
        for model in ("nemotron3", "kimi_k3"):
            fixed = _record(
                campaign,
                str(mode),
                model,
                "decode_b1_t1",
                str(MatrixVariant.C_FIXED),
            )
            phased = _record(
                campaign,
                str(mode),
                model,
                "decode_b1_t1",
                str(MatrixVariant.D_AFFINE),
            )
            fixed_phased = campaign["physical_packet_evidence"]["fixed_phased_bank_control"][model]
            components = {
                "hbm_cycles_saved": fixed["hbm_cycles"] - phased["hbm_cycles"],
                "issue_cycles_saved": fixed["recurrence_issue_cycles"] - phased["recurrence_issue_cycles"],
                "ideal_matrix_service_cycles_saved": fixed["matrix_sram_ideal_cycles"]
                - phased["matrix_sram_ideal_cycles"],
                "bank_stall_cycles_saved": fixed["bank_stall_cycles"] - phased["bank_stall_cycles"],
                "arithmetic_cycles_saved": fixed["recurrence_vector_arithmetic_cycles"]
                - phased["recurrence_vector_arithmetic_cycles"],
            }
            total_saved = fixed["cycles"] - phased["cycles"]
            if sum(components.values()) != total_saved:
                raise AssertionError(
                    f"{model}: C-to-D attribution sums to {sum(components.values())}, expected {total_saved}"
                )
            bank_only_cycles = fixed["cycles"] - components["bank_stall_cycles_saved"]
            attribution.append(
                {
                    "state_mode": mode,
                    "model": model,
                    "C_to_D_total_cycles_saved": total_saved,
                    **components,
                    "bank_stall_fraction_of_C_to_D_saved": (components["bank_stall_cycles_saved"] / total_saved),
                    "C_cycles_if_only_bank_stalls_removed": bank_only_cycles,
                    "bank_only_speedup_over_C": fixed["cycles"] / bank_only_cycles,
                    "D_prime_fixed_phased_bank_service_cycles": fixed_phased["service_cycles"],
                    "D_prime_fixed_phased_bank_stall_cycles": fixed_phased["bank_stall_cycles"],
                    "D_vs_D_prime_pure_bank_speedup": fixed_phased[
                        "compact_phase_vs_explicit_bases_bank_speedup"
                    ],
                    "interpretation": (
                        "C is a constrained single-base executable descriptor, not the fair "
                        "bank-only control. D' uses fixed diagonal wiring plus legal per-tile "
                        "base phases and matches D at 1.00x bank service. C-to-D changes "
                        "descriptor compactness, chunk issue, ideal service and KDA spill."
                    ),
                }
            )
    with (output_dir / "c_to_d_attribution.csv").open("w", newline="") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=list(attribution[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(attribution)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    simulator_root = Path(__file__).resolve().parents[2]
    campaign = build_campaign(
        compiler_root=args.compiler_root,
        simulator_root=simulator_root,
    )
    write_artifacts(campaign, args.output_dir)
    print(json.dumps({"status": campaign["status"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
