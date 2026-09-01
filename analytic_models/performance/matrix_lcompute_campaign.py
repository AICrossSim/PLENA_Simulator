"""End-to-end Matrix-SRAM L-Compute campaign for the two hybrid models.

This campaign is intentionally independent from the older Vector-SRAM
``L_CFG`` experiments.  Its architectural variable is only the physical
placement of Matrix-SRAM bank words.  In particular, C, D' and D execute the
same operation stream; only the skew used by the Matrix bank mapper changes.

The default precision mode preserves the official GPU observation that both
Nemotron Mamba and Kimi KDA recurrent state is FP32.  Such state is explicitly
streamed and is never called resident, cached, or silently stored in the BF16
Matrix SRAM.  A second, clearly labelled BF16-state design point is included
to show the architectural upper bound together with its measured accuracy
risk.
"""

from __future__ import annotations

import argparse
import csv
import functools
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
    C_MULTIROW_ORIGINAL = "C_multirow_original_fixed"
    D_PRIME_BEST_FIXED = "D_prime_best_global_fixed"
    D_AFFINE = "D_compiler_per_tile_affine"
    E_AFFINE_OVERLAP = "E_affine_plus_matrix_overlap"


class StateMode(StrEnum):
    OFFICIAL_FP32 = "official_fp32_streamed"
    BF16_CANDIDATE = "bf16_state_candidate"


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
    matrix_sram_rows: int = 256
    matrix_element_bits: int = 16
    view_slots: int = 4

    def __post_init__(self) -> None:
        if self.mlen != self.banks * self.bank_width:
            raise ValueError("MLEN must equal banks * bank_width")
        if self.banks < 1 or self.banks & (self.banks - 1):
            raise ValueError("Matrix bank count must be a power of two")
        if min(self.blen, self.hbm_bytes_per_cycle, self.hbm_burst_bytes) <= 0:
            raise ValueError("hardware dimensions and bandwidth must be positive")

    @property
    def matrix_macs_per_cycle(self) -> int:
        return self.mlen * self.blen

    @property
    def matrix_sram_bytes(self) -> int:
        return self.mlen * self.matrix_sram_rows * self.matrix_element_bits // 8

    def resource_proxies(self) -> dict[str, Any]:
        skew_bits = int(math.log2(self.banks))
        bank_word_bits = self.bank_width * self.matrix_element_bits
        packet_bits = self.mlen * self.matrix_element_bits
        rotator_stages = skew_bits
        return {
            "scope": "pre-RTL structural proxy; not area, power, timing or PPA",
            "additional_sram_payload_bytes": 0,
            "additional_cache_tags_or_replacement_bits": 0,
            "additional_mac_lanes": 0,
            "configuration_register_bits": self.view_slots * 64,
            "configured_view_slots": self.view_slots,
            "skew_bits": skew_bits,
            "parallel_bank_select_adders": self.banks,
            "bank_select_adder_width_bits": skew_bits,
            "cyclic_lane_restore_bank_words": self.banks,
            "lane_restore_word_width_bits": bank_word_bits,
            "lane_restore_mux_stages": rotator_stages,
            "conservative_one_bit_mux_equivalents": (
                self.banks * rotator_stages * bank_word_bits
            ),
            "additional_matrix_sram_read_ports_per_bank": 0,
            "additional_matrix_sram_write_ports_per_bank": 0,
            "additional_operand_staging_bytes": 0,
            "existing_vector_operand_buffer_reused": True,
            "maximum_operand_hold_bytes": packet_bits // 8,
            "matrix_to_vector_operand_staging_bits": packet_bits,
            "matrix_vector_bypass_payload_bits_per_cycle": packet_bits,
            "matrix_to_vector_operand_mux_bits": packet_bits,
            "vector_to_matrix_writeback_mux_bits": packet_bits,
            "matrix_sram_capacity_bytes": self.matrix_sram_bytes,
            "note": (
                "The mapper and cyclic restore network reuse the existing one-word-per-bank "
                "row interface. With one read port per bank, a two-source Vector operation "
                "uses the existing Vector operand buffer to hold the first restored packet "
                "while the second packet arrives; no new operand SRAM is introduced. The "
                "Matrix-to-Vector and Vector-to-Matrix bypass "
                "muxes are new datapaths, but no SRAM payload or port is added. Synthesis "
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
            programmable_skew = variant in {
                MatrixVariant.D_AFFINE,
                MatrixVariant.E_AFFINE_OVERLAP,
            }
            result[variant] = {
                "packetized_matrix_access": packet_path,
                "compiler_programmable_per_tile_skew": programmable_skew,
                "configuration_register_bits": (
                    self.view_slots * 64 if programmable_skew else 0
                ),
                "parallel_skew_address_adders": (
                    self.banks if programmable_skew else 0
                ),
                "skew_adder_width_bits": skew_bits if programmable_skew else 0,
                "cyclic_lane_restore_payload_bits": packet_bits if packet_path else 0,
                "additional_operand_staging_bytes": 0,
                "existing_vector_operand_buffer_reused": packet_path,
                "maximum_operand_hold_bytes": packet_bits // 8 if packet_path else 0,
                "matrix_sram_read_ports_per_bank": 1,
                "matrix_sram_write_ports_per_bank": 1,
                "additional_matrix_sram_read_ports_per_bank": 0,
                "additional_matrix_sram_write_ports_per_bank": 0,
                "matrix_sram_capacity_bytes": self.matrix_sram_bytes,
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
    packet_heads=32,
    elements_per_head=64,
    recurrence_rows=128,
    passes=1,
    bf16_field_loads_per_row_pass=3,
)

KIMI_PACKET = RecurrentPacketSpec(
    model="kimi_k3",
    compiler_key="kimi_k3_decode_recurrent_mixer",
    layer_type="kda",
    layers=69,
    heads=96,
    packet_heads=16,
    elements_per_head=128,
    recurrence_rows=128,
    passes=2,
    bf16_field_loads_per_row_pass=2,
)

PACKETS = {spec.model: spec for spec in (NEMOTRON_PACKET, KIMI_PACKET)}


EVIDENCE_LEVELS = {
    "isa_and_banked_matrix_sram": (
        "Executable Compiler encoding and Rust transactional implementation."
    ),
    "connected_projection": (
        "MLEN=64 synthetic numerical Matrix projection -> affine writeback -> "
        "view-qualified consumer; exact output comparison."
    ),
    "multi_token_recurrence": (
        "Four-token reduced-shape numerical Mamba and KDA recurrence with persistent "
        "Matrix-SRAM state. This is a BF16 storage candidate, not the official FP32 path."
    ),
    "official_shape_packets": (
        "Physical 2048-value Matrix-SRAM roundtrips using official head and row "
        "dimensions; D' and D differ only in the compiler-selected skew."
    ),
    "full_model_timeline": (
        "Official 52/93-layer structure, tensor dimensions, measured GPU calibration "
        "and symbolic PLENA weights in an analytic timeline."
    ),
    "not_demonstrated": (
        "No real-weight first-to-last-layer Rust execution for Nemotron 3 or Kimi K3; "
        "no RTL timing, synthesis, PPA or silicon energy measurement."
    ),
}


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
) -> tuple[int, int]:
    words_per_row = spec.elements_per_head // hardware.bank_width
    row_groups = math.ceil(words_per_row / hardware.banks)
    bank_row = (
        base_bank_row
        + tile * tile_pitch_rows
        + row * row_groups
        + word // hardware.banks
    )
    bank = (
        alpha * bank_row
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
) -> dict[str, Any]:
    """Move one real paper-shape packet through numbered physical bank words."""

    hardware = hardware or MatrixHardwarePoint()
    if not 0 < spec.packet_values <= hardware.mlen:
        raise ValueError(
            f"{spec.model}: packet has {spec.packet_values} values, outside "
            f"one MLEN={hardware.mlen} issue"
        )
    if spec.elements_per_head % hardware.bank_width:
        raise ValueError(f"{spec.model}: head row does not contain whole bank words")
    words_per_head = spec.elements_per_head // hardware.bank_width
    if not 0 <= alpha < hardware.banks or not 0 <= gamma < hardware.banks:
        raise ValueError("alpha and gamma must fit the Matrix bank index")

    cells: dict[tuple[int, int], tuple[int, ...]] = {}
    expected: list[tuple[int, ...]] = []
    bank_load: Counter[int] = Counter()
    for tile in range(spec.packet_heads):
        for word in range(words_per_head):
            logical = tuple(
                tile * spec.elements_per_head + word * hardware.bank_width + lane
                for lane in range(hardware.bank_width)
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
            )
            restored.append(cells[coord])
    if restored != expected:
        raise AssertionError("inverse lane restoration changed logical packet order")

    wrong_layout_detected = False
    wrong_alpha = (alpha + 1) % hardware.banks
    try:
        wrong = [
            cells[
                _physical_word(
                    tile=tile,
                    row=row,
                    word=word,
                    spec=spec,
                    hardware=hardware,
                    alpha=wrong_alpha,
                    gamma=gamma,
                    base_bank_row=base_bank_row,
                )
            ]
            for tile in range(spec.packet_heads)
            for word in range(words_per_head)
        ]
        wrong_layout_detected = wrong != expected
    except KeyError:
        wrong_layout_detected = True
    if not wrong_layout_detected:
        raise AssertionError("wrong Matrix skew was not detected by data movement")

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
        "map": {"alpha": alpha, "gamma": gamma},
        "service_cycles": service,
        "ideal_cycles": ideal,
        "bank_stall_cycles": service - ideal,
        "banks_touched": len(bank_load),
        "worst_bank_words": service,
        "roundtrip_values_checked": spec.packet_values,
        "wrong_alpha_changes_data": wrong_layout_detected,
        "packet_occupancy_bytes": spec.packet_values * hardware.matrix_element_bits // 8,
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
) -> dict[str, Any]:
    """Move one ordinary Matrix row/column through the same affine cells."""

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
        bank_row = row * row_groups + word // hardware.banks
        bank = (
            alpha * bank_row
            + gamma * (bank_row // hardware.banks)
            + word
        ) % hardware.banks
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
                    alpha * (row * row_groups + (word_col // hardware.bank_width) // hardware.banks)
                    + gamma
                    * (
                        (row * row_groups + (word_col // hardware.bank_width) // hardware.banks)
                        // hardware.banks
                    )
                    + word_col // hardware.bank_width
                )
                % hardware.banks,
                row * row_groups + (word_col // hardware.bank_width) // hardware.banks,
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
        MatrixVariant.C_MULTIROW_ORIGINAL: (1, 0),
        MatrixVariant.D_PRIME_BEST_FIXED: (
            int(fixed["alpha"]),
            int(fixed["gamma"]),
        ),
        # Attention's tensor view keeps the prior-work diagonal alpha=1.  It
        # does not inherit KDA's row-width-dependent alpha=4.
        MatrixVariant.D_AFFINE: (1, int(fixed["gamma"])),
    }
    records = {
        variant: {
            "moe_and_projection_row": measure_matrix_line(
                axis="row",
                rows=128,
                cols=hardware.mlen,
                hardware=hardware,
                alpha=alpha,
                gamma=gamma,
            ),
            "attention_qkt_column": measure_matrix_line(
                axis="column",
                rows=128,
                cols=hardware.mlen,
                hardware=hardware,
                alpha=alpha,
                gamma=gamma,
            ),
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
            "per_row": hardware.mlen,
            "per_column": 128,
        },
    }


def build_physical_evidence(
    hardware: MatrixHardwarePoint | None = None,
) -> dict[str, Any]:
    hardware = hardware or MatrixHardwarePoint()
    # D' is one hardwired map for the complete device.  It must preserve the
    # paper's ordinary 128-value column read at its two-cycle floor, then
    # minimise the worst normalised service across the two recurrent packets.
    # This is deliberately stricter than choosing a different fixed map per
    # model, which would already be a programmable design.
    fixed_candidates: list[tuple[tuple[float, float, int, int], int, int]] = []
    for alpha in range(hardware.banks):
        for gamma in range(hardware.banks):
            column_load = Counter(
                (
                    alpha * row
                    + gamma * (row // hardware.banks)
                )
                % hardware.banks
                for row in range(128)
            )
            column_service = max(column_load.values(), default=0)
            column_ideal = math.ceil(128 / hardware.banks)
            if column_service != column_ideal:
                continue
            services = [
                measure_packet(
                    spec,
                    hardware=hardware,
                    alpha=alpha,
                    gamma=gamma,
                )["service_cycles"]
                for spec in PACKETS.values()
            ]
            normalised = [
                service
                / math.ceil(
                    (spec.packet_heads * (spec.elements_per_head // hardware.bank_width))
                    / hardware.banks
                )
                for service, spec in zip(services, PACKETS.values(), strict=True)
            ]
            fixed_candidates.append(
                ((max(normalised), sum(normalised), alpha, gamma), alpha, gamma)
            )
    if not fixed_candidates:
        raise AssertionError("no fixed map preserves the ordinary column-read floor")
    _, fixed_alpha, fixed_gamma = min(fixed_candidates)

    result: dict[str, Any] = {
        "global_fixed_map": {
            "alpha": fixed_alpha,
            "gamma": fixed_gamma,
            "search_points": hardware.banks**2,
            "eligible_without_column_regression": len(fixed_candidates),
            "selection_rule": (
                "preserve the 128-value column-read floor; then minimise worst "
                "normalised Mamba/KDA packet service"
            ),
        }
    }
    for spec in PACKETS.values():
        original = measure_packet(spec, hardware=hardware, alpha=1, gamma=0)
        d_prime = measure_packet(
            spec,
            hardware=hardware,
            alpha=fixed_alpha,
            gamma=fixed_gamma,
        )
        affine = measure_packet(
            spec,
            hardware=hardware,
            alpha=spec.elements_per_head // hardware.bank_width,
            gamma=fixed_gamma,
        )
        if affine["service_cycles"] != affine["ideal_cycles"]:
            raise AssertionError("compiler per-tile skew did not reach the bank floor")
        result[spec.model] = {
            "C_original_fixed": original,
            "D_prime_best_global_fixed": d_prime,
            "D_compiler_per_tile_affine": affine,
            "fixed_alpha_gamma_search_points": hardware.banks**2,
            "best_fixed_service_cycles": d_prime["service_cycles"],
            "pure_layout_speedup_D_over_D_prime": (
                d_prime["service_cycles"] / affine["service_cycles"]
            ),
        }
    return result


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
        from compiler.aten.plena.matrix_packet_report import build_report as build_packet_report
        from compiler.aten.plena.matrix_prefill_handoff import (
            build_prefill_handoff_report,
        )
        from compiler.aten.plena.matrix_recurrence_lowering import (
            build_matrix_recurrence_report,
        )

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
            "issue": issue,
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
    new = compiler["matrix_recurrence"]["models"][
        "nemotron3_mamba2" if spec.model == "nemotron3" else "kimi_k3_kda"
    ]["metrics"]
    return {
        "A": int(old["baseline"]["dynamic_issued_instructions"]),
        "B": int(old["postincrement_only"]["dynamic_issued_instructions"]),
        "packet_core": int(new["dynamic_issued_instructions"]),
        "packet_reads": int(new["packet_reads"]),
        "packet_writes": int(new["packet_writes"]),
    }


def _validate_real_packet_contract(
    compiler: dict[str, Any], spec: RecurrentPacketSpec
) -> dict[str, Any]:
    stage = (
        "nemotron3_mamba2_matrix_recurrence"
        if spec.model == "nemotron3"
        else "kimi_k3_kda_matrix_recurrence"
    )
    matches = [
        case
        for case in compiler["real_packets"]["cases"]
        if case["stage"] == stage and case["lowering"] == "matrix_recurrence_affine"
    ]
    if len(matches) != 1:
        raise AssertionError(f"{spec.model}: expected one affine recurrence packet case")
    case = matches[0]
    read_groups = [
        entry
        for entry in case["coissued_histogram"]
        if entry["direction"] == "read"
    ]
    if not read_groups or not any(entry["same_cycle_operands"] == 2 for entry in read_groups):
        raise AssertionError(f"{spec.model}: no same-cycle two-source Matrix packet")
    for group in read_groups:
        for operand in group["operands"]:
            if (
                operand["tiles"] != spec.packet_heads
                or operand["elements_per_tile"] != spec.elements_per_head
            ):
                raise AssertionError(
                    f"{spec.model}: compiler packet shape {operand} does not match {spec}"
                )
    return {
        "stage": stage,
        "same_cycle_read_shapes": read_groups,
        "dynamic_packet_repeats": case["dynamic_packet_repeats"],
        "source": case["source"],
        "service_groups": case["service_groups"],
    }


def _real_fixed_map_signatures(
    *,
    compiler: dict[str, Any],
    spec: RecurrentPacketSpec,
    hardware: MatrixHardwarePoint,
) -> tuple[Counter[tuple[tuple[int, int], ...]], int]:
    """Compress the official FP32 real-lowering addresses for D' search.

    A fixed-map candidate needs only ``bank_row mod banks^2`` and ``word mod
    banks``.  Equal dynamic packets are counted once, which keeps an exhaustive
    4096-point search practical without replacing the real emitted strides by a
    synthetic packet.
    """

    signatures: Counter[tuple[tuple[int, int], ...]] = Counter()
    ideal_cycles = 0
    modulus = hardware.banks * hardware.banks
    for group in _validate_real_packet_contract(compiler, spec)["service_groups"]:
        if group["direction"] != "read":
            continue
        operands = [
            operand for operand in group["operands"] if operand["name"] == "source2"
        ]
        if not operands:
            continue
        for repeat in range(int(group["repeats"])):
            coordinates: list[tuple[int, int]] = []
            for operand in operands:
                base = operand["matrix_address"]
                stride = operand["address_stride_elements"]
                if not isinstance(base, int) or not isinstance(stride, int):
                    raise AssertionError("D' search requires resolved Matrix addresses")
                address = base + repeat * stride
                if address % hardware.mlen:
                    raise AssertionError("D' search address is not Matrix-row aligned")
                words_per_row = int(operand["view_cols"]) // hardware.bank_width
                row_groups = math.ceil(words_per_row / hardware.banks)
                base_bank_row = address // hardware.mlen
                for tile in range(int(operand["tiles"])):
                    for row in range(int(operand["view_rows"])):
                        for word in range(words_per_row):
                            bank_row = (
                                base_bank_row
                                + tile * int(operand["tile_pitch_rows"])
                                + row * row_groups
                                + word // hardware.banks
                            )
                            coordinates.append((bank_row % modulus, word % hardware.banks))
            signature = tuple(coordinates)
            signatures[signature] += 1
            ideal_cycles += math.ceil(len(coordinates) / hardware.banks)
    if not signatures:
        raise AssertionError(f"{spec.model}: no real Matrix packets for D' search")
    return signatures, ideal_cycles


def _score_fixed_map(
    signatures: Counter[tuple[tuple[int, int], ...]],
    *,
    alpha: int,
    gamma: int,
    banks: int,
) -> int:
    total = 0
    for signature, repeats in signatures.items():
        loads = [0] * banks
        for bank_row, word in signature:
            bank = (
                alpha * bank_row
                + gamma * (bank_row // banks)
                + word
            ) % banks
            loads[bank] += 1
        total += max(loads, default=0) * repeats
    return total


def apply_real_fixed_map_selection(
    *,
    compiler: dict[str, Any],
    physical: dict[str, Any],
    hardware: MatrixHardwarePoint,
) -> None:
    """Select D' on real Compiler lowerings, then rebuild its checked packets."""

    real = {
        spec.model: _real_fixed_map_signatures(
            compiler=compiler,
            spec=spec,
            hardware=hardware,
        )
        for spec in PACKETS.values()
    }
    candidates: list[tuple[tuple[float, float, int, int], int, int, dict[str, int]]] = []
    eligible = 0
    column_ideal = math.ceil(128 / hardware.banks)
    for alpha in range(hardware.banks):
        for gamma in range(hardware.banks):
            column_load = Counter(
                (alpha * row + gamma * (row // hardware.banks)) % hardware.banks
                for row in range(128)
            )
            if max(column_load.values(), default=0) != column_ideal:
                continue
            eligible += 1
            services = {
                model: _score_fixed_map(
                    signatures,
                    alpha=alpha,
                    gamma=gamma,
                    banks=hardware.banks,
                )
                for model, (signatures, _ideal) in real.items()
            }
            normalised = {
                model: services[model] / ideal
                for model, (_signatures, ideal) in real.items()
            }
            candidates.append(
                (
                    (
                        max(normalised.values()),
                        sum(normalised.values()),
                        alpha,
                        gamma,
                    ),
                    alpha,
                    gamma,
                    services,
                )
            )
    if not candidates:
        raise AssertionError("no real-lowering fixed map preserves the column floor")
    _score, fixed_alpha, fixed_gamma, services = min(candidates)
    physical["global_fixed_map"] = {
        "alpha": fixed_alpha,
        "gamma": fixed_gamma,
        "search_points": hardware.banks**2,
        "eligible_without_column_regression": eligible,
        "selection_rule": (
            "preserve the 128-value column-read floor; then minimise worst "
            "normalised service over official-FP32 Compiler dynamic addresses"
        ),
        "search_input": "real Compiler service_groups and dynamic address strides",
        "real_lowering_service_cycles": services,
    }
    for spec in PACKETS.values():
        d_prime = measure_packet(
            spec,
            hardware=hardware,
            alpha=fixed_alpha,
            gamma=fixed_gamma,
        )
        affine = measure_packet(
            spec,
            hardware=hardware,
            alpha=spec.elements_per_head // hardware.bank_width,
            gamma=fixed_gamma,
        )
        if affine["service_cycles"] != affine["ideal_cycles"]:
            raise AssertionError("real-search affine map did not reach the packet floor")
        physical[spec.model]["D_prime_best_global_fixed"] = d_prime
        physical[spec.model]["D_compiler_per_tile_affine"] = affine
        physical[spec.model]["best_fixed_service_cycles"] = d_prime["service_cycles"]
        physical[spec.model]["pure_layout_speedup_D_over_D_prime"] = (
            d_prime["service_cycles"] / affine["service_cycles"]
        )
        physical[spec.model]["fixed_map_selected_from_real_lowering"] = True


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
    """Replay every dynamic Matrix packet address from the real lowering.

    Co-issued operands share one physical read port per bank, so their bank
    loads are combined before service is calculated.  Read and write groups of
    one instruction are then joined to expose both the unavoidable port floor
    and conflict-only stalls.  Numbered bank words are written and read back for
    every operand; this is data-movement evidence, not a cycle formula alone.
    """

    contract = _validate_real_packet_contract(compiler, spec)
    groups = contract["service_groups"]
    per_issue_service: Counter[tuple[int, int]] = Counter()
    per_issue_ideal: Counter[tuple[int, int]] = Counter()
    instruction_repeats: dict[int, int] = {}
    total_values = 0
    total_bank_words = 0
    total_operands = 0
    total_groups = 0
    worst_group_service = 0
    service_histogram: Counter[tuple[str, int, int]] = Counter()

    for group in groups:
        direction = str(group["direction"])
        operands = list(group["operands"])
        if state_mode == StateMode.OFFICIAL_FP32:
            # The official state remains explicit FP32 traffic outside Matrix
            # SRAM.  Only the BF16 field operand produced by projection uses a
            # Matrix view; no state write is credited to the layout mechanism.
            if direction != "read":
                continue
            operands = [operand for operand in operands if operand["name"] == "source2"]
        if not operands:
            continue

        instruction = int(group["instruction_index"])
        repeats = int(group["repeats"])
        previous = instruction_repeats.setdefault(instruction, repeats)
        if previous != repeats:
            raise AssertionError("one Matrix instruction has inconsistent repeat counts")
        total_groups += repeats
        for repeat in range(repeats):
            bank_load: Counter[int] = Counter()
            words_this_group = 0
            for operand_index, operand in enumerate(operands):
                base = operand["matrix_address"]
                stride = operand["address_stride_elements"]
                if not isinstance(base, int) or not isinstance(stride, int):
                    raise AssertionError(
                        f"{spec.model}: unresolved dynamic Matrix address in {operand}"
                    )
                address = base + repeat * stride
                if address % hardware.mlen:
                    raise AssertionError(
                        f"{spec.model}: Matrix address {address} is not row aligned"
                    )
                rows = int(operand["view_rows"])
                cols = int(operand["view_cols"])
                tiles = int(operand["tiles"])
                pitch = int(operand["tile_pitch_rows"])
                if cols % hardware.bank_width:
                    raise AssertionError("Matrix view row is not a whole number of bank words")
                words_per_row = cols // hardware.bank_width
                row_groups = math.ceil(words_per_row / hardware.banks)
                if variant == MatrixVariant.C_MULTIROW_ORIGINAL:
                    alpha, gamma = 1, 0
                elif variant == MatrixVariant.D_PRIME_BEST_FIXED:
                    alpha, gamma = fixed_alpha, fixed_gamma
                elif variant in (MatrixVariant.D_AFFINE, MatrixVariant.E_AFFINE_OVERLAP):
                    alpha, gamma = int(operand["view_alpha"]), fixed_gamma
                else:
                    raise ValueError(f"{variant} has no Matrix packet service")

                # Each allocation has its own logical payload.  Coordinates
                # must be unique within it; reading through the same mapping
                # must return every numbered bank word in logical order.
                cells: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
                expected: list[tuple[int, int, int, int, int]] = []
                base_bank_row = address // hardware.mlen
                for tile in range(tiles):
                    for row in range(rows):
                        for word in range(words_per_row):
                            bank_row = (
                                base_bank_row
                                + tile * pitch
                                + row * row_groups
                                + word // hardware.banks
                            )
                            bank = (
                                alpha * bank_row
                                + gamma * (bank_row // hardware.banks)
                                + word
                            ) % hardware.banks
                            logical = (operand_index, repeat, tile, row, word)
                            coord = (bank, bank_row)
                            if coord in cells:
                                raise AssertionError(
                                    f"{spec.model}: Matrix view aliases at {coord}"
                                )
                            cells[coord] = logical
                            expected.append(logical)
                            bank_load[bank] += 1
                            words_this_group += 1
                restored = [
                    cells[
                        (
                            (
                                alpha
                                * (
                                    base_bank_row
                                    + tile * pitch
                                    + row * row_groups
                                    + word // hardware.banks
                                )
                                + gamma
                                * (
                                    (
                                        base_bank_row
                                        + tile * pitch
                                        + row * row_groups
                                        + word // hardware.banks
                                    )
                                    // hardware.banks
                                )
                                + word
                            )
                            % hardware.banks,
                            base_bank_row
                            + tile * pitch
                            + row * row_groups
                            + word // hardware.banks,
                        )
                    ]
                    for tile in range(tiles)
                    for row in range(rows)
                    for word in range(words_per_row)
                ]
                if restored != expected:
                    raise AssertionError("Matrix lane restoration changed logical word order")

            service = max(bank_load.values(), default=0)
            ideal = math.ceil(words_this_group / hardware.banks)
            # Matrix SRAM keeps its existing one-read/one-write port per bank.
            # A read packet and the destination writeback of the same
            # instruction therefore overlap; two read operands above already
            # share and contend for the single read port. Charge the slower
            # direction, not an impossible serial read+write sum.
            key = (instruction, repeat)
            per_issue_service[key] = max(per_issue_service[key], service)
            per_issue_ideal[key] = max(per_issue_ideal[key], ideal)
            total_bank_words += words_this_group
            total_values += words_this_group * hardware.bank_width
            total_operands += len(operands)
            worst_group_service = max(worst_group_service, service)
            service_histogram[(direction, ideal, service)] += 1

    dynamic_instructions = sum(instruction_repeats.values())
    service_cycles = sum(per_issue_service.values())
    ideal_cycles = sum(per_issue_ideal.values())
    extra_cycles_over_issue = sum(
        max(0, service - 1) for service in per_issue_service.values()
    )
    return {
        "dynamic_matrix_instructions": dynamic_instructions,
        "dynamic_service_groups": total_groups,
        "dynamic_operands": total_operands,
        "values_roundtrip_checked": total_values,
        "bank_words": total_bank_words,
        "ideal_cycles": ideal_cycles,
        "service_cycles": service_cycles,
        "bank_stall_cycles": service_cycles - ideal_cycles,
        "extra_cycles_over_issue": extra_cycles_over_issue,
        "worst_group_service_cycles": worst_group_service,
        "service_histogram": [
            {
                "direction": direction,
                "ideal_cycles": ideal,
                "service_cycles": service,
                "dynamic_groups": count,
            }
            for (direction, ideal, service), count in sorted(service_histogram.items())
        ],
        "source": "Compiler real service_groups with dynamic address strides",
    }


def attach_real_service_evidence(
    *,
    compiler: dict[str, Any],
    physical: dict[str, Any],
    hardware: MatrixHardwarePoint,
) -> None:
    """Attach exact C/D'/D service totals without changing D' selection."""

    fixed = physical["global_fixed_map"]
    for spec in PACKETS.values():
        modes: dict[str, Any] = {}
        for state_mode in StateMode:
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
                    MatrixVariant.C_MULTIROW_ORIGINAL,
                    MatrixVariant.D_PRIME_BEST_FIXED,
                    MatrixVariant.D_AFFINE,
                    MatrixVariant.E_AFFINE_OVERLAP,
                )
            }
        physical[spec.model]["real_lowering_service"] = modes


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
    original = int(physical[spec.model]["C_original_fixed"]["service_cycles"])
    fixed = int(physical[spec.model]["D_prime_best_global_fixed"]["service_cycles"])
    affine = int(physical[spec.model]["D_compiler_per_tile_affine"]["service_cycles"])

    if variant == MatrixVariant.A_ORIGINAL:
        return {
            "cycles": counts["A"] * batch_size,
            "issued": counts["A"] * batch_size,
            "packet_ops": 0,
            "service": 0,
            "ideal": 0,
            "stall": 0,
        }
    if variant == MatrixVariant.B_ARLO:
        return {
            "cycles": counts["B"] * batch_size,
            "issued": counts["B"] * batch_size,
            "packet_ops": 0,
            "service": 0,
            "ideal": 0,
            "stall": 0,
        }

    service_per_packet = {
        MatrixVariant.C_MULTIROW_ORIGINAL: original,
        MatrixVariant.D_PRIME_BEST_FIXED: fixed,
        MatrixVariant.D_AFFINE: affine,
        MatrixVariant.E_AFFINE_OVERLAP: affine,
    }[variant]
    exact_modes = physical[spec.model].get("real_lowering_service")
    if exact_modes is not None:
        exact = exact_modes[state_mode][variant]
        issued = (
            counts["B"] + 2
            if state_mode == StateMode.OFFICIAL_FP32
            else counts["packet_core"]
        )
        return {
            "cycles": (issued + int(exact["extra_cycles_over_issue"])) * batch_size,
            "issued": issued * batch_size,
            "packet_ops": int(exact["dynamic_service_groups"]) * batch_size,
            "service": int(exact["service_cycles"]) * batch_size,
            "ideal": int(exact["ideal_cycles"]) * batch_size,
            "stall": int(exact["bank_stall_cycles"]) * batch_size,
        }
    if state_mode == StateMode.OFFICIAL_FP32:
        # Only BF16 projection/temporary packets enter Matrix SRAM.  State
        # remains explicit FP32 traffic, so no state access is credited here.
        packet_ops = spec.bf16_field_packets
        issued = counts["B"] + 2  # two packed FULL configurations
        service = packet_ops * service_per_packet
        ideal = packet_ops
        cycles = issued + service - ideal
    else:
        # Architectural upper bound: every state/field packet uses the BF16
        # Matrix path.  The accuracy campaign, not this timing model, decides
        # whether this point is usable.
        packet_ops = counts["packet_reads"] + counts["packet_writes"]
        issued = counts["packet_core"]
        service = packet_ops * service_per_packet
        ideal = packet_ops
        cycles = issued + service - ideal
    return {
        "cycles": cycles * batch_size,
        "issued": issued * batch_size,
        "packet_ops": packet_ops * batch_size,
        "service": service * batch_size,
        "ideal": ideal * batch_size,
        "stall": (service - ideal) * batch_size,
    }


_MAMBA_RECURRENCE = {"mamba_state_update", "mamba_state_output"}
_KDA_RECURRENCE = {
    "kda_qk_l2norm",
    "kda_state_decay_prediction",
    "kda_delta_update_output",
    "kda_output_gate_rmsnorm",
}


def _is_recurrence(stage: StageWork, model: str) -> bool:
    return stage.name in (_MAMBA_RECURRENCE if model == "nemotron3" else _KDA_RECURRENCE)


def _projection_name(model: str) -> str:
    return "mamba_in_projection" if model == "nemotron3" else "kda_qkv_projection"


def _physical_hbm_bytes(stage: StageWork, burst: int) -> tuple[int, int]:
    read = stage.traffic.logical_hbm_read_bytes
    write = stage.traffic.logical_hbm_write_bytes
    return (
        math.ceil(read / burst) * burst if read else 0,
        math.ceil(write / burst) * burst if write else 0,
    )


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
    core = recurrent_core_metrics(
        spec=spec,
        variant=variant,
        state_mode=state_mode,
        compiler=compiler,
        physical=physical,
        batch_size=report.scenario.batch_size,
    )
    totals: dict[str, Any] = {
        "cycles": 0,
        "hbm_cycles": 0,
        "matrix_cycles": 0,
        "vector_cycles": 0,
        "overlap_cycles": 0,
        "logical_hbm_read_bytes": 0,
        "logical_hbm_write_bytes": 0,
        "physical_hbm_read_bytes": 0,
        "physical_hbm_write_bytes": 0,
        "packet_ops": 0,
        "matrix_sram_service_cycles": 0,
        "bank_stall_cycles": 0,
        "dynamic_issued_instructions": 0,
        "by_layer_type": defaultdict(lambda: defaultdict(int)),
    }
    recurrence_done: set[int] = set()
    projection_cycles: dict[int, int] = defaultdict(int)

    for stage in report.stages:
        matrix, vector = _generic_compute(stage, hardware)  # type: ignore[arg-type]
        if stage.name == _projection_name(model):
            projection_cycles[stage.layer_id] += matrix

        if report.scenario.phase == InferencePhase.DECODE and _is_recurrence(stage, model):
            matrix = vector = 0
            if stage.layer_id not in recurrence_done:
                recurrence_done.add(stage.layer_id)
                vector = core["cycles"]
                totals["dynamic_issued_instructions"] += core["issued"]
                totals["packet_ops"] += core["packet_ops"]
                totals["matrix_sram_service_cycles"] += core["service"]
                totals["bank_stall_cycles"] += core["stall"]
                if variant == MatrixVariant.E_AFFINE_OVERLAP:
                    overlap = min(projection_cycles[stage.layer_id], core["cycles"])
                    totals["overlap_cycles"] += overlap
                    totals["by_layer_type"][stage.layer_type]["overlap_cycles"] += overlap

        physical_read, physical_write = _physical_hbm_bytes(stage, hardware.hbm_burst_bytes)
        hbm = math.ceil(
            (physical_read + physical_write) / hardware.hbm_bytes_per_cycle
        )
        totals["hbm_cycles"] += hbm
        totals["matrix_cycles"] += matrix
        totals["vector_cycles"] += vector
        totals["logical_hbm_read_bytes"] += stage.traffic.logical_hbm_read_bytes
        totals["logical_hbm_write_bytes"] += stage.traffic.logical_hbm_write_bytes
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
        - totals["overlap_cycles"]
    )
    totals["latency_us_proxy"] = totals["cycles"] / (hardware.clock_hz / 1_000_000)
    totals["by_layer_type"] = {
        name: {
            **values,
            "cycles": values["hbm_cycles"]
            + values["matrix_cycles"]
            + values["vector_cycles"]
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
) -> tuple[int | None, tuple[tuple[int, int], ...]]:
    if (
        profile is not None
        and model == "nemotron3"
        and batch_size == profile.batch_size
        and context_length == profile.context_length
    ):
        key = "prefill" if phase == InferencePhase.PREFILL else "decode"
        index = 0 if key == "prefill" else decode_index
        matches = [step for step in profile.steps if (step.phase, step.index) == (key, index)]
        if len(matches) == 1 and matches[0].token_count == batch_size * sequence_length:
            return None, matches[0].unique_experts_by_layer
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
) -> list[WorkloadReport]:
    state_precision = Precision.FP32 if state_mode == StateMode.OFFICIAL_FP32 else Precision.BF16
    workload = _model(
        model,
        compiler_root,
        activation_precision=Precision.BF16,
        weight_precision=None,
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
) -> dict[str, Any]:
    reports = build_reports(
        model=model,
        phase=phase,
        batch_size=batch_size,
        tokens=tokens,
        context_length=context_length,
        state_mode=state_mode,
        compiler_root=compiler_root,
        profile=profile,
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
            "overlap_cycles",
            "logical_hbm_read_bytes",
            "logical_hbm_write_bytes",
            "physical_hbm_read_bytes",
            "physical_hbm_write_bytes",
            "packet_ops",
            "matrix_sram_service_cycles",
            "bank_stall_cycles",
            "dynamic_issued_instructions",
        )
        record = {name: sum(piece[name] for piece in pieces) for name in numeric}
        record["variant"] = variant
        record["latency_us_proxy"] = record["cycles"] / (hardware.clock_hz / 1_000_000)
        layer_types = {name for piece in pieces for name in piece["by_layer_type"]}
        record["by_layer_type"] = {
            layer_type: {
                metric: sum(
                    piece["by_layer_type"].get(layer_type, {}).get(metric, 0)
                    for piece in pieces
                )
                for metric in ("cycles", "hbm_cycles", "matrix_cycles", "vector_cycles", "overlap_cycles")
            }
            for layer_type in sorted(layer_types)
        }
        records.append(record)

    by_variant = {record["variant"]: record for record in records}
    for record in records:
        record["speedup_vs_A"] = by_variant[MatrixVariant.A_ORIGINAL]["cycles"] / record["cycles"]
        record["speedup_vs_B"] = by_variant[MatrixVariant.B_ARLO]["cycles"] / record["cycles"]
        record["speedup_vs_D_prime"] = (
            by_variant[MatrixVariant.D_PRIME_BEST_FIXED]["cycles"] / record["cycles"]
        )
        record["speedup_vs_D"] = (
            by_variant[MatrixVariant.D_AFFINE]["cycles"] / record["cycles"]
        )

    ordinary = ("attention", "moe", "mla", "latent_moe", "dense", "attn_res")
    for layer_type in ordinary:
        observed = {
            record["by_layer_type"].get(layer_type, {}).get("cycles", 0)
            for record in records
        }
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
        "layer_counts": layer_counts,
        "ordinary_attention_moe_cycles_identical": True,
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
        "nemotron_mamba_s32768": _cached_gpu_report()["b200_supplemental"][
            "mamba_precision_s32768"
        ],
        "interpretation": (
            "Official timing uses FP32 state. BF16/FP16/MX8 are storage experiments "
            "with FP32 update/reduction and are not checkpoint quality results."
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
            "reason": (
                f"row width {row_width} is not a whole {hardware.bank_width}-value "
                "Matrix bank word"
            ),
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
    original = measure_packet(
        spec,
        hardware=hardware,
        alpha=1,
        gamma=0,
    )
    affine = measure_packet(
        spec,
        hardware=hardware,
        alpha=row_width // hardware.bank_width,
        gamma=1,
    )
    return {
        "name": name,
        "row_width": row_width,
        "packet_width": packet_width,
        "supported": True,
        "C_original_fixed": original,
        "D_compiler_per_tile_affine": affine,
        "pure_service_speedup_C_over_D": (
            original["service_cycles"] / affine["service_cycles"]
        ),
        "values_checked_per_variant": packet_width,
    }


def build_layout_dse(
    *,
    hardware: MatrixHardwarePoint,
    experiments: dict[str, Any],
) -> dict[str, Any]:
    """Sweep layout geometry without changing arithmetic or adding ports."""

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
                    "bank_select_adders": point.banks,
                    "bank_select_adder_bits": int(math.log2(point.banks)),
                    "lane_restore_payload_bits": (
                        point.mlen * point.matrix_element_bits
                    ),
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
            item
            for item in experiments[mode][model][case]["records"]
            if item["variant"] == MatrixVariant.D_AFFINE
        )

    precision = {
        mode: {
            model: {
                "cycles": record(mode, model, "decode_b1_t1")["cycles"],
                "physical_hbm_read_bytes": record(mode, model, "decode_b1_t1")[
                    "physical_hbm_read_bytes"
                ],
                "bank_stall_cycles": record(mode, model, "decode_b1_t1")[
                    "bank_stall_cycles"
                ],
            }
            for model in ("nemotron3", "kimi_k3")
        }
        for mode in StateMode
    }
    batch = {
        model: [
            {
                "batch": batch_size,
                "cycles": record(
                    StateMode.OFFICIAL_FP32,
                    model,
                    f"decode_b{batch_size}_t1",
                )["cycles"],
                "bank_stall_cycles": record(
                    StateMode.OFFICIAL_FP32,
                    model,
                    f"decode_b{batch_size}_t1",
                )["bank_stall_cycles"],
            }
            for batch_size in (1, 2, 4, 8, 16)
        ]
        for model in ("nemotron3", "kimi_k3")
    }
    return {
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
    """Report the KDA identity-transpose removal separately from decode layout.

    The regular prefill timeline models the useful layer work and intentionally
    excludes the legacy identity GEMM.  This function composes that measured
    emitted-GEMM cost serially with the BF16 candidate timeline.  It charges one
    complete five-instruction view configuration per KDA layer, even though an
    implementation may keep the common descriptor live across layers.  Official
    FP32 state is not eligible because it does not reside in the BF16 Matrix SRAM.
    """

    legacy = handoff["legacy_identity_gemm"]
    view = handoff["matrix_view_handoff"]
    shape = handoff["shape"]
    legacy_cycles = int(legacy["emitted_matrix_cycles_all_kda_layers"])
    view_config_cycles = (
        int(view["configuration_dynamic_instructions"]) * int(shape["kda_layers"])
    )

    cases: dict[str, Any] = {}
    for tokens in (16, 128):
        case_name = f"prefill_b1_s{tokens}"
        records = experiments[StateMode.BF16_CANDIDATE]["kimi_k3"][case_name][
            "records"
        ]
        affine = next(
            record
            for record in records
            if record["variant"] == MatrixVariant.D_AFFINE
        )
        useful_cycles = int(affine["cycles"])
        legacy_total = useful_cycles + legacy_cycles
        view_total = useful_cycles + view_config_cycles
        cases[case_name] = {
            "useful_prefill_cycles_without_handoff": useful_cycles,
            "legacy_identity_gemm_cycles": legacy_cycles,
            "legacy_serial_total_cycles": legacy_total,
            "view_configuration_cycles_upper_bound": view_config_cycles,
            "view_serial_total_cycles": view_total,
            "serial_timeline_speedup": legacy_total / view_total,
        }

    return {
        "scope": (
            "Kimi K3 BF16/MX8 candidate only; standalone serial boundary delta, "
            "not D over D' and not credited to the official FP32 timeline"
        ),
        "legacy_logical_macs_eliminated": int(
            legacy["logical_macs_all_kda_layers"]
        ),
        "legacy_emitted_padded_macs_eliminated": int(
            legacy["emitted_padded_macs_all_kda_layers"]
        ),
        "legacy_matrix_cycles_eliminated": legacy_cycles,
        "view_handoff_macs": int(view["handoff_macs"]),
        "view_configuration_policy": (
            "conservative: reconfigure the same general view once per KDA layer"
        ),
        "official_fp32_speedup_claimed": False,
        "cases": cases,
    }


def build_campaign(
    *,
    compiler_root: Path,
    simulator_root: Path,
) -> dict[str, Any]:
    hardware = MatrixHardwarePoint()
    compiler = load_compiler_evidence(str(compiler_root))
    packet_contracts = {
        spec.model: _validate_real_packet_contract(compiler, spec)
        for spec in PACKETS.values()
    }
    physical = build_physical_evidence(hardware)
    apply_real_fixed_map_selection(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
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
    for state_mode in StateMode:
        mode_cases: dict[str, Any] = {}
        for model in ("nemotron3", "kimi_k3"):
            cases: dict[str, Any] = {}
            for batch in (1, 2, 4, 8, 16):
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
                state_mode=StateMode.OFFICIAL_FP32,
                hardware=point,
                compiler_root=compiler_root,
                compiler=compiler,
                physical=physical,
                profile=(routing if model == "nemotron3" else None),
            )
            for model in ("nemotron3", "kimi_k3")
        }
    layout_dse = build_layout_dse(
        hardware=hardware,
        experiments=experiments,
    )
    prefill_handoff_delta = build_prefill_handoff_timeline_delta(
        experiments=experiments,
        handoff=compiler["prefill_handoff"],
    )

    return {
        "schema_version": 2,
        "outcome": (
            "Outcome 1 on real Compiler lowerings: per-tile alpha beats every "
            "non-regressing global fixed wiring for Kimi K3; Nemotron D' equals D."
        ),
        "status": "COMPLETE_PRE_RTL",
        "scope": "Compiler plus analytic/transactional Simulator; no RTL, synthesis or PPA",
        "evidence_levels": EVIDENCE_LEVELS,
        "hardware": asdict(hardware),
        "resource_proxies": hardware.resource_proxies(),
        "resource_proxies_by_variant": hardware.resource_proxies_by_variant(),
        "physical_packet_evidence": physical,
        "ordinary_attention_moe_no_regression": ordinary_no_regression,
        "compiler": {
            "source_root": compiler["compiler_root"],
            "plena_settings_toml": compiler["plena_settings_toml"],
            "issue_counts": {
                spec.model: _issue_counts(compiler, spec) for spec in PACKETS.values()
            },
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
            "packet_extraction_coverage": compiler["real_packets"]["coverage"],
            "packet_capacity_contract": compiler["real_packets"]["capacity_contract"],
            "packet_scope_boundary": compiler["real_packets"]["scope_boundary"],
            "prefill_to_decode_handoff": compiler["prefill_handoff"],
        },
        "experiments": experiments,
        "bandwidth_sweep": bandwidth_sweep,
        "layout_dse": layout_dse,
        "precision_evidence": _precision_evidence(simulator_root),
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
                    "no measured batch routing trace is available; use the conservative "
                    "maximum-active-expert bound"
                )
            },
        },
        "claim_boundaries": {
            "lcompute_credit": "D over D_prime only",
            "compiler_credit": "B over A only",
            "overlap_credit": "E over D only",
            "fp32_state": "explicit HBM traffic; no Matrix-SRAM residency or cache",
            "prefill": (
                "full workload timeline supported; official FP32 receives no view credit. "
                "The emitted identity-GEMM versus zero-MAC BF16/MX8 view handoff is "
                "reported separately, not folded into D over D'."
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
    (output_dir / "state_residency.json").write_text(
        json.dumps(campaign["state_residency"], indent=2) + "\n"
    )
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
                            "speedup_vs_D_prime": record["speedup_vs_D_prime"],
                            "speedup_vs_D": record["speedup_vs_D"],
                            "bank_stall_cycles": record["bank_stall_cycles"],
                            "matrix_sram_service_cycles": record["matrix_sram_service_cycles"],
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
    for mode in StateMode:
        for model in ("nemotron3", "kimi_k3"):
            d_prime = _record(
                campaign,
                str(mode),
                model,
                "decode_b1_t1",
                str(MatrixVariant.D_PRIME_BEST_FIXED),
            )
            affine = _record(
                campaign,
                str(mode),
                model,
                "decode_b1_t1",
                str(MatrixVariant.D_AFFINE),
            )
            summary.append(
                {
                    "state_mode": mode,
                    "model": model,
                    "D_prime_cycles": d_prime["cycles"],
                    "D_cycles": affine["cycles"],
                    "full_model_speedup_D_over_D_prime": d_prime["cycles"] / affine["cycles"],
                    "D_prime_bank_stall": d_prime["bank_stall_cycles"],
                    "D_bank_stall": affine["bank_stall_cycles"],
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
