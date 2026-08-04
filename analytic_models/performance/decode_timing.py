"""Decode timing contracts and their validation gates.

Three contracts, in increasing order of what they assume the hardware does:

- `rtl_serialized` — the implemented behaviour. `mxint_systolic_mcu.sv` accepts a
  new instruction only when `!draining`, so every accumulate waits for the
  previous writeout to stream out.
- `drain_overlapped` — the accumulate keeps its measured `3 * BLEN + 11`, but the
  writeout streams behind the next accumulate. This is one concrete RTL change:
  the drain reads `acc_fp_ph`, which the next accumulate immediately overwrites,
  so overlapping it needs a second accumulator bank.
- `ideal_matrix_pipeline` — an architectural oracle in which the accumulate also
  costs only `BLEN`. It bounds the array, not any implementable design.

Only `rtl_serialized` describes the source-derived current RTL behaviour. No
full-layer timing contract has silicon evidence. The other two are labelled
wherever they are used, and none may rank hardware without matched cycle
evidence.

Timing evidence carries an explicit tier. RTL-tier evidence
(mode `rtl_serialized`) anchors the analytic model to RTL cycle counts with
an emulator cross-check. Emulator-tier evidence (mode `emulator_serialized`)
anchors the same serialized issue contract to transactional-emulator cycle
counts only; its anchors carry no RTL measurements and every consumer
surfaces the tier so the two cannot be mistaken for one another.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

RTL_SERIALIZED = "rtl_serialized"
DRAIN_OVERLAPPED = "drain_overlapped"
IDEAL_MATRIX_PIPELINE = "ideal_matrix_pipeline"
TIMING_MODES = (RTL_SERIALIZED, DRAIN_OVERLAPPED, IDEAL_MATRIX_PIPELINE)
#: Evidence mode for the serialized issue contract whose cycle anchors are
#: measured on the transactional emulator instead of the RTL simulation. It
#: prices identically to ``rtl_serialized``; only the measurement reference
#: differs, and the evidence document carries that tier explicitly.
EMULATOR_SERIALIZED = "emulator_serialized"
TIMING_EVIDENCE_MODES = (*TIMING_MODES, EMULATOR_SERIALIZED)
RTL_EVIDENCE_TIER = "rtl"
EMULATOR_EVIDENCE_TIER = "emulator"
STEP_COMPOSITION = "max_compute_memory"
TIMING_EVIDENCE_SCHEMA = "plena-decode-timing-evidence"
REQUIRED_TIMING_ANCHOR_KINDS = ("linear", "qk", "pv", "vector", "layer")
REQUIRED_TIMING_PROVENANCE_ROLES = (
    "anchors",
    "compiler",
    "analytic",
    "emulator",
    "rtl",
)
EMULATOR_TIMING_PROVENANCE_ROLES = (
    "anchors",
    "compiler",
    "analytic",
    "emulator",
)
CANONICAL_ANCHOR_MAX_ERROR_LIMIT = 0.05
CANONICAL_ANALYTICAL_MAPE_LIMIT = 0.10


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def projection_mm_events(
    *,
    rows: int,
    input_width: int,
    query_width: int,
    kv_width: int,
    mlen: int,
    blen: int,
) -> tuple[int, int, int]:
    """Return exact tiled MM counts for Q, K, and V projections."""

    values = (rows, input_width, query_width, kv_width, mlen, blen)
    if any(value <= 0 for value in values):
        raise ValueError("projection dimensions must be positive")
    if mlen % blen:
        raise ValueError("MLEN must be divisible by BLEN")
    row_tiles = math.ceil(rows / blen)
    reduction_tiles = math.ceil(input_width / mlen)
    query = row_tiles * reduction_tiles * math.ceil(query_width / blen)
    kv = row_tiles * reduction_tiles * math.ceil(kv_width / blen)
    return query, kv, kv


def matrix_issue_cycles(
    opcode: str,
    blen: int,
    mode: str,
    *,
    mlen: int | None = None,
    hlen: int | None = None,
) -> int:
    """Return the measured latency or the explicit pipeline oracle interval."""

    if blen <= 0:
        raise ValueError("BLEN must be positive")
    if mode not in TIMING_MODES:
        raise ValueError(f"unknown timing mode {mode!r}")
    if opcode in {"M_MM", "M_TMM", "M_BMM", "M_BTMM"}:
        # Only the full oracle shortens the accumulate itself; overlapping the
        # drain leaves the measured issue latency untouched.
        return blen if mode == IDEAL_MATRIX_PIPELINE else 3 * blen + 11
    if opcode == "M_MM_WO":
        return blen + 6 if mode == RTL_SERIALIZED else 1
    if opcode == "M_BMM_WO":
        if mode in {IDEAL_MATRIX_PIPELINE, DRAIN_OVERLAPPED}:
            return 1
        if mlen is None or hlen is None:
            raise ValueError("M_BMM_WO timing requires MLEN and HLEN")
        if mlen <= 0 or hlen <= 0 or mlen % hlen:
            raise ValueError("M_BMM_WO timing requires HLEN to divide MLEN")
        return (mlen // hlen) * blen + 6
    if opcode in {"M_MV", "M_TMV", "M_BMV", "M_BTMV"}:
        return blen + 9
    if opcode in {"M_MV_WO", "M_BMV_WO"}:
        if mlen is None:
            raise ValueError(f"{opcode} timing requires MLEN")
        if mlen <= 0:
            raise ValueError("MLEN must be positive")
        return mlen + 6 if mode == RTL_SERIALIZED else 1
    raise ValueError(f"unsupported matrix opcode {opcode!r}")


def cycles_to_seconds(
    cycles: int,
    *,
    frequency_hz: float,
    compute_density: float = 1.0,
    chip_count: int = 1,
) -> float:
    """Convert already-complete cycle counts without a hidden scale factor."""

    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    if not math.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if not math.isfinite(compute_density) or compute_density <= 0:
        raise ValueError("compute_density must be positive")
    if chip_count <= 0:
        raise ValueError("chip_count must be positive")
    return cycles / frequency_hz / compute_density / chip_count


@dataclass(frozen=True)
class MatrixInstructionMix:
    """Matrix instruction counts plus cycles outside the selected opcodes."""

    counts: tuple[tuple[str, int], ...]
    other_cycles: int = 0

    def __post_init__(self) -> None:
        normalized = tuple(sorted((str(name), int(count)) for name, count in self.counts))
        if len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("instruction names must be unique")
        if any(count < 0 for _, count in normalized) or self.other_cycles < 0:
            raise ValueError("instruction counts and other cycles must be non-negative")
        object.__setattr__(self, "counts", normalized)

    def cycles(self, blen: int, mode: str) -> int:
        return self.other_cycles + sum(count * matrix_issue_cycles(opcode, blen, mode) for opcode, count in self.counts)

    def serialization_gap_cycles(self, blen: int) -> int:
        return self.cycles(blen, RTL_SERIALIZED) - self.cycles(
            blen,
            IDEAL_MATRIX_PIPELINE,
        )


@dataclass(frozen=True)
class BottleneckAttribution:
    """Three-view bottleneck classification for one instruction trace."""

    timing_mode: str
    peak_compute_seconds: float
    compute_seconds: float
    ideal_compute_seconds: float
    memory_seconds: float
    physical_bytes: int
    bandwidth_bytes_per_second: float
    cycles: int
    serialization_gap_cycles: int
    timing_calibrated: bool

    @property
    def memory_bound(self) -> bool:
        return self.memory_seconds >= self.compute_seconds

    @property
    def classical_roofline_bottleneck(self) -> str:
        return "memory" if self.memory_seconds >= self.peak_compute_seconds else "compute"

    @property
    def architecture_issue_bottleneck(self) -> str:
        return "memory" if self.memory_seconds >= self.ideal_compute_seconds else "compute"

    @property
    def algorithmic_bottleneck(self) -> str:
        """Compatibility alias for the architecture-issue view."""

        return self.architecture_issue_bottleneck

    @property
    def bottleneck(self) -> str:
        if self.memory_bound:
            return "memory"
        if (
            self.timing_mode == RTL_SERIALIZED
            and self.serialization_gap_cycles > 0
            and self.architecture_issue_bottleneck == "memory"
        ):
            return "serialization"
        return "compute"

    @property
    def rankable(self) -> bool:
        return self.timing_calibrated


def attribute_bottleneck(
    mix: MatrixInstructionMix,
    *,
    blen: int,
    timing_mode: str,
    frequency_hz: float,
    physical_bytes: int,
    bandwidth_bytes_per_second: float,
    peak_compute_seconds: float,
    timing_evidence: TimingEvidence | None,
) -> BottleneckAttribution:
    """Classify from physical bytes and trace cycles without changing labels."""

    if physical_bytes < 0:
        raise ValueError("physical_bytes must be non-negative")
    if not math.isfinite(bandwidth_bytes_per_second) or bandwidth_bytes_per_second <= 0:
        raise ValueError("bandwidth_bytes_per_second must be positive")
    if not math.isfinite(peak_compute_seconds) or peak_compute_seconds < 0:
        raise ValueError("peak_compute_seconds must be finite and non-negative")
    cycles = mix.cycles(blen, timing_mode)
    ideal_cycles = mix.cycles(blen, IDEAL_MATRIX_PIPELINE)
    calibrated, _ = validate_timing_evidence(timing_mode, timing_evidence)
    result = BottleneckAttribution(
        timing_mode=timing_mode,
        peak_compute_seconds=peak_compute_seconds,
        compute_seconds=cycles_to_seconds(
            cycles,
            frequency_hz=frequency_hz,
        ),
        ideal_compute_seconds=cycles_to_seconds(
            ideal_cycles,
            frequency_hz=frequency_hz,
        ),
        memory_seconds=physical_bytes / bandwidth_bytes_per_second,
        physical_bytes=physical_bytes,
        bandwidth_bytes_per_second=bandwidth_bytes_per_second,
        cycles=cycles,
        serialization_gap_cycles=mix.serialization_gap_cycles(blen),
        timing_calibrated=calibrated,
    )
    if result.peak_compute_seconds > result.ideal_compute_seconds + 1e-18:
        raise ValueError("peak compute cannot be slower than ideal issue")
    return result


@dataclass(frozen=True)
class CycleAnchor:
    """One cross-stack cycle comparison for a single anchor program.

    Every anchor carries analytical and emulator cycles. RTL cycles and the
    RTL trace hash are present on RTL-tier anchors and absent on
    emulator-tier anchors; partial RTL evidence is rejected.
    """

    anchor_id: str
    analytical_cycles: int
    emulator_cycles: int
    rtl_cycles: int | None = None
    anchor_kind: str = "layer"
    analytical_compute_cycles: int | None = None
    analytical_memory_cycles: int | None = None
    cache_position: int | None = None
    batch: int | None = None
    physical_hbm_bytes: int | None = None
    mlen: int | None = None
    blen: int | None = None
    hlen: int | None = None
    vlen: int | None = None
    geometry_sha256: str | None = None
    precision_sha256: str | None = None
    compiler_sha256: str | None = None
    asm_sha256: str | None = None
    analytical_trace_sha256: str | None = None
    emulator_trace_sha256: str | None = None
    rtl_trace_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.anchor_id:
            raise ValueError("anchor_id must be non-empty")
        if self.anchor_kind not in REQUIRED_TIMING_ANCHOR_KINDS:
            raise ValueError(f"unsupported timing anchor kind {self.anchor_kind!r}")
        for name in ("analytical_cycles", "emulator_cycles"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rtl_cycles is not None and self.rtl_cycles <= 0:
            raise ValueError("rtl_cycles must be positive")
        layer_fields = (
            self.analytical_compute_cycles,
            self.analytical_memory_cycles,
            self.cache_position,
            self.batch,
            self.physical_hbm_bytes,
        )
        if self.anchor_kind == "layer":
            if any(value is None for value in layer_fields):
                raise ValueError("layer anchors require compute, memory, cache, batch, and HBM-byte evidence")
            for name in (
                "analytical_compute_cycles",
                "analytical_memory_cycles",
                "batch",
                "physical_hbm_bytes",
            ):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be positive")
            if self.cache_position < 0:
                raise ValueError("cache_position must be non-negative")
            if self.analytical_cycles != max(
                self.analytical_compute_cycles,
                self.analytical_memory_cycles,
            ):
                raise ValueError("layer analytical cycles must use max_compute_memory")
        elif any(value is not None for value in layer_fields):
            raise ValueError("operation anchors cannot carry whole-layer timing fields")
        core_identity_fields = (
            self.mlen,
            self.blen,
            self.hlen,
            self.vlen,
            self.geometry_sha256,
            self.precision_sha256,
            self.compiler_sha256,
            self.asm_sha256,
            self.analytical_trace_sha256,
            self.emulator_trace_sha256,
        )
        core_present = any(value is not None for value in core_identity_fields)
        if core_present or self.rtl_trace_sha256 is not None:
            if any(value is None for value in core_identity_fields):
                raise ValueError("timing anchor execution identity must be complete")
            if (self.rtl_trace_sha256 is None) != (self.rtl_cycles is None):
                raise ValueError("timing anchor execution identity must be complete")
            for name in ("mlen", "blen", "hlen", "vlen"):
                if getattr(self, name) <= 0:
                    raise ValueError(f"{name} must be positive")
            if self.mlen % self.blen:
                raise ValueError("timing anchor MLEN must be divisible by BLEN")
            if self.mlen % self.hlen:
                raise ValueError("timing anchor HLEN must divide MLEN")
            digest_names = [
                "geometry_sha256",
                "precision_sha256",
                "compiler_sha256",
                "asm_sha256",
                "analytical_trace_sha256",
                "emulator_trace_sha256",
            ]
            if self.rtl_trace_sha256 is not None:
                digest_names.append("rtl_trace_sha256")
            for name in digest_names:
                if not _is_sha256(getattr(self, name)):
                    raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @property
    def has_rtl_evidence(self) -> bool:
        return self.rtl_cycles is not None

    @property
    def emulator_rtl_error(self) -> float | None:
        if self.rtl_cycles is None:
            return None
        return abs(self.emulator_cycles - self.rtl_cycles) / self.rtl_cycles

    @property
    def analytical_rtl_error(self) -> float | None:
        if self.rtl_cycles is None:
            return None
        return abs(self.analytical_cycles - self.rtl_cycles) / self.rtl_cycles

    @property
    def analytical_emulator_error(self) -> float:
        return abs(self.analytical_cycles - self.emulator_cycles) / self.emulator_cycles

    @property
    def identity_complete(self) -> bool:
        core = all(
            value is not None
            for value in (
                self.mlen,
                self.blen,
                self.hlen,
                self.vlen,
                self.geometry_sha256,
                self.precision_sha256,
                self.compiler_sha256,
                self.asm_sha256,
                self.analytical_trace_sha256,
                self.emulator_trace_sha256,
            )
        )
        if self.has_rtl_evidence:
            return core and self.rtl_trace_sha256 is not None
        return core

    @property
    def geometry(self) -> tuple[int, int, int, int] | None:
        if not self.identity_complete:
            return None
        return self.mlen, self.blen, self.hlen, self.vlen

    @property
    def trace_identity_matched(self) -> bool:
        if not self.identity_complete:
            return False
        hashes = {self.analytical_trace_sha256, self.emulator_trace_sha256}
        if self.has_rtl_evidence:
            hashes.add(self.rtl_trace_sha256)
        return len(hashes) == 1

    @property
    def instruction_trace_sha256(self) -> str | None:
        return self.analytical_trace_sha256 if self.trace_identity_matched else None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CycleAnchor:
        return cls(
            anchor_id=str(value["anchor_id"]),
            analytical_cycles=int(value["analytical_cycles"]),
            emulator_cycles=int(value["emulator_cycles"]),
            rtl_cycles=(int(value["rtl_cycles"]) if value.get("rtl_cycles") not in (None, "") else None),
            anchor_kind=str(value["anchor_kind"]),
            analytical_compute_cycles=(
                int(value["analytical_compute_cycles"])
                if value.get("analytical_compute_cycles") not in (None, "")
                else None
            ),
            analytical_memory_cycles=(
                int(value["analytical_memory_cycles"])
                if value.get("analytical_memory_cycles") not in (None, "")
                else None
            ),
            cache_position=(int(value["cache_position"]) if value.get("cache_position") not in (None, "") else None),
            batch=(int(value["batch"]) if value.get("batch") not in (None, "") else None),
            physical_hbm_bytes=(
                int(value["physical_hbm_bytes"]) if value.get("physical_hbm_bytes") not in (None, "") else None
            ),
            mlen=(int(value["mlen"]) if value.get("mlen") not in (None, "") else None),
            blen=(int(value["blen"]) if value.get("blen") not in (None, "") else None),
            hlen=(int(value["hlen"]) if value.get("hlen") not in (None, "") else None),
            vlen=(int(value["vlen"]) if value.get("vlen") not in (None, "") else None),
            geometry_sha256=(str(value["geometry_sha256"]) if value.get("geometry_sha256") not in (None, "") else None),
            precision_sha256=(
                str(value["precision_sha256"]) if value.get("precision_sha256") not in (None, "") else None
            ),
            compiler_sha256=(str(value["compiler_sha256"]) if value.get("compiler_sha256") not in (None, "") else None),
            asm_sha256=(str(value["asm_sha256"]) if value.get("asm_sha256") not in (None, "") else None),
            analytical_trace_sha256=(
                str(value["analytical_trace_sha256"])
                if value.get("analytical_trace_sha256") not in (None, "")
                else None
            ),
            emulator_trace_sha256=(
                str(value["emulator_trace_sha256"]) if value.get("emulator_trace_sha256") not in (None, "") else None
            ),
            rtl_trace_sha256=(
                str(value["rtl_trace_sha256"]) if value.get("rtl_trace_sha256") not in (None, "") else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "anchor_id": self.anchor_id,
            "anchor_kind": self.anchor_kind,
            "analytical_cycles": self.analytical_cycles,
            "emulator_cycles": self.emulator_cycles,
            "rtl_cycles": self.rtl_cycles,
            "analytical_compute_cycles": self.analytical_compute_cycles,
            "analytical_memory_cycles": self.analytical_memory_cycles,
            "cache_position": self.cache_position,
            "batch": self.batch,
            "physical_hbm_bytes": self.physical_hbm_bytes,
            "mlen": self.mlen,
            "blen": self.blen,
            "hlen": self.hlen,
            "vlen": self.vlen,
            "geometry_sha256": self.geometry_sha256,
            "precision_sha256": self.precision_sha256,
            "compiler_sha256": self.compiler_sha256,
            "asm_sha256": self.asm_sha256,
            "analytical_trace_sha256": self.analytical_trace_sha256,
            "emulator_trace_sha256": self.emulator_trace_sha256,
            "rtl_trace_sha256": self.rtl_trace_sha256,
            "identity_complete": self.identity_complete,
            "trace_identity_matched": self.trace_identity_matched,
            "instruction_trace_sha256": self.instruction_trace_sha256,
            "emulator_rtl_error": self.emulator_rtl_error,
            "analytical_rtl_error": self.analytical_rtl_error,
            "analytical_emulator_error": self.analytical_emulator_error,
        }


@dataclass(frozen=True)
class TimingEvidence:
    """Fail-closed timing calibration for one evidence mode.

    RTL-tier modes anchor the analytic model to RTL cycle counts with an
    emulator cross-check. The ``emulator_serialized`` mode anchors the same
    serialized issue contract to emulator cycle counts only; its anchors
    carry no RTL evidence and its gates compare analytic against emulator at
    the same numeric limits.
    """

    mode: str
    anchors: tuple[CycleAnchor, ...]
    provenance_hashes: tuple[tuple[str, str], ...]
    step_composition: str = STEP_COMPOSITION
    anchor_max_error_limit: float = CANONICAL_ANCHOR_MAX_ERROR_LIMIT
    analytical_mape_limit: float = CANONICAL_ANALYTICAL_MAPE_LIMIT

    def __post_init__(self) -> None:
        if self.mode not in TIMING_EVIDENCE_MODES:
            raise ValueError(f"unknown timing mode {self.mode!r}")
        if self.step_composition != STEP_COMPOSITION:
            raise ValueError(f"timing evidence requires {STEP_COMPOSITION!r} composition")
        if (
            self.anchor_max_error_limit != CANONICAL_ANCHOR_MAX_ERROR_LIMIT
            or self.analytical_mape_limit != CANONICAL_ANALYTICAL_MAPE_LIMIT
        ):
            raise ValueError("timing evidence limits are immutable at 5% per-anchor error and 10% analytical MAPE")
        if not self.anchors:
            raise ValueError("timing evidence requires at least one anchor")
        if len({anchor.anchor_id for anchor in self.anchors}) != len(self.anchors):
            raise ValueError("timing anchor IDs must be unique")
        if self.evidence_tier == EMULATOR_EVIDENCE_TIER:
            if any(anchor.has_rtl_evidence or anchor.rtl_trace_sha256 is not None for anchor in self.anchors):
                raise ValueError("emulator-tier timing anchors must not carry RTL evidence")
        elif any(not anchor.has_rtl_evidence for anchor in self.anchors):
            raise ValueError("RTL-tier timing anchors require RTL cycle evidence")
        layer_anchors = tuple(anchor for anchor in self.anchors if anchor.anchor_kind == "layer")
        if layer_anchors:
            batches = {anchor.batch for anchor in layer_anchors}
            positions = sorted(anchor.cache_position for anchor in layer_anchors)
            if len(batches) != 1:
                raise ValueError("layer timing anchors must use one batch")
            if any(right != left + 1 for left, right in pairwise(positions)):
                raise ValueError("layer timing anchors must cover consecutive cache appends")
        if not self.provenance_hashes:
            raise ValueError("timing evidence requires provenance hashes")
        if len({name for name, _ in self.provenance_hashes}) != len(self.provenance_hashes):
            raise ValueError("timing provenance roles must be unique")
        for name, digest in self.provenance_hashes:
            if not name or not _is_sha256(digest):
                raise ValueError("provenance hashes must be named SHA-256 digests")
        if self.evidence_tier == EMULATOR_EVIDENCE_TIER and any(
            name == "rtl" for name, _ in self.provenance_hashes
        ):
            raise ValueError("emulator-tier timing evidence must not carry an RTL provenance role")

    @property
    def evidence_tier(self) -> str:
        return EMULATOR_EVIDENCE_TIER if self.mode == EMULATOR_SERIALIZED else RTL_EVIDENCE_TIER

    @property
    def required_provenance_roles(self) -> tuple[str, ...]:
        if self.evidence_tier == EMULATOR_EVIDENCE_TIER:
            return EMULATOR_TIMING_PROVENANCE_ROLES
        return REQUIRED_TIMING_PROVENANCE_ROLES

    @property
    def emulator_rtl_error(self) -> float | None:
        errors = [anchor.emulator_rtl_error for anchor in self.anchors]
        if any(error is None for error in errors):
            return None
        return max(errors)

    @property
    def anchor_max_error(self) -> float:
        """Per-anchor tightness: emulator-vs-RTL at RTL tier, analytic-vs-emulator otherwise."""

        if self.evidence_tier == EMULATOR_EVIDENCE_TIER:
            return max(anchor.analytical_emulator_error for anchor in self.anchors)
        return max(anchor.emulator_rtl_error for anchor in self.anchors)

    @property
    def analytical_mape(self) -> float:
        if self.evidence_tier == EMULATOR_EVIDENCE_TIER:
            errors = [anchor.analytical_emulator_error for anchor in self.anchors]
        else:
            errors = [anchor.analytical_rtl_error for anchor in self.anchors]
        return sum(errors) / len(errors)

    @property
    def missing_anchor_kinds(self) -> tuple[str, ...]:
        observed = {anchor.anchor_kind for anchor in self.anchors}
        return tuple(kind for kind in REQUIRED_TIMING_ANCHOR_KINDS if kind not in observed)

    @property
    def layer_anchor_count(self) -> int:
        return sum(anchor.anchor_kind == "layer" for anchor in self.anchors)

    @property
    def missing_provenance_roles(self) -> tuple[str, ...]:
        observed = {name for name, _ in self.provenance_hashes}
        return tuple(role for role in self.required_provenance_roles if role not in observed)

    @property
    def execution_identity_complete(self) -> bool:
        return all(anchor.identity_complete for anchor in self.anchors)

    @property
    def shared_geometry(self) -> bool:
        return (
            self.execution_identity_complete
            and len({(anchor.geometry, anchor.geometry_sha256) for anchor in self.anchors}) == 1
        )

    @property
    def shared_precision(self) -> bool:
        return self.execution_identity_complete and len({anchor.precision_sha256 for anchor in self.anchors}) == 1

    @property
    def shared_compiler(self) -> bool:
        return self.execution_identity_complete and len({anchor.compiler_sha256 for anchor in self.anchors}) == 1

    @property
    def compiler_provenance_matched(self) -> bool:
        compiler_digest = dict(self.provenance_hashes).get("compiler")
        return (
            self.shared_compiler
            and compiler_digest is not None
            and all(anchor.compiler_sha256 == compiler_digest for anchor in self.anchors)
        )

    @property
    def trace_identities_matched(self) -> bool:
        return all(anchor.trace_identity_matched for anchor in self.anchors)

    @property
    def execution_identity_matched(self) -> bool:
        return (
            self.execution_identity_complete
            and self.shared_geometry
            and self.shared_precision
            and self.shared_compiler
            and self.compiler_provenance_matched
            and self.trace_identities_matched
        )

    @property
    def passed(self) -> bool:
        return (
            not self.missing_anchor_kinds
            and self.layer_anchor_count >= 2
            and not self.missing_provenance_roles
            and self.execution_identity_matched
            and self.anchor_max_error <= self.anchor_max_error_limit
            and self.analytical_mape <= self.analytical_mape_limit
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": TIMING_EVIDENCE_SCHEMA,
            "mode": self.mode,
            "evidence_tier": self.evidence_tier,
            "step_composition": self.step_composition,
            "anchor_max_error_limit": self.anchor_max_error_limit,
            "analytical_mape_limit": self.analytical_mape_limit,
            "provenance_hashes": dict(self.provenance_hashes),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
        }

    @property
    def evidence_id(self) -> str:
        payload = json.dumps(
            self._content_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"timing-{hashlib.sha256(payload).hexdigest()}"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TimingEvidence:
        if value.get("schema") != TIMING_EVIDENCE_SCHEMA:
            raise ValueError("unsupported timing evidence schema")
        anchors = tuple(CycleAnchor.from_dict(anchor) for anchor in value.get("anchors", ()))
        provenance = value.get("provenance_hashes", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance_hashes must be an object")
        evidence = cls(
            mode=str(value["mode"]),
            anchors=anchors,
            provenance_hashes=tuple(sorted((str(name), str(digest)) for name, digest in provenance.items())),
            step_composition=str(value["step_composition"]),
            anchor_max_error_limit=float(
                value.get(
                    "anchor_max_error_limit",
                    CANONICAL_ANCHOR_MAX_ERROR_LIMIT,
                )
            ),
            analytical_mape_limit=float(
                value.get(
                    "analytical_mape_limit",
                    CANONICAL_ANALYTICAL_MAPE_LIMIT,
                )
            ),
        )
        declared_tier = value.get("evidence_tier")
        if declared_tier is not None and declared_tier != evidence.evidence_tier:
            raise ValueError("timing evidence tier does not match its mode")
        observed_id = value.get("evidence_id")
        if observed_id is not None and observed_id != evidence.evidence_id:
            raise ValueError("timing evidence identity mismatch")
        return evidence

    @classmethod
    def load(cls, path: str | Path) -> TimingEvidence:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, object]:
        return self._content_dict() | {
            "evidence_id": self.evidence_id,
            "mode": self.mode,
            "evidence_tier": self.evidence_tier,
            "passed": self.passed,
            "missing_anchor_kinds": list(self.missing_anchor_kinds),
            "layer_anchor_count": self.layer_anchor_count,
            "missing_provenance_roles": list(self.missing_provenance_roles),
            "execution_identity_complete": self.execution_identity_complete,
            "shared_geometry": self.shared_geometry,
            "shared_precision": self.shared_precision,
            "shared_compiler": self.shared_compiler,
            "compiler_provenance_matched": self.compiler_provenance_matched,
            "trace_identities_matched": self.trace_identities_matched,
            "execution_identity_matched": self.execution_identity_matched,
            "emulator_rtl_error": self.emulator_rtl_error,
            "anchor_max_error": self.anchor_max_error,
            "analytical_mape": self.analytical_mape,
        }


def validate_timing_evidence(
    mode: str,
    evidence: TimingEvidence | None,
) -> tuple[bool, str]:
    """Require matching, passing evidence before a mode ranks hardware.

    Emulator-tier evidence calibrates the serialized issue contract, so it
    satisfies the ``rtl_serialized`` pricing mode with a reason that names
    its tier.
    """

    if evidence is None:
        return False, "missing_timing_evidence"
    if evidence.mode != mode and not (
        mode == RTL_SERIALIZED and evidence.mode == EMULATOR_SERIALIZED
    ):
        return False, "timing_mode_mismatch"
    if not evidence.passed:
        return False, "timing_calibration_failed"
    if evidence.evidence_tier == EMULATOR_EVIDENCE_TIER:
        return True, "timing_calibrated_emulator_tier"
    return True, "timing_calibrated"


def cycle_error_summary(anchors: Sequence[CycleAnchor]) -> dict[str, float]:
    """Return the two publication timing errors for an RTL-tier anchor set."""

    anchors = tuple(anchors)
    if not anchors:
        raise ValueError("cycle error summary requires at least one anchor")
    if any(not anchor.has_rtl_evidence for anchor in anchors):
        raise ValueError("cycle error summary requires RTL cycle evidence")
    return {
        "emulator_rtl_max_error": max(anchor.emulator_rtl_error for anchor in anchors),
        "analytical_mape": sum(anchor.analytical_rtl_error for anchor in anchors) / len(anchors),
    }


__all__ = [
    "CANONICAL_ANALYTICAL_MAPE_LIMIT",
    "CANONICAL_ANCHOR_MAX_ERROR_LIMIT",
    "EMULATOR_EVIDENCE_TIER",
    "EMULATOR_SERIALIZED",
    "EMULATOR_TIMING_PROVENANCE_ROLES",
    "IDEAL_MATRIX_PIPELINE",
    "REQUIRED_TIMING_ANCHOR_KINDS",
    "REQUIRED_TIMING_PROVENANCE_ROLES",
    "RTL_EVIDENCE_TIER",
    "RTL_SERIALIZED",
    "STEP_COMPOSITION",
    "TIMING_EVIDENCE_MODES",
    "TIMING_EVIDENCE_SCHEMA",
    "TIMING_MODES",
    "BottleneckAttribution",
    "CycleAnchor",
    "MatrixInstructionMix",
    "TimingEvidence",
    "attribute_bottleneck",
    "cycle_error_summary",
    "cycles_to_seconds",
    "matrix_issue_cycles",
    "projection_mm_events",
    "validate_timing_evidence",
]
