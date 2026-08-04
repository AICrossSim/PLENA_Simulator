"""Analytic-vs-emulator calibration for the decode layer.

This is deliberately **not** trace calibration. `decode_timing.TimingEvidence`
compares both the analytic model and the emulator against RTL cycles and is the
only thing that may promote a bound to `trace-calibrated`. The artifact here
compares the analytic model against the transactional emulator, whose numerical
output is checked against a PyTorch golden but whose cycle counts have no RTL
reference. It therefore carries its own label, `emulator-calibrated`, and it
records `uncovered_fraction`: the share of measured decode-layer cycles that no
analytic term describes. A reader who sees the label cannot mistake it for full
coverage, and a reviewer can see exactly how much of the layer the agreement
covers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

EMULATOR_CALIBRATION_SCHEMA = "plena-decode-emulator-calibration"
EMULATOR_CALIBRATION_LABEL = "emulator-calibrated"
EMULATOR_CALIBRATION_STAGES = (
    "Activation load + RMSNorm",
    "Q/K/V + W_O projection + RoPE",
    "KV store",
    "Flash attention",
    "Residual add",
    "FFN (gate/up/down)",
    "LM head",
)
EMULATOR_STAGE_ERROR_LIMIT = 0.05
EMULATOR_TOTAL_ERROR_LIMIT = 0.15
EMULATOR_UNCOVERED_FRACTION_LIMIT = 0.01
EMULATOR_MEASURED_PROVENANCE_ROLES = (
    "op_stats",
    "assembly",
    "isa_lib",
    "settings",
    "run_manifest",
    "run_receipt",
    "emulator_binary",
    "compiler_artifact",
    "request_memory_calibration",
)
EMULATOR_PRECISION_ROLES = (
    "MATRIX_SRAM_TYPE",
    "VECTOR_SRAM_TYPE",
    "HBM_M_WEIGHT_TYPE",
    "HBM_M_KV_TYPE",
    "HBM_V_ACT_TYPE",
    "HBM_V_KV_TYPE",
)
CALIBRATION_SOURCE_FILES = (
    ("performance/perf_model.py", "perf_model.py"),
    ("performance/decode_cost_model.py", "decode_cost_model.py"),
    ("performance/decode_timing.py", "decode_timing.py"),
    ("performance/disagg_decode.py", "disagg_decode.py"),
    ("performance/compiler_trace_timing.py", "compiler_trace_timing.py"),
    ("disagg_serve/memory.py", "../disagg_serve/memory.py"),
    ("disagg_serve/packed_kv.py", "../disagg_serve/packed_kv.py"),
    ("performance/decode_stage_validation.py", "decode_stage_validation.py"),
    ("performance/emulator_calibration.py", "emulator_calibration.py"),
    (
        "compiler/aten/execution_trace.py",
        "../../compiler/aten/execution_trace.py",
    ),
)
EMULATOR_ANALYTIC_PROVENANCE_ROLES = tuple(
    f"analytic_source:{name}" for name, _relative_path in CALIBRATION_SOURCE_FILES
)
EMULATOR_REQUIRED_PROVENANCE_ROLES = (
    EMULATOR_MEASURED_PROVENANCE_ROLES + EMULATOR_ANALYTIC_PROVENANCE_ROLES
)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON object key: {name}")
        value[name] = item
    return value


@dataclass(frozen=True)
class StageCalibration:
    """One decode-layer stage compared against the emulator."""

    stage: str
    analytical_cycles: int
    emulator_cycles: int

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("stage must be non-empty")
        if self.analytical_cycles <= 0 or self.emulator_cycles <= 0:
            raise ValueError(f"{self.stage}: cycle counts must be positive")

    @property
    def error(self) -> float:
        return (self.analytical_cycles - self.emulator_cycles) / self.emulator_cycles

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "analytical_cycles": self.analytical_cycles,
            "emulator_cycles": self.emulator_cycles,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "StageCalibration":
        return cls(
            stage=str(value["stage"]),
            analytical_cycles=int(value["analytical_cycles"]),
            emulator_cycles=int(value["emulator_cycles"]),
        )


@dataclass(frozen=True)
class EmulatorExecutionContract:
    """Runtime choices that must match the analytic timing comparison."""

    timing_mode: str
    drain_overlapped: bool
    fp_sram_depth: int
    hbm_gen: str
    hbm_channels: int
    precision: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.timing_mode not in {"rtl_serialized", "drain_overlapped"}:
            raise ValueError("emulator calibration has no matching timing mode")
        if self.drain_overlapped != (self.timing_mode == "drain_overlapped"):
            raise ValueError("timing mode disagrees with emulator drain behavior")
        if self.fp_sram_depth <= 0:
            raise ValueError("emulator FP SRAM depth must be positive")
        if self.hbm_gen not in {"HBM2", "HBM3"} or self.hbm_channels <= 0:
            raise ValueError("emulator HBM geometry is invalid")
        names = tuple(name for name, _descriptor in self.precision)
        if names != EMULATOR_PRECISION_ROLES:
            raise ValueError("emulator precision contract is incomplete")
        for _name, descriptor in self.precision:
            try:
                decoded = json.loads(descriptor)
            except json.JSONDecodeError as error:
                raise ValueError("emulator precision descriptor is invalid") from error
            if not isinstance(decoded, Mapping) or descriptor != json.dumps(
                decoded, sort_keys=True, separators=(",", ":"), allow_nan=False
            ):
                raise ValueError("emulator precision descriptor is not canonical")

    def to_dict(self) -> dict[str, object]:
        return {
            "timing_mode": self.timing_mode,
            "drain_overlapped": self.drain_overlapped,
            "fp_sram_depth": self.fp_sram_depth,
            "hbm_gen": self.hbm_gen,
            "hbm_channels": self.hbm_channels,
            "precision": {
                name: json.loads(descriptor) for name, descriptor in self.precision
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EmulatorExecutionContract":
        precision = value.get("precision")
        if not isinstance(precision, Mapping):
            raise ValueError("emulator precision contract is missing")
        if set(precision) != set(EMULATOR_PRECISION_ROLES):
            raise ValueError("emulator precision contract is incomplete")
        drain_overlapped = value.get("drain_overlapped")
        if not isinstance(drain_overlapped, bool):
            raise ValueError("emulator drain behavior must be Boolean")
        return cls(
            timing_mode=str(value["timing_mode"]),
            drain_overlapped=drain_overlapped,
            fp_sram_depth=int(value["fp_sram_depth"]),
            hbm_gen=str(value["hbm_gen"]).upper(),
            hbm_channels=int(value["hbm_channels"]),
            precision=tuple(
                (
                    name,
                    json.dumps(
                        precision[name],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                )
                for name in EMULATOR_PRECISION_ROLES
            ),
        )


@dataclass(frozen=True)
class EmulatorCalibration:
    """Fail-closed analytic-vs-emulator agreement for one decode configuration."""

    configuration: str
    stages: tuple[StageCalibration, ...]
    uncovered_cycles: int
    provenance_hashes: tuple[tuple[str, str], ...]
    execution_contract: EmulatorExecutionContract
    stage_error_limit: float = EMULATOR_STAGE_ERROR_LIMIT
    total_error_limit: float = EMULATOR_TOTAL_ERROR_LIMIT
    uncovered_fraction_limit: float = EMULATOR_UNCOVERED_FRACTION_LIMIT

    def __post_init__(self) -> None:
        if not self.configuration:
            raise ValueError("configuration must be non-empty")
        stage_names = tuple(stage.stage for stage in self.stages)
        if stage_names != EMULATOR_CALIBRATION_STAGES:
            raise ValueError(
                "calibration stages must exactly match the canonical decode stages"
            )
        if self.uncovered_cycles < 0:
            raise ValueError("uncovered cycles must be non-negative")
        provenance_names = tuple(name for name, _digest in self.provenance_hashes)
        if len(set(provenance_names)) != len(provenance_names):
            raise ValueError("calibration provenance roles must be unique")
        missing_roles = sorted(
            set(EMULATOR_REQUIRED_PROVENANCE_ROLES) - set(provenance_names)
        )
        if missing_roles:
            raise ValueError(
                "calibration is missing required provenance roles: "
                + ", ".join(missing_roles)
            )
        for name, digest in self.provenance_hashes:
            if (
                not name
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("provenance hashes must be named SHA-256 digests")
        limits = (
            self.stage_error_limit,
            self.total_error_limit,
            self.uncovered_fraction_limit,
        )
        canonical_limits = (
            EMULATOR_STAGE_ERROR_LIMIT,
            EMULATOR_TOTAL_ERROR_LIMIT,
            EMULATOR_UNCOVERED_FRACTION_LIMIT,
        )
        if any(not math.isfinite(limit) for limit in limits) or limits != canonical_limits:
            raise ValueError(
                "calibration limits must exactly match the canonical validation limits"
            )

    @property
    def analytical_cycles(self) -> int:
        return sum(stage.analytical_cycles for stage in self.stages)

    @property
    def emulator_cycles(self) -> int:
        return sum(stage.emulator_cycles for stage in self.stages)

    @property
    def measured_layer_cycles(self) -> int:
        return self.emulator_cycles + self.uncovered_cycles

    @property
    def total_error(self) -> float:
        return (self.analytical_cycles - self.emulator_cycles) / self.emulator_cycles

    @property
    def worst_stage_error(self) -> float:
        return max(abs(stage.error) for stage in self.stages)

    @property
    def uncovered_fraction(self) -> float:
        """Share of measured decode-layer cycles no analytic term describes."""
        return self.uncovered_cycles / self.measured_layer_cycles

    @property
    def passed(self) -> bool:
        return (
            self.worst_stage_error <= self.stage_error_limit
            and abs(self.total_error) <= self.total_error_limit
            and self.uncovered_fraction <= self.uncovered_fraction_limit
        )

    @property
    def label(self) -> str:
        """The only label this artifact may carry. Never `trace-calibrated`."""
        return EMULATOR_CALIBRATION_LABEL if self.passed else "uncalibrated"

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema": EMULATOR_CALIBRATION_SCHEMA,
            "configuration": self.configuration,
            "reference": "transactional_emulator",
            "stage_error_limit": self.stage_error_limit,
            "total_error_limit": self.total_error_limit,
            "uncovered_fraction_limit": self.uncovered_fraction_limit,
            "uncovered_cycles": self.uncovered_cycles,
            "execution_contract": self.execution_contract.to_dict(),
            "provenance_hashes": dict(self.provenance_hashes),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @property
    def calibration_id(self) -> str:
        payload = json.dumps(
            self._content_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"emucal-{hashlib.sha256(payload).hexdigest()}"

    def to_dict(self) -> dict[str, object]:
        return self._content_dict() | {
            "calibration_id": self.calibration_id,
            "label": self.label,
            "passed": self.passed,
            "total_error": self.total_error,
            "worst_stage_error": self.worst_stage_error,
            "uncovered_fraction": self.uncovered_fraction,
            "measured_layer_cycles": self.measured_layer_cycles,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "EmulatorCalibration":
        if value.get("schema") != EMULATOR_CALIBRATION_SCHEMA:
            raise ValueError("unsupported emulator calibration schema")
        required_limits = {
            "stage_error_limit": EMULATOR_STAGE_ERROR_LIMIT,
            "total_error_limit": EMULATOR_TOTAL_ERROR_LIMIT,
            "uncovered_fraction_limit": EMULATOR_UNCOVERED_FRACTION_LIMIT,
        }
        for name, expected in required_limits.items():
            if name not in value or float(value[name]) != expected:
                raise ValueError(
                    "calibration limits must exactly match the canonical validation limits"
                )
        provenance = value.get("provenance_hashes", {})
        if not isinstance(provenance, Mapping):
            raise ValueError("provenance_hashes must be an object")
        calibration = cls(
            configuration=str(value["configuration"]),
            stages=tuple(
                StageCalibration.from_dict(stage) for stage in value.get("stages", ())
            ),
            uncovered_cycles=int(value["uncovered_cycles"]),
            execution_contract=EmulatorExecutionContract.from_dict(
                value["execution_contract"]
            ),
            provenance_hashes=tuple(
                sorted((str(name), str(digest)) for name, digest in provenance.items())
            ),
            stage_error_limit=float(value["stage_error_limit"]),
            total_error_limit=float(value["total_error_limit"]),
            uncovered_fraction_limit=float(value["uncovered_fraction_limit"]),
        )
        observed = value.get("calibration_id")
        if observed is not None and observed != calibration.calibration_id:
            raise ValueError("emulator calibration identity mismatch")
        return calibration

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        source_directory: str | Path | None = None,
    ) -> "EmulatorCalibration":
        """Load an artifact only when its analytical source hashes are current."""
        calibration = cls.from_dict(
            json.loads(Path(path).read_text(), object_pairs_hook=_unique_json_object)
        )
        validate_calibration_sources(calibration, source_directory)
        return calibration


def sha256_file(path: str | Path) -> str:
    """Digest an external input so a result records which input produced it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_source_hashes(
    source_directory: str | Path | None = None,
) -> dict[str, str]:
    """Hash every analytical source that gives a calibration its meaning."""
    directory = (
        Path(source_directory)
        if source_directory is not None
        else Path(__file__).resolve().parent
    )
    return {
        f"analytic_source:{name}": sha256_file(directory / relative_path)
        for name, relative_path in CALIBRATION_SOURCE_FILES
    }


def validate_calibration_sources(
    calibration: EmulatorCalibration,
    source_directory: str | Path | None = None,
) -> None:
    """Reject calibration produced by missing or different analytical code."""
    expected = calibration_source_hashes(source_directory)
    observed = dict(calibration.provenance_hashes)
    missing = sorted(name for name in expected if name not in observed)
    if missing:
        raise ValueError(
            "missing analytic source provenance: " + ", ".join(missing)
        )
    stale = sorted(
        name for name, digest in expected.items() if observed[name] != digest
    )
    if stale:
        raise ValueError("stale analytic source provenance: " + ", ".join(stale))


def describe_emulator_calibration(
    calibration: EmulatorCalibration | None,
) -> tuple[bool, str]:
    """Report the emulator-calibration state without granting RTL calibration.

    The boolean is never a licence to rank hardware: only
    `decode_timing.validate_timing_evidence` can do that, and it requires RTL
    cycles. This reports whether the analytic model agrees with the emulator
    over the stages it models.
    """
    if calibration is None:
        return False, "missing_emulator_calibration"
    if not calibration.passed:
        return False, "emulator_calibration_failed"
    return True, EMULATOR_CALIBRATION_LABEL


__all__ = [
    "EMULATOR_CALIBRATION_LABEL",
    "EMULATOR_CALIBRATION_SCHEMA",
    "EMULATOR_CALIBRATION_STAGES",
    "EMULATOR_MEASURED_PROVENANCE_ROLES",
    "EMULATOR_ANALYTIC_PROVENANCE_ROLES",
    "EMULATOR_REQUIRED_PROVENANCE_ROLES",
    "EMULATOR_PRECISION_ROLES",
    "EMULATOR_STAGE_ERROR_LIMIT",
    "EMULATOR_TOTAL_ERROR_LIMIT",
    "EMULATOR_UNCOVERED_FRACTION_LIMIT",
    "CALIBRATION_SOURCE_FILES",
    "EmulatorCalibration",
    "EmulatorExecutionContract",
    "StageCalibration",
    "calibration_source_hashes",
    "describe_emulator_calibration",
    "sha256_file",
    "validate_calibration_sources",
]
