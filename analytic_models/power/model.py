"""Event-based decode-chip energy model with explicit calibration gates."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .structural_area import StructuralAreaEvidence

POWER_MODEL_VERSION = "plena-event-power"
POWER_CALIBRATION_MANIFEST_HASH = (
    "baad3cf6e7648069f2121b06475eda40000cd6c9a069c6534c3137a00b0eb241"
)
POWER_INTERPOLATION_DOMAIN = {
    "mlen_min": 16,
    "mlen_max": 64,
    "blen_min": 4,
    "blen_max": 16,
}
EXPECTED_FIT_SUMMARY = {
    "complete_rows": 502,
    "train_rows": 332,
    "holdout_rows": 170,
    "signature_count": 80,
    "structural_area_signature_count": 72,
    "vector_area_signature_count": 7,
    "coverage_failures": 0,
}


@dataclass(frozen=True)
class CalibrationGate:
    area_median_pct: float = 10.0
    area_max_pct: float = 15.0
    dynamic_median_pct: float = 15.0
    dynamic_max_pct: float = 25.0
    leakage_median_pct: float = 15.0
    leakage_max_pct: float = 25.0
    rank_correlation: float = 0.90
    cycle_error_pct: float = 5.0
    latency_mape_pct: float = 10.0


@dataclass(frozen=True)
class EventCount:
    """One measured operation signature and its executed event count."""

    signature: str
    count: int
    mlen: int
    blen: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("event count must be non-negative")
        if self.mlen <= 0 or self.blen <= 0 or self.mlen % self.blen:
            raise ValueError("event geometry must satisfy MLEN>0, BLEN>0, MLEN%BLEN==0")


@dataclass(frozen=True)
class ValidationReport:
    area_median_pct: float | None = None
    area_max_pct: float | None = None
    dynamic_median_pct: float | None = None
    dynamic_max_pct: float | None = None
    leakage_median_pct: float | None = None
    leakage_max_pct: float | None = None
    rank_correlation: float | None = None
    cycle_error_pct: float | None = None
    latency_mape_pct: float | None = None
    missing_fields: tuple[str, ...] = ()
    passed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "missing_fields",
            tuple(sorted({str(value) for value in self.missing_fields})),
        )
        metric_names = (
            "area_median_pct",
            "area_max_pct",
            "dynamic_median_pct",
            "dynamic_max_pct",
            "leakage_median_pct",
            "leakage_max_pct",
            "rank_correlation",
            "cycle_error_pct",
            "latency_mape_pct",
        )
        for name in metric_names:
            value = getattr(self, name)
            if value is None:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} must be finite")
            if name == "rank_correlation":
                if not -1.0 <= numeric <= 1.0:
                    raise ValueError("rank_correlation must be in [-1, 1]")
            elif numeric < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, numeric)
        if self.passed and not self.meets_publication_gate:
            raise ValueError("passed validation does not satisfy publication gates")

    @property
    def meets_publication_gate(self) -> bool:
        values = (
            self.area_median_pct,
            self.area_max_pct,
            self.dynamic_median_pct,
            self.dynamic_max_pct,
            self.leakage_median_pct,
            self.leakage_max_pct,
            self.rank_correlation,
            self.cycle_error_pct,
            self.latency_mape_pct,
        )
        if self.missing_fields or any(value is None for value in values):
            return False
        gate = CalibrationGate()
        return (
            self.area_median_pct <= gate.area_median_pct
            and self.area_max_pct <= gate.area_max_pct
            and self.dynamic_median_pct <= gate.dynamic_median_pct
            and self.dynamic_max_pct <= gate.dynamic_max_pct
            and self.leakage_median_pct <= gate.leakage_median_pct
            and self.leakage_max_pct <= gate.leakage_max_pct
            and self.rank_correlation >= gate.rank_correlation
            and self.cycle_error_pct <= gate.cycle_error_pct
            and self.latency_mape_pct <= gate.latency_mape_pct
        )

    def to_dict(self) -> dict:
        return {
            "area_median_pct": self.area_median_pct,
            "area_max_pct": self.area_max_pct,
            "dynamic_median_pct": self.dynamic_median_pct,
            "dynamic_max_pct": self.dynamic_max_pct,
            "leakage_median_pct": self.leakage_median_pct,
            "leakage_max_pct": self.leakage_max_pct,
            "rank_correlation": self.rank_correlation,
            "cycle_error_pct": self.cycle_error_pct,
            "latency_mape_pct": self.latency_mape_pct,
            "missing_fields": list(self.missing_fields),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class PowerCalibration:
    """Versioned coefficients fitted from DC/SAIF and HBM measurements."""

    model_version: str
    event_energy_models: Mapping[str, tuple[float, float, float]]
    hbm_energy_j_per_byte: float
    leakage_power_model: tuple[float, float, float]
    fixed_area_mm2: float
    vector_area_models: Mapping[str, tuple[float, float, float]]
    selector_area_model: tuple[float, float, float]
    structural_area_model: StructuralAreaEvidence
    provenance_hash: str
    activity_provenance_hash: str
    artifact_catalog_sha256: str
    validation: ValidationReport
    calibration_manifest_hash: str = ""
    hbm_energy_source: str = ""
    synthesis_context: Mapping[str, str] = field(default_factory=dict)
    fit_summary: Mapping[str, int] = field(default_factory=dict)
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.model_version or not self.provenance_hash:
            raise ValueError("calibration version and provenance hash are required")
        if self.model_version != POWER_MODEL_VERSION:
            raise ValueError(f"unsupported power model version {self.model_version!r}")
        for name, value in (
            ("provenance_hash", self.provenance_hash),
            ("activity_provenance_hash", self.activity_provenance_hash),
            ("artifact_catalog_sha256", self.artifact_catalog_sha256),
            ("calibration_manifest_hash", self.calibration_manifest_hash),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.calibration_manifest_hash != POWER_CALIBRATION_MANIFEST_HASH:
            raise ValueError("calibration manifest identity mismatch")
        if not self.hbm_energy_source:
            raise ValueError("hbm_energy_source is required")
        context = {
            str(key): str(value)
            for key, value in dict(self.synthesis_context).items()
        }
        for name in (
            "dc_tool_version",
            "library_id",
            "process_corner",
            "mx_block_size",
            "hardware_fp_binding",
            "activity_generator",
        ):
            if not context.get(name):
                raise ValueError(f"synthesis_context.{name} is required")
        if context.get("mx_block_size") != "8":
            raise ValueError("synthesis_context.mx_block_size must be 8")
        object.__setattr__(self, "synthesis_context", context)
        summary = {
            str(key): int(value)
            for key, value in dict(self.fit_summary).items()
        }
        if any(
            summary.get(name) != expected
            for name, expected in EXPECTED_FIT_SUMMARY.items()
        ):
            raise ValueError("fit summary does not cover the calibration schedule")
        object.__setattr__(self, "fit_summary", summary)
        if self.source_sha256 is not None:
            if len(self.source_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in self.source_sha256
            ):
                raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        for name in (
            "hbm_energy_j_per_byte",
            "fixed_area_mm2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        leakage_values = tuple(float(value) for value in self.leakage_power_model)
        if (
            len(leakage_values) != 3
            or any(
                not math.isfinite(value) or value < 0
                for value in leakage_values
            )
            or not any(value > 0 for value in leakage_values)
        ):
            raise ValueError("invalid complete-chip leakage model")
        object.__setattr__(self, "leakage_power_model", leakage_values)
        selector_values = tuple(float(value) for value in self.selector_area_model)
        if (
            len(selector_values) != 3
            or any(
                not math.isfinite(value) or value < 0
                for value in selector_values
            )
            or not any(value > 0 for value in selector_values)
        ):
            raise ValueError("invalid selector area model")
        object.__setattr__(self, "selector_area_model", selector_values)
        if not isinstance(self.structural_area_model, StructuralAreaEvidence):
            raise TypeError("structural_area_model must be StructuralAreaEvidence")
        if not self.structural_area_model.passed:
            raise ValueError("structural area evidence does not pass")
        for name, models in (
            ("event_energy_models", self.event_energy_models),
            ("vector_area_models", self.vector_area_models),
        ):
            for signature, coefficients in models.items():
                values = tuple(float(value) for value in coefficients)
                if (
                    len(values) != 3
                    or any(not math.isfinite(value) or value < 0 for value in values)
                    or not any(value > 0 for value in values)
                ):
                    raise ValueError(f"invalid {name} coefficients for {signature!r}")
        from .calibration_manifest import VECTOR_FP, event_signatures

        if set(self.event_energy_models) != set(event_signatures()):
            raise ValueError("event-energy models do not cover the exact schedule")
        if set(self.vector_area_models) != set(VECTOR_FP):
            raise ValueError("vector-area models do not cover the exact schedule")

    @classmethod
    def from_dict(cls, raw: Mapping) -> "PowerCalibration":
        models = {
            str(k): tuple(float(x) for x in v)
            for k, v in dict(raw.get("event_energy_models", {})).items()
        }
        if any(len(v) != 3 for v in models.values()):
            raise ValueError("each event-energy model needs [constant, MLEN*BLEN, MLEN+BLEN]")
        vector_area_models = {
            str(k): tuple(float(x) for x in v)
            for k, v in dict(raw.get("vector_area_models", {})).items()
        }
        if any(len(v) != 3 for v in vector_area_models.values()):
            raise ValueError(
                "each vector-area model needs [constant, MLEN*BLEN, MLEN+BLEN]"
            )
        validation_raw = dict(raw.get("validation", {}))
        validation = ValidationReport(
            area_median_pct=validation_raw.get("area_median_pct"),
            area_max_pct=validation_raw.get("area_max_pct"),
            dynamic_median_pct=validation_raw.get("dynamic_median_pct"),
            dynamic_max_pct=validation_raw.get("dynamic_max_pct"),
            leakage_median_pct=validation_raw.get("leakage_median_pct"),
            leakage_max_pct=validation_raw.get("leakage_max_pct"),
            rank_correlation=validation_raw.get("rank_correlation"),
            cycle_error_pct=validation_raw.get("cycle_error_pct"),
            latency_mape_pct=validation_raw.get("latency_mape_pct"),
            missing_fields=tuple(validation_raw.get("missing_fields", ())),
            passed=bool(validation_raw.get("passed", False)),
        )
        return cls(
            model_version=str(raw["model_version"]),
            event_energy_models=models,
            hbm_energy_j_per_byte=float(raw.get("hbm_energy_j_per_byte", 0.0)),
            leakage_power_model=tuple(
                float(value)
                for value in raw.get("leakage_power_model", ())
            ),
            fixed_area_mm2=float(raw.get("fixed_area_mm2", 0.0)),
            vector_area_models=vector_area_models,
            selector_area_model=tuple(
                float(value) for value in raw.get("selector_area_model", ())
            ),
            structural_area_model=StructuralAreaEvidence(
                raw.get("structural_area_model", {})
            ),
            provenance_hash=str(raw.get("provenance_hash", "")),
            activity_provenance_hash=str(
                raw.get("activity_provenance_hash", "")
            ),
            artifact_catalog_sha256=str(
                raw.get("artifact_catalog_sha256", "")
            ),
            validation=validation,
            calibration_manifest_hash=str(
                raw.get("calibration_manifest_hash", "")
            ),
            hbm_energy_source=str(raw.get("hbm_energy_source", "")),
            synthesis_context={
                str(k): str(v)
                for k, v in dict(raw.get("synthesis_context", {})).items()
            },
            fit_summary={
                str(k): int(v)
                for k, v in dict(raw.get("fit_summary", {})).items()
            },
            source_sha256=None,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PowerCalibration":
        payload = Path(path).read_bytes()
        return replace(
            cls.from_dict(json.loads(payload)),
            source_sha256=hashlib.sha256(payload).hexdigest(),
        )

    def event_energy_j(self, event: EventCount) -> float:
        try:
            constant, cells, perimeter = self.event_energy_models[event.signature]
        except KeyError as exc:
            raise KeyError(f"uncalibrated event signature {event.signature!r}") from exc
        value = constant + cells * event.mlen * event.blen + perimeter * (event.mlen + event.blen)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid event energy for {event.signature!r}: {value}")
        return value

    def vector_area_mm2(
        self,
        vector_fp: str,
        *,
        mlen: int,
        blen: int,
    ) -> float:
        """Evaluate one fitted vector-area format at a legal geometry."""

        if mlen <= 0 or blen <= 0 or mlen % blen:
            raise ValueError(
                "vector geometry must satisfy MLEN>0, BLEN>0, MLEN%BLEN==0"
            )
        try:
            constant, cells, perimeter = self.vector_area_models[vector_fp]
        except KeyError as exc:
            raise KeyError(f"uncalibrated vector area {vector_fp!r}") from exc
        value = constant + cells * mlen * blen + perimeter * (mlen + blen)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid vector area for {vector_fp!r}: {value}")
        return value

    def selector_area_mm2(self, *, mlen: int, blen: int) -> float:
        """Evaluate the fitted PackedKV selector area delta."""

        if mlen <= 0 or blen <= 0 or mlen % blen:
            raise ValueError(
                "selector geometry must satisfy MLEN>0, BLEN>0, MLEN%BLEN==0"
            )
        constant, cells, perimeter = self.selector_area_model
        value = constant + cells * mlen * blen + perimeter * (mlen + blen)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid selector area: {value}")
        return value

    def complete_chip_leakage_w(self, *, mlen: int, blen: int) -> float:
        """Evaluate leakage fitted from complete-chip geometry anchors."""

        if mlen <= 0 or blen <= 0 or mlen % blen:
            raise ValueError(
                "leakage geometry must satisfy MLEN>0, BLEN>0, MLEN%BLEN==0"
            )
        constant, cells, perimeter = self.leakage_power_model
        value = constant + cells * mlen * blen + perimeter * (mlen + blen)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid complete-chip leakage: {value}")
        return value


@dataclass(frozen=True)
class PowerEstimate:
    compute_dynamic_energy_j: float
    vector_dynamic_energy_j: float
    selector_dynamic_energy_j: float
    dynamic_energy_j: float
    leakage_energy_j: float
    leakage_power_w: float
    hbm_energy_j: float
    total_energy_j: float
    average_power_w: float
    area_mm2: float
    matrix_area_mm2: float
    sram_area_mm2: float
    fixed_area_mm2: float
    vector_area_mm2: float
    selector_area_mm2: float
    array_signatures: tuple[str, ...]
    events: tuple[EventCount, ...]
    hbm_bytes: float
    area_config: Mapping[str, int]
    vector_fp: str
    selector_enabled: bool
    mlen: int
    blen: int
    calibrated: bool
    rankable: bool
    missing_signatures: tuple[str, ...] = field(default_factory=tuple)
    calibration_source_sha256: str | None = None
    calibration_provenance_hash: str = ""
    calibration_activity_provenance_hash: str = ""
    structural_area_id: str = ""
    calibration_domain: Mapping[str, int] = field(default_factory=dict)
    extrapolated: bool = False
    extrapolation_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "compute_dynamic_energy_j": self.compute_dynamic_energy_j,
            "vector_dynamic_energy_j": self.vector_dynamic_energy_j,
            "selector_dynamic_energy_j": self.selector_dynamic_energy_j,
            "dynamic_energy_j": self.dynamic_energy_j,
            "leakage_energy_j": self.leakage_energy_j,
            "leakage_power_w": self.leakage_power_w,
            "hbm_energy_j": self.hbm_energy_j,
            "total_energy_j": self.total_energy_j,
            "average_power_w": self.average_power_w,
            "area_mm2": self.area_mm2,
            "matrix_area_mm2": self.matrix_area_mm2,
            "sram_area_mm2": self.sram_area_mm2,
            "fixed_area_mm2": self.fixed_area_mm2,
            "vector_area_mm2": self.vector_area_mm2,
            "selector_area_mm2": self.selector_area_mm2,
            "array_signatures": list(self.array_signatures),
            "events": [
                {
                    "signature": event.signature,
                    "count": event.count,
                    "MLEN": event.mlen,
                    "BLEN": event.blen,
                }
                for event in self.events
            ],
            "hbm_bytes": self.hbm_bytes,
            "area_config": dict(sorted(self.area_config.items())),
            "vector_fp": self.vector_fp,
            "selector_enabled": self.selector_enabled,
            "MLEN": self.mlen,
            "BLEN": self.blen,
            "calibrated": self.calibrated,
            "rankable": self.rankable,
            "missing_signatures": list(self.missing_signatures),
            "calibration_source_sha256": self.calibration_source_sha256,
            "calibration_provenance_hash": self.calibration_provenance_hash,
            "calibration_activity_provenance_hash": (
                self.calibration_activity_provenance_hash
            ),
            "structural_area_id": self.structural_area_id,
            "calibration_domain": dict(sorted(self.calibration_domain.items())),
            "extrapolated": self.extrapolated,
            "extrapolation_reasons": list(self.extrapolation_reasons),
        }


def estimate_power(
    calibration: PowerCalibration,
    events: Iterable[EventCount],
    *,
    elapsed_s: float,
    hbm_bytes: float,
    vector_fp: str,
    area_config: Mapping[str, int],
    selector_enabled: bool = True,
) -> PowerEstimate:
    """Estimate total energy and average power for one candidate.

    Missing operation signatures are retained in the result and make the point
    non-rankable instead of silently receiving a proxy coefficient.
    """
    if elapsed_s <= 0 or not math.isfinite(elapsed_s):
        raise ValueError("elapsed_s must be finite and positive")
    if hbm_bytes < 0:
        raise ValueError("byte counts must be non-negative")

    event_values = tuple(events)
    compute_dynamic = 0.0
    vector_dynamic = 0.0
    selector_dynamic = 0.0
    missing: list[str] = []
    for event in event_values:
        try:
            energy = event.count * calibration.event_energy_j(event)
        except KeyError:
            missing.append(event.signature)
            continue
        if event.signature.startswith(("LINEAR:", "QK:", "PV:")):
            compute_dynamic += energy
        elif event.signature.startswith("VECTOR:"):
            vector_dynamic += energy
        elif event.signature == "SELECTOR:PACKED_KV":
            selector_dynamic += energy
        else:
            missing.append(f"EVENT_CLASS:{event.signature}")
    dynamic = compute_dynamic + vector_dynamic + selector_dynamic
    array_signatures = tuple(
        sorted(
            {
                event.signature
                for event in event_values
                if event.count > 0
                and event.signature.startswith(("LINEAR:", "QK:", "PV:"))
            }
        )
    )
    geometries = {(event.mlen, event.blen) for event in event_values}
    if len(geometries) > 1:
        raise ValueError("one estimate cannot mix array geometries")
    geometry = next(iter(geometries), None)
    operation_signatures = {
        operation: {
            event.signature
            for event in event_values
            if event.count > 0 and event.signature.startswith(f"{operation}:")
        }
        for operation in ("LINEAR", "QK", "PV")
    }
    for operation, signatures in operation_signatures.items():
        if len(signatures) != 1:
            missing.append(f"EVENT:{operation}")
    vector_events = sum(
        event.count
        for event in event_values
        if event.signature == f"VECTOR:{vector_fp}"
    )
    if vector_events <= 0:
        missing.append(f"EVENT:VECTOR:{vector_fp}")
    selector_events = sum(
        event.count
        for event in event_values
        if event.signature == "SELECTOR:PACKED_KV"
    )
    if selector_enabled and selector_events <= 0:
        missing.append("EVENT:SELECTOR:PACKED_KV")
    if not selector_enabled and selector_events:
        raise ValueError("selector events require selector_enabled=True")
    if geometry is None:
        missing.append("LEAKAGE:geometry")
        leakage_power_w = 0.0
        matrix_area_mm2 = 0.0
        sram_area_mm2 = 0.0
        fixed_area_mm2 = 0.0
        vector_area_mm2 = 0.0
        selector_area_mm2 = 0.0
    else:
        leakage_power_w = calibration.complete_chip_leakage_w(
            mlen=geometry[0],
            blen=geometry[1],
        )
        matrix_areas: list[float] = []
        try:
            matrix_areas = [
                calibration.structural_area_model.matrix_area_mm2(
                    signature,
                    mlen=geometry[0],
                    blen=geometry[1],
                    reference_corner=True,
                    scale_width=int(area_config.get("MX_SCALE_WIDTH", 0)),
                )
                for signature in array_signatures
            ]
        except (KeyError, TypeError, ValueError):
            missing.append("AREA:structural_matrix")
        matrix_area_mm2 = max(matrix_areas, default=0.0)
        try:
            sram_area_mm2 = calibration.structural_area_model.sram_area_mm2(
                array_signatures,
                vector_fp=vector_fp,
                area_config=area_config,
                mlen=geometry[0],
                blen=geometry[1],
            )
        except (KeyError, TypeError, ValueError):
            missing.append("AREA:sram")
            sram_area_mm2 = 0.0
        pdk_scale = float(
            calibration.structural_area_model.payload["pdk_scale_reference"]
        )
        fixed_area_mm2 = calibration.fixed_area_mm2 * pdk_scale
        try:
            vector_area_mm2 = calibration.vector_area_mm2(
                vector_fp,
                mlen=geometry[0],
                blen=geometry[1],
            ) * pdk_scale
        except KeyError:
            missing.append(f"AREA:VECTOR:{vector_fp}")
            vector_area_mm2 = 0.0
        selector_area_mm2 = (
            calibration.selector_area_mm2(
                mlen=geometry[0],
                blen=geometry[1],
            )
            * pdk_scale
            if selector_enabled
            else 0.0
        )
    leakage = leakage_power_w * elapsed_s
    hbm = hbm_bytes * calibration.hbm_energy_j_per_byte
    total = dynamic + leakage + hbm
    area = (
        fixed_area_mm2
        + matrix_area_mm2
        + vector_area_mm2
        + sram_area_mm2
        + selector_area_mm2
    )
    canonical_area_config = {
        str(key): int(value) for key, value in dict(area_config).items()
    }
    extrapolation_reasons: list[str] = []
    if geometry is not None:
        if not (
            POWER_INTERPOLATION_DOMAIN["mlen_min"]
            <= geometry[0]
            <= POWER_INTERPOLATION_DOMAIN["mlen_max"]
        ):
            extrapolation_reasons.append("MLEN")
        if not (
            POWER_INTERPOLATION_DOMAIN["blen_min"]
            <= geometry[1]
            <= POWER_INTERPOLATION_DOMAIN["blen_max"]
        ):
            extrapolation_reasons.append("BLEN")
    hardware_fp_binding = calibration.synthesis_context[
        "hardware_fp_binding"
    ]
    if vector_fp != hardware_fp_binding:
        extrapolation_reasons.append("FP_DEPENDENT_MATRIX_AND_CHIP_LOGIC")
    missing_tuple = tuple(sorted(set(missing)))
    extrapolated = bool(extrapolation_reasons)
    calibrated = (
        calibration.validation.passed
        and calibration.validation.meets_publication_gate
    )
    return PowerEstimate(
        compute_dynamic_energy_j=compute_dynamic,
        vector_dynamic_energy_j=vector_dynamic,
        selector_dynamic_energy_j=selector_dynamic,
        dynamic_energy_j=dynamic,
        leakage_energy_j=leakage,
        leakage_power_w=leakage_power_w,
        hbm_energy_j=hbm,
        total_energy_j=total,
        average_power_w=total / elapsed_s,
        area_mm2=area,
        matrix_area_mm2=matrix_area_mm2,
        sram_area_mm2=sram_area_mm2,
        fixed_area_mm2=fixed_area_mm2,
        vector_area_mm2=vector_area_mm2,
        selector_area_mm2=selector_area_mm2,
        array_signatures=array_signatures,
        events=event_values,
        hbm_bytes=hbm_bytes,
        area_config=canonical_area_config,
        vector_fp=vector_fp,
        selector_enabled=selector_enabled,
        mlen=geometry[0] if geometry is not None else 0,
        blen=geometry[1] if geometry is not None else 0,
        calibrated=calibrated,
        rankable=calibrated and not missing_tuple and not extrapolated,
        missing_signatures=missing_tuple,
        calibration_source_sha256=calibration.source_sha256,
        calibration_provenance_hash=calibration.provenance_hash,
        calibration_activity_provenance_hash=(
            calibration.activity_provenance_hash
        ),
        structural_area_id=calibration.structural_area_model.evidence_id,
        calibration_domain=POWER_INTERPOLATION_DOMAIN,
        extrapolated=extrapolated,
        extrapolation_reasons=tuple(sorted(extrapolation_reasons)),
    )


def _relative_errors(measured: Sequence[float], predicted: Sequence[float]) -> list[float]:
    if len(measured) != len(predicted):
        raise ValueError("measured and predicted sequences must have equal length")
    out = []
    for actual, estimate in zip(measured, predicted):
        if actual <= 0 or not (math.isfinite(actual) and math.isfinite(estimate)):
            raise ValueError("validation values must be finite and measured values positive")
        out.append(abs(estimate - actual) / actual * 100.0)
    return out


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank correlation needs equal sequences with at least two values")
    x, y = _ranks(left), _ranks(right)
    xm, ym = statistics.mean(x), statistics.mean(y)
    num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    den = math.sqrt(sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y))
    return 1.0 if den == 0 and x == y else (0.0 if den == 0 else num / den)


def validate_predictions(
    *,
    measured_area: Sequence[float] = (),
    predicted_area: Sequence[float] = (),
    measured_dynamic: Sequence[float] = (),
    predicted_dynamic: Sequence[float] = (),
    measured_leakage: Sequence[float] = (),
    predicted_leakage: Sequence[float] = (),
    measured_cycles: Sequence[float] = (),
    predicted_cycles: Sequence[float] = (),
    measured_latency: Sequence[float] = (),
    predicted_latency: Sequence[float] = (),
    gate: CalibrationGate | None = None,
) -> ValidationReport:
    """Evaluate publication gates on held-out synthesis and trace points."""
    gate = gate or CalibrationGate()
    missing: list[str] = []

    def errors(name: str, measured: Sequence[float], predicted: Sequence[float]):
        if len(measured) < 2 or len(predicted) < 2:
            missing.append(name)
            return None
        return _relative_errors(measured, predicted)

    area = errors("area", measured_area, predicted_area)
    dynamic = errors("dynamic_power", measured_dynamic, predicted_dynamic)
    leakage = errors("leakage_power", measured_leakage, predicted_leakage)
    cycles = errors("cycles", measured_cycles, predicted_cycles)
    latency = errors("latency", measured_latency, predicted_latency)
    rank = None
    if (
        dynamic is not None
        and len(set(measured_dynamic)) >= 2
        and len(set(predicted_dynamic)) >= 2
    ):
        rank = _spearman(measured_dynamic, predicted_dynamic)
    else:
        missing.append("ranking")

    report = ValidationReport(
        area_median_pct=statistics.median(area) if area is not None else None,
        area_max_pct=max(area) if area is not None else None,
        dynamic_median_pct=statistics.median(dynamic) if dynamic is not None else None,
        dynamic_max_pct=max(dynamic) if dynamic is not None else None,
        leakage_median_pct=(
            statistics.median(leakage) if leakage is not None else None
        ),
        leakage_max_pct=max(leakage) if leakage is not None else None,
        rank_correlation=rank,
        cycle_error_pct=max(cycles) if cycles is not None else None,
        latency_mape_pct=statistics.mean(latency) if latency is not None else None,
        missing_fields=tuple(missing),
    )
    passed = (
        not report.missing_fields
        and report.area_median_pct <= gate.area_median_pct
        and report.area_max_pct <= gate.area_max_pct
        and report.dynamic_median_pct <= gate.dynamic_median_pct
        and report.dynamic_max_pct <= gate.dynamic_max_pct
        and report.leakage_median_pct <= gate.leakage_median_pct
        and report.leakage_max_pct <= gate.leakage_max_pct
        and report.rank_correlation >= gate.rank_correlation
        and report.cycle_error_pct <= gate.cycle_error_pct
        and report.latency_mape_pct <= gate.latency_mape_pct
    )
    return ValidationReport(**{**report.to_dict(), "missing_fields": report.missing_fields, "passed": passed})
