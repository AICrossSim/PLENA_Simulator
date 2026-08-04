"""Fit event-energy coefficients and evaluate the larger-geometry holdouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .artifact_catalog import validate_artifact_catalog
from .calibration_manifest import (
    HARDWARE_FP_BINDING,
    HOLDOUT_GEOMETRIES,
    MX_BLOCK_SIZE,
    SELECTOR_SIGNATURE,
    TRAIN_GEOMETRIES,
    VECTOR_FP,
    build_manifest,
    event_signatures,
    manifest_hash,
    operand_signatures,
)
from .model import validate_predictions
from .structural_area import (
    StructuralAreaEvidence,
    build_structural_area_evidence,
)

AREA_CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "area" / "calibration"
DEFAULT_AREA_COEFFICIENTS = (
    AREA_CALIBRATION_DIR / "matrix_structural_coefficients.json"
)
DEFAULT_SRAM_MACROS = AREA_CALIBRATION_DIR / "asap7_sram_macro_table.csv"
DEFAULT_AREA_INPUTS = (
    AREA_CALIBRATION_DIR / "matrix_machine_mxint.csv",
    AREA_CALIBRATION_DIR / "matrix_machine_mxfp.csv",
)


def _number(row: dict, key: str, default: float | None = None) -> float:
    value = row.get(key, "")
    if value in (None, ""):
        if default is None:
            raise ValueError(f"missing {key}")
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {key}")
    return result


def _event_energy(row: dict) -> float:
    events = _number(row, "events")
    if events <= 0:
        raise ValueError("events must be positive")
    elapsed_s = _number(row, "cycles") * _number(row, "clock_ns", 1.0) * 1e-9
    dynamic_power = _number(row, "dynamic_power_w")
    if elapsed_s <= 0 or dynamic_power <= 0:
        raise ValueError("cycles, clock, and dynamic power must be positive")
    return dynamic_power * elapsed_s / events


def _features(row: dict) -> tuple[float, float, float]:
    mlen, blen = _number(row, "MLEN"), _number(row, "BLEN")
    return 1.0, mlen * blen, mlen + blen


def _geometry(row: dict) -> tuple[int, int]:
    return int(_number(row, "MLEN")), int(_number(row, "BLEN"))


def _nonnegative_fit(rows: list[dict]) -> tuple[float, float, float]:
    """Projected coordinate descent for three non-negative coefficients."""
    if len(rows) < 3:
        raise ValueError("at least three training geometries are required per signature")
    raw_x = [_features(row) for row in rows]
    y = [_event_energy(row) for row in rows]
    cell_scale = statistics.median(x[1] for x in raw_x)
    perimeter_scale = statistics.median(x[2] for x in raw_x)
    x = [(1.0, a / cell_scale, b / perimeter_scale) for _, a, b in raw_x]
    coefficients = [max(statistics.mean(y), 0.0), 0.0, 0.0]
    ridge = 1e-12
    for _ in range(5000):
        previous = tuple(coefficients)
        for column in range(3):
            numerator = 0.0
            denominator = ridge
            for features, target in zip(x, y):
                residual = target - sum(
                    coefficients[index] * features[index]
                    for index in range(3)
                    if index != column
                )
                numerator += features[column] * residual
                denominator += features[column] ** 2
            coefficients[column] = max(0.0, numerator / denominator)
        if max(abs(a - b) for a, b in zip(previous, coefficients)) < 1e-24:
            break
    return (
        coefficients[0],
        coefficients[1] / cell_scale,
        coefficients[2] / perimeter_scale,
    )


def _nonnegative_area_fit(rows: list[dict]) -> tuple[float, float, float]:
    """Fit non-negative area coefficients over the same geometry basis."""
    if len(rows) < 3:
        raise ValueError("at least three area geometries are required per signature")
    raw_x = [_features(row) for row in rows]
    y = [_number(row, "area_mm2") for row in rows]
    cell_scale = statistics.median(x[1] for x in raw_x)
    perimeter_scale = statistics.median(x[2] for x in raw_x)
    x = [(1.0, a / cell_scale, b / perimeter_scale) for _, a, b in raw_x]
    coefficients = [max(statistics.mean(y), 0.0), 0.0, 0.0]
    ridge = 1e-12
    for _ in range(5000):
        previous = tuple(coefficients)
        for column in range(3):
            numerator = 0.0
            denominator = ridge
            for features, target in zip(x, y):
                residual = target - sum(
                    coefficients[index] * features[index]
                    for index in range(3)
                    if index != column
                )
                numerator += features[column] * residual
                denominator += features[column] ** 2
            coefficients[column] = max(0.0, numerator / denominator)
        if max(abs(a - b) for a, b in zip(previous, coefficients)) < 1e-24:
            break
    return (
        coefficients[0],
        coefficients[1] / cell_scale,
        coefficients[2] / perimeter_scale,
    )


def _nonnegative_leakage_fit(rows: list[dict]) -> tuple[float, float, float]:
    """Fit complete-chip leakage over the calibrated geometry basis."""
    if len(rows) < 3:
        raise ValueError("at least three chip-leakage geometries are required")
    raw_x = [_features(row) for row in rows]
    y = [_number(row, "leakage_power_w") for row in rows]
    if any(value <= 0 for value in y):
        raise ValueError("complete-chip leakage measurements must be positive")
    cell_scale = statistics.median(x[1] for x in raw_x)
    perimeter_scale = statistics.median(x[2] for x in raw_x)
    x = [(1.0, a / cell_scale, b / perimeter_scale) for _, a, b in raw_x]
    coefficients = [max(statistics.mean(y), 0.0), 0.0, 0.0]
    ridge = 1e-12
    for _ in range(5000):
        previous = tuple(coefficients)
        for column in range(3):
            numerator = 0.0
            denominator = ridge
            for features, target in zip(x, y):
                residual = target - sum(
                    coefficients[index] * features[index]
                    for index in range(3)
                    if index != column
                )
                numerator += features[column] * residual
                denominator += features[column] ** 2
            coefficients[column] = max(0.0, numerator / denominator)
        if max(abs(a - b) for a, b in zip(previous, coefficients)) < 1e-24:
            break
    return (
        coefficients[0],
        coefficients[1] / cell_scale,
        coefficients[2] / perimeter_scale,
    )


def _predict_energy(model: tuple[float, float, float], row: dict) -> float:
    return sum(coefficient * feature for coefficient, feature in zip(model, _features(row)))


def _complete_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return [row for row in rows if row.get("status", "").lower() == "complete"]


def _truthy(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _is_sha256(value: object) -> bool:
    token = str(value)
    return len(token) == 64 and all(
        character in "0123456789abcdef" for character in token
    )


def _row_identity(row: dict) -> tuple[str, str, int, int, str, bool]:
    return (
        str(row.get("component", "")),
        str(row.get("signature", "")),
        *_geometry(row),
        str(row.get("split", "")),
        _truthy(row.get("selector_enabled", "")),
    )


def _manifest_identity(point) -> tuple[str, str, int, int, str, bool]:
    return (
        point.component,
        point.signature,
        point.mlen,
        point.blen,
        point.split,
        point.selector_enabled,
    )


def _audit_measurement_rows(
    rows: list[dict],
) -> tuple[list[dict], list[str], dict[str, str]]:
    expected_points = {
        _manifest_identity(point): point for point in build_manifest()
    }
    expected = set(expected_points)
    by_identity: dict[tuple[str, str, int, int, str, bool], list[dict]] = defaultdict(list)
    unexpected: list[tuple[str, str, int, int, str, bool]] = []
    for row in rows:
        identity = _row_identity(row)
        if identity in expected:
            by_identity[identity].append(row)
        else:
            unexpected.append(identity)

    failures = [
        f"missing_point:{identity}"
        for identity in sorted(expected - set(by_identity))
    ]
    failures.extend(
        f"unexpected_point:{identity}"
        for identity in sorted(set(unexpected))
    )
    failures.extend(
        f"duplicate_point:{identity}"
        for identity, values in sorted(by_identity.items())
        if len(values) != 1
    )
    usable = [
        values[0]
        for identity, values in sorted(by_identity.items())
        if identity in expected
    ]

    for row in usable:
        point = expected_points[_row_identity(row)]
        if str(row.get("point_id", "")) != point.point_id:
            failures.append(f"point_id:{_row_identity(row)}")
        if not math.isclose(
            _number(row, "clock_ns"),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            failures.append(f"clock_constraint:{_row_identity(row)}")
        if point.requires_saif:
            if str(row.get("activity_class", "")) != point.activity_class:
                failures.append(f"activity_class:{_row_identity(row)}")
            for name in ("saif_sha256", "decode_trace_sha256"):
                if not _is_sha256(row.get(name, "")):
                    failures.append(f"{name}:{_row_identity(row)}")
            if not str(row.get("saif_source_id", "")).strip():
                failures.append(f"saif_source_id:{_row_identity(row)}")
            if not str(row.get("activity_generator", "")).strip():
                failures.append(f"activity_generator:{_row_identity(row)}")
            try:
                _event_energy(row)
            except ValueError:
                failures.append(f"dynamic_measurement:{_row_identity(row)}")

    context: dict[str, str] = {}
    for name in ("dc_tool_version", "library_id", "process_corner"):
        values = {
            str(row.get(name, "")).strip()
            for row in usable
            if str(row.get(name, "")).strip()
        }
        if len(values) != 1 or any(
            not str(row.get(name, "")).strip() for row in usable
        ):
            failures.append(f"synthesis_context:{name}")
        elif values:
            context[name] = next(iter(values))
    activity_rows = [
        row
        for row in usable
        if expected_points[_row_identity(row)].requires_saif
    ]
    generators = {
        str(row.get("activity_generator", "")).strip()
        for row in activity_rows
        if str(row.get("activity_generator", "")).strip()
    }
    if len(generators) != 1 or any(
        not str(row.get("activity_generator", "")).strip()
        for row in activity_rows
    ):
        failures.append("synthesis_context:activity_generator")
    elif generators:
        context["activity_generator"] = next(iter(generators))
    block_sizes: set[int] = set()
    for row in usable:
        raw = row.get(
            "MX_BLOCK_SIZE",
            row.get("BLOCK_DIM", row.get("block_size", "")),
        )
        try:
            block_sizes.add(int(raw))
        except (TypeError, ValueError):
            failures.append(f"mx_block_size:{_row_identity(row)}")
    if block_sizes != {MX_BLOCK_SIZE}:
        failures.append("mx_block_size")
    else:
        context["mx_block_size"] = str(MX_BLOCK_SIZE)
    fp_bindings = {
        str(row.get("hardware_fp_binding", "")).strip()
        for row in usable
        if str(row.get("hardware_fp_binding", "")).strip()
    }
    if fp_bindings != {HARDWARE_FP_BINDING} or any(
        str(row.get("hardware_fp_binding", "")).strip()
        != HARDWARE_FP_BINDING
        for row in usable
    ):
        failures.append("hardware_fp_binding")
    else:
        context["hardware_fp_binding"] = HARDWARE_FP_BINDING
    return usable, sorted(set(failures)), context


def _optional_pairs(rows: list[dict], measured_key: str, predicted_key: str):
    measured, predicted = [], []
    for row in rows:
        if row.get(measured_key, "") != "" and row.get(predicted_key, "") != "":
            measured.append(_number(row, measured_key))
            predicted.append(_number(row, predicted_key))
    return measured, predicted


def _missing_geometry_coverage(
    rows: list[dict],
    *,
    component: str,
    signatures: tuple[str, ...],
    split: str,
    geometries: tuple[tuple[int, int], ...],
) -> list[str]:
    available = {
        (row.get("signature", ""), _geometry(row))
        for row in rows
        if row.get("component") == component and row.get("split") == split
    }
    return [
        f"{split}:{component}:{signature}:{mlen}x{blen}"
        for signature in signatures
        for mlen, blen in geometries
        if (signature, (mlen, blen)) not in available
    ]


def _selector_pairs(
    rows: list[dict],
    split: str,
) -> list[tuple[dict, dict]]:
    grouped: dict[tuple[int, int], dict[bool, dict]] = defaultdict(dict)
    for row in rows:
        if (
            row.get("component") != "selector"
            or row.get("split") != split
        ):
            continue
        grouped[_geometry(row)][_truthy(row.get("selector_enabled", ""))] = row
    return [
        (values[False], values[True])
        for _, values in sorted(grouped.items())
        if True in values and False in values
    ]


def _selector_delta_rows(
    rows: list[dict],
    split: str,
) -> tuple[list[dict], list[str]]:
    deltas: list[dict] = []
    failures: list[str] = []
    for disabled, enabled in _selector_pairs(rows, split):
        identity = f"{split}:{_geometry(enabled)[0]}x{_geometry(enabled)[1]}"
        comparable = (
            disabled.get("saif_sha256") == enabled.get("saif_sha256")
            and disabled.get("decode_trace_sha256")
            == enabled.get("decode_trace_sha256")
            and disabled.get("saif_source_id") == enabled.get("saif_source_id")
        )
        for key in ("events", "cycles", "clock_ns"):
            try:
                comparable = comparable and math.isclose(
                    _number(disabled, key),
                    _number(enabled, key),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            except ValueError:
                comparable = False
        if not comparable:
            failures.append(f"selector_pair:{identity}")
            continue
        try:
            dynamic_delta = _number(enabled, "dynamic_power_w") - _number(
                disabled,
                "dynamic_power_w",
            )
            area_delta = _number(enabled, "area_mm2") - _number(
                disabled,
                "area_mm2",
            )
        except ValueError:
            failures.append(f"selector_pair:{identity}")
            continue
        if dynamic_delta <= 0 or area_delta <= 0:
            failures.append(f"selector_delta:{identity}")
            continue
        row = dict(enabled)
        row.update(
            signature=SELECTOR_SIGNATURE,
            dynamic_power_w=str(dynamic_delta),
            area_mm2=str(area_delta),
        )
        deltas.append(row)
    return deltas, failures


def _activity_provenance_hash(rows: list[dict]) -> str:
    activity = [
        {
            "point_id": str(row["point_id"]),
            "activity_class": str(row["activity_class"]),
            "saif_sha256": str(row["saif_sha256"]),
            "decode_trace_sha256": str(row["decode_trace_sha256"]),
            "saif_source_id": str(row["saif_source_id"]),
            "activity_generator": str(row["activity_generator"]),
        }
        for row in rows
        if str(row.get("saif_sha256", "")).strip()
    ]
    payload = json.dumps(
        sorted(activity, key=lambda value: value["point_id"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def fit(
    path: str | Path,
    *,
    hbm_energy_j_per_byte: float | None = None,
    hbm_energy_source: str = "",
    artifact_catalog: str | Path | None = None,
    structural_area_coefficients: str | Path = DEFAULT_AREA_COEFFICIENTS,
    sram_macro_table: str | Path = DEFAULT_SRAM_MACROS,
    structural_area_inputs: tuple[str | Path, ...] = DEFAULT_AREA_INPUTS,
) -> dict:
    source = Path(path)
    rows, input_failures, synthesis_context = _audit_measurement_rows(
        _complete_rows(source)
    )
    artifact_catalog_sha256 = ""
    if artifact_catalog is None:
        input_failures.append("artifact_catalog:missing")
    else:
        try:
            artifact_catalog_sha256, catalog_failures = (
                validate_artifact_catalog(artifact_catalog, rows)
            )
            input_failures.extend(catalog_failures)
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError):
            input_failures.append("artifact_catalog:invalid")
    train = [row for row in rows if row.get("split") == "train"]
    holdout = [row for row in rows if row.get("split") == "holdout"]
    selector_train, selector_train_failures = _selector_delta_rows(
        rows,
        "train",
    )
    selector_holdout, selector_holdout_failures = _selector_delta_rows(
        rows,
        "holdout",
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in train:
        if row.get("component") in {"array", "vector"}:
            grouped[row["signature"]].append(row)
    models = {
        signature: _nonnegative_fit(group)
        for signature, group in sorted(grouped.items())
        if len(group) >= 3
    }
    if len(selector_train) >= 3:
        models[SELECTOR_SIGNATURE] = _nonnegative_fit(selector_train)

    vector_area_groups: dict[str, list[dict]] = defaultdict(list)
    for row in train:
        if row.get("component") == "vector" and row.get("area_mm2", "") != "":
            vector_area_groups[row["signature"].removeprefix("VECTOR:")].append(
                row
            )
    vector_area_models = {
        signature: _nonnegative_area_fit(group)
        for signature, group in sorted(vector_area_groups.items())
        if len(group) >= 3
    }
    selector_area_model = (
        _nonnegative_area_fit(selector_train)
        if len(selector_train) >= 3
        else None
    )
    leakage_train = [
        row for row in train if row.get("component") == "chip_leakage"
    ]
    leakage_model = (
        _nonnegative_leakage_fit(leakage_train)
        if len(leakage_train) >= 3
        else None
    )
    try:
        structural_area = build_structural_area_evidence(
            structural_area_coefficients,
            sram_macro_table,
            calibration_inputs=structural_area_inputs,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        structural_area = None

    measured_dynamic, predicted_dynamic = [], []
    for row in holdout + selector_holdout:
        model = models.get(row.get("signature", ""))
        if model is None or row.get("events", "") == "":
            continue
        elapsed_s = _number(row, "cycles") * _number(row, "clock_ns", 1.0) * 1e-9
        if elapsed_s <= 0:
            continue
        measured_dynamic.append(_number(row, "dynamic_power_w"))
        predicted_dynamic.append(_predict_energy(model, row) * _number(row, "events") / elapsed_s)

    measured_area, predicted_area = [], []
    for row in holdout:
        if (
            row.get("component") == "array"
            and structural_area is not None
            and structural_area.passed
            and row.get("area_mm2", "") != ""
        ):
            measured_area.append(_number(row, "area_mm2"))
            predicted_area.append(
                structural_area.matrix_area_mm2(
                    row["signature"],
                    mlen=_geometry(row)[0],
                    blen=_geometry(row)[1],
                    reference_corner=False,
                )
            )
        elif row.get("component") == "vector":
            model = vector_area_models.get(
                row.get("signature", "").removeprefix("VECTOR:")
            )
            if model is not None and row.get("area_mm2", "") != "":
                measured_area.append(_number(row, "area_mm2"))
                predicted_area.append(_predict_energy(model, row))

    measured_leakage, predicted_leakage = [], []
    for row in holdout:
        if row.get("component") != "chip_leakage" or leakage_model is None:
            continue
        measured_leakage.append(_number(row, "leakage_power_w"))
        predicted_leakage.append(_predict_energy(leakage_model, row))

    def component_area(component: str) -> list[float]:
        return [
            _number(row, "area_mm2")
            for row in train
            if row.get("component") == component
            and row.get("area_mm2", "") != ""
        ]

    fixed_area = component_area("fixed")
    if fixed_area:
        prediction = statistics.median(fixed_area)
        for row in holdout:
            if row.get("component") == "fixed" and row.get("area_mm2", "") != "":
                measured_area.append(_number(row, "area_mm2"))
                predicted_area.append(prediction)
    if selector_area_model is not None:
        for row in selector_holdout:
            measured_area.append(_number(row, "area_mm2"))
            predicted_area.append(_predict_energy(selector_area_model, row))

    measured_cycles, predicted_cycles = _optional_pairs(
        [row for row in holdout if row.get("component") == "cycle"],
        "rtl_cycles",
        "emulator_cycles",
    )
    measured_latency, predicted_latency = _optional_pairs(
        [row for row in holdout if row.get("component") == "latency"],
        "measured_latency_s",
        "analytical_latency_s",
    )
    validation = validate_predictions(
        measured_area=measured_area,
        predicted_area=predicted_area,
        measured_dynamic=measured_dynamic,
        predicted_dynamic=predicted_dynamic,
        measured_leakage=measured_leakage,
        predicted_leakage=predicted_leakage,
        measured_cycles=measured_cycles,
        predicted_cycles=predicted_cycles,
        measured_latency=measured_latency,
        predicted_latency=predicted_latency,
    )

    missing_coefficients: list[str] = []
    if hbm_energy_j_per_byte is None or hbm_energy_j_per_byte <= 0:
        missing_coefficients.append("hbm_energy_j_per_byte")
    if not hbm_energy_source.strip():
        missing_coefficients.append("hbm_energy_source")
    if leakage_model is None or not any(value > 0 for value in leakage_model):
        missing_coefficients.append("leakage_power_model")
    if not fixed_area or statistics.median(fixed_area) <= 0:
        missing_coefficients.append("fixed_area_mm2")
    if structural_area is None or not structural_area.passed:
        missing_coefficients.append("structural_area_model")
    if selector_area_model is None or not any(
        value > 0 for value in selector_area_model
    ):
        missing_coefficients.append("selector_area_model")
    for vector_format in VECTOR_FP:
        model = vector_area_models.get(vector_format)
        if model is None or not any(value > 0 for value in model):
            missing_coefficients.append(f"vector_area_models:{vector_format}")
    for signature in event_signatures():
        if signature not in models:
            missing_coefficients.append(f"event_energy_models:{signature}")

    coverage_failures: list[str] = [
        *input_failures,
        *selector_train_failures,
        *selector_holdout_failures,
    ]
    for component, signatures in (
        ("array", operand_signatures()),
        ("vector", tuple(f"VECTOR:{value}" for value in VECTOR_FP)),
    ):
        coverage_failures.extend(
            _missing_geometry_coverage(
                rows,
                component=component,
                signatures=signatures,
                split="train",
                geometries=TRAIN_GEOMETRIES,
            )
        )
        coverage_failures.extend(
            _missing_geometry_coverage(
                rows,
                component=component,
                signatures=signatures,
                split="holdout",
                geometries=HOLDOUT_GEOMETRIES,
            )
        )
    for component in ("fixed",):
        coverage_failures.extend(
            _missing_geometry_coverage(
                rows,
                component=component,
                signatures=(component.upper(),),
                split="train",
                geometries=TRAIN_GEOMETRIES,
            )
        )
        coverage_failures.extend(
            _missing_geometry_coverage(
                rows,
                component=component,
                signatures=(component.upper(),),
                split="holdout",
                geometries=HOLDOUT_GEOMETRIES,
            )
        )
    coverage_failures.extend(
        _missing_geometry_coverage(
            rows,
            component="chip_leakage",
            signatures=("CHIP_LEAKAGE",),
            split="train",
            geometries=TRAIN_GEOMETRIES,
        )
    )
    coverage_failures.extend(
        _missing_geometry_coverage(
            rows,
            component="chip_leakage",
            signatures=("CHIP_LEAKAGE",),
            split="holdout",
            geometries=HOLDOUT_GEOMETRIES,
        )
    )
    if len(selector_train) != len(TRAIN_GEOMETRIES):
        coverage_failures.append("train:selector")
    if len(selector_holdout) != len(HOLDOUT_GEOMETRIES):
        coverage_failures.append("holdout:selector")
    if len(measured_cycles) < 2:
        coverage_failures.append("holdout:cycle")
    if len(measured_latency) < 2:
        coverage_failures.append("holdout:latency")

    validation_dict = validation.to_dict()
    if missing_coefficients or coverage_failures:
        validation_dict["missing_fields"] = sorted(
            set(validation_dict["missing_fields"])
            | {f"coefficient:{name}" for name in missing_coefficients}
            | {f"coverage:{name}" for name in coverage_failures}
        )
        validation_dict["passed"] = False

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "model_version": "plena-event-power",
        "event_energy_models": {key: list(value) for key, value in models.items()},
        "structural_area_model": (
            structural_area.to_dict() if structural_area is not None else {}
        ),
        "vector_area_models": {
            key: list(value) for key, value in vector_area_models.items()
        },
        "selector_area_model": (
            list(selector_area_model) if selector_area_model is not None else []
        ),
        "hbm_energy_j_per_byte": float(hbm_energy_j_per_byte or 0.0),
        "leakage_power_model": (
            list(leakage_model) if leakage_model is not None else []
        ),
        "fixed_area_mm2": statistics.median(fixed_area) if fixed_area else 0.0,
        "provenance_hash": digest,
        "activity_provenance_hash": _activity_provenance_hash(rows),
        "artifact_catalog_sha256": artifact_catalog_sha256,
        "calibration_manifest_hash": manifest_hash(),
        "hbm_energy_source": hbm_energy_source.strip(),
        "synthesis_context": synthesis_context,
        "validation": validation_dict,
        "fit_summary": {
            "complete_rows": len(rows),
            "train_rows": len(train),
            "holdout_rows": len(holdout),
            "signature_count": len(models),
            "structural_area_signature_count": len(operand_signatures()),
            "vector_area_signature_count": len(vector_area_models),
            "coverage_failures": len(coverage_failures),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("measurements")
    parser.add_argument("output")
    parser.add_argument("--hbm-energy-pj-per-byte", type=float, required=True)
    parser.add_argument("--hbm-energy-source", required=True)
    parser.add_argument("--artifact-catalog", required=True)
    parser.add_argument(
        "--structural-area-coefficients",
        default=str(DEFAULT_AREA_COEFFICIENTS),
    )
    parser.add_argument(
        "--sram-macro-table",
        default=str(DEFAULT_SRAM_MACROS),
    )
    parser.add_argument(
        "--structural-area-input",
        action="append",
        default=None,
    )
    args = parser.parse_args()
    artifact = fit(
        args.measurements,
        hbm_energy_j_per_byte=args.hbm_energy_pj_per_byte * 1e-12,
        hbm_energy_source=args.hbm_energy_source,
        artifact_catalog=args.artifact_catalog,
        structural_area_coefficients=args.structural_area_coefficients,
        sram_macro_table=args.sram_macro_table,
        structural_area_inputs=tuple(
            args.structural_area_input or DEFAULT_AREA_INPUTS
        ),
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(target)


if __name__ == "__main__":
    main()
