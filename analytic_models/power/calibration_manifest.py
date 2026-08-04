"""Generate the complete DC/SAIF calibration schedule for power fitting."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

CALIBRATION_MANIFEST_SCHEMA = "plena-power-calibration-manifest"
MX_BLOCK_SIZE = 8
HARDWARE_FP_BINDING = "FP_E6M5"
SELECTOR_SIGNATURE = "SELECTOR:PACKED_KV"
MXINT_W = ("MXINT4", "MXINT8")
MXINT_A_KV = ("MXINT2", "MXINT4", "MXINT8")
MXFP = ("MXFP_E1M2", "MXFP_E2M1", "MXFP_E4M3", "MXFP_E5M2")
VECTOR_FP = (
    "FP_E3M2",
    "FP_E2M3",
    "FP_E6M5",
    "FP_E5M6",
    "FP_E4M7",
    "FP_E8M5",
    "BF16",
)
TRAIN_GEOMETRIES = ((16, 4), (16, 8), (32, 4), (32, 8))
HOLDOUT_GEOMETRIES = ((64, 8), (64, 16))
TRACE_HOLDOUTS = (
    ("cycle", "SCALED_QWEN_LAYER:APPEND1", (16, 4)),
    ("cycle", "SCALED_QWEN_LAYER:APPEND2", (32, 8)),
    ("latency", "QWEN3_32B:BATCH1", (64, 8)),
    ("latency", "QWEN3_32B:BATCH8", (64, 16)),
)
MEASUREMENT_COLUMNS = (
    "status",
    "point_id",
    "split",
    "component",
    "signature",
    "MLEN",
    "BLEN",
    "selector_enabled",
    "events",
    "cycles",
    "clock_ns",
    "dynamic_power_w",
    "leakage_power_w",
    "area_mm2",
    "rtl_cycles",
    "emulator_cycles",
    "measured_latency_s",
    "analytical_latency_s",
    "dc_tool_version",
    "library_id",
    "process_corner",
    "MX_BLOCK_SIZE",
    "hardware_fp_binding",
    "activity_class",
    "saif_sha256",
    "decode_trace_sha256",
    "saif_source_id",
    "activity_generator",
)


@dataclass(frozen=True)
class CalibrationPoint:
    point_id: str
    component: str
    signature: str
    mlen: int
    blen: int
    split: str
    selector_enabled: bool
    activity_class: str | None = None
    requires_saif: bool = False
    mx_block_size: int = MX_BLOCK_SIZE
    hardware_fp_binding: str = HARDWARE_FP_BINDING
    clock_ns: float = 1.0
    status: str = "scheduled"


def _point(
    component: str,
    signature: str,
    geometry: tuple[int, int],
    split: str,
    selector_enabled: bool = False,
    activity_class: str | None = None,
) -> CalibrationPoint:
    key = {
        "component": component,
        "signature": signature,
        "mlen": geometry[0],
        "blen": geometry[1],
        "split": split,
        "selector_enabled": selector_enabled,
        "activity_class": activity_class,
        "requires_saif": activity_class is not None,
        "mx_block_size": MX_BLOCK_SIZE,
        "hardware_fp_binding": HARDWARE_FP_BINDING,
        "clock_ns": 1.0,
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return CalibrationPoint(
        point_id=f"pwr-{digest}",
        component=component,
        signature=signature,
        mlen=geometry[0],
        blen=geometry[1],
        split=split,
        selector_enabled=selector_enabled,
        activity_class=activity_class,
        requires_saif=activity_class is not None,
        mx_block_size=MX_BLOCK_SIZE,
        hardware_fp_binding=HARDWARE_FP_BINDING,
    )


def operand_signatures() -> tuple[str, ...]:
    linear = [f"LINEAR:{w}x{a}" for w, a in itertools.product(MXINT_W, MXINT_A_KV)]
    qk = [f"QK:{kv}x{a}" for kv, a in itertools.product(MXINT_A_KV, MXINT_A_KV)]
    pv = [f"PV:{kv}x{a}" for kv, a in itertools.product(MXINT_A_KV, MXINT_A_KV)]
    linear += [f"LINEAR:{w}x{a}" for w, a in itertools.product(MXFP, MXFP)]
    qk += [f"QK:{kv}x{a}" for kv, a in itertools.product(MXFP, MXFP)]
    pv += [f"PV:{kv}x{a}" for kv, a in itertools.product(MXFP, MXFP)]
    return tuple(sorted(set(linear + qk + pv)))


def event_signatures() -> tuple[str, ...]:
    return tuple(
        sorted(
            set(operand_signatures())
            | {f"VECTOR:{vector_format}" for vector_format in VECTOR_FP}
            | {SELECTOR_SIGNATURE}
        )
    )


def build_manifest() -> list[CalibrationPoint]:
    points: list[CalibrationPoint] = []
    for split, geometries in (("train", TRAIN_GEOMETRIES), ("holdout", HOLDOUT_GEOMETRIES)):
        for signature, geometry in itertools.product(operand_signatures(), geometries):
            operation = signature.split(":", 1)[0].casefold()
            points.append(
                _point(
                    "array",
                    signature,
                    geometry,
                    split,
                    activity_class=f"qwen3_32b_decode_q1_{operation}",
                )
            )
        for fp, geometry in itertools.product(VECTOR_FP, geometries):
            points.append(
                _point(
                    "vector",
                    f"VECTOR:{fp}",
                    geometry,
                    split,
                    activity_class="qwen3_32b_decode_q1_vector",
                )
            )
        for geometry in geometries:
            points.append(_point("fixed", "FIXED", geometry, split))
        for geometry in geometries:
            points.append(
                _point(
                    "chip_leakage",
                    "CHIP_LEAKAGE",
                    geometry,
                    split,
                    selector_enabled=True,
                )
            )
    for enabled in (False, True):
        for geometry in TRAIN_GEOMETRIES + HOLDOUT_GEOMETRIES:
            points.append(
                _point(
                    "selector",
                    "PACKED_KV_SELECTOR",
                    geometry,
                    "holdout" if geometry in HOLDOUT_GEOMETRIES else "train",
                    selector_enabled=enabled,
                    activity_class="qwen3_32b_decode_q1_packedkv_selector",
                )
            )
    for component, signature, geometry in TRACE_HOLDOUTS:
        points.append(_point(component, signature, geometry, "holdout"))
    ids = [point.point_id for point in points]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate calibration point IDs")
    return sorted(points, key=lambda point: point.point_id)


def manifest_payload() -> dict:
    """Return the canonical immutable calibration schedule."""

    return {
        "schema": CALIBRATION_MANIFEST_SCHEMA,
        "clock_ns": 1.0,
        "mx_block_size": MX_BLOCK_SIZE,
        "hardware_fp_binding": HARDWARE_FP_BINDING,
        "points": [asdict(point) for point in build_manifest()],
    }


def manifest_hash() -> str:
    payload = json.dumps(
        manifest_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_manifest(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_payload()
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return target


def write_measurement_template(path: str | Path) -> Path:
    """Create the canonical one-row-per-point measurement table."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEASUREMENT_COLUMNS)
        writer.writeheader()
        for point in build_manifest():
            writer.writerow(
                {
                    "status": point.status,
                    "point_id": point.point_id,
                    "split": point.split,
                    "component": point.component,
                    "signature": point.signature,
                    "MLEN": point.mlen,
                    "BLEN": point.blen,
                    "selector_enabled": point.selector_enabled,
                    "clock_ns": point.clock_ns,
                    "MX_BLOCK_SIZE": point.mx_block_size,
                    "hardware_fp_binding": point.hardware_fp_binding,
                    "activity_class": point.activity_class or "",
                }
            )
    temporary.replace(target)
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--csv-template")
    args = parser.parse_args()
    print(write_manifest(args.output))
    if args.csv_template:
        print(write_measurement_template(args.csv_template))
