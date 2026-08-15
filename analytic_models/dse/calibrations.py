"""Versioned calibration identities required by the formal DSE profile."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
RTL_V6_AREA_CALIBRATION = (
    REPO_ROOT
    / "analytic_models/area_new/calibration/vector_rtl_v6_delta_coefficients.json"
)
RTL_V6_POWER_CALIBRATION = (
    REPO_ROOT
    / "analytic_models/power/calibration/vector_rtl_v6_power_delta.json"
)
RTL_V6_AREA_SCHEMA = "vector_rtl_v6_delta_v3"
RTL_V6_AREA_STATUS = "fitted_from_paired_rtl_v6_dc"
RTL_V6_POWER_SCHEMA = "vector_rtl_v6_action_energy_v2"
RTL_V6_POWER_STATUS = "rtl_activity_calibrated_rtl_v6_logic"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required DSE calibration is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"DSE calibration must contain a JSON object: {path}")
    return payload


def _false_checks(value: Any, prefix: str = "") -> tuple[str, ...]:
    failures: list[str] = []
    if isinstance(value, Mapping):
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            failures.extend(_false_checks(child, child_prefix))
    elif isinstance(value, bool) and not value:
        failures.append(prefix)
    return tuple(failures)


@dataclass(frozen=True)
class DSECalibrationManifest:
    """Immutable artifact identity used by study compatibility checks."""

    area_path: Path
    area_sha256: str
    area_schema: str
    area_status: str
    area_validation: Mapping[str, Any]
    power_path: Path
    power_sha256: str
    power_schema: str
    power_status: str
    power_validation: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            {
                "area_sha256": self.area_sha256,
                "area_schema": self.area_schema,
                "area_status": self.area_status,
                "power_sha256": self.power_sha256,
                "power_schema": self.power_schema,
                "power_status": self.power_status,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "dse_calibration_manifest_v1",
            "fingerprint": self.fingerprint,
            "rtl_v6_area": {
                "path": str(self.area_path),
                "sha256": self.area_sha256,
                "schema": self.area_schema,
                "status": self.area_status,
                "validation": dict(self.area_validation),
            },
            "rtl_v6_power": {
                "path": str(self.power_path),
                "sha256": self.power_sha256,
                "schema": self.power_schema,
                "status": self.power_status,
                "validation": dict(self.power_validation),
            },
        }


def load_dse_calibration_manifest() -> DSECalibrationManifest:
    """Load and validate the promoted rtl-v6 area and power artifacts."""

    area_path = Path(
        os.environ.get(
            "PLENA_AREA_NEW_VECTOR_RTL_V6_DELTA",
            RTL_V6_AREA_CALIBRATION,
        )
    ).expanduser().resolve()
    power_path = Path(
        os.environ.get(
            "PLENA_POWER_VECTOR_RTL_V6_DELTA",
            RTL_V6_POWER_CALIBRATION,
        )
    ).expanduser().resolve()
    area = _load(area_path)
    power = _load(power_path)
    area_metadata = dict(area.get("metadata") or {})
    area_checks = dict(area_metadata.get("checks") or {})
    area_failures = list(area_metadata.get("failures") or ())
    area_failures.extend(_false_checks(area_checks))
    power_failures = list(power.get("failures") or ())

    observed = {
        "area schema": (area.get("schema_version"), RTL_V6_AREA_SCHEMA),
        "area status": (area_metadata.get("status"), RTL_V6_AREA_STATUS),
        "power schema": (power.get("schema_version"), RTL_V6_POWER_SCHEMA),
        "power status": (power.get("calibration_status"), RTL_V6_POWER_STATUS),
    }
    mismatches = [
        f"{name}={actual!r}, expected {expected!r}"
        for name, (actual, expected) in observed.items()
        if actual != expected
    ]
    if area_failures:
        mismatches.append(f"area validation failures={area_failures!r}")
    if power_failures:
        mismatches.append(f"power validation failures={power_failures!r}")
    if mismatches:
        raise ValueError(
            "formal DSE requires promoted rtl-v6 calibrations: "
            + "; ".join(mismatches)
        )

    return DSECalibrationManifest(
        area_path=area_path,
        area_sha256=_sha256(area_path),
        area_schema=str(area["schema_version"]),
        area_status=str(area_metadata["status"]),
        area_validation=area_checks,
        power_path=power_path,
        power_sha256=_sha256(power_path),
        power_schema=str(power["schema_version"]),
        power_status=str(power["calibration_status"]),
        power_validation=dict(power.get("validation") or {}),
    )


__all__ = [
    "DSECalibrationManifest",
    "RTL_V6_AREA_CALIBRATION",
    "RTL_V6_AREA_SCHEMA",
    "RTL_V6_AREA_STATUS",
    "RTL_V6_POWER_CALIBRATION",
    "RTL_V6_POWER_SCHEMA",
    "RTL_V6_POWER_STATUS",
    "load_dse_calibration_manifest",
]
