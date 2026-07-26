"""Fixed paired-DC area delta for the six-stream loop AGU.

The AGU is small enough to synthesize independently.  Its area proxy combines
the standalone affine-offset sidecar with the measured frontend loop-controller
delta.  This avoids fitting the AGU against unrelated full-chip logic and keeps
the legacy address-generation path available for controlled A/B comparisons.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CALIBRATION_DIR = Path(__file__).with_name("calibration")
V1_ARTIFACT_PATH = CALIBRATION_DIR / "agu_area_delta_v1.json"
DEFAULT_ARTIFACT_PATH = V1_ARTIFACT_PATH
MODEL_VERSION = "loop_agu_paired_delta_v1"
SUPPORTED_MODEL_VERSIONS = {"loop_agu_paired_delta_v1"}


def artifact_path(
    explicit_path: str | Path | None = None,
    *,
    mode: str = "loop-agu-v1",
) -> Path:
    """Resolve an explicit, environment, or committed AGU area artifact."""
    path = explicit_path or os.environ.get("PLENA_AREA_NEW_AGU_DELTA")
    if path:
        return Path(path)
    if mode != "loop-agu-v1":
        raise ValueError(f"AGU area artifacts are unavailable for mode={mode!r}")
    return V1_ARTIFACT_PATH


def load_artifact(
    explicit_path: str | Path | None = None,
    *,
    mode: str = "loop-agu-v1",
) -> dict[str, Any]:
    """Load and validate the paired-DC AGU area artifact."""
    path = artifact_path(explicit_path, mode=mode)
    raw = json.loads(path.read_text())
    if raw.get("model_version") not in SUPPORTED_MODEL_VERSIONS:
        raise ValueError(
            f"unsupported AGU area model {raw.get('model_version')!r}; "
            f"expected one of {sorted(SUPPORTED_MODEL_VERSIONS)!r}"
        )
    return raw


def estimate_address_generation_unit_area(
    config: dict[str, Any],
    *,
    artifact: dict[str, Any] | None = None,
    artifact_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Estimate the AGU delta in square micrometres.

    The mapped point uses a 32-bit GP datapath, six affine streams, and four
    nested loop frames.  There is not enough paired data to infer a trustworthy
    width-scaling law, so out-of-domain widths retain the measured nominal area
    and are reported explicitly as structural extrapolations.
    """
    mode = str(
        config.get(
            "address_generation_mode",
            config.get("ADDRESS_GENERATION_MODE", "loop-agu-v1"),
        )
    ).strip().lower()
    if mode == "legacy":
        return {
            "area": 0.0,
            "area_proxy": 0.0,
            "area_model": MODEL_VERSION,
            "enabled": False,
            "calibration_status": "disabled_legacy_address_generation",
            "calibration_in_domain": True,
            "calibration_warnings": [],
            "breakdown": {
                "AguAffineSidecar": 0.0,
                "AguFrontendDelta": 0.0,
            },
        }
    if mode != "loop-agu-v1":
        raise ValueError(f"unsupported address_generation_mode={mode!r}")

    data = artifact or load_artifact(artifact_path_override, mode=mode)
    model_version = str(data["model_version"])
    measurements = data["measurements"]
    sidecar = float(measurements["loop_agu_state"]["area_um2"])
    frontend = float(measurements["loop_controller"]["area_delta_um2"])
    area = sidecar + frontend
    int_width = int(config.get("INT_DATA_WIDTH", 32))
    width_in_domain = int_width == int(data["calibration_domain"]["INT_DATA_WIDTH"])
    warnings: list[str] = []
    if not width_in_domain:
        warnings.append(
            f"AGU INT_DATA_WIDTH={int_width} is outside the paired DC point at 32 bits; "
            "the fixed 32-bit nominal delta is retained"
        )
    minimum_slack = min(
        float(measurements["loop_agu_state"]["wns_ps"]),
        float(measurements["loop_controller"]["current_wns_ps"]),
    )
    if minimum_slack < 10.0:
        warnings.append(
            "AGU mapped DC paths meet the 1 ns constraint with less than 10 ps margin"
        )
    return {
        "area": area,
        "area_proxy": area,
        "area_model": model_version,
        "enabled": True,
        "calibration_status": (
            "mapped_dc_paired_delta"
            if width_in_domain
            else "mapped_dc_width_extrapolation"
        ),
        "calibration_in_domain": width_in_domain,
        "calibration_warnings": warnings,
        "breakdown": {
            "AguAffineSidecar": sidecar,
            "AguFrontendDelta": frontend,
        },
        "inputs": {
            "address_generation_mode": mode,
            "INT_DATA_WIDTH": int_width,
            "stream_count": int(data["calibration_domain"]["stream_count"]),
            "loop_depth": int(data["calibration_domain"]["loop_depth"]),
        },
        "timing": {
            "clock_period_ps": float(data["corner"]["clock_period_ps"]),
            "minimum_wns_ps": minimum_slack,
            "timing_status": "met_with_low_margin",
        },
        "artifact": str(artifact_path(artifact_path_override, mode=mode)),
    }
