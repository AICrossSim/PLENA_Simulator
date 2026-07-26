"""Literature-parameterized external-memory energy estimates.

The external model consumes the physical 64-B DMA traffic reported by the HBM
service model.  It intentionally remains separate from the calibrated on-chip
HBM-controller action energy so each measurement boundary stays visible.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


POWER_DIR = Path(__file__).resolve().parent
DEFAULT_HBM3E_ENERGY = POWER_DIR / "calibration/external_memory_hbm3e_v1.json"
DECIMAL_GB = 1_000_000_000


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"expected mapping/dataclass report, got {type(value).__name__}")


@lru_cache(maxsize=8)
def _read_artifact(selected: str) -> dict[str, Any]:
    payload = json.loads(Path(selected).read_text())
    if payload.get("model") != "external_memory_hbm3e_v1":
        raise ValueError(
            f"unsupported external-memory artifact {payload.get('model')!r}"
        )
    return payload


def _load_artifact(path: str | Path | None) -> dict[str, Any]:
    selected = Path(path or DEFAULT_HBM3E_ENERGY).resolve()
    return _read_artifact(str(selected))


def _runtime_ns(timing: Mapping[str, Any], explicit: float | None) -> tuple[float, str]:
    if explicit is not None:
        if explicit <= 0:
            raise ValueError(f"runtime_ns must be positive, got {explicit}")
        return float(explicit), "explicit"
    # The DSE's fast system estimate uses the stage-wise roofline.  Scheduled
    # replay remains a validation shadow and must not silently change power.
    for name in (
        "roofline_latency_ns",
        "true_full_model_latency_ns",
        "serial_latency_ns",
        "scheduled_shadow_latency_ns",
        "compute_latency_ns",
    ):
        value = timing.get(name)
        if value is not None and float(value) > 0:
            return float(value), name
    raise ValueError("timing report has no positive system runtime")


def _config_number(
    config: Mapping[str, Any],
    names: tuple[str, ...],
    *,
    required: bool = True,
) -> float | None:
    for name in names:
        if name in config:
            return float(config[name])
    if required:
        raise ValueError(f"memory config is missing one of {names}")
    return None


def _dynamic_energy_mj(
    read_bytes: float,
    write_bytes: float,
    artifact: Mapping[str, Any],
) -> tuple[float, float]:
    read_pj = (
        read_bytes
        * 8
        * float(artifact["coefficients"]["read_energy_pj_per_bit"])
    )
    write_pj = (
        write_bytes
        * 8
        * float(artifact["coefficients"]["write_energy_pj_per_bit"])
    )
    return read_pj * 1e-9, write_pj * 1e-9


def _energy_breakdown(
    traffic: Mapping[str, Any], artifact: Mapping[str, Any]
) -> dict[str, dict[str, dict[str, float | int]]]:
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for group in ("by_role", "by_stage", "by_opcode"):
        result[group] = {}
        for key, raw_bucket in dict(traffic.get(group, {})).items():
            bucket = dict(raw_bucket)
            read_bytes = float(bucket.get("physical_read_bytes", 0))
            write_bytes = float(bucket.get("physical_write_bytes", 0))
            read_mj, write_mj = _dynamic_energy_mj(
                read_bytes, write_bytes, artifact
            )
            result[group][str(key)] = {
                **bucket,
                "read_energy_mj": read_mj,
                "write_energy_mj": write_mj,
                "dynamic_energy_mj": read_mj + write_mj,
            }
    return result


def _validate_traffic_breakdown(
    traffic: Mapping[str, Any], read_bytes: float, write_bytes: float
) -> None:
    for group in ("by_role", "by_stage", "by_opcode"):
        buckets = dict(traffic.get(group, {}))
        if not buckets:
            raise ValueError(f"HBM traffic breakdown is missing {group}")
        grouped_read = math.fsum(
            float(dict(bucket).get("physical_read_bytes", 0))
            for bucket in buckets.values()
        )
        grouped_write = math.fsum(
            float(dict(bucket).get("physical_write_bytes", 0))
            for bucket in buckets.values()
        )
        if not (
            math.isclose(
                grouped_read,
                read_bytes,
                rel_tol=1e-12,
                abs_tol=1e-6,
            )
            and math.isclose(
                grouped_write,
                write_bytes,
                rel_tol=1e-12,
                abs_tol=1e-6,
            )
        ):
            raise ValueError(
                f"HBM {group} traffic does not sum to physical totals: "
                f"breakdown=({grouped_read}, {grouped_write}), "
                f"total=({read_bytes}, {write_bytes})"
            )


def estimate_external_hbm_power(
    memory_config: Mapping[str, Any],
    timing_report: Any,
    *,
    artifact_path: str | Path | None = None,
    runtime_ns: float | None = None,
) -> dict[str, Any]:
    """Estimate HBM3E-equivalent background and physical-transfer energy."""

    config = dict(memory_config)
    timing = _mapping(timing_report)
    artifact = _load_artifact(artifact_path)
    capacity_bytes = _config_number(
        config, ("HBM_CAPACITY_BYTES", "hbm_capacity_bytes")
    )
    bandwidth_gbps = _config_number(
        config, ("HBM_BANDWIDTH_GBPS", "hbm_bandwidth_gbps")
    )
    assert capacity_bytes is not None and bandwidth_gbps is not None
    if capacity_bytes <= 0 or bandwidth_gbps <= 0:
        raise ValueError("HBM capacity and bandwidth must be positive")

    read_bytes = float(timing.get("hbm_read_bytes", 0))
    write_bytes = float(timing.get("hbm_write_bytes", 0))
    payload_read_bytes = float(timing.get("hbm_payload_read_bytes", 0))
    payload_write_bytes = float(timing.get("hbm_payload_write_bytes", 0))
    if min(read_bytes, write_bytes, payload_read_bytes, payload_write_bytes) < 0:
        raise ValueError("HBM traffic counters must be nonnegative")

    effective_runtime_ns, runtime_source = _runtime_ns(timing, runtime_ns)
    runtime_ms = effective_runtime_ns / 1e6
    capacity_gb = capacity_bytes / DECIMAL_GB
    background = artifact["coefficients"]["background_power_mw_per_gb"]
    background_power_w = {
        quantile: capacity_gb * float(background[quantile]) / 1000.0
        for quantile in ("p10", "p50", "p90")
    }
    background_energy_mj = {
        quantile: power_w * runtime_ms
        for quantile, power_w in background_power_w.items()
    }
    read_energy_mj, write_energy_mj = _dynamic_energy_mj(
        read_bytes, write_bytes, artifact
    )
    dynamic_energy_mj = read_energy_mj + write_energy_mj
    energy_mj = {
        quantile: background_energy_mj[quantile] + dynamic_energy_mj
        for quantile in ("p10", "p50", "p90")
    }
    total_physical_bytes = read_bytes + write_bytes
    total_payload_bytes = payload_read_bytes + payload_write_bytes
    achieved_bandwidth_gbps = total_physical_bytes / effective_runtime_ns
    traffic = dict(timing.get("hbm_traffic_breakdown") or {})
    if traffic:
        _validate_traffic_breakdown(traffic, read_bytes, write_bytes)

    warnings = list(artifact.get("warnings", ()))
    if capacity_bytes != 80_000_000_000 or bandwidth_gbps != 2039.0:
        warnings.append(
            "memory configuration differs from the default abstract "
            "80 GB / 2039 GB/s A100-aligned comparison"
        )
    if not traffic:
        warnings.append(
            "HBM traffic role/stage breakdown is unavailable; total physical "
            "read/write energy remains valid"
        )

    return {
        "external_memory_model": artifact["model"],
        "external_memory_technology": artifact["technology"],
        "external_memory_configuration_semantics": str(
            config.get(
                "HBM_CONFIGURATION_SEMANTICS",
                "abstract_80gb_a100_aligned",
            )
        ),
        "external_memory_calibration_status": artifact["calibration_status"],
        "external_memory_source": dict(artifact["source"]),
        "provisioned_capacity_bytes": int(capacity_bytes),
        "provisioned_capacity_gb_decimal": capacity_gb,
        "configured_bandwidth_gbps": bandwidth_gbps,
        "runtime_ns": effective_runtime_ns,
        "runtime_source": runtime_source,
        "physical_read_bytes": read_bytes,
        "physical_write_bytes": write_bytes,
        "payload_read_bytes": payload_read_bytes,
        "payload_write_bytes": payload_write_bytes,
        "physical_to_payload_traffic_ratio": (
            None
            if total_payload_bytes == 0
            else total_physical_bytes / total_payload_bytes
        ),
        "achieved_average_bandwidth_gbps": achieved_bandwidth_gbps,
        "bandwidth_utilization": achieved_bandwidth_gbps / bandwidth_gbps,
        "hbm_background_power_w": background_power_w["p50"],
        "hbm_background_power_p10_w": background_power_w["p10"],
        "hbm_background_power_p50_w": background_power_w["p50"],
        "hbm_background_power_p90_w": background_power_w["p90"],
        "hbm_background_energy_mj": background_energy_mj["p50"],
        "hbm_read_energy_mj": read_energy_mj,
        "hbm_write_energy_mj": write_energy_mj,
        "hbm_dynamic_energy_mj": dynamic_energy_mj,
        "external_hbm_energy_mj": energy_mj["p50"],
        "external_hbm_energy_p10_mj": energy_mj["p10"],
        "external_hbm_energy_p50_mj": energy_mj["p50"],
        "external_hbm_energy_p90_mj": energy_mj["p90"],
        "external_hbm_average_power_w": energy_mj["p50"] / runtime_ms,
        "external_hbm_average_power_p10_w": energy_mj["p10"] / runtime_ms,
        "external_hbm_average_power_p50_w": energy_mj["p50"] / runtime_ms,
        "external_hbm_average_power_p90_w": energy_mj["p90"] / runtime_ms,
        "external_hbm_dynamic_energy_breakdown": _energy_breakdown(
            traffic, artifact
        ),
        "uncertainty_semantics": (
            "literature background-power envelope; read/write pJ-per-bit fixed"
        ),
        "warnings": list(dict.fromkeys(warnings)),
        "boundary_uncertainties": list(
            artifact.get("boundary_uncertainties", ())
        ),
    }


__all__ = ["estimate_external_hbm_power"]
