"""Unified on-chip plus external-HBM power report."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .external_memory import estimate_external_hbm_power
from .power_model import estimate_onchip_power


def estimate_system_power(
    config: Mapping[str, Any],
    cost_trace: Any,
    timing_report: Any,
    *,
    external_memory_config: Mapping[str, Any] | None = None,
    logic_coefficients_path: str | Path | None = None,
    sram_energy_path: str | Path | None = None,
    sram_background_path: str | Path | None = None,
    external_memory_artifact_path: str | Path | None = None,
    clock_gating_mode: str = "ideal_hierarchical",
) -> dict[str, Any]:
    """Estimate PLENA on-chip and HBM3E-equivalent system energy."""

    memory = dict(config)
    if external_memory_config is not None:
        memory.update(external_memory_config)
    external = estimate_external_hbm_power(
        memory,
        timing_report,
        artifact_path=external_memory_artifact_path,
    )
    runtime_ns = float(external["runtime_ns"])
    onchip = estimate_onchip_power(
        config,
        cost_trace,
        timing_report,
        logic_coefficients_path=logic_coefficients_path,
        sram_energy_path=sram_energy_path,
        sram_background_path=sram_background_path,
        makespan_ns_override=runtime_ns,
        makespan_source_override=f"system:{external['runtime_source']}",
        clock_gating_mode=clock_gating_mode,
    )
    runtime_ms = runtime_ns / 1e6
    system_energy = {
        quantile: float(onchip[f"onchip_energy_{quantile}_mj"])
        + float(external[f"external_hbm_energy_{quantile}_mj"])
        for quantile in ("p10", "p50", "p90")
    }
    ungated_system_energy = {
        quantile: float(onchip[f"ungated_onchip_energy_{quantile}_mj"])
        + float(external[f"external_hbm_energy_{quantile}_mj"])
        for quantile in ("p10", "p50", "p90")
    }
    input_tokens = max(
        1,
        int(config.get("INPUT_TOKENS", 0))
        or int(config.get("SEQ_LEN", 1)) * int(config.get("BATCH_SIZE", 1)),
    )
    warnings = list(onchip.get("warnings", ())) + list(
        external.get("warnings", ())
    )
    excludes = [
        "package",
        "cooling",
        "board_regulator",
        "kv_link",
        "cts",
        "asap7_macro_intrinsic_leakage_calibration",
    ]
    result = dict(onchip)
    result.update(
        {
            "power_model": "plena_system_power_hbm3e_v1",
            "onchip_power_model": onchip["power_model"],
            "power_scope": (
                "onchip_logic+sram+controller+external_hbm3e_equivalent"
            ),
            "calibration_status": (
                "mixed_rtl_activity_onchip_literature_sram_and_hbm3e"
            ),
            "onchip_calibration_status": onchip["calibration_status"],
            "external_memory": external,
            "external_memory_model": external["external_memory_model"],
            "external_memory_calibration_status": external[
                "external_memory_calibration_status"
            ],
            "external_memory_configuration_semantics": external[
                "external_memory_configuration_semantics"
            ],
            "external_hbm_capacity_bytes": external[
                "provisioned_capacity_bytes"
            ],
            "external_hbm_configured_bandwidth_gbps": external[
                "configured_bandwidth_gbps"
            ],
            "hbm_background_energy_mj": external[
                "hbm_background_energy_mj"
            ],
            "hbm_read_energy_mj": external["hbm_read_energy_mj"],
            "hbm_write_energy_mj": external["hbm_write_energy_mj"],
            "external_hbm_energy_mj": external["external_hbm_energy_mj"],
            "external_hbm_energy_p10_mj": external[
                "external_hbm_energy_p10_mj"
            ],
            "external_hbm_energy_p50_mj": external[
                "external_hbm_energy_p50_mj"
            ],
            "external_hbm_energy_p90_mj": external[
                "external_hbm_energy_p90_mj"
            ],
            "external_hbm_average_power_w": external[
                "external_hbm_average_power_w"
            ],
            "external_hbm_average_power_p10_w": external[
                "external_hbm_average_power_p10_w"
            ],
            "external_hbm_average_power_p50_w": external[
                "external_hbm_average_power_p50_w"
            ],
            "external_hbm_average_power_p90_w": external[
                "external_hbm_average_power_p90_w"
            ],
            "hbm_physical_read_bytes": external["physical_read_bytes"],
            "hbm_physical_write_bytes": external["physical_write_bytes"],
            "hbm_payload_read_bytes": external["payload_read_bytes"],
            "hbm_payload_write_bytes": external["payload_write_bytes"],
            "physical_to_payload_traffic_ratio": external[
                "physical_to_payload_traffic_ratio"
            ],
            "achieved_average_bandwidth_gbps": external[
                "achieved_average_bandwidth_gbps"
            ],
            "bandwidth_utilization": external["bandwidth_utilization"],
            "external_hbm_energy_by_role": external[
                "external_hbm_dynamic_energy_breakdown"
            ].get("by_role", {}),
            "external_hbm_energy_by_stage": external[
                "external_hbm_dynamic_energy_breakdown"
            ].get("by_stage", {}),
            "external_hbm_energy_by_opcode": external[
                "external_hbm_dynamic_energy_breakdown"
            ].get("by_opcode", {}),
            "system_energy_mj": system_energy["p50"],
            "system_energy_p10_mj": system_energy["p10"],
            "system_energy_p50_mj": system_energy["p50"],
            "system_energy_p90_mj": system_energy["p90"],
            "system_average_power_w": system_energy["p50"] / runtime_ms,
            "system_average_power_p10_w": system_energy["p10"] / runtime_ms,
            "system_average_power_p50_w": system_energy["p50"] / runtime_ms,
            "system_average_power_p90_w": system_energy["p90"] / runtime_ms,
            "system_energy_per_input_token_mj": (
                system_energy["p50"] / input_tokens
            ),
            "ungated_system_energy_mj": ungated_system_energy["p50"],
            "ungated_system_energy_p10_mj": ungated_system_energy["p10"],
            "ungated_system_energy_p50_mj": ungated_system_energy["p50"],
            "ungated_system_energy_p90_mj": ungated_system_energy["p90"],
            "ungated_system_average_power_w": (
                ungated_system_energy["p50"] / runtime_ms
            ),
            "ungated_system_average_power_p10_w": (
                ungated_system_energy["p10"] / runtime_ms
            ),
            "ungated_system_average_power_p50_w": (
                ungated_system_energy["p50"] / runtime_ms
            ),
            "ungated_system_average_power_p90_w": (
                ungated_system_energy["p90"] / runtime_ms
            ),
            "ungated_system_energy_per_input_token_mj": (
                ungated_system_energy["p50"] / input_tokens
            ),
            "system_uncertainty_energy_mj": system_energy,
            "system_uncertainty_power_w": {
                key: value / runtime_ms for key, value in system_energy.items()
            },
            "system_runtime_ns": runtime_ns,
            "system_runtime_source": external["runtime_source"],
            "clock_gating_mode": onchip["clock_gating_mode"],
            "clock_gating_status": onchip["clock_gating_status"],
            "idle_clock_fraction": onchip["idle_clock_fraction"],
            "gating_overhead_included": False,
            "rtl_clock_gating_implemented": False,
            "warnings": list(dict.fromkeys(warnings)),
            "excludes": excludes,
            "boundary_uncertainties": external[
                "boundary_uncertainties"
            ],
        }
    )
    return result


__all__ = ["estimate_system_power"]
