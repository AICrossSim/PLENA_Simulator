"""Fixed-corner on-chip action-energy estimator for PLENA.

Dynamic logic is evaluated from compiler-emitted hardware actions. SRAM dynamic
energy comes from the selected ASAP7 macro tiling and Liberty read/write tables.
Logic leakage is a pre-layout area-proportional reference. Because the public
ASAP7 SRAM macros report zero leakage, SRAM background power is represented by
an explicitly labelled literature coefficient applied to allocated macro bits.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

from analytic_models.area_new import estimate_area
from analytic_models.area_new.precision import derive_compute_sides
from analytic_models.area_new.sram_model import estimate_sram_area

from .clock_work import build_clock_work
from .sram_energy import load_sram_energy_catalog, macro_energy_lookup

POWER_DIR = Path(__file__).resolve().parent
DEFAULT_LOGIC_ENERGY = POWER_DIR / "calibration/logic_energy_v2.json"
DEFAULT_AGU_ENERGY = POWER_DIR / "calibration/agu_energy_v1.json"
DEFAULT_VECTOR_RTL_V4_ENERGY = (
    POWER_DIR / "calibration/vector_rtl_v4_power_delta.json"
)
DEFAULT_VECTOR_RTL_V5_ENERGY = (
    POWER_DIR / "calibration/vector_rtl_v5_power_delta.json"
)
DEFAULT_VECTOR_RTL_V6_ENERGY = (
    POWER_DIR / "calibration/vector_rtl_v6_power_delta.json"
)
DEFAULT_SRAM_BACKGROUND = (
    POWER_DIR / "calibration/sram_background_memexplorer_v1.json"
)

# These conservative architecture priors are only the fallback used when the
# selected artifact is missing. The checked-in default is the calibrated v2
# candidate; fallback values must not be presented as measurements.
DEFAULT_LOGIC_COEFFICIENTS: dict[str, Any] = {
    "model": "onchip_action_energy_v1",
    "calibration_status": "bootstrap_pending_activity_calibration",
    "corner": {"process": "ASAP7_TT", "voltage_v": 0.7, "temperature_c": 25.0, "clock_period_ps": 1000},
    "dynamic_pj": {
        "matrix.active_mac_bit_product": 0.0025,
        "matrix.matrix_vector_bit_product": 0.0025,
        "matrix.output_bit": 0.006,
        "vector.lane_add_sub_bit": 0.010,
        "vector.lane_multiply_bit2": 0.0012,
        "vector.lane_sfu_bit2": 0.0030,
        "vector.reduction_node_bit": 0.008,
        "vector.lane_movement_bit": 0.004,
        "scalar.fp_add_sub_move_bit": 0.018,
        "scalar.fp_multiply_bit2": 0.0020,
        "scalar.fp_sfu_bit2": 0.0060,
        "scalar.integer_alu_bit": 0.010,
        "scalar.register_access_bit": 0.004,
        "control.frontend_issue": 0.20,
        "hbm.dma_issue": 1.0,
        "hbm.line": 0.20,
        "hbm.byte": 0.003,
    },
    "clock_pj_per_cycle": {
        "matrix_pe": 0.00005,
        "vector_lane": 0.0010,
        "scalar_machine": 0.20,
        "control_frontend": 0.30,
        "hbm_controller": 0.30,
    },
    "logic_leakage_mw_per_um2": 8.0e-7,
    "uncertainty_relative": {"p10": 0.55, "p50": 1.0, "p90": 1.75},
    "provenance": {
        "dynamic": "architecture prior; replace with VCD-annotated DC slopes",
        "leakage": "order-of-magnitude reference from mapped ASAP7 reports",
    },
}


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"expected mapping/dataclass report, got {type(value).__name__}")


@lru_cache(maxsize=8)
def _read_logic_coefficients(selected: str) -> dict[str, Any]:
    path = Path(selected)
    if not path.exists():
        return json.loads(json.dumps(DEFAULT_LOGIC_COEFFICIENTS))
    payload = json.loads(path.read_text())
    if payload.get("model") not in {"onchip_action_energy_v1", "onchip_action_energy_v2"}:
        raise ValueError(f"unsupported logic energy artifact {payload.get('model')!r}")
    return payload


def _load_logic_coefficients(path: str | Path | None) -> dict[str, Any]:
    selected = path or os.environ.get("PLENA_POWER_LOGIC_ENERGY") or DEFAULT_LOGIC_ENERGY
    return _read_logic_coefficients(str(Path(selected).resolve()))


@lru_cache(maxsize=4)
def _read_sram_background(selected: str) -> dict[str, Any]:
    path = Path(selected)
    if not path.exists():
        raise FileNotFoundError(f"SRAM background artifact does not exist: {path}")
    payload = json.loads(path.read_text())
    if payload.get("model") != "sram_background_memexplorer_v1":
        raise ValueError(
            f"unsupported SRAM background model {payload.get('model')!r}"
        )
    if float(payload["selected_background_power_w_per_gb"]) < 0:
        raise ValueError("SRAM background power coefficient must be nonnegative")
    return payload


def _load_sram_background(path: str | Path | None) -> dict[str, Any]:
    selected = (
        path
        or os.environ.get("PLENA_POWER_SRAM_BACKGROUND")
        or DEFAULT_SRAM_BACKGROUND
    )
    return _read_sram_background(str(Path(selected).resolve()))


@lru_cache(maxsize=4)
def _read_vector_rtl_v4_energy(selected: str) -> dict[str, Any]:
    path = Path(selected)
    return json.loads(path.read_text()) if path.exists() else {}


def _load_vector_rtl_v4_energy() -> dict[str, Any]:
    selected = (
        os.environ.get("PLENA_POWER_VECTOR_RTL_V4_DELTA")
        or DEFAULT_VECTOR_RTL_V4_ENERGY
    )
    return _read_vector_rtl_v4_energy(str(Path(selected).resolve()))


@lru_cache(maxsize=4)
def _read_vector_rtl_v5_energy(selected: str) -> dict[str, Any]:
    path = Path(selected)
    return json.loads(path.read_text()) if path.exists() else {}


def _load_vector_rtl_v5_energy() -> dict[str, Any]:
    selected = (
        os.environ.get("PLENA_POWER_VECTOR_RTL_V5_DELTA")
        or DEFAULT_VECTOR_RTL_V5_ENERGY
    )
    return _read_vector_rtl_v5_energy(str(Path(selected).resolve()))


@lru_cache(maxsize=4)
def _read_vector_rtl_v6_energy(selected: str) -> dict[str, Any]:
    path = Path(selected)
    return json.loads(path.read_text()) if path.exists() else {}


def _load_vector_rtl_v6_energy() -> dict[str, Any]:
    selected = (
        os.environ.get("PLENA_POWER_VECTOR_RTL_V6_DELTA")
        or DEFAULT_VECTOR_RTL_V6_ENERGY
    )
    return _read_vector_rtl_v6_energy(str(Path(selected).resolve()))


def _rtl_v6_action_energy(
    action: Mapping[str, Any],
    config: Mapping[str, Any],
    fp_width: int,
    *,
    quantile: str,
) -> float | None:
    """Use the dedicated rtl-v6 logic artifact when it passed validation.

    SRAM access energy is intentionally absent from this overlay and remains
    charged once by ``_sram_dynamic_energy`` from the ASAP7 Liberty catalog.
    """

    artifact = _load_vector_rtl_v6_energy()
    if artifact.get("calibration_status") != "rtl_activity_calibrated_rtl_v6_logic":
        return None
    component = str(action.get("component"))
    family = str(action.get("action"))
    action_key = f"{component}.{family}"
    count = float(action.get("count", 0.0))
    fp_scale = fp_width / 12.0
    models = artifact.get("row_action_models", {})
    if action_key in models:
        model = models[action_key]
        rows = max(
            1,
            int(
                action.get("active_lanes")
                or config.get("SOFTMAX_ROW_LANES", 1)
            ),
        )
        units = rows
        if model.get("unit") == "active_elements":
            units *= int(config.get("VLEN", config["MLEN"]))
        nominal = count * (
            float(model["fixed_pj_per_command"])
            + units * float(model["variable_pj_per_unit"])
        ) * fp_scale
    else:
        blen = int(config["BLEN"])
        packed_models = artifact.get("packed_pv_action_models", {})
        if action_key in packed_models:
            model = packed_models[action_key]
            # One M_MM_WO_PACKED_ACC drains BLEN output rows.  Each row pays
            # one fixed FIFO/mux cost plus a calibrated cost for BLEN active
            # write lanes, hence BLEN * (fixed + BLEN * slope).
            nominal = count * blen * (
                float(model["fixed_pj_per_row"])
                + blen * float(model["variable_pj_per_active_lane"])
            ) * fp_scale
        else:
            # Compatibility with the v1 overlay, whose coefficient already
            # represented one active lane and therefore scales as BLEN^2.
            packed = artifact.get("packed_pv_pj_per_active_lane", {})
            if action_key not in packed:
                return None
            nominal = count * blen * blen * float(packed[action_key]) * fp_scale
    envelope = artifact.get("activity_envelope", {})
    return nominal * float(envelope.get(quantile, 1.0))


@lru_cache(maxsize=4)
def _load_agu_coefficients(selected: str = str(DEFAULT_AGU_ENERGY)) -> dict[str, Any]:
    path = Path(selected)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if payload.get("model_version") != "loop_agu_action_energy_v1":
        raise ValueError(f"unsupported AGU energy artifact {payload.get('model_version')!r}")
    return payload


def _fp_width(config: Mapping[str, Any]) -> int:
    if "FP_SETTING" in config:
        import re

        match = re.match(r"^FP_?E(\d+)M(\d+)$", str(config["FP_SETTING"]), re.I)
        if match:
            return 1 + int(match.group(1)) + int(match.group(2))
    return 1 + int(config.get("S_FP_EXP_WIDTH", config.get("FP_EXP_WIDTH", 5))) + int(
        config.get("S_FP_MANT_WIDTH", config.get("FP_MANT_WIDTH", 6))
    )


def _trace_actions(cost_trace: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(cost_trace, Mapping):
        trace = dict(cost_trace)
        actions = [dict(action) for action in trace.get("energy_actions", ())]
    elif hasattr(cost_trace, "energy_actions"):
        # Avoid CostTrace.to_dict(): it serializes the full compressed schedule
        # and can cost seconds plus hundreds of MiB on a production Qwen trace.
        actions = [
            (
                dict(vars(action))
                if hasattr(action, "__dict__")
                else action.to_dict()
                if hasattr(action, "to_dict")
                else asdict(action)
            )
            for action in cost_trace.energy_actions
        ]
        trace = {
            "schema_version": getattr(cost_trace, "schema_version", None),
            "metadata": getattr(cost_trace, "metadata", {}),
        }
    elif hasattr(cost_trace, "to_dict"):
        trace = cost_trace.to_dict()
        actions = [dict(action) for action in trace.get("energy_actions", ())]
    else:
        raise TypeError("cost_trace must be a CostTrace or mapping")
    return actions, trace


def _makespan(timing: Mapping[str, Any], clock_period_ps: int) -> tuple[int, float, str]:
    candidates = (
        ("compute_pipeline_makespan_cycles", timing.get("compute_pipeline_makespan_cycles")),
        ("scheduled_shadow_makespan_cycles", timing.get("scheduled_shadow_makespan_cycles")),
        ("compute_resource_work_cycles", timing.get("compute_resource_work_cycles")),
    )
    for source, value in candidates:
        if value is not None and int(value) > 0:
            cycles = int(value)
            return cycles, cycles * clock_period_ps / 1000.0, source
    for source in ("roofline_latency_ns", "compute_latency_ns", "latency_ns"):
        value = timing.get(source)
        if value is not None and float(value) > 0:
            latency_ns = float(value)
            cycles = max(1, math.ceil(latency_ns * 1000.0 / clock_period_ps))
            return cycles, latency_ns, source
    raise ValueError("timing_report has no positive makespan or latency")


def _logic_action_energy(
    action: Mapping[str, Any],
    config: Mapping[str, Any],
    coefficients: Mapping[str, float],
    widths: Mapping[str, int],
) -> float:
    component = str(action["component"])
    family = str(action["action"])
    count = float(action["count"])
    mlen = int(config["MLEN"])
    blen = int(config["BLEN"])
    vlen = int(config.get("VLEN", mlen))
    fp_width = widths["fp"]
    if component == "matrix":
        if family == "array_compute":
            return count * mlen * blen * blen * widths["t"] * widths["l"] * coefficients["matrix.active_mac_bit_product"]
        if family == "matrix_vector_compute":
            return count * mlen * blen * widths["t"] * widths["l"] * coefficients["matrix.matrix_vector_bit_product"]
        if family == "cross_k_reduce":
            splits = max(0, mlen // blen - 1)
            return count * blen * blen * splits * fp_width * coefficients["matrix.output_bit"]
        return count * blen * blen * fp_width * coefficients["matrix.output_bit"]
    if component == "vector":
        lanes = int(action.get("active_lanes") or vlen)
        if family.startswith("softmax_row_") or family.endswith("_rows"):
            lanes *= vlen
        if family.startswith("lane_add_sub"):
            return count * lanes * fp_width * coefficients["vector.lane_add_sub_bit"]
        if family.startswith("lane_multiply"):
            return count * lanes * fp_width * fp_width * coefficients["vector.lane_multiply_bit2"]
        if family.startswith("lane_sfu"):
            return count * lanes * fp_width * fp_width * coefficients["vector.lane_sfu_bit2"]
        if family.startswith("reduction"):
            return count * max(1, lanes - 1) * fp_width * coefficients["vector.reduction_node_bit"]
        return count * lanes * fp_width * coefficients["vector.lane_movement_bit"]
    if component == "softmax_state":
        rows = int(action.get("active_lanes") or 1)
        add = fp_width * coefficients["scalar.fp_add_sub_move_bit"]
        mul = fp_width * fp_width * coefficients["scalar.fp_multiply_bit2"]
        sfu = fp_width * fp_width * coefficients["scalar.fp_sfu_bit2"]
        per_row = {
            "max_update": 2 * add + sfu,
            "sum_update": mul + add,
            "final_reciprocal": sfu,
        }.get(family, add)
        return count * rows * per_row
    if component == "packed_pv_accumulator":
        active = blen * blen
        return (
            count
            * active
            * fp_width
            * coefficients["vector.lane_add_sub_bit"]
        )
    if component == "scalar":
        int_width = int(config.get("INT_DATA_WIDTH", 32))
        if family == "fp_add_sub_move":
            return count * fp_width * coefficients["scalar.fp_add_sub_move_bit"]
        if family == "fp_multiply":
            return count * fp_width * fp_width * coefficients["scalar.fp_multiply_bit2"]
        if family.startswith("fp_sfu"):
            return count * fp_width * fp_width * coefficients["scalar.fp_sfu_bit2"]
        if family in {"integer_alu", "integer_multiply"}:
            return count * int_width * coefficients["scalar.integer_alu_bit"]
        return count * max(fp_width, int_width) * coefficients["scalar.register_access_bit"]
    if component == "control":
        return count * coefficients["control.frontend_issue"]
    if component == "agu":
        return count * coefficients["control.frontend_issue"]
    if component == "hbm_controller":
        return count * coefficients["hbm.dma_issue"]
    return 0.0


def _fp_format_key(config: Mapping[str, Any]) -> str:
    if "FP_SETTING" in config:
        value = str(config["FP_SETTING"]).upper()
        return value if value.startswith("FP_") else f"FP_{value}"
    exp = int(config.get("S_FP_EXP_WIDTH", config.get("FP_EXP_WIDTH", 5)))
    mant = int(config.get("S_FP_MANT_WIDTH", config.get("FP_MANT_WIDTH", 6)))
    return f"FP_E{exp}M{mant}"


def _lookup_format_coefficient(
    table: Mapping[str, Any],
    format_key: str,
    fp_width: int,
) -> float:
    """Use an exact format anchor or interpolate by total FP width."""

    if format_key in table:
        return float(table[format_key])
    parsed: list[tuple[int, int, float]] = []
    import re

    for name, raw_value in table.items():
        match = re.fullmatch(r"FP_E(\d+)M(\d+)", str(name).upper())
        if match:
            parsed.append(
                (int(match.group(1)), int(match.group(2)), float(raw_value))
            )
    target_match = re.fullmatch(r"FP_E(\d+)M(\d+)", format_key.upper())
    if target_match and len(parsed) >= 2:
        for left, right in pairwise(sorted(parsed)):
            determinant = left[0] * right[1] - right[0] * left[1]
            if determinant == 0:
                continue
            exp_c = (left[2] * right[1] - right[2] * left[1]) / determinant
            mant_c = (left[0] * right[2] - right[0] * left[2]) / determinant
            if exp_c >= 0 and mant_c >= 0:
                return (
                    exp_c * int(target_match.group(1))
                    + mant_c * int(target_match.group(2))
                )
    anchors = sorted((1 + exp + mant, value) for exp, mant, value in parsed)
    if not anchors:
        return float(table.get("default", 0.0))
    anchors.sort()
    if fp_width <= anchors[0][0]:
        return anchors[0][1] * fp_width / max(1, anchors[0][0])
    if fp_width >= anchors[-1][0]:
        return anchors[-1][1] * fp_width / max(1, anchors[-1][0])
    for (left_width, left), (right_width, right) in pairwise(anchors):
        if left_width <= fp_width <= right_width:
            fraction = (fp_width - left_width) / max(1, right_width - left_width)
            return left + fraction * (right - left)
    return anchors[-1][1]


def _activity_ratio(
    coefficients: Mapping[str, Any],
    key: str,
    quantile: str,
) -> float:
    if quantile == "nominal":
        return 1.0
    envelope = coefficients.get("activity_envelope", {})
    selected = envelope.get(key, envelope.get(key.rsplit(".", 1)[0], {}))
    return float(selected.get(quantile, 1.0))


def _logic_action_energy_v2(
    action: Mapping[str, Any],
    config: Mapping[str, Any],
    coefficients: Mapping[str, Any],
    widths: Mapping[str, Any],
    *,
    quantile: str,
) -> float:
    """Evaluate one structural v2 action under an empirical activity profile."""

    component = str(action["component"])
    family = str(action["action"])
    count = float(action["count"])
    mlen = int(config["MLEN"])
    blen = int(config["BLEN"])
    vlen = int(config.get("VLEN", mlen))
    fp_width = widths["fp"]
    rtl_v6_energy = _rtl_v6_action_energy(
        action,
        config,
        fp_width,
        quantile=quantile,
    )
    if rtl_v6_energy is not None:
        return rtl_v6_energy
    dynamic = coefficients["dynamic_nominal_pj"]
    envelope_key = f"{component}.{family}"
    if component == "matrix":
        matrix = dynamic["matrix"]
        mode = str(widths["mode"])
        if family in {"array_compute", "matrix_vector_compute"}:
            leaf = matrix[mode]["pe_cycle"]
            if mode == "mxint":
                pe_energy = (
                    float(leaf["base"])
                    + float(leaf["bit_product"]) * widths["t"] * widths["l"]
                    + float(leaf["width_sum"]) * (widths["t"] + widths["l"])
                )
            else:
                pe_energy = (
                    float(leaf["base"])
                    + float(leaf["bit_product"]) * widths["t"] * widths["l"]
                    + float(leaf["width_sum"]) * (widths["t"] + widths["l"])
                )
            split_count = max(1, mlen // blen)
            pe_cycles_per_slice = (
                blen * blen if family == "matrix_vector_compute" else blen**3
            )
            slice_energy = (
                float(leaf.get("slice_fixed", 0.0))
                + blen * float(leaf.get("feed_cycle", 0.0))
                + pe_cycles_per_slice * pe_energy
            )
            nominal = count * split_count * slice_energy
            envelope_key = f"matrix.{family}.{mode}"
        elif family == "cross_k_reduce":
            split_nodes = blen * blen * max(mlen // blen - 1, 0)
            accumulator_width = (
                fp_width
                if mode == "mxfp"
                else widths["t"] + widths["l"] + math.ceil(math.log2(max(1, blen)))
            )
            nominal = count * split_nodes * accumulator_width * float(matrix[mode]["reduce_node_bit"])
            envelope_key = f"matrix.cross_k_reduce.{mode}"
        else:
            nominal = count * blen * blen * fp_width * float(matrix[mode]["output_bit"])
            envelope_key = f"matrix.output_conversion.{mode}"
    elif component == "vector":
        lanes = int(action.get("active_lanes") or vlen)
        if family.startswith("compact_stats_"):
            lanes = int(action.get("segment_count") or lanes)
            configured_lanes = int(config.get("COMPACT_STATS_LANES", 16))
            v5_overlay = _load_vector_rtl_v5_energy()
            v5_values = v5_overlay.get(
                "dynamic_nominal_pj_per_lane_action",
                {},
            )
            use_v5 = (
                configured_lanes > 16
                and str(v5_overlay.get("calibration_status", "")).startswith(
                    "rtl_activity_calibrated"
                )
                and family in v5_values
            )
            overlay = v5_overlay if use_v5 else _load_vector_rtl_v4_energy()
            values = (
                v5_values
                if use_v5
                else overlay.get("dynamic_nominal_pj", {})
            )
            if not str(overlay.get("calibration_status", "")).startswith(
                "rtl_activity_calibrated"
            ) or family not in values:
                return 0.0
            if use_v5:
                nominal = (
                    count
                    * float(values[family])
                    * lanes
                    * fp_width
                    / 12.0
                )
            else:
                # RTL-v4 measured a 16-lane operation. Scale its action
                # energy by logical active lanes for compatibility.
                nominal = (
                    count
                    * float(values[family])
                    * lanes
                    / 16.0
                    * fp_width
                    / 12.0
                )
            ratios = overlay.get("activity_envelope", {}).get(family, {})
            return nominal * float(ratios.get(quantile, 1.0))
        row_lanes = lanes
        if family.startswith("softmax_row_"):
            family = {
                "softmax_row_subtract": "lane_add_sub_vf",
                "softmax_row_multiply": "lane_multiply_vf",
                "softmax_row_exp": "lane_sfu_exp",
            }[family]
            scale = row_lanes * vlen
        elif family.startswith("reduction"):
            if family.endswith("_rows"):
                active_nodes = row_lanes * max(1, vlen - 1)
            elif family.endswith("_segment"):
                active_nodes = max(1, vlen * int(math.log2(max(2, vlen))))
            elif family.endswith("_segments"):
                active_nodes = vlen
            else:
                active_nodes = max(
                    1, (lanes - 1) * int(math.log2(max(2, vlen)))
                )
            scale = active_nodes
        else:
            scale = lanes
        format_key = _fp_format_key(config)
        family_table = dynamic["vector"].get(family, dynamic["vector"].get("default", {}))
        nominal = count * scale * _lookup_format_coefficient(family_table, format_key, fp_width)
        # EnergyAction stores the originating ISA mnemonic in ``precision``.
        # Accept ``source_opcode`` as well for hand-authored/external traces.
        source_opcode = str(
            action.get("source_opcode") or action.get("precision") or ""
        )
        overwrite_kernel = {
            "V_RED_SUM_OVR": "reduce_sum_ovr",
            "V_RED_MAX_OVR": "reduce_max_ovr",
            "V_RED_SUM_SEG_OVR": "reduce_sum_seg_ovr",
            "V_RED_MAX_SEG_OVR": "reduce_max_seg_ovr",
        }.get(source_opcode)
        if overwrite_kernel is not None:
            overlay = _load_vector_rtl_v4_energy()
            delta = overlay.get("reduction_overwrite_delta_pj", {}).get(
                overwrite_kernel
            )
            if delta is not None:
                nominal += count * max(0.0, float(delta))
    elif component == "softmax_state":
        rows = int(action.get("active_lanes") or 1)
        scalar = dynamic["scalar"]
        format_key = _fp_format_key(config)

        def scalar_energy(name: str) -> float:
            table = scalar.get(name, scalar.get("default", {}))
            return _lookup_format_coefficient(table, format_key, fp_width)

        per_row = {
            "max_update": (
                2 * scalar_energy("fp_add_sub_move")
                + scalar_energy("fp_sfu_exp")
            ),
            "sum_update": (
                scalar_energy("fp_multiply")
                + scalar_energy("fp_add_sub_move")
            ),
            "final_reciprocal": scalar_energy("fp_sfu_reciprocal"),
        }.get(family)
        if per_row is None:
            return 0.0
        nominal = count * rows * per_row
        envelope_key = "scalar.register_or_sram_access"
    elif component == "packed_pv_accumulator":
        format_key = _fp_format_key(config)
        vector = dynamic["vector"]
        if family == "accumulate":
            table = vector.get("lane_add_sub_vv", vector.get("default", {}))
        elif family == "overwrite":
            table = vector.get("lane_movement_shift", vector.get("default", {}))
        else:
            return 0.0
        # Both modes write the complete Matrix output tile.  The overwrite
        # path selects movement rather than add energy, but has identical
        # output-cell multiplicity.
        active = blen * blen
        nominal = count * active * _lookup_format_coefficient(
            table, format_key, fp_width
        )
        envelope_key = "vector.lane_add_sub_vv"
    elif component == "scalar":
        format_key = _fp_format_key(config)
        family_table = dynamic["scalar"].get(family, dynamic["scalar"].get("default", {}))
        if family.startswith("integer"):
            nominal = count * float(family_table.get(str(config.get("INT_DATA_WIDTH", 32)), family_table.get("default", 0.0)))
        elif family in {"vector_lane_load", "vector_lane_store"}:
            nominal = count * vlen * _lookup_format_coefficient(
                family_table, format_key, fp_width
            )
        else:
            nominal = count * _lookup_format_coefficient(family_table, format_key, fp_width)
    elif component == "control":
        nominal = count * float(dynamic["control"]["frontend_issue"])
    elif component == "agu":
        agu = _load_agu_coefficients()
        action_energy = agu.get("dynamic_nominal_pj", {})
        if not action_energy:
            frontend = float(dynamic["control"]["frontend_issue"])
            scale = (
                max(1, int(action.get("active_lanes", 0)))
                if family == "agu_stream_step"
                else 0.25
                if family == "agu_offset_read"
                else 1.0
            )
            return count * frontend * scale
        if family not in action_energy:
            return 0.0
        scale = (
            max(1, int(action.get("active_lanes", 0)))
            if family == "agu_stream_step"
            else 1
        )
        nominal = count * scale * float(action_energy[family])
        if quantile == "low":
            nominal *= float(agu["activity_envelope"]["low"])
        elif quantile == "high":
            nominal *= float(agu["activity_envelope"]["high"])
        return nominal
    elif component == "hbm_controller":
        # v2 HBM coefficients are measured per accepted logical lane.  The
        # CostEmitter action carries the production DMA amount, so energy is
        # monotonic with transfer size without pretending that the fixed-
        # amount calibration identified separate line/byte coefficients.
        lanes = max(1, int(action.get("active_lanes", 0)))
        nominal = count * lanes * float(
            dynamic["hbm_controller"].get(
                family, dynamic["hbm_controller"]["default"]
            )
        )
    else:
        return 0.0
    return nominal * _activity_ratio(coefficients, envelope_key, quantile)


def _structurally_zero_action(
    action: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[bool, int]:
    """Identify emitted actions whose configured hardware census is empty."""

    if (
        str(action.get("component")) == "matrix"
        and str(action.get("action")) == "cross_k_reduce"
    ):
        mlen = int(config["MLEN"])
        blen = int(config["BLEN"])
        physical_instances = blen * blen * max(mlen // blen - 1, 0)
        return physical_instances == 0, physical_instances
    return False, 0


def _logic_component_areas(area_metrics: Mapping[str, Any]) -> dict[str, float]:
    breakdown = area_metrics.get("area_breakdown", {})
    return {
        "matrix": float(breakdown.get("MatrixMachine", 0.0)),
        "vector": float(breakdown.get("VectorMachine", 0.0)),
        "scalar": sum(
            float(breakdown.get(name, 0.0))
            for name in (
                "ScalarIntLogic",
                "ScalarFPLogic",
                "ScalarVectorBufferLogic",
                "ScalarControl",
                "ScalarRTLv3PipelineDelta",
            )
        ),
        "hbm_controller": sum(
            float(value)
            for name, value in breakdown.items()
            if str(name).startswith("HBM")
        ),
        "agu": float(breakdown.get("AddressGenerationUnit", 0.0)),
        "control": float(breakdown.get("FullChipTopResidual", 0.0)),
    }


def _logic_subcomponent_areas(
    area_metrics: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Normalize area_new hierarchy names into clock-gating domains."""

    matrix = dict((area_metrics.get("matrix_machine") or {}).get("breakdown") or {})
    vector = dict((area_metrics.get("vector_machine") or {}).get("breakdown") or {})
    scalar = dict((area_metrics.get("scalar_machine") or {}).get("breakdown") or {})
    hbm = dict((area_metrics.get("hbm_system") or {}).get("breakdown") or {})

    vector_lane = sum(
        float(vector.get(name, 0.0))
        for name in (
            "VectorElementUnit",
            "VectorLaneMantissaLogic",
            "VectorLaneExponentLogic",
            "VectorLaneQuadraticLogic",
        )
    )
    vector_reduction = sum(
        float(vector.get(name, 0.0))
        for name in ("VectorReductionUnit", "VectorReductionLogic")
    )
    vector_buffers = float(vector.get("VectorBuffers", 0.0))
    vector_control = float(vector.get("VectorTopControl", 0.0)) + float(
        vector.get("VectorControl", 0.0)
    )
    # A legacy single-number vector model cannot be partitioned safely. Keep
    # the full area in control so ideal mode only clocks it while vector work
    # is active, without inventing a lane/reduction split.
    if not any((vector_lane, vector_reduction, vector_buffers, vector_control)):
        vector_control = float(vector.get("VectorMachine", 0.0))

    return {
        "matrix": {
            "array_stack": float(matrix.get("array_stack_area", 0.0)),
            "reduce_tree": float(matrix.get("reduce_tree_area", 0.0)),
            "output_accumulator": float(
                matrix.get("output_accumulator_area", 0.0)
            ),
            "output_conversion": float(
                matrix.get("output_conversion_area", 0.0)
            ),
            "result_buffer": float(matrix.get("result_buffer_area", 0.0)),
            "io_pipeline": float(matrix.get("io_pipeline_area", 0.0)),
            "component_control": float(matrix.get("control_area", 0.0)),
        },
        "vector": {
            "lane_datapath": vector_lane,
            "reduction_tree": vector_reduction,
            "buffers": vector_buffers,
            "component_control": vector_control,
            "segment_parallel_delta": float(
                vector.get("VectorRTLv3SegmentParallelDelta", 0.0)
            ),
            "compact_stats_simd": float(vector.get("CompactStatsSIMD", 0.0)),
            "reduction_overwrite_control": float(
                vector.get("ReductionOverwriteControl", 0.0)
            ),
            "softmax_aux_row_slices": float(
                vector.get("SoftmaxAuxRowSlices", 0.0)
            ),
            "state_simd": float(vector.get("SoftmaxStateSIMD", 0.0)),
            "state_bank_control": float(
                vector.get("BankedVectorSRAMControl", 0.0)
            ),
            "packed_pv_accumulator": float(
                vector.get("PackedPVAccumulator", 0.0)
            ),
        },
        "scalar": {
            "int_datapath": float(scalar.get("ScalarIntLogic", 0.0)),
            "fp_datapath": float(scalar.get("ScalarFPLogic", 0.0)),
            "lane_access": float(
                scalar.get("ScalarVectorBufferLogic", 0.0)
            ),
            "component_control": float(scalar.get("ScalarControl", 0.0)),
            "pipeline_delta": float(
                scalar.get("ScalarRTLv3PipelineDelta", 0.0)
            ),
        },
        "hbm_controller": {
            "matrix_path": float(hbm.get("HBMMatrixPath", 0.0)),
            "vector_path": float(hbm.get("HBMVectorPath", 0.0)),
            "scale_path": float(hbm.get("HBMScalePath", 0.0)),
            "address_control": float(hbm.get("HBMAddressControl", 0.0)),
            "prefetch_writeback": float(
                hbm.get("HBMPrefetchWritebackControl", 0.0)
            ),
            "fixed_control": float(hbm.get("HBMFixedControl", 0.0)),
            "component_control": 0.0,
        },
        "agu": {
            "descriptor_storage": 0.0,
            "stride_adders": 0.0,
            "offset_read_path": 0.0,
            "component_control": float(
                (area_metrics.get("area_breakdown") or {}).get(
                    "AddressGenerationUnit", 0.0
                )
            ),
        },
        # FullChipTopResidual intentionally has no ideal clock domain. It is a
        # mixture of wrapper/interconnect logic and aggregate model residual,
        # not a measured frontend hierarchy.
        "control": {"frontend": 0.0, "component_control": 0.0},
    }


def _clock_density(
    component: str,
    widths: Mapping[str, Any],
    coefficients: Mapping[str, Any],
) -> tuple[float, float]:
    key = (
        f"matrix.{widths['mode']}"
        if component == "matrix"
        else "control"
        if component == "agu"
        else component
    )
    densities = coefficients["clock_pj_per_cycle_um2"]
    fixed = coefficients.get("clock_fixed_pj_per_cycle", {})
    return (
        float(densities.get(key, densities.get(component, 0.0))),
        float(fixed.get(key, fixed.get(component, 0.0))),
    )


def _ungated_clock_energy(
    *,
    cycles: int,
    area_metrics: Mapping[str, Any],
    widths: Mapping[str, Any],
    coefficients: Mapping[str, Any],
) -> dict[str, float]:
    areas = _logic_component_areas(area_metrics)
    result: dict[str, float] = {}
    for component, area in areas.items():
        density, fixed = _clock_density(component, widths, coefficients)
        result[component] = cycles * (area * density + fixed)
    return result


def _ideal_clock_energy(
    *,
    actions: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    timing: Mapping[str, Any],
    cycles: int,
    area_metrics: Mapping[str, Any],
    widths: Mapping[str, Any],
    coefficients: Mapping[str, Any],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, Any],
    dict[str, float],
]:
    """Return ideal clock energy, activity, ClockWork, and subcomponent energy."""

    clock_work = build_clock_work(actions, config, timing)
    if clock_work["status"] != "complete":
        examples = clock_work["unavailable"][:3]
        raise ValueError(
            "ideal hierarchical clock work is unavailable for one or more "
            f"compiler actions: {examples}"
        )
    subcomponent_areas = _logic_subcomponent_areas(area_metrics)
    work_cycles: Counter[tuple[str, str]] = Counter()
    component_active_cycles: Counter[str] = Counter()
    for record in clock_work["records"]:
        key = (str(record["component"]), str(record["subcomponent"]))
        work_cycles[key] += float(record["equivalent_full_area_cycles"])
        component_active_cycles[str(record["component"])] += float(
            record["component_active_cycles"]
        )

    energy_by_component: Counter[str] = Counter()
    energy_by_subcomponent: dict[str, float] = {}
    equivalent_area_cycles_by_component: Counter[str] = Counter()
    for component, areas in subcomponent_areas.items():
        density, fixed = _clock_density(component, widths, coefficients)
        for subcomponent, area in areas.items():
            if subcomponent == "component_control":
                # This is a real hierarchy area for Matrix/Vector/Scalar, but
                # the record also carries fixed-clock activity. Charge both
                # terms once below.
                pass
            active_cycles = min(
                float(cycles), work_cycles.get((component, subcomponent), 0.0)
            )
            area_cycles = max(0.0, area) * active_cycles
            equivalent_area_cycles_by_component[component] += area_cycles
            energy = density * area_cycles
            energy_by_component[component] += energy
            energy_by_subcomponent[f"{component}.{subcomponent}"] = energy
        fixed_cycles = min(cycles, component_active_cycles.get(component, 0))
        fixed_energy = fixed * fixed_cycles
        energy_by_component[component] += fixed_energy
        energy_by_subcomponent[f"{component}.fixed_clock"] = fixed_energy

    component_areas = _logic_component_areas(area_metrics)
    active_fraction = {}
    for component, area in component_areas.items():
        denominator = area * cycles
        active_fraction[component] = (
            0.0
            if denominator <= 0
            else min(
                1.0,
                equivalent_area_cycles_by_component.get(component, 0.0)
                / denominator,
            )
        )
    # Make the post-cap values auditable rather than exposing only raw work.
    clock_work["capped_equivalent_full_area_cycles"] = {
        f"{component}.{subcomponent}": min(
            float(cycles), raw_cycles
        )
        for (component, subcomponent), raw_cycles in sorted(work_cycles.items())
    }
    clock_work["component_active_cycles"] = {
        component: min(cycles, raw_cycles)
        for component, raw_cycles in sorted(component_active_cycles.items())
    }
    return (
        dict(energy_by_component),
        active_fraction,
        clock_work,
        energy_by_subcomponent,
    )


def _sram_dynamic_energy(
    actions: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
) -> tuple[float, dict[str, float], dict[str, Any], list[str]]:
    area = estimate_sram_area(
        dict(config),
        sram_port_model=str(
            config.get("SRAM_PORT_MODEL", "replicated-single-port")
        ),
    )
    tiling = area.get("area_sram_macro_tiling", {})
    energy_by_macro = macro_energy_lookup(dict(catalog))
    component_to_key = {
        "matrix_sram": "matrix",
        "vector_sram": "vector",
        "scalar_int_sram": "scalar_int",
        "scalar_fp_sram": "scalar_fp",
        "softmax_state_sram": "softmax_state",
    }
    breakdown: Counter[str] = Counter()
    accesses: Counter[str] = Counter()
    warnings: list[str] = []
    for action in actions:
        component = str(action.get("component"))
        if component not in component_to_key:
            continue
        kind = component_to_key[component]
        detail = tiling.get(kind)
        if not detail:
            warnings.append(f"no SRAM macro tiling for {component}")
            continue
        operation = str(action.get("action"))
        energy_key = "write_energy_pj" if operation == "write" else "read_energy_pj"
        count = float(action.get("count", 0))
        # A MatrixMachine opcode streams BLEN rows through the local array. DMA
        # actions carry their actual amount in active_lanes. Other vector/scalar
        # actions access one logical word per instruction.
        if int(action.get("active_lanes", 0)) > 0:
            row_accesses = count * int(action["active_lanes"])
        elif component == "matrix_sram" and str(action.get("precision", "")).startswith("M_"):
            row_accesses = count * int(config["BLEN"])
        else:
            row_accesses = count
        banks = [
            dict(bank)
            for bank in detail.get("banks", ())
            if not bank.get("unused_empty_bank", False)
        ]
        if banks:
            # A row-interleaved logical access selects exactly one bank. Most
            # physical configurations use the same macro and width tiling in
            # every bank. If depth rounding selects different macros, weight
            # their per-row charge by the number of logical rows mapped to
            # each bank rather than charging all banks for every access.
            total_logical_rows = math.fsum(
                float(bank.get("logical_depth", 0.0)) for bank in banks
            )
            weighted_energy = 0.0
            missing_macros: list[str] = []
            for bank in banks:
                macro_name = str(bank.get("macro"))
                macro = energy_by_macro.get(macro_name)
                if macro is None:
                    missing_macros.append(macro_name)
                    continue
                weighted_energy += (
                    float(bank.get("logical_depth", 0.0))
                    * int(bank["width_tiles"])
                    * float(macro[energy_key])
                )
            if missing_macros:
                warnings.append(
                    "no SRAM energy table entry for bank macros "
                    + ",".join(sorted(set(missing_macros)))
                )
                continue
            energy_per_logical_row = (
                weighted_energy / total_logical_rows
                if total_logical_rows > 0.0
                else 0.0
            )
        else:
            macro = energy_by_macro.get(str(detail["macro"]))
            if macro is None:
                warnings.append(
                    f"no SRAM energy table entry for {detail['macro']}"
                )
                continue
            energy_per_logical_row = int(detail["width_tiles"]) * float(
                macro[energy_key]
            )
        energy = row_accesses * energy_per_logical_row
        breakdown[component] += energy
        accesses[f"{component}.{operation}"] += row_accesses
    return sum(breakdown.values()), dict(breakdown), {
        "logical_accesses": dict(accesses),
        "macro_tiling": tiling,
        "catalog_model": catalog.get("model"),
        "sram_port_energy_model": "ideal_independent_access",
        "dual_port_overhead_included": False,
    }, warnings


def _sram_background_energy(
    sram_metadata: Mapping[str, Any],
    makespan_ns: float,
    artifact: Mapping[str, Any],
) -> tuple[float, float, float, dict[str, float]]:
    """Calculate capacity-proportional SRAM background energy.

    Decimal GB follows the source paper. ``covered_bits`` includes macro tiling
    padding and physical port copies selected by the SRAM area model.
    """

    tiling = dict(sram_metadata.get("macro_tiling") or {})
    coefficient_w_per_gb = float(
        artifact["selected_background_power_w_per_gb"]
    )
    capacity_by_component_gb = {
        str(component): float(detail.get("covered_bits", 0.0)) / 8.0e9
        for component, detail in tiling.items()
    }
    capacity_gb = math.fsum(capacity_by_component_gb.values())
    power_w = capacity_gb * coefficient_w_per_gb
    # W * ns = 1e3 pJ.
    energy_pj = power_w * makespan_ns * 1.0e3
    breakdown_pj = {
        component: capacity * coefficient_w_per_gb * makespan_ns * 1.0e3
        for component, capacity in capacity_by_component_gb.items()
    }
    return energy_pj, power_w, capacity_gb, breakdown_pj


def estimate_onchip_power(
    config: Mapping[str, Any],
    cost_trace: Any,
    timing_report: Any,
    *,
    logic_coefficients_path: str | Path | None = None,
    sram_energy_path: str | Path | None = None,
    sram_background_path: str | Path | None = None,
    makespan_ns_override: float | None = None,
    makespan_source_override: str | None = None,
    clock_gating_mode: str = "ungated",
) -> dict[str, Any]:
    """Estimate fixed-corner on-chip energy and average power.

    The estimate is returned even while logic activity coefficients are in
    bootstrap state because DSE consumes it as a shadow metric.  Callers must
    inspect ``calibration_status`` and ``warnings`` before using it as evidence.
    """

    cfg = dict(config)
    clock_period_ps = int(cfg.get("CLOCK_PERIOD_PS", 1000))
    timing = _mapping(timing_report)
    actions, trace = _trace_actions(cost_trace)
    normalized_clock_mode = str(clock_gating_mode).replace("-", "_").lower()
    if normalized_clock_mode not in {"ungated", "ideal_hierarchical"}:
        raise ValueError(
            "clock_gating_mode must be 'ungated' or 'ideal_hierarchical', got "
            f"{clock_gating_mode!r}"
        )
    coefficients = _load_logic_coefficients(logic_coefficients_path)
    model_version = str(coefficients.get("model", "onchip_action_energy_v1"))
    dynamic_coefficients = coefficients.get("dynamic_pj", {})
    sides = derive_compute_sides(
        cfg["ACT_WIDTH"],
        cfg["KV_WIDTH"],
        cfg.get("WEIGHT_WIDTH", "MXINT4"),
        default_scale_width=int(cfg.get("MX_SCALE_WIDTH", 8)),
    )
    widths = {
        "mode": str(sides["mode"]),
        "t": int(sides["t_width"]),
        "l": int(sides["l_width"]),
        "fp": _fp_width(cfg),
    }
    if makespan_ns_override is None:
        cycles, makespan_ns, makespan_source = _makespan(timing, clock_period_ps)
    else:
        if makespan_ns_override <= 0:
            raise ValueError(
                f"makespan_ns_override must be positive, got {makespan_ns_override}"
            )
        makespan_ns = float(makespan_ns_override)
        cycles = max(1, math.ceil(makespan_ns * 1000.0 / clock_period_ps))
        makespan_source = makespan_source_override or "explicit_override"
    area_metrics = estimate_area(cfg)
    stage_logic: Counter[str] = Counter()
    stage_logic_low: Counter[str] = Counter()
    stage_logic_high: Counter[str] = Counter()
    component_logic: Counter[str] = Counter()
    action_logic_by_action: Counter[str] = Counter()
    action_active_elements_by_action: Counter[str] = Counter()
    action_active_rows_by_action: Counter[str] = Counter()
    unknown_actions: Counter[str] = Counter()
    structurally_zero_actions: Counter[str] = Counter()
    structural_physical_instances: Counter[str] = Counter()
    compact_stats_actions = {
        str(action.get("action"))
        for action in actions
        if str(action.get("action")).startswith("compact_stats_")
        and float(action.get("count", 0)) > 0
    }
    rtl_v6_action_components = {
        str(action.get("component"))
        for action in actions
        if str(action.get("component"))
        in {"softmax_state", "packed_pv_accumulator", "softmax_state_sram"}
        or str(action.get("action")).startswith("softmax_row_")
        or str(action.get("action")).endswith("_rows")
    }
    unit_logic_energy_v2: dict[
        tuple[tuple[str, str], ...], tuple[float, float, float]
    ] = {}
    for action in actions:
        component = str(action.get("component"))
        if component.endswith("_sram"):
            continue
        if model_version == "onchip_action_energy_v2":
            # Stage, compiler lineage and diagnostic variants do not alter
            # the hardware action coefficient. Evaluate each physical action
            # shape once per rank, then apply its exact compressed count.
            # This is particularly important for tile-aware multi-chip power,
            # where the same action families are attributed to many kernels.
            unit_key = tuple(
                sorted(
                    (str(key), repr(value))
                    for key, value in action.items()
                    if key
                    not in {
                        "count",
                        "stage",
                        "parallel_kernel",
                        "variant",
                    }
                )
            )
            unit_energies = unit_logic_energy_v2.get(unit_key)
            if unit_energies is None:
                unit_action = dict(action)
                unit_action["count"] = 1.0
                unit_energies = (
                    _logic_action_energy_v2(
                        unit_action,
                        cfg,
                        coefficients,
                        widths,
                        quantile="nominal",
                    ),
                    _logic_action_energy_v2(
                        unit_action,
                        cfg,
                        coefficients,
                        widths,
                        quantile="low",
                    ),
                    _logic_action_energy_v2(
                        unit_action,
                        cfg,
                        coefficients,
                        widths,
                        quantile="high",
                    ),
                )
                unit_logic_energy_v2[unit_key] = unit_energies
            count = float(action.get("count", 0.0))
            energy, low_energy, high_energy = (
                value * count for value in unit_energies
            )
        else:
            energy = _logic_action_energy(action, cfg, dynamic_coefficients, widths)
            low_energy = energy
            high_energy = energy
        agu_covered = (
            component == "agu"
            and str(action.get("action"))
            in _load_agu_coefficients().get("dynamic_nominal_pj", {})
        )
        action_key = f"{component}.{action.get('action')}"
        structurally_zero, physical_instances = _structurally_zero_action(
            action, cfg
        )
        if structurally_zero and int(action.get("count", 0)):
            structurally_zero_actions[action_key] += float(action["count"])
            structural_physical_instances[action_key] += physical_instances
        elif energy == 0.0 and int(action.get("count", 0)) and not agu_covered:
            unknown_actions[action_key] += float(action["count"])
        stage = str(action.get("stage", "global"))
        stage_logic[stage] += energy
        stage_logic_low[stage] += low_energy
        stage_logic_high[stage] += high_energy
        component_logic[component] += energy
        action_logic_by_action[action_key] += energy
        count = float(action.get("count", 0.0))
        active_rows = int(action.get("active_lanes", 0) or 0)
        if active_rows <= 0:
            active_rows = int(cfg.get("SOFTMAX_ROW_LANES", 1))
        if action_key in {
            "vector.reduction_max_rows",
            "vector.reduction_sum_rows",
            "vector.softmax_row_exp",
            "vector.softmax_row_multiply",
            "vector.softmax_row_subtract",
        }:
            action_active_elements_by_action[action_key] += (
                count * active_rows * int(cfg.get("VLEN", 1))
            )
        elif component == "softmax_state":
            action_active_rows_by_action[action_key] += count * active_rows
        elif component == "packed_pv_accumulator":
            write_lanes = int(cfg.get("BLEN", cfg.get("BLOCK_DIM", 1)))
            action_active_elements_by_action[action_key] += (
                count * write_lanes * write_lanes
            )
    hbm_read_bytes = int(timing.get("hbm_read_bytes", timing.get("read_bytes", 0)))
    hbm_write_bytes = int(timing.get("hbm_write_bytes", timing.get("write_bytes", 0)))
    hbm_read_requests = int(timing.get("hbm_read_requests", 0))
    hbm_write_requests = int(timing.get("hbm_write_requests", 0))
    if model_version == "onchip_action_energy_v2":
        hbm_coefficients = coefficients["dynamic_nominal_pj"]["hbm_controller"]
        hbm_increment = (
            (hbm_read_bytes + hbm_write_bytes) * float(hbm_coefficients["byte"])
            + (hbm_read_requests + hbm_write_requests) * float(hbm_coefficients["line"])
        )
        hbm_low = hbm_increment * _activity_ratio(
            coefficients, "hbm_controller.physical_transfer", "low"
        )
        hbm_high = hbm_increment * _activity_ratio(
            coefficients, "hbm_controller.physical_transfer", "high"
        )
    else:
        hbm_increment = (
            (hbm_read_bytes + hbm_write_bytes) * dynamic_coefficients["hbm.byte"]
            + (hbm_read_requests + hbm_write_requests) * dynamic_coefficients["hbm.line"]
        )
        hbm_low = hbm_increment
        hbm_high = hbm_increment
    component_logic["hbm_controller"] += hbm_increment
    stage_logic["global/hbm_physical_lines"] += hbm_increment
    stage_logic_low["global/hbm_physical_lines"] += hbm_low
    stage_logic_high["global/hbm_physical_lines"] += hbm_high

    action_logic_pj = sum(stage_logic.values())
    action_logic_low_pj = sum(stage_logic_low.values())
    action_logic_high_pj = sum(stage_logic_high.values())
    clock_work: dict[str, Any] = {
        "schema": "compressed_clock_work_v1",
        "status": "not_requested",
        "records": [],
    }
    clock_work_error: str | None = None
    clock_active_fraction = {
        component: 1.0 for component in _logic_component_areas(area_metrics)
    }
    ideal_clock_by_subcomponent: dict[str, float] = {}
    if model_version == "onchip_action_energy_v2":
        ungated_clock_breakdown = _ungated_clock_energy(
            cycles=cycles,
            area_metrics=area_metrics,
            widths=widths,
            coefficients=coefficients,
        )
        try:
            (
                ideal_clock_breakdown,
                ideal_active_fraction,
                ideal_clock_work,
                ideal_clock_by_subcomponent,
            ) = _ideal_clock_energy(
                actions=actions,
                config=cfg,
                timing=timing,
                cycles=cycles,
                area_metrics=area_metrics,
                widths=widths,
                coefficients=coefficients,
            )
        except ValueError as exc:
            if normalized_clock_mode == "ideal_hierarchical":
                raise
            ideal_clock_breakdown = {}
            ideal_active_fraction = {}
            ideal_clock_work = {
                "schema": "compressed_clock_work_v1",
                "status": "clock_work_unavailable",
                "records": [],
                "failure_reason": str(exc),
            }
            ideal_clock_by_subcomponent = {}
            clock_work_error = str(exc)
        if normalized_clock_mode == "ideal_hierarchical":
            clock_breakdown = ideal_clock_breakdown
            clock_active_fraction = ideal_active_fraction
            clock_work = ideal_clock_work
        else:
            clock_breakdown = ungated_clock_breakdown
            # Keep ClockWork available as a shadow diagnostic even when the
            # selected semantics are the historical upper bound.
            clock_work = ideal_clock_work
    else:
        if normalized_clock_mode == "ideal_hierarchical":
            raise ValueError(
                "ideal hierarchical clock gating requires the v2 area-density "
                "clock artifact"
            )
        clocks = coefficients["clock_pj_per_cycle"]
        legacy_control_hbm = float(clocks.get("control_hbm", 0.30))
        ungated_clock_breakdown = {
            "matrix": cycles * int(cfg["MLEN"]) * int(cfg["BLEN"]) * float(clocks["matrix_pe"]),
            "vector": cycles * int(cfg.get("VLEN", cfg["MLEN"])) * float(clocks["vector_lane"]),
            "scalar": cycles * float(clocks["scalar_machine"]),
            "control": cycles * float(clocks.get("control_frontend", legacy_control_hbm)),
            "hbm_controller": cycles * float(clocks.get("hbm_controller", legacy_control_hbm)),
        }
        clock_breakdown = ungated_clock_breakdown
        ideal_clock_breakdown = {}
    for component, energy in clock_breakdown.items():
        component_logic[f"{component}_clock"] += energy
    stage_logic["global/clock_baseline"] += sum(clock_breakdown.values())
    stage_logic_low["global/clock_baseline"] += sum(clock_breakdown.values())
    stage_logic_high["global/clock_baseline"] += sum(clock_breakdown.values())
    logic_dynamic_pj = sum(stage_logic.values())
    selected_clock_pj = sum(clock_breakdown.values())
    ungated_clock_pj = sum(ungated_clock_breakdown.values())
    ideal_clock_pj = sum(ideal_clock_breakdown.values())

    catalog = load_sram_energy_catalog(sram_energy_path)
    sram_dynamic_pj, sram_breakdown, sram_metadata, sram_warnings = _sram_dynamic_energy(
        actions, cfg, catalog=catalog
    )
    sram_background_artifact = _load_sram_background(sram_background_path)
    (
        sram_background_pj,
        sram_background_power_w,
        sram_allocated_capacity_gb,
        sram_background_breakdown_pj,
    ) = _sram_background_energy(
        sram_metadata,
        makespan_ns,
        sram_background_artifact,
    )
    warnings = list(sram_warnings)
    if clock_period_ps != int(coefficients["corner"]["clock_period_ps"]):
        warnings.append(
            f"CLOCK_PERIOD_PS={clock_period_ps} is outside the fixed 1000 ps calibration; coefficients were not frequency-scaled"
        )
    sram_area_um2 = float((area_metrics.get("sram") or {}).get("area", 0.0))
    total_area_um2 = float(area_metrics.get("area", 0.0))
    logic_area_um2 = max(0.0, total_area_um2 - sram_area_um2)
    leakage_mw = logic_area_um2 * float(coefficients["logic_leakage_mw_per_um2"])
    leakage_pj = leakage_mw * makespan_ns
    total_pj = (
        logic_dynamic_pj
        + sram_dynamic_pj
        + leakage_pj
        + sram_background_pj
    )
    ungated_logic_dynamic_pj = action_logic_pj + ungated_clock_pj
    ungated_total_pj = (
        ungated_logic_dynamic_pj
        + sram_dynamic_pj
        + leakage_pj
        + sram_background_pj
    )
    if model_version == "onchip_action_energy_v2":
        residual = coefficients.get("grouped_holdout_residual", {})
        low_factor = max(0.0, 1.0 - float(residual.get("p90_relative", 0.0)))
        high_factor = 1.0 + float(residual.get("p90_relative", 0.0))
        uncertainty = {
            "p10": (
                action_logic_low_pj * low_factor
                + selected_clock_pj
                + sram_dynamic_pj
                + leakage_pj
                + sram_background_pj
            )
            * 1e-9,
            "p50": total_pj * 1e-9,
            "p90": (
                action_logic_high_pj * high_factor
                + selected_clock_pj
                + sram_dynamic_pj
                + leakage_pj
                + sram_background_pj
            )
            * 1e-9,
        }
        ungated_uncertainty = {
            "p10": (
                action_logic_low_pj * low_factor
                + ungated_clock_pj
                + sram_dynamic_pj
                + leakage_pj
                + sram_background_pj
            )
            * 1e-9,
            "p50": ungated_total_pj * 1e-9,
            "p90": (
                action_logic_high_pj * high_factor
                + ungated_clock_pj
                + sram_dynamic_pj
                + leakage_pj
                + sram_background_pj
            )
            * 1e-9,
        }
    else:
        relative = coefficients["uncertainty_relative"]
        uncertain_total_pj = total_pj - sram_background_pj
        uncertain_ungated_total_pj = ungated_total_pj - sram_background_pj
        uncertainty = {
            name: (
                uncertain_total_pj * float(relative[name])
                + sram_background_pj
            )
            * 1e-9
            for name in ("p10", "p50", "p90")
        }
        ungated_uncertainty = {
            name: (
                uncertain_ungated_total_pj * float(relative[name])
                + sram_background_pj
            )
            * 1e-9
            for name in ("p10", "p50", "p90")
        }
    input_tokens = int(cfg.get("INPUT_TOKENS", cfg.get("input_tokens", 0)))
    if not input_tokens:
        input_tokens = int(cfg.get("SEQ_LEN", cfg.get("seq_len", 1))) * int(
            cfg.get("BATCH_SIZE", cfg.get("batch_size", 1))
        )
    calibration_status = str(coefficients.get("calibration_status", "unknown"))
    agu_actions = [
        action for action in actions if str(action.get("component")) == "agu"
    ]
    agu_artifact = _load_agu_coefficients()
    agu_power_status = (
        str(agu_artifact.get("calibration_status", "agu_power_calibration_pending"))
        if agu_actions
        else "not_applicable"
    )
    if calibration_status in {"rtl_activity_calibrated_candidate", "rtl_activity_calibrated_candidate_v2"}:
        warnings.append(
            "logic energy is an RTL-activity/mapped-DC candidate; gate-level activity validation was not run"
        )
        if model_version == "onchip_action_energy_v2":
            warnings.append(
                "HBM controller energy is calibrated per accepted logical lane; separate physical-line/byte terms were not identifiable from fixed-amount points"
            )
    elif calibration_status not in {"vcd_dc_validated", "gate_level_validated"}:
        warnings.append(
            "logic dynamic coefficients are bootstrap priors pending VCD-annotated mapped-DC calibration"
        )
    if unknown_actions:
        warnings.append("one or more emitted action families lack energy coefficients")
    v5_compact_artifact = _load_vector_rtl_v5_energy()
    v5_compact_calibrated = str(
        v5_compact_artifact.get("calibration_status", "")
    ).startswith("rtl_activity_calibrated")
    compact_stats_power_status = (
        "not_applicable"
        if not compact_stats_actions
        else "rtl_activity_calibrated_rtl_v5_tiers"
        if int(cfg.get("COMPACT_STATS_LANES", 16)) > 16
        and v5_compact_calibrated
        else "rtl_activity_per_lane_extrapolation_32_64"
        if int(cfg.get("COMPACT_STATS_LANES", 16)) > 16
        else "rtl_activity_calibrated"
        if not any(
            f"vector.{family}" in unknown_actions
            for family in compact_stats_actions
        )
        else "rtl_v4_power_calibration_pending"
    )
    rtl_v6_artifact = _load_vector_rtl_v6_energy()
    rtl_v6_calibrated = (
        rtl_v6_artifact.get("calibration_status")
        == "rtl_activity_calibrated_rtl_v6_logic"
    )
    rtl_v6_power_status = (
        "not_applicable"
        if not rtl_v6_action_components
        else "rtl_activity_calibrated_rtl_v6_logic"
        if rtl_v6_calibrated
        else "structural_nonzero_proxy_pending_paired_rtl_activity"
    )
    if compact_stats_power_status == "rtl_v4_power_calibration_pending":
        warnings.append(
            "compact-stat SIMD dynamic energy is excluded pending dedicated "
            "RTL-activity/mapped-DC replay; it is not approximated as a full-width vector op"
        )
    elif compact_stats_power_status == "rtl_activity_per_lane_extrapolation_32_64":
        warnings.append(
            "rtl-v5 compact-stat dynamic energy scales the calibrated 16-lane "
            "per-active-lane coefficient to 32/64 lanes; dedicated Qwen-like "
            "RTL-activity replay for those tiers remains pending"
        )
    if rtl_v6_action_components and not rtl_v6_calibrated:
        warnings.append(
            "rtl-v6 softmax-row/state/PV dynamic energy uses nonzero "
            "structural RTL-v5 primitive coefficients; dedicated paired "
            "RTL-activity calibration remains pending"
        )
    elif rtl_v6_action_components:
        warnings.append(
            "rtl-v6 softmax-row/state/PV logic energy uses module-level "
            "RTL-activity/mapped-DC calibration; SRAM energy remains a "
            "separate ASAP7 Liberty charge and CTS/routing are excluded"
        )
    if agu_actions and not agu_artifact:
        warnings.append(
            "AGU dynamic energy uses a nonzero frontend-derived proxy pending "
            "paired RTL-activity calibration"
        )
    if clock_work_error is not None:
        warnings.append(clock_work_error)
    if not actions:
        warnings.append("CostTrace contains no EnergyAction records")
    unmodeled_clock_residual_area_um2 = float(
        (area_metrics.get("area_breakdown") or {}).get(
            "FullChipTopResidual", 0.0
        )
    )
    if normalized_clock_mode == "ideal_hierarchical":
        warnings.append(
            "ideal hierarchical clock gating is an architectural lower-bound "
            "assumption; the current RTL does not implement or validate these gates"
        )
    warnings.append(
        "SRAM background power uses the 10 W/GB lower endpoint reported by "
        "MemExplorer; public ASAP7 SRAM Liberty files do not provide intrinsic "
        "macro leakage"
    )
    return {
        "power_model": model_version,
        "power_scope": "logic+sram_dynamic+sram_background+onchip_hbm_controller",
        "compute_timing_mode": str(
            timing.get("compute_timing_mode", "legacy")
        ),
        "compute_timing_status": (
            "architectural_ideal_assumption"
            if timing.get("compute_timing_mode") == "ideal-ii1"
            else "rtl_calibrated"
            if timing.get("compute_timing_mode") == "rtl-v1"
            else "legacy"
        ),
        "compute_hazards_included": (
            timing.get("compute_timing_mode") == "rtl-v1"
        ),
        "onchip_power_semantics": (
            "logic_leakage_plus_sram_background_plus_dynamic_ideal_clock_lower_bound"
            if normalized_clock_mode == "ideal_hierarchical"
            else "logic_leakage_plus_sram_background_plus_dynamic_ungated_clock_upper_bound"
        ),
        "corner": dict(coefficients["corner"]),
        "calibration_status": calibration_status,
        "agu_power_calibration_status": agu_power_status,
        "agu_power_model": agu_artifact.get("model_version"),
        "agu_power_validation": dict(agu_artifact.get("validation", {})),
        "compact_stats_power_calibration_status": compact_stats_power_status,
        "rtl_v6_power_calibration_status": rtl_v6_power_status,
        "rtl_v6_power_model": rtl_v6_artifact.get("schema_version"),
        "rtl_v6_power_validation": dict(
            rtl_v6_artifact.get("validation", {})
        ),
        "calibration_coverage": {
            "energy_action_count": len(actions),
            "dynamic_action_instances": sum(
                float(action["count"]) for action in actions
            ),
            "unknown_actions": dict(unknown_actions),
            "structurally_zero_actions": dict(structurally_zero_actions),
            "structural_physical_instances": dict(
                structural_physical_instances
            ),
            "sram_macro_count": int(catalog.get("macro_count", 0)),
            "sram_energy_status": "liberty_internal_power",
            "sram_background_status": sram_background_artifact.get(
                "calibration_status"
            ),
            "gate_level_validation": coefficients.get("gate_level_validation", "unavailable"),
            "agu_power_calibration": agu_power_status,
        },
        "logic_dynamic_energy_mj": logic_dynamic_pj * 1e-9,
        "action_logic_dynamic_energy_mj": action_logic_pj * 1e-9,
        "clock_energy_mj": selected_clock_pj * 1e-9,
        "ideal_clock_energy_mj": ideal_clock_pj * 1e-9,
        "ungated_clock_energy_mj": ungated_clock_pj * 1e-9,
        "sram_dynamic_energy_mj": sram_dynamic_pj * 1e-9,
        "sram_background_energy_mj": sram_background_pj * 1e-9,
        "sram_leakage_energy_mj": sram_background_pj * 1e-9,
        "logic_leakage_energy_mj": leakage_pj * 1e-9,
        "onchip_energy_mj": total_pj * 1e-9,
        "onchip_average_power_w": total_pj / makespan_ns * 1e-3,
        "ungated_onchip_energy_mj": ungated_total_pj * 1e-9,
        "ungated_onchip_average_power_w": (
            ungated_total_pj / makespan_ns * 1e-3
        ),
        "ungated_onchip_energy_p10_mj": ungated_uncertainty["p10"],
        "ungated_onchip_energy_p50_mj": ungated_uncertainty["p50"],
        "ungated_onchip_energy_p90_mj": ungated_uncertainty["p90"],
        "ungated_onchip_average_power_p10_w": (
            ungated_uncertainty["p10"] * 1e6 / makespan_ns
        ),
        "ungated_onchip_average_power_p50_w": (
            ungated_uncertainty["p50"] * 1e6 / makespan_ns
        ),
        "ungated_onchip_average_power_p90_w": (
            ungated_uncertainty["p90"] * 1e6 / makespan_ns
        ),
        "clock_energy_savings_pct": (
            None
            if clock_work.get("status") == "clock_work_unavailable"
            else 0.0
            if ungated_clock_pj <= 0
            else 100.0
            * (ungated_clock_pj - ideal_clock_pj)
            / ungated_clock_pj
        ),
        "clock_gating_mode": normalized_clock_mode,
        "clock_gating_status": (
            "architectural_ideal_assumption"
            if normalized_clock_mode == "ideal_hierarchical"
            else "ungated_upper_bound"
        ),
        "idle_clock_fraction": (
            0.0 if normalized_clock_mode == "ideal_hierarchical" else 1.0
        ),
        "gating_overhead_included": False,
        "rtl_clock_gating_implemented": False,
        "clock_active_fraction_by_component": clock_active_fraction,
        "clock_work_by_subcomponent": clock_work.get(
            "capped_equivalent_full_area_cycles", {}
        ),
        "clock_energy_by_component_pj": dict(clock_breakdown),
        "ungated_clock_energy_by_component_pj": dict(
            ungated_clock_breakdown
        ),
        "ideal_clock_energy_by_component_pj": dict(ideal_clock_breakdown),
        "ideal_clock_energy_by_subcomponent_pj": ideal_clock_by_subcomponent,
        "clock_work": clock_work,
        "unmodeled_clock_residual_area_um2": (
            unmodeled_clock_residual_area_um2
        ),
        "onchip_energy_p10_mj": uncertainty["p10"],
        "onchip_energy_p50_mj": uncertainty["p50"],
        "onchip_energy_p90_mj": uncertainty["p90"],
        "onchip_average_power_p10_w": uncertainty["p10"] * 1e6 / makespan_ns,
        "onchip_average_power_p50_w": uncertainty["p50"] * 1e6 / makespan_ns,
        "onchip_average_power_p90_w": uncertainty["p90"] * 1e6 / makespan_ns,
        "energy_per_input_token_mj": total_pj * 1e-9 / max(1, input_tokens),
        "area_used_for_logic_leakage_um2": logic_area_um2,
        "logic_leakage_power_mw": leakage_mw,
        "sram_background_power_w": sram_background_power_w,
        "sram_leakage_power_w": sram_background_power_w,
        "sram_allocated_capacity_gb": sram_allocated_capacity_gb,
        "sram_background_power_w_per_gb": float(
            sram_background_artifact["selected_background_power_w_per_gb"]
        ),
        "sram_background_energy_by_component_pj": (
            sram_background_breakdown_pj
        ),
        "sram_background_model": sram_background_artifact["model"],
        "sram_background_calibration_status": sram_background_artifact[
            "calibration_status"
        ],
        "sram_background_source": dict(sram_background_artifact["source"]),
        "sram_leakage_status": "literature_parameterized_background_proxy",
        "asap7_sram_leakage_status": "unavailable_from_public_liberty",
        "makespan_cycles": cycles,
        "makespan_ns": makespan_ns,
        "makespan_source": makespan_source,
        "stage_logic_dynamic_energy_pj": dict(stage_logic),
        "component_logic_dynamic_energy_pj": dict(component_logic),
        "action_logic_dynamic_energy_by_action_pj": dict(
            action_logic_by_action
        ),
        "action_active_elements_by_action": dict(
            action_active_elements_by_action
        ),
        "action_active_rows_by_action": dict(action_active_rows_by_action),
        "component_sram_dynamic_energy_pj": sram_breakdown,
        "sram_access_metadata": sram_metadata,
        "uncertainty_energy_mj": uncertainty,
        "uncertainty_power_w": {
            name: value * 1e6 / makespan_ns for name, value in uncertainty.items()
        },
        "warnings": list(dict.fromkeys(warnings)),
        "excludes": [
            "external_hbm",
            "phy",
            "package",
            "kv_link",
            "cts",
            "asap7_macro_intrinsic_leakage_calibration",
        ],
        "pre_cts": True,
        "trace_schema_version": trace.get("schema_version"),
    }


__all__ = ["estimate_onchip_power"]
