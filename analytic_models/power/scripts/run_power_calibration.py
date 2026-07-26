#!/usr/bin/env python3
# ruff: noqa: E402
"""Low-cost mapped-DC activity calibration for on-chip action energy.

Exactly sixteen small RTL configurations are mapped once.  Multiple VCD
scenarios are then replayed against each mapped DDC, so changing activity does
not consume another synthesis license or create another large build tree.

The runner does not invent stimulus.  ``--activity-command-template`` must
invoke a Verilator/cocotb driver that writes the requested VCD, or a complete
``--vcd-root`` may be supplied.  This makes missing gate/activity collateral a
hard, recorded failure instead of silently falling back to vectorless power.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytic_models.area_new.scripts.calibration_csv import append_row
from analytic_models.area_new.scripts.calibration_runtime import (
    CalibrationJob,
    CompactExport,
    RuntimeConfig,
    run_calibration_jobs,
    stable_job_key,
)
from analytic_models.area_new.scripts.license_utils import resolve_dc_worker_count
from analytic_models.area_new.scripts.license_utils import is_dc_license_unavailable_text
from analytic_models.area_new.scripts.run_matrix_machine_calibration import (
    mxfp_mini_wrapper,
    mxfp_reduce_wrapper,
    mxint_acc_to_fp_wrapper,
    mxint_mini_wrapper,
    mxint_reduce_wrapper,
    replace_localparam,
)
from analytic_models.area_new.scripts.run_vector_machine_calibration import (
    Point as VectorPoint,
    patch_vector_config,
)
from analytic_models.area_new.scripts.run_scalar_machine_calibration import (
    Point as ScalarPoint,
    patch_scalar_config,
)
from analytic_models.area_new.scripts.run_hbm_system_calibration import (
    Point as HbmPoint,
    patch_hbm_config,
    patch_tilelink_upsizer_for_large_widths,
)
from analytic_models.power.sram_energy import DEFAULT_CATALOG, build_sram_energy_catalog

DEFAULT_RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
DEFAULT_WORKER_ROOT = Path("/tmp/plena_rtl_power_workers_v1")
GIB = 1024**3

SCENARIOS = (
    ("idle_32", "idle", 32),
    ("random_32", "random", 32),
    ("idle_128", "idle", 128),
    ("random_128", "random", 128),
    ("idle_512", "idle", 512),
    ("random_512", "random", 512),
    ("low_toggle_128", "low-toggle", 128),
    ("qwen_128", "representative-qwen", 128),
    ("mixed_holdout_128", "mixed-kernel-holdout", 128),
)

ACTIVITY_FIELDS = [
    "point_id", "point_key", "component", "scenario", "pattern", "microkernel",
    "repeat_count", "status", "window_ns", "switching_power_mw",
    "internal_power_mw", "dynamic_power_mw", "leakage_power_mw",
    "window_dynamic_energy_pj", "incremental_energy_pj", "vcd_path",
    "power_report", "activity_log", "activity_level", "features_json",
    "logic_area_um2", "holdout", "pwr_414", "pwr_415",
    "failure_reason",
]


@dataclass(frozen=True)
class PowerPoint:
    point_id: str
    component: str
    module: str
    top_module: str
    params: dict[str, Any]
    holdout: bool = False
    point_key: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "component": self.component,
            "module": self.module,
            "top_module": self.top_module,
            "params": self.params,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        object.__setattr__(self, "point_key", f"power_{self.component}_{digest}")


def build_plan() -> list[PowerPoint]:
    """Return the fixed 16-netlist v1 mapping plan."""

    points: list[PowerPoint] = []
    matrix = [
        ("mxint_b4_t4_l2", "mxint", 4, 2, None),
        ("mxint_b4_t4_l4", "mxint", 4, 4, None),
        ("mxint_b4_t8_l8", "mxint", 8, 8, None),
        ("mxfp_b4_e1m2", "mxfp", None, None, (1, 2)),
        ("mxfp_b4_e4m3", "mxfp", None, None, (4, 3)),
        ("mxint_b8_t4_l4_holdout", "mxint", 4, 4, None),
    ]
    for name, mode, t_bits, l_bits, fp in matrix:
        block = 8 if "b8" in name else 4
        params: dict[str, Any] = {"mode": mode, "BLOCK_DIM": block, "ACC_DEPTH": 16}
        if mode == "mxint":
            params.update({"T_BITS": t_bits, "L_BITS": l_bits})
        else:
            assert fp is not None
            params.update({"T_EXP": fp[0], "T_MANT": fp[1], "L_EXP": fp[0], "L_MANT": fp[1]})
        points.append(
            PowerPoint(
                point_id=f"power_matrix_{name}",
                component="matrix",
                module="mx_mini_systolic_array",
                top_module=f"power_matrix_{name}",
                params=params,
                holdout="holdout" in name,
            )
        )
    points.append(
        PowerPoint(
            "power_matrix_leaf_bundle", "matrix", "matrix_leaf_bundle",
            "power_matrix_leaf_bundle", {"leaf_bundle": True}, holdout=True,
        )
    )
    for vlen, exp, mant, holdout in ((16, 6, 5, False), (32, 6, 5, False), (64, 6, 5, False), (32, 8, 7, True)):
        points.append(
            PowerPoint(
                f"power_vector_v{vlen}_e{exp}m{mant}", "vector", "vector_machine",
                "vector_machine", {"VLEN": vlen, "V_FP_EXP_WIDTH": exp, "V_FP_MANT_WIDTH": mant}, holdout,
            )
        )
    for int_width, exp, mant, holdout in ((32, 6, 5, False), (64, 8, 7, True)):
        points.append(
            PowerPoint(
                f"power_scalar_i{int_width}_e{exp}m{mant}", "scalar", "scalar_machine",
                "scalar_machine", {"MLEN": 32, "VLEN": 32, "INT_DATA_WIDTH": int_width, "S_FP_EXP_WIDTH": exp, "S_FP_MANT_WIDTH": mant}, holdout,
            )
        )
    for mode, precision, holdout in (("mxint", "MXINT4", False), ("mxfp", "MXFP_E4M3", True)):
        points.append(
            PowerPoint(
                f"power_hbm_{mode}", "hbm", "hbm_sys", "hbm_sys",
                {
                    "MLEN": 32, "VLEN": 32, "BLEN": 8, "BLOCK_DIM": 8,
                    "ACT_WIDTH": precision, "KV_WIDTH": precision,
                    "WEIGHT_WIDTH": precision, "MX_SCALE_WIDTH": 8,
                    "HBM_M_Prefetch_Amount": 32, "HBM_V_Prefetch_Amount": 8,
                    "HBM_V_Writeback_Amount": 8,
                }, holdout,
            )
        )
    points.append(
        PowerPoint(
            "power_frontend_control", "control", "pipeline_control",
            "pipeline_control", {"MLEN": 32, "VLEN": 32, "BLEN": 8, "HLEN": 8}, False,
        )
    )
    if len(points) != 16:
        raise AssertionError(f"power calibration plan must contain 16 points, got {len(points)}")
    return points


def build_plan_v2() -> list[PowerPoint]:
    """Return the 31-netlist v2 plan, reusing all sixteen v1 configurations."""

    points = []
    for point in build_plan():
        holdout = point.holdout
        if point.point_id == "power_matrix_mxint_b8_t4_l4_holdout":
            holdout = False
        elif point.point_id == "power_vector_v64_e6m5":
            holdout = True
        points.append(replace(point, holdout=holdout))

    matrix_specs = (
        ("mxint_b2_t4_l4_v2", "mxint", 2, 4, 4, None, None, False),
        ("mxint_b16_t4_l4_v2", "mxint", 16, 4, 4, None, None, True),
        ("mxint_b4_t8_l2_v2", "mxint", 4, 8, 2, None, None, False),
        ("mxint_b4_t4_l8_v2", "mxint", 4, 4, 8, None, None, True),
        ("mxfp_b4_e2m1_v2", "mxfp", 4, None, None, (2, 1), (2, 1), False),
        ("mxfp_b4_e5m2_v2", "mxfp", 4, None, None, (5, 2), (5, 2), True),
        ("mxfp_b4_t_e4m3_l_e1m2_v2", "mxfp", 4, None, None, (4, 3), (1, 2), False),
        ("mxfp_b4_t_e1m2_l_e4m3_v2", "mxfp", 4, None, None, (1, 2), (4, 3), True),
    )
    for name, mode, block, t_bits, l_bits, t_fp, l_fp, holdout in matrix_specs:
        params: dict[str, Any] = {"mode": mode, "BLOCK_DIM": block, "ACC_DEPTH": 16}
        if mode == "mxint":
            params.update({"T_BITS": t_bits, "L_BITS": l_bits})
        else:
            assert t_fp is not None and l_fp is not None
            params.update(
                {
                    "T_EXP": t_fp[0], "T_MANT": t_fp[1],
                    "L_EXP": l_fp[0], "L_MANT": l_fp[1],
                }
            )
        points.append(
            PowerPoint(
                f"power_matrix_{name}", "matrix", "mx_mini_systolic_array",
                f"power_matrix_{name}", params, holdout,
            )
        )

    for exp, mant, holdout in ((5, 6, False), (8, 5, True)):
        points.append(
            PowerPoint(
                f"power_vector_v32_e{exp}m{mant}_v2", "vector", "vector_machine",
                "vector_machine", {"VLEN": 32, "V_FP_EXP_WIDTH": exp, "V_FP_MANT_WIDTH": mant},
                holdout,
            )
        )

    scalar_specs = (
        (16, 6, 5, False),
        (64, 6, 5, False),
        (32, 5, 6, False),
        (32, 8, 5, False),
        (64, 8, 5, True),
    )
    for int_width, exp, mant, holdout in scalar_specs:
        points.append(
            PowerPoint(
                f"power_scalar_i{int_width}_e{exp}m{mant}_v2", "scalar", "scalar_machine",
                "scalar_machine",
                {
                    "MLEN": 32, "VLEN": 32, "INT_DATA_WIDTH": int_width,
                    "S_FP_EXP_WIDTH": exp, "S_FP_MANT_WIDTH": mant,
                },
                holdout,
            )
        )
    if len(points) != 31:
        raise AssertionError(f"power calibration v2 plan must contain 31 points, got {len(points)}")
    return points


def build_agu_plan() -> list[PowerPoint]:
    """Return the standalone six-stream AGU activity calibration point."""

    return [
        PowerPoint(
            "power_loop_agu_v1",
            "agu",
            "loop_agu_state",
            "loop_agu_state",
            {
                "INT_DATA_WIDTH": 32,
                "stream_count": 6,
                "loop_depth": 4,
                "gp_register_count": 16,
            },
            False,
        )
    ]


VECTOR_V2_MICROKERNELS = (
    "add_vv", "add_vf", "add_vseg", "mul_vv", "mul_vf", "mul_vseg",
    "exp", "reciprocal", "reduce_sum", "reduce_max", "reduce_sum_seg",
    "reduce_max_seg", "reduce_sum_segs", "reduce_max_segs", "shift",
    "lane_load", "lane_store", "compact_stats_mul", "compact_stats_add",
    "compact_stats_rsqrt", "reduce_sum_ovr", "reduce_max_ovr",
    "reduce_sum_seg_ovr", "reduce_max_seg_ovr",
)
SCALAR_V2_MICROKERNELS = (
    "fp_alu", "fp_mul", "fp_exp", "fp_reciprocal", "fp_sqrt", "fp_rsqrt",
    "int_alu", "int_mul", "register_access",
)
HBM_V2_MICROKERNELS = ("matrix_prefetch", "vector_prefetch", "vector_writeback")


def _microkernel_scenarios(
    microkernels: tuple[str, ...],
    *,
    representative: bool,
) -> list[tuple[str, str, int, str]]:
    scenarios: list[tuple[str, str, int, str]] = [("idle_128", "idle", 128, "idle")]
    for microkernel in microkernels:
        scenarios.append((f"qwen_{microkernel}_128", "representative-qwen", 128, microkernel))
    scenarios.append(("qwen_mix_128", "representative-qwen", 128, "mixed"))
    if representative:
        scenarios.extend((("idle_32", "idle", 32, "idle"), ("idle_512", "idle", 512, "idle")))
        for microkernel in microkernels:
            scenarios.extend(
                (
                    (f"qwen_{microkernel}_32", "representative-qwen", 32, microkernel),
                    (f"qwen_{microkernel}_512", "representative-qwen", 512, microkernel),
                    (f"low_{microkernel}_128", "low-toggle", 128, microkernel),
                    (f"random_{microkernel}_128", "random", 128, microkernel),
                )
            )
    return scenarios


def scenarios_for_point_v2(point: PowerPoint) -> list[tuple[str, str, int, str]]:
    """Build the deterministic per-component v2 activity scenario manifest."""

    if point.component == "agu":
        return [
            ("idle_32", "idle", 32, "idle"),
            ("idle_128", "idle", 128, "idle"),
            ("idle_512", "idle", 512, "idle"),
            ("qwen_boundary_6_32", "representative-qwen", 32, "boundary_6"),
            ("qwen_boundary_6_128", "representative-qwen", 128, "boundary_6"),
            ("qwen_boundary_6_512", "representative-qwen", 512, "boundary_6"),
            ("qwen_boundary_1_128", "representative-qwen", 128, "boundary_1"),
            ("qwen_boundary_3_128", "representative-qwen", 128, "boundary_3"),
            ("qwen_offset_read_128", "representative-qwen", 128, "offset_read"),
            ("qwen_setup_1_128", "representative-qwen", 128, "setup_1"),
            ("qwen_setup_6_128", "representative-qwen", 128, "setup_6"),
            ("low_boundary_6_128", "low-toggle", 128, "boundary_6"),
            ("random_boundary_6_128", "random", 128, "boundary_6"),
        ]
    if point.component == "matrix":
        microkernels = ("leaf_bundle",) if point.params.get("leaf_bundle") else ("array_compute",)
        representative = point.point_id in {
            "power_matrix_mxint_b4_t4_l4",
            "power_matrix_mxfp_b4_e4m3",
            "power_matrix_leaf_bundle",
        }
    elif point.component == "vector":
        microkernels = VECTOR_V2_MICROKERNELS
        representative = point.point_id == "power_vector_v32_e6m5"
    elif point.component == "scalar":
        microkernels = SCALAR_V2_MICROKERNELS
        representative = point.point_id == "power_scalar_i32_e6m5"
    elif point.component == "hbm":
        microkernels = HBM_V2_MICROKERNELS
        representative = point.point_id == "power_hbm_mxint"
    else:
        microkernels = ("frontend_issue",)
        representative = True
    return _microkernel_scenarios(microkernels, representative=representative)


def _write_plan(
    points: list[PowerPoint],
    run_dir: Path,
    *,
    scenarios_by_point: dict[str, list[tuple[str, str, int, str]]] | None = None,
) -> None:
    plan_dir = run_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    fields = ["point_id", "point_key", "component", "module", "top_module", "holdout", "params_json"]
    with (plan_dir / "mapped_netlists.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow({
                "point_id": point.point_id, "point_key": point.point_key,
                "component": point.component, "module": point.module,
                "top_module": point.top_module, "holdout": int(point.holdout),
                "params_json": json.dumps(point.params, sort_keys=True),
            })
    with (plan_dir / "activity_scenarios.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "point_key", "scenario", "pattern", "repeat_count", "microkernel"])
        for point in points:
            scenarios = (
                scenarios_by_point[point.point_key]
                if scenarios_by_point is not None
                else [(name, pattern, repeats, "mixed") for name, pattern, repeats in SCENARIOS]
            )
            for scenario, pattern, repeat_count, microkernel in scenarios:
                writer.writerow(
                    [
                        point.point_id,
                        point.point_key,
                        scenario,
                        pattern,
                        repeat_count,
                        microkernel,
                    ]
                )
    contract = {
        "required_placeholders": [
            "worker_rtl", "point_id", "top_module", "scenario", "pattern",
            "repeat_count", "microkernel", "vcd", "mapped_netlist", "sdf", "holdout",
        ],
        "requirements": [
            "Generate RTL-simulation VCD using Verilator/cocotb.",
            "VCD must include clk, primary inputs, and sequential outputs under the configured strip path.",
            "Idle and random scenarios with the same repeat_count must have the same simulation window.",
            "Write a sibling .vcd.actions.json with dynamic_features and clock_features in coefficient units.",
        ],
        "actions_sidecar_schema": {
            "dynamic_features": {"vector.lane_add_sub_bit": "nonnegative feature count"},
            "clock_features": {"vector_lane": "clock cycles times instantiated lanes"},
        },
    }
    (plan_dir / "activity_generator_contract.json").write_text(json.dumps(contract, indent=2) + "\n")


def _active_tmp_paths() -> set[Path]:
    active: set[Path] = set()
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cwd = (entry / "cwd").resolve()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if "dc_shell" in command or "run_power_calibration" in command:
            active.add(cwd)
    return active


def _path_is_active(path: Path, active: set[Path]) -> bool:
    resolved = path.resolve()
    return any(cwd == resolved or resolved in cwd.parents or cwd in resolved.parents for cwd in active)


def tmp_preflight(workers: int, worker_root: Path) -> int:
    """Enforce quota headroom and lower concurrency when cleanup is insufficient."""

    def free_bytes() -> int:
        return shutil.disk_usage("/tmp").free

    required = 10 * GIB + 6 * GIB * workers
    if free_bytes() < required:
        active = _active_tmp_paths()
        for candidate in sorted(Path("/tmp").glob("plena_rtl_power_workers_v1*")):
            if candidate == worker_root or _path_is_active(candidate, active):
                continue
            shutil.rmtree(candidate, ignore_errors=True)
        for candidate in sorted(Path("/tmp").glob("area_new_power_*")):
            if _path_is_active(candidate, active):
                continue
            shutil.rmtree(candidate, ignore_errors=True)
    available = free_bytes()
    if available < 15 * GIB:
        raise RuntimeError(f"/tmp has only {available / GIB:.1f} GiB free; at least 15 GiB is required")
    allowed = max(1, int((available - 10 * GIB) // (6 * GIB)))
    return min(workers, allowed)


def _matrix_leaf_bundle() -> str:
    int_top = "power_leaf_int_reduce"
    fp_top = "power_leaf_fp_reduce"
    cvt_top = "power_leaf_int_to_fp"
    children = "\n".join([
        mxint_reduce_wrapper(int_top, 16, 2, 2),
        mxfp_reduce_wrapper(fp_top, 8, 7, 2, 2),
        mxint_acc_to_fp_wrapper(cvt_top, 24, 8, 5, 6),
    ])
    # The child ports are retained by exposing representative inputs and all
    # outputs.  This bundle is a mapping convenience, not a functional block.
    top = f"""
module power_matrix_leaf_bundle(
    input logic clk, input logic rst, input logic in_valid,
    input logic [1:0][1:0][1:0][15:0] int_data,
    input logic [1:0][1:0][1:0][15:0] fp_data,
    input logic signed [23:0] acc_in, input logic [8:0] scale_in,
    output logic [1:0][1:0][33:0] int_out,
    output logic [1:0][1:0][8:0] int_scale,
    output logic int_valid,
    output logic [1:0][1:0][15:0] fp_out,
    output logic fp_valid,
    output logic [11:0] converted
);
  logic [1:0][1:0][1:0][8:0] scales;
  assign scales = '0;
  {int_top} u_int(.clk(clk), .rst(rst), .m_in_int(int_data), .m_in_scale(scales),
    .in_valid(in_valid), .m_out_int(int_out), .m_out_scale(int_scale), .out_valid(int_valid));
  {fp_top} u_fp(.clk(clk), .rst(rst), .m_in_data(fp_data), .in_valid(in_valid),
    .m_out_data(fp_out), .out_valid(fp_valid));
  {cvt_top} u_cvt(.acc_in(acc_in), .scale_in(scale_in), .fp_out(converted));
endmodule
"""
    return children + "\n" + top


def _prepare_point(point: PowerPoint, worker_rtl: Path) -> None:
    if point.component == "matrix":
        path = worker_rtl / "src/basic_components/systolic_gemm_mx/rtl" / f"{point.top_module}.sv"
        if point.params.get("leaf_bundle"):
            text = _matrix_leaf_bundle()
        elif point.params["mode"] == "mxint":
            path = worker_rtl / "src/basic_components/systolic_gemm_mxint/rtl" / f"{point.top_module}.sv"
            text = mxint_mini_wrapper(
                point.top_module, int(point.params["T_BITS"]), int(point.params["L_BITS"]),
                int(point.params["BLOCK_DIM"]), int(point.params["ACC_DEPTH"]),
            )
        else:
            text = mxfp_mini_wrapper(
                point.top_module, int(point.params["T_EXP"]), int(point.params["T_MANT"]),
                int(point.params["L_EXP"]), int(point.params["L_MANT"]), int(point.params["BLOCK_DIM"]),
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    elif point.component == "vector":
        patch_vector_config(VectorPoint(point.point_id, point.module, point.top_module, point.params), worker_rtl)
    elif point.component == "scalar":
        patch_scalar_config(ScalarPoint(point.point_id, point.module, point.top_module, point.params), worker_rtl)
    elif point.component == "hbm":
        patch_hbm_config(HbmPoint(point.point_id, point.module, point.top_module, point.params), worker_rtl)
        patch_tilelink_upsizer_for_large_widths(worker_rtl)
    elif point.component == "control":
        configuration = worker_rtl / "src/definitions/configuration.svh"
        for name in ("MLEN", "VLEN", "BLEN", "HLEN"):
            replace_localparam(configuration, name, int(point.params[name]))


def _vcd_window_ns(path: Path) -> float:
    text = path.read_text(errors="ignore")
    unit_match = re.search(r"\$timescale\s+(\d+)\s*(fs|ps|ns|us)", text, re.I)
    scale = {"fs": 1e-6, "ps": 1e-3, "ns": 1.0, "us": 1e3}
    tick_ns = float(unit_match.group(1)) * scale[unit_match.group(2).lower()] if unit_match else 1e-3
    timestamps = re.findall(r"(?m)^#(\d+)\s*$", text)
    if not timestamps:
        raise ValueError(f"{path} has no VCD timestamps")
    return max(1e-9, int(timestamps[-1]) * tick_ns)


def _power_value(text: str, label: str) -> float:
    match = re.search(rf"{re.escape(label)}\s*=\s*([0-9.eE+-]+)\s*([munp]?W)", text)
    if match is None:
        raise ValueError(f"power report has no {label}")
    factor_mw = {"W": 1000.0, "mW": 1.0, "uW": 1e-3, "nW": 1e-6, "pW": 1e-9}
    return float(match.group(1)) * factor_mw[match.group(2)]


def _optional_power_value(text: str, label: str) -> float | None:
    try:
        return _power_value(text, label)
    except ValueError:
        return None


def _linear_r2(samples: list[tuple[float, float]]) -> float | None:
    if len(samples) < 3:
        return None
    mean_x = sum(x for x, _ in samples) / len(samples)
    mean_y = sum(y for _, y in samples) / len(samples)
    variance_x = sum((x - mean_x) ** 2 for x, _ in samples)
    if variance_x == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in samples) / variance_x
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in samples)
    total = sum((y - mean_y) ** 2 for _, y in samples)
    return 1.0 if total == 0 and residual == 0 else 1.0 - residual / total if total else None


def _activity_tcl(ddc: Path, vcd: Path, top: str, strip_path: str, report: Path) -> str:
    return f"""read_ddc {{{ddc}}}
current_design {top}
link
read_vcd -strip_path {{{strip_path}}} {{{vcd}}}
update_power
report_power -hierarchy > {{{report}}}
report_switching_activity -list_not_annotated > {{{report.with_suffix('.coverage.rpt')}}}
exit
"""


class PowerAdapter:
    name = "power"
    row_fields: ClassVar[list[str]] = [
        "component", "holdout", "mapped_ddc", "mapped_netlist", "sdf",
        "sdc", "wns_ns", "timing_status", "activity_complete", "activity_r2",
    ]

    def __init__(self, activity_template: str | None, gate_level_template: str | None, vcd_root: Path | None, strip_path: str):
        self.activity_template = activity_template
        self.gate_level_template = gate_level_template
        self.vcd_root = vcd_root
        self.strip_path = strip_path
        self.csv_lock = threading.Lock()

    def compact_exports(self) -> list[CompactExport]:
        return []

    def _generate_vcd(self, point: PowerPoint, worker_rtl: Path, scenario: str, pattern: str, repeat_count: int, destination: Path, mapped_netlist: Path | None, sdf: Path | None) -> str:
        if self.vcd_root is not None:
            source = self.vcd_root / point.point_id / f"{scenario}.vcd"
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                sidecar = source.with_suffix(source.suffix + ".actions.json")
                if sidecar.exists():
                    shutil.copy2(sidecar, destination.with_suffix(destination.suffix + ".actions.json"))
                return "rtl_vcd_mapped_dc"
        template = self.gate_level_template if point.holdout and self.gate_level_template else self.activity_template
        if template is None:
            raise FileNotFoundError(f"missing VCD for {point.point_id}/{scenario}; provide --activity-command-template or --vcd-root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = template.format(
            worker_rtl=shlex.quote(str(worker_rtl)), point_id=shlex.quote(point.point_id),
            top_module=shlex.quote(point.top_module), scenario=shlex.quote(scenario),
            pattern=shlex.quote(pattern), repeat_count=repeat_count, vcd=shlex.quote(str(destination)),
            mapped_netlist=shlex.quote(str(mapped_netlist or "")),
            sdf=shlex.quote(str(sdf or "")), holdout=int(point.holdout),
        )
        result = subprocess.run(["bash", "-lc", command], cwd=worker_rtl, text=True, capture_output=True)
        if result.returncode != 0 or not destination.exists():
            raise RuntimeError(f"activity generator failed ({result.returncode}): {result.stderr[-1000:]}")
        return "gate_level_vcd_mapped_dc" if point.holdout and self.gate_level_template else "rtl_vcd_mapped_dc"

    @staticmethod
    def _run_with_license_retry(
        command: list[str],
        *,
        cwd: Path,
        log_dir: Path,
        log_stem: str,
        worker_id: int,
        point_id: str,
        wait_sec: float,
        max_retries: int,
    ) -> subprocess.CompletedProcess[str]:
        """Run one DC command, retaining every SEC-50 retry attempt."""

        attempt = 0
        while True:
            attempt += 1
            result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / f"{log_stem}.attempt_{attempt}.stdout.log").write_text(result.stdout)
            (log_dir / f"{log_stem}.attempt_{attempt}.stderr.log").write_text(result.stderr)
            (log_dir / f"{log_stem}.stdout.log").write_text(result.stdout)
            (log_dir / f"{log_stem}.stderr.log").write_text(result.stderr)
            text = f"{result.stdout}\n{result.stderr}"
            if result.returncode == 0 or not is_dc_license_unavailable_text(text):
                return result
            if max_retries > 0 and attempt > max_retries:
                return result
            print(
                f"[license-busy] worker={worker_id} point={point_id} "
                f"attempt={attempt}; retrying in {wait_sec:g}s",
                flush=True,
            )
            time.sleep(wait_sec)

    def _run_point_impl(self, point: PowerPoint, worker_id: int, worker_rtl: Path, rtl_root: Path, run_dir: Path, cleanup_builds: bool, license_retry_wait_sec: float, license_max_retries: int) -> dict[str, Any]:
        start = time.time()
        _prepare_point(point, worker_rtl)
        command = ["nix", "develop", "-c", "bash", "-lc", f"cd {shlex.quote(str(worker_rtl))} && just synth {point.top_module} 1000 normal"]
        log_dir = run_dir / "command_logs" / point.point_key
        nix_root = Path(os.environ.get("PLENA_RTL_NIX_ROOT", str(rtl_root)))
        result = self._run_with_license_retry(
            command,
            cwd=nix_root,
            log_dir=log_dir,
            log_stem="synth",
            worker_id=worker_id,
            point_id=point.point_id,
            wait_sec=license_retry_wait_sec,
            max_retries=license_max_retries,
        )
        if result.returncode != 0:
            return {"point_id": point.point_id, "module": point.module, "top_module": point.top_module, "component": point.component, "status": "failed", "worker_id": worker_id, "elapsed_sec": time.time() - start, "failure_reason": result.stderr[-1000:]}
        latest = worker_rtl / "build/synth" / point.top_module / "latest"
        archive = run_dir / "mapped" / point.point_key
        archive.mkdir(parents=True, exist_ok=True)
        copied: dict[str, Path] = {}
        sources = {
            "mapped_ddc": latest / "out" / f"{point.top_module}_mapped.ddc",
            "mapped_netlist": latest / "out" / f"{point.top_module}_mapped.v",
            "sdf": latest / "out" / f"{point.top_module}.sdf",
            "sdc": latest / "out" / f"{point.top_module}.sdc",
            "timing": latest / "reports" / f"{point.top_module}_timing.rpt",
            "qor": latest / "reports" / f"{point.top_module}_qor.rpt",
            "area": latest / "reports" / f"{point.top_module}_area.rpt",
        }
        for name, source in sources.items():
            if source.exists():
                destination = archive / source.name
                shutil.copy2(source, destination)
                copied[name] = destination
        if "mapped_ddc" not in copied:
            raise FileNotFoundError("normal synthesis produced no mapped DDC")
        logic_area_um2 = None
        if "area" in copied:
            match = re.search(
                r"Total cell area:\s*([0-9.eE+-]+)",
                copied["area"].read_text(errors="ignore"),
            )
            logic_area_um2 = float(match.group(1)) if match else None
        scenario_rows: list[dict[str, Any]] = []
        idle_energy: dict[int, float] = {}
        for scenario, pattern, repeat_count in SCENARIOS:
            row = {"point_id": point.point_id, "point_key": point.point_key, "component": point.component, "scenario": scenario, "pattern": pattern, "repeat_count": repeat_count, "holdout": int(point.holdout), "status": "failed"}
            try:
                vcd = run_dir / "activity" / point.point_key / f"{scenario}.vcd"
                activity_level = self._generate_vcd(
                    point, worker_rtl, scenario, pattern, repeat_count, vcd,
                    copied.get("mapped_netlist"), copied.get("sdf"),
                )
                scenario_dir = run_dir / "reports" / point.point_key / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)
                report = scenario_dir / "power.rpt"
                tcl = scenario_dir / "power.tcl"
                tcl.write_text(_activity_tcl(copied["mapped_ddc"], vcd, point.top_module, self.strip_path, report))
                dc_command = "source /mnt/applications/synopsys/2024-25/scripts/SYN_2024.09-SP2_RHELx86.sh && dc_shell -f " + shlex.quote(str(tcl))
                dc = self._run_with_license_retry(
                    ["bash", "-lc", dc_command],
                    cwd=worker_rtl,
                    log_dir=scenario_dir,
                    log_stem="dc_power",
                    worker_id=worker_id,
                    point_id=f"{point.point_id}/{scenario}",
                    wait_sec=license_retry_wait_sec,
                    max_retries=license_max_retries,
                )
                activity_log = scenario_dir / "activity.log"
                activity_log.write_text(dc.stdout + "\n" + dc.stderr)
                if dc.returncode != 0 or not report.exists():
                    raise RuntimeError(f"DC activity analysis failed ({dc.returncode})")
                report_text = report.read_text(errors="ignore")
                window_ns = _vcd_window_ns(vcd)
                dynamic_mw = _power_value(report_text, "Total Dynamic Power")
                leakage_mw = _power_value(report_text, "Cell Leakage Power")
                switching_mw = _optional_power_value(report_text, "Net Switching Power")
                internal_mw = _optional_power_value(report_text, "Cell Internal Power")
                window_energy = dynamic_mw * window_ns
                if pattern == "idle":
                    idle_energy[repeat_count] = window_energy
                row.update({
                    "status": "complete", "window_ns": window_ns,
                    "dynamic_power_mw": dynamic_mw, "leakage_power_mw": leakage_mw,
                    "switching_power_mw": "" if switching_mw is None else switching_mw,
                    "internal_power_mw": "" if internal_mw is None else internal_mw,
                    "window_dynamic_energy_pj": window_energy,
                    "incremental_energy_pj": window_energy - idle_energy.get(repeat_count, window_energy),
                    "vcd_path": str(vcd), "power_report": str(report),
                    "activity_log": str(activity_log),
                    "activity_level": activity_level,
                    "logic_area_um2": (
                        "" if logic_area_um2 is None else logic_area_um2
                    ),
                    "features_json": (
                        (vcd.with_suffix(vcd.suffix + ".actions.json")).read_text().strip()
                        if vcd.with_suffix(vcd.suffix + ".actions.json").exists()
                        else ""
                    ),
                    "pwr_414": int("PWR-414" in (dc.stdout + dc.stderr)),
                    "pwr_415": int("PWR-415" in (dc.stdout + dc.stderr)),
                    "failure_reason": "",
                })
                if row["pwr_414"] or row["pwr_415"]:
                    raise RuntimeError("activity annotation incomplete (PWR-414/PWR-415)")
            except Exception as exc:
                row["status"] = "failed"
                row["failure_reason"] = repr(exc)
            scenario_rows.append(row)
            with self.csv_lock:
                append_row(run_dir / "power_calibration_points.csv", row, ACTIVITY_FIELDS)
        random_samples = [
            (float(row["repeat_count"]), float(row["incremental_energy_pj"]))
            for row in scenario_rows
            if row["status"] == "complete" and row["pattern"] == "random"
        ]
        activity_r2 = _linear_r2(random_samples)
        complete = all(row["status"] == "complete" for row in scenario_rows)
        if activity_r2 is None or activity_r2 < 0.95:
            complete = False
        wns = None
        if "timing" in copied:
            matches = re.findall(r"slack\s*\([^)]*\)\s*(-?[0-9.]+)", copied["timing"].read_text(errors="ignore"), re.I)
            wns = min(map(float, matches)) if matches else None
        if cleanup_builds:
            shutil.rmtree(worker_rtl / "build/synth" / point.top_module, ignore_errors=True)
        return {
            "point_id": point.point_id, "module": point.module, "top_module": point.top_module,
            "component": point.component, "holdout": int(point.holdout),
            "status": "complete" if complete else "failed", "worker_id": worker_id,
            "elapsed_sec": round(time.time() - start, 3),
            "mapped_ddc": str(copied.get("mapped_ddc", "")),
            "mapped_netlist": str(copied.get("mapped_netlist", "")),
            "sdf": str(copied.get("sdf", "")), "sdc": str(copied.get("sdc", "")),
            "wns_ns": "" if wns is None else wns,
            "timing_status": (
                "timing_unknown" if wns is None else "timing_unclosed" if wns < 0 else "timing_closed"
            ),
            "activity_complete": int(complete), "activity_r2": "" if activity_r2 is None else activity_r2,
            "report_dir": str(run_dir / "reports" / point.point_key),
            "command_log_dir": str(log_dir),
            "failure_reason": "" if complete else "activity scenarios failed or random-action slope R2 < 0.95",
        }

    def run_point(self, point: PowerPoint, worker_id: int, worker_rtl: Path, rtl_root: Path, run_dir: Path, cleanup_builds: bool, license_retry_wait_sec: float, license_max_retries: int) -> dict[str, Any]:
        """Run a point and guarantee per-top build cleanup on every exit path."""

        try:
            return self._run_point_impl(
                point,
                worker_id,
                worker_rtl,
                rtl_root,
                run_dir,
                cleanup_builds,
                license_retry_wait_sec,
                license_max_retries,
            )
        finally:
            if cleanup_builds:
                shutil.rmtree(worker_rtl / "build/synth" / point.top_module, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, default=DEFAULT_RTL_ROOT)
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--workers", default="4")
    parser.add_argument("--reserve", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cleanup-worker-builds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-workers", action="store_true")
    parser.add_argument(
        "--copy-to-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="archive the append-only activity CSV under analytic_models/power/calibration",
    )
    parser.add_argument("--vcd-root", type=Path)
    parser.add_argument("--activity-command-template")
    parser.add_argument(
        "--gate-level-activity-command-template",
        help="optional holdout-only simulator command using {mapped_netlist} and {sdf}",
    )
    parser.add_argument("--vcd-strip-path", default="dut")
    parser.add_argument("--license-retry-wait-sec", type=float, default=60.0)
    parser.add_argument(
        "--license-max-retries",
        type=int,
        default=0,
        help="zero retries indefinitely while DC licenses are busy",
    )
    args = parser.parse_args()
    points = build_plan()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    _write_plan(points, args.run_dir)
    build_sram_energy_catalog(output=DEFAULT_CATALOG)
    if args.dry_run:
        summary = {
            "status": "dry_run", "mapped_netlists": len(points),
            "activity_scenarios_per_netlist": len(SCENARIOS),
            "mapped_syntheses": len(points), "activity_analyses": len(points) * len(SCENARIOS),
            "gate_level_validation": "preflight_pending",
        }
        (args.run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return 0
    if args.activity_command_template is None and args.vcd_root is None:
        raise ValueError("real calibration requires --activity-command-template or --vcd-root")
    requested = resolve_dc_worker_count(args.workers, repo_root=REPO_ROOT)
    workers = tmp_preflight(requested, args.worker_root)
    if workers < requested:
        print(f"Reduced workers from {requested} to {workers} to preserve /tmp headroom")
    adapter = PowerAdapter(
        args.activity_command_template,
        args.gate_level_activity_command_template,
        args.vcd_root,
        args.vcd_strip_path,
    )
    jobs = [
        CalibrationJob(stable_job_key(adapter.name, point), adapter.name, point, adapter, "power_v1_fixed_16")
        for point in points
    ]
    exit_code = run_calibration_jobs(
        jobs, {adapter.name: adapter},
        RuntimeConfig(
            run_dir=args.run_dir, rtl_root=args.rtl_root, worker_root=args.worker_root,
            workers=workers, reserve=args.reserve, cleanup_worker_builds=args.cleanup_worker_builds,
            keep_workers=args.keep_workers, resume=args.resume,
            license_retry_wait_sec=args.license_retry_wait_sec,
            license_max_retries=args.license_max_retries,
        ),
    )
    points_csv = args.run_dir / "power_calibration_points.csv"
    if args.copy_to_calibration and points_csv.exists():
        destination = REPO_ROOT / "analytic_models/power/calibration/power_calibration_points.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(points_csv, destination)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
