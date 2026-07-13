#!/usr/bin/env python3
"""Legacy DC-based SRAM calibration flow.

The default SRAM area model now uses ASAP7 SRAM macro LIB/LEF data via
build_asap7_sram_macro_table.py. This script is kept only as an explicit
debug fallback for synthesized register-array experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from license_utils import is_dc_license_unavailable_text, resolve_dc_worker_count

ROOT = Path(__file__).resolve().parents[3]
RTL_ROOT = Path("/home/yh3525/FYP/PLENA_RTL")
DEFAULT_WORKER_ROOT = Path("/tmp/plena_rtl_area_workers")
CALIBRATION_DIR = ROOT / "analytic_models" / "area_new" / "calibration"

CSV_FIELDS = [
    "point_key",
    "point_id",
    "sram_kind",
    "precision_mode",
    "module",
    "top_module",
    "status",
    "worker_id",
    "elapsed_sec",
    "area_um2",
    "dynamic_power",
    "leakage_power",
    "total_power",
    "report_dir",
    "summary_log",
    "failure_reason",
    "ACT_WIDTH",
    "KV_WIDTH",
    "WEIGHT_WIDTH",
    "FP_SETTING",
    "DATA_WIDTH",
    "DEPTH",
    "SRAM_DEPTH",
    "MLEN",
    "VLEN",
    "BLEN",
    "BLOCK_DIM",
    "PARALLEL_DIM",
    "ELEMENT_WIDTH",
    "ACT_ELEMENT_WIDTH",
    "KV_ELEMENT_WIDTH",
    "FP_WIDTH",
    "SCALE_WIDTH",
    "BANKS",
    "PORTS",
]

FP_SETTINGS = [(3, 2), (2, 3), (6, 5), (5, 6), (4, 7), (8, 5)]
MXFP_FORMATS = [(1, 2), (2, 1), (4, 3), (5, 2)]


@dataclass(frozen=True)
class Point:
    """One legacy register-array SRAM synthesis point."""
    point_id: str
    sram_kind: str
    precision_mode: str
    module: str
    top_module: str
    params: dict[str, Any]
    point_key: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "sram_kind": self.sram_kind,
            "precision_mode": self.precision_mode,
            "module": self.module,
            "top_module": self.top_module,
            "params": self.params,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        object.__setattr__(self, "point_key", f"{self.sram_kind}_{self.precision_mode}_{digest}")


def ceil_log2(value: int) -> int:
    """Return a safe ceiling log2 used for generated address widths."""
    return max(1, (value - 1).bit_length())


def mxfp_name(exp: int, mant: int) -> str:
    """Return a canonical MXFP token."""
    return f"MXFP_E{exp}M{mant}"


def mxfp_width(exp: int, mant: int) -> int:
    """Return sign + exponent + mantissa bits."""
    return 1 + exp + mant


def fp_name(exp: int, mant: int) -> str:
    """Return a canonical internal FP token."""
    return f"FP_E{exp}M{mant}"


def representative_mxfp_profiles() -> list[dict[str, Any]]:
    """Return a small set of same-format and asymmetric debug profiles."""
    raw = [
        ((1, 2), (1, 2)),
        ((2, 1), (2, 1)),
        ((4, 3), (4, 3)),
        ((5, 2), (5, 2)),
        ((1, 2), (4, 3)),
        ((4, 3), (1, 2)),
        ((2, 1), (5, 2)),
        ((5, 2), (2, 1)),
    ]
    profiles = []
    for act, kv in raw:
        profiles.append(
            {
                "ACT_WIDTH": mxfp_name(*act),
                "KV_WIDTH": mxfp_name(*kv),
                "ACT_EXP": act[0],
                "ACT_MANT": act[1],
                "KV_EXP": kv[0],
                "KV_MANT": kv[1],
                "ACT_ELEMENT_WIDTH": mxfp_width(*act),
                "KV_ELEMENT_WIDTH": mxfp_width(*kv),
            }
        )
    return profiles


def build_plan(mode: str) -> list[Point]:
    """Build legacy matrix, vector, scalar, or combined register-array points."""
    points: list[Point] = []
    if mode in {"matrix", "all"}:
        for t_bits in [4, 8]:
            for mlen in [16, 32, 64]:
                for block_dim in [4, 8]:
                    if mlen % block_dim != 0:
                        continue
                    for depth in [32, 64, 128, 256]:
                        for parallel_dim in [1, 2, 4]:
                            top = f"area_new_matrix_sram_mxint_t{t_bits}_m{mlen}_b{block_dim}_d{depth}_p{parallel_dim}"
                            points.append(
                                Point(
                                    point_id=top,
                                    sram_kind="matrix_sram",
                                    precision_mode="mxint",
                                    module="matrix_sram_without_rounding",
                                    top_module=top,
                                    params={
                                        "WEIGHT_WIDTH": f"MXINT{t_bits}",
                                        "ELEMENT_WIDTH": t_bits,
                                        "MLEN": mlen,
                                        "BLOCK_DIM": block_dim,
                                        "SRAM_DEPTH": depth,
                                        "PARALLEL_DIM": parallel_dim,
                                        "SCALE_WIDTH": 8,
                                        "BANKS": 2 * math.ceil(mlen / parallel_dim),
                                        "PORTS": 2,
                                    },
                                )
                            )
        for exp, mant in MXFP_FORMATS:
            elem = mxfp_width(exp, mant)
            for mlen in [16, 32, 64]:
                for block_dim in [4, 8]:
                    if mlen % block_dim != 0:
                        continue
                    for depth in [32, 64, 128, 256]:
                        for parallel_dim in [1, 2, 4]:
                            top = f"area_new_matrix_sram_mxfp_e{exp}m{mant}_m{mlen}_b{block_dim}_d{depth}_p{parallel_dim}"
                            points.append(
                                Point(
                                    point_id=top,
                                    sram_kind="matrix_sram",
                                    precision_mode="mxfp",
                                    module="matrix_sram_without_rounding",
                                    top_module=top,
                                    params={
                                        "WEIGHT_WIDTH": mxfp_name(exp, mant),
                                        "WT_EXP": exp,
                                        "WT_MANT": mant,
                                        "ELEMENT_WIDTH": elem,
                                        "MLEN": mlen,
                                        "BLOCK_DIM": block_dim,
                                        "SRAM_DEPTH": depth,
                                        "PARALLEL_DIM": parallel_dim,
                                        "SCALE_WIDTH": 8,
                                        "BANKS": 2 * math.ceil(mlen / parallel_dim),
                                        "PORTS": 2,
                                    },
                                )
                            )

    if mode in {"vector", "all"}:
        mxint_profiles = [
            {
                "ACT_WIDTH": f"MXINT{act}",
                "KV_WIDTH": f"MXINT{kv}",
                "ACT_ELEMENT_WIDTH": act,
                "KV_ELEMENT_WIDTH": kv,
                "ACT_BITS": act,
                "KV_BITS": kv,
            }
            for act in [2, 4, 8]
            for kv in [2, 4, 8]
        ]
        vector_profiles = [("mxint", p) for p in mxint_profiles] + [("mxfp", p) for p in representative_mxfp_profiles()]
        for precision_mode, profile in vector_profiles:
            for fp_exp, fp_mant in FP_SETTINGS:
                fp_width = 1 + fp_exp + fp_mant
                for vlen in [32, 64, 128]:
                    for blen in [4, 8, 16]:
                        if vlen % blen != 0:
                            continue
                        for depth in [32, 64, 128, 256]:
                            top = (
                                f"area_new_vector_sram_{precision_mode}_a{profile['ACT_ELEMENT_WIDTH']}"
                                f"_k{profile['KV_ELEMENT_WIDTH']}_f{fp_width}_v{vlen}_b{blen}_d{depth}"
                            )
                            scale_blocks = math.ceil(vlen / blen)
                            width = vlen * (fp_width + profile["ACT_ELEMENT_WIDTH"] + profile["KV_ELEMENT_WIDTH"]) + 2 * scale_blocks * 8
                            points.append(
                                Point(
                                    point_id=top,
                                    sram_kind="vector_sram",
                                    precision_mode=precision_mode,
                                    module="fp_vector_sram",
                                    top_module=top,
                                    params={
                                        **profile,
                                        "FP_SETTING": fp_name(fp_exp, fp_mant),
                                        "FP_EXP": fp_exp,
                                        "FP_MANT": fp_mant,
                                        "FP_WIDTH": fp_width,
                                        "VLEN": vlen,
                                        "MLEN": vlen,
                                        "BLEN": blen,
                                        "BLOCK_DIM": blen,
                                        "SRAM_DEPTH": depth,
                                        "SCALE_WIDTH": 8,
                                        "DATA_WIDTH": width,
                                        "BANKS": 3,
                                        "PORTS": 2,
                                    },
                                )
                            )

    if mode in {"scalar", "all"}:
        for width in [16, 32, 64]:
            for depth in [16, 32, 64, 128, 256]:
                top = f"area_new_scalar_sram_int_w{width}_d{depth}"
                points.append(
                    Point(
                        point_id=top,
                        sram_kind="scalar_sram",
                        precision_mode="int",
                        module="scalar_sram",
                        top_module=top,
                        params={"DATA_WIDTH": width, "DEPTH": depth, "BANKS": 1, "PORTS": 1},
                    )
                )
        for exp, mant in sorted(set(FP_SETTINGS)):
            width = 1 + exp + mant
            for depth in [16, 32, 64, 128, 256]:
                top = f"area_new_scalar_sram_fp_e{exp}m{mant}_d{depth}"
                points.append(
                    Point(
                        point_id=top,
                        sram_kind="scalar_sram",
                        precision_mode="fp",
                        module="scalar_sram",
                        top_module=top,
                        params={
                            "FP_SETTING": fp_name(exp, mant),
                            "DATA_WIDTH": width,
                            "DEPTH": depth,
                            "BANKS": 1,
                            "PORTS": 1,
                        },
                    )
                )
    return points


def point_to_row(
    point: Point,
    *,
    status: str,
    worker_id: int | str = "",
    elapsed_sec: float | str = "",
    area_um2: float | str = "",
    dynamic_power: float | str | None = "",
    leakage_power: float | str | None = "",
    total_power: float | str | None = "",
    report_dir: str = "",
    summary_log: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    """Serialize one legacy SRAM synthesis outcome."""
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "point_key": point.point_key,
            "point_id": point.point_id,
            "sram_kind": point.sram_kind,
            "precision_mode": point.precision_mode,
            "module": point.module,
            "top_module": point.top_module,
            "status": status,
            "worker_id": worker_id,
            "elapsed_sec": elapsed_sec,
            "area_um2": area_um2,
            "dynamic_power": "" if dynamic_power is None else dynamic_power,
            "leakage_power": "" if leakage_power is None else leakage_power,
            "total_power": "" if total_power is None else total_power,
            "report_dir": report_dir,
            "summary_log": summary_log,
            "failure_reason": failure_reason,
        }
    )
    for key, value in point.params.items():
        if key in row:
            row[key] = value
    return row


def write_plan_csv(points: list[Point], path: Path) -> None:
    """Persist the planned register-array experiment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in points:
            writer.writerow(point_to_row(point, status="planned"))


def read_completed_keys(path: Path) -> set[str]:
    """Return successful point keys for resumable debug runs."""
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {str(row["point_key"]) for row in csv.DictReader(f) if row.get("status") == "complete"}


def append_row(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    """Append one attempt under a shared writer lock."""
    with lock:
        exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def create_worker_copy(worker_id: int, worker_root: Path, source_root: Path) -> Path:
    """Create one isolated lightweight RTL worker tree."""
    dest = worker_root / f"worker_{worker_id}" / "PLENA_RTL"
    if dest.exists():
        shutil.rmtree(dest)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        skip = {".git", ".venv", ".direnv", "build", "result", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
        return {name for name in names if name in skip or name.endswith(".pyc")}

    shutil.copytree(source_root, dest, ignore=ignore)
    patch_worker_rtl(dest)
    return dest


def patch_worker_rtl(worker_rtl: Path) -> None:
    """Apply calibration-local RTL fixes to the worker copy."""
    path = worker_rtl / "src/basic_components/fp_operation/rtl/fp_cp_asym_mult.sv"
    if path.exists():
        text = path.read_text()
        if "p1_signed_exp_a_ext" not in text:
            text = text.replace(
            "    logic signed [IN_EXP_WIDTH_B:0]     p1_signed_exp_b;\n"
            "    logic signed [IN_FIXED_WIDTH_B - 1:0]   p1_signed_mant_b;\n",
            "    logic signed [IN_EXP_WIDTH_B:0]     p1_signed_exp_b;\n"
            "    logic signed [IN_FIXED_WIDTH_B - 1:0]   p1_signed_mant_b;\n"
            "    logic signed [IN_EXP_WIDTH_A + 1:0] p1_signed_exp_a_ext;\n"
            "    logic signed [IN_EXP_WIDTH_B + 1:0] p1_signed_exp_b_ext;\n",
            )
            text = text.replace(
            "    logic p1_mult_valid;\n",
            "    assign p1_signed_exp_a_ext = {p1_signed_exp_a[IN_EXP_WIDTH_A], p1_signed_exp_a};\n"
            "    assign p1_signed_exp_b_ext = {p1_signed_exp_b[IN_EXP_WIDTH_B], p1_signed_exp_b};\n\n"
            "    logic p1_mult_valid;\n",
            )
            text = text.replace(".exp_a      (p1_signed_exp_a),", ".exp_a      (p1_signed_exp_a_ext),")
            text = text.replace(".exp_b      (p1_signed_exp_b),", ".exp_b      (p1_signed_exp_b_ext),")
        path.write_text(text)

    path = worker_rtl / "src/basic_components/gemv/rtl/fp_adder_tree_layer.sv"
    if path.exists():
        text = path.read_text()
        text = text.replace(".IN_EXP_WIDTH(IN_EXP_WIDTH),", ".IN_EXP_WIDTH(IN_EXP_WIDTH + 1),")
        path.write_text(text)

    for path in [
        worker_rtl / "src/basic_components/systolic_gemm_mx/rtl/mxfp_default_pe.sv",
        worker_rtl / "src/basic_components/systolic_gemm_mx/rtl/mx_default_pe.sv",
    ]:
        if not path.exists():
            continue
        text = path.read_text()
        if "MULT_EXP_WIDTH" not in text:
            text = text.replace(
                "logic [MX_L_EXP_WIDTH + MX_L_MANT_WIDTH : 0] block_mult_result;",
                "localparam int MULT_EXP_WIDTH = "
                "(MX_T_EXP_WIDTH > MX_L_EXP_WIDTH ? MX_T_EXP_WIDTH : MX_L_EXP_WIDTH);\n"
                "            localparam int MULT_MANT_WIDTH = "
                "(MX_T_MANT_WIDTH > MX_L_MANT_WIDTH ? MX_T_MANT_WIDTH : MX_L_MANT_WIDTH);\n"
                "            logic [MULT_EXP_WIDTH + MULT_MANT_WIDTH : 0] block_mult_result;",
            )
            text = text.replace(".MXFP_EXP_WIDTH     (MX_L_EXP_WIDTH),", ".MXFP_EXP_WIDTH     (MULT_EXP_WIDTH),")
            text = text.replace(".MXFP_MANT_WIDTH    (MX_L_MANT_WIDTH),", ".MXFP_MANT_WIDTH    (MULT_MANT_WIDTH),")
            path.write_text(text)


def cleanup_workers(worker_root: Path) -> None:
    """Remove temporary worker copies."""
    if worker_root.exists():
        shutil.rmtree(worker_root)


def scalar_wrapper(module: str, width: int, depth: int) -> str:
    """Generate a concrete scalar register-array wrapper."""
    aw = ceil_log2(depth)
    return f"""`timescale 1ns / 1ps
module {module}(
    input logic clk,
    input logic rst,
    input logic req,
    input logic write_en,
    input logic [{aw - 1}:0] sram_addr,
    input logic [{width - 1}:0] sram_data_in,
    output logic [{width - 1}:0] sram_data_out
);
    scalar_sram #(.DATA_WIDTH({width}), .DEPTH({depth})) dut (
        .clk(clk), .rst(rst), .req(req), .write_en(write_en),
        .sram_addr(sram_addr), .sram_data_in(sram_data_in), .sram_data_out(sram_data_out)
    );
endmodule
"""


def matrix_wrapper(point: Point) -> str:
    """Generate a concrete MatrixSRAM register-array wrapper."""
    p = point.params
    mlen = int(p["MLEN"])
    block_dim = int(p["BLOCK_DIM"])
    block_num = mlen // block_dim
    parallel_dim = int(p["PARALLEL_DIM"])
    elem = int(p["ELEMENT_WIDTH"])
    scale = int(p["SCALE_WIDTH"])
    depth = int(p["SRAM_DEPTH"])
    if point.precision_mode == "mxint":
        int_enable = 1
        exp = 4
        mant = 3
        int_width = elem
    else:
        int_enable = 0
        exp = int(p["WT_EXP"])
        mant = int(p["WT_MANT"])
        int_width = 7
    return f"""`timescale 1ns / 1ps
module {point.top_module}(
    input logic clk,
    input logic rst,
    input logic req,
    input logic transposed_read,
    input logic [31:0] sram_raddr,
    output logic [{parallel_dim - 1}:0][{mlen - 1}:0][{elem - 1}:0] element_out,
    output logic [{parallel_dim - 1}:0][{mlen - 1}:0][{scale - 1}:0] scale_out,
    input logic wen,
    output logic write_response,
    input logic [31:0] sram_waddr,
    input logic [{parallel_dim - 1}:0][{mlen - 1}:0][{elem - 1}:0] element_in,
    input logic [{parallel_dim - 1}:0][{block_num - 1}:0][{scale - 1}:0] scale_in,
    input logic [31:0] prefetch_addr,
    input logic prefetch_en,
    output logic data_not_ready
);
    matrix_sram_without_rounding #(
        .WT_MX_EXP_WIDTH({exp}),
        .WT_MX_MANT_WIDTH({mant}),
        .WT_MX_INT_ENABLE({int_enable}),
        .WT_MX_INT_WIDTH({int_width}),
        .MX_SCALE_WIDTH({scale}),
        .ON_CHIP_ADDR_WIDTH(32),
        .MLEN({mlen}),
        .BLOCK_DIM({block_dim}),
        .SRAM_DEPTH({depth}),
        .PARALLEL_DIM({parallel_dim}),
        .PREFETCH_AMOUNT(4)
    ) dut (
        .clk(clk), .rst(rst), .req(req), .transposed_read(transposed_read),
        .sram_raddr(sram_raddr), .element_out(element_out), .scale_out(scale_out),
        .wen(wen), .write_response(write_response), .sram_waddr(sram_waddr),
        .element_in(element_in), .scale_in(scale_in),
        .prefetch_addr(prefetch_addr), .prefetch_en(prefetch_en), .data_not_ready(data_not_ready)
    );
endmodule
"""


def vector_wrapper(point: Point) -> str:
    """Generate a concrete VectorSRAM register-array wrapper."""
    p = point.params
    vlen = int(p["VLEN"])
    mlen = int(p["MLEN"])
    blen = int(p["BLEN"])
    block_dim = int(p["BLOCK_DIM"])
    depth = int(p["SRAM_DEPTH"])
    fp_exp = int(p["FP_EXP"])
    fp_mant = int(p["FP_MANT"])
    fp_width = int(p["FP_WIDTH"])
    act_width = int(p["ACT_ELEMENT_WIDTH"])
    kv_width = int(p["KV_ELEMENT_WIDTH"])
    scale = int(p["SCALE_WIDTH"])
    m_blocks = mlen // block_dim
    v_blocks = vlen // block_dim
    if point.precision_mode == "mxint":
        act_int_enable = 1
        kv_int_enable = 1
        act_exp = 4
        act_mant = 3
        kv_exp = 4
        kv_mant = 3
        act_int = act_width
        kv_int = kv_width
    else:
        act_int_enable = 0
        kv_int_enable = 0
        act_exp = int(p["ACT_EXP"])
        act_mant = int(p["ACT_MANT"])
        kv_exp = int(p["KV_EXP"])
        kv_mant = int(p["KV_MANT"])
        act_int = 7
        kv_int = 7
    return f"""`timescale 1ns / 1ps
module {point.top_module}(
    input logic clk,
    input logic rst,
    input logic port_a_req,
    input logic port_a_write_en,
    input logic [31:0] port_a_addr,
    input logic select_write_data_a,
    input logic [{vlen - 1}:0][{fp_width - 1}:0] port_a_v_fp_in,
    input logic [{mlen - 1}:0][{fp_width - 1}:0] port_a_m_fp_in,
    input logic [{vlen - 1}:0] port_a_mask_in,
    output logic [{vlen - 1}:0][{fp_width - 1}:0] port_a_v_fp_out,
    output logic [{mlen - 1}:0][{act_width - 1}:0] port_a_element_out,
    output logic [{m_blocks - 1}:0][{scale - 1}:0] port_a_scale_out,
    input logic port_b_req,
    input logic port_b_write_en,
    input logic [31:0] port_b_addr,
    input logic [1:0] select_write_data_b,
    input logic [{mlen - 1}:0][{fp_width - 1}:0] port_b_fp_in,
    output logic [{vlen - 1}:0][{fp_width - 1}:0] port_b_fp_out,
    input logic [{vlen - 1}:0] port_b_mask_in,
    input logic [{vlen - 1}:0][{act_width - 1}:0] port_b_high_precision_element_in,
    input logic [{vlen - 1}:0][{kv_width - 1}:0] port_b_low_precision_element_in,
    input logic [{v_blocks - 1}:0][{scale - 1}:0] port_b_scale_in,
    input logic [1:0] port_b_mxfp_req,
    output logic port_b_mxfp_high_out_valid,
    output logic port_b_mxfp_low_out_valid,
    output logic [{vlen - 1}:0][{act_width - 1}:0] port_b_high_element_out,
    output logic [{vlen - 1}:0][{kv_width - 1}:0] port_b_low_element_out,
    output logic [{v_blocks - 1}:0][{scale - 1}:0] port_b_scale_out,
    input logic prefetch_en,
    input logic [31:0] prefetch_addr,
    output logic data_not_ready
);
    fp_vector_sram #(
        .ACT_MXFP_EXP_WIDTH({act_exp}),
        .ACT_MXFP_MANT_WIDTH({act_mant}),
        .WT_MX_EXP_WIDTH(4),
        .WT_MX_MANT_WIDTH(3),
        .KV_MX_EXP_WIDTH({kv_exp}),
        .KV_MX_MANT_WIDTH({kv_mant}),
        .MX_SCALE_WIDTH({scale}),
        .ACT_MX_INT_ENABLE({act_int_enable}),
        .ACT_MX_INT_WIDTH({act_int}),
        .WT_MX_INT_ENABLE(1),
        .WT_MX_INT_WIDTH(4),
        .KV_MX_INT_ENABLE({kv_int_enable}),
        .KV_MX_INT_WIDTH({kv_int}),
        .EXP_WIDTH({fp_exp}),
        .MANT_WIDTH({fp_mant}),
        .VLEN({vlen}),
        .MLEN({mlen}),
        .BLEN({blen}),
        .BLOCK_DIM({block_dim}),
        .SRAM_DEPTH({depth}),
        .ON_CHIP_ADDR_WIDTH(32),
        .PREFETCH_AMOUNT(4)
    ) dut (
        .clk(clk), .rst(rst),
        .port_a_req(port_a_req), .port_a_write_en(port_a_write_en), .port_a_addr(port_a_addr),
        .select_write_data_a(select_write_data_a), .port_a_v_fp_in(port_a_v_fp_in),
        .port_a_m_fp_in(port_a_m_fp_in), .port_a_mask_in(port_a_mask_in),
        .port_a_v_fp_out(port_a_v_fp_out), .port_a_element_out(port_a_element_out),
        .port_a_scale_out(port_a_scale_out),
        .port_b_req(port_b_req), .port_b_write_en(port_b_write_en), .port_b_addr(port_b_addr),
        .select_write_data_b(select_write_data_b), .port_b_fp_in(port_b_fp_in),
        .port_b_fp_out(port_b_fp_out), .port_b_mask_in(port_b_mask_in),
        .port_b_high_precision_element_in(port_b_high_precision_element_in),
        .port_b_low_precision_element_in(port_b_low_precision_element_in),
        .port_b_scale_in(port_b_scale_in), .port_b_mxfp_req(port_b_mxfp_req),
        .port_b_mxfp_high_out_valid(port_b_mxfp_high_out_valid),
        .port_b_mxfp_low_out_valid(port_b_mxfp_low_out_valid),
        .port_b_high_element_out(port_b_high_element_out),
        .port_b_low_element_out(port_b_low_element_out),
        .port_b_scale_out(port_b_scale_out),
        .prefetch_en(prefetch_en), .prefetch_addr(prefetch_addr), .data_not_ready(data_not_ready)
    );
endmodule
"""


def wrapper_text(point: Point) -> str:
    """Dispatch wrapper generation by SRAM kind."""
    if point.sram_kind == "scalar_sram":
        return scalar_wrapper(point.top_module, int(point.params["DATA_WIDTH"]), int(point.params["DEPTH"]))
    if point.sram_kind == "matrix_sram":
        return matrix_wrapper(point)
    if point.sram_kind == "vector_sram":
        return vector_wrapper(point)
    raise ValueError(f"unknown SRAM kind: {point.sram_kind}")


def write_wrapper(point: Point, rtl_root: Path) -> Path:
    """Write a generated wrapper into the worker RTL tree."""
    if point.sram_kind == "matrix_sram":
        target_dir = rtl_root / "src/memory/matrix_sram/rtl"
    elif point.sram_kind == "vector_sram":
        target_dir = rtl_root / "src/memory/vector_sram/rtl"
    elif point.sram_kind == "scalar_sram":
        target_dir = rtl_root / "src/memory/scalar_sram/rtl"
    else:
        raise ValueError(f"unknown SRAM kind: {point.sram_kind}")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{point.top_module}.sv"
    path.write_text(wrapper_text(point))
    return path


def parse_area(report: Path) -> float:
    """Parse total DC cell area in um^2."""
    if not report.exists():
        raise FileNotFoundError(report)
    match = re.search(r"Total cell area:\s*([0-9.]+)", report.read_text(errors="ignore"))
    if not match:
        raise ValueError(f"Total cell area not found in {report}")
    return float(match.group(1))


def parse_area_from_text(text: str) -> float | None:
    """Recover area from command output when no report is found."""
    match = re.search(r"Total cell area:\s*([0-9.]+)", text)
    return float(match.group(1)) if match else None


def parse_power(report: Path) -> dict[str, float | None]:
    """Parse optional DC power values."""
    if not report.exists():
        return {"dynamic_power": None, "leakage_power": None, "total_power": None}
    text = report.read_text(errors="ignore")

    def grab(pattern: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "dynamic_power": grab(r"Total Dynamic Power\s*=\s*([0-9.eE+-]+)"),
        "leakage_power": grab(r"Cell Leakage Power\s*=\s*([0-9.eE+-]+)"),
        "total_power": grab(r"Total Power\s*=\s*([0-9.eE+-]+)"),
    }


def json_safe(value: Any) -> Any:
    """Convert nested values to strict JSON-compatible data."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def copy_reports(worker_rtl: Path, point: Point, run_dir: Path) -> tuple[Path, Path | None, Path | None]:
    """Copy compact reports out of the worker build before cleanup."""
    latest = worker_rtl / f"build/synth/{point.top_module}/latest"
    dest = run_dir / "reports" / point.point_key
    dest.mkdir(parents=True, exist_ok=True)
    area_src = latest / f"reports/{point.top_module}_area.rpt"
    power_src = latest / f"reports/{point.top_module}_power.rpt"
    summary_src = latest / "logs/summary.log"
    area_dst = dest / "area.rpt"
    power_dst = dest / "power.rpt"
    summary_dst = dest / "summary.log"
    if area_src.exists():
        shutil.copy2(area_src, area_dst)
    if power_src.exists():
        shutil.copy2(power_src, power_dst)
    if summary_src.exists():
        shutil.copy2(summary_src, summary_dst)
    return area_dst, power_dst if power_dst.exists() else None, summary_dst if summary_dst.exists() else None


def summarize_synth_failure(result: subprocess.CompletedProcess[str]) -> str:
    """Return a concise reason from failed synthesis output."""
    text = f"{result.stdout}\n{result.stderr}"
    if is_dc_license_unavailable_text(text):
        return "design compiler license unavailable (SEC-50)"
    return f"synth failed with exit code {result.returncode}"


def cleanup_worker_build(worker_rtl: Path, point: Point) -> None:
    """Remove one worker-local synthesis build."""
    path = worker_rtl / f"build/synth/{point.top_module}"
    if path.exists():
        shutil.rmtree(path)


def run_point(
    point: Point,
    worker_id: int,
    worker_rtl: Path,
    rtl_root: Path,
    run_dir: Path,
    cleanup_builds: bool,
    license_retry_wait_sec: float,
    license_max_retries: int,
) -> dict[str, Any]:
    """Run one explicitly enabled register-array debug synthesis."""
    start = time.time()
    try:
        write_wrapper(point, worker_rtl)
        synth_cmd = f"cd {str(worker_rtl)!r} && just synth {point.top_module} 1000 area"
        cmd = ["nix", "develop", "-c", "bash", "-lc", synth_cmd]
        log_dir = run_dir / "command_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            attempt += 1
            result = subprocess.run(cmd, cwd=rtl_root, text=True, capture_output=True, check=False)
            (log_dir / f"{point.point_key}.attempt_{attempt}.stdout.log").write_text(result.stdout)
            (log_dir / f"{point.point_key}.attempt_{attempt}.stderr.log").write_text(result.stderr)
            (log_dir / f"{point.point_key}.stdout.log").write_text(result.stdout)
            (log_dir / f"{point.point_key}.stderr.log").write_text(result.stderr)

            if result.returncode == 0:
                break

            text = f"{result.stdout}\n{result.stderr}"
            if not is_dc_license_unavailable_text(text):
                break
            if license_max_retries > 0 and attempt > license_max_retries:
                break
            print(
                f"[license-busy] worker={worker_id} point={point.point_id} "
                f"attempt={attempt}; retrying in {license_retry_wait_sec:g}s",
                flush=True,
            )
            if cleanup_builds:
                cleanup_worker_build(worker_rtl, point)
            time.sleep(license_retry_wait_sec)
        if result.returncode != 0:
            stdout_area = parse_area_from_text(result.stdout)
            if stdout_area is not None and "Status: SUCCESS" in result.stdout:
                return point_to_row(
                    point,
                    status="complete",
                    worker_id=worker_id,
                    elapsed_sec=round(time.time() - start, 3),
                    area_um2=stdout_area,
                    failure_reason="synth reported success but report copy/summary failed",
                )
            return point_to_row(
                point,
                status="failed",
                worker_id=worker_id,
                elapsed_sec=round(time.time() - start, 3),
                failure_reason=summarize_synth_failure(result),
            )
        area_report, power_report, summary_log = copy_reports(worker_rtl, point, run_dir)
        area = parse_area(area_report)
        power = parse_power(power_report) if power_report else parse_power(Path("__missing__"))
        return point_to_row(
            point,
            status="complete",
            worker_id=worker_id,
            elapsed_sec=round(time.time() - start, 3),
            area_um2=area,
            dynamic_power=power["dynamic_power"],
            leakage_power=power["leakage_power"],
            total_power=power["total_power"],
            report_dir=str(area_report.parent),
            summary_log=str(summary_log or ""),
        )
    except Exception as exc:  # noqa: BLE001 - calibration should record failures and continue
        return point_to_row(
            point,
            status="failed",
            worker_id=worker_id,
            elapsed_sec=round(time.time() - start, 3),
            failure_reason=repr(exc),
        )
    finally:
        if cleanup_builds:
            cleanup_worker_build(worker_rtl, point)


def fit_linear(rows: list[dict[str, Any]], features: list[str], target: str = "area_um2") -> tuple[list[float], float]:
    """Fit the legacy unconstrained linear SRAM equation and MAPE."""
    if not rows:
        return [1.0] * len(features), float("nan")
    x = np.array([[float(row[name]) for name in features] for row in rows], dtype=float)
    y = np.array([float(row[target]) for row in rows], dtype=float)
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ coeffs
    mape = float(np.mean(np.abs((pred - y) / np.maximum(y, 1e-9))) * 100)
    return [float(c) for c in coeffs], mape


def completed_rows(csv_path: Path, kind: str) -> list[dict[str, Any]]:
    """Load successful rows for one logical SRAM kind."""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("status") == "complete" and row.get("sram_kind") == kind]


def _add_features(row: dict[str, Any]) -> None:
    if row["sram_kind"] == "matrix_sram":
        width = int(row["MLEN"]) * int(row["PARALLEL_DIM"]) * (int(row["ELEMENT_WIDTH"]) + int(row["SCALE_WIDTH"]))
        depth = int(row["SRAM_DEPTH"])
    elif row["sram_kind"] == "vector_sram":
        width = int(row["DATA_WIDTH"])
        depth = int(row["SRAM_DEPTH"])
    else:
        width = int(row["DATA_WIDTH"])
        depth = int(row["DEPTH"])
    banks = int(row.get("BANKS") or 1)
    ports = int(row.get("PORTS") or 1)
    row["feat_bits"] = depth * width
    row["feat_depth"] = depth
    row["feat_width"] = width
    row["feat_banks"] = banks
    row["feat_ports_width"] = ports * width
    row["feat_1"] = 1.0


def fit_and_write_coefficients(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Fit legacy register-array coefficients for explicit debugging only."""
    coeffs: dict[str, dict[str, float]] = {}
    metadata: dict[str, Any] = {"source_csv": str(csv_path), "status": "fitted_from_local_plena_rtl_synth"}
    for kind, key in [("matrix_sram", "matrix"), ("vector_sram", "vector"), ("scalar_sram", "scalar")]:
        rows = completed_rows(csv_path, kind)
        for row in rows:
            _add_features(row)
        vals, mape = fit_linear(rows, ["feat_bits", "feat_depth", "feat_width", "feat_banks", "feat_ports_width", "feat_1"])
        coeffs[key] = {"a": vals[0], "b": vals[1], "c": vals[2], "d": vals[3], "e": vals[4], "f": vals[5]}
        metadata[f"{key}_rows"] = len(rows)
        metadata[f"{key}_mape_pct"] = mape
    out = {"metadata": metadata, "coefficients": coeffs}
    path = run_dir / "sram_model_coefficients.json"
    path.write_text(json.dumps(json_safe(out), indent=2, sort_keys=True))
    if copy_to_calibration:
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, CALIBRATION_DIR / path.name)


def write_split_csvs(csv_path: Path, run_dir: Path, copy_to_calibration: bool) -> None:
    """Split successful legacy rows by SRAM kind."""
    if not csv_path.exists():
        return
    rows = list(csv.DictReader(csv_path.open(newline="")))
    for kind, name in [("matrix_sram", "matrix_sram"), ("vector_sram", "vector_sram"), ("scalar_sram", "scalar_sram")]:
        selected = [row for row in rows if row.get("status") == "complete" and row.get("sram_kind") == kind]
        if not selected:
            continue
        out = run_dir / f"{name}.csv"
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(selected)
        if copy_to_calibration:
            CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, CALIBRATION_DIR / out.name)
    if copy_to_calibration:
        shutil.copy2(csv_path, CALIBRATION_DIR / "sram_points.csv")


def run_dry_run(points: list[Point], run_dir: Path) -> None:
    """Generate wrappers and plan without running DC."""
    write_plan_csv(points, run_dir / "plans" / "calibration_plan.csv")
    wrappers = run_dir / "wrappers"
    wrappers.mkdir(parents=True, exist_ok=True)
    for point in points:
        (wrappers / f"{point.top_module}.sv").write_text(wrapper_text(point))
    print(f"Dry run wrote {len(points)} planned points to {run_dir}")


def parse_args() -> argparse.Namespace:
    """Parse legacy debug-flow options and explicit safety opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["matrix", "vector", "scalar", "all"], default="all")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rtl-root", type=Path, default=RTL_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", default="auto", help="worker count or 'auto' to use available DC licenses")
    parser.add_argument("--worker-root", type=Path, default=DEFAULT_WORKER_ROOT)
    parser.add_argument("--cleanup-worker-builds", action="store_true", default=True)
    parser.add_argument("--no-cleanup-worker-builds", dest="cleanup_worker_builds", action="store_false")
    parser.add_argument("--keep-workers", action="store_true")
    parser.add_argument("--no-copy-to-calibration", action="store_true")
    parser.add_argument(
        "--allow-register-array-synth",
        action="store_true",
        help=(
            "Allow legacy DC synthesis of behavioral SRAM/register arrays. "
            "Default is disabled because SRAM area should come from ASAP7 macro tables."
        ),
    )
    parser.add_argument(
        "--license-retry-wait-sec",
        type=float,
        default=float(os.environ.get("PLENA_DC_LICENSE_RETRY_WAIT_SEC", "60")),
        help="seconds to wait before retrying a point when DC licenses are busy",
    )
    parser.add_argument(
        "--license-max-retries",
        type=int,
        default=int(os.environ.get("PLENA_DC_LICENSE_MAX_RETRIES", "0")),
        help="max license-busy retries per point; 0 means retry indefinitely",
    )
    return parser.parse_args()


def main() -> int:
    """Run or dry-run the opt-in register-array calibration flow."""
    args = parse_args()
    run_dir: Path = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_root == DEFAULT_WORKER_ROOT:
        safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_dir.name)
        args.worker_root = args.worker_root / f"sram_{safe_run_name}_{os.getpid()}"
    points = build_plan(args.mode)
    if args.limit is not None:
        points = points[: args.limit]
    write_plan_csv(points, run_dir / "plans" / "calibration_plan.csv")
    if args.dry_run:
        run_dry_run(points, run_dir)
        return 0
    if not args.allow_register_array_synth:
        print(
            "SRAM DC calibration is disabled by default. Use "
            "analytic_models/area_new/scripts/build_asap7_sram_macro_table.py "
            "and the default asap7_sram_macro_tiling model instead. Pass "
            "--allow-register-array-synth only for explicit debug experiments.",
            file=sys.stderr,
        )
        return 2

    csv_path = run_dir / "calibration_points.csv"
    completed = read_completed_keys(csv_path) if args.resume else set()
    pending = [point for point in points if point.point_key not in completed]
    if not pending:
        print("No pending points.")
        write_split_csvs(csv_path, run_dir, not args.no_copy_to_calibration)
        fit_and_write_coefficients(csv_path, run_dir, not args.no_copy_to_calibration)
        return 0

    if args.worker_root.exists() and not args.keep_workers:
        shutil.rmtree(args.worker_root)
    args.worker_root.mkdir(parents=True, exist_ok=True)
    worker_count = resolve_dc_worker_count(args.workers, repo_root=ROOT)
    worker_paths = [create_worker_copy(i, args.worker_root, args.rtl_root) for i in range(worker_count)]
    worker_queue: queue.Queue[tuple[int, Path]] = queue.Queue()
    for item in enumerate(worker_paths):
        worker_queue.put(item)
    csv_lock = threading.Lock()

    def wrapped(point: Point) -> dict[str, Any]:
        worker_id, worker_path = worker_queue.get()
        try:
            return run_point(
                point,
                worker_id,
                worker_path,
                args.rtl_root,
                run_dir,
                args.cleanup_worker_builds,
                args.license_retry_wait_sec,
                args.license_max_retries,
            )
        finally:
            worker_queue.put((worker_id, worker_path))

    interrupted = False
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(wrapped, point): point for point in pending}
            for future in as_completed(futures):
                row = future.result()
                append_row(csv_path, row, csv_lock)
                print(f"[{row['status']}] {row['point_id']} area={row['area_um2']} reason={row['failure_reason']}")
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; compact rows already written remain resumable.", file=sys.stderr)
    finally:
        write_split_csvs(csv_path, run_dir, not args.no_copy_to_calibration)
        fit_and_write_coefficients(csv_path, run_dir, not args.no_copy_to_calibration)
        if not args.keep_workers:
            cleanup_workers(args.worker_root)
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
