"""Install and run the built-in RTL activity harness in an isolated RTL copy."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any


HERE = Path(__file__).resolve().parent
HARNESS = HERE / "power_activity_harness.py"
QWEN_MIX = HERE.parent / "calibration/qwen3_32b_action_mix_v2.json"
QWEN_V6_MIX = HERE.parent / "calibration/qwen3_32b_softmax_v6_action_mix_v1.json"
TRACE_MODE_VERSION = "verilator_struct_scopes_explicit_saif_map_v4_costtrace_mix"


@dataclass(frozen=True)
class ActivityArtifact:
    scenario: str
    pattern: str
    repeat_count: int
    microkernel: str
    vcd: Path
    sidecar: Path
    log: Path
    elapsed_sec: float
    peak_rss_kib: int | None
    fingerprint: str
    preconverted_saif: Path | None = None


def weighted_microkernel_schedule(weights: dict[str, int], count: int) -> list[str]:
    """Return a deterministic, interleaved largest-remainder action schedule.

    The calibration window is intentionally small, so the full Qwen action
    histogram must be projected to ``count`` actions. Largest remainder keeps
    family counts unbiased; the deficit scheduler then distributes each family
    throughout the window instead of grouping identical operations together.
    """

    positive = {str(name): int(value) for name, value in weights.items() if int(value) > 0}
    if count <= 0 or not positive:
        raise ValueError("weighted schedule requires positive count and weights")
    total = sum(positive.values())
    exact = {name: count * value / total for name, value in positive.items()}
    allocated = {name: int(value) for name, value in exact.items()}
    remainder = count - sum(allocated.values())
    for name in sorted(positive, key=lambda item: (-(exact[item] - allocated[item]), item))[:remainder]:
        allocated[name] += 1

    emitted = Counter()
    schedule: list[str] = []
    for slot in range(count):
        candidates = [name for name, target in allocated.items() if emitted[name] < target]
        selected = max(
            candidates,
            key=lambda name: ((slot + 1) * allocated[name] / count - emitted[name], name),
        )
        schedule.append(selected)
        emitted[selected] += 1
    if dict(emitted) != {name: value for name, value in allocated.items() if value}:
        raise AssertionError("weighted action schedule did not preserve allocated counts")
    return schedule


def _qwen_mix_payload(component: str) -> tuple[dict[str, Any], str]:
    path = QWEN_V6_MIX if component in {"softmax_v6", "packed_pv_v6"} else QWEN_MIX
    payload = json.loads(path.read_text())
    semantic_hash = str(payload.get("semantic_hash") or hashlib.sha256(path.read_bytes()).hexdigest())
    return payload, semantic_hash


def _qwen_mix_for_component(component: str, repeat_count: int) -> tuple[list[str], str]:
    payload, semantic_hash = _qwen_mix_payload(component)
    weights = payload["components"][component]["microkernel_weights"]
    return weighted_microkernel_schedule(weights, repeat_count), semantic_hash


def qwen_mix_semantic_hash(component: str = "vector") -> str:
    """Return the semantic identity required for resume-safe mixed replays."""

    _, semantic_hash = _qwen_mix_payload(component)
    return semantic_hash


def _rename_module(text: str, old: str) -> str:
    updated, count = re.subn(rf"\bmodule\s+{re.escape(old)}\b", "module power_activity_tb", text, count=1)
    if count != 1:
        raise ValueError(f"could not rename wrapper module {old}")
    return updated


def _scalar_wrapper(repo_root: Path) -> str:
    path = repo_root / "transactional_emulator/testbench/rtl_timing/wrappers/scalar_machine_timing_wrapper.sv"
    text = _rename_module(path.read_text(), "scalar_machine_timing_wrapper")
    marker = "    input  logic [FP_OPERAND_WIDTH-1:0] external_fp_wtarget,\n"
    text = text.replace(
        marker,
        marker
        + "    input  logic [3:0] scalar_int_op,\n"
        + "    input  logic [INT_OPERAND_WIDTH-1:0] int_rs1, int_rs2, int_rd,\n"
        + "    input  logic [IMM_WIDTH-1:0] int_imm,\n",
    )
    text = text.replace(".assigned_int_op           (STALL_S_INT)", ".assigned_int_op           (S_INT_OP'(scalar_int_op))")
    text = text.replace(".rs1                       ('0)", ".rs1                       (int_rs1)")
    text = text.replace(".rs2                       ('0)", ".rs2                       (int_rs2)")
    text = text.replace(".rd                        ('0)", ".rd                        (int_rd)")
    text = text.replace(".imm_in                    ('0)", ".imm_in                    (int_imm)")
    return text


def _vector_wrapper(repo_root: Path) -> str:
    path = repo_root / "transactional_emulator/testbench/rtl_timing/wrappers/vector_machine_timing_wrapper.sv"
    return _rename_module(path.read_text(), "vector_machine_timing_wrapper")


def _control_wrapper(repo_root: Path) -> str:
    path = repo_root / "transactional_emulator/testbench/rtl_timing/wrappers/pipeline_control_timing_wrapper.sv"
    text = _rename_module(path.read_text(), "pipeline_control_timing_wrapper")
    text = text.replace(
        ".fp_sram_stall_req           (fp_sram_stall_req),",
        ".fp_sram_stall_req           (fp_sram_stall_req),\n"
        "        .fp_pending_regs            ('0),\n"
        "        .fp_rob_full                (1'b0),",
    )
    text = text.replace(
        "assign decode_stage_op.update_v_waddr    = 1'b0;",
        "assign decode_stage_op.update_v_waddr    = 1'b0;\n"
        "    assign decode_stage_op.v_segment_broadcast_en = 1'b0;\n"
        "    assign decode_stage_op.v_lane_store_en = 1'b0;\n"
        "    assign decode_stage_op.v_multi_reduction_en = 1'b0;\n"
        "    assign decode_stage_op.v_element_mask_en = 1'b0;",
    )
    return text


def _agu_wrapper() -> str:
    return r'''`timescale 1ns / 1ps
`include "configuration.svh"
`include "operation.svh"

module power_activity_tb
  import configuration_pkg::*;
  import instruction_pkg::*;
(
  input logic clk, input logic rst,
  input logic config_valid,
  input logic [INT_OPERAND_WIDTH-1:0] config_reg,
  input logic [IMM_WIDTH-1:0] config_stride,
  input logic frame_start,
  input logic [INT_OPERAND_WIDTH-1:0] frame_counter_reg,
  input logic boundary_step, input logic boundary_exit,
  input logic gp_write_valid,
  input logic [INT_OPERAND_WIDTH-1:0] gp_write_addr,
  input logic [INT_OPERAND_WIDTH-1:0] gp_read_addr_1, gp_read_addr_2,
  input logic gp_read_valid_1, gp_read_valid_2,
  input logic [INT_DATA_WIDTH-1:0] gp_base_1, gp_base_2,
  output logic [INT_DATA_WIDTH-1:0] gp_resolved_1, gp_resolved_2
);
  loop_agu_state dut (
    .clk(clk), .rst(rst),
    .config_valid(config_valid), .config_reg(config_reg),
    .config_stride(config_stride), .frame_start(frame_start),
    .frame_counter_reg(frame_counter_reg),
    .boundary_step(boundary_step), .boundary_exit(boundary_exit),
    .gp_write_valid(gp_write_valid), .gp_write_addr(gp_write_addr),
    .gp_read_addr_1(gp_read_addr_1), .gp_read_addr_2(gp_read_addr_2),
    .gp_read_valid_1(gp_read_valid_1),
    .gp_read_valid_2(gp_read_valid_2),
    .gp_base_1(gp_base_1), .gp_base_2(gp_base_2),
    .gp_resolved_1(gp_resolved_1), .gp_resolved_2(gp_resolved_2)
  );
endmodule
'''


def _hbm_wrapper() -> str:
    return r'''`timescale 1ns / 1ps
`include "tl_pkg.svh"
`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"
`include "tl_util.svh"

module power_activity_tb
  import precision_pkg::*;
  import configuration_pkg::*;
  import instruction_pkg::*;
(
  input logic clk, input logic rst,
  input logic [2:0] h_op,
  input logic [ON_CHIP_ADDR_WIDTH-1:0] addr_1, addr_2,
  input logic prefetch_v_ready,
  input logic write_high_valid, write_low_valid,
  input logic [VLEN-1:0][ACT_ELEMENT_WIDTH-1:0] write_high_element,
  input logic [VLEN-1:0][KV_ELEMENT_WIDTH-1:0] write_low_element,
  input logic [VLEN/BLOCK_DIM-1:0][MX_SCALE_WIDTH-1:0] write_scale,
  output logic [31:0] accepted_lines,
  output logic [31:0] accepted_element_lines,
  output logic [31:0] accepted_scale_lines,
  output logic prefetch_m_valid, prefetch_v_valid, write_ready
);
  OP_BUNDLE exe_stage_op;
  `TL_DECLARE(HBM_ELE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_element_link);
  `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, m_scale_link);
  `TL_DECLARE(HBM_ELE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_element_link);
  `TL_DECLARE(HBM_SCALE_WIDTH, HBM_ADDR_WIDTH, SourceWidth, SinkWidth, v_scale_link);

  logic [MLEN-1:0][WT_ELEMENT_WIDTH-1:0] prefetch_m_element;
  logic [MLEN/BLOCK_DIM-1:0][MX_SCALE_WIDTH-1:0] prefetch_m_scale;
  logic [VLEN-1:0][ACT_ELEMENT_WIDTH-1:0] prefetch_v_high;
  logic [VLEN-1:0][KV_ELEMENT_WIDTH-1:0] prefetch_v_low;
  logic [VLEN/BLOCK_DIM-1:0][MX_SCALE_WIDTH-1:0] prefetch_v_scale;

  assign exe_stage_op = '{
    m_op: STALL_M,
    v_ele_op: STALL_V_ELEMENT,
    v_reduct_op: STALL_V_REDUCT,
    s_fp_op: STALL_S_FP,
    c_op: STALL_C,
    h_op: H_OP'(h_op),
    m_transposed_read: 1'b0,
    v_broadcast_en: 1'b0,
    fps1: '0,
    fps2: '0,
    fpd: '0,
    gp_reg1: '0,
    gp_reg2: '0,
    gp_rstride: '0,
    gp_rd: '0,
    addr_1: addr_1,
    addr_2: addr_2,
    update_m_waddr: 1'b0,
    update_v_waddr: 1'b0,
    v_segment_broadcast_en: 1'b0,
    v_lane_store_en: 1'b0,
    v_multi_reduction_en: 1'b0,
    v_element_mask_en: 1'b0,
    pc_tag: '0
  };

  hbm_sys dut (
    .clk(clk), .rst(rst), .exe_stage_op(exe_stage_op),
    .prefetch_m_valid(prefetch_m_valid), .prefetch_m_element(prefetch_m_element),
    .prefetch_m_scale(prefetch_m_scale), .prefetch_v_ready(prefetch_v_ready),
    .prefetch_v_valid(prefetch_v_valid),
    .prefetch_v_high_precision_element(prefetch_v_high),
    .prefetch_v_low_precision_element(prefetch_v_low), .prefetch_v_scale(prefetch_v_scale),
    .hbm_write_high_valid(write_high_valid), .hbm_write_low_valid(write_low_valid),
    .hbm_write_ready(write_ready), .hbm_write_high_element(write_high_element),
    .hbm_write_low_element(write_low_element), .hbm_write_scale(write_scale),
    `TL_CONNECT_HOST_PORT(host_m_element, m_element_link),
    `TL_CONNECT_HOST_PORT(host_m_scale, m_scale_link),
    `TL_CONNECT_HOST_PORT(host_v_element, v_element_link),
    `TL_CONNECT_HOST_PORT(host_v_scale, v_scale_link)
  );

  fake_hbm_4port #(
    .ADDR_WIDTH(HBM_ADDR_WIDTH), .ELE_DATA_WIDTH(HBM_ELE_WIDTH),
    .SCALE_DATA_WIDTH(HBM_SCALE_WIDTH), .BRAM_ADDR_WIDTH(12),
    .SourceWidth(SourceWidth), .SinkWidth(SinkWidth)
  ) memory (
    .clk(clk), .rst(rst),
    `TL_CONNECT_DEVICE_PORT(m_element, m_element_link),
    `TL_CONNECT_DEVICE_PORT(m_scale, m_scale_link),
    `TL_CONNECT_DEVICE_PORT(v_element, v_element_link),
    `TL_CONNECT_DEVICE_PORT(v_scale, v_scale_link)
  );

  always_ff @(posedge clk) begin
    if (rst) begin
      accepted_lines <= '0;
      accepted_element_lines <= '0;
      accepted_scale_lines <= '0;
    end else begin
      accepted_element_lines <= accepted_element_lines
        + (m_element_link_a_valid && m_element_link_a_ready)
        + (v_element_link_a_valid && v_element_link_a_ready);
      accepted_scale_lines <= accepted_scale_lines
        + (m_scale_link_a_valid && m_scale_link_a_ready)
        + (v_scale_link_a_valid && v_scale_link_a_ready);
      accepted_lines <= accepted_lines
        + (m_element_link_a_valid && m_element_link_a_ready)
        + (m_scale_link_a_valid && m_scale_link_a_ready)
        + (v_element_link_a_valid && v_element_link_a_ready)
        + (v_scale_link_a_valid && v_scale_link_a_ready);
    end
  end
endmodule
'''


def build_wrapper(point: Any, repo_root: Path) -> str:
    """Return a top named ``power_activity_tb`` with the mapped design at ``dut``."""
    if point.component == "scalar":
        return _scalar_wrapper(repo_root)
    if point.component == "vector":
        return _vector_wrapper(repo_root)
    if point.component == "softmax_v6":
        path = repo_root / (
            "src/vector_machine/rtl/"
            "vector_machine_rtl_v6_integration_wrapper.sv"
        )
        return _rename_module(
            path.read_text(), "vector_machine_rtl_v6_integration_wrapper"
        )
    if point.component == "packed_pv_v6":
        path = repo_root / "src/matrix_machine/rtl/packed_pv_accumulator.sv"
        return _rename_module(path.read_text(), "packed_pv_accumulator")
    if point.component == "control":
        return _control_wrapper(repo_root)
    if point.component == "agu":
        return _agu_wrapper()
    if point.component == "hbm":
        return _hbm_wrapper()
    if point.component == "matrix":
        # _prepare_point already emitted the exact mapped wrapper. Rename only
        # its top; the direct child remains named dut, matching mapped hierarchy.
        source = next((repo_root / "unused" for _ in ()), None)
        del source
        raise RuntimeError("matrix wrapper must be loaded from the patched worker")
    raise ValueError(f"unsupported component {point.component}")


def _patch_worker_trace_mode(worker_rtl: Path) -> Path:
    """Return the worker runner retained in its supported struct-trace mode.

    Verilator 5.034 needs ``--trace-structs`` for correct C++ generation of the
    HBM wrapper's packed ``OP_BUNDLE`` assignments. The resulting nested SAIF
    names are reconciled with DC by the explicit replay name map instead.
    """
    runner = worker_rtl / "tools/cfl_cocotb/runner.py"
    if "--trace-structs" not in runner.read_text():
        raise RuntimeError(f"worker runner lacks required packed-struct tracing: {runner}")
    return runner


def install_harness(point: Any, worker_rtl: Path, repo_root: Path) -> tuple[Path, str]:
    runner = _patch_worker_trace_mode(worker_rtl)
    group = worker_rtl / "src/power_activity"
    rtl_dir = group / "rtl"
    test_dir = group / "test"
    rtl_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    if point.component == "matrix":
        candidates = list(worker_rtl.glob(f"src/**/rtl/{point.top_module}.sv"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"expected one generated matrix wrapper for {point.top_module}, got {candidates}")
        wrapper = _rename_module(candidates[0].read_text(), point.top_module)
    else:
        # The rtl-v6 wrappers are parameterized by ``_prepare_point`` inside
        # the isolated RTL worker.  Reading them relative to the Simulator
        # repository both misses the file and would bypass those generated
        # parameters if a stale copy happened to exist there.
        wrapper_root = (
            worker_rtl
            if point.component in {"softmax_v6", "packed_pv_v6"}
            else repo_root
        )
        wrapper = build_wrapper(point, wrapper_root)
    wrapper_path = rtl_dir / "power_activity_tb.sv"
    wrapper_path.write_text(wrapper)
    harness_path = test_dir / "power_activity_tb.py"
    shutil.copy2(HARNESS, harness_path)
    fingerprint = hashlib.sha256(
        TRACE_MODE_VERSION.encode()
        + json.dumps(point.params, sort_keys=True, separators=(",", ":")).encode()
        + runner.read_bytes()
        + wrapper.encode()
        + HARNESS.read_bytes()
        + QWEN_MIX.read_bytes()
        + (QWEN_V6_MIX.read_bytes() if point.component in {"softmax_v6", "packed_pv_v6"} else b"")
        + (worker_rtl / "src/definitions/configuration.svh").read_bytes()
        + (worker_rtl / "src/definitions/precision.svh").read_bytes()
    ).hexdigest()
    return harness_path, fingerprint


def _peak_rss(time_report: Path) -> int | None:
    if not time_report.exists():
        return None
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_report.read_text(errors="ignore"))
    return int(match.group(1)) if match else None


def generate_activity_scenarios(
    *,
    point: Any,
    worker_rtl: Path,
    source_rtl: Path,
    repo_root: Path,
    run_dir: Path,
    scenarios: Iterable[tuple[str, str, int] | tuple[str, str, int, str]],
    verilator_jobs: int,
    on_artifact: Callable[[ActivityArtifact], None] | None = None,
    reuse_preconverted_saif: bool = False,
) -> list[ActivityArtifact]:
    harness, fingerprint = install_harness(point, worker_rtl, repo_root)
    build_root = worker_rtl / "src/power_activity/test/build/power_activity_tb/test_0"
    marker = build_root / ".power_fingerprint"
    skip_build = marker.exists() and marker.read_text().strip() == fingerprint
    # A SIGKILL can leave a partial Verilator tree before the fingerprint is
    # committed.  Such a tree is not resumable and must never be mistaken for
    # a valid compile cache.
    if build_root.exists() and not skip_build:
        shutil.rmtree(build_root, ignore_errors=True)

    artifacts: list[ActivityArtifact] = []
    for raw_scenario in scenarios:
        scenario, pattern, repeat_count = raw_scenario[:3]
        microkernel = raw_scenario[3] if len(raw_scenario) == 4 else "mixed"
        destination_dir = run_dir / "activity" / point.point_key
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"{scenario}.vcd"
        sidecar = destination.with_suffix(destination.suffix + ".actions.json")
        log_dir = run_dir / "command_logs" / point.point_key / "activity"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{scenario}.log"
        time_report = log_dir / f"{scenario}.time"
        cached_fingerprint = ""
        cached_payload: dict[str, Any] = {}
        if sidecar.exists():
            try:
                cached_payload = json.loads(sidecar.read_text())
                cached_fingerprint = cached_payload.get("activity_fingerprint", "")
            except (OSError, json.JSONDecodeError):
                cached_fingerprint = ""
                cached_payload = {}
        preconverted_saif = (
            run_dir / "reports" / point.point_key / scenario / "activity.saif.gz"
        )
        expected_completed_actions = 0 if pattern == "idle" else repeat_count
        if (
            reuse_preconverted_saif
            and sidecar.exists()
            and preconverted_saif.exists()
            and cached_payload.get("params") == point.params
            and cached_payload.get("component") == point.component
            and cached_payload.get("pattern") == pattern
            and cached_payload.get("microkernel") == microkernel
            and int(cached_payload.get("requested_actions", -1)) == repeat_count
            and cached_payload.get("accepted_actions") == expected_completed_actions
            and cached_payload.get("completed_actions") == expected_completed_actions
        ):
            # The RTL-SAIF is upstream of DC mapping and remains valid when a
            # mapping-only bug is repaired.  This explicit opt-in path avoids
            # regenerating multi-GiB VCDs while still requiring exact behavior
            # parameters and completed action accounting.
            with gzip.open(preconverted_saif, "rb") as handle:
                if not handle.read(64).lstrip().startswith(b"(SAIFILE"):
                    raise ValueError(
                        f"invalid cached RTL-SAIF for {point.point_id}/{scenario}"
                    )
            artifact = ActivityArtifact(
                scenario=scenario,
                pattern=pattern,
                repeat_count=repeat_count,
                microkernel=microkernel,
                vcd=destination,
                sidecar=sidecar,
                log=log,
                elapsed_sec=0.0,
                peak_rss_kib=None,
                fingerprint=fingerprint,
                preconverted_saif=preconverted_saif,
            )
            artifacts.append(artifact)
            if on_artifact is not None:
                on_artifact(artifact)
            continue
        if destination.exists() and sidecar.exists() and cached_fingerprint == fingerprint:
            artifact = ActivityArtifact(
                scenario=scenario,
                pattern=pattern,
                repeat_count=repeat_count,
                microkernel=microkernel,
                vcd=destination,
                sidecar=sidecar,
                log=log,
                elapsed_sec=0.0,
                peak_rss_kib=None,
                fingerprint=fingerprint,
            )
            artifacts.append(artifact)
            if on_artifact is not None:
                on_artifact(artifact)
            continue
        destination.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        test_dir = worker_rtl / "src/power_activity/test/runs" / scenario
        vcd_dir = test_dir / "vcd"
        shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        env = {
            "PLENA_POWER_COMPONENT": point.component,
            "PLENA_POWER_PATTERN": pattern,
            "PLENA_POWER_REPEATS": str(repeat_count),
            "PLENA_POWER_MICROKERNEL": microkernel,
            "PLENA_POWER_SIDECAR": str(sidecar),
            "PLENA_POWER_PARAMS_JSON": json.dumps(point.params, sort_keys=True),
            "PLENA_POWER_FINGERPRINT": fingerprint,
            "PLENA_POWER_SKIP_BUILD": "1" if skip_build else "0",
            "PLENA_POWER_TEST_DIR": str(test_dir),
            "PLENA_POWER_VCD_DIR": str(vcd_dir),
            "VERILATOR_JOBS": str(verilator_jobs),
            # cocotb forwards Verilator's -build-jobs but invokes the generated
            # makefile without -j. Bound MAKEFLAGS by the same heavy-job token.
            "MAKEFLAGS": f"-j{verilator_jobs}",
        }
        if microkernel == "mixed" and pattern == "representative-qwen":
            mix_sequence, mix_hash = _qwen_mix_for_component(point.component, repeat_count)
            env["PLENA_POWER_MIX_SEQUENCE_JSON"] = json.dumps(mix_sequence)
            env["PLENA_POWER_MIX_HASH"] = mix_hash
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        body = (
            f"source {shlex.quote(str(source_rtl / '.venv/bin/activate'))}; "
            f"cd {shlex.quote(str(worker_rtl))}; {exports} "
            f"/usr/bin/time -v -o {shlex.quote(str(time_report))} "
            f"python {shlex.quote(str(harness.relative_to(worker_rtl)))}"
        )
        import time

        started = time.monotonic()
        command = (
            ["bash", "-lc", body]
            if os.environ.get("IN_NIX_SHELL") and shutil.which("verilator")
            else ["nix", "develop", str(source_rtl), "--command", "bash", "-lc", body]
        )
        proc = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.monotonic() - started
        log.write_text(proc.stdout + "\n" + proc.stderr)
        generated = vcd_dir / "dump.vcd"
        if proc.returncode != 0 or not generated.exists() or not sidecar.exists():
            raise RuntimeError(f"activity generation failed for {point.point_id}/{scenario}; see {log}")
        shutil.move(str(generated), destination)
        if not skip_build:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(fingerprint + "\n")
            skip_build = True
        artifact = ActivityArtifact(
                scenario=scenario,
                pattern=pattern,
                repeat_count=repeat_count,
                microkernel=microkernel,
                vcd=destination,
                sidecar=sidecar,
                log=log,
                elapsed_sec=elapsed,
                peak_rss_kib=_peak_rss(time_report),
                fingerprint=fingerprint,
            )
        artifacts.append(artifact)
        if on_artifact is not None:
            on_artifact(artifact)
    return artifacts
