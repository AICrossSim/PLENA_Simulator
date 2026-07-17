#!/usr/bin/env python3
"""Run deterministic opcode timing harnesses on a temporary PLENA_RTL copy.

This tool collects raw cycle observations only.  It does not silently rewrite
the production calibration artifact: changing a fitted/structural formula is a
reviewable source change.  The JSON output contains the RTL HEAD, dirty state,
commands, logs, and parsed ``[RTL_TIMING]`` records needed for that review.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RTL_ROOT = Path(os.environ.get("PLENA_RTL_ROOT", "/home/yh3525/FYP/PLENA_RTL"))
HARNESS_TARGETS = {
    "scalar_alu_timing.py": "src/scalar_machine/test/rtl_v1_scalar_alu_timing.py",
    "scalar_sfu_timing.py": "src/scalar_machine/test/rtl_v1_scalar_sfu_timing.py",
    "vector_element_timing.py": "src/vector_machine/test/rtl_v1_vector_element_timing.py",
    "vector_reduction_timing.py": "src/vector_machine/test/rtl_v1_vector_reduction_timing.py",
    "mxint_mcu_timing.py": "src/basic_components/systolic_gemm_mxint/test/rtl_v1_mxint_mcu_timing.py",
    "mxfp_mcu_timing.py": "src/basic_components/systolic_gemm_mx/test/rtl_v1_mxfp_mcu_timing.py",
    "matrix_machine_full_timing.py": "src/matrix_machine/test/rtl_v1_matrix_machine_full_timing.py",
    "vector_machine_full_timing.py": "src/vector_machine/test/rtl_v1_vector_machine_full_timing.py",
    "scalar_machine_full_timing.py": "src/scalar_machine/test/rtl_v1_scalar_machine_full_timing.py",
    "pipeline_control_timing.py": "src/control/test/rtl_v1_pipeline_control_timing.py",
}
WRAPPER_TARGETS = {
    "matrix_machine_timing_wrapper.sv": "src/matrix_machine/rtl/matrix_machine_timing_wrapper.sv",
    "vector_machine_timing_wrapper.sv": "src/vector_machine/rtl/vector_machine_timing_wrapper.sv",
    "scalar_machine_timing_wrapper.sv": "src/scalar_machine/rtl/scalar_machine_timing_wrapper.sv",
    "pipeline_control_timing_wrapper.sv": "src/control/rtl/pipeline_control_timing_wrapper.sv",
}
TIMING_RE = re.compile(r"\[RTL_TIMING\]\s+([A-Z0-9_]+)\s*(.*)")
KEY_VALUE_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^\s,]+)")
COCOTB_FAILURE_RE = re.compile(r"TESTS=\d+\s+PASS=\d+\s+FAIL=([1-9]\d*)")


def _git(rtl_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(rtl_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _rtl_diff_sha256(rtl_root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(rtl_root), "diff", "--binary", "HEAD"],
        check=False,
        capture_output=True,
    )
    return hashlib.sha256(proc.stdout).hexdigest()


def _copy_rtl(rtl_root: Path, work_root: Path) -> Path:
    destination = work_root / "PLENA_RTL"
    shutil.copytree(
        rtl_root,
        destination,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git",
            ".direnv",
            ".venv",
            ".pytest_cache",
            "__pycache__",
            "build",
            "target",
            "PLENA_Compiler",
            "result",
            "results",
        ),
    )
    for source_name, relative_target in HARNESS_TARGETS.items():
        target = destination / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "harnesses" / source_name, target)
    for source_name, relative_target in WRAPPER_TARGETS.items():
        target = destination / relative_target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "wrappers" / source_name, target)
    return destination


def _patch_localparams(path: Path, updates: dict[str, int]) -> None:
    """Patch concrete package localparams in a disposable RTL copy."""
    text = path.read_text(encoding="utf-8")
    for name, value in updates.items():
        pattern = re.compile(
            rf"(^\s*localparam\s+(?:\w+\s+)?{re.escape(name)}\s*=\s*)([^;]+)(;)",
            re.MULTILINE,
        )
        text, count = pattern.subn(rf"\g<1>{value}\g<3>", text, count=1)
        if count != 1:
            raise ValueError(f"could not patch localparam {name} in {path}")
    path.write_text(text, encoding="utf-8")


def _full_machine_invocations(harness: str, mode: str) -> list[dict[str, Any]]:
    """Return global-package configurations required by one full top harness."""
    common_precision = {
        "WT_MX_INT_ENABLE": 0,
        "KV_MX_INT_ENABLE": 0,
        "ACT_MX_INT_ENABLE": 0,
        "WT_MX_EXP_WIDTH": 4,
        "WT_MX_MANT_WIDTH": 3,
        "KV_MX_EXP_WIDTH": 4,
        "KV_MX_MANT_WIDTH": 3,
        "ACT_MXFP_EXP_WIDTH": 4,
        "ACT_MXFP_MANT_WIDTH": 3,
        "M_FP_EXP_WIDTH": 8,
        "M_FP_MANT_WIDTH": 7,
        "V_FP_EXP_WIDTH": 8,
        "V_FP_MANT_WIDTH": 7,
        "S_FP_EXP_WIDTH": 8,
        "S_FP_MANT_WIDTH": 7,
    }
    if harness == "matrix_machine_full_timing.py":
        shapes = [(16, 4)] if mode == "smoke" else [
            (16, 4),
            (32, 4),
            (64, 4),
            (32, 8),
            (64, 8),
            (64, 16),
        ]
        return [
            {
                "tag": f"m{mlen}_b{blen}_e8m7",
                "config": {"MLEN": mlen, "VLEN": mlen, "BLEN": blen, "HLEN": min(8, mlen)},
                "precision": {**common_precision, "BLOCK_DIM": blen},
                "env": {"PLENA_TEST_MLEN": mlen, "PLENA_TEST_BLEN": blen},
            }
            for mlen, blen in shapes
        ]
    if harness in {"vector_machine_full_timing.py", "scalar_machine_full_timing.py"}:
        points = [(8, 8, 7)] if mode == "smoke" else [
            (8, 8, 7),
            (16, 8, 7),
            (32, 8, 7),
            (64, 8, 7),
            (32, 6, 5),
        ]
        return [
            {
                "tag": f"v{vlen}_e{exp}m{mant}",
                "config": {"MLEN": vlen, "VLEN": vlen, "BLEN": min(4, vlen), "HLEN": min(8, vlen)},
                "precision": {
                    **common_precision,
                    "BLOCK_DIM": min(4, vlen),
                    "V_FP_EXP_WIDTH": exp,
                    "V_FP_MANT_WIDTH": mant,
                    "S_FP_EXP_WIDTH": exp,
                    "S_FP_MANT_WIDTH": mant,
                },
                "env": {
                    "PLENA_TEST_VLEN": vlen,
                    "PLENA_TEST_FP_EXP": exp,
                    "PLENA_TEST_FP_MANT": mant,
                },
            }
            for vlen, exp, mant in points
        ]
    if harness == "pipeline_control_timing.py":
        return [
            {
                "tag": "default",
                "config": {"MLEN": 16, "VLEN": 16, "BLEN": 4, "HLEN": 8},
                "precision": {**common_precision, "BLOCK_DIM": 4},
                "env": {},
            }
        ]
    return [{"tag": "default", "config": {}, "precision": {}, "env": {}}]


def _harness_invocations(harness: str, mode: str) -> list[dict[str, Any]]:
    if harness.endswith("_full_timing.py") or harness == "pipeline_control_timing.py":
        return _full_machine_invocations(harness, mode)
    return [{"tag": "default", "config": {}, "precision": {}, "env": {}}]


def _parse_scalar(value: str) -> int | float | str:
    value = value.rstrip(".,")
    try:
        return int(value, 0)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_measurements(output: str, harness: str) -> list[dict[str, Any]]:
    measurements: list[dict[str, Any]] = []
    for line in output.splitlines():
        match = TIMING_RE.search(line)
        if not match:
            continue
        record: dict[str, Any] = {"harness": harness, "opcode": match.group(1)}
        for key, value in KEY_VALUE_RE.findall(match.group(2)):
            record[key] = _parse_scalar(value)
        measurements.append(record)
    return measurements


def _run_harness(
    *,
    source_rtl: Path,
    work_rtl: Path,
    relative_target: str,
    mode: str,
    environment: dict[str, Any],
    log_path: Path,
) -> tuple[list[str], list[dict[str, Any]]]:
    venv_activate = source_rtl / ".venv/bin/activate"
    if not venv_activate.exists():
        raise FileNotFoundError(
            f"PLENA_RTL Python environment is missing: {venv_activate}. "
            "Create it before running the cocotb calibration."
        )

    extra_environment = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in sorted(environment.items())
    )
    body = (
        f"source {shlex.quote(str(venv_activate))}; "
        f"cd {shlex.quote(str(work_rtl))}; "
        f"PLENA_TIMING_MODE={shlex.quote(mode)} "
        f"{extra_environment} "
        f"python {shlex.quote(relative_target)}"
    )
    command = ["nix", "develop", str(source_rtl), "--command", "bash", "-lc", body]
    proc = subprocess.run(command, check=False, capture_output=True, text=True)
    combined = proc.stdout + proc.stderr
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"RTL timing harness failed ({relative_target}, exit={proc.returncode}); "
            f"see {log_path}"
        )
    if COCOTB_FAILURE_RE.search(combined):
        raise RuntimeError(
            f"RTL timing cocotb test failed ({relative_target}); see {log_path}"
        )
    measurements = _parse_measurements(combined, Path(relative_target).name)
    if not measurements:
        raise RuntimeError(f"No [RTL_TIMING] records found in {log_path}")
    return command, measurements


def _cleanup_harness_builds(work_rtl: Path) -> None:
    """Remove generated simulator builds while retaining copied RTL sources."""
    for build_dir in work_rtl.glob("src/**/test/build"):
        shutil.rmtree(build_dir, ignore_errors=True)


def _point_key(harness: str, point_tag: str) -> tuple[str, str]:
    """Return the stable identity of one full-machine simulator invocation."""
    # Installed cocotb harnesses carry an ``rtl_v1_`` prefix while CLI choices
    # use the source filename. Canonicalize both forms so schema-2 artifacts can
    # be resumed without repeating their first point.
    canonical = Path(harness).name.removeprefix("rtl_v1_")
    return canonical, point_tag


def _dedupe_measurements(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest record for each harness/point/opcode observation."""
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        harness, point_tag = _point_key(
            str(record.get("harness", "")), str(record.get("point_tag", ""))
        )
        key = (harness, point_tag, str(record.get("opcode", "")))
        latest[key] = record
    return list(latest.values())


def _write_payload(
    *,
    path: Path,
    mode: str,
    selected_harnesses: list[str],
    rtl_root: Path,
    commands: list[list[str]],
    measurements: list[dict[str, Any]],
    failure: str | None,
    temporary_workdir: str | None,
) -> None:
    """Checkpoint provenance and completed points after every invocation.

    Full Machine elaboration is expensive.  Writing only in ``finally`` made a
    machine reboot or hard kill indistinguishable from an empty run.  This
    append-by-rewrite checkpoint is small (JSON metadata only) and lets
    ``--resume`` skip every point for which at least one timing record exists.
    """
    payload = {
        "schema_version": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "selected_harnesses": selected_harnesses,
        "rtl_root": str(rtl_root),
        "rtl_head": _git(rtl_root, "rev-parse", "HEAD"),
        "rtl_dirty": bool(_git(rtl_root, "status", "--porcelain")),
        "rtl_diff_sha256": _rtl_diff_sha256(rtl_root),
        "implementation_profile": "behavioral_current_rtl",
        "measurement_boundary": {
            "start": "opcode accepted at production machine control boundary",
            "ready": "architectural result-valid/write request",
            "done": "backend resource idle",
            "initiation_interval": "next independent opcode acceptance interval",
        },
        "dc_library_profile_verified": False,
        "commands": commands,
        "measurements": measurements,
        "failure": failure,
        "temporary_workdir": temporary_workdir,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rtl-root", type=Path, default=DEFAULT_RTL_ROOT)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument(
        "--harness",
        action="append",
        choices=tuple(HARNESS_TARGETS),
        help=(
            "Run only the named harness. Repeat this option to select multiple "
            "harnesses; by default all harnesses run."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip harness/point pairs already checkpointed in raw_measurements.json.",
    )
    args = parser.parse_args()

    rtl_root = args.rtl_root.resolve()
    if not (rtl_root / "flake.nix").exists():
        parser.error(f"not a PLENA_RTL checkout: {rtl_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    temporary = Path(tempfile.mkdtemp(prefix="plena_rtl_timing_", dir=args.work_root))
    raw_path = args.out_dir / "raw_measurements.json"
    commands: list[list[str]] = []
    measurements: list[dict[str, Any]] = []
    if args.resume and raw_path.exists():
        previous = json.loads(raw_path.read_text(encoding="utf-8"))
        commands = list(previous.get("commands", []))
        measurements = _dedupe_measurements(list(previous.get("measurements", [])))
    completed_points = {
        _point_key(str(record.get("harness", "")), str(record.get("point_tag", "")))
        for record in measurements
    }
    failure: str | None = None
    selected_harnesses = args.harness or list(HARNESS_TARGETS)
    try:
        work_rtl = _copy_rtl(rtl_root, temporary)
        for harness_name in selected_harnesses:
            relative_target = HARNESS_TARGETS[harness_name]
            for invocation in _harness_invocations(harness_name, args.mode):
                key = _point_key(harness_name, invocation["tag"])
                if key in completed_points:
                    print(f"Skipping completed timing point {harness_name}:{invocation['tag']}")
                    continue
                if invocation["config"]:
                    _patch_localparams(
                        work_rtl / "src/definitions/configuration.svh",
                        invocation["config"],
                    )
                if invocation["precision"]:
                    _patch_localparams(
                        work_rtl / "src/definitions/precision.svh",
                        invocation["precision"],
                    )
                log_path = args.out_dir / f"{Path(relative_target).stem}_{invocation['tag']}.log"
                command, parsed = _run_harness(
                    source_rtl=rtl_root,
                    work_rtl=work_rtl,
                    relative_target=relative_target,
                    mode=args.mode,
                    environment=invocation["env"],
                    log_path=log_path,
                )
                commands.append(command)
                for record in parsed:
                    record["point_tag"] = invocation["tag"]
                measurements.extend(parsed)
                completed_points.add(key)
                _write_payload(
                    path=raw_path,
                    mode=args.mode,
                    selected_harnesses=selected_harnesses,
                    rtl_root=rtl_root,
                    commands=commands,
                    measurements=measurements,
                    failure=None,
                    temporary_workdir=str(temporary) if args.keep_workdir else None,
                )
                if not args.keep_workdir:
                    _cleanup_harness_builds(work_rtl)
    except BaseException as error:  # also record Ctrl-C before re-raising
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        _write_payload(
            path=raw_path,
            mode=args.mode,
            selected_harnesses=selected_harnesses,
            rtl_root=rtl_root,
            commands=commands,
            measurements=measurements,
            failure=failure,
            temporary_workdir=str(temporary) if args.keep_workdir else None,
        )
        if not args.keep_workdir:
            shutil.rmtree(temporary, ignore_errors=True)

    print(f"Collected {len(measurements)} timing records in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
