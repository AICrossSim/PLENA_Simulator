"""Build measured compiler and emulator stage reports for stack validity.

Runs the decode compiler (ASM + compilation-artifact generation) and the
transactional emulator (decode-layer execution with numerical golden
comparison) as separately timestamped stages, then validates the resulting
dump with the stage-validation gate and emits one report per stage:

- ``compiler_report.json`` binds the generated assembly, compilation
  artifact, ISA library, and emulator settings by SHA-256.
- ``emulator_report.json`` binds the op-stats stream, run manifest, run
  receipt, and emulator binary by SHA-256.

Both reports carry tz-aware started/completed UTC timestamps around the real
subprocess invocation, the calibration ids they were validated against, and a
``content_hash`` over the canonical report body. The reports are portable:
they contain content hashes and ids only, never absolute paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

STAGE_REPORT_SCHEMA = "plena-stack-stage-report"
CALIBRATION_SCHEMA = "plena-decode-emulator-calibration"
RTL_SERIALIZED = "rtl_serialized"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TESTBENCH = _REPO_ROOT / "transactional_emulator" / "testbench"
_ASM_GEN = _TESTBENCH / "misc" / "decoder_decode_asm_gen.py"
_DECODE_TEST = _TESTBENCH / "misc" / "decoder_decode_test.py"
_EMULATOR_BINARY = (
    _REPO_ROOT / "transactional_emulator" / "target" / "release" / "transactional_emulator"
)

ASSEMBLY_NAME = "generated_asm_code.asm"
COMPILATION_ARTIFACT_NAME = "compilation_artifact.json"
OP_STATS_NAME = "op_stats.jsonl"
MANIFEST_NAME = "decode_run_manifest.json"
RUN_STATS_NAME = "rust_emulator_run_stats.json"


def _canonical_content_hash(value: Mapping[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_hash"}
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_calibration_ids(paths: list[Path]) -> list[str]:
    """Validate each retained calibration artifact and collect its id."""

    ids: list[str] = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError(f"{path} is not a {CALIBRATION_SCHEMA} artifact")
        if value.get("passed") is not True:
            raise ValueError(f"{path} did not pass its stage-validation gate")
        contract = value.get("execution_contract", {})
        if contract.get("timing_mode") != RTL_SERIALIZED:
            raise ValueError(
                f"{path} was calibrated under timing mode "
                f"{contract.get('timing_mode')!r}, not {RTL_SERIALIZED!r}"
            )
        calibration_id = value.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ValueError(f"{path} has no calibration id")
        ids.append(calibration_id)
    return ids


def _run_stage(command: list[str], *, cwd: Path) -> dict[str, str]:
    """Execute one stage subprocess and return its measured wall interval."""

    started = _utc_now()
    completed_process = subprocess.run(
        command,
        cwd=cwd,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    completed = _utc_now()
    if completed_process.returncode != 0:
        tail = "\n".join(completed_process.stdout.splitlines()[-25:])
        raise RuntimeError(
            f"stage command failed ({completed_process.returncode}): "
            f"{' '.join(command)}\n{tail}"
        )
    return {"started_at_utc": started, "completed_at_utc": completed}


def _stage_report(
    *,
    stage: str,
    interval: Mapping[str, str],
    command: list[str],
    artifacts: Mapping[str, str],
    calibration_ids: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": STAGE_REPORT_SCHEMA,
        "stage": stage,
        "provenance": {
            "started_at_utc": interval["started_at_utc"],
            "completed_at_utc": interval["completed_at_utc"],
            "command": list(command),
            "host": platform.node(),
        },
        "artifacts": dict(sorted(artifacts.items())),
        "calibration_ids": sorted(calibration_ids),
    }
    report["content_hash"] = _canonical_content_hash(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-size", type=int, default=128)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=_TESTBENCH / "build" / "stack_stage_reports",
        help="artifact directory for the measured compiler/emulator run",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=_REPO_ROOT / "plena_settings.toml",
    )
    parser.add_argument(
        "--isa-lib",
        type=Path,
        default=Path(__file__).resolve().parent / "customISA_lib.json",
    )
    parser.add_argument(
        "--request-memory-calibration",
        type=Path,
        default=_REPO_ROOT
        / "analytic_models"
        / "disagg_serve"
        / "calibration_dma_requests.csv",
    )
    parser.add_argument(
        "--calibration-artifact",
        type=Path,
        action="append",
        default=None,
        help="retained stage-calibration artifact to bind (repeatable)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calibration_paths = args.calibration_artifact or sorted(
        (Path(__file__).resolve().parent / "calibration").glob("decode_kv*.json")
    )
    if not calibration_paths:
        raise SystemExit("no retained calibration artifacts were supplied or found")
    retained_ids = _load_calibration_ids([Path(p) for p in calibration_paths])

    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    compiler_command = [
        sys.executable,
        str(_ASM_GEN),
        "--kv-size",
        str(args.kv_size),
        "--build-dir",
        str(build_dir),
    ]
    compiler_interval = _run_stage(compiler_command, cwd=_REPO_ROOT)
    assembly = build_dir / ASSEMBLY_NAME
    compilation_artifact = build_dir / COMPILATION_ARTIFACT_NAME
    for required in (assembly, compilation_artifact):
        if not required.is_file():
            raise SystemExit(f"compiler stage produced no {required.name}")

    emulator_command = [
        sys.executable,
        str(_DECODE_TEST),
        "--kv-size",
        str(args.kv_size),
        "--build-dir",
        str(build_dir),
    ]
    emulator_interval = _run_stage(emulator_command, cwd=_REPO_ROOT)
    op_stats = build_dir / OP_STATS_NAME
    run_manifest = build_dir / MANIFEST_NAME
    run_receipt = build_dir / RUN_STATS_NAME
    for required in (op_stats, run_manifest, run_receipt, _EMULATOR_BINARY):
        if not required.is_file():
            raise SystemExit(f"emulator stage produced no {required.name}")

    fresh_calibration = build_dir / f"stack_stage_calibration_kv{args.kv_size}.json"
    if fresh_calibration.exists():
        fresh_calibration.unlink()
    validation_command = [
        sys.executable,
        "-m",
        "analytic_models.performance.decode_stage_validation",
        "--asm",
        str(assembly),
        "--op-stats",
        str(op_stats),
        "--settings",
        str(args.settings.resolve()),
        "--isa-lib",
        str(args.isa_lib.resolve()),
        "--request-memory-calibration",
        str(args.request_memory_calibration.resolve()),
        "--emit-calibration",
        str(fresh_calibration),
    ]
    validation = subprocess.run(
        validation_command,
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if validation.returncode != 0:
        tail = "\n".join(validation.stdout.splitlines()[-25:])
        raise SystemExit(
            f"stage validation failed ({validation.returncode}):\n{tail}"
        )
    fresh_ids = _load_calibration_ids([fresh_calibration])
    bound_ids = retained_ids + fresh_ids

    compiler_report = _stage_report(
        stage="compiler",
        interval=compiler_interval,
        command=compiler_command[1:],
        artifacts={
            "assembly": _sha256_file(assembly),
            "compiler_artifact": _sha256_file(compilation_artifact),
            "isa_lib": _sha256_file(args.isa_lib.resolve()),
            "settings": _sha256_file(args.settings.resolve()),
        },
        calibration_ids=bound_ids,
    )
    emulator_report = _stage_report(
        stage="emulator",
        interval=emulator_interval,
        command=emulator_command[1:],
        artifacts={
            "op_stats": _sha256_file(op_stats),
            "run_manifest": _sha256_file(run_manifest),
            "run_receipt": _sha256_file(run_receipt),
            "emulator_binary": _sha256_file(_EMULATOR_BINARY),
        },
        calibration_ids=bound_ids,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in (
        ("compiler_report.json", compiler_report),
        ("emulator_report.json", emulator_report),
    ):
        destination = output_dir / name
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"{name}: {report['content_hash']}")
    print(f"bound calibration ids: {', '.join(bound_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
