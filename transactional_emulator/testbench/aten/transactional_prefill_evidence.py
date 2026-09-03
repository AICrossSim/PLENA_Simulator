"""Run complete S128 Mamba-2 and KDA transactional prefill evidence.

The two existing stage programs execute every chunk, carry recurrent state
between chunks, and write back every token output plus the final state.  This
driver gives them disposable build directories and preserves only a compact,
machine-readable report; the emulator's 512 MiB SRAM dumps are not artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tail(path: Path, lines: int = 80) -> str:
    content = path.read_text(errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _run_case(
    *,
    name: str,
    script: Path,
    arguments: tuple[str, ...],
    shape: dict[str, int],
    compiler_root: Path,
    temporary_root: Path,
) -> dict[str, object]:
    build_dir = temporary_root / name
    log_path = temporary_root / f"{name}.log"
    command = [
        sys.executable,
        str(script),
        *arguments,
        "--build-dir",
        str(build_dir),
    ]
    environment = {
        **os.environ,
        "NO_COLOR": "1",
        "PLENA_COMPILER_ROOT": str(compiler_root),
        "PLENA_USE_NIX_BUILD": "1",
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                (
                    str(compiler_root),
                    str(REPO_ROOT),
                    os.environ.get("PYTHONPATH"),
                ),
            )
        ),
    }
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"{name} transactional prefill failed with exit {result.returncode}:\n"
            f"{_tail(log_path)}"
        )

    stats_path = build_dir / "rust_emulator_run_stats.json"
    stats = json.loads(stats_path.read_text())
    comparison = stats.get("numerical_comparison", {})
    counters = stats.get("matrix_view_packet_counters", {})
    if not comparison.get("allclose_pass"):
        raise AssertionError(f"{name}: numerical comparison did not pass")
    if int(counters.get("bank_stall_cycles", -1)) != 0:
        raise AssertionError(f"{name}: Matrix-view bank stalls were not zero")

    return {
        "case": name,
        "shape": shape,
        "precision": {
            "hbm_inputs": "bf16",
            "matrix_sram": "bf16",
            "vector_sram": "bf16",
            "carried_state_and_spills": "bf16",
            "accumulation": "fp32",
        },
        "isa_source_lines": stats["artifacts"]["asm_source_lines"],
        "machine_words": stats["artifacts"]["machine_code_lines"],
        "rust_cycles": stats["sim_latency_cycles"],
        "hbm_bytes_read": stats["hbm_bytes_read"],
        "hbm_bytes_written": stats["hbm_bytes_written"],
        "matrix_view_packet_counters": counters,
        "numerical_comparison": comparison,
        "machine_code_sha256": _sha256(build_dir / "generated_machine_code.mem"),
        "hbm_preload_sha256": _sha256(build_dir / "hbm_for_behave_sim.bin"),
    }


def run_evidence(*, compiler_root: Path, output: Path) -> dict[str, object]:
    compiler_root = compiler_root.resolve()
    if not (compiler_root / "aten" / "plena" / "compiler.py").exists():
        raise FileNotFoundError(f"PLENA Compiler checkout is invalid: {compiler_root}")

    with tempfile.TemporaryDirectory(prefix="plena-transactional-prefill-") as tmp:
        temporary_root = Path(tmp)
        cases = [
            _run_case(
                name="nemotron_mamba2_s128",
                script=REPO_ROOT
                / "transactional_emulator/testbench/mamba2/mamba2_stage_test.py",
                arguments=("--case", "prefill_s128_full"),
                shape={
                    "batch": 1,
                    "tokens": 128,
                    "chunk": 64,
                    "heads": 1,
                    "state_dim": 64,
                    "head_dim": 64,
                },
                compiler_root=compiler_root,
                temporary_root=temporary_root,
            ),
            _run_case(
                name="kimi_kda_s128",
                script=REPO_ROOT
                / "transactional_emulator/testbench/kda/kda_stage_test.py",
                arguments=("--case", "prefill_s128_full", "--chunk", "16"),
                shape={
                    "batch": 1,
                    "tokens": 128,
                    "chunk": 16,
                    "heads": 1,
                    "key_dim": 64,
                    "value_dim": 64,
                },
                compiler_root=compiler_root,
                temporary_root=temporary_root,
            ),
        ]

    summary: dict[str, object] = {
        "schema_version": 1,
        "evidence": (
            "Compiler assembly -> assembler -> Rust transactional emulator -> "
            "all 128 outputs and final recurrent state"
        ),
        "cases": cases,
        "claim_boundary": (
            "complete recurrence/chunk prefill at reduced 64-wide geometry; "
            "not a complete real-weight Nemotron or Kimi layer"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler-root",
        type=Path,
        default=REPO_ROOT / "PLENA_Compiler",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_evidence(
        compiler_root=args.compiler_root,
        output=args.output.resolve(),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
