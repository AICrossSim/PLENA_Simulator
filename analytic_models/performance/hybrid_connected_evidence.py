"""Run and summarize the executable non-GPU hybrid L-Compute gates.

The analytic campaign uses official model dimensions and full 52/93-layer
timelines. This companion campaign answers the orthogonal functional questions
with the Rust transactional emulator:

* can Matrix final writeback and an affine Vector consumer exchange real data;
* can S128 prefill state feed one packetized decode update;
* do B=1/2/4/8/16 requests keep independent recurrent state.

The batch functional cases deliberately use reduced outer dimensions so the
full contents can be compared in CI-sized memory. Official batch dimensions and
traffic remain the responsibility of ``hybrid_lcompute_campaign``; this file
does not rename a reduced numerical test as a full-model execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SIMULATOR_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ConnectedCase:
    name: str
    command: tuple[str, ...]
    build_dir: Path
    scope: str
    require_l_cfg: bool = False
    require_conflict_free_packet: bool = False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest()


def _git_revision(path: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty_during_campaign": dirty}


def _fixed_cases(python: str) -> tuple[ConnectedCase, ...]:
    return (
        ConnectedCase(
            name="matrix_affine_writeback",
            command=(python, "transactional_emulator/testbench/aten/affine_projection_test.py"),
            build_dir=SIMULATOR_ROOT / "transactional_emulator/testbench/aten/build/affine_projection",
            scope="Matrix final writeback -> affine bank placement -> ordinary Vector lane restore",
            require_l_cfg=True,
        ),
        ConnectedCase(
            name="nemotron_mamba_s128_handoff",
            command=(
                python,
                "transactional_emulator/testbench/mamba2/mamba2_stage_test.py",
                "--case",
                "prefill_s128_decode_handoff",
            ),
            build_dir=(
                SIMULATOR_ROOT
                / "transactional_emulator/testbench/mamba2/build/mamba2_prefill_s128_decode_handoff"
            ),
            scope="two 64-token Mamba chunks -> affine state handoff -> one packetized decode step",
            require_l_cfg=True,
            require_conflict_free_packet=True,
        ),
        ConnectedCase(
            name="kimi_kda_s128_handoff",
            command=(
                python,
                "transactional_emulator/testbench/kda/kda_stage_test.py",
                "--case",
                "prefill_s128_decode_handoff",
                "--chunk",
                "16",
            ),
            build_dir=(
                SIMULATOR_ROOT
                / "transactional_emulator/testbench/kda/build/kda_prefill_s128_decode_handoff"
            ),
            scope="eight 16-token KDA chunks -> state transpose/affine handoff -> one packetized decode step",
            require_l_cfg=True,
            require_conflict_free_packet=True,
        ),
    )


def _batch_cases(python: str, batch_sizes: tuple[int, ...]) -> tuple[ConnectedCase, ...]:
    cases: list[ConnectedCase] = []
    for batch in batch_sizes:
        cases.extend(
            (
                ConnectedCase(
                    name=f"mamba_private_state_b{batch}",
                    command=(
                        python,
                        "transactional_emulator/testbench/mamba2/mamba2_stage_test.py",
                        "--case",
                        "decode_batch",
                        "--batch-size",
                        str(batch),
                    ),
                    build_dir=(
                        SIMULATOR_ROOT
                        / "transactional_emulator/testbench/mamba2/build/mamba2_decode_batch"
                    ),
                    scope=(
                        f"B{batch} independent Mamba states; reduced 4-head x 64-state x 64-head-dim "
                        "functional geometry"
                    ),
                ),
                ConnectedCase(
                    name=f"kda_private_state_b{batch}",
                    command=(
                        python,
                        "transactional_emulator/testbench/kda/kda_stage_test.py",
                        "--case",
                        "recurrent_batch",
                        "--batch-size",
                        str(batch),
                        "--mlen",
                        "128",
                        "--key-dim",
                        "128",
                    ),
                    build_dir=(
                        SIMULATOR_ROOT
                        / "transactional_emulator/testbench/kda/build/kda_recurrent_batch"
                    ),
                    scope=(
                        f"B{batch} independent KDA states; one full 128-key x 128-value head "
                        "with a request-reused FPRAM scalar window"
                    ),
                ),
            )
        )
    return tuple(cases)


def _run_case(case: ConnectedCase, *, env: dict[str, str]) -> dict[str, Any]:
    print(f"\n=== {case.name} ===", flush=True)
    subprocess.run(case.command, cwd=SIMULATOR_ROOT, env=env, check=True)

    stats_path = case.build_dir / "rust_emulator_run_stats.json"
    if not stats_path.exists():
        raise RuntimeError(f"{case.name}: missing {stats_path}")
    stats = json.loads(stats_path.read_text())
    comparison = stats.get("numerical_comparison")
    if not comparison or not comparison.get("allclose_pass"):
        raise RuntimeError(f"{case.name}: numerical comparison did not pass")

    counters = stats.get("lstream_packet_counters")
    asm_path = case.build_dir / "generated_asm_code.asm"
    if case.require_l_cfg and (not asm_path.exists() or "L_CFG" not in asm_path.read_text()):
        raise RuntimeError(f"{case.name}: L_CFG path was not exercised")
    if case.require_conflict_free_packet:
        if not counters or counters["packet_reads"] <= 0:
            raise RuntimeError(f"{case.name}: L-Compute packet path was not exercised")
        if counters["packet_service_cycles"] != counters["packet_bandwidth_floor_cycles"]:
            raise RuntimeError(f"{case.name}: packet service did not reach its bandwidth floor")
        if counters["packet_conflict_stall_cycles"] != 0:
            raise RuntimeError(f"{case.name}: affine packet still has bank conflicts")

    files = {}
    for name in ("generated_asm_code.asm", "generated_machine_code.mem", "golden_result.txt"):
        path = case.build_dir / name
        if path.exists():
            files[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}

    return {
        "name": case.name,
        "scope": case.scope,
        "command": list(case.command),
        "sim_latency_cycles": stats.get("sim_latency_cycles"),
        "hbm_bytes_read": stats.get("hbm_bytes_read"),
        "hbm_bytes_written": stats.get("hbm_bytes_written"),
        "isa_source_lines": stats.get("artifacts", {}).get("asm_source_lines"),
        "machine_code_lines": stats.get("artifacts", {}).get("machine_code_lines"),
        "lstream_packet_counters": counters,
        "numerical_comparison": comparison,
        "files": files,
    }


def run_campaign(
    *,
    compiler_root: Path,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    if tuple(sorted(set(batch_sizes))) != batch_sizes or not batch_sizes or batch_sizes[0] != 1:
        raise ValueError("batch_sizes must be unique, increasing and start at 1")
    if not (compiler_root / "aten/plena/compiler.py").exists():
        raise FileNotFoundError(f"not a PLENA Compiler checkout: {compiler_root}")

    env = {
        **os.environ,
        "PLENA_COMPILER_ROOT": str(compiler_root),
        "PYTHONPATH": os.pathsep.join(
            value
            for value in (str(SIMULATOR_ROOT), str(compiler_root), os.environ.get("PYTHONPATH", ""))
            if value
        ),
    }
    cases = (*_fixed_cases(sys.executable), *_batch_cases(sys.executable, batch_sizes))
    records = [_run_case(case, env=env) for case in cases]

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "simulator": _git_revision(SIMULATOR_ROOT),
        "compiler": _git_revision(compiler_root),
        "batch_sizes": list(batch_sizes),
        "records": records,
        "claims": {
            "matrix_to_vector_affine_handoff": "executable and numerically verified",
            "s128_prefill_to_packet_decode": "executable for Mamba and KDA",
            "request_private_state": "executable for B=1/2/4/8/16",
            "bank_conflict": "zero stalls only for the affine packet cases explicitly listed above",
        },
        "claim_boundary": {
            "functional_batch_geometry": "reduced outer dimensions; every state value is checked",
            "official_dimensions": "covered by the separate full-model analytic campaign",
            "prefill_packetization": (
                "prefill itself uses the existing chunked path; the state handoff and following decode "
                "use affine packet addressing"
            ),
            "weights": "deterministic synthetic values; this artifact is not a real-checkpoint execution",
            "gpu_routing": "not required and not used",
            "rtl": "not evaluated",
        },
    }
    report["report_sha256"] = _sha256_json(report)
    return report


def _parse_batch_sizes(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in raw.split(",") if value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler-root",
        type=Path,
        default=SIMULATOR_ROOT / "PLENA_Compiler",
    )
    parser.add_argument(
        "--batch-sizes",
        type=_parse_batch_sizes,
        default=(1, 2, 4, 8, 16),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=SIMULATOR_ROOT / "artifacts/hybrid_lcompute_connected_v1/evidence.json",
    )
    args = parser.parse_args(argv)

    report = run_campaign(
        compiler_root=args.compiler_root.resolve(),
        batch_sizes=args.batch_sizes,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {args.json_out}")
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
