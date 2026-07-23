#!/usr/bin/env python3
"""Window 1 P1 timing validation gates G1/G2/G5."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.moe_timing.replay.utils import (
    DEFAULT_OUT_ROOT,
    ensure_python_paths,
    load_json,
    write_json,
)


ensure_python_paths()

from compiler.assembler.assembly_to_binary import AssemblyToBinary  # noqa: E402
from transactional_emulator.testbench.aten.configurable import HardwareConfig  # noqa: E402
from transactional_emulator.testbench.emulator_runner import run_emulator  # noqa: E402


MLEN = 64
VLEN = 64
BLEN = 4
HLEN = 16
BROADCAST_AMOUNT = 4
HBM_SIZE = 1 << 20
SCALE_BASE = 65_536
PREFETCH_COUNT = 64
COMPUTE_REPS = 100


def write_settings(build_dir: Path) -> Path:
    hw = HardwareConfig(
        mlen=MLEN,
        vlen=VLEN,
        blen=BLEN,
        hlen=HLEN,
        broadcast_amount=BROADCAST_AMOUNT,
        dc_en=None,
        latency_profile=None,
        hbm_m_prefetch_amount=MLEN,
        hbm_v_prefetch_amount=BLEN,
        hbm_v_writeback_amount=BLEN,
    )
    return hw.write_toml(build_dir)


def assembler() -> AssemblyToBinary:
    repo_root = Path(__file__).resolve().parents[4]
    compiler_root = repo_root / "PLENA_Compiler"
    return AssemblyToBinary(
        str(compiler_root / "doc" / "operation.svh"),
        str(compiler_root / "doc" / "configuration.svh"),
    )


def setup_prefetch(stride: int, *, dest: int = 4096) -> list[str]:
    return [
        f"S_ADDI_INT gp4, gp0, {SCALE_BASE}",
        "C_SET_SCALE_REG gp4",
        f"S_ADDI_INT gp5, gp0, {stride}",
        "C_SET_STRIDE_REG gp5",
        f"S_ADDI_INT gp6, gp0, {dest}",
        "S_ADDI_INT gp1, gp0, 0",
        "C_SET_ADDR_REG gp3, gp0, gp1",
    ]


def prefetch_offsets_program(name: str, *, stride: int, offsets: list[int]) -> str:
    lines = [f"; {name}: H_PREFETCH_V address-pattern microbench"]
    lines += setup_prefetch(stride)
    for offset in offsets:
        lines += [
            f"S_ADDI_INT gp2, gp0, {offset}",
            "H_PREFETCH_V gp6, gp2, gp3, 1, 0",
        ]
    return "\n".join(lines)


def prefetch_one_program() -> str:
    return "\n".join(
        [
            "; G2 prefetch-only component",
            *setup_prefetch(VLEN),
            "S_ADDI_INT gp2, gp0, 0",
            "H_PREFETCH_V gp6, gp2, gp3, 1, 0",
        ]
    )


def compute_program(reps: int = COMPUTE_REPS) -> str:
    return "\n".join(["; G2 compute-only independent M_MV sequence"] + ["M_MV gp0, gp0, gp0"] * reps)


def combined_prefetch_compute_program() -> str:
    return "\n".join(
        [
            "; G2 prefetch then independent compute",
            *setup_prefetch(VLEN),
            "S_ADDI_INT gp2, gp0, 0",
            "H_PREFETCH_V gp6, gp2, gp3, 1, 0",
            *["M_MV gp0, gp0, gp0" for _ in range(COMPUTE_REPS)],
        ]
    )


def emit_case(out_root: Path, name: str, asm: str) -> Path:
    build_dir = out_root / "timing_gate_runs" / name
    build_dir.mkdir(parents=True, exist_ok=True)
    settings = write_settings(build_dir)
    (build_dir / "generated_asm_code.asm").write_text(asm.strip() + "\n", encoding="utf-8")
    assembler().generate_binary(
        build_dir / "generated_asm_code.asm",
        build_dir / "generated_machine_code.mem",
    )
    (build_dir / "hbm_for_behave_sim.bin").write_bytes(bytes(HBM_SIZE))
    (build_dir / "hbm_size.txt").write_text(f"{HBM_SIZE}\n", encoding="utf-8")
    (build_dir / "fp_sram.bin").write_bytes(bytes(2048))
    (build_dir / "int_sram.bin").write_bytes(bytes(4096))
    return settings


def run_case(
    out_root: Path,
    name: str,
    asm: str,
    *,
    overlap_prefetch_compute: bool = False,
    stage_profile: bool = True,
) -> dict[str, Any]:
    build_dir = out_root / "timing_gate_runs" / name
    settings = emit_case(out_root, name, asm)
    import os

    old_settings = os.environ.get("PLENA_SETTINGS_TOML")
    os.environ["PLENA_SETTINGS_TOML"] = str(settings)
    try:
        metrics = run_emulator(
            build_dir,
            hbm_size=HBM_SIZE,
            threads=1,
            stage_profile=stage_profile,
            overlap_prefetch_compute=overlap_prefetch_compute,
        )
    finally:
        if old_settings is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = old_settings
    stage_profile_path = Path(metrics.get("stage_profile_path", build_dir / "stage_profile.json"))
    result = {
        "name": name,
        "build_dir": str(build_dir),
        "sim_latency_cycles": metrics.get("sim_latency_cycles"),
        "hbm_bytes_read": metrics.get("hbm_bytes_read"),
        "hbm_bytes_written": metrics.get("hbm_bytes_written"),
        "overlap_prefetch_compute": bool(overlap_prefetch_compute),
        "stage_profile_requested": bool(stage_profile),
        "run_stats_path": metrics.get("stats_path"),
        "stage_profile_path": str(stage_profile_path),
        "stage_profile": load_json(stage_profile_path) if stage_profile_path.exists() else {},
    }
    write_json(build_dir / "timing_gate_case_result.json", result)
    return result


def g1_cases() -> dict[str, str]:
    continuous_offsets = [idx * (BLEN * VLEN) for idx in range(PREFETCH_COUNT)]
    # Keep offsets within S_ADDI_INT's compact immediate range.
    sparse_offsets = [idx * 4096 for idx in range(PREFETCH_COUNT)]
    rng = random.Random(20260701)
    random_offsets = continuous_offsets[:]
    rng.shuffle(random_offsets)
    row_hit_offsets = [0 for _ in range(PREFETCH_COUNT)]
    return {
        "g1_continuous_prefetch_v": prefetch_offsets_program(
            "continuous",
            stride=VLEN,
            offsets=continuous_offsets,
        ),
        "g1_sparse_cross_region_prefetch_v": prefetch_offsets_program(
            "sparse-cross-region",
            stride=VLEN,
            offsets=sparse_offsets,
        ),
        "g1_random_order_prefetch_v": prefetch_offsets_program(
            "random-order-same-address-set",
            stride=VLEN,
            offsets=random_offsets,
        ),
        "g1_row_hit_repeat_prefetch_v": prefetch_offsets_program(
            "row-hit-repeat",
            stride=VLEN,
            offsets=row_hit_offsets,
        ),
    }


def evaluate_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["name"]: row for row in rows}
    g1_candidates = [
        by_name["g1_sparse_cross_region_prefetch_v"],
        by_name["g1_random_order_prefetch_v"],
    ]
    continuous = by_name["g1_continuous_prefetch_v"]
    row_hit = by_name["g1_row_hit_repeat_prefetch_v"]
    same_bytes = all(row["hbm_bytes_read"] == continuous["hbm_bytes_read"] for row in [*g1_candidates, row_hit])
    random_gt_continuous = any(row["sim_latency_cycles"] > continuous["sim_latency_cycles"] for row in g1_candidates)
    random_gt_row_hit = any(row["sim_latency_cycles"] > row_hit["sim_latency_cycles"] for row in g1_candidates)
    address_sensitive = len({row["sim_latency_cycles"] for row in [*g1_candidates, continuous, row_hit]}) > 1

    prefetch_only = by_name["g2_prefetch_only"]
    compute_only = by_name["g2_compute_only"]
    combined = by_name["g2_prefetch_then_compute"]
    combined_overlap = by_name["g2_prefetch_then_compute_overlap_on"]
    prefetch_plus_compute = int(prefetch_only["sim_latency_cycles"]) + int(compute_only["sim_latency_cycles"])
    serial_hidden_cycles = prefetch_plus_compute - int(combined["sim_latency_cycles"])
    overlay_hidden_cycles = int(combined["sim_latency_cycles"]) - int(combined_overlap["sim_latency_cycles"])
    overlay_hbm_bytes_match = (
        combined["hbm_bytes_read"] == combined_overlap["hbm_bytes_read"]
        and combined["hbm_bytes_written"] == combined_overlap["hbm_bytes_written"]
    )

    m_mv = by_name["g5_single_m_mv"]
    m_mm = by_name["g5_single_m_mm"]

    profiled_rows = [row for row in rows if row.get("stage_profile_requested")]
    bytes_match = all(
        row["hbm_bytes_read"] == row["stage_profile"].get("total_hbm_bytes_read")
        and row["hbm_bytes_written"] == row["stage_profile"].get("total_hbm_bytes_written")
        for row in profiled_rows
    )
    stage_cycles_match = all(
        row["sim_latency_cycles"]
        == row["stage_profile"].get("total_simulation_cycles")
        == row["stage_profile"].get("total_stage_wall_cycles")
        for row in profiled_rows
    )

    return {
        "g1_ramulator_address_sensitivity": {
            "continuous_cycles": continuous["sim_latency_cycles"],
            "row_hit_repeat_cycles": row_hit["sim_latency_cycles"],
            "candidates": [
                {
                    "name": row["name"],
                    "cycles": row["sim_latency_cycles"],
                    "hbm_bytes_read": row["hbm_bytes_read"],
                }
                for row in g1_candidates
            ],
            "same_hbm_read_bytes": same_bytes,
            "random_or_sparse_gt_continuous": random_gt_continuous,
            "random_or_sparse_gt_row_hit_locality": random_gt_row_hit,
            "address_sensitive": address_sensitive,
            "pass": same_bytes and random_gt_continuous,
            "note": "Strict pass requires random/sparse same-byte access to be slower than the continuous unique baseline. Row-hit locality is reported separately as weaker evidence.",
        },
        "g2_prefetch_compute_overlap": {
            "prefetch_only_cycles": prefetch_only["sim_latency_cycles"],
            "compute_only_cycles": compute_only["sim_latency_cycles"],
            "prefetch_plus_compute_cycles": prefetch_plus_compute,
            "default_combined_cycles": combined["sim_latency_cycles"],
            "overlap_enabled_combined_cycles": combined_overlap["sim_latency_cycles"],
            "default_hidden_cycles": serial_hidden_cycles,
            "overlay_hidden_cycles": overlay_hidden_cycles,
            "overlap_hbm_bytes_match_default": overlay_hbm_bytes_match,
            # Assert the structural relationship rather than an absolute baseline:
            # serial_hidden_cycles == 0 means the default run is exactly
            # prefetch_only + compute_only (no accidental overlap), which is
            # robust to timing-model rebaselines that a hardcoded cycle count is not.
            "pass": (
                serial_hidden_cycles == 0
                and overlay_hidden_cycles > 0
                and int(combined_overlap["sim_latency_cycles"]) < int(combined["sim_latency_cycles"])
                and overlay_hbm_bytes_match
            ),
            "note": "Default must remain serial; only --experimental-overlap-prefetch-compute may reduce cycles for this independent prefetch+compute pattern.",
        },
        "g5_matrix_formula_self_consistency": {
            "mlen": MLEN,
            "expected_m_mv_cycles": MLEN,
            "measured_m_mv_cycles": m_mv["sim_latency_cycles"],
            "expected_m_mm_cycles": MLEN,
            "measured_m_mm_cycles": m_mm["sim_latency_cycles"],
            "pass": m_mv["sim_latency_cycles"] == MLEN and m_mm["sim_latency_cycles"] == MLEN,
        },
        "stage_profile_accounting": {
            "stage_cycles_match_total": stage_cycles_match,
            "physical_hbm_bytes_match_run_stats": bytes_match,
            "pass": stage_cycles_match and bytes_match,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    cases: dict[str, str] = {}
    cases.update(g1_cases())
    cases.update(
        {
            "g2_prefetch_only": prefetch_one_program(),
            "g2_compute_only": compute_program(),
            "g2_prefetch_then_compute": combined_prefetch_compute_program(),
            "g5_single_m_mv": "M_MV gp0, gp0, gp0\n",
            "g5_single_m_mm": "M_MM gp0, gp0, gp0\n",
        }
    )
    rows = [run_case(args.out_root, name, asm) for name, asm in cases.items()]
    rows.append(
        run_case(
            args.out_root,
            "g2_prefetch_then_compute_overlap_on",
            combined_prefetch_compute_program(),
            overlap_prefetch_compute=True,
            stage_profile=False,
        )
    )
    gates = evaluate_gates(rows)
    summary = {
        "schema_version": 1,
        "output_root": str(args.out_root),
        "hardware": {
            "mlen": MLEN,
            "vlen": VLEN,
            "blen": BLEN,
            "hlen": HLEN,
            "broadcast_amount": BROADCAST_AMOUNT,
            "hbm_size": HBM_SIZE,
            "prefetch_count": PREFETCH_COUNT,
            "compute_reps": COMPUTE_REPS,
        },
        "cases": [{key: value for key, value in row.items() if key != "stage_profile"} for row in rows],
        "gates": gates,
    }
    out_path = args.out_root / "timing_gates_summary.json"
    write_json(out_path, summary)
    print(out_path)
    print(gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
