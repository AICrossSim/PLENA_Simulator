#!/usr/bin/env python3
"""Measure HBM burst coalescing and DMA/compute overlap independently.

The runner reuses an existing compiler build directory in place. Large HBM
images are never copied, and emulator dumps are hashed then removed from a
temporary directory after every run.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
for dependency in (REPO_ROOT, REPO_ROOT / "PLENA_Tools"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from verification.check_mem import compare_vram_with_golden  # noqa: E402


VARIANTS = (
    {
        "id": "legacy_serial",
        "description": "Current 64B requests, serial instruction timing",
        "timing_model": "serial",
        "coalesce": False,
        "serialize_scoreboard": False,
    },
    {
        "id": "coalesced_serial",
        "description": "Duplicate 64B requests coalesced, serial instruction timing",
        "timing_model": "serial",
        "coalesce": True,
        "serialize_scoreboard": False,
    },
    {
        "id": "coalesced_scoreboard_serialized",
        "description": "Coalesced requests, scoreboard forced to reproduce serial timing",
        "timing_model": "scoreboard",
        "coalesce": True,
        "serialize_scoreboard": True,
    },
    {
        "id": "coalesced_scoreboard",
        "description": "Coalesced requests, dependency-aware asynchronous timing",
        "timing_model": "scoreboard",
        "coalesce": True,
        "serialize_scoreboard": False,
    },
)

SIM_RE = re.compile(r"Simulation completed\. Latency\s+([0-9.eE+-]+)ns(?:\s+cycles\s+([0-9]+))?")
HBM_RE = re.compile(
    r"HBM Statistics - Bytes read:\s*([0-9]+)\s*\|\s*"
    r"Bytes written:\s*([0-9]+)\s*\|\s*"
    r"Utilization:\s*([0-9.eE+-]+)\s*bytes/sec"
)
SCOREBOARD_RE = re.compile(r"Scoreboard timing model summary")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hbm_size(build_dir: Path) -> int:
    sidecar = build_dir / "hbm_size.txt"
    if sidecar.exists():
        value = int(sidecar.read_text().strip())
    else:
        value = 2 * (build_dir / "hbm_for_behave_sim.bin").stat().st_size
    preload = (build_dir / "hbm_for_behave_sim.bin").stat().st_size
    return max(value, ((preload + 63) // 64) * 64)


def required_inputs(build_dir: Path) -> dict[str, Path]:
    paths = {
        "opcode": build_dir / "generated_machine_code.mem",
        "hbm": build_dir / "hbm_for_behave_sim.bin",
        "fpsram": build_dir / "fp_sram.bin",
        "intsram": build_dir / "int_sram.bin",
        "settings": build_dir / "plena_settings.toml",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing emulator inputs: " + ", ".join(missing))
    return paths


def detect_weight_layout(build_dir: Path) -> str:
    layout_path = build_dir / "tensor_layouts.json"
    if not layout_path.exists():
        return "unknown"
    layouts = json.loads(layout_path.read_text())
    orders = {
        str(layout.get("storage_order", "row_major"))
        for name, layout in layouts.items()
        if str(name).startswith("W_")
    }
    if not orders:
        return "unknown"
    if len(orders) != 1:
        return "mixed:" + ",".join(sorted(orders))
    return orders.pop()


def libtorch_env(binary: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_WAIT_POLICY": "PASSIVE",
            "RUST_BACKTRACE": "1",
            "RUST_LOG": "warn,transactional_emulator=info",
        }
    )
    target = binary.parent.parent
    candidates = list(target.glob("build/torch-sys-*/out/libtorch/libtorch/lib"))
    if candidates:
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{candidates[0]}:{old}" if old else str(candidates[0])
    return env


def compare_output_to_golden(*, build_dir: Path, vram_dump: Path) -> dict[str, object]:
    params_path = build_dir / "comparison_params.json"
    golden_path = build_dir / "golden_result.txt"
    if not params_path.exists() or not golden_path.exists():
        raise FileNotFoundError("Functional gate requires comparison_params.json and golden_result.txt")

    params = json.loads(params_path.read_text())
    with (build_dir / "plena_settings.toml").open("rb") as stream:
        settings = tomllib.load(stream)
    data_type = settings["TRANSACTIONAL"]["PRECISION"]["VECTOR_SRAM_TYPE"]["DATA_TYPE"]
    exp_width = int(data_type["exponent"])
    man_width = int(data_type["mantissa"])
    total_bits = exp_width + man_width + (1 if bool(data_type.get("sign", True)) else 0)

    # The verifier is intentionally verbose. Keep raw details out of the compact
    # ablation log while retaining the scalar metrics in the run manifest.
    with contextlib.redirect_stdout(io.StringIO()):
        result = compare_vram_with_golden(
            vram_dump,
            golden_path,
            exp_width=exp_width,
            man_width=man_width,
            num_bytes_per_val=max(1, (total_bits + 7) // 8),
            row_dim=params.get("row_dim", 64),
            start_row_idx=params["start_row_idx"],
            num_batches=params["num_batches"],
            num_rows=params["num_rows"],
            elements_per_batch=params["elements_per_batch"],
            atol=params.get("atol", 0.2),
            rtol=params.get("rtol", 0.2),
            use_stride_mode=params.get("use_stride_mode", True),
            use_slice_mode=params.get("use_slice_mode", False),
            slice_per_row=params.get("slice_per_row"),
            physical_rows=params.get("physical_rows"),
            rows_per_batch=params.get("rows_per_batch"),
            active_seq=params.get("active_seq_per_batch"),
        )
    golden = result["golden_values"].float()
    simulated = result["simulated_values"].float()
    diff_rms = math.sqrt(float(((simulated - golden) ** 2).mean().item()))
    golden_rms = math.sqrt(float((golden**2).mean().item()))
    rel_rms = diff_rms / golden_rms if golden_rms else diff_rms
    rel_rms_threshold = float(params.get("rel_rms_threshold", 0.01))
    return {
        "reference_pass": rel_rms <= rel_rms_threshold,
        "rel_rms": rel_rms,
        "rel_rms_threshold": rel_rms_threshold,
        "exact_match_rate": float(result["allclose_match_rate"]),
        "legacy_verifier_pass": bool(result["allclose_pass"]),
        "mse": float(result["mse"]),
        "mae": float(result["mae"]),
        "max_error": float(result["max_error"]),
        "relative_error": float(result["relative_error"]),
    }


def run_once(
    *,
    binary: Path,
    build_dir: Path,
    inputs: dict[str, Path],
    variant: dict[str, object],
    repeat: int,
    output_dir: Path,
    temp_root: Path,
) -> dict[str, object]:
    run_dir = temp_root / f"{variant['id']}.repeat{repeat:02d}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    command = [
        str(binary),
        "--opcode",
        str(inputs["opcode"]),
        "--hbm",
        str(inputs["hbm"]),
        "--fpsram",
        str(inputs["fpsram"]),
        "--intsram",
        str(inputs["intsram"]),
        "--settings",
        str(inputs["settings"]),
        "--hbm-size",
        str(hbm_size(build_dir)),
    ]
    vram_preload = build_dir / "vram_preload.bin"
    if vram_preload.exists():
        command += ["--vram", str(vram_preload)]
    if variant["timing_model"] == "scoreboard":
        command += ["--timing-model", "scoreboard"]
    if variant["serialize_scoreboard"]:
        command.append("--scoreboard-serialize")
    if variant["coalesce"]:
        command.append("--coalesce-hbm-bursts")

    log_path = output_dir / f"{variant['id']}.repeat{repeat:02d}.log"
    started = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=run_dir,
        env=libtorch_env(binary),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )
    host_seconds = time.perf_counter() - started
    log_path.write_text(proc.stdout, encoding="utf-8")

    sim_match = SIM_RE.search(proc.stdout)
    hbm_match = HBM_RE.search(proc.stdout)
    if proc.returncode != 0 or sim_match is None or hbm_match is None:
        raise RuntimeError(
            f"{variant['id']} repeat {repeat} failed or omitted required metrics; "
            f"exit={proc.returncode}, log={log_path}"
        )

    vram_dump = run_dir / "vram_dump.bin"
    if not vram_dump.exists():
        raise RuntimeError(f"{variant['id']} repeat {repeat} emitted no VRAM dump")
    functional = compare_output_to_golden(build_dir=build_dir, vram_dump=vram_dump)

    dump_hashes = {}
    dump_sizes = {}
    for name in ("mram_dump.bin", "vram_dump.bin", "fpsram_dump.bin", "intsram_dump.bin"):
        dump = run_dir / name
        if dump.exists():
            dump_hashes[name] = sha256_file(dump)
            dump_sizes[name] = dump.stat().st_size
            dump.unlink()
    shutil.rmtree(run_dir)

    return {
        "variant": variant["id"],
        "description": variant["description"],
        "repeat": repeat,
        "timing_model": variant["timing_model"],
        "coalesce_hbm_bursts": variant["coalesce"],
        "scoreboard_serialize": variant["serialize_scoreboard"],
        "total_cycles": int(sim_match.group(2)) if sim_match.group(2) else round(float(sim_match.group(1))),
        "sim_latency_ns": float(sim_match.group(1)),
        "hbm_bytes_read": int(hbm_match.group(1)),
        "hbm_bytes_written": int(hbm_match.group(2)),
        "hbm_utilization_bytes_per_second": float(hbm_match.group(3)),
        "scoreboard_summary_present": bool(SCOREBOARD_RE.search(proc.stdout)),
        "host_wall_time_seconds": host_seconds,
        "return_code": proc.returncode,
        "dump_sha256": dump_hashes,
        "dump_sizes": dump_sizes,
        "functional": functional,
        "log_path": str(log_path),
        "command": command,
    }


def stable_value(rows: list[dict[str, object]], key: str) -> object:
    values = [row[key] for row in rows]
    if len(set(values)) != 1:
        raise RuntimeError(f"Repeat gate failed for {rows[0]['variant']} {key}: {values}")
    return values[0]


def stable_dump_hash(rows: list[dict[str, object]], name: str) -> str:
    values = [str(row["dump_sha256"][name]) for row in rows]
    if len(set(values)) != 1:
        raise RuntimeError(f"Repeat gate failed for {rows[0]['variant']} {name}: {values}")
    return values[0]


def stable_functional_value(rows: list[dict[str, object]], key: str) -> object:
    values = [row["functional"][key] for row in rows]
    if len(set(values)) != 1:
        raise RuntimeError(f"Repeat gate failed for {rows[0]['variant']} functional.{key}: {values}")
    return values[0]


def write_outputs(
    *,
    rows: list[dict[str, object]],
    output_dir: Path,
    build_dir: Path,
    binary: Path,
    repeats: int,
) -> None:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["variant"])].append(row)

    summary_rows = []
    for variant in VARIANTS:
        group = groups[str(variant["id"])]
        summary_rows.append(
            {
                "variant": variant["id"],
                "description": variant["description"],
                "total_cycles": stable_value(group, "total_cycles"),
                "hbm_bytes_read": stable_value(group, "hbm_bytes_read"),
                "hbm_bytes_written": stable_value(group, "hbm_bytes_written"),
                "vram_sha256": stable_dump_hash(group, "vram_dump.bin"),
                "functional_pass": stable_functional_value(group, "reference_pass"),
                "functional_rel_rms": stable_functional_value(group, "rel_rms"),
                "functional_exact_match_rate": stable_functional_value(group, "exact_match_rate"),
                "host_seconds_mean": sum(float(row["host_wall_time_seconds"]) for row in group) / len(group),
            }
        )

    by_id = {str(row["variant"]): row for row in summary_rows}
    legacy = by_id["legacy_serial"]
    coalesced = by_id["coalesced_serial"]
    serialized = by_id["coalesced_scoreboard_serialized"]
    async_row = by_id["coalesced_scoreboard"]

    all_hashes = {str(row["vram_sha256"]) for row in summary_rows}
    gates = {
        "repeat_cycles_identical": True,
        "repeat_hbm_bytes_identical": True,
        "functional_vram_hashes_identical": len(all_hashes) == 1,
        "all_runs_match_pytorch_golden": all(bool(row["functional_pass"]) for row in summary_rows),
        "scoreboard_serialized_matches_serial": serialized["total_cycles"] == coalesced["total_cycles"],
        "scoreboard_serialized_bytes_match_serial": serialized["hbm_bytes_read"] == coalesced["hbm_bytes_read"],
    }
    metrics = {
        "burst_coalescing_cycle_speedup": legacy["total_cycles"] / coalesced["total_cycles"],
        "burst_coalescing_physical_byte_reduction_pct": 100.0
        * (legacy["hbm_bytes_read"] - coalesced["hbm_bytes_read"])
        / legacy["hbm_bytes_read"],
        "async_incremental_cycle_speedup": coalesced["total_cycles"] / async_row["total_cycles"],
        "combined_cycle_speedup": legacy["total_cycles"] / async_row["total_cycles"],
    }

    with (output_dir / "ablation_runs.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "variant",
            "repeat",
            "timing_model",
            "coalesce_hbm_bursts",
            "scoreboard_serialize",
            "total_cycles",
            "hbm_bytes_read",
            "hbm_bytes_written",
            "hbm_utilization_bytes_per_second",
            "functional",
            "host_wall_time_seconds",
            "log_path",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    weight_layout = detect_weight_layout(build_dir)
    manifest = {
        "schema_version": 1,
        "build_dir": str(build_dir),
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "repeats": repeats,
        "weight_layout": weight_layout,
        "input_sha256": {
            name: sha256_file(build_dir / name)
            for name in (
                "generated_machine_code.mem",
                "hbm_for_behave_sim.bin",
                "fp_sram.bin",
                "int_sram.bin",
                "plena_settings.toml",
            )
        },
        "variants": summary_rows,
        "gates": gates,
        "metrics": metrics,
        "claim_boundary": (
            f"This experiment uses compiler weight layout {weight_layout!r} and isolates DMA "
            "duplicate-burst coalescing from dependency-aware overlap. Cross-layout speedup must "
            "be computed against a separately generated row-major build with identical tensors. "
            "It does not model split lanes."
        ),
    }
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    table = [
        "| 实验状态 | 总周期 | HBM 读取字节 | 相对旧串行加速 |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        table.append(
            f"| {row['variant']} | {int(row['total_cycles']):,} | "
            f"{int(row['hbm_bytes_read']):,} | "
            f"{legacy['total_cycles'] / row['total_cycles']:.4f}x |"
        )
    report = f"""# MoE 权重布局、HBM 请求合并与异步重叠消融

输入构建目录：`{build_dir}`  
权重物理布局：`{weight_layout}`  
重复次数：{repeats}

{os.linesep.join(table)}

## 独立归因

- 重复 64B 请求合并：`{metrics['burst_coalescing_cycle_speedup']:.4f}x`，HBM 物理读取减少 `{metrics['burst_coalescing_physical_byte_reduction_pct']:.2f}%`。
- 在请求合并基础上启用真实 scoreboard overlap：额外 `{metrics['async_incremental_cycle_speedup']:.4f}x`。
- 两者合计相对旧串行路径：`{metrics['combined_cycle_speedup']:.4f}x`。

## 验证门

- 三次 cycle 与 HBM bytes 一致：PASS
- 每次运行与 PyTorch golden 比较：{'PASS' if gates['all_runs_match_pytorch_golden'] else 'FAIL'}
- 四种状态 VRAM 输出 SHA256 一致：{'PASS' if gates['functional_vram_hashes_identical'] else 'FAIL'}
- 强制串行 scoreboard 与普通串行 cycle 一致：{'PASS' if gates['scoreboard_serialized_matches_serial'] else 'FAIL'}

## 结论边界

本实验使用上面标明的单一 compiler 权重布局，隔离 DMA 重复 burst 合并与依赖感知的异步重叠。跨布局收益必须与使用相同 tensor 重新生成的 row-major build 比较；本表没有混入拆 lane 模型。
"""
    (output_dir / "ABLATION_REPORT.md").write_text(report, encoding="utf-8")

    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError("Validation gates failed: " + ", ".join(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emulator-binary", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temp-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.emulator_binary.resolve()
    build_dir = args.build_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if not binary.is_file():
        raise FileNotFoundError(binary)
    inputs = required_inputs(build_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    owns_temp = args.temp_root is None
    temp_root = (
        Path(tempfile.mkdtemp(prefix="plena-layout-async-"))
        if owns_temp
        else args.temp_root.resolve()
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        rows = []
        for variant in VARIANTS:
            for repeat in range(1, args.repeats + 1):
                print(f"Running {variant['id']} repeat {repeat}/{args.repeats}", flush=True)
                row = run_once(
                    binary=binary,
                    build_dir=build_dir,
                    inputs=inputs,
                    variant=variant,
                    repeat=repeat,
                    output_dir=output_dir,
                    temp_root=temp_root,
                )
                rows.append(row)
                print(
                    f"  cycles={row['total_cycles']:,} HBM-read={row['hbm_bytes_read']:,} "
                    f"host={row['host_wall_time_seconds']:.1f}s",
                    flush=True,
                )
        write_outputs(
            rows=rows,
            output_dir=output_dir,
            build_dir=build_dir,
            binary=binary,
            repeats=args.repeats,
        )
    finally:
        if owns_temp:
            shutil.rmtree(temp_root, ignore_errors=True)
    print(f"Wrote {output_dir / 'ABLATION_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
