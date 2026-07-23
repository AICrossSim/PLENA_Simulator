"""
Shared helper for running the Rust transactional emulator and comparing results.
Used by ATen-style testbench scripts for end-to-end numerical verification.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

import tomlkit

from verification.check_mem import compare_vram_with_golden, print_comparison_results
from transactional_emulator.testbench.config_utils import update_plena_config


def _build_emulator_binary(emulator_dir: Path, binary: Path) -> None:
    """Compile the release emulator binary on demand.

    A fresh checkout / container has no `target/release/transactional_emulator`
    (the Rust `target/` dir isn't a persisted docker volume). Rather than failing
    with a dead-end error, build it here once — `cargo build` is a fast no-op when
    the binary is already up to date, so this stays out of the way on warm runs.
    """
    print(
        f"Ensuring emulator binary is current at {binary}\n"
        "Running cargo build --release (incremental/no-op when already current)...",
        file=sys.stderr,
        flush=True,
    )
    build_cmd = _release_build_command(emulator_dir)
    if build_cmd[0] == "nix":
        print(
            "Direct cargo is unavailable; using nix develop -c cargo build --release.",
            file=sys.stderr,
            flush=True,
        )
    result = subprocess.run(
        build_cmd,
        cwd=str(emulator_dir),
        env={**os.environ, "RUST_BACKTRACE": "1"},
    )
    if result.returncode != 0 or not binary.exists():
        raise FileNotFoundError(
            f"Failed to build the emulator binary (cargo exit {result.returncode}).\n"
            f"Build it manually with: cd {emulator_dir} && cargo build --release"
        )


def _release_build_command(emulator_dir: Path) -> list[str]:
    cargo_probe = subprocess.run(
        ["cargo", "--version"],
        cwd=str(emulator_dir),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if cargo_probe.returncode == 0:
        return ["cargo", "build", "--release"]
    if (emulator_dir.parent / "flake.nix").exists():
        return ["nix", "develop", "-c", "cargo", "build", "--release"]
    return ["cargo", "build", "--release"]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _run_file_suffix(run_label: str | None) -> str:
    if not run_label:
        return ""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_label.strip())
    if not safe:
        raise ValueError("run_label must contain at least one filename-safe character")
    return f".{safe}"


def run_emulator(
    build_dir: Path,
    hbm_size: int | None = None,
    threads: int | None = None,
    stage_profile: bool | None = None,
    stage_profile_out: Path | None = None,
    run_label: str | None = None,
    overlap_prefetch_compute: bool | None = None,
    dump_cwd: Path | None = None,
) -> dict:
    """Run the Rust transactional emulator with build artifacts from build_dir.

    Args:
        build_dir: directory containing generated_machine_code.mem, hbm_for_behave_sim.bin,
                   fp_sram.bin, int_sram.bin, and optionally vram_preload.bin.
        hbm_size: optional override for the emulator's HBM allocation, in bytes.
                  When set, passes --hbm-size to the emulator. Useful when
                  plena_settings.toml's TRANSACTIONAL.CONFIG.HBM_SIZE is provisioned
                  for large models (e.g. 128 GiB for LLaDA-8B) but the current
                  test only populates a small prefix — bounding HBM here keeps
                  steady-state RSS proportional to preload size rather than the
                  default capacity.
                  When None (default), it auto-sizes from `hbm_for_behave_sim.bin`'s
                  on-disk size, rounded up to the next 64-byte multiple. This
                  matches the actual preload — anything beyond is unused virtual
                  space that the emulator would otherwise lazy-commit pages into.
        stage_profile: when true, pass generated_asm_code.asm to the Rust
                       stage profiler and write stage_profile.json in build_dir.
                       When None, PLENA_EMULATOR_STAGE_PROFILE controls it.
        run_label: optional filename suffix for repeat/determinism runs. When
                   omitted, output filenames keep their historical names.
        overlap_prefetch_compute: when true, pass the off-by-default
                                   --experimental-overlap-prefetch-compute flag.
                                   When None, PLENA_EMULATOR_OVERLAP_PREFETCH_COMPUTE
                                   controls it.
        dump_cwd: optional working directory for emulator dump files. Defaults to
                  the historical emulator directory. Parallel replay can set this
                  to build_dir so vram_dump.bin/fpsram_dump.bin are not shared
                  between concurrent emulator processes.
    """
    emulator_dir = Path(__file__).parent.parent  # transactional_emulator/
    binary = emulator_dir / "target" / "release" / "transactional_emulator"
    dump_dir = Path(dump_cwd) if dump_cwd is not None else emulator_dir
    dump_dir.mkdir(parents=True, exist_ok=True)

    if stage_profile is None:
        stage_profile = _env_flag("PLENA_EMULATOR_STAGE_PROFILE")
    if overlap_prefetch_compute is None:
        overlap_prefetch_compute = _env_flag("PLENA_EMULATOR_OVERLAP_PREFETCH_COMPUTE")

    # Always rebuild before running. `cargo build --release` is a fast no-op when
    # current, and this prevents false failures from new ASM hitting a stale
    # release binary with old opcode decode logic.
    _build_emulator_binary(emulator_dir, binary)

    asm_path = build_dir / "generated_machine_code.mem"
    hbm_path = build_dir / "hbm_for_behave_sim.bin"
    fpsram_path = build_dir / "fp_sram.bin"
    intsram_path = build_dir / "int_sram.bin"
    vram_preload_path = build_dir / "vram_preload.bin"

    cmd = [
        str(binary),
        "--opcode",
        str(asm_path),
        "--hbm",
        str(hbm_path),
        "--fpsram",
        str(fpsram_path),
        "--intsram",
        str(intsram_path),
    ]
    # Deliberately NOT passing --log-level: it fully overrides RUST_LOG, and the
    # simulated-latency line ("Simulation completed. Latency ...ns" in main.rs) is logged
    # at INFO. We set RUST_LOG below to "warn,transactional_emulator=info" so that single
    # line is captured into sim_latency_ns without flooding the other modules (validated:
    # no measurable log/runtime blow-up vs plain --log-level warn).

    # HBM sizing: prefer the codegen-emitted sidecar (exact), fall back to
    # 2× preload heuristic, then TOML default (no flag = emulator reads TOML).
    hbm_size_file = build_dir / "hbm_size.txt"
    if hbm_size is None and hbm_size_file.exists():
        try:
            parsed = int(hbm_size_file.read_text().strip())
            if parsed > 0:
                hbm_size = parsed
        except (ValueError, OSError):
            pass  # fall through to heuristic
    if hbm_size is None and hbm_path.exists():
        # Heuristic fallback for builds that don't emit hbm_size.txt.
        preload_bytes = hbm_path.stat().st_size
        hbm_size = (((2 * preload_bytes) + 63) // 64) * 64
    if hbm_size is not None:
        cmd += ["--hbm-size", str(hbm_size)]

    # Per-build settings TOML: pass explicitly so the emulator reads the
    # correct config (not the global ../plena_settings.toml).
    settings_path = os.environ.get("PLENA_SETTINGS_TOML")
    if settings_path:
        cmd += ["--settings", settings_path]

    # Optional VRAM preload: inject prestaged tensor data before execution.
    if vram_preload_path.exists():
        cmd += ["--vram", str(vram_preload_path)]

    run_suffix = _run_file_suffix(run_label)
    if stage_profile_out is not None:
        profile_out_path = stage_profile_out
    elif run_suffix:
        profile_out_path = build_dir / f"stage_profile{run_suffix}.json"
    else:
        profile_out_path = build_dir / "stage_profile.json"
    if stage_profile:
        profile_asm_path = build_dir / "generated_asm_code.asm"
        if not profile_asm_path.exists():
            raise FileNotFoundError(f"stage_profile=True requires ASM comments at {profile_asm_path}")
        cmd += [
            "--stage-profile-asm",
            str(profile_asm_path),
            "--stage-profile-out",
            str(profile_out_path),
        ]
    if overlap_prefetch_compute:
        cmd.append("--experimental-overlap-prefetch-compute")

    # tch's download-libtorch stores libtorch in the Cargo build cache.
    # The binary needs LD_LIBRARY_PATH to find it at runtime.
    libtorch_pattern = str(
        emulator_dir / "target" / "release" / "build" / "torch-sys-*" / "out" / "libtorch" / "libtorch" / "lib"
    )
    libtorch_dirs = glob.glob(libtorch_pattern)
    env = {
        **os.environ,
        "RUST_BACKTRACE": "1",
        "RUST_LOG": os.environ.get("PLENA_EMULATOR_RUST_LOG", "warn,transactional_emulator=info"),
    }
    # libtorch (tch/ATen) parallelises every tensor op with an OpenMP pool that defaults to one
    # thread per core. On the emulator's tiny per-op tensors that is almost pure barrier overhead
    # (single-thread is ~6x faster here), and the spin-wait barriers melt down under
    # oversubscription when another libtorch job shares the box (e.g. a 32x32x4 sub-64 run went
    # from ~16s to 3.4h that way). PASSIVE makes idle threads sleep instead of spin (free safety);
    # `threads` caps the pool when set (run_model.py --threads, default 1).
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    if threads is not None:
        for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[_var] = str(threads)
    if libtorch_dirs:
        existing_ldpath = env.get("LD_LIBRARY_PATH", "")
        new_ldpath = libtorch_dirs[0]
        env["LD_LIBRARY_PATH"] = f"{new_ldpath}:{existing_ldpath}" if existing_ldpath else new_ldpath

    log_path = build_dir / f"rust_emulator_stdout{run_suffix}.log"
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    metrics: dict[str, object] = {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "build_dir": str(build_dir),
        "command": cmd,
        "cwd": str(dump_dir),
        "config_path": str(_current_plena_settings_path()),
        "behavior_config": _current_behavior_config_summary(),
        "hbm_size_bytes": hbm_size,
        "artifacts": _artifact_summary(build_dir, asm_path, hbm_path),
        "log_path": str(log_path),
        "stage_profile_requested": bool(stage_profile),
        "experimental_overlap_prefetch_compute": bool(overlap_prefetch_compute),
    }
    if run_label:
        metrics["run_label"] = run_label
    if stage_profile:
        metrics["stage_profile_path"] = str(profile_out_path)

    sim_latency_re = re.compile(r"Simulation completed\. Latency\s+([0-9.eE+-]+)ns(?:\s+cycles\s+([0-9]+))?")
    topology_re = re.compile(r"mlen=(\d+)\s+vlen=(\d+)\s+.*blen=(\d+)")
    hbm_stats_re = re.compile(
        r"HBM Statistics - Bytes read:\s*([0-9]+)\s*\|\s*"
        r"Bytes written:\s*([0-9]+)\s*\|\s*"
        r"Utilization:\s*([0-9.eE+-]+)\s*bytes/sec"
    )

    # Avoid copying an HBM dump from a previous debug run when the current run
    # does not enable DEBUG tracing.
    hbm_debug_dump = dump_dir / "hbm_dump.bin"
    if hbm_debug_dump.exists():
        hbm_debug_dump.unlink()

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(dump_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)

            sim_match = sim_latency_re.search(line)
            if sim_match:
                sim_latency_ns = float(sim_match.group(1))
                metrics["sim_latency_ns"] = sim_latency_ns
                metrics["sim_latency_ms"] = sim_latency_ns / 1_000_000.0
                if sim_match.group(2) is not None:
                    metrics["sim_latency_cycles"] = int(sim_match.group(2))

            topo_match = topology_re.search(line)
            if topo_match:
                metrics["emu_mlen"] = int(topo_match.group(1))
                metrics["emu_vlen"] = int(topo_match.group(2))
                metrics["emu_blen"] = int(topo_match.group(3))

            hbm_match = hbm_stats_re.search(line)
            if hbm_match:
                metrics["hbm_bytes_read"] = int(hbm_match.group(1))
                metrics["hbm_bytes_written"] = int(hbm_match.group(2))
                metrics["hbm_utilization_bytes_per_sec"] = float(hbm_match.group(3))

        return_code = proc.wait()

    ended_at = datetime.now(UTC)
    metrics["ended_at_utc"] = ended_at.isoformat()
    metrics["host_wall_time_seconds"] = time.perf_counter() - start
    metrics["return_code"] = return_code
    if stage_profile:
        metrics["stage_profile_exists"] = profile_out_path.exists()
        if profile_out_path.exists():
            metrics["stage_profile_size_bytes"] = profile_out_path.stat().st_size

    stats_path = build_dir / f"rust_emulator_run_stats{run_suffix}.json"
    metrics["stats_path"] = str(stats_path)
    stats_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Rust emulator host wall time: {metrics['host_wall_time_seconds']:.3f}s (stats: {stats_path})")

    if return_code != 0:
        raise RuntimeError(f"Transactional emulator failed (exit code {return_code})")

    # Copy emulator memory dumps to the build dir so subsequent runs don't
    # overwrite them. Source from dump_dir (may be a per-run cwd); skip the copy
    # when it already resolves to the build dir to avoid a self-copy.
    for dump_name in ("vram_dump.bin", "fpsram_dump.bin", "intsram_dump.bin", "hbm_dump.bin"):
        dump_src = dump_dir / dump_name
        dump_dst = build_dir / dump_name
        if dump_src.exists() and dump_src.resolve() != dump_dst.resolve():
            shutil.copy2(dump_src, dump_dst)

    return metrics


def run_emulator_repeat_gate(
    build_dir: Path,
    repeats: int = 3,
    hbm_size: int | None = None,
    threads: int | None = None,
    stage_profile: bool | None = None,
    overlap_prefetch_compute: bool | None = None,
) -> dict:
    """Run the same emulator artifact repeatedly and require identical cycles.

    This is intentionally opt-in: ordinary functional tests still call
    `run_emulator()` once, while timing baselines can call this helper to prove
    that a measurement is deterministic before accepting it.
    """
    if repeats < 2:
        raise ValueError("repeat gate requires at least two runs")

    build_dir = Path(build_dir)
    runs = []
    for index in range(repeats):
        label = f"repeat{index + 1:02d}"
        print(f"\n--- Repeat gate run {index + 1}/{repeats} ({label}) ---")
        runs.append(
            run_emulator(
                build_dir,
                hbm_size=hbm_size,
                threads=threads,
                stage_profile=stage_profile,
                run_label=label,
                overlap_prefetch_compute=overlap_prefetch_compute,
            )
        )

    required_keys = ("sim_latency_cycles",)
    optional_stable_keys = ("hbm_bytes_read", "hbm_bytes_written")
    series: dict[str, list[object]] = {
        key: [run.get(key) for run in runs] for key in required_keys + optional_stable_keys
    }

    missing_required = [key for key in required_keys if any(value is None for value in series[key])]
    if missing_required:
        raise RuntimeError("Repeat gate could not find required metrics: " + ", ".join(missing_required))

    stable_checks = {
        key: len(set(values)) == 1
        for key, values in series.items()
        if key in required_keys or all(value is not None for value in values)
    }
    passed = all(stable_checks.values())
    summary = {
        "schema_version": 1,
        "build_dir": str(build_dir),
        "repeats": repeats,
        "passed": passed,
        "stable_checks": stable_checks,
        "series": series,
        "run_stats_paths": [run.get("stats_path") for run in runs],
    }
    summary_path = build_dir / "rust_emulator_repeat_gate.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not passed:
        raise RuntimeError(f"Repeat gate failed; metric series written to {summary_path}: {series}")

    print(f"Repeat gate PASSED: {series} (summary: {summary_path})")
    return summary


def _current_plena_settings_path() -> Path:
    return Path(os.environ.get("PLENA_SETTINGS_TOML", Path(__file__).parents[2] / "plena_settings.toml"))


def _current_behavior_config_summary() -> dict[str, int | float | str]:
    config_path = _current_plena_settings_path()
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        config = tomlkit.load(f)
    behavior_config = config.get("TRANSACTIONAL", {}).get("CONFIG", {})
    keys = (
        "BLEN",
        "HLEN",
        "MLEN",
        "VLEN",
        "BROADCAST_AMOUNT",
        "HBM_M_Prefetch_Amount",
        "HBM_V_Prefetch_Amount",
        "HBM_V_Writeback_Amount",
        "MATRIX_SRAM_SIZE",
        "VECTOR_SRAM_SIZE",
        "HBM_SIZE",
    )
    summary = {}
    for key in keys:
        value = behavior_config.get(key, {})
        if isinstance(value, dict) and "value" in value:
            summary[key] = value["value"]
    return summary


def _artifact_summary(build_dir: Path, asm_path: Path, hbm_path: Path) -> dict[str, int]:
    summary = {}
    source_asm = build_dir / "generated_asm_code.asm"
    for key, path in (
        ("asm_source_bytes", source_asm),
        ("machine_code_bytes", asm_path),
        ("hbm_preload_bytes", hbm_path),
    ):
        if path.exists():
            summary[key] = path.stat().st_size
    if source_asm.exists():
        summary["asm_source_lines"] = sum(1 for _ in source_asm.open(encoding="utf-8", errors="replace"))
    if asm_path.exists():
        summary["machine_code_lines"] = sum(1 for _ in asm_path.open(encoding="utf-8", errors="replace"))
    return summary


def compare_emulator_output(build_dir: Path) -> tuple:
    """
    Compare emulator VRAM output against the golden reference.

    Returns:
        (results dict, params dict)
    """
    # Prefer build-dir copy (isolated), fall back to global emulator output.
    vram_file = build_dir / "vram_dump.bin"
    if not vram_file.exists():
        emulator_dir = Path(__file__).parent.parent  # transactional_emulator/
        vram_file = emulator_dir / "vram_dump.bin"
    golden_file = build_dir / "golden_result.txt"
    params_file = build_dir / "comparison_params.json"

    with open(params_file) as f:
        params = json.load(f)

    exp_width, man_width, bits_per_val = _current_vector_sram_fp_format()
    results = compare_vram_with_golden(
        vram_file,
        golden_file,
        exp_width=exp_width,
        man_width=man_width,
        num_bytes_per_val=max(1, (bits_per_val + 7) // 8),
        row_dim=params.get("row_dim", 64),
        start_row_idx=params["start_row_idx"],
        num_batches=params["num_batches"],
        num_rows=params["num_rows"],
        elements_per_batch=params["elements_per_batch"],
        atol=params.get("atol", 0.2),
        rtol=params.get("rtol", 0.2),
        use_stride_mode=params.get("use_stride_mode", True),
        use_slice_mode=params.get("use_slice_mode", False),
        slice_per_row=params.get("slice_per_row", None),
        physical_rows=params.get("physical_rows", None),
        rows_per_batch=params.get("rows_per_batch", None),
        active_seq=params.get("active_seq_per_batch", None),
    )
    return results, params


def _current_vector_sram_fp_format() -> tuple[int, int, int]:
    """Return VECTOR_SRAM_TYPE as (exp, mant, total_bits) from the active TOML."""
    config_path = _current_plena_settings_path()
    with open(config_path) as f:
        config = tomlkit.load(f)
    data_type = config["TRANSACTIONAL"]["PRECISION"]["VECTOR_SRAM_TYPE"]["DATA_TYPE"]
    exp_width = int(data_type["exponent"])
    man_width = int(data_type["mantissa"])
    sign_width = 1 if bool(data_type.get("sign", True)) else 0
    return exp_width, man_width, sign_width + exp_width + man_width


def run_and_assert(
    build_dir: Path,
    op_name: str,
    mlen: int = 64,
    blen: int = 4,
    vlen: int | None = None,
    threads: int | None = None,
    stage_profile: bool | None = None,
) -> dict:
    """
    Sync HW config, run the Rust emulator, compare output, exit(1) on failure.

    Args:
        build_dir: Path to the build directory with sim env files.
        op_name:   Operator name used in pass/fail messages.
        mlen:      Matrix tile length — synced to plena_settings.toml before running.
        blen:      Batch tile length — synced to plena_settings.toml before running.
        vlen:      Vector tile length — defaults to mlen if not specified.
    """
    if vlen is None:
        vlen = mlen
    if "PLENA_SETTINGS_TOML" not in os.environ:
        update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)

    print("\n--- Running Rust transactional emulator ---")
    run_metrics = run_emulator(build_dir, threads=threads, stage_profile=stage_profile)

    emu_mlen = run_metrics.get("emu_mlen")
    emu_blen = run_metrics.get("emu_blen")
    if emu_mlen is not None and emu_mlen != mlen:
        raise RuntimeError(
            f"Config mismatch: emulator ran at MLEN={emu_mlen} but test compiled for MLEN={mlen}. "
            f"Check PLENA_SETTINGS_TOML points to the per-build TOML."
        )
    if emu_blen is not None and emu_blen != blen:
        raise RuntimeError(
            f"Config mismatch: emulator ran at BLEN={emu_blen} but test compiled for BLEN={blen}. "
            f"Check PLENA_SETTINGS_TOML points to the per-build TOML."
        )

    print("\n--- Comparing emulator output vs golden ---")
    results, params = compare_emulator_output(build_dir)
    print_comparison_results(results, verbose=True, comparison_params=params)

    if results.get("test_pass", results.get("allclose_pass", False)):
        print(f"\n[ATen-style {op_name} test PASSED - ISA generated + emulator verified]")
    else:
        print(f"\n[ATen-style {op_name} test FAILED - emulator numerical check failed]")
        sys.exit(1)

    return run_metrics


def emulate_from_result(
    result: dict,
    build_dir: Path,
    asm_name: str,
    mlen: int = 64,
    blen: int = 4,
    vlen: int | None = None,
    threads: int | None = None,
    stage_profile: bool | None = None,
) -> dict:
    """Write sim artifacts from a compile result dict and run the Rust emulator.

    The result dict must contain: isa, input_tensors, golden_result,
    fp_preload, data_order, comparison_params. Optional: tensor_layouts,
    hbm_addrs.
    """
    from transactional_emulator.tools.create_sim_env import create_sim_env
    from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim

    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    create_sim_env(
        result.get("input_tensors", {}),
        result["isa"],
        result.get("golden_result", {"original_output": result.get("golden_output")}),
        result["fp_preload"],
        build_dir=str(build_dir),
        tensor_layouts=result.get("tensor_layouts"),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=asm_name,
        data=None,
        specified_data_order=result["data_order"],
        build_path=build_dir,
        input_tensors=result.get("input_tensors"),
        tensor_layouts=result.get("tensor_layouts"),
        hbm_addrs=result.get("hbm_addrs"),
    )

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(result["comparison_params"], f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(result["isa"])

    return run_and_assert(
        build_dir,
        asm_name,
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        threads=threads,
        stage_profile=stage_profile,
    )
