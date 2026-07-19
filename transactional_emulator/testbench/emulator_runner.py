"""
Shared helper for running the Rust transactional emulator and comparing results.
Used by ATen-style testbench scripts for end-to-end numerical verification.
"""

from __future__ import annotations

import glob
import inspect
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path

import tomlkit

from verification.check_mem import compare_vram_with_golden, print_comparison_results
from transactional_emulator.testbench.config_utils import update_plena_config


def _build_emulator_binary(emulator_dir: Path, binary: Path) -> None:
    """Incrementally compile the release emulator binary.

    A fresh checkout / container has no `target/release/transactional_emulator`
    (the Rust `target/` dir isn't a persisted docker volume). Rather than failing
    with a dead-end error, build it here.  More importantly, always asking Cargo
    to check freshness prevents a pre-existing release executable from silently
    lagging behind CLI or timing-model source changes. `cargo build` is a fast
    no-op when the binary is already up to date.
    """
    if not binary.exists():
        print(
            f"Emulator binary not found at {binary}\n"
            "Building it now (subsequent runs use Cargo's incremental check)...",
            file=sys.stderr,
            flush=True,
        )
    result = subprocess.run(
        ["cargo", "build", "--release", "--bin", "transactional_emulator"],
        cwd=str(emulator_dir),
        env={**os.environ, "RUST_BACKTRACE": "1"},
    )
    if result.returncode != 0 or not binary.exists():
        raise FileNotFoundError(
            f"Failed to build the emulator binary (cargo exit {result.returncode}).\n"
            f"Build it manually with: cd {emulator_dir} && cargo build --release"
        )


def run_emulator(
    build_dir: Path,
    hbm_size: int | None = None,
    hbm_channels: int | None = None,
    threads: int | None = None,
    profile_memory: bool = False,
    profile_memory_level: str = "opcode",
    timing_mode: str = "rtl-v1",
    event_trace: Path | None = None,
    dma_event_trace: Path | None = None,
    no_state_dumps: bool = False,
    require_rtl_validated: bool = False,
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
        hbm_channels: optional override for modeled Ramulator HBM channel count.
                  With current HBM2_2Gbps and 64-bit/channel, 128 channels is a
                  2048 GB/s A100-bandwidth-equivalent proxy, not physical A100.
        timing_mode: ``legacy`` preserves serial timing; ``rtl-v1`` enables the
                  RTL-oriented opcode timing and hazard-aware logical scheduler.
        event_trace: optional path for the issue/start/completion JSON trace.
        dma_event_trace: optional compact path containing only HBM DMA timing
                  events, suitable for CostEmitter scheduler replay.
        no_state_dumps: skip post-run memory dumps for timing-only validation.
        require_rtl_validated: preserve outputs but fail after the run when an
                  opcode is unsupported by RTL or outside calibration domain.
    """
    emulator_dir = Path(__file__).parent.parent  # transactional_emulator/
    binary = emulator_dir / "target" / "release" / "transactional_emulator"

    # Do not use existence as a freshness proxy: an old release executable can
    # accept a different CLI or contain obsolete scheduling semantics.
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
    if hbm_channels is not None:
        cmd += ["--hbm-channels", str(hbm_channels)]
    if timing_mode not in {"legacy", "rtl-v1"}:
        raise ValueError(f"Unsupported timing mode: {timing_mode!r}")
    if require_rtl_validated and timing_mode != "rtl-v1":
        raise ValueError("require_rtl_validated requires timing_mode='rtl-v1'")
    cmd += ["--timing-mode", timing_mode]
    rtl_validation_path = build_dir / "rtl_validation_summary.json"
    if timing_mode == "rtl-v1":
        rtl_validation_path.unlink(missing_ok=True)
        cmd += ["--rtl-validation-output", str(rtl_validation_path)]
    if event_trace is not None:
        event_trace = Path(event_trace)
        event_trace.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--event-trace", str(event_trace)]
    if dma_event_trace is not None:
        dma_event_trace = Path(dma_event_trace)
        dma_event_trace.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--dma-event-trace", str(dma_event_trace)]
    if no_state_dumps:
        cmd += ["--no-state-dumps"]
    if require_rtl_validated:
        cmd += ["--require-rtl-validated"]

    # Per-build settings TOML: pass explicitly so the emulator reads the
    # correct config (not the global ../plena_settings.toml).
    settings_path = os.environ.get("PLENA_SETTINGS_TOML")
    if settings_path:
        cmd += ["--settings", settings_path]

    memory_profile_path = build_dir / "memory_profile.json"
    if profile_memory:
        cmd += [
            "--profile-memory",
            "--profile-memory-level",
            profile_memory_level,
            "--profile-output",
            str(memory_profile_path),
        ]

    # Optional VRAM preload: inject prestaged tensor data before execution.
    if vram_preload_path.exists():
        cmd += ["--vram", str(vram_preload_path)]

    # tch's download-libtorch stores libtorch in the Cargo build cache.
    # The binary needs LD_LIBRARY_PATH to find it at runtime.
    libtorch_pattern = str(
        emulator_dir / "target" / "release" / "build" / "torch-sys-*" / "out" / "libtorch" / "libtorch" / "lib"
    )
    libtorch_dirs = glob.glob(libtorch_pattern)
    env = {**os.environ, "RUST_BACKTRACE": "1", "RUST_LOG": "warn,transactional_emulator=info"}
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

    log_path = build_dir / "rust_emulator_stdout.log"
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    behavior_config = _current_behavior_config_summary()
    effective_hbm_channels = int(hbm_channels or behavior_config.get("HBM_CHANNELS", 8))
    hbm_channel_width_bits = 64
    hbm_data_rate_gbps = 2
    hbm_theoretical_peak_gbps = hbm_data_rate_gbps * hbm_channel_width_bits * effective_hbm_channels // 8
    metrics: dict[str, object] = {
        "schema_version": 3,
        "started_at_utc": started_at.isoformat(),
        "build_dir": str(build_dir),
        "command": cmd,
        "cwd": str(emulator_dir),
        "config_path": str(_current_plena_settings_path()),
        "behavior_config": behavior_config,
        "hbm_size_bytes": hbm_size,
        "hbm_channels": effective_hbm_channels,
        "hbm_timing": "HBM2_2Gbps",
        "hbm_channel_width_bits": hbm_channel_width_bits,
        "hbm_theoretical_peak_gbps": hbm_theoretical_peak_gbps,
        "hbm_model_note": "A100-bandwidth-equivalent when hbm_channels=128; not physical A100 topology",
        "timing_mode": timing_mode,
        "event_trace_path": str(event_trace) if event_trace is not None else None,
        "dma_event_trace_path": (
            str(dma_event_trace) if dma_event_trace is not None else None
        ),
        "rtl_validation_path": (
            str(rtl_validation_path) if timing_mode == "rtl-v1" else None
        ),
        "require_rtl_validated": require_rtl_validated,
        "artifacts": _artifact_summary(build_dir, asm_path, hbm_path),
        "log_path": str(log_path),
    }
    if profile_memory:
        metrics["memory_profile_path"] = str(memory_profile_path)
        metrics["memory_profile_level"] = profile_memory_level

    sim_latency_re = re.compile(r"Simulation completed\. Latency\s+([0-9.eE+-]+)ns")
    detailed_latency_re = re.compile(
        r'Simulation completed\s+timing_mode="([^"]+)"\s+'
        r"latency=([0-9.eE+-]+)ns\s+functional_executor_latency=([0-9.eE+-]+)ns"
    )
    topology_re = re.compile(r"mlen=(\d+)\s+vlen=(\d+)\s+.*blen=(\d+)")
    hbm_stats_re = re.compile(
        r"HBM Statistics - Bytes read:\s*([0-9]+)\s*\|\s*"
        r"Bytes written:\s*([0-9]+)\s*\|\s*"
        r"Utilization:\s*([0-9.eE+-]+)\s*bytes/sec"
    )

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(emulator_dir),
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

            detailed_match = detailed_latency_re.search(line)
            if detailed_match:
                modeled_latency_ns = float(detailed_match.group(2))
                functional_latency_ns = float(detailed_match.group(3))
                metrics["reported_timing_mode"] = detailed_match.group(1)
                metrics["modeled_makespan_latency_ns"] = modeled_latency_ns
                metrics["modeled_makespan_latency_ms"] = modeled_latency_ns / 1_000_000.0
                metrics["functional_executor_latency_ns"] = functional_latency_ns
                metrics["functional_executor_latency_ms"] = functional_latency_ns / 1_000_000.0

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

    # Keep the compact run stats self-describing. The full trace/profile remain
    # separate artifacts, while their calibration identity and timeline summary
    # are duplicated here so downstream sweeps can audit timing quality without
    # loading every per-trial event record.
    event_payload = _load_json_if_present(event_trace)
    if event_payload is not None:
        metrics["event_trace_schema_version"] = event_payload.get("schema_version")
        metrics["timing_calibration"] = event_payload.get("timing_calibration")
        metrics["rtl_validation"] = event_payload.get("rtl_validation")
        status_counts = Counter(
            event.get("calibration_status", "missing")
            for event in event_payload.get("events", [])
        )
        metrics["timing_calibration_status_counts"] = dict(sorted(status_counts.items()))

    validation_payload = _load_json_if_present(
        rtl_validation_path if timing_mode == "rtl-v1" else None
    )
    if validation_payload is not None:
        metrics["rtl_validation"] = validation_payload

    profile_payload = _load_json_if_present(memory_profile_path if profile_memory else None)
    if profile_payload is not None:
        timeline = profile_payload.get("timeline") or {}
        metrics["timeline_profile"] = timeline
        metrics["dma_transfers"] = profile_payload.get("dma_transfers")
        if metrics.get("rtl_validation") is None:
            metrics["rtl_validation"] = timeline.get("rtl_validation")
        if not metrics.get("timing_calibration_status_counts"):
            metrics["timing_calibration_status_counts"] = timeline.get(
                "timing_calibration_status_counts"
            )
        if metrics.get("timing_calibration") is None:
            metrics["timing_calibration"] = profile_payload.get("timing_calibration")

    stats_path = build_dir / "rust_emulator_run_stats.json"
    validation_requirement_failed = bool(require_rtl_validated and return_code == 2)
    metrics["rtl_validation_requirement_failed"] = validation_requirement_failed
    stats_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Rust emulator host wall time: {metrics['host_wall_time_seconds']:.3f}s (stats: {stats_path})")

    # Exit code 2 is the documented post-run validation-policy failure. The
    # emulator has completed and written valid state, so let the testbench copy
    # VRAM and run the unchanged numerical comparison before surfacing the
    # nonzero result. Any other nonzero code is a runtime failure.
    if return_code != 0 and not validation_requirement_failed:
        raise RuntimeError(f"Transactional emulator failed (exit code {return_code})")

    # Copy vram to build dir so subsequent runs don't overwrite it.
    vram_src = emulator_dir / "vram_dump.bin"
    vram_dst = build_dir / "vram_dump.bin"
    if vram_src.exists():
        import shutil

        shutil.copy2(vram_src, vram_dst)

    return metrics


def _load_json_if_present(path: Path | None) -> dict | None:
    """Read a JSON artifact when present, leaving emulator failures intact."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
        "HBM_CHANNELS",
    )
    summary = {}
    for key in keys:
        value = behavior_config.get(key, {})
        if isinstance(value, dict) and "value" in value:
            summary[key] = value["value"]
    return summary


def _validate_compile_runtime_transfer_contract(result: dict) -> None:
    """Reject DMA amounts that differ between compiler codegen and runtime.

    H_PREFETCH_V/H_STORE_V do not encode their transfer count in the ISA.  The
    compiler therefore uses the amount to generate loop bounds and address
    increments, while the emulator reads the amount from its TOML.  A mismatch
    causes overlapping transfers and can silently overwrite adjacent VRAM.
    Older compile results do not carry this metadata and remain supported.
    """
    if "PLENA_SETTINGS_TOML" not in os.environ:
        return
    info = result.get("info")
    if not isinstance(info, dict):
        return

    runtime = _current_behavior_config_summary()
    fields = (
        ("hbm_v_prefetch_amount", "HBM_V_Prefetch_Amount"),
        ("hbm_v_writeback_amount", "HBM_V_Writeback_Amount"),
    )
    mismatches = []
    for compile_key, runtime_key in fields:
        if compile_key not in info or runtime_key not in runtime:
            continue
        compiled = int(info[compile_key])
        configured = int(runtime[runtime_key])
        if compiled != configured:
            mismatches.append(
                f"{runtime_key}: compiler={compiled}, emulator={configured}"
            )
    if mismatches:
        raise ValueError(
            "Compiler/emulator DMA transfer contract mismatch; running this "
            "ISA would corrupt VRAM address ranges: " + "; ".join(mismatches)
        )


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
    compare_kwargs = {
        "exp_width": exp_width,
        "man_width": man_width,
        "num_bytes_per_val": max(1, (bits_per_val + 7) // 8),
        "row_dim": params.get("row_dim", 64),
        "start_row_idx": params["start_row_idx"],
        "num_batches": params["num_batches"],
        "num_rows": params["num_rows"],
        "elements_per_batch": params["elements_per_batch"],
        "atol": params.get("atol", 0.2),
        "rtol": params.get("rtol", 0.2),
        "use_stride_mode": params.get("use_stride_mode", True),
        "use_slice_mode": params.get("use_slice_mode", False),
        "slice_per_row": params.get("slice_per_row", None),
        "physical_rows": params.get("physical_rows", None),
        "rows_per_batch": params.get("rows_per_batch", None),
        "active_seq": params.get("active_seq_per_batch", None),
        "batch_pack_factor": params.get("batch_pack_factor", 1),
    }
    supported_kwargs = set(inspect.signature(compare_vram_with_golden).parameters)
    compare_kwargs = {k: v for k, v in compare_kwargs.items() if k in supported_kwargs}
    results = compare_vram_with_golden(vram_file, golden_file, **compare_kwargs)
    return results, params


def _write_comparison_summary(build_dir: Path, results: dict, params: dict) -> Path:
    """Persist scalar comparison diagnostics without serializing tensor payloads."""
    scalar_keys = (
        "mse",
        "mae",
        "max_error",
        "relative_error",
        "relative_match_rate",
        "allclose_match_rate",
        "match_rate",
        "allclose_pass",
        "atol",
        "rtol",
    )
    summary = {key: results[key] for key in scalar_keys if key in results}
    for key in ("golden_shape", "simulated_shape"):
        if key in results:
            summary[key] = list(results[key])
    summary["comparison_params"] = params
    path = build_dir / "comparison_results.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


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
    hbm_channels: int | None = None,
    profile_memory: bool = False,
    profile_memory_level: str = "opcode",
    timing_mode: str = "rtl-v1",
    event_trace: Path | None = None,
    dma_event_trace: Path | None = None,
    no_state_dumps: bool = False,
    require_rtl_validated: bool = False,
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
    run_metrics = run_emulator(
        build_dir,
        threads=threads,
        hbm_channels=hbm_channels,
        profile_memory=profile_memory,
        profile_memory_level=profile_memory_level,
        timing_mode=timing_mode,
        event_trace=event_trace,
        dma_event_trace=dma_event_trace,
        no_state_dumps=no_state_dumps,
        require_rtl_validated=require_rtl_validated,
    )

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
    comparison_path = _write_comparison_summary(build_dir, results, params)
    print_comparison_results(results, verbose=True, comparison_params=params)
    print(f"Comparison metrics: {comparison_path}")

    if results.get("test_pass", results.get("allclose_pass", False)):
        print(f"\n[ATen-style {op_name} test PASSED - ISA generated + emulator verified]")
    else:
        print(f"\n[ATen-style {op_name} test FAILED - emulator numerical check failed]")
        sys.exit(1)

    if run_metrics.get("rtl_validation_requirement_failed", False):
        print(
            "Numerical comparison completed, but --require-rtl-validated rejected "
            "unsupported or out-of-calibration-domain timing. See "
            f"{build_dir / 'rtl_validation_summary.json'}.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return run_metrics


def emulate_from_result(
    result: dict,
    build_dir: Path,
    asm_name: str,
    mlen: int = 64,
    blen: int = 4,
    vlen: int | None = None,
    threads: int | None = None,
    hbm_channels: int | None = None,
    profile_memory: bool = False,
    profile_memory_level: str = "opcode",
    timing_mode: str = "rtl-v1",
    event_trace: Path | None = None,
    dma_event_trace: Path | None = None,
    no_state_dumps: bool = False,
    require_rtl_validated: bool = False,
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
    _validate_compile_runtime_transfer_contract(result)

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
        hbm_channels=hbm_channels,
        profile_memory=profile_memory,
        profile_memory_level=profile_memory_level,
        timing_mode=timing_mode,
        event_trace=event_trace,
        dma_event_trace=dma_event_trace,
        no_state_dumps=no_state_dumps,
        require_rtl_validated=require_rtl_validated,
    )
