"""
Shared helper for running the Rust transactional emulator and comparing results.
Used by ATen-style testbench scripts for end-to-end numerical verification.
"""

from __future__ import annotations

import atexit
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, UTC
from pathlib import Path

import tomlkit

from runtime_paths import settings_path, simulator_root
from verification.check_mem import compare_vram_with_golden, print_comparison_results
from transactional_emulator.testbench.config_utils import update_plena_config


_BUILD_DIRECTORY_LOCKS: dict[Path, dict[str, object]] = {}
RUN_RECEIPT_SCHEMA = 2


def _close_inherited_leases() -> None:
    for entry in _BUILD_DIRECTORY_LOCKS.values():
        handle = entry.get("handle")
        if handle is not None:
            handle.close()
    _BUILD_DIRECTORY_LOCKS.clear()


os.register_at_fork(after_in_child=_close_inherited_leases)


class BuildDirectoryLease:
    """Process-local reference to an exclusive build-directory lock."""

    def __init__(self, build_dir: Path, owner: tuple[int, int]) -> None:
        self.build_dir = build_dir
        self.owner = owner
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        entry = _BUILD_DIRECTORY_LOCKS.get(self.build_dir)
        if entry is None or entry["owner"] != self.owner:
            return
        entry["references"] = int(entry["references"]) - 1
        if entry["references"]:
            return
        handle = entry["handle"]
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        del _BUILD_DIRECTORY_LOCKS[self.build_dir]


def acquire_build_directory(build_dir: Path) -> BuildDirectoryLease:
    """Acquire a non-blocking, same-thread-reentrant lease for ``build_dir``.

    Generation, emulation, dump copying, and validation all use mutable files in
    one directory. Serialising only the emulator process is insufficient because
    a second generator can replace those inputs while the first run is active.
    """
    resolved = Path(build_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    owner = (os.getpid(), threading.get_ident())
    entry = _BUILD_DIRECTORY_LOCKS.get(resolved)
    if entry is not None:
        if entry["owner"] == owner:
            entry["references"] = int(entry["references"]) + 1
            lease = BuildDirectoryLease(resolved, owner)
            atexit.register(lease.release)
            return lease
        raise RuntimeError(
            f"Build directory {resolved} is already active in another thread. "
            "Concurrent or interleaved runs would invalidate its artifacts."
        )

    lock_path = resolved / ".transactional_emulator.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "owner metadata unavailable"
        handle.close()
        raise RuntimeError(
            f"Build directory {resolved} is already active ({owner}). "
            "Concurrent or interleaved runs would invalidate its artifacts; "
            "wait for that run to finish or use a different build directory."
        ) from error

    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "thread": threading.get_ident(),
                "started_at_utc": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    _BUILD_DIRECTORY_LOCKS[resolved] = {
        "handle": handle,
        "owner": owner,
        "references": 1,
    }
    lease = BuildDirectoryLease(resolved, owner)
    atexit.register(lease.release)
    return lease


def assert_build_directory_idle(build_dir: Path) -> None:
    """Fail if another process is producing artifacts in ``build_dir``."""
    resolved = Path(build_dir).resolve()
    if resolved in _BUILD_DIRECTORY_LOCKS:
        raise RuntimeError(f"Build directory {resolved} is already active")
    lease = acquire_build_directory(build_dir)
    lease.release()


def _emulator_root() -> Path:
    return simulator_root() / "transactional_emulator"


def _emulator_execution_directory() -> Path:
    """Shared lock directory for emulator outputs written outside a build."""
    checkout_id = hashlib.sha256(str(_emulator_root()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "plena_emulator_locks" / checkout_id


def acquire_emulator_execution() -> BuildDirectoryLease:
    """Acquire the checkout-wide lease for global dumps and live settings."""
    return acquire_build_directory(_emulator_execution_directory())


def _build_emulator_binary(emulator_dir: Path, binary: Path) -> None:
    """Compile the release emulator binary on demand.

    A fresh checkout / container has no `target/release/transactional_emulator`
    (the Rust `target/` dir isn't a persisted docker volume). Rather than failing
    with a dead-end error, build it here once — `cargo build` is a fast no-op when
    the binary is already up to date, so this stays out of the way on warm runs.
    """
    print(
        f"Emulator binary not found at {binary}\n"
        "Building it now (one-time release compile; subsequent runs reuse it)...",
        file=sys.stderr,
        flush=True,
    )
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=str(emulator_dir),
        env={**os.environ, "RUST_BACKTRACE": "1"},
    )
    if result.returncode != 0 or not binary.exists():
        raise FileNotFoundError(
            f"Failed to build the emulator binary (cargo exit {result.returncode}).\n"
            f"Build it manually with: cd {emulator_dir} && cargo build --release"
        )


def _decode_hbm_read_ledger(build_dir: Path, metrics: dict[str, object]) -> dict:
    """Attribute physical decoder reads to the prefetch instructions that issue them."""
    source_path = build_dir / "generated_asm_code.asm"
    machine_path = build_dir / "generated_machine_code.mem"
    op_stats_path = build_dir / "op_stats.jsonl"
    marker = re.compile(r"^; (?:Pipelined|Packed) ([KV]) prefetch for ")
    role_names = {"K": "key", "V": "value"}
    prefetch_pcs: dict[str, set[int]] = {"key": set(), "value": set()}
    pending_role: str | None = None
    instruction_pc = -1

    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(";"):
            match = marker.match(line)
            if match:
                if pending_role is not None:
                    raise RuntimeError("decoder prefetch tag has no instruction")
                pending_role = role_names[match.group(1)]
            continue
        instruction_pc += 1
        if pending_role is not None:
            opcode = line.split(maxsplit=1)[0]
            if opcode == "H_PREFETCH_M":
                prefetch_pcs[pending_role].add(instruction_pc)
                pending_role = None
            elif opcode not in {"S_ADDI_INT", "S_ADD_INT"}:
                raise RuntimeError(
                    "decoder prefetch tag is separated from its issue by "
                    f"non-address-setup opcode {opcode}"
                )

    if pending_role is not None or not all(prefetch_pcs.values()):
        raise RuntimeError("decoder assembly lacks attributable K/V prefetches")
    source_instruction_count = instruction_pc + 1
    machine_instruction_count = sum(
        1
        for line in machine_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if source_instruction_count != machine_instruction_count:
        raise RuntimeError(
            "decoder source-to-machine PC mapping is not one instruction per line"
        )

    pc_roles = {pc: role for role, pcs in prefetch_pcs.items() for pc in pcs}
    if len(pc_roles) != sum(len(pcs) for pcs in prefetch_pcs.values()):
        raise RuntimeError("decoder K/V prefetch PC sets overlap")

    attributed_bytes = {"key": 0, "value": 0}
    dynamic_prefetches = {"key": 0, "value": 0}
    executions_by_pc = {pc: 0 for pc in pc_roles}
    issue_bytes_by_pc = {pc: 0 for pc in pc_roles}
    service_row_total = 0
    issue_row_total = 0
    aggregate_service_total: int | None = None
    aggregate_issue_total: int | None = None
    with op_stats_path.open(encoding="utf-8") as source:
        for raw_line in source:
            record = json.loads(raw_line)
            if record.get("aggregate") is True:
                if aggregate_service_total is not None:
                    raise RuntimeError("decoder op-stats has multiple aggregate records")
                if "total_hbm_issue_rd" not in record:
                    raise RuntimeError(
                        "decoder op-stats lacks issue-origin HBM attribution"
                    )
                aggregate_service_total = int(record["total_hbm_rd"])
                aggregate_issue_total = int(record["total_hbm_issue_rd"])
                continue
            if "pc" not in record:
                continue
            if "hbm_issue_rd" not in record:
                raise RuntimeError(
                    "decoder op-stats row lacks issue-origin HBM attribution"
                )
            hbm_read = int(record.get("hbm_rd", 0))
            hbm_issue_read = int(record["hbm_issue_rd"])
            if hbm_read < 0 or hbm_issue_read < 0:
                raise RuntimeError("decoder op-stats contains negative HBM traffic")
            service_row_total += hbm_read
            issue_row_total += hbm_issue_read
            role = pc_roles.get(int(record["pc"]))
            if role is not None:
                if record.get("op") != "H_PREFETCH_M":
                    raise RuntimeError("decoder prefetch PC executed as a different opcode")
                attributed_bytes[role] += hbm_issue_read
                dynamic_prefetches[role] += 1
                executions_by_pc[int(record["pc"])] += 1
                issue_bytes_by_pc[int(record["pc"])] += hbm_issue_read

    measured_total = metrics.get("hbm_bytes_read")
    if not isinstance(measured_total, int):
        raise RuntimeError("decoder emulator receipt has no physical HBM read total")
    if aggregate_service_total is None or not (
        service_row_total == aggregate_service_total == measured_total
    ):
        raise RuntimeError(
            "decoder HBM service totals disagree between rows, aggregate, and receipt"
        )
    if aggregate_issue_total is None or not (
        issue_row_total == aggregate_issue_total == measured_total
    ):
        raise RuntimeError(
            "decoder issue-origin HBM totals disagree between rows, aggregate, and receipt"
        )
    non_attention_bytes = (
        measured_total - attributed_bytes["key"] - attributed_bytes["value"]
    )
    inactive_pcs = sorted(
        pc
        for pc in pc_roles
        if executions_by_pc[pc] == 0 or issue_bytes_by_pc[pc] == 0
    )
    if inactive_pcs:
        raise RuntimeError(
            "decoder tagged K/V prefetch PCs lack positive dynamic issue traffic: "
            + ", ".join(str(pc) for pc in inactive_pcs)
        )
    if non_attention_bytes < 0 or not all(dynamic_prefetches.values()):
        raise RuntimeError("decoder K/V attribution does not reconcile with global traffic")

    artifacts = metrics.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("decoder emulator receipt has no artifact inventory")
    return {
        "schema_version": 2,
        "provenance": "emulator issue-origin physical bytes by source-tagged instruction PC",
        "key_bytes": attributed_bytes["key"],
        "value_bytes": attributed_bytes["value"],
        "non_attention_bytes": non_attention_bytes,
        "global_bytes": measured_total,
        "issue_origin_bytes": issue_row_total,
        "key_prefetches": dynamic_prefetches["key"],
        "value_prefetches": dynamic_prefetches["value"],
        "key_prefetch_pcs": sorted(prefetch_pcs["key"]),
        "value_prefetch_pcs": sorted(prefetch_pcs["value"]),
        "source_instruction_count": source_instruction_count,
        "asm_source_sha256": artifacts.get("asm_source_sha256"),
        "config_sha256": metrics.get("config_sha256"),
        "op_stats_sha256": metrics.get("op_stats_sha256"),
        "run_manifest_sha256": metrics.get("run_manifest_sha256"),
    }


def _run_emulator_unlocked(
    build_dir: Path, hbm_size: int | None = None, threads: int | None = None
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
    """
    emulator_dir = _emulator_root()
    binary = emulator_dir / "target" / "release" / "transactional_emulator"

    if not binary.exists():
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

    # Extra emulator flags injected by calibration drivers, e.g.
    # PLENA_EMU_EXTRA_ARGS="--blocking-prefetch --op-stats /path/op_stats.jsonl".
    extra_args = os.environ.get("PLENA_EMU_EXTRA_ARGS")
    if extra_args:
        cmd += extra_args.split()

    # Per-op statistics, unless the caller already routed them somewhere else.
    # The emulator only writes this file when asked, so without it a build
    # directory keeps whatever a previous run left behind: the stage-validation
    # comparison would then read one schedule's op counts against another
    # schedule's assembly and report agreement that was never measured.
    if "--op-stats" not in cmd:
        cmd += ["--op-stats", str(build_dir / "op_stats.jsonl")]

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
    stats_path = build_dir / "rust_emulator_run_stats.json"
    vram_src = emulator_dir / "vram_dump.bin"
    stats_path.unlink(missing_ok=True)
    vram_src.unlink(missing_ok=True)
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    config_path = _current_plena_settings_path()
    metrics: dict[str, object] = {
        "schema_version": RUN_RECEIPT_SCHEMA,
        "started_at_utc": started_at.isoformat(),
        "build_dir": str(build_dir),
        "command": cmd,
        "cwd": str(emulator_dir),
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "emulator_binary_sha256": _sha256_file(binary),
        "behavior_config": _current_behavior_config_summary(cmd),
        "hbm_size_bytes": hbm_size,
        "artifacts": _artifact_summary(build_dir, asm_path, hbm_path),
        "log_path": str(log_path),
        "numerical_validation_passed": False,
    }

    sim_latency_re = re.compile(r"Simulation completed\. Latency\s+([0-9.eE+-]+)ns")
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

    if return_code != 0:
        _atomic_write_json(stats_path, metrics)
        raise RuntimeError(f"Transactional emulator failed (exit code {return_code})")

    # Copy vram to build dir so subsequent runs don't overwrite it.
    vram_dst = build_dir / "vram_dump.bin"
    if not vram_src.is_file():
        metrics["artifact_complete"] = False
        _atomic_write_json(stats_path, metrics)
        raise RuntimeError("emulator completed without producing vram_dump.bin")
    import shutil

    shutil.copy2(vram_src, vram_dst)

    op_stats_value = cmd[cmd.index("--op-stats") + 1]
    op_stats_path = Path(op_stats_value)
    if not op_stats_path.is_absolute():
        op_stats_path = (emulator_dir / op_stats_path).resolve()
    metrics["op_stats_path"] = str(op_stats_path)
    metrics["op_stats_sha256"] = _sha256_file(op_stats_path)
    metrics["vram_dump_sha256"] = _sha256_file(vram_dst)
    for key, filename in (
        ("run_manifest_sha256", "decode_run_manifest.json"),
        ("comparison_params_sha256", "comparison_params.json"),
        ("golden_result_sha256", "golden_result.txt"),
    ):
        artifact = build_dir / filename
        if artifact.is_file():
            metrics[key] = _sha256_file(artifact)
    if (build_dir / "decode_run_manifest.json").is_file():
        try:
            metrics["hbm_read_ledger"] = _decode_hbm_read_ledger(build_dir, metrics)
        except Exception as error:
            metrics["artifact_complete"] = False
            metrics["traffic_validation_error"] = str(error)
            _atomic_write_json(stats_path, metrics)
            raise
    metrics["artifact_complete"] = True

    _atomic_write_json(stats_path, metrics)
    print(f"Rust emulator host wall time: {metrics['host_wall_time_seconds']:.3f}s (stats: {stats_path})")

    return metrics


def run_emulator(build_dir: Path, hbm_size: int | None = None, threads: int | None = None) -> dict:
    """Run while owning both the build artifacts and global emulator outputs."""
    resolved = Path(build_dir).resolve()
    build_lease = acquire_build_directory(resolved)
    try:
        execution_lease = acquire_build_directory(_emulator_execution_directory())
        try:
            return _run_emulator_unlocked(
                resolved,
                hbm_size=hbm_size,
                threads=threads,
            )
        finally:
            execution_lease.release()
    finally:
        build_lease.release()


def _current_plena_settings_path() -> Path:
    return settings_path()


def _plain_toml_value(value: object) -> object:
    if hasattr(value, "unwrap"):
        value = value.unwrap()
    if isinstance(value, dict):
        return {str(key): _plain_toml_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_toml_value(item) for item in value]
    return value


def _behavior_config_summary(
    config_path: Path,
    command: list[str] | None = None,
) -> dict[str, object]:
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
        "FP_SRAM_DEPTH",
        "HBM_GEN",
        "HBM_CHANNELS",
    )
    summary: dict[str, object] = {}
    for key in keys:
        value = behavior_config.get(key, {})
        if isinstance(value, dict) and "value" in value:
            summary[key] = _plain_toml_value(value["value"])
    drain = behavior_config.get("DRAIN_OVERLAPPED", {"value": 0})
    summary["DRAIN_OVERLAPPED"] = bool(int(drain.get("value", 0)))

    precision = config.get("TRANSACTIONAL", {}).get("PRECISION", {})
    precision_keys = (
        "MATRIX_SRAM_TYPE",
        "VECTOR_SRAM_TYPE",
        "HBM_M_WEIGHT_TYPE",
        "HBM_M_KV_TYPE",
        "HBM_V_ACT_TYPE",
        "HBM_V_KV_TYPE",
    )
    summary["PRECISION"] = {
        key: _plain_toml_value(precision[key])
        for key in precision_keys
        if key in precision
    }
    if command:
        for option, key, conversion in (
            ("--hbm-gen", "HBM_GEN", str),
            ("--hbm-channels", "HBM_CHANNELS", int),
        ):
            if option in command:
                summary[key] = conversion(command[command.index(option) + 1])
    return summary


def _current_behavior_config_summary(
    command: list[str] | None = None,
) -> dict[str, object]:
    return _behavior_config_summary(_current_plena_settings_path(), command)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _mark_numerical_validation(build_dir: Path) -> dict:
    receipt_path = Path(build_dir).resolve() / "rust_emulator_run_stats.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != RUN_RECEIPT_SCHEMA
        or receipt.get("return_code") != 0
        or receipt.get("artifact_complete") is not True
    ):
        raise RuntimeError("cannot validate an incomplete emulator receipt")
    receipt["numerical_validation_passed"] = True
    _atomic_write_json(receipt_path, receipt)
    return receipt


def _artifact_summary(build_dir: Path, asm_path: Path, hbm_path: Path) -> dict[str, int | str]:
    summary = {}
    source_asm = build_dir / "generated_asm_code.asm"
    for key, path in (
        ("asm_source_bytes", source_asm),
        ("machine_code_bytes", asm_path),
        ("hbm_preload_bytes", hbm_path),
        ("fp_sram_preload_bytes", build_dir / "fp_sram.bin"),
        ("int_sram_preload_bytes", build_dir / "int_sram.bin"),
        ("vram_preload_bytes", build_dir / "vram_preload.bin"),
        ("compiler_artifact_bytes", build_dir / "compilation_artifact.json"),
    ):
        if path.exists():
            summary[key] = path.stat().st_size
            summary[key.removesuffix("_bytes") + "_sha256"] = _sha256_file(path)
    if source_asm.exists():
        summary["asm_source_lines"] = sum(1 for _ in source_asm.open(encoding="utf-8", errors="replace"))
    if asm_path.exists():
        summary["machine_code_lines"] = sum(1 for _ in asm_path.open(encoding="utf-8", errors="replace"))
    return summary


def validate_emulator_run_receipt(
    build_dir: Path,
    *,
    settings_file: Path | None = None,
) -> dict:
    """Verify that a successful run receipt binds every consumed artifact."""
    resolved = Path(build_dir).resolve()
    receipt_path = resolved / "rust_emulator_run_stats.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"emulator run receipt is missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"emulator run receipt is unreadable: {receipt_path}") from error
    if not isinstance(receipt, dict) or receipt.get("schema_version") != RUN_RECEIPT_SCHEMA:
        raise RuntimeError("emulator run receipt schema is unsupported")
    if receipt.get("return_code") != 0 or receipt.get("artifact_complete") is not True:
        raise RuntimeError("emulator run receipt does not describe a successful complete run")
    if receipt.get("numerical_validation_passed") is not True:
        raise RuntimeError("emulator run receipt has no successful numerical validation")
    if Path(str(receipt.get("build_dir", ""))).resolve() != resolved:
        raise RuntimeError("emulator run receipt names a different build directory")

    recorded_config = Path(str(receipt.get("config_path", ""))).resolve()
    expected_config = (
        Path(settings_file).resolve() if settings_file is not None else recorded_config
    )
    if recorded_config != expected_config or not expected_config.is_file():
        raise RuntimeError("emulator run receipt names a different settings file")

    def require_hash(path: Path, field: str) -> None:
        expected = receipt.get(field)
        if not path.is_file() or not isinstance(expected, str) or _sha256_file(path) != expected:
            raise RuntimeError(f"emulator run receipt hash mismatch for {path}")

    require_hash(expected_config, "config_sha256")
    command = receipt.get("command")
    if not isinstance(command, list) or not all(
        isinstance(argument, str) for argument in command
    ):
        raise RuntimeError("emulator run receipt has no valid command")
    emulator_binary_sha256 = receipt.get("emulator_binary_sha256")
    if (
        not isinstance(emulator_binary_sha256, str)
        or len(emulator_binary_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in emulator_binary_sha256
        )
    ):
        raise RuntimeError("emulator run receipt has no valid emulator binary digest")
    emulator_binary = Path(command[0])
    if not emulator_binary.is_absolute():
        emulator_binary = Path(str(receipt.get("cwd", resolved))) / emulator_binary
    if (
        not emulator_binary.is_file()
        or _sha256_file(emulator_binary) != emulator_binary_sha256
    ):
        raise RuntimeError("emulator run receipt binary digest mismatch")
    if receipt.get("behavior_config") != _behavior_config_summary(
        expected_config, command
    ):
        raise RuntimeError("emulator run receipt behavior configuration mismatch")
    require_hash(resolved / "op_stats.jsonl", "op_stats_sha256")
    require_hash(resolved / "vram_dump.bin", "vram_dump_sha256")
    if Path(str(receipt.get("op_stats_path", ""))).resolve() != (
        resolved / "op_stats.jsonl"
    ):
        raise RuntimeError("emulator run receipt names a different op-stats file")

    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError("emulator run receipt is missing its input inventory")
    for filename, field in (
        ("generated_asm_code.asm", "asm_source_sha256"),
        ("generated_machine_code.mem", "machine_code_sha256"),
        ("hbm_for_behave_sim.bin", "hbm_preload_sha256"),
        ("fp_sram.bin", "fp_sram_preload_sha256"),
        ("int_sram.bin", "int_sram_preload_sha256"),
    ):
        path = resolved / filename
        expected = artifacts.get(field)
        if not isinstance(expected, str) or _sha256_file(path) != expected:
            raise RuntimeError(f"emulator run receipt hash mismatch for {path}")
    optional_preload = resolved / "vram_preload.bin"
    if optional_preload.exists() or "vram_preload_sha256" in artifacts:
        expected = artifacts.get("vram_preload_sha256")
        if (
            not optional_preload.is_file()
            or not isinstance(expected, str)
            or _sha256_file(optional_preload) != expected
        ):
            raise RuntimeError(
                f"emulator run receipt hash mismatch for {optional_preload}"
            )
    compiler_artifact = resolved / "compilation_artifact.json"
    if compiler_artifact.exists() or "compiler_artifact_sha256" in artifacts:
        expected = artifacts.get("compiler_artifact_sha256")
        if (
            not compiler_artifact.is_file()
            or not isinstance(expected, str)
            or _sha256_file(compiler_artifact) != expected
        ):
            raise RuntimeError(
                f"emulator run receipt hash mismatch for {compiler_artifact}"
            )
    for filename, field in (
        ("decode_run_manifest.json", "run_manifest_sha256"),
        ("comparison_params.json", "comparison_params_sha256"),
        ("golden_result.txt", "golden_result_sha256"),
    ):
        path = resolved / filename
        if path.exists() or field in receipt:
            require_hash(path, field)

    manifest_path = resolved / "decode_run_manifest.json"
    manifest = None
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("decoder run manifest is unreadable") from error

    ledger = receipt.get("hbm_read_ledger")
    if (
        isinstance(manifest, dict)
        and manifest.get("hbm_read_ledger_required") is True
        and ledger is None
    ):
        raise RuntimeError("emulator run receipt is missing its HBM read ledger")
    if ledger is not None:
        if not isinstance(ledger, dict) or ledger.get("schema_version") != 2:
            raise RuntimeError("emulator run receipt has an invalid HBM read ledger")
        byte_fields = (
            "key_bytes",
            "value_bytes",
            "non_attention_bytes",
            "global_bytes",
            "issue_origin_bytes",
        )
        if any(
            not isinstance(ledger.get(field), int) or ledger[field] < 0
            for field in byte_fields
        ):
            raise RuntimeError("emulator HBM read ledger has invalid byte counts")
        if (
            ledger["key_bytes"]
            + ledger["value_bytes"]
            + ledger["non_attention_bytes"]
            != ledger["global_bytes"]
            or ledger["global_bytes"] != receipt.get("hbm_bytes_read")
            or ledger["issue_origin_bytes"] != ledger["global_bytes"]
        ):
            raise RuntimeError("emulator HBM read ledger does not reconcile")
        if ledger.get("provenance") != (
            "emulator issue-origin physical bytes by source-tagged instruction PC"
        ):
            raise RuntimeError("emulator HBM read ledger lacks issue-origin provenance")
        for ledger_field, receipt_value in (
            ("asm_source_sha256", artifacts.get("asm_source_sha256")),
            ("config_sha256", receipt.get("config_sha256")),
            ("op_stats_sha256", receipt.get("op_stats_sha256")),
            ("run_manifest_sha256", receipt.get("run_manifest_sha256")),
        ):
            if ledger.get(ledger_field) != receipt_value:
                raise RuntimeError(
                    f"emulator HBM read ledger hash mismatch for {ledger_field}"
                )
        recomputed_ledger = _decode_hbm_read_ledger(resolved, receipt)
        if ledger != recomputed_ledger:
            raise RuntimeError(
                "emulator HBM read ledger differs from hash-bound issue-origin evidence"
            )

    latency = receipt.get("sim_latency_ns")
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise RuntimeError("emulator run receipt has no valid simulated latency")
    return receipt


def compare_emulator_output(build_dir: Path) -> tuple:
    """
    Compare emulator VRAM output against the golden reference.

    Returns:
        (results dict, params dict)
    """
    build_dir = Path(build_dir).resolve()
    vram_file = build_dir / "vram_dump.bin"
    if not vram_file.exists():
        raise FileNotFoundError(
            f"{vram_file} is missing; a global emulator dump is not attributable "
            "to this build directory"
        )
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


def _run_and_assert_locked(
    build_dir: Path, op_name: str, mlen: int = 64, blen: int = 4, vlen: int | None = None, threads: int | None = None
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
    run_metrics = run_emulator(build_dir, threads=threads)

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
        _mark_numerical_validation(build_dir)
        run_metrics["numerical_validation_passed"] = True
        print(f"\n[ATen-style {op_name} test PASSED - ISA generated + emulator verified]")
    else:
        print(f"\n[ATen-style {op_name} test FAILED - emulator numerical check failed]")
        sys.exit(1)

    return run_metrics


def run_and_assert(
    build_dir: Path,
    op_name: str,
    mlen: int = 64,
    blen: int = 4,
    vlen: int | None = None,
    threads: int | None = None,
) -> dict:
    """Generate no mutable state outside an exclusive execution window."""
    resolved = Path(build_dir).resolve()
    build_lease = acquire_build_directory(resolved)
    try:
        execution_lease = acquire_build_directory(_emulator_execution_directory())
        try:
            return _run_and_assert_locked(
                resolved,
                op_name,
                mlen=mlen,
                blen=blen,
                vlen=vlen,
                threads=threads,
            )
        finally:
            execution_lease.release()
    finally:
        build_lease.release()


def _emulate_from_result_locked(
    result: dict,
    build_dir: Path,
    asm_name: str,
    mlen: int = 64,
    blen: int = 4,
    vlen: int | None = None,
    threads: int | None = None,
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

    return run_and_assert(build_dir, asm_name, mlen=mlen, blen=blen, vlen=vlen, threads=threads)


def emulate_from_result(
    result: dict,
    build_dir: Path,
    asm_name: str,
    mlen: int = 64,
    blen: int = 4,
    vlen: int | None = None,
    threads: int | None = None,
) -> dict:
    """Materialize, execute, and validate one isolated build transaction."""
    resolved = Path(build_dir).resolve()
    build_lease = acquire_build_directory(resolved)
    try:
        execution_lease = acquire_build_directory(_emulator_execution_directory())
        try:
            return _emulate_from_result_locked(
                result,
                resolved,
                asm_name,
                mlen=mlen,
                blen=blen,
                vlen=vlen,
                threads=threads,
            )
        finally:
            execution_lease.release()
    finally:
        build_lease.release()
