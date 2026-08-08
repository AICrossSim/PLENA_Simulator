"""Resumable parent orchestration for RunPod preflight and formal runs."""

from __future__ import annotations

import dataclasses
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import (
    create_environment_lock,
    load_environment_lock,
    resolve_model_revisions,
    validate_environment_lock,
    validate_runpod_persistent_paths,
    write_environment_lock,
)
from .aggregate import aggregate_point
from .inventory import validate_a100_sxm_inventory, validate_inventory_hash
from .io import completed_artifact_matches, read_json, sha256_json, write_json_atomic
from .manifest import BenchmarkManifest, BenchmarkPoint
from .runtime import runtime_point_fingerprint


def select_points(
    points: Iterable[BenchmarkPoint],
    *,
    models: set[str] | None = None,
    workloads: set[str] | None = None,
    point_ids: set[str] | None = None,
) -> tuple[BenchmarkPoint, ...]:
    selected = tuple(
        point
        for point in points
        if (not models or point.model_name in models)
        and (not workloads or point.workload_name in workloads)
        and (not point_ids or point.point_id in point_ids)
    )
    if point_ids:
        missing = point_ids - {point.point_id for point in selected}
        if missing:
            raise ValueError(f"unknown or filtered point IDs: {sorted(missing)}")
    return selected


def group_points(
    points: Iterable[BenchmarkPoint],
    *,
    revisions: dict[str, str],
    quantization: str,
) -> tuple[tuple[BenchmarkPoint, ...], ...]:
    groups: dict[str, list[BenchmarkPoint]] = defaultdict(list)
    for point in points:
        groups[
            point.engine_key(revision=revisions[point.model_name], quantization=quantization)
        ].append(point)
    return tuple(
        tuple(sorted(group, key=lambda point: (point.input_tokens, point.local_batch_size, point.point_id)))
        for _, group in sorted(groups.items())
    )


def _summary_path(output_root: Path, point: BenchmarkPoint) -> Path:
    return output_root / "points" / point.point_id / "summary.json"


def pending_points(
    points: Iterable[BenchmarkPoint],
    *,
    output_root: Path,
    revisions: dict[str, str],
    quantization: str,
    environment_hash: str,
) -> tuple[BenchmarkPoint, ...]:
    pending: list[BenchmarkPoint] = []
    for point in points:
        fingerprint = runtime_point_fingerprint(
            point,
            revision=revisions[point.model_name],
            quantization=quantization,
            environment_hash=environment_hash,
        )
        path = _summary_path(output_root, point)
        terminal = completed_artifact_matches(path, fingerprint=fingerprint)
        if path.is_file() and not terminal:
            payload = read_json(path)
            terminal = (
                payload.get("status") == "capacity_infeasible"
                and payload.get("point_fingerprint") == fingerprint
            )
            if payload.get("status") == "complete":
                raise RuntimeError(f"completed point has a stale fingerprint and will not be overwritten: {path}")
        if not terminal:
            pending.append(point)
    return tuple(pending)


def _worker_command(
    *,
    manifest: BenchmarkManifest,
    points: tuple[BenchmarkPoint, ...],
    revision: str,
    quantization: str,
    environment_hash: str,
    output_root: Path,
    physical_gpu_ids: tuple[int, ...] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "analytic_models.serving_benchmark.vllm_worker",
        "--manifest",
        str(manifest.source_path),
        "--output-root",
        str(output_root),
        "--revision",
        revision,
        "--quantization",
        quantization,
        "--environment-hash",
        environment_hash,
    ]
    for point in points:
        command.extend(("--point-id", point.point_id))
    repetition_counts = {point.repetitions for point in points}
    stages = {point.measurement_stage for point in points}
    if len(repetition_counts) != 1 or len(stages) != 1:
        raise ValueError("worker points must share one repetition count and measurement stage")
    command.extend(("--repetitions", str(next(iter(repetition_counts)))))
    command.extend(("--measurement-stage", next(iter(stages))))
    if physical_gpu_ids is not None:
        command.extend(("--physical-gpu-ids", ",".join(str(gpu) for gpu in physical_gpu_ids)))
    return command


@dataclass
class _RunningGroup:
    group_index: int
    launch_index: int
    points: tuple[BenchmarkPoint, ...]
    gpu_ids: tuple[int, ...]
    process: subprocess.Popen[str]
    log_handle: Any
    log_path: Path
    started_epoch_s: float


def _allocate_gpu_group(
    available_gpu_ids: set[int],
    required: int,
) -> tuple[int, ...] | None:
    if required <= 0:
        raise ValueError("required GPU count must be positive")
    if len(available_gpu_ids) < required:
        return None
    ordered = sorted(available_gpu_ids)
    for start in range(len(ordered) - required + 1):
        candidate = tuple(ordered[start : start + required])
        if candidate[-1] - candidate[0] == required - 1:
            return candidate
    return tuple(ordered[:required])


def _execution_groups(
    points: tuple[BenchmarkPoint, ...],
    *,
    revisions: dict[str, str],
    quantization: str,
    granularity: str,
    gpu_capacity: int = 8,
    max_parallel_groups: int = 8,
) -> tuple[tuple[BenchmarkPoint, ...], ...]:
    if granularity == "point":
        ordered = sorted(
            points,
            key=lambda point: (
                -point.tensor_parallel_size,
                -point.local_batch_size,
                -point.input_tokens,
                point.point_id,
            ),
        )
        return tuple((point,) for point in ordered)
    groups = group_points(points, revisions=revisions, quantization=quantization)
    if granularity == "sharded-engine":
        sharded: list[tuple[BenchmarkPoint, ...]] = []
        for group in groups:
            shard_count = min(
                len(group),
                max(1, gpu_capacity // group[0].tensor_parallel_size),
                max_parallel_groups,
            )
            shards: list[list[BenchmarkPoint]] = [[] for _ in range(shard_count)]
            shard_work = [0] * shard_count
            for point in sorted(
                group,
                key=lambda item: -(
                    (item.input_tokens + item.output_tokens) * item.local_batch_size
                ),
            ):
                shard_index = min(range(shard_count), key=shard_work.__getitem__)
                shards[shard_index].append(point)
                shard_work[shard_index] += (
                    point.input_tokens + point.output_tokens
                ) * point.local_batch_size
            sharded.extend(tuple(shard) for shard in shards if shard)
        groups = tuple(sharded)
    elif granularity != "engine":
        raise ValueError(f"unknown execution granularity: {granularity}")
    return tuple(
        sorted(
            groups,
            key=lambda group: (
                -group[0].tensor_parallel_size,
                -sum(
                    (point.input_tokens + point.output_tokens) * point.local_batch_size
                    for point in group
                ),
                group[0].point_id,
            ),
        )
    )


def _terminate_running_groups(running: list[_RunningGroup]) -> None:
    for worker in running:
        if worker.process.poll() is None:
            try:
                os.killpg(worker.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for worker in running:
        try:
            worker.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(worker.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.process.wait()
        worker.log_handle.close()


def _ephemeral_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _formal_execution_policy(measurement_stage: str, execution_mode: str) -> tuple[str, bool]:
    if execution_mode == "sequential":
        return "engine", False
    if execution_mode == "gpu-parallel":
        return "point", True
    if execution_mode != "auto":
        raise ValueError(f"unknown execution mode: {execution_mode}")
    if measurement_stage == "screening":
        return "sharded-engine", True
    if measurement_stage == "short-sweep":
        return "engine", True
    return "engine", False


def execute_groups(
    *,
    manifest: BenchmarkManifest,
    points: tuple[BenchmarkPoint, ...],
    revisions: dict[str, str],
    quantization: str,
    environment_hash: str,
    output_root: Path,
    resume: bool,
    physical_gpu_pool: tuple[int, ...] | None = None,
    max_concurrent_engines: int = 1,
    granularity: str = "engine",
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    selected = (
        pending_points(
            points,
            output_root=output_root,
            revisions=revisions,
            quantization=quantization,
            environment_hash=environment_hash,
        )
        if resume
        else points
    )
    if not selected:
        return []
    if max_concurrent_engines < 1:
        raise ValueError("max_concurrent_engines must be positive")
    if physical_gpu_pool is None:
        physical_gpu_pool = tuple(sorted({gpu for point in selected for gpu in point.gpu_ids}))
    if not physical_gpu_pool or len(set(physical_gpu_pool)) != len(physical_gpu_pool):
        raise ValueError("physical_gpu_pool must contain unique GPU IDs")
    if any(point.tensor_parallel_size > len(physical_gpu_pool) for point in selected):
        raise ValueError("a selected point requires more GPUs than the physical pool")

    queued = list(
        enumerate(
            _execution_groups(
                selected,
                revisions=revisions,
                quantization=quantization,
                granularity=granularity,
                gpu_capacity=len(physical_gpu_pool),
                max_parallel_groups=max_concurrent_engines,
            )
        )
    )
    outcomes: list[dict[str, Any]] = []
    running: list[_RunningGroup] = []
    available_gpu_ids = set(physical_gpu_pool)
    launch_index = 0

    try:
        while queued or running:
            launched = False
            while queued and len(running) < max_concurrent_engines:
                fitting_index = None
                assigned_gpu_ids = None
                for index, (_, group) in enumerate(queued):
                    assigned_gpu_ids = _allocate_gpu_group(
                        available_gpu_ids,
                        group[0].tensor_parallel_size,
                    )
                    if assigned_gpu_ids is not None:
                        fitting_index = index
                        break
                if fitting_index is None or assigned_gpu_ids is None:
                    break
                group_index, group = queued.pop(fitting_index)
                if any(point.tensor_parallel_size != len(assigned_gpu_ids) for point in group):
                    raise ValueError("engine group contains inconsistent tensor-parallel sizes")
                revision = revisions[group[0].model_name]
                log_path = (
                    output_root
                    / "logs"
                    / f"engine_group_{group_index:03d}_launch_{launch_index:03d}.log"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in assigned_gpu_ids)
                environment.setdefault("TOKENIZERS_PARALLELISM", "false")
                rendezvous_port = _ephemeral_local_port()
                environment["MASTER_PORT"] = str(rendezvous_port)
                environment["VLLM_PORT"] = str(rendezvous_port)
                command = _worker_command(
                    manifest=manifest,
                    points=group,
                    revision=revision,
                    quantization=quantization,
                    environment_hash=environment_hash,
                    output_root=output_root,
                    physical_gpu_ids=assigned_gpu_ids,
                )
                log_handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=manifest.source_path.parents[3],
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                for gpu in assigned_gpu_ids:
                    available_gpu_ids.remove(gpu)
                running.append(
                    _RunningGroup(
                        group_index=group_index,
                        launch_index=launch_index,
                        points=group,
                        gpu_ids=assigned_gpu_ids,
                        process=process,
                        log_handle=log_handle,
                        log_path=log_path,
                        started_epoch_s=time.time(),
                    )
                )
                launch_index += 1
                launched = True

            if launched:
                write_json_atomic(
                    output_root / "active_schedule.json",
                    {
                        "physical_gpu_pool": list(physical_gpu_pool),
                        "active": [
                            {
                                "engine_group": worker.group_index,
                                "point_ids": [point.point_id for point in worker.points],
                                "physical_gpu_ids": list(worker.gpu_ids),
                                "pid": worker.process.pid,
                                "started_epoch_s": worker.started_epoch_s,
                            }
                            for worker in running
                        ],
                    },
                )

            completed_workers = [worker for worker in running if worker.process.poll() is not None]
            if not completed_workers:
                if running:
                    time.sleep(0.25)
                    continue
                if queued:
                    raise RuntimeError("GPU scheduler deadlock: no queued group fits the physical pool")
            for worker in completed_workers:
                running.remove(worker)
                worker.log_handle.close()
                available_gpu_ids.update(worker.gpu_ids)
                returncode = worker.process.returncode
                outcome = {
                    "engine_group": worker.group_index,
                    "launch_index": worker.launch_index,
                    "point_ids": [point.point_id for point in worker.points],
                    "physical_gpu_ids": list(worker.gpu_ids),
                    "returncode": returncode,
                    "log": str(worker.log_path),
                    "started_epoch_s": worker.started_epoch_s,
                    "ended_epoch_s": time.time(),
                }
                outcomes.append(outcome)
                write_json_atomic(
                    output_root / "run_state.json",
                    {
                        "physical_gpu_pool": list(physical_gpu_pool),
                        "max_concurrent_engines": max_concurrent_engines,
                        "granularity": granularity,
                        "groups": outcomes,
                    },
                )
                if returncode not in {0, 75}:
                    _terminate_running_groups(running)
                    running.clear()
                    raise RuntimeError(f"engine group failed; inspect {worker.log_path}")
                pending = pending_points(
                    worker.points,
                    output_root=output_root,
                    revisions=revisions,
                    quantization=quantization,
                    environment_hash=environment_hash,
                )
                if returncode == 0 and pending:
                    _terminate_running_groups(running)
                    running.clear()
                    raise RuntimeError(f"engine group exited successfully with incomplete points: {pending}")
                if pending:
                    queued.append((worker.group_index, pending))
    except BaseException:
        _terminate_running_groups(running)
        raise
    write_json_atomic(output_root / "active_schedule.json", {"active": []})
    return outcomes


def run_preflight(
    *,
    manifest: BenchmarkManifest,
    inventory_path: Path,
    output_root: Path,
    environment_lock_path: Path,
    image_digest: str,
) -> dict[str, Any]:
    inventory = read_json(inventory_path)
    selected_storage_mode = str(inventory.get("storage_mode", "persistent-workspace"))
    validate_runpod_persistent_paths(
        output_root,
        environment_lock_path,
        inventory_path,
        required_storage_mode=selected_storage_mode,
    )
    if not validate_inventory_hash(inventory):
        raise RuntimeError("inventory hash mismatch; recapture the RunPod inventory")
    inventory_errors = validate_a100_sxm_inventory(inventory)
    if inventory_errors:
        raise RuntimeError("inventory is not eligible for formal preflight: " + "; ".join(inventory_errors))
    revisions = resolve_model_revisions(manifest)
    common_candidates = set.intersection(
        *(set(model.quantization_candidates) for model in manifest.models.values())
    )
    candidate_order = [
        candidate
        for candidate in next(iter(manifest.models.values())).quantization_candidates
        if candidate in common_candidates
    ]
    if not candidate_order:
        raise RuntimeError("models have no common quantization backend candidate")

    failures: list[dict[str, Any]] = []
    for quantization in candidate_order:
        candidate_root = output_root / f"candidate_{quantization}"
        provisional_hash = sha256_json(
            {
                "manifest": manifest.fingerprint,
                "revisions": revisions,
                "quantization": quantization,
                "image_digest": image_digest,
            }
        )
        try:
            execute_groups(
                manifest=manifest,
                points=manifest.preflight_points,
                revisions=revisions,
                quantization=quantization,
                environment_hash=provisional_hash,
                output_root=candidate_root,
                resume=True,
            )
        except RuntimeError as exc:
            failures.append({"quantization": quantization, "error": str(exc)})
            continue
        artifacts = []
        for point in manifest.preflight_points:
            summary_path = _summary_path(candidate_root, point)
            summary = read_json(summary_path)
            acceptable = summary.get("status") == "complete" or (
                not point.required_success and summary.get("status") == "capacity_infeasible"
            )
            if not acceptable:
                raise RuntimeError(
                    f"preflight point failed for a non-capacity reason: {point.point_id} "
                    f"({summary.get('status')})"
                )
            artifacts.append(
                {
                    "point_id": point.point_id,
                    "summary_path": str(summary_path),
                    "point_fingerprint": summary["point_fingerprint"],
                    "required_success": point.required_success,
                    "status": summary.get("status"),
                }
            )
        lock = create_environment_lock(
            manifest=manifest,
            inventory=inventory,
            resolved_revisions=revisions,
            quantization=quantization,
            image_digest=image_digest,
            preflight_artifacts=artifacts,
        )
        lock["backend_failures_before_selection"] = failures
        # Rehash after adding the diagnostic field.
        lock.pop("environment_hash", None)
        lock["environment_hash"] = sha256_json(lock)
        write_environment_lock(environment_lock_path, lock)
        return lock
    raise RuntimeError(f"all common quantization backends failed preflight: {failures}")


def run_formal(
    *,
    manifest: BenchmarkManifest,
    environment_lock_path: Path,
    output_root: Path,
    image_digest: str,
    models: set[str] | None = None,
    workloads: set[str] | None = None,
    point_ids: set[str] | None = None,
    measurement_stage: str = "screening",
    execution_mode: str = "auto",
    physical_gpu_pool: tuple[int, ...] = tuple(range(8)),
    max_concurrent_engines: int = 8,
) -> list[dict[str, Any]]:
    lock = load_environment_lock(environment_lock_path)
    validate_runpod_persistent_paths(
        output_root,
        environment_lock_path,
        required_storage_mode=str(lock.get("storage_mode", "persistent-workspace")),
    )
    errors = validate_environment_lock(lock, manifest=manifest, image_digest=image_digest)
    if errors:
        raise RuntimeError("formal run environment differs from preflight: " + "; ".join(errors))
    if measurement_stage not in {"screening", "confirmation", "short-sweep", "holdout"}:
        raise ValueError(f"unknown measurement stage: {measurement_stage}")
    if measurement_stage == "screening" and not workloads and not point_ids:
        workloads = {"primary-90000x8000"}
    elif measurement_stage == "short-sweep" and not workloads and not point_ids:
        workloads = {"short-1400x200"}
    elif measurement_stage == "holdout":
        if not point_ids:
            raise ValueError("holdout stage requires selected --point-ids")
        workloads = {"holdout-114000x5000"}
    elif measurement_stage == "confirmation" and not point_ids:
        raise ValueError("confirmation stage requires selected --point-ids")
    repetitions = 1 if measurement_stage == "screening" else 3
    points = select_points(
        manifest.formal_points,
        models=models,
        workloads=workloads,
        point_ids=point_ids,
    )
    if not points:
        raise ValueError("formal point selection is empty")
    points = tuple(
        dataclasses.replace(
            point,
            repetitions=repetitions,
            measurement_stage=measurement_stage,
        )
        for point in points
    )
    granularity, parallel = _formal_execution_policy(measurement_stage, execution_mode)
    effective_max_concurrent = max_concurrent_engines if parallel else 1
    write_json_atomic(
        output_root / "campaign.json",
        {
            "campaign": manifest.campaign,
            "manifest_hash": manifest.fingerprint,
            "environment_hash": lock["environment_hash"],
            "environment_lock": str(environment_lock_path.resolve()),
            "selected_point_ids": [point.point_id for point in points],
            "measurement_stage": measurement_stage,
            "repetitions": repetitions,
            "execution_mode": execution_mode,
            "execution_granularity": granularity,
            "physical_gpu_pool": list(physical_gpu_pool),
            "max_concurrent_engines": effective_max_concurrent,
        },
    )
    return execute_groups(
        manifest=manifest,
        points=points,
        revisions=lock["resolved_revisions"],
        quantization=lock["quantization_backend"],
        environment_hash=lock["environment_hash"],
        output_root=output_root,
        resume=True,
        physical_gpu_pool=physical_gpu_pool,
        max_concurrent_engines=effective_max_concurrent,
        granularity=granularity,
    )


def run_replica_check(
    *,
    manifest: BenchmarkManifest,
    environment_lock_path: Path,
    formal_output_root: Path,
    output_root: Path,
    point_id: str,
    gpu_groups: tuple[tuple[int, ...], ...],
    image_digest: str,
) -> dict[str, Any]:
    lock = load_environment_lock(environment_lock_path)
    validate_runpod_persistent_paths(
        output_root,
        formal_output_root,
        environment_lock_path,
        required_storage_mode=str(lock.get("storage_mode", "persistent-workspace")),
    )
    errors = validate_environment_lock(lock, manifest=manifest, image_digest=image_digest)
    if errors:
        raise RuntimeError("replica-check environment differs from preflight: " + "; ".join(errors))
    source_point = dataclasses.replace(
        manifest.point_by_id(point_id),
        repetitions=3,
        measurement_stage="replica-check",
    )
    if source_point.kind != "formal":
        raise ValueError("replica-check source must be a formal point")
    if len(gpu_groups) < 2:
        raise ValueError("replica-check requires at least two GPU groups")
    if any(len(group) != source_point.tensor_parallel_size for group in gpu_groups):
        raise ValueError("each replica GPU group must match the point tensor-parallel size")
    flattened = [gpu for group in gpu_groups for gpu in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("replica GPU groups must not overlap")
    independent_path = formal_output_root / "points" / point_id / "summary.json"
    if not independent_path.is_file():
        raise RuntimeError(f"independent formal point must complete before replica-check: {independent_path}")
    independent = aggregate_point(read_json(independent_path))

    output_root.mkdir(parents=True, exist_ok=True)
    barrier_root = output_root / "barriers" / uuid.uuid4().hex
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    revision = lock["resolved_revisions"][source_point.model_name]
    quantization = lock["quantization_backend"]
    for replica_index, group in enumerate(gpu_groups):
        label = chr(ord("a") + replica_index)
        replica_root = output_root / f"replica_{label}"
        log_path = output_root / "logs" / f"replica_{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = _worker_command(
            manifest=manifest,
            points=(source_point,),
            revision=revision,
            quantization=quantization,
            environment_hash=lock["environment_hash"],
            output_root=replica_root,
        )
        command.extend(
            (
                "--physical-gpu-ids",
                ",".join(str(gpu) for gpu in group),
                "--barrier-root",
                str(barrier_root),
                "--barrier-label",
                label,
                "--barrier-participants",
                str(len(gpu_groups)),
            )
        )
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in group)
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=manifest.source_path.parents[3],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((label, process, log_handle))
    failures: list[str] = []
    for label, process, log_handle in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode != 0:
            failures.append(label)
    if failures:
        raise RuntimeError(f"concurrent replica workers failed: {failures}; inspect {output_root / 'logs'}")

    replicas: list[dict[str, Any]] = []
    for replica_index in range(len(gpu_groups)):
        label = chr(ord("a") + replica_index)
        replicas.append(
            aggregate_point(read_json(output_root / f"replica_{label}" / "points" / point_id / "summary.json"))
        )
    concurrent_latency = max(row["median_full_request_latency_s"] for row in replicas)
    concurrent_energy = sum(row["median_complete_energy_mj"] for row in replicas)
    reconstructed_latency = independent["median_full_request_latency_s"]
    reconstructed_energy = independent["median_complete_energy_mj"] * len(replicas)
    result = {
        "schema_version": "runpod-replica-concurrency-v1",
        "point_id": point_id,
        "gpu_groups": [list(group) for group in gpu_groups],
        "replica_count": len(replicas),
        "independent": independent,
        "concurrent_replicas": replicas,
        "concurrent_global_latency_s": concurrent_latency,
        "independent_reconstruction_latency_s": reconstructed_latency,
        "latency_correction_factor": concurrent_latency / reconstructed_latency,
        "concurrent_total_energy_mj": concurrent_energy,
        "independent_reconstruction_energy_mj": reconstructed_energy,
        "energy_correction_factor": concurrent_energy / reconstructed_energy,
    }
    result["requires_concurrency_correction"] = (
        abs(result["latency_correction_factor"] - 1.0) > 0.05
        or abs(result["energy_correction_factor"] - 1.0) > 0.05
    )
    write_json_atomic(output_root / "replica_check.json", result)
    return result
