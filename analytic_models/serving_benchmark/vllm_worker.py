"""One-process offline-vLLM worker with phase and board-energy instrumentation."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gzip
import inspect
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

from .io import sha256_json, write_json_atomic
from .manifest import BenchmarkManifest, BenchmarkPoint, load_manifest
from .nvlink import DcgmNvlinkMonitor
from .phases import PhaseTracker
from .power import PowerMonitor, power_summary, write_power_marks
from .runtime import runtime_point_fingerprint


class CapacityInfeasibleError(RuntimeError):
    pass


def _vllm_classes() -> tuple[Any, Any, Any]:
    try:
        from vllm import EngineArgs, LLMEngine, SamplingParams
    except ImportError as exc:
        raise RuntimeError("vLLM is required to execute RunPod benchmark points") from exc
    return EngineArgs, LLMEngine, SamplingParams


def _supported_kwargs(callable_object: Any, values: dict[str, Any], *, required: set[str]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    supports_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if supports_var_kwargs:
        return values
    missing = sorted(name for name in required if name not in parameters)
    if missing:
        raise RuntimeError(f"installed vLLM does not support required EngineArgs fields: {missing}")
    return {name: value for name, value in values.items() if name in parameters}


def create_engine(
    points: tuple[BenchmarkPoint, ...],
    *,
    revision: str,
    quantization: str,
) -> Any:
    if not points:
        raise ValueError("worker received no points")
    first = points[0]
    for point in points[1:]:
        if point.engine_key(revision=revision, quantization=quantization) != first.engine_key(
            revision=revision, quantization=quantization
        ):
            raise ValueError("all points in a worker must share one vLLM engine configuration")

    EngineArgs, LLMEngine, _ = _vllm_classes()
    options = dict(first.engine_options)
    speculative = bool(options.pop("speculative_decoding", False))
    if speculative:
        raise ValueError("speculative decoding must remain disabled for phase measurement")
    values: dict[str, Any] = {
        "model": first.model_id,
        "revision": revision,
        "tokenizer_revision": revision,
        "dtype": first.dtype,
        "quantization": quantization,
        "tensor_parallel_size": first.tensor_parallel_size,
        "max_model_len": first.max_model_len,
        "rope_scaling": first.rope_scaling,
        "max_num_seqs": max(point.local_batch_size for point in points),
        "enable_prefix_caching": bool(options.pop("enable_prefix_caching", False)),
        "cpu_offload_gb": float(options.pop("cpu_offload_gb", 0)),
        "swap_space": float(options.pop("swap_space", 0)),
        "trust_remote_code": bool(options.pop("trust_remote_code", False)),
        **options,
    }
    required = {
        "model",
        "revision",
        "tokenizer_revision",
        "quantization",
        "tensor_parallel_size",
        "max_model_len",
    }
    if first.rope_scaling is not None:
        required.add("rope_scaling")
    engine_args = EngineArgs(**_supported_kwargs(EngineArgs, values, required=required))
    return LLMEngine.from_engine_args(engine_args)


def _token_pool(engine: Any) -> tuple[int, ...]:
    tokenizer = engine.get_tokenizer()
    vocabulary = tokenizer.get_vocab()
    special = set(getattr(tokenizer, "all_special_ids", ()))
    pool = tuple(sorted({int(token_id) for token_id in vocabulary.values()} - special))
    if not pool:
        raise RuntimeError("tokenizer has no non-special token IDs")
    return pool


def _engine_metadata(engine: Any) -> dict[str, Any]:
    cache = getattr(engine, "cache_config", None)
    if cache is None:
        cache = getattr(getattr(engine, "vllm_config", None), "cache_config", None)
    block_size = getattr(cache, "block_size", None)
    num_gpu_blocks = getattr(cache, "num_gpu_blocks", None)
    token_capacity = None
    if block_size is not None and num_gpu_blocks is not None:
        token_capacity = int(block_size) * int(num_gpu_blocks)
    scheduler = getattr(engine, "scheduler_config", None)
    if scheduler is None:
        scheduler = getattr(getattr(engine, "vllm_config", None), "scheduler_config", None)
    return {
        "cache_block_size_tokens": block_size,
        "gpu_cache_blocks_per_rank": num_gpu_blocks,
        "kv_cache_token_capacity_per_replica": token_capacity,
        "kv_capacity_fidelity": "vllm_cache_block_capacity" if token_capacity is not None else "unavailable",
        "scheduler": {
            "max_num_seqs": getattr(scheduler, "max_num_seqs", None),
            "max_num_batched_tokens": getattr(scheduler, "max_num_batched_tokens", None),
            "enable_chunked_prefill": getattr(scheduler, "enable_chunked_prefill", None),
            "num_scheduler_steps": getattr(scheduler, "num_scheduler_steps", None),
        },
    }


def deterministic_prompt_token_ids(
    token_pool: tuple[int, ...], *, length: int, seed: int, request_index: int
) -> list[int]:
    rng = random.Random((seed << 16) ^ request_index)
    return [token_pool[rng.randrange(len(token_pool))] for _ in range(length)]


def _completion_tokens(output: Any) -> tuple[int, ...]:
    completions = getattr(output, "outputs", None)
    if not completions or len(completions) != 1:
        raise RuntimeError("benchmark requires exactly one completion per request")
    return tuple(int(token) for token in completions[0].token_ids)


def _add_request(engine: Any, *, request_id: str, prompt_token_ids: list[int], sampling_params: Any) -> None:
    prompt = {"prompt_token_ids": prompt_token_ids}
    try:
        engine.add_request(request_id, prompt, sampling_params)
    except TypeError:
        engine.add_request(request_id=request_id, prompt=prompt, params=sampling_params)


def _failure_status(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    capacity_markers = (
        "outofmemory",
        "out of memory",
        "kv cache is not enough",
        "maximum number of tokens",
        "no available memory for the cache blocks",
        "does not fit in the available kv cache",
    )
    return "capacity_infeasible" if any(marker in text for marker in capacity_markers) else "failed"


def execute_batch(
    engine: Any,
    point: BenchmarkPoint,
    *,
    output_tokens: int,
    repetition_label: str,
    token_pool: tuple[int, ...],
    output_dir: Path | None,
    physical_gpu_ids: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    _, _, SamplingParams = _vllm_classes()
    request_ids = tuple(f"{point.point_id}.{repetition_label}.r{index}" for index in range(point.local_batch_size))
    sampling_values = {
        "temperature": 0.0,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "seed": point.token_seed,
    }
    sampling_params = SamplingParams(
        **_supported_kwargs(SamplingParams, sampling_values, required={"temperature", "max_tokens", "ignore_eos"})
    )
    for request_index, request_id in enumerate(request_ids):
        prompt = deterministic_prompt_token_ids(
            token_pool,
            length=point.input_tokens,
            seed=point.token_seed,
            request_index=request_index,
        )
        _add_request(engine, request_id=request_id, prompt_token_ids=prompt, sampling_params=sampling_params)

    monitor = None
    nvlink_monitor = None
    if output_dir is not None:
        monitored_gpu_ids = physical_gpu_ids or point.gpu_ids
        nvlink_monitor = DcgmNvlinkMonitor(gpu_indices=monitored_gpu_ids, output_dir=output_dir)
        nvlink_monitor.start()
        monitor = PowerMonitor(
            gpu_indices=monitored_gpu_ids,
            output_path=output_dir / "power_samples.csv.gz",
            sampling_hz=point.sampling_hz,
        )
        monitor.start()
        monitor.mark("idle_start")
        time.sleep(point.idle_baseline_seconds)
        monitor.mark("idle_end")
        nvlink_monitor.mark_start()
        start_mark = monitor.mark("request_start")
        request_start_s = start_mark.monotonic_s
    else:
        request_start_s = time.monotonic()

    tracker = PhaseTracker(request_ids=request_ids, expected_output_tokens=output_tokens)
    request_diagnostics = {
        request_id: {"max_cached_tokens": 0, "max_preemptions": 0} for request_id in request_ids
    }
    output_token_hashes: dict[str, str] = {}
    event_file = (
        gzip.open(output_dir / "engine_events.jsonl.gz", "wt", encoding="utf-8")
        if output_dir is not None
        else None
    )
    prefill_marked = False
    first_decode_marked = False
    try:
        while engine.has_unfinished_requests():
            outputs = engine.step()
            timestamp_s = time.monotonic()
            event_outputs: list[dict[str, Any]] = []
            for output in outputs:
                request_id = str(output.request_id)
                if request_id not in tracker.requests:
                    raise RuntimeError(f"vLLM returned an unknown request ID: {request_id}")
                tokens = _completion_tokens(output)
                tracker.observe(
                    request_id,
                    cumulative_tokens=len(tokens),
                    finished=bool(output.finished),
                    timestamp_s=timestamp_s,
                )
                event_outputs.append(
                    {"request_id": request_id, "cumulative_tokens": len(tokens), "finished": bool(output.finished)}
                )
                if output.finished:
                    output_token_hashes[request_id] = sha256_json(tokens)
                metrics = getattr(output, "metrics", None)
                if metrics is not None:
                    cached = getattr(metrics, "num_cached_tokens", 0) or 0
                    preemptions = getattr(metrics, "num_preemptions", 0) or 0
                    request_diagnostics[request_id]["max_cached_tokens"] = max(
                        request_diagnostics[request_id]["max_cached_tokens"], int(cached)
                    )
                    request_diagnostics[request_id]["max_preemptions"] = max(
                        request_diagnostics[request_id]["max_preemptions"], int(preemptions)
                    )
            if event_file is not None:
                event_file.write(
                    json.dumps(
                        {"monotonic_s": timestamp_s, "outputs": event_outputs},
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            if monitor is not None and tracker.all_have_first_token and not prefill_marked:
                monitor.mark("prefill_complete")
                prefill_marked = True
            if monitor is not None and tracker.all_have_second_token and not first_decode_marked:
                monitor.mark("first_decode_complete")
                first_decode_marked = True
    finally:
        marks = ()
        nvlink_summary = None
        if monitor is not None:
            if tracker.all_finished:
                monitor.mark("request_complete")
            marks = monitor.stop()
        if nvlink_monitor is not None:
            if tracker.all_finished:
                nvlink_monitor.mark_end()
            nvlink_summary = nvlink_monitor.stop()
        if event_file is not None:
            event_file.close()

    phase = tracker.summary(request_start_s=request_start_s)
    if phase["multi_token_step_observed"]:
        raise RuntimeError("multiple output tokens appeared in one engine step; decode phase fidelity is invalid")
    if any(item["max_preemptions"] for item in request_diagnostics.values()):
        raise RuntimeError("vLLM request preemption was observed")
    if any(item["max_cached_tokens"] for item in request_diagnostics.values()):
        raise RuntimeError("vLLM reported cached prompt tokens while prefix caching is disabled")
    if set(output_token_hashes) != set(request_ids):
        raise RuntimeError("missing final output token hashes")
    result: dict[str, Any] = {
        "phase": phase,
        "request_diagnostics": request_diagnostics,
        "output_token_hashes": [output_token_hashes[request_id] for request_id in request_ids],
    }
    if output_dir is not None:
        with gzip.open(output_dir / "token_timestamps.csv.gz", "wt", encoding="utf-8", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(("request_id", "output_token_index", "monotonic_s"))
            writer.writerows(tracker.token_timestamp_rows())
        write_power_marks(output_dir / "power_marks.json", marks)
        result["power"] = power_summary(output_dir / "power_samples.csv.gz", marks)
        assert nvlink_summary is not None
        result["nvlink"] = nvlink_summary
    return result


def execute_point(
    engine: Any,
    point: BenchmarkPoint,
    *,
    revision: str,
    quantization: str,
    environment_hash: str,
    output_root: Path,
    token_pool: tuple[int, ...],
    physical_gpu_ids: tuple[int, ...] | None = None,
    barrier: Any | None = None,
) -> dict[str, Any]:
    point_dir = output_root / "points" / point.point_id
    summary_path = point_dir / "summary.json"
    fingerprint = runtime_point_fingerprint(
        point,
        revision=revision,
        quantization=quantization,
        environment_hash=environment_hash,
    )
    point_dir.mkdir(parents=True, exist_ok=True)
    assigned_gpu_ids = physical_gpu_ids or point.gpu_ids
    write_json_atomic(
        point_dir / "point_manifest.json",
        {
            "point": point.as_dict(),
            "physical_gpu_ids": list(assigned_gpu_ids),
            "point_fingerprint": fingerprint,
            "resolved_revision": revision,
            "quantization": quantization,
            "environment_hash": environment_hash,
        },
    )
    try:
        warmup = None
        if point.kind != "preflight":
            warmup = execute_batch(
                engine,
                point,
                output_tokens=min(point.warmup_output_tokens, point.output_tokens),
                repetition_label="warmup",
                token_pool=token_pool,
                output_dir=None,
                physical_gpu_ids=assigned_gpu_ids,
            )
        repetitions: list[dict[str, Any]] = []
        for repetition in range(point.repetitions):
            if barrier is not None:
                barrier(repetition)
            repetition_dir = point_dir / f"repeat_{repetition:02d}"
            repetition_dir.mkdir(parents=True, exist_ok=True)
            repetitions.append(
                execute_batch(
                    engine,
                    point,
                    output_tokens=point.output_tokens,
                    repetition_label=f"repeat{repetition}",
                    token_pool=token_pool,
                    output_dir=repetition_dir,
                    physical_gpu_ids=assigned_gpu_ids,
                )
            )
        result = {
            "schema_version": "runpod-serving-point-v1",
            "status": "complete",
            "point_id": point.point_id,
            "point_fingerprint": fingerprint,
            "point": point.as_dict(),
            "physical_gpu_ids": list(assigned_gpu_ids),
            "resolved_revision": revision,
            "quantization": quantization,
            "environment_hash": environment_hash,
            "phase_fidelity": "vllm_offline_first_output_boundary",
            "decode_fidelity": "imported_kv_decode_proxy",
            "real_kv_import_performed": False,
            "engine_metadata": _engine_metadata(engine),
            "warmup": warmup,
            "repetitions": repetitions,
        }
        write_json_atomic(summary_path, result)
        return result
    except BaseException as exc:
        failure = {
            "schema_version": "runpod-serving-point-v1",
            "status": _failure_status(exc),
            "point_id": point.point_id,
            "point_fingerprint": fingerprint,
            "point": point.as_dict(),
            "physical_gpu_ids": list(assigned_gpu_ids),
            "resolved_revision": revision,
            "quantization": quantization,
            "environment_hash": environment_hash,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json_atomic(summary_path, failure)
        raise


def run_worker(
    *,
    manifest: BenchmarkManifest,
    point_ids: tuple[str, ...],
    revision: str,
    quantization: str,
    environment_hash: str,
    output_root: Path,
    physical_gpu_ids: tuple[int, ...] | None = None,
    barrier_root: Path | None = None,
    barrier_label: str | None = None,
    barrier_participants: int = 1,
    repetitions_override: int | None = None,
    measurement_stage: str | None = None,
) -> None:
    points = tuple(manifest.point_by_id(point_id) for point_id in point_ids)
    if repetitions_override is not None or measurement_stage is not None:
        points = tuple(
            dataclasses.replace(
                point,
                repetitions=repetitions_override or point.repetitions,
                measurement_stage=measurement_stage or point.measurement_stage,
            )
            for point in points
        )
    if physical_gpu_ids is not None:
        if len(physical_gpu_ids) != points[0].tensor_parallel_size:
            raise ValueError("physical GPU count must match tensor_parallel_size")
        if len(set(physical_gpu_ids)) != len(physical_gpu_ids):
            raise ValueError("physical GPU IDs must be unique")
    barrier = None
    if barrier_root is not None:
        if not barrier_label or barrier_participants < 2:
            raise ValueError("replica barrier requires a label and at least two participants")

        def wait_at_barrier(repetition: int) -> None:
            barrier_root.mkdir(parents=True, exist_ok=True)
            (barrier_root / f"repeat_{repetition:02d}.{barrier_label}.ready").touch()
            deadline = time.monotonic() + 1800.0
            pattern = f"repeat_{repetition:02d}.*.ready"
            while len(tuple(barrier_root.glob(pattern))) < barrier_participants:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for replica barrier {repetition}")
                time.sleep(0.1)

        barrier = wait_at_barrier
    engine = create_engine(points, revision=revision, quantization=quantization)
    token_pool = _token_pool(engine)
    for point in points:
        try:
            execute_point(
                engine,
                point,
                revision=revision,
                quantization=quantization,
                environment_hash=environment_hash,
                output_root=output_root,
                token_pool=token_pool,
                physical_gpu_ids=physical_gpu_ids,
                barrier=barrier,
            )
        except Exception as exc:
            optional_capacity_failure = (
                point.kind == "preflight"
                and not point.required_success
                and _failure_status(exc) == "capacity_infeasible"
            )
            if not optional_capacity_failure:
                if point.kind == "formal" and _failure_status(exc) == "capacity_infeasible":
                    raise CapacityInfeasibleError(point.point_id) from exc
                raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--quantization", required=True)
    parser.add_argument("--environment-hash", required=True)
    parser.add_argument("--point-id", action="append", required=True)
    parser.add_argument("--physical-gpu-ids", help="physical GPU IDs assigned by the parent scheduler")
    parser.add_argument("--barrier-root", type=Path)
    parser.add_argument("--barrier-label")
    parser.add_argument("--barrier-participants", type=int, default=1)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--measurement-stage")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        run_worker(
            manifest=load_manifest(args.manifest),
            point_ids=tuple(args.point_id),
            revision=args.revision,
            quantization=args.quantization,
            environment_hash=args.environment_hash,
            output_root=args.output_root,
            physical_gpu_ids=(
                tuple(int(item) for item in args.physical_gpu_ids.split(","))
                if args.physical_gpu_ids
                else None
            ),
            barrier_root=args.barrier_root,
            barrier_label=args.barrier_label,
            barrier_participants=args.barrier_participants,
            repetitions_override=args.repetitions,
            measurement_stage=args.measurement_stage,
        )
    except CapacityInfeasibleError:
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
