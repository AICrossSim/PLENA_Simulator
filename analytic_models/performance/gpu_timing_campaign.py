#!/usr/bin/env python3
"""Real-prompt vLLM timing capture with serialized, windowed NVML sampling.

Maintained successor to the immutable 2026-09-03 raw campaign script. Run as
``python -m analytic_models.performance.gpu_timing_campaign`` on the pinned GPU
capture environment. No GPU checkpoint or capture is needed to test the energy
integration and sampler synchronization.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

# Keep the V1 engine in-process so process-pollution checks are exact.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from .gpu_energy import PowerTrace


def _gpu_uuid(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode()
    # PyTorch's CUuuid string omits the NVML "GPU-" prefix.
    return str(value).lower().removeprefix("gpu-")


def _verify_cuda_device(torch_module: Any, nvml_uuid: str) -> str:
    if torch_module.cuda.device_count() != 1:
        raise RuntimeError("capture requires exactly one visible CUDA device")
    actual_uuid = getattr(torch_module.cuda.get_device_properties(0), "uuid", None)
    if actual_uuid is None or _gpu_uuid(actual_uuid) != _gpu_uuid(nvml_uuid):
        raise RuntimeError(f"CUDA/NVML GPU UUID mismatch: CUDA={actual_uuid}, NVML={nvml_uuid}")
    if torch_module.cuda.current_device() != 0:
        raise RuntimeError("capture must run on the verified CUDA device 0")
    return str(actual_uuid)


def _bind_cuda_device(nvml_uuid: str) -> None:
    # Resolve physical ordinals through NVML and bind by UUID before importing
    # vLLM or initializing CUDA. CUDA and NVML ordinal order need not agree.
    loaded_torch = sys.modules.get("torch")
    if loaded_torch is not None and loaded_torch.cuda.is_initialized():
        try:
            _verify_cuda_device(loaded_torch, nvml_uuid)
        except RuntimeError as error:
            raise RuntimeError("CUDA is already initialized; restart before selecting another GPU") from error
    os.environ["CUDA_VISIBLE_DEVICES"] = nvml_uuid


def percentile(values: list[float], pct: float) -> float | str:
    if not values:
        return "N/A"
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def descendants(pid: int) -> set[int]:
    found = {pid}
    queue = [pid]
    while queue:
        current = queue.pop()
        path = Path(f"/proc/{current}/task/{current}/children")
        try:
            children = [int(x) for x in path.read_text().split()]
        except (FileNotFoundError, PermissionError, ValueError):
            children = []
        for child in children:
            if child not in found:
                found.add(child)
                queue.append(child)
    return found


class NvmlSampler:
    """Only the background thread samples NVML; trial labels are metadata only."""

    def __init__(self, gpu_index: int, path: Path, interval_s: float, *, nvml=None) -> None:
        if interval_s <= 0 or not math.isfinite(interval_s):
            raise ValueError("power interval must be positive and finite")
        if nvml is None:
            import pynvml

            nvml = pynvml
        self.nvml = nvml
        nvml.nvmlInit()
        try:
            self.handle = nvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.uuid = nvml.nvmlDeviceGetUUID(self.handle)
            if isinstance(self.uuid, bytes):
                self.uuid = self.uuid.decode()
        except BaseException:
            nvml.nvmlShutdown()
            raise
        self.path = path
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.context: dict[str, Any] = {"trial_id": "idle", "phase": "idle"}
        self.rows: list[dict[str, Any]] = []
        self.unexpected_pids: set[int] = set()
        self.power_supported = True
        self.power_error: str | None = None
        self.worker_error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def set_context(self, **context: Any) -> None:
        with self.condition:
            self.context = context

    def _sample_once(self) -> None:
        with self.condition:
            context = dict(self.context)
        # Timestamp the actual power query, before unrelated NVML/proc queries.
        before_ns = time.perf_counter_ns()
        try:
            power = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 if self.power_supported else None
        except self.nvml.NVMLError as error:
            self.power_supported = False
            self.power_error = f"{type(error).__name__}: {error}"
            power = None
        after_ns = time.perf_counter_ns()
        util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
        memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        try:
            pids = sorted(int(p.pid) for p in self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle))
        except self.nvml.NVMLError:
            pids = []
        unexpected = sorted(set(pids) - descendants(os.getpid()))
        row = {
            "timestamp_ns": (before_ns + after_ns) // 2,
            "power_read_start_ns": before_ns,
            "power_read_end_ns": after_ns,
            "wall_time_ns": time.time_ns(),
            "gpu_uuid": self.uuid,
            "power_w": power,
            "gpu_utilization_pct": int(util.gpu),
            "memory_utilization_pct": int(util.memory),
            "memory_used_bytes": int(memory.used),
            "compute_pids": pids,
            "unexpected_compute_pids": unexpected,
            **context,
        }
        with self.condition:
            self.unexpected_pids.update(unexpected)
            self.rows.append(row)
            self.condition.notify_all()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                start = time.perf_counter()
                self._sample_once()
                self.stop_event.wait(max(0.0, self.interval_s - (time.perf_counter() - start)))
        except BaseException as error:
            with self.condition:
                self.worker_error = error
                self.condition.notify_all()

    def _wait_for_sample(self, timestamp_ns: int) -> None:
        with self.condition:
            available = self.condition.wait_for(
                lambda: (
                    self.worker_error is not None or bool(self.rows and self.rows[-1]["timestamp_ns"] >= timestamp_ns)
                ),
                timeout=max(5.0, 3 * self.interval_s),
            )
            if self.worker_error is not None:
                raise RuntimeError("NVML sampling worker failed") from self.worker_error
            if not available:
                raise RuntimeError("NVML samples do not bracket the completed batch")

    def start(self) -> None:
        before_start = time.perf_counter_ns()
        self.thread.start()
        self._wait_for_sample(before_start)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=max(5.0, 3 * self.interval_s))
        if self.thread.is_alive():
            raise RuntimeError("NVML sampler did not stop; refusing an incomplete power archive")
        try:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                fields = [
                    "timestamp_ns",
                    "power_read_start_ns",
                    "power_read_end_ns",
                    "wall_time_ns",
                    "gpu_uuid",
                    "trial_id",
                    "phase",
                    "benchmark",
                    "mode",
                    "batch_size",
                    "group_index",
                    "measurement",
                    "power_w",
                    "gpu_utilization_pct",
                    "memory_utilization_pct",
                    "memory_used_bytes",
                    "compute_pids",
                    "unexpected_compute_pids",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in sorted(self.rows, key=lambda value: value["timestamp_ns"]):
                    output = dict(row)
                    output["power_w"] = "N/A" if row["power_w"] is None else row["power_w"]
                    output["compute_pids"] = json.dumps(row["compute_pids"])
                    output["unexpected_compute_pids"] = json.dumps(row["unexpected_compute_pids"])
                    writer.writerow({key: output.get(key, "N/A") for key in fields})
        finally:
            self.nvml.nvmlShutdown()

    def trial_stats(self, trial_id: str, start_ns: int, end_ns: int) -> dict[str, Any]:
        self._wait_for_sample(end_ns)
        with self.condition:
            rows = list(self.rows)
        inside = [r for r in rows if start_ns <= r["timestamp_ns"] <= end_ns]
        energy = PowerTrace(rows).integrate(start_ns, end_ns) if self.power_supported else "N/A"
        return {
            "power_sample_count": len(inside),
            "energy_joules": energy,
            "energy_method": "sorted_interpolated_recorded_batch_window_v2",
            "energy_trial_id": trial_id,
            "gpu_memory_used_peak_bytes": max((r["memory_used_bytes"] for r in inside), default="N/A"),
        }


def build_groups(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_benchmark[sample["benchmark"]].append(sample)
    groups = []
    # Every sample gets a full 128-token B1 case.
    for benchmark in sorted(by_benchmark):
        ordered = sorted(by_benchmark[benchmark], key=lambda x: (x["prompt_length"], x["sample_id"]))
        for index, sample in enumerate(ordered):
            groups.append(
                {
                    "benchmark": benchmark,
                    "mode": "b1_long",
                    "batch_size": 1,
                    "generation_limit": 128,
                    "group_index": index,
                    "samples": [sample],
                }
            )
    # Short generation sweep, with contiguous length-sorted groups per benchmark.
    for batch_size in (1, 2, 4, 8, 16):
        for benchmark in sorted(by_benchmark):
            ordered = sorted(by_benchmark[benchmark], key=lambda x: (x["prompt_length"], x["sample_id"]))
            for index in range(0, len(ordered), batch_size):
                chunk = ordered[index : index + batch_size]
                if len(chunk) != batch_size:
                    raise RuntimeError(f"incomplete batch for {benchmark} B{batch_size}")
                groups.append(
                    {
                        "benchmark": benchmark,
                        "mode": "batch_sweep",
                        "batch_size": batch_size,
                        "generation_limit": 32,
                        "group_index": index // batch_size,
                        "samples": chunk,
                    }
                )
    return groups


def sampling_params(max_tokens: int):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=False,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )


def run_batch(llm: Any, samples: list[dict[str, Any]], max_tokens: int) -> tuple[list[dict[str, Any]], int, int]:
    params = sampling_params(max_tokens)
    group_start_ns = time.perf_counter_ns()
    state: dict[str, dict[str, Any]] = {}
    for sample in samples:
        arrival_ns = time.perf_counter_ns()
        request_id = llm._add_request({"prompt_token_ids": sample["prompt_token_ids"]}, params)
        state[request_id] = {
            "sample": sample,
            "request_id": request_id,
            "arrival_ns": arrival_ns,
            "token_timestamps_ns": [],
            "generated_token_ids": [],
            "finish_ns": None,
            "finish_reason": None,
        }
    while llm.llm_engine.has_unfinished_requests():
        outputs = llm.llm_engine.step()
        now_ns = time.perf_counter_ns()
        for output in outputs:
            if output.request_id not in state:
                continue
            item = state[output.request_id]
            for completion in output.outputs:
                ids = list(completion.token_ids)
                item["generated_token_ids"].extend(ids)
                item["token_timestamps_ns"].extend([now_ns] * len(ids))
                item["finish_reason"] = completion.finish_reason
            if output.finished:
                item["finish_ns"] = now_ns
    group_end_ns = time.perf_counter_ns()
    records = []
    padded_length = max(x["prompt_length"] for x in samples)
    for item in state.values():
        timestamps = item["token_timestamps_ns"]
        if not timestamps:
            raise RuntimeError(f"request {item['request_id']} emitted no token")
        finish_ns = item["finish_ns"] or group_end_ns
        itl_ms = [(b - a) / 1e6 for a, b in pairwise(timestamps)]
        sample = item["sample"]
        records.append(
            {
                "request_id": item["request_id"],
                "sample_id": sample["sample_id"],
                "benchmark": sample["benchmark"],
                "prompt_sha256": sample["prompt_sha256"],
                "prompt_length": sample["prompt_length"],
                "padded_prompt_length": padded_length,
                "padding_length": padded_length - sample["prompt_length"],
                "padding_materialized": False,
                "generation_limit": max_tokens,
                "generated_token_ids": item["generated_token_ids"],
                "generated_tokens": len(item["generated_token_ids"]),
                "finish_reason": item["finish_reason"],
                "arrival_ns": item["arrival_ns"],
                "first_token_ns": timestamps[0],
                "last_token_ns": timestamps[-1],
                "finish_ns": finish_ns,
                "token_timestamps_ns": timestamps,
                "ttft_ms": (timestamps[0] - item["arrival_ns"]) / 1e6,
                "itl_ms": itl_ms,
                "tpot_ms": statistics.mean(itl_ms) if itl_ms else "N/A",
                "e2e_ms": (finish_ns - item["arrival_ns"]) / 1e6,
            }
        )
    return records, group_start_ns, group_end_ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measurements", type=int, default=20)
    parser.add_argument("--power-interval-ms", type=float, default=20.0)
    parser.add_argument("--group-limit", type=int)
    args = parser.parse_args()

    if (args.output_dir / "latency_raw.jsonl").exists() or (args.output_dir / "power_raw.csv").exists():
        raise FileExistsError("use a new capture directory; raw measurements must not be overwritten")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    document = json.loads(args.samples.read_text(encoding="utf-8"))
    samples = document["samples"]
    groups = build_groups(samples)
    if args.group_limit is not None:
        groups = groups[: args.group_limit]

    sampler = NvmlSampler(args.physical_gpu, args.output_dir / "power_raw.csv", args.power_interval_ms / 1000.0)
    raw_path = args.output_dir / "latency_raw.jsonl"
    trace_path = args.output_dir / "token_traces_timing.jsonl"
    first_long_trace: dict[str, dict[str, Any]] = {}
    measurement_records: list[dict[str, Any]] = []
    try:
        _bind_cuda_device(sampler.uuid)
        import torch
        from vllm import LLM

        cuda_gpu_uuid = _verify_cuda_device(torch, sampler.uuid)
        llm = LLM(
            model=str(args.model),
            trust_remote_code=True,
            skip_tokenizer_init=True,
            tensor_parallel_size=1,
            dtype="auto",
            seed=1234,
            enforce_eager=False,
            max_model_len=8192,
            max_num_batched_tokens=131072,
            max_num_seqs=16,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            kv_cache_memory_bytes=32 * 1024**3,
            swap_space=0,
            disable_log_stats=False,
        )
        torch.cuda.synchronize()
        _verify_cuda_device(torch, sampler.uuid)
        sampler.start()
        with raw_path.open("w", encoding="utf-8") as raw:
            for group_number, group in enumerate(groups):
                for iteration in range(args.warmup + args.measurements):
                    phase = "warmup" if iteration < args.warmup else "measurement"
                    measurement = iteration if phase == "warmup" else iteration - args.warmup
                    trial_id = (
                        f"{group['benchmark']}:{group['mode']}:B{group['batch_size']}:"
                        f"G{group['group_index']}:{phase}:{measurement}"
                    )
                    context = {
                        "trial_id": trial_id,
                        "phase": phase,
                        "benchmark": group["benchmark"],
                        "mode": group["mode"],
                        "batch_size": group["batch_size"],
                        "group_index": group["group_index"],
                        "measurement": measurement,
                    }
                    sampler.set_context(**context)
                    records, group_start_ns, group_end_ns = run_batch(llm, group["samples"], group["generation_limit"])
                    torch.cuda.synchronize()
                    power = sampler.trial_stats(trial_id, group_start_ns, group_end_ns)
                    total_tokens = sum(x["generated_tokens"] for x in records)
                    batch_e2e_s = (group_end_ns - group_start_ns) / 1e9
                    batch_throughput = total_tokens / batch_e2e_s
                    energy = power["energy_joules"]
                    tokens_per_joule = total_tokens / energy if isinstance(energy, float) and energy > 0 else "N/A"
                    for record in records:
                        record.update(context)
                        record.update(
                            {
                                "record_type": "request",
                                "group_number": group_number,
                                "include_in_summary": phase == "measurement",
                                "batch_total_generated_tokens": total_tokens,
                                "batch_e2e_ms": batch_e2e_s * 1000.0,
                                "group_start_ns": group_start_ns,
                                "group_end_ns": group_end_ns,
                                "batch_throughput_tokens_per_s": batch_throughput,
                                "batch_energy_joules": energy,
                                "batch_tokens_per_joule": tokens_per_joule,
                                "request_attributed_joules_equal_share": (
                                    energy / group["batch_size"] if isinstance(energy, float) else "N/A"
                                ),
                                **power,
                            }
                        )
                        raw.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if phase == "measurement":
                            measurement_records.append(record)
                            if group["mode"] == "b1_long" and record["sample_id"] not in first_long_trace:
                                first_long_trace[record["sample_id"]] = {
                                    "benchmark": record["benchmark"],
                                    "sample_id": record["sample_id"],
                                    "prompt_sha256": record["prompt_sha256"],
                                    "generated_token_ids": record["generated_token_ids"],
                                    "generated_tokens": record["generated_tokens"],
                                    "source": "timing_b1_long_measurement_0",
                                }
                    raw.flush()
                    if sampler.unexpected_pids:
                        raise RuntimeError(
                            f"unexpected compute PIDs on selected GPU: {sorted(sampler.unexpected_pids)}"
                        )
    finally:
        sampler.set_context(trial_id="idle", phase="idle")
        sampler.stop()
    if sampler.worker_error is not None:
        raise RuntimeError("NVML sampling worker failed") from sampler.worker_error

    with trace_path.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(first_long_trace):
            handle.write(json.dumps(first_long_trace[sample_id], ensure_ascii=False) + "\n")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in measurement_records:
        key = (
            record["benchmark"],
            record["mode"],
            record["batch_size"],
            record["generation_limit"],
            record["group_index"],
        )
        grouped[key].append(record)
    summary_path = args.output_dir / "latency_summary.csv"
    fields = [
        "benchmark",
        "mode",
        "batch_size",
        "generation_limit",
        "group_index",
        "sample_ids",
        "prompt_length_min",
        "prompt_length_max",
        "padded_prompt_length",
        "warmup",
        "measurements",
        "request_measurement_count",
        "generated_tokens_total",
        "ttft_ms_median",
        "ttft_ms_p95",
        "itl_ms_median",
        "itl_ms_p95",
        "tpot_ms_median",
        "tpot_ms_p95",
        "e2e_ms_median",
        "e2e_ms_p95",
        "batch_throughput_tokens_per_s_median",
        "batch_throughput_tokens_per_s_p95",
        "gpu_memory_used_peak_bytes",
        "batch_energy_joules_median",
        "batch_energy_joules_p95",
        "batch_tokens_per_joule_median",
        "power_status",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            records = grouped[key]
            ttft = [x["ttft_ms"] for x in records]
            itl = [y for x in records for y in x["itl_ms"]]
            tpot = [x["tpot_ms"] for x in records if isinstance(x["tpot_ms"], float)]
            e2e = [x["e2e_ms"] for x in records]
            # Batch metrics are duplicated across requests; retain one per measurement.
            trials = {x["trial_id"]: x for x in records}.values()
            throughput = [x["batch_throughput_tokens_per_s"] for x in trials]
            energy = [x["batch_energy_joules"] for x in trials if isinstance(x["batch_energy_joules"], float)]
            efficiency = [x["batch_tokens_per_joule"] for x in trials if isinstance(x["batch_tokens_per_joule"], float)]
            writer.writerow(
                {
                    "benchmark": key[0],
                    "mode": key[1],
                    "batch_size": key[2],
                    "generation_limit": key[3],
                    "group_index": key[4],
                    "sample_ids": json.dumps(sorted({x["sample_id"] for x in records})),
                    "prompt_length_min": min(x["prompt_length"] for x in records),
                    "prompt_length_max": max(x["prompt_length"] for x in records),
                    "padded_prompt_length": max(x["padded_prompt_length"] for x in records),
                    "warmup": args.warmup,
                    "measurements": args.measurements,
                    "request_measurement_count": len(records),
                    "generated_tokens_total": sum(x["generated_tokens"] for x in records),
                    "ttft_ms_median": statistics.median(ttft),
                    "ttft_ms_p95": percentile(ttft, 95),
                    "itl_ms_median": statistics.median(itl) if itl else "N/A",
                    "itl_ms_p95": percentile(itl, 95),
                    "tpot_ms_median": statistics.median(tpot) if tpot else "N/A",
                    "tpot_ms_p95": percentile(tpot, 95),
                    "e2e_ms_median": statistics.median(e2e),
                    "e2e_ms_p95": percentile(e2e, 95),
                    "batch_throughput_tokens_per_s_median": statistics.median(throughput),
                    "batch_throughput_tokens_per_s_p95": percentile(throughput, 95),
                    "gpu_memory_used_peak_bytes": max(
                        (
                            x["gpu_memory_used_peak_bytes"]
                            for x in records
                            if isinstance(x["gpu_memory_used_peak_bytes"], int)
                        ),
                        default="N/A",
                    ),
                    "batch_energy_joules_median": statistics.median(energy) if energy else "N/A",
                    "batch_energy_joules_p95": percentile(energy, 95),
                    "batch_tokens_per_joule_median": statistics.median(efficiency) if efficiency else "N/A",
                    "power_status": "supported" if sampler.power_supported else "N/A",
                }
            )

    metadata = {
        "model": str(args.model),
        "model_revision": args.model.name,
        "physical_gpu_index": args.physical_gpu,
        "gpu_uuid": sampler.uuid,
        "cuda_gpu_uuid": cuda_gpu_uuid,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_nvml_uuid_verified": True,
        "single_model_instance": True,
        "tensor_parallel_size": 1,
        "optimized_serving": True,
        "enforce_eager": False,
        "routing_hooks_installed": False,
        "temperature": 0.0,
        "greedy": True,
        "ignore_eos": False,
        "warmup": args.warmup,
        "measurements": args.measurements,
        "power_interval_ms": args.power_interval_ms,
        "power_supported": sampler.power_supported,
        "power_error": sampler.power_error or "N/A",
        "unexpected_compute_pids": sorted(sampler.unexpected_pids),
        "group_count": len(groups),
        "measurement_request_records": len(measurement_records),
        "padding_note": "vLLM continuous batching does not materialize padding; padding_length is the equivalent max-length delta within each group",
        "energy_method": "sorted_interpolated_recorded_batch_window_v2",
        "power_timestamp": "midpoint of actual NVML power read; bracket retained per sample",
        "energy_attribution_note": "B1 request joules are direct trial energy; B>1 per-request joules are equal-share attribution and batch joules are authoritative",
    }
    (args.output_dir / "timing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
