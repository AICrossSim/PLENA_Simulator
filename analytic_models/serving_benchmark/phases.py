"""Version-independent extraction of prefill and decode phase boundaries."""

from __future__ import annotations

import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestPhase:
    first_token_s: float | None = None
    second_token_s: float | None = None
    finish_s: float | None = None
    output_tokens: int = 0
    multi_token_step_observed: bool = False
    token_timestamps_s: list[float] = field(default_factory=list)
    arrival_time_s: float | None = None
    first_scheduled_time_s: float | None = None
    engine_first_token_time_s: float | None = None
    engine_finished_time_s: float | None = None
    time_in_queue_s: float | None = None


def _nearest_rank_p95(values: list[float]) -> float:
    index = max(0, min(len(values) - 1, math.ceil(0.95 * len(values)) - 1))
    return sorted(values)[index]


def _phase_first_token_offsets(phase: dict[str, Any]) -> list[float]:
    offsets = phase.get("first_token_completion_offsets_s")
    if offsets is not None:
        return sorted(float(value) for value in offsets)

    request_ttft = phase.get("request_ttft_s")
    if isinstance(request_ttft, dict) and request_ttft:
        return sorted(float(value) for value in request_ttft.values())

    request_start = phase.get("request_start_s")
    requests = phase.get("requests")
    if request_start is None or not isinstance(requests, dict):
        return []
    first_tokens = [request.get("first_token_s") for request in requests.values() if isinstance(request, dict)]
    if not first_tokens or any(value is None for value in first_tokens):
        return []
    return sorted(float(value) - float(request_start) for value in first_tokens)


def enrich_phase_ttft_semantics(
    phase: dict[str, Any],
    *,
    batch_size: int,
    uniform_prompt_shape: bool,
    no_preemption: bool,
) -> dict[str, Any]:
    """Upgrade current or legacy phase data without mutating its raw artifact."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    enriched = dict(phase)
    enriched["phase_schema_version"] = "request-visible-v4"

    barrier = enriched.get("batch_first_token_barrier_latency_s")
    if barrier is None:
        barrier = enriched.get("prefill_latency_s")
    barrier = float(barrier) if barrier is not None else None
    enriched["batch_first_token_barrier_latency_s"] = barrier
    enriched["throughput_equivalent_prefill_interval_s"] = barrier / batch_size if barrier is not None else None
    enriched["throughput_equivalent_prefill_interval_semantics"] = "batch_first_token_barrier_divided_by_batch_not_ttft"

    offsets = _phase_first_token_offsets(enriched)
    if offsets:
        spacings = [right - left for left, right in itertools.pairwise(offsets)]
        enriched.setdefault("first_token_completion_offsets_s", offsets)
        enriched.setdefault("first_token_completion_spacing_s", spacings)
        enriched.setdefault("earliest_request_ttft_s", offsets[0])
        enriched.setdefault("mean_request_ttft_s", statistics.fmean(offsets))
        enriched.setdefault("median_request_ttft_s", statistics.median(offsets))
        enriched.setdefault("p95_request_ttft_s", _nearest_rank_p95(offsets))
        enriched.setdefault("latest_request_ttft_s", offsets[-1])
        enriched.setdefault("first_token_staircase_span_s", offsets[-1] - offsets[0])
        enriched.setdefault(
            "batch_barrier_over_earliest_ttft",
            barrier / offsets[0] if barrier is not None and offsets[0] > 0 else None,
        )
        request_ttft = enriched.get("request_ttft_s")
        if isinstance(request_ttft, dict) and request_ttft:
            enriched.setdefault(
                "first_submitted_request_ttft_s",
                float(next(iter(request_ttft.values()))),
            )

    exact_samples = enriched.get("scheduler_admitted_request_ttft_s")
    exact_values = (
        [float(value) for value in exact_samples.values()] if isinstance(exact_samples, dict) and exact_samples else []
    )
    if exact_values:
        enriched["mean_scheduler_admitted_ttft_s"] = statistics.fmean(exact_values)
        enriched["median_scheduler_admitted_ttft_s"] = statistics.median(exact_values)
        enriched["p95_scheduler_admitted_ttft_s"] = _nearest_rank_p95(exact_values)
    else:
        enriched["scheduler_admitted_request_ttft_s"] = None
        enriched["mean_scheduler_admitted_ttft_s"] = None
        enriched["median_scheduler_admitted_ttft_s"] = None
        enriched["p95_scheduler_admitted_ttft_s"] = None

    candidate_samples = [offsets[0], *[right - left for left, right in itertools.pairwise(offsets)]] if offsets else []
    cv_pct = (
        0.0
        if candidate_samples and statistics.fmean(candidate_samples) == 0
        else statistics.pstdev(candidate_samples) / statistics.fmean(candidate_samples) * 100.0
        if candidate_samples
        else None
    )
    one_request_per_timestamp = len(offsets) == batch_size and len(set(offsets)) == batch_size
    ratio = barrier / (offsets[0] * batch_size) if barrier is not None and offsets and offsets[0] > 0 else None
    checks = {
        "completion_count_matches_batch": len(offsets) == batch_size,
        "one_request_per_completion_timestamp": one_request_per_timestamp,
        "uniform_prompt_shape": bool(uniform_prompt_shape),
        "no_preemption": bool(no_preemption),
        "completion_interval_cv_within_5pct": cv_pct is not None and cv_pct <= 5.0,
        "barrier_ratio_within_5pct": ratio is not None and abs(ratio - 1.0) <= 0.05,
    }
    inference_valid = all(checks.values())
    reasons = [name for name, passed in checks.items() if not passed]
    enriched["serial_staircase_cv_pct"] = cv_pct
    enriched["serial_staircase_validation"] = {
        "status": "pass" if inference_valid else "rejected",
        "checks": checks,
        "rejection_reasons": reasons,
        "batch_barrier_over_earliest_times_batch": ratio,
    }
    enriched["inferred_scheduler_admitted_ttft_samples_s"] = candidate_samples if inference_valid else None
    enriched["inferred_mean_scheduler_admitted_ttft_s"] = (
        statistics.fmean(candidate_samples) if inference_valid else None
    )

    if exact_values:
        source = "measured_vllm_first_scheduled_to_first_token"
        fidelity = "exact_vllm_request_metrics"
        best_available = statistics.fmean(exact_values)
        enriched["internal_phase_marker_fidelity"] = "vllm_request_metrics_admission_marker_kv_ready_unavailable"
    elif inference_valid:
        source = "inferred_serial_staircase"
        fidelity = "proxy_from_serial_first_token_completion_staircase"
        best_available = statistics.fmean(candidate_samples)
    else:
        source = "unavailable"
        fidelity = "unavailable_no_valid_admission_marker_or_staircase"
        best_available = None
    enriched["scheduler_admitted_ttft_source"] = source
    enriched["scheduler_admitted_ttft_fidelity"] = fidelity
    enriched["scheduler_admitted_ttft_best_available_s"] = best_available

    request_tpot = enriched.get("request_tpot_s")
    request_tpot_values = (
        [float(value) for value in request_tpot.values()] if isinstance(request_tpot, dict) and request_tpot else []
    )
    if not request_tpot_values:
        requests = enriched.get("requests")
        if isinstance(requests, dict):
            derived: dict[str, float] = {}
            for request_id, request in requests.items():
                if not isinstance(request, dict):
                    continue
                output_tokens = int(
                    request.get(
                        "output_tokens",
                        enriched.get("output_tokens_per_request", 0),
                    )
                )
                first_token = request.get("first_token_s")
                finish = request.get("finish_s")
                if output_tokens <= 1 or first_token is None or finish is None:
                    continue
                derived[str(request_id)] = (float(finish) - float(first_token)) / (output_tokens - 1)
            if derived:
                enriched["request_tpot_s"] = derived
                request_tpot_values = list(derived.values())

    if request_tpot_values:
        enriched["mean_request_tpot_s"] = statistics.fmean(request_tpot_values)
        enriched["median_request_tpot_s"] = statistics.median(request_tpot_values)
        enriched["p95_request_tpot_s"] = _nearest_rank_p95(request_tpot_values)
        # Compatibility alias. Older artifacts used this name for the slowest
        # request generation makespan divided by the number of decode steps.
        enriched["mean_tpot_s"] = enriched["mean_request_tpot_s"]
        enriched["mean_tpot_legacy_alias_semantics"] = "mean_request_tpot_s"
    else:
        enriched["request_tpot_s"] = None
        enriched["mean_request_tpot_s"] = None
        enriched["median_request_tpot_s"] = None
        enriched["p95_request_tpot_s"] = None
        enriched["mean_tpot_s"] = None
        enriched["mean_tpot_legacy_alias_semantics"] = "unavailable"

    request_tbt = enriched.get("request_tbt_samples_s")
    request_tbt_values = (
        [float(value) for values in request_tbt.values() for value in values] if isinstance(request_tbt, dict) else []
    )
    if request_tbt_values:
        enriched["median_tbt_s"] = statistics.median(request_tbt_values)
        enriched["p95_tbt_s"] = _nearest_rank_p95(request_tbt_values)
    else:
        enriched.setdefault("request_tbt_samples_s", None)
    return enriched


@dataclass
class PhaseTracker:
    request_ids: tuple[str, ...]
    expected_output_tokens: int
    requests: dict[str, RequestPhase] = field(init=False)

    def __post_init__(self) -> None:
        self.requests = {request_id: RequestPhase() for request_id in self.request_ids}

    def observe(self, request_id: str, *, cumulative_tokens: int, finished: bool, timestamp_s: float) -> None:
        request = self.requests[request_id]
        if cumulative_tokens < request.output_tokens:
            raise ValueError(f"output token count regressed for {request_id}")
        delta = cumulative_tokens - request.output_tokens
        if delta > 1:
            request.multi_token_step_observed = True
        if delta == 1:
            request.token_timestamps_s.append(timestamp_s)
        if request.first_token_s is None and cumulative_tokens >= 1:
            request.first_token_s = timestamp_s
        if request.second_token_s is None and cumulative_tokens >= 2:
            request.second_token_s = timestamp_s
        request.output_tokens = cumulative_tokens
        if finished:
            request.finish_s = timestamp_s

    def observe_scheduler_metrics(self, request_id: str, metrics: Any) -> None:
        """Capture optional vLLM RequestMetrics fields for exact admission timing."""

        request = self.requests[request_id]
        fields = {
            "arrival_time_s": "arrival_time",
            "first_scheduled_time_s": "first_scheduled_time",
            "engine_first_token_time_s": "first_token_time",
            "engine_finished_time_s": "finished_time",
            "time_in_queue_s": "time_in_queue",
        }
        for destination, source in fields.items():
            value = getattr(metrics, source, None)
            if value is not None:
                setattr(request, destination, float(value))

    @staticmethod
    def _validate_scheduler_times(request_id: str, request: RequestPhase) -> None:
        ordered = (
            request.arrival_time_s,
            request.first_scheduled_time_s,
            request.engine_first_token_time_s,
            request.engine_finished_time_s,
        )
        present = [value for value in ordered if value is not None]
        if any(right < left for left, right in itertools.pairwise(present)):
            raise ValueError(f"vLLM RequestMetrics timestamps are out of order for {request_id}")
        if request.time_in_queue_s is not None and request.time_in_queue_s < 0:
            raise ValueError(f"vLLM reported a negative queue time for {request_id}")

    @property
    def all_have_first_token(self) -> bool:
        return all(request.first_token_s is not None for request in self.requests.values())

    @property
    def all_have_second_token(self) -> bool:
        return all(request.second_token_s is not None for request in self.requests.values())

    @property
    def all_finished(self) -> bool:
        return all(request.finish_s is not None for request in self.requests.values())

    def summary(self, *, request_start_s: float) -> dict[str, Any]:
        if not self.all_finished or not self.all_have_first_token:
            raise ValueError("cannot summarize an incomplete request group")
        if self.expected_output_tokens >= 2 and not self.all_have_second_token:
            raise ValueError("missing the first normal decode iteration")
        if any(request.output_tokens != self.expected_output_tokens for request in self.requests.values()):
            raise ValueError("one or more requests produced an unexpected output token count")
        for request_id, request in self.requests.items():
            self._validate_scheduler_times(request_id, request)

        request_ttft = {
            request_id: float(request.first_token_s) - request_start_s for request_id, request in self.requests.items()
        }
        sorted_request_ttft = sorted(request_ttft.values())
        sorted_first_token_s = sorted(float(request.first_token_s) for request in self.requests.values())
        first_token_completion_spacing_s = [right - left for left, right in itertools.pairwise(sorted_first_token_s)]
        exact_scheduler_ttft = {
            request_id: float(request.engine_first_token_time_s) - float(request.first_scheduled_time_s)
            for request_id, request in self.requests.items()
            if request.engine_first_token_time_s is not None and request.first_scheduled_time_s is not None
        }
        if len(exact_scheduler_ttft) != len(self.requests):
            exact_scheduler_ttft = {}
        first = max(float(request.first_token_s) for request in self.requests.values())
        second = (
            max(float(request.second_token_s) for request in self.requests.values())
            if self.expected_output_tokens >= 2
            else first
        )
        finish = max(float(request.finish_s) for request in self.requests.values())
        per_request_first_decode = [
            (float(request.second_token_s) - float(request.first_token_s)) if self.expected_output_tokens >= 2 else 0.0
            for request in self.requests.values()
        ]
        per_request_generation = [
            float(request.finish_s) - float(request.first_token_s) for request in self.requests.values()
        ]
        request_tpot = (
            {
                request_id: (float(request.finish_s) - float(request.first_token_s)) / (self.expected_output_tokens - 1)
                for request_id, request in self.requests.items()
            }
            if self.expected_output_tokens > 1
            else {}
        )
        request_tbt_samples = (
            {
                request_id: [right - left for left, right in itertools.pairwise(request.token_timestamps_s)]
                for request_id, request in self.requests.items()
            }
            if self.expected_output_tokens > 1
            else {}
        )
        first_decode_iteration_s = max(per_request_first_decode)
        measured_generation_s = max(per_request_generation)
        post_prefill_tail_s = finish - first
        post_prefill_generated_tokens = sum(
            1 for request in self.requests.values() for timestamp in request.token_timestamps_s if timestamp > first
        )
        post_prefill_rate = (
            post_prefill_generated_tokens / post_prefill_tail_s
            if post_prefill_tail_s > 0 and post_prefill_generated_tokens > 0
            else None
        )
        measured_decode_steps = max(1, self.expected_output_tokens - 1)
        proxy_remaining_tokens = measured_decode_steps * len(self.requests)
        imported_proxy_s = (
            first_decode_iteration_s + proxy_remaining_tokens / post_prefill_rate
            if post_prefill_rate is not None
            else None
        )
        token_intervals = [
            right - left
            for request in self.requests.values()
            for left, right in itertools.pairwise(request.token_timestamps_s)
        ]
        sorted_intervals = sorted(token_intervals)
        p95_index = max(0, min(len(sorted_intervals) - 1, math.ceil(0.95 * len(sorted_intervals)) - 1))
        stage_sum = first - request_start_s + measured_generation_s
        full_latency = finish - request_start_s
        overlap_s = max(0.0, stage_sum - full_latency)
        batch_first_token_barrier_latency_s = first - request_start_s
        summary = {
            "phase_schema_version": "request-visible-v4",
            "request_start_s": request_start_s,
            "batch_first_token_barrier_s": first,
            "batch_first_token_barrier_latency_s": batch_first_token_barrier_latency_s,
            # Backward-compatible aliases. These are scheduler-visible batch
            # barriers, not GPU-admitted prefill or KV-ready timestamps.
            "prefill_complete_s": first,
            "first_decode_complete_s": second,
            "request_complete_s": finish,
            "prefill_latency_s": batch_first_token_barrier_latency_s,
            "prefill_legacy_alias_semantics": "batch_first_token_barrier",
            "earliest_request_ttft_s": min(sorted_request_ttft),
            "first_submitted_request_ttft_s": request_ttft[self.request_ids[0]],
            "mean_request_ttft_s": statistics.fmean(sorted_request_ttft),
            "median_request_ttft_s": statistics.median(sorted_request_ttft),
            "p95_request_ttft_s": _nearest_rank_p95(sorted_request_ttft),
            "latest_request_ttft_s": max(sorted_request_ttft),
            "request_ttft_s": request_ttft,
            "first_token_completion_offsets_s": [timestamp - request_start_s for timestamp in sorted_first_token_s],
            "first_token_completion_spacing_s": first_token_completion_spacing_s,
            "first_token_staircase_span_s": (max(sorted_request_ttft) - min(sorted_request_ttft)),
            "batch_barrier_over_earliest_ttft": (
                batch_first_token_barrier_latency_s / min(sorted_request_ttft) if min(sorted_request_ttft) > 0 else None
            ),
            "scheduler_admission_time_s": None,
            "gpu_admitted_prefill_service_time_s": None,
            "kv_ready_time_s": None,
            "internal_phase_marker_fidelity": "unavailable_no_internal_vllm_marker",
            "scheduler_admitted_request_ttft_s": exact_scheduler_ttft or None,
            "first_decode_iteration_latency_s": first_decode_iteration_s,
            "measured_generation_latency_s": measured_generation_s,
            "full_request_latency_s": full_latency,
            "stage_sum_latency_s": stage_sum,
            "stage_reconstruction_error_pct": abs(stage_sum - full_latency) / full_latency * 100.0,
            "prefill_decode_overlap_s": overlap_s,
            "overlap_adjusted_stage_sum_latency_s": stage_sum - overlap_s,
            "overlap_adjusted_reconstruction_error_pct": (
                abs(stage_sum - overlap_s - full_latency) / full_latency * 100.0
            ),
            "imported_kv_decode_proxy_latency_s": imported_proxy_s,
            "decode_proxy_fidelity": (
                "post_global_prefill_tail_extrapolation_v1"
                if imported_proxy_s is not None
                else "unavailable_insufficient_decode_tail"
            ),
            "post_global_prefill_tail_latency_s": post_prefill_tail_s,
            "post_global_prefill_generated_tokens": post_prefill_generated_tokens,
            "post_global_prefill_output_tokens_per_s": post_prefill_rate,
            "output_tokens_per_request": self.expected_output_tokens,
            "global_output_tokens": self.expected_output_tokens * len(self.requests),
            "request_tpot_s": request_tpot or None,
            "request_tbt_samples_s": request_tbt_samples or None,
            "mean_request_tpot_s": (statistics.fmean(request_tpot.values()) if request_tpot else None),
            "median_request_tpot_s": (statistics.median(request_tpot.values()) if request_tpot else None),
            "p95_request_tpot_s": (_nearest_rank_p95(list(request_tpot.values())) if request_tpot else None),
            "mean_tpot_s": (statistics.fmean(request_tpot.values()) if request_tpot else None),
            "mean_tpot_legacy_alias_semantics": ("mean_request_tpot_s" if request_tpot else "unavailable"),
            "median_tbt_s": statistics.median(token_intervals) if token_intervals else None,
            "p95_tbt_s": sorted_intervals[p95_index] if sorted_intervals else None,
            "decode_output_tokens_per_s": (
                measured_decode_steps * len(self.requests) / measured_generation_s
                if measured_generation_s > 0
                else None
            ),
            "full_request_output_tokens_per_s": (
                self.expected_output_tokens * len(self.requests) / (finish - request_start_s)
                if finish > request_start_s
                else None
            ),
            "multi_token_step_observed": any(request.multi_token_step_observed for request in self.requests.values()),
            "requests": {
                request_id: {
                    "first_token_s": request.first_token_s,
                    "second_token_s": request.second_token_s,
                    "finish_s": request.finish_s,
                    "output_tokens": request.output_tokens,
                    "multi_token_step_observed": request.multi_token_step_observed,
                    "arrival_time_s": request.arrival_time_s,
                    "first_scheduled_time_s": request.first_scheduled_time_s,
                    "engine_first_token_time_s": request.engine_first_token_time_s,
                    "engine_finished_time_s": request.engine_finished_time_s,
                    "time_in_queue_s": request.time_in_queue_s,
                    "queue_time_s": request.time_in_queue_s,
                    "request_visible_ttft_from_vllm_metrics_s": (
                        request.engine_first_token_time_s - request.arrival_time_s
                        if request.engine_first_token_time_s is not None and request.arrival_time_s is not None
                        else None
                    ),
                    "scheduler_admitted_ttft_s": (
                        request.engine_first_token_time_s - request.first_scheduled_time_s
                        if request.engine_first_token_time_s is not None and request.first_scheduled_time_s is not None
                        else None
                    ),
                }
                for request_id, request in self.requests.items()
            },
        }
        return enrich_phase_ttft_semantics(
            summary,
            batch_size=len(self.requests),
            uniform_prompt_shape=False,
            no_preemption=False,
        )

    def token_timestamp_rows(self) -> list[tuple[str, int, float]]:
        return [
            (request_id, token_index, timestamp)
            for request_id, request in self.requests.items()
            for token_index, timestamp in enumerate(request.token_timestamps_s, start=1)
        ]
