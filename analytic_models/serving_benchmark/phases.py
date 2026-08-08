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

        first = max(float(request.first_token_s) for request in self.requests.values())
        second = (
            max(float(request.second_token_s) for request in self.requests.values())
            if self.expected_output_tokens >= 2
            else first
        )
        finish = max(float(request.finish_s) for request in self.requests.values())
        per_request_first_decode = [
            (float(request.second_token_s) - float(request.first_token_s))
            if self.expected_output_tokens >= 2
            else 0.0
            for request in self.requests.values()
        ]
        per_request_generation = [
            float(request.finish_s) - float(request.first_token_s) for request in self.requests.values()
        ]
        first_decode_iteration_s = max(per_request_first_decode)
        measured_generation_s = max(per_request_generation)
        imported_proxy_s = max(
            first_decode + generation
            for first_decode, generation in zip(
                per_request_first_decode, per_request_generation, strict=True
            )
        )
        measured_decode_steps = max(1, self.expected_output_tokens - 1)
        token_intervals = [
            right - left
            for request in self.requests.values()
            for left, right in itertools.pairwise(request.token_timestamps_s)
        ]
        sorted_intervals = sorted(token_intervals)
        p95_index = max(0, min(len(sorted_intervals) - 1, math.ceil(0.95 * len(sorted_intervals)) - 1))
        stage_sum = first - request_start_s + measured_generation_s
        full_latency = finish - request_start_s
        return {
            "request_start_s": request_start_s,
            "prefill_complete_s": first,
            "first_decode_complete_s": second,
            "request_complete_s": finish,
            "prefill_latency_s": first - request_start_s,
            "first_decode_iteration_latency_s": first_decode_iteration_s,
            "measured_generation_latency_s": measured_generation_s,
            "full_request_latency_s": full_latency,
            "stage_sum_latency_s": stage_sum,
            "stage_reconstruction_error_pct": abs(stage_sum - full_latency) / full_latency * 100.0,
            "imported_kv_decode_proxy_latency_s": imported_proxy_s,
            "output_tokens_per_request": self.expected_output_tokens,
            "global_output_tokens": self.expected_output_tokens * len(self.requests),
            "mean_tpot_s": measured_generation_s / measured_decode_steps,
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
            "multi_token_step_observed": any(
                request.multi_token_step_observed for request in self.requests.values()
            ),
            "requests": {
                request_id: {
                    "first_token_s": request.first_token_s,
                    "second_token_s": request.second_token_s,
                    "finish_s": request.finish_s,
                    "output_tokens": request.output_tokens,
                    "multi_token_step_observed": request.multi_token_step_observed,
                }
                for request_id, request in self.requests.items()
            },
        }

    def token_timestamp_rows(self) -> list[tuple[str, int, float]]:
        return [
            (request_id, token_index, timestamp)
            for request_id, request in self.requests.items()
            for token_index, timestamp in enumerate(request.token_timestamps_s, start=1)
        ]
