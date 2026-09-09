"""Deterministic queueing model for disaggregated prefill/decode serving."""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .handoff import HANDOFF_REGIMES, HandoffTime
except ImportError:  # pragma: no cover - direct script execution
    from handoff import HANDOFF_REGIMES, HandoffTime


SCHEMA = "plena-disaggregated-serving-queue/v1"


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True)
class ServingRequest:
    """One request with service costs supplied by measured or analytic models."""

    request_id: str
    arrival_s: float
    prompt_tokens: int
    generation_tokens: int
    prefill_s: float
    decode_tpot_s: float
    transfer_streamed_s: float
    transfer_bulk_s: float
    host_write_s: float
    host_read_s: float
    admission_s: float
    kv_buffer_bytes: int
    admission_bytes: int
    prefill_energy_j: float
    decode_energy_per_token_j: float
    handoff_energy_j: float
    workload_bucket: str = "default"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id is required")
        if self.prompt_tokens <= 0 or self.generation_tokens < 2:
            raise ValueError(
                "prompt_tokens must be positive and generation_tokens must include "
                "one prefill token plus at least one decode token"
            )
        if self.kv_buffer_bytes <= 0 or self.admission_bytes <= 0:
            raise ValueError("KV and admission byte counts must be positive")
        for name in (
            "arrival_s",
            "prefill_s",
            "decode_tpot_s",
            "transfer_streamed_s",
            "transfer_bulk_s",
            "host_write_s",
            "host_read_s",
            "admission_s",
            "prefill_energy_j",
            "decode_energy_per_token_j",
            "handoff_energy_j",
        ):
            _positive(
                getattr(self, name),
                name,
                allow_zero=name
                in {
                    "arrival_s",
                    "prefill_energy_j",
                    "decode_energy_per_token_j",
                    "handoff_energy_j",
                },
            )
        if not self.workload_bucket:
            raise ValueError("workload_bucket is required")

    @classmethod
    def from_handoff(
        cls,
        *,
        request_id: str,
        arrival_s: float,
        prompt_tokens: int,
        generation_tokens: int,
        prefill_s: float,
        decode_tpot_s: float,
        handoff: HandoffTime,
        host_bandwidth_bytes_per_s: float,
        prefill_energy_j: float,
        decode_energy_per_token_j: float,
        link_energy_j: float,
        workload_bucket: str = "default",
    ) -> "ServingRequest":
        host_bw = _positive(host_bandwidth_bytes_per_s, "host bandwidth")
        return cls(
            request_id=request_id,
            arrival_s=arrival_s,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            prefill_s=prefill_s,
            decode_tpot_s=decode_tpot_s,
            transfer_streamed_s=handoff.transfer_streamed_s,
            transfer_bulk_s=handoff.transfer_bulk_s,
            host_write_s=handoff.wire_bytes / host_bw,
            host_read_s=handoff.wire_bytes / host_bw,
            admission_s=handoff.admission_s,
            kv_buffer_bytes=math.ceil(handoff.wire_bytes),
            admission_bytes=math.ceil(
                handoff.wire_bytes + handoff.decode_cache_bytes
            ),
            prefill_energy_j=prefill_energy_j,
            decode_energy_per_token_j=decode_energy_per_token_j,
            handoff_energy_j=handoff.admission_energy_j + link_energy_j,
            workload_bucket=workload_bucket,
        )


@dataclass(frozen=True)
class ServingQueueConfig:
    regime: str
    prefill_replicas: int
    decode_replicas: int
    link_ports: int
    kv_buffer_capacity_bytes: int
    max_inflight_requests: int
    ttft_slo_s: float
    tpot_slo_s: float
    e2e_slo_s: float | None = None
    prefill_idle_power_w: float = 0.0
    decode_idle_power_w: float = 0.0
    link_idle_power_w: float = 0.0

    def __post_init__(self) -> None:
        if self.regime not in HANDOFF_REGIMES:
            raise ValueError(f"unsupported handoff regime {self.regime!r}")
        for name in (
            "prefill_replicas",
            "decode_replicas",
            "link_ports",
            "max_inflight_requests",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.kv_buffer_capacity_bytes < 0:
            raise ValueError("kv_buffer_capacity_bytes must be non-negative")
        if self.regime != "back_pressure" and self.kv_buffer_capacity_bytes <= 0:
            raise ValueError("buffered regimes require positive KV capacity")
        for name in (
            "ttft_slo_s",
            "tpot_slo_s",
            "prefill_idle_power_w",
            "decode_idle_power_w",
            "link_idle_power_w",
        ):
            _positive(
                getattr(self, name),
                name,
                allow_zero=name.endswith("power_w"),
            )
        if self.e2e_slo_s is not None:
            _positive(self.e2e_slo_s, "e2e_slo_s")


def poisson_arrival_times(
    *, rate_per_s: float, count: int, seed: int, start_s: float = 0.0
) -> tuple[float, ...]:
    """Generate a reproducible Poisson arrival trace."""

    rate = _positive(rate_per_s, "arrival rate")
    if isinstance(count, bool) or count <= 0:
        raise ValueError("arrival count must be positive")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    current = _positive(start_s, "start_s", allow_zero=True)
    generator = random.Random(seed)
    values = []
    for _ in range(count):
        current += generator.expovariate(rate)
        values.append(current)
    return tuple(values)


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def simulate_serving_queue(
    requests: Sequence[ServingRequest], config: ServingQueueConfig
) -> dict[str, Any]:
    """Simulate FCFS prefill, handoff, admission, and dedicated decode service."""

    ordered = tuple(sorted(requests, key=lambda item: (item.arrival_s, item.request_id)))
    if not ordered:
        raise ValueError("at least one serving request is required")
    if len({request.request_id for request in ordered}) != len(ordered):
        raise ValueError("request IDs must be unique")

    by_id = {request.request_id: request for request in ordered}
    event_heap: list[tuple[float, int, int, str, tuple[Any, ...]]] = []
    event_sequence = 0

    def event(time_s: float, priority: int, kind: str, *payload: Any) -> None:
        nonlocal event_sequence
        heapq.heappush(
            event_heap, (float(time_s), priority, event_sequence, kind, payload)
        )
        event_sequence += 1

    for request in ordered:
        event(request.arrival_s, 4, "arrival", request.request_id)

    prefill_idle = list(range(config.prefill_replicas))
    decode_idle = list(range(config.decode_replicas))
    link_idle = list(range(config.link_ports))
    heapq.heapify(prefill_idle)
    heapq.heapify(decode_idle)
    heapq.heapify(link_idle)
    prefill_wait: collections.deque[str] = collections.deque()
    transfer_wait: collections.deque[str] = collections.deque()
    decode_ready: collections.deque[str] = collections.deque()
    blocked_prefill: collections.deque[tuple[str, int, float]] = collections.deque()
    records: dict[str, dict[str, Any]] = {}
    inflight = 0
    buffer_bytes = 0
    peak_buffer_bytes = 0
    prefill_busy_s = 0.0
    prefill_stall_s = 0.0
    decode_busy_s = 0.0
    link_busy_s = 0.0
    admission_bytes = 0
    peak_inflight = 0
    peak_prefill_queue = 0
    last_time = ordered[0].arrival_s

    def reject(request_id: str, time_s: float, reason: str) -> None:
        nonlocal inflight
        record = records[request_id]
        record.update(status="rejected", rejection_reason=reason, completion_s=time_s)
        inflight -= 1

    def reserve_buffer(request_id: str, time_s: float) -> bool:
        nonlocal buffer_bytes, peak_buffer_bytes
        request = by_id[request_id]
        required = request.kv_buffer_bytes
        if buffer_bytes + required > config.kv_buffer_capacity_bytes:
            reject(request_id, time_s, "kv_buffer_capacity")
            return False
        buffer_bytes += required
        peak_buffer_bytes = max(peak_buffer_bytes, buffer_bytes)
        records[request_id]["buffer_enter_s"] = time_s
        return True

    def start_prefill(time_s: float) -> bool:
        nonlocal prefill_busy_s
        changed = False
        while prefill_idle and prefill_wait:
            server = heapq.heappop(prefill_idle)
            request_id = prefill_wait.popleft()
            request = by_id[request_id]
            finish = time_s + request.prefill_s
            records[request_id].update(
                prefill_server=server,
                prefill_start_s=time_s,
                prefill_finish_compute_s=finish,
            )
            prefill_busy_s += request.prefill_s
            event(finish, 2, "prefill_done", server, request_id)
            changed = True
        return changed

    def start_transfer(time_s: float) -> bool:
        nonlocal link_busy_s
        changed = False
        while link_idle and transfer_wait:
            server = heapq.heappop(link_idle)
            request_id = transfer_wait.popleft()
            request = by_id[request_id]
            duration = (
                request.transfer_streamed_s
                if config.regime == "fully_pipelined"
                else request.host_write_s
            )
            records[request_id].update(
                link_server=server,
                transfer_start_s=time_s,
                transfer_finish_s=time_s + duration,
            )
            link_busy_s += duration
            event(time_s + duration, 1, "transfer_done", server, request_id)
            changed = True
        return changed

    def start_decode(time_s: float) -> bool:
        nonlocal buffer_bytes, decode_busy_s, admission_bytes, link_busy_s
        changed = False
        queue: Any = blocked_prefill if config.regime == "back_pressure" else decode_ready
        while decode_idle and queue and (
            config.regime not in {"back_pressure", "host_buffered"}
            or link_idle
        ):
            server = heapq.heappop(decode_idle)
            if config.regime == "back_pressure":
                request_id, prefill_server, blocked_at = queue.popleft()
                request = by_id[request_id]
                transfer = request.transfer_bulk_s
                link_server = heapq.heappop(link_idle)
                link_busy_s += transfer
                records[request_id].update(
                    link_server=link_server,
                    transfer_start_s=time_s,
                    transfer_finish_s=time_s + transfer,
                )
                event(
                    time_s + transfer,
                    0,
                    "back_transfer_done",
                    link_server,
                    prefill_server,
                    request_id,
                    blocked_at,
                )
            else:
                request_id = queue.popleft()
                request = by_id[request_id]
                transfer = 0.0
                if config.regime == "host_buffered":
                    transfer = request.host_read_s
                    link_server = heapq.heappop(link_idle)
                    link_busy_s += transfer
                    records[request_id].update(
                        host_read_link_server=link_server,
                        host_read_start_s=time_s,
                        host_read_finish_s=time_s + transfer,
                    )
                    event(
                        time_s + transfer,
                        0,
                        "host_read_done",
                        link_server,
                        request_id,
                    )
                else:
                    buffer_bytes -= request.kv_buffer_bytes
                    if buffer_bytes < 0:
                        raise AssertionError("KV buffer accounting underflow")
                    records[request_id]["buffer_leave_s"] = time_s
            first_decode_token = (
                time_s + transfer + request.admission_s + request.decode_tpot_s
            )
            decode_tokens = request.generation_tokens - 1
            completion = (
                time_s
                + transfer
                + request.admission_s
                + decode_tokens * request.decode_tpot_s
            )
            service = completion - time_s
            decode_busy_s += service
            admission_bytes += request.admission_bytes
            records[request_id].update(
                decode_server=server,
                decode_start_s=time_s,
                first_decode_token_s=first_decode_token,
                completion_s=completion,
                status="running",
            )
            event(completion, 0, "decode_done", server, request_id)
            changed = True
        return changed

    while event_heap:
        time_s = event_heap[0][0]
        last_time = max(last_time, time_s)
        current = []
        while event_heap and event_heap[0][0] == time_s:
            current.append(heapq.heappop(event_heap))
        for _, _, _, kind, payload in current:
            if kind == "arrival":
                request_id = str(payload[0])
                records[request_id] = {
                    "request_id": request_id,
                    "arrival_s": time_s,
                    "workload_bucket": by_id[request_id].workload_bucket,
                    "status": "queued",
                }
                if inflight >= config.max_inflight_requests:
                    records[request_id].update(
                        status="rejected",
                        rejection_reason="max_inflight_requests",
                        completion_s=time_s,
                    )
                else:
                    inflight += 1
                    prefill_wait.append(request_id)
            elif kind == "prefill_done":
                server, request_id = int(payload[0]), str(payload[1])
                if config.regime == "back_pressure":
                    records[request_id]["first_token_s"] = time_s
                    blocked_prefill.append((request_id, server, time_s))
                else:
                    records[request_id]["first_token_s"] = time_s
                    heapq.heappush(prefill_idle, server)
                    if reserve_buffer(request_id, time_s):
                        transfer_wait.append(request_id)
            elif kind == "transfer_done":
                server, request_id = int(payload[0]), str(payload[1])
                heapq.heappush(link_idle, server)
                decode_ready.append(request_id)
            elif kind == "back_transfer_done":
                link_server = int(payload[0])
                server, request_id = int(payload[1]), str(payload[2])
                blocked_at = float(payload[3])
                heapq.heappush(link_idle, link_server)
                blocked = time_s - blocked_at
                prefill_stall_s += blocked
                records[request_id]["prefill_blocked_s"] = blocked
                records[request_id]["prefill_release_s"] = time_s
                heapq.heappush(prefill_idle, server)
            elif kind == "host_read_done":
                link_server, request_id = int(payload[0]), str(payload[1])
                heapq.heappush(link_idle, link_server)
                buffer_bytes -= by_id[request_id].kv_buffer_bytes
                if buffer_bytes < 0:
                    raise AssertionError("KV buffer accounting underflow")
                records[request_id]["buffer_leave_s"] = time_s
                records[request_id]["host_read_released_s"] = time_s
            elif kind == "decode_done":
                server, request_id = int(payload[0]), str(payload[1])
                heapq.heappush(decode_idle, server)
                records[request_id]["status"] = "completed"
                inflight -= 1
            else:  # pragma: no cover - internal invariant
                raise AssertionError(f"unknown queue event {kind}")

        while True:
            changed = start_decode(time_s)
            changed = start_transfer(time_s) or changed
            changed = start_prefill(time_s) or changed
            if not changed:
                break
        peak_inflight = max(peak_inflight, inflight)
        peak_prefill_queue = max(peak_prefill_queue, len(prefill_wait))

    if inflight or prefill_wait or transfer_wait or decode_ready or blocked_prefill:
        raise RuntimeError("queue simulation ended with unfinished requests")
    if buffer_bytes:
        raise RuntimeError("queue simulation ended with retained KV bytes")

    completed = [record for record in records.values() if record["status"] == "completed"]
    rejected = [record for record in records.values() if record["status"] == "rejected"]
    ttft = [record["first_token_s"] - record["arrival_s"] for record in completed]
    e2e = [record["completion_s"] - record["arrival_s"] for record in completed]
    service_tpot = [
        by_id[record["request_id"]].decode_tpot_s for record in completed
    ]
    observed_tpot = [
        (record["completion_s"] - record["first_token_s"])
        / (by_id[record["request_id"]].generation_tokens - 1)
        for record in completed
    ]
    good = []
    for record, request_ttft, request_tpot, request_e2e in zip(
        completed, ttft, observed_tpot, e2e
    ):
        within = request_ttft <= config.ttft_slo_s and request_tpot <= config.tpot_slo_s
        if config.e2e_slo_s is not None:
            within = within and request_e2e <= config.e2e_slo_s
        record["slo_met"] = within
        good.append(within)

    start_time = min(request.arrival_s for request in ordered)
    horizon = max((record["completion_s"] for record in records.values()), default=start_time) - start_time
    horizon = max(horizon, 1e-15)
    capacities = {
        "prefill": config.prefill_replicas * horizon,
        "decode": config.decode_replicas * horizon,
        "link": config.link_ports * horizon,
    }
    idle_energy = (
        max(0.0, capacities["prefill"] - prefill_busy_s)
        * config.prefill_idle_power_w
        + max(0.0, capacities["decode"] - decode_busy_s)
        * config.decode_idle_power_w
        + max(0.0, capacities["link"] - link_busy_s)
        * config.link_idle_power_w
    )
    active_energy = sum(
        by_id[record["request_id"]].prefill_energy_j
        + by_id[record["request_id"]].handoff_energy_j
        + (by_id[record["request_id"]].generation_tokens - 1)
        * by_id[record["request_id"]].decode_energy_per_token_j
        for record in completed
    )
    rejected_prefill_energy = sum(
        by_id[record["request_id"]].prefill_energy_j
        for record in rejected
        if record.get("rejection_reason") == "kv_buffer_capacity"
    )
    active_energy += rejected_prefill_energy
    total_energy = active_energy + idle_energy
    rejection_reasons = collections.Counter(
        str(record["rejection_reason"]) for record in rejected
    )

    request_rows = []
    for request in ordered:
        record = dict(records[request.request_id])
        if record["status"] == "completed":
            record.update(
                ttft_s=record["first_token_s"] - record["arrival_s"],
                decode_service_tpot_s=request.decode_tpot_s,
                observed_tpot_s=(
                    record["completion_s"] - record["first_token_s"]
                )
                / (request.generation_tokens - 1),
                e2e_s=record["completion_s"] - record["arrival_s"],
                prefill_queue_s=record["prefill_start_s"] - record["arrival_s"],
                decode_queue_s=record["decode_start_s"]
                - record["prefill_finish_compute_s"],
            )
        request_rows.append(record)

    body = {
        "schema_version": SCHEMA,
        "config": {
            "regime": config.regime,
            "prefill_replicas": config.prefill_replicas,
            "decode_replicas": config.decode_replicas,
            "prefill_decode_ratio": config.prefill_replicas / config.decode_replicas,
            "link_ports": config.link_ports,
            "kv_buffer_capacity_bytes": config.kv_buffer_capacity_bytes,
            "max_inflight_requests": config.max_inflight_requests,
            "slo": {
                "ttft_s": config.ttft_slo_s,
                "tpot_s": config.tpot_slo_s,
                "e2e_s": config.e2e_slo_s,
            },
        },
        "request_count": len(ordered),
        "completed_requests": len(completed),
        "rejected_requests": len(rejected),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "horizon_s": horizon,
        "throughput_requests_per_s": len(completed) / horizon,
        "throughput_generated_tokens_per_s": sum(
            by_id[record["request_id"]].generation_tokens for record in completed
        )
        / horizon,
        "offered_requests_per_s": len(ordered) / horizon,
        "slo_goodput_requests_per_s": sum(good) / horizon,
        "slo_goodput_generated_tokens_per_s": sum(
            by_id[record["request_id"]].generation_tokens
            for record, passed in zip(completed, good)
            if passed
        )
        / horizon,
        "slo_attainment_fraction": sum(good) / len(completed) if completed else 0.0,
        "latency": {
            "ttft_s": _distribution(ttft),
            "observed_tpot_s": _distribution(observed_tpot),
            "decode_service_tpot_s": _distribution(service_tpot),
            "e2e_s": _distribution(e2e),
        },
        "resources": {
            "prefill_utilization": prefill_busy_s / capacities["prefill"],
            "prefill_stall_fraction": prefill_stall_s / capacities["prefill"],
            "decode_utilization": decode_busy_s / capacities["decode"],
            "link_utilization": link_busy_s / capacities["link"],
            "peak_kv_buffer_bytes": peak_buffer_bytes,
            "peak_inflight_requests": peak_inflight,
            "peak_prefill_queue_requests": peak_prefill_queue,
            "admission_bytes": admission_bytes,
            "achieved_admission_bandwidth_bytes_per_s": admission_bytes / horizon,
        },
        "energy": {
            "active_j": active_energy,
            "rejected_prefill_j": rejected_prefill_energy,
            "idle_j": idle_energy,
            "total_j": total_energy,
            "joules_per_completed_request": (
                total_energy / len(completed) if completed else None
            ),
        },
        "requests": request_rows,
        "model_scope": {
            "queue_policy": (
                "FCFS within each stage; host-buffered decode reads have priority "
                "over queued host writes at a shared-link event boundary"
            ),
            "decode_service": "one request occupies one decode replica",
            "first_token_owner": "prefill",
            "observed_tpot": (
                "mean token interval from the prefill-owned first token through "
                "the final decode token, including queue/handoff/admission delay"
            ),
            "host_link": (
                "host writes and reads contend for the same configured link-port "
                "pool; read time also occupies decode service"
            ),
            "continuous_batching": False,
            "standalone_input_provenance_rankable": False,
            "external_input_receipt_required": True,
            "standalone_unrankable_reason": (
                "arrival, service, link, and power inputs are supplied by the caller"
            ),
        },
    }
    return body | {"content_hash": _canonical_hash(body)}


def simulate_sensitivity(
    templates: Sequence[ServingRequest],
    *,
    rates_per_s: Sequence[float],
    regimes: Sequence[str],
    replica_pairs: Sequence[tuple[int, int]],
    base_config: ServingQueueConfig,
    request_count: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    """Cross arrival rates, regimes, and prefill/decode replica counts."""

    if not templates:
        raise ValueError("at least one request template is required")
    outputs = []
    for rate in rates_per_s:
        arrivals = poisson_arrival_times(
            rate_per_s=rate, count=request_count, seed=seed
        )
        requests = []
        for index, arrival in enumerate(arrivals):
            source = templates[index % len(templates)]
            requests.append(
                ServingRequest(
                    **{
                        **source.__dict__,
                        "request_id": f"rate-{float(rate):g}-request-{index:06d}",
                        "arrival_s": arrival,
                    }
                )
            )
        for regime in regimes:
            for prefill_replicas, decode_replicas in replica_pairs:
                config = ServingQueueConfig(
                    **{
                        **base_config.__dict__,
                        "regime": regime,
                        "prefill_replicas": prefill_replicas,
                        "decode_replicas": decode_replicas,
                    }
                )
                result = simulate_serving_queue(requests, config)
                result.pop("content_hash", None)
                result["sensitivity"] = {
                    "arrival_rate_per_s": float(rate),
                    "seed": seed,
                }
                result["content_hash"] = _canonical_hash(result)
                outputs.append(result)
    return tuple(outputs)


def _load_requests(path: Path) -> tuple[ServingRequest, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("requests") if isinstance(value, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError("request input must contain a requests list")
    return tuple(ServingRequest(**row) for row in rows)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    config_value = json.loads(args.config.read_text(encoding="utf-8"))
    config = ServingQueueConfig(**config_value)
    result = simulate_serving_queue(_load_requests(args.requests), config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "content_hash": result["content_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
