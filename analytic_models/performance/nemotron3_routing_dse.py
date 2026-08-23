"""Profile-driven routed-expert weight-cache DSE for Nemotron 3 decode."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transactional_emulator.testbench.model_configs.loader import ModelArchConfig, load_model_config

from .nemotron3_workload import Precision, storage_bytes


PINNED_TRACE = Path(__file__).with_name("profiles") / "nemotron3_decode_routing_trace.json"
DEFAULT_CAPACITY_ENTRIES = (0, 23, 46, 92, 138, 256, 512, 1024, 2048, 2944)
MIB = 1024 * 1024


class RoutingTraceError(ValueError):
    pass


@dataclass(frozen=True)
class RoutedCacheResult:
    capacity_entries: int
    capacity_bytes: int
    prefill_active_entries: int
    prefill_resident_entries: int
    accesses: int
    hits: int
    misses: int
    weight_read_bytes: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["capacity_mib"] = self.capacity_bytes / MIB
        result["weight_read_mib"] = self.weight_read_bytes / MIB
        result["hit_rate"] = self.hit_rate
        return result


def load_routing_trace(path: Path = PINNED_TRACE) -> dict[str, Any]:
    try:
        trace = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RoutingTraceError(f"cannot read routing trace {path}: {error}") from error
    if trace.get("schema_version") != 1 or trace.get("contract") != "nemotron3-decode-routing-v1":
        raise RoutingTraceError("unsupported routing trace contract")
    shape = trace.get("shape", {})
    expected = {
        "context_tokens": 2048,
        "generated_tokens": 128,
        "recurrent_decode_steps": 127,
        "layers": 23,
        "experts": 128,
        "top_k": 6,
    }
    if shape != expected:
        raise RoutingTraceError("unexpected routing trace shape")
    layer_names = trace.get("layer_names")
    prefill = trace.get("prefill_active_experts_by_layer")
    prefill_by_length = trace.get("prefill_active_experts_by_sequence_length")
    decode = trace.get("decode_topk_by_step")
    if not isinstance(layer_names, list) or len(layer_names) != shape["layers"]:
        raise RoutingTraceError("routing trace layer names are incomplete")
    if not isinstance(prefill, list) or len(prefill) != shape["layers"]:
        raise RoutingTraceError("routing trace prefill state is incomplete")
    if not isinstance(prefill_by_length, dict) or set(prefill_by_length) != {
        "128",
        "2048",
        "8192",
    }:
        raise RoutingTraceError("routing trace prefill sequence-length coverage is incomplete")
    if not isinstance(decode, list) or len(decode) != shape["recurrent_decode_steps"]:
        raise RoutingTraceError("routing trace decode steps are incomplete")

    for layer_index, experts in enumerate(prefill):
        _validate_experts(experts, shape["experts"], f"prefill layer {layer_index}", unique=True)
    for token_count, per_layer in prefill_by_length.items():
        if not isinstance(per_layer, list) or len(per_layer) != shape["layers"]:
            raise RoutingTraceError(f"prefill S{token_count} does not cover every MoE layer")
        for layer_index, experts in enumerate(per_layer):
            _validate_experts(
                experts,
                shape["experts"],
                f"prefill S{token_count} layer {layer_index}",
                unique=True,
            )
    for step_index, step in enumerate(decode):
        if not isinstance(step, list) or len(step) != shape["layers"]:
            raise RoutingTraceError(f"decode step {step_index} does not cover every MoE layer")
        for layer_index, experts in enumerate(step):
            if not isinstance(experts, list) or len(experts) != shape["top_k"]:
                raise RoutingTraceError(f"decode step {step_index}, layer {layer_index}: invalid top-k width")
            _validate_experts(
                experts,
                shape["experts"],
                f"decode step {step_index}, layer {layer_index}",
                unique=True,
            )
    return trace


def _validate_experts(experts: object, expert_count: int, label: str, *, unique: bool) -> None:
    if not isinstance(experts, list) or not all(isinstance(expert, int) for expert in experts):
        raise RoutingTraceError(f"{label}: expert IDs must be integers")
    if any(expert < 0 or expert >= expert_count for expert in experts):
        raise RoutingTraceError(f"{label}: expert ID is out of range")
    if unique and len(experts) != len(set(experts)):
        raise RoutingTraceError(f"{label}: expert IDs must be unique")


def simulate_routed_expert_lru(
    trace: dict[str, Any],
    *,
    entry_bytes: int,
    capacity_entries: int,
    expert_order: str = "topk_rank",
) -> RoutedCacheResult:
    """Replay exact decode top-k IDs after a deterministic prefill warm start.

    The compact campaign trace preserves the prefill resident set but not the
    final prefill recency order.  We therefore seed it in layer-major,
    ascending-expert order.  Every decode access after that is exact.
    """

    if entry_bytes <= 0:
        raise ValueError("entry_bytes must be positive")
    if capacity_entries < 0:
        raise ValueError("capacity_entries must be non-negative")
    if expert_order not in {"topk_rank", "expert_id"}:
        raise ValueError("expert_order must be topk_rank or expert_id")
    prefill = trace["prefill_active_experts_by_layer"]
    decode = trace["decode_topk_by_step"]
    active_keys = [(layer, expert) for layer, experts in enumerate(prefill) for expert in experts]
    cache: OrderedDict[tuple[int, int], None] = OrderedDict()

    def insert(key: tuple[int, int]) -> bool:
        if capacity_entries == 0:
            return False
        if key in cache:
            cache.move_to_end(key)
            return True
        if len(cache) == capacity_entries:
            cache.popitem(last=False)
        cache[key] = None
        return False

    for key in active_keys:
        insert(key)
    prefill_resident_entries = len(cache)

    hits = misses = accesses = 0
    for step in decode:
        for layer, experts in enumerate(step):
            ordered_experts = experts if expert_order == "topk_rank" else sorted(experts)
            for expert in ordered_experts:
                accesses += 1
                if insert((layer, expert)):
                    hits += 1
                else:
                    misses += 1
    return RoutedCacheResult(
        capacity_entries=capacity_entries,
        capacity_bytes=capacity_entries * entry_bytes,
        prefill_active_entries=len(active_keys),
        prefill_resident_entries=prefill_resident_entries,
        accesses=accesses,
        hits=hits,
        misses=misses,
        weight_read_bytes=misses * entry_bytes,
    )


def build_report(
    arch: ModelArchConfig,
    trace_path: Path = PINNED_TRACE,
    *,
    capacity_entries: tuple[int, ...] = DEFAULT_CAPACITY_ENTRIES,
) -> dict[str, Any]:
    if arch.moe is None:
        raise ValueError("routing DSE requires a MoE architecture")
    trace = load_routing_trace(trace_path)
    moe = arch.moe
    routed_entry_bytes = storage_bytes(
        2 * arch.hidden_size * moe.intermediate_size,
        Precision.NVFP4,
    )
    shared_entry_bytes = storage_bytes(
        2 * arch.hidden_size * moe.shared_intermediate_size,
        Precision.NVFP4,
    )
    access_orders = {
        expert_order: [
            simulate_routed_expert_lru(
                trace,
                entry_bytes=routed_entry_bytes,
                capacity_entries=capacity,
                expert_order=expert_order,
            ).to_dict()
            for capacity in capacity_entries
        ]
        for expert_order in ("expert_id", "topk_rank")
    }
    shared_entries = arch.count_layers("moe") * moe.shared_experts
    decode_steps = trace["shape"]["recurrent_decode_steps"]
    return {
        "schema_version": 1,
        "status": "profile_driven_workload_calibrated_plena_cycles_uncalibrated",
        "source": trace["source"],
        "trace_shape": trace["shape"],
        "weight_format": "NVFP4 logical payload plus one FP8 scale per 16 values",
        "routed_expert": {
            "entry_bytes": routed_entry_bytes,
            "entry_mib": routed_entry_bytes / MIB,
            "total_entries": arch.count_layers("moe") * moe.num_experts,
            "access_orders": access_orders,
        },
        "shared_expert": {
            "entry_bytes": shared_entry_bytes,
            "entry_mib": shared_entry_bytes / MIB,
            "entries": shared_entries,
            "all_layers_resident_bytes": shared_entries * shared_entry_bytes,
            "all_layers_resident_mib": shared_entries * shared_entry_bytes / MIB,
            "stream_every_decode_step_bytes": decode_steps * shared_entries * shared_entry_bytes,
            "stream_every_decode_step_gib": decode_steps * shared_entries * shared_entry_bytes / (1024**3),
        },
        "assumptions": [
            "Cache keys are layer-specific routed experts; weights are read-only and use global LRU replacement.",
            "The exact 127-step top-k ID sets are replayed in model layer order.",
            "expert_id models an expert-grouped scheduler; topk_rank preserves router rank order because the final hardware order is not frozen.",
            "The compact trace preserves the prefill active set but not final recency; warm-start order is layer-major then expert ID.",
            "NVFP4 global tensor scales, alignment padding, and decompression bandwidth are not included.",
            "This models weight residency and HBM payload only; Expert/M/K compute scheduling is separate.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=PINNED_TRACE)
    parser.add_argument("--capacity-entries", default=",".join(map(str, DEFAULT_CAPACITY_ENTRIES)))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    capacities = tuple(int(item) for item in args.capacity_entries.split(",") if item)
    arch = load_model_config("nemotron3_nano_30b_a3b").arch
    report = build_report(arch, args.trace, capacity_entries=capacities)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
