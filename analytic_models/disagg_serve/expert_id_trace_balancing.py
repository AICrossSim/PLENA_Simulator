"""Held-out, trace-derived expert-ID placement for Qwen3-MoE decode.

The router and hidden state are replicated on every TP rank.  A rank filters
the top-k IDs it owns, evaluates those complete experts locally, and the
existing expert-output all-reduce combines the partial outputs.  There is no
hidden-state dispatch in this mapping.

Placement is learned from a chronological trace prefix.  All reported timing
comparisons use disjoint suffix windows.  Every artifact remains an isolated
analytic projection: it is never publication-, hardware-, or selection-
rankable without independent compiler, emulator, RTL, and power evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from .expert_placement import (
        MODEL_ID,
        MODEL_REVISION,
        NUM_EXPERTS,
        NUM_LAYERS,
        TOP_K,
        RoutingTrace,
        trace_content_hash,
    )
    from .physical_ledger import matrix_planes
    from .body_weight_layout import (
        EXPERT_ID_PARALLEL,
        EXPERT_TENSOR_PARALLEL,
        build_body_weight_physical_layout,
    )
    from .packed_kv import DENSE_SELECTOR, traffic_from_precision
    from ..performance.perf_model import PerfModel
except ImportError:  # script-style imports
    from expert_placement import (  # type: ignore[no-redef]
        MODEL_ID,
        MODEL_REVISION,
        NUM_EXPERTS,
        NUM_LAYERS,
        TOP_K,
        RoutingTrace,
        trace_content_hash,
    )
    from physical_ledger import matrix_planes  # type: ignore[no-redef]
    from body_weight_layout import (  # type: ignore[no-redef]
        EXPERT_ID_PARALLEL,
        EXPERT_TENSOR_PARALLEL,
        build_body_weight_physical_layout,
    )
    from packed_kv import DENSE_SELECTOR, traffic_from_precision  # type: ignore[no-redef]
    from perf_model import PerfModel  # type: ignore[no-redef]


REPORT_SCHEMA = "plena-qwen3-moe-expert-id-held-out-report/v1"
INDEX_SCHEMA = "plena-qwen3-moe-expert-id-held-out-index/v1"
WINDOW_OVERLAY_SCHEMA = "plena-qwen3-moe-expert-id-window-overlay/v1"
BODY_REPRICE_SCHEMA = "plena-qwen3-moe-expert-id-window-body-reprice/v1"
PLACEMENT_ALGORITHM = (
    "per_layer_frequency_lpt_capacity_exact_with_training_nonregression_fallback/v1"
)
SPLIT_POLICY = "chronological_prefix_train_suffix_held_out/v1"
WINDOW_POLICY = "consecutive_nonoverlapping_trace_steps_post_hoc_batch_proxy/v1"
EXPERT_ID_MAPPING = (
    "replicated_hidden_local_route_filter_then_output_allreduce"
)
TENSOR_MAPPING = "replicated_hidden_tensor_sharded_experts_then_output_allreduce"
FREQUENCY_HZ = 1_000_000_000
HIDDEN_SIZE = 2048
EXPERT_INTERMEDIATE_SIZE = 768
_SHA256 = re.compile(r"[0-9a-f]{64}")


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hashed_body(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("content_hash", None)
    return body | {"content_hash": canonical_hash(body)}


def _load_hashed_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    body = dict(value)
    observed = body.pop("content_hash", None)
    if observed != canonical_hash(body):
        raise ValueError(f"content hash mismatch: {path}")
    return value


def _positive_int(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _positive_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class ExpertIdBalanceConfig:
    """Fully bound topology, geometry, precision, and split configuration."""

    tensor_parallel_degree: int
    kv_parallel_degree: int
    batch_size: int
    train_step_count: int
    mlen: int
    blen: int
    vlen: int
    ffn_element_bits: int
    ffn_effective_bits: float
    mx_block_size: int
    precision_label: str
    link_generation: str
    tp_link_bandwidth_bytes_per_s: float
    tp_link_ports: int
    activation_bytes: int = 2
    weight_alignment_bytes: int = 64
    context_tokens: int = 32768
    hbm_capacity_bytes_per_chip: int = 32_000_000_000
    runtime_hbm_reserve_bytes_per_chip: int = 536_870_912
    kv_layout: str = DENSE_SELECTOR
    require_sample_disjoint_split: bool = True
    hidden_size: int = HIDDEN_SIZE
    expert_intermediate_size: int = EXPERT_INTERMEDIATE_SIZE
    num_layers: int = NUM_LAYERS
    num_experts: int = NUM_EXPERTS
    experts_per_token: int = TOP_K
    frequency_hz: int = FREQUENCY_HZ

    def __post_init__(self) -> None:
        for field in (
            "tensor_parallel_degree",
            "kv_parallel_degree",
            "batch_size",
            "train_step_count",
            "mlen",
            "blen",
            "vlen",
            "ffn_element_bits",
            "mx_block_size",
            "tp_link_ports",
            "activation_bytes",
            "weight_alignment_bytes",
            "context_tokens",
            "hbm_capacity_bytes_per_chip",
            "hidden_size",
            "expert_intermediate_size",
            "num_layers",
            "num_experts",
            "experts_per_token",
            "frequency_hz",
        ):
            _positive_int(getattr(self, field), field)
        _positive_int(
            self.runtime_hbm_reserve_bytes_per_chip,
            "runtime_hbm_reserve_bytes_per_chip",
            allow_zero=True,
        )
        if self.tensor_parallel_degree not in (1, 2, 4):
            raise ValueError("tensor_parallel_degree must be one of 1, 2, or 4")
        if self.kv_parallel_degree not in (1, 2, 4):
            raise ValueError("kv_parallel_degree must be one of 1, 2, or 4")
        if self.num_experts % self.tensor_parallel_degree:
            raise ValueError("tensor parallel degree must divide all experts")
        if self.train_step_count % self.batch_size:
            raise ValueError("train_step_count must contain complete batch windows")
        if self.mlen % self.blen:
            raise ValueError("mlen must be divisible by blen")
        if self.vlen < self.blen:
            raise ValueError("vlen must be at least blen")
        if (
            self.hidden_size != HIDDEN_SIZE
            or self.expert_intermediate_size != EXPERT_INTERMEDIATE_SIZE
            or self.num_layers != NUM_LAYERS
            or self.num_experts != NUM_EXPERTS
            or self.experts_per_token != TOP_K
            or self.frequency_hz != FREQUENCY_HZ
        ):
            raise ValueError("configuration differs from the sealed Qwen3 target")
        _positive_float(self.ffn_effective_bits, "ffn_effective_bits")
        _positive_float(
            self.tp_link_bandwidth_bytes_per_s,
            "tp_link_bandwidth_bytes_per_s",
        )
        if not isinstance(self.precision_label, str) or not self.precision_label:
            raise ValueError("precision_label must be non-empty")
        if not isinstance(self.link_generation, str) or not self.link_generation:
            raise ValueError("link_generation must be non-empty")
        if self.kv_layout != DENSE_SELECTOR:
            raise ValueError("held-out capacity sensitivity requires dense_selector KV")
        if self.require_sample_disjoint_split is not True:
            raise ValueError("held-out protocol requires sample-disjoint splits")


def hardware_binding_for_perf_model(
    perf: PerfModel,
    *,
    hardware_config_path: Path,
    custom_isa_path: Path,
) -> dict[str, Any]:
    """Bind resolved PerfModel timing to its two immutable source files."""

    config_path = hardware_config_path.resolve()
    isa_path = custom_isa_path.resolve()
    if not config_path.is_file() or not isa_path.is_file():
        raise FileNotFoundError("PerfModel hardware inputs are missing")
    latencies = dict(sorted(perf.instr.latencies.items()))
    return {
        "hardware_config_path": str(config_path),
        "hardware_config_sha256": file_hash(config_path),
        "custom_isa_path": str(isa_path),
        "custom_isa_sha256": file_hash(isa_path),
        "timing_mode": str(perf.timing_mode),
        "resolved_geometry": {
            "mlen": int(perf.mlen),
            "blen": int(perf.blen),
            "vlen": int(perf.vlen),
            "hlen": int(perf.hlen),
        },
        "instruction_latencies": latencies,
        "instruction_latencies_content_hash": canonical_hash(latencies),
    }


def _validate_hardware_binding(
    binding: Mapping[str, Any],
    perf: PerfModel,
    config: ExpertIdBalanceConfig,
) -> None:
    if not isinstance(binding, Mapping):
        raise ValueError("hardware_binding must be an object")
    for path_field, hash_field in (
        ("hardware_config_path", "hardware_config_sha256"),
        ("custom_isa_path", "custom_isa_sha256"),
    ):
        path = Path(str(binding.get(path_field, "")))
        if (
            not path.is_absolute()
            or not path.is_file()
            or file_hash(path) != binding.get(hash_field)
        ):
            raise ValueError(f"hardware binding {path_field} is missing or changed")
    geometry = binding.get("resolved_geometry")
    if geometry != {
        "mlen": config.mlen,
        "blen": config.blen,
        "vlen": config.vlen,
        "hlen": int(perf.hlen),
    }:
        raise ValueError("hardware binding geometry differs from study config")
    if (
        int(perf.mlen) != config.mlen
        or int(perf.blen) != config.blen
        or int(perf.vlen) != config.vlen
        or binding.get("timing_mode") != str(perf.timing_mode)
    ):
        raise ValueError("PerfModel differs from the hardware binding")
    latencies = dict(sorted(perf.instr.latencies.items()))
    if (
        binding.get("instruction_latencies") != latencies
        or binding.get("instruction_latencies_content_hash")
        != canonical_hash(latencies)
    ):
        raise ValueError("resolved instruction-latency binding differs")


def validate_hardware_binding_for_perf_model(
    binding: Mapping[str, Any],
    perf: PerfModel,
    config: ExpertIdBalanceConfig,
) -> None:
    """Validate a previously materialized PerfModel hardware/ISA binding."""

    _validate_hardware_binding(binding, perf, config)


def _audit_trace(trace: RoutingTrace) -> tuple[tuple[Any, ...], ...]:
    if trace.source_kind != "measured" or trace.receipt is None:
        raise ValueError("expert-ID balancing requires a measured trace receipt")
    if trace.model_id != MODEL_ID or trace.model_revision != MODEL_REVISION:
        raise ValueError("routing trace differs from the sealed model identity")
    if tuple(step.step_index for step in trace.steps) != tuple(range(len(trace.steps))):
        raise ValueError("routing steps must be contiguous and zero-based")
    expected_layers = set(range(NUM_LAYERS))
    ordered: list[tuple[Any, ...]] = []
    for step in trace.steps:
        if len(step.records) != NUM_LAYERS:
            raise ValueError("each routing step must contain exactly 48 records")
        if {record.layer for record in step.records} != expected_layers:
            raise ValueError("each routing step must cover every layer exactly once")
        if len({record.token_id for record in step.records}) != 1:
            raise ValueError("a collector step must describe exactly one token")
        if len({record.source_chip for record in step.records}) != 1:
            raise ValueError("a collector step must describe one source chip")
        rows = tuple(sorted(step.records, key=lambda record: record.layer))
        if sum(len(record.expert_ids) for record in rows) != NUM_LAYERS * TOP_K:
            raise ValueError("routing assignments do not conserve")
        ordered.append(rows)
    return tuple(ordered)


def _sample_id_for_step(step: Sequence[Any]) -> str:
    token_id = str(step[0].token_id)
    marker = ":cached-decode:"
    return token_id.rsplit(marker, 1)[0] if marker in token_id else token_id


def _validate_source_binding(
    binding: Mapping[str, Any],
    trace: RoutingTrace,
) -> None:
    if not isinstance(binding, Mapping) or binding.get("collector_verified") is not True:
        raise ValueError("source binding requires collector verification")
    paths: dict[str, Path] = {}
    for path_field, hash_field in (
        ("router_index_path", "router_index_sha256"),
        ("placement_input_path", "placement_input_sha256"),
        ("router_trace_evidence_path", "router_trace_evidence_sha256"),
    ):
        path = Path(str(binding.get(path_field, "")))
        if (
            not path.is_absolute()
            or not path.is_file()
            or file_hash(path) != binding.get(hash_field)
        ):
            raise ValueError(f"source binding {path_field} is missing or changed")
        paths[path_field] = path.resolve()
    trace_hash = trace_content_hash(trace)
    if binding.get("trace_content_hash") != trace_hash:
        raise ValueError("source binding and parsed trace hashes differ")

    index = _load_hashed_json(paths["router_index_path"])
    placement = _load_hashed_json(paths["placement_input_path"])
    evidence = _load_hashed_json(paths["router_trace_evidence_path"])
    for value, field in (
        (index, "router_index_content_hash"),
        (placement, "placement_input_content_hash"),
        (evidence, "router_trace_evidence_content_hash"),
    ):
        if value.get("content_hash") != binding.get(field):
            raise ValueError(f"source binding {field} differs")
    if (
        index.get("schema") != "plena-qwen3-moe-router-trace-index/v1"
        or placement.get("schema") != "plena-qwen3-moe-expert-placement-input/v1"
        or evidence.get("schema") != "plena-qwen3-moe-router-trace-evidence/v1"
    ):
        raise ValueError("source binding uses an unsupported collector schema")
    if (
        index.get("trace_content_hash") != trace_hash
        or evidence.get("trace_content_hash") != trace_hash
        or index.get("token_step_count") != len(trace.steps)
    ):
        raise ValueError("collector artifacts bind a different routing trace")
    index_dir = paths["router_index_path"].parent
    if (
        (index_dir / str(index.get("input_path", ""))).resolve()
        != paths["placement_input_path"]
        or (index_dir / str(index.get("artifact_path", ""))).resolve()
        != paths["router_trace_evidence_path"]
        or index.get("input_sha256") != binding.get("placement_input_sha256")
        or index.get("artifact_sha256")
        != binding.get("router_trace_evidence_sha256")
    ):
        raise ValueError("collector index paths or hashes differ from the binding")
    receipt = trace.receipt
    if (
        receipt.artifact_path != str(paths["router_trace_evidence_path"])
        or receipt.artifact_sha256
        != binding.get("router_trace_evidence_sha256")
        or receipt.subject_sha256 != trace_hash
        or receipt.sample_count != len(trace.steps)
        or not _SHA256.fullmatch(receipt.tool_revision)
    ):
        raise ValueError("routing-trace receipt differs from bound evidence")


def _balanced_widths(value: int, shards: int) -> tuple[int, ...]:
    quotient, remainder = divmod(value, shards)
    if quotient == 0:
        raise ValueError("parallel degree exceeds partitioned dimension")
    return tuple(quotient + int(rank < remainder) for rank in range(shards))


def _baseline_owner(tp: int) -> tuple[int, ...]:
    return tuple(expert_id % tp for expert_id in range(NUM_EXPERTS))


def _frequency_candidate(frequencies: Sequence[int], tp: int) -> tuple[int, ...]:
    capacity = NUM_EXPERTS // tp
    owners = [-1] * NUM_EXPERTS
    rank_loads = [0] * tp
    rank_counts = [0] * tp
    for expert_id in sorted(range(NUM_EXPERTS), key=lambda item: (-frequencies[item], item)):
        available = [rank for rank in range(tp) if rank_counts[rank] < capacity]
        rank = min(available, key=lambda item: (rank_loads[item], rank_counts[item], item))
        owners[expert_id] = rank
        rank_loads[rank] += int(frequencies[expert_id])
        rank_counts[rank] += 1
    if rank_counts != [capacity] * tp or any(owner < 0 for owner in owners):
        raise AssertionError("frequency placement failed exact rank capacity")
    return tuple(owners)


def _rank_frequency_loads(
    frequencies: Sequence[int], owners: Sequence[int], tp: int
) -> tuple[int, ...]:
    loads = [0] * tp
    for expert_id, count in enumerate(frequencies):
        loads[int(owners[expert_id])] += int(count)
    return tuple(loads)


def _rank_counts(
    records: Sequence[Any], owners: Sequence[int], tp: int
) -> tuple[Counter[int], ...]:
    result = tuple(Counter() for _ in range(tp))
    for record in records:
        for expert_id in record.expert_ids:
            result[int(owners[expert_id])][expert_id] += 1
    return result


def _histogram(counts: Mapping[int, int]) -> dict[str, int]:
    values = Counter(int(count) for count in counts.values())
    return {str(count): values[count] for count in sorted(values)}


def _expert_bytes(config: ExpertIdBalanceConfig, intermediate: int) -> int:
    physical_hidden = math.ceil(config.hidden_size / config.mlen) * config.mlen
    physical_intermediate = math.ceil(intermediate / config.mlen) * config.mlen
    common = {
        "element_bits": config.ffn_element_bits,
        "effective_bits": config.ffn_effective_bits,
        "block_size": config.mx_block_size,
        "alignment_bytes": config.weight_alignment_bytes,
    }
    gate_up = matrix_planes(
        physical_intermediate,
        physical_hidden,
        2,
        **common,
    )
    down = matrix_planes(
        physical_hidden,
        physical_intermediate,
        1,
        **common,
    )
    return gate_up.total_aligned + down.total_aligned


def _collective(config: ExpertIdBalanceConfig) -> dict[str, Any]:
    tp = config.tensor_parallel_degree
    payload = config.batch_size * config.hidden_size * config.activation_bytes
    slowest = 0 if tp == 1 else 2 * (tp - 1) * payload // tp
    system = slowest * tp * config.kv_parallel_degree
    time_s = (
        0.0
        if slowest == 0
        else slowest
        / (config.tp_link_bandwidth_bytes_per_s * config.tp_link_ports)
    )
    return {
        "mapping": "ring_allreduce",
        "count_per_layer": int(tp > 1),
        "payload_bytes": payload,
        "slowest_rank_bytes": slowest,
        "system_bytes": system,
        "time_s": time_s,
        "source_hidden_dispatch_bytes": 0,
        "source_hidden_dispatch_time_s": 0.0,
    }


def _expert_id_metrics(
    records: Sequence[Any],
    owners: Sequence[int],
    perf: PerfModel,
    config: ExpertIdBalanceConfig,
) -> dict[str, Any]:
    tp = config.tensor_parallel_degree
    capacity = NUM_EXPERTS // tp
    counts_by_rank = _rank_counts(records, owners, tp)
    assignment_counts = [sum(counts.values()) for counts in counts_by_rank]
    expected = len(records) * TOP_K
    if sum(assignment_counts) != expected:
        raise AssertionError("expert-ID assignment conservation failed")
    ledgers = [
        perf.moe_decode_expert_timing_from_histogram(
            config.hidden_size,
            config.expert_intermediate_size,
            _histogram(counts),
            owned_experts=capacity,
            expected_route_assignments=sum(counts.values()),
        )
        for counts in counts_by_rank
    ]
    cycles = [int(ledger["expert_stage_cycles"]) for ledger in ledgers]
    row_tiles = [int(ledger["expert_row_tiles"]) for ledger in ledgers]
    padding_rows = [int(ledger["expert_padding_rows"]) for ledger in ledgers]
    weight_per_expert = _expert_bytes(config, config.expert_intermediate_size)
    traffic = [len(counts) * weight_per_expert for counts in counts_by_rank]
    collective = _collective(config)
    slowest_cycles = max(cycles)
    return {
        "mapping": EXPERT_ID_MAPPING,
        "owner_capacity_per_rank": capacity,
        "assignment_count_by_rank": assignment_counts,
        "physical_assignment_count_by_rank_across_kvp": [
            count * config.kv_parallel_degree for count in assignment_counts
        ],
        "assignment_sum_semantics": "sum_across_ranks_equals_batch_times_topk",
        "active_expert_ids_by_rank": [
            sorted(counts) for counts in counts_by_rank
        ],
        "active_expert_count_by_rank": [len(counts) for counts in counts_by_rank],
        "expert_token_count_histogram_by_rank": [
            _histogram(counts) for counts in counts_by_rank
        ],
        "expert_row_tiles_by_rank": row_tiles,
        "expert_padding_rows_by_rank": padding_rows,
        "matrix_instruction_histogram_by_rank": [
            dict(ledger["matrix_instruction_histogram"]) for ledger in ledgers
        ],
        "auxiliary_instruction_histogram_by_rank": [
            dict(ledger["auxiliary_instruction_histogram"]) for ledger in ledgers
        ],
        "matrix_cycles_by_rank": [int(ledger["matrix_cycles"]) for ledger in ledgers],
        "activation_cycles_by_rank": [
            int(ledger["activation_cycles"]) for ledger in ledgers
        ],
        "auxiliary_cycles_by_rank": [
            int(ledger["auxiliary_cycles"]) for ledger in ledgers
        ],
        "expert_stage_cycles_by_rank": cycles,
        "slowest_rank_expert_stage_cycles": slowest_cycles,
        "slowest_rank_expert_stage_time_s": slowest_cycles / config.frequency_hz,
        "slowest_rank_cycle_rank": cycles.index(slowest_cycles),
        "expert_weight_bytes_per_active_expert": weight_per_expert,
        "expert_weight_hbm_bytes_by_rank": traffic,
        "slowest_rank_expert_weight_hbm_bytes": max(traffic),
        "system_expert_weight_hbm_bytes": (
            sum(traffic) * config.kv_parallel_degree
        ),
        "source_hidden_dispatch_bytes": 0,
        "expert_output_collective": collective,
        "expert_stage_plus_output_collective_time_s": (
            slowest_cycles / config.frequency_hz + float(collective["time_s"])
        ),
        "route_assignment_conservation": {
            "expected": expected,
            "observed": sum(assignment_counts),
            "logical_assignments_per_tp_group": expected,
            "physical_whole_expert_assignments_across_kvp": (
                sum(assignment_counts) * config.kv_parallel_degree
            ),
            "expected_physical_whole_expert_assignments_across_kvp": (
                expected * config.kv_parallel_degree
            ),
            "kvp_route_semantics": "identical_route_replica_not_new_trace_sample",
            "conserved": True,
        },
    }


def _tensor_metrics(
    records: Sequence[Any],
    perf: PerfModel,
    config: ExpertIdBalanceConfig,
) -> dict[str, Any]:
    tp = config.tensor_parallel_degree
    global_counts: Counter[int] = Counter()
    for record in records:
        global_counts.update(record.expert_ids)
    expected = len(records) * TOP_K
    if sum(global_counts.values()) != expected:
        raise AssertionError("tensor control route conservation failed")
    histogram = _histogram(global_counts)
    widths = _balanced_widths(config.expert_intermediate_size, tp)
    ledgers = [
        perf.moe_decode_expert_timing_from_histogram(
            config.hidden_size,
            width,
            histogram,
            owned_experts=NUM_EXPERTS,
            expected_route_assignments=expected,
            source="verified_trace_tensor_shard_global_expert_histogram",
        )
        for width in widths
    ]
    cycles = [int(ledger["expert_stage_cycles"]) for ledger in ledgers]
    traffic = [len(global_counts) * _expert_bytes(config, width) for width in widths]
    collective = _collective(config)
    slowest_cycles = max(cycles)
    return {
        "mapping": TENSOR_MAPPING,
        "local_intermediate_width_by_rank": list(widths),
        "assignment_count_by_rank": [expected] * tp,
        "physical_tensor_shard_assignment_count_by_rank_across_kvp": [
            expected * config.kv_parallel_degree
        ]
        * tp,
        "assignment_sum_semantics": (
            "each_rank_processes_a_tensor_shard_of_every_logical_assignment"
        ),
        "active_expert_ids_by_rank": [sorted(global_counts)] * tp,
        "active_expert_count_by_rank": [len(global_counts)] * tp,
        "expert_token_count_histogram_by_rank": [histogram] * tp,
        "expert_row_tiles_by_rank": [
            int(ledger["expert_row_tiles"]) for ledger in ledgers
        ],
        "expert_padding_rows_by_rank": [
            int(ledger["expert_padding_rows"]) for ledger in ledgers
        ],
        "matrix_instruction_histogram_by_rank": [
            dict(ledger["matrix_instruction_histogram"]) for ledger in ledgers
        ],
        "auxiliary_instruction_histogram_by_rank": [
            dict(ledger["auxiliary_instruction_histogram"]) for ledger in ledgers
        ],
        "matrix_cycles_by_rank": [int(ledger["matrix_cycles"]) for ledger in ledgers],
        "activation_cycles_by_rank": [
            int(ledger["activation_cycles"]) for ledger in ledgers
        ],
        "auxiliary_cycles_by_rank": [
            int(ledger["auxiliary_cycles"]) for ledger in ledgers
        ],
        "expert_stage_cycles_by_rank": cycles,
        "slowest_rank_expert_stage_cycles": slowest_cycles,
        "slowest_rank_expert_stage_time_s": slowest_cycles / config.frequency_hz,
        "slowest_rank_cycle_rank": cycles.index(slowest_cycles),
        "expert_weight_hbm_bytes_by_rank": traffic,
        "slowest_rank_expert_weight_hbm_bytes": max(traffic),
        "system_expert_weight_hbm_bytes": (
            sum(traffic) * config.kv_parallel_degree
        ),
        "source_hidden_dispatch_bytes": 0,
        "expert_output_collective": collective,
        "expert_stage_plus_output_collective_time_s": (
            slowest_cycles / config.frequency_hz + float(collective["time_s"])
        ),
        "route_assignment_conservation": {
            "expected": expected,
            "observed_once_before_tensor_sharding": sum(global_counts.values()),
            "logical_assignments_per_tp_group": expected,
            "physical_whole_expert_assignments_across_kvp": (
                expected * config.kv_parallel_degree
            ),
            "physical_tensor_shard_executions_across_tp_and_kvp": (
                expected * tp * config.kv_parallel_degree
            ),
            "tensor_shard_matrix_events_are_not_whole_expert_assignments": True,
            "kvp_route_semantics": "identical_route_replica_not_new_trace_sample",
            "conserved": True,
        },
    }


def _training_cycle_summary(
    windows: Sequence[Sequence[Sequence[Any]]],
    layer: int,
    owners: Sequence[int],
    perf: PerfModel,
    config: ExpertIdBalanceConfig,
) -> dict[str, int]:
    values = [
        int(
            _expert_id_metrics(
                [step[layer] for step in window], owners, perf, config
            )["slowest_rank_expert_stage_cycles"]
        )
        for window in windows
    ]
    return {
        "window_count": len(values),
        "total_slowest_rank_expert_stage_cycles": sum(values),
        "maximum_slowest_rank_expert_stage_cycles": max(values),
    }


def _learn_placements(
    train_steps: Sequence[Sequence[Any]],
    perf: PerfModel,
    config: ExpertIdBalanceConfig,
) -> tuple[list[dict[str, Any]], tuple[tuple[int, ...], ...]]:
    tp = config.tensor_parallel_degree
    capacity = NUM_EXPERTS // tp
    baseline = _baseline_owner(tp)
    windows = [
        train_steps[index : index + config.batch_size]
        for index in range(0, len(train_steps), config.batch_size)
    ]
    rows: list[dict[str, Any]] = []
    selected: list[tuple[int, ...]] = []
    for layer in range(NUM_LAYERS):
        frequency = [0] * NUM_EXPERTS
        for step in train_steps:
            for expert_id in step[layer].expert_ids:
                frequency[expert_id] += 1
        candidate = _frequency_candidate(frequency, tp)
        baseline_loads = _rank_frequency_loads(frequency, baseline, tp)
        candidate_loads = _rank_frequency_loads(frequency, candidate, tp)
        baseline_cycles = _training_cycle_summary(
            windows, layer, baseline, perf, config
        )
        candidate_cycles = _training_cycle_summary(
            windows, layer, candidate, perf, config
        )
        nonregression = (
            max(candidate_loads) <= max(baseline_loads)
            and candidate_cycles["total_slowest_rank_expert_stage_cycles"]
            <= baseline_cycles["total_slowest_rank_expert_stage_cycles"]
            and candidate_cycles["maximum_slowest_rank_expert_stage_cycles"]
            <= baseline_cycles["maximum_slowest_rank_expert_stage_cycles"]
        )
        candidate_objective = (
            max(candidate_loads),
            sum(load * load for load in candidate_loads),
            candidate_cycles["maximum_slowest_rank_expert_stage_cycles"],
            candidate_cycles["total_slowest_rank_expert_stage_cycles"],
            candidate,
        )
        baseline_objective = (
            max(baseline_loads),
            sum(load * load for load in baseline_loads),
            baseline_cycles["maximum_slowest_rank_expert_stage_cycles"],
            baseline_cycles["total_slowest_rank_expert_stage_cycles"],
            baseline,
        )
        use_candidate = nonregression and candidate_objective < baseline_objective
        owner = candidate if use_candidate else baseline
        selected_loads = candidate_loads if use_candidate else baseline_loads
        selected_cycles = candidate_cycles if use_candidate else baseline_cycles
        if any(owner.count(rank) != capacity for rank in range(tp)):
            raise AssertionError("selected expert ownership violates exact capacity")
        if (
            max(selected_loads) > max(baseline_loads)
            or selected_cycles["total_slowest_rank_expert_stage_cycles"]
            > baseline_cycles["total_slowest_rank_expert_stage_cycles"]
            or selected_cycles["maximum_slowest_rank_expert_stage_cycles"]
            > baseline_cycles["maximum_slowest_rank_expert_stage_cycles"]
        ):
            raise AssertionError("training-only placement regressed its baseline")
        rows.append(
            {
                "layer": layer,
                "training_expert_assignment_frequency": frequency,
                "baseline_expert_owner_by_id": list(baseline),
                "candidate_expert_owner_by_id": list(candidate),
                "selected_expert_owner_by_id": list(owner),
                "owned_expert_count_by_rank": [
                    owner.count(rank) for rank in range(tp)
                ],
                "capacity_per_rank": capacity,
                "baseline_assignment_frequency_by_rank": list(baseline_loads),
                "candidate_assignment_frequency_by_rank": list(candidate_loads),
                "selected_assignment_frequency_by_rank": list(selected_loads),
                "baseline_training_timing": baseline_cycles,
                "candidate_training_timing": candidate_cycles,
                "selected_training_timing": selected_cycles,
                "candidate_training_nonregression": nonregression,
                "selected_policy": (
                    "frequency_aware_capacity_balanced"
                    if use_candidate
                    else "cyclic_baseline_fallback"
                ),
                "held_out_steps_used_for_placement": False,
            }
        )
        selected.append(owner)
    return rows, tuple(selected)


def _aggregate_policy(
    rows: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    metrics = [row["policies"][policy] for row in rows]
    cycles = [int(value["slowest_rank_expert_stage_cycles"]) for value in metrics]
    hbm = [int(value["slowest_rank_expert_weight_hbm_bytes"]) for value in metrics]
    system_hbm = [int(value["system_expert_weight_hbm_bytes"]) for value in metrics]
    collective = [
        int(value["expert_output_collective"]["slowest_rank_bytes"])
        for value in metrics
    ]
    return {
        "observation_count": len(metrics),
        "total_slowest_rank_expert_stage_cycles": sum(cycles),
        "maximum_slowest_rank_expert_stage_cycles": max(cycles),
        "total_slowest_rank_expert_weight_hbm_bytes": sum(hbm),
        "total_system_expert_weight_hbm_bytes": sum(system_hbm),
        "total_slowest_rank_expert_output_collective_bytes": sum(collective),
    }


def _window_totals(
    rows: Sequence[Mapping[str, Any]], policy: str
) -> dict[int, int]:
    totals: dict[int, int] = {}
    for row in rows:
        window = int(row["window_index"])
        totals[window] = totals.get(window, 0) + int(
            row["policies"][policy]["slowest_rank_expert_stage_cycles"]
        )
    return totals


def _comparison(
    aggregates: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    reference: str,
) -> dict[str, Any]:
    candidate = aggregates["frequency_aware_expert_id"]
    baseline = aggregates[reference]
    candidate_windows = _window_totals(rows, "frequency_aware_expert_id")
    baseline_windows = _window_totals(rows, reference)
    all_windows = all(
        candidate_windows[index] <= baseline_windows[index]
        for index in sorted(candidate_windows)
    )
    total_nonregression = (
        candidate["total_slowest_rank_expert_stage_cycles"]
        <= baseline["total_slowest_rank_expert_stage_cycles"]
    )
    maximum_nonregression = (
        candidate["maximum_slowest_rank_expert_stage_cycles"]
        <= baseline["maximum_slowest_rank_expert_stage_cycles"]
    )
    return {
        "reference_policy": reference,
        "held_out_total_cycle_nonregression": total_nonregression,
        "held_out_maximum_cycle_nonregression": maximum_nonregression,
        "every_held_out_window_cycle_nonregression": all_windows,
        "strict_cycle_improvement_observed": (
            total_nonregression
            and maximum_nonregression
            and all_windows
            and candidate["total_slowest_rank_expert_stage_cycles"]
            < baseline["total_slowest_rank_expert_stage_cycles"]
        ),
        "headline_win_claimed": False,
        "candidate_total_cycles": candidate[
            "total_slowest_rank_expert_stage_cycles"
        ],
        "reference_total_cycles": baseline[
            "total_slowest_rank_expert_stage_cycles"
        ],
    }


def build_expert_id_balancing_report(
    trace: RoutingTrace,
    perf: PerfModel,
    config: ExpertIdBalanceConfig,
    *,
    source_binding: Mapping[str, Any],
    hardware_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Learn placements on a prefix and evaluate only disjoint suffix windows."""

    ordered = _audit_trace(trace)
    _validate_source_binding(source_binding, trace)
    _validate_hardware_binding(hardware_binding, perf, config)
    if config.train_step_count >= len(ordered):
        raise ValueError("training split leaves no held-out trace steps")
    held_out_count = len(ordered) - config.train_step_count
    complete_eval_windows, eval_tail = divmod(held_out_count, config.batch_size)
    if complete_eval_windows == 0:
        raise ValueError("held-out split has no complete batch window")

    train = ordered[: config.train_step_count]
    held_out = ordered[config.train_step_count :]
    used_held_out = held_out[: complete_eval_windows * config.batch_size]
    training_sample_ids = sorted({_sample_id_for_step(step) for step in train})
    held_out_sample_ids = sorted(
        {_sample_id_for_step(step) for step in used_held_out}
    )
    sample_overlap = sorted(set(training_sample_ids).intersection(held_out_sample_ids))
    if config.require_sample_disjoint_split and sample_overlap:
        raise ValueError(
            "training and held-out steps share collector sample IDs; choose a "
            "sample-boundary train_step_count"
        )
    placement_rows, selected_owners = _learn_placements(train, perf, config)
    baseline = _baseline_owner(config.tensor_parallel_degree)
    observations: list[dict[str, Any]] = []
    for window_index in range(complete_eval_windows):
        begin = window_index * config.batch_size
        end = begin + config.batch_size
        window = held_out[begin:end]
        global_first = config.train_step_count + begin
        global_last = config.train_step_count + end - 1
        for layer in range(NUM_LAYERS):
            records = [step[layer] for step in window]
            baseline_metrics = _expert_id_metrics(
                records, baseline, perf, config
            )
            balanced_metrics = _expert_id_metrics(
                records, selected_owners[layer], perf, config
            )
            tensor_metrics = _tensor_metrics(records, perf, config)
            observations.append(
                {
                    "window_index": window_index,
                    "first_step_index": global_first,
                    "last_step_index": global_last,
                    "layer": layer,
                    "batch_size": config.batch_size,
                    "route_assignments": config.batch_size * TOP_K,
                    "policies": {
                        "cyclic_expert_id": baseline_metrics,
                        "frequency_aware_expert_id": balanced_metrics,
                        "tensor_parallel_control": tensor_metrics,
                    },
                    "all_policy_routes_conserved": True,
                    "training_steps_used_in_observation": False,
                }
            )

    policy_names = (
        "cyclic_expert_id",
        "frequency_aware_expert_id",
        "tensor_parallel_control",
    )
    aggregates = {
        policy: _aggregate_policy(observations, policy) for policy in policy_names
    }
    vs_cyclic = _comparison(aggregates, observations, "cyclic_expert_id")
    vs_tensor = _comparison(aggregates, observations, "tensor_parallel_control")
    capacity = NUM_EXPERTS // config.tensor_parallel_degree
    expert_id_resident = (
        NUM_LAYERS
        * capacity
        * _expert_bytes(config, config.expert_intermediate_size)
    )
    tensor_widths = _balanced_widths(
        config.expert_intermediate_size, config.tensor_parallel_degree
    )
    tensor_resident = [
        NUM_LAYERS * NUM_EXPERTS * _expert_bytes(config, width)
        for width in tensor_widths
    ]
    trace_hash = trace_content_hash(trace)
    config_body = asdict(config)
    config_hash = canonical_hash(config_body)
    used_eval_count = complete_eval_windows * config.batch_size
    body = {
        "schema": REPORT_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": trace_hash,
        "source_binding": dict(source_binding),
        "hardware_binding": dict(hardware_binding),
        "study_config": config_body,
        "study_config_content_hash": config_hash,
        "precision_binding": {
            "label": config.precision_label,
            "ffn_element_bits": config.ffn_element_bits,
            "ffn_effective_bits": config.ffn_effective_bits,
            "mx_block_size": config.mx_block_size,
            "descriptor_content_hash": canonical_hash(
                {
                    "label": config.precision_label,
                    "ffn_element_bits": config.ffn_element_bits,
                    "ffn_effective_bits": config.ffn_effective_bits,
                    "mx_block_size": config.mx_block_size,
                }
            ),
        },
        "topology_binding": {
            "tensor_parallel_degree": config.tensor_parallel_degree,
            "kv_parallel_degree": config.kv_parallel_degree,
            "chip_count": (
                config.tensor_parallel_degree * config.kv_parallel_degree
            ),
            "expert_id_mapping": EXPERT_ID_MAPPING,
            "tensor_control_mapping": TENSOR_MAPPING,
            "router_execution": "replicated_on_every_tp_rank",
            "hidden_state_before_router": "replicated_after_attention_tp_allreduce",
            "expert_id_filter": "rank_local_by_owned_id",
            "source_hidden_dispatch_required": False,
            "source_hidden_dispatch_bytes": 0,
            "expert_output_collective": "one_ring_allreduce_per_layer",
            "link_generation": config.link_generation,
            "tp_link_ports": config.tp_link_ports,
            "tp_link_bandwidth_bytes_per_s": (
                config.tp_link_bandwidth_bytes_per_s
            ),
        },
        "split": {
            "policy": SPLIT_POLICY,
            "window_policy": WINDOW_POLICY,
            "training": {
                "first_step_index": 0,
                "last_step_index": config.train_step_count - 1,
                "step_count": config.train_step_count,
                "complete_window_count": config.train_step_count
                // config.batch_size,
                "step_indices_content_hash": canonical_hash(
                    {"indices": list(range(config.train_step_count))}
                ),
            },
            "held_out": {
                "first_step_index": config.train_step_count,
                "last_used_step_index": config.train_step_count
                + used_eval_count
                - 1,
                "used_step_count": used_eval_count,
                "complete_window_count": complete_eval_windows,
                "dropped_tail_step_count": eval_tail,
                "step_indices_content_hash": canonical_hash(
                    {
                        "indices": list(
                            range(
                                config.train_step_count,
                                config.train_step_count + used_eval_count,
                            )
                        )
                    }
                ),
            },
            "train_eval_overlap_count": 0,
            "training_sample_ids": training_sample_ids,
            "held_out_sample_ids": held_out_sample_ids,
            "sample_id_overlap_count": len(sample_overlap),
            "sample_id_overlap": sample_overlap,
            "sample_disjoint_required": config.require_sample_disjoint_split,
            "held_out_influences_placement": False,
        },
        "placement": {
            "algorithm": PLACEMENT_ALGORITHM,
            "capacity_per_rank": capacity,
            "baseline_policy": "expert_id_mod_tp_rank",
            "training_trace_semantics": (
                "one_logical_route_sample_per_step;kvp_copies_are_not_resampled"
            ),
            "layers": placement_rows,
            "all_layers_capacity_exact": True,
            "all_layers_training_nonregression": True,
        },
        "resident_expert_weights": {
            "scope": "all_48_layers_all_128_experts",
            "expert_id_slowest_rank_bytes": expert_id_resident,
            "expert_id_system_bytes": (
                expert_id_resident
                * config.tensor_parallel_degree
                * config.kv_parallel_degree
            ),
            "tensor_parallel_bytes_by_rank": tensor_resident,
            "tensor_parallel_slowest_rank_bytes": max(tensor_resident),
            "tensor_parallel_system_bytes": (
                sum(tensor_resident) * config.kv_parallel_degree
            ),
            "all_experts_remain_resident": True,
        },
        "capacity_crossover_request": {
            "context_tokens": config.context_tokens,
            "batch_size": config.batch_size,
            "hbm_capacity_bytes_per_chip": config.hbm_capacity_bytes_per_chip,
            "runtime_hbm_reserve_bytes_per_chip": (
                config.runtime_hbm_reserve_bytes_per_chip
            ),
            "kv_layout": config.kv_layout,
            "candidate_kvp_degrees": [1, 2, 4],
            "status": "requires_full_precision_window_body_reprice",
            "publication_rankable": False,
        },
        "held_out_evaluation": {
            "observation_count": len(observations),
            "observations": observations,
            "assignment_conservation": {
                "logical_assignments": (
                    len(observations) * config.batch_size * TOP_K
                ),
                "physical_whole_expert_assignments_across_kvp": (
                    len(observations)
                    * config.batch_size
                    * TOP_K
                    * config.kv_parallel_degree
                ),
                "kvp_route_semantics": (
                    "identical_route_replica_not_independent_placement_sample"
                ),
                "conserved": True,
            },
            "policy_aggregates": aggregates,
            "frequency_aware_vs_cyclic": vs_cyclic,
            "frequency_aware_vs_tensor_parallel": vs_tensor,
            "held_out_win_required_for_any_future_promotion": True,
            "headline_win_claimed": False,
        },
        "reprice_receipt": {
            "schema": "plena-qwen3-moe-expert-id-isolated-reprice-receipt/v1",
            "timing_source": "PerfModel_exact_rank_local_trace_histograms",
            "timing_scope": "routed_expert_stage_plus_exposed_output_collective",
            "full_tpot_repriced": False,
            "source_hidden_dispatch_bytes": 0,
            "expert_output_allreduces_per_layer": int(
                config.tensor_parallel_degree > 1
            ),
            "compiler_validated": False,
            "emulator_validated": False,
            "rtl_validated": False,
            "power_calibrated": False,
            "timing_selection_allowed": False,
        },
        "classification": {
            "evidence": "measured_trace_held_out_analytic_projection",
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
            "headline_eligible": False,
            "blockers": [
                "batch_windows_are_post_hoc_groups_not_measured_batched_forwards",
                "expert_id_mapping_lacks_full_compiler_and_emulator_receipts",
                "expert_id_mapping_lacks_rtl_timing_receipt",
                "full_decode_tpot_is_not_repriced_by_this_isolated_artifact",
                "moe_dynamic_power_lacks_matched_calibration",
            ],
        },
    }
    return _hashed_body(body)


def validate_expert_id_balancing_report(report: Mapping[str, Any]) -> None:
    """Validate immutable bindings and fail-closed structural invariants."""

    if not isinstance(report, Mapping):
        raise ValueError("expert-ID report must be an object")
    body = dict(report)
    observed = body.pop("content_hash", None)
    if observed != canonical_hash(body):
        raise ValueError("expert-ID report content hash mismatch")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported expert-ID report schema")
    if (
        report.get("model_id") != MODEL_ID
        or report.get("model_revision") != MODEL_REVISION
    ):
        raise ValueError("expert-ID report model identity differs")
    classification = report.get("classification")
    if not isinstance(classification, Mapping) or any(
        classification.get(field) is not False
        for field in (
            "publication_rankable",
            "hardware_rankable",
            "selection_eligible",
            "headline_eligible",
        )
    ):
        raise ValueError("expert-ID report must remain fail-closed")
    evaluation = report.get("held_out_evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("headline_win_claimed") is not False
    ):
        raise ValueError("expert-ID report must not claim a headline win")
    split = report.get("split")
    if (
        not isinstance(split, Mapping)
        or split.get("train_eval_overlap_count") != 0
        or split.get("sample_id_overlap_count") != 0
        or split.get("sample_id_overlap") != []
        or split.get("sample_disjoint_required") is not True
        or split.get("held_out_influences_placement") is not False
    ):
        raise ValueError("expert-ID train and held-out splits overlap")
    topology = report.get("topology_binding")
    if (
        not isinstance(topology, Mapping)
        or topology.get("source_hidden_dispatch_required") is not False
        or topology.get("source_hidden_dispatch_bytes") != 0
        or topology.get("expert_output_collective")
        != "one_ring_allreduce_per_layer"
    ):
        raise ValueError("expert-ID report transport mapping differs")
    placement = report.get("placement")
    if (
        not isinstance(placement, Mapping)
        or placement.get("all_layers_capacity_exact") is not True
        or placement.get("all_layers_training_nonregression") is not True
        or len(placement.get("layers", [])) != NUM_LAYERS
    ):
        raise ValueError("expert-ID placement proof is incomplete")
    config_value = report.get("study_config")
    if not isinstance(config_value, Mapping):
        raise ValueError("expert-ID study config is missing")
    config = ExpertIdBalanceConfig(**config_value)
    if canonical_hash(config_value) != report.get("study_config_content_hash"):
        raise ValueError("expert-ID study config hash differs")
    capacity = NUM_EXPERTS // config.tensor_parallel_degree
    for row in placement["layers"]:
        owners = row.get("selected_expert_owner_by_id")
        if (
            not isinstance(owners, list)
            or len(owners) != NUM_EXPERTS
            or any(owners.count(rank) != capacity for rank in range(config.tensor_parallel_degree))
            or row.get("held_out_steps_used_for_placement") is not False
        ):
            raise ValueError("expert-ID layer ownership violates capacity or split")
    observations = evaluation.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("expert-ID report has no held-out observations")
    expected = config.batch_size * TOP_K
    for row in observations:
        if (
            row.get("route_assignments") != expected
            or row.get("training_steps_used_in_observation") is not False
            or row.get("all_policy_routes_conserved") is not True
        ):
            raise ValueError("held-out route observation does not conserve")
        policies = row.get("policies")
        if not isinstance(policies, Mapping) or set(policies) != {
            "cyclic_expert_id",
            "frequency_aware_expert_id",
            "tensor_parallel_control",
        }:
            raise ValueError("held-out policy controls are incomplete")
        for name in ("cyclic_expert_id", "frequency_aware_expert_id"):
            metrics = policies[name]
            if (
                sum(metrics.get("assignment_count_by_rank", [])) != expected
                or sum(
                    metrics.get(
                        "physical_assignment_count_by_rank_across_kvp", []
                    )
                )
                != expected * config.kv_parallel_degree
                or metrics.get("source_hidden_dispatch_bytes") != 0
                or metrics.get("route_assignment_conservation", {}).get("conserved")
                is not True
            ):
                raise ValueError("expert-ID held-out metrics do not conserve")
    reprice = report.get("reprice_receipt")
    if not isinstance(reprice, Mapping) or any(
        reprice.get(field) is not False
        for field in (
            "full_tpot_repriced",
            "compiler_validated",
            "emulator_validated",
            "rtl_validated",
            "power_calibrated",
            "timing_selection_allowed",
        )
    ):
        raise ValueError("expert-ID reprice receipt is not fail-closed")


def audit_expert_id_balancing_report(
    report: Mapping[str, Any],
    trace: RoutingTrace,
    perf: PerfModel,
    config: ExpertIdBalanceConfig,
    *,
    source_binding: Mapping[str, Any],
    hardware_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the complete artifact and reject any source or metric drift."""

    validate_expert_id_balancing_report(report)
    expected = build_expert_id_balancing_report(
        trace,
        perf,
        config,
        source_binding=source_binding,
        hardware_binding=hardware_binding,
    )
    if dict(report) != expected:
        raise ValueError("expert-ID report differs from trace-exact recomputation")
    return expected


def build_expert_id_window_overlay(
    report: Mapping[str, Any],
    window_index: int,
    *,
    policy: str = "frequency_aware_expert_id",
    report_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a layer-exact adapter for one measured held-out batch window.

    The adapter deliberately retains 48 independent ownership/load records.
    Consumers must request a layer explicitly; collapsing layer-wise maxima
    into one fabricated global route is not supported.
    """

    validate_expert_id_balancing_report(report)
    if policy not in {"cyclic_expert_id", "frequency_aware_expert_id"}:
        raise ValueError("window overlay supports expert-ID policies only")
    _positive_int(window_index, "window_index", allow_zero=True)
    observations = [
        row
        for row in report["held_out_evaluation"]["observations"]
        if int(row["window_index"]) == window_index
    ]
    observations.sort(key=lambda row: int(row["layer"]))
    if len(observations) != NUM_LAYERS or [
        int(row["layer"]) for row in observations
    ] != list(range(NUM_LAYERS)):
        raise ValueError("selected held-out window lacks exact 48-layer coverage")
    placement = report["placement"]["layers"]
    config = report["study_config"]
    tp = int(config["tensor_parallel_degree"])
    layers: list[dict[str, Any]] = []
    for layer, observation in enumerate(observations):
        metrics = observation["policies"][policy]
        owners = (
            placement[layer]["baseline_expert_owner_by_id"]
            if policy == "cyclic_expert_id"
            else placement[layer]["selected_expert_owner_by_id"]
        )
        active = list(metrics["active_expert_count_by_rank"])
        if len(active) != tp or sum(active) != len(
            {
                expert
                for values in metrics["active_expert_ids_by_rank"]
                for expert in values
            }
        ):
            raise AssertionError("window overlay active-expert counts differ")
        layers.append(
            {
                "layer": layer,
                "expert_owner_by_id": list(owners),
                "active_experts_per_rank": active,
                "active_expert_ids_by_rank": [
                    list(values) for values in metrics["active_expert_ids_by_rank"]
                ],
                "assignment_count_by_rank": list(
                    metrics["assignment_count_by_rank"]
                ),
                "physical_assignment_count_by_rank_across_kvp": list(
                    metrics["physical_assignment_count_by_rank_across_kvp"]
                ),
                "expert_token_count_histogram_by_rank": [
                    dict(value)
                    for value in metrics["expert_token_count_histogram_by_rank"]
                ],
                "expert_stage_cycles_by_rank": list(
                    metrics["expert_stage_cycles_by_rank"]
                ),
                "matrix_instruction_histogram_by_rank": [
                    dict(value)
                    for value in metrics["matrix_instruction_histogram_by_rank"]
                ],
                "auxiliary_instruction_histogram_by_rank": [
                    dict(value)
                    for value in metrics["auxiliary_instruction_histogram_by_rank"]
                ],
                "matrix_cycles_by_rank": list(metrics["matrix_cycles_by_rank"]),
                "activation_cycles_by_rank": list(
                    metrics["activation_cycles_by_rank"]
                ),
                "auxiliary_cycles_by_rank": list(
                    metrics["auxiliary_cycles_by_rank"]
                ),
                "expert_weight_hbm_bytes_by_rank": list(
                    metrics["expert_weight_hbm_bytes_by_rank"]
                ),
                "unique_active_experts": sum(active),
                "route_assignments": int(observation["route_assignments"]),
                "physical_whole_expert_assignments_across_kvp": int(
                    metrics["route_assignment_conservation"][
                        "physical_whole_expert_assignments_across_kvp"
                    ]
                ),
                "source_hidden_dispatch_bytes": 0,
                "expert_output_collective": dict(
                    metrics["expert_output_collective"]
                ),
                "assignment_conserved": True,
            }
        )
    first = int(observations[0]["first_step_index"])
    last = int(observations[0]["last_step_index"])
    if any(
        int(row["first_step_index"]) != first
        or int(row["last_step_index"]) != last
        for row in observations
    ):
        raise AssertionError("held-out window step bounds differ across layers")

    receipt: dict[str, Any] | None = None
    blockers = [
        "layer_exact_overlay_is_an_isolated_analytic_projection",
        "full_decode_tpot_integration_requires_explicit_consumer_validation",
        "compiler_emulator_rtl_and_power_receipts_missing",
    ]
    if report_receipt is not None:
        path = Path(str(report_receipt.get("path", "")))
        if (
            not path.is_absolute()
            or not path.is_file()
            or file_hash(path) != report_receipt.get("sha256")
            or report_receipt.get("content_hash") != report["content_hash"]
        ):
            raise ValueError("window overlay report receipt is missing or changed")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded != dict(report):
            raise ValueError("window overlay report file differs from its object")
        receipt = dict(report_receipt)
    else:
        blockers.append("report_file_receipt_missing_in_memory_adapter_only")

    total_cycles = sum(
        max(int(value) for value in layer["expert_stage_cycles_by_rank"])
        for layer in layers
    )
    slowest_hbm_per_layer = sum(
        max(int(value) for value in layer["expert_weight_hbm_bytes_by_rank"])
        for layer in layers
    )
    body = {
        "schema": WINDOW_OVERLAY_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": report["trace_content_hash"],
        "report_content_hash": report["content_hash"],
        "report_receipt": receipt,
        "study_config_content_hash": report["study_config_content_hash"],
        "study_config": dict(config),
        "policy": policy,
        "window_index": window_index,
        "first_step_index": first,
        "last_step_index": last,
        "layer_count": NUM_LAYERS,
        "layers": layers,
        "full_decoder_expert_projection": {
            "sum_of_per_layer_slowest_rank_expert_stage_cycles": total_cycles,
            "sum_of_per_layer_slowest_rank_expert_stage_time_s": (
                total_cycles / int(config["frequency_hz"])
            ),
            "sum_of_per_layer_slowest_rank_expert_weight_hbm_bytes": (
                slowest_hbm_per_layer
            ),
            "source_hidden_dispatch_bytes": 0,
            "output_collective_count": NUM_LAYERS
            * int(int(config["tensor_parallel_degree"]) > 1),
            "full_tpot_repriced": False,
        },
        "adapter_contract": {
            "consumer_api": "expert_id_body_inputs_for_layer",
            "layer_specific_inputs_required": True,
            "global_active_count_collapse_allowed": False,
            "expert_owner_policy": "explicit_expert_owner_by_id_per_layer",
            "route_filter_transport": "zero_byte_rank_local_filter",
            "expert_output_collective": "one_ring_allreduce_per_layer",
        },
        "classification": {
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
            "timing_selection_allowed": False,
            "blockers": blockers,
        },
    }
    return _hashed_body(body)


def validate_expert_id_window_overlay(
    overlay: Mapping[str, Any],
    *,
    report: Mapping[str, Any] | None = None,
) -> None:
    """Validate a layer-exact overlay and its optional source report."""

    if not isinstance(overlay, Mapping):
        raise ValueError("expert-ID window overlay must be an object")
    body = dict(overlay)
    observed = body.pop("content_hash", None)
    if observed != canonical_hash(body):
        raise ValueError("expert-ID window overlay content hash mismatch")
    if overlay.get("schema") != WINDOW_OVERLAY_SCHEMA:
        raise ValueError("unsupported expert-ID window overlay schema")
    classification = overlay.get("classification")
    if not isinstance(classification, Mapping) or any(
        classification.get(field) is not False
        for field in (
            "publication_rankable",
            "hardware_rankable",
            "selection_eligible",
            "timing_selection_allowed",
        )
    ):
        raise ValueError("expert-ID window overlay must remain fail-closed")
    adapter = overlay.get("adapter_contract")
    if (
        not isinstance(adapter, Mapping)
        or adapter.get("layer_specific_inputs_required") is not True
        or adapter.get("global_active_count_collapse_allowed") is not False
    ):
        raise ValueError("expert-ID window adapter permits an unsafe collapse")
    config = ExpertIdBalanceConfig(**overlay["study_config"])
    layers = overlay.get("layers")
    if (
        not isinstance(layers, list)
        or len(layers) != NUM_LAYERS
        or [int(row.get("layer", -1)) for row in layers] != list(range(NUM_LAYERS))
    ):
        raise ValueError("expert-ID window overlay lacks exact layer coverage")
    capacity = NUM_EXPERTS // config.tensor_parallel_degree
    expected_assignments = config.batch_size * TOP_K
    for row in layers:
        owners = row.get("expert_owner_by_id")
        active = row.get("active_experts_per_rank")
        assignments = row.get("assignment_count_by_rank")
        active_ids = row.get("active_expert_ids_by_rank")
        physical_assignments = row.get(
            "physical_assignment_count_by_rank_across_kvp"
        )
        histograms = row.get("expert_token_count_histogram_by_rank")
        if (
            not isinstance(owners, list)
            or len(owners) != NUM_EXPERTS
            or any(
                owners.count(rank) != capacity
                for rank in range(config.tensor_parallel_degree)
            )
            or not isinstance(active, list)
            or len(active) != config.tensor_parallel_degree
            or not isinstance(assignments, list)
            or sum(assignments) != expected_assignments
            or not isinstance(physical_assignments, list)
            or len(physical_assignments) != config.tensor_parallel_degree
            or physical_assignments
            != [value * config.kv_parallel_degree for value in assignments]
            or sum(physical_assignments)
            != expected_assignments * config.kv_parallel_degree
            or row.get("physical_whole_expert_assignments_across_kvp")
            != expected_assignments * config.kv_parallel_degree
            or row.get("route_assignments") != expected_assignments
            or row.get("source_hidden_dispatch_bytes") != 0
            or row.get("assignment_conserved") is not True
            or not isinstance(active_ids, list)
            or [len(values) for values in active_ids] != active
            or any(len(values) != len(set(values)) for values in active_ids)
            or len(
                {
                    int(expert)
                    for values in active_ids
                    for expert in values
                }
            )
            != sum(active)
            or row.get("unique_active_experts") != sum(active)
            or not isinstance(histograms, list)
            or any(
                len(row.get(field, [])) != config.tensor_parallel_degree
                for field in (
                    "expert_token_count_histogram_by_rank",
                    "expert_stage_cycles_by_rank",
                    "matrix_instruction_histogram_by_rank",
                    "auxiliary_instruction_histogram_by_rank",
                    "matrix_cycles_by_rank",
                    "activation_cycles_by_rank",
                    "auxiliary_cycles_by_rank",
                    "expert_weight_hbm_bytes_by_rank",
                )
            )
        ):
            raise ValueError("expert-ID layer adapter violates ownership or routes")
        for rank, values in enumerate(active_ids):
            if any(owners[int(expert)] != rank for expert in values):
                raise ValueError("active expert is assigned to the wrong owner")
            histogram = histograms[rank]
            if (
                not isinstance(histogram, Mapping)
                or sum(int(count) for count in histogram.values()) != active[rank]
                or sum(
                    int(token_count) * int(count)
                    for token_count, count in histogram.items()
                )
                != assignments[rank]
                or any(
                    isinstance(token_count, bool)
                    or isinstance(count, bool)
                    or int(token_count) <= 0
                    or int(count) <= 0
                    or str(int(token_count)) != str(token_count)
                    for token_count, count in histogram.items()
                )
            ):
                raise ValueError("expert-ID rank histogram does not conserve")
    receipt = overlay.get("report_receipt")
    if receipt is not None:
        path = Path(str(receipt.get("path", "")))
        if (
            not path.is_absolute()
            or not path.is_file()
            or file_hash(path) != receipt.get("sha256")
            or receipt.get("content_hash") != overlay.get("report_content_hash")
        ):
            raise ValueError("expert-ID overlay report receipt changed")
    if report is not None:
        validate_expert_id_balancing_report(report)
        if report.get("content_hash") != overlay.get("report_content_hash"):
            raise ValueError("expert-ID overlay binds a different report")
        expected = build_expert_id_window_overlay(
            report,
            int(overlay["window_index"]),
            policy=str(overlay["policy"]),
            report_receipt=receipt,
        )
        if dict(overlay) != expected:
            raise ValueError("expert-ID window overlay differs from its report")


def expert_id_body_inputs_for_layer(
    overlay: Mapping[str, Any], layer: int
) -> dict[str, Any]:
    """Return safe physical-layout and exact-timing inputs for one layer."""

    validate_expert_id_window_overlay(overlay)
    _positive_int(layer, "layer", allow_zero=True)
    if layer >= NUM_LAYERS:
        raise ValueError("layer is outside the Qwen3 decoder")
    row = overlay["layers"][layer]
    return {
        "expert_owner_by_id": tuple(int(value) for value in row["expert_owner_by_id"]),
        "active_experts_per_rank": tuple(
            int(value) for value in row["active_experts_per_rank"]
        ),
        "physical_assignment_count_by_rank_across_kvp": tuple(
            int(value)
            for value in row["physical_assignment_count_by_rank_across_kvp"]
        ),
        "unique_experts": int(row["unique_active_experts"]),
        "expert_token_count_histogram_by_rank": tuple(
            dict(value) for value in row["expert_token_count_histogram_by_rank"]
        ),
        "expert_stage_cycles_by_rank": tuple(
            int(value) for value in row["expert_stage_cycles_by_rank"]
        ),
        "matrix_instruction_histogram_by_rank": tuple(
            dict(value) for value in row["matrix_instruction_histogram_by_rank"]
        ),
        "auxiliary_instruction_histogram_by_rank": tuple(
            dict(value) for value in row["auxiliary_instruction_histogram_by_rank"]
        ),
        "matrix_cycles_by_rank": tuple(
            int(value) for value in row["matrix_cycles_by_rank"]
        ),
        "activation_cycles_by_rank": tuple(
            int(value) for value in row["activation_cycles_by_rank"]
        ),
        "auxiliary_cycles_by_rank": tuple(
            int(value) for value in row["auxiliary_cycles_by_rank"]
        ),
        "expert_weight_hbm_bytes_by_rank": tuple(
            int(value) for value in row["expert_weight_hbm_bytes_by_rank"]
        ),
        "source_hidden_dispatch_bytes": 0,
        "expert_output_collective": dict(row["expert_output_collective"]),
        "provenance": {
            "schema": WINDOW_OVERLAY_SCHEMA,
            "overlay_content_hash": overlay["content_hash"],
            "trace_content_hash": overlay["trace_content_hash"],
            "window_index": overlay["window_index"],
            "layer": layer,
            "timing_selection_allowed": False,
        },
    }


def reprice_expert_id_window_body(
    overlay: Mapping[str, Any],
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
) -> dict[str, Any]:
    """Drive rank-local body layout and PerfModel once for each exact layer.

    This function is intentionally outside the normal DSE path.  It validates
    one content-addressed held-out overlay, invokes the physical body hook for
    each of its 48 layer-specific ownership/load records, and recomputes each
    rank's expert timing from the bound histogram.  It does not replace every
    layer with a global maximum and does not produce a full-TPOT result.
    """

    validate_expert_id_window_overlay(overlay)
    config = ExpertIdBalanceConfig(**overlay["study_config"])
    if (
        int(perf.mlen) != config.mlen
        or int(perf.blen) != config.blen
        or int(perf.vlen) != config.vlen
    ):
        raise ValueError("PerfModel geometry differs from the window overlay")
    expected_dims = {
        "hidden": HIDDEN_SIZE,
        "inter": EXPERT_INTERMEDIATE_SIZE,
        "layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "experts_per_token": TOP_K,
    }
    for field, expected in expected_dims.items():
        if int(dims.get(field, -1)) != expected:
            raise ValueError(f"decoder dimension {field} differs from the overlay")
    for field in ("heads", "kv_heads", "head_dim", "vocab"):
        if int(dims.get(field, 0)) <= 0:
            raise ValueError(f"decoder dimension {field} is missing")
    if (
        int(precision.get("ffn_elem", -1)) != config.ffn_element_bits
        or not math.isclose(
            float(precision.get("ffn_bits", -1.0)),
            config.ffn_effective_bits,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or int(precision.get("block_size", -1)) != config.mx_block_size
    ):
        raise ValueError("FFN precision differs from the window overlay")
    if config.mx_block_size != 8:
        raise ValueError("physical body reprice currently requires block-8 MX")

    one_layer_dims = dict(dims) | {"layers": 1}
    layer_rows: list[dict[str, Any]] = []
    total_cycles = 0
    total_slowest_streamed = 0
    total_system_streamed = 0
    resident_by_layer_slowest = 0
    resident_by_layer_system = 0
    total_physical_assignments = 0
    total_collective_slowest = 0
    total_collective_system = 0
    for layer in range(NUM_LAYERS):
        inputs = expert_id_body_inputs_for_layer(overlay, layer)
        layout = build_body_weight_physical_layout(
            one_layer_dims,
            precision,
            mlen=config.mlen,
            tp=config.tensor_parallel_degree,
            kvp=config.kv_parallel_degree,
            batch=config.batch_size,
            unique_experts=int(inputs["unique_experts"]),
            expert_parallel_mode=EXPERT_ID_PARALLEL,
            active_experts_per_rank=inputs["active_experts_per_rank"],
            expert_owner_by_id=inputs["expert_owner_by_id"],
            include_lm_head=False,
            alignment_bytes=config.weight_alignment_bytes,
        )
        if layout.provenance.get("expert_route_assignment_exact") is not True:
            raise AssertionError("body layout did not accept exact route assignment")
        recomputed_cycles: list[int] = []
        assignment_counts = overlay["layers"][layer]["assignment_count_by_rank"]
        for rank, histogram in enumerate(
            inputs["expert_token_count_histogram_by_rank"]
        ):
            timing = perf.moe_decode_expert_timing_from_histogram(
                config.hidden_size,
                config.expert_intermediate_size,
                histogram,
                owned_experts=NUM_EXPERTS // config.tensor_parallel_degree,
                expected_route_assignments=int(assignment_counts[rank]),
                source="held_out_window_overlay_recomputed_rank_local_histogram",
            )
            recomputed_cycles.append(int(timing["expert_stage_cycles"]))
        if tuple(recomputed_cycles) != inputs["expert_stage_cycles_by_rank"]:
            raise ValueError("overlay expert timing differs from PerfModel recomputation")
        overlay_streamed = inputs["expert_weight_hbm_bytes_by_rank"]
        layout_slowest_streamed = (
            layout.slowest_rank.ffn_streamed.total_aligned
        )
        layout_system_streamed = layout.system.ffn_streamed.total_aligned
        if (
            layout_slowest_streamed != max(overlay_streamed)
            or layout_system_streamed
            != sum(overlay_streamed) * config.kv_parallel_degree
        ):
            raise ValueError("overlay expert traffic differs from body layout")
        collective = inputs["expert_output_collective"]
        layer_slowest_cycles = max(recomputed_cycles)
        total_cycles += layer_slowest_cycles
        total_slowest_streamed += layout_slowest_streamed
        total_system_streamed += layout_system_streamed
        resident_by_layer_slowest += (
            layout.slowest_rank.ffn_resident.total_aligned
        )
        resident_by_layer_system += layout.system.ffn_resident.total_aligned
        total_physical_assignments += sum(
            inputs["physical_assignment_count_by_rank_across_kvp"]
        )
        total_collective_slowest += int(collective["slowest_rank_bytes"])
        total_collective_system += int(collective["system_bytes"])
        layer_rows.append(
            {
                "layer": layer,
                "active_experts_per_rank": list(
                    inputs["active_experts_per_rank"]
                ),
                "expert_stage_cycles_by_rank": recomputed_cycles,
                "matrix_instruction_histogram_by_rank": [
                    dict(value)
                    for value in inputs["matrix_instruction_histogram_by_rank"]
                ],
                "auxiliary_instruction_histogram_by_rank": [
                    dict(value)
                    for value in inputs["auxiliary_instruction_histogram_by_rank"]
                ],
                "slowest_rank_expert_stage_cycles": layer_slowest_cycles,
                "slowest_rank_expert_streamed_bytes": layout_slowest_streamed,
                "system_expert_streamed_bytes": layout_system_streamed,
                "slowest_rank_expert_resident_bytes": (
                    layout.slowest_rank.ffn_resident.total_aligned
                ),
                "system_expert_resident_bytes": (
                    layout.system.ffn_resident.total_aligned
                ),
                "physical_whole_expert_assignments_across_kvp": sum(
                    inputs["physical_assignment_count_by_rank_across_kvp"]
                ),
                "source_hidden_dispatch_bytes": 0,
                "expert_output_collective_slowest_rank_bytes": int(
                    collective["slowest_rank_bytes"]
                ),
                "expert_output_collective_system_bytes": int(
                    collective["system_bytes"]
                ),
                "body_layout_route_assignment_exact": True,
            }
        )

    expected_physical = (
        NUM_LAYERS
        * config.batch_size
        * TOP_K
        * config.kv_parallel_degree
    )
    if total_physical_assignments != expected_physical:
        raise AssertionError("window body reprice physical assignments differ")

    capacity_precision = dict(precision)
    capacity_precision.setdefault(
        "head_elem", int(capacity_precision.get("attn_elem", -1))
    )
    capacity_precision.setdefault(
        "head_bits", float(capacity_precision.get("attn_bits", -1.0))
    )
    if capacity_precision.get("lm_head_quantized") is not True:
        raise ValueError("capacity crossover requires the canonical local MX head")
    layer_zero = expert_id_body_inputs_for_layer(overlay, 0)
    expert_id_capacity_layout = build_body_weight_physical_layout(
        dims,
        capacity_precision,
        mlen=config.mlen,
        tp=config.tensor_parallel_degree,
        kvp=1,
        batch=config.batch_size,
        unique_experts=int(layer_zero["unique_experts"]),
        expert_parallel_mode=EXPERT_ID_PARALLEL,
        active_experts_per_rank=layer_zero["active_experts_per_rank"],
        expert_owner_by_id=layer_zero["expert_owner_by_id"],
        include_lm_head=True,
        alignment_bytes=config.weight_alignment_bytes,
    )
    tensor_capacity_layout = build_body_weight_physical_layout(
        dims,
        capacity_precision,
        mlen=config.mlen,
        tp=config.tensor_parallel_degree,
        kvp=1,
        batch=config.batch_size,
        unique_experts=int(layer_zero["unique_experts"]),
        expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
        include_lm_head=True,
        alignment_bytes=config.weight_alignment_bytes,
    )
    local_kv_heads = int(dims["kv_heads"]) // config.tensor_parallel_degree
    key = traffic_from_precision(
        kv_heads=local_kv_heads,
        head_dim=int(dims["head_dim"]),
        mlen=config.mlen,
        element_bits=int(precision.get("key_elem", precision.get("kv_elem", -1))),
        effective_bits=float(
            precision.get("key_bits", precision.get("kv_bits", -1.0))
        ),
        block_size=config.mx_block_size,
    )
    value = traffic_from_precision(
        kv_heads=local_kv_heads,
        head_dim=int(dims["head_dim"]),
        mlen=config.mlen,
        element_bits=int(
            precision.get("value_elem", precision.get("kv_elem", -1))
        ),
        effective_bits=float(
            precision.get("value_bits", precision.get("kv_bits", -1.0))
        ),
        block_size=config.mx_block_size,
    )
    kv_bytes_per_local_tp_rank_token_layer = key.storage_bytes(
        config.kv_layout
    ) + value.storage_bytes(config.kv_layout)

    def capacity_rows(layout: Any) -> list[dict[str, Any]]:
        rank_weights = layout.slowest_rank.resident.total_aligned
        group_weights = layout.tensor_parallel_group.resident.total_aligned
        rows = []
        for kvp in (1, 2, 4):
            rank_kv = (
                config.batch_size
                * NUM_LAYERS
                * math.ceil(config.context_tokens / kvp)
                * kv_bytes_per_local_tp_rank_token_layer
            )
            system_kv = (
                config.batch_size
                * NUM_LAYERS
                * config.context_tokens
                * kv_bytes_per_local_tp_rank_token_layer
                * config.tensor_parallel_degree
            )
            required = (
                rank_weights
                + rank_kv
                + config.runtime_hbm_reserve_bytes_per_chip
            )
            rows.append(
                {
                    "kv_parallel_degree": kvp,
                    "chip_count": config.tensor_parallel_degree * kvp,
                    "slowest_rank_weight_bytes": rank_weights,
                    "slowest_rank_kv_bytes": rank_kv,
                    "runtime_reserve_bytes": (
                        config.runtime_hbm_reserve_bytes_per_chip
                    ),
                    "slowest_rank_required_bytes": required,
                    "capacity_bytes_per_chip": (
                        config.hbm_capacity_bytes_per_chip
                    ),
                    "capacity_margin_bytes": (
                        config.hbm_capacity_bytes_per_chip - required
                    ),
                    "feasible": required <= config.hbm_capacity_bytes_per_chip,
                    "system_weight_bytes": group_weights * kvp,
                    "system_kv_bytes": system_kv,
                    "system_runtime_reserve_bytes": (
                        config.runtime_hbm_reserve_bytes_per_chip
                        * config.tensor_parallel_degree
                        * kvp
                    ),
                }
            )
        return rows

    expert_id_capacity_rows = capacity_rows(expert_id_capacity_layout)
    tensor_capacity_rows = capacity_rows(tensor_capacity_layout)

    def minimum_feasible(rows: Sequence[Mapping[str, Any]]) -> int | None:
        values = [
            int(row["kv_parallel_degree"])
            for row in rows
            if row["feasible"] is True
        ]
        return min(values) if values else None

    expert_id_minimum_kvp = minimum_feasible(expert_id_capacity_rows)
    tensor_minimum_kvp = minimum_feasible(tensor_capacity_rows)
    capacity_crossover = {
        "context_tokens": config.context_tokens,
        "batch_size": config.batch_size,
        "tensor_parallel_degree": config.tensor_parallel_degree,
        "hbm_capacity_bytes_per_chip": config.hbm_capacity_bytes_per_chip,
        "runtime_hbm_reserve_bytes_per_chip": (
            config.runtime_hbm_reserve_bytes_per_chip
        ),
        "kv_layout": config.kv_layout,
        "kv_bytes_per_local_tp_rank_token_layer": (
            kv_bytes_per_local_tp_rank_token_layer
        ),
        "resident_owner_identity_semantics": (
            "all_experts_have_identical_shapes_and_every_layer_owns_exactly_"
            "128_div_tp_per_rank;layer0_owner_ids_are_a_storage-equivalent_"
            "representative_only"
        ),
        "route_timing_still_uses_all_48_layer_specific_histograms": True,
        "expert_id_parallel": {
            "rows": expert_id_capacity_rows,
            "minimum_feasible_kv_parallel_degree": expert_id_minimum_kvp,
        },
        "tensor_parallel_control": {
            "rows": tensor_capacity_rows,
            "minimum_feasible_kv_parallel_degree": tensor_minimum_kvp,
        },
        "expert_id_reduces_minimum_kvp": (
            expert_id_minimum_kvp is not None
            and (
                tensor_minimum_kvp is None
                or expert_id_minimum_kvp < tensor_minimum_kvp
            )
        ),
        "analytic_sensitivity_only": True,
        "publication_rankable": False,
    }
    body = {
        "schema": BODY_REPRICE_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": overlay["trace_content_hash"],
        "overlay_content_hash": overlay["content_hash"],
        "study_config_content_hash": overlay["study_config_content_hash"],
        "precision_content_hash": canonical_hash(dict(precision)),
        "window_index": overlay["window_index"],
        "first_step_index": overlay["first_step_index"],
        "last_step_index": overlay["last_step_index"],
        "layer_count": NUM_LAYERS,
        "layers": layer_rows,
        "totals": {
            "sum_of_per_layer_slowest_rank_expert_stage_cycles": total_cycles,
            "sum_of_per_layer_slowest_rank_expert_stage_time_s": (
                total_cycles / config.frequency_hz
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_bytes": (
                total_slowest_streamed
            ),
            "system_expert_streamed_bytes": total_system_streamed,
            "slowest_rank_expert_resident_bytes": resident_by_layer_slowest,
            "system_expert_resident_bytes": resident_by_layer_system,
            "logical_route_assignments": (
                NUM_LAYERS * config.batch_size * TOP_K
            ),
            "physical_whole_expert_assignments_across_kvp": (
                total_physical_assignments
            ),
            "expected_physical_whole_expert_assignments_across_kvp": (
                expected_physical
            ),
            "source_hidden_dispatch_bytes": 0,
            "expert_output_collective_slowest_rank_bytes": (
                total_collective_slowest
            ),
            "expert_output_collective_system_bytes": total_collective_system,
        },
        "capacity_and_chip_count_crossover": capacity_crossover,
        "classification": {
            "evidence": "measured_trace_window_exact_body_analytic_projection",
            "full_tpot_repriced": False,
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
            "timing_selection_allowed": False,
            "blockers": [
                "only_routed_expert_stage_and_output_collective_are_repriced",
                "batch_window_is_a_post_hoc_trace_group",
                "compiler_emulator_rtl_and_power_receipts_missing",
            ],
        },
    }
    return _hashed_body(body)


def validate_expert_id_window_body_reprice(
    reprice: Mapping[str, Any],
    *,
    overlay: Mapping[str, Any] | None = None,
) -> None:
    """Validate an isolated layer-exact body reprice and its claim boundary."""

    if not isinstance(reprice, Mapping):
        raise ValueError("expert-ID body reprice must be an object")
    body = dict(reprice)
    observed = body.pop("content_hash", None)
    if observed != canonical_hash(body):
        raise ValueError("expert-ID body reprice content hash mismatch")
    if reprice.get("schema") != BODY_REPRICE_SCHEMA:
        raise ValueError("unsupported expert-ID body reprice schema")
    classification = reprice.get("classification")
    if not isinstance(classification, Mapping) or any(
        classification.get(field) is not False
        for field in (
            "full_tpot_repriced",
            "publication_rankable",
            "hardware_rankable",
            "selection_eligible",
            "timing_selection_allowed",
        )
    ):
        raise ValueError("expert-ID body reprice must remain fail-closed")
    layers = reprice.get("layers")
    if (
        not isinstance(layers, list)
        or len(layers) != NUM_LAYERS
        or [int(row.get("layer", -1)) for row in layers] != list(range(NUM_LAYERS))
        or any(
            row.get("body_layout_route_assignment_exact") is not True
            or row.get("source_hidden_dispatch_bytes") != 0
            for row in layers
        )
    ):
        raise ValueError("expert-ID body reprice lacks exact layer ledgers")
    totals = reprice.get("totals")
    if (
        not isinstance(totals, Mapping)
        or totals.get("source_hidden_dispatch_bytes") != 0
        or totals.get("physical_whole_expert_assignments_across_kvp")
        != totals.get("expected_physical_whole_expert_assignments_across_kvp")
    ):
        raise ValueError("expert-ID body reprice physical work does not conserve")
    crossover = reprice.get("capacity_and_chip_count_crossover")
    if (
        not isinstance(crossover, Mapping)
        or crossover.get("analytic_sensitivity_only") is not True
        or crossover.get("publication_rankable") is not False
    ):
        raise ValueError("expert-ID capacity crossover must remain analytic")
    for policy in ("expert_id_parallel", "tensor_parallel_control"):
        value = crossover.get(policy)
        rows = value.get("rows") if isinstance(value, Mapping) else None
        if (
            not isinstance(rows, list)
            or [row.get("kv_parallel_degree") for row in rows] != [1, 2, 4]
            or any(
                row.get("feasible")
                is not (
                    int(row["slowest_rank_required_bytes"])
                    <= int(row["capacity_bytes_per_chip"])
                )
                for row in rows
            )
        ):
            raise ValueError("expert-ID capacity rows are malformed")
    if overlay is not None:
        validate_expert_id_window_overlay(overlay)
        if (
            reprice.get("overlay_content_hash") != overlay.get("content_hash")
            or reprice.get("trace_content_hash") != overlay.get("trace_content_hash")
            or reprice.get("window_index") != overlay.get("window_index")
        ):
            raise ValueError("expert-ID body reprice binds a different overlay")


def audit_expert_id_window_body_reprice(
    reprice: Mapping[str, Any],
    overlay: Mapping[str, Any],
    perf: PerfModel,
    dims: Mapping[str, Any],
    precision: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute an isolated body receipt from the exact layer adapter."""

    validate_expert_id_window_body_reprice(reprice, overlay=overlay)
    expected = reprice_expert_id_window_body(overlay, perf, dims, precision)
    if dict(reprice) != expected:
        raise ValueError("expert-ID body reprice differs from recomputation")
    return expected


def _encode(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_install(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to replace different artifact: {path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(
                    f"refusing to replace concurrently installed artifact: {path}"
                )
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize_expert_id_balancing_report(
    report: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Install a content-addressed report and content-addressed index."""

    validate_expert_id_balancing_report(report)
    output = output_dir.resolve()
    report_payload = _encode(report)
    report_path = output / f"expert_id_held_out.{report['content_hash']}.json"
    _atomic_install(report_path, report_payload)
    index_body = {
        "schema": INDEX_SCHEMA,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "trace_content_hash": report["trace_content_hash"],
        "study_config_content_hash": report["study_config_content_hash"],
        "report_path": str(report_path),
        "report_sha256": hashlib.sha256(report_payload).hexdigest(),
        "report_content_hash": report["content_hash"],
        "classification": {
            "publication_rankable": False,
            "hardware_rankable": False,
            "selection_eligible": False,
        },
    }
    index = _hashed_body(index_body)
    index_payload = _encode(index)
    index_path = output / f"expert_id_held_out_index.{index['content_hash']}.json"
    _atomic_install(index_path, index_payload)
    return index | {"index_path": str(index_path), "report_path": str(report_path)}


def materialize_expert_id_window_overlay(
    overlay: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Install one layer-exact adapter under its canonical content hash."""

    validate_expert_id_window_overlay(overlay)
    payload = _encode(overlay)
    path = output_dir.resolve() / (
        f"expert_id_window_{int(overlay['window_index']):06d}."
        f"{overlay['content_hash']}.json"
    )
    _atomic_install(path, payload)
    return {
        "schema": "plena-qwen3-moe-expert-id-window-overlay-receipt/v1",
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_hash": overlay["content_hash"],
        "window_index": overlay["window_index"],
        "policy": overlay["policy"],
        "publication_rankable": False,
        "selection_eligible": False,
    }


def materialize_expert_id_window_body_reprice(
    reprice: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Install one isolated body reprice under its canonical content hash."""

    validate_expert_id_window_body_reprice(reprice)
    payload = _encode(reprice)
    path = output_dir.resolve() / (
        f"expert_id_window_body_{int(reprice['window_index']):06d}."
        f"{reprice['content_hash']}.json"
    )
    _atomic_install(path, payload)
    return {
        "schema": "plena-qwen3-moe-expert-id-window-body-reprice-receipt/v1",
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_hash": reprice["content_hash"],
        "window_index": reprice["window_index"],
        "full_tpot_repriced": False,
        "publication_rankable": False,
        "selection_eligible": False,
    }


__all__ = [
    "ExpertIdBalanceConfig",
    "BODY_REPRICE_SCHEMA",
    "INDEX_SCHEMA",
    "PLACEMENT_ALGORITHM",
    "REPORT_SCHEMA",
    "SPLIT_POLICY",
    "WINDOW_OVERLAY_SCHEMA",
    "audit_expert_id_balancing_report",
    "audit_expert_id_window_body_reprice",
    "build_expert_id_window_overlay",
    "build_expert_id_balancing_report",
    "canonical_hash",
    "expert_id_body_inputs_for_layer",
    "file_hash",
    "hardware_binding_for_perf_model",
    "validate_hardware_binding_for_perf_model",
    "materialize_expert_id_balancing_report",
    "materialize_expert_id_window_body_reprice",
    "materialize_expert_id_window_overlay",
    "validate_expert_id_balancing_report",
    "validate_expert_id_window_body_reprice",
    "validate_expert_id_window_overlay",
    "reprice_expert_id_window_body",
]
