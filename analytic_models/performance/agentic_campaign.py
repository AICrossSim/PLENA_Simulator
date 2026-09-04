"""Import the real-checkpoint Nemotron Agentic profiling campaign.

The raw campaign is an external artifact, not a source-tree dependency.  This
module validates the data used by the performance model and reduces the eager
router output to per-layer active-expert sets.  GPU latency remains baseline
evidence; it is never converted into PLENA cycles here.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hybrid_routing import RoutingFormatError, RoutingProfile, RoutingStep


NEMOTRON_AGENTIC_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
NEMOTRON_AGENTIC_REVISION = "ce1b118ae66ec705d02c241525192832eb045fd3"
NEMOTRON_MOE_LAYER_IDS = (
    1,
    3,
    6,
    8,
    10,
    13,
    15,
    17,
    20,
    22,
    24,
    27,
    29,
    31,
    34,
    36,
    38,
    40,
    43,
    45,
    47,
    49,
    51,
)

REQUIRED_CAMPAIGN_FILES = (
    "manifest.json",
    "validation.json",
    "samples.json",
    "latency_raw.jsonl",
    "routing_raw.jsonl",
    "token_traces.jsonl",
    "token_traces_timing.jsonl",
    "campaign_summary.json",
)


class AgenticCampaignFormatError(RoutingFormatError):
    """The external Agentic campaign cannot be used as model evidence."""


@dataclass(frozen=True)
class AgenticCampaignContract:
    model_id: str = NEMOTRON_AGENTIC_MODEL_ID
    revision: str = NEMOTRON_AGENTIC_REVISION
    benchmark_sample_counts: tuple[tuple[str, int], ...] = (
        ("bfcl_v3", 16),
        ("gpqa_diamond", 16),
        ("swebench_verified", 16),
    )
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16)
    moe_layer_ids: tuple[int, ...] = NEMOTRON_MOE_LAYER_IDS
    expert_count: int = 128
    top_k: int = 6
    decode_steps: int = 32
    measurements: int = 20
    generation_limit: int = 32

    @property
    def sample_count(self) -> int:
        return sum(count for _, count in self.benchmark_sample_counts)


DEFAULT_AGENTIC_CONTRACT = AgenticCampaignContract()


@dataclass(frozen=True)
class AgenticSample:
    benchmark: str
    sample_id: str
    prompt_length: int
    prompt_sha256: str
    generated_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class AgenticRoutingCounters:
    """Separate source validation coverage from the replay window actually used."""

    raw_events: int
    conservation_validated_events: int
    prefill_events: int
    decode_events: int
    fully_validated_decode_events: int
    used_decode_events: int
    ignored_decode_events: int

    def __post_init__(self) -> None:
        if (
            min(
                self.raw_events,
                self.conservation_validated_events,
                self.prefill_events,
                self.decode_events,
                self.fully_validated_decode_events,
                self.used_decode_events,
                self.ignored_decode_events,
            )
            < 0
        ):
            raise AgenticCampaignFormatError("routing counters must be non-negative")
        if self.raw_events != self.prefill_events + self.decode_events:
            raise AgenticCampaignFormatError("routing phase counters do not sum to raw events")
        if self.conservation_validated_events != self.raw_events:
            raise AgenticCampaignFormatError("not every raw routing event passed conservation checks")
        if self.fully_validated_decode_events != self.decode_events:
            raise AgenticCampaignFormatError("not every decode event received full row validation")
        if self.decode_events != self.used_decode_events + self.ignored_decode_events:
            raise AgenticCampaignFormatError("used and ignored decode counters do not cover decode events")

    def to_dict(self) -> dict[str, int]:
        return {
            "raw_events": self.raw_events,
            "conservation_validated_events": self.conservation_validated_events,
            "prefill_events": self.prefill_events,
            "decode_events": self.decode_events,
            "fully_validated_decode_events": self.fully_validated_decode_events,
            "used_decode_events": self.used_decode_events,
            "ignored_decode_events": self.ignored_decode_events,
        }


@dataclass(frozen=True)
class AgenticGpuBatchAggregate:
    """Pinned global GPU aggregate copied from the external campaign summary."""

    batch_size: int
    request_measurements: int
    trial_measurements: int
    ttft_ms_median: float
    ttft_ms_p95: float
    itl_ms_median: float
    itl_ms_p95: float
    e2e_ms_median: float
    e2e_ms_p95: float
    throughput_tokens_s_median: float
    throughput_tokens_s_p95: float
    batch_joules_median: float
    batch_joules_p95: float
    peak_memory_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class AgenticBatchGroup:
    benchmark: str
    batch_size: int
    group_index: int
    sample_ids: tuple[str, ...]
    prompt_lengths: tuple[int, ...]
    padded_context_length: int
    generation_limit: int
    gpu_ttft_ms_median: float
    gpu_itl_ms_median: float
    gpu_e2e_ms_median: float
    gpu_batch_e2e_ms_median: float
    gpu_batch_throughput_tokens_s_median: float
    gpu_batch_energy_joules_median: float

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or len(self.sample_ids) != self.batch_size:
            raise AgenticCampaignFormatError("Agentic batch membership does not match batch_size")
        if len(self.prompt_lengths) != self.batch_size or any(length <= 0 for length in self.prompt_lengths):
            raise AgenticCampaignFormatError("Agentic batch prompt lengths are incomplete")
        if self.padded_context_length != max(self.prompt_lengths):
            raise AgenticCampaignFormatError("padded context must equal the longest prompt in the batch")

    @property
    def padding_token_overhead(self) -> int:
        return self.batch_size * self.padded_context_length - sum(self.prompt_lengths)

    @property
    def padding_fraction(self) -> float:
        padded = self.batch_size * self.padded_context_length
        return self.padding_token_overhead / padded

    @property
    def key(self) -> str:
        return f"{self.benchmark}_b{self.batch_size}_g{self.group_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "benchmark": self.benchmark,
            "batch_size": self.batch_size,
            "group_index": self.group_index,
            "sample_ids": list(self.sample_ids),
            "prompt_lengths": list(self.prompt_lengths),
            "padded_context_length": self.padded_context_length,
            "generation_limit": self.generation_limit,
            "padding_token_overhead": self.padding_token_overhead,
            "padding_fraction": self.padding_fraction,
            "gpu": {
                "ttft_ms_median": self.gpu_ttft_ms_median,
                "itl_ms_median": self.gpu_itl_ms_median,
                "e2e_ms_median": self.gpu_e2e_ms_median,
                "batch_e2e_ms_median": self.gpu_batch_e2e_ms_median,
                "batch_throughput_tokens_s_median": self.gpu_batch_throughput_tokens_s_median,
                "batch_energy_joules_median": self.gpu_batch_energy_joules_median,
            },
        }


@dataclass(frozen=True)
class AgenticCampaign:
    root: Path
    model_id: str
    revision: str
    gpu_uuid: str
    expert_count: int
    top_k: int
    routing_source_sha256: str
    timing_source_sha256: str
    campaign_summary_source_sha256: str
    samples: tuple[AgenticSample, ...]
    groups: tuple[AgenticBatchGroup, ...]
    # sample -> decode step -> ((layer, active experts), ...)
    decode_routes: dict[str, tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]]
    routing_counters: AgenticRoutingCounters
    gpu_global_aggregates: tuple[AgenticGpuBatchAggregate, ...]
    timing_and_routing_tokens_identical_samples: int
    timing_and_routing_replay_window_identical_samples: int
    checksum_manifest_entries: int

    @property
    def raw_routing_events(self) -> int:
        return self.routing_counters.raw_events

    def group(self, benchmark: str, batch_size: int, group_index: int) -> AgenticBatchGroup:
        matches = [
            group
            for group in self.groups
            if (group.benchmark, group.batch_size, group.group_index) == (benchmark, batch_size, group_index)
        ]
        if len(matches) != 1:
            raise AgenticCampaignFormatError(f"no unique Agentic group {benchmark}/B{batch_size}/G{group_index}")
        return matches[0]

    def routing_profile(self, group: AgenticBatchGroup, *, decode_steps: int) -> RoutingProfile:
        if decode_steps <= 0 or decode_steps > group.generation_limit:
            raise AgenticCampaignFormatError("requested decode length exceeds the profiled batch case")
        sample_routes = [self.decode_routes[sample_id] for sample_id in group.sample_ids]
        if any(len(routes) < decode_steps for routes in sample_routes):
            raise AgenticCampaignFormatError("routing trace is shorter than the requested decode replay")

        steps = []
        for index in range(decode_steps):
            per_sample = [dict(routes[index]) for routes in sample_routes]
            layer_ids = tuple(per_sample[0])
            if any(tuple(route) != layer_ids for route in per_sample[1:]):
                raise AgenticCampaignFormatError("sample routing traces disagree on MoE layer coverage")
            merged = tuple(
                (layer_id, tuple(sorted({expert for route in per_sample for expert in route[layer_id]})))
                for layer_id in layer_ids
            )
            steps.append(
                RoutingStep(
                    phase="decode",
                    index=index,
                    token_count=group.batch_size,
                    active_experts_by_layer=merged,
                )
            )

        profile = RoutingProfile(
            model_key="nemotron3",
            model_id=self.model_id,
            revision=self.revision,
            batch_size=group.batch_size,
            context_length=group.padded_context_length,
            expert_count=self.expert_count,
            top_k=self.top_k,
            source_sha256=self.routing_source_sha256,
            steps=tuple(steps),
        )
        profile.validate_replay(
            model_key="nemotron3",
            phase="decode",
            batch_size=group.batch_size,
            context_length=group.padded_context_length,
            sequence_length=1,
            decode_steps=decode_steps,
        )
        return profile

    def to_summary(self) -> dict[str, Any]:
        benchmark_counts = Counter(sample.benchmark for sample in self.samples)
        group_counts = Counter((group.benchmark, group.batch_size) for group in self.groups)
        return {
            "contract": "nemotron-agentic-campaign-import-v2",
            "model_id": self.model_id,
            "revision": self.revision,
            "gpu_uuid": self.gpu_uuid,
            "sample_count": len(self.samples),
            "benchmark_sample_counts": dict(sorted(benchmark_counts.items())),
            "batch_group_counts": {
                f"{benchmark}_b{batch}": count for (benchmark, batch), count in sorted(group_counts.items())
            },
            "raw_routing_events": self.raw_routing_events,
            "routing_event_accounting": self.routing_counters.to_dict(),
            "decode_steps_imported_per_sample": min(map(len, self.decode_routes.values())),
            "routing_source_sha256": self.routing_source_sha256,
            "timing_source_sha256": self.timing_source_sha256,
            "campaign_summary_source_sha256": self.campaign_summary_source_sha256,
            "checksum_manifest_entries": self.checksum_manifest_entries,
            "verified_consumed_files": len(REQUIRED_CAMPAIGN_FILES),
            "timing_and_routing_tokens_identical_samples": self.timing_and_routing_tokens_identical_samples,
            "timing_and_routing_replay_window_identical_samples": (
                self.timing_and_routing_replay_window_identical_samples
            ),
            "gpu_global_aggregate_scope": "campaign_summary.timing.aggregate.all",
            "gpu_global_aggregates": {
                f"batch_b{aggregate.batch_size}": aggregate.to_dict() for aggregate in self.gpu_global_aggregates
            },
            "routing_semantics": (
                "eager routing token traces are authoritative for replay; optimized timing token traces are baseline only"
            ),
            "batch_routing_semantics": (
                "length-sorted GPU batch membership with active-expert unions reconstructed from independent "
                "real-checkpoint B1 eager traces; not a direct batched routing capture"
            ),
            "privacy": (
                "the external archive contains prompt token IDs and must not be published; "
                "only hashes, aggregate measurements and derived routing counts enter this artifact"
            ),
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AgenticCampaignFormatError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise AgenticCampaignFormatError(f"{path} must contain a JSON object")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise AgenticCampaignFormatError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _checksum_manifest(root: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    try:
        lines = (root / "SHA256SUMS").read_text().splitlines()
    except OSError as error:
        raise AgenticCampaignFormatError(f"cannot read campaign checksums: {error}") from error
    for line in lines:
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64 or name in checksums:
            raise AgenticCampaignFormatError("campaign SHA256SUMS is malformed")
        checksums[name] = digest
    return checksums


def _verify_required_checksums(root: Path, checksums: dict[str, str]) -> None:
    for name in REQUIRED_CAMPAIGN_FILES:
        expected = checksums.get(name)
        if expected is None:
            raise AgenticCampaignFormatError(f"campaign checksum is missing {name}")
        actual = _sha256(root / name)
        if actual != expected:
            raise AgenticCampaignFormatError(f"campaign checksum mismatch for {name}")


def _jsonl(path: Path):
    try:
        with path.open() as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise AgenticCampaignFormatError(f"{path.name}:{line_number}: invalid JSON") from error
                if not isinstance(row, dict):
                    raise AgenticCampaignFormatError(f"{path.name}:{line_number}: expected object")
                yield line_number, row
    except OSError as error:
        raise AgenticCampaignFormatError(f"cannot read {path}: {error}") from error


def _load_samples(
    root: Path,
    contract: AgenticCampaignContract,
) -> tuple[dict[str, AgenticSample], int, int]:
    document = _read_json(root / "samples.json")
    rows = document.get("samples")
    if not isinstance(rows, list):
        raise AgenticCampaignFormatError("samples.json has no sample list")

    token_traces: dict[str, dict[str, Any]] = {}
    for line_number, row in _jsonl(root / "token_traces.jsonl"):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in token_traces:
            raise AgenticCampaignFormatError(f"token_traces.jsonl:{line_number}: duplicate/invalid sample")
        token_traces[sample_id] = row

    timing_traces: dict[str, dict[str, Any]] = {}
    for line_number, row in _jsonl(root / "token_traces_timing.jsonl"):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in timing_traces:
            raise AgenticCampaignFormatError(f"token_traces_timing.jsonl:{line_number}: duplicate/invalid sample")
        timing_traces[sample_id] = row

    samples: dict[str, AgenticSample] = {}
    identical_count = 0
    replay_window_identical_count = 0
    for row in rows:
        sample_id = row.get("sample_id")
        trace = token_traces.get(sample_id)
        timing_trace = timing_traces.get(sample_id)
        if not isinstance(sample_id, str) or sample_id in samples or trace is None or timing_trace is None:
            raise AgenticCampaignFormatError("sample and eager token trace coverage disagree")
        prompt_ids = row.get("prompt_token_ids")
        generated_ids = trace.get("routing_generated_token_ids")
        timing_ids = trace.get("timing_generated_token_ids")
        standalone_timing_ids = timing_trace.get("generated_token_ids")
        if not isinstance(prompt_ids, list) or len(prompt_ids) != row.get("prompt_length"):
            raise AgenticCampaignFormatError(f"{sample_id}: prompt token IDs are incomplete")
        if not isinstance(generated_ids, list) or len(generated_ids) != trace.get("routing_generated_tokens"):
            raise AgenticCampaignFormatError(f"{sample_id}: eager generated token IDs are incomplete")
        if (
            not isinstance(timing_ids, list)
            or not isinstance(standalone_timing_ids, list)
            or timing_ids != standalone_timing_ids
            or len(timing_ids) != timing_trace.get("generated_tokens")
        ):
            raise AgenticCampaignFormatError(f"{sample_id}: optimized timing token traces disagree")
        if trace.get("prompt_sha256") != row.get("prompt_sha256"):
            raise AgenticCampaignFormatError(f"{sample_id}: prompt hash differs between source and routing")
        if timing_trace.get("prompt_sha256") != row.get("prompt_sha256") or timing_trace.get("benchmark") != row.get(
            "benchmark"
        ):
            raise AgenticCampaignFormatError(f"{sample_id}: timing token trace metadata disagrees")
        identical = generated_ids == timing_ids
        if trace.get("timing_vs_routing_identical") != identical:
            raise AgenticCampaignFormatError(f"{sample_id}: token-trace equality flag is incorrect")
        identical_count += int(identical)
        replay_window_identical_count += int(
            generated_ids[: contract.decode_steps] == timing_ids[: contract.decode_steps]
        )
        samples[sample_id] = AgenticSample(
            benchmark=str(row["benchmark"]),
            sample_id=sample_id,
            prompt_length=int(row["prompt_length"]),
            prompt_sha256=str(row["prompt_sha256"]),
            generated_token_ids=tuple(int(token) for token in generated_ids),
        )

    expected_counts = dict(contract.benchmark_sample_counts)
    observed_counts = Counter(sample.benchmark for sample in samples.values())
    if len(samples) != contract.sample_count or observed_counts != expected_counts:
        raise AgenticCampaignFormatError(
            f"unexpected benchmark sample coverage: {dict(observed_counts)} != {expected_counts}"
        )
    if set(token_traces) != set(samples) or set(timing_traces) != set(samples):
        raise AgenticCampaignFormatError("token traces contain unknown or missing samples")
    return samples, identical_count, replay_window_identical_count


def _load_batch_groups(
    root: Path,
    samples: dict[str, AgenticSample],
    contract: AgenticCampaignContract,
) -> tuple[AgenticBatchGroup, ...]:
    rows_by_group: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for _, row in _jsonl(root / "latency_raw.jsonl"):
        if (
            row.get("mode") == "batch_sweep"
            and row.get("phase") == "measurement"
            and row.get("include_in_summary") is True
        ):
            key = (str(row["benchmark"]), int(row["batch_size"]), int(row["group_index"]))
            rows_by_group[key].append(row)

    groups = []
    for (benchmark, batch_size, group_index), rows in sorted(rows_by_group.items()):
        if batch_size not in contract.batch_sizes:
            raise AgenticCampaignFormatError(f"unexpected timing batch size {batch_size}")
        expected_rows = contract.measurements * batch_size
        if len(rows) != expected_rows:
            raise AgenticCampaignFormatError(
                f"{benchmark}/B{batch_size}/G{group_index}: {len(rows)} rows, expected {expected_rows}"
            )
        trial_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            trial_rows[str(row["trial_id"])].append(row)
        if len(trial_rows) != contract.measurements or any(len(trial) != batch_size for trial in trial_rows.values()):
            raise AgenticCampaignFormatError("timing trials do not contain one row per batch request")

        reference = min(trial_rows.values(), key=lambda trial: int(trial[0]["measurement"]))
        members = sorted(
            ((str(row["sample_id"]), int(row["prompt_length"]), int(row["padded_prompt_length"])) for row in reference),
            key=lambda item: (item[1], item[0]),
        )
        sample_ids = tuple(member[0] for member in members)
        if len(set(sample_ids)) != batch_size or any(sample_id not in samples for sample_id in sample_ids):
            raise AgenticCampaignFormatError("timing group has duplicate or unknown samples")
        expected_members = set(sample_ids)
        if any({str(row["sample_id"]) for row in trial} != expected_members for trial in trial_rows.values()):
            raise AgenticCampaignFormatError("timing group membership changed between measurements")
        if any(samples[sample_id].benchmark != benchmark for sample_id in sample_ids):
            raise AgenticCampaignFormatError("timing group mixes benchmarks")

        prompt_lengths = tuple(member[1] for member in members)
        if any(samples[sample_id].prompt_length != length for sample_id, length in zip(sample_ids, prompt_lengths)):
            raise AgenticCampaignFormatError("timing group prompt length disagrees with samples.json")
        padded_values = {int(row["padded_prompt_length"]) for row in rows}
        generation_limits = {int(row["generation_limit"]) for row in rows}
        if len(padded_values) != 1 or generation_limits != {contract.generation_limit}:
            raise AgenticCampaignFormatError("timing group padding or generation length is inconsistent")

        batch_metric_names = (
            "batch_e2e_ms",
            "batch_throughput_tokens_per_s",
            "batch_energy_joules",
        )
        for trial in trial_rows.values():
            if any(len({float(row[name]) for row in trial}) != 1 for name in batch_metric_names):
                raise AgenticCampaignFormatError("batch-level timing metrics differ within one trial")
        unique_trials = [trial[0] for trial in trial_rows.values()]
        itl_values = [float(value) for row in rows for value in row["itl_ms"]]
        groups.append(
            AgenticBatchGroup(
                benchmark=benchmark,
                batch_size=batch_size,
                group_index=group_index,
                sample_ids=sample_ids,
                prompt_lengths=prompt_lengths,
                padded_context_length=padded_values.pop(),
                generation_limit=contract.generation_limit,
                gpu_ttft_ms_median=statistics.median(float(row["ttft_ms"]) for row in rows),
                gpu_itl_ms_median=statistics.median(itl_values),
                gpu_e2e_ms_median=statistics.median(float(row["e2e_ms"]) for row in rows),
                gpu_batch_e2e_ms_median=statistics.median(float(row["batch_e2e_ms"]) for row in unique_trials),
                gpu_batch_throughput_tokens_s_median=statistics.median(
                    float(row["batch_throughput_tokens_per_s"]) for row in unique_trials
                ),
                gpu_batch_energy_joules_median=statistics.median(
                    float(row["batch_energy_joules"]) for row in unique_trials
                ),
            )
        )

    expected_counts = dict(contract.benchmark_sample_counts)
    for benchmark, sample_count in expected_counts.items():
        expected_ids = {sample.sample_id for sample in samples.values() if sample.benchmark == benchmark}
        for batch_size in contract.batch_sizes:
            selected = [group for group in groups if (group.benchmark, group.batch_size) == (benchmark, batch_size)]
            observed_ids = [sample_id for group in selected for sample_id in group.sample_ids]
            if len(selected) != math.ceil(sample_count / batch_size) or set(observed_ids) != expected_ids:
                raise AgenticCampaignFormatError(
                    f"{benchmark}/B{batch_size}: timing groups do not cover every sample exactly once"
                )
            if len(observed_ids) != len(set(observed_ids)):
                raise AgenticCampaignFormatError(f"{benchmark}/B{batch_size}: timing sample is repeated")
    return tuple(groups)


def _load_gpu_global_aggregates(
    root: Path,
    contract: AgenticCampaignContract,
    *,
    timing_and_routing_identical_samples: int,
) -> tuple[AgenticGpuBatchAggregate, ...]:
    """Validate the exact global GPU rows quoted next to the DSE results."""

    document = _read_json(root / "campaign_summary.json")
    routing = document.get("routing")
    if not isinstance(routing, dict) or int(routing.get("timing_vs_routing_identical_samples", -1)) != (
        timing_and_routing_identical_samples
    ):
        raise AgenticCampaignFormatError("campaign summary routing identity count disagrees with token traces")
    timing = document.get("timing")
    aggregate = timing.get("aggregate") if isinstance(timing, dict) else None
    all_rows = aggregate.get("all") if isinstance(aggregate, dict) else None
    if not isinstance(all_rows, dict):
        raise AgenticCampaignFormatError("campaign summary has no timing.aggregate.all table")

    float_fields = (
        "ttft_ms_median",
        "ttft_ms_p95",
        "itl_ms_median",
        "itl_ms_p95",
        "e2e_ms_median",
        "e2e_ms_p95",
        "throughput_tokens_s_median",
        "throughput_tokens_s_p95",
        "batch_joules_median",
        "batch_joules_p95",
    )
    rows = []
    for batch_size in contract.batch_sizes:
        source = all_rows.get(f"batch_b{batch_size}")
        if not isinstance(source, dict):
            raise AgenticCampaignFormatError(f"campaign summary is missing global batch B{batch_size}")
        expected_requests = contract.sample_count * contract.measurements
        expected_trials = sum(
            math.ceil(sample_count / batch_size) * contract.measurements
            for _, sample_count in contract.benchmark_sample_counts
        )
        if int(source.get("request_measurements", -1)) != expected_requests:
            raise AgenticCampaignFormatError(f"global B{batch_size} request count is inconsistent")
        if int(source.get("trial_measurements", -1)) != expected_trials:
            raise AgenticCampaignFormatError(f"global B{batch_size} trial count is inconsistent")
        values = {name: float(source.get(name, math.nan)) for name in float_fields}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise AgenticCampaignFormatError(f"global B{batch_size} contains invalid timing or energy")
        peak_memory = source.get("peak_memory_bytes")
        if type(peak_memory) is not int or peak_memory <= 0:
            raise AgenticCampaignFormatError(f"global B{batch_size} peak memory is invalid")
        rows.append(
            AgenticGpuBatchAggregate(
                batch_size=batch_size,
                request_measurements=expected_requests,
                trial_measurements=expected_trials,
                peak_memory_bytes=peak_memory,
                **values,
            )
        )
    return tuple(rows)


def _validate_used_routing_rows(
    *,
    event: dict[str, Any],
    line_number: int,
    contract: AgenticCampaignContract,
) -> tuple[int, ...]:
    token_count = int(event.get("token_count", 0))
    ids = event.get("topk_expert_ids")
    weights = event.get("topk_weights")
    counts = event.get("expert_counts_128")
    if (
        event.get("top_k") != contract.top_k
        or not isinstance(ids, list)
        or not isinstance(weights, list)
        or len(ids) != token_count
        or len(weights) != token_count
        or not isinstance(counts, list)
        or len(counts) != contract.expert_count
    ):
        raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: routing shape mismatch")
    derived: Counter[int] = Counter()
    for row_ids, row_weights in zip(ids, weights, strict=True):
        if len(row_ids) != contract.top_k or len(set(row_ids)) != contract.top_k:
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: invalid top-k IDs")
        if any(not isinstance(expert, int) or not 0 <= expert < contract.expert_count for expert in row_ids):
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: expert ID out of range")
        if len(row_weights) != contract.top_k or any(not math.isfinite(float(weight)) for weight in row_weights):
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: invalid routing weights")
        if not math.isclose(sum(float(weight) for weight in row_weights), 1.0, abs_tol=2e-6):
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: routing weights are not normalized")
        derived.update(row_ids)
    if [derived[index] for index in range(contract.expert_count)] != counts:
        raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: expert counts disagree with IDs")
    return tuple(index for index, count in enumerate(counts) if count)


def _load_decode_routes(
    root: Path,
    samples: dict[str, AgenticSample],
    contract: AgenticCampaignContract,
    expected_event_count: int,
) -> tuple[
    dict[str, tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]],
    AgenticRoutingCounters,
]:
    routes: dict[str, dict[int, dict[int, tuple[int, ...]]]] = defaultdict(lambda: defaultdict(dict))
    prefill_layers: dict[str, set[int]] = defaultdict(set)
    generation_checked: set[str] = set()
    seen_decode_events: set[tuple[str, int, int]] = set()
    event_count = 0
    prefill_event_count = 0
    decode_event_count = 0
    fully_validated_decode_event_count = 0
    used_decode_event_count = 0
    for line_number, event in _jsonl(root / "routing_raw.jsonl"):
        event_count += 1
        sample_id = event.get("sample_id")
        sample = samples.get(sample_id)
        if sample is None:
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: unknown sample")
        if (
            event.get("benchmark") != sample.benchmark
            or event.get("prompt_sha256") != sample.prompt_sha256
            or int(event.get("prompt_length", 0)) != sample.prompt_length
            or event.get("batch") != 1
        ):
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: sample metadata mismatch")
        layer_id = int(event.get("layer_index", -1))
        if layer_id not in contract.moe_layer_ids:
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: unexpected MoE layer")
        counts = event.get("expert_counts_128")
        token_count = int(event.get("token_count", 0))
        if (
            event.get("top_k") != contract.top_k
            or not isinstance(counts, list)
            or len(counts) != contract.expert_count
            or any(type(count) is not int or count < 0 for count in counts)
            or sum(int(count) for count in counts) != token_count * contract.top_k
            or int(event.get("shared_expert_token_count", -1)) != token_count
        ):
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: routing conservation failed")
        if sample_id not in generation_checked:
            if tuple(int(token) for token in event.get("generated_token_ids", ())) != sample.generated_token_ids:
                raise AgenticCampaignFormatError(f"{sample_id}: routing event and eager token trace disagree")
            generation_checked.add(sample_id)

        phase = event.get("phase")
        if phase == "prefill":
            prefill_event_count += 1
            if event.get("decode_step") != -1 or token_count != sample.prompt_length:
                raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: invalid prefill event")
            if layer_id in prefill_layers[sample_id]:
                raise AgenticCampaignFormatError(f"{sample_id}: duplicate prefill routing layer")
            prefill_layers[sample_id].add(layer_id)
            continue
        if phase != "decode":
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: invalid phase")
        decode_event_count += 1
        step = int(event.get("decode_step", -1))
        if step < 0 or token_count != 1:
            raise AgenticCampaignFormatError(f"routing_raw.jsonl:{line_number}: invalid decode event")
        key = (sample_id, step, layer_id)
        if key in seen_decode_events:
            raise AgenticCampaignFormatError(f"{sample_id}: duplicate decode step/layer routing event")
        seen_decode_events.add(key)
        active = _validate_used_routing_rows(event=event, line_number=line_number, contract=contract)
        fully_validated_decode_event_count += 1
        if step >= contract.decode_steps:
            continue
        used_decode_event_count += 1
        if layer_id in routes[sample_id][step]:
            raise AgenticCampaignFormatError(f"{sample_id}: duplicate decode step/layer routing event")
        routes[sample_id][step][layer_id] = active

    if event_count != expected_event_count:
        raise AgenticCampaignFormatError(f"routing event count {event_count} != manifest {expected_event_count}")
    expected_layers = set(contract.moe_layer_ids)
    if generation_checked != set(samples):
        raise AgenticCampaignFormatError("routing events do not cover every sample")
    normalized = {}
    for sample_id in sorted(samples):
        if prefill_layers[sample_id] != expected_layers:
            raise AgenticCampaignFormatError(f"{sample_id}: prefill does not cover every MoE layer")
        expected_steps = set(range(contract.decode_steps))
        if set(routes[sample_id]) != expected_steps:
            raise AgenticCampaignFormatError(f"{sample_id}: decode trace is shorter than requested")
        step_rows = []
        for step in range(contract.decode_steps):
            by_layer = routes[sample_id][step]
            if set(by_layer) != expected_layers:
                raise AgenticCampaignFormatError(f"{sample_id}: decode step {step} has incomplete layers")
            step_rows.append(tuple((layer, by_layer[layer]) for layer in contract.moe_layer_ids))
        normalized[sample_id] = tuple(step_rows)
    expected_used = len(samples) * contract.decode_steps * len(contract.moe_layer_ids)
    if used_decode_event_count != expected_used:
        raise AgenticCampaignFormatError(
            f"used decode event count {used_decode_event_count} != replay contract {expected_used}"
        )
    counters = AgenticRoutingCounters(
        raw_events=event_count,
        conservation_validated_events=event_count,
        prefill_events=prefill_event_count,
        decode_events=decode_event_count,
        fully_validated_decode_events=fully_validated_decode_event_count,
        used_decode_events=used_decode_event_count,
        ignored_decode_events=decode_event_count - used_decode_event_count,
    )
    return normalized, counters


def load_agentic_campaign(
    root: Path,
    *,
    contract: AgenticCampaignContract = DEFAULT_AGENTIC_CONTRACT,
    verify_checksums: bool = True,
) -> AgenticCampaign:
    """Load and validate the campaign subset consumed by PLENA DSE."""

    root = root.resolve()
    manifest = _read_json(root / "manifest.json")
    validation = _read_json(root / "validation.json")
    model = manifest.get("model", {})
    if manifest.get("status") != "COMPLETE" or validation.get("status") != "PASS":
        raise AgenticCampaignFormatError("Agentic campaign is not complete and validated")
    if model.get("id") != contract.model_id or model.get("revision") != contract.revision:
        raise AgenticCampaignFormatError("Agentic campaign model/revision does not match the pinned contract")
    if model.get("checkpoint_type") != "real official NVFP4 checkpoint":
        raise AgenticCampaignFormatError("Agentic campaign is not a real-checkpoint run")
    if manifest.get("routing", {}).get("enforce_eager") is not True:
        raise AgenticCampaignFormatError("routing must come from the self-consistent eager run")
    if manifest.get("timing", {}).get("enforce_eager") is not False:
        raise AgenticCampaignFormatError("timing baseline must come from optimized serving")

    checksums = _checksum_manifest(root)
    if verify_checksums:
        _verify_required_checksums(root, checksums)
    samples, identical_count, replay_window_identical_count = _load_samples(root, contract)
    groups = _load_batch_groups(root, samples, contract)
    gpu_global_aggregates = _load_gpu_global_aggregates(
        root,
        contract,
        timing_and_routing_identical_samples=identical_count,
    )
    routing_manifest = manifest.get("routing", {})
    routes, routing_counters = _load_decode_routes(
        root,
        samples,
        contract,
        expected_event_count=int(routing_manifest.get("raw_event_count", -1)),
    )
    routing_gpu_uuid = str(routing_manifest.get("gpu_uuid", ""))
    timing_gpu_uuid = str(manifest.get("timing", {}).get("gpu_uuid", ""))
    if not routing_gpu_uuid or routing_gpu_uuid != timing_gpu_uuid:
        raise AgenticCampaignFormatError("timing and routing were not collected on the same pinned GPU")
    return AgenticCampaign(
        root=root,
        model_id=contract.model_id,
        revision=contract.revision,
        gpu_uuid=routing_gpu_uuid,
        expert_count=contract.expert_count,
        top_k=contract.top_k,
        routing_source_sha256=checksums["routing_raw.jsonl"],
        timing_source_sha256=checksums["latency_raw.jsonl"],
        campaign_summary_source_sha256=checksums["campaign_summary.json"],
        samples=tuple(samples[sample_id] for sample_id in sorted(samples)),
        groups=groups,
        decode_routes=routes,
        routing_counters=routing_counters,
        gpu_global_aggregates=gpu_global_aggregates,
        timing_and_routing_tokens_identical_samples=identical_count,
        timing_and_routing_replay_window_identical_samples=replay_window_identical_count,
        checksum_manifest_entries=len(checksums),
    )


__all__ = [
    "DEFAULT_AGENTIC_CONTRACT",
    "NEMOTRON_AGENTIC_MODEL_ID",
    "NEMOTRON_AGENTIC_REVISION",
    "NEMOTRON_MOE_LAYER_IDS",
    "AgenticBatchGroup",
    "AgenticCampaign",
    "AgenticCampaignContract",
    "AgenticCampaignFormatError",
    "AgenticGpuBatchAggregate",
    "AgenticRoutingCounters",
    "load_agentic_campaign",
]
