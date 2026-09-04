"""Validated routing profiles for hybrid-model workload replay.

The DSE consumes only active expert IDs.  It never fabricates routing from a
uniform distribution: measured traces are validated here, while missing batch
profiles remain explicit optimistic/conservative bounds in the campaign.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PINNED_NEMOTRON_ROUTING = Path(__file__).with_name("profiles") / "nemotron3_decode_routing_trace.json"
_LAYER_RE = re.compile(r"(?:model|backbone)\.layers\.(\d+)\..+")


class RoutingFormatError(ValueError):
    """A routing trace cannot be used as workload evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _layer_id(name: str) -> int:
    match = _LAYER_RE.fullmatch(name)
    if match is None:
        raise RoutingFormatError(f"cannot extract layer ID from {name!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class RoutingStep:
    phase: str
    index: int
    token_count: int
    active_experts_by_layer: tuple[tuple[int, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise RoutingFormatError(f"unsupported routing phase {self.phase!r}")
        if self.index < 0 or self.token_count <= 0:
            raise RoutingFormatError("routing step index/token count is invalid")
        layers = [layer_id for layer_id, _ in self.active_experts_by_layer]
        if layers != sorted(layers) or len(layers) != len(set(layers)):
            raise RoutingFormatError("routing layers must be sorted and unique")
        if any(not experts or len(experts) != len(set(experts)) for _, experts in self.active_experts_by_layer):
            raise RoutingFormatError("every routing layer needs a non-empty unique expert set")

    @property
    def unique_experts_by_layer(self) -> tuple[tuple[int, int], ...]:
        return tuple((layer_id, len(experts)) for layer_id, experts in self.active_experts_by_layer)


@dataclass(frozen=True)
class RoutingProfile:
    model_key: str
    model_id: str
    revision: str
    batch_size: int
    context_length: int
    expert_count: int
    top_k: int
    source_sha256: str
    steps: tuple[RoutingStep, ...]

    def __post_init__(self) -> None:
        if self.model_key not in {"nemotron3", "kimi_k3"}:
            raise RoutingFormatError(f"unsupported model key {self.model_key!r}")
        if min(self.batch_size, self.context_length, self.expert_count, self.top_k) <= 0:
            raise RoutingFormatError("routing profile dimensions must be positive")
        if self.top_k > self.expert_count:
            raise RoutingFormatError("routing top-k exceeds expert count")
        keys = [(step.phase, step.index) for step in self.steps]
        if len(keys) != len(set(keys)):
            raise RoutingFormatError("routing profile contains duplicate phase/step keys")
        for step in self.steps:
            for _, experts in step.active_experts_by_layer:
                if any(expert < 0 or expert >= self.expert_count for expert in experts):
                    raise RoutingFormatError("routing profile contains an out-of-range expert ID")

    def step(self, phase: str, index: int) -> RoutingStep:
        matches = [step for step in self.steps if step.phase == phase and step.index == index]
        if len(matches) != 1:
            raise RoutingFormatError(f"routing profile has no unique {phase} step {index}")
        return matches[0]

    def validate_replay(
        self,
        *,
        model_key: str,
        phase: str,
        batch_size: int,
        context_length: int,
        sequence_length: int,
        decode_steps: int,
    ) -> None:
        if model_key != self.model_key:
            raise RoutingFormatError(f"routing model {self.model_key} cannot replay {model_key}")
        if batch_size != self.batch_size or context_length != self.context_length:
            raise RoutingFormatError(
                "routing batch/context does not match the requested workload: "
                f"B{batch_size}/S{context_length} != B{self.batch_size}/S{self.context_length}"
            )
        if phase == "prefill":
            step = self.step("prefill", 0)
            if step.token_count != batch_size * sequence_length:
                raise RoutingFormatError("prefill routing token count does not match the workload")
        else:
            for index in range(decode_steps):
                step = self.step("decode", index)
                if step.token_count != batch_size:
                    raise RoutingFormatError("decode routing token count does not match batch size")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "contract": "hybrid-routing-profile-v1",
            "model_key": self.model_key,
            "model_id": self.model_id,
            "revision": self.revision,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "expert_count": self.expert_count,
            "top_k": self.top_k,
            "source_sha256": self.source_sha256,
            "steps": [
                {
                    "phase": step.phase,
                    "index": step.index,
                    "token_count": step.token_count,
                    "active_experts_by_layer": [
                        [layer_id, list(experts)] for layer_id, experts in step.active_experts_by_layer
                    ],
                }
                for step in self.steps
            ],
        }


def load_profile(path: Path) -> RoutingProfile:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RoutingFormatError(f"cannot read routing profile {path}: {error}") from error
    if document.get("contract") != "hybrid-routing-profile-v1" or document.get("schema_version") != 1:
        raise RoutingFormatError("unsupported normalized routing contract")
    steps = tuple(
        RoutingStep(
            phase=step["phase"],
            index=int(step["index"]),
            token_count=int(step["token_count"]),
            active_experts_by_layer=tuple(
                (int(layer_id), tuple(int(expert) for expert in experts))
                for layer_id, experts in step["active_experts_by_layer"]
            ),
        )
        for step in document["steps"]
    )
    return RoutingProfile(
        model_key=document["model_key"],
        model_id=document["model_id"],
        revision=document["revision"],
        batch_size=int(document["batch_size"]),
        context_length=int(document["context_length"]),
        expert_count=int(document["expert_count"]),
        top_k=int(document["top_k"]),
        source_sha256=document["source_sha256"],
        steps=steps,
    )


def load_pinned_nemotron_profile(path: Path = PINNED_NEMOTRON_ROUTING) -> RoutingProfile:
    """Adapt the independently validated B200 compact trace to the common ABI."""

    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RoutingFormatError(f"cannot read pinned Nemotron routing: {error}") from error
    if document.get("contract") != "nemotron3-decode-routing-v1" or document.get("schema_version") != 1:
        raise RoutingFormatError("unexpected pinned Nemotron routing contract")
    shape = document["shape"]
    layer_ids = tuple(_layer_id(name) for name in document["layer_names"])
    if len(layer_ids) != shape["layers"] or len(set(layer_ids)) != len(layer_ids):
        raise RoutingFormatError("pinned Nemotron layer list is incomplete or duplicated")
    prefill = document["prefill_active_experts_by_layer"]
    decode = document["decode_topk_by_step"]
    if len(prefill) != len(layer_ids) or len(decode) != shape["recurrent_decode_steps"]:
        raise RoutingFormatError("pinned Nemotron routing dimensions are incomplete")

    steps = [
        RoutingStep(
            phase="prefill",
            index=0,
            token_count=shape["context_tokens"],
            active_experts_by_layer=tuple(
                (layer_id, tuple(sorted(set(experts)))) for layer_id, experts in zip(layer_ids, prefill, strict=True)
            ),
        )
    ]
    for index, rows in enumerate(decode):
        if len(rows) != len(layer_ids):
            raise RoutingFormatError(f"pinned Nemotron decode step {index} has incomplete layers")
        steps.append(
            RoutingStep(
                phase="decode",
                index=index,
                token_count=1,
                active_experts_by_layer=tuple(
                    (layer_id, tuple(sorted(set(experts)))) for layer_id, experts in zip(layer_ids, rows, strict=True)
                ),
            )
        )
    return RoutingProfile(
        model_key="nemotron3",
        model_id=document["source"]["model"],
        revision=document["source"]["revision"],
        batch_size=1,
        context_length=shape["context_tokens"],
        expert_count=shape["experts"],
        top_k=shape["top_k"],
        source_sha256=document["source"]["raw_routing_sha256"],
        steps=tuple(steps),
    )


def normalize_routing_jsonl(
    path: Path,
    *,
    case: str,
    model_key: str,
    model_id: str,
    revision: str,
    batch_size: int,
    context_length: int,
    expert_count: int,
    top_k: int,
    expected_layer_ids: tuple[int, ...],
) -> RoutingProfile:
    """Validate profiler JSONL and reduce it to expert sets used by DSE.

    This accepts the existing Nemotron hook schema and is also the required
    handoff format for future Kimi B2/B4/B8/B16 profiling.
    """

    events: dict[tuple[str, int, int], tuple[int, ...]] = {}
    token_counts: dict[tuple[str, int], int] = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise RoutingFormatError(f"cannot read routing JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RoutingFormatError(f"routing JSONL line {line_number} is invalid: {error}") from error
        if event.get("case") != case:
            continue
        phase = event.get("phase")
        step = int(event.get("phase_step", -1))
        layer_id = _layer_id(event.get("layer_name", ""))
        token_count = int(event.get("token_count", 0))
        ids = event.get("topk_expert_ids")
        weights = event.get("topk_weights")
        counts = event.get("routed_expert_token_counts")
        if event.get("top_k") != top_k or not isinstance(ids, list) or len(ids) != token_count:
            raise RoutingFormatError(f"line {line_number}: routing shape/top-k mismatch")
        derived: Counter[int] = Counter()
        for row_index, row in enumerate(ids):
            if len(row) != top_k or len(set(row)) != top_k:
                raise RoutingFormatError(f"line {line_number}: duplicate or missing top-k IDs")
            if any(not isinstance(expert, int) or not 0 <= expert < expert_count for expert in row):
                raise RoutingFormatError(f"line {line_number}: expert ID out of range")
            derived.update(row)
            if weights is not None:
                weight_row = weights[row_index]
                if len(weight_row) != top_k or not math.isclose(sum(weight_row), 1.0, abs_tol=2e-7):
                    raise RoutingFormatError(f"line {line_number}: routing weights are not normalized")
        if counts is not None and [derived[index] for index in range(expert_count)] != counts:
            raise RoutingFormatError(f"line {line_number}: expert counts disagree with top-k IDs")
        key = (phase, step, layer_id)
        if key in events:
            raise RoutingFormatError(f"duplicate routing event {key}")
        events[key] = tuple(sorted(derived))
        step_key = (phase, step)
        if step_key in token_counts and token_counts[step_key] != token_count:
            raise RoutingFormatError(f"routing step {step_key} has inconsistent token counts")
        token_counts[step_key] = token_count

    if not events:
        raise RoutingFormatError(f"routing case {case!r} has no events")
    expected = tuple(sorted(expected_layer_ids))
    grouped: dict[tuple[str, int], dict[int, tuple[int, ...]]] = defaultdict(dict)
    for (phase, step, layer_id), experts in events.items():
        grouped[(phase, step)][layer_id] = experts
    steps = []
    for (phase, step), layers in sorted(grouped.items(), key=lambda item: (item[0][0] != "prefill", item[0][1])):
        if tuple(sorted(layers)) != expected:
            raise RoutingFormatError(f"routing {phase} step {step} does not cover the expected layers")
        steps.append(
            RoutingStep(
                phase=phase,
                index=step,
                token_count=token_counts[(phase, step)],
                active_experts_by_layer=tuple((layer_id, layers[layer_id]) for layer_id in expected),
            )
        )
    return RoutingProfile(
        model_key=model_key,
        model_id=model_id,
        revision=revision,
        batch_size=batch_size,
        context_length=context_length,
        expert_count=expert_count,
        top_k=top_k,
        source_sha256=_sha256(path),
        steps=tuple(steps),
    )
