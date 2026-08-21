"""Validate and index empirical Kimi K3 top-16 routing traces."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .nemotron3_workload import InferencePhase


KIMI_EXPERTS = 896
KIMI_TOP_K = 16
KIMI_MOE_LAYER_IDS = tuple(range(1, 93))
KIMI_HF_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"


@dataclass(frozen=True)
class KimiRoutingTrace:
    path: Path
    model_revision: str
    prompt_sha256: str
    events: dict[tuple[InferencePhase, int, int], tuple[int, ...]]
    steps_by_phase: dict[InferencePhase, int]

    def experts(
        self,
        phase: InferencePhase,
        step: int,
        layer_id: int,
    ) -> tuple[int, ...]:
        try:
            return self.events[(phase, step, layer_id)]
        except KeyError as error:
            raise ValueError(f"Kimi routing trace has no {phase.value} step={step} layer={layer_id}") from error

    def coverage(self) -> dict[str, Any]:
        return {
            "status": "empirical",
            "path": str(self.path),
            "model_revision": self.model_revision,
            "prompt_sha256": self.prompt_sha256,
            "moe_layers": len(KIMI_MOE_LAYER_IDS),
            "steps": {phase.value: count for phase, count in self.steps_by_phase.items()},
            "events": len(self.events),
        }


@lru_cache(maxsize=8)
def load_kimi_routing_trace(path: Path) -> KimiRoutingTrace:
    path = path.resolve()
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("Kimi routing trace schema_version must be 1")
    model = document.get("model", {})
    if model.get("name") != "moonshotai/Kimi-K3":
        raise ValueError("Kimi routing trace has the wrong model name")
    if model.get("experts") != KIMI_EXPERTS or model.get("top_k") != KIMI_TOP_K:
        raise ValueError("Kimi routing trace shape must be 896 experts / top-16")
    revision = model.get("revision")
    source = document.get("source", {})
    prompt_sha256 = source.get("prompt_sha256")
    if revision != KIMI_HF_REVISION:
        raise ValueError(f"Kimi routing trace revision must be {KIMI_HF_REVISION}")
    if not isinstance(prompt_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", prompt_sha256) is None:
        raise ValueError("Kimi routing trace must carry a prompt SHA256")

    declared_phase = source.get("phase")
    if declared_phase not in {"prefill", "decode", "both"}:
        raise ValueError("Kimi routing trace source phase must be prefill, decode, or both")
    if source.get("batch_size") != 1:
        raise ValueError("Kimi routing trace v1 supports batch_size=1 only")

    raw_events = document.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("Kimi routing trace has no events")
    events: dict[tuple[InferencePhase, int, int], tuple[int, ...]] = {}
    for raw in raw_events:
        try:
            phase = InferencePhase(raw["phase"])
            step = int(raw["step"])
            layer_id = int(raw["layer_id"])
            experts = tuple(int(value) for value in raw["active_experts"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("malformed Kimi routing event") from error
        if step < 0 or layer_id not in KIMI_MOE_LAYER_IDS:
            raise ValueError("Kimi routing event has an invalid step or MoE layer")
        if not experts or len(experts) > KIMI_EXPERTS:
            raise ValueError("Kimi routing event has an invalid active-expert count")
        if phase == InferencePhase.DECODE and len(experts) != KIMI_TOP_K:
            raise ValueError("Kimi decode event must contain exactly top-16 experts")
        if len(set(experts)) != len(experts):
            raise ValueError("Kimi routing event repeats an expert")
        if any(not 0 <= expert < KIMI_EXPERTS for expert in experts):
            raise ValueError("Kimi routing event has an out-of-range expert")
        key = (phase, step, layer_id)
        if key in events:
            raise ValueError("Kimi routing trace repeats a phase/step/layer event")
        events[key] = experts

    steps_by_phase: dict[InferencePhase, int] = {}
    for phase in set(key[0] for key in events):
        steps = sorted(set(key[1] for key in events if key[0] == phase))
        if steps != list(range(len(steps))):
            raise ValueError(f"Kimi {phase.value} routing steps must start at zero and be contiguous")
        for step in steps:
            layers = {key[2] for key in events if key[:2] == (phase, step)}
            if layers != set(KIMI_MOE_LAYER_IDS):
                missing = sorted(set(KIMI_MOE_LAYER_IDS) - layers)
                raise ValueError(
                    f"Kimi {phase.value} step {step} does not cover all 92 MoE layers; missing {missing[:4]}"
                )
        steps_by_phase[phase] = len(steps)

    present_phases = {phase.value for phase in steps_by_phase}
    expected_phases = {"prefill", "decode"} if declared_phase == "both" else {declared_phase}
    if present_phases != expected_phases:
        raise ValueError(
            "Kimi routing trace event phases do not match source.phase: "
            f"declared={declared_phase}, present={sorted(present_phases)}"
        )

    return KimiRoutingTrace(
        path=path,
        model_revision=revision,
        prompt_sha256=prompt_sha256,
        events=events,
        steps_by_phase=steps_by_phase,
    )


__all__ = [
    "KIMI_EXPERTS",
    "KIMI_HF_REVISION",
    "KIMI_MOE_LAYER_IDS",
    "KIMI_TOP_K",
    "KimiRoutingTrace",
    "load_kimi_routing_trace",
]
