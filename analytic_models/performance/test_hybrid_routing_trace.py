from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from .hybrid_routing_trace import KIMI_MOE_LAYER_IDS, load_kimi_routing_trace
from .hybrid_system_timeline import HybridSystemTimelineModel, ModelFamily, SystemDesign
from .nemotron3_workload import InferencePhase


def _document(*, decode_steps: int = 2) -> dict:
    events = [
        {
            "phase": "prefill",
            "step": 0,
            "layer_id": layer_id,
            "active_experts": list(range(32)),
        }
        for layer_id in KIMI_MOE_LAYER_IDS
    ]
    events.extend(
        {
            "phase": "decode",
            "step": step,
            "layer_id": layer_id,
            "active_experts": [(layer_id * 19 + step * 7 + rank * 31) % 896 for rank in range(16)],
        }
        for step in range(decode_steps)
        for layer_id in KIMI_MOE_LAYER_IDS
    )
    return {
        "schema_version": 1,
        "model": {
            "name": "moonshotai/Kimi-K3",
            "revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
            "experts": 896,
            "top_k": 16,
        },
        "source": {
            "prompt_sha256": "a" * 64,
            "phase": "both",
            "batch_size": 1,
        },
        "events": events,
    }


def _write(tmp_path, document: dict):
    path = tmp_path / "kimi-routing.json"
    path.write_text(json.dumps(document))
    return path


def test_valid_trace_is_indexed_and_covers_all_moe_layers(tmp_path) -> None:
    trace = load_kimi_routing_trace(_write(tmp_path, _document()))
    assert trace.steps_by_phase == {
        InferencePhase.PREFILL: 1,
        InferencePhase.DECODE: 2,
    }
    assert len(trace.events) == 3 * len(KIMI_MOE_LAYER_IDS)
    assert len(trace.experts(InferencePhase.PREFILL, 0, 1)) == 32
    assert len(trace.experts(InferencePhase.DECODE, 1, 92)) == 16
    assert trace.coverage()["status"] == "empirical"


def test_timeline_uses_empirical_expert_ids_without_changing_weight_shape(tmp_path) -> None:
    path = _write(tmp_path, _document())
    model = HybridSystemTimelineModel(
        ModelFamily.KIMI_K3,
        SystemDesign(),
        kimi_routing_trace_path=path,
    )
    instance = SimpleNamespace(
        phase=InferencePhase.DECODE,
        token_index=1,
    )
    assert model._experts(instance, 7) == tuple((7 * 19 + 7 + rank * 31) % 896 for rank in range(16))
    cache = model._moe_cache(InferencePhase.DECODE, prompt_tokens=128)
    assert cache.routing_source == "empirical_kimi_top16"
    assert cache.entry_bytes > 0


def test_complete_decode_timeline_consumes_every_empirical_layer_event(tmp_path) -> None:
    path = _write(tmp_path, _document(decode_steps=1))
    report = HybridSystemTimelineModel(
        ModelFamily.KIMI_K3,
        SystemDesign(),
        kimi_routing_trace_path=path,
    ).simulate(
        InferencePhase.DECODE,
        context_length=16,
        decode_tokens=1,
        include_embedding=False,
        include_lm_head=False,
    )
    assert report.moe_cache.routing_source == "empirical_kimi_top16"
    assert report.moe_cache.accesses == 92 * (16 + 2)
    assert "validated empirical top-16 trace" in report.limits[-1]


@pytest.mark.parametrize("failure", ["duplicate", "out_of_range", "missing_layer"])
def test_invalid_expert_or_layer_coverage_is_rejected(tmp_path, failure) -> None:
    document = _document()
    if failure == "duplicate":
        document["events"][len(KIMI_MOE_LAYER_IDS)]["active_experts"][1] = document["events"][len(KIMI_MOE_LAYER_IDS)][
            "active_experts"
        ][0]
    elif failure == "out_of_range":
        document["events"][0]["active_experts"][0] = 896
    else:
        document["events"] = [
            event
            for event in document["events"]
            if not (event["phase"] == "decode" and event["step"] == 1 and event["layer_id"] == 92)
        ]
    with pytest.raises(ValueError):
        load_kimi_routing_trace(_write(tmp_path, document))


def test_declared_phase_and_event_phases_must_match(tmp_path) -> None:
    document = _document()
    document["source"]["phase"] = "decode"
    with pytest.raises(ValueError, match="event phases do not match"):
        load_kimi_routing_trace(_write(tmp_path, document))


def test_wrong_model_revision_is_rejected(tmp_path) -> None:
    document = _document()
    document["model"]["revision"] = "0" * 40
    with pytest.raises(ValueError, match="routing trace revision"):
        load_kimi_routing_trace(_write(tmp_path, document))


def test_missing_requested_step_fails_instead_of_reusing_routing(tmp_path) -> None:
    trace = load_kimi_routing_trace(_write(tmp_path, _document(decode_steps=1)))
    with pytest.raises(ValueError, match="has no decode step=1"):
        trace.experts(InferencePhase.DECODE, 1, 1)
