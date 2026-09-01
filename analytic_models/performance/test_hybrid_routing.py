from __future__ import annotations

import json
from pathlib import Path

import pytest

from .hybrid_routing import (
    RoutingFormatError,
    load_pinned_nemotron_profile,
    load_profile,
    normalize_routing_jsonl,
)


def _event(*, phase: str, step: int, layer: int, rows: list[list[int]]) -> dict:
    counts = [0, 0, 0, 0]
    for row in rows:
        for expert in row:
            counts[expert] += 1
    return {
        "case": "decode_b2",
        "layer_name": f"model.layers.{layer}.mixer.experts",
        "phase": phase,
        "phase_step": step,
        "token_count": len(rows),
        "top_k": 2,
        "topk_expert_ids": rows,
        "topk_weights": [[0.6, 0.4] for _ in rows],
        "routed_expert_token_counts": counts,
    }


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events))


def test_pinned_nemotron_trace_has_exact_layer_and_step_coverage() -> None:
    profile = load_pinned_nemotron_profile()
    assert profile.model_key == "nemotron3"
    assert profile.batch_size == 1
    assert profile.context_length == 2048
    assert len(profile.steps) == 128
    assert len(profile.step("prefill", 0).active_experts_by_layer) == 23
    assert set(dict(profile.step("decode", 126).unique_experts_by_layer).values()) == {6}
    prefill_counts = dict(profile.step("prefill", 0).unique_experts_by_layer)
    assert min(prefill_counts.values()) == 105
    assert max(prefill_counts.values()) == 128
    profile.validate_replay(
        model_key="nemotron3",
        phase="decode",
        batch_size=1,
        context_length=2048,
        sequence_length=1,
        decode_steps=127,
    )


def test_generic_jsonl_normalizer_preserves_batch_expert_union(tmp_path: Path) -> None:
    source = tmp_path / "routing.jsonl"
    _write_jsonl(
        source,
        [
            _event(phase="decode", step=0, layer=1, rows=[[0, 1], [1, 2]]),
            _event(phase="decode", step=0, layer=3, rows=[[2, 3], [0, 3]]),
            _event(phase="decode", step=1, layer=1, rows=[[0, 2], [0, 3]]),
            _event(phase="decode", step=1, layer=3, rows=[[1, 2], [2, 3]]),
        ],
    )
    profile = normalize_routing_jsonl(
        source,
        case="decode_b2",
        model_key="kimi_k3",
        model_id="moonshotai/Kimi-K3",
        revision="test-revision",
        batch_size=2,
        context_length=2048,
        expert_count=4,
        top_k=2,
        expected_layer_ids=(1, 3),
    )
    assert profile.step("decode", 0).active_experts_by_layer == (
        (1, (0, 1, 2)),
        (3, (0, 2, 3)),
    )
    profile.validate_replay(
        model_key="kimi_k3",
        phase="decode",
        batch_size=2,
        context_length=2048,
        sequence_length=1,
        decode_steps=2,
    )

    normalized = tmp_path / "normalized.json"
    normalized.write_text(json.dumps(profile.to_dict()))
    assert load_profile(normalized) == profile


def test_jsonl_normalizer_rejects_counts_that_disagree_with_ids(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    event = _event(phase="decode", step=0, layer=1, rows=[[0, 1], [1, 2]])
    event["routed_expert_token_counts"] = [4, 0, 0, 0]
    _write_jsonl(source, [event])
    with pytest.raises(RoutingFormatError, match="counts disagree"):
        normalize_routing_jsonl(
            source,
            case="decode_b2",
            model_key="kimi_k3",
            model_id="moonshotai/Kimi-K3",
            revision="test-revision",
            batch_size=2,
            context_length=2048,
            expert_count=4,
            top_k=2,
            expected_layer_ids=(1,),
        )


def test_profile_rejects_batch_or_model_mismatch() -> None:
    profile = load_pinned_nemotron_profile()
    with pytest.raises(RoutingFormatError, match="cannot replay"):
        profile.validate_replay(
            model_key="kimi_k3",
            phase="decode",
            batch_size=1,
            context_length=2048,
            sequence_length=1,
            decode_steps=1,
        )
    with pytest.raises(RoutingFormatError, match="batch/context"):
        profile.validate_replay(
            model_key="nemotron3",
            phase="decode",
            batch_size=2,
            context_length=2048,
            sequence_length=1,
            decode_steps=1,
        )
