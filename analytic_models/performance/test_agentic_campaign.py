from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from .agentic_campaign import (
    AgenticCampaignContract,
    AgenticCampaignFormatError,
    load_agentic_campaign,
)


TEST_CONTRACT = AgenticCampaignContract(
    model_id="test/Nemotron",
    revision="test-revision",
    benchmark_sample_counts=(("bfcl_v3", 2),),
    batch_sizes=(1, 2),
    moe_layer_ids=(1, 3),
    expert_count=4,
    top_k=2,
    decode_steps=2,
    measurements=2,
    generation_limit=2,
)


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _counts(rows: list[list[int]]) -> list[int]:
    counts = [0] * TEST_CONTRACT.expert_count
    for row in rows:
        for expert in row:
            counts[expert] += 1
    return counts


def _routing_event(
    *,
    sample: dict,
    phase: str,
    step: int,
    layer: int,
    rows: list[list[int]],
) -> dict:
    return {
        "benchmark": sample["benchmark"],
        "sample_id": sample["sample_id"],
        "batch": 1,
        "prompt_length": sample["prompt_length"],
        "prompt_sha256": sample["prompt_sha256"],
        "phase": phase,
        "decode_step": step,
        "layer_index": layer,
        "layer_name": f"model.layers.{layer}.mixer.experts",
        "model_call_index": 0,
        "token_count": len(rows),
        "top_k": 2,
        "topk_expert_ids": rows,
        "topk_weights": [[0.6, 0.4] for _ in rows],
        "expert_counts_128": _counts(rows),
        "shared_expert_token_count": len(rows),
        "generated_token_ids": sample["generated_token_ids"],
        "generated_tokens": len(sample["generated_token_ids"]),
        "finish_reason": "length",
    }


def _timing_rows(samples: list[dict]) -> list[dict]:
    rows = []
    for batch_size in TEST_CONTRACT.batch_sizes:
        groups = [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]
        for group_index, members in enumerate(groups):
            padded = max(sample["prompt_length"] for sample in members)
            for measurement in range(TEST_CONTRACT.measurements):
                trial_id = f"bfcl_v3:B{batch_size}:G{group_index}:measurement:{measurement}"
                for request_id, sample in enumerate(members):
                    rows.append(
                        {
                            "benchmark": "bfcl_v3",
                            "mode": "batch_sweep",
                            "batch_size": batch_size,
                            "group_index": group_index,
                            "group_number": group_index,
                            "trial_id": trial_id,
                            "request_id": str(request_id),
                            "sample_id": sample["sample_id"],
                            "include_in_summary": True,
                            "record_type": "request",
                            "phase": "measurement",
                            "measurement": measurement,
                            "prompt_length": sample["prompt_length"],
                            "padded_prompt_length": padded,
                            "generation_limit": 2,
                            "ttft_ms": 10.0 + request_id,
                            "itl_ms": [2.0, 3.0],
                            "e2e_ms": 20.0 + request_id,
                            "batch_e2e_ms": 21.0,
                            "batch_throughput_tokens_per_s": 100.0 * batch_size,
                            "batch_energy_joules": 4.0 * batch_size,
                        }
                    )
    return rows


def _refresh_checksums(root: Path) -> None:
    names = (
        "manifest.json",
        "validation.json",
        "samples.json",
        "latency_raw.jsonl",
        "routing_raw.jsonl",
        "token_traces.jsonl",
        "token_traces_timing.jsonl",
    )
    lines = []
    for name in names:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    (root / "SHA256SUMS").write_text("".join(lines))


def _campaign_fixture(root: Path) -> Path:
    samples = [
        {
            "benchmark": "bfcl_v3",
            "sample_id": "bfcl_v3:a",
            "prompt_length": 2,
            "prompt_sha256": "a" * 64,
            "prompt_token_ids": [1, 2],
            "generated_token_ids": [10, 11],
        },
        {
            "benchmark": "bfcl_v3",
            "sample_id": "bfcl_v3:b",
            "prompt_length": 3,
            "prompt_sha256": "b" * 64,
            "prompt_token_ids": [3, 4, 5],
            "generated_token_ids": [12, 13],
        },
    ]
    _write_json(
        root / "manifest.json",
        {
            "status": "COMPLETE",
            "model": {
                "id": TEST_CONTRACT.model_id,
                "revision": TEST_CONTRACT.revision,
                "checkpoint_type": "real official NVFP4 checkpoint",
            },
            "routing": {
                "enforce_eager": True,
                "gpu_uuid": "GPU-test",
                "raw_event_count": 12,
            },
            "timing": {"enforce_eager": False, "gpu_uuid": "GPU-test"},
        },
    )
    _write_json(root / "validation.json", {"status": "PASS"})
    _write_json(
        root / "samples.json",
        {
            "samples": [
                {key: value for key, value in sample.items() if key != "generated_token_ids"} for sample in samples
            ]
        },
    )
    _write_jsonl(
        root / "token_traces.jsonl",
        [
            {
                "benchmark": sample["benchmark"],
                "sample_id": sample["sample_id"],
                "prompt_sha256": sample["prompt_sha256"],
                "routing_generated_token_ids": sample["generated_token_ids"],
                "routing_generated_tokens": 2,
                "timing_generated_token_ids": list(reversed(sample["generated_token_ids"])),
                "timing_vs_routing_identical": False,
            }
            for sample in samples
        ],
    )
    _write_jsonl(
        root / "token_traces_timing.jsonl",
        [
            {
                "benchmark": sample["benchmark"],
                "sample_id": sample["sample_id"],
                "prompt_sha256": sample["prompt_sha256"],
                "generated_token_ids": list(reversed(sample["generated_token_ids"])),
                "generated_tokens": 2,
                "source": "fixture",
            }
            for sample in samples
        ],
    )
    _write_jsonl(root / "latency_raw.jsonl", _timing_rows(samples))

    route_patterns = {
        "bfcl_v3:a": {
            0: {1: [0, 1], 3: [1, 2]},
            1: {1: [0, 2], 3: [2, 3]},
        },
        "bfcl_v3:b": {
            0: {1: [2, 3], 3: [0, 3]},
            1: {1: [1, 3], 3: [0, 1]},
        },
    }
    events = []
    for sample in samples:
        prefill_rows = [[0, 1] if index % 2 == 0 else [2, 3] for index in range(sample["prompt_length"])]
        for layer in TEST_CONTRACT.moe_layer_ids:
            events.append(_routing_event(sample=sample, phase="prefill", step=-1, layer=layer, rows=prefill_rows))
        for step, layers in route_patterns[sample["sample_id"]].items():
            for layer, ids in layers.items():
                events.append(_routing_event(sample=sample, phase="decode", step=step, layer=layer, rows=[ids]))
    _write_jsonl(root / "routing_raw.jsonl", events)
    _write_json(
        root / "campaign_summary.json",
        {"routing": {"timing_vs_routing_identical_samples": 0}},
    )
    _refresh_checksums(root)
    return root


def test_agentic_campaign_rebuilds_gpu_batch_membership_and_route_union(tmp_path: Path) -> None:
    campaign = load_agentic_campaign(
        _campaign_fixture(tmp_path),
        contract=TEST_CONTRACT,
    )
    assert campaign.timing_and_routing_tokens_identical_samples == 0
    assert campaign.to_summary()["batch_group_counts"] == {
        "bfcl_v3_b1": 2,
        "bfcl_v3_b2": 1,
    }
    group = campaign.group("bfcl_v3", 2, 0)
    assert group.sample_ids == ("bfcl_v3:a", "bfcl_v3:b")
    assert group.prompt_lengths == (2, 3)
    assert group.padding_token_overhead == 1
    assert group.gpu_itl_ms_median == 2.5

    profile = campaign.routing_profile(group, decode_steps=2)
    assert profile.batch_size == 2
    assert profile.context_length == 3
    assert profile.step("decode", 0).active_experts_by_layer == (
        (1, (0, 1, 2, 3)),
        (3, (0, 1, 2, 3)),
    )
    assert profile.step("decode", 1).active_experts_by_layer == (
        (1, (0, 1, 2, 3)),
        (3, (0, 1, 2, 3)),
    )


def test_agentic_campaign_rejects_route_counts_that_disagree_with_ids(tmp_path: Path) -> None:
    root = _campaign_fixture(tmp_path)
    rows = [json.loads(line) for line in (root / "routing_raw.jsonl").read_text().splitlines()]
    decode = next(row for row in rows if row["phase"] == "decode")
    decode["expert_counts_128"] = [2, 0, 0, 0]
    _write_jsonl(root / "routing_raw.jsonl", rows)
    _refresh_checksums(root)
    with pytest.raises(AgenticCampaignFormatError, match="expert counts disagree"):
        load_agentic_campaign(root, contract=TEST_CONTRACT)


def test_agentic_campaign_never_substitutes_optimized_timing_tokens_for_routing(tmp_path: Path) -> None:
    root = _campaign_fixture(tmp_path)
    traces = [json.loads(line) for line in (root / "token_traces.jsonl").read_text().splitlines()]
    traces[0]["routing_generated_token_ids"] = [99, 98]
    _write_jsonl(root / "token_traces.jsonl", traces)
    _refresh_checksums(root)
    with pytest.raises(AgenticCampaignFormatError, match="routing event and eager token trace disagree"):
        load_agentic_campaign(root, contract=TEST_CONTRACT)


def test_agentic_campaign_cross_checks_the_independent_timing_trace(tmp_path: Path) -> None:
    root = _campaign_fixture(tmp_path)
    traces = [json.loads(line) for line in (root / "token_traces_timing.jsonl").read_text().splitlines()]
    traces[0]["generated_token_ids"] = [77, 78]
    _write_jsonl(root / "token_traces_timing.jsonl", traces)
    _refresh_checksums(root)
    with pytest.raises(AgenticCampaignFormatError, match="optimized timing token traces disagree"):
        load_agentic_campaign(root, contract=TEST_CONTRACT)


def test_agentic_campaign_rejects_a_file_changed_after_manifest_hashing(tmp_path: Path) -> None:
    root = _campaign_fixture(tmp_path)
    with (root / "latency_raw.jsonl").open("a") as destination:
        destination.write("{}\n")
    with pytest.raises(AgenticCampaignFormatError, match=r"checksum mismatch for latency_raw\.jsonl"):
        load_agentic_campaign(root, contract=TEST_CONTRACT)
