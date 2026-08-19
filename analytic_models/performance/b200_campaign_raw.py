"""Ingest and independently validate the complete B200 profiling campaign.

The raw archive is intentionally kept outside Git.  This module converts it
into a small, versioned workload-calibration contract and a compact routing
trace that can be replayed by DSE without the 46 MiB JSONL source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .profile_paths import profile_relative_path


EXPECTED_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
EXPECTED_NEMOTRON_REVISION = "ce1b118ae66ec705d02c241525192832eb045fd3"
EXPECTED_KIMI_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
EXPECTED_KIMI_COMMIT = "3cb39dfd32e51c3328e2e4b4af21341247d06c43"
EXPECTED_FLASHKDA_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_LAYER_COUNTS = {"Mamba": 23, "Attention": 6, "MoE": 23}
EXPECTED_KDA_CASES = {
    "prefill_b1_s2048": (1, 2048),
    "decode_b1": (1, 1),
    "decode_b8": (8, 1),
}
EXPECTED_KDA_STAGES = (
    "qkv_projection",
    "short_conv",
    "decay_beta",
    "kda_state_update_output",
    "gate_rmsnorm",
    "out_projection",
)
EXPECTED_NEMOTRON_NCU = {
    ("Prefill", "Mamba"),
    ("Prefill", "Attention"),
    ("Prefill", "MoE"),
    ("Decode", "Mamba"),
    ("Decode", "Attention"),
    ("Decode", "MoE"),
}
EXPECTED_ROUTING_EVENTS = {
    "prefill_s128": 23,
    "prefill_s2048": 23,
    "prefill_s8192": 23,
    "decode_s2048_128": 23 * 128,
}
MATRIX_KDA_STAGES = {"qkv_projection", "gate_rmsnorm", "out_projection"}
_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.mixer\.experts")


class RawCampaignError(ValueError):
    pass


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RawCampaignError(f"cannot read {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise RawCampaignError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _close(actual: float, expected: float, *, label: str, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RawCampaignError(f"{label}: {actual} != {expected}")


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise RawCampaignError("cannot take a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _weighted(records: list[dict[str, Any]], key: str) -> float:
    duration = sum(record["duration_ns"] for record in records)
    if duration <= 0:
        raise RawCampaignError("duration-weighted metric has no positive duration")
    return sum(record[key] * record["duration_ns"] for record in records) / duration


def _remote_to_local(profiles_root: Path, remote: str) -> Path:
    try:
        relative = profile_relative_path(remote)
    except ValueError as error:
        raise RawCampaignError(
            f"artifact is outside the campaign profile root: {remote}"
        ) from error
    return profiles_root / relative


def _validate_checksum_file(root: Path, checksum_file: Path) -> int:
    checked = 0
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError as error:
            raise RawCampaignError(f"malformed checksum line in {checksum_file}: {line!r}") from error
        relative = relative.removeprefix("*").removeprefix("./")
        source = root / relative
        if _sha256(source) != expected:
            raise RawCampaignError(f"checksum mismatch: {source}")
        checked += 1
    return checked


def _validate_manifest_artifacts(campaign_root: Path, manifest: dict[str, Any]) -> int:
    profiles_root = campaign_root.parents[1]
    checked = 0
    collections = manifest.get("collections")
    if not isinstance(collections, dict):
        raise RawCampaignError("campaign manifest has no collections")
    for entries in collections.values():
        if not isinstance(entries, list):
            raise RawCampaignError("campaign collection is not a list")
        for entry in entries:
            artifacts = entry.get("artifacts")
            if not isinstance(artifacts, dict):
                raise RawCampaignError("campaign collection has no artifact map")
            for artifact in artifacts.values():
                source = _remote_to_local(profiles_root, artifact["path"])
                if source.stat().st_size != artifact["size_bytes"]:
                    raise RawCampaignError(f"artifact size mismatch: {source}")
                if _sha256(source) != artifact["sha256"]:
                    raise RawCampaignError(f"artifact hash mismatch: {source}")
                checked += 1
    return checked


def _aggregate_kernels(kernels: list[dict[str, Any]]) -> dict[str, Any]:
    launches = [launch for kernel in kernels for launch in kernel.get("launches", [])]
    if not launches:
        raise RawCampaignError("KDA stage has no kernel launches")
    return {
        "calls": len(launches),
        "time_ms": sum(launch["duration_ns"] for launch in launches) / 1e6,
        "dram_read_bytes": sum(launch["dram_read_bytes"] for launch in launches),
        "dram_write_bytes": sum(launch["dram_write_bytes"] for launch in launches),
        "l2_read_sectors": sum(launch["l2_read_sectors"] for launch in launches),
        "l2_write_sectors": sum(launch["l2_write_sectors"] for launch in launches),
        "sm_throughput_pct": _weighted(launches, "sm_throughput_pct"),
        "memory_throughput_pct": _weighted(launches, "memory_throughput_pct"),
        "achieved_occupancy_pct": _weighted(launches, "achieved_occupancy_pct"),
        "theoretical_occupancy_pct": _weighted(launches, "theoretical_occupancy_pct"),
        "registers_per_thread": _weighted(launches, "registers_per_thread"),
        "unique_kernel_names": len({kernel["kernel_name"] for kernel in kernels}),
    }


def _load_kda(campaign_root: Path) -> dict[str, Any]:
    manifest = _load_json(campaign_root / "kda/stage2-environment-source-manifest.json")
    source = manifest.get("source", {})
    if source.get("kimi_k3", {}).get("commit") != EXPECTED_KIMI_COMMIT:
        raise RawCampaignError("unexpected Kimi source commit")
    if source.get("flashkda", {}).get("commit") != EXPECTED_FLASHKDA_COMMIT:
        raise RawCampaignError("unexpected FlashKDA source commit")
    if source.get("huggingface_kimi_k3", {}).get("revision") != EXPECTED_KIMI_REVISION:
        raise RawCampaignError("unexpected Kimi Hugging Face revision")
    if manifest.get("environments", {}).get("isolation_verified") is not True:
        raise RawCampaignError("KDA and Nemotron environments were not isolated")

    comparison = _load_json(campaign_root / "kda/kda_custom_vs_official.json")
    if comparison.get("weight_mapping_bit_identical") is not True:
        raise RawCampaignError("KDA weight mapping is not bit-identical")
    comparison_cases = comparison.get("comparisons", [])
    if {case.get("sequence_length") for case in comparison_cases} != {1, 16, 256, 2048}:
        raise RawCampaignError("KDA official comparison coverage is incomplete")
    for case in comparison_cases:
        if case["output"]["max_abs"] != 0.0 or case["recurrent_state"]["max_abs"] != 0.0:
            raise RawCampaignError("KDA official output/state comparison failed")
        if any(case["conv_state"][field]["max_abs"] != 0.0 for field in ("q", "k", "v")):
            raise RawCampaignError("KDA official conv-state comparison failed")

    layout = _load_json(campaign_root / "kda/kda_projection_layout.json")
    fields = layout.get("fields", [])
    if [field.get("field_name") for field in fields] != [
        "q",
        "k",
        "v",
        "decay_low_rank",
        "decay_g",
        "beta",
        "output_gate",
        "output",
    ]:
        raise RawCampaignError("unexpected KDA projection field order")
    if any(field.get("runtime_hook", {}).get("output", {}).get("contiguous") is not True for field in fields):
        raise RawCampaignError("KDA projection runtime hook is incomplete")

    raw_stages = _load_json(campaign_root / "kda/summary/kda_ncu_summary.json")
    if not isinstance(raw_stages, list) or len(raw_stages) != 18:
        raise RawCampaignError("KDA NCU summary must contain 18 case/stage records")
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in raw_stages:
        if record.get("collection_status") != "OK":
            raise RawCampaignError("KDA NCU collection is not OK")
        case = record.get("case")
        stage = record.get("stage")
        if case not in EXPECTED_KDA_CASES or stage not in EXPECTED_KDA_STAGES:
            raise RawCampaignError(f"unexpected KDA case/stage: {case}/{stage}")
        if stage in by_case[case]:
            raise RawCampaignError(f"duplicate KDA case/stage: {case}/{stage}")
        by_case[case][stage] = _aggregate_kernels(record["kernels"])

    cases = {}
    for name, (batch, sequence_length) in EXPECTED_KDA_CASES.items():
        stages = by_case[name]
        if set(stages) != set(EXPECTED_KDA_STAGES):
            raise RawCampaignError(f"{name}: incomplete KDA stage coverage")
        totals = {
            "kernel_calls": sum(stage["calls"] for stage in stages.values()),
            "kernel_time_ms": sum(stage["time_ms"] for stage in stages.values()),
            "dram_read_bytes": sum(stage["dram_read_bytes"] for stage in stages.values()),
            "dram_write_bytes": sum(stage["dram_write_bytes"] for stage in stages.values()),
            "l2_read_sectors": sum(stage["l2_read_sectors"] for stage in stages.values()),
            "l2_write_sectors": sum(stage["l2_write_sectors"] for stage in stages.values()),
        }
        cases[name] = {
            "batch": batch,
            "sequence_length": sequence_length,
            **totals,
            "stages": stages,
        }

    return {
        "source": {
            "kimi_commit": EXPECTED_KIMI_COMMIT,
            "flashkda_commit": EXPECTED_FLASHKDA_COMMIT,
            "huggingface_revision": EXPECTED_KIMI_REVISION,
        },
        "shape": {
            "hidden_size": 7168,
            "num_heads": 96,
            "head_dim": 128,
            "conv_kernel": 4,
            "recurrent_state_dtype": "fp32",
            "recurrent_state_mib_per_batch_layer": 6.0,
            "conv_state_dtype": "bf16",
            "conv_state_mib_per_batch_layer": 0.28125,
        },
        "official_comparison": {
            "sequence_lengths": [1, 16, 256, 2048],
            "output_max_abs": 0.0,
            "recurrent_state_max_abs": 0.0,
            "conv_state_max_abs": 0.0,
            "bit_identical_weight_mappings": 14,
        },
        "projection_storage": "eight_independent_contiguous_tensors_not_packed_qkv",
        "projection_fields": [
            {
                "name": field["field_name"],
                "elements_per_token": field["projection_output_length_elements_per_token"],
                "dtype": field["projection_output_dtype"],
                "decode_consumer_order": field["decode_consumer_order"],
                "consumer": field["decode_consumer"],
            }
            for field in fields
        ],
        "cases": cases,
        "reported_sha256": {
            "projection_layout": _sha256(campaign_root / "kda/kda_projection_layout.json"),
            "official_comparison": _sha256(campaign_root / "kda/kda_custom_vs_official.json"),
            "ncu_summary": _sha256(campaign_root / "kda/summary/kda_ncu_summary.json"),
        },
    }


def _load_latency(campaign_root: Path) -> dict[str, Any]:
    source = campaign_root / "nemotron/timing-routing-data/timing/latency-raw.jsonl"
    reported = _load_json(campaign_root / "nemotron/timing-routing-data/timing/latency-summary.json")
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        for line in source.read_text().splitlines():
            record = json.loads(line)
            records[record["case"]].append(record)
    except (OSError, json.JSONDecodeError, KeyError) as error:
        raise RawCampaignError(f"cannot parse latency raw file: {error}") from error
    if set(records) != set(reported) or any(len(case) != 20 for case in records.values()):
        raise RawCampaignError("latency raw coverage is not four cases x 20 measurements")

    normalized = {}
    for name, rows in records.items():
        ttft = [float(row["ttft_ms"]) for row in rows]
        itl = [float(value) for row in rows for value in row["itl_ms"]]
        summary = reported[name]
        _close(statistics.median(ttft), summary["ttft_ms_median"], label=f"{name} TTFT median")
        _close(_percentile(ttft, 0.95), summary["ttft_ms_p95"], label=f"{name} TTFT p95")
        if len(itl) != summary["itl_sample_count"]:
            raise RawCampaignError(f"{name}: ITL sample count mismatch")
        if itl:
            _close(statistics.median(itl), summary["itl_ms_median"], label=f"{name} ITL median")
            _close(_percentile(itl, 0.95), summary["itl_ms_p95"], label=f"{name} ITL p95")
        normalized[name] = {
            "prompt_tokens": summary["prompt_tokens"],
            "output_tokens": summary["output_tokens"],
            "measurements": summary["measurements"],
            "ttft_median_ms": summary["ttft_ms_median"],
            "ttft_p95_ms": summary["ttft_ms_p95"],
            "itl_sample_count": len(itl),
            "itl_median_ms": summary["itl_ms_median"],
            "itl_p95_ms": summary["itl_ms_p95"],
            "decode_tokens_per_s": summary["decode_tokens_per_s_median"],
            "torch_peak_allocated_bytes": summary["torch_peak_memory_allocated_bytes"],
            "torch_peak_reserved_bytes": summary["torch_peak_memory_reserved_bytes"],
        }
    return normalized


def _layer_number(layer_name: str) -> int:
    match = _LAYER_RE.fullmatch(layer_name)
    if match is None:
        raise RawCampaignError(f"unexpected routing layer name: {layer_name}")
    return int(match.group(1))


def _load_routing(campaign_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = campaign_root / "nemotron/timing-routing-data/routing/moe-routing-raw.jsonl"
    reported = _load_json(campaign_root / "nemotron/timing-routing-data/routing/moe-routing-summary.json")
    events: list[dict[str, Any]] = []
    try:
        for line in source.read_text().splitlines():
            events.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as error:
        raise RawCampaignError(f"cannot parse routing raw file: {error}") from error

    event_counts = Counter(event["case"] for event in events)
    if dict(event_counts) != EXPECTED_ROUTING_EVENTS:
        raise RawCampaignError(f"unexpected routing event coverage: {dict(event_counts)}")

    case_slot_counts: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0] * 128)
    )
    case_shared = Counter()
    phase_stats: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    decode_events: list[dict[str, Any]] = []
    for event in events:
        case = event["case"]
        layer_name = event["layer_name"]
        _layer_number(layer_name)
        token_count = event["token_count"]
        ids = event["topk_expert_ids"]
        weights = event["topk_weights"]
        counts = event["routed_expert_token_counts"]
        if event["top_k"] != 6 or len(ids) != token_count or len(weights) != token_count or len(counts) != 128:
            raise RawCampaignError("routing tensor shape or top-k mismatch")
        derived = Counter(expert for row in ids for expert in row)
        for id_row, weight_row in zip(ids, weights, strict=True):
            if len(id_row) != 6 or len(set(id_row)) != 6 or not all(0 <= expert < 128 for expert in id_row):
                raise RawCampaignError("routing expert IDs are invalid")
            if len(weight_row) != 6 or not math.isclose(sum(weight_row), 1.0, abs_tol=2e-7):
                raise RawCampaignError("routing weights are not normalized")
        if [derived[index] for index in range(128)] != counts:
            raise RawCampaignError("routing expert counts do not match top-k IDs")
        shared = sum(event["shared_expert_token_counts"].values())
        if shared != token_count:
            raise RawCampaignError("shared expert did not receive every token")
        for index, count in enumerate(counts):
            case_slot_counts[case][layer_name][index] += count
        case_shared[case] += shared
        phase = phase_stats[(case, event["phase"])]
        phase["events"] += 1
        phase["tokens"] += token_count
        phase["assignments"] += sum(counts)
        phase["active_expert_sum"] += sum(count > 0 for count in counts)
        decode_events.append(event) if case == "decode_s2048_128" else None

    cases = {}
    for case, layers in case_slot_counts.items():
        if len(layers) != 23 or set(reported.get(case, {})) != set(layers):
            raise RawCampaignError(f"{case}: routing layers do not match supplied summary")
        slots = [count for counts in layers.values() for count in counts]
        for layer, counts in layers.items():
            if counts != reported[case][layer]["routed_expert_token_counts"]:
                raise RawCampaignError(f"{case}/{layer}: raw routing differs from supplied summary")
        cases[case] = {
            "routed_assignments": sum(slots),
            "mean_per_slot": statistics.mean(slots),
            "p50": statistics.median(slots),
            "p95": _percentile([float(value) for value in slots], 0.95),
            "max": max(slots),
            "never_selected_slots": sum(value == 0 for value in slots),
            "shared_tokens": case_shared[case],
        }

    pure_decode: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    prefill_active: dict[str, tuple[int, ...]] = {}
    for event in decode_events:
        layer = event["layer_name"]
        if event["phase"] == "prefill":
            prefill_active[layer] = tuple(
                index for index, count in enumerate(event["routed_expert_token_counts"]) if count
            )
        elif event["phase"] == "decode":
            if event["token_count"] != 1:
                raise RawCampaignError("decode routing event must contain one token")
            pure_decode[layer].append(tuple(event["topk_expert_ids"][0]))
        else:
            raise RawCampaignError(f"unexpected decode campaign phase: {event['phase']}")
    layer_names = sorted(pure_decode, key=_layer_number)
    if len(layer_names) != 23 or set(prefill_active) != set(layer_names):
        raise RawCampaignError("decode routing does not cover 23 MoE layers")
    if {len(steps) for steps in pure_decode.values()} != {127}:
        raise RawCampaignError("decode routing must contain 127 recurrent steps after TTFT")

    overlaps = [
        len(set(left) & set(right))
        for layer in layer_names
        for left, right in zip(pure_decode[layer], pure_decode[layer][1:], strict=False)
    ]
    unique_per_layer = [len({expert for step in pure_decode[layer] for expert in step}) for layer in layer_names]
    decode_layer_hotspots = []
    for layer in layer_names:
        counts = case_slot_counts["decode_s2048_128"][layer]
        expert = max(range(128), key=counts.__getitem__)
        decode_layer_hotspots.append({"layer": _layer_number(layer), "expert": expert, "count": counts[expert]})

    phase_breakdown = {
        f"{case}:{phase}": {
            **stats,
            "active_experts_per_event_mean": stats["active_expert_sum"] / stats["events"],
        }
        for (case, phase), stats in phase_stats.items()
    }
    summary = {
        "event_count": len(events),
        "cases": cases,
        "phase_breakdown": phase_breakdown,
        "decode_layer_hotspots": decode_layer_hotspots,
        "decode_generation": {
            "generated_tokens": 128,
            "recurrent_decode_steps": 127,
            "moe_layers": 23,
            "events": 127 * 23,
            "assignments": 127 * 23 * 6,
            "consecutive_topk_overlap_mean_of_6": statistics.mean(overlaps),
            "consecutive_topk_overlap_p50_of_6": statistics.median(overlaps),
            "consecutive_topk_overlap_p95_of_6": _percentile([float(value) for value in overlaps], 0.95),
            "unique_experts_per_layer_mean": statistics.mean(unique_per_layer),
            "unique_experts_per_layer_min": min(unique_per_layer),
            "unique_experts_per_layer_max": max(unique_per_layer),
        },
        "source_sha256": _sha256(source),
    }
    compact_trace = {
        "schema_version": 1,
        "contract": "nemotron3-decode-routing-v1",
        "source": {
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_NEMOTRON_REVISION,
            "raw_routing_sha256": summary["source_sha256"],
            "case": "decode_s2048_128",
        },
        "shape": {
            "context_tokens": 2048,
            "generated_tokens": 128,
            "recurrent_decode_steps": 127,
            "layers": 23,
            "experts": 128,
            "top_k": 6,
        },
        "layer_names": layer_names,
        "prefill_active_experts_by_layer": [list(prefill_active[layer]) for layer in layer_names],
        "decode_topk_by_step": [
            [list(pure_decode[layer][step]) for layer in layer_names] for step in range(127)
        ],
    }
    return summary, compact_trace


def _load_nsys(campaign_root: Path) -> dict[str, dict[str, float]]:
    document = _load_json(campaign_root / "nemotron/nsys-layer-type-summary.json")
    if not isinstance(document, list) or {case.get("case") for case in document} != {
        "prefill_s128",
        "prefill_s2048",
        "prefill_s8192",
        "decode_s2048_128",
    }:
        raise RawCampaignError("unexpected Nemotron NSYS coverage")
    result = {}
    for case in document:
        layers = case["layer_types"]
        if set(layers) != set(EXPECTED_LAYER_COUNTS):
            raise RawCampaignError("unexpected NSYS layer types")
        if any(layers[name]["layer_invocations"] % EXPECTED_LAYER_COUNTS[name] for name in layers):
            raise RawCampaignError("NSYS layer invocation count is inconsistent")
        result[case["case"]] = {
            name.lower(): values["total_gpu_projected_time_ns"] / 1e6 for name, values in layers.items()
        }
    return result


def _load_ncu(campaign_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    validation = _load_json(campaign_root / "nemotron/ncu-validation.json")
    if validation.get("status") != "PASS" or len(validation.get("collections", [])) != 6:
        raise RawCampaignError("Nemotron NCU validation is incomplete")
    if any(item.get("validation_status") != "PASS" for item in validation["collections"]):
        raise RawCampaignError("Nemotron NCU case failed validation")
    document = _load_json(campaign_root / "nemotron/ncu-summary/nemotron_ncu_summary.json")
    collections = document.get("collections")
    if not isinstance(collections, list) or {
        (item.get("phase"), item.get("layer_type")) for item in collections
    } != EXPECTED_NEMOTRON_NCU:
        raise RawCampaignError("Nemotron NCU does not cover prefill/decode x three layer types")
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in collections:
        if item.get("collection_status") != "OK":
            raise RawCampaignError("Nemotron NCU collection is not OK")
        phase = "prefill_s128" if item["phase"] == "Prefill" else "decode_step_s2048"
        result[phase][item["layer_type"].lower()] = {
            key: item[key]
            for key in (
                "physical_gpu_index",
                "gpu_uuid",
                "kernel_calls",
                "unique_kernel_names",
                "duration_ns",
                "dram_read_bytes",
                "dram_write_bytes",
                "l2_read_sectors",
                "l2_write_sectors",
                "sm_throughput_pct",
                "memory_throughput_pct",
                "achieved_occupancy_pct",
                "theoretical_occupancy_pct",
                "registers_per_thread",
            )
        }
    return dict(result)


def _load_environment(campaign_root: Path) -> dict[str, Any]:
    timing = _load_json(campaign_root / "nemotron/timing-routing-data/timing/environment-manifest.json")
    routing = _load_json(campaign_root / "nemotron/timing-routing-data/routing/environment-manifest.json")
    stable_fields = (
        "model_id",
        "model_revision",
        "cuda_device_name",
        "layer_counts",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "mamba_ssm_cache_dtype",
        "model_config_dtype",
        "hybrid_override_pattern",
    )
    if any(timing.get(field) != routing.get(field) for field in stable_fields):
        raise RawCampaignError("timing and routing environments disagree")
    if timing.get("model_id") != EXPECTED_MODEL or timing.get("model_revision") != EXPECTED_NEMOTRON_REVISION:
        raise RawCampaignError("unexpected Nemotron model or revision")
    if timing.get("layer_counts") != EXPECTED_LAYER_COUNTS:
        raise RawCampaignError("unexpected Nemotron layer counts")
    quantization = timing.get("hf_quant_config", {}).get("quantization", {})
    if quantization.get("quant_algo") != "NVFP4" or quantization.get("group_size") != 16:
        raise RawCampaignError("unexpected checkpoint quantization")
    exclusions = quantization.get("exclude_modules")
    if not isinstance(exclusions, list):
        raise RawCampaignError("checkpoint quantization exclusions are missing")
    mamba_projection_layers = sorted(
        {
            int(match.group(1))
            for name in exclusions
            if (match := re.fullmatch(r"backbone\.layers\.(\d+)\.mixer\.(?:in_proj|out_proj)", name))
        }
    )
    attention_projection_layers = sorted(
        {
            int(match.group(1))
            for name in exclusions
            if (match := re.fullmatch(r"backbone\.layers\.(\d+)\.mixer\.(?:q_proj|k_proj|v_proj|o_proj)", name))
        }
    )
    conv_layers = sorted(
        {
            int(match.group(1))
            for name in exclusions
            if (match := re.fullmatch(r"backbone\.layers\.(\d+)\.mixer\.conv1d", name))
        }
    )
    if len(mamba_projection_layers) != 6 or len(attention_projection_layers) != 6 or len(conv_layers) != 23:
        raise RawCampaignError("unexpected mixed-precision exclusion pattern")
    return {
        "gpu": timing["cuda_device_name"],
        "software": {
            "python": timing["python"],
            "torch": timing["packages"]["torch"],
            "torch_cuda": timing["torch_cuda"],
            "vllm": timing["packages"]["vllm"],
            "transformers": timing["packages"]["transformers"],
            "triton": timing["packages"]["triton"],
            "flashinfer": timing["packages"]["flashinfer-python"],
        },
        "state_dtype": timing["mamba_ssm_cache_dtype"],
        "activation_dtype": timing["model_config_dtype"],
        "checkpoint_quantization": {
            "default_linear_weight": "nvfp4",
            "group_size": 16,
            "kv_cache": quantization.get("kv_cache_quant_algo"),
            "excluded_modules_remain_model_dtype": True,
            "mamba_projection_bf16_layers": mamba_projection_layers,
            "attention_projection_bf16_layers": attention_projection_layers,
            "mamba_conv_bf16_layers": conv_layers,
            "lm_head_bf16": "lm_head" in exclusions,
            "raw_excluded_modules": exclusions,
        },
    }


def build_normalized_profile(campaign_root: Path, *, archive: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(campaign_root / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "COMPLETE":
        raise RawCampaignError("campaign manifest is not COMPLETE")
    if manifest.get("model") != {"id": EXPECTED_MODEL, "revision": EXPECTED_NEMOTRON_REVISION}:
        raise RawCampaignError("campaign manifest model mismatch")
    validation = manifest.get("validation", {})
    if validation.get("status") != "PASS" or validation.get("all_case_sha256_verified") is not True:
        raise RawCampaignError("campaign validation did not pass")

    top_level_checksums = _validate_checksum_file(campaign_root, campaign_root / "SHA256SUMS")
    collection_artifacts = _validate_manifest_artifacts(campaign_root, manifest)
    environment = _load_environment(campaign_root)
    kda = _load_kda(campaign_root)
    latency = _load_latency(campaign_root)
    routing, routing_trace = _load_routing(campaign_root)
    nsys = _load_nsys(campaign_root)
    ncu = _load_ncu(campaign_root)
    gpu_uuids = sorted(
        {
            layer["gpu_uuid"]
            for phase in ncu.values()
            for layer in phase.values()
        }
    )

    archive_evidence = None
    if archive is not None:
        archive_evidence = {"name": archive.name, "size_bytes": archive.stat().st_size, "sha256": _sha256(archive)}
    document = {
        "schema_version": 2,
        "campaign_status": "complete",
        "source_status": "raw_campaign_local_sha256_validated",
        "source": {
            "profile_root_relative": profile_relative_path(manifest["campaign_dir"]),
            "generated_utc": manifest["generated_utc"],
            "manifest_sha256": _sha256(campaign_root / "manifest.json"),
            "top_level_checksums_verified": top_level_checksums,
            "collection_artifacts_verified": collection_artifacts,
            "archive": archive_evidence,
        },
        "gpu": environment["gpu"],
        "gpu_uuids": gpu_uuids,
        "software": environment["software"],
        "kda": kda,
        "nemotron": {
            "model": EXPECTED_MODEL,
            "revision": EXPECTED_NEMOTRON_REVISION,
            "layer_counts": {"mamba": 23, "moe": 23, "attention": 6},
            "moe": {"routed_experts": 128, "experts_per_token": 6, "shared_experts": 1},
            "mamba_state_dtype": "fp32",
            "activation_dtype": environment["activation_dtype"],
            "checkpoint_quantization": environment["checkpoint_quantization"],
            "latency": latency,
            "nsys_projected_layer_time_ms": nsys,
            "ncu": ncu,
            "routing": routing,
            "missing": [],
        },
        "evidence_boundaries": {
            "workload_calibrated": True,
            "gpu_baseline_measured": True,
            "plena_cycle_calibrated": False,
            "rtl_frequency_or_ppa_calibrated": False,
            "notes": [
                "NCU kernel replay is physical B200 evidence, not ordinary inference latency.",
                "NSYS projected ranges may overlap and must not be summed into wall time.",
                "Decode NCU layer classes were collected on three B200 GPUs and retain per-case UUID provenance.",
                "The routing trace is one deterministic prompt campaign, not a population-level routing distribution.",
            ],
        },
    }
    return document, routing_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--profile-out", type=Path)
    parser.add_argument("--routing-trace-out", type=Path)
    args = parser.parse_args(argv)
    profile, routing = build_normalized_profile(args.campaign_root, archive=args.archive)
    rendered = json.dumps(profile, indent=2) + "\n"
    if args.profile_out is not None:
        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
        args.profile_out.write_text(rendered)
    if args.routing_trace_out is not None:
        args.routing_trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.routing_trace_out.write_text(json.dumps(routing, separators=(",", ":")) + "\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
