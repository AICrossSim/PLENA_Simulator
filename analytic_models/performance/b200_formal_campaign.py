"""Validate the normalized complete B200 KDA/Nemotron profiling campaign.

The raw archive is ingested by :mod:`b200_campaign_raw`.  This module validates
the compact checked-in contract consumed by workload models and DSE.  GPU
evidence calibrates workload shape, dtype, routing, bottleneck direction, and
the comparison baseline; it never converts NVIDIA time into PLENA cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PINNED_SUMMARY = Path(__file__).with_name("profiles") / "b200_kda_nemotron_campaign_complete.json"
EXPECTED_KDA_CASES = {"prefill_b1_s2048", "decode_b1", "decode_b8"}
EXPECTED_KDA_STAGES = {
    "qkv_projection",
    "short_conv",
    "decay_beta",
    "kda_state_update_output",
    "gate_rmsnorm",
    "out_projection",
}
KDA_MATRIX_PATH = {"qkv_projection", "gate_rmsnorm", "out_projection"}


class B200CampaignFormatError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise B200CampaignFormatError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise B200CampaignFormatError("campaign summary must be a JSON object")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(manifest: dict[str, Any], suffix: str) -> str:
    matches = [value for path, value in manifest.get("artifact_sha256", {}).items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise B200CampaignFormatError(f"Stage2 manifest has no unique hash for {suffix}")
    return matches[0]


def _close(actual: float, expected: float, *, tolerance: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise B200CampaignFormatError(f"{label}: {actual} != {expected}")


def _validate_kda(document: dict[str, Any]) -> list[dict[str, Any]]:
    kda = document.get("kda", {})
    shape = kda.get("shape", {})
    expected_shape = {
        "hidden_size": 7168,
        "num_heads": 96,
        "head_dim": 128,
        "conv_kernel": 4,
        "recurrent_state_dtype": "fp32",
        "recurrent_state_mib_per_batch_layer": 6.0,
        "conv_state_dtype": "bf16",
        "conv_state_mib_per_batch_layer": 0.28125,
    }
    if shape != expected_shape:
        raise B200CampaignFormatError("unexpected KDA shape or persistent-state precision")

    comparison = kda.get("official_comparison", {})
    if comparison.get("sequence_lengths") != [1, 16, 256, 2048]:
        raise B200CampaignFormatError("KDA official comparison coverage is incomplete")
    for field in ("output_max_abs", "recurrent_state_max_abs", "conv_state_max_abs"):
        if comparison.get(field) != 0.0:
            raise B200CampaignFormatError(f"KDA official comparison failed: {field}")

    cases = kda.get("cases")
    if not isinstance(cases, dict) or set(cases) != EXPECTED_KDA_CASES:
        raise B200CampaignFormatError("unexpected KDA campaign cases")

    normalized = []
    for name, case in cases.items():
        stages = case.get("stages")
        if not isinstance(stages, dict) or set(stages) != EXPECTED_KDA_STAGES:
            raise B200CampaignFormatError(f"{name}: unexpected stage set")
        if sum(stage["calls"] for stage in stages.values()) != case.get("kernel_calls"):
            raise B200CampaignFormatError(f"{name}: kernel calls do not sum to case total")

        stage_time = sum(stage["time_ms"] for stage in stages.values())
        stage_read = sum(stage["dram_read_bytes"] for stage in stages.values())
        stage_write = sum(stage["dram_write_bytes"] for stage in stages.values())
        _close(stage_time, case["kernel_time_ms"], tolerance=1e-5, label=f"{name} time")
        _close(stage_read, case["dram_read_bytes"], tolerance=0, label=f"{name} read")
        _close(stage_write, case["dram_write_bytes"], tolerance=0, label=f"{name} write")

        matrix_time = sum(stages[stage]["time_ms"] for stage in KDA_MATRIX_PATH)
        matrix_read = sum(stages[stage]["dram_read_bytes"] for stage in KDA_MATRIX_PATH)
        core = stages["kda_state_update_output"]
        normalized_stages = {
            stage_name: {
                **stage,
                "dram_read_mib": stage["dram_read_bytes"] / (1024**2),
                "dram_write_mib": stage["dram_write_bytes"] / (1024**2),
            }
            for stage_name, stage in stages.items()
        }
        normalized.append(
            {
                "case": name,
                "batch": case["batch"],
                "sequence_length": case["sequence_length"],
                "kernel_time_ms": case["kernel_time_ms"],
                "dram_read_bytes": case["dram_read_bytes"],
                "dram_write_bytes": case["dram_write_bytes"],
                "dram_read_mib": case["dram_read_bytes"] / (1024**2),
                "dram_write_mib": case["dram_write_bytes"] / (1024**2),
                "matrix_path_time_fraction": matrix_time / case["kernel_time_ms"],
                "matrix_path_dram_read_fraction": matrix_read / case["dram_read_bytes"],
                "state_core_time_fraction": core["time_ms"] / case["kernel_time_ms"],
                "state_core_dram_read_mib": core["dram_read_bytes"] / (1024**2),
                "stages": normalized_stages,
            }
        )
    return sorted(normalized, key=lambda item: item["case"])


def _ncu_phase_summary(ncu: dict[str, dict[str, Any]], phase: str) -> dict[str, Any]:
    cases = ncu.get(phase)
    if not isinstance(cases, dict) or set(cases) != {"mamba", "attention", "moe"}:
        raise B200CampaignFormatError(f"Nemotron NCU phase is incomplete: {phase}")
    total_time = sum(case["duration_ns"] for case in cases.values())
    total_read = sum(case["dram_read_bytes"] for case in cases.values())
    if total_time <= 0 or total_read <= 0:
        raise B200CampaignFormatError(f"Nemotron NCU phase has empty counters: {phase}")
    return {
        "total_duration_ns": total_time,
        "total_dram_read_bytes": total_read,
        "layer_types": {
            name: {
                **case,
                "duration_fraction": case["duration_ns"] / total_time,
                "dram_read_fraction": case["dram_read_bytes"] / total_read,
            }
            for name, case in cases.items()
        },
    }


def _validate_nemotron(document: dict[str, Any]) -> dict[str, Any]:
    nemotron = document.get("nemotron", {})
    if nemotron.get("model") != "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4":
        raise B200CampaignFormatError("unexpected Nemotron checkpoint model")
    if not isinstance(nemotron.get("revision"), str) or len(nemotron["revision"]) != 40:
        raise B200CampaignFormatError("Nemotron checkpoint revision is not pinned")
    layer_counts = nemotron.get("layer_counts", {})
    if layer_counts != {"mamba": 23, "moe": 23, "attention": 6}:
        raise B200CampaignFormatError("unexpected Nemotron layer pattern")
    moe = nemotron.get("moe", {})
    if moe != {"routed_experts": 128, "experts_per_token": 6, "shared_experts": 1}:
        raise B200CampaignFormatError("unexpected Nemotron MoE shape")
    if nemotron.get("mamba_state_dtype") != "fp32":
        raise B200CampaignFormatError("formal Nemotron run did not use FP32 Mamba state")

    routing = nemotron.get("routing", {})
    cases = routing.get("cases")
    if not isinstance(cases, dict) or len(cases) != 4:
        raise B200CampaignFormatError("Nemotron routing summary is incomplete")
    for name, case in cases.items():
        if case["routed_assignments"] != case["shared_tokens"] * moe["experts_per_token"]:
            raise B200CampaignFormatError(f"{name}: routed assignments disagree with top-k")
        slots = layer_counts["moe"] * moe["routed_experts"]
        _close(
            case["routed_assignments"] / slots,
            case["mean_per_slot"],
            tolerance=1e-6,
            label=f"{name} mean slot load",
        )
    expected_events = 3 * layer_counts["moe"] + 128 * layer_counts["moe"]
    if routing.get("event_count") != expected_events:
        raise B200CampaignFormatError("routing event count does not cover the reported cases")
    hotspots = routing.get("decode_layer_hotspots")
    if not isinstance(hotspots, list) or len(hotspots) != layer_counts["moe"]:
        raise B200CampaignFormatError("decode hotspots do not cover all MoE layers")

    ncu = nemotron.get("ncu", {})
    if set(ncu) != {"prefill_s128", "decode_step_s2048"}:
        raise B200CampaignFormatError("Nemotron NCU prefill/decode coverage is incomplete")
    prefill_ncu = _ncu_phase_summary(ncu, "prefill_s128")
    decode_ncu = _ncu_phase_summary(ncu, "decode_step_s2048")
    hottest_assignments = sum(item["count"] for item in hotspots)
    decode_assignments = cases["decode_s2048_128"]["routed_assignments"]
    decode_generation = routing.get("decode_generation")
    if not isinstance(decode_generation, dict) or decode_generation.get("recurrent_decode_steps") != 127:
        raise B200CampaignFormatError("pure decode routing coverage is incomplete")
    return {
        "model": nemotron["model"],
        "revision": nemotron["revision"],
        "latency": nemotron["latency"],
        "nsys_projected_layer_time_ms": nemotron["nsys_projected_layer_time_ms"],
        "ncu": {
            "prefill_s128": prefill_ncu,
            "decode_step_s2048": decode_ncu,
        },
        "routing": {
            "event_count": routing["event_count"],
            "cases": cases,
            "decode_layer_hotspots": hotspots,
            "decode_max_hotspot_count": max(item["count"] for item in hotspots),
            "decode_max_hotspot_to_mean": max(item["count"] for item in hotspots)
            / cases["decode_s2048_128"]["mean_per_slot"],
            "one_hottest_expert_per_layer_assignment_coverage": hottest_assignments / decode_assignments,
            "phase_breakdown": routing["phase_breakdown"],
            "decode_generation": decode_generation,
            "source_sha256": routing["source_sha256"],
        },
        "checkpoint_quantization": nemotron["checkpoint_quantization"],
        "moe_to_mamba_prefill_dram_read_ratio": ncu["prefill_s128"]["moe"]["dram_read_bytes"]
        / ncu["prefill_s128"]["mamba"]["dram_read_bytes"],
        "moe_to_attention_prefill_dram_read_ratio": ncu["prefill_s128"]["moe"]["dram_read_bytes"]
        / ncu["prefill_s128"]["attention"]["dram_read_bytes"],
        "moe_to_mamba_decode_dram_read_ratio": ncu["decode_step_s2048"]["moe"]["dram_read_bytes"]
        / ncu["decode_step_s2048"]["mamba"]["dram_read_bytes"],
        "missing": nemotron.get("missing", []),
    }


def build_report(path: Path = PINNED_SUMMARY) -> dict[str, Any]:
    document = _load(path)
    if document.get("schema_version") != 2:
        raise B200CampaignFormatError("unexpected campaign schema")
    if document.get("campaign_status") != "complete":
        raise B200CampaignFormatError("this parser expects the complete formal campaign")
    if document.get("source_status") != "raw_campaign_local_sha256_validated":
        raise B200CampaignFormatError("campaign source status is ambiguous")

    kda_cases = _validate_kda(document)
    decode_b1 = next(case for case in kda_cases if case["case"] == "decode_b1")
    decode_b8 = next(case for case in kda_cases if case["case"] == "decode_b8")
    return {
        "schema_version": 1,
        "campaign_status": document["campaign_status"],
        "source_status": document["source_status"],
        "gpu": document["gpu"],
        "gpu_uuids": document["gpu_uuids"],
        "kda": {
            "shape": document["kda"]["shape"],
            "projection_storage": document["kda"]["projection_storage"],
            "official_comparison": document["kda"]["official_comparison"],
            "cases": kda_cases,
            "decode_b8_to_b1_kernel_time_ratio": decode_b8["kernel_time_ms"] / decode_b1["kernel_time_ms"],
            "decode_b8_to_b1_state_core_read_ratio": decode_b8["state_core_dram_read_mib"]
            / decode_b1["state_core_dram_read_mib"],
        },
        "nemotron": _validate_nemotron(document),
        "source": document["source"],
        "evidence_boundaries": document["evidence_boundaries"],
        "reported_sha256": document["kda"].get("reported_sha256", {}),
        "precision_scope": {
            "gpu_checkpoint": "NVFP4",
            "analytic_model": (
                "packed E2M1 payload plus one FP8 E4M3 block scale per 16 values; "
                "global scales and alignment padding excluded"
            ),
            "transactional_emulator": (
                "sub-byte packing and DMA are supported, but NVIDIA finite-only E2M1/E4M3 "
                "semantics and the tensor-global FP32 scale are not implemented"
            ),
            "compiler_binary": "current Matrix weight layout remains MXFP8, not NVFP4",
        },
        "modeling_constraints": [
            "KDA full-layer DSE must include Matrix and weight-streaming stages; the recurrent core is not the sole bottleneck.",
            "Nemotron MoE placement and scheduling must consume non-uniform routing distributions.",
            "GPU physical traffic validates bottleneck direction but is not a PLENA cycle fit.",
            "The complete raw archive and all six Nemotron NCU reports are locally hash-validated.",
            "The checkpoint is mixed precision: documented BF16 exclusions must override the default NVFP4 linear format.",
            "NVFP4 is a logical traffic assumption only; current Compiler/transactional execution remains MXFP8.",
        ],
    }


def crosscheck_local_kda_stage2(
    root: Path,
    formal_summary: Path = PINNED_SUMMARY,
) -> dict[str, Any]:
    """Cross-check the locally transferred GPU2 KDA-core subset.

    This archive predates the six-stage GPU3 campaign. It independently pins
    source revisions, the official numerical/layout files, and recurrent-core
    DRAM reads; it does not validate the full-stage totals.
    """
    profiles = root / "plena-profiles" if (root / "plena-profiles").is_dir() else root
    manifest_path = profiles / "manifests" / "stage2-environment-source-manifest.json"
    traffic_path = profiles / "ncu" / "kda_memory_traffic_summary.json"
    layout_path = profiles / "kda" / "kda_projection_layout.json"
    comparison_path = profiles / "validation" / "kda_custom_vs_official.json"
    manifest = _load(manifest_path)
    traffic = _load(traffic_path)
    formal_document = _load(formal_summary)

    expected_source = formal_document["kda"]["source"]
    observed_source = manifest.get("source", {})
    source_checks = {
        "kimi_commit": observed_source.get("kimi_k3", {}).get("commit"),
        "flashkda_commit": observed_source.get("flashkda", {}).get("commit"),
        "huggingface_revision": observed_source.get("huggingface_kimi_k3", {}).get("revision"),
    }
    if source_checks != expected_source:
        raise B200CampaignFormatError("local KDA Stage2 source revisions do not match formal campaign")

    files = {
        "projection_layout": (layout_path, "kda/kda_projection_layout.json"),
        "official_comparison": (
            comparison_path,
            "validation/kda_custom_vs_official.json",
        ),
        "core_traffic_summary": (
            traffic_path,
            "ncu/kda_memory_traffic_summary.json",
        ),
    }
    hashes = {}
    for name, (local_path, manifest_suffix) in files.items():
        actual = _sha256(local_path)
        expected = _manifest_hash(manifest, manifest_suffix)
        if actual != expected:
            raise B200CampaignFormatError(f"local KDA Stage2 hash mismatch: {name}")
        hashes[name] = actual
    for name in ("projection_layout", "official_comparison"):
        if hashes[name] != formal_document["kda"]["reported_sha256"][name]:
            raise B200CampaignFormatError(f"local {name} is not the file used by the formal campaign")

    formal_cases = formal_document["kda"]["cases"]
    local_cases = traffic.get("cases", {})
    case_names = {
        "prefill_b1_s2048": "prefill_b1_s2048",
        "decode_b1": "decode_b1",
        "decode_b8": "decode_b8",
    }
    comparisons = []
    for formal_name, local_name in case_names.items():
        formal_core = formal_cases[formal_name]["stages"]["kda_state_update_output"]
        local = local_cases[local_name]
        local_read_mib = local["totals"]["dram_read_bytes"] / (1024**2)
        if local["kernel_count"] != formal_core["calls"]:
            raise B200CampaignFormatError(f"{formal_name}: local KDA core call count differs")
        formal_read_mib = formal_core["dram_read_bytes"] / (1024**2)
        _close(
            local_read_mib,
            formal_read_mib,
            tolerance=0.01,
            label=f"{formal_name} local core DRAM read MiB",
        )
        comparisons.append(
            {
                "case": formal_name,
                "kernel_calls": local["kernel_count"],
                "local_core_dram_read_mib": local_read_mib,
                "formal_core_dram_read_mib": formal_read_mib,
                "absolute_delta_mib": abs(local_read_mib - formal_read_mib),
            }
        )
    return {
        "status": "local_gpu2_kda_core_subset_matches_formal_gpu3_summary",
        "scope": "source revisions, official comparison/layout hashes, and KDA core DRAM reads only",
        "not_proven": ["six-stage KDA totals", "Nemotron latency/NSYS/NCU/routing"],
        "source": source_checks,
        "hashes": hashes,
        "cases": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="?", default=PINNED_SUMMARY)
    parser.add_argument("--kda-stage2-root", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.summary)
    if args.kda_stage2_root is not None:
        report["local_kda_stage2_crosscheck"] = crosscheck_local_kda_stage2(
            args.kda_stage2_root,
            args.summary,
        )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
