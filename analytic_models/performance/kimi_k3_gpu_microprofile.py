"""Validate and normalize the pinned Kimi K3 KDA B200 microprofile.

The GPU counters describe physical traffic seen by one implementation.  They
are deliberately kept separate from the workload model's logical tensor
traffic and are never used as PLENA cycle calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .profile_paths import profile_relative_path


EXPECTED_HF_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
EXPECTED_KIMI_COMMIT = "3cb39dfd32e51c3328e2e4b4af21341247d06c43"
EXPECTED_FLASHKDA_COMMIT = "1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b"
EXPECTED_CASES = {
    "decode_b1": (1, 1),
    "decode_b8": (8, 1),
    "prefill_b1_s2048": (1, 2048),
}
EXPECTED_FIELDS = (
    "q",
    "k",
    "v",
    "decay_low_rank",
    "decay_g",
    "beta",
    "output_gate",
    "output",
)
REQUIRED_PROFILE_FILES = {
    "validation/kda_custom_vs_official.json",
    "kda/kda_projection_layout.json",
    "ncu/kda_memory_traffic_summary.json",
}
STATE_ELEMENTS_PER_REQUEST = 96 * 128 * 128
STATE_BYTES_PER_REQUEST = STATE_ELEMENTS_PER_REQUEST * 4
PROJECTION_SIZE = 96 * 128


class KdaMicroprofileFormatError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise KdaMicroprofileFormatError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise KdaMicroprofileFormatError(f"{path} is not a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise KdaMicroprofileFormatError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _validate_manifest(root: Path, manifest: dict[str, Any]) -> set[str]:
    source = manifest.get("source", {})
    if source.get("kimi_k3", {}).get("commit") != EXPECTED_KIMI_COMMIT:
        raise KdaMicroprofileFormatError("unexpected Kimi K3 source commit")
    if source.get("flashkda", {}).get("commit") != EXPECTED_FLASHKDA_COMMIT:
        raise KdaMicroprofileFormatError("unexpected FlashKDA source commit")
    if source.get("huggingface_kimi_k3", {}).get("revision") != EXPECTED_HF_REVISION:
        raise KdaMicroprofileFormatError("unexpected Kimi K3 Hugging Face revision")
    if manifest.get("environments", {}).get("isolation_verified") is not True:
        raise KdaMicroprofileFormatError("KDA and Nemotron environments were not isolated")

    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise KdaMicroprofileFormatError("artifact manifest has no hashes")
    manifested = set()
    for path in hashes:
        try:
            manifested.add(profile_relative_path(path))
        except ValueError as error:
            raise KdaMicroprofileFormatError(f"unexpected artifact path {path!r}") from error
    if missing := REQUIRED_PROFILE_FILES - manifested:
        raise KdaMicroprofileFormatError(f"required profile files are not hashed: {sorted(missing)}")
    for remote_path, expected in hashes.items():
        try:
            relative = profile_relative_path(remote_path)
        except ValueError as error:
            raise KdaMicroprofileFormatError(f"unexpected artifact path {remote_path!r}") from error
        local = root / relative
        if _sha256(local) != expected:
            raise KdaMicroprofileFormatError(f"artifact hash mismatch: {relative}")
    return manifested


def _validate_comparison(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != 1:
        raise KdaMicroprofileFormatError("unexpected KDA comparison schema")
    source = document.get("official_source", {})
    if source.get("huggingface_model_revision") != EXPECTED_HF_REVISION:
        raise KdaMicroprofileFormatError("comparison uses a different HF revision")
    if document.get("weight_mapping_bit_identical") is not True:
        raise KdaMicroprofileFormatError("custom and official weights are not identical")
    comparisons = document.get("comparisons")
    if not isinstance(comparisons, list) or {
        item.get("sequence_length") for item in comparisons if isinstance(item, dict)
    } != {1, 16, 256, 2048}:
        raise KdaMicroprofileFormatError("official comparison has incomplete sequence coverage")
    for case in comparisons:
        for key in ("output", "recurrent_state"):
            result = case.get(key, {})
            if result.get("max_abs") != 0.0 or result.get("finite_custom") is not True:
                raise KdaMicroprofileFormatError(f"official comparison failed for {key}")
        conv = case.get("conv_state", {})
        for key in ("q", "k", "v"):
            result = conv.get(key, {})
            if result.get("max_abs") != 0.0 or result.get("finite_custom") is not True:
                raise KdaMicroprofileFormatError(f"official comparison failed for conv {key}")
    return {
        "sequence_lengths": [1, 16, 256, 2048],
        "wrapper_projection_state_bit_exact": True,
        "scope": document.get("custom_mode", {}).get("note"),
    }


def _validate_layout(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("derivation") != "official source plus runtime hooks; no PLENA Simulator mapping consulted":
        raise KdaMicroprofileFormatError("projection layout was not independently derived")
    fields = document.get("fields")
    if not isinstance(fields, list) or tuple(field.get("field_name") for field in fields) != EXPECTED_FIELDS:
        raise KdaMicroprofileFormatError("unexpected KDA projection field order")
    for field in fields:
        if not str(field.get("storage_relation", "")).startswith("independent_tensor"):
            raise KdaMicroprofileFormatError("official KDA projection is not recorded as independent tensors")
        hook = field.get("runtime_hook")
        if not isinstance(hook, dict) or hook.get("output", {}).get("contiguous") is not True:
            raise KdaMicroprofileFormatError(f"missing runtime hook for {field.get('field_name')}")
    return {
        "official_projection_storage": "independent_tensors_not_packed_qkv",
        "field_order": list(EXPECTED_FIELDS),
        "runtime_hook_case": document.get("runtime_hook_case"),
        "consumer_order": document.get("non_projection_consumer_order"),
    }


def _validate_flashkda(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("passed") is not True:
        raise KdaMicroprofileFormatError("FlashKDA comparison did not pass")
    criteria = document.get("criteria", {})
    cosine_min = criteria.get("cosine_min")
    mean_abs_max = criteria.get("mean_abs_max")
    if not isinstance(cosine_min, (int, float)) or not isinstance(mean_abs_max, (int, float)):
        raise KdaMicroprofileFormatError("FlashKDA comparison criteria are missing")
    results = document.get("results")
    if not isinstance(results, list) or {item.get("sequence_length") for item in results if isinstance(item, dict)} != {
        16,
        256,
        2048,
    }:
        raise KdaMicroprofileFormatError("FlashKDA comparison coverage is incomplete")
    for case in results:
        for key in ("output", "final_state"):
            value = case.get(key, {})
            cosine = value.get("cosine")
            mean_abs = value.get("mean_abs")
            max_abs = value.get("max_abs")
            if (
                not isinstance(cosine, (int, float))
                or not isinstance(mean_abs, (int, float))
                or not isinstance(max_abs, (int, float))
                or cosine < cosine_min
                or mean_abs > mean_abs_max
            ):
                raise KdaMicroprofileFormatError(
                    f"FlashKDA comparison failed for {key} at S={case.get('sequence_length')}"
                )
    return {
        "sequence_lengths": [16, 256, 2048],
        "passed": True,
        "criteria": criteria,
        "worst_output_max_abs": max(case["output"]["max_abs"] for case in results),
        "worst_state_max_abs": max(case["final_state"]["max_abs"] for case in results),
        "minimum_cosine": min(value["cosine"] for case in results for value in (case["output"], case["final_state"])),
    }


def _logical_kda_core_bytes(batch: int, sequence: int, *, beta_bytes: int) -> dict[str, int]:
    state_bytes = batch * STATE_BYTES_PER_REQUEST
    input_bytes = batch * sequence * (4 * PROJECTION_SIZE * 2 + 96 * beta_bytes)
    output_bytes = batch * sequence * PROJECTION_SIZE * 2
    return {
        "recurrent_state_read_bytes": state_bytes,
        "recurrent_state_write_bytes": state_bytes,
        "projected_input_read_bytes": input_bytes,
        "output_write_bytes": output_bytes,
    }


def _validate_traffic(document: dict[str, Any]) -> list[dict[str, Any]]:
    support = document.get("raw_metric_support_on_b200", {})
    if not str(support.get("dram__bytes_read.sum", "")).startswith("supported"):
        raise KdaMicroprofileFormatError("B200 DRAM read counter is unsupported")
    if not str(support.get("dram__bytes_write.sum", "")).startswith("supported"):
        raise KdaMicroprofileFormatError("B200 DRAM write counter is unsupported")
    cases = document.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(EXPECTED_CASES):
        raise KdaMicroprofileFormatError("unexpected B200 traffic cases")

    normalized = []
    for name, (batch, sequence) in EXPECTED_CASES.items():
        case = cases[name]
        totals = case.get("totals", {})
        for direction in ("read", "write"):
            sectors = totals.get(f"l2_{direction}_sectors")
            derived = totals.get(f"l2_{direction}_bytes_derived")
            if not isinstance(sectors, int) or derived != sectors * 32:
                raise KdaMicroprofileFormatError(f"invalid L2 sector conversion for {name}/{direction}")
        physical = {
            key: totals.get(key)
            for key in (
                "dram_read_bytes",
                "dram_write_bytes",
                "l2_read_sectors",
                "l2_write_sectors",
                "l2_read_bytes_derived",
                "l2_write_bytes_derived",
            )
        }
        if not all(isinstance(value, int) and value >= 0 for value in physical.values()):
            raise KdaMicroprofileFormatError(f"invalid physical counter for {name}")
        normalized.append(
            {
                "case": name,
                "batch": batch,
                "sequence_length": sequence,
                "logical_core_traffic": _logical_kda_core_bytes(
                    batch,
                    sequence,
                    beta_bytes=4 if name.startswith("decode") else 2,
                ),
                "physical_b200_traffic": physical,
                "kernels": case.get("kernels"),
                "ncu_replay_details_not_latency": case.get("decode_ncu_details"),
            }
        )
    return normalized


def build_report(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifests/stage2-environment-source-manifest.json"
    comparison_path = root / "validation/kda_custom_vs_official.json"
    flashkda_path = root / "validation/flashkda_vs_fla.json"
    layout_path = root / "kda/kda_projection_layout.json"
    traffic_path = root / "ncu/kda_memory_traffic_summary.json"

    manifest = _load_json(manifest_path)
    manifested = _validate_manifest(root, manifest)
    comparison = _validate_comparison(_load_json(comparison_path))
    flashkda = _validate_flashkda(_load_json(flashkda_path))
    flashkda["stage2_artifact_manifest_hashed"] = "validation/flashkda_vs_fla.json" in manifested
    layout = _validate_layout(_load_json(layout_path))
    traffic = _validate_traffic(_load_json(traffic_path))
    return {
        "schema_version": 1,
        "profile_scope": "real-shape random-BF16 KimiDeltaAttention wrapper on one NVIDIA B200",
        "source": {
            "kimi_git_commit": EXPECTED_KIMI_COMMIT,
            "flashkda_git_commit": EXPECTED_FLASHKDA_COMMIT,
            "huggingface_revision": EXPECTED_HF_REVISION,
        },
        "observed_state": {
            "recurrent_dtype": "fp32",
            "recurrent_bytes_per_layer_per_request": STATE_BYTES_PER_REQUEST,
            "conv_dtype": "bf16",
        },
        "official_equivalence": comparison,
        "backend_equivalence": flashkda,
        "backend_compatibility": {
            "decode": "FLA fused recurrent with FP32 beta",
            "prefill": "FlashKDA with state_v_first=True and BF16 beta",
            "caveat": (
                "the pinned HF wrapper uses transpose_state_layout=True and FP32 "
                "beta, so the profiled FlashKDA prefill is a validated compatibility "
                "adaptation rather than the unmodified wrapper path"
            ),
        },
        "projection_contract_evidence": layout,
        "memory_traffic": traffic,
        "validated_facts": [
            "the extracted wrapper is bit-exact with the official layer for output and persistent state",
            "FlashKDA prefill matches the FLA comparison within the recorded numerical criteria",
            "official q/k/v and auxiliary projections are independent tensors",
            "directional DRAM counters are raw supported B200 metrics",
            "directional L2 bytes are derived from raw 32-byte sector counts",
        ],
        "limits": [
            "physical GPU traffic is not PLENA cycle calibration",
            "logical tensor bytes and physical GPU counters are intentionally separate",
            "decode B1 DRAM write zero means dirty state remained cached, not zero logical state write",
            "the tensor-level official order does not select a PLENA per-cycle bank rotation",
            "FlashKDA prefill uses a documented compatibility adaptation for state layout and beta dtype",
            "the FlashKDA-vs-FLA comparison is supplemental archive evidence, not a Stage 2 manifest-listed formal output",
            "NCU replay duration is not an ordinary latency measurement",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(build_report(args.profile_root), indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
