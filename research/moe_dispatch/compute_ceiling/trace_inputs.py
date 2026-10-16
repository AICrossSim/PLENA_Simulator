"""Qualified routing inputs for the compute-only ceiling experiment.

Primary inputs retain the historical DS-V2-Lite split.  Requesting development
does not read heldout payloads.  Qwen inputs are supplemental microbatch replays:
individually captured one-token routes are grouped without claiming a newly
executed batched model forward pass.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PRIMARY_MANIFEST = REPO / "research/moe_dispatch/round2/results/E0/frozen_inputs.json"
QWEN_ROUTE_ROOT = Path("/scratch/shared/mcl123/plena/outputs/window1_p3/route_traces")
BATCHES = (2, 4, 8, 16, 64, 96, 128)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_workload(workload: dict[str, Any], *, shared: bool) -> None:
    batch = int(workload["batch"])
    top_k = int(workload["top_k"])
    experts = workload["experts"]
    routed = [expert for expert in experts if not expert.get("is_shared", False)]
    shared_experts = [expert for expert in experts if expert.get("is_shared", False)]
    if sum(int(expert["Me"]) for expert in routed) != batch * top_k:
        raise ValueError(f"Routed token conservation failed: {workload['id']}")
    if any(int(expert["Me"]) <= 0 for expert in experts):
        raise ValueError("Input expert lists must contain only active experts")
    if len({expert["id"] for expert in experts}) != len(experts):
        raise ValueError("Duplicate expert descriptor")
    if shared and (len(shared_experts) != 1 or int(shared_experts[0]["Me"]) != batch):
        raise ValueError("DS shared descriptor must have Me equal to batch")
    if not shared and shared_experts:
        raise ValueError("Qwen3 supplemental input has no shared expert")


def load_primary_inputs(
    split: str | None = None, *, manifest_path: str | Path = PRIMARY_MANIFEST,
) -> dict[str, Any]:
    """Return requested split(s) and a manifest, checking frozen byte hashes.

    ``split='development'`` returns ``{'development': [...], 'manifest': ...}``.
    ``None`` loads both historical sets; main search should request development
    and freeze its designs before requesting heldout.
    """
    if split not in (None, "development", "heldout"):
        raise ValueError("split must be development, heldout, or None")
    manifest_path = Path(manifest_path)
    original = json.loads(manifest_path.read_text())
    directory = Path(original["input_directory"])
    output: dict[str, Any] = {}
    loaded_hashes: dict[str, str] = {}
    requested = ("development", "heldout") if split is None else (split,)
    for requested_split in requested:
        windows: list[dict[str, Any]] = []
        for filename in (f"{requested_split}.json", f"mixed_{requested_split}.json"):
            path = directory / filename
            digest = _sha256(path)
            if digest != original["input_sha256"][filename]:
                raise ValueError(f"Frozen input hash mismatch: {path}")
            loaded_hashes[filename] = digest
            windows.extend(json.loads(path.read_text())["workloads"])
        expected_ids = original[f"{requested_split}_window_ids"]
        if [window["id"] for window in windows] != expected_ids:
            raise ValueError(f"Frozen {requested_split} window order or IDs changed")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("Duplicate frozen window IDs")
        for window in windows:
            _validate_workload(window, shared=True)
            for expert in window["experts"]:
                expected_f = 2816 if expert.get("is_shared", False) else 1408
                if int(expert["H"]) != 2048 or int(expert["F"]) != expected_f:
                    raise ValueError("Primary GEMM dimensions changed")
        output[requested_split] = windows
    output["manifest"] = {
        "primary_model": "DeepSeek-V2-Lite",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "input_directory": str(directory),
        "loaded_input_sha256": loaded_hashes,
        "loaded_splits": list(requested),
        "model": original["model"],
        "expert_count": 64,
        "precision": "BF16 inputs/weights; FP32 accumulation",
        "boundary": "post-router MoE FFN; routing capture is input, not timed",
        "weights": "shape/address metadata; no pretrained numerical weights loaded",
        "batch_counts": {
            name: dict(sorted(Counter(str(window["batch"]) for window in output[name]).items()))
            for name in requested
        },
        "validation_warning": (
            "Historical heldout captures have been used in prior experiments; "
            "results are exploratory validation, not a pristine unseen test set."
        ),
    }
    return output


def _qwen_partition(request_key: tuple[str, str]) -> str:
    # All layers of one source request remain in the same partition.
    key = "qwen-compute-ceiling-v1\0" + "\0".join(request_key)
    residue = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 3
    return "development" if residue == 0 else "heldout"


def load_qwen_inputs(
    split: str | None = "heldout", *, route_root: str | Path = QWEN_ROUTE_ROOT,
    max_windows_per_group: int | None = 3,
) -> dict[str, Any]:
    """Prepare supplemental real-route microbatches, separate from primary DSE.

    For each benchmark/layer/batch, use consecutive nonoverlapping groups of
    single-token requests, ordered by captured sample index and ID.  Each source
    request belongs to one partition across every layer.  Default selection is
    bounded to three windows per benchmark/layer/batch; ``None`` uses every full
    group.  A source token may occur at different replay batch sizes, which is
    explicitly recorded rather than treated as statistically independent data.
    """
    if split not in (None, "development", "heldout"):
        raise ValueError("split must be development, heldout, or None")
    if max_windows_per_group is not None and max_windows_per_group <= 0:
        raise ValueError("max_windows_per_group must be positive")
    route_root = Path(route_root)
    requested = ("development", "heldout") if split is None else (split,)
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    sources: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str, int]] = set()
    request_sets: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for path in sorted(route_root.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            model = record["model"]
            workload = record["workload"]
            route = record["routing"]
            if (model.get("name"), int(model.get("hidden_size", -1)),
                int(model.get("intermediate_size", -1)), int(model.get("num_experts", -1)),
                int(model.get("top_k", -1))) != ("Qwen3-30B-A3B", 2048, 768, 128, 8):
                raise ValueError("unsupported model geometry")
            if workload.get("phase") != "decode" or int(workload.get("token_count", -1)) != 1:
                raise ValueError("not a one-token decode capture")
            if route.get("source") != "true_hidden_state_router_logits_topk_softmax_greedy_decode":
                raise ValueError("unqualified routing provenance")
            indices = route["topk_indices"]
            if len(indices) != 1 or len(indices[0]) != 8 or len(set(indices[0])) != 8:
                raise ValueError("invalid top-K list")
            if any(not 0 <= int(expert_id) < 128 for expert_id in indices[0]):
                raise ValueError("expert ID out of range")
            counts = Counter(int(expert_id) for expert_id in indices[0])
            if route["expert_counts"] != [counts.get(index, 0) for index in range(128)]:
                raise ValueError("captured expert histogram disagrees with token routes")
            request_key = (str(workload["benchmark"]), str(workload["sample_id"]))
            layer = int(model["layer_index"])
            key = (*request_key, layer)
            if key in seen:
                raise ValueError("duplicate request/layer capture")
            seen.add(key)
            partition = _qwen_partition(request_key)
            request_sets[partition].add(request_key)
            item = {
                "path": str(path), "sha256": _sha256(path), "request_key": list(request_key),
                "layer": layer, "sample_index": int(workload.get("sample_index", 0)),
                "topk_indices": [int(index) for index in indices[0]],
                "topk_weights": route["topk_weights"][0], "partition": partition,
            }
            sources.append({key: value for key, value in item.items() if key not in ("topk_indices", "topk_weights")})
            if partition in requested:
                groups[(partition, request_key[0], layer)].append(item)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"path": str(path), "reason": str(error)})
    if request_sets["development"] & request_sets["heldout"]:
        raise AssertionError("Qwen request partitions overlap")
    output: dict[str, Any] = {partition: [] for partition in requested}
    for (partition, benchmark, layer), records in sorted(groups.items()):
        records.sort(key=lambda record: (record["sample_index"], record["request_key"]))
        for batch in BATCHES:
            complete_count = len(records) // batch
            count = complete_count if max_windows_per_group is None else min(complete_count, max_windows_per_group)
            for window_index in range(count):
                chunk = records[window_index * batch:(window_index + 1) * batch]
                tokens: dict[int, list[int]] = defaultdict(list)
                scores: dict[int, list[float]] = defaultdict(list)
                for token_index, source in enumerate(chunk):
                    for expert_id, score in zip(source["topk_indices"], source["topk_weights"]):
                        tokens[expert_id].append(token_index)
                        scores[expert_id].append(float(score))
                experts = [
                    {"id": expert_id, "Me": len(token_indices), "H": 2048, "F": 768,
                     "is_shared": False, "token_indices": token_indices,
                     "route_scores": scores[expert_id]}
                    for expert_id, token_indices in sorted(tokens.items())
                ]
                window = {
                    "id": f"qwen_real_microbatch_{partition}_{benchmark}_b{batch}_l{layer}_w{window_index}",
                    "batch": batch, "tokens": batch, "hidden": 2048, "top_k": 8,
                    "routed_expert_count": 128,
                    "experts": experts, "dataset": benchmark, "layer": layer, "split": partition,
                    "scope": "post-router three-GEMM SwiGLU compute geometry replay; no shared expert",
                    "routing_is_input_not_timed": True,
                    "provenance": {
                        "origin": "real_one_token_routes_replayed_as_microbatch",
                        "new_batched_model_forward_executed": False,
                        "request_keys": [record["request_key"] for record in chunk],
                        "source_paths": [record["path"] for record in chunk],
                        "source_sha256": [record["sha256"] for record in chunk],
                        "ordering": "captured sample_index, then request ID; consecutive full groups",
                    },
                }
                _validate_workload(window, shared=False)
                output[partition].append(window)
    output["manifest"] = {
        "supplemental_model": "Qwen3-30B-A3B", "route_root": str(route_root),
        "model": {"H": 2048, "routed_F": 768, "routed_topk": 8, "experts": 128,
                  "shared_experts": 0, "expert_gemms": 3},
        "precision_scope": "BF16 hardware replay; source numerical weights are not loaded",
        "qualification": "exact true-router token IDs/histograms; source one-token decode only",
        "partition_rule": "SHA256('qwen-compute-ceiling-v1', benchmark, request_id) modulo3: 0 development, otherwise heldout",
        "request_disjoint": True,
        "valid_source_records": len(sources),
        "source_request_counts": {key: len(value) for key, value in sorted(request_sets.items())},
        "max_windows_per_benchmark_layer_batch": max_windows_per_group,
        "loaded_splits": list(requested),
        "batch_counts": {partition: dict(sorted(Counter(str(window["batch"]) for window in output[partition]).items())) for partition in requested},
        "sources": sources, "rejected_sources": rejected,
        "warning": "Supplemental frozen-design evaluation only; aggregation is not newly captured batched inference. The same requests recur across replay batch sizes, so windows are correlated.",
    }
    return output


def write_input_manifest(path: str | Path, primary: dict[str, Any], **supplements: dict[str, Any]) -> None:
    """Persist reproducible input provenance without numerical trace fabrication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"primary": primary["manifest"], "supplements": {name: data["manifest"] for name, data in supplements.items()}}
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
