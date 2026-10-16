"""Frozen true-route Shared + MoE inputs and conservation/provenance checks.

Every native descriptor is retained.  The experimental inputs are archived real
routes, not new batched inference or numerical pretrained-weight evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from research.moe_dispatch.compute_ceiling.trace_inputs import (
    PRIMARY_MANIFEST,
    load_primary_inputs,
)

BATCHES = (2, 4, 8, 16, 64, 96, 128)
MODEL = {"name": "DeepSeek-V2-Lite", "H": 2048, "routed_F": 1408,
         "shared_F": 2816, "routed_topk": 6, "routed_experts": 64,
         "shared_experts": 1, "expert_gemms": 3}
DEFAULT_OUTPUT = Path("/scratch/shared/mcl123/plena/outputs/plena65536_moe_20261011")


def canonical_sha256(value: Any) -> str:
    """Hash a JSON value independently of whitespace in the archived payload."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def validate_window(window: dict[str, Any]) -> None:
    """Check true Top-6 per token, mirrored descriptors, and all GEMM shapes."""
    identifier = str(window["id"])
    batch = int(window["batch"])
    if batch not in BATCHES or int(window["hidden"]) != MODEL["H"]:
        raise ValueError(f"Unexpected batch/hidden geometry: {identifier}")
    if int(window["top_k"]) != MODEL["routed_topk"]:
        raise ValueError(f"Expected true Top-6 routes: {identifier}")
    tokens = window["tokens"]
    if len(tokens) != batch or sorted(int(t["token_index"]) for t in tokens) != list(range(batch)):
        raise ValueError(f"Token index conservation failed: {identifier}")
    route_lookup: dict[tuple[int, int], tuple[int, float]] = {}
    counts: Counter[int] = Counter()
    for token in tokens:
        token_index = int(token["token_index"])
        routes = token["routes"]
        ids = [int(route["expert_id"]) for route in routes]
        if len(ids) != 6 or len(set(ids)) != 6 or any(not 0 <= eid < 64 for eid in ids):
            raise ValueError(f"Invalid true Top-6 expert IDs: {identifier}/token{token_index}")
        if sorted(int(route["slot"]) for route in routes) != list(range(6)):
            raise ValueError(f"Invalid Top-6 route slots: {identifier}/token{token_index}")
        for route in routes:
            eid = int(route["expert_id"])
            score = float(route["score"])
            if not math.isfinite(score) or score < 0:
                raise ValueError(f"Invalid captured route score: {identifier}")
            route_lookup[(token_index, eid)] = (int(route["slot"]), score)
            counts[eid] += 1
    experts = window["experts"]
    if len({int(expert["id"]) for expert in experts}) != len(experts):
        raise ValueError(f"Duplicate expert descriptor: {identifier}")
    shared = [expert for expert in experts if expert.get("is_shared", False)]
    routed = [expert for expert in experts if not expert.get("is_shared", False)]
    if len(shared) != 1 or int(shared[0]["id"]) != -1 or int(shared[0]["Me"]) != batch:
        raise ValueError(f"Shared descriptor must process every token: {identifier}")
    if Counter({int(e["id"]): int(e["Me"]) for e in routed}) != counts:
        raise ValueError(f"Routed expert histogram disagrees with token routes: {identifier}")
    if sum(int(e["Me"]) for e in routed) != batch * 6:
        raise ValueError(f"Routed token conservation failed: {identifier}")
    for expert in experts:
        eid, m = int(expert["id"]), int(expert["Me"])
        is_shared = bool(expert.get("is_shared", False))
        h, f = int(expert["H"]), int(expert["F"])
        expected_f = MODEL["shared_F"] if is_shared else MODEL["routed_F"]
        if h != MODEL["H"] or f != expected_f or m <= 0:
            raise ValueError(f"Primary GEMM dimensions changed: {identifier}/expert{eid}")
        indices, slots, scores = (expert["token_indices"], expert["route_slots"], expert["route_scores"])
        if any(len(values) != m for values in (indices, slots, scores)):
            raise ValueError(f"Expert route descriptor length mismatch: {identifier}/expert{eid}")
        if len(set(int(index) for index in indices)) != m:
            raise ValueError(f"Duplicate expert token: {identifier}/expert{eid}")
        if is_shared:
            if sorted(int(index) for index in indices) != list(range(batch)):
                raise ValueError(f"Shared token indices changed: {identifier}")
            if any(int(slot) != -1 for slot in slots) or any(float(score) != 1.0 for score in scores):
                raise ValueError(f"Shared route descriptors changed: {identifier}")
        else:
            for index, slot, score in zip(indices, slots, scores):
                actual = (int(slot), float(score))
                if route_lookup.get((int(index), eid)) != actual:
                    raise ValueError(f"Captured token/descriptor route mismatch: {identifier}/expert{eid}")
        expected_shapes = {"gate": [f, h], "up": [f, h], "down": [h, f]}
        if set(expert["weights"]) != set(expected_shapes):
            raise ValueError(f"Expected exactly Gate/Up/Down weights: {identifier}/expert{eid}")
        for name, shape in expected_shapes.items():
            weight = expert["weights"][name]
            if (list(weight["shape_nk"]) != shape or weight["dtype"] != "BF16"
                    or int(weight["physical_bytes"]) != h * f * 2
                    or int(weight["payload_bytes"]) != h * f * 2):
                raise ValueError(f"Weight dimensions changed: {identifier}/expert{eid}/{name}")
    provenance = window.get("provenance")
    if provenance != "captured_decode" and not (
        isinstance(provenance, dict) and provenance.get("origin") == "captured_mixed"
        and provenance.get("capture_sha256") and provenance.get("request_sha256")
    ):
        raise ValueError(f"Unqualified captured routing provenance: {identifier}")
    if not window.get("routing_is_input_not_timed", False):
        raise ValueError(f"Expected post-router timing boundary: {identifier}")


def profile_window(window: dict[str, Any]) -> dict[str, Any]:
    """Count all three GEMMs for shared and real routed populations separately."""
    routed = [expert for expert in window["experts"] if not expert.get("is_shared", False)]
    shared = [expert for expert in window["experts"] if expert.get("is_shared", False)]
    counts = {int(expert["id"]): int(expert["Me"]) for expert in routed}
    histogram = [counts.get(index, 0) for index in range(64)]
    shared_macs = sum(3 * int(e["Me"]) * int(e["H"]) * int(e["F"]) for e in shared)
    routed_macs = sum(3 * int(e["Me"]) * int(e["H"]) * int(e["F"]) for e in routed)
    shared_bytes = sum(int(weight["physical_bytes"]) for e in shared for weight in e["weights"].values())
    routed_bytes = sum(int(weight["physical_bytes"]) for e in routed for weight in e["weights"].values())
    payload_sha = canonical_sha256(window)
    route_sha = canonical_sha256(window["tokens"])
    provenance = window.get("provenance")
    metadata = provenance if isinstance(provenance, dict) else {}
    dataset_match = re.search(r"(?:^|_)(bfcl|gpqa|swe)(?:_|$)", str(window["id"]))
    layer_match = re.search(r"_l(\d+)(?:_|$)", str(window["id"]))
    step_match = re.search(r"_s(\d+)(?:_|$)", str(metadata.get("decode_case", window["id"])))
    return {
        "window_id": window["id"], "split": window.get("split", metadata.get("split")), "batch": int(window["batch"]),
        "dataset": window.get("dataset", metadata.get("dataset", dataset_match.group(1) if dataset_match else None)),
        "layer": window.get("layer", int(layer_match.group(1)) if layer_match else None),
        "step": window.get("step", int(step_match.group(1)) if step_match else None),
        "routed_topk": 6, "routed_expert_count": 64,
        "active_routed_experts": len(routed), "shared_experts": len(shared),
        "routed_token_assignments": sum(histogram), "shared_token_assignments": int(window["batch"]),
        "routed_me_min_active": min(counts.values()), "routed_me_max": max(counts.values()),
        "routed_experts_me1": sum(m == 1 for m in counts.values()),
        "routed_experts_me_le4": sum(m <= 4 for m in counts.values()),
        "routed_experts_me_ge16": sum(m >= 16 for m in counts.values()),
        "routed_counts_64": histogram, "routed_me_histogram": dict(sorted(Counter(histogram).items())),
        "shared_useful_macs": shared_macs, "routed_useful_macs": routed_macs,
        "total_useful_macs": shared_macs + routed_macs,
        "shared_unique_weight_bytes": shared_bytes, "routed_unique_weight_bytes": routed_bytes,
        "total_unique_weight_bytes": shared_bytes + routed_bytes,
        "unique_weight_identity": "(expert_id, gate/up/down); address aliases do not merge experts",
        "window_payload_sha256": payload_sha, "token_routes_sha256": route_sha,
        "scope": window.get("scope"), "provenance": window.get("provenance"),
        "source_sha256": window.get("source_sha256"),
    }


def load_workloads(split: str | None = None, *, manifest_path: str | Path = PRIMARY_MANIFEST) -> dict[str, Any]:
    """Return native frozen windows and validation manifest for requested splits.

    Development-only selection never reads heldout payloads.  The primary
    loader verifies the actual file bytes against the historical frozen hashes.
    """
    result = load_primary_inputs(split, manifest_path=manifest_path)
    profiles: dict[str, list[dict[str, Any]]] = {}
    source_manifest = json.loads(Path(manifest_path).read_text())
    input_directory = Path(result["manifest"]["input_directory"])
    sources: list[dict[str, Any]] = []
    window_sources: dict[str, dict[str, Any]] = {}
    for filename, digest in result["manifest"]["loaded_input_sha256"].items():
        sources.append({"path": str(input_directory / filename), "sha256": digest})
        for index, window in enumerate(json.loads((input_directory / filename).read_text())["workloads"]):
            window_sources[str(window["id"])] = {
                "source_input_path": str(input_directory / filename),
                "source_input_sha256": digest, "source_input_window_index": index,
            }
    for name in result["manifest"]["loaded_splits"]:
        windows = result[name]
        expected_count = 18 if name == "development" else 135
        if len(windows) != expected_count or set(int(w["batch"]) for w in windows) != set(BATCHES):
            raise ValueError(f"Frozen {name} coverage changed")
        profiles[name] = []
        for window in windows:
            validate_window(window)
            profile = profile_window(window)
            profile.update(window_sources[str(window["id"])])
            # All three routed GEMMs contribute 3x the shared MAC population:
            # Top6 * F1408 == 3 * F2816, for every observed batch.
            if profile["routed_useful_macs"] != 3 * profile["shared_useful_macs"]:
                raise ValueError(f"Shared + routed useful-MAC conservation failed: {window['id']}")
            profiles[name].append(profile)
    manifest = dict(result["manifest"])
    manifest.update({
        "schema": "plena65536_moe_workloads_v1", "model": MODEL,
        "workload_scope": "Shared + all active true Top-6 routed experts; 3 GEMMs each",
        "sources": sources, "historical_origin": source_manifest["origin"],
        "validation": {"frozen_file_bytes": True, "window_order": True,
                       "per_token_top6_distinct_experts": True, "route_slots_0_to_5": True,
                       "token_descriptor_score_id_slot_equality": True,
                       "shared_covers_batch": True, "three_gemm_shape_and_weight_bytes": True,
                       "useful_macs_routed_equals_three_shared": True},
        "unique_weights": "One BF16 gate/up/down matrix per expert identity, independent of repeated loads; historical address aliases do not merge distinct experts.",
        "metric_aggregation": "GM within each batch size; then equal-weight GM across all 7 batch sizes. Also disclose sample-weighted GM.",
        "profiles": profiles,
    })
    result["manifest"] = manifest
    return result


def geometric_mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Geometric means require finite, positive, nonempty values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def batch_balanced_metrics(windows: list[dict[str, Any]], values: Mapping[str, float]) -> dict[str, Any]:
    """Aggregate a positive metric by window ID without favoring decode counts."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for window in windows:
        grouped[int(window["batch"])].append(float(values[str(window["id"])]))
    per_batch = {batch: geometric_mean(group) for batch, group in sorted(grouped.items())}
    return {"per_batch_gm": per_batch, "batch_balanced_gm": geometric_mean(per_batch.values()),
            "sample_weighted_gm": geometric_mean(value for group in grouped.values() for value in group),
            "window_count": len(windows), "batch_count": len(grouped),
            "batch_window_counts": {batch: len(group) for batch, group in sorted(grouped.items())}}


def write_workload_artifacts(inputs: dict[str, Any], output_directory: str | Path = DEFAULT_OUTPUT) -> dict[str, str]:
    """Write manifests and CSV profiles without changing any frozen input file."""
    out = Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "WORKLOAD_MANIFEST.json"
    manifest_path.write_text(json.dumps(inputs["manifest"], indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    rows = [profile for profiles in inputs["manifest"]["profiles"].values() for profile in profiles]
    csv_path = out / "workload_summary.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})
    schema = {
        "schema": "plena65536_moe_workloads_v1",
        "native_window": {"id": "str", "split": "development|heldout", "batch": "2|4|8|16|64|96|128",
                          "hidden": 2048, "top_k": 6,
                          "tokens": "list[{token_index:int,routes:list[6 x {expert_id:int0..63,slot:int0..5,score:float}]}]",
                          "experts": "list[{id:int(-1 shared),is_shared:bool,Me:int,H:2048,F:2816(shared)|1408(routed),token_indices:list[int],route_slots:list[int],route_scores:list[float],weights:{gate/up/down:{shape_nk:list[int,int],physical_bytes:int,payload_bytes:int,dtype:BF16,hbm_base:int}}}]"},
        "useful_macs": "sum_experts(3 * Me * H * F); Shared and routed are separately reported",
        "unique_weight_bytes": "sum_active_expert_id_and_matrix(H*F*2); historical hbm_base aliases are not deduplicated across distinct experts",
        "batch_balanced_gm": "exp(mean_batch(mean_window_in_batch(log(metric))))",
    }
    schema_path = out / "WORKLOAD_SCHEMA.json"
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=True) + "\n")
    return {"manifest": str(manifest_path), "summary": str(csv_path), "schema": str(schema_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "heldout"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = load_workloads(args.split)
    print(json.dumps(write_workload_artifacts(inputs, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
