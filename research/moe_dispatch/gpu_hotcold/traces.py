"""Portable export of frozen, captured DeepSeek-V2-Lite token routes.

Loading an exported file needs only Python's standard library. Exporting from
the research repository imports its frozen-input loader lazily. Never replace
missing captures with generated expert histograms.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

BATCHES = (2, 4, 8, 16, 64, 96, 128)
MODEL = {"name": "DeepSeek-V2-Lite", "H": 2048, "shared_F": 2816,
         "routed_F": 1408, "routed_topk": 6, "routed_experts": 64}
SCHEMA = "gpu_hotcold_captured_routes_v1"
DEFAULT_OUTPUT = Path("/scratch/shared/mcl123/plena/outputs/gpu_hotcold_20261013/input/heldout_routes.json")
EXPECTED_COUNTS = {2: 27, 4: 27, 8: 27, 16: 27, 64: 9, 96: 9, 128: 9}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def validate_window(window: dict[str, Any]) -> None:
    """Validate exact per-token routes, duplicated arrays, and count conservation."""
    identifier = window["id"]
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Window ID must be a nonempty string")
    batch = _integer(window["batch"], "batch")
    if batch not in BATCHES or window.get("split") != "heldout":
        raise ValueError(f"Expected a frozen heldout batch: {identifier}")
    tokens, ids, weights, counts = (window[k] for k in
                                    ("tokens", "route_ids", "route_weights", "counts_64"))
    if len(tokens) != batch or len(ids) != batch or len(weights) != batch or len(counts) != 64:
        raise ValueError(f"Route array dimensions disagree: {identifier}")
    token_indices = [_integer(t["token_index"], "token_index") for t in tokens]
    if sorted(token_indices) != list(range(batch)):
        raise ValueError(f"Token index conservation failed: {identifier}")
    actual = Counter()
    for token in tokens:
        index = token["token_index"]
        routes = token["routes"]
        if len(routes) != 6 or sorted(_integer(r["slot"], "slot") for r in routes) != list(range(6)):
            raise ValueError(f"Invalid Top-6 slots: {identifier}/{index}")
        ordered = sorted(routes, key=lambda r: r["slot"])
        token_ids = [_integer(r["expert_id"], "expert_id") for r in ordered]
        if len(set(token_ids)) != 6 or any(not 0 <= e < 64 for e in token_ids):
            raise ValueError(f"Invalid distinct Top-6 IDs: {identifier}/{index}")
        scores = [float(r["score"]) for r in ordered]
        if any(not math.isfinite(s) or s < 0 for s in scores):
            raise ValueError(f"Invalid captured score: {identifier}/{index}")
        if any(isinstance(x, bool) or not isinstance(x, int) for x in ids[index]):
            raise ValueError("route_ids must contain integers")
        if ids[index] != token_ids or weights[index] != scores:
            raise ValueError(f"Duplicated route arrays disagree: {identifier}/{index}")
        actual.update(token_ids)
    expected = [actual[e] for e in range(64)]
    if any(_integer(c, "count") < 0 for c in counts) or counts != expected or sum(counts) != 6 * batch:
        raise ValueError(f"Expert histogram / 6B conservation failed: {identifier}")
    if window.get("token_routes_sha256") != canonical_sha256(tokens):
        raise ValueError(f"Captured token payload hash changed: {identifier}")
    provenance = window.get("provenance")
    if provenance != "captured_decode" and not (
        isinstance(provenance, dict) and provenance.get("origin") == "captured_mixed"
        and provenance.get("capture_sha256") and provenance.get("request_sha256")
    ):
        raise ValueError(f"Missing captured-route provenance: {identifier}")
    source = window.get("source", {})
    digest = source.get("sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"Missing original source hash: {identifier}")


def population_metrics(window: dict[str, Any]) -> dict[str, Any]:
    """Reference arithmetic accounting, explicitly not measured GPU latency."""
    batch = window["batch"]
    counts = window["counts_64"]
    cold = [e for e, m in enumerate(counts) if 0 < m <= 2]
    hot = [e for e, m in enumerate(counts) if m > 2]
    shared_flops = 6 * batch * MODEL["H"] * MODEL["shared_F"]
    cold_flops = 6 * sum(counts[e] for e in cold) * MODEL["H"] * MODEL["routed_F"]
    hot_flops = 6 * sum(counts[e] for e in hot) * MODEL["H"] * MODEL["routed_F"]
    return {"cold_ids": cold, "hot_ids": hot, "cold_experts": len(cold),
            "hot_experts": len(hot), "inactive_experts": counts.count(0),
            "cold_token_assignments": sum(counts[e] for e in cold),
            "hot_token_assignments": sum(counts[e] for e in hot),
            "shared_gemm_flops": shared_flops, "hot_gemm_flops": hot_flops,
            "cold_gemm_flops": cold_flops,
            "total_gemm_flops": shared_flops + hot_flops + cold_flops,
            "arithmetic_scope": "3 GEMMs/expert only; not a time or utilization measurement"}


def load_routes(path: str | Path) -> dict[str, Any]:
    """Load the complete portable heldout set, refusing altered or partial inputs."""
    payload = json.loads(Path(path).read_text())
    metadata, windows = payload["metadata"], payload["windows"]
    if metadata.get("schema") != SCHEMA or metadata.get("model") != MODEL:
        raise ValueError("Unexpected trace schema or model shape")
    if metadata.get("split") != "heldout" or len(windows) != 135:
        raise ValueError("Expected all 135 frozen heldout windows")
    if len({w["id"] for w in windows}) != len(windows):
        raise ValueError("Duplicate window IDs")
    for window in windows:
        validate_window(window)
    if dict(Counter(w["batch"] for w in windows)) != EXPECTED_COUNTS:
        raise ValueError("Heldout batch coverage changed")
    if metadata.get("window_ids_sha256") != canonical_sha256([w["id"] for w in windows]):
        raise ValueError("Frozen window order changed")
    if metadata.get("windows_sha256") != canonical_sha256(windows):
        raise ValueError("Exported window payload changed")
    return payload


def export_heldout(path: str | Path = DEFAULT_OUTPUT) -> dict[str, str]:
    from research.moe_dispatch.plena65536_moe.workloads import load_workloads

    inputs = load_workloads("heldout")
    manifest = inputs["manifest"]
    profiles = {p["window_id"]: p for p in manifest["profiles"]["heldout"]}
    windows = []
    for native in inputs["heldout"]:
        profile = profiles[native["id"]]
        tokens = native["tokens"]
        ordered = sorted(tokens, key=lambda t: t["token_index"])
        routes = [sorted(t["routes"], key=lambda r: r["slot"]) for t in ordered]
        window = {"id": native["id"], "batch": native["batch"], "split": "heldout",
                  "tokens": tokens, "route_ids": [[r["expert_id"] for r in rs] for rs in routes],
                  "route_weights": [[float(r["score"]) for r in rs] for rs in routes],
                  "counts_64": profile["routed_counts_64"], "provenance": native["provenance"],
                  "token_routes_sha256": profile["token_routes_sha256"],
                  "native_window_sha256": profile["window_payload_sha256"],
                  "source": {"path": profile["source_input_path"],
                             "sha256": profile["source_input_sha256"],
                             "window_index": profile["source_input_window_index"]}}
        validate_window(window)
        windows.append(window)
    payload = {"metadata": {
        "schema": SCHEMA, "model": MODEL, "split": "heldout", "window_count": 135,
        "batch_counts": dict(Counter(w["batch"] for w in windows)),
        "source_manifest": manifest["source_manifest"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "sources": manifest["sources"],
        "window_ids_sha256": canonical_sha256([w["id"] for w in windows]),
        "windows_sha256": canonical_sha256(windows),
        "route_weights": "Captured scores retained exactly; no renormalization or generated histogram",
        "scope": "Post-router MoE replay. Captured IDs/scores, not a new full-model inference.",
        "weights_and_activations": "Not included. Benchmark separately declares numerical tensor provenance.",
        "heldout_note": manifest["validation_warning"],
        "mixed_note": "Large-B mixed windows preserve their archived captured_mixed provenance; not claimed as new native batched forwards.",
        "hotcold_threshold": {"inactive": "Me=0", "cold": "1<=Me<=2", "hot": "Me>2"},
        "flops_note": "Shared useful GEMM FLOPs fraction is 25%; no corresponding GPU time fraction is assumed."
    }, "windows": windows}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    load_routes(output)
    receipt = {"schema": SCHEMA, "status": "PASS", "file": output.name,
               "sha256": sha256(output), "windows": len(windows),
               "batch_counts": payload["metadata"]["batch_counts"],
               "source_manifest_sha256": manifest["source_manifest_sha256"],
               "source_files": manifest["sources"],
               "checks": ["Frozen source hashes and window IDs", "Top-6 distinct IDs per token",
                          "Captured score/ID/slot equality", "64-count histogram and 6B conservation",
                          "Shared/routed shape conservation", "Portable JSON readback"]}
    receipt_path = output.with_name("INPUT_MANIFEST.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"routes": str(output), "manifest": str(receipt_path), "sha256": receipt["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate:
        loaded = load_routes(args.validate)
        print(json.dumps({"status": "PASS", "windows": len(loaded["windows"]), "sha256": sha256(args.validate)}))
    else:
        print(json.dumps(export_heldout(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
