#!/usr/bin/env python3
"""Recover an exact decode route slice referenced by a grouped workload manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _resolve_source(value: str, manifest_path: Path, workspace_root: Path | None) -> Path:
    source = Path(value).expanduser()
    candidates = [source] if source.is_absolute() else [manifest_path.parent / source]
    if workspace_root is not None and not source.is_absolute():
        candidates.append(workspace_root / source)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"source_trace {value!r} was not found; tried {rendered}. "
        "Pass --workspace-root or set PLENA_WORKSPACE_ROOT for workspace-relative manifests."
    )


def _gini(counts: list[int]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    if values.size == 0 or values.sum() == 0:
        return 0.0
    diff = np.abs(values[:, None] - values[None, :]).sum()
    return float(diff / (2.0 * values.size * values.sum()))


def build_trace(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    workspace_root = args.workspace_root
    if workspace_root is None and os.environ.get("PLENA_WORKSPACE_ROOT"):
        workspace_root = Path(os.environ["PLENA_WORKSPACE_ROOT"])
    if workspace_root is not None:
        workspace_root = workspace_root.expanduser().resolve()
    source_path = _resolve_source(manifest["source_trace"], manifest_path, workspace_root)
    source_sha256 = _sha256(source_path)
    expected_source_sha256 = manifest.get("source_trace_sha256")
    if expected_source_sha256 and source_sha256 != expected_source_sha256:
        raise ValueError(
            f"source trace SHA256 mismatch: actual={source_sha256}, expected={expected_source_sha256}"
        )

    selection = manifest["source_trace_meta"]["selection"]
    start = int(selection["cohort_start"])
    stop = int(selection["cohort_stop"])
    step = int(selection["step"])
    layer_index = int(selection["layer_index"])
    with np.load(source_path, allow_pickle=False) as archive:
        indices = archive["decode_idx"][start:stop, step, layer_index, :].astype(np.int64)
        weights = archive["decode_weight"][start:stop, step, layer_index, :].astype(np.float32)
        sample_ids = archive["sample_ids"][start:stop].astype(str).tolist()
        if "valid" in archive.files:
            valid = archive["valid"][start:stop, step]
            if not bool(np.all(valid)):
                raise ValueError(f"selected route slice contains invalid decode records: {valid.tolist()}")
        if "layer_ids" in archive.files:
            recovered_layer = int(archive["layer_ids"][layer_index])
            if recovered_layer != int(manifest["layer"]):
                raise ValueError(
                    f"layer mismatch: NPZ layer_ids[{layer_index}]={recovered_layer}, manifest={manifest['layer']}"
                )

    batch = int(manifest["batch"])
    top_k = int(manifest["top_k"])
    if tuple(indices.shape) != (batch, top_k) or weights.shape != indices.shape:
        raise ValueError(
            f"route slice shape {tuple(indices.shape)} does not match batch/top_k {(batch, top_k)}"
        )
    expected_samples = manifest["source_trace_meta"].get("selected_sample_ids", [])
    if expected_samples and sample_ids != expected_samples:
        raise ValueError(f"sample ids differ: NPZ={sample_ids}, manifest={expected_samples}")

    histogram = Counter(int(value) for value in indices.reshape(-1))
    expected_histogram = {int(key): int(value) for key, value in manifest["group_sizes"].items()}
    if dict(sorted(histogram.items())) != dict(sorted(expected_histogram.items())):
        raise ValueError(
            "exact NPZ route histogram does not match grouped manifest: "
            f"NPZ={dict(sorted(histogram.items()))}, manifest={dict(sorted(expected_histogram.items()))}"
        )

    num_experts = int(manifest["num_experts"])
    if np.any(indices < 0) or np.any(indices >= num_experts):
        raise ValueError(f"expert ids must be in [0, {num_experts}), got range [{indices.min()}, {indices.max()}]")
    if not bool(np.all(np.isfinite(weights))):
        raise ValueError("route weights contain NaN or infinity")
    if any(len(set(row.tolist())) != top_k for row in indices):
        raise ValueError("one or more tokens select the same expert more than once")
    counts = [int(histogram.get(expert, 0)) for expert in range(num_experts)]
    hidden = int(args.hidden or manifest["hidden"])
    intermediate = int(args.intermediate or manifest["routed_intermediate"])
    shared_intermediate = int(
        args.shared_intermediate
        if args.shared_intermediate is not None
        else manifest.get("shared_intermediate", 0)
    )
    if hidden % args.mlen or intermediate % args.mlen or shared_intermediate % args.mlen:
        raise ValueError("hidden, intermediate, and shared intermediate must be divisible by --mlen")
    active = sorted(histogram)
    trace_id = (
        f"{_slug(manifest['model'])}_{_slug(manifest.get('workload', 'workload'))}_"
        f"b{batch}_l{manifest['layer']}_s{manifest['step']}_h{hidden}_i{intermediate}"
    )
    trace = {
        "schema_version": 2,
        "trace_id": trace_id,
        "created_by": "transactional_emulator.testbench.moe_timing.replay.build_trace_from_grouped_manifest",
        "model": {
            "name": manifest["model"],
            "layer_index": int(manifest["layer"]),
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_experts": num_experts,
            "top_k": top_k,
            "activation_policy": "standard_swiglu",
            "policy_name": str(manifest["architecture"]),
            "shared_experts": int(manifest.get("n_shared", 0)),
            "shared_intermediate_size": shared_intermediate,
            "shared_gate": str(manifest.get("shared_gate", "none")),
        },
        "workload": {
            "benchmark": manifest.get("workload", "unknown"),
            "sample_id": ",".join(sample_ids),
            "phase": "decode",
            "batch_size": batch,
            "seq_len": 1,
            "token_count": batch,
        },
        "routing": {
            "source": "exact_npz_decode_slice",
            "topk_indices": indices.tolist(),
            "topk_weights": weights.tolist(),
            "expert_counts": counts,
            "active_experts": active,
            "duplicate_factor": float(indices.size / len(active)),
            "gini": _gini(counts),
        },
        "artifacts": {
            "reference_pt": "generated_by_expert_grouping_replay",
            "l1_golden_pt": "generated_by_expert_grouping_replay",
        },
        "replay": {
            "harness_module": "transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay",
            "stage": "full_vram",
            "mlen": args.mlen,
            "vlen": args.mlen,
            "blen": args.blen,
            "emu_threads": 1,
        },
        "provenance": {
            "grouped_manifest": str(manifest_path),
            "grouped_manifest_sha256": _sha256(manifest_path),
            "source_trace": str(source_path),
            "source_trace_sha256": source_sha256,
            "slice": {
                "cohort_start": start,
                "cohort_stop": stop,
                "step": step,
                "layer_index": layer_index,
            },
            "original_dimensions": {
                "hidden": int(manifest["hidden"]),
                "routed_intermediate": int(manifest["routed_intermediate"]),
                "shared_intermediate": int(manifest.get("shared_intermediate", 0)),
            },
            "dimension_override": any(
                value is not None
                for value in (args.hidden, args.intermediate, args.shared_intermediate)
            ),
            "functional_provenance": manifest.get("functional_provenance"),
        },
        "measurement_note": (
            "Exact real-model route ids and weights for the selected NPZ slice. "
            "Weight values are supplied by the replay harness; dimension overrides are functional tests only."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--shared-intermediate", type=int)
    parser.add_argument("--mlen", type=int, default=64)
    parser.add_argument("--blen", type=int, default=4)
    args = parser.parse_args()
    trace = build_trace(args)
    print(
        json.dumps(
            {
                "trace_id": trace["trace_id"],
                "route_rows": len(trace["routing"]["topk_indices"]),
                "active_experts": len(trace["routing"]["active_experts"]),
                "duplicate_factor": trace["routing"]["duplicate_factor"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
