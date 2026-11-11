"""Convert a routing capture into the simulator's trace files <model>_b<batch>_<split>.npz.

A trace file has one row per (request group, decode step, MoE layer) with at least one live request: routes
[rows, batch, topk] int16 (live requests first, zero padding), live [rows] int16, and metadata, a JSON list of
{model, batch, live, source, group, cohort, layer, step}. Rows follow the shard order of the manifest; within a shard
they are layer-major for captures with an alive mask (capture_routes) and step-major for per-step captures.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .requests import file_sha256

TRACE_MODELS = {"qwen35": "qwen", "deepseek": "deepseek", "qwen3next": "qwen3next"}


def masked_rows(z, layers):
    """(layer, step, ids of live requests) from a capture that keeps finished requests behind an alive mask."""
    alive = z["alive"]
    for layer in layers:
        ids = z[f"ids_l{layer}"]
        for step in range(len(alive)):
            if alive[step].any():
                yield layer, step + 1, ids[step][alive[step]]


def per_step_rows(z, layers):
    """(layer, step, ids of live requests) from a capture that stores live requests per step."""
    for step, live in enumerate(z["live_counts"], start=1):
        for layer in layers:
            ids = z[f"ids_l{layer}_s{step}"]
            assert len(ids) == live, (layer, step)
            yield layer, step, ids


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("capture", type=Path, help="capture directory with manifest.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batches", help="comma-separated batch sizes to convert (default: all)")
    p.add_argument("--split", help="write every group under this split name instead of its own")
    a = p.parse_args()
    manifest = json.loads((a.capture / "manifest.json").read_text())
    model = TRACE_MODELS[manifest["kind"]]
    plan = {g["id"]: g for g in manifest["plan"]}
    topk, experts = manifest["workload"]["topk"], manifest["workload"]["experts"]
    layers = [int(layer) for layer in manifest["layers"]]
    batches = {int(b) for b in a.batches.split(",")} if a.batches else None

    traces = {}
    for shard in manifest["shards"]:
        group = plan[shard["id"]]
        batch = int(group["batch"])
        if batches and batch not in batches:
            continue
        path = a.capture / shard["file"]
        if file_sha256(path) != shard["sha256"]:
            raise ValueError(f"{path} does not match its manifest hash")
        routes, metadata = traces.setdefault((batch, a.split or group["split"]), ([], []))
        with np.load(path) as z:
            rows = per_step_rows(z, layers) if "live_counts" in z.files else masked_rows(z, layers)
            for layer, step, ids in rows:
                live = len(ids)
                assert ids.shape == (live, topk) and ids.min() >= 0 and ids.max() < experts, (path, layer, step)
                assert np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0), (path, layer, step)
                padded = np.zeros((batch, topk), np.int16)
                padded[:live] = ids
                routes.append(padded)
                metadata.append(
                    dict(
                        model=model,
                        batch=batch,
                        live=live,
                        source=group["source"],
                        group=group["id"],
                        cohort=int(group.get("cohort", 0)),
                        layer=layer,
                        step=step,
                    )
                )

    a.out.mkdir(parents=True, exist_ok=True)
    for (batch, split), (routes, metadata) in sorted(traces.items()):
        path = a.out / f"{model}_b{batch}_{split}.npz"
        live = np.array([m["live"] for m in metadata], np.int16)
        np.savez_compressed(path, routes=np.stack(routes), live=live, metadata=json.dumps(metadata))
        print(path.name, len(routes), "rows", len({m["group"] for m in metadata}), "groups", flush=True)


if __name__ == "__main__":
    main()
