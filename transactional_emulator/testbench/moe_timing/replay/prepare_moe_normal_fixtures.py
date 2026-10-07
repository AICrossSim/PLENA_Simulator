#!/usr/bin/env python3
"""Prepare small, reproducible V0 fixtures; archived routes retain their weights.

Run with torch and the actual PLENA_Tools codec on PYTHONPATH. Archived cases
use synthetic weights/ready inputs and reduced dimensions, not model execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def matrix(rows, cols, seed):
    # Integer construction avoids RNG/version dependence. Distinct blocks have
    # different scales and both signs; these are explicitly synthetic values.
    return [[(((r * 17 + c * 13 + seed * 7) % 41) - 20)
             * (2.0 ** (-8 + ((c // 8 + r + seed) % 3)))
             for c in range(cols)] for r in range(rows)]


def core(name, blen, mlen, fraction):
    return dict(id=name, blen=blen, mlen=mlen,
                vector_sram_bytes=32768 // fraction,
                accumulator_bytes=16384 // fraction,
                weight_sram_bytes=2048 // fraction)


def prepare(compiler, workspace, destination):
    exporter_path = compiler / "aten/plena/moe_normal_export.py"
    spec = importlib.util.spec_from_file_location("moe_normal_export", exporter_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    destination.mkdir(parents=True, exist_ok=True)
    common = dict(schema_version=1, dispatch_threshold=4, large_core=0,
                  global_dma_credits=8, global_dma_staging_bytes=512,
                  combine_sram_bytes=65536, clock_period_ps=1000,
                  mac_pipeline_cycles=2, vector_elements_per_cycle=16)
    architectures = [
        dict(common, name="single_b4_k24", small_core=0,
             cores=[core("single", 4, 24, 1)]),
        dict(common, name="single_b2_k48", small_core=0,
             cores=[core("single", 2, 48, 1)]),
        dict(common, name="dual_b4_k16_b2_k16", small_core=1,
             cores=[core("large", 4, 16, 2), core("small", 2, 16, 2)]),
    ]
    for architecture in architectures:
        assert sum(c["blen"] * c["mlen"] for c in architecture["cores"]) == 96
        write_json(destination / "architectures" / (architecture["name"] + ".json"), architecture)
    fixtures = []
    for shared in (False, True):
        name = "synthetic_shared" if shared else "synthetic_tails"
        arrays = exporter.demo_arrays()
        if shared:
            arrays["shared_expert"] = {"expert": 3, "weight": 0.25}
        exporter.export_workload(destination / name, name=name,
                                 provenance={"evidence_scope": "synthetic nonzero ready-input operator"},
                                 **arrays)
        fixtures.append(name)
    sources = {
        "qwen_archived_routes": "outputs/swe_grouping_sram_20260903/sram_ring_sweep/qwen_b8_h512_i256_trace.json",
        "deepseek_archived_routes": "outputs/swe_grouping_sram_20260903/panel_pool_sweep/deepseek_h512_slots2/trace.json",
    }
    for name, relative in sources.items():
        source_path = workspace / relative
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
        indices, weights = source["routing"]["topk_indices"], source["routing"]["topk_weights"]
        if len(indices) != len(weights) or any(len(a) != len(b) for a, b in zip(indices, weights)):
            raise ValueError("archived route/weight shape mismatch")
        routes = [dict(token=t, slot=s, expert=expert, weight=weights[t][s])
                  for t, row in enumerate(indices) for s, expert in enumerate(row)]
        expert_ids = sorted({route["expert"] for route in routes})
        d, e = 31, 47
        experts = [dict(id=i, gate=matrix(e, d, i * 3 + 1),
                        up=matrix(e, d, i * 3 + 2), down=matrix(d, e, i * 3 + 3))
                   for i in expert_ids]
        exporter.export_workload(destination / name, name=name,
            inputs=matrix(len(indices), d, 91), experts=experts, routes=routes,
            provenance={
                "evidence_scope": "archived route ids and weights; synthetic ready inputs and weights; reduced dimensions; routed experts only",
                "source_json": str(source_path.resolve()),
                "source_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "inherited_archive_provenance_not_reverified": source.get("provenance"),
                "synthetic_dimensions": {"input_dim": d, "expert_hidden_dim": e},
                "excluded": ["runtime router", "actual model weights and activations", "model-specific shared expert and gate", "full layer/model/agent trajectory"],
            })
        fixtures.append(name)
    campaign = dict(schema_version=1, fixtures=fixtures,
                    architectures=[a["name"] for a in architectures],
                    baseline="single_b4_k24", alternate_single_core="single_b2_k48",
                    note="Predeclared geometries, not DSE optima; all have 96 multipliers and equal total configured SRAM, but port and control area differ.",
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(destination / "campaign.json", campaign)
    print(destination / "campaign.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.compiler.resolve(), args.workspace.resolve(), args.output_dir.resolve())
