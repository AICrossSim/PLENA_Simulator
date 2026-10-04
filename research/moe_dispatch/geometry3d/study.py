"""Reproducible full declared 3-axis analytical DSE; development selection only."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from .compute import Core, enumerate_geometries, enumeration_counts, family, geometry_id, TIMING_PROFILES
from .memory import memory_budget, FabricProfile
from .model import Settings, simulate_layer


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path, value):
    path.write_bytes(canonical(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def summary(results):
    useful = sum(r["useful_macs"] for r in results)
    issued = sum(r["issued_macs"] for r in results)
    cycles = sum(r["cycles"] for r in results)
    return {"layers": len(results), "total_ms": cycles / 1e6,
            "p95_layer_ms": percentile([r["latency_ms"] for r in results], .95),
            "spatial_utilization": useful / issued,
            "wall_mac_utilization": useful / (12288 * cycles),
            "hbm_GiB": sum(r["hbm_bytes"] for r in results) / 2**30,
            "X_sram_GiB": sum(r["X_sram_bytes"] for r in results) / 2**30,
            "hbm_reload_ratio": sum(r["hbm_bytes"] for r in results) / sum(r["native_unique_bytes"] for r in results)}


def verify_capture(workloads):
    for w in workloads:
        provenance = w.get("provenance")
        captured = provenance == "captured_decode" or (isinstance(provenance, dict) and provenance.get("origin") == "captured_mixed")
        if not captured:
            raise ValueError("final performance requires captured routes")
        assert sum(e["Me"] for e in w["experts"] if not e["is_shared"]) == w["batch"] * w["top_k"]
        assert sum(e["Me"] for e in w["experts"] if e["is_shared"]) == w["batch"]
        assert all(e["H"] == 2048 and e["F"] in (1408, 2816) for e in w["experts"])
        ids = [t["token_index"] for t in w["tokens"]]
        assert ids == list(range(w["batch"]))
        routed = {}
        for t in w["tokens"]:
            assert len(t["routes"]) == w["top_k"]
            assert len({r["expert_id"] for r in t["routes"]}) == w["top_k"]
            for r in t["routes"]:
                routed.setdefault(r["expert_id"], []).append(t["token_index"])
        seen = {}
        for e in w["experts"]:
            assert e["Me"] == len(e["token_indices"])
            assert sorted(e["token_indices"]) == (ids if e["is_shared"] else routed[e["id"]])
            if not e["is_shared"]:
                seen[e["id"]] = sorted(e["token_indices"])
        assert seen == routed


def cores_from(gid):
    return tuple(Core(*map(int, part.split("x"))) for part in gid.split("+"))


def evaluate(workloads, cores, settings, policy="eft", repeat=True, detailed=False, owners=None):
    first = [simulate_layer(w, cores, settings, policy, detail=detailed,
                            fixed_owners=None if owners is None else owners[i]) for i,w in enumerate(workloads)]
    if repeat:
        second = [simulate_layer(w, cores, settings, policy, detail=detailed,
                                 fixed_owners=None if owners is None else owners[i]) for i,w in enumerate(workloads)]
        assert canonical(first) == canonical(second), "non-deterministic result"
    return first


def search_one(args):
    cores, dev, settings = args
    row = {"geometry": geometry_id(cores), "family": family(cores),
           "different_PK": len({c.pk for c in cores}) > 1, "legal": False,
           "reason": "", "development_total_ms": "", "development_p95_ms": ""}
    budget = memory_budget(cores, 128, 2048, 2816, allocation=settings.allocation,
                           buffer_limit=settings.prefetch_slots)
    if not budget.fits:
        row["reason"] = "; ".join(c.ineligibility_reason for c in budget.cores if not c.eligible) or "global budget"
    else:
        try:
            s = summary(evaluate(dev, cores, settings))
            row.update(legal=True, development_total_ms=s["total_ms"], development_p95_ms=s["p95_layer_ms"])
        except ValueError as exc:
            row["reason"] = str(exc)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("search", "select", "final", "all"), default="all")
    parser.add_argument("--compute-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev_paths = [args.inputs / "development.json", args.inputs / "mixed_development.json"]
    dev = [w for p in dev_paths for w in json.loads(p.read_text())["workloads"]]
    verify_capture(dev)
    settings = Settings(hbm=not args.compute_only, ports=not args.compute_only, control=not args.compute_only)
    geometries = enumerate_geometries()
    contract = {"scope": "prospective 3D analytical fluid DSE, not native Rust/Ramulator/RTL",
        "counts": enumeration_counts(), "domain": {"PM": [1, 16], "PN": [1, 192], "PK": [32,64,128,256,512,1024]},
        "main_multiplier_budget": 12288, "installed_SRAM_bytes": 2158592,
        "primary_settings": asdict(settings), "secondary_search": "top 4 development designs per family, equal/proportional allocation, bounded WS/OS; no claim of exhaustive allocation/dataflow optimum",
        "runtime_search": "after geometry selection: EFT, idle, RR, thresholds1/2/4/8/16, prefetch slots2/6/32 on development only",
        "timing_sensitivity": "four explicit PK latency hypotheses, frozen selected hardware; not fullspace reranking",
        "selection": "nominal fullspace development score, Round B refine finalists, freeze hardware, then development runtime tune, then heldout",
        "compute_only": args.compute_only,
        "clock": "hypothetical 1GHz; estimated cycles/1e6 = ms",
        "timing_profiles": {key: asdict(v) for key,v in TIMING_PROFILES.items()},
        "development_hashes": {p.name: sha(p) for p in dev_paths},
        "heldout_warning": "historically exposed v3/RoundA captures; split respected for this selection but not a pristine blind dataset",
        "native_layout": "BF16 W[N,K], per-row32B stride; no zero-padding HBM transfers",
        "execution_boundary": "routed expert lists/X available, zero combined-Y explicitly charged; cold weights each layer; excludes Router/attention/full generation",
        "numeric": "physical PK tree changes grouping; tolerance/reference evidence, not cross-PK bitexact or pretrained accuracy",
        "resource_model": "finite grouped FP32 outputs/pairedGU/Zchunks, installed private quotas and banks, shared credits/ingress; fluid service overlap approximates native scheduling",
        "not_claimed": ["measured silicon frequency", "equal area", "PPA", "full model E2E", "hardware global optimum", "guaranteed heterogeneity speedup"]}
    contract_path = args.out / "PREREGISTERED.json"
    if contract_path.exists():
        assert contract_path.read_bytes() == canonical(contract)
    else:
        write_json(contract_path, contract)
    if args.stage in ("search", "select", "all"):
        rows = []
        if args.stage == "select":
            with (args.out / "all_geometry_development.csv").open() as f:
                for r in csv.DictReader(f):
                    r["legal"] = r["legal"] == "True"
                    r["different_PK"] = r["different_PK"] == "True"
                    if r["legal"]:
                        r["development_total_ms"] = float(r["development_total_ms"])
                        r["development_p95_ms"] = float(r["development_p95_ms"])
                    rows.append(r)
            geometries = ()
        began = time.time()
        pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
        jobs = ((cores, dev, settings) for cores in geometries)
        iterator = pool.map(search_one, jobs, chunksize=8) if pool else map(search_one, jobs)
        for idx, row in enumerate(iterator):
            if idx % 1000 == 0:
                print(f"search {idx}/{len(geometries)}, {time.time()-began:.1f}s", flush=True)
            rows.append(row)
        if pool:
            pool.shutdown()
        write_csv(args.out / "all_geometry_development.csv", rows)
        finalists = []
        for fam in ("single", "homogeneous", "heterogeneous"):
            candidates = sorted((r for r in rows if r["legal"] and r["family"] == fam),
                                key=lambda r: (r["development_total_ms"], r["geometry"]))[:4]
            finalists += candidates
        refinement = []
        for row in finalists:
            cores = cores_from(row["geometry"])
            for allocation in ("proportional", "equal"):
                for flow in ("bounded_ws", "bounded_os"):
                    cfg = replace(settings, allocation=allocation, flow=flow)
                    try:
                        s = summary(evaluate(dev, cores, cfg))
                    except ValueError:
                        continue
                    refinement.append({"geometry": row["geometry"], "family": row["family"],
                                       "allocation": allocation, "flow": flow, **s})
        write_csv(args.out / "round_b_finalists.csv", refinement)
        frozen = []
        for fam in ("single", "homogeneous", "heterogeneous"):
            selected = min((r for r in refinement if r["family"] == fam),
                           key=lambda r: (r["total_ms"], r["geometry"], r["allocation"], r["flow"]))
            frozen.append({"label": "selected_" + fam, **selected})
        # Comparators are rerun in THIS model; prior measured numbers are not mixed.
        fixed = {"fixed_6": (Core(6,4,512),), "fixed_3+3": (Core(3,4,512),Core(3,4,512)),
                 "fixed_4+2": (Core(2,4,512),Core(4,4,512))}
        for label, cores in fixed.items():
            frozen.append({"label": label, "geometry": geometry_id(cores), "family": family(cores),
                           "allocation": "proportional", "flow": "bounded_ws"})
        # The fullspace best unequal-PK design is included even if family winner is same-PK.
        unequal = min((r for r in rows if r["legal"] and r["different_PK"]), key=lambda r:r["development_total_ms"])
        frozen.append({"label": "selected_unequal_PK", "geometry": unequal["geometry"], "family": "heterogeneous",
                       "allocation": "proportional", "flow": "bounded_ws"})
        runtime = []
        for selected in frozen:
            cores = cores_from(selected["geometry"])
            for policy in ("eft", "idle", "round_robin", "threshold_1", "threshold_2", "threshold_4", "threshold_8", "threshold_16"):
                for slots in (2, 6, 32):
                    cfg = replace(settings, allocation=selected["allocation"], flow=selected["flow"], prefetch_slots=slots)
                    s = summary(evaluate(dev, cores, cfg, policy))
                    runtime.append({"label": selected["label"], "policy": policy, "prefetch_slots": slots, **s})
            best = min((r for r in runtime if r["label"] == selected["label"]),
                       key=lambda r: (r["total_ms"], r["policy"], r["prefetch_slots"]))
            selected.update(policy=best["policy"], prefetch_slots=best["prefetch_slots"],
                            development_selected_ms=best["total_ms"])
        write_csv(args.out / "runtime_development.csv", runtime)
        write_json(args.out / "FROZEN_SELECTION.json", {"contract_sha256": sha(contract_path), "points": frozen,
                                                      "legal_geometries": sum(r["legal"] for r in rows),
                                                      "excluded_geometries": sum(not r["legal"] for r in rows)})
        print("Development selection frozen. No heldout consulted.", flush=True)
    if args.stage in ("final", "all"):
        frozen_path = args.out / "FROZEN_SELECTION.json"
        frozen = json.loads(frozen_path.read_text())["points"]
        held_paths = [args.inputs / "heldout.json", args.inputs / "mixed_heldout.json"]
        held = [w for p in held_paths for w in json.loads(p.read_text())["workloads"]]
        verify_capture(held)
        rows, details, aggregate, sensitivities, ablations = [], {}, [], [], []
        for selected in frozen:
            cores = cores_from(selected["geometry"])
            cfg = replace(settings, allocation=selected["allocation"], flow=selected["flow"], prefetch_slots=selected["prefetch_slots"])
            results = evaluate(held, cores, cfg, selected["policy"], detailed=True)
            details[selected["label"]] = results
            for r in results:
                rows.append({"label": selected["label"], "geometry": selected["geometry"], "policy": selected["policy"],
                             "workload": r["workload"], "batch": r["batch"], "latency_ms": r["latency_ms"],
                             "cycles": r["cycles"], "useful_macs": r["useful_macs"], "issued_macs": r["issued_macs"],
                             "spatial_utilization": r["spatial_utilization"], "hbm_bytes": r["hbm_bytes"],
                             "X_sram_bytes": r["X_sram_bytes"], "control_service_cycles": r["control_service_cycles"]})
            for batch in sorted({r["batch"] for r in results}):
                aggregate.append({"label": selected["label"], "geometry": selected["geometry"], "batch": batch,
                                  **summary([r for r in results if r["batch"] == batch])})
            aggregate.append({"label": selected["label"], "geometry": selected["geometry"], "batch": "all", **summary(results)})
            for name, profile in TIMING_PROFILES.items():
                r = evaluate(held, cores, replace(cfg, timing=profile), selected["policy"])
                sensitivities.append({"label": selected["label"], "geometry": selected["geometry"], "timing": name, **summary(r)})
            # Same hardware, separately remove supply, control, and local ports.
            owners = [tuple(b["core"] for b in r["bindings"]) for r in results]
            for name, altered in (("charged", cfg), ("zero_control",replace(cfg,control=False)),
                    ("ideal_HBM",replace(cfg,hbm=False)), ("ideal_ports",replace(cfg,ports=False)),
                    ("compute_only",replace(cfg,hbm=False,ports=False,control=False))):
                r = evaluate(held, cores, altered, selected["policy"], owners=owners)
                ablations.append({"label": selected["label"], "geometry": selected["geometry"], "condition": name,
                                  "ownership": "charged_fixed", **summary(r)})
        write_csv(args.out / "heldout_layers.csv", rows)
        write_csv(args.out / "heldout_summary.csv", aggregate)
        write_csv(args.out / "timing_sensitivity.csv", sensitivities)
        write_csv(args.out / "resource_oracles.csv", ablations)
        write_json(args.out / "heldout_details.json", details)
        write_json(args.out / "completion.json", {"complete": True, "repeats": 2,
                   "heldout_layers": len(held), "frozen_selection_sha256": sha(frozen_path),
                   "heldout_hashes": {p.name:sha(p) for p in held_paths},
                   "source_hashes": {p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},
                   "scope": contract["scope"]})
        print("Full declared search/frozen heldout/sensitivity/oracles finished.", flush=True)


if __name__ == "__main__":
    main()
