#!/usr/bin/env python3
"""Consolidate only completed, explicitly selected output-pool experiment runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    require(bool(rows), "empty table: " + str(path))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepared = json.loads(args.prepared.read_text())
    manifest_hash = digest(args.prepared)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    points, projections, cores = [], [], []
    seen = set()
    builds, libraries, sources, source_records = set(), set(), [], []
    executions = 0
    for result_path in args.result:
        data = json.loads(result_path.read_text())
        require(data.get("status") == "passed" and data.get("all_gates_passed") is True,
                "incomplete or invalid experiment: " + str(result_path))
        require(data["prepared_sha256"] == manifest_hash, "mixed preparation manifests")
        builds.add(data["binary_sha256"])
        sources.append(data["source_sha256"])
        executions += data["successful_native_runs"]
        source_records.append({"path": str(result_path.resolve()), "sha256": digest(result_path)})
        for case in data["cases"]:
            phase, name = case["phase"], case["name"]
            key = (phase, name)
            require(key not in seen, "duplicate completed case: " + str(key))
            seen.add(key)
            configs = prepared["phases"][phase]["architectures"]
            mapping = {json.loads(Path(c["path"]).read_text())["name"]: c for c in configs}
            require(len(mapping) == len(configs), "duplicate architecture names in preparation")
            require(len(case["observations"]) == len(configs), "missing architecture observations")
            require({o["architecture"] for o in case["observations"]} == set(mapping),
                    "architecture observations do not cover the frozen configurations")
            comparison_path = Path(case["result"])
            comparison = json.loads(comparison_path.read_text())
            require(comparison["all_gates_passed"] is True, "comparison gates did not pass")
            require(comparison["executable_sha256"] == data["binary_sha256"],
                    "comparison executable differs from phase manifest")
            libraries.update(c["native_library_sha256"] for c in comparison["comparisons"])
            source_records.append({"path": str(comparison_path.resolve()), "sha256": digest(comparison_path)})
            for observation in case["observations"]:
                label = mapping[observation["architecture"]]
                architecture = json.loads(Path(label["path"]).read_text())
                core_configs = {c["id"]: c for c in architecture["cores"]}
                require(digest(label["path"]) == label["sha256"], "configuration changed")
                pool = architecture["cores"][0]["refinement"].get("output_pool")
                common = dict(phase=phase, window=name, organization=label["organization"],
                              mode=label["mode"], dispatch=architecture["dispatch_policy"])
                points.append(dict(**common, total_ms=observation["total_ps"] / 1e9,
                                   useful_macs=observation["useful_macs"], issued_macs=observation["issued_macs"],
                                   hbm_read_bytes=observation["hbm_read_bytes"],
                                   active_cores=";".join(observation["active_cores"]),
                                   contexts_per_core=pool["output_contexts"] if pool else "",
                                   comparison=case["result"]))
                for core in observation["cores"]:
                    core_config = core_configs[core["id"]]
                    refinement_config = core_config["refinement"]
                    p, r, mt = core_config["blen"], core_config["mlen"], refinement_config["m_rows"]
                    detail = core["refinement"]
                    pool_stats = detail.get("output_pool") or {}
                    cores.append(dict(**common, core=core["id"], p=p, r=r, mt=mt, jobs=core["jobs"],
                                      useful_macs=core["useful_macs"], compute_ms=core["compute_busy_ps"] / 1e9,
                                      weight_wait_ms=core["weight_ready_wait_ps"] / 1e9,
                                      accumulator_busy_ms=detail["accumulator_port_busy_ps"] / 1e9,
                                      accumulator_wait_ms=detail["accumulator_port_wait_ps"] / 1e9,
                                      weight_port_busy_ms=detail["weight_port_busy_ps"] / 1e9,
                                      weight_port_wait_ms=detail["weight_port_wait_ps"] / 1e9,
                                      dependency_wait_ms=core["accumulator_dependency_stall_ps"] / 1e9,
                                      pipeline_drain_ms=core["pipeline_drain_ps"] / 1e9,
                                      output_finalize_ms=detail["output_finalize_elapsed_ps"] / 1e9,
                                      scheduler_ms=pool_stats.get("scheduler_busy_ps", 0) / 1e9,
                                      scheduler_visits=pool_stats.get("scheduler_visits", 0),
                                      contexts_peak=pool_stats.get("contexts_peak", detail["output_contexts_peak"]),
                                      stages_peak=pool_stats.get("operand_stages_peak", ""),
                                      pending_contexts_peak=pool_stats.get("pending_contexts_peak", ""),
                                      weight_slots_peak=core["weight_slots_peak"],
                                      weight_bytes_peak=core["weight_sram_peak_bytes"],
                                      accumulator_bytes_peak=core["accumulator_peak_bytes"]))
                    for projection in core["projections"]:
                        metrics = projection["metrics"]
                        loads = metrics["tile_loads"]
                        require(loads["count"] > 0, "nonempty projection has no weight loads")
                        m_blocks = (projection["m"] + mt - 1) // mt
                        n_bands = (projection["n"] + p - 1) // p
                        configured_pool = refinement_config.get("output_pool")
                        admitted_bands = (configured_pool["output_contexts"] // m_blocks
                                          if configured_pool else refinement_config["active_n_tiles"])
                        projections.append(dict(**common, core=core["id"], expert=projection["expert"],
                                                projection=projection["projection"], m=projection["m"],
                                                n=projection["n"], k=projection["k"],
                                                p=p, r=r, mt=mt, contexts_per_n_band=m_blocks,
                                                max_live_n_bands=min(n_bands, admitted_bands),
                                                elapsed_us=(projection["end_ps"] - projection["start_ps"]) / 1e6,
                                                tile_loads=loads["count"],
                                                load_mean_ns=loads["total_ps"] / loads["count"] / 1000,
                                                load_min_ns=loads["min_ps"] / 1000,
                                                load_max_ns=loads["max_ps"] / 1000,
                                                weight_bytes=metrics["hbm_read_bytes"],
                                                compute_us=metrics["compute_busy_ps"] / 1e6,
                                                accumulator_busy_us=metrics["accumulator_port_busy_ps"] / 1e6,
                                                accumulator_wait_us=metrics["accumulator_port_wait_ps"] / 1e6,
                                                dependency_wait_us=metrics["accumulator_dependency_stall_ps"] / 1e6,
                                                weight_port_busy_us=metrics["weight_port_busy_ps"] / 1e6,
                                                weight_port_wait_us=metrics["weight_port_wait_ps"] / 1e6,
                                                weight_wait_us=metrics["weight_ready_wait_ps"] / 1e6,
                                                pipeline_drain_us=metrics["pipeline_drain_ps"] / 1e6,
                                                output_finalize_us=metrics["output_finalize_elapsed_ps"] / 1e6,
                                                pending_contexts_peak=metrics["pending_contexts_peak"],
                                                scheduler_us=metrics["scheduler_busy_ps"] / 1e6))
    expected = {(phase, w["name"]) for phase, p in prepared["phases"].items() for w in p["windows"]}
    require(seen == expected, "not all frozen cases are complete")
    require(executions == prepared["expected_native_runs"], "native execution count differs")
    require(len(builds) == 1 and all(s == sources[0] for s in sources), "mixed binaries or source snapshots")
    require(len(libraries) == 1 and None not in libraries, "mixed or missing native libraries")
    write_csv(output / "points.csv", points)
    write_csv(output / "cores.csv", cores)
    write_csv(output / "projections.csv", projections)

    lines = ["# Bounded output scheduling: completed experiment", "",
             f"All **{executions} native executions** in the frozen A/B/WC plan passed comparison gates.", "",
             "These are cold-state normal MoE operator measurements using a fixed synthetic expert bank.",
             "A is service characterization; B uses fixed threshold placement; WC is the separately",
             "predeclared work-conserving confirmation. No full-model, physical-area or energy claim.", ""]
    for phase in ("B", "WC"):
        lines += ["## " + ("Fixed-placement mechanism comparison" if phase == "B" else "Work-conserving confirmation"), "",
                  "Times are milliseconds; all shown configurations have matched aggregate modeled resources.", "",
                  "| Window | Organization | Legacy N2 | Legacy N3 | Pool Q8 | Pool Q16 | Pool Q32 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for window in prepared["phases"][phase]["windows"]:
            for organization in ("single", "homogeneous", "heterogeneous"):
                subset = {p["mode"]: p["total_ms"] for p in points
                          if (p["phase"], p["window"], p["organization"]) == (phase, window["name"], organization)}
                values = [f"{subset[m]:.6f}" if m in subset else "—" for m in
                          ("legacy_n2", "legacy_n3", "pool_q8", "pool_q16", "pool_q32")]
                lines.append("| " + " | ".join([window["name"], organization, *values]) + " |")
        lines.append("")
    lines += ["## Service interpretation", "",
              "A's isolated-expert dual-core cases intentionally use one core; they are not comparisons",
              "of equal active PE throughput. Compare a mechanism against its own organization's control.",
              "The concurrent case exercises both fixed expert owners. Projection load statistics include",
              "waiting inside the load path and overlap other loads; their sum is not operator latency.",
              "Finite-port service and scheduler time can also overlap, so per-core waits must not be summed.", "",
              "[All points](points.csv) · [Per-core counters](cores.csv) · [Projection service data](projections.csv)", "",
              "The port 4+4/6+2 diagnosis used the earlier frozen binary and is archived separately.", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    summary = dict(status="passed", successful_native_runs=executions, prepared_sha256=manifest_hash,
                   binary_sha256=next(iter(builds)), sources=source_records, points=len(points),
                   native_library_sha256=next(iter(libraries)),
                   projection_records=len(projections), report=str(output / "REPORT.md"))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
