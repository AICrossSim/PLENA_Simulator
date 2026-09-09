#!/usr/bin/env python3
"""Independently reconcile completed V2 projections with finite service costs.

Accept explicit phase result.json files or completed comparison.json files.
An in-progress phase is allowed only because its listed cases are individually
complete and gated. No unfinished/native runs are read or launched.
"""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    payload = Path(path).read_bytes()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def ceildiv(n, d):
    return (n + d - 1) // d


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def expected_service(projection, core, clock):
    """Compute valid M/N tails, full resident weight tiles, and first-K zero.

    Each M/N output tile writes Kt times and reads Kt-1 times; the final
    materialization reads the whole contiguous M*N output once. Port transfers
    round up per operation, not once after summing all element counts.
    """
    m, n, k = [positive(projection[key], key) for key in ("m", "n", "k")]
    p, r = [positive(core[key], key) for key in ("blen", "mlen")]
    ref = core["refinement"]
    mt = positive(ref["m_rows"], "m_rows")
    wp = positive(ref["weight_read_elements_per_cycle"], "weight port")
    ap = positive(ref["accumulator_elements_per_cycle"], "accumulator port")
    positive(clock, "clock_period_ps")
    if r % 8:
        raise ValueError("local block8 weight tiles require MLEN divisible by 8")

    def extents(size, tile):
        count, tail = divmod(size, tile)
        return [(tile, count)] + ([(tail, 1)] if tail else [])

    ktiles = ceildiv(k, r)
    loads = ceildiv(n, p) * ktiles
    update_cycles = sum(mc * nc * ceildiv(mr * nr, ap)
                        for mr, mc in extents(m, mt) for nr, nc in extents(n, p))
    return dict(
        tile_load_count=loads,
        weight_port_busy_ps=2 * loads * ceildiv(p * r, wp) * clock,
        accumulator_port_busy_ps=((2 * ktiles - 1) * update_cycles + ceildiv(m * n, ap)) * clock,
    )


def self_test():
    core = dict(blen=4, mlen=8, refinement=dict(m_rows=3, weight_read_elements_per_cycle=6,
                                               accumulator_elements_per_cycle=5))
    # M tiles 3+2, N tiles 4+3: accumulator updates round to 3+2+2+2 cycles.
    # K tiles 8+2: three update-port visits plus final ceil(35/5)=7 cycles.
    actual = expected_service(dict(m=5, n=7, k=10), core, 10)
    if actual != dict(tile_load_count=4, weight_port_busy_ps=480, accumulator_port_busy_ps=340):
        raise AssertionError(actual)
    core = dict(blen=8, mlen=16, refinement=dict(m_rows=4, weight_read_elements_per_cycle=3,
                                                accumulator_elements_per_cycle=1))
    actual = expected_service(dict(m=1, n=1, k=1), core, 7)
    if actual != dict(tile_load_count=1, weight_port_busy_ps=602, accumulator_port_busy_ps=14):
        raise AssertionError(actual)


def audit(paths):
    report = dict(status="running", passed=False, inputs=[], completed_cases=0, architecture_records=0,
                  active_core_records=0, projections=0, numeric_checks=0, mismatches=[], cases=[],
                  scope="Analytic reconciliation of representative completed projections; no native reruns or wait-time summation")
    seen = set()
    for input_path in paths:
        phase, sha = load(input_path)
        report["inputs"].append(dict(path=str(input_path.resolve()), sha256=sha, status=phase.get("status")))
        if "cases" in phase:
            sources = [(item["name"], item["phase"], Path(item["result"]), item.get("observations"))
                       for item in phase["cases"]]
        else:
            sources = [(input_path.parent.name, "comparison", input_path, None)]
        for name, phase_name, comparison_path, observations in sources:
            comparison_path = comparison_path.resolve()
            if comparison_path in seen:
                raise ValueError("comparison specified more than once: " + str(comparison_path))
            seen.add(comparison_path)
            comparison, comparison_sha = load(comparison_path)
            if comparison.get("all_gates_passed") is not True or comparison.get("repeats", 0) < 2:
                raise ValueError("case is not a completed, repeated, gated comparison: " + str(comparison_path))
            if phase.get("binary_sha256", comparison["executable_sha256"]) != comparison["executable_sha256"]:
                raise ValueError("phase/comparison executable hash differs")
            summary = dict(name=name, phase=phase_name, comparison=str(comparison_path), sha256=comparison_sha,
                           architecture_records=0, active_core_records=0, projections=0, mismatches=0)
            for candidate in comparison["comparisons"]:
                architecture, result = candidate["architecture"], candidate["result"]
                arch_name = architecture["name"]
                if architecture["schema_version"] != 2 or architecture.get("matrix_timing", "pipelined") != "pipelined":
                    raise ValueError("audit supports the refined V2 pipelined contract only")
                if observations is not None:
                    matching = [o for o in observations if o["architecture"] == arch_name]
                    if len(matching) != 1 or matching[0]["cores"] != result["cores"]:
                        raise ValueError("phase observations differ from completed comparison: " + arch_name)
                summary["architecture_records"] += 1
                by_id = {core["id"]: core for core in architecture["cores"]}
                for core in result["cores"]:
                    if core["jobs"] == 0:
                        if core["projections"]:
                            raise ValueError("idle core has projection records")
                        continue
                    summary["active_core_records"] += 1
                    if len(core["projections"]) != 3 * core["jobs"]:
                        raise ValueError("completed expert is missing gate/up/down service records")
                    for projection in core["projections"]:
                        summary["projections"] += 1
                        expected = expected_service(projection, by_id[core["id"]], architecture["clock_period_ps"])
                        observed = projection["metrics"]
                        observed = dict(tile_load_count=observed["tile_loads"]["count"],
                                        weight_port_busy_ps=observed["weight_port_busy_ps"],
                                        accumulator_port_busy_ps=observed["accumulator_port_busy_ps"])
                        for metric, target in expected.items():
                            report["numeric_checks"] += 1
                            if type(observed[metric]) is not int or observed[metric] != target:
                                summary["mismatches"] += 1
                                report["mismatches"].append(dict(case=name, phase=phase_name, architecture=arch_name,
                                    core=core["id"], job=projection["job"], expert=projection["expert"],
                                    projection=projection["projection"], shape=[projection[k] for k in ("m", "n", "k")],
                                    metric=metric, expected=target, observed=observed[metric]))
            report["completed_cases"] += 1
            for key in ("architecture_records", "active_core_records", "projections"):
                report[key] += summary[key]
            report["cases"].append(summary)
    report["passed"] = report["projections"] > 0 and not report["mismatches"]
    report["status"] = "passed" if report["passed"] else "failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path, help="Explicit phase result.json or completed comparison.json paths")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test and not args.results:
        print("Two hand-calculated non-square/tail service cases passed")
        return
    if not args.results or args.output is None:
        parser.error("provide explicit result paths and --output")
    report = audit(args.results)
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "completed_cases", "architecture_records",
                                                   "projections", "numeric_checks")}))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
