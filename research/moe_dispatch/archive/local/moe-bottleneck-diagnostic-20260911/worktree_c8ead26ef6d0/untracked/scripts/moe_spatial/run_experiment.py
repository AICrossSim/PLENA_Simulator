#!/usr/bin/env python3
"""Bounded spatial-M mechanism study; never edits the accepted MoE engine.

Run with the project Python and the existing build environment. Every simulator
point runs twice. An independent integer reference and invocation audit check
values, ordering, coverage, issue intervals, finite capacity and MAC accounting.
This is ideal-interface GEMM execution, not full-model or memory simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def ceildiv(a, b):
    return (a + b - 1) // b


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def table(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@lru_cache(maxsize=None)
def scalar_reference(expert, m, n, k, seed):
    # Products are exact integer multiples of 1/128. Their k-dependence has
    # period lcm(17,19)=323. This is independent of Rust's tree/pipeline model.
    out32, out16 = [], []
    periods, tail = divmod(k, 323)
    for row in range(m):
        for col in range(n):
            products = [((row * 7 + kk * 3 + seed) % 17 - 8)
                        * ((kk * 5 + col * 11 + seed * 7 + expert * 3) % 19 - 9)
                        for kk in range(323)]
            numerator = periods * sum(products) + sum(products[:tail])
            bits = struct.unpack("<I", struct.pack("<f", numerator / 128))[0]
            out32.append(bits)
            out16.append(((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF)
    return out32, out16


def audit(req, rep):
    assert rep["requests_drained"]
    widths, pn, pk = req["m_lanes"], req["n_lanes"], req["k_lanes"]
    latency, ii = req["result_latency_cycles"], req["issue_interval_cycles"]
    capacity = ceildiv(latency, ii)
    budget = sum(widths) * pn * pk
    assert budget == req["total_multiplier_budget"] == rep["total_multipliers"]
    jobs = {j["expert"]: j for j in req["jobs"]}
    useful = sum(j["m"] * j["n"] * j["k"] for j in jobs.values())
    assert rep["useful_macs"] == useful
    coverage, per_core, last = {}, defaultdict(list), 0
    useful_check = issued_check = checks = 0
    for inv in rep["trace"]:
        c, e = inv["core"], inv["expert"]
        j, rows = jobs[e], inv["rows"]
        ns, ks = inv["n_start"], inv["k_start"]
        assert ns % pn == ks % pk == 0
        assert 0 <= ns < j["n"] and 0 <= ks < j["k"]
        assert 0 < len(rows) <= widths[c] and len(set(rows)) == len(rows)
        assert inv["valid_n"] == min(pn, j["n"] - ns)
        assert inv["valid_k"] == min(pk, j["k"] - ks)
        assert inv["completion_cycle"] == inv["issue_cycle"] + latency
        assert inv["issued_mac_slots"] == widths[c] * pn * pk
        assert inv["useful_macs"] == len(rows) * inv["valid_n"] * inv["valid_k"]
        if req["ownership"] == "pinned_expert":
            assert e in rep["cores"][c]["assigned_experts"]
        for row in rows:
            assert 0 <= row < j["m"]
            key = (e, row, ns, ks)
            assert key not in coverage
            if ks:
                assert coverage[e, row, ns, ks - pk] <= inv["issue_cycle"]
            coverage[key] = inv["completion_cycle"]
            checks += 1
        per_core[c].append(inv)
        last = max(last, inv["completion_cycle"])
        useful_check += inv["useful_macs"]
        issued_check += inv["issued_mac_slots"]
    assert last == rep["total_cycles"]
    for e, j in jobs.items():
        for row in range(j["m"]):
            for ns in range(0, j["n"], pn):
                for ks in range(0, j["k"], pk):
                    assert (e, row, ns, ks) in coverage
    assert checks == rep["dependency_order_checks"]
    assert useful_check == useful
    assert issued_check == rep["issued_mac_slots"]
    assert issued_check - useful == rep["tail_mac_slots"]
    assert len(rep["trace"]) == rep["total_invocations"]
    for c, cr in enumerate(rep["cores"]):
        invs = per_core[c]
        times = [i["issue_cycle"] for i in invs]
        assert times == cr["issue_cycles"]
        assert all(y - x >= ii for x, y in zip(times, times[1:]))
        events = []
        for i in invs:
            events.extend([(i["issue_cycle"], 1), (i["completion_cycle"], -1)])
        live = peak = 0
        for _, change in sorted(events):  # Completion before same-cycle issue.
            live += change
            peak = max(peak, live)
            assert 0 <= live <= capacity
        assert live == 0 and peak == cr["pipeline_peak"]
        assert capacity == cr["pipeline_capacity"]
        assert cr["pipeline_result_register_bytes"] == capacity * widths[c] * pn * 4
        assert cr["invocations"] == len(invs)
        assert cr["useful_macs"] == sum(i["useful_macs"] for i in invs)
        assert cr["logical_weight_elements"] == sum(i["valid_n"] * i["valid_k"] for i in invs)
        assert cr["logical_activation_elements"] == sum(len(i["rows"]) * i["valid_k"] for i in invs)
        assert sum(cr[k] for k in ["no_ready_invocation_cycles", "no_remaining_work_cycles",
                                   "issue_interval_wait_cycles", "pipeline_full_cycles"]) == last
    if req["verify_values"]:
        assert rep["numerical_bit_exact"] is True
        for j, b32, b16 in zip(req["jobs"], rep["output_fp32_bits"], rep["output_bf16_bits"]):
            ref32, ref16 = scalar_reference(j["expert"], j["m"], j["n"], j["k"], j["seed"])
            assert b32 == ref32 and b16 == ref16
    else:
        assert rep["numerical_bit_exact"] is None
        assert rep["output_fp32_bits"] is None and rep["output_bf16_bits"] is None


def request(name, widths, jobs, mode="pinned_expert", latency=25, ii=1, numerical=True):
    return dict(name=name, m_lanes=widths, n_lanes=4, k_lanes=512,
                total_multiplier_budget=12288, result_latency_cycles=latency,
                issue_interval_cycles=ii, ownership=mode, verify_values=numerical,
                record_trace=True, jobs=jobs)


def fixture(ms, n, k):
    return [dict(expert=e, m=m, n=n, k=k, seed=13 + e) for e, m in enumerate(ms)]


class Experiment:
    def __init__(self, binary, output):
        self.binary, self.output = binary, output
        self.rows, self.receipts = [], []
        for d in ["requests", "reports", "inputs"]:
            (output / d).mkdir()

    def run(self, req, family, workload):
        name = req["name"]
        rp = self.output / "requests" / f"{name}.json"
        save(rp, req)
        first = None
        hashes = []
        for repeat in range(2):
            with tempfile.TemporaryDirectory(prefix="plena-spatial-", dir="/tmp") as tmp:
                raw = Path(tmp) / "report.json"
                subprocess.run([str(self.binary), "--request", str(rp), "--output", str(raw)],
                               check=True, timeout=240)
                payload = raw.read_bytes()
                rep = json.loads(payload)
                audit(req, rep)
                hashes.append(hashlib.sha256(payload).hexdigest())
                if first is None:
                    first = rep
                    compressed = gzip.compress(payload, mtime=0)
                    (self.output / "reports" / f"{name}.json.gz").write_bytes(compressed)
                else:
                    assert rep == first and hashes[0] == hashes[1], name
        row = dict(family=family, workload=workload, shape="+".join(map(str, req["m_lanes"])),
                   ownership=req["ownership"], latency=req["result_latency_cycles"],
                   ii=req["issue_interval_cycles"], cycles=rep["total_cycles"],
                   multipliers=rep["total_multipliers"], useful_macs=rep["useful_macs"],
                   issued_mac_slots=rep["issued_mac_slots"], tail_mac_slots=rep["tail_mac_slots"],
                   invocation_utilization=rep["invocation_utilization"],
                   elapsed_utilization=rep["physical_mac_slot_utilization"],
                   invocations_per_core="/".join(str(c["invocations"]) for c in rep["cores"]),
                   result_register_bytes=sum(c["pipeline_result_register_bytes"] for c in rep["cores"]),
                   logical_weight_elements=sum(c["logical_weight_elements"] for c in rep["cores"]),
                   logical_activation_elements=sum(c["logical_activation_elements"] for c in rep["cores"]),
                   numeric=req["verify_values"], repeats=2, full_repeat_identical=True,
                   name=name)
        self.rows.append(row)
        self.receipts.append(dict(name=name, raw_report_sha256=hashes, request_sha256=digest(rp),
                                  independent_invocation_audit=True,
                                  independent_integer_numerics=req["verify_values"]))
        print(f"{len(self.rows):3d} {name}: {rep['total_cycles']} cycles, repeat/audit PASS", flush=True)
        return rep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ex = Experiment(args.binary.resolve(), out)
    shapes = [[6], [3, 3], [4, 2], [5, 1], [2, 4], [1, 5], [2, 2, 2], [1] * 6]
    # Both core-index orders of every asymmetric split: greedy core-index order
    # is a scheduler sensitivity, not an intrinsic property of a split.
    modes = ["pinned_expert", "tile_stealing"]
    toy = [("4_2", [4, 2]), ("3_3", [3, 3]), ("5_1", [5, 1]), ("2_2", [2, 2])]
    expected = {"4_2": [26, 26, 25], "3_3": [26, 25, 26],
                "5_1": [26, 26, 26], "2_2": [26, 25, 25]}
    for workload, ms in toy:
        for widths in shapes:
            for mode in modes:
                for latency, ii in [(1, 1), (25, 1), (25, 25)]:
                    if widths not in shapes[:3] and (latency, ii) != (25, 1):
                        continue
                    s = "_".join(map(str, widths))
                    name = f"toy_{workload}__{s}__{mode}__L{latency}I{ii}"
                    rep = ex.run(request(name, widths, fixture(ms, 4, 512), mode, latency, ii),
                                 "toy", workload)
                    if mode == "pinned_expert" and (latency, ii) == (25, 1) and widths in shapes[:3]:
                        assert rep["total_cycles"] == expected[workload][shapes.index(widths)]
    # More independent N outputs: measure sustained throughput, not just one
    # pipeline latency. Counterexample retained at exactly the same N and K.
    for workload, ms in toy[:2]:
        for widths in shapes:
            for mode in modes:
                s = "_".join(map(str, widths))
                name = f"sustained_{workload}__{s}__{mode}"
                ex.run(request(name, widths, fixture(ms, 512, 2048), mode), "sustained", workload)
    for widths in [shapes[0], shapes[1], shapes[2], shapes[-1]]:
        for mode in modes:
            name = f"tails__{'_'.join(map(str, widths))}__{mode}"
            ex.run(request(name, widths, fixture([5, 3, 1], 9, 1031), mode), "tails", "5_3_1")
    save(out / "mechanism_gate.json", dict(passed=True, scope="numerical compute-only mechanism",
         limitations="physical timing is an explicit analytical sensitivity; no memory or area claim"))
    table(out / "mechanism.csv", ex.rows)

    # B8 is a prefix of B32 in these archives. Do NOT call full B32 held-out.
    # Validation excludes tokens 0..7; disjoint tokens, same source family only.
    windows = {}
    raw_inputs = {}
    for batch in [8, 32]:
        path = args.windows / f"qwen_full_decode_b{batch}" / "workload.json"
        raw = json.loads(path.read_text())
        raw_inputs[batch] = raw
        (out / "inputs" / f"b{batch}_workload.json.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
        windows[f"b{batch}_full"] = (raw, raw["routes"], path)
    assert raw_inputs[8]["routes"] == [r for r in raw_inputs[32]["routes"] if r["token"] < 8]
    windows["b32_tokens8_31"] = (raw_inputs[32], [r for r in raw_inputs[32]["routes"] if r["token"] >= 8],
                                  args.windows / "qwen_full_decode_b32" / "workload.json")
    manifest = []
    for label, (raw, routes, path) in windows.items():
        counts = Counter(r["expert"] for r in routes)
        jobs = [dict(expert=e, m=m, n=raw["expert_hidden_dim"], k=raw["input_dim"], seed=13 + e)
                for e, m in sorted(counts.items())]
        manifest.append(dict(window=label, path=str(path.resolve()), sha256=digest(path),
            token_ids=sorted(set(r["token"] for r in routes)),
            expert_token_counts=dict(sorted(counts.items())), jobs=jobs,
            role={"b8_full": "selection", "b32_full": "overlapping_reference_not_held_out",
                  "b32_tokens8_31": "disjoint_token_validation_same_archive_family"}[label],
            scope="archived routed Me only; gate GEMM dimensions; no shared expert; synthetic numeric fixtures",
            archive_metadata=raw["metadata"]))
        for widths in shapes:
            for mode in modes:
                name = f"trace_{label}__{'_'.join(map(str, widths))}__{mode}"
                ex.run(request(name, widths, jobs, mode, numerical=False), "trace", label)
        # Numerically execute two representative full-dimension trace-shape
        # cases; still synthetic operands, not the old MX bank or full FFN.
        if label != "b32_full":
            widths = [4, 2] if label == "b8_full" else [1] * 6
            name = f"trace_numeric_{label}__{'_'.join(map(str, widths))}__tile_stealing"
            rep = ex.run(request(name, widths, jobs, "tile_stealing"), "trace_numeric", label)
            timing = next(r for r in ex.rows if r["family"] == "trace" and r["workload"] == label
                          and r["shape"] == "+".join(map(str, widths)) and r["ownership"] == "tile_stealing")
            assert timing["cycles"] == rep["total_cycles"]
    save(out / "input_manifest.json", manifest)
    table(out / "all_points.csv", ex.rows)
    table(out / "trace_dse.csv", [r for r in ex.rows if r["family"] == "trace"])
    # Restricted oracle: choose best observed two-way split for each batch,
    # latency and ownership. This is NOT arbitrary runtime repartitioning.
    groups = defaultdict(list)
    for row in ex.rows:
        if row["family"] not in ["toy", "sustained", "trace"]:
            continue
        if row["latency"] != 25 or row["ii"] != 1:
            continue
        groups[row["family"], row["workload"], row["ownership"]].append(row)
    challengers = []
    for (family, workload, mode), rows in groups.items():
        two_way = [r for r in rows if len(r["shape"].split("+")) == 2]
        oracle = min(two_way, key=lambda r: (r["cycles"], r["shape"]))
        uniform = min([r for r in rows if len(set(r["shape"].split("+"))) == 1],
                      key=lambda r: (r["cycles"], r["shape"]))
        hetero = min([r for r in two_way if r["shape"] != "3+3"],
                     key=lambda r: (r["cycles"], r["shape"]))
        homo = next(r for r in rows if r["shape"] == "3+3")
        challengers.append(dict(family=family, workload=workload, ownership=mode,
            homogeneous_pair_cycles=homo["cycles"], best_two_way_hetero=hetero["shape"],
            best_two_way_hetero_cycles=hetero["cycles"],
            best_uniform=uniform["shape"], best_uniform_cycles=uniform["cycles"],
            batch_partition_oracle=oracle["shape"], batch_partition_oracle_cycles=oracle["cycles"],
            oracle_scope="oracle: positive two-way partitions only; zero repartition cost"))
    table(out / "challenger.csv", challengers)
    # Choose ONCE on B8. Evaluate that shape on disjoint tokens; no validation tuning.
    selections = []
    for mode in modes:
        candidates = [r for r in ex.rows if r["family"] == "trace" and r["workload"] == "b8_full"
                      and r["ownership"] == mode]
        for category, eligible in [
            ("uniform", [r for r in candidates if len(set(r["shape"].split("+"))) == 1]),
            ("heterogeneous_pair", [r for r in candidates if len(r["shape"].split("+")) == 2 and r["shape"] != "3+3"])]:
            selected = min(eligible, key=lambda r: (r["cycles"], r["shape"]))
            validation = next(r for r in ex.rows if r["family"] == "trace"
                              and r["workload"] == "b32_tokens8_31" and r["ownership"] == mode
                              and r["shape"] == selected["shape"])
            selections.append(dict(ownership=mode, category=category, selected_shape=selected["shape"],
                                   selection_cycles=selected["cycles"], validation_cycles=validation["cycles"]))
    table(out / "selection_validation.csv", selections)
    save(out / "validation.json", dict(passed=True, points=len(ex.rows), runs=2 * len(ex.rows),
         numerical_points=sum(r["numeric"] for r in ex.rows), shape_only_points=sum(not r["numeric"] for r in ex.rows),
         bit_exact_scope="generated dyadic BF16 GEMMs, independent integer/scalar references",
         full_repeat_identity=True, dependency_coverage_and_finite_pipeline=True,
         new_hbm_experiments=0, old_engine_modified=False, receipts=ex.receipts))
    save(out / "provenance.json", dict(binary=str(ex.binary), binary_sha256=digest(ex.binary),
         runner=str(Path(__file__).resolve()), runner_sha256=digest(__file__),
         limitations=["No SRAM, DMA, HBM, quantization or nonlinear FFN timing",
                      "25-cycle latency and II are assumptions, not synthesized",
                      "Equal multipliers, not equal area or SRAM ports",
                      "Only one archive family; disjoint tokens do not prove generalization"]))
    print(f"COMPLETE: {len(ex.rows)} points x 2 repeats", flush=True)


if __name__ == "__main__":
    main()
