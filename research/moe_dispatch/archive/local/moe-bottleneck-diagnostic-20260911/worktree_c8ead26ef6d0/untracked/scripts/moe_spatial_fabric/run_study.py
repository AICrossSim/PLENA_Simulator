#!/usr/bin/env python3
"""Finite-fabric study with repeat hashes, causal audits and separate validation."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

helper_path = Path(__file__).resolve().parents[1] / "moe_spatial/run_experiment.py"
spec = importlib.util.spec_from_file_location("spatial_reference", helper_path)
helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
save, table, reference = helper.save, helper.table, helper.scalar_reference

DEFAULT = dict(weight_bpc=1024, activation_bpc=6144, accumulator_bpc=192,
    control_ports=1, issue_cycles=2, install_cycles=3, completion_cycles=2,
    broadcast=True, retain_weights=True, hop_cycles=1, slots_per_m=2,
    stages_per_core=2, descriptors=256, ready_window=32, accumulator_bytes=2097152,
    selector="affinity", control="cohort", zero_control_time=False,
    zero_weight_time=False, zero_activation_time=False, zero_accumulator_time=False,
    packed_operand_ports=False)


def sha(data): return hashlib.sha256(data).hexdigest()
def aligned(n): return (n + 31) // 32 * 32
def cdiv(a, b): return (a + b - 1) // b


def audit(req, rep):
    r, f, s = req["compute"], req["fabric"], rep["stats"]
    jobs = {j["expert"]: j for j in r["jobs"]}
    widths, pn, pk = r["m_lanes"], r["n_lanes"], r["k_lanes"]
    assert rep["drained"] and rep["total_multipliers"] == sum(widths) * pn * pk
    assert rep["useful_macs"] == sum(j["m"] * j["n"] * j["k"] for j in jobs.values())
    assert sum(c["useful_macs"] for c in rep["cores"]) == rep["useful_macs"]
    assert sum(c["invocations"] for c in rep["cores"]) == rep["invocation_audit_count"]
    assert s["cache_hits"] + s["cache_misses"] == rep["invocation_audit_count"]
    assert s["control_service_cycles"] == (0 if f["zero_control_time"] else
        s["issue_services"] * f["issue_cycles"] + s["install_services"] * f["install_cycles"]
        + s["completion_services"] * f["completion_cycles"])
    assert s["weight_budget_bytes"] == sum(widths) * f["slots_per_m"] * pn * pk * 2
    assert s["activation_budget_bytes"] == sum(widths) * f["stages_per_core"] * pk * 2
    assert s["result_budget_bytes"] == sum(widths) * pn * 4 * cdiv(r["result_latency_cycles"], r["issue_interval_cycles"])
    for kind in ["weight", "activation", "result"]:
        assert s[f"{kind}_storage_peak_bytes"] <= s[f"{kind}_budget_bytes"]
    assert s["descriptor_peak"] <= f["descriptors"]
    assert s["total_metadata_peak"] <= f["descriptors"]
    assert s["output_storage_bytes"] == sum(j["m"] * j["n"] * 4 for j in jobs.values()) <= f["accumulator_bytes"]
    assert s["simultaneous_operand_result_peak_bytes"] <= sum(s[k + "_budget_bytes"] for k in ["weight", "activation", "result"])
    for core in rep["cores"]:
        assert sum(core["states"].values()) == rep["total_cycles"]
        assert core["stage_peak"] <= f["stages_per_core"]
        assert core["pipeline_peak"] <= cdiv(r["result_latency_cycles"], r["issue_interval_cycles"])
    if r["verify_values"]:
        assert rep["numerical_bit_exact"] is True
        assert len(rep["output_fp32_bits"]) == len(r["jobs"]) == len(rep["output_bf16_bits"])
        for j, actual32, actual16 in zip(r["jobs"], rep["output_fp32_bits"], rep["output_bf16_bits"]):
            expected32, expected16 = reference(j["expert"], j["m"], j["n"], j["k"], j["seed"])
            assert actual32 == expected32 and actual16 == expected16
    else:
        assert rep["numerical_bit_exact"] is None and rep["output_fp32_bits"] is None
    if not r["record_trace"]:
        assert rep["trace"] == rep["services"] == []
        assert len(rep["invocation_sha256"]) == len(rep["service_sha256"]) == 64
        return
    ts = rep["trace"]
    assert len(ts) == rep["invocation_audit_count"]
    assert len(rep["services"]) == rep["service_audit_count"]
    for values, field in [(ts, "invocation_sha256"), (rep["services"], "service_sha256")]:
        h = hashlib.sha256()
        for value in values: h.update(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())
        assert h.hexdigest() == rep[field]
    coverage, core_issues, leases, stage_leases = {}, defaultdict(list), [], defaultdict(list)
    last_tag, ref_end = {}, defaultdict(int)
    for t in ts:
        j, c = jobs[t["expert"]], t["core"]
        ns, ks = t["n_start"], t["k_start"]
        assert ns % pn == ks % pk == 0 and ns < j["n"] and ks < j["k"]
        assert 0 < len(t["rows"]) <= widths[c] and len(set(t["rows"])) == len(t["rows"])
        assert t["valid_n"] == min(pn, j["n"] - ns) and t["valid_k"] == min(pk, j["k"] - ks)
        assert t["useful_macs"] == len(t["rows"]) * t["valid_n"] * t["valid_k"]
        assert t["issued_mac_slots"] == widths[c] * pn * pk
        assert t["admitted"] <= t["descriptor_ready"] <= t["activation_ready"] <= t["issue_cycle"]
        assert t["weight_ready"] <= t["issue_cycle"]
        assert t["mac_done"] == t["issue_cycle"] + r["result_latency_cycles"]
        assert t["mac_done"] <= t["rmw_done"] <= t["commit_cycle"] <= rep["total_cycles"]
        for row in t["rows"]:
            assert 0 <= row < j["m"]
            key = (t["expert"], row, ns, ks)
            assert key not in coverage
            if ks: assert coverage[t["expert"], row, ns, ks - pk] <= t["admitted"]
            coverage[key] = t["commit_cycle"]
        lease = (c, t["slot"]); tag = (t["expert"], ns, ks)
        assert t["slot"] < widths[c] * f["slots_per_m"]
        if lease in last_tag and last_tag[lease] != tag:
            assert ref_end[lease] < t["admitted"]  # admission precedes same-cycle MAC issue
        if t["cache_hit"]: assert last_tag[lease] == tag
        last_tag[lease] = tag; ref_end[lease] = max(ref_end[lease], t["issue_cycle"])
        core_issues[c].append(t["issue_cycle"])
        leases.extend([(t["admitted"], 1), (t["commit_cycle"], -1)])
        stage_leases[c].extend([(t["admitted"], 1), (t["issue_cycle"], -1)])
    expected_items = sum(j["m"] * cdiv(j["n"], pn) * cdiv(j["k"], pk) for j in jobs.values())
    assert len(coverage) == expected_items
    assert max(max(t["commit_cycle"] for t in ts),max(sv["end"] for sv in rep["services"])) == rep["total_cycles"]
    for c, issues in core_issues.items():
        issues.sort(); assert all(b - a >= r["issue_interval_cycles"] for a, b in zip(issues, issues[1:]))
        live = peak = 0
        for _, change in sorted(stage_leases[c], key=lambda e: (e[0], -e[1])):
            live += change; peak = max(peak, live); assert live >= 0
        assert live == 0 and peak == rep["cores"][c]["stage_peak"]
    live = peak = 0
    for _, change in sorted(leases): live += change; peak = max(peak, live); assert live >= 0
    assert live == 0 and peak == s["descriptor_peak"]
    ports, byte_ends, cycle_counts, byte_counts, misses = {}, {}, Counter(), Counter(), Counter()
    for ev in rep["services"]:
        port = (ev["resource"], ev["port"])
        assert ev["start"] >= ev["released"]
        if "byte_start" in ev:
            assert ev["resource"] in ["activation","accumulator"] and f["packed_operand_ports"]
            assert ev["byte_start"] >= byte_ends.get(port,0)
            byte_ends[port]=ev["byte_end"]
            assert ev["byte_end"]-ev["byte_start"]==ev["bytes"]
            assert ev["start"]==ev["byte_start"]//ev["byte_rate"]
            assert ev["end"]==cdiv(ev["byte_end"],ev["byte_rate"])
        else:
            assert ev["start"] >= ports.get(port,0)
        assert ev["start"] <= ev["end"] <= rep["total_cycles"]
        cycle_counts[ev["resource"]] += ev["end"] - max(ev["start"],ports.get(port,0))
        ports[port] = ev["end"]
        byte_counts[ev["resource"]] += ev["bytes"]
        if ev["resource"] == "weight":
            targets = [ts[i] for i in ev["requests"]]
            assert len(set((t["expert"], t["n_start"], t["k_start"]) for t in targets)) == 1
            assert ev["bytes"] == aligned(targets[0]["valid_n"] * targets[0]["valid_k"] * 2)
            assert len(set(t["core"] for t in targets)) == len(targets)
            if not f["broadcast"]: assert len(targets) == 1
            for t in targets:
                assert not t["cache_hit"] and t["descriptor_ready"] <= ev["end"] <= t["weight_ready"]
                misses[t["request"]] += 1
        elif ev["resource"] in ["activation", "accumulator"]:
            assert len(ev["requests"]) == 1
            t = ts[ev["requests"][0]]
            if ev["resource"] == "activation":
                assert ev["bytes"] == aligned(len(t["rows"]) * t["valid_k"] * 2)
                assert ev["end"] == t["activation_ready"]
            else:
                assert ev["bytes"] == aligned(len(t["rows"]) * t["valid_n"] * 8)
                assert ev["released"] == t["mac_done"] and ev["end"] == t["rmw_done"]
    assert misses == Counter(t["request"] for t in ts if not t["cache_hit"])
    for res in ["control", "weight", "activation", "accumulator"]:
        assert cycle_counts[res] == s[res + "_service_cycles"]
    assert byte_counts["weight"] == s["source_weight_bytes"]
    assert byte_counts["activation"] == s["activation_bytes"]
    assert byte_counts["accumulator"] == s["accumulator_rmw_bytes"]


class Study:
    def __init__(self, binary, output):
        self.binary, self.output, self.rows, self.receipts = binary, output, [], []
        for d in ["requests", "reports"]: (output / d).mkdir(parents=True)

    def run(self, req, family, window, variant="default"):
        name = req["compute"]["name"]
        path = self.output / "requests" / (name + ".json"); save(path, req)
        hashes = []
        for repeat in range(2):
            with tempfile.TemporaryDirectory(prefix="plena-fabric-", dir="/tmp") as tmp:
                raw = Path(tmp) / "result.json"
                subprocess.run([str(self.binary), "--request", str(path), "--output", str(raw)], check=True, timeout=240)
                data = raw.read_bytes(); rep = json.loads(data); audit(req, rep)
                hashes.append(sha(data))
                if repeat == 0: (self.output / "reports" / (name + ".json.gz")).write_bytes(gzip.compress(data, mtime=0))
                else: assert hashes[0] == hashes[1], name
        r, f, s = req["compute"], req["fabric"], rep["stats"]
        row = dict(name=name, family=family, window=window, variant=variant,
            shape="+".join(map(str, r["m_lanes"])), ownership=r["ownership"], control=f["control"], selector=f["selector"],
            cycles=rep["total_cycles"], useful_macs=rep["useful_macs"], issued_mac_slots=rep["issued_mac_slots"],
            elapsed_utilization=rep["useful_macs"] / (12288 * rep["total_cycles"]), numeric=r["verify_values"],
            full_trace=r["record_trace"], invocation_audit_count=rep["invocation_audit_count"],
            weight_bpc=f["weight_bpc"], control_ports=f["control_ports"], broadcast=f["broadcast"], retain=f["retain_weights"])
        row['packed']=f['packed_operand_ports']
        row.update(s); self.rows.append(row)
        self.receipts.append(dict(name=name, hashes=hashes, request_sha256=sha(path.read_bytes()),
            invocation_sha256=rep["invocation_sha256"], service_sha256=rep["service_sha256"],
            independent_full_event_audit=r["record_trace"], integer_numerical_audit=r["verify_values"]))
        table(self.output / "all_points.csv", self.rows)
        save(self.output / "progress.json", dict(points=len(self.rows), last=name))
        print(f"{len(self.rows):3d} {name}: {rep['total_cycles']} cycles PASS", flush=True)
        return row


def req(name, widths, jobs, mode, cfg=None, numeric=False, trace=False):
    f = deepcopy(DEFAULT); f.update(cfg or {})
    return dict(compute=dict(name=name, m_lanes=widths, n_lanes=4, k_lanes=512,
        total_multiplier_budget=12288, result_latency_cycles=25, issue_interval_cycles=1,
        ownership=mode, verify_values=numeric, record_trace=trace, jobs=jobs), fabric=f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    ex = Study(args.binary.resolve(), out)
    shapes = [[6], [3,3], [4,2], [5,1], [2,4], [1,5], [2,2,2], [1]*6]
    primary = [[6], [3,3], [4,2], [1]*6]
    modes = ["pinned_expert", "tile_stealing"]
    for widths in primary:
        for mode in modes:
            for ctl in ["invocation", "cohort", "tile_cohort"]:
                for selector in ["fifo", "affinity"]:
                    name=f"gate__{'_'.join(map(str,widths))}__{mode}__{ctl}__{selector}"
                    ex.run(req(name,widths,helper.fixture([5,3,1],9,1031),mode,
                        dict(control=ctl,selector=selector),numeric=True,trace=True),"gate","tails")
    manifest = json.loads(args.input_manifest.read_text())
    jobs = {x["window"]:x["jobs"] for x in manifest if x["window"] in ["b8_full","b32_tokens8_31"]}
    save(out / "input_manifest.json", [x for x in manifest if x["window"] in jobs])
    workloads = {"toy_4_2":helper.fixture([4,2],512,2048),"toy_3_3":helper.fixture([3,3],512,2048),**jobs}
    for window, work in workloads.items():
        for widths in shapes:
            for mode in modes:
                for ctl in ["invocation", "cohort", "tile_cohort"]:
                    name=f"main_{window}__{'_'.join(map(str,widths))}__{mode}__{ctl}"
                    full=window.startswith("toy") or (window=="b8_full" and widths in primary and mode=="tile_stealing" and ctl in ["cohort","tile_cohort"])
                    ex.run(req(name,widths,work,mode,dict(control=ctl),numeric=window.startswith("toy"),trace=full),"main",window)
                    if ctl=="tile_cohort":
                        ex.run(req(name+"__packed",widths,work,mode,dict(control=ctl,packed_operand_ports=True),numeric=window.startswith("toy"),trace=full),"main_packed",window)
    # Selection only on B8; keep both shape and policy fixed on validation.
    selected=[]
    for widths in primary:
        shape="+".join(map(str,widths))
        win=min((r for r in ex.rows if r["family"]=="main" and r["window"]=="b8_full" and r["shape"]==shape),key=lambda r:(r["cycles"],r["name"]))
        selected.append(win)
    save(out / "selected_per_shape.json",selected)
    for me in [1,32]:
        for widths in primary:
            for mode in modes:
                name=f"isolated_me{me}__{'_'.join(map(str,widths))}__{mode}"
                ex.run(req(name,widths,helper.fixture([me],512,2048),mode,dict(control="tile_cohort"),numeric=True),"isolated",f"me{me}")
    variants={"fifo":dict(selector="fifo"),"no_broadcast":dict(broadcast=False),
        "no_retain":dict(retain_weights=False),"neither":dict(broadcast=False,retain_weights=False),
        "weight256":dict(weight_bpc=256),"weight4096":dict(weight_bpc=4096),
        "control2ports":dict(control_ports=2),"activation1536":dict(activation_bpc=1536),
        "accumulator48":dict(accumulator_bpc=48),
        "control_low":dict(issue_cycles=1,install_cycles=1,completion_cycles=1),
        "control_high":dict(issue_cycles=4,install_cycles=6,completion_cycles=4),
        "weight_slots1":dict(slots_per_m=1),"weight_slots4":dict(slots_per_m=4),
        "stages1":dict(stages_per_core=1),"stages4":dict(stages_per_core=4),
        "descriptors64":dict(descriptors=64),
        "oracle_zero_control":dict(zero_control_time=True),"oracle_zero_weight":dict(zero_weight_time=True),
        "oracle_zero_activation":dict(zero_activation_time=True),"oracle_zero_accumulator":dict(zero_accumulator_time=True),
        "oracle_zero_control_weight":dict(zero_control_time=True,zero_weight_time=True),
        "oracle_all":dict(zero_control_time=True,zero_weight_time=True,zero_activation_time=True,zero_accumulator_time=True)}
    for selected_row in selected:
        widths=list(map(int,selected_row["shape"].split("+")))
        for window, work in jobs.items():
            for variant, change in variants.items():
                name=f"ablation_{window}__{'_'.join(map(str,widths))}__{variant}"
                cfg=dict(control=selected_row["control"]);cfg.update(change)
                ex.run(req(name,widths,work,selected_row["ownership"],cfg),"ablation",window,variant)
    # A strong-baseline follow-up: select the best uniform and heterogeneous
    # shape on B8 separately for each operand interface, then keep that choice
    # fixed on validation. Do not limit diagnostics to the hand-picked 4+2 pair.
    followups=[]
    for family in ["main","main_packed"]:
        for category in ["uniform","heterogeneous"]:
            candidates=[r for r in ex.rows if r['family']==family and r['window']=='b8_full' and
                (len(set(r['shape'].split('+')))==1 if category=='uniform' else len(r['shape'].split('+'))==2 and len(set(r['shape'].split('+')))==2)]
            chosen=min(candidates,key=lambda r:(r['cycles'],r['name']))
            followups.append(dict(family=family,category=category,chosen=chosen))
            for window,work in jobs.items():
                for variant in ['weight256','weight4096','control2ports','control_low','oracle_zero_control','oracle_zero_weight']:
                    name=f"winner_{family}_{category}_{window}__{variant}"
                    cfg=dict(control=chosen['control'],packed_operand_ports=chosen['packed']);cfg.update(variants[variant])
                    ex.run(req(name,list(map(int,chosen['shape'].split('+'))),work,chosen['ownership'],cfg),'winner_followup',window,variant)
    save(out/'selected_winners.json',followups)
    # Numerical full-dimension trace spot checks, separately from timing sweeps.
    for window, widths in [("b8_full",[4,2]),("b32_tokens8_31",[1]*6)]:
        name=f"numeric_{window}__{'_'.join(map(str,widths))}"
        row=ex.run(req(name,widths,jobs[window],"tile_stealing",dict(control="tile_cohort"),numeric=True),"numeric",window)
        baseline=next(r for r in ex.rows if r["family"]=="main" and r["window"]==window and r["shape"]==row["shape"] and r["ownership"]=="tile_stealing" and r["control"]=="tile_cohort")
        assert row["cycles"]==baseline["cycles"]
    table(out/"main.csv",[r for r in ex.rows if r["family"] in ["main","main_packed"]])
    table(out/"ablations.csv",[r for r in ex.rows if r["family"] in ["ablation","winner_followup"]])
    save(out/"validation.json",dict(passed=True,points=len(ex.rows),runs=2*len(ex.rows),
        numerical_points=sum(r["numeric"] for r in ex.rows),full_trace_points=sum(r["full_trace"] for r in ex.rows),
        all_event_timelines_audited_in_rust=True,all_repeats_full_report_and_timeline_hash_equal=True,receipts=ex.receipts))
    save(out/"provenance.json",dict(binary=str(ex.binary),binary_sha256=sha(ex.binary.read_bytes()),
        runner_sha256=sha(Path(__file__).read_bytes()),input_manifest_sha256=sha(args.input_manifest.read_bytes()),
        oracle_scope="same policy with changed timing; online assignment/cache behavior can change traffic; not fixed-trace causal bounds"))
    print(f"COMPLETE {len(ex.rows)} x 2",flush=True)


if __name__=="__main__": main()
