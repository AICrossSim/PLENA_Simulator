"""Final acceptance audit of saved evidence, without rerunning or changing it.

The default waits for required deliverables.  --partial audits everything that
is currently present and explicitly reports incomplete delivery.  Numerical
acceptance and architecture-performance gates are separate: an honest failure
to beat a baseline is not a corrupt simulation or an audit failure.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import fields
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import time

from .common import ROOT, OLD, REPO, inputs, decode_design, sha, gm, write_json, canonical, digest
from .config import Parameters, BATCHES, MODES, SEED, parameters
from .optimizer import universal_bound

ROUND2_TREE = "c2e29a178010362d4bced173cf86217c74b0613b"
BATCH_FIELDS = [f"B{b}" for b in BATCHES]
SCHEMAS = {
    "E0/repro.csv": "source credits design dispatch window_id ms_ref ms_new abs_diff".split(),
    "E2/bounds_by_window.csv": "set window_id batch bw_GBps unique_weight_MiB fetch_floor_ms compute_lb_ms vector_lb_ms port_lb_ms lb_ms lb_kind b1_frozen_ms b2_frozen_ms hetero_frozen_ms".split(),
    "E2/bounds_summary.csv": "set bw_GBps batch n_windows max_gain_vs_b1_pct max_gain_vs_b2_pct gate5_reachable".split(),
    "E2/port_tight/bounds_by_window.csv": "set window_id batch bw_GBps unique_weight_MiB fetch_floor_ms compute_lb_ms vector_lb_ms port_lb_ms lb_ms lb_kind b1_frozen_ms b2_frozen_ms hetero_frozen_ms".split(),
    "E2/port_tight/bounds_summary.csv": "set bw_GBps batch n_windows max_gain_vs_b1_pct max_gain_vs_b2_pct gate5_reachable".split(),
    "E3/ablation.csv": "config design change batch geomean_ms ratio_vs_S0 hbm_busy_pct single_fetcher_time_pct single_fetcher_GBps avg_inflight_KiB_core0 avg_inflight_KiB_core1 shared_fetch_GBps hbm_MiB refetch_tasks finish_gap_ms iso".split(),
    "E4/dse_progress.csv": "onchip_mode constraint_group family evaluated_points pruned_points current_best_ms current_lb_ms gap_pct".split(),
    "E4/union_generation_budget.csv": "onchip_mode family candidate_generation_budget node_generation_budget generated_points_with_overlap unique_attempted_points unique_successful_points actual_simulator_calls".split(),
    "E4/proof_status.csv": "onchip_mode constraint_group family proof_A_closed proof_B_closed remaining_open_regions".split(),
    "E4/bootstrap_stability.csv": "onchip_mode constraint_group family geometry objective draws selected_count".split(),
    "E4/heldout_main_table.csv": ["onchip_mode", "design", "dispatch", *BATCH_FIELDS, "all_geomean_ms"],
    "E4/heldout_main_table_pipelined.csv": ["onchip_mode", "design", "dispatch", *BATCH_FIELDS, "all_geomean_ms"],
    "E4/heldout_main_table_port_tight.csv": ["onchip_mode", "design", "dispatch", *BATCH_FIELDS, "all_geomean_ms"],
    "E4/breakdown.csv": "onchip_mode design dispatch batch hbm_busy_pct W_port_busy_pct X_port_busy_pct idle_pct finish_gap_ms".split(),
    "E4/gates.csv": "onchip_mode design dispatch ratio_vs_B1 ratio_vs_B2 enter_calibration calibrated_architecture_win".split(),
    "E4/selected_baseline_headroom.csv": "constraint_group onchip_mode dispatch batch n_windows lower_bound_GM_ms B1_GM_ms B2_GM_ms max_gain_vs_B1_pct max_gain_vs_B2_pct gate5_reachable".split(),
    "E4/cross_bw.csv": "hardware_source onchip_mode design geometry at126_geomean_ms at256_geomean_ms".split(),
    "E4/synthetic/reverse_search.csv": "point_index window_id batch bw_GBps origin exact_repeat_identical single_ms hetero_ms delta proof_complete simulator_calls".split(),
    "E5/dispatch/compare.csv": ["design", "onchip_mode", "dispatch", *BATCH_FIELDS, "all_geomean_ms", "ratio_vs_milp", "ratio_vs_B1_fixed"],
    "E5/dispatch/hbm_bytes.csv": "window_id design onchip_mode dispatch hbm_MiB hbm_MiB_milp extra_pct refetch_tasks".split(),
    "E5/dispatch/regression.csv": "window_id design onchip_mode cycles_exact hbm_bytes_exact result_digest_exact".split(),
    "E5/dispatch/development_per_window.csv": "window_id design onchip_mode t_big large_first ratio fixed_ms eft_old_ms result_digest repeat_digest reference_digest repeat_reference_digest raw_file raw_sha256".split(),
    "E5/predictor/predictor_table.csv": ["bw_GBps", "onchip_mode", "design", "method", *BATCH_FIELDS, "all_geomean_ms", "ratio_vs_no_pred", "MAE", "success_at_2", "success_at_4", "success_at_8", "late", "late_gt_64", "late_gt_256", "stall_pct"],
    "E5/sobol/sobol_indices.csv": "param S1 S1_ci ST ST_ci".split(),
    "E5/sobol/sobol_samples.csv": "sample_index weight_tile_service_cycles bank_Bpc dotstagecycles credits bw_GBps vector_scale exact_repeat_identical single_ms hetero_ms delta proof_complete simulator_calls".split(),
    "E5/sobol/flip_points.csv": "sample_index weight_tile_service_cycles bank_Bpc dotstagecycles credits vector_scale delta".split(),
}
REQUIRED = tuple(SCHEMAS) + (
    "TASK.md", "E0/SLOT_SEMANTICS.md", "E0/repeat_checks.json", "E0/METADATA.json",
    "E1/OPERATING_POINT.md", "E2/BOUNDS.md", "E2/bound_checks.json", "E2/repeat_checks.json",
    "E3/ABLATION.md", "E3/per_window.csv", "E3/repeat_checks.json", "E3/RUN_RECEIPT.json",
    "E4/selected_designs.json", "E4/DSE_PROTOCOL.json", "E4/DSE_PROTOCOL_FINAL.json", "E4/PROOF_REPORT_AUDIT.json",
    "E4/cross_bw_repeat_checks.json", "E4/synthetic/SYNTHETIC_PROTOCOL.json",
    "E4/robust_heldout/COMPLETE.json", "E4/robust_heldout/RUN_RECEIPT.json",
    "E4/robust_heldout/SUMMARY_ZH.md", "E4/robust_heldout/robust_objectives.csv",
    "E4/robust_heldout/winners.csv", "E4/robust_heldout/selection_stability.csv",
    "E4/robust_heldout/per_window.csv", "E4/robust_heldout/candidate_counts.csv",
    "E5/dispatch/per_window.json", "E5/dispatch/repeat_checks.json", "E5/dispatch/selection.json",
    "E5/dispatch/gpqa_t128_case.md", "E5/predictor/PREDICTOR.md", "E5/predictor/per_window.csv",
    "E5/predictor/repeat_checks.json", "E5/predictor/TASK_CSV_ARCHIVE.json", "E5/sobol/SOBOL_PROTOCOL.json", "E5/sobol/sobol_samples.csv",
    "REPORT_ZH.md", "README.md",
    *("figures/" + n + ".pdf" for n in ("fig_headroom_vs_bw", "fig_ablation", "fig_gpqa_inflight", "fig_main_by_batch", "fig_predictor")),
)


def read_json(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else Path(path).open()) as f:
        return json.load(f)


def read_csv(path, required=(), *, allow_empty=False):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise AssertionError(f"Missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows and not allow_empty:
        raise AssertionError("CSV has no data rows")
    return rows


def truth(value):
    return value is True or value == "True" or value == "true" or value == 1


def compressed_csv_contract(directory, manifest):
    """Verify lossless raw CSV archival independently of derived tables."""
    path = Path(directory) / manifest["archive"]
    if not truth(manifest.get("lossless")) or sha(path) != manifest["compressed_sha256"]:
        raise AssertionError("Compressed raw CSV SHA or lossless declaration differs")
    h = hashlib.sha256()
    size = 0
    with gzip.open(path, "rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
            size += len(block)
    if h.hexdigest() != manifest["uncompressed_sha256"] or size != manifest["uncompressed_bytes"]:
        raise AssertionError("Uncompressed raw CSV differs from archived original")
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = sum(1 for _ in reader)
    if rows != manifest["rows"] or ("header" in manifest and header != manifest["header"]):
        raise AssertionError("Raw CSV row count or header differs")
    return {"rows": rows, "uncompressed_bytes": size, "lossless_hash_verified": True}


def exact_reproduction(rows):
    seen = set()
    for r in rows:
        if float(r["abs_diff"]) != 0 or float(r["ms_ref"]) != float(r["ms_new"]):
            raise AssertionError(f"Nonzero reproduction difference: {r}")
        key = tuple(r[k] for k in ("source", "credits", "design", "dispatch", "window_id"))
        if key in seen:
            raise AssertionError(f"Duplicate reproduction point: {key}")
        seen.add(key)
    return len(seen)


def check_repeat_record(r):
    pairs = (("result_digest", "repeat_digest"), ("result_digest", "repeated_result_digest"),
             ("heldout_digest", "repeat_heldout_digest"), ("warmup_digest", "repeat_warmup_digest"),
             ("warmup_digest", "repeated_warmup_digest"), ("state_digest", "repeat_state_digest"),
             ("reference_digest", "repeat_reference_digest"),
             ("predictor_state_digest", "repeat_predictor_state_digest"),
             ("final_state_digest", "repeat_final_state_digest"))
    checked = 0
    for a, b in pairs:
        if a in r and b in r:
            if r[a] != r[b]:
                raise AssertionError(f"Repeat digest differs: {a}/{b}")
            checked += 1
    for k in ("exact", "exact_repeat", "exact_repeat_identical", "repeat_identical", "entire_search_repeat_identical"):
        if k in r and not truth(r[k]):
            raise AssertionError(f"Negative repeat receipt: {k}")
    if "repeats" in r and int(r["repeats"]) != 2:
        raise AssertionError("Configuration was not repeated twice")
    return checked


def certificate_contract(c):
    opens = c.get("open_regions", [])
    if c.get("proof_B_closed") and opens:
        raise AssertionError("Global-domain proof B claimed with open regions")
    if "declared_lattice_points" in c:
        closed = sum(int(x["cardinality"]) for x in c.get("certificate", []))
        pending = sum(int(x["cardinality"]) for x in opens)
        if closed + pending != int(c["declared_lattice_points"]):
            raise AssertionError("Certificate closed/open cardinality does not cover declared domain")
    if "total_campaign_simulator_calls" in c and int(c["total_campaign_simulator_calls"]) != 2 * int(c["simulator_calls"]):
        raise AssertionError("Repeated search simulator-call accounting differs")
    witnesses = [x for x in c.get("witnesses", []) if x.get("status") == "evaluated"]
    if len(witnesses) != int(c.get("successful_points", len(witnesses))):
        raise AssertionError("Successful witness count differs")
    for w in witnesses:
        check_repeat_record(w)
        if not truth(w.get("repeat_identical")):
            raise AssertionError("Search witness missing exact two-run check")
    return witnesses


def union_call_accounting(row, c0, c1):
    """Two selections refer to one already-executed pair of campaigns."""
    calls = sum(int(c["total_campaign_simulator_calls"]) for c in (c0, c1))
    if int(row["actual_simulator_calls"]) != calls:
        raise AssertionError("Union generation call count differs from actual source campaigns")
    for field, source in (("candidate_generation_budget", "candidate_budget"),
                          ("node_generation_budget", "node_budget"),
                          ("generated_points_with_overlap", "evaluated_points")):
        if int(row[field]) != sum(int(c[source]) for c in (c0, c1)):
            raise AssertionError("Union generation budget differs: " + field)
    return calls


def assert_floor(latency_ms, lower_cycles, label=""):
    cycles = float(latency_ms) * 1e6
    if not math.isfinite(cycles) or cycles <= 0:
        raise AssertionError(f"Invalid latency: {label}")
    # Floating summation tolerance, not a percentage model-error allowance.
    if cycles + max(1e-6, abs(lower_cycles) * 1e-12) < lower_cycles:
        raise AssertionError(f"Latency below global LB: {label}, {cycles} < {lower_cycles}")


def indexed_campaign_rows(rows, field, expected_count, *, partial=False):
    """Reject duplicate/relabelled points; partial progress may be a subset."""
    indexed = {}
    for row in rows:
        index = int(row[field])
        if index not in range(expected_count) or index in indexed:
            raise AssertionError("Duplicate or out-of-range campaign index: " + str(index))
        if not truth(row.get("exact_repeat_identical")):
            raise AssertionError("Campaign point lacks its complete-search two-run equality marker")
        indexed[index] = row
    if not partial and set(indexed) != set(range(expected_count)):
        raise AssertionError("Incomplete campaign row IDs: " + field)
    return indexed


def sensitivity_point_contract(data, row, expected_parameters, window_ids):
    """Independently pair one search certificate with its published result."""
    families = ("single", "homogeneous", "5+1", "4+2", "3+3")
    if set(data.get("families", {})) != set(families):
        raise AssertionError("Sensitivity point does not cover all five hardware families")
    if data.get("development_window_ids") != list(window_ids):
        raise AssertionError("Sensitivity certificate input window IDs/order differ")
    if canonical(data.get("parameters")) != canonical(expected_parameters):
        raise AssertionError("Sensitivity certificate parameters differ from its planned point")
    if data.get("seed") != SEED:
        raise AssertionError("Sensitivity certificate seed differs")
    for family, cert in data["families"].items():
        if cert.get("family") != family or canonical(cert.get("parameters")) != canonical(expected_parameters):
            raise AssertionError("Family identity or parameters differ within sensitivity point")
    closed = all(truth(c["proof_B_closed"]) for c in data["families"].values())
    if truth(data.get("proof_complete")) != closed:
        raise AssertionError("Sensitivity root proof marker differs from family certificates")
    if row is None:
        return
    if not truth(row.get("exact_repeat_identical")):
        raise AssertionError("Campaign point lacks its complete-search two-run equality marker")
    if truth(row["proof_complete"]) != closed:
        raise AssertionError("Published point proof marker differs from its certificate")
    single = data["families"]["single"]
    duals = [data["families"][name] for name in ("5+1", "4+2", "3+3")]
    chosen = min(duals, key=lambda c: (c["selected"]["score_ms"], c["family"]))
    sm = single["selected"]["score_ms"]
    hm = chosen["selected"]["score_ms"]
    values = {"single_ms": sm, "hetero_ms": hm, "delta": hm / sm - 1,
              "delta_lower": min(c["global_lb_ms"] for c in duals) / sm - 1,
              "delta_upper": hm / single["global_lb_ms"] - 1,
              "gap_pct": max(c["gap_pct"] for c in data["families"].values())}
    for key, value in values.items():
        if not math.isclose(float(row[key]), value, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("Published sensitivity result differs from certificate: " + key)
    for field, family in (("single_design", single), ("hetero_design", chosen)):
        if canonical(json.loads(row[field])) != canonical(family["selected"]["design"]):
            raise AssertionError("Published sensitivity hardware differs from certificate: " + field)
    cores = chosen["selected"]["design"]["cores"]
    distinct = len(cores) == 2 and canonical(cores[0]) != canonical(cores[1])
    if truth(row["compute_shapes_distinct"]) != distinct or row["hetero_family"] != chosen["family"]:
        raise AssertionError("Published sensitivity family/shape distinction differs")
    if int(row["evaluated_points"]) != sum(c["evaluated_points"] for c in data["families"].values()):
        raise AssertionError("Published sensitivity point count differs")
    if int(row["simulator_calls"]) != 2 * sum(c["simulator_calls"] for c in data["families"].values()):
        raise AssertionError("Published sensitivity repeated simulator-call count differs")


def single_regression_contract(rows, raw, window_ids, *, modes=MODES, partial=False):
    """Check unique protocol coverage and raw results, not only true CSV flags."""
    expected = set(itertools.product(modes, ("B0", "B1"), window_ids))
    seen = set()
    for row in rows:
        key = row["onchip_mode"], row["design"], row["window_id"]
        if key not in expected or key in seen:
            raise AssertionError("Duplicate or unexpected single-core regression protocol/window")
        seen.add(key)
        if not all(truth(row[k]) for k in ("cycles_exact", "hbm_bytes_exact", "result_digest_exact")):
            raise AssertionError("Single-core regression CSV reports non-bitexact result")
        old = raw[key + ("eft_old",)]
        new = raw[key + ("fixed",)]
        if old["cycles"] != new["cycles"] or old["hbm_bytes"] != new["hbm_bytes"] or digest(old) != digest(new):
            raise AssertionError("Single-core raw fixed/EFT results differ despite regression flags")
    if not partial and seen != expected:
        raise AssertionError("Incomplete single-core regression protocol/window coverage")
    return len(seen)


def parameters_at_saved_bw(mode, bw):
    """CSV stores the exact derived cap, while registry names are rounded."""
    closest = min((256, 390, 520), key=lambda c: abs(parameters(mode, credits=c).hbm_bandwidth - float(bw)))
    p = parameters(mode, credits=closest)
    if abs(p.hbm_bandwidth - float(bw)) > .01:
        raise AssertionError(f"Unknown saved operating point: {bw}")
    return p


def check_physical_ledger(result):
    """Check installed budgets separately from estimated inflight proxies."""
    ledger = result["ledger"]
    private = ledger["private"]
    pool = int(ledger["shared_landing_pool_B"])
    installed = sum(ledger["fixed_storage_B"].values()) + pool + sum(
        sum(private[k]) for k in ("w_bytes", "x_bytes", "acc_bytes", "z_bytes"))
    if installed != 2158592 or installed != int(ledger["installed_storage_B"]):
        raise AssertionError("Physical installed SRAM exceeds or differs from iso budget")
    for field, total in (("w_banks", 64), ("x_banks", 24), ("acc_banks", 12), ("vector_lanes", 64)):
        if sum(private[field]) != total:
            raise AssertionError("Physical port/lane budget changed: " + field)
    if ledger["main_multipliers"] != 12288:
        raise AssertionError("Physical multiplier budget changed")
    if ledger["landing_mode"] == "shared" and (sum(private["w_bytes"]) or not truth(ledger["shared_pool_counted_once"])):
        raise AssertionError("Shared pool includes free private W")
    for segment in result.get("segments", []):
        p = segment.get("pool_ledger")
        if p:
            subtotal = sum(p[k] for k in ("current_reserved_bytes", "next_reserved_bytes", "lookahead_bytes"))
            if p["pool_bytes"] != pool or subtotal != p["used_bytes"] or subtotal > pool or p["unused_bytes"] != pool - subtotal:
                raise AssertionError("Actual common byte-pool admission exceeded installed capacity")


class Audit:
    def __init__(self, root=ROOT, *, partial=False):
        self.root = Path(root)
        self.partial = partial
        self.checks = []
        self.errors = []
        self.lb_counts = Counter()
        self.lb_cache = {}
        self.lb_violations = []
        self.repeat_pairs = 0
        self.repeat_configs = 0
        self.search = Counter()
        sets = inputs()
        self.dev = sets["development"]
        self.held = sets["heldout"]
        self.by_id = {w["id"]: w for w in (*self.dev, *self.held)}

    def check(self, name, fn):
        try:
            result = fn()
            self.checks.append({"name": name, "passed": True, "details": result})
            return result
        except Exception as e:
            issue = {"name": name, "passed": False, "error": f"{type(e).__name__}: {e}"}
            self.checks.append(issue)
            self.errors.append(issue)
            return None

    def p_from_dict(self, value):
        if "weight_tile_service_cycles" in value:
            from .sensitivity import SensitivityParameters
            return SensitivityParameters(**value)
        return Parameters(**value)

    def floor(self, w, p, *, unbounded_w=False):
        k = (w["id"], json.dumps({f.name: getattr(p, f.name) for f in fields(p)}, sort_keys=True), unbounded_w)
        if k not in self.lb_cache:
            b = universal_bound(w, p)
            self.lb_cache[k] = (max(v for name, v in b["terms"].items() if name != "HBM_rowchunk_optimistic532")
                                if unbounded_w else b["lb_cycles"])
        return self.lb_cache[k]

    def point(self, stage, w, ms, p, *, unbounded_w=False):
        bound = self.floor(w, p, unbounded_w=unbounded_w)
        try:
            assert_floor(ms, bound, f"{stage}/{w['id']}")
        except AssertionError:
            self.lb_violations.append({"stage": stage, "window_id": w["id"],
                                       "latency_ms": ms, "lb_ms": bound / 1e6})
            raise
        self.lb_counts[stage] += 1

    def window(self, row):
        wid = row.get("window_id") or row.get("workload")
        if isinstance(wid, dict):
            wid = wid["id"]
        return self.by_id[wid]

    def schemas(self):
        count = 0
        for name, cols in SCHEMAS.items():
            path = self.root / name
            if path.exists():
                self.check("schema:" + name, lambda p=path, c=cols, n=name:
                           len(read_csv(p, c, allow_empty=n.endswith("flip_points.csv"))))
                count += 1
        return count

    def historical_tree(self):
        got = subprocess.check_output(["git", "rev-parse", "HEAD:research/moe_dispatch/round2"], cwd=REPO, text=True).strip()
        if got != ROUND2_TREE:
            raise AssertionError(f"Historical tree changed: {got} != {ROUND2_TREE}")
        r = subprocess.run(["git", "diff", "HEAD", "--exit-code", "--", "research/moe_dispatch/round2"], cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            raise AssertionError("Round2 worktree or index contains modifications: " + r.stdout[:1000])
        extra = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "research/moe_dispatch/round2"], cwd=REPO, text=True).strip()
        if extra:
            raise AssertionError("New files appeared in read-only round2: " + extra[:1000])
        return {"tree": got, "unchanged": True}

    def reproduction(self):
        rows = read_csv(self.root / "E0/repro.csv", SCHEMAS["E0/repro.csv"])
        n = exact_reproduction(rows)
        from .reproduce import specs
        expected = Counter((s["source"], str(s["credits"]), s["design"], s["method"]) for s in specs() for _ in s["refs"])
        actual = Counter((r["source"], r["credits"], r["design"], r["dispatch"]) for r in rows)
        if actual != expected:
            raise AssertionError("E0 configuration/window counts differ from immutable reference")
        for r in rows:
            mode = "port_tight" if r["source"].endswith("port_tight") else "pipelined"
            self.point("E0", self.window(r), r["ms_new"], parameters(mode, credits=int(r["credits"])))
        return {"rows": n, "configurations": len(actual), "nonzero_differences": 0}

    def bounds_tables(self):
        n = 0
        for mode in MODES:
            base = self.root / ("E2" if mode == "pipelined" else "E2/port_tight")
            path = base / "bounds_by_window.csv"
            if not path.exists():
                continue
            rows = read_csv(path)
            if len(rows) != 3 * (len(self.dev) + len(self.held)):
                raise AssertionError(f"Incomplete E2 rows: {mode}")
            for r in rows:
                p = parameters_at_saved_bw(mode, r["bw_GBps"])
                w = self.window(r)
                bound = self.floor(w, p) / 1e6
                if not math.isclose(bound, float(r["lb_ms"]), rel_tol=1e-12, abs_tol=1e-12):
                    raise AssertionError("Saved E2 LB differs from current global bound")
                for field in ("b1_frozen_ms", "b2_frozen_ms", "hetero_frozen_ms"):
                    self.point("E2", w, r[field], p); n += 1
        return n

    def per_window(self):
        checked = {}
        sources = (("E3/per_window.csv", "E3"), ("E5/dispatch/per_window.json", "E5_dispatch"),
                   ("E5/dispatch/old126_runtime_settings.json", "E5_old_runtime_setting"),
                   ("E5/predictor/per_window.csv", "E5_predictor"))
        for name, stage in sources:
            path = self.root / name
            if not path.exists():
                continue
            rows = read_json(path) if name.endswith(".json") else read_csv(path)
            groups = defaultdict(list)
            for r in rows:
                mode = r.get("onchip_mode", "pipelined")
                p = parameters(mode, credits=int(r["credits"])) if "credits" in r else parameters_at_saved_bw(mode, r.get("bw_GBps", 256))
                self.point(stage, self.window(r), r["latency_ms"], p,
                           unbounded_w=stage == "E3" and not truth(r["iso"]))
                self.repeat_pairs += check_repeat_record(r)
                fields_key = ("config",) if stage == "E3" else (("bw_GBps", "onchip_mode", "design", "method") if stage == "E5_predictor" else ("constraint_group", "onchip_mode", "design", "dispatch"))
                groups[tuple(str(r.get(k, "")) for k in fields_key)].append(r["window_id"])
            expected = Counter(w["id"] for w in self.held)
            if any(Counter(ids) != expected for ids in groups.values()):
                raise AssertionError("Incomplete or duplicate per-window group: " + name)
            checked[stage] = len(rows)
        return checked

    def receipts(self):
        checked = 0
        paths = [self.root / n for n in ("E0/repeat_checks.json", "E2/repeat_checks.json", "E3/repeat_checks.json",
                 "E4/cross_bw_repeat_checks.json", "E5/dispatch/repeat_checks.json", "E5/predictor/repeat_checks.json")]
        paths += list((self.root / "E5/predictor").glob("*/repeat_checks.json"))
        raw_seen = set()
        for path in paths:
            if not path.exists():
                continue
            for r in read_json(path):
                self.repeat_pairs += check_repeat_record(r)
                self.repeat_configs += 1; checked += 1
                if "hardware" in r:
                    design = decode_design(r["hardware"])
                    if design.diagnostic_unbounded_w and truth(r.get("iso", False)):
                        raise AssertionError("Infinite-W diagnostic labelled iso-resource")
                if "raw_file" in r:
                    rp = self.root / r["raw_file"]
                    if sha(rp) != r["raw_sha256"]:
                        raise AssertionError("Raw result SHA256 differs: " + str(rp))
                    raw_seen.add(rp.resolve())
                for item in r.get("conditional_oracle_checks", []):
                    if not truth(item.get("physically_replayed")):
                        raise AssertionError("Oracle row lacks independent physical replay")
                    self.repeat_pairs += check_repeat_record(item)
                if "oracle_raw_file" in r:
                    rp = self.root / r["oracle_raw_file"]
                    if sha(rp) != r["oracle_raw_sha256"]:
                        raise AssertionError("Oracle raw SHA256 differs")
        # Check raw physical records too.  During --partial this includes fully
        # written orphan files whose enclosing campaign has not yet completed.
        for sub in ("E3/raw", "E4/cross_bw_raw", "E5/dispatch/raw", "E5/predictor/raw"):
            for rp in sorted((self.root / sub).glob("*.json.gz")):
                rows = read_json(rp)
                if len(rows) != len(self.held):
                    raise AssertionError("Incomplete raw heldout sequence: " + str(rp))
                mode = "port_tight" if "port_tight" in rp.name else "pipelined"
                if sub == "E3/raw":
                    credit = 520
                elif sub == "E5/predictor/raw":
                    credit = int(rp.name.split("_")[0])
                else:
                    credit = int(rp.name.removesuffix(".json.gz").split("_")[-1])
                p = parameters(mode, credits=credit)
                for w, r in zip(self.held, rows):
                    if r.get("workload") not in (w["id"], w):
                        raise AssertionError("Raw result window order differs: " + str(rp))
                    self.point(sub + "_physical", w, r["latency_ms"], p,
                               unbounded_w=truth(r.get("ledger", {}).get("non_iso_diagnostic", False)))
                    check_physical_ledger(r)
                    for s in r.get("segments", []):
                        rate = float(s["hbm_rate_Bpc"])
                        if rate < 0 or rate > p.hbm_bandwidth + 1e-8:
                            raise AssertionError("Shared global HBM credit ceiling exceeded")
                        if not math.isclose(rate, sum(s["hbm_rate_Bpc_core"]), abs_tol=1e-8, rel_tol=1e-12):
                            raise AssertionError("Global/core HBM service rates do not conserve bytes")
        csv_manifest = self.root / "E5/predictor/TASK_CSV_ARCHIVE.json"
        archived_csv = (compressed_csv_contract(csv_manifest.parent, read_json(csv_manifest))
                        if csv_manifest.exists() else None)
        return {"configuration_receipts": checked, "digest_pairs": self.repeat_pairs,
                "raw_receipts": len(raw_seen), "archived_prediction_csv": archived_csv}

    def development_dispatch(self):
        """Audit both physically repeated fixed and EFT development references."""
        raw = {}
        for path in sorted((self.root / "E5/dispatch/development_raw").glob("*.json.gz")):
            data = read_json(path)
            if set(data) != {"fixed", "eft_old"}:
                raise AssertionError("Development raw lacks fixed/EFT reference")
            mode = "port_tight" if "port_tight" in path.name else "pipelined"
            p = parameters(mode)
            indexed = {}
            for method, sequence in data.items():
                if len(sequence) != len(self.dev):
                    raise AssertionError("Incomplete development reference sequence")
                for w, r in zip(self.dev, sequence):
                    if r["workload"] != w["id"]:
                        raise AssertionError("Development raw input ordering differs")
                    self.point("E5_dispatch_development_raw", w, r["latency_ms"], p)
                    check_physical_ledger(r)
                    indexed[method, w["id"]] = r
            raw[str(path.relative_to(self.root))] = (sha(path), indexed)
        table = self.root / "E5/dispatch/development_per_window.csv"
        groups = defaultdict(list)
        rows = read_csv(table) if table.exists() else []
        for row in rows:
            self.repeat_pairs += check_repeat_record(row)
            stored_sha, indexed = raw[row["raw_file"]]
            if stored_sha != row["raw_sha256"]:
                raise AssertionError("Development raw SHA differs")
            fixed = indexed["fixed", row["window_id"]]
            eft = indexed["eft_old", row["window_id"]]
            if digest(fixed) != row["result_digest"] or digest(eft) != row["reference_digest"]:
                raise AssertionError("Development CSV digest differs from physical raw")
            if (float(row["fixed_ms"]) != fixed["latency_ms"] or
                    float(row["eft_old_ms"]) != eft["latency_ms"] or
                    not math.isclose(float(row["ratio"]), fixed["cycles"] / eft["cycles"], rel_tol=1e-12)):
                raise AssertionError("Development candidate/reference timing differs")
            groups[tuple(row[k] for k in ("onchip_mode", "design", "t_big", "large_first"))].append(row["window_id"])
        expected = Counter(w["id"] for w in self.dev)
        if any(Counter(ids) != expected for ids in groups.values()):
            raise AssertionError("Incomplete or duplicate development candidate windows")
        if not self.partial and (len(groups) != 140 or len(raw) != 140):
            raise AssertionError("Expected 140 independently repeated development dispatch configurations")
        return {"raw_configurations": len(raw), "CSV_configurations": len(groups), "rows": len(rows)}

    def regression(self):
        rows = read_csv(self.root / "E5/dispatch/regression.csv")
        receipts = read_json(self.root / "E5/dispatch/repeat_checks.json")
        raw = {}; protocols = set()
        for receipt in receipts:
            if (receipt["constraint_group"] != "C0" or receipt["design"] not in ("B0", "B1") or
                    receipt["dispatch"] not in ("eft_old", "fixed")):
                continue
            ident = receipt["onchip_mode"], receipt["design"], receipt["dispatch"]
            if ident in protocols:
                raise AssertionError("Duplicate single-core physical regression protocol")
            protocols.add(ident)
            path = self.root / receipt["raw_file"]
            if sha(path) != receipt["raw_sha256"]:
                raise AssertionError("Single-core regression raw SHA differs")
            records = read_json(path)
            ids = [r["workload"] for r in records]
            if ids != [w["id"] for w in self.held]:
                raise AssertionError("Single-core regression raw windows/order differ")
            for r in records:
                raw[ident[:2] + (r["workload"], ident[2])] = r
        count = single_regression_contract(rows, raw, [w["id"] for w in self.held], partial=self.partial)
        return {"rows": count, "raw_protocols": len(protocols), "all_bitexact": True,
                "raw_old_fixed_rechecked": True}

    def selected_baseline_headroom(self):
        """Pair the same LB with measured new baselines, never assume monotonicity."""
        table = read_csv(self.root / "E4/selected_baseline_headroom.csv")
        expected = set(itertools.product(MODES, ("fixed", "milp"), tuple(map(str, BATCHES)) + ("all",)))
        seen = set()
        records = read_json(self.root / "E5/dispatch/per_window.json")
        measured = {(r["onchip_mode"], r["dispatch"], r["design"], r["window_id"]): float(r["latency_ms"])
                    for r in records if r["constraint_group"] == "C0" and r["design"] in ("B1", "B2")}
        bounds = {}
        for mode in MODES:
            path = self.root / ("E2" if mode == "pipelined" else "E2/port_tight") / "bounds_by_window.csv"
            bounds.update({(mode, r["window_id"]): float(r["lb_ms"]) for r in read_csv(path)
                           if r["set"] == "heldout" and float(r["bw_GBps"]) == 256})
        for row in table:
            ident = row["onchip_mode"], row["dispatch"], row["batch"]
            if ident in seen or ident not in expected or row["constraint_group"] != "C0":
                raise AssertionError("New baseline headroom has duplicate or wrong protocol")
            seen.add(ident)
            mode, dispatch, batch = ident
            windows = [w for w in self.held if batch == "all" or str(w["batch"]) == batch]
            if int(row["n_windows"]) != len(windows):
                raise AssertionError("New baseline headroom window count differs")
            lb = [bounds[mode, w["id"]] for w in windows]
            values = {"lower_bound_GM_ms": gm(lb)}
            for design in ("B1", "B2"):
                times = [measured[mode, dispatch, design, w["id"]] for w in windows]
                values[design + "_GM_ms"] = gm(times)
                values["max_gain_vs_" + design + "_pct"] = 100 * (1 - gm(a / b for a, b in zip(lb, times)))
            for field, value in values.items():
                if not math.isclose(float(row[field]), value, rel_tol=1e-12, abs_tol=1e-10):
                    raise AssertionError("New baseline headroom differs from saved paired windows: " + field)
            gate = min(values["max_gain_vs_B1_pct"], values["max_gain_vs_B2_pct"]) >= 5
            if truth(row["gate5_reachable"]) != gate:
                raise AssertionError("New baseline headroom gate differs")
        if seen != expected:
            raise AssertionError("Incomplete new baseline headroom protocols or batches")
        return {"paired_rows": len(table), "assumed_heldout_monotonicity": False}

    def one_certificate(self, c, workloads, *, repeated_entire=True, stage="E4_search", count_calls=True):
        witnesses = certificate_contract(c)
        p = self.p_from_dict(c["parameters"])
        for x in witnesses:
            d = decode_design(x["design"])
            if d.diagnostic_unbounded_w:
                raise AssertionError("Diagnostic infinite-W entered hardware search")
            if len(x["latencies_ms"]) != len(workloads):
                raise AssertionError("Search witness latency vector length differs")
            for w, ms in zip(workloads, x["latencies_ms"]):
                self.point(stage, w, ms, p)
            if not math.isclose(gm(x["latencies_ms"]), x["score_ms"], rel_tol=1e-12):
                raise AssertionError("Search witness score differs from per-window geometric mean")
        if count_calls:
            self.search["certificates"] += 1
            self.search["evaluated_points"] += int(c["evaluated_points"])
            self.search["successful_points"] += len(witnesses)
            self.search["simulator_calls_including_repeats"] += int(c.get("total_campaign_simulator_calls", 2 * int(c["simulator_calls"])))
            self.search["proof_B_closed_families"] += bool(c["proof_B_closed"])
            self.search["open_regions"] += len(c["open_regions"])
        else:
            self.search["union_selection_certificates_no_new_calls"] += 1
        return len(witnesses)

    def certificates(self):
        count = 0
        originals = {}
        for path in sorted((self.root / "E4/certificates").glob("*.json")):
            c = read_json(path)
            if not truth(c.get("entire_search_repeat_identical")):
                raise AssertionError("Missing complete-search deterministic repeat: " + str(path))
            self.one_certificate(c, self.dev); count += 1
            originals[c.get("onchip_mode", c["parameters"]["onchip_mode"]), c["constraint_group"], c["family"]] = c
        if not self.partial and count != 20:
            raise AssertionError(f"Expected twenty DSE family/mode/constraint certificates, got {count}")
        final_count = self.final_certificates(originals)
        synthetic = self.sensitivity_campaign("synthetic")
        sobol = self.sensitivity_campaign("sobol")
        return {"main_family_certificates": count, "final_union_selection_certificates": final_count,
                "synthetic_points": synthetic, "sobol_samples": sobol, **self.search}

    def sensitivity_campaign(self, kind):
        """Match every completed point to its grid/sample and summary record."""
        from .sensitivity import SensitivityParameters
        if kind == "sobol":
            from SALib.sample import sobol as sampler
            problem = {"num_vars": 5,
                       "names": ["weight_tile_service_cycles", "bank_Bpc", "dotstagecycles", "credits", "vector_scale"],
                       "bounds": [[1, 30.4], [8, 32], [1, 4], [256, 640], [.5, 2]]}
            # This regenerates only the deterministic sampling coordinates,
            # never a physical simulation or a hardware search.
            planned = sampler.sample(problem, 256, calc_second_order=False, seed=SEED)
            count, base, field, name, pattern = 1792, self.root / "E5/sobol", "sample_index", "sobol_samples.csv", "*.json.gz"
            assert len(planned) == count
        else:
            from ..round2.regions import synthetic as make_synthetic
            calibration = read_json(OLD / "results/E3/synthetic_calibration.json")
            planned = list(itertools.product((2, 4, 8, 16, 32, 64, 128, 256), range(5), (0, 1, 2, 4), ((64, 6), (128, 8), (256, 8)), (512, 1408, 2048)))
            count, base, field, name, pattern = 1440, self.root / "E4/synthetic", "point_index", "reverse_search.csv", "*.json"
            assert len(planned) == count
        rows = read_csv(base / name) if (base / name).exists() else []
        indexed = indexed_campaign_rows(rows, field, count, partial=self.partial)
        paths = {}
        for path in sorted((base / "certificates").glob(pattern)):
            index = int(path.name.split(".")[0])
            if index not in range(count) or index in paths:
                raise AssertionError("Duplicate or out-of-range sensitivity certificate ID")
            paths[index] = path
        if not self.partial and set(paths) != set(range(count)):
            raise AssertionError("Incomplete sensitivity certificate IDs: " + kind)
        if set(indexed) - set(paths):
            raise AssertionError("Published sensitivity row has no corresponding saved certificate")
        for index, path in paths.items():
            row = indexed.get(index)
            if kind == "sobol":
                tau, bank, dot, credits, vector = planned[index]
                p = SensitivityParameters(weight_tile_service_cycles=float(tau), bank_Bpc=float(bank),
                        dotstagecycles=float(dot), credits=int(round(credits)), vector_scale=float(vector))
                workloads = self.dev
                if row is not None:
                    for key in ("weight_tile_service_cycles", "bank_Bpc", "dotstagecycles", "credits", "vector_scale"):
                        if float(row[key]) != getattr(p, key):
                            raise AssertionError("Sobol CSV parameters differ from planned Saltelli sample: " + key)
            else:
                batch, level, units, (experts, topk), ffn = planned[index]
                workloads = [make_synthetic(batch, calibration["levels"][level], units, experts, topk, ffn, SEED)]
                p = SensitivityParameters()
                if row is not None and (row["window_id"] != workloads[0]["id"] or int(row["batch"]) != batch):
                    raise AssertionError("Synthetic CSV input differs from planned grid point")
            if row is not None and float(row["bw_GBps"]) != p.hbm_bandwidth:
                raise AssertionError("Sensitivity CSV effective HBM ceiling differs")
            data = read_json(path)
            expected_parameters = {f.name: getattr(p, f.name) for f in fields(p)}
            sensitivity_point_contract(data, row, expected_parameters, [w["id"] for w in workloads])
            for cert in data["families"].values():
                self.one_certificate(cert, workloads, stage="E5_sobol_search" if kind == "sobol" else "E4_synthetic_search")
        return len(paths)

    def final_certificates(self, originals):
        paths = sorted((self.root / "E4/final_certificates").glob("*.json"))
        if not paths and self.partial:
            return 0
        if len(paths) != 20:
            raise AssertionError("Expected twenty final union selection certificates")
        from .search import FAMILY_LABELS, c1_legal, key
        budgets = read_csv(self.root / "E4/union_generation_budget.csv")
        rows = {(r["onchip_mode"], r["family"]): r for r in budgets}
        if len(rows) != 10 or len(budgets) != 10:
            raise AssertionError("Union generation must have ten unique mode/family budgets")
        actual_calls = 0
        for (mode, family), row in rows.items():
            actual_calls += union_call_accounting(row, originals[mode, "C0", family], originals[mode, "C1", family])
        selected = read_json(self.root / "E4/selected_designs.json")
        final = {}
        for path in paths:
            c = read_json(path)
            mode, group, family = c["onchip_mode"], c["constraint_group"], c["family"]
            ident = mode, group, family
            if ident in final:
                raise AssertionError("Duplicate final selection family")
            final[ident] = c
            self.one_certificate(c, self.dev, stage="E4_union_selection", count_calls=False)
            sources = [originals[mode, g, family] for g in ("C0", "C1")]
            calls = sum(int(s["total_campaign_simulator_calls"]) for s in sources)
            if int(c["total_campaign_simulator_calls"]) != calls:
                raise AssertionError("Final selection repeats or omits union source calls")
            source_points = {}
            for source in sources:
                for witness in source["witnesses"]:
                    point_key = key(decode_design(witness["design"]))
                    if point_key in source_points and witness["status"] == "evaluated":
                        old = source_points[point_key]
                        if canonical(old["latencies_ms"]) != canonical(witness["latencies_ms"]):
                            raise AssertionError("Repeated cross-constraint witness changed")
                    source_points[point_key] = witness
            eligible = {k: w for k, w in source_points.items() if w["status"] == "evaluated" and
                        (group == "C0" or c1_legal(decode_design(w["design"])))}
            if {key(decode_design(w["design"])) for w in c["witnesses"]} != set(eligible):
                raise AssertionError("Final selection is not the complete eligible source union")
            best = min(eligible.values(), key=lambda w: (w["score_ms"], key(decode_design(w["design"]))))
            if key(decode_design(c["selected"]["design"])) != key(decode_design(best["design"])):
                raise AssertionError("Final union did not select its best eligible development witness")
            label = FAMILY_LABELS[family]
            if key(decode_design(selected["modes"][mode][group][label])) != key(decode_design(best["design"])):
                raise AssertionError("Frozen hardware differs from final development certificate")
        for mode, family in rows:
            if final[mode, "C0", family]["selected"]["score_ms"] > final[mode, "C1", family]["selected"]["score_ms"] + 1e-12:
                raise AssertionError("C0 source union is worse than its C1 subset")
        self.search["main_union_actual_source_calls_counted_once"] = actual_calls
        return len(paths)

    def selected(self):
        value = read_json(self.root / "E4/selected_designs.json")
        expected = {"B1", "B2", "H51", "H42", "H33"}
        count = 0
        for mode in MODES:
            for group in ("C0", "C1"):
                ds = value["modes"][mode][group]
                if set(ds) != expected:
                    raise AssertionError("Frozen selected design family set differs")
                for name, d in ds.items():
                    d = decode_design(d)
                    if d.diagnostic_unbounded_w:
                        raise AssertionError("Non-iso diagnostic entered main selection")
                    if name == "B2" and d.cores[0] != d.cores[1]:
                        raise AssertionError("Homogeneous selected cores have unequal geometries")
                    if group == "C1":
                        from .search import c1_legal
                        if not c1_legal(d):
                            raise AssertionError("C1 selected design lacks the required solo inflight capacity")
                    count += 1
        return count

    def statistical_counts(self):
        details = {}
        path = self.root / "E4/bootstrap_stability.csv"
        if path.exists():
            groups = defaultdict(int)
            for r in read_csv(path):
                if int(r["draws"]) != 200:
                    raise AssertionError("Bootstrap does not use 200 draws")
                groups[tuple(r[k] for k in ("onchip_mode", "constraint_group", "family", "objective"))] += int(r["selected_count"])
            if any(v != 200 for v in groups.values()):
                raise AssertionError("Bootstrap choice counts do not sum to 200")
            details["bootstrap_groups"] = len(groups)
        for file, count, field in (("E5/sobol/SOBOL_PROTOCOL.json", 1792, "samples_completed"), ("E4/synthetic/SYNTHETIC_PROTOCOL.json", 1440, "completed_points")):
            path = self.root / file
            if path.exists():
                r = read_json(path)
                # The actual complete Cartesian product is checked too, rather
                # than trusting the historical progress print's 1440 label.
                expected = count
                if int(r[field]) != expected or int(r["repeats"]) != 2 or ("synthetic" in file and int(r["planned_points"]) != count):
                    raise AssertionError("Incomplete sensitivity campaign: " + file)
                details[file] = r[field]
        return details

    def aggregate_tables(self):
        """Recompute published latency summaries from saved paired windows."""
        count = 0
        path = self.root / "E3/per_window.csv"
        if path.exists():
            window_rows = read_csv(path)
            s0 = {r["window_id"]: float(r["latency_ms"]) for r in window_rows if r["config"] == "S0"}
            for r in read_csv(self.root / "E3/ablation.csv"):
                rows = [x for x in window_rows if x["config"] == r["config"] and (r["batch"] == "all" or x["batch"] == r["batch"])]
                value = gm(x["latency_ms"] for x in rows)
                ratio = gm(float(x["latency_ms"]) / s0[x["window_id"]] for x in rows)
                if not math.isclose(value, float(r["geomean_ms"]), rel_tol=1e-12) or not math.isclose(ratio, float(r["ratio_vs_S0"]), rel_tol=1e-12):
                    raise AssertionError("E3 geometric-mean table differs from paired windows")
                count += 1
        path = self.root / "E5/dispatch/per_window.json"
        if path.exists():
            rs = read_json(path)
            for name in ("E5/dispatch/compare.csv", "E4/heldout_main_table.csv"):
                tab = self.root / name
                if not tab.exists():
                    continue
                for r in read_csv(tab):
                    key = (r.get("constraint_group", "C0"), r["onchip_mode"], r["design"], r["dispatch"])
                    rows = [x for x in rs if (x["constraint_group"], x["onchip_mode"], x["design"], x["dispatch"]) == key]
                    for b in (*BATCHES, "all"):
                        chosen = [x for x in rows if b == "all" or int(x["batch"]) == b]
                        field = "all_geomean_ms" if b == "all" else f"B{b}"
                        if not math.isclose(gm(x["latency_ms"] for x in chosen), float(r[field]), rel_tol=1e-12):
                            raise AssertionError(f"Published GM differs: {name}, {key}, {field}")
                        count += 1
        path = self.root / "E5/predictor/per_window.csv"
        table = self.root / "E5/predictor/predictor_table.csv"
        if path.exists() and table.exists():
            rs = read_csv(path)
            for r in read_csv(table):
                rows = [x for x in rs if (x["onchip_mode"], x["design"], x["method"]) == (r["onchip_mode"], r["design"], r["method"]) and abs(float(x["bw_GBps"]) - float(r["bw_GBps"])) < .01]
                for b in (*BATCHES, "all"):
                    chosen = [x for x in rows if b == "all" or int(x["batch"]) == b]
                    field = "all_geomean_ms" if b == "all" else f"B{b}"
                    if not math.isclose(gm(x["latency_ms"] for x in chosen), float(r[field]), rel_tol=1e-12):
                        raise AssertionError("Predictor GM differs from window values")
                    count += 1
        return {"recomputed_table_cells": count}

    def robust_heldout(self):
        """Independently audit development-frozen heldout diagnostics."""
        from .robust_heldout_audit import audit
        result = audit(self.root)
        if not result["complete"] or not result["passed"]:
            raise AssertionError("Robust heldout audit failed: " + canonical(result["failures"]))
        return {"coverage": result["coverage"],
                "max_objective_error": result["max_objective_error"],
                "selected_hardware_unchanged": True}

    def run(self):
        missing = [n for n in REQUIRED if not (self.root / n).is_file()]
        self.schemas()
        self.check("round2_readonly", self.historical_tree)
        if (self.root / "E0/repro.csv").exists():
            self.check("E0_exact_reproduction_and_LB", self.reproduction)
        self.check("E2_saved_bounds_and_reference_LB", self.bounds_tables)
        self.check("all_available_per_window_LB_and_repeat", self.per_window)
        self.check("repeat_receipts_raw_hashes_and_raw_LB", self.receipts)
        self.check("dispatch_development_fixed_and_EFT_reference_repeats", self.development_dispatch)
        self.check("all_available_search_witnesses_LB_and_certificate_scope", self.certificates)
        self.check("bootstrap_and_campaign_counts", self.statistical_counts)
        self.check("published_latency_tables_from_window_values", self.aggregate_tables)
        if (self.root / "E4/selected_baseline_headroom.csv").exists():
            self.check("new_baseline_headroom_from_same_protocol_windows", self.selected_baseline_headroom)
        if (self.root / "E5/dispatch/regression.csv").exists():
            self.check("single_core_bitexact_regression", self.regression)
        if (self.root / "E4/selected_designs.json").exists():
            self.check("frozen_main_hardware_isoresource", self.selected)
        if (self.root / "E4/robust_heldout/COMPLETE.json").exists():
            self.check("development_frozen_heldout_robust_diagnostics", self.robust_heldout)
        return {"status": "failed" if self.errors else "partial" if missing else "passed",
                "complete_delivery": not missing, "partial_requested": self.partial,
                "missing_files": missing, "checks": self.checks, "errors": self.errors,
                "lower_bound_checked_actual_values": dict(self.lb_counts),
                "lower_bound_checked_total": sum(self.lb_counts.values()),
                "lower_bound_violation_count": len(self.lb_violations), "lower_bound_violations": self.lb_violations,
                "repeat_configuration_receipts": self.repeat_configs, "repeat_digest_pairs": self.repeat_pairs,
                "search_accounting": dict(self.search),
                "scope": "BF16 post-router phase-fluid analytical evidence; not native HBM, RTL, calibrated area, or full-model timing",
                "proof_scope": "Exact resource-assignment solver is not a joint temporal-scheduling optimum. Open hardware domains are reported, never closed by an optimal leaf alone.",
                "inflight_scope": "phase-fluid rate times 65-cycle estimate, not a native discrete-credit occupancy trace"}


def write_summary(result, root=ROOT):
    lines = ["# 第三轮最终验收", "", f"状态：**{result['status']}**。完整交付：{result['complete_delivery']}。", "",
             "本验收检查已保存的逐窗口结果、资源约束、重复运行凭证及证明范围。性能是否跨过 5% 门槛单独报告；如实报告未胜出不等于实验验收失败。", "",
             "| 检查 | 结果 |", "|---|---|",
             *[f"| {x['name']} | {'通过' if x['passed'] else x['error']} |" for x in result["checks"]], "",
             f"实际逐窗口延迟与全局下界比较：{result['lower_bound_checked_total']} 条；下界违例／相关检查失败 {result['lower_bound_violation_count']}。",
             f"重复配置凭证 {result['repeat_configuration_receipts']} 项，匹配摘要对 {result['repeat_digest_pairs']} 对。", "",
             "解析模型的片段流量和在途字节是模型观测／估计，不是 Ramulator 请求级实测；离线精确分配不能代表全部时间调度的全局最优。"]
    if result["missing_files"]:
        lines += ["", "尚未完成的文件：", "", *["- " + n for n in result["missing_files"]]]
    (Path(root) / "VALIDATION_ZH.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="audit currently available files; do not claim final completion")
    ap.add_argument("--wait-timeout", type=float, default=None, help="optional seconds before reporting incomplete delivery")
    args = ap.parse_args()
    started = time.monotonic(); announced = 0
    if not args.partial:
        while True:
            missing = [n for n in REQUIRED if not (ROOT / n).is_file()]
            if not missing or (args.wait_timeout is not None and time.monotonic() - started >= args.wait_timeout):
                break
            if time.monotonic() - announced >= 30:
                print(f"Waiting for complete deliverables: {len(missing)} files; first={missing[0]}", flush=True)
                announced = time.monotonic()
            time.sleep(2)
    result = Audit(partial=args.partial).run()
    write_json(ROOT / "VALIDATION.json", result)
    write_summary(result)
    print(json.dumps({k: result[k] for k in ("status", "complete_delivery", "lower_bound_checked_total", "lower_bound_violation_count")}), flush=True)
    if result["errors"] or (not args.partial and not result["complete_delivery"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
