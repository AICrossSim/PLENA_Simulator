"""Evidence gate tests through a real fallible executable; Python 3.6+ stdlib."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "compare_moe_normal", str(Path(__file__).with_name("compare_moe_normal.py")))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)

FAKE_RUNNER = r'''
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
parser = argparse.ArgumentParser()
for name in ("workload", "architecture", "output", "hbm-channels"):
    parser.add_argument("--" + name, required=True)
args = parser.parse_args()
root = Path(__file__).parent
scenario = json.loads((root / "scenario.json").read_text())
workload = json.loads(Path(args.workload).read_text())
architecture = json.loads(Path(args.architecture).read_text())
name = architecture["name"]
state_path = root / "calls.json"
state_lock = (root / "calls.lock").open('w')
fcntl.flock(state_lock, fcntl.LOCK_EX)
state = json.loads(state_path.read_text()) if state_path.exists() else {}
repeat = state.get(name, 0)
state[name] = repeat + 1
state_path.write_text(json.dumps(state))
state_lock.close()
if scenario.get("exit_code"):
    sys.exit(scenario["exit_code"])
if scenario.get("skip_write"):
    sys.exit(0)
if scenario.get("rewrite_golden"):
    golden_path = root / "golden.json"
    golden = json.loads(golden_path.read_text())
    golden["output_f32"] = [[7.0] * 8, [7.0] * 8]
    golden["output_bf16"] = [[16608] * 8, [16608] * 8]
    golden_path.write_text(json.dumps(golden))
envelope = json.loads((root / "reports.json").read_text())[name]
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
envelope["workload_manifest"] = workload
envelope["architecture_manifest"] = architecture
envelope["provenance"] = {
    "workload_sha256": sha(args.workload), "architecture_sha256": sha(args.architecture),
    "hbm_sha256": sha(Path(args.workload).parent / workload["hbm_file"]),
    "executable_sha256": sha(__file__),
}
envelope["memory_model"]["channels"] = int(args.hbm_channels)
for mutation in scenario.get("mutations", []):
    if mutation.get("architecture", name) != name or mutation.get("repeat", repeat) != repeat:
        continue
    target = envelope
    for key in mutation["path"][:-1]:
        target = target[key]
    target[mutation["path"][-1]] = mutation["value"]
Path(args.output).write_text(json.dumps(envelope))
'''


def core_config(name, blen, mlen):
    return dict(id=name, blen=blen, mlen=mlen, vector_sram_bytes=4096,
                accumulator_bytes=1024, weight_sram_bytes=2048)


def architecture(name, cores):
    return dict(schema_version=1, name=name, cores=cores, dispatch_threshold=2,
                large_core=0, small_core=len(cores) - 1, global_dma_credits=2,
                global_dma_staging_bytes=128, combine_sram_bytes=4096,
                clock_period_ps=10, mac_pipeline_cycles=8, vector_elements_per_cycle=8)


def core_report(core, jobs, rows, total_ps, busy_ps):
    useful = rows * 3 * 8 * 8
    multipliers = core["blen"] * core["mlen"]
    return dict(id=core["id"], blen=core["blen"], mlen=core["mlen"],
                multipliers=multipliers, jobs=jobs, useful_macs=useful,
                issued_macs=useful, compute_busy_ps=busy_ps,
                accumulator_dependency_stall_ps=0, pipeline_drain_ps=10,
                pipeline_register_bytes=64, weight_ready_wait_ps=0,
                vector_wait_ps=0, vector_sram_peak_bytes=256,
                accumulator_peak_bytes=128, weight_sram_peak_bytes=256,
                weight_slots_peak=2, hbm_read_bytes=192 * jobs,
                compute_busy_fraction=busy_ps / total_ps,
                mac_utilization=useful / (multipliers * total_ps / 10))


def envelope(arch):
    single = len(arch["cores"]) == 1
    total = 1000 if single else 800
    jobs = [
        dict(job=0, expert=0, shared=False, core=arch["cores"][0]["id"], rows=2,
             start_ps=0, compute_done_ps=200, output_copied_ps=220),
        dict(job=1, expert=2, shared=True, core=arch["cores"][0]["id"], rows=2,
             start_ps=220, compute_done_ps=420, output_copied_ps=440),
        dict(job=2, expert=1, shared=False, core=arch["cores"][-1]["id"], rows=1,
             start_ps=440 if single else 0, compute_done_ps=600 if single else 180,
             output_copied_ps=620 if single else 200),
    ]
    cores = ([core_report(arch["cores"][0], 3, 5, total, 600)] if single else
             [core_report(arch["cores"][0], 2, 4, total, 600),
              core_report(arch["cores"][1], 1, 1, total, 200)])
    result = dict(schema_version=1, workload="fixture", architecture=arch["name"],
                  timing_model="analytical", timing_boundary="operator",
                  weight_format="local_mx", total_ps=total, multipliers=32,
                  useful_macs=960, issued_macs=960, hbm_read_bytes=576,
                  hbm_write_bytes=0, global_dma_inflight_peak=2,
                  global_dma_staging_peak_bytes=128, combine_sram_peak_bytes=256,
                  shared_vector_busy_ps=100, cores=cores,
                  job_completions=jobs if single else [jobs[2], jobs[0], jobs[1]],
                  output_f32=[[1.0] * 8, [-1.0] * 8],
                  output_bf16=[[16256] * 8, [49024] * 8],
                  pre_round_output_f32=[[1.0] * 8, [-1.0] * 8])
    return dict(schema_version=1, evidence_level="fixture_only", provenance={},
                memory_model=dict(name="Ramulator HBM2 preset", channels=8,
                                  upper_burst_bytes=64), result=result)


class ComparisonEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="plena-moe-gates-", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.binary = self.root / "fake_moe_runner"
        self.binary.write_text("#!" + sys.executable + "\n" + FAKE_RUNNER)
        self.binary.chmod(0o700)
        self.hbm = self.root / "weights.bin"
        self.hbm.write_bytes(bytes(1152))
        self.workload = dict(schema_version=1, name="fixture", hbm_file="weights.bin",
                             input_dim=8, expert_hidden_dim=8,
                             inputs_bf16=[[16256] * 8, [16256] * 8],
                             routes=[dict(token=0, slot=0, expert=0, weight=0.5),
                                     dict(token=0, slot=1, expert=1, weight=0.5),
                                     dict(token=1, slot=0, expert=0, weight=1.0)],
                             shared_expert=dict(expert=2, weight=1.0), metadata={})
        self.workload["experts"] = []
        for expert in range(3):
            record = dict(id=expert)
            for projection, name in enumerate(("gate", "up", "down")):
                base = (expert * 3 + projection) * 128
                record[name] = dict(rows=8, cols=8, element_base=base,
                                    scale_base=base + 64, element_row_stride=8,
                                    scale_row_stride=1)
            self.workload["experts"].append(record)
        self.workload_path = self.root / "workload.json"
        self.write(self.workload_path, self.workload)
        self.architectures = [architecture("single", [core_config("single", 4, 8)]),
                              architecture("dual", [core_config("large", 2, 8),
                                                    core_config("small", 1, 16)])]
        self.arch_paths = [self.root / "single.json", self.root / "dual.json"]
        for path, arch in zip(self.arch_paths, self.architectures):
            self.write(path, arch)
        self.reports = {arch["name"]: envelope(arch) for arch in self.architectures}
        self.write(self.root / "reports.json", self.reports)
        self.golden = dict(schema_version=1,
                           workload_sha256=compare.digest(self.workload_path),
                           hbm_sha256=compare.digest(self.hbm),
                           output_f32=[[1.0] * 8, [-1.0] * 8],
                           output_bf16=[[16256] * 8, [49024] * 8])
        self.golden_path = self.root / "golden.json"
        self.write(self.golden_path, self.golden)
        self.scenario = {}
        self.write(self.root / "scenario.json", self.scenario)
        self.output = self.root / "comparison"

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def run_comparison(self, **kwargs):
        self.write(self.root / "scenario.json", self.scenario)
        return compare.run_comparison(str(self.binary), str(self.workload_path),
                                      str(self.golden_path), self.arch_paths,
                                      self.output, **kwargs)

    def mutate(self, path, value, **filters):
        self.scenario.setdefault("mutations", []).append(dict(path=path, value=value, **filters))

    def rejected(self, **kwargs):
        with self.assertRaises((ValueError, KeyError)):
            self.run_comparison(**kwargs)
        summary = self.output / "comparison.json"
        if summary.exists():
            self.assertIsNot(compare.read_json(summary).get("all_gates_passed"), True)

    def test_complete_deterministic_reports_publish_speedup_and_hashes(self):
        summary = self.run_comparison()
        self.assertTrue(summary["all_gates_passed"])
        self.assertEqual(1.25, summary["comparisons"][1]["speedup_vs_baseline"])
        self.assertEqual(compare.digest(self.golden_path), summary["golden_sha256"])
        self.assertEqual(compare.digest(self.hbm), summary["hbm_sha256"])
        self.assertEqual({"single": 2, "dual": 2}, compare.read_json(self.root / "calls.json"))

    def test_wrong_numeric_result_is_rejected(self):
        self.mutate(["result", "output_f32", 0, 0], 7.0)
        self.rejected()

    def test_parallel_architectures_keep_order_and_repeat_gates(self):
        result = self.run_comparison(workers=2)
        self.assertEqual([r['architecture']['name'] for r in result['comparisons']], ['single','dual'])
        self.assertEqual(compare.read_json(self.root/'calls.json'),{'single':2,'dual':2})
        self.mutate(['result','total_ps'],801,architecture='dual',repeat=3)
        self.rejected(workers=2)

    def test_invalid_parallelism_is_rejected(self):
        for workers in [0,9,True,1.5]:
            with self.subTest(workers=workers): self.rejected(workers=workers)

    def enable_full_shape_resources(self):
        for path, arch in zip(self.arch_paths, self.architectures):
            arch.update(dispatch_queue_bytes=256, dispatch_cycles=1)
            parts = len(arch['cores'])
            for core in arch['cores']:
                core.update(read_cache_bytes=160 // parts, vector_sram_bytes=4096 // parts,
                            accumulator_bytes=1024 // parts, weight_sram_bytes=2048 // parts)
            report = self.reports[arch['name']]['result']
            report.update(dispatch_queue_peak_bytes=192, dispatcher_busy_ps=30)
            for core in report['cores']:
                misses = core['hbm_read_bytes'] // 64
                core.update(cache_requests=misses+2, cache_hits=2,
                            cache_port_busy_ps=(2*misses+2)*10, cache_peak_bytes=80)
            self.write(path, arch)
        self.write(self.root / 'reports.json', self.reports)

    def test_full_shape_resource_accounting_passes(self):
        self.enable_full_shape_resources()
        self.assertTrue(self.run_comparison()['all_gates_passed'])

    def test_cache_capacity_traffic_and_port_undercharging_are_rejected(self):
        self.enable_full_shape_resources()
        for metric,value in [('cache_peak_bytes',240),('cache_requests',0),
                             ('cache_port_busy_ps',0),('cache_hits',-1)]:
            with self.subTest(metric=metric):
                self.scenario = {}
                self.mutate(['result','cores',0,metric],value)
                self.rejected()

    def test_dispatch_capacity_and_service_are_required(self):
        self.enable_full_shape_resources()
        for metric,value in [('dispatch_queue_peak_bytes',0),('dispatcher_busy_ps',0)]:
            with self.subTest(metric=metric):
                self.scenario = {}
                self.mutate(['result',metric],value)
                self.rejected()

    def test_unequal_or_partly_missing_cache_budgets_fail_before_execution(self):
        self.enable_full_shape_resources()
        self.architectures[0]['cores'][0]['read_cache_bytes'] += 80
        self.write(self.arch_paths[0],self.architectures[0])
        self.rejected()
        self.assertFalse((self.root/'calls.json').exists())
        del self.architectures[0]['cores'][0]['read_cache_bytes']
        self.write(self.arch_paths[0],self.architectures[0])
        self.rejected()
        self.assertFalse((self.root/'calls.json').exists())

    def test_unequal_timing_modes_fail_before_execution(self):
        self.architectures[0]['matrix_timing'] = 'legacy_serialized'
        self.write(self.arch_paths[0],self.architectures[0])
        self.rejected()
        self.assertFalse((self.root/'calls.json').exists())

    def test_nonfinite_output_is_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.scenario = {}
                self.mutate(["result", "output_f32", 0, 0], value)
                self.rejected()

    def test_bf16_output_cannot_disagree_with_decoded_float_output(self):
        self.mutate(["result", "output_bf16", 0, 0], 0)
        self.rejected()

    def test_output_shape_must_match_workload_even_if_golden_is_empty(self):
        self.golden.update(output_f32=[], output_bf16=[])
        self.write(self.golden_path, self.golden)
        self.mutate(["result", "output_f32"], [])
        self.mutate(["result", "output_bf16"], [])
        self.rejected()

    def test_every_finite_storage_bound_is_enforced(self):
        cases = [(["global_dma_inflight_peak"], 3),
                 (["global_dma_staging_peak_bytes"], 129),
                 (["combine_sram_peak_bytes"], 4097),
                 (["cores", 0, "vector_sram_peak_bytes"], 4097),
                 (["cores", 0, "accumulator_peak_bytes"], 1025),
                 (["cores", 0, "weight_sram_peak_bytes"], 2049),
                 (["cores", 0, "weight_slots_peak"], 3)]
        for path, value in cases:
            with self.subTest(metric=path):
                self.scenario = {}
                self.mutate(["result"] + path, value)
                self.rejected()

    def test_negative_resource_counter_is_rejected(self):
        self.mutate(["result", "global_dma_staging_peak_bytes"], -64)
        self.rejected()

    def test_useful_and_issued_macs_cannot_omit_work(self):
        for metric in ("useful_macs", "issued_macs"):
            with self.subTest(metric=metric):
                self.scenario = {}
                self.mutate(["result", metric], 959)
                self.rejected()

    def test_per_core_macs_must_close_to_total(self):
        for metric in ("useful_macs", "issued_macs"):
            with self.subTest(metric=metric):
                self.scenario = {}
                self.mutate(["result", "cores", 0, metric], 959)
                self.rejected()

    def test_per_core_hbm_bytes_must_close_to_total(self):
        self.mutate(["result", "cores", 0, "hbm_read_bytes"], 512)
        self.rejected()

    def test_missing_or_duplicate_expert_completions_are_rejected(self):
        original = self.reports["single"]["result"]["job_completions"]
        for jobs in (original[:-1], original + [copy.deepcopy(original[0])]):
            with self.subTest(count=len(jobs)):
                self.scenario = {}
                self.mutate(["result", "job_completions"], jobs, architecture="single")
                self.rejected()

    def test_completed_expert_must_report_all_routed_rows(self):
        self.mutate(["result", "job_completions", 0, "rows"], 1, architecture="single")
        self.rejected()

    def test_completed_job_must_name_a_real_core(self):
        self.mutate(["result", "job_completions", 0, "core"], "missing")
        self.rejected()

    def test_completion_after_total_time_is_rejected(self):
        self.mutate(["result", "job_completions", 0, "output_copied_ps"], 1001)
        self.rejected()

    def test_nonfinite_or_negative_timings_are_rejected(self):
        for metric, value in (("total_ps", float("inf")), ("total_ps", -1),
                              ("shared_vector_busy_ps", float("nan")),
                              ("shared_vector_busy_ps", -1)):
            with self.subTest(metric=metric, value=value):
                self.scenario = {}
                self.mutate(["result", metric], value)
                self.rejected()

    def test_actual_memory_configuration_must_match_requested_hbm(self):
        for field, value in (("channels", 16), ("upper_burst_bytes", 32)):
            with self.subTest(field=field):
                self.scenario = {}
                self.mutate(["memory_model", field], value)
                self.rejected()

    def test_input_or_executable_hash_mismatch_is_rejected(self):
        for field in ("workload_sha256", "architecture_sha256", "hbm_sha256", "executable_sha256"):
            with self.subTest(field=field):
                self.scenario = {}
                self.mutate(["provenance", field], "0" * 64)
                self.rejected()

    def test_golden_must_be_bound_to_exact_workload_and_hbm(self):
        for field in ("workload_sha256", "hbm_sha256"):
            with self.subTest(field=field):
                bad = dict(self.golden)
                bad[field] = "0" * 64
                self.write(self.golden_path, bad)
                self.rejected()
        self.write(self.golden_path, self.golden)

    def test_golden_without_identity_is_rejected(self):
        del self.golden["workload_sha256"]
        self.write(self.golden_path, self.golden)
        self.rejected()

    def test_golden_changed_after_loading_cannot_be_published_as_reference(self):
        self.scenario = dict(rewrite_golden=True)
        self.rejected()

    def test_unequal_multiplier_architectures_are_rejected_before_execution(self):
        self.architectures[1]["cores"][0]["mlen"] = 16
        self.write(self.arch_paths[1], self.architectures[1])
        self.rejected()
        self.assertFalse((self.root / "calls.json").exists())

    def test_different_shared_resources_are_rejected_before_execution(self):
        original = copy.deepcopy(self.architectures[1])
        for field in ("clock_period_ps", "mac_pipeline_cycles", "vector_elements_per_cycle",
                      "global_dma_credits", "global_dma_staging_bytes", "combine_sram_bytes"):
            with self.subTest(field=field):
                altered = copy.deepcopy(original)
                altered[field] += 1
                self.write(self.arch_paths[1], altered)
                self.rejected()
                self.assertFalse((self.root / "calls.json").exists())

    def test_repeat_difference_prevents_speedup_publication(self):
        self.mutate(["result", "total_ps"], 1001, architecture="single", repeat=1)
        self.rejected()

    def test_failed_rerun_cannot_leave_previous_success_summary(self):
        self.run_comparison()
        self.mutate(["result", "output_f32", 0, 0], 7.0)
        self.rejected()

    def test_success_exit_without_new_report_cannot_reuse_previous_reports(self):
        self.run_comparison()
        self.scenario = dict(skip_write=True)
        with self.assertRaises((ValueError, OSError, KeyError)):
            self.run_comparison()
        summary = self.output / "comparison.json"
        if summary.exists():
            self.assertIsNot(compare.read_json(summary).get("all_gates_passed"), True)

    def test_invalid_run_controls_are_rejected(self):
        for kwargs in (dict(repeats=1), dict(repeats=2.5), dict(atol=-1.0),
                       dict(rtol=float("nan")), dict(hbm_channels=0),
                       dict(hbm_channels=3), dict(timeout=0), dict(timeout=float("inf"))):
            with self.subTest(kwargs=kwargs):
                self.rejected(**kwargs)


if __name__ == "__main__":
    unittest.main()
