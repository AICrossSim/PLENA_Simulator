from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from .counter_report import METRICS, aggregate_kernels, parse_kineto, parse_ncu, report, routed_gemm_rows

NVTX = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"


def fixture(rows=None, duration_unit="ns", omit=None):
    columns = ["ID", "Kernel Name", NVTX] + [c for c in METRICS.values() if c != omit]
    units = ["", "", ""] + [(duration_unit if c == "gpu__time_duration.sum" else
                                "byte" if "bytes" in c else "%") for c in columns[3:]]
    buf = io.StringIO()
    buf.write("==PROF== Connected\n==WARNING== clock note\n")
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(columns)
    writer.writerow(units)
    for i, values in enumerate(rows or [("shared", "1,000", 2000, 1000, 50, 10, 30, 40),
                                        ("shared", "3,000", 6000, 3000, 90, 30, 50, 60)]):
        stage, duration, read, write, dram, tensor, sm, occ = values
        metrics = dict(zip(METRICS.values(), (duration, read, write, dram, tensor, sm, occ)))
        writer.writerow([str(i), f"kernel_{i}", f'1 "<default domain>:measured_moe:none" "<default domain>:{stage}:none"'] +
                        [metrics[col] for col in columns[3:]])
    return buf.getvalue()


class CounterReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
        return path

    def test_wide_csv_units_stage_commas(self):
        rows = parse_ncu(self.write("test.csv", fixture()))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["duration_ns"], 1000)
        self.assertEqual(rows[1]["stage"], "shared")
        self.assertEqual(rows[0]["dram_read_bytes"], 2000)

    def test_duration_conversion(self):
        rows = parse_ncu(self.write("test.csv", fixture(duration_unit="us")))
        self.assertEqual(rows[0]["duration_ns"], 1e6)

    def test_duration_weighted_aggregation(self):
        rows = parse_ncu(self.write("test.csv", fixture()))
        out = aggregate_kernels(rows, 2, "vllm_grouped", "win")
        self.assertEqual(out[0]["kernel_count"], 2)
        self.assertEqual(out[0]["serialized_kernel_sum_us"], 4)
        self.assertEqual(out[0]["dram_util_pct"], 80)
        self.assertEqual(out[0]["tensor_pipe_util_pct"], 25)
        self.assertEqual(out[0]["dram_read_write_GBps"], 3)
        self.assertEqual(out[-1]["stage"], "all_kernel_sum")

    def test_missing_required_metric_fails(self):
        path = self.write("test.csv", fixture(omit="dram__bytes_read.sum"))
        with self.assertRaisesRegex(ValueError, "Required counters"):
            parse_ncu(path)

    def test_missing_or_error_counter_fails(self):
        for value in ("n/a", "", "nan", "-1", "inf", "<error>"):
            text = fixture(rows=[("hot", value, 100, 0, 30, 0, 20, 10)])
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_ncu(self.write("test.csv", text))

    def test_no_kernel_or_header_fails(self):
        for text in ("", "==PROF== Error\n", fixture().split('"0","kernel_0"')[0]):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                parse_ncu(self.write("test.csv", text))

    def test_unrecognized_stage_fails(self):
        with self.assertRaisesRegex(ValueError, "NVTX stage"):
            parse_ncu(self.write("test.csv", fixture(rows=[("unknown", 100, 100, 0, 10, 0, 10, 10)])))

    def test_invalid_unit_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported duration"):
            parse_ncu(self.write("test.csv", fixture(duration_unit="cycle")))

    def test_kineto_launch_runtime_driver_not_added(self):
        trace = {"traceEvents": [
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 3.0},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 9.0},
            {"ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernelEx", "dur": 5.0},
            {"ph": "X", "cat": "kernel", "name": "cudaLaunchKernel", "dur": 100.0},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaDeviceSynchronize", "dur": 1000.0},
        ]}
        rows = parse_kineto(self.write("trace.json", json.dumps(trace)), iterations=3)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["launch_calls_total"], 2)
        self.assertEqual(rows[0]["launch_api_sum_us_per_layer"], 4)
        self.assertEqual(rows[0]["launch_api_mean_us"], 6)
        self.assertTrue(rows[0]["preferred_nonduplicated_category"])
        self.assertFalse(rows[1]["preferred_nonduplicated_category"])

    def test_gemm_phase_order_and_missing_fails(self):
        rows = parse_ncu(self.write("test.csv", fixture(rows=[
            ("routed", 1000, 100, 0, 10, 5, 20, 10),
            ("routed", 2000, 200, 0, 15, 6, 22, 11)])))
        for row in rows:
            row["kernel_name"] = "fused_moe_kernel"
        output = routed_gemm_rows(list(reversed(rows)), 2, "vllm_grouped", "win")
        self.assertEqual([r["phase"] for r in output], ["Gate_Up", "Down"])
        self.assertEqual(output[0]["replay_duration_us"], 1)
        with self.assertRaisesRegex(ValueError, "two unquantized"):
            routed_gemm_rows(rows[:1], 2, "vllm_grouped")

    def test_empty_kineto_fails(self):
        with self.assertRaises(ValueError):
            parse_kineto(self.write("trace.json", json.dumps({"traceEvents": []})))

    def test_partial_explicit_and_full_missing_fails(self):
        self.write("B2_vllm_grouped.csv", fixture())
        with self.assertRaisesRegex(ValueError, "campaign incomplete|provenance missing"):
            report(self.root, self.root / "strict")
        audit = report(self.root, self.root / "partial", allow_partial=True)
        self.assertEqual(audit["status"], "PARTIAL")
        self.assertEqual(audit["ncu_points"], 1)
        self.assertTrue((self.root / "partial/GPU_COUNTERS.md").exists())
        self.assertTrue((self.root / "partial/counter_summary.csv").exists())

    def test_real_b200_sample_if_available(self):
        path = Path('/scratch/shared/mcl123/plena/outputs/gpu_hotcold_20261013/B2_vllm_grouped.csv')
        if not path.exists():
            self.skipTest("real B200 sample unavailable")
        rows = parse_ncu(path)
        self.assertGreater(len(rows), 5)
        self.assertEqual(set(r['stage'] for r in rows), {'shared', 'routed', 'combine'})
        self.assertGreater(sum(r['dram_read_bytes'] for r in rows), 33 * 1024 ** 2)


if __name__ == '__main__':
    unittest.main()
