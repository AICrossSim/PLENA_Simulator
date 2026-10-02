"""Finite-memory large legacy windows; numerical row/route order and timing.

These are constructed boundary tests, not pretrained quality measurements.
The numerical check covers the actual global SRAM row aliases and per-chunk
legacy reduction order; it does not turn the timing engine into a payload ISA.
"""
import importlib.util
import json
import subprocess
import tempfile
import unittest

import numpy as np
from frontend import COMPILER_DIR, compiler, default_binary
from numerical import bf16, bf16_bits, from_bf16, reference_expert

spec = importlib.util.spec_from_file_location("plena_chunk_boundary_planner", COMPILER_DIR / "legacy_chunks.py")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def workload(batch=400, hidden=513, f=19):
    experts = []
    for id in (-1, 0, 1):
        weights = {}
        for p, (phase, n, k) in enumerate((("gate", f, hidden), ("up", f, hidden), ("down", hidden, f))):
            weights[phase] = dict(shape_nk=[n, k], hbm_base=(id + 1) * 3 * 16777216 + p * 16777216,
                row_stride_bytes=compiler.align(k * 2), physical_bytes=n * compiler.align(k * 2))
        experts.append(dict(id=id, is_shared=id == -1, Me=batch, H=hidden, F=f,
            token_indices=list(range(batch)), route_slots=[-1 if id == -1 else id] * batch,
            route_scores=[1.0 if id == -1 else (.75 if id == 0 else .25)] * batch, weights=weights))
    return dict(id="synthetic_legacy_capacity_rows", batch=batch, hidden=hidden, top_k=2,
        tokens=[dict(token_index=t, sample_id="constructed_%d" % t,
            routes=[dict(expert_id=e, slot=e, score=.75 if e == 0 else .25) for e in (0, 1)]) for t in range(batch)],
        experts=experts)


class LegacyChunkBoundaryTests(unittest.TestCase):
    def run_timing(self, prepared, lanes):
        cfg = dict(arch="joint_v1", lanes=lanes, group=4, dispatch="fifo", split="none",
            runtime_fsm=True, arbiter="stock", record_trace=False, diagnostic_profile=True)
        with tempfile.TemporaryDirectory(prefix="plena-legacy-chunk-boundary-") as tmp:
            from pathlib import Path
            tmp = Path(tmp)
            (tmp / "workload.json").write_text(json.dumps(prepared))
            (tmp / "config.json").write_text(json.dumps(cfg))
            raws = []
            for repeat in (1, 2):
                dest = tmp / ("repeat%d.json" % repeat)
                p = subprocess.run([str(default_binary()), str(tmp / "workload.json"), str(tmp / "config.json"), str(dest)],
                    capture_output=True, timeout=120)
                self.assertEqual(p.returncode, 0, p.stderr.decode()[-6000:])
                raws.append(dest.read_bytes())
            self.assertEqual(*raws)
            return json.loads(raws[0])

    def test_real_kernel_chunks_preserve_work_drain_and_report_refetch(self):
        for lanes in ([6], [4, 2]):
            with self.subTest(lanes=lanes):
                w = workload()
                prepared = planner.prepare_legacy_workload(w, lanes)
                plan = prepared["legacy_batch_execution"]
                r = self.run_timing(prepared, lanes)
                parts = r["legacy_chunks"]["parts"]
                self.assertTrue(r["drained"] and r["ownership_k_order_capacity_checks"])
                self.assertTrue(r["legacy_chunks"]["global_routes_and_scores_preserved"])
                self.assertEqual(r["cycles"], sum(p["setup_cycles"] + p["kernel_report"]["cycles"] for p in parts))
                self.assertEqual(r["useful_macs"], plan["expected_useful_macs"])
                self.assertEqual(r["weight_bytes"], plan["weight_read_bytes"])
                self.assertEqual(r["refetch_bytes"], r["weight_bytes"] - plan["unique_weight_bytes"])
                self.assertGreater(r["refetch_bytes"], 0)
                self.assertEqual(r["dma_transactions_accepted"], r["dma_transactions_landed"])
                self.assertEqual(r["weight_bytes"], r["dma_transactions_accepted"] * 32)
                self.assertEqual(r["cycles"] - r["experts_done_cycles"], r["combine_tail_cycles"])
                self.assertEqual(r["total_chunk_combine_cycles"], sum(p["kernel_report"]["combine_tail_cycles"] for p in parts))
                self.assertTrue(r["m0_profile"]["mutually_exclusive"])
                self.assertEqual(r["m0_profile"]["hbm_sum"], r["cycles"])
                self.assertEqual(r["m0_profile"]["core_sums"], [r["cycles"]] * len(lanes))
                for c, core in enumerate(r["cores"]):
                    self.assertLessEqual(core["stats"]["workspace_peak_bytes"], core["capacity"])
                    self.assertEqual(core["stats"]["issues"], sum(p["kernel_report"]["cores"][c]["stats"]["issues"] for p in parts))
                    self.assertEqual(core["stats"]["weight_peak_bytes"], max(p["kernel_report"]["cores"][c]["stats"]["weight_peak_bytes"] for p in parts))

    def test_sram_row_aliases_preserve_bf16_outputs_and_route_sum_order(self):
        w = workload()
        prepared = planner.prepare_legacy_workload(w, [6])
        plan = prepared["legacy_batch_execution"]
        persistent = plan["persistent_cores"][0]
        rng = np.random.default_rng(20261002)
        x = bf16(rng.normal(0, .1, (w["batch"], w["hidden"])))
        tensors = {e["id"]: tuple(bf16(rng.normal(0, .1, (n, k))) for n, k in
            ((w["experts"][0]["F"], w["hidden"]), (w["experts"][0]["F"], w["hidden"]), (w["hidden"], w["experts"][0]["F"]))) for e in w["experts"]}
        outputs = {e["id"]: reference_expert(x, *tensors[e["id"]]) for e in w["experts"]}
        expected = bf16(np.add(bf16(np.add(np.multiply(outputs[0], np.float32(.75), dtype=np.float32),
            np.multiply(outputs[1], np.float32(.25), dtype=np.float32), dtype=np.float32)), outputs[-1], dtype=np.float32))
        arena = bytearray(persistent["capacity"])
        xb = persistent["original_x"]["address_bytes"]
        arena[xb:xb + x.size * 2] = bf16_bits(x).astype('<u2').tobytes()
        original_bytes = bytes(arena[xb:xb + x.size * 2])
        yb = persistent["combined_output"]["address_bytes"]
        for part in plan["chunks"]:
            start, end = part["token_range"]
            layout = part["workload"]["engine_layout"]
            result = layout["cores"][0]["result_layout"]
            xv = result["original_x"]
            local_x = from_bf16(np.frombuffer(arena, dtype='<u2', count=(end - start) * w["hidden"], offset=xv["address_bytes"]).copy().reshape(end-start, w["hidden"]))
            values = {}
            for e, inbox in zip(part["workload"]["experts"], result["inboxes"]):
                y = reference_expert(local_x[e["token_indices"]], *tensors[e["id"]])
                a = inbox["address_bytes"]
                self.assertGreaterEqual(a, persistent["reserved"])
                self.assertLessEqual(a + y.size * 4, layout["cores"][0]["reserved"])
                arena[a:a + y.size * 4] = y.astype('<f4').tobytes()
                values[e["id"]] = np.frombuffer(arena, dtype='<f4', count=y.size, offset=a).copy().reshape(y.shape)
            combined = bf16(np.add(bf16(np.add(np.multiply(values[0], np.float32(.75), dtype=np.float32),
                np.multiply(values[1], np.float32(.25), dtype=np.float32), dtype=np.float32)), values[-1], dtype=np.float32))
            dest = result["combined_output"]["address_bytes"]
            self.assertEqual(dest, yb + start * w["hidden"] * 4)
            arena[dest:dest + combined.size * 4] = combined.astype('<f4').tobytes()
            self.assertEqual(bytes(arena[xb:xb + x.size * 2]), original_bytes)
        actual = np.frombuffer(arena, dtype='<f4', count=expected.size, offset=yb).reshape(expected.shape)
        np.testing.assert_array_equal(bf16_bits(actual), bf16_bits(expected))


if __name__ == "__main__":
    unittest.main()
