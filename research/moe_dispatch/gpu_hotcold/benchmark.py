"""CUDA-event profiling of real MoE routes with synthetic BF16 tensor values.

The timed scope starts after router IDs/weights are known. vLLM is the normal
production-library baseline. Triton uses host-precomputed expert tile maps and
is explicitly a trace-specialized prototype. Splitting Hot/Cold changes its
execution plan; split timings never attribute the unsplit vLLM/grouped runtime.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from . import kernels
from .traces import load_routes

H, ROUTED_F, SHARED_F, EXPERTS, TOPK = 2048, 1408, 2816, 64, 6
PATHS = ("vllm_grouped", "vllm_split", "triton_grouped", "triton_split")


def summary(values):
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("Cannot summarize an empty timing sample")
    return {"mean": statistics.fmean(vals), "median": statistics.median(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals)}


@dataclass
class Group:
    rows: torch.Tensor
    tile_experts: torch.Tensor
    tile_mstarts: torch.Tensor
    experts: list[int]


class RoutePlan:
    """Host grouping excluded from timing; runtime GPU gather stays included."""
    def __init__(self, window, device="cuda"):
        self.window = window
        self.batch = int(window["batch"])
        ids = window["route_ids"]
        scores = window["route_weights"]
        if len(ids) != self.batch or len(scores) != self.batch:
            raise ValueError("Batch/route row mismatch")
        entries = [[] for _ in range(EXPERTS)]
        for token, route in enumerate(ids):
            if len(route) != TOPK or len(set(route)) != TOPK:
                raise ValueError("Every token must route to six distinct experts")
            if len(scores[token]) != TOPK or any(not math.isfinite(float(w)) for w in scores[token]):
                raise ValueError("Invalid route scores")
            for slot, e in enumerate(route):
                if not 0 <= int(e) < EXPERTS:
                    raise ValueError("Expert ID outside [0,64)")
                entries[int(e)].append((token, slot))
        self.counts = [len(v) for v in entries]
        if "counts_64" in window and self.counts != list(window["counts_64"]):
            raise ValueError("Captured counts do not match exact route IDs")
        starts, tokens, inverse = [0], [], [[0] * TOPK for _ in ids]
        for e, rows in enumerate(entries):
            for token, slot in rows:
                inverse[token][slot] = len(tokens)
                tokens.append(token)
            starts.append(len(tokens))
        assert sum(self.counts) == TOPK * self.batch
        make = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
        self.row_starts = make(starts)
        self.token_ids = make(tokens)
        self.pack_indices = make(inverse)
        self.route_ids = make(ids)
        self.route_weights = torch.tensor(scores, dtype=torch.float32, device=device)
        self.groups = {}
        for kind in ("all", "hot", "cold"):
            es = [e for e, m in enumerate(self.counts) if m > 0 and
                  (kind == "all" or (kind == "hot" and m > 2) or
                   (kind == "cold" and m <= 2))]
            rows, te, tm = [], [], []
            for e in es:
                rows.extend(range(starts[e], starts[e + 1]))
                for m in range(0, self.counts[e], 16):
                    te.append(e)
                    tm.append(m)
            self.groups[kind] = Group(make(rows), make(te), make(tm), es)
        self.all, self.hot, self.cold = (self.groups[k] for k in ("all", "hot", "cold"))


class BenchmarkContext:
    def __init__(self, device="cuda:0", seed=1729, enable_vllm=True):
        if not torch.cuda.is_available():
            raise RuntimeError("A real CUDA GPU is required; CPU emulation is prohibited")
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")
        self.seed = seed
        gen = torch.Generator(device=self.device).manual_seed(seed)
        def weight(shape, reduction):
            return torch.empty(shape, dtype=torch.bfloat16, device=self.device).normal_(
                generator=gen).mul_(1 / math.sqrt(reduction))
        self.shared_gu = weight((H, 2 * SHARED_F), H)
        self.shared_down = weight((SHARED_F, H), SHARED_F)
        self.routed_gu = weight((EXPERTS, H, 2 * ROUTED_F), H)
        self.routed_down = weight((EXPERTS, ROUTED_F, H), ROUTED_F)
        self.fused_experts = None
        if enable_vllm:
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
            self.fused_experts = fused_experts
            self.vllm_gu = self.routed_gu.transpose(1, 2).contiguous()
            self.vllm_down = self.routed_down.transpose(1, 2).contiguous()
        torch.cuda.synchronize(self.device)

    def metadata(self):
        p = torch.cuda.get_device_properties(self.device)
        versions = {}
        for name in ("torch", "triton", "vllm"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return {"gpu": p.name, "device": str(self.device), "compute_capability": [p.major, p.minor],
                "total_memory_bytes": p.total_memory, "sm_count": p.multi_processor_count,
                "cuda_runtime": torch.version.cuda, "software": versions,
                "python": platform.python_version(), "tensor_seed": self.seed,
                "precision": "BF16 inputs/weights; FP32 matmul accumulation",
                "weight_source": "synthetic Gaussian BF16 values, not pretrained checkpoint",
                "routing_source": "captured token IDs, six expert IDs and six scores per token",
                "scope": "post-router MoE, GPU gather/GEMMs/SiLU/weighted combine included",
                "cache_policy": "repeated full-layer warm execution; no between-component cache flush",
                "prototype_specialization": "Triton expert grouping and tile maps prepared on host outside timing",
                "production_baseline": "vLLM fused_experts includes its runtime expert alignment",
                "vllm_split_contract": "expert_map sets excluded experts to -1; matmul outputs zero; full-route alignment and activation repeat per subset; split is a controlled ablation, not normal-kernel attribution",
                "hardware_counters": "not available from CUDA Events; separate Nsight run required",
                "synthetic_bf16_weight_bytes": sum(t.numel() * t.element_size() for t in
                    [self.shared_gu, self.shared_down, self.routed_gu, self.routed_down])}


class WindowRunner:
    def __init__(self, context: BenchmarkContext, window):
        self.context, self.window = context, window
        self.plan = RoutePlan(window, context.device)
        b = self.plan.batch
        seed = context.seed + int(hashlib.sha256(window["id"].encode()).hexdigest()[:8], 16)
        gen = torch.Generator(device=context.device).manual_seed(seed)
        self.x = torch.randn((b, H), dtype=torch.bfloat16, device=context.device, generator=gen)
        empty = lambda shape: torch.empty(shape, dtype=torch.bfloat16, device=context.device)
        self.shared_gu = empty((b, 2 * SHARED_F))
        self.shared_z = empty((b, SHARED_F))
        self.shared_y = empty((b, H))
        self.shared_rows = torch.arange(b, dtype=torch.int32, device=context.device)
        self.packed_x = empty((b * TOPK, H))
        self.routed_gu = empty((b * TOPK, 2 * ROUTED_F))
        self.routed_z = empty((b * TOPK, ROUTED_F))
        self.routed_y = empty((b * TOPK, H))
        self.output = empty((b, H))
        self.expected_output = None
        self.vllm_y = None
        self.vllm_subset_y = {"hot": empty((b, H)), "cold": empty((b, H))}
        self.vllm_expert_maps = {
            kind: torch.tensor([e if e in self.plan.groups[kind].experts else -1
                                for e in range(EXPERTS)], dtype=torch.int32,
                               device=context.device)
            for kind in ("hot", "cold")}

    def shared(self):
        torch.mm(self.x, self.context.shared_gu, out=self.shared_gu)
        kernels.silu_multiply_rows(self.shared_gu, self.shared_rows, self.shared_z)
        torch.mm(self.shared_z, self.context.shared_down, out=self.shared_y)

    def routed(self, kind="all"):
        g = self.plan.groups[kind]
        if not g.experts:
            return
        kernels.gather_rows(self.x, self.plan.token_ids, g.rows, self.packed_x)
        kernels.grouped_matmul(self.packed_x, self.context.routed_gu, self.routed_gu,
                               self.plan.row_starts, g.tile_experts, g.tile_mstarts)
        kernels.silu_multiply_rows(self.routed_gu, g.rows, self.routed_z)
        kernels.grouped_matmul(self.routed_z, self.context.routed_down, self.routed_y,
                               self.plan.row_starts, g.tile_experts, g.tile_mstarts)

    def vllm_routed(self):
        if self.context.fused_experts is None:
            raise RuntimeError("vLLM backend was not loaded")
        self.vllm_y = self.context.fused_experts(
            self.x, self.context.vllm_gu, self.context.vllm_down,
            self.plan.route_weights, self.plan.route_ids, inplace=False, activation="silu")

    def vllm_routed_subset(self, kind):
        # vLLM's expert-parallel mapping explicitly zeroes excluded experts.
        # Alignment/activation still span the original route slots: this is a
        # controlled split ablation, not a disjoint attribution of normal vLLM.
        self.vllm_subset_y[kind] = self.context.fused_experts(
            self.x, self.context.vllm_gu, self.context.vllm_down,
            self.plan.route_weights, self.plan.route_ids, inplace=False,
            activation="silu", global_num_experts=EXPERTS,
            expert_map=self.vllm_expert_maps[kind])

    def vllm_split_combine(self):
        kernels.combine_subsets(self.shared_y, self.vllm_subset_y["hot"],
                                self.vllm_subset_y["cold"], self.output,
                                bool(self.plan.hot.experts), bool(self.plan.cold.experts))

    def combine(self):
        kernels.combine_routed(self.routed_y, self.plan.pack_indices, self.plan.route_weights,
                               self.shared_y, self.output)

    def vllm_combine(self):
        torch.add(self.shared_y, self.vllm_y, out=self.output)

    def get_stages(self, path):
        """(name, callable, active); inactive categories are exactly zero time."""
        if path == "vllm_grouped":
            return [("shared", self.shared, True), ("routed", self.vllm_routed, True),
                    ("combine", self.vllm_combine, True)]
        if path == "vllm_split":
            return [("shared", self.shared, True),
                    ("hot", lambda: self.vllm_routed_subset("hot"), bool(self.plan.hot.experts)),
                    ("cold", lambda: self.vllm_routed_subset("cold"), bool(self.plan.cold.experts)),
                    ("combine", self.vllm_split_combine, True)]
        if path == "triton_grouped":
            return [("shared", self.shared, True), ("routed", self.routed, True),
                    ("combine", self.combine, True)]
        if path == "triton_split":
            return [("shared", self.shared, True),
                    ("hot", lambda: self.routed("hot"), bool(self.plan.hot.experts)),
                    ("cold", lambda: self.routed("cold"), bool(self.plan.cold.experts)),
                    ("combine", self.combine, True)]
        raise ValueError(f"Unknown path: {path}")

    def run(self, path):
        for _, fn, active in self.get_stages(path):
            if active:
                fn()
        return self.output

    def reference(self):
        def ffn(x, gu, down):
            z = torch.mm(x, gu)
            gate, up = z.chunk(2, dim=1)
            # Match the fused pointwise FP32 compute -> BF16 store contract.
            z = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
            return torch.mm(z, down)
        c = self.context
        y = ffn(self.x, c.shared_gu, c.shared_down).float()
        for e in self.plan.all.experts:
            pairs = [(t, s) for t, ids in enumerate(self.window["route_ids"])
                     for s, eid in enumerate(ids) if eid == e]
            idx = torch.tensor([t for t, _ in pairs], dtype=torch.long, device=c.device)
            score = torch.tensor([self.window["route_weights"][t][s] for t, s in pairs],
                                 dtype=torch.float32, device=c.device)
            result = ffn(self.x[idx], c.routed_gu[e], c.routed_down[e]).float()
            y.index_add_(0, idx, result * score[:, None])
        return y.to(torch.bfloat16)

    def check_correctness(self, paths):
        ref = self.reference()
        self.expected_output = ref
        rows = []
        for path in paths:
            actual = self.run(path).clone()
            torch.cuda.synchronize(self.context.device)
            if not torch.isfinite(actual).all():
                raise AssertionError(f"Non-finite output: {self.window['id']} {path}")
            err = actual.float() - ref.float()
            # Different valid BF16 fused implementations need not be bitwise identical.
            torch.testing.assert_close(actual, ref, rtol=0.05, atol=0.025)
            rows.append({"window_id": self.window["id"], "batch": self.plan.batch, "path": path,
                         "max_abs_error": err.abs().max().item(),
                         "relative_l2_error": (err.norm() / ref.float().norm().clamp_min(1e-12)).item(),
                         "rtol": 0.05, "atol": 0.025, "passed": True})
        return rows


def benchmark_window(runner, path, mode, warmup=10, samples=50):
    if warmup < 10 or samples < 50:
        raise ValueError("The measurement contract requires >=10 warmups and >=50 samples")
    stages = runner.get_stages(path)
    def plain():
        runner.run(path)
    # Compile all kernels and initialize cuBLAS/vLLM outside capture and timing.
    for _ in range(warmup):
        plain()
    torch.cuda.synchronize(runner.context.device)
    graph = None
    if mode == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plain()
        execute = graph.replay
    elif mode == "eager":
        execute = plain
    else:
        raise ValueError(mode)
    for _ in range(warmup):
        execute()
    torch.cuda.synchronize(runner.context.device)
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    e2e = []
    for _ in range(samples):
        begin.record()
        execute()
        end.record()
        end.synchronize()
        e2e.append(begin.elapsed_time(end))
    raw_output_check = None
    if runner.expected_output is not None:
        torch.testing.assert_close(runner.output, runner.expected_output, rtol=0.05, atol=0.025)
        raw_output_check = True
    # Events are boundaries of ONE complete invocation, never isolated stage loops.
    active = [(n, f) for n, f, yes in stages if yes]
    events = [torch.cuda.Event(enable_timing=True, external=(mode == "graph"))
              for _ in range(len(active) + 1)]
    def instrumented():
        events[0].record()
        for i, (_, fn) in enumerate(active):
            fn()
            events[i + 1].record()
    instrumented_graph = None
    if mode == "graph":
        instrumented_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(instrumented_graph):
            instrumented()
        execute_instrumented = instrumented_graph.replay
    else:
        execute_instrumented = instrumented
    for _ in range(warmup):
        execute_instrumented()
    torch.cuda.synchronize(runner.context.device)
    stage_samples = {name: [] for name, _, _ in stages}
    totals = []
    for _ in range(samples):
        execute_instrumented()
        events[-1].synchronize()
        measured = {name: events[i].elapsed_time(events[i + 1])
                    for i, (name, _) in enumerate(active)}
        for name in stage_samples:
            stage_samples[name].append(measured.get(name, 0.0))
        totals.append(events[0].elapsed_time(events[-1]))
    instrumented_output_check = None
    if runner.expected_output is not None:
        torch.testing.assert_close(runner.output, runner.expected_output, rtol=0.05, atol=0.025)
        instrumented_output_check = True
    closure_error = max(abs(sum(stage_samples[k][i] for k in stage_samples) - totals[i])
                        for i in range(samples))
    # Event float rounding is small; no negative timings or unaccounted gaps allowed.
    if closure_error > max(0.001, max(totals) * 1e-4):
        raise AssertionError(f"Timing accounting does not close: {closure_error} ms")
    output = {"window_id": runner.window["id"], "batch": runner.plan.batch,
              "mode": mode, "path": path, "warmup": warmup, "samples": samples,
              "hot_experts": len(runner.plan.hot.experts), "cold_experts": len(runner.plan.cold.experts),
              "hot_token_assignments": sum(m for m in runner.plan.counts if m > 2),
              "cold_token_assignments": sum(m for m in runner.plan.counts if 0 < m <= 2),
              "e2e_ms": e2e, "instrumented_total_ms": totals, "stage_samples_ms": stage_samples,
              "e2e_stats_ms": summary(e2e), "instrumented_stats_ms": summary(totals),
              "stage_stats_ms": {k: summary(v) for k, v in stage_samples.items()},
              "instrumentation_delta_pct": (statistics.fmean(totals) / statistics.fmean(e2e) - 1) * 100,
              "accounting_max_abs_error_ms": closure_error,
              "raw_output_correct": raw_output_check,
              "instrumented_output_correct": instrumented_output_check,
              "category_split_is_controlled_ablation": path.endswith("_split")}
    return output


def capture_torch_profiler(runner, path, output):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, profile_memory=False) as prof:
        for _ in range(3):
            runner.run(path)
        torch.cuda.synchronize(runner.context.device)
    prof.export_chrome_trace(str(output.with_suffix(".json")))
    items = []
    for item in prof.key_averages():
        items.append({"name": item.key, "count": item.count,
                      "cpu_total_us": item.cpu_time_total,
                      "device_total_us": getattr(item, "device_time_total", 0),
                      "self_device_us": getattr(item, "self_device_time_total", 0)})
    output.with_suffix(".summary.json").write_text(json.dumps({
        "window_id": runner.window["id"], "path": path, "iterations": 3,
        "scope": "separate eager profiler run, not timing sample; not occupancy or bandwidth counters",
        "events": items}, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--routes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--order-seed", type=int, default=20261013,
                   help="Shuffles window/mode/path order only; numerical seed stays fixed")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--modes", default="eager,graph")
    p.add_argument("--paths", default=",".join(PATHS))
    p.add_argument("--batches", default="2,4,8,16,64,96,128")
    p.add_argument("--max-windows", type=int)
    p.add_argument("--correctness", choices=["all", "representative"], default="all")
    p.add_argument("--torch-profiler", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and (args.output / "measurements.jsonl").exists():
        raise FileExistsError("Use a new output directory; existing measurements are immutable")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_routes(args.routes)
    windows = data["windows"]
    batches = set(map(int, args.batches.split(",")))
    windows = [w for w in windows if int(w["batch"]) in batches]
    if args.max_windows:
        windows = windows[:args.max_windows]
    if not windows:
        raise ValueError("No matching captured windows")
    order_rng = random.Random(args.order_seed)
    order_rng.shuffle(windows)
    modes, paths = args.modes.split(","), args.paths.split(",")
    if any(p not in PATHS for p in paths) or any(m not in ("eager", "graph") for m in modes):
        raise ValueError("Unsupported path or mode")
    if args.warmup < 10 or args.samples < 50:
        raise ValueError("At least 10 warmup iterations and 50 samples are required")
    ctx = BenchmarkContext(args.device, args.seed, any(p.startswith("vllm_") for p in paths))
    metadata = ctx.metadata()
    metadata.update({"command": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "routes_sha256": hashlib.sha256(args.routes.read_bytes()).hexdigest(),
                     "window_count": len(windows), "window_order": [w["id"] for w in windows],
                     "route_metadata": data.get("metadata", {})})
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    correctness, seen, rows = [], set(), []
    with (args.output / "measurements.jsonl").open("w") as f, torch.inference_mode():
        for index, window in enumerate(windows):
            runner = WindowRunner(ctx, window)
            if args.correctness == "all" or runner.plan.batch not in seen:
                correctness.extend(runner.check_correctness(paths))
            measurements = [(mode, path) for mode in modes for path in paths]
            order_rng.shuffle(measurements)
            for mode, path in measurements:
                row = benchmark_window(runner, path, mode, args.warmup, args.samples)
                row["execution_index"] = len(rows)
                row["order_seed"] = args.order_seed
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
                print(json.dumps({"progress": f"{index+1}/{len(windows)}", "window": window["id"],
                                  "mode": mode, "path": path,
                                  "mean_ms": row["e2e_stats_ms"]["mean"],
                                  "median_ms": row["e2e_stats_ms"]["median"]}), flush=True)
            if args.torch_profiler and runner.plan.batch not in seen:
                folder = args.output / "torch_profiler"
                folder.mkdir(exist_ok=True)
                for path in paths:
                    capture_torch_profiler(runner, path, folder / f"B{runner.plan.batch}_{path}")
            seen.add(runner.plan.batch)
            (args.output / "correctness.json").write_text(json.dumps(correctness, indent=2))
            del runner
    (args.output / "COMPLETE.json").write_text(json.dumps({"status": "PASS", "points": len(rows),
        "windows": len(windows), "correctness_checks": len(correctness),
        "all_finite_positive": all(r["e2e_stats_ms"]["min"] > 0 for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
