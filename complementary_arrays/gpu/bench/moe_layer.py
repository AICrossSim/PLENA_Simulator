"""Time one MoE decode layer on recorded routes with the fused-MoE kernels of vLLM, FlashInfer, TensorRT-LLM and Marlin.

A layer is the router GEMM plus one fused MoE call in which every token also selects the shared experts, appended as
extra experts. Weights are random and kept in one copy per MoE layer of the model, used in turn so that no call finds
them in L2; calls are replayed from a CUDA graph. A window's bytes (router, shared experts and distinct routed experts,
scales included) over the HBM bandwidth give its weight-read floor.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MODELS = {
    "qwen": dict(H=2048, FE=512, E=256, K=8, S=1, layers=40),
    "deepseek": dict(H=2048, FE=1408, E=64, K=6, S=2, layers=26),
}
BATCHES = (2, 4, 8, 16, 32, 64, 96, 128)
KERNELS = ("bf16:triton", "bf16:flashinfer", "bf16:trtllm", "w8g128:triton", "w8g128:marlin", "w4g32:marlin")


def report_windows(traces, model, batch, n):
    """Held-out trace windows of 10 layers spread over the odd-indexed MoE layers and decode steps 16-32, sorted;
    n evenly spaced ones if there are more."""
    with np.load(traces / f"{model}_b{batch}_heldout.npz") as z:
        meta, routes, live = json.loads(str(z["metadata"])), z["routes"], z["live"]
    rows = [dict(r, ids=routes[i, : live[i]]) for i, r in enumerate(meta)]
    odd = set(sorted({r["layer"] for r in rows})[1::2])
    rows = [r for r in rows if r["layer"] in odd and 16 <= r["step"] <= 32]
    layers = sorted({r["layer"] for r in rows})
    keep = {layers[round(i * (len(layers) - 1) / 9)] for i in range(10)}
    rows = sorted(
        (r for r in rows if r["layer"] in keep), key=lambda r: (r["source"], str(r["group"]), r["layer"], r["step"])
    )
    if n and len(rows) > n:
        rows = [rows[i] for i in sorted(set(np.linspace(0, len(rows) - 1, n).round().astype(int).tolist()))]
    return rows


def window_weights(rows):
    """Averaging weights: sources equal, request groups equal within a source, windows equal within a group."""
    w = np.zeros(len(rows))
    src, grp = np.array([r["source"] for r in rows]), np.array([r["group"] for r in rows])
    sources = sorted(set(src))
    for s in sources:
        groups = sorted(set(grp[src == s]))
        for g in groups:
            ix = (src == s) & (grp == g)
            w[ix] = 1 / (len(sources) * len(groups) * ix.sum())
    return w


def padded_experts(m):
    """Expert count with the shared experts, rounded up to a multiple of 4 as the TensorRT-LLM kernels require."""
    return (m["E"] + m["S"] + 3) // 4 * 4


def route_tensors(torch, m, ids):
    """Top-k weights and ids with the shared experts appended, and router logits (+8 on exactly these experts, -8
    elsewhere) for kernels that route inside the call (top-k then softmax)."""
    M = ids.shape[0]
    routed = torch.as_tensor(ids.astype(np.int32), device="cuda")
    shared = torch.arange(m["E"], m["E"] + m["S"], dtype=torch.int32, device="cuda").repeat(M, 1)
    topk_ids = torch.cat([routed, shared], 1).contiguous()
    topk_weights = torch.cat(
        [
            torch.full((M, m["K"]), 1.0 / m["K"], dtype=torch.float32, device="cuda"),
            torch.ones((M, m["S"]), dtype=torch.float32, device="cuda"),
        ],
        1,
    ).contiguous()
    logits = torch.full((M, padded_experts(m)), -8.0, dtype=torch.bfloat16, device="cuda")
    logits.scatter_(1, topk_ids.long(), 8.0)
    return topk_weights, topk_ids, logits


class MoELayer:
    """L copies of one model's router and experts in one weight format, laid out for one kernel."""

    def __init__(self, torch, fmt, impl, m):
        if f"{fmt}:{impl}" not in KERNELS:
            raise ValueError(f"unsupported kernel {fmt}:{impl}")
        self.t, self.fmt, self.impl, self.m, self.L = torch, fmt, impl, m, m["layers"]
        self.experts = padded_experts(m) if impl == "trtllm" else m["E"] + m["S"]
        self.router = torch.empty(self.L, m["E"], m["H"], dtype=torch.bfloat16, device="cuda")
        self.router[:] = torch.randn(m["E"], m["H"], dtype=torch.bfloat16, device="cuda") * 0.02
        self.w = self.build()
        self.expert_bytes = self.w["bytes"] / (self.L * self.experts)
        self.shared_bytes = m["S"] * self.expert_bytes
        self.router_bytes = m["E"] * m["H"] * 2

    def fill(self, shape, dtype, src):
        """[L, *shape] tensor with every copy equal to src."""
        x = self.t.empty((self.L, *shape), dtype=dtype, device="cuda")
        x[:] = src.to(dtype)
        return x

    def build(self):
        t, E, H, FE = self.t, self.experts, self.m["H"], self.m["FE"]
        bf, u8 = t.bfloat16, t.uint8

        def rnd(*s):
            return t.randn(*s, dtype=bf, device="cuda") * 0.02

        if self.impl == "trtllm":  # BlockMajorK [E, K / 64, N, 64], the layout trtllm_bf16_moe takes
            w1 = self.fill((E, H // 64, 2 * FE, 64), bf, rnd(E, H // 64, 2 * FE, 64))
            w2 = self.fill((E, FE // 64, H, 64), bf, rnd(E, FE // 64, H, 64))
            return dict(w1=w1, w2=w2, bytes=w1.nbytes + w2.nbytes)
        if self.fmt == "bf16":
            w1, w2 = self.fill((E, 2 * FE, H), bf, rnd(E, 2 * FE, H)), self.fill((E, H, FE), bf, rnd(E, H, FE))
            return dict(w1=w1, w2=w2, qc=[None] * self.L, bytes=w1.nbytes + w2.nbytes)
        bits, g = (8, 128) if self.fmt == "w8g128" else (4, 32)
        if self.impl == "triton":  # W8A16: uint8 values with BF16 group scales and zero points
            from vllm.model_executor.layers.fused_moe.config import int8_w8a16_moe_quant_config

            def ri(*s):
                return t.randint(0, 256, s, dtype=u8, device="cuda")

            w1, w2 = self.fill((E, 2 * FE, H), u8, ri(E, 2 * FE, H)), self.fill((E, H, FE), u8, ri(E, H, FE))
            s1 = self.fill((E, 2 * FE, H // g), bf, t.rand(E, 2 * FE, H // g, device="cuda") * 1e-2)
            s2 = self.fill((E, H, FE // g), bf, t.rand(E, H, FE // g, device="cuda") * 1e-2)
            z1, z2 = (
                self.fill((E, 2 * FE, H // g), u8, ri(E, 2 * FE, H // g)),
                self.fill((E, H, FE // g), u8, ri(E, H, FE // g)),
            )
            qc = [
                int8_w8a16_moe_quant_config(
                    w1_scale=s1[l], w2_scale=s2[l], w1_zp=z1[l], w2_zp=z2[l], block_shape=[0, g]
                )
                for l in range(self.L)
            ]
            return dict(w1=w1, w2=w2, qc=qc, bytes=sum(x.nbytes for x in (w1, w2, s1, s2, z1, z2)))
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import awq_marlin_quantize, marlin_quantize
        from vllm.scalar_type import scalar_types

        out = {}
        for name, (k, n) in (("1", (H, 2 * FE)), ("2", (FE, H))):  # Marlin takes W^T: k x n
            w = t.randn(k, n, dtype=bf, device="cuda") / 10
            if bits == 4:  # 4-bit with zero points (AWQ)
                _, q, s, z = awq_marlin_quantize(w, scalar_types.uint4, g)
                out["qt"] = scalar_types.uint4
            else:  # symmetric 8-bit (GPTQ)
                _, q, s, _, _, _ = marlin_quantize(w, scalar_types.uint8b128, g, False)
                z, out["qt"] = None, scalar_types.uint8b128
            for key, v in (("q", q), ("s", s), ("z", z)):
                out[key + name] = (
                    None if v is None else self.fill((E, *v.shape), v.dtype, v.unsqueeze(0).expand((E, *v.shape)))
                )
        out["bytes"] = sum(v.nbytes for k, v in out.items() if k[0] in "qsz" and k != "qt" and v is not None)
        return out

    def __call__(self, x, routes, l):
        """One layer with weight copy l: router GEMM, then the fused MoE call."""
        topk_weights, topk_ids, logits = routes
        w, m = self.w, self.m
        self.t.mm(x, self.router[l].t())
        if self.impl == "trtllm":
            from flashinfer.fused_moe import trtllm_bf16_moe

            return trtllm_bf16_moe(
                routing_logits=logits,
                routing_bias=None,
                hidden_states=x,
                gemm1_weights=w["w1"][l],
                gemm2_weights=w["w2"][l],
                num_experts=self.experts,
                top_k=m["K"] + m["S"],
                n_group=None,
                topk_group=None,
                intermediate_size=m["FE"],
                local_expert_offset=0,
                local_num_experts=self.experts,
                routing_method_type=1,
                tune_max_num_tokens=256,
            )
        if self.impl == "flashinfer":
            from flashinfer.fused_moe import cutlass_fused_moe

            y = cutlass_fused_moe(x, topk_ids, topk_weights, w["w1"][l], w["w2"][l], x.dtype, quant_scales=[])
            return y[0] if isinstance(y, list | tuple) else y
        if self.impl == "triton":
            from vllm.model_executor.layers.fused_moe import fused_experts

            return fused_experts(
                x,
                w["w1"][l],
                w["w2"][l],
                topk_weights,
                topk_ids,
                quant_config=w["qc"][l],
                global_num_experts=self.experts,
            )
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe

        return fused_marlin_moe(
            x,
            w["q1"][l],
            w["q2"][l],
            bias1=None,
            bias2=None,
            w1_scale=w["s1"][l],
            w2_scale=w["s2"][l],
            gating_output=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=w["qt"].id,
            global_num_experts=self.experts,
            w1_zeros=None if w["z1"] is None else w["z1"][l],
            w2_zeros=None if w["z2"] is None else w["z2"][l],
            is_k_full=True,
        )

    def warm(self, x, routes):
        """FlashInfer and TensorRT-LLM choose tactics with their autotuner on a first eager call."""
        if self.impl in ("flashinfer", "trtllm"):
            from flashinfer.autotuner import autotune

            try:
                with autotune(True):
                    self(x, routes, 0)
            except Exception:  # untuned calls still use valid default tactics
                pass


def time_calls(torch, fn, n_calls, reps):
    """Microseconds per call of fn(0) ... fn(n_calls - 1), captured in one CUDA graph and replayed `reps` times."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(n_calls):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n_calls):
            fn(i)
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    us = e0.elapsed_time(e1) * 1000 / (reps * n_calls)
    del g
    return us


def tune(torch, layer, windows):
    """Fastest Triton block configuration for BF16 fused_experts on three of the windows."""
    from vllm.model_executor.layers.fused_moe import override_config

    space = [
        dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=1, num_warps=w, num_stages=st)
        for bm in (16, 32, 64)
        for bn in (32, 64, 128, 256)
        for bk in (64, 128, 256)
        for w in (4, 8)
        for st in (3, 4)
    ]
    sample = windows[:: max(1, len(windows) // 3)][:3]
    routes = [route_tensors(torch, layer.m, r["ids"]) for r in sample]
    xs = [torch.randn(r["ids"].shape[0], layer.m["H"], dtype=torch.bfloat16, device="cuda") for r in sample]
    module = sys.modules[override_config.__module__]
    results = []
    for cfg in space:
        try:
            with override_config(cfg):
                us = np.mean(
                    [
                        time_calls(torch, lambda i, x=x, rt=rt: layer(x, rt, i % layer.L), layer.L, 3)
                        for x, rt in zip(xs, routes, strict=True)
                    ]
                )
            results.append((us, cfg))
        except Exception as ex:  # configurations the kernel cannot run, e.g. too much shared memory
            results.append((float("inf"), dict(cfg, error=repr(ex)[:120])))
        finally:
            module._config = None  # override_config does not restore the default after an exception
    results.sort(key=lambda z: z[0])
    return results[0][1], results[0][0], results[:5]


def write_tuned(layer, tuned):
    """Store per-batch configurations where vLLM looks them up (VLLM_TUNED_CONFIG_FOLDER) and clear its cache."""
    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe

    folder = Path(os.environ["VLLM_TUNED_CONFIG_FOLDER"])
    folder.mkdir(parents=True, exist_ok=True)
    name = fused_moe.get_config_file_name(layer.experts, layer.m["FE"], None, None)
    (folder / name).write_text(json.dumps({str(b): c for b, c in tuned.items()}, indent=1))
    fused_moe.get_moe_configs.cache_clear()


def append_jsonl(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def device_info(torch):
    import flashinfer
    import triton
    import vllm

    p = torch.cuda.get_device_properties(0)
    return dict(
        name=p.name,
        sms=p.multi_processor_count,
        memory_GB=p.total_memory / 1e9,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        vllm=vllm.__version__,
        flashinfer=flashinfer.__version__,
        triton=triton.__version__,
    )


def measure(torch, layer, windows, reps):
    """Layer time and weight bytes of every window."""
    rows = []
    for r in windows:
        ids = r["ids"]
        routes = route_tensors(torch, layer.m, ids)
        x = torch.randn(ids.shape[0], layer.m["H"], dtype=torch.bfloat16, device="cuda")
        layer.warm(x, routes)
        us = time_calls(torch, lambda i: layer(x, routes, i % layer.L), layer.L, reps)
        unique = int(np.unique(ids).size)
        nbytes = layer.router_bytes + layer.shared_bytes + unique * layer.expert_bytes
        key = [r["source"], r["group"], r["layer"], r["step"]]
        rows.append(dict(key=key, live=int(ids.shape[0]), unique=unique, bytes=nbytes, us=us))
    return rows


def benchmark(torch, a, model, kernel, windows):
    """Tune BF16 Triton if requested, time every batch size and append the results to the output files."""
    fmt, impl = kernel.split(":")
    layer = MoELayer(torch, fmt, impl, MODELS[model])
    print(model, kernel, f"expert {layer.expert_bytes / 1e6:.3f} MB", flush=True)
    if (fmt, impl) == ("bf16", "triton") and not a.no_tune:
        tuned = {}
        for b, wins in windows.items():
            cfg, us, top = tune(torch, layer, wins)
            tuned[b] = {k: v for k, v in cfg.items() if k != "error"}
            append_jsonl(a.out / "tuning.jsonl", dict(model=model, batch=b, best_us=us, best=cfg, top=top))
        write_tuned(layer, tuned)
    for b, wins in windows.items():
        start = time.monotonic()
        rows = measure(torch, layer, wins, a.reps)
        w = window_weights(wins)
        summary = dict(
            model=model,
            format=fmt,
            kernel=impl,
            batch=b,
            windows=len(rows),
            copies=layer.L,
            layer_us=float(w @ [r["us"] for r in rows]),
            bytes=float(w @ [r["bytes"] for r in rows]),
            unique_experts=float(w @ [r["unique"] for r in rows]),
            expert_bytes=layer.expert_bytes,
            seconds=time.monotonic() - start,
        )
        append_jsonl(a.out / "summary.jsonl", summary)
        append_jsonl(a.out / "windows.jsonl", dict(model=model, format=fmt, kernel=impl, batch=b, rows=rows))
        tbps = summary["bytes"] / summary["layer_us"] / 1e6
        print(model, kernel, f"B{b}", f"{summary['layer_us']:.1f} us", f"{tbps:.2f} TB/s", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--traces", type=Path, required=True, help="directory with <model>_b<batch>_heldout.npz")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--kernels", default=",".join(KERNELS), help="format:implementation pairs")
    p.add_argument("--batches", default=",".join(map(str, BATCHES)))
    p.add_argument("--windows", type=int, default=128, help="windows per batch size")
    p.add_argument("--reps", type=int, default=5, help="graph replays per window")
    p.add_argument("--no-tune", action="store_true", help="keep vLLM's default Triton configurations for BF16")
    return p.parse_args()


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_TUNED_CONFIG_FOLDER", str(a.out / "tuned_configs"))
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config

    torch.manual_seed(37)
    batches = [int(b) for b in a.batches.split(",")]
    with set_current_vllm_config(VllmConfig()):
        append_jsonl(a.out / "device.jsonl", device_info(torch))
        for model in a.models.split(","):
            windows = {b: report_windows(a.traces, model, b, a.windows) for b in batches}
            for kernel in a.kernels.split(","):
                benchmark(torch, a, model, kernel, windows)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
