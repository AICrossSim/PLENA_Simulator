"""GPU kernel time per MoE layer and decode step on one B200, split into router, shared expert and routed experts.

B slots in steady state (workload.py) route their real hidden states; once every slot has its first token, a
torch.profiler trace of N decode steps is taken in eager mode (CUDA graphs off): vLLM's engine steps directly, SGLang
through its scheduler profiler. A kernel counts for the MoE layer only if launched inside a MoE block, and for the
router or shared expert if also inside their ranges (formats.annotate_moe); per layer = sum / N / MoE layers. Without
CUDA graphs this is a sum of kernel times, not a wall-clock time: report.py uses it only where a wall-clock cell is
void.

usage: python -m complementary_arrays.gpu.bench.kernel_time {vllm,sglang} {qwen35,deepseek} MODE --model DIR --pool POOL
       --capture DIR --out DIR [--batches 2,8,32,64,128] [--steps 64]
"""

import asyncio
import glob
import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from . import engines
from .formats import LAYERS
from .stream import parse_args

if __name__ != "__main__":
    engines.setup_from_env()

PARTS = ("router", "shared", "routed")


def attribute(trace_path):
    """Kernel time (us) by part from a Chrome trace, via the launch of each kernel and the ranges open on its thread."""
    with (gzip.open if str(trace_path).endswith(".gz") else open)(trace_path, "rt") as f:
        events = json.load(f)["traceEvents"]
    ranges, launch, kernels = defaultdict(list), {}, []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, args = e.get("cat", ""), e.get("args", {})
        if cat == "user_annotation" and e.get("name", "").startswith("cat::"):
            ranges[e["tid"]].append((e["ts"], e["ts"] + e["dur"], e["name"][5:]))
        elif cat in ("cuda_runtime", "cuda_driver") and "correlation" in args:
            launch[args["correlation"]] = (e["tid"], e["ts"])
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append((args.get("correlation"), e["dur"], e.get("name", "?")))
    total, names = defaultdict(float), defaultdict(lambda: defaultdict(float))
    for corr, dur, name in kernels:
        part = "other"
        if corr in launch:
            tid, t0 = launch[corr]
            inside = {n for s0, e0, n in ranges.get(tid, ()) if s0 <= t0 <= e0}
            if "moe" in inside:
                part = "shared" if "shared" in inside else "router" if "router" in inside else "routed"
        total[part] += dur
        names[part][name[:90]] += dur
    return dict(total=dict(total), names={p: dict(v) for p, v in names.items()})


def record(kind, res, steps):
    L = LAYERS[kind]
    part = {p: res["total"].get(p, 0.0) / steps / L for p in PARTS}
    return dict(
        moe_us_per_layer=sum(part.values()),
        router_us_per_layer=part["router"],
        shared_us_per_layer=part["shared"],
        routed_us_per_layer=part["routed"],
        no_shared_us_per_layer=part["router"] + part["routed"],
        other_us_per_step=res["total"].get("other", 0.0) / steps,
        top_kernels={p: sorted(res["names"].get(p, {}).items(), key=lambda z: -z[1])[:6] for p in PARTS},
    )


def profile_vllm(s, a, reqs, w, rng, batches, write):
    import torch
    import vllm
    from torch.profiler import ProfilerActivity, profile
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from . import formats as f

    kw = dict(
        model=f.vllm_model_dir(s["model"], s["kind"], s["mode"], s["work"]),
        max_model_len=8400,
        gpu_memory_utilization=0.85,
        max_num_seqs=max(batches),
        max_num_batched_tokens=16384,
        enable_prefix_caching=False,
        seed=37,
        trust_remote_code=True,
        enforce_eager=True,
    )
    f.vllm_engine_kwargs(s["kind"], s["mode"], kw)
    eng = LLM(**kw).llm_engine
    for B in batches:
        ids = []
        for j, i in enumerate(rng.choice(len(reqs), B, replace=B > len(reqs), p=w), start=1):
            r = reqs[i]
            k = int(rng.integers(0, len(r["out"])))
            n = max(len(r["out"]) - k, a.steps + 16)
            ids.append(f"p{B}_{j}")
            prompt = TokensPrompt(prompt_token_ids=list(r["input_ids"]) + r["out"][:k])
            eng.add_request(
                ids[-1], prompt, SamplingParams(max_tokens=n, min_tokens=n, ignore_eos=True, temperature=0.0)
            )
        first, warm = set(ids), 0
        while first and warm < 5000:  # until every slot has finished its prefill
            for o in eng.step():
                first.discard(o.request_id)
            warm += 1
        for _ in range(5):
            eng.step()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(a.steps):
                eng.step()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        trace = a.out / f"trace_{engines.tag(s)}_B{B}.json.gz"
        prof.export_chrome_trace(str(trace))
        res = attribute(trace)
        trace.unlink()
        eng.abort_request(ids)
        while eng.has_unfinished_requests():
            eng.step()
        write(
            dict(
                batch=B,
                steps=a.steps,
                warm_steps=warm,
                eager_step_ms=wall / a.steps * 1e3,
                vllm=vllm.__version__,
                **record(s["kind"], res, a.steps),
            )
        )


def profile_sglang(s, a, reqs, w, rng, batches, write):
    import sglang as sgl
    from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType

    from . import formats as f

    kw = dict(
        model_path=f.sglang_model_dir(s["model"], s["kind"], s["mode"], s["work"]),
        tp_size=1,
        mem_fraction_static=0.80,
        disable_radix_cache=True,
        max_running_requests=max(batches),
        context_length=engines.MAX_LEN,
        chunked_prefill_size=16384,
        disable_cuda_graph=True,
        random_seed=37,
        trust_remote_code=True,
        log_level="warning",
    )
    f.sglang_engine_kwargs(s["kind"], s["mode"], kw)
    llm = sgl.Engine(**kw)
    tm = llm.tokenizer_manager

    async def one_batch(B):
        started, stop = [], asyncio.Event()

        async def one(r, k):
            n = max(len(r["out"]) - k, a.steps + 64)
            sp = dict(max_new_tokens=n, min_new_tokens=n, ignore_eos=True, temperature=0.0)
            first = True
            async for _ in await llm.async_generate(
                input_ids=list(r["input_ids"]) + r["out"][:k], sampling_params=sp, stream=True
            ):
                if first:
                    started.append(1)
                    first = False
                if stop.is_set():
                    break

        pick = rng.choice(len(reqs), B, replace=False, p=w)
        tasks = [asyncio.ensure_future(one(reqs[i], int(rng.integers(0, len(reqs[i]["out"]))))) for i in pick]
        t0 = time.perf_counter()
        while len(started) < B and time.perf_counter() - t0 < 900:
            await asyncio.sleep(0.1)
        ok = len(started) == B
        await asyncio.sleep(0.5)
        d = a.out / f"trace_{engines.tag(s)}_B{B}"
        d.mkdir(exist_ok=True)
        for x in d.iterdir():
            x.unlink()
        if ok:
            req = ProfileReq(
                req_type=ProfileReqType.START_PROFILE,
                output_dir=str(d),
                num_steps=a.steps,
                activities=["CPU", "GPU"],
                profile_id=f"B{B}",
            )
            await tm.start_profile(req)
            t1, size = time.perf_counter(), -1
            while time.perf_counter() - t1 < 900:  # until the trace file exists and stops growing
                await asyncio.sleep(2.0)
                fs = glob.glob(str(d / "*.trace.json*"))
                if fs:
                    n = os.path.getsize(fs[0])
                    if n == size and n > 0:
                        await asyncio.sleep(5.0)
                        if os.path.getsize(fs[0]) == n:
                            break
                    size = n
        stop.set()
        tm.abort_request(abort_all=True)
        await asyncio.sleep(1.0)
        for t in tasks:
            t.cancel()
        await asyncio.sleep(2.0)
        fs = glob.glob(str(d / "*.trace.json*")) if ok else []
        return ok, (fs[0] if fs else None)

    for B in batches:
        ok, trace = llm.loop.run_until_complete(one_batch(B))
        rec = dict(batch=B, steps=a.steps, all_slots_started=ok, sglang=sgl.__version__)
        res = None
        for _ in range(30 if trace else 0):  # the file may still be being written
            try:
                res = attribute(trace)
                break
            except Exception:
                time.sleep(10)
        if res is not None:
            Path(trace).unlink()
            rec.update(record(s["kind"], res, a.steps))
        write(rec)
    return llm


def main():
    p = parse_args(__doc__)
    p.add_argument("--steps", type=int, default=64)
    a = p.parse_args()
    from . import workload

    a.out.mkdir(parents=True, exist_ok=True)
    s = engines.configure(a.stack, a.kind, a.mode, a.model, a.work or a.out / "work", profile=True)
    reqs, _, w, rng = workload.load(a.pool, a.capture, a.kind)
    batches = [int(b) for b in a.batches.split(",")]
    path = a.out / f"profile_{a.stack}_{a.kind}_{a.mode}.jsonl"

    def write(rec):
        rec = dict(stack=a.stack, kind=a.kind, mode=a.mode, moe_layers=LAYERS[a.kind], **rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "top_kernels"}), flush=True)
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    if a.stack == "vllm":
        profile_vllm(s, a, reqs, w, rng, batches, write)
        print("PROFILE_DONE", a.stack, a.kind, a.mode, flush=True)
    else:
        llm = profile_sglang(s, a, reqs, w, rng, batches, write)
        print("PROFILE_DONE", a.stack, a.kind, a.mode, flush=True)
        engines.stop(llm)


if __name__ == "__main__":
    main()
