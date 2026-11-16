"""Decode time per output token at a fixed concurrency B on one B200 (vLLM or SGLang), measured by client streaming.

Workload as in workload.py. Every request is streamed; the time between two consecutive output chunks divided by the
tokens of the chunk is one interval per token. An interval is decode-only if no newly admitted request was in prefill
during it (no overlap with any [admission, first token] window); intervals before every initial slot has produced its
first token are skipped, then W seconds are measured. Reported per B: mean over decode-only intervals (the metric of
the MoE-layer time) and over all intervals. --moe no_moe zeroes every MoE block, --no-shared only the shared expert, so
that (full - no_moe) / MoE layers is the MoE-layer time and (no_shared - no_moe) / MoE layers its routed part.

usage: python -m complementary_arrays.gpu.bench.stream {vllm,sglang} {qwen35,deepseek} MODE --model DIR --pool POOL
       --capture DIR --out DIR [--moe full|no_moe] [--no-shared] [--batches 2,8,32,64,128] [--window 40] [--work DIR]
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from . import engines

if __name__ != "__main__":
    engines.setup_from_env()  # SGLang's scheduler process re-imports this module


def intervals(recs, t_warm, t_end):
    wins = sorted((r["admit"], r["times"][0]) for r in recs if not r["initial"] and r["times"])
    starts, ends = np.array([w[0] for w in wins]), np.array([w[1] for w in wins])
    dec, every = [], []
    for r in recs:
        t, c = r["times"], r["counts"]
        for k in range(1, len(t)):
            a, b = t[k - 1], t[k]
            if a < t_warm or b > t_end or c[k] <= 0:
                continue
            per = (b - a) / c[k]
            every += [per] * c[k]
            if not len(starts) or not np.any((starts < b) & (ends > a)):
                dec += [per] * c[k]
    return np.array(dec), np.array(every)


def decode_periods(recs, t_warm, t_end):
    """Spans of [t_warm, t_end] in which no newly admitted request is in prefill."""
    out, cur = [], t_warm
    for s0, e0 in sorted((r["admit"], r["times"][0]) for r in recs if not r["initial"] and r["times"]):
        if e0 <= cur or s0 >= t_end:
            continue
        if s0 > cur:
            out.append((cur, min(s0, t_end)))
        cur = max(cur, e0)
    if cur < t_end:
        out.append((cur, t_end))
    return out


def energy_stats(power, recs, t_warm, t_end):
    """Board energy over the decode-only periods (power.window(a, b): samples (time, energy mJ, power mW))."""
    periods, E, T, P = decode_periods(recs, t_warm, t_end), 0.0, 0.0, []
    for a, b in periods:
        w = power.window(a, b)
        if len(w) >= 2:
            E += (w[-1][1] - w[0][1]) / 1e3
            T += w[-1][0] - w[0][0]
            P += [x[2] / 1e3 for x in w]
    every = power.window(t_warm, t_end)
    e_all = (every[-1][1] - every[0][1]) / 1e3 if len(every) >= 2 else None
    t_all = every[-1][0] - every[0][0] if len(every) >= 2 else None
    return dict(
        decode_periods=len(periods),
        decode_s=T,
        decode_energy_J=E,
        decode_power_W=E / T if T > 0 else None,
        decode_power_instant_mean_W=float(np.mean(P)) if P else None,
        window_power_W=e_all / t_all if t_all else None,
        samples=len(every),
    )


async def measure(stream, abort_all, B, reqs, order, w, rng, window, power=None):
    """Decode-only time per token at concurrency B; with a power sampler also the energy per token of the decode-only
    periods (decode-only power x decode-only time per token / B)."""
    recs, pending, tasks = [], list(order), set()

    async def one(r, k, initial):
        n = len(r["out"]) - k
        rec = dict(admit=time.perf_counter(), times=[], counts=[], initial=initial, source=r["source"])
        recs.append(rec)
        try:
            async for cnt in stream(list(r["input_ids"]) + r["out"][:k], n):
                rec["times"].append(time.perf_counter())
                rec["counts"].append(cnt)
        except asyncio.CancelledError:
            pass

    for i in rng.choice(len(reqs), B, replace=False, p=w):
        r = reqs[i]
        pending = [p for p in pending if p["request"] != r["request"]]
        tasks.add(asyncio.create_task(one(r, int(rng.integers(0, len(r["out"]))), True)))
    t_warm, t0 = None, time.perf_counter()
    while True:
        done, _ = await asyncio.wait(tasks, timeout=0.05, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            tasks.discard(d)
            if pending:
                tasks.add(asyncio.create_task(one(pending.pop(0), 0, False)))
        if t_warm is None and all(r["times"] for r in recs if r["initial"]):
            t_warm = time.perf_counter() + 0.2
        now = time.perf_counter()
        if (t_warm and now > t_warm + window) or (not pending and len(tasks) < B) or now - t0 > window + 600:
            break
    t_end = time.perf_counter()
    for t in tasks:
        t.cancel()
    await abort_all()
    await asyncio.sleep(1.0)
    dec, every = intervals(recs, t_warm or t_end, t_end)
    energy = energy_stats(power, recs, t_warm or t_end, t_end) if power else {}
    if energy.get("decode_power_W") and len(dec):
        energy["energy_per_token_mJ"] = energy["decode_power_W"] * float(dec.mean()) / B * 1e3
    return dict(
        batch=B,
        **energy,
        window_s=t_end - (t_warm or t_end),
        decode_only_tokens=len(dec),
        all_tokens=len(every),
        decode_only_ms_mean=float(dec.mean() * 1e3) if len(dec) else None,
        decode_only_ms_median=float(np.median(dec) * 1e3) if len(dec) else None,
        all_ms_mean=float(every.mean() * 1e3) if len(every) else None,
        admitted=len(recs),
        pool_exhausted=not pending,
    )


def parse_args(doc):
    p = argparse.ArgumentParser(description=doc.splitlines()[0])
    p.add_argument("stack", choices=("vllm", "sglang"))
    p.add_argument("kind", choices=("qwen35", "deepseek"))
    p.add_argument("mode")
    p.add_argument("--model", type=Path, required=True, help="checkpoint directory")
    p.add_argument("--pool", type=Path, required=True, help="pool_<kind>.pkl of capture/pool.py")
    p.add_argument("--capture", type=Path, required=True, help="reference sampled capture of capture/capture.py")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, help="directory for the linked model directories (default: OUT/work)")
    p.add_argument("--moe", choices=("full", "no_moe"), default="full")
    p.add_argument("--no-shared", action="store_true")
    p.add_argument("--batches", default="2,8,32,64,128")
    return p


def main():
    p = parse_args(__doc__)
    p.add_argument("--window", type=float, default=40.0)
    a = p.parse_args()
    from . import workload

    a.out.mkdir(parents=True, exist_ok=True)
    s = engines.configure(a.stack, a.kind, a.mode, a.model, a.work or a.out / "work", a.moe, a.no_shared)
    reqs, order, w, rng = workload.load(a.pool, a.capture, a.kind)
    batches = [int(b) for b in a.batches.split(",")]
    stream, abort_all, llm, meta = engines.engine(s, max(batches))
    meta.update(kind=a.kind, mode=a.mode, moe=a.moe, no_shared=a.no_shared, window_s=a.window)
    path = a.out / f"stream_{engines.tag(s)}.jsonl"

    async def measure_all():
        for B in batches:
            rec = dict(meta, **(await measure(stream, abort_all, B, reqs, order, w, rng, a.window)))
            print(json.dumps({k: v for k, v in rec.items() if k not in ("kwargs", "load_stats")}), flush=True)
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    engines.run_batches(s, llm, measure_all)
    print("STREAM_DONE", engines.tag(s), flush=True)
    engines.stop(llm)


if __name__ == "__main__":
    main()
