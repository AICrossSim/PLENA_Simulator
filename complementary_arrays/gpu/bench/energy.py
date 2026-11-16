"""Energy per output token on one B200 from the board's NVML energy counter, in steady state or in closed batches.

  steady  the streaming workload of stream.py; energy integrated over the decode-only periods (no newly admitted
          request in prefill); energy per token = decode-only power x decode-only time per token / B
  closed  every repetition starts B requests together (start positions as in workload.py), each generating exactly N
          tokens; energy integrated from 0.3 s after the last request's first token to 50 ms before the first request
          finishes; repetitions with new requests until MIN_S seconds of decode; energy per token = decode power x step
          time / B. Used where decode-only periods are too short for the counter (short DeepSeek-V2-Lite requests)
The MoE energy is the difference to the MoE-zeroed run of the same engine and method on the same GPU (report.py).
Before the engine starts, the idle power is recorded for 20 s.

usage: python -m complementary_arrays.gpu.bench.energy {vllm,sglang} {qwen35,deepseek} MODE --model DIR --pool POOL
       --capture DIR --out DIR --method {steady,closed} [--moe full|no_moe] [--batches 2,8,32,64,128] [--window 40]
       [--tokens 512] [--min-seconds 10] [--max-reps 8]
  The GPU is the first entry of CUDA_VISIBLE_DEVICES in PCI bus order (set CUDA_DEVICE_ORDER=PCI_BUS_ID).
"""

import asyncio
import json
import os
import threading
import time

import numpy as np

from . import engines
from . import stream as st

if __name__ != "__main__":
    engines.setup_from_env()


class Power:
    """NVML sampler: (time, energy mJ, instantaneous power mW) at `hz`."""

    def __init__(self, hz=20):
        import pynvml as nvml

        nvml.nvmlInit()
        self.nvml = nvml
        self.handle = nvml.nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]))
        self.dt, self.samples, self.stop = 1.0 / hz, [], False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        n, h = self.nvml, self.handle
        while not self.stop:
            try:
                e = n.nvmlDeviceGetTotalEnergyConsumption(h)
                p = n.nvmlDeviceGetFieldValues(h, [n.NVML_FI_DEV_POWER_INSTANT])[0].value.uiVal
                self.samples.append((time.perf_counter(), e, p))
            except Exception:
                pass
            time.sleep(self.dt)

    def window(self, a, b):
        return [x for x in self.samples if a <= x[0] <= b]


async def measure(stream, abort_all, B, reqs, w, rng, power, n_tokens, min_s, max_reps, max_len):
    E = T = 0.0
    steps, reps = [], []
    for rep in range(max_reps):
        recs, tasks, ctxs = [], [], []

        async def one(r, k, recs=recs):
            rec = dict(times=[], counts=[])
            recs.append(rec)
            try:
                async for cnt in stream(list(r["input_ids"]) + r["out"][:k], n_tokens):
                    rec["times"].append(time.perf_counter())
                    rec["counts"].append(cnt)
            except asyncio.CancelledError:
                pass

        for i in rng.choice(len(reqs), B, replace=False, p=w):
            r = reqs[i]
            k = max(0, min(int(rng.integers(0, len(r["out"]))), max_len - n_tokens - len(r["input_ids"]) - 16))
            ctxs.append(len(r["input_ids"]) + k)
            tasks.append(asyncio.create_task(one(r, k)))
        await asyncio.gather(*tasks)
        short = B - sum(sum(x["counts"]) == n_tokens for x in recs)
        if short:
            print("short requests", short, flush=True)
        ta = max(x["times"][0] for x in recs) + 0.3
        tb = min(x["times"][-1] for x in recs) - 0.05
        if tb - ta < 0.5:
            reps.append(dict(rep=rep, window_s=tb - ta, skipped=True))
            continue
        pw = power.window(ta, tb)
        if len(pw) < 2:
            continue
        e, t = (pw[-1][1] - pw[0][1]) / 1e3, pw[-1][0] - pw[0][0]
        per_req = [sum(c for tt, c in zip(x["times"], x["counts"], strict=True) if ta <= tt <= tb) for x in recs]
        st = (tb - ta) / float(np.median(per_req))
        E += e
        T += t
        steps.append(st)
        reps.append(
            dict(
                rep=rep,
                window_s=tb - ta,
                energy_J=e,
                power_W=e / t,
                step_ms=st * 1e3,
                samples=len(pw),
                ctx_mean=float(np.mean(ctxs)),
            )
        )
        await abort_all()
        await asyncio.sleep(0.5)
        if T >= min_s:
            break
    P = E / T if T > 0 else None
    st = float(np.average(steps, weights=[r["window_s"] for r in reps if not r.get("skipped")])) if steps else None
    return dict(
        batch=B,
        method="closed",
        N=n_tokens,
        decode_s=T,
        decode_energy_J=E,
        decode_power_W=P,
        decode_only_ms_mean=st * 1e3 if st else None,
        energy_per_token_mJ=P * st / B * 1e3 if P and st else None,
        reps=reps,
    )


def main():
    p = st.parse_args(__doc__)
    p.add_argument("--method", choices=("steady", "closed"), required=True)
    p.add_argument("--window", type=float, default=40.0)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--min-seconds", type=float, default=10.0)
    p.add_argument("--max-reps", type=int, default=8)
    a = p.parse_args()
    from . import workload

    a.out.mkdir(parents=True, exist_ok=True)
    s = engines.configure(a.stack, a.kind, a.mode, a.model, a.work or a.out / "work", a.moe, a.no_shared)
    reqs, order, w, rng = workload.load(a.pool, a.capture, a.kind)
    batches = [int(b) for b in a.batches.split(",")]
    power = Power()
    t0 = time.perf_counter()
    time.sleep(20.0)
    iw = power.window(t0, time.perf_counter())
    idle = dict(idle_power_W=(iw[-1][1] - iw[0][1]) / 1e3 / (iw[-1][0] - iw[0][0]), idle_samples=len(iw))
    print("IDLE", json.dumps(idle), flush=True)
    stream, abort_all, llm, meta = engines.engine(s, max(batches))
    meta.update(kind=a.kind, mode=a.mode, moe=a.moe, method=a.method, **idle)
    path = a.out / f"energy_{engines.tag(s)}.jsonl"

    async def measure_all():
        for B in batches:
            if a.method == "steady":
                res = await st.measure(stream, abort_all, B, reqs, order, w, rng, a.window, power)
            else:
                res = await measure(
                    stream, abort_all, B, reqs, w, rng, power, a.tokens, a.min_seconds, a.max_reps, engines.MAX_LEN
                )
            rec = dict(meta, **res)
            print(json.dumps({k: v for k, v in rec.items() if k not in ("kwargs", "load_stats", "reps")}), flush=True)
            with path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

    engines.run_batches(s, llm, measure_all)
    print("ENERGY_DONE", engines.tag(s), flush=True)
    engines.stop(llm)


if __name__ == "__main__":
    main()
