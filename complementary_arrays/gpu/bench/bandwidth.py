"""Read-only HBM bandwidth of one GPU against the read size, with a streaming Triton kernel.

Call i reads buffer i mod k of a 16 GiB pool (the k buffers together are at least 512 MB, four times the L2), every
call of one size is replayed from one CUDA graph, and the fastest of 15 launch shapes is kept. Small reads show the
per-kernel floor (launch and ramp), large reads the attainable read bandwidth (the read-floor reference of report.py).
torch.sum over the same buffers and a large copy are recorded for comparison.

usage: python -m complementary_arrays.gpu.bench.bandwidth OUT.json [--reps 5]
"""

import argparse
import json
import time
from pathlib import Path

SIZES_MB = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)
SHAPES = [(iters, block, warps) for iters in (1, 2, 4, 8, 16) for block, warps in ((2048, 4), (4096, 8), (8192, 8))]


def read_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ITERS: tl.constexpr):
        """Each program loads ITERS x BLOCK int32 words once (evict first) and writes one int32."""
        base = tl.program_id(0) * BLOCK * ITERS
        acc = tl.zeros([BLOCK], dtype=tl.int32)
        for i in range(ITERS):
            offs = base + i * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + offs, mask=offs < n, other=0, eviction_policy="evict_first")
        tl.store(out_ptr + tl.program_id(0), tl.sum(acc, 0))

    def launch(v, out, iters, block, warps):
        n = v.numel()
        kernel[(triton.cdiv(n, block * iters),)](v, out, n, BLOCK=block, ITERS=iters, num_warps=warps)

    return launch


def time_calls(torch, fn, n_calls, reps):
    """us per call of fn(0) .. fn(n_calls - 1), captured in one CUDA graph and replayed reps times."""
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
    return e0.elapsed_time(e1) * 1000 / (reps * n_calls)


def large_copy_and_sum(torch):
    n = 8 << 30
    x = torch.empty(n, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    x.random_()
    res = {}
    for name, fn, nbytes in (
        ("copy_read_plus_write", lambda: y.copy_(x), 2 * n),
        ("sum_read", lambda: x.view(torch.bfloat16).sum(), n),
    ):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(10):
            fn()
        e1.record()
        torch.cuda.synchronize()
        res[name + "_TBps"] = nbytes * 10 / (e0.elapsed_time(e1) / 1000) / 1e12
    return res


def sweep(torch, reps):
    launch = read_kernel()
    pool = torch.empty(16 << 30, dtype=torch.uint8, device="cuda")
    pool.random_()
    out = torch.empty(1 << 22, dtype=torch.int32, device="cuda")
    res = []
    for mb in SIZES_MB:
        n = mb << 20
        k = int(min(pool.numel() // n, max(4, -(-(512 << 20) // n))))
        views = [pool[i * n : (i + 1) * n].view(torch.int32) for i in range(k)]
        us_sum = time_calls(torch, lambda i: views[i % k].view(torch.bfloat16).sum(), k, reps)
        best = (float("inf"), None)
        for iters, block, warps in SHAPES:
            if -(-views[0].numel() // (block * iters)) < 1:
                continue
            us = time_calls(torch, lambda i, it=iters, b=block, w=warps: launch(views[i % k], out, it, b, w), k, reps)
            best = min(best, (us, dict(iters=iters, block=block, warps=warps)), key=lambda z: z[0])
        res.append(
            dict(
                MB=mb,
                buffers=k,
                us=best[0],
                TBps=n / (best[0] * 1e-6) / 1e12,
                cfg=best[1],
                torch_sum_us=us_sum,
                torch_sum_TBps=n / (us_sum * 1e-6) / 1e12,
            )
        )
        print(json.dumps(res[-1]), flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--reps", type=int, default=5)
    a = p.parse_args()
    import torch

    prop = torch.cuda.get_device_properties(0)
    device = dict(name=prop.name, sms=prop.multi_processor_count, torch=torch.__version__, cuda=torch.version.cuda)
    large = large_copy_and_sum(torch)
    torch.cuda.empty_cache()
    res = dict(device=device, large=large, sweep=sweep(torch, a.reps), t=time.time())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
