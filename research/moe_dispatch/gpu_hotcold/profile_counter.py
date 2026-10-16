"""One representative full MoE invocation for separate Nsight collection.

Counter replay durations must not replace the unprofiled CUDA-event timings.
The selected window minimizes distance to its Batch's mean expert populations;
selection never uses measured runtime. An explicit ID can override selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .traces import load_routes, population_metrics


def select_window(windows, batch, window_id=None):
    candidates = [w for w in windows if w['batch'] == batch]
    if window_id is not None:
        return next(w for w in candidates if w['id'] == window_id)
    populations = [population_metrics(w) for w in candidates]
    means = {k: statistics.fmean(p[k] for p in populations)
             for k in ('hot_experts', 'cold_experts')}
    return min(zip(candidates, populations), key=lambda item: (
        sum(abs(item[1][k] - means[k]) for k in means), item[0]['id']))[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--routes', type=Path, required=True)
    p.add_argument('--batch', type=int, required=True)
    p.add_argument('--path', default='vllm_grouped')
    p.add_argument('--window-id')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', default='all')
    p.add_argument('--torch-profiler', action='store_true',
                   help='Also collect a separate three-invocation eager Kineto trace')
    args = p.parse_args()
    import torch
    from .benchmark import BenchmarkContext, WindowRunner
    data = load_routes(args.routes)
    window = select_window(data['windows'], args.batch, args.window_id)
    ctx = BenchmarkContext(enable_vllm=args.path.startswith('vllm'))
    runner = WindowRunner(ctx, window)
    with torch.inference_mode():
        for _ in range(20):
            runner.run(args.path)
        torch.cuda.synchronize()
        # A complete invocation preserves preceding Shared/Routed cache effects.
        # Only the requested stage(s) enter the NVTX counter-collection range.
        active = []
        for name, fn, enabled in runner.get_stages(args.path):
            if not enabled:
                continue
            collect = args.stage in ('all', name)
            if collect:
                torch.cuda.nvtx.range_push('measured_moe')
                torch.cuda.nvtx.range_push(name)
                active.append(name)
            fn()
            if collect:
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        if args.torch_profiler:
            from .benchmark import capture_torch_profiler
            args.output.parent.mkdir(parents=True, exist_ok=True)
            capture_torch_profiler(runner, args.path,
                                   args.output.with_name(args.output.stem + '_kineto'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'window_id': window['id'], 'batch': args.batch, 'path': args.path,
        'stage': args.stage, 'collected_stages': active,
        'selection': 'nearest mean hot/cold expert counts; tie by window ID',
        'population': population_metrics(window), 'metadata': ctx.metadata(),
        'scope': 'single representative invocation; Nsight replay characterization, not E2E timing',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
