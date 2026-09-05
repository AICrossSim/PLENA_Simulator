#!/usr/bin/env python3
"""Run all predeclared full-shape comparisons, including causal sensitivities.

Every invocation uses fresh run directories and rechecks all evidence gates.
Parallelism is between independent simulations, never inside their executor.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from compare_moe_normal import digest, read_json, run_comparison


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def campaign(root, binary, workers=2, timeout=900, architecture_workers=4):
    manifest = read_json(root/'campaign.json')
    architectures = [root/'architectures'/(n+'.json') for n in manifest['architectures']]
    tasks = [(name, root/name, architectures) for name in manifest['fixtures']]
    # Predeclared sensitivities use the identical DeepSeek batch-32 fixture.
    # Different timing/cache budgets form separate fair comparison groups.
    for variant in ['legacy_serialized', 'cache_disabled']:
        paths = []
        for source in architectures:
            arch = read_json(source)
            if variant == 'legacy_serialized':
                arch['matrix_timing'] = variant
            else:
                for core in arch['cores']: core['read_cache_bytes'] = 0
            path = root/'sensitivity_architectures'/variant/source.name
            save(path,arch); paths.append(path)
        tasks.append(('deepseek_b32_'+variant, root/'deepseek_full_decode_b32', paths))
    declaration = dict(campaign_sha256=digest(root/'campaign.json'), executable_sha256=digest(binary),
        driver_sha256=digest(__file__), comparison_driver_sha256=digest(Path(__file__).with_name('compare_moe_normal.py')),
        workers=workers, architecture_workers=architecture_workers, timeout_per_run_seconds=timeout,
        tasks=[dict(name=name,workload=str(fixture/'workload.json'),
                    architectures=[dict(path=str(p),sha256=digest(p)) for p in paths])
               for name,fixture,paths in tasks])
    save(root/'execution_plan.json',declaration)
    outcomes = {}
    save(root/'campaign_status.json',dict(status='running', all_gates_passed=False, outcomes=outcomes))
    def execute(task):
        name, fixture, paths = task
        print('Starting '+name, flush=True); started=time.monotonic()
        try:
            result = run_comparison(binary, fixture/'workload.json', fixture/'golden.json', paths,
                root/'comparisons'/name, repeats=manifest['repeats'], hbm_channels=manifest['hbm_channels'],
                timeout=timeout, workers=architecture_workers, **manifest['numerical_tolerance'])
            return dict(status='passed',wall_seconds=time.monotonic()-started,
                comparisons=[dict(name=r['architecture']['name'],total_ps=r['result']['total_ps'],
                    output_bit_exact=all(g['output_bit_exact'] for g in r['gates']))
                    for r in result['comparisons']])
        except Exception as error:
            return dict(status='failed',error=str(error),wall_seconds=time.monotonic()-started)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute,task):task[0] for task in tasks}
        for future in as_completed(futures):
            name=futures[future]; outcomes[name]=future.result()
            print(name+': '+outcomes[name]['status'],flush=True)
            save(root/'campaign_status.json',dict(status='running',all_gates_passed=False,outcomes=outcomes))
    passed = all(o['status']=='passed' for o in outcomes.values()) and len(outcomes)==len(tasks)
    save(root/'campaign_status.json',dict(status='passed' if passed else 'failed',all_gates_passed=passed,
        execution_plan_sha256=digest(root/'execution_plan.json'), outcomes=outcomes))
    if not passed: raise RuntimeError('campaign failed; no complete benefit conclusion is authorized')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2,choices=[1,2])
    parser.add_argument('--timeout',type=float,default=900)
    parser.add_argument('--architecture-workers',type=int,default=4,choices=range(1,9))
    args=parser.parse_args()
    campaign(args.root.resolve(),args.binary.resolve(),args.workers,args.timeout,args.architecture_workers)
