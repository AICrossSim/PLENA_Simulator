#!/usr/bin/env python3
"""Controlled borrowable-credit follow-up using a frozen second executable.

The disabled control must reproduce the prior candidate exactly, including
native statistics, before this follow-up is allowed into the final report.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from compare_moe_normal import read_json, digest, require, run_comparison
from run_moe_dma_campaign import save


def campaign(root, source, binary):
    base = read_json(root/'campaign.json')
    fixtures = read_json(source/'campaign.json')['fixtures']
    names = base['architectures']
    tasks = []
    for enabled in [False, True]:
        label = 'reserved' if enabled else 'control'
        paths = []
        for name in names:
            a = read_json(root/'architectures/candidate'/(name+'.json'))
            a['dma']['fair_credits'] = enabled
            path = root/'fairness/architectures'/label/(name+'.json')
            save(path,a); paths.append(path)
        for fixture in fixtures: tasks.append((fixture,label,paths))
    plan = dict(executable_sha256=digest(binary), driver_sha256=digest(__file__),
        comparison_driver_sha256=digest(Path(__file__).with_name('compare_moe_normal.py')),
        tasks=[dict(fixture=f,label=l,architectures=[dict(path=str(p),sha256=digest(p)) for p in ps]) for f,l,ps in tasks])
    save(root/'fairness/execution_plan.json', plan)
    def execute(task):
        fixture,label,paths = task
        start=time.monotonic()
        print('Starting '+fixture+'_'+label,flush=True)
        try:
            result = run_comparison(binary,source/fixture/'workload.json',source/fixture/'golden.json',paths,
                root/'fairness/comparisons'/(fixture+'_'+label),repeats=2,hbm_channels=8,timeout=1200,workers=4,
                **base['numerical_tolerance'])
            if label == 'control':
                prior=read_json(root/'comparisons'/(fixture+'_candidate')/'comparison.json')
                for new,old in zip(result['comparisons'],prior['comparisons']):
                    metrics=dict(new['result']); front=dict(metrics['dma_frontend'])
                    for key in ['fair_credit_reserve_per_core','fair_credit_wait_ps']:
                        require(front.pop(key) == 0, 'disabled fairness charged service or reservation')
                    metrics['dma_frontend']=front
                    require(metrics == old['result'] and new['native'] == old['native'],
                            'disabled control changed prior candidate timing/values/counters')
            return dict(status='passed',wall_seconds=time.monotonic()-start,
                timings_ps={r['architecture']['name']:r['result']['total_ps'] for r in result['comparisons']})
        except Exception as error: return dict(status='failed',error=str(error),wall_seconds=time.monotonic()-start)
    outcomes={}
    # Four independent simulations at a time; the primary campaign may coexist.
    with ThreadPoolExecutor(max_workers=1) as pool:
        futures={pool.submit(execute,t):t[0]+'_'+t[1] for t in tasks}
        for f in as_completed(futures):
            name=futures[f];outcomes[name]=f.result()
            print(name+': '+outcomes[name]['status'],flush=True)
            save(root/'fairness/status.json',dict(status='running',all_gates_passed=False,outcomes=outcomes))
    passed=len(outcomes)==len(tasks) and all(o['status']=='passed' for o in outcomes.values())
    require(digest(binary)==plan['executable_sha256'],'follow-up binary changed')
    save(root/'fairness/status.json',dict(status='passed' if passed else 'failed',all_gates_passed=passed,outcomes=outcomes))
    require(passed,'fairness follow-up failed; no benefit claim')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','source','binary']: p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();campaign(a.root.resolve(),a.source.resolve(),a.binary.resolve())
