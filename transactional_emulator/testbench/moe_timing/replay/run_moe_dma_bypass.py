#!/usr/bin/env python3
"""Optimistic direct-path controls: prevent all-miss lookup cost faking a gain.

The legacy direct path models neither lookup nor return-copy service, so these
are explicitly optimistic controls, not fully port-calibrated hardware designs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from compare_moe_normal import digest,read_json,run_comparison,require
from run_moe_dma_campaign import save


def main(root, source, binary):
    base=read_json(root/'campaign.json'); fixtures=read_json(source/'campaign.json')['fixtures']
    tasks=[]
    for label,stage in [('direct2','reference'),('direct4','candidate')]:
        paths=[]
        for name in base['architectures']:
            arch=read_json(root/'architectures'/stage/(name+'.json'));arch.pop('dma')
            path=root/'bypass/architectures'/label/(name+'.json');save(path,arch);paths.append(path)
        tasks.extend((f,label,paths) for f in fixtures)
    plan=dict(executable_sha256=digest(binary),driver_sha256=digest(__file__),
        comparison_driver_sha256=digest(Path(__file__).with_name('compare_moe_normal.py')),
        scope='Optimistic direct path: zero lookup/return-copy timing, 44KiB available DMA budget; no extra memory bandwidth',
        tasks=[dict(fixture=f,label=l,architectures=[dict(path=str(p),sha256=digest(p)) for p in ps]) for f,l,ps in tasks])
    save(root/'bypass/execution_plan.json',plan)
    def run(task):
        f,label,paths=task;print('Starting '+f+'_'+label,flush=True)
        try:
            c=run_comparison(binary,source/f/'workload.json',source/f/'golden.json',paths,
                root/'bypass/comparisons'/(f+'_'+label),repeats=2,hbm_channels=8,timeout=1200,workers=4,
                **base['numerical_tolerance'])
            return dict(status='passed',timings_ps={r['architecture']['name']:r['result']['total_ps'] for r in c['comparisons']})
        except Exception as e: return dict(status='failed',error=str(e))
    outcomes={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(run,t):t[0]+'_'+t[1] for t in tasks}
        for future in as_completed(futures):
            name=futures[future];outcomes[name]=future.result();print(name+': '+outcomes[name]['status'],flush=True)
            save(root/'bypass/status.json',dict(status='running',all_gates_passed=False,outcomes=outcomes))
    passed=len(outcomes)==len(tasks) and all(o['status']=='passed' for o in outcomes.values())
    require(digest(binary)==plan['executable_sha256'],'binary identity changed')
    save(root/'bypass/status.json',dict(status='passed' if passed else 'failed',all_gates_passed=passed,outcomes=outcomes))
    require(passed,'direct-path control failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','source','binary']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(),a.source.resolve(),a.binary.resolve())
