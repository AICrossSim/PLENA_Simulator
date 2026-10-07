#!/usr/bin/env python3
"""Explicitly assumed latency/II and finite weight-bandwidth sensitivity.
These are not synthesis calibration, HBM clock changes, or native Ramulator runs.
"""
import csv,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
import run_routes as r

def main():
    windows=json.loads((r.OUT/'inputs/route_windows.json').read_text());plan=[]
    for w in windows:
        if w['split']!='primary' or w['dataset']!='swe':continue
        for shape in r.SHAPES:
            for mode in r.MODES:
                # All five organizations under equal total decoded weight rate.
                for rate in [256,512,2048,4096]:
                    variant=f'weight_bpc{rate}'
                    name=f"sensitivity_{w['name']}__{'_'.join(map(str,shape))}__{mode}__{variant}"
                    req=r.helper.req(name,shape,r.jobs(w,'routed_gate_up'),mode,dict(control='tile_cohort',weight_bpc=rate))
                    plan.append(dict(name=name,window=w['name'],model=w['model'],dataset=w['dataset'],split=w['split'],batch=w['batch'],
                        phase='routed_gate_up',shape='+'.join(map(str,shape)),mode=mode,variant=variant,request=req))
                if shape not in r.SHAPES[:3]:continue
                for latency,interval in [(1,1),(25,25)]:
                    for variant in ['pure','finite']:
                        name=f"pipeline_{w['name']}__{'_'.join(map(str,shape))}__{mode}__L{latency}_II{interval}__{variant}"
                        req=r.helper.req(name,shape,r.jobs(w,'routed_gate_up'),mode,dict(control='tile_cohort'))
                        req['compute'].update(result_latency_cycles=latency,issue_interval_cycles=interval)
                        plan.append(dict(name=name,window=w['name'],model=w['model'],dataset=w['dataset'],split=w['split'],batch=w['batch'],
                            phase='routed_gate_up',shape='+'.join(map(str,shape)),mode=mode,variant=variant,
                            latency=latency,interval=interval,request=req))
    dest=r.OUT/'sensitivity';dest.mkdir(exist_ok=True)
    r.save(dest/'plan.json',dict(points=len(plan),repeats=2,calibrated=False,
        note='Changing pipeline L/II also changes required finite result registers under the declared ceil(L/II) contract; report that budget.'))
    start=time.monotonic();rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(r.execute,t):t['name'] for t in plan}
        for f in as_completed(fs):
            try:rows.append(f.result())
            except Exception as e:
                r.save(dest/'FAILURE.json',dict(name=fs[f],error=str(e)))
                for x in fs:x.cancel()
                raise
            r.save(dest/'progress.json',dict(done=len(rows),planned=len(plan),elapsed_s=time.monotonic()-start))
            if len(rows)%20==0:print(len(rows),len(plan),flush=True)
    keys=sorted(set().union(*(x.keys() for x in rows)))
    with (dest/'all_points.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    r.save(dest/'validation.json',dict(passed=True,points=len(rows),runs=len(rows)*2,repeat_identical=True))
if __name__=='__main__':main()
