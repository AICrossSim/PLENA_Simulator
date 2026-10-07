#!/usr/bin/env python3
"""Verify Step2-disabled paths against immutable accepted native results."""
import argparse, concurrent.futures, gzip, json, os, subprocess, tempfile
from pathlib import Path
import step0 as b
import step2 as s
import diagnose_step1 as d


def main():
    p=argparse.ArgumentParser();p.add_argument('--step2',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    root=args.step2.resolve();out=root/'disabled_regression';out.mkdir(exist_ok=True);cases=[]
    for source in [s.COMMON,s.COMMON/'fixed_ownership',s.ACCEPTED]:
        for c in b.read(source/'manifest.json')['cases']:
            if source==s.COMMON and c['policy']=='threshold' and c['workload_kind']!='me1':continue
            cases.append((source,c))
    validator=d.load_validator(root/'repro/compare_moe_normal.py')
    def run(pair):
        source,c=pair;folder=out/('accepted__' if source==s.ACCEPTED else 'common__')/c['id'];folder.mkdir(parents=True,exist_ok=True)
        old=b.read(source/c['id']/'rep1.json');a,w,g=[b.read(c[k]) for k in ['architecture','workload','golden']]
        times=[]
        for rep in [1,2]:
            dest=folder/f'rep{rep}.json.gz'
            if not dest.exists():
                with tempfile.TemporaryDirectory(prefix='plena-disabled-',dir='/tmp') as tmp:
                    raw=Path(tmp)/'result.json'
                    with (folder/f'rep{rep}.log').open('w') as log:
                        subprocess.run([str(root/'repro/moe_dual_normal'),'--architecture',c['architecture'],'--workload',c['workload'],
                            '--output',str(raw),'--hbm-channels','8','--max-hbm-bytes',str(1<<30)],stdout=log,stderr=subprocess.STDOUT,
                            env=dict(os.environ,LD_LIBRARY_PATH=str(root/'repro')),timeout=1800,check=True)
                    e=b.read(raw);s.save_gzip(dest,e)
            else:e=s.read(dest)
            validator.validate_run(e,g,w,a,0,0);d.require_bit_exact(e['result'],g)
            b.require(old['result']==e['result'],'old result changed '+c['id'])
            b.require(old['memory_model']['calibration']==e['memory_model']['calibration'],'native counters changed '+c['id'])
            times.append(e['result']['total_ps'])
        b.require(times[0]==times[1],'regression repeat changed')
        b.save(folder/'validation.json',dict(status='passed',reference=str(source/c['id']/'rep1.json'),
            reference_sha256=b.digest(source/c['id']/'rep1.json'),all_old_result_fields_exact=True,native_counters_exact=True,
            outputs={q.name:b.digest(q) for q in folder.glob('*.json.gz')}))
        return dict(case=c['id'],source=str(source),time_us=times[0]/1e6,repeats=2)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(run,cases):rows.append(r);print('PASS',r['case'],r['time_us'],flush=True)
    b.csv_write(out/'measurements.csv',rows);b.save(out/'validation.json',dict(status='passed',points=len(rows),runs=2*len(rows),failed_invariants=0))

if __name__=='__main__':main()
