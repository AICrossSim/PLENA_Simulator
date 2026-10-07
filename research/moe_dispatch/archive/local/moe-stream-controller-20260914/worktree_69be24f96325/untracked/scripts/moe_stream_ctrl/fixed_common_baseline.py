#!/usr/bin/env python3
"""Freeze the common cohort work-conserving ownership, then replay both controllers."""
import argparse,concurrent.futures,copy,shutil,json
from pathlib import Path
import common_baseline as common
import step0 as b
import diagnose_step1 as d


def main():
 p=argparse.ArgumentParser();p.add_argument('--common',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
 src=args.common.resolve();out=src/'fixed_ownership';old=b.read(src/'manifest.json')
 if not (out/'manifest.json').exists():
  out.mkdir(exist_ok=True);(out/'repro').mkdir(exist_ok=True)
  for p in (src/'repro').iterdir():
   if p.is_file():shutil.copy2(p,out/'repro'/p.name)
  shutil.copy2(__file__,out/'repro'/Path(__file__).name)
  cases=[];oldcases={c['id']:c for c in old['cases']}
  for workload in ['b8','b32']:
   for org in ['single','homogeneous','heterogeneous']:
    ref=src/f'{workload}__{org}__work_conserving__cohort'/'rep1.json'
    refresult=b.read(ref)['result'];refarch=b.read(ref.parent/'architecture.json');order=d.job_order(refresult,refarch)
    for mode in ['cohort','charged_n3']:
     original=oldcases[f'{workload}__{org}__work_conserving__{mode}'];c=copy.deepcopy(original)
     c['id']=f'{workload}__{org}__fixed__{mode}';c['policy']='fixed';folder=out/c['id'];folder.mkdir()
     a=b.read(original['architecture']);a['name']=c['id'];a['diagnostic']['fixed_job_order']=order
     b.save(folder/'architecture.json',a);c['architecture']=str(folder/'architecture.json')
     c['hashes']['architecture']=b.digest(c['architecture']);c['assignment_reference']=str(ref)
     c['hashes']['assignment_reference']=b.digest(ref);cases.append(c)
  m=dict(status='prepared',cases=cases,repeats=2,planned_runs=24,artifacts={p.name:b.digest(p) for p in (out/'repro').iterdir()},
    assignment_rule='freeze Step1 cohort work-conserving owner and within-core order per input/organization',accepted_source=old['accepted_source'])
  b.save(out/'manifest.json',m)
 else:m=b.read(out/'manifest.json')
 v=d.load_validator(out/'repro/compare_moe_normal.py');rows=[];cores=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
  futures={ex.submit(common.run_case,c,out,m,v):c for c in m['cases']}
  for f in concurrent.futures.as_completed(futures):
   rr,cc=f.result();c=futures[f];a=b.read(c['architecture']);r=b.read(out/c['id']/'rep1.json')['result']
   b.require(d.job_order(r,a)==a['diagnostic']['fixed_job_order'],'frozen ownership/order changed')
   ref=b.read(c['assignment_reference'])['result']
   for n,o in zip(r['cores'],ref['cores']):
    for key in ['hbm_read_bytes','useful_macs','issued_macs','jobs']:b.require(n[key]==o[key],'fixed per-core work changed')
   if c['mode']=='cohort':b.require(r['total_ps']==ref['total_ps'],'frozen cohort timing changed')
   rows+=rr;cores+=cc;print('PASS',c['id'],rr[0]['time_us'],flush=True)
 rows.sort(key=lambda r:(r['case'],r['repeat']));cores.sort(key=lambda r:(r['case'],r['core']))
 b.csv_write(out/'measurements.csv',rows);b.csv_write(out/'core_services.csv',cores)
 import csv
 with (src/'measurements.csv').open() as f: prior=list(csv.DictReader(f))
 values={r['case']:r for r in prior if r['repeat']=='1' and r['policy']=='work_conserving'}
 values.update({r['case']:r for r in rows if r['repeat']==1})
 text=['# Step1 common baseline for Steps 2–6','',
 'Requested controls: cohort / one control port / rotating, and 2/3/2 charged N3. Fixed ownership is obtained by freezing the Step1 cohort work-conserving owner **and within-core order** for each input/organization, then replaying both controllers on that same map. The earlier threshold runs are separate diagnostics; they are not substituted for this fixed map.', '',
 '| Input | Organization | Dispatch | Cohort time (us) | / entry bound | Charged N3 time (us) | / entry bound |',
 '|---|---|---|---:|---:|---:|---:|']
 for workload in ['b8','b32']:
  for org in ['single','homogeneous','heterogeneous']:
   for policy in ['fixed','work_conserving']:
    a,z=[values[f'{workload}__{org}__{policy}__{mode}'] for mode in ['cohort','charged_n3']]
    text.append(f"| {workload} | {org} | {policy} | {float(a['time_us']):.6f} | {float(a['lower_bound_ratio']):.4f} | {float(z['time_us']):.6f} | {float(z['lower_bound_ratio']):.4f} |")
 text+=['','B8 entry lower bound: 235.008 us. B32: 317.952 us. Native 8 channels, 32 B, 1 ns/channel issue cadence; no HBM/DMA 2x here. Every requested point repeated twice; 48 native runs in the requested 24-point table. The 24 threshold diagnostics and two Me1 calibration runs are additional. All 74 runs pass numerical, native-byte, budget, drain and exact-repeat gates.', '',
 'Me1 single-cohort: 32.722 us, 768 tiles, load latency sum 81,069,000 ps, mean **105.55859375 ns**, minimum 53 ns, maximum 543 ns. Default 4L rounds upward to **422,235 ps** (422.235 ns). Thus measured mean is about 106 ns; several hundred ns describe the tail, not the mean. Calibration is frozen before Step2.', '',
 'Per-case/core L values for the fixed maps are in `fixed_ownership/core_services.csv`; work-conserving values and Me1 are in `core_services.csv`. Step2 uses matched baseline load sum/count, with threshold `ceil(multiplier * sum / count)` so no mean-rounding drift occurs.', '',
 'Per-core fixed ownership matches the cohort reference exactly, including padded MAC counts and HBM byte shares. Fixed cohort timing exactly reproduces its work-conserving source. Charged N3 may have different dynamic owners, so both fixed and dynamic comparisons are retained. All architecture JSON files and native results are archived with provenance.', '',
 'Step1 Me32 accumulator_dependency_stall 0.519 -> 1.587 us is retained as a Step4 target. No Step2 change should alter the K-dependency rule.', '']
 (src/'COMMON_BASELINE.md').write_text('\n'.join(text))
 m.update(status='complete',completed_runs=24,failed_invariants=0,hbm_bytes_changed_runs=0);b.save(out/'manifest.json',m)
 b.save(src/'common_baseline_index.json',dict(status='complete',requested_points=24,requested_runs=48,total_runs_including_threshold_and_me1=74,
  work_conserving_manifest=str(src/'manifest.json'),fixed_manifest=str(out/'manifest.json'),report=str(src/'COMMON_BASELINE.md')))
 print('COMMON TABLE COMPLETE',flush=True)

if __name__=='__main__':main()
