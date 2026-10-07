#!/usr/bin/env python3
"""Read-only independent cross-table audit; never simulates or alters source/results."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,math
P=argparse.ArgumentParser();P.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);P.add_argument('--out',type=Path);a=P.parse_args()
if a.out is None:a.out=a.root/'results/FINAL_TABLE_AUDIT.json'
R=a.root/'results';checks=[];warnings=[];missing=[];files={}
def fail(ok,message):
 checks.append({'ok':bool(ok),'check':message})
def read(name):
 p=R/name
 if not p.exists():missing.append(name);return []
 files[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 with p.open() as f:return list(csv.DictReader(f))
def near(x,y):return abs(x-y)<=max(1e-8,1e-9*max(abs(x),abs(y)))
def gm(v):return math.exp(sum(map(math.log,v))/len(v))
B=(2,4,8,16,64,96,128);M=('pipelined','port_tight','fixed_issue');bw=256*32/65
raw=read('E4/per_window.csv');table=read('E4/heldout_main_table.csv')
if raw:
 expected=dict(json.loads((R/'E0/frozen_inputs.json').read_text())['heldout_batch_counts'])
 groups={}
 for row in raw:
  k=(row['entry'],row['onchip_mode'],row['sched_type']);groups.setdefault(k,[]).append(row)
  cycles=float(row['cycles']);latency=float(row['latency_ms']);hbm=float(row['hbm_bytes']);unique=float(row['native_unique_bytes'])
  fail(near(cycles/1e6,latency),str(k)+' cycles/ms unit')
  fail(hbm>=unique-1e-6,str(k)+' HBM >= unique useful weights')
  fail(hbm/cycles<=bw+1e-6,str(k)+' effective HBM below shared credit cap')
  fail(0<float(row['spatial_util'])<=1+1e-10,str(k)+' spatial utilization legal')
  for f in ('core0_compute_busy','core1_compute_busy','w_port_busy','x_port_busy','acc_port_busy','core_finish_gap'):
   fail(0<=float(row[f])<=cycles+1e-4,str(k)+' occupancy/gap <= wall: '+f)
  fail(0<=float(row['hbm_busy_frac'])<=1+1e-9,str(k)+' HBM occupancy legal')
  if row['entry'] not in ('U1','U2'):
   d=json.loads(row['design']);fail(sum(c['pm']*c['pn']*c['pk'] for c in d['cores'])==12288,str(k)+' multiplier budget frozen')
 for k,rows in groups.items():
  fail(len(rows)==135,str(k)+' complete135window cohort')
  fail(dict(Counter(row['batch'] for row in rows))==expected,str(k)+' perbatch counts frozen')
  fail(len({row['window_id'] for row in rows})==135,str(k)+' no duplicate window IDs')
 for t in table:
  k=(t['entry'],t['onchip_mode'],t['sched_type']);rows=groups[k]
  for b in B:fail(near(float(t['B'+str(b)]),gm([float(r['latency_ms']) for r in rows if int(r['batch'])==b])),str(k)+' batchGM cross-table B'+str(b))
  fail(near(float(t['all_geomean']),gm([float(r['latency_ms']) for r in rows])),str(k)+' allGM cross-table')
  for base in ('B1','B2'):
   r0={r['window_id']:float(r['cycles']) for r in groups[(base,k[1],k[2])]}
   v=gm([float(r['cycles'])/r0[r['window_id']] for r in rows]);fail(near(float(t['ratio_vs_'+base]),v),str(k)+' pairedratio vs '+base)
 bounds=read('E1/bounds_per_window.csv')
 for r in bounds:
  for f in ('hbm_floor_unique','hbm_floor_actual','mac_floor','port_floor','task_floor','bound'):
   fail(float(r[f])<=float(r['latency_ms'])+1e-9,'E1 '+r['window_id']+'/'+r['design']+'/'+r['onchip_mode']+' legal '+f)
mic=read('E2/micro.csv')
if mic:
 fail(len(mic)==1188,'E2 expected6shapes*3flows*2expert types*11Me*3modes')
 fail(len({tuple(sorted(r.items())) for r in mic})==1188,'E2 no duplicated micro rows')
 fail(all(0<float(r['spatial_util'])<=1+1e-9 and float(r['cycles'])>0 for r in mic),'E2 finite positive latency and utilization')
for mode in M:
 p=R/f'E3/bnb_{mode}_A.json'
 if not p.exists():missing.append(str(p.relative_to(R)));continue
 files[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();obj=json.loads(p.read_text())
 for k,f in obj['families'].items():
  fail(f['covered_lattice_points']+sum(v['lattice_points'] for v in f['open_regions'])==f['declared_lattice_points'],mode+'/'+k+' BnB lattice conservation')
  fail(bool(f['proof_complete'])==(not f['open_regions']),mode+'/'+k+' proofstate matches openfrontier')
  if not f['proof_complete']:warnings.append(mode+'/'+k+' is best-evaluated with open proof; not certified global optimum')
ps=read('E5/predictor_table.csv')
if ps:
 fail(len(ps)==36,'E5 two designs*three modes*six predictors')
 fail(all(float(r['mae_pct'])>=0 and float(r['e2e_ratio_vs_oracle'])>0 for r in ps),'E5 prediction errors and reference ratios legal')
 fail(all(r.get('oracle_caveat') for r in ps),'E5 profile-guided oracle caveat retained')
whole=read('E6/model_token_e2e.csv')
if whole:
 fail(all(not r['token_ms'] and not r['non_moe_ms_per_layer'] and r['status']=='missing_matching_DeepSeek_non_MoE_layer_timing' for r in whole),'E6 matching nonMoE unavailable; whole-model fields must remain missing')
 warnings.append('E6 complete-model inference remains unavailable; table is honest MoE-layer evidence only')
obj={'scope':'Independent read-only numerical/table integrity; not RTL calibration or verification of analytical approximation','checks':len(checks),'passed':sum(c['ok'] for c in checks),'failed':[c for c in checks if not c['ok']],'missing_inputs':missing,'warnings':warnings,'input_sha256':files,'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:obj[k] for k in ('checks','passed','failed','missing_inputs','warnings')}));raise SystemExit(bool(obj['failed']))
