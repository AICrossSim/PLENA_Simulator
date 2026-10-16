"""Validate delivered tables against per-window records, budgets and receipts."""
import hashlib,subprocess,sys,importlib.metadata
import numpy as np
from .common import *
from .inputs import load
from .evaluate import selections
from .report import load_results,aggregate,strongest_map

def main():
 required=['E1/cells.csv','E2/online_compare.csv','E2/workload_balance.csv','E2/SCHEDULER.md','E3/winning_cells.md','REPORT_ZH.md','figures/fig_cells.pdf','README.md']
 for name in required:assert (ROOT/name).is_file() and (ROOT/name).stat().st_size>0,name
 ws=load();held={w['id']:w for w in ws['heldout']};rows=load_results();assert len(rows)==5*5*105*4
 delta=read_json(ROOT/'E2/delta_selection.json')['delta']
 for r in rows:
  w=held[r['window_id']];useful=2*3*2048*sum(e['Me']*e['F'] for e in w['experts'])
  assert useful==r['useful_flops']
  assert r['weight_bytes']==sum(6*2048*e['F'] for e in w['experts'])
  assert r['cycles']>=max(r['core_service'])
  assert 0<r['mac_util_pct']<=100 and 0<=r['padding_waste_pct']<100
  assert r['offline_lb_cycles']<=r['cycles']
  assert r['delta']==delta
  assert 'MODEL_INVALID' not in r['solver_status'] and 'INFEASIBLE' not in r['solver_status']
  assert abs(r['req_weight_TBps']-r['weight_bytes']/r['cycles']/1000)<1e-12
  assert sum(r['pool_work'])==sum(r['core_work'])==useful//2
  if r['scheduler']=='online':assert abs(sum(r['target_work'])-useful/2)<=1e-4
 for s in selections():
  used=sum(c['pm']*c['pn']*c['pk'] for c in s['cores']);assert used==s['used_macs']<=s['budget']
  assert used==s['budget'] or (s['org'] in ('H','H+') and len(s['cores']) in (12,13))
  assert not s['selection_uses_heldout']
  for b in BATCHES:
   rr=[r for r in rows if r['macs']==s['budget'] and r['org']==s['org'] and r['batch']==b]
   assert len(rr)==60 and len({r['geometry'] for r in rr})==1
 recomputed,_,_=aggregate(rows,strongest_map());saved=read_csv(ROOT/'E1/cells.csv');assert len(saved)==len(recomputed)==175
 for a,b in zip(saved,recomputed):
  for k,v in b.items():
   if isinstance(v,float):assert float(a[k])==v,(k,a[k],v)
   else:assert a[k]==str(v),(k,a[k],v)
 for m in BUDGETS:
  for ch in (0,64):
   r=read_json(ROOT/'E1/search'/f'{m}_{ch}'/'receipt.json');assert r['grid_repeats']==2 and r['grid_bitwise_equal'] and r['refined_repeats']==2
 receipts=list((ROOT/'E1/receipts').glob('*.json'))
 for p in receipts:
  r=read_json(p);assert r['repeats']==2 and r['bitwise_equal']
  assert digest(read_json(ROOT/'E1/windows'/p.with_suffix('.json.gz').name))==r['rows_digest']
 assert len(read_csv(ROOT/'E1/per_batch_hardware_reference.csv'))==175
 # Explicitly fail if any frozen Round2..Round5 tracked file has changed.
 repo=ROOT.parents[2]
 for d in ('round2','round3','round4','round5_largebatch_compute'):
  path=f'research/moe_dispatch/{d}'
  r=subprocess.run(['git','diff','b095a7de67a0be07802df0fb1fa2c2b3063bdc75','--',path],cwd=repo,capture_output=True,text=True,check=True)
  assert not r.stdout,f'frozen files changed: {d}'
 result=dict(passed=True,input_windows={s:len(v) for s,v in ws.items()},main_cells=175,main_window_scheduler_rows=len(rows),
  frozen_designs=25,per_batch_designs=175,receipt_files=len(receipts),all_points_repeated_exactly=True,
  all_required_files_present=True,frozen_round2_to_round5_unchanged=True,
  python=sys.version,versions={n:importlib.metadata.version(n) for n in ('numpy','scipy','ortools','matplotlib','pytest')},
  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.iterdir()) if p.suffix in ('.py','.cpp')},
  output_sha256={x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in required if x!='README.md'})
 write_json(ROOT/'VALIDATION.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
