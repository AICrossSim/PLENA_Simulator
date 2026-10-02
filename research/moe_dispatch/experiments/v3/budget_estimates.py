#!/usr/bin/env python3
"""Separate, explicitly labeled closed-form estimates; never simulation results."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'v3_reference'))
import budget_v3 as b
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study import csvwrite

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);out=[]
 for split in ('development','heldout','constructed','mixed_development','mixed_heldout'):
  if not (root/'inputs'/f'{split}.json').exists():continue
  for w in json.loads((root/'inputs'/f'{split}.json').read_text())['workloads']:
   ms=[e['Me'] for e in w['experts'] if not e['is_shared']]
   for org in ('6','3+3','4+2'):
    for prec,fmt in [('P0',b.BF16),('P1',b.FMT_V3),('P2',b.FMT_V3)]:
     for bw in (128,256,512):
      r=b.predict_layer(b.MODELS['dsv2_lite'],w['batch'],ms,fmt,b.v3_cores(org,prec),prec,bw)
      provenance=w.get('provenance','captured_decode')
      if isinstance(provenance,dict):provenance=provenance['origin']
      out.append(dict(kind='closed_form_first_order_not_simulated',format_status='reference_default_not_quality_selected',split=split,provenance=provenance,workload=w['id'],org=org,precision=prec,bw=bw,estimate_cycles=r['time'],estimate_ms=r['time']/1e6,HBM_lower_bound_cycles=r['t_hbm'],onchip_fluid_cycles=r['t_onchip'],bound=r['bound']))
 csvwrite(root/'analytical_budget_estimates.csv',out)
 print(len(out))
if __name__=='__main__':main()
