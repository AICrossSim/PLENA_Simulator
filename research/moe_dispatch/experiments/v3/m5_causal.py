#!/usr/bin/env python3
"""Small real-calibration Rust integration: causal lambda, two reset traces.

This is additional development-only control integration, not held-out novelty
evidence or a replacement for the preregistered static-rank policy ablations.
"""
import argparse,copy,json,math,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import study

def bf16_rne(value):
 bits=struct.unpack('<I',struct.pack('<f',float(value)))[0]
 rounded=(bits+0x7fff+((bits>>16)&1))&0xffff0000
 return struct.unpack('<f',struct.pack('<I',rounded))[0]

def compact_table(table):
 keys=('rank_candidates','tail_energy_per_projection','factor_bytes_per_projection','capacity_per_projection')
 result={k:copy.deepcopy(table[k]) for k in keys}
 for name in ('gate','up','down'):
  assert len(result['tail_energy_per_projection'][name])==64
  for values in result['tail_energy_per_projection'][name]:
   assert len(values)==5 and all(math.isfinite(x) and x>=0 for x in values)
   values[:]=[bf16_rne(x) for x in values]
  for values in result['factor_bytes_per_projection'][name]:assert len(values)==5 and all(math.isfinite(x) and x>=0 and x==round(x) for x in values)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--binary',required=True);ap.add_argument('--rank-table',required=True);ap.add_argument('--format-config',default='');a=ap.parse_args();root=Path(a.root)
 if a.format_config:study.load_format(a.format_config)
 table=json.loads(Path(a.rank_table).read_text());assert table.get('physical_energy_storage','').startswith('BF16 RNE'),'M5 must use lambda0 refit on the physical BF16 LUT, not float-table calibration'
 energy=compact_table(table);rank=table['uniform_reference_rank'];assert rank<32,'Maximum-rank budget has no redistribution headroom'
 cfg=study.config('BL4','OP2');assert cfg['main_bits']==table['bits'] and cfg['factor_a']=='mxint4' and cfg['factor_b']=='bf16'
 for k,nseg in [('gate',4),('up',4),('down',3)]:assert max(table['capacity_per_projection'][k])<=cfg['rank_lanes']*nseg
 cases=json.loads((root/'inputs/development.json').read_text())['workloads'];selected=[]
 for dataset in ('bfcl','gpqa','swe'):
  for batch in (2,16):selected.append(next(w for w in cases if w['batch']==batch and f'joint_test_{dataset}_' in w['id']))
 outroot=root/'m5_causal';outroot.mkdir(parents=True,exist_ok=True)
 signature={'binary_sha256':study.sha(a.binary),'study_sha256':study.sha(study.__file__),'m5_sha256':study.sha(__file__),'rank_table_sha256':study.sha(a.rank_table),'development_inputs_sha256':study.sha(root/'inputs/development.json')}
 signature.update(physical_format_sha256=study.digest(study.physical_format()),physical_format=study.physical_format())
 study.frozen(outroot/'frozen_rank_table.json',table)
 # Packed implementation: BF16 energy + uint32 byte cost per entry, uint16
 # capacity per projection/expert, candidate ranks and lambda state.
 table_bytes=64*3*5*(2+4)+64*3*2+5*2+32
 lam0=float(table['lambda0']);rows=[];sequence=[[],[]]
 # Execute two complete causal sequences from lambda0. Each individual point
 # still has the required two raw repetitions, rather than manufacturing a
 # second sequence by copying the first sequence's rows.
 for reset in (0,1):
  lam=lam0;previous_budget=None
  for order,w0 in enumerate(selected):
   w=copy.deepcopy(w0)
   for e in w['experts']:
    if not e['is_shared']:e['gate_weights']=list(e['route_scores'])
   static=study.point(w,'BL4','OP2',suite='m5_causal',variant=f'order{order}_uniform{rank}',changes={'ranks':{'routed':[min(rank,cfg['rank_lanes']*n) for n in (4,4,3)],'shared':cfg['ranks']['shared']},'rank_alloc':'static'})
   b=study.run_point(static,outroot,a.binary,signature,3600);assert b['status']=='complete',b.get('reason');baseline=b['report'];budget=int(baseline['routed_rank_factor_bytes'])
   reset_reason='sequence_start' if previous_budget is None else ('routed_budget_changed' if previous_budget!=budget else 'carry_same_budget')
   if previous_budget!=budget:lam=lam0
   changes={'rank_alloc':'gate_weighted','rank_selection':'common','rank_energy_table':energy,'rank_energy_table_bytes':table_bytes,'rank_decision_cycles':15,'lambda0':lam0,'lambda_before':lam,'lambda_bounds':table['lambda_bounds'],'lambda_eta':table['lambda_eta'],'rank_budget_bytes':budget,'rank_budget_scope':'routed'}
   dynamic=study.point(w,'BL4','OP2',suite='m5_causal',variant=f'reset{reset}_order{order}_causal',changes=changes)
   item=study.run_point(dynamic,outroot,a.binary,signature,3600);assert item['status']=='complete',item.get('reason');r=item['report'];assert r['lambda_before']==lam
   assert r['rank_budget_scope']=='routed' and r['shared_rank_factor_bytes']==baseline['shared_rank_factor_bytes']
   row={'order':order,'workload':w['id'],'tokens':w['batch'],'lambda_before':r['lambda_before'],'lambda_after':r['lambda_after'],'lambda0_calibration':lam0,'lambda_reset_reason':reset_reason,'factor_budget_scope':'routed_only','factor_budget_bytes':budget,'factor_bytes':r['rank_factor_bytes'],'routed_factor_bytes':r['routed_rank_factor_bytes'],'shared_fixed_factor_bytes':r['shared_rank_factor_bytes'],'factor_ratio':r['routed_rank_factor_bytes']/budget,'uniform_cycles':baseline['cycles'],'dynamic_cycles':r['cycles'],'time_ratio':r['cycles']/baseline['cycles'],'rank_table_bytes':table_bytes,'rank_table_reservation':cfg['control_reserve'],'control_cycles':sum(c['control_cycles'] for c in r['cores']),'rank_lanes':cfg['rank_lanes'],'table_sha256':signature['rank_table_sha256'],'quality_status':'timing/control integration; accuracy qualification requires hardware-BF16-LUT Q3','bindings':r['bindings']}
   if reset==0:rows.append(row)
   sequence[reset].append({'lambda_before':r['lambda_before'],'lambda_after':r['lambda_after'],'raw_sha256':item['receipt']['sha256']})
   lam=r['lambda_after'];previous_budget=budget;print(reset,order,w['id'],reset_reason,row['factor_ratio'],row['time_ratio'],flush=True)
 assert sequence[0]==sequence[1]
 for before,after in zip(rows[1:],rows[:-1]):assert before['lambda_before']==(after['lambda_after'] if before['factor_budget_bytes']==after['factor_budget_bytes'] else lam0)
 study.write(outroot/'causal_sequence.json',{'signature':signature,'scope':'six actual development route windows, layer13; CPU-calibrated real weights; analytical Rust timing','energy_storage':'BF16 round-to-nearest-even before candidate objective; cost uint32','sequence':rows,'repeat_traces':sequence,'two_reset_sequences_identical':True,'causal_lambda_continuity':True,'lambda_budget_change_rule':'reset to calibration lambda0 whenever uniform-active routed factor budget changes; carry only at identical budget','only_lambda_carries_across_same_budget_windows':True,'primary_six_have_budget_change_every_window':all(r['lambda_reset_reason']!='carry_same_budget' for r in rows),'no_convergence_claim':True,'no_new_accuracy_or_heldout_claim':True})
 study.csvwrite(outroot/'causal_sequence.csv',[{k:v for k,v in r.items() if k!='bindings'} for r in rows])
 # This is a separately labeled temporal replay of one real BFCL B2 window,
 # not another independent sample or a change to the primary six-window result.
 # It exercises the same-budget carry wire, which the six distinct active-expert
 # budgets above cannot exercise under the strict Q3 reset rule.
 replay_w=copy.deepcopy(selected[0])
 for e in replay_w['experts']:
  if not e['is_shared']:e['gate_weights']=list(e['route_scores'])
 replay_static=study.point(replay_w,'BL4','OP2',suite='m5_causal',variant=f'order0_uniform{rank}',changes={'ranks':{'routed':[min(rank,cfg['rank_lanes']*n) for n in (4,4,3)],'shared':cfg['ranks']['shared']},'rank_alloc':'static'})
 replay_base=study.run_point(replay_static,outroot,a.binary,signature,3600)['report'];replay_budget=int(replay_base['routed_rank_factor_bytes']);replay=[[],[]]
 for reset in (0,1):
  lam=lam0
  for step in (0,1):
   changes={'rank_alloc':'gate_weighted','rank_selection':'common','rank_energy_table':energy,'rank_energy_table_bytes':table_bytes,'rank_decision_cycles':15,'lambda0':lam0,'lambda_before':lam,'lambda_bounds':table['lambda_bounds'],'lambda_eta':table['lambda_eta'],'rank_budget_bytes':replay_budget,'rank_budget_scope':'routed'}
   p=study.point(replay_w,'BL4','OP2',suite='m5_temporal_replay',variant=f'reset{reset}_step{step}_same_real_window',changes=changes)
   item=study.run_point(p,outroot,a.binary,signature,3600);assert item['status']=='complete',item.get('reason');r=item['report']
   assert r['lambda_before']==lam and r['rank_budget_scope']=='routed'
   assert r['shared_rank_factor_bytes']==replay_base['shared_rank_factor_bytes']
   replay[reset].append({'step':step,'workload':replay_w['id'],'lambda_before':r['lambda_before'],'lambda_after':r['lambda_after'],'lambda_reset_reason':'sequence_start' if step==0 else 'carry_same_budget','routed_budget_bytes':replay_budget,'routed_spent_bytes':r['routed_rank_factor_bytes'],'cycles':r['cycles'],'raw_sha256':item['receipt']['sha256']})
   lam=r['lambda_after']
 assert replay[0]==replay[1] and replay[0][1]['lambda_before']==replay[0][0]['lambda_after']
 study.write(outroot/'temporal_replay.json',{'signature':signature,'provenance':'temporal_replay_of_one_real_development_BFCL_B2_window','independent_requests_added':0,'included_in_primary_six_or_N5_gain':False,'purpose':'test actual same-budget causal lambda carry; no convergence or generalization claim','two_actual_reset_sequences':replay,'repeats_per_point':2,'original_raw_reports_identical_across_reset_sequences':True})
 study.csvwrite(outroot/'temporal_replay.csv',replay[0])
if __name__=='__main__':main()
