#!/usr/bin/env python3
"""Fixed-threshold evidence summaries; missing populations stay unavailable.

This is a timing-evidence assessment, conditional on Q-line validation. It cannot
promote shape-only numerical claims or constructed routes to real mixed evidence.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from report import rows,gm
from study import write,csvwrite

def n4_verdict(time_ratio,traffic_ratio,default_area_difference,area_robust):
 """Original nominal threshold plus explicitly separate type sensitivity."""
 latency=time_ratio<=.95;traffic=traffic_ratio<=.85
 nominal=latency or (traffic and default_area_difference is not None and default_area_difference<=.05)
 return dict(passed=nominal,nominal_passed=nominal,latency_threshold_pass=latency,area_type_robust_pass=traffic and area_robust,area_based_claim_type_conditional=nominal and not latency and not area_robust,unconditional_organization_evidence=latency or (traffic and area_robust))

def compensation_assessment(evidence,expected_windows=120,expected_focus=6,modes=None):
 """Apply N2 only after every preregistered real-window comparison exists."""
 if modes is None:modes={'OP5':('lanes','separate','kext','offload'),'OP2':('lanes',)}
 populations={};all_complete=True
 for op,variants in modes.items():
  for mode in variants:
   group=[r for r in evidence if r['op']==op and r['mode']==mode]
   focus=[r for r in group if r['provenance']=='captured_mixed' and r['tokens']==96 and r['shared_me']>=16]
   busy=[r['sum_core_busy_overhead'] for r in group if r.get('sum_core_busy_overhead') is not None]
   cold=[r['cold_throughput_loss'] for r in group if r.get('cold_throughput_loss') is not None]
   mult=[r['rank_multiplier_overhead'] for r in group]
   complete=len(group)==expected_windows and len(focus)==expected_focus and len(busy)==expected_windows
   all_complete &= complete
   item={'paired_windows':len(group),'required_paired_windows':expected_windows,'T96_focus_windows':len(focus),'required_T96_focus_windows':expected_focus,'complete':complete,'equal_wire_bytes':all(r['equal_wire_bytes'] for r in group),'sum_core_busy_overhead_max':max(busy,default=None),'T96_layer_overhead_max':max((r['layer_overhead'] for r in focus),default=None),'cold_throughput_loss_max':max(cold,default=None),'rank_multiplier_overhead_max':max(mult,default=None)}
   if mode=='lanes':
    item['threshold_pass']=bool(busy) and max(busy)<=.03 and bool(focus) and max(r['layer_overhead'] for r in focus)<=.03 and bool(mult) and max(mult)<=.032
   else:
    item['witnesses']=[dict(workload=r['workload'],busy=r['sum_core_busy_overhead']>=.10 if r.get('sum_core_busy_overhead') is not None else False,T96_layer=r in focus and r['layer_overhead']>=.05,cold=r.get('cold_throughput_loss',0)>=.10) for r in group if (r.get('sum_core_busy_overhead',0)>=.10 or (r in focus and r['layer_overhead']>=.05) or r.get('cold_throughput_loss',0)>=.10)]
    item['threshold_pass']=bool(item['witnesses'])
   populations[f'{op}/{mode}']=item
 separate={}
 for op in modes:
  items=[item for key,item in populations.items() if key.startswith(op+'/')]
  separate[op]=all(item['threshold_pass'] and item['equal_wire_bytes'] for item in items) if all(item['complete'] for item in items) else None
 return {'observed_comparisons':len(evidence),'expected_comparisons':expected_windows*sum(len(v) for v in modes.values()),'complete':bool(all_complete),'all_equal_wire_bytes':all(r['equal_wire_bytes'] for r in evidence) if evidence else None,'byte_matching_scope':'Equal total accepted wire bytes via actual expert-tail DMA/pool padding; per-tile layout/A placement and phase timing are not identical, so this is not pure arithmetic isolation','primary_busy_metric':'Sum of mutually exclusive C1+C4+C5+C6+C7 across cores; per-core values are diagnostics of placement as well as service','onchip_focus':'Real mixed T=96, Shared Me>=16; frozen source population is correlated across layers','cold_metric':'Routed Me<=2 rows / (last cold completion - first cold binding); queueing included, equal descriptor populations required','by_mode':populations,'P1_primary_timing_passed':separate.get('OP5'),'P2_primary_timing_passed':separate.get('OP2'),'passed':all(item['threshold_pass'] and item['equal_wire_bytes'] for item in populations.values()) if all_complete else None,'reason':'Measured frozen thresholds applied' if all_complete else 'Incomplete paired real-window population; no final pass/fail'}

def assess(root):
 root=Path(root);rs=rows(root);idx={(r['workload'],r['design'],r['op'],r['port'],r['suite'],r['variant']):r for r in rs}
 frozen_mixed=json.loads((root/'inputs/mixed_heldout.json').read_text())['workloads'] if (root/'inputs/mixed_heldout.json').exists() else []
 required_mixed=sum(w['batch'] in (64,96) for w in frozen_mixed);required_focus=sum(w['batch']==96 for w in frozen_mixed)
 datasets={w.get('dataset',w.get('provenance',{}).get('dataset')) for w in frozen_mixed};capture=json.loads((root/'mixed_capture_manifest.json').read_text()) if (root/'mixed_capture_manifest.json').exists() else {}
 fixed=json.loads((root/'development_fixed_comparator.json').read_text()) if (root/'development_fixed_comparator.json').exists() else {}
 alt=fixed.get('selected');out={'scope':'conditional analytical timing; Q1 accuracy and coverage checked separately','fixed_comparator':fixed,'N1':{},'N2':{},'N3':{},'N4':{},'N5':{}}
 held=[r for r in rs if r['split']=='heldout' and r['provenance']=='captured_decode']
 mixed=[r for r in rs if r['split']=='heldout' and r['provenance']=='captured_mixed' and r['tokens'] in (64,96)]
 # Some capture bundles explicitly label the split mixed_heldout.
 mixed += [r for r in rs if r['split']=='mixed_heldout' and r['provenance']=='captured_mixed' and r['tokens'] in (64,96)]
 decode_points={r['workload'] for r in held};mixed_points={r['workload'] for r in mixed}
 out['coverage']={'heldout_decode_windows_observed':len(decode_points),'real_mixed_windows_observed':len(mixed_points),'required_decode_windows':108,'required_available_real_mixed_windows':required_mixed,'independent_mixed_prompt_requests':capture.get('independent_prefill_requests',0),'SWE_mixed_available':'swe' in datasets}
 for group,points in [('decode',held),('real_mixed',mixed)]:
  real=[]
  for r in points:
   if r['suite']=='main' and r['design']=='BL4' and r['op']=='OP2' and r['port']=='iso':
    b=idx.get((r['workload'],'BL4','OP1','iso','main','default'))
    if b:
     ratio=b['unique_weight_bytes']/r['unique_weight_bytes'];real.append((b['cycles']/r['cycles'])/ratio)
  out['N1'][group]={'paired_windows':len(real),'minimum_R_v3':min(real) if real else None,'geomean_R_v3':gm(real),'timing_threshold_pass':bool(real) and min(real)>=.85,'complete':len(real)==(108 if group=='decode' else required_mixed)}
 legacy=[]
 for r in rs:
  if r['provenance'] in ('captured_decode','captured_mixed') and r['split'] in ('heldout','mixed_heldout') and r['tokens']<=96 and r['suite']=='m0_baseline' and r['design']=='BL1' and r['variant']=='byte_only':
   b=idx.get((r['workload'],'BL1','OP0','iso','m0_baseline','normal'))
   if b:legacy.append((b['cycles']/r['cycles'])/(b['weight_bytes']/r['scaled_wire_bytes']))
 p=root/'m0/m0_comparison.csv'
 legacy_exact=bool(legacy)
 if p.exists() and not legacy:
  legacy=[float(r['compression_realization_R']) for r in csv.DictReader(p.open()) if r['organization']=='6']
 out['N1']['legacy']={'windows':len(legacy),'maximum_R_legacy':max(legacy) if legacy else None,'threshold_pass':bool(legacy) and max(legacy)<=.4,'complete':legacy_exact and len(legacy)==108+required_mixed,'fallback_historical_joint_not_FIFO':not legacy_exact}
 n1complete=all(out['N1'][g]['complete'] for g in ('decode','real_mixed','legacy'))
 out['N1']['passed']=all(out['N1'][g].get('timing_threshold_pass',out['N1'][g].get('threshold_pass')) for g in ('decode','real_mixed','legacy')) if n1complete else None
 for op in ('OP0','OP1'):
  points=[r for r in held if r['suite']=='main' and r['design'] in ('BL2','BL3','BL4') and r['op']==op and r['port']=='iso' and (op=='OP1' or r['tokens']==16)]
  vals=[r['supply_efficiency'] for r in points];cut=.95 if op=='OP0' else .90
  out['N3'][op]={'observed_points':len(vals),'eta_min':min(vals) if vals else None,'required_eta':cut,'threshold_pass':bool(vals) and min(vals)>=cut,'complete':len(vals)==(81 if op=='OP0' else 324)}
 losses={};eta_losses={}
 for r in held:
  if r['suite']=='ablation' and r['variant'].startswith('leave_out_'):
   b=idx.get((r['workload'],r['design'],r['op'],r['port'],'ablation','forward8'))
   if b:
    key=r['variant'][10:];losses.setdefault(key,[]).append(r['cycles']/b['cycles']-1)
    if b.get('supply_efficiency',0)>0:eta_losses.setdefault(key,[]).append(1-r['supply_efficiency']/b['supply_efficiency'])
 out['N3']['mechanisms']={k:{'maximum_latency_loss':max(v),'maximum_eta_loss':max(eta_losses.get(k,[]),default=None),'threshold_pass':max(v)>=.03 or max(eta_losses.get(k,[]),default=0)>=.03,'points':len(v),'criterion':'At least one working point loses >=3% latency or supply efficiency when disabled'} for k,v in losses.items()}
 n3complete=all(out['N3'][op]['complete'] for op in ('OP0','OP1')) and len(losses)==8 and all(len(v)==972 for v in losses.values())
 out['N3']['passed']=all(out['N3'][op]['threshold_pass'] for op in ('OP0','OP1')) and all(v['threshold_pass'] for v in out['N3']['mechanisms'].values()) if n3complete else None
 org=[]
 for r in mixed:
  if r['suite']=='main' and r['design']=='BL4' and r['op']=='OP2' and r['port']=='iso':
   b=idx.get((r['workload'],alt,'OP2','iso','main','default'))
   if b:
    ratios={name:r['area_'+name]/b['area_'+name] for name in ('proxy_default','proxy_accumulator_hybrid','proxy_accumulator_all_sram','proxy_accumulator_all_rf') if r.get('area_'+name) and b.get('area_'+name)}
    local_ratio=r['area_proxy_with_local_operand_broadcast_ports']/b['area_proxy_with_local_operand_broadcast_ports'] if r.get('area_proxy_with_local_operand_broadcast_ports') and b.get('area_proxy_with_local_operand_broadcast_ports') else None
    org.append({'time_ratio':r['cycles']/b['cycles'],'traffic_ratio':r['onchip_bytes']/b['onchip_bytes'],'area_ratios':ratios,'local_broadcast_area_ratio':local_ratio,'area_type_ledger_complete':r.get('area_accumulator_types_explicitly_reported')==1 and b.get('area_accumulator_types_explicitly_reported')==1})
 area_by_type={name:max((abs(r['area_ratios'][name]-1) for r in org if name in r['area_ratios']),default=None) for name in ('proxy_default','proxy_accumulator_hybrid','proxy_accumulator_all_sram','proxy_accumulator_all_rf')}
 area_robust=bool(org) and all(r['area_type_ledger_complete'] and len(r['area_ratios'])==4 for r in org) and all(x is not None and x<=.05 for x in area_by_type.values())
 out['N4']={'paired_windows':len(org),'time_ratio_geomean':gm([r['time_ratio'] for r in org]),'traffic_ratio_geomean':gm([r['traffic_ratio'] for r in org]),'maximum_area_difference':area_by_type['proxy_default'],'maximum_area_difference_by_accumulator_type':area_by_type,'area_condition_robust_to_type':area_robust,'area_proxy_scope':'Same measured timing, alternate physical-type estimates; no implemented RF/SRAM split or PPA claim','passed':None,'nominal_passed':None,'area_type_robust_pass':None,'unconditional_organization_evidence':None}
 out['N4']['maximum_area_difference_with_local_RF_broadcast_port_sensitivity']=max((abs(r['local_broadcast_area_ratio']-1) for r in org if r['local_broadcast_area_ratio'] is not None),default=None)
 out['N4']['local_RF_broadcast_scope']='Supplementary hypothetical operand-delivery port proxy; does not alter original nominal N4 criterion or establish synthesized RF fanout cost'
 if required_mixed and len(org)==required_mixed:
  out['N4'].update(n4_verdict(out['N4']['time_ratio_geomean'],out['N4']['traffic_ratio_geomean'],area_by_type['proxy_default'],area_robust))
 policy=[]
 for r in mixed:
  if r['suite']=='policies' and r['op']=='OP2' and r['variant']=='supply_ipd':
   b=idx.get((r['workload'],'BL4','OP2','iso','policies','joint'))
   if b:policy.append((r['cycles'],b['cycles']))
 out['N5']={'paired_windows':len(policy),'time_ratio_geomean':gm([a/b for a,b in policy]),'p95_ratio':float(np.percentile([a for a,b in policy],95)/np.percentile([b for a,b in policy],95)) if policy else None,'timing_passed':None,'Q3_gate_weighted':'See real-weight numerical report; never inferred from scheduling timing'}
 if required_mixed and len(policy)==required_mixed:out['N5']['timing_passed']=out['N5']['time_ratio_geomean']<=.97 or out['N5']['p95_ratio']<=.95
 comp=[];supplement=[]
 for r in held+mixed:
  if r['suite']=='compensation' and r['variant'] not in ('none','b4_none'):
   is_supplement=r['variant'].startswith('b4_')
   b=idx.get((r['workload'],'BL4',r['op'],'iso','compensation','b4_none' if is_supplement else 'none'))
   if not b:continue
   row={'mode':r['variant'],'op':r['op'],'workload':r['workload'],'tokens':r['tokens'],'shared_me':r.get('shared_me',r['tokens']),'provenance':r['provenance'],'precision_quality':r.get('precision_quality',''),'layer_overhead':r['cycles']/b['cycles']-1,'equal_wire_bytes':r['weight_bytes']==b['weight_bytes'],'rank_multiplier_overhead':r.get('area_rank_bf16_multipliers',r.get('budget_rank_multipliers',192))/12288}
   total=total_base=0
   for c in (0,1):
    active=sum(r.get(f'core{c}_states_C{x}',0) for x in (1,4,5,6,7));base=sum(b.get(f'core{c}_states_C{x}',0) for x in (1,4,5,6,7))
    if base:row[f'core{c}_busy_overhead']=active/base-1
    total+=active;total_base+=base
   row['sum_core_busy_overhead']=total/total_base-1 if total_base else None
   if r.get('cold_descriptors')==b.get('cold_descriptors') and r.get('cold_token_rows')==b.get('cold_token_rows') and r.get('cold_token_rows_per_cycle',0)>0 and b.get('cold_token_rows_per_cycle',0)>0:row['cold_throughput_loss']=1-r['cold_token_rows_per_cycle']/b['cold_token_rows_per_cycle']
   if is_supplement:
    row['mode']=row['mode'][3:];supplement.append(row)
   else:comp.append(row)
 csvwrite(root/'claim_N2_compensation_evidence.csv',comp)
 out['N2']=compensation_assessment(comp,108+required_mixed,required_focus)
 csvwrite(root/'claim_N2_P2_B4_supplement_evidence.csv',supplement)
 out['N2_P2_B4_supplement']={**compensation_assessment(supplement,108+required_mixed,required_focus,modes={'OP2':('lanes','separate','offload')}),'scope':'Supplementary precision diagnostic, not a tightened primary pass criterion','quality_status':'Separate B MXINT4 numerical validation required; default B BF16 qualification does not transfer'}
 wins=[]
 for r in comp:
  if r['mode']=='lanes':continue
  l=next((x for x in comp if x['mode']=='lanes' and x['op']==r['op'] and x['workload']==r['workload']),None)
  if l and (r['layer_overhead']<l['layer_overhead'] or (r['sum_core_busy_overhead'] is not None and l['sum_core_busy_overhead'] is not None and r['sum_core_busy_overhead']<l['sum_core_busy_overhead'])):wins.append({**r,'lanes_layer_overhead':l['layer_overhead'],'lanes_sum_core_busy_overhead':l['sum_core_busy_overhead']})
 csvwrite(root/'claim_N2_alternative_wins.csv',wins)
 write(root/'claim_timing_evidence.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();assess(a.root)
