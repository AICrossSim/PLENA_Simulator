#!/usr/bin/env python3
"""Summaries for v3 measured analytical-event results (no oracle numbers)."""
from __future__ import annotations
import argparse,csv,json,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study import csvwrite,write,digest,sha

def gm(v):return math.exp(sum(map(math.log,v))/len(v)) if v else None

def historical_supply_pairs(root,rs):
 """Pair six development windows; the old observer is historical, not heldout."""
 root=Path(root);out=[]
 for r in rs:
  if not (r['suite']=='main' and r['design']=='BL4' and r['port']=='iso' and r['op'] in ('OP0','OP1','OP2') and r.get('evaluation_split',r['split'])=='development' and int(r['tokens']) in (2,16)):continue
  path=root/'m0'/r['workload']/'4+2'/'profile_on'/'repeat0.json'
  if not path.exists():continue
  old=json.loads(path.read_text());profile=old.get('m0_profile',{})
  if not profile.get('mutually_exclusive'):continue
  for core,states in enumerate(profile['core_states']):
   row=dict(workload=r['workload'],tokens=r['tokens'],op=r['op'],core=core,legacy_cycles=old['cycles'],v3_cycles=r['cycles'],legacy_raw_sha256=sha(path),v3_point=r['point'],scope='Historical M0 BF16 4+2 vs paired developmental v3; identical routing window, different physical design/precision. Not heldout or a one-variable experiment.')
   for c in range(9):
    row[f'legacy_C{c}']=states.get(f'C{c}',0)/old['cycles']
    row[f'v3_C{c}']=r.get(f'core{core}_states_C{c}',r.get(f'core{core}_c_states_C{c}',0))/r['cycles']
   assert abs(sum(row[f'legacy_C{c}'] for c in range(9))-1)<1e-9
   assert abs(sum(row[f'v3_C{c}'] for c in range(9))-1)<1e-9
   out.append(row)
 return out

def numerical_pareto_rows(root):
 """Actual full-validation Q1 norms only; do not mix Q3 16-token errors."""
 root=Path(root);source=root/'quant/full_numerics/q1_metrics.csv'
 if not source.exists():return [],dict(status='unavailable',reason='Actual Q1 metrics not yet available')
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'v3_reference'))
 from budget_v3 import Fmt,plan_projection
 source_bytes=source.read_bytes();source_sha=__import__('hashlib').sha256(source_bytes).hexdigest()
 metrics=list(csv.DictReader(source_bytes.decode().splitlines()));out=[];seen=set();main={}
 for r in metrics:
  if r.get('scope')!='layer':continue
  key=tuple(r[k] for k in ('layer','rank_lanes','bits','method','rank','factor_a','factor_b'))
  if key in seen:raise ValueError('Duplicate actual Q1 layer candidate in figure input')
  seen.add(key);bits=int(r['bits']);L=int(r['rank_lanes']);mk=(bits,L,r['factor_a'],r['factor_b'])
  if mk not in main:
   fmt=Fmt('Q1 physical main',bits,r['factor_b'],L,factor_a=r['factor_a'])
   main[mk]=sum(count*sum(plan_projection(name,K,N,0,fmt).main_bytes for name,K,N in [('gate',2048,F),('up',2048,F),('down',F,2048)]) for count,F in ((64,1408),(1,2816)))
  out.append({**r,'layer':int(r['layer']),'rank_lanes':L,'bits':bits,'rank':int(r['rank']),'relative_error':float(r['relative_error']),'cosine':float(r['cosine']),'factor_bytes':float(r['factor_bytes']),'main_bytes':main[mk],'total_physical_weight_bytes':main[mk]+float(r['factor_bytes']),'scope':'Measured full8192-token Q1 MoE-layer output relativeF; all65 expert weight/factor bytes, no Q3 window mixing'})
 coverage=root/'quant/full_numerics/q1_coverage.json';cov=json.loads(coverage.read_text()) if coverage.exists() else {}
 complete=cov.get('complete') is True and len(out)==int(cov.get('expected_candidates',2178))==2178
 return out,dict(status='generated',quality_campaign_complete=complete,candidates=len(out),expected_candidates=2178,source=str(source),source_sha256=source_sha,coverage_sha256=sha(coverage) if coverage.exists() else None,scope='Actual full-validation Q1 numerical data. Candidate scatter/Pareto is descriptive, not accuracy qualification or causal N5 evidence.')
def rows(root,binary_sha=None,include_provisional=False):
 root=Path(root)
 active=None
 if not binary_sha and not include_provisional and (root/'active_result_signature.json').exists():
  active=json.loads((root/'active_result_signature.json').read_text());binary_sha=active['binary_sha256']
 out=[]
 for p in sorted(Path(root).glob('*.csv'),key=lambda p:p.stat().st_mtime):
  if p.name.startswith(('table_','analytical_')):continue
  with p.open() as f:
   for r in csv.DictReader(f):
    physical_match=not active or (r.get('physical_format_sha256')==active['physical_format_sha256'] if 'physical_format_sha256' in active else r.get('frozen_format_sha256','')==active.get('format_sha256',''))
    if r.get('status')=='complete' and r.get('cycles') and (not binary_sha or r.get('raw_binary')==binary_sha) and (not active or not r.get('raw_campaign') or r['raw_campaign']==digest(active)) and physical_match:
     for k,v in list(r.items()):
      if v and k!='rows':
       if v in ('True','False','true','false'):r[k]=v.lower()=='true';continue
       try:r[k]=float(v)
       except ValueError:pass
     out.append(r)
 # Filters/pilots can produce repeated point IDs: latest final rows are identical.
 result={}
 for r in out:result[str(r['point'])]=r
 return list(result.values())

def summary(root,rs):
 group=defaultdict(list)
 for r in rs:group[(r['split'],r['provenance'],r['suite'],r['design'],r['op'],r['port'],r['variant'])].append(r)
 out=[]
 for keys,items in sorted(group.items()):
  rec=dict(zip(('split','provenance','suite','design','op','port','variant'),keys));rec['windows']=len(items)
  for k in ('cycles','ms','us','supply_efficiency','weight_gbps','weight_bytes','useful_macs','issued_macs','onchip_bytes'):
   v=[float(r[k]) for r in items if r.get(k) not in (None,'')]
   if v:rec.update({k+'_geomean':gm(v) if min(v)>0 else sum(v)/len(v),k+'_min':min(v),k+'_max':max(v),k+'_p95':float(np.percentile(v,95))})
  out.append(rec)
 csvwrite(Path(root)/'table_aggregate.csv',out)
 batch_groups=defaultdict(list)
 for r in rs:
  if r['suite']=='main':batch_groups[(r['split'],r['provenance'],r.get('dataset',''),r['tokens'],r['design'],r['op'],r['port'])].append(r)
 by_batch=[]
 for key,items in sorted(batch_groups.items()):
  row=dict(zip(('split','provenance','dataset','tokens','design','op','port'),key));row.update(windows=len(items),M_N_K='+'.join(f'{m}x4x512' for m in str(items[0]['rows']).split('+')),latency_ms_geomean=gm([r['ms'] for r in items]),latency_ms_min=min(r['ms'] for r in items),latency_ms_max=max(r['ms'] for r in items),latency_ms_p95=float(np.percentile([r['ms'] for r in items],95)),precision_quality=items[0].get('precision_quality',''),factor_a=items[0].get('factor_a',''),factor_b=items[0].get('factor_b',''))
  for name in ('weight_bytes','unique_weight_bytes','supply_efficiency','onchip_bytes','host_seconds_two_repeats'):
   values=[r[name] for r in items if r.get(name) not in (None,'')]
   if values:row[name+'_mean']=sum(values)/len(values)
  by_batch.append(row)
 csvwrite(Path(root)/'table_by_batch.csv',by_batch)
 hardware=[];services=[];local_ports=[]
 for r in rs:
  if r['suite']!='main':continue
  base={k:r[k] for k in ('point','split','provenance','dataset','tokens','design','op','port','rows','precision','main_bits','factor_a','factor_b','rank_lanes','clock_ghz','hbm_bytes_per_cycle','hbm_latency_cycles','credits_installed','physical_t_chunk','dataflow','main_multiplier_budget') if k in r}
  hardware.append({**base,**{k:v for k,v in r.items() if k.startswith(('budget_','area_')) or k in ('actual_storage_peak_bytes','pool_peak_bytes','ingress_peak_bytes','credit_peak')}})
  local_ports.append({**base,**{k:v for k,v in r.items() if k.startswith('local_') or k in ('area_proxy_default','area_proxy_with_local_operand_broadcast_ports','area_canonical_port_proxy_scope')}})
  services.append({**base,**{k:v for k,v in r.items() if k.startswith('core') and any(s in k for s in ('group_wall_service','issue_to_acc_commit','pool_to_wor'))}})
 csvwrite(Path(root)/'table_hardware_and_peaks.csv',hardware)
 csvwrite(Path(root)/'table_local_operand_ports.csv',local_ports)
 csvwrite(Path(root)/'table_service_distributions.csv',services)
 csvwrite(Path(root)/'table_physical_movement.csv',[{k:v for k,v in r.items() if k.startswith('movement_') or k in ('point','split','provenance','dataset','tokens','design','op','port','useful_macs','onchip_bytes','onchip_bytes_per_useful_mac','legacy_onchip_subset_bytes','onchip_traffic_definition')} for r in rs if r['suite']=='main'])
 idx={(r['workload'],r['design'],r['op'],r['port'],r['suite'],r['variant']):r for r in rs}
 paired=[]
 for r in rs:
  if r['suite']=='main' and r['design'] in ('BL4','M8_asym62','M8_asym53'):
   alternatives=('BL2','BL3','BL5') if r['design']=='BL4' else ('M8_single','M8_homogeneous','M8_flex62','M8_flex53')
   for alt in alternatives:
    other=idx.get((r['workload'],alt,r['op'],r['port'],'main',r['variant']))
    if other:paired.append(dict(split=r['split'],provenance=r['provenance'],workload=r['workload'],tokens=r['tokens'],op=r['op'],port=r['port'],proposal=r['design'],baseline=alt,baseline_ms=other['ms'],proposal_ms=r['ms'],speedup=other['cycles']/r['cycles'],area_proposal=r.get('area_proxy_default',''),area_baseline=other.get('area_proxy_default',''),onchip_ratio=r.get('onchip_bytes',0)/max(other.get('onchip_bytes',0),1)))
 csvwrite(Path(root)/'table_organization_paired.csv',paired)
 # Paired BF16→quantized byte savings use identical organization/ports/policy.
 realization=[]
 for r in rs:
  if r['suite']=='main' and r['op']=='OP2':
   base=idx.get((r['workload'],r['design'],'OP1',r['port'],'main',r['variant']))
   if base and base.get('weight_bytes') and r.get('weight_bytes'):
    byte_ratio=base.get('unique_weight_bytes',base['weight_bytes'])/r.get('unique_weight_bytes',r['weight_bytes']);speedup=base['cycles']/r['cycles']
    realization.append(dict(split=r['split'],provenance=r['provenance'],workload=r['workload'],tokens=r['tokens'],design=r['design'],port=r['port'],BF16_ms=base['ms'],W4_ms=r['ms'],byte_ratio=byte_ratio,speedup=speedup,R_v3=speedup/byte_ratio))
 csvwrite(Path(root)/'table_realization.csv',realization)
 legacy=[]
 for r in rs:
  if r['suite']=='m0_baseline' and r['variant']=='byte_only':
   b=idx.get((r['workload'],r['design'],'OP0','iso','m0_baseline','normal'))
   if b:
    speed=b['cycles']/r['cycles'];ratio=b['weight_bytes']/r['scaled_wire_bytes']
    legacy.append(dict(split=r['split'],provenance=r['provenance'],workload=r['workload'],design=r['design'],baseline_ms=b['ms'],compression_oracle_ms=r['ms'],speedup=speed,wire_byte_ratio=ratio,R_legacy=speed/ratio,scope='nonphysical byte timing oracle; BF16 onchip unchanged'))
 csvwrite(Path(root)/'table_legacy_realization.csv',legacy)
 comp=[]
 for r in rs:
  if r['suite']=='compensation' and r['variant'] not in ('none','b4_none'):
   base=idx.get((r['workload'],r['design'],r['op'],r['port'],'compensation','b4_none' if r['variant'].startswith('b4_') else 'none'))
   if base:
    out=dict(split=r['split'],provenance=r['provenance'],workload=r['workload'],op=r['op'],mode=r['variant'],layer_overhead=r['cycles']/base['cycles']-1,none_weight_bytes=base['weight_bytes'],mode_weight_bytes=r['weight_bytes'],equal_wire_bytes=base['weight_bytes']==r['weight_bytes'],mode_padding_bytes=r.get('comp_padding_bytes',0),none_padding_bytes=base.get('comp_padding_bytes',0),cross_core_bytes=r.get('cross_core_bytes',0))
    total=total_base=0
    for i in (0,1):
     # Include only exclusive states assigned by the engine, not overlapping waits.
     busy=sum(r.get(f'core{i}_states_C{c}',r.get(f'core{i}_c_states_C{c}',0)) for c in (1,4,5,6,7))
     base_busy=sum(base.get(f'core{i}_states_C{c}',base.get(f'core{i}_c_states_C{c}',0)) for c in (1,4,5,6,7))
     if base_busy:out[f'core{i}_busy_overhead']=busy/base_busy-1
     total+=busy;total_base+=base_busy
    if total_base:out['sum_core_busy_overhead']=total/total_base-1
    if r.get('cold_descriptors')==base.get('cold_descriptors') and r.get('cold_token_rows')==base.get('cold_token_rows') and r.get('cold_token_rows_per_cycle',0)>0 and base.get('cold_token_rows_per_cycle',0)>0:out['cold_throughput_loss']=1-r['cold_token_rows_per_cycle']/base['cold_token_rows_per_cycle']
    comp.append(out)
 csvwrite(Path(root)/'table_compensation.csv',comp)
 ablation=[]
 for r in rs:
  if r['suite']=='ablation' and str(r['variant']).startswith('leave_out_'):
   full=idx.get((r['workload'],r['design'],r['op'],r['port'],'ablation','forward8'))
   if full:ablation.append(dict(split=r['split'],provenance=r['provenance'],workload=r['workload'],design=r['design'],op=r['op'],variant=r['variant'],loss=r['cycles']/full['cycles']-1,eta_loss=1-r['supply_efficiency']/full['supply_efficiency'] if full.get('supply_efficiency',0)>0 else None,full_ms=full['ms'],disabled_ms=r['ms']))
 csvwrite(Path(root)/'table_ablation_leave_one.csv',ablation)
 area_rows=[{k:v for k,v in r.items() if k.startswith(('area_','budget_')) or k in ('point','design','op','port','split','tokens')} for r in rs if r.get('area_proxy_default')]
 csvwrite(Path(root)/'table_area_components.csv',area_rows)
 area_sensitivity=[]
 seen=set()
 for r in area_rows:
  key=(r['design'],r['op'],r['port'])
  if key in seen:continue
  seen.add(key)
  terms={'main_mx':r['area_main_mx_multipliers']/12288,'main_bf16':2*r['area_main_bf16_multipliers']/12288,'rank_bf16':2*r['area_rank_bf16_multipliers']/12288,'SRAM':r['area_sram_bytes']/2097152,'RF':4*r['area_rf_bytes']/2097152,'ports':.1*r['area_read_write_port_bytes_per_cycle']/2048}
  total=sum(terms.values())
  for term,value in terms.items():
   for multiplier in (.5,1.,2.):area_sensitivity.append(dict(design=r['design'],op=r['op'],port=r['port'],varied_coefficient=term,coefficient_multiplier=multiplier,proxy=total+value*(multiplier-1),raw_component=value,default_proxy=total))
 csvwrite(Path(root)/'table_area_sensitivity.csv',area_sensitivity)
 area_types=[]
 for r in area_rows:
  for kind in ('default','accumulator_hybrid','accumulator_all_sram','accumulator_all_rf'):
   key='area_proxy_'+kind
   if key in r:area_types.append({**{k:r[k] for k in ('point','design','op','port','split','tokens') if k in r},'accumulator_type_assumption':kind,'area_proxy':r[key],'scope':'Same analytical timing; uncertain physical-type proxy, not RTL mapping or synthesized area'})
 csvwrite(Path(root)/'table_area_accumulator_type_bounds.csv',area_types)
 policy=[]
 for r in rs:
  if r['suite']=='policies' and r['variant']=='supply_ipd':
   for alt in ('fifo','earliest_finish','joint'):
    base=idx.get((r['workload'],r['design'],r['op'],r['port'],'policies',alt))
    if base:policy.append(dict(split=r['split'],provenance=r['provenance'],workload=r['workload'],op=r['op'],baseline=alt,speedup=base['cycles']/r['cycles'],baseline_ms=base['ms'],supply_ipd_ms=r['ms'],prediction_mean_absolute_error_cycles=r.get('prediction_mean_absolute_error_cycles',''),prediction_worst_underestimate_cycles=r.get('prediction_worst_underestimate_cycles','')))
 csvwrite(Path(root)/'table_policy_paired.csv',policy)
 return out,paired,realization,comp,ablation

def figures(root,rs):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 root=Path(root);dest=root/'figures';dest.mkdir(exist_ok=True)
 receipts=[]
 all_rows=rs
 evaluated=[r for r in rs if r.get('evaluation_split',r['split']) in ('heldout','mixed_heldout','constructed')]
 if evaluated:rs=evaluated
 plot_scope='heldout and separately labeled constructed' if evaluated else 'development preview; no heldout conclusion'
 def save(fig,name,note):
  fig.tight_layout();fig.savefig(dest/(name+'.svg'));fig.savefig(dest/(name+'.png'),dpi=160);plt.close(fig);receipts.append(dict(name=name,status='generated',note=note,scope=plot_scope))
 legacy=root/'m0/m0_legacy_fit.csv'
 if legacy.exists():
  fit=list(csv.DictReader(legacy.open()));x=[float(r['issues_per_tile']) for r in fit];y=[float(r['ns_per_tile']) for r in fit]
  fig,ax=plt.subplots(figsize=(7,4));ax.scatter(x,y,label='Observed layer cycles / tile')
  order=np.argsort(x);pred=[float(r['supplied_fit_ns'])/float(r['tiles']) for r in fit];ax.plot([x[i] for i in order],[pred[i] for i in order],ls='--',label='Supplied 30.4 ns/issue fit')
  for a,b,r in zip(x,y,fit):ax.annotate(r['workload'].replace('joint_test_','').replace('_l13_s7',''),(a,b),fontsize=7,xytext=(4,4),textcoords='offset points')
  ax.set_xlabel('MAC issues / unique weight tile');ax.set_ylabel('Layer cycles / tile (ns at 1 GHz)');ax.legend();save(fig,'legacy_fit','Frozen Joint single6 developmental M0 throughput fit; not a measured per-issue serial handshake')
 # Full available OP2 paired decode/mixed results; provenance shown in every label.
 items=[r for r in rs if r['suite']=='main' and r['op']=='OP2' and r['port']=='iso']
 if items:
  grouped=defaultdict(list)
  for r in items:grouped[(r['provenance'],r['tokens'],r['design'])].append(r['us'])
  fig,ax=plt.subplots(figsize=(10,4))
  for d in ('BL2','BL3','BL4','BL5'):
   labels=sorted({(k[0],int(k[1])) for k in grouped});ax.plot(range(len(labels)),[gm(grouped[(p,t,d)]) if (p,t,d) in grouped else np.nan for p,t in labels],'o-',label=d)
  labels=sorted({(k[0],int(k[1])) for k in grouped});ax.set_xticks(range(len(labels)),[f'{p}\nT={t}' for p,t in labels],rotation=30,ha='right');ax.set_ylabel('Layer latency (µs at 1 GHz)');ax.legend();save(fig,'organization','Analytical event simulation; captured decode and constructed workloads labeled separately')
 # Equal task-specified supply ports versus separately billed demand ports.
 for port in ('iso','demand'):
  subset=[r for r in rs if r['suite']=='main' and r['op']=='OP2' and r['port']==port and r['design'] in ('BL2','BL3','BL4','BL5')]
  if not subset:continue
  fig,axes=plt.subplots(1,2,figsize=(12,4));grouped=defaultdict(list)
  for r in subset:grouped[(r['provenance'],int(r['tokens']),r['design'])].append(r)
  labels=sorted({(k[0],k[1]) for k in grouped})
  for d in ('BL2','BL3','BL4','BL5'):
   for ax,field in zip(axes,('us','onchip_bytes_per_useful_mac')):
    ax.plot(range(len(labels)),[gm([r[field] for r in grouped[(p,t,d)] if r.get(field,0)>0]) if (p,t,d) in grouped else np.nan for p,t in labels],'o-',label=d)
  for ax in axes:ax.set_xticks(range(len(labels)),[f'{p}\nT={t}' for p,t in labels],rotation=35,ha='right');ax.legend()
  axes[0].set_ylabel('Layer latency (µs)');axes[1].set_ylabel('On-chip traffic / useful MAC (B)')
  save(fig,'organization_'+port,'Task-specified aggregate supply ports fixed in ISO; private accumulator source ports and demand ports separately charged. Traffic is an energy proxy, not joules.')
 real=list(csv.DictReader((root/'table_realization.csv').open())) if (root/'table_realization.csv').exists() else []
 selected_workloads={r['workload'] for r in rs}
 real=[r for r in real if r['workload'] in selected_workloads]
 legacy_file=root/'m0/m0_comparison.csv'
 legacy_real=list(csv.DictReader(legacy_file.open())) if legacy_file.exists() else []
 if real or legacy_real:
  fig,axes=plt.subplots(1,2,figsize=(12,4))
  ax=axes[0];group=defaultdict(list)
  for r in legacy_real:group[r['organization']].append(float(r['compression_realization_R']))
  keys=sorted(group);ax.bar(range(len(keys)),[gm(group[k]) for k in keys]);ax.set_xticks(range(len(keys)),keys);ax.axhline(.40,color='black',ls='--');ax.set_title('Historical development: R_legacy\nNonphysical compression timing oracle');ax.set_ylabel('Speedup / wire-byte reduction')
  ax=axes[1];group=defaultdict(list)
  for r in real:group[(r['provenance'],r['design'])].append(float(r['R_v3']))
  keys=sorted(group);ax.bar(range(len(keys)),[gm(group[k]) for k in keys]);ax.set_xticks(range(len(keys)),[f'{p}\n{d}' for p,d in keys],rotation=30,ha='right');ax.axhline(.85,color='black',ls='--');ax.set_title('R_v3: '+plot_scope);ax.set_ylabel('Speedup / unique-weight-byte reduction')
  save(fig,'realization','Left: M0 old BF16 onchip unchanged, wire-only oracle at six developmental windows. Right: actual OP1 BF16 versus OP2 P2 paired organization/port. Different cohorts and interventions, not a direct one-variable comparison.')
  receipts[-1]['legacy_source_sha256']=sha(legacy_file) if legacy_file.exists() else None
 ab=list(csv.DictReader((root/'table_ablation_leave_one.csv').open())) if (root/'table_ablation_leave_one.csv').exists() else []
 ab=[r for r in ab if r['workload'] in selected_workloads]
 if ab:
  group=defaultdict(list)
  for r in ab:group[(r['op'],r['variant'])].append(float(r['loss']))
  fig,ax=plt.subplots(figsize=(11,4));keys=sorted(group);ax.bar(range(len(keys)),[100*sum(group[k])/len(group[k]) for k in keys]);ax.set_xticks(range(len(keys)),[f'{op}\n{v[10:]}' for op,v in keys],rotation=50,ha='right');ax.axhline(3,color='black',ls='--');ax.set_ylabel('Disable-one latency loss (%)');save(fig,'ablation','Leave-one mechanism loss; overlapping counters not added')
 states=[r for r in rs if r['suite']=='main' and r['design']=='BL4' and r['port']=='iso' and r['op'] in ('OP0','OP1','OP2')]
 if states:
  fig,ax=plt.subplots(figsize=(9,4));keys=sorted({r['op'] for r in states});bottom=np.zeros(len(keys))
  for c in range(9):
   vals=[]
   for op in keys:
    sub=[r for r in states if r['op']==op];vals.append(np.mean([r.get(f'core0_states_C{c}',r.get(f'core0_c_states_C{c}',0))/r['cycles'] for r in sub]))
   ax.bar(keys,vals,bottom=bottom,label=f'C{c}');bottom+=vals
  ax.set_ylabel('Dense-core exclusive cycle fraction');ax.legend(ncol=3);save(fig,'stalls','Exclusive cycles, averaged dense core; not a global additive latency decomposition')
 historical=historical_supply_pairs(root,all_rows);csvwrite(root/'table_historical_v3_supply_states.csv',historical)
 if historical:
  fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
  labels=['C0 idle','C1 issue','C2 sent/not ready','C3 not sent','C4 operand feed','C5 X wait','C6 acc/dependency','C7 vector/context','C8 control']
  for core,ax in enumerate(axes):
   subset=[r for r in historical if r['core']==core];ops=sorted({r['op'] for r in subset});xs=[(op,kind) for op in ops for kind in ('legacy','v3')];bottom=np.zeros(len(xs))
   for c in range(9):
    vals=[np.mean([r[f'{kind}_C{c}'] for r in subset if r['op']==op]) for op,kind in xs]
    ax.bar(range(len(xs)),vals,bottom=bottom,label=labels[c]);bottom+=vals
   ax.set_xticks(range(len(xs)),[f'{op}\n{kind}' for op,kind in xs]);ax.set_title(f'Core {core}, M={4 if core==0 else 2}: six developmental windows');ax.set_ylim(0,1.01);ax.set_ylabel('Exclusive core cycle fraction')
  axes[-1].legend(ncol=3,loc='upper center',bbox_to_anchor=(.1,-.12),fontsize=8)
  save(fig,'stalls_historical_vs_v3','Only identical six B2/B16 development routing windows are paired. Old BF16 M0 repeats in each OP panel; v3 OP0/1 BF16 and OP2 quantized differ in ports/dataflow/precision. Counters use each engine\'s exclusive classification; not equal instruction semantics or global additive latency.')
  receipts[-1].update(scope='Historical/development comparison only; separate from heldout',paired_window_count=len({r['workload'] for r in historical}),sources={r['workload']:r['legacy_raw_sha256'] for r in historical},paired_v3_points=sorted({r['v3_point'] for r in historical}))
 comp=list(csv.DictReader((root/'table_compensation.csv').open())) if (root/'table_compensation.csv').exists() else []
 comp=[r for r in comp if r['workload'] in selected_workloads]
 if comp:
  group=defaultdict(list)
  for r in comp:group[(r['op'],r['mode'])].append(float(r['layer_overhead']))
  fig,ax=plt.subplots(figsize=(8,4));cost={'lanes':1.5625,'none':0,'separate':0,'kext':0,'offload':0}
  for (op,mode),vals in group.items():
   matching=[r for r in rs if r['suite']=='compensation' and r['op']==op and r['variant']==mode]
   x=100*np.mean([r.get('area_rank_bf16_multipliers_active',0)/max(sum(int(m) for m in str(r['rows']).split('+'))*4*512,1) for r in matching]) if matching else cost.get(mode,0)
   ax.scatter(x,100*np.mean(vals),label=f'{op}/{mode}')
  ax.set_xlabel('Active rank multipliers / main multiplier budget (%)');ax.set_ylabel('Layer-time overhead (%)');ax.legend();save(fig,'compensation','Timing modes retain installed rank channels; x-axis shows active dedicated rank arithmetic, not a synthesized area reduction')
 sensitivity=[r for r in rs if r['suite']=='sensitivity']
 if sensitivity:
  fig,ax=plt.subplots(figsize=(9,4));group=defaultdict(list)
  for r in sensitivity:group[(r['design'],r['variant'])].append(r['us'])
  variants=sorted({k[1] for k in group})
  for d in sorted({k[0] for k in group}):ax.plot(range(len(variants)),[gm(group[(d,v)]) for v in variants],'o-',label=d)
  ax.set_xticks(range(len(variants)),variants);ax.set_ylabel('Layer latency (µs)');ax.legend();save(fig,'sensitivity','Bandwidth and response latency remain explicit; credit adjusted by Little law')
 forward=[r for r in rs if r['suite']=='ablation' and r['variant'].startswith('forward')]
 if forward:
  fig,ax=plt.subplots(figsize=(9,4));group=defaultdict(list)
  for r in forward:group[(r['op'],int(r['variant'][7:]))].append(r['us'])
  for op in sorted({k[0] for k in group}):ax.plot(range(9),[gm(group[(op,i)]) if group.get((op,i)) else np.nan for i in range(9)],'o-',label=op)
  ax.set_xticks(range(9),['base','credit','pool','quota','pipeline','X reuse','W reuse','ports','inline'],rotation=30);ax.set_ylabel('Layer latency (µs)');ax.legend();save(fig,'ablation_forward','Full forward mechanism staircase paired with leave-one results')
 numerical,numeric_receipt=numerical_pareto_rows(root)
 if numerical:
  csvwrite(root/'table_numerical_pareto_points.csv',numerical)
  layers=sorted({r['layer'] for r in numerical});fig,axes=plt.subplots(1,len(layers),figsize=(6*len(layers),4),squeeze=False)
  for layer,ax in zip(layers,axes[0]):
   for L,marker in ((8,'o'),(16,'s')):
    points=[r for r in numerical if r['layer']==layer and r['rank_lanes']==L]
    ax.scatter([r['total_physical_weight_bytes']/2**20 for r in points],[r['relative_error'] for r in points],s=8,alpha=.25,label=f'L{L}: all measured candidates',marker=marker)
    best=math.inf;front=[]
    for r in sorted(points,key=lambda p:(p['total_physical_weight_bytes'],p['relative_error'])):
     if r['relative_error']<best:front.append(r);best=r['relative_error']
    ax.scatter([r['total_physical_weight_bytes']/2**20 for r in front],[r['relative_error'] for r in front],s=35,marker=marker,label=f'L{L}: observed nondominated points')
   ax.set_title(f'Layer {layer}: '+('complete Q1' if numeric_receipt['quality_campaign_complete'] else 'provisional Q1'));ax.set_xlabel('All65 experts: main + factor weights (MiB)');ax.set_ylabel('Actual full8192-token relativeF error');ax.legend(fontsize=8)
  save(fig,'numerical_pareto','Actual full-validation Q1 candidate scatter, main payload from budget_v3 r=0 plus measured factor bytes; no Q3 window errors, interpolation, or qualification inferred.')
  receipts[-1].update(numeric_receipt)
 else:receipts.append(dict(name='numerical_pareto',**numeric_receipt))
 write(root/'figure_receipt.json',receipts)

def coverage(root):
 root=Path(root);signature=json.loads((root/'active_result_signature.json').read_text()) if (root/'active_result_signature.json').exists() else {};binary=signature.get('binary_sha256');records={}
 for path in sorted(root.glob('*.csv'),key=lambda p:p.stat().st_mtime):
  if path.name.startswith(('table_','analytical_','claim_')):continue
  for row in csv.DictReader(path.open()):
   if 'status' not in row or not row.get('point'):continue
   if binary and row.get('raw_binary') and row['raw_binary']!=binary:continue
   # Exclusions carry no binary result. Require the CSV's execution receipt.
   receipt=root/(path.stem+'_receipt.json')
   if binary and receipt.exists() and json.loads(receipt.read_text()).get('signature',{})!=signature:continue
   if row.get('raw_campaign') and row['raw_campaign']!=digest(signature):continue
   records[row['point']]=row
 group=defaultdict(list)
 for row in records.values():group[(row.get('evaluation_split',row.get('split')),row['suite'])].append(row)
 required={}
 inventory=root/'declared_matrix_inventory.csv'
 if inventory.exists():
  for r in csv.DictReader(inventory.open()):required[(r['split'],r['suite'])]=r
 out=[]
 for key in sorted(set(group)|set(required)):
  vals=group.get(key,[]);req=required.get(key,{})
  counts={s:sum(r['status']==s for r in vals) for s in ('complete','excluded','unsupported','failed')}
  out.append(dict(evaluation_split=key[0],suite=key[1],declared_points=req.get('points',''),declared_legal=req.get('legal',''),observed_points=len(vals),pending_points=max(0,int(req.get('points',len(vals)))-len(vals)),**counts))
 csvwrite(root/'table_coverage.csv',out)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--binary-sha');p.add_argument('--include-provisional',action='store_true');a=p.parse_args();rs=rows(a.root,a.binary_sha,a.include_provisional);summary(a.root,rs);figures(a.root,rs);coverage(a.root)
 write(Path(a.root)/'analysis_receipt.json',dict(rows=len(rs),simulator_scope='analytical discrete-event FFN; not native HBM/RTL/full model',duplicated_raw_json_verification='See point receipts',input_manifest_sha256=__import__('hashlib').sha256((Path(a.root)/'input_manifest.json').read_bytes()).hexdigest()))
 print(json.dumps(dict(rows=len(rs))))
if __name__=='__main__':main()
