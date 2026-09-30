"""Audit immutable repeated reports and export comparable policy diagnostics."""
import argparse
import csv,json
from pathlib import Path


def dumpcsv(p,rows):
 if not rows:return
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)

def load(root):
 out=[]
 for d in sorted((root/'points').iterdir()):
  if not (d/'repeat_receipt.json').exists():continue
  b1=(d/'report_repeat1.json').read_bytes();b2=(d/'report_repeat2.json').read_bytes();assert b1==b2
  r=json.loads(b1);w=json.loads((d/'workload.json').read_text());cfg=r['config']
  assert r['drained'] and r['ownership_k_order_capacity_checks']
  assert r['dma_transactions_accepted']==r['dma_transactions_landed']==r['weight_bytes']//32
  assert r['credit_peak']<=cfg['credits']
  assert r['useful_macs']==sum(3*e['Me']*e['H']*e['F'] for e in w['experts'])
  if cfg['credits']>256:
   assert cfg['diagnostic_credit_expansion']
   assert r['resource_conditions']['not_eligible_for_equal_budget_claim']
  assert r['pending_window_peak']<=8
  assert len(r['dispatch_audit'])==len(w['experts'])
  assert len({x['task'] for x in r['dispatch_audit']})==len(w['experts'])
  for c in r['cores']:
   assert c['stats']['workspace_peak_bytes']<=c['capacity']
   assert c['stats']['weight_peak_bytes'] <= (10//len(cfg['lanes']))*4096
   assert c['stats']['x_peak_bytes'] <= c['m']*2048
  out.append((d,r,w))
 return out

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root', type=Path, required=True)
 parser.add_argument('--expected-points', type=int, required=True)
 args=parser.parse_args()
 root=args.root
 target=args.expected_points
 points=load(root)
 print(root,len(points),'/',target)
 assert len(points)==target, 'Incomplete campaign'
 traffic={}
 for _,r,w in points:
  signature=(r['weight_bytes'],r['useful_macs'])
  assert traffic.setdefault(w['id'],signature)==signature, 'Work or traffic changed'
 bindings=[];metrics=[];idle=[];slots=[];tier=[]
 for d,r,w in points:
  cfg=r['config'];org='+'.join(map(str,cfg['lanes']));mode=d.name.split('__')[-2]
  key={'batch':w['batch'],'organization':org,'arbiter':cfg['arbiter'],'mode':mode,
       'credits':cfg['credits'],'over_budget':cfg.get('diagnostic_credit_expansion',False)}
  for a in r['dispatch_audit']:
   assert a['actual_minus_predicted_cycles']==a['actual_finish_cycle']-a['predicted_finish_cycle']
   bindings.append({**key,**a,'eligible_cores':json.dumps(a['eligible_cores'])})
  ds=r.get('supply_diagnostics',{})
  latencymean=ds.get('credit_lease_cycles_sum',0)/max(ds.get('credit_completions',0),1)
  metrics.append({**key,'cycles':r['cycles'],'latency_ms':r['cycles']/1e6,
                  'HBM_bytes':r['weight_bytes'],
                  'extra_return_bytes':max(0,cfg['credits']-256)*32,
                  'extra_credit_tag_bytes':max(0,cfg['credits']-256)*2,
                  'late_bind_wait_cycles':r.get('late_bind_wait_cycles',0),'byte_lower_bound_us':r['weight_bytes']/256/1000,
                  'actual_weight_GBps':r['weight_bytes']/r['cycles'],
                  'gap_above_256_lower_bound_percent':(r['cycles']*256/r['weight_bytes']-1)*100,
                  'gap_above_128_credit_lower_bound_percent':(r['cycles']*128/r['weight_bytes']-1)*100 if cfg['credits']==256 else '',
                  'mean_credit_lease_ns':latencymean,'max_credit_lease_ns':ds.get('credit_lease_cycles_max',0),
                  'little_mean_lease_upper_bound_GBps':cfg['credits']*32/latencymean if latencymean else '',
                  'eligible_1':sum(len(a['eligible_cores'])==1 for a in r['dispatch_audit']),
                  'eligible_2':sum(len(a['eligible_cores'])==2 for a in r['dispatch_audit']),
                  'signed_prediction_error_mean_us':sum(a['actual_minus_predicted_cycles'] for a in r['dispatch_audit'])/len(w['experts'])/1000,
                  'absolute_prediction_error_mean_us':sum(abs(a['actual_minus_predicted_cycles']) for a in r['dispatch_audit'])/len(w['experts'])/1000,
                  'worst_underestimate_us':max([0]+[a['actual_minus_predicted_cycles'] for a in r['dispatch_audit']])/1000})
  for c,x in enumerate(ds.get('cores',[])):
   obs=x['observed_cycles'];assert sum(x['slot_occupancy_cycles'].values())==obs
   assert sum(x['idle_reason_and_other_phase'].values())==x['no_accept_cycles']
   for k,v in x['slot_occupancy_cycles'].items():
    k=int(k);cur=k//1000;nxt=k//100%10;ahead=k//10%10;free=k%10
    # single all-free=10 carries into ahead digit; special key 10.
    if k==10 and len(cfg['lanes'])==1:cur=nxt=ahead=0;free=10
    assert cur+nxt+ahead+free==10//len(cfg['lanes'])
    slots.append({**key,'core':c,'current_slots':cur,'next_slots':nxt,'phase_ahead_slots':ahead,'free_slots':free,'cycles':v,'fraction':v/obs})
   for k,v in x['idle_reason_and_other_phase'].items():
    k=int(k);reason=k//1000;other=k//100%10;wait=k%100
    idle.append({**key,'core':c,'reason':('no_legal_request','credits_full','DMA_backpressure','not_selected_or_rate_limited')[reason],
                 'other_phase':{0:'Gate',1:'Up',2:'Activation',4:'Down',5:'Output',6:'Idle',7:'No other core'}[other],
                 'other_front_wait':{0:'other',1:'previous_K',2:'X',3:'weight',4:'result_context',5:'result_drain',6:'conversion',7:'operand_feed'}[wait],
                 'cycles':v,'fraction_of_observed_core_cycles':v/obs})
   tier.append({**key,'core':c,'observed_cycles':obs,'no_accept_fraction':x['no_accept_cycles']/obs,
                'tier0_eligible_cycles':x['tier0_eligible_cycles'],'tier0_episodes':x['tier0_episodes'],
                **{f'tier{j}_grants':v for j,v in enumerate(x['tier_grants'])},
                'tier3_grant_fraction':x['tier_grants'][3]/max(sum(x['tier_grants']),1),
                'phase_ahead_tiles':x['phase_ahead_tiles'],'next_denied_for_depth':x['next_denied_for_current_depth']})
 dumpcsv(root/'binding_predictions.csv',bindings);dumpcsv(root/'metrics.csv',metrics)
 dumpcsv(root/'supply_idle.csv',idle);dumpcsv(root/'slot_occupancy.csv',slots);dumpcsv(root/'arbiter_tiers.csv',tier)
 check={'points':len(points),'runs':2*len(points),'full_reports_identical':True,
        'byte_slot_credit_owner_drain_checks':True,'work_and_weight_bytes_unchanged_across_modes':True,
        'over_budget_points':sum(r['config'].get('diagnostic_credit_expansion',False) for _,r,_ in points),'large_timing_runs_execute_payloads':False}
 (root/'validation.json').write_text(json.dumps(check,indent=2)+'\n')
 print('metrics exported')

if __name__ == "__main__": main()
