#!/usr/bin/env python3
"""Adversarial accounting checks plus FIFO-neutrality and cycle-RLE validation."""
import argparse,copy,json,os,subprocess
from pathlib import Path
import step0 as b
import step2 as s
import diagnose_step1 as d


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
 check=out/'adversarial_checks';check.mkdir(exist_ok=True);m=b.read(out/'manifest.json')
 c=next(c for c in m['cases'] if c['id']=='me1__single__threshold__w6_age4_reserved')
 env=s.read(out/c['id']/'rep1.json.gz');a,w,g=[b.read(c[k]) for k in ['architecture','workload','golden']]
 v=d.load_validator(out/'repro/compare_moe_normal.py');v.validate_run(env,g,w,a,0,0)
 def detail(e):return e['result']['cores'][0]['refinement']['split_window']
 def peak_unseen(e):
  z=detail(e);z['live_peak_classes']=[z['live_peak_bytes']-2*z['operand_stage_bytes'],0,2*z['operand_stage_bytes']]
 def third_stage(e):
  z=detail(e);z['lifetime_changes'][0][3]=3*z['operand_stage_bytes']
 mutations={
  'peak_sum_off_by_one':lambda e:detail(e).__setitem__('live_peak_bytes',detail(e)['live_peak_bytes']+1),
  'unobserved_peak_tuple':peak_unseen,
  'unpaid_third_operand':third_stage,
  'leaked_live_buffer':lambda e:detail(e).__setitem__('live_current_bytes',[1,0,0]),
  'unpaid_candidate_header':lambda e:detail(e).__setitem__('window_header_bytes',0),
  'uncharged_header_access':lambda e:detail(e).__setitem__('window_header_updates',0),
  'fixed_64_cycle_aging':lambda e:detail(e).__setitem__('aging_threshold_ps',64000),
  'unpaid_frontend_credits':lambda e:e['result']['dma_frontend'].__setitem__('reserved_bytes',e['result']['dma_frontend']['reserved_bytes']-96),
  'missing_native_sector':lambda e:e['result'].__setitem__('hbm_read_bytes',e['result']['hbm_read_bytes']-32),
  'native_request_leak':lambda e:e['memory_model']['calibration'].__setitem__('native_pending',1),
  'changed_output_bit':lambda e:e['result']['output_bf16'][0].__setitem__(0,e['result']['output_bf16'][0][0]^1),
 }
 results=[]
 for name,mutate in mutations.items():
  altered=copy.deepcopy(env);mutate(altered)
  try:v.validate_run(altered,g,w,a,0,0)
  except ValueError as ex:results.append(dict(check=name,status='rejected',reason=str(ex)))
  else:raise AssertionError('tampered result accepted: '+name)
 def r(x):return x['cores'][0]['refinement']
 configs={
  'weight_budget_one_byte_short':lambda x:x['cores'][0].__setitem__('weight_sram_bytes',detail(env)['weight_reserved_bytes']-1),
  'frontend_budget_one_byte_short':lambda x:x['dma'].__setitem__('frontend_sram_bytes',env['result']['dma_frontend']['reserved_bytes']-1),
  'accumulator_budget_one_byte_short':lambda x:x['cores'][0].__setitem__('accumulator_bytes',env['result']['cores'][0]['accumulator_peak_bytes']-1),
  'missing_cohort_pool':lambda x:r(x).__setitem__('output_pool',None),
  'invalid_aging':lambda x:r(x)['stream_ctrl']['split_window'].__setitem__('aging_multiplier',64),
  'invalid_window':lambda x:r(x)['stream_ctrl']['split_window'].__setitem__('window_tiles',7),
  'incompatible_fair_credits':lambda x:x['dma'].__setitem__('fair_credits',True),
  'forbidden_native_demand_aware':lambda x:x['dma'].__setitem__('issue_policy','demand_aware'),
 }
 for name,mutate in configs.items():
  altered=copy.deepcopy(a);mutate(altered);path=check/(name+'.json');b.save(path,altered)
  cmd=[str(out/'repro/moe_dual_normal'),'--architecture',str(path),'--workload',c['workload'],'--output',str(check/(name+'.output.json')),
       '--hbm-channels','8','--max-hbm-bytes',str(1<<30)]
  process=subprocess.run(cmd,capture_output=True,text=True,env=dict(os.environ,LD_LIBRARY_PATH=str(out/'repro')),timeout=60)
  (check/(name+'.log')).write_text(process.stdout+process.stderr)
  b.require(process.returncode!=0 and 'panicked' not in process.stderr,'invalid configuration did not fail closed: '+name)
  results.append(dict(check=name,status='rejected',returncode=process.returncode,reason=process.stderr[-1000:]))
 fifo=[]
 for window in [3,6]:
  one,two=[s.read(out/f'me1__single__threshold__w{window}_ageoff_{credit}/rep1.json.gz') for credit in ['shared','reserved']]
  b.require(one['memory_model']['calibration']==two['memory_model']['calibration'],'single core grant policy changed native order')
  b.require(one['result']['total_ps']==two['result']['total_ps'],'single FIFO is not neutral')
  fifo.append(dict(window=window,native_counters_exact=True,time_ps=one['result']['total_ps']))
 b.save(check/'validation.json',dict(status='passed',negative_checks=results,single_core_fifo=fifo))
 print('PASS',len(results),'adversarial checks, two FIFO controls')

if __name__=='__main__':main()
