#!/usr/bin/env python3
"""Recheck the unchanged 31.124 us acceptance reference with the final binary."""
import argparse,os,subprocess,tempfile
from pathlib import Path
import step0 as b
import step2 as s
import diagnose_step1 as d


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
 step0=s.COMMON.parent.parent/'step0';source=b.read(step0/'manifest.json')
 c=next(c for c in source['cases'] if c['id']=='frozen__expert_me1_single_legacy_n3')
 old=b.read(c['reference']);folder=out/'frozen_me1_n3';folder.mkdir(exist_ok=True);first=None
 a,w,g=[b.read(c[k]) for k in ['architecture','workload','golden']];v=d.load_validator(out/'repro/compare_moe_normal.py')
 for rep in [1,2]:
  dest=folder/f'rep{rep}.json.gz'
  if not dest.exists():
   with tempfile.TemporaryDirectory(prefix='plena-frozen-me1-',dir='/tmp') as tmp:
    raw=Path(tmp)/'result.json'
    cmd=[str(out/'repro/moe_dual_normal'),'--architecture',c['architecture'],'--workload',c['workload'],
         '--output',str(raw),'--hbm-channels','8','--max-hbm-bytes',str(1<<30)]
    with (folder/f'rep{rep}.log').open('w') as log:
     subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800,env=dict(os.environ,LD_LIBRARY_PATH=str(out/'repro')))
    e=b.read(raw);s.save_gzip(dest,e)
  else:e=s.read(dest)
  v.validate_run(e,g,w,a,0,0);d.require_bit_exact(e['result'],g)
  b.require(e['result']['total_ps']==31_124_000,'frozen threshold changed')
  b.require(not b.differences(old['result'],e['result']),'frozen legacy fields changed')
  b.require(old['memory_model']['calibration']==e['memory_model']['calibration'],'frozen native counters changed')
  if first:b.require(e['result']==first['result'] and e['memory_model']['calibration']==first['memory_model']['calibration'],'frozen repeats differ')
  first=e
 b.save(folder/'validation.json',dict(status='passed',time_us=31.124,repeats=2,old_fields_exact=True,native_counters_exact=True,reference=c['reference'],
    hashes={k:b.digest(c[k]) for k in ['architecture','workload','golden','reference']},outputs={p.name:b.digest(p) for p in folder.glob('*.json.gz')}))
 print('PASS frozen Me1 N3 31.124 us, two exact native repeats')

if __name__=='__main__':main()
