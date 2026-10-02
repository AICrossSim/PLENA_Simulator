#!/usr/bin/env python3
"""Run only development, stopping on genuine validation failures.

The comparator's freeze-ready receipt does not authorize heldout. The parent
researcher must commit PREREG and publish heldout_authorization.json separately.
"""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import study

def stage_receipt(root,split,suite,regex=''):
 name=f'{split}_{suite}'
 if regex:name+='_'+hashlib.sha256(('|'+regex).encode()).hexdigest()[:8]
 return root/f'{name}_receipt.json'

def verify_stage(root,split,suite,regex=''):
 path=stage_receipt(root,split,suite,regex);r=json.loads(path.read_text())
 assert r['points']==r['complete']+r['excluded'],f'Failed or unsupported legal points in {path}: {r}'
 assert r['failed']==r['unsupported']==0
 assert r['signature']==json.loads((root/'active_result_signature.json').read_text())
 return dict(path=str(path),sha256=study.sha(path),points=r['points'],complete=r['complete'],excluded=r['excluded'],host_elapsed_seconds=r['host_elapsed_seconds'],signature=r['signature'])

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=study.DEFAULT_ROOT);p.add_argument('--binary',type=Path,required=True);p.add_argument('--workers',type=int,default=12);p.add_argument('--timeout',type=int,default=3600);p.add_argument('--format-config',default='');p.add_argument('--mode',choices=('stress','comparator','development'),default='development');a=p.parse_args()
 command=[sys.executable,str(Path(__file__).with_name('study.py')),'run','--root',str(a.root),'--binary',str(a.binary),'--workers',str(a.workers),'--timeout',str(a.timeout)]
 if a.format_config:command+=['--format-config',a.format_config]
 started=time.monotonic();receipts=[]
 def run(split,suite,regex=''):
  c=command+['--split',split,'--suite',suite]
  if regex:c+=['--filter-regex',regex]
  print('START',split,suite,regex,flush=True);subprocess.run(c,check=True)
  receipts.append(verify_stage(a.root,split,suite,regex))
  study.write(a.root/'development_pipeline_progress.json',dict(mode=a.mode,receipts=receipts,host_elapsed_seconds=time.monotonic()-started,heldout_started=False))
 run('mixed_development','main',r't128.*__BL[2345]__OP2__iso__')
 if a.mode!='stress':
  run('development','dev_comparator')
  c=[sys.executable,str(Path(__file__).with_name('study.py')),'comparator','--root',str(a.root)]
  if a.format_config:c+=['--format-config',a.format_config]
  subprocess.run(c,check=True)
 if a.mode=='development':
  run('development','all');run('mixed_development','all')
 study.write(a.root/'development_pipeline_receipt.json',dict(status='complete',mode=a.mode,receipts=receipts,host_elapsed_seconds=time.monotonic()-started,heldout_started=False,launcher_sha256=study.sha(__file__)))
 print('COMPLETE development pipeline; heldout remains blocked pending committed PREREG',flush=True)
if __name__=='__main__':main()
