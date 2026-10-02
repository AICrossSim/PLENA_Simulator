#!/usr/bin/env python3
"""Run and verify every declared evaluation point after committed PREREG.

This script never creates authorization or modifies architecture parameters.
"""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import study
from run_development import verify_stage

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=study.DEFAULT_ROOT);p.add_argument('--binary',type=Path,required=True);p.add_argument('--workers',type=int,default=20);p.add_argument('--timeout',type=int,default=3600);p.add_argument('--format-config',default='');a=p.parse_args()
 auth=json.loads((a.root/'heldout_authorization.json').read_text());assert auth.get('authorized') is True and auth.get('prereg_commit') and auth.get('frozen_signature')
 started=time.monotonic();receipts=[]
 for split in ('heldout','mixed_heldout','constructed'):
  command=[sys.executable,str(Path(__file__).with_name('study.py')),'run','--root',str(a.root),'--binary',str(a.binary),'--split',split,'--suite','all','--workers',str(a.workers),'--timeout',str(a.timeout)]
  if a.format_config:command+=['--format-config',a.format_config]
  subprocess.run(command,check=True);r=verify_stage(a.root,split,'all');assert r['signature']==auth['frozen_signature'];receipts.append(r)
  study.write(a.root/'final_pipeline_progress.json',dict(receipts=receipts,host_elapsed_seconds=time.monotonic()-started,prereg_commit=auth['prereg_commit']))
 for script in ('report.py','assess.py'):subprocess.run([sys.executable,str(Path(__file__).with_name(script)),'--root',str(a.root)],check=True)
 from report import coverage
 declared=coverage(a.root)
 assert all(int(r['declared_points'])==r['observed_points'] and int(r['declared_legal'])==r['complete'] and not any(r[k] for k in ('pending_points','unsupported','failed')) for r in declared),'Final declared matrix is incomplete or contains invalid legal points'
 study.write(a.root/'final_pipeline_receipt.json',dict(status='complete',receipts=receipts,coverage=declared,host_elapsed_seconds=time.monotonic()-started,prereg_commit=auth['prereg_commit'],frozen_signature=auth['frozen_signature'],launcher_sha256=study.sha(__file__)))
 subprocess.run([sys.executable,str(Path(__file__).with_name('render_report.py')),'--root',str(a.root)],check=True)
 print('COMPLETE: all declared development and evaluation points valid and duplicated',flush=True)
if __name__=='__main__':main()
