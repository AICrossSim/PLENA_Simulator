"""Record actual campaign command, immutable source hashes, and exit status."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from .common import ROOT,write_json,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True)
    ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not cmd:ap.error('a command is required after --')
    repo=ROOT.parents[2];folder=ROOT/'results/executions';folder.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target=folder/(a.label+'_'+stamp+'.json');log=folder/(a.label+'_'+stamp+'.log')
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]='1'
    env['PYTHONHASHSEED']='20261007'
    rec={'label':a.label,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat(),
        'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'source_sha256':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},
        'frozen_input_manifest_sha256':sha(ROOT/'results/E0/frozen_inputs.json'),
        'environment':{k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED')},
        'log':str(log.relative_to(ROOT)),'returncode':None}
    write_json(target,rec);start=time.monotonic()
    with log.open('w') as f:proc=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
    rec.update(returncode=proc.returncode,elapsed_seconds=time.monotonic()-start,
        finished_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha(log))
    write_json(target,rec)
    print(json.dumps({'label':a.label,'returncode':proc.returncode,'log':str(log),'elapsed_seconds':rec['elapsed_seconds']}),flush=True)
    raise SystemExit(proc.returncode)
if __name__=='__main__':main()
