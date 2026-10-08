"""Execution receipts kept wholly outside the frozen second-round results."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
from ..common import ROOT, sha, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--label',required=True)
    ap.add_argument('command',nargs=argparse.REMAINDER);args=ap.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:ap.error('a command is required')
    directory=Path(__file__).resolve().parent;repo=ROOT.parents[2]
    folder=directory/'executions';folder.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    receipt=folder/f'{args.label}_{stamp}.json';log=folder/f'{args.label}_{stamp}.log'
    env=dict(os.environ)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[name]='1'
    env['PYTHONHASHSEED']='20261008'
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'model.py',ROOT/'predictors.py',ROOT/'optimizer.py']}
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in sorted(directory.glob('*.py'))})
    rec={'command':command,'label':args.label,'started_utc':datetime.now(timezone.utc).isoformat(),
        'execution_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        'source_sha256':sources,'input_manifest_sha256':sha(ROOT/'results/E0/frozen_inputs.json'),
        'log':str(log.relative_to(directory)),'returncode':None}
    write_json(receipt,rec);start=time.monotonic()
    with log.open('w') as f:proc=subprocess.run(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
    rec.update(returncode=proc.returncode,elapsed_seconds=time.monotonic()-start,
        finished_utc=datetime.now(timezone.utc).isoformat(),log_sha256=sha(log),
        source_unchanged_during_execution={n:sha(ROOT/n)==v for n,v in sources.items()})
    write_json(receipt,rec)
    print({'receipt':str(receipt),'returncode':proc.returncode,'elapsed_seconds':rec['elapsed_seconds']},flush=True)
    raise SystemExit(proc.returncode)


if __name__=='__main__':main()
