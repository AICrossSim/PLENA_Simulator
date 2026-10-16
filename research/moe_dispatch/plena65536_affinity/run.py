"""Reproduce the complete frozen-geometry affinity experiment and its audits."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main(out,*,reuse_search=False,skip_tests=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    prefix='research.moe_dispatch.plena65536_affinity.'
    commands=[]
    if not reuse_search:
        commands.extend([('search',out),('search',out/'repeat_search')])
    commands.extend([('report',out),('assignment_gap',out),('independent_audit',out)])
    for module,path in commands:
        flag='--output' if module=='assignment_gap' else '--out'
        subprocess.run([sys.executable,'-m',prefix+module,flag,str(path)],check=True,env=env)
    if not skip_tests:
        result=subprocess.run([sys.executable,'-m','pytest','research/moe_dispatch/plena65536_affinity','-q'],
                              env=env,text=True,capture_output=True)
        (out/'TEST_OUTPUT.txt').write_text(result.stdout+result.stderr)
        result.check_returncode()
        (out/'TEST_AUDIT.json').write_text(json.dumps({'status':'PASS','command':'pytest research/moe_dispatch/plena65536_affinity -q',
                                                       'output':result.stdout.strip()},indent=2)+'\n')
    subprocess.run([sys.executable,'-m',prefix+'delivery','--out',str(out)],check=True,env=env)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reuse-search',action='store_true');p.add_argument('--skip-tests',action='store_true')
    args=p.parse_args();main(args.out,reuse_search=args.reuse_search,skip_tests=args.skip_tests)
