#!/usr/bin/env python3
"""Run repository unit suites and retain every attempt's actual evidence.

Does not download checkpoints, start GPU benchmarks, stage files, or change
source. Historical snapshots and frozen result directories are excluded from
pytest discovery. A supplied release binary is documented separately from
fresh Rust compilation; no archive binary is represented as freshly built.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import shlex
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--compiler',type=Path)
    parser.add_argument('--binary',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--suites',nargs='+',choices=('research','analytical','rust','main_rust'),
                        default=['research','analytical','rust'])
    parser.add_argument('--cargo-target',type=Path,default=Path('/tmp/plena-round2-cargo-target'))
    parser.add_argument('--main-rust-target',type=Path,default=Path('/tmp/mcl123-plena-layout-async-target'))
    parser.add_argument('--cargo',default='cargo')
    parser.add_argument('--rustc')
    args=parser.parse_args()
    repo=args.repo.resolve();args.out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.out/stamp;out.mkdir()
    env=os.environ.copy()
    test_tmp=Path('/tmp/plena-round2-test-tmp')
    test_tmp.mkdir(parents=True,exist_ok=True)
    env['TMPDIR']=str(test_tmp)
    if args.compiler:
        env['PLENA_DISPATCH_COMPILER']=str(args.compiler.resolve()/'research/moe_dispatch')
        env['PLENA_COMPILER_ROOT']=str(args.compiler.resolve())
    if args.rustc:env['RUSTC']=args.rustc
    if args.binary:env['PLENA_DISPATCH_TEST_BINARY']=str(args.binary.resolve())
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    except subprocess.CalledProcessError:commit='unavailable'
    packages={}
    for pkg in ('pytest','torch','numpy','pydantic','matplotlib','scipy','ortools'):
        try:packages[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:packages[pkg]='missing'
    receipt={'commit_at_start':commit,'python':sys.version,'python_executable':sys.executable,
             'package_versions':packages,'suites':[],'GPU_benchmarks_run':False,
             'checkpoint_inference_run':False,'compiler_source_override':env.get('PLENA_DISPATCH_COMPILER'),
             'binary':str(args.binary.resolve()) if args.binary else None,
             'binary_sha256':sha(args.binary) if args.binary and args.binary.is_file() else None,
             'binary_scope':'explicit existing binary for Python integration; fresh Rust unit build recorded separately',
             'temporary_directory':str(test_tmp),'cargo_target':str(args.cargo_target)}
    receipt['compiler_root_override']=env.get('PLENA_COMPILER_ROOT')
    receipt['cargo_command']=args.cargo
    receipt['rustc_override']=args.rustc
    receipt['build_environment']={k:env[k] for k in (
        'PATH','LIBTORCH','LIBTORCH_CXX11_ABI','CXX','CC','LIBRARY_PATH',
        'LD_LIBRARY_PATH','CARGO_BUILD_JOBS','RUSTFLAGS','TMPDIR',
        'PLENA_COMPILER_ROOT','PLENA_DISPATCH_COMPILER',
    ) if k in env}
    receipt['main_rust_target']=str(args.main_rust_target)
    for suite in args.suites:
        junit=out/f'{suite}.xml'
        if suite=='research':
            cmd=[sys.executable,'-m','pytest','research/moe_dispatch','-q',
                 '--ignore=research/moe_dispatch/archive','--ignore=research/moe_dispatch/results',
                 '--ignore=research/moe_dispatch/round2/results','--ignore=research/moe_dispatch/round2/archive',f'--junitxml={junit}']
        elif suite=='analytical':
            cmd=[sys.executable,'-m','pytest','analytic_models','-q',f'--junitxml={junit}']
        elif suite=='rust':
            cmd=[args.cargo,'test','--locked','--manifest-path','research/moe_dispatch/rust/Cargo.toml',
                 '--target-dir',str(args.cargo_target),'--','--test-threads=2']
        else:
            cmd=[args.cargo,'test','--locked','--manifest-path','transactional_emulator/Cargo.toml',
                 '--workspace','--target-dir',str(args.main_rust_target),'--','--test-threads=2']
        start=datetime.now(timezone.utc).isoformat()
        log=out/f'{suite}.log'
        print(f'Starting {suite}; log={log}',flush=True)
        with log.open('w') as stream:
            proc=subprocess.run(cmd,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
        record={'suite':suite,'command':cmd,'started_utc':start,
                'finished_utc':datetime.now(timezone.utc).isoformat(),
                'returncode':proc.returncode,'log':str(log),'log_sha256':sha(log)}
        text=log.read_text(errors='replace')
        if junit.exists():
            tree=ET.parse(junit)
            cases=tree.findall('.//testcase')
            record.update(tests=len(cases),failures=sum(c.find('failure') is not None for c in cases),
                          errors=sum(c.find('error') is not None for c in cases),
                          skipped=sum(c.find('skipped') is not None for c in cases),
                          junit_sha256=sha(junit))
            record['failed_tests']=[{'name':c.attrib.get('name'),'class':c.attrib.get('classname'),
                'message':(c.find('failure') if c.find('failure') is not None else c.find('error')).attrib.get('message','')[:2000]}
                for c in cases if c.find('failure') is not None or c.find('error') is not None]
        if proc.returncode:
            record['initial_classification']='missing_dependency' if 'ModuleNotFoundError' in text else (
                'missing_source_or_artifact' if 'FileNotFoundError' in text else 'unit_or_build_failure_review_required')
        else:record['initial_classification']='passed'
        receipt['suites'].append(record)
        (out/'UNIT_CHECKS.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ('failed_tests','command')}),flush=True)
        if proc.returncode:print('\n'.join(text.splitlines()[-45:]),flush=True)
    receipt['commit_at_end']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    receipt['all_requested_suites_passed']=all(r['returncode']==0 for r in receipt['suites'])
    (out/'UNIT_CHECKS.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    (out/'README.md').write_text('# Unit-check attempt\n\n'
        'Each suite retains its actual exit status, JUnit cases where available, '
        'and unedited log. Missing dependencies or historical artifacts are not '
        'reported as passes. No checkpoints or GPU benchmarks are launched.\n\n'
        f'Commit at start: `{commit}`\n\n'
        '```sh\n'+shlex.join([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]])+'\n```\n\n'
        'Use a fresh output directory or inspect the timestamped new attempt. '
        'Exact compiler override, supplied binary hash and package versions are '
        'recorded in UNIT_CHECKS.json.\n')
    with (out/'PROVENANCE.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=('file','sha256','commit','scope'))
        writer.writeheader()
        for file in sorted(out.iterdir()):
            if file.is_file() and file.name!='PROVENANCE.csv':
                writer.writerow({'file':file.name,'sha256':sha(file),'commit':commit,
                                 'scope':'actual unit-test attempt; no performance claim'})
    print('Receipt: '+str(out/'UNIT_CHECKS.json'),flush=True)
    raise SystemExit(0 if receipt['all_requested_suites_passed'] else 1)


if __name__=='__main__':main()
