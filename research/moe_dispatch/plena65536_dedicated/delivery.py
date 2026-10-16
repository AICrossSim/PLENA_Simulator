"""Deterministic search receipts and portable source/input/results archive."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_repeat(output):
    output=Path(output)
    a=json.loads((output/'SEARCH_CERTIFICATE.json').read_text())
    b=json.loads((output/'repeat_search/SEARCH_CERTIFICATE.json').read_text())
    assert a['selection_sha256']==b['selection_sha256']
    checked=[]
    for path in sorted(output.glob('*_exact.npz')):
        repeat=output/'repeat_search'/path.name
        with np.load(path) as first,np.load(repeat) as second:
            assert first.files==second.files
            for field in first.files:assert np.array_equal(first[field],second[field]),(path.name,field)
        checked.append({'file':path.name,'source_sha256':sha(path),'repeat_sha256':sha(repeat),'arrays_identical':True})
    assert len(checked)==24
    receipt={'status':'PASS','complete_search_repetitions':2,'profiles':3,'families_per_profile':8,
             'selection_sha256':a['selection_sha256'],'checked':checked}
    (output/'SEARCH_REPEAT_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def package(output):
    output=Path(output);verify_repeat(output)
    for name in ('REPORT_AUDIT.json','INDEPENDENT_SEARCH_AUDIT.json'):
        assert json.loads((output/name).read_text())['status']=='PASS'
    selection=json.loads((output/'FROZEN_SELECTION.json').read_text())
    for name,digest in selection['source_sha256'].items():assert sha(HERE/name)==digest
    report=json.loads((output/'REPORT_AUDIT.json').read_text())
    assert sha(HERE/'report.py')==report['report_source_sha256']
    paths=list(HERE.glob('*.py'))+[HERE/'README.md',HERE/'requirements.txt']
    paths+=[HERE.parent/'plena65536_shared'/n for n in ('geometry.py','timing.py')]
    paths+=[HERE.parent/'plena65536_moe'/n for n in ('model.py','workloads.py')]
    paths+=[HERE.parent/'compute_ceiling'/n for n in ('assignment_dp.py','trace_inputs.py')]
    manifest_path=REPO/'research/moe_dispatch/round2/results/E0/frozen_inputs.json'
    manifest=json.loads(manifest_path.read_text());input_dir=Path(manifest['input_directory'])
    inputs=[(input_dir/name,'inputs/'+name) for name in manifest['input_sha256']]
    for path,_ in inputs:assert sha(path)==manifest['input_sha256'][path.name]
    target=output/'PLENA65536_dedicated_complete.zip'
    results=[p for p in output.iterdir() if p.is_file() and p!=target and p.name!='DELIVERY_AUDIT.json']
    script='''#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$task_dir"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r research/moe_dispatch/plena65536_dedicated/requirements.txt
.venv/bin/python -m pytest research/moe_dispatch/plena65536_dedicated -q
.venv/bin/python -m research.moe_dispatch.plena65536_dedicated.run --out reproduced_results
'''
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in paths:archive.write(p,str(p.relative_to(REPO)))
        for p,name in inputs:archive.write(p,name)
        archive.writestr('research/moe_dispatch/round2/results/E0/frozen_inputs.json',
                         json.dumps(dict(manifest,input_directory='inputs'),indent=2)+'\n')
        archive.writestr('REPRODUCE.sh',script)
        for p in results:archive.write(p,'results/'+p.name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        count=len(archive.namelist());assert count==len(set(archive.namelist()))
    receipt={'status':'PASS','archive':str(target),'archive_bytes':target.stat().st_size,
             'archive_sha256':sha(target),'archive_members':count,
             'source_sha256':{str(p.relative_to(REPO)):sha(p) for p in paths},
             'raw_input_sha256':{p.name:sha(p) for p,_ in inputs},
             'strict_affinity':'C0Sharedonly,C1allRoutedonly','old_results_modified':False,
             'native_RTL_calibrated':False,'BLEN1_is_new_scalar_topology':True}
    (output/'DELIVERY_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','archive','archive_bytes','archive_members')}),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    package(parser.parse_args().out)
