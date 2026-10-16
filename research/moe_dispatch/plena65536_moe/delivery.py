"""Verify deterministic search receipts and package source+inputs+results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_search_repeat(output):
    output=Path(output)
    first=json.loads((output/'SEARCH_CERTIFICATE.json').read_text())
    second=json.loads((output/'repeat_search/SEARCH_CERTIFICATE.json').read_text())
    assert first['selection_sha256']==second['selection_sha256']
    checked=[]
    for path in sorted(output.glob('*_search.npz')):
        repeat=output/'repeat_search'/path.name
        assert repeat.exists()
        with np.load(path) as a, np.load(repeat) as b:
            assert a.files==b.files
            for name in a.files:
                assert np.array_equal(a[name],b[name],equal_nan=True),(path.name,name)
        checked.append({'file':path.name,'sha256':sha(path),'repeat_sha256':sha(repeat),
                        'all_arrays_bitwise_identical':True})
    assert len(checked)==18
    receipt={'status':'PASS','complete_search_repetitions':2,
             'profiles':3,'domains_per_profile':6,'selection_sha256':first['selection_sha256'],
             'checked':checked,'elapsed_seconds_are_not_compared':True}
    (output/'SEARCH_REPEAT_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def verify_search_certificate(output):
    """Check every saved domain score and exact pruning inventory."""
    from research.moe_dispatch.plena65536_shared.geometry import enumerate_cores
    output=Path(output)
    frozen=json.loads((output/'FROZEN_SELECTION.json').read_text())
    cores=enumerate_cores()
    budgets=np.asarray([c.macs for c in cores],dtype=np.int64)
    rows=[]
    for profile,families in frozen['profile_optima'].items():
        for family,selection in families.items():
            if family=='original':
                continue
            path=output/f'{family}_{profile}_search.npz'
            with np.load(path) as data:
                ids,lb,actual=(data[k] for k in ('pair_ids','lower_bound_GM_cycles','exact_GM_cycles'))
                total=budgets[ids[:,0]]+np.where(ids[:,1]>=0,budgets[np.maximum(ids[:,1],0)],0)
                assert np.all(total==65536)
                is_exact=np.isfinite(actual)
                best=selection['development_GM_cycles']
                assert abs(float(actual[is_exact].min())-best)<1e-7
                assert np.all(lb[is_exact]<=actual[is_exact]+1e-7)
                assert np.all(lb[~is_exact]>best+1e-7)
                certificate=selection['certificate']
                assert len(ids)==certificate['geometries']
                assert int(is_exact.sum())==certificate['exact_evaluated']
                assert int((~is_exact).sum())==certificate['pruned_by_admissible_LP']
                rows.append({'domain':family,'profile':profile,'geometries':len(ids),
                             'integer_assignments_evaluated':int(is_exact.sum()),
                             'certified_pruned':int((~is_exact).sum()),'sha256':sha(path)})
    assert len(rows)==18
    receipt={'status':'PASS','rows':rows,'certificate_scope':frozen['contract'],
             'heldout_used_for_search':False,'equal_multiplier_budget':65536}
    (output/'CERTIFICATE_FILE_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def package(output):
    output=Path(output)
    verify_search_repeat(output)
    verify_search_certificate(output)
    for name in ('REPORT_AUDIT.json','PLOT_AUDIT.json','TIMING_REPEAT_AUDIT.json','SPLIT_DIAGNOSTIC.json'):
        assert json.loads((output/name).read_text())['status']=='PASS',name
    source_paths=list(HERE.glob('*.py'))+[HERE/'README.md',HERE/'requirements.txt']
    source_paths += [HERE.parent/'plena65536_shared'/name for name in ('geometry.py','timing.py','dataflow.py')]
    source_paths += [HERE.parent/'compute_ceiling'/name for name in ('assignment_dp.py','trace_inputs.py')]
    source_hashes={str(p.relative_to(REPO)):sha(p) for p in source_paths}
    for source,value in json.loads((output/'FROZEN_SELECTION.json').read_text())['search_source_sha256'].items():
        assert sha(HERE/source)==value,source
    original_manifest=REPO/'research/moe_dispatch/round2/results/E0/frozen_inputs.json'
    manifest=json.loads(original_manifest.read_text())
    input_dir=Path(manifest['input_directory'])
    inputs=[]
    for name,digest in manifest['input_sha256'].items():
        path=input_dir/name
        assert sha(path)==digest
        inputs.append((path,'inputs/'+name))
    portable=dict(manifest,input_directory='inputs')
    package_path=output/'plena65536_shared_plus_moe_complete.zip'
    artifact_paths=[p for p in output.iterdir() if p.is_file() and p!=package_path
                    and p.name not in ('DELIVERY_AUDIT.json','report_execution.log')]
    reproduce='''#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$task_dir"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r research/moe_dispatch/plena65536_moe/requirements.txt
.venv/bin/python -m pytest research/moe_dispatch/plena65536_moe -q
.venv/bin/python -m research.moe_dispatch.plena65536_moe.run --out reproduced_results
'''
    with zipfile.ZipFile(package_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in source_paths:
            archive.write(p,str(p.relative_to(REPO)))
        for p,name in inputs:
            archive.write(p,name)
        archive.writestr('research/moe_dispatch/round2/results/E0/frozen_inputs.json',json.dumps(portable,indent=2)+'\n')
        archive.writestr('REPRODUCE.sh',reproduce)
        for p in artifact_paths:
            archive.write(p,'results/'+p.name)
    with zipfile.ZipFile(package_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist())==len(set(archive.namelist()))
        member_count=len(archive.namelist())
    receipt={'status':'PASS','scope':'Shared + all real Top6 active MoE experts',
        'source_sha256':source_hashes,'raw_inputs':{p.name:sha(p) for p,_ in inputs},
        'archive':str(package_path),'archive_bytes':package_path.stat().st_size,
        'archive_sha256':sha(package_path),'archive_members':member_count,
        'contains_raw_routing_inputs':True,'portable_manifest':True,
        'WS_IS_are_operand_retention_extensions_not_original_PE_modes':True,
        'native_RTL_calibration_completed':False}
    (output/'DELIVERY_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','archive','archive_bytes','archive_members')}),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    package(parser.parse_args().out)
