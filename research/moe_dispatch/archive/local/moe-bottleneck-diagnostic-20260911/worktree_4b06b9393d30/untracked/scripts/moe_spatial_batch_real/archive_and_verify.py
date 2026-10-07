#!/usr/bin/env python3
"""Verify finished evidence against receipts, preserve source and review diff."""
import difflib,gzip,hashlib,json,re,shutil,subprocess,tarfile
from pathlib import Path
import numpy as np
from prepare_routes import ROOT,OUT,sha

WORK=ROOT/'review_20260921/simulator-moe-batch-real'
OLD=ROOT/'review_20260919/simulator-moe-spatial-fabric'
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def report(path,hashes):
    assert len(hashes)==2 and hashes[0]==hashes[1]
    data=gzip.decompress(path.read_bytes());assert hashlib.sha256(data).hexdigest()==hashes[0],path
    return json.loads(data)

def main():
    counts={}
    for folder in ['route_campaign','sensitivity','real_campaign','real_compute','breakdown','real_breakdown','operand_ports']:
        v=json.loads((OUT/folder/'validation.json').read_text());assert v['passed'];counts[folder]=v
    assert json.loads((OUT/'repro/regression.json').read_text())['passed']
    assert json.loads((OUT/'real_inputs/repeat_validation.json').read_text())['passed']
    window=json.loads((OUT/'inputs/route_windows.json').read_text())
    for inv in json.loads((OUT/'inputs/archive_inventory.json').read_text()):
        assert sha(Path(inv['path']))==inv['sha256']
        with np.load(inv['path'],allow_pickle=False) as a:
            valid=a['valid'];assert np.all(np.diff(valid.astype(int),axis=1)<=0)
            for w in [w for w in window if w['source']==inv['path']]:
                ss=w['stream_indices'];step=w['decode_step'];li=w['moe_layer_index']
                assert np.all(valid[ss,step])
                assert a['decode_idx'][ss,step,li].tolist()==w['routes']
                assert a['decode_weight'][ss,step,li].astype(float).tolist()==w['route_weights']
                assert a['sample_ids'][ss].tolist()==w['sample_ids']
        selected=[w for w in window if w['source']==inv['path'] and w['batch']==16]
        assert len(selected)==2 and not(set(selected[0]['stream_indices'])&set(selected[1]['stream_indices']))
    n=0
    for p in (OUT/'route_campaign/receipts').glob('*.json'):
        x=json.loads(p.read_text());name=p.stem
        assert sha(OUT/'route_campaign/requests'/p.name)==x['request_sha256']
        rep=report(OUT/'route_campaign/reports'/(name+'.json.gz'),x['repeat_sha256'])
        assert rep['total_cycles']==x['row']['cycles'];n+=1
    assert n==4800
    raw_hash={};real_count=0
    for p in (OUT/'real_campaign').glob('real_b*/validation.json'):
        x=json.loads(p.read_text());assert x['passed']
        for c in x['checks']:
            phase=c['phase'];req=p.parent/(phase+'.request.json');mf=p.parent/(phase+'.operands.json')
            assert sha(req)==c['request_sha256'] and sha(mf)==c['operands_sha256']
            rep=report(p.parent/(phase+'.report.json.gz'),c['repeat_sha256'])
            assert rep['numerical_bit_exact'] and rep['drained']
            assert rep['invocation_sha256']==c['invocation_sha256'] and rep['service_sha256']==c['service_sha256']
            for j in json.loads(mf.read_text())['jobs']:
                for kind in ['x','w']:
                    path=Path(j[kind]['path']);raw_hash.setdefault(str(path),None)
                    if raw_hash[str(path)] is None:raw_hash[str(path)]=sha(path)
                    assert raw_hash[str(path)]==j[kind]['sha256']
            real_count+=1
        assert sha(Path(x['output']['path']))==x['output']['sha256']
    assert real_count==240
    for folder,expected in [('breakdown',120),('real_breakdown',24),('operand_ports',160)]:
        seen=0
        for p in (OUT/folder).glob('*/validation.json'):
            x=json.loads(p.read_text());assert x['passed']
            rep=report(p.parent/('report.json.gz' if folder=='operand_ports' else 'trace.json.gz'),x['repeat_sha256'])
            assert rep['drained'];seen+=1
        assert seen==expected,(folder,seen)
    n=0
    for p in (OUT/'real_compute').glob('*.validation.json'):
        x=json.loads(p.read_text());rep=report(p.with_name(p.name.replace('.validation.json','.json.gz')),x['pure_repeat_sha256'])
        assert rep['total_cycles']==x['pure_cycles'] and rep['requests_drained'];n+=1
    assert n==240
    testlog=(OUT/'logs/workspace_tests.log').read_text()
    passed=sum(map(int,re.findall(r'test result: ok\. (\d+) passed',testlog)))
    assert passed==341 and 'test result: FAILED' not in testlog
    assert 'Finished' in (OUT/'logs/clippy.log').read_text()
    # The inherited normal-engine work is unchanged. This study's additions are
    # opt-in spatial-M source and experiment runners, not a new default model.
    def git(path,*args):return subprocess.check_output(['git','-C',str(path),*args])
    assert git(WORK,'rev-parse','HEAD')==git(OLD,'rev-parse','HEAD')
    assert git(WORK,'diff','--binary')==git(OLD,'diff','--binary')
    target=WORK/'scripts/moe_spatial_batch_real';target.mkdir(exist_ok=True)
    for p in (OUT/'scripts').glob('*.py'):shutil.copy2(p,target/p.name)
    shutil.copy2(OUT/'REPRODUCE.md',target/'README.md')
    shutil.copy2(OUT/'METHODS.md',WORK/'doc/moe_spatial_batch_real_contract.md')
    changed=['transactional_emulator/src/moe_spatial/mod.rs','transactional_emulator/src/moe_spatial/fabric.rs',
        'transactional_emulator/src/moe_spatial/fabric/tests.rs','transactional_emulator/src/moe_spatial/operands.rs',
        'transactional_emulator/src/bin/moe_spatial_fabric.rs','doc/moe_spatial_batch_real_contract.md']
    changed += [str(p.relative_to(WORK)) for p in target.iterdir() if p.is_file()]
    patch=[]
    for rel in sorted(changed):
        before=(OLD/rel).read_text().splitlines(keepends=True) if (OLD/rel).exists() else []
        after=(WORK/rel).read_text().splitlines(keepends=True)
        patch.extend(difflib.unified_diff(before,after,fromfile='a/'+rel if before else '/dev/null',tofile='b/'+rel))
    (OUT/'repro/changes_vs_frozen_spatial_fabric.patch').write_text(''.join(patch))
    manifest={rel:sha(WORK/rel) for rel in changed};save(OUT/'repro/new_source_manifest.json',manifest)
    with tarfile.open(OUT/'repro/source_snapshot.tar.gz','w:gz') as tar:
        for p in sorted(WORK.rglob('*')):
            rel=p.relative_to(WORK)
            if p.is_file() and '.git' not in rel.parts and '__pycache__' not in rel.parts:
                tar.add(p,arcname=str(rel),recursive=False)
    # Preserve exact pure binary rather than depending on a mutable build cache.
    shutil.copy2('/tmp/plena-moe-dual-core-target/release/moe_spatial_m',OUT/'repro/moe_spatial_m')
    audit=dict(passed=True,route_and_sensitivity_receipts=4800,real_value_phases=240,trace_receipts=144,
        actual_value_operand_port_receipts=160,real_pure_receipts=240,source_archives_checked=9,
        route_windows_reconstructed=72,disjoint_holdout_verified=True,source_and_operand_hashes_verified=True,
        inherited_tracked_source_unchanged=True,rust_test_invocations=passed,source_worktree=str(WORK),
        validated_campaign_runs=json.loads((OUT/'summary/conclusions.json').read_text())['validated_campaign_simulator_runs'])
    save(OUT/'archive_audit.json',audit)
    setup=json.loads((OUT/'setup.json').read_text());setup.update(status='complete',report=str(OUT/'REPORT_ZH.md'),audit=str(OUT/'archive_audit.json'))
    save(OUT/'setup.json',setup)
    save(OUT/'DELIVERABLES.json',dict(report='REPORT_ZH.md',methods='METHODS.md',reproduction='REPRODUCE.md',
        tables='summary',source=str(WORK),patch='repro/changes_vs_frozen_spatial_fabric.patch',
        raw_roots=['route_campaign','sensitivity','real_campaign','real_compute','breakdown','real_breakdown','operand_ports'],
        archive_audit='archive_audit.json',status='complete'))
    inventory={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and
        '__pycache__' not in p.parts and p.name!='artifact_manifest.json' and str(p.relative_to(OUT))!='logs/archive.log'}
    save(OUT/'artifact_manifest.json',inventory)
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
