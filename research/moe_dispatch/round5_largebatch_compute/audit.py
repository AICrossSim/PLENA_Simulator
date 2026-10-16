"""Check task accounting, bounds, heldout separation and table identities."""
import csv,hashlib,json,math,gzip
from .common import ROOT,BUDGETS,BATCHES,write_json,digest
from .inputs import load
from .model import Core,configurations,build_tasks
from .search import weights

def rows(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return list(csv.DictReader(f))

def audit():
    from .pack_bounds import pack
    pack()
    data=load();cases=[];raw_count=0;selected_count=0;grid_count=0
    for split,ws in data.items():
        assert len(ws)==90
        assert len({w['group'] for w in ws})==5
        assert len({w['id'] for w in ws})==len(ws)
        for b in BATCHES:assert sum(w['batch']==b for w in ws)==15
        for w in ws:
            e=w['experts'];assert sum(x['Me'] for x in e if not x['is_shared'])==6*w['batch']
            assert sum(x['Me'] for x in e if x['is_shared'])==w['batch']
            assert all(len(set(r))==6 and min(r)>=0 and max(r)<64 for r in w['token_routes'])
    assert not {q for w in data['development'] for q in w['source_requests']} & {q for w in data['heldout'] for q in w['source_requests']}
    for total in BUDGETS:
        nc=sum(1 for _ in configurations(total))
        for mode in ('matrix','full_scaled'):
            for gran in ('expert','shared64'):
                p=ROOT/'results'/f'{total}_{mode}_{gran}';sel=json.loads((p/'selected.json').read_text())
                receipt=json.loads((p/'receipt.json').read_text());held=json.loads((p/'holdout_receipt.json').read_text())
                assert receipt['candidates']==nc and receipt['grid_bitwise_equal']
                assert receipt['dev_digest']==digest(data['development'])
                assert held['input_digest']==digest(data['heldout']) and held['all_point_repeats_exact']
                assert held['selection_digest']==digest(sel)
                assert len(sel)==20;selected_count+=len(sel);grid_count+=nc
                for s in sel.values():
                    assert sum(Core(**c).macs for c in s['cores'])==total
                    assert not s['selection_uses_heldout']
                    assert s['dev_global_lower_cycles']<=s['dev_upper_cycles']+1e-6
                rr=rows(p/'holdout.csv.gz');raw_count+=len(rr)
                assert len(rr)==90*20*(7 if mode=='full_scaled' else 6)
                assert (p/'runtime_fix_receipt.json').exists()
                for r in rr:
                    c=int(r['cycles']);u=int(r['useful_MACs']);issued=int(r['issued_MACs']);b=int(r['batch'])
                    assert int(r['lower_cycles'])<=c and c>0 and issued>=u
                    assert u==3*2048*1408*8*b
                    assert math.isclose(float(r['ms']),c/1e6)
                    assert math.isclose(float(r['GFLOPs']),2*u/c)
                    assert math.isclose(float(r['utilization_pct']),100*u/(total*c))
                    assert float(r['utilization_pct'])<=100+1e-6
                    assert all(x<=c for x in json.loads(r['core_cycles']))
                    assert r['repeat_exact']=='True'
                cases.append(dict(total_macs=total,mode=mode,granularity=gran,candidates=nc,heldout_rows=len(rr)))
    for r in rows(ROOT/'report/main.csv'):
        assert math.isclose(float(r['speedup_vs_single']),float(r['single_ms'])/float(r['heterogeneous_ms']))
        assert math.isclose(float(r['speedup_vs_homogeneous']),float(r['homogeneous_ms'])/float(r['heterogeneous_ms']))
    files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*')) if p.is_file() and p.suffix in ('.py','.cpp','.json','.csv','.npz','.gz') and p.name!='AUDIT.json'}
    write_json(ROOT/'AUDIT.json',dict(passed=True,cases=cases,enumerated_config_case_count=grid_count,selected_family_case_count=selected_count,
        heldout_metric_rows=raw_count,development_windows=90,heldout_windows=90,
        all_declared_invariants_passed=True,source_and_result_sha256=files,
        limitation='Accounting/reproducibility audit, not RTL calibration or area/timing closure.'))
    print('AUDIT PASSED',grid_count,'configuration cases;',raw_count,'holdout metric rows',flush=True)

if __name__=='__main__':audit()
