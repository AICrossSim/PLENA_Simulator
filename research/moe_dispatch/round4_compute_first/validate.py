"""Independently audit delivered ledgers/rows, not just producer exit codes."""
import hashlib,json,math,platform,subprocess,sys
from pathlib import Path
from .common import ROOT,inputs,write_json,digest
from .requirements import load_rows
from .evaluate import selected,unique,io_bytes,flops
from .config import TOTAL_MACS

def run():
    expected=['E1/selected_designs.json','E1/compute_by_window.csv','E1/compute_main_table.csv',
       'E2/sram_requirement.csv','E2/balance_bandwidth.csv','E3/four_metrics.csv','E3/four_metrics_ZH.md',
       'E3/system_by_window.csv','E4/crossover.csv','E4/fig_speedup_vs_bw.pdf','E4/HETERO_VS_PERTASK_DATAFLOW.md',
       'REPORT_ZH.md','README.md']
    assert all((ROOT/f).is_file() and (ROOT/f).stat().st_size for f in expected)
    ws=inputs(); held={w['id']:w for w in ws['heldout']};sd=selected()
    assert len(ws['development'])==18 and len(held)==135
    for w in list(held.values())+ws['development']:
        assert sum(e['Me'] for e in w['experts'] if not e.get('is_shared'))==6*w['batch']
        assert sum(e['Me'] for e in w['experts'] if e.get('is_shared'))==w['batch']
    for s in sd.values():
        assert sum(c['pm']*c['pn']*c['pk'] for c in s['cores'])==TOTAL_MACS
        assert all(1<=c['pm']<=16 and c['pn']<=s['pn_cap'] for c in s['cores'])
        d=s['real_sram_design'];assert d is not None
        assert sum(sum(d[k]) for k in ('w_bytes','x_bytes','acc_bytes','z_bytes'))==532*1024
        assert [sum(d[k]) for k in ('w_banks','x_banks','acc_banks','vector_lanes')]==[64,24,12,64]
    e1=load_rows(ROOT/'E1/compute_by_window.csv');e3=load_rows(ROOT/'E3/system_by_window.csv')
    assert len(e1)==len(e3)==len(sd)*135*2
    ix={(r['mode'],r['design'],r['window_id']):r for r in e1}
    for r in e1+e3:
        w=held[r['window_id']];n=int(r['core_count']);T=r['Tc_ms']*1e6
        assert r['flops']==flops(w) and r['unique_bytes']==unique(w)
        assert r['hbm_bytes']>=unique(w)+io_bytes(w)
        assert sum(r[f'core{j}_tasks'] for j in range(n))==len(w['experts'])
        assert abs(sum(r[f'core{j}_work_pct'] for j in range(n))-100)<1e-8
        assert 0<r['mac_util_pct']<=100+1e-7
        if r['mode']=='B':assert r['hbm_bytes']==unique(w)+io_bytes(w)
        if r['mode']=='A':assert T+1e-6>=ix['B',r['design'],r['window_id']]['Tc_ms']*1e6
        if r['mode']=='system':
            assert T+1e-3>=r['hbm_bytes']/r['bw_GBps']
            assert r['credits']*32 >=65*r['bw_GBps']
    counts={}
    for stage in ('E1','E3'):
        checks=json.loads((ROOT/stage/'repeat_checks.json').read_text())
        assert len(checks)==len(sd)*270 and all(c['repeat_exact'] for c in checks)
        assert len({(c['design'],c['window'],c['mode']) for c in checks})==len(checks)
        counts[stage]=len(checks)
    # Independently recompute every E3 aggregate: use GM of per-window maxima,
    # never max of aggregate GMs.
    metrics=load_rows(ROOT/'E3/four_metrics.csv')
    for q in metrics:
        rs=[r for r in e1 if r['mode']==q['mode'] and r['design']==q['design'] and (q['batch']=='all' or r['batch']==q['batch'])]
        est=[max(r['Tc_ms'],r['hbm_bytes']/(q['bw_GBps']*1e6)) for r in rs]
        val=math.exp(sum(math.log(t) for t in est)/len(est))
        assert math.isclose(val,q['est_layer_ms'],rel_tol=1e-12)
        assert q['eq_fetch_GBps']<=q['bw_GBps']*(1+1e-12)
    repo=ROOT.parents[2]
    subprocess.run(['git','diff','--exit-code','a33903ccdc47b3982cd0ce54f30277e9649529bf','--',
       'research/moe_dispatch/round2','research/moe_dispatch/round3'],cwd=repo,check=True,capture_output=True)
    old={name:subprocess.check_output(['git','rev-parse',f'HEAD:research/moe_dispatch/{name}'],cwd=repo,text=True).strip() for name in ('round2','round3')}
    tests=subprocess.run([sys.executable,'-m','pytest',str(ROOT/'test_round4.py'),'-q'],cwd=repo,capture_output=True,text=True)
    (ROOT/'TEST_RESULTS.txt').write_text(tests.stdout+tests.stderr)
    assert tests.returncode==0,tests.stdout+tests.stderr
    receipt=dict(status='PASS',required_files=expected,selected_designs=len(sd),windows=135,
        ledger_rows=len(e1)+len(e3),repeat_points=counts,four_metric_aggregates=len(metrics),
        second_third_round_unchanged=True,tests=tests.stdout.strip(),
        no_native_datapath_calibration=True)
    write_json(ROOT/'VALIDATION.json',receipt)
    provenance=dict(python=sys.version,platform=platform.platform(),input_sha256=digest(ws),
        frozen_old_tree=old,scope='phase-fluid analytical simulation; uncalibrated hardware timing',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))},
        result_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.rglob('*'))
                      if p.is_file() and p.suffix in ('.csv','.json','.md','.pdf','.gz') and p.name not in ('PROVENANCE.json','COMMITS.json')})
    write_json(ROOT/'PROVENANCE.json',provenance)
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
