"""Pair saved global lower bounds with final selected baseline observations."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from .common import ROOT, gm, sha, write_csv, write_json
from .config import BATCHES, MODES


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def run():
    result_path=ROOT/'E5/dispatch/per_window.json'
    result=json.loads(result_path.read_text())
    result=[r for r in result if r['constraint_group']=='C0' and r['design'] in ('B1','B2') and r['dispatch'] in ('fixed','milp')]
    indices={tuple(r[k] for k in ('onchip_mode','design','dispatch','window_id')):r for r in result}
    assert len(indices)==1080 and len(result)==1080,'All two modes x two baselines x two policies x 135 windows required'
    paths={mode:ROOT/('E2/' if mode=='pipelined' else 'E2/port_tight/')/'bounds_by_window.csv' for mode in MODES}
    bound={}
    for mode,path in paths.items():
        data=[r for r in read_csv(path) if r['set']=='heldout' and abs(float(r['bw_GBps'])-256)<1e-9]
        assert len(data)==135
        for r in data:bound[mode,r['window_id']]=r
    rows=[];paired=[]
    for mode in MODES:
        for dispatch in ('fixed','milp'):
            ids=[k[3] for k in indices if k[:3]==(mode,'B1',dispatch)]
            assert len(ids)==135 and len(set(ids))==135
            for wid in ids:
                b=bound[mode,wid];lb=float(b['lb_ms']);a=indices[mode,'B1',dispatch,wid];c=indices[mode,'B2',dispatch,wid]
                assert lb<=min(a['latency_ms'],c['latency_ms'])+1e-8,'Saved baseline below mandatory floor'
                paired.append(dict(onchip_mode=mode,dispatch=dispatch,window_id=wid,batch=a['batch'],lb_ms=lb,
                    B1_ms=a['latency_ms'],B2_ms=c['latency_ms'],LB_over_B1=lb/a['latency_ms'],LB_over_B2=lb/c['latency_ms']))
            for batch in (*BATCHES,'all'):
                rs=[r for r in paired if r['onchip_mode']==mode and r['dispatch']==dispatch and (batch=='all' or r['batch']==batch)]
                g1=100*(1-gm(r['LB_over_B1'] for r in rs));g2=100*(1-gm(r['LB_over_B2'] for r in rs))
                rows.append(dict(constraint_group='C0',onchip_mode=mode,dispatch=dispatch,batch=batch,n_windows=len(rs),
                    lower_bound_GM_ms=gm(r['lb_ms'] for r in rs),B1_GM_ms=gm(r['B1_ms'] for r in rs),B2_GM_ms=gm(r['B2_ms'] for r in rs),
                    max_gain_vs_B1_pct=g1,max_gain_vs_B2_pct=g2,gate5_reachable=g1>=5 and g2>=5,
                    basis='saved global LB paired with final development-selected C0 hardware under same dispatch/mode/256 cap; heldout observations'))
    out=ROOT/'E4';write_csv(out/'selected_baseline_headroom.csv',rows);write_csv(out/'selected_baseline_headroom_by_window.csv',paired)
    write_json(out/'SELECTED_BASELINE_HEADROOM_RECEIPT.json',dict(source_files={str(p.relative_to(ROOT)):sha(p) for p in [result_path,ROOT/'E4/selected_designs.json',*paths.values()]},
        script_sha256=sha(Path(__file__)),simulation_rerun=False,second_round_modified=False,rows=len(rows),paired_window_rows=len(paired),
        formula='100 * (1 - geometric_mean(global_lower_bound_ms/window_baseline_ms)); gate>=5% against BOTH B1 and B2',
        scope='heldout conditional mandatory-floor headroom; separate from development proof A and historical E2 frozen references'))
    print(json.dumps([r for r in rows if r['batch']=='all'],indent=2),flush=True)

if __name__=='__main__':run()
