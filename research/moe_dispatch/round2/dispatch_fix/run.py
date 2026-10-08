"""Repeated frozen-hardware dispatch repair campaign, development selection first."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from ..common import (ROOT, canonical, decode_design, encode_design, gmean,
                      inputs, read_csv, sha, write_csv, write_json)
from ..model import Parameters, simulate as old_simulate, task_cost
from ..optimizer import solve_assignment
from ..predictors import Predictor
from ..run import designs_from_selection, selection_file
from .runtime import simulate as fixed_simulate, refetch_factor

DIRECTORY=Path(__file__).resolve().parent
NAMES=('B0','B1','B2','best_hetero','fixed_4+2')
MODES=('pipelined','port_tight')
CONFIGS=tuple((t,flag) for flag in (False,True) for t in (2,3,4,6,8))
_WS=None
_DESIGNS=None


def _init_worker():
    global _WS,_DESIGNS
    _WS=inputs()
    selection=selection_file()
    _DESIGNS={mode:{name:designs_from_selection(selection,mode)[name] for name in NAMES}
              for mode in MODES}


def digest(result):
    return hashlib.sha256(canonical(result).encode()).hexdigest()


def _refetch_rows(w,d,p,result):
    rows=[]
    binds={b['expert_index']:b for b in result['bindings']}
    for t in result['tasks']:
        i=t['expert_index'];e=w['experts'][i];c=t['core']
        co=task_cost(e,d,c,p)
        if co.hbm_bytes<=co.unique_hbm_bytes:
            continue
        choices=[]
        for ci in range(len(d.cores)):
            try: candidate=task_cost(e,d,ci,p)
            except ValueError: continue
            choices.append((candidate.hbm_bytes,ci,candidate))
        _,best,bc=min(choices,key=lambda x:(x[0],x[1]))
        b=binds[i]
        rows.append({'expert_index':i,'expert_id':e.get('id',i),'Me':e['Me'],
            'is_shared':bool(e.get('is_shared',False)),'core':c,
            'refetch_factor':refetch_factor(co),'best_refetch_factor':refetch_factor(bc),
            'best_refetch_core':best,'chosen_hbm_bytes':co.hbm_bytes,
            'minimum_hbm_bytes':bc.hbm_bytes,'excess_B':co.hbm_bytes-bc.hbm_bytes,
            'z_chunks':co.z_chunks,'spill_bytes':co.spill_bytes,
            'decision_kind':b.get('decision_kind','original_eft_or_lpt'),
            'bind_cycle':b['bind_cycle'],'candidate_comparisons':b.get('candidate_comparisons',[])})
    return rows


def summarize(w,d,p,result,dispatch,repeat_digest,solver=None):
    refetch=_refetch_rows(w,d,p,result)
    shared=[]
    for t in result['tasks']:
        e=w['experts'][t['expert_index']]
        if not e.get('is_shared',False):continue
        co=task_cost(e,d,t['core'],p)
        shared.append({'expert_id':e.get('id',t['expert_index']),'core':t['core'],
            'start_cycles':t['start'],'start_ms':t['start']/1e6,
            'finish_cycles':t['finish'],'nominal_cycles':co.isolated_cycles,
            'refetch_factor':refetch_factor(co),'z_chunks':co.z_chunks})
    return {'window_id':w['id'],'batch':w['batch'],'dataset':w.get('dataset'),
        'dispatch':dispatch,'cycles':result['cycles'],'latency_ms':result['latency_ms'],
        'hbm_bytes':result['hbm_bytes'],'native_unique_bytes':result['native_unique_bytes'],
        'refetch_tasks':len(refetch),'refetch_details':refetch,'shared_tasks':shared,
        'result_digest':digest(result),'repeat_digest':repeat_digest,
        'solver_status':None if solver is None else solver['status'],
        'solver_assignment_digest':None if solver is None else digest(solver),
        'core_finish_cycles':result['core_finish_cycles'],
        'hbm_busy_frac':result['hbm_busy_frac'],'core_compute_busy':result['core_compute_busy'],
        'dispatch_deferrals':len(result.get('dispatch_decisions',[])),
        'bindings':result['bindings'],
        'tasks':[{'expert_index':t['expert_index'],'core':t['core'],'start':t['start'],
                  'finish':t['finish'],'predicted_cycles':t['predicted_cycles'],
                  'nominal_cycles':t['nominal_cycles']} for t in result['tasks']]}


def _baseline_job(job):
    mode,name=job;d=_DESIGNS[mode][name];p=Parameters(onchip_mode=mode)
    out=[]
    for w in _WS['heldout']:
        a=old_simulate(w,d,p);b=old_simulate(w,d,p)
        assert canonical(a)==canonical(b),(mode,name,w['id'],'old repeat')
        r=summarize(w,d,p,a,'eft_old',digest(b));r.update(design=name,onchip_mode=mode)
        out.append(r)
        a_s=solve_assignment(w,d,p);b_s=solve_assignment(w,d,p)
        assert canonical(a_s)==canonical(b_s),(mode,name,w['id'],'assignment repeat')
        a=old_simulate(w,d,p,owners=a_s['owners']);b=old_simulate(w,d,p,owners=b_s['owners'])
        assert canonical(a)==canonical(b),(mode,name,w['id'],'milp repeat')
        r=summarize(w,d,p,a,'milp',digest(b),a_s);r.update(design=name,onchip_mode=mode)
        out.append(r)
    return out


def _sequence(ws,d,p,dispatch,t_big,large_first,predictor):
    func=fixed_simulate if dispatch=='fixed' else old_simulate
    kw={'t_big':t_big,'large_first':large_first} if dispatch=='fixed' else {'policy':'eft'}
    # Single hardware explicitly retains E4's predictor=None path, both old/fixed.
    if len(d.cores)==1:predictor=None
    return [func(w,d,p,predictor=predictor,**kw) for w in ws]


def _selection_job(job):
    mode,name,t_big,flag=job;d=_DESIGNS[mode][name];p=Parameters(onchip_mode=mode)
    seq=[];states=[]
    for repeat in range(2):
        pred=Predictor('ours') if len(d.cores)>1 else None
        rs=_sequence(_WS['development'],d,p,'fixed',t_big,flag,pred)
        seq.append(rs);states.append(None if pred is None else vars(pred)|{'rng':repr(pred.rng.getstate())})
    assert canonical(seq[0])==canonical(seq[1]),(mode,name,t_big,flag,'dev repeat')
    # Completion/state determinism is checked without serializing RNG object.
    state_digest=[]
    for state in states:
        if state is None:state_digest.append(None)
        else:
            state=dict(state);state.pop('rng',None)
            state_digest.append(hashlib.sha256(repr(state).encode()).hexdigest())
    assert state_digest[0]==state_digest[1]
    old=[old_simulate(w,d,p) for w in _WS['development']]
    rows=[]
    for w,a,b,r in zip(_WS['development'],seq[0],seq[1],old):
        rows.append({'design':name,'onchip_mode':mode,'t_big':t_big,'large_first':flag,
            'window_id':w['id'],'batch':w['batch'],'fixed_cycles':a['cycles'],
            'old_cycles':r['cycles'],'ratio':a['cycles']/r['cycles'],
            'result_digest':digest(a),'repeat_digest':digest(b)})
    return rows


def _heldout_job(job):
    mode,name,dispatch,t_big,flag=job;d=_DESIGNS[mode][name];p=Parameters(onchip_mode=mode)
    sequences=[];warm_hashes=[]
    for repeat in range(2):
        pred=Predictor('ours') if len(d.cores)>1 else None
        warm=_sequence(_WS['development'],d,p,dispatch,t_big,flag,pred)
        warm_hashes.append(digest(warm))
        sequences.append(_sequence(_WS['heldout'],d,p,dispatch,t_big,flag,pred))
    assert canonical(sequences[0])==canonical(sequences[1]),(mode,name,dispatch,'heldout repeat')
    assert warm_hashes[0]==warm_hashes[1]
    out=[]
    for w,a,b in zip(_WS['heldout'],sequences[0],sequences[1]):
        r=summarize(w,d,p,a,dispatch,digest(b));r.update(design=name,onchip_mode=mode,
            predictor='ours' if len(d.cores)>1 else 'none_single_compatibility',
            warmup_digest=warm_hashes[0])
        out.append(r)
    return out


def select(development_rows,window_ids):
    grouped={cfg:{wid:[] for wid in window_ids} for cfg in CONFIGS}
    for r in development_rows:grouped[r['t_big'],r['large_first']][r['window_id']].append(math.log(r['ratio']))
    per_window={cfg:[sum(by_w[wid])/len(by_w[wid]) for wid in window_ids]
                for cfg,by_w in grouped.items()}
    scores={cfg:math.exp(sum(logs)/len(logs)) for cfg,logs in per_window.items()}
    best=min(CONFIGS,key=lambda cfg:(scores[cfg],cfg[1],cfg[0]))
    rng=np.random.default_rng(20261008);counts={cfg:0 for cfg in CONFIGS}
    for _ in range(200):
        indices=rng.integers(0,len(window_ids),size=len(window_ids))
        cfg=min(CONFIGS,key=lambda c:(sum(per_window[c][j] for j in indices)/len(indices),c[1],c[0]))
        counts[cfg]+=1
    return {'chosen':{'t_big':best[0],'large_first':best[1]},'bootstrap_draws':200,
        'bootstrap_seed':20261008,'development_windows':window_ids,
        'objective':'paired log-ratio GM fixed/eft_old; average across all five designs and both modes per window',
        'selected_development_ratio':scores[best],
        'protocol':'fresh ours; process canonical 18 development windows once per candidate; reset and repeat entire sequence; final heldout predictor warmed on 18 development windows',
        'threshold_inactive_when_large_first_false':True,
        'scores':[{'t_big':cfg[0],'large_first':cfg[1],'ratio':scores[cfg]} for cfg in CONFIGS],
        'bootstrap_selection_counts':[{'t_big':cfg[0],'large_first':cfg[1],'count':counts[cfg],
             'fraction':counts[cfg]/200} for cfg in CONFIGS],
        'per_window_log_ratios':[{'t_big':cfg[0],'large_first':cfg[1],'logs':per_window[cfg]}
                                for cfg in CONFIGS]}


def _check_references(rows):
    refs={(r['entry'],r['onchip_mode'],r['sched_type'],r['window_id']):r
          for r in read_csv(ROOT/'results/E4/per_window.csv')}
    checks=[]
    for r in rows:
        kind='runtime' if r['dispatch']=='eft_old' else 'milp'
        ref=refs[r['design'],r['onchip_mode'],kind,r['window_id']]
        checks.append({'design':r['design'],'onchip_mode':r['onchip_mode'],
            'dispatch':r['dispatch'],'window_id':r['window_id'],
            'cycles_exact':r['cycles']==float(ref['cycles']),
            'hbm_bytes_exact':r['hbm_bytes']==int(ref['hbm_bytes'])})
    assert all(c['cycles_exact'] and c['hbm_bytes_exact'] for c in checks),'E4 reproduction differs'
    return checks


def freeze():
    files=[ROOT/n for n in ('model.py','predictors.py','optimizer.py','common.py','run.py')]
    files += [ROOT/'results/E0/frozen_inputs.json',ROOT/'results/E3/FROZEN_SELECTION.json',
              ROOT/'results/E4/per_window.csv',ROOT/'results/E4/heldout_main_table.csv']
    repo=ROOT.parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    manifest={'base_commit':commit,'source_hashes':{str(f.relative_to(ROOT)):sha(f) for f in files},
        'modes':{m:{n:encode_design(d) for n,d in _DESIGNS[m].items()} for m in MODES},
        'mode_B2_choice':'user selected per-mode E4 frozen hardware; port_tight B2=3x16x128 pair',
        'parameters':{m:asdict(Parameters(onchip_mode=m)) for m in MODES},
        'window_counts':{split:len(_WS[split]) for split in ('development','heldout')},
        'scope':'post-router BF16 MoE phase-fluid estimate; no hardware/cost/input change',
        'single_compatibility':'direct delegation to original E4 EFT without learned predictor',
        'predictor_control':'E4 old baseline has no predictor; separate EFT+ours rows isolate predictor effect'}
    write_json(DIRECTORY/'frozen_designs.json',manifest)
    return manifest


def verify_freeze(manifest):
    checks={name:sha(ROOT/name)==hashval for name,hashval in manifest['source_hashes'].items()}
    assert all(checks.values()),'frozen source/result hash changed'
    return checks


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--jobs',type=int,default=6)
    ap.add_argument('--stage',choices=('all','baseline','selection','heldout'),default='all')
    args=ap.parse_args();_init_worker();manifest=freeze()
    stage=args.stage
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=_init_worker) as pool:
        if stage in ('all','baseline'):
            rows=[]
            for result in pool.map(_baseline_job,[(m,n) for m in MODES for n in NAMES]):
                rows.extend(result);print('baseline',len(rows),'rows',flush=True)
            write_json(DIRECTORY/'baseline_per_window.json',rows)
            write_csv(DIRECTORY/'e4_reproduction.csv',_check_references(rows))
        if stage in ('all','selection'):
            dev=[]
            jobs=[(m,n,t,flag) for t,flag in CONFIGS for m in MODES for n in NAMES]
            for j,result in enumerate(pool.map(_selection_job,jobs)):
                dev.extend(result)
                if (j+1)%10==0:print('development',j+1,'/',len(jobs),flush=True)
            write_csv(DIRECTORY/'development_per_window.csv',dev)
            chosen=select(dev,[w['id'] for w in _WS['development']])
            write_json(DIRECTORY/'selection.json',chosen)
            print('selected',chosen['chosen'],'ratio',chosen['selected_development_ratio'],flush=True)
        if stage in ('all','heldout'):
            chosen=json.loads((DIRECTORY/'selection.json').read_text())
            selection_sha=sha(DIRECTORY/'selection.json');cfg=chosen['chosen']
            rows=json.loads((DIRECTORY/'baseline_per_window.json').read_text())
            jobs=[(m,n,kind,cfg['t_big'],cfg['large_first']) for m in MODES
                  for n in NAMES for kind in ('fixed','eft_ours_control')]
            for j,result in enumerate(pool.map(_heldout_job,jobs)):
                rows.extend(result);print('heldout',j+1,'/',len(jobs),flush=True)
            assert sha(DIRECTORY/'selection.json')==selection_sha,'selection changed during heldout'
            write_json(DIRECTORY/'per_window.json',rows)
            old={(r['design'],r['onchip_mode'],r['window_id']):r for r in rows if r['dispatch']=='eft_old'}
            regression=[]
            for r in rows:
                if r['design'] not in ('B0','B1') or r['dispatch']!='fixed':continue
                ref=old[r['design'],r['onchip_mode'],r['window_id']]
                regression.append({'design':r['design'],'onchip_mode':r['onchip_mode'],'window_id':r['window_id'],
                    'cycles_exact':r['cycles']==ref['cycles'],'full_digest_exact':r['result_digest']==ref['result_digest']})
            assert all(r['cycles_exact'] and r['full_digest_exact'] for r in regression),'single regression'
            write_json(DIRECTORY/'repeat_checks.json',{'all_evaluated_configs_repeated':True,
                'all_per_window_digests_match':all(r['result_digest']==r['repeat_digest'] for r in rows),
                'heldout_rows':len(rows),'development_rows':len(read_csv(DIRECTORY/'development_per_window.csv')),
                'single_regressions':len(regression),'all_single_bit_exact':True,
                'frozen_sha_checks':verify_freeze(manifest),'selection_sha256':selection_sha,
                'predictor_sequences':'independent fresh state per configuration/repeat; serial canonical window order'})
    print('done',stage,flush=True)


if __name__=='__main__':main()
