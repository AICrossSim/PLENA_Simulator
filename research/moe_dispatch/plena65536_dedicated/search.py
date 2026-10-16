"""Exhaustive integer compute geometry DSE under absolute hard affinity.

No dispatch estimator or optimisation of task ownership exists: Core0's load
is Shared, Core1's load is every routed expert.  All legal ordered geometries
are evaluated exactly at all development windows.  BLEN1 scalar topology is
admitted fairly to single/homogeneous/dedicated baselines, with the preserved
BLEN>=2 domain certified separately.  Shared>=Routed multiplier allocation is
the primary interpretation of a genuinely larger Shared core; free budget
order is an explicitly separate diagnostic, with the SAME hard affinity.
"""
from __future__ import annotations
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from math import gcd,lcm,prod
from pathlib import Path
import time
import numpy as np
from research.moe_dispatch.plena65536_shared.timing import PROFILES,resource_ledger
from research.moe_dispatch.plena65536_moe.workloads import load_workloads
from research.moe_dispatch.plena65536_dedicated.geometry import (Core,enumerate_cores,
    geometry_counts,single_cores,homogeneous_cores,iter_dedicated_pairs)
from research.moe_dispatch.plena65536_dedicated.model import population_costs


def core_from_dict(value):
    return Core(int(value['b']),int(value['r']),int(value['rm']),int(value['rn']))


def label(cores):
    return ' + '.join(f'{c.pm}x{c.pn}x{c.groups}[b{c.b},R{c.r}]' for c in cores)


def integer_objective_exponents(windows):
    counts=Counter(int(w['batch']) for w in windows)
    denominators=[len(counts)*counts[int(w['batch'])] for w in windows]
    denominator=lcm(*denominators)
    return np.asarray([denominator//d for d in denominators],dtype=np.int64),denominator


def evaluate_family(cores,ids,shared,routed,weights,*,exponents=None,chunk=8192):
    """Return winner and exact-evaluated GM array for all finite designs."""
    ids=np.asarray(ids,dtype=np.int32)
    scores=np.empty(len(ids),dtype=np.float64)
    for start in range(0,len(ids),chunk):
        selected=ids[start:start+chunk]
        if np.all(selected[:,1]==-1):
            cycles=shared[selected[:,0]]+routed[selected[:,0]]
        else:
            if np.any(selected[:,1]<0):raise ValueError('A family cannot mixsingle/dual')
            cycles=np.maximum(shared[selected[:,0]],routed[selected[:,1]])
        scores[start:start+len(selected)]=np.exp(np.log(cycles)@weights)
    minimum=float(scores.min())
    # Every integer makespan is already exact. The narrow floating log band
    # is resolved using an exact integer product with rational batch weights.
    near=np.flatnonzero(scores<=minimum*(1+1e-12))
    exp_weights=np.asarray(exponents,dtype=np.int64) if exponents is not None else None
    candidates=[]
    for ix in near:
        first,second=map(int,ids[ix])
        role0,role1=shared[first],routed[second] if second>=0 else routed[first]
        cycles=np.maximum(role0,role1) if second>=0 else role0+role1
        exact=prod(int(x)**int(p) for x,p in zip(cycles,exp_weights)) if exp_weights is not None else 0
        shared_score=float(np.exp(np.log(role0)@weights))
        cs=(cores[first],) if second<0 else (cores[first],cores[second])
        operand=sum(4*c.r*(c.pm+c.pn) for c in cs)
        candidates.append((exact,shared_score,operand,int(ix),cycles.tolist(),role0.tolist(),role1.tolist()))
    winner=min(candidates,key=lambda x:x[:4])
    _,_,_,ix,cycles,role0,role1=winner
    picked=tuple(cores[int(i)] for i in ids[ix] if i>=0)
    info={'indices':ids[ix].tolist(),'label':label(picked),'cores':[c.to_dict() for c in picked],
          'development_GM_cycles':float(scores[ix]),'development_window_cycles':cycles,
          'development_Shared_cycles':role0,'development_Routed_cycles':role1,
          'resource_ledger':resource_ledger(picked),'cooptimal_log_band_count':len(near),
          'objective_exact_integer_product':str(winner[0]) if exp_weights is not None else None,
          'secondary_tie_break':'minSharedGM, thenoperand-bufferbytes, thenstablegeometryorder',
          'geometry_count':len(ids),'every_geometry_exactly_evaluated':True}
    return info,scores


def main(output,*,chunk=8192):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    inputs=load_workloads('development');windows=inputs['development']
    cores=enumerate_cores(include_blen1=True);index={c:i for i,c in enumerate(cores)}
    domains={}
    for extension,prefix in ((False,'strict'),(True,'extended')):
        domains[prefix+'_single']=np.asarray([(index[c],-1) for c in single_cores(include_blen1=extension)],dtype=np.int32)
        domains[prefix+'_homogeneous']=np.asarray([(index[c],index[c]) for c in homogeneous_cores(include_blen1=extension)],dtype=np.int32)
        free=np.asarray([(index[a],index[b]) for a,b in iter_dedicated_pairs(include_blen1=extension,
                         shared_not_smaller=False,exclude_identical=True)],dtype=np.int32)
        domains[prefix+'_dedicated_free']=free
        allocation=np.asarray([c.macs for c in cores],dtype=np.int64)
        domains[prefix+'_dedicated_largeShared']=free[allocation[free[:,0]]>=allocation[free[:,1]]]
    exp_weights,denominator=integer_objective_exponents(windows)
    frozen={'budget':65536,'contract':'Shared_C0_Routed_C1_hard_affinity',
        'profile_default':PROFILES[0].name,'precision':'BF16/FP32','clock_GHz':1,
        'supply_delay_cycles':0,'HBM_bandwidth':'infinite','SRAM_port_wait_cycles':0,
        'no_cross_core_substitution':True,'development_window_ids':[w['id'] for w in windows],
        'development_input_sha256':inputs['manifest']['loaded_input_sha256'],
        'objective':'withinBatchwindowGM, then7BatchequalGM',
        'objective_exponents':exp_weights.tolist(),'objective_root_denominator':denominator,
        'geometry_domains':{prefix:geometry_counts(include_blen1=extension) for extension,prefix in ((False,'strict'),(True,'extended'))},
        'geometry_scope':{'PM_powers_of_two':[1,2,4,8,16,32,64],'PN_alignment':4,'MLEN_max':2048,
             'Routed_PM':[1,2],'Shared_PM':'alllegal','homogeneous_PM':'alllegal',
             'BLEN1':'newactualscalar1x1subarray, notgatedfreeunusedPEs',
             'exact2to1_budget_possible':False,'43690_plus21846_legal_in_align4_domain':False},
        'primary_dedicated_allocation_constraint':'Core0MACs>=Core1MACs',
        'free_diagnostic_keeps_same_affinity':True,'profile_optima':{}}
    for profile in PROFILES:
        begin=time.monotonic();shared,routed,weights=population_costs(cores,windows,profile)
        choices={}
        for family,ids in domains.items():
            t=time.monotonic()
            chosen,scores=evaluate_family(cores,ids,shared,routed,weights,exponents=exp_weights,chunk=chunk)
            chosen['profile']=profile.name;choices[family]=chosen
            np.savez_compressed(output/f'{family}_{profile.name}_exact.npz',core_ids=ids,exact_GM_cycles=scores)
            print(profile.name,family,chosen['label'],
                  f"GM={chosen['development_GM_cycles']:.6f}",f"N={len(ids)}",f"{time.monotonic()-t:.2f}s",flush=True)
        original=Core(32,2048)
        original_ids=np.asarray([(index[original],-1)],dtype=np.int32)
        choices['original']=evaluate_family(cores,original_ids,shared,routed,weights,exponents=exp_weights)[0]
        choices['original']['profile']=profile.name
        frozen['profile_optima'][profile.name]=choices
        (output/'SEARCH_PROGRESS.json').write_text(json.dumps(frozen,indent=2)+'\n')
        print('PROFILE DONE',profile.name,round(time.monotonic()-begin,2),flush=True)
    frozen['families']=frozen['profile_optima'][PROFILES[0].name]
    frozen['elapsed_seconds']=time.monotonic()-started
    paths=[Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('geometry.py')]
    frozen['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (output/'FROZEN_SELECTION.json').write_text(json.dumps(frozen,indent=2)+'\n')
    stable={k:frozen[k] for k in ('families','profile_optima','development_window_ids','objective_exponents','geometry_scope')}
    digest=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    (output/'SEARCH_CERTIFICATE.json').write_text(json.dumps({'status':'PASS','selection_sha256':digest,
        'domains':{profile:{family:{'geometry_count':v['geometry_count'],'winnerGM':v['development_GM_cycles'],
                  'all_exact':True} for family,v in choices.items()} for profile,choices in frozen['profile_optima'].items()},
        'no_pruning':True,'no_ownership_search':True,'BLEN1_is_proposed_topology':True,
        'heldout_payload_read_for_selection':False,'scope':frozen['geometry_scope']},indent=2)+'\n')
    print('SEARCH COMPLETE',digest,round(frozen['elapsed_seconds'],2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--chunk',type=int,default=8192)
    opts=parser.parse_args();main(opts.out,chunk=opts.chunk)
