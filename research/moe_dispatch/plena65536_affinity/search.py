"""Complete finite-domain development-only geometry/affinity grid search.

A evaluates every ordered geometry and all four fixed thresholds exactly.
B evaluates a declared planning portfolio using costs of the CURRENT known
router descriptors: whole-expert greedy EFT or one of four hard boundaries.
An admissible assignment lower bound only skips configurations which cannot
beat an actually scheduled incumbent. This certifies the finite grid and
specified algorithms, not arbitrary dispatch algorithms or calibrated RTL.
"""
from __future__ import annotations
import argparse
from collections import Counter
from math import prod
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from research.moe_dispatch.plena65536_dedicated.geometry import Core,enumerate_cores,single_cores,homogeneous_cores
from research.moe_dispatch.plena65536_dedicated.search import integer_objective_exponents
from research.moe_dispatch.plena65536_moe.model import window_histograms,cost_table,fractional_pair_bound,expert_key
from research.moe_dispatch.plena65536_moe.workloads import load_workloads
from research.moe_dispatch.plena65536_shared.timing import PROFILES,resource_ledger

THRESHOLDS=(1,2,4,8)
DEFAULT_OUTPUT=Path('/scratch/shared/mcl123/plena/outputs/plena65536_affinity_20261013')

def core_from_dict(d):return Core(*(int(d[k]) for k in ('b','r','rm','rn')))
def label(cs):return ' + '.join(f'{c.pm}x{c.pn}x{c.groups}[b{c.b},R{c.r}]' for c in cs)

def ordered_pair_ids(cores):
    by={}
    for i,c in enumerate(cores):by.setdefault(c.macs,[]).append(i)
    chunks=[]
    for n,a in sorted(by.items()):
        b=by.get(65536-n,[])
        if not b:continue
        left=np.repeat(np.asarray(a,dtype=np.int32),len(b))
        right=np.tile(np.asarray(b,dtype=np.int32),len(a))
        keep=left!=right
        chunks.append(np.column_stack((left[keep],right[keep])))
    return np.concatenate(chunks)

def guarded_fractional_pair_bound(costs,ids,hist):
    """Admissible dual with a tighter *per-pair cross-product* overflow proof.

    Numerators use Amax*Bmax*job_count, never Amax squared. This matters for
    legal extremely unequal core budgets. Arbitrary dual multipliers remain
    valid even if floating-point sorting changes their selection.
    """
    a=costs[ids[:,0]];b=costs[ids[:,1]]
    jobs=max(sum(map(int,h)) for h in hist)
    am=a.max(axis=1);bm=b.max(axis=1);limit=np.iinfo(np.int64).max
    if np.any(am>limit//bm//max(1,jobs)) or np.any(am+bm>2**53):
        raise OverflowError('Per-pair LP cross-product proof failed')
    order=np.argsort(a/b,axis=1,kind='stable')
    a=np.take_along_axis(a,order,axis=1);b=np.take_along_axis(b,order,axis=1)
    answer=np.zeros((len(ids),len(hist)),dtype=np.float64)
    row=np.arange(len(ids))
    for wi,h in enumerate(hist):
        n=h[order]
        ca=np.cumsum(a*n,axis=1);cb=np.sum(b*n,axis=1)[:,None]-np.cumsum(b*n,axis=1)
        ix=np.argmax(ca>=cb,axis=1);av=a[row,ix];bv=b[row,ix]
        numerator=np.sum(np.minimum(bv[:,None]*a,av[:,None]*b)*n,axis=1,dtype=np.int64)
        quotient=numerator.astype(float)/(av+bv)
        answer[:,wi]=np.maximum(0,np.nextafter(quotient*(1-8*np.finfo(float).eps),-np.inf))
    return answer

def threshold_populations(cores,types,hist,costs):
    # Expert types distinguish Shared F=2816 from Routed F=1408.
    dense=[];stream=[]
    for t in THRESHOLDS:
        mask=np.asarray([f==2816 or m>t for m,h,f in types],dtype=np.int64)
        dense.append(costs@(hist*mask).T)
        stream.append(costs@(hist*(1-mask)).T)
    return np.stack(dense),np.stack(stream)

def greedy_vectors(ids,costs,windows,types,threshold):
    """Independent vector replay of deterministic whole-expert greedy EFT."""
    type_id={key:i for i,key in enumerate(types)}
    result=np.zeros((len(ids),len(windows)),dtype=np.int64)
    for wi,w in enumerate(windows):
        loads0=np.zeros(len(ids),dtype=np.int64);loads1=loads0.copy()
        tasks=sorted(w['experts'],key=lambda e:(-3*int(e['Me'])*int(e['H'])*int(e['F']),int(e['id'])))
        for e in tasks:
            ti=type_id[expert_key(e)]
            a=costs[ids[:,0],ti];b=costs[ids[:,1],ti]
            finish0=loads0+a;finish1=loads1+b
            preferred0=bool(e.get('is_shared',False)) or int(e['Me'])>threshold
            take0=(finish0<finish1)|((finish0==finish1)&preferred0)
            loads0=np.where(take0,finish0,loads0)
            loads1=np.where(take0,loads1,finish1)
        result[:,wi]=np.maximum(loads0,loads1)
    return result

def hard_vectors(ids,dense,stream):
    return np.maximum(dense[:,ids[:,0],:],stream[:,ids[:,1],:])

def score(cycles,weights):return np.exp(np.log(cycles)@weights)

def exact_key(cycles,exponents):return prod(int(c)**int(e) for c,e in zip(cycles,exponents))

def tie_key(cores,ids,threshold):
    cs=[cores[int(i)] for i in ids if i>=0]
    return (sum(4*c.r*(c.pm+c.pn) for c in cs),tuple(int(i) for i in ids),int(threshold))

def pick(candidates,cores,weights,exponents):
    # candidates=(cycles, pair_ids, threshold, policy, source_index)
    scores=[float(score(np.asarray(c[0]),weights)) for c in candidates]
    low=min(scores)
    near=[i for i,v in enumerate(scores) if v<=low*(1+1e-12)]
    i=min(near,key=lambda i:(exact_key(candidates[i][0],exponents),tie_key(cores,candidates[i][1],candidates[i][2])))
    cycles,ids,t,policy,source=candidates[i]
    cs=tuple(cores[int(i)] for i in ids if i>=0)
    return {'cores':[c.to_dict() for c in cs],'label':label(cs),'policy':policy,'threshold':int(t),
        'indices':list(map(int,ids)),'source_index':int(source),'development_window_cycles':list(map(int,cycles)),
        'development_GM_cycles':scores[i],'objective_exact_integer_product':str(exact_key(cycles,exponents)),
        'resource_ledger':resource_ledger(cs),'profile':PROFILES[0].name}

def top_candidates(scores,cycles_for,ids,policy):
    low=float(np.min(scores));locations=np.argwhere(scores<=low*(1+1e-12))
    if scores.ndim==1:
        return [(cycles_for(int(i[0]),0),ids[int(i[0])],2,policy,int(i[0])) for i in locations]
    return [(cycles_for(int(i),int(t)),ids[int(i)],THRESHOLDS[int(t)],policy,int(i)) for i,t in locations]

def main(out=DEFAULT_OUTPUT,*,chunk=4096):
    started=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    inputs=load_workloads('development');windows=inputs['development']
    types,hist,weights=window_histograms(windows);exponents,root=integer_objective_exponents(windows)
    cores=enumerate_cores();index={c:i for i,c in enumerate(cores)}
    pairs=ordered_pair_ids(cores)
    singles=np.asarray([(index[c],-1) for c in single_cores()],dtype=np.int32)
    hom=np.asarray([(index[c],index[c]) for c in homogeneous_cores()],dtype=np.int32)
    costs=cost_table(cores,types,PROFILES[0]);dense,stream=threshold_populations(cores,types,hist,costs)
    print('DOMAIN',len(cores),'cores',len(singles),'single',len(hom),'hom',len(pairs),'orderedhetero',flush=True)
    families={};diags={}
    sc=costs[singles[:,0]]@hist.T;ss=score(sc,weights)
    families['single']=pick(top_candidates(ss,lambda i,t:sc[i],singles,'greedy_eft'),cores,weights,exponents)
    oi=index[Core(32,2048)];oc=costs[oi]@hist.T
    families['original']=pick([(oc,(oi,-1),2,'greedy_eft',0)],cores,weights,exponents)
    np.savez_compressed(out/'single_scores.npz',pair_ids=singles,GM_cycles=ss,window_cycles=sc)
    hg=greedy_vectors(hom,costs,windows,types,2);hs=score(hg,weights)
    families['homogeneous']=pick(top_candidates(hs,lambda i,t:hg[i],hom,'greedy_eft'),cores,weights,exponents)
    ha=hard_vectors(hom,dense,stream).min(axis=0)
    hb=np.stack([np.minimum(greedy_vectors(hom,costs,windows,types,t),ha) for t in THRESHOLDS],axis=1)
    hbs=score(hb,weights)
    families['homogeneous_B']=pick(top_candidates(hbs,lambda i,t:hb[i,t],hom,'workload_aware'),cores,weights,exponents)
    np.savez_compressed(out/'homogeneous_scores.npz',pair_ids=hom,greedy_GM=hs,portfolio_GM=hbs)
    # All A candidates and current-window best-threshold diagnostic.
    fixed_scores=np.empty((len(pairs),4),dtype=np.float64)
    adaptive_scores=np.empty(len(pairs),dtype=np.float64)
    lp_scores=np.empty(len(pairs),dtype=np.float64)
    last=time.monotonic()
    for start in range(0,len(pairs),chunk):
        ids=pairs[start:start+chunk];h=hard_vectors(ids,dense,stream)
        fixed_scores[start:start+len(ids)]=score(h,weights).T
        adaptive_scores[start:start+len(ids)]=score(h.min(axis=0),weights)
        lb=guarded_fractional_pair_bound(costs,ids,hist)
        lp_scores[start:start+len(ids)]=score(lb,weights)*(1-16*np.finfo(float).eps)
        if time.monotonic()-last>15:
            print('GRID',start+len(ids),'/',len(pairs),'elapsed',round(time.monotonic()-started,1),flush=True);last=time.monotonic()
    families['heterogeneous_A']=pick(top_candidates(fixed_scores,
        lambda i,t:np.maximum(dense[t,pairs[i,0]],stream[t,pairs[i,1]]),pairs,'hard_threshold'),cores,weights,exponents)
    diags['heterogeneous_adaptive_threshold']=pick(top_candidates(adaptive_scores,
        lambda i,t:hard_vectors(pairs[i:i+1],dense,stream).min(axis=0)[0],pairs,'adaptive_threshold'),cores,weights,exponents)
    # Fair resource-aware greedy and declared portfolio share a valid LP bound.
    order=np.argsort(lp_scores,kind='stable');evaluated=np.zeros(len(pairs),dtype=bool)
    greedy_scores=np.full((len(pairs),4),np.nan,dtype=np.float64)
    portfolio_scores=np.full((len(pairs),4),np.nan,dtype=np.float64)
    best_g=best_b=float('inf');g_candidates=[];b_candidates=[]
    for block in range(0,len(order),128):
        group=order[block:block+128]
        group=group[lp_scores[group]<=max(best_g,best_b)+1e-7]
        if not len(group):break
        ids=pairs[group];hardbest=hard_vectors(ids,dense,stream).min(axis=0)
        for ti,t in enumerate(THRESHOLDS):
            g=greedy_vectors(ids,costs,windows,types,t);b=np.minimum(g,hardbest)
            gs=score(g,weights);bs=score(b,weights)
            greedy_scores[group,ti]=gs;portfolio_scores[group,ti]=bs
            new_g=min(best_g,float(gs.min()));new_b=min(best_b,float(bs.min()))
            if new_g<best_g*(1-1e-12):g_candidates=[]
            if new_b<best_b*(1-1e-12):b_candidates=[]
            best_g=new_g;best_b=new_b
            for j in np.flatnonzero(gs<=best_g*(1+1e-12)):
                g_candidates.append((g[j],ids[j],t,'greedy_eft',int(group[j])))
            for j in np.flatnonzero(bs<=best_b*(1+1e-12)):
                b_candidates.append((b[j],ids[j],t,'workload_aware',int(group[j])))
        evaluated[group]=True
    families['heterogeneous_B']=pick(b_candidates,cores,weights,exponents)
    diags['heterogeneous_greedy']=pick(g_candidates,cores,weights,exponents)
    print('WINNERS',{name:(v['label'],v['threshold'],v['development_GM_cycles']) for name,v in families.items()},flush=True)
    print('B EXACT',int(evaluated.sum()),'PRUNED',int((~evaluated).sum()),flush=True)
    # Freeze cross-policy parameters ONLY using development data.
    for name,v in list(families.items())+list(diags.items()):
        ids=np.asarray([v['indices']],dtype=np.int32)
        if ids[0,1]<0:continue
        h=hard_vectors(ids,dense,stream)[:,0,:];hscore=score(h,weights)
        v['best_A_threshold_for_geometry']=int(THRESHOLDS[min(range(4),key=lambda j:(exact_key(h[j],exponents),THRESHOLDS[j]))])
        b=np.stack([np.minimum(greedy_vectors(ids,costs,windows,types,t)[0],h.min(axis=0)) for t in THRESHOLDS])
        v['best_B_threshold_for_geometry']=int(THRESHOLDS[min(range(4),key=lambda j:(exact_key(b[j],exponents),THRESHOLDS[j]))])
        v['development_batch_thresholds']={}
        for batch in sorted({w['batch'] for w in windows}):
            mask=np.asarray([w['batch']==batch for w in windows])
            ix=min(range(4),key=lambda j:(prod(int(v) for v in h[j,mask]),THRESHOLDS[j]))
            v['development_batch_thresholds'][str(batch)]=int(THRESHOLDS[ix])
    np.savez_compressed(out/'heterogeneous_search.npz',pair_ids=pairs,fixed_GM_cycles=fixed_scores,
        adaptive_threshold_GM_cycles=adaptive_scores,LP_lower_GM_cycles=lp_scores,
        greedy_GM_cycles=greedy_scores,portfolio_GM_cycles=portfolio_scores,portfolio_evaluated=evaluated)
    counts={'individual_cores':len(cores),'single':len(singles),'homogeneous':len(hom),
        'ordered_heterogeneous':len(pairs),'fixed_threshold_geometry_parameter_points':len(pairs)*4,
        'B_geometry_parameter_points_in_domain':len(pairs)*4,'B_exact_geometry_count':int(evaluated.sum()),
        'B_pruned_geometry_count':int((~evaluated).sum())}
    minimum_pruned=float(lp_scores[~evaluated].min()) if (~evaluated).any() else None
    assert minimum_pruned is None or minimum_pruned>max(families['heterogeneous_B']['development_GM_cycles'],diags['heterogeneous_greedy']['development_GM_cycles'])+1e-7
    frozen={'schema':'plena65536_affinity_v1','families':families,'diagnostics':diags,'budget':65536,
        'precision':'BF16/FP32; declared precision/topology extension, not native main-paper MX RTL',
        'clock_GHz':1,'profile_default':PROFILES[0].name,'dataflow':'OS','supply_latency_cycles':0,
        'thresholds':list(THRESHOLDS),'scope':'Shared+allactiveRouted; whole-expert tasks; no migration or task splitting',
        'policy_A':'Shared or Me>T->C0; else C1; fixed T selected on development',
        'policy_B':'minimum of current-known-shape greedy EFT and four hard-threshold schedules; greedy first on equal makespan; zero planner cost',
        'greedy_order':'descending usefulMACs, then expertID; smallest predicted finish; equal finish uses threshold affinity',
        'policy_B_is_hardware_implemented':False,'uses_actual_future_service_times':False,
        'geometry_scope':{'PM':[1,2,4,8,16,32,64],'PN_alignment':4,'MLEN':[1,2,4,8,16,32,64,128,256,512,1024,2048],
            'G':'MLEN/BLEN','both_roles_all_legal_PM':True,'all_legal_budget_splits':True,
            'Dense_is_role_not_requirement_of_more_MACs':True,'BLEN1':'explicit scalar-subarray extension'},
        'development_window_ids':[w['id'] for w in windows],'development_input_sha256':inputs['manifest']['loaded_input_sha256'],
        'objective':'windowGM within each batch, equalGM across7batches','objective_exponents':exponents.tolist(),
        'objective_root_denominator':root,'counts':counts,'elapsed_seconds':time.monotonic()-started,
        'minimum_pruned_lower_bound_GM_cycles':minimum_pruned}
    frozen['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('model.py'))}
    (out/'FROZEN_SELECTION.json').write_text(json.dumps(frozen,indent=2)+'\n')
    stable={k:frozen[k] for k in ('families','diagnostics','development_window_ids','objective_exponents','geometry_scope')}
    digest=hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
    (out/'SEARCH_CERTIFICATE.json').write_text(json.dumps({'status':'PASS','selection_sha256':digest,'counts':counts,
        'A_every_grid_point_evaluated':True,'B_all_grid_points_enumerated':True,'B_pruning_admissible_fractional_assignment_bound':True,
        'minimum_pruned_lower_bound_GM_cycles':minimum_pruned,'B_winner_GM_cycles':families['heterogeneous_B']['development_GM_cycles'],
        'heldout_read_during_selection':False,'certifies_finite_geometry_and_declared_algorithm_grid_not_all_schedulers':True},indent=2)+'\n')
    print('SEARCH COMPLETE',digest,round(frozen['elapsed_seconds'],2),flush=True)
    return frozen

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=DEFAULT_OUTPUT);p.add_argument('--chunk',type=int,default=4096)
    a=p.parse_args();main(a.out,chunk=a.chunk)
