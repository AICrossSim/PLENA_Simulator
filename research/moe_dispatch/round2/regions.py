"""Calibrated synthetic workload regions and full Saltelli parameter scope.

Search certificates travel with every grid/sample. A time-limited point is
NOT labeled a globally optimized point; downstream indices report that limit.
"""
from __future__ import annotations
import argparse,itertools,json,math,time
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize_scalar
from .common import *
from .optimizer import search_workloads


def synthetic(batch,alpha,shared_units,E,topk,F,seed=20261007):
    rng=np.random.default_rng(seed)
    prob=np.maximum(rng.gamma(float(alpha),1,E),1e-12);prob/=prob.sum()
    # Sampling WITHOUT replacement within each token: exact top-k routing.
    scores=np.log(prob)[None,:]-np.log(-np.log(np.clip(rng.random((batch,E)),1e-12,1-1e-12)))
    picked=np.argpartition(scores,-topk,axis=1)[:,-topk:]
    es=[]
    for expert in range(E):
        ids=np.nonzero(np.any(picked==expert,axis=1))[0].tolist()
        if ids:es.append({'id':expert,'Me':len(ids),'H':2048,'F':F,'is_shared':False,'token_indices':ids})
    if shared_units:es.insert(0,{'id':-1,'Me':batch,'H':2048,'F':int(shared_units)*F,'is_shared':True,'token_indices':list(range(batch))})
    w={'id':f'synthetic_B{batch}_a{alpha:.8g}_S{shared_units}_E{E}_K{topk}_F{F}_seed{seed}',
       'batch':batch,'hidden':2048,'top_k':topk,'experts':es,
       'tokens':[{'token_index':i,'routes':[{'expert_id':int(e)} for e in picked[i]]} for i in range(batch)],
       'provenance':{'origin':'synthetic_calibrated_region_only','seed':seed}}
    assert sum(e['Me'] for e in es if not e['is_shared'])==batch*topk
    return w


def me_hist(w):
    a=np.bincount([e['Me'] for e in w['experts'] if not e['is_shared']],minlength=w['batch']+1).astype(float)
    a+=1e-8;return a/a.sum()

def calibration(dev):
    # Fit the concentrated endpoint to SWE development captures, never heldout.
    swe=[w for w in dev if 'swe' in w['id'].lower()]
    if not swe:raise ValueError('SWE development trace required')
    def loss(logalpha):
        alpha=math.exp(logalpha);values=[]
        for j,w in enumerate(swe):
            ref=me_hist(w);n=sum(not e['is_shared'] for e in w['experts']);pred=[]
            for seed in range(8):
                v=synthetic(w['batch'],alpha,2,64,6,1408,20261007+31*j+seed)
                h=me_hist(v);dist=sum(not e['is_shared'] for e in v['experts'])
                pred.append(np.sum(ref*np.log(ref/h))+(dist-n)**2/max(1,n*n))
            values.append(np.mean(pred))
        return float(np.mean(values))
    fit=minimize_scalar(loss,bounds=(math.log(.003),math.log(100)),method='bounded',options={'xatol':.015})
    lo=math.exp(fit.x);hi=100.0
    levels=np.geomspace(hi,lo,5).tolist();rows=[]
    for w in dev:
        ref=me_hist(w);routed=sum(not e['is_shared'] for e in w['experts'])
        for level,alpha in enumerate(levels):
            ss=[synthetic(w['batch'],alpha,2,64,6,1408,20261007+j) for j in range(16)]
            meanhist=np.mean([me_hist(v) for v in ss],axis=0);distinct=np.mean([sum(not e['is_shared'] for e in v['experts']) for v in ss])
            rows.append({'window_id':w['id'],'batch':w['batch'],'concentration_level':level,'alpha':alpha,
                'real_distinct':routed,'synthetic_mean_distinct':float(distinct),
                'distinct_relative_error':float((distinct-routed)/routed),'me_hist_KL_real_to_synthetic':float(np.sum(ref*np.log(ref/meanhist)))})
    return {'levels':levels,'concentrated_fit_loss':float(fit.fun),'fit_source_ids':[w['id'] for w in swe],
            'fit_samples_per_window':8,'synthetic_draws_for_validation':16,'seed':20261007},rows


def _search_job(job):
    idx,w,params,cap,initial=job
    # Full legal region coverage is tracked by search_workloads; time limit does not shrink its declared domain.
    a=search_workloads([w],params,delta=.02,time_limit_s=cap,initial_designs=initial,seed=20261007)
    a["resume_workloads"]=[w]
    return idx,a

def _family(result,name):
    x=result.get('families',result.get('best',{})).get(name)
    if not x:return None
    return x

def grid(args):
    ws=inputs();cal,rows=calibration(ws['development']);out=ROOT/'results/E3'
    write_json(out/'synthetic_calibration.json',cal);write_csv(out/'synthetic_calibration.csv',rows)
    selection=json.loads((out/'FROZEN_SELECTION.json').read_text())
    initial=[decode_design(v['design']) for v in selection['modes']['pipelined'].values() if isinstance(v,dict) and 'design' in v]
    combos=list(itertools.product((2,4,8,16,32,64,128,256),range(5),(0,1,2,4),((64,6),(128,8),(256,8)),(512,1408,2048),(126.03076923076924,252.06153846153848,504.12307692307695)))
    old=read_csv(out/'workload_map.csv') if (out/'workload_map.csv').exists() else []
    done={int(r['point_index']) for r in old};queue=[]
    for index,(batch,level,S,(E,K),F,bw) in enumerate(combos):
        if index in done:continue
        w=synthetic(batch,cal['levels'][level],S,E,K,F)
        p=Parameters(hbm_Bpc=bw,credits=math.ceil(bw*65/32))
        queue.append((index,w,p,args.point_seconds,initial))
    results=list(old)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
      for index,res in pool.map(_search_job,queue,chunksize=1):
        batch,level,S,(E,K),F,bw=combos[index]
        write_json(out/'search_certificates/grid'/f'{index:04d}.json',res)
        a=_family(res,'single');b=_family(res,'heterogeneous');c=_family(res,'homogeneous')
        row={'point_index':index,'batch':batch,'concentration':level,'alpha':cal['levels'][level],
            'shared_units':S,'E':E,'topk':K,'F':F,'bw_or_mac_scale':bw,
            'best_single_ms':a['geomean_ms'] if a else None,'best_hetero_ms':b['geomean_ms'] if b else None,
            'best_homo_ms':c['geomean_ms'] if c else None,
            'delta_vs_single_pct':100*(b['geomean_ms']/a['geomean_ms']-1) if a and b and a.get('geomean_ms') and b.get('geomean_ms') else None,
            'delta_vs_homo_pct':100*(b['geomean_ms']/c['geomean_ms']-1) if b and c and b.get('geomean_ms') and c.get('geomean_ms') else None,
            'hetero_design':canonical(b['design']) if b else None,
            'proof_complete':res.get('proof_complete',False),'open_lb_ms':res.get('open_lb_ms'),
            'gap_pct':res.get('gap_pct'),'search_seconds':res.get('elapsed_seconds'),
            'hypothetical_bw_override':True,'model_scope':'synthetic_only'}
        results.append(row)
        if len(results)%32==0:write_csv(out/'workload_map.csv',sorted(results,key=lambda r:int(r['point_index'])));print('grid',len(results),'/4320',flush=True)
    write_csv(out/'workload_map.csv',sorted(results,key=lambda r:int(r['point_index'])))
    valid=[r for r in results if r['delta_vs_single_pct'] is not None and r['delta_vs_single_pct']!='']
    best=min(valid,key=lambda r:float(r['delta_vs_single_pct']))
    write_json(out/'workload_grid_summary.json',{'planned_points':len(combos),'completed_points':len(results),
        'certified_points':sum(str(r['proof_complete']).lower()=='true' for r in results),'best_evaluated_point':best,
        'scope':'full workload grid; time-limited full-domain hardware searches; uncertified points are candidate estimates'})


def _sobol_job(job):
    index,sample,dev,cap,initial=job
    ii,bank,dot,credits,vector=sample
    p=Parameters(tile_issue_cycles=float(ii),bank_Bpc=float(bank),dotstagecycles=float(dot),credits=int(round(credits)),vector_scale=float(vector))
    res=search_workloads(dev,p,delta=.02,time_limit_s=cap,initial_designs=initial,seed=20261007)
    res['resume_workloads']=dev
    a=_family(res,'single');b=_family(res,'heterogeneous')
    write_json(ROOT/'results/E3/search_certificates/sobol'/f'{index:04d}.json',res)
    return {'sample_index':index,'tile_issue_cycles':ii,'bank_Bpc':bank,'dotstagecycles':dot,'credits':int(round(credits)),
        'vector_scale':vector,'delta':b['geomean_ms']/a['geomean_ms']-1,
        'single_ms':a['geomean_ms'],'hetero_ms':b['geomean_ms'],
        'single_design':canonical(a['design']),'hetero_design':canonical(b['design']),
        'proof_complete':res.get('proof_complete',False),'gap_pct':res.get('gap_pct'),
        'open_lb_ms':res.get('open_lb_ms')}


def sobol(args):
    from SALib.sample import sobol as sampler
    from SALib.analyze import sobol as analyzer
    out=ROOT/'results/E3';ws=inputs();selection=json.loads((out/'FROZEN_SELECTION.json').read_text())
    initial=[decode_design(v['design']) for v in selection['modes']['pipelined'].values() if isinstance(v,dict) and 'design' in v]
    problem={'num_vars':5,'names':['tile_issue_cycles','bank_Bpc','dotstagecycles','credits','vector_scale'],
        'bounds':[[1,30.4],[8,32],[1,4],[256,512],[.5,2]]}
    samples=sampler.sample(problem,256,calc_second_order=False,seed=20261007)
    assert len(samples)==1792
    old=read_csv(out/'sobol_samples.csv') if (out/'sobol_samples.csv').exists() else [];done={int(r['sample_index']) for r in old}
    rows=list(old);jobs=[(i,s.tolist(),ws['development'],args.point_seconds,initial) for i,s in enumerate(samples) if i not in done]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
      for row in pool.map(_sobol_job,jobs,chunksize=1):
        rows.append(row)
        if len(rows)%16==0:write_csv(out/'sobol_samples.csv',sorted(rows,key=lambda r:int(r['sample_index'])));print('sobol',len(rows),'/1792',flush=True)
    rows=sorted(rows,key=lambda r:int(r['sample_index']));write_csv(out/'sobol_samples.csv',rows)
    values=np.array([float(r['delta']) for r in rows]);si=analyzer.analyze(problem,values,calc_second_order=False,seed=20261007)
    write_csv(out/'sobol.csv',[{'param':name,'S1':si['S1'][j],'S1_ci':si['S1_conf'][j],
        'ST':si['ST'][j],'ST_ci':si['ST_conf'][j],
        'all_searches_certified':all(str(r['proof_complete']).lower()=='true' for r in rows),
        'scope':'indices of time-limited best-evaluated designs if certificates remain open'} for j,name in enumerate(problem['names'])])
    write_json(out/'sobol_protocol.json',{'problem':problem,'baseN':256,'samples':1792,'second_order':False,
        'hardware_reoptimized_per_sample':True,'effective_credits_rounded_to_integer':True,
        'certified_samples':sum(str(r['proof_complete']).lower()=='true' for r in rows)})


def flip(args):
    out=ROOT/'results/E3';dev=inputs()['development'];selection=json.loads((out/'FROZEN_SELECTION.json').read_text())
    initial=[decode_design(v['design']) for v in selection['modes']['pipelined'].values() if isinstance(v,dict) and 'design' in v]
    ranges={'tile_issue_cycles':np.linspace(1,30.4,21),'bank_Bpc':np.linspace(8,32,21),
            'dotstagecycles':np.linspace(1,4,21),'credits':np.linspace(256,512,21),'vector_scale':np.linspace(.5,2,21)}
    jobs=[];metadata=[]
    for key,values in ranges.items():
      for value in values:
        p=replace(Parameters(),**{key:int(round(value)) if key=='credits' else float(value)})
        i=len(jobs);jobs.append((i,dev,p,args.point_seconds,initial));metadata.append((key,float(value)))
    defresult=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
      for i,res in pool.map(_flip_job,jobs,chunksize=1):
        key,value=metadata[i];a=_family(res,'single');b=_family(res,'heterogeneous')
        defresult.append({'param':key,'value':value,'delta':b['geomean_ms']/a['geomean_ms']-1,
            'single_ms':a['geomean_ms'],'hetero_ms':b['geomean_ms'],'proof_complete':res['proof_complete'],
            'gap_pct':res['gap_pct'],'slice':'all other parameters frozen at main defaults'})
        write_json(out/'search_certificates/flip'/f'{i:03d}.json',res)
    write_csv(out/'flip_samples.csv',defresult)
    crossings=[]
    for key in ranges:
      rows=sorted((r for r in defresult if r['param']==key),key=lambda r:r['value'])
      for target in (0,-.05):
        found=[]
        for a,b in zip(rows,rows[1:]):
          if (a['delta']-target)*(b['delta']-target)<=0 and a['delta']!=b['delta']:
            t=a['value']+(target-a['delta'])*(b['value']-a['value'])/(b['delta']-a['delta'])
            found.append({'param':key,'target_delta':target,'value':t,'bracket_low':a['value'],
                'bracket_high':b['value'],'status':'interpolated_between_sampled_candidates',
                'proof_complete':a['proof_complete'] and b['proof_complete']})
        crossings.extend(found or [{'param':key,'target_delta':target,'value':None,'bracket_low':None,
            'bracket_high':None,'status':'no_crossing_in_evaluated_range','proof_complete':False}])
    write_csv(out/'flip_boundary.csv',crossings)

def _flip_job(job):
    i,dev,p,cap,initial=job
    res=search_workloads(dev,p,delta=.02,time_limit_s=cap,initial_designs=initial,seed=20261007)
    res['resume_workloads']=dev
    return i,res

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['grid','sobol','flip'],required=True);ap.add_argument('--jobs',type=int,default=24);ap.add_argument('--point-seconds',type=float,default=2)
    args=ap.parse_args();globals()[args.stage](args)
if __name__=='__main__':main()
