"""Calibrated synthetic workload regions and full Saltelli parameter scope.

Search certificates travel with every grid/sample. A time-limited point is
NOT labeled a globally optimized point; downstream indices report that limit.
"""
from __future__ import annotations
import argparse,itertools,json,math,time
from dataclasses import asdict,replace
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
    # Both endpoint targets are selected from development captures only.
    swe=[w for w in dev if 'swe' in w['id'].lower()]
    if not swe:raise ValueError('SWE development trace required')
    cohorts={}
    for w in dev:
        name=w['id'].lower()
        if 'swe' in name:continue
        cohort=next((key for key in ('bfcl','gpqa') if key in name),
                    'non_swe_mixed' if 'mixed' in name else 'other_development')
        cohorts.setdefault(cohort,[]).append(w)
    if not cohorts:raise ValueError('Non-SWE development trace required for diffuse endpoint')
    def route_entropy(w):
        counts=np.asarray([e['Me'] for e in w['experts'] if not e['is_shared']],dtype=float)
        probs=counts/counts.sum()
        ceiling=min(64,w['batch']*w.get('top_k',6))
        return float(-np.sum(probs*np.log(probs))/math.log(ceiling))
    scores={name:float(np.mean([route_entropy(w) for w in rows])) for name,rows in cohorts.items()}
    diffuse_name=min(cohorts,key=lambda name:(-scores[name],name))
    diffuse=cohorts[diffuse_name]
    def fit_endpoint(windows):
        def loss(logalpha):
            alpha=math.exp(logalpha);values=[]
            for j,w in enumerate(windows):
                ref=me_hist(w);n=sum(not e['is_shared'] for e in w['experts']);pred=[]
                for seed in range(8):
                    v=synthetic(w['batch'],alpha,2,64,6,1408,20261007+31*j+seed)
                    h=me_hist(v);dist=sum(not e['is_shared'] for e in v['experts'])
                    pred.append(np.sum(ref*np.log(ref/h))+(dist-n)**2/max(1,n*n))
                values.append(np.mean(pred))
            return float(np.mean(values))
        result=minimize_scalar(loss,bounds=(math.log(.003),math.log(100)),method='bounded',options={'xatol':.015})
        return {'alpha':math.exp(result.x),'loss':float(result.fun),'source_ids':[w['id'] for w in windows]}
    concentrated_fit=fit_endpoint(swe);diffuse_fit=fit_endpoint(diffuse)
    lo=min(concentrated_fit['alpha'],diffuse_fit['alpha']);hi=max(concentrated_fit['alpha'],diffuse_fit['alpha'])
    levels=np.geomspace(hi,lo,5).tolist();rows=[]
    for w in dev:
        ref=me_hist(w);routed=sum(not e['is_shared'] for e in w['experts'])
        for level,alpha in enumerate(levels):
            ss=[synthetic(w['batch'],alpha,2,64,6,1408,20261007+j) for j in range(16)]
            meanhist=np.mean([me_hist(v) for v in ss],axis=0);distinct=np.mean([sum(not e['is_shared'] for e in v['experts']) for v in ss])
            rows.append({'window_id':w['id'],'batch':w['batch'],'concentration_level':level,'alpha':alpha,
                'real_distinct':routed,'synthetic_mean_distinct':float(distinct),
                'distinct_relative_error':float((distinct-routed)/routed),'me_hist_KL_real_to_synthetic':float(np.sum(ref*np.log(ref/meanhist)))})
    return {'levels':levels,'concentrated_fit_loss':concentrated_fit['loss'],'fit_source_ids':concentrated_fit['source_ids'],
            'diffuse_fit_loss':diffuse_fit['loss'],'diffuse_fit_source_ids':diffuse_fit['source_ids'],
            'endpoint_fits':{'concentrated_swe':concentrated_fit,'diffuse_development':diffuse_fit},
            'fitted_endpoint_order_matches_expected':diffuse_fit['alpha']>=concentrated_fit['alpha'],
            'selection_protocol':{'scope':'development only; no heldout input',
                'diffuse_rule':'Highest cohort mean normalized routed-token entropy among non-SWE development cohorts; lexical tie break',
                'entropy_normalization':'Shannon entropy of expert Me / total routed tokens, divided by log(min(64, batch * topk))',
                'cohort_entropy_scores':scores,'selected_diffuse_cohort':diffuse_name,
                'cohort_source_ids':{name:[w['id'] for w in windows] for name,windows in cohorts.items()},
                'endpoint_label_scope':'Empirical diffuse and SWE fits; finite alpha does not assert exactly uniform routing',
                'loss':'Mean across endpoint windows of KL(real Me histogram || synthetic histogram) plus squared relative distinct-count error',
                'alpha_fit_bounds':[.003,100],'log_alpha_xatol':.015,
                'fit_seed_rule':'20261007 + 31 * endpoint_window_index + draw_index',
                'grid_level_order':'Five geometric levels from larger fitted alpha to smaller fitted alpha'},
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

def delta_intervals(result):
    """Bound ratios of family optima using independently certified LB/U pairs."""
    bounds={};fields={}
    for family,label in (('single','single'),('heterogeneous','hetero'),('homogeneous','homo')):
        row=_family(result,family) or {}
        lower=row.get('certified_global_lb_ms');upper=row.get('geomean_ms')
        lower=float(lower) if lower is not None else None
        upper=float(upper) if upper is not None else None
        if lower is not None and (not math.isfinite(lower) or lower<0):
            raise ValueError('Invalid certified family lower bound')
        if upper is not None and (not math.isfinite(upper) or upper<=0):
            raise ValueError('Invalid family incumbent latency')
        if lower is not None and upper is not None and lower>upper+max(1e-12,abs(upper)*1e-10):
            raise ValueError('Certified family lower bound exceeds incumbent latency')
        if lower is not None and upper is not None:lower=min(lower,upper)
        bounds[family]=(lower,upper)
        fields[label+'_certified_lb_ms']=lower
        fields[label+'_incumbent_ms']=upper
    lh,uh=bounds['heterogeneous']
    for baseline,label in (('single','single'),('homogeneous','homo')):
        lb,ub=bounds[baseline]
        fields['delta_lower_vs_'+label+'_pct']=100*(lh/ub-1) if lh is not None and ub else None
        fields['delta_upper_vs_'+label+'_pct']=100*(uh/lb-1) if uh is not None and lb else None
    fields['candidate_delta_scope']='Incumbent ratio is a candidate estimate; family-optimum ratio lies in [LB_hetero / U_baseline, U_hetero / LB_baseline] when the displayed bounds exist'
    return fields


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
            'hypothetical_bw_override':True,'model_scope':'synthetic_only',**delta_intervals(res)}
        results.append(row)
        if len(results)%32==0:write_csv(out/'workload_map.csv',sorted(results,key=lambda r:int(r['point_index'])));print('grid',len(results),'/4320',flush=True)
    write_csv(out/'workload_map.csv',sorted(results,key=lambda r:int(r['point_index'])))
    valid=[r for r in results if r['delta_vs_single_pct'] is not None and r['delta_vs_single_pct']!='']
    best=min(valid,key=lambda r:float(r['delta_vs_single_pct']))
    write_json(out/'workload_grid_summary.json',{'planned_points':len(combos),'completed_points':len(results),
        'certified_points':sum(str(r['proof_complete']).lower()=='true' for r in results),'best_evaluated_point':best,
        'scope':'full workload grid; time-limited full-domain hardware searches; uncertified points are candidate estimates'})


def _sobol_job(job):
    from .sensitivity import SensitivityParameters
    index,sample,dev,cap,initial=job
    tau,bank,dot,credits,vector=sample
    p=SensitivityParameters(weight_tile_service_cycles=float(tau),bank_Bpc=float(bank),dotstagecycles=float(dot),credits=int(round(credits)),vector_scale=float(vector))
    res=search_workloads(dev,p,delta=.02,time_limit_s=cap,initial_designs=initial,seed=20261007)
    res['resume_workloads']=dev
    a=_family(res,'single');b=_family(res,'heterogeneous')
    write_json(ROOT/'results/E3/search_certificates/sobol'/f'{index:04d}.json',res)
    interval=delta_intervals(res)
    return {'sample_index':index,'weight_tile_service_cycles':tau,'bank_Bpc':bank,'dotstagecycles':dot,'credits':int(round(credits)),
        'vector_scale':vector,'timing_model_sha256':p.timing_model_sha256,'delta':b['geomean_ms']/a['geomean_ms']-1,
        'single_ms':a['geomean_ms'],'hetero_ms':b['geomean_ms'],
        'single_design':canonical(a['design']),'hetero_design':canonical(b['design']),
        'proof_complete':res.get('proof_complete',False),'gap_pct':res.get('gap_pct'),
        'open_lb_ms':res.get('open_lb_ms'),**interval,
        'delta_lower':interval['delta_lower_vs_single_pct']/100 if interval['delta_lower_vs_single_pct'] is not None else None,
        'delta_upper':interval['delta_upper_vs_single_pct']/100 if interval['delta_upper_vs_single_pct'] is not None else None}


def sobol(args):
    from .sensitivity import SensitivityParameters,source_sha256
    from SALib.sample import sobol as sampler
    from SALib.analyze import sobol as analyzer
    out=ROOT/'results/E3';ws=inputs();selection=json.loads((out/'FROZEN_SELECTION.json').read_text())
    initial=[decode_design(v['design']) for v in selection['modes']['pipelined'].values() if isinstance(v,dict) and 'design' in v]
    problem={'num_vars':5,'names':['weight_tile_service_cycles','bank_Bpc','dotstagecycles','credits','vector_scale'],
        'bounds':[[1,30.4],[8,32],[1,4],[256,512],[.5,2]]}
    samples=sampler.sample(problem,256,calc_second_order=False,seed=20261007)
    assert len(samples)==1792
    old=read_csv(out/'sobol_samples.csv') if (out/'sobol_samples.csv').exists() else []
    if any('weight_tile_service_cycles' not in r or r.get('timing_model_sha256')!=source_sha256() for r in old):
        raise ValueError('Sobol checkpoint uses a different frontend timing model; preserve it separately before starting a new campaign')
    done={int(r['sample_index']) for r in old}
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
        'baseline_parameters':asdict(SensitivityParameters()),
        'parameter_class':'SensitivityParameters','timing_model_sha256':source_sha256(),
        'weight_tile_service_semantics':'Shared W frontend total bandwidth min(64 * bank_Bpc, 4096 / tau); per-core share w_banks / 64; datapath issue interval stays 1',
        'universal_W_bound_scope':'Ignores the extra frontend cap; remains conservative but may be looser',
        'delta_interval_semantics':'True family-optimum H/S delta lies in [LB_H/U_S - 1, U_H/LB_S - 1]; delta_lower/delta_upper are fractions, *_pct fields are percentages',
        'certified_samples':sum(str(r['proof_complete']).lower()=='true' for r in rows)})


def flip(args):
    from .sensitivity import SensitivityParameters,source_sha256
    out=ROOT/'results/E3';dev=inputs()['development'];selection=json.loads((out/'FROZEN_SELECTION.json').read_text())
    initial=[decode_design(v['design']) for v in selection['modes']['pipelined'].values() if isinstance(v,dict) and 'design' in v]
    ranges={'weight_tile_service_cycles':np.linspace(1,30.4,21),'bank_Bpc':np.linspace(8,32,21),
            'dotstagecycles':np.linspace(1,4,21),'credits':np.linspace(256,512,21),'vector_scale':np.linspace(.5,2,21)}
    jobs=[];metadata=[]
    for key,values in ranges.items():
      for value in values:
        p=replace(SensitivityParameters(),**{key:int(round(value)) if key=='credits' else float(value)})
        i=len(jobs);jobs.append((i,dev,p,args.point_seconds,initial));metadata.append((key,float(value)))
    defresult=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
      for i,res in pool.map(_flip_job,jobs,chunksize=1):
        key,value=metadata[i];a=_family(res,'single');b=_family(res,'heterogeneous')
        interval=delta_intervals(res)
        defresult.append({'param':key,'value':value,'delta':b['geomean_ms']/a['geomean_ms']-1,
            'single_ms':a['geomean_ms'],'hetero_ms':b['geomean_ms'],'proof_complete':res['proof_complete'],
            'gap_pct':res['gap_pct'],'timing_model_sha256':source_sha256(),
            'slice':'all other parameters frozen at main defaults',**interval,
            'delta_lower':interval['delta_lower_vs_single_pct']/100 if interval['delta_lower_vs_single_pct'] is not None else None,
            'delta_upper':interval['delta_upper_vs_single_pct']/100 if interval['delta_upper_vs_single_pct'] is not None else None})
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
    write_json(out/'flip_protocol.json',{'planned_points':105,'completed_points':len(defresult),
        'baseline_parameters':asdict(SensitivityParameters()),
        'parameter_class':'SensitivityParameters','timing_model_sha256':source_sha256(),
        'ranges':{key:values.tolist() for key,values in ranges.items()},
        'effective_credits_rounded_to_integer':True,
        'weight_tile_service_semantics':'Shared W frontend total bandwidth min(64 * bank_Bpc, 4096 / tau); per-core share w_banks / 64; datapath issue interval stays 1',
        'universal_W_bound_scope':'Ignores the extra frontend cap; remains conservative but may be looser',
        'slice':'All other parameters fixed at main defaults; tau defaults to 1 cycle',
        'delta_interval_semantics':'True family-optimum H/S delta lies in [LB_H/U_S - 1, U_H/LB_S - 1]; delta_lower/delta_upper are fractions, *_pct fields are percentages',
        'boundary_scope':'Linear interpolation of evaluated candidate ratios; open certificates are not global family optima'})

def _flip_job(job):
    i,dev,p,cap,initial=job
    res=search_workloads(dev,p,delta=.02,time_limit_s=cap,initial_designs=initial,seed=20261007)
    res['resume_workloads']=dev
    return i,res

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['grid','sobol','flip'],required=True);ap.add_argument('--jobs',type=int,default=24);ap.add_argument('--point-seconds',type=float,default=2)
    args=ap.parse_args();globals()[args.stage](args)
if __name__=='__main__':main()
