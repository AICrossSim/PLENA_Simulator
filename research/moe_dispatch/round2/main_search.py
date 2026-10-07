"""Full-domain certificates, single-family exhaustive seeds, and LB audit."""
from __future__ import annotations
import argparse,itertools,json,math,random,time
from dataclasses import asdict,replace
from concurrent.futures import ProcessPoolExecutor
from .common import *
from .optimizer import evaluate_design,region_bound,search_workloads
from .search import _workload_hash,_engine_hash
from research.moe_dispatch.geometry3d.compute import enumerate_geometries,enumeration_counts


def _point(job):
    d,dev,p=job
    try:
        first=[evaluate_design(w,d,p) for w in dev]
        if any(not r.get('legal',True) for r in first):return {'design':encode_design(d),'geometry':d.geometry,'family':d.family,'invalid':'an expert has no physical core'}
        second=[evaluate_design(w,d,p) for w in dev]
        assert canonical(first)==canonical(second),'allocation/replay repeat mismatch'
        return {'design':encode_design(d),'geometry':d.geometry,'flows':d.flows,'family':d.family,
            'geomean_ms':gmean(r['milp_sched']['latency_ms'] for r in first),
            'latencies_ms':[r['milp_sched']['latency_ms'] for r in first],
            'runtime_latencies_ms':[r['runtime']['latency_ms'] for r in first],
            'allocations_optimal':all(r['assignment']['optimal'] for r in first),
            'lb_ms':[r['lb_cycles']/1e6 for r in first],'repeat_identical':True,
            'parameters':asdict(p),'workload_sha256':_workload_hash(dev),'engine_sha256':_engine_hash()}
    except (ValueError,RuntimeError) as error:return {'design':encode_design(d),'geometry':d.geometry,'family':d.family,'invalid':str(error)}


def _validity(job):
    index,d,dev,p=job;rows=[]
    # Full-region bound, as used to cut the declared space, not only a singleton-specific bound.
    for w in dev:
        intervals={name:tuple((v,max(v,v+1024)) if name.endswith('_bytes') else (v,v+1) for v in getattr(d,name)) for name in ('w_bytes','x_bytes','acc_bytes','z_bytes','w_banks','x_banks','acc_banks','vector_lanes')}
        lb=region_bound(w,p,geometries=(d.cores,),intervals=intervals)
        first=simulate(w,d,p);second=simulate(w,d,p);assert canonical(first)==canonical(second)
        l=lb['lb_cycles'];t=first['cycles']
        rows.append({'sample_index':index,'design':canonical(encode_design(d)),'window_id':w['id'],
            'lb_ms':l/1e6,'sim_ms':t/1e6,'ok':l<=t+1e-6,'binding_term':lb['binding_term']})
        if l>t+1e-6:raise AssertionError(('LB validity failure',rows[-1]))
    return rows


def random_designs(n,seed=20261007):
    rng=random.Random(seed);geoms=enumerate_geometries();seen=set();result=[];attempts=0
    while len(result)<n:
        attempts+=1
        cs=geoms[rng.randrange(len(geoms))]
        if len(cs)==1:d=Design(cs,flows=(rng.choice(('OS','WS','IS')),))
        else:
            def split(total,quantum=1):
                a=rng.randrange(1,total//quantum);return (a*quantum,total-a*quantum)
            d=Design(cs,flows=tuple(rng.choice(('OS','WS','IS')) for _ in cs),
                w_bytes=split(40*1024,1024),x_bytes=split(12*1024,1024),acc_bytes=split(96*1024,1024),
                z_bytes=split(384*1024,1024),w_banks=split(64),x_banks=split(24),acc_banks=split(12),vector_lanes=split(64))
        key=canonical(encode_design(d))
        if key in seen:continue
        # Audit only physically feasible instantiated designs, track rejected configurations separately.
        try:
            for c in range(len(cs)):task_cost({'Me':1,'H':2048,'F':1408},d,c,Parameters())
        except ValueError:continue
        seen.add(key);result.append(d)
    return result,attempts


def validity(args):
    dev=inputs()['development'];ds,attempts=random_designs(2000);p=Parameters();rows=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
      for rs in pool.map(_validity,[(i,d,dev,p) for i,d in enumerate(ds)],chunksize=1):
        rows.extend(rs)
        if len(rows)%3600==0:print('LB audit',len(rows),'/36000',flush=True)
    out=ROOT/'results/E3';write_csv(out/'lb_validity.csv',rows)
    write_json(out/'lb_validity_protocol.json',{'concrete_feasible_designs':2000,'sampling_attempts':attempts,
        'development_windows':18,'checks':len(rows),'all_ok':all(r['ok'] for r in rows),
        'repeats':2,'seed':20261007,'warning':'Random checks support implementation confidence, not a mathematical proof'})


def seed_candidates(dev,p,jobs):
    geoms=enumerate_geometries();candidates=[]
    # Exhaust the finite single geometry×dataflow family; single capacities/ports are uniquely fixed.
    for cs in geoms:
      if len(cs)==1:
       for f in ('OS','WS','IS'):candidates.append(Design(cs,flows=(f,)))
    # Seed every identical-core geometry with all nine flows, plus declared reference asymmetric geometries.
    for cs in geoms:
      if len(cs)==2 and cs[0]==cs[1]:
       for fs in itertools.product(('OS','WS','IS'),repeat=2):candidates.append(Design(cs,flows=fs))
    reference=((Core(4,4,512),Core(2,4,512)),(Core(1,2,64),Core(5,19,128)),
               (Core(5,8,256),Core(1,8,256)),(Core(4,16,128),Core(2,16,128)),
               (Core(1,32,128),Core(2,32,128)))
    for cs in reference:
      for fs in itertools.product(('OS','WS','IS'),repeat=2):
       for profile in ('default','balanced','stream_bandwidth'):
        d=Design(cs,flows=fs)
        if profile=='balanced':
            d=replace(d,w_bytes=(20*1024,20*1024),w_banks=(32,32),x_banks=(12,12),acc_banks=(6,6),vector_lanes=(32,32))
        elif profile=='stream_bandwidth':
            small=min(range(2),key=lambda c:cs[c].macs)
            def roles(a,b):return (a,b) if small==0 else (b,a)
            d=replace(d,w_bytes=roles(24*1024,16*1024),w_banks=roles(32,32),x_banks=roles(4,20),
                acc_banks=roles(4,8),x_bytes=roles(2*1024,10*1024),z_bytes=roles(32*1024,352*1024),
                acc_bytes=roles(16*1024,80*1024),vector_lanes=roles(8,56))
        candidates.append(d)
    dedup={canonical(encode_design(d)):d for d in candidates};points=[]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
      for i,row in enumerate(pool.map(_point,[(d,dev,p) for d in dedup.values()],chunksize=1)):
        points.append(row)
        if i%32==0:print('seed',p.onchip_mode,i,'/',len(dedup),flush=True)
    return points


def main_search(args):
    dev=inputs()['development'];out=ROOT/'results/E3';out.mkdir(parents=True,exist_ok=True)
    frozen={'modes':{},'selection_split':'development only','objective':'paired window geometric mean',
        'BF16_only':True,'all_hardware_fixed_before_heldout':True};cert=[];leaves=[];summaries=[]
    for mode in MODES:
      p=Parameters(onchip_mode=mode);cache=out/f'seed_points_{mode}.json'
      points=json.loads(cache.read_text()) if cache.exists() else []
      if not points or any('invalid' not in r and (r.get('parameters')!=asdict(p) or r.get('workload_sha256')!=_workload_hash(dev) or r.get('engine_sha256')!=_engine_hash()) for r in points):
          points=seed_candidates(dev,p,args.jobs)
      write_json(cache,points)
      legal=[r for r in points if 'invalid' not in r];initial=[];m={}
      for family in ('single','homogeneous','heterogeneous'):
          best=min((r for r in legal if r['family']==family),key=lambda r:r['geomean_ms']);m[family]=best;initial.append(best)
      for family,quota in (('5+1',2048),('4+2',4096),('2+4',4096)):
          ss=[r for r in legal if r['family']=='heterogeneous' and min(c['pm']*c['pn']*c['pk'] for c in r['design']['cores'])==quota]
          if ss:m[family]=min(ss,key=lambda r:r['geomean_ms']);initial.append(m[family])
      initial.extend(sorted(legal,key=lambda r:r['geomean_ms'])[:12])
      for delta,proof in ((.05,'A'),(0.0,'B')):
          result=search_workloads(dev,p,delta=delta,time_limit_s=args.search_seconds,
              initial_designs=initial,seed=20261007,
              target_families=('single','homogeneous','heterogeneous','5+1','4+2'))
          write_json(out/f'bnb_{mode}_{proof}.json',result)
          summaries.append({'onchip_mode':mode,'proof':proof,'result':result})
          for region_index,r in enumerate(result.get('certificate',[])):
            cert.append({'proof':proof,'onchip_mode':mode,'region_id':f'{mode}_{proof}_{region_index}',
                'region_desc':canonical({k:v for k,v in r.items() if k not in ('lb_ms','incumbent_ms','status')}),
                'lb_geomean_ms':r['lb_ms'],'incumbent_ms':r.get('incumbent_ms'),
                'pruned_reason':r['status'],**r})
          for r in result.get('leaves',[]):leaves.append({'proof':proof,'onchip_mode':mode,**r})
          for family,x in result.get('families',{}).items():
            if x is not None and x.get('geomean_ms',x.get('score_ms',float('inf')))<m.get(family,{}).get('geomean_ms',float('inf')):
                m[family]={**x,'geomean_ms':x.get('geomean_ms',x.get('score_ms'))}
          initial=[x for family,x in m.items() if isinstance(x,dict) and 'design' in x]
      if '4+2' in m:
          m['2+4']={**m['4+2'],'search_family':'2+4',
              'mirror_alias_of':'4+2','alias_scope':'same physical MAC-ratio domain with swappable roles and all private resource cuts'}
      single_points=[r for r in points if r['family']=='single']
      single_exact=len(single_points)==132 and all('invalid' in r or r.get('allocations_optimal',False) for r in single_points)
      m['single']['proof_status']='complete_single_geometry_dataflow_scope' if single_exact else 'single_all_geometries_evaluated_with_unresolved_allocations'
      m['single']['proof_complete']=single_exact
      m['all_family_optima_certified']=bool(result.get('proof_complete',False))
      m['search_scope']='full declared domain; open regional frontiers remain if time cap reached'
      frozen['modes'][mode]=m
      print('search mode frozen',mode,flush=True)
    write_json(out/'FROZEN_SELECTION.json',frozen);write_json(out/'bnb_summary.json',{'runs':summaries,'geometry_counts':enumeration_counts(),
        'all_family_proofs_complete':all(s['result'].get('proof_complete',False) for s in summaries),
        'proof_A_conclusions':{s['onchip_mode']:{'global_delta_certificate':s['result'].get('global_delta_proof',False),
           'scope':'geometric mean across18 development windows; factor1.05 convention; not heldout calibration',
           'family_proofs_complete':s['result'].get('proof_complete',False)} for s in summaries if s['proof']=='A'},
        'coverage_pct':min((x['coverage_pct'] for s in summaries for x in s['result']['families'].values()),default=0),
        'total_recorded_regions':sum(len(s['result']['certificate'])+sum(len(x['open_regions']) for x in s['result']['families'].values()) for s in summaries),
        'pruned_regions':sum(len(s['result']['certificate']) for s in summaries),
        'capacity_resolution_B':1024,'vector_domain':'integer positive split of existing64 elements/cycle total',
        'mandatory_math_corrections':['Never force slow-but-legal core ownership','Region reload LB uses largest feasible buffers',
            'CP-SAT is resource assignment relaxation; cold HBM waits never summed percore']})
    write_csv(out/'bnb_certificate.csv',cert);write_csv(out/'bnb_leaves.csv',leaves)
    # All seed points are concrete full evaluations, kept separate from the proof frontier.
    write_csv(out/'seed_leaves.csv',[{'onchip_mode':mode,'design':canonical(r['design']),'family':r['family'],
        'geomean_ms':r.get('geomean_ms'),'invalid_reason':r.get('invalid',''),'repeat_identical':r.get('repeat_identical',False)}
        for mode in MODES for r in json.loads((out/f'seed_points_{mode}.json').read_text())])


def schedule_gaps(args):
    dev=inputs()['development'];held=inputs()['heldout'];sel=json.loads((ROOT/'results/E3/FROZEN_SELECTION.json').read_text());rows=[]
    for mode,m in sel['modes'].items():
      p=Parameters(onchip_mode=mode)
      for family in ('single','homogeneous','heterogeneous'):
        d=decode_design(m[family]['design'])
        for w in dev+held:
          a=evaluate_design(w,d,p);b=evaluate_design(w,d,p);assert canonical(a)==canonical(b)
          rows.append({'design':family,'geometry':d.geometry,'window_id':w['id'],'batch':w['batch'],'onchip_mode':mode,
            'T_lb':a['lb_cycles']/1e6,'T_milp_sched':a['milp_sched']['latency_ms'],'T_runtime_eft':a['runtime']['latency_ms'],
            'gap_sched_pct':a['gap_sched_pct'],'gap_runtime_pct':a['gap_runtime_pct'],
            'solver_status':a['assignment']['status'],'units':'ms'})
    write_csv(ROOT/'results/E3/schedule_gaps.csv',rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('search','validity','gaps'),required=True);ap.add_argument('--jobs',type=int,default=24);ap.add_argument('--search-seconds',type=float,default=120)
    args=ap.parse_args();{'search':main_search,'validity':validity,'gaps':schedule_gaps}[args.stage](args)
if __name__=='__main__':main()
