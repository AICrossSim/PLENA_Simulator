"""Full finite-domain Shared-only search with safe fractional split bounds.

Native paper cores and output-replicated topology extensions are reported
separately. Every shortlisted split is solved exactly for its declared
contiguous row/column contracts; no unconstrained optimal-scheduler claim.
"""
import argparse, csv, gzip, hashlib, json, math, time, shutil
from pathlib import Path
import numpy as np
from .geometry import (Core, BUDGET, enumerate_cores, single_cores,
    homogeneous_cores, iter_heterogeneous_pairs, geometry_counts,
    enumerate_native_designs)
from .timing import BATCHES, H, F, PROFILES, partition, resource_ledger

def gm(xs):
    return float(np.exp(np.mean(np.log(xs))))

def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def csv_out(path, rows):
    rows=list(rows)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def config(cores):
    return ' + '.join(f'{c.pm}x{c.pn}x{c.groups}[b{c.b},R{c.r}]' for c in cores)

def core_cost_arrays(cores, profile):
    b=np.array([c.b for c in cores],float)[:,None]
    pm=np.array([c.pm for c in cores],float)[:,None]
    pn=np.array([c.pn for c in cores],float)[:,None]
    r=np.array([c.r for c in cores],float)[:,None]
    d=np.array([profile.drain(c) for c in cores],float)[:,None]
    cm=np.ceil(np.array(BATCHES)[None,:]/pm)
    nk_h=np.ceil(H/r); nk_f=np.ceil(F/r)
    if profile.serialized:
        sh=nk_h*b+d; sf=nk_f*b+d
        intercept=np.zeros_like(d)
    else:
        sh=nk_h*b; sf=nk_f*b
        intercept=d
    gu=2*cm*np.ceil(F/pn)*sh + intercept
    down=cm*np.ceil(H/pn)*sf + intercept
    # Relax only M- or N-ceil. K padding and native b-step cost remain priced.
    a_gu=2*cm*sh/pn; a_down=cm*sf/pn
    a_rows=(2*np.ceil(F/pn)*sh+np.ceil(H/pn)*sf)/pm
    return gu,down,a_gu,a_down,a_rows,intercept

def fractional_pair(total,a0,d0,a1,d1,e0,e1):
    x=(a1*total+d1-d0)/(a0+a1)
    interior=(a0*a1*total+a1*d0+a0*d1)/(a0+a1)
    interior=np.where((x>=0)&(x<=total),interior,np.inf)
    return np.minimum(np.minimum(e0,e1),interior)

def lower_bounds(pairs, arrays):
    gu,down,ag,ad,ar,d=arrays
    i,j=pairs[:,0],pairs[:,1]
    col=fractional_pair(F,ag[i],d[i],ag[j],d[j],gu[i],gu[j])
    col+=fractional_pair(H,ad[i],d[i],ad[j],d[j],down[i],down[j])
    row=fractional_pair(np.array(BATCHES)[None,:],ar[i],2*d[i],ar[j],2*d[j],
                         gu[i]+down[i],gu[j]+down[j])
    return np.exp(np.mean(np.log(np.minimum(col,row)),axis=1))

def evaluate(cores, profile):
    plans=[partition(cores,b,profile) for b in BATCHES]
    return gm([p['cycles'] for p in plans]),plans

def solve_domain(name, candidates, ids, bounds, profiles=PROFILES):
    chosen={}; certificates=[]; evaluations={}
    for pi,profile in enumerate(profiles):
        ordered=np.argsort(bounds[pi],kind='stable')
        best=float('inf'); best_ix=None; exact=0
        for ix in ordered:
            if bounds[pi][ix] > best+1e-8:
                break
            cs=tuple(candidates[int(k)] for k in ids[ix] if k>=0)
            score,plans=evaluate(cs,profile)
            evaluations[(int(ix),pi)]=(score,plans)
            exact+=1
            if score < best-1e-8:
                best,best_ix=score,int(ix)
        cs=tuple(candidates[int(k)] for k in ids[best_ix] if k>=0)
        chosen[profile.name]={'indices':ids[best_ix].tolist(),'cores':[c.to_dict() for c in cs],
            'geometry':config(cs),'GM_cycles':best,'plans':evaluations[(best_ix,pi)][1]}
        certificates.append({'domain':name,'profile':profile.name,'geometries':len(ids),
            'exact_split_evaluations':exact,'best_GM_cycles':best,
            'excluded_only_by_valid_lower_bound':len(ids)-exact})
    # Timing-robust minimax regret: compare against each profile's own optimum.
    opt=np.array([chosen[p.name]['GM_cycles'] for p in profiles])
    robust_bound=np.max(np.array(bounds)/opt[:,None],axis=0)
    best=float('inf'); best_ix=None; exact=0; best_scores=None
    for ix in np.argsort(robust_bound,kind='stable'):
        if robust_bound[ix] > best+1e-10:
            break
        cs=tuple(candidates[int(k)] for k in ids[ix] if k>=0)
        scores=[]
        for pi,p in enumerate(profiles):
            if (int(ix),pi) not in evaluations:
                evaluations[(int(ix),pi)]=evaluate(cs,p)
            scores.append(evaluations[(int(ix),pi)][0])
        score=float(np.max(np.array(scores)/opt)); exact+=1
        if score < best-1e-10:
            best,best_ix,best_scores=score,int(ix),scores
    cs=tuple(candidates[int(k)] for k in ids[best_ix] if k>=0)
    chosen['timing_robust']={'indices':ids[best_ix].tolist(),'cores':[c.to_dict() for c in cs],
        'geometry':config(cs),'max_regret':best-1,'profile_GM_cycles':dict(zip([p.name for p in profiles],best_scores)),
        'plans':[partition(cs,b,PROFILES[0]) for b in BATCHES]}
    certificates.append({'domain':name,'profile':'timing_robust','geometries':len(ids),
        'exact_split_evaluations':exact,'best_GM_cycles':gm(best_scores),
        'excluded_only_by_valid_lower_bound':len(ids)-exact})
    return chosen,certificates

def pruned_main(out):
    started=time.monotonic();out.mkdir(parents=True,exist_ok=True)
    cores=enumerate_cores(); index={c:i for i,c in enumerate(cores)}
    print('domain',geometry_counts(),flush=True)
    dump(out/'GEOMETRY_DOMAIN.json',geometry_counts())
    dump(out/'core_domain.json',[c.to_dict() for c in cores])
    pair_ids=np.array([(index[a],index[b]) for a,b in iter_heterogeneous_pairs()],dtype=np.int32)
    domains={
      'extended_single':np.array([(index[c],-1) for c in single_cores()],dtype=np.int32),
      'extended_homogeneous':np.array([(index[c],index[c]) for c in homogeneous_cores()],dtype=np.int32),
      'extended_heterogeneous':pair_ids,
    }
    native=enumerate_native_designs()
    for family, designs in native.items():
        ds=[(c,) if isinstance(c,Core) else tuple(c) for c in designs]
        domains['native_'+family]=np.array([(index[cs[0]],index[cs[1]] if len(cs)>1 else -1) for cs in ds],dtype=np.int32)
    arrays=[core_cost_arrays(cores,p) for p in PROFILES]
    results={};certs=[];all_bounds={}
    for name,ids in domains.items():
        if ids.shape[1]!=2: raise ValueError('badindices')
        bs=[]
        for pi,p in enumerate(PROFILES):
            if name.endswith('single'):
                gu,down,*_=arrays[pi]
                bound=np.exp(np.mean(np.log(gu[ids[:,0]]+down[ids[:,0]]),axis=1))
            else:
                bound=lower_bounds(ids,arrays[pi])
            bs.append(bound)
        results[name],c=solve_domain(name,cores,ids,bs)
        certs.extend(c);all_bounds[name]=bs
        print(name,results[name][PROFILES[0].name]['geometry'],results[name][PROFILES[0].name]['GM_cycles'],flush=True)
    dump(out/'FROZEN_SELECTION.json',results)
    csv_out(out/'search_certificates.csv',certs)
    # Full-domain lower-bound record; exact shortlisted scores in frozen results.
    with gzip.open(out/'all_geometry_bounds.csv.gz','wt',newline='') as f:
        w=csv.writer(f);w.writerow(['domain','core0_index','core1_index']+[p.name+'_GM_lower_bound_cycles' for p in PROFILES])
        for name,ids in domains.items():
            bs=all_bounds[name]
            for ix,(i,j) in enumerate(ids):
                w.writerow([name,int(i),int(j)]+[float(v[ix]) for v in bs])
    rows=[]
    for name,selection in results.items():
        for profile in PROFILES:
            cs=tuple(Core(**{k:c[k] for k in ('b','r','rm','rn')}) for c in selection[profile.name]['cores'])
            for batch in BATCHES:
                x=partition(cs,batch,profile)
                rows.append({'domain':name,'profile':profile.name,'batch':batch,
                    'geometry':config(cs),'compute_cycles':x['cycles'],'compute_ms':x['cycles']/1e6,
                    'partition':x['kind'],'gu_split':x['gu_split'],'down_split':x['down_split'],'row_split':x['row_split']})
    csv_out(out/'selected_timing_by_batch.csv',rows)
    dump(out/'COMPLETION_SEARCH.json',{'status':'PASS','elapsed_seconds':time.monotonic()-started,
        'native_calibrated':False,'domain_complete':True,'source_precision':'BF16/FP32 variant of paper topology',
        'optimization_contract':'exact contiguous row or GU/Down output-column partition; seven batches equal GM',
        'counts':geometry_counts(),'not_an_unrestricted_scheduling_optimum':True})

def finalize_exact(out):
    """Export full-enumeration evidence and a neutral co-optimal tie rule."""
    cores=enumerate_cores()
    selection=json.loads((out/'FROZEN_SELECTION.json').read_text())
    certificates=[];coopt={};timing_rows=[]
    for name, selected in selection.items():
        raw=np.load(out/(name+'_all_exact_scores.npz'))
        ids=raw['pair_ids'];scores=raw['profile_GM_cycles']
        for pi,p in enumerate(PROFILES):
            best=float(scores[pi].min())
            tied=np.flatnonzero(scores[pi]<=best+1e-8)
            ix=int(tied[0])
            if name.endswith('heterogeneous'):
                # No performance distinction exists between exact ties.
                # Prefer the least unequal multiplier split, then stable index.
                ix=min((int(i) for i in tied),key=lambda i:(
                    abs(cores[int(ids[i,0])].macs-cores[int(ids[i,1])].macs),i))
            cs=tuple(cores[int(i)] for i in ids[ix] if i>=0)
            selected[p.name]={'indices':ids[ix].tolist(),'cores':[c.to_dict() for c in cs],
                'geometry':config(cs),'GM_cycles':float(scores[pi,ix]),
                'plans':[partition(cs,b,p) for b in BATCHES],
                'exact_cooptimal_count':len(tied),
                'tie_break':'closest multiplier balance; stable index; no measured area preference'}
            certificates.append({'domain':name,'profile':p.name,'geometries':len(ids),
                'exact_split_evaluations':len(ids),'best_GM_cycles':best,
                'excluded_only_by_valid_lower_bound':0,'exact_cooptimal_count':len(tied),
                'within1pct_count':int(np.count_nonzero(scores[pi]<=1.01*best))})
            for b in BATCHES:
                plan=partition(cs,b,p)
                timing_rows.append({'domain':name,'profile':p.name,'batch':b,
                    'geometry':config(cs),'compute_cycles':plan['cycles'],
                    'compute_ms':plan['cycles']/1e6,'partition':plan['kind'],
                    'gu_split':plan['gu_split'],'down_split':plan['down_split'],'row_split':plan['row_split']})
            if pi==0:
                coopt[name]={'best_GM_cycles':best,'exact_cooptimal_count':len(tied),
                    'within1pct_count':int(np.count_nonzero(scores[pi]<=1.01*best)),
                    'cooptimal_geometries':[config(tuple(cores[int(k)] for k in ids[int(i)] if k>=0)) for i in tied]}
    dump(out/'FROZEN_SELECTION.json',selection)
    dump(out/'COOPTIMAL_GEOMETRIES.json',coopt)
    dump(out/'core_domain.json',[c.to_dict() for c in cores])
    dump(out/'GEOMETRY_DOMAIN.json',geometry_counts())
    csv_out(out/'search_certificates.csv',certificates)
    csv_out(out/'selected_timing_by_batch.csv',timing_rows)
    audit=json.loads((out/'VECTOR_AUDIT.json').read_text())
    dump(out/'COMPLETION_SEARCH.json',{'status':'PASS','backend':'complete vectorized integer enumeration',
        'elapsed_seconds':audit['elapsed_seconds'],'domain_complete':True,'native_calibrated':False,
        'source_precision':'BF16/FP32 variant of paper topology','counts':geometry_counts(),
        'optimization_contract':'exact contiguous row or GU/Down output-column partition; seven batches equal GM',
        'tie_break':'among exact ties prefer closest multiplier balance; no area optimality claim',
        'not_an_unrestricted_scheduling_optimum':True})

def main(out, backend='vector', import_audit=None):
    if backend=='pruned':
        if import_audit: raise ValueError('audit import applies only to vector backend')
        return pruned_main(out)
    out.mkdir(parents=True,exist_ok=True)
    if import_audit:
        for p in import_audit.iterdir():
            if p.suffix=='.npz' or p.name in ('FROZEN_SELECTION.json','VECTOR_AUDIT.json','REPEAT_AUDIT.json'):
                shutil.copy2(p,out/p.name)
    else:
        from .vector_search import main as vector_main
        vector_main(out)
    finalize_exact(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--backend',choices=('vector','pruned'),default='vector')
    ap.add_argument('--import-audit',type=Path)
    args=ap.parse_args();main(args.out,args.backend,args.import_audit)
