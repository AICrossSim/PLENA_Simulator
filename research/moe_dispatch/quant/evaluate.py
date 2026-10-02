#!/usr/bin/env python3
"""Real-weight Q0--Q3 evaluator; rejects missing or leaking calibration/eval data.

Full matrix defaults match TASK_V3. --allow-incomplete is a provenance-labeled
pilot only and cannot freeze precision. Q4 is a separate model evaluation tool.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threadpoolctl import threadpool_limits
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v3_reference'))
import ref_numerics as rn
import budget_v3 as bv

# Shards change only scheduling of independent experts/layers, never arithmetic.
BLAS_THREADS = 2
HARDWARE_BF16_POLICY = False


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()


class Checkpoint:
    def __init__(self,path,expected=None):
        self.path=Path(path);self.index=json.loads((self.path/'model.safetensors.index.json').read_text())['weight_map'];self.records={};self.expected=expected or {}
    def weight(self,layer,e,phase):
        stem='shared_experts' if e==-1 else f'experts.{e}'
        name=f'model.layers.{layer}.mlp.{stem}.{phase}_proj.weight';shard=self.path/self.index[name]
        with safe_open(shard,framework='pt',device='cpu') as f:t=f.get_tensor(name)
        if t.dtype!=torch.bfloat16:raise ValueError(f'{name}: weight is not BF16')
        raw=t.contiguous().view(torch.uint16).numpy().tobytes();hashvalue=hashlib.sha256(raw).hexdigest()
        prior=self.expected.get(name)
        if prior and hashvalue!=prior['sha256']:raise ValueError(f'{name}: captured tensor hash mismatch')
        self.records[name]={'shard':str(shard.resolve()),'sha256':hashvalue,'shape':list(t.shape),'dtype':'BF16',
                            'expected_hash_verified':bool(prior),'hash_origin':'local BF16 checkpoint tensor'}
        return t.float().numpy()


def load_capture(path):
    path=Path(path);manifest=json.loads(path.read_text());arrays={};request_hashes=[];count=0
    for shard in manifest['shards']:
        p=Path(shard['path']);p=p if p.is_absolute() else path.parent/p
        if sha(p)!=shard['sha256']:raise ValueError(f'capture hash mismatch {p}')
        if shard.get('prefix_truncated_smoke'):raise ValueError('throughput smoke is not a calibration capture')
        request_hashes.append(shard['request_sha256']);count+=shard['tokens']
        with np.load(p) as z:
            for layer in manifest['layers']:
                row=arrays.setdefault(int(layer),{'x':[],'routes':[],'gates':[]})
                for key,prefix in [('x','x'),('routes','routes'),('gates','gates')]:row[key].append(z[f'{prefix}_l{layer}'])
    arrays={l:{k:np.concatenate(v) for k,v in row.items()} for l,row in arrays.items()}
    return manifest,arrays,set(request_hashes),count


def cosine(a,b):
    a=np.asarray(a,np.float64).reshape(-1);b=np.asarray(b,np.float64).reshape(-1)
    return float(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30))


def factor_a(A,fmt):
    if np.asarray(A).size==0:return np.asarray(A,np.float32)
    return rn.mxint8_split(A,0)[3] if fmt=='mxint8s' else rn.quantize_factor(A,fmt,0)


def write_csv(path,rows):
    if not rows:return
    # Preserve the previous verified checkpoint on ENOSPC or interruption.
    # Temporary files are task-local and the published name changes atomically.
    path=Path(path);temporary=path.with_name(path.name+f'.checkpoint.{os.getpid()}')
    try:
        with temporary.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def hardware_q3_group_complete(folder,layer,bit,method,L,tokens):
    """Only reuse complete physical-LUT Q3 groups under the existing contract."""
    wanted={(u,t,s) for u in (16,32) for t in range(0,tokens,16)
            for s in ('uniform','frequency_static','gate_weighted_budget_oracle','gate_weighted_causal')}
    try:
        for kind in ('hardware_bf16','rank_vector_comparison'):
            p=folder/f'q3_{kind}_l{layer}_w{bit}_{method}_L{L}.csv'
            with p.open() as f:rr=list(csv.DictReader(f))
            keys=[(int(r['uniform_rank']),int(r['window_start']),r['strategy']) for r in rr]
            if set(keys)!=wanted or len(keys)!=len(wanted):return False
            if any(int(r['layer'])!=layer or int(r['bits'])!=bit or r['method']!=method or int(r['rank_lanes'])!=L for r in rr):return False
        for uniform in (16,32):
            table=json.loads((folder/f'rank_table_hw_bf16_l{layer}_w{bit}_{method}_uniform{uniform}.json').read_text())
            if not table['origin'].startswith('physical BF16-energy LUT') or table['lambda_fit']['method']!='bisection_on_mean_actual_development_window_bytes':return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def q1_coverage(rows,layers,bits,ranks,methods,lane_modes,factor_a,factor_b):
    """Exhaustive requested-candidate coverage; clipped duplicates still count."""
    expected=set()
    for layer in layers:
        for bit in list(bits)+[8]:
            for method in (['rtn'] if bit==8 else methods):
                for L in lane_modes:
                    rr=[0] if method=='rtn' else list(ranks)+([96] if L==16 and 96 not in ranks else [])
                    for r in rr:
                        for af in (['mxint4'] if method=='rtn' else factor_a):
                            for bf in (['bf16'] if method=='rtn' else factor_b):
                                expected.add((int(layer),int(L),int(bit),method,int(r),af,bf))
    counts=Counter((int(v['layer']),int(v['rank_lanes']),int(v['bits']),v['method'],int(v['rank']),v['factor_a'],v['factor_b'])
                   for v in rows if v['scope']=='layer')
    actual=set(counts);missing=sorted(expected-actual);unexpected=sorted(actual-expected)
    duplicates=sorted(k for k,n in counts.items() if n!=1)
    return {'schema':'plena_v3_q1_coverage_v1','expected_candidates':len(expected),'completed_candidates':len(actual),
            'complete':not missing and not unexpected and not duplicates,'missing_candidates':missing,
            'unexpected_candidates':unexpected,'duplicate_candidates':duplicates,
            'key_fields':['layer','rank_lanes','main_bits','method','requested_rank','factor_a','factor_b'],
            'scope':'one completed actual layer-FFN output metric per requested combination; effective-rank duplicate math may be reused'}


def original_z_hw(x,w,gate):
    """Original BF16-weight Gate/Up with the same K order and gate-folded Z contract."""
    g=rn._segment_matmul(x,w['g'].T);u=rn._segment_matmul(x,w['u'].T)
    sigmoid=np.empty_like(g);positive=g>=0
    sigmoid[positive]=1/(1+np.exp(-g[positive]));ex=np.exp(g[~positive]);sigmoid[~positive]=ex/(1+ex)
    return rn.bf16_round((g*sigmoid*u)*np.asarray(gate,np.float32)[:,None])


def gate_up_parts(x,w):
    return {k:[(x[:,lo:lo+512]@w[k][:,lo:lo+512].T).astype(np.float32)
               for lo in range(0,x.shape[1],512)] for k in ('g','u')}


def ffn_reuse_gate_up(x,w,fs,gate,L,parts):
    """Same hardware FP32 sequence, reusing factor-independent main Gate/Up partials."""
    values={}
    for k in ('g','u'):
        A,B=fs[k];r=A.shape[1];segments=rn.rank_segments(r,x.shape[1],L)
        if r>L*bv.nseg(x.shape[1]):return rn.expert_ffn_hw(x,w['g'],w['u'],w['d'],fs,gate,L)
        U=rn.bf16_round(rn._segment_matmul(x,A));acc=np.zeros_like(parts[k][0])
        for p,R in zip(parts[k],segments):
            partial=p.copy()
            if R:partial=(partial+U[:,R]@B[R,:]).astype(np.float32)
            acc=np.add(acc,partial,dtype=np.float32)
        values[k]=acc
    g=values['g'];sigmoid=np.empty_like(g);positive=g>=0
    sigmoid[positive]=1/(1+np.exp(-g[positive]));eg=np.exp(g[~positive]);sigmoid[~positive]=eg/(1+eg)
    z=rn.bf16_round((g*sigmoid*values['u'])*np.asarray(gate,np.float32)[:,None])
    return rn.fused_rank_lane_gemm_hw(z,w['d'],*fs['d'],L=L)


def layer_problem(cp,layer,cal,ev):
    """Down calibration/eval activations are produced by original BF16 weights."""
    problems={};weights={};out={};indices={};gates={};cold=[]
    for e in range(64):
        ic=np.where(np.any(cal['routes']==e,axis=1))[0];ie=np.where(np.any(ev['routes']==e,axis=1))[0]
        problems[e]={'cal_indices':ic,'eval_indices':ie};cold.append({'layer':layer,'expert':e,'calibration_tokens':len(ic),'evaluation_tokens':len(ie),'cold':len(ic)<32})
    for e in list(range(64))+[-1]:
        w={k:cp.weight(layer,e,name) for k,name in [('g','gate'),('u','up'),('d','down')]};weights[e]=w
        ic=np.arange(len(cal['x'])) if e==-1 else problems[e]['cal_indices'];ie=np.arange(len(ev['x'])) if e==-1 else problems[e]['eval_indices']
        xc=rn.bf16_round(cal['x'][ic]);xe=rn.bf16_round(ev['x'][ie]);indices[e]=ie
        if e==-1:
            gates[e]=np.ones(len(ie),np.float32);gc=np.ones(len(ic),np.float32)
        else:
            gates[e]=np.asarray([ev['gates'][i,list(ev['routes'][i]).index(e)] for i in ie],np.float32)
            gc=np.asarray([cal['gates'][i,list(cal['routes'][i]).index(e)] for i in ic],np.float32)
        zc=original_z_hw(xc,w,gc);ze=original_z_hw(xe,w,gates[e])
        problems[e]={'xc':xc,'xe':xe,'zc':zc,'ze':ze,'ic':ic,'ie':ie,'gc':gc,'ge':gates[e]}
        out[e]=rn.expert_ffn_hw(xe,w['g'],w['u'],w['d'],gate=gates[e]);problems[e]['gold']=out[e]
        if e%8==0:print(f'layer{layer} original expert{e}: cal{len(ic)} eval{len(ie)}',flush=True)
    return problems,weights,out,indices,gates,cold


def evaluate_cold_statistics(layer,problems,ws,cal,pool_down,lanes):
    rows=[]
    for e in range(64):
        p=problems[e]
        if len(p['xc'])>=32 or not len(p['xe']):continue
        w=ws[e];qw={k:rn.mx_quantize(W,4)[2] for k,W in w.items()}
        baseline=rn.expert_ffn_hw(p['xe'],w['g'],w['u'],w['d'],gate=p['ge'])
        for label,method,use_self in [('expert_own','qera_exact',True),('layer_pooled','qera_exact',False),('pooled_diagonal','qera_approx',False)]:
            if use_self and not len(p['xc']):
                rows.append({'layer':layer,'expert':e,'calibration_tokens':0,'evaluation_tokens':len(p['xe']),'statistics':label,'relative_error':'','cosine':'','status':'no_expert_samples'})
                continue
            fs={}
            for k in ('g','u','d'):
                xc=(p['zc'] if k=='d' else p['xc']) if use_self else (pool_down if k=='d' else cal['x'])
                r=min(8,lanes*bv.nseg(w[k].shape[1]));A,B,_=rn.lowrank_factors(w[k],qw[k],xc,r,method)
                fs[k]=(factor_a(A,'mxint4'),rn.quantize_b_per_segment(B,rn.rank_segments(r,w[k].shape[1],lanes),'bf16'))
            y=rn.expert_ffn_hw(p['xe'],qw['g'],qw['u'],qw['d'],fs,gate=p['ge'],L=lanes)
            rows.append({'layer':layer,'expert':e,'calibration_tokens':len(p['xc']),'evaluation_tokens':len(p['xe']),'statistics':label,
                         'relative_error':rn.rel_err(y,baseline),'cosine':cosine(y,baseline),'status':'measured_real'})
    return rows


def fit_lambda_window_budget(cal,te,rb,uniform_index):
    """Calibration-only bisection over real per-window discrete allocations."""
    E=rb.shape[0];windows=[];freq=np.zeros(E)
    for start in range(0,len(cal['x']),16):
        w=np.zeros(E);active=np.zeros(E,bool)
        for route,score in zip(cal['routes'][start:start+16],cal['gates'][start:start+16]):
            for ee,g in zip(route,score):w[ee]+=float(g)**2;active[ee]=True;freq[ee]+=1
        windows.append((w,active))
    calw=np.stack([w for w,_ in windows]);active=np.stack([a for _,a in windows]);freq/=max(1,freq.sum())
    mean_window_budget=float((active*rb[None,:,uniform_index]).sum(axis=1).mean());low,high=1e-16,1e2
    for _ in range(120):
        mid=np.sqrt(low*high);ch=np.argmin(calw[:,:,None]*te[None,:,:]+mid*rb[None,:,:],axis=2)
        mean_spent=float((rb[np.arange(E)[None,:],ch]*active).sum(axis=1).mean())
        if mean_spent>mean_window_budget:low=mid
        else:high=mid
    ch=np.argmin(calw[:,:,None]*te[None,:,:]+high*rb[None,:,:],axis=2)
    mean_spent=float((rb[np.arange(E)[None,:],ch]*active).sum(axis=1).mean())
    fit={'method':'bisection_on_mean_actual_development_window_bytes','window_tokens':16,'window_count':len(windows),
         'mean_uniform_window_bytes':mean_window_budget,'mean_allocated_window_bytes':mean_spent,'byte_delta':mean_spent-mean_window_budget}
    return float(high),fit,freq


def rank_calibration(layer,bit,method,ws,factors,cal,lanes,uniform_rank=32):
    """Projection-specific rank capacities, real packed bytes, calibration-only spectra."""
    choices=np.array([0,8,16,24,32]);E=64;phase_names={'g':'gate','u':'up','d':'down'}
    phase_te={};phase_bytes={};phase_caps={}
    for k,name in phase_names.items():
        caps=np.array([min(lanes*bv.nseg(ws[e][k].shape[1]),factors[e][k][0].shape[1]) for e in range(E)])
        phase_caps[name]=caps
        phase_te[name]=np.array([rn.tail_energy(factors[e][k][2],[min(int(r),int(caps[e])) for r in choices]) for e in range(E)])
        values=[]
        for e in range(E):
            W=ws[e][k];base=bv.plan_projection(k,W.shape[1],W.shape[0],0,bv.Fmt('q',bit,'bf16',lanes,'mxint4'))
            row=[]
            for r in choices:
                P=bv.plan_projection(k,W.shape[1],W.shape[0],min(int(r),int(caps[e])),bv.Fmt('q',bit,'bf16',lanes,'mxint4'))
                row.append(P.a_bytes+P.b_sep_bytes+P.main_bytes-base.main_bytes)
            values.append(row)
        phase_bytes[name]=np.asarray(values,float)
    te=sum(phase_te.values());rb=sum(phase_bytes.values());capidx=np.full(E,4)
    # Fit lambda to mean actual development-window bytes. Averaging gate
    # weights before the argmin is invalid because discrete allocation is
    # nonlinear and an inactive expert must not consume this window's budget.
    uniform_index=int(np.where(choices==uniform_rank)[0][0]);budget=rb[:,uniform_index].sum()
    high,fit,freq=fit_lambda_window_budget(cal,te,rb,uniform_index)
    table={'schema':'plena_v3_rank_calibration_v2','layer':layer,'bits':bit,'method':method,'rank_candidates':choices.tolist(),
           'tail_energy_per_projection':{k:v.tolist() for k,v in phase_te.items()},
           'factor_bytes_per_projection':{k:v.tolist() for k,v in phase_bytes.items()},
           'capacity_per_projection':{k:v.tolist() for k,v in phase_caps.items()},
           'tail_energy':te.tolist(),'factor_bytes':rb.tolist(),'cap_index':capidx.tolist(),
           'lambda0':float(high),'lambda_eta':0.5,'lambda_bounds':[float(high)/100,float(high)*100],
           'uniform_reference_rank':uniform_rank,'uniform_factor_budget_bytes':float(budget),
           'lambda_fit':fit,
           'shared_ranks':dict(zip(('gate','up','down'),bv.DEFAULT_RANKS[lanes]['shared'])),
           'origin':'calibration-only singular spectra and routing gate statistics','calibration_tokens':len(cal['x']),
           'initial_control_state':{'lambda':float(high),'windows_observed':0}}
    return table,choices,te,rb,capidx,freq


def evaluate_rank_policies(layer,bit,method,cpdata,ws,factors,cal,ev,lanes,outdir,budget_ranks=(16,32),quantized_weights=None,gu_parts=None,include_software=True):
    """Actual FFN errors; default 32/32/24 and lower-byte operating points stay separate.

    The exact-budget oracle is labeled separately from the causal lambda feedback
    controller. A discrete allocation may spend less than its cap: byte deltas are
    explicit, so approximate equal-byte comparisons are never hidden.
    """
    E=64;rows=[];hardware_rows=[];vector_rows=[];quant=quantized_weights or {e:{k:rn.mx_quantize(W,bit)[2] for k,W in w.items()} for e,w in ws.items()}
    table,choices,te,rb,capidx,freq=rank_calibration(layer,bit,method,ws,factors,cal,lanes)
    hardware_te=sum(rn.bf16_round(np.asarray(v,np.float32)).astype(np.float64) for v in table['tail_energy_per_projection'].values())
    (outdir/f'rank_table_l{layer}_w{bit}_{method}.json').write_text(json.dumps(table,indent=2)+'\n')
    # Prequantize factor prefixes once, rather than redoing full matrices each window.
    factor_cache={}
    def get_factors(e,index):
        key=(int(e),int(index))
        if key not in factor_cache:
            fs={}
            for k in ('g','u','d'):
                A,B,_=factors[e][k];r=min(A.shape[1],lanes*bv.nseg(ws[e][k].shape[1]),int(choices[index]) if e!=-1 else bv.DEFAULT_RANKS[lanes]['shared'][('g','u','d').index(k)])
                fs[k]=(factor_a(A[:,:r],'mxint4'),rn.quantize_b_per_segment(B[:r,:],rn.rank_segments(r,ws[e][k].shape[1],lanes),'bf16'))
            factor_cache[key]=fs
        return factor_cache[key]
    full_outputs={};local_maps={}
    def policy_worker(e):
        positions=cpdata[e]['ie'];x=cpdata[e]['xe'];gs=cpdata[e]['ge']
        mapping=np.full(len(ev['x']),-1,int);mapping[positions]=np.arange(len(positions))
        parts=gu_parts[e] if gu_parts is not None else gate_up_parts(x,quant[e]);values={}
        for index in ([4] if e==-1 else range(len(choices))):
            values[index]=ffn_reuse_gate_up(x,quant[e],get_factors(e,index),gs,lanes,parts)
        return e,mapping,values
    with threadpool_limits(limits=BLAS_THREADS),ThreadPoolExecutor(max_workers=8) as pool:
        for e,mapping,values in pool.map(policy_worker,list(range(E))+[-1]):
            local_maps[e]=mapping;full_outputs[e]=values
    for uniform_rank in budget_ranks:
        tb,_,_,_,_,_=rank_calibration(layer,bit,method,ws,factors,cal,lanes,uniform_rank)
        (outdir/f'rank_table_l{layer}_w{bit}_{method}_uniform{uniform_rank}.json').write_text(json.dumps(tb,indent=2)+'\n')
        ui=int(np.where(choices==uniform_rank)[0][0]);lam0=tb['lambda0'];lam=lam0
        hardware_lam0,hardware_fit,_=fit_lambda_window_budget(cal,hardware_te,rb,ui) if HARDWARE_BF16_POLICY else (lam0,tb['lambda_fit'],freq)
        hardware_lam=hardware_lam0
        if HARDWARE_BF16_POLICY:
            hwtable=dict(tb)
            hwtable['tail_energy_per_projection']={k:rn.bf16_round(np.asarray(v,np.float32)).tolist() for k,v in tb['tail_energy_per_projection'].items()}
            hwtable['tail_energy']=hardware_te.tolist();hwtable['lambda0']=hardware_lam0;hwtable['lambda_fit']=hardware_fit
            hwtable['lambda_bounds']=[hardware_lam0/100,hardware_lam0*100]
            hwtable['initial_control_state']={'lambda':hardware_lam0,'windows_observed':0}
            hwtable['software_float_lambda0']=lam0;hwtable['software_float_lambda_fit']=tb['lambda_fit']
            hwtable['physical_energy_storage']='BF16 RNE perprojection; decoded values summed; exact uint32 bytecost'
            hwtable['origin']='physical BF16-energy LUT and actual development-window routed factor bytes; validation never used to calibrate'
            weights=[];actives=[]
            for lo in range(0,len(cal['x']),16):
                ww=np.zeros(E);rr=cal['routes'][lo:lo+16];gg=cal['gates'][lo:lo+16]
                for routes_,scores_ in zip(rr,gg):
                    for ee,g in zip(routes_,scores_):ww[ee]+=float(g)**2
                weights.append(ww);actives.append(np.isin(np.arange(E),np.unique(rr)))
            weights=np.asarray(weights);actives=np.asarray(actives)
            idx=np.argmin(weights[:,:,None]*hardware_te[None,:,:]+lam0*rb[None,:,:],axis=2)
            diagnostic_mean=float((rb[np.arange(E)[None,:],idx]*actives).sum(axis=1).mean())
            hwtable['float_lambda0_on_hardware_diagnostic']={'mean_allocated_window_bytes':diagnostic_mean,
                'mean_uniform_window_bytes':hardware_fit['mean_uniform_window_bytes'],
                'byte_delta':diagnostic_mean-hardware_fit['mean_uniform_window_bytes']}
            (outdir/f'rank_table_hw_bf16_l{layer}_w{bit}_{method}_uniform{uniform_rank}.json').write_text(json.dumps(hwtable,indent=2)+'\n')
        previous_budget=None;update_count=0
        for start in range(0,len(ev['x']),16):
            x=ev['x'][start:start+16];routes=ev['routes'][start:start+16];scores=ev['gates'][start:start+16];active=np.unique(routes)
            wgt=np.zeros(E)
            for rr,gg in zip(routes,scores):
                for ee,g in zip(rr,gg):wgt[ee]+=float(g)**2
            budget=float(rb[active,ui].sum())
            if previous_budget is not None and budget!=previous_budget:lam=lam0
            if lam<lam0/100 or lam>lam0*100:lam=lam0
            if previous_budget is not None and budget!=previous_budget:hardware_lam=hardware_lam0
            if hardware_lam<hardware_lam0/100 or hardware_lam>hardware_lam0*100:hardware_lam=hardware_lam0
            policies={'uniform':np.full(E,ui)}
            for strategy,importance in [('frequency_static',freq),('gate_weighted_budget_oracle',wgt)]:
                low,high=1e-16,1e2
                for _ in range(100):
                    mid=np.sqrt(low*high);ch=rn.allocate_ranks(importance[active],te[active],rb[active],mid,capidx[active])
                    if rb[active,ch].sum()>budget:low=mid
                    else:high=mid
                idx=np.zeros(E,int);idx[active]=rn.allocate_ranks(importance[active],te[active],rb[active],high,capidx[active]);policies[strategy]=idx
            causal=np.zeros(E,int);causal[active]=rn.allocate_ranks(wgt[active],te[active],rb[active],lam,capidx[active]);policies['gate_weighted_causal']=causal
            hardware_policies={'uniform':np.full(E,ui)}
            if HARDWARE_BF16_POLICY:
                for strategy,importance in [('frequency_static',freq),('gate_weighted_budget_oracle',wgt)]:
                    low,high=1e-16,1e2
                    for _ in range(100):
                        mid=np.sqrt(low*high);ch=rn.allocate_ranks(importance[active],hardware_te[active],rb[active],mid,capidx[active])
                        if rb[active,ch].sum()>budget:low=mid
                        else:high=mid
                    idx=np.zeros(E,int);idx[active]=rn.allocate_ranks(importance[active],hardware_te[active],rb[active],high,capidx[active]);hardware_policies[strategy]=idx
                idx=np.zeros(E,int);idx[active]=rn.allocate_ranks(wgt[active],hardware_te[active],rb[active],hardware_lam,capidx[active]);hardware_policies['gate_weighted_causal']=idx
            golden=np.zeros((len(x),x.shape[1]),np.float32)
            parts=[]
            for e in list(active)+[-1]:
                positions=np.arange(len(x)) if e==-1 else np.where(np.any(routes==e,axis=1))[0]
                gs=np.ones(len(positions),np.float32) if e==-1 else np.array([scores[i,list(routes[i]).index(e)] for i in positions],np.float32)
                local=local_maps[e][start+positions];golden[positions]+=cpdata[e]['gold'][local];parts.append((e,positions,local))
            def output_metrics(idx):
                out=np.zeros_like(golden)
                for e,positions,local in parts:
                    out[positions]+=full_outputs[e][idx[e] if e!=-1 else 4][local]
                return rn.rel_err(out,golden),cosine(out,golden)
            for strategy,idx in policies.items():
                error,cos=output_metrics(idx)
                spent=float(rb[active,idx[active]].sum())
                if include_software:rows.append({'layer':layer,'bits':bit,'method':method,'window_start':start,'tokens':len(x),'strategy':strategy,
                    'uniform_rank':uniform_rank,'uniform_projection_ranks':f"{uniform_rank}/{uniform_rank}/{min(uniform_rank,table['capacity_per_projection']['down'][0])}",
                    'factor_bytes':spent,'uniform_budget_bytes':budget,'byte_delta':spent-budget,
                    'relative_error':error,'cosine':cos,'lambda0_calibration':lam0,
                    'lambda_before':lam if strategy=='gate_weighted_causal' else '',
                    'warmup':update_count<8,'scope':'actual_real_FFN_not_singular_energy_proxy'})
                if HARDWARE_BF16_POLICY:
                    hi=hardware_policies[strategy];same=bool(np.array_equal(idx[active],hi[active]))
                    herror,hcos=(error,cos) if same else output_metrics(hi)
                    hspent=float(rb[active,hi[active]].sum())
                    hardware_rows.append({'layer':layer,'bits':bit,'method':method,'window_start':start,'tokens':len(x),'strategy':strategy,
                        'uniform_rank':uniform_rank,'factor_bytes':hspent,'uniform_budget_bytes':budget,'byte_delta':hspent-budget,
                        'relative_error':herror,'cosine':hcos,'lambda0_calibration':hardware_lam0,
                        'lambda_before':hardware_lam if strategy=='gate_weighted_causal' else '',
                        'warmup':update_count<8,'same_rank_vector_as_float':same,
                        'changed_active_experts':int(np.count_nonzero(idx[active]!=hi[active])),
                        'rank_vector_sha256':hashlib.sha256(hi.astype('<u1').tobytes()).hexdigest(),
                        'scope':'actual_real_FFN_BF16_RNE_projection_energy_LUT_uint32_bytes; commonexpert_rank; sharedfull; routedonlylambda'})
                    vector_rows.append({'layer':layer,'bits':bit,'method':method,'rank_lanes':lanes,'uniform_rank':uniform_rank,'window_start':start,
                        'strategy':strategy,'same':same,'active_experts':len(active),'changed_active_experts':int(np.count_nonzero(idx[active]!=hi[active])),
                        'float_rank_vector_sha256':hashlib.sha256(idx.astype('<u1').tobytes()).hexdigest(),
                        'bf16_rank_vector_sha256':hashlib.sha256(hi.astype('<u1').tobytes()).hexdigest()})
            causal_bytes=float(rb[active,causal[active]].sum())
            lam*=float(np.exp(np.clip(0.5*(causal_bytes/max(1,budget)-1),-20,20)))
            if lam<lam0/100 or lam>lam0*100:lam=lam0
            if HARDWARE_BF16_POLICY:
                hidx=hardware_policies['gate_weighted_causal'];hbytes=float(rb[active,hidx[active]].sum())
                hardware_lam*=float(np.exp(np.clip(.5*(hbytes/max(1,budget)-1),-20,20)))
                if hardware_lam<hardware_lam0/100 or hardware_lam>hardware_lam0*100:hardware_lam=hardware_lam0
            previous_budget=budget;update_count+=1
        (outdir/f'lambda_state_l{layer}_w{bit}_{method}_uniform{uniform_rank}.json').write_text(json.dumps({
            'lambda0':lam0,'lambda_next':lam,'windows_observed':update_count,'lambda_eta':0.5,
            'uniform_rank':uniform_rank,'budget_last':previous_budget,'origin':'causal numerical-validation replay; never used for frozen calibration constants'},indent=2)+'\n')
        if HARDWARE_BF16_POLICY:
            (outdir/f'lambda_state_bf16_l{layer}_w{bit}_{method}_uniform{uniform_rank}.json').write_text(json.dumps({
                'lambda0':hardware_lam0,'lambda_next':hardware_lam,'windows_observed':update_count,'uniform_rank':uniform_rank,
                'budget_last':previous_budget,'lambda0_origin':'physically BF16-energy development-only actual-window byte fit',
                'energy_storage':'each projection value BF16 RNE then sum; bytecost uint32; common expert rank',
                'origin':'causal numeric validation; never used to fit frozen calibration'},indent=2)+'\n')
    if HARDWARE_BF16_POLICY:
        write_csv(outdir/f'q3_hardware_bf16_l{layer}_w{bit}_{method}_L{lanes}.csv',[dict(r,rank_lanes=lanes) for r in hardware_rows])
        write_csv(outdir/f'q3_rank_vector_comparison_l{layer}_w{bit}_{method}_L{lanes}.csv',vector_rows)
    return rows


def source_statistics(X,method):
    XX=np.asarray(X,np.float64);K=XX.shape[1];eps=1e-6
    if method in ('l2qer','qera_approx'):
        scale=np.mean(np.abs(XX),axis=0) if method=='l2qer' else np.sqrt(np.mean(XX*XX,axis=0))
        return np.maximum(scale,eps*max(scale.max(),eps))
    if method=='qera_exact':
        R=XX.T@XX/XX.shape[0];R+=eps*np.trace(R)/K*np.eye(K)
        values,Q=np.linalg.eigh(R);values=np.maximum(values,eps*values.max())
        return (Q*np.sqrt(values))@Q.T,(Q/np.sqrt(values))@Q.T
    raise ValueError(method)


def cached_source_factors(W,Wq,X,r,method,cache,source_key):
    """Same closed forms as ref_numerics; source statistics reused across W bits/Gate/Up/L."""
    if method=='lqer':return rn.lowrank_factors(W,Wq,X,r,method)
    E=(np.asarray(W,np.float64)-np.asarray(Wq,np.float64)).T;key=(source_key,method)
    if key not in cache:cache[key]=source_statistics(X,method)
    if method=='qera_exact':
        Rh,Rih=cache[key];U,s,B=rn._svd_factors(Rh@E,r);return Rih@U,B,s
    scale=cache[key];U,s,B=rn._svd_factors(scale[:,None]*E,r);return U/scale[:,None],B,s


def parallel_factor_bases(ws,deq,problems,cal,pooled_down,method,max_rank,capacity_lanes,cache,workers=8):
    """Deterministic ordered expert results; one worker owns all three projections.

    BLAS2×8workers bounds basis construction to16cores. Common pooled source
    statistics are completed outside the pool. Remaining source keys belong to
    one expert worker, so Gate/Up reuse has no concurrent writer.
    """
    if method!='lqer' and any(len(problems[e]['xc'])<32 for e in ws):
        for kind,X in [('gu',cal['x']),('d',pooled_down)]:
            key=(('pooled',kind),method)
            if key not in cache:cache[key]=source_statistics(X,method)
    def worker(e):
        factors={};p=problems[e]
        for k,W in ws[e].items():
            xc=p['zc'] if k=='d' else p['xc'];cold=len(xc)<32
            if cold:xc=pooled_down if k=='d' and e!=-1 else problems[-1]['zc'] if k=='d' else cal['x']
            key=(e if not cold else 'pooled','d' if k=='d' else 'gu')
            factors[k]=cached_source_factors(W,deq[e][k],xc,min(max_rank,capacity_lanes*bv.nseg(W.shape[1])),method,cache,key)
        return e,factors
    result={}
    with threadpool_limits(limits=BLAS_THREADS),ThreadPoolExecutor(max_workers=workers) as pool:
        for e,factors in pool.map(worker,list(ws)):
            result[e]=factors
            if e%8==0:print(f'parallel {method} completed expert{e}',flush=True)
    return result


def parallel_candidate(layer,bit,method,L,rank,af,bf,ws,deq,factors,problems,gates,x64,
                       projection_original,projection_quantized,gu_parts,projection_cache,
                       output_cache,rtn_outputs,signatures,workers=8):
    """One expert owns its caches; ordered collection preserves the serial merge.

    Each worker computes its unchanged three projections and FFN using BLAS2.
    Shared dictionaries have disjoint expert keys: no two workers read/write the
    same cache entry, and the caller never overlaps candidate invocations.
    """
    def worker(e):
        w=ws[e];fs={};rs=[];local=[];factor_bytes=0
        for k in ('g','u','d'):
            W=w[k];K,N=W.shape[1],W.shape[0];r=0 if method=='rtn' else min(rank,L*bv.nseg(K));rs.append(r)
            if r:
                A,B,_=factors[e][k];A=factor_a(A[:,:r],af);segs=rn.rank_segments(r,K,L)
                B=rn.quantize_b_per_segment(B[:r,:],segs,bf)
            else:A=np.zeros((K,0),np.float32);B=np.zeros((0,N),np.float32)
            fs[k]=(A,B)
            P=bv.plan_projection(k,K,N,r,bv.Fmt('q',bit,bf,L,af))
            factor_bytes+=P.a_bytes+P.b_sep_bytes+P.main_bytes-bv.plan_projection(k,K,N,0,bv.Fmt('q',bit)).main_bytes
            if len(problems[e]['xe']):
                key=(e,k,L,r,af,bf,method) if r else (e,k,0,0,'rtn','rtn','rtn')
                if key not in projection_cache:
                    yp=projection_quantized[e][k]+(x64[e][k]@A.astype(np.float64))@B.astype(np.float64)
                    base=projection_original[e][k];projection_cache[key]=(rn.rel_err(yp,base),cosine(yp,base))
                error,cos=projection_cache[key]
                local.append({'scope':'projection','layer':layer,'rank_lanes':L,'expert':e,'projection':k,'bits':bit,
                    'method':method,'rank':r,'factor_a':af,'factor_b':bf,'relative_error':error,'cosine':cos,
                    'factor_bytes':P.a_bytes+P.b_sep_bytes})
        signature=tuple(rs);key=(e,L,signature,af,bf)
        if not any(signature) and e in rtn_outputs:y=rtn_outputs[e]
        elif key in output_cache:y=output_cache[key]
        else:
            y=ffn_reuse_gate_up(problems[e]['xe'],deq[e],fs,gates[e],L,gu_parts[e])
            if not any(signature):rtn_outputs[e]=y
            elif signatures[e][signature]>1:output_cache[key]=y
        if len(problems[e]['xe']):
            local.append({'scope':'expert','layer':layer,'rank_lanes':L,'expert':e,'projection':'ffn','bits':bit,
                'method':method,'rank':rank,'factor_a':af,'factor_b':bf,'relative_error':rn.rel_err(y,problems[e]['gold']),
                'cosine':cosine(y,problems[e]['gold']),'factor_bytes':0})
        return e,y,local,factor_bytes
    outputs={};rows=[];total=0
    with threadpool_limits(limits=BLAS_THREADS),ThreadPoolExecutor(max_workers=workers) as pool:
        for e,y,local,bytes_ in pool.map(worker,list(ws)):
            outputs[e]=y;rows.extend(local);total+=bytes_
    return outputs,rows,total


def main():
    global BLAS_THREADS,HARDWARE_BF16_POLICY
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--calibration',type=Path,required=True);ap.add_argument('--evaluation',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--allow-incomplete',action='store_true');ap.add_argument('--expected-tensors',type=Path)
    ap.add_argument('--bits',default='4,3');ap.add_argument('--ranks',default='0,8,16,24,32,48,64');ap.add_argument('--methods',default='rtn,qera_approx,qera_exact,lqer,l2qer')
    ap.add_argument('--resume-q1',type=Path,help='reuse complete Q1 candidate rows after a Q3-only calibration fix; requires same captures/checkpoint')
    ap.add_argument('--resume-q3',type=Path,help='reuse only complete corrected-Q3 groups with an actual-window-lambda provenance receipt')
    ap.add_argument('--factor-a',default='mxint4,mxint8s,bf16');ap.add_argument('--factor-b',default='bf16,mxint8');ap.add_argument('--rank-lanes',default='8',help='8, 16, or 8,16; shared SVD basis for multiple modes');ap.add_argument('--factor-workers',type=int,default=8)
    ap.add_argument('--layers',default='2,13,26',help='Independent full-candidate layer shard; all captures are still validated against Q0')
    ap.add_argument('--blas-threads',type=int,default=2,help='Per-expert BLAS concurrency; does not alter mathematical algorithms or samples')
    ap.add_argument('--supplemental-b4',action='store_true',help='Separate real default-rank MXINT4-B checks; never added to the required Q1 matrix')
    ap.add_argument('--supplemental-only',action='store_true',help='Only the explicit MXINT4-B diagnostic; cannot freeze Q1')
    ap.add_argument('--q3-bf16-hardware',action='store_true',help='Separate actual-FFN hardware-BF16 energy-LUT policies using resident cached outputs')
    ap.add_argument('--q3-only',action='store_true',help='Q3 software/hardware supplement only; never re-evaluate or freeze Q1')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);lane_modes=list(map(int,str(args.rank_lanes).split(',')))
    BLAS_THREADS=args.blas_threads
    HARDWARE_BF16_POLICY=args.q3_bf16_hardware
    if BLAS_THREADS<1:raise ValueError('BLAS threads must be positive')
    if any(L not in (8,16) for L in lane_modes):raise ValueError('rank lanes must be8 or16')
    cm,cal,ch,cn=load_capture(args.calibration);em,ev,eh,en=load_capture(args.evaluation)
    if ch & eh:raise ValueError('calibration/eval request hashes overlap')
    model_config_hash=sha(args.model/'config.json')
    for manifest in (cm,em):
        if manifest.get('model_config_sha256') and manifest['model_config_sha256']!=model_config_hash:
            raise ValueError('capture model config does not match evaluated checkpoint')
    complete=cn>=32768 and en>=8192 and set(cal)=={2,13,26} and set(ev)=={2,13,26}
    if not complete and not args.allow_incomplete:raise ValueError('Q0 requires >=32768 calibration + >=8192 eval tokens at all three layers')
    selected_layers=list(map(int,args.layers.split(',')))
    if len(set(selected_layers))!=len(selected_layers) or not set(selected_layers)<=cal.keys()&ev.keys():raise ValueError('layer shard contains duplicate or uncaptured layers')
    expected=json.loads(args.expected_tensors.read_text())['tensor_provenance'] if args.expected_tensors else None
    cp=Checkpoint(args.model,expected);rows=[];coldrows=[];rankrows=[];q3actual=[];q2rows=[];supplemental=[];bits=list(map(int,args.bits.split(',')));ranks=list(map(int,args.ranks.split(',')))
    completed=set()
    if args.resume_q1:
        receipt=json.loads(args.resume_q1.with_suffix('.receipt.json').read_text())
        for key,path in [('q1_sha256',args.resume_q1),('calibration_manifest_sha256',args.calibration),
                         ('validation_manifest_sha256',args.evaluation),('model_config_sha256',args.model/'config.json')]:
            if receipt.get(key)!=sha(path):raise ValueError(f'Q1 resume provenance mismatch: {key}')
        if args.expected_tensors and receipt.get('inventory_sha256')!=sha(args.expected_tensors):
            raise ValueError('Q1 resume tensor inventory mismatch')
        existing=list(csv.DictReader(args.resume_q1.open()))
        def signature(row):return tuple(row[k] for k in ('layer','rank_lanes','bits','method','rank','factor_a','factor_b'))
        for row in existing:
            # Projection rows record clipped rank rather than the requested
            # expert rank. Retain the whole complete CSV prefix; an interrupted
            # candidate can only occur before its atomic CSV checkpoint.
            for k in ('layer','rank_lanes','bits','rank','factor_bytes'):row[k]=int(row[k])
            if row['expert']!='all':row['expert']=int(row['expert'])
            for k in ('relative_error','cosine'):row[k]=float(row[k])
            rows.append(row)
            if row['scope']=='layer':completed.add(signature(row))
        print(f'resumed {len(rows)} exact Q1 rows from {args.resume_q1}; Q3 is recomputed',flush=True)
    completed_q3=set()
    if args.resume_q3:
        receipt=json.loads(args.resume_q3.with_suffix('.receipt.json').read_text())
        if receipt.get('policy_lambda_fit_version')!='actual_window_bytes_v1':raise ValueError('Q3 resume uses obsolete lambda calibration')
        for key,path in [('q3_sha256',args.resume_q3),('calibration_manifest_sha256',args.calibration),
                         ('validation_manifest_sha256',args.evaluation),('model_config_sha256',args.model/'config.json')]:
            if receipt.get(key)!=sha(path):raise ValueError(f'Q3 resume provenance mismatch: {key}')
        groups={}
        for row in csv.DictReader(args.resume_q3.open()):
            for k in ('layer','rank_lanes','bits','window_start','tokens','uniform_rank'):row[k]=int(row[k])
            for k in ('factor_bytes','uniform_budget_bytes','byte_delta','relative_error','cosine','lambda0_calibration'):row[k]=float(row[k])
            if row['lambda_before']:row['lambda_before']=float(row['lambda_before'])
            row['warmup']=row['warmup']=='True'
            key=(row['layer'],row['bits'],row['method'],row['rank_lanes']);groups.setdefault(key,[]).append(row)
        for key,rr in groups.items():
            wanted={(u,start,s) for u in (16,32) for start in range(0,len(ev[key[0]]['x']),16)
                    for s in ('uniform','frequency_static','gate_weighted_budget_oracle','gate_weighted_causal')}
            actual=[(r['uniform_rank'],r['window_start'],r['strategy']) for r in rr]
            if set(actual)==wanted and len(actual)==len(wanted):completed_q3.add(key);q3actual.extend(rr)
        print(f'resumed{len(completed_q3)} complete corrected-Q3 groups ({len(q3actual)} rows)',flush=True)
    for layer in sorted(selected_layers,key=lambda l:(l!=13,l)):
        print(f'layer{layer} load original problems',flush=True);layer_start=time.monotonic()
        p,ws,gold,ids,gates,cold=layer_problem(cp,layer,cal[layer],ev[layer]);coldrows+=cold
        pooled_down=np.concatenate([v['zc'] for e,v in p.items() if e!=-1 and len(v['zc'])],axis=0)
        for L in lane_modes:
            q2rows += [dict(row,rank_lanes=L) for row in evaluate_cold_statistics(layer,p,ws,cal[layer],pooled_down,L)]
        write_csv(args.output/'q2_statistics_comparison.csv',q2rows)
        gold_layer=rn.combine_hw(gold,ids,list(gold),len(ev[layer]['x']));statistics_cache={}
        x64={e:{'g':p[e]['xe'].astype(np.float64),'u':p[e]['xe'].astype(np.float64),'d':p[e]['ze'].astype(np.float64)} for e in ws}
        projection_original={e:{k:x64[e][k]@W.T.astype(np.float64) for k,W in w.items()} for e,w in ws.items()}
        for bit in ([4] if args.supplemental_only else (bits if args.q3_only else bits+[8])):
            deq={e:{k:rn.mx_quantize(v,bit)[2] for k,v in w.items()} for e,w in ws.items()}
            projection_quantized={e:{k:x64[e][k]@W.T.astype(np.float64) for k,W in w.items()} for e,w in deq.items()}
            gu_parts={e:gate_up_parts(p[e]['xe'],w) for e,w in deq.items()}
            projection_cache={};rtn_outputs={}
            for method in (['rtn','qera_approx'] if args.supplemental_only else (['rtn'] if bit==8 else args.methods.split(','))):
                if args.q3_only and method=='rtn':continue
                # A verified resumed method can avoid repeating its exact SVD.
                # Missing requested points/groups still take the unchanged path.
                requested_keys={(layer,L,bit,method,r,af,bf) for L in lane_modes
                    for r in ([0] if method=='rtn' else ranks+([96] if L==16 and 96 not in ranks else []))
                    for af in (['mxint4'] if method=='rtn' else args.factor_a.split(','))
                    for bf in (['bf16'] if method=='rtn' else args.factor_b.split(','))}
                complete_method=bool(requested_keys) and requested_keys<=completed
                complete_software=method=='rtn' or all((layer,bit,method,L) in completed_q3 for L in lane_modes)
                complete_hardware=not HARDWARE_BF16_POLICY or method=='rtn' or all(
                    hardware_q3_group_complete(args.output/f'L{L}',layer,bit,method,L,len(ev[layer]['x'])) for L in lane_modes)
                # Supplemented RTN/QERA defaults have their own receipt and are
                # not silently skipped by this main-matrix recovery mechanism.
                needs_supplement=(args.supplemental_b4 or args.supplemental_only) and bit==4 and method in ('rtn','qera_approx')
                if complete_method and complete_software and complete_hardware and not needs_supplement and not args.q3_only:
                    print(f'layer{layer} W{bit} {method}: exact complete checkpoint reused; no SVD repeated',flush=True)
                    continue
                print(f'layer{layer} W{bit} {method} basis start',flush=True)
                output_cache={}
                # SVD bases can be sliced into nested rank prefixes; computed once per expert/method.
                factors={}
                if method!='rtn':
                    factors=parallel_factor_bases(ws,deq,p,cal[layer],pooled_down,method,
                        max(ranks+[96] if 16 in lane_modes else ranks),max(lane_modes),statistics_cache,args.factor_workers)
                for L in lane_modes:
                    lane_output=args.output/f'L{L}';lane_output.mkdir(exist_ok=True)
                    if method!='rtn':
                        table,*_=rank_calibration(layer,bit,method,ws,factors,cal[layer],L)
                        (lane_output/f'rank_table_l{layer}_w{bit}_{method}.json').write_text(json.dumps(table,indent=2)+'\n')
                    requested=[0] if method=='rtn' else ranks+([96] if L==16 and 96 not in ranks else [])
                    signatures={e:Counter(tuple(min(rr,L*bv.nseg(w[k].shape[1])) for k in ('g','u','d')) for rr in requested) for e,w in ws.items()}
                    if (args.supplemental_b4 or args.supplemental_only) and bit==4 and method in ('rtn','qera_approx'):
                        diagnostic_rank=0 if method=='rtn' else (48 if L==8 else 96)
                        outputs,local_rows,factor_bytes=parallel_candidate(layer,bit,method,L,diagnostic_rank,'mxint4','mxint4',ws,deq,factors,p,gates,x64,
                            projection_original,projection_quantized,gu_parts,projection_cache,output_cache,rtn_outputs,signatures,args.factor_workers)
                        layer_out=rn.combine_hw(outputs,ids,list(outputs),len(ev[layer]['x']))
                        supplemental.extend(local_rows)
                        supplemental.append({'scope':'layer','layer':layer,'rank_lanes':L,'expert':'all','projection':'moe','bits':bit,'method':method,
                            'rank':diagnostic_rank,'factor_a':'mxint4','factor_b':'mxint4','relative_error':rn.rel_err(layer_out,gold_layer),
                            'cosine':cosine(layer_out,gold_layer),'factor_bytes':factor_bytes})
                        write_csv(args.output/'supplemental_mxint4_b_default.csv',supplemental)
                    if args.supplemental_only:continue
                    for rank in ([] if args.q3_only else requested):
                        for af in (['mxint4'] if method=='rtn' else args.factor_a.split(',')):
                            for bf in (['bf16'] if method=='rtn' else args.factor_b.split(',')):
                                candidate=(layer,L,bit,method,rank,af,bf)
                                if candidate in completed:
                                    continue
                                outputs,local_rows,factor_bytes=parallel_candidate(layer,bit,method,L,rank,af,bf,ws,deq,factors,p,gates,x64,
                                    projection_original,projection_quantized,gu_parts,projection_cache,output_cache,rtn_outputs,signatures,args.factor_workers)
                                rows.extend(local_rows)
                                layer_out=rn.combine_hw(outputs,ids,list(outputs),len(ev[layer]['x']))
                                rows.append({'scope':'layer','layer':layer,'rank_lanes':L,'expert':'all','projection':'moe','bits':bit,'method':method,'rank':rank,'factor_a':af,'factor_b':bf,'relative_error':rn.rel_err(layer_out,gold_layer),'cosine':cosine(layer_out,gold_layer),'factor_bytes':factor_bytes})
                                write_csv(args.output/'q1_metrics.csv',rows)
                                print(f'layer{layer} W{bit} {method} L{L} r{rank} A{af} B{bf} layererror{rows[-1]["relative_error"]:.6g}',flush=True)
                    if method!='rtn':
                        if (layer,bit,method,L) not in completed_q3 or HARDWARE_BF16_POLICY:
                            q3actual += [dict(row,rank_lanes=L) for row in evaluate_rank_policies(layer,bit,method,p,ws,factors,cal[layer],ev[layer],L,lane_output,quantized_weights=deq,gu_parts=gu_parts,
                                include_software=(layer,bit,method,L) not in completed_q3)]
                        write_csv(args.output/'q3_actual_ffn.csv',q3actual)
                        tb,choices,te,rb,capidx,freq=rank_calibration(layer,bit,method,ws,factors,cal[layer],L)
                        weights=np.zeros(64)
                        for rr,gg in zip(cal[layer]['routes'],cal[layer]['gates']):
                            for ee,g in zip(rr,gg):weights[ee]+=float(g)**2
                        idx=rn.allocate_ranks(weights,te,rb,tb['lambda0'],capidx)
                        rankrows.append({'layer':layer,'rank_lanes':L,'bits':bit,'method':method,'budget_bytes':tb['uniform_factor_budget_bytes'],
                            'allocated_bytes':float(rb[np.arange(64),idx].sum()),'uniform_objective':float((weights*te[:,4]).sum()),
                            'gate_weighted_objective':float((weights*te[np.arange(64),idx]).sum()),'lambda0':tb['lambda0'],
                            'scope':'surrogate_calibration_tail_energy_not_layer_error'})
    if args.supplemental_only:
        (args.output/'supplemental_provenance.json').write_text(json.dumps({'schema':'plena_v3_mxint4_b_default_diagnostic_v1',
            'layers':selected_layers,'q0_complete':complete,'calibration_tokens':cn,'evaluation_tokens':en,
            'calibration_manifest_sha256':sha(args.calibration),'validation_manifest_sha256':sha(args.evaluation),
            'model_config_sha256':model_config_hash,'inventory_sha256':sha(args.expected_tensors) if args.expected_tensors else None,
            'accuracy_scope':'default-rank RTN/lanes only, full-Z hardware numerical order; separate from required Q1 matrix',
            'can_freeze':False},indent=2)+'\n')
        return
    if args.q3_only:
        write_csv(args.output/'q3_rank_proxy.csv',rankrows)
        (args.output/'q3_only_provenance.json').write_text(json.dumps({'schema':'plena_v3_q3_only_provenance_v1','q0_complete':complete,
            'layers':selected_layers,'calibration_tokens':cn,'evaluation_tokens':en,'calibration_manifest_sha256':sha(args.calibration),
            'validation_manifest_sha256':sha(args.evaluation),'model_config_sha256':model_config_hash,
            'inventory_sha256':sha(args.expected_tensors) if args.expected_tensors else None,'hardware_bf16':HARDWARE_BF16_POLICY,
            'source':'real checkpoint/captured rows; complete Q3 matrices; no Q1 candidates reevaluated','can_freeze':False},indent=2)+'\n')
        return
    write_csv(args.output/'q1_metrics.csv',rows);write_csv(args.output/'q2_cold_counts.csv',coldrows);write_csv(args.output/'q3_rank_proxy.csv',rankrows)
    coverage=q1_coverage(rows,selected_layers,bits,ranks,args.methods.split(','),lane_modes,args.factor_a.split(','),args.factor_b.split(','))
    (args.output/'q1_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    if not coverage['complete']:raise AssertionError('required Q1 matrix is incomplete; inspect q1_coverage.json')
    (args.output/'q2_status.json').write_text(json.dumps({'cold_experts':sum(int(r['cold']) for r in coldrows),
        'measured_cold_comparisons':sum(r['status']=='measured_real' for r in q2rows),
        'no_samples_expert_own':sum(r['status']=='no_expert_samples' for r in q2rows),
        'status':'measured' if q2rows else 'no_calibration_expert_below_32_tokens_in_this_capture'},indent=2)+'\n')
    layer_rows=[r for r in rows if r['scope']=='layer'];rtn={(r['layer'],r['rank_lanes']):r['relative_error'] for r in layer_rows if r['bits']==4 and r['method']=='rtn'}
    candidates={}
    for row in layer_rows:
        if row['method']=='rtn' or row['bits']==8:continue
        key=(row['bits'],row['method'],row['rank'],row['factor_a'],row['factor_b'],row['rank_lanes'])
        candidates.setdefault(key,[]).append(row)
    passing=[]
    for key,rr in candidates.items():
        if {r['layer'] for r in rr}!={2,13,26}:continue
        if all(r['relative_error']<=0.5*rtn.get((r['layer'],r['rank_lanes']),0) and r['cosine']>=0.999 for r in rr):
            main_bytes=sum(bv.plan_projection(k,W.shape[1],W.shape[0],0,bv.Fmt('q',key[0],'bf16',key[5],'mxint4')).main_bytes
                           for w in ws.values() for k,W in w.items())
            passing.append({'bits':key[0],'method':key[1],'rank':key[2],'factor_a':key[3],'factor_b':key[4],
                            'rank_lanes':key[5],'factor_bytes_all_experts_max_layer':max(r['factor_bytes'] for r in rr),
                            'total_physical_weight_bytes':main_bytes+max(r['factor_bytes'] for r in rr),
                            'precision':'P1' if key[3]=='bf16' else 'P2','layers':[2,13,26]})
    passing.sort(key=lambda v:v['total_physical_weight_bytes'])
    can_freeze=complete and set(selected_layers)=={2,13,26} and bool(passing)
    (args.output/'freeze_candidates.json').write_text(json.dumps({'q0_complete':complete,'accuracy_target':{'error_vs_mxint4_rtn_max':0.5,'cosine_min':0.999},
        'passing_candidates':passing,'can_freeze':can_freeze,'scope':'same configuration across three layers; development request subdivisions only'},indent=2)+'\n')
    result={'schema':'plena_v3_quant_report_v1','q0_complete':complete,'calibration_tokens':cn,'evaluation_tokens':en,
            'request_hash_disjoint':True,'layers':sorted(cal.keys()&ev.keys()),'source':'real BF16 weights and captured real activations',
            'rank_lane_modes':lane_modes,'factor_workers':args.factor_workers,'basis_blas_threads':BLAS_THREADS,
            'evaluated_layers':selected_layers,'full_accuracy_matrix_complete':set(selected_layers)=={2,13,26},
            'resumed_q1_from':str(args.resume_q1) if args.resume_q1 else None,
            'resumed_q1_sha256':sha(args.resume_q1) if args.resume_q1 else None,
            'layer_baseline':'original BF16 checkpoint weights under the hardware FP32/BF16 storage contract; not end-to-end model perplexity',
            'tensor_provenance':cp.records,'precision_can_be_frozen':can_freeze,'q4_status':'not_run_no_gpu' if not torch.cuda.is_available() else 'required_separate_script'}
    (args.output/'q0_provenance.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
