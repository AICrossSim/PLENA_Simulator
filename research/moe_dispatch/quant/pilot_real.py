#!/usr/bin/env python3
"""Historical layer1 real-weight/X16 smoke, explicitly not Q0--Q3 completion."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent));from evaluate import Checkpoint,cosine,write_csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v3_reference'));import ref_numerics as rn


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);start=time.monotonic();m=json.loads(a.manifest.read_text());xp=Path(m['x']['path'])
    raw=xp.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=m['x']['sha256']:raise ValueError('historical X hash mismatch')
    x=(np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(16,2048);rr=np.asarray(m['routes']);gg=np.asarray(m['route_weights'],np.float32)
    cp=Checkpoint(a.model,m['tensor_provenance']);experts=sorted(set(rr[8:].reshape(-1).tolist()))+[-1];ids={};gates={};gold={};qgold={};states={};rows=[]
    for e in experts:
        ic=np.arange(8) if e==-1 else np.where(np.any(rr[:8]==e,axis=1))[0];ie=np.arange(8) if e==-1 else np.where(np.any(rr[8:]==e,axis=1))[0]
        ids[e]=ie.tolist();gates[e]=np.ones(len(ie),np.float32) if e==-1 else np.array([gg[i+8,list(rr[i+8]).index(e)] for i in ie],np.float32)
        xc=x[ic] if len(ic) else x[:8];xe=x[8:][ie];w={k:cp.weight(1,e,p) for k,p in [('g','gate'),('u','up'),('d','down')]}
        zc=rn.bf16_round(rn.silu(xc@w['g'].T)*(xc@w['u'].T));qw={k:rn.mx_quantize(W,4)[2] for k,W in w.items()};fs={}
        for k in ('g','u','d'):
            rank=32 if k!='d' else (48 if e==-1 else 24);X=zc if k=='d' else xc
            A,B,_=rn.lowrank_factors(w[k],qw[k],X,rank,'qera_approx');A=rn.quantize_factor(A,'mxint4',0);B=rn.bf16_round(B);fs[k]=(A,B)
        gold[e]=rn.expert_ffn_hw(xe,w['g'],w['u'],w['d'],gate=gates[e]);qgold[e]=rn.expert_ffn_hw(xe,qw['g'],qw['u'],qw['d'],gate=gates[e])
        y,state=rn.expert_ffn_hw(xe,qw['g'],qw['u'],qw['d'],fs,gates[e],return_state=True);states[e]=y
        fp64=rn.expert_ffn(xe,qw['g'],qw['u'],qw['d'],fs,gates[e]);rows.append({'scope':'expert','layer':1,'expert':e,'calibration_routed_tokens':len(ic),'evaluation_routed_tokens':len(ie),'rtn_error':rn.rel_err(qgold[e],gold[e]),'qera_approx_error':rn.rel_err(y,gold[e]),'qera_cosine':cosine(y,gold[e]),'hardware_vs_fp64':rn.rel_err(y,fp64)})
    combined=rn.combine_hw(gold,ids,experts,8);rtn=rn.combine_hw(qgold,ids,experts,8);qera=rn.combine_hw(states,ids,experts,8)
    result={'scope':'historical_real_layer1_pilot_only','layer':1,'calibration_tokens':8,'evaluation_tokens':8,'checkpoint_weights_and_X_hash_verified':True,
            'rtn_layer_relative_error':rn.rel_err(rtn,combined),'qera_approx_layer_relative_error':rn.rel_err(qera,combined),'qera_cosine':cosine(qera,combined),
            'relative_error_ratio':rn.rel_err(qera,combined)/rn.rel_err(rtn,combined),'max_hardware_vs_fp64':max(r['hardware_vs_fp64'] for r in rows),
            'q0_complete':False,'q1_freeze_eligible':False,'reason':'layer1 historical8+8 distinct requests; required layers2/13/26 and32k+8ktokens not yet completed',
            'seconds':time.monotonic()-start,'tensor_provenance':cp.records}
    (a.output/'pilot_report.json').write_text(json.dumps(result,indent=2)+'\n');write_csv(a.output/'pilot_experts.csv',rows)
    print(json.dumps({k:v for k,v in result.items() if k!='tensor_provenance'}))

if __name__=='__main__':main()
