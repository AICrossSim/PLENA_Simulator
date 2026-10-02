#!/usr/bin/env python3
"""Materialize real default compiler payloads and replay decoded bytes only.

This validates packing/ABI, not quantization accuracy acceptance. Factor bases
are computed from real development calibration and original BF16 checkpoint
weights. No synthetic factors or original-W substitute enter the decoded replay.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import evaluate as q


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--evaluation',type=Path,required=True);ap.add_argument('--inventory',type=Path,required=True)
    ap.add_argument('--compiler',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--layer',type=int,default=13)
    args=ap.parse_args();sys.path.insert(0,str(args.compiler));import compiler_v3 as c
    _,cal,ch,cn=q.load_capture(args.calibration);_,ev,eh,en=q.load_capture(args.evaluation)
    if ch&eh:raise ValueError('numeric calibration/validation overlap')
    inventory=json.loads(args.inventory.read_text())['tensor_provenance'];cp=q.Checkpoint(args.model,inventory)
    args.output.mkdir(exist_ok=True,parents=True);experts=[]
    for e in (0,-1):
        data=cal[args.layer];vd=ev[args.layer]
        ic=np.arange(len(data['x'])) if e==-1 else np.where(np.any(data['routes']==e,axis=1))[0]
        ie=np.arange(len(vd['x'])) if e==-1 else np.where(np.any(vd['routes']==e,axis=1))[0]
        xc=q.rn.bf16_round(data['x'][ic]);xe=q.rn.bf16_round(vd['x'][ie[:16]])
        gc=np.ones(len(ic),np.float32) if e==-1 else np.array([data['gates'][i,list(data['routes'][i]).index(e)] for i in ic],np.float32)
        ge=np.ones(len(xe),np.float32) if e==-1 else np.array([vd['gates'][i,list(vd['routes'][i]).index(e)] for i in ie[:16]],np.float32)
        ws={k:cp.weight(args.layer,e,name) for k,name in [('g','gate'),('u','up'),('d','down')]}
        deq={k:q.rn.mx_quantize(W,4)[2] for k,W in ws.items()};zc=q.original_z_hw(xc,ws,gc)
        ranks=(32,32,48) if e==-1 else (32,32,24);decoded={};factors={};direct={};items=[]
        for (k,W),r in zip(ws.items(),ranks):
            A,B,_=q.rn.lowrank_factors(W,deq[k],zc if k=='d' else xc,r,'qera_approx')
            fmt=q.bv.FMT_V3;raw,meta=c.pack_projection(W,A,B,fmt,name=k);Wb,Ab,Bb=c.unpack_projection(raw,meta,fmt)
            ar=q.factor_a(A,'mxint4');br=q.rn.quantize_b_per_segment(B,q.rn.rank_segments(r,W.shape[1],8),'bf16')
            if not (np.array_equal(Wb,deq[k]) and np.array_equal(Ab,ar) and np.array_equal(Bb,br)):
                raise AssertionError('actual payload decode differs from exact frozen quantizer contract')
            stem=f'layer{args.layer}_expert{e}_{k}';p=args.output/f'{stem}.bin';p.write_bytes(raw)
            (args.output/f'{stem}.json').write_text(json.dumps(meta,indent=2)+'\n')
            decoded[k]=Wb;factors[k]=(Ab,Bb);direct[k]=(ar,br)
            items.append({'projection':k,'shape_nk':list(W.shape),'rank':r,'payload_bytes':len(raw),'payload_sha256':q.sha(p),
                'all_decoded_elements_bit_exact':True,'file':str(p.resolve())})
            print(f'real layer{args.layer} expert{e} {k} packed {len(raw)}B',flush=True)
        actual=q.rn.expert_ffn_hw(xe,decoded['g'],decoded['u'],decoded['d'],factors,ge,8)
        gold=q.rn.expert_ffn_hw(xe,deq['g'],deq['u'],deq['d'],direct,ge,8)
        if not np.array_equal(actual,gold):raise AssertionError('decoded-byte FFN replay is not bit-exact')
        total=sum(x['payload_bytes'] for x in items);expected=9889920 if e==-1 else 4970112
        if total!=expected:raise AssertionError(f'actual full expert payload bytes {total} != {expected}')
        experts.append({'expert':e,'is_shared':e==-1,'calibration_tokens':len(ic),'validation_replay_tokens':len(xe),
            'payload_bytes':total,'expected_default_bytes':expected,'projections':items,'decoded_byte_ffn_bit_exact':True})
    report={'schema':'plena_v3_real_compiler_payload_receipt_v1','layer':args.layer,'experts':experts,'source':'real BF16 checkpoint weights and actual captured development activations',
        'factor_method':'QERAapprox using expert-owned calibration; Down statistics from original BF16 gate-folded Z',
        'format':{'main':'MXINT4','A':'MXINT4','B':'BF16','L':8},'q0_tokens':[cn,en],
        'calibration_manifest_sha256':q.sha(args.calibration),'validation_manifest_sha256':q.sha(args.evaluation),
        'inventory_sha256':q.sha(args.inventory),'compiler_source_sha256':q.sha(args.compiler/'compiler_v3.py'),'verified_tensor_provenance':cp.records,
        'accuracy_acceptance_claim':False,'scope':'ABI/packing and decoded-byte FFN validation only; provisional timing format already fails Q1 default accuracy target'}
    (args.output/'packing_receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('actual routed/shared payload and FFN byte replay complete',flush=True)


if __name__=='__main__':main()
