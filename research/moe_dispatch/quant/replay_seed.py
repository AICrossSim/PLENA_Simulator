#!/usr/bin/env python3
"""Generate hardware-order gold for a deterministic small numeric replay."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v3_reference'))
import ref_numerics as rn


def make_seed(seed=20261002,streamed=False):
    rng=np.random.default_rng(seed);T,d,I,r=3,544,160,8
    x=rn.bf16_round(rng.normal(size=(T,d)).astype(np.float32)*.1)
    ws={k:rn.mx_quantize(rng.normal(size=(I,d) if k!='d' else (d,I)).astype(np.float32)*.025,4)[2] for k in ('g','u','d')}
    fs={k:(rn.quantize_factor(rng.normal(size=(w.shape[1],r)).astype(np.float32)*.02,'mxint4',0),rn.bf16_round(rng.normal(size=(r,w.shape[0])).astype(np.float32)*.02)) for k,w in ws.items()}
    gate=rng.uniform(.1,.7,T).astype(np.float32);out,state=rn.expert_ffn_hw(x,ws['g'],ws['u'],ws['d'],fs,gate,L=4,z_mode='streamed' if streamed else 'full',return_state=True)
    fp64=rn.expert_ffn(x,ws['g'],ws['u'],ws['d'],fs,gate)
    return {'schema':'plena_moe_v3_numeric_replay_v1','origin':'synthetic_seeded_correctness_not_model_accuracy','seed':seed,'x':x.tolist(),
            'weights':{k:v.tolist() for k,v in ws.items()},'factors':{k:{'a':a.tolist(),'b':b.tolist()} for k,(a,b) in fs.items()},
            'L':4,'gate':gate.tolist(),'z_mode':'streamed' if streamed else 'full','output':out.tolist(),
            'state':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()},
            'algorithm_relative_error':rn.rel_err(out,fp64),'hardware_rtol':1e-5,'algorithm_rtol':5e-3}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--streamed',action='store_true');a=ap.parse_args()
    v=make_seed(streamed=a.streamed);assert v['algorithm_relative_error']<v['algorithm_rtol'];a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(v,sort_keys=True)+'\n')
    print(json.dumps({'output':str(a.output),'algorithm_relative_error':v['algorithm_relative_error']}))

if __name__=='__main__':main()
