"""Integration test: reused graphs must read new routes, maps, and activations."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from .precision import *
from .common import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);rows=[]
    with torch.inference_mode():
        rp,sp=layer_weights(a.model,1)
        for precision in PRECISIONS:
            rw,sw=ExpertWeights(rp,precision),ExpertWeights(sp,precision)
            for batch in (2,16):
                z=np.load(a.capture/f'bfcl_b{batch}_c0.npz');graphs={}
                for i,step in enumerate(z['profile_steps'][:2]):
                    x=torch.from_numpy(z['x_l1'][i].copy()).view(torch.bfloat16).cuda();ids=z['ids_l1'][step-1];sc=z['scores_l1'][step-1]
                    for path in ('grouped','split'):
                        r=Runner(rw,sw,x,ids,sc);ref=r.run(path).clone();key=(r.hot,r.cold,path)
                        if key not in graphs:graphs[key]=GraphBench(r,path,10)
                        bench=graphs[key];bench.update(x,ids,sc);timing,y=bench.measure()
                        error=error_metrics(y,ref)
                        if not error['finite'] or error['relative_l2']>.02:raise ValueError('graph reuse mismatch '+str(error))
                        rows.append(dict(precision=precision,batch=batch,step=int(step),path=path,**error));write_json(a.out,rows)
                del graphs
            print('PASS',precision,flush=True)
    print('GPU GRAPH TESTS PASS',len(rows),flush=True)
if __name__=='__main__':main()
