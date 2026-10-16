"""Probe actual kernels and numeric error on a captured input, never fake timing."""
import argparse, gc, json, traceback
from pathlib import Path
import numpy as np
import torch
from .common import write_json
from .precision import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--shard',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--precisions',default=','.join(PRECISIONS));a=p.parse_args()
    torch.set_num_threads(4)
    z=np.load(a.shard);x=torch.from_numpy(z['x_l1'][0].copy()).view(torch.bfloat16).cuda()
    ids=z['ids_l1'][0];scores=z['scores_l1'][0];rows=[]
    with torch.inference_mode():
        rp,sp=layer_weights(a.model,1)
        base=Runner(ExpertWeights(rp,'bf16'),ExpertWeights(sp,'bf16'),x,ids,scores)
        ref=base.run('grouped').clone();del base
        for precision in a.precisions.split(','):
            try:
                runner=Runner(ExpertWeights(rp,precision),ExpertWeights(sp,precision),x,ids,scores)
                paths={}
                for path in ('grouped','split'):
                    timing,y=measure(runner,path)
                    err=error_metrics(y,ref)
                    paths[path]=dict(**err,mean_ms=sum(timing['e2e_ms'])/len(timing['e2e_ms']),
                        instrumentation_output_relative_l2=timing['instrumentation_output_relative_l2'])
                if precision=='int4_w4a16':
                    def dequant(w,s):
                        q=torch.stack((w&15,w>>4),dim=-1).flatten(-2).float()-8
                        return (q*s.float().repeat_interleave(128,dim=-1)).to(torch.bfloat16)
                    r1=dequant(runner.routed_weights.w1,runner.routed_weights.s1)
                    r2=dequant(runner.routed_weights.w2,runner.routed_weights.s2)
                    s1=dequant(runner.shared_weights.w1,runner.shared_weights.s1)
                    s2=dequant(runner.shared_weights.w2,runner.shared_weights.s2)
                    def ffn(inp,w1,w2):
                        g,u=torch.nn.functional.linear(inp,w1).chunk(2,-1)
                        z=(torch.nn.functional.silu(g.float())*u.float()).to(torch.bfloat16)
                        return torch.nn.functional.linear(z,w2)
                    numeric=ffn(x,s1[0],s2[0]).float()
                    for e in sorted(set(ids.reshape(-1).tolist())):
                        ts,slots=np.where(ids==e);t=torch.as_tensor(ts,device='cuda')
                        part=ffn(x[t],r1[e],r2[e]).float()*torch.as_tensor(scores[ts,slots],device='cuda')[:,None]
                        numeric.index_add_(0,t,part)
                    qerror=error_metrics(runner.run('grouped'),numeric)
                    if qerror['relative_l2']>.03:raise ValueError('packed W4 disagrees with dequantized independent reference '+str(qerror))
                    paths['dequantized_reference']=qerror
                row=dict(precision=precision,status='supported',paths=paths)
                del runner;gc.collect();torch.cuda.empty_cache()
            except Exception as e:
                row=dict(precision=precision,status='failed',error=repr(e),traceback=traceback.format_exc())
            rows.append(row);write_json(a.out,rows);print(json.dumps(row),flush=True)
if __name__=='__main__':main()
