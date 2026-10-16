"""Representative real CUDA kernel traces; not hardware utilization counters."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import torch
from .common import write_csv,write_json
from .precision import PRECISIONS,layer_weights,ExpertWeights,Runner

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);rows=[];inventory=[]
    with torch.inference_mode():
        for precision in PRECISIONS:
            rp,sp=layer_weights(a.model,13);rw,sw=ExpertWeights(rp,precision),ExpertWeights(sp,precision)
            del rp,sp
            for batch in (2,16,128,256):
                with np.load(a.capture/f'bfcl_b{batch}_c0.npz') as z:
                    x=torch.from_numpy(z['x_l13'][0].copy()).view(torch.bfloat16).cuda();ids=z['ids_l13'][0];scores=z['scores_l13'][0]
                runner=Runner(rw,sw,x,ids,scores)
                for _ in range(10):runner.run('grouped')
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):runner.run('grouped')
                graph.replay();torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                    for _ in range(5):graph.replay()
                    torch.cuda.synchronize()
                path=a.out/f'{precision}_b{batch}_l13.json';prof.export_chrome_trace(str(path))
                trace=json.loads(path.read_text());kernels=[r for r in trace['traceEvents'] if r.get('cat')=='kernel' and r.get('ph')=='X']
                if not kernels:raise ValueError('no real CUDA kernels recorded')
                for k in kernels:rows.append(dict(precision=precision,batch=batch,layer=13,kernel=k['name'],duration_us=k['dur'],stream=k.get('args',{}).get('stream','')))
                inventory.append(dict(precision=precision,batch=batch,layer=13,kernel_events=len(kernels),graph_replays=5,trace=path.name))
                print('KERNEL TRACE',precision,batch,len(kernels),flush=True)
                del graph,runner,x;gc.collect();torch.cuda.empty_cache()
            del rw,sw;gc.collect();torch.cuda.empty_cache()
    write_csv(a.out/'kernel_events.csv',rows);write_json(a.out/'COMPLETE.json',dict(points=inventory,
        scope='layer13, BFCL cohort0 decode step1; grouped CUDA Graph; profiler tracing overhead included, do not replace timing campaign',
        limitation='CUDA kernel names/durations only; no Tensor Core occupancy or physical HBM counters'))

if __name__=='__main__':main()
