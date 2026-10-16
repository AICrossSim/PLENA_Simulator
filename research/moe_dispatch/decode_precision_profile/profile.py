"""Layer-matched real-weight/real-activation post-router GPU replay."""
from __future__ import annotations
import argparse, gc, gzip, json, random, time
from pathlib import Path
import numpy as np
import torch
from .common import *
from .precision import *

def main():
    import triton,vllm
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--precision',choices=PRECISIONS,required=True);p.add_argument('--layers',default=','.join(map(str,LAYERS)))
    p.add_argument('--repeat',type=int,default=1);p.add_argument('--warmup',type=int,default=10);p.add_argument('--samples',type=int,default=50)
    p.add_argument('--max-groups',type=int);p.add_argument('--max-steps',type=int)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    manifest=json.loads((a.capture/'capture_manifest.json').read_text())
    if not manifest['complete']:raise ValueError('capture incomplete')
    if manifest['dtype']!='bf16':raise ValueError('primary precision comparison must use common BF16 routes and activations')
    groups=manifest['shards'][:a.max_groups];layers=list(map(int,a.layers.split(',')))
    steps=manifest['profile_steps'][:a.max_steps]
    torch.manual_seed(20261016)
    meta=dict(schema='real_decode_precision_profile_v1',precision=a.precision,repeat=a.repeat,
        gpu=torch.cuda.get_device_name(),device_properties=str(torch.cuda.get_device_properties(0)),torch=torch.__version__,triton=triton.__version__,vllm=vllm.__version__,
        capture_manifest_sha256=sha(a.capture/'capture_manifest.json'),layers=layers,steps=steps,
        groups=[g['id'] for g in groups],warmup=a.warmup,samples=a.samples,
        source_sha256={n:sha(Path(__file__).parent/n) for n in ('profile.py','precision.py','common.py')},
        mode='CUDA Graph',paths=['grouped','split'],weight_source='actual layer checkpoint tensors',
        activation_source='captured post-attention layer input, native BF16 autoregressive decode',
        quantization='RTN; W8A8 per-output-channel weights/dynamic per-token activations; WNA16 K-group128 weights/BF16 activations',
        model_quality='local FFN output error only; no claim of full-model quantized quality',
        cache='warm layer replay; no intervening cache flush',
        scope='post-router Shared + Routed FFN + combine; route already known; category maps built outside timed replay',
        split_contract='two masked vLLM calls repeat alignment and common work; not attribution of unsplit grouped runtime',
        backend='vLLM fused_experts_impl; actual Triton/CUDA WNA16 dispatch chosen by installed library',complete=False)
    metap=a.out/'metadata.json'
    if metap.exists():
        old=json.loads(metap.read_text())
        for k in ('precision','repeat','layers','steps','groups','capture_manifest_sha256','source_sha256'):
            if old[k]!=meta[k]:raise ValueError('resume metadata mismatch '+k)
    else:write_json(metap,meta)
    for group in groups:
        if sha(a.capture/group['file'])!=group['sha256']:raise ValueError('capture shard changed')
    completed=[]
    with torch.inference_mode():
        for li in layers:
            dest=a.out/f'layer_{li:02d}.jsonl.gz';receipt=a.out/f'layer_{li:02d}.complete.json'
            if receipt.exists():
                r=json.loads(receipt.read_text())
                if sha(dest)!=r['sha256']:raise ValueError('changed prior layer')
                completed.append(r);continue
            print('LAYER',li,'LOAD',a.precision,flush=True);started=time.monotonic()
            rpair,spair=layer_weights(a.model,li)
            routed,shared=ExpertWeights(rpair,a.precision),ExpertWeights(spair,a.precision)
            stored_bytes=routed.storage_bytes()+shared.storage_bytes()
            del rpair,spair;gc.collect();torch.cuda.empty_cache()
            points=0;graphs={}
            shuffled=list(groups);random.Random(20261016+a.repeat+li).shuffle(shuffled)
            with gzip.open(str(dest)+'.tmp','wt') as out:
                for group in shuffled:
                    shard=a.capture/group['file']
                    with np.load(shard) as z:
                        allids=z[f'ids_l{li}'];allsc=z[f'scores_l{li}'];xx=z[f'x_l{li}'];yy=z[f'y_l{li}']
                    for si,step in enumerate(steps):
                        ids,sc=allids[step-1],allsc[step-1];validate_routes(ids,sc)
                        x=torch.from_numpy(xx[si].copy()).view(torch.bfloat16).cuda()
                        ref=torch.from_numpy(yy[si].copy()).view(torch.bfloat16).cuda()
                        counts=np.bincount(ids.reshape(-1),minlength=64)
                        kind=(group['batch'],bool((counts>2).any()),bool(((counts>0)&(counts<=2)).any()))
                        results={}
                        for path in ('grouped','split'):
                            key=(*kind,path)
                            if key not in graphs:graphs[key]=GraphBench(Runner(routed,shared,x.clone(),ids,sc),path,a.warmup)
                            bench=graphs[key];bench.update(x,ids,sc)
                            timings,y=bench.measure(a.warmup,a.samples);err=error_metrics(y,ref)
                            if not err['finite']:raise ValueError('nonfinite result')
                            if a.precision in ('bf16','fp16') and err['relative_l2']>.05:raise ValueError('unquantized layer replay mismatch '+str(err))
                            results[path]=y
                            row=dict(window_id=f'{group["id"]}_s{step}_l{li}',dataset=group['dataset'],cohort=group['cohort'],
                                batch=group['batch'],step=step,layer=li,phase='decode',precision=a.precision,repeat=a.repeat,
                                path=path,model_weight_storage_MiB=stored_bytes/2**20,
                                active_weight_storage_MiB=(shared.storage_bytes()+routed.storage_bytes()/64*int((counts>0).sum()))/2**20,
                                **population(ids),**err,**timings)
                            out.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');points+=1
                        consistency=error_metrics(results['split'],results['grouped'])
                        if consistency['relative_l2']>.03:raise ValueError('split/grouped numerical mismatch '+str(consistency))
                        del x,ref,results
                    print('PROFILE',a.precision,a.repeat,li,group['id'],points,flush=True)
            Path(str(dest)+'.tmp').replace(dest)
            r=dict(layer=li,points=points,sha256=sha(dest),seconds=time.monotonic()-started)
            write_json(receipt,r);completed.append(r)
            del routed,shared,graphs,bench;gc.collect();torch.cuda.empty_cache()
    expected=len(layers)*len(groups)*len(steps)*2
    if sum(r['points'] for r in completed)!=expected:raise ValueError('incomplete profile')
    write_json(a.out/'COMPLETE.json',dict(points=expected,layers=completed,metadata_sha256=sha(metap)))
    print('PROFILE COMPLETE',a.precision,a.repeat,expected,flush=True)

if __name__=='__main__':main()
