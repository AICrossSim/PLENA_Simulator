"""Real pretrained weights, deterministic fake-quant numerical diagnostic.

Synthetic X is deliberately labelled; this is NOT perplexity/accuracy, not
QERA/LQER calibration, and does not qualify W4 for an architecture claim.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
from safetensors import safe_open
from ..geometry3d.study import write_csv,write_json


def quantize(weight,bits,group=128):
    n,k=weight.shape;pad=(-k)%group
    a=torch.nn.functional.pad(weight.float(),(0,pad)).reshape(n,-1,group)
    limit=2**(bits-1)-1
    scale=(a.abs().amax(dim=-1)/limit).clamp_min(torch.finfo(torch.float16).tiny).half()
    q=(a/scale.float().unsqueeze(-1)).round().clamp(-limit,limit).to(torch.int8)
    w=(q.float()*scale.float().unsqueeze(-1)).reshape(n,-1)[:,:k].to(torch.bfloat16).float()
    return q,scale,w


def run(weights,out):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    index=weights/'model.safetensors.index.json';meta=json.loads(index.read_text());rows=[]
    keys=[k for k in meta['weight_map'] if 'layers.1.mlp.' in k and ('.shared_experts.' in k or '.experts.0.' in k)]
    for key in sorted(keys):
        shard=weights/meta['weight_map'][key]
        with safe_open(str(shard),framework='pt',device='cpu') as f:weight=f.get_tensor(key).float()
        gen=torch.Generator().manual_seed(20261005)
        x=torch.randn((8,weight.shape[1]),generator=gen).to(torch.bfloat16).float()
        reference=x@weight.T
        for bits in (8,4):
            def measure():
                q,scale,w=quantize(weight,bits)
                pred=x@w.T
                return {'weight_NMSE':float((w-weight).square().sum()/weight.square().sum()),
                        'synthetic_X_output_NMSE':float((pred-reference).square().sum()/reference.square().sum()),
                        'weight_max_abs_error':float((w-weight).abs().max()),
                        'scale_bytes':scale.numel()*2,'logical_packed_code_bytes':(q.numel()*bits+7)//8}
            a=measure();b=measure();assert a==b
            rows.append({'tensor':key,'source_shard':shard.name,'source_tensor_SHA256':hashlib.sha256(weight.numpy().tobytes()).hexdigest(),
                         'N':weight.shape[0],'K':weight.shape[1],'weight_bits':bits,'group':128,
                         'scale_format':'FP16','decoded_format':'BF16','repeat_exact':True,**a,
                         'scope':'pretrained weights plus synthetic BF16 X; NOT captured activation/task quality'})
    write_csv(out/'real_weight_quant_diagnostic.csv',rows)
    write_json(out/'QUANT_QUALITY_STATUS.json',{'tensors':len(keys),'points':len(rows),'repeats':2,
        'weight_index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),
        'trained_model_task_quality_qualified':False,'W4_architecture_claim_allowed':False,
        'reason':'only6realweightmatrices, syntheticX; no full-model perplexity/accuracy or QERA low-rank reconstruction',
        'next_required':'freeze format/scales/calibration; run real calibration activations and heldout model quality, state degradation threshold'})
    print('Real-weight numerical diagnostic complete; trained-model quality remains unqualified.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.weights,a.out)
