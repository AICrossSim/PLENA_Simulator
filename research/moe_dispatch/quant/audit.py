#!/usr/bin/env python3
"""Hash every routed/shared BF16 tensor in layers2/13/26, inventory available captures."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from safetensors import safe_open


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();torch.set_num_threads(2);start=time.monotonic()
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map'];records={};missing=[]
    for l in (2,13,26):
        for e in list(range(64))+[-1]:
            stem='shared_experts' if e==-1 else f'experts.{e}'
            for phase in ('gate','up','down'):
                name=f'model.layers.{l}.mlp.{stem}.{phase}_proj.weight';p=a.model/index[name]
                with safe_open(p,framework='pt',device='cpu') as f:t=f.get_tensor(name)
                if t.dtype!=torch.bfloat16:raise ValueError('expected BF16 '+name)
                raw=t.contiguous().view(torch.uint16).numpy().tobytes();records[name]={'sha256':hashlib.sha256(raw).hexdigest(),'shard':p.name,'shape':list(t.shape),'dtype':'BF16','bytes':len(raw)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report={'schema':'plena_v3_weight_inventory_v1','layers':[2,13,26],'experts_per_layer':65,'tensor_count':len(records),'tensor_provenance':records,
            'model_index_sha256':hashlib.sha256((a.model/'model.safetensors.index.json').read_bytes()).hexdigest(),
            'hash_basis':'exact BF16 tensor bytes read from local model shards; generated Q0 provenance, not an earlier capture claim',
            'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'seconds':time.monotonic()-start}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='tensor_provenance'}))

if __name__=='__main__':main()
