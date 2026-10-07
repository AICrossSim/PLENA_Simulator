#!/usr/bin/env python3
"""Restore disposable BF16 exports from the original local checkpoint tensors."""
import hashlib,json
from pathlib import Path
import torch
from safetensors import safe_open
from prepare_routes import OUT

def main():
    m=json.loads((OUT/'real_inputs/manifest.json').read_text())
    for tag,export in m['exported_weights'].items():
        expert,phase=tag.rsplit('_',1)
        prefix='model.layers.1.mlp.shared_experts' if expert=='shared' else f'model.layers.1.mlp.experts.{expert}'
        key=f'{prefix}.{phase}_proj.weight';source=m['tensor_provenance'][key]
        p=Path(export['path']);p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            assert hashlib.sha256(p.read_bytes()).hexdigest()==export['sha256'],p
            continue
        with safe_open(source['shard'],framework='pt',device='cpu') as f:v=f.get_tensor(key)
        assert list(v.shape)==source['shape'] and v.dtype==torch.bfloat16
        raw=v.contiguous().view(torch.uint16).numpy().tobytes()
        assert hashlib.sha256(raw).hexdigest()==source['sha256']==export['sha256']
        p.write_bytes(raw)
    print('RESTORE/VERIFY PASS',len(m['exported_weights']))
if __name__=='__main__':main()
