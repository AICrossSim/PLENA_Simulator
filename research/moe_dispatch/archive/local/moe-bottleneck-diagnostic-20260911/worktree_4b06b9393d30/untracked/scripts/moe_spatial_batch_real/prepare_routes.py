#!/usr/bin/env python3
"""Inventory every valid captured routing decision; select declared nested windows.

Timing coverage is deliberately separate from inventory coverage. No synthetic
tokens, invalid padded tokens, or values inferred from capture timing are used.
"""
from pathlib import Path
from collections import Counter
import hashlib, json, csv
import numpy as np

ROOT = Path('/scratch/shared/mcl123/plena')
OUT = ROOT / 'outputs/moe_spatial_batch_real_20260921'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''): h.update(x)
    return h.hexdigest()

def main():
    inventory=[]; windows=[]
    for p in sorted((ROOT/'outputs/real_shared_moe_routes_20260819').glob('*/*.npz')):
        with np.load(p,allow_pickle=False) as a:
            m=json.loads(str(a['meta'])); idx=a['decode_idx']; valid=a['valid']; weights=a['decode_weight']
            assert idx.shape[:2]==valid.shape and idx.shape==weights.shape
            model='qwen' if 'qwen' in p.parent.name else 'deepseek' if 'deepseek' in p.parent.name else 'nemotron'
            dataset='bfcl' if 'bfcl' in p.name else 'gpqa' if 'gpqa' in p.name else 'swe'
            active=idx[valid]
            assert np.all((active>=0)&(active<m['routed_experts']))
            assert np.all(np.diff(np.sort(active,axis=-1),axis=-1)!=0)
            assert np.all(np.isfinite(weights[valid])) and np.all(weights[valid]>=0)
            counts=np.bincount(active.reshape(-1),minlength=m['routed_experts'])
            inv=dict(model=model,dataset=dataset,path=str(p),sha256=sha(p),streams=idx.shape[0],
                steps=idx.shape[1],layers=idx.shape[2],top_k=idx.shape[3],valid_stream_steps=int(valid.sum()),
                valid_routed_pairs=int(active.size),hidden=m['hidden_size'],intermediate=m['moe_intermediate_size'],
                shared_intermediate=m['shared_expert_intermediate_size'],capture_batch=m['capture_batch_size'],
                expert_pair_counts=counts.tolist(),metadata=m)
            inventory.append(inv)
            # Two independent groups, at first/middle layer and early/later decode.
            # Choice is fixed before observing simulator results.
            for split,step,li,offset in [('primary',0,0,0),('holdout',8,idx.shape[2]//2,16)]:
                streams=np.flatnonzero(valid[:,step])[offset:offset+16]
                assert len(streams)==16
                for batch in [2,4,8,16]:
                    ss=streams[:batch]; decisions=idx[ss,step,li]
                    c=Counter(map(int,decisions.reshape(-1)))
                    common=dict(model=model,dataset=dataset,split=split,batch=batch,source=str(p),source_sha256=inv['sha256'],
                        original_capture_batch=inv['capture_batch'],decode_step=step,moe_layer_index=li,
                        model_layer_id=int(a['layer_ids'][li]),stream_indices=ss.tolist(),sample_ids=a['sample_ids'][ss].tolist(),
                        hidden=inv['hidden'],intermediate=inv['intermediate'],shared_intermediate=inv['shared_intermediate'],
                        routed_pairs=batch*idx.shape[3],expert_counts=dict(sorted(c.items())),
                        routes=decisions.tolist(),route_weights=weights[ss,step,li].astype(float).tolist(),
                        evidence='captured routing; rebatched; full matrix dimensions; timing-only, no original X/W in NPZ')
                    assert sum(c.values())==common['routed_pairs']
                    common['name']=f'{model}_{dataset}_{split}_b{batch}'
                    windows.append(common)
    assert len(inventory)==9 and len(windows)==72
    for name,data in [('archive_inventory',inventory),('route_windows',windows)]:
        (OUT/'inputs'/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    with (OUT/'inputs/coverage.csv').open('w') as f:
        keys=['model','dataset','streams','steps','layers','top_k','valid_stream_steps','valid_routed_pairs','hidden','intermediate','shared_intermediate','capture_batch']
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:x[k] for k in keys} for x in inventory)
    print(json.dumps(dict(archives=len(inventory),inventoried_pairs=sum(x['valid_routed_pairs'] for x in inventory),selected_windows=len(windows))))

if __name__=='__main__':main()
