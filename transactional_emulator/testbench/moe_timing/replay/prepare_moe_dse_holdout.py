#!/usr/bin/env python3
"""Freeze unseen route coordinates before ranking; export actual codec + oracle."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from compare_moe_normal import digest,read_json,require
from run_moe_dma_campaign import save,scale_layout


def prepare(root,source,compiler):
    sys.path.insert(0,str(compiler/'aten/plena'))
    from moe_full_shape_export import export_full_shape
    torch.set_num_threads(1)
    selections=[]
    for model in ['qwen','deepseek']:
        training=read_json(source/(model+'_full_decode_b32')/'workload.json')
        p=training['metadata']['provenance']; npz=Path(p['source_npz'])
        require(digest(npz)==p['source_npz_sha256'],'route archive changed')
        with np.load(str(npz),allow_pickle=False) as z:
            ids=z['decode_idx'];weights=z['decode_weight'];valid=z['valid'];meta=json.loads(str(z['meta']))
        step=p['step'];layer=p['layer_index'];first=min(p['stream_indices'])
        used=set(p['stream_indices'])
        for batch,offset in [(8,64),(32,96)]:
            available=np.flatnonzero(valid[:,step]);selected=available[available>=first+offset][:batch]
            require(len(selected)==batch and not set(selected).intersection(used),'holdout unavailable/disjointness failure')
            used.update(selected.tolist())
            provenance=dict(p,stream_indices=selected.tolist(),partition='holdout',
                selection_rule='first valid streams >= original cohort start +64 for B8; +96 for B32; same layer and decode step; frozen before DSE ranking',
                disjoint_from_training_and_other_holdout_batch=True)
            top_ids=ids[selected,step,layer];top_weights=weights[selected,step,layer].astype(np.float32)
            routes=[dict(token=t,slot=k,expert=int(e),weight=float(top_weights[t,k]))
                for t,row in enumerate(top_ids) for k,e in enumerate(row)]
            selections.append(dict(name=model+'_holdout_b'+str(batch),tokens=batch,
                input_dim=meta['hidden_size'],expert_hidden_dim=meta['moe_intermediate_size'],routes=routes,provenance=provenance))
    save(root/'holdout_selection.json',dict(selection_frozen=True,fixtures=selections))
    for selection in selections:
        name=selection['name']
        print('Exporting '+name,flush=True)
        export_full_shape(root/'holdout/raw'/name,**selection)
        scale_layout(root/'holdout/raw'/name,root/'holdout/scale_odd32'/name)
    records={}
    for selection in selections:
        name=selection['name']
        for layout in ['raw','scale_odd32']:
            folder=root/'holdout'/layout/name;w=read_json(folder/'workload.json')
            records[layout+'/'+name]=dict(path=str(folder),workload_sha256=digest(folder/'workload.json'),
                golden_sha256=digest(folder/'golden.json'),hbm_sha256=digest(folder/w['hbm_file']))
    save(root/'holdout_manifest.json',dict(fixtures=records,selection_sha256=digest(root/'holdout_selection.json'),
        generator_sha256=digest(__file__),scope='Disjoint captured route coordinates, same two model families and synthetic value generator; not independent full-model trajectories'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--compiler',type=Path,required=True)
    a=p.parse_args();prepare(a.root.resolve(),a.source.resolve(),a.compiler.resolve())
