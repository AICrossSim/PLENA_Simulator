#!/usr/bin/env python3
"""Predeclare a full-dimension Rust MoE campaign from validated route archives."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def configurations():
    common=dict(schema_version=1,dispatch_threshold=8,large_core=0,small_core=0,
        global_dma_credits=64,global_dma_staging_bytes=4096,combine_sram_bytes=4*1024*1024,
        clock_period_ps=1000,mac_pipeline_cycles=16,vector_elements_per_cycle=512,
        dispatch_policy='work_conserving',dispatch_queue_bytes=16384,dispatch_cycles=1)
    def core(name,b,k,parts):
        return dict(id=name,blen=b,mlen=k,vector_sram_bytes=4*1024*1024//parts,
            accumulator_bytes=1024*1024//parts,weight_sram_bytes=65536//parts,
            read_cache_bytes=40960//parts)
    values=[]
    for b,k in [(32,128),(16,256),(8,512),(4,1024)]:
        values.append(dict(common,name=f'single_b{b}_k{k}',cores=[core('single',b,k,1)]))
    values.append(dict(common,name='homogeneous_b16_k128',small_core=1,
        cores=[core('large',16,128,2),core('small',16,128,2)]))
    values.append(dict(common,name='heterogeneous_b16_k192_b8_k128',small_core=1,
        cores=[core('large',16,192,2),core('small',8,128,2)]))
    values.append(dict(values[-1],name='heterogeneous_fixed_threshold',dispatch_policy='threshold'))
    for a in values:
        assert sum(c['blen']*c['mlen'] for c in a['cores'])==4096
    return values


def prepare(args):
    sys.path.insert(0,str(args.compiler/'aten/plena'))
    from moe_full_shape_export import export_full_shape
    torch.set_num_threads(1)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    configs=configurations()
    for a in configs:save(out/'architectures'/(a['name']+'.json'),a)
    fixtures=[]
    archives={
        'qwen':'outputs/swe_grouping_sram_20260903/sram_ring_sweep/qwen_b8_h512_i256_trace.json',
        'deepseek':'outputs/swe_grouping_sram_20260903/panel_pool_sweep/deepseek_h512_slots2/trace.json'}
    for model,relative in archives.items():
        archive_path=args.workspace/relative
        archive=json.loads(archive_path.read_text()); provenance=archive['provenance']
        source=Path(provenance['source_trace']); actual_sha=sha(source)
        if actual_sha!=provenance['source_trace_sha256']:
            raise ValueError('original NPZ identity mismatch: '+str(source))
        with np.load(source,allow_pickle=False) as z:
            ids,weights,valid=z['decode_idx'],z['decode_weight'],z['valid']
            meta=json.loads(str(z['meta']));layer_ids=z['layer_ids']
        s=provenance['slice']; start,step,layer=s['cohort_start'],s['step'],s['layer_index']
        if ids[start:start+8,step,layer].tolist()!=archive['routing']['topk_indices']:
            raise ValueError('derived archive ids do not match original NPZ')
        if weights[start:start+8,step,layer].astype(np.float32).tolist()!=archive['routing']['topk_weights']:
            raise ValueError('derived archive weights do not match original NPZ')
        available=np.flatnonzero(valid[:,step]); available=available[available>=start]
        for batch in [8,32]:
            selected=available[:batch]
            if len(selected)!=batch or (batch==8 and selected.tolist()!=list(range(start,start+8))):
                raise ValueError('required original valid decode cohort is unavailable')
            top_ids=ids[selected,step,layer];top_weights=weights[selected,step,layer].astype(np.float32)
            routes=[dict(token=t,slot=k,expert=int(expert),weight=float(top_weights[t,k]))
                    for t,row in enumerate(top_ids) for k,expert in enumerate(row)]
            name=f'{model}_full_decode_b{batch}'
            evidence=dict(source_npz=str(source),source_npz_sha256=actual_sha,
                source_archive_json=str(archive_path),source_archive_sha256=sha(archive_path),
                stream_indices=selected.tolist(),step=step,layer_index=layer,model_layer_id=int(layer_ids[layer]),
                original_capture_batch_size=meta['capture_batch_size'],
                routing_scope='rebatch of actual captured decode decisions; not a captured batch-32 model forward or prefill trace',
                model_id=meta['model_id'],hidden_size=meta['hidden_size'],expert_hidden_size=meta['moe_intermediate_size'])
            print('Exporting',name,'active_experts',len(set(top_ids.flatten().tolist())),flush=True)
            export_full_shape(out/name,input_dim=meta['hidden_size'],expert_hidden_dim=meta['moe_intermediate_size'],
                tokens=batch,routes=routes,name=name,provenance=evidence)
            fixtures.append(name)
            save(out/'preparation_progress.json',dict(completed=fixtures))
    save(out/'campaign.json',dict(schema_version=1,fixtures=fixtures,
        architectures=[a['name'] for a in configs],repeats=2,hbm_channels=8,
        numerical_tolerance=dict(atol=1e-6,rtol=0.01),generator_sha256=sha(__file__),
        scope='Full source-model D/F dimensions; actual archived decode routes rebatched; synthetic weights and ready inputs; routed FFN only'))
    print('Prepared',out/'campaign.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['compiler','workspace','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    prepare(parser.parse_args())
