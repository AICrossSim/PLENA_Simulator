"""Deterministic request-disjoint route windows, selected without timing results."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from frontend import compiler

CAPTURES = {
    'bfcl': 'deepseek_v2_lite_shared_moe_bfcl_v3_batch16_decode16.npz',
    'gpqa': 'deepseek_v2_lite_shared_moe_gpqa_diamond_batch16_decode16.npz',
    'swe': 'deepseek_v2_lite_shared_moe_swe_bench_full_test_batch16_decode16.npz',
}


def build_window(a, meta, indices, step, layer, name):
    positions={};tokens=[]
    li=list(a['layer_ids']).index(layer)
    for token,i in enumerate(indices):
        assert a['valid'][i,step]
        ids=a['decode_idx'][i,step,li];scores=a['decode_weight'][i,step,li]
        assert len(set(map(int,ids)))==meta['top_k']
        routes=[]
        for slot,(eid,score) in enumerate(zip(ids,scores)):
            eid,score=int(eid),float(score)
            assert 0<=eid<meta['routed_experts'] and np.isfinite(score) and score>=0
            positions.setdefault(eid,[]).append((token,slot,score))
            routes.append(dict(expert_id=eid,slot=slot,score=score))
        tokens.append(dict(token_index=token,sample_id=str(a['sample_ids'][i]),routes=routes))
    experts=[];h=meta['hidden_size'];batch=len(indices)
    for eid in sorted(positions)+[-1]:
        f=meta['shared_expert_intermediate_size'] if eid==-1 else meta['moe_intermediate_size']
        assignments=positions[eid] if eid!=-1 else [(t,-1,1.) for t in range(batch)]
        weights={}
        for phase,(key,n,k) in enumerate((('gate',f,h),('up',f,h),('down',h,f))):
            stride=compiler.align(2*k)
            assert n*stride<=compiler.EXPERT_PHASE_STRIDE
            weights[key]=dict(shape_nk=[n,k],dtype='BF16',row_stride_bytes=stride,
                              physical_bytes=n*stride,payload_bytes=2*n*k,
                              hbm_base=((meta['routed_experts'] if eid==-1 else eid)*3+phase)*compiler.EXPERT_PHASE_STRIDE,
                              payload_scope='shape/address only; no numerical pretrained weights loaded')
        e=dict(id=eid,is_shared=eid==-1,Me=len(assignments),H=h,F=f,
               token_indices=[v[0] for v in assignments],route_slots=[v[1] for v in assignments],
               route_scores=[v[2] for v in assignments],weights=weights)
        e['dag']=compiler.expert_dag(e);experts.append(e)
    return dict(id=name,batch=batch,hidden=h,top_k=meta['top_k'],tokens=tokens,experts=experts,
                routing_is_input_not_timed=True,route_scores_renormalized=False,
                scope='offline window of distinct captured decode requests; not a new batched model execution')


def prepare(base, output):
    data={};sources=[];manifest=[];all_ids={}
    for dataset, filename in CAPTURES.items():
        path=base/'deepseek-v2-lite-chat'/filename
        a=np.load(path,allow_pickle=False);meta=json.loads(a['meta'].item())
        assert meta['shared_gate']=='none' and meta['top_k']==6 and meta['hidden_size']==2048
        sources.append(dict(dataset=dataset,path=str(path),sha256=compiler.sha256(path),metadata=meta,
                            prefill_excluded='aggregate counts lack per-token route order; outside bounded current storage scope'))
        data[dataset]=(a,meta)
    # Different datasets for selection/validation, heldout sources and requests
    # fixed here. No timing or route-hotness criterion enters sample selection.
    plans=[('design','bfcl',0),('validation','gpqa',0),('heldout','bfcl',7),('heldout','swe',7)]
    workloads={k:[] for k in ('design','validation','heldout')}
    for split,dataset,step in plans:
        a,meta=data[dataset]
        pool=[]
        for i,sample in enumerate(a['sample_ids']):
            ident=dataset+':'+str(sample)
            code=hashlib.sha256(('plena-robust-v1:'+ident).encode()).hexdigest()
            bucket=int(code[:8],16)%10
            selected=(bucket<6 if split=='design' else 6<=bucket<8 if split=='validation' else bucket>=8)
            if selected and a['valid'][i,step]:pool.append((code,i,ident))
        pool.sort();assert len(pool)>=30
        cursor=0
        for bi,batch in enumerate((2,4,8,16)):
            chosen=pool[cursor:cursor+batch];cursor+=batch
            assert len(chosen)==batch
            layer=(1,13,1,13)[bi]
            name=f'{split}_{dataset}_b{batch}_l{layer}_s{step}'
            indices=[x[1] for x in chosen]
            for _,_,ident in chosen:
                assert ident not in all_ids, 'Request leakage across windows/splits'
                all_ids[ident]=split
            w=build_window(a,meta,indices,step,layer,name)
            workloads[split].append(w)
            manifest.append(dict(id=name,split=split,dataset=dataset,batch=batch,layer=layer,decode_step=step,
                                 sample_ids=[x[2] for x in chosen],stream_indices=indices,
                                 model='DeepSeek-V2-Lite',phase='decode',offline_rebatch=True))
    exclusions=[dict(model='Nemotron3-Nano',reason='relu2 expert graph differs from implemented SwiGLU; excluded from model-equivalent timing'),
                dict(model='Qwen3.5',reason='captured FP8 model includes Shared sigmoid gate absent in current BF16 flow; excluded from model-equivalent timing')]
    output.mkdir(parents=True,exist_ok=True)
    for split,ws in workloads.items():
        (output/f'{split}_workloads.json').write_text(json.dumps(dict(schema=compiler.SCHEMA,workloads=ws),indent=2)+'\n')
    result=dict(selection='SHA256(plena-robust-v1:dataset:sample_id); buckets 0..5/6..7/8..9; unique requests',
                sources=sources,windows=manifest,excluded_models=exclusions,
                supported_scope='one model, three datasets, two layers and decode steps; no cross-model robustness claim')
    (output/'workload_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return workloads,result
