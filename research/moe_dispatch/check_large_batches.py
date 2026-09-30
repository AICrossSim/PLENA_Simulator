"""Rebatch recorded decode rows, then reject infeasible SRAM plans before timing.

Rebatching is a trace-driven scenario, not a captured B256 model execution.
No routes or payloads are invented; missing weight payload hashes stay absent.
"""
import argparse
import copy
import csv
import json
from pathlib import Path
import numpy as np
from frontend import compiler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=compiler.DEFAULT_WORKLOADS)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    a=np.load(args.capture,allow_pickle=False)
    meta=json.loads(a['meta'].item())
    layer=list(a['layer_ids']).index(1)
    indices=np.flatnonzero(a['valid'][:,0])
    assert len(indices)>=256 and meta['hidden_size']==2048 and meta['top_k']==6
    ref=json.loads(args.reference.read_text())['workloads'][-1]
    routed=next(e for e in ref['experts'] if not e['is_shared'])
    shared=next(e for e in ref['experts'] if e['is_shared'])
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]; loads=[]; workloads=[]
    for batch in (32,64,128,256):
        tokens=[]; positions={}
        for t,i in enumerate(indices[:batch]):
            routes=[]
            for slot,(eid,score) in enumerate(zip(a['decode_idx'][i,0,layer],a['decode_weight'][i,0,layer])):
                eid,score=int(eid),float(score)
                assert 0<=eid<64 and np.isfinite(score)
                positions.setdefault(eid,[]).append((t,slot,score))
                routes.append({'expert_id':eid,'slot':slot,'score':score})
            tokens.append({'token_index':t,'sample_id':str(a['sample_ids'][i]),'routes':routes})
        experts=[]
        for eid in sorted(positions)+[-1]:
            e=copy.deepcopy(shared if eid==-1 else routed)
            p=positions[eid] if eid!=-1 else [(t,-1,1.) for t in range(batch)]
            e.update(id=eid,Me=len(p),is_shared=eid==-1,token_indices=[t for t,_,_ in p],
                     route_slots=[s for _,s,_ in p],route_scores=[s for _,_,s in p])
            for phase,(name,w) in enumerate(e['weights'].items()):
                # Preserve existing expert/phase address formula and BF16 row layout.
                phase=('gate','up','down').index(name)
                w['hbm_base']=((64 if eid==-1 else eid)*3+phase)*compiler.EXPERT_PHASE_STRIDE
                w['tensor_name']=f'model.layers.1.mlp.'+('shared_experts' if eid==-1 else f'experts.{eid}')+f'.{name}_proj.weight'
                for k in ('source_sha256','source_shard','source_hash_verified_by'): w.pop(k,None)
                w['payload_scope']='shape/address metadata only; weights not loaded in timing model'
            e['dag']=compiler.expert_dag(e)
            experts.append(e)
            loads.append({'batch':batch,'expert':eid,'shared':eid==-1,'Me':len(p)})
        w={'id':f'archived_decode_rebatch_b{batch}','batch':batch,'hidden':2048,'top_k':6,
           'tokens':tokens,'experts':experts,'routing_is_input_not_timed':True,
           'route_scores_renormalized':False}
        workloads.append(w)
        for lanes in ((6,),(3,3),(4,2)):
            plan=compiler.compile_workload(w,lanes,'whole',4)
            feasible=all(any(v['cores'][0]['storage']['feasible'] for v in e['candidates']) for e in plan['experts'])
            for c,m in enumerate(lanes):
                peaks=[e['candidates'][c]['cores'][0]['storage']['peak_private_bytes'] for e in plan['experts']]
                cap=plan['budget']['private_accumulator_bytes'][c]
                rows.append({'batch':batch,'organization':'+'.join(map(str,lanes)),'core':c,'M':m,
                             'capacity_bytes':cap,'maximum_peak_bytes':max(peaks),
                             'experts_fitting_this_core':sum(p<=cap for p in peaks),
                             'expert_count':len(experts),'whole_workload_feasible':feasible,
                             'timed':False,'reason':'capacity rejection' if not feasible else 'eligible for timing'})
    provenance={'capture_path':str(args.capture.resolve()),'capture_sha256':compiler.sha256(args.capture),
                'decode_step':0,'layer':1,'stream_indices':indices[:256].tolist(),
                'scope':'rebatch distinct recorded requests, not new B32/B64/B128/B256 GPU inference',
                'model_metadata':meta}
    (args.output/'workloads.json').write_text(json.dumps({'schema':compiler.SCHEMA,'provenance':provenance,'workloads':workloads},indent=2)+'\n')
    for name,data in [('capacity.csv',rows),('expert_loads.csv',loads)]:
        with (args.output/name).open('w',newline='') as f:
            out=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n");out.writeheader();out.writerows(data)
    print(json.dumps({'batch_organization_points':12,'feasible_points':sum(r['whole_workload_feasible'] for r in rows if r['core']==0),'output':str(args.output)}))

if __name__=='__main__': main()
