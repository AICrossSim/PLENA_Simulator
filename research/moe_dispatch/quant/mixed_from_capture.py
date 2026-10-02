#!/usr/bin/env python3
"""Join 16 captured decode rows with a continuous real prompt prefill prefix.

This does not concatenate independent request tokens pretending they are one
prompt. Prefill rows preserve a single request and adjacent original positions.
The architectural decode rows are separately captured real routing inputs.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture-manifest',type=Path,required=True);ap.add_argument('--decode-workloads',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--split',default='development');ap.add_argument('--layer',type=int,default=13)
    ap.add_argument('--dataset',default='bfcl');ap.add_argument('--step',type=int);ap.add_argument('--layers',help='comma separated, defaults to --layer')
    args=ap.parse_args();m=json.loads(args.capture_manifest.read_text());s=m['shards'][0];p=Path(s['path']);p=p if p.is_absolute() else args.capture_manifest.parent/p
    if hashlib.sha256(p.read_bytes()).hexdigest()!=s['sha256']:raise ValueError('capture payload hash mismatch')
    archive=json.loads(args.decode_workloads.read_text());layers=list(map(int,args.layers.split(','))) if args.layers else [args.layer]
    allcases=[]
    for layer in layers:
        allcases += build_cases(archive,m,s,p,args.dataset,layer,args.step,args.split)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({'schema':'plena_moe_supply_v3_mixed_v1','workloads':allcases},indent=2)+'\n')
    print(json.dumps({'path':str(args.output),'cases':len(allcases),'sizes':sorted({w['batch'] for w in allcases})}))


def build_cases(archive,m,s,p,dataset,layer,step,split):
    base=next(w for w in archive['workloads'] if w['batch']==16 and w['layer']==layer and w['dataset']==dataset and (step is None or w['step']==step))
    if s['request_id'] in base['request_ids']:raise ValueError('prefill request reused among concurrent decode requests')
    template={e['id']:e for e in base['experts']};single=next(e for e in base['experts'] if not e['is_shared']);sh=next(e for e in base['experts'] if e['is_shared'])
    with np.load(p) as z:rr=z[f'routes_l{layer}'];gg=z[f'gates_l{layer}']
    cases=[]
    for T in (64,96,128):
        count=T-16
        if len(rr)<count:continue
        tokens=copy.deepcopy(base['tokens']);positions={}
        for j in range(count):
            tokens.append({'token_index':16+j,'sample_id':s['request_id'],'prompt_position':j,
                           'routes':[{'expert_id':int(e),'slot':slot,'score':float(g)} for slot,(e,g) in enumerate(zip(rr[j],gg[j]))]})
        for token in tokens:
            for r in token['routes']:positions.setdefault(r['expert_id'],[]).append((token['token_index'],r['slot'],r['score']))
        experts=[]
        for e,rows in sorted(positions.items()):
            ex=copy.deepcopy(template.get(e,single));ex.update({'id':e,'Me':len(rows),'token_indices':[r[0] for r in rows],
                    'route_slots':[r[1] for r in rows],'route_scores':[r[2] for r in rows],'is_shared':False})
            ex.pop('dag',None)
            # Existing per-expert HBM addresses are provenance only; v3 repacks per its descriptors.
            experts.append(ex)
        ex=copy.deepcopy(sh);ex.update({'Me':T,'token_indices':list(range(T)),'route_slots':[-1]*T,'route_scores':[1.]*T});ex.pop('dag',None);experts.append(ex)
        case=copy.deepcopy(base);case.update({'id':f'v3_captured_mixed_{split}_{dataset}_t{T}_l{layer}','batch':T,'experts':experts,'tokens':tokens,
                'scope':'16 archived real decode tokens + continuous real CPU BF16 prefill prefix from one request',
                'provenance':{'origin':'captured_mixed','split':split,'dataset':dataset,'decode_case':base['id'],
                              'prefill_request_id':s['request_id'],'request_sha256':s['request_sha256'],'prefill_positions':[0,count],
                              'capture_sha256':s['sha256'],'continuous_prompt':True,'capture_prefix_smoke':s.get('prefix_truncated_smoke',False)}})
        cases.append(case)
    return cases

if __name__=='__main__':main()
