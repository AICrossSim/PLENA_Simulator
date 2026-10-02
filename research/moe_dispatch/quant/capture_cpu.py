#!/usr/bin/env python3
"""Capture real BF16 MoE inputs/routes at layers 2/13/26 using a local checkpoint.

Requests are canonical archived messages re-tokenized for DeepSeek; token-level
records preserve contiguous prompt positions and SHA256 of the original request.
Use --smoke-tokens to measure CPU feasibility (not a calibration data completion).
"""
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--archives',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--split',choices=['development','heldout','both'],default='development')
    ap.add_argument('--target-tokens',type=int,default=32768);ap.add_argument('--smoke-tokens',type=int,default=0);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--eval-tokens',type=int,default=8192)
    ap.add_argument('--resume',action='store_true',help='verify and retain completed real request shards')
    ap.add_argument('--canonical-jsonl',type=Path,help='additional exact-hash recovered canonical requests, e.g. SWE')
    ap.add_argument('--canonical-sha256',help='required provenance hash for --canonical-jsonl')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    selection=json.loads(args.selection.read_text());wanted=set(selection['development'])|set(selection['heldout']) if args.split=='both' else set(selection[args.split]);other=set() if args.split=='both' else set(selection['heldout' if args.split=='development' else 'development'])
    if wanted & other: raise ValueError('request split leakage')
    rows={}
    sources=[]
    if args.canonical_jsonl:
        source_hash=hashlib.sha256(args.canonical_jsonl.read_bytes()).hexdigest()
        if not args.canonical_sha256 or source_hash!=args.canonical_sha256:raise ValueError('canonical request-source hash mismatch')
        sources.append({'path':str(args.canonical_jsonl.resolve()),'sha256':source_hash})
        with args.canonical_jsonl.open() as f:
            for line in f:
                v=json.loads(line);bench=v.get('benchmark','')
                prefix='swe' if bench=='swe_bench' else 'bfcl' if bench.startswith('bfcl') else 'gpqa'
                identity=prefix+':'+v['sample_id']
                if identity in wanted:rows[identity]=v
    for p in sorted(args.archives.glob('*model_inputs.jsonl.gz')):
        with gzip.open(p,'rt') as f:
            for line in f:
                v=json.loads(line);prefix='bfcl' if v.get('benchmark','').startswith('bfcl') else 'gpqa'
                identity=prefix+':'+v['sample_id']
                if identity in wanted and identity not in rows: rows[identity]=v
    if not rows: raise ValueError('no selected request content available')
    start=time.monotonic();tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    print('loading BF16 checkpoint',flush=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cpu').eval()
    print('loaded',round(time.monotonic()-start,2),flush=True)
    captured={};handles=[]
    for layer in (2,13,26):
        def hook(module,inputs,layer=layer): captured[layer]=inputs[0].detach().cpu().clone()
        handles.append(model.model.layers[layer].mlp.register_forward_pre_hook(hook))
    phases=['development','heldout'] if args.split=='both' else [args.split]
    for phase in phases:
        phase_wanted=set(selection[phase])
        phase_output=args.output/phase if args.split=='both' else args.output
        phase_output.mkdir(parents=True,exist_ok=True)
        phase_target=args.eval_tokens if phase=='heldout' else args.target_tokens
        shards=[];count=0;records=[]
        if (phase_output/'manifest.json').exists():
            if not args.resume:raise ValueError('existing capture requires --resume or a new output directory')
            prior=json.loads((phase_output/'manifest.json').read_text())
            for row in prior['shards']:
                saved=Path(row['path'])
                if hashlib.sha256(saved.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('existing shard hash mismatch')
                if row['request_id'] not in phase_wanted:raise ValueError('resume selection mismatch')
                records.append(row);shards.append(saved);count+=row['tokens']
        done={row['request_id'] for row in records}
        if count>=phase_target:continue
        for identity,v in sorted(((key,row) for key,row in rows.items() if key in phase_wanted),key=lambda kv:hashlib.sha256(kv[0].encode()).hexdigest()):
            if identity in done:continue
            messages=v['messages']
            if v.get('function_schema'): messages=[{'role':'system','content':'You may use these functions: '+json.dumps(v['function_schema'],ensure_ascii=False)}]+messages
            ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
            full_tokens=ids.shape[1]
            if args.smoke_tokens: ids=ids[:,:args.smoke_tokens]
            else: ids=ids[:,:min(full_tokens,4096,phase_target-count)]
            # Whole prompt is forwarded once; selected stored positions are its continuous prefix.
            n=min(ids.shape[1],phase_target-count);stamp=time.monotonic();captured.clear()
            with torch.inference_mode(): model.model(input_ids=ids,use_cache=False)
            elapsed=time.monotonic()-stamp;request_sha=hashlib.sha256(identity.encode()).hexdigest(); shard=phase_output/f'capture_{len(shards):04d}.npz';payload={}
            for layer,x in captured.items():
                x=x[:,:n,:].reshape(n,-1)
                with torch.inference_mode():
                    result=model.model.layers[layer].mlp.gate(x[None,:,:])
                routes,scores=result[:2]
                payload[f'x_l{layer}']=x.float().numpy();payload[f'routes_l{layer}']=routes.reshape(n,-1).numpy();payload[f'gates_l{layer}']=scores.reshape(n,-1).float().numpy()
            payload['token_positions']=np.arange(n);np.savez(shard,**payload);raw=shard.read_bytes()
            r={'path':str(shard.resolve()),'sha256':hashlib.sha256(raw).hexdigest(),'request_id':identity,'request_sha256':request_sha,
               'identity_sha256':request_sha,'request_sha256_semantics':'legacy identity-string SHA; not a content hash',
               'canonical_record_sha256':hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
               'forward_input_token_ids_sha256':hashlib.sha256(ids.cpu().numpy().astype('<i8').tobytes()).hexdigest(),
               'tokens':n,'full_prompt_tokens':full_tokens,'positions':[0,n],'continuous':True,'prefix_truncated_smoke':bool(args.smoke_tokens),'prefix_truncated_for_budget':n<full_tokens,'forward_seconds':elapsed}
            records.append(r);shards.append(shard);count+=n
            manifest={'schema':'plena_moe_v3_numeric_capture_v1','model':str(args.model.resolve()),'layers':[2,13,26],
                      'split':phase,'token_count':count,'required_tokens':phase_target,'complete':count>=phase_target and not args.smoke_tokens,
                      'origin':'real checkpoint BF16 forward; canonical archived messages re-tokenized; CPU','cuda_available':torch.cuda.is_available(),
                      'canonical_request_sources':sources,'selection_sha256':hashlib.sha256(args.selection.read_bytes()).hexdigest(),
                      'model_config_sha256':hashlib.sha256((args.model/'config.json').read_bytes()).hexdigest(),'shards':records,'seconds':time.monotonic()-start}
            (phase_output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(identity,n,'tokens',round(elapsed,3),'s',round(n/elapsed,3),'token/s',flush=True)
            if args.smoke_tokens or count>=phase_target: break
    for h in handles:h.remove()

if __name__=='__main__':main()
