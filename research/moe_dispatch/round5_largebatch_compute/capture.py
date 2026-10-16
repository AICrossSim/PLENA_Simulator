"""Fresh real BF16 routes. No fabricated, repeated, or resampled routing rows."""
import argparse, gzip, hashlib, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = Path('/scratch/shared/mcl123/plena/weights/deepseek-v2-lite-chat')
ARCHIVES = Path('/scratch/shared/mcl123/plena/paper_artifacts')
LAYERS = (2, 13, 26)

def sha(data): return hashlib.sha256(data).hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def select(tok):
    """Four long BFCL requests plus a real multi-request GPQA pack per split.

    Identity-hash split/ordering, then token-length qualification, before any
    routes or timing results are accessed. GPQA records are independent model
    forwards; only post-router records are packed into one MoE token batch.
    """
    out=[]; sources=[]
    for dataset,filename in [('bfcl','bfcl_v3_model_inputs.jsonl.gz'),('gpqa','gpqa_diamond_model_inputs.jsonl.gz')]:
        path=ARCHIVES/filename;sources.append({'path':str(path),'sha256':sha(path.read_bytes())})
        with gzip.open(path,'rt') as f: records=[json.loads(line) for line in f]
        records.sort(key=lambda r:sha(('round5-order:'+dataset+':'+r['sample_id']).encode()))
        for split in ('development','heldout'):
            count=0; ntokens=0;selected_ids=set()
            for r in records:
                ident=dataset+':'+r['sample_id']
                if ident in selected_ids:continue
                assigned='development' if int(sha(('round5-split:'+ident).encode()),16)%2==0 else 'heldout'
                if assigned!=split:continue
                messages=r['messages']
                if r.get('function_schema'):
                    messages=[{'role':'system','content':'You may use these functions: '+json.dumps(r['function_schema'],ensure_ascii=False)}]+messages
                ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                if dataset=='bfcl' and len(ids)<2048:continue
                n=2048 if dataset=='bfcl' else min(len(ids),2048-ntokens)
                out.append({'request_id':ident,'dataset':dataset,'split':split,
                            'group':f'{dataset}_{count}' if dataset=='bfcl' else 'gpqa_pack',
                            'canonical_record_sha256':sha(canonical(r)),
                            'full_prompt_tokens':len(ids),'input_ids':ids[:n],
                            'tokens':n,'positions':[0,n]})
                count+=1;ntokens+=n
                selected_ids.add(ident)
                if (dataset=='bfcl' and count==4) or (dataset=='gpqa' and ntokens==2048):break
            assert (count==4 if dataset=='bfcl' else ntokens==2048)
    assert not ({r['request_id'] for r in out if r['split']=='development'} &
                {r['request_id'] for r in out if r['split']=='heldout'})
    assert len({r['request_id'] for r in out})==len(out)
    return {'schema':'round5_real_prefill_selection_v1','sources':sources,'requests':out,
            'selection_rule':'hash request split/order; four BFCL prompts >=2048 tokens per split; GPQA real independent prompt pack of2048 tokens per split',
            'not_decode_batch':True,'layers':LAYERS}

def main():
    import numpy as np
    import torch, transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    ap=argparse.ArgumentParser();ap.add_argument('--threads',type=int,default=4);ap.add_argument('--select-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    out=ROOT/'inputs';out.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False)
    selection_path=out/'selection.json'
    if selection_path.exists(): selection=json.loads(selection_path.read_text())
    else:selection=select(tok);write(selection_path,selection)
    assert len({r['request_id'] for r in selection['requests']})==len(selection['requests'])
    print('selected',len(selection['requests']),'requests',sum(r['tokens'] for r in selection['requests']),'tokens',flush=True)
    if args.select_only:return
    print('loading original BF16 checkpoint',flush=True);start=time.monotonic()
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=False,
            dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cpu').eval()
    print('loaded seconds',time.monotonic()-start,flush=True)
    captured={};handles=[]
    for layer in LAYERS:
        def hook(module,inputs,result,layer=layer):
            captured[layer]=(result[0].detach().cpu().numpy().astype(np.uint8),result[1].detach().float().cpu().numpy())
        handles.append(model.model.layers[layer].mlp.gate.register_forward_hook(hook))
    receipt_path=out/'capture_manifest.json'
    receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'schema':'round5_actual_BF16_forward_routes_v1','selection_sha256':sha(selection_path.read_bytes()),
        'torch':torch.__version__,'transformers':transformers.__version__,
        'model_config_sha256':sha((MODEL/'config.json').read_bytes()),
        'model_index_sha256':sha((MODEL/'model.safetensors.index.json').read_bytes()),
        'device':'CPU','shards':[],'complete':False}
    completed={r['request_id']:r for r in receipt['shards']}
    for index,r in enumerate(selection['requests']):
        name=f'route_{index:03d}.npz';path=out/name
        if r['request_id'] in completed:
            assert sha(path.read_bytes())==completed[r['request_id']]['sha256'];continue
        ids=torch.tensor([r['input_ids']],dtype=torch.long);captured.clear();t=time.monotonic()
        with torch.inference_mode():model.model(input_ids=ids,use_cache=False)
        assert set(captured)==set(LAYERS)
        data={}
        for layer,(routes,gates) in captured.items():
            routes=routes.reshape(r['tokens'],6);gates=gates.reshape(r['tokens'],6)
            assert (np.diff(np.sort(routes,axis=1),axis=1)>0).all()
            data[f'routes_l{layer}']=routes;data[f'gates_l{layer}']=gates
        np.savez_compressed(path,**data)
        row={k:v for k,v in r.items() if k!='input_ids'}
        row.update(file=name,sha256=sha(path.read_bytes()),input_ids_sha256=sha(ids.numpy().astype('<i8').tobytes()),forward_seconds=time.monotonic()-t,cpu_threads=args.threads)
        receipt['shards'].append(row);write(receipt_path,receipt)
        print(index,r['request_id'],r['tokens'],'tokens',round(row['forward_seconds'],2),'seconds',flush=True)
    receipt['complete']=True;write(receipt_path,receipt)
    print('capture complete',flush=True)

if __name__=='__main__':main()
