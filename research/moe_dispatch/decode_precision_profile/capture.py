"""Real autoregressive batches: all 26 MoE layers, exact IDs/scores/inputs.

Prefill is never counted as decode. Batch is concurrent independent requests.
Greedy generation is continued for a fixed 32 steps, including after EOS;
this is explicitly a fixed-population profiling workload, not serving latency.
"""
from __future__ import annotations
import argparse, gc, json, time, types
from pathlib import Path
import numpy as np
import torch
from .common import *

SOURCES = {'bfcl': 'bfcl_v3_stratified_512.jsonl',
           'gpqa': 'gpqa_diamond.jsonl', 'swe': 'swe_bench_full_test_2294.jsonl'}

def select(tokenizer, source, max_tokens, batches, cohorts):
    rows = [json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    unique = {}
    for row in rows: unique.setdefault(str(row['sample_id']), row)
    rows = sorted(unique.values(), key=lambda r: canonical_sha(['decode_profile_20261016', r['sample_id']]))
    selected = []
    for row in rows[:max(batches)*cohorts]:
        messages = list(row['messages'])
        tools = row.get('tools') or row.get('function_schema')
        if tools:
            messages = [{'role':'system', 'content':'Available functions: '+json.dumps(tools, ensure_ascii=False)}] + messages
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if hasattr(ids, 'keys'): ids = ids['input_ids']
        if torch.is_tensor(ids): ids = ids.reshape(-1).tolist()
        if ids and isinstance(ids[0], list): ids = ids[0]
        full = len(ids)
        if full > max_tokens:
            head = max_tokens//4; ids = ids[:head]+ids[-(max_tokens-head):]
        selected.append(dict(request_id=str(row['sample_id']), record_sha256=canonical_sha(row),
            full_prompt_tokens=full, prompt_tokens=len(ids), input_ids=ids,
            truncation='head_quarter+tail_three_quarters' if full > max_tokens else 'none'))
    return selected

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True); p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--dtype', choices=['bf16','fp16'], default='bf16')
    p.add_argument('--batches', default=','.join(map(str,BATCHES))); p.add_argument('--datasets',default='bfcl,gpqa,swe')
    p.add_argument('--cohorts',type=int,default=2); p.add_argument('--steps',type=int,default=32)
    p.add_argument('--max-prompt',type=int,default=1024); p.add_argument('--resume',action='store_true')
    p.add_argument('--expert-precision',choices=['native','fp8_w8a8','int8_w8a8','int8_w8a16','int4_w4a16'],default='native')
    a = p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    batches = tuple(map(int,a.batches.split(','))); dtype = torch.bfloat16 if a.dtype=='bf16' else torch.float16
    torch.set_num_threads(4); torch.manual_seed(20261016)
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    plan=[]; skips=[]; sources=[]
    for dataset in a.datasets.split(','):
        path=a.sources/SOURCES[dataset]; sources.append(dict(dataset=dataset,path=str(path),sha256=sha(path)))
        selected=select(tokenizer,path,a.max_prompt,batches,a.cohorts)
        for b in batches:
            for cohort in range(a.cohorts):
                items=selected[cohort*b:(cohort+1)*b]
                if len(items)!=b:
                    skips.append(dict(dataset=dataset,batch=b,cohort=cohort,reason='insufficient distinct source requests'));continue
                plan.append(dict(id=f'{dataset}_b{b}_c{cohort}',dataset=dataset,batch=b,cohort=cohort,requests=items))
    manifest=dict(schema='fresh_decode_all_layers_v1',dtype=a.dtype,layers=list(LAYERS),steps=a.steps,
        expert_precision=a.expert_precision,
        profile_steps=[s for s in PROFILE_STEPS if s<=a.steps],batches=list(batches),sources=sources,
        model=str(a.model),model_config_sha256=sha(a.model/'config.json'),
        model_index_sha256=sha(a.model/'model.safetensors.index.json'),
        torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
        population='native concurrent-request batches; one generated input token/request/decode step',
        eos_policy='fixed-length greedy rollout; continue after EOS, positions recorded',
        prefill='separate phase; excluded from decode routes and profile inputs',
        max_prompt_tokens=a.max_prompt,planned_groups=plan,skipped_groups=skips,shards=[],complete=False)
    receipt=a.out/'capture_manifest.json'
    if receipt.exists():
        if not a.resume: raise ValueError('immutable output; use a new destination or resume')
        old=json.loads(receipt.read_text())
        for k in ('dtype','layers','steps','profile_steps','planned_groups','sources','max_prompt_tokens','expert_precision'):
            if old.get(k,'native' if k=='expert_precision' else None)!=manifest[k]: raise ValueError('resume plan mismatch: '+k)
        manifest['shards']=old['shards']
        for s in manifest['shards']:
            if sha(a.out/s['file'])!=s['sha256']: raise ValueError('changed completed shard')
    write_json(receipt,manifest)
    print('LOAD',a.dtype,len(plan),'groups',flush=True)
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,
        dtype=dtype,device_map={'':0},attn_implementation='sdpa').eval()
    layers={i:model.model.layers[i].mlp for i in LAYERS}
    state={'step':0}; captured={}; inputs={}; outputs={}; prefill={}; patch_checks=[]
    # Keep the native attention/router/Shared code. Replace only routed expert
    # execution with vLLM's unquantized grouped implementation, validating every
    # layer against its original expert-loop on a deterministic numeric fixture.
    with torch.inference_mode():
        for li,moe in layers.items():
            ex=moe.experts; x=torch.randn(2,H,device='cuda',dtype=dtype)*.1
            _,score,ids=moe.gate(x)
            reference=ex(x,ids,score)
            got=fused_experts(x,ex.gate_up_proj,ex.down_proj,score,ids,inplace=False)
            rel=float((got.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8))
            if not torch.isfinite(got).all() or rel>.04: raise ValueError(f'fused implementation mismatch layer {li}: {rel}')
            patch_checks.append(dict(layer=li,relative_l2=rel))
            def forward(self,x,ids,scores):
                return fused_experts(x,self.gate_up_proj,self.down_proj,scores,ids,inplace=False)
            ex.forward=types.MethodType(forward,ex)
            if a.expert_precision!='native':
                from .precision import ExpertWeights
                rw=ExpertWeights((ex.gate_up_proj,ex.down_proj),a.expert_precision)
                sw=ExpertWeights((torch.cat((moe.shared_experts.gate_proj.weight,moe.shared_experts.up_proj.weight),dim=0)[None],moe.shared_experts.down_proj.weight[None]),a.expert_precision)
                def quant_moe(self, hidden_states, rw=rw, sw=sw):
                    shape=hidden_states.shape;x=hidden_states.reshape(-1,H)
                    _,scores,ids=self.gate(hidden_states)
                    y=rw(x,ids,scores)
                    sy=sw(x,torch.zeros((len(x),1),device=x.device,dtype=torch.int64),torch.ones((len(x),1),device=x.device))
                    return (y+sy).view(shape)
                moe.forward=types.MethodType(quant_moe,moe)
            def hook(module,args,result,li=li):
                _,scores,ids=result; b=state['batch']
                if state['step']==0:
                    # Mask left padding when accounting prefill routes.
                    valid=state['mask'].bool().reshape(-1)
                    prefill[li]=torch.bincount(ids[valid].reshape(-1),minlength=64).cpu().numpy()
                    return
                if ids.shape!=(b,6):raise ValueError('not a pure one-token decode call')
                captured[li].append((ids.detach().cpu().numpy().astype(np.uint8), scores.detach().float().cpu().numpy()))
                if state['step'] in manifest['profile_steps']:
                    x=args[0].detach().reshape(b,H)
                    if not torch.isfinite(x).all():raise ValueError(f'nonfinite input layer {li}')
                    inputs[li].append(x.contiguous().view(torch.uint16).cpu().numpy())
            moe.gate.register_forward_hook(hook)
            def output_hook(module,args,result,li=li):
                if state['step'] in manifest['profile_steps']:
                    y=result.detach().reshape(state['batch'],H)
                    if not torch.isfinite(y).all():raise ValueError(f'nonfinite output layer {li}')
                    outputs[li].append(y.contiguous().view(torch.uint16).cpu().numpy())
            moe.register_forward_hook(output_hook)
    manifest['fused_expert_numeric_checks']=patch_checks;write_json(receipt,manifest)
    completed={s['id'] for s in manifest['shards']}
    for group in plan:
        if group['id'] in completed:continue
        started=time.monotonic();b=group['batch']; state.update(step=0,batch=b)
        width=max(r['prompt_tokens'] for r in group['requests']);pad=tokenizer.pad_token_id or tokenizer.eos_token_id
        ids=torch.full((b,width),pad,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
        for j,r in enumerate(group['requests']):
            n=len(r['input_ids']);ids[j,-n:]=torch.tensor(r['input_ids'],device='cuda');mask[j,-n:]=1
        state['mask']=mask;captured={li:[] for li in LAYERS};inputs={li:[] for li in LAYERS};outputs={li:[] for li in LAYERS};prefill={}
        generated=[];consumed=[];eos_positions=[];cache=None
        with torch.inference_mode():
            positions=(mask.cumsum(-1)-1).clamp_min(0)
            result=model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=True,logits_to_keep=1)
            cache=result.past_key_values;token=result.logits[:,-1].argmax(-1,keepdim=True);generated.append(token[:,0].cpu().numpy())
            for step in range(1,a.steps+1):
                state['step']=step;consumed.append(token[:,0].cpu().numpy())
                mask=torch.cat((mask,torch.ones((b,1),device='cuda',dtype=mask.dtype)),dim=1)
                position=mask.sum(-1,keepdim=True)-1
                result=model(input_ids=token,attention_mask=mask,position_ids=position,past_key_values=cache,use_cache=True,logits_to_keep=1)
                if not torch.isfinite(result.logits).all():raise ValueError('nonfinite generation logits')
                cache=result.past_key_values;token=result.logits[:,-1].argmax(-1,keepdim=True);generated.append(token[:,0].cpu().numpy())
        arrays=dict(layers=np.array(LAYERS),profile_steps=np.array(manifest['profile_steps']),
            generated_tokens=np.stack(generated),decode_input_tokens=np.stack(consumed),
            prompt_lengths=np.array([r['prompt_tokens'] for r in group['requests']]))
        for li in LAYERS:
            ri=np.stack([t[0] for t in captured[li]]);sc=np.stack([t[1] for t in captured[li]])
            validate_routes(ri,sc)
            arrays[f'ids_l{li}']=ri;arrays[f'scores_l{li}']=sc;arrays[f'x_l{li}']=np.stack(inputs[li]);arrays[f'y_l{li}']=np.stack(outputs[li]);arrays[f'prefill_counts_l{li}']=prefill[li]
        path=a.out/(group['id']+'.npz');np.savez_compressed(path,**arrays)
        shard=dict(id=group['id'],dataset=group['dataset'],batch=b,cohort=group['cohort'],file=path.name,
            sha256=sha(path),seconds=time.monotonic()-started,decode_windows=a.steps*len(LAYERS),
            profile_windows=len(manifest['profile_steps'])*len(LAYERS),max_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
        manifest['shards'].append(shard);write_json(receipt,manifest)
        print('CAPTURE',json.dumps(shard),flush=True)
        del result,cache,ids,mask,token;gc.collect();torch.cuda.empty_cache()
    manifest['complete']=len(manifest['shards'])==len(plan);write_json(receipt,manifest)
    print('CAPTURE COMPLETE',len(plan),flush=True)

if __name__=='__main__':main()
