#!/usr/bin/env python3
"""Capture actual layer-1 MoE operands using the local DeepSeek checkpoint.

Runs the COMPLETE prompt through embedding, decoder 0 and attention/norm of
decoder 1. Only the final prompt token is kept. This is last-token prefill, not
an autoregressive rollout or the older archived decode routing population.
"""
import json, hashlib, time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoTokenizer, DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2DecoderLayer,DeepseekV2RotaryEmbedding,DeepseekV2MoEGate)

ROOT=Path('/scratch/shared/mcl123/plena');OUT=ROOT/'outputs/moe_spatial_batch_real_20260921'
WEIGHTS=ROOT/'weights/deepseek-v2-lite-chat'
CACHE=Path('/tmp/plena-batch-real-operands-20260921');CACHE.mkdir(exist_ok=True)
torch.set_num_threads(6);torch.set_num_interop_threads(1)
index=json.loads((WEIGHTS/'model.safetensors.index.json').read_text())['weight_map']
provenance={}
def tensor(name):
    shard=WEIGHTS/index[name]
    with safe_open(shard,framework='pt',device='cpu') as f:v=f.get_tensor(name)
    assert v.dtype==torch.bfloat16
    raw=v.contiguous().view(torch.uint16).numpy().tobytes()
    provenance[name]=dict(shard=str(shard),shape=list(v.shape),dtype='BF16',sha256=hashlib.sha256(raw).hexdigest())
    return v

def raw_tensor(path,v):
    assert v.dtype==torch.bfloat16
    b=v.contiguous().view(torch.uint16).numpy().tobytes();path.write_bytes(b)
    return dict(path=str(path),sha256=hashlib.sha256(b).hexdigest())

def main():
    start=time.monotonic();config=DeepseekV2Config.from_pretrained(WEIGHTS,local_files_only=True)
    config._attn_implementation='sdpa'
    with torch.device('meta'):
        layers=[DeepseekV2DecoderLayer(config,i) for i in range(2)]
    layers[1].mlp=torch.nn.Identity()
    for i,layer in enumerate(layers):
        state={k:tensor(f'model.layers.{i}.{k}') for k in layer.state_dict()}
        layer.load_state_dict(state,strict=True,assign=True);layer.eval()
    embedding=tensor('model.embed_tokens.weight')
    rotary=DeepseekV2RotaryEmbedding(config).eval()
    gate=DeepseekV2MoEGate(config).eval()
    gate.load_state_dict({'weight':tensor('model.layers.1.mlp.gate.weight')},assign=True)
    tokenizer=AutoTokenizer.from_pretrained(WEIGHTS,local_files_only=True,trust_remote_code=False)
    examples=[json.loads(line) for line in (OUT/'inputs/BFCL_v3_simple.json').read_text().splitlines()][:16]
    captured=[];items=[]
    with torch.inference_mode():
        for number,row in enumerate(examples):
            # Tool schema is part of the model's actual input; no tool is executed.
            messages=[{'role':'system','content':'You may use these functions: '+json.dumps(row['function'],ensure_ascii=False)}]+row['question'][0]
            ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt')
            length=ids.shape[1];position=torch.arange(length).unsqueeze(0)
            h=torch.nn.functional.embedding(ids,embedding)
            pos=rotary(h,position)
            # SDPA is causally masked by the official attention implementation.
            h=layers[0](h,position_ids=position,position_embeddings=pos,use_cache=False)
            residual=h
            h,_=layers[1].self_attn(layers[1].input_layernorm(h),position_ids=position,position_embeddings=pos,use_cache=False)
            x=layers[1].post_attention_layernorm(residual+h)[:,-1:,:].contiguous()
            captured.append(x[0,0].clone())
            items.append(dict(sample_id=row['id'],input_tokens=length,input_token_ids=ids[0].tolist(),truncated=False))
            print(f'captured {row["id"]}, {length} tokens, {time.monotonic()-start:.1f}s',flush=True)
        x=torch.stack(captured);routes,weights=gate(x[:,None,:])
    assert x.shape==(16,2048) and routes.shape==(16,6)
    dst=OUT/'real_inputs';dst.mkdir(exist_ok=True)
    xfile=raw_tensor(dst/'x16.bf16',x)
    torch.save(dict(x=x,routes=routes,route_weights=weights),dst/'capture.pt')
    # Export only selected expert weights. Large temporary copies are disposable
    # and reconstructable from tensor SHA hashes in the permanent manifest.
    exported={}
    for e in sorted(set(routes.reshape(-1).tolist()))+['shared']:
        prefix=f'model.layers.1.mlp.experts.{e}' if e!='shared' else 'model.layers.1.mlp.shared_experts'
        for phase in ['gate','up','down']:
            name=f'{prefix}.{phase}_proj.weight';v=tensor(name)
            exported[f'{e}_{phase}']=raw_tensor(CACHE/f'{e}_{phase}.bf16',v)
    result=dict(model='deepseek-v2-lite-chat',model_layer=1,dtype='BF16',phase='last-token prefill',
        transformers_version=__import__('transformers').__version__,torch_version=torch.__version__,
        source_dataset=str(OUT/'inputs/BFCL_v3_simple.json'),examples=items,x=xfile,routes=routes.tolist(),
        route_weights=weights.tolist(),tensor_provenance=provenance,exported_weights=exported,
        prefix='embedding -> full decoder0 -> decoder1 attention + post_attention_norm',
        complete_model_inference=False,archived_decode_trace_reproduction=False,
        precision='BF16 weights/activations; standard model FP32 norm/router/softmax',host_seconds=time.monotonic()-start)
    (dst/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CAPTURE COMPLETE',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
