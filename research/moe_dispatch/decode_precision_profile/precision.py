"""Explicit vLLM fused-MoE precision paths, using real checkpoint weights.

FP8/INT8 use dynamic per-token activation quantization. W4A16/W8A16 retain
BF16 activations and dequantize packed weights inside the GEMM kernel.
These are simple RTN quantizers, not calibrated AWQ/GPTQ model releases.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from safetensors import safe_open
from .common import *

PRECISIONS=('bf16','fp16','fp8_w8a8','int8_w8a8','int8_w8a16','int4_w4a16')

def layer_weights(model, layer):
    model=Path(model); index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    def get(name):
        with safe_open(model/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name)
    p=f'model.layers.{layer}.mlp.'
    gu=[]; down=[]
    for e in range(64):
        prefix=p+f'experts.{e}.'
        gu.append(torch.cat([get(prefix+'gate_proj.weight'),get(prefix+'up_proj.weight')],dim=0))
        down.append(get(prefix+'down_proj.weight'))
    shared_gu=torch.cat([get(p+'shared_experts.gate_proj.weight'),get(p+'shared_experts.up_proj.weight')],dim=0)[None]
    shared_down=get(p+'shared_experts.down_proj.weight')[None]
    return (torch.stack(gu).cuda(),torch.stack(down).cuda()),(shared_gu.cuda(),shared_down.cuda())

def quant_weight(w, precision):
    if precision in ('bf16','fp16'):
        return w.to(torch.bfloat16 if precision=='bf16' else torch.float16).contiguous(),None
    if precision in ('fp8_w8a8','int8_w8a8'):
        qdtype=torch.float8_e4m3fn if precision.startswith('fp8') else torch.int8
        bound=448. if precision.startswith('fp8') else 127.
        scale=(w.float().abs().amax(-1,keepdim=True)/bound).clamp_min(1e-12)
        values=(w.float()/scale).clamp(-bound,bound)
        if qdtype==torch.int8:values=values.round()
        return values.to(qdtype).contiguous(),scale.contiguous()
    bits=4 if precision=='int4_w4a16' else 8
    group=128; shape=w.shape; wg=w.float().reshape(*shape[:-1],shape[-1]//group,group)
    # The kernel interprets unsigned storage with an implicit midpoint zero.
    bound=2**(bits-1)-1;mid=2**(bits-1)
    scale=(wg.abs().amax(-1)/bound).clamp_min(1e-12).to(torch.bfloat16)
    q=(wg/scale.float()[...,None]).round().clamp(-mid,bound).add(mid).to(torch.uint8).reshape(shape)
    if bits==4:q=q[...,::2] | (q[...,1::2]<<4)
    return q.contiguous(),scale.contiguous()

class ExpertWeights:
    def __init__(self, pair, precision):
        self.precision=precision;self.w1,self.s1=quant_weight(pair[0],precision);self.w2,self.s2=quant_weight(pair[1],precision)
        self.kw={}
        if precision=='fp8_w8a8':self.kw=dict(use_fp8_w8a8=True,per_channel_quant=True)
        if precision=='int8_w8a8':self.kw=dict(use_int8_w8a8=True,per_channel_quant=True)
        if precision=='int8_w8a16':self.kw=dict(use_int8_w8a16=True,block_shape=[0,128])
        if precision=='int4_w4a16':self.kw=dict(use_int4_w4a16=True,block_shape=[0,128])
        if self.s1 is not None:self.kw.update(w1_scale=self.s1,w2_scale=self.s2)
    def __call__(self,x,ids,scores,expert_map=None):
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl
        return fused_experts_impl(x,self.w1,self.w2,scores,ids,inplace=False,
            global_num_experts=self.w1.shape[0],expert_map=expert_map,**self.kw)
    def storage_bytes(self):
        return sum(v.numel()*v.element_size() for v in (self.w1,self.w2,self.s1,self.s2) if v is not None)

class Runner:
    def __init__(self,routed,shared,x,ids,scores):
        self.routed_weights,self.shared_weights=routed,shared
        self.x=x.to(torch.float16 if routed.precision=='fp16' else torch.bfloat16).contiguous()
        self.ids=torch.as_tensor(ids.astype('int64'),device='cuda');self.scores=torch.as_tensor(scores,device='cuda')
        b=len(ids); counts=torch.bincount(self.ids.reshape(-1),minlength=64).cpu().tolist()
        self.hot=any(c>2 for c in counts);self.cold=any(0<c<=2 for c in counts)
        self.maps={kind:torch.tensor([e if (c>2 if kind=='hot' else 0<c<=2) else -1 for e,c in enumerate(counts)],device='cuda',dtype=torch.int32) for kind in ('hot','cold')}
        self.shared_ids=torch.zeros((b,1),device='cuda',dtype=torch.int64);self.shared_scores=torch.ones((b,1),device='cuda')
        self.parts={};self.output=None
    def shared(self):self.parts['shared']=self.shared_weights(self.x,self.shared_ids,self.shared_scores)
    def routed(self,kind):self.parts[kind]=self.routed_weights(self.x,self.ids,self.scores,None if kind=='all' else self.maps[kind])
    def combine(self,path):
        self.output=self.parts['shared']+self.parts['all'] if path=='grouped' else self.parts['shared']+(self.parts['hot'] if self.hot else 0)+(self.parts['cold'] if self.cold else 0)
    def stages(self,path):
        if path=='grouped':return [('shared',self.shared),('routed',lambda:self.routed('all')),('combine',lambda:self.combine(path))]
        return [('shared',self.shared)]+([('hot',lambda:self.routed('hot'))] if self.hot else [])+([('cold',lambda:self.routed('cold'))] if self.cold else [])+[('combine',lambda:self.combine(path))]
    def run(self,path):
        for _,fn in self.stages(path):fn()
        return self.output

def error_metrics(y,ref):
    y=y.float();ref=ref.float();err=y-ref
    return dict(finite=bool(torch.isfinite(y).all()),relative_l2=float(err.norm()/ref.norm().clamp_min(1e-12)),
        max_abs_error=float(err.abs().max()),cosine=float(torch.nn.functional.cosine_similarity(y.reshape(1,-1),ref.reshape(1,-1)).item()))

def measure(runner,path,warmup=10,samples=50):
    if warmup<10 or samples<50:raise ValueError('need >=10 warmup and >=50 samples')
    for _ in range(warmup):runner.run(path)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):runner.run(path)
    for _ in range(warmup):graph.replay()
    begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    e2e=[]
    for _ in range(samples):
        begin.record();graph.replay();end.record();end.synchronize();e2e.append(begin.elapsed_time(end))
    output=runner.output.clone()
    stages=runner.stages(path);events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(len(stages)+1)]
    marked=torch.cuda.CUDAGraph()
    with torch.cuda.graph(marked):
        events[0].record()
        for i,(_,fn) in enumerate(stages):fn();events[i+1].record()
    for _ in range(warmup):marked.replay()
    parts={n:[] for n in ('shared','hot','cold','routed','combine')};totals=[]
    for _ in range(samples):
        marked.replay();events[-1].synchronize();vals={n:events[i].elapsed_time(events[i+1]) for i,(n,_) in enumerate(stages)}
        t=events[0].elapsed_time(events[-1]);totals.append(t)
        if abs(sum(vals.values())-t)>max(0.001,t*1e-4):raise ValueError('stage accounting does not close')
        for n in parts:parts[n].append(vals.get(n,0.))
    torch.cuda.synchronize()
    err=error_metrics(runner.output,output)
    # WNA16 replay showed small output variation in the integration probe.
    # Record it under an explicit numeric tolerance, not bitwise equivalence.
    tolerance=.02 if runner.routed_weights.precision in ('int4_w4a16','int8_w8a16') else 1e-5
    if not err['finite'] or err['relative_l2']>tolerance:raise ValueError('instrumentation changed output: '+str(err))
    return dict(e2e_ms=e2e,instrumented_ms=totals,stage_ms=parts,
        instrumentation_output_relative_l2=err['relative_l2']),output

class GraphBench:
    """Reuse graph structure; update real input/route buffers between windows.

    Kernel specialization depends on shapes and presence of categories only.
    Expert IDs, scores, activation values and masks remain device inputs.
    """
    def __init__(self, runner, path, warmup):
        self.runner,self.path=runner,path
        for _ in range(warmup):runner.run(path)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):runner.run(path)
        self.plain_output=runner.output
        self.stages=runner.stages(path)
        self.events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(len(self.stages)+1)]
        self.marked=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.marked):
            self.events[0].record()
            for i,(_,fn) in enumerate(self.stages):fn();self.events[i+1].record()
        self.marked_output=runner.output
        self.begin,self.end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    def update(self,x,ids,scores):
        r=self.runner;r.x.copy_(x);r.ids.copy_(torch.as_tensor(ids.astype('int64'),device='cuda'));r.scores.copy_(torch.as_tensor(scores,device='cuda'))
        counts=torch.bincount(r.ids.reshape(-1),minlength=64)
        if bool((counts>2).any())!=r.hot or bool(((counts>0)&(counts<=2)).any())!=r.cold:raise ValueError('category graph mismatch')
        es=torch.arange(64,device='cuda',dtype=torch.int32)
        r.maps['hot'].copy_(torch.where(counts>2,es,-1));r.maps['cold'].copy_(torch.where((counts>0)&(counts<=2),es,-1))
    def measure(self,warmup=10,samples=50):
        if warmup<10 or samples<50:raise ValueError('insufficient timing samples')
        for _ in range(warmup):self.graph.replay()
        e2e=[]
        for _ in range(samples):
            self.begin.record();self.graph.replay();self.end.record();self.end.synchronize();e2e.append(self.begin.elapsed_time(self.end))
        output=self.plain_output.clone()
        for _ in range(warmup):self.marked.replay()
        parts={n:[] for n in ('shared','hot','cold','routed','combine')};totals=[]
        for _ in range(samples):
            self.marked.replay();self.events[-1].synchronize()
            vals={n:self.events[i].elapsed_time(self.events[i+1]) for i,(n,_) in enumerate(self.stages)}
            total=self.events[0].elapsed_time(self.events[-1]);totals.append(total)
            if abs(sum(vals.values())-total)>max(.001,total*1e-4):raise ValueError('stage closure failure')
            for n in parts:parts[n].append(vals.get(n,0.))
        err=error_metrics(self.marked_output,output)
        tolerance=.02 if self.runner.routed_weights.precision in ('int4_w4a16','int8_w8a16') else 1e-5
        if not err['finite'] or err['relative_l2']>tolerance:raise ValueError('instrumentation output discrepancy '+str(err))
        return dict(e2e_ms=e2e,instrumented_ms=totals,stage_ms=parts,instrumentation_output_relative_l2=err['relative_l2']),output
