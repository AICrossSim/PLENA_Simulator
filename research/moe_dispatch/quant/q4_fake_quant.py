#!/usr/bin/env python3
"""GPU-only whole-model fake-quant perplexity using exported frozen factor cache.

Refuses to silently omit missing expert factor files. --rank0 is an explicit RTN
baseline. Downstream task evaluation requires its separate benchmark harness.
"""
import argparse,json,math,sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v3_reference'))
import ref_numerics as rn


class CompensatedLinear(torch.nn.Module):
    def __init__(self,layer,wq,A,B):
        super().__init__();self.register_buffer('weight',wq.to(layer.weight.device,layer.weight.dtype))
        self.register_buffer('A',torch.as_tensor(A,device=layer.weight.device,dtype=layer.weight.dtype));self.register_buffer('B',torch.as_tensor(B,device=layer.weight.device,dtype=layer.weight.dtype))
    def forward(self,x):
        y=torch.nn.functional.linear(x,self.weight)
        if self.A.shape[1]:y=y+(x@self.A)@self.B
        return y


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--eval-text',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--factor-cache',type=Path);ap.add_argument('--rank0',action='store_true');ap.add_argument('--bits',type=int,default=4);ap.add_argument('--max-length',type=int,default=2048);a=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Q4 requested but no CUDA device is available')
    if not a.rank0 and a.factor_cache is None:raise ValueError('frozen all-layer factor cache is required')
    model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='auto').eval();tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    texts=[json.loads(line)['text'] if line.lstrip().startswith('{') else line.strip() for line in a.eval_text.read_text().splitlines() if line.strip()]
    def ppl():
        loss,tokens=0.,0
        with torch.inference_mode():
            for text in texts:
                ids=tok(text,return_tensors='pt',truncation=True,max_length=a.max_length).input_ids.to(model.device)
                if ids.shape[1]<2:continue
                r=model(input_ids=ids,labels=ids,use_cache=False);loss+=float(r.loss)*(ids.shape[1]-1);tokens+=ids.shape[1]-1
        if not tokens:raise ValueError('no evaluation tokens')
        return math.exp(loss/tokens),tokens
    baseline,n=ppl();count=0
    for name,module in list(model.named_modules()):
        if not isinstance(module,torch.nn.Linear) or '.mlp.' not in name or not name.endswith(('gate_proj','up_proj','down_proj')):continue
        W=module.weight.detach().float().cpu().numpy();wq=torch.from_numpy(rn.mx_quantize(W,a.bits)[2]);K,N=W.shape[1],W.shape[0]
        if a.rank0:A=np.zeros((K,0),np.float32);B=np.zeros((0,N),np.float32)
        else:
            p=a.factor_cache/(name.replace('.','_')+'.npz')
            if not p.exists():raise FileNotFoundError('Q4 factor omitted: '+str(p))
            with np.load(p) as z:A,B=z['A'],z['B']
        parent=model
        path=name.split('.')
        for part in path[:-1]:parent=getattr(parent,part)
        setattr(parent,path[-1],CompensatedLinear(module,wq,A,B));count+=1
    quantized,_=ppl();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'baseline_ppl':baseline,'quantized_ppl':quantized,'relative_increase':quantized/baseline-1,'tokens':n,'quantized_linears':count,'bits':a.bits,'rank0':a.rank0},indent=2)+'\n')

if __name__=='__main__':main()
