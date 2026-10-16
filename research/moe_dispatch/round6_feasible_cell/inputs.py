"""Reuse frozen R5 windows; append distinct real post-router groups for B4096."""
import copy,hashlib,json
import numpy as np
from .common import ROOT,R5,read_json,write_json,write_csv,digest,BATCHES,gm
def workload_summary(data):
 rows=[]
 for split,ws in data.items():
  for b in BATCHES:
   ww=[w for w in ws if w['batch']==b];ee=[[e for e in w['experts'] if not e['is_shared']] for w in ww]
   rows.append(dict(split=split,batch=b,windows=len(ww),routed_mean_tokens=6*b/64,
    active_experts_mean=sum(len(es) for es in ee)/len(ee),shared_useful_work_pct=25,
    cold_useful_work_pct=100*sum(sum(e['Me'] for e in es if e['Me']<=2) for es in ee)/(len(ee)*8*b),
    max_routed_tokens_mean=sum(max(e['Me'] for e in es) for es in ee)/len(ee),
    unique_weight_MiB_GM=gm([33+16.5*len(es) for es in ee])))
 write_csv(ROOT/'E1/workload.csv',rows)
def build():
 data=read_json(R5/'inputs/windows.json');out=copy.deepcopy(data);summary=[]
 for split,ws in data.items():
  groups=sorted({w['group'] for w in ws})
  for w in out[split]:w['source_groups']=[w['group']]
  for i,g in enumerate(groups):
   g2=groups[(i+1)%len(groups)]
   for layer in (2,13,26):
    a=next(w for w in ws if w['group']==g and w['layer']==layer and w['batch']==2048)
    b=next(w for w in ws if w['group']==g2 and w['layer']==layer and w['batch']==2048)
    assert not set(a['source_requests'])&set(b['source_requests'])
    routes=np.array(a['token_routes']+b['token_routes'],dtype=np.uint8);h=np.bincount(routes.ravel(),minlength=64)
    es=[dict(id=e,Me=int(m),H=2048,F=1408,is_shared=False) for e,m in enumerate(h) if m]
    es.append(dict(id=64,Me=4096,H=2048,F=2816,is_shared=True))
    out[split].append(dict(id=f'{split}_{g}+{g2}_l{layer}_b4096',split=split,batch=4096,layer=layer,
      group=f'{g}+{g2}',source_groups=[g,g2],dataset=a['dataset'] if a['dataset']==b['dataset'] else 'bfcl+gpqa',
      experts=es,token_routes=routes.tolist(),source_requests=a['source_requests']+b['source_requests'],
      routing_sha256=hashlib.sha256(routes.tobytes()).hexdigest(),
      scope='Concatenation AFTER routing of two distinct original2048-token groups; no repeated token records; not a new attention forward.',
      source_windows=[a['id'],b['id']]))
  assert len(out[split])==105
  for w in out[split]:
   h=[e['Me'] for e in w['experts'] if not e['is_shared']]
   assert sum(h)==6*w['batch']
   summary.append(dict(split=split,window_id=w['id'],batch=w['batch'],layer=w['layer'],group=w['group'],
    source_groups=json.dumps(w['source_groups']),active=len(h),cold=sum(m<=2 for m in h),
    cold_work_pct=100*sum(m for m in h if m<=2)/(8*w['batch']),shared_work_pct=25,max_Me=max(h)))
 assert not {r for w in out['development'] for r in w['source_requests']} & {r for w in out['heldout'] for r in w['source_requests']}
 write_json(ROOT/'inputs/windows.json.gz',out);write_csv(ROOT/'inputs/windows.csv',summary)
 write_json(ROOT/'inputs/manifest.json',dict(round5_commit='b095a7de67a0be07802df0fb1fa2c2b3063bdc75',
  round5_input_sha256=hashlib.sha256((R5/'inputs/windows.json').read_bytes()).hexdigest(),digest=digest(out),
  counts={s:len(v) for s,v in out.items()},batches=BATCHES,windows_per_batch=15,
  b4096='Distinct original request groups concatenated only after real routing. Source groups overlap across windows; bootstrap original groups.'))
 workload_summary(out)
 return out
def load():
 p=ROOT/'inputs/windows.json.gz'
 return read_json(p) if p.exists() else build()
if __name__=='__main__':build()
