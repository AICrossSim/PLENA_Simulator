"""Freeze real post-router windows from request-disjoint new model captures."""
import hashlib,json
from collections import defaultdict
import numpy as np
from .common import ROOT,BATCHES,write_json,write_csv,digest

def build():
    p=ROOT/'inputs';manifest=json.loads((p/'capture_manifest.json').read_text())
    assert manifest['complete'],'capture not complete'
    groups=defaultdict(list);seen={'development':set(),'heldout':set()}
    for s in manifest['shards']:
        assert s['request_id'] not in seen[s['split']],'duplicate request in capture'
        path=p/s['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==s['sha256']
        seen[s['split']].add(s['request_id']);groups[s['split'],s['group']].append(s)
    assert not seen['development'] & seen['heldout']
    data={'development':[],'heldout':[]};summary=[]
    for (split,group),shards in sorted(groups.items()):
        for layer in (2,13,26):
            r=np.concatenate([np.load(p/s['file'])[f'routes_l{layer}'] for s in shards])
            assert r.shape==(2048,6)
            for b in BATCHES:
                # Fixed start independent of routing values/latency results.
                start=int(digest([split,group,layer,b]),16)%(2048-b+1)
                window=r[start:start+b];hist=np.bincount(window.ravel(),minlength=64)
                assert hist.sum()==6*b and max(hist)<=b
                experts=[dict(id=i,Me=int(m),H=2048,F=1408,is_shared=False) for i,m in enumerate(hist) if m]
                experts.append(dict(id=64,Me=b,H=2048,F=2816,is_shared=True))
                w=dict(id=f'{split}_{group}_l{layer}_b{b}',split=split,dataset=shards[0]['dataset'],
                    group=group,layer=layer,batch=b,experts=experts,
                    source_requests=[s['request_id'] for s in shards],positions=[start,start+b],
                    token_routes=window.tolist(),routing_sha256=hashlib.sha256(window.tobytes()).hexdigest(),
                    scope='real contiguous BFCL prefill segment' if group.startswith('bfcl') else 'post-router packing of real independent GPQA prefill requests')
                data[split].append(w)
                summary.append(dict(split=split,window_id=w['id'],dataset=w['dataset'],batch=b,layer=layer,
                    group=group,active=int((hist>0).sum()),cold=int(((hist>0)&(hist<=2)).sum()),
                    hot=int((hist>2).sum()),max_Me=int(max(hist)),mean_Me=float(hist.mean()),
                    cold_flops_pct=100*int(hist[(hist>0)&(hist<=2)].sum())/(8*b),
                    shared_flops_pct=25.0,useful_flops=2*sum(3*e['Me']*e['H']*e['F'] for e in experts)))
    write_json(p/'windows.json',data);write_csv(p/'window_statistics.csv',summary)
    write_json(p/'INPUT_MANIFEST.json',dict(capture_manifest_sha256=hashlib.sha256((p/'capture_manifest.json').read_bytes()).hexdigest(),
        input_digest=digest(data),windows={s:len(v) for s,v in data.items()},
        batches=BATCHES,requests_disjoint=True,group_count={s:len({w['group'] for w in v}) for s,v in data.items()},
        note='Prefill token batches, not2048 independent decode requests. Requests/layers recur across B; use group-cluster bootstrap. Timing heldout chosen by identity hash before new routes/search; requests may have appeared in historical unrelated studies.'))
    print({s:len(v) for s,v in data.items()})

def load():return json.loads((ROOT/'inputs/windows.json').read_text())
if __name__=='__main__':build()
