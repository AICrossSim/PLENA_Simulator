#!/usr/bin/env python3
"""EXACT full-archive Me histograms for each batch; not a timing simulation."""
import csv,json
from pathlib import Path
import numpy as np
from prepare_routes import ROOT,OUT

def main():
    rows=[]
    for item in json.loads((OUT/'inputs/archive_inventory.json').read_text()):
        with np.load(item['path'],allow_pickle=False) as a:
            idx=a['decode_idx'];valid=a['valid'];E=item['metadata']['routed_experts'];L=item['layers'];topk=item['top_k']
            for b in [2,4,8,16]:
                hist=np.zeros(b+1,dtype=np.int64);windows=0;pairs=0;dropped=0
                for step in range(idx.shape[1]):
                    x=idx[valid[:,step],step];n=len(x)//b*b;dropped+=(len(x)-n)*L*topk
                    if not n:continue
                    q=x[:n].reshape(n//b,b,L,topk).transpose(0,2,1,3).reshape(-1,b*topk)
                    encoded=q.astype(np.int64)+np.arange(len(q),dtype=np.int64)[:,None]*E
                    counts=np.bincount(encoded.reshape(-1),minlength=len(q)*E)
                    assert counts.max()<=b
                    hist+=np.bincount(counts,minlength=b+1)
                    windows+=len(q);pairs+=int(counts.sum())
                assert int(hist@np.arange(b+1))==pairs and int(hist.sum())==windows*E
                assert pairs+dropped==item['valid_routed_pairs']
                active=int(hist[1:].sum())
                row=dict(model=item['model'],dataset=item['dataset'],batch=b,layer_batch_windows=windows,
                    routed_pairs=pairs,dropped_tail_pairs=dropped,active_expert_jobs=active,
                    mean_active_experts=active/windows,frac_expert_jobs_M1=float(hist[1]/active),
                    frac_pairs_in_M1=float(hist[1]/pairs),mean_M_when_active=pairs/active,
                    Me_histogram=json.dumps(hist.tolist()),scope='full valid archive distribution, rebatching, not timing runs')
                rows.append(row)
                print(item['model'],item['dataset'],b,windows,flush=True)
    with (OUT/'inputs/population_by_batch.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'inputs/population_validation.json').write_text(json.dumps(dict(passed=True,points=len(rows),
        all_valid_pairs_accounted_in_complete_windows_or_explicit_tail=True,latency_claim=False),indent=2)+'\n')
if __name__=='__main__':main()
