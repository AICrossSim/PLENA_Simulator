"""Report the development-only best exact geometry at every budget split."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from research.moe_dispatch.plena65536_dedicated.geometry import enumerate_cores
from research.moe_dispatch.plena65536_dedicated.search import label
from research.moe_dispatch.plena65536_dedicated.model import population_costs
from research.moe_dispatch.plena65536_moe.workloads import load_workloads


def main(output):
    output=Path(output);cores=enumerate_cores(include_blen1=True)
    selected=json.loads((output/'FROZEN_SELECTION.json').read_text())
    shared,_,weights=population_costs(cores,load_workloads('development')['development'])
    shared_scores=np.exp(np.log(shared)@weights)
    buffers=np.asarray([4*c.r*(c.pm+c.pn) for c in cores],dtype=np.int64)
    rows=[]
    for prefix in ('strict','extended'):
        with np.load(output/f'{prefix}_dedicated_free_pipelined_tree1_exact.npz') as archive:
            ids=archive['core_ids'];scores=archive['exact_GM_cycles']
            budget=np.asarray([cores[int(i)].macs for i in ids[:,0]],dtype=np.int64)
            # Each identical budget's feasible set is repriced in full; select
            # its actual minimum, never interpolate between budget ratios.
            order=np.lexsort((np.arange(len(ids)),buffers[ids[:,0]]+buffers[ids[:,1]],
                              shared_scores[ids[:,0]],scores,budget))
            sorted_budget=budget[order]
            first=np.r_[True,sorted_budget[1:]!=sorted_budget[:-1]]
            for ix in order[first]:
                ix=int(ix)
                a,b=(cores[int(i)] for i in ids[ix])
                assert a.macs+b.macs==65536 and b.pm in (1,2)
                rows.append({'domain':prefix,'Shared_MACs':a.macs,'Routed_MACs':b.macs,
                    'Shared_over_Routed_budget':a.macs/b.macs,
                    'Shared_GE_Routed':a.macs>=b.macs,
                    'development_GM_ms':float(scores[ix])/1e6,
                    'Shared_shape':label((a,)),'Routed_shape':label((b,)),
                    'selection_split':'development_only','hard_affinity':True})
    with (output/'split_frontier.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    checks={}
    for prefix in ('strict','extended'):
        points=[r for r in rows if r['domain']==prefix]
        assert abs(min(r['development_GM_ms'] for r in points)*1e6-selected['families'][prefix+'_dedicated_free']['development_GM_cycles'])<1e-7
        assert abs(min(r['development_GM_ms'] for r in points if r['Shared_GE_Routed'])*1e6-selected['families'][prefix+'_dedicated_largeShared']['development_GM_cycles'])<1e-7
        checks[prefix]={'legal_budget_splits':len(points),'all_candidates_consumed':True}
    (output/'SPLIT_FRONTIER_AUDIT.json').write_text(json.dumps({'status':'PASS','counts':checks,
        'heldout_used':False,'source':'exact exhaustive saved candidate scores'},indent=2)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
