"""Describe actual exact-ownership queues, without inventing a runtime policy."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from research.moe_dispatch.plena65536_moe.workloads import load_workloads
from research.moe_dispatch.plena65536_moe.search import core_from_dict
from research.moe_dispatch.plena65536_moe.model import schedule


def main(output):
    output = Path(output)
    frozen = json.loads((output/'FROZEN_SELECTION.json').read_text())
    windows = load_workloads('heldout')['heldout']
    rows = []
    for name, selection in frozen['families'].items():
        cores = tuple(core_from_dict(c) for c in selection['cores'])
        for window in windows:
            plan = schedule(cores,window)
            for ci,core in enumerate(cores):
                jobs = [j for j in plan['jobs'] if j['core_id']==ci]
                shared = [j for j in jobs if j['is_shared']]
                routed = [j for j in jobs if not j['is_shared']]
                rows.append({'design':name,'window_id':window['id'],'batch':window['batch'],
                    'core_id':ci,'PM':core.pm,'PN':core.pn,'G':core.groups,'BLEN':core.b,'MLEN':core.r,
                    'multipliers':core.macs,'shared_owned':bool(shared),
                    'shared_start_cycles':shared[0]['start'] if shared else '',
                    'shared_cycles':shared[0]['cycles'] if shared else 0,
                    'cold_M1_M2_experts':sum(j['M']<=2 for j in routed),
                    'hot_M_gt2_experts':sum(j['M']>2 for j in routed),
                    'routed_tokens':sum(j['M'] for j in routed),
                    'mean_routed_M':sum(j['M'] for j in routed)/len(routed) if routed else 0,
                    'max_routed_M':max((j['M'] for j in routed),default=0),
                    'core_load_cycles':plan['core_cycles'][ci],
                    'whole_window_cycles':plan['cycles'],
                    'core_idle_fraction':1-plan['core_cycles'][ci]/plan['cycles']})
    with (output/'dispatch_queues.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for name, selection in frozen['families'].items():
        summary[name]=[]
        for ci in range(len(selection['cores'])):
            own=[r for r in rows if r['design']==name and r['core_id']==ci]
            summary[name].append({'core_id':ci,'shared_owner_windows':sum(r['shared_owned'] for r in own),
                'total_windows':len(own),'cold_expert_tasks':sum(r['cold_M1_M2_experts'] for r in own),
                'hot_expert_tasks':sum(r['hot_M_gt2_experts'] for r in own),
                'routed_token_assignments':sum(r['routed_tokens'] for r in own)})
    (output/'DISPATCH_QUEUE_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
