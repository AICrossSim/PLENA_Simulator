#!/usr/bin/env python3
"""Repeat complete-grid winners; evaluate holdout and bounded timing sensitivity.

Primary winners are frozen BEFORE any holdout measurement. Sensitivity compares
those fixed winners, and does not claim each is optimal under changed timing.
"""
import argparse
import copy
import json
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from compare_moe_normal import digest,read_json,require
from run_moe_dma_campaign import save
from run_moe_joint_dse import FIXTURES,execute_point,load_report,resource_gate


def numerical_result_signature(envelope):
    # Paths, configuration labels and executable metadata may differ. Every
    # numerical value, timestamp, busy counter and native statistic must match.
    result=copy.deepcopy(envelope['result']);result.pop('architecture',None)
    return dict(result=result,memory_model=envelope['memory_model'])


def main(root,workers):
    status=read_json(root/'status.json');require(status['status']=='passed','grid incomplete')
    plan=read_json(root/'campaign.json');ranking=read_json(root/'ranking.json')
    require(status['passed']==status['expected']==plan['expected_runs'],'coverage differs')
    holdout=read_json(root/'holdout_manifest.json')
    winners={c:r['point'] for c,r in ranking['winners'].items()}
    tasks=[]
    for category,point in winners.items():
        for repeat in [1,2]:
            for fixture in FIXTURES:
                tasks.append(dict(kind='repeat',category=category,point=point,fixture=fixture,
                    label=point['id']+'_repeat'+str(repeat)))
        for fixture in sorted({k.split('/')[1] for k in holdout['fixtures']}):
            record=holdout['fixtures'][point['axes']['layout']+'/'+fixture]
            for repeat in [1,2]:
                tasks.append(dict(kind='holdout',category=category,point=point,fixture=fixture,
                    fixture_override=record,label=point['id']+'_holdout_repeat'+str(repeat)))
        original=read_json(root/'architectures'/(point['id']+'.json'))
        for mode in ['activation512','activation2048','lookup_ii1','legacy_serialized','optimistic_direct']:
            architecture=copy.deepcopy(original)
            if mode.startswith('activation'):
                budget=int(mode[len('activation'):])
                for core in architecture['cores']:
                    core['activation_elements_per_cycle']=budget*core['blen']*core['mlen']//4096
            elif mode=='lookup_ii1':architecture['dma']['lookup_ii_cycles']=1
            elif mode=='legacy_serialized':architecture['matrix_timing']='legacy_serialized'
            else:architecture.pop('dma')
            label=point['id']+'_'+mode;architecture['name']=label
            path=root/'finalists/architectures'/(label+'.json');save(path,architecture)
            for fixture in FIXTURES:
                tasks.append(dict(kind='sensitivity',category=category,mode=mode,point=point,
                    fixture=fixture,label=label,architecture_override=str(path),architecture_sha256=digest(path)))
    execution=dict(winners=winners,tasks=tasks,grid_sha256=digest(root/'campaign.json'),
        ranking_sha256=digest(root/'ranking.json'),holdout_sha256=digest(root/'holdout_manifest.json'),
        driver_sha256=digest(__file__),selection_rule='only four category winners by search-set geometric mean; no holdout tuning')
    save(root/'finalists/execution_plan.json',execution)
    outcomes=[]
    def execute(t):
        if 'architecture_override' in t:
            require(digest(t['architecture_override'])==t['architecture_sha256'],'sensitivity architecture changed')
        value=execute_point(root,plan,t['point'],t['fixture'],label=t['label'],
            architecture_override=Path(t['architecture_override']) if 'architecture_override' in t else None,
            fixture_override=t.get('fixture_override'))
        if 'architecture_override' in t:
            require(digest(t['architecture_override'])==t['architecture_sha256'],'sensitivity architecture changed during execution')
        return dict(task=t,outcome=value)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(execute,t) for t in tasks]
        for future in as_completed(futures):
            outcomes.append(future.result())
            save(root/'finalists/status.json',dict(status='running',completed=len(outcomes),expected=len(tasks),
                failed=[o for o in outcomes if o['outcome']['status']!='passed']))
            print(str(len(outcomes))+'/'+str(len(tasks)),flush=True)
    require(all(o['outcome']['status']=='passed' for o in outcomes),'finalist execution failed')
    repeat_gates=[]
    for category,point in winners.items():
        for fixture in FIXTURES:
            baseline=numerical_result_signature(load_report(root/'runs'/point['id']/fixture/'report.json.gz'))
            for repeat in [1,2]:
                other=numerical_result_signature(load_report(root/'runs'/(point['id']+'_repeat'+str(repeat))/fixture/'report.json.gz'))
                require(baseline==other,'search/repeat timing, counters or values differ: '+point['id']+'/'+fixture)
            repeat_gates.append(dict(category=category,fixture=fixture,executions=3,all_values_and_counters_exact=True))
        for fixture in sorted({k.split('/')[1] for k in holdout['fixtures']}):
            reports=[numerical_result_signature(load_report(root/'runs'/(point['id']+'_holdout_repeat'+str(r))/fixture/'report.json.gz')) for r in [1,2]]
            require(reports[0]==reports[1],'holdout repeats differ')
            repeat_gates.append(dict(category=category,fixture=fixture,executions=2,all_values_and_counters_exact=True))
    save(root/'finalists/status.json',dict(status='passed',completed=len(outcomes),expected=len(tasks),
        outcomes=outcomes,repeat_gates=repeat_gates))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=24)
    a=p.parse_args();require(1<=a.workers<=40,'workers');main(a.root.resolve(),a.workers)
