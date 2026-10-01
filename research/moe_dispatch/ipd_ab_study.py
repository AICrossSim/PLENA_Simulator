#!/usr/bin/env python3
"""Fixed-budget BF16 dispatch study on the archived, already-seen route windows.

This is a controlled reanalysis, not a new held-out test. Timing is executed by
the Rust analytical simulator; Python only prepares points and paired summaries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil

import joint_study
import run_experiments as run
from robust_study import csv_out

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / 'results/joint_runtime_20260930'
POLICIES = ('dynamic', 'window_lpt_ect', 'joint', 'ipd_no_quota', 'ipd')


def cases():
    designs = [d for d in json.loads((ARCHIVE/'frozen_hardware.json').read_text())
               if d['budget_group'] == 'M6']
    windows = json.loads((ARCHIVE/'inputs/test_workloads.json').read_text())['workloads']
    assert len(designs) == 3 and len(windows) == 12
    result = []
    for w in windows:
        w = joint_study.reorder(w)
        for d in designs:
            for policy in POLICIES:
                cfg = joint_study.configuration(d, 'joint' if policy.startswith('ipd') else policy)
                if policy.startswith('ipd'):
                    cfg['dispatch'] = 'ipd'
                    cfg['ipd_credit_quotas'] = policy == 'ipd'
                assert cfg['credits'] == 256 and cfg['hbm_latency_ns'] == 64
                assert not cfg['tail_partition'] and cfg['split'] == 'none'
                key = f'{w["id"]}__{d["id"]}__{policy}'
                result.append(dict(key=key,suite='archived_reanalysis',organization='+'.join(map(str,d['lanes'])),
                    mode=policy,condition='fixed_budget_bf16_ipd_ab',workload=w,
                    config=cfg,resources=d['resources'],design=d))
    return result


def row(item):
    p, r = item['point'], item['report']
    cycles = r['cycles']
    multiplier_count = sum(p['config']['lanes']) * 4 * 512
    out = dict(point=p['key'],workload=p['workload']['id'],shape=p['organization'],policy=p['mode'],
        batch=p['workload']['batch'],latency_ms=r['time_ms_at_1ghz'],
        hbm_weight_bytes=r['weight_bytes'],useful_macs=r['useful_macs'],
        hbm_GBps=r['weight_bytes']/cycles,
        hbm_util_configured=r['weight_bytes']/(256*cycles),
        hbm_util_credit_roof=r['weight_bytes']/(128*cycles),
        useful_mac_fraction_nominal_peak=r['useful_macs']/(multiplier_count*cycles),
        spatial_mac_fraction=r['useful_macs']/r['issued_macs'],
        credit_peak=r['credit_peak'],ipd_quota_block_cycles=r['ipd_quota_block_cycles'],
        control_cycles_sum=sum(c['stats']['control_cycles'] for c in r['cores']),
        two_core_finish_gap_cycles=max(c['stats']['done_cycle'] for c in r['cores'])
            -min(c['stats']['done_cycle'] for c in r['cores']),
        repeat_equal=True,report_sha256=item['report_sha256'])
    for i,c in enumerate(r['cores']):
        stats=c['stats']
        out[f'core{i}_arithmetic_active_fraction_observer']=stats['arithmetic_active_cycles']/cycles
        out[f'core{i}_arithmetic_window_hbm_accept_fraction_observer']=stats['arithmetic_window_hbm_accept_cycles']/cycles
        out[f'core{i}_mac_issue_hbm_accept_cycles']=stats['mac_issue_hbm_accept_cycles']
        out[f'core{i}_useful_mac_fraction_nominal_peak']=stats['useful_macs']/(c['m']*2048*cycles)
        out[f'core{i}_weight_not_ready_fraction']=stats['front_states'].get('weight_not_ready',0)/cycles
    return out


def execute(root, binary, workers, filter_text):
    ps=[p for p in cases() if filter_text in p['key']]
    assert ps
    source_paths=run.source_paths()
    sources={name:run.file_sha(path) for name,path in source_paths.items()}
    for name in ('frozen_hardware.json','inputs/test_workloads.json','workload_manifest.json'):
        sources['archive/'+name]=run.file_sha(ARCHIVE/name)
    prov=dict(binary_sha256=run.file_sha(binary),sources=sources,source_bundle_sha256=run.digest(sources))
    target=root/'provenance'/(prov['binary_sha256'][:12]+'_'+prov['source_bundle_sha256'][:12])
    target.mkdir(parents=True,exist_ok=True)
    frozen=target/'moe-dispatch-analytical-v1'
    if not frozen.exists(): shutil.copy2(binary,frozen)
    assert run.file_sha(frozen)==prov['binary_sha256']
    for name,path in source_paths.items():
        dest=target/'sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copy2(path,dest)
        assert run.file_sha(dest)==sources[name]
    run.write_json(target/'manifest.json',prov)
    rows=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run.run_point,p,root/'runs',frozen,prov,2,1800,False):p for p in ps}
        for future in as_completed(futures):
            item=future.result()
            rows.append(row(item))
            if len(rows)%5==0 or len(rows)==len(ps):
                print(f'{len(rows)}/{len(ps)}',flush=True)
                csv_out(root/'points.csv',sorted(rows,key=lambda x:x['point']))
    run.write_json(root/'receipt.json',dict(points=len(ps),repetitions=2,
        fixed_budget=True,source_bundle_sha256=prov['source_bundle_sha256'],
        binary_sha256=prov['binary_sha256'],archived_inputs_already_seen=True))


def report(root):
    import csv
    with (root/'points.csv').open() as f: rows=list(csv.DictReader(f))
    lookup={(r['workload'],r['shape'],r['policy']):r for r in rows}
    grouped=defaultdict(list)
    for r in rows: grouped[(r['shape'],r['policy'])].append(r)
    summary=[]
    for (shape,policy),rs in sorted(grouped.items()):
        def speed(ref_shape,ref_policy):
            ratios=[float(lookup[(r['workload'],ref_shape,ref_policy)]['latency_ms'])/float(r['latency_ms'])
                    for r in rs if (r['workload'],ref_shape,ref_policy) in lookup]
            return math.exp(sum(map(math.log,ratios))/len(ratios)) if len(ratios)==len(rs) else None
        def worst(ref_shape,ref_policy):
            ratios=[float(lookup[(r['workload'],ref_shape,ref_policy)]['latency_ms'])/float(r['latency_ms'])
                    for r in rs if (r['workload'],ref_shape,ref_policy) in lookup]
            return max(0,1/min(ratios)-1) if len(ratios)==len(rs) else None
        item=dict(shape=shape,policy=policy,windows=len(rs),
            latency_geomean_ms=math.exp(sum(math.log(float(r['latency_ms'])) for r in rs)/len(rs)),
            mean_hbm_util_configured=sum(float(r['hbm_util_configured']) for r in rs)/len(rs),
            mean_hbm_util_credit_roof=sum(float(r['hbm_util_credit_roof']) for r in rs)/len(rs),
            speedup_over_same_shape_joint=speed(shape,'joint'),
            worst_slowdown_vs_same_shape_joint=worst(shape,'joint'),
            speedup_over_same_shape_dynamic=speed(shape,'dynamic'))
        if policy=='ipd':
            item['speedup_over_single_same_policy']=speed('6','ipd')
            item['worst_slowdown_vs_single_same_policy']=worst('6','ipd')
            item['speedup_over_homogeneous_same_policy']=speed('3+3','ipd')
            item['worst_slowdown_vs_homogeneous_same_policy']=worst('3+3','ipd')
            item['quota_incremental_speedup']=speed(shape,'ipd_no_quota')
        for b in (2,4,8,16):
            subset=[float(r['latency_ms']) for r in rs if int(r['batch'])==b]
            item[f'B{b}_mean_ms']=sum(subset)/len(subset) if subset else None
        summary.append(item)
    # Every strategy and organization must perform identical useful work and
    # read the same immutable BF16 weights for each frozen route window.
    by_workload=defaultdict(list)
    for r in rows: by_workload[r['workload']].append(r)
    for workload,rs in by_workload.items():
        assert len({int(r['hbm_weight_bytes']) for r in rs})==1, workload
        assert len({int(r['useful_macs']) for r in rs})==1, workload
    csv_out(root/'summary.csv',summary)
    run.write_json(root/'summary.json',summary)
    print(json.dumps(dict(points=len(rows),groups=len(summary),complete=len(rows)==180,
        archived_inputs_already_seen=True),sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('run','report','list'))
    p.add_argument('--output',type=Path,default=Path('/tmp/plena-ipd-ab-study'))
    p.add_argument('--binary',type=Path,default=run.DEFAULT_BINARY)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--filter',default='')
    a=p.parse_args()
    if a.stage=='list':
        for item in cases():
            if a.filter in item['key']: print(item['key'])
    elif a.stage=='run': execute(a.output,a.binary,a.workers,a.filter)
    else: report(a.output)

if __name__=='__main__': main()
