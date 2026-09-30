#!/usr/bin/env python3
"""Predeclared bounded-dispatch experiment; captured routes, analytical timing.

New test requests exclude every request used in the earlier robust study.
Physical hardware is frozen; all policies receive identical descriptor order,
storage, port, and shared supply budgets. No test-time hardware selection.
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
import time

import numpy as np
import run_experiments as run
from robust_study import csv_out
from robust_workloads import build_window

OLD_ROOT = Path('/scratch/shared/mcl123/plena/outputs/moe_robust_fixed_20260930')
DEFAULT_ROOT = Path('/tmp/plena-joint-runtime-20260930')
CORE_POLICIES = ('fifo', 'dynamic', 'feedback', 'joint')
REFERENCES = ('window_lpt_ect', 'selected_tail_reference')
MAIN = CORE_POLICIES + REFERENCES
ABLATIONS = ('joint_no_late', 'joint_no_pair', 'joint_no_feedback', 'joint_no_prefetch')
POLICIES = MAIN + ABLATIONS
PLAN = {
    'schema': 'plena_joint_runtime_study_v1',
    'scope': 'Captured DeepSeek-V2-Lite decode routing; analytical complete FFN with router excluded. Not native Ramulator, pretrained numerical execution, or full model inference.',
    'hardware': 'Six prior frozen physical/G configurations, each retained unchanged except common 256 B joint-control reserve within the existing 2 MiB arena. No further hardware search.',
    'control': 'joint_state_bytes=256; control_bytes total=4352; reserved identically for every policy; feedback_state_bytes=96 remains separately accounted.',
    'task_granularity': 'Whole experts only, tail_partition=false for every primary policy and ablation; selected_tail_reference separately preserves the prior frozen tail setting.',
    'input': 'Stable Shared-first expert descriptors for every policy, otherwise existing routed expert order. One descriptor/cycle and bounded 8-entry window remain. Original Shared-last ordering is a development diagnostic.',
    'main_policies': MAIN,
    'ablations': ABLATIONS,
    'joint_parameters': 'Default bounded joint controller: late binding, pair choice, completion feedback enabled; age_limit=8, margin=64 cycles. No per-organization tuning.',
    'supply': '256 B/ns, 64 ns response, 256 shared 32 B credits, stock arbitration, finite SRAM landing. Current/Next prefetch enabled except declared ablation.',
    'development': 'Earlier four design windows only. Earlier heldout was seen and is not reused as an independent test.',
    'test_selection': 'Exclude all request identities in previous robust manifest; sort valid decode-step7 requests by SHA256(plena-runtime-v2:dataset:sample_id). Take disjoint consecutive groups B2/B4/B8/B16 in each of BFCL/GPQA/SWE, all layer13. No performance or route-hotness selection.',
    'references': 'window_lpt_ect uses the same bounded joint window with pairing, late binding and feedback disabled; it is a critical-task plus ECT baseline, not a HEFT DAG implementation. selected_tail_reference uses feedback with the previous frozen tail flag and the same new control reserve.',
    'test_count': '12 fresh windows x 6 frozen hardware x 10 policies x 2 repeats = 1440 executions.',
    'predictor_state': 'Cold reset per independent window; updates only from completed tasks. No claim of training over earlier requests.',
    'repetition': 'Exactly two executions; complete raw JSON byte equality required. Host wall time is not simulated latency.',
    'reporting': 'Paired per-window latency and geometric mean speedups, worst slowdown, decision opportunity, prediction error, Shared binding, storage/bytes/checks; no additive global wall-time interpretation of per-core overlapping counters.',
}


def reorder(w, shared_first=True):
    out = copy.deepcopy(w)
    out.pop('engine_layout', None)
    if shared_first:
        out['experts'].sort(key=lambda e: not e.get('is_shared', False))
    out['descriptor_order'] = 'shared_first' if shared_first else 'source_shared_last'
    return out


def prepare(root, old_root=OLD_ROOT):
    old_manifest = json.loads((old_root/'workload_manifest.json').read_text())
    used = {ident for w in old_manifest['windows'] for ident in w['sample_ids']}
    frozen = json.loads((old_root/'frozen_designs.json').read_text())
    designs = copy.deepcopy(frozen['designs'])
    for d in designs:
        d['prior_selected_tail_partition'] = d['tail_partition']
        d['tail_partition'] = False
        d['id'] = d['physical_id'] + '_joint_budget'
        d['resources']['joint_state_bytes'] = 256
        d['resources']['control_bytes'] = [4352//len(d['lanes'])]*len(d['lanes'])
    assert len(designs) == 6
    inputs = root/'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    dev = json.loads((old_root/'inputs/design_workloads.json').read_text())['workloads']
    test, manifest, selected = [], [], set()
    sources = []
    for source in old_manifest['sources']:
        dataset = source['dataset']
        path = Path(source['path'])
        assert run.file_sha(path) == source['sha256'], 'Captured source changed'
        sources.append(copy.deepcopy(source))
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(a['meta'].item())
            pool = []
            for i, sid in enumerate(a['sample_ids']):
                ident = dataset+':'+str(sid)
                if ident not in used and bool(a['valid'][i, 7]):
                    code = hashlib.sha256(('plena-runtime-v2:'+ident).encode()).hexdigest()
                    pool.append((code, i, ident))
            pool.sort()
            assert len(pool) >= 30
            cursor = 0
            for batch in (2, 4, 8, 16):
                entries = pool[cursor:cursor+batch]
                cursor += batch
                identities = [x[2] for x in entries]
                assert not set(identities) & (used | selected)
                selected.update(identities)
                name = f'joint_test_{dataset}_b{batch}_l13_s7'
                w = build_window(a, meta, [x[1] for x in entries], 7, 13, name)
                test.append(w)
                manifest.append(dict(id=name, dataset=dataset, batch=batch, layer=13, decode_step=7,
                                     sample_ids=identities, stream_indices=[x[1] for x in entries],
                                     split='fresh_test', offline_rebatch=True))
    bundle = dict(plan=PLAN, previous_manifest_sha256=run.file_sha(old_root/'workload_manifest.json'),
                  previous_frozen_sha256=run.file_sha(old_root/'frozen_designs.json'),
                  excluded_request_count=len(used), excluded_request_ids=sorted(used), sources=sources,
                  fresh_request_count=len(selected), windows=manifest)
    files = {
        root/'plan.json': PLAN,
        root/'frozen_hardware.json': designs,
        inputs/'development_workloads.json': dict(workloads=dev),
        inputs/'test_workloads.json': dict(workloads=test),
        root/'workload_manifest.json': bundle,
    }
    for path, data in files.items():
        if path.exists():
            assert json.loads(path.read_text()) == json.loads(json.dumps(data)), f'Refusing to change frozen file {path}'
        else:
            run.write_json(path, data)
    hashes = {str(p.relative_to(root)): run.file_sha(p) for p in files}
    receipt = root/'prepare_receipt.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['sha256'] == hashes
    else:
        run.write_json(receipt, dict(sha256=hashes, timing_seen=False,
                                    note='Manifest is frozen before new timing; hardware inherited from an earlier study.'))
    print(json.dumps(dict(designs=len(designs), development=len(dev), fresh_windows=len(test),
                          fresh_requests=len(selected), excluded_requests=len(used)), sort_keys=True))


def check_plan(root):
    receipt = json.loads((root/'prepare_receipt.json').read_text())
    for name, sha in receipt['sha256'].items():
        assert run.file_sha(root/name) == sha, f'Frozen input changed: {name}'
    assert json.loads((root/'plan.json').read_text()) == json.loads(json.dumps(PLAN))


def configuration(d, policy):
    dispatch = policy if policy in CORE_POLICIES else 'feedback' if policy == 'selected_tail_reference' else 'joint'
    cfg = dict(run.BASE, lanes=d['lanes'], group=d['group'], dispatch=dispatch,
               split='none', window=8, runtime_fsm=True, next_prefetch=policy != 'joint_no_prefetch',
               arbiter='stock', tail_partition=False)
    if policy == 'selected_tail_reference':
        cfg['tail_partition'] = d['prior_selected_tail_partition']
    if dispatch == 'joint':
        cfg.update(joint_late_bind=policy != 'joint_no_late', joint_pairing=policy != 'joint_no_pair',
                   joint_feedback=policy != 'joint_no_feedback',joint_age_limit=8,joint_margin_cycles=64)
    if policy == 'window_lpt_ect':
        cfg.update(joint_late_bind=False,joint_pairing=False,joint_feedback=False)
    return cfg


def points(root, split, policy_set, order):
    designs = json.loads((root/'frozen_hardware.json').read_text())
    windows = json.loads((root/'inputs'/f'{split}_workloads.json').read_text())['workloads']
    policies = CORE_POLICIES+('window_lpt_ect',) if policy_set=='pilot' else MAIN if policy_set == 'main' else ABLATIONS if policy_set == 'ablations' else POLICIES
    result = []
    for d in designs:
        for w in windows:
            w = reorder(w, order == 'shared_first')
            for policy in policies:
                result.append(dict(key=f'{split}__{order}__{w["id"]}__{d["id"]}__{policy}', suite=split,
                    organization='+'.join(map(str,d['lanes'])), mode=policy,
                    condition='equal_budget_joint_runtime_'+order, workload=w,
                    config=configuration(d,policy), resources=d['resources'], design=d))
    return result


def snapshot(root, binary):
    paths = run.source_paths()
    sources = {name: run.file_sha(p) for name,p in paths.items()}
    sha = run.file_sha(binary)
    prov = dict(binary_sha256=sha, sources=sources, source_bundle_sha256=run.digest(sources))
    target = root/'provenance'/(sha[:12]+'_'+prov['source_bundle_sha256'][:12])
    target.mkdir(parents=True, exist_ok=True)
    executable = target/'moe-dispatch-analytical-v1'
    if not executable.exists():
        shutil.copy2(binary, executable)
        for name, src in paths.items():
            dst = target/'sources'/name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        run.write_json(target/'manifest.json', prov)
    assert run.file_sha(executable) == sha
    return executable, prov


def extended_row(item):
    row = run.summary_row(item)
    d, r = item['point']['design'], item['report']
    row.update(design_id=d['id'], architecture=d['architecture'], budget_group=d['budget_group'],
               descriptor_order=item['point']['workload']['descriptor_order'])
    audits = r.get('dispatch_audit', [])
    shared = [a for a in audits if a.get('expert') == -1]
    row['shared_bind_cycle'] = min([a['cycle'] for a in shared], default='')
    row['shared_owner'] = shared[0]['core'] if len(shared) == 1 else ''
    row['shared_finish_cycle'] = max([a.get('actual_finish_cycle', 0) for a in shared], default='')
    row['two_core_choice_fraction'] = sum(len(a.get('eligible_cores', [])) > 1 for a in audits)/len(audits) if audits else 0
    row['credit_occupancy_lower_bound_cycles'] = math.ceil(r['weight_bytes']*item['point']['config']['hbm_latency_ns']/(32*item['point']['config']['credits']))
    row['latency_over_credit_lower_bound'] = r['cycles']/row['credit_occupancy_lower_bound_cycles']
    row['core_finish_gap_cycles'] = max(c['stats']['done_cycle'] for c in r['cores'])-min(c['stats']['done_cycle'] for c in r['cores'])
    # Full candidate audits remain in the raw report. Flatten finite controller
    # counters into the table instead of embedding a large JSON history in CSV.
    diagnostic = r.get('joint_diagnostics', {})
    for k,v in diagnostic.items():
        if isinstance(v,(int,float,bool)):
            row['joint_'+k] = v
    row['joint_audit_count'] = len(diagnostic.get('audit', []))
    observed = [c for a in diagnostic.get('audit', []) for c in a.get('candidates', [])]
    row['joint_candidate_observations'] = len(observed)
    row['joint_candidate_potential_both_fraction'] = sum(c.get('potential_mask',0)==3 for c in observed)/len(observed) if observed else 0
    row['joint_candidate_due_both_fraction'] = sum(c.get('eligible_mask',0)==3 for c in observed)/len(observed) if observed else 0
    return row


def execute(root, binary, split, policy_set, order, workers, filter_text=''):
    check_plan(root)
    ps = points(root, split, policy_set, order)
    if filter_text:
        ps = [p for p in ps if filter_text in p['key']]
    assert ps
    binary, prov = snapshot(root,binary)
    stage = f'{split}_{order}_{policy_set}' + ('_filter_'+hashlib.sha256(filter_text.encode()).hexdigest()[:8] if filter_text else '')
    run.write_json(root/f'{stage}_start.json', dict(binary_sha256=prov['binary_sha256'],
        source_bundle_sha256=prov['source_bundle_sha256'], points=len(ps), repeats=2,
        point_keys=[p['key'] for p in ps], plan_sha256=run.file_sha(root/'plan.json')))
    rows, fronts = [], []
    started = time.monotonic()
    def one(p):
        return run.run_point(p, root/'runs', binary, prov, 2, 1800, False)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(one,p):p for p in ps}
        for future in as_completed(jobs):
            item = future.result()
            rows.append(extended_row(item))
            fronts.extend(run.front_rows(item))
            if len(rows)%6 == 0 or len(rows) == len(ps):
                print(f'{stage} {len(rows)}/{len(ps)} host_s={time.monotonic()-started:.1f}', flush=True)
                csv_out(root/f'{stage}_results.csv', sorted(rows,key=lambda r:r['point']))
                run.write_json(root/f'{stage}_rows.json',sorted(rows,key=lambda r:r['point']))
    csv_out(root/f'{stage}_front_states.csv',fronts)
    run.write_json(root/f'{stage}_receipt.json',dict(points=len(rows),runs=2*len(rows),
        complete=True,repeats_equal=True,binary_sha256=prov['binary_sha256'],
        source_bundle_sha256=prov['source_bundle_sha256']))


def report(root):
    check_plan(root)
    by_point = {}
    for path in sorted(root.glob('*_rows.json')):
        for row in json.loads(path.read_text()):
            old = by_point.get(row['point'])
            assert old is None or old == row, 'Different results for the same declared point'
            by_point[row['point']] = row
    rows = list(by_point.values())
    csv_out(root/'all_results.csv', sorted(rows,key=lambda r:r['point']))
    test = [r for r in rows if r['suite']=='test' and r['descriptor_order']=='shared_first']
    groups = defaultdict(list)
    for r in test:
        groups[(r['budget_group'],r['organization'],r['mode'])].append(r)
    lookup = {(r['budget_group'],r['organization'],r['mode'],r['workload']):r for r in test}
    summary = []
    for (budget,org,policy),rs in sorted(groups.items()):
        s = dict(budget_group=budget,organization=org,policy=policy,windows=len(rs),
                 latency_geomean_ms=math.exp(math.fsum(math.log(r['latency_ms']) for r in rs)/len(rs)),
                 two_core_choice_fraction_mean=math.fsum(r['two_core_choice_fraction'] for r in rs)/len(rs))
        for baseline in MAIN:
            ratios = [lookup[(budget,org,baseline,r['workload'])]['latency_ms']/r['latency_ms']
                      for r in rs if (budget,org,baseline,r['workload']) in lookup]
            if len(ratios)==len(rs):
                s[f'speedup_over_{baseline}'] = math.exp(math.fsum(map(math.log,ratios))/len(ratios))
                s[f'worst_slowdown_percent_vs_{baseline}'] = max(0,(1/min(ratios)-1)*100)
        for b in (2,4,8,16):
            subset=[r['latency_ms'] for r in rs if r['batch']==b]
            s[f'B{b}_mean_ms']=math.fsum(subset)/len(subset) if subset else ''
        summary.append(s)
    csv_out(root/'summary.csv',summary)
    run.write_json(root/'summary.json',summary)
    text=['# Bounded joint runtime study','',
          'Captured-route Rust analytical FFN timings. Router, native Ramulator, pretrained payloads, and whole-model inference are outside this timing scope. Physical designs are fixed; every policy has the same added control reservation and Shared-first descriptor stream.','',
          '| Budget | Shape | Policy | Windows | B2 ms | B4 ms | B8 ms | B16 ms | vs dynamic |',
          '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        f=lambda x:f'{x:.6f}' if isinstance(x,(int,float)) else str(x)
        text.append('| '+' | '.join(str(s[k]) for k in ('budget_group','organization','policy','windows'))+' | '+
                    ' | '.join(f(s[k]) for k in ('B2_mean_ms','B4_mean_ms','B8_mean_ms','B16_mean_ms'))+' | '+f(s.get('speedup_over_dynamic',''))+' |')
    text += ['', 'Each batch entry is an arithmetic mean of three dataset windows when coverage is complete. Speedups are paired-window geometric means; values above 1 are faster. Missing or partial coverage must not be presented as the final study.', '',
             'The four ablations separately disable late binding, pair selection, feedback, or Next prefetch. Per-core front-state counters overlap arithmetic and the other core; they are not additive wall-time components.', '',
             f'Archived completed points: {len(rows)}; fresh-test points: {len(test)}; expected fresh-test points: 720.',
             'No automatic architectural-success claim is generated. Interpret measured latency, controller cost, binding opportunity, supply bounds and bytes together.']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(completed_points=len(rows),test_points=len(test),expected_test_points=720)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','run','report'))
    p.add_argument('--output',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--old-root',type=Path,default=OLD_ROOT)
    p.add_argument('--binary',type=Path,default=run.DEFAULT_BINARY)
    p.add_argument('--split',choices=('development','test'),default='development')
    p.add_argument('--policies',choices=('pilot','main','ablations','all'),default='all')
    p.add_argument('--order',choices=('shared_first','source_shared_last'),default='shared_first')
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--filter',default='')
    a=p.parse_args()
    assert a.workers>0
    if a.stage=='prepare': prepare(a.output,a.old_root)
    elif a.stage=='run':
        assert not (a.split=='test' and a.order!='shared_first'), 'Order diagnostic is development-only'
        execute(a.output,a.binary,a.split,a.policies,a.order,a.workers,a.filter)
    else: report(a.output)


if __name__=='__main__': main()
