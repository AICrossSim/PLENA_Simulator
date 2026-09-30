#!/usr/bin/env python3
"""Fixed-hardware study. No test-time selection; all reports are analytical."""
from __future__ import annotations
import argparse, copy, csv, itertools, json, math, shutil, subprocess, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import run_experiments as run

sys.path.insert(0, str(run.COMPILER_PATH.parent))
from robust_space import designs

POLICIES = ('fifo', 'dynamic', 'feedback')
TOYS = ([4,2], [3,3], [2,2], [8,2], [1,1], [8,8])
PLAN = {
    'version': 2, 'timing_scope': 'independent Rust analytical FFN, router excluded; not native Ramulator',
    'search': 'all generated physical points, G=2 or 4; all three policies; no timing pruning',
    'selection': 'For each budget and architecture, rank hardware by design geometric-mean feedback latency. Validate top 3 (or all if fewer), select minimum validation geometric-mean feedback latency; tie by design latency then id. No worst-case cutoff. Report worst and Pareto separately.',
    'supply': 'stock arbiter, 256 credits, 64ns response, 256B/ns serialization; early Next prefetch; surplus rules off',
    'feedback': 'per-core four Me bins, Q8 service correction; initialize 1.0 each independent window; update only completed Current tasks, 2 control cycles; 96B reserved for every policy',
    'test': '8 disjoint request windows; one frozen hardware per budget/architecture, three policies; never reselect using test timing',
    'assignment_oracle': 'enumerate all legal whole-expert owners, fixed input order, representative G4 resource point; no global optimality claim',
    'numeric_scope': 'small timed trace replay only; large real-route timings have no captured activation/weight payload',
    'conditional_E': 'Whole-assignment oracle leaves 433706/152830 idle tail cycles in M6 homogeneous/heterogeneous representative four-expert subset. Before design/test timing, add common tail_partition on/off. Last unbound expert only; wait for both cores, two disjoint proportional output ranges, full Z barrier and charged copy. No migration. Same shape/supply budgets. Re-evaluate all designs; single tail mode is a no-op.',
}

def configurations():
    out=[]
    for physical in designs():
        for tail in ((False,) if len(physical['lanes'])==1 else (False,True)):
            d=copy.deepcopy(physical);d['physical_id']=d['id'];d['tail_partition']=tail
            d['id'] += '_tail' if tail else '_whole'
            out.append(d)
    return out

def csv_out(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=keys);w.writeheader()
        w.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in rows)

def load_windows(root, split):
    return json.loads((root/'inputs'/f'{split}_workloads.json').read_text())['workloads']

def point(d,w,policy,stage,owners=None):
    cfg=dict(run.BASE,lanes=d['lanes'],group=d['group'],dispatch=policy,split='none',
             window=8,runtime_fsm=True,next_prefetch=True,arbiter='stock')
    cfg['tail_partition']=d.get('tail_partition',False)
    suffix=''
    if owners is not None:
        cfg['fixed_assignment']=list(owners);suffix='__a'+''.join(map(str,owners))
    return dict(key=f"{stage}__{w['id']}__{d['id']}__{policy}{suffix}",suite=stage,
                organization='+'.join(map(str,d['lanes'])),mode=policy,condition='formal_equal_budget',
                workload=w,config=cfg,resources=d['resources'],design=d)

def freeze(root,binary,stage):
    sources=run.source_inventory(); sha=run.file_sha(binary)
    snap=root/'provenance'/f'{stage}__{sha[:12]}_{run.digest(sources)[:12]}'
    snap.mkdir(parents=True,exist_ok=True)
    p=dict(binary_sha256=sha,sources=sources,source_bundle_sha256=run.digest(sources),plan=PLAN)
    for name,src in run.source_paths().items():
        dst=snap/'sources'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    target=snap/'moe-dispatch-analytical-v1';shutil.copy2(binary,target)
    run.write_json(snap/'manifest.json',p)
    return target,p

def run_points(points,root,binary,stage,workers):
    binary,prov=freeze(root,binary,stage)
    rows=[];rejects=[];front=[];start=time.monotonic()
    def one(p):
        # Compile first: illegal storage is a capacity result, never a latency.
        try:
            layout=run.attach_layout(p)['engine_layout']
            for c in layout['cores']:
                if c['reserved']>c['capacity']:raise ValueError(f"core{c['core']} persistent result reserve exceeds capacity")
            for e,candidates in enumerate(layout['whole']):
                owners=p['config'].get('fixed_assignment')
                legal=candidates[owners[e]]['feasible'] if owners is not None else any(c['feasible'] for c in candidates)
                if not legal:raise ValueError(f"expert index {e} has no feasible requested whole owner")
        except ValueError as e: return None, str(e)
        return run.run_point(p,root/stage,binary,prov,2,1800,False),None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(one,p):p for p in points}
        for f in as_completed(jobs):
            p=jobs[f]; item,reason=f.result()
            d=p['design']
            meta=dict(design_id=d['id'],budget_group=d['budget_group'],architecture=d['architecture'],stage=stage)
            if reason:
                rejects.append(dict(point=p['key'],workload=p['workload']['id'],policy=p['mode'],**meta,reason=reason))
            else:
                row=run.summary_row(item);row.update(meta)
                row['owners']=[a['core'] for a in item['report'].get('dispatch_audit',[])]
                row['feedback_updates']=item['report'].get('feedback_updates',0)
                row['tail_partition']=p['config'].get('tail_partition',False)
                row['tail_partition_count']=item['report'].get('tail_partition_count',0)
                rows.append(row);front.extend(run.front_rows(item))
            n=len(rows)+len(rejects)
            if n%10==0 or n==len(points):
                print(f'{stage} {n}/{len(points)} accepted={len(rows)} rejected={len(rejects)} host_s={time.monotonic()-start:.1f}',flush=True)
                csv_out(root/f'{stage}_results.csv',sorted(rows,key=lambda r:r['point']))
                csv_out(root/f'{stage}_rejections.csv',rejects)
    csv_out(root/f'{stage}_front_states.csv',front)
    run.write_json(root/f'{stage}_rows.json',rows)
    return rows,rejects

def representatives():
    out=[]
    for lanes in run.frontend.ROBUST_ORGANIZATIONS:
        out.append(next(d for d in designs() if d['lanes']==list(lanes) and d['group']==4
                        and d['arena_policy']=='proportional' and d['bank_policy']=='proportional'))
    return out

def toy(ms):
    w=copy.deepcopy(run.toy_workloads()[0]);w['id']='toy_me'+'_'.join(map(str,ms));w['batch']=sum(ms)
    w['input']['shape'][0]=sum(ms);w['tokens']=[];at=0
    for e,m in zip(w['experts'],ms):
        ids=list(range(at,at+m));e.update(Me=m,token_indices=ids,route_slots=[0]*m,route_scores=[1.0]*m)
        e['dag']=run.frontend.expert_dag(e)
        w['tokens'] += [dict(token_index=t,sample_id=f'synthetic-{t}',routes=[dict(expert_id=e['id'],slot=0,score=1.0)]) for t in ids]
        at+=m
    return w

def diagnostic(root,binary,workers):
    binary,prov=freeze(root,binary,'compute')
    out=[];points=[]
    # One projection; enumerate all owners, preserving within-core expert order.
    for ms in TOYS:
        w=toy(ms)
        for d in representatives():
            for owners in itertools.product(range(len(d['lanes'])),repeat=len(ms)):
                key=f"{w['id']}__{d['id']}__{''.join(map(str,owners))}"
                args=dict(lanes=d['lanes'],Me=ms,owners=owners,N=128,K=512,group=4)
                path=root/'compute'/key;path.mkdir(parents=True,exist_ok=True)
                run.write_json(path/'input.json',args)
                for i in (1,2):subprocess.run([str(binary),'--compute',str(path/'input.json'),str(path/f'report{i}.json')],check=True)
                assert (path/'report1.json').read_bytes()==(path/'report2.json').read_bytes()
                r=json.loads((path/'report1.json').read_text())
                out.append(dict(workload=w['id'],budget_group=d['budget_group'],architecture=d['architecture'],
                                lanes=d['lanes'],owners=owners,**r))
                points.append(point(d,w,'fixed','assignment',owners))
    csv_out(root/'compute_mechanism.csv',out)
    # Representative route subset: retain original IDs, scores, shapes. This is not a full layer.
    full=next(w for w in load_windows(root,'design') if w['batch']==4)
    partial=copy.deepcopy(full);partial['id']='partial4_'+full['id']
    routed=[e for e in partial['experts'] if not e['is_shared']][:3]
    partial['experts']=routed+[e for e in partial['experts'] if e['is_shared']]
    ids={e['id'] for e in partial['experts']}
    for t in partial['tokens']: t['routes']=[r for r in t['routes'] if r['expert_id'] in ids]
    partial['scope']='oracle-only partial experts from captured routes; not complete MoE layer'
    for d in representatives():
        for owners in itertools.product(range(len(d['lanes'])),repeat=len(partial['experts'])):
            points.append(point(d,partial,'fixed','assignment',owners))
    rows,rejects=run_points(points,root,binary,'assignment',workers)
    csv_out(root/'assignment_search.csv',rows)

def rank(rows,policy='feedback'):
    groups=defaultdict(list)
    for r in rows:
        if r['mode']==policy:groups[r['design_id']].append(r)
    ranks=[]
    for did,rs in groups.items():
        # Capacity failures remove a design from selection, not from the coverage report.
        if len(rs)!=4:continue
        ranks.append(dict(design_id=did,budget_group=rs[0]['budget_group'],architecture=rs[0]['architecture'],
                          geomean_cycles=math.exp(math.fsum(math.log(r['latency_cycles_at_1ghz']) for r in rs)/len(rs)),
                          worst_cycles=max(r['latency_cycles_at_1ghz'] for r in rs)))
    return sorted(ranks,key=lambda r:(r['geomean_cycles'],r['design_id']))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['plan','diagnostic','elasticity','design','validation','heldout'])
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--binary',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=16);a=ap.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=True)
    space=configurations()
    if a.stage=='plan':
        run.write_json(root/'selection_plan.json',PLAN)
        run.write_json(root/'all_designs.json',space)
        csv_out(root/'design_space.csv',[dict(**d,legal=True,reason='') for d in space])
        print('physical configurations',len(designs()),'with granularity',len(space));return
    assert json.loads((root/'selection_plan.json').read_text())==PLAN,'selection plan changed'
    if a.stage=='diagnostic':diagnostic(root,a.binary,a.workers);return
    if a.stage=='elasticity':
        ws=[toy(ms) for ms in TOYS]
        # Complete B4 layer here, unlike the assignment-search partial subset.
        ws += [w for w in load_windows(root,'design') if w['batch']==4]
        ps=[]
        for d in representatives():
            for tail in ((False,) if len(d['lanes'])==1 else (False,True)):
                v=dict(d,id=d['id']+('_tail' if tail else '_whole'),tail_partition=tail)
                ps += [point(v,w,p,'elasticity') for w in ws for p in POLICIES]
        run_points(ps,root,a.binary,'elasticity',a.workers);return
    if a.stage=='design':selected=space
    elif a.stage=='validation':
        ranking=rank(json.loads((root/'design_rows.json').read_text()));chosen=[];counts=defaultdict(int)
        for r in ranking:
            key=(r['budget_group'],r['architecture'])
            if counts[key]<3:chosen.append(r['design_id']);counts[key]+=1
        run.write_json(root/'validation_shortlist.json',dict(selection=PLAN['selection'],candidates=chosen,ranking=ranking))
        selected=[d for d in space if d['id'] in chosen]
    else:
        frozen=json.loads((root/'frozen_designs.json').read_text());selected=frozen['designs']
    split=a.stage
    if split=='heldout':
        ps=[]
        for d in selected:
            for tail in ((False,) if len(d['lanes'])==1 else (False,True)):
                v=dict(d,id=d['id']+f'_ablate_tail{int(tail)}',tail_partition=tail,frozen_id=d['id'])
                ps += [point(v,w,p,split) for w in load_windows(root,split) for p in POLICIES]
        points=ps
    else:points=[point(d,w,p,split) for d in selected for w in load_windows(root,split) for p in POLICIES]
    rows,rejects=run_points(points,root,a.binary,split,a.workers)
    if a.stage=='validation':
        ranking=rank(rows);chosen={}
        design_scores={r['design_id']:r['geomean_cycles'] for r in rank(json.loads((root/'design_rows.json').read_text()))}
        ranking.sort(key=lambda r:(r['geomean_cycles'],design_scores[r['design_id']],r['design_id']))
        for r in ranking:chosen.setdefault((r['budget_group'],r['architecture']),r['design_id'])
        assert len(chosen)==6,'some architecture/budget lacks complete validation coverage'
        run.write_json(root/'frozen_designs.json',dict(selection=PLAN,designs=[d for d in space if d['id'] in chosen.values()],
            validation_ranking=ranking,heldout_timing_seen=False,binary_sha256=run.file_sha(a.binary)))
    if a.stage=='heldout':csv_out(root/'dispatch_ablation.csv',rows)

if __name__=='__main__':main()
