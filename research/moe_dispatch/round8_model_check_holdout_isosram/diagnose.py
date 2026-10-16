"""Read-only replay and algebraic attribution of the three timing contracts.

No geometry search, held-out resampling or equal-SRAM experiment is run here.
The decomposition is an accounting identity, not measured RTL stall counters.
"""
from pathlib import Path
from collections import Counter
import csv
import gzip
import hashlib
import inspect
import json
import math
import subprocess

from research.moe_dispatch.round7_64k_dualcore_dse.model import Core, gemm, task_cost
from research.moe_dispatch.round7_64k_dualcore_dse.trace import load_inputs
from research.moe_dispatch.scaling_g64.geometry import Core as LegacyCore
from research.moe_dispatch.scaling_g64.runtime import schedule as legacy_schedule
from research.moe_dispatch.plena65536_dedicated.model import job_cycles
from research.moe_dispatch.plena65536_shared.timing import phase, PROFILES
from research.moe_dispatch.round5_largebatch_compute.model import projection

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
R7=ROOT.parent/'round7_64k_dualcore_dse'
CFG=json.loads((ROOT/'config.json').read_text())
OUT=ROOT/'results'
MODELS=CFG['models']

def ceildiv(x,y): return (x+y-1)//y
def gm(xs): return math.exp(math.fsum(map(math.log,xs))/len(xs))
def mean(xs): return math.fsum(xs)/len(xs)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(name,data):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def write_csv(name,rows):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def decompose(model,shape,m,k,n):
    pm,pn,g=shape
    nm,nn,nk=ceildiv(m,pm),ceildiv(n,pn),ceildiv(k,g)
    pure=nm*nn*nk
    if model=='legacy_phase':
        c=LegacyCore.from_shape(pm,pn,g)
        total=phase(c,m,n,k,1,PROFILES[0])
        # G is GROUP COUNT here. The temporal feed width is b*G, for b clocks.
        packing=nm*nn*(ceildiv(k,c.r)*c.b-nk)
        reduction=g.bit_length()-1
        recurrence=0
        other=packing+c.b+11
        groups=1
        q_sizes=[]
    elif model=='round7_os':
        c=Core(pm,pn,g)
        total=gemm(c,m,k,n,'plena_os',0)['cycles']
        q_sizes=[min(8,nn-j) for j in range(0,nn,8)]
        groups=nm*len(q_sizes)
        L=3+2*(g.bit_length()-1)
        recurrence=nm*sum((nk-1)*max(0,L-q) for q in q_sizes)
        # Residual drain L-1 = 2*log2(G) + 2. This is an algebraic convention:
        # the source exposes only a combined latency, not physical counters.
        reduction=groups*2*(g.bit_length()-1)
        packing=0
        other=2*groups
    elif model=='ceil':
        total=gemm(Core(pm,pn,g),m,k,n,'issue_formula',0)['cycles']
        reduction=recurrence=other=packing=0;groups=0;q_sizes=[]
    else: raise ValueError(model)
    assert pure+reduction+recurrence+other==total
    return dict(model=model,shape='x'.join(map(str,shape)),M=m,K=k,N=n,
                pure_cycles=pure,reduction_drain_cycles=reduction,
                accumulation_dependency_cycles=recurrence,other_cycles=other,
                total_cycles=total,packing_extra_cycles=packing,
                pure_us=pure/1000,reduction_drain_us=reduction/1000,
                accumulation_dependency_us=recurrence/1000,other_us=other/1000,
                total_us=total/1000,output_groups=groups,q_sizes=json.dumps(q_sizes))

def source_record(obj):
    obj=inspect.unwrap(obj)
    p=Path(inspect.getsourcefile(obj)).resolve()
    return dict(path=str(p),function=obj.__qualname__,line=inspect.getsourcelines(obj)[1],sha256=sha(p))

def sources():
    from research.moe_dispatch.round7_64k_dualcore_dse.legacy import run as old_replay
    refs={k:source_record(v) for k,v in dict(legacy_replay=old_replay,
          legacy_runtime=legacy_schedule,legacy_job=job_cycles,legacy_phase=phase,
          r7_gemm=gemm,r7_task=task_cost,r5_projection=projection).items()}
    paths=[
      'research/moe_dispatch/geometry3d/compute.py',
      'research/moe_dispatch/round2/model.py',
      'research/moe_dispatch/round4_compute_first/model.py',
      'research/moe_dispatch/round5_largebatch_compute/model.py',
      'research/moe_dispatch/plena65536_shared/timing.py',
      'research/moe_dispatch/scaling_g64/runtime.py',
      'research/moe_dispatch/round7_64k_dualcore_dse/model.py']
    histories={p:subprocess.check_output(['git','log','--reverse','--format=%h %ad %s',
                      '--date=iso-strict','--',p],cwd=REPO,text=True).splitlines() for p in paths}
    rtl=Path('/scratch/shared/mcl123/plena/worktrees/pr_candidates/PLENA_RTL_fake_hbm_capacity')
    files=[REPO/'analytic_models/performance/customISA_lib.json',
           REPO/'transactional_emulator/src/matrix_machine.rs',
           REPO/'transactional_emulator/src/matrix_service.rs',
           rtl/'src/basic_components/systolic_gemm_fp/rtl/fp_sum_across_sa.sv',
           rtl/'src/basic_components/systolic_gemm_fp/rtl/fp_default_pe.sv',
           rtl/'src/basic_components/gemv/rtl/fp_adder_tree.sv']
    return dict(functions=refs,history=histories,evidence_files={str(p):sha(p) for p in files},
                paper='https://arxiv.org/html/2509.09505v3#S3.SS2',
                paper_section='III-B, Figures 5/6; pipelined tiles, OS PE partials, M_SUM across subarrays',
                interpretation='Paper supports mechanisms, not fixed 8 contexts or a 15-cycle per-K recurrence.')

def run():
    frozen={str(p.relative_to(R7)):sha(p) for p in R7.rglob('*') if p.is_file()}
    windows,meta=load_inputs(write=False)
    original=json.load(gzip.open(R7/'inputs/original_windows.json.gz','rt'))
    assert [w['id'] for w in windows]==[w['id'] for w in original]
    for w,nw in zip(original,windows):
        old={(e['id'],e['Me'],e['F']) for e in w['experts'] if not e['is_shared']}
        new={(e['id'],e['m'],1408) for e in nw['tasks'] if not e['shared']}
        assert old==new
    gemms=[decompose(mod,s,*mnk) for mod in MODELS for s in CFG['shapes'] for mnk in CFG['gemms_M_K_N']]
    assert gemms==[decompose(mod,s,*mnk) for mod in MODELS for s in CFG['shapes'] for mnk in CFG['gemms_M_K_N']]
    write_csv('gemm_breakdown.csv',gemms)
    c=Core(4,256,64);oldc=LegacyCore.from_shape(4,256,64)
    b2=[];case={};tasks=[]
    for w,nw in zip(original,windows):
        if w['batch']!=2: continue
        old=legacy_schedule((oldc,),w,profile=CFG['legacy_profile'],details=True,**CFG['legacy_runtime'])
        again=legacy_schedule((oldc,),w,profile=CFG['legacy_profile'],details=True,**CFG['legacy_runtime'])
        assert old==again
        oldsum=sum(job_cycles(oldc,e,PROFILES[0]) for e in w['experts'])
        assert old['cycles']==oldsum+old['controller_delay_cycles']
        assert old['controller_delay_cycles']==5
        row=dict(window=w['id'],active_routed=len(w['experts'])-1,
                 legacy_task_count=len(w['experts']),r7_task_count=len(nw['tasks']),
                 legacy_compute_us=oldsum/1000,legacy_control_us=old['controller_delay_cycles']/1000,
                 legacy_total_us=old['cycles']/1000)
        for mod in ['plena_os','issue_formula']:
            pieces=[decompose('round7_os' if mod=='plena_os' else 'ceil',[4,256,64],t['m'],k,n)
                    for t in nw['tasks'] for k,n in [(2048,1408),(2048,1408),(1408,2048)]]
            for part in ['pure','reduction_drain','accumulation_dependency','other','total']:
                row[mod+'_'+part+'_us']=sum(p[part+'_cycles'] for p in pieces)/1000
            row[mod+'_merged_shared_us']=sum(task_cost(c,e['Me'],mod,0,e['F'])['cycles'] for e in w['experts'])/1000
        b2.append(row)
        if w['id']==CFG['case_window']:
            case=dict(row=row,original=w,normalized=nw,runtime=old)
            for e in w['experts']:
                if e['is_shared']:continue
                tasks.append(dict(task='R'+str(e['id']),M=e['Me'],old_F=e['F'],new_F=1408,
                    legacy_us=job_cycles(oldc,e,PROFILES[0])/1000,
                    r7_os_us=task_cost(c,e['Me'],'plena_os',0)['cycles']/1000,
                    ceil_us=task_cost(c,e['Me'],'issue_formula',0)['cycles']/1000))
            shared=w['experts'][0];assert shared['is_shared']
            tasks.insert(0,dict(task='Shared A+B (legacy fused)',M=2,old_F=2816,new_F='1408+1408',
                legacy_us=job_cycles(oldc,shared,PROFILES[0])/1000,
                r7_os_us=2*task_cost(c,2,'plena_os',0)['cycles']/1000,
                ceil_us=2*task_cost(c,2,'issue_formula',0)['cycles']/1000))
    write_csv('b2_windows.csv',b2);write_csv('b2_case_tasks.csv',tasks);write_json('results/case_window.json',case)
    write_json('results/model_sources.json',sources())
    agg={key:dict(AM_us=mean([r[key] for r in b2]),GM_us=gm([r[key] for r in b2]))
         for key in b2[0] if key.endswith('_us') and all(r[key]>0 for r in b2)}
    assert abs(agg['legacy_total_us']['GM_us']-7.605794395941989)<1e-12
    assert abs(agg['plena_os_total_us']['AM_us']-16.506925925925923)<1e-12
    assert abs(agg['issue_formula_total_us']['AM_us']-7.072592592592593)<1e-12
    counts=Counter(r['active_routed'] for r in b2)
    write_json('results/b2_summary.json',dict(statistics=agg,active_histogram=dict(sorted(counts.items())),
                   active_mean=mean([r['active_routed'] for r in b2]),task_counts_match_except_shared=True))
    inventory=[]
    for src in meta['sources']:
        path=Path(src['path']);ws=json.loads(path.read_text())['workloads']
        for b in sorted({w['batch'] for w in ws}):
            sub=[w for w in ws if w['batch']==b]
            inventory.append(dict(path=str(path),batch=b,count=len(sub),
                layers=','.join(map(str,sorted({w['layer'] for w in sub}))),
                decode_steps=','.join(map(str,sorted({w['step'] for w in sub}))),
                composition='distinct requests, one token each' if b<=16 else f'16 decode + {b-16} contiguous prefill',
                capture_prefix_smoke_windows=sum(isinstance(w.get('provenance'),dict) and w['provenance'].get('capture_prefix_smoke',False) for w in sub)))
    write_csv('trace_inventory.csv',inventory)
    assert frozen=={str(p.relative_to(R7)):sha(p) for p in R7.rglob('*') if p.is_file()}
    changes=subprocess.check_output(['git','diff','HEAD','--name-only','--',
            'research/moe_dispatch/round2','research/moe_dispatch/round3',
            'research/moe_dispatch/round4_compute_first','research/moe_dispatch/round5_largebatch_compute',
            'research/moe_dispatch/round6_feasible_cell'],cwd=REPO,text=True)
    assert not changes
    write_json('results/VALIDATION.json',dict(passed=True,gemm_comparisons=len(gemms),b2_windows=len(b2),
                original_windows_checked=len(windows),repeated_replays_identical=True,
                round7_all_files_unchanged=True,round2_to_6_tracked_files_unchanged=True,
                source_round7_hashes=frozen,stage2_executed=False,stage3_executed=False))
    print(json.dumps(dict(case=case['row'],b2_statistics=agg,active_histogram=dict(counts)),indent=2))

if __name__=='__main__':run()
