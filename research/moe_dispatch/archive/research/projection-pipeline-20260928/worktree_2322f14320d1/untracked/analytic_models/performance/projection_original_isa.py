"""Source-bound whole-program search over existing bounded Matrix opcodes.

M_MM's transpose form is M_TMM. This experiment extends the explicit
rectangular-view service contract; it does not certify untouched PLENA RTL.
Projection hardware extensions are disabled in every arm. All candidates
are priced as complete programs, including real memory events. Mixed choices
are re-executed and may lose to a uniform schedule. No stage minima sum.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from functools import partial
from dataclasses import replace, asdict
import hashlib
import json
from pathlib import Path
from .ltile_platform import ExecutionProfile
from .ltile_dma import DmaBackend
from .ltile_services import Services
from .ltile_layers import compiler_api
from .projection_campaign import write_csv


def execute(spec,args,templates=(),control='fsm'):
    compiler_api(args.compiler)
    kind,batch,schedule,k=spec
    profile=ExecutionProfile(hbm_controllers=16,projection_schedule=schedule,
                             projection_k_tile=k,projection_template_overrides=templates)
    backend=DmaBackend(args.memory_root/'ltile_memory',args.memory_root/'ramulator.json',args.cache/'memory')
    r=Services(args.compiler,profile,backend,args.cache/'services').layer(
        kind,batch,control,'BF16',supply='native' if control=='fsm' else 'packed')
    assert not profile.matrix.weight_replay and profile.matrix.projection_segments==1
    assert r['components']['total']==sum(r['components'][x] for x in ('issue','scalar','sram','arithmetic','dependency','dma'))
    assert sum(s['total'] for s in r['sections'])==r['components']['total']
    name=f'{kind}_b{batch}_{schedule}_k{k}_{"mixed" if templates else "uniform"}_{control}'
    path=args.output/(name+'.json');path.write_text(json.dumps(r,separators=(',',':'))+'\n')
    categories={s['name']:s['category'] for s in r['metadata']['stages']}
    projection=sum(s['total'] for s in r['sections'] if categories[s['name']]=='projection')
    row=dict(model=kind,batch=batch,schedule=schedule,k_tile=k,mixed=bool(templates),control=control,
             projection_cycles=projection,**r['components'],hbm_read_bytes=r['hbm_read_bytes'],hbm_write_bytes=r['hbm_write_bytes'],
             result=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    print(name,row['total'],flush=True)
    return row,r,profile


def finalize_group(pool, args):
    """Reprice a composed program, then pair its projection with both recurrences."""
    compiler_api(args.compiler)
    m, b = pool[0][0]['model'], pool[0][0]['batch']
    names=[s['name'] for s in pool[0][1]['metadata']['stages'] if s['matrix_shape']]
    selections=[]
    for name in names:
        best=min(pool,key=lambda item:next(s['total'] for s in item[1]['sections'] if s['name']==name))
        selections.append((name,best[2].projection_schedule,best[2].projection_k_tile))
    uniform=min(pool,key=lambda item:item[0]['total'])
    mixed=execute((m,b,'transposed',1024),args,tuple(selections))
    best=min((uniform,mixed),key=lambda item:item[0]['total'])
    paired=execute((m,b,best[2].projection_schedule,best[2].projection_k_tile),args,best[2].projection_template_overrides,'old_isa')
    total,old=best[0]['total'],paired[0]['total']
    mv,tmv,mm=(next(item[0] for item in pool if item[0]['schedule']==s) for s in ('resident','transposed','matrix'))
    table=dict(model=m,batch=b,previous_tmv_native_ms=tmv['total']/1e6,
        selected_vector_ms=old/1e6,selected_native_ms=total/1e6,
        compiler_speedup_recurrence_fixed=tmv['total']/total,recurrent_speedup_projection_fixed=old/total,
        selected_projection_ms=best[0]['projection_cycles']/1e6,
        resident_projection_ms=mv['projection_cycles']/1e6,tmv_projection_ms=tmv['projection_cycles']/1e6,
        mm_projection_ms=mm['projection_cycles']/1e6,
        selected_hbm_bytes=best[0]['hbm_read_bytes']+best[0]['hbm_write_bytes'],result=best[0]['result'])
    choice=dict(model=m,batch=b,profile=asdict(best[2]),selection=selections,
                mixed_selected=best is mixed,vector=paired[0]['result'],native=best[0]['result'])
    stages=[dict(model=m,batch=b,arm=arm,**s) for arm,r in (('vector',paired[1]),('native',best[1])) for s in r['sections']]
    return [mixed[0],paired[0]],table,stages,choice


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--compiler',required=True,type=Path)
    p.add_argument('--memory-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--batches',nargs='+',type=int,default=[1,2,4,8,16])
    p.add_argument('--resume',action='store_true',help='Resume only an unfinished campaign using source-bound caches')
    args=p.parse_args()
    if (args.output/'manifest.json').exists():
        raise ValueError('completed campaigns are immutable')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    compiler_api(args.compiler)
    config=json.loads((args.memory_root/'ramulator.json').read_text())
    if len(config['memory_system']['controllers'])!=16:
        raise ValueError('this paired campaign requires the frozen 16-controller memory profile')
    specs=[(m,b,s,k) for m in ('mamba','kda') for b in args.batches
           for s,k in (('resident',256),('transposed',1024),('matrix',1024))]
    with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
        observed=list(pool.map(partial(execute,args=args),specs))
    candidates=[row for row,_,_ in observed]; tables=[]; stages=[]; choices=[]
    groups=[[(row,r,h) for row,r,h in observed if row['model']==m and row['batch']==b]
            for m in ('mamba','kda') for b in args.batches]
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context("spawn")) as pool:
        finalized=list(pool.map(partial(finalize_group,args=args),groups))
    for extra,table,stage_rows,choice in finalized:
        candidates.extend(extra);tables.append(table);stages.extend(stage_rows);choices.append(choice)
    write_csv(args.output/'candidates.csv',candidates)
    write_csv(args.output/'summary.csv',tables)
    write_csv(args.output/'stages.csv',stages)
    (args.output/'selection.json').write_text(json.dumps(choices,indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps(dict(clock_hz=10**9,weight='BF16',
        scope='input norm through output projection; excludes outer residual/MoE/full model',
        projection_hardware_added_bytes=0,dma_status='32/32 candidate; original RTL does not certify credits',
        matrix_abi='bounded edge4 / K1024 rectangular view; original instruction words, candidate BF16 service',
        memory_sha256=hashlib.sha256((args.memory_root/'ramulator.json').read_bytes()).hexdigest(),
        sources=observed[0][1]['identity']['sources']),indent=2)+'\n')

if __name__=='__main__':main()
