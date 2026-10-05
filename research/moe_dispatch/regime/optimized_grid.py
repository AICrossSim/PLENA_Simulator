"""Reoptimize the single baseline before judging each regime's headroom."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,json
from .search import development,init_worker,point,score_point,settings_for,source_hashes,GROUPS
from .campaign import evaluate,result_rows,aggregate
from ..geometry3d.compute import enumerate_geometries
from ..geometry3d.study import cores_from,write_json,write_csv


def job(args): return score_point(*args)


def run(inputs,out,workers=16):
    out.mkdir(parents=True,exist_ok=True);dev=development(inputs)
    singles=[g for g in enumerate_geometries() if len(g)==1]
    rows=[];frozen=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(dev,)) as pool:
        for credits in (256,512):
            for fmt in ('BF16','W8','W4'):
                tasks=[(point(g,flow=flow,slots=slots,gu=gu,down=down),credits,fmt) for g in singles
                        for flow in ('bounded_ws','bounded_os') for slots in (2,6,32)
                        for gu in GROUPS for down in GROUPS]
                got=list(pool.map(job,tasks,chunksize=8));rows +=[{'credits':credits,'weight_format':fmt,**r} for r in got]
                best=min((r for r in got if r['legal']),key=lambda r:(r['score_ms'],r['geometry'],r['flow'],r['prefetch_slots']))
                frozen.append({'credits':credits,'weight_format':fmt,**best})
                print(credits,fmt,best['geometry'],best['flow'],best['prefetch_slots'],best['score_ms'],flush=True)
    write_json(out/'FROZEN_OPTIMIZED_SINGLE.json',{'points':frozen,'heldout_used_for_selection':False,
        'scope':'all44singlegeometries x2boundedflows x3prefetchdepths x25GU/Down group limits,valid SRAM spans; same development objective',
        'source_sha256_before_heldout':source_hashes(),
        'development_input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.glob('*development.json')}})
    write_json(out/'single_development_candidates.json',rows)
    held=[w for name in ('heldout.json','mixed_heldout.json') for w in json.loads((inputs/name).read_text())['workloads']]
    results=[]
    for p in frozen:
        s=settings_for(p['credits'],p['weight_format'],p)
        for split,ws in (('development',dev),('heldout',held)):
            results+=result_rows(evaluate(ws,cores_from(p['geometry']),s,p['policy']),s,
                credits=p['credits'],weight_format=p['weight_format'],split=split,geometry=p['geometry'])
    write_csv(out/'optimized_grid_windows.csv',results)
    grid=aggregate(results,('credits','weight_format','split','batch','geometry'))
    for r in grid:
        r['search_eligible_development']=r['split']=='development' and r['architecture_headroom_percent']>=10
        r['space_status']='candidate_space' if r['architecture_headroom_percent']>=10 else ('below_3pct_assumed_margin' if r['architecture_headroom_percent']<3 else 'limited_space_3_to_10pct')
        r['scope']='search bound uses unique weights/peak MAC/necessary minimum port traffic; conditional bound uses actual mapped traffic'
    write_csv(out/'optimized_grid.csv',grid)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();run(a.inputs,a.out,a.workers)
