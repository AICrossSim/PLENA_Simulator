"""Exhaustive finite-domain geometry x dataflow search; selection uses dev only."""
import argparse, itertools, json, time
from collections import defaultdict
from dataclasses import asdict
from .common import ROOT, inputs, gm, digest, write_csv, write_json
from .config import Core,TOTAL_MACS,PK_VALUES
from .compute import FLOWS,pure,shape_id,shape_dict,allocate_real,cost

def candidates(cap):
    by=defaultdict(list)
    for m in range(1,17):
        for n in range(1,cap+1):
            for k in PK_VALUES:
                c=Core(m,n,k)
                if c.macs<=TOTAL_MACS: by[c.macs].append(c)
    for nc,name in ((1,'single'),(2,'homo2'),(4,'homo4')):
        for c in by[TOTAL_MACS//nc]:
            for f in FLOWS: yield name,(c,)*nc,(f,)*nc
    for ratio in (1,2,3,5,7):
        if TOTAL_MACS%(ratio+1): continue
        small=TOTAL_MACS//(ratio+1); large=TOTAL_MACS-small
        for a in by[large]:
            for b in by[small]:
                if ratio==1 and a>=b: continue
                for fs in itertools.product(FLOWS,repeat=2):
                    yield f'hetero{ratio}1',(a,b),fs

def run():
    out=ROOT/'E1'; out.mkdir(exist_ok=True)
    dev=inputs()['development']; selected={}; receipts=[]
    for cap in (256,512):
        start=time.monotonic(); rows=[]; best={}
        for index,(family,cs,fs) in enumerate(candidates(cap)):
            # Exact repeated execution for every geometry/dataflow candidate.
            a=[pure(w,cs,fs,False) for w in dev]
            b=[pure(w,cs,fs,False) for w in dev]
            assert a==b
            score=gm(a)
            row=dict(pn_cap=cap,family=family,geometry=shape_id(cs),flows='/'.join(fs),
                     dev_GM_cycles=score,repeat_sha256=digest(a),repeat_exact=True)
            rows.append(row)
            keys=[family]
            if family=='single': keys.append('single_pertask' if fs==('PER_TASK',) else 'single_fixed')
            # All fixed-flow minima retained for the dataflow attribution.
            if family=='single': keys.append('single_'+fs[0].lower())
            for key in keys:
                candidate=(score,shape_id(cs),fs)
                if key not in best or candidate<best[key][0]: best[key]=(candidate,cs,fs)
            if index%5000==0: print(f'PN{cap}: {index} candidates, {time.monotonic()-start:.1f}s',flush=True)
        for family,(_,cs,fs) in best.items():
            name=f'pn{cap}_{family}'
            try:
                real=asdict(allocate_real(cs,fs,name)); feasible=True; reason=''
            except ValueError as e:
                real=None; feasible=False; reason=str(e)
            selected[name]=dict(design=name,family=family,pn_cap=cap,cores=shape_dict(cs),
                flows=list(fs),macs=[c.macs for c in cs],dev_GM_cycles=best[family][0][0],
                real_sram_design=real,real_capacity_feasible=feasible,capacity_reason=reason)
        write_csv(out/f'search_pn{cap}.csv',rows)
        receipts.append(dict(pn_cap=cap,candidates=len(rows),windows=18,runs_each=2,
                             sha256=digest(rows),wall_seconds=time.monotonic()-start))
        print(f'PN{cap} finished: {len(rows)} candidates',flush=True)
    write_json(out/'selected_designs.json',selected)
    write_csv(out/'infeasible_families.csv',[
        dict(pn_cap=cap,family=f'hetero{r}1',total_macs=TOTAL_MACS,status='no_exact_integer_split',
             reason=f'{TOTAL_MACS} is not divisible by {r+1}; no rounding is permitted')
        for cap in (256,512) for r in (2,5)])
    write_json(out/'search_receipt.json',dict(runs=receipts,total_macs=TOTAL_MACS,
        illegal_families={'hetero21':'65536 is not divisible by 3','hetero51':'65536 is not divisible by 6'},
        objective='geometric mean across all 18 development windows',
        optimality_scope='exhaustive declared shape/flow grid, fixed deterministic scheduler; not schedule/PPA optimum',
        inputs_hash=digest(dev)))
    print(json.dumps({k:(v['cores'],v['flows'],v['dev_GM_cycles'],v['real_capacity_feasible']) for k,v in selected.items()},indent=2),flush=True)

if __name__=='__main__': run()
