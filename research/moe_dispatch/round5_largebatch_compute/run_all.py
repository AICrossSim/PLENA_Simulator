"""Resumable orchestrator. Model capture must finish before results are read."""
import argparse,json,os,subprocess,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from .common import ROOT,BUDGETS,write_json

def case(args):
    total,mode,gran,refine,work=args
    from .inputs import load
    from .search import run_case
    from .evaluate import evaluate_case
    data=load();run_case(total,mode,gran,data['development'],refine_limit=refine)
    evaluate_case(ROOT/'results'/f'{total}_{mode}_{gran}',data['heldout'],work)
    from .runtime_fix import retrofit
    retrofit(ROOT/'results'/f'{total}_{mode}_{gran}')
    return [total,mode,gran]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--refine',type=int,default=12);ap.add_argument('--work',type=float,default=.08);a=ap.parse_args()
    print('Waiting for real model capture to complete',flush=True)
    while True:
        p=ROOT/'inputs/capture_manifest.json'
        if p.exists():
            try:
                if json.loads(p.read_text())['complete']:break
            except (json.JSONDecodeError,KeyError):pass
        time.sleep(20)
    from .inputs import build
    build();start=time.monotonic()
    args=[(b,m,g,a.refine,a.work) for b in BUDGETS for m in ('matrix','full_scaled') for g in ('expert','shared64')]
    errors=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(case,x):x for x in args}
        for future in as_completed(futures):
            try:print('CASE COMPLETE',future.result(),flush=True)
            except Exception:
                error=dict(case=futures[future],traceback=traceback.format_exc());errors.append(error)
                print('CASE FAILED',error,flush=True)
    if errors:
        write_json(ROOT/'FAILED_CASES.json',errors)
        raise RuntimeError(f'{len(errors)} cases failed; completed independent cases remain resumable')
    from .pack_bounds import pack
    pack()
    from .report import build as report
    report()
    from .audit import audit
    audit()
    write_json(ROOT/'run_receipt.json',dict(cases=len(args),wall_seconds=time.monotonic()-start,complete=True))
    print('ALL RESULTS AND AUDIT COMPLETE',flush=True)

if __name__=='__main__':main()
