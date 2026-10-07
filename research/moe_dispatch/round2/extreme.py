"""E3 continuous CMA-ES workload search and full-delta revalidation.

Synthetic routing is used only to locate possible benefit regions, not as a
replacement for the captured model traces. Discrete batch/E/F/shared choices
are explicit projections of continuous CMA parameters and every hardware
search preserves full domain/open certificates. δ=0 requests exact traversal;
a time cap cannot be relabeled an optimality proof.
"""
from __future__ import annotations
import argparse,json,math
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def decode_parameters(unit,alpha_low,alpha_high):
    z=np.clip(np.asarray(unit,dtype=float),0,1)
    if len(z)!=6:raise ValueError("six continuous parameters required")
    # Batch powers and expert counts are integer workload semantics; the
    # continuous search has quantized plateaus rather than fake fractional tokens.
    batch=int(np.clip(round(2**(1+7*z[0])),2,256))
    alpha=math.exp(math.log(alpha_low)+(math.log(alpha_high)-math.log(alpha_low))*z[1])
    E=int(np.clip(round(64+192*z[3]),64,256))
    F=int(np.clip(32*round((512+1536*z[4])/32),512,2048))
    return {"batch":batch,"alpha":alpha,"shared_units":4*float(z[2]),"E":E,
            "topk":6 if E==64 else 8,"F":F,"bw":126.03076923076924*(1+3*float(z[5]))}


def make_workload(par,seed=20261007):
    from .regions import synthetic
    w=synthetic(par["batch"],par["alpha"],0,par["E"],par["topk"],par["F"],seed)
    sharedF=int(round(par["shared_units"]*par["F"]))
    if sharedF:
        w["experts"].insert(0,{"id":-1,"Me":par["batch"],"H":2048,"F":sharedF,
                              "is_shared":True,"token_indices":list(range(par["batch"]))})
    w["id"]="cma_"+w["id"]+f"_S{par['shared_units']:.8g}"
    return w


def _distance(w,par,dev):
    rows=[]
    def hist(v):
        h=np.bincount([e["Me"] for e in v["experts"] if not e.get("is_shared",False)],minlength=257).astype(float)+1e-9
        return h/h.sum()
    a=hist(w)
    for v in dev:
        b=hist(v);nb=sum(not e.get("is_shared",False) for e in v["experts"])
        rows.append({"window_id":v["id"],"batch":v["batch"],"batch_log2_distance":abs(math.log2(w["batch"])-math.log2(v["batch"])),
            "Me_hist_KL_synthetic_to_real":float(np.sum(a*np.log(a/b))),
            "distinct_synthetic":sum(not e.get("is_shared",False) for e in w["experts"]),"distinct_real":nb})
    nearest=min(rows,key=lambda r:(r["batch_log2_distance"],r["Me_hist_KL_synthetic_to_real"],r["window_id"]))
    return {"nearest_real_window":nearest,"all_development_distances":rows,
            "distance_scope":"token-count histogram and batch only; no claim synthetic prompts resemble captured language tasks"}


def _search_point(job):
    par,initial,seconds=job
    from .model import Parameters
    from .optimizer import search_workloads
    w=make_workload(par)
    p=Parameters(hbm_Bpc=par["bw"],credits=math.ceil(par["bw"]*65/32))
    result=search_workloads([w],p,delta=.02,time_limit_s=seconds,initial_designs=initial)
    result["resume_workloads"]=[w]
    return w,p,result


def run(args):
    import cma
    from .common import ROOT,inputs,write_json,write_csv,decode_design
    from .optimizer import search_workloads
    from .model import Parameters
    out=ROOT/"results/E3";ws=inputs();cal=json.loads((out/"synthetic_calibration.json").read_text())
    selection=json.loads((out/"FROZEN_SELECTION.json").read_text())
    initial=[decode_design(selection["modes"]["pipelined"][f]["design"]) for f in ("single","homogeneous","heterogeneous")]
    alpha_low=min(cal["levels"]);alpha_high=max(cal["levels"])
    # At most 500 objective calls, including duplicates; CMA remains causal
    # and deterministic for a fixed seed. Cache repeated projected workloads.
    budget=min(500,args.max_evaluations)
    es=cma.CMAEvolutionStrategy([.45,.5,.5,.35,.6,.35],.24,
        {"bounds":[0,1],"seed":20261007,"verbose":-9,"maxfevals":budget,"popsize":8})
    rows=[];best=None;cache={}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
      while not es.stop() and len(rows)<budget:
        points=es.ask()[:budget-len(rows)]
        pars=[decode_parameters(point,alpha_low,alpha_high) for point in points]
        keys=[json.dumps(par,sort_keys=True) for par in pars]
        missing={key:par for key,par in zip(keys,pars) if key not in cache}
        # Parallelism is within one generation only. All measured values are
        # delivered to tell before the next ask; there is no future feedback.
        jobs=[(par,initial,args.point_seconds) for par in missing.values()]
        results=pool.map(_search_point,jobs,chunksize=1)
        for key,result in zip(missing,results):cache[key]=result
        vals=[]
        for point,par,key in zip(points,pars,keys):
            w,p,result=cache[key]
            a=result["families"]["single"];b=result["families"]["heterogeneous"];h=result["families"]["homogeneous"]
            value=b["geomean_ms"]/a["geomean_ms"]-1
            row={"evaluation":len(rows),**par,"delta_vs_single":value,"delta_vs_homo":b["geomean_ms"]/h["geomean_ms"]-1,
                 "best_single_ms":a["geomean_ms"],"best_hetero_ms":b["geomean_ms"],"best_homo_ms":h["geomean_ms"],
                 "single_design":json.dumps(a["design"],sort_keys=True),"hetero_design":json.dumps(b["design"],sort_keys=True),
                 "proof_complete":result["proof_complete"],"gap_pct":result["gap_pct"],"full_domain_search":True}
            rows.append(row);vals.append(value)
            write_json(out/"cma_certificates"/f"evaluation_{row['evaluation']:04d}.json",result)
            if best is None or value<best[0]:best=(value,par,w,p,result)
        write_csv(out/"workload_cma_evaluations.csv",rows)
        print("CMA",len(rows),"/",budget,flush=True)
        if len(points)==es.popsize:es.tell(points,vals)
        else:break
    if best is None:raise RuntimeError("CMA made no workload evaluation")
    value,par,w,p,result=best
    finalinitial=[decode_design(result["families"][f]["design"]) for f in ("single","homogeneous","heterogeneous")]
    exact=search_workloads([w],p,delta=0,time_limit_s=args.verify_seconds,initial_designs=finalinitial)
    a=exact["families"]["single"];b=exact["families"]["heterogeneous"];h=exact["families"]["homogeneous"]
    write_csv(out/"workload_cma_evaluations.csv",rows)
    write_json(out/"workload_extreme.json",{"best_evaluated_workload_parameters":par,"workload":w,
        "delta_vs_single":b["geomean_ms"]/a["geomean_ms"]-1,"delta_vs_homo":b["geomean_ms"]/h["geomean_ms"]-1,
        "requested_verification_delta":0,"verification_full_domain_certificate":exact,
        "distance_from_real":_distance(w,par,ws["development"]),"CMA_evaluations":len(rows),"max_allowed_evaluations":500,
        "CMA_stop":{str(k):str(v) for k,v in es.stop().items()},"continuous_parameter_projection":
            "B rounded2..256; E rounded64..256; F nearest32; Shared continuous equivalent width rounded integer; no fractional tokens",
        "scope":"synthetic workload exploration; best evaluated candidate until all full-domain open regions close",
        "hypothetical_HBM_bandwidth_override":True})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--max-evaluations",type=int,default=500);ap.add_argument("--jobs",type=int,default=8)
    ap.add_argument("--point-seconds",type=float,default=1);ap.add_argument("--verify-seconds",type=float,default=120)
    run(ap.parse_args())
if __name__=="__main__":main()
