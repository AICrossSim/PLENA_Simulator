"""Compare actual oversized storage/spill paths to saved original sources."""
from __future__ import annotations
from dataclasses import replace
import time
from .check_equivalence import baseline_modules
from research.moe_dispatch.round3.common import ROOT, frozen_designs, canonical, digest, write_json, write_csv, sha
from research.moe_dispatch.round3.config import parameters
from research.moe_dispatch.round3 import model, runtime
from research.moe_dispatch.round2.predictors import Predictor
from research.moe_dispatch.round2.dispatch_fix.predictor_ablation_20261008.nominal_control import NominalPredictor


def run():
    before = baseline_modules()
    # Explicit synthetic controls only; do not present them as captured input.
    w={"id":"performance_synthetic_B256", "batch":256, "hidden":2048, "top_k":2,
       "experts":[{"id":-1,"Me":256,"H":2048,"F":512,"is_shared":True},
                  *[{"id":i,"Me":128,"H":2048,"F":512,"is_shared":False} for i in range(4)]]}
    wide={**w,"id":"performance_synthetic_H3072", "hidden":3072,
          "experts":[{**e,"H":3072} for e in w["experts"]]}
    rows=[];timing={False:[0.0,0.0],True:[0.0,0.0]}
    for mode in ("pipelined","port_tight"):
        ds=frozen_designs(mode);h=ds["best_hetero"]
        designs=[("B1",ds["B1"]),("best_hetero",h),
                 ("shared_hetero",replace(h,w_bytes=(0,0),landing_mode="shared",landing_pool_bytes=sum(h.w_bytes)))]
        p=parameters(mode)
        for name,d in designs:
            for workload in (w,wide):
                chunks=len(model.storage_chunks(workload));assert chunks>1
                for detail in (False,True):
                    for method,factory in (("ours",Predictor),("nominal",NominalPredictor),("owners",None)):
                        owners=[i%len(d.cores) for i in range(len(workload["experts"]))] if factory is None else None
                        oldpred=factory("ours") if factory is Predictor else factory() if factory else None
                        newpred=factory("ours") if factory is Predictor else factory() if factory else None
                        args=dict(dispatch="fixed",owners=owners,detail=detail)
                        start=time.monotonic();old=before["runtime"].simulate(workload,d,p,predictor=oldpred,**args);timing[detail][0]+=time.monotonic()-start
                        start=time.monotonic();new=runtime.simulate(workload,d,p,predictor=newpred,**args);timing[detail][1]+=time.monotonic()-start
                        assert canonical(old)==canonical(new),(mode,name,workload["id"],detail,method)
                        rows.append(dict(mode=mode,design=name,workload=workload["id"],storage_chunks=chunks,detail=detail,method=method,old_digest=digest(old),new_digest=digest(new),exact=True))
    out=ROOT/"diagnostics/performance"
    write_csv(out/"chunk_equivalence.csv",rows)
    receipt=dict(cases=len(rows),all_canonical_bitexact=True,source_sha256={n:sha(ROOT/n) for n in ("model.py","runtime.py","optimizer.py")},timing={str(k):dict(old_seconds=v[0],new_seconds=v[1]) for k,v in timing.items()},scope="Synthetic explicit storage-chunk/spill regression; not captured benchmark performance")
    write_json(out/"CHUNK_EQUIVALENCE.json",receipt);print(receipt,flush=True)

if __name__=="__main__":run()
