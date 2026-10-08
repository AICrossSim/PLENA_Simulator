"""Check previously known feasible WS witnesses against new DSE incumbents."""
from __future__ import annotations
from dataclasses import asdict
import json
from .common import ROOT, inputs, frozen_designs, write_csv, write_json, metadata, canonical
from .config import MODES, parameters
from .search import _evaluate, matches, FAMILIES


def run():
    dev=inputs()["development"];rows=[]
    for mode in MODES:
        for name,d in frozen_designs(mode,common_ws=True).items():
            if name not in ("B1","B2","best_hetero"):
                continue
            result,status,raw,calls=_evaluate(d,dev,parameters(mode),10)
            assert status=="evaluated"
            family=next(f for f in FAMILIES if matches(d.cores,f))
            certificate=ROOT/"E4/certificates"/f"{mode}_C0_{family.replace('+','_')}.json"
            incumbent=json.loads(certificate.read_text())["selected"]["score_ms"] if certificate.exists() else None
            rows.append({"onchip_mode":mode,"design":name,"family":family,
                "old_common_WS_dev_ms":result["score_ms"],"new_incumbent_dev_ms":incumbent,
                "known_witness_better":incumbent is not None and result["score_ms"]<incumbent-1e-12,
                "repeat_identical":True,"simulator_calls":calls,
                "hardware":canonical(asdict(d)),"latencies_ms":canonical(result["latencies_ms"]),
                "allocation_statuses":canonical(result["solver_statuses"])})
    out=ROOT/"E4/diagnostics"
    write_csv(out/"incumbent_check.csv",rows)
    write_json(out/"INCUMBENT_AUDIT.json",metadata({"scope":"18developmentwindows; commonWSallorganizations",
        "rows":len(rows),"known_better_count":sum(r["known_witness_better"] for r in rows)}))


if __name__=="__main__":
    run()
