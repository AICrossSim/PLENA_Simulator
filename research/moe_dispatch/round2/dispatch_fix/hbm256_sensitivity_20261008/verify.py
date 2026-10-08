"""Read-only independent validation of the 520-credit sensitivity outputs."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

from ...common import ROOT, BATCHES, decode_design, encode_design, gmean, inputs, sha, write_json

HERE=Path(__file__).resolve().parent


def main():
    with (HERE/"per_window.csv").open() as f: rows=list(csv.DictReader(f))
    with (HERE/"compare.csv").open() as f: compare=list(csv.DictReader(f))
    metadata=json.loads((HERE/"METADATA.json").read_text())
    receipts=json.loads((HERE/"repeat_checks.json").read_text())
    frozen=json.loads((HERE.parent/"frozen_designs.json").read_text())
    checked=0
    def check(condition,label):
        nonlocal checked
        assert condition,label
        checked+=1
    check(len(rows)==1620,"12x135 rows")
    check(len(compare)==96,"12x8 summaries")
    check(len(receipts)==12,"12 repeat receipts")
    check(len({(r["onchip_mode"],r["design"],r["predictor"],r["window_id"]) for r in rows})==1620,"unique rows")
    ws=inputs()
    expectedids={w["id"] for w in ws["heldout"]}
    for receipt in receipts:
        m,n,p=receipt["onchip_mode"],receipt["design"],receipt["predictor"]
        rs=[r for r in rows if (r["onchip_mode"],r["design"],r["predictor"])==(m,n,p)]
        check({r["window_id"] for r in rs}==expectedids,"all frozen inputs")
        check(receipt["heldout_digest"]==receipt["repeat_heldout_digest"],"full repeat")
        check(receipt["warmup_digest"]==receipt["repeat_warmup_digest"],"warm repeat")
        check(receipt["final_predictor_state_digest"]==receipt["repeat_final_predictor_state_digest"],"state repeat")
        old=encode_design(decode_design(frozen["modes"][m][n])); hw=receipt["hardware"]
        # JSON manifests list tuples while dataclass encode returns tuples.
        check(json.loads(json.dumps({k:v for k,v in old.items() if k!="flows"}))=={k:v for k,v in hw.items() if k!="flows"},"frozen compute/SRAM")
        check(all(x=="WS" for x in hw["flows"]),"common WS")
        check(receipt["parameters"]["credits"]==520 and receipt["parameters"]["hbm_Bpc"]==256,"520/256 parameters")
    for r in rows:
        check(r["result_digest"]==r["repeat_digest"],"perwindow digest")
        cycles=float(r["cycles"]); by=int(r["hbm_bytes"])
        check(math.isclose(float(r["latency_ms"]),cycles/1e6,rel_tol=1e-14),"ns/ms")
        check(math.isclose(float(r["achieved_hbm_GB_s"]),by/cycles,rel_tol=1e-14),"bandwidth identity")
        check(by/cycles<=256+1e-7,"shared bandwidth cap")
    for row in compare:
        rs=[r for r in rows if (r["onchip_mode"],r["design"],r["predictor"])==
            (row["onchip_mode"],row["design"],row["predictor"]) and (row["batch"]=="all" or r["batch"]==row["batch"])]
        check(len(rs)==int(row["windows"]),"summary windows")
        check(math.isclose(gmean(float(r["latency_ms"]) for r in rs),float(row["new256_geomean_ms"]),rel_tol=1e-13),"GM recomputed")
        check(math.isclose(float(row["ratio_new_vs_old"]),float(row["new256_geomean_ms"])/float(row["old126_geomean_ms"]),rel_tol=1e-13),"paired ratio identity")
    for name,value in metadata["source_sha256"].items(): check(sha(ROOT/name)==value,"unchanged source/input baseline")
    with gzip.open(HERE/"task_predictions.csv.gz","rb") as f: raw=f.read()
    check(hashlib.sha256(raw).hexdigest()==metadata["task_records_raw_sha256"],"raw gzip hash")
    check(sha(HERE/"task_predictions.csv.gz")==metadata["task_records_compressed_sha256"],"gzip hash")
    taskrows=list(csv.DictReader(raw.decode().splitlines()))
    check(len(taskrows)==metadata["task_prediction_rows"],"task count")
    check(set((r["onchip_mode"],r["design"],r["predictor"]) for r in taskrows)==set((r["onchip_mode"],r["design"],r["predictor"]) for r in rows),"task config coverage")
    write_json(HERE/"VERIFY.json",{"checks_passed":checked,"failed":[],"window_rows":len(rows),"task_rows":len(taskrows),
        "source_and_inputs_unchanged":True,"compute_private_storage_unchanged":True,"all_repeats_exact":True,
        "native_hbm_calibrated":False,"area_claim_eligible":False})
    print({"checks_passed":checked,"failed":[],"tasks":len(taskrows)},flush=True)


if __name__=="__main__": main()
