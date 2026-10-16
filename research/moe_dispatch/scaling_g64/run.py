#!/usr/bin/env python3
"""Execute fixed scaling matrix on every captured held-out window, twice."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .configs import configurations, serializable_configs, POLICIES, PROFILES
from .runtime import schedule
from research.moe_dispatch.plena65536_moe.workloads import load_workloads, canonical_sha256
from research.moe_dispatch.plena65536_shared.timing import resource_ledger


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--profiles",nargs="+",default=list(PROFILES))
    args=ap.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"records.jsonl").exists():
        raise RuntimeError("Existing results preserved. Choose an unused output directory.")
    start=time.monotonic()
    windows=load_workloads("heldout")["heldout"]
    assert len(windows)==135
    configs=configurations()
    config_data=serializable_configs(configs)
    for c, data in zip(configs,config_data):
        data["resources"]=resource_ledger(c["cores"])
    dump(out/"configs.json",config_data)
    dump(out/"PROTOCOL.json",{
        "frequency_hz":1000000000,"data_supply_latency_cycles":0,
        "memory_timing_modeled":False,"compute_datapath_native_calibrated":False,
        "BF16_FP32_numerical_datapath_in_timing_model":False,
        "window_count":len(windows),"window_counts_by_batch":dict(Counter(w["batch"] for w in windows)),
        "windows_sha256":canonical_sha256(windows),"window_ids":[w["id"] for w in windows],
        "workload":"Shared + all active routed experts, real archived DeepSeek-V2-Lite Top6",
        "heldout_reused_in_prior_research":True,"hardware_selection_on_heldout":False,
        "same_config_all_batches":True,"threshold":2,"queue_depth_per_logical_group":8,
        "context_cycles":4,"policies":POLICIES,"profiles":args.profiles,
        "task_granularity":"one entire expert, 2 Gate/Up +1 Down; no concurrent within-expert split",
        "input_order":"Shared first, then routed expert ID, no offline LPT",
        "geometry_mapping":"G=MLEN/BLEN, BLEN=PM, MLEN=PM*64, PN/BLEN column replicas",
        "aggregation":"GM within each batch then equal-weight GM across7batches",
        "repetitions":2,"comparison_scope":"iso MACs, not iso synthesized area/energy",
        "source_paper":"https://arxiv.org/html/2509.09505v3",
        "unsupported_prompt_claims_removed":["guaranteed heterogeneous win","1GHz single-cycle G64 reduction proof",
             "G64 guarantees all K divide MLEN","one4-rowE equals two independent2-rowE",
             "1:1MAC allocation guarantees50:50usefulwork","MACcounts equivalentto TPUv4/v6"],
    })
    count=0
    aggregate=hashlib.sha256()
    point_checks=[]
    with (out/"records.jsonl").open("w") as stream:
        for config in configs:
            for profile in args.profiles:
                if profile not in PROFILES:raise ValueError(profile)
                policy_items = POLICIES.items() if profile == "pipelined_tree1" else [("full",POLICIES["full"])]
                for policy,kwargs in policy_items:
                    for window in windows:
                        common=dict(groups=config["groups"],profile=profile,threshold=2,queue_depth=8,
                                    context_cycles=4,details=True,**kwargs)
                        first=schedule(config["cores"],window,**common)
                        second=schedule(config["cores"],window,**common)
                        a=json.dumps(first,sort_keys=True,separators=(",",":"),allow_nan=False)
                        b=json.dumps(second,sort_keys=True,separators=(",",":"),allow_nan=False)
                        assert a==b,(config["design_id"],profile,policy,window["id"])
                        expected=sum(3*e["Me"]*e["H"]*e["F"] for e in window["experts"])
                        assert expected==first["useful_MACs"]
                        assert expected <= config["scale"]*first["cycles"]
                        row=dict(design_id=config["design_id"],scale=config["scale"],role=config["role"],
                                 variant=config["variant"],profile=profile,policy=policy,window_id=window["id"],
                                 batch=window["batch"],cores=[c.to_dict() for c in config["cores"]],
                                 groups=[list(g) for g in config["groups"]],result=first)
                        line=json.dumps(row,sort_keys=True,separators=(",",":"),allow_nan=False)
                        stream.write(line+"\n")
                        aggregate.update((line+"\n").encode())
                        point_checks.append([config["design_id"],profile,policy,window["id"],hashlib.sha256(a.encode()).hexdigest()])
                        count+=1
                stream.flush()
                print(f"Completed {config['design_id']} {profile}; points={count}, repeats={2*count}",flush=True)
    dump(out/"repeat_sha256.json",point_checks)
    source_root=Path(__file__).resolve().parent
    dump(out/"run_receipt.json",{
        "completed":True,"unique_points":count,"executions":count*2,
        "all_repetitions_byte_identical":True,"records_sha256":aggregate.hexdigest(),
        "elapsed_seconds":time.monotonic()-start,
        "branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(source_root.glob("*.py"))},
    })
    print(f"Done: {count} points, all twice identical. {out}",flush=True)


if __name__=="__main__":main()
