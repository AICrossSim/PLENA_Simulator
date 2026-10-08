"""Interpret and fairly union two equally-budgeted development campaigns.

This report pass distinguishes local5% incumbent bounds from the requested
joint5% improvement gate against B1/B2, fixes common-reference robust metrics,
and avoids claiming simultaneous independent slot capacity in a shared pool.
As-run certificates stay unchanged. C0 selects all actual C0+C1 witnesses;
C1 selects the legal subset of that same union, so a finite sampling policy
cannot make a stricter constraint appear physically better than its superset.
"""
from __future__ import annotations
import json
from dataclasses import asdict
import numpy as np

from .common import ROOT, inputs, write_csv, write_json, read_csv, metadata, canonical
from .config import MODES, SEED
from .search import FAMILIES, FAMILY_LABELS, decode, key, c1_legal
from ..round2.robust import objectives, bootstrap_choices


def excluded_joint_five_percent(lb, b1, b2):
    # Equality still permits exactly5%, which meets the user gate.
    x1=lb>.95*b1+1e-12
    x2=lb>.95*b2+1e-12
    return x1,x2,x1 or x2


def union_witnesses(c0, c1):
    """Union actual evaluations, validating duplicates without replay reuse.

    Both original campaigns have already executed both entire searches and
    both physical evaluations per point. This reporting function does not
    claim additional simulator calls and never overwrites as-run certificates.
    """
    points={}
    for group,result in (("C0",c0),("C1",c1)):
        assert result["entire_search_repeat_identical"]
        for row in result["witnesses"]:
            ident=key(decode(row["design"]))
            if ident in points:
                old=points[ident]
                assert old["status"]==row["status"],"duplicate evaluation status changed"
                if row["status"]=="evaluated":
                    for field in ("latencies_ms","score_ms","solver_statuses","allocation_optimal"):
                        assert canonical(old[field])==canonical(row[field]),"duplicate numerical result changed"
                old["generated_by"].append(group)
            else:
                points[ident]={**row,"design":json.loads(ident),"generated_by":[group]}
            if row["status"]=="evaluated":
                assert row["repeat_identical"] and len(row["latencies_ms"])==18
    return [points[k] for k in sorted(points)]


def run():
    out=ROOT/"E4"
    original_path=out/"selected_designs_as_run.json"
    if original_path.exists():
        selection=json.loads(original_path.read_text())
    else:
        selection=json.loads((out/"selected_designs.json").read_text())
        write_json(original_path,selection)
    dev=inputs()["development"]
    expected_ids=[w["id"] for w in dev]
    assert len(dev)==18
    assert selection["selection_metadata"]["development_ids"]==expected_ids
    protocol=json.loads((out/"DSE_PROTOCOL.json").read_text())
    assert protocol["jobs"]==20 and protocol["entire_search_repeats"]==2
    assert protocol["selection"]["development_ids"]==expected_ids
    certs={}
    for mode in MODES:
        for group in ("C0","C1"):
            for family in FAMILIES:
                path=out/"certificates"/f"{mode}_{group}_{family.replace('+','_')}.json"
                certs[mode,group,family]=json.loads(path.read_text())
                assert len(certs[mode,group,family]["selected"]["latencies_ms"])==18
                assert certs[mode,group,family]["entire_search_repeat_identical"]
                label=FAMILY_LABELS[family]
                assert canonical(certs[mode,group,family]["selected"]["design"])==canonical(selection["modes"][mode][group][label])
    final={};union_meta=[];repeat_rows=[]
    for mode in MODES:
        for family in FAMILIES:
            original0=certs[mode,"C0",family];original1=certs[mode,"C1",family]
            points=union_witnesses(original0,original1)
            assert canonical(points)==canonical(union_witnesses(original0,original1)),"union selection input repeat mismatch"
            successful=[r for r in points if r["status"]=="evaluated"]
            calls=sum(r["total_campaign_simulator_calls"] for r in (original0,original1))
            generated=sum(r["evaluated_points"] for r in (original0,original1))
            union_meta.append({"onchip_mode":mode,"family":family,
                "candidate_generation_budget":sum(r["candidate_budget"] for r in (original0,original1)),
                "node_generation_budget":sum(r["node_budget"] for r in (original0,original1)),
                "generated_points_with_overlap":generated,"unique_attempted_points":len(points),
                "unique_successful_points":len(successful),"actual_simulator_calls":calls})
            for group in ("C0","C1"):
                pool=[r for r in successful if group=="C0" or c1_legal(decode(r["design"]))]
                assert pool,"union must contain a legal repeated witness"
                best=min(pool,key=lambda r:(r["score_ms"],key(decode(r["design"]))))
                source=certs[mode,group,family]
                label=FAMILY_LABELS[family]
                selection["modes"][mode][group][label]=best["design"]
                selection["development_scores_ms"][mode][group][label]=best["score_ms"]
                final[mode,group,family]={"family":family,"constraint_group":group,
                    "onchip_mode":mode,"selected":best,"witnesses":pool,
                    "parameters":source["parameters"],"candidate_budget":512,"node_budget":4096,
                    "evaluated_points":len(points),"successful_points":len(pool),
                    "visited_nodes":sum(r["visited_nodes"] for r in (original0,original1)),
                    "declared_lattice_points":source["declared_lattice_points"],
                    "pruned_lattice_points":source["pruned_lattice_points"],
                    "certificate":source["certificate"],
                    "simulator_calls":calls//2,"total_campaign_simulator_calls":calls,
                    "global_lb_ms":source["global_lb_ms"],
                    "gap_pct":100*(best["score_ms"]/source["global_lb_ms"]-1),
                    "proof_B_closed":source["proof_B_closed"],"open_regions":source["open_regions"],
                    "scope":"best eligible witness in union of two equally-budgeted actual development campaigns",
                    "as_run_sources":[f"certificates/{mode}_{g}_{family.replace('+','_')}.json" for g in ("C0","C1")],
                    "union_counts":union_meta[-1],"eligible_unique_points":len(pool),
                    "physical_call_accounting":"union source calls counted once per mode/family; C0/C1 are two selections of same executed pool",
                    "frontier_scope":"original constraint group's as-run full-domain frontier; valid pruning retained, no new proof closure from union",
                    "no_additional_simulator_calls_in_report_pass":True,
                    "entire_search_repeat_identical":True}
                assert best["score_ms"]<=source["selected"]["score_ms"]+1e-12
                repeat_rows.append({"onchip_mode":mode,"constraint_group":group,"family":family,
                    "both_C0_C1_entire_search_repeat_identical":True,
                    "every_eligible_witness_physical_repeat_identical":all(w["repeat_identical"] for w in pool),
                    "union_repeat_identical":True,"final_selected_design":canonical(best["design"]),
                    "repeat_scope":"both original campaigns physically rerun; union reporting repeated without claiming new physical calls"})
            assert final[mode,"C0",family]["selected"]["score_ms"]<=final[mode,"C1",family]["selected"]["score_ms"]+1e-12
    for mode in MODES:
        scores=selection["development_scores_ms"][mode]["C0"]
        selection["best_hetero_by_mode"][mode]=min(("H51","H42","H33"),key=lambda name:(scores[name],name))
    proof=[];stability=[];diagnostic=[];progress=[]
    for mode in MODES:
        common=np.asarray(final[mode,"C0","single"]["selected"]["latencies_ms"])
        for group in ("C0","C1"):
            b1=final[mode,group,"single"]["selected"]["score_ms"]
            b2=final[mode,group,"homogeneous"]["selected"]["score_ms"]
            for family in FAMILIES:
                r=final[mode,group,family];lb=r["global_lb_ms"]
                x1,x2,joint=excluded_joint_five_percent(lb,b1,b2)
                r["proof_A_closed"]=joint
                write_json(out/"final_certificates"/f"{mode}_{group}_{family.replace('+','_')}.json",r)
                proof.append({"onchip_mode":mode,"constraint_group":group,"family":family,
                    "proof_A_delta":.05,"proof_A_closed":joint,"proof_B_delta":0,
                    "proof_B_closed":r["proof_B_closed"],"remaining_open_regions":len(r["open_regions"]),
                    "gap_pct":r["gap_pct"],"global_lb_ms":lb,"baseline_B1_ms":b1,"baseline_B2_ms":b2,
                    "max_gain_vs_B1_pct":100*(1-lb/b1),"max_gain_vs_B2_pct":100*(1-lb/b2),
                    "exclude_5pct_vs_B1":x1,"exclude_5pct_vs_B2":x2,
                    "local_5pct_incumbent_certificate":lb>=.95*r["selected"]["score_ms"],
                    "proof_A_scope":"exclude jointly >=5% faster than development-selected B1 andB2; one excluded baseline suffices",
                    "proof_B_scope":"fixed allocation algorithm + physical LPT replay objective; not joint temporal scheduling optimum",
                    "status":"proofB closed" if r["proof_B_closed"] else "equal deterministicwork; full domain open"})
                label=FAMILY_LABELS[family]
                details=selection["design_details"][mode][group][label]
                design=decode(selection["modes"][mode][group][label])
                details.update(geometry=design.geometry,landing_mode=design.landing_mode,
                    landing_pool_KiB=design.landing_pool_bytes/1024,
                    capacities_KiB={field:[x/1024 for x in getattr(design,field)] for field in ("w_bytes","x_bytes","acc_bytes","z_bytes")},
                    shape_equal_homogeneous=len(design.cores)==2 and design.cores[0]==design.cores[1])
                maxima=[design.effective_w_bytes(i)//c.w_slice_bytes for i,c in enumerate(design.cores)]
                details.update(W_slots=[design.w_bytes[i]//c.w_slice_bytes for i,c in enumerate(design.cores)],
                    isolated_max_slots=maxima,proof_A_closed=joint,
                    shared_current_reserved_B=sum(c.w_slice_bytes for c in design.cores) if design.landing_mode=="shared" else 0,
                    slot_note="shared: privateW=0; isolatedmaxima notsimultaneous capacities; common byte pool counted once" if design.landing_mode=="shared" else "privatephysicalslots")
                pool=r["witnesses"]
                pool.sort(key=lambda row:key(decode(row["design"])))
                matrix=[np.asarray(row["latencies_ms"])/common for row in pool]
                counts=bootstrap_choices(matrix,[w["batch"] for w in dev],draws=200,seed=SEED)
                for i,row in enumerate(pool):
                    m=objectives(matrix[i],[w["batch"] for w in dev])
                    diagnostic.append({"onchip_mode":mode,"constraint_group":group,"family":family,
                        "geometry":row["geometry"],"design":canonical(row["design"]),"geomean_ms":row["score_ms"],
                        "ratio_vs_common_B1":m["geomean"],"cvar10_ratio":m["cvar10"],"minimax_batch_ratio":m["minimax"],
                        "p95_latency_ms":float(np.quantile(row["latencies_ms"],.95)),
                        "selected_by_primary_objective":key(decode(row["design"]))==key(design),
                        "generated_by":"+".join(row["generated_by"]),
                        "common_reference":"C0 development-selected B1 MILP+LPT vector within mode"})
                    for name,v in counts.items():
                        stability.append({"onchip_mode":mode,"constraint_group":group,"family":family,
                            "geometry":row["geometry"],"design":canonical(row["design"]),"objective":name,
                            "draws":200,"selected_count":v[i],"selected_share":v[i]/200,
                            "common_reference":"C0 development-selected B1 MILP+LPT vector within mode",
                            "candidate_scope":"equally-budgeted C0+C1 development witness union; not full-domain optimum probability"})
                progress.append({"onchip_mode":mode,"constraint_group":group,"family":family,
                    **r["union_counts"],"eligible_unique_points":len(pool),
                    "evaluated_points":r["union_counts"]["unique_attempted_points"],
                    "successful_points":len(pool),"repeated_simulator_calls":r["union_counts"]["actual_simulator_calls"],
                    "current_best_ms":r["selected"]["score_ms"],"current_lb_ms":lb,
                    "gap_pct":r["gap_pct"],"open_regions":len(r["open_regions"]),
                    "pruned_points":certs[mode,group,family]["pruned_lattice_points"],
                    "as_run_pruned_points":certs[mode,group,family]["pruned_lattice_points"],
                    "declared_points":certs[mode,group,family]["declared_lattice_points"],
                    "exact_repeat_identical":True,"scope":"actual witness union; original as-run frontiers retained",
                    "call_count_scope":"per mode/family shared generation; do not sum both constraint rows"})
    selection["selection_metadata"].update(
        robust_diagnostic_reference="same C0 development-selected B1 window vector across all families andconstraintsets within mode",
        proof_A_definition="joint5% gain against B1/B2, notfamily incumbent",
        proof_B_definition="algorithm-specific allocation+LPT physical replay objective; notglobaljoint temporal optimum",
        final_selection_policy="C0+C1 actual development witness union per mode/family; C1 restricted to same union's eligible points",
        combined_candidate_generation_budget_per_family=512,combined_node_generation_budget_per_family=4096,
        no_heldout_access_in_union_selection=True)
    write_json(out/"selected_designs.json",selection)
    write_csv(out/"proof_status.csv",proof)
    write_csv(out/"bootstrap_stability.csv",stability)
    write_csv(out/"development_candidates.csv",diagnostic)
    if not (out/"dse_progress_as_run.csv").exists():
        write_csv(out/"dse_progress_as_run.csv",read_csv(out/"dse_progress.csv"))
    write_csv(out/"dse_progress.csv",progress)
    write_csv(out/"union_generation_budget.csv",union_meta)
    write_csv(out/"dse_repeat_compare.csv",repeat_rows)
    write_json(out/"DSE_PROTOCOL_FINAL.json",metadata({
        "selection":selection["selection_metadata"],"jobs":20,"entire_search_repeats":2,
        "as_run_protocol":"DSE_PROTOCOL.json","certificate_directory":"final_certificates",
        "final_selection_scope":"full equal-work C0+C1 actual witness union; C1 checks the same union",
        "all_proof_B_closed":all(r["proof_B_closed"] for r in final.values()),
        "total_simulator_calls":sum(x["actual_simulator_calls"] for x in union_meta),
        "no_extra_simulator_calls_for_union":True}))
    write_json(out/"PROOF_REPORT_AUDIT.json",metadata({"numeric_replays_unchanged":True,
        "selected_hardware_may_change_with_dev_union":True,"all_C0_superset_scores_no_worse_than_C1":True,
        "joint_gate_A_rows":len(proof),"proof_A_closed_rows":sum(x["proof_A_closed"] for x in proof),
        "proof_B_closed_rows":sum(x["proof_B_closed"] for x in proof),
        "bootstrap_draws":200,"common_reference_verified":True,
        "raw_certificate_terminology":"proof_A_closed rawfield islocalincumbent5%; headlineproof_status usesjointB1/B2gate",
        "actual_source_simulator_calls":sum(x["actual_simulator_calls"] for x in union_meta)}))
    marker=out/"DSE_COMPLETE.json"
    temporary=marker.with_suffix(".json.tmp")
    write_json(temporary,{"all20certificates_have18development_windows":True,
        "all20fullsearchrepetitions_identical":True,"numeric_hardware_selected_before_heldout":True,
        "development_ids":expected_ids,"jointproofAandcommonrobustreference_reported":True,
        "private_and_shared_slots_separated":True,"selection_uses_equal_C0_C1_witness_union":True})
    temporary.replace(marker)


if __name__=="__main__":
    run()
