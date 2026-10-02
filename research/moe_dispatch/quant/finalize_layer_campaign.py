#!/usr/bin/env python3
"""Supplement a verified full-layer merge with Q2 and MXINT4-B receipts."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import evaluate as q


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--calibration',type=Path,required=True)
    ap.add_argument('--evaluation',type=Path,required=True);ap.add_argument('--supplement-layer13',type=Path,required=True);ap.add_argument('--wait',action='store_true')
    args=ap.parse_args()
    if args.wait:
        while True:
            state=json.loads((args.output/'job_status.json').read_text())
            diag=json.loads((args.supplement_layer13/'job_status.json').read_text())
            if state['status']=='completed' and diag['status']=='completed':break
            if diag['status']=='failed':raise RuntimeError('MXINT4-B diagnostic failed')
            if (args.output/'shard_orchestrator.json').exists():
                orchestrator=json.loads((args.output/'shard_orchestrator.json').read_text())
                if orchestrator.get('status')=='failed':raise RuntimeError('layer merge failed')
            time.sleep(15)
    coverage=json.loads((args.output/'q1_coverage.json').read_text())
    if not coverage['complete'] or coverage['completed_candidates']!=2178:raise ValueError('required full Q1 matrix is not complete')
    _,cal,_,cn=q.load_capture(args.calibration);_,ev,_,en=q.load_capture(args.evaluation)
    cold=[]
    for layer in (2,13,26):
        for e in range(64):
            n=int(np.any(cal[layer]['routes']==e,axis=1).sum());m=int(np.any(ev[layer]['routes']==e,axis=1).sum())
            cold.append({'layer':layer,'expert':e,'calibration_tokens':n,'evaluation_tokens':m,'cold':n<32})
    q.write_csv(args.output/'q2_cold_counts.csv',cold)
    # Frequency quartiles use calibration counts only. Expert errors are
    # inherited from the actual full Q1 default/RTN FFNs, not a surrogate.
    metrics=list(csv.DictReader((args.output/'q1_metrics.csv').open()));frequency_rows=[]
    for layer in (2,13,26):
        counts={r['expert']:r['calibration_tokens'] for r in cold if r['layer']==layer};ordered=sorted(counts,key=lambda e:(counts[e],e))
        for L in (8,16):
            for method,requested in [('rtn',0),('qera_approx',48 if L==8 else 96)]:
                selected={int(r['expert']):r for r in metrics if r['scope']=='expert' and int(r['layer'])==layer and int(r['rank_lanes'])==L and r['bits']=='4' and r['method']==method and int(r['rank'])==requested and r['factor_a']=='mxint4' and r['factor_b']=='bf16' and int(r['expert'])>=0}
                if set(selected)!=set(range(64)):raise ValueError('frequency audit lacks actual default expert metrics')
                for quartile in range(4):
                    ids=ordered[quartile*16:(quartile+1)*16];errors=[float(selected[e]['relative_error']) for e in ids]
                    frequency_rows.append({'layer':layer,'rank_lanes':L,'method':method,'requested_rank':requested,'calibration_frequency_quartile':quartile+1,'experts':len(ids),
                        'calibration_tokens_min':min(counts[e] for e in ids),'calibration_tokens_max':max(counts[e] for e in ids),
                        'relative_error_mean':float(np.mean(errors)),'relative_error_median':float(np.median(errors)),'relative_error_p90':float(np.quantile(errors,.9)),
                        'relative_error_min':min(errors),'relative_error_max':max(errors),'scope':'actual Q1 default expert errors grouped by development calibration frequency; naturally-cold own/pooled comparison not applicable'})
    q.write_csv(args.output/'q2_frequency_group_errors.csv',frequency_rows)
    count=sum(int(r['cold']) for r in cold)
    if count:raise ValueError('unexpected naturally cold experts: must merge real Q2 measurements rather than mark N/A')
    (args.output/'q2_status.json').write_text(json.dumps({'cold_experts':0,'measured_cold_comparisons':0,
        'no_samples_expert_own':0,'status':'no_calibration_expert_below_32_tokens_in_this_capture',
        'scope':'all192 routed expert/layer populations audited from real calibration; no artificial starvation is treated as naturally cold'},indent=2)+'\n')
    supplement=[]
    for layer in (2,13,26):
        folder=args.supplement_layer13 if layer==13 else args.output/'layer_shards'/f'layer{layer}'
        supplement+=list(csv.DictReader((folder/'supplemental_mxint4_b_default.csv').open()))
    layerrows=[r for r in supplement if r['scope']=='layer']
    wanted={(l,L,m) for l in (2,13,26) for L in (8,16) for m in ('rtn','qera_approx')}
    keys=[(int(r['layer']),int(r['rank_lanes']),r['method']) for r in layerrows]
    if set(keys)!=wanted or len(keys)!=len(wanted):raise ValueError('supplemental MXINT4-B default rows incomplete/duplicate')
    q.write_csv(args.output/'supplemental_mxint4_b_default.csv',supplement)
    target={}
    for row in layerrows:
        if row['method']=='rtn':target[(row['layer'],row['rank_lanes'])]=.5*float(row['relative_error'])
    checks=[dict(row,error_target=target[(row['layer'],row['rank_lanes'])],cosine_target=.999,
        passes=float(row['relative_error'])<=target[(row['layer'],row['rank_lanes'])] and float(row['cosine'])>=.999)
        for row in layerrows if row['method']=='qera_approx']
    (args.output/'supplemental_mxint4_b_quality.json').write_text(json.dumps({'schema':'plena_v3_mxint4_b_default_quality_v1',
        'q0_complete':True,'calibration_tokens':cn,'validation_tokens':en,'all_checks_complete':True,'checks':checks,
        'all_targets_pass':all(r['passes'] for r in checks),'scope':'actual default rank, MXINT4A/MXINT4B, RTN vs lanes, full-Z numerical order; not additional Q1 candidates or model perplexity',
        'calibration_manifest_sha256':q.sha(args.calibration),'validation_manifest_sha256':q.sha(args.evaluation)},indent=2)+'\n')
    # Proxy rows are explicitly calibration-only surrogates, distinct from Q3
    # actual layer-FFN measurements. Reconstructing them uses no future gates.
    proxy=[]
    for L in (8,16):
        for layer in (2,13,26):
            for bit in (4,3):
                for method in ('qera_approx','qera_exact','lqer','l2qer'):
                    table=json.loads((args.output/f'L{L}'/f'rank_table_l{layer}_w{bit}_{method}.json').read_text())
                    weights=np.zeros(64)
                    for rr,gg in zip(cal[layer]['routes'],cal[layer]['gates']):
                        for e,g in zip(rr,gg):weights[e]+=float(g)**2
                    te=np.asarray(table['tail_energy']);rb=np.asarray(table['factor_bytes']);caps=np.asarray(table['cap_index'])
                    idx=q.rn.allocate_ranks(weights,te,rb,table['lambda0'],caps)
                    proxy.append({'layer':layer,'rank_lanes':L,'bits':bit,'method':method,'budget_bytes':table['uniform_factor_budget_bytes'],
                        'allocated_bytes':float(rb[np.arange(64),idx].sum()),'uniform_objective':float((weights*te[:,4]).sum()),
                        'gate_weighted_objective':float((weights*te[np.arange(64),idx]).sum()),'lambda0':table['lambda0'],
                        'scope':'surrogate_calibration_tail_energy_not_layer_error'})
    q.write_csv(args.output/'q3_rank_proxy.csv',proxy)
    (args.output/'postmerge_receipt.json').write_text(json.dumps({'schema':'plena_v3_postmerge_receipt_v1','q2_naturally_cold':count,
        'mxint4_b_default_checks':len(checks),'q3_calibration_proxy_rows':len(proxy),'all_complete':True},indent=2)+'\n')
    print(json.dumps({'completed':True,'naturally_cold':count,'mxint4_b_pass':all(r['passes'] for r in checks)}))


if __name__=='__main__':main()
