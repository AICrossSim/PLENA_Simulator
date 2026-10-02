#!/usr/bin/env python3
"""Verify and archive all actual BF16-LUT Q3 rows and vector comparisons."""
import argparse,csv,json,time
import numpy as np
from pathlib import Path
import evaluate as q
from run_layer_shards import verify_q3


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--layer13',type=Path,required=True);ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    if args.wait:
        while True:
            a=json.loads((args.output/'job_status.json').read_text());b=json.loads((args.layer13/'job_status.json').read_text())
            if a['status']=='completed' and b['status']=='completed':break
            if b['status']=='failed':raise RuntimeError('layer13 hardware Q3 supplement failed')
            time.sleep(15)
    hardware=[];vectors=[];software_control=[];sources=[]
    for layer in (2,13,26):
        folder=args.layer13 if layer==13 else args.output/'layer_shards'/f'layer{layer}'
        software_control += list(csv.DictReader((folder/'q3_actual_ffn.csv').open()))
        for L in (8,16):
            for bit in (4,3):
                for method in ('qera_approx','qera_exact','lqer','l2qer'):
                    for kind,target in [('hardware_bf16',hardware),('rank_vector_comparison',vectors)]:
                        p=folder/f'L{L}'/f'q3_{kind}_l{layer}_w{bit}_{method}_L{L}.csv'
                        target+=list(csv.DictReader(p.open()));sources.append({'path':str(p.resolve()),'sha256':q.sha(p)})
    tables=[]
    for layer in (2,13,26):
        folder=args.layer13 if layer==13 else args.output/'layer_shards'/f'layer{layer}'
        for L in (8,16):
            for bit in (4,3):
                for method in ('qera_approx','qera_exact','lqer','l2qer'):
                    for uniform in (16,32):
                        p=folder/f'L{L}'/f'rank_table_hw_bf16_l{layer}_w{bit}_{method}_uniform{uniform}.json'
                        t=json.loads(p.read_text())
                        if not t['origin'].startswith('physical BF16-energy LUT'):raise ValueError('hardware table uses obsolete float calibration')
                        tables.append({'path':str(p.resolve()),'sha256':q.sha(p),'lambda0':t['lambda0'],'lambda_fit':t['lambda_fit'],
                            'software_float_lambda0':t['software_float_lambda0'],'float_lambda0_on_hardware_diagnostic':t['float_lambda0_on_hardware_diagnostic']})
    coverage=verify_q3(hardware,[2,13,26]);verify_q3(software_control,[2,13,26])
    wanted={(int(r['layer']),int(r['rank_lanes']),int(r['bits']),r['method'],int(r['uniform_rank']),int(r['window_start']),r['strategy']) for r in hardware}
    keys=[(int(r['layer']),int(r['rank_lanes']),int(r['bits']),r['method'],int(r['uniform_rank']),int(r['window_start']),r['strategy']) for r in vectors]
    if set(keys)!=wanted or len(keys)!=len(wanted):raise ValueError('BF16/float vector proof has missing/duplicate rows')
    q.write_csv(args.output/'q3_hardware_bf16_actual_ffn.csv',hardware);q.write_csv(args.output/'q3_hardware_vector_comparison.csv',vectors)
    q.write_csv(args.output/'q3_software_for_hardware_control.csv',software_control)
    # An unequal-byte error improvement is not an equal-budget result. Report
    # causal before/after-warmup rows, actual byte deltas and exact-byte pairs.
    paired=[];reference={}
    for r in hardware:
        key=(r['layer'],r['rank_lanes'],r['bits'],r['method'],r['uniform_rank'],r['window_start'])
        if r['strategy']=='uniform':reference[key]=r
    for layer in (2,13,26):
        for L in (8,16):
            for bit in (4,3):
                for method in ('qera_approx','qera_exact','lqer','l2qer'):
                    for uniform in (16,32):
                        selected=[r for r in hardware if int(r['layer'])==layer and int(r['rank_lanes'])==L and int(r['bits'])==bit and r['method']==method and int(r['uniform_rank'])==uniform and r['strategy']=='gate_weighted_causal']
                        for population,rows in [('all',selected),('after8warmup',[r for r in selected if r['warmup']=='False'])]:
                            baseline=[reference[(r['layer'],r['rank_lanes'],r['bits'],r['method'],r['uniform_rank'],r['window_start'])] for r in rows]
                            same=[(r,b) for r,b in zip(rows,baseline) if float(r['factor_bytes'])==float(b['factor_bytes'])]
                            err=np.mean([float(r['relative_error']) for r in rows]);ref=np.mean([float(r['relative_error']) for r in baseline])
                            matched=float(np.mean([float(r['relative_error']) for r,b in same])/np.mean([float(b['relative_error']) for r,b in same])) if same else ''
                            paired.append({'layer':layer,'rank_lanes':L,'bits':bit,'method':method,'uniform_rank':uniform,'population':population,'windows':len(rows),
                                'mean_actual_error_ratio':float(err/ref),'mean_actual_factor_bytes':float(np.mean([float(r['factor_bytes']) for r in rows])),
                                'mean_uniform_factor_bytes':float(np.mean([float(b['factor_bytes']) for b in baseline])),
                                'mean_byte_delta':float(np.mean([float(r['factor_bytes'])-float(b['factor_bytes']) for r,b in zip(rows,baseline)])),
                                'exact_byte_matched_windows':len(same),'exact_byte_matched_error_ratio':matched,
                                'scope':'different actual bytes are diagnostic; only exact-byte pairs are an equal-byte subset; correlated captured windows'})
    q.write_csv(args.output/'q3_hardware_budget_pair_summary.csv',paired)
    summaries=[]
    for strategy in ('uniform','frequency_static','gate_weighted_budget_oracle','gate_weighted_causal'):
        for budget in (16,32):
            vv=[r for r in vectors if r['strategy']==strategy and int(r['uniform_rank'])==budget]
            summaries.append({'strategy':strategy,'uniform_rank':budget,'windows':len(vv),
                'changed_windows':sum(r['same']!='True' for r in vv),'changed_expert_decisions':sum(int(r['changed_active_experts']) for r in vv),
                'all_vectors_identical':all(r['same']=='True' for r in vv)})
    receipt={'schema':'plena_v3_hardware_q3_complete_receipt_v1','complete':True,'coverage':coverage,
        'vector_comparisons':len(vectors),'summary':summaries,'sources':sources,'physical_calibration_tables':tables,
        'table':'BF16 RNE perprojection energies; exact uint32 byte costs; commonexpert rank; Shared stays full',
        'lambda':'physical BF16 projection energy LUT refit on actual development-window routed-only factor bytes; Shared fixed separate; software float constant retained as diagnostic',
        'quality_evidence':'actual FFN output reconstructed from resident rank-prefix outputs; software and physical-LUT policy errors kept separate',
        'no_silent_quality_transfer':True}
    (args.output/'q3_hardware_complete_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'completed':True,'actual_ffn_rows':len(hardware),'vector_rows':len(vectors)}))


if __name__=='__main__':main()
