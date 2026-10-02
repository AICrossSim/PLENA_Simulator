#!/usr/bin/env python3
"""Detect complete first-layer candidate rejection without stopping other layers."""
import argparse,csv,io,json,time
from pathlib import Path
import evaluate as q


def proof(output,layer=13):
    raw=(output/'q1_metrics.csv').read_text();rows=list(csv.DictReader(io.StringIO(raw)))
    relevant=[r for r in rows if r.get('scope')=='layer' and int(r['layer'])==layer]
    coverage=q.q1_coverage(relevant,[layer],[4,3],[0,8,16,24,32,48,64],
        ['rtn','qera_approx','qera_exact','lqer','l2qer'],[8,16],['mxint4','mxint8s','bf16'],['bf16','mxint8'])
    if not coverage['complete']:return None
    rtn={int(r['rank_lanes']):float(r['relative_error']) for r in relevant if r['bits']=='4' and r['method']=='rtn'}
    candidates=[r for r in relevant if r['method']!='rtn' and r['bits']!='8']
    passing=[r for r in candidates if float(r['relative_error'])<=.5*rtn[int(r['rank_lanes'])] and float(r['cosine'])>=.999]
    return {'schema':'plena_v3_complete_layer_accuracy_proof_v1','layer':layer,'scope':'real development numerical validation; fixed candidate search only',
        'coverage':coverage,'tested_accuracy_candidates':len(candidates),'passing_candidates':passing,
        'all_three_layer_candidates_eliminated':not passing,'targets':{'error_vs_w4_rtn_max':.5,'cosine_min':.999},
        'best_error_candidate':min(candidates,key=lambda r:float(r['relative_error'])),
        'best_cosine_candidate':max(candidates,key=lambda r:float(r['cosine'])),
        'evidence':{'measured':'all required candidate outputs at this layer','deduced':'if all fail here, no candidate can pass the same all-three-layer conjunction',
                    'not_claimed':'future formats/methods or other models cannot work'},
        'layer_candidate_rows':relevant,'full_other_layer_matrix_continues':True}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--layer',type=int,default=13)
    ap.add_argument('--wait',action='store_true');ap.add_argument('--timeout-hours',type=float,default=8);args=ap.parse_args();start=time.monotonic()
    while True:
        try:value=proof(args.output,args.layer)
        except (KeyError,ValueError,FileNotFoundError):value=None  # Worker may be checkpointing the CSV.
        if value is not None:
            p=args.output/f'q1_layer{args.layer}_accuracy_proof.json';p.write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps({'path':str(p),'candidates':value['coverage']['completed_candidates'],
                              'passing':len(value['passing_candidates']),'all_eliminated':value['all_three_layer_candidates_eliminated']}));break
        if not args.wait:print('layer candidate coverage is incomplete');break
        if time.monotonic()-start>args.timeout_hours*3600:raise TimeoutError('layer matrix is still incomplete')
        time.sleep(15)


if __name__=='__main__':main()
