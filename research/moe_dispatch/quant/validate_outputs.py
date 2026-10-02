#!/usr/bin/env python3
"""Verify the full required numerical matrix, independently of the worker process."""
import argparse,csv,json,time
from pathlib import Path
import evaluate as q


def validate(output, lane_modes):
    path=output/'q1_metrics.csv'
    rows=list(csv.DictReader(path.open())) if path.exists() else []
    result=q.q1_coverage(rows,[2,13,26],[4,3],[0,8,16,24,32,48,64],
        ['rtn','qera_approx','qera_exact','lqer','l2qer'],lane_modes,
        ['mxint4','mxint8s','bf16'],['bf16','mxint8'])
    result['metric_rows']=len(rows)
    result['layers_with_actual_expert_metrics']={str(l):sorted({int(r['expert']) for r in rows if r['scope']=='expert' and int(r['layer'])==l}) for l in (2,13,26)}
    result['worker_receipt']=json.loads((output/'job_status.json').read_text()) if (output/'job_status.json').exists() else None
    result['accuracy_freeze_receipt']=json.loads((output/'freeze_candidates.json').read_text()) if (output/'freeze_candidates.json').exists() else None
    (output/'q1_coverage.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rank-lanes',default='8,16');ap.add_argument('--wait',action='store_true')
    ap.add_argument('--timeout-hours',type=float,default=8);args=ap.parse_args();start=time.monotonic()
    if args.wait:
        while True:
            state=json.loads((args.output/'job_status.json').read_text())
            if state['status'] in ('completed','failed'):break
            if time.monotonic()-start>args.timeout_hours*3600:raise TimeoutError('numerical matrix is still unfinished')
            time.sleep(15)
    result=validate(args.output,list(map(int,args.rank_lanes.split(','))))
    print(json.dumps({k:result[k] for k in ('expected_candidates','completed_candidates','complete','metric_rows')}))
    if args.wait and not result['complete']:raise SystemExit('full Q1 matrix did not complete')


if __name__=='__main__':main()
