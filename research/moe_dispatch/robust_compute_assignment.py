#!/usr/bin/env python3
"""Three-GEMM resident-operand oracle for the real four-expert diagnostic.

Sums each expert's Gate/Up/Down projection service on its assigned core. Per-core
expert order is fixed; vector/copy/ingress service is zero. This is deliberately
not a substitute for the full analytical FFN or a globally optimal scheduler.
"""
import argparse, itertools, json, subprocess
from pathlib import Path
import run_experiments as run
from robust_study import representatives,csv_out,freeze

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--binary',type=Path,required=True);a=ap.parse_args();root=a.output
    rows=json.loads((root/'assignment_rows.json').read_text())
    first=next(r for r in rows if r['workload'].startswith('partial4_'))
    w=json.loads((Path(first['raw_directory'])/'workload.json').read_text())
    binary,provenance=freeze(root,a.binary,'compute_real')
    cache={};out=[]
    def projection(m,me,n,k):
        key=(m,me,n,k)
        if key not in cache:
            path=root/'compute_real'/('_'.join(map(str,key)));path.mkdir(parents=True,exist_ok=True)
            run.write_json(path/'input.json',dict(lanes=[m],Me=[me],owners=[0],N=n,K=k,group=4))
            reports=[]
            for i in (1,2):
                subprocess.run([str(binary),'--compute',str(path/'input.json'),str(path/f'report{i}.json')],check=True)
                reports.append(json.loads((path/f'report{i}.json').read_text()))
            assert reports[0]==reports[1]
            cache[key]=reports
        return cache[key]
    for d in representatives():
        for owners in itertools.product(range(len(d['lanes'])),repeat=len(w['experts'])):
            repeats=[]
            for repeat in (0,1):
                end=[0]*len(d['lanes']);useful=issued=issues=0
                for e,c in zip(w['experts'],owners):
                    for n,k in ((e['F'],e['H']),(e['F'],e['H']),(e['H'],e['F'])):
                        r=projection(d['lanes'][c],e['Me'],n,k)[repeat]
                        end[c]+=r['cycles'];useful+=r['useful_macs'];issued+=r['issued_macs'];issues+=sum(r['issues_per_core'])
                repeats.append(dict(cycles=max(end),core_finish=end,useful_macs=useful,issued_macs=issued,
                                    padding_macs=issued-useful,issues=issues))
            assert repeats[0]==repeats[1]
            out.append(dict(workload=w['id'],budget_group=d['budget_group'],architecture=d['architecture'],
                            lanes=d['lanes'],owners=owners,**repeats[0],repeats_equal=True,
                            scope='oracle: three resident GEMMs per expert; zero vector/copy/ingress; fixed per-core expert order',
                            latency_ms=max(end)/1e6))
    csv_out(root/'assignment_compute_real.csv',out)
    csv_out(root/'assignment_search.csv',[dict(r,scope='full analytical fixed-order whole-expert assignment') for r in rows]+out)
    run.write_json(root/'compute_real_verification.json',dict(primitive_projection_cases=len(cache),runs=2*len(cache),
                    composed_assignment_cases=len(out),complete_repeats_match=True,binary_sha256=provenance['binary_sha256']))
    print('real subset compute oracle',len(out),'assignments;',len(cache),'primitive projections repeated twice')

if __name__=='__main__':main()
