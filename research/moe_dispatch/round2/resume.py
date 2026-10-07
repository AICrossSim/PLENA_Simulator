"""Continue preserved B&B intervals without changing their frozen domain."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from .common import inputs,write_json,ROOT
from .sensitivity import parameters_from_dict
from .search import search_workloads,_workload_hash

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--certificate',type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=3600);ap.add_argument('--workload-json',type=Path)
    ap.add_argument('--output',type=Path)
    a=ap.parse_args();source=json.loads(a.certificate.read_text())
    result=source.get('verification_full_domain_certificate',source)
    resume=result.get('resume',result)
    if a.workload_json:
        ws=json.loads(a.workload_json.read_text());ws=ws if isinstance(ws,list) else [ws]
    elif 'workload' in source:ws=[source['workload']]
    elif 'resume_workloads' in result:ws=result['resume_workloads']
    else:ws=inputs()['development']
    if _workload_hash(ws)!=resume['workload_sha256']:
        raise ValueError('Workload hash differs. Supply the exact saved workload with --workload-json; no new routing sample may replace a certificate.')
    p=parameters_from_dict(resume['parameters'])
    r=search_workloads(ws,p,delta=resume['delta'],time_limit_s=a.seconds,
        target_families=tuple(resume['families']),resume_state=result)
    r['resume_workloads']=ws
    out=a.output or a.certificate.with_name(a.certificate.stem+'_continued.json')
    write_json(out,r)
    print(json.dumps({'output':str(out),'proof_complete':r['proof_complete'],
        'gap_pct':r['gap_pct'],'open_lb_ms':r['open_lb_ms']},indent=2))
if __name__=='__main__':main()
