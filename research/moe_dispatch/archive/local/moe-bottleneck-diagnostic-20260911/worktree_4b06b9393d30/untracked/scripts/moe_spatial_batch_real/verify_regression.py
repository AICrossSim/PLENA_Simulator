#!/usr/bin/env python3
"""Frozen-output regression and external-operand CLI rejection contracts."""
import json,hashlib,subprocess,tempfile
from pathlib import Path
import run_routes as r

NEW=r.OUT/'repro/moe_spatial_fabric_real'
def run(binary,request,output,operands=None):
    cmd=[str(binary),'--request',str(request),'--output',str(output)]
    if operands:cmd+=['--operands',str(operands)]
    return subprocess.run(cmd,capture_output=True,timeout=120)
def main():
    checks=[]
    with tempfile.TemporaryDirectory(prefix='plena-regression-',dir='/tmp') as tmp:
        d=Path(tmp)
        for shape in [[6],[3,3],[4,2],[2,2,2],[1]*6]:
            for mode in r.MODES:
                for control in ['cohort','tile_cohort']:
                    req=r.helper.req('unchanged',shape,r.helper.helper.fixture([4,2,1],7,1031),mode,
                        dict(control=control),numeric=True,trace=True)
                    p=d/'request.json';r.save(p,req)
                    for b,o in [(r.FAB,d/'old.json'),(NEW,d/'new.json')]:
                        proc=run(b,p,o);assert proc.returncode==0,proc.stderr
                    assert (d/'old.json').read_bytes()==(d/'new.json').read_bytes()
                    checks.append(dict(shape=shape,mode=mode,control=control,full_output_identical=True))
        req=r.helper.req('loader',[6],[dict(expert=3,m=1,n=4,k=512,seed=13)],'pinned_expert',numeric=True)
        r.save(d/'request.json',req)
        raw=b'\x80\x3f' # 1.0 BF16
        (d/'x.bf16').write_bytes(raw*512);(d/'w.bf16').write_bytes(raw*2048)
        files={k:dict(path=k+'.bf16',sha256=r.digest((d/(k+'.bf16')).read_bytes())) for k in ['x','w']}
        valid=dict(schema='plena_bf16_x_mk_w_nk_v1',jobs=[dict(expert=3,**files)])
        r.save(d/'operands.json',valid)
        p=run(NEW,d/'request.json',d/'valid.json',d/'operands.json');assert p.returncode==0,p.stderr
        out=json.loads((d/'valid.json').read_text());assert out['numerical_bit_exact']
        failures=[]
        for case in ['hash','expert','schema','missing','short','nonfinite','verification_off','overwrite_manifest','overwrite_weight']:
            m=json.loads(json.dumps(valid));rq=json.loads(json.dumps(req));output=d/'rejected.json'
            for k,count in [('x',512),('w',2048)]:(d/(k+'.bf16')).write_bytes(raw*count)
            if case=='hash':m['jobs'][0]['x']['sha256']='0'*64
            elif case=='expert':m['jobs'][0]['expert']=4
            elif case=='schema':m['schema']='unknown'
            elif case=='missing':m['jobs'][0]['x']['path']='absent.bf16'
            elif case=='short':(d/'x.bf16').write_bytes(raw)
            elif case=='nonfinite':
                (d/'x.bf16').write_bytes(b'\x80\x7f'*512)
                m['jobs'][0]['x']['sha256']=r.digest((d/'x.bf16').read_bytes())
            elif case=='verification_off':rq['compute']['verify_values']=False
            elif case=='overwrite_manifest':output=d/'operands.json'
            elif case=='overwrite_weight':output=d/'w.bf16'
            r.save(d/'operands.json',m);r.save(d/'request.json',rq)
            before=output.read_bytes() if output.exists() else None
            proc=run(NEW,d/'request.json',output,d/'operands.json')
            assert proc.returncode!=0,case
            if before is None:assert not output.exists()
            else:assert output.read_bytes()==before
            failures.append(dict(case=case,rejected=True,error=proc.stderr.decode().strip()))
    r.save(r.OUT/'repro/regression.json',dict(passed=True,frozen_full_output_cases=checks,loader_rejections=failures,
        new_binary_sha256=r.digest(NEW.read_bytes()),old_binary_sha256=r.digest(r.FAB.read_bytes())))
    print('REGRESSION PASS',len(checks),len(failures))
if __name__=='__main__':main()
