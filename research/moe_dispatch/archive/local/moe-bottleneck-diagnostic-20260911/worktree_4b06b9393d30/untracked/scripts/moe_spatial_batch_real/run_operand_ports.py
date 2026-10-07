#!/usr/bin/env python3
"""Meaningful operand-port probes when 4x byte bandwidth leaves ceil(bytes/rate)=1.
Every oracle and packed-port run uses ACTUAL BF16 values and must match charged
FP32 outputs, not just nominal MAC/byte counters. No throughput claim from a
timing-inert bandwidth sweep is accepted.
"""
import csv,gzip,json,subprocess,tempfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import run_routes as r

DEST=r.OUT/'operand_ports';BIN=r.OUT/'repro/moe_spatial_fabric_real'
VARIANTS={
    'oracle_zero_activation':dict(zero_activation_time=True),
    'oracle_zero_accumulator':dict(zero_accumulator_time=True),
    'oracle_both_operand_ports':dict(zero_activation_time=True,zero_accumulator_time=True),
    'optimistic_packed_operand_ports':dict(packed_operand_ports=True),
}
def execute(source,variant,cfg):
    name=source.name+'__'+variant
    folder=DEST/name;folder.mkdir(exist_ok=True)
    req=json.loads((source/'routed_gate.request.json').read_text());req['compute']['name']=name
    req['fabric'].update(cfg);r.save(folder/'request.json',req)
    charged=json.loads(gzip.decompress((source/'routed_gate.report.json.gz').read_bytes()))
    hs=[]
    for repeat in range(2):
        with tempfile.TemporaryDirectory(prefix='plena-operand-ports-',dir='/tmp') as tmp:
            p=Path(tmp)/'out.json';q=subprocess.run([str(BIN),'--request',str(folder/'request.json'),
                '--operands',str(source/'routed_gate.operands.json'),'--output',str(p)],capture_output=True,timeout=900)
            assert q.returncode==0,q.stderr
            raw=p.read_bytes();rep=json.loads(raw)
        hs.append(r.digest(raw))
        if repeat==0:(folder/'report.json.gz').write_bytes(gzip.compress(raw,mtime=0))
        else:assert hs[0]==hs[1]
        assert rep['numerical_bit_exact'] is True
        assert rep['output_fp32_bits']==charged['output_fp32_bits'] and rep['output_bf16_bits']==charged['output_bf16_bits']
    timing=json.loads(json.dumps(req));timing['compute']['verify_values']=False
    rr={**rep,'numerical_bit_exact':None,'output_fp32_bits':None,'output_bf16_bits':None}
    r.helper.audit(timing,rr)
    b,shape,mode=source.name.split('__');s=rep['stats'];bs=charged['stats']
    row=dict(name=name,batch=int(b.removeprefix('real_b')),shape=shape.replace('_','+'),mode=mode,variant=variant,
        charged_cycles=charged['total_cycles'],cycles=rep['total_cycles'],speedup=charged['total_cycles']/rep['total_cycles'],
        source_byte_delta=s['source_weight_bytes']-bs['source_weight_bytes'],activation_byte_delta=s['activation_bytes']-bs['activation_bytes'],
        rmw_byte_delta=s['accumulator_rmw_bytes']-bs['accumulator_rmw_bytes'],
        invocation_delta=rep['invocation_audit_count']-charged['invocation_audit_count'],
        activation_service_cycles=s['activation_service_cycles'],accumulator_service_cycles=s['accumulator_service_cycles'])
    r.save(folder/'validation.json',dict(passed=True,row=row,repeat_sha256=hs,charged_output_bit_exact=True,
        operands_source=str(source/'routed_gate.operands.json'),binary_sha256=r.digest(BIN.read_bytes())))
    print(name,'PASS',flush=True);return row
def main():
    DEST.mkdir(exist_ok=True);sources=sorted(p for p in (r.OUT/'real_campaign').glob('real_b*') if p.is_dir());assert len(sources)==40
    plan=[(p,k,v) for p in sources for k,v in VARIANTS.items()]
    r.save(DEST/'plan.json',dict(points=len(plan),repeats=2,scope='actual-input routed gate only; nonphysical timing or optimistic sub-beat packing',
        reason='at least one cycle per positive request makes 4x byte-rate sweep timing-inert for requests <=6144B X / <=192B RMW'))
    rows=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(execute,*p) for p in plan]
        for f in as_completed(fs):
            rows.append(f.result());r.save(DEST/'progress.json',dict(done=len(rows),planned=len(plan)))
    with (DEST/'all_points.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    r.save(DEST/'validation.json',dict(passed=True,points=len(rows),runs=2*len(rows),all_actual_outputs_match_charged=True,all_repeats_identical=True))
if __name__=='__main__':main()
