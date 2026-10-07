#!/usr/bin/env python3
"""Actual, connected BF16 MoE values: Rust gate/up/down plus host vector ops.

Time is reported ONLY for GEMM phases. SiLU, routing/scatter/weighted combine
execute numerically in PyTorch but receive NO claimed accelerator timing.
"""
import argparse, csv, gzip, hashlib, importlib.util, json, subprocess, tempfile, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import torch

ROOT=Path('/scratch/shared/mcl123/plena');OUT=ROOT/'outputs/moe_spatial_batch_real_20260921'
WORK=ROOT/'review_20260921/simulator-moe-batch-real';DEST=OUT/'real_campaign'
BINARY=OUT/'repro/moe_spatial_fabric_real'
spec=importlib.util.spec_from_file_location('fabric_audit',WORK/'scripts/moe_spatial_fabric/run_study.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
meta=json.loads((OUT/'real_inputs/manifest.json').read_text())
capture=torch.load(OUT/'real_inputs/capture.pt',map_location='cpu',weights_only=True)
def digest(b):return hashlib.sha256(b).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def raw_tensor(p,v):
    assert v.dtype==torch.bfloat16
    b=v.contiguous().view(torch.uint16).numpy().tobytes();p.write_bytes(b)
    return dict(path=str(p),sha256=digest(b))
weights={}
for tag,f in meta['exported_weights'].items():
    b=Path(f['path']).read_bytes();assert digest(b)==f['sha256']
    phase=tag.split('_')[-1];shared=tag.startswith('shared_');inter=2816 if shared else 1408
    n,k=(2048,inter) if phase=='down' else (inter,2048)
    weights[tag]=torch.from_numpy(np.frombuffer(b,dtype='<u2').copy()).view(torch.bfloat16).reshape(n,k)

def metric(actual,expected):
    a=actual.double();e=expected.double();error=a-e
    return dict(max_abs=float(error.abs().max()),relative_l2=float(error.norm()/e.norm().clamp_min(1e-30)),
        mse=float((error*error).mean()),bf16_equal_fraction=float((actual==expected).float().mean()))

def execute(batch,shape,mode):
    name=f"real_b{batch}__{'_'.join(map(str,shape))}__{mode}"
    dst=DEST/name;dst.mkdir(parents=True,exist_ok=True)
    receipt=dst/'validation.json'
    if receipt.exists():
        result=json.loads(receipt.read_text());assert result['passed']
        return result
    start=time.monotonic();x=capture['x'][:batch];routes=capture['routes'][:batch];rw=capture['route_weights'][:batch]
    experts=sorted(set(routes.reshape(-1).tolist()))
    positions={e:(routes==e).nonzero() for e in experts}
    routed_x={e:x[positions[e][:,0]].contiguous() for e in experts}
    phase_rows=[];phase_checks=[];all_outputs={}
    def phase(kind,inputs,shared=False):
        label=('shared_' if shared else 'routed_')+kind
        jobs=[];ops=[];ordered=list(inputs);inter=2816 if shared else 1408
        n,k=(2048,inter) if kind=='down' else (inter,2048)
        for e in ordered:
            inp=inputs[e];expert=10000 if e=='shared' else e
            jobs.append(dict(expert=expert,m=inp.shape[0],n=n,k=k,seed=13))
            ops.append(dict(expert=expert,x=raw_tensor(dst/f'{label}_{e}_x.bf16',inp),w=meta['exported_weights'][f'{e}_{kind}']))
        request=helper.req(f'{name}__{label}',shape,jobs,mode,dict(control='tile_cohort'),numeric=True)
        path=dst/f'{label}.request.json';opfile=dst/f'{label}.operands.json'
        save(path,request);save(opfile,dict(schema='plena_bf16_x_mk_w_nk_v1',jobs=ops))
        hs=[]
        for repeat in range(2):
            with tempfile.TemporaryDirectory(prefix='plena-real-',dir='/tmp') as tmp:
                output=Path(tmp)/'result.json'
                p=subprocess.run([str(BINARY),'--request',str(path),'--operands',str(opfile),'--output',str(output)],
                    capture_output=True,timeout=1200)
                if p.returncode:raise RuntimeError(f'{name}/{label}: {p.stderr.decode()[-2000:]}')
                data=output.read_bytes();rep=json.loads(data)
            hs.append(digest(data))
            if repeat==0:(dst/f'{label}.report.json.gz').write_bytes(gzip.compress(data,mtime=0))
            else:assert hs[0]==hs[1]
        # Original timing/capacity assertions without its synthetic-value reference.
        tr=json.loads(json.dumps(request));tr['compute']['verify_values']=False
        auditrep={**rep,'numerical_bit_exact':None,'output_fp32_bits':None,'output_bf16_bits':None}
        helper.audit(tr,auditrep)
        assert rep['numerical_bit_exact'] is True
        outputs={};checks=[]
        for e,bits,rounded in zip(ordered,rep['output_fp32_bits'],rep['output_bf16_bits']):
            a=torch.from_numpy(np.array(bits,dtype=np.uint32).view(np.float32)).reshape(inputs[e].shape[0],n)
            w=weights[f'{e}_{kind}'];xx=inputs[e]
            reference=xx.double()@w.double().T
            magnitude=xx.double().abs()@w.double().abs().T
            scaled=float(((a.double()-reference).abs()/magnitude.clamp_min(1e-30)).max())
            assert scaled<=5e-6,(name,label,e,scaled)
            bf=a.to(torch.bfloat16)
            assert bf.view(torch.uint16).reshape(-1).tolist()==rounded
            outputs[e]=bf;checks.append(dict(expert=e,fp64_scaled_error=scaled,**metric(a,reference)))
        all_outputs[label]=rep['output_fp32_bits']
        phase_rows.append(dict(name=name,batch=batch,shape='+'.join(map(str,shape)),mode=mode,phase=label,
            cycles=rep['total_cycles'],macs=rep['useful_macs'],**rep['stats']))
        phase_checks.append(dict(phase=label,repeat_sha256=hs,request_sha256=digest(path.read_bytes()),
            operands_sha256=digest(opfile.read_bytes()),canonical_tree_bit_exact=True,fp64_checks=checks,
            invocation_sha256=rep['invocation_sha256'],service_sha256=rep['service_sha256']))
        print(f'{name} {label} PASS {rep["total_cycles"]} cycles',flush=True)
        return outputs
    gate=phase('gate',routed_x);up=phase('up',routed_x)
    # PyTorch BF16 SiLU then BF16 multiply, matching the official model MLP.
    z={e:torch.nn.functional.silu(gate[e])*up[e] for e in experts}
    down=phase('down',z)
    sg=phase('gate',{'shared':x},True);su=phase('up',{'shared':x},True)
    sd=phase('down',{'shared':torch.nn.functional.silu(sg['shared'])*su['shared']},True)['shared']
    # Restore top-k slot order before FP32 weighted combination and BF16 rounding.
    routed=torch.empty((batch,6,2048),dtype=torch.bfloat16)
    for e in experts:routed[positions[e][:,0],positions[e][:,1]]=down[e]
    combined=(routed.float()*rw[:,:,None]).sum(1).to(torch.bfloat16)+sd
    # Independent PyTorch model equation using original weights, no simulator intermediates.
    ref_routed=torch.empty_like(routed)
    for e in experts:
        xx=routed_x[e];g=torch.nn.functional.linear(xx,weights[f'{e}_gate']);u=torch.nn.functional.linear(xx,weights[f'{e}_up'])
        yy=torch.nn.functional.linear(torch.nn.functional.silu(g)*u,weights[f'{e}_down'])
        ref_routed[positions[e][:,0],positions[e][:,1]]=yy
    sgold=torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,weights['shared_gate']))*
        torch.nn.functional.linear(x,weights['shared_up']),weights['shared_down'])
    golden=(ref_routed.float()*rw[:,:,None]).sum(1).to(torch.bfloat16)+sgold
    error=metric(combined,golden);assert error['relative_l2']<=0.01,error
    outputfile=raw_tensor(dst/'moe_output.bf16',combined)
    signature=digest(json.dumps(all_outputs,separators=(',',':')).encode())
    result=dict(name=name,passed=True,batch=batch,shape='+'.join(map(str,shape)),mode=mode,
        matrix_phases=phase_rows,checks=phase_checks,all_projection_outputs_sha256=signature,
        output=outputfile,versus_pytorch=error,expert_counts={e:len(positions[e]) for e in experts},
        total_gemm_phase_cycles=sum(p['cycles'] for p in phase_rows),host_wall_seconds=time.monotonic()-start,
        timing_scope='sum of six serial isolated GEMM phase simulations; no vector/router/merge timing; not layer end-to-end',
        numerical_scope='connected complete layer-1 MoE including shared experts, BF16 nonlinear, FP32 routing combine',
        binary_sha256=digest(BINARY.read_bytes()),prefix_source=str(OUT/'real_inputs/manifest.json'))
    save(receipt,result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    DEST.mkdir(exist_ok=True)
    plan=[(b,s,m) for b in [2,4,8,16] for s in [[6],[3,3],[4,2],[2,2,2],[1]*6] for m in ['pinned_expert','tile_stealing']]
    save(DEST/'plan.json',dict(configurations=len(plan),gemm_phases_each=6,repeats=2,
        fp64_projection_scaled_error_limit=5e-6,final_moe_relative_l2_limit=.01,
        bit_exact_required='between architectures and against canonical FP32 K512 tree, not against arbitrary torch GEMM reduction'))
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fs={pool.submit(execute,*task):task for task in plan}
        for f in as_completed(fs):
            try:r=f.result()
            except Exception as e:
                save(DEST/'FAILURE.json',dict(task=fs[f],error=str(e)))
                for other in fs:other.cancel()
                raise
            results.append(r);save(DEST/'progress.json',dict(done=len(results),planned=len(plan),last=r['name']))
    for b in [2,4,8,16]:
        rs=[r for r in results if r['batch']==b]
        assert len(rs)==10
        assert len(set(r['output']['sha256'] for r in rs))==1
        assert len(set(r['all_projection_outputs_sha256'] for r in rs))==1
    rows=[p for r in results for p in r['matrix_phases']]
    with (DEST/'all_phases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(DEST/'validation.json',dict(passed=True,configurations=len(results),gemm_runs=12*len(results),
        architecture_outputs_bit_exact=True,repeats_bit_exact=True,all_fp64_checks_pass=True,
        worst_final_relative_l2=max(r['versus_pytorch']['relative_l2'] for r in results)))
    print('REAL CAMPAIGN COMPLETE',len(results),flush=True)
if __name__=='__main__':main()
