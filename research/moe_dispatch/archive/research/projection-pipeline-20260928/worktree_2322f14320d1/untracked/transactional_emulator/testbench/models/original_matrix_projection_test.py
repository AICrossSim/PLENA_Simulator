"""Machine-code calibration of bounded existing M_MM/M_TMM, not M_MM.P."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from analytic_models.performance.ltile_platform import ExecutionProfile
from transactional_emulator.testbench.aten.recurrent_gate_test import Arena, bf
from transactional_emulator.testbench.aten.recurrent_conv_test import run_program, read
from transactional_emulator.testbench.aten.matrix_projection_test import reference
from compiler.aten.plena.isa_matrix_projection import Projection
from compiler.aten.plena.isa_projection_software import lower_matrix_projection


def case(root,runtime,memory,b,k,n,ktile=1024,transpose=True):
    rng=np.random.default_rng(31007+b+k+n)
    arena=Arena()
    zero=arena.add(np.zeros(2048))
    p=Projection(0,0,0,0,k,n,ktile)
    xs=[bf(rng.normal(0,.2,k)) for _ in range(b)]
    inputs=[arena.add(np.pad(x,(0,p.input_values-k))) for x in xs]
    w=bf(rng.normal(0,.1,(k,n)))
    padded=np.pad(w,((0,(-k)%32),(0,(-n)%4)))
    packets=[]
    for col in range(0,n,4):
        for k0 in range(0,k,ktile):
            rows=(min(ktile,k-k0)+31)//32*32
            block=padded[k0:k0+rows,col:col+4]
            if transpose:
                packets.append(block.T.copy().reshape(-1))
            else:
                packets.append(np.pad(block,((0,0),(0,28))).reshape(-1))
    weights=arena.add(np.concatenate(packets))
    outputs=[arena.add(np.full(p.output_values,7),output=True) for _ in xs]
    p=replace(p,inputs=inputs[0],weights=weights,outputs=outputs[0],zero=zero)
    text=lower_matrix_projection(p,inputs,outputs,transpose=transpose)
    controllers=len(json.loads((memory/'ramulator.json').read_text())['memory_system']['controllers'])
    profile=ExecutionProfile(hbm_controllers=controllers,projection_schedule='matrix',projection_k_tile=ktile)
    image,result=run_program(root,runtime,memory,arena,text,profile=profile)
    for x,a in zip(xs,outputs):
        np.testing.assert_array_equal(read(image,a,n),reference(x,w,ktile,profile.matrix))
        np.testing.assert_array_equal(read(image,a+n*2,p.output_values-n),0)
    return dict(case=root.name,status='passed',dimensions=dict(batch=b,k=k,n=n,k_tile=ktile,transpose=transpose),
                checked_values=b*n,logical_weight_bytes=k*n*2,packed_weight_bytes=sum(x.size*2 for x in packets),**result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',required=True,type=Path)
    p.add_argument('--memory-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cases=[(b,1057,35,1024,True) for b in (1,2,4,8,16)]
    cases += [(4,65,7,256,False),(16,7168,35,1024,True),(16,12288,35,1024,True),
              (4,1057,2051,1024,True),(16,2049,2051,1024,True)]
    results=[]
    for i,(b,k,n,kt,tr) in enumerate(cases):
        r=case(args.output/f'case{i}',args.runtime,args.memory_root,b,k,n,kt,tr)
        results.append(r)
        print(r['case'],r['status'],flush=True)
        (args.output/'validation.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
