"""Synthetic, distinct-head machine checks for fused Vector alternatives.

No model weights, hardware RTL or task-quality assertion. Prepared BF16 inputs
exercise all 128 state rows, KDA's original-state two-pass dependency and the
same BF16 pairwise product tree; Compiler emits the executed machine code.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from compiler.aten.plena.vector_recurrence_candidate import lower_vector_candidate_group
from compiler.aten.plena.recurrent_coefficients import CompactCoefficientLoader
from compiler.aten.plena.ltile_v2 import Options, lower_group
from transactional_emulator.testbench.aten.recurrent_gate_test import Arena,bf,digest
from transactional_emulator.testbench.aten.recurrent_conv_test import run_program,read
from analytic_models.performance.ltile_platform import ExecutionProfile
from analytic_models.performance.ltile_cost import Machine


def tree(x):
    x=bf(x)
    while len(x)>1:x=bf(x[0::2]+x[1::2])
    return x[0]


def case(kind,level,alu,root,runtime,memory,lanes=256):
    # Every ladder level and width receives the same per-model payload.
    rng=np.random.default_rng(620100+(100 if kind=="kda" else 0))
    width=64 if kind=="mamba" else 128;heads=2048//width
    a=Arena()
    state=bf(rng.normal(0,.03,(128,heads,width)))
    x=bf(rng.normal(0,.05,(heads,width)))
    delta=bf(rng.uniform(.001,.08,heads if kind=="mamba" else (heads,128)))
    b=bf(rng.normal(0,.04,(heads//8,128) if kind=="mamba" else (heads,128)))
    c=bf(rng.normal(0,.04,b.shape))
    scalar=np.zeros(2048,np.float32);skip=np.zeros(2048,np.float32)
    if kind=="mamba":
        scalar[1:heads*2:2]=bf(rng.uniform(.02,.1,heads))
        skip[:heads*2:2]=1;skip[1:heads*2:2]=bf(rng.normal(0,.1,heads))
        u=bf(x*scalar[1:heads*2:2,None])
        expanded_b=np.repeat(b,8,axis=0).T[:,:,None]
        expanded_c=np.repeat(c,8,axis=0).T[:,:,None]
        expected=bf((state-delta[None,:,None]*state)+expanded_b*u[None])
        output=bf(tree(bf(expected*expanded_c))+skip[1:heads*2:2,None]*x)
    else:
        scalar[:heads*2:2]=bf(rng.uniform(.1,.8,heads))
        decayed=state-delta.T[:,:,None]*state
        prediction=tree(bf(decayed*b.T[:,:,None]))
        u=bf((x-prediction)*scalar[:heads*2:2,None])
        expected=bf(decayed+b.T[:,:,None]*u[None])
        output=tree(bf(expected*c.T[:,:,None]))
    zero=a.add(np.zeros(2048,np.float32));onehot=a.add(np.r_[1,np.zeros(2047)].astype(np.float32))
    # Matrix-view DMA uses canonical head-major storage; streamed Vector
    # groups use row/head/column order. Both contain the same logical state.
    state_image=np.swapaxes(state,0,1) if level==5 else state
    initial=a.add(state_image,output=True);xp=a.add(x);sp=a.add(scalar);sk=a.add(skip)
    def field(values):return a.add(np.pad(values.reshape(-1),(0,2048-values.size)))
    dp,bp,cp=field(delta),field(b),field(c)
    out=a.add(np.zeros(2048,np.float32),output=True)
    if kind=="mamba":
        native=[(dp,0,1,0,0),(bp,0,128,1,3),(cp,0,128,1,3)]
        mapping={}
        for name,p,offset in (("a",dp,0),("dt",sp,1),("d",sk,1)):
            mapping[name,0]=[(p,(2*h+offset) if name!="a" else h,h*width,width) for h in range(heads)]
        for row in range(128):
            for name,p in (("b",bp),("c",cp)):
                mapping[name,row]=[(p,h//8*128+row,h*width,8*width) for h in range(0,heads,8)]
    else:
        native=[(p,0,128,1,0) for p in (dp,bp,cp)]
        mapping={("beta",0):[(sp,2*h,h*width,width) for h in range(heads)]}
        for row in range(128):
            for name,p in (("decay",dp),("key",bp),("query",cp)):
                mapping[name,row]=[(p,h*128+row,h*width,width) for h in range(heads)]
    intervals=sorted({(start,n) for entries in mapping.values() for _,_,start,n in entries})
    masks={}
    for start,n in intervals:
        v=np.zeros(2048,np.float32);v[start:start+n]=1;masks[start,n]=a.add(v)
    loader=CompactCoefficientLoader(mapping,masks,onehot) if level==1 else None
    if level==5:
        text='\n'.join(lower_group(Options(kind),dict(states=[initial],input=xp,output=out,
            scalar=sp,skip=sk,native=native)).lines)
    else:
        text=lower_vector_candidate_group(kind,level=level,state_base=initial,input_base=xp,output_base=out,
            zero_base=zero,scalar_base=sp,skip_base=sk,native=None if level==1 else native,coefficient_loader=loader)
    config=json.loads((memory/"ramulator.json").read_text())
    profile=ExecutionProfile(machine=replace(ExecutionProfile().machine,vector_rec_alu=alu,lanes=lanes),
        hbm_controllers=len(config["memory_system"]["controllers"]))
    image,result=run_program(root,runtime,memory,a,text,machine=profile.machine,profile=profile)
    actual_s=read(image,initial,expected.size)
    actual_s=np.swapaxes(actual_s.reshape(heads,128,width),0,1) if level==5 else actual_s.reshape(expected.shape)
    actual_o=read(image,out,2048).reshape(output.shape)
    if not np.array_equal(actual_s,expected) or not np.array_equal(actual_o,output):
        raise AssertionError(dict(state_max=float(np.max(np.abs(actual_s-expected))),output_max=float(np.max(np.abs(actual_o-output)))))
    evidence=dict(kind=kind,level=level,alu=alu,lanes=lanes,exact=True,values=int(expected.size+output.size),
        state_sha256=__import__('hashlib').sha256(actual_s.tobytes()).hexdigest(),components=result["prediction"],
        output_sha256=__import__('hashlib').sha256(actual_o.tobytes()).hexdigest(),
        input_sha256=digest(root/"hbm_for_behave_sim.bin"),runtime_sha256=digest(runtime),
        storage="native Matrix SRAM" if level==5 else "Vector SRAM only",
        HBM_state_order="head/row/column" if level==5 else "row/head/column",
        scope="prepared synthetic128rows; state/private addresses and BF16 tree; not realweights/longchain/RTL")
    (root/"numerical_check.json").write_text(json.dumps(evidence,indent=2)+"\n")
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",type=Path,required=True);p.add_argument("--memory-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--levels",nargs="+",type=int,choices=range(1,6),default=[1,2,4])
    p.add_argument("--alus",nargs="+",choices=["shared","dedicated"],default=["shared"])
    p.add_argument("--lanes",nargs="+",type=int,choices=[128,256,512],default=[256])
    p.add_argument("--models",nargs="+",choices=["mamba","kda"],default=["mamba","kda"])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    checks=[]
    for kind in args.models:
        for level in args.levels:
            for alu in args.alus:
                for lanes in args.lanes:
                    checks.append(case(kind,level,alu,args.output/f"{kind}_l{level}_{alu}_L{lanes}",args.runtime,args.memory_root,lanes))
    (args.output/"summary.json").write_text(json.dumps(checks,indent=2)+"\n")
    for kind in args.models:
        matched=[c for c in checks if c["kind"]==kind]
        assert len({(c["state_sha256"],c["output_sha256"]) for c in matched})==1, "ladder/width changes arithmetic"
    print(json.dumps(dict(checks=len(checks),exact=all(c["exact"] for c in checks))))


if __name__=="__main__":main()
