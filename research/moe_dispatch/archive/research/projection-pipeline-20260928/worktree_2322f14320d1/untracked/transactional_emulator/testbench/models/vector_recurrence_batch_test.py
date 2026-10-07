"""Distinct KDA request states through one finite reused Vector workspace.

Compiled prepared-input recurrence only, not projection, realweights or a
full-model benchmark. Six timing components and physical DMA are also checked.
"""
import argparse, hashlib, json
from dataclasses import replace
from pathlib import Path
import numpy as np
from compiler.aten.plena.vector_recurrence_candidate import lower_vector_candidate_group
from transactional_emulator.testbench.aten.recurrent_gate_test import Arena,bf,digest
from transactional_emulator.testbench.aten.recurrent_conv_test import run_program,read
from transactional_emulator.testbench.models.vector_recurrence_candidate_test import tree
from analytic_models.performance.ltile_platform import ExecutionProfile


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--runtime',type=Path,required=True);p.add_argument('--memory-root',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True);p.add_argument('--batch',type=int,default=16)
 args=p.parse_args();assert args.batch in [1,2,4,8,16]
 rng=np.random.default_rng(620116);a=Arena();zero=a.add(np.zeros(2048,np.float32))
 refs=[];program=[];regions=[]
 for request in range(args.batch):
  s=bf(rng.normal(0,.03,(128,16,128)));x=bf(rng.normal(0,.05,(16,128)))
  delta=bf(rng.uniform(.001,.08,(16,128)));key=bf(rng.normal(0,.04,(16,128)));query=bf(rng.normal(0,.04,(16,128)))
  scalar=np.zeros(2048,np.float32);scalar[:32:2]=bf(rng.uniform(.1,.8,16))
  decayed=s-delta.T[:,:,None]*s;pred=tree(bf(decayed*key.T[:,:,None]))
  u=bf((x-pred)*scalar[:32:2,None]);state=bf(decayed+key.T[:,:,None]*u[None]);out=tree(bf(state*query.T[:,:,None]))
  sp=a.add(s,output=True);xp=a.add(x);sc=a.add(scalar);fields=[a.add(v) for v in [delta,key,query]]
  op=a.add(np.zeros(2048,np.float32),output=True)
  refs.append((sp,op,state,out));regions.append(dict(request=request,state_base=sp,state_bytes=s.size*2,output_base=op))
  program.append(lower_vector_candidate_group('kda',level=2,state_base=sp,input_base=xp,output_base=op,
   zero_base=zero,scalar_base=sc,skip_base=sc,native=[(f,0,128,1,0) for f in fields]))
 assert len({r['state_base'] for r in regions})==args.batch
 assert all(a0['state_base']+a0['state_bytes']<=a1['state_base'] for a0,a1 in zip(regions,regions[1:]))
 config=json.loads((args.memory_root/'ramulator.json').read_text())
 profile=ExecutionProfile(hbm_controllers=len(config['memory_system']['controllers']))
 image,result=run_program(args.output,args.runtime,args.memory_root,a,'\n'.join(program),machine=profile.machine,profile=profile)
 checks=[]
 for i,(sp,op,state,out) in enumerate(refs):
  actual_s=read(image,sp,state.size).reshape(state.shape);actual_o=read(image,op,2048).reshape(out.shape)
  assert np.array_equal(actual_s,state) and np.array_equal(actual_o,out),f'request {i} contaminated'
  checks.append(dict(request=i,state_sha256=hashlib.sha256(actual_s.tobytes()).hexdigest(),output_sha256=hashlib.sha256(actual_o.tobytes()).hexdigest()))
 assert len({c['state_sha256'] for c in checks})==args.batch
 evidence=dict(batch=args.batch,kind='kda',level=2,exact=True,checks=checks,regions=regions,
  runtime_sha256=digest(args.runtime),program_sha256=digest(args.output/'generated_machine_code.mem'),
  components=result['prediction'],scope='prepared synthetic16 private128row state groups, reused Vector SRAM; no projection/fullweights/fullmodel/RTL')
 (args.output/'batch_private_check.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps(dict(batch=args.batch,exact=True)))

if __name__=='__main__':main()
