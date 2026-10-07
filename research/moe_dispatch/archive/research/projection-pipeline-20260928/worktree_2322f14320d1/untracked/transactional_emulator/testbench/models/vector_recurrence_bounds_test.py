"""Malformed compiled FSM program must fail before issuing a 129th leaf."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
from compiler.aten.plena.vector_recurrence_candidate import CandidateEmitter,VectorRecurrencePrimitive as P
from compiler.aten.plena.ltile_native import CoefficientView
from transactional_emulator.testbench.aten.recurrent_gate_test import AssemblyToBinary,COMPILER_ROOT,digest
from analytic_models.performance.ltile_platform import ExecutionProfile


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--runtime',type=Path,required=True);p.add_argument('--settings',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 e=CandidateEmitter('kda',4);e.configure(rows=32)
 e.coefficient(2,CoefficientView(0,128,1,0,16,2048));e.execute(P.TREE_RESET)
 for _ in range(128):e.execute(P.TREE_ACC)
 # Valid descriptors/data ranges isolate the excess-leaf failure.
 e.execute(P.READOUT_PRODUCT,destination=5,source=20)
 asm=a.output/'invalid.asm';asm.write_text('\n'.join(e.lines)+'\n')
 binary=a.output/'invalid.mem';AssemblyToBinary(str(COMPILER_ROOT/'doc/operation.svh'),str(COMPILER_ROOT/'doc/configuration.svh')).generate_binary(str(asm),str(binary))
 for name,n in [('hbm.bin',4096),('fp.bin',64),('int.bin',64)]: (a.output/name).write_bytes(bytes(n))
 env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',RUST_LOG='warn');env.update(ExecutionProfile().runtime_environment())
 if os.environ.get('LIBTORCH'):env['LD_LIBRARY_PATH']=str(Path(os.environ['LIBTORCH'])/'lib')+':'+env.get('LD_LIBRARY_PATH','')
 cmd=[str(a.runtime),'--opcode',str(binary),'--hbm',str(a.output/'hbm.bin'),'--fpsram',str(a.output/'fp.bin'),'--intsram',str(a.output/'int.bin'),'--settings',str(a.settings),'--hbm-size','4096']
 result=subprocess.run(cmd,cwd=a.output,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
 (a.output/'run.log').write_text(result.stdout)
 assert result.returncode!=0 and 'Vector FSM exceeds 128 tree leaves' in result.stdout,result.stdout
 evidence=dict(rejected=True,exit_code=result.returncode,program_sha256=digest(binary),runtime_sha256=digest(a.runtime),command=cmd,
  scope='malformed FSM rejected by preflight tree bound; all descriptors/data ranges otherwise valid; no silicon assertion')
 (a.output/'negative_check.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence))

if __name__=='__main__':main()
