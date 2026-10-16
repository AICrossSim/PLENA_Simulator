"""Reproduce full Round6 locally; git commits/push are deliberately explicit."""
import argparse,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from .common import ROOT,BUDGETS

def invoke(module,args=(),log=None):
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
 cmd=[sys.executable,'-m','research.moe_dispatch.round6_feasible_cell.'+module,*map(str,args)]
 if log:
  with open(log,'w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 else:subprocess.run(cmd,env=env,check=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','search','evaluate','report'],default='all');a=ap.parse_args()
 if a.stage in ('all','search'):
  invoke('inputs')
  with ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(lambda b:invoke('search',['--budget',b],f'/tmp/plena-round6-search-{b}.log'),BUDGETS))
 if a.stage in ('all','evaluate'):
  invoke('evaluate',['--tune-only'])
  with ThreadPoolExecutor(max_workers=5) as ex:list(ex.map(lambda b:invoke('evaluate',['--budget',b,'--appendix'],f'/tmp/plena-round6-evaluate-{b}.log'),BUDGETS))
 if a.stage in ('all','report'):invoke('deliver')

if __name__=='__main__':main()
