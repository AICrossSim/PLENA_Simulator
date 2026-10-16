"""Reproduce inputs, all searches, legacy comparison, tables and final audit."""
import subprocess
import sys
from .common import CFG,REPO
from .trace import load_inputs

def main():
    load_inputs()
    package='research.moe_dispatch.round7_64k_dualcore_dse'
    for model in CFG['models']:
        for overhead in CFG['overheads_per_gemm']:
            subprocess.run([sys.executable,'-m',package+'.search','--model',model,'--overhead',str(overhead)],cwd=REPO,check=True)
    for stage in ('legacy','report','audit'):
        subprocess.run([sys.executable,'-m',package+'.'+stage],cwd=REPO,check=True)

if __name__=='__main__':main()
