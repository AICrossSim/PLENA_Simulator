#!/usr/bin/env python3
"""Static scientific figures from measured CSVs; no interpolated data."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from prepare_routes import OUT

def main():
    dst=OUT/'figures';dst.mkdir(exist_ok=True)
    rs=list(csv.DictReader((OUT/'summary/real_moe_values_and_cycles.csv').open()))
    shapes=['6','3+3','4+2','2+2+2','1+1+1+1+1+1'];labels=['6','3+3','4+2','2+2+2','6 x 1']
    fig,axs=plt.subplots(2,2,figsize=(10.8,7.5),sharey=True)
    for ax,b in zip(axs.flat,[2,4,8,16]):
        a={r['shape']:r for r in rs if int(r['batch'])==b and r['mode']=='pinned_expert'}
        x=np.arange(5)
        ax.bar(x-.19,[float(a[s]['pure_gemm_phase_sum'])/1000 for s in shapes],.38,label='Compute only',color='#4788b7')
        ax.bar(x+.19,[float(a[s]['finite_gemm_phase_sum'])/1000 for s in shapes],.38,label='Finite interfaces + control',color='#d68b35')
        ax.set_xticks(x,labels);ax.set_title(f'Batch {b}',loc='left',fontweight='bold')
        ax.set_ylabel('Six serial GEMM phases (1,000 cycles)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Actual DeepSeek MoE operands: compute gains can shrink or reverse',fontweight='bold')
    fig.text(.5,.015,'BF16, pinned expert ownership, equal 12,288 multipliers. Router/nonlinear/merge timing excluded; not whole-layer latency.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ['svg','pdf','png']:fig.savefig(dst/f'real_batch_comparison.{ext}',dpi=180)
    plt.close(fig)
    a=[r for r in csv.DictReader((OUT/'summary/real_issue_state_partition.csv').open()) if int(r['batch'])==8 and r['shape'] in ['3+3','4+2']]
    fig,ax=plt.subplots(figsize=(9,3.8));left=np.zeros(2)
    for key,label,color in [('issue_actor_cycles','Issue actor','#4788b7'),('wait_control','Control issue/install wait','#d68b35'),
        ('wait_weight_source','Weight-source wait','#70a885'),('other','Other','#aaaaaa')]:
        values=np.array([float(x[key])/1000 for x in a]);ax.barh([0,1],values,left=left,label=label,color=color);left+=values
    for i,t in enumerate(left):ax.text(t+3,i,f'{t:.3f}',va='center')
    ax.set_yticks([0,1],[x['shape'] for x in a]);ax.invert_yaxis();ax.set_xlim(0,445)
    ax.set_xlabel('Sum of each serial phase\'s last-core issue states (1,000 cycles)')
    ax.set_title('Actual B8: fewer issue cycles, more exposed waiting',loc='left',fontweight='bold',pad=42)
    ax.legend(ncol=4,loc='lower left',bbox_to_anchor=(0,1.01),fontsize=8,frameon=False)
    fig.text(.5,.015,'Mutually exclusive priority classification; not independently removable causal delays or total MAC busy time.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ['svg','pdf','png']:fig.savefig(dst/f'real_b8_waits.{ext}',dpi=180)
    plt.close(fig)
if __name__=='__main__':main()
