#!/usr/bin/env python3
"""Standalone scientific figure from the fully audited result summary."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root):
    data=json.loads((root/'summary.json').read_text())
    if data['status']!='passed':raise ValueError('complete audited summary required')
    categories=['single','homogeneous','equal_pe_different_shape','large_small']
    labels=['Single core','Identical pair','Equal-PE, different shapes','Large + small']
    colors=['#263d61','#599ba8','#b3a17f','#cf7756']
    fig,axes=plt.subplots(1,2,figsize=(13.6,4.5),sharey=True)
    for ax,partition,title in zip(axes,['primary','holdout'],['Search inputs','Unseen route windows']):
        times=data['scores'][partition]['timings_ps']
        names=sorted(times['single'],key=lambda n:(0 if n.startswith('qwen') else 1,8 if n.endswith('b8') else 32))
        x=np.arange(len(names));width=.19
        for i,(category,label,color) in enumerate(zip(categories,labels,colors)):
            values=[times[category][n]/1e9 for n in names]
            ax.bar(x+(i-1.5)*width,values,width=width,label=label,color=color)
        ax.set_xticks(x,['Qwen\nB8','Qwen\nB32','DeepSeek\nB8','DeepSeek\nB32'])
        ax.set_title(title,loc='left',fontweight='bold');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('MoE FFN latency (ms), lower is faster')
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.03))
    fig.text(.01,.015,'4096 multipliers; equal total SRAM and HBM. One fixed winner per category. Synthetic values + archived routes.\nRust numerical execution + Ramulator; compute/SRAM timing is analytical, not RTL-calibrated.',fontsize=8,color='#555555')
    fig.tight_layout(rect=[0,.1,1,.94])
    fig.savefig(root/'latency_comparison.png',dpi=180,bbox_inches='tight')
    fig.savefig(root/'latency_comparison.svg',bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);plot(p.parse_args().root.resolve())
