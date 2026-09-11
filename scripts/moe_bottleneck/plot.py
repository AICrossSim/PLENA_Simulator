#!/usr/bin/env python3
"""Export measured simulator sensitivities; no fitted or predicted values."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/scratch/shared/mcl123/plena/outputs/moe_bottleneck_20260911')
rows=list(csv.DictReader((ROOT/'sensitivity.csv').open()))
by={(r['case'],r['profile']):float(r['total_us']) for r in rows}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(15,4.7))
colors=['#394b59','#7656a8','#287da8','#cf863b']
labels=['Baseline','MAC service 2x','HBM clock 2x','Accumulator port 2x']
profiles=['baseline','mac2','hbm_clock2','accumulator2']
x=np.arange(3);width=.19
for i,(profile,label,color) in enumerate(zip(profiles,labels,colors)):
 vals=[by[(f'expert_me{m}_single_legacy_n3',profile)] for m in (1,8,32)]
 axes[0].bar(x+(i-1.5)*width,vals,width,label=label,color=color)
axes[0].set_xticks(x,['1','8','32']);axes[0].set_xlabel('Tokens per expert (Me)')
axes[0].set_ylabel('Simulated latency (us)');axes[0].set_title('Single expert / single N3 core')
axes[0].legend(frameon=False,fontsize=8)
for ax,batch in zip(axes[1:],[8,32]):
 x=np.arange(4);width=.25
 configs=['single_legacy_n3','single_pool_q32','heterogeneous_legacy_n2','heterogeneous_pool_q32']
 for i,(profile,label,color) in enumerate(zip(['baseline','supply2','all2'],['Baseline','Supply/control 2x','Supply/control + MAC 2x'],colors)):
  vals=[by[(f'qwen_full_decode_b{batch}_{c}',profile)]/1000 for c in configs]
  ax.bar(x+(i-1)*width,vals,width,color=color,label=label)
 ax.set_xticks(x,['Single\nN3','Single\nQ32','Big/small\nN2','Big/small\nQ32'])
 ax.set_ylabel('Simulated latency (ms)');ax.set_title(f'Archived route window B{batch}')
 ax.legend(frameon=False,fontsize=8)
fig.suptitle('PLENA MoE: the limiting resource changes with workload',fontsize=15,y=.99)
fig.text(.5,.015,'Counterfactual service experiments, not equal-cost hardware speedups. Same shapes, encoded bytes and per-core job order.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.95))
fig.savefig(ROOT/'bottleneck_summary.png',dpi=180)
fig.savefig(ROOT/'bottleneck_summary.svg')
