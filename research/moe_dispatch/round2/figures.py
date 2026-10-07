"""Ten scientific figures from frozen round2 result tables.

This reader never runs simulations or synthesizes data. Resource occupancy is
plotted with separate bars: overlapping counters are not wall-time components.
Unclosed search certificates are labelled best-evaluated candidates throughout.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

MODES=('pipelined','port_tight','fixed_issue')
BATCHES=(2,4,8,16,64,96,128)
NAMES={'B0':'B0 original single','B1':'B1 tuned single','B2':'B2 tuned homogeneous',
       'fixed_3+3':'Fixed 3+3','fixed_4+2':'Fixed 4+2','previous_asym':'Previous asymmetric',
       'best_5+1':'Selected 5+1','best_4+2':'Selected 4+2','best_2+4':'Selected 2+4 (mirror)',
       'best_hetero':'Selected heterogeneous','U1':'U1 shape diagnostic','U2':'U2 weight-read diagnostic'}
FLOWS=('OS','WS','IS')
COLORS=('#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9')
FAMILIES=('single','homogeneous','heterogeneous')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})


def number(x):
    if x is None or x=='':return float('nan')
    return float(x)


def truth(x):return x is True or str(x).lower()=='true'


def gm(values):
    a=np.asarray(list(values),float)
    return float(np.exp(np.log(a).mean())) if len(a) and np.all(a>0) else float('nan')


class Figures:
    def __init__(self,root,out=None):
        self.root=Path(root);self.out=Path(out) if out else self.root/'figures'
        self.inputs={};self.outputs=[]
    def csv(self,name):
        p=self.root/'results'/name
        if not p.exists():raise FileNotFoundError(f'Figure input missing: {p}')
        self.inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        with p.open() as f:rows=list(csv.DictReader(f))
        if not rows:raise ValueError(f'Empty figure input: {p}')
        return rows
    def js(self,name):
        p=self.root/'results'/name
        if not p.exists():raise FileNotFoundError(f'Figure input missing: {p}')
        self.inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    def save(self,fig,name,caption):
        self.out.mkdir(parents=True,exist_ok=True)
        fig.text(.01,.008,caption,ha='left',va='bottom',fontsize=8,wrap=True)
        fig.tight_layout(rect=(0,.065,1,.98))
        for ext in ('pdf','png'):
            p=self.out/f'{name}.{ext}';meta={'CreationDate':None,'ModDate':None} if ext=='pdf' else None;fig.savefig(p,dpi=200,bbox_inches='tight',metadata=meta);self.outputs.append(p)
        plt.close(fig)
    @staticmethod
    def title(mode):return mode+(' (non-iso-resource reference)' if mode=='fixed_issue' else '')

    def headroom(self):
        rows=self.csv('E1/bounds_per_window.csv')
        fields=('latency_ms','hbm_floor_unique','hbm_floor_actual','mac_floor','port_floor','task_floor')
        labels=('B1 latency','Unique-weight HBM floor','Actual-traffic HBM floor','Useful MAC floor','Mandatory port floor','Largest-task floor')
        fig,axes=plt.subplots(1,3,figsize=(15,4.4))
        for ax,mode in zip(axes,MODES):
            ss=[r for r in rows if r['design']=='B1' and r['onchip_mode']==mode]
            for j,(field,label) in enumerate(zip(fields,labels)):
                vals=[gm(number(r[field]) for r in ss if int(r['batch'])==b) for b in BATCHES]
                ax.plot(range(len(BATCHES)),vals,marker='o',lw=2 if j==0 else 1,label=label,color=COLORS[j])
            ax.set_xticks(range(len(BATCHES)),[f'B{b}' for b in BATCHES]);ax.set_ylabel('Geometric-mean latency / floor (ms)');ax.set_title(self.title(mode));ax.grid(axis='y',alpha=.2)
        axes[0].legend(fontsize=7,loc='upper left')
        self.save(fig,'fig_headroom','Analytical estimates, held-out windows; 1 cycle = 1 ns. Floors are separate lower bounds, not additive time components.')

    def main_bars(self):
        rows=self.csv('E4/heldout_main_table.csv')
        names=('B0','B1','B2','fixed_3+3','fixed_4+2','best_5+1','best_4+2','best_2+4','best_hetero','U1','U2')
        fig,axes=plt.subplots(1,3,figsize=(20,5));width=.075;x=np.arange(len(BATCHES))
        for ax,mode in zip(axes,MODES):
            ss=[r for r in rows if r['onchip_mode']==mode and r['sched_type']=='runtime'];base=next(r for r in ss if r['entry']=='B1')
            for j,name in enumerate(names):
                r=next(r for r in ss if r['entry']==name)
                vals=[number(base[f'B{b}'])/number(r[f'B{b}']) for b in BATCHES]
                ax.bar(x+(j-(len(names)-1)/2)*width,vals,width,color=COLORS[j%len(COLORS)],
                    hatch=('//' if name.startswith('U') else '..' if j>=6 else None),label=NAMES[name])
            ax.axhline(1,color='black',lw=.8);ax.set_xticks(x,[f'B{b}' for b in BATCHES]);ax.set_ylabel('Speedup relative to B1');ax.set_title(self.title(mode));ax.grid(axis='y',alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.99),ncol=6,fontsize=7)
        self.save(fig,'fig_main_bars','Runtime analytical estimates on held-out windows; one development-selected design is frozen across batches. Selected designs are best-evaluated candidates when search proofs remain open; no RTL calibration.')

    def breakdown(self):
        rows=self.csv('E4/per_window.csv');names=('B1','B2','best_hetero')
        fields=('cycles','hbm_busy_cycles','w_port_busy','x_port_busy','acc_port_busy','core0_compute_busy','core1_compute_busy')
        labels=('Wall time','HBM occupancy','W occupancy','X occupancy','Accumulator occupancy','Core 0 compute','Core 1 compute')
        fig,axes=plt.subplots(1,3,figsize=(15,4.6));x=np.arange(3);width=.115
        for ax,mode in zip(axes,MODES):
            for j,(field,label) in enumerate(zip(fields,labels)):
                vals=[]
                for name in names:
                    ss=[r for r in rows if r['entry']==name and r['onchip_mode']==mode and r['sched_type']=='runtime']
                    if not ss:raise ValueError(f'Breakdown rows missing: {mode}/{name}')
                    a=[number(r['cycles'])*number(r['hbm_busy_frac']) if field=='hbm_busy_cycles' else number(r[field]) for r in ss]
                    vals.append(float(np.mean(a))/1e6)
                ax.bar(x+(j-3)*width,vals,width,color=COLORS[j%len(COLORS)] if j<6 else '#777777',label=label)
            ax.set_xticks(x,['B1','B2','Selected hetero']);ax.set_ylabel('Mean duration / occupancy (ms)');ax.set_title(self.title(mode));ax.grid(axis='y',alpha=.2)
        axes[0].legend(fontsize=7)
        self.save(fig,'fig_breakdown','Separate, non-additive bars: HBM/W/X/accumulator/compute counters overlap and are not native stall categories. Means use identical held-out windows; 1e6 cycles = 1 ms.')

    def bnb_coverage(self):
        fig,axes=plt.subplots(3,2,figsize=(12,10))
        for k,mode in enumerate(MODES):
            obj=self.js(f'E3/bnb_{mode}_A.json');cert=obj['certificate']
            history=self.root/'results/E3'/f'seed_points_{mode}_partial.jsonl'
            if not history.exists():
                raise FileNotFoundError('Actual completed seed history missing: '+str(history))
            self.inputs[str(history)]=hashlib.sha256(history.read_bytes()).hexdigest()
            expected_hash=obj.get('resume',{}).get('engine_sha256')
            seeds=[json.loads(line) for line in history.read_text().splitlines() if line.strip()]
            seeds=[r for r in seeds if r.get('engine_sha256')==expected_hash]
            extra=self.root/'results/E3'/f'bnb_{mode}_B.json'
            leaves=list(obj['leaves'])
            if extra.exists():leaves+=self.js(f'E3/bnb_{mode}_B.json')['leaves']
            for j,family in enumerate(FAMILIES):
                per=defaultdict(int);total=obj['families'][family]['declared_lattice_points']
                for r in cert:
                    if r['family']==family and r['status']=='lower_bound_pruned':per[int(r['depth'])]+=int(r['lattice_points'])
                depths=sorted(per);vals=np.cumsum([per[d] for d in depths])/total*100 if depths else []
                axes[k,0].step(depths,vals,where='post',color=COLORS[j],label=family)
                points=[number(r['geomean_ms']) for r in seeds if r.get('family')==family and 'invalid' not in r]
                points += [number(r['geomean_ms']) for r in leaves if r.get('family')==family and r.get('geomean_ms') is not None]
                if points:axes[k,1].plot(np.arange(1,len(points)+1),np.minimum.accumulate(points),color=COLORS[j],label=family)
            for ax in axes[k]:ax.set_title(self.title(mode));ax.grid(alpha=.2)
            axes[k,0].set_ylabel('Declared lattice pruned by bounds (%)');axes[k,0].set_xlabel('Region depth');axes[k,0].set_ylim(0,100)
            axes[k,1].set_ylabel('Incumbent concrete score (ms)');axes[k,1].set_xlabel('Completed evaluations: seeds, proof A, proof B');axes[k,1].set_yscale('log')
        axes[0,0].legend();axes[0,1].legend()
        self.save(fig,'fig_bnb_coverage','Left: proof-A bound-pruned lattice by region depth. Right: incumbent history from actual seed completion log then A/B leaf evaluation order, separately for each family; open regions remain unless certified.')

    def workload_map(self):
        rows=self.csv('E3/workload_map.csv');cal=self.csv('E3/synthetic_calibration.csv')
        batches=(2,4,8,16,32,64,128,256);widths=sorted({number(r['bw_or_mac_scale']) for r in rows})
        fig,axes=plt.subplots(1,len(widths),figsize=(5*len(widths),5.8),squeeze=False)
        for ax,bw in zip(axes[0],widths):
            a=np.full((len(batches),5),np.nan);closed=np.zeros(a.shape,bool)
            for r in rows:
                if int(r['E'])==64 and int(r['topk'])==6 and int(r['F'])==1408 and int(r['shared_units'])==2 and abs(number(r['bw_or_mac_scale'])-bw)<1e-6:
                    bi=batches.index(int(r['batch']));ci=int(r['concentration']);a[bi,ci]=number(r['delta_vs_single_pct']);closed[bi,ci]=truth(r['proof_complete'])
            image=ax.imshow(np.ma.masked_invalid(a),origin='lower',aspect='auto',cmap='RdBu_r',vmin=-20,vmax=20)
            for (i,j),v in np.ndenumerate(a):
                if np.isfinite(v):
                    ax.text(j,i,f'{v:+.1f}'+('' if closed[i,j] else '*'),ha='center',va='center',fontsize=7)
            closest={}
            for r in cal:
                k=r['window_id'];v=number(r['me_hist_KL_real_to_synthetic'])
                if k not in closest or v<closest[k][0]:closest[k]=(v,int(r['concentration_level']),int(r['batch']))
            locations=defaultdict(int)
            for _,level,batch in closest.values():locations[level,math.log2(batch)-1]+=1
            if math.isclose(bw,256*32/65,rel_tol=1e-5):
                for (level,y),count in locations.items():ax.scatter(level,y,s=30+6*count,facecolors='none',edgecolors='black',lw=1.5)
            ax.set_xticks(range(5),['0\ndiffuse','1','2','3','4\nconcentrated']);ax.set_yticks(range(len(batches)),[f'B{b}' for b in batches]);ax.set_xlabel('Calibrated synthetic concentration level');ax.set_title(f'Effective HBM {bw:.1f} GB/s')
            fig.colorbar(image,ax=ax,label='Selected hetero / single latency change (%)')
        self.save(fig,'fig_workload_map','Synthetic region only: E=64, top-6, routed F=1408, Shared=2 units. Both concentration endpoints are fitted on development captures; diffuse does not mean perfectly uniform. Circles appear only in the actual ~126 GB/s regime and mark nearest calibrated capture histogram levels (mixed B96 on log2 batch axis); other bandwidth panels are hypothetical. * = open hardware-search proof; candidate estimate, not certified optimum.')

    def sobol(self):
        rows=self.csv('E3/sobol.csv');x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(9,4.8))
        for j,(val,ci,label) in enumerate((('S1','S1_ci','First order'),('ST','ST_ci','Total order'))):
            ax.bar(x+(j-.5)*.32,[number(r[val]) for r in rows],.32,yerr=[number(r[ci]) for r in rows],capsize=3,color=COLORS[j],label=label)
        ax.set_xticks(x,[r['param'].replace('_','\n') for r in rows]);ax.set_ylabel('Sobol sensitivity index');ax.legend();ax.grid(axis='y',alpha=.2)
        complete=all(truth(r['all_searches_certified']) for r in rows)
        self.save(fig,'fig_sobol','Hardware is reoptimized per parameter sample. '+('All sample proofs close at the declared 2% improvement tolerance; these are not exact optima.' if complete else 'Indices describe best-evaluated candidates from time-limited searches; they do not certify global-optimum sensitivity.')+' Error bars are estimator confidence intervals.')

    def flip_boundary(self):
        rows=self.csv('E3/flip_samples.csv');flips=self.csv('E3/flip_boundary.csv');params=list(dict.fromkeys(r['param'] for r in rows));fig,axes=plt.subplots(2,3,figsize=(14,8.1))
        for ax,key in zip(axes.flat,params):
            rs=sorted((r for r in rows if r['param']==key),key=lambda r:number(r['value']))
            x=[number(r['value']) for r in rs];y=[100*number(r['delta']) for r in rs];ax.plot(x,y,color=COLORS[0],lw=1.5)
            if all(r.get('delta_lower') not in (None,'') and r.get('delta_upper') not in (None,'') for r in rs):
                ax.fill_between(x,[100*number(r['delta_lower']) for r in rs],[100*number(r['delta_upper']) for r in rs],color=COLORS[0],alpha=.12)
            for xx,yy,r in zip(x,y,rs):ax.plot(xx,yy,'o',color=COLORS[0],markerfacecolor=COLORS[0] if truth(r['proof_complete']) else 'white',ms=4)
            ax.axhline(0,color='black',lw=.7);ax.axhline(-5,color=COLORS[2],lw=.7,ls='--')
            for r in flips:
                if r['param']==key and np.isfinite(number(r['value'])):ax.axvline(number(r['value']),color=COLORS[1],alpha=.5,ls=':')
            ax.set_xlabel(key);ax.set_ylabel('Hetero / single latency change (%)');ax.grid(alpha=.2)
        for ax in list(axes.flat)[len(params):]:ax.set_visible(False)
        self.save(fig,'fig_flip_boundary','One parameter varies; hardware is reoptimized at each sampled point. Hollow markers: open search proof. Shading, when available: legal family-optimum ratio intervals from lower bounds and executable incumbents, not statistical confidence. Dotted crossings interpolate sampled candidates; thresholds are not RTL-calibrated wins.')

    def dataflow_grid(self):
        rows=self.csv('E2/layer_grid.csv');names=('B1','fixed_4+2','previous_asym','best_hetero');fig,axes=plt.subplots(4,3,figsize=(12,12))
        for i,name in enumerate(names):
          for j,mode in enumerate(MODES):
            ax=axes[i,j];rs=[r for r in rows if r['design']==name and r['onchip_mode']==mode and r['batch']=='all'];single=name=='B1';a=np.full((3,1 if single else 3),np.nan)
            for r in rs:a[FLOWS.index(r['df_big']),0 if single else FLOWS.index(r['df_small'])]=number(r['ratio_vs_OS_OS'])
            ax.imshow(np.ma.masked_invalid(a),cmap='YlOrRd',aspect='auto',vmin=min(1,np.nanmin(a)),vmax=max(1.0001,np.nanmax(a)))
            for (y,x),v in np.ndenumerate(a):
                if np.isfinite(v):ax.text(x,y,f'{v:.3f}x',ha='center',va='center',fontsize=9)
            ax.set_yticks(range(3),FLOWS);ax.set_xticks(range(a.shape[1]),['single'] if single else FLOWS);ax.set_ylabel('Single-core flow' if single else 'Large-core flow');ax.set_xlabel('Small-core flow' if not single else '');ax.set_title(NAMES[name]+'\n'+self.title(mode),fontsize=9)
        self.save(fig,'fig_dataflow_grid','Held-out, all-window paired geometric latency ratio relative to the same design OS/OS (single: OS). Colors normalize within each panel; compare the annotated ratios across panels. Capacity, ports, geometry and policy stay frozen; dataflow benefits are not additive. Large/small labels follow multiplier counts (core 0 first on a tie); B1 has no second core.')

    def me_crossover(self):
        rows=self.csv('E2/micro.csv');fig,axes=plt.subplots(3,3,figsize=(14,11))
        shapes=list(dict.fromkeys(r['shape'] for r in rows))
        for i,flow in enumerate(FLOWS):
          for j,mode in enumerate(MODES):
            ax=axes[i,j];curves={}
            for k,shape in enumerate(shapes):
                rs=sorted((r for r in rows if r['shape']==shape and r['dataflow']==flow and r['expert_type']=='routed' and r['onchip_mode']==mode),key=lambda r:int(r['Me']))
                if not rs:continue
                x=np.array([int(r['Me']) for r in rs]);y=np.array([number(r['cycles']) for r in rs]);curves[shape]=(x,y)
                macs=math.prod(map(int,shape.split('x')));ax.plot(x,y,'o-',ms=2.5,lw=1,label=f'{shape} ({macs:,} mult.)',color=COLORS[k%len(COLORS)])
            if curves:
                grid=next(iter(curves.values()))[0];winners=[min(curves,key=lambda s:curves[s][1][n]) for n in range(len(grid))]
                for n in range(1,len(grid)):
                    if winners[n]!=winners[n-1]:ax.axvspan(grid[n-1],grid[n],color='black',alpha=.08)
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Tokens per routed expert Me');ax.set_ylabel('Isolated expert latency (cycles)');ax.set_title(flow+' / '+self.title(mode),fontsize=9);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=6)
        self.save(fig,'fig_me_crossover','Micro-only latency includes finite HBM, private SRAM ports, compute and cold-start/phase service. Each individual core receives full declared private quotas; shapes have different multiplier counts, so this is not an iso-resource layer comparison or a compute-only timing experiment. Shaded intervals bracket changes in the fastest sampled core; no exact crossover is inferred between discrete Me samples.')

    def predictor(self):
        rows=self.csv('E5/predictor_table.csv');designs=('best_hetero','fixed_4+2');order=('random','static','btb','ema','ours','oracle');fig,axes=plt.subplots(4,3,figsize=(14,11))
        for di,name in enumerate(designs):
          for mi,mode in enumerate(MODES):
            ss=[r for r in rows if r['design']==name and r['onchip_mode']==mode];rr=[next(r for r in ss if r['predictor']==p) for p in order];x=np.arange(len(order))
            ax=axes[2*di,mi];ax.bar(x,[number(r['mae_pct']) for r in rr],color=COLORS);ax.set_ylabel('Mean absolute relative error (%)');ax.set_title(NAMES[name]+' / '+self.title(mode),fontsize=9);ax.grid(axis='y',alpha=.2)
            ax2=axes[2*di+1,mi];ax2.bar(x,[number(r['e2e_ratio_vs_oracle']) for r in rr],color=COLORS);ax2.axhline(1,color='black',lw=.8);ax2.set_ylabel('MoE latency / frozen-ours timing reference');ax2.grid(axis='y',alpha=.2)
            for target in (ax,ax2):target.set_xticks(x,order,rotation=25,ha='right')
        self.save(fig,'fig_predictor','Same development warmup and held-out sequence, repeated with fresh initial states. MAE measures task duration prediction; latency is post-router MoE only. Primary oracle physically replays the frozen actual ours owners, binding, prefetch and phase-release plan, so its latency equals that same plan. It measures conditional timing accuracy, not perfect-prediction dispatch performance or an optimal-scheduling upper bound; floating replay residuals are retained. No circuit area claims.')

    def manifest(self):
        if not self.outputs:return
        self.out.mkdir(parents=True,exist_ok=True)
        obj={'scope':'BF16 post-router phase-fluid analytical figures; no RTL/native timing claimed',
             'figure_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'input_sha256':self.inputs,'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.outputs}}
        (self.out/'FIGURE_PROVENANCE.json').write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')

METHODS=('headroom','main_bars','breakdown','bnb_coverage','workload_map','sobol','flip_boundary','dataflow_grid','me_crossover','predictor')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--out',type=Path);ap.add_argument('--only',nargs='+',choices=METHODS)
    args=ap.parse_args();plots=Figures(args.root,args.out)
    for name in args.only or METHODS:
        getattr(plots,name)();print('generated fig_'+name,flush=True)
    plots.manifest()

if __name__=='__main__':main()
