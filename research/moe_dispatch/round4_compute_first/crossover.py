"""Exact piecewise-log breakpoints for the explicitly estimated E4 curves."""
import math
import numpy as np
from .common import ROOT,write_csv,write_json,gm,table
from .requirements import load_rows
from .evaluate import selected
from .config import BATCHES

def roof(rows,bw):
    return np.maximum(np.array([r['Tc_ms'] for r in rows]),np.array([r['hbm_bytes'] for r in rows])/(bw*1e6))

def crossing(candidate,baselines,reduction,max_bw=9830):
    # log(GM(base/candidate)) is affine between traffic/compute breakpoints.
    # Search all intervals: no monotonicity assumption or coarse-grid miss.
    cuts={1e-9,max_bw}
    for rs in (candidate,*baselines):
        for r in rs:
            b=r['hbm_bytes']/(r['Tc_ms']*1e6)
            if 1e-9<b<max_bw:cuts.add(b)
    cuts=sorted(cuts); target=-math.log1p(-reduction)
    def values(b):
        c=np.log(roof(candidate,b))
        return [float((np.log(roof(base,b))-c).mean())-target for base in baselines]
    for a,b in zip(cuts,cuts[1:]):
        va,vb=values(a),values(b);lo,hi=0.,1.
        for x,y in zip(va,vb):
            slope=y-x
            if abs(slope)<1e-12:
                if x< -1e-12:lo=2;break
            elif slope>0:lo=max(lo,-x/slope)
            else:hi=min(hi,-x/slope)
        if lo<=hi+1e-12 and lo<=1+1e-12:
            z=math.exp(math.log(a)+max(0,min(1,lo))*(math.log(b)-math.log(a)))
            return 0.0 if z<=1.001e-9 else z
    return None

def run():
    rows=load_rows(ROOT/'E1/compute_by_window.csv');sd=selected();out=ROOT/'E4';out.mkdir(exist_ok=True)
    ix={(r['mode'],r['design'],r['window_id']):r for r in rows}
    results=[];per=[];wins=[]
    rng=np.random.default_rng(20261016)
    for cap in (256,512):
        base=f'pn{cap}_single'; homo=min((f'pn{cap}_homo2',f'pn{cap}_homo4'),key=lambda k:sd[k]['dev_GM_cycles'])
        for mode in ('B','A'):
            for family in ('hetero11','hetero31','hetero71'):
                name=f'pn{cap}_{family}'
                for batch in (*BATCHES,'all'):
                    rs=sorted([r for r in rows if r['mode']==mode and r['design']==name and (batch=='all' or r['batch']==batch)],key=lambda r:r['window_id'])
                    refs=[[ix[mode,b,r['window_id']] for r in rs] for b in
                          (base,f'pn{cap}_homo2',f'pn{cap}_homo4')]
                    for red in (.05,.10):
                        both=crossing(rs,refs,red)
                        single_cross=crossing(rs,[refs[0]],red)
                        homo_cross=crossing(rs,refs[1:],red)
                        results.append(dict(pn_cap=cap,mode=mode,design=name,batch=batch,
                            threshold_latency_reduction_pct=red*100,single_baseline=base,
                            homo_baseline=f'pn{cap}_homo2 AND pn{cap}_homo4',
                            min_bw_vs_single_GBps=single_cross if single_cross is not None else '不存在',
                            min_bw_vs_homo_GBps=homo_cross if homo_cross is not None else '不存在',
                            min_bw_vs_both_GBps=both if both is not None else '不存在',
                            range_GBps='(0,9830]',scope='ideal-overlap estimate only; not calibrated system win'))
                    if batch=='all':
                        logratios=np.array([[math.log(a['Tc_ms']/r['Tc_ms']) for a,r in zip(ref,rs)] for ref in refs])
                        sample=rng.integers(0,len(rs),size=(2000,len(rs)))
                        reductions=1-np.exp(-logratios[:,sample].mean(axis=2))
                        means=1-np.exp(-logratios.mean(axis=1))
                        lower=np.quantile(reductions,.025,axis=1)
                        wins.append(dict(pn_cap=cap,mode=mode,design=name,
                            reduction_vs_single_pct=100*means[0],reduction_vs_homo_pct=100*min(means[1:]),
                            lower95_vs_single_pct=100*lower[0],lower95_vs_homo_pct=100*float(np.quantile(np.min(reductions[1:],axis=0),.025)),
                            enters_calibration=bool(min(means)>=.05),
                            numerical_10pct_5pctCI=bool(min(means)>=.10 and min(lower)>=.05),
                            calibrated=False,verdict='未校准，不宣布架构胜出'))
                        fixed=[ix[mode,f'pn{cap}_single_fixed',r['window_id']] for r in rs]
                        switch=[ix[mode,f'pn{cap}_single_pertask',r['window_id']] for r in rs]
                        h=gm(r['Tc_ms'] for r in rs);f=gm(r['Tc_ms'] for r in fixed);s=gm(r['Tc_ms'] for r in switch)
                        per.append(dict(pn_cap=cap,mode=mode,hetero=name,fixed_single_ms=f,pertask_single_ms=s,hetero_ms=h,
                            pertask_speedup=f/s,hetero_speedup=f/h,hetero_vs_pertask=s/h,
                            recovered_fraction=(f-s)/(f-h) if f>h else 'not_applicable'))
    write_csv(out/'crossover.csv',results);write_csv(out/'paired_confidence.csv',wins);write_csv(out/'pertask_comparison.csv',per)
    # System gates compare BOTH homogeneous organizations. A pure-compute
    # winner is not necessarily the strongest finite-memory baseline.
    if (ROOT/'E3/system_by_window.csv').exists():
        system=load_rows(ROOT/'E3/system_by_window.csv')
        sx={(r['design'],r['window_id'],int(r['bw_GBps'])):r for r in system}
        sysci=[]
        for cap in (256,512):
            for bw in (256,1229):
                for fam in ('hetero11','hetero31','hetero71'):
                    name=f'pn{cap}_{fam}'
                    for batch in (*BATCHES,'all'):
                        rs=[r for r in system if r['design']==name and r['bw_GBps']==bw and (batch=='all' or r['batch']==batch)]
                        ll=np.array([[math.log(sx[f'pn{cap}_{b}',r['window_id'],bw]['Tc_ms']/r['Tc_ms']) for r in rs] for b in ('single','homo2','homo4')])
                        ids=rng.integers(0,len(rs),size=(2000,len(rs)));rr=1-np.exp(-ll[:,ids].mean(axis=2))
                        reduction=1-np.exp(-ll.mean(axis=1))
                        lows=np.quantile(rr,.025,axis=1)
                        sysci.append(dict(design=name,pn_cap=cap,bw_GBps=bw,batch=batch,
                            **{f'reduction_vs_{b}_pct':100*reduction[j] for j,b in enumerate(('single','homo2','homo4'))},
                            **{f'lower95_vs_{b}_pct':100*lows[j] for j,b in enumerate(('single','homo2','homo4'))},
                            enters_calibration=bool(min(reduction)>=.05),
                            nominal_10pct_and_CI5pct=bool(min(reduction)>=.10 and min(lows)>=.05),
                            calibrated=False))
        write_csv(out/'system_paired_confidence.csv',sysci)
    text=['# 异构与单核按任务切换数据流',
          '所有硬件/策略在开发集选择后冻结。下表为135窗口GM；按任务策略根据Me和冻结成本规则选择OS/WS/IS，无留出集调参。切换控制开销在纯计算B忽略，在系统回放由任务绑定成本覆盖；没有物理实现验证。',
          table(['PN上限','模式','异构族','固定流单核ms','按任务单核ms','异构ms','异构/按任务单核加速','按任务收回收益比例'],
            [[r['pn_cap'],r['mode'],r['hetero'],f"{r['fixed_single_ms']:.6f}",f"{r['pertask_single_ms']:.6f}",f"{r['hetero_ms']:.6f}",f"{r['hetero_vs_pertask']:.4f}",r['recovered_fraction']] for r in per]),
          '收回比例=(固定流单核时间−按任务单核时间)/(固定流单核时间−异构时间)。分母≤0时不定义。大于1表示单核策略已超过该异构；负值表示按任务反而更慢。',
          '带宽曲线中的max(Tc,bytes/BW)忽略启动、争用与动态派工。Tc来自一种固定调度；它是该调度的理想重叠投影，不能冒充所有可能调度的全局下界。实际系统时间仅引用E3。']
    (out/'HETERO_VS_PERTASK_DATAFLOW.md').write_text('\n\n'.join(text)+'\n')
    plot(rows,sd)

def plot(rows,sd):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    xs=np.geomspace(64,9830,300)
    with PdfPages(ROOT/'E4/fig_speedup_vs_bw.pdf') as pdf:
        for cap in (256,512):
            for mode in ('B','A'):
                fig,axes=plt.subplots(1,3,figsize=(17,4.5),sharey=True)
                for ax,fam in zip(axes,('hetero11','hetero31','hetero71')):
                    for batch in BATCHES:
                        a=sorted([r for r in rows if r['mode']==mode and r['design']==f'pn{cap}_single' and r['batch']==batch],key=lambda r:r['window_id'])
                        b=sorted([r for r in rows if r['mode']==mode and r['design']==f'pn{cap}_{fam}' and r['batch']==batch],key=lambda r:r['window_id'])
                        ys=[gm(roof(a,x)/roof(b,x)) for x in xs]
                        ax.plot(xs,ys,label=f'B{batch}')
                    for red in (.05,.10):ax.axhline(1/(1-red),color='gray',ls='--',lw=.8)
                    ticks=(256,512,1229,2458,9830)
                    for x in ticks:ax.axvline(x,color='gray',ls=':',lw=.5)
                    ax.set_xscale('log');ax.set_title(fam);ax.set_xlabel('HBM GB/s');ax.grid(alpha=.2)
                    ax.set_xticks(ticks,['256\n1xHBM2','512\n2xHBM2','1229\n1xHBM3E','2458\n2xHBM3E','9830\n~8xHBM3E'],fontsize=7)
                axes[0].set_ylabel('Single / hetero latency (estimate)');axes[-1].legend(fontsize=7)
                fig.suptitle(f'PN <= {cap}, mode {mode}; dashed: 5% / 10% latency reduction; not measured system speedup')
                fig.tight_layout();pdf.savefig(fig)
                if cap==256 and mode=='B':fig.savefig(ROOT/'E4/fig_speedup_vs_bw.png',dpi=180)
                plt.close(fig)

if __name__=='__main__':run()
