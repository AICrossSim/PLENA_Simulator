"""E2 lower bounds and frozen-hardware headroom, not new-hardware selection."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from .common import (ROOT, OLD, inputs, read_csv, write_csv, write_json, gm,
                     digest, canonical, frozen_designs, metadata, table)
from .config import MODES, BATCHES, parameters
from .runtime import simulate
from .optimizer import universal_bound, unique_hbm_bytes
from ..round2.predictors import Predictor


FIELDS = ['set','window_id','batch','bw_GBps','unique_weight_MiB','fetch_floor_ms',
          'compute_lb_ms','vector_lb_ms','port_lb_ms','lb_ms','lb_kind',
          'b1_frozen_ms','b2_frozen_ms','hetero_frozen_ms']
SUMMARY_FIELDS = ['set','bw_GBps','batch','n_windows','max_gain_vs_b1_pct',
                  'max_gain_vs_b2_pct','gate5_reachable']


def frozen_reference(spec):
    mode, credits, name = spec
    ws = inputs()
    d = frozen_designs(mode, common_ws=True)[name]
    p = parameters(mode, credits=credits)
    outputs = []
    for _ in range(2):
        pred = Predictor('ours')
        dev = [simulate(w,d,p,dispatch='fixed_legacy',predictor=pred) for w in ws['development']]
        held = ([simulate(w,d,p,dispatch='fixed_legacy',predictor=pred) for w in ws['heldout']]
                if credits == 390 else [])
        outputs.append(dict(dev=dev,heldout=held))
    assert canonical(outputs[0]) == canonical(outputs[1])
    result = {('dev',w['id']):r['latency_ms'] for w,r in zip(ws['development'],outputs[0]['dev'])}
    if credits == 390:
        result.update({('heldout',w['id']):r['latency_ms'] for w,r in zip(ws['heldout'],outputs[0]['heldout'])})
    else:
        source = (OLD/'dispatch_fix/hbm256_sensitivity_20261008/per_window.csv' if credits == 520 else
                  OLD/'dispatch_fix/predictor_ablation_20261008/per_window.csv')
        rows = [r for r in read_csv(source) if r['onchip_mode']==mode and r['design']==name and r['predictor']=='ours']
        assert len(rows) == 135
        result.update({('heldout',r['window_id']):float(r['latency_ms']) for r in rows})
    receipt = dict(mode=mode,credits=credits,design=name,repeats=2,
                   result_digest=digest(outputs[0]),repeat_digest=digest(outputs[1]),
                   dev_fresh=True,heldout_fresh=credits==390,
                   heldout_reference='fresh' if credits==390 else str(source.relative_to(ROOT.parent)))
    return result, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs',type=int,default=4)
    args = parser.parse_args()
    ws = inputs()
    specs = [(mode,c,name) for mode in MODES for c in (256,390,520)
             for name in ('B1','B2','best_hetero')]
    references, receipts = {}, []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for spec,(values,receipt) in zip(specs,pool.map(frozen_reference,specs)):
            references[spec]=values;receipts.append(receipt)
            print(dict(reference=spec,windows=len(values)),flush=True)
    violations = []
    mode_rows = {}
    for mode in MODES:
        rows = []
        for credits in (256,390,520):
            p = parameters(mode,credits=credits)
            for set_name, key in (('dev','development'),('heldout','heldout')):
                for w in ws[key]:
                    bound = universal_bound(w,p)
                    terms = bound['terms']
                    refs = {n:references[mode,credits,n][set_name,w['id']] for n in ('B1','B2','best_hetero')}
                    for name,ms in refs.items():
                        if ms*1e6+1e-6 < bound['lb_cycles']:
                            violations.append(dict(mode=mode,credits=credits,name=name,window=w['id'],ms=ms,lb=bound['lb_cycles']/1e6))
                    rows.append(dict(set=set_name,window_id=w['id'],batch=w['batch'],bw_GBps=p.hbm_bandwidth,
                        unique_weight_MiB=unique_hbm_bytes(w)/2**20,
                        fetch_floor_ms=terms['hbm_unique']/1e6,compute_lb_ms=terms['mac']/1e6,
                        vector_lb_ms=terms['vector_mandatory']/1e6,
                        port_lb_ms=max(terms['W_mandatory'],terms['X_mandatory'],terms['acc_mandatory'])/1e6,
                        lb_ms=bound['lb_cycles']/1e6,lb_kind=bound['binding_term'],
                        b1_frozen_ms=refs['B1'],b2_frozen_ms=refs['B2'],hetero_frozen_ms=refs['best_hetero']))
        summaries = []
        for set_name in ('dev','heldout'):
            for credits in (256,390,520):
                bw = parameters(mode,credits=credits).hbm_bandwidth
                for batch in (*BATCHES,'all'):
                    rs = [r for r in rows if r['set']==set_name and r['bw_GBps']==bw and (batch=='all' or r['batch']==batch)]
                    if not rs:
                        continue
                    a = 100*(1-gm(r['lb_ms']/r['b1_frozen_ms'] for r in rs))
                    b = 100*(1-gm(r['lb_ms']/r['b2_frozen_ms'] for r in rs))
                    summaries.append(dict(set=set_name,bw_GBps=bw,batch=batch,n_windows=len(rs),
                                          max_gain_vs_b1_pct=a,max_gain_vs_b2_pct=b,gate5_reachable=a>=5 and b>=5))
        destination = ROOT/'E2' if mode=='pipelined' else ROOT/'E2/port_tight'
        write_csv(destination/'bounds_by_window.csv',rows,FIELDS)
        write_csv(destination/'bounds_summary.csv',summaries,SUMMARY_FIELDS)
        mode_rows[mode]=(rows,summaries)
    write_json(ROOT/'E2/repeat_checks.json',receipts)
    write_json(ROOT/'E2/bound_checks.json',dict(violations=violations,violation_count=len(violations),
        checked_reference_points=sum(len(rows)*3 for rows,_ in mode_rows.values()),
        expanded_bound_scope='full-row GU/Down lifetime; optimistic simultaneous Z532KiB and retainedW532KiB; private/shared pool allowed'))
    assert not violations, violations[:5]
    lines=['# E2：先算下界，再判断搜索空间','',
           'BF16；18 开发＋135 留出窗口。各窗口先算比值，再取几何平均；单位 ms、GB/s、MiB。冻结硬件统一 WS、旧 fixed_legacy＋ours，开发暖机顺序与提交 7eb58061 一致。126/256 的留出值只读引用，192 全部窗口及三个点的开发窗口补跑两次。',
           '', '唯一字节与模型相同：每个被选专家和 Shared 的 Gate/Up/Down 各读一次，K 行按 32 B 请求打包。不把 PN×PK 片上补齐额外当成 HBM 字节。',
           '', '全局下界取 HBM、MAC、向量、W/X/累加必需端口，以及可证明有效的乐观行分块下界最大值。W 最少三次片上服务；向量工作量为 ΣMe(3F+2H)。端口占用相互重叠，下界取最大值而非相加。',
           '', '旧区域下界的最大 Z384KiB、最大 W40KiB 不再有效。新下界为每个任务同时免费授予 Z532KiB 与可保留 W532KiB；这超过真实共享总预算，因此只会低估重读。现模型仍要求一组完整行执行完 Gate/Up 后再 Down，故每个额外行分块需再次经过未保留权重。此论证不适用于未来跨 N 融合／更改中间值生命周期。','']
    for mode,(_,summary) in mode_rows.items():
        lines += [f'## {mode} 留出集最大可能收益','',
                  table(['GB/s','Batch','窗口数','相对冻结 B1 上限 %','相对冻结 B2 上限 %','能否到 5%'],
                        [[f"{r['bw_GBps']:.2f}",r['batch'],r['n_windows'],f"{r['max_gain_vs_b1_pct']:.4f}",
                          f"{r['max_gain_vs_b2_pct']:.4f}",r['gate5_reachable']]
                         for r in summary if r['set']=='heldout']),'']
    lines += ['本表是相对旧冻结基线的最大可能收益。开发集目标改善不保证留出集也改善；只有最终基线在相同窗口、相同控制协议下确实更快时，相对它的上限才会缩小。E4 的证明与最终门槛须分别用其同协议的新基线判断。若某 batch 可达 5%，不能据总体值宣称每个 batch 都不可能；实际胜出仍需同时跨过 B1、B2 的总体配对门槛。',
              '',f'已检查冻结参考点 {sum(len(rows)*3 for rows,_ in mode_rows.values())} 个，下界违例 {len(violations)}。全部新设计的逐窗口检查由最终验收追加。']
    (ROOT/'E2/BOUNDS.md').write_text('\n'.join(lines)+'\n')
    write_json(ROOT/'E2/METADATA.json',metadata(dict(references=receipts,violations=0)))
    plot(mode_rows['pipelined'][1])


def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    for batch in (*BATCHES,'all'):
        rs=sorted([r for r in summary if r['set']=='heldout' and r['batch']==batch],key=lambda r:r['bw_GBps'])
        ax.plot([r['bw_GBps'] for r in rs],[r['max_gain_vs_b1_pct'] for r in rs],marker='o',
                linewidth=2.5 if batch=='all' else 1,label='All (paired GM)' if batch=='all' else f'B{batch}')
    ax.axhline(5,color='black',linestyle='--',label='5% calibration gate')
    ax.set(xlabel='Analytical shared HBM service cap (GB/s)',ylabel='Maximum possible reduction vs frozen B1 (%)')
    ax.grid(alpha=.25);ax.legend(ncol=3,fontsize=8);fig.tight_layout()
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/fig_headroom_vs_bw.pdf');plt.close(fig)


if __name__=='__main__':
    main()
