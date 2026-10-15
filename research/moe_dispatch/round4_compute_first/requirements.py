"""E2 inverse provisioning, E3 matched-window roofline metrics."""
import argparse,json,math
from collections import defaultdict
from .common import ROOT,read_csv,write_csv,write_json,gm,table
from .config import BATCHES,BANDWIDTHS
from .evaluate import selected

def load_rows(path):
    rs=read_csv(path)
    for r in rs:
        for k,v in list(r.items()):
            if v=='': continue
            try:r[k]=float(v)
            except ValueError:pass
    return rs

def group(rs):
    out=defaultdict(list)
    for r in rs:
        out[r['mode'],r['design'],int(r['batch'])].append(r)
        out[r['mode'],r['design'],'all'].append(r)
    return out

def e2():
    rows=load_rows(ROOT/'E1/compute_by_window.csv'); groups=group(rows)
    index={(r['mode'],r['design'],r['window_id']):r for r in rows}
    sr=[];bw=[]
    for (mode,name,batch),rs in groups.items():
        rates=[r['hbm_bytes']/(r['Tc_ms']*1e6) for r in rs]
        balance=gm(rates)
        bw.append(dict(mode=mode,design=name,batch=batch,windows=len(rs),Tc_ms=gm(r['Tc_ms'] for r in rs),
            hbm_MiB=gm(r['hbm_bytes']/2**20 for r in rs),unique_MiB=gm(r['unique_bytes']/2**20 for r in rs),
            balance_GBps=balance,balance_unique_GBps=gm(r['unique_bytes']/(r['Tc_ms']*1e6) for r in rs),
            balance_unique_with_io_GBps=gm(r['unique_with_io_bytes']/(r['Tc_ms']*1e6) for r in rs),
            max_balance_GBps=max(rates),HBM2_stacks=math.ceil(balance/256),
            HBM3_stacks=math.ceil(balance/819),HBM3E_stacks=math.ceil(balance/1229),
            HBM3E_stacks_for_all_windows=math.ceil(max(rates)/1229),
            feasible='现实可行（仅带宽数量判据）' if balance<=8*1229 else '现实不可行（超过8堆栈）',
            all_windows_feasible=max(rates)<=8*1229))
        if mode!='B':continue
        a=[index['A',name,r['window_id']] for r in rs]
        rec=dict(design=name,batch=batch,windows=len(rs),
                 sram_total_peak_KiB=max(r['sram_total_peak'] for r in rs)/1024,
                 sram_total_GM_KiB=gm(r['sram_total_peak'] for r in rs)/1024,
                 sram_over_round3=max(r['sram_total_peak'] for r in rs)/2158592,
                 sram_GM_over_round3=gm(r['sram_total_peak'] for r in rs)/2158592,
                 Tc_modeA_over_modeB=gm(x['Tc_ms']/r['Tc_ms'] for x,r in zip(a,rs)),
                 hbm_bytes_modeA_over_modeB=gm(x['hbm_bytes']/r['hbm_bytes'] for x,r in zip(a,rs)))
        for k in ('W','X','acc','Z'):
            rec[k+'_KiB']=max(r['sram_peak_'+k] for r in rs)/1024
            rec[k+'_GM_KiB']=gm(r['sram_peak_'+k] for r in rs)/1024
        for k,den in [('W',1024),('X',384),('acc',192)]:
            for stat,key in [('peak','port_peak_'),('avg','port_avg_'),('issue_peak','port_issue_peak_')]:
                rec[f'port_{k}_{stat}_Bpc']=max(r[key+k] for r in rs)
                rec[f'port_{k}_{stat}_GM_Bpc']=gm(r[key+k] for r in rs)
            rec[f'port_{k}_over_round3']=rec[f'port_{k}_peak_Bpc']/den
        rec['port_over_round3']=max(rec[f'port_{k}_over_round3'] for k in ('W','X','acc'))
        sr.append(rec)
    write_csv(ROOT/'E2/sram_requirement.csv',sr);write_csv(ROOT/'E2/balance_bandwidth.csv',bw)
    write_json(ROOT/'E2/DEFINITIONS.json',dict(capacity='simultaneous task-reservation envelope, a sufficient provisioning policy, not minimum-SRAM proof',
        peak_ports='maximum simultaneous private phase-fluid demand envelope; issue_peak columns give unsmoothed issue interface demand',
        HBM_rates='decimal GB/s; 1 B/cycle at 1GHz =1GB/s; MiB=2**20B',
        feasibility='only requested <=8 HBM3E-stack bandwidth criterion, not packaging, capacity, area or power validation',
        nominal_stack_GBps={'HBM2_model':256,'HBM3':819,'HBM3E':1229},
        last_sweep_point=9830,exact_8_stack_rate=9832))

def relation(vals,higher=False):
    names=['a','b','c']; order=sorted(range(3),key=lambda j:vals[j],reverse=higher)
    out=names[order[0]]
    for l,r in zip(order,order[1:]):
        close=abs(vals[l]/vals[r]-1)<.01
        out+=('≈' if close else ('>' if higher else '<'))+names[r]
    return out

def e3():
    rows=load_rows(ROOT/'E1/compute_by_window.csv'); groups=group(rows)
    sys=load_rows(ROOT/'E3/system_by_window.csv')
    ix={(r['design'],r['window_id'],int(r['bw_GBps'])):r for r in sys}
    sd=selected()
    def attribution(s):
        d=sd[s['design']]['real_sram_design'];n=int(s['core_count']);t=s['Tc_ms']
        floors={'HBM':s['hbm_bytes']/(s['bw_GBps']*1e6),
                'matrix':max(s[f'core{j}_matrix_ms'] for j in range(n))}
        for k,field in [('W','w_banks'),('X','x_banks'),('acc','acc_banks')]:
            floors[k]=max(s[f'core{j}_port_avg_{k}']*t/(d[field][j]*16) for j in range(n))
        critical=max(floors,key=floors.get)
        return {**{f'sim_{k}_service_floor_ms':v for k,v in floors.items()},
                'dominant_resource_floor':critical,'sim_over_resource_floor':t/floors[critical]}
    result=[]; deviations=[]; wr=[]
    for (mode,name,batch),rs in groups.items():
        for bw in BANDWIDTHS:
            estimates=[max(r['Tc_ms'],r['hbm_bytes']/(bw*1e6)) for r in rs]
            sim=[ix.get((name,r['window_id'],bw)) for r in rs]
            have=all(s is not None for s in sim)
            limited=sum(r['Tc_ms']>=r['hbm_bytes']/(bw*1e6) for r in rs)
            out=dict(mode=mode,bw_GBps=bw,design=name,batch=batch,windows=len(rs),
                Tc_ms=gm(r['Tc_ms'] for r in rs),fetch_ms=gm(r['hbm_bytes']/(bw*1e6) for r in rs),
                est_layer_ms=gm(estimates),eq_fetch_GBps=gm(r['hbm_bytes']/(t*1e6) for r,t in zip(rs,estimates)),
                eq_GFLOPs=gm(r['flops']/(t*1e6) for r,t in zip(rs,estimates)),
                bound_by='计算' if limited==len(rs) else '取数' if limited==0 else '混合',compute_bound_windows=limited,
                sim_layer_ms=gm(s['Tc_ms'] for s in sim) if have else '',
                sim_over_est=gm(s['Tc_ms']/t for s,t in zip(sim,estimates)) if have else '',
                sim_eq_fetch_GBps=gm(s['hbm_bytes']/(s['Tc_ms']*1e6) for s in sim) if have else '',
                sim_eq_GFLOPs=gm(s['flops']/(s['Tc_ms']*1e6) for s in sim) if have else '',
                sim_uses_real_sram=True if have else '',
                estimate_scope='ideal overlap bound for projected traffic/schedule; not measured E2E')
            result.append(out)
            if batch=='all':
                for r,t,s in zip(rs,estimates,sim):
                    v=dict(mode=mode,design=name,window_id=r['window_id'],batch=int(r['batch']),bw_GBps=bw,
                        Tc_ms=r['Tc_ms'],hbm_bytes=r['hbm_bytes'],est_layer_ms=t,
                        sim_layer_ms=s['Tc_ms'] if s else '',sim_over_est=s['Tc_ms']/t if s else '')
                    wr.append(v)
                    if s and abs(s['Tc_ms']/t-1)>.05:
                        reasons=[]
                        traffic=s['hbm_bytes']/r['hbm_bytes']
                        if traffic>1.001:reasons.append('finite SRAM / changed assignment increases weight rereads or spills')
                        if mode=='B':reasons.append('real SRAM W/X/acc ports replace infinite ports')
                        reasons.append('cold HBM latency, finite in-flight tile slots, phase dependencies and fixed online admission')
                        if s['Tc_ms']<t:reasons.append('finite-bandwidth dispatch changes ownership; projected schedule is not a universal optimal lower bound')
                        deviations.append(dict(**v,sim_hbm_over_est_hbm=traffic,
                                               **attribution(s),reason='; '.join(reasons)))
    write_csv(ROOT/'E3/four_metrics.csv',result);write_csv(ROOT/'E3/four_metrics_by_window.csv',wr)
    write_csv(ROOT/'E3/deviations_over_5pct.csv',deviations)
    md=['# 第四轮四项指标：逐窗口计算后取几何平均',
        '单位：带宽 GB/s、时间 ms、有效算力 GFLOP/s。a=开发集选定单核，b=开发集较优同构（2/4核），c=开发集较优异构。≈ 表示差距小于1%。',
        '四项指标使用理想重叠估计时间；256/1229 的实际系统时间另列。模式B的系统列也统一使用真实SRAM，不能称为无限SRAM实测。']
    sd=selected()
    for cap in (256,512):
        names=[f'pn{cap}_single',min((f'pn{cap}_homo2',f'pn{cap}_homo4'),key=lambda k:sd[k]['dev_GM_cycles']),
               min((f'pn{cap}_hetero11',f'pn{cap}_hetero31',f'pn{cap}_hetero71'),key=lambda k:sd[k]['dev_GM_cycles'])]
        md.append(f'\nPN≤{cap}：'+', '.join(f'{a}={n}' for a,n in zip('abc',names)))
        for mode in ('B','A'):
            for bw in BANDWIDTHS:
                md.append(f'\n**模式{mode}，{bw} GB/s**')
                tr=[]
                for batch in (*BATCHES,'all'):
                    rr=[next(r for r in result if r['mode']==mode and r['bw_GBps']==bw and r['design']==name and r['batch']==batch) for name in names]
                    cells=[' / '.join(f'{r[k]:.5f}' for r in rr) for k in ('eq_fetch_GBps','fetch_ms','eq_GFLOPs','Tc_ms')]
                    rel='；'.join(label+':'+relation([r[k] for r in rr],higher) for label,k,higher in
                        [('BW','eq_fetch_GBps',True),('Fetch','fetch_ms',False),('FLOPs','eq_GFLOPs',True),('Compute','Tc_ms',False)])
                    actual=' / '.join(f"{r['sim_layer_ms']:.5f}" for r in rr) if bw in (256,1229) else '未跑系统；仅估计'
                    tr.append([batch,*cells,'BW a<b≈c；Fetch a≈b≈c；FLOPs a<b<c；Compute a>b>c（旧假设）',rel,actual])
                md.append(table(['Batch','Equivalent Fetch BW a/b/c','Fetch time a/b/c','Equivalent FLOPs a/b/c','Compute time a/b/c','原预期（未作为结果）','实际估计关系','完整系统时间 a/b/c'],tr))
    (ROOT/'E3/four_metrics_ZH.md').write_text('\n\n'.join(md)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['E2','E3']);a=ap.parse_args()
    (e2 if a.stage=='E2' else e3)()
