"""Generate the numerical Chinese report from delivered CSVs, no pasted scores."""
import json
from .common import ROOT,table,write_json
from .requirements import load_rows
from .evaluate import selected

def run():
    sd=selected();main=load_rows(ROOT/'E1/compute_main_table.csv');bw=load_rows(ROOT/'E2/balance_bandwidth.csv')
    sr=load_rows(ROOT/'E2/sram_requirement.csv');four=load_rows(ROOT/'E3/four_metrics.csv')
    cross=load_rows(ROOT/'E4/crossover.csv');per=load_rows(ROOT/'E4/pertask_comparison.csv')
    ci=load_rows(ROOT/'E4/paired_confidence.csv');deviations=load_rows(ROOT/'E3/deviations_over_5pct.csv')
    def find(rs,**kv):return next(r for r in rs if all(r[k]==v for k,v in kv.items()))
    fmt=lambda x:f'{float(x):.6f}'
    cap=256
    names=[f'pn{cap}_single',min((f'pn{cap}_homo2',f'pn{cap}_homo4'),key=lambda k:sd[k]['dev_GM_cycles']),
           min((f'pn{cap}_hetero11',f'pn{cap}_hetero31',f'pn{cap}_hetero71'),key=lambda k:sd[k]['dev_GM_cycles'])]
    refs=[find(main,mode='B',design=n,batch='all') for n in names]
    needs=[find(bw,mode='B',design=n,batch='all') for n in names]
    stores=[find(sr,design=n,batch='all') for n in names]
    pp=find(per,pn_cap=256,mode='B',hetero=names[2])
    q1='；'.join(f"{n}: {r['geometry']} / {r['flows']}，{fmt(r['Tc_ms'])} ms" for n,r in zip(names,refs))
    q1+=f"。异构/单核延迟比 {refs[2]['ratio_vs_single']:.5f}；异构/最佳同构 {refs[2]['ratio_vs_homo']:.5f}。预算65,536，已按论文纠正。"
    q2='；'.join(f"{n}: 平衡带宽GM {r['balance_GBps']/1000:.3f} TB/s，容量峰值 {s['sram_total_peak_KiB']/1024:.2f} MiB（{s['sram_over_round3']:.2f}×），W/X/Acc端口峰值 {s['port_W_peak_Bpc']:.0f}/{s['port_X_peak_Bpc']:.0f}/{s['port_acc_peak_Bpc']:.0f} B/cycle；{r['feasible']}" for n,r,s in zip(names,needs,stores))
    q3=[]
    system_names=[f'pn{cap}_single',f'pn{cap}_homo2',f'pn{cap}_homo4',names[2]]
    for b in (256,1229):
        rr=[find(four,mode='A',bw_GBps=b,design=n,batch='all') for n in system_names]
        q3.append(f"{b} GB/s：单核/同构二核/同构四核/异构完整系统时间="+'/'.join(fmt(r['sim_layer_ms']) for r in rr)+' ms；四项指标详见本报告下表（含B/A）。')
    q4='；'.join(f"模式{m}、降延迟{int(t)}%："+str(find(cross,pn_cap=256,mode=m,design=names[2],batch='all',threshold_latency_reduction_pct=t)['min_bw_vs_both_GBps']) for m in ('B','A') for t in (5,10))+'（0–9,830 GB/s范围内，同时对最优单核/同构）'
    q5=f"固定流单核 {fmt(pp['fixed_single_ms'])} ms；按任务单核 {fmt(pp['pertask_single_ms'])} ms；异构 {fmt(pp['hetero_ms'])} ms；按任务相对固定流加速 {pp['pertask_speedup']:.4f}×，按任务单核收回异构相对固定流收益比例 {pp['recovered_fraction']}。"
    doc=['# 第四轮结果：纯计算DSE与供数需求反推',
         '**主口径：65,536乘法器，BF16/FP32，PN≤256，135个留出窗口GM。模式B假设HBM带宽与SRAM容量/端口均无限，不是系统加速。**',
         table(['第一页问题','数值回答'],[[1,q1],[2,q2],[3,' '.join(q3)],[4,q4],[5,q5]]),
         '**结论边界：**开发集选定的硬件全部冻结后回放。未完成原生Rust/RTL datapath校准或PPA综合，不宣布架构胜出。任何“计算更快”只对声明的模型与空间成立。',
         '## 1. 选择、尺寸及留出纯计算结果']
    tr=[]
    for c in (256,512):
        for f in ('single','homo2','homo4','hetero11','hetero31','hetero71'):
            r=find(main,mode='B',design=f'pn{c}_{f}',batch='all')
            tr.append([c,f,r['geometry'],r['flows'],fmt(r['Tc_ms']),f"{r['eq_GFLOPs']:.2f}",f"{r['mac_util_pct']:.2f}",f"{1/r['ratio_vs_single']:.5f}",f"{1/r['ratio_vs_homo']:.5f}"])
    doc.append(table(['PN上限','组织','各核PM×PN×PK','数据流','Tc ms','有效GFLOP/s','有效MAC利用率%','相对单核加速','相对较优同构加速'],tr))
    doc.append('2:1、5:1没有合法精确整数配比；1:1异构是同MAC不同几何。PN512对所有核同时开放。总枚举数量及重复校验见E1/search_receipt.json。')
    tr=[]
    for batch in (2,4,8,16,64,96,128,'all'):
        rr=[find(main,mode='B',design=n,batch=batch) for n in names]
        tr.append([batch,*[fmt(r['Tc_ms']) for r in rr],f"{rr[0]['Tc_ms']/rr[2]['Tc_ms']:.4f}",f"{rr[1]['Tc_ms']/rr[2]['Tc_ms']:.4f}"])
    doc.append(table(['Batch','单核ms','最佳同构ms','最佳异构ms','单/异加速','同/异加速'],tr))
    doc+=['## 2. 无限SRAM假设的代价',
          '容量列是同时保留任务缓冲的需求上界，不是最小SRAM证明。端口峰值是phase-fluid供数包络；E2另列发射接口burst需求。真实SRAM预算约2.06MiB，W/X/Acc总端口1024/384/192 B/cycle。',
          table(['设计','容量峰值MiB','容量倍数','W/X/Acc端口峰值B/cycle','最大端口倍数','A/B计算时间','A/B HBM字节','所需HBM GM TB/s','HBM3E堆栈GM判据'],
                [[n,f"{s['sram_total_peak_KiB']/1024:.3f}",f"{s['sram_over_round3']:.2f}",
                  '/'.join(f"{s[f'port_{k}_peak_Bpc']:.0f}" for k in ('W','X','acc')),f"{s['port_over_round3']:.2f}",
                  f"{s['Tc_modeA_over_modeB']:.2f}",f"{s['hbm_bytes_modeA_over_modeB']:.2f}",f"{r['balance_GBps']/1000:.3f}",int(r['HBM3E_stacks'])]
                 for n,s,r in zip(names,stores,needs)])]
    doc.append('## 3. 四项指标与完整系统回放')
    doc.append('下表同时列出同构二核和四核：纯计算最优不代表有限供数下也最优。“取数BW/有效FLOPs”使用理想重叠估计；完整系统列是有限容量/端口/HBM/fixed派工的独立事件仿真，两者不得混用。')
    doc.append('模式A的Tc包含SRAM端口服务，因此A表中的“计算”表示片上服务一侧受限，不能据此称为MAC阵列计算受限；具体HBM/W/X/Acc瓶颈已在逐窗口偏差表量化。')
    tr=[]
    for mode in ('B','A'):
        for b in (256,1229):
            for n in system_names:
                r=find(four,mode=mode,bw_GBps=b,design=n,batch='all')
                tr.append([mode,b,n,fmt(r['Tc_ms']),fmt(r['fetch_ms']),f"{r['eq_fetch_GBps']:.2f}",f"{r['eq_GFLOPs']:.2f}",
                           fmt(r['est_layer_ms']),fmt(r['sim_layer_ms']),f"{r['sim_over_est']:.3f}",r['bound_by']])
    doc.append(table(['模式','HBM GB/s','设计','Compute ms','Fetch ms','Eq Fetch GB/s','Eq GFLOP/s','估计ms','系统ms','系统/估计','估计瓶颈'],tr))
    sysci=load_rows(ROOT/'E4/system_paired_confidence.csv')
    doc.append(table(['HBM GB/s','异构族','相对单核降延迟%','相对同构2核降延迟%','相对同构4核降延迟%','三基线均达5%'],
        [[r['bw_GBps'],r['design'],f"{r['reduction_vs_single_pct']:.3f}",f"{r['reduction_vs_homo2_pct']:.3f}",f"{r['reduction_vs_homo4_pct']:.3f}",r['enters_calibration']]
         for r in sysci if r['pn_cap']==256 and r['batch']=='all']))
    doc.append(f'共 {len(deviations)} 条逐窗口/估计模式/带宽记录的偏差绝对值超过5%，全列在 E3/deviations_over_5pct.csv。大偏差来自有限片上端口、容量重读、HBM启动/在途窗口及派工变化。A对照保持B的归属，E3会在线重分配，因此A的估计也不是对所有在线调度的全局最优下界。')
    doc.append('## 4. 带宽门槛、数据流替代收益与统计')
    doc.append(table(['模式','异构族','降延迟5%所需GB/s','降延迟10%所需GB/s'],
       [[m,f,find(cross,pn_cap=256,mode=m,design=f'pn256_{f}',batch='all',threshold_latency_reduction_pct=5)['min_bw_vs_both_GBps'],
              find(cross,pn_cap=256,mode=m,design=f'pn256_{f}',batch='all',threshold_latency_reduction_pct=10)['min_bw_vs_both_GBps']]
        for m in ('B','A') for f in ('hetero11','hetero31','hetero71')]))
    doc.append(q5)
    doc.append(table(['模式','异构族','相对单核降延迟%','相对同构降延迟%','95%下界：单/同','进入校准5%门槛','已校准'],
       [[r['mode'],r['design'],f"{r['reduction_vs_single_pct']:.3f}",f"{r['reduction_vs_homo_pct']:.3f}",
          f"{r['lower95_vs_single_pct']:.3f}/{r['lower95_vs_homo_pct']:.3f}",r['enters_calibration'],'否'] for r in ci if r['pn_cap']==256]))
    doc+=['## 5. 完整交付与限制',
          'E1：开发集全域搜索、冻结硬件、逐窗口计算/流量/每核存储端口/负载；E2：所有组织逐batch及all的带宽和存储反推；E3：5档带宽×2模式四指标，256/1229完整系统回放；E4：全部异构族逐batch交叉点、四页带宽曲线、单核按任务切换对照。',
          '所有DSE候选与所有选定设计的留出点重复两次并逐位核对。输入来源、运行环境、源码哈希、第二/三轮只读树校验及测试结果见PROVENANCE.json和VALIDATION.json。',
          '形状/数据流重配置可实现性、SRAM bank面积、路由扇出、BF16归约时序未综合。完整系统仿真仍是解析phase-fluid模型，不能替代Rust/native校准。固定容量分配未做额外搜索。按任务切换的面积和专门切换延迟未计价。HBM堆栈“现实可行”仅指任务书的≤8堆栈数量判据。']
    (ROOT/'REPORT_ZH.md').write_text('\n\n'.join(doc)+'\n')
    write_json(ROOT/'REPORT_NUMBERS.json',dict(primary_names=names,compute=refs,bandwidth=needs,sram=stores,pertask=pp))

if __name__=='__main__':run()
