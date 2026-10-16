"""Nine requested tables, both caps, both timing contracts and both overheads."""
import math
from collections import defaultdict
from .common import CFG,ROOT,read_json,save_json,save_csv,gm,table_files,markdown
from .model import Core,references,geometry,task_cost
from .scheduler import greedy,optimal
from .metrics import gm_row
from .legacy import PUBLISHED

ORG={'single':'单核最优','homo':'同构双核最优','hetero':'异形双核最优'}
REF={'PLENA_32x32x64_reference':'PLENA 32×32×64几何参照','old_single':'旧单核',
     'old_homo':'旧同构双核','old_hetero':'旧异构双核','old_stream_homo':'旧纯流式同构'}
TITLES={1:'硬件配置汇总',2:'各batch计算性能',3:'HBM取数',4:'SRAM取数',
        5:'每个batch单独最优shape',6:'贪心与精确最优调度',7:'Top-10候选',
        8:'MemExplorer参考带宽',9:'与旧结果的一致性检查'}

def memory_reference():
    return [dict(存储='SRAM（1 die）',每单元容量='256 MB',每单元带宽_TBps=4,
                 来源='MemExplorer v2, Table II, PDF p4'),
            dict(存储='3D堆叠SRAM（1层）',每单元容量='1 GB',每单元带宽_TBps=8,
                 来源='MemExplorer v2, Table II, PDF p4'),
            dict(存储='HBM3E 8H（1 stack）',每单元容量='24 GB',每单元带宽_TBps=1,
                 来源='MemExplorer v2, Table II, PDF p4'),
            dict(存储='HBM4 12H（1 stack）',每单元容量='36 GB',每单元带宽_TBps=2,
                 来源='MemExplorer v2, Table II, PDF p4')]

def scenario_tables(model,overhead,cap,ws,legacy):
    directory=ROOT/'results'/f'{model}_overhead{overhead}'
    data=read_json(directory/'all_candidates.json');bygeo={r['geometry']:r for r in data}
    sels=read_json(directory/'selected.json')[str(cap)]
    chosen={o:bygeo[v['geometry']] for o,v in sels.items()}
    bybatch={o:{r['batch']:r for r in x['by_batch']} for o,x in chosen.items()}
    for o,x in chosen.items():bybatch[o]['GM']=gm_row(x['by_batch'])
    tables={i:[] for i in range(1,10)}
    configs=[(ORG[o],tuple(Core(**c) for c in chosen[o]['cores']),chosen[o]) for o in ORG]
    configs += [(REF[label],cs,bygeo[geometry(cs)]) for label,cs in references()]
    for label,cs,x in configs:
        tables[1].append(dict(组织=label,核1形状=cs[0].shape,核2形状=cs[1].shape if len(cs)>1 else '—',
          乘法器分配='+'.join(str(c.macs) for c in cs),峰值_TFLOPS=2*CFG['total_macs']*CFG['clock_hz']/1e12,
          SRAM_Peak_Feed_TBps=sum(c.peak_feed for c in cs),七batch_GM_Compute_us=x['gm_compute_us'],
          当前cap是否合法=all(c.pn<=cap for c in cs),参与本组织搜索=not label.startswith('旧') and not label.startswith('PLENA')))
    for b in list(CFG['batches'])+['GM']:
        for o in ORG:
            r=bybatch[o][b]
            tables[2].append(dict(Batch=b,组织=ORG[o],Compute_Time_us=r['compute_us'],Equivalent_FLOPs_GFLOP=r['equivalent_GFLOP'],
              Equivalent_Throughput_TFLOPS=r['throughput_TFLOPS'],MAC_Utilization_pct=r['utilization_pct'],
              相对单核加速比=bybatch['single'][b]['compute_us']/r['compute_us'],
              相对同构加速比=bybatch['homo'][b]['compute_us']/r['compute_us']))
            if b=='GM':continue
            tables[3].append(dict(Batch=b,组织=ORG[o],窗口数=r['windows'],Active_Experts=r['active_experts'],
              Weight_Bytes_MB=r['weight_MB'],Compute_Time_us=r['compute_us'],HBM_Equivalent_Fetch_BW_TBps=r['hbm_req_TBps'],
              HBM_Fetch_Time_us=r['hbm_fetch_us'],Roofline_Latency_us=r['roofline_us'],Bound=r['bound'],
              compute_bound窗口数=r['compute_bound_windows']))
            tables[4].append(dict(Batch=b,组织=ORG[o],SRAM_Peak_Feed_TBps=r['sram_peak_TBps'],
              SRAM_Bytes_per_Layer_MB=r['sram_MB'],SRAM_Equivalent_Fetch_BW_TBps=r['sram_req_TBps'],
              SRAM_Fetch_Time_us=r['sram_fetch_us'],SRAM_Die_Equivalent=r['sram_die_equivalent'],
              非overhead周期全部满宽读取上界_MB=r['sram_fullslot_upper_MB'],
              非overhead周期全部满宽读取上界_TBps=r['sram_fullslot_upper_TBps'],
              同时考虑HBM与SRAM的理想下界_us=r['hierarchy_roofline_us']))
    for o in ORG:
        candidates=[x for x in data if cap in x['caps'] and x['org']==o]
        for b in CFG['batches']:
            best=min(candidates,key=lambda x:(next(r['compute_us'] for r in x['by_batch'] if r['batch']==b),x['geometry']))
            t=next(r['compute_us'] for r in best['by_batch'] if r['batch']==b)
            tables[5].append(dict(Batch=b,组织=ORG[o],该batch最优形状=best['geometry'],该形状_Compute_us=t,
                                 七batch最优形状_Compute_us=bybatch[o][b]['compute_us'],
                                 损失_pct=(bybatch[o][b]['compute_us']/t-1)*100))
        top=sorted(candidates,key=lambda x:(x['gm_compute_us'],x['geometry']))[:10]
        for rank,x in enumerate(top,1):
            tables[7].append(dict(组织=ORG[o],排名=rank,形状=x['geometry'],七batch_GM_Compute_us=x['gm_compute_us'],
                比本组织最优慢_pct=(x['gm_compute_us']/top[0]['gm_compute_us']-1)*100,
                **{f'B{r["batch"]}_us':r['compute_us'] for r in x['by_batch']}))
    greedy_records=[]
    for o in ('homo','hetero'):
        cs=tuple(Core(**c) for c in chosen[o]['cores']);means=[]
        for b in CFG['batches']:
            values=[]
            for w in ws:
                if w['batch']!=b:continue
                p=[[task_cost(c,t['m'],model,overhead)['cycles'] for c in cs] for t in w['tasks']]
                a=greedy(p);z=greedy(p);assert a==z
                exact=optimal(p);assert a['cycles']>=exact['cycles']
                values.append(a['cycles']/1000)
                greedy_records.append(dict(window=w['id'],batch=b,org=o,geometry=geometry(cs),
                    greedy_cycles=a['cycles'],optimal_cycles=exact['cycles'],owner=a['owner']))
            g=math.fsum(values)/len(values);t=bybatch[o][b]['compute_us'];means.append(g)
            tables[6].append(dict(Batch=b,组织=ORG[o],最优调度_Compute_us=t,贪心_Compute_us=g,差距_pct=(g/t-1)*100))
        t=bybatch[o]['GM']['compute_us'];g=gm(means)
        tables[6].append(dict(Batch='GM',组织=ORG[o],最优调度_Compute_us=t,贪心_Compute_us=g,差距_pct=(g/t-1)*100))
    tables[8]=memory_reference()
    for label,cs in references()[1:]:
        for idx,b in enumerate(CFG['batches']):
            r=next(x for x in bygeo[geometry(cs)]['by_batch'] if x['batch']==b)
            old=PUBLISHED.get(label,[None]*7)[idx]
            attrib=next(x for x in legacy if x['model']==model and x['label']==label and int(x['batch'])==b)
            tables[9].append(dict(Batch=b,参照=REF[label],形状=geometry(cs),旧值_us=old*1000 if old else '未提供',
                新值_算术均值_us=r['compute_us'],偏差_pct=(r['compute_us']/(old*1000)-1)*100 if old else 'N/A',
                旧代码重放_GM_us=float(attrib['legacy_replay_GM_us']),
                旧代码重放_AM_us=float(attrib['legacy_runtime_us']),
                旧成本_最优调度_AM_us=float(attrib['legacy_cost_optimal_us']),
                新成本_合并Shared_AM_us=float(attrib['new_cost_merged_shared_us']),
                新成本_双Shared_AM_us=float(attrib['new_cost_two_shared_us']),
                归因列额外overhead=0,当前cap是否合法=all(c.pn<=cap for c in cs)))
    out=directory/f'cap{cap}';out.mkdir(exist_ok=True)
    for i,rows in tables.items():table_files(out,f'table{i}',rows)
    save_json(out/'greedy_by_window.json.gz',greedy_records)
    return tables,chosen,bybatch

def all_reports():
    import csv
    ws=read_json(ROOT/'inputs/windows.json.gz')
    with (ROOT/'results/legacy_attribution.csv').open() as f:legacy=list(csv.DictReader(f))
    all_tables={};selection=[]
    for model in CFG['models']:
        for overhead in CFG['overheads_per_gemm']:
            for cap in CFG['pn_caps']:
                tables,chosen,bybatch=scenario_tables(model,overhead,cap,ws,legacy)
                all_tables[(model,overhead,cap)]=tables
                for o,x in chosen.items():
                    selection.append(dict(model=model,overhead=overhead,cap=cap,org=o,geometry=x['geometry'],
                        GM_compute_us=x['gm_compute_us'],speedup_vs_single=chosen['single']['gm_compute_us']/x['gm_compute_us'],
                        speedup_vs_homo=chosen['homo']['gm_compute_us']/x['gm_compute_us'],
                        GM_required_HBM_TBps=gm([r['hbm_req_TBps'] for r in x['by_batch']]),
                        GM_required_SRAM_TBps=gm([r['sram_req_TBps'] for r in x['by_batch']]),
                        SRAM_peak_TBps=sum(Core(**c).peak_feed for c in x['cores'])))
    table_files(ROOT/'results','sensitivity_summary',selection)
    write_document(all_tables,selection)
    print('All nine tables written for both caps, both timing contracts, both overheads.',flush=True)

def write_document(tables,selection):
    def selected(model,cap,org,overhead=0):
        return next(r for r in selection if (r['model'],r['cap'],r['org'],r['overhead'])==(model,cap,org,overhead))
    a=selected('plena_os',256,'hetero');h=selected('plena_os',256,'homo');s=selected('plena_os',256,'single')
    f=selected('issue_formula',256,'hetero');fh=selected('issue_formula',512,'homo');fp=selected('issue_formula',512,'hetero')
    lines=['# 64K乘法器：单核／同构双核／异形双核穷举与访存反推','',
      '## 输入、耗时函数和统计口径','',
      '- 原始trace清单：`research/moe_dispatch/round2/results/E0/frozen_inputs.json`；实际JSON路径和SHA256见`../inputs/manifest.json`。原始格式为`workloads[]`，每窗含`tokens[].routes[]`和`experts[].Me`，两者逐一核对，全部满足ΣMe=6B。原始副本与标准化输入均已归档。',
      '- **135是全部窗口数，不是每个batch135个。** B2/B4/B8/B16各27窗，B64/B96/B128各9窗；没有复制窗口、重新裁剪或构造随机分布。',
      '- B2–B16：BFCL／GPQA／SWE各数据集，层2／13／26，decode步3／7／11；沿用每窗保存的B个真实请求token和顺序。B64／96／128：每个数据集×层一窗，由16个原始decode token，加同一请求连续prefill前缀48／80／112 token组成，decode步7。**大batch是混合窗口，不能称为纯decode批次。** 具体请求ID、时间步和prefill位置见逐窗清单。',
      '- 本次按用户要求直接在历史135窗上穷举与报告，没有另行开发/测试划分。因此“最优”只表示这批已知窗口和声明模型上的穷举最优，不是独立留出泛化成绩。',
      '- 本地HF配置已核对：hidden=2048，intermediate=1408，64选6，n_shared_experts=2。历史F2816合并Shared替换为两个F1408、M=B的独立完整任务；权重字节和有效MAC总量不变，两份Down结果需相加，按本任务要求不计向量与合并耗时。',
      '- **主表优先复用第五轮`model.projection(..., paired=False, flow="OS")`**：三次独立Gate/Up/Down，保留8个OS输出上下文、归约与累加递推；G映射为PK，是真实每次发射的K宽度。每GEMM另加参数overhead。不是RTL校准的硬件实测。',
      '- **完整附表严格执行任务书ceil公式**：三维tile数量乘积加overhead。该公式没有原OS函数的归约/累加等待。两种模型分别完整穷举，结果不能混成一张无标签的性能表。',
      '- 本次G=64为K方向乘积数；更早`scaling_g64`把G解释为MLEN/BLEN，MLEN=PM×G。这是旧表与本次结果不能直接等同的关键原因。32×32×64仅保留为要求的几何参照，本表不是重新复现PLENA原论文整机。',
      '- 所有核使用相同约束：PM任意合法正整数，PN为16倍数，G=64，总MAC严格65536；cap256共126候选（5单核、5同构、116异形），cap512共156候选（6、6、144），等预算异形仅去除交换核编号的重复。另保留旧异构的原核编号顺序用于参照。',
      '- 双核最优分配使用无时间截断的精确Pareto动态规划：以核0负载和从核1移走的负载为状态，合并等成本任务并仅删除严格支配状态；返回具体owner和闭合的最优值。每个点独立运行两遍完全一致。贪心只在选型后单独评价。',
      '- 每batch所有指标先逐窗算，再取算术平均；七batch综合取这些均值的等权GM。加速比是对应平均延迟的比，GM行是七batch对应指标的GM。不能用“平均字节÷平均耗时”替代平均带宽，不能用GFLOP/us直接当TFLOPS：需乘1000。',
      '- BF16操作数2B、1GHz；峰值131.072 TFLOPS；不计向量、Router、attention、派工、核间输出合并、容量/端口争用、DMA固定延迟与启动尾部；乘法器等量不等于面积等量。',
      '- HBM只算每层每个激活专家的三份BF16权重各读一次，忽略激活/输出HBM流量。HBM需求是平均必要带宽，不是能保证零气泡的充分条件。',
      '- SRAM主值按**实际发射次数×满读宽**计数，包含padding槽位，不把未发射的OS递推等待或额外overhead当读取。附列同时给出按所有非overhead周期满宽读取的上包络；在ceil公式下两者完全相同。未计累加器读写、输出写回和控制流量，故不是全部片上互连流量。',
      '- HBM和SRAM参考均4TB/s，采用十进制MB/TB。Roofline列按要求仅为max(计算,HBM读取)，并额外给出max(计算,HBM读取,SRAM读取)的理想下界，避免忽略SRAM瓶颈。SRAM Die Equivalent是带宽倍数，不是已验证的die数量、容量或PPA。',
      '- [MemExplorer v2 Table II，PDF第4页](https://arxiv.org/pdf/2604.16007v2#page=4)已核对：SRAM 256MB/4TB/s；3D SRAM 1GB/8TB/s；HBM3E 24GB/1TB/s；HBM4 36GB/2TB/s。它们是该论文的建模参考值。','',
      '## 结论（九句话）','',
      f'1. 复用OS时序、cap256、overhead0时，异形双核`{a["geometry"]}`最快，七batch GM为{a["GM_compute_us"]:.6f}µs，相对最优单核{a["speedup_vs_single"]:.4f}×、相对最优同构{a["speedup_vs_homo"]:.4f}×；并非每个batch都胜出。',
      f'2. 该异形的七batch GM平均必要HBM带宽为{a["GM_required_HBM_TBps"]:.3f}TB/s、SRAM发射供数带宽为{a["GM_required_SRAM_TBps"]:.3f}TB/s，峰值SRAM读宽为{a["SRAM_peak_TBps"]:.3f}TB/s。',
      f'3. 相对单核，这两个平均带宽分别为{a["GM_required_HBM_TBps"]/s["GM_required_HBM_TBps"]:.3f}倍、{a["GM_required_SRAM_TBps"]/s["GM_required_SRAM_TBps"]:.3f}倍；相对同构分别为{a["GM_required_HBM_TBps"]/h["GM_required_HBM_TBps"]:.3f}倍、{a["GM_required_SRAM_TBps"]/h["GM_required_SRAM_TBps"]:.3f}倍。',
      '4. 仅按4TB/s HBM筛选，复用OS模型中B2–B64全为memory-bound，B96单核/同构为窗口混合、异形仍为memory-bound，B128三者为compute-bound；这不代表同时只有4TB/s SRAM时也能计算受限。',
      '5. 把SRAM也限制到4TB/s后，各最优组织各batch的SRAM供数需求均超过额度，因此本报告不宣称实现了对应纯计算速度。',
      '6. 复用OS模型把cap放宽到512后，三种组织的最优几何均未改变，overhead16重新选型也未改变这些几何。',
      f'7. 严格ceil公式的cap256最优异形是`{f["geometry"]}`，GM为{f["GM_compute_us"]:.6f}µs，对同构{f["speedup_vs_homo"]:.4f}×；在4TB/s HBM下全部batch都为memory-bound。',
      f'8. 严格ceil公式放宽cap到512后，最优同构`{fh["geometry"]}`的GM为{fh["GM_compute_us"]:.6f}µs，优于最优异形的{fp["GM_compute_us"]:.6f}µs，说明“异构胜出”依赖几何边界与时序契约。',
      '9. 旧表差异通过逐步重放原调度、改最优分配、改周期函数、拆分Shared和GM→算术平均分别记录，没有调参拟合旧数；两套模型的完整数据都保留。','',
      '## 跨模型、列宽与overhead汇总','',markdown(selection),'']
    for model in CFG['models']:
        for overhead in CFG['overheads_per_gemm']:
            for cap in CFG['pn_caps']:
                lines += [f'## {model} / cap={cap} / overhead={overhead}周期每GEMM','']
                if model=='plena_os':
                    lines+=['复用OS累加时序；SRAM主值只计发射，附上包络计所有非overhead周期。','']
                else:lines+=['严格任务书ceil公式；每个非overhead计算周期对应一次满宽读取。','']
                for i in range(1,10):
                    lines += [f'### 表{i}：{TITLES[i]}','',markdown(tables[(model,overhead,cap)][i]),'']
                    if i==7:lines+=['单核/同构合法候选不足10个时，列出全部候选，不补假行。','']
                    if i==9:lines+=['旧纯流式同构没有给出历史数值，标“未提供”；重放结果是真实新运行，不冒充旧值。归因阶段统一overhead0，主“新值”列使用当前情景overhead。','']
    lines+=['## 精确调度与旧值归因明细','',
      '全部候选逐窗口owner、核负载、cycles及流量位于各情景`all_windows.json.gz`；全部候选目标及逐batch均值位于`all_candidates.json`；贪心逐窗口记录位于各cap目录。',
      '旧值重放及分阶段归因见`legacy_by_window.csv`和`legacy_attribution.csv`。原旧数据是合并Shared、旧G映射、有限队列在线调度、逐batch GM；当前主目标是两个Shared、G=K宽、精确最优分配、逐batch算术平均，因此差异不能解释为同一仿真条件下的性能回归。','']
    (ROOT/'results/results.md').write_text('\n'.join(lines))
    save_json(ROOT/'results/summary.json',selection)

if __name__=='__main__':all_reports()
