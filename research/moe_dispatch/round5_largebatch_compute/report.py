"""Generate transparent Chinese result tables and static scientific figures."""
import csv,json,math,gzip
from collections import defaultdict
import numpy as np
from .common import ROOT,BUDGETS,BATCHES,write_csv,write_json,gm,table

def read(p):
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return list(csv.DictReader(f))

METRICS=('ms','GFLOPs','utilization_pct','row_fill_pct','padding_pct','idle_MAC_pct','offline_gap_pct',
         'unique_weight_MiB','unique_weight_BW_TBs','service_imbalance','vector_excess_core_cycles')

def summarize(rows):
    out={}
    for metric in METRICS:
        # Times, throughput and utilization use paired GM. Zero-valued waste
        # counters use arithmetic means, with equal batch weighting for GM row.
        values=[float(r[metric]) for r in rows]
        out[metric]=gm(values) if metric in ('ms','GFLOPs','utilization_pct','row_fill_pct','unique_weight_MiB','unique_weight_BW_TBs','service_imbalance') else sum(values)/len(values)
    return out

def aggregation(rows):
    by={b:summarize([r for r in rows if int(r['batch'])==b]) for b in BATCHES}
    out={}
    for k in METRICS:
        out[k]=gm([by[b][k] for b in BATCHES]) if k in ('ms','GFLOPs','utilization_pct','row_fill_pct','unique_weight_MiB','unique_weight_BW_TBs','service_imbalance') else sum(by[b][k] for b in BATCHES)/len(BATCHES)
    by['GM']=out;return by

def paired_interval(base,candidate):
    base={r['window_id']:r for r in base};groups=sorted({r['group'] for r in candidate})
    logs={g:[] for g in groups};low=[];high=[]
    for r in candidate:
        b=base[r['window_id']];logs[r['group']].append(math.log(float(b['cycles'])/float(r['cycles'])))
        low.append(math.log(float(b['lower_cycles'])/float(r['cycles'])))
        high.append(math.log(float(b['cycles'])/float(r['lower_cycles'])))
    v=np.array([np.mean(logs[g]) for g in groups]);rng=np.random.default_rng(20261016)
    bs=np.exp(v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1))
    return dict(speedup=float(np.exp(v.mean())),ci95_low=float(np.quantile(bs,.025)),ci95_high=float(np.quantile(bs,.975)),
        scheduling_bound_speedup_low=math.exp(sum(low)/len(low)),scheduling_bound_speedup_high=math.exp(sum(high)/len(high)),cluster_groups=len(groups))

def build():
    from .runtime_fix import retrofit_all
    retrofit_all()
    from .robustness import build as robustness
    organization_stability=robustness()
    all_agg=[];main=[];online_main=[];frozen=[];comparison=[];runtime=[];controls=[];cert=[];work=[];cases={};dataset_results=[];stability=[];dual=[]
    for total in BUDGETS:
        for mode in ('matrix','full_scaled'):
            for gran in ('expert','shared64'):
                path=ROOT/'results'/f'{total}_{mode}_{gran}'
                sel=json.loads((path/'selected.json').read_text());rows=read(path/'holdout.csv.gz');by=defaultdict(list)
                for r in rows:by[r['family'],r['control'],r['scheduler']].append(r)
                aggs={k:aggregation(v) for k,v in by.items()}
                best_homo=min((f for f in sel if f.startswith('homo')),key=lambda f:sel[f]['dev_upper_cycles'])
                best_hetero=min((f for f in sel if f.startswith('hetero')),key=lambda f:sel[f]['dev_upper_cycles'])
                chosen={'single':'single','homogeneous':best_homo,'heterogeneous':best_hetero}
                cases[total,mode,gran]=(sel,by,aggs,chosen)
                for r in read(path/'development_bootstrap.csv'):
                    stability.append(dict(total_macs=total,mode=mode,granularity=gran,**r))
                for family,s in sel.items():
                    cert.append(dict(total_macs=total,mode=mode,granularity=gran,family=family,geometry=s['geometry'],
                        dev_upper_ms=s['dev_upper_cycles']/1e6,dev_global_lower_ms=s['dev_global_lower_cycles']/1e6,
                        dse_gap_pct=s['dse_gap_pct'],candidates=s['candidates']))
                    for key,a in aggs.items():
                        if key[0]!=family:continue
                        for b,q in a.items():all_agg.append(dict(total_macs=total,mode=mode,granularity=gran,family=family,geometry=s['geometry'],control=key[1],scheduler=key[2],batch=b,**q))
                for org,f in chosen.items():
                    s=sel[f];scope=[x for name,x in sel.items() if (name=='single' if org=='single' else name.startswith('homo' if org=='homogeneous' else 'hetero'))]
                    gap=100*(s['dev_upper_cycles']/min(x['dev_global_lower_cycles'] for x in scope)-1)
                    frozen.append(dict(total_macs=total,mode=mode,granularity=gran,organization=org,family=f,geometry=s['geometry'],dev_GM_ms=s['dev_upper_cycles']/1e6,dse_family_gap_pct=s['dse_gap_pct'],organization_global_gap_pct=gap))
                for b in (*BATCHES,'GM'):
                    single=aggs['single','primary','offline'][b]['ms'];h2=aggs['homo2','primary','offline'][b]['ms'];x2=aggs['hetero2','primary','offline'][b]['ms'];hb=aggs[best_homo,'primary','offline'][b]['ms']
                    dual.append(dict(total_macs=total,mode=mode,granularity=gran,batch=b,single_ms=single,homo2_ms=h2,hetero2_ms=x2,
                        hetero2_geometry=sel['hetero2']['geometry'],hetero2_vs_single=single/x2,hetero2_vs_homo2=h2/x2,hetero2_vs_best_homo=hb/x2))
                for b in (*BATCHES,'GM'):
                    q={org:aggs[f,'primary','offline'][b] for org,f in chosen.items()}
                    work_floor=gm([1056*bb*65536/total/1e6 for bb in BATCHES]) if b=='GM' else 1056*b*65536/total/1e6
                    main.append(dict(total_macs=total,mode=mode,granularity=gran,batch=b,
                        ideal_MAC_work_lower_ms=work_floor,
                        single_ms=q['single']['ms'],homogeneous_ms=q['homogeneous']['ms'],heterogeneous_ms=q['heterogeneous']['ms'],
                        speedup_vs_single=q['single']['ms']/q['heterogeneous']['ms'],speedup_vs_homogeneous=q['homogeneous']['ms']/q['heterogeneous']['ms'],
                        single_util_pct=q['single']['utilization_pct'],homogeneous_util_pct=q['homogeneous']['utilization_pct'],heterogeneous_util_pct=q['heterogeneous']['utilization_pct'],
                        single_row_fill_pct=q['single']['row_fill_pct'],homogeneous_row_fill_pct=q['homogeneous']['row_fill_pct'],heterogeneous_row_fill_pct=q['heterogeneous']['row_fill_pct']))
                    rt={org:aggs[f,'primary','runtime'][b] for org,f in chosen.items()}
                    online_main.append(dict(total_macs=total,mode=mode,granularity=gran,batch=b,
                        single_ms=rt['single']['ms'],homogeneous_ms=rt['homogeneous']['ms'],heterogeneous_ms=rt['heterogeneous']['ms'],
                        speedup_vs_single=rt['single']['ms']/rt['heterogeneous']['ms'],speedup_vs_homogeneous=rt['homogeneous']['ms']/rt['heterogeneous']['ms']))
                cand=by[best_hetero,'primary','offline']
                for dataset in ('bfcl','gpqa'):
                    ds={org:aggregation([r for r in by[f,'primary','offline'] if r['dataset']==dataset])['GM']['ms'] for org,f in chosen.items()}
                    dataset_results.append(dict(total_macs=total,mode=mode,granularity=gran,dataset=dataset,
                        single_ms=ds['single'],homogeneous_ms=ds['homogeneous'],heterogeneous_ms=ds['heterogeneous'],
                        speedup_vs_single=ds['single']/ds['heterogeneous'],speedup_vs_homogeneous=ds['homogeneous']/ds['heterogeneous']))
                for org,f in [('single','single'),('homogeneous',best_homo),('homo2','homo2')]:
                    comparison.append(dict(total_macs=total,mode=mode,granularity=gran,baseline=org,**paired_interval(by[f,'primary','offline'],cand)))
                for org,f in chosen.items():
                    a=aggs[f,'primary','offline']['GM'];rt=aggs[f,'primary','runtime']['GM'];npf=aggs[f,'primary','runtime_no_preconfig']['GM']
                    naive=aggs[f,'primary','runtime_naive_window']['GM']
                    runtime.append(dict(total_macs=total,mode=mode,granularity=gran,organization=org,family=f,naive_window_ms=naive['ms'],
                        offline_ms=a['ms'],runtime_ms=rt['ms'],no_preconfig_ms=npf['ms'],runtime_over_offline=rt['ms']/a['ms'],preconfig_speedup=npf['ms']/rt['ms']))
                    for control in ('uniform_OS','fixed_vector64'):
                        if (f,control,'offline') in aggs:
                            c=aggs[f,control,'offline']['GM'];controls.append(dict(total_macs=total,mode=mode,granularity=gran,organization=org,control=control,primary_ms=a['ms'],control_ms=c['ms'],slowdown=c['ms']/a['ms']))
                for r in cand:
                    for geom,v in json.loads(r['pool_metrics']).items():work.append(dict(total_macs=total,mode=mode,granularity=gran,window_id=r['window_id'],batch=r['batch'],geometry=geom,layer_cycles=int(r['cycles']),tail_wait_cycles=int(r['cycles'])-v['maximum'],**v))
    out=ROOT/'report';out.mkdir(exist_ok=True)
    for name,rows in [('all_aggregates',all_agg),('main',main),('online_main',online_main),('frozen_hardware',frozen),('paired_comparisons',comparison),('runtime',runtime),('controls',controls),('certificates',cert),('heterogeneous_pool_work',work),('dataset_sensitivity',dataset_results),('development_bootstrap',stability),('dual_core',dual)]:write_csv(out/f'{name}.csv',rows)
    write_json(out/'results.json',dict(main=main,online_main=online_main,frozen_hardware=frozen,comparisons=comparison,runtime=runtime,controls=controls))
    wb=defaultdict(list)
    for r in work:wb[r['total_macs'],r['mode'],r['granularity'],r['batch'],r['geometry']].append(r)
    balance=[]
    for (total,mode,gran,b,geom),rr in wb.items():
        mean=lambda k:sum(float(r[k]) for r in rr)/len(rr)
        balance.append(dict(total_macs=total,mode=mode,granularity=gran,batch=b,geometry=geom,
            cores=rr[0]['cores'],MACs=rr[0]['MACs'],work_pct=mean('work_pct'),idle_pct=mean('idle_pct'),
            last_core_finish_ms=mean('maximum')/1e6,tail_wait_ms=mean('tail_wait_cycles')/1e6))
    write_csv(out/'work_balance.csv',balance)
    text=[]
    text+=['# 第五轮：大Batch × 大算力，纯计算DSE完整结果','',
        '本报告的延迟是明确时序契约下的架构模型结果，不是RTL、ASIC或GPU实测。供数零等待，1 GHz，BF16输入/FP32累加。等乘法器，不等于等面积、等功耗。',
        '', '## 实验口径', '',
        '- 新采集原模型真实前向路由：BFCL长提示词与GPQA独立请求路由打包；每组2048个实际Token。Batch指MoE层的Token批量，不是B个独立解码请求。',
        '- 开发90窗口、留出90窗口；各含5个请求组 × 3层 × 6个Batch。开发/留出请求不重叠。Batch等权，层与请求组在Batch内等权；不沿用旧135窗口的80%小Batch权重。',
        '- PM={1,2,4,8,16,32,64}；PN=8..256步长8；PK={32,64,128,256,512}；每核至少4096且4096对齐。所有组织使用相同合法域。',
        '- 单核、同构2/4/8/16核、两种几何组成的异构2～16核全部枚举。所有组织最多16核；同构的其他核数不能整除严格预算与4096粒度。异构池的两种核数及所有合法整数算力切分也枚举。这里的“最优”只针对声明的有限空间，并保留调度及搜索误差界限。',
        '- 主模式允许每个专家从OS/WS/IS中选择计算时间最短的循环顺序，所有组织权限相同，不收切换成本；另有冻结硬件的统一OS对照。这是几何潜力模型，不是固定OS芯片的已实现性能。',
        '- 矩阵模式只计GEMM；完整专家模式另计SiLU/乘法/输出合并的向量服务，并假设阶段内理想流水重叠。向量吞吐按每65536乘法器64元素/周期同比扩大。Router、跨核互连和搬运不计。',
        '- 整专家派工与Shared按64 Token拆分分别在开发集选型，冻结到全部留出Batch。路由专家本轮不拆分。两种粒度是独立的实验条件，不允许在留出集逐Batch换硬件。',
        '- 离线给可行调度上界和严格下界；不能将存在非零间隙的点称为精确全局最优。运行时使用已知形状、初始LPT顺序、8项前瞻、每核Current+Next、每周期1个描述符，同一策略适用于所有组织。上下文配置2周期是显式假设。',
        '- 统计区间使用5个请求组做2000次聚类bootstrap，保留跨Batch/层相关性。只有5组独立聚类，区间只描述本样本，不代表跨模型泛化。','']
    text+=['所有加速比均为基线时间/异构时间，大于1表示异构更快。先逐窗口求整层makespan，再取Batch内GM，最后各Batch等权；不对“平均核时间”再取最大值。','']
    largest=max(BUDGETS)
    last=lambda rows,mode,gran:next(r for r in rows if r['total_macs']==largest and r['mode']==mode and r['granularity']==gran and r['batch']=='GM')
    potential=last(main,'matrix','expert');full=last(main,'full_scaled','expert');actual=last(online_main,'full_scaled','expert');shard=last(main,'matrix','shared64')
    text+=['## 本轮可成立的结论','',
        f"- 在{largest:,}乘法器下，离线整专家分配的异构矩阵加速比为最优同构的{potential['speedup_vs_homogeneous']:.4f}×；加入同比扩展的向量服务后为{full['speedup_vs_homogeneous']:.4f}×。这是声明模型下的几何潜力。",
        f"- 同一完整专家硬件改用当前在线准入规则后，异构/同构加速比为{actual['speedup_vs_homogeneous']:.4f}×：单核{actual['single_ms']:.6f}ms、同构{actual['homogeneous_ms']:.6f}ms、异构{actual['heterogeneous_ms']:.6f}ms。不能把离线潜力当作在线已实现收益。",
        f"- 允许Shared按64行拆分并重新公平选型后，最大预算的矩阵加速比为{shard['speedup_vs_homogeneous']:.4f}×。这只是一项任务粒度敏感性，不是穷尽所有切分策略。",
        '- 最优已找到候选往往是一个较大核配多个较小核；原始“大小双核”有单独表，不能混称。所有非零搜索间隙均保留，不宣称全空间精确最优或物理芯片胜出。','']
    for mode,label in [('matrix','矩阵'),('full_scaled','完整专家')]:
        large=[next(r for r in main if r['total_macs']==largest and r['mode']==mode and r['granularity']=='shared64' and r['batch']==b) for b in (512,1024,2048)]
        text.append(f"Shared允许拆块时，最大预算在B512/B1024/B2048的{label}加速比分别为"+'/'.join(f"{r['speedup_vs_homogeneous']:.4f}×" for r in large)+'。大Batch下的收益明显小于整专家不可拆结果，不能只用六Batch GM宣称普遍胜出。')
    online_shard=last(online_main,'full_scaled','shared64')
    text+=['',f"Shared分块条件下，最大预算的当前在线完整专家加速比仅{online_shard['speedup_vs_homogeneous']:.4f}×，低于同构。当前控制器对任务粒度变化缺乏稳健性，也尚未兑现这一条件下的离线几何潜力。",'']
    text+=['指标说明：核空闲率指没有任务服务的时间，任务内部的累加递推或向量等待另计；它不等于MAC利用率的补数。`service_imbalance`比较有工作核心的最大/最小服务量，零工作核心另见`unused_cores`。离线最优映射可能不唯一，因此工作分配账本描述的是实际返回的那份合法映射。','']
    stats=read(ROOT/'inputs/window_statistics.csv');workload=[]
    for b in BATCHES:
        rr=[r for r in stats if r['split']=='heldout' and int(r['batch'])==b]
        def avg(k):return sum(float(r[k]) for r in rr)/len(rr)
        workload.append(dict(batch=b,windows=len(rr),mean_Me_including_zero=6*b/64,
            active_experts=avg('active'),cold_experts=avg('cold'),hot_experts=avg('hot'),
            mean_largest_Me=avg('max_Me'),cold_total_flops_pct=avg('cold_flops_pct'),shared_flops_pct=25))
    write_csv(out/'workload.csv',workload)
    text+=['## 实际留出负载分布','',table(['Batch','窗口数','平均Me（含0）','活跃专家','Me≤2专家','Me>2专家','每窗最大Me均值','冷专家占总FLOPs%','Shared占总FLOPs%'],
        [[r['batch'],r['windows'],r['mean_Me_including_zero'],f"{r['active_experts']:.2f}",f"{r['cold_experts']:.2f}",f"{r['hot_experts']:.2f}",f"{r['mean_largest_Me']:.2f}",f"{r['cold_total_flops_pct']:.3f}",25] for r in workload]),'']
    for gran,title in [('expert','整专家作为不可拆任务'),('shared64','允许Shared按64 Token行分块')]:
        text+=['## '+title,'','### 开发集冻结硬件','']
        rr=[r for r in frozen if r['granularity']==gran]
        text.append(table(['MAC预算','计算模式','组织','冻结尺寸：核数×(PM×PN×PK)','开发GM ms','跨核数选型搜索间隙%'],[[r['total_macs'],r['mode'],r['organization'],r['geometry'],f"{r['dev_GM_ms']:.6f}",f"{r['organization_global_gap_pct']:.3f}"] for r in rr]))
        for mode,label in [('matrix','矩阵计算'),('full_scaled','完整专家计算，向量吞吐同比扩展')]:
            text+=['','### '+label+'：六Batch等权GM','']
            rr=[r for r in main if r['mode']==mode and r['granularity']==gran and r['batch']=='GM']
            text.append(table(['MAC预算','单核ms','最优同构ms','异构ms','异/单加速比','异/同加速比','利用率：单/同/异%'],[[r['total_macs'],f"{r['single_ms']:.6f}",f"{r['homogeneous_ms']:.6f}",f"{r['heterogeneous_ms']:.6f}",f"{r['speedup_vs_single']:.4f}×",f"{r['speedup_vs_homogeneous']:.4f}×",'/'.join(f"{r[k]:.2f}" for k in ('single_util_pct','homogeneous_util_pct','heterogeneous_util_pct'))] for r in rr]))
            text+=['','### '+label+'：逐Batch数据','']
            rr=[r for r in main if r['mode']==mode and r['granularity']==gran and r['batch']!='GM']
            text.append(table(['MAC预算','Batch','单核ms','最优同构ms','异构ms','异/单','异/同'],[[r['total_macs'],r['batch'],f"{r['single_ms']:.6f}",f"{r['homogeneous_ms']:.6f}",f"{r['heterogeneous_ms']:.6f}",f"{r['speedup_vs_single']:.4f}×",f"{r['speedup_vs_homogeneous']:.4f}×"] for r in rr]))
    text+=['','## 异构实际工作分工与尾部等待','',
        '下面按核类型池列出有效MAC工作占比与调度空闲率，均为窗口算术平均；尾部等待为层完成时刻减去该类型中最后一个核的完成时刻。几何标签按实际尺寸给出，不把任何核预先称为专职Shared核或专职冷专家核。','']
    rr=[r for r in balance if r['mode']=='matrix']
    text.append(table(['MAC预算','粒度','Batch','核类型','核数','有效工作%','调度空闲%','最后完成ms','尾部等待ms'],[[r['total_macs'],r['granularity'],r['batch'],r['geometry'],r['cores'],f"{r['work_pct']:.2f}",f"{r['idle_pct']:.2f}",f"{r['last_core_finish_ms']:.6f}",f"{r['tail_wait_ms']:.6f}"] for r in rr]))
    text+=['','## 如果坚持双核，结果如何','',
        '上面的异构允许2～16核、两种几何。下面单独保留最优已找到双核，防止把多核池收益误认为原始大小双核收益。','']
    rr=[r for r in dual if r['mode']=='matrix' and r['batch']=='GM']
    text.append(table(['MAC预算','粒度','异构双核尺寸','双核异构GM ms','对单核加速比','对同构双核','对最佳同构核数'],[[r['total_macs'],r['granularity'],r['hetero2_geometry'],f"{r['hetero2_ms']:.6f}",f"{r['hetero2_vs_single']:.4f}",f"{r['hetero2_vs_homo2']:.4f}",f"{r['hetero2_vs_best_homo']:.4f}"] for r in rr]))
    text+=['','## 统计与调度最优性界限','',
        '调度界限列为：基线下界/候选可行时间 至 基线可行时间/候选下界；它与bootstrap统计区间是两类不同不确定性。','']
    rr=[r for r in comparison if r['mode']=='matrix' and r['baseline']=='homogeneous']
    text.append(table(['MAC预算','粒度','异构/同构加速比','聚类95%CI','调度界限区间'],[[r['total_macs'],r['granularity'],f"{r['speedup']:.4f}×",f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]",f"[{r['scheduling_bound_speedup_low']:.4f}, {r['scheduling_bound_speedup_high']:.4f}]"] for r in rr]))
    text+=['','## 跨核数与几何的开发集稳定性','',
        '以请求组为单位做200次开发集bootstrap；每次比较该组织全部已枚举候选的可行调度得分。该分析包含不同核数，但仍受未完全求解候选的调度上界误差影响。','']
    rr=[r for r in organization_stability if r['mode']=='matrix']
    text.append(table(['MAC预算','粒度','组织','原尺寸再次最佳%','原尺寸距最佳≤1%的频率%','p95后悔值%'],[[r['total_macs'],r['granularity'],r['organization'],f"{r['exact_geometry_win_pct']:.1f}",f"{r['within_one_percent_pct']:.1f}",f"{r['p95_regret_pct']:.3f}"] for r in rr]))
    text+=['','## 运行时兑现程度','',
        '固定开发集选中的硬件，仅改变派工；没有用留出结果调整硬件。保留首版朴素窗口EFT：它在队列满时只比较可立即准入的核，并在8项内优先选择完成最早的短任务，可能推迟Shared并强制溢出。修正版保持LPT优先级，比较所有核（包括暂满队列）的预计完成时刻，必要时等待，并只对可安全准入的任务做有限前瞻回填；所有组织使用同一规则。修正属于事后发现的通用准入语义修复，没有按留出Batch拟合阈值。预配置只能隐藏假定的控制设置周期，不隐藏本轮已设为零的访存延迟。','']
    text+=['修复朴素准入的收益不计作异构几何的独立贡献。公平在线比较必须给单核、同构、异构同样的修正策略。离线列是有界近优解；若在线偶尔找到更好的合法分配，在线/离线可略低于1。','']
    rr=[r for r in runtime if r['mode']=='matrix']
    text.append(table(['MAC预算','粒度','组织','离线ms','朴素窗口ms','修正准入ms','修正/离线','预配置加速比'],[[r['total_macs'],r['granularity'],r['organization'],f"{r['offline_ms']:.6f}",f"{r['naive_window_ms']:.6f}",f"{r['runtime_ms']:.6f}",f"{r['runtime_over_offline']:.4f}",f"{r['preconfig_speedup']:.5f}"] for r in rr]))
    text+=['','三种组织全部使用修正在线派工后的公平GM对比：','']
    rr=[r for r in online_main if r['batch']=='GM']
    text.append(table(['MAC预算','计算模式','粒度','单核ms','同构ms','异构ms','异构对单核','异构对同构'],[[r['total_macs'],r['mode'],r['granularity'],f"{r['single_ms']:.6f}",f"{r['homogeneous_ms']:.6f}",f"{r['heterogeneous_ms']:.6f}",f"{r['speedup_vs_single']:.4f}",f"{r['speedup_vs_homogeneous']:.4f}"] for r in rr]))
    diagnostics=[]
    for (total,mode,gran),(sel,by,aggs,chosen) in cases.items():
        if mode!='full_scaled' or gran!='expert':continue
        f=chosen['heterogeneous']
        for scheduler in ('offline','runtime'):
            rr=by[f,'primary',scheduler]
            for geom in json.loads(rr[0]['pool_metrics']):
                pools=[json.loads(r['pool_metrics'])[geom] for r in rr]
                diagnostics.append(dict(total_macs=total,scheduler=scheduler,geometry=geom,
                    cores=pools[0]['cores'],MACs=pools[0]['MACs'],
                    work_pct=sum(p['work_pct'] for p in pools)/len(pools),
                    idle_pct=sum(p['idle_pct'] for p in pools)/len(pools)))
    write_csv(out/'runtime_pool_diagnostics.csv',diagnostics)
    text+=['','完整专家模式最大预算的派工差距归因（窗口算术平均）：','']
    rr=[r for r in diagnostics if r['total_macs']==largest]
    text.append(table(['派工','核类型','核数','类型总MACs','有效工作%','调度空闲%'],[[r['scheduler'],r['geometry'],r['cores'],r['MACs'],f"{r['work_pct']:.2f}",f"{r['idle_pct']:.2f}"] for r in rr]))
    text+=['','## 固定64元素/周期向量吞吐对照','',
        '硬件几何冻结，以下对照不重新选型；超过64K时这是故意不扩展向量单元的诊断。主模式求解工作额度0.08，对照0.02，均保留严格下界；例如64K时固定/同比向量吞吐相同，少量延迟差别来自可行调度的求解残差，不应解释为硬件效应。','']
    rr=[r for r in controls if r['control']=='fixed_vector64']
    text.append(table(['MAC预算','粒度','组织','同比向量ms','固定向量ms','减速比'],[[r['total_macs'],r['granularity'],r['organization'],f"{r['primary_ms']:.6f}",f"{r['control_ms']:.6f}",f"{r['slowdown']:.3f}"] for r in rr]))
    text+=['','## 冻结硬件的统一OS对照','',
        '各组织保持同一份开发集选出的硬件；统一OS仅替换循环顺序，未重新做OS专用几何DSE。因此本表衡量数据流假设敏感性，不能称作“最优OS对最优OS”。','']
    rr=[r for r in controls if r['control']=='uniform_OS' and r['granularity']=='expert']
    text.append(table(['MAC预算','计算模式','组织','自动数据流ms','统一OS ms','减速比'],[[r['total_macs'],r['mode'],r['organization'],f"{r['primary_ms']:.6f}",f"{r['control_ms']:.6f}",f"{r['slowdown']:.3f}"] for r in rr]))
    supply=[]
    for (total,mode,gran),(sel,by,aggs,chosen) in cases.items():
        for org,f in chosen.items():
            raw=by[f,'primary','offline'];a=aggs[f,'primary','offline']['GM']
            supply.append(dict(total_macs=total,mode=mode,granularity=gran,organization=org,
                compute_GM_ms=a['ms'],unique_weight_GM_MiB=a['unique_weight_MiB'],
                largest_window_unique_weight_MiB=max(float(r['unique_weight_MiB']) for r in raw),
                necessary_average_weight_BW_TBs=a['unique_weight_BW_TBs'],
                unique_weight_service_GM_ms_at_256GBs=a['unique_weight_MiB']*2**20/256e6))
    write_csv(out/'supply_requirements.csv',supply)
    text+=['','## 供数需求反推：必要条件，不是访存仿真','',
        '带宽列逐窗口用“唯一活跃BF16权重字节/计算时间”计算后取等权GM，假设每份权重只从供数层读取一次。它不包括实际数据流重读、激活、部分和、瞬时突发和互连；也不代表HBM必须提供该带宽，若权重预驻留则由片上层承担。容量列是整层活跃权重全部驻留所需容量；流式分块无需同时容纳整层。','']
    rr=[r for r in supply if r['mode']=='matrix' and r['granularity']=='expert']
    text.append(table(['MAC预算','组织','计算GM ms','唯一权重GM MiB','窗口最大权重MiB','必要平均权重TB/s','256GB/s权重服务GM ms'],[[r['total_macs'],r['organization'],f"{r['compute_GM_ms']:.6f}",f"{r['unique_weight_GM_MiB']:.2f}",f"{r['largest_window_unique_weight_MiB']:.2f}",f"{r['necessary_average_weight_BW_TBs']:.3f}",f"{r['unique_weight_service_GM_ms_at_256GBs']:.4f}"] for r in rr]))
    text+=['','## 完整文件','',
        '- `main.csv`：两种粒度、两种计算模式的逐Batch与GM主表。',
        '- `all_aggregates.csv`：所有单/同/异核数、统一OS、向量对照、在线与离线。',
        '- `frozen_hardware.csv`：所有主表硬件及开发选型值。',
        '- `certificates.csv`与每个结果目录的`all_candidates.csv.gz`：完整候选空间、上下界、选型间隙。',
        '- `heterogeneous_pool_work.csv`：逐窗口各核类型有效工作量和空闲时间。',
        '- 每个结果目录`holdout.csv.gz`：逐窗口原始整数周期、映射、有效/填充MACs、运行时与离线结果。',
        '- `unique_weight_BW_TBs`仅为“唯一活跃权重字节/纯计算时间”的必要平均带宽下界；不是完整SRAM端口需求，更不是已验证的端到端性能。','']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    plots(main,out)

def plots(main,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    all_z=[r['speedup_vs_homogeneous'] for r in main if r['mode']=='matrix' and r['batch']!='GM']
    norm=TwoSlopeNorm(1,min(.95,min(all_z)),max(1.05,max(all_z)))
    for ax,gran,title in zip(axes,('expert','shared64'),('Whole experts','Shared split into 64-token tasks')):
        rows=[r for r in main if r['mode']=='matrix' and r['granularity']==gran and r['batch']!='GM']
        z=np.array([[next(r['speedup_vs_homogeneous'] for r in rows if r['total_macs']==m and r['batch']==b) for b in BATCHES] for m in BUDGETS])
        im=ax.imshow(z,cmap='RdYlGn',norm=norm,aspect='auto')
        ax.set_xticks(range(6),[str(b) for b in BATCHES]);ax.set_yticks(range(5),['64K','128K','256K','512K','1M'])
        ax.set_xlabel('Tokens per MoE layer');ax.set_ylabel('Physical multiplier budget');ax.set_title(title)
        for y in range(5):
            for x in range(6):ax.text(x,y,f'{z[y,x]:.3f}x',ha='center',va='center',fontsize=10)
        fig.colorbar(im,ax=ax,label='Heterogeneous speedup vs best homogeneous')
    fig.suptitle('Ideal-supply analytical compute model; hardware frozen on development')
    fig.savefig(out/'advantage_map.png',dpi=180);fig.savefig(out/'advantage_map.pdf');plt.close(fig)

if __name__=='__main__':build()
