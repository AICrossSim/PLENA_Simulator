"""Chinese report and the conditional E3 decision; no assumed victory."""
from .common import *
from .report import build,load_results,group_rows,strongest_map
from .evaluate import selections

def fmt(x,n=4):return f'{float(x):.{n}f}'
def label(r):return f"{r['macs']//1024}K / B{r['batch']} / {r['org']}"

def scheduler_doc():
 d=read_json(ROOT/'E2/delta_selection.json');delta=d['delta']
 text=f'''# 剩余工作量配额派工器

冻结参数 δ = **{delta}**。在105个开发窗口、五档算力和五种组织上统一选择；目标是各预算/组织/batch等权的在线÷离线配对GM。候选0.02/0.05/0.10/0.20，逐点记录见 `delta_development.csv.gz`。留出结果不参与选参。

Router已提供全部Me。控制器估计的是周期和两类核应承担的有效MAC工作量，不猜Expert ID。

1. 将每个专家（或Shared行块）在每类核上的成本算出。两类核的连续分配松弛按相对成本排序、平衡池内平均周期；只保留两类核的总工作量份额和层完成时刻估计，丢弃松弛中的逐任务归属。**不调用离线整数求解器，也不读取其owner列表。**
2. Shared需广播时，比较不同接收核数量的计算＋广播时间和剩余routed工作量，选择预计整层更早完成的接收核组。主实验按共同分块广播契约执行，详见METHOD.md。
3. Routed描述符按最小预计执行时间降序排列，窗口深度8；每核仅Current和Next两个上下文。保留队首优先、队满时等待，以及窗口内其他合法任务旁路。
4. 首先比较所有核的预计完成时刻。如果最快选择使某类核已接工作量超过(1+δ)×目标份额，则考虑未超额的另一类核。若它仍能在预计整层排空截止时间的(1+δ)倍内完成，就分过去；否则允许逃逸到原先最快的核。
5. 每次绑定增记该核类型的有效工作量；每个任务完成从对应剩余周期计数中扣除；已接有效工作量是层内累计值，不随任务完成减少。这里成本模型已准确知道执行成本，无另外训练的时长预测器。

原任务的“除非其他核都会更晚完成”若直接解释为“比当前任务的最小EFT更晚”，会让配额永远无效，因为其他候选本来就不会早于最小EFT。本实现把逃逸门槛明确改为**整层预测排空截止时间**，这是为使规则非空而做的语义澄清，不能隐瞒成完全逐字执行。

```text
costs = estimate_all_experts(router_counts, frozen_shapes)
targets, layer_deadline = fractional_pool_balance(costs)
plan_common_shared_broadcast_cohort()
pending = LPT(routed_tasks)
while pending:
    for task in first_8(pending):
        best = argmin_core(predicted_finish(core, task))
        if exceeds_pool_quota(best, task):
            alternate = earliest_finish_among_under_quota_pools(task)
            if alternate.finish <= (1 + delta) * forecast_layer_finish:
                best = alternate
        if best.Current_and_Next_full:
            continue  # 不强塞慢核，允许窗口内其他任务先走
        bind(task, best)
        accepted_work[type(best)] += useful_MACs(task)
        break
    else:
        wait_until_a_context_retires()
```

硬件状态：每类核目标份额、已接有效MAC工作量、预测剩余周期各64bit；每核Current/Next上下文、预计结束计数；8项前瞻描述符（Expert ID、Me、行区间、类型、预计周期）；共享广播接收掩码与块引用计数。两类核份额生成需要一次按相对成本排序/连续分配，可由顺序控制器和定点除法实现，**不是已经综合出的几个免费比较器**。每次发射计1周期，Next预配置计2周期且可隐藏。层前另计每(task,type)成本查表1周期、bitonic排序网络逐级1周期、每任务份额扫描1周期，以及Shared接收核数量逐方案扫描的周期；新旧在线共同承担基础成本查表/排序费用。此为公开的控制器服务假设，完整门级延迟/面积尚未综合，应在下一阶段校准。

旧在线列：调用第五轮冻结的LPT/EFT准入函数取得owner和每核routed顺序，再按相同Shared广播cohort重放，以避免拿无广播旧时间与收费新时间混比。原第五轮零广播周期另存 `original_r5_no_broadcast_ms`。`quota_ablation_ms` 固定新的Shared接收核规划，仅关闭份额门控，隔离配额本身的贡献。

`workload_balance.csv` 同时列出模型预测份额、在线实际份额、离线事后份额和各核空闲率。最后一项只用于诊断，绝未进入在线决策。
'''
 (ROOT/'E2/SCHEDULER.md').write_text(text)

def e3_no_winners(cells):
 ps=[r for r in cells if r['org'] in ('P','P+')];wins=[r for r in ps if r['crit_A10'] and r['crit_B'] and r['crit_C']]
 if wins:return wins
 # Multiplicative distance to all four numeric inequalities; no arbitrary
 # cherry-picking by latency alone.
 def gap(r):return max(r['req_weight_TBps']/10,1.1/r['speedup_vs_strongest_homo_offline'],1.05/r['speedup_vs_strongest_homo_online'],r['online_over_offline']/1.05)-1
 distinct={}
 for r in ps:
  key=r['macs'],r['batch']
  if key not in distinct or gap(r)<gap(distinct[key]):distinct[key]=r
 close=sorted(distinct.values(),key=lambda r:(gap(r),r['macs'],r['batch'],r['org']))[:3]
 close=[dict(r,bandwidth_excess_TBps=max(0,r['req_weight_TBps']-10),
  offline_gain_shortfall_pp=max(0,100*(1.10-r['speedup_vs_strongest_homo_offline'])),
  online_gain_shortfall_pp=max(0,100*(1.05-r['speedup_vs_strongest_homo_online'])),
  online_lag_excess_pp=max(0,100*(r['online_over_offline']-1.05))) for r in close]
 first={'A':0,'B':0,'C':0}
 for r in ps:first['A' if not r['crit_A10'] else 'B' if not r['crit_B'] else 'C']+=1
 exclusion=[]
 for r in ps:
  reason='A: even feasible-upper-time bandwidth exceeds10' if not r['crit_A10'] else 'B: H upper / P strict lower <1.10' if r['speedup_possible_upper']<1.10 else 'C: measured online speedup <1.05' if r['speedup_vs_strongest_homo_online']<1.05 else 'C: online / offline feasible upper >1.05' if r['online_over_offline']>1.05 else 'UNRESOLVED offline interval'
  exclusion.append(dict(macs=r['macs'],batch=r['batch'],org=r['org'],certified_excluded=not reason.startswith('UNRESOLVED'),reason=reason))
 unresolved=[r for r in exclusion if not r['certified_excluded']]
 lines=['# E3：没有同时满足 A、B、C 的格子','',
  '时间均为纯计算、假设普通供数充足；显式计入10TB/s额外Shared广播。A使用10TB/s平均权重供给档，3.3TB/s为附加更严档。',
  '',f'按A→B→C顺序检查，首先失败：A {first["A"]} 个，B {first["B"]} 个，C {first["C"]} 个（合计{len(ps)}个异构组织×格子）。',
  '',f'利用严格上下界及实测在线时间，可排除 {len(ps)-len(unresolved)}/{len(ps)} 个冻结组织×格子；仍因求解区间无法排除的有 {len(unresolved)} 个。逐格证明条件见exclusion_bounds.csv。该结论只覆盖冻结硬件，不推广为任意未选硬件均不可能。',
  '', '每个物理格子先取P/P+中违约较小者，再按所有不等式中最大的倍数违约幅度排序，列出最近三个不同格子；这是事后诊断，不是重新选型：','',
  table(['格子','需求 TB/s / ≤10','离线加速 / ≥1.10','在线加速 / ≥1.05','在线÷离线 / ≤1.05','最大违约'],
   [[label(r),fmt(r['req_weight_TBps']),fmt(r['speedup_vs_strongest_homo_offline']),fmt(r['speedup_vs_strongest_homo_online']),fmt(r['online_over_offline']),f'{100*gap(r):.2f}%'] for r in close]),
  '', '具体缺口如下；百分数之差使用百分点，零表示该条件已满足。所有加速比分母均为开发集预先选定的最强同构。','',
  table(['格子','带宽超额TB/s','离线收益差距(百分点)','在线收益差距(百分点)','在线落后离线超额(百分点)'],[[label(r),fmt(r['bandwidth_excess_TBps'],3),fmt(r['offline_gain_shortfall_pp'],2),fmt(r['online_gain_shortfall_pp'],2),fmt(r['online_lag_excess_pp'],2)] for r in close]),
  '',table(['最近格子','离线加速比严格下界','可能上界'],[[label(r),fmt(r['speedup_certified_lower']),fmt(r['speedup_possible_upper'])] for r in close]),
  '', '“没有找到”仅针对本次声明的有限搜索空间、广播契约和冻结配置；如果求解器区间跨过1.10，不能据可行上界 alone 排除尚未找到的更优分配。',
  '', '因为没有通过A+B+C的格子，用户规定的赢家专项5/20TB/s、Shared32/128与全部专家可拆敏感性不触发，不能把它们写成已经跑过。200次开发集来源组bootstrap仍对全部冻结组织完成，见下表。',
  '', '以下稳定性是全候选可行上界分数的重采样选型；并非声称每个候选都已经证明整数分配最优。','']
 br=[]
 for s in selections():
  rows=read_csv(ROOT/'E1/search'/f"{s['budget']}_{s['chunk']}"/'bootstrap.csv');r=next(x for x in rows if x['org']==s['org'])
  br.append([s['budget']//1024,s['org'],s['geometry'],fmt(r['same_hardware_pct'],1),fmt(r['within_1pct_pct'],1),fmt(r['p95_regret_pct'],2)])
 lines.append(table(['MAC(K)','组织','冻结几何','相同硬件选中%','距重采样最佳≤1%次数%','p95损失%'],br))
 (ROOT/'E3').mkdir(exist_ok=True);(ROOT/'E3/winning_cells.md').write_text('\n'.join(lines)+'\n')
 write_csv(ROOT/'E3/exclusion_bounds.csv',exclusion)
 write_json(ROOT/'E3/decision.json',dict(winners=[],first_failure=first,nearest=close,conditional_sensitivity_triggered=False,bootstrap_repeats=200,
  certified_excluded=len(ps)-len(unresolved),unresolved_frozen_cells=unresolved))
 return []

def final_report(cells):
 ps=[r for r in cells if r['org'] in ('P','P+')];wins=[r for r in ps if r['crit_A10'] and r['crit_B'] and r['crit_C']];ab=[r for r in ps if r['crit_A10'] and r['crit_B']]
 cert=[r for r in wins if r['speedup_certified_lower']>=1.1 and r['req_weight_upper_from_lb_TBps']<=10 and r['crit_C_certified']]
 by={(r['macs'],r['batch'],r['org']):r for r in cells};strong=strongest_map();comparison=read_csv(ROOT/'E2/online_compare.csv')
 hs=[r for r in comparison if r['org'] in ('H','H+')];pp=[r for r in comparison if r['org'] in ('P','P+')]
 names=lambda rr:'；'.join(label(r) for r in rr) or '无'
 conclusion='存在达到判据的候选格子；仍需有限端口/面积校准。' if wins else '本轮没有找到同时通过A、B、C的冻结设计格子。'+('存在离线A+B通过格子，但C未兑现。' if ab else '离线收益与带宽条件尚未同时成立。')
 refs=read_csv(ROOT/'E1/per_batch_hardware_reference.csv');refwins=[r for r in refs if r['org'] in ('P','P+') and all(r[k]=='True' for k in ('crit_A10','crit_B','crit_C'))]
 feasible=[r for r in ps if r['crit_A10']];beneficial=[r for r in ps if r['crit_B']]
 best_a=max(feasible,key=lambda r:r['speedup_vs_strongest_homo_offline']) if feasible else None
 least_b=min(beneficial,key=lambda r:r['req_weight_TBps']) if beneficial else None
 effects=read_csv(ROOT/'E1/shared_split_effect.csv');focus=next(r for r in effects if int(r['macs'])==1048576 and int(r['batch'])==2048)
 online_focus=next(r for r in comparison if int(r['macs'])==1048576 and int(r['batch'])==2048 and r['org']=='P')
 online_small=next(r for r in comparison if int(r['macs'])==1048576 and int(r['batch'])==64 and r['org']=='P')
 lines=['# 第六轮：带宽可行、最强同构与在线兑现','',
 '**全部时间：纯计算、假设普通供数充足；主口径矩阵＋按MAC同比扩展的向量运算，额外Shared广播端口10TB/s。A10与A3.3是必要带宽条件，不是实际HBM/SRAM端到端验证。**','',
 table(['第一页必须回答的问题','本轮答案'],[
  ['1. 同时满足A10、B、C？',(names(wins)+f'；其中上下界也确认全部判据的有{len(cert)}格。') if wins else '无（35个物理格子×P/P+）。E3逐格给出带宽/收益上下界或在线时间的排除依据。'],
  ['2. 只看带宽可行与离线收益（A10+B）？',names(ab)],
  ['3. Shared拆分同构削弱多少异构收益？',f"1M/B2048：第五轮固定P不变，对同构的加速从{float(focus['round5_H_over_P']):.4f}×降为{float(focus['round5_Hplus_over_same_P']):.4f}×；第六轮同口径为{float(focus['round6_H_over_P']):.4f}×→{float(focus['round6_Hplus_over_P']):.4f}×。全部格子见shared_split_effect.csv及下表。"],
  ['4. 新在线÷离线是多少？',f"异构P/P+范围 {min(float(r['new_over_offline']) for r in pp):.4f}–{max(float(r['new_over_offline']) for r in pp):.4f}；同构H/H+范围 {min(float(r['new_over_offline']) for r in hs):.4f}–{max(float(r['new_over_offline']) for r in hs):.4f}。1M/B2048的P从旧{float(online_focus['old_over_offline']):.4f}降到{float(online_focus['new_over_offline']):.4f}，仍未达到≤1.05。"],
  ['5. 结论',conclusion]]),
  '',(f"带宽≤10TB/s的异构候选中，最高离线加速为 **{best_a['speedup_vs_strongest_homo_offline']:.4f}×**（{label(best_a)}）。" if best_a else '没有通过A10的异构候选。')+(f" 达到≥10%离线收益的候选中，最低所需权重带宽为 **{least_b['req_weight_TBps']:.3f}TB/s**（{label(least_b)}）。" if least_b else '没有达到≥10%离线收益的候选。'),
 '', '## 判据与可信范围','',
 'A10：唯一BF16权重字节÷离线计算时间≤10TB/s；A3.3使用3.3TB/s。B：最强同构离线÷异构离线≥1.10。C：最强同构在线÷异构在线≥1.05，且异构在线÷离线≤1.05。',
 '', 'offline_ms是求解器找到的可行上界，offline_lb_ms是严格下界。没有闭合的间隔明确保留；严谨收益区间为同构下界/异构上界至同构上界/异构下界。不能将此模型中的候选称为已经综合验证的架构胜出。',
 '', '每档S/H/H+/P/P+各自在开发集上独立选一套硬件，跨所有batch冻结。每格最强同构组织H或H+由该batch开发集成绩决定，留出集不参与。B4096是不同请求组真实路由的后拼接，开发/留出各105窗口，每batch15个。',
 '', '## 冻结硬件','',
 table(['MAC(K)','组织','核数','完整几何（数量×PM×PN×PK）','使用MAC','未用额度','开发选型残余界差%'],[[s['budget']//1024,s['org'],len(s['cores']),s['geometry'],s['used_macs'],s['unused_macs'],fmt(s['dse_gap_pct'],3)] for s in selections()]),
  '', '12/13个完全相同的4096对齐核无法用满2的幂次预算，枚举所有不超过预算的合法几何并计入闲置额度；其他候选精确用满预算。乘法器相等不等于综合面积相等。',
  '', '## 真实负载与瓶颈背景','',
  table(['Batch','每专家平均Token（含未激活）','活跃专家均值','Shared有效工作量%','Me≤2有效工作量%','每窗最大routed Me均值'],[
   [r['batch'],fmt(r['routed_mean_tokens'],0),fmt(r['active_experts_mean'],2),r['shared_useful_work_pct'],fmt(r['cold_useful_work_pct'],5),fmt(r['max_routed_tokens_mean'],2)] for r in read_csv(ROOT/'E1/workload.csv') if r['split']=='heldout']),
  '', '大batch摊薄了每次权重读取，却也使routed矩阵更规整：Shared工作量固定25%，Me≤2任务的有效工作量在B512后已很小。是否仍有异构收益，必须由几何/尾部分配实测决定，不能从冷专家个数直接推导。',
 '', '## 异构逐格判据（矩阵＋向量）','',
 table(['MAC(K)','Batch','组织','离线ms [下界]','在线ms','需W TB/s','离线加速/强同构','在线加速/强同构','在线/离线','A10/A3.3/B/C'],[
  [r['macs']//1024,r['batch'],r['org'],f"{r['offline_ms']:.6f} [{r['offline_lb_ms']:.6f}]",fmt(r['online_ms'],6),fmt(r['req_weight_TBps'],3),fmt(r['speedup_vs_strongest_homo_offline']),fmt(r['speedup_vs_strongest_homo_online']),fmt(r['online_over_offline']),'/'.join('是' if r[k] else '否' for k in ('crit_A10','crit_A3p3','crit_B','crit_C'))] for r in ps]),
 '', '## Shared拆分同构对收益的影响','',
 table(['MAC(K)','Batch','H ms','H+ ms','开发冻结的强同构','P/H 加速','P/H+ 加速','收益变化（百分点）'],[
  [m//1024,b,fmt(by[m,b,'H']['offline_ms'],6),fmt(by[m,b,'H+']['offline_ms'],6),strong[m,b],fmt(by[m,b,'H']['offline_ms']/by[m,b,'P']['offline_ms']),fmt(by[m,b,'H+']['offline_ms']/by[m,b,'P']['offline_ms']),fmt(100*(by[m,b,'H']['offline_ms']-by[m,b,'H+']['offline_ms'])/by[m,b,'P']['offline_ms'],2)] for m in BUDGETS for b in BATCHES]),
  '', '正值表示允许Shared拆分使同构更快，因而缩小异构优势；负值表示广播/阶段同步让该冻结H+反而更慢。这个比较重新在第六轮同一口径下计算，不能把第五轮无广播的时间直接混入。',
  '', 'E1/shared_split_effect.csv另外读取第五轮已提交的原始结果：每一轮都固定该轮的整专家P分母，只替换H→H+。该文件覆盖全部35格，新增B4096的第五轮数据标NA；跨轮差异还包含七batch重新选型和广播收费，不能全部归因于Shared拆分。',
  '', '## 在线兑现与归因','',
  f"1M/B2048的P：平均最长核服务量比离线多 {(float(online_focus['new_assignment_critical_mean_ms'])-float(online_focus['offline_assignment_critical_mean_ms']))*1000:.3f}µs，另有启动 {float(online_focus['new_setup_mean_ms'])*1000:.3f}µs、残余等待 {float(online_focus['new_residual_wait_mean_ms'])*1000:.3f}µs。这些分解项采用窗口算术均值以保持可加，主性能比较仍用GM。在当前不可拆专家、8项前瞻与Current/Next准入规则下，剩余差距主要出现在具体任务归属造成的核间失配和排队等待。",' ',
  f"新规则也有退步的格子：1M/B64的P在线/离线由 {float(online_small['old_over_offline']):.4f} 变为 {float(online_small['new_over_offline']):.4f}，不能宣称统一改善。",'',
 table(['MAC(K)','Batch','组织','旧在线ms','新在线ms','离线ms','旧/离线','新/离线','关闭配额ms'],[[int(r['macs'])//1024,r['batch'],r['org'],fmt(r['old_online_ms'],6),fmt(r['new_online_ms'],6),fmt(r['offline_ms'],6),fmt(r['old_over_offline']),fmt(r['new_over_offline']),fmt(r['quota_ablation_ms'],6)] for r in comparison]),
 '', '旧在线保留第五轮的owner及routed顺序，再用共同Shared广播契约回放；其原始无广播周期另列。新在线按整层预计排空期限执行配额逃逸，不读取离线具体分配。配额关闭列保持新Shared接收核规划不变，便于分离两项改动。硬件状态与伪代码见E2/SCHEDULER.md。',
  '', '## 供数与模型边界','',
  f'每个batch单独在开发集选型的上限参考中，A10+B+C通过 {len(refwins)} 个组织×格子。完整几何与判据见E1/per_batch_hardware_reference.csv；这类参考允许不同batch更换硬件，不能替代主协议的跨batch冻结结果。',
 '峰值驻留是活跃权重块的双缓冲容量，非整层权重常驻，也不包含激活/累加SRAM。加激活/输出的带宽仅统计专家边界BF16输入读、FP32输出写。普通输入理想、广播块内计算均匀化、阶段屏障及分析型阵列时序均属于本轮模型假设，不能声称完成真实SRAM端口闭环。',
 '', '额外Shared副本收费会延长计算时间，因而降低“平均所需带宽”。E1/cells.csv同时保留无广播owner计算带宽；E1/zero_broadcast_appendix.csv给冻结硬件的免广播对照，避免把拖慢后的A通过当作免费物理收益。',
 '', '## 文件与复现','',
 '- E1/cells.csv：175行全组织主表；matrix_appendix.csv：纯矩阵附表；per_batch_hardware_reference.csv：每batch开发选型上限参考。',
 '- E1/search/：全几何候选、严格下界、冻结选型、200次来源组bootstrap和重复运行记录。',
 '- E1/windows/：每个留出窗口的上下界、归属、服务周期、广播、在线份额、流量和利用率。',
 '- E2/online_compare.csv、workload_balance.csv、SCHEDULER.md：新旧在线、工作量和硬件规则。',
 '- E3/winning_cells.md：通过格子专项敏感性或无赢家时最近三格与失败原因。',
 '- figures/fig_cells.pdf：开发选定异构组织的三项热力图，并分别附P/P+；fig_cells.png为首图。',
 '- README.md：提交号和完整运行命令；METHOD.md：计时/广播/统计契约。','']
 (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines))

def main():
 cells=build();scheduler_doc();wins=e3_no_winners(cells)
 if wins:
  write_json(ROOT/'E3/pending_winners.json',wins)
  print('WINNERS REQUIRE CONDITIONAL SENSITIVITY',len(wins),flush=True)
 final_report(cells)
if __name__=='__main__':main()
