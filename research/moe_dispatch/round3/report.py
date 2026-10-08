"""Build the Chinese final report only from completed numerical outputs."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from .common import ROOT, REPO, OLD, read_csv, table, write_json, metadata, sha
from .config import BATCHES, MODES


def f(value, digits=4):
    return f'{float(value):.{digits}f}'


def rows(path):
    return read_csv(ROOT/path)


def headline_rows():
    return [r for r in rows('E4/heldout_main_table.csv')
            if r['constraint_group']=='C0' and r['onchip_mode']=='pipelined' and r['dispatch']=='fixed']


def shape(d):
    return '+'.join(f"{c['pm']}×{c['pn']}×{c['pk']}" for c in d['cores'])


def capacities(d, key):
    return '+'.join(f(x/1024,0) for x in d[key])


def main():
    selected=json.loads((ROOT/'E4/selected_designs.json').read_text())
    validation=json.loads((ROOT/'VALIDATION.json').read_text())
    bound=rows('E2/bounds_summary.csv')
    heldbound=[r for r in bound if r['set']=='heldout' and float(r['bw_GBps'])==256]
    whole=next(r for r in heldbound if r['batch']=='all')
    newheadroom=[r for r in rows('E4/selected_baseline_headroom.csv')
                 if r['onchip_mode']=='pipelined' and r['dispatch']=='fixed']
    newwhole=next(r for r in newheadroom if r['batch']=='all')
    ablation=[r for r in rows('E3/ablation.csv') if r['batch']=='all']
    ab={r['config']:r for r in ablation}
    attribution=json.loads((ROOT/'E3/attribution.json').read_text())
    dispatch=rows('E5/dispatch/compare.csv')
    dmain={r['design']:r for r in dispatch if r['onchip_mode']=='pipelined' and r['dispatch']=='fixed'}
    best=selected['best_hetero_by_mode']['pipelined']
    predictor=rows('E5/predictor/predictor_table.csv')
    pred=[r for r in predictor if float(r['bw_GBps'])==256 and r['onchip_mode']=='pipelined']
    pmap={(r['design'],r['method']):r for r in pred}
    gates=[r for r in rows('E4/gates.csv') if r['constraint_group']=='C0' and r['onchip_mode']=='pipelined' and r['dispatch']=='fixed']
    proofs=rows('E4/proof_status.csv')
    robust_receipt=json.loads((ROOT/'E4/robust_heldout/RUN_RECEIPT.json').read_text())
    robust_global=[r for r in rows('E4/robust_heldout/winners.csv')
                   if r['family']=='all' and r['objective']=='geomean']
    synthetic=rows('E4/synthetic/reverse_search.csv')
    synthetic_audit=json.loads((ROOT/'E4/synthetic/COMPLETION_AUDIT.json').read_text())
    sobol=rows('E5/sobol/sobol_samples.csv')
    sobol_audit=json.loads((ROOT/'E5/sobol/COMPLETION_AUDIT.json').read_text())
    synthetic_batches=[]
    for batch in sorted({int(r['batch']) for r in synthetic}):
        group=[r for r in synthetic if int(r['batch'])==batch]
        reversals=[r for r in group if float(r['delta'])<0]
        synthetic_batches.append([f'B{batch}',len(group),len(reversals),
            sum(r['compute_shapes_distinct']=='True' for r in reversals),
            f(max([0.0,*[-100*float(r['delta']) for r in reversals]])),
            sum(r['proof_complete']=='True' for r in group)])
    qtable=[
        ['Q1 最大可能收益',f"当前模型下，整体相对本轮 B1 至多 {f(newwhole['max_gain_vs_B1_pct'])}%，相对 B2 至多 {f(newwhole['max_gain_vs_B2_pct'])}%；达不到同时快 5%。B96/128 单独仍有空间。",'E4/selected_baseline_headroom.csv'],
        ['Q2 W 槽是否主因',f"H0 {f(ab['H0']['geomean_ms'])} → H2 {f(ab['H2']['geomean_ms'])} ms；W∞可回收 {f(100*attribution['recovery_fraction'],2)}%，未达 80% 标准。",'E3/attribution.json'],
        ['Q3 重新搜索后',f"B1 {f(dmain['B1']['all_geomean_ms'])}；B2 {f(dmain['B2']['all_geomean_ms'])}；开发集选定异构 {best} {f(dmain[best]['all_geomean_ms'])} ms。通过双基线 5% 门槛的异构 {sum(r['enter_calibration']=='True' for r in gates)} 个。",'E4/heldout_main_table.csv; gates.csv'],
        ['Q4 在线/离线',f"B2 {f(dmain['B2']['ratio_vs_milp'],6)}；{best} {f(dmain[best]['ratio_vs_milp'],6)}。1.01 验收按实际比值判断，未达标项另列原因。",'E5/dispatch/compare.csv'],
        ['Q5 预测器收益',f"ours 相对 nominal 的延迟变化：B1 {f(100*(float(pmap['B1','ours']['ratio_vs_no_pred'])-1))}%，B2 {f(100*(float(pmap['B2','ours']['ratio_vs_no_pred'])-1))}%，{best} {f(100*(float(pmap[best,'ours']['ratio_vs_no_pred'])-1))}%。负值才是更快。",'E5/predictor/predictor_table.csv']]
    lines=['# 第三轮：256 GB/s 供数上限下的完整评估','',
           '本报告使用 BF16、18 个开发窗口和 135 个既有留出窗口；1 模型周期＝1 ns。所有 ms 为路由完成后的 MoE FFN 解析估计，包含 Gate/Up、SiLU、Down 和汇合，不包含完整模型的 attention/router/norm，也不是原生 HBM 或 RTL 实测。',
           '',table(['问题','数值答案','数据来源'],qtable),'',
           '以下区分“126 下选出、在 256 下评估”和“256 下重新搜索”。不同控制／供数计数可以重叠，不能相加成延迟；不同消融收益也不能相加。',
           '', '## 1. 工作点与前端','',
           '`min(256,credits×32/65) GB/s`：256／390／520 个额度分别对应 126.030769／192／256。三种组织共用同一前端。520 点要求平均每周期 8 个 32 B 请求、足够标签及返回落地吞吐，尚无 RTL／面积验证；详见 E1/OPERATING_POINT.md。',
           '', '主设计固定 12,288 个乘法器、2,158,592 B 存储、W/X/累加 bank 总量 64/24/12、向量吞吐 64 元素/周期。可交换的私有／落地容量总量为 532 KiB，其余公共存储固定。等乘法器、容量、端口不能代替等综合面积。',
           '', '## 2. 下界与能够争取的空间','',
           table(['Batch','窗口数','相对冻结 B1 最大收益 %','相对冻结 B2 最大收益 %','双基线 5% 是否可能'],
                 [[r['batch'],r['n_windows'],f(r['max_gain_vs_b1_pct']),f(r['max_gain_vs_b2_pct']),r['gate5_reachable']] for r in heldbound]),
           '', '下界按每窗口的唯一权重、MAC、向量、必需片上端口及乐观 532 KiB 行分块计算，取最大值。旧 Z384/W40 的区域限制已经删除。新的行分块下界同时免费授予每个任务 Z532 和 W532，故是乐观放宽；其有效性依赖当前完整行的 Gate/Up→Down 生命周期，不能用于未来的跨列融合。',
           '', '整体能否胜出按全部窗口配对几何平均判断；单个大 batch 仍有局部空间。本轮全体已检查结果的下界验收见 VALIDATION.json。',
           '', 'E2 上限的参照是历史冻结基线及其原控制协议。开发集重新调优不保证留出集也更快，因此 E4 的开发集证明和留出集门槛分别使用最终同协议 B1/B2，不能直接套用历史上限。',
           '', '下面把同一个逐窗口 LB 与第三轮冻结硬件的实际 fixed 结果重新配对；不修改 E2 历史表，也不假设开发集调优必然改善留出集。',
           '', table(['Batch','窗口数','LB GM ms','本轮 B1 GM ms','最多比 B1 快 %','最多比 B2 快 %','双基线 5% 是否可能'],
                     [[r['batch'],r['n_windows'],f(r['lower_bound_GM_ms']),f(r['B1_GM_ms']),
                       f(r['max_gain_vs_B1_pct']),f(r['max_gain_vs_B2_pct']),r['gate5_reachable']] for r in newheadroom]),
           '', '## 3. 权重缓冲消融：126 下选出、在 256 下评估','',
           table(['配置','等资源','GM ms','相对单核','说明'],
                 [[r['config'],r['iso'],f(r['geomean_ms']),f(r['ratio_vs_S0'],6),r['change']] for r in ablation]),
           '',f"可回收比例 {f(100*attribution['recovery_fraction'],2)}%，按任务书标准不能写 W 槽不足是主因。余下的重读、核完成差及端口服务积分见 E3/ABLATION.md；它们只能作为诊断，未被分离成互斥的因果时间。W∞保持原行分块和阶段生命周期，只放宽有效 W 驻留／前瞻窗口，不自动引入跨阶段完整专家缓存。",
           '', '曲线中的在途量是“当前流体供数率×65 ns”的服务等价代理量，冷响应等待为 0，不能据此验证逐请求槽位占用。共享池实际有限容量由独立字节预留账本检查。single_fetcher 指只有一个核获得正供数的区间；Shared 供数率是大核执行 Shared 期间全芯片 HBM 的速率，包含另一核。',
           '', '## 4. 256 下重新搜索的硬件','']
    for mode in MODES:
        lines += [f'### {mode}，C0 主约束','',
                  table(['组织','PM×PN×PK','数据流','落地','W KiB','X KiB','累加 KiB','Z KiB','W/X/累加 banks','向量份额'],
                        [[name,shape(d),'+'.join(d['flows']),d['landing_mode'],
                          ('共享池 '+f(d['landing_pool_bytes']/1024,0) if d['landing_mode']=='shared' else capacities(d,'w_bytes')),
                          capacities(d,'x_bytes'),capacities(d,'acc_bytes'),capacities(d,'z_bytes'),
                          '/'.join('+'.join(map(str,d[k])) for k in ('w_banks','x_banks','acc_banks')),
                          '+'.join(map(str,d['vector_lanes']))]
                         for name,d in selected['modes'][mode]['C0'].items()]),'']
    lines += ['每核维度、存储和端口按 core0/core1 顺序列出；W 池只计算一次，共享池的各核单独最大槽数不能同时相加。H33 允许与同构空间重叠；若最终核心形状相同，不能把标签 H33 本身当作异构贡献。',
              '', 'H51／H42／H33 表示两核乘法器预算的无序划分族 5+1／4+2／3+3，不表示 core0/core1 必须按这个顺序排列，也不能当成 PM 行数。具体核顺序以每行 core0/core1 的形状和资源列表为准。PM 对应一次处理的 token 行数，PN 是输出列宽，PK 是物理点积宽度；本轮三个维度均有变化。',
              '', '共享落地池开放的是字节容量共享；本轮仍保留各核冻结的 W bank／读端口份额，没有免费借用另一核的端口。C1 检查扣除计算块后的物理在途空间不少于 16 KiB，它不保证实际供数达到 252 GB/s：尾块有效载荷、阶段边界和片上读端口仍可能限制速率。',
              '', '主目标是 18 开发窗口上 MILP 资源分配＋物理 LPT 回放延迟的几何平均。CVaR10、按 batch 最坏比值和 200 次 bootstrap 为诊断，统一使用同一 B1 参考向量。选择后硬件冻结，不按留出 batch 更换配置。',
              '', '实际覆盖限制必须同时看 SEARCH_COVERAGE_ZH.md：每组 256 个实评点均来自初始点／seed，分支定界尚未解析任何单点叶子。候选生成只有六套容量总额 profile，容量份额和 bank 份额仍主要按等分／算力比例耦合；声明的独立容量、bank 和数据流大空间尚未被充分覆盖。C0/C1 联合重选对每族使用相同生成预算，不能代替全空间优化。',
              '',table(['模式','约束','族','A：双基线 5%','B：全局零差距','剩余差距 %','开放区域'],
                       [[r['onchip_mode'],r['constraint_group'],r['family'],r['proof_A_closed'],r['proof_B_closed'],f(r['gap_pct']),r['remaining_open_regions']] for r in proofs]),
              '', 'A 的闭合表示可以排除同时比 B1 和 B2 快 5%，不表示该族已找到全局最优。B 未闭合时，本报告只称“等预算搜索中已评估的最好设计”，开放区域和下界保留在 certificates/。内层 CP-SAT 的 OPTIMAL 只证明资源分配松弛最优，LPT 是一个合法回放，不证明联合时序调度全局最优。',
              '', '### 开发集近优候选的留出集鲁棒诊断','',
              f"另对开发集距离各族最好已测点不超过 1% 的 {robust_receipt['group_candidate_rows']} 个组内候选进行留出诊断，按模式和完整物理配置去重为 {robust_receipt['unique_points']} 点。{robust_receipt['reused_points']} 点复用已校验的双遍回放，{robust_receipt['new_points']} 点补做两遍，共新增 {robust_receipt['new_physical_window_passes']} 次逐窗口物理解析回放。",
              '', '以同模式本轮冻结 B1 为共同参照，比较窗口延迟比的几何平均、CVaR10 和最坏 batch；200 次 bootstrap 只重采样开发窗口。结果和后验赢家见 E4/robust_heldout/SUMMARY_ZH.md、robust_objectives.csv、winners.csv 和 selection_stability.csv。留出赢家只作诊断，未替换 selected_designs 或主表，也不是新的盲测选择。',
              '', table(['模式','约束','跨族近优候选数','GM 后验赢家形状','三目标是否同一完整配置'],
                        [[r['onchip_mode'],r['constraint_group'],r['candidate_count'],r['geometry'],
                          r['three_objectives_same_physical_design']] for r in robust_global]),
              '', '### C0 主表：全部为 ms、留出集几何平均，fixed 派工','',
              table(['模式','组织',*[f'B{b}' for b in BATCHES],'全部 GM'],
                    [[r['onchip_mode'],r['design'],*[f(r[f'B{b}']) for b in BATCHES],f(r['all_geomean_ms'])]
                     for r in rows('E4/heldout_main_table.csv') if r['constraint_group']=='C0' and r['dispatch']=='fixed']),
              '', 'MILP 回放、C1、资源积分和双基线置信区间见同目录主表／breakdown／gates。在线/离线和预测器表采用各自注明的控制设置，不能把它们的值混为同一测量。',
              '',table(['异构候选','/B1','/B2','进入校准'],
                       [[r['design'],f(r['ratio_vs_B1'],6),f(r['ratio_vs_B2'],6),r['enter_calibration']] for r in gates]),
              '', '5% 是同时针对两个调优基线的进入校准门槛；10% 和置信区间下界 5% 是校准后的宣称胜出门槛。未做 Rust/native 校准及综合，不能宣称校准后的胜出或面积／能耗收益。',
              '', '### 同协议交叉带宽表','',
              table(['选择带宽','模式','组织','126 ms','256 ms'],
                    [[r['hardware_source'],r['onchip_mode'],r['design'],
                      f(r['at126_geomean_ms']),f(r['at256_geomean_ms'])] for r in rows('E4/cross_bw.csv')]),
              '', '交叉表对新旧硬件统一使用本轮 fixed 和本轮开发集选定参数；单核关闭预测器以保持派工回归逐位一致，双核使用 ours。旧硬件保留原 E4 数据流；历史 7eb58061 的共同 WS／旧 runtime 协议值保留在 E0/E2，不与交叉表强行相等。',
              '', '## 5. 派工与预测器','',
              '新 fixed 在选择时计入重读的共享总线代价；本任务自身重读已经在成本里，只追加对并发其他用户的带宽外部代价，避免重复计费。所有核都需重读时，先选重读倍数最小者，同倍数再比较完成时刻。实际执行仍按有限资源积分计时，预测不释放资源。',
              '',table(['模式','组织','fixed ms','在线/离线','是否 ≤1.01'],
                       [[r['onchip_mode'],r['design'],f(r['all_geomean_ms']),f(r['ratio_vs_milp'],6),float(r['ratio_vs_milp'])<=1.01]
                        for r in dispatch if r['dispatch']=='fixed']),
              '', '所有在线 HBM 比离线多 2% 以上的窗口逐一列在 E5/dispatch/excess_hbm_windows.csv；单核逐位回归见 regression.csv。GPQA B128 的 Shared 归属、开始时刻、字节和延迟见 gpqa_t128_case.md。离线回放是参照，不是物理全局最优，在线比它更快也不矛盾。',
              '', 'H51 的具体准入缺口见 E5/dispatch/DISPATCH_DIAGNOSIS.md：GPQA B128 在 3.691454 ms 绑定一个 Me=18 的任务时，已预测大核完成于 3.766665 ms、小核于 8.632654 ms，但大核暂时不能接单，仍把任务给了小核。两核都无需重读，当前“允许等待”的规则没有覆盖这种情形。该案例不能归因于 ETA 不准；本轮保留原冻结控制逻辑及失败结果，没有用未经评测的新等待规则替换主表。',
              '', '预测器在拉数前估计服务时长和完成时刻，不猜已经由 Router 确定的专家 ID。nominal 不学习，保留与其他方法相同的 25%/50%/75% 进度检查与有限 Next 准入。随机方法随机估计时长，不随机选择核。126 对照沿用旧硬件和 fixed_legacy；256 用新硬件和新 fixed，方法优劣仅在同工作点内配对比较。',
              '',table(['组织','方法','全部 GM ms','/nominal','MAE %','late %','等待空转 %'],
                       [[r['design'],r['method'],f(r['all_geomean_ms']),f(r['ratio_vs_no_pred'],6),
                         f(100*float(r['MAE'])),f(100*float(r['late'])),f(r['stall_pct'])] for r in pred]),
              '', 'oracle 对 ours 冻结行动计划进行独立物理回放，使用真实时长计算误差并重新验证供数、有限池和完成时刻；它不是能改变任务归属的先知调度上界。predictor_table.csv 含分 batch 延迟和整体指标；accuracy_by_batch.csv 含分 batch 的准确率、success@W、late 和 late>64/256 指标。预测更准不保证层延迟更低；慢下来的方法同时报告重读字节、等待和派工变化。',
              '', '## 6. 敏感性与合成反向搜索','',
              table(['参数','Sobol 一阶 S1','S1 置信半宽','总效应 ST','ST 置信半宽'],
                    [[r['param'],f(r['S1']),f(r['S1_ci']),f(r['ST']),f(r['ST_ci'])] for r in rows('E5/sobol/sobol_indices.csv')]),
              '', 'Sobol 使用 N=256、五参数、1,792 个样本，每点重新选几何、数据流、私有／共享容量与端口并重复完整过程。在途额度范围 256–640，其余范围沿第二轮。若搜索证明开放，指数衡量的是等预算搜索程序及其最好已测候选的敏感性，不能声称全局最优硬件的 Sobol 指数。翻转点逐一列在 flip_points.csv。',
              '', '表中保留有限样本的原始估计值和置信半宽，未把负的一阶估计或超过 1 的总效应裁剪为比例。置信区间较宽，不能据此给出精确的重要性排序；总效应也不能相加成延迟归因。硬件重新选择与有限搜索候选的跳变都包含在这个响应函数里。',
              '', '响应量使用同一组 18 个开发窗口：delta = 三个双核比例族（5+1、4+2、3+3）中最好已评估候选的开发集延迟几何平均 / 最好已评估单核的开发集延迟几何平均 − 1。负值仅表示该预算内已测双核候选更快；H33（3+3）允许两核计算形状相同，因此不能把双核族标签解释为严格异构。指数不使用留出窗口选硬件。',
              '', table(['扫描','完整点数','资源模型回放次数（含双遍）','全局零差距证明闭合点数'],
                        [['Sobol',sobol_audit['completed_points'],
                          sobol_audit['physical_simulator_calls_both_repeats'],sobol_audit['closed_global_proofs']],
                         ['合成反向搜索',synthetic_audit['completed_points'],
                          synthetic_audit['physical_simulator_calls_both_repeats'],synthetic_audit['closed_global_proofs']]]),
              '', f"Sobol 的已测 delta 范围为 {f(100*min(float(r['delta']) for r in sobol))}%～{f(100*max(float(r['delta']) for r in sobol))}%，观察到的双核／单核排序反转点为 {sum(float(r['delta'])<0 for r in sobol)} 个。S1／ST／置信半宽从完整 delta 序列独立重算，最大逐值差为 {sobol_audit['maximum_recomputed_index_difference']}；证书、全量索引、源冻结和重复核验见 E5/sobol/COMPLETION_AUDIT.json。",
              '', f"获选双核中计算形状不同的样本有 {sobol_audit['selected_dual_compute_shape_counts']['distinct']} 个、相同的有 {sobol_audit['selected_dual_compute_shape_counts']['same']} 个，后者属于 H33 的同构重叠空间。逐窗口下界另独立检查 {sobol_audit['independent_development_window_lowerbound_checks']} 项，违例 {sobol_audit['lowerbound_violations']}。样本内的局部 5% incumbent 证书只限制该族已测可执行参照点的改进幅度，不是主表同时对 B1/B2 的 5% 胜出门槛。",
              '', '### 合成负载：反转只用于探索，不进入主表','',
              table(['合成 Batch','完整点数','双核已测领先点数','其中计算形状不同','最大已测领先 %','全局证明 B 闭合点数'],synthetic_batches),
              '', '以上领先只是在每族相同有限搜索预算内比较已测候选，不能写成对全局最优单核的胜出。完整逐点区间、硬件、证明和重复记录保留在 E4/synthetic/reverse_search.csv、SYNTHETIC_ZH.md 与 COMPLETION_AUDIT.json。合成数据只探索可能的工作区间，不进入真实留出主表，也不能代替真实模型精度／推理验证。',
              '', '## 7. 验收与局限','',
              '完整自动验收结果：','', '```json',json.dumps(validation,ensure_ascii=False,indent=2),'```','',
              '模型是相位流体解析近似；共享池在事件边界分配有限字节窗口，并未模拟逐 DRAM 请求返回、真实 bank 地址、交叉开关或全部控制电路。端口与计算占用是重叠积分。当前生命周期和下界都不代表将来更改 compiler 融合后的架构。没有 RTL／面积／功耗综合，也没有完整模型每 token 计时。本轮留出集已被此前调试访问，不能当成全新盲测。',
              '', '所有固定设计跨所有 batch 使用同一组硬件和已冻结 runtime 参数。第二轮结果只读引用，不改写；重复与执行命令、输入哈希见各阶段 receipts 和 README.md。']
    lines += ['', '测试程序使用了经过逐位等价验证的运行加速：省去未输出的诊断序列化，缓存静态几何排序，以及将原整数 DFS 按相同遍历、剪枝和并列规则用 C 执行。CP-SAT 的约束和确定性工作预算、两次完整搜索及各窗口的两次物理回放均保留。这里的 C 整数遍历不是 native HBM／RTL 仿真或硬件校准；证据见 diagnostics/performance/ 和 native_enum_conformance 执行回执。']
    (ROOT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write_json(ROOT/'REPORT_METADATA.json',metadata(dict(report_source='completed E0-E5 artifacts',
                         numerical_summary_not_hand_entered=True,validation_sha256=sha(ROOT/'VALIDATION.json'))))
    main_plot()
    readme()


def main_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rs=headline_rows();names=['B1','B2','H51','H42','H33']
    lookup={r['design']:r for r in rs}
    fig,ax=plt.subplots(figsize=(10,4.6));x=np.arange(len(BATCHES));width=.16
    for i,name in enumerate(names):
        ax.bar(x+(i-2)*width,[float(lookup[name][f'B{b}']) for b in BATCHES],width,label=name)
    ax.set_xticks(x,[f'B{b}' for b in BATCHES]);ax.set(ylabel='MoE layer analytical latency (ms, GM)',xlabel='Held-out batch')
    ax.set_title('BF16 pipelined; 256 GB/s service cap; 256-selected C0; fixed dispatch')
    ax.legend(ncol=5);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(ROOT/'figures/fig_main_by_batch.pdf');plt.close(fig)


def readme():
    logs=subprocess.check_output(['git','log','--format=%h %s'],cwd=REPO,text=True).splitlines()
    hashes={stage:next((line.split()[0] for line in logs if ' round3:' in line and line.endswith(f'({stage})')),None)
            for stage in ('E0','E1','E2','E3','E4','E5')}
    commands={
        'E0':'python -m research.moe_dispatch.round3.reproduce --jobs 6',
        'E1':'阅读 E1/OPERATING_POINT.md（纯规格文档）',
        'E2':'python -m research.moe_dispatch.round3.bounds --jobs 4',
        'E3':'python -m research.moe_dispatch.round3.ablation --jobs 3',
        'E4':'python -m research.moe_dispatch.round3.dse --jobs 8 --candidates 256 --nodes 2048；python -m research.moe_dispatch.round3.finalize_dse；python -m research.moe_dispatch.round3.sensitivity --stage synthetic --jobs 8 --candidates 8 --nodes 32；python -m research.moe_dispatch.round3.robust_heldout --jobs 3',
        'E5':'python -m research.moe_dispatch.round3.evaluations --stage dispatch --jobs 6；python -m research.moe_dispatch.round3.evaluations --stage cross --jobs 6；python -m research.moe_dispatch.round3.evaluations --stage predictor --jobs 6；python -m research.moe_dispatch.round3.sensitivity --stage sobol --jobs 32 --candidates 8 --nodes 32'}
    text=['# 第三轮可复现入口','',
          '分支 `research/moe-supply-first-v3`；所有新代码／数据在本目录，第二轮树未修改。使用 Python 3.11 虚拟环境 `/tmp/plena-round2-venv/bin/python`，工作目录为 simulator worktree 根。BF16；解析估计，非 native／RTL。',
          '', '实际 Python、依赖版本和宿主 C 编译器见 ENVIRONMENT.json；requirements-repro.txt 锁定本轮直接依赖。宿主 C 编译器只用于整数搜索加速，不模拟 HBM 或 RTL。',
          '',table(['阶段','提交','运行命令','交付目录'],[[s,hashes[s] or '尚未入提交账本',commands[s],s+'/'] for s in commands]),
          '', '上表命令用实际 Python 路径替换 python。E4 与 E5 的主表共用冻结后的完整回放结果；跨带宽采用同一新控制协议，126 预测器历史对照采用 fixed_legacy。在线派工／预测方法从相同初始状态先运行 18 开发窗口再跑 135 留出窗口；离线 MILP＋LPT 参照没有历史预测状态，不做历史预热。',
          '', '真实执行命令、时间、退出码、日志、源和输入 SHA256 保存在 executions/；被修正的报告或尝试保留原 receipt／archive，不覆盖第二轮证据。',
          '', '数值输出全部完成后的初次收尾顺序：`python -m research.moe_dispatch.round3.validate --partial` → `python -m research.moe_dispatch.round3.report` → `python -m research.moe_dispatch.round3.figure_clarify` → `python -m research.moe_dispatch.round3.validate`。完整验收通过后再生成报告和绘图清单，使报告引用完整验收；已有完整交付可直接运行 validate。测试命令：`python -m pytest research/moe_dispatch/round3 -q`。',
          '', 'REPORT_ZH.md 给出 Q1–Q5、主表与局限。figures/ 包含五份 PDF。全局零差距证明未闭合时，必须引用 proof_status.csv 的开放区域与剩余差距，不能把最好已评估候选称全局最优。']
    (ROOT/'README.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':
    main()
