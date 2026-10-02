#!/usr/bin/env python3
"""Render measured artifacts without promoting incomplete or conditional evidence."""
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import report,study

DESIGN_ORDER=('BL2','BL3','BL4','BL5')

def jsonfile(path):
 return json.loads(path.read_text()) if path.is_file() else {}

def csvfile(path):
 return list(csv.DictReader(path.open())) if path.is_file() else []

def number(value,digits=4):
 return '—' if value is None else f'{float(value):.{digits}f}'

def verdict(value):
 return '尚未完成／不能判定' if value is None else ('通过' if value else '不通过')

def link(path):return f'[{path.name}]({path.resolve()})'

def n5_verdict(timing,quality,physical_format_sha256=None):
 if physical_format_sha256 and quality.get('physical_format_sha256')!=physical_format_sha256:return None
 if timing is None or not quality.get('complete') or quality.get('N5_quality_pass') is None:return None
 return bool(timing and quality['N5_quality_pass'])

def organization_groups(rs):
 grouped=defaultdict(dict)
 for r in rs:
  population=r.get('evaluation_split',r.get('split'))
  if population not in ('heldout','mixed_heldout') or r['suite']!='main' or r['op']!='OP2' or r['port']!='iso' or r['design'] not in DESIGN_ORDER:continue
  grouped[(r['provenance'],int(r['tokens']))].setdefault(r['design'],{})[r['workload']]=r
 result=[]
 for (provenance,tokens),groups in sorted(grouped.items()):
  common=set.intersection(*(set(groups.get(d,{})) for d in DESIGN_ORDER))
  values={d:report.gm([groups[d][w]['ms'] for w in sorted(common)]) if common else None for d in DESIGN_ORDER}
  result.append((provenance,tokens,len(common),values))
 return result

def render(root):
 root=Path(root).resolve();rs=report.rows(root)
 claims=jsonfile(root/'claim_timing_evidence.json');coverage=report.coverage(root)
 signature=jsonfile(root/'active_result_signature.json');fixed=jsonfile(root/'development_fixed_comparator.json')
 ready=jsonfile(root/'development_freeze_ready_receipt.json');fmt=jsonfile(root/'frozen_format.json')
 q3path=root/'quant/full_numerics/q3_qualification_summary.json';q3=jsonfile(q3path)
 q3completion=jsonfile(root/'quant/full_numerics/q3_hardware_complete_receipt.json')
 m0=jsonfile(root/'m0/m0_receipt.json');m5=jsonfile(root/'m5_causal/causal_sequence.json')
 native=jsonfile(root/'compensation_native/receipt.json');final=jsonfile(root/'final_pipeline_receipt.json')
 if claims.get('fixed_comparator',{}).get('signature')!=signature:claims={}
 if fixed.get('signature')!=signature:fixed={}
 if m5.get('signature',{}).get('binary_sha256')!=signature.get('binary_sha256'):m5={}
 if native.get('signature',{}).get('binary_sha256')!=signature.get('binary_sha256'):native={}
 if final.get('frozen_signature')!=signature:final={}
 complete=sum(int(r.get('complete',0)) for r in coverage)
 legal=sum(int(r.get('declared_legal') or 0) for r in coverage)
 excluded=sum(int(r.get('excluded',0)) for r in coverage)
 failed=sum(int(r.get('failed',0))+int(r.get('unsupported',0)) for r in coverage)
 pending=sum(int(r.get('pending_points',0)) for r in coverage)
 main_done=bool(coverage) and not pending and not failed and complete==legal
 state='完整矩阵已完成' if main_done else '进行中；表中只展示已经实际完成的点'
 n1=claims.get('N1',{});n2=claims.get('N2',{});n3=claims.get('N3',{});n4=claims.get('N4',{});n5=claims.get('N5',{})
 evidence=[
  ('N1 量化节省是否兑现',n1.get('passed'),f"单核旧供数 R 最大={number(n1.get('legacy',{}).get('maximum_R_legacy'))}；v3 解码 R 最小={number(n1.get('decode',{}).get('minimum_R_v3'))}；混合 R 最小={number(n1.get('real_mixed',{}).get('minimum_R_v3'))}", '旧单核≤0.40，v3各要求窗口≥0.85'),
  ('N2 秩通道补偿开销',n2.get('passed'),f"{n2.get('observed_comparisons',0)}/{n2.get('expected_comparisons','—')}组配对；P1={verdict(n2.get('P1_primary_timing_passed'))}；P2={verdict(n2.get('P2_primary_timing_passed'))}",'秩通道总核忙开销≤3%，T96层开销≤3%，乘法器≤3.2%；替代方案存在规定损失见证'),
  ('N3 供数机制',n3.get('passed'),f"OP0 B16 η最小={number(n3.get('OP0',{}).get('eta_min'))}；OP1 η最小={number(n3.get('OP1',{}).get('eta_min'))}",'η≥0.95／0.90；每个机制至少一处留一损失≥3%'),
  ('N4 特化大小核组织',n4.get('nominal_passed'),f"混合层时间比={number(n4.get('time_ratio_geomean'))}；搬运比={number(n4.get('traffic_ratio_geomean'))}；面积代理最大差={number(n4.get('maximum_area_difference'))}",'对开发集冻结的最强替代：时间≤0.95，或搬运≤0.85且名义面积差≤5%'),
  ('N5 运行时与动态秩',n5_verdict(n5.get('timing_passed'),q3,signature.get('physical_format_sha256')),f"IPD/joint时间比={number(n5.get('time_ratio_geomean'))}，p95比={number(n5.get('p95_ratio'))}；数值条件={verdict(q3.get('N5_quality_pass'))}",'时间≤0.97或p95≤0.95；另须因果完整总体的等字节误差≤0.90或等误差因子字节≤0.80'),
 ]
 milestone=[]
 suite_complete=defaultdict(int);suite_legal=defaultdict(int)
 for r in coverage:
  suite_complete[r['suite']]+=int(r.get('complete',0));suite_legal[r['suite']]+=int(r.get('declared_legal') or 0)
 def counts(suite):return f"{suite_complete[suite]}/{suite_legal[suite]}合法点，均需两次原始报告"
 milestone.extend([
  ('M0 旧架构计量', '已有冻结诊断' if m0 else '未取得完成收据',link(root/'m0/m0_receipt.json') if m0 else '—'),
  ('M1 BF16供数',counts('ablation'),link(root/'table_ablation_leave_one.csv')),
  ('M2 真实权重量化',fmt.get('quality_status','尚无最终资格文件'),link(root/'frozen_format.json') if fmt else '资格未冻结；不能声称保持精度'),
  ('M3 量化通路与补偿',counts('compensation'),link(root/'claim_N2_compensation_evidence.csv')),
  ('M4 数据流与组织',counts('main'),link(root/'table_organization_paired.csv')),
  ('M5 Runtime',counts('policies')+('；真实LUT双重置序列已验证' if m5.get('two_reset_sequences_identical') else '；实际LUT整合尚未完成'),link(root/'table_policy_paired.csv')),
  ('M6 留出与报告', '完整留出完成' if final.get('status')=='complete' and main_done else f'尚有{pending}个声明点未完成',link(root/'table_coverage.csv')),
 ])
 organizations=[]
 for provenance,tokens,count,values in organization_groups(rs):
  alt=fixed.get('selected');ratio=values['BL4']/values[alt] if values.get('BL4') and values.get(alt) else None
  organizations.append('| '+' | '.join([('真实解码' if provenance=='captured_decode' else '真实混合'),str(tokens),str(count)]+[number(values[d],6) for d in DESIGN_ORDER]+[number(ratio)])+' |')
 selected=fixed.get('selected_by_budget',{})
 comparator=[]
 for d,cycles in fixed.get('geometric_mean_cycles',{}).items():
  comparator.append(f'| {d} | {number(cycles/1e6,6)} | '+('冻结选择' if d in selected.values() else '同预算备选')+' |')
 mechanisms=[]
 for name,v in sorted(n3.get('mechanisms',{}).items()):
  mechanisms.append(f"| {name} | {number(v.get('maximum_latency_loss'))} | {number(v.get('maximum_eta_loss'))} | {verdict(v.get('threshold_pass'))} |")
 quality=fmt.get('quality_status','timing_candidate_not_accuracy_validated')
 lines=[
 '# PLENA-MoE Supply-first v3 实施与评估报告',
 f'状态：**{state}**。主矩阵完成 **{complete}/{legal}** 个合法点、对应 **{complete*2}** 次已验证原始运行；已观察声明排除{excluded}点，失败／不支持{failed}点。矩阵是否完整以覆盖收据为准。',
 '所有时间都是 Rust 有限资源离散事件分析模型的单个 MoE FFN 层时间，起点是路由结果已经就绪，终点是全部专家输出合并完成。Router执行、Attention和全模型生成时间未计入。1 GHz下1周期=1 ns，1,000,000周期=1 ms；主机运行秒数不等于硬件延迟。',
 '## 实际组织及冻结比较对象',
 '尺寸顺序统一为M×N×K。M6组织总主乘法器都是12,288，秩通道另计。共享HBM、落地池和激活存储；两个核有自己的操作数寄存器与有限累加空间。ISO固定任务指定的总供数端口，私有累加读写端口仍随核形状变化并明确计费。',
 '| 组织 | 物理尺寸M×N×K | 数据流 |\n|---|---|---|\n| BL2 单核 | 6×4×512 | 可切换WS/IS，2上下文 |\n| BL3 同构 | 3×4×512＋3×4×512 | 两核可切换WS/IS |\n| BL4 特化大小核 | 4×4×512＋2×4×512 | 大核WS、小核IS |\n| BL5 可切换大小核 | 4×4×512＋2×4×512 | 两核可切换WS/IS |',
 f"开发集12个窗口的比较对象冻结为M6={selected.get('M6','未冻结')}、M8={selected.get('M8','未冻结')}，以后不逐测试窗口选择最快基线。下表是开发集选择证据，不能当作留出结论。",
 '| 开发候选 | 层延迟几何平均ms | 选择 |\n|---|---:|---|\n'+'\n'.join(comparator),
 '## 真实留出结果：相同OP2、ISO端口',
 '每行对同一组真实窗口求几何平均。比值小于1表示BL4更快，大于1表示冻结替代更快。只有四种组织全部完成才形成完整配对；T128属于单列压力测试，N4/N5主要比较T64/96。',
 '| 来源 | Token数T | 已配对窗口 | 单核6 ms | 同构3+3 ms | 特化4+2 ms | 可切换4+2 ms | 特化/冻结替代 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'+('\n'.join(organizations) if organizations else '| 留出尚未运行／完成 | — | 0 | — | — | — | — | — |'),
 '## M0–M6交付证据',
 '实现与测量完成，不代表论文门槛通过；负结果同样保留。',
 '| 里程碑 | 当前完成状态 | 证据 |\n|---|---|---|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in milestone),
 '## N1–N5：按原阈值判定',
 '| 主张 | 判定 | 实测值 | 原判定标准 |\n|---|---|---|---|\n'+'\n'.join(f'| {name} | {verdict(value)} | {values} | {criterion} |' for name,value,values,criterion in evidence),
 f"N4名义判定保持原5%面积代理门槛；类型稳健性另报为 **{verdict(n4.get('area_type_robust_pass'))}**。若名义通过但依赖RF/SRAM类型假设，不能声称无条件等面积硬件优势。N5调度时间判定为 **{verdict(n5.get('timing_passed'))}**，数值资格必须另行完成，不能由时间推断。",
 '### 供数机制的负结果也保留',
 '| 留一机制 | 最大时间损失比例 | 最大η损失比例 | ≥3%门槛 |\n|---|---:|---:|---|\n'+'\n'.join(mechanisms),
 '### 数值与补偿边界',
 f'实际物理格式：`{json.dumps(signature.get("physical_format",{}),sort_keys=True)}`。最终资格状态：**{quality}**。P2使用低位宽MX权重和BF16 X/U/Z，factor_a是低秩矩阵A，不是主输入activation量化；不是W4A4。',
 f"硬件BF16 LUT完整Q3收据：{'已完成' if q3completion.get('complete') else '尚未完成'}；N5数值结论：{q3.get('reason','尚无完整资格结论')}。字节口径：{q3.get('byte_scope','待资格收据明确')}。等字节后验子集、oracle等误差选择、未对齐实际字节的误差收益不能转为完整因果总体的通过。",
 f"N2使用相同位宽/因子格式及相同总wire字节，但padding在专家尾部真实传输，不保留每tile布局、A位置和消费次序，因此不是纯算术单变量隔离。开发集native-wire敏感性完成{native.get('native_points',0)}/42点、{native.get('native_raw_runs',0)}原始运行；见{link(root/'compensation_native/native_wire_sensitivity.csv')}。B-MXINT4补充组具有独立数值条件，不能继承默认B-BF16资格。",
 'M5真实LUT的六窗口因活动专家字节预算改变而每次重置λ0，没有跨窗口自学习收敛证据；单个真实窗口的temporal replay只验证相同预算的因果λ传递，不增加独立样本。实际预算超支、不同rank及负的时间收益均保留于causal_sequence.csv。',
 '## 覆盖、可复现性与局限',
 f"最终binary SHA：`{signature.get('binary_sha256','未冻结')}`。Timing signature包含实际编译器源码、输入、五项物理格式和runner；质量文件SHA独立。每个合法点的两次完整原始JSON必须字节相同，gzip解压后哈希仍校验。预注册提交：`{final.get('prereg_commit','尚未完成正式留出收据')}`。",
 f'全覆盖见{link(root/"table_coverage.csv")}；精确原始运行和不可用组合见每点receipt/unavailable。存储迁移仅改变文件所在文件系统，不改变逻辑路径、输入、格式或阈值；{link(root/"whole_output_storage_relocation_receipt.json")}包含文件SHA核验，{link(root/"storage_resume_audit_v6.json")}记录中断报告保留。早期失败候选和ENOSPC尝试保留，不能混入当前通过的时序点。',
 '真实混合留出27窗口来自只有3个独立prefill请求，并跨层/长度复用，不能报告为27个独立模型请求。混合窗口同时就绪的FFN集合不等于完整连续batching的到达过程。没有原生HBM/Ramulator、RTL频率、SRAM宏面积、功耗或整模型tokens/J实测；面积与能量只提供明确口径的相对代理。局部WOR/XOR广播端口和RF扇出没有综合证据，另作敏感性。',
 '结论只采用上述已经完成的原阈值判定。若异构组织、Runtime或数值条件不通过，降低对应主张，保留实测数据和适用范围；不通过单独调参、更换测试配置或缩小矩阵制造胜出。',
 ]
 path=root/'REPORT_V3.md';path.write_text('\n\n'.join(lines)+'\n');return path

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(render(a.root))
