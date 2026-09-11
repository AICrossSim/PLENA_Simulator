#!/usr/bin/env python3
"""Render a concise Chinese evidence report from completed experiment artifacts."""
import csv,json,shutil
from pathlib import Path
from run import OUT,sha,save

def loadcsv(p):return list(csv.DictReader(p.open()))

def main():
    m=json.loads((OUT/'manifest.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    adaptive=json.loads((OUT/'adaptive_dispatch/manifest.json').read_text())
    assert m['status']==adaptive['status']=='passed'
    rows=loadcsv(OUT/'sensitivity.csv');by={(r['case'],r['profile']):r for r in rows}
    ar=loadcsv(OUT/'adaptive_dispatch/measurements.csv')
    ab={(r['case'],r['profile']):r for r in ar if r['repeat']=='1'}
    for r in ar:
        base=by[(r['case'],'baseline')]
        assert r['useful_macs']==base['useful_macs']
    lines=['# PLENA MoE：计算与供数瓶颈诊断（2026-09-11）','',
      '**当前配置主要受供数、片上端口和控制/反馈限制；接近计算受限的理想供数对照也已测到，但本次相同预算的大小核未胜过同样优化的单核。动态派工还暴露出小核接到不合适的大任务、核内扫描开销过高的问题。**','',
      f"完成 {m['completed_runs']} 次固定派工实验（220 个点，各重复 2 次），以及 {adaptive['completed_runs']} 次恢复原有 work-conserving 派工的补充实验。全部数值输出与独立 golden 完全一致，重复结果一致；所有 native HBM 请求均完成。另有 46 项 MoE Rust 测试和 4 项 HBM 诊断测试通过。",'',
      '**结论边界：这是现有 Compiler 导出的 MoE 算子在 Rust 中的数值与时序实验；使用归档路由、合成权重和解析计算时序。没有运行完整模型或测量实物芯片。诊断中的“2×”是改变服务时间，不能当成同面积硬件的实际收益。**','',
      '## 1. 测了什么','',
      '- 单专家 Me=1、8、32：单核 N3。多专家 B8、B32：单核 N3、单核 Q32、大小核 N2、大小核 Q32。',
      '- N3/N2 表示旧实现的 3/2 个活跃输出 N tile；Q32 表示每核 32 个独立的 (M块,N块) 输出上下文、与两个 operand 暂存分离。这里 N3 不表示逻辑矩阵 N=3。',
      '- B8：17 个活跃专家、64 个 token→expert 分配，Me=1～8。B32：23 个活跃专家、256 个分配，Me=1～30。',
      '- gate/up 的逻辑 (M,K,N)=(Me,2048,512)；down=(Me,512,2048)。所有对照固定输入与路由、权重地址、精度和 SRAM 容量。',
      '- 单核物理 (P,R,Mt)=(8,512,4)；大小核为 (6,512,4)+(4,256,1)，合计均为 4096 个乘法器。Mt 是时间上处理的行数，不是额外的物理 PE 轴。',
      '- 合计权重 SRAM 64 KiB、向量 SRAM 4 MiB、accumulator SRAM 1 MiB；共用 8 个 native HBM 控制器，32B native 事务；core 时钟固定 1ns。',
      '- 权重为本地 E4M3 elements + E8M0 scales、block8，按原 MLEN 读取并实际解码成 BF16。格式与内存布局不变。','',
      '## 2. 分别加快哪里，总时间会怎样','',
      '下表为原时间÷修改后时间；1.00× 表示几乎没有收益。大小核 N2 的任务分配固定，以隔离单项资源。','',
      '| 输入 / 配置 | 原时间 μs | MAC 服务 2× | HBM 时钟 2× | HBM+DMA 2× | accumulator 端口 2× |','|---|---:|---:|---:|---:|---:|']
    selected=[('单专家 Me=1','expert_me1_single_legacy_n3'),('单专家 Me=8','expert_me8_single_legacy_n3'),('单专家 Me=32','expert_me32_single_legacy_n3'),('B8 单核 N3','qwen_full_decode_b8_single_legacy_n3'),('B8 大小核 N2','qwen_full_decode_b8_heterogeneous_legacy_n2'),('B32 单核 N3','qwen_full_decode_b32_single_legacy_n3'),('B32 大小核 N2','qwen_full_decode_b32_heterogeneous_legacy_n2')]
    for label,c in selected:
        values=[f"{float(by[(c,p)]['speedup']):.3f}×" for p in ['mac2','hbm_clock2','hbm_clock_dma2','accumulator2']]
        lines.append('| '+label+f" | {float(by[(c,'baseline')]['total_us']):.3f} | "+' | '.join(values)+' |')
    lines+=['','小 Me 对 HBM 供数时序更敏感；Me=32 时，accumulator 端口的影响明显增加。只加快 MAC 的收益有限，说明当前没有把 MAC 吞吐用满。应称为随任务变化的供数、片上端口和反馈限制，不能统一称为“HBM 带宽已经跑满”。','',
      'HBM 时钟 2× 同时改变存储器域的延迟和吞吐上限，DMA 入口保持原速；它不是纯带宽变量。另已分别测试列命令间隔、返回延迟、全部 bank 时序，以及 DMA、权重端口、activation、共享 vector、调度器；全部结果在 sensitivity.csv。列间隔变短后仍有 controller 命令发射与 DMA 入口限制，因此不保证实际带宽翻倍。','',
      '## 3. 接近计算受限时，大小核是否获益','',
      '“理想供数”移除 native HBM 及 native 入口等待，并把非 MAC 的供数、端口、vector、调度服务加快 64×；保留有限 SRAM、槽位、数据依赖及 MAC 反馈。它用于找瓶颈，不是可制造配置，也不等于零开销理论下界。','',
      '| 输入 / 同为 Q32 | 理想供数时间 μs | 再加快 MAC 2× 后 μs | MAC 加速效果 |','|---|---:|---:|---:|']
    for b in (8,32):
        for org,label in [('single','单核'),('heterogeneous','大小核')]:
            c=f'qwen_full_decode_b{b}_{org}_pool_q32'
            a=float(by[(c,'ideal_supply64')]['total_us']);z=float(by[(c,'ideal_supply64_mac2')]['total_us'])
            lines.append(f'| B{b} {label} | {a:.3f} | {z:.3f} | {a/z:.3f}× |')
    lines+=['','这组对照可检验计算服务的敏感性；必须同时比较同为 Q32 的单核。单核也能用较多独立输出隐藏反馈等待。理想供数下，单核 Q32 的 useful MAC / (4096 × 总周期) 在 B8/B32 约为 91.5% / 94.6%，接近本模型的计算吞吐上限；实际资源配置远低于这一水平。当前 N/K 与单核 P/R 对齐，单核无 N/K padding 浪费；大小核的总 PE 相同，拆分本身没有额外算力。物理 P/R 与逻辑 M/N/K 必须分开讨论；尤其不能将小 Me 自动解释成大核有大量物理 M 维 PE 闲置。','',
      '## 4. 公平比较双方的结果','',
      '每列在本次两个单核实现、两个大小核实现中各取较短时间，不代表全设计空间最优。原配置中的单核与大小核总资源预算一致。各诊断配置让双方接受相同服务加速，但不估算加速所需面积和功耗。','',
      '| 场景 | B8 单核 / 大小核 μs | B32 单核 / 大小核 μs |','|---|---:|---:|']
    for profile,label in [('baseline','原配置'),('hbm_clock_dma2','HBM+DMA 2×，固定派工'),('all2','供数/控制与 MAC 2×，固定派工'),('ideal_supply64','理想供数，固定派工')]:
        cells=[]
        for b in (8,32):
            single=min(float(by[(f'qwen_full_decode_b{b}_single_{mode}',profile)]['total_us']) for mode in ('legacy_n3','pool_q32'))
            dual=min(float(by[(f'qwen_full_decode_b{b}_heterogeneous_{mode}',profile)]['total_us']) for mode in ('legacy_n2','pool_q32'))
            cells.append(f'{single:.3f} / {dual:.3f}')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    for profile,label in [('hbm_clock_dma2','HBM+DMA 2×，重新派工'),('all2','供数/控制与 MAC 2×，重新派工'),('ideal_supply64','理想供数，重新派工')]:
        cells=[]
        for b in (8,32):
            single=min(float(ab[(f'qwen_full_decode_b{b}_single_{mode}',profile)]['total_us']) for mode in ('legacy_n3','pool_q32'))
            dual=min(float(ab[(f'qwen_full_decode_b{b}_heterogeneous_{mode}',profile)]['total_us']) for mode in ('legacy_n2','pool_q32'))
            cells.append(f'{single:.3f} / {dual:.3f}')
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','重新派工使用现有的 M 阈值偏好及空闲核心取任务规则，未进行最优任务分配搜索。它可检查原结论是否仅由固定分配造成；不能排除更好的派工或其他 M/N/K 形状下存在异构优势。','',
      '### 重新派工暴露出的具体限制','',
      'B32、大小核 Q32、HBM+DMA 2×：固定派工为 1.139ms，恢复原有 work-conserving 派工后为 3.091ms。小核在约 0.932ms 时接走 expert 185（Me=30），该任务到约 3.090ms 才完成。',
      '小核 Mt=1，因此该专家需要 30 个 M 上下文；Q32 容量允许它执行，但核内选择逻辑每次从较小 M 索引逐项扫描、每次检查计 1 周期。小核整个窗口的 scheduler 服务从约 0.597ms 增至 2.499ms。计数会与后台搬运重叠，不能直接相加；但任务归属和扫描成本清楚暴露了只按“空闲且放得下”派工的限制。',
      '这不是新的 HBM 配置变慢；同一输入、同一 HBM/端口服务，仅恢复动态派工就触发了较差的任务放置。对应结果在 adaptive_dispatch/qwen_full_decode_b32_heterogeneous_pool_q32/hbm_clock_dma2.rep1.json；核内扫描见 output_pool.rs，派工见 engine.rs。','',
      '## 5. 下一步架构工作','',
      '1. 先针对现有数据证实的供数等待、accumulator 服务和反馈等待优化，并给单核应用同样优化。把 HBM 返回、DMA 入队/复制、解码、SRAM 端口等时间分开记录，不用 weight_ready_wait 直接推断 HBM 带宽饱和。',
      '2. 派工应考虑任务实际 (Me,N,K)、两核的服务时间和排队、剩余槽位、预计供数完成时间。以预计完成时间选择核心，联合控制预取，检验是否改善末尾慢任务；核内使用有容量和成本建模的 ready 位图/队列或旋转游标，避免重复扫描已不可发射的上下文。现有仅按 M 偏好并取空闲任务的规则不构成完整架构机制。',
      '3. 单独检验可产生形状匹配收益的 N/K 尾部和 M 分布，分别扫描逻辑任务形状与物理 P/R/Mt，保持总 PE、SRAM 和端口预算公平。仅改变成 compute-bound 并不自动产生大小核收益。','',
      '## 6. 校验与复现','',
      f"- {summary['independent_port_service_checks']} 项独立端口服务审计通过；所有固定派工对照的 useful/issued MAC 数不变。HBM 字节发生变化的运行数：{summary['hbm_bytes_changed_runs']}。",
      '- 11 组默认配置的两次重复都精确复现旧归档总周期。更改时序的每次结果核对 BF16、FP32 和最终舍入前 FP32 三份输出。重复检查完整 result 和 native telemetry。',
      '- 主实验 manifest.json、measurements.csv、sensitivity.csv、core_services.csv；补充实验 adaptive_dispatch/manifest.json 与 measurements.csv。每个点保留 architecture、完整结果、日志和 SHA256。',
      '- repro/ 保存实际使用的 binary、native library、源码 patch、测试日志和 runner。旧工作树及旧结果保留。本次未改 Compiler 导出契约，未涉及 RTL，也没有创建/更新线上 PR。',
      '- native 微测试验证默认首读 30ns、仅返回延迟减半后 22ns、HBM 时钟减半后 15ns；事务大小始终 32B。512 事务流受入口发射率约束，列间隔减半不等于吞吐翻倍。',
      '- 初次准备因误用系统 Python 3.6 失败，未运行实验；随后使用项目 Python 3.11。主实验从 2 个进程改为 8 个进程，仅改变宿主并行度，未改冻结输入、二进制或模拟时序；重复一致性覆盖这一点。','',
      '![汇总](bottleneck_summary.png)','']
    (OUT/'RESULT_ZH.md').write_text('\n'.join(lines))
    src=Path(__file__).parent
    for name in ('run.py','adaptive.py','analyze.py','plot.py','report.py','README.md'):
        shutil.copy2(src/name,OUT/'repro'/('final_'+name))
    save(OUT/'repro/final_scripts.json',{name:sha(src/name) for name in ('run.py','adaptive.py','analyze.py','plot.py','report.py','README.md')})
    print('WROTE',OUT/'RESULT_ZH.md')

if __name__=='__main__':main()
