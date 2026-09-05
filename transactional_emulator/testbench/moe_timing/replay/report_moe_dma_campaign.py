#!/usr/bin/env python3
"""Publish only complete, gated DMA, fairness and optimistic-bypass campaigns."""
import argparse
import csv
from pathlib import Path
from compare_moe_normal import read_json, digest, require
from run_moe_dma_campaign import save

SINGLE_PREFIX='single_'
HET='heterogeneous_b16_k192_b8_k128'
HOMO='homogeneous_b16_k128'
FIXTURES=['qwen_full_decode_b8','qwen_full_decode_b32','deepseek_full_decode_b8','deepseek_full_decode_b32']
LABELS=['Qwen B8','Qwen B32','DeepSeek B8','DeepSeek B32']


def build(root):
    plans=[(root, 'campaign_status.json', 17), (root/'fairness','status.json',8), (root/'bypass','status.json',8)]
    all_rows=[]; groups={}; runs=0; input_hashes={}; native_hashes=set()
    for folder,status_file,expected in plans:
        status=read_json(folder/status_file)
        require(status['status']=='passed' and status['all_gates_passed'] and len(status['outcomes'])==expected,
                'incomplete required campaign: '+str(folder))
        for name,outcome in status['outcomes'].items():
            require(outcome['status']=='passed','failed point: '+name)
            path=folder/'comparisons'/name/'comparison.json'; comparison=read_json(path)
            require(comparison['all_gates_passed'] and comparison['status']=='passed' and comparison['repeats']==2,
                    'comparison gate/repeat mismatch')
            require(len(comparison['comparisons'])==6,'architecture coverage incomplete')
            raw=list(Path(comparison['run_directory']).glob('arch*_repeat*.json'))
            require(len(raw)==12,'raw repeat coverage incomplete')
            input_hashes[str(path)]=digest(path);runs+=len(raw)
            fixture=next(f for f in FIXTURES if name.startswith(f+'_'));stage=name[len(fixture)+1:]
            groups[(fixture,stage)]={r['architecture']['name']:r for r in comparison['comparisons']}
            for row in comparison['comparisons']:
                require(all(g['passed'] and g['output_bit_exact'] and g['max_absolute_error']==0 for g in row['gates']),
                        'final report requires bit-exact nonzero MoE validation')
                s=row['result'];n=row['native'];native_hashes.add(row['native_library_sha256'])
                controllers=n['native_stats']['memory_system']['controller']
                requests=sum(c['num_read_reqs_served'] for c in controllers)
                require(requests*32==s['hbm_read_bytes'],'native byte mismatch at report time')
                front=s.get('dma_frontend') or {}
                all_rows.append(dict(fixture=fixture,stage=stage,architecture=row['architecture']['name'],
                    total_ms=s['total_ps']/1e9,hbm_MB=s['hbm_read_bytes']/1e6,
                    achieved_GBps=s['hbm_read_bytes']*1000/s['total_ps'],native_reads=requests,
                    native_rejections=sum(n['rejected_per_channel']),
                    native_mean_read_latency_cycles=sum(c['read_latency'] for c in controllers)/max(requests,1),
                    dma_line_requests=front.get('line_requests',0),merged_sectors=front.get('merged_sectors',0),
                    useful_copy_bytes=front.get('useful_copy_bytes',0),
                    weight_ready_wait_sum_ms=sum(c['weight_ready_wait_ps'] for c in s['cores'])/1e9,
                    compute_busy_sum_ms=sum(c['compute_busy_ps'] for c in s['cores'])/1e9,
                    dma_reserved_bytes=front.get('reserved_bytes',0)))
    require(runs==396 and len(native_hashes)==1,'full coverage or common native library failed')
    with (root/'all_points.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    def time_of(f,stage,arch):return groups[(f,stage)][arch]['result']['total_ps']
    def best(f,arches,stages):
        choices=[(time_of(f,s,a),a,s) for s in stages if (f,s) in groups for a in arches]
        return min(choices)
    summaries=[]
    for f in FIXTURES:
        singles=[a for a in groups[(f,'reference')] if a.startswith(SINGLE_PREFIX)]
        stages=['reference','candidate','scale_layout','reserved','port_only','sector','coalesce','credits128','slots3']
        single=best(f,singles,stages);hetero=best(f,[HET],stages);homo=best(f,[HOMO],stages)
        direct=best(f,singles,['direct2','direct4'])
        summaries.append(dict(fixture=f,best_single=dict(ps=single[0],architecture=single[1],stage=single[2]),
            best_heterogeneous=dict(ps=hetero[0],stage=hetero[2]),best_homogeneous=dict(ps=homo[0],stage=homo[2]),
            heterogeneous_speedup_vs_best_single=single[0]/hetero[0],
            heterogeneous_time_overhead_percent=(hetero[0]/single[0]-1)*100,
            hetero_dma_speedup_vs_counted_reference=time_of(f,'reference',HET)/hetero[0],
            hetero_dma_speedup_vs_optimistic_direct2=time_of(f,'direct2',HET)/hetero[0],
            hetero_dma_speedup_vs_optimistic_direct4=time_of(f,'direct4',HET)/hetero[0],
            optimistic_direct_best_single_ps=direct[0],
            fair_heterogeneous_change_percent=(time_of(f,'reserved',HET)/time_of(f,'control',HET)-1)*100))
    wins=sum(s['heterogeneous_speedup_vs_best_single']>1 for s in summaries)
    save(root/'summary.json',dict(all_gates_passed=True,runs=runs,comparison_groups=33,
        heterogeneous_wins=wins,results=summaries,native_library_sha256=next(iter(native_hashes)),
        report_driver_sha256=digest(__file__),comparison_artifact_sha256=input_hashes))
    lines=['# MoE DMA 实现与完整尺寸 Rust 重测', '',
        '**结论：搬运优化已有可重复的数值执行证据；当前固定大小核配置在所测窗口中胜过最佳单核 %d/4。**'%wins,
        '所有数字均为同一个已校准 Ramulator 后端与分析式核心时序模型的结果，不能直接当成 RTL/芯片或完整模型加速。', '',
        '完成 33 组对照、396 次运行；四种单核、同构双核、异构双核，每点重复两次。全部输出与独立 oracle 位精确相同；原生 32B 请求完成数、实际读取字节、SRAM/端口额度和重复性检查通过。', '',
        '## 大小核是否成立', '',
        '以下在每个架构已经测试的配置中选最短时间。三个双核/单核类别均允许使用相同优化。DeepSeek B32 还包含逐项消融点；没有宣称已经搜索所有核心形状。直接路径的乐观对照单列，不与完整计费路径混为一种硬件实现。', '',
        '| 窗口 | 最佳单核 ms | 当前大小核最佳 ms | 大小核耗时增加 | 大小核最佳配置 |',
        '|---|---:|---:|---:|---|']
    for label,s in zip(LABELS,summaries):
        lines.append('| %s | %.4f | %.4f | %.1f%% | %s |'%(label,s['best_single']['ps']/1e9,s['best_heterogeneous']['ps']/1e9,s['heterogeneous_time_overhead_percent'],s['best_heterogeneous']['stage']))
    lines+=['','最佳单核形状及配置：','']
    lines+=['- %s：`%s` / `%s`。'%(label,s['best_single']['architecture'],s['best_single']['stage']) for label,s in zip(LABELS,summaries)]
    lines+=['','## 搬运本身是否有益','','同一个大小核，逐步改善 DMA 后与三种对照比较；倍数大于 1 表示改进更快。direct2/direct4 不计查询和返回复制端口服务，故是有利于对照的乐观边界，不是完整硬件计时。', '',
        '| 窗口 | 对完整计费 2-slot 对照 | 对乐观 direct2 | 对乐观 direct4 | 保留额度开关的耗时变化 |',
        '|---|---:|---:|---:|---:|']
    for label,s in zip(LABELS,summaries):
        lines.append('| %s | %.3f× | %.3f× | %.3f× | %+.2f%% |'%(label,s['hetero_dma_speedup_vs_counted_reference'],s['hetero_dma_speedup_vs_optimistic_direct2'],s['hetero_dma_speedup_vs_optimistic_direct4'],s['fair_heterogeneous_change_percent']))
    lines+=['','正的保留额度耗时变化表示变慢，不能算作加速。控制组关闭该开关后，逐项复现上一二进制的所有时间、输出与原生后端计数。', '',
        '## 各项贡献：DeepSeek B32','','| 阶段 | B8/K512 单核 ms | 同构双核 ms | 大小核 ms | 大小核读取 MB |', '|---|---:|---:|---:|---:|']
    f=FIXTURES[-1]
    for stage in ['direct2','reference','port_only','sector','coalesce','credits128','slots3','candidate','scale_layout','reserved','direct4']:
        lines.append('| %s | %.4f | %.4f | %.4f | %.3f |'%(stage,time_of(f,stage,'single_b8_k512')/1e9,time_of(f,stage,HOMO)/1e9,time_of(f,stage,HET)/1e9,groups[(f,stage)][HET]['result']['hbm_read_bytes']/1e6))
    lines+=['','阶段说明：reference=64 credits/2 slots/完整行/全局队列；port_only 仅分通道；sector 仅请求需要的 32B 半行；coalesce 再加在途合并；credits128 扩大在途窗口；slots3=3 个预取槽；candidate=4 槽；scale_layout 再旋转 scale 行通道；reserved 是原布局 candidate 加可借用的每核最低额度。所有完整 DMA 计费配置总预算均为 44KiB，staging 包含在其中。', '',
        '## 为何大小核仍可能更慢','','HBM 峰值没有增加。以 Qwen B8 的原布局 candidate 为例：','']
    for arch in ['single_b8_k512',HET]:
        r=groups[(FIXTURES[0],'candidate')][arch]['result']
        lines.append('- `%s`：读取 %.3f MB，平均 %.2f GB/s，耗时 %.4f ms。'%(arch,r['hbm_read_bytes']/1e6,r['hbm_read_bytes']*1000/r['total_ps'],r['total_ps']/1e9))
    lines+=['','这说明当前异构切块同时承受额外事务与较低的有效吞吐；并非仅仅把同一份计算平均分给两个核。局部 MX block8 下，K=192 的每行 scale 是 24B，K=128 是 16B，均会与 32B 原生事务发生碎片；K=512 的 scale 是 64B。部分浪费能由在途合并消除，但收益受预取窗口及端口服务限制。这是对计数与布局的解释，尚不能把全部剩余耗时归因于某一个具体物理电路。', '',
        '## 正确的下一步','','1. 保留校准、计数、有限预取、sector/MSHR 和布局开关；逐点选择已验证的配置，不把所有开关默认打开。',
        '2. 当前大小核参数不作为已成功的硬件设计冻结。以数据搬运完成时间为约束，重新联合搜索核心形状、K 与 scale 原生事务对齐、每核 SRAM 分配和专家 M 分布；同构双核也需扩大形状搜索。',
        '3. 先校准各 MLEN 的激活/权重端口吞吐与 MAC 流水时序，再谈等面积/等功耗优势或 RTL。不能用相同乘法器数代替物理成本比较。',
        '4. E/S deadline/DRR 专门仲裁、持久 scale cache、transpose、输入/输出 DMA 竞争和完整模型运行仍未实现；由等待/事务指标决定优先级，而非预设它们一定能救回大小核。', '',
        '## 证据边界与复现','','- Qwen：D=2048、F=512；DeepSeek：D=2048、F=1408。B8/B32 使用归档 decode 路由重组，原记录为 batch16；不是完整 batch32 实际 forward 或 prefill。',
        '- 权重和输入为确定性非零合成值，使用仓库实际 E4M3/E8M0 block8 codec、Rust 三个 GEMM/SwiGLU/加权 combine。没有执行训练权重的完整模型。',
        '- 端到端边界为输入/路由已经 ready、权重驻留 HBM，至 BF16 输出 ready；不含 router、初始装载和输出写回。核心时序仍为分析模型。',
        '- 查询 bank 本轮保守地占用 2 个周期，尚未建模原设计的 2-cycle latency / 1-cycle initiation interval；需要 SRAM 端口与流水线校准，不能把本轮代价当成物理实现定值。',
        '- 与旧 16B wrapper 报告不直接计算加速比；旧数据保留。',
        '- `all_points.csv`：198 个配置条目（含关闭开关的复现对照）；`summary.json`：完整覆盖验收和报告输入哈希；`comparisons/`、`fairness/`、`bypass/`：原始两次重复。',
        '- `repro/`：实际两个 Rust 二进制与同一原生库、构建环境、源码证据和测试日志；`doc/moe_dma_implementation_v1_zh.md`：接口/资源与未实现项。','']
    (root/'RESULT_ZH.md').write_text('\n'.join(lines))
    print('Published',root/'RESULT_ZH.md', 'runs=',runs,'heterogeneous_wins=',wins)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    build(p.parse_args().root.resolve())
