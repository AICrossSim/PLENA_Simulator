#!/usr/bin/env python3
"""Publish a complete report only after primary and adaptive evidence closes."""
import argparse
import csv
from pathlib import Path
from compare_moe_normal import read_json, require, digest
from run_moe_full_campaign import save


def report(root):
    for name in ['campaign_status.json','cache_followup_status.json']:
        require(read_json(root/name)['all_gates_passed'] is True, name+' is incomplete')
    manifest=read_json(root/'campaign.json'); fixtures=manifest['fixtures']
    main={n:n for n in fixtures}
    bypass={n:n+'_cache_disabled' for n in fixtures}
    bypass['deepseek_full_decode_b32']='deepseek_b32_cache_disabled'
    required=list(main.values())+list(bypass.values())+['deepseek_b32_legacy_serialized']
    summaries={n:read_json(root/'comparisons'/n/'comparison.json') for n in required}
    for name,s in summaries.items():
        require(s['all_gates_passed'] is True and s['status']=='passed',name+' failed')
        require([r['architecture']['name'] for r in s['comparisons']]==manifest['architectures'],
                'incomplete architecture coverage: '+name)
        require(s['repeats']==2 and all(len(r['gates'])==2 and all(g['passed'] for g in r['gates'])
                    for r in s['comparisons']), 'incomplete gates: '+name)
    require(len({s['executable_sha256'] for s in summaries.values()})==1,'binary identity differs')
    for n in fixtures:
        require(summaries[main[n]]['workload_sha256']==summaries[bypass[n]]['workload_sha256'],
                'cache comparison workload differs')
    singles=[n for n in manifest['architectures'] if n.startswith('single_')]
    homo='homogeneous_b16_k128';hetero='heterogeneous_b16_k192_b8_k128'
    def index(name):return {r['architecture']['name']:r for r in summaries[name]['comparisons']}
    def elapsed(row):return row['result']['total_ps']
    compact=[]; core_rows=[]; decisions=[]
    for name,s in summaries.items():
        rows=index(name);best=min((rows[n] for n in singles),key=elapsed)
        for arch,row in rows.items():
            r=row['result'];clock=row['architecture']['clock_period_ps']
            compact.append(dict(comparison=name,architecture=arch,total_ms=r['total_ps']/1e9,
                fastest_tested_single=best['architecture']['name'],speedup_vs_fastest_single=elapsed(best)/elapsed(row),
                useful_macs=r['useful_macs'],issued_macs=r['issued_macs'],
                global_useful_mac_utilization=r['useful_macs']*clock/(r['multipliers']*r['total_ps']),
                useful_fraction_of_issued_macs=r['useful_macs']/r['issued_macs'],hbm_read_bytes=r['hbm_read_bytes'],
                effective_hbm_GBps_over_operator=r['hbm_read_bytes']*1000/r['total_ps'],
                all_outputs_bit_exact=all(g['output_bit_exact'] for g in row['gates'])))
            for c in r['cores']:core_rows.append(dict(comparison=name,architecture=arch,**c))
    for n in fixtures:
        cached=index(main[n]);uncached=index(bypass[n])
        best=min([x[k] for x in [cached,uncached] for k in singles],key=elapsed)
        pair=min([cached[hetero],uncached[hetero]],key=elapsed)
        equal=min([cached[homo],uncached[homo]],key=elapsed)
        decisions.append(dict(fixture=n,fastest_tested_single=best['architecture']['name'],
            single_ms=elapsed(best)/1e9,heterogeneous_ms=elapsed(pair)/1e9,homogeneous_ms=elapsed(equal)/1e9,
            heterogeneous_speedup=elapsed(best)/elapsed(pair),homogeneous_speedup=elapsed(best)/elapsed(equal),
            heterogeneous_vs_homogeneous=elapsed(equal)/elapsed(pair)))
    def write_csv(path,rows):
        with path.open('w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_csv(root/'architecture_metrics.csv',compact);write_csv(root/'core_metrics.csv',core_rows)
    validated_runs=sum(len(r['gates']) for s in summaries.values() for r in s['comparisons'])
    bit_exact=all(r['all_outputs_bit_exact'] for r in compact)
    save(root/'benefit_summary.json',dict(all_gates_passed=True,comparisons=len(summaries),
        validated_runs=validated_runs, all_outputs_bit_exact=bit_exact,
        decisions=decisions,metrics=compact,
        input_comparisons=[dict(path=str(root/'comparisons'/n/'comparison.json'),
                               sha256=digest(root/'comparisons'/n/'comparison.json')) for n in required]))
    wins=sum(d['heterogeneous_speedup']>1 for d in decisions)
    hetero_wins=sum(d['heterogeneous_vs_homogeneous']>1 for d in decisions)
    conclusion=(f'结论：当前 3072+1024 大小核在四组窗口中胜过强单核 {wins}/4 组、胜过同构双核 {hetero_wins}/4 组。'
                + ('本次未证明这组大小核有收益，不冻结该比例；这不等于否定所有异构配置。' if wins==0 else '收益范围以以下数据和计时假设为限。'))
    lines=['# MoE 大小核 Rust 完整尺寸验证结果（2026-09-05）','',conclusion,'',
        '## 完整 goal 与完成范围','',
        '在同一共享 HBM、相同 4096 个矩阵乘法器和相同总 SRAM 预算下，完整执行固定路由 MoE 的 Gate/Up → SwiGLU → Down → 加权合并；比较单核、同构双核和大小核的耗时、利用率与资源占用。必须数值正确、资源不越界、重复运行确定；允许结论为没有收益。','',
        '已实现独立 normal buffer / accumulator、有限双缓冲、共享 DMA、动态接任务、可关闭的有限缓存及两种矩阵计时。入口是独立 Rust 数值实验 `moe_dual_normal`；尚未接入旧 ISA，也未实现 transpose/attention/RTL。','',
        '## 如何读取结果','',
        '下表对每种核配置分别取“启用缓存/关闭缓存”中较快的一种，并让单核在四种形状中取最快者，避免把缓存本身的额外开销误算成大小核优势。每个启用/关闭缓存的比较组内部都保持相同资源预算；关闭组所有核均不配置缓存。跨组最优值用于排除较弱基线，不能声称两组 SRAM 配置完全一样。加速比大于 1 才更快；1.25× 对应耗时下降 20%。','',
        '| 负载 | 最快单核形状 | 单核 ms | 同构双核 ms | 大小核 ms | 大小核/单核加速比 | 大小核/同构加速比 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for d in decisions:
        lines.append('| {fixture} | {fastest_tested_single} | {single_ms:.6f} | {homogeneous_ms:.6f} | {heterogeneous_ms:.6f} | {heterogeneous_speedup:.4f}× | {heterogeneous_vs_homogeneous:.4f}× |'.format(**d))
    lines+=['',f'在这四组有限配置和路由窗口中，优化后的大小核胜过最快单核 {wins}/4 组，胜过同构双核 {hetero_wins}/4 组。这是该 Rust 时序模型中的结果，不是已验证硬件加速，也不能外推为最优架构或论文 novelty。','',
        '## 全部主比较与缓存对照','',
        '四种单核分别为 B32/K128、B16/K256、B8/K512、B4/K1024；同构为两个 B16/K128；大小核为 B16/K192 + B8/K128。最后一行保留固定阈值分配，其余使用有空闲核就继续接任务的调度。下表单位均为 ms，未只选有利配置。','']
    for label,mapping in [('主比较：每核一个缓存端口',main),('对照：全部关闭缓存',bypass)]:
        lines+=['### '+label,'','| 配置 | '+' | '.join(fixtures)+' |','|---|'+'---:|'*len(fixtures)]
        for arch in manifest['architectures']:
            lines.append('| '+arch+' | '+' | '.join('{:.6f}'.format(elapsed(index(mapping[n])[arch])/1e9) for n in fixtures)+' |')
        lines.append('')
    lines+=['## 保守串行计时对照','',
        'DeepSeek batch 32；每条宏 tile 收费 MLEN+16 周期，取消流水重叠。其他资源与主比较一致。它仅模拟旧 MatrixMachine 指令计时公式的形式，不是复现旧 SRAM/ISA，也未采用旧默认 overhead=0。','',
        '| 配置 | 耗时 ms | 相对本组最快单核 |','|---|---:|---:|']
    for r in compact:
        if r['comparison']=='deepseek_b32_legacy_serialized':
            lines.append('| {architecture} | {total_ms:.6f} | {speedup_vs_fastest_single:.4f}× |'.format(**r))
    lines+=['','## 收益归因与限制','',
        '- 主比较中 Qwen 的 K512/K1024 单核缓存命中为零，仍为每次查询和插入各收费一周期。单核一个缓存端口、双核两个端口，会改变吞吐；因此不能把这组差距全部解释为大小核减少计算浪费。关闭缓存检查正是为排除这一点。',
        '- 原定敏感性只含 DeepSeek batch 32；发现上述瓶颈后，追加其余全部三组的缓存关闭对照。这是明确标记的诊断扩展，保留原主比较和所有负结果。',
        '- 双核共用同一个 Ramulator HBM2，HBM 没有翻倍。重叠请求、任务分配、片上端口和 padding 会改变实际完成时间。HBM 字节及每核等待/利用率详见 CSV；等待项存在重叠，不能简单相加。',
        '- 相同乘法器和 SRAM 容量不等于相同面积。每核独立控制、缓存端口以及不同宽度的 SRAM 读口尚未综合；未建模 bank conflict 与完整 accumulator 端口冲突。',
        '- 两种矩阵服务时间和共享 vector actor 都是明确的分析模型；Ramulator 负责实际请求的存储时序。不能把程序跑通直接当成 RTL 时序已校准。','',
        '## 数值、数据与复现证据','',
        '四组维度为 Qwen D=2048/F=512、DeepSeek D=2048/F=1408，token 数分别 8/32。原 NPZ 哈希、原有 8-token 路由切片已核对；32-token 是对 batch-16 采集结果的重组，不是新采集的 batch-32 forward，更不是 prefill。权重和输入为非零合成值，实际使用 PLENA 本地 E4M3/E8M0 block-8 编码；不能称为真实模型精度或 OCP MX 合规结果。完整尺寸性能只覆盖 routed experts；shared expert 在功能测试覆盖。','',
        f'共 {len(summaries)} 组比较、{validated_runs} 次有效 Rust 运行，每个配置运行两次；验证独立参考数值、文件/二进制身份、资源上限、请求与 MAC 计数以及重复确定性。全部最终 BF16 输出与独立参考逐位一致：{bit_exact}。Rust 13、Compiler 导出 21、Python 证据门禁 33 个相关测试通过；记录在 `repro/verification.json`。旧中断尝试目录保留，不计入通过次数。','',
        '- `architecture_metrics.csv`：全部架构的耗时、HBM 实读字节、有效 MAC 利用率、padding 比率。',
        '- `core_metrics.csv`：每核任务、缓存命中、端口时间、等待及 SRAM 峰值。',
        '- `comparisons/*/comparison.json`：带数值门禁及原始运行目录的完整结果。',
        '- `execution_plan.json`、`cache_followup_plan.json`：预先声明与追加诊断的界限。',
        '- `repro/moe_dual_normal`、`repro/build_environment.sh`：本次二进制及构建环境。','',
        '## 后续正确的顺序','',
        ('1. 本次不冻结 3072+1024 大小核比例，保留经过缓存开关比较的单核作为强基线；停止引用原先带缓存比较中的加速作为大小核有效性的结论。' if wins==0 else '1. 保留强单核基线，逐项限定当前比例的收益范围；不能只引用带缓存比较中的加速。'),
        '2. 优先校准矩阵发射间隔、normal SRAM/accumulator 端口和 DMA/cache 吞吐，再在同样端口/面积预算下搜索单核、同构和异构配置。若同构更快，就将同构保留为必须战胜的基线。',
        '3. 增加独立路由窗口、真实 prefill 和 shared-expert 完整尺寸执行，用未参与配置选择的窗口验证；当前四组结果不能代替全模型。',
        '4. 在已验证的 normal 路径上接入 Compiler/旧 ISA 的多核任务描述与执行，再开展多小核、transpose/attention 和 RTL/PPA；不把这些后续项算作本次已完成。','']
    (root/'RESULT_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print(root/'RESULT_ZH.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    report(parser.parse_args().root.resolve())
