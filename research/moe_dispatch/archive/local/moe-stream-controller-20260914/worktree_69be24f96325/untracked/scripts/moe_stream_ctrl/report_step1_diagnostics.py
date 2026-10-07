#!/usr/bin/env python3
"""Summarize the step1 diagnostic matrix; performance thresholds stay frozen."""
import argparse
from pathlib import Path
import re

import diagnose_step1 as diag
from step0 import read, save, require, csv_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    validation = read(root / 'validation.json')
    require(validation['status'] == 'passed' and validation['completed_runs'] == 22,
            'complete the two-repeat native matrix before reporting')
    manifest = read(root / 'manifest.json')
    runs = {c['id']: read(root / c['id'] / 'rep1.json')['result'] for c in manifest['cases']}
    architectures = {c['id']: read(c['architecture']) for c in manifest['cases']}

    def core(case, index=0): return runs[case]['cores'][index]
    def detail(case, index=0): return core(case, index)['refinement']
    def stream(case, index=0): return detail(case, index).get('stream_ctrl') or {}
    def pool(case, index=0): return detail(case, index).get('output_pool') or {}
    def time(case): return runs[case]['total_ps']

    n3, o2, o3, z2, z3 = ('me32_n3', 'me32_event_rotating_o2', 'me32_event_rotating_o3',
                           'me32_zero_event_o2', 'me32_zero_event_o3')
    require(time(n3) == 70_532_000 and time(o2) == 78_070_000, 'frozen Me32 reference drifted')
    s = stream(o2)
    event_rows = []
    for kind, name in enumerate(manifest['event_kind_order']):
        count = s['event_counts_by_kind'][kind]
        event_rows.append(dict(kind=name, count=count,
            update_cycles=s['event_update_cycles_by_kind'][kind],
            cycles_per_event=s['event_update_cycles_by_kind'][kind]/count,
            service_ps=s['event_service_ps_by_kind'][kind],
            control_wait_ps=s['event_control_wait_ps_by_kind'][kind],
            queue_residence_ps=s['event_queue_residence_ps_by_kind'][kind],
            mean_producer_to_update_ns=s['event_delivery_ps_by_kind'][kind]/count/1000,
            max_producer_to_update_ns=s['event_delivery_max_ps_by_kind'][kind]/1000))
    csv_write(root / 'D2_events.csv', event_rows)
    busy = sum(s['event_service_ps_by_kind'])
    event_summary = dict(events=s['events_processed'], update_cycles=sum(s['event_update_cycles_by_kind']),
        mean_update_cycles_per_event=sum(s['event_update_cycles_by_kind'])/s['events_processed'],
        busy_ps=busy, overlap_with_mac_ps=s['event_service_mac_overlap_ps'],
        overlap_fraction_of_event_service=s['event_service_mac_overlap_ps']/busy,
        overlap_fraction_of_mac_service=s['event_service_mac_overlap_ps']/core(o2)['compute_busy_ps'],
        no_mac_overlap_ps=s['event_service_no_mac_ps'], fifo_capacity=s['event_capacity'],
        fifo_peak=s['event_queue_peak'], producer_full_fifo_delay_ps=s['event_queue_wait_ps'],
        blocked_producers_peak=s['blocked_producers_peak'],
        boundary='Half-open event-control-port service intervals intersected with actual MAC service intervals; no feedback or port-acquire wait included.')
    save(root / 'D2_event_summary.json', event_summary)

    # Ordered counterfactual decomposition, not a sum of overlapping busy times.
    decomposition = []
    for label, before, after in [('D1 semantic correction (not applicable)', o2, o2),
                                 ('D2 zero event-update service at O2', o2, z2),
                                 ('D3 O2 to O3 after D2', z2, z3),
                                 ('Residual versus frozen N3', z3, n3)]:
        row = dict(component=label, before=before, after=after,
                   contribution_ps=time(before)-time(after), contribution_us=(time(before)-time(after))/1e6)
        for key in ('weight_ready_wait_ps', 'accumulator_dependency_stall_ps', 'pipeline_drain_ps'):
            row[key+'_delta'] = core(before)[key]-core(after)[key]
        row['accumulator_port_wait_ps_delta'] = detail(before)['accumulator_port_wait_ps']-detail(after)['accumulator_port_wait_ps']
        row['scheduler_service_ps_delta'] = pool(before).get('scheduler_busy_ps', 0)-pool(after).get('scheduler_busy_ps', 0)
        row['scheduler_port_wait_ps_delta'] = stream(before).get('scheduler_port_wait_ps', 0)-stream(after).get('scheduler_port_wait_ps', 0)
        decomposition.append(row)
    require(sum(r['contribution_ps'] for r in decomposition) == time(o2)-time(n3), 'decomposition does not reconcile')
    csv_write(root / 'D4_decomposition.csv', decomposition)
    interaction = dict(order='D1 (no correction), D2 at O2, D3 with event service already removed, residual',
        standalone_third_operand_gain_ps=time(o2)-time(o3),
        third_operand_gain_after_zero_event_service_ps=time(z2)-time(z3),
        interaction_ps=(time(o2)-time(o3))-(time(z2)-time(z3)),
        alternate_order_ps=[0, time(o2)-time(o3), time(o3)-time(z3), time(z3)-time(n3)],
        warning='Counterfactual timing changes readiness and port arbitration. These are conditional net wall-time effects, not independent additive hardware penalties. The residual has not been uniquely attributed.')
    save(root / 'D4_interaction.json', interaction)

    resources = []
    for case in (n3, 'me32_pool_scan', o2, o3):
        c, a = core(case), architectures[case]['cores'][0]
        r = a['refinement']; slots = a['weight_slots']
        stages = (r.get('output_pool') or {}).get('operand_stages', r['active_n_tiles'])
        slot_bytes = a['blen'] * a['mlen'] * 25 // 8
        reservation = slots*slot_bytes + stages*a['blen']*a['mlen']*2
        require(reservation == c['weight_sram_peak_bytes'], 'observed weight peak differs')
        resources.append(dict(case=case, p=a['blen'], r=a['mlen'], mt=r['m_rows'], slots=slots,
            operand_stages=stages, packed_and_decoded_bytes_per_slot=slot_bytes,
            operand_bytes_per_stage=a['blen']*a['mlen']*2, weight_reserved_bytes=reservation,
            weight_budget_bytes=a['weight_sram_bytes'], weight_margin_bytes=a['weight_sram_bytes']-reservation,
            control_reserved_bytes=detail(case)['output_context_peak_bytes'],
            accumulator_peak_bytes=c['accumulator_peak_bytes'], accumulator_budget_bytes=a['accumulator_bytes'],
            accumulator_margin_bytes=a['accumulator_bytes']-c['accumulator_peak_bytes']))
    csv_write(root / 'D3_resources.csv', resources)

    fixed = []
    for name in ('b32_fixed_scan', 'b32_fixed_rotating', 'b32_fixed_lowest'):
        for index, c in enumerate(runs[name]['cores']):
            fixed.append(dict(case=name, core=c['id'], jobs=c['jobs'],
                rows=sum(j['rows'] for j in runs[name]['job_completions'] if j['core']==c['id']),
                hbm_read_bytes=c['hbm_read_bytes'], scheduler_ps=pool(name,index)['scheduler_busy_ps'],
                total_ps=time(name), accumulator_peak_bytes=c['accumulator_peak_bytes'],
                accumulator_margin_bytes=architectures[name]['cores'][index]['accumulator_bytes']-c['accumulator_peak_bytes']))
    csv_write(root / 'B32_fixed_dispatch.csv', fixed)
    frozen = read(diag.RUN/'step1/adaptive_b32_hbm_clock_dma2/rep1.json')['result']
    fixed_order = diag.job_order(frozen, architectures['b32_fixed_scan'])
    save(root/'B32_fixed_assignment.json', dict(source=str(diag.RUN/'step1/adaptive_b32_hbm_clock_dma2/rep1.json'),
        core_order=[c['id'] for c in architectures['b32_fixed_scan']['cores']], job_order=fixed_order,
        jobs=sorted(frozen['job_completions'], key=lambda j:j['start_ps']),
        verified_all_fixed_runs=True))

    gates = [dict(name='Me32 normal rotating O2', observed_ps=time(o2), limit_ps=70_532_000,
                  comparator='<=', passed=time(o2)<=70_532_000)]
    for mode in ('rotating','lowest'):
        cost = pool('b32_fixed_'+mode,1)['scheduler_busy_ps']
        gates.append(dict(name='Fixed B32 small scheduler '+mode, observed_ps=cost, limit_ps=700_000_000,
                          comparator='<', passed=cost<700_000_000))
    acceptance = dict(status='failed_performance_stopped', invariants_passed=True, gates=gates,
        d1_correction_applied=False, reason='Both N3 and pool already wait for accumulator writeback completion.',
        residual_ps=time(z3)-time(n3), step2_started=False,
        next_action='Stop at step1 and wait for the user decision; do not change frozen thresholds.')
    require(not all(g['passed'] for g in gates), 'report text assumes failed gates')
    save(root/'acceptance.json', acceptance)
    manifest.update(status='completed_invariants_passed_performance_failed', completed_runs=22,
                    step2_started=False, acceptance=str(root/'acceptance.json'))
    save(root/'manifest.json', manifest)

    def table(headers, rows):
        return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                         ['| '+' | '.join(map(str,row))+' |' for row in rows])

    measured = table(['Me32 配置','总时间 µs','用途'], [
        ['冻结 N3',f'{time(n3)/1e6:.3f}','验收基准'],
        ['原扫描 Q32',f'{time("me32_pool_scan")/1e6:.3f}','旧 pool 对照'],
        ['事件 Q32 / 旋转 / O2',f'{time(o2)/1e6:.3f}','正式验收配置'],
        ['事件 Q32 / 最低位 / O2',f'{time("me32_event_lowest_o2")/1e6:.3f}','另一选择器'],
        ['事件 Q32 / 旋转 / O3',f'{time(o3)/1e6:.3f}','D3 容量诊断'],
        ['事件更新零服务 / 旋转 / O2',f'{time(z2)/1e6:.3f}','D2 诊断，不用于验收'],
        ['事件更新零服务 / 旋转 / O3',f'{time(z3)/1e6:.3f}','D4 交叉诊断，不用于验收']])
    event_table = table(['事件','个数','更新周期/事件','服务 µs','FIFO 等待 µs（累积）'],
                       [[r['kind'],r['count'],f"{r['cycles_per_event']:.0f}",f"{r['service_ps']/1e6:.3f}",
                         f"{r['queue_residence_ps']/1e6:.3f}"] for r in event_rows])
    decomposition_table = table(['项','有序对照','对 7.538 µs 差距的贡献'],[
        ['D1','两者都等写回完成，无修正','0.000 µs'],
        ['D2',f'{time(o2)/1e6:.3f} → {time(z2)/1e6:.3f}',f'{(time(o2)-time(z2))/1e6:.3f} µs'],
        ['D3（在 D2 后）',f'{time(z2)/1e6:.3f} → {time(z3)/1e6:.3f}',f'{(time(z2)-time(z3))/1e6:.3f} µs'],
        ['残差',f'{time(z3)/1e6:.3f} − {time(n3)/1e6:.3f}',f'{(time(z3)-time(n3))/1e6:.3f} µs'],
        ['合计','','7.538 µs']])
    wait_table = table(['配置','权重就绪等待 µs','依赖等待 µs','acc 端口等待 µs','控制端口等待 µs','scheduler 服务 µs'],
        [[case,f'{core(case)["weight_ready_wait_ps"]/1e6:.3f}',
          f'{core(case)["accumulator_dependency_stall_ps"]/1e6:.3f}',
          f'{detail(case)["accumulator_port_wait_ps"]/1e6:.3f}',
          '未建模' if case==n3 else f'{stream(case)["scheduler_port_wait_ps"]/1e6:.3f}',
          '未单独计费' if case==n3 else f'{pool(case)["scheduler_busy_ps"]/1e6:.3f}'] for case in (n3,o2,o3,z2,z3)])
    resource_table = table(['配置','权重预算内占用 / 余量 B','控制记录 B','accumulator 总占用 / 余量 B'],
        [[r['case'],f"{r['weight_reserved_bytes']:,} / {r['weight_margin_bytes']:,}",r['control_reserved_bytes'],
          f"{r['accumulator_peak_bytes']:,} / {r['accumulator_margin_bytes']:,}"] for r in resources])
    b32_table = table(['固定归属配置','大核 scheduler ms','小核 scheduler ms','总时间 ms'],
        [[case,f'{pool(case)["scheduler_busy_ps"]/1e9:.6f}',f'{pool(case,1)["scheduler_busy_ps"]/1e9:.6f}',
          f'{time(case)/1e9:.6f}'] for case in ('b32_fixed_scan','b32_fixed_rotating','b32_fixed_lowest')])
    test_log = (root/'repro/cargo-test.log').read_text()
    tests = sum(map(int,re.findall(r'test result: ok\. (\d+) passed',test_log)))
    report = f'''**Step1 补充诊断：D1–D4 与固定派工 B32**

**结论：Step1 仍未通过，停在 Step1，未进入 Step2。** Me32 正式配置仍为 78.070 µs，高于冻结 N3 的 70.532 µs；固定原始专家归属后，两种选择器的小核 scheduler 都是 1.078272 ms，高于 0.7 ms。验收数字没有调整。

本次只增加观测、显式诊断开关和复现脚本。Compiler 导出、权重 bank/布局、native Ramulator 源码与库、DMA 策略/额度、原数据通路保持不变。B32 继续使用用户指定的冻结 HBM clock + DMA 2× 条件；Me32 使用原 native 条件。没有启用 Step2、demand_aware、部分和、槽生命周期拆分或 PR。

**1. D1：原假设不成立，不能把 prev_k_done 提前到入队时刻。**

N3 的选择条件读取每个 M 记录的 `feedback`：[engine.rs:1500](repro/engine.rs#L1500)–1509；发射前要求该位为真。完成协程在 MAC+feedback 延迟之后调用 `refined_port_work(...).await`，**等待 accumulator 写服务结束，才把 feedback 设为真**（[engine.rs:1553](repro/engine.rs#L1553)–1563）。`refined_port_work` 自身先获得串行端口，再等待实际端口服务时间（engine.rs:1313–1334）。

旧 pool 同样要求 `context.ready`（[output_pool.rs:312](repro/output_pool.rs#L312)–325），同样在写服务 `.await` 结束后设为真（output_pool.rs:378–405）。两者在 tile 的所有 M 消费者发射后推进 band 的 K 指针，但不会因此清除每记录的写回依赖（engine.rs:1581–1587；output_pool.rs:425–434）。

事件版仍在写回服务结束后发送 `record_k_done`（[event_ready.rs:705](repro/event_ready.rs#L705)–722），事件消费者再更新 `prev_done`。所以**依赖条件相同，完成通知的传播成本不同**。额外的事件排队/更新成本归 D2，不重复归 D1。未应用任何 K 依赖语义修正。

数值顺序依据：每记录 `next_k` 单调增加，只有前次写完成才能再次发射；每次发射内 `kk=0..kr` 升序累加。同一输出的 K 顺序和端口 read→MAC/feedback→write→下一 K read 顺序不变。此模拟器在发射处进行数值加法，用独立端口/反馈模型收费；本次没有改这个抽象。

{measured}

**2. D2：事件服务占用与暴露在总延迟上的代价不同。**

Me32 单核 Q32 / 旋转 / O2：C=ceil(32/4)=8，gate/up 各 4 段 K，down 1 段 K；全投影合计 384 个 N band、768 个权重 tile、6,144 次 MAC 发射。一次权重 tile 有“已解码”和“已安装 operand”两个通知阶段。

{event_table}

合计 **7,680 个事件、19,200 个更新周期，平均 2.5 周期/事件，服务 19.200 µs**。全体 scheduler 服务仍是 40.320 µs，事件服务是它的子集。W=3 的 FIFO 峰值为 3；因满队列阻塞生产者的累计延后为 **0 周期**。FIFO 中正常排队等待累计 10.004 µs，与“队列满阻塞生产者”不同。

按实际时间区间求交，事件服务有 **8.847 µs 与 MAC 服务占用重叠，占事件服务的 {event_summary['overlap_fraction_of_event_service']:.2%}**（占 MAC 服务的 {event_summary['overlap_fraction_of_mac_service']:.2%}）；10.353 µs 未与 MAC 服务重叠。这里 MAC 区间不含反馈延迟、acc 端口等待或选择器服务；“不与 MAC 重叠”也不等于都在关键路径上。

写回完成到 `record_k_done` 更新完成，平均 **{event_rows[2]['mean_producer_to_update_ns']:.3f} ns**、最大 {event_rows[2]['max_producer_to_update_ns']:.0f} ns。6,144 个完成通知的延迟总和 26.035 µs 包含并行等待，不能作为总时间加项。

为测净效应，显式开启 `diagnostic.event_updates_zero_cost`：只把这三类事件的描述符更新服务设为 0，保留 FIFO 容量、更新数量、控制端口互斥、写回完成依赖、其他 actor 服务和全部数据/HBM 操作。它是诊断 oracle，**不是可接受的硬件实现，也不用于放行**。78.070 → 76.182 µs，净节省 1.888 µs；不能声称节省全部 19.200 µs。该反事实改变后续就绪和仲裁时刻，净效应已包含这些交互。

**3. D3：第 3 级 operand 没有单独改善当前配置。**

Me32 单核为 `(P,R,Mt)=(8,512,4)`。你给出的 9,600 B/槽和 6,144 B/级是异构大核 `(6,512,4)` 的数值，不能直接套在单核。单核每槽 element 4,096 B + scale 512 B + decoded BF16 8,192 B = 12,800 B；每级 operand 为 8,192 B。因此 O3 总计 **3×12,800 + 3×8,192 = 62,976 B**，在 64 KiB 内，余量 2,560 B。

{resource_table}

208 B 事件控制仍与 `128Q+64(slots+stages)` 计入同一 **accumulator** 预算。O2 的控制记录 4,416→4,624 B，accumulator 余量 777,120→776,912 B。O3 另加 64 B stage 记录，控制共 4,688 B，accumulator 余量 776,848 B；vector 不增加。

单独 O2→O3：78.070→78.940 µs，**慢 0.870 µs**。acc 端口等待 4.046→4.735 µs，权重就绪等待 4.939→5.058 µs，依赖等待 0.519→0.672 µs；所以没有“补回一个 latch 就消除差距”的证据。默认契约仍为最多 2 级；O3 必须显式开启 `diagnostic.allow_three_operand_stages`，并通过真实容量检查。

**4. D4：7.538 µs 的有序分解与尚未唯一归因的残差。**

{decomposition_table}

上表按“先 D2，后 D3”逐项相减，精确相加为 7.538 µs。**D3 的 0.205 µs 只在事件服务已归零的反事实条件下成立**。正常服务下它是 −0.870 µs 的收益，二者相差 1.075 µs。若先做 D3，分解变为 0 − 0.870 + 2.963 + 5.445 = 7.538 µs。两种顺序都保留在 JSON，不能把条件收益当成互相独立的机制收益。

{wait_table}

逐项“前−后”的等待计数器变化见 [D4_decomposition.csv](D4_decomposition.csv)。这些等待可能跨 actor 重叠，统计边界也不同，**不能把等待列相加凑总时间**；表中时间贡献来自完整运行的差值。

残差 5.445 µs 仍未被唯一分配给单一结构原因。代码已确认还有两类差异：

- **控制收费不同。** N3 的选择/状态处理在主循环里没有独立的 scheduler 周期收费；事件版即便把事件服务归零，其他 actor 的准入、绑定、发射和释放仍收费 **21.120 µs**。其中 `2×6144 + 3×768 = 14.592 µs` 是发射 actor 的描述符服务，另外是准入与绑定服务。21.120 µs 是服务占用，不是 5.445 µs 的同义数，更不是已经测出的独立壁钟惩罚。冻结 N3 的 70.532 µs 保持原样。
- **轮转和供数组织不同。** N3 先轮转 3 个 N group，再在组内找 M 记录（engine.rs:1500–1509）；事件版在平铺的 Q 位图上旋转，Q32/C8 可准入 4 个 band，但只有 3 个权重槽。Me32 实测独立输出切换 N3 为 **3,746 次**、事件版为 **765 次**；两者不只是扫描实现不同。增加 operand 数也会改变就绪、端口仲裁与取数时序，容量收益不能脱离调度来判断。

三种实现的 MAC 服务都是 24.576 µs、acc 端口服务 49.152 µs、权重端口服务 6.144 µs、最终输出处理 12.480 µs，Me32 HBM 读取都是 **3,538,944 B**。这说明原始算术/端口工作量没变；剩余净差距不能解释成多算或多读 HBM，也不能宣称已证明 output 解耦本身无效。

**5. B32：固定原始专家归属后，0.7 ms 门槛未通过。**

从冻结 `adaptive_dispatch/…/hbm_clock_dma2` 的 job start 记录提取每核 job ID 顺序，使用现有 `diagnostic.fixed_job_order` 重放。三种固定版本同时锁定专家归属及核内顺序，保留原 dispatcher 服务成本。大核 11 专家/208 行，小核 12 专家/48 行；每核 useful/issued MAC、权重读取字节及任务顺序均逐项核验。

{b32_table}

固定扫描对照重现原来的总时间 3.091489 ms 和小核 scheduler 2.498892 ms。固定归属后，事件版小核 scheduler 降为 **1.078272 ms**，确实比原扫描少，但**两种选择器都未达到 <0.7 ms**。大核反而由 0.364680 增至 0.419142 ms，反映了新控制服务本身的代价。

此前动态事件版小核只有 18 行，scheduler 是 0.569088 ms。它不构成原来 48 行任务的验收证据。固定归属总时间虽从 3.091489 降为 1.575799 ms，也不能代替 scheduler 门槛。

服务计数可独立重算：每核 `Σbands(C+1) + 8×tile数 + 5×issue数`。固定小核为 **46,080 + 8×36,864 + 5×147,456 = 1,078,272 周期**；1 ns/周期即 1.078272 ms。这不是 DMA 等待混入的时间。HBM 全局读取仍为 **81,395,712 B**，大核 38,928,384 B、小核 42,467,328 B。

因此，在当前工作量和每类控制访问费用不变的条件下，仅换选择器或增加等待重叠不能让这项服务总周期低于 0.7 ms。进一步设计需要重新检查每次发射/完成事件是否存在可合并的控制记录访问，并重新证明状态更新及收费正确；本轮没有擅自实施这种机制修改。

208 B/core 同预算检查：固定大核 accumulator 245,656→245,864 B，512 KiB 内余量 278,632→278,424 B；固定小核 251,072→251,280 B，余量 273,216→273,008 B。两核权重、vector 容量不变。

**6. 验证与复现。**

- Rust workspace **{tests} 测试通过**；Clippy `--workspace --all-targets -- -D warnings` 通过。
- 11 配置×2 = **22 次 native 运行**，其中 4 次为明确标记的零事件服务反事实，其余 18 次保留实际服务收费；全体 BF16、FP32、最终舍入前 FP32 均 bit-exact。
- HBM 字节变化 0 次；native pending 全为 0；每配置两次总周期、完整 result 和 native calibration 一致。N3、旧 Q32、两种 Me32 事件选择器、原动态 B32 的冻结字段均复现，观测没有改变其旧计数/时序。
- 核尺寸、SRAM 预算、bank 文件、32 B native 请求、每通道入口节拍及冻结 B32 HBM/DMA 2× 条件由架构与 provenance/hash 锁定。`repro/libramulator.so` 与 Step0/Step1 同 SHA256；最终补丁不含 Ramulator 源码改动。
- 首轮两个 oracle 被旧的投影级收费断言拒绝；修正为 oracle 的明确费用公式后，对全部结果重新校验并补齐第二次运行。没有改结果、阈值或普通收费断言；首次日志和校验器保留于 `repro/attempt1`。

复现：先 `source {root}/repro/build-env.sh`，用 `repro/source.patch` 在基准提交的干净工作树恢复源码后构建 `cargo build --release --bin moe_dual_normal`。使用本目录已归档二进制重验/补跑：

```bash
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python {root}/repro/diagnose_step1.py --output {root}
```

每个配置的 `architecture.json`、两份原始 `rep*.json`、`validation.json` 都在对应子目录；入口为 [manifest.json](manifest.json)、[measurements.csv](measurements.csv)、[core_services.csv](core_services.csv)、[acceptance.json](acceptance.json)。本报告原始数据和计数器足以复算 D2/D3/D4。

**停点：提交以上分解与证据，等待决定。** 未放松 K 依赖，未改验收门槛，未进入 Step2。若继续解释 5.445 µs 残差，需要另行决定如何对比 N3 的控制收费与 N/M 轮转顺序；本轮没有把这一未决定的实验或新机制擅自加入。
'''
    def absolute_link(match):
        target = match.group(1)
        if target.startswith(('/', 'https://', 'http://')):
            return match.group(0)
        return '](' + str(root/target.replace('#L', ':')) + ')'
    report = re.sub(r'\]\(([^)]+)\)', absolute_link, report)
    (root/'REPORT.md').write_text(report)
    status = read(diag.RUN/'run.json')
    status['status'] = 'step1_diagnostics_failed_gates_stopped'
    status['step1_diagnostics'] = str(root/'REPORT.md')
    status['step1_diagnostics_acceptance'] = str(root/'acceptance.json')
    save(diag.RUN/'run.json', status)
    print(str(root/'REPORT.md'))


if __name__ == '__main__':
    main()
