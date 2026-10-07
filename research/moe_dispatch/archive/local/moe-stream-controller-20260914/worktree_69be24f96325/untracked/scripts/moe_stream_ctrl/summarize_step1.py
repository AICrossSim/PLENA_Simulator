#!/usr/bin/env python3
"""Audit step1 artifacts, account service/space, and report a failed gate honestly."""
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess

import step0 as base
from step1 import RUN, SOURCE


def main():
    root = RUN / 'step1'
    manifest = base.read(root / 'manifest.json')
    acceptance = base.read(root / 'acceptance.json')
    base.require(acceptance['completed_runs'] == 94 and acceptance['invariants_passed'], 'step1 invariants incomplete')
    for name, key in [('moe_dual_normal','binary_sha256'),('libramulator.so','native_library_sha256'),
                      ('step1.py','runner_sha256'),('step0.py','helper_sha256'),('compare_moe_normal.py','validator_sha256')]:
        base.require(base.digest(root/'repro'/name) == manifest[key], 'archived artifact changed: '+name)
    log = (root/'repro/cargo-test-workspace.log').read_text()
    groups = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed;', log)
    passed = sum(int(a) for a,b in groups)
    base.require(passed == 268 and all(int(b)==0 for a,b in groups), 'workspace test failure')
    clippy = (root/'repro/clippy-workspace.log').read_text()
    base.require('Finished ' in clippy and not re.search(r'^(error|warning):',clippy,re.M), 'Clippy failed')
    summaries, dispatch, startup = [], [], []
    selected = ['frozen__expert_me32_single_pool_q32','adaptive_b32_hbm_clock_dma2']
    for name in selected:
        for mode in ['off','rotating','lowest']:
            key = name + ('' if mode=='off' else '__'+mode)
            d = base.read(root/key/'rep1.json')
            r, architecture = d['result'], d['architecture_manifest']
            for c, config in zip(r['cores'],architecture['cores']):
                detail=c['refinement'];p=detail['output_pool'];s=detail.get('stream_ctrl')
                jobs=[j for j in r['job_completions'] if j['core']==c['id']]
                if s:
                    admission=sum((e['n']+config['blen']-1)//config['blen'] *
                                  (1+(e['m']+detail['m_rows']-1)//detail['m_rows']) for e in c['projections'])
                    expected_cycles=admission + 8*p['tile_admissions']+5*p['context_updates']
                    base.require(expected_cycles==p['scheduler_visits'], 'scheduler accounting formula differs')
                    base.require(p['control_reserved_bytes']==4416+208, 'control capacity changed')
                    summaries.append(dict(case=key,core=c['id'],selector=mode,admission_cycles=admission,
                                          tile_lifecycle_cycles=8*p['tile_admissions'],
                                          fanout_cycles=p['context_updates'],
                                          issue_and_completion_cycles=4*p['context_updates'],
                                          scheduler_cycles=expected_cycles))
                    arrivals=[]
                    for job in jobs:
                        gate=next(e for e in c['projections'] if e['job']==job['job'] and e['projection']=='gate')
                        arrivals.append((gate['metrics']['first_tile_arrival_ps']-job['start_ps'])/1000)
                    startup.append(dict(case=key,core=c['id'],experts=len(arrivals),
                                        dma_speedup=architecture.get('diagnostic',{}).get('dma_speedup',1),
                                        hbm_profile=architecture.get('diagnostic',{}).get('hbm_profile','native'),
                                        mean_first_tile_latency_ns=sum(arrivals)/len(arrivals)))
                if name=='adaptive_b32_hbm_clock_dma2':
                    dispatch.append(dict(selection=mode,core=c['id'],jobs=len(jobs),rows=sum(j['rows'] for j in jobs),
                                         max_me=max(j['rows'] for j in jobs),hbm_bytes=c['hbm_read_bytes'],
                                         scheduler_us=p['scheduler_busy_ps']/1e6,total_us=r['total_ps']/1e6))
    base.csv_write(root/'scheduler_accounting.csv',summaries)
    base.csv_write(root/'dispatch_work.csv',dispatch)
    base.csv_write(root/'first_tile_summary.csv',startup)
    decisions=base.read(RUN/'run.json')['decisions']
    base.save(root/'authorized_decisions.json',decisions)
    core_rows=list(csv.DictReader((root/'core_services.csv').open()))
    lines=['# Step 1 — 事件就绪控制器实现与验收', '',
           '**结论：实现和正确性验证完成，整体性能验收未通过；按用户要求停在 step1，未进入 step2。**', '',
           '默认旋转策略把 Me32 单核 Q32 从 97.115 µs 降至 78.070 µs，但没有达到 ≤70.532 µs。指定 B32 诊断中，小核 scheduler 服务从 2.498892 ms 降至 0.569088 ms，通过 <0.7 ms。两项必须同时通过，不能用后者抵消前者。', '',
           '## 验收结果（每点两遍相同）', '',
           '| 验收项 | 冻结对照 | 旋转游标（默认） | 最低置位位 | 判定 |', '|---|---:|---:|---:|---|',
           '| Me32 单核 Q32 总时间，目标 ≤70.532 µs | 97.115 µs | 78.070 µs | 80.338 µs | 两种均未通过 |',
           '| B32 小核 scheduler 服务，目标 <0.700 ms | 2.498892 ms | 0.569088 ms | 0.569088 ms | 两种均通过 |',
           '| B32 整体算子时间（附带观察） | 3.091489 ms | 0.989397 ms | 1.006532 ms | 不替代上述验收项 |', '',
           '**B32 以上三点均使用同一冻结的 HBM clock 2× + DMA 2× 诊断配置，且 `fixed_job_order` 缺省、`dispatch_policy=work_conserving`。** 这些时间不是 native 默认时序结果，不应宣传为 native 配置的端到端加速。Me32 使用 native 配置。两边比较时未增添带宽、槽数或算力。', '',
           '## 已实现的逻辑', '',
           '原有路由和派工器仍决定专家归属。每个核仍按 gate → up → SwiGLU → down 执行；每个 N 带准入完整 M cohort，沿全局 K 顺序累加。新增的部分是下面三条控制路径：', '',
           '```text',
           '原 HBM/DMA → 原打包+解码槽 → tile_decoded 通知',
           '                                    ↓',
           '事件 actor：处理有限队列 → 逐周期更新上下文就绪位',
           '                                    ↑',
           'operand actor：等空闲 stage → 经原 SRAM 端口装入操作数',
           '                                    ↓',
           'issue actor：旋转/最低置位选择 → 原 MAC → 原 FP32 写回',
           '                                    └→ record_k_done 通知',
           '```', '',
           '事件 FIFO 深度 W=3；一个 tile 影响 C 个上下文时，逐一进行 C 次有成本的更新。三个 actor 可以分别推进，但控制记录访问共用一个串行控制端口，数据读写仍竞争原权重/accumulator 端口。队列满时，完成事件留在已有 slot/context 记录中等待，不存在无限事件缓冲。', '',
           '两阶段 tile 通知分别表示“权重已解码”和“实际已装入 operand stage”；第二阶段完成前不能发射。不会把收到通知等同于免费拿到操作数。新增代码在 `event_ready.rs`，旧路径通过可选开关继续保留。实现细节及周期表在工作树 `doc/moe_event_ready_step1.md`。', '',
           '## 208 B 的预算位置与余量', '',
           '**预算位置明确为每核 accumulator SRAM，与原控制记录位于同一预算；不是 vector SRAM，也不是新增预算。**', '',
           '`原控制 = 128×32 + 64×(3+2) = 4,416 B`', '',
           '`新增 = 三张 32-bit 位图按 64-bit 对齐 24 B + activation 位图 8 B + 3×16 B 事件 + actor 64 B + expert 64 B = 208 B`', '',
           '`新控制 = 4,624 B/core`。其余 accumulator 数据、pending FP32 结果和 pipeline 都继续计费。', '',
           '下表在**相同的本次实际任务分配**下扣除/加入 208 B，避免把派工导致的最大 Me 变化误当成控制空间变化。两种选择策略的本表数值相同。', '',
           '| 用例/核 | accumulator 总预算 B | 不含新增控制的完整占用 B | 含新增控制的完整占用 B | 原余量 B | 新余量 B |',
           '|---|---:|---:|---:|---:|---:|']
    for name in selected:
        key=name+'__rotating'
        for row in core_rows:
            if row['case']==key and row['repeat']=='1':
                label=('Me32 单核' if name==selected[0] else 'B32')+'/'+row['core']
                fields=['budget_bytes','same_jobs_without_event_control_bytes','accumulator_peak_bytes','budget_free_before_bytes','budget_free_after_bytes']
                lines.append('| '+label+' | '+' | '.join(f'{int(row[f]):,}' for f in fields)+' |')
    lines += ['', 'B32 冻结对照的实际 accumulator 峰值为大核 245,656 B、小核 251,072 B；本次实际峰值为大核 254,056 B、小核 30,096 B。差异包含最大 Me 的变化，不能只用两次峰值相减来判断 208 B 是否正确计费。完整逐核数据在 `core_services.csv`。', '',
              '## 为什么一项通过、一项未通过', '',
              '**Me32：三 actor 并行推进缩短了时间，但逐记录控制更新仍有成本。** 一次发射或 K 完成都分别更新 context 和 band；一个 tile 还要经历准入、装载、fan-out、释放。就绪位图消除了查找时的线性轮询，并没有消除这些状态更新。', '',
              '| Me32 单核服务 | 冻结 Q32 | 本次两种策略 |', '|---|---:|---:|',
              '| scheduler 服务 | 38.463 µs | 40.320 µs |',
              '| accumulator 数据端口服务 | 49.152 µs | 49.152 µs |',
              '| MAC busy 服务 | 24.576 µs | 24.576 µs |', '',
              'scheduler 服务没有下降：本次按 `Σ(C_b+1)+8×tile数+5×context发射数` 计费，Me32 恰为 `384×9+8×768+5×6144=40,320` 周期；其中 fan-out 的 6,144 周期全部纳入原 `scheduler_busy_ps`，未扣除与其他 actor 重叠的周期。额外预算和数据端口检查均通过，所以这不是“没计费才变快”。', '',
              '总时间虽然下降，但上述服务会部分重叠，不能直接相加或把 19.045 µs 的下降全部归为扫描修复。该实现同时包含就绪选择和三 actor 拆分，本次没有用单独消融声称各自贡献。它仍比冻结 N3 的 70.532 µs 慢约 10.7%，故 step1 失败。', '',
              '**B32：除了控制方式变化，还有 work-conserving 派工结果变化。** 派工算法未修改，但核心更早/更晚变空闲，会改变哪些专家由它领取。', '',
              '| 配置/核 | 专家数 | 分配的 token 行数 | 最大 Me | 权重读取 B | scheduler µs |', '|---|---:|---:|---:|---:|---:|']
    for row in dispatch:
        if row['selection']=='lowest':continue
        lines.append(f"| {row['selection']}/{row['core']} | {row['jobs']} | {row['rows']} | {row['max_me']} | {row['hbm_bytes']:,} | {row['scheduler_us']:.3f} |")
    lines += ['', '小核从 48 行降到 18 行，最大 Me 从 30 降到 3；因此 2.498892→0.569088 ms 是用户指定的动态派工场景下的实测服务总量，不能解释成同一批工作调度效率提升了 4.39×。总体仍执行相同 23 个专家、256 次 token 分配，native 权重读取仍为 81,395,712 B。', '',
              '这些结果支持“当前控制组织影响性能”，并不证明异构双核优于同资源单核，也不能把未达标笼统归咎于 HBM controller。与 native、同构和新共同基线的完整比较属于后续步骤，本次尚未进入。', '',
              '## 实际测试及不变条件', '',
              '- `cargo test --workspace --locked --offline`：268 项通过（新增 5 项）；Clippy `--all-targets -- -D warnings` 通过。',
              '- 47 配置各两遍，共 94 次：84 次重跑全部 step0 冻结控制，2 次指定动态 B32 冻结诊断，8 次新控制器。90 次使用 native Ramulator，4 次为冻结理想供数参考。',
              '- 28 组遗留 native 对照继续通过。历史 demand-aware 配置仅用于旧路径兼容性回归；新机制全部保持 native `per_channel`，demand-aware 因 44 KiB 前端预算排除，不作为新方案性能对照。',
              '- 94 次 BF16、FP32、最终舍入前 FP32 都与 golden 一致，FP32 另核对位模式；所有配置两遍总时间及 native 计数器完全一致。',
              '- HBM 字节变化 0 次；90 次 native 全部 drain；SRAM、额度、stage、context、事件容量和端口服务检查全部通过。',
              '- 单元测试覆盖长 fan-out 期间实际出现的队列回压、单槽、低反馈延迟、慢端口、尾块、两种选择器的跨 word/回绕选择，以及预算少 1 B 时拒绝执行。', '',
              '测量边界保持输入和路由已经就绪、冷状态起算；不是完整模型、router 或请求时延。实际源码 patch、binary、native library 和 SHA256 已归档。', '',
              '## 后续决定已保存，未实施', '',
              '`authorized_decisions.json` 保存用户批准的 A–E 决定：前端窗口 aging 保持 per_channel；ECT 用 MAC 与 B/ns；保留整 cohort；整 MLEN 取数、核内拆 K；Z 从原核私有 vector 经共享向量单元复制到接收核。它们覆盖 step0 中相应待定建议，后续无需重复询问这些已经决定的事项。', '',
              '`expert_first_tile_arrivals.csv` 记录每专家首个完整 packed tile 到达时间（DMA 复制完成、尚未解码）相对该专家开始的延迟；`first_tile_summary.csv` 按配置/核汇总，供后续 ECT 的 fixed_overhead 使用。须保留 native 与 HBM-clock/DMA-2x 标签，不能把不同 timing profile 的均值混用。', '',
              '**当前停止点：Me32 未达到 ≤70.532 µs。** 下一次若继续，应先审查 step1 中逐 context/band 的重复状态更新及 issue actor 的串行等待，确认能否在同样的预算和每周期更新限制下减少服务。不能改统计口径、增加隐藏端口，或先打开 step2 来掩盖这一项失败。', '',
              '## 复现入口', '',
              f'工作树：`{SOURCE}`。`scripts/moe_stream_ctrl/step1.py --output <新的输出目录> --binary <归档binary> --workers 4` 会创建相同实验矩阵并各跑两遍；先创建 `<新目录>/repro` 用于保存日志。它只读引用冻结输入。直接使用已有输出目录会校验现有 result，`reused_result` 字段对此明确标记。', '',
              '汇总：`measurements.csv`；逐核/预算：`core_services.csv`；scheduler 周期分解：`scheduler_accounting.csv`；派工负载：`dispatch_work.csv`；每点完整数据在对应目录的 `rep1.json`、`rep2.json`；最终判据在 `acceptance.json`。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(__file__,root/'repro/summarize_step1.py')
    shutil.copy2(SOURCE/'doc/moe_event_ready_step1.md',root/'repro/implementation.md')
    shutil.copy2(RUN/'step0/repro/build-env.sh',root/'repro/build-env.sh')
    # Capture all changed code and its review documentation, including new files.
    patch=subprocess.check_output(['git','diff','--binary','HEAD'],cwd=SOURCE)
    names=['transactional_emulator/src/moe_normal/event_ready.rs','doc/moe_event_ready_step1.md']
    names += [str(p.relative_to(SOURCE)) for p in (SOURCE/'scripts/moe_stream_ctrl').iterdir() if p.suffix in ('.py','.sh')]
    for name in sorted(names):
        p=subprocess.run(['git','diff','--no-index','--binary','/dev/null',name],cwd=SOURCE,capture_output=True,check=False)
        base.require(p.returncode==1,'patch failed: '+name)
        patch+=p.stdout
    (root/'repro/source.patch').write_bytes(patch)
    reference=ROOT_REFERENCE=SOURCE.parents[1]/'review_20260911/simulator-moe-bottleneck'
    subprocess.run(['git','apply','--check',str(root/'repro/source.patch')],cwd=reference,check=True)
    base.require(not subprocess.check_output(['git','diff','HEAD','--','transactional_emulator'],cwd=ROOT_REFERENCE),'frozen source changed')
    manifest.update(source_patch_sha256=base.digest(root/'repro/source.patch'),
                    summary_runner_sha256=base.digest(root/'repro/summarize_step1.py'),
                    workspace_tests_passed=passed,clippy_warnings_denied=True,
                    scheduler_formula_audit='passed for both selectors and both acceptance cases',
                    source_patch_apply_check='passed against frozen reference (read-only check)',
                    report_sha256=base.digest(root/'REPORT.md'))
    base.save(root/'manifest.json',manifest)
    run=base.read(RUN/'run.json');run.update(status='step1_failed_performance_stopped',report=str(root/'REPORT.md'),
                                           completed_step1_implementation=True,step2_started=False)
    base.save(RUN/'run.json',run)
    print(json.dumps(dict(step1_status=acceptance['status'],invariants_passed=True,tests=passed,runs=94,
                          next_step='stop_at_step1',report=str(root/'REPORT.md'))))


if __name__=='__main__': main()
