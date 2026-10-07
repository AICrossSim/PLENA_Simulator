#!/usr/bin/env python3
"""Audit completed step0 artifacts and write the pre-implementation report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    manifest = read(root / 'manifest.json')
    require(manifest['status'] == 'passed' and manifest['completed_runs'] == 84, 'regression incomplete')
    for file, field in [('moe_dual_normal', 'binary_sha256'), ('libramulator.so', 'native_library_sha256'),
                        ('runner.py', 'runner_sha256'), ('compare_moe_normal.py', 'validator_sha256')]:
        require(digest(root / 'repro' / file) == manifest[field], 'repro artifact changed: ' + file)
    count = 0
    cases = {}
    input_hashes = {}
    for case in manifest['cases']:
        for field, expected in case['input_hashes'].items():
            path = case[field]
            if path not in input_hashes:
                input_hashes[path] = digest(path)
            require(input_hashes[path] == expected, 'input/reference modified: ' + path)
        golden = read(case['golden'])
        for repeat in (1, 2):
            result = read(root / case['id'] / f'rep{repeat}.json')['result']
            for field in ('output_f32', 'pre_round_output_f32'):
                actual, expected = result[field], golden[field]
                require(len(actual) == len(expected), 'FP32 row count differs')
                for a, b in zip(actual, expected):
                    require(len(a) == len(b), 'FP32 column count differs')
                    require(all(struct.pack('<f', x) == struct.pack('<f', y) for x, y in zip(a, b)),
                            case['id'] + ': FP32 bit pattern mismatch')
            count += 1
        cases[case['id']] = result
    require(count == 84, 'bit-pattern audit incomplete')
    expected_table = [
        ('expert_me1_single_legacy_n3', 'Me=1', 'Single N3', '31.124'),
        ('expert_me8_single_legacy_n3', 'Me=8', 'Single N3', '38.376'),
        ('expert_me32_single_legacy_n3', 'Me=32', 'Single N3', '70.532'),
        ('expert_me32_single_pool_q32', 'Me=32', 'Single Q32', '97.115'),
        ('qwen_full_decode_b8_single_legacy_n3', 'B8', 'Single N3', '582.155'),
        ('qwen_full_decode_b8_single_pool_q32', 'B8', 'Single Q32', '596.846'),
        ('qwen_full_decode_b8_heterogeneous_legacy_n2', 'B8', 'Heterogeneous N2', '828.061'),
        ('qwen_full_decode_b8_heterogeneous_pool_q32', 'B8', 'Heterogeneous Q32', '745.106'),
        ('qwen_full_decode_b8_single_pool_q32__ideal_supply64', 'B8', 'Ideal single Q32', '53.734'),
        ('qwen_full_decode_b32_single_legacy_n3', 'B32', 'Single N3', '1003.224'),
        ('qwen_full_decode_b32_single_pool_q32', 'B32', 'Single Q32', '1177.071'),
        ('qwen_full_decode_b32_heterogeneous_legacy_n2', 'B32', 'Heterogeneous N2', '1319.428'),
        ('qwen_full_decode_b32_heterogeneous_pool_q32', 'B32', 'Heterogeneous Q32', '1433.075'),
        ('qwen_full_decode_b32_single_pool_q32__ideal_supply64', 'B32', 'Ideal single Q32', '207.761'),
    ]
    table = []
    for key, workload, organization, expected in expected_table:
        result = cases['frozen__' + key]
        observed = result['total_ps'] / 1e6
        require(f'{observed:.3f}' == expected, 'task table mismatch: ' + key)
        table.append(dict(workload=workload, organization=organization, task_us=expected,
                          actual_ps=result['total_ps'], actual_us=observed, repeated_exactly=True))
    with (root / 'frozen_table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    bounds = []
    for workload, expected_bytes in [('b8', 60162048), ('b32', 81395712)]:
        result = cases[f'frozen__qwen_full_decode_{workload}_single_legacy_n3']
        require(result['hbm_read_bytes'] == expected_bytes, 'native byte count changed')
        lower = expected_bytes / (8 * 32) / 1000
        bounds.append(dict(workload=workload.upper(), bytes=expected_bytes, channels=8, native_bytes=32,
                           issue_period_ps=1000, aggregate_admission_gbps=256, admission_lower_us=lower,
                           single_n3_us=result['total_ps'] / 1e6,
                           single_n3_over_lower=result['total_ps'] / 1e6 / lower,
                           step2_stop_threshold_us=1.3 * lower))
    save(root / 'admission_bounds.json', bounds)
    organizations = {'single': [(8, 512)], 'homogeneous': [(4, 512), (4, 512)],
                     'heterogeneous': [(6, 512), (4, 256)]}
    resource_checks = []
    for name, cores in organizations.items():
        row = dict(organization=name, window_per_core=6, budget=45056)
        for policy, fragment_bytes, extra in [('per_channel', 24, 0), ('demand_aware', 40, 5632)]:
            required = 128 * 120 + 8192 + extra
            for p, r in cores:
                fragments = (r + 63) // 64 + ((r + 7) // 8 + 63) // 64 + 2
                required += 6 * (fragment_bytes * p * fragments + 128)
            row[policy + '_bytes'] = required
            row[policy + '_fits'] = required <= row['budget']
        resource_checks.append(row)
    save(root / 'resource_formula_checks.json', resource_checks)
    test_log = (root / 'repro/cargo-test-workspace.log').read_text()
    groups = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed;', test_log)
    passed, failed = (sum(int(group[i]) for group in groups) for i in (0, 1))
    require(passed == 263 and failed == 0, 'Rust test count mismatch')
    for name in ('clippy-workspace.log', 'release.log'):
        log = (root / 'repro' / name).read_text()
        require('Finished ' in log and not re.search(r'^error:', log, re.M), name + ' failed')
    source = Path(manifest['source_tree'])
    reference = source.parents[1] / 'review_20260911/simulator-moe-bottleneck'
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    reference_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=reference, text=True).strip()
    require(source_commit == reference_commit == manifest['source_commit'], 'source base changed')
    for tree in (source, reference):
        delta = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'transactional_emulator'], cwd=tree)
        require(not delta, 'simulation source changed: ' + str(tree))
    audit = dict(status='passed', rust_tests=passed, rust_failures=failed, clippy_warnings_denied=True,
                 legacy_cases=28, legacy_runs=56, all_runs=count, native_runs=80, ideal_reference_runs=4,
                 bf16_bit_exact_runs=84, fp32_bit_pattern_exact_runs=count, repeat_exact_runs=84,
                 native_drained_runs=80, changed_hbm_bytes_runs=0, task_table_cells_reproduced=14,
                 reference_input_hashes_rechecked=len(input_hashes), simulator_source_diff_empty=True,
                 implementation_status='not_started_awaiting_task_section_7_confirmation')
    save(root / 'acceptance.json', audit)
    lines = ['# Step 0 — 冻结复现与实施前资源核对', '',
             '**状态：step0 验收通过；step1–7 尚未实施，等待任务 §7.4 要求的资源契约确认。**', '',
             '本次新增独立工作树、回归工具和报告。冻结模拟器源码、Compiler 契约、权重镜像、原生 HBM 库及其时序均未修改；没有创建或更新线上 PR。', '',
             f'- 工作树：`{source}`', f'- 基线提交：`{source_commit}`',
             f'- 输出目录：`{root.parent}`', '', '## 验收', '',
             '| 项目 | 本次实测结果 |', '|---|---|',
             '| cargo test --workspace --locked --offline | 263 passed，0 failed |',
             '| cargo clippy --workspace --all-targets --locked --offline -- -D warnings | 通过 |',
             '| release 二进制 | 当前独立工作树重新构建，原件及 SHA256 已归档 |',
             '| 28 组遗留 native 对照 | 每组两次，共 56 次；所有旧结果字段和 native 计数器一致 |',
             '| §0.3 冻结表 | 14 个数字全部复现，各两次 |',
             '| 合计 | 42 配置 × 2 = 84 次，其中 80 native、4 ideal 参考 |',
             '| 数值 | 84 次 BF16、FP32、舍入前 FP32 均一致；另逐值检查 FP32 位模式 |',
             '| 不变条件 | HBM 字节变化 0 次；80 次 native 全部 drain；两遍总时间及 native 计数器完全一致 |',
             '| 资源 | 原校验器的 SRAM、DMA、端口和数值检查通过；新增机制尚未启用 |', '',
             '首轮测试脚本遇到旧校验器仅接受旧 HBM 显示名称的问题。新二进制输出的是 `Ramulator HBM2 with explicit timing profile`；已验证 native profile、全部服务倍率为 1 和原生计数器，再只对传入旧校验器的显示名做适配。没有改原始 result。首轮失败日志、runner、manifest 保留在 `repro/attempt1-*`，最终通过结果在 `manifest.json`。', '',
             '## 冻结表：两遍相同', '', '| 输入 | 组织 | 任务表 µs | 本次实际 µs |', '|---|---|---:|---:|']
    for row in table:
        lines.append(f"| {row['workload']} | {row['organization']} | {row['task_us']} | {row['actual_us']:.6f} |")
    lines += ['', '理想供数是任务明确要求复现的诊断参考，不参与 native 可达性能判断。任务中的两个理想时间保留三位小数；上表也给出了实际 ps 对应的精确小数。', '',
              '## DMA 入口下界', '',
              '`8 通道 × 32 B / 1 ns = 256 GB/s` 是当前模型入口的合计最大发射率，不等于声称 HBM 芯片的标称带宽。', '',
              '| 输入 | 本次 native 读取 B | 字节/入口峰值 µs | 单核 N3 / 下界 |', '|---|---:|---:|---:|']
    for row in bounds:
        lines.append(f"| {row['workload']} | {row['bytes']:,} | {row['admission_lower_us']:.3f} | {row['single_n3_over_lower']:.3f}× |")
    lines += ['', '**B32 = 81,395,712 / 256,000 = 317.952 µs。** 这是理想连续入口服务下界；真实运行还要经历返回延迟、数据就绪和输出收尾。不能把与下界的差值全部算成可消除开销，也不能仅凭这个倍数宣称纯 memory-bound。', '',
              '任务规定的 step2 中间停止阈值：B8 单核 ≤ 1.3 × 235.008 = **305.5104 µs** 时先报告并等待是否继续。当前 B8 单核 Q32 仍为 596.846 µs，这次未实施加速机制，不形成新收益结论。', '',
              '## 资源发现与待确认方案', '',
              '完整字段、生命周期和字节公式见 [RESOURCE_CONTRACT_PROPOSAL.md](RESOURCE_CONTRACT_PROPOSAL.md)。', '',
              '1. Step1 建议保留 Q=32、原权重槽和数据路径，新增有界事件队列、就绪位图、三个 actor。W=3 时新增 **208 B/core** 控制状态，计入现有 accumulator 预算；事件影响多个上下文时逐周期更新，不能免费广播 Q 次写入。',
              '2. Step2 的六个打包槽与两个 operand stage 在四种核上都满足权重 SRAM 容量。',
              '3. 但若同时启用现有 `demand_aware` DMA 元数据结构，前端预留变为单核 **51,072 B**、同构 **51,840 B**、异构 **53,280 B**，均超过 **45,056 B** 的固定预算。这是尚未实施的候选方案约束冲突，不是冻结基线验收失败。',
              '4. 建议先在新前端的有限 tile 窗口内做 aging，保留冻结 native `per_channel` 策略；另一选择是单独设计有限的按需 fragment 描述符生成。此项到 step2 前需确认，不能私自增加 SRAM。',
              '5. 后续还需明确 ECT 的 MAC/FLOP 单位、Me 上下文语义、短 K 的原生字节复用和共享 Z 的可见性。已逐项列出选项与影响；未据此实现后续步骤。', '',
              '## 重跑及证据', '',
              '`repro/` 保存 binary、native library、build-env、两个验证脚本和日志；SHA256 在 manifest。`measurements.csv` 为 84 次汇总，`core_services.csv` 为逐核服务，`frozen_table.csv` 对应任务表；每个 case 保留两个完整 result、日志和 validation。', '',
              f'完整重新验证入口：`bash {source}/scripts/moe_stream_ctrl/reproduce_step0.sh`。它创建新的 UUID 目录、重新执行 Rust 检查和 84 次模拟，再生成报告；已做 bash 语法检查，此封装脚本没有再启动一轮重复实验。', '',
              '也可使用 Python 3.11 分别运行 `step0.py` 和 `summarize_step0.py`。只读引用 manifest 中的冻结 workload/golden/bank；不要直接运行会写回旧目录的旧回归脚本。最后一次汇总复用了已经完成的 84 份 result 来加强校验和归档，`reused_result` 已标注；CSV 中 `wall_seconds` 是本次调用处理时间，不能当作纯模拟器主机运行时间。', '',
              '## 下一道门', '',
              '按任务 §7.4 “等待确认后再实现”，请确认资源说明中的 **step1 契约**。确认后才能实施 step1 并检验 Me32 单核 Q32 ≤ 70.532 µs，以及指定 B32 动态派工诊断的小核 scheduler 服务 < 0.7 ms。任一未通过，报告原因并停在该步。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    shutil.copy2(__file__, root / 'repro/summarize_step0.py')
    shutil.copy2(Path(__file__).with_name('reproduce_step0.sh'), root / 'repro/reproduce_step0.sh')
    manifest['acceptance_audit'] = audit
    manifest['resource_proposal_sha256'] = digest(root / 'RESOURCE_CONTRACT_PROPOSAL.md')
    manifest['summary_runner_sha256'] = digest(root / 'repro/summarize_step0.py')
    manifest['reproduction_wrapper_sha256'] = digest(root / 'repro/reproduce_step0.sh')
    save(root / 'manifest.json', manifest)
    run = read(root.parent / 'run.json')
    run.update(status='step0_passed_awaiting_contract_confirmation', completed_step=0,
               report=str(root / 'REPORT.md'), resource_proposal=str(root / 'RESOURCE_CONTRACT_PROPOSAL.md'))
    save(root.parent / 'run.json', run)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
