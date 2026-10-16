"""Audit and report the finite-cache Route A analytical experiment.

Ranking uses within-batch window geometric means, then an equally weighted
seven-batch geometric mean.  The separate additive breakdown uses arithmetic
means at both levels: independently computed GMs are never added/subtracted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

BATCHES = (2, 4, 8, 16, 64, 96, 128)
COUNTS = {2: 27, 4: 27, 8: 27, 16: 27, 64: 9, 96: 9, 128: 9}
DESIGNS = ('single', 'homo', 'hetero')
SHARED_BYTES = 33 * 1024**2
NUMERIC = (
    'batch', 'bandwidth_gbs', 'total_cycles', 'ideal_compute_cycles',
    'stall_cycles', 'hbm_bytes', 'unique_weight_bytes', 'shared_request_bytes',
    'shared_hit_bytes', 'shared_evicted_bytes', 'refetch_bytes',
    'hbm_busy_cycles', 'core0_compute_cycles', 'core1_compute_cycles',
)
LABELS = {'single': '单核', 'homo': '同构', 'hetero': '异构'}
POLICY_LABELS = {
    'fair_lru': '公平 LRU + 预取',
    'fair_pinned': '公平 Shared 固定驻留 + 预取',
    'proposed_asymmetric': '用户提出的不对称策略（混杂对照）',
    'fair_pinned_hard': '公平固定驻留 + 硬阈值分流',
    'ablate_no_prefetch': '公平 LRU，无预取',
}


def gm(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('GM needs a nonempty finite nonnegative sequence')
    return 0.0 if 0 in values else math.exp(math.fsum(map(math.log, values)) / len(values))


def mean(values):
    values = list(values)
    if not values:
        raise ValueError('Cannot average empty data')
    return math.fsum(values) / len(values)


def percentile(values, quantile):
    values = sorted(values)
    index = (len(values) - 1) * quantile
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def paired_speedup(reference, candidate, bootstrap=200, seed=20261013):
    """reference/candidate speedup; preserve pairing and equal batch weights."""
    key = lambda r: (int(r['batch']), r['window_id'])
    ref = {key(r): float(r['total_cycles']) for r in reference}
    cand = {key(r): float(r['total_cycles']) for r in candidate}
    if len(ref) != len(reference) or len(cand) != len(candidate):
        raise ValueError('Duplicate window in paired comparison')
    if ref.keys() != cand.keys() or not ref:
        raise ValueError('Paired comparison must contain identical nonempty windows')
    logs = defaultdict(list)
    for batch, window in sorted(ref):
        a, b = ref[(batch, window)], cand[(batch, window)]
        if min(a, b) <= 0:
            raise ValueError('Latency must be positive')
        logs[batch].append(math.log(a / b))
    score = math.exp(mean(mean(v) for v in logs.values()))
    rng = random.Random(seed)
    replicas = []
    for _ in range(bootstrap):
        replicas.append(math.exp(mean(
            mean(rng.choice(values) for _ in values) for values in logs.values())))
    return score, percentile(replicas, .025), percentile(replicas, .975)


def load_rows(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Input CSV is empty')
    for row in rows:
        for key in NUMERIC:
            if key not in row or row[key] == '':
                raise ValueError(f'Missing field {key}')
            row[key] = float(row[key])
        row['resident_reload_bytes'] = float(row.get('resident_reload_bytes') or 0)
        row['bandwidth_utilization'] = float(row.get('bandwidth_utilization') or
                                             row['hbm_bytes'] / (row['total_cycles'] * row['bandwidth_gbs']))
        row['batch'] = int(row['batch'])
    return rows


def validate_rows(rows, full_heldout=True):
    keys = set()
    grouped = defaultdict(list)
    for row in rows:
        for field in NUMERIC:
            if not math.isfinite(float(row[field])) or float(row[field]) < 0:
                raise ValueError(f'Invalid nonnegative field {field}')
        key = (row['policy'], row['cache_state'], row['bandwidth_gbs'],
               row['design'], row['window_id'])
        if key in keys:
            raise ValueError(f'Duplicate point: {key}')
        keys.add(key)
        if row['design'] not in DESIGNS or row['batch'] not in BATCHES:
            raise ValueError('Unrecognized design or batch')
        if row['cache_state'] not in ('cold', 'shared_resident'):
            raise ValueError('Unrecognized initial cache state')
        total, compute, stall = (float(row[x]) for x in
                                 ('total_cycles', 'ideal_compute_cycles', 'stall_cycles'))
        if total <= 0 or not math.isclose(total, compute + stall, rel_tol=1e-10, abs_tol=1e-5):
            raise ValueError('total must equal ideal same-assignment compute plus exposed stall')
        if row['hbm_busy_cycles'] > total + 1e-5:
            raise ValueError('Shared HBM bus occupancy cannot exceed layer span')
        if row['shared_hit_bytes'] > row['shared_request_bytes']:
            raise ValueError('Shared hit bytes exceed requested bytes')
        byte_floor = row['unique_weight_bytes'] - (
            SHARED_BYTES if row['cache_state'] == 'shared_resident' else 0)
        if row['hbm_bytes'] + 1e-5 < max(0, byte_floor):
            raise ValueError('HBM traffic below initial-state unique-byte lower bound')
        transfer_floor = row['hbm_bytes'] / row['bandwidth_gbs']
        if total + 1e-5 < transfer_floor:
            raise ValueError('Layer latency below shared HBM byte/bandwidth lower bound')
        grouped[key[:4]].append(row)
    for key, points in grouped.items():
        by_batch = {b: sum(p['batch'] == b for p in points) for b in BATCHES}
        if full_heldout and by_batch != COUNTS:
            raise ValueError(f'Incomplete heldout batch coverage: {key}: {by_batch}')
    scenarios = defaultdict(dict)
    for key, points in grouped.items():
        scenarios[key[:3]][key[3]] = {(p['batch'], p['window_id']) for p in points}
    for key, by_design in scenarios.items():
        if set(by_design) != set(DESIGNS):
            raise ValueError(f'Missing design in scenario: {key}')
        if len({frozenset(v) for v in by_design.values()}) != 1:
            raise ValueError(f'Unpaired windows across designs: {key}')
    return {'points': len(rows), 'scenario_count': len(scenarios),
            'window_count': len({r['window_id'] for r in rows}),
            'full_heldout_validation': full_heldout}


def aggregate(rows, field, geometric=True):
    by_batch = defaultdict(list)
    for row in rows:
        by_batch[row['batch']].append(float(row[field]))
    operation = gm if geometric else mean
    return operation(operation(v) for v in by_batch.values())


def main_tables(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['policy'], row['cache_state'], row['bandwidth_gbs'])].append(row)
    main, breakdown = [], []
    for (policy, state, bw), points in sorted(groups.items()):
        for batch in [*BATCHES, '7-Batch GM']:
            selected = points if isinstance(batch, str) else [p for p in points if p['batch'] == batch]
            if not selected:
                continue
            designs = {d: [p for p in selected if p['design'] == d] for d in DESIGNS}
            result = {'policy': policy, 'cache_state': state, 'bandwidth_gbs': bw, 'batch': batch}
            for d, values in designs.items():
                result[f'{d}_ms'] = aggregate(values, 'total_cycles') / 1e6
            for d in ('single', 'homo'):
                speed, lower, upper = paired_speedup(designs[d], designs['hetero'])
                result[f'hetero_speedup_vs_{d}'] = speed
                result[f'hetero_speedup_vs_{d}_ci95_low'] = lower
                result[f'hetero_speedup_vs_{d}_ci95_high'] = upper
            main.append(result)
            for d, values in designs.items():
                metrics = []
                for p in values:
                    q = dict(p)
                    q['shared_hit_pct'] = 100 * p['shared_hit_bytes'] / p['shared_request_bytes'] if p['shared_request_bytes'] else 0
                    q['hbm_bus_busy_pct'] = 100 * p['hbm_busy_cycles'] / p['total_cycles']
                    q['hbm_payload_bw_util_pct'] = 100 * p['hbm_bytes'] / (p['total_cycles'] * p['bandwidth_gbs'])
                    q['resident_reload_bytes'] = float(p.get('resident_reload_bytes') or 0)
                    q['total_reload_bytes'] = p['refetch_bytes'] + q['resident_reload_bytes']
                    q['hbm_lower_bound_cycles'] = p['hbm_bytes'] / p['bandwidth_gbs']
                    metrics.append(q)
                out = {'policy': policy, 'cache_state': state, 'bandwidth_gbs': bw,
                       'batch': batch if isinstance(batch, int) else '7-Batch equal AM', 'design': d}
                for dest, source, scale in (
                    ('compute_ms_AM', 'ideal_compute_cycles', 1e6),
                    ('stall_ms_AM', 'stall_cycles', 1e6),
                    ('total_ms_AM', 'total_cycles', 1e6),
                    ('hbm_MiB_AM', 'hbm_bytes', 1024**2),
                    ('unique_weight_MiB_AM', 'unique_weight_bytes', 1024**2),
                    ('hbm_byte_lower_bound_ms_AM', 'hbm_lower_bound_cycles', 1e6),
                    ('shared_hit_pct_AM', 'shared_hit_pct', 1),
                    ('dma_occupied_pct_AM', 'hbm_bus_busy_pct', 1),
                    ('hbm_payload_bw_util_pct_AM', 'hbm_payload_bw_util_pct', 1),
                    ('shared_evicted_MiB_AM', 'shared_evicted_bytes', 1024**2),
                    ('refetch_MiB_AM', 'refetch_bytes', 1024**2),
                    ('initial_resident_reload_MiB_AM', 'resident_reload_bytes', 1024**2),
                    ('total_reload_MiB_AM', 'total_reload_bytes', 1024**2),
                    ('core0_compute_ms_AM', 'core0_compute_cycles', 1e6),
                    ('core1_compute_ms_AM', 'core1_compute_cycles', 1e6),
                ):
                    out[dest] = aggregate(metrics, source, geometric=False) / scale
                out['total_ms_GM'] = aggregate(metrics, 'total_cycles') / 1e6
                assert math.isclose(out['total_ms_AM'], out['compute_ms_AM'] + out['stall_ms_AM'])
                breakdown.append(out)
    return main, breakdown


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'No rows for {path}')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    if not rows:
        return '无该条件的数据。'
    keys = list(rows[0])
    lines = ['| ' + ' | '.join(keys) + ' |', '|' + '|'.join('---' for _ in keys) + '|']
    for row in rows:
        formatted = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                value = f'{value:.6f}' if 'ms' in key else f'{value:.4f}'
            formatted.append(str(value).replace('|', '\\|'))
        lines.append('| ' + ' | '.join(formatted) + ' |')
    return '\n'.join(lines)


def chat_rows(rows):
    return [{
        'Batch': r['batch'], '单核 ms': r['single_ms'], '同构 ms': r['homo_ms'],
        '异构 ms': r['hetero_ms'], '异构相对单核加速比': r['hetero_speedup_vs_single'],
        '异构相对同构加速比': r['hetero_speedup_vs_homo'],
    } for r in rows]


def build_report(source, output, full_heldout=True):
    source, output = Path(source), Path(output)
    rows = load_rows(source)
    audit = validate_rows(rows, full_heldout=full_heldout)
    main, breakdown = main_tables(rows)
    output.mkdir(parents=True, exist_ok=True)
    ablation = [r for r in main if r['batch'] == '7-Batch GM' and r['bandwidth_gbs'] == 512]
    sensitivity = [r for r in main if r['policy'] in ('fair_lru', 'fair_pinned')]
    tables = {'main': main, 'breakdown': breakdown, 'policy_ablation': ablation, 'sensitivity': sensitivity}
    optional_hashes = {}
    latency_path = source.parent / 'latency_sensitivity.csv'
    if latency_path.exists():
        latency_rows = load_rows(latency_path)
        validate_rows(latency_rows, full_heldout=full_heldout)
        zero_latency, _ = main_tables(latency_rows)
        normal_latency = [r for r in main if r['policy'] == 'fair_lru' and r['bandwidth_gbs'] == 512]
        tables['request_latency_sensitivity'] = [dict(r, request_latency_ns=latency)
                                                  for latency, values in ((0, zero_latency), (64, normal_latency))
                                                  for r in values]
        optional_hashes['latency_sensitivity_sha256'] = hashlib.sha256(latency_path.read_bytes()).hexdigest()
    sram_path = source.parent / 'sram_port_necessary_bounds.csv'
    if sram_path.exists():
        with sram_path.open(newline='', encoding='utf-8') as handle:
            sram_rows = list(csv.DictReader(handle))
        sram_fields = ('ideal_compute_cycles', 'os_weight_sram_bytes', 'os_activation_sram_bytes',
                       'required_average_sram_TBs', 'port_floor_ms_at_8TBs',
                       'port_floor_ms_at_32TBs', 'port_floor_ms_at_128TBs')
        for row in sram_rows:
            row['batch'] = int(row['batch'])
            for field in sram_fields:
                row[field] = float(row[field])
        tables['sram_necessary_bounds'] = []
        for batch in [*BATCHES, '7-Batch GM']:
            for design in DESIGNS:
                selected = [r for r in sram_rows if r['design'] == design
                            and (isinstance(batch, str) or r['batch'] == batch)]
                if selected:
                    item = {'batch': batch, 'design': design}
                    item.update({field: aggregate(selected, field) for field in sram_fields})
                    tables['sram_necessary_bounds'].append(item)
        optional_hashes['sram_port_necessary_bounds_sha256'] = hashlib.sha256(sram_path.read_bytes()).hexdigest()
    for name, values in tables.items():
        csv_write(output / f'{name}.csv', values)
    audit.update({'input_csv': str(source.resolve()),
                  'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'bootstrap_replicates': 200, 'bootstrap_seed': 20261013,
                  'ranking_aggregation': 'GM within each batch, equally weighted GM across batches',
                  'breakdown_aggregation': 'AM within each batch, equally weighted AM across batches',
                  'clock_ghz': 1, 'total_MACs': 65536, 'total_SRAM_MiB': 64,
                  'weight_cache_MiB': 56, 'activation_and_accumulator_reserve_MiB': 8,
                  'physical_area_claim': False, 'native_simulator_calibrated': False})
    audit.update(optional_hashes)
    try:
        from openpyxl import Workbook
        book = Workbook()
        book.remove(book.active)
        for name, values in tables.items():
            sheet = book.create_sheet(name)
            fields = list(values[0])
            sheet.append(fields)
            for row in values:
                sheet.append([row.get(k) for k in fields])
            sheet.freeze_panes = 'A2'
            sheet.auto_filter.ref = sheet.dimensions
        book.save(output / 'CACHE_ROUTE_A_完整结果.xlsx')
        audit['excel_created'] = True
    except ImportError:
        audit['excel_created'] = False
    primary = [r for r in main if r['policy'] == 'fair_lru' and r['cache_state'] == 'cold' and r['bandwidth_gbs'] == 512]
    parts = [
        '# 有限缓存 Route A：固定硬件三组织对比',
        '本报告是**分析型事件仿真**，片上端口/互连仍理想化；不是 GPU 实测、RTL 或已校准的完整物理端到端结果。'
        '硬件形状沿用指定冻结点，本轮未在有限缓存条件下重新做硬件 DSE，因此“最优”只指此前选型范围。',
        '| 组织 | 物理尺寸 PM×PN×G | MAC 分配 |\n|---|---|---|\n'
        '| 单核 | 1×128×512 | 65,536 |\n| 同构 | 1×128×256 + 1×128×256 | 32,768 + 32,768 |\n'
        '| 异构 | 1×704×64 + 1×160×128 | 45,056 + 20,480 |',
        '**公平资源口径：**总 SRAM 64 MiB，统一预留 8 MiB 给激活/累加器，权重缓存 56 MiB；'
        '三种组织使用相同外部带宽及允许的 DMA/预取能力。等 MAC 与等 SRAM 容量不等于等面积。',
        '**冻结调度口径：**fair_lru/fair_pinned 均沿用先前冻结的 workload_aware、T=1；本轮不选新参数。'
        'proposed_asymmetric 的单核为 workload_aware T=1，同构为 greedy EFT T=2，异构为硬阈值 T=2；'
        'fair_pinned_hard 对全部组织统一硬阈值 T=2。',
        '**固定驻留并非排他私有：**56 MiB 权重区内固定 Shared 33 MiB，其余 23 MiB 是两核都可使用的流式空间，'
        '因为 Dense 核也需要读取 Hot 专家权重。预留 8 MiB 采用流式 Up/SiLU、原地 Z 与最终合并；'
        'B128 的解析工作区账本含本地寄存器与 1 MiB 元数据：单核 5.564 MiB、同构 7.095 MiB、异构 7.091 MiB，'
        '统一按不超过 7.2 MiB 核算，均位于 8 MiB 预留内。**本轮不计 SRAM 端口或片上互连等待。**',
        '**DMA 时序：**每个模型请求条带增加 64 ns 抽象服务开销，并非实测 HBM 控制器时序；'
        '独立的零固定延迟消融用于识别几何造成的请求次数差异。',
        '**协议修正：**单核也可由独立 DMA 与计算重叠，不能人为禁止；固定 Shared 的策略也必须给单核和同构。'
        'Shared 的 33 MiB 在 cold 实验中需要首次加载。shared_resident 是人为给定的相同初态，不代表真实跨层/跨请求 100% 命中。'
        '31 MiB 切成两个 15.5 MiB 缓冲不能各自装入一个 16.5 MiB 路由专家，本实现按块缓存和预取。',
        '## 公平主表：512 GB/s，冷缓存，三者均 LRU + 预取',
        markdown_table(chat_rows(primary)),
    ]
    overall = next((r for r in primary if r['batch'] == '7-Batch GM'), None)
    if overall:
        s = overall['hetero_speedup_vs_homo']
        lo, hi = overall['hetero_speedup_vs_homo_ci95_low'], overall['hetero_speedup_vs_homo_ci95_high']
        parts.append(f'异构相对同构配对加速比为 **{s:.6f}×**，200 次按 Batch 分层、按窗口配对 bootstrap 的 95% 区间 '
                     f'[{lo:.6f}, {hi:.6f}]。点估计{("达到" if s >= 1.15 else "未达到")} 1.15×；'
                     f'置信区间下界{("达到" if lo >= 1.15 else "未达到")} 1.15×。')
    for policy in ('fair_pinned', 'proposed_asymmetric', 'fair_pinned_hard', 'ablate_no_prefetch'):
        parts.append(f'## {POLICY_LABELS[policy]}：512 GB/s')
        for state in ('cold', 'shared_resident'):
            selected = [r for r in main if r['policy'] == policy and r['cache_state'] == state and r['bandwidth_gbs'] == 512]
            if selected:
                parts.extend([f'初态：`{state}`。', markdown_table(chat_rows(selected))])
    parts.extend([
        '**混杂对照的含义：**proposed_asymmetric 让单核不预取、同构使用贪心调度、异构固定 Shared 且硬阈值分流，'
        '同时改变了缓存、调度和预取权限。该组即使异构胜出，也不能归因于异构几何本身。',
        '## 访存与停顿拆解（512 GB/s，公平 LRU，cold）',
        '这里采用逐 Batch 等权**算术平均**，因此 `总时间 = 同一任务归属的理想计算时间 + 暴露的停顿`。'
        '停顿包含供数导致的等待及其传播，不是各核等待时间之和。',
        markdown_table([{'组织': LABELS[r['design']], '计算 ms': r['compute_ms_AM'],
                         '停顿 ms': r['stall_ms_AM'], '总时间 ms': r['total_ms_AM'],
                         'HBM MiB': r['hbm_MiB_AM'], 'Shared 字节命中 %': r['shared_hit_pct_AM'],
                         '有效外部带宽利用 %': r['hbm_payload_bw_util_pct_AM'],
                         'DMA 占用 %': r['dma_occupied_pct_AM'],
                         'Shared 驱逐 MiB': r['shared_evicted_MiB_AM'], '实际重读 MiB': r['total_reload_MiB_AM']}
                        for r in breakdown if r['policy'] == 'fair_lru' and r['cache_state'] == 'cold'
                        and r['bandwidth_gbs'] == 512 and r['batch'] == '7-Batch equal AM']),
        '**驱逐与重读分开：**一个权重块使用完后被逐出，不代表该层内需要再次读取；只有随后确实再次请求并从片外加载才计入 refetch。'
        'Shared 命中率按每个权重条带首次请求时已驻留的字节计数，不包括条带内部跨 M 的复用；'
        '因此 cold 的 0% 表示初次请求需要搬入，不表示每个 token 都要重读权重。'
        '有效外部带宽利用率 = 实际 HBM 字节/(峰值带宽×整层时间)；'
        'DMA 占用率还包含每请求固定延迟，两者不能互换。refetch 记录本窗口已加载后再次加载的字节，'
        'initial_resident_reload 单列初始驻留数据被逐出后重新加载；实际重读为两者之和。',
    ])
    if 'request_latency_sensitivity' in tables:
        parts.extend([
            '## 每条带固定 DMA 延迟敏感性：0 ns 与 64 ns',
            '同一 512 GB/s、fair_lru、相同冻结硬件与任务归属。每个条带承担一次固定开销；'
            '因此几何的请求数差异可能改变排序。此项是未校准的模型假设敏感性，不能解读为实际 HBM 延迟测量。',
            markdown_table([{'初态': r['cache_state'], '固定延迟 ns': r['request_latency_ns'],
                             '单核 GM ms': r['single_ms'], '同构 GM ms': r['homo_ms'],
                             '异构 GM ms': r['hetero_ms'], '异构相对单核加速比': r['hetero_speedup_vs_single'],
                             '异构相对同构加速比': r['hetero_speedup_vs_homo']}
                            for r in tables['request_latency_sensitivity'] if r['batch'] == '7-Batch GM']),
        ])
    if 'sram_necessary_bounds' in tables:
        parts.extend([
            '## 片上供数必要条件反推（不是端口计时实验）',
            '下面按已声明 OS 操作数读取账本计算权重与激活的供数要求，并给出有限聚合读带宽对应的服务下界。'
            'OS 的循环次序和核内权重复用会改变此账本；当前外存仿真采用条带优先、条带内处理全部 M，'
            '所以这张旧 M 外循环操作数账本不是当前缓存命中次数，也未纳入主表时序。'
            '它不含累加器端口、跨核互连、广播和 bank 冲突，不能作为足够条件。',
            markdown_table([{'组织': LABELS[r['design']], '理想计算 GM ms': r['ideal_compute_cycles'] / 1e6,
                             '平均所需片上带宽 GM TB/s': r['required_average_sram_TBs'],
                             '8 TB/s 端口下界 GM ms': r['port_floor_ms_at_8TBs'],
                             '32 TB/s 端口下界 GM ms': r['port_floor_ms_at_32TBs'],
                             '128 TB/s 端口下界 GM ms': r['port_floor_ms_at_128TBs']}
                            for r in tables['sram_necessary_bounds'] if r['batch'] == '7-Batch GM']),
        ])
    parts.extend([
        '## 统计与复现',
        f'输入 {audit["points"]} 点，{audit["window_count"]} 个不同留出窗口。每个 Batch 先对窗口取 GM，再对 7 个 Batch 等权取 GM；'
        '速度比较基于同窗口配对比值。bootstrap 只表达当前 trace 样本波动，不代表仿真模型误差。',
        '冷缓存下逐窗口检查 `HBM 字节 ≥ 唯一权重字节` 与 `总时间 ≥ 实际 HBM 字节/带宽`；'
        'warm 初态只允许减去已驻留的 Shared 字节。额外容量、无限带宽或负停顿不被用于制造结果。',
        '完整逐 Batch、策略与带宽扫描见 `main.csv`、`breakdown.csv`、`policy_ablation.csv`、`sensitivity.csv`；'
        '原始逐窗口输入见 `per_window.csv`。硬件效率、能耗和面积结论仍需带有限片上端口/时序的模型及综合校准。',
    ])
    (output / 'REPORT.md').write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
    (output / 'REPORT_AUDIT.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true', help='Only for unit/smoke tests; skips 135-window count check')
    args = parser.parse_args()
    print(json.dumps(build_report(args.input, args.output, not args.allow_partial), indent=2))


if __name__ == '__main__':
    main()
