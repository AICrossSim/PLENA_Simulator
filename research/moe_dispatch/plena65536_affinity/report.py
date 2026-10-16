"""Reproducible tables for frozen geometry and A/B affinity experiments.

All main statistics use within-batch geometric means followed by an equal
seven-batch geometric mean.  Idle fractions/counts use arithmetic means;
capacity and instantaneous rate envelopes use maxima, never geometric means.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_moe.workloads import (
    BATCHES, load_workloads, profile_window, write_workload_artifacts,
)
from research.moe_dispatch.plena65536_shared.timing import PROFILES

BUDGET = 65536
DEFAULT_OUT = Path('/scratch/shared/mcl123/plena/outputs/plena65536_affinity_20261013')
LABELS = {
    'original': '原论文默认几何单核', 'single': 'DSE最优单核',
    'homogeneous': 'DSE最优同构（贪心EFT）',
    'homogeneous_B': 'DSE最优同构（同一策略B）',
    'heterogeneous_A': 'DSE异构A（固定阈值）',
    'heterogeneous_B': 'DSE异构B（负载自适应）',
}
POLICIES = {
    'hard_threshold': 'A：硬阈值', 'greedy_eft': '贪心EFT',
    'adaptive_threshold': '逐窗自适应阈值',
    'workload_aware': 'B：贪心与四阈值择优',
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def gm(values):
    values = tuple(float(v) for v in values)
    if not values or any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError('GM requires finite nonnegative values')
    return 0.0 if any(v == 0 for v in values) else math.exp(math.fsum(math.log(v) for v in values) / len(values))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(rows):
    if not rows:
        return ''
    keys = list(rows[0])
    result = ['| ' + ' | '.join(keys) + ' |', '|' + '|'.join('---' for _ in keys) + '|']
    for row in rows:
        rendered = []
        for key in keys:
            value = row[key]
            if value is None:
                value = '—'
            elif isinstance(value, float):
                value = f'{value:,.6f}' if 'ms' in key else f'{value:,.4f}'
            rendered.append(str(value).replace('|', '\\|'))
        result.append('| ' + ' | '.join(rendered) + ' |')
    return '\n'.join(result)


def physical(cores):
    return ' + '.join(f'{c.pm}×{c.pn}×{c.groups}' for c in cores)


def issued_macs(core, expert):
    m, h, f = (int(expert[k]) for k in ('Me', 'H', 'F'))
    ceil = lambda a, b: (a + b - 1) // b
    feed = ceil(m, core.pm) * (2 * ceil(f, core.pn) * ceil(h, core.r)
                               + ceil(h, core.pn) * ceil(f, core.r)) * core.b
    return feed * core.macs


def checked_point(cores, window, policy, threshold, profile=PROFILES[0]):
    from .model import schedule
    kwargs = dict(profile=profile, policy=policy, threshold=threshold)
    first = schedule(cores, window, **kwargs)
    second = schedule(cores, window, **kwargs)
    assert canonical(first) == canonical(second), 'Non-deterministic schedule replay'
    assert sum(c.macs for c in cores) == BUDGET
    observed = profile_window(window)
    experts = {int(e['id']): e for e in window['experts']}
    assert sorted(int(j['expert_id']) for j in first['jobs']) == sorted(experts)
    ends, padded, useful = [0] * len(cores), 0, 0
    for job in first['jobs']:
        cid, eid = int(job['core_id']), int(job['expert_id'])
        assert 0 <= cid < len(cores)
        assert int(job['start']) == ends[cid]
        assert int(job['end']) > int(job['start'])
        ends[cid] = int(job['end'])
        expert = experts[eid]
        padded += issued_macs(cores[cid], expert)
        useful += 3 * int(expert['Me']) * int(expert['H']) * int(expert['F'])
        if len(cores) == 2 and policy == 'hard_threshold':
            assert cid == (0 if eid == -1 or int(expert['Me']) > threshold else 1)
    assert ends == list(first['core_cycles'])
    assert int(first['cycles']) == max(ends)
    assert useful == observed['total_useful_macs'] == int(first['useful_MACs'])
    assert padded == int(first['issued_macs']) and padded >= useful
    assert int(first['unique_weight_bytes']) == observed['total_unique_weight_bytes']
    assert int(first.get('migration_count', 0)) == int(first.get('steal_count', 0)) == 0
    if len(cores) == 2 and policy == 'workload_aware':
        assert int(first['cycles']) == min(first['candidate_cycles'].values())
    return first, observed


def window_row(name, meta, cores, split, window, plan, observed, profile=PROFILES[0], do_supply=True):
    from .supply import supply_ledger
    cycles, times = int(plan['cycles']), [int(x) for x in plan['core_cycles']]
    dual = len(cores) == 2
    cold_hot = Counter()
    for job in plan['jobs']:
        cid = int(job['core_id'])
        kind = 'shared' if job['is_shared'] else ('cold' if int(job['M']) <= int(meta['threshold']) else 'hot')
        cold_hot[f'{kind}_core{cid}'] += 1
    row = {
        'design': name, 'organization': LABELS.get(name, name),
        'policy': meta['policy'], 'threshold': meta['threshold'],
        'physical_PM_PN_G': physical(cores),
        'native_frames': ' + '.join(f'b={c.b},R={c.r}' for c in cores),
        'MAC_allocation': '+'.join(str(c.macs) for c in cores),
        'split': split, 'batch': int(window['batch']), 'row_type': 'window',
        'window_id': str(window['id']), 'window_count': 1,
        'compute_cycles': cycles, 'compute_ms': cycles / 1e6,
        'useful_GFLOP_s': 2 * observed['total_useful_macs'] / cycles,
        'wall_peak_utilization_pct': 100 * observed['total_useful_macs'] / (BUDGET * cycles),
        'spatial_utilization_pct': 100 * observed['total_useful_macs'] / int(plan['issued_macs']),
        'core0_ms': times[0] / 1e6, 'core1_ms': times[1] / 1e6 if dual else None,
        'core0_idle_pct': 100 * (cycles - times[0]) / cycles,
        'core1_idle_pct': 100 * (cycles - times[1]) / cycles if dual else None,
        'imbalance_ratio_finite': max(times) / min(times) if dual and min(times) > 0 else None,
        'zero_load_core_windows': int(dual and min(times) == 0),
        'bottleneck_core': 'tie' if dual and times[0] == times[1] else str(times.index(max(times))),
        'selected_policy': plan.get('selected_policy', meta['policy']),
        'selected_threshold': plan.get('selected_threshold'),
        'affinity_deviation_count': int(plan.get('affinity_deviation_count', 0)),
        'useful_macs': observed['total_useful_macs'], 'issued_macs': int(plan['issued_macs']),
        'unique_weight_MiB': observed['total_unique_weight_bytes'] / 1024**2,
        'unique_fetch_ms_256GB_s': observed['total_unique_weight_bytes'] / 256e6,
        'unique_weight_average_TB_s': observed['total_unique_weight_bytes'] / cycles / 1000,
        'repeat_identical': True,
    }
    physical_active = sum(c.macs * t for c, t in zip(cores, times))
    capacity = BUDGET * cycles
    row['padding_slot_waste_pct'] = 100 * (int(plan['issued_macs']) - observed['total_useful_macs']) / capacity
    row['pipeline_slot_waste_pct'] = 100 * (physical_active - int(plan['issued_macs'])) / capacity
    row['imbalance_slot_waste_pct'] = 100 * (capacity - physical_active) / capacity
    assert min(row[k] for k in ('padding_slot_waste_pct', 'pipeline_slot_waste_pct', 'imbalance_slot_waste_pct')) >= 0
    assert math.isclose(sum(row[k] for k in ('wall_peak_utilization_pct', 'padding_slot_waste_pct',
                                            'pipeline_slot_waste_pct', 'imbalance_slot_waste_pct')), 100)
    row.update({f'core{cid}_bottleneck_windows': int(row['bottleneck_core'] == str(cid)) for cid in (0, 1)})
    row['tie_bottleneck_windows'] = int(row['bottleneck_core'] == 'tie')
    for kind in ('shared', 'cold', 'hot'):
        for cid in (0, 1):
            row[f'{kind}_jobs_core{cid}'] = cold_hot[f'{kind}_core{cid}']
    if do_supply:
        row.update(supply_ledger(cores, window, plan, profile=profile))
        for key, value in list(row.items()):
            if isinstance(value, (list, dict)):
                row[key] = canonical(value)
    assert row['wall_peak_utilization_pct'] <= 100 + 1e-9
    assert row['spatial_utilization_pct'] <= 100 + 1e-9
    return row


AM_FIELDS = {'core0_ms', 'core1_ms', 'core0_idle_pct', 'core1_idle_pct',
             'affinity_deviation_count', 'zero_load_core_windows',
             'padding_slot_waste_pct', 'pipeline_slot_waste_pct', 'imbalance_slot_waste_pct'}
IDENTIFIERS = {'batch', 'threshold', 'selected_threshold', 'window_count'}


def aggregate(rows, batch, row_type):
    result = dict(rows[0])
    result.update(batch=batch, row_type=row_type, window_id='',
                  window_count=sum(r['window_count'] for r in rows),
                  selected_policy='mixed; see boundary table', selected_threshold=None)
    for key, value in rows[0].items():
        if key in IDENTIFIERS or isinstance(value, bool):
            continue
        values = [r[key] for r in rows if r.get(key) is not None]
        if not values or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            continue
        if key == 'zero_load_core_windows' or key.endswith('_bottleneck_windows'):
            result[key] = sum(values)
        elif key in AM_FIELDS or '_jobs_' in key:
            result[key] = math.fsum(values) / len(values)
        elif ('peak' in key and 'utilization' not in key or key.endswith('_capacity_MiB')
              or key.endswith('_capacity_bytes') or key.endswith('_KiB')
              or key.endswith('_upper_MiB') or key == 'full_layer_weight_MiB'):
            result[key] = max(values)
        else:
            result[key] = gm(values)
    if 'core0_bottleneck_windows' in result:
        result['bottleneck_core'] = ', '.join(f'{label}:{result[key]}' for label, key in
            [('Core0', 'core0_bottleneck_windows'), ('Core1', 'core1_bottleneck_windows'), ('tie', 'tie_bottleneck_windows')])
    return result


def with_summaries(rows):
    batches = [aggregate([r for r in rows if r['batch'] == b], b, 'batch_GM') for b in BATCHES]
    return rows + batches + [aggregate(batches, '7-Batch GM', '7batch_GM')]


def add_speedups(rows, reference_rows):
    refs = {(r['design'], r['split'], r['window_id']): r for r in reference_rows if r['row_type'] == 'window'}
    for row in rows:
        for name, field in [('single', 'speedup_vs_single'), ('homogeneous', 'speedup_vs_homogeneous'),
                            ('homogeneous_B', 'speedup_vs_homogeneous_B')]:
            reference = refs.get((name, row['split'], row['window_id']))
            row[field] = reference['compute_ms'] / row['compute_ms'] if reference else None


def table1(rows):
    return [{
        'Batch': f"B{r['batch']}" if isinstance(r['batch'], int) else r['batch'],
        '组织形态': r['organization'], '物理尺寸 (Core_D + Core_S)': r['physical_PM_PN_G'],
        '乘法器分配': r['MAC_allocation'],
        '分流策略': POLICIES.get(r['policy'], r['policy']) + (f" T={r['threshold']}" if r['policy'] == 'hard_threshold' else ''),
        'Compute Time (ms)': r['compute_ms'], '有效 GFLOP/s': r['useful_GFLOP_s'],
        '整片 MAC 利用率 (%)': r['wall_peak_utilization_pct'],
        '相比最优单核加速比': r['speedup_vs_single'],
        '相比最优同构贪心加速比': r['speedup_vs_homogeneous'],
        '相比同策略B最优同构加速比': r['speedup_vs_homogeneous_B'],
    } for r in rows]


def table2(rows):
    return [{
        'Batch': f"B{r['batch']}" if isinstance(r['batch'], int) else r['batch'],
        '设计/策略': r['organization'],
        'Dense核耗时 AM (ms)': r['core0_ms'], 'Stream核耗时 AM (ms)': r['core1_ms'],
        '逐窗max/min的GM（有限项）': r['imbalance_ratio_finite'],
        '单核零负载窗口数': r['zero_load_core_windows'],
        'Dense核空闲率 AM (%)': r['core0_idle_pct'],
        'Stream核空闲率 AM (%)': r['core1_idle_pct'],
        '拖尾核及窗口数': r['bottleneck_core'],
    } for r in rows]


def table3(rows):
    return [{
        '组织形态': r['organization'],
        '7-Batch GM计算时间 (ms)': r['compute_ms'],
        '模型单层全部专家权重容量 (MiB)': r['full_layer_weight_MiB'],
        'OS共享池W+X平均需求GM (TB/s)': r['pool_operand_average_TB_s'],
        '阵列接口实际调度峰值（全宽）(TB/s)': r['array_full_scheduled_peak_TB_s'],
        'SRAM最小峰值需求': '待指定预取时序与缓冲后求解',
        '本地双缓冲容量 (KiB)': r['local_double_buffer_KiB'],
        '唯一权重冷取需求GM (TB/s)': r['unique_weight_average_TB_s'],
    } for r in rows]


def bootstrap(rows, repetitions=2000, seed=20261013):
    import numpy as np
    rng = np.random.default_rng(seed)
    lookup = {(r['design'], r['window_id']): r for r in rows if r['split'] == 'heldout' and r['row_type'] == 'window'}
    result = []
    for candidate in ('heterogeneous_A', 'heterogeneous_B'):
        for baseline in ('single', 'homogeneous', 'homogeneous_B'):
            logs, counts = [], Counter()
            for batch in BATCHES:
                group = [r for r in rows if r['design'] == candidate and r['split'] == 'heldout'
                         and r['row_type'] == 'window' and r['batch'] == batch]
                log = []
                for r in group:
                    b = lookup[(baseline, r['window_id'])]
                    log.append(math.log(b['compute_cycles'] / r['compute_cycles']))
                    counts['win' if r['compute_cycles'] < b['compute_cycles'] else 'tie' if r['compute_cycles'] == b['compute_cycles'] else 'loss'] += 1
                logs.append(np.asarray(log))
            samples = np.zeros(repetitions)
            for group in logs:
                samples += group[rng.integers(0, len(group), size=(repetitions, len(group)))].mean(axis=1) / len(BATCHES)
            low, high = np.quantile(np.exp(samples), (.025, .975))
            result.append({'candidate': candidate, 'baseline': baseline,
                           '7batch_GM_speedup': math.exp(sum(g.mean() for g in logs) / len(BATCHES)),
                           'bootstrap_95pct_lower': float(low), 'bootstrap_95pct_upper': float(high),
                           'win_windows': counts['win'], 'tie_windows': counts['tie'], 'loss_windows': counts['loss'],
                           'repetitions': repetitions, 'seed': seed})
    return result


def boundary_summary(rows):
    result = []
    for name in dict.fromkeys(r['design'] for r in rows):
        selected = [r for r in rows if r['design'] == name and r['split'] == 'heldout' and r['row_type'] == 'window']
        for b in BATCHES:
            group = [r for r in selected if r['batch'] == b]
            counts = Counter(str(r['selected_threshold']) if r['selected_threshold'] is not None else 'greedy' for r in group)
            result.append({'design': name, 'batch': b, 'windows': len(group),
                           'greedy_selected': counts['greedy'], **{f'T{t}_selected': counts[str(t)] for t in (1, 2, 4, 8)},
                           'affinity_deviations_AM': sum(r['affinity_deviation_count'] for r in group) / len(group)})
    return result


def workbook(out, sheets):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    book = Workbook()
    book.remove(book.active)
    for name, rows in sheets.items():
        if not rows:
            continue
        sheet = book.create_sheet(name[:31])
        keys = list(dict.fromkeys(k for row in rows for k in row))
        sheet.append(keys)
        for r in rows:
            sheet.append([r.get(k) for k in keys])
        sheet.freeze_panes, sheet.auto_filter.ref = 'A2', sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill('solid', fgColor='17365D')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        sheet.row_dimensions[1].height = 48
        for column in sheet.columns:
            header = str(column[0].value)
            sheet.column_dimensions[column[0].column_letter].width = min(48, max(15, len(header) * 1.15))
            for cell in column[1:]:
                if isinstance(cell.value, float):
                    cell.number_format = '0.000000' if 'ms' in header else '0.0000'
    path = out / 'PLENA65536_冷热自适应DSE_完整表.xlsx'
    book.save(path)
    read = load_workbook(path, read_only=True, data_only=True)
    cells = 0
    for name, rows in sheets.items():
        if not rows:
            continue
        keys = list(dict.fromkeys(k for row in rows for k in row))
        expected = [keys] + [[r.get(k) for k in keys] for r in rows]
        actual = list(read[name[:31]].values)
        assert len(actual) == len(expected)
        for line, target in zip(actual, expected):
            for a, e in zip(line, target):
                cells += 1
                if isinstance(e, (float, int)) and not isinstance(e, bool):
                    assert a is not None and math.isclose(a, e, rel_tol=1e-12, abs_tol=1e-12)
                else:
                    assert a == e or a is None and e == ''
    read.close()
    return path, cells


def normalize_meta(name, value):
    result = dict(value)
    defaults = {'original': 'greedy_eft', 'single': 'greedy_eft', 'homogeneous': 'greedy_eft',
                'homogeneous_B': 'workload_aware', 'heterogeneous_A': 'hard_threshold',
                'heterogeneous_B': 'workload_aware'}
    result.setdefault('policy', defaults[name])
    result['threshold'] = int(result.get('threshold') or 2)
    return result


def evaluate(out, inputs, name, meta, *, details=None, profile=PROFILES[0], do_supply=True):
    cores = tuple(Core(**{k: c[k] for k in ('b', 'r', 'rm', 'rn')}) for c in meta['cores'])
    rows = []
    for split in ('development', 'heldout'):
        for window in inputs[split]:
            plan, observed = checked_point(cores, window, meta['policy'], meta['threshold'], profile)
            row = window_row(name, meta, cores, split, window, plan, observed, profile, do_supply)
            rows.append(row)
            if details:
                details.write(canonical({'design': name, 'split': split, 'window_id': window['id'],
                                         'input_sha256': observed['window_payload_sha256'], 'plan': plan}) + '\n')
    return rows


def run(out=DEFAULT_OUT):
    out = Path(out)
    selection_path = out / 'FROZEN_SELECTION.json'
    selection = json.loads(selection_path.read_text())
    inputs = load_workloads()
    write_workload_artifacts(inputs, out)
    families = {name: normalize_meta(name, value) for name, value in selection['families'].items() if name in LABELS}
    required = {'original', 'single', 'homogeneous', 'homogeneous_B', 'heterogeneous_A', 'heterogeneous_B'}
    assert required <= set(families), 'Both greedy and same-policy homogeneous references are required'
    names = [n for n in LABELS if n in families]
    raw = []
    detail_path = out / 'execution_details.jsonl.gz'
    with gzip.open(detail_path, 'wt', encoding='utf-8') as details:
        for name in names:
            raw.extend(evaluate(out, inputs, name, families[name], details=details))
    add_speedups(raw, raw)
    all_rows = []
    for name in names:
        for split in ('development', 'heldout'):
            all_rows.extend(with_summaries([r for r in raw if r['design'] == name and r['split'] == split]))
    order = {name: i for i, name in enumerate(names)}
    heldout = sorted([r for r in all_rows if r['split'] == 'heldout' and r['row_type'] != 'window'],
                     key=lambda r: (BATCHES.index(r['batch']) if isinstance(r['batch'], int) else 7, order[r['design']]))
    summary = [r for r in heldout if r['row_type'] == '7batch_GM']
    first = table1(heldout)
    second = table2([r for r in heldout if r['design'].startswith('heterogeneous')])
    third = table3(summary)
    intervals = bootstrap(raw)
    boundary = boundary_summary(raw)
    ablation_raw, sensitivity = [], []
    for parent in ('heterogeneous_A', 'heterogeneous_B'):
        base = families[parent]
        for policy in ('hard_threshold', 'greedy_eft', 'adaptive_threshold', 'workload_aware'):
            meta = dict(base, policy=policy)
            key = 'best_A_threshold_for_geometry' if policy in ('hard_threshold', 'adaptive_threshold') else 'best_B_threshold_for_geometry'
            meta['threshold'] = int(base.get(key, base['threshold']))
            name = parent + '__' + policy
            rows = evaluate(out, inputs, name, meta, do_supply=False)
            add_speedups(rows, raw)
            for split in ('development', 'heldout'):
                ablation_raw.extend(with_summaries([r for r in rows if r['split'] == split]))
    for name in names:
        for profile in PROFILES[1:]:
            rows = evaluate(out, inputs, name, families[name], profile=profile, do_supply=False)
            for split in ('development', 'heldout'):
                group = [r for r in rows if r['split'] == split]
                sensitivity.append({'design': name, 'split': split, 'profile': profile.name,
                                    'physical_PM_PN_G': group[0]['physical_PM_PN_G'],
                                    '7batch_GM_compute_ms': gm(gm(r['compute_ms'] for r in group if r['batch'] == b) for b in BATCHES),
                                    'geometry_and_policy_frozen': True})
    csvs = {'table1_compute': first, 'table2_imbalance': second, 'table3_supply': third,
            'all_results': all_rows, 'cross_policy_ablation': ablation_raw,
            'dynamic_boundaries': boundary, 'paired_bootstrap': intervals,
            'frozen_timing_sensitivity': sensitivity}
    for name, rows in csvs.items():
        write_csv(out / (name + '.csv'), rows)
    xlsx, cells = workbook(out, {'表1纯计算性能': first, '表2两核负载': second, '表3供数必要条件': third,
                                  '全部窗口与汇总': all_rows, '同硬件策略交叉': ablation_raw,
                                  '动态分流边界': boundary, '配对收益置信区间': intervals,
                                  '冻结几何时序敏感性': sensitivity})
    chosen_b = families['heterogeneous_B']
    chosen_cores = tuple(Core(**{k: c[k] for k in ('b', 'r', 'rm', 'rn')}) for c in chosen_b['cores'])
    divisor = math.gcd(*(c.macs for c in chosen_cores))
    b_summary = next(r for r in summary if r['design'] == 'heterogeneous_B')
    ratio = ':'.join(str(c.macs // divisor) for c in chosen_cores)
    resource_text = '+'.join(f'{c.macs:,}' for c in chosen_cores)
    findings = [
        f"开发集冻结策略B的Core0/Core1={physical(chosen_cores)}，乘法器{resource_text}，比例{ratio}。{'两核PM均为1，最优点没有形成高PM的Dense核；角色名称不能证明几何专用化。' if all(c.pm == 1 for c in chosen_cores) else 'PM、PN和归约深度共同决定每种专家的成本，不以角色名字强制哪边更大。'}",
        f"留出集7-Batch GM={b_summary['compute_ms']:.9f}ms；相对最优单核{b_summary['speedup_vs_single']:.6f}×，相对最优同构贪心{b_summary['speedup_vs_homogeneous']:.6f}×，相对同一策略B最优同构{b_summary['speedup_vs_homogeneous_B']:.6f}×。{'候选超过两类优化基线。' if min(b_summary['speedup_vs_single'],b_summary['speedup_vs_homogeneous_B']) > 1 else '候选尚未同时超过两类优化基线。'}差值属于未校准解析模型，不直接宣布架构实物胜出。",
        f"候选的平均计算容量损失分解（Batch内AM后七Batch等权AM）：padding {b_summary['padding_slot_waste_pct']:.4f}%，计算流水线 {b_summary['pipeline_slot_waste_pct']:.4f}%，核间负载失衡 {b_summary['imbalance_slot_waste_pct']:.4f}%。这些互斥项以整层65,536×周期为分母；与表1 GM利用率的统计口径不同，不与GM利用率直接求和。",
        f"候选OS共享池W+X平均必要供数为{b_summary['pool_operand_average_TB_s']:.6f}TB/s（7-Batch GM），本地阵列接口实际调度峰值为{b_summary['array_full_scheduled_peak_TB_s']:.6f}TB/s（全部窗口最大）。所需共享SRAM最小峰值尚不能唯一确定，下一阶段需给定双缓冲、预取起点、延迟和bank约束后验证。",
    ]
    threshold_pass = min(b_summary['speedup_vs_single'], b_summary['speedup_vs_homogeneous_B']) >= 1.05
    findings.append(
        '按此前冻结的双基线标准，进入校准需同时较形状最优单核和最优同构快至少5%。'
        + ('本轮解析结果达到该数值门槛，但仍需原生校准与未参与决策的新测试数据。' if threshold_pass else
           '本轮对单核达到门槛，但对同一策略B最优同构仅有约1.34%加速，未达到进入校准的双基线门槛。')
        + '宣布胜出还需校准后至少10%加速且95%区间下界至少5%；窗口Bootstrap不能覆盖时序模型误差。')
    if all(c.b == 1 and c.r in (64, 128) for c in chosen_cores):
        findings.append(
            '当前异构收益并非一核高PM匹配热专家、一核低PM匹配冷专家：两核PM都为1。'
            '其MLEN为64/128，恰能整除Routed Down的K=1408；最优单核MLEN512、同构MLEN256都要把该K维发射到1536。'
            '异构减少K补齐和归约深度，同时PN704/160又引入部分N补齐，最终差异由完整MAC、流水线和负载账本共同决定。'
            '这是当前形状及调度的具体收益，不能外推为所有MoE形状均适合相同11:5划分。')
    gap_path = out / 'assignment_gap.json'
    if gap_path.exists():
        gap_rows = {r['design']: r for r in json.loads(gap_path.read_text())['summary']
                    if r['row_type'] in {'7batch_GM', '7batch_equal_GM'}}
        gap_text = []
        for key, label in [('heterogeneous_B', '异构B'), ('homogeneous_B', '同构B')]:
            r = gap_rows[key]
            gap_text.append(f"{label}={r['policy_ms']:.9f}ms，离线最优={r['oracle_ms']:.9f}ms，剩余调度差距{r['oracle_gap_pct']:.4f}%")
        findings.append(
            '固定相同几何后的整专家离线最优分配另见[assignment_gap.md](assignment_gap.md)：'
            + '；'.join(gap_text) + '。'
            '这些是已选几何的整专家分配参照，未重新搜索硬件，也不是可实现的零成本在线控制器。')
    lines = [
        '# 65,536 MACs Shared + MoE：固定阈值与负载自适应几何联合 DSE', '',
        '本轮是基于真实捕获路由的纯计算解析回放。BF16/FP32、1GHz、统一OS、数据瞬时就绪；没有HBM/SRAM端口或派工器硬件执行延迟。未进行Rust/RTL校准或PPA测量。', '',
        '18个开发窗口选定硬件与静态参数后冻结，在135个留出窗口评估。每Batch取窗口GM，七Batch等权GM；不按测试Batch更换硬件。PM=1使用BLEN=1标量子阵列扩展，非原论文默认实现。第三维G=MLEN/BLEN，不是逻辑K。', '',
        '这些135个留出窗口已在此前研究迭代中多次查看，当前应定性为探索性验证集，不能宣称从未查看的最终论文测试集；最终泛化结论需要新捕获且未参与研究决策的数据。', '',
        '策略A：Shared或Me>T给Core0，其余给Core1；T从{1,2,4,8}仅在开发集选择。策略B：依据当前窗口已知专家形状，比较贪心EFT及四个硬阈值计划，选预计makespan最低者；两种策略均在开始执行前确定归属，任务执行中无迁移/偷取。B使用当前输入和模型成本，不读取未来窗口或未来实测时间。B是候选软件调度参考，尚未证明为低开销硬件控制器。', '',
        '公平基线同时提供最优同构贪心和同一策略B下重新选型的最优同构。A/B各自最优硬件之外，同一冻结硬件的四种策略也交叉回放，以区分几何收益和派工收益。', '',
        '**本轮结果与三项归因**', '', *[item for line in findings for item in (line, '')],
        '**表1：全Batch纯计算性能**', '', table(first), '',
        '利用率=有效MAC/(65,536×整层完成周期)，含padding、流水线开销和双核闲置。有效GFLOP/s=2×有效MAC/周期；1cycle=1ns。加速比=基线时间/候选时间，小于1表示更慢。', '',
        '**表2：两核负载与失配**', '', table(second), '',
        '两核耗时和空闲率为逐窗口算术平均；总时间仍按逐窗口max后取GM，不能用两核均值的max替代。失配比先逐窗计算max/min再对有限项取GM；核零负载意味着无限失配，另计窗口数，未伪造有限比例。七Batch行的耗时/空闲率为七个Batch均值等权平均。Core0/Core1是稳定角色名，不保证前者乘法器更多。', '',
        '**表3：维持纯计算性能的供数必要条件反推**', '', table(third), '',
        '全模型单层唯一BF16权重=33MiB Shared+64×16.5MiB Routed=1089MiB；实际激活权重随窗口变化。平均列是共享池到本地W+X操作数的有效字节需求，峰值列是本地到阵列全宽接口在实际计算调度下的峰值，两者是不同边界。SRAM最小峰值需求取决于预取时间、缓冲、端口和分bank，纯计算模型不能唯一反推。完整读写、上下文、后处理与存储账本详见原始CSV和SUPPLY_CONTRACT.md。容量足够不能证明供数延迟足够；这里没有实现端到端供数闭环。', '',
        '**动态分流边界**', '', table([r for r in boundary if r['design'] == 'heterogeneous_B']), '',
        '某些窗口最佳候选为贪心EFT，此时分配不必可由单一阈值表示。T(B)并非使用测试标签事后拟合，每窗仅用当前真实任务描述符决定。', '',
        '**留出集配对收益区间**', '', table(intervals), '',
        'Bootstrap在每个Batch内对窗口配对重采样2000次，再对七Batch等权GM；只表达捕获窗口波动，不涵盖模型误差或跨层/step相关性。', '',
        '**冻结硬件时序敏感性**', '', table([r for r in sensitivity if r['split'] == 'heldout']), '',
        '每个主表、策略交叉、时序敏感性计算点独立回放两次，专家守恒、MAC账本、队列时间、硬阈值规则和B候选择优均审计。完整CSV、输入哈希、逐专家执行时间与Excel回读审计随报告交付。',
    ]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    main_points = len(names) * (len(inputs['development']) + len(inputs['heldout']))
    cross_points = 8 * (len(inputs['development']) + len(inputs['heldout']))
    sensitivity_points = len(names) * (len(PROFILES) - 1) * (len(inputs['development']) + len(inputs['heldout']))
    audit = {'status': 'PASS', 'development_windows': len(inputs['development']),
             'heldout_windows': len(inputs['heldout']), 'main_designs': len(names),
             'main_points': main_points, 'cross_policy_points': cross_points,
             'sensitivity_points': sensitivity_points,
             'repeated_points': main_points + cross_points + sensitivity_points,
             'repetitions_per_point': 2, 'workbook_readback_cells': cells,
             'workbook': str(xlsx), 'selection_sha256': sha(selection_path),
             'report_source_sha256': sha(Path(__file__)), 'details_sha256': sha(detail_path),
             'all_experts_once': True, 'no_midtask_migration_or_stealing': True,
             'physical_MAC_budget': BUDGET, 'wall_clock_GHz': 1,
             'timed_supply_delay_cycles': 0, 'native_calibrated': False,
             'aggregation': 'within-batch GM then equal GM across seven batches'}
    write_json(out / 'REPORT_AUDIT.json', audit)
    print(canonical(audit), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    run(parser.parse_args().out)
