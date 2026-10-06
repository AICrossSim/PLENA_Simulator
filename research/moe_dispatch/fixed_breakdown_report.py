"""Export the fixed-shape BF16 comparison with explicitly scoped diagnostics.

All figures are analytical post-router FFN results. Independent service
demands and counterfactual timings are never stacked into wall-clock time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from .geometry3d.study import canonical, cores_from
from .regime.campaign import evaluate, load_inputs
from .regime.metrics import bounds, core_rows
from .regime.resources import Partition
from .regime.search import point, settings_for


BATCHES = (2, 4, 8, 16)
DESIGNS = {
    'a_single6': ('单核 6', '6x4x512'),
    'b_homogeneous3+3': ('同构 3+3', '3x4x512+3x4x512'),
    'c_heterogeneous4+2': ('异构 4+2', '2x4x512+4x4x512'),
}
ORACLES = ('ideal_HBM', 'ideal_ports', 'zero_control', 'compute_only')
GROUP_CASES = ('GU_only', 'Down_only', 'both_halved', 'lookahead_guard')
COLORS = ('#2563ad', '#df8c24', '#27876a')
PHASE_IDENTITY = ('core', 'expert_index', 'phase', 'compute', 'issues',
                  'useful_macs', 'issued_macs', 'hbm_bytes', 'weight_unique',
                  'w_port_bytes', 'x_port_bytes', 'acc_port_bytes',
                  'activation_bytes', 'peak_w_bytes', 'peak_x_bytes',
                  'peak_accumulator_bytes', 'decoder_elements')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_file(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def make_settings(organization):
    cores = cores_from(DESIGNS[organization][1])
    fraction = cores[0].pm / sum(c.pm for c in cores)
    partition = None if len(cores) == 1 else Partition(
        w=.5, x=fraction, acc=fraction, z=fraction,
        wb=.5, xb=fraction, ab=fraction)
    return cores, settings_for(256, 'BF16', point(cores, partition=partition, gu=2, down=4))


def verify_oracle(charged, alternative):
    fields = ('hbm_bytes', 'native_unique_bytes', 'useful_macs', 'issued_macs',
              'X_sram_bytes', 'global_activation_bytes', 'budget', 'compulsory_traffic')
    assert charged['workload'] == alternative['workload']
    for field in fields:
        assert canonical(charged[field]) == canonical(alternative[field]), field
    assert [b['core'] for b in charged['bindings']] == [b['core'] for b in alternative['bindings']]
    phase_keys = lambda r: sorted(tuple(p[k] for k in PHASE_IDENTITY) for p in r['phases'])
    assert phase_keys(charged) == phase_keys(alternative)


def lookahead_limits(settings, budget, workloads):
    """Static compiler guard for this fluid model, without new storage.

    Use useful average native bytes, never padded operand capacity, when
    deriving lead depth. This is not a per-request tail-window proof.
    One common GU/Down cap is chosen conservatively across a design's cores
    and all captured projection dimensions, independently of measured times.
    """
    cores = budget['cores']
    lead_bytes = settings.fabric.landing_credit_bandwidth_upper_bound * (
        settings.fabric.hbm_latency_cycles + 1)
    dimensions = {e['H'] for w in workloads for e in w['experts']}
    down_dimensions = {e['F'] for w in workloads for e in w['experts']}
    gu_limits, down_limits, records = [], [], []
    for c in cores:
        average = lambda k: c['pn'] * k * 2 / math.ceil(k / c['pk'])
        gu_bytes = min(average(k) for k in dimensions)
        down_bytes = min(average(k) for k in down_dimensions)
        gu_spare = math.ceil(lead_bytes / gu_bytes)
        down_spare = math.ceil(lead_bytes / down_bytes)
        gu = max(1, min(settings.gu_group_limit, (c['w_slots'] - gu_spare) // 2))
        down = max(1, min(settings.down_group_limit, c['w_slots'] - down_spare))
        gu_limits.append(gu)
        down_limits.append(down)
        records.append({'pm': c['pm'], 'pn': c['pn'], 'pk': c['pk'],
                        'physical_W_slots': c['w_slots'], 'lead_bytes': lead_bytes,
                        'average_GU_useful_W_bytes': gu_bytes,
                        'average_Down_useful_W_bytes': down_bytes,
                        'required_GU_spare_slots': gu_spare,
                        'required_Down_spare_slots': down_spare,
                        'GU_limit': gu, 'Down_limit': down})
    return min(gu_limits), min(down_limits), records


def collect(source, inputs, out):
    source_methods = json.loads((source / 'METHODS.json').read_text())
    assert source_methods['precision'].startswith('BF16')
    for name, digest in source_methods['input_sha256'].items():
        assert hashlib.sha256((inputs / name).read_bytes()).hexdigest() == digest, name
    workloads = [w for w in load_inputs(inputs)['heldout'] if w['batch'] in BATCHES]
    assert len(workloads) == 108
    wm = {w['id']: w for w in workloads}
    (out / 'raw').mkdir()
    (out / 'source_data').mkdir()
    (out / 'inputs').mkdir()
    # Include the actual inputs so the exported package can be rerun.
    for name in source_methods['input_sha256']:
        shutil.copy2(inputs / name, out / 'inputs' / name)
    for p in source.iterdir():
        if p.is_file():
            shutil.copy2(p, out / 'source_data' / p.name)
    all_rows, service_rows, core_configs, quota_rows = [], [], [], []
    controls, groups, guard_records, receipts = [], [], [], []
    for organization in DESIGNS:
        cores, settings = make_settings(organization)
        original = json.loads((source / (organization + '_charged.json')).read_text())
        charged = evaluate(workloads, cores, settings, detail=True)
        assert canonical(charged) == canonical(original), 'source result replay mismatch'
        owners = [tuple(b['core'] for b in r['bindings']) for r in charged]
        dump(out / 'raw' / (organization + '_charged.json'), charged)
        receipts.append({'organization': organization, 'case': 'charged', 'windows': 108,
            'complete_repeats': 2, 'exact_original_baseline_reproduction': True})
        for c, memory in enumerate(charged[0]['budget']['cores']):
            core_configs.append({'organization': organization, 'core': c,
                'PM': memory['pm'], 'PN': memory['pn'], 'PK': memory['pk'],
                'multipliers': memory['pm'] * memory['pn'] * memory['pk'],
                'W_KiB': memory['w_capacity_bytes'] / 1024, 'W_slots': memory['w_slots'],
                'X_KiB': memory['x_register_bytes'] / 1024, 'X_buffers': memory['x_buffers'],
                'accumulator_KiB': memory['accumulator_bytes'] / 1024,
                'Z_KiB': memory['z_bytes'] / 1024,
                'W_banks': memory['w_banks'], 'X_banks': memory['x_banks'],
                'accumulator_banks': memory['accumulator_banks']})
        quota_rows += [{'organization': organization, 'structure': name, 'bytes': size,
                        'KiB': size / 1024} for name, size in charged[0]['budget']['structures'].items()]
        alt_results = {}
        settings_cases = {'ideal_HBM': replace(settings, hbm=False),
                          'ideal_ports': replace(settings, ports=False),
                          'zero_control': replace(settings, control=False),
                          'compute_only': replace(settings, hbm=False, ports=False, control=False)}
        for name, alternative in settings_cases.items():
            got = evaluate(workloads, cores, alternative, owners=owners, detail=True)
            for a, b in zip(charged, got):
                verify_oracle(a, b)
            if name == 'compute_only':
                old = json.loads((source / (organization + '_compute_oracle.json')).read_text())
                assert canonical(got) == canonical(old)
            alt_results[name] = got
            dump(out / 'raw' / (organization + '_' + name + '.json'), got)
            receipts.append({'organization': organization, 'case': name, 'windows': 108,
                'complete_repeats': 2, 'timing_only_work_owners_traffic_budget_verified': True})
        gu, down, guard = lookahead_limits(settings, charged[0]['budget'], workloads)
        guard_records.extend({'organization': organization, **r} for r in guard)
        compiler_cases = {'GU_only': (1, 4), 'Down_only': (2, 2),
                          'both_halved': (1, 2), 'lookahead_guard': (gu, down)}
        for name, (gu_cap, down_cap) in compiler_cases.items():
            alternative = replace(settings, gu_group_limit=gu_cap, down_group_limit=down_cap)
            got = evaluate(workloads, cores, alternative, owners=owners, detail=True)
            dump(out / 'raw' / (organization + '_' + name + '.json'), got)
            for a, b in zip(charged, got):
                for key in ('useful_macs', 'issued_macs', 'hbm_bytes', 'native_unique_bytes', 'budget'):
                    assert canonical(a[key]) == canonical(b[key]), key
                assert [x['core'] for x in a['bindings']] == [x['core'] for x in b['bindings']]
                assert all(p['peak_w_bytes'] <= b['budget']['cores'][p['core']]['w_capacity_bytes']
                           and p['peak_x_bytes'] <= b['budget']['cores'][p['core']]['x_register_bytes']
                           and p['peak_accumulator_bytes'] <= b['budget']['cores'][p['core']]['accumulator_bytes']
                           for p in b['phases'])
                groups.append({'organization': organization, 'case': name,
                    'workload': a['workload'], 'batch': a['batch'],
                    'GU_group_limit': gu_cap, 'Down_group_limit': down_cap,
                    'baseline_ms': a['latency_ms'], 'alternative_ms': b['latency_ms'],
                    'time_reduction_percent': 100 * (1 - b['cycles'] / a['cycles']),
                    'HBM_bytes': b['hbm_bytes'], 'HBM_bytes_unchanged': True,
                    'baseline_X_sram_bytes': a['X_sram_bytes'],
                    'alternative_X_sram_bytes': b['X_sram_bytes'],
                    'baseline_compute_service_ms': bounds(a, settings)['compute_fixed_owner_lb_ms'],
                    'alternative_compute_service_ms': bounds(b, alternative)['compute_fixed_owner_lb_ms'],
                    'scope': 'compiler mapping intervention,not timing-only oracle;same charged owners'})
            receipts.append({'organization': organization, 'case': name, 'windows': 108,
                'complete_repeats': 2, 'HBM_MAC_budget_owners_unchanged': True,
                'activation_reuse_compute_order_and_control_service_may_change': True})
        for i, r in enumerate(charged):
            w = wm[r['workload']]
            provenance = w.get('provenance', {})
            dataset = provenance.get('dataset', '') if isinstance(provenance, dict) else ''
            routed = [e for e in w['experts'] if not e['is_shared']]
            b = bounds(r, settings)
            oracle = alt_results['compute_only'][i]
            core_services = core_rows(r, settings)
            row = {'organization': organization, 'workload': r['workload'],
                'dataset': dataset, 'batch': r['batch'],
                'windows': 1, 'routed_hot_Me_gt2': sum(e['Me'] > 2 for e in routed),
                'routed_cold_Me_1_or2': sum(e['Me'] <= 2 for e in routed),
                'shared_experts': sum(e['is_shared'] for e in w['experts']),
                'total_ms': r['latency_ms'], 'HBM_lower_bound_ms': b['hbm_lb_ms'],
                'compute_service_ms': b['compute_fixed_owner_lb_ms'],
                'compute_oracle_ms': oracle['latency_ms'],
                'HBM_MB': r['hbm_bytes'] / 1e6, 'HBM_bytes': r['hbm_bytes'],
                'unique_HBM_bytes': r['native_unique_bytes'],
                'useful_MAC': r['useful_macs'], 'issued_MAC': r['issued_macs'],
                'X_sram_bytes': r['X_sram_bytes'],
                'max_core_W_port_ms': max(s['W_port_service_ms'] for s in core_services),
                'max_core_X_port_ms': max(s['X_port_service_ms'] for s in core_services),
                'max_core_accumulator_ms': max(s['acc_port_service_ms'] for s in core_services),
                'core_finish_gap_ms': (max(r['core_finish_cycles']) - min(r['core_finish_cycles'])) / 1e6,
                'spatial_utilization_percent': 100 * r['useful_macs'] / r['issued_macs'],
                'effective_fetch_GBps_wall': r['hbm_bytes'] / r['cycles'],
                'effective_GFLOPs_wall': 2 * r['useful_macs'] / r['cycles'],
                'GFLOPs_compute_oracle': 2 * r['useful_macs'] / oracle['cycles'],
                'actual_native_fetch_span_ms': None, 'actual_native_HBM_busy_ms': None}
            # Capture dataset is also encoded in the immutable workload ID.
            if not row['dataset']:
                row['dataset'] = next(d for d in ('bfcl', 'gpqa', 'swe') if d in r['workload'])
            all_rows.append(row)
            service_rows.extend({'organization': organization, **s} for s in core_services)
            controls.append({'organization': organization, 'workload': r['workload'], 'batch': r['batch'],
                'baseline_ms': r['latency_ms'],
                **{name + '_ms': result[i]['latency_ms'] for name, result in alt_results.items()}})
    return workloads, all_rows, service_rows, core_configs, quota_rows, controls, groups, guard_records, receipts


def aggregate(rows, by_dataset=False):
    result = []
    for batch in BATCHES:
        for organization in DESIGNS:
            selectors = ('bfcl', 'gpqa', 'swe') if by_dataset else ('all',)
            for dataset in selectors:
                rs = [r for r in rows if r['batch'] == batch and r['organization'] == organization
                      and (dataset == 'all' or r['dataset'] == dataset)]
                assert rs
                row = {'batch': batch, 'organization': organization, 'dataset': dataset, 'windows': len(rs)}
                for key in ('routed_hot_Me_gt2', 'routed_cold_Me_1_or2', 'shared_experts', 'total_ms',
                            'HBM_lower_bound_ms', 'compute_service_ms', 'compute_oracle_ms', 'HBM_MB',
                            'max_core_W_port_ms', 'max_core_X_port_ms', 'max_core_accumulator_ms', 'core_finish_gap_ms'):
                    row[key] = mean(r[key] for r in rs)
                row['p50_total_ms'] = float(np.quantile([r['total_ms'] for r in rs], .5))
                row['p95_total_ms'] = float(np.quantile([r['total_ms'] for r in rs], .95))
                wall = sum(r['total_ms'] for r in rs) * 1e6
                oracle = sum(r['compute_oracle_ms'] for r in rs) * 1e6
                row['effective_fetch_GBps_wall'] = sum(r['HBM_bytes'] for r in rs) / wall
                row['effective_GFLOPs_wall'] = 2 * sum(r['useful_MAC'] for r in rs) / wall
                row['GFLOPs_compute_oracle'] = 2 * sum(r['useful_MAC'] for r in rs) / oracle
                row['spatial_utilization_percent'] = 100 * sum(r['useful_MAC'] for r in rs) / sum(r['issued_MAC'] for r in rs)
                row['over_HBM_lower_bound_percent'] = 100 * (sum(r['total_ms'] for r in rs) /
                                                           sum(r['HBM_lower_bound_ms'] for r in rs) - 1)
                result.append(row)
    return result


def summarize_controls(rows, groups=False):
    result = []
    for batch in BATCHES:
        for organization in DESIGNS:
            cases = GROUP_CASES if groups else ('oracle_controls',)
            for case in cases:
                rs = [r for r in rows if r['batch'] == batch and r['organization'] == organization
                      and (not groups or r['case'] == case)]
                row = {'batch': batch, 'organization': organization, 'case': case,
                       'windows': len(rs), 'baseline_ms': mean(r['baseline_ms'] for r in rs)}
                if groups:
                    row.update(GU_group_limit=rs[0]['GU_group_limit'], Down_group_limit=rs[0]['Down_group_limit'],
                        alternative_ms=mean(r['alternative_ms'] for r in rs),
                        HBM_bytes_unchanged=True,
                        mean_X_bytes_change=mean(r['alternative_X_sram_bytes'] - r['baseline_X_sram_bytes'] for r in rs),
                        baseline_compute_service_ms=mean(r['baseline_compute_service_ms'] for r in rs),
                        alternative_compute_service_ms=mean(r['alternative_compute_service_ms'] for r in rs))
                    row['time_reduction_percent'] = 100 * (1 - row['alternative_ms'] / row['baseline_ms'])
                else:
                    for name in ORACLES:
                        row[name + '_ms'] = mean(r[name + '_ms'] for r in rs)
                        row[name + '_reduction_percent'] = 100 * (1 - row[name + '_ms'] / row['baseline_ms'])
                result.append(row)
    return result


LABELS = {'batch': 'Batch', 'organization': '组织', 'dataset': '数据集', 'windows': '窗口数',
    'routed_hot_Me_gt2': '路由热专家均数\nMe>2', 'routed_cold_Me_1_or2': '路由冷专家均数\nMe=1或2',
    'shared_experts': 'Shared专家均数', 'total_ms': '模型总延迟\nms', 'HBM_lower_bound_ms': '取数下限\nms',
    'compute_service_ms': '固定归属计算服务\nms', 'compute_oracle_ms': '理想供数计算对照\nms',
    'HBM_MB': 'HBM权重读取\nMB', 'max_core_W_port_ms': '最大核W端口服务\nms',
    'max_core_X_port_ms': '最大核X端口服务\nms', 'max_core_accumulator_ms': '最大核累加服务\nms',
    'core_finish_gap_ms': '两核完成时间差\nms', 'effective_fetch_GBps_wall': '等效取数吞吐\nGB/s，总延迟口径',
    'effective_GFLOPs_wall': '有效计算吞吐\nGFLOP/s，总延迟口径',
    'GFLOPs_compute_oracle': '计算对照吞吐\nGFLOP/s，对照时间口径',
    'spatial_utilization_percent': '有效MAC/发射MAC\n%', 'over_HBM_lower_bound_percent': '高于取数下限\n%',
    'baseline_ms': '原配置模型延迟\nms', 'alternative_ms': '对照模型延迟\nms',
    'time_reduction_percent': '对照降低延迟\n%', 'actual_native_fetch_span_ms': '实际native取数跨度\n未测',
    'actual_native_HBM_busy_ms': '实际native HBM忙碌\n未测', 'p50_total_ms': '窗口延迟p50\nms',
    'p95_total_ms': '窗口延迟p95\nms'}


def workbook(path, summary, dataset, rows, cores, quotas, controls, groups, services, workload_rows, guard):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    tables = [('主结果', summary, '固定形状；解析FFN总延迟；分项服务和神谕不能相加'),
              ('输入与专家数', workload_rows, 'Me>0均活跃；冷专家Me=1或2；Shared独立统计；均数不是单窗口整数'),
              ('硬件配置', cores, '尺寸顺序M×N×K；相同MAC/SRAM不等于相同面积；总2,158,592 B'),
              ('存储账本', quotas, '共享与私有分开；容量和bank带宽分别收费'),
              ('计时神谕', controls, '只改变计时；锁定原归属、工作量、字节和存储；节省不可相加'),
              ('驻留组对照', groups, '编译器改变映射顺序；三种组织同规则；HBM不变，X流量和计算反馈可能改变'),
              ('预取空间规则', guard, '按阶段平均native有效字节计算；native K尾窗口尚未校准'),
              ('按数据集', dataset, '保持相同窗口，不跨旧表或不同层取加速比'),
              ('逐窗口', rows, '324个charged窗口；输入与路由已就绪；不含Router/Attention/生成'),
              ('每核服务', services, '服务需求重叠；不是native停顿分解，不能相加成总延迟')]
    notes = [
        ('结果性质', '解析模型；未native/RTL校准；未验证完整FFN数值或bit-exact'),
        ('精度与时钟', '仅BF16输入/权重、FP32累加；假设1GHz，1周期=1ns'),
        ('输入', '捕获DeepSeek-V2-Lite BFCL/GPQA/SWE路由，B2/4/8/16各27窗口'),
        ('计时边界', 'post-router Gate/Up、SiLU/Z、Down与输出合并；不含Router/Attention/生成'),
        ('取数', '256个共享32B信用，64周期响应；模型连续上限126.030769 GB/s'),
        ('取数下限', 'HBM字节/模型上限；不是实际取数跨度或总线忙碌时间'),
        ('取数等效吞吐', '总HBM字节/总延迟；不是HBM活跃时带宽'),
        ('总延迟计算吞吐', '2×有效MAC/总延迟；一次乘加=2 FLOP'),
        ('计算对照吞吐', '2×有效MAC/理想供数计算对照时间；与前者分母不同'),
        ('控制神谕', '固定charged归属，单独关闭HBM/端口/控制计时；不可加总节省'),
        ('编译器对照', '只减小驻留N组，参数在所有batch冻结；同预算，不增加SRAM或带宽'),
        ('胜负范围', '固定6/3+3/4+2；未与形状优化单核比较，不宣称异构硬件胜出'),
        ('缺少的物理数据', '实际native取数跨度/HBM忙碌、逐地址bank冲突、功耗/面积：未测'),
        ('统计', '平均与p50/p95来自窗口分布，不是重复运行波动或模型误差'),
        ('复现', '看METHODS.json、README.md、source_data与raw；所有新点完整重复两次')]
    tables.insert(0, ('说明与字段', [{'项目': a, '说明': b} for a, b in notes], '先读此页，再读主结果'))
    for name, data, note in tables:
        ws = wb.create_sheet(name)
        keys = list(data[0])
        last = get_column_letter(len(keys))
        ws.merge_cells(f'A1:{last}1')
        ws['A1'] = 'MoE固定形状BF16对照 · ' + name
        ws['A1'].fill = PatternFill('solid', fgColor='18334D')
        ws['A1'].font = Font(name='Microsoft YaHei', size=15, bold=True, color='FFFFFF')
        ws.row_dimensions[1].height = 29
        ws.merge_cells(f'A2:{last}2')
        ws['A2'] = note
        ws['A2'].alignment = Alignment(wrap_text=True, vertical='center')
        ws['A2'].font = Font(name='Microsoft YaHei', size=10, color='334B60')
        ws.row_dimensions[2].height = 33
        for c, key in enumerate(keys, 1):
            cell = ws.cell(4, c, LABELS.get(key, key))
            cell.font = Font(name='Microsoft YaHei', bold=True, color='FFFFFF', size=10)
            cell.fill = PatternFill('solid', fgColor='34566E')
            cell.alignment = Alignment(wrap_text=True, vertical='center')
            width = 20 if key not in ('workload', 'scope', '说明') else (42 if key == 'workload' else 75)
            ws.column_dimensions[get_column_letter(c)].width = width
        ws.row_dimensions[4].height = 35
        for rn, record in enumerate(data, 5):
            for c, key in enumerate(keys, 1):
                value = record[key]
                if key == 'organization': value = DESIGNS[value][0]
                if value is None: value = '未测'
                cell = ws.cell(rn, c, value)
                cell.font = Font(name='Microsoft YaHei', size=10)
                cell.alignment = Alignment(vertical='center', wrap_text=isinstance(value, str))
                if rn % 2: cell.fill = PatternFill('solid', fgColor='EEF4F7')
                if isinstance(value, float): cell.number_format = '0.0000'
            ws.row_dimensions[rn].height = 25 if name != '说明与字段' else 32
        ws.freeze_panes = 'C5' if len(keys) > 2 else 'A5'
        ws.auto_filter.ref = f'A4:{last}{4 + len(data)}'
        ws.sheet_view.showGridLines = False
        ws.print_title_rows = '1:4'
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.page_setup.orientation = 'landscape'
        ws.page_setup.paperSize = ws.PAPERSIZE_A3
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
    wb.save(path)


def plot_figures(out, summary, controls, groups, workload_rows):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.backends.backend_pdf import PdfPages
    font = Path('/usr/share/fonts/google-droid/DroidSansFallback.ttf')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    find = lambda data, org, batch: next(r for r in data if r['organization'] == org and r['batch'] == batch)
    figures = []
    fig = plt.figure(figsize=(14.5, 10.0), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, .95))
    total, compute, input_ax, rate_ax = [fig.add_subplot(grid[r, c]) for r, c in ((0,0),(0,1),(1,0),(1,1))]
    for i, org in enumerate(DESIGNS):
        x = np.arange(4) + (i - 1) * .24
        for ax, key in ((total, 'total_ms'), (compute, 'compute_oracle_ms')):
            values = [find(summary, org, b)[key] for b in BATCHES]
            bars = ax.bar(x, values, width=.22, color=COLORS[i], label=DESIGNS[org][0])
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)
    lower = [find(summary, 'a_single6', b)['HBM_lower_bound_ms'] for b in BATCHES]
    total.plot(np.arange(4), lower, 'D--', color='#35434e', ms=5, lw=1.2, label='同字节取数下限')
    for ax in (total, compute):
        ax.set_xticks(range(4), ['B2', 'B4', 'B8', 'B16'])
        ax.set_xlabel('输入 Batch')
        ax.set_ylabel('时间（ms）')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    total.set_title('① 原配置总延迟：单核更快', loc='left', fontweight='bold')
    compute.set_title('② 理想供数计算对照：双核计算更快', loc='left', fontweight='bold')
    total.legend(fontsize=10, ncol=2, frameon=False, loc='upper left')
    input_ax.axis('off')
    input_ax.set_title('③ 输入与共同取数下限（每Batch 27窗口）', loc='left', fontweight='bold', pad=14)
    values = [[f"B{r['batch']}", f"{r['hot_mean']:.2f}", f"{r['cold_mean']:.2f}",
               '1', f"{r['HBM_MB']:.1f}", f"{r['HBM_lower_ms']:.4f}"] for r in workload_rows]
    table = input_ax.table(cellText=values, colLabels=['Batch','热专家\nMe>2','冷专家\nMe=1或2',
        'Shared','权重\nMB','取数下限\nms'], cellLoc='center', bbox=(0, .29, 1, .61))
    table.auto_set_font_size(False); table.set_fontsize(11)
    style_table(table)
    input_ax.text(0, .18, '专家数是窗口平均值；冷专家仍然活跃。\n两核共用HBM；分核不减少这些权重字节。',
                  transform=input_ax.transAxes, fontsize=11, va='top', color='#445769')
    rate_ax.axis('off')
    rate_ax.set_title('④ B8吞吐：必须区分两种时间分母', loc='left', fontweight='bold', pad=14)
    values = [[DESIGNS[org][0], f"{find(summary,org,8)['effective_fetch_GBps_wall']:.2f}",
               f"{find(summary,org,8)['GFLOPs_compute_oracle']:.0f}",
               f"{find(summary,org,8)['effective_GFLOPs_wall']:.2f}"] for org in DESIGNS]
    table = rate_ax.table(cellText=values, colLabels=['组织','取数等效\nGB/s（总延迟）',
        '计算对照\nGFLOP/s','端到端\nGFLOP/s'], cellLoc='center', bbox=(0,.29,1,.61))
    table.auto_set_font_size(False); table.set_fontsize(11)
    style_table(table)
    rate_ax.text(0, .18, '同字节和有效MAC时，两个总延迟口径的\n吞吐排序都由总延迟的倒数决定。', transform=rate_ax.transAxes,
                 fontsize=11, va='top', color='#445769')
    fig.suptitle('MoE固定形状对照：计算收益能否转化为总延迟收益？\n'
        'BF16｜a=6×4×512；b=3×4×512两核；c=4×4×512+2×4×512', fontsize=18, fontweight='bold')
    fig.supxlabel('解析 post-Router FFN｜假设1 GHz｜HBM上限126.03 GB/s｜未native校准｜计算对照与取数下限不能相加', fontsize=11)
    figures.append(('overview', fig))
    fig = plt.figure(figsize=(16, 9), layout='constrained')
    ax = fig.add_subplot()
    ax.axis('off')
    values = []
    for batch in BATCHES:
        for org in DESIGNS:
            r = find(summary, org, batch)
            values.append([f'B{batch}', DESIGNS[org][0], f"{r['total_ms']:.4f}",
                f"{r['HBM_lower_bound_ms']:.4f}", f"{r['compute_oracle_ms']:.4f}",
                f"{r['effective_fetch_GBps_wall']:.2f}", f"{r['effective_GFLOPs_wall']:.2f}",
                f"{r['GFLOPs_compute_oracle']:.1f}", f"{r['spatial_utilization_percent']:.1f}"])
    table = ax.table(cellText=values, colLabels=['Batch', '组织', '原配置\n总延迟 ms',
        'HBM取数\n下限 ms', '计算对照\n时间 ms', '取数等效\nGB/s', '总延迟口径\nGFLOP/s',
        '计算对照口径\nGFLOP/s', '发射空间\n利用率 %'], cellLoc='center',
        colWidths=[.065,.12,.115,.115,.115,.11,.12,.13,.11], bbox=(0,.19,1,.72))
    table.auto_set_font_size(False); table.set_fontsize(12)
    style_table(table)
    ax.text(0,.12,'108个捕获路由窗口；每个Batch 27个，时间列为窗口均值。精度仅BF16输入/权重、FP32累加。\n'
        '取数下限、计算对照不是总延迟的两个可加分项；实际native取数跨度与HBM忙碌时间尚未测量。\n'
        '发射空间利用率只统计已发射的MAC槽，不包含阵列空等周期；计算对照并非实际运行中的计算占比。',
        transform=ax.transAxes, va='top', fontsize=12, color='#445769')
    fig.suptitle('MoE固定形状BF16：将预期排序改为实际模型结果\n'
        'a 单核6×4×512｜b 同构3×4×512两核｜c 异构4×4×512+2×4×512', fontsize=19)
    fig.supxlabel('共同预算：12,288乘法器 · 2,158,592 B存储 · 126.03 GB/s模型HBM上限 · 假设1 GHz · 未native校准',fontsize=12)
    figures.append(('comparison_table',fig))
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), layout='constrained')
    labels = ('原配置','理想HBM','理想片上端口','零控制')
    for i, org in enumerate(DESIGNS):
        r = find(controls, org, 8)
        values = [r['baseline_ms'], r['ideal_HBM_ms'], r['ideal_ports_ms'], r['zero_control_ms']]
        bars = axes[0].bar(np.arange(4)+(i-1)*.24, values, .22, color=COLORS[i], label=DESIGNS[org][0])
        axes[0].bar_label(bars, fmt='%.2f', fontsize=10, padding=3)
        for name, style in (('baseline_ms','--'),('alternative_ms','-')):
            values = [next(r for r in groups if r['organization']==org and r['batch']==b and r['case']=='lookahead_guard')[name]
                      for b in BATCHES]
            axes[1].plot(range(4), values, style, marker='o', color=COLORS[i], lw=2,
                         label=DESIGNS[org][0] + ('原配置' if name=='baseline_ms' else '保留预取空间'))
    axes[0].set_xticks(range(4), labels); axes[0].set_title('B8单项计时神谕（固定原归属）', loc='left')
    axes[0].legend(frameon=False, fontsize=10)
    axes[1].set_xticks(range(4), ['B2','B4','B8','B16'])
    axes[1].set_title('共同编译器规则：预留足够权重前瞻空间', loc='left')
    axes[1].legend(frameon=False, fontsize=9, ncol=2)
    for ax in axes:
        ax.set_ylabel('模型时间（ms）'); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1]*1.2)
    fig.suptitle('瓶颈诊断与驻留组对照：硬件、精度、带宽预算不变', fontsize=18, fontweight='bold')
    fig.supxlabel('左：非物理计时神谕，节省不可相加。右：映射改变，X复用/计算反馈可能改变；不是异构专属优化。', fontsize=11)
    figures.append(('causal_controls', fig))
    with PdfPages(out / 'REPORT_FIGURES.pdf') as pdf:
        for name, fig in figures:
            fig.savefig(out / (name + '.png'), dpi=180, facecolor='white')
            svg = out / (name + '.svg')
            fig.savefig(svg, facecolor='white')
            svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
            pdf.savefig(fig, facecolor='white')
            plt.close(fig)


def style_table(table):
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        if row == 0:
            cell.set_facecolor('#284960'); cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#eef4f7' if row % 2 else '#f8fafb')
            cell.set_text_props(color='#263b4c')


def run(source, inputs, out):
    assert not out.exists() or not any(out.iterdir()), 'use a new empty output directory'
    out.mkdir(parents=True, exist_ok=True)
    ws, rows, services, cores, quotas, controls, groups, guard, checks = collect(source, inputs, out)
    summary, dataset = aggregate(rows), aggregate(rows, by_dataset=True)
    control_summary, group_summary = summarize_controls(controls), summarize_controls(groups, groups=True)
    workloads = []
    for batch in BATCHES:
        group = [w for w in ws if w['batch']==batch]
        routed = [[e for e in w['experts'] if not e['is_shared']] for w in group]
        ref = next(r for r in summary if r['batch']==batch)
        workloads.append({'batch':batch,'windows':len(group),
            'hot_mean':mean(sum(e['Me']>2 for e in es) for es in routed),
            'cold_mean':mean(sum(e['Me']<=2 for e in es) for es in routed), 'Shared_mean':1,
            'Me1_mean':mean(sum(e['Me']==1 for e in es) for es in routed),
            'Me2_mean':mean(sum(e['Me']==2 for e in es) for es in routed),
            'HBM_MB':ref['HBM_MB'],'HBM_lower_ms':ref['HBM_lower_bound_ms']})
    products={'SUMMARY':summary,'BY_DATASET':dataset,'WINDOWS':rows,'CORE_SERVICES':services,
              'CONFIG':cores,'SRAM_LEDGER':quotas,'WORKLOADS':workloads,'ORACLES':control_summary,
              'ORACLES_WINDOWS':controls,'GROUP_ABLATIONS':group_summary,
              'GROUP_ABLATIONS_WINDOWS':groups,'LOOKAHEAD_RULE':guard}
    for name, data in products.items(): csv_file(out / (name+'.csv'),data)
    workbook(out/'DATA.xlsx',summary,dataset,rows,cores,quotas,control_summary,group_summary,services,workloads,guard)
    plot_figures(out,summary,control_summary,group_summary,workloads)
    timing_modules = Path(__file__).parent
    code = ['regime/model.py','regime/resources.py','regime/metrics.py','regime/search.py','regime/campaign.py',
            'geometry3d/compute.py','geometry3d/memory.py','geometry3d/study.py','fixed_breakdown_report.py']
    (out/'source').mkdir()
    for rel in code:
        dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(timing_modules/rel,dest)
    methods={'complete':True,'repeats':2,'captured_windows':108,'windows_per_batch':27,
        'analytical_model_points':2916,'physical_native_calibration':False,'full_model_generation':False,
        'full_FFN_numerics_verified':False,'precision':'BF16/FP32 only','assumed_clock_GHz':1,
        'native_fetch_span_status':'not measured; null values explicitly retained',
        'statistics':'arithmetic means and empirical p50/p95 of captured windows; not calibrated model error',
        'comparison_domain':'fixed6/3+3/4+2 with whole-expert EFT; not optimized-single architecture victory',
        'new_defaults_installed':False,'controller_or_predictor_added':False,
        'source_sha256':{rel:hashlib.sha256((timing_modules/rel).read_bytes()).hexdigest() for rel in code},
        'python':platform.python_version(),'checks':checks,
        'guard_scope':'static compiler uses actual average native useful slice bytes and credit-latency product; not per-request K-tail proof',
        'group_interventions':'fixed charged owners/HBM/MAC/budget;X traffic/control service/compute order may change',
        'oracle_scope':'same charged owners/work/traffic/reservations;only declared timing changed'}
    dump(out/'METHODS.json',methods)
    readme(out,summary,group_summary)
    verify_exports(out,summary,rows,cores)
    dump(out/'MANIFEST.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(out.rglob('*')) if p.is_file() and p != out/'MANIFEST.json'})
    print(json.dumps({'out':str(out),'model_points':2916,'repeats':2,
                      'summary_rows':12,'group_cases':len(group_summary),'native':False},ensure_ascii=False))


def readme(out, summary, groups):
    lines=['# MoE固定形状BF16：图表与完整数据', '',
        '本次完整导出四个batch、三种固定组织的解析结果；未宣称native或完整模型测量。', '',
        '先看comparison_table.png、overview.png和DATA.xlsx的“说明与字段／主结果”。causal_controls.png分别展示计时神谕和映射对照，不能把神谕节省相加。', '',
        '| Batch | 单核6 ms | 同构3+3 ms | 异构4+2 ms | 共同HBM下限 ms |',
        '|---|---:|---:|---:|---:|']
    for b in BATCHES:
        rs=[next(r for r in summary if r['batch']==b and r['organization']==o) for o in DESIGNS]
        lines.append(f"| B{b} | {rs[0]['total_ms']:.4f} | {rs[1]['total_ms']:.4f} | {rs[2]['total_ms']:.4f} | {rs[0]['HBM_lower_bound_ms']:.4f} |")
    lines += ['', '## 如何读这些数据', '',
        '- 仅BF16输入/权重、FP32累加，尺寸按M×N×K；总乘法器12,288、总存储2,158,592 B。',
        '- 108个捕获post-router FFN窗口，B2/4/8/16各27个；Shared单列。Me=1/2仍是活跃冷专家，B2的Me>2必为0。',
        '- 取数下限=实际HBM字节/模型上限126.030769 GB/s；不是实际取数跨度或忙碌时间。',
        '- 取数等效GB/s和端到端GFLOP/s均除以总延迟；计算对照GFLOP/s除以非物理计算对照时间。一次MAC=2FLOP。',
        '- 计时神谕保持原归属、工作量、字节与容量；服务需求重叠，不能拼成堆叠的墙钟占比。',
        '- 驻留组对照改变遍历，三种组织同样适用；HBM/MAC/预算不变，X流量与反馈隐藏可能改变。',
        '- 双核每核5槽。GU原驻留4块留下1个预取槽；平均4KiB/65ns约63GB/s，低于全局126GB/s。Down有K尾，按有效字节计算需要3个前瞻槽。',
        '- 保留预取空间规则从槽数和模型带宽延迟积推导，单核保留原2/4组，双核改为1/2组，跨batch不换配置。该规则在native的K尾窗口上仍需校准。',
        '- 不取最大收益而单独偏袒异构；同构也会受益，不能据此宣称异构必要。此处基线是固定6×4×512，未与形状调优单核比较。',
        '- 2916个模型点各完整重复两次；完整数值、RTL、native排空、实际HBM跨度、面积能耗均未测。', '',
        '## 对照之后的总延迟（固定原归属）', '',
        '| Batch | 单核6 ms | 同构3+3 ms | 异构4+2 ms |', '|---|---:|---:|---:|']
    for b in BATCHES:
        rs=[next(r for r in groups if r['batch']==b and r['organization']==o and r['case']=='lookahead_guard') for o in DESIGNS]
        lines.append(f"| B{b} | {rs[0]['alternative_ms']:.4f} | {rs[1]['alternative_ms']:.4f} | {rs[2]['alternative_ms']:.4f} |")
    lines += ['', '## 文件与复现', '',
        'DATA.xlsx含说明、主结果、输入、配置、存储、计时神谕、驻留组、按数据集、逐窗口和每核服务。对应CSV保留英文稳定字段与明确单位，raw含每个原始模型结果，inputs/source_data/source保留输入、原始基线与关键源码。', '',
        '在匹配simulator研究分支、安装NumPy/matplotlib/openpyxl后运行：', '', '```sh',
        'python -m research.moe_dispatch.fixed_breakdown_report \\',
        '  --source /absolute/path/to/unpacked/source_data \\',
        '  --inputs /absolute/path/to/unpacked/inputs \\',
        '  --out /absolute/path/to/new-empty-output', '```', '',
        'METHODS.json与MANIFEST.json记录来源与所有文件哈希；脚本拒绝覆盖已有输出。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')


def verify_exports(out, summary, rows, cores):
    from openpyxl import load_workbook
    assert len(rows)==324 and len(summary)==12
    for b in BATCHES:
        rs=[r for r in summary if r['batch']==b]
        assert len(rs)==3 and all(r['windows']==27 for r in rs)
        assert len({r['HBM_MB'] for r in rs})==1
        for r in rs:
            assert r['HBM_lower_bound_ms']<=r['total_ms']
            assert math.isclose(r['effective_fetch_GBps_wall'],r['HBM_MB']/r['total_ms'],rel_tol=1e-12)
        if b==2: assert all(r['routed_hot_Me_gt2']==0 for r in rs)
    for o in DESIGNS:
        cs=[r for r in cores if r['organization']==o]
        assert sum(c['multipliers'] for c in cs)==12288
        for field,total in (('W_KiB',40),('X_KiB',12),('accumulator_KiB',96),('Z_KiB',384),
                            ('W_banks',64),('X_banks',24),('accumulator_banks',12)):
            assert sum(c[field] for c in cs)==total
    book=load_workbook(out/'DATA.xlsx',data_only=True,read_only=True)
    assert book['主结果'].max_row==16
    assert book['逐窗口'].max_row==328
    book.close()
    dump(out/'EXPORT_CHECKS.json',{'complete':True,'summary_rows':12,'window_rows':324,
        'input_counts_units_sram_and_banks_verified':True,'workbook_rows_verified':True,
        'native_or_bitexact_claimed':False})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.source,args.inputs,args.out)
