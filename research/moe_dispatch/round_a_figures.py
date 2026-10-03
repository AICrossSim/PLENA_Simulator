#!/usr/bin/env python3
"""Plot completed Round A results without selecting or changing a configuration.

These are scientific figures for a prospective ideal-operand compute/vector
model. Equal main MAC count does not imply equal physical area, SRAM ports or
HBM service. This script writes only to ROOT/figures and never runs a campaign.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

# The optional pure model import must not create files in its source tree.
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SELECTIONS = (
    'fixed_single_6', 'fixed_homogeneous_3_3', 'fixed_heterogeneous_4_2',
    'best_single_eft', 'best_homogeneous_eft', 'best_heterogeneous_eft',
)
LABELS = (
    'Fixed single 6', 'Fixed homogeneous 3+3', 'Fixed asymmetric 4+2',
    'Selected single', 'Selected homogeneous', 'Selected asymmetric',
)
COLORS = {'single': '#2864A8', 'homogeneous': '#DA8131', 'heterogeneous': '#237D6C'}
MARKERS = {'single': 's', 'homogeneous': '^', 'heterogeneous': 'o'}
NOTE = ('Ideal operands; post-Router compute + shared vector model. '
        'No HBM timing, SRAM-port timing, paid dispatch or PPA.')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def cores(geometry):
    result = [tuple(map(int, text.split('x'))) for text in geometry.split('+')]
    require(all(len(core) == 3 for core in result), 'Geometry must be MxNxK')
    require(sum(math.prod(core) for core in result) == 12288,
            'Figures require the fixed 12,288-main-MAC geometry budget')
    return result


def geometry_label(geometry, multiline=False):
    # Display wider token axis first, without changing canonical source IDs.
    ordered = sorted(cores(geometry), reverse=True)
    separator = '\n+ ' if multiline else ' + '
    return separator.join('×'.join(map(str, core)) for core in ordered)


def family(geometry):
    shapes = cores(geometry)
    return 'single' if len(shapes) == 1 else (
        'homogeneous' if shapes[0] == shapes[1] else 'heterogeneous')


def save(fig, dest, name, sources, scope, extra=None):
    paths = []
    for extension in ('png', 'svg', 'pdf'):
        path = dest / f'{name}.{extension}'
        metadata = {'Date': None} if extension == 'svg' else (
            {'CreationDate': None, 'ModDate': None} if extension == 'pdf' else {})
        fig.savefig(path, dpi=180, bbox_inches='tight', metadata=metadata)
        paths.append(dict(path=str(path.resolve()), bytes=path.stat().st_size,
                          sha256=sha(path)))
    plt.close(fig)
    return dict(name=name, status='generated', scope=scope,
                source_artifacts=sources, outputs=paths, **(extra or {}))


def matched_heldout(root, freeze):
    summaries = {row['selection']: row for row in read_json(root / 'heldout_summary.json')
                 if row['split'] == 'all_heldout'}
    layers = read_json(root / 'heldout_layers.json')
    common = None
    result = []
    for name in SELECTIONS:
        row = summaries[name]
        selection = freeze['selected_points'][name]
        require(row['geometry'] == selection['geometry'] and row['policy'] == selection['policy'] == 'eft',
                'Displayed hardware must be the frozen common-EFT selections')
        selected = [value for value in layers if value['selection'] == name]
        keys = {(value['split'], value['workload']) for value in selected}
        require(len(keys) == len(selected) == row['layers'] == 135,
                'A displayed selection does not have 135 unique matched layer windows')
        require(all(value['geometry'] == row['geometry'] and value['policy'] == 'eft'
                    for value in selected), 'Heldout rows do not match frozen geometry/policy')
        common = keys if common is None else common
        require(keys == common, 'Displayed configurations use different heldout windows')
        for field, layer_field in [('total_compute_layer_cycles', 'cycles'),
                                  ('useful_macs', 'useful_macs'),
                                  ('issued_macs', 'issued_macs')]:
            require(row[field] == sum(value[layer_field] for value in selected),
                    f'Published aggregate differs from per-layer results: {name}/{field}')
        require(row['spatial_utilization'] == row['useful_macs'] / row['issued_macs'],
                'Spatial utilization definition changed')
        require(row['total_compute_layer_ms_at_1ghz'] == row['total_compute_layer_cycles'] / 1e6,
                'Millisecond/cycle conversion changed')
        cores(row['geometry'])
        result.append(row)
    require(len({row['useful_macs'] for row in result}) == 1,
            'Compared configurations do not perform identical useful MAC work')
    return result, sorted(common)


def heldout_figure(root, dest, freeze, sources):
    rows, common = matched_heldout(root, freeze)
    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    x = [0, 1, 2, 3.5, 4.5, 5.5]
    heights = [row['total_compute_layer_ms_at_1ghz'] for row in rows]
    colors = [COLORS[family(row['geometry'])] for row in rows]
    bars = ax.bar(x, heights, color=colors, width=.74, edgecolor='white', linewidth=.8)
    for position, row, bar in zip(x, rows, bars):
        ax.text(position, bar.get_height() + max(heights)*.021,
                f"{bar.get_height():.3f} ms\nMAC spatial {row['spatial_utilization']:.1%}",
                ha='center', va='bottom', fontsize=9)
    ax.axvline(2.76, color='#AAAAAA', linestyle='--', linewidth=.9)
    ax.set_xticks(x, [label + '\n' + geometry_label(row['geometry'], multiline=True)
                     for label, row in zip(LABELS, rows)], fontsize=9)
    ax.set_ylim(0, max(heights)*1.22)
    ax.set_ylabel('Sum of 135 matched layer latencies (ms at 1 GHz)')
    ax.set_xlabel('Core shapes: M×N×K; 12,288 main MACs in every organization')
    ax.grid(axis='y', alpha=.23)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(.22, 1.03, 'Original shape comparison', transform=ax.transAxes,
            ha='center', fontsize=10, fontweight='bold')
    ax.text(.75, 1.03, 'Hardware selected on development traces', transform=ax.transAxes,
            ha='center', fontsize=10, fontweight='bold')
    fig.suptitle('Round A: fixed shapes versus trace-selected geometry',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.text(.5, -.015, NOTE + '\n135 correlated captured windows; these sums are not a full-model latency.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout()
    csv_write(dest / 'heldout_six_configuration_plot_data.csv', [
        {key: row[key] for key in ('selection', 'geometry', 'policy', 'layers',
          'total_compute_layer_cycles', 'total_compute_layer_ms_at_1ghz',
          'p95_layer_ms_at_1ghz', 'spatial_utilization', 'wall_mac_utilization',
          'useful_macs', 'issued_macs')} for row in rows])
    write_json(dest / 'matched_heldout_windows.json', common)
    return save(fig, dest, 'round_a_heldout_shape_comparison', sources,
                'Matched135 captured post-Router layer windows; ideal-operand compute/vector only',
                dict(configurations=6, layers_per_configuration=135,
                     policy='eft', hardware_selection='development only; frozen before heldout read',
                     main_MACs_per_configuration=12288, area_equivalence_claim=False,
                     HBM_E2E_claim=False))


def frontier_figure(root, dest, freeze, sources):
    rows = [row for row in read_json(root / 'development_search.json') if row['policy'] == 'eft']
    require(len(rows) == len({row['geometry'] for row in rows}) == 96,
            'The development EFT search must contain all 96 unique geometries')
    require(all(row['main_MAC_budget'] == 12288 and row['layers'] == 18 for row in rows),
            'Development plot does not share the declared MAC budget/trace population')
    front = [row for row in rows if not any(
        other['total_compute_layer_cycles'] <= row['total_compute_layer_cycles'] and
        other['spatial_utilization'] >= row['spatial_utilization'] and
        (other['total_compute_layer_cycles'] < row['total_compute_layer_cycles'] or
         other['spatial_utilization'] > row['spatial_utilization']) for other in rows)]
    published_front = csv_rows(root / 'latency_utilization_frontier.csv')
    require({row['geometry'] for row in front} == {row['geometry'] for row in published_front},
            'Nondominated set differs from the published fixed-MAC latency/utilization frontier')
    fig, ax = plt.subplots(figsize=(9, 6))
    for name in ('single', 'homogeneous', 'heterogeneous'):
        group = [row for row in rows if row['family'] == name]
        ax.scatter([100*row['spatial_utilization'] for row in group],
                   [row['total_compute_layer_ms_at_1ghz'] for row in group],
                   s=58 if name != 'heterogeneous' else 36,
                   color=COLORS[name], marker=MARKERS[name], alpha=.78,
                   edgecolors='white', linewidths=.55,
                   label=f'{name.capitalize()} ({len(group)} geometries)')
    line = sorted(front, key=lambda row: row['spatial_utilization'])
    ax.plot([100*row['spatial_utilization'] for row in line],
            [row['total_compute_layer_ms_at_1ghz'] for row in line],
            color='#363636', linestyle='--', linewidth=1.1,
            label='Latency / spatial-utilization nondominated set')
    # Highlight only configurations selected by the recorded development rule.
    offsets = {'single': (-100, 65), 'homogeneous': (-130, 35),
               'heterogeneous': (-155, 8)}
    for name in ('single', 'homogeneous', 'heterogeneous'):
        gid = freeze['selected_points'][f'best_{name}_eft']['geometry']
        row = next(value for value in rows if value['geometry'] == gid)
        point = (100*row['spatial_utilization'], row['total_compute_layer_ms_at_1ghz'])
        ax.scatter(*point, s=150, marker='*', color=COLORS[name], edgecolors='#222222',
                   linewidths=.7, zorder=4)
        ax.annotate(geometry_label(gid), point, xytext=offsets[name],
                    textcoords='offset points', fontsize=8, color=COLORS[name],
                    arrowprops=dict(arrowstyle='-', color=COLORS[name], linewidth=.6))
    ax.set_xlabel('MAC spatial utilization = useful / issued MAC operations (%)')
    ax.set_ylabel('Sum of 18 development layer latencies (ms, log scale)')
    ax.set_yscale('log')
    ax.grid(which='major', alpha=.23)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', fontsize=8, framealpha=.92)
    ax.set_title('Round A: geometry signal under a fixed MAC budget',
                 fontsize=13, fontweight='bold')
    fig.text(.5, -.03, '96 geometries; common EFT policy; 12,288 main MACs each. '
             'This is not an area/PPA Pareto frontier.\n' + NOTE,
             ha='center', fontsize=8.5, color='#555555')
    fig.tight_layout()
    csv_write(dest / 'development_eft_plot_data.csv', [{key: row[key] for key in (
        'geometry', 'family', 'policy', 'main_MAC_budget', 'layers',
        'total_compute_layer_cycles', 'total_compute_layer_ms_at_1ghz',
        'spatial_utilization', 'wall_mac_utilization')} for row in rows])
    return save(fig, dest, 'round_a_development_latency_utilization', sources,
                'All96 development-only geometry points under common EFT; fixed MAC, not fixed area',
                dict(points=96, layers_per_point=18, policy='eft',
                     frontier_geometry_ids=[row['geometry'] for row in front],
                     geometry_search_not_extended=True, area_or_energy_Pareto_claim=False))


def optional_crossover(root, dest, freeze, sources):
    pins = read_json(root / 'SOURCE_PINS.json')
    model_pin = pins['python_model']
    model_path = Path(model_pin['path'])
    require(sha(model_path) == model_pin['sha256'] == freeze['source_sha256'],
            'Optional expert curves must use the exact recorded Round A model')
    spec = importlib.util.spec_from_file_location('round_a_figure_exact_model', model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.Model(**freeze['scope']['model'])
    shape_set = sorted({shape for name in ('single', 'homogeneous', 'heterogeneous')
                        for shape in cores(freeze['selected_points'][f'best_{name}_eft']['geometry'])})
    dimensions = Counter()
    for pin in freeze['development_sources'].values():
        path = Path(pin['source_path'])
        require(sha(path) == pin['sha256'], 'Captured projection metadata changed')
        for workload in read_json(path)['workloads']:
            for expert in workload['experts']:
                dimensions[(expert['H'], expert['F'], bool(expert['is_shared']))] += 1
    rows, crossings = [], {}
    for h, f, is_shared in sorted(dimensions):
        previous, changed = None, []
        for m in range(1, 129):
            values = []
            for shape in shape_set:
                cost = module.expert_cost(m, h, f, module.Core(*shape), model)
                values.append((cost['private_estimate_cycles'], shape))
                rows.append(dict(Me=m, H=h, F=f, is_shared=is_shared,
                                 core_M_N_K='x'.join(map(str, shape)),
                                 private_estimate_cycles=cost['private_estimate_cycles'],
                                 scope='Synthetic Me sweep at captured H/F; isolated private-cost estimate; no queue contention'))
            fastest = min(values)[1]
            if previous is not None and fastest != previous:
                changed.append(m)
            previous = fastest
        crossings[f'H{h}_F{f}_shared{int(is_shared)}'] = changed
    csv_write(dest / 'selected_shape_isolated_expert_Me_sweep.csv', rows)
    if not any(crossings.values()):
        return dict(name='round_a_selected_shape_crossover', status='not_generated',
                    reason='No fastest isolated selected-core service crossover in Me1..128 at the captured H/F dimensions; no threshold is inferred.',
                    scope='Optional synthetic sensitivity, not extra captured workload observations',
                    H_F_populations=[dict(H=h, F=f, is_shared=s, observed_expert_descriptors=count)
                                     for (h, f, s), count in sorted(dimensions.items())],
                    crossings=crossings, exact_model_source=model_pin,
                    retained_sweep_csv=dict(path=str((dest/'selected_shape_isolated_expert_Me_sweep.csv').resolve()),
                                           sha256=sha(dest/'selected_shape_isolated_expert_Me_sweep.csv')))
    fig, axes = plt.subplots(1, len(dimensions), figsize=(6*len(dimensions), 4.7), squeeze=False)
    for ax, (h, f, is_shared) in zip(axes[0], sorted(dimensions)):
        for shape in shape_set:
            values = [row for row in rows if row['H'] == h and row['F'] == f and
                      row['is_shared'] == is_shared and row['core_M_N_K'] == 'x'.join(map(str, shape))]
            ax.plot([row['Me'] for row in values], [row['private_estimate_cycles'] for row in values],
                    label='×'.join(map(str, shape)))
        ax.set_title(f'H={h}, F={f}; ' + ('Shared' if is_shared else 'Routed'))
        ax.set_xlabel('Synthetic tokens per expert Me')
        ax.set_ylabel('Isolated private-cost estimate (cycles)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Selected-core service sensitivity; no queuing or HBM', fontsize=13)
    fig.tight_layout()
    return save(fig, dest, 'round_a_selected_shape_crossover', sources,
                'Synthetic Me1..128 at captured H/F; recorded pure Round A expert_cost function',
                dict(crossings=crossings, threshold_selection_performed=False,
                     captured_extra_observations_claim=False))


def run(root):
    root = Path(root).resolve()
    require(root.name == 'plena-round-a-dse-20261003-v2',
            'This implementation is pinned to the independently validated Round A v2 directory')
    files = ('CONCLUSIONS.json', 'FROZEN_ROUND_A.json', 'SOURCE_PINS.json',
             'VALIDATION_RECEIPT.json', 'input_provenance.json', 'heldout_summary.json',
             'heldout_layers.json', 'development_search.json', 'latency_utilization_frontier.csv')
    source_hashes = {name: sha(root/name) for name in files}
    source_stat = {name: (root/name).stat().st_mtime_ns for name in files}
    conclusions = read_json(root/'CONCLUSIONS.json')
    freeze = read_json(root/'FROZEN_ROUND_A.json')
    validation = read_json(root/'VALIDATION_RECEIPT.json')
    require(conclusions['complete'] is True and validation['complete'] is True and
            validation['errors'] == validation['failures'] == 0 and
            conclusions['repeat_full_json_exact'] is True and
            conclusions['freeze_sha256'] == source_hashes['FROZEN_ROUND_A.json'] and
            freeze['source_pins_sha256'] == source_hashes['SOURCE_PINS.json'],
            'Round A completed validation/freeze provenance is missing')
    for pin in validation['current_artifacts']:
        require(sha(pin['path']) == pin['sha256'], 'Validated campaign artifact changed')
    require(freeze['scope']['main_MAC_budget'] == 12288 and
            freeze['heldout_opened_in_this_run_before_freeze'] is False,
            'Geometry budget or development-before-heldout ordering changed')
    dest = root/'figures'
    dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10, 'axes.titlesize': 13,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    sources = [{'path': str((root/name).resolve()), 'sha256': value}
               for name, value in source_hashes.items()]
    figures = [heldout_figure(root, dest, freeze, sources),
               frontier_figure(root, dest, freeze, sources),
               optional_crossover(root, dest, freeze, sources)]
    require(all(sha(root/name) == value and (root/name).stat().st_mtime_ns == source_stat[name]
                for name, value in source_hashes.items()), 'Plotting modified the campaign')
    artifacts = [{'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(dest.iterdir()) if path.is_file() and path.name != 'figure_receipt.json']
    receipt = dict(schema='plena_round_a_v2_scientific_figures_v1', status='complete',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        scope='Ideal compute/vector diagnostic geometry screening; no HBM/PPA/area Pareto claim',
        script_path=str(Path(__file__).resolve()), script_sha256=sha(__file__),
        reproducible_cli=[sys.executable, str(Path(__file__).resolve()), '--root', str(root)],
        simulator_campaigns_launched=0, new_hardware_selection_performed=False,
        geometry_policy_or_result_changes=False, frozen_v3_outputs_touched=False,
        source_artifacts=sources, original_results_hash_and_mtime_unchanged=True,
        figures=figures, artifacts=artifacts,
        matplotlib_version=matplotlib.__version__)
    write_json(dest/'figure_receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2))
