#!/usr/bin/env python3
"""Calibrated full-shape DMA ablation; all policies also run on single cores.

Reuses immutable original numerical fixtures. The optional layout is a byte
permutation with unchanged quantization, explicitly checked before execution.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import time
from compare_moe_normal import digest, read_json, require, run_comparison


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def scale_layout(source, target):
    original = read_json(source/'workload.json')
    manifest = copy.deepcopy(original)
    image_path = source/original['hbm_file']
    source_hash = digest(image_path)
    require(source_hash == original['metadata']['hbm_sha256'], 'source image identity differs')
    image = image_path.read_bytes()
    packed = bytearray()
    for old, new in zip(original['experts'], manifest['experts']):
        for name in ['gate', 'up', 'down']:
            a, b = old[name], new[name]
            for stream, width in [('element', a['cols']), ('scale', (a['cols']+7)//8)]:
                packed.extend(b'\0' * (-len(packed) % 64))
                base = len(packed)
                # Element storage remains unchanged in width/stride. Only scales
                # get the next odd number of 32B sectors to rotate row starts.
                stride = a[stream+'_row_stride']
                if stream == 'scale':
                    sectors = (width + 31)//32
                    stride = (sectors if sectors % 2 else sectors + 1)*32
                b[stream+'_base'] = base
                b[stream+'_row_stride'] = stride
                for row in range(a['rows']):
                    start = a[stream+'_base'] + row*a[stream+'_row_stride']
                    values = image[start:start+width]
                    require(len(values) == width, 'truncated source row')
                    packed.extend(values)
                    packed.extend(b'\0' * (stride-width))
                    require(packed[base+row*stride:base+row*stride+width] == values, 'layout changed row values')
    packed.extend(b'\0' * (-len(packed) % 64))
    target.mkdir(parents=True, exist_ok=True)
    out = target/manifest['hbm_file']
    out.write_bytes(packed)
    manifest['metadata']['hbm_sha256'] = digest(out)
    manifest['metadata']['dma_layout_transform'] = dict(source_manifest_sha256=digest(source/'workload.json'),
        source_image_sha256=source_hash, kind='scale_stride_odd_32B_sectors',
        original_bytes=len(image), padded_bytes=len(packed), all_logical_row_bytes_equal=True)
    save(target/'workload.json', manifest)
    golden = read_json(source/'golden.json')
    require(golden['workload_sha256'] == digest(source/'workload.json') and golden['hbm_sha256'] == source_hash,
            'source golden identity differs')
    golden['workload_sha256'] = digest(target/'workload.json')
    golden['hbm_sha256'] = digest(out)
    save(target/'golden.json', golden)


def prepare(root, source):
    base = read_json(source/'campaign.json')
    names = [n for n in base['architectures'] if n != 'heterogeneous_fixed_threshold']
    stages = ['reference', 'port_only', 'sector', 'coalesce', 'credits128', 'slots3', 'candidate']
    for stage_index, stage in enumerate(stages):
        for name in names:
            a = read_json(source/'architectures'/(name+'.json'))
            a['dma'] = dict(issue_policy='global_fifo' if stage_index == 0 else 'per_channel',
                sector_reads=stage_index >= 2, coalesce=stage_index >= 3, frontend_sram_bytes=45056)
            a['global_dma_credits'] = 64 if stage_index < 4 else 128
            a['global_dma_staging_bytes'] = a['global_dma_credits'] * 64
            slots = 2 if stage_index < 5 else (3 if stage_index == 5 else 4)
            for c in a['cores']:
                c['read_cache_bytes'] = 0
                c['weight_slots'] = slots
            # Same 64KiB total matrix SRAM; only the asymmetric candidate
            # needs redistribution to accommodate four large-core slots.
            if len(a['cores']) == 2 and a['cores'][0]['mlen'] == 192 and slots == 4:
                a['cores'][0]['weight_sram_bytes'] = 49152
                a['cores'][1]['weight_sram_bytes'] = 16384
            save(root/'architectures'/stage/(name+'.json'), a)
    for fixture in base['fixtures']:
        scale_layout(source/fixture, root/'scale_layout'/fixture)
    tasks = []
    for stage in ['reference', 'candidate']:
        tasks.extend(dict(name=fixture+'_'+stage, fixture=str(source/fixture), stage=stage) for fixture in base['fixtures'])
    for stage in stages[1:-1]:
        tasks.append(dict(name='deepseek_full_decode_b32_'+stage, fixture=str(source/'deepseek_full_decode_b32'), stage=stage))
    for fixture in base['fixtures']:
        tasks.append(dict(name=fixture+'_scale_layout', fixture=str(root/'scale_layout'/fixture), stage='candidate'))
    plan = dict(schema_version=1, tasks=tasks, architectures=names, repeats=2, hbm_channels=8,
        numerical_tolerance=base['numerical_tolerance'], source_campaign_sha256=digest(source/'campaign.json'),
        scope=base['scope'], configured_dma_total_sram_bytes=45056,
        note='staging is a subset of frontend SRAM; equal MAC counts are not equal area or calibrated RTL throughput')
    save(root/'campaign.json', plan)
    return plan


def campaign(root, source, binary, workers):
    plan = prepare(root, source)
    binary_hash = digest(binary)
    save(root/'execution_plan.json', dict(campaign=plan, executable_sha256=binary_hash,
        campaign_driver_sha256=digest(__file__), comparison_driver_sha256=digest(Path(__file__).with_name('compare_moe_normal.py')),
        architecture_files={str(p.relative_to(root)):digest(p) for p in sorted((root/'architectures').glob('*/*.json'))}))
    outcomes = {}
    def execute(task):
        fixture = Path(task['fixture'])
        paths = [root/'architectures'/task['stage']/(n+'.json') for n in plan['architectures']]
        print('Starting '+task['name'], flush=True)
        start = time.monotonic()
        try:
            comparison = run_comparison(binary, fixture/'workload.json', fixture/'golden.json', paths,
                root/'comparisons'/task['name'], repeats=2, hbm_channels=8, timeout=1200, workers=4,
                **plan['numerical_tolerance'])
            return dict(status='passed', wall_seconds=time.monotonic()-start,
                timings_ps={r['architecture']['name']:r['result']['total_ps'] for r in comparison['comparisons']})
        except Exception as error:
            return dict(status='failed', error=str(error), wall_seconds=time.monotonic()-start)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, task):task['name'] for task in plan['tasks']}
        for future in as_completed(futures):
            name = futures[future]
            outcomes[name] = future.result()
            print(name+': '+outcomes[name]['status'], flush=True)
            save(root/'campaign_status.json', dict(status='running', all_gates_passed=False, outcomes=outcomes))
    passed = len(outcomes) == len(plan['tasks']) and all(o['status'] == 'passed' for o in outcomes.values())
    require(digest(binary) == binary_hash, 'binary changed during campaign')
    save(root/'campaign_status.json', dict(status='passed' if passed else 'failed', all_gates_passed=passed, outcomes=outcomes))
    require(passed, 'incomplete/failed campaign; no validated benefit conclusion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=[1,2])
    args = parser.parse_args()
    campaign(args.root.resolve(), args.source.resolve(), args.binary.resolve(), args.workers)
