#!/usr/bin/env python3
"""Reproduce frozen PLENA controls without writing to their source directories."""

import argparse
import concurrent.futures
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/scratch/shared/mcl123/plena')
SOURCE = Path(__file__).resolve().parents[2]
OLD = ROOT / 'outputs/moe_output_pool_20260909'
DIAG = ROOT / 'outputs/moe_bottleneck_20260911'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def differences(old, new, prefix=''):
    if isinstance(old, dict):
        if not isinstance(new, dict):
            return [prefix]
        return [p for k, v in old.items()
                for p in differences(v, new.get(k), prefix + '/' + k)]
    if isinstance(old, list):
        if not isinstance(new, list) or len(old) != len(new):
            return [prefix]
        return [p for i, (a, b) in enumerate(zip(old, new))
                for p in differences(a, b, prefix + '/' + str(i))]
    return [] if old == new else [prefix]


def load_compare():
    path = SOURCE / 'transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py'
    spec = importlib.util.spec_from_file_location('step0_compare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_native_reference(envelope, golden, workload, architecture, compare):
    # The frozen diagnostic binary renamed the display label while retaining
    # the native profile. Keep the archived result untouched and adapt only
    # this label for the older validator; all counters/timing gates still run.
    diagnostic = architecture.get('diagnostic') or {}
    require(not diagnostic.get('ideal_hbm', False), 'native case selected ideal supply')
    require(diagnostic.get('hbm_profile', 'native') == 'native', 'native HBM profile changed')
    for field in ('mac_speedup', 'activation_speedup', 'weight_port_speedup',
                  'accumulator_speedup', 'scheduler_speedup', 'dma_speedup', 'vector_speedup'):
        require(diagnostic.get(field, 1) == 1, 'native reference has a service speedup: ' + field)
    require(envelope['memory_model']['name'] in (
        'Ramulator HBM2 preset', 'Ramulator HBM2 with explicit timing profile'),
        'unrecognized native memory model')
    adapted = dict(envelope, memory_model=dict(envelope['memory_model'], name='Ramulator HBM2 preset'))
    compare.validate_run(adapted, golden, workload, architecture, 0, 0)


def prepare(output, binary, library):
    require(not (output / 'manifest.json').exists(), 'use a fresh output directory')
    repro = output / 'repro'
    repro.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, repro / 'moe_dual_normal')
    shutil.copy2(library, repro / 'libramulator.so')
    shutil.copy2(__file__, repro / 'runner.py')
    shutil.copy2(SOURCE / 'transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py',
                 repro / 'compare_moe_normal.py')
    (repro / 'source.patch').write_bytes(subprocess.check_output(
        ['git', 'diff', '--binary', 'HEAD', '--', 'transactional_emulator'], cwd=SOURCE))
    source_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    cases = []
    legacy = read(OLD / 'legacy_regression/baseline.json')
    require(len(legacy['runs']) == 28, 'legacy case list changed')
    for item in legacy['runs']:
        name = 'legacy__' + item['window'] + '__' + Path(item['architecture']).stem
        cases.append(dict(id=name, kind='legacy', architecture_source=item['architecture'],
                          workload=item['workload'], golden=str(Path(item['workload']).with_name('golden.json')),
                          reference=item['old_report'], reference_kind='envelope', ideal=False))
    for item in read(DIAG / 'manifest.json')['cases']:
        cases.append(dict(id='frozen__' + item['id'], kind='frozen',
                          architecture_source=str(DIAG / item['id'] / 'baseline.arch.json'),
                          workload=item['window']['workload']['path'], golden=item['window']['golden']['path'],
                          reference=str(DIAG / item['id'] / 'baseline.rep1.json'),
                          reference_kind='envelope', ideal=False))
    service = OLD / 'prepared/results/A_13e2c8e189054ef28c442f8681ba54fa/A/expert_me32/comparison.json'
    cases.append(dict(id='frozen__expert_me32_single_pool_q32', kind='frozen',
                      architecture_source=str(OLD / 'prepared/architectures/output_pool_threshold_pool_q32_single.json'),
                      workload=str(OLD / 'prepared/service/expert_me32/workload.json'),
                      golden=str(OLD / 'prepared/service/expert_me32/golden.json'),
                      reference=str(service), reference_kind='comparison',
                      reference_architecture='output_pool_threshold_pool_q32_single', ideal=False))
    for window in ('b8', 'b32'):
        original = next(c for c in cases if c['id'] == f'frozen__qwen_full_decode_{window}_single_pool_q32')
        item = copy.deepcopy(original)
        item.update(id=item['id'] + '__ideal_supply64', kind='frozen_ideal_oracle', ideal=True,
                    architecture_source=str(DIAG / f'qwen_full_decode_{window}_single_pool_q32/ideal_supply64.arch.json'),
                    reference=str(DIAG / f'qwen_full_decode_{window}_single_pool_q32/ideal_supply64.rep1.json'))
        cases.append(item)
    for c in cases:
        folder = output / c['id']
        folder.mkdir()
        c['architecture'] = str(folder / 'architecture.json')
        shutil.copy2(c['architecture_source'], c['architecture'])
        c['input_hashes'] = {k: digest(c[k]) for k in ('architecture', 'workload', 'golden', 'reference')}
        reference = read(c['reference'])
        if c['reference_kind'] == 'comparison':
            selected = next(x for x in reference['comparisons']
                            if x['architecture']['name'] == c['reference_architecture'])
            expected = dict(result=selected['result'], calibration=selected['native'])
        else:
            expected = dict(result=reference['result'], calibration=reference['memory_model']['calibration'])
        save(folder / 'expected.json', expected)
        c['expected_sha256'] = digest(folder / 'expected.json')
    require(len(cases) == 42, 'step0 case count changed')
    manifest = dict(schema_version=1, status='prepared', source_commit=source_commit,
                    source_tree=str(SOURCE), simulation_patch_sha256=digest(repro / 'source.patch'),
                    runner_sha256=digest(repro / 'runner.py'), binary_sha256=digest(repro / 'moe_dual_normal'),
                    validator_sha256=digest(repro / 'compare_moe_normal.py'),
                    native_library_sha256=digest(repro / 'libramulator.so'), repeats=2,
                    planned_cases=len(cases), planned_runs=2 * len(cases),
                    note='Ideal-supply runs only reproduce explicitly requested frozen references; native timing is unchanged.',
                    cases=cases)
    save(output / 'manifest.json', manifest)
    return manifest


def run_case(c, output, manifest, compare):
    folder = output / c['id']
    for field, value in c['input_hashes'].items():
        require(digest(c[field]) == value, 'input hash changed: ' + c[field])
    require(digest(folder / 'expected.json') == c['expected_sha256'], 'expected result changed')
    expected = read(folder / 'expected.json')
    arch, workload, golden = (read(c[k]) for k in ('architecture', 'workload', 'golden'))
    first = None
    rows, cores = [], []
    for repeat in (1, 2):
        path = folder / f'rep{repeat}.json'
        command = [str(output / 'repro/moe_dual_normal'), '--workload', c['workload'],
                   '--architecture', c['architecture'], '--output', str(path),
                   '--hbm-channels', '8', '--max-hbm-bytes', str(1 << 30)]
        start = time.monotonic()
        reused_result = path.exists()
        if not reused_result:
            env = dict(os.environ, LD_LIBRARY_PATH=str(output / 'repro'))
            with path.with_suffix('.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)
        d = read(path)
        result, cal, provenance = d['result'], d['memory_model']['calibration'], d['provenance']
        for field in ('output_bf16', 'output_f32', 'pre_round_output_f32'):
            require(result[field] == golden[field], c['id'] + ': golden mismatch ' + field)
        for field, expected_hash in (
            ('executable_sha256', manifest['binary_sha256']),
            ('native_library_sha256', manifest['native_library_sha256']),
            ('architecture_sha256', c['input_hashes']['architecture']),
            ('workload_sha256', c['input_hashes']['workload']),
            ('hbm_sha256', workload['metadata']['hbm_sha256']),
        ):
            require(provenance[field] == expected_hash, 'provenance mismatch: ' + field)
        delta = differences(expected['result'], result, '/result')
        delta += differences(expected['calibration'], cal, '/calibration')
        require(not delta, c['id'] + ': frozen fields differ ' + str(delta[:10]))
        if not c['ideal']:
            validate_native_reference(d, golden, workload, arch, compare)
            require(cal['native_pending'] == 0, 'native did not drain')
        else:
            require(cal is None, 'ideal reference unexpectedly has native telemetry')
        if first is not None:
            require(result == first['result'] and cal == first['memory_model']['calibration'],
                    'repeat changed: ' + c['id'])
        first = d
        nominal_lower_us = result['hbm_read_bytes'] / 256e3
        row = dict(case=c['id'], kind=c['kind'], repeat=repeat, total_ps=result['total_ps'],
                   total_us=result['total_ps'] / 1e6, frozen_us=expected['result']['total_ps'] / 1e6,
                   hbm_read_bytes=result['hbm_read_bytes'], hbm_bytes_changed=False,
                   useful_macs=result['useful_macs'], issued_macs=result['issued_macs'],
                   effective_mac_utilization=result['useful_macs'] / (result['multipliers'] * result['total_ps'] / arch['clock_period_ps']),
                   nominal_admission_lower_us=nominal_lower_us,
                   time_over_native_lower=None if c['ideal'] else result['total_ps'] / 1e6 / nominal_lower_us,
                   golden_bit_exact=True, old_fields_identical=True, native_drained=None if c['ideal'] else True,
                   reused_result=reused_result,
                   wall_seconds=round(time.monotonic() - start, 3), result_path=str(path))
        rows.append(row)
        for core in result['cores']:
            detail = core.get('refinement') or {}
            pool = detail.get('output_pool') or {}
            cores.append(dict(case=c['id'], repeat=repeat, core=core['id'], jobs=core['jobs'],
                              hbm_read_bytes=core['hbm_read_bytes'],
                              byte_share=core['hbm_read_bytes'] / result['hbm_read_bytes'],
                              compute_busy_ps=core['compute_busy_ps'],
                              weight_ready_wait_ps=core['weight_ready_wait_ps'],
                              accumulator_dependency_stall_ps=core['accumulator_dependency_stall_ps'],
                              scheduler_busy_ps=pool.get('scheduler_busy_ps', 0),
                              weight_sram_peak_bytes=core['weight_sram_peak_bytes'],
                              vector_sram_peak_bytes=core['vector_sram_peak_bytes'],
                              accumulator_peak_bytes=core['accumulator_peak_bytes']))
    save(folder / 'validation.json', dict(status='passed', repeats=2, measurements=rows))
    return rows, cores


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--workers', default=4, type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    require(1 <= args.workers <= 8, 'use 1–8 host worker processes')
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
    else:
        manifest = prepare(output, args.binary.resolve(), args.library.resolve())
    for name, key in [('moe_dual_normal', 'binary_sha256'), ('libramulator.so', 'native_library_sha256')]:
        require(digest(output / 'repro' / name) == manifest[key], 'frozen executable changed')
    compare = load_compare()
    manifest.update(status='running', workers=args.workers)
    save(manifest_path, manifest)
    rows, cores, errors = [], [], []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, c, output, manifest, compare): c for c in manifest['cases']}
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            try:
                new_rows, new_cores = future.result()
                rows.extend(new_rows)
                cores.extend(new_cores)
                print(f"PASS {case['id']} {new_rows[0]['total_us']:.6f} us", flush=True)
            except Exception as exc:
                errors.append(dict(case=case['id'], error=str(exc)))
                print(f"FAIL {case['id']}: {exc}", flush=True)
            save(output / 'progress.json', dict(completed_runs=len(rows), planned_runs=manifest['planned_runs'], errors=errors))
    rows.sort(key=lambda x: (x['case'], x['repeat']))
    cores.sort(key=lambda x: (x['case'], x['repeat'], x['core']))
    if rows:
        csv_write(output / 'measurements.csv', rows)
        csv_write(output / 'core_services.csv', cores)
    manifest.update(status='failed' if errors else 'passed', completed_runs=len(rows),
                    numerical_bit_exact_runs=len(rows), legacy_comparisons=sum(c['kind'] == 'legacy' for c in manifest['cases']),
                    native_runs=sum(r['kind'] != 'frozen_ideal_oracle' for r in rows),
                    ideal_reference_runs=sum(r['kind'] == 'frozen_ideal_oracle' for r in rows),
                    hbm_bytes_changed_runs=sum(r['hbm_bytes_changed'] for r in rows),
                    errors=errors, elapsed_wall_seconds=round(time.monotonic() - started, 3))
    save(manifest_path, manifest)
    require(not errors and len(rows) == manifest['planned_runs'], 'step0 regression did not pass')
    print(json.dumps({k: manifest[k] for k in ('status', 'completed_runs', 'native_runs', 'ideal_reference_runs')}), flush=True)


if __name__ == '__main__':
    main()
