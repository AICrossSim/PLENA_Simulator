#!/usr/bin/env python3
"""Native step1 acceptance; all frozen controls and both selectors, two repeats."""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import step0 as base

ROOT = base.ROOT
SOURCE = base.SOURCE
RUN = ROOT / 'outputs/moe_stream_ctrl_20260914/5cec1f6917944c7d8d96cb5ac7f79bc7'


def prepare(output, binary):
    repro = output / 'repro'
    repro.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, repro / 'moe_dual_normal')
    shutil.copy2(RUN / 'step0/repro/libramulator.so', repro / 'libramulator.so')
    shutil.copy2(__file__, repro / 'step1.py')
    shutil.copy2(SOURCE / 'scripts/moe_stream_ctrl/step0.py', repro / 'step0.py')
    shutil.copy2(SOURCE / 'transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py', repro / 'compare_moe_normal.py')
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=SOURCE)
    additions = ['transactional_emulator/src/moe_normal/event_ready.rs']
    additions += [str(p.relative_to(SOURCE)) for p in (SOURCE / 'scripts/moe_stream_ctrl').iterdir()
                  if p.suffix in ('.py', '.sh')]
    for name in sorted(additions):
        p = subprocess.run(['git', 'diff', '--no-index', '--binary', '/dev/null', name],
                           cwd=SOURCE, capture_output=True, check=False)
        base.require(p.returncode == 1, 'new source patch failed: ' + name)
        patch += p.stdout
    (repro / 'source.patch').write_bytes(patch)
    cases = copy.deepcopy(base.read(RUN / 'step0/manifest.json')['cases'])
    for c in cases:
        c['variant'] = 'off'
        c['expected_source'] = str(RUN / 'step0' / c['id'] / 'expected.json')
    diag = ROOT / 'outputs/moe_bottleneck_20260911/adaptive_dispatch/qwen_full_decode_b32_heterogeneous_pool_q32'
    prior = base.read(diag / 'hbm_clock_dma2.rep1.json')
    diag_expected = output / 'adaptive_expected.json'
    base.save(diag_expected, dict(result=prior['result'], calibration=prior['memory_model']['calibration']))
    cases.append(dict(id='adaptive_b32_hbm_clock_dma2', kind='adaptive_reference', variant='off', ideal=False,
                      architecture_source=str(diag / 'hbm_clock_dma2.arch.json'),
                      workload=prior['provenance']['workload_path'],
                      golden=str(Path(prior['provenance']['workload_path']).with_name('golden.json')),
                      expected_source=str(diag_expected)))
    selected = ['frozen__expert_me32_single_pool_q32', 'adaptive_b32_hbm_clock_dma2']
    for key in selected:
        reference = next(c for c in cases if c['id'] == key)
        for selection in ('rotating', 'lowest'):
            candidate = copy.deepcopy(reference)
            candidate.update(id=key + '__' + selection, variant=selection, kind='step1', baseline_id=key)
            cases.append(candidate)
    for c in cases:
        folder = output / c['id']
        folder.mkdir()
        original = base.read(c['architecture_source'])
        architecture = copy.deepcopy(original)
        if c['variant'] != 'off':
            architecture['name'] += '__event_' + c['variant']
            for core in architecture['cores']:
                base.require(core['weight_slots'] == 3 and core['refinement']['output_pool']['output_contexts'] == 32,
                             'step1 acceptance shape changed')
                core['refinement']['stream_ctrl'] = dict(event_ready=True, selection=c['variant'])
            check = copy.deepcopy(architecture)
            check['name'] = original['name']
            for core in check['cores']:
                del core['refinement']['stream_ctrl']
            base.require(check == original, 'unexpected data-path or diagnostic changes')
        c['architecture'] = str(folder / 'architecture.json')
        base.save(c['architecture'], architecture)
        shutil.copy2(c['expected_source'], folder / 'expected.json')
        c['hashes'] = {key: base.digest(c[key]) for key in ('architecture_source', 'architecture', 'workload', 'golden', 'expected_source')}
    manifest = dict(status='prepared', source_tree=str(SOURCE), base_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip(), cases=cases, repeats=2,
        planned_runs=len(cases)*2, binary_sha256=base.digest(repro / 'moe_dual_normal'),
        native_library_sha256=base.digest(repro / 'libramulator.so'), source_patch_sha256=base.digest(repro / 'source.patch'),
        runner_sha256=base.digest(repro / 'step1.py'), helper_sha256=base.digest(repro / 'step0.py'),
        validator_sha256=base.digest(repro / 'compare_moe_normal.py'),
        native_policy='per_channel for step1; unchanged legacy policies for frozen regression',
        diagnostics='The user-required adaptive B32 control and candidates alone retain frozen HBM clock/DMA 2x; ideal runs are frozen references only.',
        gate='Any step1 invariant or performance failure stops progression to step2.')
    base.save(output / 'manifest.json', manifest)
    return manifest


def run_case(c, output, manifest, validator):
    for field, digest in c['hashes'].items():
        base.require(base.digest(c[field]) == digest, 'input changed: ' + field)
    folder = output / c['id']
    architecture, workload, golden = (base.read(c[k]) for k in ('architecture', 'workload', 'golden'))
    expected = base.read(folder / 'expected.json')
    base.require(base.digest(folder / 'expected.json') == c['hashes']['expected_source'], 'expected result changed')
    first = None
    rows, core_rows, arrivals = [], [], []
    for repeat in (1, 2):
        path = folder / f'rep{repeat}.json'
        command = [str(output / 'repro/moe_dual_normal'), '--workload', c['workload'], '--architecture', c['architecture'],
                   '--output', str(path), '--hbm-channels', '8', '--max-hbm-bytes', str(1 << 30)]
        resumed = path.exists()
        start = time.monotonic()
        if not resumed:
            with path.with_suffix('.log').open('w') as log:
                subprocess.run(command, env=dict(os.environ, LD_LIBRARY_PATH=str(output / 'repro')),
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        d = base.read(path)
        r, cal = d['result'], d['memory_model']['calibration']
        for key in ('output_bf16', 'output_f32', 'pre_round_output_f32'):
            base.require(r[key] == golden[key], 'golden differs: ' + key)
        for key in ('output_f32', 'pre_round_output_f32'):
            base.require(all(struct.pack('<f', x) == struct.pack('<f', y)
                             for a, b in zip(r[key], golden[key]) for x, y in zip(a, b)), 'FP32 bits differ')
        for key, value in [('executable_sha256', manifest['binary_sha256']),
                           ('native_library_sha256', manifest['native_library_sha256']),
                           ('architecture_sha256', c['hashes']['architecture']),
                           ('workload_sha256', c['hashes']['workload']),
                           ('hbm_sha256', workload['metadata']['hbm_sha256'])]:
            base.require(d['provenance'][key] == value, 'provenance mismatch ' + key)
        base.require(r['hbm_read_bytes'] == expected['result']['hbm_read_bytes'], 'HBM byte invariant failed')
        base.require(r['useful_macs'] == expected['result']['useful_macs'], 'useful work changed')
        if c['variant'] == 'off':
            delta = base.differences(expected['result'], r) + base.differences(expected['calibration'], cal)
            base.require(not delta, 'disabled path differs from frozen reference: ' + str(delta[:5]))
        if not c['ideal']:
            validator.validate_run(d, golden, workload, architecture, 0, 0)
            base.require(cal['native_pending'] == 0, 'native HBM did not drain')
        else:
            base.require(cal is None, 'ideal reference unexpectedly used native HBM')
        if first is not None:
            base.require(r == first['result'] and cal == first['memory_model']['calibration'], 'repeat differs')
        first = d
        dma_period = (architecture['clock_period_ps'] + architecture.get('diagnostic', {}).get('dma_speedup', 1)-1) // architecture.get('diagnostic', {}).get('dma_speedup', 1)
        lower = r['hbm_read_bytes'] * dma_period / (8 * 32) / 1e6
        row = dict(case=c['id'], variant=c['variant'], kind=c['kind'], repeat=repeat, total_ps=r['total_ps'],
                   total_us=r['total_ps']/1e6, baseline_us=expected['result']['total_ps']/1e6,
                   speedup=expected['result']['total_ps']/r['total_ps'], hbm_read_bytes=r['hbm_read_bytes'],
                   hbm_bytes_changed=False, admission_lower_us=None if c['ideal'] else lower,
                   time_over_lower=None if c['ideal'] else r['total_ps']/1e6/lower,
                   useful_macs=r['useful_macs'], issued_macs=r['issued_macs'],
                   effective_mac_utilization=r['useful_macs']/(r['multipliers']*r['total_ps']/architecture['clock_period_ps']),
                   golden_bit_exact=True, native_drained=None if c['ideal'] else True,
                   host_elapsed_s=time.monotonic()-start, reused_result=resumed, result_path=str(path))
        rows.append(row)
        for core, config in zip(r['cores'], architecture['cores']):
            detail = core.get('refinement') or {}
            pool = detail.get('output_pool') or {}
            stream = detail.get('stream_ctrl') or {}
            old = next(x for x in expected['result']['cores'] if x['id'] == core['id'])
            extra = stream.get('additional_control_bytes', 0)
            # Resource before/after uses the same actual job assignment; the
            # original dynamic dispatch may have a different maximum Me.
            before = core['accumulator_peak_bytes'] - (extra if core['jobs'] else 0)
            info = dict(case=c['id'], variant=c['variant'], repeat=repeat, core=core['id'], jobs=core['jobs'],
                        hbm_read_bytes=core['hbm_read_bytes'], weight_byte_share=core['hbm_read_bytes']/r['hbm_read_bytes'],
                        compute_busy_ps=core['compute_busy_ps'], scheduler_busy_ps=pool.get('scheduler_busy_ps', 0),
                        baseline_scheduler_busy_ps=((old.get('refinement') or {}).get('output_pool') or {}).get('scheduler_busy_ps',0),
                        weight_ready_wait_ps=core['weight_ready_wait_ps'],
                        accumulator_dependency_stall_ps=core['accumulator_dependency_stall_ps'],
                        weight_port_busy_ps=detail.get('weight_port_busy_ps',0),
                        accumulator_port_busy_ps=detail.get('accumulator_port_busy_ps',0),
                        budget='accumulator', budget_bytes=config['accumulator_bytes'],
                        control_reserved_bytes=pool.get('control_reserved_bytes',0), additional_control_bytes=extra,
                        frozen_actual_accumulator_peak_bytes=old['accumulator_peak_bytes'],
                        same_jobs_without_event_control_bytes=before,
                        accumulator_peak_bytes=core['accumulator_peak_bytes'],
                        budget_free_before_bytes=config['accumulator_bytes']-before,
                        budget_free_after_bytes=config['accumulator_bytes']-core['accumulator_peak_bytes'],
                        vector_sram_peak_bytes=core['vector_sram_peak_bytes'], weight_sram_peak_bytes=core['weight_sram_peak_bytes'])
            for field in ('events_enqueued','events_processed','event_queue_peak','event_queue_wait_ps','blocked_producers_peak',
                          'fanout_writes','event_fanout_busy_ps','descriptor_reads','descriptor_writes','selector_visits',
                          'selector_busy_ps','scheduler_port_wait_ps','operand_fill_busy_ps','operand_fill_elapsed_ps'):
                info[field] = stream.get(field,0)
            info['descriptor_reads_per_tile'] = stream.get('descriptor_reads',0) / max(1,pool.get('tile_admissions',0))
            core_rows.append(info)
            if c['variant'] != 'off':
                for job in r['job_completions']:
                    if job['core'] != core['id']: continue
                    gate = next(p for p in core['projections'] if p['job'] == job['job'] and p['projection'] == 'gate')
                    arrival = gate['metrics']['first_tile_arrival_ps']
                    base.require(job['start_ps'] <= arrival <= gate['end_ps'], 'first arrival outside job')
                    arrivals.append(dict(case=c['id'], repeat=repeat, core=core['id'], job=job['job'], expert=job['expert'],
                                         me=job['rows'], job_start_ps=job['start_ps'], first_tile_arrival_ps=arrival,
                                         first_tile_arrival_latency_ns=(arrival-job['start_ps'])/1000))
    base.save(folder / 'validation.json', dict(status='passed', repeated_exactly=True, rows=rows))
    return rows, core_rows, arrivals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN/'step1')
    parser.add_argument('--binary', type=Path, default=Path('/tmp/plena-moe-dual-core-target/release/moe_dual_normal'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    base.require(1 <= args.workers <= 8, 'invalid host worker count')
    manifest = base.read(output/'manifest.json') if (output/'manifest.json').exists() else prepare(output,args.binary)
    validator = base.load_compare()
    manifest.update(status='running', workers=args.workers)
    base.save(output/'manifest.json',manifest)
    rows, cores, arrivals, errors = [], [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case,c,output,manifest,validator): c for c in manifest['cases']}
        for future in concurrent.futures.as_completed(futures):
            c = futures[future]
            try:
                r, co, ar = future.result()
                rows.extend(r); cores.extend(co); arrivals.extend(ar)
                print(f"PASS {c['id']} {r[0]['total_us']:.6f} us", flush=True)
            except Exception as exc:
                errors.append(dict(case=c['id'],error=str(exc)))
                print(f"FAIL {c['id']}: {exc}",flush=True)
            base.save(output/'progress.json',dict(completed_runs=len(rows),planned_runs=manifest['planned_runs'],errors=errors))
    for name, values in [('measurements',rows),('core_services',cores),('expert_first_tile_arrivals',arrivals)]:
        if values:
            values.sort(key=lambda x:(x['case'],x['repeat'],x.get('core',''),x.get('job',0)))
            base.csv_write(output/(name+'.csv'),values)
    gates=[]
    for mode in ('rotating','lowest'):
        me=next((r for r in rows if r['case']=='frozen__expert_me32_single_pool_q32__'+mode),None)
        small=next((r for r in cores if r['case']=='adaptive_b32_hbm_clock_dma2__'+mode and r['core']=='core1'),None)
        gates.extend([dict(selection=mode,gate='Me32 single Q32 time',observed_ps=None if me is None else me['total_ps'],
                           limit_ps=70532000,passed=me is not None and me['total_ps']<=70532000),
                      dict(selection=mode,gate='Adaptive B32 small scheduler service',observed_ps=None if small is None else small['scheduler_busy_ps'],
                           limit_ps=700000000,passed=small is not None and small['scheduler_busy_ps']<700000000)])
    invariants=not errors and len(rows)==manifest['planned_runs']
    status='passed' if invariants and all(g['passed'] for g in gates) else 'failed'
    acceptance=dict(status=status,invariants_passed=invariants,performance_gates=gates,
                    completed_runs=len(rows),planned_runs=manifest['planned_runs'],
                    native_runs=sum(not r['kind']=='frozen_ideal_oracle' for r in rows),
                    hbm_bytes_changed_runs=sum(r['hbm_bytes_changed'] for r in rows),errors=errors,
                    next_step='stop_at_step1' if status=='failed' else 'step1_passed_report_before_step2')
    base.save(output/'acceptance.json',acceptance)
    manifest.update(acceptance)
    base.save(output/'manifest.json',manifest)
    run=base.read(RUN/'run.json');run['status']='step1_'+status;run['step1_acceptance']=str(output/'acceptance.json')
    base.save(RUN/'run.json',run)
    print(json.dumps(acceptance),flush=True)


if __name__=='__main__': main()
