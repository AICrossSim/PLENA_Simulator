#!/usr/bin/env python3
"""Step1-only native D1-D4 and fixed-assignment controls; no step2 actions.

The zero-event-cost factorial points are causal oracles, never acceptance runs.
All native memory timing, traffic, numerical order and capacity checks remain.
"""
import argparse
import concurrent.futures
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time

import step0 as base

RUN = base.ROOT / 'outputs/moe_stream_ctrl_20260914/5cec1f6917944c7d8d96cb5ac7f79bc7'


def load_validator(path):
    spec = importlib.util.spec_from_file_location('diagnostic_compare', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job_order(result, arch):
    return [[j['job'] for j in sorted(result['job_completions'], key=lambda j: j['start_ps'])
             if j['core'] == c['id']] for c in arch['cores']]


def require_bit_exact(result, golden):
    for key in ('output_bf16', 'output_f32', 'pre_round_output_f32'):
        base.require(result[key] == golden[key], 'golden mismatch ' + key)
        if key != 'output_bf16':
            base.require(all(struct.pack('<f', x) == struct.pack('<f', y)
                             for a, b in zip(result[key], golden[key]) for x, y in zip(a, b)),
                         'f32 bit pattern mismatch ' + key)


def source_archive(repro):
    source = base.SOURCE
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=source)
    additions = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=source,
                                        text=True).splitlines()
    for name in additions:
        if Path(name).suffix not in ('.rs', '.py', '.sh', '.md'):
            continue
        delta = subprocess.run(['git', 'diff', '--no-index', '--binary', '/dev/null', name],
                               cwd=source, capture_output=True, check=False)
        base.require(delta.returncode == 1, 'could not archive ' + name)
        patch += delta.stdout
    (repro / 'source.patch').write_bytes(patch)
    # Save the full audited modules, so line citations survive later edits.
    for name in ('engine.rs', 'output_pool.rs', 'event_ready.rs', 'types.rs'):
        shutil.copy2(source / 'transactional_emulator/src/moe_normal' / name, repro / name)
    shutil.copy2(RUN / 'step0/repro/build-env.sh', repro / 'build-env.sh')


def prepare(output, binary):
    base.require(output.parent == RUN / 'step1', 'diagnostics must remain under step1')
    repro = output / 'repro'
    repro.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binary, repro / 'moe_dual_normal')
    shutil.copy2(RUN / 'step1/repro/libramulator.so', repro / 'libramulator.so')
    shutil.copy2(__file__, repro / 'diagnose_step1.py')
    shutil.copy2(base.SOURCE / 'scripts/moe_stream_ctrl/step0.py', repro / 'step0.py')
    shutil.copy2(base.SOURCE / 'transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py',
                 repro / 'compare_moe_normal.py')
    source_archive(repro)
    previous = {c['id']: c for c in base.read(RUN / 'step1/manifest.json')['cases']}
    me_old = previous['frozen__expert_me32_single_pool_q32']
    n3 = previous['frozen__expert_me32_single_legacy_n3']
    b32 = previous['adaptive_b32_hbm_clock_dma2']
    definitions = [
        ('me32_n3', n3, None, 0, False, False, True),
        ('me32_pool_scan', me_old, None, 2, False, False, True),
        ('me32_event_rotating_o2', me_old, 'rotating', 2, False, False, True),
        ('me32_event_lowest_o2', me_old, 'lowest', 2, False, False, True),
        ('me32_event_rotating_o3', me_old, 'rotating', 3, False, False, False),
        ('me32_zero_event_o2', me_old, 'rotating', 2, True, False, False),
        ('me32_zero_event_o3', me_old, 'rotating', 3, True, False, False),
        ('b32_adaptive_rotating', b32, 'rotating', 2, False, False, True),
        ('b32_fixed_scan', b32, None, 2, False, True, False),
        ('b32_fixed_rotating', b32, 'rotating', 2, False, True, False),
        ('b32_fixed_lowest', b32, 'lowest', 2, False, True, False),
    ]
    cases = []
    for name, origin, selection, stages, zero, fixed, compatible in definitions:
        folder = output / name
        folder.mkdir()
        architecture = base.read(origin['architecture'])
        frozen = base.read(RUN / 'step1' / origin['id'] / 'rep1.json')
        compatibility = RUN / 'step1' / origin['id'] / 'rep1.json'
        if selection:
            # Keep the frozen display name, making fieldwise compatibility exact.
            compatibility = RUN / 'step1' / (origin['id'] + '__' + selection) / 'rep1.json'
            architecture = base.read(compatibility.parent / 'architecture.json')
        changes = []
        if fixed:
            order = job_order(frozen['result'], architecture)
            architecture.setdefault('diagnostic', {})['fixed_job_order'] = order
            changes.append('fixed_job_order from frozen adaptive job starts')
        if stages == 3:
            architecture.setdefault('diagnostic', {})['allow_three_operand_stages'] = True
            for core in architecture['cores']:
                core['refinement']['output_pool']['operand_stages'] = 3
                core['refinement']['operand_latch_bytes'] = 3 * core['blen'] * core['mlen'] * 2
            changes.append('diagnostic third operand stage, charged to existing budgets')
        if zero:
            architecture.setdefault('diagnostic', {})['event_updates_zero_cost'] = True
            changes.append('zero event-update service CAUSAL ORACLE, excluded from acceptance')
        base.save(folder / 'architecture.json', architecture)
        base.save(folder / 'expected.json', dict(result=frozen['result'], calibration=frozen['memory_model']['calibration']))
        c = dict(id=name, architecture=str(folder / 'architecture.json'), workload=origin['workload'],
                 golden=origin['golden'], expected=str(folder / 'expected.json'),
                 compatibility=str(compatibility) if compatible else None, origin=origin['id'],
                 selection=selection, stages=stages, zero_event_cost=zero, fixed_assignment=fixed,
                 changes=changes)
        c['hashes'] = {k: base.digest(c[k]) for k in ('architecture', 'workload', 'golden', 'expected')}
        if compatible:
            c['hashes']['compatibility'] = base.digest(compatibility)
        cases.append(c)
    manifest = dict(status='prepared', source_tree=str(base.SOURCE),
                    base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=base.SOURCE, text=True).strip(),
                    scope='D1-D4 and B32 fixed assignment, no step2', cases=cases, repeats=2,
                    planned_runs=len(cases)*2, frozen_step1=str(RUN / 'step1'),
                    artifacts={p.name: base.digest(p) for p in repro.iterdir()
                               if p.is_file() and p.suffix != '.log'},
                    event_kind_order=['decoded_slot', 'operand_installed', 'record_k_done'])
    base.save(output / 'manifest.json', manifest)
    return manifest


def run_case(c, output, manifest, validator):
    folder = output / c['id']
    for key, value in c['hashes'].items():
        base.require(base.digest(c[key]) == value, 'input changed: ' + key)
    arch, workload, golden, expected = [base.read(c[k]) for k in ('architecture', 'workload', 'golden', 'expected')]
    first = None
    rows, core_rows = [], []
    for repeat in (1, 2):
        path = folder / f'rep{repeat}.json'
        command = [str(output / 'repro/moe_dual_normal'), '--workload', c['workload'],
                   '--architecture', c['architecture'], '--output', str(path), '--hbm-channels', '8',
                   '--max-hbm-bytes', str(1 << 30)]
        start = time.monotonic()
        if not path.exists():
            with path.with_suffix('.log').open('w') as log:
                subprocess.run(command, env=dict(os.environ, LD_LIBRARY_PATH=str(output / 'repro')),
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        envelope = base.read(path)
        r = envelope['result']
        calibration = envelope['memory_model']['calibration']
        require_bit_exact(r, golden)
        provenance = envelope['provenance']
        for key, value in (
            ('executable_sha256', manifest['artifacts']['moe_dual_normal']),
            ('native_library_sha256', manifest['artifacts']['libramulator.so']),
            ('architecture_sha256', c['hashes']['architecture']),
            ('workload_sha256', c['hashes']['workload']),
            ('hbm_sha256', workload['metadata']['hbm_sha256']),
        ):
            base.require(provenance[key] == value, 'provenance mismatch ' + key)
        for key in ('hbm_read_bytes', 'hbm_write_bytes', 'useful_macs'):
            base.require(r[key] == expected['result'][key], 'work/traffic changed ' + key)
        if c['id'] != 'b32_adaptive_rotating':
            base.require(r['issued_macs'] == expected['result']['issued_macs'], 'issued MACs changed at fixed work')
        validator.validate_run(envelope, golden, workload, arch, 0, 0)
        base.require(calibration['native_pending'] == 0, 'native requests did not drain')
        if c['compatibility']:
            old = base.read(c['compatibility'])
            difference = base.differences(old['result'], r) + base.differences(old['memory_model']['calibration'], calibration)
            base.require(not difference, 'instrumentation changed frozen fields: ' + str(difference[:8]))
        if c['fixed_assignment']:
            base.require(job_order(r, arch) == arch['diagnostic']['fixed_job_order'], 'fixed ownership/order changed')
            for new, old in zip(r['cores'], expected['result']['cores']):
                for key in ('jobs', 'hbm_read_bytes', 'useful_macs', 'issued_macs'):
                    base.require(new[key] == old[key], 'per-core frozen work changed: ' + key)
        if first is not None:
            base.require(first['result'] == r and first['memory_model']['calibration'] == calibration,
                         'two repeats differ')
        first = envelope
        rows.append(dict(case=c['id'], repeat=repeat, total_ps=r['total_ps'], total_us=r['total_ps']/1e6,
                         hbm_read_bytes=r['hbm_read_bytes'], useful_macs=r['useful_macs'], issued_macs=r['issued_macs'],
                         native_pending=calibration['native_pending'], bit_exact=True, hbm_bytes_changed=False,
                         zero_event_cost=c['zero_event_cost'], fixed_assignment=c['fixed_assignment'],
                         compatible_frozen_fields=bool(c['compatibility']), host_elapsed_s=time.monotonic()-start,
                         result_path=str(path)))
        for core, cfg in zip(r['cores'], arch['cores']):
            d = core['refinement']; pool = d.get('output_pool') or {}; stream = d.get('stream_ctrl') or {}
            row = dict(case=c['id'], repeat=repeat, core=core['id'], jobs=core['jobs'],
                       rows=sum(j['rows'] for j in r['job_completions'] if j['core'] == core['id']),
                       hbm_read_bytes=core['hbm_read_bytes'], weight_budget=cfg['weight_sram_bytes'],
                       accumulator_budget=cfg['accumulator_bytes'])
            for field in ('compute_busy_ps', 'weight_ready_wait_ps', 'accumulator_dependency_stall_ps',
                          'pipeline_drain_ps', 'vector_wait_ps', 'activation_supply_busy_ps',
                          'vector_sram_peak_bytes', 'accumulator_peak_bytes', 'weight_sram_peak_bytes'):
                row[field] = core.get(field, 0)
            for field in ('weight_port_busy_ps', 'weight_port_wait_ps', 'accumulator_port_busy_ps',
                          'accumulator_port_wait_ps', 'output_finalize_elapsed_ps', 'operand_latch_reserved_bytes'):
                row[field] = d.get(field, 0)
            for field in ('scheduler_busy_ps', 'scheduler_visits', 'tile_admissions', 'band_admissions',
                          'context_updates', 'control_reserved_bytes', 'operand_stages_peak'):
                row[field] = pool.get(field, 0)
            for field, value in stream.items():
                if isinstance(value, list):
                    for kind, item in enumerate(value): row[field + '_' + str(kind)] = item
                elif field != 'budget': row[field] = value
            core_rows.append(row)
    base.save(folder / 'validation.json', dict(status='passed', repeats_exact=True, rows=rows))
    return rows, core_rows


def write_union_csv(path, rows):
    fields = sorted({k for r in rows for k in r})
    base.csv_write(path, [{k: r.get(k) for k in fields} for r in rows])


def negative_checks(output, manifest, validator):
    """Ensure the diagnostic exceptions do not bypass the normal invariants."""
    cases = {c['id']: c for c in manifest['cases']}
    checks = []
    for case, mutation in [('me32_zero_event_o2', 'remove_zero_cost_flag'),
                           ('me32_event_rotating_o3', 'remove_stage3_opt_in'),
                           ('me32_event_rotating_o2', 'corrupt_event_service'),
                           ('me32_event_rotating_o2', 'corrupt_hbm_bytes'),
                           ('me32_event_rotating_o2', 'corrupt_pre_round_f32')]:
        c = cases[case]
        a, w, g = (base.read(c[k]) for k in ('architecture', 'workload', 'golden'))
        envelope = base.read(output / case / 'rep1.json')
        r = envelope['result']
        if mutation == 'remove_zero_cost_flag': a['diagnostic'].pop('event_updates_zero_cost')
        if mutation == 'remove_stage3_opt_in': a['diagnostic'].pop('allow_three_operand_stages')
        if mutation == 'corrupt_event_service': r['cores'][0]['refinement']['stream_ctrl']['event_service_ps_by_kind'][2] += 1000
        if mutation == 'corrupt_hbm_bytes': r['hbm_read_bytes'] += 32
        if mutation == 'corrupt_pre_round_f32': r['pre_round_output_f32'][0][0] += 1
        # Target the accounting guard, not just the manifest equality check.
        envelope['architecture_manifest'] = copy.deepcopy(a)
        try:
            require_bit_exact(r, g)
            validator.validate_run(envelope, g, w, a, 0, 0)
        except ValueError as exc:
            checks.append(dict(case=case, mutation=mutation, rejected=True, reason=str(exc)))
        else:
            raise ValueError('invalid report accepted: ' + mutation)
    base.save(output / 'validator_negative_checks.json', dict(passed=True, checks=checks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('/tmp/plena-moe-dual-core-target/release/moe_dual_normal'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    base.require(1 <= args.workers <= 4, 'use at most four native host processes')
    output = args.output.resolve()
    manifest = base.read(output / 'manifest.json') if (output / 'manifest.json').exists() else prepare(output, args.binary)
    for name, digest in manifest['artifacts'].items():
        base.require(base.digest(output / 'repro' / name) == digest, 'archived artifact changed ' + name)
    validator = load_validator(output / 'repro/compare_moe_normal.py')
    rows, cores, errors = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, c, output, manifest, validator): c for c in manifest['cases']}
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            try:
                r, c = future.result(); rows.extend(r); cores.extend(c)
                print(f"PASS {case['id']}: {r[0]['total_us']:.6f} us", flush=True)
            except Exception as exc:
                errors.append(dict(case=case['id'], error=str(exc)))
                print(f"FAIL {case['id']}: {exc}", flush=True)
            base.save(output / 'progress.json', dict(completed_runs=len(rows), planned_runs=manifest['planned_runs'], errors=errors))
    rows.sort(key=lambda r: (r['case'], r['repeat']))
    cores.sort(key=lambda r: (r['case'], r['repeat'], r['core']))
    write_union_csv(output / 'measurements.csv', rows)
    write_union_csv(output / 'core_services.csv', cores)
    base.save(output / 'validation.json', dict(status='passed' if not errors else 'failed',
              completed_runs=len(rows), planned_runs=manifest['planned_runs'], errors=errors,
              step2_started=False, oracle_runs=sum(r['zero_event_cost'] for r in rows)))
    base.require(not errors and len(rows) == manifest['planned_runs'], 'diagnostic validation failed')
    negative_checks(output, manifest, validator)


if __name__ == '__main__':
    main()
