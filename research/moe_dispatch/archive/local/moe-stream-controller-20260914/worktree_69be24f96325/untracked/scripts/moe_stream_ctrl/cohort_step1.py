#!/usr/bin/env python3
"""Native Step1 cohort correction, charged N3 and selector/port diagnostics."""
import argparse
import concurrent.futures
import copy
import json
from pathlib import Path
import shutil
import subprocess

import step0 as base
import diagnose_step1 as diag

RUN = diag.RUN
PREVIOUS = RUN / 'step1/diagnostics_0ca4cbfb6004438ea1d86a9a5102090e'


def prepare(output, binary):
    base.require(output.parent == RUN / 'step1', 'must stay under step1')
    base.require(not (output / 'manifest.json').exists(), 'fresh manifest required')
    repro = output / 'repro'
    repro.mkdir(exist_ok=True)
    shutil.copy2(binary, repro / 'moe_dual_normal')
    shutil.copy2(RUN / 'step1/repro/libramulator.so', repro / 'libramulator.so')
    for name in ('step0.py', 'diagnose_step1.py', 'cohort_step1.py'):
        shutil.copy2(base.SOURCE / 'scripts/moe_stream_ctrl' / name, repro / name)
    shutil.copy2(base.SOURCE / 'transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py',
                 repro / 'compare_moe_normal.py')
    diag.source_archive(repro)
    for name in ('cohort.rs', 'control_port.rs', 'output_pool_tests.rs'):
        shutil.copy2(base.SOURCE / 'transactional_emulator/src/moe_normal' / name, repro / name)
    old = {c['id']: c for c in base.read(PREVIOUS / 'manifest.json')['cases']}
    # id, original accepted case, selector, cohort, ports, charge N3, exact old fields
    definitions = [
        ('me32_n3', 'me32_n3', None, False, 1, False, True),
        ('me32_n3_charged_p1', 'me32_n3', None, False, 1, True, False),
        ('me32_n3_charged_p2', 'me32_n3', None, False, 2, True, False),
        ('me32_record_rotating_o2', 'me32_event_rotating_o2', 'rotating', False, 1, False, True),
        ('me32_record_band_o2', 'me32_event_rotating_o2', 'band_rotating', False, 1, False, False),
        ('me32_record_rotating_o3', 'me32_event_rotating_o3', 'rotating', False, 1, False, True),
        ('me32_record_band_o3', 'me32_event_rotating_o3', 'band_rotating', False, 1, False, False),
        ('me32_cohort_rotating_p1', 'me32_event_rotating_o2', 'rotating', True, 1, False, False),
        ('me32_cohort_lowest_p1', 'me32_event_rotating_o2', 'lowest', True, 1, False, False),
        ('me32_cohort_band_p1', 'me32_event_rotating_o2', 'band_rotating', True, 1, False, False),
        ('me32_cohort_rotating_p2', 'me32_event_rotating_o2', 'rotating', True, 2, False, False),
        ('b32_fixed_scan', 'b32_fixed_scan', None, False, 1, False, True),
        ('b32_fixed_record_rotating', 'b32_fixed_rotating', 'rotating', False, 1, False, True),
        ('b32_fixed_cohort_rotating_p1', 'b32_fixed_rotating', 'rotating', True, 1, False, False),
        ('b32_fixed_cohort_lowest_p1', 'b32_fixed_lowest', 'lowest', True, 1, False, False),
        ('b32_fixed_cohort_rotating_p2', 'b32_fixed_rotating', 'rotating', True, 2, False, False),
    ]
    cases = []
    for name, origin_id, selection, cohort, ports, charge, compatible in definitions:
        origin = old[origin_id]
        folder = output / name
        folder.mkdir()
        architecture = base.read(origin['architecture'])
        architecture.setdefault('diagnostic', {}).update(control_ports=ports, charge_legacy_control=charge)
        if selection:
            for core in architecture['cores']:
                core['refinement']['stream_ctrl'].update(selection=selection, cohort_control=cohort)
        base.save(folder / 'architecture.json', architecture)
        shutil.copy2(origin['expected'], folder / 'expected.json')
        c = dict(id=name, architecture=str(folder / 'architecture.json'), workload=origin['workload'],
                 golden=origin['golden'], expected=str(folder / 'expected.json'),
                 compatibility=str(PREVIOUS / origin_id / 'rep1.json') if compatible else None,
                 origin=origin_id, selection=selection, cohort=cohort, control_ports=ports, charge_legacy=charge,
                 stages=origin['stages'], zero_event_cost=False, fixed_assignment=origin['fixed_assignment'])
        c['hashes'] = {k: base.digest(c[k]) for k in ('architecture', 'workload', 'golden', 'expected')}
        if compatible: c['hashes']['compatibility'] = base.digest(c['compatibility'])
        cases.append(c)
    manifest = dict(status='prepared', scope='Step1 cohort correction only', source_tree=str(base.SOURCE),
                    base_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=base.SOURCE,text=True).strip(),
                    cases=cases, repeats=2, planned_runs=2*len(cases), previous_diagnostics=str(PREVIOUS),
                    artifacts={p.name: base.digest(p) for p in repro.iterdir() if p.is_file() and p.suffix != '.log'})
    base.save(output / 'manifest.json', manifest)
    return manifest


def negative_checks(output, manifest, validator):
    cases = {c['id']: c for c in manifest['cases']}
    checks = []
    for case, mutation in [
        ('me32_cohort_rotating_p1', 'drop_mask_cost'),
        ('me32_cohort_rotating_p1', 'drop_sequencer_cost'),
        ('me32_cohort_rotating_p1', 'lose_writeback'),
        ('me32_cohort_rotating_p2', 'unfund_second_port'),
        ('me32_n3_charged_p1', 'free_legacy_issue'),
        ('me32_n3_charged_p1', 'remove_charge_opt_in'),
        ('me32_record_band_o3', 'remove_o3_opt_in'),
        ('b32_fixed_cohort_rotating_p1', 'corrupt_hbm_bytes'),
        ('b32_fixed_cohort_rotating_p1', 'corrupt_fp32'),
    ]:
        c = cases[case]
        a, w, g = (base.read(c[k]) for k in ('architecture', 'workload', 'golden'))
        e = base.read(output / case / 'rep1.json')
        r = e['result']; core = r['cores'][0]; d = core['refinement']
        if mutation == 'drop_mask_cost': d['stream_ctrl']['completion_mask_cycles'] -= 1
        elif mutation == 'drop_sequencer_cost': d['stream_ctrl']['sequencer_busy_ps'] -= 1000
        elif mutation == 'lose_writeback': d['stream_ctrl']['completed_contexts'] -= 1
        elif mutation == 'unfund_second_port': core['accumulator_peak_bytes'] -= 64
        elif mutation == 'free_legacy_issue': d['legacy_control']['service_ps_by_kind'][0] = 0
        elif mutation == 'remove_charge_opt_in': a['diagnostic']['charge_legacy_control'] = False
        elif mutation == 'remove_o3_opt_in': a['diagnostic']['allow_three_operand_stages'] = False
        elif mutation == 'corrupt_hbm_bytes': r['hbm_read_bytes'] += 32
        elif mutation == 'corrupt_fp32': r['pre_round_output_f32'][0][0] += 1
        try:
            validator.validate_run(e, g, w, a, 0, 0)
            diag.require_bit_exact(r, g)
        except (ValueError, AssertionError) as error:
            checks.append(dict(case=case, mutation=mutation, rejected=True, reason=str(error)))
        else:
            raise ValueError('validator accepted ' + mutation)
    base.save(output / 'negative_checks.json', checks)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('/tmp/plena-moe-dual-core-target/release/moe_dual_normal'))
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args(); output = args.output.resolve()
    manifest = base.read(output / 'manifest.json') if (output / 'manifest.json').exists() else prepare(output, args.binary)
    for name, digest in manifest['artifacts'].items():
        base.require(base.digest(output / 'repro' / name) == digest, 'archived artifact changed: ' + name)
    validator = diag.load_validator(output / 'repro/compare_moe_normal.py')
    rows, cores = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(diag.run_case, c, output, manifest, validator): c for c in manifest['cases']}
        for future in concurrent.futures.as_completed(futures):
            run_rows, core_rows = future.result()
            rows.extend(run_rows); cores.extend(core_rows)
            print(f"PASS {futures[future]['id']}: {run_rows[0]['total_us']:.6f} us", flush=True)
    rows.sort(key=lambda r:(r['case'],r['repeat'])); cores.sort(key=lambda r:(r['case'],r['repeat'],r['core']))
    diag.write_union_csv(output / 'measurements.csv', rows)
    diag.write_union_csv(output / 'core_services.csv', cores)
    checks = negative_checks(output, manifest, validator)
    results = {c['id']:base.read(output / c['id'] / 'rep1.json')['result'] for c in manifest['cases']}
    charged = results['me32_n3_charged_p1']['total_ps']
    event = results['me32_cohort_rotating_p1']['total_ps']
    fixed = results['b32_fixed_cohort_rotating_p1']
    fixed_arch = base.read(output / 'b32_fixed_cohort_rotating_p1/architecture.json')
    small = fixed['cores'][fixed_arch['small_core']]
    service = small['refinement']['output_pool']['scheduler_busy_ps']
    gates = dict(me32_event_ps=event, charged_n3_ps=charged, frozen_n3_ps=70_532_000,
                 me32_pass=event <= charged, fixed_b32_small_control_service_ps=service,
                 fixed_b32_pass=service < 700_000_000,
                 acceptance_pass=event <= charged and service < 700_000_000)
    base.save(output / 'acceptance.json', gates)
    manifest.update(status='complete', completed_runs=len(rows), failed_invariants=0,
                    hbm_bytes_changed_runs=0, negative_checks=len(checks), acceptance=gates)
    base.save(output / 'manifest.json', manifest)
    print(json.dumps(gates, indent=2), flush=True)

if __name__ == '__main__': main()
