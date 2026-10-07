#!/usr/bin/env python3
"""Nonphysical oracle diagnosis; frozen Step2 defaults and reports are immutable.

A: B8/B32 x single/homo/hetero x fixed/work-conserving x 2x2.
B: B8 fixed descriptor-only/lifecycle-only control timing waivers.
C: Me1/32 on single or isolated large/small, retaining allocated resources.
"""
import argparse
import concurrent.futures
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import step0 as b
import step2 as split
import diagnose_step1 as diag

FROZEN = diag.RUN / 'step2/split_window_charged_f80ec7b41f824877a84f05919eb14e3a'
EXPERTS = b.ROOT / 'outputs/moe_output_pool_20260909/prepared/service'
EXPECTED = dict(single=417087000, homogeneous=486315000, heterogeneous=531456000)
CONDITIONS = [('charged', 'charged', False), ('zero_control', 'zero_control', False),
              ('ideal_supply', 'charged', True), ('both', 'zero_control', True)]


def prepare(out, binary):
    out.mkdir(parents=True, exist_ok=True)
    repro = out / 'repro'; repro.mkdir(exist_ok=True)
    shutil.copy2(binary, repro / 'moe_dual_normal')
    shutil.copy2(FROZEN / 'repro/libramulator.so', repro / 'libramulator.so')
    for filename in ('oracle.py', 'step0.py', 'step2.py', 'diagnose_step1.py'):
        shutil.copy2(b.SOURCE / 'scripts/moe_stream_ctrl' / filename, repro / filename)
    shutil.copy2(b.SOURCE / 'transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py', repro / 'compare_moe_normal.py')
    diag.source_archive(repro)
    shutil.copytree(b.SOURCE / 'transactional_emulator/src/moe_normal', repro / 'moe_normal', dirs_exist_ok=True)
    shutil.copy2(b.SOURCE / 'transactional_emulator/src/bin/moe_dual_normal.rs', repro / 'moe_dual_normal.rs')
    for f in ('Cargo.toml', 'Cargo.lock', 'transactional_emulator/Cargo.toml'):
        src = b.SOURCE / f
        if src.exists():
            target = repro / 'workspace' / f; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, target)
    originals = [c for c in b.read(FROZEN / 'manifest.json')['cases']
                 if c['window'] == 6 and c['aging'] == 4 and c['weighted'] and c['workload_kind'] in ('b8', 'b32')]
    cases = []

    def add(original, group, condition, control, ideal, me=None, active=None):
        organization = original['organization'] if me is None else ('single' if active is None else ('large' if active == 0 else 'small'))
        policy = original['policy'] if me is None else 'isolated_fixed'
        workload = original['workload_kind'] if me is None else f'me{me}'
        case_id = f'oracle_{group}__{workload}__{organization}__{policy}__{condition}'
        folder = out / case_id; folder.mkdir(exist_ok=True)
        arch = b.read(original['architecture']); arch['name'] = case_id
        d = arch.setdefault('diagnostic', {})
        d.update(oracle_control=control, ideal_hbm=ideal)
        for field in ('event_updates_zero_cost',):
            b.require(not d.get(field, False), 'unexpected older timing oracle')
        for field in ('scheduler_speedup', 'dma_speedup', 'mac_speedup', 'weight_port_speedup',
                      'accumulator_speedup', 'activation_speedup', 'vector_speedup'):
            b.require(d.get(field, 1) == 1, 'nonunit service factor')
        if me is not None:
            slot = 0 if active is None else active
            d['fixed_job_order'] = [[0] if i == slot else [] for i in range(len(arch['cores']))]
            work, golden = EXPERTS / f'expert_me{me}/workload.json', EXPERTS / f'expert_me{me}/golden.json'
        else:
            work, golden = Path(original['workload']), Path(original['golden'])
        b.save(folder / 'architecture.json', arch)
        ref = FROZEN / original['id'] / 'rep1.json.gz'
        c = dict(id=case_id, group=group, workload_kind=workload, organization=organization, policy=policy,
                 condition=condition, oracle_control=control, ideal_supply=ideal, architecture=str(folder / 'architecture.json'),
                 workload=str(work), golden=str(golden), reference=str(ref), active_core=(0 if active is None else active) if me else None,
                 me=me, L_reference=original['reference'], expected_bytes=3538944 if me else (60162048 if workload == 'b8' else 81395712),
                 original_architecture=original['architecture'])
        c['hashes'] = {k: b.digest(c[k]) for k in ('architecture', 'workload', 'golden', 'reference', 'original_architecture')}
        cases.append(c)

    for original in originals:
        for condition, control, ideal in CONDITIONS:
            add(original, 'A', condition, control, ideal)
    for original in originals:
        if original['workload_kind'] == 'b8' and original['policy'] == 'fixed':
            for condition in ('desc_only', 'lifecycle_only'):
                add(original, 'B', condition, condition, False)
    for me in (1, 32):
        for org, active in (('single', None), ('heterogeneous', 0), ('heterogeneous', 1)):
            original = next(c for c in originals if c['workload_kind'] == 'b8' and c['policy'] == 'fixed' and c['organization'] == org)
            for condition in ('charged', 'zero_control'):
                add(original, 'C', condition, condition, False, me, active)
    b.require(len(cases) == 66, 'oracle case matrix changed')
    manifest = dict(schema_version=1, evidence_class='nonphysical_oracle_diagnostic', status='prepared',
                    source_tree=str(b.SOURCE), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=b.SOURCE, text=True).strip(),
                    frozen_step2=str(FROZEN), repeats=2, planned_cases=len(cases), planned_runs=2 * len(cases),
                    charged_gate_expected_ps=EXPECTED, g1_calibration='B8 fixed Step1 L retained per allocated core; no recalibration',
                    g1_isolation='heterogeneous resources retained; other core idle via fixed ownership; single retains its full original budget',
                    native_library_unchanged=b.digest(repro / 'libramulator.so') == b.digest(FROZEN / 'repro/libramulator.so'),
                    cases=cases, artifacts={str(p.relative_to(repro)): b.digest(p) for p in repro.rglob('*') if p.is_file()})
    b.require(manifest['native_library_unchanged'], 'native library changed')
    b.save(out / 'manifest.json', manifest)
    return manifest


def invariants(c, arch, result, reference):
    for field in ('hbm_read_bytes', 'hbm_write_bytes', 'useful_macs', 'output_bf16', 'output_f32', 'pre_round_output_f32'):
        if c['group'] != 'C':
            b.require(result[field] == reference[field], 'frozen work differs: ' + field)
    b.require(result['hbm_read_bytes'] == c['expected_bytes'] and result['hbm_write_bytes'] == 0, 'native-sized bytes changed')
    if c['policy'] in ('fixed', 'isolated_fixed'):
        b.require(diag.job_order(result, arch) == arch['diagnostic']['fixed_job_order'], 'fixed ownership/order changed')
        if c['group'] != 'C':
            for actual, old in zip(result['cores'], reference['cores']):
                for field in ('jobs', 'hbm_read_bytes', 'useful_macs', 'issued_macs'):
                    b.require(actual[field] == old[field], 'per-core work differs: ' + field)
    if c['group'] == 'A' and c['condition'] == 'charged':
        adapted = dict(result, architecture=reference['architecture'])
        differences = b.differences(reference, adapted)
        b.require(not differences, 'charged implementation changed frozen result: ' + str(differences[:10]))
        if c['workload_kind'] == 'b8' and c['policy'] == 'fixed':
            b.require(result['total_ps'] == EXPECTED[c['organization']], 'mandatory B8 charged gate failed')


def core_rows(c, result, architecture):
    rows = []
    for observed, cfg in zip(result['cores'], architecture['cores']):
        detail = observed['refinement']; pool = detail['output_pool']; stream = detail['stream_ctrl']; split_report = detail['split_window']
        categories = (stream.get('oracle_control') or {}).get('categories', {})
        row = dict(case=c['id'], evidence_class='nonphysical_oracle_diagnostic', group=c['group'], workload=c['workload_kind'],
                   organization=c['organization'], policy=c['policy'], condition=c['condition'], core=observed['id'],
                   jobs=observed['jobs'], token_rows=sum(j['rows'] for j in result['job_completions'] if j['core'] == observed['id']),
                   p=cfg['blen'], r=cfg['mlen'], mt=detail['m_rows'], tiles=pool['tile_admissions'], issues=pool['context_updates'],
                   bursts=stream['burst_starts'], mask_cycles=stream['completion_mask_cycles'], nominal_control_cycles=pool['scheduler_visits'],
                   control_service_ps=pool['scheduler_busy_ps'], control_occupancy_ps=stream['control_occupancy_ps'],
                   control_service_fraction=pool['scheduler_busy_ps'] / result['total_ps'],
                   compute_busy_ps=observed['compute_busy_ps'], useful_macs=observed['useful_macs'], issued_macs=observed['issued_macs'],
                   weight_ready_wait_ps=observed['weight_ready_wait_ps'], accumulator_dependency_stall_ps=observed['accumulator_dependency_stall_ps'],
                   accumulator_port_busy_ps=detail['accumulator_port_busy_ps'], accumulator_port_wait_ps=detail['accumulator_port_wait_ps'],
                   hbm_read_bytes=observed['hbm_read_bytes'], weight_budget_bytes=cfg['weight_sram_bytes'],
                   simultaneous_weight_peak_bytes=split_report['live_peak_bytes'], reserved_weight_bytes=split_report['weight_reserved_bytes'],
                   accumulator_peak_bytes=observed['accumulator_peak_bytes'], accumulator_budget_bytes=cfg['accumulator_bytes'],
                   vector_peak_bytes=observed['vector_sram_peak_bytes'], vector_budget_bytes=cfg['vector_sram_bytes'],
                   event_queue_peak=stream['event_queue_peak'], output_contexts_peak=pool['contexts_peak'])
        for name in ('admission', 'header', 'load', 'arrival', 'install', 'operand_bind', 'packed_retire', 'operand_retire', 'burst_start', 'burst_complete', 'mask'):
            category = categories.get(name, dict(accesses=0, nominal_cycles=0, paid_ps=0, wait_ps=0))
            for key, value in category.items(): row[name + '_' + key] = value
        rows.append(row)
    return rows


def run_case(c, out, manifest, validator):
    folder = out / c['id']
    for field, expected in c['hashes'].items(): b.require(b.digest(c[field]) == expected, 'input changed ' + field)
    arch, workload, golden = [b.read(c[k]) for k in ('architecture', 'workload', 'golden')]
    reference_envelope = split.read(c['reference']); reference = reference_envelope['result']
    first = None; rows = []
    for repeat in (1, 2):
        path = folder / f'rep{repeat}.json.gz'
        if not path.exists():
            with tempfile.TemporaryDirectory(prefix='plena-oracle-', dir='/tmp') as temp:
                raw = Path(temp) / 'result.json'
                command = [str(out / 'repro/moe_dual_normal'), '--architecture', c['architecture'], '--workload', c['workload'],
                           '--output', str(raw), '--hbm-channels', '8', '--max-hbm-bytes', str(1 << 30)]
                with (folder / f'rep{repeat}.log').open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800, check=True,
                                   env=dict(os.environ, LD_LIBRARY_PATH=str(out / 'repro')))
                envelope = b.read(raw); split.save_gzip(path, envelope)
        else:
            envelope = split.read(path)
        validator.validate_run(envelope, golden, workload, arch, 0, 0)
        result = envelope['result']; diag.require_bit_exact(result, golden)
        invariants(c, arch, result, reference)
        native = envelope['memory_model']['calibration']
        if c['group'] == 'A' and c['condition'] == 'charged':
            b.require(native == reference_envelope['memory_model']['calibration'], 'charged native telemetry changed')
        for key, value in (('executable_sha256', manifest['artifacts']['moe_dual_normal']),
                           ('native_library_sha256', manifest['artifacts']['libramulator.so']),
                           ('architecture_sha256', c['hashes']['architecture']), ('workload_sha256', c['hashes']['workload']),
                           ('hbm_sha256', workload['metadata']['hbm_sha256'])):
            b.require(envelope['provenance'][key] == value, 'provenance mismatch ' + key)
        if first is not None:
            b.require(first['result'] == result and first['memory_model'] == envelope['memory_model'], 'non-deterministic oracle repeat')
        first = envelope
        row = dict(case=c['id'], evidence_class='nonphysical_oracle_diagnostic', group=c['group'], repeat=repeat,
                   workload=c['workload_kind'], organization=c['organization'], policy=c['policy'], condition=c['condition'],
                   oracle_control=c['oracle_control'], ideal_supply=c['ideal_supply'], total_ps=result['total_ps'], time_us=result['total_ps']/1e6,
                   hbm_read_bytes=result['hbm_read_bytes'], hbm_write_bytes=result['hbm_write_bytes'], useful_macs=result['useful_macs'],
                   issued_macs=result['issued_macs'], native_mode='bypassed_oracle' if c['ideal_supply'] else 'unchanged_native',
                   native_drained='not_applicable' if c['ideal_supply'] else native['native_pending'] == 0,
                   request_drain_pass=True, bit_exact=True, simultaneous_storage_pass=True,
                   control_service_ps_by_core=json.dumps([core['refinement']['output_pool']['scheduler_busy_ps'] for core in result['cores']], separators=(',', ':')),
                   core_jobs=json.dumps([core['jobs'] for core in result['cores']], separators=(',', ':')),
                   hbm_bytes_by_core=json.dumps([core['hbm_read_bytes'] for core in result['cores']], separators=(',', ':')),
                   result_path=str(path))
        if c['group'] == 'C':
            cfg=arch['cores'][c['active_core']]; active=result['cores'][c['active_core']]
            detail=active['refinement']; pool=detail['output_pool']; cohort=(c['me']+detail['m_rows']-1)//detail['m_rows']
            projections=active['projections']; read_elements=sum(p['m']*p['n']*((p['k']+cfg['mlen']-1)//cfg['mlen']-1) for p in projections)
            write_elements=sum(p['m']*p['n']*((p['k']+cfg['mlen']-1)//cfg['mlen']) for p in projections)
            final_elements=sum(p['m']*p['n'] for p in projections)
            row.update(me=c['me'], active_core=active['id'], p=cfg['blen'], r=cfg['mlen'], mt=detail['m_rows'],
                       tile_count=pool['tile_admissions'], nominal_control_cycles=pool['scheduler_visits'],
                       control_service_us=pool['scheduler_busy_ps']/1e6, control_service_fraction=pool['scheduler_busy_ps']/result['total_ps'],
                       compute_service_us=active['compute_busy_ps']/1e6, accumulator_port_busy_us=detail['accumulator_port_busy_ps']/1e6,
                       accumulator_port_wait_us=detail['accumulator_port_wait_ps']/1e6,
                       derived_rmw_read_bytes=4*read_elements, derived_rmw_write_bytes=4*write_elements,
                       derived_rmw_total_bytes=4*(read_elements+write_elements), derived_final_read_bytes=4*final_elements,
                       cohort_records=cohort, q_available_n_bands=pool['contexts_capacity']//cohort,
                       window_available_n_bands=min(pool['contexts_capacity']//cohort,cfg['refinement']['stream_ctrl']['split_window']['window_tiles']),
                       inactive_core_resources_retained=len(arch['cores'])>1, L_reference=c['L_reference'])
        rows.append(row)
    cores=core_rows(c,first['result'],arch)
    for core in first['result']['cores']:
        values=core['refinement']['split_window']['lifetime_changes']
        cycle_ranges=split.cycle_peaks(values,arch['clock_period_ps'],first['result']['total_ps'])
        b.require(max((sum(x[2:]) for x in cycle_ranges),default=0)==core['refinement']['split_window']['live_peak_bytes'], 'cycle peak inconsistent')
        split.save_gzip(folder / ('cycle_peaks_' + core['id'] + '.json.gz'), dict(clock_period_ps=arch['clock_period_ps'],
                       columns=['cycle_start','cycle_end_exclusive','packed_bytes','ready_operand_bytes','decode_destination_bytes'],ranges=cycle_ranges))
    diag.write_union_csv(folder/'measurements.csv',rows);diag.write_union_csv(folder/'core_services.csv',cores)
    b.save(folder/'validation.json',dict(status='passed',evidence_class='nonphysical_oracle_diagnostic',repeats_exact=True,
           bit_exact=True,request_drained=True,native_drained='not_applicable' if c['ideal_supply'] else True,
           hbm_bytes_unchanged=True,output_hashes={p.name:b.digest(p) for p in folder.glob('*.json.gz')}))
    return rows,cores


def aggregate(out,manifest):
    rows=[];cores=[]
    for c in manifest['cases']:
        folder=out/c['id']
        if not (folder/'validation.json').exists():continue
        for name,dest in (('measurements.csv',rows),('core_services.csv',cores)):
            with (folder/name).open() as f:dest.extend(csv.DictReader(f))
    if rows:
        diag.write_union_csv(out/'measurements.csv',rows);diag.write_union_csv(out/'core_services.csv',cores)
        a=[r for r in rows if r['group']=='A']
        lookup={(r['workload'],r['organization'],r['policy'],r['condition'],r['repeat']):r for r in a}
        for row in a:
            single=lookup.get((row['workload'],'single',row['policy'],row['condition'],row['repeat']))
            row['time_over_same_condition_single']=float(row['time_us'])/float(single['time_us']) if single else None
        if a:diag.write_union_csv(out/'oracle_2x2.csv',a)
        bb=[r for r in rows if r['group']=='B']+[dict(r) for r in a if r['workload']=='b8' and r['policy']=='fixed' and r['condition'] in ('charged','zero_control')]
        for row in bb:
            charged=lookup.get(('b8',row['organization'],'fixed','charged',row['repeat']))
            zero=lookup.get(('b8',row['organization'],'fixed','zero_control',row['repeat']))
            if charged:
                delta=float(charged['time_us'])-float(row['time_us']);row.update(saved_us=delta,saved_fraction_of_charged=delta/float(charged['time_us']))
                if zero:
                    denominator=float(charged['time_us'])-float(zero['time_us'])
                    row['saved_fraction_of_zero_control_gain']=delta/denominator if denominator else None
        if bb:diag.write_union_csv(out/'control_breakdown.csv',bb)
        cc=[r for r in rows if r['group']=='C']
        if cc:diag.write_union_csv(out/'g1_extremes.csv',cc)
    manifest.update(completed_runs=len(rows),status='complete' if len(rows)==manifest['planned_runs'] else 'in_progress')
    b.save(out/'manifest.json',manifest)


def is_gate(c):
    return c['group']=='A' and c['workload_kind']=='b8' and c['policy']=='fixed' and c['condition']=='charged'


def run_batch(cases,out,manifest,validator,workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(run_case,c,out,manifest,validator):c for c in cases}
        for future in concurrent.futures.as_completed(pending):
            rows,_=future.result();print('PASS',pending[future]['id'],rows[0]['time_us'],'us',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--binary',type=Path,default=Path('/tmp/plena-moe-dual-core-target/release/moe_dual_normal'))
    parser.add_argument('--group',choices=['prepare','charged_gate','a','b','c','all'],default='charged_gate')
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();out=args.output.resolve()
    manifest=b.read(out/'manifest.json') if (out/'manifest.json').exists() else prepare(out,args.binary.resolve())
    for file,h in manifest['artifacts'].items():b.require(b.digest(out/'repro'/file)==h,'archived artifact changed: '+file)
    if args.group=='prepare':print('PREPARED',len(manifest['cases']),'oracle cases');return
    validator=diag.load_validator(out/'repro/compare_moe_normal.py')
    try:
        # Mandatory first: no oracle launch before all three fixed B8 repeats pass.
        gates=[c for c in manifest['cases'] if is_gate(c)]
        run_batch(gates,out,manifest,validator,args.workers)
        b.save(out/'charged_gate.json',dict(status='passed',cases=[c['id'] for c in gates],expected_ps=EXPECTED,repeats=2))
        if args.group!='charged_gate':
            selected=[c for c in manifest['cases'] if not is_gate(c) and (args.group=='all' or c['group'].lower()==args.group)]
            run_batch(selected,out,manifest,validator,args.workers)
    except Exception as exc:
        b.save(out/'FAILURE.json',dict(status='failed',group=args.group,error=str(exc)))
        raise
    finally:aggregate(out,manifest)
    print('ORACLE GROUP COMPLETE',args.group,flush=True)


if __name__=='__main__':main()
