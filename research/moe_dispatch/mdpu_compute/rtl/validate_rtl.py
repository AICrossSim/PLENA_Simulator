#!/usr/bin/env python3
"""Build outside scratch and validate synthesizable M-DPU control leaves."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def replay_vectors(src, out):
    sys.path.insert(0, str(src.parents[3]))
    from research.moe_dispatch.mdpu_compute.runtime import schedule, prepare_experts, estimate_cycles
    from research.moe_dispatch.plena65536_moe.workloads import load_workloads
    from research.moe_dispatch.plena65536_dedicated.geometry import Core
    frozen = json.loads((out.parent / 'FROZEN_SELECTION.json').read_text())
    windows = load_workloads('heldout')['heldout']
    path = out / 'runtime_replay_vectors.txt'
    cases = 0
    settings = (('mdpu',True,False),('mdpu',False,False),
                ('hard_threshold',True,False),('mdpu',True,True),('mdpu',False,True))
    configs = [(family, frozen['families'][family], settings)
               for family in ('compact_homo_mdpu','fair_homo_mdpu','hetero_mdpu')]
    guarded_path = out.parent / 'guarded' / 'FROZEN_SELECTION.json'
    if guarded_path.exists():
        guarded = json.loads(guarded_path.read_text())
        configs.extend(('guarded_' + family, guarded['families'][family],
                        (('mdpu',True,True),('mdpu',False,True),('hard_threshold',True,False)))
                       for family in ('compact_homo_mdpu','fair_homo_mdpu','hetero_mdpu'))
    with path.open('w') as stream:
        for family, config, settings in configs:
            cores = [Core(**{k: c[k] for k in ('b','r','rm','rn')}) for c in config['cores']]
            for policy, prefetch, wait_best in settings:
                for window in windows:
                    result = schedule(cores, window, policy=policy, prefetch=prefetch,
                                      threshold=config['threshold'],wait_best=wait_best)
                    experts = prepare_experts(window)
                    by_id = {j['expert_id']:j for j in result['jobs']}
                    bypass = int(policy == 'mdpu')
                    stream.write(f"{family}:{policy}:{prefetch}:{wait_best}:{window['id']} {len(experts)} "
                                 f"{bypass} {bypass} {int(prefetch)} {int(wait_best)} {config['threshold']} {result['cycles']}\n")
                    for e in experts:
                        j = by_id[e['id']]
                        eid = 127 if e['id'] == -1 else e['id']
                        stream.write(f"{eid} {e['Me']} {int(e.get('is_shared',False))} "
                                     f"{estimate_cycles(cores[0],e)} {estimate_cycles(cores[1],e)} "
                                     f"{j['core_id']} {j['start']} {j['end']} {j['queue_owner']} {j['queue_entry']} {j['claim']}\n")
                    cases += 1
    return path, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    src = Path(__file__).resolve().parent
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tool = shutil.which('verilator')
    if tool is None:
        raise RuntimeError('Verilator is required; validation was not run')
    version = subprocess.check_output([tool, '--version'], text=True).strip()
    replay_path, replay_count = replay_vectors(src, out)
    results = {}
    targets = [('selector', ['mdpu_selector.sv']),
               ('estimator', ['mdpu_estimator.sv']),
               ('queue_engine', ['mdpu_selector.sv', 'mdpu_queue_engine.sv'])]
    with tempfile.TemporaryDirectory(prefix='mdpu-rtl-', dir='/tmp') as tmp:
        for name, sources in targets:
            top = 'mdpu_' + name
            bench = 'queue_test.cpp' if name == 'queue_engine' else name + '_test.cpp'
            build = Path(tmp) / name
            build.mkdir()
            cmd = [tool, '--cc', '--exe', '--top-module', top,
                   '-Wall', '-Wno-UNUSED', '--Mdir', str(build),
                   '-CFLAGS', '-std=c++17 -O2']
            cmd += [str(src / s) for s in sources] + [str(src / bench)]
            result = subprocess.run(cmd, cwd=build, capture_output=True, text=True)
            (out / (name + '_build.log')).write_text(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f'{name} build failed; see {out}')
            make = subprocess.run(['make', '-C', str(build), '-f', 'V' + top + '.mk', '-j', '2'],
                                  capture_output=True, text=True)
            with (out / (name + '_build.log')).open('a') as log:
                log.write(make.stdout + make.stderr)
            if make.returncode:
                raise RuntimeError(f'{name} C++ build failed; see {out}')
            command = [str(build / ('V' + top))]
            if name == 'queue_engine': command.append(str(replay_path))
            run = subprocess.run(command, cwd=build,
                                 capture_output=True, text=True)
            (out / (name + '_test.log')).write_text(run.stdout + run.stderr)
            if run.returncode:
                raise RuntimeError(f'{name} simulation failed; see {out}')
            results[name] = json.loads(run.stdout)
    ledger = {
        'scope': 'Declared RTL state; not post-synthesis physical area or power',
        'fifo_count': 2, 'fifo_depth_each': 8, 'descriptor_bits': 80,
        'descriptor_fields_bits': {'expert_id': 7, 'tokens': 8, 'shared': 1,
                                   'dense_cost': 32, 'stream_cost': 32},
        'fifo_storage_bits': 2 * 8 * 80, 'shadow_storage_bits': 2 * 80,
        'pointer_bits': 4 * 3, 'count_bits': 2 * 4,
        'prepare_counter_bits': 2 * 3, 'shadow_valid_bits': 2,
        'queued_service_scoreboard_bits': 64,
        'declared_sequential_state_bits': 1532,
        'declared_sequential_state_bytes': 191.5,
        'integrated_declared_state_bits_including_core_countdowns_and_config': 1774,
        'integrated_declared_state_bytes_including_core_countdowns_and_config': 221.75,
        'excluded_integration_state': {
            'two_current_execution_countdowns_bits': 64,
            'two_cores_two_expert_kind_coefficients_bits': 128,
            'two_cores_tail_and_PM_shift_config_bits': 38,
            'threshold_and_mode_bits': 12,
        },
        'scope_boundary': 'Current execution countdowns supplied by compute cores; queued-work scoreboards, drain estimation, task dispatch, FIFOs, shadow readiness, and bypass implemented here.',
        'logic': ['Two 33-bit completion adders and a comparator in ingress selection',
                  'Two depth-8 ring FIFOs with head-only guarded bypass',
                  'Two independent four-word context preparation state machines',
                  'Coefficient-based ceil(Me/PM) service estimator with generic16x32-bit multiply; constant folding requires hardwired coefficients'],
        'context_words': 4,
        'latency_meaning': 'Designed four-word context setup; not calibrated PLENA context-switch latency',
        'area_mm2': None, 'power_w': None, 'fmax_ghz': None,
        'technology_synthesis_run': False, 'timing_closure_1GHz_proven': False,
        'notes': ['Queues may be implemented as FFs or small RAM macros; no claim of zero storage cost.',
                  'Verilator compiles RTL to C++ for functional simulation; it does not synthesize gates or prove 1GHz timing.',
                  'Unused metadata can be pruned by synthesis; this is a source-level upper accounting, not a netlist cell count.'],
    }
    validation = {
        'passed': True, 'tool': version, 'results': results,
        'scope': 'Selector arithmetic and tie/capacity behavior, service estimator, dual FIFO/shadow context protocol simulation',
        'full_runtime_cycle_equivalence': True,
        'runtime_replay_cases': replay_count,
        'runtime_replay_scope': 'Every expert queue binding/time, core assignment/start/end and layer makespan; all135heldout x3originalfrozengeometries x5settings, plus all135 x3guarded-frozengeometries x3guard/no-prefetch/hard-settings when guarded selection exists; defaultcontext4.',
        'not_validated': ['Physical matrix datapath timing', 'Arbitrary parameter settings outside validated replay domain',
                          'Technology timing closure, mm2, watts, energy'],
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(src.iterdir()) if p.suffix in {'.sv', '.cpp', '.py'}},
        'runtime_source_sha256': hashlib.sha256((src.parent/'runtime.py').read_bytes()).hexdigest(),
        'runtime_replay_vectors_sha256': hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        'frozen_inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (out.parent/'FROZEN_SELECTION.json',
                                          out.parent/'guarded'/'FROZEN_SELECTION.json') if p.exists()},
    }
    (out / 'hardware_resources.json').write_text(json.dumps(ledger, indent=2) + '\n')
    (out / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
