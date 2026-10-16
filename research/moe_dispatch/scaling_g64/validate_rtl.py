"""Replay new scaling schedules against the unchanged dual-worker M-DPU RTL.

This is a control-protocol test. Verilator does not validate floating-point
datapath timing, physical implementation, area, power, or 1 GHz closure.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def make_vectors(output):
    from .runtime import estimate_cycles, prepare_experts
    from .geometry import Core
    from research.moe_dispatch.plena65536_moe.workloads import load_workloads

    windows = {w['id']: w for w in load_workloads('heldout')['heldout']}
    directory = output / 'rtl'
    directory.mkdir(parents=True, exist_ok=True)
    streams, counts, skipped = {}, {}, 0
    try:
        for line in (output / 'records.jsonl').open():
            entry = json.loads(line)
            if entry['profile'] != 'pipelined_tree1':
                continue
            if len(entry['cores']) != 2 or entry['groups'] != [[0], [1]]:
                skipped += 1
                continue
            policy = entry['policy']
            assert policy in {'static', 'naive', 'no_preconfig', 'full'}, policy
            name = entry['design_id']
            if name not in streams:
                streams[name] = (directory / (name + '_vectors.txt')).open('w')
                counts[name] = 0
            stream = streams[name]
            cores = [Core(**{k: c[k] for k in ('b', 'r', 'rm', 'rn')}) for c in entry['cores']]
            experts = prepare_experts(windows[entry['window_id']])
            result = entry['result']
            jobs = {j['expert_id']: j for j in result['jobs']}
            bypass = int(policy != 'static')
            prefetch = int(policy != 'no_preconfig')
            wait = int(policy in {'full', 'no_preconfig'})
            stream.write(f"{name}:{policy}:{entry['window_id']} {len(experts)} "
                         f"{bypass} {bypass} {prefetch} {wait} 2 {result['cycles']}\n")
            for e in experts:
                j = jobs[e['id']]
                eid = 127 if e['id'] == -1 else e['id']
                stream.write(f"{eid} {e['Me']} {int(e.get('is_shared', False))} "
                             f"{estimate_cycles(cores[0], e)} {estimate_cycles(cores[1], e)} "
                             f"{j['core_id']} {j['start']} {j['end']} {j['queue_owner']} "
                             f"{j['queue_entry']} {j['claim']}\n")
            counts[name] += 1
    finally:
        for stream in streams.values():
            stream.close()
    return directory, counts, skipped


def strengthen_claim_replay(original, destination):
    """Add claim-clock checks omitted by the archived C++ replay harness.

    The SV modules remain unchanged. The archived bench already parsed each
    expected claim clock but only asserted ingress and compute issue clocks.
    Tracking accepted FIFO IDs makes internal dequeue clocks observable too.
    """
    text = original.read_text()
    needle = 'std::array<unsigned,2> busy{0,0};unsigned pos=0,done=0,actual_end=0;'
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '\n        std::deque<unsigned> audit_fifo[2];')
    needle = 'unsigned eid[2]={d.issue_expert0,d.issue_expert1},cost[2]={d.issue_cost0,d.issue_cost1};'
    assert text.count(needle) == 1
    text = text.replace(needle, needle + '''
            bool claim[2]={bool(d.claim_valid0),bool(d.claim_valid1)};
            bool stolen[2]={bool(d.stolen0),bool(d.stolen1)};
            for(unsigned c=0;c<2;++c)if(claim[c]){
                unsigned q=stolen[c]?1-c:c;
                check(!audit_fifo[q].empty(), name+" claim from empty audit FIFO");
                unsigned id=audit_fifo[q].front();audit_fifo[q].pop_front();
                if(claimclock[id]!=t || owners[id]!=c){
                    std::cerr<<name<<" claim expert="<<id<<" actual="<<c<<","<<t
                        <<" expected="<<owners[id]<<","<<claimclock[id]<<"\\n";
                    throw std::runtime_error("runtime replay claim-clock mismatch");
                }
            }
''')
    needle = '                ++pos;\n            }\n            for(unsigned c=0;c<2;++c)'
    assert text.count(needle) == 1
    text = text.replace(needle, '                audit_fifo[d.accepted_stream].push_back(id);\n' + needle)
    destination.write_text(text)
    return destination


def validate(output, jobs=4):
    tool = shutil.which('verilator')
    if not tool:
        raise RuntimeError('Verilator unavailable: RTL validation was not performed')
    src = Path(__file__).resolve().parents[1] / 'mdpu_compute' / 'rtl'
    directory, counts, skipped = make_vectors(output)
    enhanced_bench = strengthen_claim_replay(src / 'queue_test.cpp', directory / 'queue_claim_test.cpp')
    results = {}
    with tempfile.TemporaryDirectory(prefix='scaling-g64-rtl-', dir='/tmp') as temporary:
        binaries = {}
        for name, sources in [('selector', ['mdpu_selector.sv']),
                              ('estimator', ['mdpu_estimator.sv']),
                              ('queue_engine', ['mdpu_selector.sv', 'mdpu_queue_engine.sv'])]:
            top = 'mdpu_' + name
            bench = 'queue_test.cpp' if name == 'queue_engine' else name + '_test.cpp'
            build = Path(temporary) / name
            build.mkdir()
            command = [tool, '--cc', '--exe', '--top-module', top, '-Wall', '-Wno-UNUSED',
                       '--Mdir', str(build), '-CFLAGS', '-std=c++17 -O2']
            command += [str(src / s) for s in sources] + [str(enhanced_bench if name == 'queue_engine' else src / bench)]
            generated = subprocess.run(command, cwd=build, capture_output=True, text=True)
            log = generated.stdout + generated.stderr
            if generated.returncode:
                (directory / (name + '_build.log')).write_text(log)
                raise RuntimeError(name + ': Verilator translation failed')
            compiled = subprocess.run(['make', '-C', str(build), '-f', 'V' + top + '.mk', '-j', '2'],
                                      capture_output=True, text=True)
            log += compiled.stdout + compiled.stderr
            (directory / (name + '_build.log')).write_text(log)
            if compiled.returncode:
                raise RuntimeError(name + ': C++ compilation failed')
            binaries[name] = build / ('V' + top)
            if name != 'queue_engine':
                run = subprocess.run([str(binaries[name])], capture_output=True, text=True)
                (directory / (name + '_test.log')).write_text(run.stdout + run.stderr)
                if run.returncode:
                    raise RuntimeError(name + ': test failed')
                results[name] = json.loads(run.stdout)

        def replay(name):
            run = subprocess.run([str(binaries['queue_engine']), str(directory / (name + '_vectors.txt'))],
                                 capture_output=True, text=True)
            (directory / (name + '_test.log')).write_text(run.stdout + run.stderr)
            if run.returncode:
                raise RuntimeError(name + ': cycle replay failed; see ' + str(directory))
            value = json.loads(run.stdout)
            assert value['runtime_trace_replays'] == counts[name]
            return name, value

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results['queue_engine_by_design'] = dict(pool.map(replay, sorted(counts)))
    validation = {
        'passed': True,
        'tool': subprocess.check_output([tool, '--version'], text=True).strip(),
        'results': results,
        'runtime_trace_replays': sum(counts.values()),
        'replays_by_design': counts,
        'excluded_single_or_multilane_points': skipped,
        'scope': 'Unchanged RTL, all recorded main-profile two-worker schedules: ingress destination and cycle, FIFO claim worker/cycle, issue worker/start/end, and final makespan; static/naive/no-preconfiguration/full policies.',
        'claim_clock_checks_added_to_archived_bench': True,
        'enhanced_queue_bench_sha256': sha(enhanced_bench),
        'multilane_RTL_equivalence': False,
        'numerical_datapath_validated_by_this_test': False,
        'timing_closure_1GHz_proven': False,
        'PPA_synthesis_performed': False,
        'source_sha256': {p.name: sha(p) for p in sorted(src.iterdir()) if p.suffix in {'.sv', '.cpp'}},
        'validation_driver_sha256': sha(__file__),
        'runtime_source_sha256': sha(Path(__file__).parent / 'runtime.py'),
        'records_sha256': sha(output / 'records.jsonl'),
        'vectors_sha256': {n: sha(directory / (n + '_vectors.txt')) for n in counts},
    }
    (directory / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    return validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(validate(args.output.resolve(), args.jobs), indent=2))
