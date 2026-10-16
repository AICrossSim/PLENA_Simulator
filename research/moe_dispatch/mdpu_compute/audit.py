"""Independent output audit and cycle-by-cycle control reference.

No claimed silicon calibration: this verifies consistency of the declared
analytical arithmetic and finite-state dispatch contracts and their outputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from random import Random

from .domain import core_from_dict, domains
from .runtime import schedule, schedule_from_costs
from research.moe_dispatch.plena65536_moe.workloads import load_workloads

BATCHES = (2, 4, 8, 16, 64, 96, 128)


def geometric(values):
    values = list(values)
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def balanced(windows, values):
    return geometric(geometric(v for w, v in zip(windows, values) if int(w['batch']) == b) for b in BATCHES)


def array_cost(core, expert, profile):
    """Re-derive tile counts independently instead of calling job_cycles."""
    ceil = lambda a, b: (a + b - 1) // b
    m, h, f = (int(expert[k]) for k in ('Me', 'H', 'F'))
    records = (2 * ceil(m, core.pm) * ceil(f, core.pn), ceil(m, core.pm) * ceil(h, core.pn))
    feeds = (ceil(h, core.r) * core.b, ceil(f, core.r) * core.b)
    stages = int(math.log2(core.groups))
    drain = core.b + 11 + stages * (2 if profile == 'pipelined_tree2' else 1)
    cycles = sum(r * (feed + drain) if profile == 'serialized_output' else r * feed + drain
                 for r, feed in zip(records, feeds))
    slots = sum(r * feed for r, feed in zip(records, feeds)) * core.macs
    return cycles, slots


def tick_reference(costs, experts, depth, setup, prefetch, bypass, threshold, wait_best=False):
    """Small independent fixed-clock interpreter; deliberately no event jumps."""
    nc, n = len(costs), len(experts)
    q = [[] for _ in costs]
    nxt = [None] * nc
    current = [None] * nc
    busy = [0] * nc
    finish = [0] * nc
    peak = [0] * nc
    source = done = steals = 0
    for t in range(20000):
        for c in range(nc):
            if current[c] is not None and current[c][1] == t:
                current[c] = None
                done += 1
        if done == n:
            return {'cycles': max(finish), 'core_cycles': busy,
                    'core_finish_cycles': finish, 'queue_max_occupancy': peak, 'steal_count': steals}

        def committed(c):
            end = current[c][1] if current[c] is not None else t
            if nxt[c] is not None:
                end = max(end, nxt[c][1]) + costs[c][nxt[c][0]]
            return end

        # A zero-cycle context is combinational, so allow its ready transition
        # within this clock; finite preparation never chains in a single clock.
        again = True
        while again:
            again = False
            for c in range(nc):
                if current[c] is None and nxt[c] is not None and nxt[c][1] <= t:
                    i = nxt[c][0]
                    finish[c] = t + costs[c][i]
                    busy[c] += costs[c][i]
                    current[c] = (i, finish[c])
                    nxt[c] = None
                    again = True
            legal = [nxt[c] is None and (prefetch or current[c] is None) for c in range(nc)]
            for c in range(nc):
                if legal[c] and q[c] and q[c][0][1] <= t:
                    i, _ = q[c].pop(0)
                    nxt[c] = (i, t + setup)
                    legal[c] = False
                    again = True
            if nc == 2 and bypass:
                for c in range(nc):
                    o = 1 - c
                    if legal[c] and not q[c] and q[o] and q[o][0][1] <= t:
                        i = q[o][0][0]
                        if max(committed(c), t + setup) + costs[c][i] < max(committed(o), t + setup) + costs[o][i]:
                            q[o].pop(0)
                            nxt[c] = (i, t + setup)
                            legal[c] = False
                            steals += 1
                            again = True
        if source < n:
            pref = 0 if nc == 1 or experts[source].get('is_shared') or experts[source]['Me'] > threshold else 1
            legal = [c for c in range(nc) if len(q[c]) < depth and (bypass or c == pref)]
            if legal:
                predicted = []
                for c in range(nc):
                    end = committed(c)
                    for i, visible in q[c]:
                        end = max(end, visible) + costs[c][i] + (0 if prefetch else setup)
                    predicted.append(max(end, t + 1 + setup) + costs[c][source])
                c = min(legal, key=lambda x: (predicted[x], x != pref))
                if wait_best and bypass:
                    c = min(range(nc), key=lambda x: (predicted[x], x != pref))
                if len(q[c]) < depth:
                    q[c].append((source, t + 1))
                    peak[c] = max(peak[c], len(q[c]))
                    source += 1
    raise AssertionError('Cycle reference did not terminate')


def verify_tick_model():
    from research.moe_dispatch.plena65536_dedicated.geometry import Core
    rng = Random(71013)
    single = (Core(1, 512, 1, 128),)
    dual = (Core(1, 256, 1, 128),) * 2
    cases = 0
    for nc in (1, 2):
        for depth in (1, 2, 8):
            for setup in (0, 4, 16):
                for prefetch in (False, True):
                    for bypass in (False, True):
                        for wait_best in (False, True):
                            experts = [{'id': i - 1, 'Me': rng.randint(1, 8), 'H': 8, 'F': 8, 'is_shared': i == 0}
                                       for i in range(12)]
                            costs = [[rng.randint(1, 100) for _ in experts] for _ in range(nc)]
                            reference = tick_reference(costs, experts, depth, setup, prefetch, bypass, 2, wait_best)
                            actual = schedule_from_costs(single if nc == 1 else dual, experts, costs,
                                                        queue_depth=depth, context_cycles=setup,
                                                        prefetch=prefetch, bypass=bypass, wait_best=wait_best)
                            for key in reference:
                                assert reference[key] == actual[key], (key, reference, actual)
                            cases += 1
    return cases


def verify_jobs(cores, window, result):
    experts = {int(e['id']): e for e in window['experts']}
    jobs = result['jobs']
    assert {j['expert_id'] for j in jobs} == set(experts)
    assert len(jobs) == len(experts)
    assert len({j['queue_entry'] for j in jobs}) == len(jobs), 'More than one source descriptor in a cycle'
    busy, ends, issued, useful = ([0] * len(cores) for _ in range(4))
    calculated_overlap = 0
    for j in jobs:
        c = j['core_id']
        e = experts[j['expert_id']]
        cycles, slots = array_cost(cores[c], e, result['profile'])
        assert cycles == j['cycles'] == j['end'] - j['start']
        assert slots == j['issued_macs']
        assert j['queue_visible'] == j['queue_entry'] + 1
        assert j['claim'] >= j['queue_visible']
        assert j['context_ready'] == j['claim'] + result['context_cycles']
        assert j['start'] >= j['context_ready']
        assert j['M'] == e['Me'] and j['H'] == e['H'] and j['F'] == e['F']
        busy[c] += cycles
        ends[c] = max(ends[c], j['end'])
        issued[c] += slots
        useful[c] += 3 * int(e['Me']) * int(e['H']) * int(e['F'])
    for c in range(len(cores)):
        core_jobs = sorted((j for j in jobs if j['core_id'] == c), key=lambda j: j['start'])
        assert all(a['end'] <= b['start'] for a, b in zip(core_jobs, core_jobs[1:]))
        context_intervals = sorted((j['claim'], j['start']) for j in core_jobs)
        assert all(a[1] <= b[0] for a, b in zip(context_intervals, context_intervals[1:])), 'Shadow overwrite'
        for j in core_jobs:
            calculated_overlap += sum(max(0, min(j['context_ready'], old['end']) - max(j['claim'], old['start']))
                                      for old in core_jobs if old is not j)
        # All descriptors leave their original FIFO as heads, whether claimed
        # locally or stolen. Half-open interval capacity includes input slots.
        queued = [j for j in jobs if j['queue_owner'] == c]
        arrivals = sorted(queued, key=lambda j: j['queue_entry'])
        removals = sorted(queued, key=lambda j: (j['claim'], j['queue_entry']))
        assert [j['expert_id'] for j in arrivals] == [j['expert_id'] for j in removals], 'Non-head dequeue'
        clocks = sorted(set(j['queue_entry'] for j in queued))
        occupancy = max((sum(j['queue_entry'] <= t < j['claim'] for j in queued) for t in clocks), default=0)
        assert occupancy == result['queue_max_occupancy'][c] <= result['queue_depth']
    assert busy == result['core_cycles'] and ends == result['core_finish_cycles']
    assert max(ends) == result['cycles']
    assert sum(issued) == result['issued_macs']
    assert sum(useful) == result['useful_MACs']
    assert result['useful_MACs'] <= result['issued_macs']
    assert calculated_overlap == result['context_overlap_cycles']
    assert result['context_exposed_cycles'] + calculated_overlap == len(jobs) * result['context_cycles']
    assert math.isclose(result['spatial_utilization'], sum(useful) / sum(issued), rel_tol=1e-13)
    assert math.isclose(result['wall_peak_utilization'], sum(useful) / (65536 * result['cycles']), rel_tol=1e-13)
    assert result['cycles'] >= sum(useful) / 65536
    atomic_lower_bound = max(min(array_cost(c, e, result['profile'])[0] for c in cores) for e in experts.values())
    assert result['cycles'] >= atomic_lower_bound
    assert result['supply_delay_cycles'] == result['migration_count'] == 0
    assert sum(useful) == sum(3 * int(e['Me']) * int(e['H']) * int(e['F']) for e in experts.values())


def audit(out):
    ticks = verify_tick_model()
    selection = json.loads((out / 'FROZEN_SELECTION.json').read_text())
    contract = json.loads((out / 'EXPERIMENT_CONTRACT.json').read_text())
    complete = json.loads((out / 'COMPLETION.json').read_text())
    dev = load_workloads('development')['development']
    heldout = load_workloads('heldout')['heldout']
    windows = {w['id']: w for w in heldout}
    domain = domains()
    assert len(dev) == 18 and len(heldout) == 135
    assert [w['id'] for w in dev] == selection['development_window_ids']
    assert not set(selection['development_window_ids']) & set(windows)
    assert complete['frozen_selection_sha256'] == hashlib.sha256((out / 'FROZEN_SELECTION.json').read_bytes()).hexdigest()
    search_rows = list(csv.DictReader((out / 'development_search.csv').open()))
    frozen = {}
    def replay(cores, window, policy, threshold, details=True):
        # Independently translate the named experiment variant to the optional
        # runtime admission gate; avoid relying on the runner's wrapper.
        return schedule(cores, window, policy='mdpu' if policy == 'mdpu_guarded' else policy,
                        threshold=threshold, details=details, wait_best=policy == 'mdpu_guarded')

    for family, chosen in selection['families'].items():
        cores = tuple(core_from_dict(c) for c in chosen['cores'])
        assert sum(c.macs for c in cores) == 65536
        assert all(c.macs % 4096 == 0 and c.pn <= 2048 and c.r <= 2048 for c in cores)
        domain_name = contract['family_policies'][family][0]
        assert cores in domain[domain_name]
        vals = [replay(cores, w, chosen['policy'], chosen['threshold'], details=False)['cycles'] for w in dev]
        assert vals == chosen['development_window_cycles']
        score = balanced(dev, vals)
        assert math.isclose(score, chosen['development_GM_cycles'], rel_tol=1e-13)
        minimum = min(float(r['development_GM_cycles']) for r in search_rows if r['family'] == family)
        assert math.isclose(score, minimum, rel_tol=1e-13)
        frozen[family] = cores
    repeat = {(r['design'], r['variant'], r['window_id']): r
              for r in csv.DictReader((out / 'repeat_check.csv').open())}
    assert all(r['two_runs_exactly_equal'] == 'True' for r in repeat.values())
    seen = set()
    main_rows = main_jobs = 0
    totals = 0
    for line in (out / 'results.jsonl').open():
        entry = json.loads(line)
        row, result = entry['row'], entry['result']
        key = row['design'], row['variant'], row['window_id']
        assert key in repeat and key not in seen
        seen.add(key)
        w = windows[row['window_id']]
        assert row['cycles'] == result['cycles']
        assert math.isclose(row['effective_GFLOPs'], 2 * result['useful_MACs'] / result['cycles'], rel_tol=1e-13)
        assert math.isclose(row['chip_utilization_pct'], 100 * result['wall_peak_utilization'], rel_tol=1e-13)
        assert result['cycles'] >= result['useful_MACs'] / 65536
        if row['variant'] == 'main':
            verify_jobs(frozen[row['design']], w, result)
            encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
            assert hashlib.sha256(encoded.encode()).hexdigest() == repeat[key]['result_sha256']
            rerun = replay(frozen[row['design']], w, row['policy'], row['threshold'])
            assert rerun == result
            main_rows += 1
            main_jobs += len(result['jobs'])
        totals += 1
    assert totals == len(repeat) == complete['rows']
    assert main_rows == len(frozen) * len(heldout) == complete['main_data_rows']
    output = {'status': 'PASS', 'independent_tick_reference_cases': ticks,
              'development_windows': len(dev), 'heldout_windows': len(heldout),
              'selected_families': len(frozen), 'all_rows_checked': totals,
              'main_rows_independently_replayed': main_rows,
              'main_whole_expert_jobs_checked': main_jobs,
              'exact_second_run_receipts': len(repeat),
              'checks': ['exact resource budgets and domain membership',
                         'development-only selected score reproduces global minimum of enumerated domain',
                         'whole-expert conservation; no active migration or same-core overlap',
                         'finite FIFO capacity, FIFO-head claims and shadow-context exclusivity',
                         'input visibility, setup latency and overlap',
                         'independent arithmetic tile count, issued slots, useful MAC and utilization',
                         'longest atomic job and full-chip throughput lower bounds',
                         'all recorded repeated hashes; main full rerun equality'],
              'limitation': 'Internal consistency of a declared timing model; no native-cycle, synthesis-area or power calibration.'}
    if contract.get('wait_best_queue_admission'):
        output['guarded_variant_scope'] = ('Geometry/threshold selection uses development windows only; '
                                           'the admission mechanism itself was revised after inspecting the prior heldout failure. '
                                           'These heldout windows are regression evidence, not fresh generalization evidence.')
    (out / 'INDEPENDENT_AUDIT.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('/home/mcl123/plena-outputs/mdpu_compute_20261013'))
    p.add_argument('--tick-only', action='store_true')
    a = p.parse_args()
    if a.tick_only:
        print({'independent_tick_reference_cases': verify_tick_model(), 'status': 'PASS'})
    else:
        audit(a.out)
