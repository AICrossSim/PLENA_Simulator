"""Independent arithmetic, schedule and BF16 tiling audit for fixed-G scaling.

The schedule audit covers every recorded real routing window. Numerical tests
cover synthetic BF16 operands and tiled GEMM indexing, not pretrained model
accuracy or a native PLENA floating-point pipeline.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from random import Random


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def array_cost(c, expert, profile):
    """Independent expansion of PM/PN padding and BLEN*G reduction chunks."""
    ceil = lambda a, b: (a + b - 1) // b
    m, h, f = (int(expert[k]) for k in ('Me', 'H', 'F'))
    records = (2 * ceil(m, c.pm) * ceil(f, c.pn), ceil(m, c.pm) * ceil(h, c.pn))
    feed = (ceil(h, c.r) * c.b, ceil(f, c.r) * c.b)
    stage_cost = 2 if profile == 'pipelined_tree2' else 1
    drain = c.b + 11 + (c.groups.bit_length() - 1) * stage_cost
    cycles = sum(a * (b + drain) if profile == 'serialized_output' else a * b + drain
                 for a, b in zip(records, feed))
    slots = sum(a * b for a, b in zip(records, feed)) * c.macs
    return cycles, slots


def tick_reference(costs, experts, depth, setup, prefetch, bypass, threshold, wait_best=False):
    """Independent per-clock two-worker interpreter; no event jumping."""
    nc, n = len(costs), len(experts)
    q = [[] for _ in costs]
    nxt, current = [None] * nc, [None] * nc
    busy, finish, peak = [0] * nc, [0] * nc, [0] * nc
    source = done = steals = 0
    for t in range(20000):
        for c in range(nc):
            if current[c] is not None and current[c][1] == t:
                current[c] = None
                done += 1
        if done == n:
            return {'cycles': max(finish), 'core_cycles': busy, 'core_finish_cycles': finish,
                    'queue_max_occupancy': peak, 'steal_count': steals}
        dequeued = [False] * nc

        def committed(c):
            end = current[c][1] if current[c] is not None else t
            if nxt[c] is not None:
                end = max(end, nxt[c][1]) + costs[c][nxt[c][0]]
            return end

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
                if legal[c] and q[c] and q[c][0][1] <= t and not dequeued[c]:
                    i, _ = q[c].pop(0)
                    nxt[c] = (i, t + setup)
                    dequeued[c] = True
                    legal[c] = False
                    again = True
            if nc == 2 and bypass:
                for c in range(nc):
                    o = 1 - c
                    if legal[c] and not q[c] and q[o] and q[o][0][1] <= t and not dequeued[o]:
                        i = q[o][0][0]
                        if max(committed(c), t + setup) + costs[c][i] < max(committed(o), t + setup) + costs[o][i]:
                            q[o].pop(0)
                            nxt[c] = (i, t + setup)
                            dequeued[o] = True
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
    raise AssertionError('Fixed-clock reference did not terminate')


def verify_tick_model():
    from .geometry import Core
    from .runtime import schedule_from_costs
    rng = Random(261015)
    cases = 0
    for scale in (65536, 131072, 262144):
        for nc in (1, 2):
            pm = scale // (nc * 256 * 64)
            cores = (Core(pm, pm * 64, 1, 256 // pm),) * nc
            for depth in (1, 2, 8):
                for setup in (0, 4, 16):
                    for prefetch in (False, True):
                        for bypass in (False, True):
                            for wait_best in (False, True):
                                experts = [{'id': i - 1, 'Me': rng.randint(1, 32), 'H': 8, 'F': 8,
                                            'is_shared': i == 0} for i in range(12)]
                                costs = [[rng.randint(1, 100) for _ in experts] for _ in range(nc)]
                                reference = tick_reference(costs, experts, depth, setup, prefetch, bypass, 2, wait_best)
                                actual = schedule_from_costs(cores, experts, costs, queue_depth=depth,
                                                             context_cycles=setup, prefetch=prefetch,
                                                             bypass=bypass, wait_best=wait_best)
                                for key in reference:
                                    assert reference[key] == actual[key], (scale, nc, key, reference, actual)
                                cases += 1
    return cases


def pooled_tick_reference(costs, experts, groups, depth, setup, prefetch, bypass, wait_best):
    """Clock-stepped two-FIFO reference for three/four independently active workers."""
    nc, n = len(costs), len(experts)
    queues = [[], []]
    current, shadow = [None] * nc, [None] * nc
    busy, finished, peaks = [0] * nc, [0] * nc, [0, 0]
    source = retired = steals = 0
    for clock in range(20000):
        for c in range(nc):
            if current[c] and current[c][1] == clock:
                current[c] = None
                retired += 1
        if retired == n:
            return dict(cycles=max(finished), core_cycles=busy, core_finish_cycles=finished,
                        queue_max_occupancy=peaks, steal_count=steals)
        dequeued = [False, False]

        def committed(c):
            end = current[c][1] if current[c] else clock
            if shadow[c]:
                i, ready = shadow[c]
                end = max(end, ready) + costs[c][i]
            return end

        def eligible(c):
            return shadow[c] is None and (prefetch or current[c] is None)

        def claim_cost(c, i):
            return max(committed(c), clock + setup) + costs[c][i]

        def pull(c, g):
            i, _ = queues[g].pop(0)
            shadow[c] = (i, clock + setup)
            dequeued[g] = True

        progress = True
        while progress:
            progress = False
            for c in range(nc):
                if current[c] is None and shadow[c] and shadow[c][1] <= clock:
                    i, _ = shadow[c]
                    shadow[c] = None
                    finished[c] = clock + costs[c][i]
                    current[c] = (i, finished[c])
                    busy[c] += costs[c][i]
                    progress = True
            for g in range(2):
                candidates = [c for c in groups[g] if eligible(c)]
                if candidates and queues[g] and queues[g][0][1] <= clock and not dequeued[g]:
                    i = queues[g][0][0]
                    c = min(candidates, key=lambda c: (claim_cost(c, i), c))
                    pull(c, g)
                    progress = True
            if bypass:
                for g in range(2):
                    o = 1 - g
                    candidates = [c for c in groups[g] if eligible(c)]
                    if candidates and not queues[g] and queues[o] and queues[o][0][1] <= clock and not dequeued[o]:
                        i = queues[o][0][0]
                        c = min(candidates, key=lambda c: (claim_cost(c, i), c))
                        if claim_cost(c, i) < min(claim_cost(c2, i) for c2 in groups[o]):
                            pull(c, o)
                            steals += 1
                            progress = True
        if source < n:
            estimates = []
            for g in range(2):
                ready = {c: committed(c) for c in groups[g]}
                port = clock + int(dequeued[g])
                for i, visible in queues[g]:
                    arrival = max(port, visible)

                    def complete(c):
                        start = max(ready[c], arrival + setup) if prefetch else max(ready[c], arrival) + setup
                        return start + costs[c][i]

                    c = min(groups[g], key=lambda c: (complete(c), c))
                    ready[c] = complete(c)
                    port = arrival + 1
                arrival = max(port, clock + 1)
                estimates.append(min((max(ready[c], arrival + setup) if prefetch else max(ready[c], arrival) + setup)
                                     + costs[c][source] for c in groups[g]))
            preferred = 0 if experts[source].get('is_shared') or experts[source]['Me'] > 2 else 1
            choices = [g for g in range(2) if len(queues[g]) < depth and (bypass or g == preferred)]
            if choices:
                considered = range(2) if bypass and wait_best else choices
                g = min(considered, key=lambda g: (estimates[g], g != preferred))
                if len(queues[g]) < depth:
                    queues[g].append((source, clock + 1))
                    peaks[g] = max(peaks[g], len(queues[g]))
                    source += 1
    raise AssertionError('Pooled clock reference did not terminate')


def verify_pooled_tick_model():
    from .geometry import Core
    from .runtime import schedule_from_costs
    rng = Random(9261015)
    systems = [((Core(4, 256, 1, 64), Core(1, 64, 1, 512), Core(1, 64, 1, 512)), ((0,), (1, 2))),
               ((Core(2, 128, 1, 256),) * 4, ((0, 1), (2, 3)))]
    cases = 0
    for cores, groups in systems:
        for depth in (1, 2, 8):
            for setup in (0, 4, 16):
                for prefetch in (False, True):
                    for bypass in (False, True):
                        for wait_best in (False, True):
                            experts = [{'id': i - 1, 'Me': rng.randint(1, 32), 'H': 8, 'F': 8,
                                        'is_shared': i == 0} for i in range(16)]
                            costs = [[rng.randint(1, 100) for _ in experts] for _ in cores]
                            expected = pooled_tick_reference(costs, experts, groups, depth, setup, prefetch, bypass, wait_best)
                            actual = schedule_from_costs(cores, experts, costs, groups=groups, queue_depth=depth,
                                                         context_cycles=setup, prefetch=prefetch,
                                                         bypass=bypass, wait_best=wait_best)
                            for key in expected:
                                assert expected[key] == actual[key], (key, groups, depth, setup, expected, actual)
                            cases += 1
    return cases


def verify_jobs(cores, groups, window, result):
    experts = {int(e['id']): e for e in window['experts']}
    jobs = result['jobs']
    budget = sum(c.macs for c in cores)
    maximum_cost = max(array_cost(c, e, result['profile'])[0] for c in cores for e in experts.values())
    assert maximum_cost * result['queue_depth'] < 2 ** 32, 'FIFO service scoreboard overflows RTL width'
    assert result['cycles'] < 2 ** 32, 'Current/finish counters exceed declared width'
    assert len(jobs) == len(experts)
    assert {j['expert_id'] for j in jobs} == set(experts)
    assert len({j['queue_entry'] for j in jobs}) == len(jobs)
    busy, ends, issued, useful = ([0] * len(cores) for _ in range(4))
    overlap = 0
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
        assert (j['M'], j['H'], j['F']) == (e['Me'], e['H'], e['F'])
        busy[c] += cycles
        ends[c] = max(ends[c], j['end'])
        issued[c] += slots
        useful[c] += 3 * int(e['Me']) * int(e['H']) * int(e['F'])
    for c in range(len(cores)):
        cj = sorted((j for j in jobs if j['core_id'] == c), key=lambda j: j['start'])
        assert all(a['end'] <= b['start'] for a, b in zip(cj, cj[1:])), 'Worker overlap'
        contexts = sorted((j['claim'], j['start']) for j in cj)
        assert all(a[1] <= b[0] for a, b in zip(contexts, contexts[1:])), 'Shadow overwrite'
        for j in cj:
            overlap += sum(max(0, min(j['context_ready'], old['end']) - max(j['claim'], old['start']))
                           for old in cj if old is not j)
    for g in range(len(groups)):
        queued = [j for j in jobs if j['queue_owner'] == g]
        arrivals = sorted(queued, key=lambda j: j['queue_entry'])
        removals = sorted(queued, key=lambda j: (j['claim'], j['queue_entry']))
        assert [j['expert_id'] for j in arrivals] == [j['expert_id'] for j in removals], 'Non-head dequeue'
        assert len({j['claim'] for j in queued}) == len(queued), 'Multiple dequeues from one FIFO in a clock'
        clocks = {j['queue_entry'] for j in queued}
        occupancy = max((sum(j['queue_entry'] <= t < j['claim'] for j in queued) for t in clocks), default=0)
        assert occupancy == result['queue_max_occupancy'][g] <= result['queue_depth']
    assert busy == result['core_cycles'] and ends == result['core_finish_cycles']
    assert max(ends) == result['cycles']
    assert sum(issued) == result['issued_macs']
    assert sum(useful) == result['useful_MACs'] <= result['issued_macs']
    assert overlap == result['context_overlap_cycles']
    assert result['context_exposed_cycles'] + overlap == len(jobs) * result['context_cycles']
    assert math.isclose(result['spatial_utilization'], sum(useful) / sum(issued), rel_tol=1e-13)
    assert math.isclose(result['wall_peak_utilization'], sum(useful) / (budget * result['cycles']), rel_tol=1e-13)
    assert result['cycles'] >= sum(useful) / budget
    atomic_bound = max(min(array_cost(c, e, result['profile'])[0] for c in cores) for e in experts.values())
    assert result['cycles'] >= atomic_bound
    assert result['supply_delay_cycles'] == result['migration_count'] == 0
    assert sum(useful) == sum(3 * int(e['Me']) * int(e['H']) * int(e['F']) for e in experts.values())
    return len(jobs)


def verify_bf16_tiles(cores):
    """Real BF16-rounded inputs, FP32 accumulation, independent integer answer.

    Small integer inputs are exactly BF16 representable and their sums fit
    FP32, so padding/indexing errors cannot hide behind loose tolerances.
    This intentionally is not a model-accuracy or general rounding proof.
    """
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    import numpy as np
    rng = np.random.default_rng(261015)

    def bf16(a):
        bits = np.asarray(a, dtype=np.float32).view(np.uint32)
        rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & 1)
        return (rounded & np.uint32(0xffff0000)).view(np.float32)

    cases = []
    for c in cores:
        shapes = [(c.pm + 1, c.pn + 3, c.r + 3), (1, 1, 1), (2, 7, 65)]
        shapes += [(min(c.pm + 1, 3), 17, k) for k in (1408, 2048, 2816)]
        for m, n, k in shapes:
            x_int = rng.integers(-2, 3, size=(m, k), dtype=np.int64)
            w_int = rng.integers(-2, 3, size=(k, n), dtype=np.int64)
            x, w = bf16(x_int), bf16(w_int)
            answer = np.zeros((m, n), dtype=np.float32)
            for mi in range(0, m, c.pm):
                mm = min(c.pm, m - mi)
                for ni in range(0, n, c.pn):
                    nn = min(c.pn, n - ni)
                    acc = np.zeros((c.pm, c.pn), dtype=np.float32)
                    for ki in range(0, k, c.r):
                        kk = min(c.r, k - ki)
                        xt = np.zeros((c.pm, c.r), dtype=np.float32)
                        wt = np.zeros((c.r, c.pn), dtype=np.float32)
                        xt[:mm, :kk] = x[mi:mi + mm, ki:ki + kk]
                        wt[:kk, :nn] = w[ki:ki + kk, ni:ni + nn]
                        acc += xt @ wt
                    answer[mi:mi + mm, ni:ni + nn] = acc[:mm, :nn]
            expected = x_int @ w_int
            assert np.array_equal(answer, expected), (c, m, n, k)
            cases.append({'pm': c.pm, 'pn': c.pn, 'G': c.groups, 'BLEN': c.b, 'reduction_chunk': c.r,
                          'M': m, 'N': n, 'K': k, 'exact_match': True})
    return {'passed': True, 'cases': len(cases), 'case_details': cases,
            'precision': 'BF16-rounded synthetic inputs, FP32 tiled accumulation; exact independent int64 reference.',
            'not_covered': ['Real model weights or end-to-end perplexity/accuracy',
                            'Native BF16 multiplier/adder rounding at each tree node',
                            'Native matrix unit cycle timing or synthesis']}


def audit(output):
    from .geometry import Core
    from research.moe_dispatch.plena65536_moe.workloads import load_workloads
    windows = {w['id']: w for w in load_workloads('heldout')['heldout']}
    assert len(windows) == 135
    assert Counter(w['batch'] for w in windows.values()) == {2: 27, 4: 27, 8: 27, 16: 27, 64: 9, 96: 9, 128: 9}
    ticks = verify_tick_model()
    pooled_ticks = verify_pooled_tick_model()
    receipt = json.loads((output / 'run_receipt.json').read_text())
    assert receipt['completed'] and receipt['all_repetitions_byte_identical']
    assert receipt['records_sha256'] == sha(output / 'records.jsonl')
    repeated = {(d, policy, profile, window): digest for d, profile, policy, window, digest
                in json.loads((output / 'repeat_sha256.json').read_text())}
    rows = jobs = 0
    geometries, seen = {}, set()
    by_design = Counter()
    coverage = {}
    for line in (output / 'records.jsonl').open():
        entry = json.loads(line)
        key = (entry['design_id'], entry['policy'], entry['profile'], entry['window_id'])
        assert key not in seen, ('Duplicate result', key)
        seen.add(key)
        cores = [Core(**{k: c[k] for k in ('b', 'r', 'rm', 'rn')}) for c in entry['cores']]
        groups = entry['groups']
        assert sorted(w for group in groups for w in group) == list(range(len(cores)))
        assert sum(c.macs for c in cores) == entry['scale']
        assert all(c.groups == 64 for c in cores)
        for c in cores:
            geometries[(c.b, c.r, c.rm, c.rn)] = c
        result = entry['result']
        encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
        assert repeated[key] == hashlib.sha256(encoded.encode()).hexdigest()
        coverage.setdefault(key[:3], set()).add(entry['window_id'])
        assert result['profile'] == entry['profile']
        jobs += verify_jobs(cores, groups, windows[entry['window_id']], result)
        if entry['policy'] == 'static' and len(groups) == 2:
            owner_group = {w: g for g, ws in enumerate(groups) for w in ws}
            for job in result['jobs']:
                expected = 0 if job['is_shared'] or job['M'] > 2 else 1
                assert owner_group[job['core_id']] == expected
        rows += 1
        by_design[entry['design_id']] += 1
    assert seen == set(repeated)
    assert rows == receipt['unique_points'] and receipt['executions'] == 2 * rows
    assert all(ids == set(windows) for ids in coverage.values()), 'Missing heldout windows'
    configs = json.loads((output / 'configs.json').read_text())
    protocol = json.loads((output / 'PROTOCOL.json').read_text())
    expected_sets = {(c['design_id'], policy, profile) for c in configs for profile in protocol['profiles']
                     for policy in (protocol['policies'] if profile == 'pipelined_tree1' else ['full'])}
    assert set(coverage) == expected_sets, 'Missing design/profile/policy cross product'
    numerical = verify_bf16_tiles([geometries[k] for k in sorted(geometries)])
    (output / 'numerical_validation.json').write_text(json.dumps(numerical, indent=2) + '\n')
    result = {
        'status': 'PASS', 'heldout_windows': 135, 'rows_checked': rows,
        'whole_expert_jobs_checked': jobs, 'rows_by_design': dict(by_design),
        'byte_identical_repeat_receipts_verified': len(repeated),
        'independent_fixed_clock_reference_cases': ticks,
        'independent_multilane_fixed_clock_reference_cases': pooled_ticks,
        'unique_physical_geometries': len(geometries),
        'BF16_tiled_numerical_cases': numerical['cases'],
        'checks': ['Independent BLEN*G tile arithmetic and useful/padded MAC conservation',
                   'Single source descriptor per cycle and FIFO-head claims',
                   'Per-worker compute exclusivity, current/next context exclusivity',
                   'Per-logical-queue depth limit and at most one dequeue per clock',
                   'Exact total MAC budgets and G=64',
                   '32-bit service/countdown scoreboard bounds for every recorded workload',
                   'Every real expert executes exactly once; no migration',
                   'Full-chip throughput and atomic task lower bounds',
                   'Static-affinity assignment; guarded schedule independent clock reference',
                   'BF16 input / FP32 output tiled GEMM edge and full-model K-length tests'],
        'limitations': ['No native PLENA cycle calibration', 'No silicon timing, area, power or energy measurement',
                        'Synthetic BF16 numerical tests are not end-to-end pretrained-model validation',
                        'Multilane E logical-queue extension has no RTL implementation in this experiment'],
        'records_sha256': sha(output / 'records.jsonl'),
        'source_sha256': {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
    }
    (output / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.output.resolve()), indent=2))
