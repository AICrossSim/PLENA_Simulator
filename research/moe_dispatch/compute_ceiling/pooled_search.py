"""Stronger compute-only DSE with cross-expert ready-record interleaving.

This is a separate sensitivity campaign, never an overwrite of the primary
whole-expert search. All geometries receive valid issue-load bounds. Exact
integer ownership minimizes issued work, then a deterministic ready-record
policy supplies a feasible cycle upper bound. Neither ownership nor the
ready policy proves optimal pooled instruction scheduling.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import math
from pathlib import Path
import time

import numpy as np

from .ready_pool import schedule_ready_pool
from .search import (BATCHES, BUDGET, FAMILIES, Shape, balanced, canonical,
                     csv_out, domain, dump, expert_cost, fractional_bounds,
                     gid, gm, grouped_window, optimal_counts, paired_ci,
                     timing, vector_schedule)


FINAL_DRAIN = 20  # dot20 + commit1 = 21; last issue at n-1 completes at n+20.


def issue_costs(shapes, signatures):
    """One PM*PN*PK dot issue, including real M/N/K padding, costs one II."""
    pm, pn, pk = (np.array([getattr(c, x) for c in shapes], dtype=np.int64)[:, None]
                  for x in ('pm', 'pn', 'pk'))
    m, h, f = (np.array([s[i] for s in signatures], dtype=np.int64)[None, :]
               for i in range(3))
    nm = (m + pm - 1) // pm
    return (2 * nm * ((f + pn - 1) // pn) * ((h + pk - 1) // pk)
            + nm * ((h + pn - 1) // pn) * ((f + pk - 1) // pk))


def exact_issue_assignment(window, cores):
    """Return exact issue-load makespan and its deterministic whole-job owners."""
    signatures, grouped = grouped_window(window)
    counts = tuple(len(grouped[s]) for s in signatures)
    costs = [tuple(expert_cost(*s, c, 'flat20')['issues'] for s in signatures)
             for c in cores]
    if len(cores) == 1:
        makespan, count0 = sum(a * n for a, n in zip(costs[0], counts)), counts
    else:
        makespan, count0 = optimal_counts(*costs, counts)
    owned = [[] for _ in cores]
    owner_rows = []
    for j, s in enumerate(signatures):
        for i, expert in enumerate(grouped[s]):
            owner = 0 if i < count0[j] else 1
            owned[owner].append(expert)
            owner_rows.append({'expert_id': expert['id'], 'Me': expert['Me'],
                               'is_shared': expert.get('is_shared', False),
                               'owner': owner, 'issues': costs[owner][j]})
    loads = [sum(expert_cost(e['Me'], e['H'], e['F'], c, 'flat20')['issues']
                 for e in jobs) for c, jobs in zip(cores, owned)]
    assert max(loads, default=0) == makespan
    return makespan, owned, owner_rows


def workload_key(experts):
    # Record order affects RR. Preserve the exact dimensional sequence after
    # the policy's ID-based tie break, while dropping irrelevant ID values.
    jobs = sorted(experts, key=lambda e: (-3*e['Me']*e['H']*e['F'], repr(e['id'])))
    return tuple((e['Me'], e['H'], e['F']) for e in jobs)


@lru_cache(maxsize=4096)
def cached_core_replay(key, core):
    # Deterministic lexical IDs preserve the dimensional order on equal-work
    # jobs. Only metrics are reused: cached expert identities are never output.
    experts = [dict(id=f'{i:08d}', Me=m, H=h, F=f) for i, (m, h, f) in enumerate(key)]
    result = schedule_ready_pool(experts, core)
    return {k: v for k, v in result.items() if k != 'expert_timelines'}


def pooled_layer(window, cores, *, cached=False):
    issue_floor, owned, ownership = exact_issue_assignment(window, cores)
    replays = [(cached_core_replay(workload_key(jobs), core) if cached else
                schedule_ready_pool(jobs, core)) for jobs, core in zip(owned, cores)]
    actual = max((r['cycles'] for r in replays), default=0)
    # A fair fallback is chosen independently on each core, with the same
    # issue-optimal owner assignment. This does not use the old cycle owners.
    core_cycles = [r['best_feasible_cycles'] for r in replays]
    cycles = max(core_cycles, default=0)
    lb = issue_floor + FINAL_DRAIN if issue_floor else 0
    assert cycles >= lb
    useful = sum(r['useful_macs'] for r in replays)
    issued = sum(r['issued_macs'] for r in replays)
    return {
        'window_id': window['id'], 'batch': window['batch'], 'geometry': gid(cores),
        'cycles': cycles, 'compute_ms': cycles / 1e6,
        'ready_pool_cycles': actual,
        'issue_assignment_floor_cycles': lb,
        'core_cycles': core_cycles,
        'core_issues': [r['issues'] for r in replays],
        'core_schedule_source': [r['best_feasible_source'] for r in replays],
        'issues': sum(r['issues'] for r in replays),
        'useful_macs': useful, 'issued_macs': issued, 'padding_macs': issued - useful,
        'spatial_utilization': useful / issued if issued else 1.,
        'wall_mac_utilization': useful / (BUDGET * cycles) if cycles else 1.,
        'peak_FP32_partial_bytes_per_core': [r['best_feasible_peak_partial_records'] * c.record_bytes
                                            for r, c in zip(replays, cores)],
        'ready_pool_peak_FP32_partial_bytes_per_core': [r['peak_live_partial_bytes'] for r in replays],
        'installed_record_descriptors_per_core': [r['installed_record_descriptors_if_precreated']
                                                   for r in replays],
        'ready_pool_issue_gap_cycles_per_core': [r['issue_gap_cycles_before_last_issue'] for r in replays],
        'finish_gap_cycles': max(core_cycles) - min(core_cycles) if len(cores) == 2 else 0,
        'ownership': ownership,
        'scope': 'feasible RR/whole-expert fallback with exact issue-load ownership; not optimal pooled scheduling',
    }


class PooledSearch:
    def __init__(self, windows, out, max_actual=8):
        self.windows, self.out, self.max_actual = windows, out, max_actual
        self.shapes, indices, family = domain()
        a, b = indices[:, 0], np.maximum(indices[:, 1], 0)
        aligned = np.array([c.pn % 4 == 0 for c in self.shapes])
        valid = aligned[a] & ((indices[:, 1] < 0) | aligned[b])
        self.indices, self.family = indices[valid], family[valid]
        self.signatures = sorted({(e['Me'], e['H'], e['F']) for w in windows
                                  for e in w['experts'] if e['Me']})
        self.cost = issue_costs(self.shapes, self.signatures)
        a, b = self.indices[:, 0], np.maximum(self.indices[:, 1], 0)
        lower = []
        for w in windows:
            counts = np.array([sum((e['Me'], e['H'], e['F']) == s for e in w['experts'])
                               for s in self.signatures])
            active = counts > 0
            if not active.any():
                lower.append(np.zeros(len(self.indices), dtype=np.int64))
                continue
            ca, cb = self.cost[a][:, active], self.cost[b][:, active]
            lb = fractional_bounds(ca, cb, counts[active])
            lb = np.where(self.indices[:, 1] < 0, (ca * counts[active]).sum(axis=1), lb)
            lower.append(np.ceil(lb - 1e-7).astype(np.int64) + FINAL_DRAIN)
        self.lb = np.stack(lower, axis=1)
        self.lower = balanced(self.lb, windows)
        self.eft_issue = vector_schedule(windows, self.signatures, self.cost, self.indices)
        self.eft_floor = self.eft_issue + np.where(self.eft_issue > 0, FINAL_DRAIN, 0)
        self.upper_issue_floor = balanced(self.eft_floor, windows)
        self.exact_floors, self.actual = {}, {}
        self.stats = []

    def cores(self, index):
        return tuple(self.shapes[j] for j in self.indices[index] if j >= 0)

    def exact_floor(self, index):
        if index not in self.exact_floors:
            values = np.array([exact_issue_assignment(w, self.cores(index))[0]
                               for w in self.windows], dtype=np.int64)
            values += np.where(values > 0, FINAL_DRAIN, 0)
            assert np.all(values >= self.lb[index])
            assert np.all(values <= self.eft_floor[index])
            self.exact_floors[index] = values
            self.upper_issue_floor[index] = balanced(values, self.windows)
        return self.exact_floors[index]

    def replay(self, index):
        if index not in self.actual:
            began = time.monotonic()
            results = [pooled_layer(w, self.cores(index), cached=True) for w in self.windows]
            values = np.array([r['cycles'] for r in results], dtype=np.int64)
            assert np.all(values >= self.exact_floor(index))
            self.actual[index] = (float(balanced(values, self.windows)), results)
            print(canonical({'event_replay': gid(self.cores(index)),
                             'family': FAMILIES[int(self.family[index])],
                             'score_cycles': self.actual[index][0],
                             'seconds': time.monotonic() - began}), flush=True)
        return self.actual[index][0]

    def explicit_seeds(self, family):
        if family == 0:
            candidates = [((1, 96, 128),)]
        elif family == 1:
            candidates = [((1, 128, 48), (1, 128, 48)),
                          ((1, 48, 128), (1, 48, 128))]
        else:
            big = [(1, 64, 128), (1, 128, 64)]
            small = [(1, 32, 128), (1, 64, 64), (1, 128, 32)]
            candidates = [tuple(sorted((a, b))) for a in big for b in small]
        lookup = {tuple((c.pm, c.pn, c.pk) for c in self.cores(int(i))): int(i)
                  for i in np.flatnonzero(self.family == family)}
        return [lookup[key] for key in candidates if key in lookup]

    def optimize(self):
        chosen = {}
        for fi, family in enumerate(FAMILIES):
            ids = np.flatnonzero(self.family == fi)
            seed = int(ids[np.argmin(self.upper_issue_floor[ids])])
            floor_best, floor_score = seed, float(balanced(self.exact_floor(seed), self.windows))
            floor_solves = 0
            # Certify only the issue-assignment floor: no instruction schedule
            # optimality is inferred from solving this integer assignment.
            for index in ids[np.argsort(self.lower[ids], kind='stable')]:
                index = int(index)
                if self.lower[index] > floor_score + 1e-8:
                    break
                score = float(balanced(self.exact_floor(index), self.windows))
                floor_solves += 1
                if (score, gid(self.cores(index))) < (floor_score, gid(self.cores(floor_best))):
                    floor_best, floor_score = index, score
            best, best_score = floor_best, self.replay(floor_best)
            explicit = sorted(set(self.explicit_seeds(fi)),
                              key=lambda i: (self.lower[i], gid(self.cores(i))))
            remaining = [int(i) for i in ids[np.argsort(self.lower[ids], kind='stable')]
                         if int(i) not in explicit and int(i) != floor_best]
            for index in explicit + remaining:
                if index in self.actual:
                    continue
                if self.lower[index] > best_score + 1e-8:
                    continue
                exact_lb = float(balanced(self.exact_floor(index), self.windows))
                if exact_lb > best_score + 1e-8:
                    continue
                count = sum(int(self.family[i]) == fi for i in self.actual)
                if count >= self.max_actual:
                    break
                score = self.replay(index)
                if (score, gid(self.cores(index))) < (best_score, gid(self.cores(best))):
                    best, best_score = index, score
            chosen[family] = best
            # The minimum of exact issue floors over the complete geometry
            # domain is a stronger valid lower bound on all pooled schedules
            # under the declared whole-expert ownership restriction.
            fractional_min = float(self.lower[ids].min())
            event_count = sum(int(self.family[i]) == fi for i in self.actual)
            stats = {'family': family, 'domain_geometries': len(ids),
                     'exact_issue_floor_points': floor_solves,
                     'issue_floor_best_geometry': gid(self.cores(floor_best)),
                     'family_min_fractional_LB_cycles': fractional_min,
                     'family_min_valid_LB_cycles': floor_score,
                     'issue_assignment_floor_domain_certified': True,
                     'actual_event_geometries': event_count,
                     'actual_event_budget': self.max_actual,
                     'best_feasible_geometry': gid(self.cores(best)),
                     'best_feasible_score_cycles': best_score,
                     'certified_relative_gap': best_score / floor_score - 1,
                     'gap_against_fractional_min': best_score / fractional_min - 1,
                     'pooled_schedule_global_optimality_claimed': False,
                     'all_domain_lower_bounds_reported': True}
            self.stats.append(stats)
            print(canonical(stats), flush=True)
        return chosen


def summarize(label, rows, windows):
    result = []
    for batch in list(BATCHES) + ['all']:
        rr = [r for r in rows if batch == 'all' or r['batch'] == batch]
        ww = [w for w in windows if batch == 'all' or w['batch'] == batch]
        values = np.array([r['cycles'] for r in rr])
        if not rr:
            continue
        value = balanced(values, ww) if batch == 'all' else gm(values)
        result.append({'label': label, 'geometry': rr[0]['geometry'], 'batch': batch,
                       'windows': len(rr), 'cycles_geomean': float(value),
                       'compute_ms_geomean': float(value / 1e6),
                       'p95_ms': float(np.quantile(values, .95, method='higher') / 1e6),
                       'spatial_utilization': sum(r['useful_macs'] for r in rr) /
                                              sum(r['issued_macs'] for r in rr),
                       'wall_mac_utilization': sum(r['useful_macs'] for r in rr) /
                                               (BUDGET * sum(values)),
                       'FP32_partial_peak_KiB': max(sum(r['peak_FP32_partial_bytes_per_core'])
                                                   for r in rr) / 1024,
                       'ready_pool_FP32_partial_peak_KiB': max(
                           sum(r['ready_pool_peak_FP32_partial_bytes_per_core']) for r in rr) / 1024,
                       'issue_floor_cycles_geomean': float(balanced(
                           np.array([r['issue_assignment_floor_cycles'] for r in rr]), ww)),
                       'finish_gap_cycles_mean': float(np.mean([r['finish_gap_cycles'] for r in rr]))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-actual-per-family', type=int, default=8)
    args = parser.parse_args()
    if args.max_actual_per_family < 1:
        parser.error('--max-actual-per-family must be >= 1')
    args.out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    from .trace_inputs import load_primary_inputs
    inputs = load_primary_inputs(split='development')
    dev = inputs['development']
    assert len(dev) == 18
    contract = {
        'budget_MACs': BUDGET, 'precision': 'BF16 products / FP32 partials',
        'timing': 'flat20: dot20+commit1=21 latency, II1, assumed 1GHz',
        'geometry_domain': 'complete current domain: positive M; N multiple4; K16..2048 multiple16; sum MACs=12288',
        'memory': 'all operands resident; unlimited pooled contexts and independent output records',
        'ownership': 'whole expert stays on one core; exact integer assignment minimizes issue-load makespan',
        'schedule': 'RR ready records across experts; per-output nextK waits own commit; own expert GU barrier before Down',
        'fallback': 'same-owner serial whole-expert schedule allowed independently per core',
        'lower_bound': 'full-domain fractional issue assignment +20; certify minimum exact integer issue floor',
        'upper_bound': 'actual feasible event replay; not an optimal pooled schedule',
        'excluded': ['HBM', 'SRAM/NoC ports', 'DMA', 'dispatch/context-switch cost', 'SiLU', 'combine'],
        'selection': '18 development windows only, equal batch-weight log mean; freeze before heldout load',
        'event_budget_per_family': args.max_actual_per_family,
        'native_calibrated': False, 'main_frozen_results_modified': False,
        'cross_expert_row_copacking': False, 'shared_column_splitting': False,
        'manifest': inputs['manifest'],
    }
    dump(args.out / 'CONTRACT.json', contract)
    search = PooledSearch(dev, args.out, args.max_actual_per_family)
    chosen = search.optimize()
    selected = {name: search.cores(index) for name, index in chosen.items()}
    dump(args.out / 'FROZEN_SELECTION.json', {
        'configs': {name: [{'pm': c.pm, 'pn': c.pn, 'pk': c.pk} for c in cores]
                    for name, cores in selected.items()},
        'frozen_before_heldout_load': True,
        'selection_is_feasible_upper_not_pooled_schedule_optimum': True,
        'development_certificates': search.stats,
    })
    csv_out(args.out / 'development_certificates.csv', search.stats)
    csv_out(args.out / 'development_all_geometry_bounds.csv', (
        {'geometry': gid(search.cores(i)), 'family': FAMILIES[int(search.family[i])],
         'fractional_LB_cycles': float(search.lower[i]),
         'exact_issue_floor_cycles': float(balanced(search.exact_floors[i], dev))
                                     if i in search.exact_floors else '',
         'event_replayed': i in search.actual,
         'event_feasible_upper_cycles': search.actual[i][0] if i in search.actual else '',
         'within_selected_upper_by_fractional_LB': search.lower[i] <= search.actual[chosen[
             FAMILIES[int(search.family[i])]]][0] + 1e-8,
         'pool_optimality_claimed': False} for i in range(len(search.indices))))
    csv_out(args.out / 'development_near_optimal_bounds.csv', (
        {'geometry': gid(search.cores(i)), 'family': FAMILIES[int(search.family[i])],
         'valid_fractional_LB_cycles': float(search.lower[i]),
         'exact_issue_floor_cycles': float(balanced(search.exact_floors[i], dev))
                                     if i in search.exact_floors else '',
         'best_family_feasible_upper_cycles': search.actual[chosen[FAMILIES[int(search.family[i])]]][0],
         'actual_event_replayed': i in search.actual}
        for i in range(len(search.indices)) if search.lower[i] <= search.actual[chosen[
            FAMILIES[int(search.family[i])]]][0] + 1e-8))
    for family, index in chosen.items():
        dump(args.out / f'development_{family}.json', search.actual[index][1])
    # Only after writing the frozen development selection is heldout loaded.
    held = load_primary_inputs(split='heldout')['heldout']
    assert len(held) == 135
    named, table, window_rows = {}, [], []
    for label, cores in selected.items():
        label_start = time.monotonic()
        results = [pooled_layer(w, cores, cached=False) for w in held]
        optimal_counts.cache_clear()
        # Re-run the actual events, never replace this repeat with cache hits.
        again = [pooled_layer(w, cores, cached=False) for w in held]
        assert canonical(results) == canonical(again)
        named[label] = results
        dump(args.out / f'heldout_{label}.json', results)
        table.extend(summarize(label, results, held))
        for row in results:
            window_rows.append({
                'label': label, 'geometry': row['geometry'], 'window_id': row['window_id'],
                'batch': row['batch'], 'compute_ms': row['compute_ms'],
                'cycles': row['cycles'], 'ready_pool_cycles': row['ready_pool_cycles'],
                'issue_assignment_floor_cycles': row['issue_assignment_floor_cycles'],
                'issues': row['issues'], 'useful_macs': row['useful_macs'],
                'padding_macs': row['padding_macs'],
                'partial_peak_KiB': sum(row['peak_FP32_partial_bytes_per_core']) / 1024,
                'ready_pool_partial_peak_KiB': sum(row['ready_pool_peak_FP32_partial_bytes_per_core']) / 1024,
                'core_partial_peak_KiB': canonical([x / 1024 for x in row['peak_FP32_partial_bytes_per_core']]),
                'core_cycles': canonical(row['core_cycles']),
                'core_issues': canonical(row['core_issues']),
                'core_schedule_source': canonical(row['core_schedule_source']),
                'repeat_exactly_equal': True})
        print(canonical({'heldout_complete': label, 'geometry': gid(cores),
                         'windows': len(held), 'event_repeats': 2,
                         'seconds': time.monotonic() - label_start}), flush=True)
    csv_out(args.out / 'compute_by_batch.csv', table)
    csv_out(args.out / 'heldout_per_window.csv', window_rows)
    comparisons = []
    for candidate, baseline in [('homogeneous', 'single'), ('heterogeneous', 'single'),
                                ('heterogeneous', 'homogeneous')]:
        for batch in list(BATCHES) + ['all']:
            indexes = [i for i, w in enumerate(held) if batch == 'all' or w['batch'] == batch]
            windows = [held[i] for i in indexes]
            c = np.array([named[candidate][i]['cycles'] for i in indexes])
            b = np.array([named[baseline][i]['cycles'] for i in indexes])
            speed = float(balanced(b / c, windows))
            low, high = paired_ci(c, b, windows)
            comparisons.append({'candidate': candidate, 'baseline': baseline,
                                'batch': batch, 'speedup': speed,
                                'latency_reduction_pct': (1 - 1 / speed) * 100,
                                'speedup_CI95_low': low, 'speedup_CI95_high': high,
                                'native_calibrated': False})
    csv_out(args.out / 'comparisons.csv', comparisons)
    certificates = '\n'.join(
        f"| {s['family']} | {s['best_feasible_geometry']} | {s['best_feasible_score_cycles']:.3f} | "
        f"{s['family_min_valid_LB_cycles']:.3f} | {100*s['certified_relative_gap']:.4f}% | "
        f"{s['actual_event_geometries']} / {s['domain_geometries']} |" for s in search.stats)
    (args.out / 'SUMMARY.md').write_text(
        '# Cross-expert pooled-context compute sensitivity\n\n'
        'This separate campaign permits independent expert output records to interleave on a core. '
        'It assumes unlimited resident operands and partial contexts, II=1 and completion latency21. '
        'No same-issue row co-packing, Shared-column split, memory timing or context-switch cost is included.\n\n'
        '| Family | Frozen geometry (M×N×K) | Development feasible upper cycles | Valid family lower cycles | Certified relative gap | Event points / full domain |\n'
        '|---|---|---:|---:|---:|---:|\n' + certificates + '\n\n'
        'The full geometry domain received valid fractional issue-load bounds. The minimum exact '
        'integer issue-assignment floor was certified by bound pruning; it remains a lower bound '
        'on pooled execution. Event replay gives feasible uppers, not globally optimal instruction schedules. '
        'The gap upper/lower−1 bounds how much further the development objective could improve within '
        'this geometry domain and whole-expert ownership contract.\n\n'
        'Development selection was frozen before loading 135 heldout windows. Every frozen heldout '
        'point ran two full event replays and matched exactly. compute_by_batch.csv reports B2/B4/B8/B16/'
        'B64/B96/B128 and an equal-batch-weight geometric mean; heldout_per_window.csv reports actual '
        'partial-storage peaks. These peaks are requirements, not evidence that existing SRAM suffices.\n\n'
        'Do not promote this sensitivity into a physical speedup claim: arithmetic II, clock, ports, '
        'NoC, capacity, SiLU and dispatch still require implementation/calibration.\n')
    dump(args.out / 'COMPLETION.json', {
        'elapsed_seconds': time.monotonic() - began, 'development_windows': len(dev),
        'heldout_windows': len(held), 'heldout_actual_event_repeats': 2,
        'repeat_exactly_equal': True, 'domain_geometries': len(search.indices),
        'frozen_before_heldout_load': True, 'main_results_unchanged': True,
        'actual_schedule_optimality_claimed': False, 'native_calibrated': False,
        'cache_used_only_for_development_event_replays': True})
    print(canonical({'complete': str(args.out), 'seconds': time.monotonic() - began}), flush=True)


if __name__ == '__main__':
    main()
