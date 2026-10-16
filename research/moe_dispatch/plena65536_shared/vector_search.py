"""Independent, complete-domain vectorised integer Shared partition search.

No fractional bounds or geometric pruning are used.  Each declared design is
evaluated at all seven batches, with the same row/column ownership contracts
as timing.partition.  This is an audit backend, not a new timing model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .geometry import (Core, enumerate_cores, enumerate_native_designs,
                       geometry_counts, homogeneous_cores,
                       iter_heterogeneous_pairs, single_cores)
from .timing import BATCHES, F, H, PROFILES, partition


def _ceil(x, divisor):
    return (x + divisor - 1) // divisor


def _step(x, width, slope, intercept):
    return np.where(x > 0, _ceil(x, width) * slope + intercept, 0)


def _cut(total, shape, width0, slope0, intercept0, width1, slope1, intercept1):
    """Vectorised exact min_x max(step0(x),step1(total-x)), integer x."""
    total = np.broadcast_to(np.asarray(total, dtype=np.int64), shape)
    lo = np.zeros(shape, dtype=np.int64)
    hi = total.copy()
    # Integer bisection needs no more than ceil(log2(max(total)+1)) steps.
    for _ in range(int(total.max()).bit_length()):
        mid = (lo + hi) // 2
        crossing = (_step(mid, width0, slope0, intercept0) >=
                    _step(total-mid, width1, slope1, intercept1))
        active = lo < hi
        hi = np.where(active & crossing, mid, hi)
        lo = np.where(active & ~crossing, mid + 1, lo)
    assert np.all(lo == hi)
    best = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
    best_cut = total.copy()
    for cut in (np.zeros(shape, dtype=np.int64), total, lo, np.maximum(0, lo - 1)):
        cost = np.maximum(_step(cut, width0, slope0, intercept0),
                          _step(total-cut, width1, slope1, intercept1))
        take = (cost < best) | ((cost == best) & (cut < best_cut))
        best = np.where(take, cost, best)
        best_cut = np.where(take, cut, best_cut)
    return best, best_cut


def exact_pairs(cores, pair_ids, profile=PROFILES[0], *, batches=BATCHES):
    """Exact per-batch compute cycles and cuts for an N-by-2 core-id list."""
    ids = np.asarray(pair_ids, dtype=np.int32)
    if ids.ndim != 2 or ids.shape[1] != 2 or np.any(ids < 0):
        raise ValueError('exact_pairs requires nonnegative N-by-2 indices')
    shape = (len(ids), len(batches))
    sizes = np.asarray(batches, dtype=np.int64)[None, :]
    fields = []
    for side in (0, 1):
        selected = [cores[int(i)] for i in ids[:, side]]
        b = np.asarray([c.b for c in selected], dtype=np.int64)[:, None]
        pm = np.asarray([c.pm for c in selected], dtype=np.int64)[:, None]
        pn = np.asarray([c.pn for c in selected], dtype=np.int64)[:, None]
        reduction = np.asarray([c.r for c in selected], dtype=np.int64)[:, None]
        drain = np.asarray([profile.drain(c) for c in selected], dtype=np.int64)[:, None]
        feed_h, feed_f = _ceil(H, reduction)*b, _ceil(F, reduction)*b
        intercept = np.zeros_like(drain) if profile.serialized else drain
        if profile.serialized:
            feed_h, feed_f = feed_h + drain, feed_f + drain
        cm = _ceil(sizes, pm)
        row_slope = 2*_ceil(F, pn)*feed_h + _ceil(H, pn)*feed_f
        fields.append((pm, pn, row_slope, 2*intercept,
                       2*cm*feed_h, cm*feed_f, intercept))
    a, b = fields
    row, row_cut = _cut(sizes, shape, a[0], a[2], a[3], b[0], b[2], b[3])
    gu, gu_cut = _cut(F, shape, a[1], a[4], a[6], b[1], b[4], b[6])
    down, down_cut = _cut(H, shape, a[1], a[5], a[6], b[1], b[5], b[6])
    columns = gu + down
    # Scalar timing.partition uses lexical 'columns' before 'rows' on ties.
    rows_selected = row < columns
    return {'cycles': np.minimum(row, columns), 'rows': rows_selected,
            'row_split': row_cut, 'gu_split': gu_cut, 'down_split': down_cut}


def _config(cores):
    return ' + '.join(f'{c.pm}x{c.pn}x{c.groups}[b{c.b},R{c.r}]' for c in cores)


def _selection(cores, ids, profile, score):
    got = tuple(cores[int(i)] for i in ids if i >= 0)
    return {'indices': [int(i) for i in ids], 'cores': [c.to_dict() for c in got],
            'geometry': _config(got), 'GM_cycles': float(score),
            'plans': [partition(got, batch, profile) for batch in BATCHES]}


def evaluate_domain(cores, ids, chunk=8192):
    ids = np.asarray(ids, dtype=np.int32)
    scores = np.empty((len(PROFILES), len(ids)), dtype=np.float64)
    single = np.all(ids[:, 1] == -1)
    for pi, profile in enumerate(PROFILES):
        for offset in range(0, len(ids), chunk):
            picked = ids[offset:offset+chunk]
            if single:
                cycles = np.asarray([[partition((cores[int(i)],), batch, profile)['cycles']
                                      for batch in BATCHES] for i in picked[:, 0]], dtype=np.int64)
            else:
                cycles = exact_pairs(cores, picked, profile)['cycles']
            scores[pi, offset:offset+len(picked)] = np.exp(np.mean(np.log(cycles), axis=1))
    selections = {}
    optimum = scores.min(axis=1)
    for pi, profile in enumerate(PROFILES):
        minimum = float(optimum[pi])
        # Same deterministic absolute tolerance as scalar solve_domain.
        ix = int(np.flatnonzero(scores[pi] <= minimum + 1e-8)[0])
        selections[profile.name] = _selection(cores, ids[ix], profile, scores[pi, ix])
    regret = np.max(scores / optimum[:, None], axis=0)
    robust = int(np.flatnonzero(regret <= regret.min() + 1e-10)[0])
    got = tuple(cores[int(i)] for i in ids[robust] if i >= 0)
    selections['timing_robust'] = {
        'indices': ids[robust].tolist(), 'cores': [c.to_dict() for c in got],
        'geometry': _config(got), 'max_regret': float(regret[robust] - 1),
        'profile_GM_cycles': dict(zip([p.name for p in PROFILES], scores[:, robust].tolist())),
        'plans': [partition(got, batch, PROFILES[0]) for batch in BATCHES]}
    return selections, scores


def main(output: Path, chunk: int = 8192):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    cores = enumerate_cores()
    index = {core: i for i, core in enumerate(cores)}
    ids = np.asarray([(index[a], index[b]) for a, b in iter_heterogeneous_pairs()], dtype=np.int32)
    domains = {
        'extended_single': np.asarray([(index[c], -1) for c in single_cores()], dtype=np.int32),
        'extended_homogeneous': np.asarray([(index[c], index[c]) for c in homogeneous_cores()], dtype=np.int32),
        'extended_heterogeneous': ids,
    }
    for family, designs in enumerate_native_designs().items():
        values = [(c,) if isinstance(c, Core) else tuple(c) for c in designs]
        domains['native_'+family] = np.asarray([(index[cs[0]], index[cs[1]] if len(cs)>1 else -1)
                                               for cs in values], dtype=np.int32)
    all_results, audit_rows = {}, []
    for name, domain in domains.items():
        local_started = time.monotonic()
        selections, scores = evaluate_domain(cores, domain, chunk)
        all_results[name] = selections
        np.savez_compressed(output/(name+'_all_exact_scores.npz'),
                            pair_ids=domain, profile_GM_cycles=scores)
        for pi, profile in enumerate(PROFILES):
            audit_rows.append({'domain': name, 'profile': profile.name,
                               'evaluated_geometries': len(domain),
                               'batch_points': len(domain)*len(BATCHES),
                               'best_GM_cycles': float(scores[pi].min()),
                               'winner': selections[profile.name]['geometry']})
        print(name, 'completed', round(time.monotonic()-local_started, 2),
              selections[PROFILES[0].name]['geometry'], flush=True)
    write = lambda name, value: (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    write('FROZEN_SELECTION.json', all_results)
    write('VECTOR_AUDIT.json', {'status': 'PASS', 'method': 'all geometries exact, no pruning',
                              'counts': geometry_counts(), 'results': audit_rows,
                              'elapsed_seconds': time.monotonic()-started,
                              'profile_names': [p.name for p in PROFILES],
                              'batch_sizes': list(BATCHES), 'chunk': chunk,
                              'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in [Path(__file__), Path(__file__).with_name('timing.py'),
                                                          Path(__file__).with_name('geometry.py')]}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=8192)
    options = parser.parse_args()
    main(options.out, options.chunk)
