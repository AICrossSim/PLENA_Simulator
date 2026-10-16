"""Development-only exhaustive geometry DSE with certified assignment pruning.

Every geometry receives a fractional lower bound. Only a lower bound that
cannot beat an actually scheduled incumbent is pruned; all competitive
designs receive exact integer assignment per window. The certificate is
scoped to whole-expert ownership and the explicit finite replication domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from research.moe_dispatch.plena65536_shared.geometry import (Core, enumerate_cores, single_cores,
    homogeneous_cores, iter_heterogeneous_pairs, enumerate_native_designs,
    geometry_counts)
from research.moe_dispatch.plena65536_shared.timing import PROFILES, resource_ledger
from research.moe_dispatch.compute_ceiling.assignment_dp import exact_two_machine
from .model import window_histograms, cost_table, fractional_pair_bound
from .workloads import load_workloads


def core_from_dict(value):
    return Core(int(value['b']), int(value['r']), int(value['rm']), int(value['rn']))


def geometry_label(cores):
    return ' + '.join(f'{c.pm}x{c.pn}x{c.groups}[BLEN={c.b},MLEN={c.r}]' for c in cores)


def selection(cores, score, cycles, profile, certificate=None):
    return {'label': geometry_label(cores), 'cores': [c.to_dict() for c in cores],
            'development_GM_cycles': float(score), 'development_window_cycles': list(map(int, cycles)),
            'profile': profile.name, 'resource_ledger': resource_ledger(cores),
            'certificate': certificate}


def evaluate_family(cores, ids, costs, hist, weights, profile, *, chunk=4096):
    ids = np.asarray(ids, dtype=np.int32)
    lower = np.empty(len(ids), dtype=np.float64)
    single = bool(np.all(ids[:, 1] == -1))
    if single:
        window_cycles = costs[ids[:, 0]] @ hist.T
        scores = np.exp(np.log(window_cycles) @ weights)
        best = int(np.argmin(scores))
        cert = {'status': 'exact', 'geometries': len(ids), 'exact_evaluated': len(ids),
                'pruned_by_admissible_LP': 0, 'ownership': 'entire_expert'}
        return selection((cores[int(ids[best, 0])],), scores[best], window_cycles[best], profile, cert), scores, scores
    for start in range(0, len(ids), chunk):
        bound = fractional_pair_bound(costs, ids[start:start+chunk], hist)
        lower[start:start+len(bound)] = np.exp(np.log(bound) @ weights)
    order = np.argsort(lower, kind='stable')
    incumbent = float('inf')
    best = None
    best_cycles = None
    exact_count = 0
    cache = {}
    exact_scores = np.full(len(ids), np.nan)
    nonzero = [np.flatnonzero(n) for n in hist]
    for ix in order:
        if lower[ix] > incumbent + 1e-7:
            break
        first, second = map(int, ids[ix])
        a, b = costs[first], costs[second]
        key = a.tobytes()+b.tobytes()
        if key in cache:
            cycles = cache[key]
        else:
            cycles = np.asarray([exact_two_machine(a[nz].tolist(), b[nz].tolist(), hist[wi,nz].tolist())[0]
                                 for wi,nz in enumerate(nonzero)], dtype=np.int64)
            cache[key] = cycles
        score = float(np.exp(np.log(cycles) @ weights))
        exact_scores[ix] = score
        exact_count += 1
        if (score, int(ix)) < (incumbent, best if best is not None else len(ids)):
            incumbent, best, best_cycles = score, int(ix), cycles
    assert best is not None
    pruned = len(ids)-exact_count
    finite_unchecked = lower[np.isnan(exact_scores)]
    certificate = {'status': 'exact_with_admissible_pruning', 'geometries': len(ids),
                   'exact_evaluated': exact_count, 'distinct_exact_cost_signatures': len(cache),
                   'pruned_by_admissible_LP': pruned,
                   'minimum_pruned_lower_bound': float(finite_unchecked.min()) if pruned else None,
                   'winner_score': incumbent, 'ownership': 'entire_expert',
                   'does_not_prove_optimal_arbitrary_output_or_token_splitting': True}
    return selection(tuple(cores[int(i)] for i in ids[best]), incumbent, best_cycles, profile, certificate), lower, exact_scores


def main(output, *, chunk=4096):
    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    windows = load_workloads('development')['development']
    types, hist, weights = window_histograms(windows)
    cores = enumerate_cores()
    index = {c: i for i,c in enumerate(cores)}
    domains = {
        'extended_single': np.asarray([(index[c],-1) for c in single_cores()], dtype=np.int32),
        'extended_homogeneous': np.asarray([(index[c],index[c]) for c in homogeneous_cores()], dtype=np.int32),
        'extended_heterogeneous': np.asarray([(index[a],index[b]) for a,b in iter_heterogeneous_pairs()], dtype=np.int32)}
    for family, configs in enumerate_native_designs().items():
        configs = [(c,) if isinstance(c,Core) else c for c in configs]
        domains['native_'+family] = np.asarray([(index[c[0]],index[c[1]] if len(c)>1 else -1) for c in configs], dtype=np.int32)
    original = Core(32,2048)
    frozen = {'budget': 65_536, 'contract': 'whole_expert_exact', 'scope': 'post-router Shared+all active routed experts; ideal operands',
              'profile_default': PROFILES[0].name, 'development_window_ids': [w['id'] for w in windows],
              'geometry_domain': geometry_counts(), 'profile_optima': {},
              'precision': 'BF16 inputs/weights, FP32 accumulation: precision variant, not PLENA paper main MX4 precision'}
    for profile in PROFILES:
        local = time.monotonic()
        costs = cost_table(cores,types,profile)
        chosen = {}
        for family, ids in domains.items():
            begin = time.monotonic()
            best, bounds, exact_scores = evaluate_family(cores,ids,costs,hist,weights,profile,chunk=chunk)
            chosen[family] = best
            np.savez_compressed(output/f'{family}_{profile.name}_search.npz', pair_ids=ids,
                                lower_bound_GM_cycles=bounds, exact_GM_cycles=exact_scores)
            print(profile.name,family,best['label'],f"GM={best['development_GM_cycles']:.6f}",
                  best['certificate'],f'{time.monotonic()-begin:.1f}s',flush=True)
        orig_cycles = costs[index[original]] @ hist.T
        chosen['original'] = selection((original,), np.exp(np.log(orig_cycles)@weights),orig_cycles,profile,
                                      {'status':'fixed_paper_geometry'})
        frozen['profile_optima'][profile.name] = chosen
        (output/'SEARCH_PROGRESS.json').write_text(json.dumps(frozen,indent=2)+'\n')
        print('profile complete',profile.name,f'{time.monotonic()-local:.1f}s',flush=True)
    frozen['families'] = frozen['profile_optima'][PROFILES[0].name]
    frozen['elapsed_seconds'] = time.monotonic()-started
    frozen['search_source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('model.py')]}
    (output/'FROZEN_SELECTION.json').write_text(json.dumps(frozen,indent=2)+'\n')
    digest = hashlib.sha256(json.dumps({k:frozen[k] for k in ('families','profile_optima','development_window_ids')},sort_keys=True).encode()).hexdigest()
    (output/'SEARCH_CERTIFICATE.json').write_text(json.dumps({'status':'PASS','selection_sha256':digest,'counts':geometry_counts(),
      'contract':'whole-expert exact integer assignment; arbitrary task splitting not certified',
      'profiles':{p:{f:v['certificate'] for f,v in cs.items()} for p,cs in frozen['profile_optima'].items()},
      'elapsed_seconds':frozen['elapsed_seconds']},indent=2)+'\n')
    print('SEARCH COMPLETE',digest,round(frozen['elapsed_seconds'],2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--chunk',type=int,default=4096)
    opts = parser.parse_args()
    main(opts.out,chunk=opts.chunk)
