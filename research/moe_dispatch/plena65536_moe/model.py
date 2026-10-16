"""Complete Shared+routed expert queues under the declared PLENA timing.

The primary contract assigns entire three-GEMM experts to a core.  Exact
integer two-machine assignment optimises the whole window, not mean task
latency.  All weights/activations are ideally available; no HBM, SRAM-port,
router, vector-SiLU or dispatch latency is timed.  This is a compute ceiling
within a stated ownership contract, not a full inference benchmark.
"""
from __future__ import annotations

from collections import defaultdict
from math import exp, log
import numpy as np

from research.moe_dispatch.plena65536_shared.geometry import Core, BUDGET
from research.moe_dispatch.plena65536_shared.timing import phase, PROFILES
from research.moe_dispatch.compute_ceiling.assignment_dp import exact_two_machine


def expert_key(expert):
    return int(expert['Me']), int(expert['H']), int(expert['F'])


def job_cycles(core, expert, profile=PROFILES[0]):
    m, h, f = expert_key(expert)
    return phase(core, m, f, h, 2, profile) + phase(core, m, h, f, 1, profile)


def useful_macs(window):
    return sum(3*int(e['Me'])*int(e['H'])*int(e['F']) for e in window['experts'])


def unique_weight_bytes(window):
    return sum(6*int(e['H'])*int(e['F']) for e in window['experts'])


def group_window(window):
    groups = defaultdict(list)
    for e in window['experts']:
        groups[expert_key(e)].append(e)
    return [(key, sorted(es, key=lambda e: int(e['id']))) for key, es in sorted(groups.items())]


def schedule(cores, window, profile=PROFILES[0]):
    """Exact whole-expert makespan and deterministic legal serial queues."""
    cores = tuple(cores)
    if len(cores) not in (1, 2) or sum(c.macs for c in cores) != BUDGET:
        raise ValueError('One/two cores must total exactly 65,536 multipliers')
    grouped = group_window(window)
    prototypes = [es[0] for _, es in grouped]
    counts = [len(es) for _, es in grouped]
    costs = [[job_cycles(c, e, profile) for e in prototypes] for c in cores]
    if len(cores) == 1:
        chosen = counts
        optimum = sum(a*n for a, n in zip(costs[0], counts))
    else:
        optimum, chosen = exact_two_machine(costs[0], costs[1], counts)
    queues = [[] for _ in cores]
    for j, (_, experts) in enumerate(grouped):
        for offset, expert in enumerate(experts):
            owner = 0 if offset < chosen[j] else 1
            queues[owner].append(expert)
    jobs = []
    loads = []
    for owner, queue in enumerate(queues):
        # Queue order cannot change a serial ideal-compute makespan; LPT is
        # deterministic and makes the large-task placement easy to inspect.
        queue.sort(key=lambda e: (-job_cycles(cores[owner], e, profile), int(e['id'])))
        now = 0
        for e in queue:
            duration = job_cycles(cores[owner], e, profile)
            jobs.append({'expert_id': int(e['id']), 'is_shared': bool(e.get('is_shared', False)),
                         'core_id': owner, 'M': int(e['Me']), 'H': int(e['H']), 'F': int(e['F']),
                         'start': now, 'end': now+duration, 'cycles': duration})
            now += duration
        loads.append(now)
    assert max(loads) == optimum
    assert sorted(j['expert_id'] for j in jobs) == sorted(int(e['id']) for e in window['experts'])
    return {'cycles': int(optimum), 'jobs': jobs, 'core_cycles': loads,
            'contract': 'whole_expert_exact', 'profile': profile.name,
            'useful_MACs': useful_macs(window), 'unique_weight_bytes': unique_weight_bytes(window)}


def balanced_score(cycles, windows):
    """Equal weight to each batch; equal log weight to windows within it."""
    groups = defaultdict(list)
    for value, window in zip(cycles, windows):
        groups[int(window['batch'])].append(log(float(value)))
    return exp(sum(sum(v)/len(v) for v in groups.values()) / len(groups))


def cost_table(cores, expert_types, profile=PROFILES[0]):
    """All geometry-by-type costs, identical to scalar job_cycles."""
    b = np.asarray([c.b for c in cores], dtype=np.int64)[:, None]
    pm = np.asarray([c.pm for c in cores], dtype=np.int64)[:, None]
    pn = np.asarray([c.pn for c in cores], dtype=np.int64)[:, None]
    r = np.asarray([c.r for c in cores], dtype=np.int64)[:, None]
    d = np.asarray([profile.drain(c) for c in cores], dtype=np.int64)[:, None]
    types = np.asarray(expert_types, dtype=np.int64)
    m, h, f = (types[:, i][None, :] for i in range(3))
    ceil = lambda a, z: (a+z-1)//z
    gu = 2*ceil(m, pm)*ceil(f, pn)
    down = ceil(m, pm)*ceil(h, pn)
    if profile.serialized:
        return gu*(ceil(h, r)*b+d)+down*(ceil(f, r)*b+d)
    return gu*ceil(h, r)*b+down*ceil(f, r)*b+2*d


def window_histograms(windows):
    types = sorted({expert_key(e) for w in windows for e in w['experts']})
    index = {key: i for i, key in enumerate(types)}
    counts = np.zeros((len(windows), len(types)), dtype=np.int64)
    weights = np.zeros(len(windows), dtype=np.float64)
    batch_counts = defaultdict(int)
    for w in windows:
        batch_counts[int(w['batch'])] += 1
    for wi, w in enumerate(windows):
        for e in w['experts']:
            counts[wi, index[expert_key(e)]] += 1
        weights[wi] = 1 / (len(batch_counts)*batch_counts[int(w['batch'])])
    return types, counts, weights


def fractional_pair_bound(costs, ids, histograms):
    """Certified dual lower bounds for divisible whole-job ownership.

    A floating-point sort identifies a useful dual multiplier; correctness
    never depends on that sort.  For every t in [0,1], an integer assignment
    has makespan >= sum_j n_j*min(t*a_j,(1-t)*b_j).  At the frontier's crossing
    type, t=b/(a+b).  The dual numerator is computed with exact int64 products
    and sums.  A checked conservative bound excludes every possible overflow.
    Conversion/division receive a relative downward guard of 8 machine eps,
    followed by a downward ULP.  Unsafe input raises instead of certifying
    floating-point arithmetic or pruning from an unproved approximation.
    """
    costs = np.asarray(costs)
    ids = np.asarray(ids, dtype=np.int32)
    histograms = np.asarray(histograms)
    if costs.ndim != 2 or histograms.ndim != 2 or costs.shape[1] != histograms.shape[1]:
        raise ValueError('cost and histogram matrices must have equal type dimensions')
    if costs.dtype.kind not in 'iu' or histograms.dtype.kind not in 'iu':
        raise ValueError('certified LP requires integer costs and counts')
    if ids.ndim != 2 or ids.shape[1] != 2:
        raise ValueError('pair IDs must be an N-by-2 matrix')
    if np.any(histograms < 0):
        raise ValueError('histogram counts must be nonnegative')
    answer = np.zeros((len(ids), len(histograms)), dtype=np.float64)
    if not len(ids) or not len(histograms):
        return answer
    a, b = costs[ids[:, 0]], costs[ids[:, 1]]
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError('whole-job costs must be positive')
    # Python integers calculate the proof bound before any int64 product.
    maximum_jobs = max(sum(map(int, hist)) for hist in histograms)
    if not maximum_jobs:
        return answer
    maximum_cost = max(int(a.max()), int(b.max()))
    integer_limit = int(np.iinfo(np.int64).max)
    if (maximum_cost * maximum_cost * maximum_jobs > integer_limit
            or 2 * maximum_cost > 2**53):
        raise OverflowError('certified LP dual exceeds exact int64/float denominator contract; no pruning bound returned')
    a, b = a.astype(np.int64, copy=False), b.astype(np.int64, copy=False)
    order = np.argsort(a/b, axis=1, kind='stable')
    a = np.take_along_axis(a, order, axis=1)
    b = np.take_along_axis(b, order, axis=1)
    for wi, hist in enumerate(histograms):
        n = hist[order].astype(np.int64, copy=False)
        ca = np.cumsum(a*n, axis=1)
        cb = np.sum(b*n, axis=1)[:, None]-np.cumsum(b*n, axis=1)
        ix = np.argmax(ca >= cb, axis=1)
        row = np.arange(len(ids))
        av, bv = a[row, ix], b[row, ix]
        numerator = np.sum(np.minimum(bv[:, None]*a, av[:, None]*b)*n,
                           axis=1, dtype=np.int64)
        quotient = numerator.astype(np.float64)/(av+bv)
        guarded = quotient*(1-8*np.finfo(np.float64).eps)
        answer[:, wi] = np.maximum(0, np.nextafter(guarded, -np.inf))
    return answer
