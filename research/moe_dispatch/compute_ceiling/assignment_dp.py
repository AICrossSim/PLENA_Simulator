"""Deterministic exact two-machine assignment with integer grouped tasks.

The dense DP is a 0/1 knapsack over gcd-scaled machine-0 load.  For each
exact load it keeps the largest removed machine-1 load.  Binary chunks
represent every integer count from zero through the supplied group count.
Only states that can improve a feasible integer makespan are retained.

This is an exact pseudo-polynomial solver, not an approximation.  Explicit
resource/overflow failures must not be interpreted as certified solutions.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import gcd, prod
from operator import index

import numpy as np


MAX_DENSE_STATES = 10_000_000
MAX_ENUMERATION_POINTS = 100_000
INT64_MAX = int(np.iinfo(np.int64).max)


class ExactDPResourceError(RuntimeError):
    """Exact dense proof exceeds the declared memory budget."""


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _chunks(count):
    step = 1
    while count:
        take = min(step, count)
        yield take
        count -= take
        step *= 2


def _feasible_bound(a, b, counts, base0, base1):
    """Integer feasible assignments; the bound never uses a fractional job."""
    all0 = max(base0 + sum(v*n for v,n in zip(a,counts)), base1)
    all1 = max(base0, base1 + sum(v*n for v,n in zip(b,counts)))
    best = min(all0, all1)
    order = sorted(range(len(counts)),
                   key=lambda j: (-max(a[j], b[j]), Fraction(a[j], b[j]), j))
    l0, l1 = base0, base1
    small = sum(counts) <= MAX_ENUMERATION_POINTS
    for j in order:
        pieces = (1 for _ in range(counts[j])) if small else _chunks(counts[j])
        for n in pieces:
            left, right = max(l0 + a[j]*n, l1), max(l0, l1 + b[j]*n)
            if (left, l0 + a[j]*n, 0) <= (right, l1 + b[j]*n, 1):
                l0 += a[j]*n
            else:
                l1 += b[j]*n
    return min(best, max(l0,l1))


def _enumerate(a, b, counts, base0, base1):
    """Small exact fallback for huge/coprime costs without dtype conversion."""
    best = None
    total1 = sum(v*n for v,n in zip(b,counts))
    for x in product(*(range(n+1) for n in counts)):
        l0 = base0 + sum(v*n for v,n in zip(a,x))
        l1 = base1 + total1 - sum(v*n for v,n in zip(b,x))
        key = max(l0,l1), l0, x
        if best is None or key < best:
            best = key
    return best[0], best[2]


def exact_two_machine(c0, c1, counts, base0=0, base1=0):
    """Return ``(optimal integer makespan, group counts assigned to core 0)``.

    Costs must be positive integers, counts/baseline loads nonnegative
    integers.  Group identity is retained, including groups with zero jobs.
    Dense ties keep the old state; final ties prefer the smaller core-0 load.
    Packed per-chunk decisions reconstruct a deterministic legal assignment.
    All byte/cost products are computed in Python integers before int64 use.
    """
    a = tuple(_integer(v,"c0",1) for v in c0)
    b = tuple(_integer(v,"c1",1) for v in c1)
    counts = tuple(_integer(v,"count") for v in counts)
    base0, base1 = _integer(base0,"base0"), _integer(base1,"base1")
    if len(a) != len(b) or len(a) != len(counts):
        raise ValueError("cost/count vectors must have equal length")
    if not any(counts):
        return max(base0,base1), (0,)*len(counts)

    g0 = g1 = 0
    for va,vb,n in zip(a,b,counts):
        if n:
            g0, g1 = gcd(g0,va), gcd(g1,vb)
    bound = _feasible_bound(a,b,counts,base0,base1)
    cap = (bound-base0)//g0
    assert cap >= 0
    # No useful state can exceed the sum of all machine-0 job loads.
    cap = min(cap, sum(v*n for v,n in zip(a,counts))//g0)
    total1 = sum(v*n for v,n in zip(b,counts))
    profit_total = total1//g1
    too_big = cap+1 > MAX_DENSE_STATES
    overflow = profit_total > INT64_MAX
    if too_big or overflow:
        if prod(n+1 for n in counts) <= MAX_ENUMERATION_POINTS:
            return _enumerate(a,b,counts,base0,base1)
        if overflow:
            raise OverflowError("gcd-scaled removed-load DP exceeds int64; no approximate result returned")
        raise ExactDPResourceError(
            f"exact assignment requires {cap+1:,} dense states; limit {MAX_DENSE_STATES:,}")

    values = np.full(cap+1,-1,dtype=np.int64)
    values[0] = 0
    trace = []
    for j,(va,vb,n) in enumerate(zip(a,b,counts)):
        for number in _chunks(n):
            weight, profit = va//g0*number, vb//g1*number
            if weight > cap:
                continue  # This whole chunk can only remain on machine 1.
            source = values[:-weight]
            # Fresh candidate values guarantee 0/1 rather than unbounded reuse.
            candidate = np.where(source >= 0,source+profit,-1)
            improved = candidate > values[weight:]
            decision = np.packbits(improved,bitorder="little")
            np.copyto(values[weight:],candidate,where=improved)
            trace.append((j,number,weight,decision))

    reachable = np.flatnonzero(values >= 0)
    if (base0 + cap*g0 <= INT64_MAX and base1 + total1 <= INT64_MAX):
        completion = np.maximum(base0 + reachable*g0,
                                base1 + total1 - values[reachable]*g1)
        best_index = int(np.argmin(completion))
        target = int(reachable[best_index])
        finish = int(completion[best_index])
    else:
        # The DP benefits still fit int64, but absolute times need Python ints.
        finish,target = min((max(base0+int(i)*g0,
                                 base1+total1-int(values[i])*g1),int(i))
                            for i in reachable)

    selected = [0]*len(counts)
    remaining = target
    for j,number,weight,decision in reversed(trace):
        offset = remaining-weight
        if offset >= 0 and (int(decision[offset>>3]) >> (offset&7)) & 1:
            selected[j] += number
            remaining -= weight
    assert remaining == 0, "packed DP trace must terminate at the empty state"
    assert all(0 <= x <= n for x,n in zip(selected,counts))
    actual0 = base0 + sum(v*x for v,x in zip(a,selected))
    actual1 = base1 + sum(v*(n-x) for v,n,x in zip(b,counts,selected))
    assert actual0 == base0+target*g0
    assert sum(v*x for v,x in zip(b,selected)) == int(values[target])*g1
    assert finish == max(actual0,actual1) and finish <= bound
    return finish, tuple(selected)
