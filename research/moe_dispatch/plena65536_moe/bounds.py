"""Exact rational lower bounds for the complete Shared plus routed window.

The fractional assignment here is a relaxation, never an executable plan.
The default also relaxes M padding, so the bound admits either token-row or
output-column slicing.  Preserving M padding is only valid when a task keeps
all its original token rows on every participating core.
"""
from __future__ import annotations

from fractions import Fraction
from math import nextafter
from operator import index


def fractional_two_machine_exact(cost0, cost1, counts=None):
    """Exact continuous two-machine load optimum, with divisible task groups.

    Increasing allocation to machine zero in ascending cost0/cost1 order is
    the fractional-knapsack frontier.  Its intersection with equal machine
    loads minimizes their maximum.  Costs and returned cycles are rational.
    """
    a = tuple(Fraction(value) for value in cost0)
    b = tuple(Fraction(value) for value in cost1)
    if len(a) != len(b) or any(value <= 0 for value in a + b):
        raise ValueError("equal length positive cost vectors required")
    if counts is None:
        counts = (1,) * len(a)
    else:
        counts = tuple(index(value) for value in counts)
    if len(counts) != len(a) or any(value < 0 for value in counts):
        raise ValueError("equal length nonnegative integer counts required")
    left = Fraction(0)
    right = sum((value * count for value, count in zip(b, counts)), Fraction(0))
    if not right:
        return Fraction(0)
    for j in sorted(range(len(a)), key=lambda j: (a[j] / b[j], j)):
        take = min(Fraction(counts[j]), (right - left) / (a[j] + b[j]))
        left += take * a[j]
        right -= take * b[j]
        if left == right:
            return left
    raise AssertionError("fractional assignment frontier must cross equal loads")


def fractional_two_machine(cost0, cost1, counts=None):
    """Downward-rounded float lower bound for divisible whole-job ownership.

    With complete expert costs, this bounds exact integer whole-job assignment.
    It does not bound schedules that split a job between cores, because their
    completion drains and tile padding may differ.  The rational computation
    precedes conversion; one ULP toward negative infinity ensures the reported
    float never rounds above the exact relaxation.  Empty input returns zero.
    """
    value = fractional_two_machine_exact(cost0, cost1, counts)
    return nextafter(float(value), float("-inf")) if value else 0.0


def _phase_feed(c, me, n, k, factor, profile, preserve_m_padding):
    rows = Fraction((me + c.pm - 1) // c.pm) if preserve_m_padding else Fraction(me, c.pm)
    feed = ((k + c.r - 1) // c.r) * c.b
    if profile is not None and profile.serialized:
        feed += profile.drain(c)
    return rows * Fraction(n, c.pn) * factor * feed


def full_window_feed_lower_bound(experts, cores, profile=None, *, preserve_m_padding=False):
    """Bound arbitrary window scheduling under the declared matrix pipeline.

    Every active Shared and routed expert contributes its two GU projections
    and dependent Down projection.  Per-phase N padding is relaxed, as are
    pipelined completion drains and integer ownership.  The isolated expert
    GU/Down barrier is retained in a critical-path lower bound.  No global GU
    barrier is imposed, so different experts may overlap phases freely.

    A serialized profile prices its per-record drain in the relaxed slope.
    """
    cores = tuple(cores)
    if len(cores) not in (1, 2):
        raise ValueError("one or two cores required")
    active = [e for e in experts if e["Me"] > 0]
    phase_costs = [[] for _ in cores]
    critical = Fraction(0)
    useful = 0
    for e in active:
        me, h, f = (index(e[name]) for name in ("Me", "H", "F"))
        if me < 1 or h < 1 or f < 1:
            raise ValueError("positive active dimensions required")
        gu = [_phase_feed(c, me, f, h, 2, profile, preserve_m_padding) for c in cores]
        down = [_phase_feed(c, me, h, f, 1, profile, preserve_m_padding) for c in cores]
        for side in range(len(cores)):
            phase_costs[side].extend((gu[side], down[side]))
        if len(cores) == 1:
            expert_critical = gu[0] + down[0]
        else:
            expert_critical = fractional_two_machine_exact(gu[:1], gu[1:]) + fractional_two_machine_exact(down[:1], down[1:])
        critical = max(critical, expert_critical)
        useful += 3 * me * h * f
    if len(cores) == 1:
        load = sum(phase_costs[0], Fraction(0))
    else:
        load = fractional_two_machine_exact(*phase_costs)
    physical = Fraction(useful, sum(c.macs for c in cores))
    return max(load, critical, physical)
