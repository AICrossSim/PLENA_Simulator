"""Distance of dispatch policies from the optimum in an abstract two-stage model of one layer window.

Jobs are the shared expert and every routed expert active in the window (the router is left out, it is the same for
every policy). Stage 1 is HBM, one machine: job j on array c fetches its weights for ceil(bytes_jc / H) cycles. Stage 2
are the arrays, unrelated machines: compute takes the cycles of the job's compiled program. Compute may start once the
job's fetch has started and cannot end before the fetch ends; an array holds at most `buffer` jobs between fetch start
and compute end (current and next). Against the engine this model fetches serially at full bandwidth, may fetch a next
job completely ahead and has no port, converter, vector-phase or control-cycle detail. Policies 9, 11, 12 and 13 are
re-simulated in the model with their selection rules on the engine's compiled estimates; CP-SAT gives the optimal
makespan of the same model, or a lower bound when the time limit ends the search.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..sim import workload as wl
from ..sim.engine import Environment, program_metrics
from . import results as rs
from .organisations import ORGS, expand

NEVER = 10**15


def task_tables(model: str, shapes, env: Environment, batch: int, token_counts) -> dict:
    """(estimate, compute cycles, HBM bytes) of every job kind (token count, or 'shared') on every array; vector lanes
    in proportion to the multipliers and SRAM by the chip's split, as in the engine."""
    w, nc = wl.MODELS[model], len(shapes)
    macs, lanes, used = sum(s[1] * s[2] for s in shapes), [], 0
    for c in range(nc):
        v = 2048 - used if c == nc - 1 else max(1, int(2048 * shapes[c][1] * shapes[c][2] / macs))
        lanes.append(v)
        used += v
    share = env.share if nc == 2 else 1000
    tables = {}
    for c in range(nc):
        capacity = int((env.capacity - 6 * batch * w["hidden"]) * (share if c == 0 else 1000 - share) / 1000)
        for M in [*token_counts, "shared"]:
            shared = M == "shared"
            p = program_metrics(shapes[c], batch if shared else M, shared, w, env, capacity, lanes[c])
            if p is not None:
                tables[c, M] = (p["estimate"], float(p["cycles"]), float(p["hbm_bytes"]))
    return tables


def jobs(window: dict) -> list[tuple]:
    """(kind, expert): the shared expert first, then every active routed expert with its token count as kind."""
    experts, tokens = np.unique(window["ids"], return_counts=True)
    return [("shared", -1)] + [(int(m), int(e)) for e, m in zip(experts, tokens)]


def list_schedule(policy: int, J, T, nc: int, hbm: int, lookahead: int, buffer: int = 2, macs=None) -> int:
    """Makespan of the abstract model under dispatch policy 9, 11, 12 or 13."""

    def est(c, k):
        return T[c, k][0] if (c, k) in T else 1e15

    def raw(c, k):
        return T[c, k][1] if (c, k) in T else 1e15

    def fetch(c, k):
        return int(np.ceil(T[c, k][2] / hbm)) if (c, k) in T else NEVER

    def kind(j):
        return J[j][0]

    if policy == 9:  # longest first by the smaller estimate
        order = sorted(range(len(J)), key=lambda j: -min(est(c, kind(j)) for c in range(nc)))
    else:  # shared expert first, then routed experts by token count, descending
        order = sorted(
            range(len(J)), key=lambda j: (0 if kind(j) == "shared" else 1, -(J[j][0] if kind(j) != "shared" else 10**6))
        )
    t = 0
    cur, nxt, start = [None] * nc, [None] * nc, [0] * nc
    fstart, fend, core_of = {}, {}, {}
    hbm_free, hq, queue, assigned, done = 0, [], [], set(), 0
    front, back, side = 0, len(order) - 1, False
    aref = next((min(fetch(c, kind(j)) for c in range(nc)) for j in range(len(J)) if kind(j) != "shared"), None) or 1

    def rem(c):
        return 0 if cur[c] is None else max(0, start[c] + round(est(c, kind(cur[c]))) - t)

    def committed(c):
        return rem(c) + (round(est(c, kind(nxt[c]))) if nxt[c] is not None else 0)

    def outstanding_fetch(c):
        d = 0
        for j in (cur[c], nxt[c]):
            if j is not None and j not in fend:
                d += fetch(c, kind(j))
            elif j is not None and fend[j] > t:
                d += fend[j] - t
        return d

    def fill():
        nonlocal front, back, side
        if policy == 12:  # the window takes tasks alternately from the hot and the cold end
            while len(queue) < lookahead and front <= back:
                j = order[back] if side else order[front]
                if side:
                    back -= 1
                else:
                    front += 1
                side = not side
                if j not in assigned:
                    queue.append(j)
        else:
            while len(queue) < lookahead and front < len(order):
                j = order[front]
                front += 1
                if j not in assigned:
                    queue.append(j)

    shared_done, ends = [False], [None, None]
    routed = [j for j in order if kind(j) != "shared"]

    def choose13(c, must):
        """Two-ended: the array with more multipliers takes the hot end, the other the cold end; returns a job."""
        o = 1 - c
        if ends[0] is None:
            ends[0], ends[1] = 0, len(routed) - 1
        sj = next(j for j in range(len(J)) if kind(j) == "shared")
        if not shared_done[0]:
            f0 = rem(0) + round(est(0, "shared"))
            f1 = rem(1) + round(est(1, "shared")) if nc == 2 else NEVER
            pref = 0 if f0 <= f1 else 1
            if pref == c:
                shared_done[0] = True
                return sj
            if must and nc == 2 and cur[pref] is not None:
                fi, fb = (f1 if c else f0), min(f0, f1)
                if fi <= fb + (fb >> 1):
                    shared_done[0] = True
                    return sj
        if ends[0] > ends[1]:
            return None
        if nc == 1 or macs[c] > macs[o] or (macs[c] == macs[o] and c == 0):
            j = routed[ends[0]]
            ends[0] += 1
        else:
            j = routed[ends[1]]
            ends[1] -= 1
        return j

    def choose(c, must, next_slot):
        """Queue index of the job for array c (policies 9, 11, 12), or None."""
        if not queue:
            return None
        if policy in (9, 11):
            best, br, bz = None, -1, -1
            for qi, j in enumerate(queue):
                f0 = rem(0) + round(est(0, kind(j)))
                f1 = rem(1) + round(est(1, kind(j))) if nc == 2 else NEVER
                cc = 0 if f0 <= f1 else 1
                if (cc, kind(j)) not in T:
                    cc = 1 - cc
                if cc != c or (not next_slot and cur[cc] is not None):
                    continue
                r, z = (abs(f0 - f1) if nc == 2 else 0), est(c, kind(j))
                if r > br or (r == br and z > bz):
                    best, br, bz = qi, r, z
            if best is None and must and nc == 2:
                bd, bz = 1e30, -1
                for qi, j in enumerate(queue):
                    if (c, kind(j)) not in T:
                        continue
                    f0, f1 = rem(0) + round(est(0, kind(j))), rem(1) + round(est(1, kind(j)))
                    fi, fb = (f0 if c == 0 else f1), min(f0, f1)
                    if policy == 9:
                        if not fi <= 1.5 * fb:
                            continue
                        d = fi / max(1.0, fb)
                    else:
                        if fi > fb + (fb >> 1):
                            continue
                        d = fi - fb
                    z = est(c, kind(j))
                    if d < bd or (d == bd and z > bz):
                        best, bd, bz = qi, d, z
            return best
        o, Rc = 1 - c, committed(c)
        Ro = committed(o) if nc == 2 else NEVER
        want_hot = not (Rc - outstanding_fetch(c) > aref and (nc == 1 or Ro - outstanding_fetch(o) > aref))
        best, bk, bz = None, 1e30, -1
        for qi, j in enumerate(queue):
            if (c, kind(j)) not in T:
                continue
            pc = round(est(c, kind(j)))
            fo = Ro + round(est(o, kind(j))) if nc == 2 else NEVER
            if Rc + pc > fo:
                continue
            e = pc - fetch(c, kind(j))
            key = -e if want_hot else e
            if key < bk or (key == bk and pc > bz):
                best, bk, bz = qi, key, pc
        if best is None and must:
            bd, bz = 1e30, -1
            for qi, j in enumerate(queue):
                if (c, kind(j)) not in T:
                    continue
                fc = Rc + round(est(c, kind(j)))
                fo = Ro + round(est(o, kind(j))) if nc == 2 else NEVER
                fb = min(fc, fo)
                if fc > fb + (fb >> 1):
                    continue
                d, z = fc - fb, est(c, kind(j))
                if d < bd or (d == bd and z > bz):
                    best, bd, bz = qi, d, z
        return best

    def take(c, j, slot):
        assigned.add(j)
        core_of[j] = c
        hq.append((t, c, j))
        if slot == "current":
            cur[c], start[c] = j, t
        else:
            nxt[c] = j

    computing, finished, iterations = [None] * nc, 0, 0
    fill()
    while done < len(J):
        iterations += 1
        if iterations > 100000:
            raise RuntimeError("the abstract schedule did not converge")
        fill()
        progress = True
        while progress:  # free arrays: promote the next job or dispatch a new one
            progress = False
            for c in range(nc):
                if cur[c] is not None:
                    continue
                if nxt[c] is not None:
                    cur[c], nxt[c], start[c], progress = nxt[c], None, t, True
                    continue
                if policy == 13:
                    j = choose13(c, True)
                    if j is not None:
                        take(c, j, "current")
                        progress = True
                    continue
                qi = choose(c, True, False)
                if qi is not None:
                    take(c, queue.pop(qi), "current")
                    progress = True
                    fill()
        for c in range(nc if buffer >= 2 else 0):  # next job once the current job's fetch has completed
            if cur[c] is not None and nxt[c] is None and cur[c] in fend and fend[cur[c]] <= t:
                if policy == 13:
                    j = choose13(c, False)
                    if j is not None:
                        take(c, j, "next")
                    continue
                qi = choose(c, False, True)
                if qi is not None:
                    take(c, queue.pop(qi), "next")
                    fill()
        if hbm_free <= t and hq:  # one fetch at a time
            if policy == 12 and len({x[1] for x in hq}) > 1:  # the array whose current work ends first goes first
                hq.sort(key=lambda x: (rem(x[1]) if cur[x[1]] is not None and cur[x[1]] != x[2] else 0, x[0], x[1]))
            else:
                hq.sort(key=lambda x: (x[0], x[1]))
            _, c, j = hq.pop(0)
            fstart[j] = max(t, hbm_free)
            fend[j] = fstart[j] + fetch(c, kind(j))
            hbm_free = fend[j]
        for c in range(nc):  # compute starts once the current job's fetch has started
            j = cur[c]
            if j is not None and j in fstart and computing[c] is None:
                s = max(t, fstart[j])
                computing[c] = (j, max(s + round(raw(c, kind(j))), fend[j]))
        t = min(
            [
                x
                for x in [hbm_free if hbm_free > t else None] + [x[1] for x in computing if x]
                if x is not None and x > t
            ],
            default=t + 1,
        )
        for c in range(nc):
            if computing[c] is not None and computing[c][1] <= t:
                finished = max(finished, computing[c][1])
                computing[c], cur[c] = None, None
                done += 1
    return finished


def optimum(J, T, nc: int, hbm: int, time_limit: float = 30.0, workers: int = 4, buffer: int = 2):
    """CP-SAT makespan of the abstract model: (objective, best bound, status)."""
    from ortools.sat.python import cp_model

    m = cp_model.CpModel()
    a = {(j, c): int(np.ceil(T[c, k][2] / hbm)) for j, (k, _) in enumerate(J) for c in range(nc) if (c, k) in T}
    p = {(j, c): round(T[c, k][1]) for j, (k, _) in enumerate(J) for c in range(nc) if (c, k) in T}
    horizon = sum(max(a[j, c] for c in range(nc) if (j, c) in a) for j in range(len(J))) + sum(
        max(p[j, c] for c in range(nc) if (j, c) in p) for j in range(len(J))
    )
    fetches, compute, occupancy, ends, chosen = [], {c: [] for c in range(nc)}, {c: [] for c in range(nc)}, [], {}
    for j in range(len(J)):
        options = []
        for c in range(nc):
            if (j, c) not in a:
                continue
            x = m.NewBoolVar(f"x{j}_{c}")
            options.append(x)
            chosen[j, c] = x
            F, S, E = (m.NewIntVar(0, horizon, "") for _ in range(3))
            fetches.append(m.NewOptionalFixedSizeIntervalVar(F, a[j, c], x, ""))
            compute[c].append(m.NewOptionalIntervalVar(S, m.NewIntVar(p[j, c], horizon, ""), E, x, ""))
            occupancy[c].append(m.NewOptionalIntervalVar(F, m.NewIntVar(0, horizon, ""), E, x, ""))
            m.Add(S >= F).OnlyEnforceIf(x)
            m.Add(E >= F + a[j, c]).OnlyEnforceIf(x)
            ends.append((E, x))
        m.AddExactlyOne(options)
    m.AddNoOverlap(fetches)
    for c in range(nc):
        m.AddNoOverlap(compute[c])
        m.AddCumulative(occupancy[c], [1] * len(occupancy[c]), buffer)
    makespan = m.NewIntVar(0, horizon, "makespan")
    for E, x in ends:
        m.Add(makespan >= E).OnlyEnforceIf(x)
    m.Add(makespan >= sum(min(a[j, c] for c in range(nc) if (j, c) in a) for j in range(len(J))))
    for c in range(nc):
        m.Add(makespan >= sum(p[j, c] * chosen[j, c] for j in range(len(J)) if (j, c) in chosen))
    m.Minimize(makespan)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds, s.parameters.num_workers = time_limit, workers
    status = s.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, s.StatusName(status)
    return s.ObjectiveValue(), s.BestObjectiveBound(), s.StatusName(status)


def oracle(
    root: Path,
    out: Path,
    model: str,
    fmt: str,
    P: int,
    org: str,
    batches,
    n_windows: int,
    buffer: int,
    time_limit: float,
    workers: int,
) -> list[dict]:
    if ORGS[org].arrays != 2 or ORGS[org].streams != 2:
        raise ValueError("the abstract model has one task stream per array")
    cell = rs.cell_name(model, fmt, P)
    chip = rs.read_json(rs.final_path(out, cell, org))["best_main"]
    env = replace(Environment.from_dict(chip["env"]), shared_array=ORGS[org].shared_array)
    shapes, records = expand(ORGS[org], chip["base"]), []
    macs, nc = [s[1] * s[2] for s in shapes], len(shapes)
    for B in batches:
        rows = wl.windows(root, model, B, "report")
        rows = [rows[i] for i in sorted(set(np.linspace(0, len(rows) - 1, n_windows).round().astype(int)))]
        tokens = sorted({int(x) for r in rows for x in np.unique(r["ids"], return_counts=True)[1]})
        T = task_tables(model, shapes, env, B, tokens)
        for wi, r in enumerate(rows):
            J = jobs(r)
            makespans = {
                f"p{q}": list_schedule(q, J, T, nc, env.H, env.lookahead, buffer, macs) for q in (9, 11, 12, 13)
            }
            t0 = time.time()
            opt, bound, status = optimum(J, T, nc, env.H, time_limit, workers, buffer)
            floor = sum(min(math.ceil(T[c, k][2] / env.H) for c in range(nc) if (c, k) in T) for k, _ in J)
            records.append(
                dict(
                    cell=cell,
                    org=org,
                    B=B,
                    window=wi,
                    buffer=buffer,
                    jobs=len(J),
                    floor=floor,
                    opt=opt,
                    bound=bound,
                    lb=max(floor, bound or 0),
                    status=status,
                    solve_s=time.time() - t0,
                    **makespans,
                )
            )
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt", choices=sorted(wl.FORMATS))
    ap.add_argument("P", type=int)
    ap.add_argument("org", choices=("hom11", "het11"))
    ap.add_argument("--batches", type=int, nargs="+", default=(16, 32, 64))
    ap.add_argument("--windows", type=int, default=12, help="report windows per batch, evenly spaced")
    ap.add_argument("--buffer", type=int, default=2, help="jobs an array holds between fetch start and compute end")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    records = oracle(a.data, a.out, a.model, a.fmt, a.P, a.org, a.batches, a.windows, a.buffer, a.time_limit, a.workers)
    rs.write_json(
        Path(a.out) / "oracle" / f"{rs.cell_name(a.model, a.fmt, a.P)}_{a.org}_buffer{a.buffer}.json", records
    )
    for B in a.batches:
        rows = [r for r in records if r["B"] == B]
        print(
            B,
            {k: round(float(np.mean([r[k] / r["lb"] for r in rows])), 4) for k in ("p9", "p11", "p12", "p13")},
            flush=True,
        )


if __name__ == "__main__":
    main()
