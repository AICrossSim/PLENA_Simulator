"""Candidate chips of one cell, pre-ranked with the analytic model.

A candidate is a base: one array (single), two identical arrays (hom) or a shorter and a taller array (het). The
het rules are
  a  for heights m1 < m2 in HEIGHTS and a multiplier split s of the budget: the best standalone core of height m1 within
     s +- 2.5% of the budget and the best height-m2 partner that brings the pair to 85-100% of the budget;
  b  height-only pairs (m, n, k) + (c*m, n, k), c in {2, 4};
  c  the 30 most similar pairs whose heights differ by 2x or 4x (N and K ratios within 0.8-1.25).
Hom and single candidates are the het cores resized to half or all of the budget, filled up with the analytic
ranking to the number of het candidates. The extension adds pairs of every vocabulary height, ranked by the pair
model. The control stages restrict hom to >= 97% of the budget (fullhom) and het to at most the multipliers of the
reference hom chip (hetcap).
"""

from __future__ import annotations

import math

from .shapes import AnalyticModel, lcm

HEIGHTS = (2, 4, 8, 16)
SPLITS = [round(0.30 + 0.05 * i, 2) for i in range(9)]
EXTENSION_SPLITS = [round(0.20 + 0.05 * i, 2) for i in range(13)]
EXTENSION_SIZE = 200
SMALL_N = 128


def base_macs(base, kind: str) -> int:
    return sum(n * k for _, n, k in base) * (2 if kind == "hom" and len(base) == 1 else 1)


class Ranking:
    """Analytic times of shapes and pairs on the dev histograms of B32 and B128, cached per shape."""

    def __init__(self, model: AnalyticModel, kappa: float):
        self.model, self.kappa = model, kappa
        self.hists = model.histograms((32, 128))
        self.shapes = [tuple(int(x) for x in r) for r in model.shapes]
        self.macs = model.macs.astype(int)
        self._single = {}

    def single(self, i: int) -> float:
        if i not in self._single:
            self._single[i] = self.time(self.model.single, (i,))
        return self._single[i]

    def time(self, fn, args) -> float:
        return self.model.relative_time(self.hists, fn, args, self.kappa)


def rule_candidates(rank: Ranking, P: int) -> dict:
    """het (rules a, b, c), hom and single candidates; C = number of het candidates."""
    sh, mac = rank.shapes, rank.macs

    def ok_pair(a, b):
        return (
            a[0] <= b[0] and 0.85 * P <= a[1] * a[2] + b[1] * b[2] <= P and a[1] + b[1] <= 1024 and a[2] + b[2] <= 8192
        )

    het = []
    for m1 in HEIGHTS:
        for m2 in HEIGHTS:
            if m2 <= m1:
                continue
            for s in SPLITS:
                c1 = [i for i, x in enumerate(sh) if x[0] == m1 and (s - 0.025) * P <= mac[i] <= (s + 0.025) * P]
                if not c1:
                    continue
                i = min(c1, key=rank.single)
                c2 = [j for j, x in enumerate(sh) if x[0] == m2 and 0.85 * P - mac[i] <= mac[j] <= P - mac[i]]
                if c2:
                    het.append(("a", sh[i], sh[min(c2, key=rank.single)]))
    for m in (2, 4, 8):
        for c in (2, 4):
            for x in sh:
                if x[0] == c * m and 0.85 * P <= 2 * x[1] * x[2] <= P and 2 * x[1] <= 1024:
                    het.append(("b", (m, x[1], x[2]), x))
    near, by_height = [], {}
    for x in sh:
        by_height.setdefault(x[0], []).append(x)
    for a in sh:
        for b in by_height.get(2 * a[0], []) + by_height.get(4 * a[0], []):
            if (
                (a[1], a[2]) != (b[1], b[2])
                and 0.8 <= a[1] / b[1] <= 1.25
                and 0.8 <= a[2] / b[2] <= 1.25
                and ok_pair(a, b)
            ):
                near.append((abs(math.log(a[1] / b[1])) + abs(math.log(a[2] / b[2])), a, b))
    near.sort(key=lambda z: z[0])
    het += [("c", a, b) for _, a, b in near[:30]]
    seen, pairs = set(), []
    for rule, a, b in het:
        if (a, b) not in seen and ok_pair(a, b):
            seen.add((a, b))
            pairs.append(dict(base=[list(a), list(b)], rule=rule))
    C = len(pairs)

    cores = sorted({tuple(c) for p in pairs for c in p["base"]})

    def resize(c, budget, nmax):
        m, _, k = c
        u = lcm(m, 16)
        n2 = min(nmax, (budget // k) // u * u)
        return (m, n2, k) if n2 >= u and n2 * k >= 0.85 * budget else None

    hom, single = [], []
    for c in cores:
        x = resize(c, P // 2, 512)
        if x and [list(x), list(x)] not in hom:
            hom.append([list(x), list(x)])
        y = resize(c, P, 1024)
        if y and [list(y)] not in single:
            single.append([list(y)])
    model = rank.model
    if len(hom) < C:
        pool = sorted(
            (rank.time(model.homogeneous, (i,)), sh[i])
            for i in range(len(sh))
            if 0.85 * P / 2 <= mac[i] <= P / 2 and 2 * sh[i][1] <= 1024
        )
        for _, x in pool:
            if len(hom) >= C:
                break
            if [list(x), list(x)] not in hom:
                hom.append([list(x), list(x)])
    if len(single) < C:
        pool = sorted((rank.single(i), sh[i]) for i in range(len(sh)) if 0.85 * P <= mac[i] <= P and sh[i][1] <= 1024)
        for _, x in pool:
            if len(single) >= C:
                break
            if [list(x)] not in single:
                single.append([list(x)])
    return dict(
        het=pairs,
        hom=[dict(base=b, rule="het-core" if i < len(cores) else "mnk") for i, b in enumerate(hom[: max(C, 1)])],
        single=[dict(base=b, rule="") for b in single[: max(C, 1)]],
        C=C,
    )


def extension_pairs(rank: Ranking, P: int, keep_small_n: bool = False) -> tuple[set, list]:
    """Pairs of every vocabulary height m1 < m2 over multiplier splits of 20-80%: per (heights, split) the 3 best
    standalone cores and for each the 3 best partners (plus the best core with n <= SMALL_N outside them when
    keep_small_n). Returns the pool and its best EXTENSION_SIZE pairs by the pair model (indices into the vocabulary)."""
    sh, mac = rank.shapes, rank.macs

    def top(cands):
        cands = sorted(cands, key=rank.single)
        return cands[:3] + ([i for i in cands[3:] if sh[i][1] <= SMALL_N][:1] if keep_small_n else [])

    heights = sorted({x[0] for x in sh})
    pairs = set()
    for m1 in heights:
        for m2 in heights:
            if m2 <= m1:
                continue
            for sp in EXTENSION_SPLITS:
                for i in top(
                    [i for i, x in enumerate(sh) if x[0] == m1 and (sp - 0.025) * P <= mac[i] <= (sp + 0.025) * P]
                ):
                    for j in top(
                        [j for j, x in enumerate(sh) if x[0] == m2 and 0.85 * P - mac[i] <= mac[j] <= P - mac[i]]
                    ):
                        a, b = sh[i], sh[j]
                        if 0.85 * P <= a[1] * a[2] + b[1] * b[2] <= P and a[1] + b[1] <= 1024 and a[2] + b[2] <= 8192:
                            pairs.add((i, j))
    # equal times keep the set's iteration order (fixed for tuples of ints); the top 200 of some cells depend on it
    ranked = sorted(pairs, key=lambda ij: rank.time(rank.model.pair, ij))[:EXTENSION_SIZE]
    return pairs, ranked


def extend(cands: list[dict], rank: Ranking, ranked: list) -> int:
    """Append the ranked pairs that are not candidates yet (rule 'ext'); returns how many were added."""
    have = {tuple(map(tuple, c["base"])) for c in cands}
    added = 0
    for i, j in ranked:
        if (rank.shapes[i], rank.shapes[j]) not in have:
            have.add((rank.shapes[i], rank.shapes[j]))
            cands.append(dict(base=[list(rank.shapes[i]), list(rank.shapes[j])], rule="ext"))
            added += 1
    return added
