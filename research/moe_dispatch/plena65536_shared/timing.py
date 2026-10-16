"""Declared analytical timing, not a calibrated native PLENA measurement.

The native M_MM steady-state interval b and first-tile 2*b+9 come from
analytic_models/performance/customISA_lib.json.  M_SUM is priced explicitly:
ceil(log2(R/b)) registered add stages and two output/commit cycles.  A
shadow PE-partial bank and pipelined tree permit overlap across output tiles.
OS/WS/IS retention loops use the same compute pipeline with ideal context
ports; their different context and operand costs are reported separately.
"""
from dataclasses import dataclass
from math import ceil
from .geometry import Core

BATCHES = (2, 4, 8, 16, 64, 96, 128)
H, F = 2048, 2816

def divup(x, y):
    return (x + y - 1) // y

@dataclass(frozen=True)
class Profile:
    name: str
    tree_stage: int = 1
    serialized: bool = False

    def drain(self, c: Core):
        return c.b + 9 + (c.groups.bit_length() - 1) * self.tree_stage + 2

PROFILES = (Profile('pipelined_tree1'), Profile('pipelined_tree2', 2),
            Profile('serialized_output', 1, True))

def phase(c: Core, m: int, n: int, k: int, factor: int = 1,
          profile: Profile = PROFILES[0]) -> int:
    if m == 0 or n == 0:
        return 0
    records = factor * divup(m, c.pm) * divup(n, c.pn)
    feed = divup(k, c.r) * c.b
    drain = profile.drain(c)
    return records * (feed + drain) if profile.serialized else records * feed + drain

def shared(c, batch, profile=PROFILES[0]):
    return phase(c, batch, F, H, 2, profile) + phase(c, batch, H, F, 1, profile)

def _split_monotone(total, left, right):
    """Exact optimum of max(left(x),right(total-x)), monotone staircases."""
    lo, hi = 0, total
    while lo < hi:
        mid = (lo + hi) // 2
        if left(mid) >= right(total - mid):
            hi = mid
        else:
            lo = mid + 1
    choices = {0, total, lo, max(0, lo-1)}
    return min(((max(left(x), right(total-x)), x) for x in choices))

def partition(cores, batch, profile=PROFILES[0]):
    """Compare exact contiguous token rows and paired-GU output columns.

    Column contract: a GU barrier exposes complete shared Z, then both cores
    may compute disjoint Down output columns.  It needs no cross-core partials.
    Row contract: each core finishes all three GEMMs for its own token rows.
    Runtime slicing can vary with B; geometry and flow settings stay frozen.
    """
    if len(cores) == 1:
        return {'cycles': shared(cores[0], batch, profile), 'kind': 'single',
                'gu_split': F, 'down_split': H, 'row_split': batch}
    a, b = cores
    row_cycles, row_cut = _split_monotone(
        batch, lambda m: shared(a,m,profile), lambda m: shared(b,m,profile))
    gu, gu_cut = _split_monotone(F,
        lambda n: phase(a,batch,n,H,2,profile),
        lambda n: phase(b,batch,n,H,2,profile))
    down, down_cut = _split_monotone(H,
        lambda n: phase(a,batch,n,F,1,profile),
        lambda n: phase(b,batch,n,F,1,profile))
    column = {'cycles': gu+down, 'kind': 'columns', 'gu_split': gu_cut,
              'down_split': down_cut, 'row_split': None}
    row = {'cycles': row_cycles, 'kind': 'rows', 'gu_split': None,
           'down_split': None, 'row_split': row_cut}
    return min((column,row), key=lambda x:(x['cycles'],x['kind']))

def regions(cores, batch, plan):
    from .dataflow import Region
    if plan['kind'] == 'single':
        return {name:[Region(0,0,batch,0,n)] for name,n in [('gate',F),('up',F),('down',H)]}
    if plan['kind'] == 'rows':
        cut=plan['row_split']
        return {name:[r for r in (Region(0,0,cut,0,n),Region(1,cut,batch,0,n))
                      if r.m_stop > r.m_start and r.n_stop > r.n_start]
                for name,n in [('gate',F),('up',F),('down',H)]}
    return {name:[r for r in (Region(0,0,batch,0,cut),Region(1,0,batch,cut,n))
                  if r.m_stop > r.m_start and r.n_stop > r.n_start]
            for name,n,cut in [('gate',F,plan['gu_split']),('up',F,plan['gu_split']),
                               ('down',H,plan['down_split'])]}

def useful_macs(batch):
    return 3*batch*H*F

def resource_ledger(cores):
    return {
        'multipliers':sum(c.macs for c in cores),
        'PE_FP32_bytes':sum(4*c.macs for c in cores),
        'shadow_FP32_bytes':sum(4*c.macs for c in cores),
        'M_SUM_adders':sum(c.pm*c.pn*(c.groups-1) for c in cores),
        'M_SUM_tree_FP32_register_bytes':sum(4*c.pm*c.pn*(c.groups-1) for c in cores),
        'full_operand_double_buffer_bytes':sum(4*c.r*(c.pm+c.pn) for c in cores),
        'native_square_groups':sum(c.rm*c.rn*c.groups for c in cores),
        'area_calibrated':False,
    }
