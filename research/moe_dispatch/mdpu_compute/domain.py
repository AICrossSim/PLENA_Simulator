"""Declared compact-family and equal-capability control domains.

This is a BF16/FP32 topology extension, not a claim that every replication
is native PLENA RTL.  BLEN>=16 is a requested family restriction, not a
paper-imposed physical legality rule.  Exact 4096-MAC allocation blocks
replace the impossible 43690/21846 split.
"""
from research.moe_dispatch.plena65536_dedicated.geometry import Core, enumerate_cores

BUDGET = 65536
THRESHOLDS = (1, 2, 4, 8)


def domains():
    cores = tuple(c for c in enumerate_cores()
                  if c.macs % 4096 == 0 and c.pn <= 2048)
    dense = tuple(c for c in cores if c.b in (16, 32)
                  and c.pm in (16, 32) and c.pn >= 32)
    stream = tuple(c for c in cores if c.pm in (1, 2))
    return {
        'original': ((Core(32, 2048),),),
        'compact_single': tuple((c,) for c in cores if c.macs == BUDGET
                               and c.b in (16, 32) and c.pm in (16, 32)
                               and c.pn in (32, 64)),
        'compact_homo': tuple((c, c) for c in cores if c.macs == BUDGET//2
                             and c.b in (8, 16) and c.pm in (8, 16)),
        'hetero': tuple((a, b) for a in dense for b in stream
                        if a.macs + b.macs == BUDGET),
        'fair_single': tuple((c,) for c in cores if c.macs == BUDGET),
        'fair_homo': tuple((c, c) for c in cores if c.macs == BUDGET//2),
        'fair_native_single': tuple((c,) for c in cores
                                   if c.macs == BUDGET and c.b >= 2),
        'fair_native_homo': tuple((c, c) for c in cores
                                 if c.macs == BUDGET//2 and c.b >= 2),
    }


def core_from_dict(d):
    return Core(*(int(d[k]) for k in ('b', 'r', 'rm', 'rn')))


def label(cores):
    return ' + '.join(f'{c.pm}x{c.pn}x{c.groups}[b{c.b},R{c.r}]' for c in cores)


def validate(cores):
    assert len(cores) in (1, 2)
    assert sum(c.macs for c in cores) == BUDGET
    assert all(c.macs % 4096 == 0 and c.pn <= 2048 for c in cores)

