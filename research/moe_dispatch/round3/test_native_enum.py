import random
from dataclasses import replace
from research.moe_dispatch.round3.native_enum import (
    exact_visit, _python_visit, enable_exact_native_enum, disable_exact_native_enum)
from research.moe_dispatch.round3 import optimizer
from research.moe_dispatch.round3.common import inputs, frozen_designs, canonical
from research.moe_dispatch.round3.config import parameters


def bounds(alternatives,nr):
    ng=len(alternatives);suffix=[[0]*nr for _ in range(ng+1)];deps=[0]*(ng+1)
    for g in reversed(range(ng)):
        suffix[g]=[suffix[g+1][j]+min(a[1][j] for a in alternatives[g]) for j in range(nr)]
        deps[g]=max(deps[g+1],min(a[2] for a in alternatives[g]))
    return suffix,deps


def test_native_dfs_traversal_ties_and_counts():
    r=random.Random(20261008)
    for _ in range(400):
        ng=r.randrange(1,8);nr=r.randrange(0,13)
        alternatives=[[(tuple([a,4-a]),[r.randrange(0,1000) for _ in range(nr)],r.randrange(1000))
                       for a in range(r.randrange(1,5))] for _ in range(ng)]
        suffix,deps=bounds(alternatives,nr);initial=[r.randrange(100) for _ in range(nr)]
        assert exact_visit(alternatives,suffix,deps,initial)==_python_visit(alternatives,suffix,deps,initial)
    tied=[[((0,1),[0,0],0),((1,0),[0,0],0)]]*5
    s,d=bounds(tied,2)
    got=exact_visit(tied,s,d,[0,0])
    assert got==_python_visit(tied,s,d,[0,0])
    assert got[1]==((0,1),)*5


def test_large_coefficients_use_exact_python_fallback():
    huge=1<<90
    a=[[((0,1),[huge,2],1),((1,0),[1,huge],0)]]*3
    s,d=bounds(a,2)
    assert exact_visit(a,s,d,[0,0])==_python_visit(a,s,d,[0,0])


def test_real_assignments_and_replays_bit_exact():
    dev=inputs()['development']
    try:
        for mode,credit in [('pipelined',256),('pipelined',520),('port_tight',520)]:
            params=parameters(mode,credits=credit)
            designs=frozen_designs(mode,common_ws=True)
            # Includes small decode, mixed Shared+hot/cold experts, and one-core
            # cases where the full assignment model degenerates to one choice.
            for name in ('B1','B2','best_hetero'):
                design=designs[name]
                variants=[design]
                if len(design.cores)==2:
                    variants.append(replace(design,landing_mode='shared',
                                            landing_pool_bytes=sum(design.w_bytes),w_bytes=(0,0)))
                for candidate in variants:
                    for w in (dev[0],dev[3],dev[-3],dev[-1]):
                        disable_exact_native_enum()
                        original=optimizer.evaluate_design(w,candidate,params)
                        enable_exact_native_enum()
                        native=optimizer.evaluate_design(w,candidate,params)
                        assert canonical(original)==canonical(native)
    finally:
        disable_exact_native_enum()
