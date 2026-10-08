"""Compare every development seed with the captured original search source."""
from __future__ import annotations
import hashlib
import itertools
import sys
import types

from .common import ROOT, inputs, metadata, sha, write_json
from .search import FAMILIES, seed_designs, key, _ordered_geometry_cache


def main():
    source = ROOT / "archive/source_snapshots/E4_main_before_cache/search.py"
    original = types.ModuleType("research.moe_dispatch.round3._original_search_cache_check")
    original.__package__ = "research.moe_dispatch.round3"
    original.__file__ = str(source)
    sys.modules[original.__name__] = original
    exec(compile(source.read_text(), str(source), "exec"), original.__dict__)
    workloads = inputs()["development"]
    rows = []
    _ordered_geometry_cache.cache_clear()
    for family, constraint in itertools.product(FAMILIES, ("C0", "C1")):
        old = original.seed_designs(family, workloads, constraint=constraint)
        new = seed_designs(family, workloads, constraint=constraint)
        digest = hashlib.sha256()
        count = 0
        for index, (a, b) in enumerate(itertools.zip_longest(old, new)):
            if a is None or b is None or key(a) != key(b):
                raise AssertionError((family, constraint, index, "seed sequence changed"))
            digest.update((key(a) + "\n").encode())
            count += 1
        rows.append({"family": family, "constraint": constraint,
                     "complete_seed_count": count, "identical": True,
                     "ordered_seed_sha256": digest.hexdigest()})
        print(f"{family} {constraint}: all {count} seeds identical", flush=True)
    cache = _ordered_geometry_cache.cache_info()
    assert cache.misses == len(FAMILIES) and cache.hits == len(FAMILIES)
    write_json(ROOT / "diagnostics/geometry_cache/EQUIVALENCE.json", metadata({
        "original_search_sha256": sha(source), "rows": rows,
        "development_ids": [w["id"] for w in workloads],
        "full_generators_compared": True, "cache_info": cache._asdict(),
        "no_solver_or_physical_evaluation_cached": True,
    }))


if __name__ == "__main__":
    main()
