"""Compact completed candidate archives without changing chosen hardware.

True completion times are integer cycles, so ceil(a valid fractional lower
bound) remains a valid lower bound. The original continuous GM lower bound
is retained in all_candidates.csv.gz and selected.json. Upper bounds and all
timing results are unchanged. This is archive encoding, not a new model.
"""
import json
import numpy as np
from .common import ROOT,write_json

def pack():
    saved=0
    for p in (ROOT/'results').glob('*/development_bounds.npz'):
        if not (p.parent/'development_bootstrap.csv').exists():continue
        with np.load(p) as a:
            lo=a['lower'];hi=a['upper']
        if lo.dtype==np.int32:continue
        rounded=np.ceil(lo-1e-7).astype(np.int32)
        assert np.all(rounded<=hi)
        temp=p.with_name('development_bounds.packed.npz');old=p.stat().st_size
        np.savez_compressed(temp,lower=rounded,upper=hi.astype(np.int32))
        temp.replace(p);saved+=old-p.stat().st_size
        write_json(p.parent/'bound_archive_encoding.json',dict(lower='integer-cycle ceil of original fractional lower bound',
            upper='unchanged exact integer-cycle feasible schedules',
            original_continuous_objectives='all_candidates.csv.gz and selected.json',changes_hardware_or_results=False))
    print('Candidate archive MiB saved',round(saved/2**20,2),flush=True)

if __name__=='__main__':pack()
