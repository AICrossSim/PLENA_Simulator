"""Unrestricted MAC-throughput bound, independent of whole-expert owners."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .search import BATCHES, BUDGET, balanced, csv_out, dump, gm
from .trace_inputs import load_primary_inputs


def run(out):
    out = Path(out)
    windows = load_primary_inputs('heldout')['heldout']
    useful = [sum(3 * e['Me'] * e['H'] * e['F'] for e in w['experts']) for w in windows]
    mac_bound = np.array([math.ceil(x / BUDGET) for x in useful])
    pipeline_bound = mac_bound + 20
    rows = []
    for family in ('single', 'homogeneous', 'heterogeneous'):
        result = json.loads((out / f'heldout_{family}.json').read_text())
        byid = {r['window_id']: r for r in result}
        cycles = np.array([byid[w['id']]['cycles'] for w in windows])
        assert np.all(cycles >= pipeline_bound)
        for b in (*BATCHES, 'all'):
            mask = np.array([b == 'all' or w['batch'] == b for w in windows])
            ww = [w for w, flag in zip(windows, mask) if flag]
            reduce = lambda x: float(balanced(x, ww)) if b == 'all' else gm(x)
            ratio = reduce(pipeline_bound[mask] / cycles[mask])
            rows.append({'family': family, 'batch': b,
                         'MAC_only_lower_bound_ms': reduce(mac_bound[mask]) / 1e6,
                         'same_L21_II1_lower_bound_ms': reduce(pipeline_bound[mask]) / 1e6,
                         'frozen_design_compute_ms': reduce(cycles[mask]) / 1e6,
                         'best_possible_speedup_at_same_peak': 1 / ratio,
                         'maximum_possible_latency_reduction_pct': 100 * (1 - ratio),
                         'allows_Shared_split_and_unrestricted_task_interleaving': True,
                         'bound_is_not_a_feasible_schedule': True})
    csv_out(out / 'global_MAC_ceiling.csv', rows)
    dump(out / 'GLOBAL_MAC_CEILING.json', {
        'status': 'PASS', 'window_count': len(windows),
        'resource': '12288 useful BF16 multiplications per cycle, same 1GHz and II1',
        'bound': 'ceil(sum useful GEMM MACs / 12288) +20 final drain under common L21',
        'scope': 'even if Shared or routed experts can split arbitrarily and all padding can disappear',
        'MAC_only_bound_without_final_drain_also_reported': True,
        'not_a_feasible_program_or_physical_hardware_guarantee': True,
        'native_calibrated': False,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
