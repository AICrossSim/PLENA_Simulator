"""Frozen-geometry finite-cache Route A experiment; no hardware search.

Times are analytical 1 GHz cycles, not native simulator or GPU measurements.
Each captured window has independent cold or explicitly Shared-resident state.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor

from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_moe.workloads import load_workloads
from research.moe_dispatch.plena65536_shared.timing import resource_ledger

BATCHES = (2, 4, 8, 16, 64, 96, 128)
MIB = 1024**2
DESIGNS = {
    'single': (Core(1, 512, 1, 128),),
    'homo': (Core(1, 256, 1, 128), Core(1, 256, 1, 128)),
    'hetero': (Core(1, 64, 1, 704), Core(1, 128, 1, 160)),
}
SCENARIOS = ('fair_lru', 'fair_pinned', 'proposed_asymmetric',
             'fair_pinned_hard', 'ablate_no_prefetch')


def settings(scenario, design):
    result = dict(policy='workload_aware', threshold=1, prefetch=True,
                  cache_policy='lru')
    if scenario in ('fair_pinned', 'fair_pinned_hard'):
        result['cache_policy'] = 'pin_shared'
    if scenario == 'fair_pinned_hard':
        result.update(policy='hard_threshold', threshold=2)
    elif scenario == 'ablate_no_prefetch':
        result['prefetch'] = False
    elif scenario == 'proposed_asymmetric':
        if design == 'single':
            result['prefetch'] = False
        elif design == 'homo':
            result.update(policy='greedy_eft', threshold=2)
        else:
            result.update(policy='hard_threshold', threshold=2,
                          cache_policy='pin_shared')
    elif scenario not in ('fair_lru', 'fair_pinned'):
        raise ValueError(scenario)
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def numeric_row(window, design, scenario, bw, state, result, latency):
    row = {k: v for k, v in result.items()
           if isinstance(v, (str, int, float, bool)) or v is None}
    row.update(window_id=window['id'], batch=window['batch'], design=design,
               policy=scenario, bandwidth_gbs=bw, cache_state=state,
               request_latency_ns=latency)
    # Engine policy describes assignment; scenario identifies the fair ablation.
    row['policy'] = scenario
    row['assignment'] = settings(scenario, design)['policy']
    row['threshold'] = settings(scenario, design)['threshold']
    row['prefetch_enabled'] = settings(scenario, design)['prefetch']
    row['cache_policy'] = settings(scenario, design)['cache_policy']
    if 'core_cycles' in result:
        for i, cycles in enumerate(result['core_cycles']):
            row[f'core{i}_compute_cycles'] = cycles
    row.setdefault('core1_compute_cycles', 0)
    return row


def supply_floor(window, design):
    """Read-only necessary port budget for the declared OS repeated M loop."""
    from research.moe_dispatch.plena65536_affinity.model import schedule
    cores = DESIGNS[design]
    plan = schedule(cores, window, policy='workload_aware', threshold=1)
    by_id = {e['id']: e for e in window['experts']}
    w_bytes = x_bytes = 0
    for job in plan['jobs']:
        e = by_id[job['expert_id']]
        c = cores[job['core_id']]
        m, h, f = e['Me'], e['H'], e['F']
        m_tiles = (m + c.pm - 1) // c.pm
        w_bytes += 6 * h * f * m_tiles
        x_bytes += 2 * m * (2 * ((f + c.pn - 1)//c.pn) * h
                           + ((h + c.pn - 1)//c.pn) * f)
    return dict(window_id=window['id'], batch=window['batch'], design=design,
                ideal_compute_cycles=plan['cycles'], os_weight_sram_bytes=w_bytes,
                os_activation_sram_bytes=x_bytes,
                required_average_sram_TBs=(w_bytes+x_bytes)/plan['cycles']/1000,
                port_floor_ms_at_8TBs=(w_bytes+x_bytes)/8e12*1000,
                port_floor_ms_at_32TBs=(w_bytes+x_bytes)/32e12*1000,
                port_floor_ms_at_128TBs=(w_bytes+x_bytes)/128e12*1000)


def run_pair(task):
    from research.moe_dispatch.cache_route_a.model import run_window
    window, design, scenario, bw, state, latency = task
    kw = dict(**settings(scenario, design), initial=state,
              bandwidth_gbs=bw, request_latency_ns=latency)
    first = run_window(DESIGNS[design], window, **kw)
    if first != run_window(DESIGNS[design], window, **kw):
        raise AssertionError('Repeat mismatch')
    return numeric_row(window, design, scenario, bw, state, first, latency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--inputs', type=Path,
                    help='Portable saved workload JSON; default validates frozen source inputs')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    if args.inputs:
        data = json.loads(args.inputs.read_text())
        windows, provenance = data['windows'], data['manifest']
        from research.moe_dispatch.plena65536_moe.workloads import validate_window
        for window in windows:
            validate_window(window)
    else:
        data = load_workloads('heldout')
        windows, provenance = data['heldout'], data['manifest']
    if len(windows) != 135:
        raise ValueError('Expected all 135 frozen heldout windows')
    if args.smoke:
        windows = [next(w for w in windows if w['batch'] == b) for b in BATCHES]
    payload = {'windows': windows, 'manifest': provenance}
    (out/'inputs.json').write_text(json.dumps(payload, sort_keys=True)+'\n')
    import research.moe_dispatch.cache_route_a.model as model
    source = Path(model.__file__)
    config = {
        'status': 'SMOKE' if args.smoke else 'COMPLETE_EXPECTED',
        'scope': 'uncalibrated compute + finite capacity + shared external DMA model; ideal internal ports',
        'total_multipliers': 65536, 'frequency_GHz': 1,
        'sram_total_MiB': 64, 'reserved_working_MiB': 8, 'weight_cache_MiB': 56,
        'shared_weight_MiB': 33, 'routed_weight_MiB': 16.5,
        'pinning_organization': '33 MiB pinned Shared + 23 MiB shared streaming weight pool accessible by either core',
        'working_reserve_contract': 'stream Up/SiLU, reuse Gate storage for Z, direct final output merge; do not allocate whole Up and per-expert Y simultaneously',
        'working_peak_B128_MiB_bound': 7.2,
        'cache_states': ['cold', 'shared_resident'],
        'warm_state_scope': 'Shared starts loaded for ALL designs; independent window, not a real temporal cache trace',
        'main_bandwidth_GBs': 512, 'request_latency_ns': 64,
        'secondary_bandwidth_GBs': [256, 1024],
        'same_params_all_hardware': True, 'repetitions': 2,
        'geometry_source': '/scratch/shared/mcl123/plena/outputs/plena65536_affinity_20261013/FROZEN_SELECTION.json',
        'geometries': {k: [c.to_dict() for c in cs] for k, cs in DESIGNS.items()},
        'local_register_buffer_ledger': {k: resource_ledger(cs) for k, cs in DESIGNS.items()},
        'scenarios': {s: {k: settings(s,k) for k in DESIGNS} for s in SCENARIOS},
        'input_sha256': hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),
        'engine_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'reference_compute_ms': {'single': .022383588, 'homo_same_policy': .021391022,
                                 'hetero': .021107933},
        'limitations': ['No PPA/equal-area proof', 'No calibrated HBM bank/credit/controller timing',
                        'No SRAM bank/NoC/SiLU/router/combine timing',
                        'Column-strip OS loop differs from older m-outer supply ledger',
                        'No observed consecutive decode-window cache reuse assumption'],
    }
    (out/'CONFIG.json').write_text(json.dumps(config, indent=2)+'\n')
    rows, latency_rows = [], []
    completed = 0
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for bw in (512, 256, 1024):
            scenarios = SCENARIOS if bw == 512 else ('fair_lru', 'fair_pinned')
            for state in ('cold', 'shared_resident'):
                for scenario in scenarios:
                    for design in DESIGNS:
                        tasks = [(w, design, scenario, bw, state, 64) for w in windows]
                        rows.extend(pool.map(run_pair, tasks, chunksize=4))
                        completed += len(windows)
                        print(json.dumps({'completed': completed, 'bandwidth': bw,
                                          'state': state, 'scenario': scenario, 'design': design,
                                          'elapsed_s': round(time.monotonic()-start, 1)}), flush=True)
        # Zero per-burst latency separates request count from cache effects.
        for state in ('cold', 'shared_resident'):
            for design in DESIGNS:
                tasks = [(w, design, 'fair_lru', 512, state, 0) for w in windows]
                latency_rows.extend(pool.map(run_pair, tasks, chunksize=4))
    write_csv(out/'per_window.csv', rows)
    write_csv(out/'latency_sensitivity.csv', latency_rows)
    write_csv(out/'sram_port_necessary_bounds.csv',
              [supply_floor(w,d) for w in windows for d in DESIGNS])
    if hashlib.sha256(source.read_bytes()).hexdigest() != config['engine_sha256']:
        raise AssertionError('Engine changed during run; regenerate outputs')
    receipt = {'status': 'SMOKE' if args.smoke else 'PASS', 'windows': len(windows),
               'main_points': len(rows), 'latency_sensitivity_points': len(latency_rows),
               'repeats': 2, 'all_repeats_exactly_equal': True,
               'engine_sha256': config['engine_sha256'],
               'elapsed_seconds': time.monotonic()-start}
    (out/'COMPLETE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
