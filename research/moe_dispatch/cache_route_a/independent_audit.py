"""Independent conservation checks for the finite-cache experiment artifacts.

No search, frozen output or engine implementation is changed.  Analytical
floors are recomputed from input matrix dimensions rather than engine fields.
Small representative event replays independently check DMA serialization,
cache payload accounting, dependency readiness and compute cost conservation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

MIB = 2**20
BATCHES = (2, 4, 8, 16, 64, 96, 128)


def gm(values):
    values = list(values)
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def close(a, b, *, tolerance=1e-6):
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=tolerance), (a, b)


def weight_bytes(window):
    return sum(6 * e['H'] * e['F'] for e in window['experts'])


def useful_macs(window):
    return sum(3 * e['Me'] * e['H'] * e['F'] for e in window['experts'])


def cost(core, expert):
    """Rebuild inherited feed/drain cost from scalar fields, without phase()."""
    div = lambda x, y: (x + y - 1) // y
    m, h, f = (expert[k] for k in ('Me', 'H', 'F'))
    b, r = core.b, core.r
    drain = b + 11 + int(math.log2(r // b))
    return (2 * div(m, core.pm) * div(f, core.pn) * div(h, r) * b
            + div(m, core.pm) * div(h, core.pn) * div(f, r) * b + 2 * drain)


def event_audit(cores, window, *, cache_policy, initial, prefetch):
    from .model import WeightCache, run_window

    class CheckedCache(WeightCache):
        checks = 0

        def verify(self):
            entries = list(self.entries.values())
            assert all(e.refs >= 0 for e in entries)
            assert self.used_bytes == sum(e.nbytes for e in entries)
            assert self.nonpinned_bytes == sum(e.nbytes for e in entries if not e.pinned)
            assert self.used_bytes <= self.capacity_bytes
            assert self.nonpinned_bytes <= self.capacity_bytes - self.pinned_bytes
            self.checks += 1

        def reserve(self, *args, **kwargs):
            result = super().reserve(*args, **kwargs)
            self.verify()
            return result

        def acquire(self, *args, **kwargs):
            result = super().acquire(*args, **kwargs)
            self.verify()
            return result

        def release(self, *args, **kwargs):
            result = super().release(*args, **kwargs)
            self.verify()
            return result

    cache = CheckedCache(56*MIB, cache_policy,
                         33*MIB if cache_policy == 'pin_shared' else 0)
    result = run_window(cores, window, policy='workload_aware', threshold=1,
                        cache_policy=cache_policy, initial=initial,
                        prefetch=prefetch, cache=cache, debug=True)
    dma_intervals = []
    compute_intervals = defaultdict(list)
    ready = {}
    last_time = -1
    total_bytes = 0
    for event in result['events']:
        key = tuple(event['key'])
        now = event['time_ns']
        assert now >= last_time
        last_time = now
        if event['event'] == 'cache_hit':
            # Fresh caches can initially hit only Shared when explicitly warm.
            if key not in ready:
                assert initial == 'shared_resident' and key[1] == -1
                ready[key] = 0
        elif event['event'] == 'dma_start':
            close(event['end_ns'] - now, 64 + event['bytes']/512)
            dma_intervals.append((now, event['end_ns']))
            total_bytes += event['bytes']
        elif event['event'] == 'dma_end':
            ready[key] = now
        elif event['event'] == 'compute_start':
            assert key in ready and ready[key] <= now
            compute_intervals[event['core']].append((now, event['end_ns']))
    for intervals in [dma_intervals, *compute_intervals.values()]:
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    assert total_bytes == result['hbm_bytes']
    close(sum(e-s for s, e in dma_intervals), result['dma_busy_ns'])
    experts = {e['id']: e for e in window['experts']}
    for core_id, core in enumerate(cores):
        expected = sum(cost(core, experts[eid]) for eid in result['core_expert_ids'][core_id])
        close(sum(e-s for s, e in compute_intervals[core_id]), expected)
        assert expected == result['core_compute_cycles'][core_id]
    assert sum(map(len, result['core_expert_ids'])) == len(experts)
    # Zero external transfer latency must restore exactly the inherited compute.
    ideal = run_window(cores, window, policy='workload_aware', threshold=1,
                       bandwidth_gbs=math.inf, request_latency_ns=0)
    assert ideal['total_cycles'] == ideal['ideal_compute_cycles']
    return {'window_id': window['id'], 'cache_policy': cache_policy,
            'initial': initial, 'prefetch': prefetch, 'events': len(result['events']),
            'cache_mutation_checks': cache.checks, 'dma_bursts': len(dma_intervals),
            'peak_cache_bytes': result['cache_peak_bytes'],
            'ideal_memory_exact': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    inputs = json.loads((out/'inputs.json').read_text())
    config = json.loads((out/'CONFIG.json').read_text())
    complete = json.loads((out/'COMPLETE.json').read_text())
    windows = inputs['windows']
    by_id = {w['id']: w for w in windows}
    assert len(windows) == len(by_id) == 135
    assert Counter(w['batch'] for w in windows) == {2:27, 4:27, 8:27, 16:27, 64:9, 96:9, 128:9}
    assert hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest() == config['input_sha256']
    assert complete['all_repeats_exactly_equal'] and complete['repeats'] == 2
    from .run import DESIGNS
    working_storage = {}
    for design, cores in DESIGNS.items():
        assert sum(c.pm*c.pn*c.groups for c in cores) == 65536
        assert [c.to_dict() for c in cores] == config['geometries'][design]
        # A sufficient allocation for a declared streamed-Up/SiLU contract:
        # one full Gate output and BF16 Z per active expert, no full Up/Y;
        # Down strips merge directly into one global FP32 output.  Two-core
        # maximum conservatively combines Shared with an Me=B routed expert.
        batch = 128
        active_widths = [2816] if len(cores) == 1 else [2816, 1408]
        x = len(cores) * batch * 2048 * 2
        gate_z = sum(batch*f*6 for f in active_widths)
        combined_y = batch*2048*4
        local = sum(8*c.macs + 4*c.pm*c.pn*(c.groups-1)
                    + 4*c.r*(c.pm+c.pn) for c in cores)
        metadata_reserve = MIB
        peak = x + gate_z + combined_y + local + metadata_reserve
        assert peak <= 8*MIB
        working_storage[design] = {
            'conservative_B128_working_bytes':peak,
            'input_gathers':x, 'gate_fp32_and_z_bf16':gate_z,
            'global_output_fp32':combined_y, 'local_registers_and_buffers':local,
            'metadata_reserve':metadata_reserve,
            'contract':'stream Up/SiLU; direct Down output merge; ideal ports/nonlinear timing'}
    with (out/'per_window.csv').open() as f:
        rows = list(csv.DictReader(f))
    with (out/'latency_sensitivity.csv').open() as f:
        sensitivity = list(csv.DictReader(f))
    assert len(rows) == 7290 and len(sensitivity) == 810
    assert complete['main_points'] == len(rows)
    assert complete['latency_sensitivity_points'] == len(sensitivity)
    identities = set()
    for row in rows + sensitivity:
        identity = tuple(row[k] for k in ('window_id','design','policy','bandwidth_gbs',
                                           'cache_state','request_latency_ns'))
        assert identity not in identities
        identities.add(identity)
        window = by_id[row['window_id']]
        n = lambda key: float(row[key])
        unique = weight_bytes(window)
        assert n('unique_weight_bytes') == unique
        assert n('useful_MACs') == useful_macs(window)
        assert n('sram_total_bytes') == 64*MIB
        assert n('sram_reserved_bytes') == 8*MIB
        assert n('weight_capacity_bytes') == 56*MIB
        assert n('cache_peak_bytes') <= n('weight_capacity_bytes')
        assert n('cache_final_bytes') <= n('cache_peak_bytes')
        assert n('pinned_reservation_bytes') == (33*MIB if row['cache_policy']=='pin_shared' else 0)
        close(n('latency_ms')*1e6, n('total_cycles'))
        close(n('compute_ms')*1e6, n('compute_cycles'))
        close(n('stall_cycles'), n('total_cycles')-n('compute_cycles'))
        close(n('stall_ms')*1e6, n('stall_cycles'))
        close(n('compute_cycles'), max(n('core0_compute_cycles'), n('core1_compute_cycles')))
        assert n('total_cycles') + 1e-6 >= n('hbm_bytes')/n('bandwidth_gbs')
        assert n('total_cycles') + 1e-6 >= n('compute_cycles')
        close(n('dma_busy_ns'), n('hbm_bytes')/n('bandwidth_gbs') +
              n('dma_requests')*n('request_latency_ns'))
        assert n('total_cycles') + 1e-6 >= n('dma_busy_ns')
        close(n('effective_bandwidth_gbs'), n('hbm_bytes')/n('total_cycles'))
        close(n('shared_hit_bytes')+n('shared_miss_bytes'), 33*MIB)
        assert n('refetch_bytes') == 0  # Each distinct matrix strip used once.
        if row['cache_state'] == 'cold':
            assert n('initial_resident_bytes') == n('explicitly_seeded_bytes') == 0
            assert n('shared_hit_bytes') == 0 and n('shared_miss_bytes') == 33*MIB
            assert n('hbm_bytes') == unique
        else:
            assert n('initial_shared_resident_bytes') == 33*MIB
            assert n('explicitly_seeded_bytes') == 33*MIB
            close(n('hbm_bytes'), unique - 33*MIB + n('resident_reload_bytes'))
            if row['cache_policy'] == 'pin_shared':
                assert n('shared_hit_fraction') == 1
                assert n('hbm_bytes') == unique-33*MIB
    primary = [r for r in rows if r['policy']=='fair_lru' and r['bandwidth_gbs']=='512'
               and r['cache_state']=='cold']
    compute = {}
    expected = {'single':.022383588, 'homo':.021391022, 'hetero':.021107933}
    for design in DESIGNS:
        batch = [gm(float(r['compute_ms']) for r in primary
                    if r['design']==design and int(r['batch'])==b) for b in BATCHES]
        compute[design] = gm(batch)
        close(compute[design], expected[design], tolerance=1e-9)
    floors = []
    for b in BATCHES:
        group = [w for w in windows if w['batch']==b]
        floors.append({'batch':b, 'windows':len(group),
                       'unique_weight_GM_MiB':gm(weight_bytes(w) for w in group)/MIB,
                       'cold_HBM_floor_GM_ms_at512':gm(weight_bytes(w)/512e6 for w in group),
                       'shared_warm_HBM_floor_GM_ms_at512':gm((weight_bytes(w)-33*MIB)/512e6 for w in group),
                       'exclusive_shared_pin_floor_speedup':gm(weight_bytes(w)/(weight_bytes(w)-33*MIB) for w in group)})
    paired = []
    for scenario in config['scenarios']:
        for state in config['cache_states']:
            subset = [r for r in rows if r['policy']==scenario and r['cache_state']==state
                      and r['bandwidth_gbs']=='512']
            lookup = {(r['window_id'],r['design']):r for r in subset}
            for baseline in ('single','homo'):
                by_batch = [gm(float(lookup[w['id'],baseline]['latency_ms']) /
                                  float(lookup[w['id'],'hetero']['latency_ms'])
                                  for w in windows if w['batch']==b) for b in BATCHES]
                paired.append({'policy':scenario, 'cache_state':state,
                               'baseline':baseline, 'hetero_speedup_7batch_GM':gm(by_batch),
                               'exceeds_15_percent':gm(by_batch)>=1.15})
    zero_latency = []
    cold_gaps = []
    for source, latency in ((rows,64), (sensitivity,0)):
        subset = [r for r in source if r['policy']=='fair_lru' and r['cache_state']=='cold'
                  and r['bandwidth_gbs']=='512']
        for design in DESIGNS:
            by_batch_latency = [gm(float(r['latency_ms']) for r in subset
                                  if r['design']==design and int(r['batch'])==b)
                                for b in BATCHES]
            by_batch_gap = [gm(float(r['total_cycles']) /
                              (weight_bytes(by_id[r['window_id']])/512)
                              for r in subset if r['design']==design and int(r['batch'])==b)
                            for b in BATCHES]
            cold_gaps.append({'request_latency_ns':latency,'design':design,
                              'latency_7batch_GM_ms':gm(by_batch_latency),
                              'ratio_to_compulsory_HBM_lower_bound':gm(by_batch_gap)})
    zero_lookup = {(r['window_id'],r['design'],r['cache_state']):r for r in sensitivity}
    for state in config['cache_states']:
        for baseline in ('single','homo'):
            ratios = [gm(float(zero_lookup[w['id'],baseline,state]['latency_ms']) /
                         float(zero_lookup[w['id'],'hetero',state]['latency_ms'])
                         for w in windows if w['batch']==b) for b in BATCHES]
            zero_latency.append({'cache_state':state,'baseline':baseline,
                                 'hetero_speedup_7batch_GM':gm(ratios)})
    timelines = []
    for design, cores in DESIGNS.items():
        for b in (2,128):
            window = next(w for w in windows if w['batch']==b)
            for cache_policy, initial, prefetch in (
                    ('lru','cold',True), ('lru','shared_resident',True),
                    ('pin_shared','shared_resident',True), ('lru','cold',False)):
                result = event_audit(cores,window,cache_policy=cache_policy,
                                     initial=initial,prefetch=prefetch)
                result['design'] = design
                timelines.append(result)
    receipt = {'status':'PASS', 'checked_rows':len(rows)+len(sensitivity),
               'windows':len(windows), 'compute_7batch_GM_ms':compute,
               'analytic_HBM_floors':floors, 'paired_comparisons':paired,
               'zero_macro_request_latency_comparisons':zero_latency,
               'cold_gap_to_HBM_lower_bound':cold_gaps,
               'event_replays':timelines,
               'sufficient_working_storage_ledger':working_storage,
               'exclusive_shared_pin_floor_7batch_speedup':gm(r['exclusive_shared_pin_floor_speedup'] for r in floors),
               'meaning':'Independent conservation and scalar/event consistency, not native-cycle or PPA calibration',
               'scope':'Finite shared DMA and weight capacity; ideal internal ports; frozen compute-optimal shapes',
               'hashes':{name:hashlib.sha256((out/name).read_bytes()).hexdigest()
                         for name in ('inputs.json','CONFIG.json','per_window.csv','latency_sensitivity.csv')}}
    (out/'INDEPENDENT_AUDIT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'PASS','checked_rows':len(rows)+len(sensitivity),
                      'event_replays':len(timelines),'compute_7batch_GM_ms':compute,
                      'paired_comparisons':paired},indent=2))


if __name__ == '__main__':
    main()
