"""E0: replay immutable old evidence using the new central parameters."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from .common import (ROOT, OLD, read_csv, write_csv, write_json, inputs,
                     decode_design, frozen_designs, canonical, digest, metadata)
from .config import MODES, parameters
from .runtime import simulate
from ..round2.predictors import Predictor
from ..round2.dispatch_fix.predictor_ablation_20261008.nominal_control import NominalPredictor


def specs():
    records = read_csv(OLD / 'results/E4/per_window.csv')
    groups = {}
    for r in records:
        if r['onchip_mode'] not in MODES or r['sched_type'] != 'runtime' or r['entry'] in ('U1', 'U2'):
            continue
        key = (r['entry'], r['onchip_mode'])
        groups.setdefault(key, []).append(r)
    result = [dict(source='round2_E4_' + mode, credits=256, mode=mode,
                   design=name, hardware=rows[0]['design'], method='eft_old', refs=rows,
                   warm=False) for (name, mode), rows in sorted(groups.items())]
    paths = [(256, OLD/'dispatch_fix/predictor_ablation_20261008/per_window.csv'),
             (256, OLD/'dispatch_fix/predictor_ablation_20261008/nominal_control/per_window.csv'),
             (520, OLD/'dispatch_fix/hbm256_sensitivity_20261008/per_window.csv')]
    groups = {}
    for credits, path in paths:
        for r in read_csv(path):
            if r['predictor'] not in ('ours', 'nominal'):
                continue
            key = credits, r['onchip_mode'], r['design'], r['predictor']
            groups.setdefault(key, []).append(r)
    for (credits, mode, name, method), rows in sorted(groups.items()):
        result.append(dict(source='7eb58061_' + mode, credits=credits, mode=mode,
                           design=name, hardware=None, method=method, refs=rows, warm=True))
    assert sum(s['warm'] for s in result) == 24
    return result


def job(spec):
    ws = inputs()
    by_id = {w['id']: w for w in ws['heldout']}
    p = parameters(spec['mode'], credits=spec['credits'])
    d = (decode_design(spec['hardware']) if spec['hardware'] else
         frozen_designs(spec['mode'], common_ws=True)[spec['design']])
    outputs, warms = [], []
    for _ in range(2):
        pred = (NominalPredictor() if spec['method'] == 'nominal' else
                Predictor('ours') if spec['method'] == 'ours' else None)
        kw = dict(dispatch='fixed_legacy' if spec['warm'] else 'eft_old',
                  predictor=pred, t_big=3, large_first=True)
        warm = [simulate(w, d, p, **kw) for w in ws['development']] if spec['warm'] else []
        warms.append(digest(warm))
        outputs.append([simulate(by_id[r['window_id']], d, p, **kw) for r in spec['refs']])
    assert canonical(outputs[0]) == canonical(outputs[1]) and warms[0] == warms[1]
    rows = []
    for ref, new in zip(spec['refs'], outputs[0]):
        diff = abs(new['latency_ms'] - float(ref['latency_ms']))
        assert diff == 0, (spec['source'], spec['design'], spec['method'], ref['window_id'], diff)
        assert new['hbm_bytes'] == float(ref['hbm_bytes'])
        rows.append(dict(source=spec['source'], credits=p.credits, design=spec['design'],
                         dispatch=spec['method'], window_id=ref['window_id'],
                         ms_ref=float(ref['latency_ms']), ms_new=new['latency_ms'], abs_diff=diff))
    receipt = {k:v for k,v in spec.items() if k not in ('refs', 'hardware')}
    receipt.update(windows=len(rows), repeats=2, warmup_digest=warms[0],
                   repeated_warmup_digest=warms[1], result_digest=digest(outputs[0]),
                   repeated_result_digest=digest(outputs[1]), exact=True)
    return rows, receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs', type=int, default=6)
    args = parser.parse_args()
    groups = specs()
    rows, receipts = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for spec, (part, receipt) in zip(groups, pool.map(job, groups)):
            rows.extend(part); receipts.append(receipt)
            print(dict(done=(spec['source'],spec['credits'],spec['design'],spec['method']),
                       windows=len(part), exact=True), flush=True)
    write_csv(ROOT/'E0/repro.csv', rows,
              ['source','credits','design','dispatch','window_id','ms_ref','ms_new','abs_diff'])
    write_json(ROOT/'E0/repeat_checks.json', receipts)
    write_json(ROOT/'E0/METADATA.json', metadata(dict(configurations=len(groups),
               reproduced_rows=len(rows), nonzero_differences=0, repeats=2,
               mode_interpretation='All permitted modes: task twelve pipelined points plus twelve port_tight points; fixed_issue excluded.')))
    print(dict(rows=len(rows), nonzero=0), flush=True)


if __name__ == '__main__':
    main()
