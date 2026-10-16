"""Replay a stronger common issue policy on predefined representative windows.

Keep hardware and whole-expert ownership frozen. This is a feasible scheduling
diagnostic, not a replacement geometry search or a global schedule optimum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ready_pool import schedule_ready_pool
from .search import BATCHES, Shape, canonical, csv_out, dump
from .trace_inputs import load_primary_inputs, load_qwen_inputs


LABELS = ('best_single', 'best_homogeneous', 'best_heterogeneous')


def run(out):
    out = Path(out)
    frozen = json.loads((out / 'FROZEN_SELECTION.json').read_text())['configs']
    sources = {'DeepSeek-V2-Lite': load_primary_inputs('heldout')['heldout'],
               'Qwen3-30B-A3B': load_qwen_inputs('heldout')['heldout']}
    rows, details = [], []
    for model, windows in sources.items():
        # First stored window in each batch, declared before inspecting outcomes.
        representatives = [next(w for w in windows if w['batch'] == b) for b in BATCHES]
        for label in LABELS:
            prefix = 'heldout' if model == 'DeepSeek-V2-Lite' else 'qwen'
            baseline = {r['window_id']: r for r in json.loads((out / f'{prefix}_{label}.json').read_text())}
            cores = tuple(Shape(**s) for s in frozen[label])
            for w in representatives:
                old = baseline[w['id']]
                experts = {e['id']: e for e in w['experts']}
                jobs = [[experts[a['expert_id']] for a in old['assignments'] if a['owner'] == c]
                        for c in range(len(cores))]
                first = [schedule_ready_pool(es, c) for es, c in zip(jobs, cores)]
                second = [schedule_ready_pool(es, c) for es, c in zip(jobs, cores)]
                assert canonical(first) == canonical(second)
                assert [r['whole_expert_baseline_cycles'] for r in first] == old['core_cycles']
                assert sum(r['issued_macs'] for r in first) == old['issued_macs']
                assert sum(r['useful_macs'] for r in first) == old['useful_macs']
                pooled = max(r['cycles'] for r in first)
                fallback = max(r['best_feasible_cycles'] for r in first)
                row = {'model': model, 'batch': w['batch'], 'window_id': w['id'],
                       'design': label, 'geometry': old['geometry'],
                       'whole_expert_cycles': old['cycles'], 'ready_pool_cycles': pooled,
                       'best_feasible_cycles': fallback, 'best_feasible_ms': fallback / 1e6,
                       'serial_over_best_feasible': old['cycles'] / fallback,
                       'pool_partial_peak_KiB': sum(r['peak_live_partial_bytes'] for r in first) / 1024,
                       'whole_expert_partial_peak_KiB': sum(old['peak_FP32_partial_bytes_per_core']) / 1024,
                       'pooled_issue_descriptor_count': sum(r['installed_record_descriptors_if_precreated'] for r in first),
                       'two_repeats_identical': True}
                rows.append(row)
                details.append({'row': row, 'per_core': first})
                print(canonical(row), flush=True)
    csv_out(out / 'ready_pool_diagnostic.csv', rows)
    dump(out / 'READY_POOL_DIAGNOSTIC.json', {
        'scope': 'one predefined representative per batch and model; frozen ownership and geometry; feasible policy only',
        'repeats': 2, 'case_count': len(rows), 'cases': details,
        'not_included_in_main_search': True,
        'supply_warning': 'Interleaving changes locality and live contexts. Main OS reverse-supply traffic is not transferred to this schedule.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
