#!/usr/bin/env python3
"""Conservative N5 quality assessment from complete physical-LUT FFN evidence.

No equal-byte subset or quality-selected window is promoted to a population
claim. The original per-window byte scope is routed factors only; a separate
total-factor ratio includes the always-present Shared factors.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path


DEFAULT = {'main_bits': 4, 'factor_a': 'mxint4', 'factor_b': 'bf16',
           'rank_lanes': 8, 'ranks': {'routed': [32, 32, 24],
                                     'shared': [32, 32, 48]}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values)


def format_for(bits, lanes):
    return {'main_bits': bits, 'factor_a': 'mxint4', 'factor_b': 'bf16',
            'rank_lanes': lanes,
            'ranks': {'routed': [4 * lanes, 4 * lanes, 3 * lanes],
                      'shared': [4 * lanes, 4 * lanes, 6 * lanes]}}


def physical_sha(config):
    # Match study.py canonical(): normalized JSON followed by one newline.
    return hashlib.sha256(json.dumps(config, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode() + b'\n').hexdigest()


def assess_rows(rows):
    groups = defaultdict(dict)
    for row in rows:
        group = tuple(row[k] for k in ('layer', 'rank_lanes', 'bits', 'method'))
        ident = (int(row['window_start']), int(row['uniform_rank']), row['strategy'])
        if ident in groups[group]:
            raise ValueError('duplicate physical Q3 row')
        groups[group][ident] = row
    wanted = {(str(l), str(L), str(b), m) for l in (2, 13, 26)
        for L in (8, 16) for b in (4, 3)
        for m in ('qera_approx', 'qera_exact', 'lqer', 'l2qer')}
    if set(groups) != wanted:
        raise ValueError('physical Q3 group coverage incomplete')
    comparisons = []
    for (layer, lanes, bits, method), population in sorted(groups.items()):
        keys = {(s, u, p) for s in range(0, 8192, 16) for u in (16, 32)
            for p in ('uniform', 'frequency_static', 'gate_weighted_budget_oracle',
                      'gate_weighted_causal')}
        if set(population) != keys:
            raise ValueError('physical Q3 window coverage incomplete')
        shared = {8: 698496, 16: 1396992}[int(lanes)]
        for baseline_rank in (16, 32):
            baseline = [population[(s, baseline_rank, 'uniform')]
                        for s in range(0, 8192, 16)]
            be = mean([float(r['relative_error']) for r in baseline])
            bb = mean([float(r['factor_bytes']) for r in baseline])
            for candidate_rank in (16, 32):
                candidate = [population[(s, candidate_rank, 'gate_weighted_causal')]
                             for s in range(0, 8192, 16)]
                ce = mean([float(r['relative_error']) for r in candidate])
                cb = mean([float(r['factor_bytes']) for r in candidate])
                equal = all(float(a['factor_bytes']) == float(b['factor_bytes'])
                            for a, b in zip(candidate, baseline))
                delta = [float(a['factor_bytes']) - float(b['factor_bytes'])
                         for a, b in zip(candidate, baseline)]
                comparisons.append({'layer': int(layer), 'rank_lanes': int(lanes),
                    'bits': int(bits), 'method': method,
                    'uniform_reference_rank': baseline_rank,
                    'causal_requested_budget_rank': candidate_rank,
                    'windows': 512, 'mean_window_error_ratio': ce / be,
                    'mean_uniform_window_error': be, 'mean_causal_window_error': ce,
                    'mean_routed_factor_bytes_uniform': bb,
                    'mean_routed_factor_bytes_causal': cb,
                    'shared_fixed_factor_bytes': shared,
                    'mean_routed_factor_byte_ratio': cb / bb,
                    'mean_total_factor_byte_ratio': (cb + shared) / (bb + shared),
                    'mean_routed_byte_delta': mean(delta),
                    'worst_routed_byte_overrun': max(delta),
                    'all_windows_exact_equal_bytes': equal,
                    'equal_bytes_10pct_error_pass': equal and ce <= .9 * be,
                    'equal_error_20pct_total_factor_byte_pass': ce <= be
                        and (cb + shared) <= .8 * (bb + shared),
                    'equal_error_20pct_routed_byte_diagnostic': ce <= be and cb <= .8 * bb,
                    'scope': 'entire512-window population, no post-hoc sample selection'})
    return comparisons


def build_summary(output, frozen=None):
    source = output / 'q3_hardware_bf16_actual_ffn.csv'
    receipt_path = output / 'q3_hardware_complete_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if not receipt['complete'] or receipt['coverage']['actual_ffn_rows'] != 196608:
        raise ValueError('hardware Q3 must be exhaustively completed')
    comparisons = assess_rows(list(csv.DictReader(source.open())))
    by_signature = defaultdict(list)
    for r in comparisons:
        signature = tuple(r[k] for k in ('rank_lanes', 'bits', 'method',
                            'uniform_reference_rank', 'causal_requested_budget_rank'))
        by_signature[signature].append(r)
    groups = []
    for key, population in sorted(by_signature.items()):
        if {r['layer'] for r in population} != {2, 13, 26} or len(population) != 3:
            raise ValueError('comparison is not complete across all three layers')
        lanes, bits, method, baseline, candidate = key
        equal_byte_pass = all(r['equal_bytes_10pct_error_pass'] for r in population)
        equal_error_pass = all(r['equal_error_20pct_total_factor_byte_pass'] for r in population)
        groups.append({'rank_lanes': lanes, 'main_bits': bits, 'method': method,
            'uniform_reference_rank': baseline, 'causal_requested_budget_rank': candidate,
            'physical_format': format_for(bits, lanes),
            'physical_format_sha256': physical_sha(format_for(bits, lanes)),
            'equal_bytes_error_pass_all_layers': equal_byte_pass,
            'equal_error_total_factor_bytes_pass_all_layers': equal_error_pass,
            'N5_quality_pass_for_this_group': equal_byte_pass or equal_error_pass})
    current = json.loads(frozen.read_text())['config'] if frozen and frozen.exists() else DEFAULT
    current_hash = physical_sha(current)
    applicable = [r for r in groups if r['physical_format_sha256'] == current_hash
                  and r['method'] == 'qera_approx']
    if not applicable:
        raise ValueError('current physical format has no matching hardware evidence')
    passed = any(r['N5_quality_pass_for_this_group'] for r in applicable)
    return {'schema': 'plena_v3_hardware_q3_quality_qualification_v1',
        'complete': True, 'N5_quality_pass': passed,
        'reason': 'matching physical format and QERA-approx meet a full-population three-layer quality criterion'
                  if passed else 'matching physical format and QERA-approx do not meet either full-population three-layer criterion',
        'physical_format': current, 'physical_format_sha256': current_hash,
        'method': 'qera_approx', 'LUT': 'BF16 RNE per projection, uint32 factor byte costs',
        'rank_scope': 'one common rank per expert; routed choices0/8/16/24/32, clipped at projection capacity; Shared fixed full',
        'policy': 'gate_weighted_causal; actual development-window lambda calibration, routed-only causal feedback',
        'byte_scope': 'CSV bytes are routed factors only; N5 equal-error factor-byte ratio adds fixed Shared factor bytes to both sides; main weights excluded',
        'full_population_vs_posthoc_subset': 'full512-window population for each group; exact-byte equality must hold in every window; no quality-selected subset',
        'metric': 'arithmetic mean of512 actual16-token FFN relative Frobenius errors; not full8192-row error, perplexity or downstream model accuracy',
        'thresholds': {'equal_bytes_error_max': .90, 'equal_error_total_factor_bytes_max': .80},
        'data_scope': 'three real development-validation layers2/13/26,8192 captured tokens; calibration request identities disjoint within development pool',
        'accuracy_qualification_separate': True,
        'source_receipt': str(receipt_path.resolve()), 'source_receipt_sha256': sha(receipt_path),
        'source_csv_sha256': sha(source), 'actual_ffn_rows': 196608,
        'matching_current_groups': applicable, 'all_group_diagnostics': groups,
        'comparisons': comparisons,
        'diagnostic_caveats': ['all requested budgets are retained; cross-budget group choice is a post-hoc diagnostic',
            'frequency_static is a calibration-frequency-importance budget oracle proxy, not a fixed physical rank table',
            'error gains at unequal bytes and exact-byte subsets are not population equal-byte evidence']}


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.checkpoint.{os.getpid()}')
    try:
        with temporary.open('w') as f:
            json.dump(value, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--frozen-format', type=Path); ap.add_argument('--wait', action='store_true')
    args = ap.parse_args(); target = args.output / 'q3_qualification_summary.json'
    while not (args.output / 'q3_hardware_complete_receipt.json').exists():
        if not args.wait:
            raise FileNotFoundError('complete hardware Q3 receipt unavailable')
        if not target.exists():
            atomic_json(target, {'schema': 'plena_v3_hardware_q3_quality_qualification_v1',
                'complete': False, 'N5_quality_pass': None,
                'reason': 'full three-layer physical-LUT actual-FFN evidence still running',
                'physical_format_sha256': physical_sha(DEFAULT),
                'byte_scope': 'routed-only CSV; final total-factor comparison includes Shared',
                'full_population_vs_posthoc_subset': 'pending full population; no partial or subset claim'})
        time.sleep(15)
    summary = build_summary(args.output, args.frozen_format)
    atomic_json(target, summary)
    print(json.dumps({'complete': True, 'N5_quality_pass': summary['N5_quality_pass'],
                      'path': str(target), 'sha256': sha(target)}), flush=True)


if __name__ == '__main__':
    main()
