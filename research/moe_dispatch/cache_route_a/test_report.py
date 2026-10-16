from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from .report import (
    SHARED_BYTES, aggregate, build_report, csv_write, gm, main_tables,
    paired_speedup, validate_rows,
)


def point(design='single', window='w0', batch=2, total=100_000, compute=10_000,
          cache_state='cold'):
    weight = 40 * 1024**2
    actual = weight if cache_state == 'cold' else weight - SHARED_BYTES
    return {
        'window_id': window, 'batch': batch, 'design': design,
        'policy': 'fair_lru', 'bandwidth_gbs': 512, 'cache_state': cache_state,
        'total_cycles': total, 'ideal_compute_cycles': compute,
        'stall_cycles': total - compute, 'hbm_bytes': actual,
        'unique_weight_bytes': weight, 'shared_request_bytes': SHARED_BYTES,
        'shared_hit_bytes': 0 if cache_state == 'cold' else SHARED_BYTES,
        'shared_evicted_bytes': 0, 'refetch_bytes': 0,
        'hbm_busy_cycles': actual / 512, 'core0_compute_cycles': compute,
        'core1_compute_cycles': 0,
    }


class ReportTests(unittest.TestCase):
    def test_geomean_zeros_and_invalid_inputs(self):
        self.assertEqual(gm([0, 2]), 0)
        self.assertAlmostEqual(gm([1, 4]), 2)
        for values in ([], [-1], [math.inf], [math.nan]):
            with self.assertRaises(ValueError):
                gm(values)

    def test_batches_equal_weight_despite_counts(self):
        rows = [point(window=f'w{i}', batch=2, total=100_000) for i in range(9)]
        rows.append(point(window='b4', batch=4, total=400_000))
        self.assertAlmostEqual(aggregate(rows, 'total_cycles'), 200_000)
        self.assertEqual(aggregate(rows, 'total_cycles', geometric=False), 250_000)

    def test_bootstrap_pairs_not_independent_samples(self):
        ref = [point(window=f'w{i}', total=x) for i, x in enumerate((100, 10000, 900000))]
        cand = [point(window=f'w{i}', total=x / 2) for i, x in enumerate((100, 10000, 900000))]
        point_estimate, lower, upper = paired_speedup(ref, cand)
        for value in (point_estimate, lower, upper):
            self.assertAlmostEqual(value, 2)

    def test_bootstrap_retains_equal_batch_weight(self):
        ref = [point(window=f'w{i}', batch=2, total=200) for i in range(9)]
        ref.append(point(window='b4', batch=4, total=800))
        cand = [point(window=f'w{i}', batch=2, total=100) for i in range(9)]
        cand.append(point(window='b4', batch=4, total=100))
        self.assertAlmostEqual(paired_speedup(ref, cand)[0], 4)
        self.assertEqual(paired_speedup(ref, cand), paired_speedup(ref, cand))

    def test_missing_pairs_and_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            paired_speedup([point(window='a')], [point(window='b')])
        with self.assertRaises(ValueError):
            paired_speedup([point(), point()], [point()])

    def test_full_heldout_required_unless_smoke(self):
        rows = [point(d) for d in ('single', 'homo', 'hetero')]
        self.assertEqual(validate_rows(rows, False)['points'], 3)
        with self.assertRaisesRegex(ValueError, 'Incomplete heldout'):
            validate_rows(rows)

    def test_cold_lower_bound_and_warm_initial_state(self):
        rows = [point(d, cache_state='shared_resident') for d in ('single', 'homo', 'hetero')]
        validate_rows(rows, False)
        rows[0]['cache_state'] = 'cold'
        with self.assertRaisesRegex(ValueError, 'unique-byte'):
            validate_rows(rows, False)

    def test_transfer_and_stall_bounds(self):
        for field, value, message in (
            ('total_cycles', 1, 'total must equal'),
            ('hbm_busy_cycles', 200_000, 'bus occupancy'),
            ('shared_hit_bytes', SHARED_BYTES + 1, 'hit bytes'),
        ):
            rows = [point(d) for d in ('single', 'homo', 'hetero')]
            rows[0][field] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_rows(rows, False)
        rows = [point(d, total=10_000, compute=1_000) for d in ('single', 'homo', 'hetero')]
        for r in rows:
            r['hbm_busy_cycles'] = 100
        with self.assertRaisesRegex(ValueError, 'byte/bandwidth'):
            validate_rows(rows, False)

    def test_duplicate_and_design_pairing(self):
        rows = [point(d) for d in ('single', 'homo', 'hetero')]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            validate_rows(rows + [point()], False)
        with self.assertRaisesRegex(ValueError, 'Missing design'):
            validate_rows(rows[:2], False)
        rows[-1]['window_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            validate_rows(rows, False)

    def test_breakdown_am_additivity_not_independent_gm_subtraction(self):
        rows = []
        for design in ('single', 'homo', 'hetero'):
            rows.extend([point(design, 'a', total=100_000, compute=10_000),
                         point(design, 'b', total=400_000, compute=390_000)])
        main, breakdown = main_tables(rows)
        aggregate_row = next(r for r in breakdown if r['design'] == 'single' and r['batch'] == '7-Batch equal AM')
        self.assertAlmostEqual(aggregate_row['compute_ms_AM'], .2)
        self.assertAlmostEqual(aggregate_row['stall_ms_AM'], .05)
        self.assertAlmostEqual(aggregate_row['total_ms_AM'], .25)
        self.assertAlmostEqual(aggregate_row['total_ms_GM'], .2)
        self.assertAlmostEqual(main[-1]['hetero_speedup_vs_homo'], 1)

    def test_payload_bandwidth_distinct_from_dma_and_initial_reload(self):
        rows = [point(d) for d in ('single', 'homo', 'hetero')]
        for row in rows:
            row['hbm_busy_cycles'] = 99_000
            row['refetch_bytes'] = 1024**2
            row['resident_reload_bytes'] = 2 * 1024**2
        _, breakdown = main_tables(rows)
        out = breakdown[0]
        self.assertAlmostEqual(out['dma_occupied_pct_AM'], 99)
        self.assertAlmostEqual(out['hbm_payload_bw_util_pct_AM'], 81.92)
        self.assertEqual(out['refetch_MiB_AM'], 1)
        self.assertEqual(out['initial_resident_reload_MiB_AM'], 2)
        self.assertEqual(out['total_reload_MiB_AM'], 3)

    def test_complete_report_outputs_scope_and_conditional_claim(self):
        rows = [point(d) for d in ('single', 'homo', 'hetero')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            csv_write(path / 'per_window.csv', rows)
            audit = build_report(path / 'per_window.csv', path / 'report', full_heldout=False)
            self.assertEqual(audit['points'], 3)
            self.assertFalse(audit['physical_area_claim'])
            for name in ('main.csv', 'breakdown.csv', 'policy_ablation.csv',
                         'sensitivity.csv', 'REPORT.md', 'REPORT_AUDIT.json'):
                self.assertTrue((path / 'report' / name).is_file(), name)
            text = (path / 'report' / 'REPORT.md').read_text()
            self.assertIn('未达到 1.15×', text)
            self.assertIn('不是 GPU 实测', text)
            self.assertIn('不能归因于异构几何本身', text)


if __name__ == '__main__':
    unittest.main()
