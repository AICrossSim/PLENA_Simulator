"""Qualification must not turn subset or routed-only gains into N5 evidence."""
import q3_qualification as q
import pytest


def test_population_byte_equality_and_shared_cost_are_required():
    rows = []
    for layer in (2, 13, 26):
        for lanes in (8, 16):
            for bits in (4, 3):
                for method in ('qera_approx', 'qera_exact', 'lqer', 'l2qer'):
                    for start in range(0, 8192, 16):
                        for budget in (16, 32):
                            for policy in ('uniform', 'frequency_static', 'gate_weighted_budget_oracle', 'gate_weighted_causal'):
                                # An apparent21% routed-byte gain shrinks below
                                #20% after the required fixed Shared factors.
                                # One exact-byte window must not turn the
                                #remaining511 unequal windows into evidence.
                                byte_count = 790 if policy == 'gate_weighted_causal' and start else 1000
                                rows.append({'layer': str(layer), 'rank_lanes': str(lanes),
                                    'bits': str(bits), 'method': method, 'window_start': str(start),
                                    'uniform_rank': str(budget), 'strategy': policy,
                                    'factor_bytes': str(byte_count),
                                    'relative_error': '.8' if policy == 'gate_weighted_causal' else '1'})
    summaries = q.assess_rows(rows)
    assert len(summaries) == 192
    assert all(r['equal_error_20pct_routed_byte_diagnostic'] for r in summaries)
    assert not any(r['equal_error_20pct_total_factor_byte_pass'] for r in summaries)
    assert not any(r['equal_bytes_10pct_error_pass'] for r in summaries)
    assert q.physical_sha(q.DEFAULT) == 'd38b472f954f9dee23881b0c37549f3c1c21736347f3c358bbd466fbef5c54e1'
    with pytest.raises(ValueError, match='window coverage incomplete'):
        q.assess_rows(rows[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        q.assess_rows(rows + rows[:1])
