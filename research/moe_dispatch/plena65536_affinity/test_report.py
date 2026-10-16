"""Statistical and numerical contracts that prevent misleading result tables."""
import math

from .report import aggregate, gm, table


def row(*, cycles, a, b, batch=2, ident='w'):
    makespan = max(a, b)
    return dict(batch=batch, row_type='window', window_id=ident, window_count=1,
                compute_cycles=cycles, compute_ms=cycles/1e6,
                core0_ms=a/1e6, core1_ms=b/1e6,
                core0_idle_pct=100*(makespan-a)/makespan,
                core1_idle_pct=100*(makespan-b)/makespan,
                imbalance_ratio_finite=makespan/min(a,b) if min(a,b) else None,
                zero_load_core_windows=int(min(a,b)==0),
                core0_bottleneck_windows=int(a>b),core1_bottleneck_windows=int(b>a),
                tie_bottleneck_windows=int(a==b),
                bottleneck_core='0' if a>b else '1' if b>a else 'tie',
                array_full_scheduled_peak_TB_s=a,
                local_double_buffer_KiB=b,
                selected_policy='hard_threshold',selected_threshold=2)


def test_compute_makespan_is_aggregated_before_core_means():
    rows = [row(cycles=100,a=100,b=25),row(cycles=100,a=25,b=100)]
    result = aggregate(rows,2,'batch_GM')
    assert math.isclose(result['compute_ms'],100/1e6)
    assert math.isclose(result['core0_ms'],62.5/1e6)
    assert result['compute_ms'] != max(result['core0_ms'],result['core1_ms'])
    assert result['imbalance_ratio_finite']==4
    assert result['core0_idle_pct']==37.5
    assert result['array_full_scheduled_peak_TB_s']==100


def test_zero_load_is_counted_without_fabricated_finite_imbalance():
    rows = [row(cycles=100,a=100,b=0),row(cycles=100,a=25,b=100)]
    result = aggregate(rows,2,'batch_GM')
    assert result['zero_load_core_windows']==1
    assert result['imbalance_ratio_finite']==4
    result2 = aggregate([row(cycles=100,a=100,b=0)],2,'batch_GM')
    assert result2['imbalance_ratio_finite'] is None
    assert '—' in table([{'ratio':result2['imbalance_ratio_finite']}])


def test_seven_batch_weighting_does_not_weight_by_window_count():
    first=aggregate([row(cycles=4,a=4,b=4) for _ in range(27)],2,'batch_GM')
    second=aggregate([row(cycles=16,a=16,b=16) for _ in range(9)],64,'batch_GM')
    result=aggregate([first,second],'7-Batch GM','7batch_GM')
    assert result['compute_cycles']==gm([4,16])
    assert math.isclose(result['compute_cycles'],8)
    assert result['window_count']==36
    assert result['tie_bottleneck_windows']==36

