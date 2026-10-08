"""Check saved-value conversion and truthful figure labels, without simulation."""
import copy
import pytest
from .figure_clarify import curve_points, gpqa_figure, plt_module


def saved_segments():
    return [dict(start=0., end=1e6, hbm_rate_Bpc=0., hbm_rate_Bpc_core=[0., 0.], inflight_bytes=[0., 0.]),
            dict(start=1e6, end=2e6, hbm_rate_Bpc=32., hbm_rate_Bpc_core=[0., 32.], inflight_bytes=[0., 2080.])]


def test_saved_units_and_curve_values_are_preserved_without_mutating_input():
    segments = saved_segments(); before = copy.deepcopy(segments)
    points = curve_points(segments)
    assert points == dict(time_ms=[0., 1.], core0_proxy_KiB=[0., 0.],
                          core1_proxy_KiB=[0., 2.03125], global_HBM_GBps=[0., 32.])
    assert segments == before


def test_bad_proxy_and_nonconserved_global_rate_are_rejected():
    segments = saved_segments(); segments[1]["inflight_bytes"][1] = 2081.
    with pytest.raises(AssertionError, match="65ns service-equivalent proxy"):
        curve_points(segments)
    segments = saved_segments(); segments[1]["hbm_rate_Bpc"] = 64.
    with pytest.raises(AssertionError, match="global/core HBM rates differ"):
        curve_points(segments)


def test_pdf_plot_labels_explain_proxy_and_show_unchanged_saved_series():
    points = curve_points(saved_segments())
    fig = gpqa_figure({"H0": points})
    try:
        left, right = fig.axes
        text = "\n".join([t.get_text() for t in fig.texts] + [left.get_ylabel(), right.get_ylabel()])
        assert "65 ns" in text and "not outstanding-request occupancy" in text
        assert "exactly one positive supply stream" in text
        assert "not one core with in-flight requests" in text
        assert "KiB; proxy" in text and "GB/s; analytical" in text
        assert list(left.lines[1].get_xdata()) == points["time_ms"]
        assert list(left.lines[1].get_ydata()) == points["core1_proxy_KiB"]
        assert list(right.lines[0].get_ydata()) == points["global_HBM_GBps"]
    finally:
        plt_module().close(fig)
