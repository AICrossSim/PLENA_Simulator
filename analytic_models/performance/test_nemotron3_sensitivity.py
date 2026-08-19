from __future__ import annotations

from .nemotron3_sensitivity import build_report, render_markdown


def test_quick_sensitivity_sweep_has_all_three_levels() -> None:
    report = build_report(quick=True)
    assert report["projection_path"]["design_count"] == 128
    assert report["decode_system"]["design_count"] == 4
    assert report["prefill_candidates"]["design_count"] == 4
    # The real requirement is 23 layers x 1,097,728 B = 24.08 MiB. The swept
    # grid jumps 24 -> 32, so the grid point must not be reported as the
    # requirement.
    threshold = report["decode_system"]["full_cache_threshold"]["bf16"]
    assert threshold["entry_bytes"] == 1_097_728
    assert threshold["resident_entries"] == 23
    assert threshold["required_mib"] == 25
    assert threshold["smallest_swept_full_mib"] == 32
    markdown = render_markdown(report)
    assert "最小无停顿 FIFO" in markdown
    assert "BF16 Decode 决策点" in markdown
    assert "layer residency map" in markdown
    assert "不能从本轮得出的结论" in markdown
