from __future__ import annotations

import math

from .nemotron3_precision import run_state_precision_experiment


def test_precision_experiment_reports_finite_errors_and_compression() -> None:
    report = run_state_precision_experiment(tokens=12, num_heads=4, head_dim=2, state_dim=128, groups=2)
    by_storage = {result["storage"]: result for result in report["results"]}
    assert set(by_storage) == {"bf16", "fp16", "mx8_b128"}
    assert by_storage["bf16"]["compression_vs_fp32"] == 2.0
    assert by_storage["mx8_b128"]["compression_vs_fp32"] > 3.9
    for result in by_storage.values():
        for metric_group in ("state_error", "output_error"):
            assert all(math.isfinite(value) for value in result[metric_group].values())
            assert result[metric_group]["relative_l2"] >= 0
