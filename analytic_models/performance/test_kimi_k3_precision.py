from __future__ import annotations

import math

from .kimi_k3_precision import run_kda_precision_experiment


def test_kda_precision_experiment_reports_finite_error_and_compression() -> None:
    report = run_kda_precision_experiment(
        tokens=8,
        num_heads=2,
        key_dim=128,
        value_dim=4,
    )
    by_storage = {result["storage"]: result for result in report["results"]}
    assert set(by_storage) == {"bf16", "fp16", "mx8_b128"}
    assert by_storage["bf16"]["compression_vs_fp32"] == 2.0
    assert by_storage["mx8_b128"]["compression_vs_fp32"] > 3.9
    for result in by_storage.values():
        for metric_group in ("state_error", "output_error"):
            assert all(math.isfinite(value) for value in result[metric_group].values())
