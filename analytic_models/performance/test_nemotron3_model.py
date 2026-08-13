import json

from analytic_models.performance.nemotron3_model import _parser, build_document, main


def test_workload_cli_document_uses_real_hybrid_model() -> None:
    args = _parser().parse_args(["--mode", "workload", "--body-only"])
    document = build_document(args)

    assert document["calibration"]["status"] == "uncalibrated_no_gpu_or_rtl"
    assert document["workload"]["layer_counts"] == {"mamba": 23, "moe": 23, "attention": 6}


def test_sweep_cli_builds_only_valid_cache_combinations() -> None:
    args = _parser().parse_args(
        [
            "--mode",
            "sweep",
            "--body-only",
            "--decode-tokens",
            "2",
            "--sweep-layouts",
            "row_major,group_major_skewed",
            "--sweep-broadcasts",
            "0,1",
            "--sweep-cache-mib",
            "0,64",
            "--sweep-cache-policies",
            "none,lru",
            "--sweep-state-dim-lanes",
            "8",
        ]
    )
    document = build_document(args)

    assert document["design_count"] == 8
    assert all(result["metrics"]["calibrated"] is False for result in document["results"])
    assert document["results"] == sorted(document["results"], key=lambda result: result["metrics"]["total_cycles"])


def test_cli_writes_machine_readable_report(tmp_path) -> None:
    output = tmp_path / "report.json"
    status = main(["--mode", "dse", "--body-only", "--json-out", str(output)])

    assert status == 0
    parsed = json.loads(output.read_text())
    assert parsed["mode"] == "dse"
    assert len(parsed["results"]) == 1
