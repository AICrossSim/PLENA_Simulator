"""``python -m analytic_models.stacked_dram`` end to end."""

from __future__ import annotations

import json

import pytest
from stacked_dram_fixtures import EXAMPLES, ISA_LIB, LLAMA_3_1_8B, SETTINGS

from analytic_models.stacked_dram.__main__ import main

STACKED_EXAMPLE = str(EXAMPLES / "fictional_stacked_dram.json")


def test_describe_tabulates_stack_heights(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["describe", "--profile", STACKED_EXAMPLE, "--total-layers", "4,8", "--connected-layers", "2,8", "--json"])
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert [(row["total_layers"], row["connected_layers"]) for row in rows] == [(4, 2), (8, 2), (8, 8)]

    assert main(["describe", "--profile", STACKED_EXAMPLE]) == 0
    assert "fictional-stacked-dram" in capsys.readouterr().out


def test_estimate_with_a_profile_and_with_a_fixed_bandwidth(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    model = tmp_path / "llama-3.1-8b.json"
    model.write_text(json.dumps(LLAMA_3_1_8B))
    common = ["--model-path", str(model), "--config", str(SETTINGS), "--isa-lib", str(ISA_LIB)]
    common += ["--batch-size", "1", "--input-seq", "64", "--output-seq", "2"]

    assert main(["estimate", "--profile", STACKED_EXAMPLE, "--total-layers", "8", "--json", *common]) == 0
    stacked = json.loads(capsys.readouterr().out)
    assert stacked["memory"]["total_layers"] == 8 and stacked["memory"]["connected_layers"] == 8
    assert stacked["ttft_seconds"] > 0 and stacked["tps"] > 0

    assert main(["estimate", "--fixed-bandwidth-gbs", "100", "--fixed-capacity-gib", "64", *common]) == 0
    text = capsys.readouterr().out
    assert "TTFT" in text and "fits" in text


def test_invalid_input_is_reported_without_a_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="no --connected-layers value fits"):
        main(["describe", "--profile", STACKED_EXAMPLE, "--total-layers", "2", "--connected-layers", "4"])
    with pytest.raises(SystemExit, match="error: "):
        main(["describe", "--profile", "missing-profile.json"])


def test_layer_overrides_need_a_stacked_profile(tmp_path) -> None:
    model = tmp_path / "llama.json"
    model.write_text(json.dumps(LLAMA_3_1_8B))
    with pytest.raises(SystemExit):
        main(
            [
                "estimate",
                "--fixed-bandwidth-gbs",
                "100",
                "--total-layers",
                "4",
                "--model-path",
                str(model),
                "--config",
                str(SETTINGS),
                "--isa-lib",
                str(ISA_LIB),
            ]
        )
