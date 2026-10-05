"""Command line of the multi-chip model."""

from __future__ import annotations

import json

import pytest
from distributed_fixtures import EXAMPLES, GPT_OSS_20B, ISA_LIB, LLAMA_3_1_8B, SETTINGS, STACKED_DRAM_EXAMPLES

from analytic_models.distributed.__main__ import main


def _estimate_args(tmp_path, config: dict, noc: str, *extra: str) -> list[str]:
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(config))
    return [
        "estimate",
        "--model-path", str(model_path),
        "--config", str(SETTINGS),
        "--isa-lib", str(ISA_LIB),
        "--noc", str(EXAMPLES / f"{noc}.json"),
        "--batch-size", "8",
        "--input-seq", "128",
        "--output-seq", "2",
        *extra,
    ]  # fmt: skip


def test_describe_noc(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["describe-noc", "--noc", str(EXAMPLES / "illustrative_ring_switch_32.json")]) == 0
    text = capsys.readouterr().out
    assert "32 devices" in text and "ring" in text
    assert main(["describe-noc", "--noc", str(EXAMPLES / "illustrative_gpu_node_8.json"), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["num_devices"] == 8


def test_estimate_dense_with_a_memory_profile(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    profile = str(STACKED_DRAM_EXAMPLES / "fictional_stacked_dram.json")
    args = _estimate_args(tmp_path, LLAMA_3_1_8B, "illustrative_gpu_node_8", "--profile", profile, "--tp", "8")
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "tp=8 ep=1 dp=1 pp=1 cp=1 on 8 devices" in text
    assert "TTFT" in text and "NoC energy" in text
    assert main([*args, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["plan"]["tp"] == 8 and data["tps"] > 0


def test_estimate_moe_with_a_fixed_bandwidth(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    args = _estimate_args(
        tmp_path, GPT_OSS_20B, "illustrative_gpu_node_8",
        "--fixed-bandwidth-gbs", "1000", "--fixed-capacity-gib", "64",
        "--tp", "2", "--ep", "4", "--routing", "random", "--comm-overlap", "full",
    )  # fmt: skip
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "MoE             tp=1 ep=8 dp=1" in text
    assert "routing random" in text


def test_invalid_input_is_reported_without_a_traceback(tmp_path) -> None:
    node = "illustrative_gpu_node_8"
    with pytest.raises(SystemExit, match="error: the plan uses 4 devices but NoC profile"):
        main(_estimate_args(tmp_path, LLAMA_3_1_8B, node, "--fixed-bandwidth-gbs", "100", "--tp", "4"))
    with pytest.raises(SystemExit, match="requires --fixed-bandwidth-gbs"):
        main(_estimate_args(tmp_path, LLAMA_3_1_8B, node, "--profile", "x", "--fixed-capacity-gib", "1"))
