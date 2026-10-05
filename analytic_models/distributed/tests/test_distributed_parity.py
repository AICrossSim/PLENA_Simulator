"""One device reproduces PLENA's single-chip estimates."""

from __future__ import annotations

import contextlib
import io
import json

import pytest
from distributed_fixtures import (
    FICTIONAL_HBM,
    GPT_OSS_20B,
    ISA_LIB,
    LLAMA_3_1_8B,
    REPO_ROOT,
    SETTINGS,
    STACKED_DRAM_EXAMPLES,
    UNBOUNDED,
    gpt_oss,
    llama,
    perf_model,
    precision,
    switch_noc,
)

from analytic_models.distributed import ParallelPlan, estimate_distributed
from analytic_models.stacked_dram import DecoderShape, estimate_decoder_latency, load_memory_profile


@pytest.fixture(scope="module")
def perf():
    return perf_model()


@pytest.fixture(scope="module")
def prec():
    return precision()


@pytest.mark.parametrize("memory_name", ["unbounded", "fixed", "stacked"])
@pytest.mark.parametrize("include_lm_head", [False, True])
def test_llama_on_one_device_matches_the_single_chip_estimate(perf, prec, memory_name, include_lm_head) -> None:
    memory = {
        "unbounded": UNBOUNDED,
        "fixed": FICTIONAL_HBM,
        "stacked": load_memory_profile(STACKED_DRAM_EXAMPLES / "fictional_stacked_dram.json"),
    }[memory_name]
    shape = DecoderShape.from_hf_config(LLAMA_3_1_8B, name="llama-3.1-8b")
    kwargs = {"batch_size": 4, "input_seq_len": 1024, "output_seq_len": 6, "include_lm_head": include_lm_head}
    single = estimate_decoder_latency(shape, perf, prec, memory, **kwargs)
    result = estimate_distributed(llama(), ParallelPlan(), switch_noc(1), perf, prec, memory, **kwargs)

    assert result.ttft_seconds == pytest.approx(single.ttft_seconds, rel=1e-12)
    assert result.tps == pytest.approx(single.tps, rel=1e-12)
    assert result.tps_per_sequence == pytest.approx(single.tps / 4, rel=1e-12)
    assert result.weight_bytes_per_device == pytest.approx(single.weight_footprint_bytes, rel=1e-12)
    assert result.kv_cache_bytes_per_device == pytest.approx(single.kv_cache_footprint_bytes, rel=1e-12)
    prefill = result.prefill.stages[0]
    assert prefill.read_bytes == pytest.approx(single.prefill.read_bytes, rel=1e-12)
    assert prefill.write_bytes == pytest.approx(single.prefill.write_bytes, rel=1e-12)
    assert prefill.comm_seconds == {}
    assert result.noc_energy_pj_per_decode_token == 0.0


def test_unbounded_bandwidth_reproduces_llama_model(monkeypatch, tmp_path, perf, prec) -> None:
    config_path = tmp_path / "llama-3.1-8b.json"
    config_path.write_text(json.dumps(LLAMA_3_1_8B))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "analytic_models" / "performance"))
    from llama_model import LLaMAModel
    from perf_model import load_hardware_config_from_toml as load_script_config

    for batch, seq, out in [(4, 2048, 4), (1, 512, 3)]:
        reference = LLaMAModel(str(config_path), load_script_config(str(SETTINGS)), str(ISA_LIB), batch, seq, out)
        with contextlib.redirect_stdout(io.StringIO()):
            ttft, tps = reference.compute_performance(verbose=False)
        result = estimate_distributed(
            llama(), ParallelPlan(), switch_noc(1), perf, prec, UNBOUNDED,
            batch_size=batch, input_seq_len=seq, output_seq_len=out,
        )  # fmt: skip
        assert result.ttft_seconds == pytest.approx(ttft, rel=1e-12)
        assert result.tps == pytest.approx(tps, rel=1e-12)


def test_unbounded_bandwidth_reproduces_gpt_oss_model(monkeypatch, tmp_path, perf, prec) -> None:
    config_path = tmp_path / "gpt-oss-20b.json"
    config_path.write_text(json.dumps(GPT_OSS_20B))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "analytic_models" / "performance"))
    from gpt_oss_model import GPTOssModel
    from perf_model import load_hardware_config_from_toml as load_script_config

    for batch, seq, out in [(4, 1024, 4), (1, 100, 3), (8, 2048, 2)]:
        reference = GPTOssModel(str(config_path), load_script_config(str(SETTINGS)), str(ISA_LIB), batch, seq, out)
        with contextlib.redirect_stdout(io.StringIO()):
            ttft, tps = reference.compute_performance(verbose=False)
        result = estimate_distributed(
            gpt_oss(), ParallelPlan(), switch_noc(1), perf, prec, UNBOUNDED,
            batch_size=batch, input_seq_len=seq, output_seq_len=out,
        )  # fmt: skip
        assert result.ttft_seconds == pytest.approx(ttft, rel=1e-12)
        assert result.tps == pytest.approx(tps, rel=1e-12)
        assert result.moe["prefill"]["imbalance"] == 1.0
