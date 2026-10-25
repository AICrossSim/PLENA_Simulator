"""Memory-aware decoder latency on one PLENA chip."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import replace

import pytest
from stacked_dram_fixtures import ISA_LIB, LLAMA_3_1_8B, REPO_ROOT, SETTINGS, fictional_dram

from analytic_models.performance.perf_model import PerfModel, load_hardware_config_from_toml
from analytic_models.stacked_dram import (
    DecoderShape,
    DramEnergyConfig,
    FixedBandwidthMemory,
    HbmStoragePrecision,
    StackedDramModel,
    decode_block_traffic,
    element_bits,
    estimate_decoder_latency,
)

UNBOUNDED = FixedBandwidthMemory(name="unbounded", bandwidth_bytes_per_s=1e30, capacity_bytes=None)


@pytest.fixture(scope="module")
def perf() -> PerfModel:
    return PerfModel(load_hardware_config_from_toml(SETTINGS), str(ISA_LIB))


@pytest.fixture(scope="module")
def precision() -> HbmStoragePrecision:
    return HbmStoragePrecision.from_settings(SETTINGS)


@pytest.fixture(scope="module")
def llama() -> DecoderShape:
    return DecoderShape.from_hf_config(LLAMA_3_1_8B, name="llama-3.1-8b")


def test_storage_precision_reads_the_analytic_hbm_formats(precision: HbmStoragePrecision) -> None:
    # The shipped settings store weights, KV and activations as MXFP8: an e4m3
    # element plus one e8m0 scale per block of eight.
    mxfp8 = (8 + 8 / 8) / 8
    assert precision.weight_bytes == pytest.approx(mxfp8)
    assert precision.kv_read_bytes == pytest.approx(mxfp8)
    assert precision.kv_write_bytes == pytest.approx(mxfp8)
    assert precision.activation_bytes == pytest.approx(mxfp8)


@pytest.mark.parametrize(
    ("spec", "bits"),
    [
        ({"format": "Plain", "DATA_TYPE": {"type": "Fp", "sign": True, "exponent": 8, "mantissa": 7}}, 16),
        ({"format": "Plain", "DATA_TYPE": {"type": "Int", "width": 32}}, 32),
        (
            {
                "format": "Mx",
                "block": 32,
                "ELEM": {"type": "Fp", "sign": True, "exponent": 2, "mantissa": 1},
                "SCALE": {"type": "Fp", "sign": False, "exponent": 8, "mantissa": 0},
            },
            4 + 8 / 32,
        ),
        ({"type": "Fp", "sign": True, "exponent": 5, "mantissa": 10}, 16),
    ],
)
def test_element_bits(spec: dict, bits: float) -> None:
    assert element_bits(spec) == pytest.approx(bits)


def test_unbounded_bandwidth_reproduces_llama_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path, perf, precision, llama
) -> None:
    config_path = tmp_path / "llama-3.1-8b.json"
    config_path.write_text(json.dumps(LLAMA_3_1_8B))
    monkeypatch.syspath_prepend(str(REPO_ROOT / "analytic_models" / "performance"))
    from llama_model import LLaMAModel
    from perf_model import load_hardware_config_from_toml as load_script_config

    for batch, seq, out in [(4, 2048, 16), (1, 512, 3), (8, 4096, 2)]:
        reference = LLaMAModel(
            str(config_path),
            load_script_config(str(SETTINGS)),
            str(ISA_LIB),
            batch_size=batch,
            input_seq_len=seq,
            output_seq_len=out,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            ttft, tps = reference.compute_performance(verbose=False)
        estimate = estimate_decoder_latency(
            llama, perf, precision, UNBOUNDED, batch_size=batch, input_seq_len=seq, output_seq_len=out
        )
        assert estimate.ttft_seconds == pytest.approx(ttft, rel=1e-12)
        assert estimate.tps == pytest.approx(tps, rel=1e-12)
        assert estimate.decode.memory_bound_seconds == 0.0


def test_memory_bound_decode_is_traffic_over_bandwidth(perf, precision, llama) -> None:
    bandwidth = 1e6  # slow enough that compute is negligible
    memory = FixedBandwidthMemory(name="slow", bandwidth_bytes_per_s=bandwidth, capacity_bytes=None)
    batch, seq, out = 2, 128, 3
    estimate = estimate_decoder_latency(
        llama, perf, precision, memory, batch_size=batch, input_seq_len=seq, output_seq_len=out
    )

    expected = 0.0
    for step in range(out):
        reads, writes = decode_block_traffic(llama, precision, batch_size=batch, kv_len=seq + step)
        expected += llama.num_hidden_layers * (sum(reads.values()) + sum(writes.values())) / bandwidth
    assert estimate.decode.seconds == pytest.approx(expected)
    assert estimate.decode.memory_bound_seconds == pytest.approx(estimate.decode.seconds)
    assert estimate.tps == pytest.approx(batch * out / expected)


def test_serial_overlap_is_never_faster_than_the_roofline(perf, precision, llama) -> None:
    memory = FixedBandwidthMemory(name="mid", bandwidth_bytes_per_s=5e11, capacity_bytes=None)
    kwargs = {"batch_size": 4, "input_seq_len": 512, "output_seq_len": 4}
    roofline = estimate_decoder_latency(llama, perf, precision, memory, **kwargs)
    serial = estimate_decoder_latency(llama, perf, precision, memory, overlap_policy="serial", **kwargs)
    assert serial.ttft_seconds >= roofline.ttft_seconds
    assert serial.decode.seconds == pytest.approx(serial.decode.compute_seconds + serial.decode.memory_seconds)
    with pytest.raises(ValueError):
        estimate_decoder_latency(llama, perf, precision, memory, overlap_policy="max", **kwargs)


def test_stacked_dram_prices_traffic_with_its_usable_bandwidth(perf, precision, llama) -> None:
    stacked = StackedDramModel(name="fictional", config=fictional_dram(connected_layers=8))
    fixed = FixedBandwidthMemory(
        name="same-bandwidth",
        bandwidth_bytes_per_s=stacked.usable_bandwidth_bytes_per_s,
        capacity_bytes=stacked.capacity_bytes,
    )
    kwargs = {"batch_size": 4, "input_seq_len": 512, "output_seq_len": 4}
    via_stack = estimate_decoder_latency(llama, perf, precision, stacked, **kwargs)
    via_fixed = estimate_decoder_latency(llama, perf, precision, fixed, **kwargs)
    assert via_stack.ttft_seconds == pytest.approx(via_fixed.ttft_seconds)
    assert via_stack.tps == pytest.approx(via_fixed.tps)
    assert via_stack.memory["kind"] == "stacked_dram"


def test_wave_quantisation_only_adds_traffic(perf, precision, llama) -> None:
    exact = StackedDramModel(name="fictional", config=fictional_dram())
    quantised = replace(exact, config=fictional_dram(apply_wave_quantization=True))
    kwargs = {"batch_size": 1, "input_seq_len": 64, "output_seq_len": 2}
    base = estimate_decoder_latency(llama, perf, precision, exact, **kwargs)
    rounded = estimate_decoder_latency(llama, perf, precision, quantised, **kwargs)
    assert rounded.decode.read_bytes >= base.decode.read_bytes
    assert rounded.decode.write_bytes > base.decode.write_bytes


def test_capacity_and_energy_are_reported(perf, precision, llama) -> None:
    memory = FixedBandwidthMemory(
        name="tiny",
        bandwidth_bytes_per_s=1e12,
        capacity_bytes=2**30,
        energy=DramEnergyConfig(read_pj_per_bit=2.47, write_pj_per_bit=2.71),
    )
    estimate = estimate_decoder_latency(
        llama, perf, precision, memory, batch_size=1, input_seq_len=64, output_seq_len=1
    )
    assert estimate.fits_in_memory is False
    assert any("exceed the memory capacity" in warning for warning in estimate.warnings)
    expected_pj = 8 * (estimate.decode.read_bytes * 2.47 + estimate.decode.write_bytes * 2.71)
    assert estimate.decode.dram_energy_pj == pytest.approx(expected_pj)
    assert json.loads(json.dumps(estimate.to_dict()))["fits_in_memory"] is False


def test_lm_head_adds_its_weights_to_every_token(perf, precision, llama) -> None:
    kwargs = {"batch_size": 1, "input_seq_len": 64, "output_seq_len": 3}
    without = estimate_decoder_latency(llama, perf, precision, UNBOUNDED, **kwargs)
    with_head = estimate_decoder_latency(llama, perf, precision, UNBOUNDED, include_lm_head=True, **kwargs)
    head_bytes = llama.vocab_size * llama.hidden_size * precision.weight_bytes
    assert with_head.decode.read_bytes - without.decode.read_bytes == pytest.approx(3 * head_bytes)
    assert with_head.decode.compute_seconds > without.decode.compute_seconds


def test_shapes_are_read_from_hf_configs() -> None:
    qwen_like = dict(LLAMA_3_1_8B, hidden_size=5120, num_attention_heads=64, head_dim=128)
    shape = DecoderShape.from_hf_config(qwen_like, name="wide-heads")
    assert shape.head_dim == 128 and shape.q_width == 8192
    with pytest.raises(ValueError, match="mixture-of-experts"):
        DecoderShape.from_hf_config(dict(LLAMA_3_1_8B, num_local_experts=32), name="moe")
    with pytest.raises(ValueError):
        DecoderShape.from_hf_config(dict(LLAMA_3_1_8B, num_key_value_heads=5), name="uneven")


def test_projection_shape_mismatch_is_flagged(perf, precision) -> None:
    shape = DecoderShape.from_hf_config(
        dict(LLAMA_3_1_8B, num_hidden_layers=1, num_attention_heads=32, head_dim=64), name="narrow-heads"
    )
    estimate = estimate_decoder_latency(
        shape, perf, precision, UNBOUNDED, batch_size=1, input_seq_len=8, output_seq_len=1
    )
    assert any("head_dim" in warning for warning in estimate.warnings)
