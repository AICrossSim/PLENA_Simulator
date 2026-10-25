"""``--device-num``: the devices are data-parallel replicas of one PLENA chip.

The busiest replica serves ``ceil(batch / device_num)`` sequences and sets TTFT
and the decode time; TPS counts the tokens of the whole batch.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
from pathlib import Path

import pytest

PERF_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PERF_DIR.parents[1]
SETTINGS = REPO_ROOT / "plena_settings.toml"
ISA_LIB = PERF_DIR / "customISA_lib.json"

# Public HuggingFace dimensions of Llama-3.1-8B and gpt-oss-20b, as in PLENA_Compiler's Model_Lib.
LLAMA_3_1_8B = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 32,
    "intermediate_size": 14336,
    "vocab_size": 128256,
}
GPT_OSS_20B = {
    "hidden_size": 2880,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "num_hidden_layers": 24,
    "intermediate_size": 2880,
    "vocab_size": 201088,
    "num_local_experts": 32,
    "experts_per_token": 4,
    "sliding_window": 128,
    "layer_types": ["sliding_attention", "full_attention"] * 12,
}


@pytest.fixture(scope="module")
def scripts():
    sys.path.insert(0, str(PERF_DIR))
    try:
        perf_model = importlib.import_module("perf_model")
        llama_model = importlib.import_module("llama_model")
        gpt_oss_model = importlib.import_module("gpt_oss_model")
    finally:
        sys.path.remove(str(PERF_DIR))
    hardware = perf_model.load_hardware_config_from_toml(str(SETTINGS))
    return {
        "llama": (llama_model.LLaMAModel, LLAMA_3_1_8B),
        "gpt_oss": (gpt_oss_model.GPTOssModel, GPT_OSS_20B),
    }, hardware


def performance(scripts, tmp_path, family: str, batch: int, devices: int) -> tuple[float, float]:
    models, hardware = scripts
    cls, config = models[family]
    config_path = tmp_path / f"{family}.json"
    config_path.write_text(json.dumps(config))
    model = cls(str(config_path), hardware, str(ISA_LIB), batch, 256, 3, device_num=devices)
    with contextlib.redirect_stdout(io.StringIO()):
        return model.compute_performance(verbose=False)


@pytest.mark.parametrize("family", ["llama", "gpt_oss"])
def test_replicas_multiply_throughput_and_keep_ttft(scripts, tmp_path, family) -> None:
    ttft_one, tps_one = performance(scripts, tmp_path, family, batch=2, devices=1)
    ttft, tps = performance(scripts, tmp_path, family, batch=8, devices=4)

    assert ttft == pytest.approx(ttft_one, rel=1e-12)
    assert tps == pytest.approx(4 * tps_one, rel=1e-12)


@pytest.mark.parametrize("family", ["llama", "gpt_oss"])
def test_more_devices_than_sequences_leaves_devices_idle(scripts, tmp_path, family) -> None:
    ttft_one, tps_one = performance(scripts, tmp_path, family, batch=1, devices=1)
    ttft, tps = performance(scripts, tmp_path, family, batch=3, devices=8)

    assert ttft == pytest.approx(ttft_one, rel=1e-12)
    assert tps == pytest.approx(3 * tps_one, rel=1e-12)


@pytest.mark.parametrize("family", ["llama", "gpt_oss"])
def test_uneven_batch_is_paced_by_the_busiest_device(scripts, tmp_path, family) -> None:
    ttft_three, tps_three = performance(scripts, tmp_path, family, batch=3, devices=1)
    ttft, tps = performance(scripts, tmp_path, family, batch=10, devices=4)

    assert ttft == pytest.approx(ttft_three, rel=1e-12)
    assert tps == pytest.approx(10 / 3 * tps_three, rel=1e-12)


def test_llada_replicas_keep_the_inference_time(scripts, tmp_path) -> None:
    models, hardware = scripts
    cls, config = models["llama"]
    config_path = tmp_path / "llama.json"
    config_path.write_text(json.dumps(config))
    one = cls(str(config_path), hardware, str(ISA_LIB), 2, 256, 3, device_num=1)
    four = cls(str(config_path), hardware, str(ISA_LIB), 8, 256, 3, device_num=4)

    time_one, rate_one = one.compute_llada_inference(512, 4, verbose=False)
    time_four, rate_four = four.compute_llada_inference(512, 4, verbose=False)

    assert time_four == pytest.approx(time_one, rel=1e-12)
    assert rate_four == pytest.approx(4 * rate_one, rel=1e-12)


@pytest.mark.parametrize("family", ["llama", "gpt_oss"])
def test_zero_devices_is_rejected(scripts, tmp_path, family) -> None:
    with pytest.raises(ValueError, match="device_num"):
        performance(scripts, tmp_path, family, batch=4, devices=0)
