from pathlib import Path

import pytest

from compiler.aten.program_sink import TraceInstruction

from analytic_models.latency.timing import (
    IdealII1TimingProvider,
    MainTimingConfig,
    MainTimingProvider,
)


ROOT = Path(__file__).resolve().parents[3]


def _item(opcode: str, *, variant=()):
    return TraceInstruction(
        stage="decoder/layer/test",
        opcode=opcode,
        operands=(),
        variant=variant,
        active=None,
        sram=(),
        multiplicity=1,
    )


def test_main_config_matches_transactional_settings():
    config = MainTimingConfig.from_toml(ROOT / "plena_settings.toml")
    assert (config.mlen, config.blen, config.vlen) == (64, 4, 64)
    assert config.period_picos == 1_000
    assert config.vector_sum_cycles == 8


def test_main_provider_mirrors_emulator_special_cases():
    config = MainTimingConfig.from_toml(ROOT / "plena_settings.toml")
    provider = MainTimingProvider(config)
    assert provider.latency_picos(_item("M_MM"), {}) == 64_000
    assert provider.latency_picos(_item("M_MM_WO"), {}) == 1_000
    assert provider.latency_picos(_item("S_MAP_V_FP"), {}) == 128_000
    assert provider.latency_picos(
        _item("V_TOPK", variant=(("expert_count", "128"),)), {}
    ) == 512_000


def test_ideal_ii1_preserves_matrix_structural_timing():
    config = MainTimingConfig.from_toml(ROOT / "plena_settings.toml")
    provider = IdealII1TimingProvider(config)
    assert provider.latency_picos(_item("V_RED_SUM"), {}) == 1_000
    assert provider.latency_picos(_item("S_MAP_V_FP"), {}) == 1_000
    assert provider.latency_picos(_item("M_MM"), {}) == 64_000


def test_unknown_opcode_fails_closed():
    provider = MainTimingProvider(MainTimingConfig.from_toml(ROOT / "plena_settings.toml"))
    with pytest.raises(ValueError, match="no entry"):
        provider.latency_picos(_item("V_FUTURE"), {})
