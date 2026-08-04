from __future__ import annotations

import math

import pytest

from .handoff import HandoffTime, evaluate_handoff_regimes


def _handoff() -> HandoffTime:
    return HandoffTime(
        wire_bytes=64e9,
        decode_cache_bytes=16e9,
        link_bw=450e9,
        transfer_bulk_s=64e9 / 450e9,
        transfer_streamed_s=(64e9 / 32) / 450e9,
        admission_s=0.02,
        admission_energy_j=0.05,
        admission_calibrated=False,
        admission_calibration_id=None,
    )


def _evaluate(generation_tokens: int = 128):
    return evaluate_handoff_regimes(
        _handoff(),
        prompt_tokens=4096,
        generation_tokens=generation_tokens,
        precision="MXINT4",
        prefill_latency_s=0.25,
        decode_tpot_s=0.002,
        decode_ready_delay_s=0.20,
        prefill_energy_j=4.0,
        decode_energy_per_token_j=0.08,
        prefill_stall_power_w=50.0,
        decode_idle_power_w=20.0,
    )


def test_handoff_regimes_account_for_stalls_and_host_round_trip() -> None:
    pipelined, back_pressure, host_buffered = _evaluate()
    assert pipelined.regime == "fully_pipelined"
    assert back_pressure.regime == "back_pressure"
    assert host_buffered.regime == "host_buffered"
    assert pipelined.prefill_utilization == 1.0
    assert back_pressure.prefill_utilization < 1.0
    assert host_buffered.prefill_utilization == 1.0
    assert back_pressure.ttft_s > pipelined.ttft_s
    assert host_buffered.ttft_s > back_pressure.ttft_s
    assert math.isclose(host_buffered.host_spill_s, 2.0, rel_tol=1e-12)
    assert host_buffered.energy_j > pipelined.energy_j
    assert all(result.energy_tier == "analytic_anchored" for result in _evaluate())


def test_balance_ratio_responds_to_generation_length() -> None:
    short = _evaluate(generation_tokens=32)[0]
    long = _evaluate(generation_tokens=256)[0]
    assert long.prefill_decode_ratio < short.prefill_decode_ratio
    assert math.isclose(
        short.prefill_decode_ratio,
        0.25 / (0.02 + 32 * 0.002),
        rel_tol=1e-12,
    )
    assert set(short.to_dict()) >= {
        "ttft_s",
        "energy_j",
        "prefill_utilization",
        "prefill_decode_ratio",
        "energy_tier",
    }


def test_handoff_regime_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="decode TPOT"):
        evaluate_handoff_regimes(
            _handoff(),
            prompt_tokens=4096,
            generation_tokens=128,
            precision="MXINT4",
            prefill_latency_s=0.25,
            decode_tpot_s=0.0,
            decode_ready_delay_s=0.0,
            prefill_energy_j=0.0,
            decode_energy_per_token_j=0.0,
        )
