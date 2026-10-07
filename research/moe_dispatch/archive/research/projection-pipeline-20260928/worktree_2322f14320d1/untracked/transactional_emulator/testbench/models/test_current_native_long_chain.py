"""Independent checks of SR order and BF16 store boundaries."""

import numpy as np

from transactional_emulator.testbench.models.current_native_long_chain import (
    LaneRandom, SEED, stochastic_bf16, update_reference,
)
from transactional_emulator.testbench.aten.recurrent_gate_test import bf


def test_lfsr_order_is_row_then_subgroup_and_continues_between_groups():
    rng = LaneRandom()
    actual = rng.thresholds(32, 3, 64)
    seeds = [SEED, (SEED + 17 * 0x9E3779B9) & 0xFFFFFFFF | 1]
    for row in range(3):
        for chunk in range(8):
            for j, lane in enumerate((0, 17)):
                linear = chunk * 256 + lane
                assert actual[linear // 64, row, linear % 64] == seeds[j] & 0xFFFF
                seeds[j] = (seeds[j] >> 1) ^ (0x80200003 if seeds[j] & 1 else 0)
    following = rng.thresholds(32, 1, 64)
    assert following[0, 0, 0] == seeds[0] & 0xFFFF
    assert following[0, 0, 17] == seeds[1] & 0xFFFF
    assert rng.draws_per_lane == 32


def test_sr_signed_values_match_integer_store_rule():
    values = np.asarray([1.003, -1.003, 0.000123, -0.000123], np.float32)
    thresholds = np.asarray([0, 65535, 1, 32768], np.uint16)
    actual = stochastic_bf16(values, thresholds).view(np.uint32)
    for i, bits in enumerate(values.view(np.uint32).tolist()):
        expected = (((bits >> 16) + (int(thresholds[i]) < (bits & 65535))) & 65535) << 16
        assert actual[i] == expected


def test_fused_update_does_not_round_decay_before_outer_term():
    state = np.ones((64, 128, 64), np.float32)
    delta = np.full(64, 1 / 512, np.float32)
    b = np.ones((64, 128), np.float32)
    # BF16(.004) is exactly .003997802734375. Rounding 1-delta first
    # produces 1; adding this outer term then crosses half a BF16 ulp at 1.
    # Keeping the decay until the final store leaves the value below it.
    x = np.full((64, 64), bf(np.float32(0.004)).item(), np.float32)
    dt = np.ones(64, np.float32)
    c = np.zeros((64, 128), np.float32)
    skip = np.zeros(64, np.float32)
    updated, output = update_reference(state, delta, b, x, dt, c, skip, "rn", LaneRandom())
    assert np.array_equal(updated, state)
    naive = bf(bf(state - bf(delta[:, None, None] * state)) + bf(b[:, :, None] * x[:, None, :]))
    assert np.array_equal(naive, np.full_like(state, 1.0078125))
    assert np.count_nonzero(output) == 0
