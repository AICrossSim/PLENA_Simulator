"""Precision, independent scalar reduction, cancellation and tail checks."""
import json

import numpy as np
import pytest

from .compute import Core, PK_VALUES
from .numerical import (bf16, bf16_bits, cancellation_counterexample,
                                  dot_tree, error_statistics, projection_bf16_fp32,
                                  reference_fp64, validation_report, write_report)


def scalar_tree_reference(x, w, pk):
    """Independent per-output scalar tree/segment oracle with FP32 rounding."""
    xq, wq = bf16(x), bf16(w)
    out = np.zeros((len(xq), len(wq)), dtype=np.float32)
    for m in range(len(xq)):
        for n in range(len(wq)):
            acc = np.float32(0)
            for first in range(0, xq.shape[1], pk):
                values = [np.float32(xq[m, k] * wq[n, k])
                          for k in range(first, min(first + pk, xq.shape[1]))]
                values += [np.float32(0)] * (pk - len(values))
                while len(values) > 1:
                    values = [np.float32(values[i] + values[i + 1])
                              for i in range(0, len(values), 2)]
                acc = np.float32(acc + values[0])
            out[m, n] = acc
    return out


def test_bf16_round_nearest_ties_even_zero_and_special_values():
    values = np.array([1 + 1 / 256, 1 + 3 / 256, -0.0, np.inf, -np.inf], dtype=np.float32)
    actual = bf16(values)
    assert actual[0] == 1
    assert actual[1] == 1 + 1 / 64
    assert np.signbit(actual[2])
    assert np.isposinf(actual[3]) and np.isneginf(actual[4])
    tiny_nan_payload = np.array([0x7F800001], dtype=np.uint32).view(np.float32)
    assert np.isnan(bf16(tiny_nan_payload))[0]
    assert bf16_bits(np.array([2 ** 24, 1, -(2 ** 24), -1], dtype=np.float32)).dtype == np.uint16


def test_balanced_tree_has_full_physical_zero_tail_and_fp32_additions():
    p = np.array([2 ** 24, 1, -(2 ** 24), -1], dtype=np.float32)
    assert dot_tree(p, 32) == 0
    serial = np.float32(0)
    for value in p:
        serial = np.float32(serial + value)
    assert serial == -1
    assert dot_tree(np.pad(p, (0, 28)), 32) == dot_tree(p, 32)
    with pytest.raises(ValueError):
        dot_tree(np.ones(33, dtype=np.float32), 32)


def test_same_precision_different_pk_counterexample_at_two_power24():
    x, w = cancellation_counterexample()
    np.testing.assert_array_equal(bf16(x), x)
    np.testing.assert_array_equal(bf16(w), w)
    assert reference_fp64(x, w)[0, 0] == 0
    results = {pk: projection_bf16_fp32(x, w, Core(1, 1, pk))[0, 0] for pk in PK_VALUES}
    assert results == {32: -1, 64: 0, 128: 0, 256: 0, 512: 0, 1024: 0}
    assert projection_bf16_fp32(x, w, Core(1, 1, 32), True)[0, 0] == -1


def test_random_matrices_match_independent_scalar_oracle_with_m_n_k_tails():
    rng = np.random.default_rng(9217)
    for m, n, k in ((1, 1, 1), (5, 7, 67), (3, 4, 1031)):
        x = rng.normal(size=(m, k)).astype(np.float32)
        w = rng.normal(size=(n, k)).astype(np.float32)
        for pk in PK_VALUES:
            out = projection_bf16_fp32(x, w, Core(2, 3, pk))
            gold = scalar_tree_reference(x, w, pk)
            np.testing.assert_array_equal(out.view(np.uint32), gold.view(np.uint32))
            assert out.shape == (m, n)
            stats = error_statistics(out, reference_fp64(x, w))
            assert stats["relative_l2_error"] < 1e-5
            np.testing.assert_array_equal(projection_bf16_fp32(x, w, Core(2, 3, pk), True), bf16(gold))


def test_pm_pn_shape_does_not_change_per_output_reduction_order():
    rng = np.random.default_rng(198)
    x = rng.normal(size=(5, 83)).astype(np.float32)
    w = rng.normal(size=(7, 83)).astype(np.float32)
    for pk in PK_VALUES:
        narrow = projection_bf16_fp32(x, w, Core(1, 1, pk))
        wide = projection_bf16_fp32(x, w, Core(4, 9, pk))
        np.testing.assert_array_equal(narrow.view(np.uint32), wide.view(np.uint32))


def test_fp64_reference_uses_quantized_operands_and_statistics_zero_case():
    x = np.array([[1.003]], dtype=np.float32)
    w = np.array([[1.003]], dtype=np.float32)
    assert reference_fp64(x, w)[0, 0] == 1
    assert float((x.astype(np.float64) @ w.astype(np.float64).T)[0, 0]) != 1
    zero = error_statistics(np.zeros((1, 1)), np.zeros((1, 1)))
    assert zero["relative_l2_error"] == zero["max_absolute_error"] == 0


def test_seeded_report_has_all_pk_differences_errors_and_stable_json(tmp_path):
    report = validation_report()
    assert report["same_precision_does_not_require_bitwise_equivalence"]
    assert report["counterexample"]["fp32_outputs_by_pk"]["32"] == -1
    assert len(report["random_matrix_cases"]) == 3
    assert all([r["pk"] for r in c["results_by_pk"]] == list(PK_VALUES)
               for c in report["random_matrix_cases"])
    assert any(r["fp32_different_bits_vs_pk512"]
               for c in report["random_matrix_cases"] for r in c["results_by_pk"])
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_report(first)
    write_report(second)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text()) == report


def test_invalid_shapes_or_nonfinite_inputs_fail():
    core = Core(1, 1, 64)
    for x, w in (([], [[1]]), ([[1, 2]], [[1]]), ([[np.nan]], [[1]])):
        with pytest.raises(ValueError):
            projection_bf16_fp32(x, w, core)
    with pytest.raises(ValueError):
        error_statistics(np.ones((1, 1)), np.ones((1, 2)))
