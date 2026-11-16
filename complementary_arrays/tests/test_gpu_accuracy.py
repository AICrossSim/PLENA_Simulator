"""Answer parsing, paired differences and KV-cache format names of the accuracy summaries."""

import pytest

from complementary_arrays.gpu.accuracy.kv_cache import parse
from complementary_arrays.gpu.accuracy.summarize import fmt_delta, gpqa_answer, paired


def test_gpqa_answer():
    assert gpqa_answer("reasoning\nAnswer: c") == "C"
    assert gpqa_answer("so the answer is (B), not A)") == "B"  # last pattern position wins
    assert gpqa_answer("**D)** is right") == "D"
    assert gpqa_answer("no letter here") is None


def test_paired():
    d, excludes_zero = paired([1, 1, 1, 0], [0, 0, 0, 0])
    assert d == pytest.approx(75.0) and excludes_zero
    assert fmt_delta((0.0, False)) == "0.0" and fmt_delta((-5.2, True)) == "-5.2*"


def test_kv_cache_parts():
    assert parse("mxint4") == dict(k="mxint4", v="mxint4", c="mxint4", pe="mxint4")
    assert parse("mxint4_pe") == dict(k="bf16", v="bf16", c="bf16", pe="mxint4")
    with pytest.raises(ValueError):
        parse("mxint3")
