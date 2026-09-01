from __future__ import annotations

import argparse

import pytest

from .hybrid_connected_evidence import (
    _batch_cases,
    _fixed_cases,
    _parse_batch_sizes,
    _sha256_json,
)


def test_fixed_cases_cover_matrix_mamba_and_kda() -> None:
    cases = _fixed_cases("python")

    assert [case.name for case in cases] == [
        "matrix_affine_writeback",
        "nemotron_mamba_s128_handoff",
        "kimi_kda_s128_handoff",
    ]
    assert all(case.require_l_cfg for case in cases)
    assert not cases[0].require_conflict_free_packet
    assert all(case.require_conflict_free_packet for case in cases[1:])


def test_batch_cases_cover_each_model_at_every_requested_batch() -> None:
    cases = _batch_cases("python", (1, 2, 4, 8, 16))

    assert len(cases) == 10
    assert cases[0].name == "mamba_private_state_b1"
    assert cases[1].name == "kda_private_state_b1"
    assert cases[-2].name == "mamba_private_state_b16"
    assert cases[-1].name == "kda_private_state_b16"
    assert all(not case.require_conflict_free_packet for case in cases)


def test_batch_size_parser_is_strictly_numeric() -> None:
    assert _parse_batch_sizes("1,2,4") == (1, 2, 4)
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated integers"):
        _parse_batch_sizes("1,two")


def test_canonical_hash_does_not_depend_on_key_order() -> None:
    assert _sha256_json({"a": 1, "b": 2}) == _sha256_json({"b": 2, "a": 1})
