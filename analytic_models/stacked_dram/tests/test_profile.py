"""JSON memory profiles."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from stacked_dram_fixtures import EXAMPLES

from analytic_models.stacked_dram import (
    FixedBandwidthMemory,
    StackedDramModel,
    load_memory_profile,
    memory_from_dict,
)

STACKED_EXAMPLE = EXAMPLES / "fictional_stacked_dram.json"
FIXED_EXAMPLE = EXAMPLES / "fictional_fixed_bandwidth.json"


def _stacked_profile() -> dict:
    return json.loads(STACKED_EXAMPLE.read_text())


def test_examples_load_and_record_their_provenance() -> None:
    stacked = load_memory_profile(STACKED_EXAMPLE)
    assert isinstance(stacked, StackedDramModel)
    assert stacked.config.bank_timing is not None
    assert stacked.buffering is not None and stacked.thermal is not None and stacked.energy is not None
    assert stacked.provenance["profile_sha256"] == hashlib.sha256(STACKED_EXAMPLE.read_bytes()).hexdigest()
    assert stacked.provenance["profile_path"] == str(STACKED_EXAMPLE)
    assert "fictional" in stacked.provenance["source"]

    fixed = load_memory_profile(FIXED_EXAMPLE)
    assert isinstance(fixed, FixedBandwidthMemory)
    assert fixed.capacity_bytes == 12_345_678_901


def test_profile_matches_the_python_description() -> None:
    data = _stacked_profile()
    memory = memory_from_dict(data)
    assert memory.config.total_layers == data["dram"]["total_layers"]
    assert memory.config.bank_timing.row_bytes == data["dram"]["bank_timing"]["row_bytes"]
    assert memory.thermal.dynamic_power_exponent == data["thermal"]["dynamic_power_exponent"]
    assert memory.energy.write_pj_per_bit == data["energy"]["write_pj_per_bit"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda profile: profile["dram"].update(totl_layers=4),
        lambda profile: profile["dram"]["bank_timing"].update(row_bytez=1),
        lambda profile: profile.update(extra=1),
        lambda profile: profile["thermal"].pop("static_power_w"),
        lambda profile: profile.pop("dram"),
    ],
)
def test_unknown_or_missing_keys_are_rejected(mutate) -> None:
    profile = copy.deepcopy(_stacked_profile())
    mutate(profile)
    with pytest.raises(ValueError):
        memory_from_dict(profile)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("schema_version", 2, ValueError),
        ("kind", "magic", ValueError),
        ("provenance", "free text", TypeError),
        ("dram", [], TypeError),
    ],
)
def test_header_fields_are_checked(key: str, value: object, error: type[Exception]) -> None:
    profile = _stacked_profile()
    profile[key] = value
    with pytest.raises(error):
        memory_from_dict(profile)


def test_json_booleans_are_not_numbers() -> None:
    profile = _stacked_profile()
    profile["dram"]["total_layers"] = True
    with pytest.raises(TypeError):
        memory_from_dict(profile)
