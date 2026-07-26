from __future__ import annotations

import json
from argparse import Namespace

import pytest

from analytic_models.dse.artifacts import (
    canonical_json_sha256,
    compact_trial_record,
    load_json,
    persist_trial_record,
)
from analytic_models.dse.cli import model_profile_consistency
from analytic_models.dse.domain import (
    LEGAL_BLENS_BY_MLEN,
    canonical_sram_choices,
    valid_blen_values,
    valid_mlen_values,
)
from analytic_models.dse.objective import (
    OBJECTIVE_DIRECTIONS,
    ObjectiveValues,
)
from analytic_models.dse.profiles import CURRENT_DSE_PROFILE


def test_current_profile_matches_formal_four_objective_stack() -> None:
    assert OBJECTIVE_DIRECTIONS == (
        "minimize",
        "minimize",
        "minimize",
        "maximize",
    )
    assert CURRENT_DSE_PROFILE.compute_timing == "ideal-ii1"
    assert CURRENT_DSE_PROFILE.hbm_model == "hbm-dma-v4"
    assert CURRENT_DSE_PROFILE.multi_chip_model == "tile-aware-tp-cp-ep-v3"
    assert CURRENT_DSE_PROFILE.sram_port_model == "ideal-dual-port"


def test_objective_values_follow_optuna_order() -> None:
    values = ObjectiveValues.from_trial_record(
        {
            "latency_ms": 1,
            "area_mm2": 2,
            "system_energy_nominal_mj": 3,
            "accuracy_score": 0.9,
        }
    )
    assert values.as_optuna_values() == (1.0, 2.0, 3.0, 0.9)


def test_canonical_domain_contains_only_legal_topologies() -> None:
    assert tuple(LEGAL_BLENS_BY_MLEN) == (256, 512, 1024, 2048, 4096, 8192)
    for mlen, blens in LEGAL_BLENS_BY_MLEN.items():
        assert valid_blen_values(mlen) == blens
        assert all(blen <= mlen and mlen % blen == 0 for blen in blens)
    assert 8192 not in valid_mlen_values(8)
    assert 4096 not in valid_mlen_values(16)


def test_canonical_sram_choices_merge_aliases() -> None:
    class Plan:
        def __init__(self, tiles: int, resident: int):
            self.matrix_sram_tiles = tiles
            self.resident_prefix_blocks = resident

    def derive_policy(*, policy: str, **_: object) -> Plan:
        return Plan(2, 0) if policy in {"streaming", "alias"} else Plan(8, 3)

    choices = canonical_sram_choices(
        policies=("streaming", "alias", "kv-50"),
        k_blocks=4,
        mlen=512,
        projection_tiles=8,
        derive_policy=derive_policy,
    )
    assert len(choices) == 2
    assert choices[0]["policy_aliases"] == ("streaming", "alias")


def test_compact_artifact_is_hash_stable_and_resume_safe(tmp_path) -> None:
    record = {
        "trial": 3,
        "state": "complete",
        "latency_ms": 4.0,
        "area_mm2": 5.0,
        "system_energy_nominal_mj": 6.0,
        "accuracy_score": 0.95,
        "power_shadow": {"large": list(range(100))},
    }
    compact = compact_trial_record(record)
    assert "power_shadow" not in compact
    persist_trial_record(tmp_path, record, artifact_retention="compact")
    assert load_json(tmp_path / "trial_record.json") == compact
    assert load_json(tmp_path / "trial_detail.json.gz") == record
    assert canonical_json_sha256(record) == canonical_json_sha256(
        json.loads(json.dumps(record))
    )


def test_named_profile_rejects_mislabeled_override() -> None:
    args = Namespace(
        model_profile="current-dse-v1",
        compiler_compute_timing="rtl-v1",
        compiler_trace_granularity="affine-block-summary-v1",
        multi_chip_model="tile-aware-tp-cp-ep-v3",
        clock_gating_mode="ideal-hierarchical",
    )
    consistent, mismatches = model_profile_consistency(args)
    assert not consistent
    assert any("compiler_compute_timing" in mismatch for mismatch in mismatches)


def test_unknown_mlen_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside the canonical DSE domain"):
        valid_blen_values(123)
