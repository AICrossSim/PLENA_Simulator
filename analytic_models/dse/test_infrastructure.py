from __future__ import annotations

import json
import math
import sqlite3
from argparse import Namespace

import pytest
import optuna

from analytic_models.dse.artifacts import (
    DSECacheDirectories,
    GLOBAL_DSE_CACHE_SCHEMA,
    build_physical_candidate_bank,
    build_json_cache_metadata,
    cache_entry_path,
    canonical_json_sha256,
    compact_trial_record,
    finalize_compact_artifacts,
    json_cache_metadata_path,
    load_or_create_json_cache_metadata,
    load_cached_json,
    load_json,
    materialize_sqlite_database,
    persist_trial_record,
    write_json,
)
from analytic_models.dse.cli import model_profile_consistency
from analytic_models.dse.calibrations import (
    RTL_V6_AREA_SCHEMA,
    RTL_V6_AREA_STATUS,
    RTL_V6_POWER_SCHEMA,
    RTL_V6_POWER_STATUS,
    load_dse_calibration_manifest,
)
from analytic_models.dse.domain import (
    DEFAULT_MLEN_VALUES,
    LEGAL_BLENS_BY_MLEN,
    SHAPE_DOMAIN_POLICY,
    canonical_sram_choices,
    scale_chip_counts_for_reference,
    valid_blen_values,
    valid_mlen_values,
)
from analytic_models.dse.objective import (
    OBJECTIVE_DIRECTIONS,
    OBJECTIVE_FIELDS,
    OBJECTIVE_NORMALIZATION,
    ObjectiveValues,
    area_budget_constraints,
)
from analytic_models.dse.profiles import CURRENT_DSE_PROFILE
from analytic_models.dse.precision_search import (
    build_matrix_datapath_signatures,
    conditional_precision_variant_param_name,
    matrix_datapath_signature_distance,
)
from analytic_models.dse.results import pareto_front_records
from analytic_models.dse.softmax_resource_analysis import _matched_r_rows


def test_current_profile_matches_formal_latency_energy_stack() -> None:
    assert OBJECTIVE_DIRECTIONS == ("minimize", "minimize")
    assert OBJECTIVE_NORMALIZATION == "identity"
    assert OBJECTIVE_FIELDS == (
        "prefill_latency_ms",
        "prefill_system_energy_mj_ideal",
    )
    assert CURRENT_DSE_PROFILE.compute_timing == "ideal-ii1"
    assert CURRENT_DSE_PROFILE.hbm_model == "hbm-dma-v4"
    assert CURRENT_DSE_PROFILE.multi_chip_model == "tile-aware-dp-tp-ep-v4"
    assert CURRENT_DSE_PROFILE.sram_port_model == "ideal-dual-port"
    assert CURRENT_DSE_PROFILE.vector_scalar_schedule == "rtl-v6"
    assert CURRENT_DSE_PROFILE.softmax_vector_schedule == "multi-row-v1"
    assert CURRENT_DSE_PROFILE.softmax_state_schedule == "row-bank-simd-v3"
    assert CURRENT_DSE_PROFILE.pv_accumulation_schedule == "direct-packed-rmw-v1"
    assert CURRENT_DSE_PROFILE.softmax_row_lanes == (1, 2, 4, 8, 16)
    assert CURRENT_DSE_PROFILE.softmax_row_isa_tiers == (1, 2, 4, 8)
    assert CURRENT_DSE_PROFILE.softmax_row_model_tiers == (1, 2, 4, 8, 16)
    assert CURRENT_DSE_PROFILE.allowed_weight_element_bits == (4,)
    assert CURRENT_DSE_PROFILE.softmax_row_issue_schedule == "wavefront-v1"
    assert CURRENT_DSE_PROFILE.fidelity == (
        "vector_machine_integrated_area_power_calibrated_"
        "full_core_top_level_not_run_r16_structural_model_tier"
    )


def test_current_profile_uses_promoted_rtl_v6_calibrations() -> None:
    manifest = load_dse_calibration_manifest()
    assert manifest.area_schema == RTL_V6_AREA_SCHEMA
    assert manifest.area_status == RTL_V6_AREA_STATUS
    assert manifest.power_schema == RTL_V6_POWER_SCHEMA
    assert manifest.power_status == RTL_V6_POWER_STATUS
    assert CURRENT_DSE_PROFILE.area_calibration_schema == manifest.area_schema
    assert CURRENT_DSE_PROFILE.power_calibration_schema == manifest.power_schema
    assert len(manifest.area_sha256) == 64
    assert len(manifest.power_sha256) == 64
    assert len(manifest.fingerprint) == 64


def test_rtl_v6_calibration_override_is_validated_and_fingerprinted(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = load_dse_calibration_manifest()
    payload = json.loads(installed.power_path.read_text())
    payload["schema_version"] = "unpromoted-test-schema"
    override = tmp_path / "power.json"
    override.write_text(json.dumps(payload))
    monkeypatch.setenv("PLENA_POWER_VECTOR_RTL_V6_DELTA", str(override))

    with pytest.raises(ValueError, match="power schema"):
        load_dse_calibration_manifest()


def test_softmax_matched_r_analysis_rejects_confounded_trials() -> None:
    base = {
        "precision_profile": "w_mxint4__act_mxint4__kv_mxint4__fp_e6m5",
        "MLEN": 2048,
        "VLEN": 2048,
        "BLEN": 128,
        "INT_DATA_WIDTH": 32,
        "chip_count": 8,
        "tp_degree": 4,
        "dp_degree": 2,
        "ep_degree": 1,
        "nvlink_port_count": 1,
        "matrix_sram_config_id": "kv25",
        "MATRIX_SRAM_TILES": 46,
        "matrix_sram_policy": "kv-25",
        "COMPACT_STATS_LANES": 16,
        "new_total_silicon_area_mm2": 800.0,
        "system_energy_nominal_mj": 100.0,
        "audit_fidelity": "recorded-row-lanes-recomputed-current-area-model",
    }
    rows = [
        {
            **base,
            "softmax_row_lanes": 1,
            "softmax_elements_per_cycle": 2048,
            "latency_ms": 10.0,
        },
        {
            **base,
            "softmax_row_lanes": 4,
            "softmax_elements_per_cycle": 8192,
            "latency_ms": 7.0,
            "system_energy_nominal_mj": 96.0,
            "new_total_silicon_area_mm2": 810.0,
        },
        {
            **base,
            "precision_profile": "different",
            "softmax_row_lanes": 8,
            "softmax_elements_per_cycle": 16384,
            "latency_ms": 6.0,
        },
    ]
    matched = _matched_r_rows(rows)
    assert [row["softmax_row_lanes"] for row in matched] == [1, 4]
    assert matched[1]["latency_delta_pct_vs_r1"] == pytest.approx(-30.0)
    assert matched[1]["energy_delta_pct_vs_r1"] == pytest.approx(-4.0)
    assert matched[1]["area_delta_pct_vs_r1"] == pytest.approx(1.25)


def test_objective_values_follow_optuna_order() -> None:
    values = ObjectiveValues.from_trial_record(
        {
            "latency_ms": 1,
            "area_mm2": 2,
            "system_energy_nominal_mj": 3,
            "accuracy_score": 0.9,
        }
    )
    assert values.as_optuna_values() == (1.0, 3.0)


def test_explicit_identity_normalized_values_take_precedence() -> None:
    values = ObjectiveValues.from_trial_record(
        {
            "latency_ms": 10,
            "system_energy_nominal_mj": 20,
            "normalized_latency": 1.25,
            "normalized_energy": 0.75,
        }
    )
    assert values.as_optuna_values() == (1.25, 0.75)


def test_canonical_prefill_objectives_take_precedence_over_aliases() -> None:
    values = ObjectiveValues.from_trial_record(
        {
            "latency_ms": 10,
            "system_energy_nominal_mj": 20,
            "normalized_latency": 9,
            "normalized_energy": 19,
            "prefill_latency_ms": 1.5,
            "prefill_system_energy_mj_ideal": 2.5,
        }
    )
    assert values.as_optuna_values() == (1.5, 2.5)


def test_precision_search_groups_kv_variants_under_one_pe_signature() -> None:
    base = {
        "name": "kv4",
        "accuracy_score": 0.98,
        "WEIGHT_WIDTH": {"kind": "MXINT", "width": 4, "scale_width": 8},
        "ACT_WIDTH": {"kind": "MXINT", "width": 4, "scale_width": 8},
        "KV_WIDTH": {"kind": "MXINT", "width": 4, "scale_width": 8},
        "FP_SETTING": {"exp": 5, "mant": 6},
    }
    kv8 = {
        **base,
        "name": "kv8",
        "KV_WIDTH": {"kind": "MXINT", "width": 8, "scale_width": 8},
    }
    act8 = {
        **base,
        "name": "act8",
        "ACT_WIDTH": {"kind": "MXINT", "width": 8, "scale_width": 8},
    }
    signatures, mapping = build_matrix_datapath_signatures(
        [base, kv8, act8]
    )
    by_id = {signature.signature_id: signature for signature in signatures}

    assert len(signatures) == 2
    assert mapping["kv4"] == mapping["kv8"]
    assert mapping["kv4"] != mapping["act8"]
    signature = by_id[mapping["kv4"]]
    assert signature.pe_bit_product == 16
    assert signature.profile_names == ("kv4", "kv8")
    assert (
        matrix_datapath_signature_distance(
            signature.signature_id,
            signature.signature_id,
            by_id,
        )
        == 0.0
    )
    assert conditional_precision_variant_param_name(
        signature.signature_id
    ).startswith("PRECISION_VARIANT_")


def test_canonical_domain_contains_only_legal_topologies() -> None:
    assert SHAPE_DOMAIN_POLICY == (
        "rtl_legal_full_shape_area_capacity_at_evaluation_v1"
    )
    assert tuple(LEGAL_BLENS_BY_MLEN) == (256, 512, 1024, 2048, 4096, 8192)
    for mlen, blens in LEGAL_BLENS_BY_MLEN.items():
        assert valid_blen_values(mlen) == blens
        assert all(blen <= mlen and mlen % blen == 0 for blen in blens)
    assert valid_blen_values(4096)[-1] == 1024
    assert valid_blen_values(8192)[-1] == 1024
    for chips in (1, 2, 4, 8, 16, 32, 64):
        assert valid_mlen_values(chips) == DEFAULT_MLEN_VALUES


def test_per_reference_scaling_does_not_change_local_shape_domain() -> None:
    normalized = (1, 2, 4, 8, 16)
    physical_r1 = scale_chip_counts_for_reference(
        normalized,
        reference_a100_count=1,
        mode="per-a100-reference",
    )
    physical_r4 = scale_chip_counts_for_reference(
        normalized,
        reference_a100_count=4,
        mode="per-a100-reference",
    )

    for chips_r1, chips_r4 in zip(physical_r1, physical_r4, strict=True):
        assert valid_mlen_values(chips_r1) == valid_mlen_values(chips_r4)


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
    assert load_json(tmp_path / "trial_record.json.gz") == compact
    assert not (tmp_path / "trial_record.json").exists()
    assert not (tmp_path / "trial_detail.json.gz").exists()
    assert canonical_json_sha256(record) == canonical_json_sha256(
        json.loads(json.dumps(record))
    )


def test_compact_finalization_compresses_resume_artifacts(tmp_path) -> None:
    for trial in (0, 1):
        trial_dir = tmp_path / f"trial_{trial:04d}"
        record = {
            "trial": trial,
            "state": "complete",
            "latency_ms": 4.0 + trial,
            "area_mm2": 5.0,
            "system_energy_nominal_mj": 6.0,
            "accuracy_score": 0.95,
            "power_shadow": {"large": list(range(100))},
        }
        persist_trial_record(trial_dir, record, artifact_retention="compact")

    database = tmp_path / "study.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE evidence(value INTEGER)")
    connection.execute("INSERT INTO evidence VALUES (7)")
    connection.commit()
    connection.close()
    (tmp_path / "all_trials.csv").write_text("trial,state\n0,complete\n")
    (tmp_path / "trials.jsonl").write_text('{"trial": 0}\n')
    (tmp_path / "worker_resources.jsonl").write_text('{"rss": 1}\n')

    manifest = finalize_compact_artifacts(
        tmp_path,
        retained_trial_ids={1},
    )

    assert not (tmp_path / "trial_0000" / "trial_record.json").exists()
    compressed_record = tmp_path / "trial_0000" / "trial_record.json.gz"
    assert compressed_record.exists()
    assert load_json(compressed_record)["trial"] == 0
    assert not (tmp_path / "trial_0001" / "trial_record.json").exists()
    assert (tmp_path / "trial_0001" / "trial_record.json.gz").exists()
    assert not (tmp_path / "trial_0001" / "trial_detail.json.gz").exists()
    assert (tmp_path / "all_trials.csv.gz").exists()
    assert (tmp_path / "trials.jsonl.gz").exists()
    assert not (tmp_path / "worker_resources.jsonl").exists()
    assert not database.exists()
    assert (tmp_path / "study.sqlite3.gz").exists()
    assert manifest["bytes_after_cleanup"]["total"] > 0

    restored = materialize_sqlite_database(tmp_path)
    connection = sqlite3.connect(restored)
    assert connection.execute("SELECT value FROM evidence").fetchone() == (7,)
    connection.close()


def test_shared_json_cache_metadata_reuses_immutable_hashes(tmp_path) -> None:
    payload = {"large": list(range(1000)), "nested": {"value": 7}}
    report_path = tmp_path / "report.json.gz"
    write_json(report_path, payload)

    expected = build_json_cache_metadata(report_path, payload)
    first = load_or_create_json_cache_metadata(report_path, payload)
    assert first == expected
    assert json_cache_metadata_path(report_path).exists()

    # A valid sidecar is authoritative for an immutable shared cache entry;
    # the caller does not need to serialize the full payload on every hit.
    second = load_or_create_json_cache_metadata(
        report_path,
        {"not": "the already-loaded report"},
    )
    assert second == first


def test_versioned_cross_study_cache_directories(tmp_path) -> None:
    directories = DSECacheDirectories.create(tmp_path / "cache")
    assert directories.root.name == GLOBAL_DSE_CACHE_SCHEMA
    assert directories.compiler_reports.is_dir()
    assert directories.compiler_traces.is_dir()
    assert directories.compiler_v4_work.is_dir()
    assert directories.area_reports.is_dir()

    path = cache_entry_path(directories.compiler_reports, "abc")
    write_json(path, {"result": 7})
    assert load_cached_json(path) == {"result": 7}
    path.write_bytes(b"not a gzip member")
    assert load_cached_json(path) is None


def test_physical_candidate_bank_excludes_runtime_topology() -> None:
    common = {
        "state": "complete",
        "model_config": "qwen3.json",
        "precision_profile": "w4a4",
        "MLEN": 2048,
        "VLEN": 2048,
        "BLEN": 128,
        "INT_DATA_WIDTH": 32,
        "matrix_sram_tiles": 8,
        "softmax_row_lanes": 8,
        "chip_count": 16,
        "nvlink_port_count": 2,
        "accuracy_score": 0.97,
    }
    records = [
        {
            **common,
            "trial": 1,
            "dp_degree": 2,
            "tp_degree": 8,
            "ep_degree": 1,
            "latency_ms": 10.0,
            "system_energy_nominal_mj": 20.0,
        },
        {
            **common,
            "trial": 2,
            "dp_degree": 4,
            "tp_degree": 4,
            "ep_degree": 1,
            "latency_ms": 8.0,
            "system_energy_nominal_mj": 22.0,
        },
    ]
    bank = build_physical_candidate_bank(records)
    assert len(bank) == 1
    assert bank[0]["source_trials"] == [1, 2]
    assert len(bank[0]["source_runtime_topologies"]) == 2
    assert bank[0]["best_latency_ms"] == 8.0
    assert bank[0]["best_system_energy_nominal_mj"] == 20.0
    assert "dp_degree" not in bank[0]["physical_design"]


def test_pareto_front_records_respects_area_and_two_objectives() -> None:
    def record(trial: int, latency: float, energy: float, area: float = -1.0):
        return {
            "trial": trial,
            "state": "complete",
            "latency_ms": latency,
            "system_energy_nominal_mj": energy,
            "area_budget_constraint_mm2": area,
        }

    records = [
        record(0, 1.0, 4.0),
        record(1, 2.0, 3.0),
        record(5, 2.0, 3.0),
        record(2, 3.0, 5.0),
        record(3, 4.0, 2.0),
        record(4, 0.5, 1.0, area=0.1),
    ]

    assert [item["trial"] for item in pareto_front_records(records)] == [0, 1, 3]


def test_area_constraint_callback_uses_durable_user_attribute() -> None:
    pruned = optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED)
    assert math.isinf(area_budget_constraints(pruned)[0])

    sampler = optuna.samplers.TPESampler(
        constraints_func=area_budget_constraints,
        n_startup_trials=1,
        seed=7,
    )
    study = optuna.create_study(
        directions=("minimize", "minimize"),
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        trial.set_user_attr("area_budget_constraint_mm2", -2.0)
        return (1.0, 2.0)

    study.optimize(objective, n_trials=1)

    assert study.trials[0].system_attrs["constraints"] == (-2.0,)
    assert [trial.number for trial in study.best_trials] == [0]


@pytest.mark.parametrize("name", ("report.json", "report.json.gz"))
def test_json_artifacts_are_atomically_replaced(tmp_path, name: str) -> None:
    path = tmp_path / name
    original = {"complete": True, "values": list(range(32))}
    write_json(path, original)

    with pytest.raises(TypeError):
        write_json(path, {"not_json_serializable": {1, 2, 3}})

    assert load_json(path) == original
    assert not tuple(tmp_path.glob(f".{name}.tmp.*"))


def test_compact_failed_trial_preserves_traceback() -> None:
    record = {
        "trial": 17,
        "state": "failed",
        "reason": "ValueError: diagnostic",
        "traceback": "Traceback (most recent call last):\nValueError: diagnostic\n",
        "large_detail": list(range(4096)),
    }

    compact = compact_trial_record(record)

    assert compact["traceback"] == record["traceback"]
    assert "large_detail" not in compact


def test_named_profile_rejects_mislabeled_override() -> None:
    args = Namespace(
        model_profile="current-dse-v1",
        compiler_compute_timing="rtl-v1",
        compiler_trace_granularity="affine-block-summary-v1",
        multi_chip_model="tile-aware-tp-cp-ep-v3",
        clock_gating_mode="ideal-hierarchical",
        vector_scalar_schedule="rtl-v6",
        softmax_vector_schedule="multi-row-v1",
        softmax_state_schedule="row-bank-simd-v3",
        pv_accumulation_schedule="direct-packed-rmw-v1",
        softmax_row_issue_schedule="wavefront-v1",
    )
    consistent, mismatches = model_profile_consistency(args)
    assert not consistent
    assert any("compiler_compute_timing" in mismatch for mismatch in mismatches)


def test_unknown_mlen_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside the canonical DSE domain"):
        valid_blen_values(123)
