from __future__ import annotations

import functools
import os
from pathlib import Path

from .matrix_lcompute_campaign import (
    EVIDENCE_LEVELS,
    KIMI_PACKET,
    NEMOTRON_PACKET,
    MatrixHardwarePoint,
    MatrixVariant,
    StateMode,
    _dse_packet,
    apply_real_fixed_map_selection,
    attach_real_service_evidence,
    build_ordinary_no_regression_evidence,
    build_prefill_handoff_timeline_delta,
    build_physical_evidence,
    measure_packet,
    load_compiler_evidence,
    recurrent_core_metrics,
    run_ablation,
)
from .nemotron3_workload import InferencePhase


COMPILER_ROOT = Path(
    os.environ.get(
        "PLENA_COMPILER_ROOT",
        Path(__file__).resolve().parents[2] / "PLENA_Compiler",
    )
).resolve()


@functools.lru_cache(maxsize=1)
def _real_physical_evidence() -> dict:
    hardware = MatrixHardwarePoint()
    compiler = load_compiler_evidence(str(COMPILER_ROOT))
    physical = build_physical_evidence(hardware)
    apply_real_fixed_map_selection(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
    attach_real_service_evidence(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
    return physical


def _compiler_fixture() -> dict:
    return {
        "issue": {
            "assembly": {
                NEMOTRON_PACKET.compiler_key: {
                    "baseline": {"dynamic_issued_instructions": 92_399},
                    "postincrement_only": {"dynamic_issued_instructions": 51_311},
                },
                KIMI_PACKET.compiler_key: {
                    "baseline": {"dynamic_issued_instructions": 215_387},
                    "postincrement_only": {"dynamic_issued_instructions": 116_219},
                },
            }
        },
        "matrix_recurrence": {
            "models": {
                "nemotron3_mamba2": {
                    "metrics": {
                        "dynamic_issued_instructions": 2_321,
                        "packet_reads": 1_024,
                        "packet_writes": 256,
                    }
                },
                "kimi_k3_kda": {
                    "metrics": {
                        "dynamic_issued_instructions": 10_779,
                        "packet_reads": 4_608,
                        "packet_writes": 1_536,
                    }
                },
            }
        },
    }


def test_real_packets_move_values_and_reach_the_expected_bank_floor() -> None:
    hardware = MatrixHardwarePoint()
    cases = (
        (NEMOTRON_PACKET, 2, 2),
        (KIMI_PACKET, 4, 4),
    )
    for spec, fixed_cycles, affine_alpha in cases:
        fixed = measure_packet(spec, hardware=hardware, alpha=1, gamma=0)
        affine = measure_packet(
            spec,
            hardware=hardware,
            alpha=affine_alpha,
            gamma=1,
        )
        assert fixed["roundtrip_values_checked"] == 2048
        assert affine["roundtrip_values_checked"] == 2048
        assert fixed["service_cycles"] == fixed_cycles
        assert affine["service_cycles"] == affine["ideal_cycles"] == 1
        assert affine["wrong_alpha_changes_data"] is True


def test_fair_d_prime_pitch_matches_per_view_alpha_at_both_packet_floors() -> None:
    evidence = build_physical_evidence()
    assert evidence["global_fixed_map"]["alpha"] == 1
    assert evidence["global_fixed_map"]["gamma"] == 0
    assert evidence["global_fixed_map"]["pitch_by_model"] == {
        "nemotron3": 2,
        "kimi_k3": 4,
    }
    assert evidence["nemotron3"]["fixed_alpha_gamma_search_points"] == 4096
    assert evidence["kimi_k3"]["fixed_alpha_gamma_search_points"] == 4096
    assert evidence["nemotron3"]["implemented_colayout_speedup_over_pitch1"] == 2
    assert evidence["kimi_k3"]["implemented_colayout_speedup_over_pitch1"] == 4
    assert evidence["nemotron3"]["alpha_upper_bound_speedup_over_implemented"] == 1
    assert evidence["kimi_k3"]["alpha_upper_bound_speedup_over_implemented"] == 1
    for model in ("nemotron3", "kimi_k3"):
        capacity = evidence[model]["implemented_colayout_capacity"]
        assert capacity["capacity_overhead_rows"] == 0
        assert capacity["aliases"] == 0
        assert capacity["values_roundtrip_checked"] == 262_144


def test_real_lowering_service_replays_interleaved_pitch_and_matches_d() -> None:
    physical = _real_physical_evidence()
    nemotron = physical["nemotron3"]["real_lowering_service"][
        StateMode.OFFICIAL_FP32
    ]
    kimi = physical["kimi_k3"]["real_lowering_service"][StateMode.OFFICIAL_FP32]

    assert nemotron[MatrixVariant.C_MULTIROW_ORIGINAL]["bank_stall_cycles"] == 768
    assert nemotron[MatrixVariant.D_PRIME_BEST_FIXED]["bank_stall_cycles"] == 0
    assert nemotron[MatrixVariant.D_AFFINE]["bank_stall_cycles"] == 0
    assert kimi[MatrixVariant.C_MULTIROW_ORIGINAL]["bank_stall_cycles"] == 9_216
    assert kimi[MatrixVariant.D_PRIME_BEST_FIXED]["bank_stall_cycles"] == 0
    assert kimi[MatrixVariant.D_AFFINE]["bank_stall_cycles"] == 0
    assert kimi[MatrixVariant.D_AFFINE]["values_roundtrip_checked"] == 6_291_456


def test_c_d_prime_and_d_keep_the_same_issue_stream() -> None:
    physical = build_physical_evidence()
    compiler = _compiler_fixture()
    for spec in (NEMOTRON_PACKET, KIMI_PACKET):
        metrics = {
            variant: recurrent_core_metrics(
                spec=spec,
                variant=variant,
                state_mode=StateMode.OFFICIAL_FP32,
                compiler=compiler,
                physical=physical,
                batch_size=1,
            )
            for variant in (
                MatrixVariant.C_MULTIROW_ORIGINAL,
                MatrixVariant.D_PRIME_BEST_FIXED,
                MatrixVariant.D_AFFINE,
            )
        }
        assert len({record["issued"] for record in metrics.values()}) == 1
        assert metrics[MatrixVariant.C_MULTIROW_ORIGINAL]["issued"] == metrics[
            MatrixVariant.D_PRIME_BEST_FIXED
        ]["issued"]
        assert metrics[MatrixVariant.C_MULTIROW_ORIGINAL]["cycles"] >= metrics[
            MatrixVariant.D_PRIME_BEST_FIXED
        ]["cycles"]
        assert metrics[MatrixVariant.D_AFFINE]["stall"] == 0
        assert metrics[MatrixVariant.D_PRIME_BEST_FIXED]["stall"] == 0


def test_complete_nemotron_and_kimi_decode_timelines_keep_ordinary_layers_identical() -> None:
    hardware = MatrixHardwarePoint()
    physical = build_physical_evidence(hardware)
    compiler = _compiler_fixture()
    for model, expected in (
        ("nemotron3", {"mamba": 23, "moe": 23, "attention": 6}),
        ("kimi_k3", {"kda": 69, "mla": 24, "latent_moe": 92, "dense": 1}),
    ):
        result = run_ablation(
            model=model,
            phase=InferencePhase.DECODE,
            batch_size=1,
            tokens=1,
            context_length=2048,
            state_mode=StateMode.OFFICIAL_FP32,
            hardware=hardware,
            compiler_root=COMPILER_ROOT,
            compiler=compiler,
            physical=physical,
        )
        assert result["ordinary_attention_moe_cycles_identical"] is True
        for layer_type, count in expected.items():
            assert result["layer_counts"][layer_type] == count
        by_variant = {record["variant"]: record for record in result["records"]}
        assert by_variant[MatrixVariant.D_AFFINE]["bank_stall_cycles"] == 0
        assert by_variant[MatrixVariant.D_PRIME_BEST_FIXED]["bank_stall_cycles"] == 0
        assert by_variant[MatrixVariant.D_AFFINE][
            "speedup_vs_implemented_colayout"
        ] == 1
        assert by_variant[MatrixVariant.E_AFFINE_OVERLAP][
            "speedup_vs_implemented_colayout"
        ] >= 1


def test_prefill_is_supported_but_decode_only_packet_optimisation_is_a_noop() -> None:
    hardware = MatrixHardwarePoint()
    physical = build_physical_evidence(hardware)
    for model in ("nemotron3", "kimi_k3"):
        result = run_ablation(
            model=model,
            phase=InferencePhase.PREFILL,
            batch_size=1,
            tokens=16,
            context_length=16,
            state_mode=StateMode.OFFICIAL_FP32,
            hardware=hardware,
            compiler_root=COMPILER_ROOT,
            compiler=_compiler_fixture(),
            physical=physical,
        )
        assert len({record["cycles"] for record in result["records"]}) == 1


def test_prefill_handoff_delta_is_separate_and_never_credits_fp32() -> None:
    handoff = {
        "shape": {"kda_layers": 69},
        "legacy_identity_gemm": {
            "logical_macs_all_kda_layers": 13_891_534_848,
            "emitted_padded_macs_all_kda_layers": 56_899_726_737_408,
            "emitted_matrix_cycles_all_kda_layers": 868_220_928,
        },
        "matrix_view_handoff": {
            "configuration_dynamic_instructions": 5,
            "handoff_macs": 0,
            "value_evidence": {"values_checked": 16_384},
        },
    }
    experiments = {
        StateMode.BF16_CANDIDATE: {
            "kimi_k3": {
                f"prefill_b1_s{tokens}": {
                    "records": [
                        {
                            "variant": MatrixVariant.D_AFFINE,
                            "cycles": useful,
                        }
                    ]
                }
                for tokens, useful in ((16, 1000), (128, 2000))
            }
        }
    }
    report = build_prefill_handoff_timeline_delta(
        experiments=experiments,
        handoff=handoff,
    )
    assert report["official_fp32_speedup_claimed"] is False
    assert report["performance_claim_withdrawn"] is True
    assert report["legacy_matrix_cycles_formula_not_used_for_speedup"] == 868_220_928
    assert report["view_handoff_macs"] == 0
    assert report["view_configuration_instructions_if_repeated_per_layer"] == 345
    assert report["values_moved_and_compared"] == 16_384
    assert report["cases"] == {}


def test_resource_contract_has_no_cache_private_sram_or_new_macs() -> None:
    resources = MatrixHardwarePoint().resource_proxies()
    assert resources["additional_sram_payload_bytes"] == 0
    assert resources["additional_cache_tags_or_replacement_bits"] == 0
    assert resources["additional_mac_lanes"] == 0
    assert resources["configuration_register_bits"] == 256
    assert resources["additional_operand_staging_bytes"] == 0
    assert resources["existing_vector_operand_buffer_reused"] is True

    by_variant = MatrixHardwarePoint().resource_proxies_by_variant()
    assert all(
        resources["fixed_diagonal_address_adders_existing"] == 64
        for resources in by_variant.values()
    )
    assert all(
        resources["incremental_fixed_diagonal_address_adders"] == 0
        for resources in by_variant.values()
    )
    assert by_variant[MatrixVariant.D_PRIME_BEST_FIXED][
        "configuration_register_bits"
    ] == 256
    assert by_variant[MatrixVariant.D_PRIME_BEST_FIXED][
        "compiler_programmable_tile_pitch"
    ] is True
    assert by_variant[MatrixVariant.D_PRIME_BEST_FIXED][
        "additional_programmable_skew_address_adders"
    ] == 0
    assert by_variant[MatrixVariant.D_AFFINE]["configuration_register_bits"] == 256
    assert by_variant[MatrixVariant.D_AFFINE]["architectural_variant"] is False
    assert by_variant[MatrixVariant.D_AFFINE][
        "counterfactual_programmable_alpha"
    ] is True
    assert by_variant[MatrixVariant.D_AFFINE]["layout_added_sram_payload_bytes"] == 0
    assert by_variant[MatrixVariant.D_AFFINE]["additional_operand_staging_bytes"] == 0
    assert by_variant[MatrixVariant.D_AFFINE][
        "existing_vector_operand_buffer_reused"
    ] is True
    assert by_variant[MatrixVariant.E_AFFINE_OVERLAP][
        "overlap_requires_runtime_scheduler"
    ] is False
    assert by_variant[MatrixVariant.E_AFFINE_OVERLAP]["architectural_variant"] is True


def test_dse_packet_sweeps_move_values_instead_of_only_applying_a_formula() -> None:
    hardware = MatrixHardwarePoint()
    mamba = _dse_packet(
        name="mamba_packet_512",
        row_width=64,
        packet_width=512,
        hardware=hardware,
    )
    kda = _dse_packet(
        name="kda_packet_2048",
        row_width=128,
        packet_width=2048,
        hardware=hardware,
    )
    assert mamba["values_checked_per_variant"] == 512
    assert kda["values_checked_per_variant"] == 2048
    assert mamba["D_implemented_colayout"]["bank_stall_cycles"] == 0
    assert kda["D_implemented_colayout"]["bank_stall_cycles"] == 0
    assert mamba["implemented_colayout_speedup_over_pitch1"] == 2
    assert kda["implemented_colayout_speedup_over_pitch1"] == 4


def test_ordinary_attention_and_moe_matrix_lines_do_not_regress() -> None:
    hardware = MatrixHardwarePoint()
    compiler = load_compiler_evidence(str(COMPILER_ROOT))
    physical = build_physical_evidence(hardware)
    apply_real_fixed_map_selection(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
    evidence = build_ordinary_no_regression_evidence(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
    assert evidence["all_service_cycles_identical"] is True
    assert evidence["values_checked"] == {
        "per_row": 2048 * hardware.banks,
        "per_column": 128 * hardware.banks,
    }
    assert evidence["allocation_base_phases_checked"] == hardware.banks
    assert set(evidence["source_stages"]) == {
        "gqa_attention_qkt",
        "mla_attention_qkt",
        "moe_gate_projection",
        "latent_moe_gate_projection",
    }


def test_evidence_levels_do_not_overclaim_full_real_weight_execution() -> None:
    assert "Four-token reduced-shape" in EVIDENCE_LEVELS["multi_token_recurrence"]
    assert "symbolic PLENA weights" in EVIDENCE_LEVELS["full_model_timeline"]
    assert "No real-weight first-to-last-layer Rust execution" in EVIDENCE_LEVELS[
        "not_demonstrated"
    ]
