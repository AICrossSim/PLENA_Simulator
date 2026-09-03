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
    attach_real_service_evidence,
    build_ordinary_no_regression_evidence,
    build_prefill_handoff_timeline_delta,
    build_static_overlap_feasibility,
    build_physical_evidence,
    measure_packet,
    load_compiler_evidence,
    measure_fixed_phased_packet,
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
    attach_real_service_evidence(
        compiler=compiler,
        physical=physical,
        hardware=hardware,
    )
    return physical


@functools.lru_cache(maxsize=1)
def _real_compiler_evidence() -> dict:
    return load_compiler_evidence(str(COMPILER_ROOT))


def _compiler_fixture() -> dict:
    return {
        "hybrid_l_tile_schedule": {
            "variants": {
                "affine": {
                    "nemotron3": {
                        "recurrent_layer_count": 23,
                        "all_recurrent_layers_emit_l_tile": True,
                        "l_tile_exec_count": 23 * 8,
                        "assembly_sha256": "fixture-nemotron",
                        "architectural_boundary": {},
                    },
                    "kimi_k3": {
                        "recurrent_layer_count": 69,
                        "all_recurrent_layers_emit_l_tile": True,
                        "l_tile_exec_count": 69 * 30,
                        "assembly_sha256": "fixture-kimi",
                        "architectural_boundary": {},
                    },
                }
            }
        },
        "issue": {
            "assembly": {
                NEMOTRON_PACKET.compiler_key: {
                    "baseline": {"dynamic_issued_instructions": 92_399},
                    "postincrement_only": {"dynamic_issued_instructions": 51_311},
                },
                KIMI_PACKET.compiler_key: {
                    "baseline": {"dynamic_issued_instructions": 201_948},
                    "postincrement_only": {"dynamic_issued_instructions": 103_356},
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
                    },
                    "capacity_points": {
                        str(1024 * 1024): {
                            "affine": {"metrics": {"l_tile_exec_count": 8}}
                        }
                    },
                },
                "kimi_k3_kda": {
                    "metrics": {
                        "dynamic_issued_instructions": 10_779,
                        "packet_reads": 4_608,
                        "packet_writes": 1_536,
                    },
                    "capacity_points": {
                        str(1024 * 1024): {
                            "affine": {"metrics": {"l_tile_exec_count": 30}}
                        }
                    },
                },
            }
        },
    }


def test_active_lcompute_point_is_uniform_bf16() -> None:
    hardware = MatrixHardwarePoint()
    assert list(StateMode) == [StateMode.PLENA_BF16]
    assert hardware.matrix_element_bits == 16
    assert hardware.matrix_bank_word_bits == 512
    assert hardware.matrix_bank_word_beats == 1
    assert hardware.matrix_sram_bytes == 1024 * 1024


def test_real_packets_move_values_and_reach_the_expected_bank_floor() -> None:
    hardware = MatrixHardwarePoint()
    cases = (
        (NEMOTRON_PACKET, 2),
        (KIMI_PACKET, 4),
    )
    for spec, words_per_head in cases:
        fixed = measure_packet(
            spec,
            hardware=hardware,
            alpha=1,
            gamma=0,
            tile_pitch_rows=words_per_head,
        )
        affine = measure_packet(
            spec,
            hardware=hardware,
            alpha=words_per_head,
            gamma=1,
        )
        assert fixed["roundtrip_values_checked"] == 2048
        assert affine["roundtrip_values_checked"] == 2048
        assert fixed["service_cycles"] == fixed["ideal_cycles"] == 1
        assert affine["service_cycles"] == affine["ideal_cycles"] == 1
        assert affine["wrong_alpha_changes_data"] is True


def test_strongest_fixed_phased_control_matches_affine_bank_coordinates() -> None:
    """D' owns all ordinary base-placement freedom available to the compiler."""

    hardware = MatrixHardwarePoint()
    expected_values = {
        "nemotron3": 32 * 128 * 64,
        "kimi_k3": 16 * 128 * 128,
    }
    for spec in (NEMOTRON_PACKET, KIMI_PACKET):
        control = measure_fixed_phased_packet(spec, hardware=hardware)
        assert control["service_cycles"] == control["ideal_cycles"] == 1
        assert control["bank_stall_cycles"] == 0
        assert control["banks_touched"] == hardware.banks
        assert control["physical_rows"] == spec.recurrence_rows
        assert control["capacity_bytes"] == hardware.matrix_sram_bytes // 2
        assert control["roundtrip_values_checked"] == expected_values[spec.model]
        assert control["same_physical_coordinates_as_affine_tile_skew"] is True
        assert control["programmable_skew_bank_speedup"] == 1.0


def test_fixed_and_affine_packet_controls_have_explicit_freedoms() -> None:
    evidence = build_physical_evidence()
    assert evidence["global_fixed_map"]["alpha"] == 1
    assert evidence["global_fixed_map"]["gamma"] == 0
    assert evidence["degrees_of_freedom"]["C"]["compiler_controls"] == [
        "base_bank_phase",
        "tile_pitch_rows",
        "group_phase",
        "chunking",
    ]
    assert "row_skew" in evidence["degrees_of_freedom"]["D"][
        "compiler_controls"
    ]
    assert "tile_skew" in evidence["degrees_of_freedom"]["D"][
        "compiler_controls"
    ]
    assert evidence["degrees_of_freedom"]["D_prime"]["compact_single_descriptor"] is False
    for model in ("nemotron3", "kimi_k3"):
        control = evidence["fixed_phased_bank_control"][model]
        assert control["bank_stall_cycles"] == 0
        assert control["programmable_skew_bank_speedup"] == 1.0


def test_real_lowering_service_replays_fixed_and_affine_paths() -> None:
    physical = _real_physical_evidence()
    nemotron = physical["nemotron3"]["real_lowering_service"][
        StateMode.PLENA_BF16
    ]
    kimi = physical["kimi_k3"]["real_lowering_service"][StateMode.PLENA_BF16]

    # Fixed diagonal cannot separate the adjacent row/word phases in Mamba's
    # compact update/C DMA blocks. This is the only measured programmable-skew
    # bank benefit at the BF16 point.
    assert nemotron[MatrixVariant.C_FIXED]["bank_stall_cycles"] == 256
    assert nemotron[MatrixVariant.D_AFFINE]["bank_stall_cycles"] == 0
    assert kimi[MatrixVariant.C_FIXED]["bank_stall_cycles"] == 0
    assert kimi[MatrixVariant.D_AFFINE]["bank_stall_cycles"] == 0
    state_values = KIMI_PACKET.heads * KIMI_PACKET.recurrence_rows * KIMI_PACKET.elements_per_head
    assert kimi[MatrixVariant.D_AFFINE]["logical_values_replayed"] >= 2 * state_values


def test_c_and_d_keep_the_same_math_while_affine_removes_chunking_and_stalls() -> None:
    physical = _real_physical_evidence()
    compiler = _real_compiler_evidence()
    for spec in (NEMOTRON_PACKET, KIMI_PACKET):
        metrics = {
            variant: recurrent_core_metrics(
                spec=spec,
                variant=variant,
                state_mode=StateMode.PLENA_BF16,
                compiler=compiler,
                physical=physical,
                batch_size=1,
            )
            for variant in (MatrixVariant.C_FIXED, MatrixVariant.D_AFFINE)
        }
        assert metrics[MatrixVariant.C_FIXED]["logical_state_values"] == metrics[
            MatrixVariant.D_AFFINE
        ]["logical_state_values"]
        assert metrics[MatrixVariant.C_FIXED]["arithmetic_element_ops"] == metrics[
            MatrixVariant.D_AFFINE
        ]["arithmetic_element_ops"]
        assert metrics[MatrixVariant.C_FIXED]["issued"] > metrics[
            MatrixVariant.D_AFFINE
        ]["issued"]
        assert metrics[MatrixVariant.C_FIXED]["cycles"] > metrics[
            MatrixVariant.D_AFFINE
        ]["cycles"]
        assert metrics[MatrixVariant.C_FIXED]["stall"] >= 0
        assert metrics[MatrixVariant.D_AFFINE]["stall"] <= metrics[
            MatrixVariant.C_FIXED
        ]["stall"]
        assert metrics[MatrixVariant.D_AFFINE]["stall"] == 0


def test_complete_nemotron_and_kimi_decode_timelines_keep_ordinary_layers_identical() -> None:
    hardware = MatrixHardwarePoint()
    physical = _real_physical_evidence()
    compiler = _real_compiler_evidence()
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
            state_mode=StateMode.PLENA_BF16,
            hardware=hardware,
            compiler_root=COMPILER_ROOT,
            compiler=compiler,
            physical=physical,
        )
        assert result["ordinary_attention_moe_cycles_identical"] is True
        assert result["compiler_l_tile_schedule"][
            "all_recurrent_layers_emit_l_tile"
        ] is True
        assert result["compiler_l_tile_schedule"]["recurrent_layer_count"] == (
            23 if model == "nemotron3" else 69
        )
        for layer_type, count in expected.items():
            assert result["layer_counts"][layer_type] == count
        by_variant = {record["variant"]: record for record in result["records"]}
        coefficient_prep = {
            record["recurrence_coefficient_prep_cycles"]
            for record in result["records"]
        }
        assert coefficient_prep == ({0} if model == "nemotron3" else {4_485})
        assert {
            record["recurrence_coefficient_prep_elementwise_ops"]
            for record in result["records"]
        } == ({0} if model == "nemotron3" else {5_107_104})
        assert {
            record["recurrence_coefficient_prep_exp_ops"]
            for record in result["records"]
        } == ({0} if model == "nemotron3" else {1_702_368})
        ordinary_vector_cycles = {
            record["vector_cycles"]
            - (
                record["recurrence_cycles"]
                if record["variant"]
                in {MatrixVariant.A_ORIGINAL, MatrixVariant.B_ARLO}
                else 0
            )
            for record in result["records"]
        }
        assert len(ordinary_vector_cycles) == 1
        assert by_variant[MatrixVariant.D_AFFINE]["bank_stall_cycles"] == 0
        assert by_variant[MatrixVariant.D_AFFINE]["speedup_vs_C_fixed"] >= 1
        assert by_variant[MatrixVariant.E_AFFINE_OVERLAP]["cycles"] == by_variant[
            MatrixVariant.D_AFFINE
        ]["cycles"]
        assert by_variant[MatrixVariant.E_AFFINE_OVERLAP][
            "speedup_vs_D_affine"
        ] == 1


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
            state_mode=StateMode.PLENA_BF16,
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
        StateMode.PLENA_BF16: {
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
    assert resources["existing_matrix_to_vector_datapath_reused"] is True
    assert resources["additional_payload_datapath_bits"] == 0
    assert resources["sequencer_state_bits_upper_bound"] == 256
    assert resources["segment_scalar_broadcast_lanes"] == 32
    assert resources["segment_scalar_broadcast_mux_input_bits"] == 512

    by_variant = MatrixHardwarePoint().resource_proxies_by_variant()
    assert all(record["fixed_diagonal_address_adders_existing"] == 64 for record in by_variant.values())
    assert all(
        record["incremental_fixed_diagonal_address_adders"] == 0
        for record in by_variant.values()
    )
    assert by_variant[MatrixVariant.C_FIXED][
        "configuration_register_bits"
    ] == 256
    assert by_variant[MatrixVariant.A_ORIGINAL]["architectural_variant"] is False
    assert by_variant[MatrixVariant.B_ARLO]["architectural_variant"] is False
    assert by_variant[MatrixVariant.C_FIXED]["architectural_variant"] is True
    assert by_variant[MatrixVariant.C_FIXED][
        "compiler_programmable_tile_pitch"
    ] is True
    assert by_variant[MatrixVariant.D_AFFINE]["configuration_register_bits"] == 256
    assert by_variant[MatrixVariant.D_AFFINE]["architectural_variant"] is True
    assert by_variant[MatrixVariant.D_AFFINE][
        "compiler_programmable_alpha"
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


def test_one_mib_point_cannot_claim_unemitted_static_overlap() -> None:
    feasibility = build_static_overlap_feasibility(
        compiler=_real_compiler_evidence(),
        hardware=MatrixHardwarePoint(),
    )
    assert feasibility["variant_e_credit_allowed"] is False
    assert feasibility["models"]["nemotron3"]["minimum_additional_bytes"] >= 0
    assert feasibility["models"]["kimi_k3"]["minimum_additional_bytes"] >= 0
    assert all(
        record["fits_same_capacity"] is False
        for record in feasibility["models"].values()
    )


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
    assert mamba["D_affine"]["bank_stall_cycles"] == 0
    assert kda["D_affine"]["bank_stall_cycles"] == 0
    assert mamba["C_fixed"]["bank_stall_cycles"] == 0
    assert kda["C_fixed"]["bank_stall_cycles"] == 0
    assert mamba["affine_speedup_over_fixed"] == 1
    assert kda["affine_speedup_over_fixed"] == 1


def test_ordinary_attention_and_moe_matrix_lines_do_not_regress() -> None:
    hardware = MatrixHardwarePoint()
    compiler = load_compiler_evidence(str(COMPILER_ROOT))
    physical = build_physical_evidence(hardware)
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
    assert "Four-token official recurrence geometry" in EVIDENCE_LEVELS[
        "multi_token_recurrence"
    ]
    assert "symbolic PLENA weights" in EVIDENCE_LEVELS["full_model_timeline"]
    assert "No real-weight first-to-last-layer Rust execution" in EVIDENCE_LEVELS[
        "not_demonstrated"
    ]
