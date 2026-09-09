"""CPU-only tests for the matched decode-geometry artifact."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from .matched_decode_comparison import (
    ANCESTRY_SCHEMA,
    BF16_ORACLE_SCHEMA,
    HANDOFF_SCHEMA,
    INPUT_SCHEMA,
    NUMERICAL_RECEIPT_SCHEMA,
    POINT_SCHEMA,
    ancestry_receipt_id,
    bf16_oracle_receipt_id,
    build_comparison,
    content_hash,
    handoff_receipt_id,
    load_comparison,
    numerical_receipt_id,
    write_comparison,
)


def _numerical(profile_id: str, mlen: int, nominal: dict) -> dict:
    body = {
        "schema_version": NUMERICAL_RECEIPT_SCHEMA,
        "profile_id": profile_id,
        "evaluated_mlen": mlen,
        "nominal_precision_sha256": content_hash(nominal),
        "method_contract_sha256": "a" * 64,
        "source_receipt_sha256": str(mlen)[-1] * 64,
        "accuracy_scope": "same_split_teacher_forced_cached_decode",
        "sample_set_sha256": "6" * 64,
        "scored_tokens": 4096,
        "candidate_mean_token_nll": 2.01 if mlen == 1024 else 2.02,
        "bf16_mean_token_nll": 2.0,
        "state": "succeeded",
        "hardware_bit_parity_verified": False,
        "publication_rankable": False,
    }
    return {**body, "receipt_id": numerical_receipt_id(body)}


def _point(role: str, *, mlen: int, blen: int, tpot: float) -> dict:
    architecture = {
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
    }
    nominal = {
        "weight_format": "MXINT8",
        "activation_format": "MXINT8",
        "key_format": "MXINT8",
        "value_format": "MXINT8",
        "vector_format": "FP_E6M5",
        "block_size": 8,
    }
    candidate = {
        "MLEN": mlen,
        "BLEN": blen,
        "VLEN": mlen,
        "HLEN": 128,
        "BATCH": 8,
        "HBM_CHANNELS": 32,
        "HBM_GENERATION": "HBM2",
        "CHIP_COUNT": 8,
        "TP": 8,
        "KVP": 1,
        "LINK_PORTS": 1,
        "SRAM_POLICY": "streaming",
        "KV_HEAD_REUSE": False,
        "DRAIN_OVERLAPPED": False,
        "EXPERT_PARALLEL_MODE": "tensor_parallel",
    }
    profile_id = "dqp-" + ("1" if mlen == 1024 else "2") * 64
    return {
        "schema_version": POINT_SCHEMA,
        "role": role,
        "model": {
            "name": "Qwen/Qwen3-30B-A3B-Thinking-2507",
            "revision": "1" * 40,
            "tokenizer_revision": "2" * 40,
            "model_architecture": architecture,
            "architecture_sha256": content_hash(architecture),
        },
        "nominal_precision": nominal,
        "numerical_receipt": _numerical(profile_id, mlen, nominal),
        "hardware": candidate,
        "workload": {
            "scope": "steady_state_cached_q1",
            "query_length": 1,
            "input_seq": 512,
            "output_seq": 3072,
            "stride": 1,
            "runtime_hbm_reserve_bytes": 536870912,
            "kv_layout": "dense_selector",
        },
        "phase_contract": {
            "prefill_precision": "BF16",
            "decode_query_length": 1,
            "first_token_owner": "prefill",
            "decode_kv_admission": "quantize_once",
        },
        "clock_hz": 1e9,
        "resource_receipt": {
            "matrix_pe_equivalents_per_chip": mlen * blen,
            "aggregate_multiplier_count": 8 * 65_536,
            "system_area_mm2": 2000.0 if mlen == 1024 else 2100.0,
            "aggregate_hbm_capacity_bytes": 8 * 8 * 16 * 1024**3,
            "aggregate_hbm_bandwidth_bytes_per_s": 8 * 32 * 64e9,
            "aggregate_area_limit_mm2": 2500.0,
            "aggregate_hbm_capacity_limit_bytes": 8 * 8 * 16 * 1024**3,
            "aggregate_hbm_bandwidth_limit_bytes_per_s": 8 * 32 * 64e9,
            "resource_budget_sha256": "b" * 64,
            "resource_budget_feasible": True,
            "runtime_feasible": True,
            "timing_complete": True,
            "body_timing_complete": True,
            "broader_publication_rankable": False,
        },
        "output_head": {
            "location": "decode_local_mx_head",
            "semantic_contract_sha256": "c" * 64,
            "geometry_receipt_sha256": ("5" if mlen == 1024 else "4") * 64,
            "evaluated_mlen": mlen,
            "local_cost_complete": True,
            "idealizations": [],
        },
        "routing": {
            "kind": "routed_moe",
            "routing_source_kind": "analytic_expected",
            "routing_source_receipt_sha256": "d" * 64,
            "routing_semantics_sha256": "7" * 64,
            "placement_policy_sha256": "3" * 64,
            "geometry_timing_receipt_sha256": (
                "2" if mlen == 1024 else "1"
            ) * 64,
            "expert_parallel_mode": "tensor_parallel",
            "resident_expert_count": 128,
            "model_expert_count": 128,
            "timing_complete": True,
        },
        "timing": {
            "tpot_ms": tpot,
            "timing_tier": "stage_calibrated_analytic",
            "timing_evidence_id": "timing-" + str(mlen),
            "execution_mode": "legacy_aggregate_bandwidth",
            "metric_scope": "whole_model_decode_step_local_mx_head",
            "timing_valid": True,
        },
        "source": {
            "artifact_sha256": "e" * 64,
            "record_sha256": ("f" if mlen == 1024 else "0") * 64,
            "evaluator_id": "evaluator-id",
            "evaluator_provenance_sha256": "9" * 64,
        },
    }


def _input() -> dict:
    specialized = _point(
        "decode_specialized", mlen=1024, blen=64, tpot=8.0
    )
    shared = _point(
        "shared_plena_geometry", mlen=2048, blen=32, tpot=12.0
    )
    wire_elements = 2 * 48 * 4 * 128 * 512 * 8
    wire_bytes = wire_elements * 2
    link_bandwidth = 450e9
    handoff_body = {
        "schema_version": HANDOFF_SCHEMA,
        "model": specialized["model"],
        "workload_sha256": content_hash(specialized["workload"]),
        "phase_contract_sha256": content_hash(specialized["phase_contract"]),
        "source_point_record_sha256": "f" * 64,
        "input_artifact_id": "prefill-handoff-" + "3" * 64,
        "input_artifact_sha256": "3" * 64,
        "analysis_sha256": "4" * 64,
        "source_kind": "measured_prefill_handoff",
        "regime": "back_pressure",
        "transfer_mode": "bulk",
        "admission_scope": "full_bf16_read_plus_packed_write",
        "layers": 48,
        "kv_heads": 4,
        "head_dim": 128,
        "prompt_tokens": 512,
        "batch": 8,
        "wire_bits": 16,
        "wire_bytes": wire_bytes,
        "decode_cache_bytes": wire_elements,
        "decode_cache_effective_bits_per_element": 8.0,
        "nominal_precision_sha256": content_hash(
            specialized["nominal_precision"]
        ),
        "link_generation": "nvlink4",
        "link_bandwidth_bytes_per_s": link_bandwidth,
        "link_ports_used": 1,
        "effective_link_bandwidth_bytes_per_s": link_bandwidth,
        "transfer_ms": wire_bytes / link_bandwidth * 1000.0,
        "admission_bytes": wire_bytes + wire_elements,
        "admission_bandwidth_bytes_per_s": 900e9,
        "admission_bandwidth_policy": "measured_admission_artifact",
        "admission_bandwidth_source_sha256": "5" * 64,
        "admission_calibrated": False,
        "admission_calibration_id": None,
        "admission_evidence_tier": "declared_analytic",
        "admission_ms": (wire_bytes + wire_elements) / 900e9 * 1000.0,
        "decode_ready_wait_ms": 3.0,
        "publication_rankable": False,
    }
    bf16_body = {
        "schema_version": BF16_ORACLE_SCHEMA,
        "model": specialized["model"],
        "profile_id": "dqp-" + "8" * 64,
        "source_receipt_sha256": "7" * 64,
        "accuracy_scope": "same_split_teacher_forced_cached_decode",
        "evaluation_protocol_sha256": "1" * 64,
        "dataset_sha256": "2" * 64,
        "prompt_manifest_sha256": "3" * 64,
        "seed_receipt_sha256": "4" * 64,
        "mean_nll_receipt_sha256": "5" * 64,
        "sample_set_sha256": "6" * 64,
        "scored_tokens": 4096,
        "mean_token_nll": 2.0,
        "latency_role": "accuracy_only_not_hardware_priced",
        "state": "succeeded",
    }
    ancestry_body = {
        "schema_version": ANCESTRY_SCHEMA,
        "selected_source_receipt_id": "selected-decode-source-" + "a" * 64,
        "selected_source_profile_id": specialized["numerical_receipt"][
            "profile_id"
        ],
        "selected_candidate_id": "hw-" + content_hash(specialized["hardware"]),
        "selected_hardware": dict(specialized["hardware"]),
        "selected_replay_record_sha256": "8" * 64,
        "derived_specialized_profile_id": specialized["numerical_receipt"][
            "profile_id"
        ],
        "numerical_derivation_receipt_sha256": "9" * 64,
        "derived_specialized_candidate_id": (
            "hw-" + content_hash(specialized["hardware"])
        ),
        "derived_specialized_hardware": dict(specialized["hardware"]),
        "derivation_rule": (
            "preserve_all_axes_except_blen_set_blen_to_65536_div_mlen"
        ),
        "selected_was_already_multiplier_matched": True,
    }
    return {
        "schema_version": INPUT_SCHEMA,
        "arms": {
            "decode_specialized": specialized,
            "shared_plena_geometry": shared,
        },
        "selection_ancestry": {
            **ancestry_body,
            "receipt_id": ancestry_receipt_id(ancestry_body),
        },
        "handoff": {
            **handoff_body,
            "receipt_id": handoff_receipt_id(handoff_body),
        },
        "bf16_accuracy_oracle": {
            **bf16_body,
            "receipt_id": bf16_oracle_receipt_id(bf16_body),
        },
    }


class MatchedDecodeComparisonTests(unittest.TestCase):
    def test_derives_only_matched_decode_latency_metrics(self) -> None:
        artifact = build_comparison(_input())
        result = artifact["result"]
        self.assertEqual(
            result["comparison_status"],
            "multiplier_and_common_envelope_matched_area_disclosed",
        )
        self.assertEqual(result["evidence_class"], "analytic_nonpublication")
        self.assertFalse(result["publication_rankable"])
        latency = result["latency_metrics"]
        handoff_ms = (
            _input()["handoff"]["transfer_ms"]
            + _input()["handoff"]["admission_ms"]
            + 3.0
        )
        self.assertAlmostEqual(
            latency["one_time_handoff_ms"], handoff_ms
        )
        self.assertEqual(latency["handoff_amortization_decode_steps"], 3071)
        self.assertAlmostEqual(
            latency["handoff_amortized_decode_side_service_ms"],
            8.0 + handoff_ms / 3071,
        )
        self.assertAlmostEqual(
            latency["raw_tpot_speedup_shared_over_specialized"], 1.5
        )
        self.assertAlmostEqual(
            result["accuracy_metrics"][
                "shared_plena_geometry_relative_perplexity_vs_bf16"
            ],
            __import__("math").exp(0.02),
        )
        self.assertFalse(
            result["resource_fairness"]["system_area_mm2"]["exact_match"]
        )
        self.assertNotIn(
            "goodput", json.dumps(artifact["result"]["latency_metrics"]).casefold()
        )

    def test_non_geometry_mismatch_fails_closed(self) -> None:
        value = _input()
        value["arms"]["shared_plena_geometry"]["hardware"]["BATCH"] = 16
        with self.assertRaisesRegex(ValueError, "non-geometry"):
            build_comparison(value)

    def test_source_mlen_receipt_cannot_authorize_paper_geometry(self) -> None:
        value = _input()
        receipt = value["arms"]["shared_plena_geometry"]["numerical_receipt"]
        receipt["evaluated_mlen"] = 1024
        receipt["receipt_id"] = numerical_receipt_id(receipt)
        with self.assertRaisesRegex(ValueError, "MLEN differs"):
            build_comparison(value)

    def test_handoff_cannot_undercount_authenticated_model_dimensions(self) -> None:
        value = _input()
        receipt = value["handoff"]
        receipt["layers"] = 24
        receipt["wire_bytes"] //= 2
        receipt["receipt_id"] = handoff_receipt_id(receipt)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            build_comparison(value)

    def test_aggregate_multiplier_receipt_is_recomputed(self) -> None:
        value = _input()
        value["arms"]["decode_specialized"]["resource_receipt"][
            "aggregate_multiplier_count"
        ] += 1
        with self.assertRaisesRegex(ValueError, "differs from geometry"):
            build_comparison(value)

    def test_ancestry_cannot_authorize_unequal_matrix_resources(self) -> None:
        value = _input()
        specialized = value["arms"]["decode_specialized"]
        specialized["hardware"]["BLEN"] = 32
        specialized["resource_receipt"]["matrix_pe_equivalents_per_chip"] = 32768
        specialized["resource_receipt"]["aggregate_multiplier_count"] = 8 * 32768
        with self.assertRaisesRegex(ValueError, "derived specialized hardware"):
            build_comparison(value)

    def test_common_envelope_tamper_fails_closed(self) -> None:
        value = _input()
        value["arms"]["shared_plena_geometry"]["resource_receipt"][
            "aggregate_hbm_capacity_limit_bytes"
        ] += 1
        with self.assertRaisesRegex(ValueError, "resource assumption"):
            build_comparison(value)

    def test_over_ceiling_pair_retains_points_but_suppresses_ratios(self) -> None:
        value = _input()
        for point in value["arms"].values():
            resource = point["resource_receipt"]
            resource["aggregate_area_limit_mm2"] = 1000.0
            resource["resource_budget_feasible"] = False
        artifact = build_comparison(value)
        result = artifact["result"]
        self.assertEqual(
            result["comparison_status"],
            "multiplier_or_common_envelope_unmatched",
        )
        self.assertIsNone(
            result["latency_metrics"]["raw_tpot_speedup_shared_over_specialized"]
        )

    def test_asymmetric_ceiling_outcome_is_retained_without_a_ratio(self) -> None:
        value = _input()
        shared = value["arms"]["shared_plena_geometry"]["resource_receipt"]
        shared["aggregate_area_limit_mm2"] = 2050.0
        value["arms"]["decode_specialized"]["resource_receipt"][
            "aggregate_area_limit_mm2"
        ] = 2050.0
        shared["resource_budget_feasible"] = False
        artifact = build_comparison(value)
        result = artifact["result"]
        self.assertEqual(
            result["comparison_status"],
            "multiplier_or_common_envelope_unmatched",
        )
        self.assertIsNone(
            result["latency_metrics"]["raw_tpot_speedup_shared_over_specialized"]
        )
        self.assertTrue(
            value["arms"]["decode_specialized"]["resource_receipt"][
                "resource_budget_feasible"
            ]
        )
        self.assertFalse(shared["resource_budget_feasible"])

    def test_numerical_method_mismatch_fails_closed(self) -> None:
        value = _input()
        receipt = value["arms"]["shared_plena_geometry"]["numerical_receipt"]
        receipt["method_contract_sha256"] = "0" * 64
        receipt["receipt_id"] = numerical_receipt_id(receipt)
        with self.assertRaisesRegex(ValueError, "different numerical methods"):
            build_comparison(value)

    def test_numerical_receipt_cannot_claim_hardware_bit_parity(self) -> None:
        value = _input()
        receipt = value["arms"]["decode_specialized"]["numerical_receipt"]
        receipt["hardware_bit_parity_verified"] = True
        receipt["receipt_id"] = numerical_receipt_id(receipt)
        with self.assertRaisesRegex(ValueError, "not hardware bit parity"):
            build_comparison(value)

    def test_routing_semantics_and_source_must_match(self) -> None:
        for field in ("routing_source_receipt_sha256", "routing_semantics_sha256"):
            value = _input()
            value["arms"]["shared_plena_geometry"]["routing"][field] = "0" * 64
            with self.assertRaisesRegex(ValueError, "routing policy"):
                build_comparison(value)

    def test_geometry_receipts_must_be_arm_specific(self) -> None:
        for section, field, message in (
            ("output_head", "geometry_receipt_sha256", "local-head geometry"),
            ("routing", "geometry_timing_receipt_sha256", "routing geometry"),
        ):
            value = _input()
            specialized = value["arms"]["decode_specialized"][section]
            value["arms"]["shared_plena_geometry"][section][field] = specialized[
                field
            ]
            with self.assertRaisesRegex(ValueError, message):
                build_comparison(value)

    def test_admission_time_must_match_full_traffic(self) -> None:
        value = _input()
        receipt = value["handoff"]
        receipt["admission_ms"] *= 0.5
        receipt["receipt_id"] = handoff_receipt_id(receipt)
        with self.assertRaisesRegex(ValueError, "admission time differs"):
            build_comparison(value)

    def test_analytic_admission_must_use_matched_hbm_roofline(self) -> None:
        value = _input()
        receipt = value["handoff"]
        receipt["source_kind"] = "config_bound_analytic_handoff"
        receipt["regime"] = "serial_transfer_plus_admission_no_queue_wait"
        receipt["input_artifact_id"] = "analytic-handoff-" + receipt[
            "input_artifact_sha256"
        ]
        receipt["decode_ready_wait_ms"] = 0.0
        receipt["admission_bandwidth_policy"] = (
            "matched_candidate_aggregate_hbm_roofline"
        )
        receipt["admission_bandwidth_source_sha256"] = content_hash(
            value["arms"]["decode_specialized"]["resource_receipt"]
        )
        receipt["receipt_id"] = handoff_receipt_id(receipt)
        with self.assertRaisesRegex(ValueError, "matched HBM envelope"):
            build_comparison(value)

    def test_analytic_zero_queue_wait_is_explicit_and_tamper_checked(self) -> None:
        value = _input()
        receipt = value["handoff"]
        receipt["source_kind"] = "config_bound_analytic_handoff"
        receipt["regime"] = "serial_transfer_plus_admission_no_queue_wait"
        receipt["input_artifact_id"] = "analytic-handoff-" + receipt[
            "input_artifact_sha256"
        ]
        receipt["decode_ready_wait_ms"] = 0.0
        receipt["admission_bandwidth_policy"] = (
            "matched_candidate_aggregate_hbm_roofline"
        )
        receipt["admission_bandwidth_source_sha256"] = content_hash(
            value["arms"]["decode_specialized"]["resource_receipt"]
        )
        receipt["admission_bandwidth_bytes_per_s"] = value["arms"][
            "decode_specialized"
        ]["resource_receipt"]["aggregate_hbm_bandwidth_bytes_per_s"]
        receipt["admission_ms"] = receipt["admission_bytes"] / receipt[
            "admission_bandwidth_bytes_per_s"
        ] * 1000.0
        receipt["receipt_id"] = handoff_receipt_id(receipt)
        artifact = build_comparison(value)
        self.assertTrue(
            artifact["result"]["handoff_contract"][
                "queue_wait_idealized_to_zero"
            ]
        )
        receipt["decode_ready_wait_ms"] = 1.0
        receipt["receipt_id"] = handoff_receipt_id(receipt)
        with self.assertRaisesRegex(ValueError, "zero queue-wait"):
            build_comparison(value)

    def test_illegal_selected_derivation_fails_closed(self) -> None:
        value = _input()
        ancestry = value["selection_ancestry"]
        ancestry["selected_hardware"]["MLEN"] = 1000
        ancestry["selected_candidate_id"] = "hw-" + content_hash(
            ancestry["selected_hardware"]
        )
        ancestry["receipt_id"] = ancestry_receipt_id(ancestry)
        with self.assertRaisesRegex(ValueError, "integer matched BLEN"):
            build_comparison(value)

    def test_immutable_writer_and_replay_detect_tampering(self) -> None:
        artifact = build_comparison(_input())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            write_comparison(path, artifact)
            self.assertEqual(load_comparison(path), artifact)
            tampered = json.loads(path.read_bytes())
            tampered["result"]["latency_metrics"][
                "decode_specialized_tpot_ms"
            ] = 1.0
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "replay"):
                load_comparison(path)


if __name__ == "__main__":
    unittest.main()
    ancestry_receipt_id,
