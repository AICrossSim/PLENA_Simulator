"""Fail-closed matched decode-geometry comparison contracts.

This module deliberately compares *decode-stage geometry*, not complete
aggregated and disaggregated serving systems.  It has no throughput-SLO or
goodput model.  The shared-PLENA arm is the published PLENA geometry
(``BLEN=32, MLEN=VLEN=2048``); every non-geometry assumption must match the
decode-specialized arm.

The comparison consumes already priced, evidence-bound points.  Repricing is
owned by the software adapter in ``decode_dse.software``.  Keeping arithmetic
and validation here makes the result portable while preventing a reporting
script from silently weakening the fairness contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


INPUT_SCHEMA = "plena-matched-decode-geometry-input/v1"
ARTIFACT_SCHEMA = "plena-matched-decode-geometry-comparison/v1"
POINT_SCHEMA = "plena-decode-stage-point/v1"
NUMERICAL_RECEIPT_SCHEMA = "plena-exact-mlen-numerical-receipt/v1"
BF16_ORACLE_SCHEMA = "plena-bf16-accuracy-oracle-receipt/v1"
HANDOFF_SCHEMA = "plena-handoff-service-receipt/v1"
ANCESTRY_SCHEMA = "plena-matched-specialized-derivation/v1"
CLAIM_SCOPE = "controlled_decode_stage_geometry_ablation"

SHARED_PLENA_GEOMETRY = {"MLEN": 2048, "BLEN": 32, "VLEN": 2048}
SHARED_PLENA_PE_EQUIVALENTS_PER_CHIP = 65_536
HANDOFF_LINK_BANDWIDTH_BYTES_PER_S = {
    "nvlink3": 300e9,
    "nvlink4": 450e9,
    "ualink": 400e9,
    "pcie5": 64e9,
}

_POINT_FIELDS = {
    "schema_version",
    "role",
    "model",
    "nominal_precision",
    "numerical_receipt",
    "hardware",
    "workload",
    "phase_contract",
    "clock_hz",
    "resource_receipt",
    "output_head",
    "routing",
    "timing",
    "source",
}
_MODEL_FIELDS = {
    "name",
    "revision",
    "tokenizer_revision",
    "model_architecture",
    "architecture_sha256",
}
_NUMERICAL_FIELDS = {
    "schema_version",
    "profile_id",
    "evaluated_mlen",
    "nominal_precision_sha256",
    "method_contract_sha256",
    "source_receipt_sha256",
    "accuracy_scope",
    "sample_set_sha256",
    "scored_tokens",
    "candidate_mean_token_nll",
    "bf16_mean_token_nll",
    "state",
    "hardware_bit_parity_verified",
    "publication_rankable",
    "receipt_id",
}
_WORKLOAD_FIELDS = {
    "scope",
    "query_length",
    "input_seq",
    "output_seq",
    "stride",
    "runtime_hbm_reserve_bytes",
    "kv_layout",
}
_RESOURCE_FIELDS = {
    "matrix_pe_equivalents_per_chip",
    "aggregate_multiplier_count",
    "system_area_mm2",
    "aggregate_hbm_capacity_bytes",
    "aggregate_hbm_bandwidth_bytes_per_s",
    "aggregate_area_limit_mm2",
    "aggregate_hbm_capacity_limit_bytes",
    "aggregate_hbm_bandwidth_limit_bytes_per_s",
    "resource_budget_sha256",
    "resource_budget_feasible",
    "runtime_feasible",
    "timing_complete",
    "body_timing_complete",
    "broader_publication_rankable",
}
_OUTPUT_HEAD_FIELDS = {
    "location",
    "semantic_contract_sha256",
    "geometry_receipt_sha256",
    "evaluated_mlen",
    "local_cost_complete",
    "idealizations",
}
_ROUTING_FIELDS = {
    "kind",
    "routing_source_kind",
    "routing_source_receipt_sha256",
    "routing_semantics_sha256",
    "placement_policy_sha256",
    "geometry_timing_receipt_sha256",
    "expert_parallel_mode",
    "resident_expert_count",
    "model_expert_count",
    "timing_complete",
}
_TIMING_FIELDS = {
    "tpot_ms",
    "timing_tier",
    "timing_evidence_id",
    "execution_mode",
    "metric_scope",
    "timing_valid",
}
_SOURCE_FIELDS = {
    "artifact_sha256",
    "record_sha256",
    "evaluator_id",
    "evaluator_provenance_sha256",
}
_HANDOFF_FIELDS = {
    "schema_version",
    "model",
    "workload_sha256",
    "phase_contract_sha256",
    "source_point_record_sha256",
    "input_artifact_id",
    "input_artifact_sha256",
    "analysis_sha256",
    "source_kind",
    "regime",
    "transfer_mode",
    "admission_scope",
    "layers",
    "kv_heads",
    "head_dim",
    "prompt_tokens",
    "batch",
    "wire_bits",
    "wire_bytes",
    "decode_cache_bytes",
    "decode_cache_effective_bits_per_element",
    "nominal_precision_sha256",
    "link_generation",
    "link_bandwidth_bytes_per_s",
    "link_ports_used",
    "effective_link_bandwidth_bytes_per_s",
    "transfer_ms",
    "admission_bytes",
    "admission_bandwidth_bytes_per_s",
    "admission_bandwidth_policy",
    "admission_bandwidth_source_sha256",
    "admission_calibrated",
    "admission_calibration_id",
    "admission_evidence_tier",
    "admission_ms",
    "decode_ready_wait_ms",
    "publication_rankable",
    "receipt_id",
}
_BF16_FIELDS = {
    "schema_version",
    "model",
    "profile_id",
    "source_receipt_sha256",
    "accuracy_scope",
    "evaluation_protocol_sha256",
    "dataset_sha256",
    "prompt_manifest_sha256",
    "seed_receipt_sha256",
    "mean_nll_receipt_sha256",
    "sample_set_sha256",
    "scored_tokens",
    "mean_token_nll",
    "latency_role",
    "state",
    "receipt_id",
}
_ANCESTRY_FIELDS = {
    "schema_version",
    "selected_source_receipt_id",
    "selected_source_profile_id",
    "selected_candidate_id",
    "selected_hardware",
    "selected_replay_record_sha256",
    "derived_specialized_profile_id",
    "numerical_derivation_receipt_sha256",
    "derived_specialized_candidate_id",
    "derived_specialized_hardware",
    "derivation_rule",
    "selected_was_already_multiplier_matched",
    "receipt_id",
}


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + ("\n" if newline else "")
    ).encode("utf-8")


def content_hash(value: Any) -> str:
    """Return the canonical SHA-256 used by every comparison receipt."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields differ from the schema")
    return json.loads(_canonical_bytes(dict(value)))


def _sha256(value: Any, label: str) -> str:
    token = str(value)
    if (
        len(token) != 64
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return token


def _positive_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _nonnegative_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _reject_goodput(value: Any, *, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "goodput" in str(key).casefold():
                raise ValueError(f"{path}.{key} is outside the metric whitelist")
            _reject_goodput(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_goodput(item, path=f"{path}[{index}]")


def numerical_receipt_id(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_id", None)
    return "mlen-numerical-" + content_hash(body)


def bf16_oracle_receipt_id(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_id", None)
    return "bf16-accuracy-oracle-" + content_hash(body)


def handoff_receipt_id(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_id", None)
    return "handoff-service-" + content_hash(body)


def ancestry_receipt_id(receipt: Mapping[str, Any]) -> str:
    body = dict(receipt)
    body.pop("receipt_id", None)
    return "matched-specialized-derivation-" + content_hash(body)


def _validate_numerical_receipt(
    raw: Any,
    *,
    nominal_precision: Mapping[str, Any],
    hardware_mlen: int,
) -> dict[str, Any]:
    receipt = _object(raw, _NUMERICAL_FIELDS, "numerical receipt")
    if receipt["schema_version"] != NUMERICAL_RECEIPT_SCHEMA:
        raise ValueError("unsupported numerical receipt schema")
    if not isinstance(receipt["profile_id"], str) or not receipt["profile_id"]:
        raise ValueError("numerical receipt profile identity is missing")
    if _positive_integer(receipt["evaluated_mlen"], "evaluated MLEN") != hardware_mlen:
        raise ValueError("numerical receipt MLEN differs from hardware MLEN")
    expected_precision_hash = content_hash(nominal_precision)
    if receipt["nominal_precision_sha256"] != expected_precision_hash:
        raise ValueError("numerical receipt nominal precision differs")
    _sha256(receipt["method_contract_sha256"], "numerical method contract")
    _sha256(receipt["source_receipt_sha256"], "numerical source receipt")
    if not isinstance(receipt["accuracy_scope"], str) or not receipt["accuracy_scope"]:
        raise ValueError("numerical accuracy scope must be explicit")
    _sha256(receipt["sample_set_sha256"], "numerical sample set")
    _positive_integer(receipt["scored_tokens"], "numerical scored tokens")
    for name in ("candidate_mean_token_nll", "bf16_mean_token_nll"):
        receipt[name] = _nonnegative_number(receipt[name], name)
    if receipt["state"] != "succeeded":
        raise ValueError("exact-MLEN numerical receipt did not succeed")
    if receipt["hardware_bit_parity_verified"] is not False:
        raise ValueError(
            "v1 numerical receipt is MLEN rounding evidence, not hardware bit parity"
        )
    _boolean(receipt["publication_rankable"], "numerical publication status")
    if receipt["receipt_id"] != numerical_receipt_id(receipt):
        raise ValueError("numerical receipt identity is inconsistent")
    return receipt


def _validate_point(raw: Any, expected_role: str) -> dict[str, Any]:
    point = _object(raw, _POINT_FIELDS, f"{expected_role} point")
    if point["schema_version"] != POINT_SCHEMA or point["role"] != expected_role:
        raise ValueError(f"{expected_role} point identity is invalid")

    model = _object(point["model"], _MODEL_FIELDS, "model")
    if not isinstance(model["name"], str) or not model["name"]:
        raise ValueError("model name must be explicit")
    for name in ("revision", "tokenizer_revision"):
        revision = model[name]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise ValueError(f"model {name} must be an exact lowercase git SHA")
    architecture = model["model_architecture"]
    if not isinstance(architecture, Mapping) or not architecture:
        raise ValueError("model architecture must be a non-empty object")
    architecture = json.loads(_canonical_bytes(dict(architecture)))
    required_dimensions = {
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
    }
    if not required_dimensions.issubset(architecture):
        raise ValueError("model architecture lacks decode dimensions")
    for name in required_dimensions:
        _positive_integer(architecture[name], f"model architecture {name}")
    for name in ("num_experts", "num_experts_per_tok"):
        if name in architecture:
            _positive_integer(architecture[name], f"model architecture {name}")
    model["model_architecture"] = architecture
    if model["architecture_sha256"] != content_hash(architecture):
        raise ValueError("model architecture SHA-256 differs from its dimensions")
    point["model"] = model

    nominal = point["nominal_precision"]
    if not isinstance(nominal, Mapping) or not nominal:
        raise ValueError("nominal physical precision must be a non-empty object")
    nominal = json.loads(_canonical_bytes(dict(nominal)))
    point["nominal_precision"] = nominal

    hardware = point["hardware"]
    if not isinstance(hardware, Mapping):
        raise TypeError("hardware candidate must be an object")
    hardware = json.loads(_canonical_bytes(dict(hardware)))
    required_hardware = {
        "MLEN",
        "BLEN",
        "VLEN",
        "HLEN",
        "BATCH",
        "HBM_CHANNELS",
        "HBM_GENERATION",
        "CHIP_COUNT",
        "TP",
        "KVP",
        "LINK_PORTS",
        "SRAM_POLICY",
        "KV_HEAD_REUSE",
        "DRAIN_OVERLAPPED",
    }
    if not required_hardware.issubset(hardware) or set(hardware) - (
        required_hardware | {"EXPERT_PARALLEL_MODE"}
    ):
        raise ValueError("hardware candidate fields differ from the matched schema")
    for name in (
        "MLEN",
        "BLEN",
        "VLEN",
        "HLEN",
        "BATCH",
        "HBM_CHANNELS",
        "CHIP_COUNT",
        "TP",
        "KVP",
    ):
        _positive_integer(hardware[name], f"hardware {name}")
    if hardware["VLEN"] != hardware["MLEN"]:
        raise ValueError("matched decode candidates require VLEN == MLEN")
    if (
        hardware["MLEN"] % hardware["BLEN"]
        or hardware["MLEN"] % hardware["HLEN"]
        or not hardware["BLEN"] <= hardware["HLEN"] <= hardware["MLEN"]
    ):
        raise ValueError("hardware matrix/head tiling geometry is illegal")
    if hardware["TP"] * hardware["KVP"] != hardware["CHIP_COUNT"]:
        raise ValueError("candidate CHIP_COUNT must equal TP * KVP")
    if (
        isinstance(hardware["LINK_PORTS"], bool)
        or not isinstance(hardware["LINK_PORTS"], int)
        or hardware["LINK_PORTS"] < 0
    ):
        raise ValueError("hardware LINK_PORTS must be non-negative")
    required_ports = int(hardware["TP"] > 1) + int(hardware["KVP"] > 1)
    if (
        (hardware["CHIP_COUNT"] == 1 and hardware["LINK_PORTS"] != 0)
        or (
            hardware["CHIP_COUNT"] > 1
            and hardware["LINK_PORTS"] < required_ports
        )
    ):
        raise ValueError("hardware link ports cannot serve the active topology")
    for name in ("HBM_GENERATION", "SRAM_POLICY"):
        if not isinstance(hardware[name], str) or not hardware[name]:
            raise ValueError(f"hardware {name} must be explicit")
    for name in ("KV_HEAD_REUSE", "DRAIN_OVERLAPPED"):
        _boolean(hardware[name], f"hardware {name}")
    if "EXPERT_PARALLEL_MODE" in hardware and (
        not isinstance(hardware["EXPERT_PARALLEL_MODE"], str)
        or not hardware["EXPERT_PARALLEL_MODE"]
    ):
        raise ValueError("expert parallel mode must be explicit")
    point["hardware"] = hardware

    point["numerical_receipt"] = _validate_numerical_receipt(
        point["numerical_receipt"],
        nominal_precision=nominal,
        hardware_mlen=hardware["MLEN"],
    )

    workload = _object(point["workload"], _WORKLOAD_FIELDS, "workload")
    if workload["scope"] != "steady_state_cached_q1" or workload["query_length"] != 1:
        raise ValueError("comparison workload must be steady-state cached Q=1")
    for name in ("input_seq", "output_seq", "stride"):
        _positive_integer(workload[name], f"workload {name}")
    if workload["output_seq"] < 2:
        raise ValueError("handoff amortization requires at least two output tokens")
    if (
        isinstance(workload["runtime_hbm_reserve_bytes"], bool)
        or not isinstance(workload["runtime_hbm_reserve_bytes"], int)
        or workload["runtime_hbm_reserve_bytes"] < 0
    ):
        raise ValueError("runtime HBM reserve must be non-negative")
    if not isinstance(workload["kv_layout"], str) or not workload["kv_layout"]:
        raise ValueError("KV layout must be explicit")

    phase = point["phase_contract"]
    if not isinstance(phase, Mapping) or not phase:
        raise ValueError("phase contract must be explicit")
    phase = json.loads(_canonical_bytes(dict(phase)))
    if phase.get("decode_query_length") != 1:
        raise ValueError("phase contract must bind decode Q=1")
    if phase.get("first_token_owner") != "prefill":
        raise ValueError("handoff amortization requires prefill-owned first token")
    point["phase_contract"] = phase

    point["clock_hz"] = _positive_number(point["clock_hz"], "clock")

    resource = _object(
        point["resource_receipt"], _RESOURCE_FIELDS, "resource receipt"
    )
    expected_pes = hardware["MLEN"] * hardware["BLEN"]
    if resource["matrix_pe_equivalents_per_chip"] != expected_pes:
        raise ValueError("matrix PE-equivalent receipt differs from geometry")
    multiplier_count = _positive_integer(
        resource["aggregate_multiplier_count"],
        "aggregate multiplier count",
    )
    if multiplier_count != expected_pes * hardware["CHIP_COUNT"]:
        raise ValueError("aggregate multiplier count differs from geometry")
    resource["system_area_mm2"] = _positive_number(
        resource["system_area_mm2"], "system area"
    )
    for name in (
        "aggregate_hbm_capacity_bytes",
        "aggregate_hbm_capacity_limit_bytes",
    ):
        _positive_integer(resource[name], name)
    for name in (
        "aggregate_hbm_bandwidth_bytes_per_s",
        "aggregate_area_limit_mm2",
        "aggregate_hbm_bandwidth_limit_bytes_per_s",
    ):
        resource[name] = _positive_number(resource[name], name)
    _sha256(resource["resource_budget_sha256"], "resource budget")
    for name in (
        "resource_budget_feasible",
        "runtime_feasible",
        "timing_complete",
        "body_timing_complete",
        "broader_publication_rankable",
    ):
        _boolean(resource[name], f"resource {name}")
    if not all(
        resource[name]
        for name in (
            "runtime_feasible",
            "timing_complete",
            "body_timing_complete",
        )
    ):
        raise ValueError("decode point lacks complete feasible timing evidence")
    within_limits = (
        resource["system_area_mm2"] <= resource["aggregate_area_limit_mm2"]
        and resource["aggregate_hbm_capacity_bytes"]
        <= resource["aggregate_hbm_capacity_limit_bytes"]
        and resource["aggregate_hbm_bandwidth_bytes_per_s"]
        <= resource["aggregate_hbm_bandwidth_limit_bytes_per_s"]
    )
    if resource["resource_budget_feasible"] is not within_limits:
        raise ValueError("resource feasibility flag differs from the explicit ledger")

    head = _object(point["output_head"], _OUTPUT_HEAD_FIELDS, "output head")
    if head["location"] != "decode_local_mx_head":
        raise ValueError("comparison requires the local decode MX head")
    _sha256(head["semantic_contract_sha256"], "local output-head semantics")
    _sha256(head["geometry_receipt_sha256"], "local output-head geometry receipt")
    if head["evaluated_mlen"] != hardware["MLEN"]:
        raise ValueError("local output-head receipt MLEN differs from hardware")
    if _boolean(head["local_cost_complete"], "local-head completeness") is not True:
        raise ValueError("local output-head cost is incomplete")
    if not isinstance(head["idealizations"], list) or any(
        not isinstance(value, str) for value in head["idealizations"]
    ):
        raise TypeError("output-head idealizations must be a string list")

    routing = _object(point["routing"], _ROUTING_FIELDS, "routing")
    if routing["kind"] not in {"dense", "routed_moe"}:
        raise ValueError("routing kind is unsupported")
    if routing["routing_source_kind"] not in {
        "analytic_expected",
        "verified_trace",
    }:
        raise ValueError("routing source kind is unsupported")
    _sha256(
        routing["routing_source_receipt_sha256"],
        "routing source receipt",
    )
    _sha256(routing["routing_semantics_sha256"], "routing semantics")
    _sha256(routing["placement_policy_sha256"], "routing placement policy")
    _sha256(
        routing["geometry_timing_receipt_sha256"],
        "routing geometry timing receipt",
    )
    if _boolean(routing["timing_complete"], "routing timing completeness") is not True:
        raise ValueError("routing timing evidence is incomplete")
    if routing["kind"] == "dense":
        if architecture.get("num_experts", 1) > 1:
            raise ValueError("routed-MoE architecture cannot use dense routing")
        if any(
            routing[name] is not None
            for name in (
                "expert_parallel_mode",
                "resident_expert_count",
                "model_expert_count",
            )
        ):
            raise ValueError("dense routing cannot carry expert fields")
    else:
        if architecture.get("num_experts", 1) <= 1:
            raise ValueError("dense architecture cannot use routed-MoE routing")
        if (
            not isinstance(routing["expert_parallel_mode"], str)
            or not routing["expert_parallel_mode"]
        ):
            raise ValueError("routed MoE expert mapping must be explicit")
        resident = _positive_integer(
            routing["resident_expert_count"], "resident expert count"
        )
        total = _positive_integer(routing["model_expert_count"], "model expert count")
        if resident != total:
            raise ValueError("comparison requires every model expert to remain resident")
        if total != architecture["num_experts"]:
            raise ValueError("routing expert count differs from model architecture")
        if hardware.get("EXPERT_PARALLEL_MODE") != routing["expert_parallel_mode"]:
            raise ValueError("hardware and routing expert-parallel modes differ")

    timing = _object(point["timing"], _TIMING_FIELDS, "timing")
    timing["tpot_ms"] = _positive_number(timing["tpot_ms"], "decode TPOT")
    for name in (
        "timing_tier",
        "timing_evidence_id",
        "execution_mode",
        "metric_scope",
    ):
        if not isinstance(timing[name], str) or not timing[name]:
            raise ValueError(f"timing {name} must be explicit")
    if timing["metric_scope"] != "whole_model_decode_step_local_mx_head":
        raise ValueError("TPOT metric scope does not include the local MX head")
    if _boolean(timing["timing_valid"], "timing validity") is not True:
        raise ValueError("decode timing evidence is not valid")

    source = _object(point["source"], _SOURCE_FIELDS, "point source")
    for name in (
        "artifact_sha256",
        "record_sha256",
        "evaluator_provenance_sha256",
    ):
        _sha256(source[name], f"source {name}")
    if not isinstance(source["evaluator_id"], str) or not source["evaluator_id"]:
        raise ValueError("evaluator identity must be explicit")
    return point


def _validate_handoff(
    raw: Any,
    *,
    specialized: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _object(raw, _HANDOFF_FIELDS, "handoff receipt")
    if receipt["schema_version"] != HANDOFF_SCHEMA:
        raise ValueError("unsupported handoff-service schema")
    model = _object(receipt["model"], _MODEL_FIELDS, "handoff model")
    if model != specialized["model"]:
        raise ValueError("handoff and decode model identities differ")
    if receipt["workload_sha256"] != content_hash(specialized["workload"]):
        raise ValueError("handoff workload binding differs from decode")
    if receipt["phase_contract_sha256"] != content_hash(
        specialized["phase_contract"]
    ):
        raise ValueError("handoff phase binding differs from decode")
    if receipt["source_point_record_sha256"] != specialized["source"][
        "record_sha256"
    ]:
        raise ValueError("handoff source-point binding differs")
    input_sha = _sha256(receipt["input_artifact_sha256"], "handoff input")
    source_kind = receipt["source_kind"]
    prefixes = {
        "measured_prefill_handoff": "prefill-handoff-",
        "config_bound_analytic_handoff": "analytic-handoff-",
    }
    if source_kind not in prefixes or receipt["input_artifact_id"] != (
        prefixes[source_kind] + input_sha
    ):
        raise ValueError("handoff input artifact identity is invalid")
    _sha256(receipt["analysis_sha256"], "handoff analysis")
    if receipt["regime"] not in {
        "back_pressure",
        "serial_transfer_plus_admission_no_queue_wait",
    }:
        raise ValueError(
            "handoff-amortized decode-side service time requires the "
            "sealed serial-transfer regime"
        )
    if source_kind == "measured_prefill_handoff" and receipt["regime"] != (
        "back_pressure"
    ):
        raise ValueError("measured handoff requires the back-pressure regime")
    if source_kind == "config_bound_analytic_handoff" and (
        receipt["regime"] != "serial_transfer_plus_admission_no_queue_wait"
        or receipt["decode_ready_wait_ms"] != 0
        or receipt["publication_rankable"] is not False
    ):
        raise ValueError("analytic handoff must disclose its zero queue-wait bound")
    if receipt["transfer_mode"] != "bulk":
        raise ValueError("handoff service must use the full bulk transfer")
    if receipt["admission_scope"] != "full_bf16_read_plus_packed_write":
        raise ValueError("handoff service must include full KV admission")
    for name in ("layers", "kv_heads", "head_dim", "prompt_tokens", "batch"):
        _positive_integer(receipt[name], f"handoff {name}")
    architecture = specialized["model"]["model_architecture"]
    expected_dimensions = {
        "layers": architecture["num_hidden_layers"],
        "kv_heads": architecture["num_key_value_heads"],
        "head_dim": architecture["head_dim"],
    }
    if any(receipt[name] != value for name, value in expected_dimensions.items()):
        raise ValueError("handoff KV dimensions differ from model architecture")
    if receipt["prompt_tokens"] != specialized["workload"]["input_seq"]:
        raise ValueError("handoff prompt length differs from decode workload")
    if receipt["batch"] != specialized["hardware"]["BATCH"]:
        raise ValueError("handoff batch differs from decode candidate")
    if receipt["wire_bits"] != 16:
        raise ValueError("prefill-to-decode wire precision must be BF16")
    _positive_number(receipt["wire_bytes"], "wire bytes")
    _positive_number(receipt["decode_cache_bytes"], "decode-cache bytes")
    expected_wire = (
        2
        * receipt["layers"]
        * receipt["kv_heads"]
        * receipt["head_dim"]
        * receipt["prompt_tokens"]
        * receipt["batch"]
        * 2
    )
    if receipt["wire_bytes"] != expected_wire:
        raise ValueError("BF16 handoff wire ledger does not conserve KV bytes")
    if receipt["decode_cache_bytes"] > receipt["wire_bytes"]:
        raise ValueError("packed decode cache cannot exceed the BF16 wire ledger")
    effective_bits = _positive_number(
        receipt["decode_cache_effective_bits_per_element"],
        "decode-cache effective bits",
    )
    elements = expected_wire / 2
    expected_cache = elements * effective_bits / 8
    if not math.isclose(
        float(receipt["decode_cache_bytes"]),
        expected_cache,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise ValueError("packed decode-cache ledger is inconsistent")
    if receipt["nominal_precision_sha256"] != content_hash(
        specialized["nominal_precision"]
    ):
        raise ValueError("handoff decode-cache precision binding differs")
    generation = receipt["link_generation"]
    if generation not in HANDOFF_LINK_BANDWIDTH_BYTES_PER_S:
        raise ValueError("handoff link generation is unsupported")
    declared_link_bw = _positive_number(
        receipt["link_bandwidth_bytes_per_s"], "handoff link bandwidth"
    )
    if declared_link_bw != HANDOFF_LINK_BANDWIDTH_BYTES_PER_S[generation]:
        raise ValueError("handoff link bandwidth differs from its generation")
    ports = _positive_integer(receipt["link_ports_used"], "handoff link ports")
    if ports != 1:
        raise ValueError("v1 handoff receipt requires one unstriped direct link")
    effective_link_bw = _positive_number(
        receipt["effective_link_bandwidth_bytes_per_s"],
        "effective handoff link bandwidth",
    )
    if effective_link_bw != declared_link_bw * ports:
        raise ValueError("effective handoff bandwidth differs from link ports")
    for name in ("transfer_ms", "admission_ms", "decode_ready_wait_ms"):
        receipt[name] = _nonnegative_number(receipt[name], f"handoff {name}")
    if receipt["transfer_ms"] <= 0 or receipt["admission_ms"] <= 0:
        raise ValueError("handoff transfer and admission must be positive")
    expected_transfer_ms = float(receipt["wire_bytes"]) / effective_link_bw * 1000
    if not math.isclose(
        receipt["transfer_ms"],
        expected_transfer_ms,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("bulk handoff time differs from the BF16 wire ledger")
    admission_bytes = _positive_number(
        receipt["admission_bytes"], "handoff admission bytes"
    )
    expected_admission_bytes = (
        float(receipt["wire_bytes"]) + float(receipt["decode_cache_bytes"])
    )
    if not math.isclose(
        admission_bytes,
        expected_admission_bytes,
        rel_tol=1e-12,
        abs_tol=1e-6,
    ):
        raise ValueError("full handoff admission traffic is inconsistent")
    admission_bandwidth = _positive_number(
        receipt["admission_bandwidth_bytes_per_s"],
        "handoff admission bandwidth",
    )
    policy = receipt["admission_bandwidth_policy"]
    _sha256(
        receipt["admission_bandwidth_source_sha256"],
        "handoff admission-bandwidth source",
    )
    if source_kind == "config_bound_analytic_handoff":
        if policy != "matched_candidate_aggregate_hbm_roofline":
            raise ValueError("analytic handoff admission policy differs")
        if not math.isclose(
            admission_bandwidth,
            float(
                specialized["resource_receipt"][
                    "aggregate_hbm_bandwidth_bytes_per_s"
                ]
            ),
            rel_tol=1e-12,
            abs_tol=1e-6,
        ) or receipt["admission_bandwidth_source_sha256"] != content_hash(
            specialized["resource_receipt"]
        ):
            raise ValueError(
                "analytic admission bandwidth differs from the matched HBM envelope"
            )
    elif policy != "measured_admission_artifact":
        raise ValueError("measured handoff admission policy differs")
    expected_admission_ms = admission_bytes / admission_bandwidth * 1000
    if not math.isclose(
        receipt["admission_ms"],
        expected_admission_ms,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("handoff admission time differs from its full traffic ledger")
    calibrated = _boolean(
        receipt["admission_calibrated"], "handoff admission calibration status"
    )
    calibration_id = receipt["admission_calibration_id"]
    if calibrated:
        if not isinstance(calibration_id, str) or not calibration_id.startswith(
            "admission-"
        ):
            raise ValueError("calibrated handoff admission lacks its identity")
        _sha256(calibration_id.removeprefix("admission-"), "admission calibration")
    elif calibration_id is not None:
        raise ValueError("uncalibrated handoff admission cannot carry calibration")
    if (
        not isinstance(receipt["admission_evidence_tier"], str)
        or not receipt["admission_evidence_tier"]
    ):
        raise ValueError("handoff admission evidence tier must be explicit")
    _boolean(receipt["publication_rankable"], "handoff publication status")
    if receipt["receipt_id"] != handoff_receipt_id(receipt):
        raise ValueError("handoff receipt identity is inconsistent")
    return receipt


def _validate_bf16_oracle(
    raw: Any,
    *,
    specialized: Mapping[str, Any],
    shared: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _object(raw, _BF16_FIELDS, "BF16 accuracy oracle")
    if receipt["schema_version"] != BF16_ORACLE_SCHEMA:
        raise ValueError("unsupported BF16 accuracy-oracle schema")
    model = _object(receipt["model"], _MODEL_FIELDS, "BF16 oracle model")
    if model != specialized["model"] or model != shared["model"]:
        raise ValueError("BF16 oracle and decode model identities differ")
    if not isinstance(receipt["profile_id"], str) or not receipt["profile_id"]:
        raise ValueError("BF16 oracle profile identity is missing")
    _sha256(receipt["source_receipt_sha256"], "BF16 source receipt")
    if not isinstance(receipt["accuracy_scope"], str) or not receipt["accuracy_scope"]:
        raise ValueError("BF16 oracle accuracy scope must be explicit")
    for name in (
        "evaluation_protocol_sha256",
        "dataset_sha256",
        "prompt_manifest_sha256",
        "seed_receipt_sha256",
        "mean_nll_receipt_sha256",
        "sample_set_sha256",
    ):
        _sha256(receipt[name], f"BF16 oracle {name}")
    _positive_integer(receipt["scored_tokens"], "BF16 scored tokens")
    receipt["mean_token_nll"] = _nonnegative_number(
        receipt["mean_token_nll"], "BF16 mean token NLL"
    )
    if receipt["latency_role"] != "accuracy_only_not_hardware_priced":
        raise ValueError("BF16 oracle cannot carry PLENA latency")
    if receipt["state"] != "succeeded":
        raise ValueError("BF16 accuracy oracle did not succeed")
    for point in (specialized, shared):
        numerical = point["numerical_receipt"]
        if (
            numerical["accuracy_scope"] != receipt["accuracy_scope"]
            or numerical["sample_set_sha256"] != receipt["sample_set_sha256"]
            or numerical["scored_tokens"] != receipt["scored_tokens"]
            or numerical["bf16_mean_token_nll"] != receipt["mean_token_nll"]
        ):
            raise ValueError("exact-MLEN numerical receipt and BF16 oracle differ")
    if receipt["receipt_id"] != bf16_oracle_receipt_id(receipt):
        raise ValueError("BF16 oracle receipt identity is inconsistent")
    return receipt


def _validate_selection_ancestry(
    raw: Any,
    *,
    specialized: Mapping[str, Any],
    shared: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = _object(raw, _ANCESTRY_FIELDS, "selection ancestry")
    if receipt["schema_version"] != ANCESTRY_SCHEMA:
        raise ValueError("unsupported selection-ancestry schema")
    if (
        not isinstance(receipt["selected_source_receipt_id"], str)
        or not receipt["selected_source_receipt_id"].startswith(
            "selected-decode-source-"
        )
    ):
        raise ValueError("selected-source receipt identity is invalid")
    if (
        not isinstance(receipt["selected_source_profile_id"], str)
        or not receipt["selected_source_profile_id"]
    ):
        raise ValueError("selected source profile identity is missing")
    if receipt["derived_specialized_profile_id"] != specialized[
        "numerical_receipt"
    ]["profile_id"]:
        raise ValueError("derived profile and specialized numerical identity differ")
    _sha256(
        receipt["numerical_derivation_receipt_sha256"],
        "specialized numerical derivation receipt",
    )
    selected = receipt["selected_hardware"]
    derived = receipt["derived_specialized_hardware"]
    if not isinstance(selected, Mapping) or not isinstance(derived, Mapping):
        raise TypeError("selection-ancestry hardware must be objects")
    selected = json.loads(_canonical_bytes(dict(selected)))
    derived = json.loads(_canonical_bytes(dict(derived)))
    if set(selected) != set(specialized["hardware"]) or set(derived) != set(
        specialized["hardware"]
    ):
        raise ValueError("selection-ancestry hardware schemas differ")
    if receipt["selected_candidate_id"] != "hw-" + content_hash(selected):
        raise ValueError("selected candidate identity differs from its hardware")
    if receipt["derived_specialized_candidate_id"] != "hw-" + content_hash(derived):
        raise ValueError("derived candidate identity differs from its hardware")
    _sha256(receipt["selected_replay_record_sha256"], "selected replay record")
    if receipt["derivation_rule"] != (
        "preserve_all_axes_except_blen_set_blen_to_65536_div_mlen"
    ):
        raise ValueError("unsupported matched-specialized derivation rule")
    if 65_536 % selected["MLEN"]:
        raise ValueError("selected MLEN cannot derive an integer matched BLEN")
    expected_blen = 65_536 // selected["MLEN"]
    expected = dict(selected)
    expected["BLEN"] = expected_blen
    if derived != expected or derived != specialized["hardware"]:
        raise ValueError("derived specialized hardware differs from the sealed rule")
    expected_unchanged = selected == derived
    if receipt["selected_was_already_multiplier_matched"] is not expected_unchanged:
        raise ValueError("selection ancestry unchanged flag differs")
    if derived == shared["hardware"]:
        raise ValueError("selected derivation makes the geometry ablation a no-op")
    if receipt["receipt_id"] != ancestry_receipt_id(receipt):
        raise ValueError("selection-ancestry receipt identity is inconsistent")
    receipt["selected_hardware"] = selected
    receipt["derived_specialized_hardware"] = derived
    return receipt


def _matched_hardware(specialized: Mapping[str, Any], shared: Mapping[str, Any]) -> None:
    allowed = {"MLEN", "BLEN", "VLEN"}
    if set(specialized) != set(shared):
        raise ValueError("candidate schemas differ between comparison arms")
    mismatched = sorted(
        key
        for key in specialized
        if key not in allowed and specialized[key] != shared[key]
    )
    if mismatched:
        raise ValueError(
            "non-geometry hardware assumptions differ: " + ",".join(mismatched)
        )
    for name, expected in SHARED_PLENA_GEOMETRY.items():
        if shared[name] != expected:
            raise ValueError(f"shared PLENA arm requires {name}={expected}")


def _matched_points(specialized: Mapping[str, Any], shared: Mapping[str, Any]) -> None:
    for name in (
        "model",
        "nominal_precision",
        "workload",
        "phase_contract",
        "clock_hz",
    ):
        if specialized[name] != shared[name]:
            raise ValueError(f"comparison arm {name} assumptions differ")
    for name in ("location", "semantic_contract_sha256", "idealizations"):
        if specialized["output_head"][name] != shared["output_head"][name]:
            raise ValueError(f"comparison arm output-head semantics {name} differ")
    for name in (
        "kind",
        "routing_source_kind",
        "routing_source_receipt_sha256",
        "routing_semantics_sha256",
        "placement_policy_sha256",
        "expert_parallel_mode",
        "resident_expert_count",
        "model_expert_count",
    ):
        if specialized["routing"][name] != shared["routing"][name]:
            raise ValueError(f"comparison arm routing policy {name} differs")
    _matched_hardware(specialized["hardware"], shared["hardware"])
    for name in (
        "resource_budget_sha256",
        "runtime_feasible",
        "timing_complete",
        "body_timing_complete",
        "aggregate_hbm_capacity_bytes",
        "aggregate_hbm_bandwidth_bytes_per_s",
        "aggregate_area_limit_mm2",
        "aggregate_hbm_capacity_limit_bytes",
        "aggregate_hbm_bandwidth_limit_bytes_per_s",
    ):
        if specialized["resource_receipt"][name] != shared["resource_receipt"][name]:
            raise ValueError(f"comparison arm resource assumption {name} differs")
    for name in ("timing_tier", "execution_mode", "metric_scope", "timing_valid"):
        if specialized["timing"][name] != shared["timing"][name]:
            raise ValueError(f"comparison arm timing contract {name} differs")
    if specialized["source"]["evaluator_id"] != shared["source"]["evaluator_id"]:
        raise ValueError("comparison arms use different evaluators")
    if (
        specialized["source"]["evaluator_provenance_sha256"]
        != shared["source"]["evaluator_provenance_sha256"]
    ):
        raise ValueError("comparison arms use different evaluator provenance")
    if (
        specialized["numerical_receipt"]["accuracy_scope"]
        != shared["numerical_receipt"]["accuracy_scope"]
    ):
        raise ValueError("comparison arms use different accuracy scopes")
    if (
        specialized["numerical_receipt"]["method_contract_sha256"]
        != shared["numerical_receipt"]["method_contract_sha256"]
    ):
        raise ValueError("comparison arms use different numerical methods")
    if specialized["hardware"]["MLEN"] != shared["hardware"]["MLEN"]:
        for label, left, right in (
            (
                "exact-MLEN numerical",
                specialized["numerical_receipt"]["receipt_id"],
                shared["numerical_receipt"]["receipt_id"],
            ),
            (
                "local-head geometry",
                specialized["output_head"]["geometry_receipt_sha256"],
                shared["output_head"]["geometry_receipt_sha256"],
            ),
            (
                "routing geometry timing",
                specialized["routing"]["geometry_timing_receipt_sha256"],
                shared["routing"]["geometry_timing_receipt_sha256"],
            ),
        ):
            if left == right:
                raise ValueError(f"comparison arms reuse one {label} receipt")


def build_comparison(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a matched pair and derive the only allowed latency metrics."""

    _reject_goodput(raw)
    source = _object(
        raw,
        {
            "schema_version",
            "arms",
            "selection_ancestry",
            "handoff",
            "bf16_accuracy_oracle",
        },
        "comparison input",
    )
    if source["schema_version"] != INPUT_SCHEMA:
        raise ValueError("unsupported matched-comparison input schema")
    arms = _object(
        source["arms"],
        {"decode_specialized", "shared_plena_geometry"},
        "comparison arms",
    )
    specialized = _validate_point(arms["decode_specialized"], "decode_specialized")
    shared = _validate_point(arms["shared_plena_geometry"], "shared_plena_geometry")
    _matched_points(specialized, shared)
    ancestry = _validate_selection_ancestry(
        source["selection_ancestry"],
        specialized=specialized,
        shared=shared,
    )
    handoff = _validate_handoff(
        source["handoff"], specialized=specialized
    )
    bf16 = _validate_bf16_oracle(
        source["bf16_accuracy_oracle"],
        specialized=specialized,
        shared=shared,
    )

    shared_pes = shared["resource_receipt"]["matrix_pe_equivalents_per_chip"]
    if shared_pes != SHARED_PLENA_PE_EQUIVALENTS_PER_CHIP:
        raise ValueError("shared PLENA PE-equivalent receipt is inconsistent")
    specialized_pes = specialized["resource_receipt"][
        "matrix_pe_equivalents_per_chip"
    ]
    lanes_equal = specialized_pes == shared_pes
    multipliers_equal = (
        specialized["resource_receipt"]["aggregate_multiplier_count"]
        == shared["resource_receipt"]["aggregate_multiplier_count"]
    )
    common_envelope_matched = (
        lanes_equal
        and multipliers_equal
        and specialized["resource_receipt"]["resource_budget_feasible"]
        and shared["resource_receipt"]["resource_budget_feasible"]
    )

    specialized_area = specialized["resource_receipt"]["system_area_mm2"]
    shared_area = shared["resource_receipt"]["system_area_mm2"]
    area_exact = math.isclose(
        specialized_area,
        shared_area,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
    area_delta = shared_area - specialized_area
    area_ratio = shared_area / specialized_area

    handoff_one_time_ms = (
        handoff["transfer_ms"]
        + handoff["admission_ms"]
        + handoff["decode_ready_wait_ms"]
    )
    decode_steps = specialized["workload"]["output_seq"] - 1
    amortized = handoff_one_time_ms / decode_steps
    specialized_tpot = specialized["timing"]["tpot_ms"]
    shared_tpot = shared["timing"]["tpot_ms"]
    handoff_amortized_service = specialized_tpot + amortized

    producer_publication_flags = all(
        point["resource_receipt"]["broader_publication_rankable"]
        and point["numerical_receipt"]["publication_rankable"]
        for point in (specialized, shared)
    ) and handoff["publication_rankable"]
    # v1 validates the normalized comparison and replays its arithmetic, but
    # cannot authenticate arbitrary producer-specific evidence merely because
    # a caller supplied plausible SHA strings and booleans.  It is therefore
    # always an analytic, non-publication receipt.  A future producer adapter
    # may define a distinct verified schema after replaying every source.
    publication_rankable = False
    evidence_class = "analytic_nonpublication"

    matched_contract = {
        "model": specialized["model"],
        "nominal_precision_sha256": content_hash(
            specialized["nominal_precision"]
        ),
        "workload": specialized["workload"],
        "phase_contract": specialized["phase_contract"],
        "clock_hz": specialized["clock_hz"],
        "candidate_non_geometry": {
            key: value
            for key, value in specialized["hardware"].items()
            if key not in {"MLEN", "BLEN", "VLEN"}
        },
        "resource_budget_sha256": specialized["resource_receipt"][
            "resource_budget_sha256"
        ],
        "output_head_semantics": {
            key: specialized["output_head"][key]
            for key in ("location", "semantic_contract_sha256", "idealizations")
        },
        "routing_policy": {
            key: specialized["routing"][key]
            for key in (
                "kind",
                "routing_source_kind",
                "routing_source_receipt_sha256",
                "routing_semantics_sha256",
                "placement_policy_sha256",
                "expert_parallel_mode",
                "resident_expert_count",
                "model_expert_count",
            )
        },
        "timing_method": {
            key: specialized["timing"][key]
            for key in ("timing_tier", "execution_mode", "metric_scope")
        },
        "evaluator_id": specialized["source"]["evaluator_id"],
        "evaluator_provenance_sha256": specialized["source"][
            "evaluator_provenance_sha256"
        ],
    }
    result = {
        "claim_scope": CLAIM_SCOPE,
        "comparison_status": (
            "multiplier_and_common_envelope_matched_area_disclosed"
            if common_envelope_matched
            else "multiplier_or_common_envelope_unmatched"
        ),
        "evidence_class": evidence_class,
        "publication_rankable": publication_rankable,
        "producer_publication_flags_all_true": producer_publication_flags,
        "publication_blocker": (
            "v1_normalized_receipt_does_not_authenticate_producer_artifacts"
        ),
        "paper_geometry": dict(SHARED_PLENA_GEOMETRY),
        "selection_ancestry": {
            "receipt_id": ancestry["receipt_id"],
            "selected_candidate_id": ancestry["selected_candidate_id"],
            "selected_source_profile_id": ancestry[
                "selected_source_profile_id"
            ],
            "derived_specialized_profile_id": ancestry[
                "derived_specialized_profile_id"
            ],
            "derived_specialized_candidate_id": ancestry[
                "derived_specialized_candidate_id"
            ],
            "selected_was_already_multiplier_matched": ancestry[
                "selected_was_already_multiplier_matched"
            ],
            "derived_arm_is_controlled_ablation_not_absolute_winner": True,
        },
        "matched_contract_sha256": content_hash(matched_contract),
        "resource_fairness": {
            "matrix_pe_equivalents_per_chip": {
                "decode_specialized": specialized_pes,
                "shared_plena_geometry": shared_pes,
                "matched": lanes_equal,
            },
            "aggregate_multiplier_count": {
                "decode_specialized": specialized["resource_receipt"][
                    "aggregate_multiplier_count"
                ],
                "shared_plena_geometry": shared["resource_receipt"][
                    "aggregate_multiplier_count"
                ],
                "matched": multipliers_equal,
            },
            "system_area_mm2": {
                "decode_specialized": specialized_area,
                "shared_plena_geometry": shared_area,
                "delta_shared_minus_specialized": area_delta,
                "ratio_shared_over_specialized": area_ratio,
                "exact_match": area_exact,
            },
            "aggregate_hbm_capacity_bytes": {
                "decode_specialized": specialized["resource_receipt"][
                    "aggregate_hbm_capacity_bytes"
                ],
                "shared_plena_geometry": shared["resource_receipt"][
                    "aggregate_hbm_capacity_bytes"
                ],
                "delta_shared_minus_specialized": (
                    shared["resource_receipt"]["aggregate_hbm_capacity_bytes"]
                    - specialized["resource_receipt"][
                        "aggregate_hbm_capacity_bytes"
                    ]
                ),
                "common_limit": specialized["resource_receipt"][
                    "aggregate_hbm_capacity_limit_bytes"
                ],
            },
            "aggregate_hbm_bandwidth_bytes_per_s": {
                "decode_specialized": specialized["resource_receipt"][
                    "aggregate_hbm_bandwidth_bytes_per_s"
                ],
                "shared_plena_geometry": shared["resource_receipt"][
                    "aggregate_hbm_bandwidth_bytes_per_s"
                ],
                "delta_shared_minus_specialized": (
                    shared["resource_receipt"][
                        "aggregate_hbm_bandwidth_bytes_per_s"
                    ]
                    - specialized["resource_receipt"][
                        "aggregate_hbm_bandwidth_bytes_per_s"
                    ]
                ),
                "common_limit": specialized["resource_receipt"][
                    "aggregate_hbm_bandwidth_limit_bytes_per_s"
                ],
            },
            "common_area_limit_mm2": specialized["resource_receipt"][
                "aggregate_area_limit_mm2"
            ],
            "iso_area_claimed": False,
            "comparison_ratio_allowed_under_common_envelope": (
                common_envelope_matched
            ),
        },
        "latency_metrics": {
            "decode_specialized_tpot_ms": specialized_tpot,
            "shared_plena_geometry_tpot_ms": shared_tpot,
            "one_time_handoff_ms": handoff_one_time_ms,
            "handoff_amortization_decode_steps": decode_steps,
            "amortized_handoff_ms_per_decode_token": amortized,
            # This is a request-level amortized decode-side service metric.
            # It is intentionally not called TPOT: the one-time handoff does
            # not change steady-state inter-token latency.
            "handoff_amortized_decode_side_service_ms": (
                handoff_amortized_service
            ),
            "raw_tpot_speedup_shared_over_specialized": (
                shared_tpot / specialized_tpot
                if common_envelope_matched
                else None
            ),
            "handoff_inclusive_speedup_shared_over_specialized": (
                shared_tpot / handoff_amortized_service
                if common_envelope_matched
                else None
            ),
        },
        "accuracy_contract": {
            "bf16_role": "accuracy_only_not_hardware_priced",
            "bf16_oracle_receipt_id": bf16["receipt_id"],
            "decode_specialized_profile_id": specialized["numerical_receipt"][
                "profile_id"
            ],
            "shared_plena_geometry_profile_id": shared["numerical_receipt"][
                "profile_id"
            ],
            "nominal_precision_equal": True,
            "exact_mlen_receipts_distinct": (
                specialized["numerical_receipt"]["receipt_id"]
                != shared["numerical_receipt"]["receipt_id"]
            ),
        },
        "handoff_contract": {
            "source_kind": handoff["source_kind"],
            "regime": handoff["regime"],
            "queue_wait_idealized_to_zero": handoff["source_kind"]
            == "config_bound_analytic_handoff",
        },
        "accuracy_metrics": {
            "scope": bf16["accuracy_scope"],
            "sample_set_sha256": bf16["sample_set_sha256"],
            "scored_tokens": bf16["scored_tokens"],
            "bf16_mean_token_nll": bf16["mean_token_nll"],
            "decode_specialized_mean_token_nll": specialized[
                "numerical_receipt"
            ]["candidate_mean_token_nll"],
            "shared_plena_geometry_mean_token_nll": shared[
                "numerical_receipt"
            ]["candidate_mean_token_nll"],
            "decode_specialized_relative_perplexity_vs_bf16": math.exp(
                specialized["numerical_receipt"]["candidate_mean_token_nll"]
                - bf16["mean_token_nll"]
            ),
            "shared_plena_geometry_relative_perplexity_vs_bf16": math.exp(
                shared["numerical_receipt"]["candidate_mean_token_nll"]
                - bf16["mean_token_nll"]
            ),
        },
        "metric_whitelist": [
            "decode_tpot",
            "one_time_handoff",
            "handoff_amortized_decode_side_service",
            "bf16_relative_nll_perplexity",
        ],
        "explicit_nonclaims": [
            "not_end_to_end_aggregated_vs_disaggregated_serving",
            "not_prefill_performance",
            "not_goodput",
            "not_accuracy_improvement_from_topology",
            "not_directly_normalized_to_published_plena_tps_or_ttft",
            "exact_mlen_numerical_receipt_is_mase_matrix_rounding_not_full_geometry_or_rtl_parity",
            "task_accuracy_requires_separate_authenticated_task_receipts",
            "not_iso_area_when_reported_system_areas_differ",
            *(
                ["analytic_handoff_assumes_zero_decode_queue_wait"]
                if handoff["source_kind"] == "config_bound_analytic_handoff"
                else []
            ),
        ],
    }
    canonical_input = json.loads(_canonical_bytes(source))
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "input": canonical_input,
        "result": result,
        "input_sha256": content_hash(canonical_input),
    }
    digest = content_hash(body)
    return {
        **body,
        "artifact_id": "matched-decode-geometry-" + digest,
        "content_sha256": digest,
    }


def load_comparison(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and fully replay an immutable comparison artifact."""

    artifact = json.loads(Path(path).read_bytes())
    if not isinstance(artifact, Mapping):
        raise TypeError("comparison artifact must be an object")
    required = {
        "schema_version",
        "input",
        "result",
        "input_sha256",
        "artifact_id",
        "content_sha256",
    }
    if set(artifact) != required or artifact.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("comparison artifact fields differ from the schema")
    rebuilt = build_comparison(artifact["input"])
    if artifact != rebuilt:
        raise ValueError("comparison artifact replay or content hash differs")
    return rebuilt


def write_comparison(
    path: str | os.PathLike[str],
    artifact: Mapping[str, Any],
) -> Path:
    """Atomically create an immutable, replay-validated artifact."""

    rebuilt = build_comparison(artifact["input"])
    if dict(artifact) != rebuilt:
        raise ValueError("refusing to write a non-replayable comparison artifact")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(rebuilt, newline=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(
                f"refusing to replace a different comparison artifact: {destination}"
            )
        return destination
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, 0o644)
        os.link(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


__all__ = [
    "ANCESTRY_SCHEMA",
    "ARTIFACT_SCHEMA",
    "BF16_ORACLE_SCHEMA",
    "CLAIM_SCOPE",
    "HANDOFF_SCHEMA",
    "INPUT_SCHEMA",
    "NUMERICAL_RECEIPT_SCHEMA",
    "POINT_SCHEMA",
    "SHARED_PLENA_GEOMETRY",
    "SHARED_PLENA_PE_EQUIVALENTS_PER_CHIP",
    "ancestry_receipt_id",
    "bf16_oracle_receipt_id",
    "build_comparison",
    "content_hash",
    "handoff_receipt_id",
    "load_comparison",
    "numerical_receipt_id",
    "write_comparison",
]
