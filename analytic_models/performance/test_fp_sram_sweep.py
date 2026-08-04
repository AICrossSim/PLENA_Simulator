"""The scalar FP SRAM depth trade on the KV read plane."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomlkit

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from disagg_serve.packed_kv import (  # noqa: E402
    DENSE_COMPILER,
    DENSE_SELECTOR,
    traffic_from_precision,
)
from area.calibration_provenance import build_area_calibration_audit  # noqa: E402
from fp_sram_sweep import (  # noqa: E402
    CONSTANT_SLOTS,
    RTL_FP_SRAM_DEPTH,
    SCALARS_PER_ROW,
    depth_for_reuse,
    evaluate,
    state_slots,
)
from decode_results_table import (  # noqa: E402
    Row,
    build_point,
    build_results_document,
    build_parser as build_results_parser,
    capacity_match,
    modelled_row,
    results_source_provenance,
    roofline_rows,
    validate_calibration_inputs,
)
from decode_timing import DRAIN_OVERLAPPED, RTL_SERIALIZED  # noqa: E402
from disagg_decode import (  # noqa: E402
    DEVICES,
    evaluate as evaluate_decode,
    load_precision_points,
)
from disagg_serve.decode_power import memory_power_watts  # noqa: E402
from disagg_serve.hbm_technology import hbm_technology  # noqa: E402

# Qwen3-32B on the headline array: MLEN 1024, HLEN 128, BLEN 4, 8 KV heads.
GEOMETRY = dict(broadcast_heads=8, kv_heads=8, blen=4)


def test_the_read_amplification_is_the_kv_head_count() -> None:
    """`dense_compiler` re-reads the packed row once per KV head; that is the target."""
    layout = traffic_from_precision(
        kv_heads=8, head_dim=128, mlen=1024, element_bits=4, effective_bits=5.0
    )
    assert layout.read_element_bytes(DENSE_COMPILER) == 8 * layout.read_element_bytes(
        DENSE_SELECTOR
    )
    assert layout.read_element_bytes(DENSE_SELECTOR) == layout.storage_element_bytes(
        DENSE_SELECTOR
    )


def test_reuse_costs_scalar_slots_in_proportion() -> None:
    """Doubling the live groups doubles the softmax state above the constants."""
    previous = None
    for groups in (1, 2, 4, 8):
        needed = depth_for_reuse(GEOMETRY["broadcast_heads"], GEOMETRY["blen"], groups)
        assert needed == CONSTANT_SLOTS + SCALARS_PER_ROW * 4 * 8 * groups
        if previous is not None:
            assert needed - CONSTANT_SLOTS == 2 * (previous - CONSTANT_SLOTS)
        previous = needed


def test_the_rtl_depth_already_affords_four_live_groups() -> None:
    """Half the amplification is available with no RTL change at all."""
    point = evaluate(RTL_FP_SRAM_DEPTH, **GEOMETRY)
    assert point.groups_live == 4
    assert point.kv_read_factor == 2
    assert point.live_slots == 390 <= RTL_FP_SRAM_DEPTH
    assert point.query_tile == GEOMETRY["blen"]


def test_one_read_per_token_needs_774_slots() -> None:
    """The full reduction is 1.51x the RTL depth, not a different order of magnitude."""
    needed = depth_for_reuse(GEOMETRY["broadcast_heads"], GEOMETRY["blen"], 8)
    assert needed == 774
    assert needed > RTL_FP_SRAM_DEPTH
    assert (needed - RTL_FP_SRAM_DEPTH) * 12 / 8 < 512  # under half a KiB more
    assert evaluate(needed, **GEOMETRY).kv_read_factor == 1


def test_the_query_tile_is_one_block_not_the_whole_mlen_row() -> None:
    """Row tiles split the batch, so one full M_BTMM query block must fit."""
    assert state_slots(GEOMETRY["blen"], 8, 1) == 102
    assert evaluate(RTL_FP_SRAM_DEPTH, **GEOMETRY).query_tile == GEOMETRY["blen"]


def test_live_state_never_exceeds_the_depth() -> None:
    for depth in (102, 198, 390, 512, 774, 1024, 2048):
        point = evaluate(depth, **GEOMETRY)
        if point is not None:
            assert point.live_slots <= depth


def test_a_depth_too_small_for_one_query_block_is_rejected() -> None:
    """One block of four rows across eight head lanes needs 96 slots plus constants."""
    assert evaluate(CONSTANT_SLOTS + 95, **GEOMETRY) is None
    assert evaluate(CONSTANT_SLOTS + 96, **GEOMETRY) is not None


def test_results_default_to_the_scalar_feasible_target_geometry() -> None:
    args = build_results_parser().parse_args([])
    assert args.mlen == 1024
    assert args.blen == 4
    assert args.block == 8
    assert args.models == "qwen3-32b"


def test_results_use_the_batched_baseline_for_co_design_ordering() -> None:
    args = build_results_parser().parse_args([])
    reference_capacity = (
        DEVICES["a100"]["hbm_gb"]
        * DEVICES["a100"]["count"]
        * 1e9
    )
    chips, batch = capacity_match(args, args.models, reference_capacity)
    unsupported = modelled_row(
        args,
        args.models,
        chips,
        batch,
        False,
        DENSE_COMPILER,
        RTL_SERIALIZED,
    )
    baseline = modelled_row(
        args,
        args.models,
        chips,
        batch,
        True,
        DENSE_COMPILER,
        RTL_SERIALIZED,
    )
    overlap = modelled_row(
        args,
        args.models,
        chips,
        batch,
        True,
        DENSE_COMPILER,
        DRAIN_OVERLAPPED,
    )
    one_read = modelled_row(
        args,
        args.models,
        chips,
        batch,
        True,
        DENSE_SELECTOR,
        RTL_SERIALIZED,
    )

    assert unsupported.evidence_tier == "analytic unsupported"
    assert "(unsupported)" in unsupported.configuration
    assert baseline.evidence_tier == "analytic baseline"
    assert overlap.evidence_tier == one_read.evidence_tier == "analytic co-design"
    assert baseline.precision == "MXINT4/MXINT4/BF16/MXINT4"
    assert "selector timing + KV read 1x" in one_read.configuration
    assert baseline.power_w == pytest.approx(42.2077605396844)
    assert baseline.tokens_per_joule == pytest.approx(1.969461469033609)
    for row in (overlap, one_read):
        assert math.isnan(row.area_mm2)
        assert math.isnan(row.power_w)
        assert math.isnan(row.tokens_per_joule)
    assert not hasattr(baseline, "ttft_ms")
    for row in (unsupported, baseline, overlap, one_read):
        assert row.tps == pytest.approx(row.batch * 1e3 / row.tpot_ms)
    assert baseline.tps > unsupported.tps
    for row in (overlap, one_read):
        assert row.tps > baseline.tps
        assert row.tpot_ms < baseline.tpot_ms
        assert row.first_decode_ms < baseline.first_decode_ms

    plena_peak = next(
        row
        for row in roofline_rows(args, args.models, chips)
        if row.device.startswith("PLENA")
    )
    assert (args.mlen, args.blen, chips, batch) == (1024, 4, 7, 229)
    assert plena_peak.area_mm2 == pytest.approx(0.237)
    assert plena_peak.tps == pytest.approx(692.9461782832954)


def test_multi_chip_memory_power_counts_aggregate_traffic_once() -> None:
    args = build_results_parser().parse_args([])
    reference_capacity = (
        DEVICES["a100"]["hbm_gb"] * DEVICES["a100"]["count"] * 1e9
    )
    chips, batch = capacity_match(args, args.models, reference_capacity)
    args.chips, args.batch = chips, batch
    model_path, dims, hardware, memory, precision = build_point(args, args.models)
    result = evaluate_decode(
        model_path,
        dims,
        hardware,
        args.isa_lib,
        memory,
        precision,
        batch,
        args.input_seq,
        args.output_seq,
        stride=max(1, args.output_seq // 256),
        n_chips=chips,
        kv_layout=DENSE_COMPILER,
        batch_packed_attention=True,
        hbm_gen=args.hbm_gen,
        hbm_channels=args.hbm_channels,
        timing_mode=RTL_SERIALIZED,
    )
    expected_memory_watts = memory_power_watts(
        hbm_technology(args.hbm_gen),
        capacity_bytes=result["hbm_capacity"],
        read_bytes_per_second=result["read_bytes_per_second"],
        write_bytes_per_second=result["write_bytes_per_second"],
    )
    assert result["power"].memory_watts == pytest.approx(expected_memory_watts)


def test_area_audit_is_bound_to_the_retained_aggregate_evidence() -> None:
    repository = Path(__file__).resolve().parents[2]
    audit = build_area_calibration_audit(repository)
    validation = audit["aggregate_validation"]
    assert validation["passed"] == validation["total"] == 5
    assert validation["gates"]["anchor"]["observed_mm2"] == pytest.approx(0.237)
    assert audit["evidence_grade"] == "aggregate_area_tables_without_raw_dc_reports"
    assert audit["publication_receipt_complete"] is False


def test_results_json_is_strict_complete_and_content_addressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repository)
    args = build_results_parser().parse_args([])
    requested = dict(vars(args))
    requested["json"] = Path("/tmp/nonsemantic-results-destination.json")
    geometries = {
        "qwen3-32b": {
            "hidden_size": 5120,
            "head_dim": 128,
            "query_heads": 64,
            "kv_heads": 8,
        }
    }
    row = Row(
        model="qwen3-32b",
        device="PLENA x7",
        configuration="batched-q + KV read per-head",
        precision="MXINT4/MXINT4/BF16/MXINT4",
        batch=229,
        first_decode_ms=1.0,
        tps=2.0,
        tpot_ms=3.0,
        power_w=float("nan"),
        tokens_per_joule=float("nan"),
        area_mm2=float("nan"),
        evidence_tier="analytic co-design",
    )
    calibration = SimpleNamespace(
        calibration_id="calibration-id",
        label="emulator-calibrated",
        provenance_hashes=(
            ("isa_lib", hashlib.sha256(Path(args.isa_lib).read_bytes()).hexdigest()),
            ("settings", hashlib.sha256(Path(args.config).read_bytes()).hexdigest()),
        ),
        execution_contract=SimpleNamespace(
            to_dict=lambda: {
                "timing_mode": RTL_SERIALIZED,
                "drain_overlapped": False,
                "fp_sram_depth": 512,
            }
        ),
    )
    document = build_results_document(
        args=args,
        requested_arguments=requested,
        rows=[row],
        models=["qwen3-32b"],
        geometries=geometries,
        calibration=calibration,
        calibration_hash="a" * 64,
    )

    # Python's permissive JSON NaN spelling is not RFC 8259 JSON.
    encoded = json.dumps(document, allow_nan=False, sort_keys=True)
    assert "NaN" not in encoded
    assert document["rows"][0]["power_w"] is None
    assert document["rows"][0]["tokens_per_joule"] is None
    assert document["rows"][0]["area_mm2"] is None

    evaluation = document["evaluation"]
    assert "json" not in evaluation["arguments"]
    assert evaluation["arguments"]["hlen"] == 0
    assert evaluation["arguments"]["input_seq"] == 256
    assert evaluation["arguments"]["output_seq"] == 16384
    assert evaluation["arguments"]["roofline_kv_layout"] == DENSE_SELECTOR
    assert evaluation["resolved_geometry"]["qwen3-32b"]["hlen"] == 128
    assert evaluation["resolved_geometry"]["qwen3-32b"]["mlen"] == 1024
    assert evaluation["resolved_geometry"]["qwen3-32b"]["blen"] == 4
    assert evaluation["resolved_system"]["qwen3-32b"]["plena_chips"] == 7
    assert evaluation["resolved_system"]["qwen3-32b"]["instruction_batch"] == 229
    assert {point["timing_mode"] for point in evaluation["instruction_points"]} == {
        RTL_SERIALIZED,
        DRAIN_OVERLAPPED,
    }
    assert {point["kv_layout"] for point in evaluation["instruction_points"]} == {
        DENSE_COMPILER,
        DENSE_SELECTOR,
    }

    sources = document["provenance"]["sources"]
    for name in (
        "decode_results_table",
        "disagg_decode",
        "perf_model",
        "packed_q1_timing",
        "decode_timing",
        "packed_kv",
        "physical_ledger",
        "decode_power",
        "hbm_technology",
        "memory_model",
        "area_calibration",
    ):
        assert len(sources[name]["sha256"]) == 64
        assert sources[name]["repository_path"]
    assert document["device_references"]["a100"]["source_url"].startswith(
        "https://www.nvidia.com/"
    )
    assert document["device_references"]["h100"]["source_url"].startswith(
        "https://www.nvidia.com/"
    )
    assert document["device_references"]["a100"]["hbm_tbs"] == pytest.approx(2.039)
    assert document["device_references"]["a100"]["peak_tflops"] == pytest.approx(312.0)
    assert document["device_references"]["h100"]["hbm_tbs"] == pytest.approx(3.35)
    assert document["device_references"]["h100"]["peak_tflops"] == pytest.approx(989.5)
    power_assumptions = document["model_assumptions"]["plena_power"]
    assert power_assumptions["evidence_tier"] == (
        "analytic sensitivity anchored to literature-reported model output"
    )
    assert power_assumptions["reference_configuration"]["source_url"] == (
        "https://arxiv.org/pdf/2604.16007"
    )
    assert power_assumptions["reference_configuration"]["reference_total_watts"] == (
        pytest.approx(300.09)
    )
    assert power_assumptions["hbm_energy"]["energy_source_url"] == (
        "https://arxiv.org/pdf/2604.16007"
    )
    calibration_provenance = document["provenance"]["emulator_calibration"]
    assert calibration_provenance["execution_contract"]["timing_mode"] == (
        RTL_SERIALIZED
    )
    assert calibration_provenance["measured_input_hashes"]["settings"] == (
        hashlib.sha256(Path(args.config).read_bytes()).hexdigest()
    )

    content_hash = document.pop("content_hash")
    assert content_hash == _canonical_hash(document)

    relative_args = build_results_parser().parse_args(
        [
            "--model-lib", "compiler/doc/Model_Lib",
            "--config", "plena_settings.toml",
            "--isa-lib", "analytic_models/performance/customISA_lib.json",
            "--emulator-calibration",
            "analytic_models/performance/calibration/decode_kv1024.json",
            "--json", "relative-output.json",
        ]
    )
    relative_document = build_results_document(
        args=relative_args,
        requested_arguments=dict(vars(relative_args)),
        rows=[row],
        models=["qwen3-32b"],
        geometries=geometries,
        calibration=calibration,
        calibration_hash="a" * 64,
    )
    relative_hash = relative_document.pop("content_hash")
    assert relative_hash == content_hash
    assert relative_document == document


def test_results_source_provenance_changes_with_a_source_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = results_source_provenance({"probe": source})
    source.write_text("value = 2\n", encoding="utf-8")
    second = results_source_provenance({"probe": source})
    assert first["probe"]["sha256"] != second["probe"]["sha256"]


def test_results_reject_calibration_from_different_settings_or_isa() -> None:
    args = build_results_parser().parse_args([])
    current = {
        "settings": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "isa_lib": hashlib.sha256(Path(args.isa_lib).read_bytes()).hexdigest(),
    }
    validate_calibration_inputs(
        SimpleNamespace(provenance_hashes=tuple(sorted(current.items()))), args
    )
    for role in ("settings", "isa_lib"):
        mutated = dict(current)
        mutated[role] = "0" * 64
        with pytest.raises(ValueError, match=role):
            validate_calibration_inputs(
                SimpleNamespace(provenance_hashes=tuple(sorted(mutated.items()))),
                args,
            )


def test_headline_broadcast_matches_the_head_lanes() -> None:
    settings = Path(__file__).resolve().parents[2] / "plena_settings.toml"
    with settings.open() as source:
        config = tomlkit.load(source)["ANALYTIC"]["CONFIG"]
    mlen = int(config["MLEN"]["value"])
    hlen = int(config["HLEN"]["value"])
    broadcast = int(config["BROADCAST_AMOUNT"]["value"])
    assert mlen % hlen == 0
    assert broadcast == mlen // hlen


def _canonical_hash(value: dict) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_precision_source(root: Path, *, legacy_schema: bool = False) -> Path:
    csv_path = root / "accuracy_cost.csv"
    canonical_csv = (
        "profile_id,kind,weight_format,activation_format,key_format,"
        "value_format,vector_format,state,mean_nll\n"
        "dqp-example,quantized,MXINT4,E4M3,MXINT2,MXINT2,FP_E3M2,"
        "succeeded,9.5\n"
        "dqp-control,vector_bf16_control,MXINT4,E4M3,MXINT2,MXINT2,"
        "BF16,succeeded,8.0\n"
        "dqp-reference,bf16_reference,BF16,BF16,BF16,BF16,BF16,"
        "succeeded,7.0\n"
    )
    legacy_csv = (
        "tag,cont_ppl,attn_w_bits,ffn_w_bits,kv_bits,act_bits,block\n"
        "mxint4,9.5,5,5,5,5,8\n"
    )
    csv_path.write_text(
        legacy_csv if legacy_schema else canonical_csv,
        encoding="utf-8",
    )
    model = {
        "name": "Qwen/Qwen3-32B",
        "revision": "1" * 40,
        "tokenizer_revision": "2" * 40,
        "dtype": "bfloat16",
    }
    datasets = {
        "evaluation": {
            "name": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "revision": "3" * 40,
            "split": "validation",
        }
    }
    workspace_body = {
        "schema_version": "decode-sweep-provenance",
        "created_at_utc": "2026-08-01T00:00:00Z",
        "manifest_hash": "4" * 64,
        "run_plan_hash": "5" * 64,
        "quantizer_provenance_hash": "6" * 64,
        "model": model,
        "datasets": datasets,
    }
    workspace_path = root / "workspace_provenance.json"
    workspace_path.write_text(
        json.dumps(
            workspace_body | {"content_hash": _canonical_hash(workspace_body)}
        ),
        encoding="utf-8",
    )
    results_body = {
        "schema_version": "decode-sweep-results-provenance",
        "created_at_utc": "2026-08-01T00:01:00Z",
        "model": model,
        "datasets": datasets,
        "manifest_hash": workspace_body["manifest_hash"],
        "run_plan_hash": workspace_body["run_plan_hash"],
        "quantizer_provenance_hash": workspace_body[
            "quantizer_provenance_hash"
        ],
        "workspace_provenance": {
            "path": workspace_path.name,
            "content_hash": _canonical_hash(workspace_body),
        },
        "tables": [
            {
                "filename": csv_path.name,
                "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                "size_bytes": csv_path.stat().st_size,
            }
        ],
    }
    (root / "sweep_results_provenance.json").write_text(
        json.dumps(results_body | {"content_hash": _canonical_hash(results_body)}),
        encoding="utf-8",
    )
    return csv_path


def test_precision_source_requires_matching_checksum_bound_provenance(
    tmp_path: Path,
) -> None:
    source = _write_precision_source(tmp_path)
    points = load_precision_points(source, expected_model="qwen3-32b")
    assert len(points) == 1
    assert points[0]["tag"] == "dqp-example"
    assert math.log(points[0]["ppl"]) == pytest.approx(9.5)
    assert (points[0]["attn_elem"], points[0]["act_elem"], points[0]["kv_elem"]) == (
        4,
        8,
        2,
    )
    assert (points[0]["attn_bits"], points[0]["act_bits"], points[0]["kv_bits"]) == (
        5,
        9,
        3,
    )

    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_precision_points(source, expected_model="qwen3-32b")


def test_precision_source_rejects_a_different_model(tmp_path: Path) -> None:
    source = _write_precision_source(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        load_precision_points(source, expected_model="llama3-8b")


def test_precision_source_rejects_the_retired_csv_schema(tmp_path: Path) -> None:
    source = _write_precision_source(tmp_path, legacy_schema=True)
    with pytest.raises(ValueError, match="canonical numerical-results schema"):
        load_precision_points(source, expected_model="qwen3-32b")
