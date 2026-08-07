"""Tests for the full-model artifact-set producer and the native compile path.

The two full-scale compile tests are gated behind
``PLENA_FULL_MODEL_COMPILE_TESTS=1`` because each drives a real 32-layer
native lowering; they exist precisely because nothing else in either suite
compiles a full-depth model.
"""

from __future__ import annotations

import copy
import functools
import os
import time

import pytest

from compiler_trace_timing import (
    FULL_MODEL_DECODE_SCOPE,
    FullModelDecodeArtifactSet,
    FullModelDecodeLazyArtifactGenerator,
    HBMOperatingPoint,
    canonical_sha256,
    native_decode_compiler_source_sha256,
)
from full_model_artifact_build import build_full_model_decode_artifact_set

FULL_COMPILE_ENV = "PLENA_FULL_MODEL_COMPILE_TESTS"

requires_full_compile = pytest.mark.skipif(
    os.environ.get(FULL_COMPILE_ENV) != "1",
    reason=f"full 32-layer native compiles run only with {FULL_COMPILE_ENV}=1",
)


def _tiny_point_descriptor(
    *,
    batch: int = 2,
    settings_tag: str = "tiny",
) -> dict[str, object]:
    return {
        "schema_version": "plena-compiler-trace-point-v1",
        "artifact_scope": FULL_MODEL_DECODE_SCOPE,
        "model": {
            "model_json_sha256": canonical_sha256({"model": "tiny"}),
            "dimensions": {
                "hidden": 16,
                "inter": 32,
                "layers": 1,
                "heads": 2,
                "kv_heads": 2,
                "head_dim": 8,
                "vocab": 32,
            },
            "layer_scope": "all_decoder_layers",
            "output_head_location": "prefill_chip",
        },
        "precision": {
            "specification": {"attn_elem": 4, "ffn_elem": 4, "kv_elem": 4},
            "weight_format": "mxint",
            "kv_format": "mxint",
            "block_size": 8,
            "mac_bits": 4,
        },
        "hardware": {
            "array_geometry": {
                "mlen": 16,
                "blen": 2,
                "vlen": 16,
                "hlen": 8,
            },
            "hbm_timing_geometry": HBMOperatingPoint("HBM2", 8, 2.0).to_dict(),
            "configuration": {},
            "memory_configuration": {},
            "overrides": {"TP": 1, "KVP": 1},
            "topology": {
                "tp": 1,
                "kvp": 1,
                "chip_count": 1,
                "explicit_topology": True,
                "legacy_ideal_parallelism": False,
                "link_ports": 0,
                "sram_policy": "streaming",
                "link_generation": "p2p_100GBs",
                "architecture_knobs_explicit": True,
                "kv_head_reuse": True,
                "drain_overlapped": False,
            },
        },
        "serving": {
            "batch": batch,
            "input_tokens": 17,
            "generation_tokens": 1,
            "sample_stride": 1,
            "kv_layout": "dense_selector",
            "runtime_hbm_reserve_bytes": 0,
        },
        "compiler": {
            "settings_sha256": canonical_sha256({"settings": settings_tag}),
            "latency_library_sha256": canonical_sha256({"latency": "tiny"}),
            "timing_mode": "rtl_serialized",
            "frequency_hz": 1.0e9,
        },
    }


def _llama8b_point_descriptor() -> dict[str, object]:
    """Llama-8B dimensions at the cheapest legal full-depth geometry."""

    descriptor = _tiny_point_descriptor(batch=1)
    descriptor["model"] = {
        "model_json_sha256": canonical_sha256({"model": "llama8b"}),
        "dimensions": {
            "hidden": 4096,
            "inter": 14336,
            "layers": 32,
            "heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "vocab": 128256,
        },
        "layer_scope": "all_decoder_layers",
        "output_head_location": "external_bf16_service",
    }
    descriptor["hardware"] = {
        "array_geometry": {
            "mlen": 1024,
            "blen": 8,
            "vlen": 1024,
            "hlen": 128,
        },
        "hbm_timing_geometry": HBMOperatingPoint("HBM2", 8, 2.0).to_dict(),
        "configuration": {},
        "memory_configuration": {},
        "overrides": {"TP": 8, "KVP": 1},
        "topology": {
            "tp": 8,
            "kvp": 1,
            "chip_count": 8,
            "explicit_topology": True,
            "legacy_ideal_parallelism": False,
            "link_ports": 1,
            "sram_policy": "streaming",
            "link_generation": "nvlink4",
            "architecture_knobs_explicit": True,
            "kv_head_reuse": False,
            "drain_overlapped": False,
        },
    }
    descriptor["serving"] = {
        "batch": 1,
        "input_tokens": 512,
        "generation_tokens": 1,
        "sample_stride": 1,
        "kv_layout": "dense_selector",
        "runtime_hbm_reserve_bytes": 0,
    }
    return descriptor


def test_dry_run_returns_receipt_and_writes_nothing(tmp_path) -> None:
    destination = tmp_path / "artifact_set.json"
    receipt = build_full_model_decode_artifact_set(
        ((point, range(17, 20)) for point in (_tiny_point_descriptor(),)),
        destination,
        dry_run=True,
    )
    assert not destination.exists()
    assert receipt["point_count"] == 1
    assert receipt["family_count"] == 1
    assert receipt["record_count"] == 0
    assert receipt["records_materialized"] == "lazy_at_consume"
    assert receipt["native_compile_calls"] == 1
    assert receipt["unique_lowering_keys"] == 1
    assert receipt["context_start"] == 17
    assert receipt["context_stop"] == 20
    assert receipt["compiler_source_sha256"] == native_decode_compiler_source_sha256()


def test_real_build_round_trips_through_the_loader(tmp_path) -> None:
    destination = tmp_path / "artifact_set.json"
    points = (_tiny_point_descriptor(batch=2), _tiny_point_descriptor(batch=4))
    receipt = build_full_model_decode_artifact_set(
        ((point, range(17, 20)) for point in points),
        destination,
        dry_run=False,
    )
    assert destination.is_file()
    loaded = FullModelDecodeArtifactSet.load(destination)
    assert loaded.artifact_set_id == receipt["artifact_set_id"]
    assert loaded.compiler_source_sha256 == receipt["compiler_source_sha256"]
    assert len(loaded.families) == 1
    assert loaded.records == ()
    # distinct batches are distinct native compiles and distinct lowerings
    assert receipt["native_compile_calls"] == 2
    assert receipt["unique_lowering_keys"] == 2
    assert receipt["point_count"] == 2


def test_distinct_settings_seal_distinct_families(tmp_path) -> None:
    points = (
        _tiny_point_descriptor(settings_tag="tiny"),
        _tiny_point_descriptor(settings_tag="other"),
    )
    receipt = build_full_model_decode_artifact_set(
        ((point, range(17, 20)) for point in points),
        tmp_path / "artifact_set.json",
        dry_run=True,
    )
    assert receipt["family_count"] == 2


def test_blocked_point_fails_the_whole_build(tmp_path) -> None:
    blocked = _tiny_point_descriptor()
    blocked["model"]["dimensions"]["num_experts"] = 2  # type: ignore[index]
    with pytest.raises(ValueError, match="mixture_of_experts_not_lowered"):
        build_full_model_decode_artifact_set(
            ((point, range(17, 20)) for point in (blocked,)),
            tmp_path / "artifact_set.json",
            dry_run=True,
        )


def test_context_axis_validation(tmp_path) -> None:
    point = _tiny_point_descriptor()
    destination = tmp_path / "artifact_set.json"
    with pytest.raises(ValueError, match="non-empty"):
        build_full_model_decode_artifact_set(
            ((point, range(20, 17)),), destination, dry_run=True
        )
    with pytest.raises(ValueError, match="positive"):
        build_full_model_decode_artifact_set(
            ((point, (0, 1)),), destination, dry_run=True
        )
    with pytest.raises(ValueError, match="one exact context axis"):
        build_full_model_decode_artifact_set(
            ((point, range(17, 20)), (point, range(17, 21))),
            destination,
            dry_run=True,
        )
    with pytest.raises(ValueError, match="no exact points"):
        build_full_model_decode_artifact_set((), destination, dry_run=True)


def _patched_dynamic_instruction_limit(monkeypatch, limit: int) -> None:
    import compiler.aten.plena.compiler as plena_compiler

    original = plena_compiler.build_request_memory_trace
    if isinstance(original, functools.partial):
        original = original.func
    monkeypatch.setattr(
        plena_compiler,
        "build_request_memory_trace",
        functools.partial(original, max_dynamic_instructions=limit),
    )


def test_full_model_decode_exceeds_default_dynamic_instruction_guard(tmp_path) -> None:
    """A full-depth compile trips the runaway-trace resource guard by default."""

    descriptor = _llama8b_point_descriptor()
    receipt = build_full_model_decode_artifact_set(
        ((descriptor, range(512, 515)),),
        tmp_path / "artifact_set.json",
        dry_run=False,
    )
    loaded = FullModelDecodeArtifactSet.load(tmp_path / "artifact_set.json")
    generator = FullModelDecodeLazyArtifactGenerator(loaded.families[0])
    assert receipt["family_count"] == 1
    with pytest.raises(ValueError, match="dynamic instruction limit"):
        generator.instantiate(descriptor, context_tokens=512)


@requires_full_compile
def test_full_model_decode_compiles_within_wall_clock(tmp_path, monkeypatch) -> None:
    """One full 32-layer native compile completes inside the wall-clock bound."""

    _patched_dynamic_instruction_limit(monkeypatch, 2_000_000_000)
    descriptor = _llama8b_point_descriptor()
    build_full_model_decode_artifact_set(
        ((descriptor, range(512, 515)),),
        tmp_path / "artifact_set.json",
        dry_run=False,
    )
    loaded = FullModelDecodeArtifactSet.load(tmp_path / "artifact_set.json")
    generator = FullModelDecodeLazyArtifactGenerator(loaded.families[0])
    started = time.monotonic()
    record = generator.instantiate(descriptor, context_tokens=512)
    elapsed = time.monotonic() - started
    assert int(record.compiler_receipt["batch_size"]) == 1
    assert record.context_tokens == 512
    assert elapsed < 300.0, f"32-layer native compile took {elapsed:.1f}s"
