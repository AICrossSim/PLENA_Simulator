"""Production contracts for compiler-trace decode timing."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from analytic_models.disagg_serve.memory import (
    REQUEST_FEATURES,
    DMARequestDescriptor,
    RequestLatencyModel,
    RequestModelFit,
)
from analytic_models.performance.compiler_trace_timing import (
    COMPILER_TRACE,
    FULL_MODEL_DECODE_SCOPE,
    LEGACY_AGGREGATE_BANDWIDTH,
    REFERENCE_DECODE_SCOPE,
    REQUEST_STREAM_COMPOSITION_SCHEMA,
    ArrayGeometry,
    BoundCompilerTrace,
    CompilerTraceTimingProvider,
    CompilerTraceTimingRequest,
    FullModelDecodeArtifactBinder,
    FullModelDecodeArtifactBuildPlan,
    FullModelDecodeArtifactFamily,
    FullModelDecodeLazyArtifactGenerator,
    FullModelDecodeArtifactSet,
    FullModelDecodeArtifactSetBuilder,
    FullModelDecodeBatchResolution,
    HBMOperatingPoint,
    ReferenceDecodeArtifactBuilder,
    ReferenceDecodeLowering,
    RequestDescriptorRun,
    RequestMemorySidecar,
    RequestModelStageMemoryPricer,
    ResidencyAdjustedStageMemoryPricer,
    TraceRequestBinding,
    canonical_sha256,
    create_full_model_decode_artifact_runtime,
    full_model_decode_batch_resolution,
    full_model_decode_context_resolution,
    full_model_decode_lowering_key,
    full_model_decode_family_key,
    full_model_decode_generator_blockers,
    full_model_decode_native_template_key,
    request_memory_sidecar_from_compiler,
    resolve_decode_step_timing,
    trace_entry_fingerprint,
)
from compiler.aten.execution_trace import (
    HBM_READ,
    CompilationArtifact,
    ExecutionTrace,
    build_execution_trace,
)
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.packed_kv import PackedKVLayout
from compiler.aten.plena_frontend import compile_native_hf_decoder


COMPILER_SHA256 = canonical_sha256({"compiler": "test fixture"})
INPUTS_SHA256 = canonical_sha256(
    {
        "model": "test decoder",
        "precision": "mxint8",
        "topology": {"tp": 1, "kvp": 1},
    }
)
LATENCY_SHA256 = canonical_sha256({"latency_library": "test fixture"})


class _RequestModel:
    stream_composition_schema = REQUEST_STREAM_COMPOSITION_SCHEMA
    calibration_id = "request-latency-" + canonical_sha256(
        {"request_model": "test fixture"}
    )

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, descriptor: DMARequestDescriptor) -> float:
        self.calls += 1
        return 5e-9 if descriptor.direction == "read" else 2e-9

    def predict_stream(self, runs):
        predictions = []
        for descriptor, repetitions in runs:
            seconds = self.predict(descriptor) * repetitions
            predictions.append(
                SimpleNamespace(
                    seconds=seconds,
                    isolated_seconds=seconds,
                    carried_row_conflicts=0,
                )
            )
        return tuple(predictions)


def _trace() -> ExecutionTrace:
    assembly = """\
; Normalize (rms) activation
C_LOOP_START gp1, 2
H_PREFETCH_V gp2, gp3, a0, 0, 0
V_ADD_VV gp4, gp5, gp6, 0
C_LOOP_END gp1
; Store output from VRAM to HBM
H_STORE_V gp2, gp3, a0, 0, 0
"""
    return build_execution_trace(
        assembly,
        mlen=64,
        blen=4,
        vlen=64,
        hlen=16,
        default_element_bits=8,
        default_block_size=8,
        default_scale_bits=8,
    )


def _descriptor(
    entry,
    *,
    index: int,
    hbm: HBMOperatingPoint,
) -> DMARequestDescriptor:
    rows, elements_per_row = entry.dma_shape
    element_bytes_per_row = elements_per_row
    return DMARequestDescriptor(
        opcode=entry.opcode,
        hbm_generation=hbm.generation,
        channels=hbm.channels,
        address=index * 4096,
        rows=rows,
        elements_per_row=elements_per_row,
        stride_bytes=element_bytes_per_row,
        element_bits=8,
        direction="read" if entry.dma_direction == HBM_READ else "write",
        pin_rate_gbps=hbm.pin_rate_gbps,
        tensor=entry.tensor,
        scale_bits=8,
        block_size=8,
        scale_address=(1 << 20) + index * 4096,
        scale_stride_bytes=elements_per_row // 8,
    )


def _sidecar(
    trace: ExecutionTrace,
    hbm: HBMOperatingPoint,
) -> RequestMemorySidecar:
    bindings = []
    for index, entry in enumerate(trace.entries):
        if not entry.dma_bytes:
            continue
        descriptor = _descriptor(entry, index=index, hbm=hbm)
        bindings.append(
            TraceRequestBinding(
                trace_entry_index=index,
                trace_entry_sha256=trace_entry_fingerprint(entry),
                runs=(
                    RequestDescriptorRun(
                        descriptor,
                        repetitions=entry.dynamic_count,
                    ),
                ),
            )
        )
    return RequestMemorySidecar(
        trace_assembly_sha256=trace.assembly_sha256,
        geometry=ArrayGeometry.from_trace(trace),
        bindings=tuple(bindings),
    )


def _request(
    hbm: HBMOperatingPoint | None = None,
) -> CompilerTraceTimingRequest:
    return CompilerTraceTimingRequest(
        compiler_inputs_sha256=INPUTS_SHA256,
        compiler_source_sha256=COMPILER_SHA256,
        context_tokens=128,
        batch=1,
        geometry=ArrayGeometry(64, 4, 64, 16),
        hbm=hbm or HBMOperatingPoint("HBM2", 8, 2.0),
        frequency_hz=1e9,
    )


def _provider(
    trace: ExecutionTrace,
    request_model: _RequestModel,
    *,
    sidecar: RequestMemorySidecar | None,
    builder_calls: list[CompilerTraceTimingRequest] | None = None,
    with_memory_pricer: bool = True,
) -> CompilerTraceTimingProvider:
    def build(request: CompilerTraceTimingRequest) -> BoundCompilerTrace:
        if builder_calls is not None:
            builder_calls.append(request)
        return BoundCompilerTrace(trace, COMPILER_SHA256, sidecar)

    return CompilerTraceTimingProvider(
        build,
        {
            "C_LOOP_START": 1,
            "C_LOOP_END": 1,
            "H_PREFETCH_V": 1,
            "H_STORE_V": 1,
            "V_ADD_VV": 3,
        },
        latency_library_sha256=LATENCY_SHA256,
        stage_memory_pricer=(
            RequestModelStageMemoryPricer(request_model)
            if with_memory_pricer
            else None
        ),
    )


def test_compiler_trace_mode_prices_stages_and_reports_exact_provenance() -> None:
    trace = _trace()
    request = _request()
    request_model = _RequestModel()
    sidecar = _sidecar(trace, request.hbm)
    result = _provider(trace, request_model, sidecar=sidecar).evaluate(request)

    assert [stage.stage for stage in result.stages] == ["RMSNorm", "KV store"]
    # Both fixture stages carry vector DMA, which the consuming instruction
    # waits on, so every memory cycle is exposed on top of compute.
    assert [
        (
            stage.compute_cycles,
            stage.matrix_memory_cycles,
            stage.vector_memory_cycles,
            stage.cycles,
        )
        for stage in result.stages
    ] == [(11, 0, 10, 21), (1, 0, 2, 3)]
    assert result.total_cycles == 24
    assert result.total_seconds == 24e-9
    assert result.step_composition == "max_compute_matrix_dma_plus_vector_dma"
    stream_pricer = RequestModelStageMemoryPricer(request_model)
    assert stream_pricer.calibration_id == "request-latency-" + canonical_sha256(
        {
            "base_request_calibration_id": request_model.calibration_id,
            "request_stream_composition": REQUEST_STREAM_COMPOSITION_SCHEMA,
        }
    )
    assert result.provenance == {
        "schema_version": "plena-compiler-trace-timing-v1",
        "execution_mode": "compiler_trace",
        "reason": "compiler_trace_timing_validated",
        "request_id": request.request_id,
        "trace_assembly_sha256": trace.assembly_sha256,
        "compiler_inputs_sha256": INPUTS_SHA256,
        "compiler_source_sha256": COMPILER_SHA256,
        "latency_library_sha256": LATENCY_SHA256,
        "request_memory_sidecar_sha256": sidecar.sidecar_sha256,
        "memory_calibration_id": stream_pricer.calibration_id,
        "base_memory_calibration_id": request_model.calibration_id,
        "request_stream_composition": REQUEST_STREAM_COMPOSITION_SCHEMA,
        "artifact_scope": None,
        "geometry": {"mlen": 64, "blen": 4, "vlen": 64, "hlen": 16},
        "hbm": request.hbm.to_dict(),
        "frequency_hz": 1e9,
        "step_composition": "max_compute_matrix_dma_plus_vector_dma",
    }
    # Request-memory evidence remains beside the settled nine-field trace key.
    assert all(len(entry.key) == 9 for entry in trace.entries)


def test_provider_reuses_algebraic_trace_and_descriptor_runs_by_exact_request() -> None:
    trace = _trace()
    request = _request()
    calls: list[CompilerTraceTimingRequest] = []
    request_model = _RequestModel()
    provider = _provider(
        trace,
        request_model,
        sidecar=_sidecar(trace, request.hbm),
        builder_calls=calls,
    )

    first, second, third = provider.prepare((request, request, request))
    assert first is second is third
    assert calls == [request]
    # One prediction per unique DMA entry; repetition is multiplied algebraically.
    assert request_model.calls == 2
    assert provider.cache_info() == {"hits": 2, "misses": 1, "size": 1}


def test_request_identity_separates_context_points() -> None:
    request = _request()
    other = replace(request, context_tokens=256)
    assert request.request_id != other.request_id
    assert request != other


@pytest.mark.parametrize("missing", ["sidecar", "calibration"])
def test_compiler_trace_mode_fails_closed_when_evidence_is_absent(missing: str) -> None:
    trace = _trace()
    request = _request()
    request_model = _RequestModel()
    provider = _provider(
        trace,
        request_model,
        sidecar=None if missing == "sidecar" else _sidecar(trace, request.hbm),
        with_memory_pricer=missing != "calibration",
    )
    match = "sidecar" if missing == "sidecar" else "calibrated"
    with pytest.raises(RuntimeError, match=match):
        provider.evaluate(request)


def test_sidecar_rejects_stale_entry_and_hbm_operating_point() -> None:
    trace = _trace()
    request = _request()
    sidecar = _sidecar(trace, request.hbm)
    first = sidecar.bindings[0]
    stale = replace(
        sidecar,
        bindings=(
            replace(first, trace_entry_sha256="0" * 64),
            *sidecar.bindings[1:],
        ),
    )
    with pytest.raises(ValueError, match="entry identity is stale"):
        stale.validate(trace, request.hbm)

    different_rate = HBMOperatingPoint("HBM2", 8, 2.4)
    with pytest.raises(ValueError, match="operating point differs"):
        sidecar.validate(trace, different_rate)


def test_request_memory_pricer_rejects_uncalibrated_hbm_geometry() -> None:
    trace = _trace()
    hbm = HBMOperatingPoint("HBM2", 8, 2.0, channel_width_bits=128)
    request = _request(hbm)
    provider = _provider(
        trace,
        _RequestModel(),
        sidecar=_sidecar(trace, hbm),
    )
    with pytest.raises(ValueError, match="HBM geometry differs"):
        provider.evaluate(request)


def test_compiler_sidecar_converts_without_reconstructing_dma_addresses() -> None:
    compiler = PlenaCompiler(
        mlen=64,
        blen=4,
        hbm_element_width=4,
        hbm_block_size=8,
        hbm_scale_width=8,
    )
    source = compiler.input(
        "activation",
        (1, 64),
        physical_shape=(4, 64),
        precision_role="activation",
    )
    compiler.load_batch(source)
    artifact = compiler.compile_with_trace()
    hbm = HBMOperatingPoint("HBM2", 8, 2.0)

    bound = BoundCompilerTrace.from_compilation_artifact(
        artifact,
        compiler_source_sha256=COMPILER_SHA256,
        hbm=hbm,
    )

    assert bound.request_memory is not None
    bound.request_memory.validate(bound.execution_trace, hbm)
    compiler_request = artifact.request_memory.bindings[0].runs[0].request
    descriptor = bound.request_memory.bindings[0].runs[0].descriptor
    assert descriptor.address == compiler_request.address
    assert descriptor.scale_address == compiler_request.scale_address
    assert descriptor.stride_bytes == compiler_request.stride_bytes

    serialized = artifact.to_dict()
    restored = CompilationArtifact.from_dict(serialized)
    assert restored == artifact
    trace_value = serialized["execution_trace"]
    assert isinstance(trace_value, dict)
    trace_value["assembly_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="assembly and trace differ"):
        CompilationArtifact.from_dict(serialized)


def test_request_memory_pricer_requires_content_addressed_calibration() -> None:
    class WeaklyIdentifiedModel:
        calibration_id = "request-latency-unsealed"

        @staticmethod
        def predict(descriptor: DMARequestDescriptor) -> float:
            del descriptor
            return 0.0

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        RequestModelStageMemoryPricer(WeaklyIdentifiedModel())


@pytest.mark.parametrize(
    "context_tokens,expected_carried_conflicts",
    ((128, 0), (256, 40), (512, 120), (1024, 136)),
)
def test_reference_request_stream_carries_exact_open_row_state(
    tmp_path,
    context_tokens: int,
    expected_carried_conflicts: int,
) -> None:
    from transactional_emulator.testbench.misc.decoder_decode_asm_gen import (
        generate_decode_asm,
    )

    simulator_root = Path(__file__).resolve().parents[2]
    generated = generate_decode_asm(
        kv_size=context_tokens,
        hidden=64,
        inter=128,
        head_dim=16,
        build_dir=str(tmp_path),
        settings_toml=str(simulator_root / "plena_settings.toml"),
        verbose=False,
    )
    artifact = generated["compilation_artifact"]
    assert artifact.request_memory is not None
    hbm = HBMOperatingPoint("HBM2", 8, 2.0)
    sidecar = request_memory_sidecar_from_compiler(
        artifact.request_memory,
        hbm,
    )

    row_conflict_index = REQUEST_FEATURES.index("row_conflicts")
    coefficients = [0.0] * len(REQUEST_FEATURES)
    coefficients[row_conflict_index] = 1e-9
    opcodes = {
        run.descriptor.opcode
        for binding in sidecar.bindings
        for run in binding.runs
    }
    base_calibration_id = "request-latency-" + "b" * 64
    request_model = RequestLatencyModel(
        tuple(
            RequestModelFit(
                opcode=opcode,
                hbm_generation="HBM2",
                channels=8,
                coefficients_s=tuple(coefficients),
                training_points=1,
            )
            for opcode in sorted(opcodes)
        ),
        base_calibration_id,
    )
    pricer = RequestModelStageMemoryPricer(request_model)

    # Mirror the pricer: an affine run stands for distinct addresses, so each
    # repetition enters the stream with its own descriptor.
    ordered = []
    for binding in sorted(
        sidecar.bindings,
        key=lambda item: item.trace_entry_index,
    ):
        stage = artifact.execution_trace.entries[binding.trace_entry_index].stage
        for run in binding.runs:
            if run.address_varying:
                ordered.extend(
                    (stage, run.descriptor_at(index), 1)
                    for index in range(run.repetitions)
                )
            else:
                ordered.append((stage, run.descriptor, run.repetitions))
    predictions = request_model.predict_stream(
        tuple(
            (descriptor, repetitions)
            for _stage, descriptor, repetitions in ordered
        )
    )
    observed_conflicts = sum(
        prediction.carried_row_conflicts
        for (stage, _descriptor, _repetitions), prediction in zip(
            ordered,
            predictions,
            strict=True,
        )
        if stage == "KV store"
    )
    isolated_store_seconds = sum(
        request_model.predict(descriptor) * repetitions
        for stage, descriptor, repetitions in ordered
        if stage == "KV store"
    )
    priced_store_seconds = pricer.price_trace(
        artifact.execution_trace,
        sidecar,
        hbm,
    )["KV store"]

    assert observed_conflicts == expected_carried_conflicts
    assert priced_store_seconds - isolated_store_seconds == pytest.approx(
        expected_carried_conflicts * 1e-9
    )
    assert request_model.calibration_id == base_calibration_id
    assert pricer.base_calibration_id == base_calibration_id
    assert pricer.calibration_id != base_calibration_id


def _full_model_point_descriptor() -> dict[str, object]:
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
            "hbm_timing_geometry": HBMOperatingPoint(
                "HBM2",
                8,
                2.0,
            ).to_dict(),
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
            "batch": 2,
            "input_tokens": 17,
            "generation_tokens": 1,
            "sample_stride": 1,
            "kv_layout": "dense_selector",
            "runtime_hbm_reserve_bytes": 0,
        },
        "compiler": {
            "settings_sha256": canonical_sha256({"settings": "tiny"}),
            "latency_library_sha256": canonical_sha256({"latency": "tiny"}),
            "timing_mode": "rtl_serialized",
            "frequency_hz": 1.0e9,
        },
    }


def _full_model_compilation_result() -> dict[str, object]:
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        rms_norm_eps=1.0e-5,
        rope_theta=10000.0,
        vocab_size=32,
        model_type="llama",
    )
    return compile_native_hf_decoder(
        SimpleNamespace(config=config, layers=[None]),
        seq_len=1,
        batch_size=2,
        num_layers=1,
        mlen=16,
        blen=2,
        hlen=8,
        broadcast_amount=2,
        attention_head_packing=True,
        packed_kv_layout=PackedKVLayout(
            kv_heads=2,
            head_dim=8,
            mlen=16,
            element_bits=4,
        ),
        decode_context_tokens=17,
        external_packed_kv_cache=True,
        trace_only=True,
        output_head_location="prefill_chip",
        weight_element_bits=4,
    )


def test_full_model_artifact_set_round_trip_and_exact_context_binding(
    tmp_path,
    monkeypatch,
) -> None:
    descriptor = _full_model_point_descriptor()
    compilation_result = _full_model_compilation_result()
    builder = FullModelDecodeArtifactSetBuilder(COMPILER_SHA256)
    builder.add(
        descriptor,
        context_tokens=17,
        compilation_result=compilation_result,
        critical_rank={
            "tensor_parallel_rank": 0,
            "kv_parallel_rank": 0,
            "kv_token_sharding": "round_robin",
            "owns_current_token": True,
        },
    )
    mismatched_result = {
        **compilation_result,
            "info": {
                **compilation_result["info"],
                "decode_context_tokens": 18,
                "local_decode_context_tokens": 18,
                "local_cache_position": 17,
            },
    }
    with pytest.raises(ValueError, match="append address differs"):
        builder.add(
            descriptor,
            context_tokens=18,
            compilation_result=mismatched_result,
            critical_rank={
                "tensor_parallel_rank": 0,
                "kv_parallel_rank": 0,
                "kv_token_sharding": "round_robin",
                "owns_current_token": True,
            },
        )
    artifact_set = builder.build()
    path = artifact_set.write(tmp_path / "compiler-trace-artifacts.json")
    restored = FullModelDecodeArtifactSet.load(path)
    point_id = canonical_sha256(descriptor)
    _, lowering_id, _ = full_model_decode_lowering_key(descriptor)
    binder = FullModelDecodeArtifactBinder(restored)
    request = binder.bind(descriptor)(17)

    assert restored.artifact_set_id.startswith("compiler-trace-artifacts-")
    assert restored.artifact_set_id == artifact_set.artifact_set_id
    assert restored.contexts(lowering_id) == (17,)
    assert request.compiler_inputs_sha256 == point_id
    assert request.compiler_lowering_sha256 == lowering_id
    assert request.context_tokens == 17
    assert request.batch == 2
    assert request.geometry == ArrayGeometry(16, 2, 16, 8)
    lazy_request = binder.bind(descriptor)(18)
    assert lazy_request.context_tokens == 18
    lazy_record = restored.resolve_record(descriptor, context_tokens=18)
    assert lazy_record.context_tokens == 18
    assert restored.contexts(lowering_id) == (17, 18)
    assert lazy_record.compilation_artifact != compilation_result["compilation_artifact"]

    nonshaping_variant = json.loads(json.dumps(descriptor))
    nonshaping_variant["model"]["output_head_location"] = (
        "external_bf16_service"
    )
    nonshaping_variant["hardware"]["hbm_timing_geometry"]["channels"] = 16
    nonshaping_variant["hardware"]["topology"]["link_generation"] = (
        "p2p_200GBs"
    )
    nonshaping_variant["serving"]["generation_tokens"] = 64
    variant_request = binder.bind(nonshaping_variant)(17)
    assert variant_request.compiler_inputs_sha256 != point_id
    assert variant_request.compiler_lowering_sha256 == lowering_id

    monkeypatch.setattr(
        "analytic_models.performance.compiler_trace_timing."
        "native_decode_compiler_source_sha256",
        lambda: "b" * 64,
    )
    with pytest.raises(ValueError, match="different compiler source"):
        create_full_model_decode_artifact_runtime(path)

    value = restored.to_dict()
    value["records"][0]["lowering_contract"]["query_tokens"] = 2
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="q_len=1"):
        FullModelDecodeArtifactSet.load(path)


def test_full_model_artifact_dry_run_counts_lowering_keys_not_aliases() -> None:
    descriptor = _full_model_point_descriptor()
    nonshaping_variant = json.loads(json.dumps(descriptor))
    nonshaping_variant["hardware"]["hbm_timing_geometry"].update(
        {"generation": "HBM3", "channels": 16, "pin_rate_gbps": 3.2}
    )
    nonshaping_variant["hardware"]["configuration"]["HBM_SIZE"] = 32
    nonshaping_variant["hardware"]["topology"]["drain_overlapped"] = True
    nonshaping_variant["serving"]["sample_stride"] = 8
    shaping_variant = json.loads(json.dumps(descriptor))
    shaping_variant["serving"]["batch"] = 4

    plan = FullModelDecodeArtifactBuildPlan.from_point_contexts(
        (
            (descriptor, (17, 33)),
            (nonshaping_variant, (17, 65)),
            (shaping_variant, (17,)),
        )
    )
    summary = plan.to_dict()

    assert plan.full_point_count == 3
    assert plan.unique_lowering_key_count == 2
    assert plan.unique_native_template_count == 1
    assert plan.unique_batch_record_count == 2
    assert plan.exact_point_context_resolution_count == 5
    assert plan.unique_lowering_context_resolution_count == 4
    assert plan.context_artifact_count == 0
    assert summary["materialized_alias_records"] == 0
    assert summary["materialized_context_rows"] == 0
    assert summary["materialized_batch_alias_records"] == 0
    assert summary["compile_count_formula"] == (
        "|unique (native template, batch) keys|"
    )
    assert summary["unique_compiler_family_artifacts"] == 1
    assert summary["unique_native_trace_templates"] == 1
    assert summary["unique_exact_batch_records"] == 2
    assert summary["unique_lazy_trace_instantiations"] == 2
    assert summary["projected_trace_generation_calls"] == 2
    assert summary["exact_point_context_resolutions"] == 5
    assert summary["unique_lowering_context_resolutions"] == 4
    assert summary["projected_trace_bytes"] == 331_776
    assert summary["compiler_trace_preflight_feasible"] is True


@pytest.mark.parametrize(
    ("context_tokens", "full_blocks", "tail", "blocks", "rows", "append"),
    (
        (15, 0, 15, 1, 16, 14),
        (16, 1, 0, 1, 16, 15),
        (17, 1, 1, 2, 32, 16),
        (32, 2, 0, 2, 32, 31),
        (33, 2, 1, 3, 48, 32),
    ),
)
def test_full_model_context_resolution_covers_loop_and_tail_boundaries(
    context_tokens: int,
    full_blocks: int,
    tail: int,
    blocks: int,
    rows: int,
    append: int,
) -> None:
    descriptor = _full_model_point_descriptor()
    resolution = full_model_decode_context_resolution(
        descriptor,
        context_tokens=context_tokens,
    )

    assert (
        resolution.local_full_block_count,
        resolution.local_tail_columns,
        resolution.local_cache_block_count,
        resolution.local_cache_rows_per_batch,
        resolution.local_append_token_index,
    ) == (full_blocks, tail, blocks, rows, append)
    assert resolution.has_masked_tail is (tail > 0)
    assert resolution.attention_block_count == blocks
    assert resolution.to_dict()["materialized_context_rows"] == 0


@pytest.mark.parametrize("batch", (1, 8, 256))
def test_full_model_batch_resolution_is_exact_and_non_materialized(
    batch: int,
) -> None:
    descriptor = _full_model_point_descriptor()
    descriptor["serving"]["batch"] = batch
    resolution = full_model_decode_batch_resolution(descriptor)

    assert resolution == FullModelDecodeBatchResolution.resolve(batch)
    assert resolution.native_template_batch == 1
    assert resolution.independent_slab_count == batch
    assert resolution.resolved_active_rows == batch
    assert resolution.is_identity is (batch == 1)
    assert resolution.to_dict()["slab_ordinal_range"] == {
        "start": 0,
        "stop": batch,
        "step": 1,
    }
    assert resolution.to_dict()["requires_exact_batch_record"] is True
    assert resolution.to_dict()["artifact_alias_permitted"] is False
    assert resolution.to_dict()["materialized_batch_alias_records"] == 0


@pytest.mark.parametrize(
    ("context_tokens", "owner", "local_tokens", "local_tail", "local_append"),
    ((31, 2, 8, 8, 7), (32, 3, 8, 8, 7), (33, 0, 9, 9, 8)),
)
def test_context_resolution_binds_the_round_robin_critical_rank(
    context_tokens: int,
    owner: int,
    local_tokens: int,
    local_tail: int,
    local_append: int,
) -> None:
    descriptor = _full_model_point_descriptor()
    descriptor["hardware"]["topology"].update(
        {"tp": 1, "kvp": 4, "chip_count": 4}
    )
    resolution = full_model_decode_context_resolution(
        descriptor,
        context_tokens=context_tokens,
    )

    assert resolution.kv_parallel_degree == 4
    assert resolution.kv_parallel_rank == owner
    assert resolution.local_context_tokens == local_tokens
    assert resolution.local_tail_columns == local_tail
    assert resolution.local_append_token_index == local_append
    assert resolution.local_cache_rows_per_batch == 16
    assert resolution.global_append_token_index == context_tokens - 1


def test_batch_free_template_identity_does_not_alias_exact_records() -> None:
    descriptor = _full_model_point_descriptor()
    batch_one = json.loads(json.dumps(descriptor))
    batch_one["serving"]["batch"] = 1
    batch_256 = json.loads(json.dumps(descriptor))
    batch_256["serving"]["batch"] = 256

    assert full_model_decode_lowering_key(batch_one)[1] != (
        full_model_decode_lowering_key(batch_256)[1]
    )
    assert full_model_decode_native_template_key(batch_one)[1] == (
        full_model_decode_native_template_key(batch_256)[1]
    )
    assert canonical_sha256(batch_one) != canonical_sha256(batch_256)


def test_full_model_family_is_compact_and_generator_geometry_fails_closed() -> None:
    descriptor = _full_model_point_descriptor()
    family = FullModelDecodeArtifactFamily.from_point_descriptor(
        descriptor,
        compiler_source_sha256=COMPILER_SHA256,
    )
    restored = FullModelDecodeArtifactFamily.from_dict(family.to_dict())
    family_json, family_key_sha256, key = full_model_decode_family_key(descriptor)

    assert restored == family
    assert family.family_key_json == family_json
    assert canonical_sha256(key) == family_key_sha256
    assert "dimensions" not in key["model_template"]
    assert full_model_decode_generator_blockers(descriptor) == ()

    distributed = json.loads(json.dumps(descriptor))
    distributed["hardware"]["topology"].update(
        {"tp": 2, "chip_count": 2}
    )
    assert full_model_decode_generator_blockers(distributed) == ()

    unsupported = json.loads(json.dumps(descriptor))
    unsupported["hardware"]["array_geometry"]["hlen"] = 16
    assert "packed_attention_head_geometry_unsupported" in (
        full_model_decode_generator_blockers(unsupported)
    )
    plan = FullModelDecodeArtifactBuildPlan.from_point_contexts(
        ((unsupported, (17,)),)
    )
    assert plan.compiler_trace_preflight_feasible is False
    assert any(
        "packed_attention_head_geometry_unsupported" in blocker
        for blocker in plan.preflight_blockers
    )


def test_full_model_residency_is_priced_outside_native_lowering() -> None:
    artifact = _full_model_compilation_result()["compilation_artifact"]
    hbm = HBMOperatingPoint("HBM2", 8, 2.0)
    sidecar = request_memory_sidecar_from_compiler(
        artifact.request_memory,
        hbm,
    )
    base = RequestModelStageMemoryPricer(_RequestModel())
    streaming = base.price_trace(artifact.execution_trace, sidecar, hbm)
    projection = ResidencyAdjustedStageMemoryPricer(
        base,
        "projection_resident",
    ).price_trace(artifact.execution_trace, sidecar, hbm)
    kv_resident = ResidencyAdjustedStageMemoryPricer(
        base,
        "kv_resident_100",
    ).price_trace(artifact.execution_trace, sidecar, hbm)

    assert sum(projection.values()) < sum(streaming.values())
    assert sum(kv_resident.values()) < sum(streaming.values())
    assert projection["Q/K/V + W_O projection"] < (
        streaming["Q/K/V + W_O projection"]
    )
    assert "Flash attention" not in kv_resident
    assert "KV store" not in kv_resident


@pytest.mark.parametrize(
    ("element_bits", "mlen", "context_tokens", "batch"),
    ((2, 16, 17, 1), (4, 16, 33, 2), (8, 32, 65, 3)),
)
def test_lazy_family_generation_equals_direct_native_compilation(
    element_bits: int,
    mlen: int,
    context_tokens: int,
    batch: int,
) -> None:
    descriptor = _full_model_point_descriptor()
    descriptor["precision"]["specification"].update(
        {
            "attn_elem": element_bits,
            "ffn_elem": element_bits,
            "kv_elem": element_bits,
        }
    )
    descriptor["hardware"]["array_geometry"].update(
        {"mlen": mlen, "vlen": mlen}
    )
    descriptor["serving"]["batch"] = batch
    family = FullModelDecodeArtifactFamily.from_point_descriptor(
        descriptor,
        compiler_source_sha256=COMPILER_SHA256,
    )
    generated = FullModelDecodeLazyArtifactGenerator(family).instantiate(
        descriptor,
        context_tokens=context_tokens,
    )
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        rms_norm_eps=1.0e-5,
        rope_theta=10000.0,
        vocab_size=32,
        model_type="llama",
    )
    direct = compile_native_hf_decoder(
        SimpleNamespace(config=config, layers=[None]),
        seq_len=1,
        batch_size=batch,
        num_layers=1,
        mlen=mlen,
        blen=2,
        hlen=8,
        broadcast_amount=mlen // 8,
        attention_head_packing=True,
        packed_kv_layout=PackedKVLayout(
            kv_heads=2,
            head_dim=8,
            mlen=mlen,
            element_bits=element_bits,
        ),
        decode_context_tokens=context_tokens,
        external_packed_kv_cache=True,
        trace_only=True,
        output_head_location="prefill_chip",
        weight_element_bits=element_bits,
    )

    assert generated.compilation_artifact == direct["compilation_artifact"]
    assert generated.compiler_receipt == direct["info"]


def test_lazy_generation_seals_qk_norm_formats_and_split_kv_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _full_model_point_descriptor()
    descriptor["model"]["dimensions"].update(
        {"model_type": "qwen3", "qk_norm": True}
    )
    descriptor["precision"].update(
        {
            "weight_format": "mxfp",
            "kv_format": "split",
            "key_format": "mxint",
            "value_format": "mxfp",
        }
    )
    descriptor["precision"]["specification"].update(
        {"kv_elem": 8, "key_elem": 4, "value_elem": 8}
    )
    family = FullModelDecodeArtifactFamily.from_point_descriptor(
        descriptor,
        compiler_source_sha256=COMPILER_SHA256,
    )
    generated = FullModelDecodeLazyArtifactGenerator(family).instantiate(
        descriptor,
        context_tokens=33,
    )
    receipt = generated.compiler_receipt

    assert receipt["compiled_qk_norm"] is True
    assert (
        receipt["qk_norm_segment_width"],
        receipt["qk_norm_reciprocal_fp_offset"],
        receipt["qk_norm_affine_storage_shape"],
        receipt["qk_norm_affine_pattern"],
    ) == (
        8,
        6,
        [4, 16],
        "shared_head_weight_repeated_per_mlen_vector",
    )
    assert (
        receipt["weight_storage_format"],
        receipt["kv_storage_format"],
        receipt["key_storage_format"],
        receipt["value_storage_format"],
    ) == ("mxfp", "mxint", "mxint", "mxfp")
    assert (
        receipt["packed_key_element_bits"],
        receipt["packed_value_element_bits"],
    ) == (4, 8)
    assert receipt["packed_key_layout_id"] != receipt["packed_value_layout_id"]

    format_alias = json.loads(json.dumps(descriptor))
    format_alias["precision"].update(
        {
            "weight_format": "mxint",
            "key_format": "mxfp",
            "value_format": "mxint",
        }
    )
    assert full_model_decode_family_key(format_alias)[1] == (
        full_model_decode_family_key(descriptor)[1]
    )
    assert full_model_decode_lowering_key(format_alias)[1] == (
        full_model_decode_lowering_key(descriptor)[1]
    )
    assert canonical_sha256(format_alias) != canonical_sha256(descriptor)

    alias_generated = FullModelDecodeLazyArtifactGenerator(family).instantiate(
        format_alias,
        context_tokens=33,
    )
    assert alias_generated.compilation_artifact == generated.compilation_artifact
    assert (
        alias_generated.compiler_receipt["weight_storage_format"],
        alias_generated.compiler_receipt["key_storage_format"],
        alias_generated.compiler_receipt["value_storage_format"],
    ) == ("mxint", "mxfp", "mxint")

    artifact_set = FullModelDecodeArtifactSet((generated,))
    monkeypatch.setattr(
        FullModelDecodeLazyArtifactGenerator,
        "instantiate",
        lambda *args, **kwargs: pytest.fail(
            "a receipt-only storage alias must not invoke the compiler"
        ),
    )
    rebound = artifact_set.resolve_record(format_alias, context_tokens=33)
    assert rebound.compilation_artifact is generated.compilation_artifact
    assert rebound.record_sha256 != generated.record_sha256
    assert rebound.compiler_receipt == alias_generated.compiler_receipt


def test_lazy_generation_seals_rank_local_partition_and_disabled_reuse() -> None:
    descriptor = _full_model_point_descriptor()
    descriptor["hardware"]["topology"].update(
        {
            "tp": 2,
            "kvp": 2,
            "chip_count": 4,
            "kv_head_reuse": False,
        }
    )
    assert full_model_decode_generator_blockers(descriptor) == ()
    family = FullModelDecodeArtifactFamily.from_point_descriptor(
        descriptor,
        compiler_source_sha256=COMPILER_SHA256,
    )
    generated = FullModelDecodeLazyArtifactGenerator(family).instantiate(
        descriptor,
        context_tokens=18,
    )
    receipt = generated.compiler_receipt

    assert (
        receipt["tensor_parallel_degree"],
        receipt["tensor_parallel_rank"],
        receipt["kv_parallel_degree"],
        receipt["kv_parallel_rank"],
    ) == (2, 0, 2, 1)
    assert (
        receipt["local_num_heads"],
        receipt["local_num_kv_heads"],
        receipt["local_inter_dim"],
    ) == (1, 1, 16)
    assert receipt["tensor_parallel_query_head_range"] == [0, 1]
    assert receipt["tensor_parallel_kv_head_range"] == [0, 1]
    assert receipt["local_decode_context_tokens"] == 9
    assert receipt["local_cache_position"] == 8
    assert receipt["cache_rows_per_batch"] == 16
    assert receipt["compiled_kv_head_reuse"] is False
    assert receipt["external_collectives"] == [
        "attention_output_all_reduce",
        "ffn_down_output_all_reduce",
        "attention_logsumexp_reduce",
    ]


def test_large_exact_batch_record_seals_query_tiling_and_cache_slabs() -> None:
    descriptor = _full_model_point_descriptor()
    descriptor["serving"]["batch"] = 256
    family = FullModelDecodeArtifactFamily.from_point_descriptor(
        descriptor,
        compiler_source_sha256=COMPILER_SHA256,
    )
    record = FullModelDecodeLazyArtifactGenerator(family).instantiate(
        descriptor,
        context_tokens=17,
    )
    receipt = record.compiler_receipt

    assert receipt["batch_size"] == 256
    assert receipt["active_rows"] == 256
    assert receipt["rows_per_batch"] == 16
    assert receipt["compile_seq_rows"] == 4_096
    assert receipt["cache_rows_per_batch"] == 32
    for cache in ("K_cache_0", "V_cache_0"):
        stores = [
            entry
            for entry in record.compilation_artifact.execution_trace.entries
            if entry.tensor == cache and entry.opcode == "H_STORE_V"
        ]
        assert sum(entry.dynamic_count for entry in stores) == 256


def test_reference_builder_compiles_exact_scope_and_rejects_serving_batch() -> None:
    builder = ReferenceDecodeArtifactBuilder(
        ReferenceDecodeLowering(
            intermediate_size=256,
            vocabulary_size=256,
        )
    )
    hbm = HBMOperatingPoint("HBM2", 8, 2.0)
    request = builder.request(
        context_tokens=128,
        hbm=hbm,
        frequency_hz=1e9,
    )
    artifact = builder(request)
    scoped_provider = CompilerTraceTimingProvider(
        builder,
        {"C_BREAK": 1},
        latency_library_sha256=LATENCY_SHA256,
        stage_memory_pricer=None,
    )

    assert builder.scope == REFERENCE_DECODE_SCOPE
    assert scoped_provider.artifact_scope == REFERENCE_DECODE_SCOPE
    assert builder.scope != FULL_MODEL_DECODE_SCOPE
    assert artifact.request_memory is not None
    artifact.request_memory.validate(artifact.execution_trace, hbm)
    dma_roles = {
        entry.precision_mode
        for entry in artifact.execution_trace.entries
        if entry.dma_bytes
    }
    assert dma_roles == {"activation", "weight", "key", "value"}
    assert sum(
        run.repetitions
        for binding in artifact.request_memory.bindings
        for run in binding.runs
    ) == 76

    with pytest.raises(ValueError, match="independent-request serving batch"):
        builder(replace(request, batch=1))


def test_execution_mode_resolver_has_no_implicit_fallback() -> None:
    trace = _trace()
    request = _request()
    provider = _provider(
        trace,
        _RequestModel(),
        sidecar=_sidecar(trace, request.hbm),
    )
    resolved = resolve_decode_step_timing(
        COMPILER_TRACE,
        trace_timing_provider=provider,
        trace_request=request,
    )
    assert resolved.total_seconds == 24e-9
    assert resolved.reason == "compiler_trace_timing_validated"

    legacy = resolve_decode_step_timing(
        LEGACY_AGGREGATE_BANDWIDTH,
        legacy_compute_seconds=7e-9,
        legacy_memory_seconds=11e-9,
    )
    assert legacy.total_seconds == 11e-9
    assert legacy.reason == "legacy_aggregate_bandwidth_compatibility"

    with pytest.raises(RuntimeError, match="provider and exact request"):
        resolve_decode_step_timing(COMPILER_TRACE)
    with pytest.raises(ValueError, match="rejects legacy"):
        resolve_decode_step_timing(
            COMPILER_TRACE,
            trace_timing_provider=provider,
            trace_request=request,
            legacy_compute_seconds=1.0,
        )
    with pytest.raises(ValueError, match="rejects compiler trace"):
        resolve_decode_step_timing(
            LEGACY_AGGREGATE_BANDWIDTH,
            trace_timing_provider=provider,
            trace_request=request,
            legacy_compute_seconds=1.0,
            legacy_memory_seconds=1.0,
        )
