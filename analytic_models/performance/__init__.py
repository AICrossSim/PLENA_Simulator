from __future__ import annotations

from importlib import import_module

__all__ = [
    "COMPILER_TRACE",
    "FULL_MODEL_DECODE_SCOPE",
    "FULL_MODEL_BATCH_RESOLUTION_MODE",
    "FULL_MODEL_CONTEXT_RESOLUTION_MODE",
    "LEGACY_AGGREGATE_BANDWIDTH",
    "CompilerTraceTimingProvider",
    "CompilerTraceTimingRequest",
    "FullModelDecodeArtifactRuntime",
    "FullModelDecodeArtifactSet",
    "FullModelDecodeArtifactBuildPlan",
    "FullModelDecodeBatchResolution",
    "FullModelDecodeContextResolution",
    "FullModelDecodeArtifactFamily",
    "FullModelDecodeLazyArtifactGenerator",
    "ReferenceDecodeArtifactBuilder",
    "ReferenceDecodeLowering",
    "ReferenceDecodeTimingRuntime",
    "DecodeCost",
    "DecodeCostModel",
    "HardwareConfig",
    "InstructionLatency",
    "PerfModel",
    "StageCost",
    "build_pipelined_latency",
    "create_reference_decode_timing_runtime",
    "create_full_model_decode_artifact_runtime",
    "full_model_decode_lowering_key",
    "full_model_decode_batch_resolution",
    "full_model_decode_context_resolution",
    "full_model_decode_family_key",
    "full_model_decode_native_template_key",
    "full_model_decode_generator_blockers",
    "load_hardware_config_from_toml",
    "resolve_decode_step_timing",
    "validate_packed_q1_execution_trace",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    if name in {
        "COMPILER_TRACE",
        "FULL_MODEL_DECODE_SCOPE",
        "FULL_MODEL_BATCH_RESOLUTION_MODE",
        "FULL_MODEL_CONTEXT_RESOLUTION_MODE",
        "LEGACY_AGGREGATE_BANDWIDTH",
        "CompilerTraceTimingProvider",
        "CompilerTraceTimingRequest",
        "FullModelDecodeArtifactRuntime",
        "FullModelDecodeArtifactSet",
        "FullModelDecodeArtifactBuildPlan",
        "FullModelDecodeBatchResolution",
        "FullModelDecodeContextResolution",
        "FullModelDecodeArtifactFamily",
        "FullModelDecodeLazyArtifactGenerator",
        "ReferenceDecodeArtifactBuilder",
        "ReferenceDecodeLowering",
        "ReferenceDecodeTimingRuntime",
        "create_reference_decode_timing_runtime",
        "create_full_model_decode_artifact_runtime",
        "full_model_decode_lowering_key",
        "full_model_decode_batch_resolution",
        "full_model_decode_context_resolution",
        "full_model_decode_family_key",
        "full_model_decode_native_template_key",
        "full_model_decode_generator_blockers",
        "resolve_decode_step_timing",
    }:
        module = ".compiler_trace_timing"
    elif name in {
        "DecodeCost",
        "DecodeCostModel",
        "StageCost",
        "validate_packed_q1_execution_trace",
    }:
        module = ".decode_cost_model"
    else:
        module = ".perf_model"
    return getattr(import_module(module, __name__), name)
