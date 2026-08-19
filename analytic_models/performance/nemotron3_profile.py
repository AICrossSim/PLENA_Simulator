"""Validated input contract for future Nemotron 3 GPU profiling results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .nemotron3_workload import InferencePhase


CANONICAL_STAGES = {
    "embedding",
    "block_rms_norm",
    "mamba_in_projection",
    "mamba_conv1d",
    "mamba_dt_exp",
    "mamba_state_update",
    "mamba_state_output",
    "mamba_state_update_output_fused",
    "mamba_gate_group_rms_norm",
    "mamba_out_projection",
    "attention_qkv_projection",
    "attention_qk_softmax_pv",
    "attention_out_projection",
    "moe_router_topk",
    "moe_routed_experts",
    "moe_shared_expert",
    "block_residual",
    "lm_head",
    "other",
}


class ProfileFormatError(ValueError):
    pass


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileFormatError(f"{path} must be an object")
    return value


def _string(mapping: dict[str, Any], key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ProfileFormatError(f"{path}.{key} must be a non-empty string")
    return value


def _number(
    mapping: dict[str, Any],
    key: str,
    path: str,
    *,
    integer: bool = False,
    minimum: int | float = 0,
) -> int | float:
    value = mapping.get(key)
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected) or value < minimum:
        kind = "integer" if integer else "number"
        raise ProfileFormatError(f"{path}.{key} must be a {kind} >= {minimum}")
    return value


@dataclass(frozen=True)
class ProfileEnvironment:
    gpu_model: str
    gpu_count: int
    driver_version: str
    cuda_version: str
    framework_version: str
    dtype: str
    state_dtype: str

    @classmethod
    def from_dict(cls, raw: Any) -> ProfileEnvironment:
        data = _mapping(raw, "environment")
        return cls(
            gpu_model=_string(data, "gpu_model", "environment"),
            gpu_count=int(_number(data, "gpu_count", "environment", integer=True, minimum=1)),
            driver_version=_string(data, "driver_version", "environment"),
            cuda_version=_string(data, "cuda_version", "environment"),
            framework_version=_string(data, "framework_version", "environment"),
            dtype=_string(data, "dtype", "environment"),
            state_dtype=_string(data, "state_dtype", "environment"),
        )


@dataclass(frozen=True)
class KernelMeasurement:
    canonical_stage: str
    layer_type: str
    kernel_name: str
    calls: int
    total_time_us: float
    dram_read_bytes: int
    dram_write_bytes: int

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> KernelMeasurement:
        path = f"kernels[{index}]"
        data = _mapping(raw, path)
        canonical_stage = _string(data, "canonical_stage", path)
        if canonical_stage not in CANONICAL_STAGES:
            raise ProfileFormatError(f"{path}.canonical_stage is not recognized: {canonical_stage}")
        layer_type = _string(data, "layer_type", path)
        if layer_type not in {"mamba", "moe", "attention", "model", "other"}:
            raise ProfileFormatError(f"{path}.layer_type is not recognized: {layer_type}")
        return cls(
            canonical_stage=canonical_stage,
            layer_type=layer_type,
            kernel_name=_string(data, "kernel_name", path),
            calls=int(_number(data, "calls", path, integer=True, minimum=1)),
            total_time_us=float(_number(data, "total_time_us", path)),
            dram_read_bytes=int(_number(data, "dram_read_bytes", path, integer=True)),
            dram_write_bytes=int(_number(data, "dram_write_bytes", path, integer=True)),
        )


@dataclass(frozen=True)
class ProfileScenario:
    phase: InferencePhase
    batch_size: int
    input_sequence_length: int
    context_length: int
    generated_tokens: int
    warmup_iterations: int
    measured_iterations: int
    ttft_us: float
    mean_token_latency_us: float
    kernels: tuple[KernelMeasurement, ...]

    @classmethod
    def from_dict(cls, raw: Any, index: int) -> ProfileScenario:
        path = f"scenarios[{index}]"
        data = _mapping(raw, path)
        try:
            phase = InferencePhase(_string(data, "phase", path))
        except ValueError as error:
            raise ProfileFormatError(f"{path}.phase must be prefill or decode") from error
        kernels_raw = data.get("kernels")
        if not isinstance(kernels_raw, list) or not kernels_raw:
            raise ProfileFormatError(f"{path}.kernels must be a non-empty list")
        return cls(
            phase=phase,
            batch_size=int(_number(data, "batch_size", path, integer=True, minimum=1)),
            input_sequence_length=int(_number(data, "input_sequence_length", path, integer=True)),
            context_length=int(_number(data, "context_length", path, integer=True)),
            generated_tokens=int(_number(data, "generated_tokens", path, integer=True)),
            warmup_iterations=int(_number(data, "warmup_iterations", path, integer=True)),
            measured_iterations=int(_number(data, "measured_iterations", path, integer=True, minimum=1)),
            ttft_us=float(_number(data, "ttft_us", path)),
            mean_token_latency_us=float(_number(data, "mean_token_latency_us", path)),
            kernels=tuple(KernelMeasurement.from_dict(item, kernel_idx) for kernel_idx, item in enumerate(kernels_raw)),
        )

    def aggregate_stages(self) -> dict[str, dict[str, float | int]]:
        totals: dict[str, dict[str, float | int]] = {}
        for kernel in self.kernels:
            stage = totals.setdefault(
                kernel.canonical_stage,
                {"calls": 0, "total_time_us": 0.0, "dram_read_bytes": 0, "dram_write_bytes": 0},
            )
            stage["calls"] += kernel.calls
            stage["total_time_us"] += kernel.total_time_us
            stage["dram_read_bytes"] += kernel.dram_read_bytes
            stage["dram_write_bytes"] += kernel.dram_write_bytes
        return totals


@dataclass(frozen=True)
class GpuProfile:
    schema_version: int
    model_id: str
    model_revision: str
    environment: ProfileEnvironment
    scenarios: tuple[ProfileScenario, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> GpuProfile:
        data = _mapping(raw, "profile")
        schema_version = int(_number(data, "schema_version", "profile", integer=True))
        if schema_version != 1:
            raise ProfileFormatError(f"unsupported schema_version {schema_version}; expected 1")
        scenarios_raw = data.get("scenarios")
        if not isinstance(scenarios_raw, list) or not scenarios_raw:
            raise ProfileFormatError("profile.scenarios must be a non-empty list")
        return cls(
            schema_version=schema_version,
            model_id=_string(data, "model_id", "profile"),
            model_revision=_string(data, "model_revision", "profile"),
            environment=ProfileEnvironment.from_dict(data.get("environment")),
            scenarios=tuple(ProfileScenario.from_dict(item, index) for index, item in enumerate(scenarios_raw)),
        )


def load_gpu_profile(path: Path) -> GpuProfile:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileFormatError(f"cannot read profile {path}: {error}") from error
    return GpuProfile.from_dict(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and summarize a Nemotron 3 GPU profile")
    parser.add_argument("profile", type=Path)
    args = parser.parse_args(argv)
    profile = load_gpu_profile(args.profile)
    print(f"valid profile: {profile.model_id}@{profile.model_revision} on {profile.environment.gpu_model}")
    for scenario in profile.scenarios:
        totals = scenario.aggregate_stages()
        kernel_time = sum(float(stage["total_time_us"]) for stage in totals.values())
        dram_bytes = sum(int(stage["dram_read_bytes"]) + int(stage["dram_write_bytes"]) for stage in totals.values())
        print(
            f"{scenario.phase}: batch={scenario.batch_size}, context={scenario.context_length}, "
            f"kernel_time={kernel_time:.1f} us, DRAM={dram_bytes / (1024 * 1024):.2f} MiB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
