"""Schema and expansion for the formal RunPod benchmark manifest."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import sha256_json


SCHEMA_VERSION = "runpod-serving-v1"


@dataclass(frozen=True)
class ContextProfile:
    name: str
    max_model_len: int
    rope_scaling: dict[str, Any] | None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    revision: str | None
    dtype: str
    quantization_candidates: tuple[str, ...]
    context_profiles: dict[str, ContextProfile]
    workload_context_profiles: dict[str, str]


@dataclass(frozen=True)
class WorkloadSpec:
    name: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class BenchmarkPoint:
    point_id: str
    kind: str
    measurement_stage: str
    required_success: bool
    model_name: str
    model_id: str
    revision: str | None
    dtype: str
    context_profile: str
    max_model_len: int
    rope_scaling: dict[str, Any] | None
    workload_name: str
    input_tokens: int
    output_tokens: int
    tensor_parallel_size: int
    local_batch_size: int
    gpu_ids: tuple[int, ...]
    warmup_repetitions: int
    warmup_output_tokens: int
    repetitions: int
    token_seed: int
    sampling_hz: float
    idle_baseline_seconds: float
    engine_options: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.as_dict())

    def engine_key(self, *, revision: str, quantization: str) -> str:
        return sha256_json(
            {
                "model_id": self.model_id,
                "revision": revision,
                "dtype": self.dtype,
                "quantization": quantization,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_ids": self.gpu_ids,
                "max_model_len": self.max_model_len,
                "rope_scaling": self.rope_scaling,
                "engine_options": self.engine_options,
            }
        )


@dataclass(frozen=True)
class BenchmarkManifest:
    source_path: Path
    campaign: str
    defaults: dict[str, Any]
    models: dict[str, ModelSpec]
    workloads: dict[str, WorkloadSpec]
    formal_points: tuple[BenchmarkPoint, ...]
    preflight_points: tuple[BenchmarkPoint, ...]
    raw: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.raw)

    def points(self, *, include_preflight: bool = False) -> tuple[BenchmarkPoint, ...]:
        if include_preflight:
            return self.formal_points + self.preflight_points
        return self.formal_points

    def point_by_id(self, point_id: str) -> BenchmarkPoint:
        matches = [point for point in self.points(include_preflight=True) if point.point_id == point_id]
        if len(matches) != 1:
            raise KeyError(f"unknown or duplicate benchmark point: {point_id}")
        return matches[0]


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _positive_float(value: Any, label: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _parse_models(raw: dict[str, Any]) -> dict[str, ModelSpec]:
    models: dict[str, ModelSpec] = {}
    for name, item_value in _require_mapping(raw.get("models"), "models").items():
        item = _require_mapping(item_value, f"models.{name}")
        profiles: dict[str, ContextProfile] = {}
        for profile_name, profile_value in _require_mapping(
            item.get("context_profiles"), f"models.{name}.context_profiles"
        ).items():
            profile = _require_mapping(profile_value, f"context profile {profile_name}")
            rope_scaling = profile.get("rope_scaling")
            if rope_scaling is not None:
                rope_scaling = dict(_require_mapping(rope_scaling, "rope_scaling"))
            profiles[profile_name] = ContextProfile(
                name=profile_name,
                max_model_len=_positive_int(profile.get("max_model_len"), "max_model_len"),
                rope_scaling=rope_scaling,
            )
        candidates = tuple(str(candidate) for candidate in item.get("quantization_candidates", ()))
        if not candidates:
            raise ValueError(f"models.{name}.quantization_candidates must not be empty")
        workload_profiles = {
            str(workload): str(profile)
            for workload, profile in _require_mapping(
                item.get("workload_context_profiles"),
                f"models.{name}.workload_context_profiles",
            ).items()
        }
        unknown_profiles = set(workload_profiles.values()) - set(profiles)
        if unknown_profiles:
            raise ValueError(f"model {name} references unknown context profiles: {sorted(unknown_profiles)}")
        models[name] = ModelSpec(
            name=name,
            model_id=str(item["model_id"]),
            revision=None if item.get("revision") in {None, "", "resolve"} else str(item["revision"]),
            dtype=str(item.get("dtype", "half")),
            quantization_candidates=candidates,
            context_profiles=profiles,
            workload_context_profiles=workload_profiles,
        )
    return models


def _parse_workloads(raw: dict[str, Any]) -> dict[str, WorkloadSpec]:
    workloads: dict[str, WorkloadSpec] = {}
    for name, item_value in _require_mapping(raw.get("workloads"), "workloads").items():
        item = _require_mapping(item_value, f"workloads.{name}")
        workloads[name] = WorkloadSpec(
            name=name,
            input_tokens=_positive_int(item.get("input_tokens"), "input_tokens"),
            output_tokens=_positive_int(item.get("output_tokens"), "output_tokens"),
        )
    return workloads


def _point(
    *,
    kind: str,
    point_id: str,
    model: ModelSpec,
    workload_name: str,
    input_tokens: int,
    output_tokens: int,
    context_profile_name: str,
    tensor_parallel_size: int,
    local_batch_size: int,
    gpu_ids: tuple[int, ...],
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    required_success: bool = True,
) -> BenchmarkPoint:
    overrides = overrides or {}
    context = model.context_profiles[context_profile_name]
    if input_tokens + output_tokens > context.max_model_len:
        raise ValueError(
            f"{point_id}: input+output exceeds max_model_len "
            f"({input_tokens}+{output_tokens}>{context.max_model_len})"
        )
    if tensor_parallel_size != len(gpu_ids):
        raise ValueError(f"{point_id}: gpu_ids length must equal tensor_parallel_size")
    if len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
        raise ValueError(f"{point_id}: gpu_ids must be unique non-negative indices")
    engine_options = dict(_require_mapping(defaults.get("engine_options", {}), "defaults.engine_options"))
    engine_options.update(_require_mapping(overrides.get("engine_options", {}), "engine_options"))
    return BenchmarkPoint(
        point_id=point_id,
        kind=kind,
        measurement_stage=kind,
        required_success=required_success,
        model_name=model.name,
        model_id=model.model_id,
        revision=model.revision,
        dtype=model.dtype,
        context_profile=context_profile_name,
        max_model_len=context.max_model_len,
        rope_scaling=context.rope_scaling,
        workload_name=workload_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tensor_parallel_size=tensor_parallel_size,
        local_batch_size=local_batch_size,
        gpu_ids=gpu_ids,
        warmup_repetitions=_nonnegative_int(
            overrides.get("warmup_repetitions", defaults.get("warmup_repetitions", 1)),
            "warmup_repetitions",
        ),
        warmup_output_tokens=_positive_int(
            overrides.get("warmup_output_tokens", defaults.get("warmup_output_tokens", 32)),
            "warmup_output_tokens",
        ),
        repetitions=_positive_int(overrides.get("repetitions", defaults.get("repetitions", 3)), "repetitions"),
        token_seed=int(overrides.get("token_seed", defaults.get("token_seed", 20260808))),
        sampling_hz=_positive_float(
            overrides.get("sampling_hz", defaults.get("sampling_hz", 20.0)), "sampling_hz"
        ),
        idle_baseline_seconds=_positive_float(
            overrides.get("idle_baseline_seconds", defaults.get("idle_baseline_seconds", 2.0)),
            "idle_baseline_seconds",
        ),
        engine_options=engine_options,
    )


def _expand_formal_points(
    raw: dict[str, Any],
    *,
    models: dict[str, ModelSpec],
    workloads: dict[str, WorkloadSpec],
    defaults: dict[str, Any],
) -> tuple[BenchmarkPoint, ...]:
    points: list[BenchmarkPoint] = []
    for row_index, row_value in enumerate(raw.get("formal_matrix", ())):
        row = _require_mapping(row_value, f"formal_matrix[{row_index}]")
        model = models[str(row["model"])]
        tp = _positive_int(row.get("tensor_parallel_size"), "tensor_parallel_size")
        gpu_ids = tuple(int(index) for index in row.get("gpu_ids", range(tp)))
        batches = tuple(_positive_int(batch, "local_batch_size") for batch in row.get("local_batches", ()))
        if not batches:
            raise ValueError(f"formal_matrix[{row_index}].local_batches must not be empty")
        selected_workloads = tuple(str(name) for name in row.get("workloads", workloads))
        for workload_name in selected_workloads:
            workload = workloads[workload_name]
            profile_name = model.workload_context_profiles[workload_name]
            for batch in batches:
                point_id = f"{model.name}.{workload_name}.tp{tp}.b{batch}"
                points.append(
                    _point(
                        kind="formal",
                        point_id=point_id,
                        model=model,
                        workload_name=workload_name,
                        input_tokens=workload.input_tokens,
                        output_tokens=workload.output_tokens,
                        context_profile_name=profile_name,
                        tensor_parallel_size=tp,
                        local_batch_size=batch,
                        gpu_ids=gpu_ids,
                        defaults=defaults,
                        overrides=row,
                        required_success=True,
                    )
                )
    return tuple(points)


def _expand_preflight_points(
    raw: dict[str, Any],
    *,
    models: dict[str, ModelSpec],
    defaults: dict[str, Any],
) -> tuple[BenchmarkPoint, ...]:
    points: list[BenchmarkPoint] = []
    for row_index, row_value in enumerate(raw.get("preflight_points", ())):
        row = _require_mapping(row_value, f"preflight_points[{row_index}]")
        model = models[str(row["model"])]
        tp = _positive_int(row.get("tensor_parallel_size"), "tensor_parallel_size")
        profile_name = str(row["context_profile"])
        points.append(
            _point(
                kind="preflight",
                point_id=str(row["id"]),
                model=model,
                workload_name=str(row.get("workload_name", "preflight")),
                input_tokens=_positive_int(row.get("input_tokens"), "input_tokens"),
                output_tokens=_positive_int(row.get("output_tokens"), "output_tokens"),
                context_profile_name=profile_name,
                tensor_parallel_size=tp,
                local_batch_size=_positive_int(row.get("local_batch_size"), "local_batch_size"),
                gpu_ids=tuple(int(index) for index in row.get("gpu_ids", range(tp))),
                defaults=defaults,
                overrides={**row, "warmup_output_tokens": row.get("output_tokens"), "repetitions": 1},
                required_success=bool(row.get("required_success", True)),
            )
        )
    return tuple(points)


def load_manifest(path: str | Path) -> BenchmarkManifest:
    source_path = Path(path).resolve()
    raw_value = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    raw = _require_mapping(raw_value, "manifest")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {raw.get('schema_version')!r}")
    defaults = dict(_require_mapping(raw.get("defaults", {}), "defaults"))
    models = _parse_models(raw)
    workloads = _parse_workloads(raw)
    for model in models.values():
        missing = set(workloads) - set(model.workload_context_profiles)
        if missing:
            raise ValueError(f"model {model.name} has no context profile for workloads: {sorted(missing)}")
    formal_points = _expand_formal_points(raw, models=models, workloads=workloads, defaults=defaults)
    preflight_points = _expand_preflight_points(raw, models=models, defaults=defaults)
    all_ids = [point.point_id for point in formal_points + preflight_points]
    duplicates = sorted({point_id for point_id in all_ids if all_ids.count(point_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate point IDs: {duplicates}")
    return BenchmarkManifest(
        source_path=source_path,
        campaign=str(raw["campaign"]),
        defaults=defaults,
        models=models,
        workloads=workloads,
        formal_points=formal_points,
        preflight_points=preflight_points,
        raw=raw,
    )
