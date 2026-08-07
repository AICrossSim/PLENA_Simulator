"""Build the sealed full-model decode artifact set from exact points.

The producer seals one content-addressed family per distinct family key and
writes a records-empty artifact set; exact lowering/context records
instantiate lazily at consume time through
``FullModelDecodeArtifactSet.resolve_record``. Contexts are validated
without materialisation so planning stays cheap at full study scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

from compiler_trace_timing import (
    FullModelDecodeArtifactFamily,
    FullModelDecodeArtifactSet,
    full_model_decode_family_key,
    full_model_decode_generator_blockers,
    full_model_decode_lowering_key,
    full_model_decode_native_template_key,
    native_decode_compiler_source_sha256,
)


def _validated_context_bounds(contexts: object) -> tuple[int, int, int, int]:
    """Return (start, stop, step, count) without materialising the axis."""

    if isinstance(contexts, range):
        start, stop, step = contexts.start, contexts.stop, contexts.step
        if step < 1:
            raise ValueError("context ranges must have a positive step")
        if start < 1:
            raise ValueError("context tokens must be positive")
        if stop <= start:
            raise ValueError("context ranges must be non-empty")
        return start, stop, step, len(contexts)
    values = tuple(int(item) for item in contexts)  # type: ignore[arg-type]
    if not values:
        raise ValueError("context axes must be non-empty")
    if any(item < 1 for item in values):
        raise ValueError("context tokens must be positive")
    return min(values), max(values) + 1, 0, len(values)


def build_full_model_decode_artifact_set(
    point_contexts: Iterable[tuple[object, object]],
    destination: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Seal families for every exact point and write the lazy artifact set.

    ``point_contexts`` yields ``(point_descriptor, contexts)`` pairs. Every
    point must be free of native-generator blockers; a single blocked point
    fails the whole build before anything is written. The returned receipt
    carries the artifact identity and the deduplicated compile accounting.
    """

    compiler_source = native_decode_compiler_source_sha256()

    families: dict[str, FullModelDecodeArtifactFamily] = {}
    lowering_keys: set[str] = set()
    native_compile_keys: set[tuple[str, int]] = set()
    point_count = 0
    context_bounds: tuple[int, int, int, int] | None = None

    for point, contexts in point_contexts:
        point_count += 1
        bounds = _validated_context_bounds(contexts)
        if context_bounds is None:
            context_bounds = bounds
        elif bounds != context_bounds:
            raise ValueError("all points must share one exact context axis")

        blockers = full_model_decode_generator_blockers(point)
        if blockers:
            raise ValueError(
                "native generation is blocked for an exact point: "
                + ", ".join(sorted(blockers))
            )

        _, family_sha256, _ = full_model_decode_family_key(point)
        if family_sha256 not in families:
            families[family_sha256] = FullModelDecodeArtifactFamily.from_point_descriptor(
                point,
                compiler_source_sha256=compiler_source,
            )

        _, lowering_sha256, lowering = full_model_decode_lowering_key(point)
        lowering_keys.add(lowering_sha256)
        serving = lowering["serving"]
        if not isinstance(serving, Mapping):
            raise TypeError("exact point lowering serving section is malformed")
        _, template_sha256, _ = full_model_decode_native_template_key(point)
        native_compile_keys.add((template_sha256, int(serving["batch"])))

    if point_count == 0 or context_bounds is None:
        raise ValueError("artifact generation received no exact points")

    artifact_set = FullModelDecodeArtifactSet((), families=families.values())

    if not dry_run:
        artifact_set.write(destination)

    return {
        "artifact_set_id": artifact_set.artifact_set_id,
        "compiler_source_sha256": compiler_source,
        "family_count": len(families),
        "record_count": 0,
        "records_materialized": "lazy_at_consume",
        "native_compile_calls": len(native_compile_keys),
        "unique_lowering_keys": len(lowering_keys),
        "point_count": point_count,
        "context_start": context_bounds[0],
        "context_stop": context_bounds[1],
        "context_step": context_bounds[2],
        "context_count": context_bounds[3],
    }
