"""Validate raw calibration artifacts before fitting power coefficients."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

ARTIFACT_CATALOG_SCHEMA = "plena-power-artifact-catalog"
CONTEXT_ARTIFACT_KINDS = frozenset(
    {
        "constraints",
        "library_manifest",
        "rtl_source_manifest",
        "tool_log",
    }
)
POINT_ARTIFACT_KINDS = {
    "array": frozenset(
        {"dc_report", "saif", "decode_trace", "synthesis_log"}
    ),
    "vector": frozenset(
        {"dc_report", "saif", "decode_trace", "synthesis_log"}
    ),
    "selector": frozenset(
        {"dc_report", "saif", "decode_trace", "synthesis_log"}
    ),
    "fixed": frozenset({"dc_report", "synthesis_log"}),
    "chip_leakage": frozenset({"dc_report", "synthesis_log"}),
    "cycle": frozenset({"rtl_trace", "emulator_trace"}),
    "latency": frozenset({"measured_trace", "analytical_trace"}),
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_confined(base: Path, raw: object) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact paths must be confined relative paths")
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError("artifact path escapes the catalog directory") from exc
    return resolved


def _validate_artifact(raw: object, *, base: Path) -> tuple[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "kind",
        "path",
        "size_bytes",
        "sha256",
    }:
        raise ValueError("artifact fields differ from the catalog schema")
    kind = str(raw["kind"])
    if not kind:
        raise ValueError("artifact kind must be non-empty")
    path = _resolve_confined(base, raw["path"])
    if not path.is_file():
        raise FileNotFoundError(f"calibration artifact is missing: {path}")
    size = raw["size_bytes"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
    ):
        raise ValueError(f"calibration artifact size mismatch: {path}")
    digest = str(raw["sha256"])
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("artifact SHA-256 is invalid")
    if _sha256_file(path) != digest:
        raise ValueError(f"calibration artifact checksum mismatch: {path}")
    return kind, digest


def validate_artifact_catalog(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, tuple[str, ...]]:
    """Return the catalog hash and any coverage/binding failures."""

    source = Path(path).resolve()
    payload = source.read_bytes()
    raw = json.loads(payload)
    if not isinstance(raw, Mapping):
        raise TypeError("artifact catalog root must be an object")
    body = dict(raw)
    content_hash = str(body.pop("content_hash", ""))
    if hashlib.sha256(_canonical_bytes(body)).hexdigest() != content_hash:
        raise ValueError("artifact catalog content hash mismatch")
    if set(body) != {
        "schema_version",
        "context_artifacts",
        "records",
    }:
        raise ValueError("artifact catalog fields differ from the schema")
    if body["schema_version"] != ARTIFACT_CATALOG_SCHEMA:
        raise ValueError("unsupported artifact catalog schema")

    context = body["context_artifacts"]
    if not isinstance(context, list):
        raise TypeError("context_artifacts must be a list")
    context_kinds: dict[str, str] = {}
    for item in context:
        kind, digest = _validate_artifact(item, base=source.parent)
        if kind in context_kinds:
            raise ValueError(f"context artifacts repeat kind {kind}")
        context_kinds[kind] = digest
    failures: list[str] = []
    if set(context_kinds) != CONTEXT_ARTIFACT_KINDS:
        failures.append("artifact_catalog:context_coverage")

    records = body["records"]
    if not isinstance(records, list):
        raise TypeError("artifact catalog records must be a list")
    by_point: dict[str, Mapping[str, str]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "point_id",
            "artifacts",
        }:
            raise ValueError("catalog record fields differ from the schema")
        point_id = str(record["point_id"])
        if not point_id or point_id in by_point:
            raise ValueError("catalog point IDs must be non-empty and unique")
        artifacts = record["artifacts"]
        if not isinstance(artifacts, list):
            raise TypeError("catalog record artifacts must be a list")
        kinds: dict[str, str] = {}
        for artifact in artifacts:
            kind, digest = _validate_artifact(
                artifact,
                base=source.parent,
            )
            if kind in kinds:
                raise ValueError(
                    f"catalog point {point_id} repeats artifact kind {kind}"
                )
            kinds[kind] = digest
        by_point[point_id] = kinds

    expected_ids = {str(row.get("point_id", "")) for row in rows}
    if set(by_point) != expected_ids:
        failures.append("artifact_catalog:point_coverage")
    for row in rows:
        point_id = str(row.get("point_id", ""))
        component = str(row.get("component", ""))
        required = POINT_ARTIFACT_KINDS.get(component)
        if required is None:
            failures.append(f"artifact_catalog:component:{component}")
            continue
        artifacts = by_point.get(point_id, {})
        if set(artifacts) != required:
            failures.append(f"artifact_catalog:artifacts:{point_id}")
            continue
        if component in {"array", "vector", "selector"}:
            if artifacts["saif"] != str(row.get("saif_sha256", "")):
                failures.append(f"artifact_catalog:saif:{point_id}")
            if artifacts["decode_trace"] != str(
                row.get("decode_trace_sha256", "")
            ):
                failures.append(
                    f"artifact_catalog:decode_trace:{point_id}"
                )
    return hashlib.sha256(payload).hexdigest(), tuple(sorted(set(failures)))


__all__ = [
    "ARTIFACT_CATALOG_SCHEMA",
    "CONTEXT_ARTIFACT_KINDS",
    "POINT_ARTIFACT_KINDS",
    "validate_artifact_catalog",
]
