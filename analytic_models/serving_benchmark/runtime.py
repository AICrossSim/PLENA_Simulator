"""Runtime fingerprints shared by the parent runner and GPU workers."""

from __future__ import annotations

from typing import Any

from .io import sha256_json
from .manifest import BenchmarkPoint


def runtime_point_fingerprint(
    point: BenchmarkPoint,
    *,
    revision: str,
    quantization: str,
    environment_hash: str,
) -> str:
    return sha256_json(
        {
            "point": point.as_dict(),
            "resolved_revision": revision,
            "quantization": quantization,
            "environment_hash": environment_hash,
        }
    )


def point_runtime_record(
    point: BenchmarkPoint,
    *,
    revision: str,
    quantization: str,
    environment_hash: str,
) -> dict[str, Any]:
    return {
        "point_id": point.point_id,
        "point_fingerprint": runtime_point_fingerprint(
            point,
            revision=revision,
            quantization=quantization,
            environment_hash=environment_hash,
        ),
        "resolved_revision": revision,
        "quantization": quantization,
        "environment_hash": environment_hash,
    }
