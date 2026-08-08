"""Deterministic and atomic artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def completed_artifact_matches(path: Path, *, fingerprint: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except (OSError, ValueError, TypeError):
        return False
    return payload.get("status") == "complete" and payload.get("point_fingerprint") == fingerprint
