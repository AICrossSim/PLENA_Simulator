"""Build immutable decode timing evidence from matched cycle anchors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

try:
    from .decode_timing import (
        CycleAnchor,
        EMULATOR_SERIALIZED,
        EMULATOR_TIMING_PROVENANCE_ROLES,
        REQUIRED_TIMING_PROVENANCE_ROLES,
        TIMING_EVIDENCE_MODES,
        TimingEvidence,
    )
except ImportError:
    from decode_timing import (
        CycleAnchor,
        EMULATOR_SERIALIZED,
        EMULATOR_TIMING_PROVENANCE_ROLES,
        REQUIRED_TIMING_PROVENANCE_ROLES,
        TIMING_EVIDENCE_MODES,
        TimingEvidence,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_path(raw_path: object, anchors_path: Path, field: str) -> Path:
    if raw_path in (None, ""):
        raise ValueError(f"{field} must name a raw evidence file")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = anchors_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} raw evidence is missing: {path}")
    return path


def _validate_geometry_manifest(path: Path, row: dict[str, object]) -> None:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"geometry_path must be a JSON manifest: {path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("geometry manifest must be a JSON object")
    for name in ("mlen", "blen", "hlen", "vlen"):
        if name not in manifest:
            raise ValueError(f"geometry manifest misses {name}: {path}")
        if int(row[name]) != int(manifest[name]):
            raise ValueError(f"anchor {name}={row[name]} does not match geometry manifest value {manifest[name]}")


def _load_anchors(
    path: Path,
    *,
    compiler_sha256: str,
    emulator_tier: bool,
) -> tuple[CycleAnchor, ...]:
    with path.open(newline="") as source:
        rows = tuple(csv.DictReader(source))
    required = {
        "anchor_id",
        "anchor_kind",
        "analytical_cycles",
        "analytical_compute_cycles",
        "analytical_memory_cycles",
        "cache_position",
        "batch",
        "physical_hbm_bytes",
        "emulator_cycles",
        "rtl_cycles",
        "mlen",
        "blen",
        "hlen",
        "vlen",
        "geometry_path",
        "precision_path",
        "asm_path",
        "analytical_trace_path",
        "emulator_trace_path",
        "rtl_trace_path",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            "anchor CSV misses timing decomposition, geometry, precision, "
            "compiler/ASM, trace-identity, cache, batch, HBM-byte, or "
            "cross-stack cycle columns"
        )
    anchors = []
    derived_fields = {
        "geometry_sha256": "geometry_path",
        "precision_sha256": "precision_path",
        "asm_sha256": "asm_path",
        "analytical_trace_sha256": "analytical_trace_path",
        "emulator_trace_sha256": "emulator_trace_path",
        "rtl_trace_sha256": "rtl_trace_path",
    }
    for row in rows:
        values = dict(row)
        values["compiler_sha256"] = compiler_sha256
        if emulator_tier:
            if values.get("rtl_cycles") not in (None, "") or values.get("rtl_trace_path") not in (None, ""):
                raise ValueError("emulator-tier anchor rows must leave the RTL columns empty")
            derived = {
                field: source
                for field, source in derived_fields.items()
                if field != "rtl_trace_sha256"
            }
        else:
            derived = derived_fields
        geometry_path = _evidence_path(values.get("geometry_path"), path, "geometry_path")
        _validate_geometry_manifest(geometry_path, values)
        for digest_field, path_field in derived.items():
            evidence_path = (
                geometry_path
                if path_field == "geometry_path"
                else _evidence_path(values.get(path_field), path, path_field)
            )
            values[digest_field] = _sha256(evidence_path)
        anchors.append(CycleAnchor.from_dict(values))
    return tuple(anchors)


def _provenance(
    values: list[str],
    anchors_path: Path,
    *,
    emulator_tier: bool,
) -> tuple[tuple[str, str], ...]:
    entries = {"anchors": _sha256(anchors_path)}
    for value in values:
        if "=" not in value:
            raise ValueError("provenance must use name=path")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        if not name or name in entries:
            raise ValueError("provenance names must be non-empty and unique")
        entries[name] = _sha256(path)
    required = EMULATOR_TIMING_PROVENANCE_ROLES if emulator_tier else REQUIRED_TIMING_PROVENANCE_ROLES
    if emulator_tier and "rtl" in entries:
        raise ValueError("emulator-tier timing evidence must not carry an RTL provenance role")
    missing = tuple(role for role in required if role not in entries)
    if missing:
        raise ValueError("timing evidence requires provenance roles: " + ", ".join(missing))
    return tuple(sorted(entries.items()))


def _atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            if path.read_text() != payload:
                raise FileExistsError(f"refusing to replace different timing evidence: {path}")
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=TIMING_EVIDENCE_MODES, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--provenance", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    emulator_tier = args.mode == EMULATOR_SERIALIZED
    provenance = _provenance(
        args.provenance,
        args.anchors,
        emulator_tier=emulator_tier,
    )
    evidence = TimingEvidence(
        mode=args.mode,
        anchors=_load_anchors(
            args.anchors,
            compiler_sha256=dict(provenance)["compiler"],
            emulator_tier=emulator_tier,
        ),
        provenance_hashes=provenance,
    )
    _atomic_write(args.out, evidence.to_dict())
    print(
        f"{args.out}: passed={evidence.passed} "
        f"evidence_tier={evidence.evidence_tier} "
        f"anchor_max_error={evidence.anchor_max_error:.4f} "
        f"analytical_mape={evidence.analytical_mape:.4f}"
    )
    return 0 if evidence.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
