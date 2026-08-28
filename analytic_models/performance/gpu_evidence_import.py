"""Import compact, path-independent evidence from the local GPU archives.

The profiler reports themselves stay outside Git.  This importer verifies each
archive before extracting the small CSV/JSON contracts consumed by tests and
the performance-model documentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROFILE_DIR = Path(__file__).with_name("profiles")


@dataclass(frozen=True)
class Archive:
    name: str
    size_bytes: int
    sha256: str


RTX5090 = Archive(
    "NEMOTRON3_NANO_5090.tar.gz",
    17_055_133,
    "e0a33f64c6366c557afb44eb77bd788cc6623f53357c1556578991a5bba08227",
)
B200_FORMAL = Archive(
    "kda_nemotron_nvfp4_campaign_gpu3_attempt2_COMPLETE_20260819T043954Z.tar.gz",
    123_292_703,
    "eac1d2637ff82286365070a40e21e260d222b53b0c3b3b28172f5ef925ec15c9",
)
B200_KDA_STAGE2 = Archive(
    "plena-kda-stage2-b200-20260817T030439Z.tar.gz",
    18_843_087,
    "6fada27aeaff8b359ad540d5e9bcf5655331d64cc2d27094b70da2d6bf55906c",
)
B200_SUPPLEMENTAL = Archive(
    "plena_precision_kimi_components_20260819T053646Z_COMPLETE.tar.gz",
    163_560_901,
    "1bb93c19e558081c1fba25d60fcf2e0e7e2e0eb7ba3b768405773363c96790f0",
)


RTX5090_MEMBERS = {
    "profiling_notes.md": "results/NEMOTRON3_NANO_5090/profiling_notes.md",
    "mamba_layer_latency.csv": "results/NEMOTRON3_NANO_5090/mamba_layer_latency.csv",
    "mamba_layer_latency_meta.json": ("results/NEMOTRON3_NANO_5090/mamba_layer_latency_meta.json"),
    "ncu_mamba_decode_b1.csv": "results/NEMOTRON3_NANO_5090/ncu_mamba_decode_b1.csv",
    "ncu_mamba_decode_b8.csv": "results/NEMOTRON3_NANO_5090/ncu_mamba_decode_b8.csv",
    "ncu_mamba_prefill.csv": "results/NEMOTRON3_NANO_5090/ncu_mamba_prefill.csv",
    "nvtx_stage_cuda_kernel_summary.json": (
        "results/NEMOTRON3_NANO_5090/nsys_nvtx_stage_summary/nvtx_stage_cuda_kernel_summary.json"
    ),
}

_SUPPLEMENTAL_ROOT = "plena_precision_kimi_components_20260819T053646Z"
SUPPLEMENTAL_MEMBERS = {
    "kimi_component_latency.csv": (f"{_SUPPLEMENTAL_ROOT}/kimi_components/summary/latency_summary.csv"),
    "kimi_component_ncu.csv": (f"{_SUPPLEMENTAL_ROOT}/kimi_components/summary/ncu_summary.csv"),
    "kimi_component_nsys.csv": (f"{_SUPPLEMENTAL_ROOT}/kimi_components/summary/nsys_summary.csv"),
    "kimi_component_parity.json": (f"{_SUPPLEMENTAL_ROOT}/kimi_components/parity_attempt2/parity.json"),
    "mamba_precision.csv": (f"{_SUPPLEMENTAL_ROOT}/mamba_precision/summary/mamba_precision_aggregate.csv"),
    "nemotron_prefill_mamba_stages.csv": (f"{_SUPPLEMENTAL_ROOT}/nemotron_traffic_gap/prefill_mamba_traffic_gap.csv"),
    "source_validation.json": f"{_SUPPLEMENTAL_ROOT}/validation.json",
}
SUPPLEMENTAL_MANIFEST = f"{_SUPPLEMENTAL_ROOT}/manifest.json"


class GpuEvidenceImportError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive(path: Path, expected: Archive) -> None:
    if not path.is_file():
        raise GpuEvidenceImportError(f"missing archive: {path}")
    if path.stat().st_size != expected.size_bytes:
        raise GpuEvidenceImportError(f"unexpected size for {expected.name}")
    if _sha256(path) != expected.sha256:
        raise GpuEvidenceImportError(f"SHA256 mismatch for {expected.name}")


def _read_member(archive: tarfile.TarFile, member: str) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise GpuEvidenceImportError(f"archive member is not a file: {member}")
    return extracted.read()


def _without_path_column(payload: bytes, column: str) -> bytes:
    source = io.StringIO(payload.decode())
    reader = csv.DictReader(source)
    if reader.fieldnames is None or column not in reader.fieldnames:
        raise GpuEvidenceImportError(f"CSV is missing provenance column {column}")
    fields = [field for field in reader.fieldnames if field != column]
    destination = io.StringIO()
    writer = csv.DictWriter(destination, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in reader:
        writer.writerow({field: row[field] for field in fields})
    return destination.getvalue().encode()


def _normalize_text(payload: bytes) -> bytes:
    """Keep checked-in text deterministic across profiler host platforms."""
    return payload.decode().replace("\r\n", "\n").replace("\r", "\n").encode()


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _import_5090(artifact_root: Path, output: Path) -> dict[str, dict[str, str]]:
    archive_path = artifact_root / RTX5090.name
    _verify_archive(archive_path, RTX5090)
    imported = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for target, member in RTX5090_MEMBERS.items():
            payload = _read_member(archive, member)
            if target == "mamba_layer_latency_meta.json":
                document = json.loads(payload)
                document.pop("official_source_dir", None)
                payload = (json.dumps(document, indent=2) + "\n").encode()
            else:
                payload = _normalize_text(payload)
            relative = Path("rtx5090_nemotron_mamba") / target
            imported[relative.as_posix()] = {
                "archive_member": member,
                "sha256": _write(output / relative, payload),
            }
    return imported


def _import_supplemental(artifact_root: Path, output: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    archive_path = artifact_root / B200_SUPPLEMENTAL.name
    _verify_archive(archive_path, B200_SUPPLEMENTAL)
    imported = {}
    drop_columns = {
        "kimi_component_latency.csv": "source",
        "kimi_component_ncu.csv": "result_dir",
        "kimi_component_nsys.csv": "report",
    }
    with tarfile.open(archive_path, "r:gz") as archive:
        source_manifest = json.loads(_read_member(archive, SUPPLEMENTAL_MANIFEST))
        for target, member in SUPPLEMENTAL_MEMBERS.items():
            payload = _read_member(archive, member)
            if target in drop_columns:
                payload = _without_path_column(payload, drop_columns[target])
            else:
                payload = _normalize_text(payload)
            relative = Path("b200_supplemental") / target
            imported[relative.as_posix()] = {
                "archive_member": member,
                "sha256": _write(output / relative, payload),
            }
    metadata = {
        "gpu": source_manifest["gpu"],
        "revisions": source_manifest["revisions"],
        "counts": source_manifest["counts"],
        "classifications": source_manifest["classifications"],
        "limitations": source_manifest["limitations"],
    }
    return imported, metadata


def import_evidence(artifact_root: Path, output: Path = PROFILE_DIR) -> dict[str, Any]:
    # The formal and Stage2 archives are consumed by b200_campaign_raw.py and
    # crosscheck_local_kda_stage2(), respectively. Verify them here as part of
    # the single reproducibility entry point even though this importer does not
    # duplicate those parsers.
    _verify_archive(artifact_root / B200_FORMAL.name, B200_FORMAL)
    _verify_archive(artifact_root / B200_KDA_STAGE2.name, B200_KDA_STAGE2)
    imported = _import_5090(artifact_root, output)
    supplemental, metadata = _import_supplemental(artifact_root, output)
    imported.update(supplemental)

    pinned = {
        "b200_kda_nemotron_campaign_complete.json": (
            "fa8cffba3b1dc47ebb202bb410d74b2ffe7af468b62fd05056733a1050cf4871"
        ),
        "nemotron3_decode_routing_trace.json": ("ba8533831bd88cf209aaf4b8e4d2f927889358c51699eb081ff183c69844d0cc"),
    }
    for relative, expected_hash in pinned.items():
        path = output / relative
        if _sha256(path) != expected_hash:
            raise GpuEvidenceImportError(f"pinned formal profile changed: {relative}")
        imported[relative] = {
            "archive_member": "generated by b200_campaign_raw.py from the formal archive",
            "sha256": expected_hash,
        }

    archives = {
        name: {
            "name": archive.name,
            "size_bytes": archive.size_bytes,
            "sha256": archive.sha256,
            "role": role,
        }
        for name, archive, role in (
            ("rtx5090_mamba", RTX5090, "official-shape random-weight Mamba mixer"),
            ("b200_formal", B200_FORMAL, "full Nemotron checkpoint and KDA component"),
            ("b200_kda_stage2", B200_KDA_STAGE2, "independent KDA core cross-check"),
            (
                "b200_supplemental",
                B200_SUPPLEMENTAL,
                "Mamba precision plus Kimi MLA/LatentMoE components",
            ),
        )
    }
    manifest = {
        "schema_version": 1,
        "archives": archives,
        "supplemental_source": metadata,
        "imported_files": dict(sorted(imported.items())),
        "evidence_boundaries": {
            "gpu_workload_and_baseline": "measured",
            "plena_cycles": "not calibrated by these files",
            "plena_rtl_frequency_area_power": "not measured",
            "language_quality_after_state_quantization": "not measured",
        },
    }
    _write(output / "gpu_sources.json", (json.dumps(manifest, indent=2) + "\n").encode())
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path, default=PROFILE_DIR)
    args = parser.parse_args(argv)
    manifest = import_evidence(args.artifact_root.resolve(), args.output.resolve())
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
