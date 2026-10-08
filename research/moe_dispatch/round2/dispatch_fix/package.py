"""Archive large evidence without changing any byte of the original records."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil


DIRECTORY = Path(__file__).resolve().parent
RAW_NAMES = ("baseline_per_window.json", "per_window.json")


def digest_stream(stream):
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def digest_file(path):
    with path.open("rb") as stream:
        return digest_stream(stream)


def main():
    archives = []
    for name in RAW_NAMES:
        original = DIRECTORY / name
        archive = DIRECTORY / (name + ".gz")
        if original.exists():
            original_sha, original_bytes = digest_file(original)
            with original.open("rb") as source, archive.open("wb") as target:
                with gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as packed:
                    shutil.copyfileobj(source, packed, length=1024 * 1024)
        else:
            # The archive is the complete numerical evidence in a fresh checkout.
            with gzip.open(archive, "rb") as source:
                original_sha, original_bytes = digest_stream(source)
        with gzip.open(archive, "rb") as source:
            restored_sha, restored_bytes = digest_stream(source)
        assert (restored_sha, restored_bytes) == (original_sha, original_bytes)
        archive_sha, archive_bytes = digest_file(archive)
        archives.append({"original": name, "original_sha256": original_sha,
                         "original_bytes": original_bytes, "archive": archive.name,
                         "archive_sha256": archive_sha, "archive_bytes": archive_bytes,
                         "roundtrip_exact": True, "gzip_mtime": 0})
    (DIRECTORY / "RAW_ARCHIVES.json").write_text(json.dumps(
        {"created_utc": datetime.now(timezone.utc).isoformat(), "archives": archives},
        indent=2, sort_keys=True) + "\n")
    with (DIRECTORY / "PROVENANCE.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file", "bytes", "sha256"))
        writer.writeheader()
        for path in sorted(DIRECTORY.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(DIRECTORY).as_posix()
            if relative in {*RAW_NAMES, "PROVENANCE.csv"}:
                continue
            # package execution receipts are still being closed by their wrapper.
            if relative.startswith("executions/package_"):
                continue
            digest, size = digest_file(path)
            writer.writerow({"file": relative, "bytes": size, "sha256": digest})
    print(json.dumps({"archives": archives, "provenance": "PROVENANCE.csv"}, indent=2))


if __name__ == "__main__":
    main()
