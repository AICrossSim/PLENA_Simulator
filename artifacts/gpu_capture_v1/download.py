#!/usr/bin/env python3
"""Download and verify the GPU capture assets named in catalog.json."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import urllib.request


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, record):
    if path.stat().st_size != record['archive_bytes']:
        raise ValueError(f'Archive size mismatch: {path.name}')
    if sha256(path) != record['archive_sha256']:
        raise ValueError(f'Archive checksum mismatch: {path.name}')


def extract(path, destination):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe archive path: {member.name}')
            if not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsupported archive member: {member.name}')
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'Archive path escapes destination: {member.name}')
        for member in members:
            target = destination.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('wb') as output:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
    # Every bundle supplies checksums for its published members.
    for checksums in destination.glob('*/SHA256SUMS'):
        for line in checksums.read_text().splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(None, 1)
            relative = relative.lstrip('* ')
            target = checksums.parent / relative
            if not target.resolve().is_relative_to(checksums.parent.resolve()):
                raise ValueError(f'Unsafe checksum path: {relative}')
            if sha256(target) != expected:
                raise ValueError(f'Published member checksum mismatch: {target}')


def main():
    catalog = json.loads(Path(__file__).with_name('catalog.json').read_text())
    entries = {entry['id']: entry for entry in catalog['captures']}
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--campaign', choices=entries)
    choice.add_argument('--all', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('gpu-captures'))
    parser.add_argument('--verify-only', action='store_true', help='Verify archives already present; no network.')
    parser.add_argument('--extract', action='store_true', help='Extract verified archives and verify member checksums.')
    args = parser.parse_args()
    selected = list(entries.values()) if args.all else ([entries[args.campaign]] if args.campaign else [])
    if not selected:
        for name, entry in entries.items():
            print(f"{name}: {entry['archive_bytes'] / 1024**2:.1f} MiB; {entry['scope']}")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in selected:
        target = args.output / entry['asset_name']
        if not target.exists() and not args.verify_only:
            partial = target.with_suffix(target.suffix + '.partial')
            request = urllib.request.Request(entry['download_url'], headers={'User-Agent': 'PLENA-GPU-evidence'})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    output.write(block)
            verify(partial, entry)
            partial.replace(target)
        verify(target, entry)
        print(f"Verified {entry['id']}")
        if args.extract:
            extract(target, args.output)


if __name__ == '__main__':
    main()
