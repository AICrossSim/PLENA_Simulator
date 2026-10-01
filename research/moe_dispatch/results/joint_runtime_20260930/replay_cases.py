#!/usr/bin/env python3
"""Replay published cases twice and compare complete original report bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    root = Path(__file__).resolve().parent
    for name, expected in json.loads((root / 'FILES_SHA256.json').read_text()).items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Evidence hash mismatch: {name}')
    cases = sorted((root / 'replay_cases').iterdir())
    if len(cases) != 6:
        raise RuntimeError(f'Expected six cases, found {len(cases)}')
    with tempfile.TemporaryDirectory(prefix='plena-joint-replay-') as directory:
        for index, case in enumerate(cases):
            expected = (case / 'report_repeat1.json').read_bytes()
            if expected != (case / 'report_repeat2.json').read_bytes():
                raise RuntimeError(f'Original repeats differ: {case.name}')
            for repeat in (1, 2):
                output = Path(directory) / f'{index}-{repeat}.json'
                subprocess.run([str(binary), str(case / 'workload.json'),
                                str(case / 'config.json'), str(output)],
                               check=True, capture_output=True, timeout=600)
                if output.read_bytes() != expected:
                    raise RuntimeError(f'Replay mismatch: {case.name}, repeat {repeat}')
            print(f'PASS {case.name}', flush=True)
    print('Six cases, two repeats each: exact archived report equality.')


if __name__ == '__main__':
    main()
