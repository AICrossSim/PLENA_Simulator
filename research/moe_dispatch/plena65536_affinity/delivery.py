"""Archive frozen A/B geometry results, sources, captured inputs and receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_repeat(output):
    import numpy as np
    output = Path(output)
    first = json.loads((output / 'FROZEN_SELECTION.json').read_text())
    repeat = json.loads((output / 'repeat_search/FROZEN_SELECTION.json').read_text())
    assert first['families'] == repeat['families'], 'Frozen design or parameter selection changed'
    checked = []
    for path in sorted(output.glob('*.npz')):
        counterpart = output / 'repeat_search' / path.name
        assert counterpart.is_file(), path.name
        with np.load(path) as a, np.load(counterpart) as b:
            assert a.files == b.files
            for key in a.files:
                assert np.array_equal(a[key], b[key], equal_nan=True), (path.name, key)
        checked.append({'file': path.name, 'arrays_identical': True,
                        'source_sha256': sha(path), 'repeat_sha256': sha(counterpart)})
    receipt = {'status': 'PASS', 'complete_search_repetitions': 2,
               'frozen_families_identical': True, 'npz_arrays': checked,
               'first_selection_sha256': sha(output / 'FROZEN_SELECTION.json'),
               'repeat_selection_sha256': sha(output / 'repeat_search/FROZEN_SELECTION.json')}
    (output / 'SEARCH_REPEAT_AUDIT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def package(output):
    output = Path(output)
    report = json.loads((output / 'REPORT_AUDIT.json').read_text())
    assert report['status'] == 'PASS'
    assert sha(HERE / 'report.py') == report['report_source_sha256']
    assert sha(output / 'FROZEN_SELECTION.json') == report['selection_sha256']
    repeat = verify_repeat(output)
    manifest_path = REPO / 'research/moe_dispatch/round2/results/E0/frozen_inputs.json'
    manifest = json.loads(manifest_path.read_text())
    input_root = Path(manifest['input_directory'])
    inputs = [(input_root / name, 'inputs/' + name) for name in manifest['input_sha256']]
    for path, _ in inputs:
        assert sha(path) == manifest['input_sha256'][path.name]
    # Carry source dependencies without previous experiment output directories.
    sources = []
    for directory in ('plena65536_affinity', 'plena65536_dedicated', 'plena65536_shared', 'plena65536_moe', 'compute_ceiling'):
        parent = HERE.parent / directory
        sources.extend(sorted(parent.glob('*.py')))
        if directory == 'plena65536_affinity':
            sources.extend(path for path in (parent / 'README.md', parent / 'requirements.txt') if path.exists())
    target = output / 'PLENA65536_affinity_complete.zip'
    results = [path for path in output.rglob('*') if path.is_file() and path != target
               and path.name not in {'DELIVERY_AUDIT.json', 'PORTABLE_ARCHIVE_AUDIT.json'}]
    script = '''#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$task_dir"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r research/moe_dispatch/plena65536_affinity/requirements.txt
.venv/bin/python -m pytest research/moe_dispatch/plena65536_affinity -q
.venv/bin/python -m research.moe_dispatch.plena65536_affinity.run --out reproduced_results
'''
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sources:
            archive.write(path, str(path.relative_to(REPO)))
        for path, name in inputs:
            archive.write(path, name)
        archive.writestr('research/moe_dispatch/round2/results/E0/frozen_inputs.json',
                         json.dumps(dict(manifest, input_directory='inputs'), indent=2) + '\n')
        archive.writestr('REPRODUCE.sh', script)
        for path in results:
            archive.write(path, 'results/' + str(path.relative_to(output)))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(set(names))
    receipt = {'status': 'PASS', 'archive': str(target), 'archive_bytes': target.stat().st_size,
               'archive_sha256': sha(target), 'archive_members': len(names),
               'source_sha256': {str(p.relative_to(REPO)): sha(p) for p in sources},
               'raw_input_sha256': {p.name: sha(p) for p, _ in inputs},
               'complete_search_repetitions': repeat['complete_search_repetitions'],
               'native_RTL_calibrated': False, 'timed_SRAM_implemented': False,
               'BLEN1_is_scalar_topology_extension': True}
    (output / 'DELIVERY_AUDIT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'archive', 'archive_bytes', 'archive_members')}), flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    package(parser.parse_args().out)
