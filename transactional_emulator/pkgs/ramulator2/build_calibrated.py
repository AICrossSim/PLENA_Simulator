#!/usr/bin/env python3
"""Build the pinned Ramulator source with PLENA's versioned calibration C API.

Dependencies must be supplied locally; CMake FetchContent never downloads them.
Use default.nix for the exact upstream source and dependency revisions/hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(root):
    records = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and '.git' not in path.relative_to(root).parts:
            records.append(str(path.relative_to(root)) + '\0' + digest(path) + '\n')
    return hashlib.sha256(''.join(records).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['source','yaml_source','fmt_source','build_root']:
        p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--cmake',default='cmake')
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args()
    if not 1 <= a.jobs <= 32: p.error('jobs must be 1..32')
    source=a.source.resolve(); root=a.build_root.resolve()
    expected_source = 'cec7b2dc82c9ae78d9c388a9cbc35d1ef6456f6757967f02d00a19497007f7c4'
    if source_identity(source) != expected_source: p.error('source tree is not the pinned Ramulator revision; refusing mislabeled calibration')
    if root.exists(): p.error('build-root must be new; existing builds are preserved')
    capi=Path(__file__).resolve().parent
    shutil.copytree(str(source),str(root))
    root.chmod(0o755)
    for path in root.rglob('*'):
        path.chmod(path.stat().st_mode | 0o200 | (0o100 if path.is_dir() else 0))
    for name in ['ramulator_capi.cc','ramulator_capi.h']:
        shutil.copyfile(str(capi/name),str(root/'src/ramulator/frontend/impl'/name))
    cmake_file=root/'src/ramulator/frontend/CMakeLists.txt'
    text=cmake_file.read_text()
    if 'impl/ramulator_capi.cc' not in text:
        if 'impl/external.cpp' not in text: raise RuntimeError('unexpected upstream frontend CMake layout')
        cmake_file.write_text(text.replace('impl/external.cpp','impl/external.cpp\nimpl/ramulator_capi.cc'))
    command=[a.cmake,'-S',str(root),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release',
        '-DRAMULATOR_PYTHON_BINDINGS=OFF',
        '-DFETCHCONTENT_SOURCE_DIR_YAML-CPP='+str(a.yaml_source.resolve()),
        '-DFETCHCONTENT_SOURCE_DIR_FMT='+str(a.fmt_source.resolve())]
    subprocess.run(command,check=True)
    subprocess.run([a.cmake,'--build',str(root/'build'),'-j',str(a.jobs)],check=True)
    evidence=dict(configure_command=command, library_sha256=digest(root/'libramulator.so'),
        upstream_revision='b3efdc5019a312874961a8c226097eb0581f2b5f',
        source_path=str(source), source_files={str(f.relative_to(source)):digest(f) for f in sorted((source/'src/ramulator').rglob('*')) if f.is_file()},
        capi_files={n:digest(capi/n) for n in ['ramulator_capi.cc','ramulator_capi.h']})
    (root/'build_provenance.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')


if __name__ == '__main__': main()
