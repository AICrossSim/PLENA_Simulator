"""Build the pinned Ramulator source with the matching CAPI v2 (offline inputs).

Example: python build.py --source /path/to/ramulator2 --fmt /path/to/fmt10
  --yaml /path/to/yaml-cpp --out /path/to/new/build --cmake /path/to/cmake
  --cxx /path/to/g++
Source revision: b3efdc5019a312874961a8c226097eb0581f2b5f. A previous five-symbol
CAPI .so is incompatible; the runtime checks ABI, sector size and clock before use.
"""
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','fmt','yaml','out'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--cmake',default='cmake');ap.add_argument('--cxx',default='g++');ap.add_argument('--jobs',type=int,default=8)
    a=ap.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    source=a.out/'ramulator_src';assert not source.exists(),'use a fresh build destination'
    inventory={str(p.relative_to(a.source)):sha(p) for p in a.source.rglob('*') if p.is_file()}
    shutil.copytree(a.source,source)
    for p in [source,*source.rglob('*')]:p.chmod(p.stat().st_mode|0o200)
    here=Path(__file__).resolve().parent
    for name in ('ramulator_capi.cc','ramulator_capi.h'):shutil.copy2(here/name,source/'src/ramulator'/name)
    with (source/'CMakeLists.txt').open('a') as f:
        f.write('\ntarget_sources(ramulator PRIVATE src/ramulator/ramulator_capi.cc)\ntarget_link_libraries(ramulator PRIVATE dl)\n')
    configure=[a.cmake,'-S',str(source),'-B',str(a.out/'cmake'),'-DRAMULATOR_PYTHON_BINDINGS=OFF',
        '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER='+a.cxx,
        '-DFETCHCONTENT_SOURCE_DIR_FMT='+str(a.fmt.resolve()),
        '-DFETCHCONTENT_SOURCE_DIR_YAML-CPP='+str(a.yaml.resolve())]
    build=[a.cmake,'--build',str(a.out/'cmake'),'--target','ramulator','-j'+str(a.jobs)]
    for name,cmd in (('configure',configure),('build',build)):
        with (a.out/(name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt={'upstream_revision':'b3efdc5019a312874961a8c226097eb0581f2b5f',
        'source':str(a.source.resolve()),'source_hashes':inventory,
        'wrapper_hashes':{name:sha(here/name) for name in ('ramulator_capi.cc','ramulator_capi.h')},
        'configure_argv':configure,'build_argv':build,
        'native_library':str(source/'libramulator.so'),'native_library_sha256':sha(source/'libramulator.so')}
    (a.out/'native_build.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:receipt[k] for k in ('native_library','native_library_sha256')}))
if __name__=='__main__':main()
