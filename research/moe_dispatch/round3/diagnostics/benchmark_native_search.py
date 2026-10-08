"""Full-search equivalence against the pre-optimization diagnostic certificate."""
import gzip
import json
import time
from research.moe_dispatch.round3 import sensitivity
from research.moe_dispatch.round3.common import ROOT, inputs, frozen_designs, canonical, write_json, sha
from research.moe_dispatch.round3.native_enum import enable_exact_native_enum


def main():
    original=ROOT/'diagnostics/profile/diagnostic_sample_0000.json.gz'
    before=json.loads(gzip.decompress(original.read_bytes()))
    destination=ROOT/'diagnostics/native_full_search'
    sensitivity.ROOT=destination
    spec=(0,[15,16,2,520,1],inputs()['development'],8,32,
          list(frozen_designs('pipelined').values()))
    enable_exact_native_enum()
    started=time.monotonic()
    result=sensitivity._sobol_job(spec)
    elapsed=time.monotonic()-started
    after_file=destination/'E5/sobol/certificates/0000.json.gz'
    after=json.loads(gzip.decompress(after_file.read_bytes()))
    assert canonical(before)==canonical(after),'full-search result changed'
    write_json(destination/'EQUIVALENCE.json',dict(
        before_sha256=sha(original),after_sha256=sha(after_file),
        full_search_result_bit_exact=True,elapsed_seconds=elapsed,result=result,
        note='Two full searches and every per-window double allocation/replay retained; diagnostic only, not Saltelli sample 0'))
    print('Full search bit-exact; elapsed seconds:',elapsed,flush=True)


if __name__=='__main__':main()
