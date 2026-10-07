#!/usr/bin/env python3
"""Offline address/budget arithmetic for the DMA design, NOT a timing model.

Mapping is conditional on the pinned source contract: 32B native transactions,
8 controllers, CacheLineInterleave with interleave_bits=0. The loaded library
still requires an independent runtime capability/command-count check.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def sectors(base, stride, rows, count, granule):
    result=set()
    for row in range(rows):
        begin=base+row*stride
        result.update(range(begin//granule,(begin+count-1)//granule+1))
    return result


def histogram(native_sectors):
    counts=Counter(s%8 for s in native_sectors)
    return [counts[i] for i in range(8)]


def budget(lines):
    components=dict(tile_descriptors=32*128,tile_states=16*128,line_table=lines*32,
        waiters=2*lines*24,native_queue_tracker=2*lines*16,response_payload=lines*64,
        scale_retention=5*1024,pipeline_payload=1024,queue_arbitration_counters=1024)
    return dict(line_slots=lines,bytes_by_region=components,total_bytes=sum(components.values()))


def audit(workload_path,output):
    encoded=workload_path.read_bytes();w=json.loads(encoded);view=w['experts'][0]['gate']
    assert view['cols']==2048, 'this documented example is specifically D=2048'
    assert view['scale_row_stride']==256
    rows=[]
    for name,b,k in [('single_b4_k1024',4,1024),('large_b16_k192',16,192),('small_b8_k128',8,128)]:
        assert k%8==0
        for kind,count,stride,base in [('element',k,view['element_row_stride'],view['element_base']),
                                      ('scale',k//8,view['scale_row_stride'],view['scale_base'])]:
            needed=sectors(base,stride,b,count,32)
            upper=sectors(base,stride,b,count,64)
            covered={2*line+offset for line in upper for offset in range(2)}
            assert needed<=covered
            padded_stride=stride+32
            changed=sectors(base,padded_stride,b,count,32)
            rows.append(dict(core=name,kind=kind,rows=b,valid_bytes_per_row=count,
                original_stride=stride,needed_native_sector_count=len(needed),
                needed_native_sector_histogram=histogram(needed),upper_64B_line_count=len(upper),
                upper_64B_covered_sector_histogram=histogram(covered),
                candidate_stride=padded_stride,candidate_needed_sector_histogram=histogram(changed)))
    budgets=[budget(128),budget(512)]
    assert [b['total_bytes'] for b in budgets]==[35*1024,101*1024]
    bdp=[dict(target_GBps=bw,latency_ns=lat,required_bytes=bw*lat,
              minimum_lines_64B=math.ceil(bw*lat/64),minimum_native_32B=math.ceil(bw*lat/32))
         for bw in [64,128,256] for lat in [50,100,200]]
    output.mkdir(parents=True,exist_ok=True)
    result=dict(status='offline_arithmetic_checked_not_runtime_validated',
        scope='first Gate tile, one expert, fixed-source conditional mapping; no latency or speedup prediction',
        workload=str(workload_path),workload_sha256=hashlib.sha256(encoded).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        backend_assumptions=dict(native_bytes=32,controllers=8,interleave_bits=0,
            pinned_commit='b3efdc5019a312874961a8c226097eb0581f2b5f',runtime_verified=False),
        address_examples=rows,budgets=budgets,bdp_examples_not_measurements=bdp)
    (output/'design_arithmetic.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('Checked: 6 address examples; 35/101 KiB budget sums; 9 illustrative BDP points. No performance claim.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();audit(args.workload.resolve(),args.output_dir.resolve())
