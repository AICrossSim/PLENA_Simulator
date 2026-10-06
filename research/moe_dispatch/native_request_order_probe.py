"""Native memory-only diagnostic: same address multiset, different DMA burst order.

This isolates two weight streams' interference. No compute/SRAM runtime is modeled;
these numbers are NOT layer latencies or an optimized architecture result.
"""
import argparse, ctypes as c, hashlib, json, re
from collections import Counter
from pathlib import Path
from .native_profile import hbm_config,write

def run(library,burst):
    lib=c.CDLL(str(library));lib.ramulator_new.argtypes=[c.c_char_p];lib.ramulator_new.restype=c.c_void_p
    lib.ramulator_finalize.argtypes=[c.c_void_p]
    lib.ramulator_tick.argtypes=[c.c_void_p]
    cbtype=c.CFUNCTYPE(None,c.c_void_p)
    lib.ramulator_request.argtypes=[c.c_void_p,c.c_uint64,c.c_bool,cbtype,c.c_void_p,c.c_int];lib.ramulator_request.restype=c.c_bool
    lib.ramulator_tx_bytes.argtypes=[c.c_void_p];lib.ramulator_tx_bytes.restype=c.c_uint32
    lib.ramulator_stats.argtypes=[c.c_void_p,c.c_void_p,c.c_uint64];lib.ramulator_stats.restype=c.c_uint64
    assert lib.ramulator_capi_version()==2
    raw=lib.ramulator_new(json.dumps(hbm_config()).encode());assert raw and lib.ramulator_tx_bytes(raw)==32
    offsets=[n*4096+k*2+byte for ng in range(0,128,4) for k in range(0,2048,512)
             for n in range(ng,ng+4) for byte in range(0,1024,32)]
    order=[base+offset for at in range(0,len(offsets),burst) for base in (0,16*1024**2)
           for offset in offsets[at:at+burst]]
    address_sha=hashlib.sha256(b''.join(a.to_bytes(8,'little') for a in sorted(order))).hexdigest()
    pending=set();completed=[];errors=[];now=0;cursor=0;rejected=0
    @cbtype
    def done(ptr):
        serial=int(ptr)-1
        if serial not in pending:errors.append(serial)
        else:pending.remove(serial)
        completed.append(serial)
    while cursor<len(order) or pending:
        for _ in range(8):
            if cursor==len(order) or len(pending)>=256:break
            if lib.ramulator_request(raw,order[cursor],False,done,c.c_void_p(cursor+1),32):
                pending.add(cursor);cursor+=1
            else:rejected+=1;break
        lib.ramulator_tick(raw);now+=1
        assert now<1000000
    assert not errors and len(completed)==len(order)==len(set(completed))
    size=lib.ramulator_stats(raw,None,0);buf=c.create_string_buffer(size)
    assert lib.ramulator_stats(raw,buf,size)==size
    stats=buf.value.decode()
    def values(key):return [int(v) for v in re.findall(r'^\s*'+key+r':\s*(\d+)\s*$',stats,re.M)]
    assert sum(values('total_num_read_requests'))==sum(values('num_read_reqs_served'))==len(order)
    row={'burst_sectors_per_stream':burst,'cycles':now,'bytes':len(order)*32,
        'bandwidth_GBps':len(order)*32/now,'address_multiset_sha256':address_sha,
        'accepted_and_completed':len(order),'rejected_attempts':rejected,
        **{k:sum(values(k)) for k in ('row_hits','row_misses','row_conflicts')},
        'scope':'two weight streams; native memory only; no MAC/SRAM execution'}
    lib.ramulator_finalize(raw);return row

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--library',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();rows=[]
    for burst in (1,8,32,128,512):
        r=run(a.library,burst);assert r==run(a.library,burst);rows.append(r)
    assert len({r['address_multiset_sha256'] for r in rows})==1
    write(a.out,{'rows':rows,'exact_repeats':2,'all_same_address_multiset':True})
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
