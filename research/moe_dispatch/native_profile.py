"""Live native-HBM campaign for the fixed BF16 6 / 3+3 / 4+2 comparison.

No architecture search or fitted throughput: callbacks enter the existing runtime
event queue. Inputs/routes are resident; scope is routed FFN, not full-model E2E.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, math, statistics, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .frontend import compiler, COMPILER_PATH

ORGS = {'single6':[6], 'homo33':[3,3], 'heter42':[4,2]}

def hbm_config():
    """Exact resolved native HBM2_2000/8GB timing contract of the prior Rust wrapper.
    One 1ns memory tick, eight controllers, 32B sectors, no timing speedup oracle.
    """
    constraints = [
        [1,[3,5],[3,5],2], [1,[4,6],[4,6],2],
        [1,[3,5],[3,5],2], [1,[4,6],[4,6],2],
        [1,[3,5],[4,6],13], [1,[4,6],[3,5],13],
        [1,[3],[2],5], [1,[4],[2],23], [1,[0],[0],4],
        [1,[0],[0],15,4], [1,[0],[2],34], [1,[2],[0],14],
        [1,[0],[7],48], [1,[1,2],[7],14], [1,[5],[7],19],
        [1,[6],[7],37], [1,[7],[0,2],350], [1,[8],[0],8],
        [1,[0],[8],4], [2,[3,5],[3,5],4], [2,[4,6],[4,6],4],
        [2,[4,6],[3,5],15], [2,[0],[0],4], [3,[0],[0],48],
        [3,[0],[3,5],14], [3,[0],[4,6],12], [3,[0],[1],34],
        [3,[1],[0],14], [3,[3],[1],5], [3,[4],[1],23],
        [3,[5],[0],19], [3,[6],[0],37], [3,[8],[0],160],
        [3,[0],[8],48], [3,[1],[8],14],
    ]
    dram={'impl':'HBM2','channel_width':64,
        'org':{'dq':64,'count':[1,2,4,4,65536,128]},
        'timing':[2000,2,14,14,12,14,34,48,16,5,5,2,4,4,4,6,8,15,350,160,8,3900,122,1000.0],
        'read_latency':16,'timing_constraints':constraints}
    controller={'impl':'HBM12','scheduler':{'impl':'FRFCFS'},
        'refresh_manager':{'impl':'AllBank'},'row_policy':{'impl':'Open'},
        'addr_mapper':{'impl':'MOP4CLXOR'},'dram':dram}
    return {'frontend':{'impl':'External','clock_ratio':1},
        'memory_system':{'impl':'GenericDRAM','clock_ratio':1,
            'controllers':[copy.deepcopy(controller) for _ in range(8)],
            'channel_mapper':{'impl':'CacheLineInterleave'}}}

def config(lanes, backend='native', trace=False):
    cfg={'lanes':lanes,'group':4,'dispatch':'dynamic','split':'none',
        'runtime_fsm':True,'next_prefetch':True,'arbiter':'stock','window':8,
        'hbm_bytes_per_ns':256,'hbm_latency_ns':64,'credits':256,
        'ideal_hbm':False,'ideal_onchip':False,'control_cost':True,
        'onchip_bytes_per_ns':384,'vector_elements_per_ns':32,'dot_tail_ns':20,
        'record_trace':trace,'live_profile':True,'max_cycles':200000000}
    if backend=='native': cfg.update(native_hbm=True,native_hbm_config=hbm_config())
    return cfg

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def run_point(point, binary, out, backend):
    w,org=point
    directory=out/'points'/org/w['id'];directory.mkdir(parents=True,exist_ok=True)
    lanes=ORGS[org];w=copy.deepcopy(w);w['engine_layout']=compiler.engine_layout(w,lanes,4)
    cfg=config(lanes,backend)
    write(directory/'workload.json',w);write(directory/'config.json',cfg)
    reports=[];host=[]
    for repeat in (1,2):
        output=directory/f'repeat{repeat}.json';start=time.monotonic()
        with (directory/f'repeat{repeat}.log').open('w') as log:
            proc=subprocess.run([str(binary),str(directory/'workload.json'),str(directory/'config.json'),str(output)],
                stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        if proc.returncode: raise RuntimeError(f'{directory}: return {proc.returncode}; see log')
        host.append(time.monotonic()-start);reports.append(json.loads(output.read_text()))
    assert reports[0]==reports[1],f'repeat mismatch: {directory}'
    r=reports[0];p=r['live_timing_profile']
    assert r['drained'] and r['ownership_k_order_capacity_checks']
    assert r['credit_peak']<=256
    assert r['weight_bytes']==32*r['dma_transactions_accepted']==32*r['dma_transactions_landed']
    assert p['weight_callbacks_returned']==r['dma_transactions_landed']
    if backend=='native':
        n=r['native_hbm'];assert n['pending']==0 and n['accepted']==n['completed']==r['dma_transactions_accepted']
    receipt={'full_report_exact_repeat':True,'repeats':2,'host_seconds':host,
        'workload_sha256':digest(directory/'workload.json'),'config_sha256':digest(directory/'config.json'),
        'report_sha256':digest(directory/'repeat1.json'),'binary_sha256':digest(binary)}
    write(directory/'repeat_receipt.json',receipt)
    row={'workload':w['id'],'batch':w['batch'],'organization':org,'backend':backend,
        'useful_macs':r['useful_macs'],'weight_bytes':r['weight_bytes'],
        'total_ms':r['cycles']/1e6,'fetch_ms':p['fetch_span_cycles']/1e6,
        'compute_active_ms':p['mac_active_union_cycles']/1e6,
        'operand_mac_pipeline_ms':p['operand_and_mac_pipeline_union_cycles']/1e6,
        'fetch_compute_overlap_ms':p['fetch_mac_overlap_cycles']/1e6,
        'combine_tail_ms':r['combine_tail_cycles']/1e6,
        'fetch_GBps':p['fetch_bandwidth_GBps'],'compute_active_GFLOPs':p['compute_active_GFLOPs'],
        'wall_GFLOPs':p['wall_GFLOPs'],'request_latency_mean_ns':p['request_latency_mean_cycles'],
        'request_latency_p95_ns':p['request_latency_p95_cycles'],'request_latency_max_ns':p['request_latency_max_cycles'],
        'native_backpressure_cycles':p['native_frontend_backpressure_cycles'],
        'memory_outstanding_cycles':p['memory_outstanding_cycles'],
        'spatial_utilization':r['useful_macs']/r['issued_macs'],
        'routed_experts_Me_gt2':sum(e['Me']>2 for e in w['experts'] if not e['is_shared']),
        'routed_experts_Me_le2':sum(e['Me']<=2 for e in w['experts'] if not e['is_shared']),
        'shared_experts':sum(e['is_shared'] for e in w['experts']),
        'credit_peak':r['credit_peak'],'requests':r['dma_transactions_accepted'],
        'repeat_exact':True,'host_seconds':sum(host)}
    row['core_details']=[{'core':i,'m':c['m'],'stats':c['stats'],
        'live':p['cores'][i], 'weight_bank_words':c['weight_bank_words'],
        'x_bank_words':c['x_bank_words'],'workspace_bank_words':c['workspace_bank_words']}
        for i,c in enumerate(r['cores'])]
    return row

def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='core_details']);writer.writeheader()
        writer.writerows({k:v for k,v in r.items() if k!='core_details'} for r in rows)

def aggregate(rows):
    ans=[]
    for batch in (2,4,8,16):
        for org in ORGS:
            rs=[r for r in rows if r['batch']==batch and r['organization']==org]
            if not rs:continue
            row={'batch':batch,'organization':org,'windows':len(rs)}
            for key in ('total_ms','fetch_ms','compute_active_ms','operand_mac_pipeline_ms',
                        'fetch_compute_overlap_ms','combine_tail_ms','request_latency_mean_ns',
                        'native_backpressure_cycles','spatial_utilization',
                        'routed_experts_Me_gt2','routed_experts_Me_le2','shared_experts'):
                row[key]=statistics.mean(r[key] for r in rs)
            row['fetch_GBps']=sum(r['weight_bytes'] for r in rs)/(sum(r['fetch_ms'] for r in rs)*1e6)
            row['compute_active_GFLOPs']=2*sum(r['useful_macs'] for r in rs)/(sum(r['compute_active_ms'] for r in rs)*1e6)
            row['wall_GFLOPs']=2*sum(r['useful_macs'] for r in rs)/(sum(r['total_ms'] for r in rs)*1e6)
            row['total_p95_ms']=sorted(r['total_ms'] for r in rs)[math.ceil(.95*len(rs))-1]
            row['mean_weight_MiB']=statistics.mean(r['weight_bytes']/2**20 for r in rs)
            ans.append(row)
    return ans

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--jobs',type=int,default=8)
    ap.add_argument('--backend',choices=['native','aggregate'],default='native')
    ap.add_argument('--limit',type=int,default=0,help='smoke only; 0 = all frozen heldout windows')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    ws=json.loads((args.inputs/'heldout.json').read_text())['workloads']
    ws=[w for w in ws if w['batch'] in (2,4,8,16)]
    if args.limit: ws=ws[:args.limit]
    points=[(w,org) for w in ws for org in ORGS]
    source=Path(__file__).resolve().parent
    files=list((source/'rust/src').rglob('*.rs'))+[source/'rust/Cargo.toml',source/'rust/Cargo.lock',Path(__file__),COMPILER_PATH]
    manifest={'schema':'native_fixed_profile_campaign_v1','backend':args.backend,'window_ids':[w['id'] for w in ws],
        'points':len(points),'repeats':2,'limited_smoke':bool(args.limit),
        'binary_sha256':digest(args.binary),'input_sha256':digest(args.inputs/'heldout.json'),
        'sources':{str(p):digest(p) for p in files},'orchestrator_wall_start':time.time(),
        'scope':'captured routing; resident X/routes through gate/up/SiLU/down/ordered combine; not full-model E2E',
        'hardware':'BF16; sum PM=6, PN=4, PK=512; 12288 multipliers; same total SRAM/banks/credits; group4; same dynamic dispatcher and stock arbiter'}
    write(args.out/'manifest.json',manifest);rows=[];errors=[]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(run_point,p,args.binary.resolve(),args.out,args.backend):p for p in points}
        for future in as_completed(futures):
            try: rows.append(future.result())
            except Exception as exc:errors.append({'point':str(futures[future][0]['id'])+'/'+futures[future][1],'error':str(exc)})
            if (len(rows)+len(errors))%6==0 or errors:
                print(json.dumps({'completed':len(rows),'failed':len(errors),'total':len(points),'last':rows[-1]['workload'] if rows else ''}),flush=True)
                write(args.out/'progress.json',{'completed':len(rows),'errors':errors,'total':len(points)})
    rows.sort(key=lambda r:(r['batch'],r['workload'],r['organization']))
    write(args.out/'windows.json',rows);csv_write(args.out/'windows.csv',rows)
    summary=aggregate(rows);write(args.out/'by_batch.json',summary);csv_write(args.out/'by_batch.csv',summary)
    manifest.update(completed=len(rows),errors=errors,orchestrator_wall_end=time.time(),
        exact_repeats_all=len(rows)==len(points) and not errors,
        sources_unchanged=all(digest(p)==manifest['sources'][str(p)] for p in files))
    write(args.out/'manifest.json',manifest)
    if errors:raise SystemExit(f'{len(errors)} failures; see manifest')

if __name__=='__main__':main()
