#!/usr/bin/env python3
"""Step2 native split-window measurements; exact repeats and frozen Step1 L."""
import argparse, concurrent.futures, copy, gzip, json, os, shutil, subprocess, tempfile
from pathlib import Path
import step0 as b
import diagnose_step1 as diag

COMMON = diag.RUN/'step1/common_baseline_8f37c0e1bac74dfea2b27aa9e855dd4b'
ACCEPTED = diag.RUN/'step1/cohort_revision_538305a622d64dabb75e79205bbc0e11'


def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as f:return json.load(f)
    return b.read(path)


def save_gzip(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0,compresslevel=6) as out:
            out.write(json.dumps(value,separators=(',',':'),allow_nan=False).encode())
    tmp.replace(path)


def cycle_peaks(changes, period, total):
    """[start inclusive, end exclusive, packed, operand, decode], peak per cycle.

    Keep a simultaneously observed tuple at the maximum summed occupancy in
    each clock cycle; never sum independent component peaks. Idle cycles RLE.
    """
    result=[];current=[0,0,0];cycle=0;peak=current
    def emit(start,end,values):
        if end<=start:return
        if result and result[-1][1]==start and result[-1][2:]==values:result[-1][1]=end
        else:result.append([start,end,*values])
    for at,*values in changes:
        next_cycle=at//period
        if next_cycle>cycle:
            emit(cycle,cycle+1,peak);emit(cycle+1,next_cycle,current)
            cycle=next_cycle;peak=current
        current=values
        if sum(values)>sum(peak):peak=values
    emit(cycle,cycle+1,peak)
    emit(cycle+1,max(cycle+1,(total+period-1)//period),current)
    return result


def sources():
    result=[]
    for root in [COMMON,COMMON/'fixed_ownership']:
        for c in b.read(root/'manifest.json')['cases']:
            if c['mode']!='cohort':continue
            if c['policy']=='threshold' and c['workload_kind']!='me1':continue
            result.append((root,c))
    return result


def prepare(out):
    repro=out/'repro';repro.mkdir(exist_ok=True)
    shutil.copy2('/tmp/plena-moe-dual-core-target/release/moe_dual_normal',repro/'moe_dual_normal')
    shutil.copy2('/tmp/plena-moe-dma-native/libramulator.so',repro/'libramulator.so')
    for filename in ['step2.py','regress_step2.py','step0.py','diagnose_step1.py']:
        shutil.copy2(b.SOURCE/'scripts/moe_stream_ctrl'/filename,repro/filename)
    shutil.copy2(b.SOURCE/'transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py',repro/'compare_moe_normal.py')
    shutil.copy2(ACCEPTED/'repro/build-env.sh',repro/'build-env.sh')
    diag.source_archive(repro)
    # Entire operator source includes new untracked modules; patch alone is insufficient.
    shutil.copytree(b.SOURCE/'transactional_emulator/src/moe_normal',repro/'moe_normal')
    cases=[]
    for root,c in sources():
        reference=root/c['id']/'rep1.json';envelope=read(reference)
        source_a=b.read(c['architecture'])
        for window in [3,6]:
            for aging,weighted in [(2,True),(4,True),(8,True),(None,True),(None,False)]:
                label=f"w{window}_age{aging or 'off'}_{'reserved' if weighted else 'shared'}"
                case_id=f"{c['workload_kind']}__{c['organization']}__{c['policy']}__{label}"
                folder=out/case_id;folder.mkdir()
                a=copy.deepcopy(source_a);a['name']=case_id;a['dma']['reserved_byte_credits']=weighted
                for cfg,core in zip(a['cores'],envelope['result']['cores']):
                    l=core['tile_loads'];b.require(l['count']>0,'inactive core has no measured L')
                    cfg['refinement']['stream_ctrl'].update(split_slot_lifetime=True,split_window=dict(
                        window_tiles=window,load_latency_sum_ps=l['total_ps'],load_latency_samples=l['count'],aging_multiplier=aging))
                b.save(folder/'architecture.json',a)
                case={k:c[k] for k in ['workload_kind','organization','policy','workload','golden','lower_bound_us']}
                case.update(id=case_id,window=window,aging=aging,weighted=weighted,architecture=str(folder/'architecture.json'),
                            reference=str(reference),baseline_us=envelope['result']['total_ps']/1e6)
                case['hashes']={k:b.digest(case[k]) for k in ['architecture','workload','golden','reference']}
                cases.append(case)
    m=dict(status='prepared',repeats=2,cases=cases,planned_runs=2*len(cases),common_baseline=str(COMMON),
           gates=dict(me1_max_us=26.5,frozen_me1_us=31.124,b8_bound_us=235.008,b8_stop_ratio=1.3),
           artifacts={str(p.relative_to(repro)):b.digest(p) for p in repro.rglob('*') if p.is_file()})
    b.save(out/'manifest.json',m);return m


def run_case(c,out,m,validator):
    folder=out/c['id'];a,w,g=[read(c[k]) for k in ['architecture','workload','golden']];reference=read(c['reference'])['result']
    for key,h in c['hashes'].items():b.require(b.digest(c[key])==h,'input changed '+key)
    first=None;rows=[]
    for rep in [1,2]:
        dest=folder/f'rep{rep}.json.gz'
        if not dest.exists():
            with tempfile.TemporaryDirectory(prefix='plena-step2-',dir='/tmp') as temp:
                raw=Path(temp)/'result.json'
                cmd=[str(out/'repro/moe_dual_normal'),'--architecture',c['architecture'],'--workload',c['workload'],
                     '--output',str(raw),'--hbm-channels','8','--max-hbm-bytes',str(1<<30)]
                with (folder/f'rep{rep}.log').open('w') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=1800,check=True,
                                   env=dict(os.environ,LD_LIBRARY_PATH=str(out/'repro')))
                e=read(raw);save_gzip(dest,e)
        else:e=read(dest)
        validator.validate_run(e,g,w,a,0,0);r=e['result'];diag.require_bit_exact(r,g)
        for field in ['output_bf16','output_f32','pre_round_output_f32','useful_macs','hbm_read_bytes','hbm_write_bytes']:
            b.require(r[field]==reference[field],'matched Step1 invariant differs: '+field)
        if c['policy']=='fixed':
            b.require(diag.job_order(r,a)==a['diagnostic']['fixed_job_order'],'fixed owner/order changed')
            for nr,old in zip(r['cores'],reference['cores']):
                for field in ['hbm_read_bytes','useful_macs','issued_macs','jobs']:
                    b.require(nr[field]==old[field],'fixed per-core work changed '+field)
        native=e['memory_model']['calibration']
        for key,expected in [('executable_sha256',m['artifacts']['moe_dual_normal']),('native_library_sha256',m['artifacts']['libramulator.so']),
                             ('workload_sha256',c['hashes']['workload']),('architecture_sha256',c['hashes']['architecture']),('hbm_sha256',w['metadata']['hbm_sha256'])]:
            b.require(e['provenance'][key]==expected,'provenance changed '+key)
        if first:b.require(first['result']==r and first['memory_model']['calibration']==native,'nondeterministic repeat')
        first=e
        rows.append(dict(case=c['id'],repeat=rep,workload=c['workload_kind'],organization=c['organization'],policy=c['policy'],window=c['window'],
            aging=c['aging'],weighted=c['weighted'],time_us=r['total_ps']/1e6,baseline_us=c['baseline_us'],
            speedup=c['baseline_us']/(r['total_ps']/1e6),lower_bound_ratio=r['total_ps']/1e6/c['lower_bound_us'] if c['lower_bound_us'] else None,
            hbm_read_bytes=r['hbm_read_bytes'],useful_macs=r['useful_macs'],issued_macs=r['issued_macs'],
            dma_native_admission_wait_ps=native['admission_wait_ps'],shared_vector_busy_ps=r['shared_vector_busy_ps']))
    cores=[]
    for core,cfg in zip(first['result']['cores'],a['cores']):
        d=core['refinement'];s=d['split_window'];p=d['output_pool'];ctrl=d['stream_ctrl']
        admissions=sum(x['metrics']['band_admissions']*((x['m']+d['m_rows']-1)//d['m_rows']+1) for x in core['projections'])
        row=dict(case=c['id'],core=core['id'],tiles=p['tile_admissions'],issues=p['context_updates'],bursts=ctrl['burst_starts'],
            admission_cycles=admissions,tile_cycles=10*p['tile_admissions'],fanout_cycles=p['context_updates'],
            burst_cycles=4*ctrl['burst_starts'],mask_cycles=ctrl['completion_mask_cycles'],control_service_ps=p['scheduler_busy_ps'],
            L_ps=cfg['refinement']['stream_ctrl']['split_window']['load_latency_sum_ps']/cfg['refinement']['stream_ctrl']['split_window']['load_latency_samples'],
            weight_ready_wait_ps=core['weight_ready_wait_ps'],accumulator_dependency_stall_ps=core['accumulator_dependency_stall_ps'],
            weight_budget_bytes=cfg['weight_sram_bytes'],weight_sram_peak_bytes=core['weight_sram_peak_bytes'],
            control_reserved_bytes=p['control_reserved_bytes'],accumulator_peak_bytes=core['accumulator_peak_bytes'],accumulator_budget_bytes=cfg['accumulator_bytes'],
            frontend_reserved_bytes=first['result']['dma_frontend']['reserved_bytes'],frontend_budget_bytes=a['dma']['frontend_sram_bytes'],
            core_hbm_read_bytes=core['hbm_read_bytes'],jobs=core['jobs'],token_rows=sum(j['rows'] for j in first['result']['job_completions'] if j['core']==core['id']))
        row.update({k:v for k,v in s.items() if not isinstance(v,(list,dict))})
        row.update({f'packed_arrival_{k}':v for k,v in s['packed_arrivals'].items()})
        b.require(admissions+row['tile_cycles']+row['fanout_cycles']+row['burst_cycles']+row['mask_cycles']==p['scheduler_visits'],'service formula mismatch')
        cycles=cycle_peaks(s['lifetime_changes'],a['clock_period_ps'],first['result']['total_ps'])
        b.require(max((sum(v[2:]) for v in cycles),default=0)==s['live_peak_bytes'],'cycle peak changed')
        save_gzip(folder/f"cycle_peaks_{core['id']}.json.gz",dict(clock_period_ps=a['clock_period_ps'],
           columns=['cycle_start','cycle_end_exclusive','packed_bytes','ready_operand_bytes','decode_destination_bytes'],ranges=cycles))
        cores.append(row)
    b.csv_write(folder/'measurements.csv',rows);b.csv_write(folder/'core_services.csv',cores)
    b.save(folder/'validation.json',dict(status='passed',repeats_exact=True,bit_exact=True,native_drained=True,hbm_bytes_unchanged=True,
             outputs={p.name:b.digest(p) for p in folder.glob('*.json.gz')}))
    return rows,cores


def aggregate(out,m):
    import csv
    rows=[];cores=[]
    for c in m['cases']:
        folder=out/c['id']
        if not (folder/'validation.json').exists():continue
        for filename,dest in [('measurements.csv',rows),('core_services.csv',cores)]:
            with (folder/filename).open() as f:dest.extend(csv.DictReader(f))
    if rows:b.csv_write(out/'measurements.csv',rows);b.csv_write(out/'core_services.csv',cores)
    m.update(completed_runs=len(rows),status='complete' if len(rows)==m['planned_runs'] else 'in_progress')
    b.save(out/'manifest.json',m)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--group',choices=['me1','default','all'],default='me1');args=p.parse_args();out=args.output.resolve()
    m=read(out/'manifest.json') if (out/'manifest.json').exists() else prepare(out)
    for file,h in m['artifacts'].items():b.require(b.digest(out/'repro'/file)==h,'archived source changed '+file)
    validator=diag.load_validator(out/'repro/compare_moe_normal.py')
    selected=[c for c in m['cases'] if args.group=='all' or args.group=='me1' and c['workload_kind']=='me1'
              or args.group=='default' and c['window']==6 and c['aging']==4 and c['weighted']]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures={ex.submit(run_case,c,out,m,validator):c for c in selected}
            for f in concurrent.futures.as_completed(futures):
                rr,cc=f.result();print('PASS',futures[f]['id'],rr[0]['time_us'],'us',flush=True)
    finally:aggregate(out,m)
    print('GROUP COMPLETE',args.group,len(selected)*2,'runs',flush=True)

if __name__=='__main__':main()
