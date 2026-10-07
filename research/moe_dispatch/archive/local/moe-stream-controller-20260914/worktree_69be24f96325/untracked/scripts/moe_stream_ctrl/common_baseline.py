#!/usr/bin/env python3
"""Freeze Step1 common controls before changing the Step2 data lifetime."""
import argparse, concurrent.futures, copy, json, os, shutil, subprocess, time, uuid
from pathlib import Path
import step0 as b
import diagnose_step1 as diag

ROOT=diag.RUN
ACCEPTED=ROOT/'step1/cohort_revision_538305a622d64dabb75e79205bbc0e11'
TEMPLATES=b.ROOT/'outputs/moe_output_pool_20260909/prepared/architectures'


def prepare(out):
    out.mkdir();repro=out/'repro';repro.mkdir()
    for name in ['moe_dual_normal','libramulator.so','compare_moe_normal.py','source.patch','engine.rs','output_pool.rs',
                 'event_ready.rs','cohort.rs','control_port.rs','types.rs','build-env.sh']:
        shutil.copy2(ACCEPTED/'repro'/name,repro/name)
    for name in ['common_baseline.py','step0.py','diagnose_step1.py']:
        shutil.copy2(b.SOURCE/'scripts/moe_stream_ctrl'/name,repro/name)
    cases=[]
    inputs=[('b8',b.ROOT/'outputs/moe_refinement_20260909/fixed_bank_full_qwen/windows/qwen_full_decode_b8',235.008),
            ('b32',b.ROOT/'outputs/moe_refinement_20260909/fixed_bank_full_qwen/windows/qwen_full_decode_b32',317.952),
            ('me1',b.ROOT/'outputs/moe_output_pool_20260909/prepared/service/expert_me1',None)]
    for workload,folder,lower in inputs:
        for org in ['single','homogeneous','heterogeneous']:
            for policy in ['threshold','work_conserving']:
                for mode in ['cohort','charged_n3']:
                    if workload=='me1' and (org,policy,mode)!=('single','threshold','cohort'):continue
                    name=f'{workload}__{org}__{policy}__{mode}'
                    dest=out/name;dest.mkdir()
                    a=b.read(TEMPLATES/f'output_pool_threshold_pool_q32_{org}.json')
                    a.update(name=name,dispatch_policy=policy)
                    a['diagnostic']=dict(control_ports=1,charge_legacy_control=mode=='charged_n3')
                    for core in a['cores']:
                        r=core['refinement'];core['weight_slots']=3
                        if mode=='cohort':
                            r['stream_ctrl']=dict(event_ready=True,cohort_control=True,selection='rotating')
                        else:
                            r.pop('output_pool',None);r.pop('stream_ctrl',None);r['active_n_tiles']=3
                            r['operand_latch_bytes']=3*core['blen']*core['mlen']*2
                    b.save(dest/'architecture.json',a)
                    c=dict(id=name,workload_kind=workload,organization=org,policy=policy,mode=mode,
                           architecture=str(dest/'architecture.json'),workload=str(folder/'workload.json'),
                           golden=str(folder/'golden.json'),lower_bound_us=lower)
                    c['hashes']={k:b.digest(c[k]) for k in ['architecture','workload','golden']}
                    cases.append(c)
    m=dict(status='prepared',accepted_source=str(ACCEPTED),cases=cases,repeats=2,planned_runs=2*len(cases),
           fixed_semantics='frozen threshold Me>=8 to large else small; no timing-dependent owner changes',
           artifacts={p.name:b.digest(p) for p in repro.iterdir() if p.is_file()})
    b.save(out/'manifest.json',m);return m


def run_case(c,out,m,validator):
    for k,h in c['hashes'].items():b.require(b.digest(c[k])==h,'input changed '+k)
    a,w,g=[b.read(c[k]) for k in ['architecture','workload','golden']];folder=out/c['id'];first=None;rows=[]
    for rep in [1,2]:
        p=folder/f'rep{rep}.json'
        if not p.exists():
            cmd=[str(out/'repro/moe_dual_normal'),'--architecture',c['architecture'],'--workload',c['workload'],
                 '--output',str(p),'--hbm-channels','8','--max-hbm-bytes',str(1<<30)]
            with p.with_suffix('.log').open('w') as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,LD_LIBRARY_PATH=str(out/'repro')),
                               timeout=1800,check=True)
        e=b.read(p);r=e['result'];native=e['memory_model']['calibration']
        validator.validate_run(e,g,w,a,0,0);diag.require_bit_exact(r,g)
        for key,expected in [('executable_sha256',m['artifacts']['moe_dual_normal']),
                             ('native_library_sha256',m['artifacts']['libramulator.so']),
                             ('architecture_sha256',c['hashes']['architecture']),('workload_sha256',c['hashes']['workload']),
                             ('hbm_sha256',w['metadata']['hbm_sha256'])]:
            b.require(e['provenance'][key]==expected,'provenance mismatch '+key)
        expected_bytes={'b8':60_162_048,'b32':81_395_712,'me1':3_538_944}[c['workload_kind']]
        b.require(r['hbm_read_bytes']==expected_bytes and r['hbm_write_bytes']==0,'native bytes changed')
        b.require(native['native_pending']==0,'native not drained')
        if first:b.require(r==first['result'] and native==first['memory_model']['calibration'],'repeats differ')
        first=e
        rows.append(dict(case=c['id'],repeat=rep,workload=c['workload_kind'],organization=c['organization'],policy=c['policy'],
                         mode=c['mode'],time_us=r['total_ps']/1e6,lower_bound_ratio=r['total_ps']/1e6/c['lower_bound_us'] if c['lower_bound_us'] else None,
                         hbm_read_bytes=r['hbm_read_bytes'],useful_macs=r['useful_macs'],issued_macs=r['issued_macs']))
    b.save(folder/'validation.json',dict(status='passed',repeats_exact=True,bit_exact=True,native_drained=True))
    core_rows=[]
    for core,cfg in zip(first['result']['cores'],a['cores']):
        loads=core['tile_loads'];d=core['refinement'];p=d.get('output_pool')or{};legacy=d.get('legacy_control')or{}
        core_rows.append(dict(case=c['id'],core=core['id'],jobs=core['jobs'],
          token_rows=sum(j['rows'] for j in first['result']['job_completions'] if j['core']==core['id']),
          tiles=loads['count'],load_total_ps=loads['total_ps'],L_ps=loads['total_ps']/loads['count'] if loads['count'] else None,
          load_min_ps=loads['min_ps'],load_max_ps=loads['max_ps'],
          control_service_ps=p.get('scheduler_busy_ps',sum(legacy.get('service_ps_by_kind',[]))),
          accumulator_dependency_stall_ps=core['accumulator_dependency_stall_ps'],hbm_read_bytes=core['hbm_read_bytes'],
          accumulator_peak_bytes=core['accumulator_peak_bytes'],weight_sram_peak_bytes=core['weight_sram_peak_bytes']))
    return rows,core_rows


def report(out,m,rows,cores):
    b.csv_write(out/'measurements.csv',rows);b.csv_write(out/'core_services.csv',cores)
    text=['# Step1 common baseline — frozen before Step2','',
      'Default cohort, one control port and rotating selector versus 2/3/2 charged N3. Native standard HBM/DMA settings; no HBM-clock/DMA 2x diagnostic here. Each point runs twice. Fixed means the frozen threshold assignment (Me>=8 to large, otherwise small), identical across controller variants. Work-conserving may change ownership with execution progress; ownership and per-core bytes are recorded.', '',
      '| Input | Organization | Dispatch | Cohort time (us) / entry bound | Charged N3 time (us) / entry bound |',
      '|---|---|---|---:|---:|']
    values={r['case']:r for r in rows if r['repeat']==1}
    for workload in ['b8','b32']:
        for org in ['single','homogeneous','heterogeneous']:
            for policy in ['threshold','work_conserving']:
                a,z=[values[f'{workload}__{org}__{policy}__{mode}'] for mode in ['cohort','charged_n3']]
                text.append(f"| {workload} | {org} | {policy} | {a['time_us']:.6f} / {a['lower_bound_ratio']:.4f} | {z['time_us']:.6f} / {z['lower_bound_ratio']:.4f} |")
    text+=['','Entry bounds are B8 **235.008 us**, B32 **317.952 us**, from native bytes divided by 8*32 B/ns. They are unchanged across organizations. Total work stays fixed; padded issued MACs can differ with organization/assignment.', '',
      '## Me1 calibration and fixed aging references','',
      f"Me1 single-core cohort time: **{values['me1__single__threshold__cohort']['time_us']:.6f} us**. The Step2 target remains **<=26.5 us**, compared with frozen zero-control N3 31.124 us, not this new baseline.", '',
      'L is measured from tile load issue through all native responses, MX decode and BF16 placement, before the stationary operand-copy event. It is a per-core arithmetic mean over completed tiles. For each Step2 case, use its corresponding Step1 cohort case/core L, frozen across W and aging scans. No Step2 result recalibrates L. A core with zero tiles has no sample; none is silently filled with another core\'s latency.', '',
      '| Cohort case | Core | Tiles | Mean L (ns) | Min / max (ns) |','|---|---|---:|---:|---:|']
    for r in cores:
        if r['case'].endswith('__cohort'):
            L='n/a' if r['L_ps'] is None else f"{r['L_ps']/1000:.3f}"
            text.append(f"| {r['case']} | {r['core']} | {r['tiles']} | {L} | {r['load_min_ps']/1000:.3f} / {r['load_max_ps']/1000:.3f} |")
    text+=['','All resource/numerical/native-drain/repetition/provenance gates passed. This table is the immutable Step1 reference for later mechanism attribution. Configuration shapes, budgets and original full results accompany the table. Step1 Me32 accumulator feedback wait increased 0.519->1.587 us after cohort grouping; retain it as a Step4 local-partial-sum target, without changing it in Step2.','']
    (out/'REPORT.md').write_text('\n'.join(text))
    m.update(status='complete',completed_runs=len(rows),failed_invariants=0,hbm_bytes_changed_runs=0)
    b.save(out/'manifest.json',m)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    out=a.output.resolve();m=b.read(out/'manifest.json') if (out/'manifest.json').exists() else prepare(out)
    for name,h in m['artifacts'].items():b.require(b.digest(out/'repro'/name)==h,'archived artifact changed')
    validator=diag.load_validator(out/'repro/compare_moe_normal.py');rows=[];cores=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(run_case,c,out,m,validator):c for c in m['cases']}
        for f in concurrent.futures.as_completed(futures):
            rr,cc=f.result();rows+=rr;cores+=cc
            print(f"PASS {futures[f]['id']}: {rr[0]['time_us']:.6f} us",flush=True)
    rows.sort(key=lambda r:(r['case'],r['repeat']));cores.sort(key=lambda r:(r['case'],r['core']))
    report(out,m,rows,cores)
    print('COMPLETE',len(rows),'runs',flush=True)

if __name__=='__main__':main()
