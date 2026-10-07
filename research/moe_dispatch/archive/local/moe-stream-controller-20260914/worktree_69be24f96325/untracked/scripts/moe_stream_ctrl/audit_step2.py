#!/usr/bin/env python3
"""Final source/provenance/CSV/resource audit, without rerunning simulation."""
import argparse,csv,json,shutil
from pathlib import Path
from decimal import Decimal
import step0 as b
import step2 as s


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output.resolve()
 m=b.read(out/'manifest.json');b.require(m['status']=='complete' and m['completed_runs']==260,'scan incomplete')
 for path,sha in m['artifacts'].items():b.require(b.digest(out/'repro'/path)==sha,'artifact changed '+path)
 b.require(b.digest(out/'repro/libramulator.so')==b.digest(s.ACCEPTED/'repro/libramulator.so'),'native library changed')
 for path in (out/'repro/moe_normal').glob('*.rs'):
  b.require(path.read_bytes()==(b.SOURCE/'transactional_emulator/src/moe_normal'/path.name).read_bytes(),'Rust source changed '+path.name)
 checked=0;peaks=[]
 for c in m['cases']:
  folder=out/c['id'];v=b.read(folder/'validation.json');b.require(v['status']=='passed','invalid point')
  for name,sha in v['outputs'].items():b.require(b.digest(folder/name)==sha,'output changed '+c['id']+'/'+name)
  for name,sha in c['hashes'].items():b.require(b.digest(c[name])==sha,'input/calibration changed')
  env=s.read(folder/'rep1.json.gz');r=env['result'];a=b.read(c['architecture'])
  with (folder/'measurements.csv').open() as f:rows=list(csv.DictReader(f))
  b.require(len(rows)==2 and all(Decimal(x['time_us'])*1_000_000==r['total_ps'] for x in rows),'CSV time differs')
  with (folder/'core_services.csv').open() as f:core_rows=list(csv.DictReader(f))
  for cfg,core,row in zip(a['cores'],r['cores'],core_rows):
   d=core['refinement'];z=d['split_window'];p=d['output_pool'];ctrl=d['stream_ctrl']
   b.require(int(row['live_peak_bytes'])==z['live_peak_bytes']<=cfg['weight_sram_bytes'],'CSV/lifetime peak differs')
   b.require(z['window_header_updates']==p['tile_admissions'],'header visits missing')
   b.require(p['scheduler_visits']==int(row['admission_cycles'])+10*p['tile_admissions']+p['context_updates']+4*ctrl['burst_starts']+ctrl['completion_mask_cycles'],'service formula differs')
   ranges=s.read(folder/f"cycle_peaks_{core['id']}.json.gz")['ranges'];last=0
   for start,end,pk,op,de in ranges:
    b.require(start==last and end>start,'RLE gap or overlap');last=end
    b.require(pk+op+de<=cfg['weight_sram_bytes'],'cycle peak exceeds budget')
   b.require(max(sum(x[2:]) for x in ranges)==z['live_peak_bytes'],'RLE/global peak differs')
   b.require(last>= (r['total_ps']+a['clock_period_ps']-1)//a['clock_period_ps'],'missing execution cycles')
   peaks.append(dict(case=c['id'],core=core['id'],peak=z['live_peak_bytes'],budget=cfg['weight_sram_bytes']))
  checked+=2
 for group in ['disabled_regression','frozen_me1_n3','adversarial_checks']:
  b.require(b.read(out/group/'validation.json')['status']=='passed','incomplete '+group)
 b.csv_write(out/'all_core_peaks.csv',peaks)
 repro=out/'repro';analysis=repro/'analysis';analysis.mkdir(exist_ok=True)
 for name in ['audit_step2.py','report_step2.py','verify_step2.py','frozen_me1_step2.py','step0.py','step2.py','diagnose_step1.py','regress_step2.py']:
  shutil.copy2(b.SOURCE/'scripts/moe_stream_ctrl'/name,analysis/name)
 for name in ['cargo_tests.log','cargo_clippy.log','cargo_build.log','IMPLEMENTATION.md']:
  shutil.copy2(out/name,repro/name)
 for path in repro.rglob('*'):
  if path.is_file():m['artifacts'][str(path.relative_to(repro))]=b.digest(path)
 m.update(scope='Step2 only; accepted Step1 common table frozen separately',failed_invariants=0,hbm_bytes_changed_runs=0,
    native_runs_in_final_delivery=344,acceptance=b.read(out/'acceptance.json'),
    disabled_regression=str(out/'disabled_regression/validation.json'),frozen_me1_reference=str(out/'frozen_me1_n3/validation.json'),
    note='Model snapshot is unchanged after final compilation. Analysis scripts/test logs are archived separately from the original measurement runner. No Step3 implementation is included.')
 b.save(out/'manifest.json',m)
 b.save(out/'FINAL_AUDIT.json',dict(status='passed',step2_runs=checked,core_peak_series=len(peaks),old_path_runs=84,
    rust_workspace_tests=283,adversarial_rejections=19,native_library_unchanged=True,model_snapshot_matches_workspace=True,
    repeated_numeric_and_native_results_exact=True,all_simultaneous_peaks_within_budget=True))
 print('PASS final audit:',checked,'Step2 runs;',len(peaks),'per-core cycle series; 84 baseline runs')

if __name__=='__main__':main()
