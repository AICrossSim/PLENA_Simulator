#!/usr/bin/env python3
"""Development-only native-wire compensation sensitivity; not a new N2 gate.

Six captured B2/B16 windows × (five P1 modes + two default-format P2 modes).
Original equal-total-wire runs are read from the active main campaign when ready;
only the 42 native-wire points are additional simulation, each repeated twice.
"""
from __future__ import annotations
import argparse,copy,json,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import study

def plans(root):
 ps=study.points(root,'development','compensation')
 selected=[p for p in ps if p['workload']['batch'] in (2,16) and not p['variant'].startswith('b4_')]
 assert len(selected)==42 and all(not p['unavailable_reason'] for p in selected)
 assert len({p['workload']['id'] for p in selected})==6
 out=[]
 for original in selected:
  native=copy.deepcopy(original);native.update(suite='compensation_native',key=original['key'].replace('compensation__','compensation_native__',1));native['config']['comp_equal_bytes']=False
  out.append((original,native))
 return out

def busy(r):return sum(sum(c.get('states',{}).get(f'C{x}',0) for x in (1,4,5,6,7)) for c in r['cores'])

def equal_report(root,point,signature):
 dest=Path(root)/'raw'/study.digest(signature)[:16]/point['key'];receipt=dest/'receipt.json'
 if not receipt.exists():return None
 value=json.loads(receipt.read_text());assert value['signature']==signature and value['repeats']==2 and value['bit_identical']
 data=study.rawbytes(dest,1);assert data==study.rawbytes(dest,2)
 assert study.hashlib.sha256(data).hexdigest()==value['sha256']
 report=json.loads(data);assert study.validate(report,point['config'])
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=study.DEFAULT_ROOT);p.add_argument('--binary',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--format-config',default='');p.add_argument('--report-only',action='store_true');a=p.parse_args()
 fmt=Path(a.format_config) if a.format_config else a.root/'frozen_format.json'
 if a.format_config or fmt.exists():study.load_format(fmt)
 physical=study.campaign_signature(a.root,a.binary)
 signature={**physical,'native_sensitivity_runner_sha256':study.sha(__file__)}
 outroot=a.root/'compensation_native';outroot.mkdir(exist_ok=True);planned=plans(a.root)
 reports={};equal={};receipts=[]
 def run(pair):
  original,native=pair
  if a.report_only:
   dest=outroot/'raw'/study.digest(signature)[:16]/native['key'];receipt=json.loads((dest/'receipt.json').read_text());assert receipt['signature']==signature
   data=study.rawbytes(dest,1);assert data==study.rawbytes(dest,2) and study.hashlib.sha256(data).hexdigest()==receipt['sha256']
   r=json.loads(data);assert study.validate(r,native['config']);item={'status':'complete','report':r,'receipt':receipt}
  else:item=study.run_point(native,outroot,a.binary,signature,3600)
  assert item['status']=='complete',f"{native['key']}: {item.get('reason')}"
  return original,native,item
 with ThreadPoolExecutor(max_workers=a.workers) as workers:
  for future in as_completed([workers.submit(run,pair) for pair in planned]):
   original,native,item=future.result();key=(original['workload']['id'],original['op'],original['variant'])
   reports[key]=item['report'];equal[key]=equal_report(a.root,original,physical)
   receipts.append({'point':native['key'],'original_json_sha256':item['receipt']['sha256']})
   print(len(reports),'/42 native-wire complete',flush=True)
 rows=[]
 for original,native in planned:
  w=original['workload'];key=(w['id'],original['op'],original['variant']);r=reports[key];e=equal[key]
  base=reports[(w['id'],original['op'],'none')];eb=equal[(w['id'],original['op'],'none')]
  row={'workload':w['id'],'tokens':w['batch'],'precision':native['config']['precision'],'op':original['op'],'mode':original['variant'],'main_bits':native['config']['main_bits'],'factor_a':native['config']['factor_a'],'factor_b':native['config']['factor_b'],'rank_lanes':native['config']['rank_lanes'],'native_cycles':r['cycles'],'native_ms':r['cycles']/1e6,'native_wire_bytes':r['weight_bytes'],'native_padding_bytes':r['comp_padding_bytes'],'native_sum_core_busy_cycles':busy(r),'native_layer_overhead_vs_none':r['cycles']/base['cycles']-1,'native_busy_overhead_vs_none':busy(r)/busy(base)-1,'equal_total_wire_available':e is not None,'scope':'Six development windows only; native payload layout and compensation differ; diagnostic sensitivity, not a replacement N2 threshold or pure arithmetic isolation'}
  assert row['native_padding_bytes']==0
  if e is not None:
   row.update(equal_total_wire_cycles=e['cycles'],equal_total_wire_ms=e['cycles']/1e6,equal_total_wire_bytes=e['weight_bytes'],equal_padding_bytes=e['comp_padding_bytes'],equal_sum_core_busy_cycles=busy(e),equal_vs_native_layer_difference=e['cycles']/r['cycles']-1,equal_vs_native_busy_difference=busy(e)/busy(r)-1)
   if eb is not None:row.update(equal_layer_overhead_vs_none=e['cycles']/eb['cycles']-1,equal_busy_overhead_vs_none=busy(e)/busy(eb)-1,layer_overhead_shift_due_to_wire_matching=(e['cycles']/eb['cycles']-1)-(r['cycles']/base['cycles']-1))
  rows.append(row)
 study.csvwrite(outroot/'native_wire_sensitivity.csv',rows)
 study.write(outroot/'receipt.json',{'signature':signature,'quality_receipt_sha256':study.FORMAT_RECEIPT['sha256'] if study.FORMAT_RECEIPT else None,'native_points':42,'native_raw_runs':84,'native_all_valid_bit_identical':True,'equal_total_wire_pairs_observed':sum(e is not None for e in equal.values()),'complete_equal_comparison':all(e is not None for e in equal.values()),'heldout_executed':False,'changes_to_main_inventory':False,'no_new_N2_gate':True,'original_json_receipts':sorted(receipts,key=lambda r:r['point'])})
 print('Complete native-wire sensitivity; equal-wire comparison is complete only when all main development raw receipts are present',flush=True)
if __name__=='__main__':main()
