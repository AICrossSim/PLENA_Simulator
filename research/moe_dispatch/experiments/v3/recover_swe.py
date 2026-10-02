#!/usr/bin/env python3
"""Recover original SWE canonical requests; reject any byte-hash mismatch."""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd
import requests

REPO='princeton-nlp/SWE-bench_bm25_27K'
REVISION='d4667abc38f15bccd9a99ba1aa56fc635477d274'
EXPECTED='c6d11a02b328689daf16806a39d1f68805313f6561a20d7c024d756ec59022d5'
PARQUET_SHA='39561931d28a8fb9cf767623b7d09a5f491ec06bd9feebf49b34516d9af3815b'

def main():
 p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);a=p.parse_args();root=Path(a.output_root);root.mkdir(parents=True,exist_ok=True)
 parquet=root/'test.parquet';url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/test-00000-of-00001.parquet'
 if not parquet.exists():
  with requests.get(url,stream=True,timeout=120) as res:
   res.raise_for_status()
   with parquet.open('wb') as handle:
    for chunk in res.iter_content(1048576):handle.write(chunk)
 assert hashlib.sha256(parquet.read_bytes()).hexdigest()==PARQUET_SHA,'Source parquet changed'
 rows=[]
 for position,row in enumerate(pd.read_parquet(parquet).to_dict('records')):
  prompt=row.get('text') or row.get('input') or max((v for v in row.values() if isinstance(v,str)),key=len,default='')
  if not prompt:continue
  rows.append({'benchmark':'swe_bench','sample_id':str(row.get('instance_id') or row.get('id') or f'swe_{position}'),'category':str(row.get('repo') or 'unknown_repo'),'messages':[{'role':'user','content':'Analyze the following repository issue and retrieved code. Explain the root cause and propose a concrete patch.\n\n'+prompt}],'tools':[],'source_position':position})
 rows.sort(key=lambda row:row['sample_id']);assert len(rows)==2294
 data=''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows).encode();actual=hashlib.sha256(data).hexdigest();assert actual==EXPECTED,'Cannot establish exact original request content'
 output=root/'swe_bench_full_test_2294.jsonl'
 if output.exists():assert output.read_bytes()==data
 else:output.write_bytes(data)
 receipt={'dataset':REPO,'revision':REVISION,'source_url':url,'source_parquet_sha256':PARQUET_SHA,'rows':len(rows),'canonical_file':str(output.resolve()),'canonical_sha256':actual,'expected_original_canonical_sha256':EXPECTED,'exact_match':True,'prepare_source':'/scratch/shared/mcl123/plena/analysis/multivendor_moe_capture/prepare_workloads.py'}
 (root/'recovery_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
