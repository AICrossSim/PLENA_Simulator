#!/usr/bin/env python3
"""Emit diagnostic physical-format freeze after exhaustive first-layer rejection."""
import argparse,datetime,hashlib,json,time
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proof',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    while not args.proof.exists():
        if not args.wait:raise FileNotFoundError(args.proof)
        time.sleep(15)
    proof=json.loads(args.proof.read_text())
    if not proof['coverage']['complete'] or proof['coverage']['completed_candidates']!=726:raise ValueError('first-layer matrix not exhaustive')
    if not proof['all_three_layer_candidates_eliminated'] or proof['passing_candidates']:raise ValueError('first-layer rejection does not justify a diagnostic default freeze')
    config={'main_bits':4,'factor_a':'mxint4','factor_b':'bf16','rank_lanes':8,'ranks':{'routed':[32,32,24],'shared':[32,32,48]}}
    receipt={'schema':'plena_v3_physical_format_freeze_v1','config':config,
        'quality_status':'diagnostic_non_accuracy_preserving','accuracy_preserving_qualified':False,
        'qualification':{'measured':'all726 requested layer13 candidates evaluated on real numerical-validation data',
            'deduced':'no candidate can satisfy the same all-three-layer conjunction because none passes layer13',
            'not_claimed':'full three-layer matrix completion, model perplexity, downstream accuracy, or impossibility of future methods'},
        'quality_receipt_path':str(args.proof.resolve()),'quality_receipt_sha256':sha(args.proof),
        'physical_format_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n').hexdigest(),
        'accuracy_targets':proof['targets'],'full_three_layer_candidates_expected':2178,'full_matrix_continues':True,
        'numeric_split':'request-hash-disjoint32k/8k subdivisions within development identities; heldout folder is not architectural timing heldout',
        'gold_contract':'supplied expert_ffn_hw BF16X/U/Z plus FP32 accumulation and gate folding; not HF end-to-end model accuracy',
        'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if old.get('config')!=config:raise ValueError('refusing to overwrite a different physical configuration')
        if old.get('quality_receipt_sha256')==receipt['quality_receipt_sha256']:
            print(json.dumps({'unchanged':True,'path':str(args.output),'sha256':sha(args.output)}));return
        backup=args.output.with_name(args.output.stem+'_before_complete_layer13'+args.output.suffix)
        if not backup.exists():backup.write_bytes(args.output.read_bytes())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'path':str(args.output.resolve()),'sha256':sha(args.output),'quality_receipt_sha256':receipt['quality_receipt_sha256'],'physical_format_sha256':receipt['physical_format_sha256'],'quality_status':receipt['quality_status']}),flush=True)


if __name__=='__main__':main()
