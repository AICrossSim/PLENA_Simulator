#!/usr/bin/env python3
"""Prepare and replay a no-prefetch, multi-layer Rust FFN baseline.

Captured adjacent-layer routes require the original .npz file. A workload
bundle can instead be supplied for protocol tests. External inter-layer time
and HBM bytes must be supplied, never inferred from one FFN's duration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

import joint_study
import run_experiments as run
from robust_workloads import build_window
from frontend import compiler

ARCHIVE=Path(__file__).resolve().parent/'results/joint_runtime_20260930'
SHAPES={'6':[6],'3+3':[3,3],'4+2':[4,2]}


def captured_layers(path:Path, layer_ids:list[int], step:int, batch:int):
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(a['meta'].item())
        available=set(map(int,a['layer_ids']))
        assert all(layer in available for layer in layer_ids)
        assert all(b==a+1 for a,b in zip(layer_ids,layer_ids[1:])), 'layers must be adjacent'
        excluded=set(json.loads((ARCHIVE/'workload_manifest.json').read_text())['excluded_request_ids'])
        manifest=json.loads((ARCHIVE/'workload_manifest.json').read_text())
        for w in manifest['windows']:
            excluded.update(w['sample_ids'])
        candidates=[]
        for i,sid in enumerate(a['sample_ids']):
            ident=str(sid)
            if bool(a['valid'][i,step]) and not any(ident == x or x.endswith(':'+ident) for x in excluded):
                key=hashlib.sha256(('plena-multilayer-v1:'+ident).encode()).hexdigest()
                candidates.append((key,i,ident))
        candidates.sort()
        assert len(candidates)>=batch, 'not enough unused valid captured requests'
        chosen=candidates[:batch]
        indices=[item[1] for item in chosen]
        workloads=[build_window(a,meta,indices,step,layer,
            f'multilayer_l{layer}_s{step}_b{batch}') for layer in layer_ids]
        # The single-layer route builder uses expert-local placeholder weight
        # addresses. Consecutive layers have different weights; give each a
        # disjoint address namespace before the Compiler plans DMA requests.
        layer_stride=(meta['routed_experts']+1)*3*compiler.EXPERT_PHASE_STRIDE
        for layer,w in zip(layer_ids,workloads):
            for expert in w['experts']:
                for weight in expert['weights'].values():
                    weight['hbm_base'] += layer*layer_stride
        source=dict(path=str(path.resolve()),sha256=run.file_sha(path),
            sample_ids=[item[2] for item in chosen],layer_ids=layer_ids,step=step,
            offline_rebatch=True,layer_weight_address_stride_bytes=layer_stride,
            weight_addresses_are_shape_only=True)
    return workloads,source


def prepare(args):
    if args.toy:
        assert not args.capture and not args.workloads_json
        first=run.toy_workloads()[0]
        second=copy.deepcopy(first)
        second['id']=first['id']+'_synthetic_next'
        workloads=[first,second]
        source=dict(synthetic=True,note='two toy FFNs for protocol validation only; no real adjacent layers')
    elif args.capture:
        layers=[int(x) for x in args.layer_ids.split(',')]
        assert len(layers)>=2
        workloads,source=captured_layers(args.capture,layers,args.step,args.batch)
    else:
        assert args.workloads_json, 'provide --capture or --workloads-json'
        payload=json.loads(args.workloads_json.read_text())
        workloads=payload['workloads']
        assert len(workloads)>=2
        source=dict(path=str(args.workloads_json.resolve()),sha256=run.file_sha(args.workloads_json),
            note='caller-supplied workloads; adjacent real layers are not verified')
    designs=json.loads((ARCHIVE/'frozen_hardware.json').read_text())
    design=next(d for d in designs if d['budget_group']=='M6' and d['lanes']==SHAPES[args.shape])
    cfg=joint_study.configuration(design,'joint')
    if args.policy=='ipd':
        cfg.update(dispatch='ipd',ipd_credit_quotas=True)
    cfg.update(profile_bin_cycles=args.profile_bin_cycles,record_trace=False)
    assert cfg['credits']==256 and cfg['hbm_bytes_per_ns']==256
    rows=[]
    for i,w in enumerate(workloads):
        # Match the single-layer experiment's descriptor order and exact
        # frozen compiler resources for each independent layer.
        w=joint_study.reorder(w)
        w['engine_layout']=compiler.engine_layout(w,design['lanes'],design['group'],design['resources'])
        gap=args.gap_cycles if i+1<len(workloads) else 0
        gap_bytes=args.gap_hbm_bytes if i+1<len(workloads) else 0
        rows.append(dict(workload=w,gap_after_cycles=gap,gap_hbm_bytes=gap_bytes))
    request=dict(schema='plena_moe_multilayer_baseline_input_v1',config=cfg,layers=rows,
                 opportunity_tail_cycles=args.opportunity_tail_cycles)
    args.output.mkdir(parents=True,exist_ok=True)
    run.write_json(args.output/'input.json',request)
    run.write_json(args.output/'source.json',dict(source=source,design=design['id'],
        input_sha256=run.file_sha(args.output/'input.json'),
        source_bundle_sha256=run.digest(run.source_inventory()),
        gap_origin='caller-provided; zero means no modeled Attention or Router interval',
        timing_scope='sequential analytical FFNs, no cross-layer prefetch'))
    print(json.dumps(dict(layers=len(rows),shape=args.shape,policy=args.policy,
        input=str(args.output/'input.json')),sort_keys=True))


def replay(args):
    binary=args.binary.resolve()
    assert binary.is_file()
    input_path=args.output/'input.json'
    source=json.loads((args.output/'source.json').read_text())
    assert run.file_sha(input_path)==source['input_sha256']
    assert run.digest(run.source_inventory())==source['source_bundle_sha256'], 'sources changed after preparation'
    reports=[]
    for repeat in (1,2):
        path=args.output/f'report_repeat{repeat}.json'
        subprocess.run([str(binary),'--multilayer-baseline',str(input_path),str(path)],check=True)
        reports.append(path)
    assert run.file_sha(reports[0])==run.file_sha(reports[1]), 'non-deterministic baseline'
    data=json.loads(reports[0].read_text())
    run.write_json(args.output/'receipt.json',dict(binary_sha256=run.file_sha(binary),
        input_sha256=source['input_sha256'],report_sha256=run.file_sha(reports[0]),
        source_bundle_sha256=source['source_bundle_sha256'],
        repeats_equal=True,layers=len(data['layers']),no_cross_layer_prefetch=True,
        total_cycles=data['total_cycles']))
    print(json.dumps(dict(layers=len(data['layers']),total_cycles=data['total_cycles'],
        configured_hbm_byte_utilization=data['configured_hbm_byte_utilization']),sort_keys=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','run'))
    p.add_argument('--capture',type=Path)
    p.add_argument('--workloads-json',type=Path)
    p.add_argument('--toy',action='store_true',help='reproducible two-layer protocol fixture')
    p.add_argument('--layer-ids',default='12,13,14')
    p.add_argument('--step',type=int,default=7)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--shape',choices=tuple(SHAPES),default='4+2')
    p.add_argument('--policy',choices=('joint','ipd'),default='joint')
    p.add_argument('--gap-cycles',type=int,default=0)
    p.add_argument('--gap-hbm-bytes',type=int,default=0)
    p.add_argument('--profile-bin-cycles',type=int,default=1024)
    p.add_argument('--opportunity-tail-cycles',type=int,default=65536)
    p.add_argument('--output',type=Path,default=Path('/tmp/plena-multilayer-baseline'))
    p.add_argument('--binary',type=Path,default=run.DEFAULT_BINARY)
    a=p.parse_args()
    assert a.gap_cycles>=0 and a.gap_hbm_bytes>=0 and a.profile_bin_cycles>0
    assert a.opportunity_tail_cycles>0
    if a.stage=='prepare':prepare(a)
    else:replay(a)

if __name__=='__main__':main()
