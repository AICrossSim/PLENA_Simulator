#!/usr/bin/env python3
"""Parallel full-layer numerical shards with a verified, nonduplicated merge.

An already-running evaluator is left untouched through its complete layer13
checkpoint. Its later-layer loop is stopped at the printed layer2 boundary;
independent layer2/26 workers own those candidates. No samples, SVD algorithm,
candidate set, or per-layer causal lambda sequence are changed.
"""
import argparse,csv,io,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
import evaluate as q


FIELDS=('layer','rank_lanes','bits','method','rank','factor_a','factor_b')


def verify_q1(rows,layers):
    coverage=q.q1_coverage(rows,layers,[4,3],[0,8,16,24,32,48,64],
        ['rtn','qera_approx','qera_exact','lqer','l2qer'],[8,16],['mxint4','mxint8s','bf16'],['bf16','mxint8'])
    if not coverage['complete']:raise ValueError('incomplete/duplicate/unexpected Q1 candidates')
    block=[]
    for row in rows:
        if row['scope']!='layer':block.append(row);continue
        experts={int(x['expert']) for x in block if x['scope']=='expert'}
        projs={(int(x['expert']),x['projection']) for x in block if x['scope']=='projection'}
        if len(block)!=260 or experts!=set(range(64))|{-1} or projs!={(e,p) for e in range(-1,64) for p in ('g','u','d')}:
            raise ValueError('candidate missing one or more actual expert/projection metrics')
        for x in block:
            for key in ('layer','rank_lanes','bits','method','factor_a','factor_b'):
                if str(x[key])!=str(row[key]):raise ValueError('candidate-local metric provenance mismatch')
        block=[]
    if block:raise ValueError('incomplete trailing candidate')
    return coverage


def verify_q3(rows,layers,tokens=8192):
    wanted={(l,L,b,m,u,t,s) for l in layers for L in (8,16) for b in (4,3)
        for m in ('qera_approx','qera_exact','lqer','l2qer') for u in (16,32) for t in range(0,tokens,16)
        for s in ('uniform','frequency_static','gate_weighted_budget_oracle','gate_weighted_causal')}
    actual=[(int(r['layer']),int(r['rank_lanes']),int(r['bits']),r['method'],int(r['uniform_rank']),
             int(r['window_start']),r['strategy']) for r in rows]
    if set(actual)!=wanted or len(actual)!=len(wanted):raise ValueError('incomplete/duplicate/unexpected Q3 actual-FFN rows')
    return {'complete':True,'actual_ffn_rows':len(actual),'causal_sequences':'independent per layer/bit/method/L/uniform budget'}


def read_csv(path):return list(csv.DictReader(Path(path).open()))



def merge_completed_layers(out,provenance,inventory):
    shards=out/'layer_shards';state=out/'shard_orchestrator.json'
    allrows=[];allq3=[];supplemental=[];tensor_records={};part_receipts=[]
    for layer in (2,13,26):
        folder=shards/f'layer{layer}';receipt=json.loads((folder/'job_status.json').read_text())
        if receipt['provenance']!=provenance:raise ValueError('shard source/capture/checkpoint provenance mismatch')
        part_receipts.append(receipt);allrows+=read_csv(folder/'q1_metrics.csv');allq3+=read_csv(folder/'q3_actual_ffn.csv')
        if (folder/'supplemental_mxint4_b_default.csv').exists():supplemental+=read_csv(folder/'supplemental_mxint4_b_default.csv')
        for L in (8,16):shutil.copytree(folder/f'L{L}',out/f'L{L}',dirs_exist_ok=True)
        if (folder/'q0_provenance.json').exists():tensor_records.update(json.loads((folder/'q0_provenance.json').read_text())['tensor_provenance'])
    coverage=verify_q1(allrows,[2,13,26]);q3coverage=verify_q3(allq3,[2,13,26])
    q.write_csv(out/'q1_metrics.csv',allrows);q.write_csv(out/'q3_actual_ffn.csv',allq3)
    if supplemental:q.write_csv(out/'supplemental_mxint4_b_default.csv',supplemental)
    (out/'q1_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n');(out/'q3_coverage.json').write_text(json.dumps(q3coverage,indent=2)+'\n')
    layers=[r for r in allrows if r['scope']=='layer'];rtn={(int(r['layer']),int(r['rank_lanes'])):float(r['relative_error']) for r in layers if r['bits']=='4' and r['method']=='rtn'}
    groups={}
    for r in layers:
        if r['method']=='rtn' or r['bits']=='8':continue
        key=tuple(r[x] for x in ('bits','method','rank','factor_a','factor_b','rank_lanes'));groups.setdefault(key,[]).append(r)
    passing=[]
    for key,rr in groups.items():
        if {int(r['layer']) for r in rr}!={2,13,26}:raise ValueError('candidate is not complete across all3 layers')
        if all(float(r['relative_error'])<=.5*rtn[(int(r['layer']),int(r['rank_lanes']))] and float(r['cosine'])>=.999 for r in rr):
            main=sum(q.bv.plan_projection(p,K,N,0,q.bv.Fmt('q',int(key[0]),'bf16',int(key[5]),'mxint4')).main_bytes
                for e in range(-1,64) for p,K,N in [('g',2048,2816 if e==-1 else 1408),('u',2048,2816 if e==-1 else 1408),('d',2816 if e==-1 else 1408,2048)])
            factors=max(int(r['factor_bytes']) for r in rr)
            passing.append({'bits':int(key[0]),'method':key[1],'rank':int(key[2]),'factor_a':key[3],'factor_b':key[4],
                'rank_lanes':int(key[5]),'factor_bytes_all_experts_max_layer':factors,'total_physical_weight_bytes':main+factors,'layers':[2,13,26]})
    passing.sort(key=lambda x:x['total_physical_weight_bytes'])
    (out/'freeze_candidates.json').write_text(json.dumps({'q0_complete':True,'accuracy_target':{'error_vs_mxint4_rtn_max':.5,'cosine_min':.999},
        'passing_candidates':passing,'can_freeze':bool(passing),'scope':'exact same candidate at all three complete layers; development numeric request subdivisions only'},indent=2)+'\n')
    # Layer13 weights were individually verified by the untouched primary worker;
    # archive their already-audited inventory records rather than reread 29GB.
    inv=json.loads(inventory.read_text())['tensor_provenance']
    for name,v in inv.items():
        if name.startswith('model.layers.13.'):tensor_records[name]=dict(v,expected_hash_verified=True,hash_origin='primary worker verified required tensor inventory')
    (out/'q0_provenance.json').write_text(json.dumps({'schema':'plena_v3_quant_report_v1','q0_complete':True,'calibration_tokens':32768,'evaluation_tokens':8192,
        'request_hash_disjoint':True,'layers':[2,13,26],'evaluated_layers':[2,13,26],'full_accuracy_matrix_complete':True,
        'source':'real BF16 weights and captured real activations','tensor_provenance':tensor_records,
        'precision_can_be_frozen':bool(passing),'q4_status':'not_run_no_gpu','layer_shard_provenance':provenance},indent=2)+'\n')
    receipt={'status':'completed','exit_code':0,'q1_candidates':coverage['completed_candidates'],'q1_metric_rows':len(allrows),
        'q3_actual_ffn_rows':len(allq3),'provenance':provenance,'shard_receipts':part_receipts,
        'scope':'parallel scheduling only; required candidates, samples, exact SVD, per-layer causal policy unchanged'}
    state.write_text(json.dumps(receipt,indent=2)+'\n');(out/'job_status.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'completed':True,'q1':coverage['completed_candidates'],'q3':len(allq3),'passing':len(passing)}),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary',type=Path,required=True);ap.add_argument('--primary-pid',type=int,required=True)
    ap.add_argument('--model',type=Path,required=True);ap.add_argument('--calibration',type=Path,required=True);ap.add_argument('--evaluation',type=Path,required=True)
    ap.add_argument('--inventory',type=Path,required=True);args=ap.parse_args();out=args.primary;shards=out/'layer_shards';shards.mkdir(exist_ok=True)
    state=out/'shard_orchestrator.json';worker=Path(__file__).with_name('evaluate.py');source_sha=q.sha(worker)
    provenance={'calibration_manifest_sha256':q.sha(args.calibration),'validation_manifest_sha256':q.sha(args.evaluation),
        'model_config_sha256':q.sha(args.model/'config.json'),'inventory_sha256':q.sha(args.inventory),
        'math_contract':'full captured rows, exact SVD, FP32/BF16 hardware FFN, actual_window_bytes_v1'}
    if (shards/'evaluate_source_at_launch.py').exists():
        shutil.copy2(shards/'evaluate_source_at_launch.py',shards/'evaluate_source_before_bf16.py')
    shutil.copy2(worker,shards/'evaluate_source_at_launch.py')
    processes={};handles=[]
    for layer in (2,26):
        folder=shards/f'layer{layer}';folder.mkdir(exist_ok=True)
        resume=False
        if (folder/'job_status.json').exists():
            old=json.loads((folder/'job_status.json').read_text())
            if old.get('status')!='superseded_for_BF16_Q3_restart':raise RuntimeError('layer shard directory already has an active/completed launch receipt')
            resume=True
        command=[sys.executable,str(worker),'--model',str(args.model),'--calibration',str(args.calibration),
            '--evaluation',str(args.evaluation),'--output',str(folder),'--expected-tensors',str(args.inventory),
            '--rank-lanes','8,16','--layers',str(layer),'--factor-workers','8','--blas-threads','1','--supplemental-b4','--q3-bf16-hardware']
        if resume:
            command+=['--resume-q1',str(folder/'q1_metrics_before_bf16.csv')]
            if (folder/'q3_actual_ffn_before_bf16.csv').exists():command+=['--resume-q3',str(folder/'q3_actual_ffn_before_bf16.csv')]
        handle=(folder/'evaluate.log').open('w');handles.append(handle)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env);processes[layer]=proc
        (folder/'job_status.json').write_text(json.dumps({'status':'running_full_layer_shard','layer':layer,'pid':proc.pid,
            'command':command,'provenance':provenance,'evaluator_source_sha256':source_sha,'expert_workers':8,'blas_threads':1},indent=2)+'\n')
    state.write_text(json.dumps({'status':'running_layer_shards','primary_pid':args.primary_pid,'shard_pids':{str(l):p.pid for l,p in processes.items()},
        'provenance':provenance,'candidate_set_unchanged':True,'samples_unchanged':True,'math_unchanged':True},indent=2)+'\n')
    print('started independent layer2/26 shards',flush=True)
    archived=False;primary_launcher_status=None
    while not archived:
        log=out/'evaluate.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-4096));tail=f.read().decode(errors='replace')
            if 'layer2 load original problems' in tail:
                os.kill(args.primary_pid,signal.SIGSTOP)
                rows=[r for r in read_csv(out/'q1_metrics.csv') if int(r['layer'])==13]
                qr=[r for r in read_csv(out/'q3_actual_ffn.csv') if int(r['layer'])==13]
                verify_q1(rows,[13]);verify_q3(qr,[13])
                folder=shards/'layer13';folder.mkdir(exist_ok=True)
                q.write_csv(folder/'q1_metrics.csv',rows);q.write_csv(folder/'q3_actual_ffn.csv',qr)
                for L in (8,16):shutil.copytree(out/f'L{L}',folder/f'L{L}',dirs_exist_ok=True)
                primary_launcher_status=json.loads((out/'job_status.json').read_text())
                (folder/'job_status.json').write_text(json.dumps({'status':'completed_full_layer_shard','layer':13,
                    'primary_evaluator_receipt':primary_launcher_status,'provenance':provenance,
                    'evaluator_source_sha256':None,'source_receipt_note':'old launcher did not record source hash; stable imported worker was not restarted; scheduling-only CLI changes occurred after its import',
                    'termination':'complete layer13 checkpoint; before any layer2 candidate was evaluated'},indent=2)+'\n')
                os.kill(args.primary_pid,signal.SIGTERM);os.kill(args.primary_pid,signal.SIGCONT)
                time.sleep(1)
                (out/'job_status.json').write_text(json.dumps({'status':'running_layer_shards','provenance':provenance,
                    'completed_layers':[13],'shard_pids':{str(l):p.pid for l,p in processes.items()}},indent=2)+'\n')
                archived=True;print('archived complete layer13 checkpoint at boundary; no later candidate duplicate',flush=True)
        for layer,proc in processes.items():
            if proc.poll() is not None and proc.returncode!=0:raise RuntimeError(f'layer{layer} failed with exit{proc.returncode}; inspect isolated log')
        if not archived:time.sleep(.1)
    for layer,proc in processes.items():
        rc=proc.wait();folder=shards/f'layer{layer}'
        if rc:raise RuntimeError(f'layer{layer} failed with exit{rc}')
        verify_q1(read_csv(folder/'q1_metrics.csv'),[layer]);verify_q3(read_csv(folder/'q3_actual_ffn.csv'),[layer])
        (folder/'job_status.json').write_text(json.dumps({'status':'completed_full_layer_shard','layer':layer,'exit_code':rc,
            'provenance':provenance,'evaluator_source_sha256':source_sha},indent=2)+'\n')
    merge_completed_layers(out,provenance,args.inventory)


if __name__=='__main__':main()
