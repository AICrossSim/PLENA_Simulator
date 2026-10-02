#!/usr/bin/env python3
"""Recover exact completed numerical checkpoints after host ENOSPC.

Discard only unfinished CSV tails, preserve failed original bytes, and evaluate
every missing requested candidate on the same captures and mathematical code.
"""
import argparse,csv,io,json,math,os,shutil,subprocess,sys,time
from collections import defaultdict
from pathlib import Path
import evaluate as q
from run_layer_shards import verify_q1,verify_q3,merge_completed_layers


def complete_lines(path):
    raw=path.read_bytes();end=raw.rfind(b'\n')
    return raw,list(csv.DictReader(io.StringIO(raw[:end+1].decode()))) if end>=0 else []


def partial_q1(rows,layer):
    kept=[];block=[];candidates=set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):break
        if row['scope']!='layer':block.append(row);continue
        if int(row['layer'])!=layer or len(block)!=260:raise ValueError('invalid complete candidate block')
        experts={int(x['expert']) for x in block if x['scope']=='expert'}
        projs={(int(x['expert']),x['projection']) for x in block if x['scope']=='projection'}
        if experts!=set(range(64))|{-1} or projs!={(e,p) for e in range(-1,64) for p in ('g','u','d')}:raise ValueError('candidate expert/projection coverage failed')
        for x in block+[row]:
            for key in ('layer','rank_lanes','bits','method','factor_a','factor_b'):
                if x[key]!=row[key]:raise ValueError('candidate provenance differs within block')
            if not all(math.isfinite(float(x[k])) for k in ('relative_error','cosine','factor_bytes')):raise ValueError('nonfinite metric')
        ident=tuple(row[k] for k in ('layer','rank_lanes','bits','method','rank','factor_a','factor_b'))
        if ident in candidates:raise ValueError('duplicate completed candidate')
        candidates.add(ident);kept.extend(block+[row]);block=[]
    coverage=q.q1_coverage(kept,[layer],[4,3],[0,8,16,24,32,48,64],['rtn','qera_approx','qera_exact','lqer','l2qer'],[8,16],['mxint4','mxint8s','bf16'],['bf16','mxint8'])
    if coverage['unexpected_candidates'] or coverage['duplicate_candidates']:raise ValueError('unexpected recovered candidate')
    return kept,coverage


def partial_q3(rows,layer):
    groups=defaultdict(list)
    for row in rows:
        if None in row or any(v is None for v in row.values()):break
        groups[(int(row['layer']),int(row['bits']),row['method'],int(row['rank_lanes']))].append(row)
    wanted={(u,t,s) for u in (16,32) for t in range(0,8192,16) for s in ('uniform','frequency_static','gate_weighted_budget_oracle','gate_weighted_causal')}
    kept=[]
    for key,group in groups.items():
        if key[0]!=layer:raise ValueError('Q3 recovery contains another layer')
        keys=[(int(r['uniform_rank']),int(r['window_start']),r['strategy']) for r in group]
        if set(keys)==wanted and len(keys)==len(wanted):kept+=group
    return kept


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True);ap.add_argument('--evaluation',type=Path,required=True);ap.add_argument('--inventory',type=Path,required=True);args=ap.parse_args()
    out=args.output;shards=out/'layer_shards';source=Path(__file__).with_name('evaluate.py')
    provenance={'calibration_manifest_sha256':q.sha(args.calibration),'validation_manifest_sha256':q.sha(args.evaluation),'model_config_sha256':q.sha(args.model/'config.json'),'inventory_sha256':q.sha(args.inventory),'math_contract':'full captured rows, exact SVD, FP32/BF16 hardware FFN, actual_window_bytes_v1'}
    launches=[];processes={};handles=[]
    for layer in (13,2,26):
        folder=out if layer==13 else shards/f'layer{layer}';archive=folder/'failed_enospc_original';archive.mkdir(exist_ok=True)
        for name in ('q1_metrics.csv','q3_actual_ffn.csv','evaluate.log','job_status.json'):
            p=folder/name
            if p.exists() and not (archive/name).exists():shutil.copy2(p,archive/name)
        raw,rows=complete_lines(archive/'q1_metrics.csv');complete,coverage=partial_q1(rows,layer)
        q1=folder/'q1_recovered_complete.csv';q.write_csv(q1,complete)
        receipt=dict(provenance,q1_sha256=q.sha(q1),completed_candidates=coverage['completed_candidates'],original_failed_file_sha256=q.sha(archive/'q1_metrics.csv'),original_failed_bytes=len(raw),discarded_unfinished_rows=len(rows)-len(complete),reason='host ENOSPC recovery; exact complete261-row blocks only; no arithmetic/sample/candidate reduction')
        q1.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        _,qr=complete_lines(archive/'q3_actual_ffn.csv');qr=partial_q3(qr,layer);q3=folder/'q3_recovered_complete.csv';q.write_csv(q3,qr)
        q3.with_suffix('.receipt.json').write_text(json.dumps(dict(provenance,q3_sha256=q.sha(q3),rows=len(qr),policy_lambda_fit_version='actual_window_bytes_v1',reason='only complete actual-FFN Q3 groups reused'),indent=2)+'\n')
        command=[sys.executable,str(source),'--model',str(args.model),'--calibration',str(args.calibration),'--evaluation',str(args.evaluation),'--output',str(folder),'--expected-tensors',str(args.inventory),'--rank-lanes','8,16','--layers',str(layer),'--factor-workers','8','--blas-threads','2' if layer==13 else '1','--resume-q1',str(q1),'--resume-q3',str(q3)]
        # MXINT4-B default diagnostics were already completed and their original
        # files are retained. No diagnostic is silently regenerated or removed.
        if layer!=13:
            supplement=list(csv.DictReader((folder/'supplemental_mxint4_b_default.csv').open()))
            if len(supplement)!=1044 or len([r for r in supplement if r['scope']=='layer'])!=4:raise ValueError('existing supplementary defaults are incomplete')
            command+=['--q3-bf16-hardware']
        handle=(folder/'evaluate_after_enospc.log').open('w');handles.append(handle)
        env=dict(os.environ,TMPDIR='/tmp',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        p=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,env=env);processes[layer]=p
        item=dict(layer=layer,pid=p.pid,command=command,source_sha256=q.sha(source),verified_complete_candidates=coverage['completed_candidates'],verified_q3_rows=len(qr),provenance=provenance)
        launches.append(item)
        if layer!=13:(folder/'job_status.json').write_text(json.dumps(dict(item,status='running_exact_ENOSPC_recovery'),indent=2)+'\n')
        print(json.dumps({'launched':layer,'pid':p.pid,'complete_candidates':coverage['completed_candidates'],'q3_rows':len(qr)}),flush=True)
    (out/'enospc_recovery_launch.json').write_text(json.dumps({'schema':'plena_v3_exact_enospc_recovery_v1','status':'running','launches':launches,'source_changes':'atomic CSV publish; skip only verified complete method and Q3 checkpoints; arithmetic unchanged'},indent=2)+'\n')
    (out/'job_status.json').write_text(json.dumps({'status':'running_exact_ENOSPC_recovery','provenance':provenance,'workers':launches},indent=2)+'\n')
    pending=set(processes)
    while pending:
        for layer in sorted(pending):
            rc=processes[layer].poll()
            if rc is None:continue
            if rc:raise RuntimeError(f'layer{layer} recovery failed exit{rc}; preserve all checkpoints')
            source_folder=out if layer==13 else shards/f'layer{layer}'
            verify_q1(list(csv.DictReader((source_folder/'q1_metrics.csv').open())),[layer]);verify_q3(list(csv.DictReader((source_folder/'q3_actual_ffn.csv').open())),[layer])
            if layer==13:
                folder=shards/'layer13';folder.mkdir(exist_ok=True)
                for name in ('q1_metrics.csv','q3_actual_ffn.csv','q0_provenance.json'):shutil.copy2(out/name,folder/name)
                for L in (8,16):shutil.copytree(out/f'L{L}',folder/f'L{L}',dirs_exist_ok=True)
            else:folder=source_folder
            launch=next(x for x in launches if x['layer']==layer)
            (folder/'job_status.json').write_text(json.dumps(dict(launch,status='completed_full_layer_shard',exit_code=0),indent=2)+'\n')
            pending.remove(layer);print(f'layer{layer} complete all726 candidates and65536 softwareQ3 rows',flush=True)
        if pending:time.sleep(5)
    merge_completed_layers(out,provenance,args.inventory)
    (out/'enospc_recovery_launch.json').write_text(json.dumps({'schema':'plena_v3_exact_enospc_recovery_v1','status':'completed','launches':launches,'full_candidate_count':2178,'source_changes':'atomic CSV publish; only verified completed arithmetic reused'},indent=2)+'\n')


if __name__=='__main__':main()
