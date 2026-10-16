"""Audited coverage, Hot/Cold work, real GPU time and precision error tables."""
from __future__ import annotations
import argparse, collections, gzip, json, statistics, math
from pathlib import Path
import numpy as np
from .common import *
EXPECTED_PRECISIONS={'bf16','fp16','fp8_w8a8','int8_w8a8','int8_w8a16','int4_w4a16'}

def aggregate(rows,keys,metrics):
    groups=collections.defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in keys)].append(r)
    output=[]
    for key,values in sorted(groups.items()):
        row=dict(zip(keys,key));row['observations']=len(values)
        for metric in metrics:row[metric]=statistics.fmean(r[metric] for r in values)
        output.append(row)
    return output

def markdown(rows,columns):
    def v(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(v(r.get(c,'—')) for c in columns)+' |' for r in rows])

def capture_report(root,out):
    rows=[];coverage=[];requests=[];manifests={}
    for folder in sorted(root.glob('capture_*')):
        path=folder/'capture_manifest.json'
        if not path.exists():continue
        m=json.loads(path.read_text());precision=m.get('expert_precision','native')
        if precision=='native':precision=m['dtype']
        if m['layers']!=list(LAYERS) or m['steps']!=32 or m['profile_steps']!=list(PROFILE_STEPS):raise ValueError('capture coverage plan differs')
        if len(m['planned_groups'])!=57:raise ValueError('capture cohort coverage differs')
        manifests[precision]=(folder,m)
        plans={g['id']:g for g in m['planned_groups']}
        if len(plans)!=len(m['planned_groups']):raise ValueError('duplicate capture group')
        if m['complete'] and {s['id'] for s in m['shards']}!=set(plans):raise ValueError('missing capture groups')
        for g in plans.values():
            ids=[r['request_id'] for r in g['requests']]
            if len(ids)!=g['batch'] or len(set(ids))!=len(ids):raise ValueError('non-distinct request batch')
        for s in m['shards']:
            file=folder/s['file']
            if sha(file)!=s['sha256']:raise ValueError('capture modified '+str(file))
            with np.load(file) as z:
                generated=z['generated_tokens'];eos=generated==100001
                for layer in m['layers']:
                    ids=z[f'ids_l{layer}'];scores=z[f'scores_l{layer}'];validate_routes(ids,scores)
                    if ids.shape!=(m['steps'],s['batch'],6):raise ValueError('not complete native decode shape')
                    for step in range(1,m['steps']+1):
                        after_eos=float(eos[:step].any(axis=0).mean())
                        rows.append(dict(precision=precision,dataset=s['dataset'],batch=s['batch'],cohort=s['cohort'],
                            layer=layer,step=step,post_eos_request_fraction=after_eos,
                            expert_tokens_64=json.dumps(np.bincount(ids[step-1].reshape(-1),minlength=64).tolist(),separators=(',',':')),
                            **population(ids[step-1])))
            coverage.append(dict(precision=precision,dataset=s['dataset'],batch=s['batch'],cohort=s['cohort'],
                layers=len(m['layers']),decode_steps=m['steps'],decode_windows=m['steps']*len(m['layers']),
                profile_steps=len(m['profile_steps']),native_batch=True,phase='decode',capture_complete=m['complete']))
            for r in plans[s['id']]['requests']:
                requests.append(dict(precision=precision,dataset=s['dataset'],batch=s['batch'],cohort=s['cohort'],
                    request_id=r['request_id'],full_prompt_tokens=r['full_prompt_tokens'],prompt_tokens=r['prompt_tokens'],truncated=r['truncation']!='none'))
    metrics=['hot_experts','cold_experts','inactive_experts','hot_assignments','cold_assignments',
        'shared_flops_pct','hot_flops_pct','cold_flops_pct','unique_bf16_weight_MiB','post_eos_request_fraction']
    write_csv(out/'coverage.csv',coverage);write_csv(out/'request_provenance.csv',requests)
    rq=collections.defaultdict(list)
    for r in requests:rq[r['precision'],r['dataset']].append(r)
    write_csv(out/'request_summary.csv',[dict(precision=k[0],dataset=k[1],
        distinct_requests=len({r['request_id'] for r in v}),request_group_memberships=len(v),
        truncated_memberships=sum(r['truncated'] for r in v),max_original_prompt_tokens=max(r['full_prompt_tokens'] for r in v))
        for k,v in sorted(rq.items())])
    write_csv(out/'decode_windows.csv',rows)
    batch=aggregate(rows,['precision','batch'],metrics)
    write_csv(out/'workload_by_batch.csv',batch)
    write_csv(out/'workload_by_layer.csv',aggregate(rows,['precision','dataset','batch','layer'],metrics))
    write_csv(out/'workload_by_step.csv',aggregate(rows,['precision','batch','step'],metrics))
    # Free-running precision comparisons: equal prompts, generated prefixes may
    # diverge. This is not isolated router numeric sensitivity or task accuracy.
    comparisons=[]
    if 'bf16' in manifests:
        bf,bm=manifests['bf16'];base={s['id']:s for s in bm['shards']}
        for precision,(folder,m) in manifests.items():
            if precision=='bf16':continue
            for s in m['shards']:
                if s['id'] not in base:continue
                with np.load(bf/base[s['id']]['file']) as a,np.load(folder/s['file']) as b:
                    agree=float((a['generated_tokens']==b['generated_tokens']).mean())
                    for layer in m['layers']:
                        x=a[f'ids_l{layer}'];y=b[f'ids_l{layer}']
                        intersection=(x[...,None]==y[...,None,:]).any(-1).sum(-1)
                        comparisons.append(dict(precision=precision,dataset=s['dataset'],batch=s['batch'],cohort=s['cohort'],layer=layer,
                            top6_overlap_pct=100*float(intersection.mean())/6,identical_top6_set_pct=100*float((intersection==6).mean()),
                            generated_token_agreement_pct=100*agree))
    write_csv(out/'precision_route_comparison.csv',comparisons)
    return batch,coverage,manifests

def profile_report(root,out):
    points=[];inventory=[];seen=set()
    reference_path=root/'capture_bf16/capture_manifest.json'
    reference=json.loads(reference_path.read_text()) if reference_path.exists() else None
    for folder in sorted((root/'profiles').glob('*/*')):
        meta_path=folder/'metadata.json'
        if not meta_path.exists():continue
        m=json.loads(meta_path.read_text());n=0
        if reference:
            if m['capture_manifest_sha256']!=sha(reference_path):raise ValueError('different route source in precision comparison')
            if set(m['groups'])!={r['id'] for r in reference['shards']} or m['layers']!=list(LAYERS) or m['steps']!=list(PROFILE_STEPS):raise ValueError('incomplete campaign plan')
        for receipt in sorted(folder.glob('layer_*.complete.json')):
            r=json.loads(receipt.read_text());file=folder/f'layer_{r["layer"]:02d}.jsonl.gz'
            if sha(file)!=r['sha256']:raise ValueError('profile modified')
            with gzip.open(file,'rt') as f:
                for line in f:
                    row=json.loads(line);e2e=row.pop('e2e_ms');total=row.pop('instrumented_ms');parts=row.pop('stage_ms')
                    key=(row['precision'],row['repeat'],row['window_id'],row['path'])
                    if key in seen:raise ValueError('duplicate GPU measurement '+str(key))
                    seen.add(key)
                    if len(e2e)!=m['samples'] or len(total)!=m['samples'] or any(len(v)!=m['samples'] for v in parts.values()):raise ValueError('missing timing samples')
                    if m['warmup']<10 or m['samples']<50:raise ValueError('insufficient repeats')
                    row.update(e2e_us=statistics.fmean(e2e)*1000,e2e_median_us=statistics.median(e2e)*1000,
                        e2e_p95_us=float(np.percentile(e2e,95))*1000,
                        instrumented_us=statistics.fmean(total)*1000,samples=len(e2e))
                    for name,values in parts.items():row[name+'_us']=statistics.fmean(values)*1000
                    for name in parts:row[name+'_time_pct']=100*row[name+'_us']/row['instrumented_us']
                    if abs(sum(row[k+'_us'] for k in parts)-row['instrumented_us'])>.01:raise ValueError('stage totals inconsistent')
                    if not row['finite']:raise ValueError('nonfinite profile result')
                    points.append(row);n+=1
        expected=len(m['layers'])*len(m['groups'])*len(m['steps'])*2
        complete=(folder/'COMPLETE.json').exists()
        if complete and n!=expected:raise ValueError('incorrect completion receipt')
        if complete:
            receipt=json.loads((folder/'COMPLETE.json').read_text())
            if receipt['metadata_sha256']!=sha(meta_path):raise ValueError('metadata changed after run')
        inventory.append(dict(precision=m['precision'],repeat=m['repeat'],points=n,expected_points=expected,complete=complete))
    write_csv(out/'profile_coverage.csv',inventory);write_csv(out/'profile_points.csv',points)
    metrics=['e2e_us','e2e_median_us','e2e_p95_us','instrumented_us','shared_us','hot_us','cold_us','routed_us','combine_us',
        'hot_experts','cold_experts','shared_flops_pct','hot_flops_pct','cold_flops_pct',
        'relative_l2','max_abs_error','cosine','instrumentation_output_relative_l2','active_weight_storage_MiB']
    tables={}
    if points:
        for name,keys in [('profile_by_batch',['precision','path','batch']),
                          ('profile_by_layer',['precision','path','batch','layer']),
                          ('profile_by_step',['precision','path','batch','step']),
                          ('profile_by_dataset',['precision','path','dataset','batch'])]:
            result=aggregate(points,keys,metrics)
            for row in result:
                for part in ('shared','hot','cold','routed','combine'):
                    row[part+'_time_pct']=100*row[part+'_us']/row['instrumented_us']
            write_csv(out/(name+'.csv'),result);tables[name]=result
        paired=collections.defaultdict(dict)
        for r in points:paired[r['precision'],r['window_id'],r['path']][r['repeat']]=r
        stability=[]
        for key,repeats in sorted(paired.items()):
            if set(repeats)!={1,2}:continue
            x,y=repeats[1]['e2e_us'],repeats[2]['e2e_us']
            stability.append(dict(precision=key[0],window_id=key[1],path=key[2],repeat1_us=x,repeat2_us=y,
                absolute_relative_difference_pct=100*abs(x-y)/((x+y)/2)))
        write_csv(out/'repeat_stability.csv',stability)
        stable_groups=collections.defaultdict(list)
        for r in stability:stable_groups[r['precision'],r['path']].append(r['absolute_relative_difference_pct'])
        write_csv(out/'repeat_stability_summary.csv',[dict(precision=k[0],path=k[1],paired_windows=len(v),
            difference_pct_median=float(np.median(v)),difference_pct_p95=float(np.percentile(v,95)),
            difference_pct_max=max(v)) for k,v in sorted(stable_groups.items())])
        errors=collections.defaultdict(list)
        for r in points:
            if r['path']=='grouped':errors[r['precision'],r['batch']].append(r['relative_l2'])
        write_csv(out/'precision_error.csv',[dict(precision=k[0],batch=k[1],points=len(v),
            relative_l2_mean=statistics.fmean(v),relative_l2_p95=float(np.percentile(v,95)),relative_l2_max=max(v))
            for k,v in sorted(errors.items())])
        baselines={(r['window_id'],r['path'],r['repeat']):r for r in points if r['precision']=='bf16'}
        speedups=collections.defaultdict(list)
        for r in points:
            key=r['window_id'],r['path'],r['repeat']
            if key in baselines:
                speedups[r['precision'],r['path'],r['batch']].append(baselines[key]['e2e_us']/r['e2e_us'])
        speed_rows=[]
        for (precision,path,batch),values in sorted(speedups.items()):
            speed_rows.append(dict(precision=precision,path=path,batch=batch,paired_points=len(values),
                speedup_vs_bf16_GM=math.exp(statistics.fmean(math.log(v) for v in values))))
        write_csv(out/'precision_speedups.csv',speed_rows)
    return tables,inventory

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    batch,coverage,manifests=capture_report(a.root,a.out);tables,inventory=profile_report(a.root,a.out)
    cap_complete=set(manifests)==EXPECTED_PRECISIONS and all(m['complete'] for _,m in manifests.values())
    profile_complete={(r['precision'],r['repeat']) for r in inventory}=={(p,r) for p in EXPECTED_PRECISIONS for r in (1,2)} and all(r['complete'] for r in inventory)
    lines=['# 全层真实 Decode 与多精度 GPU Profile','',
        f'状态：采集 {"完成" if cap_complete else "进行中"}；六精度×两遍 GPU 回放 {"完成" if profile_complete else "进行中，不能当完整结果"}。',
        '', '本次使用 DeepSeek-V2-Lite-Chat 真实权重、真实逐层输入和真实自回归路由。26 个 MoE 层（1–26，0 为 Dense）；每组 32 个 decode 步。B 表示同一解码步的独立请求数，未用 prefill 或复制请求填大 Batch。',
        '', '每个数据集每个 Batch 最多两个不重叠请求组。GPQA 只有198个独立请求，因此 B128 只有一组，B256 不提供 GPQA；缺口不伪造补齐。各 Batch 可能重用同一请求，层和解码步也相关，窗口数不等于独立样本数。',
        '', '提示词最多1024 token，超长时保留前1/4和后3/4，截断逐请求记录。固定生成32个decode步（EOS后仍继续），post_eos_request_fraction单独报告；不能称为自然停止条件的服务负载。',
        '', 'GPU性能：固定 BF16 路由和输入，匹配真实层权重，单层post-router回放，CUDA Graph，10次预热、50次采样、两次完整重复。每层每组采样decode步1/4/8/16/32；其余32步均保留路由统计。',
        '', 'BF16/FP16是完整模型相应浮点精度的路由。FP8/INT8/INT4若有额外路由采集，表示仅Shared和Routed FFN量化，attention等仍为BF16、router运算FP32。量化使用RTN，不是已校准AWQ/GPTQ模型；局部输出误差或生成一致率不能代替任务准确率。',
        '', 'Shared在本轮也通过同精度的vLLM expert路径执行。相比旧表的Shared torch.mm路径，后端已改变，不能把前后绝对时延直接当优化收益。',
        '', 'Split模式把Hot/Cold分开调用，重复部分对齐工作；只能解释该拆分调用的阶段占比。正式整体性能看grouped的无内部计时标记耗时。计时标记总耗时单独给出，所有时间占比使用同一次标记调用为分母。',
        '', 'W8A16/W4A16压缩权重后在GEMM内解量化，激活BF16；不声称是INT4×INT4张量核吞吐。FP8/INT8 W8A8包含动态激活量化。默认内核调优配置缺失的形状可能并非该精度最快实现。没有硬件计数器证据的行不标记memory/compute bound。',
        '', '## BF16真实Decode负载（全部32步）','',markdown([r for r in batch if r['precision']=='bf16'],
            ['batch','observations','hot_experts','cold_experts','shared_flops_pct','hot_flops_pct','cold_flops_pct','post_eos_request_fraction']),
        '', '## 实测多精度耗时','']
    if tables:
        lines.append(markdown(tables['profile_by_batch'],['precision','path','batch','observations','e2e_us','instrumented_us','shared_time_pct','hot_time_pct','cold_time_pct','relative_l2']))
    else:lines.append('尚无完成的逐层回放结果。')
    lines+=['','完整逐层、逐步、逐数据集、重复稳定性及精度误差见同目录CSV。']
    (a.out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write_json(a.out/'AUDIT.json',dict(capture_complete=cap_complete,profile_complete=profile_complete,
        capture_windows=sum(r['decode_windows'] for r in coverage),profile_inventory=inventory,
        checks=['Top-6 distinct IDs','6B conservation','26 layers x 32 decode steps x 57 request groups per precision',
            'distinct requests within each batch','source hashes','common BF16 route source across GPU precisions',
            '50 timing samples per point','unique precision/repeat/window/path keys','no mixed prefill/decode','same-invocation stage closure']))
    print(json.dumps(dict(capture_complete=cap_complete,profile_complete=profile_complete,capture_windows=sum(r['decode_windows'] for r in coverage))))
if __name__=='__main__':main()
