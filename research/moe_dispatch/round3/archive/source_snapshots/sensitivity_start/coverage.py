"""Read-only audit of actual evaluated witnesses, distinct from declared space."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
from .common import ROOT, canonical, write_csv, write_json, sha

BANKS=("w_banks","x_banks","acc_banks","vector_lanes")
CAPS=("w_bytes","x_bytes","acc_bytes","z_bytes")
FAMILIES=("single","homogeneous","5+1","4+2","3+3")
MODES=("pipelined","port_tight")


def identity(d):
    return canonical({**d,"label":""})


def merge(cs):
    rows={}
    for c in cs:
        for w in c["witnesses"]:
            k=identity(w["design"])
            if k in rows:
                assert rows[k]["status"]==w["status"]
                if w["status"]=="evaluated":
                    assert rows[k]["latencies_ms"]==w["latencies_ms"]
            else:rows[k]=w
    return list(rows.values())


def distinct(ds,fn):
    return sorted({canonical(fn(d)) for d in ds})


def summarize(stage,mode,group,family,witnesses,source):
    attempted=[w["design"] for w in witnesses]
    ds=[w["design"] for w in witnesses if w["status"]=="evaluated"]
    flows=Counter(canonical(d["flows"]) for d in ds)
    landings=Counter(d["landing_mode"] for d in ds)
    cuts={f:distinct(ds,lambda d,f=f:d[f]) for f in BANKS}
    capacity_totals=distinct(ds,lambda d:[sum(d[f])+(d["landing_pool_bytes"] if f=="w_bytes" else 0) for f in CAPS])
    return dict(stage=stage,onchip_mode=mode,constraint_group=group,family=family,
        unique_attempted_points=len(attempted),unique_successful_points=len(ds),
        successful_geometry_pairs=len(distinct(ds,lambda d:d["cores"])),
        attempted_geometry_pairs=len(distinct(attempted,lambda d:d["cores"])),
        declared_geometry_pairs=source["geometry_count"],
        actual_flow_combinations=len(flows),flow_combination_counts=canonical(flows),
        mixed_dataflow_points=sum(len(set(d["flows"]))>1 for d in ds),
        private_points=landings["private"],shared_pool_points=landings["shared"],
        capacity_cut_combinations=len(distinct(ds,lambda d:[d[f] for f in CAPS]+[d["landing_pool_bytes"]])),
        total_capacity_profiles=len(capacity_totals),total_capacity_profiles_B=canonical(capacity_totals),
        weight_storage_allocations=len(distinct(ds,lambda d:[d["w_bytes"],d["landing_pool_bytes"]])),
        x_storage_allocations=len(distinct(ds,lambda d:d["x_bytes"])),
        acc_storage_allocations=len(distinct(ds,lambda d:d["acc_bytes"])),
        z_storage_allocations=len(distinct(ds,lambda d:d["z_bytes"])),
        actual_joint_bank_combinations=len(distinct(ds,lambda d:[d[f] for f in BANKS])),
        actual_joint_bank_cuts=canonical(distinct(ds,lambda d:[d[f] for f in BANKS])),
        independent_cartesian_cells_of_observed_marginal_cuts=__import__('math').prod(len(v) for v in cuts.values()),
        **{f+"_cut_count":len(v) for f,v in cuts.items()},
        **{f+"_actual_cuts":canonical(v) for f,v in cuts.items()},
        shared_joint_bank_combinations=len(distinct([d for d in ds if d["landing_mode"]=="shared"],lambda d:[d[f] for f in BANKS])),
        private_joint_bank_combinations=len(distinct([d for d in ds if d["landing_mode"]=="private"],lambda d:[d[f] for f in BANKS])),
        proof_B_closed=source["proof_B_closed"],scope="actual successful witnesses only; declared independent variables not exhaustive evaluated coverage")


def run():
    root=ROOT/"E4"
    original={};final={};files=[]
    for dirname,target in (("certificates",original),("final_certificates",final)):
        for path in sorted((root/dirname).glob('*.json')):
            c=json.loads(path.read_text());mode=c.get('onchip_mode',c['parameters']['onchip_mode'])
            target[mode,c['constraint_group'],c['family']]=c
            files.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path)))
    assert len(original)==20 and len(final)==20,'All frozen source and final certificates required before publishing coverage'
    rows=[];union_rows=[]
    for mode in MODES:
        for family in FAMILIES:
            c0,c1=(original[mode,g,family] for g in ('C0','C1'))
            union=merge([c0,c1])
            r=summarize('actual_source_union',mode,'C0+C1',family,union,c0);union_rows.append(r);rows.append(r)
            for group in ('C0','C1'):
                c=original[mode,group,family]
                rows.append(summarize('original_campaign',mode,group,family,c['witnesses'],c))
                c=final[mode,group,family]
                rows.append(summarize('final_eligible_selection',mode,group,family,c['witnesses'],original[mode,group,family]))
    write_csv(root/'SEARCH_COVERAGE.csv',rows)
    source_calls=sum(c['total_campaign_simulator_calls'] for c in original.values())
    selected_mixed=sum(len(set(c['selected']['design']['flows']))>1 for c in final.values())
    md=['# 第三轮搜索的实际覆盖','',
        '统计对象为20份原始完整重复搜索凭证及20份最终联合筛选凭证，均取 status=evaluated 的实际物理回放候选；不把声明域、下界剪枝节点或无结果尝试当成实测点。C0/C1有重叠，因此下表按设计完整字段去重。最终筛选没有产生新的模拟调用。',
        '',f'原始搜索实际物理回放调用（含两次完整搜索和每点的两次回放）：{source_calls:,}。原始证明B开放 {sum(not c["proof_B_closed"] for c in original.values())}/20；最终20个选择中 mixed dataflow 数为{selected_mixed}。',
        '', '5+1 / 4+2 / 3+3 指每核乘法器预算份额，DSE后不等于物理 M 高度5/1、4/2、3/3。',
        '', '| 模式 | 家族 | 实评成功点 | 实评几何 / 声明几何 | dataflow组合数 | mixed flow点 | private / shared点 | 容量切分组合 | 总容量profile数 | 联合端口切分组合 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in union_rows:
        md.append(f'| {r["onchip_mode"]} | {r["family"]} | {r["unique_successful_points"]} | {r["successful_geometry_pairs"]}/{r["declared_geometry_pairs"]} | {r["actual_flow_combinations"]} | {r["mixed_dataflow_points"]} | {r["private_points"]}/{r["shared_pool_points"]} | {r["capacity_cut_combinations"]} | {r["total_capacity_profiles"]} | {r["actual_joint_bank_combinations"]} |')
    md += ['', '这些数字说明实际搜到了哪些候选，不能据此说全局最优。完整字段、原始C0/C1与最终C0/C1的分别计数均见 SEARCH_COVERAGE.csv。',
           '', '| 模式 | 家族 | W bank实际切分 | X bank实际切分 | accumulator bank实际切分 | vector lane实际切分 |',
           '|---|---|---|---|---|---|']
    for r in union_rows:
        fmt=lambda f:', '.join('['+','.join(map(str,json.loads(x)))+']' for x in json.loads(r[f+'_actual_cuts']))
        md.append(f'| {r["onchip_mode"]} | {r["family"]} | {fmt("w_banks")} | {fmt("x_banks")} | {fmt("acc_banks")} | {fmt("vector_lanes")} |')
    md += ['', '每个数组按core0、core1排列。独立边际切分的笛卡尔积不是实际联合测试点；CSV同时给出联合数和观测边际笛卡尔积数，避免把两者混为一谈。',
           '', '## 生成规则中的限制', '',
           '1. 主搜索种子只使用6套W/X/acc/Z总容量profile；每个类型使用同一算力比例或反向比例，未覆盖1KiB步长的任意跨类型容量组合。私有W满足最小tile/C1条件时可从Z借容量，这是额外局部调整，不等于独立搜索全部容量分区。',
           '2. 四种端口配比共同采用等分或算力比例。seed的奇数schedule_index同时选择shared与等分bank，偶数同时选择private与算力bank；landing和bank存在相关性。初始冻结设计可能提供少数额外组合，实际数量以上表为准。',
           '3. seed只生成WS/WS、IS/IS、OS/OS；双核分别WS/IS、OS/WS等混合dataflow并未系统搜索。个别初始候选的混合flow不代表独立搜索。',
           '4. 原始20组全部先使用initial+seed耗尽256点候选预算，再遍历2048个声明域节点。实际20组certificate数组均为空，没有resolved singleton或剪枝证书；所有实评点来自initial+seed。树仍为完整开放frontier，没有额外提供任意端口/容量切分的已评估候选。完整容量/端口变量在声明域中合法，不等于它们已经充分优化。',
           '', '因此当前结果应写“在等预算、固定生成规则下最好已评估候选”。可报告下界确实排除的性能门槛，但不能把未关闭的proof B写成独立联合优化的最优值，也不能仅凭该有限候选集否定所有异构设计或混合dataflow。',
           '', '此审计不修改硬件、选择、成本模型或结果；留出结果尚未全部生成时不计算新设计的最终性能结论。来源SHA见SEARCH_COVERAGE_RECEIPT.json。']
    (root/'SEARCH_COVERAGE_ZH.md').write_text('\n'.join(md)+'\n')
    write_json(root/'SEARCH_COVERAGE_RECEIPT.json',dict(original_certificates=20,final_certificates=20,csv_rows=len(rows),source_actual_simulator_calls=source_calls,source_files=files,search_source_sha256=sha(ROOT/'search.py'),interpretation='read-only actual witness audit; not exhaustive independent-variable optimization'))
    print({'rows':len(rows),'source_calls':source_calls,'original_open':sum(not c['proof_B_closed'] for c in original.values()),'selected_mixed':selected_mixed})

if __name__=='__main__':run()
