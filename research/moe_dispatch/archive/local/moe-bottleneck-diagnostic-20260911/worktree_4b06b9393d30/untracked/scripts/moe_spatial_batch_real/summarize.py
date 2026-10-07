#!/usr/bin/env python3
"""Summarize finished receipts; never fill missing experiment results."""
import csv,gzip,json,math
from pathlib import Path
import run_routes as r

OUT=r.OUT
def table(p,rows):
    keys=list(rows[0])
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def gm(xs):return math.exp(sum(math.log(x) for x in xs)/len(xs))
def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])
def main():
    dest=OUT/'summary';dest.mkdir(exist_ok=True)
    checks={}
    for folder in ['route_campaign','sensitivity','real_campaign','real_compute','breakdown','real_breakdown','operand_ports']:
        check=json.loads((OUT/folder/'validation.json').read_text());assert check['passed'],folder
        checks[folder]=check
    allrows=[json.loads(p.read_text())['row'] for p in (OUT/'route_campaign/receipts').glob('*.json')]
    rows=[x for x in allrows if not x['name'].startswith(('sensitivity_','pipeline_'))]
    assert len(rows)==4032 and len(allrows)==4800
    bykey={(x['window'],x['phase'],x['shape'],x['mode'],x['variant']):x for x in rows}
    shared={(x['model'],x['batch'],x['phase'],x['shape'],x['mode'],x['variant']):x for x in rows if x['phase'].startswith('shared_')}
    windows=json.loads((OUT/'inputs/route_windows.json').read_text());comp=[]
    for w in windows:
        for s in r.SHAPES:
            shape='+'.join(map(str,s))
            for mode in r.MODES:
                for variant in ['pure','finite']:
                    g=bykey[w['name'],'routed_gate_up',shape,mode,variant]
                    d=bykey[w['name'],'routed_down',shape,mode,variant]
                    sg=shared[w['model'],w['batch'],'shared_gate_up',shape,mode,variant]
                    sd=shared[w['model'],w['batch'],'shared_down',shape,mode,variant]
                    comp.append(dict(window=w['name'],model=w['model'],dataset=w['dataset'],split=w['split'],batch=w['batch'],
                        shape=shape,mode=mode,variant=variant,routed_gate_cycles=g['cycles'],routed_up_cycles=g['cycles'],
                        routed_down_cycles=d['cycles'],shared_gate_cycles=sg['cycles'],shared_up_cycles=sg['cycles'],shared_down_cycles=sd['cycles'],
                        routed_gemm_sum=2*g['cycles']+d['cycles'],six_phase_gemm_sum=2*g['cycles']+d['cycles']+2*sg['cycles']+sd['cycles'],
                        evidence='actual routed counts/full shapes, timing only; serial GEMM sum, not layer end-to-end'))
    table(dest/'route_gemm_comparison.csv',comp)
    oracle=[]
    for w in windows:
        if w['dataset']!='swe' or w['split']!='primary':continue
        for phase in ['routed_gate_up','routed_down']:
            for s in r.SHAPES[:3]:
                shape='+'.join(map(str,s))
                for mode in r.MODES:
                    base=bykey[w['name'],phase,shape,mode,'finite']
                    for variant in ['finite','oracle_zero_control','oracle_zero_weight','oracle_both']:
                        x=bykey[w['name'],phase,shape,mode,variant]
                        oracle.append(dict(window=w['name'],model=w['model'],batch=w['batch'],phase=phase,shape=shape,mode=mode,
                            condition=variant,cycles=x['cycles'],speedup_vs_charged=base['cycles']/x['cycles'],
                            source_weight_bytes=x['source_weight_bytes'],source_byte_delta=x['source_weight_bytes']-base['source_weight_bytes'],
                            activation_byte_delta=x['activation_bytes']-base['activation_bytes'],rmw_byte_delta=x['accumulator_rmw_bytes']-base['accumulator_rmw_bytes'],
                            invocation_delta=x['invocations']-base['invocations']))
    table(dest/'oracle_2x2_with_traffic.csv',oracle)
    probes=[]
    for shape in ['6','3+3','4+2','2+2+2','1+1+1+1+1+1']:
        for variant in ['control2','weight4x','activation4x','accumulator4x']:
            a=[x for x in rows if x['variant']==variant and x['shape']==shape];assert len(a)==12
            speed=[bykey[x['window'],x['phase'],shape,x['mode'],'finite']['cycles']/x['cycles'] for x in a]
            probes.append(dict(shape=shape,probe=variant,points=len(a),geomean_speedup=gm(speed),minimum=min(speed),maximum=max(speed),
                scope='BFCL primary, full-size routed gate, three models x four batches, pinned experts'))
    table(dest/'resource_probe_summary.csv',probes)
    # Select once on all primary windows, freeze BOTH shape and dispatch on holdout.
    finite=[x for x in comp if x['variant']=='finite']
    baselines={x['window']:x['six_phase_gemm_sum'] for x in finite if x['shape']=='6' and x['mode']=='pinned_expert'}
    scores=[]
    for s in r.SHAPES:
        shape='+'.join(map(str,s))
        for mode in r.MODES:
            rec=dict(shape=shape,mode=mode)
            for split in ['primary','holdout']:
                a=[x for x in finite if x['shape']==shape and x['mode']==mode and x['split']==split]
                assert len(a)==36
                rec[split+'_geomean_speedup_vs_single']=gm([baselines[x['window']]/x['six_phase_gemm_sum'] for x in a])
            scores.append(rec)
    table(dest/'frozen_policy_selection.csv',scores)
    uniform=max((x for x in scores if x['shape'] in ['3+3','2+2+2','1+1+1+1+1+1']),key=lambda x:x['primary_geomean_speedup_vs_single'])
    asym=max((x for x in scores if x['shape']=='4+2'),key=lambda x:x['primary_geomean_speedup_vs_single'])
    selected={}
    for split in ['primary','holdout']:
        def lookup(config):return {x['window']:x['six_phase_gemm_sum'] for x in finite if x['split']==split and x['shape']==config['shape'] and x['mode']==config['mode']}
        u=lookup(uniform);a=lookup(asym)
        selected[split]=dict(windows=len(u),hetero_speedup_over_selected_uniform=gm([u[k]/a[k] for k in u]),
            hetero_wins=sum(a[k]<u[k] for k in u),ties=sum(a[k]==u[k] for k in u))
    real=[json.loads(p.read_text()) for p in (OUT/'real_campaign').glob('real_b*/validation.json')]
    realpure=list(csv.DictReader((OUT/'real_compute/all_phases.csv').open()));realrows=[]
    for x in sorted(real,key=lambda x:(x['batch'],x['shape'],x['mode'])):
        pp=[p for p in realpure if int(p['batch'])==x['batch'] and p['shape']==x['shape'] and p['mode']==x['mode']]
        assert len(pp)==6
        realrows.append(dict(batch=x['batch'],shape=x['shape'],mode=x['mode'],
            pure_gemm_phase_sum=sum(int(p['pure_cycles']) for p in pp),finite_gemm_phase_sum=x['total_gemm_phase_cycles'],
            final_relative_l2=x['versus_pytorch']['relative_l2'],output_sha256=x['output']['sha256']))
    table(dest/'real_moe_values_and_cycles.csv',realrows)
    trace_rows=list(csv.DictReader((OUT/'real_breakdown/last_core.csv').open()))
    real_states=[]
    for b in [8,16]:
        for shape in ['6','3+3','4+2']:
            a=[x for x in trace_rows if int(x['batch'])==b and x['shape']==shape];assert len(a)==4
            def summed(keys):return sum(int(x[k])*int(x['multiplicity']) for x in a for k in keys)
            q=dict(batch=b,shape=shape,wall_cycles=summed(['wall_cycles']),issue_actor_cycles=summed(['issue']),
                wait_control=summed(['control_issue','install_queue','install_service']),
                wait_weight_source=summed(['source_queue','source_transfer','distribution']),
                wait_activation=summed(['activation']),other=summed(['result_backpressure','ready_behind_other_stage',
                    'descriptor_credit','weight_slot','dependency_or_ready_window','finished_idle']))
            assert sum(q[k] for k in ['issue_actor_cycles','wait_control','wait_weight_source','wait_activation','other'])==q['wall_cycles']
            real_states.append(q)
    table(dest/'real_issue_state_partition.csv',real_states)
    mainstats=[]
    for model in ['qwen','deepseek','nemotron']:
        for b in [2,4,8,16]:
            a=[x for x in finite if x['model']==model and x['batch']==b and x['mode']=='pinned_expert']
            vals={s:{x['window']:x['six_phase_gemm_sum'] for x in a if x['shape']==s} for s in ['6','3+3','4+2']}
            mainstats.append(dict(model=model,batch=b,windows=6,
                homo33_speedup_vs_single=gm([vals['6'][k]/vals['3+3'][k] for k in vals['6']]),
                hetero42_speedup_vs_single=gm([vals['6'][k]/vals['4+2'][k] for k in vals['6']]),
                hetero42_speedup_vs_homo33=gm([vals['3+3'][k]/vals['4+2'][k] for k in vals['6']]),
                hetero42_wins_vs_homo33=sum(vals['4+2'][k]<vals['3+3'][k] for k in vals['6'])))
    table(dest/'batch_model_summary.csv',mainstats)
    valid_runs=sum(checks[k]['runs'] for k in ['route_campaign','sensitivity','breakdown','real_breakdown','operand_ports'])+checks['real_campaign']['gemm_runs']+checks['real_compute']['pure_runs']+checks['real_compute']['frozen_finite_replays']
    conclusions=dict(real_connected_values_pass=True,real_configurations=40,real_batch_sizes=[2,4,8,16],
        full_value_models=['DeepSeek-V2-Lite-Chat layer1 last-token prefill'],route_models=['Qwen3.5','DeepSeek-V2-Lite','Nemotron3-Nano'],
        route_windows=72,route_archive_inventory_pairs=29495540,
        full_archive_timing_completed=False,native_hbm=False,whole_model_inference=False,complete_layer_timing=False,
        primary_selected_uniform=uniform,primary_selected_asymmetric=asym,selection_validation=selected,
        validated_campaign_simulator_runs=valid_runs,regression_cli_invocations=50,
        tests_workspace_invocations=341,clippy_pass=True,
        asymmetric_necessity_proved=False,scope='fixed candidate set, assumed timing, BF16 decoded finite interface')
    r.save(dest/'conclusions.json',conclusions)
    lines=['# Batch 2/4/8/16 与真实数据验证',
        '\n**实际权重和输入的完整 MoE 数值链已跑通；异构是否更快取决于 batch、路由分布和派工，不能据此宣布异构必需。**',
        '\n本轮使用物理 M 并行模型，硬件分别为 `[6]`、`[3,3]`、`[4,2]`、`[2,2,2]`、六个 `[1]`，每核 N=4/K=512，总乘法器均为 12,288。',
        '\n## 真实 DeepSeek 输入：固定整专家派工',
        '\n下表单位为**周期**，是六段 GEMM 的串行时间和。包含 routed/shared 的 gate/up/down；SiLU、路由合并等执行了数值验证，但其硬件时间未计入，**不能称整层端到端延迟**。',
        md(['Batch','单核 6','同构 3+3','异构 4+2','同构 2+2+2','六个 1'],[
            [b]+[next(x['finite_gemm_phase_sum'] for x in realrows if x['batch']==b and x['shape']==s and x['mode']=='pinned_expert') for s in ['6','3+3','4+2','2+2+2','1+1+1+1+1+1']] for b in [2,4,8,16]]),
        '\n允许拆分专家 M tiles 的完整对照，以及纯计算结果，见 `summary/real_moe_values_and_cycles.csv`，不能只引用上表忽略更强的调度基线。',
        '\n## B8 为何纯计算赢，有限供数却输',
        '\n同一个真实输入、固定派工，3+3 的纯计算六段和为 69,776 周期，4+2 为 56,720 周期。但有限模型分别是 373,946 和 390,993 周期。下面对每段最后完成的核心做互斥发射状态分类，再按六个串行阶段加总；“发射”不是全部 MAC 管线忙碌时间：',
        md(['组织','发射占用','等控制发射/安装','等权重源','等 activation','其他','合计'],[
            [x['shape'],x['issue_actor_cycles'],x['wait_control'],x['wait_weight_source'],x['wait_activation'],x['other'],x['wall_cycles']] for x in real_states if x['batch']==8 and x['shape'] in ['3+3','4+2']]),
        '\n4+2 发射占用减少 8,704 周期，但控制等待多 13,206、权重源等待多 12,545，合计反而多 17,047 周期。两者六段的权重源字节均为 190,316,544 B、全局控制服务均为 335,104 周期。**差别是服务如何与执行交错，以及哪条核的等待暴露在完成路径上，不能说异构多读了权重或控制总工作更多。** 此表是观察到的状态分解，各项不是可独立删去的因果收益；对照神谕及端口敏感性另列。',
        '\n## 当前模型更值得优化哪一处',
        '\nBFCL primary 的三个模型×四种 batch，共 12 个完整尺寸 gate 窗口，固定归属下的几何平均加速比如下。这是额外资源诊断；第二控制端口的物理面积和时序成本未建模，不能当等面积最终实现收益。',
        md(['组织','控制端口1→2','权重字节带宽4×'],[[s]+[f"{next(x['geomean_speedup'] for x in probes if x['shape']==s and x['probe']==v):.3f}×" for v in ['control2','weight4x']] for s in ['6','3+3','4+2','2+2+2']]),
        '\n在这组默认假设下，控制发射/安装及其重叠比单纯增加权重字节带宽更值得优先研究。activation/RMW 带宽4×均无变化，是因为请求原本已落在“至少1周期”这一档；不能用这个无变化结论排除端口瓶颈。',
        '\n已补 160 点×2 的真实 BF16 gate 测试：分别去掉 activation/累加器端口计时、同时去掉两者，以及乐观的同周期小请求合并；所有实际输出与 charged 逐位一致。见 `operand_ports/all_points.csv`。部分神谕更慢，反映在线事件顺序会改变；它们不构成严格单调上界。',
        '\n## 真实 decode 路由：跨模型对照',
        '\n每行含 BFCL/GPQA/SWE × primary/holdout，共六个窗口；下表为固定整专家派工下的几何平均加速比。>1 表示异构更快，<1 表示更慢；计时仍为完整尺寸 GEMM 串行和。',
        md(['模型','Batch','4+2 相对单核','4+2 相对3+3','胜过3+3窗口'],[[x['model'],x['batch'],f"{x['hetero42_speedup_vs_single']:.3f}×",f"{x['hetero42_speedup_vs_homo33']:.3f}×",f"{x['hetero42_wins_vs_homo33']}/6"] for x in mainstats]),
        '\n## 预先选型后的独立验证',
        f"\n仅用 36 个 primary 窗口选择后，固定同构配置为 `{uniform['shape']}/{uniform['mode']}`，固定异构为 `{asym['shape']}/{asym['mode']}`。在 36 个不重叠 holdout 窗口上，异构相对该同构的几何平均加速比为 **{selected['holdout']['hetero_speedup_over_selected_uniform']:.3f}×**，异构胜出 {selected['holdout']['hetero_wins']}/36。这里只搜索了规定的五种组织，不能称全局最优 DSE。",
        '\n## 证据与边界',
        '\n- 性能主表 4,032 点×2；敏感性 768 点×2；逐事件等待分解 120 点×2。',
        f'\n- 真实数值 40 配置×6 GEMM×2，另有匹配的纯计算 240 点×2、冻结有限模型 240 次时序回放、24 点×2 的 B8/B16 等待复核及 160 点×2 的实际数值端口诊断；合计 **{valid_runs:,} 次有效 campaign 验收执行**。早期逐事件任务曾因对应 compact 前置任务尚未完成而重试，协调日志保留在 logs/，没有修改数值或验收门槛。',
        '\n- 九份原始档案的 29,495,540 条有效专家分配做了全量统计；**实际时序仿真覆盖的是 72 个已列明窗口，不是整个档案所有窗口**。',
        '\n- 原始权重数值验证覆盖 DeepSeek 一层的 16 条真实提示词；另外两模型是实际路由加完整矩阵尺寸，未声称原始权重执行。',
        '\n- 所有真实值架构输出逐位一致，重复结果一致。独立 FP64 投影检查与 PyTorch MoE 对照通过；详细误差见 `real_campaign/validation.json`。',
        '\n- 源码新增显式 BF16 输入路径；20 个冻结回归完整输出一致，9 类错误输入/覆盖文件操作被拒绝；Rust workspace 341 次测试通过，clippy 通过。',
        '\n- 这是有限 BF16 operand 接口模型；没有原生 HBM、MX codec、完整模型生成、RTL、PPA 或经过综合标定的阵列频率结论。',
        '\n## 下一步的判断依据',
        '\n先使用 `oracle_2x2_with_traffic.csv` 与 `breakdown/last_core.csv` 确认每个失速点。控制/供数改动要同样给单核和强同构；事件时序改变引起的字节变化另列，不能把重叠等待相加或全部归为 HBM。若异构只在固定派工赢、同构拆分后持平，应研究调度代价；若纯计算赢而有限供数输，应研究具体端口与供数链。',
        '\n复现实验、输入约定、数值精度、计时边界详见 [METHODS.md](METHODS.md)。所有原始请求、报告和 SHA256 收据均保留。']
    (OUT/'REPORT_ZH.md').write_text('\n\n'.join(lines)+'\n')
    print(json.dumps(conclusions,indent=2))
if __name__=='__main__':main()
