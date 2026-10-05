"""Independent artifact checks, provenance registry, structural bill and report."""
from pathlib import Path
import argparse,csv,hashlib,json,math,shutil
from collections import defaultdict
import numpy as np
from .area import features
from .search import settings_for
from .metrics import paired_bootstrap
from ..geometry3d.study import cores_from,write_json,write_csv


def read(path):return list(csv.DictReader(path.open()))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def gm(values):
    values=list(values);return math.exp(sum(math.log(v) for v in values)/len(values))


def independently_check(details,workloads,credits,fmt):
    checked=0
    for r in details:
        w=workloads[r['workload']]
        useful=sum(3*e['Me']*e['H']*e['F'] for e in w['experts'])
        assert r['useful_macs']==useful
        assert sum(p['useful_macs'] for p in r['phases'])==useful
        assert r['issued_macs']>=useful
        assert abs(r['latency_ms']-r['cycles']/1e6)<1e-12
        bits={'BF16':16,'W8':8,'W4':4}[fmt]
        def bytes_nk(n,k):
            code=((k*bits+7)//8+31)//32*32
            scales=0 if bits==16 else (((k+127)//128*2+31)//32)*32
            return n*(code+scales)
        unique=sum(2*bytes_nk(e['F'],e['H'])+bytes_nk(e['H'],e['F']) for e in w['experts'])
        assert unique==r['native_unique_bytes']
        assert r['hbm_bytes']>=unique and r['hbm_bytes']%32==0
        assert r['budget']['total_bytes']==2158592
        assert sum(r['budget']['structures'].values())==2158592
        assert sum(c['w_capacity_bytes'] for c in r['budget']['cores'])==40960
        assert sum(c['x_register_bytes'] for c in r['budget']['cores'])==12288
        assert sum(c['accumulator_bytes'] for c in r['budget']['cores'])==98304
        assert sum(c['z_bytes'] for c in r['budget']['cores'])==393216
        assert sum(c['w_banks'] for c in r['budget']['cores'])==64
        assert sum(c['x_banks'] for c in r['budget']['cores'])==24
        assert sum(c['accumulator_banks'] for c in r['budget']['cores'])==12
        for p in r['phases']:
            m=r['budget']['cores'][p['core']]
            assert p['peak_w_bytes']<=m['w_capacity_bytes']
            assert p['peak_x_bytes']<=m['x_register_bytes']
            assert p['peak_accumulator_bytes']<=m['accumulator_bytes']
            # Logical record counts alone must not hide output-word padding
            # for PN values not divisible by four. Check a conservative
            # row-aligned 16B backing for every selected live record group.
            record=m['pm']*m['pn']*4
            rounded_record=m['pm']*((m['pn']*4+15)//16)*16
            assert p['peak_accumulator_bytes']//record*rounded_record<=m['accumulator_bytes']
            assert p['started']<=p['stream_completed']<=p['completed']<=r['cycles']*(1+1e-9)
        cap=min(256,credits*32/65)
        assert unique/cap<=r['cycles']
        assert r['cycles']==max(r['core_finish_cycles'])
        checked+=1
    return checked


def provenance(out):
    rows=[]
    specs=[
        ('geometry_v6','/scratch/shared/mcl123/plena/final_artifacts/moe_geometry3d_20261004_v6/payload/system/heldout_layers.csv','Python3Dfluid','153captureddev/test,135heldout','postrouter GU/SiLU/Z/Down/combine','256credits,64cyclefixedresponse','same input/timing,945exactmatches'),
        ('runtime_20260930','/scratch/shared/mcl123/plena/outputs/moe_runtime_fsm_20260930/performance.csv','Rustfiniteanalytical','BFCLroutingprefix B2/4/8/16','postrouter fullFFN,olderbufferprotocol','256credits,64cyclefixedresponse','different inputs/arenas/dataflow;do not cross-table ratio'),
        ('native_20260923','/scratch/shared/mcl123/plena/outputs/moe_band_prefetch_20260923/six_phase_sums.csv','Rustanalytical and native HBM2','oneMoElayer,BFCL B2/4/8/16','sixindependentcoldGEMMs;noSiLU/Z/combine','native row/bank/refresh or28/80cyclehypotheses','different boundary and address order;not new3Dcalibration'),
        ('policy_20260930','/scratch/shared/mcl123/plena/outputs/moe_runtime_policy_ablation_20260930/metrics.csv','Rustfiniteanalytical','olderBFCL B2/4/8/16','olderfullFFN/runtimepolicy','256/512diagnostic credits','not comparable to new3Dheldout without input/protocol matching')]
    for name,path,backend,inputs,boundary,hbm,comparison in specs:
        p=Path(path);rs=read(p) if p.exists() else []
        rows.append({'table':name,'path':path,'rows':len(rs) if p.exists() else 'unavailable',
            'sha256':sha(p) if p.exists() else 'unavailable','backend':backend,
            'input_scope':inputs,'timing_boundary':boundary,'HBM_model':hbm,'comparison_status':comparison})
    rows +=[{'table':name,'path':'not identified','rows':'unknown','sha256':'unknown','backend':'unknown',
        'input_scope':'unknown','timing_boundary':'unknown','HBM_model':'unknown','comparison_status':'excluded until raw table/config/input hashes identified'}
        for name in ('user_referenced_Rust_720rows','user_referenced_screenshot')]
    write_csv(out/'TABLE_PROVENANCE.csv',rows)


def run(root,inputs):
    out=root/'audit';out.mkdir(parents=True,exist_ok=True)
    selected=json.loads((root/'search/FROZEN_SELECTION.json').read_text())['points']
    ws={w['id']:w for p in inputs.glob('*heldout.json') for w in json.loads(p.read_text())['workloads']}
    n=0;arena=[];services=[];result_map={}
    for p in selected:
        sub=root/'search'/f"{p['weight_format']}_c{p['credits']}"
        results=json.loads((sub/f"heldout_{p['family']}.json").read_text())
        result_map[(p['credits'],p['weight_format'],p['family'])]=results
        n+=independently_check(results,ws,p['credits'],p['weight_format'])
        s=settings_for(p['credits'],p['weight_format'],p)
        arena.append({'credits':p['credits'],'weight_format':p['weight_format'],'family':p['family'],
                     'geometry':p['geometry'],**features(cores_from(p['geometry']),s)})
        bybatch=defaultdict(list)
        for r in results:bybatch[r['batch']].append(r)
        for batch,rs in sorted(bybatch.items()):
            total=sum(r['cycles'] for r in rs)
            ctr=defaultdict(float)
            for r in rs:
                for c in r['exclusive_stream_counters']:
                    for k,v in c.items():ctr[k]+=v
            services.append({'credits':p['credits'],'weight_format':p['weight_format'],'family':p['family'],
                'batch':batch,'windows':len(rs),'mean_latency_ms':total/len(rs)/1e6,
                'mean_HBM_bytes':sum(r['hbm_bytes'] for r in rs)/len(rs),
                'HBM_effective_GBps':sum(r['hbm_bytes'] for r in rs)/total,
                'issued_spatial_utilization':sum(r['useful_macs'] for r in rs)/sum(r['issued_macs'] for r in rs),
                'wall_MAC_utilization':sum(r['useful_macs'] for r in rs)/(12288*total),
                'mean_core_finish_difference_ms':sum(max(r['core_finish_cycles'])-min(r['core_finish_cycles']) for r in rs)/len(rs)/1e6,
                'dominant_exclusive_fluid_limiter':max((k for k in ctr if k not in ('idle','hbm_startup')),key=lambda k:ctr[k]),
                'scope':'fluid service limiter,not native measured array/HBM activity;overlap counters not additive'})
    write_csv(out/'STRUCTURAL_BILL.csv',arena);write_csv(out/'ATTRIBUTION_BY_BATCH.csv',services)
    dataset_rows=[]
    for credits,fmt in sorted({(p['credits'],p['weight_format']) for p in selected}):
        for dataset in sorted({w['dataset'] for w in ws.values()}):
            chosen=lambda fam:[r for r in result_map[(credits,fmt,fam)] if ws[r['workload']]['dataset']==dataset]
            for ref in ('single','homogeneous'):
                a,b=chosen('heterogeneous'),chosen(ref)
                dataset_rows.append({'credits':credits,'weight_format':fmt,'dataset':dataset,
                    'baseline':ref,'windows':len(a),**paired_bootstrap(a,b)})
    write_csv(root/'search/paired_gates_by_dataset.csv',dataset_rows)
    from .selection_audit import run as audit_selection
    audit_selection(root,inputs)
    from .oracle_audit import run as audit_oracles
    audit_oracles(root,inputs)
    provenance(out)
    tools={x:shutil.which(x) for x in ('yosys','dc_shell','genus','vivado','verilator')}
    write_json(out/'CAPABILITIES.json',{'tools':tools,'ASIC_area_model_synthesized':False,
        'area_gate_active':False,'reason':'Vivado is FPGA tooling; no technology-qualified BF16/FP32 logic and SRAM macro model for these geometries',
        'native':'olderliveHBM2 available; currentresearchRust hardcodesPN4PK512; oldernative supports commonPN/PK; no matching fusedFFN independent-PN/PK3Dadapter',
        'required_calibration':'same input addresses/packing/ownership/privatebanks/pairedGU+Z+Down+combine boundary;6frozenpoints only if5percent versus BOTH baseline gates pass',
        'HBM_cap_is_native_measurement':False,'scope':'126.030769/252.061538 are credits*32/(64+1) continuous upper bounds with modeled landing and backpressure'})
    write_json(out/'INDEPENDENT_AUDIT.json',{'complete':True,'checked_selected_heldout_windows':n,
        'capture_windows':len(ws),'unique_bytes_rederived_from_dimensions_and_format':True,
        'full_installed_private_quotas_and_banks_checked':True,'phase_live_set_capacity_bounds_checked':True,
        'selected_accumulator_row_word_padding_fits':True,
        'actual_per_cycle_native_occupancy_measured':False,'new3Dpretrained_fullFFN_bitexact_tested':False,
        'scope':'independent arithmetic/traffic/capacity/phase-completion audit,not physical timing calibration'})
    figures(root,out,services)
    report(root,selected,out)


def figures(root,out,services):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read(root/'search/paired_gates_by_batch.csv')
    regimes=sorted({(int(r['credits']),r['weight_format']) for r in rows})
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharey=True)
    for ax,(credits,fmt) in zip(axes.flat,regimes):
        for ref,color in (('single','#1565c0'),('homogeneous','#e07a1f')):
            rs=sorted((r for r in rows if int(r['credits'])==credits and r['weight_format']==fmt and r['baseline']==ref),key=lambda r:int(r['batch']))
            xs=list(range(len(rs)));ys=[float(r['time_reduction_percent']) for r in rs]
            ax.plot(xs,ys,'o-',color=color,label='vs '+ref)
            # Percentile intervals need not contain the point estimate.
            ax.vlines(xs,[float(r['bootstrap95_low_percent']) for r in rs],
                      [float(r['bootstrap95_high_percent']) for r in rs],color=color)
            ax.set_xticks(xs,[r['batch'] for r in rs])
        ax.axhline(0,color='#888',lw=1);ax.axhline(5,color='#aaa',ls='--',lw=1);ax.axhline(10,color='#666',ls=':',lw=1)
        ax.set_title(f'{fmt}, {credits} credits');ax.set_xlabel('Batch / window tokens');ax.grid(axis='y',alpha=.15)
    axes[0,0].legend();axes[0,0].set_ylabel('Heterogeneous latency reduction (%)');axes[1,0].set_ylabel('Positive = faster')
    fig.suptitle('Frozen heldout analytical results; ideal decoder for W8/W4; not native calibration')
    fig.tight_layout();fig.savefig(out/'heldout_comparison.png',dpi=180);plt.close(fig)


def report(root,selected,out):
    gates=read(root/'search/paired_gates.csv');overall=read(root/'search/heldout_totals.csv')
    windows=read(root/'search/heldout_windows.csv')
    baselines={(r['credits'],r['weight_format'],r['family']):r for r in overall}
    regs=sorted({(r['credits'],r['weight_format']) for r in gates})
    lines=['# 统一基准、工作区间与独立资源搜索（2026-10-05）','',
        '**本轮是三维解析模型的筛选结果，尚未宣布异构硬件胜出。** 判断同时对比本次搜索范围内优化后的单核和同构；面积/能耗档未启用。',
        '', '| 格式／信用 | 单核尺寸 M×N×K | 同构尺寸 | 异构尺寸 | 异构对单核降低延迟 | 对同构降低延迟 | 双基线 ≥5% 入校准 |',
        '|---|---|---|---|---:|---:|---|']
    pass_regs=[]
    for credits,fmt in regs:
        ps={p['family']:p for p in selected if str(p['credits'])==credits and p['weight_format']==fmt}
        gs={r['baseline']:r for r in gates if r['credits']==credits and r['weight_format']==fmt}
        passed=all(r['entry_5percent']=='True' for r in gs.values())
        if passed:pass_regs.append((credits,fmt))
        lines.append(f"| {fmt}／{credits} | `{ps['single']['geometry']}` | `{ps['homogeneous']['geometry']}` | `{ps['heterogeneous']['geometry']}` | {float(gs['single']['time_reduction_percent']):+.2f}% | {float(gs['homogeneous']['time_reduction_percent']):+.2f}% | {'达到，匹配 backend 待接通' if passed else '未达到'} |")
    lines +=['','降低延迟按每窗口配对延迟比的几何平均计算，正数更快。每区间选定一套硬件/配比/调度，并冻结跨 B2/4/8/16/64/96/128；表中各区间是不同设计场景，不能拼成一颗动态换形状的芯片。',
        '', '| 格式／信用 | 单核几何平均 ms | 同构几何平均 ms | 异构几何平均 ms |',
        '|---|---:|---:|---:|']
    for credits,fmt in regs:
        means={fam:gm(float(r['latency_ms']) for r in windows if r['credits']==credits
                     and r['weight_format']==fmt and r['family']==fam)
               for fam in ('single','homogeneous','heterogeneous')}
        lines.append(f"| {fmt}／{credits} | {means['single']:.4f} | {means['homogeneous']:.4f} | {means['heterogeneous']:.4f} |")
    lines +=['','上表每值汇总同一组135窗口；不是完整模型生成耗时。逐batch的算术平均ms、权重字节和下限另见`heldout_by_batch.csv`，不要把两种平均混用。',
        '',f'进入校准的区间数：**{len(pass_regs)}／{len(regs)}**。宣布胜出仍须：校准后相对两个强基线降低延迟 ≥10%，配对窗口 bootstrap95% 下界 ≥5%；W4 还须真实模型质量验证。',
        '', '## 输入与计时边界','',
        '18 个开发窗口、135 个留出窗口，来自已捕获 DeepSeek-V2-Lite 的 BFCL/GPQA/SWE 路由，混合窗口含真实预填充前缀。H=2048，routed F=1408，shared F=2816，top-k=6。是离线路由窗口回放，不是重新执行的完整批推理。输入/路由已就绪开始，计 Gate/Up、SiLU、Z、Down和输出合并的服务；不计Router、Attention和完整模型生成。此解析模型不验证逐字合并顺序或完整数值结果。旧留出集曾被历史实验访问，不称为全新盲测。',
        '', '全部周期按假设 1GHz 换成 ms。126.03/252.06 GB/s 是固定64周期响应、32B请求、256/512信用及落地周期推导的连续供数上限，不是 Ramulator 实测带宽。512信用采用固定8KiB返回区加背压的解析假设，额外512B标签计入16KiB控制区；没有冒充已实现的物理512信用前端。',
        '', '## 同预算与强基线','',
        '固定12,288个主乘法器、2,158,592B存储、共享HBM和64/24/12个W/X/累加bank，每bank16B/cycle。W40KiB、X12KiB、累加96KiB、Z384KiB的两核私有配比与各类bank配比独立搜索，其余共享输入/输出/返回/控制/路由区保持原账本。相同乘法器和容量不代表相同面积。',
        '', '每区间枚举16,763个PM/PN/PK组合：PM1–16、PN1–192、每核独立PK32/64/128/256/512/1024。全几何初筛比较等分/按MAC比例；完整单核44形状和同构41形状额外扫描两种有界遍历、预取2/6/32，以及Gate/Up和Down各自的N组上限{auto,1,2,4,8}共25对。每族前8形状加更强基线种子做独立配比分阶段搜索：七轴七值，三起点、两轮坐标搜索及32个组合探针，再在开发集选运行时策略与分块参数。**这是覆盖完整几何、有限联合配比搜索；不宣称联合全局最优。**',
        '', '端口计费修正：旧v6按填充后的M/N/K整块搬运收费，Rust的SRAM访问实际使用有效m/n/k跨度。本轮只搬运有效W/X元素、累加按16B字对齐；完整tile容量预留、完整发射乘法槽和精度均保留。默认兼容模式仍精确复现旧v6的945个窗口结果，旧报告未被修改。对应Rust源码为`research/moe_dispatch/rust/src/main.rs`中的`wspans`与`xspans`；有效跨度计费对三种组织一致。它校正了计费口径，尚不等于新模型已接通native。',
        '', '目标为所有开发窗口的配对延迟比几何平均，不用总ms选设计。每点完整结果重复两次一致。近优1%候选与开发集bootstrap选型稳定性逐区间保存；bootstrap只刻画窗口抽样变化，不能替代模型误差，也不能把相关窗口当作独立模型重复。',
        '', '派工策略是现有的EFT、谁先空、轮转和Me阈值对照；本轮没有新增在线训练预测器。EFT使用私有服务估计，实际共享供数通过流式模型竞争。固定点神谕锁住charged归属，再改变HBM/端口/控制计时；不能把分项节省直接相加。',
        '', '本轮派工单位是整专家，各投影留在同核；Z放不下时显式按M分段并重读权重。不含跨核N带拆分、共享部分和或动态bank租借。未找到达标候选只适用于这个声明的搜索域，不能推出所有未来异构调度架构都无效。',
        '', '## 下限与归因','',
        'HBM有两列：实际读取字节/供数上限、唯一权重字节/供数上限。前者依赖重读次数，不能当成禁止新数据流减少重读的绝对极限。端口也有两列：实际映射服务量/端口带宽用于解释当前等待；必要最小搬运量/总端口带宽用于搜索入场判断。几何搜索余量=优化单核时间/max(唯一权重HBM下限、峰值MAC下限、必要端口下限)−1，开发集某batch达到10%才准入该区间。',
        '', '各核计算/依赖服务、W/X/累加端口服务、结束时间差、流式主限制及有效带宽保存在明细。计算服务不是实测阵列有效忙碌时间；HBM下限/墙钟是等效忙碌比例，不是原生HBM忙碌计数。重叠项不能相加组成墙钟分解。',
        '', '仍是阶段级连续进度、有限记录和阶段屏障的近似；不精确复现Current/Next FSM、逐地址bank冲突或非整字PN的真实访存布局。已额外检查选定点的累加记录按每行16B保守对齐后也能放下，具体SRAM地址和字使能时序仍须匹配后端校准。',
        '', '## 量化、面积与校准状态','',
        'W8/W4是按行32B对齐、group128 FP16 scale、乐观跨片段合并及理想解码的运输假设；私有W槽仍存BF16，压缩不凭空增加槽位。另有128/512元素每周期的有限解码敏感性。真实模型第一MoE层6个预训练矩阵的重复数值检查已完成，X是合成BF16输入，只报告权重和输出NMSE，不能充当困惑度/任务精度。',
        '', '现有Rust研究模型固定PN4/PK512；旧native可用共同PN/PK的独立GEMM，但尚无本轮独立PN/PK、GU融合、Z与combine全边界匹配接口。因此没有编造3D/native校准误差。达到入校准门槛的区间才冻结6个具体点及接口契约；未达到的区间按门槛跳过。',
        '', 'Vivado可用，但它是FPGA工具；缺少这组BF16/FP32阵列与SRAM宏的技术匹配综合面积模型。只交付乘法器、加法树、bank、控制器和容量账本，拒绝用乘法器数代替mm²或J。面积/能耗节省20%那档保持关闭。',
        '', '## 产物索引','',
        '- `optimized_grid/optimized_grid.csv`：各区间强单核、分batch下限和搜索空间。',
        '- `search/heldout_by_batch.csv`、`paired_gates_by_batch.csv`：平均ms与配对收益/置信区间。',
        '- `search/*/independent_allocations.csv`、`within1percent.csv`、`selection_stability_complete.csv`：配比探索和完整配置的稳定性。',
        '- `search/heldout_core_services.csv`、`fixed_owner_oracles.csv`：服务需求与单变量神谕。',
        '- `search/timing_decoder_sensitivity.csv`：PK时序/解码假设敏感性。',
        '- `quant/real_weight_quant_diagnostic.csv`、`audit/STRUCTURAL_BILL.csv`：数值诊断及硬件账本。',
        '- `audit/TABLE_PROVENANCE.csv`：旧表来源；未识别的截图和720行表不作交叉加速比。',
        '- `audit/INDEPENDENT_AUDIT.json`、各冻结文件：字节/存储/完成时序及来源记录。','']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True)
    a=p.parse_args();run(a.root,a.inputs)
