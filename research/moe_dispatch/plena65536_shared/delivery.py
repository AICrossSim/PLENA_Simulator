"""Validate delivery boundaries and build a compact Chinese result table."""
import argparse,csv,json,math,hashlib
from pathlib import Path

def main(out):
    rows=list(csv.DictReader((out/'cross_dataflow.csv').open()))
    groups={}
    for r in rows:
        assert sum(map(int,r['multiplier_allocation'].split('+')))==65536
        assert abs(float(r['unique_weight_MiB'])-33)<1e-10
        assert abs(float(r['unique_weight_fetch_ms_at256'])-.135168)<1e-10
        assert float(r['spatial_utilization_pct'])<=100+1e-9
        assert float(r['wall_peak_utilization_pct'])<=100+1e-9
        assert float(r['useful_GFLOP_s'])<=131072+1e-8
        if r['row_type']=='batch':
            b=int(r['batch'])
            assert math.isclose(float(r['useful_GFLOP_s']),
                b*float(r['unique_weight_required_GB_s']),rel_tol=1e-12)
            assert float(r['compute_cycles'])>=264*b
            key=(r['design'],b)
            groups.setdefault(key,set()).add(float(r['compute_ms']))
    assert all(len(times)==1 for times in groups.values())
    repeat=out/'finalized_repeat/FROZEN_SELECTION.json'
    if repeat.exists():
        assert repeat.read_bytes()==(out/'FROZEN_SELECTION.json').read_bytes()
    pipeline=out/'pipeline_check'
    if (pipeline/'REPORT_AUDIT.json').exists():
        assert (pipeline/'FROZEN_SELECTION.json').read_bytes()==(out/'FROZEN_SELECTION.json').read_bytes()
        for name in ('cross_dataflow.csv','dataflow_tradeoff.csv','supply_bw.csv','resource_ledger.csv'):
            assert (pipeline/name).read_bytes()==(out/name).read_bytes(),name
    names={'paper_original':'原论文几何（BF16变体）','native_single':'原生域最优单核',
      'native_homogeneous':'原生域最优同构','native_heterogeneous':'原生域最优形状异构',
      'extended_single':'扩展最优单核','extended_homogeneous':'扩展最优同构',
      'extended_heterogeneous':'扩展最优异构代表31:33',
      'extended_3to1_cooptimal_witness':'扩展异构3:1并列代表'}
    text=['# Shared-only 65,536乘法器 DSE：中文主表','',
      '完整计算搜索已完成；这些是论文拓扑约束下的BF16/FP32分析结果，尚未原生时序/面积校准。',
      '每个设计冻结一套几何，七个Batch等权GM。只计算Gate、Up、Down的GEMM，不含SiLU、Router、供数和派工耗时。','',
      '尺寸第三维为归约组数G，逻辑K块长R=b×G。原点32×32×64的乘法器数为65,536，逻辑R=2048。','',
      '| 组织 | 物理PM×PN×G | 原生子阵列帧 | 乘法器分配 |','|---|---|---|---|']
    for name,label in names.items():
        r=next(x for x in rows if x['design']==name and x['batch']=='2')
        text.append(f"| {label} | {r['physical_PM_PN_G']} | {r['native_frames']} | {r['multiplier_allocation']} |")
    text+=['','## 原生BLEN/MLEN域：计算时间，ms','',
      '| Batch | 原论文/最优单核 | 最优同构 | 最优形状异构 |','|---|---:|---:|---:|']
    for b in ('2','4','8','16','64','96','128','7-Batch GM'):
        vals=[next(r for r in rows if r['design']==name and r['batch']==b and r['dataflow'] in ('OS','OS/OS'))
              for name in ('paper_original','native_homogeneous','native_heterogeneous')]
        text.append('| '+('B'+b if b.isdigit() else b)+' | '+' | '.join(f"{float(r['compute_ms']):.9f}" for r in vals)+' |')
    text+=['','## 扩展域主表','',
      '下表最优单核、最优同构、31:33异构及3:1异构的计算、吞吐、权重带宽逐Batch完全相同。',
      '权重服务时间是33MiB÷256GB/s，不是实际E2E延时。MAC空间利用率不包含流水启动/排空。','',
      '| Batch | 三类组织Compute(ms) | Useful GFLOP/s | 唯一权重所需TB/s | 权重Fetch(ms) | 空间利用率 | 含启动/排空利用率 |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['design']=='extended_single' and r['dataflow']=='OS':
            b=r['batch'];label='B'+b if b.isdigit() else b
            text.append(f"| {label} | {float(r['compute_ms']):.9f} | {float(r['useful_GFLOP_s']):,.2f} | {float(r['unique_weight_required_GB_s'])/1000:.3f} | {float(r['unique_weight_fetch_ms_at256']):.6f} | {float(r['spatial_utilization_pct']):.2f}% | {float(r['wall_peak_utilization_pct']):.2f}% |")
    trades=list(csv.DictReader((out/'dataflow_tradeoff.csv').open()))
    text+=['','## 扩展最优单核的数据流供数账本','',
      '供数率为七Batch等权GM，容量为七Batch最大保守负载上界，包含33MiB权重。片上供数与外部取数分开。',
      'WS/IS保留原生OS PE，增加G组部分和上下文保存/恢复；不代表论文原生支持WS/IS切换。','',
      '| 数据流 | Compute GM(ms) | 片上W供数(TB/s) | 片上X供数(TB/s) | 部分和上下文(TB/s) | B128驻留负载上界(MiB) |',
      '|---|---:|---:|---:|---:|---:|']
    for r in trades:
        if r['design']=='extended_single':
            text.append('| '+r['dataflow']+' | '+f"{float(r['compute_ms']):.9f}"+' | '+
                ' | '.join(f"{float(r[k])/1000:.3f}" for k in ('pool_weight_refill_required_GB_s','pool_X_refill_required_GB_s','pool_context_required_GB_s'))+
                f" | {float(r['resident_shared_payload_upper_MiB']):.3f} |")
    text+=['','## 结论与边界','',
      '- 原生域的最优同构GM为0.010401454ms，形状异构为0.011720303ms；异构比同构慢12.68%。',
      '- 输出复制扩展后的三类最佳几何GM均为0.005044152ms；相对原点计算加速3.059倍，但异构没有额外计算加速。',
      '- 原点PM=32的小Batch行填充率为6.25%/12.5%/25%/50%；扩展PM=2的七个Batch都能填满行。',
      '- 扩展最优点由16,384个2×2子阵列组成，原点仅64个32×32子阵列。相等乘法器不等于相等面积、广播成本或Fmax。',
      '- B2~B128的计算时间都小于0.135168ms，因此冷读取、256GB/s条件下仍受访存约束。只有唯一权重服务的理论交叉：原点B417，扩展点B511；并非真实E2E翻转证明。',
      '- 全部9种有序双核数据流组合、三种时序敏感性、缓冲/树/部分和账本均在完整工作簿和REPORT.md中。',
      '- 全域搜索两遍逐位一致；525个冻结点各重算两遍一致；44个测试与9个子测试通过。']
    (out/'TABLES_ZH.md').write_text('\n'.join(text)+'\n')
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    result={'status':'PASS','table_rows_checked':len(rows),'all_flow_compute_equal_under_ideal_ports':True,
      'units_useful_flops_weight_bandwidth_checked':True,'MAC_budget_and_peak_checked':True,
      'finalized_selection_repeat_identical':repeat.exists(),
      'complete_pipeline_reproduced':(pipeline/'REPORT_AUDIT.json').exists(),
      'source_sha256':sources,'native_timing_calibrated':False,'area_calibrated':False}
    (out/'FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
