"""Verify complete native runs and render the requested four measured columns."""
import argparse, collections, csv, hashlib, json, math, random, statistics
from pathlib import Path
import yaml
from .native_profile import ORGS,write,csv_write,digest

def geomean(xs):return math.exp(statistics.mean(math.log(x) for x in xs))
def paired_ci(xs):
    rng=random.Random(20261006);samples=sorted(geomean(rng.choices(xs,k=len(xs))) for _ in range(4000))
    return [samples[100],samples[3899]]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--campaign',type=Path,required=True)
    ap.add_argument('--native-build',type=Path,required=True);a=ap.parse_args();out=a.campaign
    manifest=json.loads((out/'manifest.json').read_text());rows=json.loads((out/'windows.json').read_text())
    summary=json.loads((out/'by_batch.json').read_text())
    assert manifest['completed']==manifest['points']==len(rows)==324
    assert manifest['exact_repeats_all'] and manifest['sources_unchanged'] and not manifest['errors']
    assert len({r['workload'] for r in rows})==108
    native=json.loads(a.native_build.read_text());assert digest(native['native_library'])==native['native_library_sha256']
    percore=[];byid=collections.defaultdict(list);native_rows=[]
    for row in rows:
        p=out/'points'/row['organization']/row['workload']
        receipt=json.loads((p/'repeat_receipt.json').read_text());r=json.loads((p/'repeat1.json').read_text())
        assert r==json.loads((p/'repeat2.json').read_text())
        assert receipt['binary_sha256']==manifest['binary_sha256']
        assert receipt['report_sha256']==digest(p/'repeat1.json')
        assert receipt['config_sha256']==digest(p/'config.json') and receipt['workload_sha256']==digest(p/'workload.json')
        assert r['native_hbm']['library_path']==native['native_library']
        s=yaml.safe_load(r['native_hbm']['native_stats_yaml'])['memory_system'];controllers=s['controller']
        count=sum(c['num_read_reqs_served'] for c in controllers)
        assert len(controllers)==8 and count==s['total_num_read_requests']==r['dma_transactions_landed']
        assert all(c['cycles']==r['native_hbm']['ticks'] for c in controllers)
        stats={k:sum(c[k] for c in controllers) for k in ('row_hits','row_misses','row_conflicts')}
        assert sum(stats.values())==count
        nr={'workload':row['workload'],'batch':row['batch'],'organization':row['organization'],
            **stats,'requests':count,'row_hit_percent':100*stats['row_hits']/count,
            'row_conflict_percent':100*stats['row_conflicts']/count,
            'X_stage_MiB':sum(c['stats']['x_stage_bytes'] for c in r['cores'])/2**20,
            'W_MiB':r['weight_bytes']/2**20,
            'core_finish_gap_ms':(max(c['stats']['done_cycle'] for c in r['cores'])-min(c['stats']['done_cycle'] for c in r['cores']))/1e6}
        native_rows.append(nr);byid[row['workload']].append(nr)
        budget=r['physical_budget'];assert sum(budget['weight_slots'])==10
        assert sum(budget['weight_banks'])==64 and sum(budget['x_banks'])==24 and sum(budget['acc_banks'])==12
        assert sum(budget['acc_bytes'])==2*1024**2 and sum(budget['control_bytes'])==4096
        for i,c in enumerate(r['cores']):
            st=c['stats'];live=r['live_timing_profile']['cores'][i];front=st['front_states']
            assert sum(front.values())==r['experts_done_cycles']
            assert live['operand_and_mac_pipeline_union_cycles']==st['arithmetic_active_cycles']
            percore.append({'workload':row['workload'],'batch':row['batch'],'organization':row['organization'],
                'core':i,'PM':c['m'],'PN':4,'PK':512,
                'finish_ms':st['done_cycle']/1e6,'MAC_union_ms':live['mac_active_union_cycles']/1e6,
                'operand_MAC_pipeline_union_ms':live['operand_and_mac_pipeline_union_cycles']/1e6,
                'front_weight_not_ready_ms':front.get('weight_not_ready',0)/1e6,
                'front_operand_feed_ms':front.get('operand_feed',0)/1e6,
                'front_X_not_ready_ms':front.get('x_not_ready',0)/1e6,
                'front_K_dependency_ms':front.get('previous_k_commit',0)/1e6,
                'front_idle_no_expert_ms':front.get('idle_no_expert',0)/1e6,
                'front_other_ms':sum(v for k,v in front.items() if k not in ('weight_not_ready','operand_feed','x_not_ready','previous_k_commit','idle_no_expert'))/1e6,
                'front_sum_ms':sum(front.values())/1e6,
                'X_stage_bytes':st['x_stage_bytes'],'W_bytes':st['weight_bytes'],
                'W_SRAM_service_word_ops':c['weight_bank_words'],
                'X_SRAM_service_word_ops':c['x_bank_words'],
                'workspace_service_word_ops':c['workspace_bank_words'],
                'word_bytes':16,'partial_sum_RMW_bytes':st['accumulator_rmw_bytes']})
    for workload,nrs in byid.items():
        assert len(nrs)==3 and len({r['W_MiB'] for r in nrs})==1
        assert len({r['X_stage_MiB'] for r in nrs})==1,(workload,'X traffic differs')
    csv_write(out/'native_memory_windows.csv',native_rows);write(out/'core_breakdown.json',percore);csv_write(out/'core_breakdown.csv',percore)
    memory=[]
    for batch in (2,4,8,16):
        for org in ORGS:
            rs=[r for r in native_rows if r['batch']==batch and r['organization']==org]
            total=sum(r['requests'] for r in rs)
            memory.append({'batch':batch,'organization':org,'windows':len(rs),
                'row_hit_percent':100*sum(r['row_hits'] for r in rs)/total,
                'row_conflict_percent':100*sum(r['row_conflicts'] for r in rs)/total,
                'X_stage_MiB':statistics.mean(r['X_stage_MiB'] for r in rs),
                'W_MiB':statistics.mean(r['W_MiB'] for r in rs),
                'core_finish_gap_ms':statistics.mean(r['core_finish_gap_ms'] for r in rs)})
    csv_write(out/'native_memory_by_batch.csv',memory);write(out/'native_memory_by_batch.json',memory)
    paired=[]
    for batch in (2,4,8,16):
        times={org:{r['workload']:r['total_ms'] for r in rows if r['batch']==batch and r['organization']==org} for org in ORGS}
        for org in ('homo33','heter42'):
            ratios=[times['single6'][w]/times[org][w] for w in times['single6']]
            paired.append({'batch':batch,'baseline':'single6','candidate':org,'windows':len(ratios),
                'speedup_geomean':geomean(ratios),'bootstrap_95_CI':paired_ci(ratios)})
    write(out/'paired_latency.json',paired)
    lines=['# 原生 HBM 与 Rust 运行时：固定 BF16 对照实测', '',
        '输入：108 个留出路由窗口，BFCL／GPQA／SWE，各 batch 27 个；三组织共 324 点，每点独立运行两次（648 次）。',
        '硬件形状顺序为 M×N×K：6×4×512；3×4×512 两核；4×4×512＋2×4×512。',
        '计时范围为路由完成、X 已在片上之后，Gate／Up → SiLU → Down → 按路由顺序合并；不含 Router、Attention 或完整模型生成。',
        'HBM：真实 Ramulator2 CAPI v2 回调，8 通道 HBM2_2000，内存／核心周期均 1 ns；32 B 请求，全系统 256 信用。',
        '计算与 SRAM：原有 Rust 事件模型（dot 尾延迟 20 周期、bank 端口和依赖计费）；不是 RTL 或芯片实测。大路由窗口只执行时间标签，小数值测试回放实际调度验证运算。', '',
        '| Batch | 组织 | 取数带宽 GB/s | 取数时间 ms | 有效计算 GFLOP/s | 计算活跃时间 ms | 总时间 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    labels={'single6':'单核 6','homo33':'同构 3+3','heter42':'异构 4+2'}
    for r in summary:
        lines.append(f"| B{r['batch']} | {labels[r['organization']]} | {r['fetch_GBps']:.2f} | {r['fetch_ms']:.4f} | {r['compute_active_GFLOPs']:.2f} | {r['compute_active_ms']:.4f} | {r['total_ms']:.4f} |")
    lines += ['', '取数时间：首个成功提交的权重请求到最后一个 HBM 返回回调，包含期间无请求和等待；最后 SRAM 落地写入另在总时间中计费。',
        '计算活跃时间：每次操作数就绪至 Dot 完成区间的并集，跨核心重叠只算一次；是这次带内存运行的活跃时间，不是理想供数耗时。',
        '取数带宽＝权重总字节÷上述取数时间；有效计算速度＝2×有效 MAC÷上述计算活跃时间。GB/s、GFLOP/s 为十进制。',
        '时间列为 27 窗口算术平均；速度列为总工作量÷总计时。取数与计算重叠，不能相加得到总时间。', '',
        '| Batch | 组织 | HBM 行命中 % | 行冲突 % | X 搬运 MiB | 权重 MiB | 两核完成差 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in memory:lines.append(f"| B{r['batch']} | {labels[r['organization']]} | {r['row_hit_percent']:.2f} | {r['row_conflict_percent']:.2f} | {r['X_stage_MiB']:.2f} | {r['W_MiB']:.2f} | {r['core_finish_gap_ms']:.4f} |")
    lines += ['', '行命中与冲突来自原生控制器统计；多个约束可以共同限制完成时间，不能仅凭占比把全部差距归因于 HBM。',
        'core_breakdown.csv 中的 front_* 是互斥的发射前端状态，每核求和等于专家阶段的时间；它们可能与后端 MAC 活跃重叠，不能同 MAC_union_ms 相加。',
        '权重槽 10 总槽（单核 10；双核 5＋5）、X 双缓冲 12 KiB、W SRAM 40 KiB＋共享返回区 8 KiB、累加／工作区 2 MiB；W／X／工作区 bank 总数分别 64／24／12。',
        '使用相同 dynamic 派工、stock 仲裁、N 分组 4、Current/Next 机制；这是一组冻结配置的实测，并未代表经过重新 DSE 的最优组织。']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write(out/'verification.json',{'all_324_points_verified':True,'runs':648,'exact_full_report_repeats':True,
        'equal_weight_and_X_bytes_every_window':True,'equal_gross_resource_budget':True,
        'native_read_counts_callbacks_landings_conserved':True,'all_8_controller_cycles_match_core_clock':True,
        'per_core_front_state_partition_exact':True,'pipeline_intervals_match_original_active_counter':True,
        'native_library_sha256':native['native_library_sha256'],'native_build_receipt_sha256':digest(a.native_build)})
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
