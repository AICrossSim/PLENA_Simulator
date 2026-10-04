"""Generate evidence-linked Chinese tables; never choose hardware on heldout."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .compute import Core
from .memory import memory_budget
from .study import cores_from


LABELS = {"selected_single":"开发集选定单核", "selected_homogeneous":"开发集选定同构",
          "selected_heterogeneous":"开发集选定异构", "selected_unequal_PK":"选定不同PK异构",
          "fixed_6":"固定单核6", "fixed_3+3":"固定同构3+3", "fixed_4+2":"固定异构4+2"}


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def render(system: Path, compute: Path, destination: Path, durable: str):
    frozen = json.loads((system / "FROZEN_SELECTION.json").read_text())
    totals = {r["label"]: r for r in csv_rows(system / "heldout_summary.csv") if r["batch"] == "all"}
    ctotals = {r["label"]: r for r in csv_rows(compute / "heldout_summary.csv") if r["batch"] == "all"}
    chosen = {p["label"]: p for p in frozen["points"]}
    single = float(totals["selected_single"]["total_ms"])
    homo = float(totals["selected_homogeneous"]["total_ms"])
    hetero = float(totals["selected_heterogeneous"]["total_ms"])
    outcome = (f"异构相对选定同构减少耗时 {(1-hetero/homo)*100:.2f}%，相对选定单核减少 {(1-hetero/single)*100:.2f}%。"
               if hetero < min(single,homo) else
               f"选定异构/单核耗时为 {hetero/single:.4f}×，异构/同构为 {hetero/homo:.4f}×；没有证明异构同时胜过两种基线。")
    text = ["# 完整三维几何搜索：解析结果与边界", "",
            "**已取消固定 PK=512。** 本次完整枚举物理 PM×PN×PK，允许两核三个轴都不同。" + outcome,
            "", "这里的毫秒来自统一解析模型及假设的 1 GHz 时钟，不是新尺寸的 Rust/native Ramulator、RTL 或实芯片测量。旧 v3 和旧固定 PK Round A 的结果未修改，也不与本表混用。",
            "", "## 搜索与真实输入", "",
            "物理域：PM=1..16，PN=1..192，PK∈{32,64,128,256,512,1024}。主乘法器总数严格为 12,288；单核/同构/异构都使用同一 2,158,592 B 存储和总端口、带宽预算。完整域有 16,763 点：44 单核、41 同构、16,678 异构，其中 12,016 点两核 PK 不同。",
            f"主配置容量检查通过 {frozen['legal_geometries']:,} 点，排除 {frozen['excluded_geometries']:,} 点；排除点保留具体原因，不给它们填虚构延迟。每个合法点在 18 个开发层窗口上跑两遍并比较完整结果。随后各类前四点公平搜索等分/比例分配及两种有界循环，再冻结硬件、调运行时、评估 135 个留出层窗口。二次映射搜索只覆盖这些候选，不宣称整个联合空间全局最优。",
            "", "输入是已有 DeepSeek-V2-Lite 捕获的真实 token→expert 路由，H=2048、routed F=1408、shared F=2816、Top-k=6；覆盖 B2/B4/B8/B16 和混合窗口64/96/128。检查逐 token 路由与每专家 Me 完全一致，没有用随机 Zipf 作为性能输入。层/窗口有相关性，且留出数据在历史实验中已使用，所以不称全新盲测。",
            "", "## 本次完整供数模型的结果", "",
            "下表总时间是同一 135 个层窗口的累计值；平均值=总时间/135。所有尺寸顺序统一为 **PM×PN×PK**。MAC 空间利用率只计算发射内有效 MAC，不等于整层墙钟利用率。", "",
            "| 配置 | 冻结物理尺寸 | 总时间 ms | 平均层 ms | 空间利用率 | HBM GiB | 权重重复倍率 |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for label in ("fixed_6", "fixed_3+3", "fixed_4+2", "selected_single", "selected_homogeneous", "selected_heterogeneous", "selected_unequal_PK"):
        r = totals[label]
        text.append(f"| {LABELS[label]} | `{r['geometry']}` | {float(r['total_ms']):.3f} | {float(r['total_ms'])/int(r['layers']):.4f} | {float(r['spatial_utilization'])*100:.1f}% | {float(r['hbm_GiB']):.3f} | {float(r['hbm_reload_ratio']):.3f}× |")
    text += ["", "资源按硬件固定，不随测试 batch 改容量。运行时允许根据实际 Me 调整同一套有界分块与任务归属。", "",
             "| 选定配置 | W容量 KiB / 实际槽数 | X容量 KiB | 累加容量 KiB | Z容量 KiB | W/X/累加 bank数 | 调度 / 预取槽上限 |",
             "|---|---|---|---|---|---|---|"]
    for label in ("selected_single", "selected_homogeneous", "selected_heterogeneous"):
        p = chosen[label]
        b = memory_budget(cores_from(p["geometry"]),128,2048,2816,allocation=p["allocation"],buffer_limit=p["prefetch_slots"])
        values = ["＋".join(f"{getattr(c,k)/1024:.2f}" for c in b.cores) for k in ("w_capacity_bytes","x_register_bytes","accumulator_bytes","z_bytes")]
        slots = "+".join(str(c.w_slots) for c in b.cores)
        banks = "+".join(f"{c.w_banks}/{c.x_banks}/{c.accumulator_banks}" for c in b.cores)
        text.append(f"| {LABELS[label]} | {values[0]} / {slots} | {values[1]} | {values[2]} | {values[3]} | {banks} | {p['policy']} / {p['prefetch_slots']} |")
    text += ["", "## 不同 batch 的平均层耗时 ms", "",
             "| Batch/窗口 | 固定6 | 固定3+3 | 固定4+2 | 选定单核 | 选定同构 | 选定异构 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    allrows = csv_rows(system / "heldout_summary.csv")
    for batch in (2,4,8,16,64,96,128):
        bylabel = {r["label"]:r for r in allrows if r["batch"] == str(batch)}
        means = [float(bylabel[l]["total_ms"])/int(bylabel[l]["layers"]) for l in ("fixed_6","fixed_3+3","fixed_4+2","selected_single","selected_homogeneous","selected_heterogeneous")]
        text.append(f"| {batch} | " + " | ".join(f"{v:.4f}" for v in means) + " |")
    toy = json.loads((system / "teacher_toy_geometry3d.json").read_text())
    text += ["", "## 导师小实验：Me=4、2", "",
             "两专家分别计算一次 N=128、K=512 投影；理想操作数、整专家固定归属、每核最多8个输出记录。点积20周期、提交1周期、发射间隔1周期；这是上述三维模型的独立机制验证，不是整层计时。", "",
             "| 物理组织 | 两专家M波次 | 周期 | 有效MAC | 补齐浪费MAC | 发射内空间利用率 |",
             "|---|---|---:|---:|---:|---:|"]
    for r in toy["rows"]:
        text.append(f"| `{r['geometry']}` | {r['M_waves_per_expert']} | {r['cycles']} | {r['useful_macs']:,} | {r['padding_macs']:,} | {r['spatial_utilization']*100:.1f}% |")
    text += ["", "4+2在这个限定实验里为112周期，另外两种为224周期；额外M波次和补齐浪费的解释成立。有限上下文、投影形状、任务归属和访存条件变化后，不能直接把这个2×加速比套到整层。旧草稿的192/100周期不是本模型的实测结果。"]
    text += ["", "## 隔离计算形状的独立搜索", "",
             "这组另行用开发集选硬件，关闭 HBM、SRAM 端口和控制服务；保留有限上下文/中间值与共享向量消费者。它不是无容量限制的 FLOPs/峰值公式。", "",
             "| 配置 | 另行冻结物理尺寸 | 同一留出集总时间 ms | 空间利用率 |",
             "|---|---|---:|---:|"]
    for l in ("selected_single","selected_homogeneous","selected_heterogeneous","selected_unequal_PK"):
        r=ctotals[l]
        text.append(f"| {LABELS[l]} | `{r['geometry']}` | {float(r['total_ms']):.3f} | {float(r['spatial_utilization'])*100:.1f}% |")
    text += ["", "## 在等归属条件下定位瓶颈", "",
             "以下移除约束时固定 charged 的专家归属和队列次序。等待/服务可以重叠，不能把控制周期、HBM服务和计算周期相加当墙钟时间。消融数字也不能相加。", "",
             "| 冻结配置 | 原样 ms | 零控制 ms | 理想HBM ms | 理想片上端口 ms | 计算/向量 ms |",
             "|---|---:|---:|---:|---:|---:|"]
    oracle=csv_rows(system / "resource_oracles.csv")
    for l in ("selected_single","selected_homogeneous","selected_heterogeneous"):
        vals={r["condition"]:float(r["total_ms"]) for r in oracle if r["label"]==l}
        text.append(f"| {LABELS[l]} | " + " | ".join(f"{vals[c]:.3f}" for c in ("charged","zero_control","ideal_HBM","ideal_ports","compute_only")) + " |")
    text += ["", "## 冻结几何的点积时序敏感性", "",
             "只替换计时假设，不在留出集上重新挑选硬件。数值为同一135层窗口的累计ms。每级假设影响各PK树深变化，尚待RTL/精确模拟校准。", "",
             "| 配置 | flat20 | 每级1周期 | 默认每级2周期 | 每级4周期 |",
             "|---|---:|---:|---:|---:|"]
    sensitivity = csv_rows(system / "timing_sensitivity.csv")
    for l in ("selected_single", "selected_homogeneous", "selected_heterogeneous"):
        vals = {r["timing"]:float(r["total_ms"]) for r in sensitivity if r["label"] == l}
        text.append(f"| {LABELS[l]} | " + " | ".join(f"{vals[t]:.3f}" for t in ("conservative_flat20", "log2_stage1", "log2_stage2", "log2_stage4")) + " |")
    text += ["", "## 机制、验证和限制", "",
             "权重保留原 BF16 W[N,K] 的32B行对齐地址；物理PK变化不会凭空增加有效权重。M/N/K尾块占物理缓冲，零填充不从HBM读取。X在一个N组内复用，权重在有限M组内复用；Z装不下整专家时按行分块并重新计权重读取。配对Gate/Up一直保留到该组SiLU/product完成，Down输出直接经过消费者进入共用Y，不保留未计费的完整FP32中间矩阵。",
             "HBM标称256GB/s、延迟64ns、256个32B额度，在本连续模型里受额度限制至126.03GB/s；不是每核各拿256GB/s。没有空闲权重槽时，填充与最后读取串行收费；有空槽才使用有限前瞻。共享HBM、激活、向量和控制服务均回收未用份额。",
             "不同PK的点积延迟使用继承PK512=20周期的显式推导假设，并提供flat20及每级1/2/4周期敏感性。这些点尚未做RTL综合/频率验证；相同MAC/SRAM容量不代表等面积。平均端口限制不等于精确bank冲突时序。只有小型SharedHBM案例逐32B验证，完整搜索采用流体解析近似。",
             "数值参考实现核对BF16输入、FP32树和K分段累加；不同PK不保证bit-exact，反例已归档。小矩阵检查不替代预训练模型质量评估，本次没有重新运行模型推理或router。极小合成Me=H=F=1的私有估计存在不足一周期的边界差异；实际性能输入不包含这个尺寸。",
             "这批结果用于选择下一轮需要精确模拟和综合的候选；不能据此声称已证明异构硬件的必要性、普适优越性或硅片加速比。要进入论文主结论，需将少量冻结候选接回逐tile控制/端口/native HBM时序，校准各PK点积与实际提交延迟，并补 trained-model 数值质量和PPA。",
             "", "## 复现与文件", "",
             f"完整原始表、逐层数据和源码快照：[本地结果]({durable})。计算形状、时序敏感性、资源消融均为本次新模型；旧冻结结果未改。",
             "", "```bash",
             "/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python -m research.moe_dispatch.geometry3d.study --inputs INPUTS --out OUTPUT --stage search --workers 8",
             "/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python -m research.moe_dispatch.geometry3d.study --inputs INPUTS --out OUTPUT --stage final",
             "# 对独立计算搜索，两条命令均加 --compute-only",
             "/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python -m pytest research/moe_dispatch/geometry3d -q",
             "```", ""]
    destination.write_text("\n".join(text))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--system",type=Path,required=True)
    p.add_argument("--compute",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--durable",required=True)
    a=p.parse_args()
    render(a.system,a.compute,a.out,a.durable)


if __name__=="__main__": main()
