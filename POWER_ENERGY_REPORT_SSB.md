# PLENA 能耗估算报告 — single_stream_block (SSB)

_工具: `tools/power/plena_isa_energy.py` · 数据源: George analytic · 结构: 真实 .isa · 2026-06-01_

> 姊妹报告: [`POWER_ENERGY_REPORT_DSB.md`](POWER_ENERGY_REPORT_DSB.md)(double_stream_block)。
> 方法论、硬件配置、数据源、假设三条与 DSB **完全一致**,本文只换 ir 目录
> (`managerbuild_SSB_2/ir`)并标注 SSB 特有差异。

## 0. 方法论 — 数据用 George,结构用真实 .isa

本模型**完全复刻 George analytic 的两条腿**(cycle 公式 + 功耗系数),但把 George
"用模型维度公式猜指令数" 换成 "读 `managerbuild_SSB_2/ir/<kernel>/<kernel>.isa`
真实数指令(含 cloop 展开)"。结论与 George **同源可对话**,但精度更高。

```
真实 .isa (15 kernel)
  ├─ 1. 逐条指令 + cloop 展开  →  各类指令真实条数        [真实数据, 替代 George 维度公式]
  ├─ 2. 每条指令 cycle  ←─ George customISA_lib.json (performance 新版)   [George 数据①]
  ├─ 3. 按 opcode 前缀分桶: M_→MCU/MSRAM  V_→Vector/VSRAM  S_→标量  H_→HBM
  ├─ 4. compute_time = Σ(M_/V_/S_/C_ cycle)/freq
  │     memory_time  = emulator 实测 transmission (KERNEL_REPORT_SSB §1.3)
  ├─ 5. total = max(compute, memory)                     [George 重叠口径]
  └─ 6. 能耗 = 组件功耗 × 组件活跃时间                     [George power_model 数据②]
         MCU 活跃 = M_* 指令 cycle (指令级精确, 非 George 段级近似)
```

## 1. 硬件配置 (plena_settings.toml [BEHAVIOR])

| MLEN | HLEN | BLEN | VLEN | HBM_WIDTH | MX | freq |
|---|---|---|---|---|---|---|
| 1024 | 128 | 128 | 1024 | 1024 B | E4M3/E8M0 | 1 GHz |

## 2. 数据源 (全部 George)

- **cycle 公式**: `analytic_models/performance/customISA_lib.json` (origin/experiment, 新版)
  - 关键: `M_MM=BLEN=128`, **`M_BMM=M_BTMM=(MLEN//BLEN)²×BLEN=8192`**
  - ⚠️ 此版与 simulator 的 `M_BMM=overhead+MLEN=1024` **差 8 倍** — 见 §5 假设。
- **功耗系数**: `analytic_models/power_model/plena_power_model.py` (origin/experiment)
  - Vector/SRAM: 二次多项式 (VLEN/MLEN 拟合自 DC 综合, ≤64)
  - MCU: 两种取值见下。

## 3. 结果

> **memory = emulator 实测 transmission**(KERNEL_REPORT_SSB §1.3),已接入工具
> (`--trans ssb`,run.sh 默认 auto 按 ir_dir 名选 SSB 表)。整链 transmission
> = 16.52 ms,与 KERNEL_REPORT_SSB §1.5 的 16.5 ms 一致。

### 3a. MCU = 100 W (目标值口径)

| 层 | compute | memory | total(Σmax) | 能耗 | 平均功率 |
|---|---|---|---|---|---|
| 整链 | 34.40 ms | 16.52 ms | **43.56 ms** | **0.179 J** | **4.1 W** |

| 组件 | 能耗 | 占比 |
|---|---|---|
| MCU (systolic) | 144.4 mJ | 80.5% |
| Vector SRAM | 24.6 mJ | 13.7% |
| Vector Unit | 8.7 mJ | 4.9% |
| Matrix SRAM | 1.7 mJ | 0.9% |

### 3b. MCU = M×K 修正模型 (RTL 实锤 PE=M×K, c≈17 mW/PE)

| 层 | compute | memory | total | 能耗 | 平均功率 |
|---|---|---|---|---|---|
| 整链 | 34.40 ms | 16.52 ms | 43.56 ms | **3.26 J** | **74.9 W** |

MCU 占 98.9%(大尺寸外推主导)。两版差异**仅在 MCU 功耗取值**(100 W vs 修正外推值);
两版 compute/memory/total 时间相同。

### 3c. Top 能耗 kernel (MCU=100W, SSB_2 实跑)

| kernel | matmul cyc | vector cyc | scalar cyc | #H | comp_us | mem_us | bound | 能耗 |
|---|---|---|---|---|---|---|---|---|
| **flash_attention** | 5,263,684 | 4,825,088 | 12,629,343 | 128 | 18636 | 1790 | comp | **69.12 mJ** |
| linear2 (proj+mlp_out) | 262,152 | 81,920 | 149,103 | 224 | 493 | 1842 | **mem** | 27.65 mJ |
| linear_k | 131,080 | 81,920 | 148,514 | 144 | 362 | 1200 | **mem** | 14.03 mJ |
| linear_mlp | 131,080 | 81,920 | 148,514 | 144 | 362 | 1200 | **mem** | 14.03 mJ |
| linear_q | 131,080 | 81,920 | 148,514 | 144 | 362 | 1200 | **mem** | 14.03 mJ |
| linear_v | 131,080 | 81,920 | 148,514 | 144 | 362 | 1200 | **mem** | 14.03 mJ |
| rope_k | 65,544 | 90,112 | 173,235 | 136 | 329 | 1052 | **mem** | 7.32 mJ |
| rope_q | 65,544 | 90,112 | 173,235 | 136 | 329 | 1052 | **mem** | 7.32 mJ |
| gelu | 0 | 966,656 | 3,454,087 | 64 | 5862 | 606 | comp | 4.38 mJ |
| qknorm_k | 0 | 630,784 | 2,328,408 | 96 | 3221 | 755 | comp | 2.52 mJ |
| qknorm_q | 0 | 630,784 | 2,328,408 | 96 | 3221 | 755 | comp | 2.52 mJ |
| layernorm | 0 | 204,800 | 517,974 | 128 | 723 | 903 | **mem** | 0.74 mJ |
| concat | 0 | 0 | 793 | 128 | 1 | 1164 | **mem** | 0.66 mJ |
| residual_gate | 0 | 16,384 | 49,950 | 128 | 66 | 904 | **mem** | 0.53 mJ |
| modulate | 0 | 16,384 | 49,951 | 128 | 66 | 903 | **mem** | 0.53 mJ |

**关键观察**: `flash_attention` 一个 kernel 占 **69.1 mJ ≈ 全链 39%**(matmul + vector
+ 海量 scalar,且 compute 18.6ms 远 > 搬运 1.8ms,是唯一的纯 compute 巨头);`linear2`
因融合 proj + mlp_out(K 维最大)排第二 27.7 mJ。matmul 集中在 flash + 5 个 linear →
吃掉几乎全部 MCU 能耗。**接入实测 transmission 后,5 个 linear、2 个 rope、layernorm/
concat/modulate/residual_gate 全部翻成 memory-bound**(搬运 > 计算)——这些 kernel 的
墙钟由 HBM hop 决定,不是算力;`gelu/qknorm` 仍是 compute-bound(逐元素 scalar 链长)。
`gelu/qknorm/layernorm/...` **零 matmul**(MCU 不启动),能耗近 0。

## 4. SSB 与 DSB 的差异(为什么数不一样)

| | DSB (double_stream) | SSB (single_stream, 本报告) |
|---|---|---|
| kernel 数 | ~40(img+txt 双流,各一套 lin/rope/qknorm/mlp) | **15**(单流,一套) |
| 整链 compute | 35.49 ms | **34.40 ms** |
| 整链 memory (emulator 实测) | 28.32 ms | **16.52 ms** |
| total (max 重叠) | 55.36 ms | **43.56 ms** |
| 能耗 (100W) | 0.186 J | **0.179 J** |
| 能耗 (M×K) | 3.27 J | **3.26 J** |

两边的 memory 都来自 emulator 实测 transmission(DSB §1.3 / SSB §1.3),已分别接入工具
的 `DSB_TRANSMISSION_NS` / `SSB_TRANSMISSION_NS` 表(`--trans auto` 按 ir_dir 名选)。
SSB 的 16.52 ms 与 KERNEL_REPORT_SSB §1.5 的 16.5 ms 一致。

能耗几乎与 DSB 持平(0.179 vs 0.186 J)是合理的:SSB 虽少一半 kernel,但单流序列是
joint 8828(含 txt),`linear2` 融合了更大的 K 维,且 flash_attention 同样在 joint 序列上
跑——matmul 总量与 DSB 双流相当。**能耗几乎不受 memory 影响**(能耗=功耗×指令活跃 cycle,
与 max 重叠口径无关);memory 只抬高 total 时间 / 拉低平均功率。

## 5. 三个必须标注的假设 (与 DSB 同;发 George 时一并确认)

1. **MCU 功耗取值**: 100W 为目标值(来源待 George 确认); M×K 修正版是从 DC 综合 M=4
   外推到 M=128×K=1024(~1000× PE),公式形式经 RTL 验证(PE=M×K,非原 power_model
   的 M²×K bug),但绝对值待大尺寸重新综合校准。

2. **cycle 用 George performance 版 (M_BMM=8192)**: 与 simulator 的 1024 差 8 倍。
   RTL 推导(`(MLEN//BLEN)² tile × ~M 拍`)偏向 8192。两套是真实建模分歧,需 George
   确认真机一条 M_BMM 到底多少拍。

3. **memory 口径**: George 原版 memory_time 是 `bytes/带宽` 静态除法; 本报告用 emulator
   实测 transmission 替代(更准,KERNEL_REPORT_SSB §1.3,已接入工具 `SSB_TRANSMISSION_NS`)。
   ⚠️ **transmission 是 simulator(M_BMM=1024)口径实测的,而本报告 compute 用 George
   performance(M_BMM=8192)**,两层不同源——这是已知建模分歧(见假设 2),total=43.56 ms
   是"George compute + simulator memory + max 重叠"的混合口径,非单一来源。max 重叠假设
   HBM 与 compute 完全 pipeline(George 口径); simulator 实测是串行相加(KERNEL_REPORT
   §1.5 = 53.0 ms,prefetch 阻塞 await,不重叠),两者对 pipeline 假设相反。

   另: DC 综合**无 switching activity 标注**(默认 toggle), 功耗本身偏低估(power_model
   作者自注); scalar/control 逻辑 + HBM 控制器功耗未单列(偏低估)。

## 6. 复现

```bash
# 一键(默认即 SSB_2):
tools/power/run_energy.sh                       # MCU=100W
tools/power/run_energy.sh managerbuild_SSB_2/ir formula   # MCU M×K 修正

# 等价手动:
git show origin/experiment:analytic_models/performance/customISA_lib.json > /tmp/customISA_perf.json
python3 tools/power/plena_isa_energy.py managerbuild_SSB_2/ir --customisa /tmp/customISA_perf.json --mcu-power-w 100
python3 tools/power/plena_isa_energy.py managerbuild_SSB_2/ir --customisa /tmp/customISA_perf.json
```
