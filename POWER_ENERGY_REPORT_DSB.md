# PLENA 能耗估算报告 — double_stream_block (DSB)

_工具: `tools/power/plena_isa_energy.py` · 数据源: George analytic · 结构: 真实 .isa · 2026-05-31_

## 0. 方法论 — 数据用 George,结构用真实 .isa

本模型**完全复刻 George analytic 的两条腿**(cycle 公式 + 功耗系数),但把 George
"用模型维度公式猜指令数" 换成 "读 `managerbuild_DSB/ir/<kernel>/<kernel>.isa`
真实数指令(含 cloop 展开)"。结论与 George **同源可对话**,但精度更高。

```
真实 .isa (40 kernel)
  ├─ 1. 逐条指令 + cloop 展开  →  各类指令真实条数        [真实数据, 替代 George 维度公式]
  ├─ 2. 每条指令 cycle  ←─ George customISA_lib.json (performance 新版)   [George 数据①]
  ├─ 3. 按 opcode 前缀分桶: M_→MCU/MSRAM  V_→Vector/VSRAM  S_→标量  H_→HBM
  ├─ 4. compute_time = Σ(M_/V_/S_/C_ cycle)/freq
  │     memory_time  = emulator transmission (KERNEL_REPORT_DSB §1.3)   [实测搬运]
  ├─ 5. total = max(compute, memory)                     [George 重叠口径]
  └─ 6. 能耗 = 组件功耗 × 组件活跃时间                     [George power_model 数据②]
         MCU 活跃 = M_* 指令 cycle (指令级精确, 非 George 段级近似)
```

**比 George 精确在:** ①指令条数真实(非维度公式猜) ②能拆 matmul/vector/scalar
逐 kernel(George 累加成标量、拆不开) ③MCU 活跃时间数真实 M_* 指令(George 按段近似)
④memory/compute 靠 opcode 前缀直接分(George 需两套维度公式)。

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

### 3a. MCU = 100 W (目标值口径)

| 层 | compute | memory | total(Σmax) | 能耗 | 平均功率 |
|---|---|---|---|---|---|
| 整链 | 35.49 ms | 28.32 ms | **55.36 ms** | **0.186 J** | **3.4 W** |

| 组件 | 能耗 | 占比 |
|---|---|---|
| MCU (systolic) | 144.4 mJ | 77.5% |
| Vector SRAM | 31.3 mJ | 16.8% |
| Vector Unit | 9.1 mJ | 4.9% |
| Matrix SRAM | 1.7 mJ | 0.9% |

### 3b. MCU = M×K 修正模型 (2234 W, RTL 实锤 PE=M×K, c≈17 mW/PE)

| 层 | compute | memory | total | 能耗 | 平均功率 |
|---|---|---|---|---|---|
| 整链 | 35.49 ms | 28.32 ms | 55.36 ms | **3.27 J** | **59.0 W** |

MCU 占 98.7%(2234W 外推主导)。两版差异**仅在 MCU 功耗取值**(100 vs 2234 W)。

### 3c. Top 能耗 kernel (MCU=100W)

| kernel | matmul cyc | vector cyc | #H | comp_us | mem_us | bound | 能耗 |
|---|---|---|---|---|---|---|---|
| **flash** | 5,263,684 | 4,825,088 | 128 | 18636 | 1790 | comp | 69.1 mJ |
| linear ×12 (linq/link/linv/proj/mlpin/mlpout ×2流) | 65,540 | 40,960 | 72 | 181 | 600 | **mem** | 7.0 mJ each |
| rope ×4 | 32,772 | 45,056 | 68 | 165 | 526 | **mem** | 3.7 mJ each |
| gelu ×2 | 0 | 483,328 | 32 | 2931 | 303 | comp | 2.2 mJ each |
| qknorm ×4 | 0 | 315,392 | 48 | 1611 | 377 | comp | 1.3 mJ each |
| concat/split ×4 | 0 | 0 | 256 | 1.6 | 2423 | **mem** | 1.4 mJ each |
| norm/mod/res ×16 | 0 | 8K~102K | 64 | 33~361 | 452 | **mem** | <0.4 mJ each |

**关键观察**: flash + 12 linear 贡献几乎全部 matmul → 吃掉全部 MCU 能耗;gelu/qknorm/norm/concat
**零 matmul**(MCU 不启动)能耗近 0。大量 linear/rope/norm/concat 是 **memory-bound**
(mem > compute, George max 口径下时间由搬运决定)。

## 4. 与 simulator 报告的对比

| 口径 | M_BMM cycle | compute/memory | total | 能耗(MCU=100W) |
|---|---|---|---|---|
| simulator (你的报告) | 1024 | 串行相加 | 65.99 ms | — (无功耗模型) |
| **George 标准 (本报告)** | 8192 | max 重叠 | 55.36 ms | 0.186 J |

## 5. 三个必须标注的假设 (发 George 时一并确认)

1. **MCU 功耗取值**: 100W 为目标值(来源待 George 确认); M×K 修正版 2234W 是从
   DC 综合 M=4 外推到 M=128×K=1024(~1000× PE),公式形式经 RTL 验证(PE=M×K,
   非原 power_model 的 M²×K bug),但绝对值待大尺寸重新综合校准。

2. **cycle 用 George performance 版 (M_BMM=8192)**: 与 simulator 的 1024 差 8 倍。
   RTL 推导(`(MLEN//BLEN)² tile × ~M 拍`)偏向 8192。两套是真实建模分歧,需 George
   确认真机一条 M_BMM 到底多少拍。

3. **memory 用 emulator transmission + George max 重叠**: George 原版 memory_time 是
   `bytes/带宽` 静态除法; 此处用 emulator 实测 transmission 替代(更准)。max 重叠
   假设 HBM 与 compute 完全 pipeline(George 口径); **simulator 实测是串行不重叠**
   (prefetch 阻塞 await), 两者对 pipeline 假设相反。

   另: DC 综合**无 switching activity 标注**(默认 toggle), 功耗本身偏低估(power_model
   作者自注); scalar/control 逻辑 + HBM 控制器功耗未单列(偏低估)。

## 6. 复现

```bash
# 导出 George performance 版 cycle 公式
git show origin/experiment:analytic_models/performance/customISA_lib.json > /tmp/customISA_perf.json
# MCU=100W
python3 tools/power/plena_isa_energy.py managerbuild_DSB/ir --customisa /tmp/customISA_perf.json --mcu-power-w 100
# MCU=M×K 修正
python3 tools/power/plena_isa_energy.py managerbuild_DSB/ir --customisa /tmp/customISA_perf.json
```
