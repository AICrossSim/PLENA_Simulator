# PLENA 能耗估算工具 — 说明文档

> 数据用 George 的 analytic 模型,结构用真实 `.isa`。结论与 George **同源可对话**,
> 但因为输入是真实编译产物(而非维度公式猜),精度更高、可拆分。

---

## 0. 目录文件

| 文件 | 作用 |
|---|---|
| `plena_isa_energy.py` | 核心:读 `.isa` → cycle 分桶 → 套功耗 → 能耗 |
| `run_energy.sh` | 一键封装(自动导出 George cycle 表 + 跑) |
| `README.md` | 本文档 |
| `ssb_power_estimate.py` | 早期 SSB 估算草稿(已被 `plena_isa_energy.py` 取代,保留备查) |

---

## 1. 快速上手

```bash
# 默认: 跑 managerbuild_DSB/ir, MCU=100W
tools/power/run_energy.sh

# 指定 ir 目录
tools/power/run_energy.sh managerbuild_SSB/ir

# 指定 MCU 功耗(W)
tools/power/run_energy.sh managerbuild_DSB/ir 100

# MCU 用 M×K 公式(外推 2234W)而非固定值
tools/power/run_energy.sh managerbuild_DSB/ir formula
```

直接调核心脚本(需先导出 cycle 表):
```bash
git show origin/experiment:analytic_models/performance/customISA_lib.json > /tmp/ci.json
python3 tools/power/plena_isa_energy.py managerbuild_DSB/ir --customisa /tmp/ci.json --mcu-power-w 100
```

---

## 2. 设计:两条腿 + 真实 .isa

复刻 George analytic 的两条独立计算腿,但把 George "用模型维度公式**猜**指令数"
换成 "读真实 `.isa` **数**指令(含 cloop 展开)"。

```
真实 .isa (managerbuild_*/ir/<kernel>/<kernel>.isa)
  │
  ├─【腿A: compute】
  │   1. 逐条指令 + cloop 嵌套展开  →  各 opcode 真实条数
  │   2. 每条指令 cycle  ←─ George customISA_lib.json (表1)
  │   3. 按 opcode 前缀分桶:
  │        M_* → matmul (MCU + Matrix SRAM)
  │        V_* → vector (Vector Unit + Vector SRAM)
  │        S_* → scalar (标量逻辑, 功耗忽略)
  │        C_* → control (循环/setreg, 功耗忽略)
  │        H_* → memory  (HBM DMA, 不计入 compute cycle)
  │   4. compute_us = (matmul+vector+scalar+control) cycle / freq
  │
  ├─【腿B: memory】 (与 compute 完全独立)
  │   5. memory_us = emulator 实测 transmission (KERNEL_REPORT §1.3)
  │       (George 原版是 bytes/带宽 静态除法; 这里用实测搬运, 更准)
  │
  ├─【合成时间】George 重叠口径
  │   6. 每个 kernel: total_us = max(compute_us, memory_us)   ← kernel 内重叠
  │      整链:       total    = Σ max(compute_i, memory_i)   ← kernel 间串行
  │      ⚠️ 注意是"逐 kernel 先 max 再求和", 不是 max(Σcompute, Σmemory)
  │
  └─【能耗】 George power_model (表2)
      7. 能耗 = 组件功耗(W) × 组件活跃时间(s)
           MCU   活跃 = matmul cycle / freq      (指令级精确, 非 George 段级近似)
           Vector活跃 = vector cycle / freq
           MSRAM 活跃 = matmul cycle / freq
           VSRAM 活跃 = total_us                 (SRAM 全程通电, 漏电底盘主导)
         整链能耗 = Σ 各 kernel 各组件能耗
```

> **关于利用率(utilization):** George 的 sys_model 有 `systolic_util`
> (prefill=100%, decode=`min(batch/blen,1)`),作为乘子挂在 MCU 活跃段上。
> 本工具**隐含 prefill systolic_util=100%(等于不乘)**,所以没有显式的 util 因子。
> DSB/SSB 全是 prefill(Open-Sora 无 decode/无 KV cache),util 恒为 1,
> 与 George 在此场景下结果一致。若将来要算 **decode**(MCU 未填满, util<100%),
> 需在 MCU/Matrix-SRAM 能耗上另乘 `systolic_util`。

### 比 George 精确在哪

| 维度 | George analytic | 本工具 |
|---|---|---|
| 指令条数 | 模型维度公式**猜** | **真实 .isa 数**(含 cloop) |
| 拆 matmul/vector/scalar | 累加成标量,**拆不开** | **逐 kernel 分桶** |
| MCU 活跃时间 | 按 segment 近似(整段算) | **指令级**(只数 M_*) |
| memory/compute 区分 | 两套维度公式 | **opcode 前缀直接分** |
| memory 时间 | bytes/带宽 静态估 | **emulator 实测 transmission** |

---

## 3. 数据源(全部 George,原样照搬)

### 表1 — 指令 cycle:`analytic_models/performance/customISA_lib.json`

origin/experiment 分支(工作树无,`run_energy.sh` 自动 `git show` 导出)。
52 条指令,每条一个 `pipelined` 公式,`eval` 套硬件参数得整数 cycle。关键值
(MLEN=1024, BLEN=128, DC_EN=1):

| 指令 | pipelined 公式 | 值 |
|---|---|---|
| M_MM / M_TMM / M_TMM_A | `BLEN` | 128 |
| **M_BMM / M_BTMM** | `BLEN × (MLEN//BLEN)²` | **8192** |
| M_MV/TMV/BMV/BTMV | `1 + BLEN` | 129 |
| M_*_WO | `1` | 1 |
| V_ADD/SUB | `VECTOR_ADD_CYCLES` | 1 |
| V_MUL | `VECTOR_MUL_CYCLES` | 1 |
| V_EXP/RECI | `1 + VECTOR_EXP_CYCLES` | 2 |
| V_RED_SUM | `VECTOR_SUM_CYCLES` | 8 |
| V_RED_MAX | `1 + VECTOR_MAX_CYCLES` | 5 |
| S_* (标量) | 1~2 | |
| C_* (控制) | 1 | |
| **H_PREFETCH/STORE** | `1` | 1 (≈免费, memory 走腿B) |

`VECTOR_ADD_CYCLES` 等不是常数, 是 toml `[BEHAVIOR.LATENCY]` 的变量(DC_EN 开/关两档),
本工具内嵌 DC_EN=1 档的值(见 `plena_isa_energy.py` 的 `HW` dict)。

### 表2 — 组件功耗:`analytic_models/power_model/plena_power_model.py`

origin/experiment 分支。组件级(每块硬件满载多少瓦),DC 综合拟合。**原样照搬**
Vector/SRAM 三个公式;MCU 见下面"唯一改动"。

```python
# Vector Unit [W]  (拟合 VLEN=8..64, R²=0.999998)
P_vec = (-1.7746135753e-07*VLEN**2 + 5.4279213710e-04*VLEN - 4.9166666667e-05) * activity

# Matrix SRAM [W]  (拟合 MLEN=8..64 @depth128, R²=0.99997)
P_msram = (1.9583044211e-06*MLEN**2 + 2.4835249306e-03*MLEN + 5.2254044409e-04) * (depth/128)

# Vector SRAM [W]  (拟合 VLEN=8..64 @depth128, R²=0.999996; +0.245W 漏电底盘)
P_vsram = (-3.1305759318e-08*VLEN**2 + 3.4435611618e-04*VLEN + 0.24520172086) * (depth/128)
```

DC 综合原始数据点(拟合来源):Vector VLEN=8→4.3mW / 64→34.0mW;
Matrix SRAM MLEN=8→20.4mW / 64→167.7mW;Vector SRAM VLEN=8→248mW / 64→267mW。

#### MCU 唯一改动:M²×K → M×K

George 原版 `P_mcu = 4.26e-3 × M² × K`。**RTL 实锤 PE 总数 = M×K**
(`mx_systolic_mcu.sv`: `SYS_ARRAY_AMOUNT=K/M` 个子阵列 × 每个 `M×M` PE = M×K),
原 M² 是 bug(只在 M=4 单点拟合,M²和M被系数吸收分不出)。本工具改为:
```python
P_mcu = 17.044e-3 × M × K     # c≈17mW/PE, 由 M=4 三点反推 (16.99/16.88/17.27, 高度一致)
                              # RTL: M=BLEN(batch), K=MLEN  (matrix_machine.sv:197-199)
```
`--mcu-power-w <值>` 可直接给定固定瓦数,**绕过此公式**(默认 run_energy.sh 用 100W)。

---

## 4. 当前结果 (managerbuild_DSB, MLEN=1024, MCU=100W)

| 层 | compute | memory | total(Σmax) | 能耗 | 平均功率 |
|---|---|---|---|---|---|
| 整链 | 35.49 ms | 28.32 ms | **55.36 ms** | **0.186 J** | **3.4 W** |

| 组件 | 能耗 | 占比 |
|---|---|---|
| MCU (systolic) | 144.4 mJ | 77.5% |
| Vector SRAM | 31.3 mJ | 16.8% |
| Vector Unit | 9.1 mJ | 4.9% |
| Matrix SRAM | 1.7 mJ | 0.9% |

(MCU 改 `formula` → 2234W → 整链 3.27J / 平均 59W; 两版仅差 MCU 功耗取值)

---

## 5. 必须标注的假设(发 George 对齐时一并确认)

1. **MCU 功耗取值**:默认 100W 为目标值(来源待确认)。M×K 公式版 2234W 是
   DC 综合 M=4 外推到 M=128×K=1024(~1000× PE);公式形式经 RTL 验证(PE=M×K,
   非原 M²×K bug),但绝对值待大尺寸重新综合校准。

2. **cycle 用 George performance 版 M_BMM=8192**:与 simulator 的
   `overhead+MLEN=1024` **差 8 倍**。三个分支(本地/main/experiment)的 simulator
   全是 1024,analytic 表独立用 8192,George 同一 experiment 分支里两套也不一致。
   RTL 推导(`(MLEN//BLEN)² tile × ~M 拍`)偏向 8192。真机一条 M_BMM 多少拍需 George 确认。

3. **memory = emulator transmission + George max 重叠**:max 重叠假设 HBM 与 compute
   完全 pipeline(George 口径);**simulator 实测是串行不重叠**(prefetch 阻塞 await),
   两者对 pipeline 假设相反。另:DC 综合**无 switching activity 标注**(默认 toggle),
   功耗本身偏低估(power_model 作者自注);scalar/control + HBM 控制器功耗未单列(偏低估)。

---

## 6. 复用到其他 build

工具按 opcode 前缀分桶,与 kernel 名无关,任何 `managerbuild_*/ir` 都能跑。
唯一需注意:`memory_us` 用的 `DSB_TRANSMISSION_NS` 表硬编在 `plena_isa_energy.py`
里(DSB 专用)。跑别的 block 时,需把对应 build 的逐 kernel transmission
(来自其 `KERNEL_REPORT_*.md §1.3`)替换进去,或设为 0(纯 compute 视角)。
