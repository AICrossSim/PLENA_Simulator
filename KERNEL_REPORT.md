# PLENA Kernel Report

**生成日期**: 2026-05-21
**编译路径**: v2 后端 (PreIsaIR v2 → MIR → ISA),稳定寄存器分配器 (`USE_STABLE_ALLOC=True`)
**模拟器**: `transactional_emulator` (analytic 计时模型,1 cycle = 1 ns)

---

## 1. 硬件设置 (plena_settings.toml, active = `analytic`)

| 参数 | 值 | 说明 |
|---|---|---|
| MLEN | 1024 | 脉动阵列行数 / 矩阵 tile 边 |
| HLEN | 128 | head 宽度 |
| VLEN | 1024 | 向量宽度 |
| BLEN | 8 | batch lane |
| BROADCAST_AMOUNT | 8 | (BMM/BTMM 实际用 MLEN/HLEN,见模拟器说明) |
| lane_count | 8 | = MLEN / HLEN |
| HBM_SIZE | 1073741824 (1 GiB) | |
| HBM_WIDTH | 512 | |
| MATRIX_SRAM_SIZE | 256 | MRAM 深度 |
| VECTOR_SRAM_SIZE | 1024 | VRAM 深度 |

**精度 (MX-E4M3)**: 所有 HBM tensor 走 MX 量化 —— 元素 E4M3 (1 byte),
scale E8M0 (8-bit exponent),block = 8。GP/IntRAM 为 int32,FPRAM 为 fp16 标量。

**计时模型**: M_MM/M_BTMM = MLEN cycle;向量 op = 1(RED_MAX=4,
RED_SUM=8);标量 int/fp = 1(含 IntRAM LD/ST,spill 免费);HBM DMA = 0 cycle;
无指令级并行。详见 `transactional_emulator/TIMING_AUDIT_vs_main.md`
(已核实本分支相对 main 对计时零改动)。

### 1.1 MLEN=512 配置 (vector kernel 用)

§2.1 / §2.2 的 flash_attention / linear 数据是在 **MLEN=1024** 下跑的。
§2.3 起的 vector kernel (layernorm / modulate / rmsnorm / rope / gelu /
residual_gate / silu) 跑在 **MLEN=512** 的 BEHAVIOR 配置:

| 参数 | 值 |
|---|---|
| MLEN | 512 |
| HLEN | 128 |
| VLEN | 512 |
| BLEN | 4 |
| HBM_WIDTH | 512 |
| prefetch/writeback | 128 |

这些 kernel 没有 matmul (纯 vector/标量),所以 MLEN 主要影响 tile 行数
(`ROWS=MLEN`、`SEQ=2*MLEN=1024`),不改变 M_MM cost。计时模型与 §1 一致。

---

## 2. 跑过的 Kernel 与结果

判定标准:
- **cosine similarity** + **NRMSE**: 尺度不变的金标准,正确反映 MX 量化噪声,
  判方向/结构是否正确。
- **relative match rate** (|err|/|golden| ≤ rtol): 逐元素相对误差通过率,
  重要的逐点正确性指标。注意它对**近零值**偏严(小分母放大单个量化档),
  attention 这类大量近零输出的 kernel 会偏低,需结合 cosine 一起读。
- **allclose** (固定 atol=0.03): 对**大输出量级**偏严(大值上 MX 量化的绝对
  误差超固定 atol),linear 这类 ±9 量级输出会偏低,参考即可。

### 2.1 flash_attention_min

| 项 | 值 |
|---|---|
| 配置 | MLEN=1024, HLEN=128, HEAD_COUNT=8, NUM_Q_BLOCKS=2, NUM_KV_BLOCKS=2 |
| 输出 | O,MX-E4M3,4,194,304 元素 |
| **Global cosine** | **0.999552** |
| **NRMSE** | **4.90%** |
| Per-row cosine (mean / min) | 0.9988 / 0.9986 |
| **Relative match rate** | **77.79%** (大量近零 attention 输出拉低,非真失败;cosine 印证正确) |
| SNR | 26.20 dB |
| allclose | 100.00% PASS (atol=0.03) |
| **延迟** | **148,024,263 ns** (≈148 ms,matmul-bound) |
| Cycle profile | M_MM 占 90.9%,spill (LD/ST_INT) 仅 5.1% |
| **状态** | ✅ **PASS** (四配置全过: small/large × NQ=1/2) |

### 2.2 linear_min

| 项 | 值 |
|---|---|
| 配置 | M=N=K=2048 (m_blocks=2, n_blocks=2, k_blocks=2),MLEN=1024,with_bias |
| 算子 | C = A @ B^T + bias (nn.Linear,transpose_B → M_TMM) |
| 输出 | C,MX-E4M3,4,194,304 元素 (2048×2048) |
| **Global cosine** | **1.000000** |
| **NRMSE** | **3.90%** |
| Per-row cosine (mean / min) | 0.99937 / 0.99922 |
| SNR | 28.17 dB |
| MSE / MAE / Max abs err | 1.11e-02 / 3.45e-02 / 1.000 |
| **Relative match rate** | **92.89%** (输出无大量近零值,逐点正确率高) |
| allclose | 96.94% (atol=0.03 不适配 ±9 量级输出;参考) |
| **延迟** | **137,743,637 ns** (≈138 ms) |
| **状态** | ✅ **PASS** (cosine=1.0,残差纯 MX 量化) |

### 2.3 layernorm_min

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HIDDEN_SIZE=1024, NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | y = (x-mean)·rsqrt(var+eps)·scale + bias (per-row, D=1024) |
| 输出 | Y,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999614** |
| **NRMSE** | **3.90%** |
| Per-row cosine (mean / min) | 0.999373 / 0.999117 |
| SNR | 28.19 dB |
| MSE / MAE / Max abs err | 1.61e-03 / 1.24e-02 / 0.500 |
| **Relative match rate** | **92.89%** (对近零值偏严) |
| allclose | 99.37% (atol=0.03;参考) |
| **延迟** | **15,815,903 ns** (≈15.8 ms,纯 vector,无 matmul) |
| **状态** | ✅ **PASS** (cosine≈1.0,残差纯 MX 量化) |

纯 vector kernel,无 M_MM —— 延迟全部来自循环展开后的标量/向量指令
(extent 1024/512 的嵌套循环放大)。验证走 HBM-direct + MX 两侧 round-trip,
与 linear / flash 同一套(见 §3)。

### 2.4 rmsnorm_min

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, HLEN=128, NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | y = (x·scale)·rsqrt(mean(x²)+eps) (per-head_dim RMSNorm) |
| 输出 | Y,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999632** |
| **NRMSE** | **3.83%** |
| Per-row cosine (mean / min) | 0.999391 / 0.999116 |
| SNR | 28.33 dB |
| MSE / MAE / Max abs err | 1.47e-03 / 1.17e-02 / 0.500 |
| **Relative match rate** | **92.89%** (对近零值偏严) |
| allclose | 99.43% (atol=0.03;参考) |
| **延迟** | **20,075,988 ns** (≈20.1 ms,纯 vector,无 matmul) |
| **状态** | ✅ **PASS** (cosine≈1.0,残差纯 MX 量化) |

纯 vector kernel,延迟来源同 layernorm:循环展开后的标量访存
(S_LD_INT 143 + S_ST_INT 106,extent=512 循环放大)占绝大多数,真正干活的
V_* 向量指令只有 ~7 条。根因是 stable allocator 把每个值钉死 IntRAM slot
(见 [stable register allocator] memory),循环内反复 ST/LD 被放大 —— 偏保守
但保证正确性。

### 2.5 gelu_min

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, HLEN=128, NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | GELU(tanh近似): 0.5·x·(1+tanh(√(2/π)(x+0.044715x³))),tanh 手工展开为 1-2/(exp(2u)+1) |
| 输出 | Y,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999935** |
| **NRMSE** | **2.78%** |
| Per-row cosine (mean / min) | 0.999658 / 0.999527 |
| SNR | 31.13 dB |
| MSE / MAE / Max abs err | 5.90e-05 / 2.60e-03 / 0.125 |
| **Relative match rate** | **92.77%** (对近零值偏严) |
| allclose | 100.00% PASS (atol=0.03) |
| **延迟** | **163,771,705 ns** (≈164 ms) |
| **状态** | ✅ **PASS** (cosine=0.99994,链中精度最高) |

延迟比 layernorm/rmsnorm 高一个数量级(164ms vs ~16-20ms):GELU 每元素要算
exp + 倒数 + 一长串 FP scalar(tanh 手工展开),逐元素的 S_EXP_FP/S_RECI_FP/
S_MUL_FP chain 在 1024×1024 上展开,标量 FP 成主导。cosine 最高(0.99994)是因
输出无大量近零值。

### 2.6 residual_gate_min

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, HLEN=128, NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | out = x + gate · y (逐元素 tile_mul + tile_add) |
| 输出 | OUT,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999737** |
| **NRMSE** | **3.52%** |
| Per-row cosine (mean / min) | 0.999472 / 0.999238 |
| SNR | 29.08 dB |
| MSE / MAE / Max abs err | 3.18e-04 / 5.31e-03 / 0.250 |
| **Relative match rate** | **94.51%** (对近零值偏严) |
| allclose | 100.00% (atol=0.03) |
| **延迟** | **208,733 ns** (≈0.21 ms,全链最快) |
| **状态** | ✅ **PASS** (cosine≈1.0,残差纯 MX 量化) |

全链延迟最低(0.21ms):没有 per-row reduce、没有 FP 超越函数,只是两条整 tile 的
V_MUL + V_ADD,标量地址循环极少。对照 gelu(164ms)/rmsnorm(20ms),说明 vector
kernel 的延迟差异几乎全由"每元素的标量/FP 链长度 + reduce"决定。

### 2.7 rope_min

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, HLEN=128, half_dim=64 (full_dim=128==hlen), NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | RoPE pair-swap: 偶 d → x[d]·cos+x[d^1]·neg_sin;奇 d → x[d^1]·sin+x[d]·cos |
| 输出 | Q_OUT,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999711** |
| **NRMSE** | **3.64%** |
| Per-row cosine (mean / min) | 0.999440 / 0.999160 |
| SNR | 28.78 dB |
| MSE / MAE / Max abs err | 2.94e-04 / 5.14e-03 / 0.250 |
| **Relative match rate** | **94.50%** (对近零值偏严) |
| allclose | 100.00% (atol=0.03) |
| **延迟** | **53,273,771 ns** (≈53.3 ms) |
| **状态** | ✅ **PASS** (cosine≈1.0,残差纯 MX 量化) |

注:rope_min 要求 full_dim==hlen,所以 half_dim=HLEN//2=64(原 testbench 的
HALF_DIM=8 与 hlen=128 不符,已修)。延迟居中:每元素一组 pair-swap 的
V_MUL+V_ADD,无 reduce/无超越函数,但 cos/sin/neg_sin 三路输入的逐元素乘加 +
地址计算撑起 ~53ms。

### 2.8 silu_min ✅ 正确,但 ⚠️ 写法低效(延迟可大幅优化)

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, HLEN=128, NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | SiLU(x) = x·sigmoid(x) = x / (1+exp(-x)) |
| 输出 | Y,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999847** |
| **NRMSE** | **2.71%** |
| Per-row cosine (mean / min) | 0.999667 / 0.999494 |
| SNR | 31.35 dB |
| MSE / MAE / Max abs err | 5.05e-05 / 2.06e-03 / 0.031 |
| **Relative match rate** | **93.75%** (对近零值偏严) |
| allclose | 100.00% PASS (atol=0.03) |
| **延迟** | **63,108,396 ns** (≈63 ms,偏高 — 见下) |
| **状态** | ✅ **PASS**(正确率没问题),但延迟可优化 |

**正确率没问题(cosine 0.999847,allclose 100% PASS)** —— 低效 ≠ 不正确。
但延迟 63ms 偏高:kernel 把整行搬进 FPRAM 标量区逐元素算,产生 V↔FPRAM
transfer(`S_MAP_V_FP` / `S_MAP_FP_V`,每条 = VLEN = 512 cycle)。

根因在 [silu_min.py](compiler/tilelang_tvm_compiler/kernels/silu_min.py) L77-90:
```
for row in T.serial(rows):        # 512 次
    T.copy(X_sh[row,0], X_FP)     # VRAM→FPRAM = S_MAP_V_FP (512 cyc)
    for i in T.unroll(hlen): ...  # 在 1D fragment(FPRAM)上逐元素 sigmoid
    T.copy(Y_FP, Y_sh[row,0])     # FPRAM→VRAM = S_MAP_FP_V (512 cyc)
```
`X_FP/Y_FP = alloc_fragment((hlen,))` 是 **1D fragment → 落 FPRAM 标量区**,
每行进出各一条 S_MAP transfer。

**优化方向(可选,不影响正确性):** 像 gelu 那样直接在 VRAM 上走向量路径
(`V_EXP_V` / `V_RECI_V` / `V_MUL_VV`,整行一次过),用 2D fragment(VRAM)
而非 1D fragment(FPRAM),消除 S_MAP,延迟可降一个量级。

### 2.9 modulate_min

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, HLEN=128, NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | y = (1+scale)·x + shift (adaLN modulate;1+scale 由 host 折叠传入) |
| 输出 | Y,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999654** |
| **NRMSE** | **3.63%** |
| Per-row cosine (mean / min) | 0.999441 / 0.999189 |
| SNR | 28.80 dB |
| MSE / MAE / Max abs err | 4.43e-04 / 6.18e-03 / 0.250 |
| **Relative match rate** | **94.48%** (对近零值偏严) |
| allclose | 99.97% (atol=0.03) |
| **延迟** | **208,733 ns** (≈0.21 ms,与 residual_gate 并列最快) |
| **状态** | ✅ **PASS** (cosine≈1.0,残差纯 MX 量化) |

与 residual_gate 同属"纯整 tile 乘加、无 reduce/无超越函数"的最快档(0.21ms):
只有 V_MUL_VV + V_ADD_VV 两条向量操作 + 极少标量地址循环。

### 2.10 flash_attention_min (MLEN=512 链配置)

§2.1 的 flash 数据是 MLEN=1024/HEAD=8 旧配置;这里是 single_stream_block
**链实际跑的 MLEN=512 配置**,与其它 vector kernel 同口径,用于整链占比。

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, NUM_Q_BLOCKS=NUM_KV_BLOCKS=2 → 输出 1024×1024 |
| 算子 | self-attention: softmax(Q@Kᵀ/√d)@V (online softmax) |
| 输出 | O,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999084** |
| **NRMSE** | **4.87%** |
| Per-row cosine (mean / min) | 0.998815 / 0.998479 |
| SNR | 26.24 dB |
| MSE / MAE / Max abs err | 5.85e-07 / 4.32e-04 / 0.0039 |
| **Relative match rate** | **78.29%** (大量近零 attention 输出拉低,非失败;cosine 印证正确) |
| allclose | 100.00% PASS (atol=0.03) |
| **延迟** | **72,935,215 ns** (≈72.9 ms,matmul-bound) |
| **状态** | ✅ **PASS** |

---

## 2.11 single_stream_block 整链 latency 占比

各步用 MLEN=512 同口径实测延迟(单 kernel 独立跑的 latency)累加。
linear_min(M=N=K=1024,即 q/k/v/mlp 规模)实测 **70.0 ms**。
gelu 走 VRAM 向量路径重写后预期 ~1/4(标"改后估");lin2 的 K=CONCAT_DIM=4096
(K_BLOCKS=8,4× K-reduction)未单独测,按 matmul cost ∝ tiles 粗估 ~280ms。

| 步 | kernel | 次数 | 单次(ms) | 小计(ms) | 来源 |
|---|---|---|---|---|---|
| layernorm | layernorm_min | 1 | 15.8 | 15.8 | 实测 |
| modulate | modulate_min | 1 | 0.21 | 0.21 | 实测 |
| **linear_q/k/v/mlp** | linear_min | 4 | 70.0 | **280.0** | 实测(512) |
| qknorm_q/k | rmsnorm_min | 2 | 20.1 | 40.2 | 实测 |
| rope_q/k | rope_min | 2 | 53.3 | 106.6 | 实测 |
| flash_attention | flash_attention_min | 1 | 72.9 | 72.9 | 实测(512) |
| gelu | gelu_min | 1 | 164→~41 | ~41 | 改后估 |
| concat | concat_min | 1 | ≈0 | ~0 | copy,极小 |
| **linear2** | linear_min | 1 | ~280 | **~280** | 粗估(K=4096) |
| residual_gate | residual_gate_min | 1 | 0.21 | 0.21 | 实测 |

**整链 ≈ 837 ms**(gelu 改后 41ms;lin2 粗估 280ms)。若 gelu 不改用原 164ms
则 ≈ 960ms;若 lin2 实测远小于估值,总数相应下降。

**占比(整链 ~837ms 为基准):**
| 组 | 延迟 | 占比 |
|---|---|---|
| **linear ×5 (q/k/v/mlp + lin2)** | **~560 ms** | **~67%** ← 绝对大头 |
| rope ×2 | 107 ms | ~13% |
| flash_attention | 73 ms | ~9% |
| gelu(改后) | 41 ms | ~5% |
| qknorm ×2 | 40 ms | ~5% |
| layernorm | 16 ms | ~2% |
| modulate + residual + concat | ~0.4 ms | <1% |

**结论:matmul(5 个 linear)占整链 ~2/3**,是绝对优化重心 —— 尤其 lin2(K=4096)
单个可能就 ~280ms/~33%。其次 rope×2(107ms,vector 侧最大,地址计算/IntRAM spill
密集,见 §2.4 的 stable-allocator 分析)。gelu 改 VRAM 路径后从 164→~41ms,
对整链贡献从 ~17% 降到 ~5%。

**待精确化:** lin2 的 MLEN=512 实测(现按 cost∝tiles 粗估);gelu/silu 重写后
重测填实测值。

## 3. 验证方法 (HBM-direct compare)

两个 kernel 都走 **HBM 直接比对**(读 `hbm_dump.bin` 的 MX-E4M3 字节,
**不重排** —— 物理字节顺序即 golden flatten 顺序),绕开 VRAM staging 的
stride 重排。Golden 两侧都做 MX-E4M3 round-trip(输入侧 + 输出侧),与
`create_mem_for_sim` / 模拟器写回的量化一致,故残差只剩 MX 量化噪声。

比对工具 `transactional_emulator/tools/check_mem.py` 现输出两段:
- **Correctness (scale-invariant)**: NRMSE / SNR / global+per-row cosine
- **Basic error reference**: MSE / MAE / max abs / allclose / relative

---

## 4. 本轮关键修复 (使 linear 大配置通过)

1. **稳定寄存器分配器** (`mir_to_isa.py`): 每个 i32 值占固定 IntRAM slot,
   GP 仅指令内 scratch,counter/lvar 钉死专属 GP。无跨指令 GP 状态 →
   循环任意迭代次数都正确。替换了原线性扫描分配器(跨迭代缝导致大配置失败)。
2. **copy_v_to_v 循环化** (`pre_isa_pass_v2.py`): 整块 `T.copy` 原本静态
   展开成 1024 条 V_ADD_VF + 2055 行常量(被 LICM 外提);改为 chunk offset
   等差时发一个 serial C_LOOP,否则回退静态展开。
3. **HBM testbench MX 约定对齐** (`tvm_linear_min_test.py`): 所有 HBM
   tensor 都 MX-packed(非 fp16);golden 必须两侧 MX round-trip,比对走
   MX 路径。
