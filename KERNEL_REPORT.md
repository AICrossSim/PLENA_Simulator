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

#### 2.2.1 linear_min — BTMM 路径 (2026-05-23 重写)

linear_min 重写为走 **M_BTMM** 而非 M_MM_WO。M_BTMM 把 K 按 hlen 切成
`lane_count = mlen/hlen` 个部分积累加进硬件累加器 (hm_accum +=),完整结果
= lane_count 个 mlen×mlen tile 之和。新加的 `split_btmm_materialize` 前端
pass 自动分配 `(lane_count*mlen, mlen)` scratch,K 循环后发一次 M_BMM_WO
drain 到 scratch,再用 C_LOOP (over mlen rows) 把 lane_count 个 tile 用
V_ADD 累加进 C_loc。新 gemm kind `btmm_mm` (非 async,不走 multilane fuse),
区别于 flash 的 head-fused async `btmm`。

| 项 | 值 |
|---|---|
| 配置 | M=N=K=1024 (2×2×2),MLEN=512,HLEN=128 (lane_count=4),with_bias |
| 输出 | C,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999639** |
| **NRMSE** | **3.93%** |
| Per-row cosine (mean / min) | 0.99936 / 0.99911 |
| **Relative match rate** | **92.64%** (输出无大量近零值,逐点正确率高) |
| allclose | 97.83% (atol=0.03 对 ±9 量级输出技术性 FAIL;参考) |
| **延迟** | **380,654 ns** (≈381 μs) |
| **状态** | ✅ **PASS** (与 matmul 路径同等 cosine/NRMSE 质量) |

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
| 吞吐 | 1,584,153 tokens/s (1024 tokens / 646402 ns) |
| **延迟** | **646,402 ns** (≈646 μs,纯 vector,无 matmul) |
| **状态** | ✅ **PASS** (cosine≈1.0,残差纯 MX 量化) |

纯 vector kernel,无 M_MM —— 延迟全部来自标量/向量指令。延迟较早期
版本 (≈15.8 ms) 大幅下降,得益于多语句 for-row 体的 loop fission(消除
整块 op 被外层 for-row 重复 N 倍的冗余) + 循环化。
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
| 吞吐 | 919,549 tokens/s (1024 tokens / 1113589 ns) |
| **延迟** | **1,113,589 ns** (≈1.11 ms,纯 vector,无 matmul) |
| **状态** | ✅ **PASS** (cosine≈1.0,残差纯 MX 量化) |

纯 vector kernel。延迟较早期版本 (≈20 ms) 大幅下降 (~18×),同 layernorm:
多语句 for-row 体的 loop fission 消除了整块 op 被外层 for-row 重复 N 倍的
冗余 + 循环化。

### 2.5 gelu_min

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, HLEN=128, NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | GELU(tanh近似): 0.5·x·(1+tanh(√(2/π)(x+0.044715x³))),tanh 手工展开为 1-2/(exp(2u)+1) |
| 输出 | Y,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999789** |
| **NRMSE** | **3.61%** |
| Per-row cosine (mean / min) | 0.999485 / 0.999363 |
| SNR | 28.84 dB |
| MSE / MAE / Max abs err | 9.99e-05 / 3.68e-03 / 0.125 |
| **Relative match rate** | **92.77%** (对近零值偏严) |
| allclose | 100.00% PASS (atol=0.03) |
| 吞吐 | 445,820 tokens/s (1024 tokens / 2296892 ns) |
| **延迟** | **2,296,892 ns** (≈2.3 ms) |
| **状态** | ✅ **PASS** (cosine=0.99979) |

延迟较早期版本 (≈164 ms) 大幅下降 (~71×):activation 链重写为直接在 2D
shared(VRAM) 上做向量运算 + fission 向量化,不再逐元素 FP scalar 展开。
exp/reci/mul 走整块 V_EXP_V/V_RECI_V/V_*_VF,标量 FP 不再主导。

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

RoPE 重写为 **shuffle-matrix matmul 形式** (2026-05-23),消除原先的 FPRAM
pair-swap。把 pair-swap 表达成 `OUT = X⊙COS + (X@P)⊙SGN_SIN`:P 是
pair-swap 置换矩阵(块对角 2×2 [[0,1],[1,0]]),`X@P` 走 btmm_mm 路径(单次,
P 块对角不跨 MLEN 边界,只需共享 512×512 对角块);两个 ⊙ 是整块 V_MUL,
末尾 V_ADD。SGN_SIN 由 host 预合成(even=-sin, odd=+sin)。BSHD 折叠成
BS·1·(H*D) 像 linear 一样大块处理。全 whole-tile,零 FPRAM/零 V↔FPRAM MAP。

| 项 | 值 |
|---|---|
| 配置 | MLEN=512, HEAD_COUNT=8, HLEN=128, H*D=1024, NUM_S_BLOCKS=2 → 输出 1024×1024 |
| 算子 | OUT = X⊙COS + (X@P)⊙SGN_SIN (P=pair-swap 置换矩阵;数学上严格等价原 pair-swap,实测 diff=0) |
| 输出 | Q_OUT,MX-E4M3,1,048,576 元素 (1024×1024) |
| **Global cosine** | **0.999713** |
| **NRMSE** | **3.63%** |
| Per-row cosine (mean / min) | 0.999442 / 0.999160 |
| SNR | 28.80 dB |
| MSE / MAE / Max abs err | 2.93e-04 / 5.12e-03 / 0.250 |
| **Relative match rate** | **94.52%** (对近零值偏严) |
| allclose | 100.00% PASS (atol=0.03) |
| 吞吐 | 2,941,895 tokens/s (1024 tokens / 348075 ns) |
| **延迟** | **348,075 ns** (≈348 μs) |
| **状态** | ✅ **PASS** (cosine≈1.0,残差纯 MX 量化) |

延迟较原 FPRAM pair-swap 版 (≈53.3 ms) 暴降 **~153×**:原版每行搬进 FPRAM
逐元素标量 FMA(V↔FPRAM MAP + 标量链)是瓶颈;shuffle-matrix 版用一次 BTMM
(P 是 0/1 置换,MX 无损) + 3 条整块向量 op,纯 whole-tile。P 为 0/1 置换故
X@P 无额外量化误差,cosine 与原版持平(0.9997)。

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

## 2.11 single_stream_block 整链 latency 占比 (2026-05-23 更新)

各步用 MLEN=512 同口径实测延迟(单 kernel 独立跑的 latency)累加。本次
更新反映两组重大优化:(1) linear 改走 **BTMM** 路径(§2.2.1),q/k/v/mlp
从 70ms → 0.38ms;(2) activation/norm 链 fission 向量化重写,layernorm
15.8→0.65ms、rmsnorm 20→1.11ms、gelu 164→2.3ms。lin2(K=CONCAT_DIM=4096,
K_BLOCKS=8 ≈ 4× K-reduction)按 BTMM 单 tile 0.38ms × ~4 粗估 ~1.5ms。

| 步 | kernel | 次数 | 单次(ms) | 小计(ms) | 来源 |
|---|---|---|---|---|---|
| layernorm | layernorm_min | 1 | 0.646 | 0.646 | 实测(新) |
| modulate | modulate_min | 1 | 0.21 | 0.21 | 实测 |
| linear_q/k/v/mlp | linear_min(BTMM) | 4 | 0.381 | 1.52 | 实测(新) |
| qknorm_q/k | rmsnorm_min | 2 | 1.11 | 2.22 | 实测(新) |
| rope_q/k | rope_min(shuffle-mm) | 2 | 0.348 | 0.696 | 实测(新) |
| **flash_attention** | flash_attention_min | 1 | 72.9 | **72.9** | 旧值·未重测 |
| gelu | gelu_min | 1 | 2.3 | 2.3 | 实测(新) |
| concat | concat_min | 1 | ≈0 | ~0 | copy,极小 |
| linear2 | linear_min(BTMM) | 1 | ~1.5 | ~1.5 | 粗估(K=4096) |
| residual_gate | residual_gate_min | 1 | 0.21 | 0.21 | 实测 |

**整链 ≈ 82 ms**(只剩 flash 用旧值)。

**占比(整链 ~82ms 为基准):**
| 组 | 延迟 | 占比 |
|---|---|---|
| **flash_attention** | **73 ms** | **~89%** ← 绝对大头(未重测) |
| gelu | 2.3 ms | ~2.8% |
| qknorm ×2 | 2.2 ms | ~2.7% |
| linear ×5 (q/k/v/mlp + lin2) | ~3.0 ms | ~3.7% |
| rope ×2 | 0.70 ms | <1% |
| layernorm | 0.65 ms | <1% |
| modulate + residual | 0.42 ms | <1% |

**结论(再次翻转):** rope 改 shuffle-matrix 后从 ~107ms(57%)暴降到 0.70ms
(<1%) —— 见 §2.7,~153×。现在 **flash_attention 单个就占 ~89%**,是唯一剩
下的大头。flash 是 matmul-bound,且仍是优化前的旧值,重测 + 走 BTMM 优化是
下一个重心。其余(linear/gelu/norm/rope)经 BTMM + 向量化重写后合计仅 ~9ms。

**待精确化:** flash 仍是旧值(现在主导整链,优先级最高);lin2 按 BTMM
cost∝tiles 粗估,可单独实测。

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
